import mlx.core as mx
import numpy as np
import pytest

from anvil import run
from anvil.kernels.metropolis import RandomWalkMetropolis
from anvil.precision import (
    ChunkedGaussianLogLike,
    PrecisionPolicy,
    chunked_sum,
    validate_precision,
)
from anvil.targets import make_transit_target


def _terms_case(n_data=300_000, seed=0):
    """fp32 terms of roughly unit magnitude and one sign, where naive
    sequential accumulation visibly loses precision."""
    rng = np.random.default_rng(seed)
    terms64 = -0.5 * rng.chisquare(1, size=(4, n_data))
    terms32 = terms64.astype(np.float32)
    ref = terms32.astype(np.float64).sum(axis=-1)  # exact sum of fp32 terms
    return terms32, ref


def test_chunked_sum_tree_accuracy():
    terms32, ref = _terms_case()
    t32 = mx.array(terms32)

    def term_fn(s, e):
        return t32[:, s:e]

    out = np.array(
        chunked_sum(term_fn, terms32.shape[1], PrecisionPolicy()),
        dtype=np.float64,
    )
    # |sum| ~ 6e5; fp32 representation floor there is ~0.06
    assert np.abs(out - ref).max() < 0.5


def test_chunked_sum_fp64_anchor_beats_or_matches_tree():
    terms32, ref = _terms_case()
    t32 = mx.array(terms32)

    def term_fn(s, e):
        return t32[:, s:e]

    n = terms32.shape[1]
    err_tree = np.abs(np.array(
        chunked_sum(term_fn, n, PrecisionPolicy(reduction="fp32_tree")),
        dtype=np.float64) - ref).max()
    err_anchor = np.abs(np.array(
        chunked_sum(term_fn, n, PrecisionPolicy(reduction="fp64_anchor")),
        dtype=np.float64) - ref).max()
    assert err_anchor <= err_tree + 1e-9
    # the anchor's only fp32 rounding is per-chunk sums + the final cast
    assert err_anchor < 0.2


def test_mixed_stream_ops_compile():
    """fp64-on-CPU ops inside mx.compile work (plan risk #1, resolved)."""

    def f(x):
        p = mx.sum(x, axis=-1)
        s = p.astype(mx.float64, stream=mx.cpu)
        return mx.sum(s, stream=mx.cpu)

    out = mx.compile(f)(mx.ones((4, 100)))
    assert out.item() == 400.0
    assert out.dtype == mx.float64


def test_transit_conditioning_story():
    """The precision harness flags the naive model and passes the
    offset-conditioned one — the package's core numerical claim."""
    rng = np.random.default_rng(1)
    reports = {}
    for cond in ("naive", "epoch_centered"):
        tt = make_transit_target(n_data=100_000, seed=0, conditioning=cond)
        u_truth = tt.transform.from_model_np(tt.truth_model)
        u = mx.array(
            (u_truth + 0.05 * rng.standard_normal((16, 6))).astype(np.float32)
        )
        reports[cond] = validate_precision(tt.target, u)
    assert reports["naive"].median_abs_err > 3.0
    assert reports["epoch_centered"].median_abs_err < 1.0
    assert (
        reports["epoch_centered"].median_abs_err
        < 0.2 * reports["naive"].median_abs_err
    )


def test_gaussian_loglike_matches_numpy():
    rng = np.random.default_rng(3)
    x = np.linspace(0, 1, 5000)
    y = 2.0 * x + 0.5 + 0.01 * rng.standard_normal(5000)
    yerr = np.full(5000, 0.01)

    def line(v, xx):
        return v[:, 0:1] * xx[None, :] + v[:, 1:2]

    ll = ChunkedGaussianLogLike(line, x, y, yerr, PrecisionPolicy(chunk_size=1024))
    v32 = np.array([[2.0, 0.5], [1.9, 0.6]], dtype=np.float32)
    v = mx.array(v32)
    got = np.array(ll(v), dtype=np.float64)
    # reference from the fp32-rounded parameter values the mx paths receive
    p = v32.astype(np.float64)
    m = p[:, 0:1] * x[None, :] + p[:, 1:2]
    want = -0.5 * (((y[None, :] - m) / 0.01) ** 2).sum(axis=-1)
    np.testing.assert_allclose(got, want, rtol=1e-4, atol=0.5)
    hi = np.array(ll.hi(v))
    # fp64 summation-order differences (MLX reduction vs numpy pairwise)
    np.testing.assert_allclose(hi, want, rtol=1e-7)


@pytest.mark.slow
def test_reanchor_end_to_end():
    tt = make_transit_target(n_data=20_000, seed=0)
    u_truth = tt.transform.from_model_np(tt.truth_model)
    u0 = mx.array(
        (u_truth + 1e-3 * np.random.default_rng(0).standard_normal((256, 6)))
        .astype(np.float32)
    )
    res = run(
        RandomWalkMetropolis(tt.target), tt.target, u0,
        n_warmup=200, n_samples=50, seed=1, reanchor_every=25,
    )
    lp = res.get_log_prob()
    assert np.all(np.isfinite(lp))
    # cached log_prob should track the fp64 truth closely at the end
    lp_hi = np.array(tt.target.log_prob_hi(res.final_state["u"]))
    drift = np.abs(np.array(res.final_state["log_prob"], dtype=np.float64) - lp_hi)
    assert np.median(drift) < 1.0
