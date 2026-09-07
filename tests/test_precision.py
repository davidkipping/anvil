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
    got = np.array(ll(v), dtype=np.float64) + ll.log_offset_const
    # reference from the fp32-rounded parameter values the mx paths receive
    p = v32.astype(np.float64)
    m = p[:, 0:1] * x[None, :] + p[:, 1:2]
    want = -0.5 * (((y[None, :] - m) / 0.01) ** 2).sum(axis=-1)
    np.testing.assert_allclose(got, want, rtol=1e-4, atol=0.5)
    hi = np.array(ll.hi(v)) + ll.log_offset_const
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


def test_hi_path_is_genuinely_float64():
    """Regression guard: mx.array() of a float64 numpy array silently
    yields float32 unless dtype= is named. That once made the whole
    'float64 verification path' a second float32 path, giving the
    precision harness a silent error floor of ~1 ulp of |logL|."""
    tt = make_transit_target(n_data=20_000, seed=0)
    u = mx.array(
        tt.transform.from_model_np(tt.truth_model)[None, :].astype(np.float32)
    )
    assert tt.target.log_prob_hi(u).dtype == mx.float64
    assert tt.loglike.hi(tt.transform.to_model(u)).dtype == mx.float64

    # and it must actually carry fp64 information: perturbing a parameter
    # far below fp32 resolution must move the fp64 logL but not the fp32 one
    u64 = np.array(u, dtype=np.float64)
    u64[0, 2] += 1e-9                      # << fp32 eps at this magnitude
    lp_a = float(np.array(tt.target.log_prob_hi(u), dtype=np.float64)[0])
    v_a = mx.array(tt.transform.model_np(np.array(u, dtype=np.float64)),
                   dtype=mx.float64)
    v_b = mx.array(tt.transform.model_np(u64), dtype=mx.float64)
    lp_b_hi = float(np.array(tt.loglike.hi(v_b))[0])
    lp_a_hi = float(np.array(tt.loglike.hi(v_a))[0])
    assert lp_a_hi != lp_b_hi, "fp64 path insensitive to sub-fp32 perturbation"
    assert np.isfinite(lp_a)


def test_recentring_is_an_exact_constant_offset():
    """Recentring must shift log_prob by exactly -N/2 and nothing else:
    every difference the sampler takes has to be unchanged."""
    rng = np.random.default_rng(0)
    x = np.linspace(0, 1, 20_000)
    y = 2.0 * x + 0.5 + 0.01 * rng.standard_normal(20_000)
    yerr = np.full(20_000, 0.01)

    def line(v, xx):
        return v[:, 0:1] * xx[None, :] + v[:, 1:2]

    # near-truth: chi2/N ~ 1, the regime recentring targets
    v = mx.array(np.array([[2.0, 0.5], [2.0004, 0.4998]], dtype=np.float32))
    off = ChunkedGaussianLogLike(line, x, y, yerr,
                                 PrecisionPolicy(recenter=False))
    on = ChunkedGaussianLogLike(line, x, y, yerr,
                                PrecisionPolicy(recenter=True))
    assert off.log_offset_const == 0.0
    assert on.log_offset_const == -0.5 * 20_000

    # float64 paths must agree exactly once the constant is reinstated
    a = np.array(off.hi(v)) + off.log_offset_const
    b = np.array(on.hi(v)) + on.log_offset_const
    np.testing.assert_allclose(a, b, rtol=1e-12)

    # and the recentred float32 value must be far smaller in magnitude,
    # which is the entire point (smaller ulp for downstream differences)
    m_off = np.abs(np.array(off(v), dtype=np.float64)).max()
    m_on = np.abs(np.array(on(v), dtype=np.float64)).max()
    assert m_on < 0.05 * m_off

    # graceful degradation: when the model fits badly (chi2/N >> 1) there is
    # no large constant to remove, so recentring is simply a no-op gain --
    # it must never be WORSE than the plain form
    bad = mx.array(np.array([[1.9, 0.55]], dtype=np.float32))
    assert (np.abs(np.array(on(bad), dtype=np.float64)).max()
            <= np.abs(np.array(off(bad), dtype=np.float64)).max())


def _short_run(n_data=20_000, n_chains=128, seed=0):
    from anvil import run
    from anvil.kernels.ensemble import EnsembleKernel
    tt = make_transit_target(n_data=n_data, seed=seed)
    u0 = mx.array(
        (tt.transform.from_model_np(tt.truth_model)
         + 1e-3 * np.random.default_rng(1).standard_normal((n_chains, 6))
         ).astype(np.float32))
    res = run(EnsembleKernel(tt.target, seed=0), tt.target, u0,
              n_warmup=300, n_samples=60, seed=2)
    return tt, res


def test_certify_reports_and_corrects():
    from anvil import certify
    tt, res = _short_run()
    draws = res.get_chain(flat=True)
    cert = certify(tt.target, draws, n_probe=64, target_ess=1e4,
                   names=tt.transform.names)

    assert cert.n_probe == 64 and cert.n_draws == draws.shape[0]
    assert cert.err_sd > 0 and np.isfinite(cert.err_sd)
    assert 0.0 < cert.is_retention <= 1.0
    assert cert.bias_std.shape == (6,)
    assert "ESS" in cert.verdict and str(cert)

    # the correction is exactly mean - Cov(f, err), and tiny here
    shift = np.abs(cert.corrected_mean - cert.raw_mean)
    assert np.all(shift < 0.5 * draws.std(axis=0))

    # correct() on the draws themselves must reproduce corrected_mean
    np.testing.assert_allclose(cert.correct(draws), cert.corrected_mean,
                               rtol=1e-10, atol=1e-14)
    # and it must work for a derived quantity of a different width
    phys = tt.transform.model_np(draws.astype(np.float64))
    assert cert.correct(phys).shape == (6,)
    assert np.isfinite(cert.correct(phys)).all()


def test_certify_requires_float64_path_and_valid_shapes():
    from anvil import certify
    from anvil.targets import correlated_gaussian
    tgt, mu, _ = correlated_gaussian(3, seed=0)
    with pytest.raises(ValueError, match="log_prob_hi"):
        certify(tgt, np.zeros((10, 3)))
    tt, res = _short_run()
    with pytest.raises(ValueError, match="n_draws, dim"):
        certify(tt.target, np.zeros(10))
    cert = certify(tt.target, res.get_chain(flat=True), n_probe=32)
    with pytest.raises(ValueError, match="same sample"):
        cert.correct(np.zeros(7))
