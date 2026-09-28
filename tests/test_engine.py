"""Engine-level tests, chiefly that the sampling pipeline is invisible.

Pipelining changes *when* arrays are evaluated, never what is computed, so
every result must be bit-identical at any depth. These tests assert that
exactly rather than statistically — a missed synchronisation would
otherwise corrupt stored frames silently.
"""

import mlx.core as mx
import numpy as np
import pytest

from anvil import run
from anvil.kernels.chees import ChEESHMC
from anvil.kernels.ensemble import EnsembleKernel
from anvil.kernels.metropolis import RandomWalkMetropolis
from anvil.surrogate import TrainingArchive
from anvil.targets import correlated_gaussian, make_transit_target


def _setup(dim=4, n=128, seed=0):
    target, mu, cov = correlated_gaussian(dim, rho=0.5, seed=seed)
    u0 = mx.array(
        (mu + np.random.default_rng(seed + 1).standard_normal((n, dim)))
        .astype(np.float32))
    return target, u0


def _kernels(target):
    return {
        "rwm": lambda: RandomWalkMetropolis(target),
        "ensemble": lambda: EnsembleKernel(target, seed=3),
        "chees": lambda: ChEESHMC(target),
    }


@pytest.mark.parametrize("kernel_name", ["rwm", "ensemble", "chees"])
def test_pipelining_is_bit_identical(kernel_name):
    target, u0 = _setup()
    make = _kernels(target)[kernel_name]
    kw = dict(n_warmup=40, n_samples=25, seed=7)
    ref = run(make(), target, u0, pipeline=0, **kw)
    for depth in (1, 2, 4):
        got = run(make(), target, u0, pipeline=depth, **kw)
        np.testing.assert_array_equal(ref.get_chain(), got.get_chain())
        np.testing.assert_array_equal(ref.get_log_prob(), got.get_log_prob())
        np.testing.assert_array_equal(ref.accept_fraction, got.accept_fraction)
        assert ref.extras["n_divergent"] == got.extras["n_divergent"]
        np.testing.assert_array_equal(
            np.array(ref.final_state["u"]), np.array(got.final_state["u"]))


def test_pipelining_respects_thinning():
    target, u0 = _setup()
    kw = dict(n_warmup=30, n_samples=9, thin=3, seed=8)
    ref = run(RandomWalkMetropolis(target), target, u0, pipeline=0, **kw)
    got = run(RandomWalkMetropolis(target), target, u0, pipeline=2, **kw)
    assert ref.get_chain().shape[0] == 9
    np.testing.assert_array_equal(ref.get_chain(), got.get_chain())


@pytest.mark.parametrize("depth,n_samples", [(4, 10), (3, 7), (8, 2), (2, 1)])
def test_pipeline_drains_completely(depth, n_samples):
    """Frames still in flight at the end must be stored, in order, even
    when the sample count is not a multiple of the depth."""
    target, u0 = _setup()
    kw = dict(n_warmup=10, n_samples=n_samples, seed=9)
    ref = run(RandomWalkMetropolis(target), target, u0, pipeline=0, **kw)
    got = run(RandomWalkMetropolis(target), target, u0, pipeline=depth, **kw)
    assert got.get_chain().shape[0] == n_samples
    np.testing.assert_array_equal(ref.get_chain(), got.get_chain())


def test_pipelining_preserves_archive_order():
    target, u0 = _setup()
    kw = dict(n_warmup=20, n_samples=12, seed=10)
    archives = {}
    for depth in (0, 3):
        arc = TrainingArchive(dim=4, capacity=10_000)
        run(RandomWalkMetropolis(target), target, u0, pipeline=depth,
            archive=arc, **kw)
        archives[depth] = arc.as_arrays()
    for a, b in zip(archives[0], archives[3]):
        np.testing.assert_array_equal(a, b)


def test_pipelining_with_reanchoring_matches_blocking():
    """Re-anchoring runs on the host, so the pipeline must be drained
    before it; the result must still be bit-identical."""
    tt = make_transit_target(n_data=5_000, seed=0)
    u0 = mx.array(
        (tt.transform.from_model_np(tt.truth_model)
         + 1e-3 * np.random.default_rng(0).standard_normal((64, 6)))
        .astype(np.float32))
    kw = dict(n_warmup=12, n_samples=20, seed=11, reanchor_every=5)
    ref = run(RandomWalkMetropolis(tt.target), tt.target, u0, pipeline=0, **kw)
    got = run(RandomWalkMetropolis(tt.target), tt.target, u0, pipeline=2, **kw)
    assert np.all(np.isfinite(ref.get_chain()))
    np.testing.assert_array_equal(ref.get_chain(), got.get_chain())


def test_auto_declines_pipelining_on_an_expensive_target(capsys):
    """The knob is auto by default and must not hold graphs in flight when
    the eval barrier is already negligible."""
    tt = make_transit_target(n_data=400_000, seed=0)
    u0 = mx.array(
        (tt.transform.from_model_np(tt.truth_model)
         + 1e-3 * np.random.default_rng(0).standard_normal((512, 6)))
        .astype(np.float32))
    run(RandomWalkMetropolis(tt.target), tt.target, u0,
        n_warmup=2, n_samples=10, seed=12, progress=1000)
    assert "not pipelining" in capsys.readouterr().out


def test_auto_enables_pipelining_on_a_cheap_target(capsys):
    target, u0 = _setup(n=64)
    run(RandomWalkMetropolis(target), target, u0,
        n_warmup=2, n_samples=30, seed=13, progress=1000)
    assert "pipelining at depth" in capsys.readouterr().out
