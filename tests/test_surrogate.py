import mlx.core as mx
import numpy as np

from anvil import run
from anvil.kernels.ensemble import EnsembleKernel
from anvil.surrogate import (
    ArchivingLogDensity,
    SwitchableLogDensity,
    TrainingArchive,
)
from anvil.targets import correlated_gaussian


def test_archive_ring_buffer():
    a = TrainingArchive(dim=2, capacity=100)
    for i in range(30):
        a.record(np.full((10, 2), i, dtype=float), np.full(10, i, dtype=float))
    assert len(a) == 100
    u, lp = a.as_arrays()
    assert u.shape == (100, 2) and lp.shape == (100,)
    # newest 100 of the 300 recorded rows survive: values 20..29
    assert lp.min() >= 20 and lp.max() == 29


def test_engine_records_archive_during_run():
    target, mu, _ = correlated_gaussian(3, seed=0)
    archive = TrainingArchive(dim=3, capacity=50_000)
    u0 = mx.array(np.random.default_rng(1).normal(
        size=(64, 3)).astype(np.float32) + mu)
    run(EnsembleKernel(target, seed=0), target, u0,
        n_warmup=20, n_samples=30, seed=2, archive=archive)
    assert len(archive) == 30 * 64
    u, lp = archive.as_arrays()
    # recorded pairs are consistent: re-evaluating gives the same log_prob
    check = np.array(target.log_prob(mx.array(u[:100].astype(np.float32))),
                     dtype=np.float64)
    np.testing.assert_allclose(lp[:100], check, atol=1e-3)


def test_switchable_defaults_to_exact():
    target, mu, _ = correlated_gaussian(2, seed=1)
    sw = SwitchableLogDensity(target)
    u = mx.zeros((5, 2))
    np.testing.assert_array_equal(
        np.array(sw.log_prob(u)), np.array(target.log_prob(u))
    )
    assert not sw.use_surrogate
