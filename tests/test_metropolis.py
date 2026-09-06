import mlx.core as mx
import numpy as np
import pytest

from anvil import EnsembleSampler, run
from anvil.diagnostics import ess_bulk, split_rhat
from anvil.kernels.metropolis import RandomWalkMetropolis
from anvil.logdensity import FunctionLogDensity


def make_correlated_gaussian(dim=10, rho=0.5, seed=0):
    """Target with known moments: N(mu, C), C = D^1/2 R D^1/2, R equicorrelated."""
    rng = np.random.default_rng(seed)
    mu = rng.normal(size=dim)
    scales = np.exp(rng.uniform(-1, 1, size=dim))
    corr = np.full((dim, dim), rho) + (1 - rho) * np.eye(dim)
    cov = np.outer(scales, scales) * corr
    prec = np.linalg.inv(cov)
    mu_mx = mx.array(mu.astype(np.float32))
    prec_mx = mx.array(prec.astype(np.float32))

    def log_prob(u):
        d = u - mu_mx
        return -0.5 * mx.sum((d @ prec_mx) * d, axis=-1)

    return log_prob, mu, cov


@pytest.mark.slow
def test_rwm_recovers_gaussian_moments():
    dim, n_chains = 10, 2048
    log_prob, mu, cov = make_correlated_gaussian(dim)
    target = FunctionLogDensity(log_prob, dim)
    u0 = mx.array(
        (mu + np.random.default_rng(1).normal(size=(n_chains, dim))).astype(
            np.float32
        )
    )
    res = run(
        RandomWalkMetropolis(target),
        target,
        u0,
        n_warmup=3000,
        n_samples=400,
        thin=10,
        seed=42,
    )
    chain = res.get_chain()  # (400, 2048, dim)
    # RWM on a correlated 10-D target has tau ~ several hundred iterations;
    # this is the walking skeleton, so require "clearly mixing", and rely on
    # the ESS-scaled moment checks below for actual correctness.
    assert np.all(split_rhat(chain) < 1.05)
    ess = ess_bulk(chain)
    assert np.all(ess > 3000)

    flat = res.get_chain(flat=True).astype(np.float64)
    # Monte Carlo standard error from measured ESS, 4-sigma tolerance
    mcse_mean = np.sqrt(np.diag(cov) / ess)
    assert np.all(np.abs(flat.mean(axis=0) - mu) < 4 * mcse_mean + 1e-3)
    sd_true = np.sqrt(np.diag(cov))
    assert np.all(np.abs(flat.std(axis=0) - sd_true) / sd_true < 0.1)


def test_seed_reproducibility():
    dim = 3
    log_prob, mu, _ = make_correlated_gaussian(dim)
    target = FunctionLogDensity(log_prob, dim)
    u0 = mx.zeros((64, dim))
    kw = dict(n_warmup=50, n_samples=40, seed=11)
    c1 = run(RandomWalkMetropolis(target), target, u0, **kw).get_chain()
    c2 = run(RandomWalkMetropolis(target), target, u0, **kw).get_chain()
    np.testing.assert_array_equal(c1, c2)
    c3 = run(RandomWalkMetropolis(target), target, u0, n_warmup=50, n_samples=40,
             seed=12).get_chain()
    assert not np.array_equal(c1, c3)


def test_acceptance_in_sane_range():
    dim = 5
    log_prob, mu, _ = make_correlated_gaussian(dim)
    target = FunctionLogDensity(log_prob, dim)
    u0 = mx.array(np.random.default_rng(0).normal(size=(256, dim)).astype(np.float32))
    res = run(RandomWalkMetropolis(target), target, u0, n_warmup=800, n_samples=200,
              seed=1)
    mean_accept = float(res.accept_fraction.mean())
    assert 0.1 < mean_accept < 0.45  # adapted toward 0.234


def test_ensemble_sampler_facade():
    dim, nwalkers = 4, 128
    log_prob, mu, _ = make_correlated_gaussian(dim)
    sampler = EnsembleSampler(nwalkers, dim, log_prob, seed=5)
    p0 = np.random.default_rng(0).normal(size=(nwalkers, dim))
    sampler.run_mcmc(p0, 100, warmup=200)
    chain = sampler.get_chain()
    assert chain.shape == (100, nwalkers, dim)
    assert sampler.get_chain(discard=50).shape == (50, nwalkers, dim)
    assert sampler.get_chain(flat=True).shape == (100 * nwalkers, dim)
    assert sampler.get_log_prob().shape == (100, nwalkers)
    assert sampler.acceptance_fraction.shape == (nwalkers,)


def test_facade_rejects_numpy_log_prob():
    def bad_log_prob(theta):  # classic emcee-style per-walker numpy fn
        return -0.5 * float(np.sum(np.asarray(theta) ** 2))

    with pytest.raises(TypeError, match="batched"):
        EnsembleSampler(32, 3, bad_log_prob)
