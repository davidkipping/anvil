import mlx.core as mx
import numpy as np
import pytest

from anvil import DEMove, EnsembleSampler, StretchMove, run
from anvil.diagnostics import ess_bulk, split_rhat
from anvil.kernels.ensemble import EnsembleKernel
from anvil.targets import correlated_gaussian, rosenbrock


def _init_ball(mu, n, dim, seed=0, scale=1.0):
    rng = np.random.default_rng(seed)
    return mx.array((mu + scale * rng.standard_normal((n, dim))).astype(np.float32))


@pytest.mark.slow
def test_stretch_recovers_gaussian_moments():
    dim, n_walkers = 5, 512
    target, mu, cov = correlated_gaussian(dim, rho=0.4, seed=0)
    u0 = _init_ball(mu, n_walkers, dim, seed=1)
    kernel = EnsembleKernel(target, moves=[(StretchMove(), 1.0)], seed=0)
    res = run(kernel, target, u0, n_warmup=1500, n_samples=500, thin=6, seed=2)
    chain = res.get_chain()
    assert np.all(split_rhat(chain) < 1.03)
    ess = ess_bulk(chain)
    assert np.all(ess > 2000)
    flat = res.get_chain(flat=True).astype(np.float64)
    sd_true = np.sqrt(np.diag(cov))
    mcse = sd_true / np.sqrt(ess)
    assert np.all(np.abs(flat.mean(axis=0) - mu) < 4 * mcse + 2e-3)
    assert np.all(np.abs(flat.std(axis=0) - sd_true) / sd_true < 0.06)


@pytest.mark.slow
def test_mixed_stretch_de_moves():
    dim, n_walkers = 4, 256
    target, mu, cov = correlated_gaussian(dim, rho=0.3, seed=3)
    u0 = _init_ball(mu, n_walkers, dim, seed=4)
    kernel = EnsembleKernel(
        target, moves=[(StretchMove(), 0.7), (DEMove(), 0.3)], seed=1
    )
    res = run(kernel, target, u0, n_warmup=1200, n_samples=400, thin=5, seed=5)
    flat = res.get_chain(flat=True).astype(np.float64)
    sd_true = np.sqrt(np.diag(cov))
    assert np.all(np.abs(flat.mean(axis=0) - mu) < 0.1 * sd_true)
    assert np.all(np.abs(flat.std(axis=0) - sd_true) / sd_true < 0.08)


@pytest.mark.slow
def test_rosenbrock_mean_recovery():
    target, mean = rosenbrock()
    n_walkers = 512
    u0 = _init_ball(np.array([1.0, 2.0]), n_walkers, 2, seed=6)
    kernel = EnsembleKernel(target, seed=2)
    res = run(kernel, target, u0, n_warmup=3000, n_samples=800, thin=6, seed=7)
    flat = res.get_chain(flat=True).astype(np.float64)
    # E[x]=1, E[y]=1+sigma_x^2=2; sd(y) is large (heavy banana tail)
    assert abs(flat[:, 0].mean() - mean[0]) < 0.05
    assert abs(flat[:, 1].mean() - mean[1]) < 0.15


class _BrokenStretchMove(StretchMove):
    """Stretch move WITHOUT the z^(d-1) Jacobian factor: not a valid MCMC
    kernel; the ensemble must visibly contract."""

    def propose(self, key, S, C):
        prop, extra = super().propose(key, S, C)
        return prop, mx.zeros_like(extra)


@pytest.mark.slow
def test_z_jacobian_factor_is_load_bearing():
    dim, n_walkers = 8, 512
    target, mu, cov = correlated_gaussian(dim, rho=0.0, seed=8)
    sd_true = np.sqrt(np.diag(cov))
    u0 = _init_ball(mu, n_walkers, dim, seed=9)

    def run_with(move, seed):
        kernel = EnsembleKernel(target, moves=[(move, 1.0)], seed=seed)
        res = run(kernel, target, u0, n_warmup=1500, n_samples=400, thin=4,
                  seed=seed)
        return res.get_chain(flat=True).astype(np.float64).std(axis=0)

    sd_good = run_with(StretchMove(), 10)
    sd_bad = run_with(_BrokenStretchMove(), 11)
    assert np.all(np.abs(sd_good - sd_true) / sd_true < 0.05)
    # dropping the factor biases the sampled scale noticeably
    assert np.mean(np.abs(sd_bad - sd_true) / sd_true) > 0.10


@pytest.mark.slow
def test_parity_with_emcee():
    emcee = pytest.importorskip("emcee")
    dim, n_walkers, n_steps = 3, 64, 3000
    target, mu, cov = correlated_gaussian(dim, rho=0.5, seed=12)
    prec = np.linalg.inv(cov)

    def log_prob_np(theta):
        d = theta - mu
        return -0.5 * d @ prec @ d

    rng = np.random.default_rng(13)
    p0 = mu + rng.standard_normal((n_walkers, dim))

    es = emcee.EnsembleSampler(n_walkers, dim, log_prob_np)
    es.run_mcmc(p0, n_steps, progress=False)
    emcee_flat = es.get_chain(discard=n_steps // 2, flat=True)

    ours = EnsembleSampler(n_walkers, dim, target.log_prob, seed=14)
    ours.run_mcmc(p0, n_steps // 2, warmup=n_steps // 2)
    ours_flat = ours.get_chain(flat=True).astype(np.float64)

    sd_true = np.sqrt(np.diag(cov))
    for flat, label in ((emcee_flat, "emcee"), (ours_flat, "anvil")):
        assert np.all(np.abs(flat.mean(axis=0) - mu) < 0.15 * sd_true), label
        assert np.all(np.abs(flat.std(axis=0) - sd_true) / sd_true < 0.10), label
    # the two posteriors agree with each other
    assert np.all(
        np.abs(emcee_flat.mean(axis=0) - ours_flat.mean(axis=0)) < 0.15 * sd_true
    )


def test_walker_count_constraints():
    target, mu, _ = correlated_gaussian(4, seed=15)
    kernel = EnsembleKernel(target)
    with pytest.raises(ValueError, match="even"):
        kernel.init(mx.random.key(0), mx.zeros((7, 4)), target)
    with pytest.raises(ValueError, match="2\\*dim"):
        kernel.init(mx.random.key(0), mx.zeros((6, 4)), target)
    with pytest.warns(UserWarning, match="4\\*dim"):
        kernel.init(mx.random.key(0), mx.zeros((10, 4)), target)


def test_facade_uses_stretch_and_accepts_moves():
    target, mu, _ = correlated_gaussian(3, seed=16)
    sampler = EnsembleSampler(
        64, 3, target.log_prob,
        moves=[(StretchMove(a=1.7), 0.5), (DEMove(), 0.5)], seed=17,
    )
    p0 = np.random.default_rng(18).normal(size=(64, 3))
    sampler.run_mcmc(p0, 50, warmup=100)
    assert sampler.get_chain().shape == (50, 64, 3)
    assert 0.05 < float(sampler.acceptance_fraction.mean()) < 0.9
