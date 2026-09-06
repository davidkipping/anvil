import mlx.core as mx
import numpy as np
import pytest

from anvil import HMCSampler, run
from anvil.diagnostics import ess_bulk, nested_rhat, split_rhat
from anvil.kernels.chees import ChEESHMC
from anvil.kernels.metropolis import RandomWalkMetropolis
from anvil.targets import correlated_gaussian, neals_funnel


def _ball(mu, n, dim, seed=0):
    rng = np.random.default_rng(seed)
    return mx.array((mu + rng.standard_normal((n, dim))).astype(np.float32))


@pytest.mark.slow
def test_chees_recovers_gaussian_moments():
    dim, n_chains = 10, 1024
    target, mu, cov = correlated_gaussian(dim, rho=0.5, seed=0)
    u0 = _ball(mu, n_chains, dim, seed=1)
    res = run(ChEESHMC(target), target, u0, n_warmup=500, n_samples=300, seed=2)
    chain = res.get_chain()
    assert res.extras["n_divergent"] == 0
    assert np.all(split_rhat(chain) < 1.01)
    ess = ess_bulk(chain)
    # HMC with adapted trajectories should give high per-draw efficiency
    assert ess.min() > 0.2 * chain.shape[0] * chain.shape[1]
    flat = res.get_chain(flat=True).astype(np.float64)
    sd = np.sqrt(np.diag(cov))
    assert np.all(np.abs(flat.mean(0) - mu) < 0.02 * sd)
    assert np.all(np.abs(flat.std(0) - sd) / sd < 0.02)
    # preconditioner recovered per-dimension scales
    sigma_est = np.array(mx.sqrt(res.final_params["inv_mass"]))
    assert np.all(np.abs(sigma_est - sd) / sd < 0.05)


@pytest.mark.slow
def test_chees_beats_rwm_on_correlated_target():
    dim, n_chains = 16, 512
    target, mu, cov = correlated_gaussian(dim, rho=0.9, seed=3)
    u0 = _ball(mu, n_chains, dim, seed=4)
    n_warm, n_samp = 600, 200

    res_h = run(ChEESHMC(target), target, u0, n_warmup=n_warm,
                n_samples=n_samp, seed=5)
    res_r = run(RandomWalkMetropolis(target), target, u0, n_warmup=n_warm,
                n_samples=n_samp, seed=6)
    ess_h = ess_bulk(res_h.get_chain()).min()
    ess_r = ess_bulk(res_r.get_chain()).min()
    # strongly correlated target: gradients + long trajectories dominate
    assert ess_h > 10 * ess_r


@pytest.mark.slow
def test_funnel_is_flagged_not_silently_wrong():
    """Neal's funnel defeats a single global step size + diagonal
    preconditioner (known ChEES limitation). The requirement is that the
    run *tells* us: divergences and/or bad R-hat, and a v-marginal that
    visibly fails to reach the funnel's true N(0, 9) tails would be caught
    by the diagnostics rather than passing silently."""
    dim, n_chains = 10, 512
    target = neals_funnel(dim)
    u0 = _ball(np.zeros(dim), n_chains, dim, seed=7)
    res = run(ChEESHMC(target), target, u0, n_warmup=800, n_samples=300, seed=8)
    chain = res.get_chain()
    v = chain[..., -1]
    v_sd = v.reshape(-1).std()
    flagged = (
        res.extras["n_divergent"] > 0
        or split_rhat(chain).max() > 1.01
        or nested_rhat(chain, n_superchains=32).max() > 1.01
    )
    biased = abs(v_sd - 3.0) > 0.3
    assert flagged or not biased, (
        f"funnel sampled with v_sd={v_sd:.2f} (true 3.0) but nothing flagged"
    )


def test_warmup_freeze_and_reproducibility():
    target, mu, _ = correlated_gaussian(3, seed=9)
    u0 = _ball(mu, 128, 3, seed=10)
    kw = dict(n_warmup=200, n_samples=50, seed=11)
    r1 = run(ChEESHMC(target), target, u0, **kw)
    r2 = run(ChEESHMC(target), target, u0, **kw)
    np.testing.assert_array_equal(r1.get_chain(), r2.get_chain())
    eps = float(r1.final_params["step_size"].item())
    T = float(r1.final_params["traj_length"].item())
    assert 0.01 < eps < 10.0
    assert eps <= T + 1e-6 or T > 0.0
    assert np.isfinite(eps) and np.isfinite(T)


def test_hmc_facade():
    target, mu, _ = correlated_gaussian(4, seed=12)
    sampler = HMCSampler(256, 4, target.log_prob, seed=13)
    p0 = np.random.default_rng(14).normal(size=(256, 4)) + mu
    sampler.run_mcmc(p0, 100, warmup=300)
    assert sampler.get_chain().shape == (100, 256, 4)
    assert sampler.n_divergent >= 0
    assert 0.4 < float(sampler.acceptance_fraction.mean()) <= 1.0


def test_nested_rhat_behavior():
    rng = np.random.default_rng(15)
    # well-mixed: all chains iid from the same distribution
    good = rng.normal(size=(20, 64, 2))
    assert np.all(nested_rhat(good, n_superchains=8) < 1.02)
    # broken: one superchain systematically offset
    bad = good.copy()
    bad[:, :8, 0] += 5.0
    assert nested_rhat(bad, n_superchains=8)[0] > 1.1
    # works even with a single draw per chain
    single_draw = rng.normal(size=(1, 64))
    assert nested_rhat(single_draw, n_superchains=8)[0] < 1.2


def test_chees_requires_grad():
    from anvil.logdensity import FunctionLogDensity

    target = FunctionLogDensity(lambda u: -mx.sum(u * u, -1), 2,
                                supports_grad=False)
    with pytest.raises(ValueError, match="gradient"):
        ChEESHMC(target)
