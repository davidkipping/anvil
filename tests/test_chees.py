import mlx.core as mx
import numpy as np
import pytest

from anvil import HMCSampler, run
from anvil.diagnostics import ess_bulk, nested_rhat, split_rhat
from anvil.kernels.chees import ChEESHMC
from anvil.kernels.metropolis import RandomWalkMetropolis
from anvil.logdensity import FunctionLogDensity
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
@pytest.mark.parametrize("dense", [False, True])
def test_funnel_is_flagged_not_silently_wrong(dense):
    """Neal's funnel defeats a single global step size and a single global
    mass matrix (known ChEES limitation). The requirement is that the run
    *tells* us: divergences and/or bad R-hat, and a v-marginal that visibly
    fails to reach the funnel's true N(0, 9) tails would be caught by the
    diagnostics rather than passing silently.

    Run for the dense preconditioner too. A dense mass matrix is a *global*
    assumption about posterior shape, so the funnel is exactly where it
    could make things worse; the bar is that it must not turn a detectable
    failure into a silent one."""
    dim, n_chains = 10, 512
    target = neals_funnel(dim)
    u0 = _ball(np.zeros(dim), n_chains, dim, seed=7)
    res = run(ChEESHMC(target, dense=dense), target, u0,
              n_warmup=800, n_samples=300, seed=8)
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
        f"funnel (dense={dense}) sampled with v_sd={v_sd:.2f} (true 3.0) "
        "but nothing flagged"
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


# --- dense (full-covariance) preconditioner ---------------------------------

def test_dense_momentum_covariance_identity():
    """The load-bearing test for the dense preconditioner.

    p = (z @ B)/sd must have covariance Sigma^-1, and _sigma_p must apply
    Sigma exactly. Getting the transpose wrong here produces a chain that
    looks healthy step by step but samples the wrong distribution -- it was
    caught in development only because R-hat blew up to 56.
    """
    from anvil.adaptation.moments import (
        dense_factors, init_dense_moments, update_dense_moments)
    from anvil.kernels.chees import ChEESHMC

    rng = np.random.default_rng(0)
    d = 6
    A = rng.standard_normal((d, d))
    Sigma_true = A @ A.T + d * np.eye(d)
    X = rng.multivariate_normal(np.zeros(d), Sigma_true, size=20_000)
    ms = init_dense_moments(d)
    for t in range(1, 120):
        ms = update_dense_moments(ms, X, t)
    sd, R, B = dense_factors(ms)
    sd_n = np.array(sd, dtype=np.float64)
    R_n = np.array(R, dtype=np.float64)
    B_n = np.array(B, dtype=np.float64)
    Sigma = R_n * np.outer(sd_n, sd_n)

    # Sigma p, applied in the factored float32-safe form
    p = mx.array(rng.standard_normal((7, d)).astype(np.float32))
    got = np.array(ChEESHMC._sigma_p(p, sd, R), dtype=np.float64)
    want = np.array(p, dtype=np.float64) @ Sigma
    assert np.abs(got - want).max() / np.abs(want).max() < 1e-5

    # Cov((z @ B)/sd) == Sigma^-1
    z = rng.standard_normal((400_000, d))
    emp = np.cov(((z @ B_n) / sd_n).T)
    inv = np.linalg.inv(Sigma)
    assert np.abs(emp - inv).max() / np.abs(inv).max() < 0.02


@pytest.mark.slow
def test_dense_beats_diagonal_on_correlated_gaussian():
    """Where a dense mass matrix should shine: strong linear correlation,
    no curvature. It must cut the leapfrog count and lift ESS per gradient,
    while still recovering the right posterior."""
    dim, n_chains = 10, 512
    target, mu, cov = correlated_gaussian(dim, rho=0.99, seed=20)
    u0 = _ball(mu, n_chains, dim, seed=21)
    out = {}
    for label, kern in (("diag", ChEESHMC(target, max_leapfrog=128)),
                        ("dense", ChEESHMC(target, max_leapfrog=128, dense=True))):
        res = run(kern, target, u0, n_warmup=400, n_samples=300, seed=22)
        chain = res.get_chain()
        eps = float(res.final_params["step_size"].item())
        T = float(res.final_params["traj_length"].item())
        L = max(1, int(0.5 * T / eps))
        out[label] = dict(ess=ess_bulk(chain).min(), L=L, res=res,
                          rhat=split_rhat(chain).max(),
                          flat=res.get_chain(flat=True).astype(np.float64))

    # the mechanism: dense removes the correlation, so trajectories shorten
    assert out["dense"]["L"] <= 4, out["dense"]["L"]
    assert out["dense"]["L"] < out["diag"]["L"]
    # and the payoff, in samples per gradient evaluation
    eff = lambda o: o["ess"] / o["L"]
    assert eff(out["dense"]) > 5 * eff(out["diag"])

    # correctness is not traded away for it
    sd_true = np.sqrt(np.diag(cov))
    for label in ("diag", "dense"):
        o = out[label]
        assert o["rhat"] < 1.01, (label, o["rhat"])
        assert o["res"].extras["n_divergent"] == 0, label
        mcse = sd_true / np.sqrt(ess_bulk(o["res"].get_chain()))
        assert np.all(np.abs(o["flat"].mean(0) - mu) < 4 * mcse + 1e-3), label
        assert np.all(np.abs(o["flat"].std(0) / sd_true - 1) < 0.05), label

    # the two samplers agree with each other
    assert np.all(np.abs(out["diag"]["flat"].mean(0)
                         - out["dense"]["flat"].mean(0)) < 0.05 * sd_true)


def test_dense_falls_back_when_too_few_chains():
    """A cross-chain covariance estimated from too few chains is noise;
    the kernel must say so and degrade to the diagonal path."""
    target, mu, _ = correlated_gaussian(8, seed=23)
    kern = ChEESHMC(target, dense=True)
    u0 = _ball(mu, 16, 8, seed=24)            # 2*dim chains: not enough
    with pytest.warns(UserWarning, match="4\\*dim"):
        res = run(kern, target, u0, n_warmup=30, n_samples=10, seed=25)
    assert kern.dense is False
    assert "inv_mass" in res.final_params and "corr" not in res.final_params


@pytest.mark.slow
def test_dense_survives_a_degenerate_covariance():
    """A duplicated coordinate makes the correlation matrix singular. The
    ridge escalation must absorb it rather than crashing mid-warmup."""
    dim = 6

    def log_prob(u):                       # u[:, 5] is unconstrained-but-tied
        d = u[:, :5]
        return -0.5 * mx.sum(d * d, axis=-1) - 0.5 * (u[:, 5] - u[:, 0]) ** 2 * 1e6

    target = FunctionLogDensity(log_prob, dim)
    u0 = _ball(np.zeros(dim), 256, dim, seed=26)
    res = run(ChEESHMC(target, dense=True), target, u0,
              n_warmup=120, n_samples=40, seed=27)
    assert np.all(np.isfinite(res.get_chain()))
    assert np.isfinite(float(res.final_params["step_size"].item()))


def test_dense_is_reproducible():
    target, mu, _ = correlated_gaussian(4, rho=0.9, seed=28)
    u0 = _ball(mu, 128, 4, seed=29)
    kw = dict(n_warmup=60, n_samples=20, seed=30)
    a = run(ChEESHMC(target, dense=True), target, u0, **kw).get_chain()
    b = run(ChEESHMC(target, dense=True), target, u0, **kw).get_chain()
    np.testing.assert_array_equal(a, b)


def test_whitened_shape_separates_correlated_from_curved():
    from anvil import whitened_shape
    rng = np.random.default_rng(31)
    A = rng.standard_normal((5, 5))
    cov = A @ A.T + 5 * np.eye(5)
    flat = rng.multivariate_normal(np.zeros(5), cov, size=60_000)
    skew, exkurt = whitened_shape(flat)
    assert np.abs(skew).max() < 0.1 and np.abs(exkurt).max() < 0.2

    x = rng.standard_normal(60_000)
    banana = np.column_stack([x, x ** 2 + 0.3 * rng.standard_normal(60_000)])
    skew, exkurt = whitened_shape(banana)
    assert np.abs(skew).max() > 1.0 and np.abs(exkurt).max() > 2.0

    with pytest.raises(ValueError, match="n_draws, dim"):
        whitened_shape(np.zeros(10))
