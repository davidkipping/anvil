"""anvil.optimize: batched maximum likelihood / MAP.

What is tested directly: each objective reaches its own analytic optimum
(likelihood alone, likelihood plus priors, and the u-space density with its
Jacobian), on targets where those three differ; an optimum on a bound is
flagged rather than chased; a start that cannot converge says so; and a
start reported converged really is at the optimum -- the failure L-BFGS
alone showed on Elome's ill-conditioned transit model.
"""

import math

import mlx.core as mx
import numpy as np
import pytest

import anvil
from anvil import (FunctionLogDensity, ParamSpec, Transform,
                   TransformedLogDensity, optimize, run)
from anvil.kernels.chees import ChEESHMC
from anvil.priors import Gaussian, LogUniform
from anvil.targets import (correlated_gaussian, make_transit_target,
                           neals_funnel, rosenbrock)

GRAD_METHODS = ["mixed", "lbfgs", "newton"]


# -- reaching the optimum ---------------------------------------------------

@pytest.mark.parametrize("dim", [3, 10, 30])
@pytest.mark.parametrize("method", GRAD_METHODS)
def test_gaussian_mode_and_hessian(dim, method):
    t, mu, cov = correlated_gaussian(dim, rho=0.9, seed=0)
    rng = np.random.default_rng(1)
    r = optimize(t, mu + 3 * rng.standard_normal((64, dim)),
                 objective="density", method=method)
    sd = np.sqrt(np.diag(cov))
    assert r.converged
    assert np.mean(r.status == "converged") > 0.7
    # tol bounds the gain left (1e-2 nats, i.e. < 0.15 sd), and the last
    # Newton step on a quadratic lands far inside it
    assert np.max(np.abs(r.u - mu) / sd) < (0.15 if method == "lbfgs" else 1e-3)
    prec = np.linalg.inv(cov)
    assert np.max(np.abs(r.hessian_u + prec)) < 1e-3 * np.max(np.abs(prec))
    np.testing.assert_allclose(r.cov_u, cov, rtol=1e-3, atol=1e-4 * cov.max())


def test_rosenbrock_mode_not_mean():
    rb, mean = rosenbrock(1.0, 0.1)
    rng = np.random.default_rng(0)
    r = optimize(rb, 1.5 * rng.standard_normal((128, 2)), objective="density")
    assert r.converged
    # the mode is (1, 1); the mean is (1, 2)
    np.testing.assert_allclose(r.u, [1.0, 1.0], atol=0.02)
    assert r.value > -1e-2
    assert abs(mean[1] - 2.0) < 1e-12


# -- what is maximised ------------------------------------------------------

def _two_param_target():
    """x on [0, 10] with a Gaussian(5, 1) prior; y on [0.1, 100] with a
    LogUniform prior. Likelihood N(x; 3, 1) N(y; 2, 0.5)."""
    specs = [ParamSpec("x", 0.0, 10.0, prior=Gaussian(5.0, 1.0)),
             ParamSpec("y", 0.1, 100.0, prior=LogUniform())]
    tr = Transform(specs)

    def loglike(v):
        return -0.5 * (v[:, 0] - 3.0) ** 2 - 0.5 * ((v[:, 1] - 2.0) / 0.5) ** 2

    def loglike_hi(v):
        return -0.5 * (v[:, 0] - 3.0) ** 2 - 0.5 * ((v[:, 1] - 2.0) / 0.5) ** 2

    return TransformedLogDensity(loglike, tr, model_log_prob_hi=loglike_hi), tr


def _density_mode_grid(target, tr):
    """The u-space mode by brute force in float64: likelihood + priors +
    log-Jacobian on a fine grid around the model-space MAP."""
    v_map = np.array([4.0, 1.0 + math.sqrt(0.75)])
    u_map = tr.from_model_np(v_map[None])[0]
    best = None
    for span in (0.5, 0.02, 0.0008):
        g = np.linspace(-span, span, 401)
        c = u_map if best is None else best
        U = np.stack(np.meshgrid(c[0] + g, c[1] + g, indexing="ij"), -1).reshape(-1, 2)
        V = tr.model_np(U)
        lp = (-0.5 * (V[:, 0] - 3) ** 2 - 0.5 * ((V[:, 1] - 2) / 0.5) ** 2
              + tr.log_prior_np(V) + tr.log_det_jac_np(U))
        best = U[np.argmax(lp)]
    return best


def test_objectives_reach_their_own_optima():
    target, tr = _two_param_target()
    rng = np.random.default_rng(0)
    u0 = rng.standard_normal((64, 2))
    mle = optimize(target, u0, objective="mle")
    mp = optimize(target, u0, objective="map")
    dens = optimize(target, u0, objective="density")
    for r in (mle, mp, dens):
        assert r.converged

    # MLE: the likelihood's peak, priors and Jacobian ignored
    np.testing.assert_allclose(mle.model, [3.0, 2.0], atol=1e-4)
    # MAP in model space: x from N(3,1)N(5,1) -> 4; y from
    # -2(y-2)^2 - log y -> y^2 - 2y + 1/4 = 0
    np.testing.assert_allclose(mp.model, [4.0, 1.0 + math.sqrt(0.75)], atol=1e-4)
    # the u-space mode includes the Jacobian, and so differs from both
    u_star = _density_mode_grid(target, tr)
    np.testing.assert_allclose(dens.u, u_star, atol=2e-3)
    assert np.max(np.abs(dens.model - mp.model)) > 0.05


def test_log_value_constants():
    target, tr = _two_param_target()
    u0 = np.random.default_rng(0).standard_normal((32, 2))
    mle = optimize(target, u0, objective="mle")
    mp = optimize(target, u0, objective="map")
    v = mp.model
    ll = -0.5 * (v[0] - 3) ** 2 - 0.5 * ((v[1] - 2) / 0.5) ** 2
    # MAP's absolute value is the NORMALISED model-space log posterior density
    norm_prior = sum(p.log_density_np(np.array([v[i]]))[0]
                     for i, p in enumerate(tr.priors))
    assert mp.log_value == pytest.approx(ll + norm_prior, abs=1e-6)
    assert mle.log_value == pytest.approx(0.0, abs=1e-6)        # ll at its peak
    assert mle.value == mle.log_value                             # no constants


def test_mle_needs_a_separable_likelihood():
    t, mu, _ = correlated_gaussian(3, rho=0.5, seed=0)
    with pytest.raises(ValueError, match="objective='density'"):
        optimize(t, mu)
    with pytest.raises(ValueError, match="objective must be"):
        optimize(t, mu, objective="ml")


def test_starts_default_to_prior_draws_and_need_a_proper_prior():
    target, _ = _two_param_target()
    r = optimize(target, objective="mle", n_starts=16, seed=3)
    assert r.n_starts == 16 and r.converged
    free = TransformedLogDensity(lambda v: -0.5 * mx.sum(v * v, axis=-1),
                                 Transform([ParamSpec("a"), ParamSpec("b")]))
    with pytest.raises(ValueError, match="improper"):
        optimize(free)
    r = optimize(free, np.array([1.0, -2.0]), n_starts=8)      # a single point is spread
    assert r.n_starts == 8
    np.testing.assert_allclose(r.model, [0.0, 0.0], atol=1e-4)


# -- bounds, failures, multimodality ----------------------------------------

def test_optimum_on_a_bound_is_flagged_not_chased():
    specs = [ParamSpec("p", 0.0, 1.0), ParamSpec("q", 0.0, 1.0)]
    target = TransformedLogDensity(
        lambda v: 5.0 * v[:, 0] - 0.5 * ((v[:, 1] - 0.3) / 0.05) ** 2,
        Transform(specs))
    r = optimize(target, np.random.default_rng(0).standard_normal((32, 2)),
                 max_iter=300)
    assert r.converged
    assert r.at_bound.tolist() == [True, False]
    assert r.model[0] > 1.0 - 1e-5
    assert r.model[1] == pytest.approx(0.3, abs=0.05 * 0.05)   # 0.05 sd
    assert np.isnan(r.sd[0]) and np.isfinite(r.sd[1])
    # sampler starts keep off the bound itself
    u = np.array(r.ball(8))
    assert np.all(np.abs(u[:, 0]) < 9.0)
    assert "[on bound]" in str(r)


def test_no_maximum_is_reported_not_hung():
    # rises without bound along x0 (Neal's funnel will not do: with x = 0
    # its density peaks at v = -(dim - 1) * 9 / 2)
    t = FunctionLogDensity(lambda u: u[:, 0] - 0.5 * u[:, 1] ** 2, 2)
    r = optimize(t, np.random.default_rng(0).standard_normal((16, 2)),
                 objective="density", max_iter=150)
    assert not r.converged
    assert not np.any(r.status == "converged")
    assert r.n_iter.max() <= 150


def test_funnel_mode():
    r = optimize(neals_funnel(5), np.random.default_rng(0).standard_normal((32, 5)),
                 objective="density")
    assert r.converged
    # sd of v at the mode is 3
    np.testing.assert_allclose(r.u, [0, 0, 0, 0, -18.0], atol=0.1)


def test_infeasible_starts_fail_cleanly():
    def lp(u):
        return mx.where(u[:, 0] > 0, -0.5 * mx.sum((u - 1.0) ** 2, axis=-1), -mx.inf)
    t = FunctionLogDensity(lp, 2)
    u0 = np.array([[-1.0, 0.0]] * 4 + [[0.5, 0.5]] * 4)
    r = optimize(t, u0, objective="density")
    assert list(r.status[:4]) == ["failed"] * 4
    np.testing.assert_allclose(r.u, [1.0, 1.0], atol=1e-4)
    with pytest.raises(RuntimeError, match="every start failed"):
        optimize(t, u0[:4], objective="density")


def test_multimodal_reports_both_optima():
    # two separated Gaussians; the one at +3 is higher by 2 nats
    def lp(u):
        x = u[:, 0]
        a = -0.5 * (x - 3.0) ** 2 / 0.25
        b = -0.5 * (x + 3.0) ** 2 / 0.25 - 2.0
        return mx.logaddexp(a, b)
    t = FunctionLogDensity(lp, 1)
    r = optimize(t, np.linspace(-6, 6, 64)[:, None], objective="density")
    assert r.u[0] == pytest.approx(3.0, abs=1e-3)
    assert len(r.optima) == 2
    assert r.optima[0]["u"][0] == pytest.approx(3.0, abs=1e-3)
    assert r.optima[1]["u"][0] == pytest.approx(-3.0, abs=1e-3)
    assert r.optima[0]["value"] - r.optima[1]["value"] == pytest.approx(2.0, abs=1e-3)
    assert "multimodality" in str(r)


# -- gradient-free ----------------------------------------------------------

def test_de_reaches_the_optimum_without_gradients():
    _, mu, cov = correlated_gaussian(3, rho=0.5, seed=0)
    prec = mx.array(np.linalg.inv(cov).astype(np.float32))
    m = mx.array(mu.astype(np.float32))

    def lp(u):
        d = u - m
        return -0.5 * mx.sum((d @ prec) * d, axis=-1)
    t = FunctionLogDensity(lp, 3, supports_grad=False)
    rng = np.random.default_rng(0)
    u0 = mu + 2 * rng.standard_normal((64, 3))
    with pytest.raises(ValueError, match="method='de'"):
        optimize(t, u0, objective="density")
    r = optimize(t, u0, objective="density", method="de", max_iter=2000)
    assert r.converged
    assert np.max(np.abs(r.u - mu) / np.sqrt(np.diag(cov))) < 0.15
    assert r.hessian_u is None


# -- feeding the samplers ---------------------------------------------------

def test_starts_seed_a_chees_run():
    t, mu, cov = correlated_gaussian(5, rho=0.7, seed=2)
    r = optimize(t, mu + 2 * np.random.default_rng(0).standard_normal((32, 5)),
                 objective="density")
    for u0 in (r.ball(64, spread=0.05), r.laplace_starts(64)):
        assert u0.shape == (64, 5) and u0.dtype == mx.float32
        res = run(ChEESHMC(t), t, u0, n_warmup=300, n_samples=200, seed=1)
        chain = res.get_chain()
        d = anvil.diagnose(chain)
        assert np.max(d.rhat) < 1.05
        np.testing.assert_allclose(chain.reshape(-1, 5).mean(0), mu,
                                   atol=0.1 * np.sqrt(np.diag(cov)).max())


def test_laplace_starts_pull_back_bad_draws():
    t, mu, _ = correlated_gaussian(4, rho=0.9, seed=0)
    r = optimize(t, mu + np.random.default_rng(0).standard_normal((16, 4)),
                 objective="density")
    u = np.array(r.laplace_starts(256, scale=20.0, sigma_cap=100.0, pull_nats=5.0))
    lp = np.array(t.log_prob(mx.array(u.astype(np.float32))))
    # 12 halvings from scale 20 leave every draw well inside 5 nats
    assert np.all(lp > r.value - 5.0)


# -- plumbing ---------------------------------------------------------------

def test_progress_callback_and_small_batches(capsys):
    t, mu, _ = correlated_gaussian(4, rho=0.5, seed=0)
    seen = []
    r = optimize(t, mu + np.random.default_rng(0).standard_normal((40, 4)),
                 objective="density", max_rows=16, progress=1,
                 callback=lambda phase, it, info: seen.append((phase, it, info)))
    assert r.converged
    assert seen and seen[0][0] == "optimize" and set(seen[0][2]) >= {"best", "n_active"}
    assert "optimize" in capsys.readouterr().out
    assert r.n_grad_rows > 0 and r.n_value_rows > 0


def test_physical_units_and_sd():
    tt = make_transit_target(n_data=5_000, seed=0)
    u_true = tt.transform.from_model_np(tt.truth_model[None])[0]
    r = optimize(tt.target, u_true + 0.1 * np.random.default_rng(0).standard_normal((32, u_true.size)))
    assert r.converged
    np.testing.assert_allclose(r.physical, tt.transform.to_physical(r.model))
    assert np.all(np.isfinite(r.sd)) and np.all(r.sd > 0)
    # the float64 value is what log_likelihood_hi gives at the best point
    v_hi = float(np.array(tt.target.log_likelihood_hi(mx.array(r.u[None].astype(np.float32))))[0])
    assert r.value == pytest.approx(v_hi, abs=1e-9)
    assert r.log_value == pytest.approx(v_hi + tt.target.model_log_norm, abs=1e-9)


def test_transit_methods_agree():
    tt = make_transit_target(n_data=20_000, seed=0)
    u_true = tt.transform.from_model_np(tt.truth_model[None])[0]
    u0 = u_true + 0.25 * np.random.default_rng(0).standard_normal((128, u_true.size))
    mixed = optimize(tt.target, u0, method="mixed")
    newton = optimize(tt.target, u0, method="newton")
    assert mixed.converged and newton.converged
    assert mixed.value == pytest.approx(newton.value, abs=1e-3)
    assert np.mean(mixed.status == "converged") > 0.8
    # the MLE is not the truth: the noise moves it, by a few nats here
    truth = float(np.array(tt.target.log_likelihood_hi(
        mx.array(u_true[None].astype(np.float32))))[0])
    assert mixed.value > truth + 1.0


def test_sampler_draws_unchanged_by_the_new_accessor():
    # log_likelihood is a new public view of model_log_prob; log_prob must be
    # untouched (the 0.7.1 golden reference in test_priors guards the draws)
    target, tr = _two_param_target()
    u = mx.array(np.random.default_rng(0).standard_normal((8, 2)).astype(np.float32))
    v = tr.to_model(u)
    expect = target.log_likelihood(u) + tr.log_det_jac(u) + tr.log_prior(v)
    np.testing.assert_array_equal(np.array(target.log_prob(u)), np.array(expect))


# -- the case that set the design -------------------------------------------

@pytest.mark.slow
def test_elome_lbfgs_does_not_claim_false_convergence():
    ea = pytest.importorskip("elome.anvil")
    tt = ea.make_target(n_data=20_000, seed=0)
    u_true = tt.transform.from_model_np(np.asarray(tt.truth_model)[None])[0]
    u0 = u_true + 0.25 * np.random.default_rng(0).standard_normal((128, u_true.size))
    mixed = optimize(tt.target, u0, method="mixed")
    lbfgs = optimize(tt.target, u0, method="lbfgs")
    assert mixed.converged
    assert np.mean(mixed.status == "converged") > 0.9
    # L-BFGS alone stops short on this condition-number-8e6 problem; it
    # must say so instead of calling those starts converged
    # starts whose own basin holds the global optimum (mixed, from the same
    # start, reached it) but which L-BFGS left well over tol = 0.01 nats
    # below it; other starts may sit at genuine local optima, correctly
    # converged, and one within ~tol may be called converged by design
    top = mixed.values.max()
    short = (lbfgs.values < top - 0.05) & (mixed.values > top - 0.01)
    assert short.sum() > 50
    assert not np.any(lbfgs.status[short] == "converged")


# -- 0.9.1: review fixes ----------------------------------------------------

def test_lbfgs_takes_steps_far_shorter_than_its_grid():
    # posterior sd 1e-4 in a unit box: ~4e-4 in u, 25 times below the
    # shortest step the line search used to try along the unit gradient,
    # so before 0.9.1 no L-BFGS start could move at all
    t = TransformedLogDensity(lambda v: -0.5 * ((v[:, 0] - 0.5) / 1e-4) ** 2,
                              Transform([ParamSpec("p", 0.0, 1.0)]))
    u0 = 0.01 * np.random.default_rng(0).standard_normal((16, 1))
    r = optimize(t, u0, method="lbfgs")
    assert np.all(r.status == "converged")
    assert r.model[0] == pytest.approx(0.5, abs=2e-5)
    # beside a well-scaled parameter the problem is ill-conditioned (1e6),
    # which is what "mixed" is for
    specs = [ParamSpec("p", 0.0, 1.0), ParamSpec("q", -5.0, 5.0)]
    t2 = TransformedLogDensity(
        lambda v: -0.5 * ((v[:, 0] - 0.5) / 1e-4) ** 2 - 0.5 * (v[:, 1] - 1.0) ** 2,
        Transform(specs))
    u0 = np.random.default_rng(0).standard_normal((16, 2)) * [0.01, 1.0]
    r = optimize(t2, u0)
    assert np.all(r.status == "converged")
    assert r.model[0] == pytest.approx(0.5, abs=2e-5)
    assert r.model[1] == pytest.approx(1.0, abs=0.02)


def test_optimum_beside_an_infeasible_edge_still_converges():
    # the optimum (u = 0, sd 0.01) is 5e-4 from a wall: a central-difference
    # point 1e-3 out lands past it
    def lp(u):
        q = -0.5 * mx.sum((u / 0.01) ** 2, axis=-1)
        return mx.where(u[:, 0] > 5e-4, -mx.inf, q)
    t = FunctionLogDensity(lp, 2)
    u0 = -0.02 + 0.005 * np.random.default_rng(0).standard_normal((16, 2))
    r = optimize(t, u0, objective="density")
    assert r.converged
    assert not np.any(r.status == "failed")
    np.testing.assert_allclose(r.u, [0.0, 0.0], atol=1e-4)
    assert r.cov_u is not None
    np.testing.assert_allclose(np.sqrt(np.diag(r.cov_u)), [0.01, 0.01], rtol=0.05)


def test_best_point_is_certified_by_converged_starts_at_its_peak():
    from anvil.optimize import _Evaluator, _certified
    t = FunctionLogDensity(lambda u: -0.5 * mx.sum(u * u, axis=-1)
                           + 3.0 * mx.exp(-0.5 * mx.sum((u - 6.0) ** 2, axis=-1) / 0.01), 2)
    ev = _Evaluator(t.log_prob, compile_step=False, max_rows=64)
    no = np.zeros(2, bool)
    # the highest start stalled 0.008 below the peak at the origin; a
    # converged one sits 0.009 below that, on the same peak
    X = np.array([[0.1, 0.06], [0.13, 0.03], [3.0, 3.0]])
    F = ev.value(X)
    status = np.array(["stalled", "converged", "converged"], dtype=object)
    assert F[0] > F[1] > F[0] - 1e-2
    assert _certified(ev, X, F, status, 0, 1e-2, no, no)
    # a converged start as high but on another peak certifies nothing
    X2 = np.array([[0.1, 0.06], [6.0, 6.0]])
    F2 = ev.value(X2)
    F2[1] = F2[0] - 1e-3
    assert not _certified(ev, X2, F2, status[:2], 0, 1e-2, no, no)


def test_one_optimum_on_a_bound_is_counted_once():
    specs = [ParamSpec("p", 0.0, 1.0), ParamSpec("q", 0.0, 1.0)]
    t = TransformedLogDensity(
        lambda v: 5.0 * v[:, 0] - 0.5 * ((v[:, 1] - 0.3) / 0.05) ** 2,
        Transform(specs))
    r = optimize(t, np.random.default_rng(0).standard_normal((32, 2)), max_iter=300)
    assert r.at_bound[0]
    assert len(r.optima) == 1
    assert r.optima[0]["n_starts"] > 16
    assert "multimodality" not in str(r)


def test_one_broad_optimum_is_counted_once():
    # sd 3 in u: converged starts sit well over 0.05 apart
    t = FunctionLogDensity(lambda u: -0.5 * mx.sum((u / 3.0) ** 2, axis=-1), 2)
    r = optimize(t, 10 * np.random.default_rng(0).standard_normal((64, 2)),
                 objective="density", method="lbfgs")
    assert len(r.optima) == 1


def test_de_partners_are_distinct():
    from anvil.optimize import _de_partners
    rng = np.random.default_rng(0)
    for n in (4, 5, 64):
        for _ in range(200):
            r1, r2 = _de_partners(rng, n)
            i = np.arange(n)
            assert np.all((r1 != i) & (r2 != i) & (r1 != r2))
            assert r1.min() >= 0 and r2.max() < n


def test_result_keeps_no_compiled_graph():
    from anvil.optimize import _Evaluator
    t, mu, _ = correlated_gaussian(3, rho=0.5, seed=0)
    r = optimize(t, mu + np.random.default_rng(0).standard_normal((8, 3)),
                 objective="density")
    held = [v for v in vars(r).values()
            if isinstance(getattr(v, "__self__", None), _Evaluator)
            or isinstance(v, _Evaluator)]
    assert held == []
    assert np.array(r.laplace_starts(16)).shape == (16, 3)


def test_a_float64_path_that_fails_everywhere_falls_back_to_float32():
    # the float64 rescoring used to call nanargmax on all-NaN values and
    # raise after the whole search had finished
    specs = [anvil.ParamSpec("a", -5.0, 5.0), anvil.ParamSpec("b", -5.0, 5.0)]

    def lp(v):
        return -0.5 * mx.sum((v - 1.0) ** 2, axis=-1)

    t = anvil.TransformedLogDensity(lp, anvil.Transform(specs),
                                    model_log_prob_hi=lambda v: mx.full(v.shape[:1], mx.nan))
    with pytest.warns(RuntimeWarning, match="float64"):
        r = anvil.optimize(t, n_starts=16, seed=0)
    assert np.allclose(r.model, 1.0, atol=1e-2) and np.isfinite(r.value)
