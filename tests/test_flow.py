"""Flow mode (anvil.flow): the coupling flow is a bijection with the right
log-determinant; a target in flow coordinates samples the original
posterior exactly and is recorded in the original coordinates; a fit never
comes back worse than whitening; the report tells a curved posterior from a
Gaussian one; flow importance sampling recovers a closed-form ln Z; a resume
refuses another flow's coordinates; and the EnsembleSampler switch.

The test posterior is a banana: x ~ N(1, 1), y ~ N(0, 10^2) a priori, and
y | x ~ N(x^2, 0.1^2) as the likelihood. Its evidence is a 1-D integral and
its posterior can be drawn exactly.
"""

import math
import warnings

import mlx.core as mx
import numpy as np
import pytest

import anvil
from anvil import ParamSpec, Transform, TransformedLogDensity
from anvil.flow import Flow, FlowTarget, refine, train
from anvil.kernels.ensemble import DEMove, EnsembleKernel, StretchMove
from anvil.priors import Gaussian

SY = 0.1
SMALL = dict(n_layers=4, hidden=32)


def _banana():
    specs = [ParamSpec("x", prior=Gaussian(1.0, 1.0)),
             ParamSpec("y", prior=Gaussian(0.0, 10.0))]

    def ll(v):
        return -0.5 * ((v[:, 1] - v[:, 0] ** 2) / SY) ** 2

    const = -math.log(SY) - 0.5 * math.log(2 * math.pi)
    return TransformedLogDensity(ll, Transform(specs), model_log_norm=const)


def _banana_log_z():
    # Z = int N(x; 1, 1) N(x^2; 0, 10^2 + SY^2) dx
    x = np.linspace(-9.0, 11.0, 400_001)
    s2 = 100.0 + SY ** 2
    f = (np.exp(-0.5 * (x - 1) ** 2) / math.sqrt(2 * math.pi)
         * np.exp(-0.5 * x ** 4 / s2) / math.sqrt(2 * math.pi * s2))
    return math.log(np.trapezoid(f, x))


def _banana_draws(n, seed=0):
    rng = np.random.default_rng(seed)
    x = 1.0 + rng.standard_normal(8 * n)
    w = np.exp(-0.5 * x ** 4 / (100.0 + SY ** 2))
    x = x[rng.choice(len(x), n, p=w / w.sum())]
    var = 1.0 / (1.0 / 100.0 + 1.0 / SY ** 2)
    y = var * x ** 2 / SY ** 2 + math.sqrt(var) * rng.standard_normal(n)
    return np.stack([x, y], axis=1).astype(np.float32)


@pytest.fixture(scope="module")
def banana_flow():
    """A flow fitted to exact banana draws (u = model values here)."""
    f = Flow(2, **SMALL, seed=1)
    info = f.fit(_banana_draws(40_000), max_steps=2000, seed=1)
    return f, info


def _perturbed_flow(dim=3, seed=0):
    f = Flow(dim, n_layers=4, hidden=16, seed=seed)
    rng = np.random.default_rng(seed)
    for p in f.params:
        p["W3"] = mx.array(0.3 * rng.standard_normal(p["W3"].shape).astype(np.float32))
        p["b3"] = mx.array(0.3 * rng.standard_normal(p["b3"].shape).astype(np.float32))
    f.mu = mx.array([0.5, -1.0, 2.0][:dim])
    return f


# -- the flow itself ------------------------------------------------------------

def test_flow_is_a_bijection_with_the_right_log_determinant():
    f = _perturbed_flow()
    z = mx.random.normal((64, 3), key=mx.random.key(0))
    u, ld = f.forward(z)
    z2, ld_inv = f.inverse(u)
    np.testing.assert_allclose(np.array(z2), np.array(z), atol=2e-5)
    np.testing.assert_allclose(np.array(ld_inv), -np.array(ld), atol=2e-5)
    # log|det J| against the Jacobian itself, one point at a time
    for i in range(4):
        J = np.stack([np.array(mx.vjp(lambda a: f.forward(a)[0], [z[i:i + 1]],
                                      [mx.array(np.eye(3, dtype=np.float32)[k:k + 1])])[1][0])[0]
                      for k in range(3)])
        assert abs(np.linalg.slogdet(J.astype(np.float64))[1] - float(ld[i].item())) < 1e-4


def test_sample_and_log_q_agree():
    f = _perturbed_flow()
    u, logq = f.sample(256, seed=3)
    np.testing.assert_allclose(np.array(f.log_q(u)), np.array(logq), atol=1e-4)


def test_untrained_flow_is_the_identity_and_dim_one_is_refused():
    f = Flow(4)
    z = mx.random.normal((8, 4), key=mx.random.key(1))
    u, ld = f.forward(z)
    np.testing.assert_allclose(np.array(u), np.array(z), atol=1e-6)
    np.testing.assert_allclose(np.array(ld), 0.0, atol=1e-6)
    with pytest.raises(ValueError, match="dim >= 2"):
        Flow(1)


def test_save_load_round_trip(tmp_path):
    f = _perturbed_flow()
    f.save(tmp_path / "f.npz")
    g = Flow.load(tmp_path / "f.npz")
    assert g.fingerprint == f.fingerprint
    z = mx.random.normal((16, 3), key=mx.random.key(2))
    np.testing.assert_array_equal(np.array(g.forward(z)[0]), np.array(f.forward(z)[0]))
    assert _perturbed_flow(seed=1).fingerprint != f.fingerprint


# -- fitting --------------------------------------------------------------------

def test_fit_learns_the_banana(banana_flow):
    f, info = banana_flow
    # a curved posterior: the flow gains several nats over whitening
    assert info["gain_nats"] > 1.0
    assert not info["diverged"]


def test_fit_never_returns_worse_than_whitening():
    # a learning rate this large diverges at every retry
    f = Flow(2, **SMALL, seed=0)
    info = f.fit(_banana_draws(5000), lr=50.0, max_steps=300, retries=1, seed=0)
    assert info["gain_nats"] >= 0.0
    assert info["retries"] == 1 and info["diverged"]
    u = mx.array(_banana_draws(100, seed=5))
    assert np.all(np.isfinite(np.array(f.log_q(u))))


# -- sampling in flow coordinates -------------------------------------------------

def test_flow_target_samples_the_posterior_and_records_in_u(banana_flow):
    f, _ = banana_flow
    t = _banana()
    ft = f.wrap(t)
    kern = EnsembleKernel(ft, moves=[(StretchMove(), 0.5), (DEMove(), 0.5)])
    rng = np.random.default_rng(0)
    p0 = f.to_latent(_banana_draws(256, seed=9)[rng.permutation(256)])
    res = anvil.run(kern, ft, p0, n_warmup=200, n_samples=600, thin=2, seed=0)
    chain = res.get_chain()
    # recorded log-probabilities are the target's own at the recorded draws
    lp = np.array(t.log_prob(mx.array(chain[-1])))
    np.testing.assert_allclose(res.get_log_prob()[-1], lp, rtol=1e-4, atol=1e-3)
    # and the draws are the banana's, to Monte Carlo accuracy
    ref = _banana_draws(200_000, seed=4)
    flat = chain.reshape(-1, 2)
    for q in (5, 50, 95):
        d = np.abs(np.percentile(flat, q, axis=0) - np.percentile(ref, q, axis=0))
        assert np.all(d < 0.06 * ref.std(axis=0)), (q, d)
    assert anvil.diagnose(chain).rhat.max() < 1.02
    assert res.target_tag == ft.resume_tag


def test_resume_refuses_another_flows_coordinates(banana_flow, tmp_path):
    f, _ = banana_flow
    t = _banana()
    ft = f.wrap(t)
    res = anvil.run(EnsembleKernel(ft), ft, f.start(64), n_warmup=10, n_samples=10)
    # the same flow (also after a save/load) resumes
    f.save(tmp_path / "f.npz")
    same = Flow.load(tmp_path / "f.npz").wrap(t)
    anvil.run(EnsembleKernel(same), same, resume=res, n_samples=5)
    res.save_state(tmp_path / "s.npz")
    assert anvil.load_state(tmp_path / "s.npz").target_tag == ft.resume_tag
    # another flow, or the target itself, does not
    other = _perturbed_flow(dim=2).wrap(t)
    with pytest.raises(ValueError, match="other coordinates"):
        anvil.run(EnsembleKernel(other), other, resume=res, n_samples=5)
    with pytest.raises(ValueError, match="other coordinates"):
        anvil.run(EnsembleKernel(t), t, resume=anvil.load_state(tmp_path / "s.npz"),
                  n_samples=5)


# -- training and the report ------------------------------------------------------

def test_train_from_the_prior_reports_a_curved_posterior():
    t = _banana()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        f = train(t, rounds=1, n_chains=128, round_warmup=50, round_samples=250,
                  smc_kwargs=dict(n_particles=2048), fit_kwargs=dict(max_steps=1500),
                  **SMALL)
    r = f.report
    assert r.smc is not None and r.worthwhile and r.gain_nats > 1.0
    assert r.is_efficiency > 0.1
    assert f.pool.shape[1] == 2
    assert "fit gain" in str(r)


def test_report_says_no_flow_needed_for_a_gaussian():
    # strongly correlated but Gaussian: whitening alone removes it
    A = np.array([[1.0, 0.0], [0.9, 0.3]])
    Ainv_T = mx.array(np.linalg.inv(A).T.astype(np.float32))
    specs = [ParamSpec("a", prior=Gaussian(0.0, 50.0)), ParamSpec("b", prior=Gaussian(0.0, 50.0))]
    t = TransformedLogDensity(lambda v: -0.5 * mx.sum((v @ Ainv_T) ** 2, axis=1),
                              Transform(specs))
    draws = (np.random.default_rng(0).standard_normal((20_000, 2)) @ A.T).astype(np.float32)
    with pytest.warns(RuntimeWarning, match="no flow needed"):
        f = refine(t, draws, fit_kwargs=dict(max_steps=1000), **SMALL)
    assert not f.report.worthwhile and f.report.gain_nats < 1.0


def test_train_needs_a_proper_prior_or_draws():
    t = anvil.FunctionLogDensity(lambda u: -0.5 * mx.sum(u * u, axis=1), 2)
    with pytest.raises(ValueError, match="proper priors"):
        train(t)


# -- evidence ---------------------------------------------------------------------

def test_flow_evidence_recovers_a_closed_form_log_z(banana_flow):
    f, _ = banana_flow
    ev = anvil.flow_evidence(_banana(), f, n=65536, seed=0)
    assert ev.method == "flow importance sampling"
    assert ev.details["efficiency"] > 0.3
    assert abs(ev.log_z - _banana_log_z()) < max(5 * ev.log_z_se, 0.01)


# -- the EnsembleSampler switch -----------------------------------------------------

def test_ensemble_sampler_with_a_flow(banana_flow):
    f, _ = banana_flow
    s = anvil.EnsembleSampler(128, 2, target=_banana(), flow=f, seed=0)
    s.run_mcmc(None, 300, warmup=100)
    chain = s.get_chain()
    assert chain.shape == (300, 128, 2)
    # in u, not latent: the banana's y sits near x^2 + ~1
    assert abs(np.median(chain[..., 1]) - np.median(_banana_draws(50_000)[:, 1])) < 0.3
    # initial_state in u is mapped in
    s.run_mcmc(_banana_draws(128, seed=3), 10)


def test_ensemble_sampler_flow_arguments_are_checked(banana_flow):
    f, _ = banana_flow
    fn = lambda u: -0.5 * mx.sum(u * u, axis=1)          # noqa: E731
    with pytest.raises(ValueError, match="proper priors"):
        anvil.EnsembleSampler(64, 2, fn, flow=True)
    with pytest.raises(ValueError, match="kernel="):
        anvil.EnsembleSampler(64, 2, target=_banana(), flow=f,
                              kernel=EnsembleKernel(_banana()))
    with pytest.raises(TypeError, match="initial_state"):
        anvil.EnsembleSampler(64, 2, fn).run_mcmc(None, 10)
    # plain use is unchanged
    s = anvil.EnsembleSampler(64, 2, fn)
    s.run_mcmc(np.random.default_rng(0).standard_normal((64, 2)), 20)
    assert s.flow is None and s.get_chain().shape == (20, 64, 2)
    with pytest.raises(TypeError, match="no flow mode"):
        anvil.HMCSampler(64, 2, fn, flow=True)
