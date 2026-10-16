"""Explicit, normalised priors.

Three properties carry the feature and are tested directly rather than
inferred: every prior integrates to one with its float64 constant; a prior
composes correctly with each kind of ParamSpec map through the samplers (a
flat likelihood must return the prior); and declaring no prior leaves draws
bit-identical to anvil 0.7.1, which every downstream package relies on.
"""

import math
import pathlib
from dataclasses import replace

import mlx.core as mx
import numpy as np
import pytest

import anvil
from anvil import ParamSpec, Transform, TransformedLogDensity, run
from anvil.kernels.chees import ChEESHMC
from anvil.kernels.ensemble import EnsembleKernel
from anvil.priors import (Gaussian, ImproperFlat, LogUniform,
                          TruncatedGaussian, Uniform)

_ROOT = pathlib.Path(__file__).resolve().parent.parent
_trap = getattr(np, "trapezoid", None) or np.trapz

#: (label, prior bound to its box, quadrature range)
BOUND = [
    ("uniform", lambda: Uniform().bind(-2.0, 5.0, "x"), (-2.0, 5.0)),
    ("loguniform", lambda: LogUniform().bind(0.1, 10.0, "x"), (0.1, 10.0)),
    ("gaussian", lambda: Gaussian(0.3, 0.7).bind(-math.inf, math.inf, "x"),
     (0.3 - 12 * 0.7, 0.3 + 12 * 0.7)),
    ("trunc_box", lambda: Gaussian(0.5, 1.0).bind(-1.0, 1.0, "x"), (-1.0, 1.0)),
    ("trunc_half", lambda: Gaussian(0.0, 1.0).bind(0.0, math.inf, "x"), (0.0, 40.0)),
    ("trunc_tail", lambda: Gaussian(0.0, 1.0).bind(4.0, math.inf, "x"), (4.0, 40.0)),
]


# -- each prior on its own --------------------------------------------------

@pytest.mark.parametrize("label, make, rng_", BOUND, ids=[b[0] for b in BOUND])
def test_every_prior_integrates_to_one(label, make, rng_):
    p = make()
    v = np.linspace(*rng_, 2_000_001)
    assert abs(_trap(np.exp(p.log_density_np(v)), v) - 1.0) < 1e-6


@pytest.mark.parametrize("label, make, rng_", BOUND, ids=[b[0] for b in BOUND])
def test_graph_term_matches_its_float64_twin(label, make, rng_):
    p = make()
    v = np.linspace(rng_[0] + 1e-3, min(rng_[1], rng_[0] + 20) - 1e-3, 1001)
    np.testing.assert_allclose(np.array(p.log_pdf(mx.array(v.astype(np.float32)))),
                               p.log_pdf_np(v), rtol=1e-4, atol=1e-4)
    with mx.stream(mx.cpu):            # the dtype polymorphism log_prob_hi relies on
        lp64 = np.array(p.log_pdf(mx.array(v, dtype=mx.float64)))
    np.testing.assert_allclose(lp64, p.log_pdf_np(v), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("label, make, rng_", BOUND, ids=[b[0] for b in BOUND])
def test_draws_have_the_analytic_moments_and_stay_in_the_box(label, make, rng_):
    p = make()
    n = 200_000
    x = p.sample(n, np.random.default_rng(1))
    assert np.all((x >= p.lo) & (x <= p.hi))
    assert abs(x.mean() - p.mean) < 6 * p.sd / math.sqrt(n)
    assert abs(x.std() / p.sd - 1.0) < 0.02


def test_flat_priors_add_nothing_to_the_graph():
    assert not Uniform().bind(0, 1, "x").has_in_graph_term
    assert LogUniform().bind(0.1, 1, "x").has_in_graph_term
    assert Gaussian(0, 1).has_in_graph_term


# -- resolution against the ParamSpec box ------------------------------------

def test_defaults_are_uniform_on_a_box_and_improper_otherwise():
    t = Transform([ParamSpec("a", 0.0, 2.0), ParamSpec("b", lo=0.0),
                   ParamSpec("c")])
    assert isinstance(t.priors[0], Uniform) and t.priors[0].is_proper
    assert isinstance(t.priors[1], ImproperFlat)
    assert isinstance(t.priors[2], ImproperFlat)
    assert not t.is_proper
    with pytest.raises(ValueError, match=r"improper prior on \['b', 'c'\]"):
        t.sample_prior(4, np.random.default_rng(0))


def test_a_gaussian_on_a_box_is_truncated_with_the_right_constant():
    g = Gaussian(0.5, 1.0)
    tg = ParamSpec("x", -1.0, 1.0, prior=g)
    p = Transform([tg]).priors[0]
    assert isinstance(p, TruncatedGaussian)
    Z = 0.5 * (math.erf(0.5 / math.sqrt(2)) - math.erf(-1.5 / math.sqrt(2)))
    assert abs(p.mass_inside - Z) < 1e-14
    assert abs(p.log_norm - (g.log_norm - math.log(Z))) < 1e-14
    # and on an unbounded spec it stays a plain Gaussian
    assert type(Transform([ParamSpec("y", prior=g)]).priors[0]) is Gaussian


def test_log_prior_norm_is_the_sum_of_the_constants():
    t = Transform([ParamSpec("a", -2.0, 5.0), ParamSpec("b", 0.1, 10.0, prior=LogUniform()),
                   ParamSpec("c", prior=Gaussian(0.3, 0.7))])
    assert t.is_proper
    assert abs(t.log_prior_norm - sum(p.log_norm for p in t.priors)) < 1e-15


@pytest.mark.parametrize("make, match", [
    (lambda: ParamSpec("x", 0.0, 1.0, prior=LogUniform()), "lo > 0"),
    (lambda: ParamSpec("x", lo=1.0, prior=LogUniform()), "finite hi"),
    (lambda: ParamSpec("x", 1.0, 2.0, report_offset=1.0, prior=LogUniform()), "offset"),
    (lambda: ParamSpec("x", prior=Uniform()), "finite box"),
    (lambda: ParamSpec("x", 0.0, 2.0, prior=Uniform(0.0, 1.0)), "box is the prior's support"),
    (lambda: ParamSpec("x", 0.25, 0.75, prior=Uniform(0.0, 1.0)), "box is the prior's support"),
    (lambda: Gaussian(0.0, -1.0), "sigma > 0"),
    (lambda: ParamSpec("x", 50.0, 60.0, prior=Gaussian(0.0, 1.0)), "no measurable mass"),
])
def test_construction_errors_are_specific(make, match):
    with pytest.raises(ValueError, match=match):
        make()


def test_from_physical_converts_through_the_reporting_map():
    spec = ParamSpec("P", report_offset=3.45, report_scale=1.0)
    g = Gaussian.from_physical(3.456, 1e-4, spec)
    assert abs(g.mu - 0.006) < 1e-12 and g.sigma == 1e-4


# -- composition with the Transform -----------------------------------------

def _mixed():
    """One parameter per prior x box type."""
    return Transform([
        ParamSpec("a", -2.0, 5.0),                                 # Uniform, bounded
        ParamSpec("b", 0.1, 10.0, prior=LogUniform()),             # LogUniform, bounded
        ParamSpec("c", prior=Gaussian(0.3, 0.7)),                  # Gaussian, free
        ParamSpec("d", -1.0, 1.0, prior=Gaussian(0.5, 1.0)),       # truncated, bounded
        ParamSpec("e", lo=0.0, prior=Gaussian(0.0, 1.0)),          # truncated, half
    ])


def test_transform_prior_graph_matches_its_twin_and_draws_are_usable():
    t = _mixed()
    assert t.is_proper and t.has_prior_terms
    rng = np.random.default_rng(3)
    v = t.sample_prior_model(2000, rng)
    np.testing.assert_allclose(np.array(t.log_prior(mx.array(v.astype(np.float32)))),
                               t.log_prior_np(v), rtol=1e-4, atol=1e-3)
    u = t.sample_prior(2000, rng)
    assert u.dtype == mx.float32 and u.shape == (2000, 5)
    assert np.all(np.isfinite(np.array(u)))
    back = t.model_np(np.array(u, dtype=np.float64))
    lo = np.array([s.lo for s in t.specs]); hi = np.array([s.hi for s in t.specs])
    assert np.all((back >= lo) & (back <= hi))


def test_prior_transform_is_the_hypercube_interface():
    t = _mixed()
    cube = np.random.default_rng(4).uniform(size=(100_000, 5))
    v = t.prior_transform(cube)
    np.testing.assert_allclose(v.mean(0), t.prior_mean, atol=0.05 * t.prior_sd.max())
    np.testing.assert_allclose(v.std(0), t.prior_sd, rtol=0.02)


def test_gradient_through_a_prior_matches_finite_differences():
    t = _mixed()
    tgt = TransformedLogDensity(lambda v: mx.zeros(v.shape[:1]), t)
    u = np.array([[0.3, -0.4, 0.2, 0.5, -0.3]])
    _, g = tgt.log_prob_and_grad(mx.array(u.astype(np.float32)))
    f = lambda uu: float(t.log_prior_np(t.model_np(uu))[0] + t.log_det_jac_np(uu)[0])
    h = 1e-6
    fd = np.array([(f(u + h * e) - f(u - h * e)) / (2 * h) for e in np.eye(5)])
    np.testing.assert_allclose(np.array(g)[0], fd, rtol=1e-3, atol=1e-4)


# -- composition with a real target ------------------------------------------

def _transit_with(prior_on_p_off=None):
    from anvil.targets import make_transit_target
    tt = make_transit_target(n_data=20_000, seed=0)
    specs = [replace(s, prior=prior_on_p_off) if s.name == "p_off" and prior_on_p_off else s
             for s in tt.transform.specs]
    t = Transform(specs)
    lik = tt.loglike
    return tt, t, TransformedLogDensity(lik, t, model_log_prob_hi=lik.hi,
                                        model_log_norm=lik.log_offset_const + lik.log_norm_const)


def test_the_precision_harness_still_agrees_with_a_prior():
    tt0, _, plain = _transit_with(None)
    p_off = next(s for s in tt0.transform.specs if s.name == "p_off")
    _, _, withp = _transit_with(Gaussian(0.5 * (p_off.lo + p_off.hi), 0.1 * (p_off.hi - p_off.lo)))
    u = mx.array(np.random.default_rng(5).standard_normal((32, plain.dim)).astype(np.float32) * 0.01)
    e0 = np.abs(np.array(plain.log_prob(u)) - np.array(plain.log_prob_hi(u))).max()
    e1 = np.abs(np.array(withp.log_prob(u)) - np.array(withp.log_prob_hi(u))).max()
    assert e1 <= e0 + 1e-3, (e0, e1)


def test_absolute_density_is_reconstructed_in_float64():
    tt, t, tgt = _transit_with(None)
    u = np.random.default_rng(6).standard_normal((8, tgt.dim)) * 0.01
    stored = np.array(tgt.log_prob(mx.array(u.astype(np.float32))), dtype=np.float64)
    v = t.model_np(u)
    with mx.stream(mx.cpu):
        ll_abs = (np.array(tt.loglike.hi(mx.array(v, dtype=mx.float64)))
                  + tt.loglike.log_offset_const + tt.loglike.log_norm_const)
    want = ll_abs + t.log_prior_np(v) + t.log_prior_norm + t.log_det_jac_np(u)
    # the constants must be exact: check them on the float64 path
    hi = np.array(tgt.log_prob_hi(mx.array(u.astype(np.float32))), dtype=np.float64)
    u32 = u.astype(np.float32).astype(np.float64)          # the same input point
    v32 = t.model_np(u32)
    with mx.stream(mx.cpu):
        ll32 = (np.array(tt.loglike.hi(mx.array(v32, dtype=mx.float64)))
                + tt.loglike.log_offset_const + tt.loglike.log_norm_const)
    want32 = ll32 + t.log_prior_np(v32) + t.log_prior_norm + t.log_det_jac_np(u32)
    np.testing.assert_allclose(hi + tgt.log_norm_const, want32, rtol=1e-12)
    # and the float32 path agrees to float32 precision -- these random points
    # sit far from the posterior where |log L| ~ 1e6 and an ulp is 0.06
    np.testing.assert_allclose(stored + tgt.log_norm_const, want, rtol=1e-6)


# -- bit-identity: the guarantee every downstream package relies on ----------

def test_no_prior_leaves_the_graph_untouched():
    from anvil.targets import make_transit_target
    tt = make_transit_target(n_data=20_000, seed=0)
    assert not tt.transform.has_prior_terms and tt.transform.is_proper
    u = mx.array(np.random.default_rng(2).standard_normal((64, 6)).astype(np.float32) * 0.01)
    v = tt.transform.to_model(u)
    np.testing.assert_array_equal(np.array(tt.target.log_prob(u)),
                                  np.array(tt.loglike(v) + tt.transform.log_det_jac(u)))
    explicit = Transform([replace(s, prior=Uniform()) for s in tt.transform.specs])
    t2 = TransformedLogDensity(tt.loglike, explicit, model_log_prob_hi=tt.loglike.hi)
    a = run(ChEESHMC(tt.target), tt.target, u, n_warmup=40, n_samples=30, seed=7)
    b = run(ChEESHMC(t2), t2, u, n_warmup=40, n_samples=30, seed=7)
    np.testing.assert_array_equal(a.get_chain(), b.get_chain())
    np.testing.assert_array_equal(a.get_log_prob(), b.get_log_prob())


def test_draws_are_bit_identical_to_the_0_7_1_reference():
    ref = np.load(_ROOT / "tests" / "data" / "transit_chees_ref.npz")
    if str(ref["mlx_version"]) != mx.__version__:
        pytest.skip(f"reference made with mlx {ref['mlx_version']}, running {mx.__version__}")
    from anvil.targets import make_transit_target
    tt = make_transit_target(n_data=20_000, seed=0)
    u0 = mx.array(np.random.default_rng(2).standard_normal((64, tt.target.dim)).astype(np.float32) * 0.01)
    res = run(ChEESHMC(tt.target), tt.target, u0, n_warmup=40, n_samples=30, seed=7)
    np.testing.assert_array_equal(res.get_chain(), ref["chain"])
    np.testing.assert_array_equal(res.get_log_prob(), ref["log_prob"])


# -- end to end: a flat likelihood must return the prior ---------------------

@pytest.mark.slow
@pytest.mark.parametrize("kind", ["chees", "ensemble"])
def test_samplers_recover_every_prior_through_every_map(kind):
    """The check that the prior term and each ParamSpec map's Jacobian
    compose correctly in u-space: with no data the posterior IS the prior."""
    t = _mixed()
    tgt = TransformedLogDensity(lambda v: mx.zeros(v.shape[:1]), t)
    u0 = t.sample_prior(512, np.random.default_rng(7))
    k = ChEESHMC(tgt) if kind == "chees" else EnsembleKernel(tgt, seed=3)
    res = run(k, tgt, u0, n_warmup=1000, n_samples=500 if kind == "chees" else 2000, seed=8)
    v = t.model_np(res.get_chain(flat=True).astype(np.float64))
    np.testing.assert_allclose(v.mean(0), t.prior_mean, atol=0.05 * t.prior_sd.max())
    np.testing.assert_allclose(v.std(0), t.prior_sd, rtol=0.10)
