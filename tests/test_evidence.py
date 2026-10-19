"""Optional evidence calculations: tempered SMC, bridge sampling, Laplace.

Each method is checked against a closed-form ln Z. The targets cover four
kinds of prior, a posterior with two narrow mirror modes and a broad ridge
holding most of the mass, and a likelihood that is -inf over half its box.
SMC's group error bar is checked for calibration across seeds, not only
for size, and its stall count for firing when particles stop moving.
"""

import math

import mlx.core as mx
import numpy as np
import pytest

import anvil
from anvil import (FunctionLogDensity, ParamSpec, Transform,
                   TransformedLogDensity)
from anvil.evidence import (TemperedLogDensity, bridge_evidence,
                            laplace_evidence, smc)
from anvil.priors import Gaussian, LogUniform


def _Phi(x):
    return 0.5 * math.erfc(-x / math.sqrt(2))


def _separable(s=0.01):
    """Normalised Gaussian likelihood per dimension, under a Uniform box, an
    unbounded Gaussian, a LogUniform and a box-truncated Gaussian prior."""
    specs = [ParamSpec("a", -5.0, 5.0),
             ParamSpec("b", prior=Gaussian(0.3, 1.0)),
             ParamSpec("c", 0.1, 10.0, prior=LogUniform()),
             ParamSpec("d", 0.0, 2.0, prior=Gaussian(0.5, 0.5))]
    m = np.array([1.2, 0.5, 2.0, 0.6])
    sd = np.array([s, s, 3 * s, s])
    m32, isd = mx.array(m.astype(np.float32)), mx.array((1 / sd).astype(np.float32))

    def ll(v):
        z = (v - m32) * isd
        return -0.5 * mx.sum(z * z, axis=-1)

    const = float(-np.log(sd).sum() - 2 * math.log(2 * math.pi))
    t = TransformedLogDensity(ll, Transform(specs), model_log_norm=const)
    lz = math.log((_Phi((5 - m[0]) / sd[0]) - _Phi((-5 - m[0]) / sd[0])) / 10.0)
    s2 = sd[1] ** 2 + 1.0
    lz += -0.5 * math.log(2 * math.pi * s2) - 0.5 * (m[1] - 0.3) ** 2 / s2
    x = np.linspace(m[2] - 12 * sd[2], m[2] + 12 * sd[2], 200_001)
    f = np.exp(-0.5 * ((x - m[2]) / sd[2]) ** 2) / (sd[2] * math.sqrt(2 * math.pi)) \
        / (x * math.log(100.0))
    lz += math.log(np.sum(0.5 * (f[1:] + f[:-1]) * np.diff(x)))
    s2 = sd[3] ** 2 + 0.25
    lz += (-0.5 * math.log(2 * math.pi * s2) - 0.5 * (m[3] - 0.5) ** 2 / s2
           - math.log(_Phi(3.0) - _Phi(-1.0)))
    return t, lz


def _mirror_ridge(w=(0.08, 0.08, 0.84), extra=4):
    """Box prior; two narrow mirror modes and a curved ridge through the
    centre carrying most of the mass. Z = sum(w) / box volume, exactly."""
    d = 2 + extra
    specs = [ParamSpec(f"x{i}", -1.0, 1.0) for i in range(d)]
    lw = [math.log(x) for x in w]

    def logn(z, s):
        return -0.5 * (z / s) ** 2 - math.log(s * math.sqrt(2 * math.pi))

    def ll(v):
        x, y, e = v[:, 0], v[:, 1], v[:, 2:]
        a = lw[0] + logn(x - 0.5, 0.01) + logn(y - 0.3, 0.01)
        b = lw[1] + logn(x + 0.5, 0.01) + logn(y - 0.3, 0.01)
        c = lw[2] + logn(x, 0.15) + logn(y - (4.0 * x * x - 0.4), 0.01)
        return mx.logaddexp(mx.logaddexp(a, b), c) + mx.sum(logn(e, 0.05), axis=-1)

    return TransformedLogDensity(ll, Transform(specs)), math.log(sum(w)) - d * math.log(2.0)


def _conjugate(d=3):
    """Unbounded parameters (u = v), Gaussian priors N(0, P^2), normalised
    Gaussian likelihood N(v; m, S^2): Gaussian posterior, Z in closed form."""
    P = np.array([1.0, 2.0, 0.5])[:d]
    S = np.array([0.1, 0.3, 0.05])[:d]
    m = np.array([0.4, -1.0, 0.2])[:d]
    specs = [ParamSpec(f"x{i}", prior=Gaussian(0.0, float(P[i]))) for i in range(d)]
    m32, iS = mx.array(m.astype(np.float32)), mx.array((1 / S).astype(np.float32))

    def ll(v):
        z = (v - m32) * iS
        return -0.5 * mx.sum(z * z, axis=-1)

    const = float(-np.log(S).sum() - 0.5 * d * math.log(2 * math.pi))
    t = TransformedLogDensity(ll, Transform(specs), model_log_norm=const)
    v = P ** 2 + S ** 2
    lz = float(np.sum(-0.5 * np.log(2 * math.pi * v) - 0.5 * m ** 2 / v))
    post_mean = m * P ** 2 / v
    post_sd = np.sqrt(P ** 2 * S ** 2 / v)
    return t, lz, post_mean, post_sd


# -- the tempered density ------------------------------------------------------

def test_tempered_endpoints():
    t, _ = _separable()
    u = mx.array(np.random.default_rng(0).standard_normal((16, 4)).astype(np.float32))
    v = t.transform.to_model(u)
    one = np.array(TemperedLogDensity(t, 1.0).log_prob(u))
    np.testing.assert_allclose(one, np.array(t.log_prob(u)), rtol=1e-6, atol=1e-3)
    zero = np.array(TemperedLogDensity(t, 0.0).log_prob(u))
    prior_u = np.array(t.transform.log_det_jac(u) + t.transform.log_prior(v))
    np.testing.assert_allclose(zero, prior_u, rtol=1e-6, atol=1e-5)


# -- SMC -----------------------------------------------------------------------

@pytest.mark.parametrize("make", [_separable, _mirror_ridge], ids=["separable", "mirror_ridge"])
def test_smc_recovers_analytic_evidence(make):
    t, lz = make()
    r = smc(t, seed=0)
    assert abs(r.log_z - lz) < 0.1
    assert r.u.shape == (8192, t.dim)
    assert len(r.log_z_groups) == 1 and math.isnan(r.log_z_se)


def test_smc_weights_modes_it_cannot_cross():
    t, _ = _mirror_ridge()
    r = smc(t, seed=1)
    v = r.model
    left = np.mean((np.abs(v[:, 0] + 0.5) < 0.05) & (np.abs(v[:, 1] - 0.3) < 0.05))
    right = np.mean((np.abs(v[:, 0] - 0.5) < 0.05) & (np.abs(v[:, 1] - 0.3) < 0.05))
    assert left == pytest.approx(0.08, abs=0.03)
    assert right == pytest.approx(0.08, abs=0.03)


def test_smc_group_error_bar_is_calibrated():
    t, lz = _separable()
    z = np.array([(r.log_z - lz) / r.log_z_se
                  for r in (smc(t, 16384, n_groups=4, seed=100 + s) for s in range(20))])
    # err/se is t-distributed with n_groups - 1 = 3 dof: P(|t| < 2) = 0.86,
    # median |t| = 0.77. Its mean square has no finite variance, so it is
    # not tested.
    assert np.mean(np.abs(z) < 2) >= 0.7
    assert 0.3 < np.median(np.abs(z)) < 1.6


def test_smc_adds_only_the_likelihood_constant():
    t, lz = _separable()
    shifted = TransformedLogDensity(t._fn, t.transform,
                                    model_log_norm=t.model_log_norm + 5.0)
    assert smc(shifted, seed=3).log_z == pytest.approx(smc(t, seed=3).log_z + 5.0, abs=1e-6)


def test_nested_identity_parameter_the_data_ignore():
    t, lz = _separable()
    specs = t.transform.specs + [ParamSpec("unused", -3.0, 7.0)]
    t2 = TransformedLogDensity(lambda v: t._fn(v[:, :4]), Transform(specs),
                               model_log_norm=t.model_log_norm)
    a, b = smc(t, seed=4), smc(t2, seed=5)
    assert abs(a.log_z - b.log_z) < 0.1


def test_smc_infeasible_half_of_the_box():
    # L = 0 for v < 0.5: the evidence is the mass of N(0.7, 0.05) above 0.5
    t = TransformedLogDensity(
        lambda v: mx.where(v[:, 0] > 0.5, -0.5 * ((v[:, 0] - 0.7) / 0.05) ** 2, -mx.inf),
        Transform([ParamSpec("x", 0.0, 1.0)]),
        model_log_norm=-math.log(0.05 * math.sqrt(2 * math.pi)))
    lz = math.log(_Phi(0.3 / 0.05) - _Phi(-0.2 / 0.05))
    r = smc(t, n_particles=2048, seed=0)
    assert abs(r.log_z - lz) < 0.05


def test_smc_warns_when_groups_disagree():
    t, _ = _mirror_ridge()
    # groups of 16 particles cannot hold three modes between them
    with pytest.warns(RuntimeWarning, match="groups disagree"):
        smc(t, n_particles=128, n_groups=8, seed=0)


def _banana(w):
    """Box prior; a parabolic ridge y = 2x^2 - 0.5 of width w, plus four
    narrow Gaussian dimensions. Z in closed form (box truncation < 0.5%)."""
    specs = [ParamSpec(f"x{i}", -1.0, 1.0) for i in range(6)]

    def ll(v):
        x, y, e = v[:, 0], v[:, 1], v[:, 2:]
        return (-0.5 * (x / 0.3) ** 2 - 0.5 * ((y - (2.0 * x * x - 0.5)) / w) ** 2
                - 0.5 * mx.sum((e / 0.05) ** 2, axis=-1))

    r2p = math.sqrt(2 * math.pi)
    lz = math.log(0.3 * r2p * w * r2p * (0.05 * r2p) ** 4) - 6 * math.log(2.0)
    return TransformedLogDensity(ll, Transform(specs)), lz


def test_smc_warns_when_particles_stop_moving():
    # a ridge 0.0005 wide: late in the run under 5% of moves are accepted and
    # the particles stay where resampling left them -- the failure that
    # biases ln Z low (here by about 0.16 nats) -- and smc says so
    t, _ = _banana(0.0005)
    with pytest.warns(RuntimeWarning, match="stopped exploring"):
        r = smc(t, 2048, seed=0)
    assert r.stalled > 0 and "decorrelating" in str(r)


@pytest.mark.parametrize("make", [lambda: _banana(0.002), _mirror_ridge],
                         ids=["wider_ridge", "mirror_ridge"])
def test_smc_no_stall_where_particles_move(make):
    # the mirror modes never exchange particles, so their coordinates stay
    # correlated; acceptance stays high, and that is not a stall
    t, lz = make()
    r = smc(t, 2048, seed=0)
    assert r.stalled == 0 and abs(r.log_z - lz) < 0.1


def test_smc_needs_a_proper_prior():
    with pytest.raises(ValueError, match="TransformedLogDensity"):
        smc(FunctionLogDensity(lambda u: -0.5 * mx.sum(u * u, axis=-1), 2))
    free = TransformedLogDensity(lambda v: -0.5 * mx.sum(v * v, axis=-1),
                                 Transform([ParamSpec("a"), ParamSpec("b")]))
    with pytest.raises(ValueError, match="improper"):
        smc(free)


def test_smc_reports_float32_error_when_it_can():
    t, _, _, _ = _conjugate()
    hi = TransformedLogDensity(t._fn, t.transform, model_log_norm=t.model_log_norm,
                               model_log_prob_hi=t._fn)
    assert smc(t, seed=0).precision is None
    p = smc(hi, seed=0, n_probe=16).precision
    assert p["n"] == 16 and abs(p["mean"]) < 1e-3


# -- bridge sampling -------------------------------------------------------------

def test_bridge_exact_on_exact_draws():
    t, lz, mu, sd = _conjugate()
    draws = mu + sd * np.random.default_rng(0).standard_normal((20_000, 3))
    b = bridge_evidence(t, draws)
    assert abs(b.log_z - lz) < max(3 * b.log_z_se, 1e-3)
    assert abs(b.log_z - lz) < 0.01
    assert b.details["proposal_ess"] > 0.5 * b.details["n_proposal"]


def test_bridge_from_smc_and_from_a_chees_run():
    t, lz, mu, sd = _conjugate()
    b = bridge_evidence(t, smc(t, seed=2))
    assert abs(b.log_z - lz) < 0.05
    res = anvil.run(anvil.ChEESHMC(t), t,
                    mx.array((mu + 0.1 * sd * np.random.default_rng(1)
                              .standard_normal((64, 3))).astype(np.float32)),
                    n_warmup=300, n_samples=300, seed=3)
    b = bridge_evidence(t, res)
    assert abs(b.log_z - lz) < max(4 * b.log_z_se, 0.02)
    assert b.details["tau"] >= 1.0


# -- Laplace -----------------------------------------------------------------------

def test_laplace_exact_on_a_gaussian_posterior():
    t, lz, _, _ = _conjugate()
    r = laplace_evidence(t, u0=np.zeros(3), n_starts=8)
    assert r.log_z == pytest.approx(lz, abs=1e-3)
    assert r.method == "laplace" and np.isnan(r.log_z_se)


def test_laplace_needs_the_density_mode():
    t, _, _, _ = _conjugate()
    opt = anvil.optimize(t, np.zeros(3), n_starts=4)          # objective="mle"
    with pytest.raises(ValueError, match="objective='density'"):
        laplace_evidence(t, opt)
