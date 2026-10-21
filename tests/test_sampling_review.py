"""Fixes from the 0.10.2 review of sampling mode, each failing on 0.10.1."""

import math

import mlx.core as mx
import numpy as np
import pytest

import anvil
import anvil.kernels.chees as C
from anvil import ParamSpec, Transform, TransformedLogDensity
from anvil.adaptation.chees_criterion import chees_gradient
from anvil.adaptation.moments import (DenseMomentsState, dense_factors,
                                      init_moments, update_moments)


def _gauss(dim):
    return anvil.FunctionLogDensity(lambda u: -0.5 * mx.sum(u * u, axis=-1), dim)


@pytest.mark.parametrize("n", [100, 1000])
def test_masked_adaptation_is_bit_identical_when_every_chain_is_live(n):
    # 0.10.1's mask divided a masked sum by n, which equals mx.mean only for
    # power-of-two n: draws at 100 chains differed from 0.10.0 by up to 8e-3
    rng = np.random.default_rng(0)
    u = mx.array(rng.standard_normal((n, 4)).astype(np.float32))
    live = mx.ones((n,), dtype=mx.bool_)
    a = update_moments(init_moments(4), u, 3)
    b = update_moments(init_moments(4), u, 3, include=live)
    assert np.array_equal(np.array(a.m1), np.array(b.m1))
    assert np.array_equal(np.array(a.m2), np.array(b.m2))
    q1 = mx.array(rng.standard_normal((n, 4)).astype(np.float32))
    v = mx.array(rng.standard_normal((n, 4)).astype(np.float32))
    acc = mx.array(rng.uniform(size=n).astype(np.float32))
    assert np.array_equal(np.array(chees_gradient(u, q1, v, acc, 0.7)),
                          np.array(chees_gradient(u, q1, v, acc, 0.7, include=live)))


def test_a_chain_beyond_float32_exp_range_does_not_freeze_adaptation():
    # a lower-bounded parameter started exactly on its bound sits at
    # u = -690: the upper branch's exp(-u) overflowed, its discarded inf
    # back-propagated as NaN, and the step size went to 0 for every chain
    tr = Transform([ParamSpec("s", lo=0.0), ParamSpec("a"), ParamSpec("b"), ParamSpec("c")])
    t = TransformedLogDensity(lambda v: -0.5 * mx.sum((v - 1.0) ** 2, axis=-1), tr)
    v0 = 1.0 + 0.3 * np.random.default_rng(4).standard_normal((64, 4))
    v0[:, 0] = np.abs(v0[:, 0])
    v0[0, 0] = 0.0
    u0 = tr.from_model_np(v0).astype(np.float32)
    g = np.array(t.log_prob_and_grad(mx.array(u0[:1]))[1])
    assert np.all(np.isfinite(g))
    r = anvil.run(C.ChEESHMC(t), t, mx.array(u0), n_warmup=200, n_samples=20, seed=0)
    assert float(r.final_params["step_size"].item()) > 0.05
    assert np.array(r.get_chain())[:, 1:, 1].std() > 0.1     # the chains move


def test_large_free_and_bounded_u_give_finite_gradients():
    # far out in u, but each parameter's own map still finite there (an
    # upper-bounded column at u = -100 really is at -inf, so it is not here)
    tr = Transform([ParamSpec("a", 0.0, 1.0), ParamSpec("b"), ParamSpec("c", hi=2.0),
                    ParamSpec("d", lo=-1.0)])
    t = TransformedLogDensity(lambda v: -0.5 * mx.sum(v * v, axis=-1) * 1e-30, tr)
    u = mx.array(np.array([[100.0, 100.0, 60.0, -100.0],
                           [-100.0, -100.0, -40.0, 40.0]], np.float32))
    lp, g = t.log_prob_and_grad(u)
    assert np.all(np.isfinite(np.array(lp))) and np.all(np.isfinite(np.array(g)))


def test_dense_factors_hand_the_kernel_the_ridged_matrix():
    # momentum came from (R + ridge)^-1 while the kinetic energy used R:
    # not a valid HMC kernel, variances low by ~ridge / lambda_min(R)
    m1 = np.zeros(2)
    m2 = np.array([[1.0, 0.9], [0.9, 1.0]])
    sd, R, B = dense_factors(DenseMomentsState(m1=m1, m2=m2), ridge=0.3)
    R, B = np.array(R, np.float64), np.array(B, np.float64)
    np.testing.assert_allclose(np.diag(R), 1.3, rtol=1e-6)
    np.testing.assert_allclose(B.T @ B, np.linalg.inv(R), rtol=1e-5)


def test_nested_rhat_uses_the_published_denominators():
    # Margossian et al. (2022), Def. 2.2: 1/(K-1) and 1/(M-1), not means
    rng = np.random.default_rng(0)
    n, k, j = 3, 4, 2
    x = rng.standard_normal((n, k * j)) + np.repeat(rng.standard_normal(k), j)
    cm = x.mean(0).reshape(k, j)
    sm = cm.mean(1)
    b = ((sm - sm.mean()) ** 2).sum() / (k - 1)
    bw = ((cm - sm[:, None]) ** 2).sum(1) / (j - 1)
    w = x.var(0, ddof=1).reshape(k, j).mean(1)
    want = math.sqrt(1 + b / (bw + w).mean())
    got = float(anvil.diagnostics.nested_rhat(x, n_superchains=k)[0])
    assert got == pytest.approx(want, rel=1e-12)


def test_reusing_a_kernel_object_reproduces_the_run():
    t = _gauss(4)
    u0 = mx.array(np.random.default_rng(1).standard_normal((64, 4)).astype(np.float32))
    k = C.ChEESHMC(t)
    a = anvil.run(k, t, u0, n_warmup=50, n_samples=20, seed=3).get_chain()
    b = anvil.run(k, t, u0, n_warmup=50, n_samples=20, seed=3).get_chain()
    assert np.array_equal(a, b)
    ek = anvil.EnsembleKernel(t, moves=[(anvil.StretchMove(), .5), (anvil.DEMove(), .5)])
    a = anvil.run(ek, t, u0, n_warmup=0, n_samples=30, seed=3).get_chain()
    b = anvil.run(ek, t, u0, n_warmup=0, n_samples=30, seed=3).get_chain()
    assert np.array_equal(a, b)


@pytest.mark.skipif(not mx.metal.is_available(), reason="fixed-point needs Metal")
def test_fixed_point_reduction_keeps_failures_and_large_terms():
    # the kernel clamped every term to +-2**43 * 2**-31: NaN, -inf and a
    # genuine -1e5 all summed to about -8195.5
    from anvil.precision import PrecisionPolicy, chunked_sum
    pol = PrecisionPolicy(reduction="fixed_point", chunk_size=64)
    base = np.full((4, 100), -0.5, np.float32)
    base[1, 7] = np.nan
    base[2, 50] = -np.inf
    base[3, 99] = -1e5
    out = np.array(chunked_sum(lambda a, b: mx.array(base[:, a:b]), 100, pol))
    assert out[0] == pytest.approx(-50.0)
    assert np.isnan(out[1]) and out[2] == -np.inf
    assert out[3] == pytest.approx(-1e5 - 49.5, rel=1e-6)


# -- 0.11.0: the diagonal preconditioner's moments -------------------------------

@pytest.mark.parametrize("mu,sd", [(3.0, 1e-3), (1.0, 3e-4), (10.0, 1e-3), (0.0, 3e-5)])
def test_preconditioner_scale_survives_float32_cancellation_and_has_no_floor(mu, sd):
    # E[u^2] - E[u]^2 in float32 read 10x small at u = 3 (sd 1e-3), 3x large
    # at u = 1 (sd 3e-4); and a floor of 1e-4 in u made sd 3e-5 read 1e-4
    from anvil.adaptation.moments import stddev
    rng = np.random.default_rng(0)
    ms = init_moments(1)
    for t in range(1, 301):
        u = mx.array((mu + sd * rng.standard_normal((1024, 1))).astype(np.float32))
        ms = update_moments(ms, u, t)
    assert float(stddev(ms).item()) == pytest.approx(sd, rel=0.05)


def test_a_tight_parameter_no_longer_shrinks_everyone_s_step():
    # sd 3e-5 in one coordinate used to be floored at 1e-4, and the step size
    # then fell from ~1.0 to ~0.4 for every coordinate
    sd = mx.array(np.array([3e-5, 1, 1, 1], np.float32))
    t = anvil.FunctionLogDensity(lambda u: -0.5 * mx.sum((u / sd) ** 2, axis=-1), 4)
    u0 = mx.array((np.random.default_rng(0).standard_normal((128, 4)) * np.array([3e-5, 1, 1, 1])).astype(np.float32))
    r = anvil.run(C.ChEESHMC(t), t, u0, n_warmup=400, n_samples=100, seed=0)
    sigma = np.sqrt(np.array(r.final_params["inv_mass"]))
    assert sigma[0] == pytest.approx(3e-5, rel=0.1)
    assert float(r.final_params["step_size"].item()) > 0.7
