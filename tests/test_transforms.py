import math

import mlx.core as mx
import numpy as np
import pytest

from anvil import run
from anvil.kernels.metropolis import RandomWalkMetropolis
from anvil.transforms import ParamSpec, Transform, TransformedLogDensity

SPECS = [
    ParamSpec("bounded", lo=-2.0, hi=5.0),
    ParamSpec("lower", lo=0.0, scale=2.0),
    ParamSpec("upper", hi=3.0, scale=0.5),
    ParamSpec("free", loc=10.0, scale=4.0),
]


def test_round_trip_all_bound_types():
    tr = Transform(SPECS)
    rng = np.random.default_rng(0)
    u = rng.normal(size=(100, 4)) * 2
    v = tr.model_np(u)
    np.testing.assert_allclose(tr.from_model_np(v), u, atol=1e-9)
    # model values respect bounds
    assert np.all(v[:, 0] > -2) and np.all(v[:, 0] < 5)
    assert np.all(v[:, 1] > 0)
    assert np.all(v[:, 2] < 3)


def test_mx_and_np_paths_agree():
    tr = Transform(SPECS)
    rng = np.random.default_rng(1)
    u = rng.normal(size=(50, 4)).astype(np.float32)
    v_mx = np.array(tr.to_model(mx.array(u)), dtype=np.float64)
    v_np = tr.model_np(u)
    np.testing.assert_allclose(v_mx, v_np, rtol=2e-5, atol=2e-5)
    lj_mx = np.array(tr.log_det_jac(mx.array(u)), dtype=np.float64)
    lj_np = tr.log_det_jac_np(u)
    np.testing.assert_allclose(lj_mx, lj_np, rtol=1e-4, atol=1e-4)


def test_log_det_jac_matches_finite_differences():
    tr = Transform(SPECS)
    rng = np.random.default_rng(2)
    u = rng.normal(size=(20, 4))
    eps = 1e-6
    lj = tr.log_det_jac_np(u)
    fd = np.zeros_like(lj)
    for d in range(4):
        up, um = u.copy(), u.copy()
        up[:, d] += eps
        um[:, d] -= eps
        dv = (tr.model_np(up)[:, d] - tr.model_np(um)[:, d]) / (2 * eps)
        fd += np.log(np.abs(dv))
    np.testing.assert_allclose(lj, fd, rtol=1e-5, atol=1e-5)


def test_to_physical_restores_large_offsets_in_fp64():
    t_ref = 2_457_000.0
    tr = Transform([ParamSpec("t0_off", lo=-0.5, hi=0.5, report_offset=t_ref)])
    v = np.array([[0.1234567891]])
    x = tr.to_physical(v)
    # fp64 keeps ~1e-9 day precision at BJD magnitudes; fp32 could not
    assert abs(x[0, 0] - (t_ref + 0.1234567891)) < 1e-8


@pytest.mark.slow
def test_flat_model_density_samples_uniform_within_bounds():
    # constant model-space log-prob + Jacobian => uniform over (lo, hi)
    lo, hi = -2.0, 5.0
    tr = Transform([ParamSpec("a", lo=lo, hi=hi)])
    target = TransformedLogDensity(lambda v: mx.zeros(v.shape[:1]), tr)
    u0 = mx.array(np.random.default_rng(0).normal(size=(1024, 1)).astype(np.float32))
    res = run(RandomWalkMetropolis(target), target, u0,
              n_warmup=800, n_samples=300, thin=4, seed=3)
    v = tr.model_np(res.get_chain(flat=True))
    mean, var = v.mean(), v.var()
    width = hi - lo
    assert abs(mean - (lo + hi) / 2) < 0.05 * width
    assert abs(var - width**2 / 12) / (width**2 / 12) < 0.1
