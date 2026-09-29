"""A chain that reaches a bounded parameter's edge must be able to leave.

Both halves of the trap were real (measured in anvil-gp's
docs/anvil-chees-boundary.md): the bounded log-Jacobian saturated to -inf
in float32, and the divergence veto then rejected the one proposal that
would have rescued the chain — so the state was absorbing and silent.
"""

import math

import mlx.core as mx
import numpy as np

from anvil import ParamSpec, Transform, TransformedLogDensity, run
from anvil.kernels.chees import ChEESHMC


def _flat_bounded(dim=2, lo=-1.0, hi=1.0):
    """Flat prior over a box. In u-space the posterior is exactly the
    standard logistic distribution (mean 0, sd pi/sqrt(3)), because the
    log-Jacobian *is* the whole log density."""
    tr = Transform([ParamSpec(f"p{i}", lo=lo, hi=hi) for i in range(dim)])
    return TransformedLogDensity(lambda v: mx.zeros(v.shape[0]), tr), tr


LOGISTIC_SD = math.pi / math.sqrt(3.0)


def test_bounded_log_jacobian_is_finite_where_the_sigmoid_form_is_not():
    _, tr = _flat_bounded(dim=1)
    u = np.array([[0.0], [10.0], [15.0], [18.0], [25.0], [80.0], [-80.0]])
    got = np.array(tr.log_det_jac(mx.array(u.astype(np.float32))))
    assert np.all(np.isfinite(got)), got
    # mx.sigmoid saturates at 18 in float32 -- the reason the old form died
    assert float(mx.sigmoid(mx.array(18.0, dtype=mx.float32)).item()) == 1.0


def test_it_agrees_with_the_float64_reference_everywhere():
    """2a changes the expression's conditioning, not its value: a run that
    never approaches a boundary cannot move."""
    _, tr = _flat_bounded(dim=3)
    rng = np.random.default_rng(0)
    u = rng.normal(0.0, 2.0, size=(4096, 3))            # ordinary sampling range
    got = np.array(tr.log_det_jac(mx.array(u.astype(np.float32))))
    ref = tr.log_det_jac_np(u)
    np.testing.assert_allclose(got, ref, rtol=0, atol=3e-5)
    # and out where float64 is the only thing still honest
    far = np.array([[14.0, 20.0, 30.0], [-30.0, 0.0, 22.0]])
    np.testing.assert_allclose(
        np.array(tr.log_det_jac(mx.array(far.astype(np.float32)))),
        tr.log_det_jac_np(far), rtol=1e-6)


def test_chees_chains_started_at_the_boundary_rejoin_the_bulk():
    target, _ = _flat_bounded(dim=2)
    n = 128
    rng = np.random.default_rng(1)
    u0 = rng.normal(0.0, 0.5, size=(n, 2)).astype(np.float32)
    stuck = slice(0, 8)
    u0[0:4] = 15.0      # saturated in the old float32 log-Jacobian
    u0[4:8] = 25.0      # far past it
    res = run(ChEESHMC(target), target, mx.array(u0),
              n_warmup=200, n_samples=200, seed=3)

    final = np.array(res.final_state["u"])
    chain = res.get_chain()                              # (n_kept, n, dim)
    assert np.all(np.isfinite(chain))
    # the once-stuck chains are back inside the bulk ...
    assert np.max(np.abs(final[stuck])) < 3 * LOGISTIC_SD, final[stuck]
    # ... having actually moved, not merely started finite
    assert np.all(np.abs(chain[-1, stuck] - u0[stuck]) > 1.0)
    # ... and they are moving, not parked
    assert np.all(res.accept_fraction[stuck] > 0.1)
    assert res.extras["n_divergent"] == 0
    assert res.extras["divergent_per_chain"].shape == (n,)


def test_the_rest_of_the_ensemble_is_undisturbed_and_the_marginal_is_right():
    """Eight boundary chains out of 128 must not corrupt the answer: the
    u-space marginal of a flat prior over a box is the standard logistic."""
    target, _ = _flat_bounded(dim=2)
    rng = np.random.default_rng(2)
    u0 = rng.normal(0.0, 0.5, size=(128, 2)).astype(np.float32)
    u0[0:8] = 15.0
    res = run(ChEESHMC(target), target, mx.array(u0),
              n_warmup=400, n_samples=600, seed=4)
    draws = res.get_chain(discard=100, flat=True)
    assert abs(draws.mean()) < 0.1
    assert abs(draws.std() - LOGISTIC_SD) / LOGISTIC_SD < 0.08
