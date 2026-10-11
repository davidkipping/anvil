"""Lost chains must be reported, not passed.

A target that reports a failed evaluation as a constant creates a flat
region; walkers that land there drift outwards for good. anvil cannot make
such a target well-behaved, but it can refuse to accept overflowed positions
and it must say so rather than return a verdict about warmup length. The
reproducer is anvil-gp's (docs/anvil-warmup-lost-walkers.md): a 4-D normal
failing beyond radius 25, half of 64 walkers started at radius 30.
"""

import mlx.core as mx
import numpy as np
import pytest

import anvil
from anvil.diagnostics import lost_chains
from anvil.kernels.ensemble import EnsembleKernel
from anvil.kernels.metropolis import RandomWalkMetropolis

DIM, RADIUS, FAIL = 4, 25.0, -1e30


def _target(sloped):
    def model_log_prob(v):
        r2 = mx.sum(v * v, axis=-1)
        if sloped:
            failed = FAIL * (1.0 + mx.stop_gradient(mx.log1p(r2 / DIM)))
        else:
            failed = mx.full(r2.shape, FAIL, dtype=v.dtype)
        return mx.where(mx.sqrt(r2) < RADIUS, -0.5 * r2, failed)
    tr = anvil.Transform([anvil.ParamSpec(f"x{i}") for i in range(DIM)])
    return anvil.TransformedLogDensity(model_log_prob, tr)


def _u0():
    rng = np.random.default_rng(5)
    u0 = (0.3 * rng.standard_normal((64, DIM))).astype(np.float32)
    u0[:32] = 15.0
    return mx.array(u0)


@pytest.fixture(scope="module")
def constant_run():
    t = _target(sloped=False)
    return t, anvil.run(EnsembleKernel(t, seed=2), t, _u0(),
                        n_warmup=400, n_samples=600, seed=3)


def test_no_position_ever_overflows(constant_run):
    _, res = constant_run
    assert np.all(np.isfinite(res.get_chain()))
    assert np.all(np.isfinite(np.array(res.final_state["u"])))


def test_warmup_report_names_the_lost_chains_instead_of_a_length_verdict(
        constant_run):
    _, res = constant_run
    with np.errstate(all="raise"):                 # no inf/inf in the settle test
        rep = anvil.warmup_report(res)
    assert rep.verdict == "LOST CHAINS", rep
    assert len(rep.lost) > 0
    assert "longer warmup will not" in rep.suggestion.lower()
    # exactly the chains sitting at the failure value
    lp_end = res.warmup_trace["log_prob"]
    np.testing.assert_array_equal(sorted(rep.lost), np.flatnonzero(lp_end <= FAIL / 2))


def test_diagnose_lists_lost_chains_beside_rhat(constant_run):
    _, res = constant_run
    d = anvil.diagnose(res.get_chain(), log_prob=res.get_log_prob())
    final_lp = np.array(res.final_state["log_prob"])
    np.testing.assert_array_equal(d.lost, np.flatnonzero(final_lp <= FAIL / 2))
    assert "LOST CHAINS" in str(d)
    # without log_prob only non-finite draws can be detected -- none here
    assert anvil.diagnose(res.get_chain()).lost.size == 0


def test_the_healthy_control_flags_nothing():
    """The sloped failure value is the fix the docs recommend."""
    t = _target(sloped=True)
    res = anvil.run(EnsembleKernel(t, seed=2), t, _u0(),
                    n_warmup=400, n_samples=600, seed=3)
    rep = anvil.warmup_report(res)
    assert rep.verdict != "LOST CHAINS" and rep.lost == ()
    assert anvil.diagnose(res.get_chain(), log_prob=res.get_log_prob()).lost.size == 0


def test_rwm_rejects_an_overflowed_proposal():
    """Same flat region, random-walk kernel: positions stay finite."""
    t = _target(sloped=False)
    k = RandomWalkMetropolis(t, initial_scale=1e37)     # proposals overflow
    res = anvil.run(k, t, _u0(), n_warmup=20, n_samples=20, seed=1)
    assert np.all(np.isfinite(res.get_chain()))


# -- the check itself, on synthetic inputs with known answers ---------------

def test_lost_chains_on_known_inputs():
    rng = np.random.default_rng(0)
    lp = -0.5 * rng.chisquare(4, size=512)
    assert lost_chains(lp).size == 0                    # healthy: nothing
    lp2 = lp.copy(); lp2[[3, 99]] = FAIL; lp2[7] = -np.inf; lp2[8] = np.nan
    assert lost_chains(lp2).tolist() == [3, 7, 8, 99]
    fin = np.ones(512, bool); fin[200] = False
    assert lost_chains(lp, positions_finite=fin).tolist() == [200]


def test_lost_chains_still_works_when_most_chains_are_lost():
    """The median goes blind once half are lost; the fixed floor does not."""
    lp = np.full(64, FAIL); lp[:10] = -2.0
    assert lost_chains(lp).tolist() == list(range(10, 64))


def test_lost_chains_uses_per_chain_medians_for_draw_arrays():
    lp = np.full((50, 8), -2.0); lp[:, 5] = FAIL; lp[0, 2] = FAIL  # one bad draw
    assert lost_chains(lp).tolist() == [5]


# -- a record that disagrees with the target at a float32 feasibility edge --
#
# anvil-gp saw ChEES chains whose recorded log-probability was -17,000 while
# the target evaluated eagerly at their positions gave the failure value
# -6.8e30: compiled and eager evaluation differ at the edge. The record is
# then a feasible value and every check that reads it is blind.

class _EdgeTarget(anvil.LogDensity):
    """A target whose compiled and eager evaluations disagree on a region:
    eager reports failure there, compiled reports an ordinary value."""
    dim = 2

    def __init__(self):
        self.eager_mode = True

    def log_prob(self, u):
        base = -0.5 * mx.sum(u * u, axis=-1)
        in_edge = u[:, 0] > 5.0
        failed = mx.full(base.shape, -1e30, dtype=base.dtype)
        # mx.compile traces with tracers: emulate the disagreement by
        # keying on a flag that differs between the two evaluations
        return mx.where(in_edge, failed if self.eager_mode else base - 17000.0, base)


def test_lost_at_end_catches_a_chain_whose_record_is_wrong():
    from anvil.engine import _eager_log_prob, _lost_at_end
    from anvil.kernels.chees import ChEESHMC

    t = _EdgeTarget()
    k = ChEESHMC(t)
    u = mx.array(np.random.default_rng(0).standard_normal((64, 2)).astype(np.float32))
    u = mx.concatenate([u[:61], mx.array([[6.0, 0.0]] * 3, dtype=mx.float32)])
    t.eager_mode = False                       # the record is written "compiled"
    st = k.init(mx.random.key(0), u, t)
    rec = np.array(st["log_prob"])
    assert np.all(rec > -1e5)                  # the record sees nothing wrong
    t.eager_mode = True                        # the eager re-evaluation sees it
    assert (_eager_log_prob(k, st, t)[61:] <= -1e29).all()
    assert _lost_at_end(k, st, t).tolist() == [61, 62, 63]


def test_a_healthy_run_reports_no_lost_chains_in_extras():
    t = _target(sloped=True)
    res = anvil.run(EnsembleKernel(t, seed=2), t, _u0() * 0.0 + 0.3,
                    n_warmup=50, n_samples=20, seed=3)
    assert res.extras["lost_chains"].size == 0
    assert "log_prob_eager" in res.warmup_trace


def test_extras_lost_chains_matches_the_constant_failure_run(constant_run):
    _, res = constant_run
    final = np.array(res.final_state["log_prob"])
    np.testing.assert_array_equal(res.extras["lost_chains"],
                                  np.flatnonzero(final <= FAIL / 2))
