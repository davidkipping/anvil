"""Engine-level tests, chiefly that the sampling pipeline is invisible.

Pipelining changes *when* arrays are evaluated, never what is computed, so
every result must be bit-identical at any depth. These tests assert that
exactly rather than statistically — a missed synchronisation would
otherwise corrupt stored frames silently.
"""

import mlx.core as mx
import numpy as np
import pytest

from anvil import run
from anvil.kernels.chees import ChEESHMC
from anvil.kernels.ensemble import EnsembleKernel
from anvil.kernels.metropolis import RandomWalkMetropolis
from anvil.surrogate import TrainingArchive
from anvil.targets import correlated_gaussian, make_transit_target


def _setup(dim=4, n=128, seed=0):
    target, mu, cov = correlated_gaussian(dim, rho=0.5, seed=seed)
    u0 = mx.array(
        (mu + np.random.default_rng(seed + 1).standard_normal((n, dim)))
        .astype(np.float32))
    return target, u0


def _kernels(target):
    return {
        "rwm": lambda: RandomWalkMetropolis(target),
        "ensemble": lambda: EnsembleKernel(target, seed=3),
        "chees": lambda: ChEESHMC(target),
    }


@pytest.mark.parametrize("kernel_name", ["rwm", "ensemble", "chees"])
def test_pipelining_is_bit_identical(kernel_name):
    target, u0 = _setup()
    make = _kernels(target)[kernel_name]
    kw = dict(n_warmup=40, n_samples=25, seed=7)
    ref = run(make(), target, u0, pipeline=0, **kw)
    for depth in (1, 2, 4):
        got = run(make(), target, u0, pipeline=depth, **kw)
        np.testing.assert_array_equal(ref.get_chain(), got.get_chain())
        np.testing.assert_array_equal(ref.get_log_prob(), got.get_log_prob())
        np.testing.assert_array_equal(ref.accept_fraction, got.accept_fraction)
        assert ref.extras["n_divergent"] == got.extras["n_divergent"]
        np.testing.assert_array_equal(
            np.array(ref.final_state["u"]), np.array(got.final_state["u"]))


def test_pipelining_respects_thinning():
    target, u0 = _setup()
    kw = dict(n_warmup=30, n_samples=9, thin=3, seed=8)
    ref = run(RandomWalkMetropolis(target), target, u0, pipeline=0, **kw)
    got = run(RandomWalkMetropolis(target), target, u0, pipeline=2, **kw)
    assert ref.get_chain().shape[0] == 9
    np.testing.assert_array_equal(ref.get_chain(), got.get_chain())


@pytest.mark.parametrize("depth,n_samples", [(4, 10), (3, 7), (8, 2), (2, 1)])
def test_pipeline_drains_completely(depth, n_samples):
    """Frames still in flight at the end must be stored, in order, even
    when the sample count is not a multiple of the depth."""
    target, u0 = _setup()
    kw = dict(n_warmup=10, n_samples=n_samples, seed=9)
    ref = run(RandomWalkMetropolis(target), target, u0, pipeline=0, **kw)
    got = run(RandomWalkMetropolis(target), target, u0, pipeline=depth, **kw)
    assert got.get_chain().shape[0] == n_samples
    np.testing.assert_array_equal(ref.get_chain(), got.get_chain())


def test_pipelining_preserves_archive_order():
    target, u0 = _setup()
    kw = dict(n_warmup=20, n_samples=12, seed=10)
    archives = {}
    for depth in (0, 3):
        arc = TrainingArchive(dim=4, capacity=10_000)
        run(RandomWalkMetropolis(target), target, u0, pipeline=depth,
            archive=arc, **kw)
        archives[depth] = arc.as_arrays()
    for a, b in zip(archives[0], archives[3]):
        np.testing.assert_array_equal(a, b)


def test_pipelining_with_reanchoring_matches_blocking():
    """Re-anchoring runs on the host, so the pipeline must be drained
    before it; the result must still be bit-identical."""
    tt = make_transit_target(n_data=5_000, seed=0)
    u0 = mx.array(
        (tt.transform.from_model_np(tt.truth_model)
         + 1e-3 * np.random.default_rng(0).standard_normal((64, 6)))
        .astype(np.float32))
    kw = dict(n_warmup=12, n_samples=20, seed=11, reanchor_every=5)
    ref = run(RandomWalkMetropolis(tt.target), tt.target, u0, pipeline=0, **kw)
    got = run(RandomWalkMetropolis(tt.target), tt.target, u0, pipeline=2, **kw)
    assert np.all(np.isfinite(ref.get_chain()))
    np.testing.assert_array_equal(ref.get_chain(), got.get_chain())


def test_auto_selection_follows_the_measured_cost(capsys, monkeypatch):
    """Auto-selection must key off measured per-iteration cost. The
    threshold is monkeypatched rather than the hardware timed, so this
    tests the decision logic and not how fast the runner happens to be
    (a CI box with no GPU makes even a trivial target ~1 ms/iteration)."""
    import anvil.engine as E

    target, u0 = _setup(n=64)
    kw = dict(n_warmup=2, n_samples=20, seed=13, progress=1000)

    monkeypatch.setattr(E, "_PIPELINE_THRESHOLD_S", 1e3)      # everything is cheap
    enabled = run(RandomWalkMetropolis(target), target, u0, **kw)
    assert "pipelining at depth" in capsys.readouterr().out

    monkeypatch.setattr(E, "_PIPELINE_THRESHOLD_S", 1e-12)    # nothing is cheap
    declined = run(RandomWalkMetropolis(target), target, u0, **kw)
    assert "not pipelining" in capsys.readouterr().out

    # and the two agree exactly, which is the property that matters
    np.testing.assert_array_equal(enabled.get_chain(), declined.get_chain())


def test_explicit_depth_overrides_auto(capsys):
    """An explicit int must be honoured without any timing probe."""
    target, u0 = _setup(n=64)
    kw = dict(n_warmup=2, n_samples=20, seed=14, progress=1000)
    run(RandomWalkMetropolis(target), target, u0, pipeline=2, **kw)
    out = capsys.readouterr().out
    assert "pipelining at depth" not in out and "not pipelining" not in out


# --- warmup diagnostics -----------------------------------------------------

def test_warmup_trace_is_recorded_and_disableable():
    target, u0 = _setup(n=64)
    r = run(RandomWalkMetropolis(target), target, u0,
            n_warmup=100, n_samples=5, seed=20, warmup_probes=20)
    tr = r.warmup_trace
    assert sorted(tr) == ["accept", "iter", "mean", "sd", "step_size"]
    assert len(tr["iter"]) == 20 and tr["iter"][-1] == 100
    assert tr["sd"].shape == (20, 4) and tr["mean"].shape == (20, 4)
    assert np.all(np.isfinite(tr["accept"]))

    off = run(RandomWalkMetropolis(target), target, u0,
              n_warmup=100, n_samples=5, seed=20, warmup_probes=0)
    assert off.warmup_trace is None
    # and the trace must not perturb the run
    np.testing.assert_array_equal(r.get_chain(), off.get_chain())


class _FakeResults:
    """Minimal stand-in: warmup_report only reads the trace and n_chains."""

    def __init__(self, sd, accept, step_size, n_chains):
        n = len(sd)
        self.n_chains = n_chains
        self.warmup_trace = {
            "iter": np.arange(1, n + 1) * 10,
            "sd": np.asarray(sd, dtype=np.float64),
            "mean": np.zeros_like(np.asarray(sd, dtype=np.float64)),
            "accept": np.full(n, accept, dtype=np.float64),
            "step_size": np.full(n, step_size, dtype=np.float64),
        }


def test_warmup_report_verdicts_on_known_traces():
    """The verdict logic is pure, so exercise it on traces whose right
    answer is known by construction rather than on runs whose behaviour
    has to be reverse-engineered."""
    from anvil import warmup_report
    n = 30

    # still expanding at the end -> too short
    grow = np.linspace(0.1, 3.0, n)[:, None] * np.ones((1, 3))
    rep = warmup_report(_FakeResults(grow, 0.4, 0.1, 512))
    assert rep.verdict == "TOO SHORT" and rep.settled_at is None

    # expands, then flat for most of the run -> settled early
    early = np.concatenate([np.linspace(0.1, 1.0, 5), np.ones(n - 5)])[:, None] \
        * np.ones((1, 3))
    rep = warmup_report(_FakeResults(early, 0.4, 0.1, 512))
    assert rep.verdict == "LONGER THAN NEEDED"
    assert rep.settled_at is not None and rep.settled_at <= 60

    # expands for most of it, settling only near the end -> about right
    late = np.concatenate([np.linspace(0.1, 1.0, 22), np.ones(n - 22)])[:, None] \
        * np.ones((1, 3))
    rep = warmup_report(_FakeResults(late, 0.4, 0.1, 512))
    assert rep.verdict == "OK"

    # never moved at all -> cannot tell converged from stuck
    flat = np.ones((n, 3)) * 2.45
    rep = warmup_report(_FakeResults(flat, 0.98, 7e-6, 256))
    assert rep.verdict == "INCONCLUSIVE"
    assert "never changed" in rep.suggestion
    # the tell for a stalled HMC run is a TINY step size with HIGH
    # acceptance, so the report must mention the step size
    assert "step size" in rep.suggestion

    # acceptance collapsed, but the spread did move -> a different failure
    rep = warmup_report(_FakeResults(grow, 0.001, 0.1, 512))
    assert rep.verdict == "FAILED" and "acceptance" in rep.suggestion


def test_warmup_report_on_a_real_run_is_sane():
    from anvil import warmup_report
    target, u0 = _setup(n=128)
    r = run(RandomWalkMetropolis(target), target, u0,
            n_warmup=300, n_samples=10, seed=23)
    rep = warmup_report(r)
    assert rep.verdict in ("OK", "LONGER THAN NEEDED")
    assert rep.n_chains == 128 and rep.n_warmup == 300
    assert 0.0 < rep.final_accept <= 1.0 and str(rep)


def test_warmup_report_tolerance_scales_with_chain_count():
    """The cross-chain spread is itself a noisy estimate; demanding tighter
    agreement than that noise floor would never succeed."""
    from anvil import warmup_report
    target, u0_small = _setup(n=32)
    _, u0_big = _setup(n=512)
    small = warmup_report(run(RandomWalkMetropolis(target), target, u0_small,
                              n_warmup=200, n_samples=5, seed=26))
    big = warmup_report(run(RandomWalkMetropolis(target), target, u0_big,
                            n_warmup=200, n_samples=5, seed=26))
    assert small.noise_floor > big.noise_floor
    assert abs(small.noise_floor - 1 / np.sqrt(2 * 31)) < 1e-9


def test_warmup_report_requires_a_trace():
    from anvil import warmup_report
    target, u0 = _setup(n=64)
    r = run(RandomWalkMetropolis(target), target, u0,
            n_warmup=20, n_samples=5, seed=27, warmup_probes=0)
    with pytest.raises(ValueError, match="warmup_probes"):
        warmup_report(r)


# -- per-chain divergences and the progress callback ----------------------


def test_divergences_are_reported_per_chain_as_well_as_summed():
    """A scalar count cannot tell one pathological chain from a diffusely
    unhappy ensemble; the funnel produces both."""
    from anvil.targets import neals_funnel

    target = neals_funnel(5)
    u0 = mx.array(
        np.random.default_rng(0).standard_normal((128, 5)).astype(np.float32)
        * 0.5)
    res = run(ChEESHMC(target), target, u0, n_warmup=200, n_samples=200, seed=2)
    per_chain = res.extras["divergent_per_chain"]
    assert per_chain.shape == (128,)
    assert np.all(per_chain >= 0)
    assert per_chain.sum() == res.extras["n_divergent"]
    assert res.extras["n_divergent"] > 0        # the funnel is meant to be hard


def test_callback_sees_both_phases_with_the_numbers_progress_prints(capsys):
    target, u0 = _setup()
    seen = []
    res = run(RandomWalkMetropolis(target), target, u0,
              n_warmup=40, n_samples=40, seed=1,
              callback=lambda ph, it, info: seen.append((ph, it, info)))
    assert capsys.readouterr().out == ""        # a callback is not a printer
    phases = [p for p, _, _ in seen]
    assert phases.count("warmup") == 10 and phases.count("sample") == 10
    assert [it for p, it, _ in seen if p == "warmup"] == list(range(4, 41, 4))
    for phase, _, info in seen:
        assert info["total"] == 40
        assert 0.0 <= info["accept"] <= 1.0
        assert info["rate"] > 0 and info["elapsed"] > 0 and info["eta"] >= 0
        assert info["step_size"] is None       # RWM has no step_size param
        assert ("n_divergent" in info) == (phase == "sample")


def test_callback_follows_an_explicit_progress_cadence(capsys):
    target, u0 = _setup()
    seen = []
    run(ChEESHMC(target), target, u0, n_warmup=20, n_samples=20, seed=1,
        progress=5, callback=lambda ph, it, info: seen.append((ph, it, info)))
    assert capsys.readouterr().out != ""        # progress=5 still prints
    assert [it for p, it, _ in seen if p == "warmup"] == [5, 10, 15, 20]
    assert all(info["step_size"] > 0 for _, _, info in seen)


def test_no_callback_and_no_progress_stays_silent(capsys):
    target, u0 = _setup()
    run(RandomWalkMetropolis(target), target, u0, n_warmup=10, n_samples=10)
    assert capsys.readouterr().out == ""
