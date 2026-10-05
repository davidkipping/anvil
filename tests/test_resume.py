"""Continuing a run instead of restarting it.

The point of a resume is that the second segment is the *same* Markov
chain as the first: same frozen adaptation, and — the part that is easy to
get silently wrong — a key stream that carries on rather than replaying
the warmup keys. Both are asserted here directly.
"""

import re

import mlx.core as mx
import numpy as np
import pytest

import anvil
from anvil import load_state, run
from anvil.kernels.chees import ChEESHMC
from anvil.kernels.ensemble import EnsembleKernel
from anvil.kernels.metropolis import RandomWalkMetropolis
from anvil.targets import correlated_gaussian


def _setup(dim=4, n=128, seed=0):
    target, mu, cov = correlated_gaussian(dim, rho=0.5, seed=seed)
    u0 = mx.array(
        (mu + np.random.default_rng(seed + 1).standard_normal((n, dim)))
        .astype(np.float32))
    return target, u0, mu, cov


def _kernels(target):
    return {
        "rwm": lambda: RandomWalkMetropolis(target),
        "ensemble": lambda: EnsembleKernel(target, seed=3),
        "chees": lambda: ChEESHMC(target),
    }


@pytest.mark.parametrize("kernel_name", ["rwm", "ensemble", "chees"])
def test_resume_freezes_the_adaptation_exactly(kernel_name):
    """No re-adaptation: the continuation's params are the ones it was
    handed, bit for bit."""
    target, u0, _, _ = _setup()
    make = _kernels(target)[kernel_name]
    res1 = run(make(), target, u0, n_warmup=60, n_samples=20, seed=1)
    res2 = run(make(), target, resume=res1, n_warmup=0, n_samples=20)
    assert set(res1.final_params) == set(res2.final_params)
    for k, v in res1.final_params.items():
        np.testing.assert_array_equal(np.array(v), np.array(res2.final_params[k]))
    # and it starts exactly where the first segment stopped
    np.testing.assert_array_equal(
        np.array(res1.final_state["u"]), np.array(res2.get_chain()[0]) * 0
        + np.array(res1.final_state["u"]))


@pytest.mark.parametrize("kernel_name", ["rwm", "ensemble", "chees"])
def test_resume_recovers_the_same_posterior_as_one_long_run(kernel_name):
    """W + 2N in one go, against W + N then a resume of N."""
    target, u0, mu, cov = _setup(dim=3, n=256)
    make = _kernels(target)[kernel_name]
    W, N = 300, 200
    long_run = run(make(), target, u0, n_warmup=W, n_samples=2 * N, seed=5)
    first = run(make(), target, u0, n_warmup=W, n_samples=N, seed=5)
    second = run(make(), target, resume=first, n_warmup=0, n_samples=N)

    split = np.concatenate([first.get_chain(flat=True),
                            second.get_chain(flat=True)])
    whole = long_run.get_chain(flat=True)
    sd = np.sqrt(np.diag(cov))
    # in units of the posterior sd: MC error on the mean of ~5e4 correlated
    # draws. Generous, deliberately — the failure this guards against
    # (continuing from stale state, or re-adapting) is gross, not marginal.
    assert np.max(np.abs(split.mean(0) - whole.mean(0)) / sd) < 0.15
    np.testing.assert_allclose(split.std(0), whole.std(0), rtol=0.15)
    # the resumed segment is a continuation, not a replay
    assert not np.array_equal(first.get_chain(), second.get_chain())


@pytest.mark.parametrize("kernel_name", ["rwm", "ensemble", "chees"])
def test_a_resumed_segment_equals_the_uninterrupted_run_bit_for_bit(kernel_name):
    """The exact form of the statistical test above: W+N then resume N must
    equal W+2N bit for bit. Same keys (iters_consumed offsets them), same
    frozen parameters -- the only thing that could differ is the cache run()
    recomputes eagerly on resume against what the compiled step had left."""
    target, u0, _, _ = _setup()
    make = _kernels(target)[kernel_name]
    W, N = 40, 15
    whole = run(make(), target, u0, n_warmup=W, n_samples=2 * N, seed=7)
    first = run(make(), target, u0, n_warmup=W, n_samples=N, seed=7)
    second = run(make(), target, resume=first, n_samples=N)
    np.testing.assert_array_equal(
        np.concatenate([first.get_chain(), second.get_chain()]),
        whole.get_chain())
    np.testing.assert_array_equal(
        np.concatenate([first.get_log_prob(), second.get_log_prob()]),
        whole.get_log_prob())


def test_the_chunked_transit_likelihood_resumes_bit_for_bit_too():
    """Eager vs compiled evaluation is where a reduction could in principle
    round differently; the chunked tree reduction is the project's own
    hardest case, so it is pinned rather than inferred from the Gaussian.
    The stored log-density is asserted as well as the chain: a 1-ulp
    difference in the refreshed cache would show there long before it
    flipped an accept decision in 15 draws."""
    from anvil.targets import make_transit_target

    target = make_transit_target(n_data=20_000, seed=0).target
    u0 = mx.array(np.random.default_rng(2).standard_normal((64, target.dim))
                  .astype(np.float32) * 0.01)
    W, N = 40, 15
    whole = run(ChEESHMC(target), target, u0, n_warmup=W, n_samples=2 * N, seed=7)
    first = run(ChEESHMC(target), target, u0, n_warmup=W, n_samples=N, seed=7)
    second = run(ChEESHMC(target), target, resume=first, n_samples=N)
    np.testing.assert_array_equal(
        np.concatenate([first.get_chain(), second.get_chain()]),
        whole.get_chain())
    np.testing.assert_array_equal(
        np.concatenate([first.get_log_prob(), second.get_log_prob()]),
        whole.get_log_prob())


def test_resume_does_not_replay_the_first_segment_keys():
    """The iteration offset is the whole point: drop it and the resumed
    segment redraws the keys warmup already used."""
    from dataclasses import replace

    target, u0, _, _ = _setup()
    res1 = run(RandomWalkMetropolis(target), target, u0,
               n_warmup=40, n_samples=15, seed=2)
    kept = run(RandomWalkMetropolis(target), target,
               resume=res1, n_warmup=0, n_samples=15)
    dropped = run(RandomWalkMetropolis(target), target,
                  resume=replace(res1.resume_state(), iteration=0),
                  n_warmup=0, n_samples=15)
    assert not np.array_equal(kept.get_chain(), dropped.get_chain())
    # and the offset is exactly the iterations already consumed
    assert res1.iters_consumed == 40 + 15
    assert res1.resume_state().iteration == 55


def test_iteration_count_accumulates_across_resumes_and_thinning():
    target, u0, _, _ = _setup()
    r1 = run(RandomWalkMetropolis(target), target, u0,
             n_warmup=20, n_samples=5, thin=3, seed=4)
    assert r1.iters_consumed == 20 + 15
    r2 = run(RandomWalkMetropolis(target), target, resume=r1,
             n_warmup=0, n_samples=7)
    assert r2.iters_consumed == 35 + 7
    r3 = run(RandomWalkMetropolis(target), target, resume=r2,
             n_warmup=0, n_samples=2)
    assert r3.iters_consumed == 44
    # three consecutive segments, no two alike
    chains = [np.array(r.get_chain()[0]) for r in (r1, r2, r3)]
    assert not np.array_equal(chains[0], chains[1])
    assert not np.array_equal(chains[1], chains[2])


def test_resume_segment_statistics_describe_the_segment_only():
    target, u0, _, _ = _setup()
    r1 = run(ChEESHMC(target), target, u0, n_warmup=60, n_samples=30, seed=6)
    r2 = run(ChEESHMC(target), target, resume=r1, n_warmup=0, n_samples=10)
    assert r2.accept_fraction.shape == r1.accept_fraction.shape
    assert np.all(r2.accept_fraction <= 1.0)
    assert r2.extras["divergent_per_chain"].shape == (r1.n_chains,)
    assert r2.get_chain().shape[0] == 10
    assert r2.n_warmup == 0


# -- persistence ----------------------------------------------------------


@pytest.mark.parametrize("dense", [False, True])
def test_save_load_round_trip_is_bit_identical(tmp_path, dense):
    target, u0, _, _ = _setup(dim=3, n=256)
    res = run(ChEESHMC(target, dense=dense), target, u0,
              n_warmup=80, n_samples=10, seed=9)
    path = tmp_path / "state.npz"
    res.save_state(path)
    loaded = load_state(path)
    rs = res.resume_state()

    assert loaded.kernel == "ChEESHMC" == rs.kernel
    assert (loaded.iteration, loaded.seed, loaded.n_chains, loaded.dim) == (
        rs.iteration, rs.seed, rs.n_chains, rs.dim)
    assert loaded.kernel_ckpt == rs.kernel_ckpt
    assert set(loaded.params) == set(rs.params)
    if dense:
        # the (dim, dim) factors are exactly the warmup work being skipped
        assert loaded.params["corr"].shape == (3, 3)
        assert loaded.params["lrinv"].shape == (3, 3)
    else:
        assert "inv_mass" in loaded.params
    for k, v in rs.params.items():
        np.testing.assert_array_equal(np.array(v), np.array(loaded.params[k]))
        assert loaded.params[k].dtype == v.dtype
    for k, v in rs.state.items():
        np.testing.assert_array_equal(np.array(v), np.array(loaded.state[k]))
        assert loaded.state[k].dtype == v.dtype


@pytest.mark.parametrize("dense", [False, True])
def test_a_loaded_state_resumes_identically_to_the_in_memory_one(
        tmp_path, dense):
    target, u0, _, _ = _setup(dim=3, n=256)
    res = run(ChEESHMC(target, dense=dense), target, u0,
              n_warmup=80, n_samples=10, seed=11)
    res.save_state(tmp_path / "s.npz")
    a = run(ChEESHMC(target, dense=dense), target, resume=res,
            n_warmup=0, n_samples=12)
    b = run(ChEESHMC(target, dense=dense), target,
            resume=load_state(tmp_path / "s.npz"), n_warmup=0, n_samples=12)
    np.testing.assert_array_equal(a.get_chain(), b.get_chain())
    np.testing.assert_array_equal(a.get_log_prob(), b.get_log_prob())


def test_dense_resume_needs_no_re_derivation_of_the_preconditioner():
    """A dense run resumed by a kernel constructed with dense=False (and
    vice versa) follows the saved params, not the constructor."""
    target, u0, _, _ = _setup(dim=3, n=256)
    res = run(ChEESHMC(target, dense=True), target, u0,
              n_warmup=80, n_samples=10, seed=12)
    assert "corr" in res.final_params
    k = ChEESHMC(target, dense=False)
    out = run(k, target, resume=res, n_warmup=0, n_samples=10)
    assert k.dense is True
    np.testing.assert_array_equal(np.array(res.final_params["corr"]),
                                  np.array(out.final_params["corr"]))
    assert np.all(np.isfinite(out.get_chain()))


def test_chees_jitter_sequence_continues_rather_than_restarting():
    target, u0, _, _ = _setup()
    res = run(ChEESHMC(target), target, u0, n_warmup=30, n_samples=20, seed=13)
    assert res.kernel_ckpt["halton_iter"] == 50
    k = ChEESHMC(target)
    run(k, target, resume=res, n_warmup=0, n_samples=7)
    assert k._iter == 57


def test_ensemble_move_mixing_continues_across_a_resume():
    target, u0, _, _ = _setup()
    kern = EnsembleKernel(target, moves=[(anvil.StretchMove(), 0.5),
                                         (anvil.DEMove(), 0.5)], seed=3)
    res = run(kern, target, u0, n_warmup=10, n_samples=10, seed=14)
    assert res.kernel_ckpt["move_draws"] == 20
    k2 = EnsembleKernel(target, moves=[(anvil.StretchMove(), 0.5),
                                       (anvil.DEMove(), 0.5)], seed=3)
    run(k2, target, resume=res, n_warmup=0, n_samples=5)
    assert k2._moves_drawn == 25
    # the replayed chooser sits exactly where 25 uninterrupted draws would
    import random as _r
    ref = _r.Random(3 ^ 0x5EED)
    for _ in range(25):
        ref.choices(range(2), [0.5, 0.5])
    assert k2._chooser.getstate() == ref.getstate()


# -- refusals -------------------------------------------------------------


def test_resume_with_warmup_raises():
    target, u0, _, _ = _setup()
    res = run(RandomWalkMetropolis(target), target, u0,
              n_warmup=10, n_samples=5, seed=1)
    with pytest.raises(ValueError, match="n_warmup must be 0"):
        run(RandomWalkMetropolis(target), target, resume=res,
            n_warmup=50, n_samples=5)


def test_mismatched_dim_raises():
    target, u0, _, _ = _setup(dim=4)
    res = run(RandomWalkMetropolis(target), target, u0,
              n_warmup=10, n_samples=5, seed=1)
    other, _, _ = correlated_gaussian(6, rho=0.5, seed=0)
    with pytest.raises(ValueError, match=r"dim=4.*dim=6"):
        run(RandomWalkMetropolis(other), other, resume=res,
            n_warmup=0, n_samples=5)


def test_mismatched_n_chains_raises():
    target, u0, _, _ = _setup(n=128)
    res = run(RandomWalkMetropolis(target), target, u0,
              n_warmup=10, n_samples=5, seed=1)
    wrong = mx.zeros((64, 4))
    with pytest.raises(ValueError, match="configuration mismatch"):
        run(RandomWalkMetropolis(target), target, wrong, resume=res,
            n_warmup=0, n_samples=5)


def test_mismatched_kernel_raises():
    target, u0, _, _ = _setup()
    res = run(ChEESHMC(target), target, u0, n_warmup=20, n_samples=5, seed=1)
    with pytest.raises(ValueError, match="written by ChEESHMC"):
        run(RandomWalkMetropolis(target), target, resume=res,
            n_warmup=0, n_samples=5)


def test_missing_u0_and_no_resume_raises():
    target, _, _, _ = _setup()
    with pytest.raises(TypeError, match="needs initial positions"):
        run(RandomWalkMetropolis(target), target, n_warmup=5, n_samples=5)


def test_resume_rejects_the_wrong_kind_of_object():
    target, u0, _, _ = _setup()
    with pytest.raises(TypeError, match="Results or a ResumeState"):
        run(RandomWalkMetropolis(target), target, resume={"u": u0},
            n_warmup=0, n_samples=5)


def test_unreadable_format_is_reported_not_crashed(tmp_path):
    p = tmp_path / "bad.npz"
    np.savez(p, format=np.array(99))
    with pytest.raises(ValueError, match=re.escape("state format 99")):
        load_state(p)


def test_a_corrupt_state_is_caught_on_load(tmp_path):
    target, u0, _, _ = _setup(n=128, dim=4)
    res = run(RandomWalkMetropolis(target), target, u0,
              n_warmup=10, n_samples=5, seed=1)
    res.save_state(tmp_path / "s.npz")
    with np.load(tmp_path / "s.npz") as z:
        d = dict(z)
    d["n_chains"] = np.array(999)          # as if written by another config
    np.savez(tmp_path / "s2.npz", **d)
    with pytest.raises(ValueError, match="inconsistent"):
        load_state(tmp_path / "s2.npz")


def test_seed_defaults_to_the_saved_one_but_can_be_overridden():
    target, u0, _, _ = _setup()
    res = run(RandomWalkMetropolis(target), target, u0,
              n_warmup=20, n_samples=5, seed=17)
    same = run(RandomWalkMetropolis(target), target, resume=res,
               n_warmup=0, n_samples=5)
    assert same.seed == 17
    other = run(RandomWalkMetropolis(target), target, resume=res,
                n_warmup=0, n_samples=5, seed=18)
    assert other.seed == 18
    assert not np.array_equal(same.get_chain(), other.get_chain())


def test_the_proposed_spelling_works_without_naming_n_warmup():
    """`run(kernel, target, resume=res, n_samples=N)` — the fresh-run
    default of 500 warmup iterations is not a request to re-adapt."""
    target, u0, _, _ = _setup()
    res1 = run(RandomWalkMetropolis(target), target, u0,
               n_warmup=20, n_samples=10, seed=19)
    res2 = run(RandomWalkMetropolis(target), target, resume=res1, n_samples=10)
    assert res2.n_warmup == 0
    assert res2.iters_consumed == 40
    np.testing.assert_array_equal(np.array(res1.final_params["step_scale"]),
                                  np.array(res2.final_params["step_scale"]))


# -- moving the chains between segments -----------------------------------
#
# The point of with_positions is that a caller composing its own exact move
# (a Gibbs sweep over a conditional anvil cannot see, a mode hop) does not
# have to know which per-chain quantities a kernel caches. Writing u by hand
# and leaving the cache behind is a silently wrong answer, so these tests
# check both that the refreshed state is right and that the stale one is
# visibly different.


def _moved(rng, n=128, dim=4, scale=1.0):
    return mx.array(rng.standard_normal((n, dim)).astype(np.float32) * scale)


@pytest.mark.parametrize("kernel_name", ["rwm", "ensemble", "chees"])
def test_with_positions_rebuilds_exactly_what_init_would_have(kernel_name):
    target, u0, _, _ = _setup()
    make = _kernels(target)[kernel_name]
    res = run(make(), target, u0, n_warmup=40, n_samples=10, seed=21)
    rs = res.resume_state()
    new = _moved(np.random.default_rng(99))

    moved = rs.with_positions(new, target)
    reference = make().init(mx.random.key(0), new, target)
    assert set(moved.state) == set(reference)
    for k, v in reference.items():
        np.testing.assert_array_equal(np.array(moved.state[k]), np.array(v))


@pytest.mark.parametrize("kernel_name", ["rwm", "ensemble", "chees"])
def test_resuming_a_moved_state_matches_a_run_started_there(kernel_name):
    """Same frozen params, same key-stream position, chains at `new`: the
    moved state and a state built from scratch at `new` must sample
    identically, bit for bit."""
    from dataclasses import replace

    target, u0, _, _ = _setup()
    make = _kernels(target)[kernel_name]
    res = run(make(), target, u0, n_warmup=40, n_samples=10, seed=22)
    rs = res.resume_state()
    new = _moved(np.random.default_rng(100))

    a = run(make(), target, resume=rs.with_positions(new, target), n_samples=12)
    built = replace(rs, state=make().init(mx.random.key(0), new, target))
    b = run(make(), target, resume=built, n_samples=12)
    np.testing.assert_array_equal(a.get_chain(), b.get_chain())
    np.testing.assert_array_equal(a.get_log_prob(), b.get_log_prob())


def test_a_hand_moved_state_is_repaired_rather_than_sampled_stale():
    """Setting u and leaving log_prob/grad describing where the chain used to
    be is the brittle way to move chains. It used to sample a different
    answer, silently; since run() recomputes the cached log-density on
    resume (see tests/test_stale_target.py) it is repaired instead, and
    matches the with_positions path exactly.

    with_positions is still the contract: it validates the move, and it is
    what a kernel whose refresh() cannot rebuild its state needs."""
    from dataclasses import replace

    target, u0, _, _ = _setup()
    res = run(ChEESHMC(target), target, u0, n_warmup=40, n_samples=10, seed=23)
    rs = res.resume_state()
    new = _moved(np.random.default_rng(101), scale=2.0)

    refreshed = rs.with_positions(new, target)
    stale = replace(rs, state={**rs.state, "u": new})      # the brittle version

    # the hand-written cache really is wrong, by far more than the ~1-unit
    # scale of a Metropolis accept decision for a typical chain -- which is
    # what makes repairing it at run start worth the one extra evaluation
    gap = np.abs(np.array(stale.state["log_prob"])
                 - np.array(refreshed.state["log_prob"]))
    assert np.median(gap) > 1.0, np.median(gap)

    a = run(ChEESHMC(target), target, resume=refreshed, n_samples=10)
    b = run(ChEESHMC(target), target, resume=stale, n_samples=10)
    np.testing.assert_array_equal(a.get_chain(), b.get_chain())
    np.testing.assert_array_equal(a.get_log_prob(), b.get_log_prob())


def test_a_kernel_that_caches_more_must_say_how_to_refresh_it():
    from dataclasses import replace

    target, u0, _, _ = _setup()
    res = run(ChEESHMC(target), target, u0, n_warmup=20, n_samples=5, seed=24)
    rs = replace(res.resume_state(),
                 state={**res.final_state, "hessian_est": res.final_state["u"]})
    with pytest.raises(NotImplementedError, match=r"caches \['hessian_est'\]"):
        rs.with_positions(_moved(np.random.default_rng(1)), target)


def test_with_positions_leaves_the_key_stream_and_the_original_alone():
    target, u0, _, _ = _setup()
    res = run(ChEESHMC(target), target, u0, n_warmup=40, n_samples=10, seed=25)
    rs = res.resume_state()
    before = np.array(rs.state["u"])
    moved = rs.with_positions(_moved(np.random.default_rng(2)), target)

    assert moved is not rs
    assert (moved.iteration, moved.seed) == (rs.iteration, rs.seed)
    assert (moved.kernel, moved.kernel_ckpt) == (rs.kernel, rs.kernel_ckpt)
    for k, v in rs.params.items():
        np.testing.assert_array_equal(np.array(v), np.array(moved.params[k]))
    np.testing.assert_array_equal(before, np.array(rs.state["u"]))   # not in place
    # and the moved state is itself persistable
    assert moved.n_chains == rs.n_chains and moved.dim == rs.dim


def test_with_positions_accepts_numpy_and_casts_to_float32():
    target, u0, _, _ = _setup()
    res = run(ChEESHMC(target), target, u0, n_warmup=20, n_samples=5, seed=26)
    new = np.random.default_rng(3).standard_normal((128, 4))      # float64
    moved = res.resume_state().with_positions(new, target)
    assert moved.state["u"].dtype == mx.float32
    np.testing.assert_allclose(np.array(moved.state["u"]), new, rtol=1e-6)


def test_with_positions_rejects_a_wrong_shape_or_non_finite_move():
    target, u0, _, _ = _setup()
    rs = run(ChEESHMC(target), target, u0,
             n_warmup=20, n_samples=5, seed=27).resume_state()
    with pytest.raises(ValueError, match=r"shape \(128, 4\)"):
        rs.with_positions(mx.zeros((64, 4)), target)
    bad = np.zeros((128, 4), np.float32)
    bad[3] = np.inf
    with pytest.raises(ValueError, match="not all finite"):
        rs.with_positions(bad, target)


def test_with_positions_refuses_a_move_outside_the_support():
    """A chain whose log-probability is -inf can only escape by luck, so
    moving one there is reported rather than accepted."""
    import mlx.core as _mx

    from anvil.logdensity import FunctionLogDensity

    def boxed(u):
        inside = _mx.all(_mx.abs(u) < 10.0, axis=-1)
        return _mx.where(inside, -0.5 * _mx.sum(u * u, axis=-1), -_mx.inf)

    target = FunctionLogDensity(boxed, dim=2)
    u0 = mx.array(np.random.default_rng(4).standard_normal((64, 2))
                  .astype(np.float32))
    rs = run(ChEESHMC(target), target, u0,
             n_warmup=30, n_samples=5, seed=28).resume_state()
    out = np.zeros((64, 2), np.float32)
    out[:5] = 50.0
    with pytest.raises(ValueError, match="not finite at the new positions"):
        rs.with_positions(out, target)


def test_with_positions_needs_to_know_the_kernel():
    from dataclasses import replace

    target, u0, _, _ = _setup()
    rs = run(ChEESHMC(target), target, u0,
             n_warmup=20, n_samples=5, seed=29).resume_state()
    unknown = replace(rs, kernel="SomeKernelFromAnotherPackage")
    with pytest.raises(ValueError, match="no imported Kernel subclass"):
        unknown.with_positions(_moved(np.random.default_rng(5)), target)
    # ... but an instance settles it
    moved = unknown.with_positions(_moved(np.random.default_rng(5)), target,
                                   kernel=ChEESHMC(target))
    assert sorted(moved.state) == ["grad", "log_prob", "u"]


def _bimodal(sep=6.0, w=0.7):
    """Two well-separated Gaussians in dim 0 (weights w, 1-w) and a
    standard normal in dim 1. ChEES cannot cross a gap of 6 sd."""
    import math

    from anvil.logdensity import FunctionLogDensity

    def lp(u):
        x0, x1 = u[:, 0], u[:, 1]
        return (mx.logaddexp(math.log(w) - 0.5 * (x0 - sep) ** 2,
                             math.log1p(-w) - 0.5 * (x0 + sep) ** 2)
                - 0.5 * x1 * x1)

    return FunctionLogDensity(lp, dim=2)


def _grid_gibbs(u, target, rng, lo=-12.0, hi=12.0, n=481):
    """Redraw dim 0 from its conditional on a grid, with an independence
    Metropolis-Hastings correction — the move the seam exists for, written
    the way a caller would write it (host-side, its own RNG)."""
    u = np.array(u, dtype=np.float32)
    n_ch = u.shape[0]
    grid = np.linspace(lo, hi, n, dtype=np.float32)

    cand = np.repeat(u[:, None, :], n, axis=1)
    cand[:, :, 0] = grid
    lp_grid = np.array(target.log_prob(
        mx.array(cand.reshape(-1, u.shape[1])))).reshape(n_ch, n)
    w = np.exp(lp_grid - lp_grid.max(1, keepdims=True))
    w /= w.sum(1, keepdims=True)

    idx = (rng.random((n_ch, 1)) < np.cumsum(w, 1)).argmax(1)
    prop = u.copy()
    prop[:, 0] = grid[idx]
    cur = np.clip(np.rint((u[:, 0] - lo) / (hi - lo) * (n - 1)).astype(int),
                  0, n - 1)
    rows = np.arange(n_ch)
    # q is the discretized conditional, so the proposal is not symmetric
    log_ratio = ((lp_grid[rows, idx] - lp_grid[rows, cur])
                 + (np.log(w[rows, cur]) - np.log(w[rows, idx])))
    take = np.log(rng.random(n_ch)) < log_ratio
    return mx.array(np.where(take[:, None], prop, u))


def test_an_exact_move_between_segments_recovers_mode_weights():
    """The end-to-end claim: ChEES alternated with an exact move through
    with_positions is still one Markov chain, and samples the mixture ChEES
    alone cannot. Weights 0.7/0.3, all chains started in the heavy mode."""
    target = _bimodal()
    rng = np.random.default_rng(7)
    u0 = mx.array(np.stack([6.0 + 0.1 * rng.standard_normal(128),
                            rng.standard_normal(128)], axis=1)
                  .astype(np.float32))

    # control: ChEES alone, same budget. Every chain stays where it started.
    alone = run(ChEESHMC(target), target, u0,
                n_warmup=300, n_samples=1000, seed=31)
    assert (alone.get_chain()[..., 0] > 0).mean() > 0.999

    res = run(ChEESHMC(target), target, u0,
              n_warmup=300, n_samples=200, seed=31)
    segments = []
    for _ in range(5):
        rs = res.resume_state()
        moved = rs.with_positions(_grid_gibbs(rs.state["u"], target, rng),
                                  target)
        res = run(ChEESHMC(target), target, resume=moved, n_samples=200)
        segments.append(res.get_chain())

    draws = np.concatenate(segments)
    heavy = (draws[..., 0] > 0).mean()
    assert abs(heavy - 0.7) < 0.06, heavy
    # both modes are resolved, not merely visited
    for sign, centre in ((+1, 6.0), (-1, -6.0)):
        x = draws[..., 0][np.sign(draws[..., 0]) == sign]
        assert abs(x.mean() - centre) < 0.15
        assert abs(x.std() - 1.0) < 0.15
    # the nuisance dimension is unharmed by the move
    assert abs(draws[..., 1].mean()) < 0.05
    assert abs(draws[..., 1].std() - 1.0) < 0.05
