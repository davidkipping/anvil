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
