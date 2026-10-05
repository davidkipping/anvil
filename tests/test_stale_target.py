"""A target that changes between runs must not be sampled as it used to be.

``mx.compile`` freezes everything a traced function reads that is not an
argument, so a kernel that compiles at construction keeps returning the
log-density of the target it first traced. The failure is silent in the way
that matters: a small change to the target still accepts normally and
converges, to the *old* posterior, with healthy R-hat and ESS. These tests
drive the public ``run(..., resume=...)`` path with the same kernel object,
which is how a Gibbs-within-HMC scheme uses anvil.

Reported by SquishierPlanet, whose limb-darkening block Gibbs step had been
conditioning on its initial value for a whole fit.
"""

import warnings

import mlx.core as mx
import numpy as np
import pytest

import anvil
from anvil.kernels.base import Kernel
from anvil.kernels.chees import ChEESHMC
from anvil.kernels.ensemble import EnsembleKernel
from anvil.kernels.metropolis import RandomWalkMetropolis

N_CHAINS = 512


class ShiftedGaussian(anvil.LogDensity):
    """Unit Gaussian whose mean lives on the target, not in the arguments --
    the shape of any conditional the sampler cannot see."""

    dim = 2

    def __init__(self, mu=0.0, per_chain=False, n_chains=N_CHAINS):
        self.per_chain = per_chain
        self.n_chains = n_chains
        self.set_mu(mu)

    def set_mu(self, mu):
        self.mu = (mx.full((self.n_chains, 1), float(mu)) if self.per_chain
                   else mx.array(mu, dtype=mx.float32))

    def log_prob(self, u):
        d = u - self.mu
        return -0.5 * mx.sum(d * d, axis=-1)


def _u0(seed, shift=0.0, n=N_CHAINS):
    rng = np.random.default_rng(seed)
    return mx.array((shift + rng.normal(size=(n, 2))).astype(np.float32))


def _segment(kernel, target, res, shift, n_samples=400):
    """Change the target, move the chains, resume -- the Gibbs pattern."""
    target.set_mu(shift)
    rs = res.resume_state().with_positions(_u0(99, shift), target)
    return anvil.run(kernel, target, resume=rs, n_warmup=0, n_samples=n_samples)


@pytest.mark.parametrize("shift", [5.0, 0.5])
@pytest.mark.parametrize("kind", ["chees", "ensemble"])
def test_a_changed_target_is_sampled_as_it_now_is(shift, kind):
    """shift=5 used to freeze the chains (0.3% moving); shift=0.5 used to
    sample the old target with every diagnostic looking healthy."""
    target = ShiftedGaussian(0.0)
    kernel = (ChEESHMC(target) if kind == "chees"
              else EnsembleKernel(target, seed=3))
    res = anvil.run(kernel, target, _u0(0), n_warmup=300, n_samples=200, seed=1)
    chain = _segment(kernel, target, res, shift).get_chain()

    assert abs(chain.mean() - shift) < 0.05, chain.mean()
    assert abs(chain.std() - 1.0) < 0.06, chain.std()
    # and the chains are actually moving, not parked where with_positions put
    # them: a frozen ensemble also has the right mean, by construction
    moving = np.mean(np.any(chain[1:] != chain[:-1], axis=-1))
    assert moving > 0.3, moving


def test_a_per_chain_block_held_on_the_target():
    """The reported case: a Gibbs block held per chain, (n_chains, 1)."""
    target = ShiftedGaussian(0.0, per_chain=True)
    kernel = ChEESHMC(target)
    res = anvil.run(kernel, target, _u0(0), n_warmup=300, n_samples=200, seed=1)
    chain = _segment(kernel, target, res, 2.0).get_chain()
    assert abs(chain.mean() - 2.0) < 0.05, chain.mean()
    assert abs(chain.std() - 1.0) < 0.06, chain.std()


def test_several_segments_track_the_target_each_time():
    """The actual usage is a loop, so check it does not go stale on segment
    three after being right on segment one."""
    target = ShiftedGaussian(0.0)
    kernel = ChEESHMC(target)
    res = anvil.run(kernel, target, _u0(0), n_warmup=300, n_samples=100, seed=1)
    for shift in (1.0, -2.0, 4.0, 0.25):
        res = _segment(kernel, target, res, shift, n_samples=200)
        got = res.get_chain().mean()
        assert abs(got - shift) < 0.06, (shift, got)


def test_a_fresh_run_with_a_reused_kernel_object_is_not_stale():
    """Not only resumes: init() does not retrace either."""
    target = ShiftedGaussian(0.0)
    kernel = ChEESHMC(target)
    anvil.run(kernel, target, _u0(0), n_warmup=200, n_samples=50, seed=1)
    target.set_mu(3.0)
    chain = anvil.run(kernel, target, _u0(1, 3.0),
                      n_warmup=200, n_samples=200, seed=2).get_chain()
    assert abs(chain.mean() - 3.0) < 0.06, chain.mean()


def test_rwm_was_never_exposed():
    """Kernels the engine compiles (self_compiled=False) get a fresh
    mx.compile per run, so they were already safe. Worth pinning: it is the
    reason the fix lives in the kernels that compile themselves."""
    assert RandomWalkMetropolis.self_compiled is False
    assert EnsembleKernel.self_compiled is True and ChEESHMC.self_compiled is True
    target = ShiftedGaussian(0.0)
    kernel = RandomWalkMetropolis(target)
    res = anvil.run(kernel, target, _u0(0), n_warmup=300, n_samples=200, seed=1)
    chain = _segment(kernel, target, res, 0.5).get_chain()
    assert abs(chain.mean() - 0.5) < 0.06, chain.mean()


@pytest.mark.parametrize("kind", ["chees", "ensemble", "rwm"])
def test_retracing_does_not_perturb_an_unchanged_target(kind):
    """The retrace must cost nothing in results: resuming with the same
    kernel object must match resuming with a freshly built one, bit for bit,
    including the frozen parameters and the key stream."""
    def make(t):
        return {"chees": lambda: ChEESHMC(t),
                "ensemble": lambda: EnsembleKernel(t, seed=3),
                "rwm": lambda: RandomWalkMetropolis(t)}[kind]()

    target = ShiftedGaussian(0.0)
    kernel = make(target)
    res = anvil.run(kernel, target, _u0(0), n_warmup=200, n_samples=50, seed=1)
    rs = res.resume_state()
    reused = anvil.run(kernel, target, resume=rs, n_warmup=0, n_samples=60)
    rebuilt = anvil.run(make(target), target, resume=rs, n_warmup=0, n_samples=60)

    np.testing.assert_array_equal(reused.get_chain(), rebuilt.get_chain())
    np.testing.assert_array_equal(reused.get_log_prob(), rebuilt.get_log_prob())
    np.testing.assert_array_equal(reused.accept_fraction, rebuilt.accept_fraction)
    assert reused.iters_consumed == rebuilt.iters_consumed
    for k, v in res.final_params.items():
        np.testing.assert_array_equal(np.array(v),
                                      np.array(reused.final_params[k]))


def test_a_self_compiled_kernel_that_forgets_to_retrace_is_warned():
    """Third-party kernels are the remaining exposure, so make it loud."""
    class Forgetful(ChEESHMC):
        retrace = Kernel.retrace          # inherit the base no-op

    target = ShiftedGaussian(0.0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        anvil.run(Forgetful(target), target, _u0(0), n_warmup=10, n_samples=5,
                  seed=1)
    assert any("does not override retrace" in str(w.message) for w in caught)


@pytest.mark.xfail(
    strict=True,
    reason="Retracing happens once per run, so a target mutated DURING a run "
           "(from a callback) is not picked up; reanchor_every does not rescue "
           "it either. This asserts the desired behaviour so the day it starts "
           "working is loud (XPASS strict) and the docs get updated, without "
           "pinning the limitation as if it were correct.")
def test_a_mid_run_change_would_be_tracked():
    target = ShiftedGaussian(0.0)
    kernel = ChEESHMC(target)
    res = anvil.run(kernel, target, _u0(0), n_warmup=300, n_samples=100, seed=1)
    chain = anvil.run(kernel, target, resume=res.resume_state(), n_warmup=0,
                      n_samples=300,
                      callback=lambda *a: target.set_mu(5.0)).get_chain()
    # the callback first fires at iteration 30 of 300 (the progress cadence),
    # so judge the tail: a sampler that tracked the change would sit at 5
    # there, while the whole-run mean could never reach it
    assert abs(chain[-150:].mean() - 5.0) < 0.1, chain[-150:].mean()
