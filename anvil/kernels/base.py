"""Kernel protocol: every sampler is init/step/adapt over batched state.

``step`` must be pure and side-effect free (compilable): all tunable
quantities enter through ``params`` as mx arrays so adaptation can change
them without retracing, and the return value is a pair of plain dicts
(``mx.compile`` handles dict pytrees but not custom classes). The engine owns
the iteration loop; kernels never loop over iterations themselves.

Step info convention — dict with per-chain arrays of shape (n_chains,):
    "accept_prob"  Metropolis acceptance probability of the proposal
    "accepted"     whether the proposal was taken (bool)
plus kernel-specific extras (e.g. "diverged" for HMC).
"""

from __future__ import annotations

import warnings
from typing import Any

import mlx.core as mx

from ..logdensity import LogDensity
from ..state import ChainState

StepInfo = dict[str, mx.array]


class Kernel:
    #: kernels that require log_prob gradients set this
    needs_grad: bool = False
    #: kernels that manage their own mx.compile (e.g. host-side move
    #: mixing) set this so the engine does not wrap step() again
    self_compiled: bool = False

    def init(self, key: mx.array, u0: mx.array, target: LogDensity) -> ChainState:
        """Build the initial chain state (caches log_prob etc.)."""
        raise NotImplementedError

    def step(
        self, key: mx.array, state: ChainState, params: dict[str, mx.array]
    ) -> tuple[ChainState, StepInfo]:
        """Advance every chain one step. Pure; compilable. Kernels needing
        several random draws split ``key`` internally with mx.random.split."""
        raise NotImplementedError

    def init_adapt(self, state: ChainState) -> Any:
        """Initial adaptation state (any small pytree); None if non-adaptive."""
        return None

    def adapt(
        self, adapt_state: Any, state: ChainState, info: StepInfo, t: int
    ) -> Any:
        """Warmup-only update of the adaptation state after iteration ``t``
        (1-based)."""
        return adapt_state

    def make_params(self, adapt_state: Any, warmup: bool) -> dict[str, mx.array]:
        """Extract the params dict ``step`` consumes. With ``warmup=False``
        return the frozen (iterate-averaged) values."""
        return {}

    # -- continuation (engine.run(..., resume=...)) ------------------------

    def attach(self, target: LogDensity, state: ChainState,
               params: dict[str, mx.array]) -> None:
        """Prepare to continue an existing run: whatever ``init`` sets up
        besides the chain state itself. The engine calls this instead of
        ``init`` on a resume, so a kernel that caches ``target`` (all of
        them do) must not rely on ``init`` having run."""
        self.target = target

    def retrace(self) -> None:
        """Discard compiled graphs so the next step re-reads the target.

        ``mx.compile`` freezes everything a traced function reads that is
        *not* an argument: arrays held on the target, Python floats, closure
        variables. A kernel that compiles once at construction therefore goes
        on sampling the target **as it was when it was first traced** --
        which, if the target changed in between, is a converged,
        healthy-looking run of the wrong distribution rather than a crash.
        The engine calls this at the start of every :func:`anvil.run`, so a
        target that changes between runs is handled by default.

        Kernels the engine compiles itself (``self_compiled = False``) need
        nothing here: ``run`` wraps ``step`` afresh each call. Kernels that
        compile their own graphs must override this, and are warned if they
        do not."""
        if self.self_compiled and type(self).retrace is Kernel.retrace:
            warnings.warn(
                f"{type(self).__name__} sets self_compiled=True but does not "
                f"override retrace(), so whatever its compiled graphs read "
                f"from the target is frozen at the first trace. If the target "
                f"can change between runs, this samples the old one silently.",
                stacklevel=2,
            )

    @classmethod
    def refresh(cls, state: ChainState, u: mx.array,
                target: LogDensity) -> ChainState:
        """Rebuild the chain state at new positions ``u``: every cached
        per-chain quantity recomputed there, nothing carried over.

        This is what makes it safe to *move* chains between segments (a
        Gibbs sweep, a mode-hopping proposal, a reparameterization) without
        the caller knowing which quantities a kernel caches. A classmethod
        because the answer is a property of the kernel's state layout, not
        of a tuned instance -- which is what lets
        :meth:`anvil.ResumeState.with_positions` reach it from a saved
        state by name alone.

        The default covers the ``{"u", "log_prob"}`` layout and *refuses*
        anything else: a kernel that caches more must say how to recompute
        it, because the alternative is resuming from a stale cache, which
        is a wrong answer rather than a crash."""
        return cls._check_refresh(state, {"u": u, "log_prob": target.log_prob(u)})

    @classmethod
    def _check_refresh(cls, state: ChainState, fresh: ChainState) -> ChainState:
        stale = sorted(set(state) - set(fresh))
        if stale:
            raise NotImplementedError(
                f"{cls.__name__} caches {stale} in its chain state, which "
                f"refresh() does not recompute. Override refresh() (a "
                f"classmethod) to rebuild it at the new positions -- copying "
                f"it across a move would resume from a stale cache, which is "
                f"a silently wrong answer, not a crash.")
        return fresh

    def checkpoint(self) -> dict[str, float]:
        """Host-side counters to carry across a resume -- a quasi-random
        jitter index, a move-mixing draw count. Scalars only; anything
        array-shaped belongs in the chain state or the params."""
        return {}

    def restore(self, ckpt: dict[str, float]) -> None:
        """Inverse of :meth:`checkpoint`. Missing keys mean an older or
        different kernel wrote the state: start from the default."""
        return None
