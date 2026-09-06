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
