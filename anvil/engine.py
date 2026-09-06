"""The shared driver loop: warmup (adapting) then sampling (frozen).

Kernels supply pure ``step`` functions; the engine owns iteration, the
``mx.eval`` cadence that bounds lazy-graph growth, storage handoff, and
acceptance bookkeeping. One loop serves every kernel.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import mlx.core as mx
import numpy as np

from .kernels.base import Kernel
from .logdensity import LogDensity
from .rng import KeyStream
from .state import ChainState
from .storage import MemoryBackend


@dataclass
class Results:
    backend: MemoryBackend
    n_chains: int
    dim: int
    n_warmup: int
    thin: int
    accept_fraction: np.ndarray  # (n_chains,) mean over sampling phase
    final_state: ChainState
    final_params: dict[str, mx.array]
    extras: dict[str, Any] = field(default_factory=dict)

    def get_chain(self, discard: int = 0, thin: int = 1, flat: bool = False):
        return self.backend.get_chain(discard, thin, flat)

    def get_log_prob(self, discard: int = 0, thin: int = 1, flat: bool = False):
        return self.backend.get_log_prob(discard, thin, flat)


def run(
    kernel: Kernel,
    target: LogDensity,
    u0: mx.array,
    *,
    n_warmup: int = 500,
    n_samples: int = 1000,
    thin: int = 1,
    seed: int = 0,
    storage: MemoryBackend | None = None,
    compile_step: bool = True,
    progress: bool = False,
    reanchor_every: int = 0,
    archive=None,
) -> Results:
    """Run ``kernel`` on ``target`` from initial positions ``u0``
    ((n_chains, dim), float32). Records ``n_samples`` states per chain, one
    every ``thin`` post-warmup iterations.

    ``reanchor_every`` > 0 recomputes the cached log_prob of the current
    states through the target's float64 path every that many iterations
    (requires ``target.log_prob_hi``), so fp32 rounding drift cannot
    accumulate in the chain's accept/reject bookkeeping.

    ``archive`` (a :class:`~anvil.surrogate.TrainingArchive`) records
    every stored (u, log_prob) frame as future emulator training data —
    this happens on already-evaluated host copies, off the hot path."""
    u0 = u0.astype(mx.float32) if u0.dtype != mx.float32 else u0
    n_chains, dim = u0.shape
    keys = KeyStream(seed)

    state = kernel.init(keys.init_key(), u0, target)
    adapt_state = kernel.init_adapt(state)
    params = kernel.make_params(adapt_state, warmup=True)
    if compile_step and not kernel.self_compiled:
        step = mx.compile(kernel.step)
    else:
        step = kernel.step

    if reanchor_every and not hasattr(target, "log_prob_hi"):
        raise ValueError(
            "reanchor_every > 0 requires a target with a float64 path "
            "(log_prob_hi); pass model_log_prob_hi to TransformedLogDensity"
        )

    def reanchor(st: ChainState) -> ChainState:
        lp = target.log_prob_hi(st["u"]).astype(mx.float32, stream=mx.cpu)
        return {**st, "log_prob": lp}

    # -- warmup: adapt every iteration ------------------------------------
    for t in range(n_warmup):
        state, info = step(keys.key(t), state, params)
        adapt_state = kernel.adapt(adapt_state, state, info, t + 1)
        params = kernel.make_params(adapt_state, warmup=True)
        if reanchor_every and (t + 1) % reanchor_every == 0:
            state = reanchor(state)
        mx.eval(*state.values(), *params.values())
        if progress and (t + 1) % max(1, n_warmup // 10) == 0:
            print(f"warmup {t + 1}/{n_warmup}")

    params = kernel.make_params(adapt_state, warmup=False)
    mx.eval(*params.values())

    # -- sampling: frozen params ------------------------------------------
    if storage is None:
        storage = MemoryBackend()
    storage.reserve(n_samples, n_chains, dim)
    accept_sum = mx.zeros((n_chains,))
    divergent_sum = mx.zeros((n_chains,))
    total_iters = n_samples * thin
    for t in range(total_iters):
        state, info = step(keys.key(n_warmup + t), state, params)
        accept_sum = accept_sum + info["accept_prob"]
        if "diverged" in info:
            divergent_sum = divergent_sum + info["diverged"]
        if reanchor_every and (t + 1) % reanchor_every == 0:
            state = reanchor(state)
        # evaluate every iteration to bound lazy-graph growth
        mx.eval(*state.values(), accept_sum, divergent_sum)
        if (t + 1) % thin == 0:
            # arrays are already evaluated; np.array() just copies out
            u_np, lp_np = np.array(state["u"]), np.array(state["log_prob"])
            storage.append(u_np, lp_np)
            if archive is not None:
                archive.record(u_np, lp_np)
        if progress and (t + 1) % max(1, total_iters // 10) == 0:
            print(f"sample {t + 1}/{total_iters}")

    return Results(
        backend=storage,
        n_chains=n_chains,
        dim=dim,
        n_warmup=n_warmup,
        thin=thin,
        accept_fraction=np.array(accept_sum) / max(1, total_iters),
        final_state=state,
        final_params=params,
        extras={"n_divergent": int(np.array(divergent_sum).sum())},
    )
