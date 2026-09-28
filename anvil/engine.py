"""The shared driver loop: warmup (adapting) then sampling (frozen).

Kernels supply pure ``step`` functions; the engine owns iteration, the
``mx.eval`` cadence that bounds lazy-graph growth, storage handoff, and
acceptance bookkeeping. One loop serves every kernel.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import mlx.core as mx
import numpy as np

from .kernels.base import Kernel
from .logdensity import LogDensity
from .rng import KeyStream
from .state import ChainState
from .storage import MemoryBackend


#: iterations timed before deciding whether to pipeline, and the mean
#: per-iteration cost below which pipelining is worth its extra memory.
#: Below ~1 ms the blocking mx.eval round-trip (~167 us floor) dominates;
#: above ~2 ms it is noise and the in-flight graphs are not worth holding.
_PIPELINE_PROBE = 8
_PIPELINE_THRESHOLD_S = 1e-3
_PIPELINE_DEPTH = 2


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
    progress: bool | int = False,
    reanchor_every: int = 0,
    archive=None,
    pipeline: int | str = "auto",
) -> Results:
    """Run ``kernel`` on ``target`` from initial positions ``u0``
    ((n_chains, dim), float32). Records ``n_samples`` states per chain, one
    every ``thin`` post-warmup iterations.

    ``reanchor_every`` > 0 recomputes the cached log_prob of the current
    states through the target's float64 path every that many iterations
    (requires ``target.log_prob_hi``). Note this is insurance for
    log-densities that can go *stale* — a surrogate retrained mid-run, or
    any non-deterministic evaluation — NOT a remedy for float32 rounding:
    cached values are always fresh evaluations of a deterministic
    function, so rounding cannot accumulate, and re-anchoring does not
    change the distribution being sampled. It is expensive (the float64
    path can cost ~1000x the float32 one); default off.

    ``archive`` (a :class:`~anvil.surrogate.TrainingArchive`) records
    every stored (u, log_prob) frame as future emulator training data —
    this happens on already-evaluated host copies, off the hot path.

    ``progress``: False for silence; True for ticks every 10% of each
    phase; an int N for a tick every N iterations. Ticks are flushed
    (safe to ``tail -f`` through a redirected log) and report rate, ETA,
    mean acceptance, and the running divergence count.

    ``pipeline``: depth of the sampling-phase evaluation pipeline. Each
    iteration normally ends in a blocking ``mx.eval``, a GPU round-trip
    with a ~167 us floor regardless of the work it waits on. Letting
    ``pipeline`` iterations stay in flight -- and copying stored frames out
    that many steps behind, by which time they have landed -- removes it:
    measured 1.9-2.5x when a target's GPU work per iteration is a few
    hundred microseconds, and 1.0x once it reaches milliseconds, where the
    barrier is already noise. Peak memory was unchanged at every depth on
    the targets measured, so the default "auto" (time the first few
    iterations, pipeline only if it will pay) is a conservative default
    rather than a necessary one; pass an int to override. Results are
    bit-identical at any depth -- only the timing of evaluation changes,
    which the engine tests assert directly. Sampling only: warmup keeps its
    barrier because adaptation reads freshly-computed parameters on the
    host, and pipelining it would mean adapting from stale ones."""
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

    def _tick_every(n_total):
        if progress is True:
            return max(1, n_total // 10)
        return max(1, int(progress))

    def _fmt_eta(seconds):
        m, s = divmod(int(seconds), 60)
        h, m = divmod(m, 60)
        return f"{h:d}:{m:02d}:{s:02d}"

    # -- warmup: adapt every iteration ------------------------------------
    t_phase = time.perf_counter()
    for t in range(n_warmup):
        state, info = step(keys.key(t), state, params)
        adapt_state = kernel.adapt(adapt_state, state, info, t + 1)
        params = kernel.make_params(adapt_state, warmup=True)
        if reanchor_every and (t + 1) % reanchor_every == 0:
            state = reanchor(state)
        mx.eval(*state.values(), *params.values())
        if progress and (t + 1) % _tick_every(n_warmup) == 0:
            el = time.perf_counter() - t_phase
            rate = (t + 1) / el
            acc = float(np.array(info["accept_prob"]).mean())
            print(f"[warmup] {t + 1}/{n_warmup} | {rate:.2f} it/s | "
                  f"accept {acc:.2f} | elapsed {_fmt_eta(el)} | "
                  f"eta {_fmt_eta((n_warmup - t - 1) / rate)}", flush=True)

    params = kernel.make_params(adapt_state, warmup=False)
    mx.eval(*params.values())

    # -- sampling: frozen params ------------------------------------------
    if storage is None:
        storage = MemoryBackend()
    storage.reserve(n_samples, n_chains, dim)
    accept_sum = mx.zeros((n_chains,))
    divergent_sum = mx.zeros((n_chains,))
    total_iters = n_samples * thin

    # frames waiting for their iteration to finish on the GPU; entries are
    # device arrays, so the host copy still happens exactly once, just
    # `depth` iterations later, by which time the work is already done
    pending: deque = deque()

    def _flush_one(frame):
        t_i, u_dev, lp_dev = frame
        if (t_i + 1) % thin:
            return
        # np.array() forces evaluation, but for a pipelined frame the
        # iteration finished several steps ago, so it is a pure copy
        u_np, lp_np = np.array(u_dev), np.array(lp_dev)
        storage.append(u_np, lp_np)
        if archive is not None:
            archive.record(u_np, lp_np)

    def _drain():
        while pending:
            _flush_one(pending.popleft())

    depth = 0 if pipeline == "auto" else max(0, int(pipeline))
    auto = pipeline == "auto"

    t_phase = time.perf_counter()
    for t in range(total_iters):
        state, info = step(keys.key(n_warmup + t), state, params)
        accept_sum = accept_sum + info["accept_prob"]
        if "diverged" in info:
            divergent_sum = divergent_sum + info["diverged"]
        if reanchor_every and (t + 1) % reanchor_every == 0:
            # the float64 path runs on the host: everything in flight has
            # to land first
            _drain()
            mx.eval(*state.values(), accept_sum, divergent_sum)
            state = reanchor(state)
        if depth:
            # keep the queue moving without blocking on it
            mx.async_eval(*state.values(), accept_sum, divergent_sum)
            pending.append((t, state["u"], state["log_prob"]))
            if len(pending) > depth:
                _flush_one(pending.popleft())
        else:
            # evaluate every iteration to bound lazy-graph growth
            mx.eval(*state.values(), accept_sum, divergent_sum)
            _flush_one((t, state["u"], state["log_prob"]))
        if auto and t + 1 == min(_PIPELINE_PROBE, total_iters):
            # decide once, from measured cost: pipelining trades peak
            # memory for latency, and only pays when latency dominates
            per_iter = (time.perf_counter() - t_phase) / (t + 1)
            auto = False
            if per_iter < _PIPELINE_THRESHOLD_S:
                depth = _PIPELINE_DEPTH
                if progress:
                    print(f"[sample] pipelining at depth {depth} "
                          f"({per_iter * 1e6:.0f} us/iter)", flush=True)
            elif progress:
                print(f"[sample] not pipelining ({per_iter * 1e3:.1f} ms/iter "
                      "— the eval barrier is already negligible)", flush=True)
        if progress and (t + 1) % _tick_every(total_iters) == 0:
            _drain()
            mx.eval(accept_sum, divergent_sum)
            el = time.perf_counter() - t_phase
            rate = (t + 1) / el
            acc = float(np.array(accept_sum).mean()) / (t + 1)
            ndiv = int(np.array(divergent_sum).sum())
            print(f"[sample] {t + 1}/{total_iters} | {rate:.2f} it/s | "
                  f"accept {acc:.2f} | divergences {ndiv} | "
                  f"elapsed {_fmt_eta(el)} | "
                  f"eta {_fmt_eta((total_iters - t - 1) / rate)}", flush=True)

    _drain()
    mx.eval(accept_sum, divergent_sum)

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
