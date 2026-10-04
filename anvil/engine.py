"""The shared driver loop: warmup (adapting) then sampling (frozen).

Kernels supply pure ``step`` functions; the engine owns iteration, the
``mx.eval`` cadence that bounds lazy-graph growth, storage handoff, and
acceptance bookkeeping. One loop serves every kernel.
"""

from __future__ import annotations

import inspect
import os
import time
import warnings
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


#: bump when the .npz layout changes incompatibly
_STATE_FORMAT = 1


def _anvil_version() -> str:
    from . import __version__          # deferred: anvil/__init__ imports us
    return __version__


def _to_mx(a: np.ndarray) -> mx.array:
    # mx.array() on a float64 numpy array silently returns float32; naming
    # the dtype is the only way to keep a float64 round-trip exact
    return mx.array(a, dtype=mx.float64) if a.dtype == np.float64 else mx.array(a)


def _kernel_class(name: str) -> type[Kernel]:
    """The Kernel subclass called ``name``. Found by walking the subclass
    tree rather than a registry, so a user-defined kernel works as soon as
    its module is imported."""
    seen: dict[str, type[Kernel]] = {}
    stack = [Kernel]
    while stack:
        cls = stack.pop()
        seen.setdefault(cls.__name__, cls)
        stack.extend(cls.__subclasses__())
    if name not in seen:
        raise ValueError(
            f"no imported Kernel subclass is named {name!r}, so there is no "
            f"way to know what this state caches; import the module that "
            f"defines it, or pass kernel= an instance of it "
            f"(known: {', '.join(sorted(seen))})")
    return seen[name]


@dataclass
class ResumeState:
    """A run frozen between iterations: chain state, frozen params, and the
    position reached in the key stream.

    Produced by :meth:`Results.resume_state` or :func:`load_state`, consumed
    by ``run(..., resume=...)``. ``iteration`` is the total number of engine
    iterations already drawn from the key stream (warmup plus
    ``n_samples * thin``); continuing from it is what keeps a resumed
    segment's randomness disjoint from the segment before it.
    """

    state: ChainState
    params: dict[str, mx.array]
    iteration: int
    seed: int
    n_chains: int
    dim: int
    kernel: str
    kernel_ckpt: dict[str, float] = field(default_factory=dict)
    anvil_version: str = ""

    # -- validation --------------------------------------------------------

    def check(self, target=None, u0=None, kernel=None) -> None:
        """Raise a legible error rather than let a mismatch crash inside a
        compiled step."""
        u = self.state.get("u")
        if u is None:
            raise ValueError("resume state has no 'u' (chain positions)")
        if tuple(u.shape) != (self.n_chains, self.dim):
            raise ValueError(
                f"resume state is inconsistent: declares {self.n_chains} "
                f"chains x {self.dim} dims, but 'u' has shape {tuple(u.shape)}"
            )
        lp = self.state.get("log_prob")
        if lp is not None and tuple(lp.shape) != (self.n_chains,):
            raise ValueError(
                f"resume state is inconsistent: 'log_prob' has shape "
                f"{tuple(lp.shape)}, expected ({self.n_chains},)")
        if kernel is not None and type(kernel).__name__ != self.kernel:
            raise ValueError(
                f"resume state was written by {self.kernel}, but this run "
                f"uses {type(kernel).__name__}: one kernel's frozen params "
                f"mean nothing to another. To change kernels, start a fresh "
                f"run (with warmup) from the saved positions instead.")
        t_dim = getattr(target, "dim", None)
        if t_dim is not None and int(t_dim) != self.dim:
            raise ValueError(
                f"resume state has dim={self.dim}, but the target has "
                f"dim={int(t_dim)}")
        if u0 is not None and tuple(u0.shape) != (self.n_chains, self.dim):
            raise ValueError(
                f"resume state holds {self.n_chains} chains x {self.dim} "
                f"dims, but u0 has shape {tuple(u0.shape)}. u0 is not used on "
                f"a resume, so this is a configuration mismatch, not an "
                f"initialization.")

    # -- moving the chains -------------------------------------------------

    def with_positions(self, u, target: LogDensity, kernel=None
                       ) -> "ResumeState":
        """A copy of this state with the chains moved to ``u``
        ((n_chains, dim), unconstrained space), every cached per-chain
        quantity the kernel keeps recomputed there, and everything else --
        frozen params, iteration, seed, kernel -- unchanged.

        For composing an exact move of your own with anvil's sampling: draw
        new positions however you like (a Gibbs sweep over a conditional
        that anvil cannot see, a mode-hopping proposal, a
        reparameterization), hand them in here, and resume. The result is
        still one Markov chain, so its segments pool exactly as resumed
        segments already do.

        What this exists to prevent is the hand-written version: writing
        ``state["u"]`` and leaving the cached ``log_prob`` (and, for ChEES,
        ``grad``) describing where the chain *used* to be. That resumes from
        a stale cache — a wrong answer that no diagnostic flags, because
        every array still has the right shape and the chain still moves.
        The recomputation goes through the kernel's
        :meth:`~anvil.kernels.base.Kernel.refresh`, so a kernel that caches
        something this layout does not know about raises rather than
        silently copying it.

        The key stream is untouched: ``iteration`` and ``seed`` are carried
        over, so the next segment draws exactly the keys it would have drawn
        without the move. Randomness for the move itself is the caller's.

        **The target may also have changed**, which is the other half of a
        Gibbs scheme: a block the target *holds* rather than samples. That is
        safe between runs — ``run`` retraces the kernel and recomputes the
        cached log-density on every call — but only between them. Change the
        target before the ``run`` that should see it, never from inside one.
        """
        u = (u.astype(mx.float32) if isinstance(u, mx.array)
             else mx.array(np.asarray(u, dtype=np.float32)))
        if tuple(u.shape) != (self.n_chains, self.dim):
            raise ValueError(
                f"new positions must have shape ({self.n_chains}, "
                f"{self.dim}) to match the resumed chains, got "
                f"{tuple(u.shape)}")
        if not bool(mx.all(mx.isfinite(u)).item()):
            bad = int(np.array(~mx.isfinite(u).all(axis=-1)).sum())
            raise ValueError(
                f"new positions are not all finite ({bad} of "
                f"{self.n_chains} chains); a non-finite position cannot be "
                f"refreshed into a usable state")

        cls = type(kernel) if kernel is not None else _kernel_class(self.kernel)
        fn = kernel.refresh if kernel is not None else cls.refresh
        if kernel is None and not inspect.ismethod(fn):
            raise TypeError(
                f"{cls.__name__}.refresh is an instance method, so it cannot "
                f"be reached from a saved state by name; declare it a "
                f"classmethod, or pass kernel= an instance")
        fresh = fn(self.state, u, target)
        mx.eval(*fresh.values())

        lp = fresh.get("log_prob")
        if lp is not None and not bool(mx.all(mx.isfinite(lp)).item()):
            bad = int(np.array(~mx.isfinite(lp)).sum())
            raise ValueError(
                f"the target is not finite at the new positions for {bad} of "
                f"{self.n_chains} chains. Resuming from those would strand "
                f"them: a chain whose current log-probability is -inf can "
                f"only move if a proposal happens to land somewhere finite. "
                f"Keep the move inside the support, or leave those chains "
                f"where they were.")
        return ResumeState(
            state=fresh,
            params=dict(self.params),
            iteration=self.iteration,
            seed=self.seed,
            n_chains=self.n_chains,
            dim=self.dim,
            kernel=self.kernel,
            kernel_ckpt=dict(self.kernel_ckpt),
            anvil_version=self.anvil_version,
        )

    # -- persistence -------------------------------------------------------

    def save(self, path) -> None:
        """Write to ``path`` as a plain ``.npz`` — arrays under transparent
        names, no pickle, so another process (or another language) can read
        it and a mismatch can be *reported* rather than raised from inside a
        deserializer."""
        out: dict[str, np.ndarray] = {
            "format": np.array(_STATE_FORMAT),
            "anvil_version": np.array(self.anvil_version or _anvil_version()),
            "kernel": np.array(self.kernel),
            "iteration": np.array(int(self.iteration)),
            "seed": np.array(int(self.seed)),
            "n_chains": np.array(int(self.n_chains)),
            "dim": np.array(int(self.dim)),
        }
        for k, v in self.state.items():
            out[f"state__{k}"] = np.array(v)
        for k, v in self.params.items():
            out[f"params__{k}"] = np.array(v)
        for k, v in self.kernel_ckpt.items():
            out[f"ckpt__{k}"] = np.array(float(v))
        path = os.fspath(path)
        np.savez(path, **out)
        # np.savez appends .npz to a bare name; hand back what we promised
        if not path.endswith(".npz") and os.path.exists(path + ".npz"):
            os.replace(path + ".npz", path)


def load_state(path) -> ResumeState:
    """Read a state written by :meth:`Results.save_state` back into a
    :class:`ResumeState` accepted by ``run(..., resume=...)``."""
    with np.load(os.fspath(path), allow_pickle=False) as z:
        fmt = int(z["format"]) if "format" in z else -1
        if fmt != _STATE_FORMAT:
            raise ValueError(
                f"{path}: state format {fmt}, this anvil reads "
                f"{_STATE_FORMAT}")
        state = {k[len("state__"):]: _to_mx(z[k])
                 for k in z.files if k.startswith("state__")}
        params = {k[len("params__"):]: _to_mx(z[k])
                  for k in z.files if k.startswith("params__")}
        ckpt = {k[len("ckpt__"):]: float(z[k])
                for k in z.files if k.startswith("ckpt__")}
        rs = ResumeState(
            state=state, params=params, kernel_ckpt=ckpt,
            iteration=int(z["iteration"]), seed=int(z["seed"]),
            n_chains=int(z["n_chains"]), dim=int(z["dim"]),
            kernel=str(z["kernel"]), anvil_version=str(z["anvil_version"]),
        )
    rs.check()
    return rs


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
    #: cheap per-probe record of the warmup phase, for warmup_report():
    #: keys "iter", "sd", "mean", "accept", "step_size"
    warmup_trace: dict[str, np.ndarray] | None = None
    extras: dict[str, Any] = field(default_factory=dict)
    #: effective RNG seed, and total iterations drawn from its key stream
    #: (this run's *and* any it resumed) -- together they are what lets a
    #: continuation keep off the keys already spent
    seed: int = 0
    iters_consumed: int = 0
    kernel: str = ""
    kernel_ckpt: dict[str, float] = field(default_factory=dict)

    def get_chain(self, discard: int = 0, thin: int = 1, flat: bool = False):
        return self.backend.get_chain(discard, thin, flat)

    def get_log_prob(self, discard: int = 0, thin: int = 1, flat: bool = False):
        return self.backend.get_log_prob(discard, thin, flat)

    def resume_state(self) -> ResumeState:
        """The handle for continuing these chains: pass it (or this
        ``Results``) as ``run(..., resume=...)``."""
        return ResumeState(
            state=dict(self.final_state),
            params=dict(self.final_params),
            iteration=int(self.iters_consumed),
            seed=int(self.seed),
            n_chains=int(self.n_chains),
            dim=int(self.dim),
            kernel=self.kernel,
            kernel_ckpt=dict(self.kernel_ckpt),
            anvil_version=_anvil_version(),
        )

    def save_state(self, path) -> None:
        """Persist enough to continue this run in another process
        (``.npz``, no pickle). See :func:`anvil.load_state`."""
        self.resume_state().save(path)


def _as_resume_state(obj) -> ResumeState:
    if isinstance(obj, ResumeState):
        return obj
    if isinstance(obj, Results):
        return obj.resume_state()
    raise TypeError(
        f"resume= takes a Results or a ResumeState from anvil.load_state(), "
        f"not {type(obj).__name__}")


def run(
    kernel: Kernel,
    target: LogDensity,
    u0: mx.array | None = None,
    *,
    n_warmup: int | None = None,
    n_samples: int = 1000,
    thin: int = 1,
    seed: int | None = None,
    storage: MemoryBackend | None = None,
    compile_step: bool = True,
    progress: bool | int = False,
    reanchor_every: int = 0,
    archive=None,
    pipeline: int | str = "auto",
    warmup_probes: int = 40,
    callback=None,
    resume: "Results | ResumeState | None" = None,
) -> Results:
    """Run ``kernel`` on ``target`` from initial positions ``u0``
    ((n_chains, dim), float32). Records ``n_samples`` states per chain, one
    every ``thin`` post-warmup iterations, after ``n_warmup`` adapting ones
    (default 500, or 0 when continuing a run with ``resume``).

    ``reanchor_every`` > 0 recomputes the cached log_prob of the current
    states through the target's float64 path every that many iterations
    (requires ``target.log_prob_hi``). It is NOT a remedy for float32
    rounding: cached values are always fresh evaluations of a deterministic
    function, so rounding cannot accumulate, and re-anchoring does not
    change the distribution being sampled. Its use is a log-density whose
    *evaluation* is not deterministic. It is expensive (the float64 path can
    cost ~1000x the float32 one); default off.

    It is also **not** a way to pick up a target that changes mid-run, which
    an earlier version of this docstring suggested: re-anchoring refreshes
    the cached log-density while the kernel's compiled proposal keeps the
    target it was traced with, so the Metropolis step would then compare two
    different targets. Measured, not reasoned: with the target's mean moved
    from 0 to 0.5 by a callback and ``reanchor_every=10``, the draws came
    back centred on 0.026. See the paragraph below.

    **A target that changes between runs** — a Gibbs block the target holds
    rather than samples, a tempering beta, a swapped dataset, a retrained
    surrogate — is handled: ``run`` calls :meth:`~anvil.kernels.base.Kernel.retrace`
    at the start of every call, which discards the compiled graphs that froze
    the target at their first trace, and a resume also recomputes the cached
    log-density. Both are no-ops in effect for an unchanged target, so draws
    are bit-identical. A target changed *during* a run is not picked up until
    the next ``run`` call, so drive such a scheme as a sequence of segments
    (see :meth:`ResumeState.with_positions`) rather than from a callback.

    ``resume``: a :class:`Results` (or a :class:`ResumeState` from
    :func:`load_state`) to *continue* rather than start. The chains pick up
    at their final positions with the adaptation frozen where it stopped, so
    ``n_warmup`` defaults to 0 there — passing a nonzero one raises rather
    than silently discarding it, and ``u0`` is not used (if given it is only
    checked for a matching shape). ``kernel.init`` is not called;
    ``kernel.attach`` is, so a kernel object built for the resumed run is
    configured from the saved params rather than from its constructor
    arguments. The key stream continues from the resumed state's iteration
    count, so a continuation never replays the keys the first segment
    already spent, and ``seed`` defaults to the one recorded in the state
    (pass one explicitly to override). ``accept_fraction`` and the
    divergence counts describe the resumed segment alone.

    ``archive`` (a :class:`~anvil.surrogate.TrainingArchive`) records
    every stored (u, log_prob) frame as future emulator training data —
    this happens on already-evaluated host copies, off the hot path.

    ``progress``: False for silence; True for ticks every 10% of each
    phase; an int N for a tick every N iterations. Ticks are flushed
    (safe to ``tail -f`` through a redirected log) and report rate, ETA,
    mean acceptance, and the running divergence count.

    ``callback``: optional ``callback(phase, iteration, info)``, invoked on
    the same cadence as ``progress`` with ``phase`` in ``{"warmup",
    "sample"}`` and ``info`` carrying the numbers the progress line reports
    (``total``, ``rate``, ``accept``, ``elapsed``, ``eta``, ``step_size``,
    and ``n_divergent`` while sampling). Supplying one does not turn
    printing on, so a caller that renders its own progress can pass
    ``progress=False`` and still be driven; if ``progress`` is False the
    cadence is the one ``progress=True`` would have used.

    ``warmup_probes``: how many times during warmup to record cross-chain
    spread, acceptance and step size, for
    :func:`anvil.diagnostics.warmup_report` to judge afterwards whether
    warmup was long enough (or far longer than needed). Costs one host
    sync per probe -- negligible at the default 40 -- and 0 disables it.

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
    if u0 is not None:
        u0 = u0.astype(mx.float32) if u0.dtype != mx.float32 else u0

    if resume is None:
        if u0 is None:
            raise TypeError(
                "run() needs initial positions u0, unless resume= is given")
        n_chains, dim = u0.shape
        n_warmup = 500 if n_warmup is None else int(n_warmup)
        seed = 0 if seed is None else int(seed)
        keys = KeyStream(seed)
        iter0 = 0
        # a kernel object reused across runs holds compiled graphs that froze
        # the target as it was at their first trace; init() does not retrace
        kernel.retrace()
        state = kernel.init(keys.init_key(), u0, target)
        adapt_state = kernel.init_adapt(state)
        params = kernel.make_params(adapt_state, warmup=True)
    else:
        rs = _as_resume_state(resume)
        # a continuation does not adapt, so the fresh-run default of 500 is
        # not a request for warmup -- but an explicit one is, and honouring
        # it is impossible, so say so
        if n_warmup:
            raise ValueError(
                f"resume= continues frozen adaptation, so n_warmup must be 0 "
                f"(got {n_warmup}). Re-adapting would make the continuation a "
                f"different Markov chain, whose draws cannot honestly be "
                f"concatenated with the first segment's -- if that is what you "
                f"want, start a fresh run from the saved positions.")
        n_warmup = 0
        rs.check(target=target, u0=u0, kernel=kernel)
        n_chains, dim = rs.n_chains, rs.dim
        # continue the same stream: with n_warmup=0 a naive resume would
        # redraw the original run's *warmup* keys, correlating the
        # continuation with the adaptation phase it is supposed to follow
        seed = rs.seed if seed is None else int(seed)
        keys = KeyStream(seed)
        iter0 = int(rs.iteration)
        state, params, adapt_state = dict(rs.state), dict(rs.params), None
        kernel.attach(target, state, params)
        kernel.restore(rs.kernel_ckpt)
        # The target may have changed since this state was written -- a Gibbs
        # block it holds rather than samples, a tempering beta, a swapped
        # dataset, a retrained surrogate. Two things would otherwise be stale,
        # and both are silent rather than loud: the kernel's compiled graphs
        # (which froze the target at their first trace) and the cached
        # log-density in the state. Retrace, then recompute the cache. For an
        # unchanged target both are exactly what they already were, so draws
        # are bit-identical.
        kernel.retrace()
        try:
            state = kernel.refresh(state, state["u"], target)
        except NotImplementedError as exc:
            warnings.warn(
                f"resuming without refreshing the cached log-density, which "
                f"is stale if the target changed since the state was written: "
                f"{exc}", stacklevel=2)
        mx.eval(*state.values(), *params.values())
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
        if progress is True or (not progress and callback is not None):
            return max(1, n_total // 10)
        return max(1, int(progress))

    ticking = bool(progress) or callback is not None

    def _step_size_of(pars):
        p = pars.get("step_size")
        return float(p.item()) if p is not None else None

    def _fmt_eta(seconds):
        m, s = divmod(int(seconds), 60)
        h, m = divmod(m, 60)
        return f"{h:d}:{m:02d}:{s:02d}"

    # -- warmup: adapt every iteration ------------------------------------
    probe_every = (max(1, n_warmup // warmup_probes)
                   if warmup_probes and n_warmup else 0)
    trace: dict[str, list] = {k: [] for k in
                              ("iter", "sd", "mean", "accept", "step_size")}
    t_phase = time.perf_counter()
    for t in range(n_warmup):
        state, info = step(keys.key(iter0 + t), state, params)
        adapt_state = kernel.adapt(adapt_state, state, info, t + 1)
        params = kernel.make_params(adapt_state, warmup=True)
        if reanchor_every and (t + 1) % reanchor_every == 0:
            state = reanchor(state)
        mx.eval(*state.values(), *params.values())
        if probe_every and (t + 1) % probe_every == 0:
            u = state["u"]
            trace["iter"].append(t + 1)
            trace["sd"].append(np.array(mx.std(u, axis=0)))
            trace["mean"].append(np.array(mx.mean(u, axis=0)))
            trace["accept"].append(float(mx.mean(info["accept_prob"]).item()))
            trace["step_size"].append(
                float(params["step_size"].item())
                if "step_size" in params else float("nan"))
        if ticking and (t + 1) % _tick_every(n_warmup) == 0:
            el = time.perf_counter() - t_phase
            rate = (t + 1) / el
            acc = float(np.array(info["accept_prob"]).mean())
            eta = (n_warmup - t - 1) / rate
            if progress:
                print(f"[warmup] {t + 1}/{n_warmup} | {rate:.2f} it/s | "
                      f"accept {acc:.2f} | elapsed {_fmt_eta(el)} | "
                      f"eta {_fmt_eta(eta)}", flush=True)
            if callback is not None:
                callback("warmup", t + 1, {
                    "total": n_warmup, "rate": rate, "accept": acc,
                    "elapsed": el, "eta": eta,
                    "step_size": _step_size_of(params)})

    if resume is None:
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
        state, info = step(keys.key(iter0 + n_warmup + t), state, params)
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
        if ticking and (t + 1) % _tick_every(total_iters) == 0:
            _drain()
            mx.eval(accept_sum, divergent_sum)
            el = time.perf_counter() - t_phase
            rate = (t + 1) / el
            acc = float(np.array(accept_sum).mean()) / (t + 1)
            ndiv = int(np.array(divergent_sum).sum())
            if callback is not None:
                callback("sample", t + 1, {
                    "total": total_iters, "rate": rate, "accept": acc,
                    "elapsed": el, "eta": (total_iters - t - 1) / rate,
                    "n_divergent": ndiv, "step_size": _step_size_of(params)})
            if progress:
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
        warmup_trace=({k: np.array(v) for k, v in trace.items()}
                      if trace["iter"] else None),
        extras={
            "n_divergent": int(np.array(divergent_sum).sum()),
            # per-chain resolution: a chain that both sits low in
            # log-probability and diverges on most proposals is stuck at a
            # boundary, whereas one that sits low with healthy acceptance is
            # in a secondary mode. The scalar above cannot tell those apart.
            "divergent_per_chain": np.array(divergent_sum),
        },
        seed=seed,
        iters_consumed=iter0 + n_warmup + total_iters,
        kernel=type(kernel).__name__,
        kernel_ckpt=kernel.checkpoint(),
    )
