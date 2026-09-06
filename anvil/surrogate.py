"""Seams for neural likelihood emulation (v2, BAMBI-style).

The plan (Graff et al. 2012, arXiv:1110.2997, modernized): while the exact
likelihood runs, accumulate (parameters, log-likelihood) pairs; train an
MLX network on them in the background; once its held-out error is far
below the Metropolis decision scale, let it stand in for the expensive
likelihood (always keeping periodic exact-likelihood audits). Deployment
on the Neural Engine via Core ML is an optional final step.

v1 ships only the interfaces and the data collection so the engine never
needs to change: everything consumes :class:`~anvil.logdensity.LogDensity`,
so a trained surrogate slots in behind :class:`SwitchableLogDensity`.
"""

from __future__ import annotations

import numpy as np

import mlx.core as mx

from .logdensity import LogDensity


class TrainingArchive:
    """Host-side ring buffer of (u, log_prob) evaluation pairs.

    Stored in float64 numpy (values come from wherever the caller chooses —
    typically the fp32 production path; pair with the fp64 path for
    highest-quality training targets). Ring semantics: once ``capacity`` is
    reached, oldest entries are overwritten, keeping the archive biased
    toward the posterior typical set that late-chain states occupy.
    """

    def __init__(self, dim: int, capacity: int = 1_000_000):
        self.dim = dim
        self.capacity = int(capacity)
        self._u = np.empty((self.capacity, dim), dtype=np.float64)
        self._lp = np.empty(self.capacity, dtype=np.float64)
        self._n = 0       # total ever recorded
        self._head = 0    # next write position

    def record(self, u: np.ndarray, log_prob: np.ndarray) -> None:
        u = np.atleast_2d(np.asarray(u, dtype=np.float64))
        lp = np.atleast_1d(np.asarray(log_prob, dtype=np.float64))
        m = u.shape[0]
        if m > self.capacity:  # keep only the newest slice
            u, lp, m = u[-self.capacity:], lp[-self.capacity:], self.capacity
        end = self._head + m
        if end <= self.capacity:
            self._u[self._head:end] = u
            self._lp[self._head:end] = lp
        else:
            k = self.capacity - self._head
            self._u[self._head:] = u[:k]
            self._lp[self._head:] = lp[:k]
            self._u[: end - self.capacity] = u[k:]
            self._lp[: end - self.capacity] = lp[k:]
        self._head = end % self.capacity
        self._n += m

    def __len__(self) -> int:
        return min(self._n, self.capacity)

    def as_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        n = len(self)
        return self._u[:n].copy(), self._lp[:n].copy()


class ArchivingLogDensity(LogDensity):
    """Wrap a density; record evaluations into a TrainingArchive.

    NOTE: recording forces host evaluation, so this wrapper must NOT be
    used inside an ``mx.compile``-d sampler step (pass ``compile_step=
    False`` to the engine, or — preferred — record at storage cadence with
    ``engine.run(..., archive=...)``, which costs nothing extra). This
    wrapper suits uncompiled evaluation loops such as prior-predictive
    sweeps or optimizer traces.
    """

    def __init__(self, inner: LogDensity, archive: TrainingArchive | None = None,
                 every: int = 1):
        self.inner = inner
        self.dim = inner.dim
        self.supports_grad = inner.supports_grad
        self.archive = archive or TrainingArchive(inner.dim)
        self.every = int(every)
        self._calls = 0

    def log_prob(self, u: mx.array) -> mx.array:
        lp = self.inner.log_prob(u)
        self._calls += 1
        if self._calls % self.every == 0:
            self.archive.record(np.array(u, dtype=np.float64),
                                np.array(lp, dtype=np.float64))
        return lp

    def log_prob_and_grad(self, u: mx.array):
        lp, g = self.inner.log_prob_and_grad(u)
        self._calls += 1
        if self._calls % self.every == 0:
            self.archive.record(np.array(u, dtype=np.float64),
                                np.array(lp, dtype=np.float64))
        return lp, g

    def __getattr__(self, name):
        # forward log_prob_hi and other optional protocol pieces
        return getattr(self.inner, name)


class Surrogate(LogDensity):
    """Interface a trained emulator must implement (v2).

    In addition to ``log_prob``, a surrogate must know its own
    trustworthiness: ``error_estimate`` returns a per-point predictive
    error in log-likelihood units, compared against the Metropolis
    decision scale to decide when the surrogate may stand in.
    """

    def error_estimate(self, u: mx.array) -> mx.array:
        raise NotImplementedError


class SwitchableLogDensity(LogDensity):
    """Exact <-> surrogate switch. v1: always exact; the seam exists so a
    v2 controller can flip ``use_surrogate`` (globally or per batch) once
    the surrogate's error estimate clears the threshold."""

    def __init__(self, exact: LogDensity, surrogate: Surrogate | None = None,
                 error_threshold: float = 0.1):
        self.exact = exact
        self.surrogate = surrogate
        self.error_threshold = float(error_threshold)
        self.use_surrogate = False
        self.dim = exact.dim
        self.supports_grad = exact.supports_grad

    def log_prob(self, u: mx.array) -> mx.array:
        if self.use_surrogate and self.surrogate is not None:
            return self.surrogate.log_prob(u)
        return self.exact.log_prob(u)

    def log_prob_and_grad(self, u: mx.array):
        if self.use_surrogate and self.surrogate is not None:
            return self.surrogate.log_prob_and_grad(u)
        return self.exact.log_prob_and_grad(u)
