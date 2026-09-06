"""Cross-chain running moments for diagonal preconditioning.

With thousands of chains, the cross-chain sample moments at a single
iteration are already low-variance estimates of the posterior moments, so a
simple exponential moving average with decay ``beta_t = t / (t + 8)``
(fast forgetting early in warmup, slow later — the schedule used by the
ChEES-HMC reference implementation) replaces Stan-style windowing.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx


@dataclass
class MomentsState:
    m1: mx.array  # (dim,) EMA of cross-chain mean
    m2: mx.array  # (dim,) EMA of cross-chain second moment


def init_moments(dim: int) -> MomentsState:
    return MomentsState(m1=mx.zeros((dim,)), m2=mx.zeros((dim,)))


def update_moments(ms: MomentsState, u: mx.array, t: int) -> MomentsState:
    """EMA update from the current (n_chains, dim) positions; ``t`` is the
    1-based warmup iteration."""
    beta = t / (t + 8.0)
    mean = mx.mean(u, axis=0)
    sq = mx.mean(u * u, axis=0)
    return MomentsState(
        m1=beta * ms.m1 + (1.0 - beta) * mean,
        m2=beta * ms.m2 + (1.0 - beta) * sq,
    )


def stddev(ms: MomentsState, reg: float = 1e-8) -> mx.array:
    """Per-dimension posterior scale estimate."""
    var = mx.maximum(ms.m2 - ms.m1 * ms.m1, reg)
    return mx.sqrt(var)
