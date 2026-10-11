"""Nesterov dual averaging of the leapfrog step size.

The Hoffman & Gelman (2014) scheme, driven by the *harmonic* mean of the
per-chain acceptance probabilities (as in the ChEES-HMC reference
implementation): the harmonic mean is dragged down hard by straggler
chains with near-zero acceptance, forcing the step size small enough that
every chain mixes, not just the average one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mlx.core as mx


@dataclass
class DualAveragingState:
    log_eps: mx.array      # scalar, current step size
    log_eps_bar: mx.array  # scalar, averaged iterate (used after freeze)
    h_bar: mx.array        # scalar, running error average
    mu: float              # shrinkage anchor, log(10 * eps0)


def init_dual_averaging(eps0: float) -> DualAveragingState:
    return DualAveragingState(
        log_eps=mx.array(math.log(eps0)),
        log_eps_bar=mx.array(math.log(eps0)),
        h_bar=mx.array(0.0),
        mu=math.log(10.0 * eps0),
    )


def harmonic_mean(a: mx.array, floor: float = 1e-10,
                  include: mx.array | None = None) -> mx.array:
    """Harmonic mean of per-chain acceptance probabilities.

    ``include`` (bool, per chain) restricts the mean to those chains. It
    exists for one case: a chain whose acceptance is zero at *every* step
    size -- one sitting at a non-finite log-density or a target's failure
    sentinel. The harmonic mean is meant to let a straggler pull the step
    size down until it mixes, but such a chain never mixes, so it pins the
    statistic near ``n_chains * floor`` and dual averaging shrinks the step
    without limit: measured, ONE such chain of 64 took ChEES from 1e-2 to
    7e-7. When every chain is included, or none is, the plain harmonic mean
    over all chains is returned unchanged, bit for bit.
    """
    full = 1.0 / mx.mean(1.0 / (a + floor))
    if include is None:
        return full
    w = include.astype(a.dtype)
    n = mx.sum(w)
    masked = n / mx.maximum(mx.sum(w / (a + floor)), 1e-30)
    use_full = (n == a.shape[0]) | (n == 0)
    return mx.where(use_full, full, masked)


def update_dual_averaging(
    da: DualAveragingState,
    accept_stat: mx.array,
    t: int,
    target: float = 0.651,
    gamma: float = 0.05,
    t0: float = 10.0,
    kappa: float = 0.75,
) -> DualAveragingState:
    """One warmup update from the iteration's acceptance statistic; ``t``
    is the 1-based warmup iteration."""
    eta = 1.0 / (t + t0)
    err = target - accept_stat
    h_bar = (1.0 - eta) * da.h_bar + eta * err
    log_eps = da.mu - math.sqrt(t) / gamma * h_bar
    w = t ** (-kappa)
    log_eps_bar = w * log_eps + (1.0 - w) * da.log_eps_bar
    return DualAveragingState(
        log_eps=log_eps, log_eps_bar=log_eps_bar, h_bar=h_bar, mu=da.mu
    )
