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


def harmonic_mean(a: mx.array, floor: float = 1e-10) -> mx.array:
    """Harmonic mean of per-chain acceptance probabilities."""
    return 1.0 / mx.mean(1.0 / (a + floor))


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
