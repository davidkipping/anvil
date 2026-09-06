"""Vectorized adaptive random-walk Metropolis.

The walking-skeleton kernel: simplest possible move, exercises the whole
engine (batched state, params, adaptation, storage, diagnostics) and serves
as the baseline all fancier kernels must beat in the benchmarks.

Proposal: ``u' = u + s · σ ⊙ ξ`` with global scale ``s`` (Robbins-Monro
toward 23.4% acceptance) and per-dimension scales ``σ`` from cross-chain
moments.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mlx.core as mx

from ..adaptation.moments import MomentsState, init_moments, stddev, update_moments
from ..logdensity import LogDensity
from ..state import ChainState, tree_where
from .base import Kernel, StepInfo

_TARGET_ACCEPT = 0.234


@dataclass
class RWMAdaptState:
    log_scale: mx.array  # scalar
    moments: MomentsState
    frozen_sigma: mx.array  # (dim,)


class RandomWalkMetropolis(Kernel):
    needs_grad = False

    def __init__(self, target: LogDensity, initial_scale: float = 0.5):
        self.target = target
        self.initial_scale = initial_scale

    def init(self, key: mx.array, u0: mx.array, target: LogDensity) -> ChainState:
        self.target = target
        lp = target.log_prob(u0)
        return {"u": u0, "log_prob": lp}

    def step(self, key, state, params):
        k_prop, k_acc = mx.random.split(key)
        u = state["u"]
        scale = params["step_scale"] * params["sigma"]  # (dim,)
        prop = u + scale * mx.random.normal(u.shape, key=k_prop, dtype=u.dtype)
        lp_prop = self.target.log_prob(prop)
        log_ratio = lp_prop - state["log_prob"]
        accept_prob = mx.minimum(1.0, mx.exp(log_ratio))
        log_uniform = mx.log(
            mx.random.uniform(shape=log_ratio.shape, key=k_acc, dtype=u.dtype)
        )
        accepted = log_uniform < log_ratio
        new_state = tree_where(
            accepted, {"u": prop, "log_prob": lp_prop}, state
        )
        return new_state, {"accept_prob": accept_prob, "accepted": accepted}

    def init_adapt(self, state: ChainState) -> RWMAdaptState:
        dim = state["u"].shape[1]
        return RWMAdaptState(
            log_scale=mx.array(math.log(self.initial_scale)),
            moments=init_moments(dim),
            frozen_sigma=mx.ones((dim,)),
        )

    def adapt(self, a: RWMAdaptState, state, info: StepInfo, t: int) -> RWMAdaptState:
        # Robbins-Monro on the global log-scale toward the RWM-optimal 0.234.
        lr = t ** (-0.6)
        mean_accept = mx.mean(info["accept_prob"])
        log_scale = a.log_scale + lr * (mean_accept - _TARGET_ACCEPT)
        moments = update_moments(a.moments, state["u"], t)
        # hold sigma at ones early on: with point-like initializations the
        # cross-chain spread is not yet a posterior-scale estimate
        sigma = stddev(moments) if t >= 50 else a.frozen_sigma
        return RWMAdaptState(log_scale=log_scale, moments=moments, frozen_sigma=sigma)

    def make_params(self, a: RWMAdaptState, warmup: bool) -> dict[str, mx.array]:
        return {"step_scale": mx.exp(a.log_scale), "sigma": a.frozen_sigma}
