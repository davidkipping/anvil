"""ChEES-HMC: GPU-friendly Hamiltonian Monte Carlo with cross-chain
adaptation (Hoffman & Sountsov, AISTATS 2021).

Design for vectorization: the trajectory-length jitter is a *per-iteration
scalar* from the Halton sequence, so every chain runs the same number of
leapfrog steps each iteration — there is no per-chain control flow
anywhere. The leapfrog count L = ceil(h*T/eps) is host-side control flow,
so the kernel compiles a *single* leapfrog step (with eps and the mass
matrix as array inputs, so adaptation never retraces) and drives it from a
Python loop; per-launch overhead is noise next to a batched
gradient+likelihood evaluation over thousands of chains.

Cost of the design: one tiny host sync per iteration to read the two
scalars (eps, T) that determine L.

Warmup adapts three things simultaneously from cross-chain statistics —
step size (dual averaging on the harmonic-mean acceptance), trajectory
length (ChEES criterion), and a diagonal preconditioner (EMA moments) —
then all three freeze at their averaged iterates.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mlx.core as mx

from ..adaptation.chees_criterion import (
    ChEESState,
    chees_gradient,
    init_chees,
    update_chees,
)
from ..adaptation.dual_averaging import (
    DualAveragingState,
    harmonic_mean,
    init_dual_averaging,
    update_dual_averaging,
)
from ..adaptation.moments import MomentsState, init_moments, stddev, update_moments
from ..halton import halton_jitter
from ..logdensity import LogDensity
from ..state import ChainState
from .base import Kernel, StepInfo


@dataclass
class ChEESAdaptState:
    da: DualAveragingState
    chees: ChEESState
    moments: MomentsState
    frozen_sigma: mx.array  # (dim,) preconditioner scales
    t: int = 0


class ChEESHMC(Kernel):
    needs_grad = True
    self_compiled = True

    def __init__(
        self,
        target: LogDensity,
        step_size: float = 0.1,
        traj_length: float = 1.0,
        max_leapfrog: int = 1000,
        target_accept: float = 0.651,
        divergence_threshold: float = 1000.0,
        precondition_after: int = 50,
    ):
        if not target.supports_grad:
            raise ValueError("ChEES-HMC requires a gradient-capable target")
        self.target = target
        self.eps0 = float(step_size)
        self.T0 = float(traj_length)
        self.max_leapfrog = int(max_leapfrog)
        self.target_accept = float(target_accept)
        self.div_threshold = float(divergence_threshold)
        self.precondition_after = int(precondition_after)
        self._iter = 0
        self._last_h = 1.0
        self._leapfrog_c = mx.compile(self._leapfrog)
        self._finish_c = mx.compile(self._finish)

    # -- pure compiled pieces ---------------------------------------------

    def _leapfrog(self, q, p, g, eps, inv_mass):
        """One leapfrog step; eps (scalar) and inv_mass (dim,) are array
        inputs so adaptation changes them without retracing."""
        p_half = p + 0.5 * eps * g
        q_new = q + eps * inv_mass * p_half
        lp, g_new = self.target.log_prob_and_grad(q_new)
        p_new = p_half + 0.5 * eps * g_new
        return q_new, p_new, g_new, lp

    def _finish(self, key, q0, lp0, g0, p0, q1, lp1, g1, p1, inv_mass):
        """Accept/reject and state assembly (one compiled launch)."""
        ke0 = 0.5 * mx.sum(inv_mass * p0 * p0, axis=-1)
        ke1 = 0.5 * mx.sum(inv_mass * p1 * p1, axis=-1)
        dH = (lp1 - ke1) - (lp0 - ke0)
        dH_safe = mx.where(mx.isfinite(dH), dH, -mx.inf)
        accept_prob = mx.minimum(1.0, mx.exp(dH_safe))
        diverged = ~mx.isfinite(dH) | (dH < -self.div_threshold)
        log_u = mx.log(mx.random.uniform(shape=dH.shape, key=key))
        accept = (log_u < dH_safe) & ~diverged
        acc = accept[:, None]
        new_state = {
            "u": mx.where(acc, q1, q0),
            "log_prob": mx.where(accept, lp1, lp0),
            "grad": mx.where(acc, g1, g0),
        }
        info: StepInfo = {
            "accept_prob": accept_prob,
            "accepted": accept,
            "diverged": diverged,
            "q_prev": q0,
            "q_prop": q1,
            "v_end": inv_mass * p1,
        }
        return new_state, info

    # -- Kernel protocol ---------------------------------------------------

    def init(self, key, u0: mx.array, target: LogDensity) -> ChainState:
        self.target = target
        lp, g = target.log_prob_and_grad(u0)
        return {"u": u0, "log_prob": lp, "grad": g}

    def step(self, key, state, params):
        eps_f = float(params["step_size"].item())
        T_f = float(params["traj_length"].item())
        h = halton_jitter(self._iter)
        self._iter += 1
        self._last_h = h
        L = max(1, min(self.max_leapfrog, math.ceil(h * T_f / eps_f)))

        k_mom, k_acc = mx.random.split(key)
        u = state["u"]
        inv_mass = params["inv_mass"]          # (dim,) = sigma^2
        # p ~ N(0, M) with M = diag(1/sigma^2)
        p0 = mx.random.normal(u.shape, key=k_mom) / mx.sqrt(inv_mass)

        eps = params["step_size"]
        q, p, g, lp = u, p0, state["grad"], state["log_prob"]
        for _ in range(L):
            q, p, g, lp = self._leapfrog_c(q, p, g, eps, inv_mass)
        return self._finish_c(
            k_acc, u, state["log_prob"], state["grad"], p0, q, lp, g, p,
            inv_mass,
        )

    def init_adapt(self, state: ChainState) -> ChEESAdaptState:
        dim = state["u"].shape[1]
        return ChEESAdaptState(
            da=init_dual_averaging(self.eps0),
            chees=init_chees(self.T0),
            moments=init_moments(dim),
            frozen_sigma=mx.ones((dim,)),
        )

    def adapt(self, a: ChEESAdaptState, state, info: StepInfo, t: int):
        da = update_dual_averaging(
            a.da, harmonic_mean(info["accept_prob"]), t,
            target=self.target_accept,
        )
        grad = chees_gradient(
            info["q_prev"], info["q_prop"], info["v_end"],
            info["accept_prob"], self._last_h,
        )
        chees = update_chees(
            a.chees, grad, t, mx.exp(da.log_eps),
            max_leapfrog=self.max_leapfrog,
        )
        moments = update_moments(a.moments, state["u"], t)
        sigma = stddev(moments) if t >= self.precondition_after else a.frozen_sigma
        return ChEESAdaptState(
            da=da, chees=chees, moments=moments, frozen_sigma=sigma, t=t
        )

    def make_params(self, a: ChEESAdaptState, warmup: bool):
        if warmup:
            log_eps, log_T = a.da.log_eps, a.chees.log_T
        else:
            log_eps, log_T = a.da.log_eps_bar, a.chees.log_T_bar
        sigma = a.frozen_sigma
        return {
            "step_size": mx.exp(log_eps),
            "traj_length": mx.exp(log_T),
            "inv_mass": sigma * sigma,
        }
