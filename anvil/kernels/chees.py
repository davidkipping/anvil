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
import warnings
from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import numpy as np

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
from ..adaptation.moments import (
    DenseMomentsState,
    MomentsState,
    dense_factors,
    init_dense_moments,
    init_moments,
    stddev,
    update_dense_moments,
    update_moments,
)
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
    # dense preconditioner only (all None on the diagonal path)
    dense_moments: DenseMomentsState | None = None
    corr: mx.array | None = None    # (dim, dim) correlation R
    lrinv: mx.array | None = None   # (dim, dim) inv(chol(R))


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
        dense: bool = False,
        ridge: float = 1e-6,
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
        self.dense = bool(dense)
        self.ridge = float(ridge)
        self._iter = 0
        self._last_h = 1.0
        self._compile_kernels()

    def _compile_kernels(self):
        if self.dense:
            self._leapfrog_c = mx.compile(self._leapfrog_dense)
            self._finish_c = mx.compile(self._finish_dense)
        else:
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
        # A chain sitting on a non-finite log-probability (a hard prior
        # edge, a model that returns -inf) must be able to leave. Its dH is
        # +inf for exactly the proposal that rescues it, and treating
        # "not finite" as "divergent" would reject that move and trap the
        # chain permanently. So: a move out of a non-finite current state
        # into a finite one is always accepted, and a chain that is already
        # stuck is not counted as diverging -- it is not the proposal's
        # fault. Moves the other way (finite -> non-finite) are divergences
        # as before.
        stuck = ~mx.isfinite(lp0)
        rescue = stuck & mx.isfinite(lp1)
        dH_safe = mx.where(mx.isfinite(dH), dH,
                           mx.where(rescue, mx.inf, -mx.inf))
        accept_prob = mx.minimum(1.0, mx.exp(dH_safe))
        diverged = (~mx.isfinite(dH) | (dH < -self.div_threshold)) & ~stuck
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

    # -- dense-preconditioner variants -------------------------------------
    #
    # M^-1 is the full covariance Sigma = S R S. Every use of it goes through
    # _sigma_p, which applies the factored form so float32 only ever touches
    # the well-conditioned correlation matrix R.

    @staticmethod
    def _sigma_p(p, s, R):
        """Sigma @ p for row-vector batches: S(R(S p))."""
        return ((p * s) @ R) * s

    def _leapfrog_dense(self, q, p, g, eps, s, R):
        p_half = p + 0.5 * eps * g
        q_new = q + eps * self._sigma_p(p_half, s, R)
        lp, g_new = self.target.log_prob_and_grad(q_new)
        return q_new, p_half + 0.5 * eps * g_new, g_new, lp

    def _finish_dense(self, key, q0, lp0, g0, p0, q1, lp1, g1, p1, s, R):
        ke0 = 0.5 * mx.sum(p0 * self._sigma_p(p0, s, R), axis=-1)
        ke1 = 0.5 * mx.sum(p1 * self._sigma_p(p1, s, R), axis=-1)
        dH = (lp1 - ke1) - (lp0 - ke0)
        # A chain sitting on a non-finite log-probability (a hard prior
        # edge, a model that returns -inf) must be able to leave. Its dH is
        # +inf for exactly the proposal that rescues it, and treating
        # "not finite" as "divergent" would reject that move and trap the
        # chain permanently. So: a move out of a non-finite current state
        # into a finite one is always accepted, and a chain that is already
        # stuck is not counted as diverging -- it is not the proposal's
        # fault. Moves the other way (finite -> non-finite) are divergences
        # as before.
        stuck = ~mx.isfinite(lp0)
        rescue = stuck & mx.isfinite(lp1)
        dH_safe = mx.where(mx.isfinite(dH), dH,
                           mx.where(rescue, mx.inf, -mx.inf))
        accept_prob = mx.minimum(1.0, mx.exp(dH_safe))
        diverged = (~mx.isfinite(dH) | (dH < -self.div_threshold)) & ~stuck
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
            "v_end": self._sigma_p(p1, s, R),
        }
        return new_state, info

    # -- Kernel protocol ---------------------------------------------------

    def init(self, key, u0: mx.array, target: LogDensity) -> ChainState:
        self.target = target
        n_chains, dim = u0.shape
        if self.dense:
            # a cross-chain covariance needs comfortably more chains than
            # dimensions to be usable; below that, silently degrading to the
            # diagonal preconditioner is far better than inverting noise
            if n_chains < 4 * dim:
                warnings.warn(
                    f"dense=True needs n_chains >= 4*dim for a usable "
                    f"cross-chain covariance (got {n_chains} chains, dim "
                    f"{dim}); falling back to the diagonal preconditioner",
                    stacklevel=2,
                )
                self.dense = False
                self._compile_kernels()
            elif dim > 512:
                warnings.warn(
                    f"dense=True costs O(dim^2) per leapfrog step and a "
                    f"dim x dim host factorization per warmup iteration; at "
                    f"dim={dim} that may outweigh the preconditioning gain",
                    stacklevel=2,
                )
        lp, g = target.log_prob_and_grad(u0)
        return {"u": u0, "log_prob": lp, "grad": g}

    def step(self, key, state, params):
        eps_f = float(params["step_size"].item())
        T_f = float(params["traj_length"].item())
        h = halton_jitter(self._iter)
        self._iter += 1
        self._last_h = h
        if eps_f > 0.0 and math.isfinite(T_f):
            L = max(1, min(self.max_leapfrog, math.ceil(h * T_f / eps_f)))
        else:
            # Adaptation has collapsed -- dual averaging can drive the step
            # size to float32 underflow when a subset of chains sits on a
            # non-finite log-probability and drags the harmonic-mean
            # acceptance to zero. Take a single (inert) step rather than
            # dividing by zero: the run then ends with a stalled-chain
            # signature that warmup_report and the divergence counts can
            # describe, instead of a traceback from inside the sampler.
            L = 1

        k_mom, k_acc = mx.random.split(key)
        u = state["u"]
        eps = params["step_size"]
        z = mx.random.normal(u.shape, key=k_mom)

        if self.dense:
            # p ~ N(0, Sigma^-1): Cov((z @ B)/sd) = B^T B / (sd sd^T) with
            # B = inv(chol(R)), so B^T B = R^-1 and the whole thing is
            # (S R S)^-1. Getting this transpose wrong is silent -- it
            # samples a valid-looking chain from the wrong distribution.
            s_arr, R = params["sigma"], params["corr"]
            p0 = (z @ params["lrinv"]) / s_arr
            extra = (s_arr, R)
        else:
            inv_mass = params["inv_mass"]      # (dim,) = sigma^2
            # p ~ N(0, M) with M = diag(1/sigma^2)
            p0 = z / mx.sqrt(inv_mass)
            extra = (inv_mass,)

        q, p, g, lp = u, p0, state["grad"], state["log_prob"]
        for _ in range(L):
            q, p, g, lp = self._leapfrog_c(q, p, g, eps, *extra)
        return self._finish_c(
            k_acc, u, state["log_prob"], state["grad"], p0, q, lp, g, p,
            *extra,
        )

    def attach(self, target, state, params) -> None:
        self.target = target
        # The params decide the preconditioner, not the constructor: a run
        # that asked for dense= and was downgraded at init (too few chains)
        # wrote diagonal params, and continuing it densely would read a
        # "corr" that is not there. Reconcile, recompiling if we moved.
        dense = "corr" in params
        if dense != self.dense:
            self.dense = dense
            self._compile_kernels()

    def retrace(self, target=None) -> None:
        # _leapfrog and _finish both close over self.target, so their traced
        # graphs hold whatever it contained at the first call. init() and
        # attach() also recompile when the preconditioner type flips; that is
        # kept for kernels driven without run() (tests do), and costs nothing
        # here because mx.compile is lazy -- a discarded wrapper never traced.
        if target is not None:
            self.target = target
        self._compile_kernels()

    @classmethod
    def refresh(cls, state, u, target) -> ChainState:
        # one forward+backward pass, exactly as init does: the cached
        # gradient is as position-dependent as the log-probability, and
        # carrying either across a move is the bug this exists to stop
        lp, g = target.log_prob_and_grad(u)
        return cls._check_refresh(state, {"u": u, "log_prob": lp, "grad": g})

    def checkpoint(self) -> dict[str, float]:
        # the Halton jitter is a *sequence*, not a draw: restarting it would
        # replay the same trajectory lengths the first segment already used
        return {"halton_iter": float(self._iter)}

    def restore(self, ckpt: dict[str, float]) -> None:
        self._iter = int(ckpt.get("halton_iter", 0))

    def init_adapt(self, state: ChainState) -> ChEESAdaptState:
        dim = state["u"].shape[1]
        return ChEESAdaptState(
            da=init_dual_averaging(self.eps0),
            chees=init_chees(self.T0),
            moments=init_moments(dim),
            frozen_sigma=mx.ones((dim,)),
            dense_moments=init_dense_moments(dim) if self.dense else None,
            corr=mx.eye(dim) if self.dense else None,
            lrinv=mx.eye(dim) if self.dense else None,
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

        dm, corr, lrinv = a.dense_moments, a.corr, a.lrinv
        if self.dense:
            # one host copy per warmup iteration; microseconds at realistic
            # sizes, and it never touches the compiled step
            dm = update_dense_moments(dm, np.array(state["u"]), t)
            if t >= self.precondition_after:
                for ridge in (self.ridge, 100.0 * self.ridge, 1e4 * self.ridge):
                    try:
                        sigma, corr, lrinv = dense_factors(dm, ridge)
                        break
                    except np.linalg.LinAlgError:
                        continue
                # if every ridge failed, keep the previous factors: a warmup
                # iteration with a stale preconditioner is recoverable, a
                # crash is not

        return ChEESAdaptState(
            da=da, chees=chees, moments=moments, frozen_sigma=sigma, t=t,
            dense_moments=dm, corr=corr, lrinv=lrinv,
        )

    def make_params(self, a: ChEESAdaptState, warmup: bool):
        if warmup:
            log_eps, log_T = a.da.log_eps, a.chees.log_T
        else:
            log_eps, log_T = a.da.log_eps_bar, a.chees.log_T_bar
        sigma = a.frozen_sigma
        out = {"step_size": mx.exp(log_eps), "traj_length": mx.exp(log_T)}
        if self.dense:
            out["sigma"] = sigma
            out["corr"] = a.corr
            out["lrinv"] = a.lrinv
        else:
            out["inv_mass"] = sigma * sigma
        return out
