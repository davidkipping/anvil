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
import numpy as np


@dataclass
class MomentsState:
    m1: mx.array  # (dim,) EMA of cross-chain mean
    m2: mx.array  # (dim,) EMA of the cross-chain variance about each
    #                iteration's own mean (centred: see update_moments)


def init_moments(dim: int) -> MomentsState:
    return MomentsState(m1=mx.zeros((dim,)), m2=mx.zeros((dim,)))


def update_moments(ms: MomentsState, u: mx.array, t: int,
                   include: mx.array | None = None) -> MomentsState:
    """EMA update from the current (n_chains, dim) positions; ``t`` is the
    1-based warmup iteration. ``include`` (n_chains,) bool leaves chains out
    -- a dead chain far from the bulk would otherwise set the scale; with
    none included the moments are left as they were.

    The variance is accumulated about each iteration's own cross-chain
    mean, not as E[u^2] - E[u]^2: in float32 that difference cancels for a
    narrow posterior away from u = 0 (before 0.11.0 the scale came out 10x
    small at u = 3 with sd 1e-3, and 3x large at u = 1 with sd 3e-4)."""
    beta = t / (t + 8.0)
    if include is None:
        mean = mx.mean(u, axis=0)
        d = u - mean
        var = mx.mean(d * d, axis=0)
        return MomentsState(
            m1=beta * ms.m1 + (1.0 - beta) * mean,
            m2=beta * ms.m2 + (1.0 - beta) * var,
        )
    w = include.astype(u.dtype)[:, None]
    n = mx.sum(w)
    nz = mx.maximum(n, 1.0)
    # every chain included: the unmasked mean exactly, so a healthy run is
    # bit-identical to one without the mask (sum/n is not, unless n is a
    # power of two)
    full = n == u.shape[0]
    mean = mx.where(full, mx.mean(u, axis=0),
                    mx.sum(mx.where(include[:, None], u, 0.0), axis=0) / nz)
    d = u - mean
    dz = mx.where(include[:, None], d, 0.0)
    var = mx.where(full, mx.mean(d * d, axis=0), mx.sum(dz * dz, axis=0) / nz)
    any_ = n > 0
    return MomentsState(
        m1=mx.where(any_, beta * ms.m1 + (1.0 - beta) * mean, ms.m1),
        m2=mx.where(any_, beta * ms.m2 + (1.0 - beta) * var, ms.m2),
    )


def stddev(ms: MomentsState, reg: float = 1e-30) -> mx.array:
    """Per-dimension posterior scale estimate. ``reg`` only keeps a
    coordinate with no spread at all from a zero scale; before 0.11.0 it
    was 1e-8 (a floor of 1e-4 in u), which made every parameter narrower
    than that in u read as 1e-4 and shrank the shared step size for all."""
    return mx.sqrt(mx.maximum(ms.m2, reg))


# --- dense (full-covariance) preconditioning --------------------------------
#
# Kept on the host in float64 on purpose. The covariance itself is badly
# conditioned (measured cond ~1e7 on a transit posterior) and must never be
# handed to float32; what the GPU graph receives is the factored form
# Sigma = S R S, in which only the correlation matrix R (cond ~1e3) is an
# array. The matrices are dim x dim, so the host work is negligible beside a
# batched likelihood.


@dataclass
class DenseMomentsState:
    m1: np.ndarray  # (dim,)      float64 EMA of the cross-chain mean
    m2: np.ndarray  # (dim, dim)  float64 EMA of the second-moment matrix


def init_dense_moments(dim: int) -> DenseMomentsState:
    return DenseMomentsState(m1=np.zeros(dim), m2=np.eye(dim))


def update_dense_moments(
    ms: DenseMomentsState, u: np.ndarray, t: int, include: np.ndarray | None = None
) -> DenseMomentsState:
    """EMA update from the current (n_chains, dim) positions, same
    ``beta = t/(t+8)`` schedule as the diagonal path. ``u`` is host numpy;
    ``include`` leaves chains out as in :func:`update_moments`."""
    beta = t / (t + 8.0)
    u = np.asarray(u, dtype=np.float64)
    if include is not None:
        u = u[np.asarray(include, dtype=bool)]
        if u.shape[0] == 0:
            return ms
    return DenseMomentsState(
        m1=beta * ms.m1 + (1.0 - beta) * u.mean(axis=0),
        m2=beta * ms.m2 + (1.0 - beta) * (u.T @ u) / u.shape[0],
    )


def dense_factors(ms: DenseMomentsState, ridge: float = 1e-6):
    """Factor the EMA covariance into the arrays the kernel consumes.

    Returns ``(sd, R, B)`` as float32 mx arrays, where Sigma = S R S and
    B = inv(chol(R)), with R already carrying the ridge: the momentum draw
    ``(z @ B) / sd`` has covariance (S R S)^-1 and the kinetic energy uses
    S R S, so both sides must see the same R. (Before 0.10.2 the energy used
    R without the ridge: posterior variances ~ridge/lambda_min(R) low,
    9% at the first escalation on a rho = 0.999 pair.)

    Raises ``numpy.linalg.LinAlgError`` if R is not positive definite at
    this ridge, so the caller can escalate or fall back.
    """
    cov = ms.m2 - np.outer(ms.m1, ms.m1)
    sd = np.sqrt(np.maximum(np.diag(cov), 1e-30))
    R = cov / np.outer(sd, sd)
    R = 0.5 * (R + R.T)                       # kill asymmetry from rounding
    np.fill_diagonal(R, 1.0)
    R = R + ridge * np.eye(len(sd))
    Lr = np.linalg.cholesky(R)
    B = np.linalg.inv(Lr)
    return (
        mx.array(sd.astype(np.float32)),
        mx.array(R.astype(np.float32)),
        mx.array(B.astype(np.float32)),
    )
