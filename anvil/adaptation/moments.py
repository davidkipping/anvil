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
    ms: DenseMomentsState, u: np.ndarray, t: int
) -> DenseMomentsState:
    """EMA update from the current (n_chains, dim) positions, same
    ``beta = t/(t+8)`` schedule as the diagonal path. ``u`` is host numpy."""
    beta = t / (t + 8.0)
    u = np.asarray(u, dtype=np.float64)
    return DenseMomentsState(
        m1=beta * ms.m1 + (1.0 - beta) * u.mean(axis=0),
        m2=beta * ms.m2 + (1.0 - beta) * (u.T @ u) / u.shape[0],
    )


def dense_factors(ms: DenseMomentsState, ridge: float = 1e-6):
    """Factor the EMA covariance into the arrays the kernel consumes.

    Returns ``(sd, R, B)`` as float32 mx arrays, where Sigma = S R S and
    B = inv(chol(R)). The momentum draw uses B via ``(z @ B) / sd``, which
    has covariance ``B^T B / (sd sd^T) = Sigma^-1`` as required.

    Raises ``numpy.linalg.LinAlgError`` if R is not positive definite at
    this ridge, so the caller can escalate or fall back.
    """
    cov = ms.m2 - np.outer(ms.m1, ms.m1)
    sd = np.sqrt(np.maximum(np.diag(cov), 1e-30))
    R = cov / np.outer(sd, sd)
    R = 0.5 * (R + R.T)                       # kill asymmetry from rounding
    np.fill_diagonal(R, 1.0)
    Lr = np.linalg.cholesky(R + ridge * np.eye(len(sd)))
    B = np.linalg.inv(Lr)
    return (
        mx.array(sd.astype(np.float32)),
        mx.array(R.astype(np.float32)),
        mx.array(B.astype(np.float32)),
    )
