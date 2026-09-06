"""Base-2 van der Corput (1-D Halton) sequence.

ChEES-HMC jitters the trajectory length with a *shared per-iteration scalar*
drawn from this low-discrepancy sequence rather than a pseudo-random uniform:
the sequence fills [0, 1) evenly so short warmups still explore a balanced set
of trajectory lengths, and every chain uses the same value each iteration,
which keeps the leapfrog count identical across chains (SIMD-friendly).
"""

from __future__ import annotations


def van_der_corput(n: int) -> float:
    """n-th element (n >= 1) of the base-2 van der Corput sequence.

    van_der_corput(1..4) == 1/2, 1/4, 3/4, 1/8.
    """
    if n < 1:
        raise ValueError(f"van der Corput index must be >= 1, got {n}")
    v = 0.0
    denom = 1.0
    while n:
        denom *= 2.0
        n, rem = divmod(n, 2)
        v += rem / denom
    return v


def halton_jitter(iteration: int) -> float:
    """Trajectory-length jitter in [0, 1) for a 0-based iteration counter."""
    return van_der_corput(iteration + 1)
