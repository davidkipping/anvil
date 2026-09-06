"""Reproducible random-key management on top of ``mx.random``.

MLX uses JAX-style splittable Threefry keys. We derive every key
deterministically from ``(seed, iteration, role)`` so that any single
iteration of a run can be reproduced in isolation (used by the precision
harness and when debugging a misbehaving step) without replaying the chain
of splits that led to it.
"""

from __future__ import annotations

import mlx.core as mx

_GOLDEN = 0x9E3779B97F4A7C15  # 64-bit golden-ratio constant for seed mixing
_MASK64 = (1 << 64) - 1


def _mix(*words: int) -> int:
    """SplitMix64-style avalanche of a tuple of ints into one 64-bit seed."""
    h = _MASK64
    for w in words:
        h = (h ^ (w & _MASK64)) * _GOLDEN & _MASK64
        h ^= h >> 31
        h = h * 0xBF58476D1CE4E5B9 & _MASK64
        h ^= h >> 27
    return h


class KeyStream:
    """Deterministic supplier of ``mx.random`` keys.

    Roles namespace independent random consumers within one iteration
    (proposal noise, momentum draw, acceptance uniform, partner choice, ...).
    """

    PROPOSAL = 0
    ACCEPT = 1
    MOMENTUM = 2
    PARTNER = 3
    INIT = 4

    def __init__(self, seed: int):
        self.seed = int(seed)

    def key(self, iteration: int, role: int = 0) -> mx.array:
        return mx.random.key(_mix(self.seed, iteration, role))

    def init_key(self) -> mx.array:
        return self.key(-1, self.INIT)
