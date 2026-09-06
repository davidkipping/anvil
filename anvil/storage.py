"""Chain storage backends.

Layout is ``(n_kept, n_chains, dim)`` float32 for positions plus
``(n_kept, n_chains)`` for log-prob. With thousands of chains the honest
default is few kept steps per chain (the many-short-chains workflow this
hardware favors); thinning is first-class.
"""

from __future__ import annotations

import numpy as np


class MemoryBackend:
    """Preallocated host-side numpy storage."""

    def __init__(self):
        self._u: np.ndarray | None = None
        self._lp: np.ndarray | None = None
        self._n = 0

    def reserve(self, n_kept: int, n_chains: int, dim: int) -> None:
        self._u = np.empty((n_kept, n_chains, dim), dtype=np.float32)
        self._lp = np.empty((n_kept, n_chains), dtype=np.float32)
        self._n = 0

    def append(self, u: np.ndarray, log_prob: np.ndarray) -> None:
        if self._u is None:
            raise RuntimeError("MemoryBackend.reserve() was never called")
        self._u[self._n] = u
        self._lp[self._n] = log_prob
        self._n += 1

    @property
    def n_kept(self) -> int:
        return self._n

    def get_chain(self, discard: int = 0, thin: int = 1, flat: bool = False) -> np.ndarray:
        out = self._u[discard : self._n : thin]
        if flat:
            return out.reshape(-1, out.shape[-1])
        return out

    def get_log_prob(self, discard: int = 0, thin: int = 1, flat: bool = False) -> np.ndarray:
        out = self._lp[discard : self._n : thin]
        if flat:
            return out.reshape(-1)
        return out


class NpyStreamBackend(MemoryBackend):
    """Storage streamed to memory-mapped ``.npy`` files on disk.

    Same interface as MemoryBackend; use when
    ``n_kept * n_chains * dim * 4`` bytes exceeds comfortable RAM. Files
    ``<prefix>_chain.npy`` and ``<prefix>_log_prob.npy`` are standard numpy
    files readable later with ``np.load(..., mmap_mode='r')``.
    """

    def __init__(self, prefix: str):
        super().__init__()
        self.prefix = str(prefix)

    def reserve(self, n_kept: int, n_chains: int, dim: int) -> None:
        self._u = np.lib.format.open_memmap(
            f"{self.prefix}_chain.npy", mode="w+", dtype=np.float32,
            shape=(n_kept, n_chains, dim),
        )
        self._lp = np.lib.format.open_memmap(
            f"{self.prefix}_log_prob.npy", mode="w+", dtype=np.float32,
            shape=(n_kept, n_chains),
        )
        self._n = 0

    def flush(self) -> None:
        if self._u is not None:
            self._u.flush()
            self._lp.flush()
