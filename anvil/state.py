"""Batched chain state and small pytree helpers.

The chain state is a flat ``dict[str, mx.array]`` whose arrays all carry the
chain dimension first:

    "u"        (n_chains, dim)  float32 — positions in unbounded sampling space
    "log_prob" (n_chains,)      float32 — cached log density at "u"

Kernels may stash extra entries (e.g. ChEES caches "grad"). Keeping the state
a plain dict keeps it directly compatible with ``mx.compile`` and ``mx.eval``.
"""

from __future__ import annotations

import mlx.core as mx

ChainState = dict[str, mx.array]


def tree_where(mask: mx.array, new: ChainState, old: ChainState) -> ChainState:
    """Per-chain select between two states. ``mask`` has shape (n_chains,)."""
    out: ChainState = {}
    for k, new_v in new.items():
        old_v = old[k]
        m = mask.reshape((-1,) + (1,) * (new_v.ndim - 1))
        out[k] = mx.where(m, new_v, old_v)
    return out


def tree_eval(*trees: ChainState) -> None:
    """Force evaluation of every array in the given states."""
    arrays = [v for t in trees for v in t.values()]
    mx.eval(*arrays)
