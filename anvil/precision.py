"""Float32 conditioning tools: chunked reductions, fp64 anchoring, and the
precision validation harness.

The strategy: per-datum log-likelihood terms are computed in float32 on the
GPU, summed chunk-wise (a two-level tree, error O(eps*sqrt(n_chunks)) rather
than O(eps*n) for naive accumulation). Optionally the cross-chunk sum runs
in float64 on the CPU stream *inside the same lazy graph* ("fp64_anchor").
The default keeps the hot compiled step pure-fp32 and instead re-anchors
the *cached* log_prob against a full-fp64 CPU evaluation every K iterations
(the engine's ``reanchor_every``), so rounding drift cannot accumulate in
the Markov chain's accept/reject bookkeeping.

``validate_precision`` is the user-facing trust feature: it evaluates the
production fp32 path and a full-fp64 CPU path at given points and reports
the discrepancy in units that matter (the ~1-unit scale of Metropolis
accept decisions).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import mlx.core as mx
import numpy as np


@dataclass
class PrecisionPolicy:
    #: "fp32_tree": chunked fp32 sums only (compilable, default).
    #: "fp64_anchor": cross-chunk sum in fp64 on the CPU stream (uncompiled).
    reduction: Literal["fp32_tree", "fp64_anchor"] = "fp32_tree"
    chunk_size: int = 65536
    #: engine-level cadence for re-anchoring cached log_prob in fp64; 0 = off
    reanchor_every: int = 100


def chunked_sum(
    term_fn: Callable[[int, int], mx.array],
    n_data: int,
    policy: PrecisionPolicy,
) -> mx.array:
    """Sum per-datum terms over the data axis with controlled rounding.

    ``term_fn(start, stop)`` returns the fp32 terms for one data chunk,
    shape (n_chains, stop - start). Returns (n_chains,) fp32.
    """
    c = policy.chunk_size
    partials = [
        mx.sum(term_fn(s, min(s + c, n_data)), axis=-1)
        for s in range(0, n_data, c)
    ]
    if len(partials) == 1:
        stacked = partials[0][:, None]
    else:
        stacked = mx.stack(partials, axis=-1)  # (n_chains, n_chunks)
    if policy.reduction == "fp64_anchor":
        s64 = mx.sum(
            stacked.astype(mx.float64, stream=mx.cpu), axis=-1, stream=mx.cpu
        )
        return s64.astype(mx.float32, stream=mx.cpu)
    return mx.sum(stacked, axis=-1)


class ChunkedGaussianLogLike:
    """Independent-Gaussian log-likelihood over a large dataset, chunked.

    ``model_fn(v, x)`` is the user's batched, dtype-polymorphic MLX model:
    parameters ``v`` (n_chains, dim) and abscissa ``x`` (m,) — or (c, m) for
    multi-channel abscissae such as (epoch-centered time, orbit number) —
    -> predictions (n_chains, m). It must use only MLX ops on its arguments
    so the same code runs fp32-on-GPU (production) and fp64-on-CPU
    (anchor/validation).

    Data enter as float64 numpy (the caller has already centered/scaled
    them into well-conditioned model units); fp32 copies feed the GPU path.
    The parameter-independent normalization sum(log(2*pi*sigma^2))/2 is kept
    as a float64 host scalar, available as ``.log_norm_const`` (it cancels
    in MH ratios and is deliberately excluded from the graph).
    """

    def __init__(
        self,
        model_fn: Callable[[mx.array, mx.array], mx.array],
        x: np.ndarray,
        y: np.ndarray,
        yerr: np.ndarray,
        policy: PrecisionPolicy | None = None,
    ):
        self.model_fn = model_fn
        self.policy = policy or PrecisionPolicy()
        self._x64 = np.asarray(x, dtype=np.float64)
        self._y64 = np.asarray(y, dtype=np.float64)
        self._yerr64 = np.asarray(yerr, dtype=np.float64)
        self.n_data = self._y64.size

        self._x32 = mx.array(self._x64.astype(np.float32))
        self._y32 = mx.array(self._y64.astype(np.float32))
        self._w32 = mx.array((1.0 / self._yerr64).astype(np.float32))

        self.log_norm_const = float(
            -0.5 * np.sum(np.log(2.0 * np.pi * self._yerr64**2))
        )

    def __call__(self, v: mx.array) -> mx.array:
        """fp32 chunked chi-squared log-likelihood (n_chains,)."""

        def term_fn(s: int, e: int) -> mx.array:
            m = self.model_fn(v, self._x32[..., s:e])
            r = (self._y32[s:e] - m) * self._w32[s:e]
            return -0.5 * r * r

        return chunked_sum(term_fn, self.n_data, self.policy)

    #: chain-block size for the float64 path; bounds peak memory, since the
    #: fp64 CPU graph can hold dozens of (block, chunk)-sized intermediates
    hi_chain_block: int = 64

    def hi(self, v: mx.array) -> mx.array:
        """Full-float64 CPU-stream evaluation of the same likelihood.

        Processed in (chain-block x data-chunk) tiles with an eval per
        tile: an expensive model's fp64 graph over all chains x all data
        at once can transiently allocate tens of GB.
        """
        n_chains = v.shape[0]
        c = self.policy.chunk_size
        out = np.zeros(n_chains, dtype=np.float64)
        with mx.stream(mx.cpu):
            v64 = v.astype(mx.float64)
            for cb in range(0, n_chains, self.hi_chain_block):
                vb = v64[cb : cb + self.hi_chain_block]
                acc = mx.zeros(vb.shape[:1], dtype=mx.float64)
                for s in range(0, self.n_data, c):
                    e = min(s + c, self.n_data)
                    # dtype= is REQUIRED: mx.array() of a float64 numpy
                    # array silently yields float32, which would make this
                    # "float64 path" a second float32 path.
                    x = mx.array(self._x64[..., s:e], dtype=mx.float64)
                    y = mx.array(self._y64[s:e], dtype=mx.float64)
                    w = mx.array(1.0 / self._yerr64[s:e], dtype=mx.float64)
                    r = (y - self.model_fn(vb, x)) * w
                    acc = acc - 0.5 * mx.sum(r * r, axis=-1)
                    mx.eval(acc)  # free this tile's graph before the next
                out[cb : cb + vb.shape[0]] = np.array(acc)
            return mx.array(out, dtype=mx.float64)


@dataclass
class PrecisionReport:
    n_points: int
    median_abs_err: float
    max_abs_err: float
    median_logl_magnitude: float

    def __str__(self) -> str:
        if self.max_abs_err < 0.1:
            verdict = ("OK: fp32 error is far below the ~1-unit scale of "
                       "Metropolis accept decisions")
        elif self.max_abs_err < 1.0:
            verdict = ("ACCEPTABLE: fp32 error is below the ~1-unit scale of "
                       "Metropolis accept decisions; residual distortion of "
                       "the sampled posterior is small. Errors in the "
                       "log-density *difference* between nearby states (what "
                       "accept/reject actually uses) are typically smaller "
                       "still.")
        else:
            verdict = ("WARNING: fp32 error reaches the scale of Metropolis "
                       "accept decisions — improve model conditioning "
                       "(offset parameters/data as in the docs) and/or "
                       "enable the fp64_anchor reduction")
        return (
            f"PrecisionReport over {self.n_points} states\n"
            f"  |logL| typical magnitude : {self.median_logl_magnitude:.4g}\n"
            f"  |fp32 - fp64| median     : {self.median_abs_err:.4g}\n"
            f"  |fp32 - fp64| max        : {self.max_abs_err:.4g}\n"
            f"  {verdict}"
        )


def validate_precision(target, u: mx.array) -> PrecisionReport:
    """Compare the production fp32 log_prob path against the fp64 path.

    ``target`` must expose ``log_prob(u)`` (fp32 production path) and
    ``log_prob_hi(u)`` (float64 CPU path). ``u``: (n_points, dim) states.
    """
    lp32 = np.array(target.log_prob(u), dtype=np.float64)
    lp64 = np.array(target.log_prob_hi(u), dtype=np.float64)
    err = np.abs(lp32 - lp64)
    return PrecisionReport(
        n_points=int(u.shape[0]),
        median_abs_err=float(np.median(err)),
        max_abs_err=float(err.max()),
        median_logl_magnitude=float(np.median(np.abs(lp64))),
    )
