"""Parameter-space transforms and the float32 conditioning discipline.

Three spaces:

* **u** — unbounded sampling space, float32, what kernels see. Kept
  O(1)-conditioned by construction.
* **model space** — the units the user's MLX log-likelihood consumes.
  These must already be well-conditioned float32 units (e.g. transit epoch
  as *days since a reference time you subtracted from your data on the CPU
  in float64*, never absolute BJD ~2.45e6). The bijectors u → model space
  live in the float32 GPU graph and carry their log-Jacobians.
* **reporting space** — true physical units, float64, host-side only:
  ``x = report_offset + report_scale * model_value``. This is where large
  offsets (BJD zero-points, absolute fluxes) are reinstated. It never
  enters the GPU graph.

Sampling in u with the Jacobian terms included means a flat prior over a
bounded model-space parameter is exact; additional priors can be added in
the user's log_prob (model space) as MLX ops.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mlx.core as mx
import numpy as np

from .logdensity import LogDensity


@dataclass
class ParamSpec:
    """One parameter's bounds/scaling, in model-space units.

    Exactly one of these shapes applies:
      * lo and hi both finite  -> bounded, sigmoid-mapped
      * lo finite, hi = +inf   -> lower-bounded, exp-mapped (scaled)
      * lo = -inf, hi finite   -> upper-bounded, mirrored exp
      * both infinite          -> unbounded, affine-mapped via loc/scale

    ``loc``/``scale`` set the u-space conditioning for unbounded and
    half-bounded parameters (guess the posterior location/width to within
    an order of magnitude; adaptation does the rest). ``report_offset`` /
    ``report_scale`` are float64 host-side constants for converting model
    units to physical reporting units.
    """

    name: str
    lo: float = -math.inf
    hi: float = math.inf
    loc: float = 0.0
    scale: float = 1.0
    report_offset: float = 0.0
    report_scale: float = 1.0

    def __post_init__(self):
        if not self.hi > self.lo:
            raise ValueError(f"{self.name}: hi must exceed lo")
        if not self.scale > 0:
            raise ValueError(f"{self.name}: scale must be positive")


class Transform:
    """Batched u -> model-space map with log-Jacobian, plus fp64 reporting."""

    def __init__(self, specs: list[ParamSpec]):
        self.specs = specs
        self.dim = len(specs)
        self.names = [s.name for s in specs]

        lo = np.array([s.lo for s in specs])
        hi = np.array([s.hi for s in specs])
        self._bounded = np.isfinite(lo) & np.isfinite(hi)
        self._lower = np.isfinite(lo) & ~np.isfinite(hi)
        self._upper = ~np.isfinite(lo) & np.isfinite(hi)
        self._free = ~np.isfinite(lo) & ~np.isfinite(hi)

        # float32 in-graph constants (model space is fp32-safe by contract)
        self._lo32 = mx.array(np.where(np.isfinite(lo), lo, 0.0).astype(np.float32))
        self._hi32 = mx.array(np.where(np.isfinite(hi), hi, 0.0).astype(np.float32))
        self._width32 = mx.array(
            np.where(self._bounded, hi - lo, 1.0).astype(np.float32)
        )
        self._loc32 = mx.array(np.array([s.loc for s in specs], dtype=np.float32))
        self._scale32 = mx.array(np.array([s.scale for s in specs], dtype=np.float32))
        for m, arr in (
            ("bounded", self._bounded), ("lower", self._lower),
            ("upper", self._upper), ("free", self._free),
        ):
            setattr(self, f"_m_{m}", mx.array(arr))

        # float64 host-side reporting constants
        self._rep_off = np.array([s.report_offset for s in specs], dtype=np.float64)
        self._rep_scale = np.array([s.report_scale for s in specs], dtype=np.float64)

    # -- in-graph (fp32, batched, differentiable) -------------------------

    def to_model(self, u: mx.array) -> mx.array:
        """(n, dim) unbounded -> (n, dim) model space."""
        sig = mx.sigmoid(u)
        bounded = self._lo32 + self._width32 * sig
        lower = self._lo32 + self._scale32 * mx.exp(u)
        upper = self._hi32 - self._scale32 * mx.exp(-u)
        free = self._loc32 + self._scale32 * u
        out = mx.where(self._m_bounded, bounded, free)
        out = mx.where(self._m_lower, lower, out)
        out = mx.where(self._m_upper, upper, out)
        return out

    def log_det_jac(self, u: mx.array) -> mx.array:
        """(n, dim) -> (n,) log-abs-determinant of d(model)/du, summed
        over dimensions."""
        sig = mx.sigmoid(u)
        # d/du [lo + w*sigmoid(u)] = w * sig * (1 - sig)
        lj_bounded = mx.log(self._width32) + mx.log(sig) + mx.log1p(-sig)
        lj_lower = mx.log(self._scale32) + u        # d/du [lo + s e^u]
        lj_upper = mx.log(self._scale32) - u        # d/du [hi - s e^-u]
        lj_free = mx.log(self._scale32)
        lj = mx.where(self._m_bounded, lj_bounded, mx.broadcast_to(lj_free, u.shape))
        lj = mx.where(self._m_lower, lj_lower, lj)
        lj = mx.where(self._m_upper, lj_upper, lj)
        return mx.sum(lj, axis=-1)

    # -- host-side (fp64, reporting) --------------------------------------

    def to_physical(self, model_values: np.ndarray) -> np.ndarray:
        """Model-space draws (numpy, any shape ending in dim) -> float64
        physical reporting units."""
        v = np.asarray(model_values, dtype=np.float64)
        return self._rep_off + self._rep_scale * v

    def model_np(self, u: np.ndarray) -> np.ndarray:
        """Host-side fp64 replica of ``to_model`` (used for reporting and
        the precision harness)."""
        u = np.asarray(u, dtype=np.float64)
        lo = np.array([s.lo for s in self.specs])
        hi = np.array([s.hi for s in self.specs])
        loc = np.array([s.loc for s in self.specs])
        scale = np.array([s.scale for s in self.specs])
        sig = 1.0 / (1.0 + np.exp(-u))
        out = np.where(self._bounded, np.where(np.isfinite(lo), lo, 0.0)
                       + np.where(self._bounded, hi - lo, 1.0) * sig,
                       loc + scale * u)
        out = np.where(self._lower, np.where(np.isfinite(lo), lo, 0.0)
                       + scale * np.exp(u), out)
        out = np.where(self._upper, np.where(np.isfinite(hi), hi, 0.0)
                       - scale * np.exp(-u), out)
        return out

    def log_det_jac_np(self, u: np.ndarray) -> np.ndarray:
        """Host-side fp64 replica of ``log_det_jac`` (precision harness)."""
        u = np.asarray(u, dtype=np.float64)
        lo = np.array([s.lo for s in self.specs])
        hi = np.array([s.hi for s in self.specs])
        scale = np.array([s.scale for s in self.specs])
        width = np.where(self._bounded, hi - lo, 1.0)
        # log(sigmoid(u)) and log(1-sigmoid(u)), computed stably
        log_sig = -np.logaddexp(0.0, -u)
        log_1msig = -np.logaddexp(0.0, u)
        lj = np.where(self._bounded, np.log(width) + log_sig + log_1msig,
                      np.log(scale))
        lj = np.where(self._lower, np.log(scale) + u, lj)
        lj = np.where(self._upper, np.log(scale) - u, lj)
        return lj.sum(axis=-1)

    def from_model_np(self, v: np.ndarray) -> np.ndarray:
        """Host-side inverse of ``to_model`` (for building initial states
        from model-space guesses)."""
        v = np.asarray(v, dtype=np.float64)
        lo = np.array([s.lo for s in self.specs])
        hi = np.array([s.hi for s in self.specs])
        loc = np.array([s.loc for s in self.specs])
        scale = np.array([s.scale for s in self.specs])
        with np.errstate(divide="ignore", invalid="ignore"):
            p = (v - lo) / np.where(self._bounded, hi - lo, 1.0)
            u_bounded = np.log(p) - np.log1p(-p)
            u_lower = np.log(np.maximum(v - lo, 1e-300) / scale)
            u_upper = -np.log(np.maximum(hi - v, 1e-300) / scale)
        u_free = (v - loc) / scale
        out = np.where(self._bounded, u_bounded, u_free)
        out = np.where(self._lower, u_lower, out)
        out = np.where(self._upper, u_upper, out)
        return out


class TransformedLogDensity(LogDensity):
    """Wrap a model-space log density into an unbounded-space LogDensity.

    ``model_log_prob`` is the user's batched MLX function over model-space
    parameters ``(n, dim) -> (n,)`` (likelihood + any explicit priors);
    the Jacobian of the u -> model map is added automatically.
    """

    def __init__(self, model_log_prob, transform: Transform,
                 supports_grad: bool = True, model_log_prob_hi=None):
        self._fn = model_log_prob
        self._fn_hi = model_log_prob_hi
        self.transform = transform
        self.dim = transform.dim
        self.supports_grad = supports_grad

    def log_prob(self, u: mx.array) -> mx.array:
        v = self.transform.to_model(u)
        return self._fn(v) + self.transform.log_det_jac(u)

    @property
    def has_hi(self) -> bool:
        return self._fn_hi is not None

    def log_prob_hi(self, u: mx.array) -> mx.array:
        """Float64 CPU-path evaluation (requires ``model_log_prob_hi``)."""
        if self._fn_hi is None:
            raise NotImplementedError(
                "no float64 path: pass model_log_prob_hi= to "
                "TransformedLogDensity (e.g. ChunkedGaussianLogLike.hi)"
            )
        u_np = np.array(u, dtype=np.float64)
        with mx.stream(mx.cpu):
            # dtype= is REQUIRED (see precision.py): without it the
            # float64 parameters and Jacobian silently round to float32.
            v64 = mx.array(self.transform.model_np(u_np), dtype=mx.float64)
            lp = self._fn_hi(v64)
            jac = mx.array(self.transform.log_det_jac_np(u_np),
                           dtype=mx.float64)
            return lp + jac
