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
from .priors import ImproperFlat, LogUniform, Prior, Uniform


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
    #: a :class:`~anvil.priors.Prior`, in model units (the units of lo/hi).
    #: None means the implied default: Uniform on a finite box, and an
    #: improper flat density otherwise. Declared last so positional
    #: construction (``ParamSpec("q1", 0.0, 1.0)``) keeps working.
    prior: Prior | None = None

    def __post_init__(self):
        if not self.hi > self.lo:
            raise ValueError(f"{self.name}: hi must exceed lo")
        if not self.scale > 0:
            raise ValueError(f"{self.name}: scale must be positive")
        if self.prior is not None:
            if isinstance(self.prior, LogUniform) and self.report_offset != 0.0:
                raise ValueError(
                    f"{self.name}: LogUniform is a prior on the model quantity "
                    f"itself and cannot sit on an offset "
                    f"(report_offset={self.report_offset})")
            self.prior.bind(self.lo, self.hi, self.name)   # validate now


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
        # Centre/half-width form of the bounded map. Algebraically
        # identical to lo + (hi-lo)*sigmoid(u) (since tanh(u/2) = 2*sigmoid(u)-1)
        # but its float32 error is relative to the parameter's distance from
        # the box CENTRE rather than absolute in the box WIDTH — the latter
        # dominated anvil's total float32 error budget for parameters whose
        # posterior sits well inside a wide box.
        # finite stand-ins first: (-inf + inf) is nan even when np.where
        # would discard it
        _lo_f = np.where(self._bounded, lo, 0.0)
        _hi_f = np.where(self._bounded, hi, 1.0)
        self._mid32 = mx.array(((_lo_f + _hi_f) / 2.0).astype(np.float32))
        self._half32 = mx.array(((_hi_f - _lo_f) / 2.0).astype(np.float32))
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

        # Priors, imposed by addition: each spec's map is unchanged, and the
        # log-posterior gains the prior's v-dependent term in the graph plus a
        # float64 host normaliser outside it. See anvil.priors.
        self.priors: list[Prior] = [self._resolve_prior(s) for s in specs]
        #: True when every parameter has a normalisable prior (no improper
        #: flat density on an unbounded or half-bounded spec)
        self.is_proper: bool = all(p.is_proper for p in self.priors)
        #: sum of the priors' float64 normalising constants. Adding it (and
        #: the likelihood's own constants) to a stored log_prob gives the
        #: absolute log posterior density. Meaningful only when is_proper.
        self.log_prior_norm: float = float(sum(p.log_norm for p in self.priors))
        self._prior_cols = [i for i, p in enumerate(self.priors)
                            if p.has_in_graph_term]
        #: False when no prior adds anything to the graph (only flat ones):
        #: then TransformedLogDensity's graph is exactly what it was before
        #: priors existed, and so are its draws.
        self.has_prior_terms: bool = bool(self._prior_cols)

    @staticmethod
    def _resolve_prior(spec: ParamSpec) -> Prior:
        if spec.prior is None:
            if math.isfinite(spec.lo) and math.isfinite(spec.hi):
                return Uniform().bind(spec.lo, spec.hi, spec.name)
            return ImproperFlat().bind(spec.lo, spec.hi, spec.name)
        return spec.prior.bind(spec.lo, spec.hi, spec.name)

    # -- priors -----------------------------------------------------------

    def log_prior(self, v: mx.array) -> mx.array:
        """(n, dim) model space -> (n,): the v-dependent part of the log
        prior, in the float32 graph (dtype follows ``v``). Zero when
        ``has_prior_terms`` is False -- callers skip it then."""
        out = None
        for i in self._prior_cols:
            t = self.priors[i].log_pdf(v[:, i])
            out = t if out is None else out + t
        return out if out is not None else mx.zeros(v.shape[:1], dtype=v.dtype)

    def log_prior_np(self, v: np.ndarray) -> np.ndarray:
        """Float64 twin of :meth:`log_prior`, summed over every parameter
        (so ``-inf`` outside any prior's support)."""
        v = np.atleast_2d(np.asarray(v, dtype=np.float64))
        return sum(p.log_pdf_np(v[:, i]) for i, p in enumerate(self.priors))

    def prior_transform(self, cube: np.ndarray) -> np.ndarray:
        """Map points of the unit hypercube ``(n, dim)`` to model space
        through each prior's inverse CDF -- the interface nested samplers
        (MultiNest, dynesty) expect, and how :meth:`sample_prior` draws.
        Requires every prior to be proper."""
        if not self.is_proper:
            bad = [s.name for s, p in zip(self.specs, self.priors) if not p.is_proper]
            raise ValueError(f"improper prior on {bad}: give those parameters "
                             "a finite box or an explicit prior")
        cube = np.atleast_2d(np.asarray(cube, dtype=np.float64))
        return np.stack([p.ppf_np(cube[:, i]) for i, p in enumerate(self.priors)],
                        axis=-1)

    def sample_prior_model(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """``n`` float64 prior draws in model space, ``(n, dim)``."""
        v = self.prior_transform(rng.uniform(size=(n, self.dim)))
        # keep draws strictly inside each box, so the inverse map to u is
        # finite (a uniform draw of exactly 0 would map to u = -inf)
        lo = np.array([s.lo for s in self.specs]); hi = np.array([s.hi for s in self.specs])
        return np.clip(v, np.nextafter(lo, np.inf), np.nextafter(hi, -np.inf))

    def sample_prior(self, n: int, rng: np.random.Generator) -> mx.array:
        """``n`` prior draws mapped to u-space, as a float32 ``mx.array``
        ready for ``run(u0=...)``. Not used as a default anywhere: a prior
        draw is rarely a good start for a data-dominated posterior."""
        u = self.from_model_np(self.sample_prior_model(n, rng))
        return mx.array(u.astype(np.float32))

    @property
    def prior_mean(self) -> np.ndarray:
        return np.array([p.mean for p in self.priors], dtype=np.float64)

    @property
    def prior_sd(self) -> np.ndarray:
        return np.array([p.sd for p in self.priors], dtype=np.float64)

    # -- in-graph (fp32, batched, differentiable) -------------------------

    def to_model(self, u: mx.array) -> mx.array:
        """(n, dim) unbounded -> (n, dim) model space."""
        bounded = self._mid32 + self._half32 * mx.tanh(0.5 * u)
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
        # d/du [lo + w*sigmoid(u)] = w * sig * (1 - sig), whose log is
        # log(w) + log(sig) + log1p(-sig). Written that way it is -inf for
        # |u| >= 18, because mx.sigmoid(18) is exactly 1.0 in float32 — and
        # a parameter pinned against its bound is then unrecoverable, since
        # every proposal out of -inf reads as a divergence. The identity
        #     log(sig) + log1p(-sig) == -|u| - 2*log1p(exp(-|u|))
        # is finite and accurate for every float32 u (and more accurate
        # than the sigmoid form well before it saturates: at u = 15 it
        # gives -15.0000 against -15.0261). The host float64 replica in
        # log_det_jac_np already used the equivalent logaddexp form.
        au = mx.abs(u)
        lj_bounded = (mx.log(self._width32) - au
                      - 2.0 * mx.log1p(mx.exp(-au)))
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
                 supports_grad: bool = True, model_log_prob_hi=None,
                 model_log_norm: float = 0.0):
        self._fn = model_log_prob
        self._fn_hi = model_log_prob_hi
        self.transform = transform
        self.dim = transform.dim
        self.supports_grad = supports_grad
        #: float64 host constant the likelihood keeps out of its graph (e.g.
        #: a ChunkedGaussianLogLike's log_offset_const + log_norm_const)
        self.model_log_norm = float(model_log_norm)

    def log_prob(self, u: mx.array) -> mx.array:
        v = self.transform.to_model(u)
        lp = self._fn(v) + self.transform.log_det_jac(u)
        # appended only when some prior has a v-dependent term: with flat
        # priors alone the graph -- and so every draw -- is exactly as before
        if self.transform.has_prior_terms:
            lp = lp + self.transform.log_prior(v)
        return lp

    def log_likelihood(self, u: mx.array) -> mx.array:
        """``model_log_prob`` alone at ``u``: no Jacobian and no declared
        prior. This is what :func:`anvil.optimize` maximises for
        ``objective="mle"``. A prior written into ``model_log_prob`` itself,
        rather than declared on a :class:`ParamSpec`, is part of it."""
        return self._fn(self.transform.to_model(u))

    def log_likelihood_hi(self, u: mx.array) -> mx.array:
        """Float64 CPU-path twin of :meth:`log_likelihood` (requires
        ``model_log_prob_hi``)."""
        if self._fn_hi is None:
            raise NotImplementedError(
                "no float64 path: pass model_log_prob_hi= to "
                "TransformedLogDensity (e.g. ChunkedGaussianLogLike.hi)"
            )
        u_np = np.array(u, dtype=np.float64)
        with mx.stream(mx.cpu):
            return self._fn_hi(mx.array(self.transform.model_np(u_np),
                                        dtype=mx.float64))

    @property
    def log_norm_const(self) -> float:
        """Float64 constant that turns a stored log_prob into the absolute
        log posterior density: ``res.get_log_prob() + target.log_norm_const``
        = log L(v) + log p(v) + log|dv/du|. The priors' normalisers plus
        ``model_log_norm``; meaningful only when :attr:`is_proper`."""
        return self.transform.log_prior_norm + self.model_log_norm

    @property
    def is_proper(self) -> bool:
        return self.transform.is_proper

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
            v_np = self.transform.model_np(u_np)
            v64 = mx.array(v_np, dtype=mx.float64)
            lp = self._fn_hi(v64)
            jac = mx.array(self.transform.log_det_jac_np(u_np),
                           dtype=mx.float64)
            out = lp + jac
            if self.transform.has_prior_terms:
                # computed in float64 from the float64 v, never cast up
                # from float32, so the precision harness compares like
                # with like
                out = out + mx.array(self.transform.log_prior_np(v_np),
                                     dtype=mx.float64)
            return out
