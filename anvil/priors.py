"""Explicit, normalised priors for single parameters.

A prior is declared on a parameter's :class:`~anvil.transforms.ParamSpec`::

    ParamSpec("a", lo=1.5, hi=200.0, prior=LogUniform())
    ParamSpec("p_off", prior=Gaussian(0.0, 3e-4))

and is imposed by **addition**: the sampler keeps walking the usual unbounded
``u`` (through the spec's existing map), and the log-posterior gains the
prior's log-density. Each prior splits that density in two, following anvil's
float32 discipline:

* ``log_pdf(v)`` -- the part that depends on ``v``, evaluated in the float32
  GPU graph (zero, and absent from the graph, for a flat prior);
* ``log_norm`` -- the normalising constant, a float64 host scalar that never
  enters the graph, exactly like ``ChunkedGaussianLogLike.log_norm_const``.

``log p(v) = log_pdf(v) + log_norm`` on the support, ``-inf`` outside it.

**The ParamSpec box is the support.** A prior with explicit bounds must match
the box; with none it adopts the box. A Gaussian on a bounded or half-bounded
spec therefore *is* a truncated Gaussian: the in-graph term is unchanged, and
only ``log_norm`` gains ``-log Z`` (Z = the Gaussian mass inside the box). A
Gaussian on an unbounded spec is a plain Gaussian.

**Priors are in model units** -- the units of ``lo``/``hi``, i.e. the offset
quantity the model samples, not the physical reporting units. Use
:meth:`Gaussian.from_physical` to convert.

Why addition rather than an inverse-CDF ("prior transform") map from the unit
hypercube, as nested samplers use: both sample the same posterior, but HMC
pays for the *shape* of the space it walks. Measured with ChEES, a Gaussian
prior that dominates the data needs 5 leapfrog steps per draw through the
inverse-CDF map against 1 by addition (the u-space posterior becomes
logistic-tailed rather than Gaussian), and a LogUniform prior 0.55-0.80x the
efficiency. Every prior still carries its inverse CDF (``ppf_np``), so
:meth:`~anvil.transforms.Transform.prior_transform` provides the
hypercube interface for prior draws and nested-sampling cross-checks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import mlx.core as mx
import numpy as np

from .diagnostics import norm_ppf

_LOG_SQRT_2PI = 0.5 * math.log(2.0 * math.pi)
_SQRT2 = math.sqrt(2.0)


def _same(a: float, b: float) -> bool:
    if math.isinf(a) or math.isinf(b):
        return a == b
    return math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-300)


def _check_box(prior, plo, phi, lo, hi, name) -> None:
    if not (_same(plo, lo) and _same(phi, hi)):
        raise ValueError(
            f"{name}: {type(prior).__name__} has support ({plo}, {phi}) but "
            f"the ParamSpec box is ({lo}, {hi}). The box is the prior's "
            f"support: set lo/hi to match, or omit the prior's bounds.")


def _phi(x: float) -> float:
    """Standard normal density; 0 at +-inf."""
    return 0.0 if math.isinf(x) else math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _normal_mass(alpha: float, beta: float) -> float:
    """Phi(beta) - Phi(alpha), using whichever tail avoids cancellation."""
    if alpha > 0.0:          # both in the upper tail: Q(alpha) - Q(beta)
        return 0.5 * (math.erfc(alpha / _SQRT2) - math.erfc(beta / _SQRT2))
    if beta < 0.0:           # both in the lower tail
        return 0.5 * (math.erfc(-beta / _SQRT2) - math.erfc(-alpha / _SQRT2))
    return 0.5 * (math.erf(beta / _SQRT2) - math.erf(alpha / _SQRT2))


@dataclass(frozen=True)
class Prior:
    """Base class: a normalised prior on one scalar parameter.

    Subclasses provide ``lo``/``hi`` (support, model units), ``is_proper``,
    ``has_in_graph_term``, ``log_norm`` (float64 host), ``log_pdf(v)`` (MLX,
    elementwise, dtype-preserving), ``log_pdf_np(v)``, ``ppf_np(x)`` (inverse
    CDF on [0, 1]), ``mean``, ``sd`` and ``bind(lo, hi, name)``, which checks
    the prior against a ParamSpec box and returns the prior actually in force.
    All constants are stored as Python floats: MLX keeps an array's dtype
    when combined with a Python float, so ``log_pdf`` is float32 on the GPU
    and float64 on the CPU stream with no casting (a numpy scalar would
    instead force evaluation and break ``mx.compile``)."""

    is_proper = True
    has_in_graph_term = False

    def log_pdf(self, v: mx.array) -> mx.array:
        return mx.zeros_like(v)

    def log_density_np(self, v: np.ndarray) -> np.ndarray:
        """The full normalised log-density, float64: log_pdf_np + log_norm."""
        return self.log_pdf_np(v) + self.log_norm

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """``n`` float64 draws in model space, by inverse-CDF sampling."""
        return self.ppf_np(rng.uniform(size=n))


@dataclass(frozen=True)
class ImproperFlat(Prior):
    """A flat density with no normalisation: the implied prior of an
    unbounded or half-bounded ParamSpec that declares none. Sampling is fine;
    an evidence is undefined, which ``Transform.is_proper`` reports."""

    lo: float = -math.inf
    hi: float = math.inf
    is_proper = False
    log_norm = 0.0

    def bind(self, lo, hi, name):
        if math.isfinite(lo) and math.isfinite(hi):
            raise ValueError(f"{name}: the box ({lo}, {hi}) is finite; "
                             "use Uniform() for a flat prior on it")
        return ImproperFlat(float(lo), float(hi))

    def log_pdf_np(self, v):
        v = np.asarray(v, dtype=np.float64)
        return np.where((v >= self.lo) & (v <= self.hi), 0.0, -np.inf)

    def ppf_np(self, x):
        raise ValueError("an improper prior has no draws or inverse CDF")

    def sample(self, n, rng):
        raise ValueError("an improper prior has no draws or inverse CDF")
    mean = property(lambda self: math.nan)
    sd = property(lambda self: math.nan)


@dataclass(frozen=True)
class Uniform(Prior):
    """Flat on ``(lo, hi)``; bounds default to the ParamSpec box, which must
    be finite. No in-graph term: anvil's bounded map plus its Jacobian already
    give a flat density, so a Uniform prior changes only the host constant
    ``-log(hi - lo)`` -- and draws are bit-identical to declaring nothing."""

    lo: float | None = None
    hi: float | None = None
    log_norm: float = field(init=False, default=math.nan)

    def __post_init__(self):
        if self.lo is not None and self.hi is not None:
            if not (math.isfinite(self.lo) and math.isfinite(self.hi)):
                raise ValueError("Uniform needs finite bounds")
            if not self.hi > self.lo:
                raise ValueError("Uniform needs hi > lo")
            object.__setattr__(self, "log_norm", -math.log(self.hi - self.lo))

    def bind(self, lo, hi, name):
        if not (math.isfinite(lo) and math.isfinite(hi)):
            raise ValueError(f"{name}: Uniform needs a finite box, got ({lo}, {hi})")
        if self.lo is not None or self.hi is not None:
            _check_box(self, self.lo if self.lo is not None else lo,
                       self.hi if self.hi is not None else hi, lo, hi, name)
        return Uniform(float(lo), float(hi))

    def log_pdf_np(self, v):
        v = np.asarray(v, dtype=np.float64)
        return np.where((v >= self.lo) & (v <= self.hi), 0.0, -np.inf)

    def ppf_np(self, x):
        return self.lo + (self.hi - self.lo) * np.asarray(x, dtype=np.float64)

    @property
    def mean(self):
        return 0.5 * (self.lo + self.hi)

    @property
    def sd(self):
        return (self.hi - self.lo) / math.sqrt(12.0)


@dataclass(frozen=True)
class LogUniform(Prior):
    """Jeffreys prior on a scale: ``p(v) ∝ 1/v`` on ``(lo, hi)`` with
    ``0 < lo < hi < inf``; bounds default to the ParamSpec box."""

    lo: float | None = None
    hi: float | None = None
    has_in_graph_term = True
    log_norm: float = field(init=False, default=math.nan)

    def __post_init__(self):
        if self.lo is not None and self.hi is not None:
            if not (self.lo > 0.0):
                raise ValueError(f"LogUniform needs lo > 0, got {self.lo}")
            if not math.isfinite(self.hi):
                raise ValueError("LogUniform needs a finite hi")
            if not self.hi > self.lo:
                raise ValueError("LogUniform needs hi > lo")
            object.__setattr__(self, "log_norm",
                               -math.log(math.log(self.hi / self.lo)))

    def bind(self, lo, hi, name):
        if not (lo > 0.0):
            raise ValueError(f"{name}: LogUniform needs lo > 0, got box lo={lo}")
        if not math.isfinite(hi):
            raise ValueError(f"{name}: LogUniform needs a finite hi, got box hi={hi}")
        if self.lo is not None or self.hi is not None:
            _check_box(self, self.lo if self.lo is not None else lo,
                       self.hi if self.hi is not None else hi, lo, hi, name)
        return LogUniform(float(lo), float(hi))

    def log_pdf(self, v):
        return -mx.log(v)

    def log_pdf_np(self, v):
        v = np.asarray(v, dtype=np.float64)
        inside = (v >= self.lo) & (v <= self.hi)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(inside, -np.log(np.where(inside, v, 1.0)), -np.inf)

    def ppf_np(self, x):
        return self.lo * np.exp(np.asarray(x, dtype=np.float64)
                                * math.log(self.hi / self.lo))

    @property
    def mean(self):
        return (self.hi - self.lo) / math.log(self.hi / self.lo)

    @property
    def sd(self):
        L = math.log(self.hi / self.lo)
        return math.sqrt((self.hi ** 2 - self.lo ** 2) / (2.0 * L) - self.mean ** 2)


@dataclass(frozen=True)
class Gaussian(Prior):
    """Normal(mu, sigma) in model units. On an unbounded ParamSpec it is a
    plain Gaussian; on a bounded or half-bounded one, :meth:`bind` returns the
    :class:`TruncatedGaussian` that the box makes it, with the right
    normalising constant."""

    mu: float = 0.0
    sigma: float = 1.0
    has_in_graph_term = True
    lo = -math.inf
    hi = math.inf

    def __post_init__(self):
        if not (self.sigma > 0.0 and math.isfinite(self.sigma)):
            raise ValueError(f"Gaussian needs a finite sigma > 0, got {self.sigma}")
        object.__setattr__(self, "mu", float(self.mu))
        object.__setattr__(self, "sigma", float(self.sigma))

    @classmethod
    def from_physical(cls, mu: float, sigma: float, spec) -> "Gaussian":
        """A Gaussian stated in a spec's *physical* units (``x = report_offset
        + report_scale * v``), converted to model units in float64."""
        return cls((mu - spec.report_offset) / spec.report_scale,
                   sigma / abs(spec.report_scale))

    @property
    def log_norm(self):
        return -math.log(self.sigma) - _LOG_SQRT_2PI

    def bind(self, lo, hi, name):
        if math.isinf(lo) and math.isinf(hi):
            return self
        return TruncatedGaussian(self.mu, self.sigma, float(lo), float(hi)).bind(lo, hi, name)

    def log_pdf(self, v):
        z = (v - self.mu) / self.sigma
        return -0.5 * z * z

    def log_pdf_np(self, v):
        z = (np.asarray(v, dtype=np.float64) - self.mu) / self.sigma
        return -0.5 * z * z

    def ppf_np(self, x):
        return self.mu + self.sigma * norm_ppf(np.asarray(x, dtype=np.float64))

    mean = property(lambda self: self.mu)
    sd = property(lambda self: self.sigma)


@dataclass(frozen=True)
class TruncatedGaussian(Gaussian):
    """Normal(mu, sigma) restricted to ``(lo, hi)`` (either may be infinite);
    bounds default to the ParamSpec box. ``mass_inside`` is the Gaussian
    probability of the box, Z; ``log_norm`` includes ``-log Z``."""

    lo: float | None = None
    hi: float | None = None
    mass_inside: float = field(init=False, default=math.nan)

    def __post_init__(self):
        super().__post_init__()
        if self.lo is not None and self.hi is not None:
            if not self.hi > self.lo:
                raise ValueError("TruncatedGaussian needs hi > lo")
            a = (self.lo - self.mu) / self.sigma
            b = (self.hi - self.mu) / self.sigma
            Z = _normal_mass(a, b)
            if not (Z > 0.0 and math.isfinite(Z)):
                raise ValueError(
                    f"the box ({self.lo}, {self.hi}) holds no measurable mass "
                    f"of Normal({self.mu}, {self.sigma}) (Z = {Z})")
            object.__setattr__(self, "mass_inside", Z)

    def bind(self, lo, hi, name):
        if self.lo is not None or self.hi is not None:
            _check_box(self, self.lo if self.lo is not None else lo,
                       self.hi if self.hi is not None else hi, lo, hi, name)
        try:
            return TruncatedGaussian(self.mu, self.sigma, float(lo), float(hi))
        except ValueError as exc:
            raise ValueError(f"{name}: {exc}") from None

    @property
    def log_norm(self):
        return -math.log(self.sigma) - _LOG_SQRT_2PI - math.log(self.mass_inside)

    def log_pdf_np(self, v):
        v = np.asarray(v, dtype=np.float64)
        z = (v - self.mu) / self.sigma
        return np.where((v >= self.lo) & (v <= self.hi), -0.5 * z * z, -np.inf)

    def ppf_np(self, x):
        x = np.asarray(x, dtype=np.float64)
        a = (self.lo - self.mu) / self.sigma
        b = (self.hi - self.mu) / self.sigma
        Z = self.mass_inside
        if a > 0.0:
            # box in the upper tail: work with upper-tail probabilities,
            # which are representable where 1 - p would round to 0
            qa = 0.5 * math.erfc(a / _SQRT2)
            z = -norm_ppf(qa - Z * x)
        else:
            pa = 0.5 * math.erfc(-a / _SQRT2)
            z = norm_ppf(pa + Z * x)
        return np.clip(self.mu + self.sigma * z, self.lo, self.hi)

    @property
    def mean(self):
        a = (self.lo - self.mu) / self.sigma
        b = (self.hi - self.mu) / self.sigma
        return self.mu + self.sigma * (_phi(a) - _phi(b)) / self.mass_inside

    @property
    def sd(self):
        a = (self.lo - self.mu) / self.sigma
        b = (self.hi - self.mu) / self.sigma
        Z = self.mass_inside
        ta = 0.0 if math.isinf(a) else a * _phi(a)
        tb = 0.0 if math.isinf(b) else b * _phi(b)
        m = (_phi(a) - _phi(b)) / Z
        return self.sigma * math.sqrt(1.0 + (ta - tb) / Z - m * m)


__all__ = ["Prior", "ImproperFlat", "Uniform", "LogUniform", "Gaussian",
           "TruncatedGaussian"]
