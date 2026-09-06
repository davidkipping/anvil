"""Built-in target densities for tests, examples, and benchmarks.

Analytic targets with known moments (Gaussian, Rosenbrock, funnel) validate
sampler correctness; the transit target exercises the full data-heavy
pipeline (chunked fp32 likelihood over ~1e5 points with an fp64 anchor path)
the package is optimized for.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mlx.core as mx
import numpy as np

from ..logdensity import FunctionLogDensity, LogDensity
from ..precision import ChunkedGaussianLogLike, PrecisionPolicy
from ..transforms import ParamSpec, Transform, TransformedLogDensity


def correlated_gaussian(dim: int = 10, rho: float = 0.5, seed: int = 0):
    """N(mu, C) with equicorrelation rho and log-uniform scales.

    Returns (LogDensity, mu, cov).
    """
    rng = np.random.default_rng(seed)
    mu = rng.normal(size=dim)
    scales = np.exp(rng.uniform(-1, 1, size=dim))
    corr = np.full((dim, dim), rho) + (1 - rho) * np.eye(dim)
    cov = np.outer(scales, scales) * corr
    prec = np.linalg.inv(cov)
    mu_mx = mx.array(mu.astype(np.float32))
    prec_mx = mx.array(prec.astype(np.float32))

    def log_prob(u):
        d = u - mu_mx
        return -0.5 * mx.sum((d @ prec_mx) * d, axis=-1)

    return FunctionLogDensity(log_prob, dim), mu, cov


def rosenbrock(sigma_x: float = 1.0, sigma_y: float = 1.0):
    """2-D Rosenbrock/banana with tractable moments.

    x ~ N(1, sigma_x^2), y | x ~ N(x^2, sigma_y^2), so
    E[x] = 1, E[y] = 1 + sigma_x^2. Returns (LogDensity, mean_vec).
    """
    sx2, sy2 = sigma_x**2, sigma_y**2

    def log_prob(u):
        x, y = u[:, 0], u[:, 1]
        return -0.5 * ((x - 1.0) ** 2 / sx2 + (y - x * x) ** 2 / sy2)

    mean = np.array([1.0, 1.0 + sx2])
    return FunctionLogDensity(log_prob, 2), mean


def neals_funnel(dim: int = 10):
    """Neal's funnel: v ~ N(0, 3^2); x_i | v ~ N(0, e^v), i < dim-1.

    Parameter order: u = [x_0 .. x_{dim-2}, v]. Returns (LogDensity,).
    Marginal of v is exactly N(0, 9) — the classic HMC stress test.
    """

    def log_prob(u):
        x, v = u[:, :-1], u[:, -1]
        lp_v = -0.5 * v * v / 9.0
        lp_x = -0.5 * mx.sum(x * x, axis=-1) * mx.exp(-v) - 0.5 * (dim - 1) * v
        return lp_v + lp_x

    return FunctionLogDensity(log_prob, dim)


# --------------------------------------------------------------------------
# Transit target: trapezoid transit + Gaussian noise over a long light curve
# --------------------------------------------------------------------------

def trapezoid_flux(v: mx.array, t: mx.array) -> mx.array:
    """Batched periodic trapezoid transit model (dtype-polymorphic MLX).

    Model-space parameters v (n_chains, 6):
        0: t0     mid-transit time [days since reference epoch]
        1: period [days]
        2: depth  fractional transit depth
        3: dur    full-width at half-depth duration [days]
        4: tau    ingress/egress duration [days]
        5: f0     out-of-transit baseline flux

    t: (m,) times [days since the same reference epoch]. Returns (n, m).
    """
    t0, period = v[:, 0:1], v[:, 1:2]
    depth, dur = v[:, 2:3], v[:, 3:4]
    tau, f0 = v[:, 4:5], v[:, 5:6]
    phase = mx.remainder(t[None, :] - t0 + 0.5 * period, period) - 0.5 * period
    # linear ramp between contact points: full depth inside |phase| < dur/2 -
    # tau/2, zero outside |phase| > dur/2 + tau/2
    ramp = (0.5 * dur + 0.5 * tau - mx.abs(phase)) / tau
    dip = depth * mx.clip(ramp, 0.0, 1.0)
    return f0 - dip


def make_offset_flux(period_ref: float):
    """Fully offset-conditioned trapezoid model (float32-safe).

    Two conditioning failures make the naive model lose ~10s of logL units
    in float32 (both found by ``validate_precision``):

    * folding absolute times: ``k * (P - P_ref)`` with an *absolute* period
      parameter multiplies P's float32 representation error (~2.4e-7 at
      P~3.5 d) by orbit numbers up to ~26;
    * an O(1) baseline flux: every model operation on ~1.0 leaves few-ulp
      noise that the residual weight 1/yerr amplifies by ~2000x.

    So the model space is *offsets*: parameters are t0_off = t0 - t0_ref
    (O(0.1) days), p_off = P - P_ref, df0 = f0 - 1, and the ordinate is
    baseline-subtracted flux deviation. The float64 CPU preprocessing bakes
    the references into the per-datum abscissae once.

    Model-space parameters v (n_chains, 6):
        0: t0_off  mid-transit offset from t0_ref [days]
        1: p_off   period offset from period_ref [days]
        2: depth   fractional transit depth
        3: dur     full-width duration [days]
        4: tau     ingress/egress duration [days]
        5: df0     baseline flux deviation from 1

    ``x`` is (2, m): row 0 = dt = t - t0_ref - k*period_ref (O(P/2) days),
    row 1 = k (orbit numbers, exact small integers). Predicts the flux
    *deviation* ``df0 - dip`` matching data ``y - 1``.
    """
    # plain Python float: a numpy scalar mixed into MLX ops routes through
    # numpy's ufunc machinery, which force-evaluates arrays mid-mx.compile
    period_ref = float(period_ref)

    def flux_dev(v: mx.array, x: mx.array) -> mx.array:
        dt, k = x[0], x[1]
        t0_off, p_off = v[:, 0:1], v[:, 1:2]
        depth, dur = v[:, 2:3], v[:, 3:4]
        tau, df0 = v[:, 4:5], v[:, 5:6]
        phase = dt[None, :] - (t0_off + k[None, :] * p_off)
        # re-wrap in case a large offset pushed a datum into the next orbit
        period = period_ref + p_off
        phase = phase - period * mx.round(phase / period)
        ramp = (0.5 * dur + 0.5 * tau - mx.abs(phase)) / tau
        dip = depth * mx.clip(ramp, 0.0, 1.0)
        return df0 - dip

    return flux_dev


def epoch_center_times(
    t_model: np.ndarray, t0_ref: float, period_ref: float
) -> np.ndarray:
    """Float64 CPU preprocessing: times -> (2, m) [dt, orbit number]."""
    t_model = np.asarray(t_model, dtype=np.float64)
    k = np.round((t_model - t0_ref) / period_ref)
    dt = t_model - t0_ref - k * period_ref
    return np.stack([dt, k])


@dataclass
class TransitTarget:
    """Synthetic transit-fitting problem, built in well-conditioned units.

    The float64 -> model-unit preprocessing happens here on the CPU: times
    are generated as absolute BJD-like values (~2.457e6 days), centered to
    a reference epoch, and (for the default conditioning) reduced to
    per-orbit residuals and baseline-subtracted fluxes before ever touching
    float32.
    """

    target: TransformedLogDensity
    transform: Transform
    loglike: ChunkedGaussianLogLike
    truth_model: np.ndarray  # (6,) true params, model units
    t_ref: float             # reference epoch subtracted from times [days]
    t_model: np.ndarray      # (m,) float64 centered times
    y: np.ndarray            # (m,) float64 fluxes (absolute, ~1.0)


def make_transit_target(
    n_data: int = 100_000,
    yerr: float = 5e-4,
    seed: int = 0,
    policy: PrecisionPolicy | None = None,
    conditioning: str = "epoch_centered",
) -> TransitTarget:
    """Build the synthetic transit problem.

    ``conditioning="epoch_centered"`` (default) uses the float32-safe
    offset parameterization over per-orbit centered abscissae;
    ``"naive"`` fits absolute parameters to absolute fluxes by folding
    whole-baseline times in the graph — deliberately kept as the
    badly-conditioned counterexample the precision harness flags.
    """
    rng = np.random.default_rng(seed)

    # -- float64 world: absolute times, generation, centering -------------
    t_ref = 2_457_000.0  # BJD-like zero-point, hopeless in float32
    baseline_days = 90.0
    t_abs = t_ref + np.sort(rng.uniform(0.0, baseline_days, size=n_data))
    t_model = t_abs - t_ref  # well-conditioned model units

    truth = np.array([
        1.2345,   # t0 [days since t_ref]
        3.456,    # period [days]
        0.010,    # depth (1%)
        0.120,    # duration [days]
        0.015,    # ingress [days]
        1.0,      # baseline flux (normalized)
    ])

    v64 = mx.array(truth[None, :])
    flux_true = np.array(
        trapezoid_flux(v64, mx.array(t_model))[0], dtype=np.float64
    )
    y = flux_true + yerr * rng.standard_normal(n_data)

    yerr_arr = np.full(n_data, yerr)

    if conditioning == "epoch_centered":
        # references: plausible initial estimates, deliberately not the truth
        t0_ref = float(truth[0]) - 0.011
        period_ref = float(truth[1]) + 0.0007
        transform = Transform([
            ParamSpec("t0_off", lo=-0.5, hi=0.5, report_offset=t_ref + t0_ref),
            ParamSpec("p_off", lo=-0.1, hi=0.1, report_offset=period_ref),
            ParamSpec("depth", lo=0.0, hi=0.05),
            ParamSpec("dur", lo=0.01, hi=0.5),
            ParamSpec("tau", lo=0.001, hi=0.1),
            ParamSpec("df0", lo=-0.01, hi=0.01, report_offset=1.0),
        ])
        model_fn = make_offset_flux(period_ref=period_ref)
        x = epoch_center_times(t_model, t0_ref=t0_ref, period_ref=period_ref)
        y_fit = y - 1.0  # float64 baseline subtraction on the CPU
        truth_model = np.array([
            truth[0] - t0_ref, truth[1] - period_ref,
            truth[2], truth[3], truth[4], truth[5] - 1.0,
        ])
    elif conditioning == "naive":
        transform = Transform([
            ParamSpec("t0", lo=truth[0] - 0.5, hi=truth[0] + 0.5,
                      report_offset=t_ref),
            ParamSpec("period", lo=truth[1] - 0.1, hi=truth[1] + 0.1),
            ParamSpec("depth", lo=0.0, hi=0.05),
            ParamSpec("dur", lo=0.01, hi=0.5),
            ParamSpec("tau", lo=0.001, hi=0.1),
            ParamSpec("f0", lo=0.99, hi=1.01),
        ])
        model_fn, x, y_fit, truth_model = trapezoid_flux, t_model, y, truth
    else:
        raise ValueError(f"unknown conditioning: {conditioning!r}")

    loglike = ChunkedGaussianLogLike(model_fn, x, y_fit, yerr_arr, policy)
    target = TransformedLogDensity(
        loglike, transform, model_log_prob_hi=loglike.hi
    )
    return TransitTarget(
        target=target,
        transform=transform,
        loglike=loglike,
        truth_model=truth_model,
        t_ref=t_ref,
        t_model=t_model,
        y=y,
    )
