"""Spherical vs oblate evidence on Elome transit fits: the validation behind
anvil.evidence.

Builds two targets on one simulated light curve and one limb-darkening law
(hybrid4, uniform on its curvature simplex): a spherical planet (t0, r, b,
a/R*, q1..q4) and an oblate one, the same plus the oblateness in the form
production fits sample, (fc, fs) = sqrt(f) (cos 2 theta, sin 2 theta) on the
disc fc^2 + fs^2 <= F_MAX. Since dfc dfs = df dtheta, a uniform prior on the
disc is uniform in (f, theta); the box-to-disc ratio 4/pi goes in the
constant. The spherical model is the oblate one at f = 0.

For each dataset (no oblateness, and f = 0.1) it reports SMC's ln Z for both
models, the stages it flagged as stalled, and the wall time. References for
the no-oblateness data at 10,000 points (see docs/evidence.md):

    spherical  84874.36   SMC, five runs at 2,048 and 8,192 particles agree
    oblate     84871.8 +/- 0.1   independent of SMC's tempering path: a
               conditional-evidence integral over the (fc, fs) disc

Needs Elome (../elome). anvil itself does not depend on it.

Run:  .venv/bin/python benchmarks/evidence_elome.py [n_data] [spherical|oblate ...]
"""

from __future__ import annotations

import math
import sys

import mlx.core as mx
import numpy as np

from anvil import ParamSpec, Transform, TransformedLogDensity
from anvil.evidence import smc
from anvil.precision import ChunkedGaussianLogLike
from elome.ld import hybrid4_from_q
from elome.metal import flux_dev_from_tau

PERIOD = 5.0
F_MAX = 0.5          # keeps r <= (1 - f)^1.5 for every r in the box


def _model(oblate: bool):
    def fn(v, x):
        tau = x[None, :] - v[:, 0:1]
        kw = {}
        if oblate:
            fc, fs = v[:, 8], v[:, 9]
            kw = dict(f=mx.minimum(fc * fc + fs * fs, F_MAX), theta=0.5 * mx.arctan2(fs, fc))
        return flux_dev_from_tau(tau, PERIOD, v[:, 3], v[:, 2], v[:, 1],
                                 limb_dark="hybrid4", u=hybrid4_from_q(v[:, 4:8]),
                                 exp_time=0.0, integration="none", **kw)
    return fn


def _specs(oblate: bool):
    s = [ParamSpec("t0", -0.02, 0.02), ParamSpec("r", 0.05, 0.2),
         ParamSpec("b", 0.0, 0.9), ParamSpec("a", 5.0, 30.0)]
    s += [ParamSpec(f"q{i + 1}", 0.0, 1.0) for i in range(4)]
    if oblate:
        R = math.sqrt(F_MAX)
        s += [ParamSpec("fc", -R, R), ParamSpec("fs", -R, R)]
    return s


def make(n_data=10_000, yerr=5e-5, f_true=0.1, theta_deg=30.0, seed=0):
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(-0.2, 0.2, n_data))
    th2 = 2 * math.radians(theta_deg)
    truth = np.array([0.0012, 0.10, 0.30, 12.0, 0.4, 0.5, 0.5, 0.5,
                      math.sqrt(f_true) * math.cos(th2), math.sqrt(f_true) * math.sin(th2)])
    with mx.stream(mx.cpu):
        dev = np.array(_model(True)(mx.array(truth[None], dtype=mx.float64),
                                    mx.array(x, dtype=mx.float64))[0], dtype=np.float64)
    y = dev + yerr * rng.standard_normal(n_data)
    out = {}
    for name, oblate in (("spherical", False), ("oblate", True)):
        lik = ChunkedGaussianLogLike(_model(oblate), x, y, np.full(n_data, yerr))
        if oblate:
            lp, lp_hi = _disc(lik), _disc(lik.hi, mx.cpu)
            const = lik.log_offset_const + lik.log_norm_const + math.log(4 / math.pi)
        else:
            lp, lp_hi = lik, lik.hi
            const = lik.log_offset_const + lik.log_norm_const
        out[name] = TransformedLogDensity(lp, Transform(_specs(oblate)),
                                          model_log_prob_hi=lp_hi, model_log_norm=const)
    return out


def _disc(lp, stream=None):
    """-inf outside the disc fc^2 + fs^2 <= F_MAX (float64 on the CPU stream
    for the float64 path, which the GPU does not run)."""
    def g(v):
        if stream is None:
            return mx.where(v[:, 8] ** 2 + v[:, 9] ** 2 <= F_MAX, lp(v), -mx.inf)
        with mx.stream(stream):
            v = mx.array(v, dtype=mx.float64)
            return mx.where(v[:, 8] ** 2 + v[:, 9] ** 2 <= F_MAX, lp(v), -mx.inf)
    return g


def main(n_data: int, models: list[str]) -> None:
    for f_true in (0.0, 0.1):
        targets = make(n_data=n_data, f_true=f_true)
        print(f"== data f = {f_true}, {n_data} points ==", flush=True)
        lz = {}
        for name in models:
            r = smc(targets[name], seed=0)
            lz[name] = r.log_z
            print(f"{name:9s} SMC ln Z {r.log_z:.3f}  ({r.n_stages} stages, "
                  f"{sum(r.moves)} moves, {r.stalled} stalled, {r.wall_time:.0f} s)", flush=True)
        if len(lz) == 2:
            print(f"ln B(sph/obl) {lz['spherical'] - lz['oblate']:+.3f}", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 10_000,
         sys.argv[2:] or ["spherical", "oblate"])
