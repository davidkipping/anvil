"""Flow mode on Elome transit fits: the validation behind anvil.flow.

Three targets on simulated light curves, each fitted with the law it was
simulated from and that law's uniform prior:

    obl4  oblate planet, hybrid4 limb darkening (curvature simplex);
          (fc, fs) = sqrt(f) (cos 2 theta, sin 2 theta) on the disc
          fc^2 + fs^2 <= F_MAX, as in benchmarks/evidence_elome.py
    sph4  spherical planet, hybrid4
    sph2  spherical planet, hybrid2 (exact physical triangle)

hybrid4 targets reproduce benchmarks/evidence_elome.py's data and
parameters, so its references apply: at 10,000 points ln Z = 84871.8 +/- 0.1
(oblate) and 84874.36 (spherical).

For one target, size and seed: anvil.flow.train (SMC from the prior, four
refinement rounds, final fit), the report, the ensemble in flow coordinates
(512 walkers, 500 + 10,000 iterations, every fifth kept), R-hat and ESS,
anvil.flow_evidence, and for comparison the plain ensemble, run as long and
started from the flow's last training round (a better start than SMC's
particles, which the research runs used).

Measured with the research prototype these functions were taken from (3
seeds; docs/flow.md): obl4 at 2,000 points, flow ensemble R-hat 1.015-1.024
at 190 ESS/s median, plain ensemble R-hat 4.1; flow IS ln Z 16934.142-.145.

Needs Elome (../elome). anvil itself does not depend on it.

Run:  .venv/bin/python benchmarks/flow_elome.py <obl4|sph4|sph2> [n_data] [seed]
"""

from __future__ import annotations

import math
import sys
import time

import mlx.core as mx
import numpy as np

import anvil
from anvil import ParamSpec, Transform, TransformedLogDensity
from anvil.kernels.ensemble import DEMove, EnsembleKernel, StretchMove
from anvil.precision import ChunkedGaussianLogLike
from elome.ld import hybrid2_from_q, hybrid4_from_q
from elome.metal import flux_dev_from_tau

PERIOD = 5.0
F_MAX = 0.5
N_Q = {"hybrid2": 2, "hybrid4": 4}
Q_TRUE = {"hybrid2": [0.4, 0.5], "hybrid4": [0.4, 0.5, 0.5, 0.5]}
TARGETS = {"obl4": ("hybrid4", True), "sph4": ("hybrid4", False), "sph2": ("hybrid2", False)}


def _weights(law, q):
    if law == "hybrid2":
        w1, w2 = hybrid2_from_q(q[:, 0], q[:, 1])
        return mx.stack([w1, w2], axis=1)
    return hybrid4_from_q(q)


def _model(law, oblate):
    k = N_Q[law]

    def fn(v, x):
        tau = x[None, :] - v[:, 0:1]
        kw = {}
        if oblate:
            fc, fs = v[:, 4 + k], v[:, 5 + k]
            kw = dict(f=mx.minimum(fc * fc + fs * fs, F_MAX), theta=0.5 * mx.arctan2(fs, fc))
        return flux_dev_from_tau(tau, PERIOD, v[:, 3], v[:, 2], v[:, 1], limb_dark=law,
                                 u=_weights(law, v[:, 4:4 + k]), exp_time=0.0,
                                 integration="none", **kw)
    return fn


def _disc(lp, j, stream=None):
    def g(v):
        if stream is None:
            return mx.where(v[:, j] ** 2 + v[:, j + 1] ** 2 <= F_MAX, lp(v), -mx.inf)
        with mx.stream(stream):
            v = mx.array(v, dtype=mx.float64)
            return mx.where(v[:, j] ** 2 + v[:, j + 1] ** 2 <= F_MAX, lp(v), -mx.inf)
    return g


def make(key, n_data=2000, yerr=5e-5, seed=0):
    law, oblate = TARGETS[key]
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(-0.2, 0.2, n_data))
    truth = np.array([0.0012, 0.10, 0.30, 12.0, *Q_TRUE[law], 0.0, 0.0])   # f = 0
    with mx.stream(mx.cpu):
        dev = np.array(_model(law, True)(mx.array(truth[None], dtype=mx.float64),
                                         mx.array(x, dtype=mx.float64))[0], dtype=np.float64)
    y = dev + yerr * rng.standard_normal(n_data)
    lik = ChunkedGaussianLogLike(_model(law, oblate), x, y, np.full(n_data, yerr))
    const = lik.log_offset_const + lik.log_norm_const
    specs = [ParamSpec("t0", -0.02, 0.02), ParamSpec("r", 0.05, 0.2),
             ParamSpec("b", 0.0, 0.9), ParamSpec("a", 5.0, 30.0)]
    specs += [ParamSpec(f"q{i + 1}", 0.0, 1.0) for i in range(N_Q[law])]
    lp, lp_hi = lik, lik.hi
    if oblate:
        R = math.sqrt(F_MAX)
        specs += [ParamSpec("fc", -R, R), ParamSpec("fs", -R, R)]
        j = 4 + N_Q[law]
        lp, lp_hi = _disc(lik, j), _disc(lik.hi, j, mx.cpu)
        const += math.log(4 / math.pi)
    return TransformedLogDensity(lp, Transform(specs), model_log_prob_hi=lp_hi,
                                 model_log_norm=const)


def _row(name, chain, wall):
    d = anvil.diagnose(chain)
    ess = float(d.ess_bulk.min())
    print(f"  {name:28s} R-hat {d.rhat.max():6.3f}  min ESS {ess:8.0f}  "
          f"{wall:6.0f} s  ESS/s {ess / wall:8.1f}", flush=True)


def main(key, n_data=2000, seed=0):
    target = make(key, n_data)
    print(f"== {key}, {n_data} points, seed {seed} ==", flush=True)
    t0 = time.perf_counter()
    flow = anvil.flow.train(target, seed=seed, progress=False)
    print(f"  trained in {time.perf_counter() - t0:.0f} s", flush=True)
    print(flow.report, flush=True)

    ev = anvil.flow_evidence(target, flow, seed=seed + 7)
    print(f"  {ev}", flush=True)

    moves = [(StretchMove(), 0.5), (DEMove(), 0.5)]
    ft = flow.wrap(target)
    t1 = time.perf_counter()
    res = anvil.run(EnsembleKernel(ft, moves=moves), ft, flow.start(512, seed=seed),
                    n_warmup=500, n_samples=2000, thin=5, seed=10 * seed + 1)
    _row("ensemble, flow coordinates", res.get_chain(), time.perf_counter() - t1)

    t2 = time.perf_counter()
    plain = anvil.run(EnsembleKernel(target, moves=moves), target,
                      flow.pool[np.random.default_rng(seed).choice(len(flow.pool), 512)],
                      n_warmup=500, n_samples=2000, thin=5, seed=10 * seed + 3)
    _row("plain ensemble", plain.get_chain(), time.perf_counter() - t2)


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], int(a[1]) if len(a) > 1 else 2000, int(a[2]) if len(a) > 2 else 0)
