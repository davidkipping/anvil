"""Timing + accuracy of the log-likelihood reduction policies.

Decides the default PrecisionPolicy: if fp64_anchor costs little, users
with marginal conditioning can enable it freely; the shipped default stays
fp32_tree + periodic re-anchoring.

Run:  .venv/bin/python benchmarks/bench_reduction.py
"""

from __future__ import annotations

import time

import mlx.core as mx
import numpy as np

from anvil.precision import PrecisionPolicy, validate_precision
from anvil.targets import make_transit_target


def bench(n_chains: int, n_data: int) -> None:
    rng = np.random.default_rng(0)
    print(f"\n== {n_chains} chains x {n_data} data points ==")
    for reduction in ("fp32_tree", "fp64_anchor", "fixed_point"):
        tt = make_transit_target(
            n_data=n_data, seed=0,
            policy=PrecisionPolicy(reduction=reduction),
        )
        u_truth = tt.transform.from_model_np(tt.truth_model)
        u = mx.array(
            (u_truth + 3e-3 * rng.standard_normal((n_chains, 6)))
            .astype(np.float32)
        )
        # compile: the engine always runs the likelihood inside a compiled
        # step, and the eager path is ~10x slower (it streams a full-size
        # temporary per elementwise op), so timing eager would overstate
        # the shipped cost by an order of magnitude
        log_prob = mx.compile(tt.target.log_prob)
        lp = log_prob(u)
        mx.eval(lp)
        t0 = time.perf_counter()
        reps = 20
        for _ in range(reps):
            lp = log_prob(u)
            mx.eval(lp)
        dt = (time.perf_counter() - t0) / reps * 1000
        rep = validate_precision(tt.target, u[:8])
        print(f"  {reduction:>12s}: {dt:7.1f} ms/eval   "
              f"median |fp32-fp64| = {rep.median_abs_err:.3g}")


if __name__ == "__main__":
    bench(1024, 100_000)
    bench(4096, 100_000)
    bench(1024, 1_000_000)
