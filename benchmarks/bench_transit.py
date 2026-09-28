"""Headline benchmark: anvil on the GPU vs emcee on the CPU.

Task: fit the 6-parameter trapezoid transit model to a synthetic light
curve of ``N_DATA`` points (default 1e5). Metric: minimum bulk ESS per
wall-clock second, warmup/burn-in included.

The emcee likelihood is vectorized float64 numpy (the strong CPU
baseline a careful astronomer would write), evaluated per walker.

Run:  .venv/bin/python benchmarks/bench_transit.py [--quick]
"""

from __future__ import annotations

import argparse
import time

import mlx.core as mx
import numpy as np

import anvil
from anvil import run
from anvil.kernels.chees import ChEESHMC
from anvil.kernels.ensemble import DEMove, EnsembleKernel, StretchMove
from anvil.targets import make_transit_target
from anvil.targets.builtin import epoch_center_times

N_DATA = 100_000


def numpy_offset_loglike(tt):
    """Vectorized float64 numpy replica of the offset transit likelihood
    (per single parameter vector) for emcee."""
    x = epoch_center_times(
        tt.t_model,
        t0_ref=float(tt.transform.specs[0].report_offset - tt.t_ref),
        period_ref=float(tt.transform.specs[1].report_offset),
    )
    dt, k = x[0], x[1]
    y = tt.y - 1.0
    yerr = 5e-4
    period_ref = float(tt.transform.specs[1].report_offset)
    lo = np.array([s.lo for s in tt.transform.specs])
    hi = np.array([s.hi for s in tt.transform.specs])

    def log_prob(theta):
        if np.any(theta <= lo) or np.any(theta >= hi):
            return -np.inf
        t0_off, p_off, depth, dur, tau, df0 = theta
        period = period_ref + p_off
        phase = dt - (t0_off + k * p_off)
        phase = phase - period * np.round(phase / period)
        ramp = (0.5 * dur + 0.5 * tau - np.abs(phase)) / tau
        dip = depth * np.clip(ramp, 0.0, 1.0)
        r = (y - (df0 - dip)) / yerr
        return -0.5 * np.sum(r * r)

    return log_prob


def bench_anvil(tt, kernel_name, n_chains, n_warmup, n_samples, thin=1):
    u_truth = tt.transform.from_model_np(tt.truth_model)
    rng = np.random.default_rng(0)
    # tight init ball, matching what emcee gets below (standard practice:
    # initialize near a preliminary fit)
    u0 = mx.array(
        (u_truth + 1e-3 * rng.standard_normal((n_chains, 6))).astype(np.float32)
    )
    if kernel_name == "chees":
        kernel = ChEESHMC(tt.target, max_leapfrog=128, dense=True)
    else:
        # 50/50 stretch + DE roughly halves the autocorrelation of stretch
        # alone on this target
        kernel = EnsembleKernel(
            tt.target, moves=[(StretchMove(), 0.5), (DEMove(), 0.5)], seed=0)
    t0 = time.perf_counter()
    res = run(kernel, tt.target, u0, n_warmup=n_warmup, n_samples=n_samples,
              thin=thin, seed=1)
    wall = time.perf_counter() - t0
    chain = res.get_chain()
    diag = anvil.diagnose(chain)
    ess, rhat = diag.ess_bulk, diag.rhat
    return {
        "label": f"anvil-{kernel_name} (GPU, {n_chains} chains)",
        "wall_s": wall,
        "ess_min": float(ess.min()),
        "rhat_max": float(rhat.max()),
        "ess_per_s": float(ess.min() / wall),
        "extra": f"divergent={res.extras.get('n_divergent', 0)}"
        if kernel_name == "chees" else "",
    }


def bench_emcee(tt, n_walkers, n_steps, n_burn):
    import emcee

    log_prob = numpy_offset_loglike(tt)
    rng = np.random.default_rng(2)
    u_truth = tt.transform.from_model_np(tt.truth_model)
    v_truth = tt.transform.model_np(u_truth)
    lo = np.array([s.lo for s in tt.transform.specs])
    hi = np.array([s.hi for s in tt.transform.specs])
    width = hi - lo
    p0 = np.clip(
        v_truth + 1e-3 * width * rng.standard_normal((n_walkers, 6)),
        lo + 1e-6 * width, hi - 1e-6 * width,
    )
    sampler = emcee.EnsembleSampler(n_walkers, 6, log_prob)
    t0 = time.perf_counter()
    sampler.run_mcmc(p0, n_steps, progress=False)
    wall = time.perf_counter() - t0
    chain = sampler.get_chain(discard=n_burn)
    diag = anvil.diagnose(chain)
    ess, rhat = diag.ess_bulk, diag.rhat
    return {
        "label": f"emcee (CPU fp64 numpy, {n_walkers} walkers)",
        "wall_s": wall,
        "ess_min": float(ess.min()),
        "rhat_max": float(rhat.max()),
        "ess_per_s": float(ess.min() / wall),
        "extra": "",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    tt = make_transit_target(n_data=N_DATA, seed=0)
    print(f"transit benchmark: {N_DATA} data points, 6 parameters\n")

    results = []
    # Configurations are chosen so every row CONVERGES (R-hat < 1.01).
    # The ensemble decorrelates in ~10^2 iterations, so it needs long
    # chains and few walkers; ChEES decorrelates in ~1 and needs the
    # opposite. Comparing an unconverged run's ESS/s is meaningless.
    if args.quick:
        results.append(bench_anvil(tt, "chees", 512, 300, 200))
        results.append(bench_anvil(tt, "ensemble", 128, 500, 8000))
        results.append(bench_emcee(tt, 64, 2000, 1000))
    else:
        results.append(bench_anvil(tt, "chees", 512, 400, 300))
        results.append(bench_anvil(tt, "ensemble", 128, 500, 30000))
        results.append(bench_emcee(tt, 64, 12000, 6000))

    print(f"{'sampler':>44s} {'wall[s]':>8s} {'minESS':>9s} "
          f"{'rhat':>7s} {'ESS/s':>9s}")
    base = None
    for r in results:
        print(f"{r['label']:>44s} {r['wall_s']:>8.1f} {r['ess_min']:>9.0f} "
              f"{r['rhat_max']:>7.3f} {r['ess_per_s']:>9.1f}  {r['extra']}")
        if base is None:
            base = r
    emcee_row = results[-1]
    for r in results[:-1]:
        print(f"\n{r['label']}: {r['ess_per_s'] / emcee_row['ess_per_s']:.0f}x "
              "the ESS/s of emcee")


if __name__ == "__main__":
    main()
