"""End-to-end example: fit a 6-parameter transit model to a synthetic
100,000-point light curve with both sampler families.

Demonstrates the full anvil workflow:
  1. float64 CPU preprocessing into well-conditioned model units
     (handled inside make_transit_target; see targets/builtin.py for the
     offset discipline worth copying into your own likelihoods),
  2. the precision harness (check before you trust),
  3. thousands of GPU chains with ChEES-HMC and the stretch move,
  4. diagnostics and reporting in physical units.

Run:  .venv/bin/python examples/transit_lightcurve.py
"""

import time

import mlx.core as mx
import numpy as np

from anvil import run, validate_precision
from anvil.diagnostics import ess_bulk, split_rhat, summary
from anvil.kernels.chees import ChEESHMC
from anvil.kernels.ensemble import EnsembleKernel
from anvil.targets import make_transit_target

N_CHAINS = 1024

tt = make_transit_target(n_data=100_000, seed=42)
names = tt.transform.names

# -- 1+2: trust check on the float32 GPU likelihood ------------------------
u_guess = tt.transform.from_model_np(tt.truth_model)
rng = np.random.default_rng(0)
u0 = mx.array(
    (u_guess + 1e-3 * rng.standard_normal((N_CHAINS, 6))).astype(np.float32)
)
print(validate_precision(tt.target, u0[:32]), "\n")

# -- 3: sample with both kernels -------------------------------------------
for label, kernel, n_warmup, n_samples, thin in (
    ("stretch", EnsembleKernel(tt.target, seed=0), 1500, 400, 2),
    ("ChEES-HMC", ChEESHMC(tt.target, max_leapfrog=128), 300, 150, 1),
):
    t0 = time.perf_counter()
    res = run(kernel, tt.target, u0, n_warmup=n_warmup, n_samples=n_samples,
              thin=thin, seed=1, reanchor_every=100)
    wall = time.perf_counter() - t0
    chain = res.get_chain()
    ess = ess_bulk(chain)
    print(f"== {label}: {wall:.1f}s, min ESS {ess.min():.0f} "
          f"({ess.min() / wall:.0f} ESS/s), "
          f"max R-hat {split_rhat(chain).max():.4f}")

    # -- 4: report in physical units (float64 reinstates absolute epochs) --
    flat_model = res.get_chain(flat=True).astype(np.float64)
    phys = tt.transform.to_physical(tt.transform.model_np(flat_model))
    mid = phys.mean(axis=0)
    sd = phys.std(axis=0)
    truth_phys = tt.transform.to_physical(tt.truth_model)
    for i, n in enumerate(names):
        print(f"  {n:>8s} = {mid[i]:.12g} +/- {sd[i]:.2g}   "
              f"(truth {truth_phys[i]:.12g})")
    print()

print("sampling-space (u) summary of the ChEES-HMC run:")
print(summary(chain, names=names))
