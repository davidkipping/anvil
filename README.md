# anvil

MCMC sampling engine optimized for Apple Silicon, built on
[MLX](https://github.com/ml-explore/mlx).

**The premise.** On Apple Silicon the GPU wins by running **thousands of
parallel chains** over **data-heavy likelihoods** (10⁴–10⁶ points) in
well-conditioned float32, while the CPU — which has float64 — owns offsets,
large constants, preprocessing, re-anchoring, and diagnostics. Unified
memory makes the split free: no transfers, and MLX lets float64 CPU
operations live inside the same lazy graph as float32 GPU work.

## Two sampler families, one engine

- **ChEES-HMC** (flagship) — Hamiltonian Monte Carlo with cross-chain
  adaptation of step size (dual averaging on the harmonic-mean acceptance),
  trajectory length (the ChEES criterion), and a diagonal preconditioner
  (Hoffman & Sountsov, AISTATS 2021). No per-chain control flow anywhere:
  the trajectory jitter is a shared per-iteration Halton scalar, so
  thousands of chains advance in lockstep. Requires an MLX-differentiable
  log density. On smooth targets it reaches ~60% ESS per draw.
- **Ensemble moves** (robust fallback) — vectorized Goodman-Weare stretch
  and differential-evolution moves with detailed-balance-correct red-black
  half-ensemble updates. Gradient-free, affine-invariant, near-drop-in for
  emcee workflows; the right choice for likelihoods with kinks, plateaus,
  or suspect gradients.

## Quick start

```python
import mlx.core as mx
import numpy as np
import anvil

# a batched MLX log-probability: (n_chains, dim) -> (n_chains,)
mu = mx.array([1.0, -0.5, 2.0])
def log_prob(theta):
    d = theta - mu
    return -0.5 * mx.sum(d * d, axis=-1)

sampler = anvil.EnsembleSampler(2048, 3, log_prob)   # stretch move
# sampler = anvil.HMCSampler(2048, 3, log_prob)      # ChEES-HMC
p0 = np.random.default_rng(0).normal(size=(2048, 3))
sampler.run_mcmc(p0, 500, warmup=500)
chain = sampler.get_chain()          # (500, 2048, 3) numpy
print(anvil.diagnostics.summary(chain))
```

The one hard difference from emcee: `log_prob` receives **all chains at
once** as an `mx.array` and must be built from `mlx.core` ops (that is
what puts it on the GPU). A per-walker numpy function raises a
`TypeError` with porting hints at construction.

## The float32 discipline (read this once)

Metal GPUs have no float64. Sampling correctness in float32 is a
*conditioning* problem, and this package hands you a working discipline:

1. **Preprocess in float64 on the CPU.** Center times to a reference epoch
   (`t - t_ref`, never absolute BJD), normalize fluxes, subtract
   baselines. Only O(1)-conditioned arrays enter the GPU graph.
2. **Sample offsets, not absolutes.** Parameters like `t0` and `P` enter
   the model as `t0 - t0_ref`, `P - P_ref`; `ParamSpec.report_offset`
   (float64, host-side) reinstates absolute units on output. See
   `anvil/targets/builtin.py` for a worked transit example — the
   naive absolute-parameter model loses ~10 units of log-likelihood to
   float32 rounding; the offset model loses ~0.2.
3. **Reduce with structure.** `ChunkedGaussianLogLike` sums per-datum
   terms chunk-wise (tree error O(ε√n)); the optional `fp64_anchor`
   policy does the cross-chunk sum in float64 on the CPU stream for
   little cost (~0–20%).
4. **Verify, don't hope.** `anvil.validate_precision(target, u)`
   compares the production float32 path against a float64 CPU path and
   reports the error against the ~1-unit scale of Metropolis accept
   decisions:

   ```
   PrecisionReport over 32 states
     |logL| typical magnitude : 5.1e+04
     |fp32 - fp64| median     : 0.0015
     |fp32 - fp64| max        : 0.0079
     OK: fp32 error is far below the ~1-unit scale of Metropolis ...
   ```
5. **Re-anchor only when the density can go stale.**
   `run(..., reanchor_every=N)` refreshes cached log-probabilities through
   the float64 path. Float32 rounding does *not* drift (cached values are
   fresh evaluations of a deterministic function), so this is insurance
   for adaptive/surrogate densities, not for rounding — and it is
   expensive. Default off.

## Performance

Benchmark: 6-parameter trapezoid transit fit over a 100,000-point light
curve on an M2 Max (30-core GPU); metric is minimum bulk ESS per second of
total wall time (warmup included) versus emcee running vectorized float64
numpy. See `benchmarks/bench_transit.py`.

<!-- BENCH_TABLE -->

Every configuration is chosen so that it **converges** (R-hat < 1.01) —
comparing the ESS/s of an unconverged run is meaningless. The two
samplers want opposite settings to get there: the ensemble decorrelates
in ~10² iterations so it needs long chains and few walkers, while
ChEES-HMC decorrelates in ~1 and wants the opposite.

Honest caveats: sustained GPU throughput swings ~2× with thermal state,
and emcee's own ESS/s varies several-fold across repeats on the same
machine, so treat the ratios as order-of-magnitude. The trapezoid model's
kinks and plateaus are near ChEES's worst case; on smooth targets it
reaches ~60% ESS per draw (see `tests/test_chees.py`).

For a science-grade worked example on a real transit model, including a
converged head-to-head and a curvature read-out explaining the result,
run `examples/metalplanet_hotjupiter.py`.

Rules of thumb from profiling:

- The regime that pays is (many chains) × (much data). A few dozen walkers
  on a cheap analytic posterior will not beat the CPU.
- Data-heavy likelihood evaluations are memory-bandwidth-bound; `mx.compile`
  fusion (automatic inside kernel steps) is worth ~5× over eager ops.
- ChEES-HMC costs L gradient evaluations per step but repays with an
  order-of-magnitude better per-draw efficiency on smooth targets; the
  stretch move is the safer pick for kinky or plateau-ridden posteriors
  (transit ingress/egress edges, box-like models).

## Diagnostics

`anvil.diagnostics` implements split rank-normalized R-hat, bulk ESS
(Geyer-truncated, FFT), and **nested R-hat** (Margossian et al. 2022) for
the many-short-chains regime this hardware favors — with thousands of
chains you need only tens of post-warmup draws per chain, and nested R-hat
remains informative there. Neal's funnel and similar pathologies are
flagged by the diagnostics rather than silently mis-sampled
(`tests/test_chees.py::test_funnel_is_flagged_not_silently_wrong`).

## Roadmap: neural likelihood emulation

The `surrogate` module carries the seams for BAMBI-style emulation
([Graff et al. 2012](https://arxiv.org/abs/1110.2997)): pass
`run(..., archive=TrainingArchive(dim))` and every stored state becomes a
training pair at zero extra cost. v2 will train an MLX network on the
archive, compare its `error_estimate` against the Metropolis decision
scale, and let `SwitchableLogDensity` substitute it for the expensive
likelihood (with periodic exact audits) — optionally compiled to Core ML
for the Neural Engine.

## Install (development)

```bash
python -m venv .venv
.venv/bin/pip install -e ".[test]"
.venv/bin/python -m pytest -m "not slow"   # fast tests
.venv/bin/python -m pytest                 # full statistical suite (~1 min)
.venv/bin/python examples/transit_lightcurve.py
```

Requires macOS on Apple Silicon, Python ≥ 3.10, MLX ≥ 0.30.
