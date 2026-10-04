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
  trajectory length (the ChEES criterion), and a preconditioner —
  diagonal by default, or the full cross-chain covariance with
  `dense=True` (Hoffman & Sountsov, AISTATS 2021). No per-chain control
  flow anywhere:
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
3. **Recentre, then reduce with structure.** A chi-squared over N points
   sums to ≈ −N/2, so anvil sums `0.5·(1 − r²)` and keeps the exact −N/2
   as a float64 host constant: the float32 quantity is then O(√(N/2))
   rather than O(N/2), and the ulp of every stored log-probability falls
   ~128×. `ChunkedGaussianLogLike` then sums chunk-wise (tree error
   O(ε√n)). Reduction tricks are the *small* lever here — conditioning is
   the large one; see the precision guide for the measured breakdown.
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
5. **Re-anchor only for a non-deterministic evaluation.**
   `run(..., reanchor_every=N)` refreshes cached log-probabilities through
   the float64 path. Float32 rounding does *not* drift (cached values are
   fresh evaluations of a deterministic function), so this is not insurance
   for rounding — and it is expensive. Default off. It is also *not* how you
   pick up a target that changed: see "Changing the target between runs".

## Performance

Benchmark: 6-parameter trapezoid transit fit over a 100,000-point light
curve on an M2 Max (30-core GPU); metric is minimum bulk ESS per second of
total wall time (warmup included) versus emcee running vectorized float64
numpy. See `benchmarks/bench_transit.py`.

| sampler | wall [s] | min ESS | max R-hat | ESS/s | vs emcee |
|---|---:|---:|---:|---:|---:|
| ensemble, stretch+DE (GPU, 128 walkers) | 36 | 123,917 | 1.001 | 3,469 | **215×** |
| ChEES-HMC, dense mass (GPU, 512 chains) | 75 | 90,156 | 1.003 | 1,203 | **75×** |
| emcee (CPU fp64 numpy, 64 walkers) | 336 | 5,427 | 1.014 | 16 | — |

Both anvil rows are **converged** (R-hat ≤ 1.003) because comparing the
ESS/s of an unconverged run is meaningless. The two samplers need
opposite settings to get there: the ensemble decorrelates in ~10²
iterations so it wants long chains and few walkers, while ChEES-HMC
decorrelates in ~1 and wants the reverse. The emcee row is given 12,000
iterations and still sits at R-hat 1.014 — slightly short of the same
bar, and left that way because closing it costs another several minutes
of CPU and would only widen the gap.

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

## Changing the target between runs

A target may hold state its log-density reads but the sampler never sees: a
Gibbs block sampled elsewhere, a tempering β, a swapped dataset, a retrained
surrogate. Change it between `run` calls and anvil picks it up:

```python
target.conditional = new_block            # whatever your log_prob reads
res2 = anvil.run(kernel, target, resume=res.resume_state().with_positions(u, target),
                 n_samples=400)
```

This needs saying because getting it wrong is silent. `mx.compile` freezes
everything a traced function reads that is not an argument, so a kernel that
compiles once kept sampling the target it first traced — and a *small*
change still accepts normally and converges, to the old posterior, with
healthy R-hat and ESS. `run` now calls `kernel.retrace()` on every call and
recomputes the cached log-density on resume; for an unchanged target both are
no-ops in effect, so draws are bit-identical. Custom kernels that compile
their own graphs must override `retrace()`, and are warned if they do not.

The limit: a target changed *during* a run (from a `callback`) is not picked
up until the next `run` call, and `reanchor_every` does not help — it
refreshes the cache while the compiled proposal keeps the old target. Drive
such a scheme as a sequence of segments.

## Extending a run

Short of effective samples? Extend the same chains rather than start over:

```python
res1 = anvil.run(kernel, target, u0, n_warmup=400, n_samples=200, seed=1)
res2 = anvil.run(kernel, target, resume=res1, n_samples=400)
res1.save_state("run.npz")                 # .npz, no pickle
state = anvil.load_state("run.npz")        # ... in another process
```

`res.resume_state().with_positions(u, target)` moves the chains somewhere
else first, recomputing every cached quantity there — the seam for
alternating anvil's sampling with an exact move of your own (a Gibbs sweep
over a conditional anvil cannot see, a mode hop across a gap HMC will not
cross).

A resume continues from the final positions with the adaptation frozen
(step size, trajectory length, diagonal or dense preconditioner), so a
target needing 16k draws per chain pays warmup once rather than once per
doubling round. It also continues the **key stream**: keys come from
`(seed, iteration, role)`, so a naive `n_warmup=0` rerun would redraw the
warmup keys — every `Results` carries `iters_consumed` and a continuation
offsets from it. `n_warmup` defaults to 0 on a resume and an explicit
nonzero one raises: re-adapting would make the continuation a different
Markov chain, whose draws could not honestly be concatenated with the first
segment's.

## Diagnostics and after-the-fact checks

`anvil.diagnose(chain)` returns R-hat and bulk ESS from a single shared
pass (rank normalization is ~85% of the work, so don't compute it twice).
Alongside it:

| call | question it answers |
|---|---|
| `anvil.diagnose` | did it converge, and how many effective samples? |
| `anvil.warmup_report` | was warmup long enough — or far longer than needed? |
| `anvil.whitened_shape` | is the posterior merely correlated, or curved? (predicts whether `dense=True` will pay) |
| `anvil.validate_precision` | is the float32 likelihood accurate enough to trust? |
| `anvil.certify` | how much did float32 bias the posterior, and what is it corrected? |

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

`VERSIONS.md` records what changed in each version. If you depend on anvil
from another package, feature-detect the capability you need
(`"resume" in inspect.signature(anvil.run).parameters`) rather than
comparing version strings — but check `anvil.__version__` against
`VERSIONS.md` if an API you expect appears to be missing, since a
non-editable install can be stale.
