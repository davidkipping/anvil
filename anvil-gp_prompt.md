# Task: build `anvil-gp` — batched Gaussian-process likelihoods for the anvil MCMC engine on Apple Silicon

You are building a Gaussian-process likelihood component for **anvil**, an
existing, tested MCMC engine that runs thousands of parallel chains on the
Apple GPU via MLX in conditioned float32, with float64 support work on the
CPU. Read this whole brief before writing code — it encodes hard
constraints from two completed sibling projects (the engine itself, and
the MetalPlanet transit model), and the central algorithmic choice here is
genuinely open and must be settled by measurement, not taste.

## Mission

Time-series GP likelihoods — celerite-class kernels (sums of damped
exponential/oscillator terms: SHO, real, complex; quasi-periodic as a sum
of terms) — evaluated for **thousands of parameter vectors per call** on
the GPU, with gradients, so that anvil samples **GP hyperparameters
JOINTLY with mean-model parameters**. Joint sampling is a hard
requirement: designs that require freezing/pre-fitting hyperparameters
are out of scope and must not shape the architecture.

Primary use case: transit/RV fitting where the mean model is MetalPlanet
(`metalplanet.anvil.make_quad_transit_flux`) and the GP absorbs stellar
variability + instrumental noise, N = 10⁴–10⁶ points, J (kernel latent
dimension) ≈ 2–8, 5–20 total parameters, 1024–4096 chains.

## Context you must absorb first

Read, in this order (all paths relative to the repo root
`/Users/dkipping/Storage1/Work/Documents/Transit_Work/CODES/`):

1. `anvil/README.md` and `anvil/docs/precision.md` — the engine, its
   float32 discipline, and the precision-harness contract.
2. `anvil/anvil/precision.py` — `ChunkedGaussianLogLike` is the
   white-noise likelihood your GP likelihood will *replace* (a GP
   factorization is global over the data; chunked accumulation does not
   transfer, but the fp32-conditioning *mindset* does).
3. `anvil/anvil/logdensity.py`, `anvil/anvil/transforms.py` — the
   `LogDensity` contract you plug into, and the ParamSpec/Transform layer
   that will carry hyperparameter bounds (sample scales in log space via
   half-bounded specs).
4. `MetalPlanet/README.md` and `MetalPlanet/docs/sampler-integration.md`
   — the batching rule and the proven "thread-per-chain fused Metal
   kernel + MLX-graph reference implementation + fp64 oracle" layering.
   MetalPlanet is the architectural template for this project.
5. `anvil/examples/metalplanet_hotjupiter.py` — the end-to-end pattern
   your deliverable must extend with a GP noise model.

## The central algorithmic problem

celerite2's O(N·J²) factorization is a **sequential recursion over data
points** — the opposite of GPU-shaped. Three consequences:

- CPU celerite2 cannot be called from the hot loop (not MLX: invisible
  to autodiff, unbatchable). It is the **numerical oracle only**.
- A naive MLX transliteration (Python loop building an N-step graph) is
  infeasible at N = 10⁵ — graph construction cost alone rules it out.
- The parallelism must therefore be *reorganized*. Two candidate designs,
  and an early milestone must benchmark both far enough to pick:

**Design A — thread-per-chain sequential Metal kernel** (the MetalPlanet
pattern). One GPU thread runs the full O(N·J²) recursion for one chain.
The chain axis supplies the parallelism the data axis refuses to give.
Forward kernel + a second kernel implementing the celerite2 paper's
closed-form backward recursions (Foreman-Mackey et al. 2017/2018 +
celerite2 paper give the reverse-mode sweeps). Risks to quantify:
occupancy at only 1024–4096 threads of long-running work; register
pressure at J = 8; fp32 error growth over 10⁵ recursion steps.

**Design B — associative-scan state-space formulation in pure MLX.**
celerite kernels are exactly linear Gaussian state-space models; the
Kalman-filter likelihood admits an associative-scan reformulation
(Särkkä & García-Fernández 2021, "Temporal parallelization of Bayesian
smoothers"), implementable as ~log₂N batched stages of elementwise ops —
parallel over data AND chains, no Metal shader authorship, and the same
code runs float64-on-CPU for verification. Risks to quantify: memory
traffic of log₂N full-size stages; numerical behavior of the scan
element composition in fp32; autodiff memory vs a hand-written VJP. The
deep-learning SSM literature (S4/S5/LRU-style associative scans) is
directly relevant prior art for scan-on-GPU mechanics.

Whatever wins, the layering is fixed (copy MetalPlanet):
a **pure-MLX-graph reference implementation always exists** (dtype-
polymorphic: fp32 GPU and fp64 CPU from the same code), the fast path
(Metal kernel or tuned scan) must pass parity tests against it, and
celerite2 (CPU, fp64) adjudicates both.

## Also weigh: is celerite even the right kernel class for this hardware?

celerite's design point is CPU-sequential O(N). On a batched-GPU
architecture other exact-or-controlled approximations may dominate, and
you should evaluate at least one as a candidate before committing:

- **Reduced-rank / Hilbert-space GPs** (Solin & Särkkä basis expansions,
  as in Stan/PyMC "HSGP"): m basis functions turn the marginal
  likelihood into batched dense linear algebra — O(N·m + m³) of
  matmul-shaped work, trivially batched over chains, trivially
  autodiff-able, approximation error controlled by m and the boundary
  factor. celerite-class kernels have rational spectral densities, so
  the required spectral evaluations are closed-form.
- FFT/Toeplitz methods for (near-)regularly sampled data (MLX has FFTs).
- Anything the reviews of this brief surface (the brief is reviewed by
  independent agents before implementation starts; incorporate their
  findings).

Decision rule: exactness matters (this is inference, not prediction), so
an approximate method is admissible only with a measurable, reportable
accuracy knob and a demonstration that posterior distortion is below
Monte-Carlo error on the fiducial problem.

## Contract

```python
class CeleriteLogLike:            # name per chosen design
    def __init__(self, mean_fn, x, y, yerr, kernel_spec, policy=None): ...
    def __call__(self, v: mx.array) -> mx.array   # (n_chains, dim) -> (n_chains,)
    def hi(self, v: mx.array) -> mx.array         # float64 CPU path (tiled!)
```

- `v` packs mean-model parameters and GP hyperparameters in one vector;
  a small spec object maps slices to roles. Residuals are per-chain
  (`y - mean_fn(v_mean, x)`) — the factorization runs per chain.
- Include the parameter-independent normalization; log|K| depends on
  hyperparameters and must be in-graph — no dropping constants as the
  white-noise likelihood does.
- Must satisfy anvil's engine contract: pure MLX in the hot path,
  `mx.compile`-safe (no data-dependent Python control flow, no numpy
  scalars in the graph — a `np.float64` closure constant force-evaluates
  mid-compile), fixed iteration counts everywhere.
- `.hi` must be **tiled over chains** (see
  `ChunkedGaussianLogLike.hi` after commit 0a00adb): an untiled fp64
  graph at 1024 chains × 10⁵ points transiently allocated ~49 GB in an
  earlier incident. Budget ≈1 GB peak.
- Kipping-style reparameterizations welcome at the Transform layer;
  hyperparameter priors are the user's job via ParamSpec bounds + explicit
  in-model terms.

## Precision requirements (gates, not suggestions)

- fp32 vs fp64 |Δ logL| measured by `anvil.validate_precision` on the
  fiducial problem: median well below 1 at the posterior typical set.
- The recursion/scan's fp32 error growth over N = 10⁵ must be measured
  and reported as a function of N and of kernel timescale/duration
  ratios (long-memory kernels stress the propagators). Compensated or
  fp64-anchored accumulation of the log-det term if needed.
- Gradient checks: analytic/VJP gradients vs fp64 central differences
  at ~10³ points including hyperparameter extremes (Q ≪ 1, Q ≫ 1,
  w0·baseline ≫ 1); zero NaNs across 10⁵ random draws.

## Validation and success criteria

1. **Numeric parity**: logL and all gradients vs celerite2 (CPU fp64)
   across a grid of kernels (SHO low/high Q, real, sums, +jitter),
   N ∈ {10³, 10⁴, 10⁵}, irregular sampling. fp64 path: ~1e-9 relative;
   fp32 path: within the measured, reported error budget.
2. **Statistical**: joint injection-recovery — MetalPlanet hot Jupiter
   (the fiducial case in `examples/metalplanet_hotjupiter.py`) plus an
   injected SHO stellar-variability signal and jitter, all parameters
   sampled jointly with BOTH anvil samplers; truth recovered within
   uncertainties; ChEES-HMC with zero divergences; cross-sampler
   posterior agreement.
3. **Performance**: report ESS/s vs the white-noise fiducial run, and
   likelihood-evaluation throughput vs a per-chain-loop celerite2 CPU
   baseline (target: ≥ 100× at 1024 chains × 10⁵ points).
4. Thermal note: sustained-GPU benchmarks on this M2 Max vary ~2× with
   thermal state; benchmark rested, report medians.

## Project mechanics

- New sibling repo directory `anvil-gp/` (import name `anvilgp`), own
  `pyproject.toml`; depends on mlx + numpy; anvil imported lazily for
  the integration layer (mirror `metalplanet/anvil.py`); celerite2 is a
  test-only dependency. Use the existing venv pattern
  (`.venv/bin/python`, editable installs).
- Milestones, each independently green: (M1) kernel-term algebra +
  pure-MLX reference implementation (whatever formulation), fp64 parity
  vs celerite2 at small N; (M2) the Design A vs Design B (vs alternative
  kernel class) decision benchmark — a short written record of the
  measurements and the choice; (M3) the fast path at full scale +
  parity + precision gates; (M4) gradients + ChEES-HMC integration;
  (M5) joint injection-recovery example + docs + benchmark report.
- MLX gotchas already paid for (do not rediscover): numpy-scalar
  poison; `.item()`/`np.array()` force-evals inside compile;
  both-branch sanitization for `mx.where` gradients including masked
  *denominators*; MLX fp64 `sin`/`cos`/`exp` are only float32-accurate
  (see `MetalPlanet/metalplanet/trig.py` — you will need accurate fp64
  exp/cos for the verification path of damped oscillators); `mx.compile`
  retraces on shape change; per-op dispatch floor ~0.2–0.7 ms.
