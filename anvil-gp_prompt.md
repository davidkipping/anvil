# Task: build `anvil-gp` — batched Gaussian-process likelihoods for the anvil MCMC engine on Apple Silicon

**Revision 2.** This brief was reviewed by three independent expert agents
(GPU/Metal architecture; GP methodology; adversarial red team — the last
built and *measured* a prototype kernel and probed MLX 0.32.2 directly).
Their convergent findings are folded in below; measured numbers are
labeled as such. Where this brief prescribes rather than asks, the
prescription is measurement-backed — do not relitigate it without new
measurements.

## Mission

Time-series GP likelihoods — celerite-class kernels (sums of damped
exponential/SHO terms) — evaluated for **thousands of parameter vectors
per call** on the Apple GPU, with gradients, so that anvil samples **GP
hyperparameters JOINTLY with mean-model parameters**. Joint sampling is a
hard requirement: designs that require freezing/pre-fitting
hyperparameters are out of scope and must not shape the architecture.

Primary use case: transit/RV fitting where the mean model is MetalPlanet
(`metalplanet.anvil.make_quad_transit_flux`) and the GP absorbs stellar
variability + instrument systematics. Scales: N = 10⁴–10⁵ points
(N = 10⁶ is a stretch goal — see Watchdog note), J (kernel latent
dimension) 2–8, 10–25 total parameters, 1024–4096 chains. Small-N RV
datasets (N ~ 10²–10³) are in scope via a trivial secondary backend
(§Backends).

## Context you must absorb first

Read, in order (paths relative to
`/Users/dkipping/Storage1/Work/Documents/Transit_Work/CODES/`):

1. `anvil/README.md`, `anvil/docs/precision.md` — the engine and its
   float32 discipline.
2. `anvil/anvil/precision.py` — `ChunkedGaussianLogLike` is the
   white-noise likelihood you *replace*. NOTE: its `.hi` tiles over the
   data axis; a GP factorization is global-sequential over data, so your
   `.hi` tiles over CHAINS only (≤16-chain fp64 tiles), streaming the
   data axis in recursion order. Budget ≈1 GB peak (history: an untiled
   fp64 pass once transiently allocated 49 GB).
3. `anvil/anvil/logdensity.py`, `anvil/anvil/transforms.py` — the
   `LogDensity` contract and the ParamSpec/Transform layer (half-bounded
   exp-mapped specs exist and are what log-scale hyperparameters use).
4. `MetalPlanet/README.md`, `MetalPlanet/docs/sampler-integration.md`,
   `MetalPlanet/metalplanet/metal.py` — the proven layering
   (MLX-graph reference + fused Metal fast path behind
   `mx.custom_function` + fp64 oracle) and what `mx.fast.metal_kernel`
   usage actually looks like. **Occupancy caveat**: MetalPlanet's
   kernels parallelize over chains × points (~10⁷ threads); its
   "thread-per-chain" label does NOT license a 1024-thread kernel
   (see The Architecture).
5. `anvil/examples/metalplanet_hotjupiter.py` — the end-to-end pattern
   your deliverable extends.

## Statistical foundation (state this in your docs)

Every celerite kernel is exactly a linear Gaussian state-space model
(CARMA correspondence; closed-form stationary covariance P∞ for
initialization). The Kalman prediction-error decomposition and the
celerite semiseparable Cholesky compute the *same* marginal likelihood —
identical log-det, identical quadratic form, in exact arithmetic. The
choice between them is purely numerics + GPU scheduling, and the
reference implementation and fast path need not share a recursion — they
need only agree numerically. Bonus of the SSM route (adopt it): exact
Matérn-3/2 and Matérn-5/2 terms (not celerite-representable), and a
propagator that is **smooth through Q = 1/2** — the celerite coefficient
parametrization is singular at critical damping (1/√(4Q²−1)), which a
sampler crossing Q = 1/2 in fp32 cannot tolerate; build Φ(Δt) from the
2×2 oscillator block with cos and sinc-form sin(dΔt)/d (series switch at
small d) instead.

## The architecture (measurement-settled; three layers)

**Layer 1 — hot path: a blocked-scan Metal kernel ("Design C").**
Neither of the two "obvious" designs survives measurement:

- *Thread-per-chain sequential kernel (Design A)*: REFUTED as primary.
  A compute-shape-faithful synthetic kernel measured **212 ms** forward
  at 1024 chains × 10⁵ × J=4 (`metal::precise`), **flat from 1024→4096
  chains** (pure latency-bound; ~9% GPU utilization — 1024 threads is
  ~1 simdgroup per core on a 30-core M2 Max), i.e. only ~26× the
  measured celerite2 CPU loop baseline (5.8 s) and ~5–10× at J=8 —
  an order of magnitude short of the ≥100× gate.
- *Pure-MLX associative scan over all N (Design B)*: REFUTED as primary
  by arithmetic. MLX has no scan primitive (verified); a hand-built
  log₂N-stage scan materializes ~2m²+3m floats/point of elements —
  **21 GB at m=4, 86 GB at m=8** for 1024 chains × 10⁵ — and moves
  ≥100 GB/eval against ~400 GB/s bandwidth. Also verified:
  `mx.linalg.cholesky`/`inv` are **CPU-only with no VJP** on MLX 0.32.2,
  so the combine's small-matrix inverses must be hand-rolled closed
  forms.

**Design C fuses their strengths** (this is the Mamba/S5 chunked-scan
pattern, with published GP precedent — Corenflos, Zhao & Särkkä 2021):

1. Pass 1 (Metal): threads = chains × data-blocks (block ≈ 128–1024
   points → ~10⁵–10⁶ threads: full occupancy; the machine absorbed 32×
   more threads for 2.9× the time in the probe). Each thread
   sequentially *condenses* its block into one affine-Gaussian scan
   element (A, b, C, η, J, log-normalizer). Per-step observation
   updates are rank-1 (Sherman–Morrison keeps it O(m²)/point);
   celerite propagators are block-diagonal 2×2 damped rotations, so the
   element stays register-resident.
2. Pass 2 (tiny): compose the ~10²–10³ block elements per chain
   (sequential per-chain kernel or small tree — negligible cost).
3. Pass 3 (Metal): re-run each block seeded by its prefix state,
   accumulating innovation log-variances and normalized residuals.
4. Backward: block boundaries are the checkpoints (~10² MB total);
   the backward kernel recomputes within blocks in reverse order.
   NEVER reconstruct forward state by inverse recursion (division by
   the propagator overflows exactly where it underflowed — data gaps).

Wrap the whole thing in `mx.custom_function` with the hand VJP, exactly
as `metalplanet/metal.py` does. Predicted forward at 1024 × 10⁵:
~10–50 ms (`metal::precise`) — 100–500× the CPU-loop baseline.
Investigate `metal::fast::` transcendentals as a measured **4.5× lever**
(probe data) — unlike MetalPlanet, transcendentals dominate this
kernel; quantify the accuracy cost before adopting.

**Layer 2 — reference path: the associative-*reduction* formulation in
pure MLX.** Key correction from review: the likelihood needs only a
tree *reduction* to the root (elements with log-normalizers, composed
pairwise, terminated against P∞), NOT a prefix scan — total allocation
≈ 2× the leaf level, log-det summed as a pairwise tree (the
fp32-friendliest accumulation). This is the dtype-polymorphic MLX-graph
reference the layering demands: it IS the `.hi` fp64 path (tiled ≤16
chains), the fp32 parity baseline, and the same block-composition
algebra the Metal kernel uses — one algebra, two executors. Run it
fp64-on-CPU and fp32-on-GPU from the same code.

**Layer 3 — oracles:** celerite2 (CPU fp64; verified: arm64 wheel
installs and runs on this machine, 5.6–6.0 ms per N=10⁵ compute+logL;
accepts duplicate timestamps, rejects unsorted) for values.
**celerite2's numpy backend exposes no gradients** — gradient parity
needs `celerite2.jax` (+ `jax[cpu]` as a test-only dependency) or fp64
central differences. For hyperparameter corners where celerite2 itself
degrades (near-critical Q, near-duplicate timescales), the oracle is a
dense fp64/mpmath Cholesky at N ≤ 10³.

**Pre-registered numerical fallback:** if fp32 covariance-form element
composition fails the precision gates, switch to square-root (Cholesky-
factor) elements — parallel square-root filters are published (Yaghoobi
et al., SIAM SISC; arXiv:2406.05188) at ~2–3× flops. Decide by
measurement at M2, not by taste.

## fp32 prescriptions (hard requirements, not contingencies)

1. **Reference-subtracted log-det.** log|K| ≈ −1.5×10⁶ at N=10⁵ and
   500 ppm errors; naive fp32 accumulation loses ~20 logL units
   (arithmetic in review). Accumulate `log(d_n / yerr_n²)` — an
   O(0.01–5) quantity — in-thread with Kahan compensation, and add the
   fp64 host constant Σ log(yerr_n²) outside the graph. This is anvil's
   offset discipline applied to the log-det.
2. **Quadratic form** (≈ N): compensated in-thread accumulation in the
   Metal kernel; pairwise tree in the reduction path.
3. **Process noise via expm1**: never Q_step = P∞ − Φ P∞ Φᵀ
   (catastrophic cancellation at ω₀Δt ≪ 1); use closed forms on
   `expm1(−2cΔt)`. (Verified: MLX fp64 `expm1` and `log` ARE true fp64;
   fp64 `exp`/`sin`/`cos` are only fp32-accurate — an fp64-accurate
   `exp` must be WRITTEN for the verification path
   (Cody–Waite + 2ᵏ scaling; `metalplanet/trig.py` covers sincos only —
   budget this in M1).)
4. **Positive-definiteness failure is a defined, gradient-safe
   outcome.** With 4096 chains, warmup WILL visit fp32-infeasible
   corners (σ_GP/yerr up to 10–300 with timescale ≫ cadence makes the
   innovation-variance update a catastrophic cancellation; φ rounds to
   exactly 1.0 for cΔt ≲ 6×10⁻⁸). Requirements: floor d_n at
   ~yerr²·ε with both-branch `mx.where` sanitization; on failure return
   large-negative *finite* logL with zeroed gradients; test that chains
   initialized in the failure region recover; **map the fp32
   feasibility boundary** over (σ_GP/yerr, timescale/cadence, Q) and
   derive documented ParamSpec bound guidance from the map.
5. **Error-growth study design**: celerite kernels are exponentially
   forgetting — recursion error *saturates* at a level set by
   τ_damp/Δt, not N. Scan τ/Δt and τ/T (long-memory kernels, τ ≳ T,
   are the danger axis); a flat error-vs-N curve is the expected
   result, not a suspicious one. Decompose reported error into log-det
   vs quadratic-form contributions (they fail differently).

## Contract

```python
class GPLogLike:
    def __init__(self, mean_fn, t, x_mean, y, yerr, kernel_spec,
                 param_map, policy=None): ...
    def __call__(self, v: mx.array) -> mx.array   # (n_chains, dim) -> (n_chains,)
    def hi(self, v: mx.array) -> mx.array          # fp64 CPU, CHAIN-tiled
```

- `t` is the strictly-sorted physical time vector (fp64 host; Δt
  precomputed fp64, entering the graph as well-conditioned fp32);
  `x_mean` is the mean model's own abscissa (e.g. MetalPlanet's (2, m)
  epoch-centered channels). These are DIFFERENT arrays — do not
  conflate them.
- `param_map` maps slices of `v` to roles: mean-model block, per-term
  hyperparameters, **per-instrument jitters and mean offsets (v1
  features — the first thing real users need)**. Jitter enters the
  observation variance per chain; `yerr` alone is not the diagonal.
- Duplicate timestamps (Δt = 0) are legal (celerite2 accepts them) and
  must not trip the φ-hazards; unsorted input raises.
- Include the full normalization: log|K| depends on hyperparameters.
  Parity tests compare ABSOLUTE logL (a wrong constant is invisible to
  MCMC and to Δ-based checks).
- Mean-model gradients come almost free: ∂logL/∂mean = α = (K+Σ)⁻¹r,
  chain-ruled into MetalPlanet's existing analytic VJP. State this
  split — it decouples the hard (hyperparameter) gradient engineering
  from the solved (transit) path.
- Engine rules: pure MLX/Metal hot path, `mx.compile`-safe, fixed
  iteration counts, no numpy scalars in the graph, no `.item()` in
  compiled code. Pad tree sizes to powers of two to avoid retraces.
- Conditional mean / `predict` for detrending figures: OUT of the hot
  path; an offline celerite2 fp64 predict is fine for M5's plots and
  should be provided as a documented utility.
- 2D/multiband GPs: explicitly out of scope for v1.

## Kernel set (v1)

SHO term (smooth through Q = 1/2 via the SSM propagator), real/exp
term, **RotationTerm composite** (two SHOs at P and P/2 with mixture
fraction — the celerite2/exoplanet standard), Matérn-3/2 and Matérn-5/2
(free on the SSM route), sums of terms, per-instrument jitter. For
sums of same-type terms, impose frequency ordering to kill permutation
multimodality — if anvil's Transform layer lacks an ordered-vector
transform, file/implement it as an engine work item. Exposure-time
integration of the kernel (long-cadence smearing) is a documented known
limitation for v1 (S+LEAF handles it; cite as the v2 path).

## Parametrization and priors (ship as defaults, not user homework)

- Per SHO term sample **(log σ, log ρ, log Q)** with σ² the stationary
  variance and ρ = 2π/ω₀ — never (S0, ω0, Q) raw (high-Q posteriors are
  axis-aligned in the former, curved ridges in the latter).
  Granulation: fix Q = 1/√2, sample (σ, ρ). Rotation: (log σ, log P,
  logit f, log Q, log ΔQ) per celerite2 convention.
- Sample log-parameters as **offsets from fp64 host references**
  (anvil rule 2 applied to hyperparameters) — conditions the graph and
  centers the shared preconditioner.
- Default proper priors (overridable): log σ_GP ~ N(log RMS(y), 1.5²);
  log jitter ~ N(log median(yerr), 1²); log ρ uniform on
  [log 2Δt_med, log T/2]; log Q ~ N(log 3, 1²) truncated ≲ few×10².
  Rationale to document: with thousands of chains, improper flat
  amplitude priors guarantee stray chains on the σ→0 ridge that drag
  the harmonic-mean step-size adaptation down for everyone — **prior
  propriety is a throughput feature**.
- Named identifiability ridges (document each): σ_GP–jitter as
  ρ → cadence; GP–baseline as ρ → T; GP–transit-depth when ρ ~ transit
  duration; period harmonics (P, P/2, 2P) multimodality — seed chains
  within one periodogram-selected mode (a host-side FFT/periodogram
  initializer is the legitimate use of Whittle machinery; the Whittle
  likelihood itself fails the exactness rule).
- Geometry note (calibrates fears): the marginalized GP has NO
  Neal's-funnel latent hierarchy — remaining geometry is mild ridges
  that ChEES's shared diagonal preconditioner + trajectory adaptation
  handle, conditional on the parametrization above.

## Backends beyond the primary (scoped)

- **Small-N dense**: batched dense Cholesky per chain for N ≲ 10³ (RV
  datasets): 1024 × 512² fp32 ≈ 1 GB — near-free to add, strictly
  better there. Hand-rolled fixed-form kernel or mlx-addons-style; core
  MLX Cholesky is CPU-only.
- **HSGP** (Solin & Särkkä basis GPs): admissible only as a *secondary
  cross-check* for low-Q smooth kernels (m ≲ 100). Blockers, recorded
  so it is not re-litigated: m scales ~linearly with Q (sharp spectral
  peaks) and with baseline/timescale ratio; published (m, c) guidance
  is not calibrated for quasi-periodic kernels; approximation deficit
  launders into biased Q / inflated jitter invisibly to prediction
  checks; requires per-chain m×m factorizations MLX cannot GPU-execute
  natively. Any use requires posterior-level A/B against the exact
  likelihood.
- REJECTED (fail the exactness rule; do not revisit): FFT/Toeplitz &
  Whittle (gaps break Toeplitz; quasi-likelihood ≠ posterior),
  SKI/BBMM/stochastic-Lanczos log-dets (noisy likelihood breaks
  Metropolis; not valid pseudo-marginal), RFF, SVGP.

## Milestones

- **M1 — algebra + reference path.** Kernel-term algebra (SSM
  propagators via expm1/sinc forms), the tree-reduction reference
  implementation (dtype-polymorphic), fp64-accurate `exp` helper, fp64
  parity vs celerite2 at N ∈ {10³, 10⁴} including absolute logL, sums
  of terms, duplicate timestamps, Q ∈ {0.1, 1/√2, 1/2 ± ε, 10, 100}.
- **M2 — decision benchmark (day-scale, not week-scale).** Design B
  full-scale is settled by the arithmetic above — do NOT fully
  implement it; benchmark: (1) forward wall-time grid over
  n_chains ∈ {256, 1024, 4096, 16384} × N ∈ {10⁴, 10⁵} × m ∈ {2, 4, 8}
  for Design C vs the Design-A probe, with the chain-scaling curve as
  the occupancy diagnostic; (2) value+grad wall time AND
  `mx.get_peak_memory`; (3) fp32 error scaling per the saturation
  design, decomposed; (4) roofline placement (achieved GB/s and
  FLOP/s) to validate the cost models; (5) end-to-end compiled
  ChEES leapfrog cost + trace/compile time + dispatch count; plus the
  `metal::fast` accuracy study and the celerite2 baselines
  (single-core loop AND 12-core multiprocess — the honest ≥100×
  denominator is the multiprocess one).
- **M3 — fast path at scale.** Blocked-scan kernel, parity vs the
  reference, precision gates, fp32 feasibility map, PSD-failure
  behavior tests.
- **M4 — gradients + engine integration.** Checkpointed backward
  kernel; gradient parity vs celerite2.jax (or fp64 central
  differences) at ~10³ points including Q ≪ 1, Q ≫ 1, ω₀Δt ≫ 1,
  ω₀T ≫ 1, jitter → 0; zero NaNs across 10⁵ draws; ChEES-HMC runs.
- **M5 — joint validation + docs.** (a) Fiducial: MetalPlanet hot
  Jupiter + injected SHO variability + jitter, both samplers, truth
  recovered, zero divergences, cross-sampler agreement. (b)
  **Adversarial geometry tests**: zero/weak-amplitude injection
  (posterior piles against the bound, zero divergences) and
  timescale ≈ cadence injection (σ_GP–jitter ridge sampled correctly).
  (c) **Calibration**: SBC rank checks at N = 10³ via the fp64
  reference path (or ≥20 injection realizations with coverage
  counting) — one recovery at one truth has ~no power against subtle
  likelihood bias. (d) Docs, benchmark report, ESS/s vs the
  white-noise fiducial.

**Watchdog note**: macOS kills GPU dispatches at ~5 s (unconfigurable).
Segment data loops so no single dispatch approaches it; N = 10⁶ is a
stretch goal contingent on the blocked kernel + dispatch splitting.

## Project mechanics

Sibling repo dir `anvil-gp/` (import name `anvilgp`); depends on
mlx + numpy; anvil imported lazily (mirror `metalplanet/anvil.py`);
test-only deps: celerite2, `jax[cpu]` (gradient oracle), pytest.
Existing venv pattern. Thermal note: sustained-GPU benchmarks on this
M2 Max swing ~2×; benchmark rested, medians of ≥7, isolated processes.

MLX gotchas already paid for (do not rediscover): numpy-scalar poison;
`.item()`/`np.array()` force-evals in compile; both-branch `mx.where`
sanitization INCLUDING masked denominators; `mx.compile` retraces on
shape change; ~0.2–0.7 ms dispatch floor; fp64 exp/sin/cos are
fp32-accurate (log/expm1 are true fp64); no scan primitive; linalg
CPU-only without VJPs.

## Prior art (verified: no GP implementation exists on MLX — you are first)

Foundations: Särkkä & García-Fernández 1905.13002 (temporal
parallelization); Corenflos, Zhao & Särkkä 2102.09964 (parallel-in-time
GP regression — Design B/C's published precedent, >10× GPU);
Foreman-Mackey et al. 1703.09710 (celerite), 1801.10156 (backprop);
Kelly et al. 1402.5978 (CARMA/SSM); Yaghoobi et al. 2207.00426 +
2406.05188 (parallel square-root filters — the fallback); 2502.11686
(orthogonal-transform smoothers); 2511.10363 (GPU prefix-sum Kalman
benchmarks incl. Metal). Implementations to read: tinygp quasisep (JAX;
its docs *require fp64* for the serial celerite solver — a warning),
celerite2 + celerite2.jax, dynamax parallel LGSSM,
tfp.experimental.parallel_filter, EEA-sensors/parallel-gps,
S+LEAF (2201.02440; integrated kernels, multi-series), mamba.py (MLX
pscan mechanics), S5 2208.04933 / Mamba 2312.00752 (chunked-scan kernel
pattern). HSGP: Solin & Särkkä 1401.5508; Riutort-Mayol et al.
2004.11408. MLX gaps: issues #1781 (Metal Cholesky NYI), #1238
(Metal inv); mlx-addons (third-party batched Metal linalg).
