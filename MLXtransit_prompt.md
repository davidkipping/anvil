# Task: build `mlxtransit` — an MLX-native, fully differentiable transit light-curve model (Agol, Luger & Foreman-Mackey 2019)

You are implementing a ground-up exoplanet transit light-curve model in
Apple's MLX array framework, designed from day one to plug into an
existing MCMC engine called **anvil** (already built and tested, in
this same directory tree). Read this whole document before writing code —
it encodes hard-won constraints from building the engine, and violating
them produces failures that are painful to diagnose.

## Why this model, why MLX

The engine (`anvil`) runs thousands of MCMC chains on the Apple
Silicon GPU in float32, with float64 conditioning handled on the CPU. Its
flagship sampler is ChEES-HMC, which needs gradients of the
log-likelihood. Its current demonstration model is a trapezoid — kinky,
plateau-ridden, a deliberate worst case. The real science model is
**Agol, Luger & Foreman-Mackey 2019, AJ 159, 123
(https://arxiv.org/abs/1908.03222; "ALFM19")**: analytic, closed-form
transit flux for polynomial limb darkening of arbitrary order, with
**analytic partial derivatives** with respect to the radius ratio r, the
impact parameter b, and every limb-darkening coefficient, plus
numerically stable formulations of every regime. It is the ideal target
because it is smooth (HMC-friendly), it needs no special functions beyond
complete elliptic integrals (computable with fixed-iteration algorithms —
GPU-friendly), and its analytic derivatives let us bypass
reverse-mode autodiff's memory cost with a custom VJP.

Fetch and work from the actual paper (and its reference implementations —
Agol's `Limbdark.jl`, and the `jaxoplanet` package, which is the closest
architectural cousin: a JAX implementation facing identical
vectorization/differentiability constraints). Do not trust equation
numbers quoted from memory — verify against the paper.

## The integration contract (non-negotiable)

anvil consumes models through this exact shape (see
`anvil/precision.py::ChunkedGaussianLogLike` and
`anvil/targets/builtin.py` for the working trapezoid example):

```python
model_fn(v: mx.array, x: mx.array) -> mx.array
# v: (n_chains, n_params)  — one parameter vector per chain
# x: (n_abscissa_channels, m) or (m,) — per-datum quantities for ONE chunk
# returns: (n_chains, m) predicted flux (or flux deviation)
```

Requirements:

1. **Batched over chains.** Slice parameters as `v[:, i:i+1]` (keepdims)
   and broadcast against `x[..., None, :]`-style time axes. Never loop
   over chains.
2. **Dtype-polymorphic.** The same function must run float32 (GPU
   production path) and float64 (CPU verification path). Never hard-code a
   dtype; derive constants from input dtypes or use Python floats. The
   engine's precision harness (`anvil.validate_precision`) runs your
   model both ways and reports the error — it must pass.
3. **Pure MLX ops, compile-safe.** The function is traced by
   `mx.compile`. Absolutely no `numpy` calls, no `.item()`, no
   `np.array(mx_array)`, no data-dependent Python `if`/`while` on array
   values inside the model. All regime selection is `mx.where` on masks.
4. **No numpy scalars in the graph.** A closure constant that is
   `np.float64` (e.g. anything read out of a numpy array without
   `float(...)`) silently routes MLX ops through numpy's ufunc machinery
   and force-evaluates mid-compile, throwing
   "Attempting to eval an array during function transformations".
   Convert every captured constant to a plain Python `float`/`int`.
5. **Fixed iteration counts.** Any iterative algorithm (Bulirsch `cel`
   for elliptic integrals, Newton/Householder for Kepler's equation) must
   run a *fixed* number of iterations chosen for float32 convergence
   (these algorithms converge quadratically; ~8–12 iterations is
   typically enough — verify). Data-dependent stopping breaks batching
   and compilation.

## The float32 conditioning discipline (learned the hard way)

The engine samples in float32 on a GPU with no float64. Two conditioning
bugs cost the trapezoid model ~10 units of log-likelihood before being
found by the precision harness; bake the fixes in from the start:

1. **Offset parameters, never absolutes.** Time-like parameters enter the
   model as offsets from float64 references handled outside the graph:
   `t0_off = t0 - t0_ref`, `p_off = P - P_ref`. Rationale: an absolute
   period P ~ 3.5 d has float32 representation error ~2.4e-7 d, and orbit
   number k multiplies it (k·δP ≈ 4e-6 d of phase error by orbit 26) —
   which the residual weight 1/σ amplifies thousands of times.
2. **Epoch-centered abscissae.** The float64 CPU preprocessing assigns
   each datum an orbit number `k = round((t - t0_ref)/P_ref)` and a
   residual `dt = t - t0_ref - k·P_ref` ONCE; the graph computes
   `phase = dt - (t0_off + k·p_off)` — all O(1) quantities. See
   `epoch_center_times` and `make_offset_flux` in
   `anvil/targets/builtin.py` and imitate the pattern.
3. **Return flux deviation.** Predict `f - 1` (matching baseline-
   subtracted data), so the ordinate is O(depth), not O(1). Offer a
   convenience wrapper that adds the baseline back for standalone use.

## Algorithmic roadmap (verify details against ALFM19)

Structure the computation the way the paper does:

- **Separation of concerns:** an orbit module maps
  (dt, k; t0_off, p_off, and orbit shape parameters) → sky-projected
  separation z(t) in stellar radii; the photometric core maps
  (z, r, limb-darkening coefficients) → flux. Keep these independently
  testable.
- **Green's-basis decomposition:** the limb-darkening profile (polynomial
  in μ of order N) is transformed to a basis in which the occulted flux
  is `F = Σ_n g_n · s_n(r, b)`; the `s_n` solution vector is where all
  the work lives. The g-transformation is a small linear map — precompute
  its matrix on the host in float64.
- **Base cases:** `s_0` (uniform disk: the classic lens-shaped
  overlap area, arccos/sqrt forms) and `s_1` (linear limb darkening:
  complete elliptic integrals). Higher `s_n` follow recursion relations
  upward from the base cases. For a v1 that covers most practical use,
  quadratic limb darkening (N=2, i.e. s_0, s_1, s_2) is sufficient —
  design the API for arbitrary N but ship N ≤ 2 first, then extend.
- **Elliptic integrals via Bulirsch's `cel`** (as in the paper and
  Limbdark.jl): an iterative arithmetic-geometric-mean-style algorithm.
  Implement it batched, dtype-polymorphic, with a fixed iteration count;
  this single routine is the numerical heart — unit-test it against
  scipy.special (ellipk/ellipe and the general cel) across the argument
  range the transit geometry produces, in both dtypes.
- **Regime handling:** the geometry has distinct analytic regimes — no
  overlap (z ≥ 1+r), full transit annulus (z ≤ 1−r), partial overlap,
  planet covering stellar center, total occultation for r > 1 (not
  needed for planets — you may cleanly restrict to r < 1). ALFM19 give
  reformulations that are stable near the boundaries (z ≈ 1±r, z ≈ r,
  z ≈ 0, r ≈ 0.5). Implement regimes as masked `mx.where` branches, and
  read the paper's stability sections carefully — the naive formulas lose
  catastrophic precision exactly at ingress/egress in float32, which is
  where transit timing information lives.

### The double-`mx.where` trick (critical for gradients)

`mx.where(mask, f(x), g(x))` evaluates BOTH branches everywhere. If the
inactive branch produces `nan`/`inf` (e.g. `sqrt(negative)`,
`acos(>1)`), the *gradient* becomes NaN even though the forward value is
fine — NaN × 0 = NaN in the backward pass. Every regime branch must
sanitize its inputs first:

```python
safe = mx.where(mask, arg, benign_value_inside_domain)
out  = mx.where(mask, branch_fn(safe), other_branch)
```

Apply this systematically; it is the number-one source of silent HMC
divergences in JAX transit codes and will be here too. Clamp every
`sqrt`, `acos`, `log`, and division argument.

## Gradient strategy (two stages)

1. **Stage 1 — autodiff-clean.** Write the model so `mx.grad` works
   end-to-end (that is what the double-where discipline buys). anvil
   gets per-chain gradients via
   `mx.vjp(model, [u], [ones])` — one forward+backward for the whole
   batch. Validate: gradients vs central finite differences in float64
   on the CPU stream at ~1e3 random points covering all regimes,
   including points *at* regime boundaries.
2. **Stage 2 — analytic custom VJP.** ALFM19 provide closed-form
   ∂F/∂r, ∂F/∂b, ∂F/∂u_n. Wrap the photometric core in
   `mx.custom_function` with a `.vjp` built from these formulas.
   Motivation (measured in anvil): reverse-mode autodiff through the
   elementwise graph is memory-bandwidth-bound — the backward pass costs
   ~10× a fused forward at 1024 chains × 1e5 points. An analytic VJP
   computes the three partials in the forward-pass style and contracts
   them directly, and should roughly halve the leapfrog cost of
   ChEES-HMC. Keep Stage 1 as the correctness oracle for Stage 2.

## Parameterization (for the sampling layer, not the core)

The photometric core takes physical (r, b or z, u_1..u_N). Provide
transform helpers for the sampling-friendly parameterizations, because
anvil samples in unbounded space via `anvil.transforms.ParamSpec`:

- **Quadratic limb darkening: offer the Kipping (2013) (q1, q2)
  triangular parameterization** (uniform box in [0,1]² maps to exactly
  the physical u1,u2 region). Implement q→u as MLX ops inside the model
  wrapper so it is differentiable, with the (trivial, constant) Jacobian
  handled by the bounded ParamSpec transforms.
- Orbit: v1 = circular orbit parameterized by
  (t0_off, p_off, r, b, a/R★ or duration-based variant — pick one,
  document why). v2 = eccentric orbits via a fixed-iteration Kepler
  solver (Newton or the Markley/Householder scheme; fixed 3–5 iterations
  from a good starter is plenty and stays differentiable; jaxoplanet's
  solver is prior art worth reading).

## Validation plan (write these tests as you go)

1. **cel unit tests** vs scipy across the full argument range, fp32 and
   fp64.
2. **Limits and identities:** r→0 gives flux ≡ 1; z ≥ 1+r gives exactly
   1; z ≤ r−1... (not applicable for r<1); uniform LD (all u=0) matches
   the closed-form lens area; total flux conservation of the LD profile
   normalization; symmetry F(z) even in z sign convention.
3. **Cross-code check:** validate flux against `batman`
   (Kreidberg 2015, quadratic case, crank its precision up) and/or
   `jaxoplanet`'s `limb_dark_light_curve` at ~1e4 (r, b, u1, u2, z)
   combinations; target fp64 agreement ~1e-9 relative, fp32 within a few
   ×1e-6 of its own fp64 path.
4. **Precision harness:** build the target with
   `anvil.ChunkedGaussianLogLike` + `anvil.TransformedLogDensity`
   (`model_log_prob_hi=loglike.hi` for the float64 path) and require
   `validate_precision` to report max |Δ logL| well under 1 for a
   100k-point synthetic light curve — the trapezoid achieves ~0.2;
   beat it, since this model is smoother.
5. **Gradient checks** as in Stage 1 above, plus: zero NaNs in gradients
   across 1e5 random points including regime boundaries (this catches
   double-where violations).
6. **End-to-end injection–recovery:** synthesize a Kepler-like light
   curve (absolute BJD times ~2.457e6 — the conditioning must survive
   this), fit with both `anvil.HMCSampler` (expect zero divergences
   now that the model is smooth — this is the acceptance criterion the
   trapezoid could not meet) and `EnsembleSampler`; recover truth within
   uncertainties; compare ESS/s of both.

## Project mechanics

- New package directory `mlxtransit/` (sibling of `anvil/` inside
  this repo, own `pyproject.toml`, depends on `mlx`, `numpy`;
  `anvil` is a test/integration dependency only — the core must be
  importable standalone).
- Use the existing venv: `.venv/bin/python`, `.venv/bin/pip`,
  `.venv/bin/python -m pytest`. Machine: M2 Max, MLX ≥ 0.32.
- Read first: `anvil/targets/builtin.py` (the integration pattern to
  imitate), `anvil/precision.py`, `anvil/transforms.py`, and the
  MLX-gotchas notes in `README.md`.
- Milestones, each independently green: (M1) `cel` + uniform-source
  s_0 with tests; (M2) s_1 + quadratic s_2, flux assembly, cross-code
  validation; (M3) autodiff cleanliness + gradient tests; (M4) anvil
  integration target + precision harness + injection–recovery; (M5)
  analytic custom VJP + benchmark vs autodiff; (M6, optional) arbitrary
  N, exposure-time integration (ALFM19 §exposure-averaging), eccentric
  orbits.
- Benchmark culture: measure before optimizing; the engine's benchmarks
  (`benchmarks/`) show the patterns. Sustained-GPU thermal throttling on
  this machine swings timings ~2× — benchmark on a rested machine.

## Success criteria

A smooth, science-grade, MLX-native transit model that (1) matches
reference codes to fp64 tolerance, (2) passes the fp32 precision harness
with margin, (3) runs ChEES-HMC with zero divergences and beats the
gradient-free stretch move in ESS/s on the same problem — the smooth
model is where HMC should win outright — and (4) exposes a clean
standalone API others can adopt independent of the sampler.
