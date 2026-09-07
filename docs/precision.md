# The float32 discipline

Metal GPUs have no float64 hardware. anvil's position: float32 MCMC is a
*conditioning* problem, not a hardware defect — and it hands you a
working discipline plus the instrument to verify it. This page is the
distilled version; every rule below traces to an error the
{func}`anvil.validate_precision` harness actually caught during
development.

## Why float32 goes wrong (concretely)

Two real failures from anvil's own transit test model, both invisible
without the harness:

1. **Whole-baseline phase folding.** Computing `(t - t0) mod P` with `t`
   spanning 90 days quantizes time at ~0.7 s in float32. Amplified by the
   residual weight `1/σ`, that cost ~12 units of log-likelihood over
   10⁵ points.
2. **Absolute parameters.** With an absolute period, the model computes
   `k · (P − P_ref)` per datum; float32 representation error of
   `P ≈ 3.5 d` is ~2.4×10⁻⁷ d, and orbit numbers `k ≲ 26` multiply it
   into a systematic phase error — again tens of log-likelihood units.

The Metropolis accept/reject decision operates on Δ log-likelihood at a
scale of order 1. Errors approaching that scale distort the sampled
posterior.

## The rules

1. **Preprocess in float64 on the CPU.** Center times to a reference
   epoch, normalize fluxes, subtract baselines, precompute per-datum
   constants (orbit numbers, epoch-centered residuals). Only
   O(1)-conditioned float32 arrays enter the GPU graph.
2. **Sample offsets, never absolutes.** Parameters like epochs, periods,
   and baselines enter the model as offsets from float64 references
   (`t0 − t0_ref`, `P − P_ref`, `f0 − 1`). Set
   `ParamSpec(report_offset=...)` and the reporting layer reinstates
   absolute units in float64 on output. Measured on the 100k-point
   transit target at the posterior typical set, rules 1–2 take the
   median error from **2.3 to 0.0015** log-likelihood units — a factor
   of ~1500, and the difference between a `WARNING` verdict and an `OK`
   one.
3. **Fit the deviation, not the signal-plus-baseline.** Predict `f − 1`
   against baseline-subtracted data so the ordinate is O(depth).
4. **Recentre the likelihood itself** (on by default;
   `PrecisionPolicy(recenter=False)` opts out). A chi-squared over N
   points is a sum of N terms whose expectation is −N/2, so the float32
   graph would carry a ~5×10⁴ number for a 10⁵-point fit even though
   every parameter-dependent part of it is O(√(N/2)) ≈ 200. anvil sums
   `0.5·(1 − r²)` instead of `−0.5·r²` and keeps the exact −N/2 as a
   float64 host constant (`log_offset_const`), alongside the Gaussian
   normalization it already excluded. Measured: the float32 ulp of the
   stored log-probability falls ~128×, so every difference the sampler
   takes — Metropolis ratios, and **HMC's energy difference, which
   `validate_precision` cannot observe because it is formed downstream
   of the likelihood** — is resolved that much more finely. End-to-end
   this is worth 2.8× on an accurate model and little on a model whose
   own arithmetic dominates, it costs nothing measurable, and its value
   grows with N (an 81× reduction in accumulation error at N = 10⁶).
   Two caveats: when the model fits badly (χ²/N ≫ 1) there is no large
   constant to remove and the scheme simply degrades to the plain form,
   never worse; and anything wanting an *absolute* log-likelihood must
   add `log_offset_const + log_norm_const` back in float64.
5. **Reduce with structure — but know its ceiling, which is low.**
   {class}`anvil.precision.ChunkedGaussianLogLike` sums per-datum terms
   chunk-wise (tree error O(ε√n) instead of O(εn)). The `fp64_anchor`
   policy additionally performs the cross-chunk sum in float64 on the CPU
   stream *inside the same lazy graph*, for a measured 0–20% cost — but
   read the fine print: **it only does anything when there are many
   chunks.** At the default `chunk_size=65536` over 10⁵ points there are
   two chunks, and the exact sum of two float32 values rounds identically
   whether added in float32 or in float64 — the two policies are then
   *bit-identical* (verified). Shrinking to `chunk_size=1024` makes it
   real but buys only ~6% of the error dispersion. Measured error
   contributions on the transit targets: the summation accounts for only
   2–40% of the total, the rest being per-datum model arithmetic and
   parameter representation. Reduction tricks are the smallest lever
   here; conditioning (rules 1–3) is the large one.

   `chunk_size` (default 16384) does have a second, non-accuracy role
   worth knowing: the chunk loop unrolls into a single graph, so in
   reverse mode every chunk's saved activations are live at once and
   **peak gradient memory falls roughly with the chunk size**. Measured at
   1024 chains × 10⁵ points, 65536 → 16384 costs nothing in forward time,
   is marginally faster for gradients, and cuts peak memory 1.4–1.7×. If a
   gradient-based run is memory-bound — 4096 chains × 10⁵ points reaches
   ~19 GB at 65536 — lower it further before reducing the chain count.
   Below ~4096 the dispatch count begins to cost at large N.

   `PrecisionPolicy(reduction="fixed_point")` takes the reduction as far
   as it can go: a custom Metal kernel accumulates the terms as int64
   fixed-point (multiples of 2⁻³⁰), which is **exact** — integer addition
   never rounds — and therefore *order-independent*, so the result is
   bit-identical however the data axis is chunked and whether or not the
   graph is compiled. The float32 tree is neither. It costs nothing
   measurable, and the only rounding left is the unavoidable one of
   returning a float32.

   Ship it only if you want that reproducibility, because **it buys no
   measured end-to-end accuracy on any target bundled here.** Recentring
   (rule 4) already dropped the reduction below the dominant term, which
   is the float32 cancellation in the residual `y − m` itself: for a
   well-fitting model the residual is a small difference of two O(1)
   numbers, so its relative error is set by the *data* magnitude, not by
   the residual's. No summation algorithm can reach that, and it is where
   the remaining error lives.
6. **Re-anchor — for adaptive densities, not for float32 rounding.**
   `run(..., reanchor_every=N)` recomputes the cached log-probabilities of
   the current states through the float64 CPU path every N iterations. Be
   clear about what this does and does not buy. anvil never *accumulates*
   a cached log-probability — every stored value is a fresh evaluation of
   a function that is bitwise deterministic and invariant to batch shape,
   position and compilation (verified) — so there is no drift mechanism
   for float32 rounding to exploit, and re-anchoring does not change the
   distribution the chain samples. It is genuine insurance where the
   log-density really can go stale: a surrogate/emulator that is retrained
   mid-run (see {mod}`anvil.surrogate`), or any non-deterministic
   evaluation. For a deterministic float32 likelihood it is expensive
   (a float64 pass can cost 1000× a float32 one for a GPU-kernel model)
   and unnecessary; the default is off.

The same principle applies *inside* the engine, and it is worth knowing
about because it once dominated everything else. A bounded parameter's
map used to be evaluated as `lo + (hi − lo)·sigmoid(u)`, whose float32
error scales with the **box width** rather than with the parameter's own
magnitude — so a tightly-constrained epoch inside a generous ±0.5-day
box inherited ~26 ulp of noise, and that single effect contributed
80–97% of anvil's total float32 error budget. It is now evaluated in
centre/half-width form, `mid + half·tanh(u/2)` — the identical map, but
with error relative to the distance from the box centre. Free, and worth
a factor of 7–56× end to end. The lesson generalizes: **prefer the
algebraic form whose rounding is relative to the quantity you care
about.**

## The instrument

```python
report = anvil.validate_precision(target, u_points)
print(report)
```

evaluates the production float32 GPU path and a full float64 CPU path at
the given states and reports the discrepancy against the ~1-unit decision
scale, with a three-tier verdict (OK / ACCEPTABLE / WARNING). Run it at
initialization points *and* at converged states before trusting any new
likelihood. The float64 path comes for free when you build on
`ChunkedGaussianLogLike` (its `.hi` method) and pass it as
`model_log_prob_hi` to {class}`anvil.TransformedLogDensity` — the same
model code runs in both precisions because MLX ops are
dtype-polymorphic.

## Certifying (and removing) the residual bias

The float32 log-density is a *deterministic* function of the parameters —
the same point always gives the same value, independent of batch shape,
row position, and compilation (all verified). A sampler driven by it is
therefore not an approximate sampler for your posterior; it is an **exact
sampler for a slightly tilted one**, `π·exp(err)`. That is a much more
tractable situation than it sounds, because the bias it puts into any
posterior mean has a closed form:

```
E_tilted[f] − E_true[f]  =  Cov(f, err)  + O(err²)
```

So evaluating `err` on a small random subset of your stored draws both
*measures* the bias and *removes* it — for a few hundred float64
evaluations over the entire run, rather than one per iteration:

```python
cert = anvil.certify(target, results.get_chain(flat=True),
                     target_ess=1e5, names=transform.names)
print(cert)                  # per-parameter bias vs Monte Carlo error
cert.corrected_mean          # bias-corrected posterior means
cert.correct(any_quantity)   # correct any derived quantity too
```

The report states the bias in units of the Monte Carlo standard error at
your target ESS, which is the only comparison that means anything —
"small" is meaningless without asking *small compared to what*:

```
  float32 error dispersion (sd) : 0.00543   [mean +0.00022, irrelevant: it cancels]
  exact-reweighting ESS retained: 0.999971
  probes used / needed          : 256 / 16
  bias at ESS = 100,000 (units of MC standard error):
              q1   +0.195   (+0.000618 posterior sd)
               r   -0.188   (-0.000595 posterior sd)
  ACCEPTABLE: bias is 0.20x the Monte Carlo standard error at ESS=100,000
```

Only the error's *dispersion* matters; a constant offset is absorbed
entirely by the normalization, which is why the mean is reported but
flagged as irrelevant. The probe count needs only `n_probe >> err_sd² ×
ESS` — often a few dozen — and the report tells you if you under-probed.

The same numbers price the exact alternative: `is_retention` is the
fraction of ESS that full importance reweighting would keep. At these
error levels it is 0.99997, i.e. reweighting is statistically free — it
is only the *evaluation* cost (one float64 call per stored draw, ~100×
more than certifying) that makes the covariance correction the better
buy.

## What errors remain, and why they are acceptable

For a well-conditioned 10⁵-point chi-squared, the residual float32 error
is ~0.002–0.005 log-likelihood units at the typical set. Three mitigating
facts: the error in the *difference* between nearby states (what
accept/reject uses) is smaller than the pointwise error because the error
field varies smoothly with parameters; decisions with |Δ| ≫ 1 are
insensitive to it entirely; and re-anchoring bounds any cumulative
effect. The harness makes all of this measurable rather than assumed —
if it reports WARNING, fix the model's conditioning before sampling.
