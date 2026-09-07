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
4. **Reduce with structure.** {class}`anvil.precision.ChunkedGaussianLogLike`
   sums per-datum terms chunk-wise (tree error O(ε√n) instead of O(εn));
   the `fp64_anchor` policy additionally performs the cross-chunk sum in
   float64 on the CPU stream *inside the same lazy graph*, for a measured
   0–20% cost.
5. **Re-anchor.** `run(..., reanchor_every=100)` recomputes the cached
   log-probabilities of the current states through the float64 CPU path
   periodically, so rounding drift can never accumulate along the chain.

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

## What errors remain, and why they are acceptable

For a well-conditioned 10⁵-point chi-squared, the residual float32 error
is ~0.002–0.005 log-likelihood units at the typical set. Three mitigating
facts: the error in the *difference* between nearby states (what
accept/reject uses) is smaller than the pointwise error because the error
field varies smoothly with parameters; decisions with |Δ| ≫ 1 are
insensitive to it entirely; and re-anchoring bounds any cumulative
effect. The harness makes all of this measurable rather than assumed —
if it reports WARNING, fix the model's conditioning before sampling.
