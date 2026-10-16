# Priors

Every parameter in anvil has a prior, whether or not you write one. Since
0.8.0 you can state it explicitly — and anvil can then tell you whether the
prior is **proper**, what its **normalising constant** is, and draw from it.
That is what a Bayesian evidence, a prior-predictive check, or simply an
honest methods section needs.

## Declaring a prior

Priors attach to the parameter's {class}`~anvil.ParamSpec`:

```python
from anvil import ParamSpec, Transform
from anvil.priors import Uniform, LogUniform, Gaussian

transform = Transform([
    ParamSpec("t0_off", -0.02, 0.02),                  # Uniform on the box (the default)
    ParamSpec("p_off", prior=Gaussian(0.0, 3e-4)),      # plain Gaussian (unbounded)
    ParamSpec("a", 1.5, 200.0, prior=LogUniform()),     # Jeffreys on a scale
    ParamSpec("b", 0.0, 1.2, prior=Gaussian(0.3, 0.2)), # Gaussian restricted to the box
])
```

| prior | density | where it fits |
|---|---|---|
| `Uniform()` | flat on the box | location parameters with hard limits (the default) |
| `LogUniform()` | ∝ 1/v on the box, `0 < lo < hi < ∞` | scales spanning decades (a/R★, amplitudes) |
| `Gaussian(mu, sigma)` | Normal | an independent measurement of the parameter |

## How anvil imposes it: by addition

anvil's samplers never walk the parameter directly; they walk an unbounded
`u`, mapped to the model value `v` by the spec's transform (so HMC can never
step outside a box). A prior is imposed by **adding** its log-density to the
log-likelihood, on top of that unchanged map:

```
log_prob(u) = log L(v) + log p(v) + log |dv/du|
```

The prior's log-density is split the way the rest of anvil splits float32 and
float64 work: the part that depends on `v` is in the GPU graph (nothing at all
for a flat prior), and the **normalising constant is a float64 host scalar**
that never enters it — exactly like `ChunkedGaussianLogLike.log_norm_const`.

Nested samplers (MultiNest, dynesty) instead map the unit hypercube through
each prior's inverse CDF. That samples the same posterior, but HMC pays for the
shape of the space it walks, and measured with ChEES the inverse-CDF map is
never the better buy: a Gaussian prior that dominates the data costs 5 leapfrog
steps per draw through it against 1 by addition, because the u-space posterior
becomes logistic-tailed instead of Gaussian; a LogUniform costs 0.55–0.80×.
The hypercube *interface* is still provided — see [Prior draws](#prior-draws).

## Defaults, and what "improper" means

A parameter that declares nothing gets:

- **`Uniform` on a finite box.** Its draws are bit-identical to anvil before
  priors existed: a flat prior adds nothing to the graph, only the constant
  `−log(hi − lo)` on the host.
- **`ImproperFlat` on a half-bounded or unbounded spec.** A flat density over
  an infinite range has no normalisation. Sampling is unaffected, but an
  evidence is undefined, and `Transform.is_proper` is `False` so code that
  needs a proper prior can refuse.

## The box is the support

A prior with explicit bounds must match its `ParamSpec` box, and a prior
without bounds adopts it. Narrower than the box is refused as well as wider:
it would put a wall of zero density inside the sampled space.

So a **Gaussian on a bounded or half-bounded parameter is a truncated
Gaussian** — not by choice, but because the box already says the parameter
cannot leave it. Its in-graph term is the same `−½((v − μ)/σ)²`, so the
sampled posterior is identical; only the normalising constant gains `−log Z`,
with Z the Gaussian mass inside the box (`prior.mass_inside`). If you want a
plain Gaussian, declare the parameter unbounded.

## Units: model units, on the offset

Priors are stated in **model units** — the units of `lo` and `hi`, which are
the offsets the model samples ("sample offsets, never absolutes",
{doc}`precision`). A prior on a period of 3.456 ± 0.0001 d sits on `p_off`:

```python
spec = ParamSpec("p_off", report_offset=period_ref)
ParamSpec("p_off", report_offset=period_ref,
          prior=Gaussian.from_physical(3.456, 1e-4, spec))   # converts in float64
```

Writing the Gaussian on the absolute period instead would cost float32 about
three digits of resolution on `(v − 3.456)`. `LogUniform` on an offset spec is
refused: Jeffreys on an offset is a mistake.

## Where the constants live

| quantity | float64 host attribute |
|---|---|
| one prior's normaliser | `prior.log_norm` |
| all priors | `transform.log_prior_norm` |
| the likelihood's own constants | `TransformedLogDensity(..., model_log_norm=...)` |
| both | `target.log_norm_const` |

The **absolute log posterior density** of a stored draw is then
`results.get_log_prob() + target.log_norm_const` — the likelihood, the
normalised prior and the Jacobian, with every constant restored.

(prior-draws)=
## Prior draws

```python
v = transform.prior_transform(cube)          # (n, dim) hypercube -> model space
v = transform.sample_prior_model(n, rng)     # prior draws in model space
u0 = transform.sample_prior(n, rng)          # the same, mapped to u, ready for run(u0=)
```

`prior_transform` is the MultiNest/dynesty interface: each prior's inverse CDF
applied column by column, so a nested sampler can be handed exactly anvil's
prior. anvil never uses prior draws as a default starting point — for a
data-dominated posterior they are a poor start.

## What this does not cover

- **Correlated (joint) priors.** Write them into `model_log_prob` as before;
  they sample correctly but are invisible to `log_norm_const`.
- **Reparameterisations** such as uniform-in-q limb darkening: those are a
  `Uniform` prior on the sampled q, which is what you get by default.
- **`FunctionLogDensity` and the emcee facade**, which have no model space or
  bounds: there `log_prob` *is* the posterior, as in emcee.

## Writing your own

Subclass {class}`anvil.priors.Prior` with `lo`, `hi`, `log_norm`,
`log_pdf(v)` (MLX, elementwise; build constants from Python floats so it works
in float32 on the GPU and float64 on the CPU), `log_pdf_np(v)` (float64, `-inf`
outside the support), `ppf_np(x)` (inverse CDF), `mean`, `sd`, and
`bind(lo, hi, name)` to check it against a box. Set `has_in_graph_term = False`
only if `log_pdf` is identically zero.
