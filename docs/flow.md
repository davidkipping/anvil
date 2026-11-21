# Flow mode: sampling a curved posterior in straightened coordinates

Some posteriors are curved: a banana-shaped ridge, a funnel, a degeneracy
that bends. A dense mass matrix or an affine-invariant move removes linear
correlation and nothing else, so on a curved posterior both ChEES and the
ensemble crawl ({func}`~anvil.whitened_shape` tells the two cases apart, see
{doc}`samplers`). Flow mode trains a normalizing flow that maps the
posterior close to a standard normal, and samples there. It is off unless
you turn it on.

```python
flow = anvil.flow.train(target)                # SMC from the prior, then refinement
print(flow.report)                             # is a flow worth it here?
ft = flow.wrap(target)                         # the same posterior, flow coordinates
kernel = anvil.EnsembleKernel(ft, moves=[(anvil.StretchMove(), 0.5), (anvil.DEMove(), 0.5)])
res = anvil.run(kernel, ft, flow.start(512), n_warmup=500, n_samples=2000, thin=5)
res.get_chain()                                # in the target's coordinates, as usual
anvil.flow_evidence(target, flow)              # optional ln Z
```

Or, in the emcee-style interface:

```python
sampler = anvil.EnsembleSampler(512, ndim, target=target, flow=True)
sampler.run_mcmc(None, 2000, warmup=500)       # trains the flow first, prints its report
sampler.get_chain()
```

## Why it is exact

The flow is a bijection `T` from a latent `z` to anvil's sampling
coordinates `u`. For any fixed bijection,

    log p_z(z) = log p_u(T(z)) + log |det dT/dz|

is the same posterior written in other coordinates. Samples of it, mapped
through `T`, are samples of the original, however good or bad `T` is. The
priors are untouched: they live inside `log p_u`. All the flow decides is
how hard the sampling is. It must be frozen before the draws that are kept,
which is why training and sampling are separate steps.

{class}`~anvil.flow.FlowTarget` is that latent target. {func}`~anvil.run`
records its draws through `T`, so `get_chain()` and `get_log_prob()` read
as if the original target had been sampled. The chain state and any resume
state stay in the flow's coordinates and carry the flow's fingerprint. A
resume with a different flow, or with the unwrapped target, raises.

## What `train` does

1. **Draws that start from the prior.** {func}`~anvil.smc` on the target,
   which therefore needs a {class}`~anvil.TransformedLogDensity` with proper
   priors. This step is deliberate. A flow trained on draws from one basin
   (an optimiser's, say) fits that basin. Sampling then speeds up there and
   reports healthy diagnostics while missing the rest of the posterior: on
   an oblate transit fit, such a run gave R-hat 1.05 and a median impact
   parameter of 0.05-0.14, against 0.26-0.30 from runs that started from the
   prior. You may pass your own `draws=` instead, but they carry the same
   risk.
2. **Refinement rounds** (default 4). In each round a flow is fitted to the
   latest draws. The ensemble then runs 200 + 1,000 iterations in its
   coordinates, and the next fit uses the last two rounds' draws. This moves
   the training set from SMC's particles towards the posterior itself.
3. **A final fit, then frozen.** The flow is whitening followed by 8 affine
   coupling layers of width 64, about 40,000 parameters at 10 dimensions.
   It is trained by maximum likelihood with a 10% hold-out and early
   stopping, in 10-35 s on the GPU. A fit can never come back worse than
   whitening alone: the whitening is its first checkpoint, and a fit that
   diverges is retried at a lower learning rate.
4. **The report.** Read it before sampling.

## Reading the report

The oblate fit at 2,000 points (`benchmarks/flow_elome.py obl4`):

```text
anvil flow: fit gain 12.75 nats over whitening, importance efficiency 0.491, 184320 training draws
  smc: ln Z 16933.883, 7 of 24 stages stalled
  time: smc 491 s, rounds 94 s, final fit 24 s, report 0 s
  WARNING: smc stalled in 7 stages; its particles may under-cover the posterior, which the refinement rounds then have to correct
```

| field | meaning | measured |
|---|---|---|
| fit gain | hold-out log-likelihood of the flow minus that of a linear map | 0.02-0.11 nats on a near-Gaussian transit posterior; 5-6 on a curved limb-darkening prior; 12-14 on oblate fits |
| importance efficiency | effective sample size of the weights target/flow, as a fraction of the draws | 0.8-0.99 when the flow was excellent; 0.03-0.32 on oblate fits that still sampled well |

**"Small fit gain"** appears when the fit gain is under 1 nat. The posterior
is then close to Gaussian after a linear map, and with a cheap likelihood a
plain sampler (dense ChEES, or the ensemble) is faster. On a spherical
transit fit with near-Gaussian posteriors, plain samplers ran 2-4x faster
than the ensemble in flow coordinates, which pays about 1.5 ms per iteration
for evaluating the flow. With an expensive likelihood a small gain can
still pay: on two-mode Gaussian mixtures (2 and 6 dimensions, modes 6-10 sd
apart), fit gains of 0.5-0.9 nats gave the ensemble in flow coordinates
1.5-3.4x the effective samples per iteration of the plain ensemble, with
2-5x the mode switches. Those toys cost almost nothing to evaluate, so
there the flow was 3-5x slower per second. Compare the two on a short run.

The importance efficiency was the one number that tracked the flow's
quality through every test. It cannot see posterior mass the training draws
never reached, which is why training starts from the prior. Nothing in the
report predicts the slow tails described under Limits; the run's R-hat is
the check for those.

From that flow, the ensemble in flow coordinates reached R-hat 1.011 at
348 ESS/s (500 + 10,000 iterations), while the plain ensemble, run as long
from the flow's own training draws, sat at R-hat 5.4.

## Measured

Transit fits through Elome, 512 walkers, 3 seeds per row. The plain samplers
started from the same SMC particles. R-hat and ESS per second are medians
over seeds; "seeds agree" is the worst 5-95% quantile of any parameter
against the pooled flow runs, in posterior sd.

| fit | sampler | R-hat | ESS/s | seeds agree to |
|---|---|---|---|---|
| oblate, hybrid4, 2,000 points | ensemble, flow coordinates | 1.017 | 190 | 0.02 sd |
| | plain ensemble | 4.1 | (not converged) | 0.70 sd |
| | ChEES, dense or whitened | 2.8-3.1 | (not converged) | 0.26-0.53 sd |
| oblate, hybrid4, 10,000 points | ensemble, flow coordinates | 1.026 | 43 | 0.07 sd |
| | plain ensemble | 4.8 | (not converged) | 1.3 sd |
| spherical, hybrid4, 10,000 points | ensemble, flow coordinates | 1.004 | 5,108 | 0.02 sd |
| | ChEES, flow coordinates | 1.002 | 1,834 | 0.03 sd |
| | plain ensemble / dense ChEES | 1.47 / 1.16 | (55 / 19) | 0.30 / 0.23 sd |
| spherical, hybrid2, 10,000 points | ensemble, flow coordinates | 1.003 | 7,833 | 0.00 sd |
| | plain ensemble / dense ChEES | 1.003 / 1.002 | 26,666 / 15,879 | 0.01 sd |

ChEES in "whitened" coordinates uses the flow's linear part alone, trained
on the same draws. It did no better than plain dense ChEES on the oblate
fit, so the gain comes from the flow's nonlinear layers.

## How much faster

Sampling speed once the flow is trained: effective samples per second
(worst parameter), median of 3 seeds, 512 walkers, M2 Max, transit fits
through Elome. The ratio is against the fastest plain sampler (the
ensemble, or ChEES with a dense mass matrix), started from the same SMC
particles and run as long.

| fit | flow ensemble | fastest plain sampler | speedup |
|---|---|---|---|
| oblate, hybrid4, 2,000 points | 190 ESS/s, R-hat 1.017 | none converged (R-hat 2.8-4.1) | no plain answer to compare |
| oblate, hybrid4, 10,000 points | 43 ESS/s, R-hat 1.026 | none converged (R-hat 4.8-35) | no plain answer to compare |
| spherical, hybrid4, 2,000 points | 6,669 ESS/s, R-hat 1.004 | ensemble 171 (R-hat 1.48) | 39x, a lower bound |
| spherical, hybrid4, 10,000 points | 5,108 ESS/s, R-hat 1.004 | ensemble 55 (R-hat 1.47) | 94x, a lower bound |
| spherical, hybrid2, 2,000 points | 9,809 ESS/s, R-hat 1.003 | dense ChEES 42,844 (R-hat 1.002) | 0.23x: slower |
| spherical, hybrid2, 10,000 points | 7,833 ESS/s, R-hat 1.003 | ensemble 26,666 (R-hat 1.003) | 0.29x: slower |

The curved-fit ratios are lower bounds, because effective sample sizes from
runs that have not converged are overestimates.

**Including the training.** Training is a fixed cost paid before any
sample:

- **Spherical hybrid4, 10,000 points.** Training took 137 s, 61 s of it SMC.
  After that, 10,000 effective samples took 2 s. The plain ensemble, at its
  overstated rate, would need 180 s for the same, and it had not converged.
- **Oblate, 10,000 points.** Training took about 2,300 s, of which SMC was
  2,080 s; 10,000 effective samples then took 230 s. No plain sampler
  converged in any run tried, including ChEES runs of 9-51 minutes.
- **Spherical hybrid2.** The flow is pure overhead, and the report warns of
  a small fit gain before you sample.

**Per-iteration overhead.** Evaluating the flow adds about 1.5 ms to each
ensemble iteration of 512 walkers. That is what makes the near-Gaussian
fit slower (22 s against 6 s for the same run at 10,000 points). It is
negligible next to an expensive likelihood: the oblate fit costs 20-66 ms
per batched evaluation.

## Memory

Peak MLX memory (unified, GPU-side) per phase, measured on the spherical
hybrid4 fit at 10,000 points with 512 walkers
(`research/flow_reparam/exp_memory.py` in the repository):

| phase | peak |
|---|---|
| plain ensemble | 118 MB |
| ensemble in flow coordinates | 119 MB |
| plain ChEES, dense mass matrix | 775 MB |
| ChEES in flow coordinates | 335 MB |
| `smc` alone | 316 MB |
| `train()` overall (SMC, rounds, fit, report) | 316 MB |
| fitting a flow on 102,400 draws | 58 MB |
| `flow_evidence`, 262,144 draws | 79 MB |

Flow mode adds no memory pressure of its own:

- The flow is 40,000 parameters (156 kB).
- Sampling in flow coordinates costs what sampling did before.
- `train()` peaks at SMC's level.
- The training draws held on the host are a few MB.

The two importance-sampling steps (the report's efficiency and
`flow_evidence`) evaluate the likelihood 1,024 draws at a time. Their peak
grows with that block size times the data size, at about 8 bytes per row
per data point on a transit likelihood: 79 MB at 10,000 points, 392 MB at
50,000. `flow_evidence(max_rows=...)` changes the block size; 8,192 rows
were 10-15% faster, but took 629 MB and 3.1 GB at those sizes.

## Limits

- **Slow tails on the hardest fits.** On the oblate fits, R-hat ended at
  1.015-1.027 rather than 1.01. Between 5 and 12 of the 512 walkers made
  long excursions through the tails. They were in real posterior mass, 0.3-5
  nats below the median log-probability and still moving, not lost, so do
  not prune them. One more fit on the production draws
  ({func}`~anvil.flow.refine`) and a run four times longer took the
  2,000-point fit to R-hat 1.006. The 10,000-point fit stayed at 1.013 with
  either remedy, and a 12 x 128 flow did not help either. Check R-hat as
  always.
- **ChEES in flow coordinates** is exact and matched the ensemble on the
  spherical fits. On the oblate fits it was 20-40x slower than the
  ensemble: the few walkers in badly mapped tails set its shared step size.
  Use the ensemble (`EnsembleSampler(flow=...)` is ensemble-only).
- **The cost is mostly SMC.** On the 10,000-point oblate fit SMC took
  2,080 s, the refinement rounds 210 s and production 285 s, so the start
  dominates. No plain sampler converged there in any time tried, but if
  you already have draws that started from the prior, `train(draws=...)`
  skips SMC.
- **`dim >= 2`**: a coupling layer splits the coordinates in two.
- Float64 certification ({func}`~anvil.certify`) applies to the original
  target, not to a {class}`~anvil.flow.FlowTarget`.

## Refining and reusing a flow

```python
flow2 = anvil.flow.refine(target, res)        # one more fit, on production draws
ft2 = flow2.wrap(target)
kernel2 = anvil.EnsembleKernel(ft2, moves=[(anvil.StretchMove(), 0.5), (anvil.DEMove(), 0.5)])
res2 = anvil.run(kernel2, ft2, flow2.start(512), n_warmup=500, n_samples=8000, thin=5)
flow2.save("flow.npz")                        # .npz, no pickle
flow3 = anvil.flow.Flow.load("flow.npz")
```

A run on a refined flow is a new Markov chain, so start it with warmup
rather than by resuming the old one; `run` refuses a resume across flows.

## Evidence

{func}`~anvil.flow_evidence` is importance sampling with the flow as the
proposal: seconds of work once a flow exists, and the most accurate
estimator on every curved fit tested. On the 10,000-point oblate fit the
three seeds gave 84871.773-84871.788 (standard error 0.007-0.012) against
an independent reference of 84871.8 +/- 0.1. Default SMC there was
0.1-0.45 nats low and Laplace 5.7 nats low. Like everything in
{mod}`anvil.evidence`, nothing computes it unless you call it. Its standard
error comes from the effective sample size and can read somewhat small: on
a test banana, one seed in three landed 2.9 standard errors from the
closed form, though only 0.003 nats from it.

## Hardware

The flow is small and runs on the GPU in milliseconds, alongside the
likelihood. The Neural Engine is not usable for it: it computes in 16-bit
floats only, has no public gradient or training path, and has no MLX
backend.
