# Evidence: which model does the data prefer?

The Bayesian evidence, Z = ∫ L(θ) p(θ) dθ, is what compares two models
fitted to the same data. Their ratio is the Bayes factor. anvil can
estimate it, as an **optional, separate calculation**. {func}`anvil.run`
and the samplers never compute it, and nothing about sampling changes
when you use it.

```python
res = anvil.smc(target)            # ln Z, and posterior draws
print(res)                         # anvil.smc: ln Z = ... (one population)
lnB = anvil.smc(round_planet).log_z - anvil.smc(oblate_planet).log_z
```

## Three tools

| call | what it does | cost | trust it when |
|---|---|---|---|
| {func}`~anvil.smc` | tempered sequential Monte Carlo from the prior | a fraction of a sampling run | it reports no stalled stages |
| {func}`~anvil.bridge_evidence` | bridge sampling on posterior draws you already have | seconds | the posterior is close to unimodal, and the draws weight its modes correctly |
| {func}`~anvil.laplace_evidence` | Gaussian at the mode | an optimisation | as a floor only |

### Tempered SMC

SMC starts from draws of the prior and carries them to the posterior through
a sequence of tempered densities, p_β ∝ L^β p, as β goes from 0 to 1. Each
stage does three things:

1. **Reweight.** β rises by as much as keeps the particles' importance
   weights L^Δβ from collapsing.
2. **Resample.** Particles are resampled by those weights.
3. **Move.** Each particle takes stretch moves and differential-evolution
   moves at the new β until every coordinate has decorrelated from where
   the stage began (at most 200 sweeps).

The evidence is the product of each stage's mean weight. At β = 1 the
particles are posterior draws. The default is one population of 8,192
particles.

**How it differs from nested sampling.** Both go from prior to posterior,
but SMC never has to draw from a likelihood-constrained region. Its moves
only need a smooth density, so all of them run in one batched GPU call.

**How it differs from thermodynamic integration.** TI integrates E_β[ln L]
over β and needs a fine ladder of temperatures. SMC multiplies the ratios
between neighbouring temperatures, so it can take a few large, adaptive
steps.

**Mode masses come from the weights.** They are settled while the modes
still overlap at small β, not by chains crossing between modes. This is
why SMC is the tool for posteriors MCMC cannot certify.

### When SMC is biased: particles that stop moving

On a narrow, curved posterior, late in the run, few moves are accepted.
The particles stay close to where resampling left them, the part of the
posterior they have not reached is under-counted, and ln Z comes out low.
Nothing in the estimate itself shows it, and more particles do not cure
it. On a 2,000-point version of the oblate fit below, with no oblateness
in the data, against a reference of 16934.13:

| particles | ln Z − reference | stages flagged |
|---|---|---|
| 2,048 | −0.30 | 9 of 25 |
| 4,096 | −0.17 | — |
| 8,192 | −0.21 | — |

In the flagged stages fewer than 5% of moves were accepted, and the
particles were still correlated above 0.95 with where the stage began
after 200 sweeps. Gradient moves did not help: ChEES-HMC from the same
particles at β = 1 had not finished warming up after 17 minutes, longer
than a whole SMC run.

`smc` counts such stages, in `SMCResult.stalled`, and warns when there are
any. A stage is flagged when it ends with some coordinate still correlated
above 0.9 with the stage's start and fewer than 5% of its moves accepted.
The acceptance test is what spares multimodal
posteriors: a coordinate that tells modes apart stays correlated however
well particles move within their modes. The flag detects the symptom, not
the bias itself: on a thin parabolic ridge in the test suite (6 dimensions,
2,048 particles) it fired in 6 or 7 of 13 stages on every one of 5 seeds,
yet ln Z came out unbiased there (−0.01 ± 0.04 nats over the 5 seeds); a
ridge four times wider was never flagged (−0.02 ± 0.03). On the oblate fit
above, the same symptom came with a bias of 0.17–0.30. Both thresholds cut
a continuous failure, so a run just on the other side of them is not much
safer; `SMCResult.unsettled` counts the stages that used all their moves
still correlated, whatever their acceptance.

What to do with a warning:

- **Treat ln Z as good to a few tenths of a nat.** A Bayes factor of
  several nats is still decided. One within about half a nat of zero is
  not.
- **Straighten the posterior if you can.** Stretch and DE moves are
  affine-invariant, so correlations cost nothing; curvature is what stops
  them. A parameterisation in which the ridge is closer to straight helps
  every sampler.

A run without the warning is not proof of accuracy, only the absence of
the one failure anvil can detect.

**Groups and an error bar.** `n_groups=G` splits the particles into G
independent groups, run in the same batched calls, and gives a standard
error from their spread (err/se follows a t distribution with G − 1
degrees of freedom). Each group must be large enough by itself: smaller
groups are each more biased, and their spread cannot show a bias they
share. On the full oblate fit below, four groups of 4,096 came out 0.3 to
0.5 nats low, with an error bar that did not cover it. `smc` warns when its groups disagree
by more than 0.5 nats, but groups that agree can all be too small.

### Bridge sampling

{func}`~anvil.bridge_evidence` takes posterior draws and returns ln Z by
Meng and Wong's optimal bridge between them and a Student-t fitted to half
the draws, then again with the halves swapped, and averages the two. The
average was 15–25% more accurate than one fold on a Gaussian posterior,
and its error bar adds the spread between the folds to the formula's
(calibrated: rms of error / error bar 0.8–1.0 over 80 seeds, 500 to
20,000 draws). The draws can be:

- a {class}`~anvil.Results`, whose autocorrelation the error bar accounts
  for;
- an {class}`~anvil.SMCResult`;
- an array of draws.

It is exact on near-Gaussian posteriors and takes seconds. Two limits:

- **It knows only the mode weights its draws have.** Given ChEES chains
  that did not mix between modes, it returns a wrong ln Z with a small
  error bar.
- **It warns when the Student-t proposal overlaps the posterior poorly.**
  On the oblate transit fit it did.

### Laplace

{func}`~anvil.laplace_evidence` places a Gaussian at the mode of the
posterior in u, using {func}`anvil.optimize` with `objective="density"`.
On a spherical transit fit it was 2.4 nats low, so it is a floor, not an
answer.

## What ln Z includes

- **The priors declared on each {class}`~anvil.ParamSpec`, normalised.** A
  proper prior is required; an unbounded parameter needs an explicit prior.
- **The likelihood's constants, only if declared.** A
  {class}`~anvil.precision.ChunkedGaussianLogLike` keeps them out of its
  graph. Pass them as
  `TransformedLogDensity(..., model_log_norm=lik.log_offset_const + lik.log_norm_const)`.
  Without that, ln Z is offset by a constant. The constant cancels between
  models fitted to the same data with the same likelihood, which is the
  usual comparison.
- **Anything written into `model_log_prob`.** A prior or penalty written
  there is part of the likelihood as far as anvil can tell, and is not
  normalised. Elome's eccentric targets fold a constraint penalty in this
  way.

The float32 error in log L is reported, not corrected: `res.precision`, at
a few posterior draws, when the target has a float64 path. On the transit
fits below it was under 0.01 nats.

## Measured: a round planet against an oblate one

`benchmarks/evidence_elome.py` fits two models to one simulated Elome light
curve: 10,000 points, hybrid4 limb darkening. The spherical model has 8
parameters. The oblate model adds the oblateness as
(√f cos 2θ, √f sin 2θ) on a disc, which is uniform in f and θ, and is the
spherical model at f = 0.

With no oblateness in the data:

| model | reference ln Z | how the reference was found | default `smc` |
|---|---|---|---|
| spherical | 84874.36 | five SMC runs at 2,048 and 8,192 particles, within about 0.1 of each other | 84874.34, 77 s, none flagged |
| oblate | 84871.8 ± 0.1 | an integral over the oblateness disc of 8-parameter evidences: best fit and Gaussian volume at 20,160 points, corrected by SMC at 40 of them | not measured at full size; see the 2,000-point table above |

The reference Bayes factor is ln B(spherical / oblate) = +2.55 ± 0.1:
with no oblateness in the data, the simpler model is preferred, by about
13 to 1. With f = 0.1 injected, the oblate model wins by about 195 nats,
and an error of a few tenths changes nothing.

Cost on the spherical fit: 77 s for `smc`, against 609 s for a 500 + 500
ChEES run of 512 chains on the same target.
