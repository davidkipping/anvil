# anvil

**MCMC sampling forged for Apple Silicon.** anvil runs thousands of
parallel Markov chains on the Apple GPU via
[MLX](https://github.com/ml-explore/mlx), in carefully conditioned
float32 — while the CPU, which has float64, owns offsets, large
constants, preprocessing, re-anchoring, and diagnostics. Unified memory
makes the split free.

The regime anvil is built for: **moderate dimension (10–50 parameters) ×
data-heavy likelihoods (10⁴–10⁶ points) × thousands of chains** — a
transit fit, an RV fit, any chi-squared over a long time series. In that
regime the measured advantage over a tuned CPU emcee workflow is one to
two orders of magnitude in effective samples per second on a single
M2 Max.

## Two sampler families, one engine

- **ChEES-HMC** — Hamiltonian Monte Carlo with cross-chain adaptation of
  step size, trajectory length, and preconditioning — diagonal by
  default, or the full covariance with `dense=True`
  (Hoffman & Sountsov 2021). No per-chain control flow: thousands of
  chains advance in lockstep. Requires an MLX-differentiable log density.
- **Ensemble moves** — vectorized Goodman-Weare stretch and
  differential-evolution moves with detailed-balance-correct red-black
  updates. Gradient-free, affine-invariant, robust; near-drop-in for
  emcee workflows.

## Answering the questions a run raises

Sampling is the easy half. anvil ships the measurements that tell you
whether to believe the answer, each one built around a trap that caught
this project out at least once:

| call | question |
|---|---|
| {func}`~anvil.diagnostics.diagnose` | did it converge, and how many effective samples? |
| {func}`~anvil.diagnostics.warmup_report` | was warmup long enough — or far longer than needed? |
| {func}`~anvil.diagnostics.whitened_shape` | correlated, or curved? (predicts whether `dense=True` pays) |
| {func}`~anvil.precision.validate_precision` | is the float32 likelihood accurate enough to trust? |
| {func}`~anvil.precision.certify` | how much did float32 bias the posterior, and what is it corrected? |

## Where to start

```{toctree}
:maxdepth: 2

quickstart
precision
samplers
api
```
