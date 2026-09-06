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
  step size, trajectory length, and preconditioning
  (Hoffman & Sountsov 2021). No per-chain control flow: thousands of
  chains advance in lockstep. Requires an MLX-differentiable log density.
- **Ensemble moves** — vectorized Goodman-Weare stretch and
  differential-evolution moves with detailed-balance-correct red-black
  updates. Gradient-free, affine-invariant, robust; near-drop-in for
  emcee workflows.

## Where to start

```{toctree}
:maxdepth: 2

quickstart
precision
samplers
api
```
