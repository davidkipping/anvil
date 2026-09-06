# Quickstart

## Install

anvil requires macOS on Apple Silicon, Python ≥ 3.10, and MLX ≥ 0.30.

```bash
pip install -e ".[test]"        # from a clone, until a PyPI release
python -m pytest -m "not slow"  # sanity check
```

## The likelihood contract

Your log-probability is **batched MLX**: it receives every chain's
parameters at once as an `mx.array` of shape `(n_chains, dim)` and
returns `(n_chains,)`, built from `mlx.core` operations. That single
convention is what puts the whole ensemble's likelihood evaluation on the
GPU as one dispatch.

```python
import mlx.core as mx
import numpy as np
import anvil

mu = mx.array([1.0, -0.5, 2.0])

def log_prob(theta):                  # (n_chains, 3) -> (n_chains,)
    d = theta - mu
    return -0.5 * mx.sum(d * d, axis=-1)
```

A per-walker numpy function (the emcee habit) raises a `TypeError` with
porting hints at construction — there is deliberately no silent slow path.

## Sampling

```python
sampler = anvil.EnsembleSampler(2048, 3, log_prob)   # stretch move
p0 = np.random.default_rng(0).normal(size=(2048, 3))
sampler.run_mcmc(p0, 500, warmup=500)
chain = sampler.get_chain()                          # (500, 2048, 3)
print(anvil.diagnostics.summary(chain))
```

Swap `EnsembleSampler` for `HMCSampler` to use ChEES-HMC when your
log-probability is smooth and MLX-differentiable. Unlike emcee, warmup
(adaptation) and recording are strictly separated: `warmup=` iterations
adapt and are discarded, then the recorded phase runs with the sampler
frozen, so the stored chain is a valid MCMC sample.

## Thinking in many short chains

With thousands of chains you need far fewer post-warmup draws per chain
than a 32-walker workflow — a few hundred is typically plenty. Judge
convergence with `anvil.diagnostics.split_rhat`, `ess_bulk`, and — in the
very-short-chain limit — `nested_rhat` (group chains into superchains by
shared initialization).

## Bounded parameters and physical units

Use {class}`anvil.ParamSpec` and {class}`anvil.Transform` to sample
bounded or half-bounded parameters in an unbounded space (Jacobians
included automatically via {class}`anvil.TransformedLogDensity`), and to
reinstate float64 physical units (absolute epochs, baselines) on output.
See {doc}`precision` — the transform layer is half of the float32 story.

## A complete worked problem

`examples/transit_lightcurve.py` fits a 6-parameter transit model to a
synthetic 100,000-point light curve with both sampler families, checks
the float32 likelihood against float64, and reports posteriors in
absolute BJD-scale units. It is the template to copy for your own
data-heavy fits.
