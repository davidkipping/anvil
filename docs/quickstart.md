# Quickstart

## Install

anvil requires macOS on Apple Silicon, Python ≥ 3.10, and MLX ≥ 0.30.

```bash
pip install -e ".[test]"        # from a clone; the package name is anvil-mcmc
python -m pytest -m "not slow"  # sanity check
pip install -e ".[docs]"        # to build these docs: sphinx -W -b html docs out
```

MLX sets the minimum macOS version; see its installation notes.

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
frozen, so the stored chain is a valid MCMC sample. Always pass `warmup=`
(or `tune=True`) to `HMCSampler`: without it ChEES runs unadapted, and it
warns. `anvil.run` itself defaults to 500 warmup iterations and 1000
recorded ones.

## Thinking in many short chains

With thousands of chains you need far fewer post-warmup draws per chain
than a 32-walker workflow — a few hundred is typically plenty. Judge
convergence with `anvil.diagnose(chain, log_prob=res.get_log_prob())` —
R-hat, bulk ESS and lost chains in one pass — and, in the
very-short-chain limit, `anvil.diagnostics.nested_rhat` (group chains into
superchains by shared initialization). A run too large for memory can
stream its draws to disk:
`anvil.run(..., storage=anvil.storage.NpyStreamBackend("run1"))` writes
`run1_chain.npy` and `run1_log_prob.npy`.

## Bounded parameters and physical units

Use {class}`anvil.ParamSpec` and {class}`anvil.Transform` to sample
bounded or half-bounded parameters in an unbounded space (Jacobians
included automatically via {class}`anvil.TransformedLogDensity`), and to
reinstate float64 physical units (absolute epochs, baselines) on output.
See {doc}`precision` — the transform layer is half of the float32 story.

Each parameter can also carry an explicit prior — `Uniform` (the default on a
box), `LogUniform` or `Gaussian` — via `ParamSpec(prior=...)`; see {doc}`priors`.

## A data-heavy likelihood

For a model evaluated over many data points, build on
{class}`~anvil.precision.ChunkedGaussianLogLike`: it sums the Gaussian
terms in chunks with float32 under control, and its `.hi` method is the
same model in float64 for checking it. The model function is batched over
chains and uses only MLX ops, so one function serves both precisions:

```python
from anvil.precision import ChunkedGaussianLogLike

x = np.linspace(-1.0, 1.0, 20_000)                    # float64, centred
y = 0.3 + 1.2 * x + 0.01 * np.random.default_rng(0).standard_normal(x.size)

def model(v, x):                                      # (n, 2), (m,) -> (n, m)
    return v[:, :1] + v[:, 1:2] * x[None, :]

lik = ChunkedGaussianLogLike(model, x, y, np.full(x.size, 0.01))
target = anvil.TransformedLogDensity(
    lik, anvil.Transform([anvil.ParamSpec("a", -1.0, 1.0),
                          anvil.ParamSpec("b", 0.0, 3.0)]),
    model_log_prob_hi=lik.hi,
    model_log_norm=lik.log_offset_const + lik.log_norm_const)
print(anvil.validate_precision(target, mx.zeros((64, 2))))
res = anvil.run(anvil.ChEESHMC(target), target, mx.zeros((512, 2)))
```

`model_log_norm` declares the likelihood's constants, which only the
evidence needs; sampling and optimisation do not.

## A complete worked problem

`examples/transit_lightcurve.py` fits a 6-parameter transit model to a
synthetic 100,000-point light curve with both sampler families, checks
the float32 likelihood against float64, and reports posteriors in
absolute BJD-scale units. It is the template to copy for your own
data-heavy fits. (`examples/elome_hotjupiter.py` does the same with a
physical transit model and needs the separate `elome` package.)
