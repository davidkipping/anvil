# Samplers: which, when, and how

## ChEES-HMC (gradient-based flagship)

{class}`anvil.ChEESHMC` implements Hoffman & Sountsov's (AISTATS 2021)
GPU-native HMC. Three quantities adapt simultaneously during warmup from
*cross-chain* statistics — a step size (dual averaging on the harmonic
mean of per-chain acceptance, target 0.651), a shared trajectory length
(stochastic ascent on the ChEES criterion), and a diagonal preconditioner
(EMA of cross-chain moments) — then all freeze for the recorded phase.

The design keeps every chain's control flow identical: the
trajectory-length jitter is a single per-iteration scalar from a Halton
sequence, so all chains run the same number of leapfrog steps each
iteration. On smooth targets this reaches ~60% ESS per draw with zero
divergences.

**Practical guidance:**

- Requires an MLX-differentiable log density (autodiff is automatic; see
  the gradient notes below).
- **Initialize tightly** — a small ball (~10⁻³ in scaled units) around a
  preliminary optimizer fit. On likelihoods with flat plateaus (e.g. a
  transit model where a chain's proposed transits miss the data
  entirely), wide initializations strand chains at zero gradient, which
  collapses the shared step size and stalls the run.
- Cap `max_leapfrog` (≈128) so a misadapted warmup cannot spend 1000
  gradient evaluations per iteration.
- Watch `results.extras["n_divergent"]`; nonzero means the geometry is
  fighting you (funnels, cusps). Neal's funnel is deliberately kept as a
  known-hard test: anvil's requirement is that diagnostics *flag* it,
  not that it be silently sampled.

## Ensemble moves (gradient-free fallback)

{class}`anvil.EnsembleKernel` runs Goodman-Weare stretch and
differential-evolution moves, vectorized with the red-black scheme: half
the ensemble updates simultaneously against the frozen other half, then
vice versa — two batched likelihood calls per iteration, detailed balance
intact. The stretch move is affine-invariant (identical behavior under
any linear reparameterization) and needs no tuning; DE adds occasional
mode-hopping proposals.

Use it when: gradients are unavailable, the model has kinks or plateaus,
or you want a robust first look from a wide initialization. Constraints:
even walker count, `n_walkers ≥ 2·dim` (enforced), and thousands of
walkers to feed the GPU.

```python
kernel = anvil.EnsembleKernel(
    target,
    moves=[(anvil.StretchMove(a=2.0), 0.7), (anvil.DEMove(), 0.3)],
)
```

A sensible expensive-posterior workflow: burn in with the stretch move
from a wide initialization, then hand the tightened ensemble to ChEES-HMC
for production draws — both kernels share the same state layout.

## Gradients and autodiff

Any likelihood written end-to-end in MLX ops is differentiable via
`mx.grad`/`mx.vjp`; anvil obtains all per-chain gradients in one
forward+backward pass. Caveats that matter in practice:

- External calls (numpy, compiled libraries) break the chain — MLX ops
  only.
- Iterative solvers need *fixed* iteration counts.
- Branch with `mx.where`, and sanitize the arguments of *both* branches:
  `mx.where` evaluates both sides, and a NaN in the inactive branch
  poisons the gradient even though the forward value is fine.
- Never let a numpy scalar into the graph (convert closure constants
  with `float(...)`): it silently forces evaluation and breaks
  `mx.compile`.

## Measured performance

Benchmark (M2 Max, 6-parameter transit over 10⁵ points, all runs
converged; `benchmarks/bench_transit.py`):

| sampler | min ESS | max R-hat | ESS/s |
|---|---:|---:|---:|
| stretch, 2048 walkers (GPU) | 48,952 | 1.033 | 612 |
| ChEES-HMC, 1024 chains (GPU) | 80,313 | 1.009 | 100–330 |
| emcee, 64 walkers (CPU, fp64 numpy) | 1,207 | 1.041 | 2–10 |

Sustained-GPU thermal throttling swings wall-clock ~2× on laptop-class
hardware; the conservative cross-run claims are ≥60× (stretch) and ≥10×
(ChEES-HMC) over emcee's best measurement. The trapezoid's kinks are
near ChEES-HMC's worst case; on smooth models the ranking flips.
