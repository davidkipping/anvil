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

### Why ChEES-HMC can lose to a gradient-free move

On the bundled transit problem the stretch move wins on wall-clock
despite ChEES-HMC having ~80× better ESS *per draw*. The reason is worth
understanding, because it tells you which sampler to reach for.

ChEES-HMC adapts a **diagonal** preconditioner. That removes each
parameter's scale but cannot remove *correlations between* parameters —
and the transit posterior is full of them (`b`–`a` −0.96, `q1`–`q2`
−0.95, `r`–`b` +0.92). Measured, the diagonal preconditioner does its
job perfectly (it matches every posterior sd to three digits) and still
leaves a condition number of ~500–1500 in the correlation matrix. HMC
must then buy roughly √cond leapfrog steps per draw to cross the
posterior — measured mean L ≈ 39, against √1546 = 39.3. The stretch move
is *affine-invariant*, so it pays nothing for those correlations; that,
and nothing else, is its advantage here.

The natural fix is a dense (full-covariance) mass matrix, and on
correlated posteriors it works exactly as theory predicts — measured on
a correlated Gaussian, L collapses from 36 to 1 and ESS per gradient
improves **40×**. Whether it rescues a *transit* posterior depends on the system. For the
hot Jupiter in `examples/metalplanet_hotjupiter.py` it does: that
posterior is correlated but essentially uncurved (max whitened skew
0.07), and dense takes it from 38 leapfrog steps per draw to 3, a
measured 6.8× in ESS/s. For a longer-baseline warm Jupiter it does not:
there the scaled semi-major axis keeps skew 2.2 and excess kurtosis 13.6
after whitening, so the posterior is **curved** rather than merely
correlated, no global mass matrix can linearize it, L stays at 30 and the
gain falls to 1.85×.

**The practical rule.** Use {func}`anvil.whitened_shape` on a pilot
sample. It whitens by the sample covariance — removing exactly the linear
part a dense mass matrix would remove — and reports what is left:

```python
skew, exkurt = anvil.whitened_shape(pilot.get_chain(flat=True))
```

* both near zero → correlated but Gaussian-ish; turn on `dense=True`;
* large → the posterior is curved, no global preconditioner will save
  you, and an affine-invariant ensemble move is likely the better buy
  (better still, reparameterize the curvature away). Measured: ~0.01 on a
  correlated Gaussian, 1.9 on the transit posterior.

### Using a dense mass matrix

```python
sampler = anvil.HMCSampler(n_chains, dim, log_prob, dense=True)
# or, at the kernel level:
kernel = anvil.ChEESHMC(target, dense=True)
```

Off by default; enabling it changes nothing else about the sampler. The
covariance is estimated across chains during warmup and frozen with
everything else. It is held factored as Σ = S·R·S so that float32 only
ever sees the correlation matrix R, never the covariance itself — on the
transit posterior those have condition numbers of ~10³ and ~10⁷
respectively, and the second would not survive single precision.

Measured gains, ESS per gradient evaluation:

| posterior | leapfrog steps: diagonal → dense | gain |
|---|---:|---:|
| correlated Gaussian, ρ = 0.9 | 11 → 3 | 5× |
| correlated Gaussian, ρ = 0.99 | 36 → 1 | 40× |
| hot-Jupiter transit, uncurved (max \|skew\| 0.07) | 38 → 3 | **6.8×** |
| warm-Jupiter transit, curved (\|skew\| 2.2) | 30 → 30 | 1.85× |

The last two rows are both transit fits, and the difference between them
is entirely curvature — which is why it is worth measuring rather than
guessing.

Two guards fire automatically. With fewer than `4 × dim` chains the
cross-chain covariance is mostly noise, so the kernel warns and falls back
to the diagonal preconditioner. Above `dim = 512` it warns that the
O(dim²) work per leapfrog step may outweigh the preconditioning gain. A
singular correlation matrix is absorbed by an escalating ridge rather than
raising.

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
near ChEES-HMC's worst case; on smooth models the ranking flips — and
see the correlation/curvature discussion above for *why* the
gradient-free move competes at all here.

## Diagnostics cost

{func}`anvil.diagnose` returns R-hat and bulk ESS from one shared pass;
prefer it to calling {func}`~anvil.diagnostics.split_rhat` and
{func}`~anvil.diagnostics.ess_bulk` separately, which repeats the rank
normalization that dominates the work. Measured at 400 draws × 2048
chains × 8 parameters: 1685 ms → 109 ms. Worth knowing because on cheap
targets the diagnostics used to cost several times the sampling run they
described.
