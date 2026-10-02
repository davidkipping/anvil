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

## How much warmup do you need?

Warmup is pure overhead — it produces no samples — but cutting it too far
biases everything downstream, so guessing is the wrong move in both
directions. `run()` records a cheap trace of the warmup phase (cross-chain
spread, acceptance, step size; one host sync per probe, 40 by default) and
{func}`anvil.warmup_report` reads it back:

```python
res = anvil.run(kernel, target, u0, n_warmup=1000, n_samples=500)
print(anvil.warmup_report(res))
```

```
WarmupReport over 1000 warmup iterations
  cross-chain spread settled : iteration 310
  final mean acceptance      : 0.374
  spread-estimate noise floor: 6.3% (128 chains)
  LONGER THAN NEEDED: the spread settled by iteration 310; roughly 620
  warmup iterations would do for this problem and initialization
```

Two traps it is built around, both of which caught out an earlier attempt
at this:

- **The cross-chain spread is itself a noisy estimate**, with relative
  error ~1/√(2·n_chains) — 6% at 128 chains. A fixed 5% convergence band
  is *below the noise floor* and would never be satisfied, making every
  run look unconverged. The tolerance scales with chain count.
- **A stalled chain looks perfectly converged**, because a chain that has
  stopped moving has a perfectly stable spread. Worse, a stalled *HMC*
  chain has acceptance near **one**, not near zero, since arbitrarily
  small steps are always accepted — so acceptance alone points the wrong
  way. The test that works is whether the spread ever moved off its
  starting value at all; if it did not, the verdict is `INCONCLUSIVE`
  rather than a confident "cut your warmup". A real stalled run looks like
  this: spread frozen at the initialization ball, acceptance 0.98, step
  size 7×10⁻⁶.

Verdicts are `OK`, `LONGER THAN NEEDED`, `TOO SHORT`, `INCONCLUSIVE` and
`FAILED`. Pass `warmup_probes=0` to skip the recording entirely.

Note what this deliberately is *not*: automatic warmup termination. Given
that a stalled run is indistinguishable from a converged one on spread
alone, stopping warmup automatically would risk silently biased posteriors
— so anvil measures and tells you, and leaves the decision with you.

## Pipelining the sampling loop

Every sampling iteration normally ends in a blocking `mx.eval`, which is a
GPU round-trip with a ~167 µs floor *regardless of how much work it waits
on*. When a target's per-iteration GPU work is comparable to that floor,
the barrier — not the likelihood — is what you are paying for.

`run(..., pipeline=N)` keeps `N` iterations in flight and copies stored
frames out `N` steps behind, by which time they have landed. Measured at
512–1024 chains, median of interleaved repeats:

| GPU work / iteration | speedup at depth 2 |
|---|---:|
| ~200–300 µs (analytic posteriors, small-N fits) | **1.9–2.5×** |
| ~1.7 ms | 1.0× |
| ~8 ms (100k-point transit) | 1.0× |

Depth 2 is the sweet spot; depth 1 captures about half of it and depth 4
adds nothing. Peak memory was identical at every depth on the targets
measured. The default `pipeline="auto"` times the first few iterations and
enables it only where it pays, reporting the decision when `progress` is
on; pass an integer to override.

**Results are bit-identical at any depth** — pipelining changes when
arrays are evaluated, never what is computed, and `tests/test_engine.py`
asserts exact equality across all three kernels rather than checking it
statistically.

This applies to the sampling phase only. Warmup keeps its barrier because
adaptation reads freshly-computed parameters back on the host each
iteration, so pipelining it would mean adapting from stale ones — a
change in what the sampler does, not just when.

## Continuing a run instead of restarting it

The usual convergence loop is: run, check R-hat and ESS, and if they are
short, run longer. Restarting pays warmup again every round and — worse —
produces a *different* Markov chain, whose draws cannot honestly be
concatenated with the first attempt's. `resume=` extends the same chains
with the same frozen adaptation:

```python
res1 = anvil.run(kernel, target, u0, n_warmup=400, n_samples=200, seed=1)
if anvil.diagnose(res1.get_chain()).min_ess < 400:
    res2 = anvil.run(kernel, target, resume=res1, n_samples=400)
    draws = np.concatenate([res1.get_chain(), res2.get_chain()])
```

`resume=` takes a `Results` (or a `ResumeState` from
{func}`anvil.load_state`) and continues from its final positions, cached
log-probabilities and gradients, with `final_params` — step size,
trajectory length, and the diagonal or dense preconditioner — held exactly
as warmup left them. `n_warmup` defaults to 0 on a resume, since the
fresh-run default of 500 is not a request to re-adapt; an *explicit*
nonzero one raises rather than being silently discarded, because
re-adapting would break precisely the continuity being asked for. `u0` is
not used (pass it anyway and it is checked for a matching shape). On a 400-warmup, 512-chain ChEES run,
extending rather than restarting removes a ~50% tax from every round.

**The key stream continues.** This is the part that is easy to get silently
wrong. Keys are derived from `(seed, iteration, role)`, and the sampling
loop draws `key(n_warmup + t)` — so a naive resume with `n_warmup=0` would
redraw the *warmup* keys, correlating the continuation with the adaptation
phase it is meant to follow. Every `Results` therefore carries
`iters_consumed` (warmup plus `n_samples × thin`, accumulated across
resumes) and a continuation offsets from it. `seed` defaults to the one
recorded in the state; pass a different one to fork a segment deliberately.

`accept_fraction` and both divergence counts describe the resumed segment
alone — aggregate across rounds yourself.

Ensemble runs resume too (their `make_params` is empty, so there is little
to freeze), and the host-side bookkeeping that is not in the params rides
along in `kernel_ckpt`: ChEES's Halton jitter index, the ensemble's
move-mixing draw count. Restarting either of those would replay a sequence
the first segment had already used.

### Moving the chains between segments

A resume does not have to continue from where the chains stopped.
`ResumeState.with_positions(u, target)` returns the same state with the
chains somewhere else and every cached per-chain quantity recomputed
there:

```python
rs = res.resume_state()
u_new = my_exact_move(np.array(rs.state["u"]))        # your move, your RNG
res2 = anvil.run(kernel, target, resume=rs.with_positions(u_new, target),
                 n_samples=400)
```

This is the seam for composing a move anvil cannot see with anvil's
sampling. The motivating case is multimodality that HMC cannot cross: if
your model has a parameter whose conditional you can enumerate — a
per-epoch transit time whose likelihood term depends on that time alone,
say — you can redraw it exactly on a grid spanning the prior, with a
Metropolis-Hastings correction, and land in any mode with the right
probability. ChEES alternated with an exact move is still one Markov
chain, so the segments pool exactly as resumed segments already do.
Measured by a downstream caller on a weakly constrained transit time: total
variation from the exact marginal 0.162 → 0.019, R-hat 1.63 → 1.05, bulk
ESS 827 → 7,570, for 20% more wall clock.

**Why this is a method and not a line of your own code.** Moving chains by
hand means writing `state["u"]` and leaving the cached `log_prob` — and,
for ChEES, the cached `grad` — describing where the chain *used* to be. The
run continues, every array has the right shape, no diagnostic fires, and
the answer is wrong. `with_positions` recomputes through the kernel's
`refresh`, so a kernel that caches something beyond `{u, log_prob, grad}`
raises rather than copying it across: `refresh` is the contract that makes
a future cached quantity a loud failure instead of a quiet one.

What it validates: the shape of `u`, that the new positions are finite, and
that the target is finite at all of them (a chain whose current
log-probability is `-inf` can only escape by luck, so moving one there is
reported rather than accepted). What it leaves alone: `iteration` and
`seed`, so the next segment draws exactly the keys it would have drawn
without the move — the randomness for the move itself is yours. It returns
a new `ResumeState`; the original is untouched and still usable for
diagnostics.

### Across process boundaries

```python
res.save_state("run_state.npz")
state = anvil.load_state("run_state.npz")
res2 = anvil.run(kernel, target, resume=state, n_samples=400)
```

A plain `.npz`, **no pickle**: positions, cached log-prob and gradient, the
params (including the dense `corr` and `lrinv` as `(dim, dim)` matrices),
the iteration counter, the seed, the chain/dimension counts, and a
kernel/version tag. Everything is stored under a transparent name so a
caller can *read* the tag and refuse a mismatch rather than discover it
from a traceback. anvil checks what it can itself: a `dim` that disagrees
with the target, a `u0` whose shape disagrees with the saved chains, a
state written by a different kernel, or an internally inconsistent file
each raise a specific error naming both sides.

## Bounded parameters at their edges

A chain that reaches a bounded parameter's boundary used to stay there.
Two independent causes, both fixed:

- `log_det_jac` computed the bounded branch as `log(sig) + log1p(-sig)`,
  which is `-inf` for `|u| ≳ 18` because `mx.sigmoid(18.0)` is exactly
  `1.0` in float32. The identity `-|u| - 2·log1p(exp(-|u|))` is finite for
  every float32 `u`, and is *more* accurate well before saturation
  (−15.0000 against −15.0261 at `u = 15`). The value is unchanged
  elsewhere, so a run with no boundary chains cannot move.
- ChEES rejected a proposal flagged divergent even when the current state
  was already non-finite — vetoing the one move that rescues the chain. A
  divergence now only vetoes a proposal from a *finite* state.

Grazing transit geometries and low-signal timing offsets sit at their
bounds by construction, so with hundreds of chains a few arriving at an
edge is routine; before this, each was lost for the rest of the run,
silently, and dragged the shared step size down for everyone else.
`tests/test_boundary.py` starts chains at `u = 15` and `u = 25` and
requires them to rejoin the bulk.

## Diagnostics cost

{func}`anvil.diagnose` returns R-hat and bulk ESS from one shared pass;
prefer it to calling {func}`~anvil.diagnostics.split_rhat` and
{func}`~anvil.diagnostics.ess_bulk` separately, which repeats the rank
normalization that dominates the work. Measured at 400 draws × 2048
chains × 8 parameters: 1685 ms → 109 ms. Worth knowing because on cheap
targets the diagnostics used to cost several times the sampling run they
described.

### At large draw counts

Past a couple of million draws the diagnostics used to cost more memory
than the sampling did, and the ranking became the whole bill. Both are
fixed; measured on an M2 Max at 512 chains, against an AR(1) target:

| draws × chains × dim | before | after |
|---|---|---|
| 16,400 × 512 × 8 | 11.0 s, 11.3 GB device | **1.1 s, 3.8 GB** |
| 16,400 × 512 × 16 | 22.8 s, 22.6 GB device | **2.4 s, 3.8 GB** |
| 16,400 × 512 × 105 | ~130 s (one parameter at a time) | **16.2 s, 3.8 GB** |
| 4,000 × 512 × 8 | 0.23 s, 1.2 GB | 0.23 s, 1.2 GB |

R-hat and ESS are unchanged: against an exact float64 per-parameter
reference, max |ΔR̂| is 7×10⁻¹⁰ and max relative ΔESS is 1×10⁻⁷.

Two things were wrong. **Memory** grew as draws × chains × dim, because
`_autocov` took the FFT of every chain of every parameter at once — but
bulk ESS only ever uses the autocovariance *averaged over chains*, so that
average is now accumulated over chunks of chains and parameters are scored
in groups. Peak is set by `memory_budget` (default 2 GiB; the observed
device peak is about twice it) rather than by the problem size, so 105
parameters cost no more than 8.

**Speed** was MLX's `argsort`, and the interesting part is *why*:

```python
x = mx.array(np.random.standard_normal((2_095_105, 2)).astype("f4"))
o = np.array(mx.argsort(x, axis=0))
np.array_equal(np.sort(o[:, 0]), np.arange(len(o)))   # False
```

A strided multi-column `argsort(axis=0)` stops returning a permutation
above `1023 × 2048 = 2,095,104` rows — a tile counter, not a row counter,
and the same number at dim 2, 3, 8 and 105. A *contiguous 1-D* sort is
exact to at least 2²⁷ rows. The failure is silent: just past the limit
indices repeat, and further past it they come back as 2143289344, the bit
pattern of float32 NaN.

anvil previously guarded on `rows > 2**21`, which was wrong in both
directions — it let the corrupt multi-column sort run for rows in
(2,095,104, 2²¹], which is exactly where **512 chains × 4096 draws**
lands, and it sent every safe single-column sort above 2²¹ to a numpy
fallback 55× slower than the GPU. Ranking now goes parameter by parameter
through the 1-D sort (20 ms against numpy's 1.14 s at 8.4 M draws), which
is both correct at every size and the whole speedup.

MLX's sort is stable, so the ranks are bit-identical to
`np.argsort(kind="stable")` — verified with ties as the rule rather than
the exception (3 M draws taking 8 distinct values). The normal scores are
evaluated on the GPU in float32, except for the outermost 10⁻⁴ of each
tail, which is redone on the host in float64 because that is where the
quantile function is steep enough for float32 to lose 0.03 — those ranks
are known a priori, so only their positions come back from the device.
