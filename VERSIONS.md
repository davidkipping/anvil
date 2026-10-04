# anvil versions

`anvil.__version__` is the single source of truth (`pyproject.toml` reads
it via `[tool.hatch.version]`), and `tests/test_version.py` fails if the
top row of this table disagrees with it.

**Every change to the public API bumps the version and adds a row here.**
Downstream packages (turin, anvil-gp) feature-detect rather than compare
version strings, which is the right thing to do — but they also *report*
`anvil.__version__`, and a non-editable install gives no other signal that
it is stale. Shipping an API addition without a bump cost turin a
debugging detour; see 0.2.0.

| version | commit | date | what changed |
|---|---|---|---|
| 0.4.0 | this | 2026-10-04 | **Correctness, silent:** `mx.compile` froze whatever a kernel's traced graphs read from the target, so a target that changed between runs (a Gibbs block it holds, a tempering beta, a swapped dataset, a retrained surrogate) went on being sampled as it was at the first trace — a small change gave a converged, healthy-looking run of the *old* posterior. `run` now calls `Kernel.retrace()` on every call and recomputes the cached log-density on resume; `ChEESHMC` and `EnsembleKernel` implement it, and a `self_compiled` kernel that does not override it is warned. Bit-identical for an unchanged target; ~2 ms, no drift over 100 segments. Also corrects the `reanchor_every` docs, which recommended it for a surrogate retrained mid-run — measured, it does not work, because re-anchoring refreshes the cache while the compiled proposal keeps the old target. Reported by SquishierPlanet. |
| 0.3.0 | this | 2026-10-02 | **`diagnose` correctness**: MLX's multi-column `argsort(axis=0)` silently stops permuting above 1023×2048 rows, which the old `rows > 2**21` guard did not catch — 512 chains × 4096 draws produced corrupted ranks, and so corrupted R-hat and ESS. Ranking above the limit now uses the contiguous 1-D sort, exact to 2²⁷ rows. **`diagnose` cost**: 11.0 s → 1.1 s and 11.3 GB → 3.8 GB at 16,400×512×8; ~130 s → 17.3 s at dim 105. `diagnose`/`split_rhat`/`ess_bulk` take `memory_budget=` (default 2 GiB), parameters are scored in groups and the chain-averaged autocovariance is accumulated over chunks of chains, so peak no longer grows with `dim`. Results unchanged: max \|ΔR̂\| 7e-10, max relative ΔESS 1e-7 against an exact float64 reference. |
| 0.2.0 | `849159f`+ | 2026-10-02 | **Resumable runs**: `run(..., resume=)`, `Results.save_state`/`anvil.load_state`, `Results.iters_consumed`/`seed`/`kernel`/`kernel_ckpt`, `ResumeState`. **Moving chains between segments**: `ResumeState.with_positions`, `Kernel.refresh`. **Kernel protocol**: `attach`/`checkpoint`/`restore`. **Bounded parameters**: non-saturating log-Jacobian, and a divergence no longer vetoes an escape from a non-finite state. `extras["divergent_per_chain"]`; `run(..., callback=)`. `n_warmup` and `seed` became `int | None`. |
| 0.1.0.dev0 | `104e754` | 2026-09-28 | Initial: ChEES-HMC (diagonal and `dense=True`) and ensemble moves on one engine, float32 conditioning discipline with a float64 CPU path, `diagnose`/`warmup_report`/`whitened_shape`/`validate_precision`/`certify`, `run(pipeline=)`, surrogate seams. |

## The 0.2.0 mistake, recorded

Everything in the 0.2.0 row shipped across two pushes (`f97f583`,
2026-09-29 and `849159f`, 2026-10-01) **while `__version__` still read
`0.1.0.dev0`**. Two consequences, both avoidable:

- a downstream package with a non-editable install saw no version change
  and had no reason to reinstall, so it ran against the old API and had to
  work out why the new one was missing;
- the version string cannot distinguish the two pushes after the fact, so
  a `0.1.0.dev0` reported by an older environment is ambiguous between
  three different API surfaces. Feature detection resolves it; the version
  string cannot.

Hence the single source of truth, this file, and the test that ties them
together. The fix for the ambiguity is not retroactive version numbers that
no installed anvil ever reported — it is the commit column above, and
bumping from here on.
