# Saving results: what was run, and how

Every result anvil returns carries `metadata`, a record of how it was made,
and every result can be saved with it. The files are plain `.npz` archives
(no pickle) or JSON, readable without anvil.

```python
res = anvil.run(kernel, target, u0, n_warmup=500, n_samples=2000, seed=3)
res.save("run.npz")                      # draws + how they were made
saved = anvil.load_results("run.npz")
saved.get_chain(), saved.names, saved.metadata["kernel"]

flow.save("flow.npz")                    # weights, training record, report, training draws
anvil.smc(target).save("smc.npz")        # particles, ln Z, stage path, the call
anvil.flow_evidence(target, flow).save("evidence.json")
```

## What is recorded

| block | contents |
|---|---|
| `environment` | anvil, MLX, NumPy and Python versions; the chip (e.g. "Apple M2 Max"), OS and MLX device; the time (UTC) |
| `arguments` | every argument of the call, as passed: for `run`, warmup, samples, thinning, seed, `compile_step`, `pipeline`, storage, whether it resumed, and the starting positions by shape and hash |
| `resolved` (runs) | what the run then used: walkers, dimension, the seed in force, compilation (and whether MLX refused it), the pipeline depth chosen, the iteration it resumed from, iterations consumed |
| `kernel` (runs) | the sampler's class and settings: each ensemble move with its parameters and weight, or ChEES's step size, trajectory length, target acceptance, dense or diagonal |
| `target` | class and dimension; for a {class}`~anvil.TransformedLogDensity`, every parameter's name, bounds, u-space scaling, reporting units and the prior in force (and whether it was declared or implied); the model function's name; the declared normalising constant. For a flow target: the flow (size, fingerprint, how it was trained, its report) and the target it wraps |
| `wall_time_s` (runs) | the run's wall time |

A run's file also holds `chain` and `log_prob` (as `get_chain()` and
`get_log_prob()`), `accept_fraction`, the final positions (`final_u`, in the
target's own coordinates like the chain; for a run in a flow's coordinates
the latent ones are `final_z`), the adapted parameters at the end of
warmup (step size, mass matrix) and the diagnostics (divergences, lost
chains, whether compilation was refused).

A flow's file holds, besides its weights, the report and the `training`
record: the call that made it ({func}`~anvil.flow.train`,
{func}`~anvil.flow.refine` or {meth}`~anvil.flow.Flow.fit`), all of its
arguments including the SMC and fit settings, the final fit's settings, the
target, and the time each phase took. It also holds the training draws, so
a loaded flow starts where the original would (`save(path, pool=False)`
leaves them out; they are 4 MB for 100,000 draws at 10 parameters).

Arguments that are data are summarised rather than copied: an array by its
shape (any array over 256 elements), a result by its kind, creation time
and seed, a flow by its fingerprint. A function is recorded by its name
alone, even one that holds data (a dataclass with `__call__`, an
`mlx.nn.Module`). Non-finite numbers are written as the strings "inf",
"-inf" and "nan", since strict JSON has no tokens for them; anvil's loaders
turn them back into floats.

## What is not recorded

**The model function's code, the data and the starting positions.** A
function is saved by its name (`model_log_prob: "mymodule.transit_loglike"`),
not its source, and the light curve it closes over is not saved at all.
The starting positions are recorded by shape and hash only, enough to
check a reproduction started from the same place but not to recreate it.
Keep the script that built the target, the data and the start with the
results.

## Reproducing a result

- **A run:** the same target, starting positions, `run()` arguments and
  seed, and for flow mode the saved flow, reproduce the draws bit for bit
  on the same machine and versions. The metadata says what the arguments
  and settings were, and identifies the start by its hash.
- **A flow:** keep the file. Retraining from the same seed gives the same
  flow only on the same chip and the same MLX and anvil versions.
- **Continuing a run:** `res.save_state(path)` writes the resume state (see
  {func}`anvil.load_state`); `res.save(path)` is the record of the draws.
  Save both if you may continue, and in flow mode the flow as well: a
  resume state works only with the flow that wrote it.
