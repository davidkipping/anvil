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
| `arguments` | every argument of the call, as passed: for `run`, warmup, samples, thinning, seed, `compile_step`, `pipeline`, storage, whether it resumed |
| `resolved` (runs) | what the run then used: walkers, dimension, the seed in force, compilation, the pipeline depth chosen, the iteration it resumed from, iterations consumed |
| `kernel` (runs) | the sampler's class and settings: each ensemble move with its parameters and weight, or ChEES's step size, trajectory length, target acceptance, dense or diagonal |
| `target` | class and dimension; for a {class}`~anvil.TransformedLogDensity`, every parameter's name, bounds, u-space scaling, reporting units and the prior in force (and whether it was declared or implied); the model function's name; the declared normalising constant. For a flow target: the flow (size, fingerprint, how it was trained, its report) and the target it wraps |
| `wall_time_s` (runs) | the run's wall time |

A run's file also holds `chain` and `log_prob` (as `get_chain()` and
`get_log_prob()`), `accept_fraction`, the final positions, the adapted
parameters at the end of warmup (step size, mass matrix) and the
diagnostics (divergences, lost chains, whether compilation was refused).

A flow's file holds, besides its weights, the report and the `training`
record: the call that made it ({func}`~anvil.flow.train`,
{func}`~anvil.flow.refine` or {meth}`~anvil.flow.Flow.fit`), all of its
arguments including the SMC and fit settings, the final fit's settings, the
target, and the time each phase took. It also holds the training draws, so
a loaded flow starts where the original would (`save(path, pool=False)`
leaves them out; they are 4 MB for 100,000 draws at 10 parameters).

Arguments that are data are summarised rather than copied: an array by its
shape, a result by its kind, creation time and seed, a flow by its
fingerprint.

## What is not recorded

**The model function's code and the data.** A function is saved by its
name (`model_log_prob: "mymodule.transit_loglike"`), not its source, and
the light curve it closes over is not saved at all. Keep the script that
built the target, and the data, with the results.

## Reproducing a result

- **A run:** the same target, the same `run()` arguments and seed, and for
  flow mode the saved flow, reproduce the draws bit for bit on the same
  machine and versions. The metadata says what all of those were.
- **A flow:** keep the file. Retraining from the same seed gives the same
  flow only on the same chip and the same MLX and anvil versions.
- **Continuing a run:** `res.save_state(path)` writes the resume state (see
  {func}`anvil.load_state`); `res.save(path)` is the record of the draws.
  Save both if you may continue.
