# Joint fits: several targets, one parameter vector

Twelve visits of one planet, each with its own mid-time, normalisation and
noise model but one transit shape. Two instruments observing one signal.
Several epochs with shared physics. All of these are one composition:
several log-densities, each reading its own columns of a joint parameter
vector, with some columns shared. {func}`anvil.joint` builds it.

```python
jt = anvil.joint(
    {"v1": t1, "v2": t2, "v3": t3},          # TransformedLogDensity each
    shared={"ror": ["v1", "v2", "v3"],       # same name in every member
            "b":   ["v1", "v2", "v3"],
            "t0":  {"v1": "t0", "v2": "tmid", "v3": "t0"}},   # or per-member names
)
jt.names            # ['ror', 'b', 't0', 'v1.c', 'v1.sigma', 'v2.c', ...]
jt.member_columns("v2")                      # the joint columns v2 reads
res = anvil.run(anvil.ChEESHMC(jt), jt, u0)  # any tool, unchanged
```

## What it does

The joint columns are the shared parameters, in the order `shared` lists
them, then each member's unshared parameters in member order, named
`"<member>.<name>"`. Each member reads its own columns through one static
gather, and the members' **model-space** functions are summed. The joint
{class}`~anvil.Transform` then applies one Jacobian and one set of declared
priors.

That last point is the trap the function exists to avoid. Each member's
`log_prob(u)` already includes its own Jacobian and priors, so summing the
members' `log_prob` would count a shared parameter's Jacobian and prior
once per member. `joint` sums `model_log_prob` instead, which is the
likelihood plus whatever prior was written into the model function, and
nothing else. (`model_log_prob` and `model_log_prob_hi` are public on
{class}`~anvil.TransformedLogDensity` since 0.12.0 for exactly this use.)

Gradients need no extra code: the gather's adjoint scatter-adds, so a
shared column receives the sum of every member's gradient.

## What carries over

- **The float64 path** exists when every member has one, and is the same
  sum over the members' `model_log_prob_hi`; {func}`anvil.validate_precision`
  and {func}`anvil.certify` work as for any target.
- **Constants:** `model_log_norm` is the members' sum, so
  {func}`anvil.smc`'s ln Z is absolute when every member declared its
  likelihood's constants (for a {class}`~anvil.precision.ChunkedGaussianLogLike`,
  `lik.log_offset_const + lik.log_norm_const`), as for a single target.
- **Gradients** are supported when every member supports them.
- **Failed evaluations** keep their meaning: a member returning at or below
  the failure floor (−1e29) puts the joint value there too, and its inward
  slope on the shared columns survives the sum.
- {func}`anvil.optimize` with `objective="mle"` maximises the pooled
  likelihood; `"map"` adds the joint priors once.

## Rules

- The members' {class}`~anvil.ParamSpec`s for a shared parameter must agree
  in everything but the name (bounds, scaling, reporting units, prior).
  Two models disagreeing about one quantity's prior is a modelling decision
  to settle at the source, and `joint` refuses it with the two specs
  printed.
- The priors are compared as resolved, so `prior=None` on a finite box
  agrees with an explicit `Uniform()`.
- A member's parameter can be shared under one joint name only.
- Parameters that merely have the same name in two members stay separate
  (`v1.c`, `v2.c`) unless `shared` lists them. A `shared` entry naming one
  member is a rename.
- A member must not override `log_prob`: the sum is over `model_log_prob`,
  so tempering or an added term belongs in the model function. `joint`
  refuses such a member.
- Parameter names in a {class}`~anvil.Transform` must be unique (checked
  since 0.12.0). If a generated `"<member>.<name>"` collides with another
  joint name, the error says which two sources produced it.
- The result is a {class}`~anvil.JointLogDensity`, a
  {class}`~anvil.TransformedLogDensity` with `members`, `shared` and
  `member_columns()` added. A joint start is assembled by placing each
  member's start into its columns; a joint draw is split the same way.

## Memory and compilation

The joint target is one graph, so every member's intermediates are live
together: its peak memory is the members' sum, not the largest member's.
Measured with 12 Gaussian-likelihood members, the gradient's peak was 6 to
8 times that of evaluating the members one at a time, and roughly 20 to 30
bytes per chain per data point over all members. Chunking inside a member
does not bound it, because the chunks unroll into the same graph. The
levers are the number of chains and, for {func}`anvil.smc` and
{func}`anvil.optimize`, `max_rows`. Size a joint fit by the total number
of data points.

The members' values are combined through a stack, not a chained sum. A
chained sum is elementwise, and MLX would fuse every member's tail into
one Metal kernel and refuse the result (see
[the compile fallback](samplers.md)) for eight or more members that each
compile alone. That was 0.12.0's behaviour, fixed in 0.12.1.
