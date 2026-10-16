# Optimization: maximum likelihood and MAP

anvil's third mode finds a best fit rather than a posterior.
{func}`anvil.optimize` runs many starting points at once, the way the samplers
run many chains: every objective and gradient evaluation is one batched call
over all the starts still moving.

```python
res = anvil.optimize(target)            # MLE from 128 prior draws
print(res)                              # best fit, +/- Laplace sd, flags
u0 = res.ball(512)                      # start ChEES around it
run(ChEESHMC(target), target, u0)
```

## What is maximised

`objective=` chooses one of three functions. They have different maxima
whenever a prior is declared or a parameter is bounded:

| `objective` | maximises | needs |
|---|---|---|
| `"mle"` (default) | the log-likelihood alone: no prior, no Jacobian | a {class}`~anvil.TransformedLogDensity` |
| `"map"` | likelihood + the priors declared on each {class}`~anvil.ParamSpec`, in model units | a {class}`~anvil.TransformedLogDensity` |
| `"density"` | the target's own `log_prob` | any {class}`~anvil.LogDensity` |

A sampler's `log_prob` is the posterior **in u**, the unbounded space anvil
samples in. It includes the log-Jacobian of the map from u to your model
units. Its maximum is therefore neither the maximum-likelihood point nor the
MAP you would compute in your own units: it pulls bounded parameters away from
their edges. `"mle"` and `"map"` leave the Jacobian out.

The search still runs in u, so no step can leave a box. Without the Jacobian
the optimum in u maps exactly onto the model-space optimum, since a
reparameterisation moves a function's maximum but not its value.

`"mle"` drops the priors declared on ParamSpecs. A prior written into your
`model_log_prob` itself is part of what anvil sees as the likelihood, and
stays in.

A plain {class}`~anvil.FunctionLogDensity` is one function, so anvil cannot
separate likelihood from prior in it; use `objective="density"`.

## Methods

| `method` | what it does | when |
|---|---|---|
| `"mixed"` (default) | L-BFGS for the large moves, then damped Newton to finish | gradient targets |
| `"lbfgs"` | L-BFGS only | well-conditioned problems, or high dimension |
| `"newton"` | damped Newton only | small dimension, starts already close |
| `"de"` | differential evolution | targets without gradients |

The default was chosen by measurement. Two targets were used: the
20,000-point transit likelihood in `anvil.targets` and Elome's spherical
transit model. The score is the share of 128 starts in a ball that ended
within 0.01 nats of a float64 scipy reference optimum:

| target | Adam (fixed step) | L-BFGS | Newton | mixed |
|---|---|---|---|---|
| transit, 20k points | 0% | 89% | 66% | **95%** (0.5 s) |
| Elome, 20k points | 0% | 0% | 80% | **98%** (0.6 s) |

What the rows show:

- **Elome's curvature in u spans a condition number of about 8×10⁶.** L-BFGS
  stops 0.01–0.1 nats short there.
- **Newton alone gets there from fewer starts**, because far from the optimum
  the Hessian is a poor guide.
- **Adam with a fixed step ends every start more than 0.6 nats short.** With
  a fixed step it keeps bouncing in the stiff directions. turin recorded the
  same failure on sharp posteriors, hundreds to thousands of nats short. Adam
  is not offered.

Newton's Hessian comes from central differences of batched float32 gradients,
at 2 × dim gradient rows per start per iteration. A recorded case showed an
autodiff Hessian with the wrong sign. The small dense algebra is done on the
host in float64. At dozens of parameters and up, that cost adds up, and
`"lbfgs"` becomes the better choice.

## Starts

```python
anvil.optimize(target)                      # 128 draws from the (proper) prior
anvil.optimize(target, u_guess)             # one point (dim,), spread by start_spread
anvil.optimize(target, U)                   # your own starts, (n, dim), in u
```

**Many starts beat good starts.** From wide prior draws on the transit
targets only 18–34% of starts reach the global optimum, and that is a
property of the likelihood:

- a start whose transit misses the data climbs to the "no transit" plateau
  by shrinking the planet;
- others stop in genuine local optima.

With 128 starts, enough get through that the best is found every time.
Failed starts cost little. A problem-specific global step belongs in your
code, passed in as `u0`; turin's grid sweep over transit times is one.

## Reading the result

{class}`~anvil.OptimizeResult` gives the best point in three forms:

- `u`, in sampling space;
- `model`, in your model units;
- `physical`, in reporting units.

The values at that point:

- **`value`** is the objective at the best point, on the target's float64
  path when it has one. The best is chosen by rescoring the leading starts in
  float64, because float32 cannot rank starts that all sit at the optimum.
- **`log_value`** adds back every float64 constant the graph leaves out:
  - for `"mle"`, the absolute log-likelihood;
  - for `"map"` with proper priors, the normalised model-space log posterior
    density.

Per start, the result records:

- `values` and `status`:
  - `"converged"`: a local maximum, to within `tol` nats;
  - `"saddle"`: stationary or flat, but not a maximum;
  - `"stalled"`: it stopped climbing without passing the test;
  - `"max_iter"`;
  - `"failed"`: the objective was non-finite at the start.
- `decrement`: the gain a Newton step still predicted, in nats.

`optima` lists the distinct optima the starts found. The printed summary warns
when a second optimum lies within 10 nats of the best.

`at_bound` marks parameters whose optimum is on a bound, a common case for a
maximum-likelihood fit. The point is moved onto the bound, and the
uncertainty for that parameter is NaN.

`hessian_u`, `cov_u`, `cov_model` and `sd` give the curvature at the best
point and the Laplace covariance, in u, model and physical units.

## Convergence and float32

Near the optimum of a data-heavy likelihood, float32 cannot resolve the last
step's change in value. The gradient still resolves the optimum. So:

- **Convergence is judged by the Newton decrement**: the gain the next Newton
  step predicts, in nats.
- **The line search accepts a step whose value has not fallen by more than
  float32 resolution.**

The decrement has a floor of its own:
- **2×10⁻⁵ nats on the transit target**;
- **6×10⁻⁴ to 1.2×10⁻³ on Elome**, at 20k and 100k points. Elome's analytic
  gradient is not quite the derivative of its float32 values.

That is why the default `tol` is 10⁻² nats. A converged start still takes
the Newton step it has just computed, which lands it far inside `tol`: the
best points on both targets were within 10⁻⁶ nats of the float64 optimum.

If a target's floor is above `tol`, starts at the optimum end `"stalled"`;
raise `tol`. If many starts end `"stalled"` with `method="lbfgs"`, the
problem is ill-conditioned; use `"mixed"`.

A start at a saddle, or on a non-concave stretch such as Rosenbrock off its
valley, also steps along the direction of most negative curvature. So it
does not stop at a stationary point that is not a maximum. An example is
f = fc² + fs², which is stationary at the origin.

## Starting a sampler from the fit

```python
u0 = res.ball(512, spread=1e-3)     # a tight ball: round posteriors
u0 = res.laplace_starts(512)        # Laplace draws: curved ones
```

ChEES adapts its step size and mass matrix from the spread of its chains, so
a tight ball around a curved posterior can leave every chain frozen.
`laplace_starts` follows turin's recipe:

- draw from half the Laplace width;
- cap each principal sd at 0.5 in u;
- pull back halfway any draw more than 25 nats below the best, up to 12
  times.

A parameter whose optimum is on a bound is started at |u| = 8 on that side,
close to the bound but where the sampler's Jacobian term is not saturated.
