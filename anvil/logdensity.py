"""The pluggable target-density interface.

Everything in the engine consumes a :class:`LogDensity`; likelihood emulators
(BAMBI-style surrogates, v2) will slot in behind the same interface without
engine changes.

The batched contract: ``log_prob`` maps ``(n_chains, dim) -> (n_chains,)``
using MLX operations only. Chains must not interact — that is what makes the
summed-gradient trick valid and every kernel's vectorization correct.
"""

from __future__ import annotations

import mlx.core as mx


class LogDensity:
    """Base class for batched MLX log densities (unnormalized posteriors)."""

    #: dimensionality of the parameter space
    dim: int
    #: whether gradients may be requested (smooth, MLX-differentiable model)
    supports_grad: bool = True

    def log_prob(self, u: mx.array) -> mx.array:
        """Batched log density: (n_chains, dim) -> (n_chains,)."""
        raise NotImplementedError

    def log_prob_and_grad(self, u: mx.array) -> tuple[mx.array, mx.array]:
        """Per-chain value and gradient in one forward+backward pass.

        Because chains are independent, pulling back a vector of ones gives
        each chain its own gradient (the Jacobian is block-diagonal).
        """
        outputs, vjps = mx.vjp(
            self.log_prob, [u], [mx.ones(u.shape[:1], dtype=u.dtype)]
        )
        return outputs[0], vjps[0]


class FunctionLogDensity(LogDensity):
    """Wrap a plain batched callable ``(n, dim) -> (n,)`` as a LogDensity."""

    def __init__(self, fn, dim: int, supports_grad: bool = True):
        self._fn = fn
        self.dim = dim
        self.supports_grad = supports_grad

    def log_prob(self, u: mx.array) -> mx.array:
        return self._fn(u)


class CountingLogDensity(LogDensity):
    """Wrap another density and count evaluations (used by benchmarks).

    Counts batched *calls* and *chain-evaluations* separately; a call over
    4096 chains is one call and 4096 evaluations.
    """

    def __init__(self, inner: LogDensity):
        self.inner = inner
        self.dim = inner.dim
        self.supports_grad = inner.supports_grad
        self.n_calls = 0
        self.n_evals = 0
        self.n_grad_calls = 0

    def log_prob(self, u: mx.array) -> mx.array:
        self.n_calls += 1
        self.n_evals += u.shape[0]
        return self.inner.log_prob(u)

    def log_prob_and_grad(self, u: mx.array):
        self.n_grad_calls += 1
        self.n_evals += u.shape[0]
        return self.inner.log_prob_and_grad(u)


def validate_batched_signature(fn, dim: int) -> None:
    """Check that ``fn`` honors the batched MLX contract; raise with a porting
    hint otherwise (emcee users habitually pass per-walker numpy functions)."""
    probe = mx.zeros((3, dim), dtype=mx.float32)
    try:
        out = fn(probe)
    except Exception as exc:  # noqa: BLE001 - reporting, not handling
        raise TypeError(
            "log_prob_fn raised when called with a batched mx.array of shape "
            f"(3, {dim}). anvil requires a *batched MLX* log-probability: "
            "it receives every chain's parameters at once as an mx.array of "
            "shape (n_chains, dim) and must return an mx.array of shape "
            "(n_chains,) built from mlx.core operations (not numpy). "
            f"Original error: {exc!r}"
        ) from exc
    if not isinstance(out, mx.array) or out.shape != (3,):
        raise TypeError(
            "log_prob_fn must be batched: return an mx.array of shape (n_chains,); got "
            f"{type(out).__name__} with shape "
            f"{getattr(out, 'shape', 'n/a')}. If you are porting from emcee, "
            "replace numpy with mlx.core ops and drop any per-walker loop — "
            "the function is called once for the whole ensemble."
        )
