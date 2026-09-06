"""emcee-familiar facade over the batched engine.

Near-drop-in mental model: build a sampler with ``(nwalkers, ndim,
log_prob_fn)``, call ``run_mcmc``, read draws with ``get_chain``. The one
hard requirement that differs from emcee: ``log_prob_fn`` must be *batched
MLX* — it receives all walkers at once as an ``mx.array`` of shape
``(nwalkers, ndim)`` and returns shape ``(nwalkers,)``. A helpful TypeError
fires at construction if it does not.

Unlike emcee, adaptation ("tuning") and recording are separated: warmup
steps adapt the move and are never recorded, then recording runs with the
move frozen (this keeps the recorded chain a valid MCMC sample).
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np

from . import engine
from .kernels.base import Kernel
from .kernels.chees import ChEESHMC
from .kernels.ensemble import EnsembleKernel, StretchMove
from .logdensity import FunctionLogDensity, LogDensity, validate_batched_signature


class EnsembleSampler:
    def __init__(
        self,
        nwalkers: int,
        ndim: int,
        log_prob_fn,
        *,
        moves=None,
        kernel: Kernel | None = None,
        seed: int = 0,
    ):
        validate_batched_signature(log_prob_fn, ndim)
        self.nwalkers = int(nwalkers)
        self.ndim = int(ndim)
        self.target = FunctionLogDensity(log_prob_fn, ndim)
        self.seed = seed
        self._moves = moves  # consumed once ensemble moves land (M3)
        self._kernel = kernel
        self._results: engine.Results | None = None

    def _make_kernel(self) -> Kernel:
        if self._kernel is not None:
            return self._kernel
        moves = self._moves or [(StretchMove(), 1.0)]
        return EnsembleKernel(self.target, moves=moves, seed=self.seed)

    def run_mcmc(
        self,
        initial_state,
        nsteps: int,
        *,
        warmup: int | None = None,
        tune: bool = False,
        thin_by: int = 1,
        progress: bool = False,
    ) -> engine.Results:
        """Run the chain. ``initial_state``: (nwalkers, ndim) array-like.

        ``warmup`` iterations adapt and are discarded; if None, defaults to
        ``nsteps`` when ``tune=True`` else 0. ``nsteps`` states per walker
        are then recorded, one every ``thin_by`` frozen iterations.
        """
        p0 = mx.array(np.asarray(initial_state, dtype=np.float32))
        if p0.shape != (self.nwalkers, self.ndim):
            raise ValueError(
                f"initial_state must have shape ({self.nwalkers}, {self.ndim}), "
                f"got {tuple(p0.shape)}"
            )
        if warmup is None:
            warmup = nsteps if tune else 0
        self._results = engine.run(
            self._make_kernel(),
            self.target,
            p0,
            n_warmup=warmup,
            n_samples=nsteps,
            thin=thin_by,
            seed=self.seed,
            progress=progress,
        )
        return self._results

    # -- emcee-style accessors -------------------------------------------

    def _require_results(self) -> engine.Results:
        if self._results is None:
            raise RuntimeError("run_mcmc() has not been called yet")
        return self._results

    def get_chain(self, *, discard: int = 0, thin: int = 1, flat: bool = False):
        return self._require_results().get_chain(discard, thin, flat)

    def get_log_prob(self, *, discard: int = 0, thin: int = 1, flat: bool = False):
        return self._require_results().get_log_prob(discard, thin, flat)

    @property
    def acceptance_fraction(self) -> np.ndarray:
        return self._require_results().accept_fraction


class HMCSampler(EnsembleSampler):
    """ChEES-HMC front door with the same verbs as EnsembleSampler.

    Requires an MLX-differentiable log_prob. "Walkers" are independent HMC
    chains that share adaptation statistics during warmup; run thousands.
    A meaningful ``warmup`` (or ``tune=True``) is required — all adaptation
    happens there and is frozen for the recorded steps.
    """

    def __init__(
        self,
        nwalkers: int,
        ndim: int,
        log_prob_fn=None,
        *,
        target: LogDensity | None = None,
        seed: int = 0,
        **kernel_kwargs,
    ):
        if target is not None:
            self.target = target
            self.nwalkers, self.ndim, self.seed = int(nwalkers), int(ndim), seed
            self._moves, self._results = None, None
        else:
            super().__init__(nwalkers, ndim, log_prob_fn, seed=seed)
        self._kernel_kwargs = kernel_kwargs
        self._kernel = None

    def _make_kernel(self) -> Kernel:
        return ChEESHMC(self.target, **self._kernel_kwargs)

    @property
    def n_divergent(self) -> int:
        return self._require_results().extras.get("n_divergent", 0)
