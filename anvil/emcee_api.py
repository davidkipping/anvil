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

import warnings

import mlx.core as mx
import numpy as np

from . import engine
from .kernels.base import Kernel
from .kernels.chees import ChEESHMC
from .kernels.ensemble import DEMove, EnsembleKernel, StretchMove
from .logdensity import FunctionLogDensity, LogDensity, validate_batched_signature


class EnsembleSampler:
    """emcee-style front door to the ensemble kernel: ``nwalkers`` walkers
    in ``ndim`` dimensions, ``log_prob_fn`` a batched MLX function
    ``(nwalkers, ndim) -> (nwalkers,)`` (checked at construction).
    ``moves``: a list of ``(move, weight)``, default the stretch move alone;
    ``kernel`` replaces the ensemble kernel outright. ``run_mcmc`` returns
    a :class:`~anvil.Results`; ``get_chain``, ``get_log_prob`` and
    ``acceptance_fraction`` read it as emcee's do. Stretch and DE moves have
    nothing to tune, so ``warmup`` only discards burn-in here.

    ``target``: a :class:`~anvil.LogDensity` (for instance a
    :class:`~anvil.TransformedLogDensity`) in place of ``log_prob_fn``.

    **Flow mode** (``flow=``, off by default), for posteriors with curved
    degeneracies: the walkers move in the coordinates of a normalizing flow
    (:mod:`anvil.flow`), and ``get_chain`` reads in the target's own.
    ``flow=True`` trains one at the first ``run_mcmc`` with
    :func:`anvil.flow.train` -- SMC from the prior, then refinement rounds,
    minutes to an hour on an expensive likelihood -- which needs ``target=`` a
    :class:`~anvil.TransformedLogDensity` with proper priors; the trained
    flow is kept as the ``flow`` attribute, its report printed, and later runs reuse
    it. ``flow=`` a trained :class:`~anvil.flow.Flow` uses that one. The
    default moves become stretch and differential evolution, half each
    (the mix measured), and ``initial_state`` may be None to start from
    the flow's training draws (saved with the flow by ``Flow.save``; a
    flow without them -- ``save(pool=False)``, a file from before 0.14.0,
    or one refitted with ``Flow.fit`` -- starts from its own distribution,
    and needs warmup); a later ``run_mcmc(None, ...)`` continues the
    previous run. ``kernel=`` cannot
    be combined with a flow: a kernel is built on the target it samples.
    For control over training (``rounds``, ``smc_kwargs``, ...) call
    :func:`anvil.flow.train` yourself and pass ``flow=`` the result."""

    def __init__(
        self,
        nwalkers: int,
        ndim: int,
        log_prob_fn=None,
        *,
        target: LogDensity | None = None,
        moves=None,
        kernel: Kernel | None = None,
        seed: int = 0,
        flow=False,
    ):
        if (log_prob_fn is None) == (target is None):
            raise TypeError("give log_prob_fn or target=, not both or neither")
        self.nwalkers = int(nwalkers)
        self.ndim = int(ndim)
        if target is None:
            validate_batched_signature(log_prob_fn, ndim)
            target = FunctionLogDensity(log_prob_fn, ndim)
        elif int(target.dim) != self.ndim:
            raise ValueError(f"target has dim {target.dim}, but ndim={self.ndim}")
        self.target = target
        self.seed = seed
        self._moves = moves
        self._kernel = kernel
        self._results: engine.Results | None = None
        self._init_flow(flow)

    def _init_flow(self, flow) -> None:
        from .flow import Flow
        from .transforms import TransformedLogDensity

        if flow is False or flow is None:
            self.flow = None
            self._train_flow = False
            return
        if self._kernel is not None:
            raise ValueError("kernel= cannot be combined with flow=: a kernel is "
                             "built on the target it samples, and in flow mode that "
                             "is the target in the flow's coordinates")
        if isinstance(flow, Flow):
            if flow.dim != self.ndim:
                raise ValueError(f"the flow has dim {flow.dim}, but ndim={self.ndim}")
            self.flow, self._train_flow = flow, False
        elif flow is True:
            if not (isinstance(self.target, TransformedLogDensity)
                    and self.target.is_proper):
                raise ValueError(
                    "flow=True trains the flow from the prior (anvil.smc), so it needs "
                    "target= a TransformedLogDensity with proper priors; or pass flow= a "
                    "Flow trained with anvil.flow.train(..., draws=...)")
            if self.ndim < 2:
                raise ValueError("a coupling flow needs ndim >= 2")
            #: the flow in use (trained at the first run_mcmc for flow=True)
            self.flow, self._train_flow = None, True
        else:
            raise TypeError(f"flow= takes True, False or an anvil.flow.Flow, "
                            f"not {type(flow).__name__}")
        # checked here, not at the first run_mcmc: flow=True trains first,
        # for minutes to an hour
        if self.nwalkers % 2:
            raise ValueError("ensemble moves require an even number of walkers")
        if self.nwalkers < 2 * self.ndim:
            raise ValueError(f"nwalkers={self.nwalkers} < 2*ndim={2 * self.ndim}: each "
                             f"half of the ensemble must span the parameter space")

    def _sampled_target(self) -> LogDensity:
        return self.target if self.flow is None else self.flow.wrap(self.target)

    def _make_kernel(self, target: LogDensity | None = None) -> Kernel:
        if self._kernel is not None:
            return self._kernel
        target = self.target if target is None else target
        default = ([(StretchMove(), 0.5), (DEMove(), 0.5)] if self.flow is not None
                   else [(StretchMove(), 1.0)])
        return EnsembleKernel(target, moves=self._moves or default, seed=self.seed)

    def run_mcmc(
        self,
        initial_state,
        nsteps: int,
        *,
        warmup: int | None = None,
        tune: bool = False,
        thin_by: int = 1,
        progress: bool = False,
        compile_step: bool | None = None,
    ) -> engine.Results:
        """Run the chain. ``initial_state``: (nwalkers, ndim) array-like.

        ``warmup`` iterations adapt and are discarded; if None, defaults to
        ``nsteps`` when ``tune=True`` else 0. ``nsteps`` states per walker
        are then recorded, one every ``thin_by`` frozen iterations.
        ``compile_step`` is passed to :func:`anvil.run` (False runs the
        step uncompiled, for a ``log_prob`` that leaves MLX).

        In flow mode ``initial_state`` is in the target's own coordinates
        (it is mapped into the flow's), and may be None. ``flow=True``
        trains the flow at the first call. With None, the first run starts
        from the flow's training draws: ``run_mcmc(None, nsteps,
        warmup=...)``. A later call with None continues the previous run
        where it stopped, as one Markov chain (frozen, so ``warmup`` must
        be 0), and ``get_chain`` then holds the new segment.
        """
        if initial_state is not None:
            # checked before any training: flow=True trains for minutes to
            # an hour, and a wrong shape found after it would waste that
            shape = np.shape(initial_state)
            if shape != (self.nwalkers, self.ndim):
                raise ValueError(
                    f"initial_state must have shape ({self.nwalkers}, {self.ndim}), "
                    f"got {tuple(shape)}")
        elif self.flow is None and not self._train_flow:
            raise TypeError("run_mcmc() needs initial_state (it may be None "
                            "only in flow mode)")
        if self._train_flow and self.flow is None:
            from .flow import train
            self.flow = train(self.target, n_chains=self.nwalkers, seed=self.seed,
                              progress=progress,
                              compile_step=True if compile_step is None else compile_step)
            print(self.flow.report, flush=True)
        sampled = self._sampled_target()
        if warmup is None:
            warmup = nsteps if tune else 0
        resume, p0 = None, None
        if initial_state is None:
            last = self._results
            if last is not None and last.target_tag == getattr(sampled, "resume_tag", ""):
                # continue, as emcee does: restarting from the training draws
                # with the same seed would replay the first run draw for draw
                resume = last
                if warmup:
                    raise ValueError(
                        "run_mcmc(None, ...) continues the previous run, which cannot "
                        "warm up again (its draws would no longer be one Markov chain); "
                        "for a fresh run with warmup, start from the last positions: "
                        "run_mcmc(sampler.get_chain()[-1], nsteps, warmup=...)")
            else:
                p0 = self.flow.start(self.nwalkers, seed=self.seed)
                if warmup == 0 and (self.flow.pool is None or last is not None):
                    why = ("this flow has no training draws (pool), so the walkers start "
                           "from the flow's own distribution" if self.flow.pool is None else
                           "the flow changed since the last run, so this is a new chain "
                           "from the flow's training draws")
                    warnings.warn(f"{why}, and with warmup=0 its way to the posterior "
                                  f"is recorded; pass warmup=", RuntimeWarning, stacklevel=2)
        else:
            p0 = mx.array(np.asarray(initial_state, dtype=np.float32))
            if self.flow is not None:
                p0 = self.flow.to_latent(p0)
        if p0 is not None and p0.shape != (self.nwalkers, self.ndim):
            raise ValueError(
                f"initial_state must have shape ({self.nwalkers}, {self.ndim}), "
                f"got {tuple(p0.shape)}"
            )
        if warmup == 0 and isinstance(self, HMCSampler):
            warnings.warn(
                "HMCSampler.run_mcmc with no warmup: ChEES adapts its step "
                "size, trajectory length and preconditioner only during "
                "warmup, so this run uses the starting values (step 0.1, "
                "length 1.0) unadapted; pass warmup= or tune=True",
                RuntimeWarning, stacklevel=2)
        self._results = engine.run(
            self._make_kernel(sampled),
            sampled,
            p0,
            n_warmup=warmup,
            n_samples=nsteps,
            thin=thin_by,
            seed=None if resume is not None else self.seed,
            progress=progress,
            compile_step=compile_step,
            resume=resume,
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
        if "flow" in kernel_kwargs:
            raise TypeError("HMCSampler has no flow mode; use EnsembleSampler(flow=...), "
                            "or wrap the target yourself (anvil.flow)")
        # flow mode is the ensemble sampler's: ChEES trailed it 20-40x in
        # flow coordinates on the oblate fit (anvil.flow docstring)
        super().__init__(nwalkers, ndim, log_prob_fn, target=target, seed=seed)
        self._kernel_kwargs = kernel_kwargs

    def _make_kernel(self, target: LogDensity | None = None) -> Kernel:
        return ChEESHMC(self.target if target is None else target, **self._kernel_kwargs)

    @property
    def n_divergent(self) -> int:
        return self._require_results().extras.get("n_divergent", 0)
