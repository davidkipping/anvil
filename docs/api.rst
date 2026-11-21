API reference
=============

Samplers (user-facing)
----------------------

.. automodule:: anvil.emcee_api
   :members: EnsembleSampler, HMCSampler

Engine
------

.. automodule:: anvil.engine
   :members: run, Results, ResumeState, load_state, load_results, SavedResults

Saved metadata
--------------

.. automodule:: anvil.metadata
   :members: environment, describe_target, describe_kernel, jsonable, dumps, loads, summary, record

Optimization
------------

.. automodule:: anvil.optimize
   :members: optimize, OptimizeResult

Evidence (optional)
-------------------

.. automodule:: anvil.evidence
   :members: smc, SMCResult, bridge_evidence, laplace_evidence,
             flow_evidence, EvidenceResult, TemperedLogDensity

Flow mode (optional)
--------------------

.. automodule:: anvil.flow
   :members: train, refine, Flow, FlowTarget, FlowReport

Kernels
-------

.. automodule:: anvil.kernels.chees
   :members: ChEESHMC

.. automodule:: anvil.kernels.ensemble
   :members: EnsembleKernel, StretchMove, DEMove, EnsembleMove

.. automodule:: anvil.kernels.metropolis
   :members: RandomWalkMetropolis

.. automodule:: anvil.kernels.base
   :members: Kernel

Log densities and transforms
----------------------------

.. automodule:: anvil.logdensity
   :members: LogDensity, FunctionLogDensity, CountingLogDensity

.. automodule:: anvil.transforms
   :members: ParamSpec, Transform, TransformedLogDensity, joint, JointLogDensity

.. automodule:: anvil.priors
   :members: Prior, Uniform, LogUniform, Gaussian, TruncatedGaussian,
             ImproperFlat

Precision
---------

.. automodule:: anvil.precision
   :members: PrecisionPolicy, chunked_sum, ChunkedGaussianLogLike,
             PrecisionReport, validate_precision, Certificate, certify

Diagnostics
-----------

.. automodule:: anvil.diagnostics
   :members: diagnose, Diagnostics, split_rhat, ess_bulk, nested_rhat,
             summary, whitened_shape, warmup_report, WarmupReport,
             lost_chains

Storage
-------

.. automodule:: anvil.storage
   :members: MemoryBackend, NpyStreamBackend

Surrogate seams (v2 preview)
----------------------------

.. automodule:: anvil.surrogate
   :members: TrainingArchive, ArchivingLogDensity, Surrogate,
             SwitchableLogDensity

Built-in targets
----------------

.. automodule:: anvil.targets.builtin
   :members: correlated_gaussian, rosenbrock, neals_funnel,
             make_transit_target, TransitTarget, trapezoid_flux,
             make_offset_flux, epoch_center_times
