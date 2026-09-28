API reference
=============

Samplers (user-facing)
----------------------

.. automodule:: anvil.emcee_api
   :members: EnsembleSampler, HMCSampler

Engine
------

.. automodule:: anvil.engine
   :members: run, Results

Kernels
-------

.. automodule:: anvil.kernels.chees
   :members: ChEESHMC

.. automodule:: anvil.kernels.ensemble
   :members: EnsembleKernel, StretchMove, DEMove, EnsembleMove

.. automodule:: anvil.kernels.metropolis
   :members: RandomWalkMetropolis

Log densities and transforms
----------------------------

.. automodule:: anvil.logdensity
   :members: LogDensity, FunctionLogDensity, CountingLogDensity

.. automodule:: anvil.transforms
   :members: ParamSpec, Transform, TransformedLogDensity

Precision
---------

.. automodule:: anvil.precision
   :members: PrecisionPolicy, chunked_sum, ChunkedGaussianLogLike,
             PrecisionReport, validate_precision, Certificate, certify

Diagnostics
-----------

.. automodule:: anvil.diagnostics
   :members: diagnose, Diagnostics, split_rhat, ess_bulk, nested_rhat,
             summary, whitened_shape, warmup_report, WarmupReport

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
