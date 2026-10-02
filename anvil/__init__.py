"""anvil: MCMC sampling engine optimized for Apple Silicon via MLX.

Thousands of parallel chains on the GPU in well-conditioned float32;
float64 offset handling and diagnostics on the CPU; one shared engine
under gradient-based (ChEES-HMC) and gradient-free (ensemble) kernels.
"""

from . import diagnostics
from .diagnostics import (Diagnostics, WarmupReport, diagnose,
                          warmup_report, whitened_shape)
from .emcee_api import EnsembleSampler, HMCSampler
from .engine import Results, ResumeState, load_state, run
from .kernels import ChEESHMC, DEMove, EnsembleKernel, StretchMove
from .logdensity import FunctionLogDensity, LogDensity
from .precision import (Certificate, PrecisionPolicy, certify,
                        validate_precision)
from .transforms import ParamSpec, Transform, TransformedLogDensity

__version__ = "0.2.0"

__all__ = [
    "ChEESHMC",
    "DEMove",
    "EnsembleKernel",
    "EnsembleSampler",
    "HMCSampler",
    "FunctionLogDensity",
    "LogDensity",
    "ParamSpec",
    "Certificate",
    "Diagnostics",
    "PrecisionPolicy",
    "Results",
    "ResumeState",
    "StretchMove",
    "Transform",
    "WarmupReport",
    "TransformedLogDensity",
    "diagnose",
    "diagnostics",
    "load_state",
    "certify",
    "run",
    "validate_precision",
    "warmup_report",
    "whitened_shape",
]
