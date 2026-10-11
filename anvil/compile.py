"""``mx.compile`` with a way out when MLX refuses the graph.

MLX fuses long elementwise chains into one Metal kernel. Past a certain
length the fused kernel needs more argument buffers than Metal provides,
and instead of splitting the fusion MLX raises::

    RuntimeError: [compile] Too many inputs/outputs fused in the Metal
    Compiled primitive which exhausted the available argument buffers ...

That is an MLX limitation, not a property of the target: the same function
evaluates correctly uncompiled, value and gradient. Two facts shape the
handling here, both measured on a real target (SquishierPlanet's collapsed
hybrid5 limb-darkening model, MLX 0.32.3):

* the error is raised at **evaluation**, not when the compiled function is
  called -- the call returns lazy arrays as usual -- so it cannot be caught
  around the call;
* it is raised again on every subsequent call of the same compiled
  function, so nothing is gained by retrying.

:class:`CompiledOrEager` therefore evaluates its outputs on the first call
only, and if that evaluation is refused it warns once and runs the plain
function from then on, evaluating each call so a large uncompiled graph
never spans more than one step. The sampled distribution is the same
either way; individual draws differ at float32 round-off, because fused
kernels -- and MLX's compiled random draws, measured 6e-6 apart from
eager ones with the same key -- round differently. Kernels remember a refusal so later retraces do not pay for the
failed trace again.
"""

from __future__ import annotations

import warnings

import mlx.core as mx

_REFUSAL_PREFIX = "[compile]"


def is_compile_refusal(exc: BaseException) -> bool:
    """True for the MLX compile errors worth falling back from -- any
    ``RuntimeError`` MLX's compiler reports, which it prefixes ``[compile]``
    -- and False for everything else, which should propagate as it is."""
    return isinstance(exc, RuntimeError) and str(exc).startswith(_REFUSAL_PREFIX)


# indirection so tests can inject a refusal without needing a target large
# enough to provoke MLX for real
_compile = mx.compile
_evaluate = mx.eval


class CompiledOrEager:
    """``mx.compile(fn)``, falling back to ``fn`` if MLX refuses the graph.

    ``enabled=False`` skips compilation outright (``compile_step=False``).
    ``on_refusal`` is called once if the compiled graph is refused, so the
    owner can remember not to try again.
    """

    def __init__(self, fn, *, name: str, enabled: bool = True,
                 on_refusal=None):
        self.fn = fn
        self.name = name
        self.compiled = bool(enabled)
        self._compiled_fn = _compile(fn) if enabled else None
        self._verified = not enabled
        self._on_refusal = on_refusal

    def _eager(self, *args, **kwargs):
        # Uncompiled, evaluate every call. Without fusion a large target's
        # graph is thousands of small arrays per call, and a kernel that
        # chains L such calls before the engine's per-iteration evaluation
        # (ChEES's leapfrog) exceeds Metal's buffer limit -- measured as
        # "[metal::malloc] Resource limit (499000) exceeded" on the target
        # that provoked all this. One sync per call keeps the live graph to
        # a single step; this path is the slow one already.
        out = self.fn(*args, **kwargs)
        _evaluate(out)
        return out

    def __call__(self, *args, **kwargs):
        if not self.compiled:
            return self._eager(*args, **kwargs)
        out = self._compiled_fn(*args, **kwargs)
        if self._verified:
            return out
        try:
            # MLX reports the refusal here, not at the call above
            _evaluate(out)
        except RuntimeError as exc:
            if not is_compile_refusal(exc):
                raise
            self.compiled = False
            self._compiled_fn = None
            warnings.warn(
                f"MLX refused to compile {self.name} and the engine is "
                f"continuing with it uncompiled. The sampled distribution is the "
                f"same; draws differ at float32 round-off, because fused "
                f"kernels and MLX's compiled random draws round differently "
                f"(~1e-6). Each "
                f"step is slower. This is an MLX limitation -- a long "
                f"elementwise chain fused into one Metal kernel exceeds the "
                f"kernel's argument-buffer limit -- not an error in the "
                f"target. Pass compile_step=False to skip the attempt. "
                f"({str(exc).splitlines()[0][:120]})",
                RuntimeWarning, stacklevel=3)
            if self._on_refusal is not None:
                self._on_refusal()
            return self._eager(*args, **kwargs)
        self._verified = True
        return out
