"""What a result records about how it was made, for reproducibility.

Every saved result (:meth:`anvil.Results.save`, :meth:`anvil.flow.Flow.save`,
:meth:`anvil.EvidenceResult.save`, :meth:`anvil.SMCResult.save`) carries a
JSON block built from these pieces:

* :func:`environment` -- anvil, MLX, NumPy and Python versions, the chip
  and operating system, and the time (UTC).
* :func:`describe_target` -- the target's class and dimension; for a
  :class:`~anvil.TransformedLogDensity` its parameters (names, bounds,
  scalings and priors) and the model function's name; for a flow target
  the flow and the target it wraps.
* :func:`describe_kernel` -- the sampler's class and settings, including
  each ensemble move and its weight.

What none of this can record is the model function's code or the data: a
function is saved by name only. Keep the script that built the target
with the results.
"""

from __future__ import annotations

import dataclasses
import datetime
import json
import math
import platform
import sys

import mlx.core as mx
import numpy as np

#: version of the metadata layout, bumped when a key changes meaning
FORMAT = 1


def environment() -> dict:
    """Versions, machine and time."""
    from . import __version__

    try:
        chip = mx.device_info().get("device_name", "")
    except Exception:                                   # noqa: BLE001
        chip = ""
    return {
        "anvil": __version__,
        "mlx": getattr(mx, "__version__", ""),
        "numpy": np.__version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "chip": chip,
        "device": str(mx.default_device()),
        "created_utc": datetime.datetime.now(datetime.timezone.utc)
        .isoformat(timespec="seconds"),
    }


def _name_of(fn) -> str:
    if fn is None:
        return ""
    mod = getattr(fn, "__module__", None) or ""
    name = getattr(fn, "__qualname__", None) or type(fn).__name__
    return f"{mod}.{name}" if mod else name


#: arrays with more elements than this are recorded by shape and dtype
_MAX_LISTED = 256
_NON_FINITE = {"inf": math.inf, "-inf": -math.inf, "nan": math.nan}


def jsonable(obj):
    """``obj`` as plain JSON types: NumPy scalars and small arrays to
    numbers and lists (arrays over 256 elements by shape and dtype),
    non-finite floats to the strings "inf", "-inf" and "nan" (strict JSON
    has no tokens for them; :func:`loads` turns them back), dataclasses to
    dicts, anything else to its ``repr``. A callable -- a model function,
    even one that is a dataclass or an ``mlx.nn.Module`` holding data -- is
    recorded by its name only."""
    if obj is None or isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        return f if math.isfinite(f) else repr(f)
    if isinstance(obj, (np.ndarray, mx.array)):
        if obj.size > _MAX_LISTED:
            return {"array": list(obj.shape), "dtype": str(obj.dtype)}
        obj = np.array(obj)
        return [jsonable(v) for v in obj.tolist()] if obj.ndim else jsonable(obj.item())
    # before the dataclass and dict branches: a callable model's fields are
    # its data, which a record must not copy
    if callable(obj):
        return _name_of(obj)
    if dataclasses.is_dataclass(obj):
        return {"class": type(obj).__name__,
                **{f.name: jsonable(getattr(obj, f.name)) for f in dataclasses.fields(obj)}}
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    return repr(obj)


def dumps(meta: dict) -> str:
    """``meta`` as a JSON string, through :func:`jsonable`."""
    return json.dumps(jsonable(meta), indent=1, sort_keys=False)


def _decode(obj):
    if isinstance(obj, str):
        return _NON_FINITE.get(obj, obj)
    if isinstance(obj, dict):
        return {k: _decode(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_decode(v) for v in obj]
    return obj


def loads(text: str):
    """Read what :func:`dumps` wrote, with "inf", "-inf" and "nan" back as
    floats."""
    return _decode(json.loads(text))


def _described(obj, fallback: dict) -> dict:
    # a description is a record, never a reason for a finished run to fail:
    # a subclass whose describe() breaks is recorded as such
    d = getattr(obj, "describe", None)
    if not callable(d):
        return fallback
    try:
        return jsonable(d())
    except Exception as exc:                            # noqa: BLE001
        return {**fallback, "describe_error": f"{type(exc).__name__}: {exc}"}


def describe_target(target) -> dict:
    """The target's own ``describe()`` if it has one, else its class and
    dimension."""
    return _described(target, {"class": type(target).__name__,
                               "dim": jsonable(getattr(target, "dim", None))})


def describe_kernel(kernel) -> dict:
    """The kernel's own ``describe()`` (class and settings), or its class."""
    return _described(kernel, {"class": type(kernel).__name__})


def summary(value):
    """A call argument as a saved result records it: small values as
    themselves, arrays by shape, objects that describe themselves by their
    description, results and other objects by class."""
    if value is None or isinstance(value, (bool, int, float, str, np.integer, np.floating,
                                           np.bool_)):
        return jsonable(value)
    if isinstance(value, (np.ndarray, mx.array)):
        return {"array": list(value.shape), "dtype": str(value.dtype)}
    if callable(getattr(value, "describe", None)) and not isinstance(value, type):
        return _described(value, {"class": type(value).__name__})
    if callable(value):                                  # before dict: an nn.Module is one
        return _name_of(value)
    if isinstance(value, dict):
        return {str(k): summary(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [summary(v) for v in value] if len(value) <= 16 else {"list": len(value)}
    meta = getattr(value, "metadata", None)
    if isinstance(meta, dict) and meta:                  # a saved-able result
        return {"class": type(value).__name__, "kind": meta.get("kind"),
                "created_utc": meta.get("environment", {}).get("created_utc"),
                "seed": meta.get("resolved", meta.get("arguments", {})).get("seed")}
    return {"class": type(value).__name__}


def record(kind: str, arguments: dict, target=None, skip=("target",)) -> dict:
    """The metadata block of a result made by ``kind`` (a function name)
    with ``arguments`` (its ``locals()`` at entry)."""
    return {"format": FORMAT, "kind": kind, "environment": environment(),
            "arguments": {k: summary(v) for k, v in arguments.items() if k not in skip},
            "target": describe_target(target) if target is not None else None}
