"""MLX can refuse to compile a graph; anvil must not die of it.

The refusal is a RuntimeError prefixed "[compile]", raised at evaluation
rather than at the compiled call, and raised again on every call. A real
provoking target (a six-generator collapsed limb-darkening model) is too
large to ship as a test, so the refusal is injected through
anvil.compile._evaluate and the end-to-end behaviour is asserted against
the one thing the fallback promises: the same draws as compile_step=False.
"""

import warnings

import mlx.core as mx
import numpy as np
import pytest

import anvil
import anvil.compile as C
from anvil.kernels.chees import ChEESHMC
from anvil.kernels.ensemble import EnsembleKernel
from anvil.kernels.metropolis import RandomWalkMetropolis
from anvil.targets import correlated_gaussian

REFUSAL = RuntimeError("[compile] Too many inputs/outputs fused in the Metal "
                       "Compiled primitive which exhausted the available "
                       "argument buffers for the kernel.")


def _setup(dim=3, n=64):
    target, mu, _ = correlated_gaussian(dim, rho=0.5, seed=0)
    u0 = mx.array((mu + np.random.default_rng(1).standard_normal((n, dim)))
                  .astype(np.float32))
    return target, u0


def _kernels(target):
    return {"chees": lambda: ChEESHMC(target),
            "ensemble": lambda: EnsembleKernel(target, seed=3),
            "rwm": lambda: RandomWalkMetropolis(target)}


class _RefuseFirst:
    """Stand-in for mx.eval that refuses the first evaluation it sees."""

    def __init__(self):
        self.calls = 0

    def __call__(self, *args):
        self.calls += 1
        if self.calls == 1:
            raise REFUSAL
        return mx.eval(*args)


# -- the wrapper itself ----------------------------------------------------

def test_wrapper_falls_back_once_and_keeps_the_answer(monkeypatch):
    f = lambda x: x * 2 + 1
    monkeypatch.setattr(C, "_evaluate", _RefuseFirst())
    w = C.CompiledOrEager(f, name="a test function")
    with pytest.warns(RuntimeWarning, match="MLX refused to compile a test function"):
        out = w(mx.array([1.0, 2.0]))
    np.testing.assert_array_equal(np.array(out), [3.0, 5.0])
    assert w.compiled is False
    with warnings.catch_warnings():
        warnings.simplefilter("error")           # a second call must be silent
        np.testing.assert_array_equal(np.array(w(mx.array([0.0]))), [1.0])


def test_wrapper_lets_other_errors_through(monkeypatch):
    def boom(*a):
        raise RuntimeError("something else entirely")
    monkeypatch.setattr(C, "_evaluate", boom)
    w = C.CompiledOrEager(lambda x: x, name="x")
    with pytest.raises(RuntimeError, match="something else"):
        w(mx.array([1.0]))
    assert w.compiled is True                    # not a refusal: no fallback


def test_disabled_wrapper_never_compiles():
    w = C.CompiledOrEager(lambda x: x + 1, name="x", enabled=False)
    assert w.compiled is False and w._compiled_fn is None
    assert float(w(mx.array(1.0)).item()) == 2.0


# -- compile_step now reaches every kernel ---------------------------------

def test_compile_step_false_reaches_the_self_compiling_kernels():
    """Before this, the flag only wrapped kernels the engine compiles, so
    compile_step=False silently did nothing for ChEES and the ensemble --
    the workaround a user would reach for first was a no-op."""
    target, u0 = _setup()
    k = ChEESHMC(target)
    anvil.run(k, target, u0, n_warmup=5, n_samples=3, seed=1, compile_step=False)
    assert k._leapfrog_c.compiled is False and k._finish_c.compiled is False
    e = EnsembleKernel(target, seed=3)
    anvil.run(e, target, u0, n_warmup=5, n_samples=3, seed=1, compile_step=False)
    assert all(not c.compiled for c in e._compiled)
    # and the default still compiles
    k2 = ChEESHMC(target)
    anvil.run(k2, target, u0, n_warmup=5, n_samples=3, seed=1)
    assert k2._leapfrog_c.compiled is True


def test_chees_compiled_and_uncompiled_draws_are_bit_identical():
    """ChEES draws its momentum OUTSIDE the compiled leapfrog, so compiled
    and eager runs agree exactly. That is the strong form of the fallback's
    promise, where it holds."""
    target, u0 = _setup()
    kw = dict(n_warmup=30, n_samples=20, seed=4)
    a = anvil.run(ChEESHMC(target), target, u0, **kw)
    b = anvil.run(ChEESHMC(target), target, u0, compile_step=False, **kw)
    np.testing.assert_array_equal(a.get_chain(), b.get_chain())
    np.testing.assert_array_equal(a.get_log_prob(), b.get_log_prob())
    assert a.extras["compile_refused"] is False
    assert b.extras["compile_refused"] is False


@pytest.mark.parametrize("kind", ["ensemble", "rwm"])
def test_kernels_that_draw_inside_the_graph_agree_to_round_off(kind):
    """mx.random.normal with the SAME key returns values 6e-6 apart compiled
    vs eager (measured), so kernels drawing inside their compiled step cannot
    be bit-identical to their uncompiled selves. What the fallback promises
    for them is the same distribution: one step agrees to round-off with no
    accept decision flipped, and a run agrees on the posterior."""
    target, mu, cov = correlated_gaussian(3, rho=0.5, seed=0)
    u0 = mx.array((mu + np.random.default_rng(1).standard_normal((256, 3)))
                  .astype(np.float32))
    make = _kernels(target)[kind]
    k = make()
    st = k.init(mx.random.key(0), u0, target)
    params = k.make_params(k.init_adapt(st), warmup=True)
    fn = k.step if kind == "rwm" else k._make_step(k.moves[0][0])
    key = mx.random.key(11)
    e_state, e_info = fn(key, st, params)
    c_state, c_info = C.CompiledOrEager(fn, name="x")(key, st, params)
    mx.eval(e_state, e_info, c_state, c_info)
    assert np.abs(np.array(e_state["u"]) - np.array(c_state["u"])).max() < 1e-4
    np.testing.assert_array_equal(np.array(e_info["accepted"]),
                                  np.array(c_info["accepted"]))

    kw = dict(n_warmup=200, n_samples=300, seed=4)
    a = anvil.run(make(), target, u0, **kw).get_chain(flat=True)
    b = anvil.run(make(), target, u0, compile_step=False, **kw).get_chain(flat=True)
    sd = np.sqrt(np.diag(cov))
    assert np.abs(a.mean(0) - b.mean(0)).max() < 0.1 * sd.max()
    np.testing.assert_allclose(a.std(0), b.std(0), rtol=0.1)


# -- the fallback, end to end ----------------------------------------------

@pytest.mark.parametrize("kind", ["chees", "ensemble", "rwm"])
def test_a_refused_graph_falls_back_and_matches_uncompiled(kind, monkeypatch):
    target, u0 = _setup()
    make = _kernels(target)[kind]
    kw = dict(n_warmup=30, n_samples=20, seed=5)
    expected = anvil.run(make(), target, u0, compile_step=False, **kw)

    monkeypatch.setattr(C, "_evaluate", _RefuseFirst())
    k = make()
    with pytest.warns(RuntimeWarning, match="MLX refused to compile"):
        got = anvil.run(k, target, u0, **kw)
    assert got.extras["compile_refused"] is True
    assert k.compile_refused is True
    np.testing.assert_array_equal(got.get_chain(), expected.get_chain())
    np.testing.assert_array_equal(got.get_log_prob(), expected.get_log_prob())


def test_a_refusal_is_remembered_across_runs(monkeypatch):
    """A later run or resume on the same kernel must not retrace the graph
    MLX already rejected -- that costs a failed trace and a second warning."""
    target, u0 = _setup()
    monkeypatch.setattr(C, "_evaluate", _RefuseFirst())
    k = ChEESHMC(target)
    with pytest.warns(RuntimeWarning):
        res = anvil.run(k, target, u0, n_warmup=10, n_samples=5, seed=6)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        again = anvil.run(k, target, resume=res, n_samples=5)
        fresh = anvil.run(k, target, u0, n_warmup=10, n_samples=5, seed=6)
    assert again.extras["compile_refused"] and fresh.extras["compile_refused"]
    assert k._leapfrog_c.compiled is False
