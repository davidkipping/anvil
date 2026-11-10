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


# -- run() inputs: NumPy u0, the kernel's own compile_step, targets that
# -- leave MLX (anvil-gp's note of 2026-11-06) -----------------------------

def test_run_accepts_a_numpy_u0():
    """run() used to call u0.astype(mx.float32) on whatever it was given, so a
    NumPy start died with "Cannot interpret 'mlx.core.float32' as a data
    type" -- while EnsembleSampler.run_mcmc took NumPy happily."""
    target, u0 = _setup()
    u0_np = np.array(u0, dtype=np.float64)
    kw = dict(n_warmup=5, n_samples=3, seed=1)
    a = anvil.run(RandomWalkMetropolis(target), target, u0, **kw).get_chain()
    b = anvil.run(RandomWalkMetropolis(target), target, u0_np, **kw).get_chain()
    c = anvil.run(RandomWalkMetropolis(target), target, u0_np.tolist(), **kw).get_chain()
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(a, c)


def test_run_honours_the_kernels_own_compile_step():
    """run() wrote its own default of True over the attribute, so a user who
    set kernel.compile_step = False was silently compiled anyway."""
    target, u0 = _setup()
    kw = dict(n_warmup=5, n_samples=3, seed=1)
    e = EnsembleKernel(target, seed=3)
    e.compile_step = False
    anvil.run(e, target, u0, **kw)
    assert e.compile_step is False and all(not c.compiled for c in e._compiled)
    # an explicit argument still wins, either way
    anvil.run(e, target, u0, compile_step=True, **kw)
    assert e.compile_step is True and all(c.compiled for c in e._compiled)
    e2 = EnsembleKernel(target, seed=3)
    anvil.run(e2, target, u0, compile_step=False, **kw)
    assert e2.compile_step is False
    # the untouched default is still compiled
    e3 = EnsembleKernel(target, seed=3)
    anvil.run(e3, target, u0, **kw)
    assert e3.compile_step is True and all(c.compiled for c in e3._compiled)


def _leaves_mlx():
    from anvil import ParamSpec, Transform, TransformedLogDensity

    def fn(v):          # a log density computed on the host
        return mx.array(-0.5 * np.sum(np.array(v) ** 2, axis=-1))
    tr = Transform([ParamSpec(f"p{i}", -5.0, 5.0) for i in range(3)])
    return TransformedLogDensity(fn, tr)


def test_a_target_that_leaves_mlx_gets_the_compile_step_hint():
    """MLX's own error ("[eval] Attempting to eval an array during function
    transformations") names neither anvil nor the remedy."""
    target = _leaves_mlx()
    u0 = mx.array(np.random.default_rng(1).standard_normal((16, 3)).astype(np.float32))
    kw = dict(n_warmup=3, n_samples=2, seed=1)
    for make in (lambda: RandomWalkMetropolis(target),
                 lambda: EnsembleKernel(target, seed=3)):
        with pytest.raises(ValueError, match="compile_step=False") as info:
            anvil.run(make(), target, u0, **kw)
        assert C.is_eval_under_compile(info.value.__cause__)
        # ... and the remedy works
        anvil.run(make(), target, u0, compile_step=False, **kw)
    # a fusion refusal is still a refusal, not this
    assert not C.is_eval_under_compile(REFUSAL)


# -- from the three-part review of 0.12.0 (shipped as 0.12.1) ---------------

def test_u0_errors_name_u0():
    target, u0 = _setup()
    k = lambda: RandomWalkMetropolis(target)
    with pytest.raises(ValueError, match=r"u0 must have shape \(n_chains, dim\)"):
        anvil.run(k(), target, np.zeros(3), n_samples=2)
    with pytest.raises(ValueError, match="u0 has 4 columns but the target has 3"):
        anvil.run(k(), target, np.zeros((8, 4)), n_samples=2)
    with pytest.raises(TypeError, match="u0 must be"):
        anvil.run(k(), target, [[1.0, 2.0, 3.0], [1.0, 2.0]], n_samples=2)
    with pytest.raises(TypeError, match="u0 must be"):
        anvil.run(k(), target, "hello", n_samples=2)
    # a float64 mx.array (CPU-only dtype) is converted like NumPy float64
    with mx.stream(mx.cpu):
        u64 = mx.array(np.array(u0, dtype=np.float64), dtype=mx.float64)
    kw = dict(n_warmup=5, n_samples=3, seed=1)
    np.testing.assert_array_equal(anvil.run(k(), target, u64, **kw).get_chain(),
                                  anvil.run(k(), target, u0, **kw).get_chain())


def test_compile_step_must_be_a_bool_or_none():
    target, u0 = _setup()
    with pytest.raises(TypeError, match="compile_step must be True, False or None"):
        anvil.run(RandomWalkMetropolis(target), target, u0, n_samples=2, compile_step="no")
    k = RandomWalkMetropolis(target)
    k.compile_step = None
    with pytest.raises(TypeError, match="kernel.compile_step must be True or False"):
        anvil.run(k, target, u0, n_samples=2)


def test_the_emcee_wrappers_and_bridge_evidence_take_compile_step():
    """The hint names compile_step=False; these entry points had no such
    argument, and bridge_evidence called itself the optimizer."""
    from anvil.emcee_api import EnsembleSampler
    target = _leaves_mlx()
    p0 = np.random.default_rng(1).standard_normal((16, 3))

    def fn(u):          # leaves MLX
        return mx.array(-0.5 * np.sum(np.array(u) ** 2, axis=-1))
    s = EnsembleSampler(16, 3, fn)
    with pytest.raises(ValueError, match="compile_step=False"):
        s.run_mcmc(p0, 3)
    assert s.run_mcmc(p0, 3, compile_step=False).get_chain().shape == (3, 16, 3)
    r = anvil.run(EnsembleKernel(target, seed=3), target, p0, n_warmup=0,
                  n_samples=60, compile_step=False)
    draws = r.get_chain(flat=True)
    with pytest.raises(ValueError, match="the bridge-sampling target"):
        anvil.bridge_evidence(target, draws)
    ev = anvil.bridge_evidence(target, draws, compile_step=False)
    assert np.isfinite(ev.log_z)


def test_the_hint_does_not_send_gradient_users_to_compile_step_false():
    target = _leaves_mlx()
    u0 = mx.array(np.random.default_rng(1).standard_normal((16, 3)).astype(np.float32))
    with pytest.raises(ValueError) as info:
        anvil.run(ChEESHMC(target), target, u0, n_warmup=3, n_samples=2, seed=1)
    msg = str(info.value)
    assert "ChEES" in msg and "no gradient flows" in msg
    # a user's own ValueError that merely starts "[eval]" is not relabelled
    assert not C.is_eval_under_compile(ValueError("[eval] my own message"))
