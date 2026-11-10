"""anvil.joint: several TransformedLogDensity members, one parameter vector,
some columns shared. Checked against the same model written by hand."""

import math
import warnings

import mlx.core as mx
import numpy as np
import pytest

import anvil
from anvil import ParamSpec, Transform, TransformedLogDensity, joint
from anvil.priors import Gaussian


def _member(x, y):
    """Straight line y = a + b x with its own noise scale ``c``: ``a`` and
    ``b`` are shared between datasets, ``c`` is per dataset."""
    x32, y32 = mx.array(x.astype(np.float32)), mx.array(y.astype(np.float32))
    x64, y64 = mx.array(x), mx.array(y)
    n = x.size

    def ll(v, xx, yy):
        r = (yy[None, :] - (v[:, 0:1] + v[:, 1:2] * xx[None, :])) / v[:, 2:3]
        return -0.5 * mx.sum(r * r, axis=-1) - n * mx.log(v[:, 2])

    specs = [ParamSpec("a", -1.0, 1.0), ParamSpec("b", 0.0, 3.0, prior=Gaussian(1.0, 1.0)),
             ParamSpec("c", 0.01, 0.5)]
    return TransformedLogDensity(lambda v: ll(v, x32, y32), Transform(specs),
                                 model_log_prob_hi=lambda v: ll(v, x64, y64),
                                 model_log_norm=-0.5 * n * math.log(2 * math.pi))


def _data(seed, n=200, c=0.05):
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(-1, 1, n))
    return x, 0.3 + 1.2 * x + c * rng.standard_normal(n)


@pytest.fixture
def two():
    x1, y1 = _data(0, c=0.05)
    x2, y2 = _data(1, c=0.08)
    t1, t2 = _member(x1, y1), _member(x2, y2)
    jt = joint({"d1": t1, "d2": t2}, shared={"a": ["d1", "d2"], "b": ["d1", "d2"]})
    return t1, t2, jt


def test_layout_and_names(two):
    t1, t2, jt = two
    assert jt.names == ["a", "b", "d1.c", "d2.c"]
    assert jt.member_columns("d1").tolist() == [0, 1, 2]
    assert jt.member_columns("d2").tolist() == [0, 1, 3]
    assert isinstance(jt, TransformedLogDensity) and jt.dim == 4
    assert jt.has_hi and jt.supports_grad and jt.is_proper
    assert jt.model_log_norm == t1.model_log_norm + t2.model_log_norm


def test_joint_equals_the_hand_written_model(two):
    t1, t2, jt = two
    u = mx.array(np.random.default_rng(3).standard_normal((32, 4)).astype(np.float32))
    v = jt.transform.to_model(u)
    # model-space sum of the members, each reading its columns
    want = (t1.model_log_prob(v[:, [0, 1, 2]]) + t2.model_log_prob(v[:, [0, 1, 3]]))
    np.testing.assert_array_equal(np.array(jt.model_log_prob(v)), np.array(want))
    # the joint log_prob applies ONE Jacobian and ONE set of priors
    tr = jt.transform
    want = want + tr.log_det_jac(u) + tr.log_prior(v)
    np.testing.assert_allclose(np.array(jt.log_prob(u)), np.array(want), rtol=1e-6, atol=1e-4)
    # summing the members' own log_prob would count a and b twice
    double = np.array(t1.log_prob(u[:, [0, 1, 2]]) + t2.log_prob(u[:, [0, 1, 3]]))
    assert not np.allclose(double, np.array(jt.log_prob(u)), atol=1e-3)


def test_shared_columns_get_the_summed_gradient(two):
    t1, t2, jt = two
    u = mx.array(np.random.default_rng(4).standard_normal((8, 4)).astype(np.float32))
    _, g = jt.log_prob_and_grad(u)
    g = np.array(g)
    # finite differences on the float64 path
    u64 = np.array(u, dtype=np.float64)
    h = 1e-4
    for k in range(4):
        e = np.zeros(4); e[k] = h
        fd = (np.array(jt.log_prob_hi(mx.array((u64 + e).astype(np.float32)))) -
              np.array(jt.log_prob_hi(mx.array((u64 - e).astype(np.float32))))) / (2 * h)
        np.testing.assert_allclose(g[:, k], fd, rtol=2e-2, atol=2e-2)


def test_float64_path_and_constant(two):
    t1, t2, jt = two
    u = mx.array(0.1 * np.random.default_rng(5).standard_normal((16, 4)).astype(np.float32))
    lo, hi = np.array(jt.log_prob(u)), np.array(jt.log_prob_hi(u))
    assert np.max(np.abs(lo - hi)) < 1e-2
    assert jt.log_norm_const == pytest.approx(jt.transform.log_prior_norm + jt.model_log_norm)
    # a member without a float64 path removes the joint one
    t3 = TransformedLogDensity(t1.model_log_prob, t1.transform)
    j2 = joint({"d1": t3, "d2": t2}, shared={"a": ["d1", "d2"], "b": ["d1", "d2"]})
    assert not j2.has_hi
    with pytest.raises(NotImplementedError):
        j2.log_prob_hi(u)


def test_a_failed_member_fails_the_joint_point_and_keeps_its_slope():
    # member 2 fails (sloped sentinel below -1e29) where its own column is
    # negative; the joint value is then below the floor, and the shared
    # column's gradient still carries member 2's inward slope
    def fail_fn(v):
        bad = v[:, 0] < 0
        return mx.where(bad, -1e30 * (1.0 + mx.stop_gradient(mx.abs(v[:, 0]))) + 1e29 * v[:, 0],
                        -0.5 * mx.sum(v * v, axis=-1))
    t1 = TransformedLogDensity(lambda v: -0.5 * mx.sum(v * v, axis=-1),
                               Transform([ParamSpec("s", prior=Gaussian(0.0, 1.0)),
                                          ParamSpec("p", prior=Gaussian(0.0, 1.0))]))
    t2 = TransformedLogDensity(fail_fn, Transform([ParamSpec("s", prior=Gaussian(0.0, 1.0))]))
    jt = joint({"m1": t1, "m2": t2}, shared={"s": ["m1", "m2"]})
    u = mx.array(np.array([[-0.5, 0.2], [0.5, 0.2]], np.float32))
    lp, g = jt.log_prob_and_grad(u)
    lp, g = np.array(lp), np.array(g)
    assert lp[0] <= -1e29 and np.isfinite(lp[1])
    assert g[0, 0] > 1e28          # pushed back toward s >= 0 along the shared column


def test_shared_specs_must_agree_and_names_must_exist(two):
    t1, t2, _ = two
    other = TransformedLogDensity(t2.model_log_prob, Transform([
        ParamSpec("a", -2.0, 2.0), t2.transform.specs[1], t2.transform.specs[2]]))
    with pytest.raises(ValueError, match="must agree"):
        joint({"d1": t1, "d2": other}, shared={"a": ["d1", "d2"]})
    with pytest.raises(ValueError, match="no parameter"):
        joint({"d1": t1, "d2": t2}, shared={"a": {"d1": "a", "d2": "zz"}})
    with pytest.raises(ValueError, match="no member"):
        joint({"d1": t1}, shared={"a": ["d1", "d9"]})
    # different member-side names for one shared parameter are allowed
    renamed = TransformedLogDensity(t2.model_log_prob, Transform([
        ParamSpec("alpha", -1.0, 1.0), t2.transform.specs[1], t2.transform.specs[2]]),
        model_log_prob_hi=t2.model_log_prob_hi, model_log_norm=t2.model_log_norm)
    j = joint({"d1": t1, "d2": renamed}, shared={"a": {"d1": "a", "d2": "alpha"}, "b": ["d1", "d2"]})
    assert j.names == ["a", "b", "d1.c", "d2.c"] and j.shared["a"] == {"d1": "a", "d2": "alpha"}


def test_transform_rejects_duplicate_names():
    with pytest.raises(ValueError, match="duplicate"):
        Transform([ParamSpec("x", 0.0, 1.0), ParamSpec("x", 0.0, 2.0)])


def test_tools_take_a_joint_target(two):
    t1, t2, jt = two
    # optimize: the MLE pools both datasets for a and b
    r = anvil.optimize(jt, n_starts=32, seed=0)
    assert r.converged and abs(r.model[0] - 0.3) < 0.02 and abs(r.model[1] - 1.2) < 0.03
    assert abs(r.model[2] - 0.05) < 0.01 and abs(r.model[3] - 0.08) < 0.015
    # run: ChEES on the joint target, posterior of the shared slope
    res = anvil.run(anvil.ChEESHMC(jt), jt, r.ball(64), n_warmup=150, n_samples=100, seed=1)
    v = jt.transform.model_np(res.get_chain(flat=True))
    assert abs(v[:, 1].mean() - 1.2) < 0.03
    # smc: finite ln Z with the summed constant, no stalls
    s = anvil.smc(jt, n_particles=1024, n_probe=4, seed=0)
    assert np.isfinite(s.log_z) and s.stalled == 0 and s.precision is not None
    # precision tools
    rep = anvil.validate_precision(jt, mx.array(r.ball(16)))
    assert rep is not None


# -- from the three-part review of 0.12.0 (shipped as 0.12.1) ---------------

def _quad(names, centres, widths, lo=-10.0, hi=10.0):
    """Independent Gaussians in each parameter, with a float64 twin."""
    c32, w32 = mx.array(np.float32(centres)), mx.array(np.float32(widths))
    c64 = mx.array(np.float64(centres), dtype=mx.float64)
    w64 = mx.array(np.float64(widths), dtype=mx.float64)
    return TransformedLogDensity(
        lambda v: -0.5 * mx.sum(((v - c32) / w32) ** 2, axis=-1),
        Transform([ParamSpec(n, lo, hi) for n in names]),
        model_log_prob_hi=lambda v: -0.5 * mx.sum(((v - c64) / w64) ** 2, axis=-1))


def test_permuted_members_against_float64_by_hand():
    """Members in different parameter orders, shared in an order that is
    neither's, one under a different name: a layout that mixed up joint
    order and member order would show against the hand calculation."""
    m1 = _quad(["c", "a", "b"], [0.5, -1.0, 2.0], [0.3, 0.7, 1.1])
    m2 = _quad(["beta", "k", "a"], [1.5, 0.2, -0.4], [0.9, 0.4, 0.6])
    jt = joint({"m1": m1, "m2": m2},
               shared={"b": {"m1": "b", "m2": "beta"}, "a": ["m1", "m2"]})
    assert jt.names == ["b", "a", "m1.c", "m2.k"]
    assert jt.member_columns("m1").tolist() == [2, 1, 0]
    assert jt.member_columns("m2").tolist() == [0, 3, 1]
    u = mx.array(np.random.default_rng(7).standard_normal((16, 4)).astype(np.float32))
    hi = jt.log_prob_hi(u)
    assert hi.dtype == mx.float64
    u32 = np.array(u, dtype=np.float64)
    v = jt.transform.model_np(u32)
    b, a, c, k = v.T
    want = (-0.5 * (((c - 0.5) / 0.3) ** 2 + ((a + 1.0) / 0.7) ** 2 + ((b - 2.0) / 1.1) ** 2)
            - 0.5 * (((b - 1.5) / 0.9) ** 2 + ((k - 0.2) / 0.4) ** 2 + ((a + 0.4) / 0.6) ** 2))
    want = want + jt.transform.log_det_jac_np(u32) + jt.transform.log_prior_np(v)
    np.testing.assert_allclose(np.array(hi), want, rtol=1e-12, atol=1e-10)
    np.testing.assert_allclose(np.array(jt.log_prob(u)),
                               want, rtol=1e-4, atol=1e-2)
    # the public float64 function works outside a CPU stream too
    v64 = mx.array(v, dtype=mx.float64)
    assert jt.model_log_prob_hi(v64).dtype == mx.float64


def test_smc_log_z_of_a_joint_target_is_the_analytic_one():
    """Two Gaussian likelihoods sharing x, one with its own y, uniform on
    a box of side 20: ln Z is known. A prior counted twice would be
    ln(20) = 3.0 low."""
    s1, s2, sy, d = 0.5, 0.8, 0.3, 0.6
    m1 = _quad(["x", "y"], [0.0, 1.0], [s1, sy])
    m2 = _quad(["x"], [d], [s2])
    jt = joint({"m1": m1, "m2": m2}, shared={"x": ["m1", "m2"]})
    want = (math.log(math.sqrt(2 * math.pi) * s1 * s2 / math.hypot(s1, s2))
            - 0.5 * d * d / (s1 * s1 + s2 * s2)
            + math.log(math.sqrt(2 * math.pi) * sy) - 2 * math.log(20.0))
    s = anvil.smc(jt, n_particles=4096, seed=0)
    assert abs(s.log_z - want) < 0.1


def test_a_member_parameter_cannot_be_shared_twice():
    """The later entry used to overwrite the column: the first sharing was
    dropped while jt.shared still reported it."""
    a, b = _quad(["x", "y"], [0, 0], [1, 1]), _quad(["x", "z"], [0, 0], [1, 1])
    with pytest.raises(ValueError, match="already shared as 'x'"):
        joint({"a": a, "b": b}, shared={"x": ["a", "b"], "x2": {"a": "x", "b": "z"}})


def test_the_joint_does_not_follow_the_callers_dict():
    a, b = _quad(["x", "y"], [0, 0], [1, 1]), _quad(["x", "z"], [0, 0], [1, 1])
    md = {"a": a, "b": b}
    jt = joint(md, shared={"x": ["a", "b"]})
    u = mx.array(np.random.default_rng(2).standard_normal((4, 3)).astype(np.float32))
    before = np.array(jt.log_prob(u))
    md["b"] = _quad(["x", "z"], [5, 5], [0.1, 0.1])
    md["c"] = _quad(["q"], [0], [1])
    np.testing.assert_array_equal(np.array(jt.log_prob(u)), before)
    assert list(jt.members) == ["a", "b"] and jt.members["b"] is b


def test_a_member_overriding_log_prob_is_refused():
    class Tempered(TransformedLogDensity):
        def log_prob(self, u):
            return 0.5 * super().log_prob(u)
    t = Tempered(lambda v: -0.5 * mx.sum(v * v, axis=-1),
                 Transform([ParamSpec("x", 0.0, 1.0)]))
    with pytest.raises(TypeError, match="overrides log_prob"):
        joint({"t": t})
    # a joint is itself a legitimate member
    a = _quad(["x", "y"], [0, 0], [1, 1])
    inner = joint({"a": a})
    assert joint({"j": inner, "b": _quad(["w"], [0], [1])}).names == ["j.a.x", "j.a.y", "b.w"]


def test_bad_arguments_get_messages_that_name_the_problem():
    a, b = _quad(["x", "y"], [0, 0], [1, 1]), _quad(["x", "z"], [0, 0], [1, 1])
    with pytest.raises(TypeError, match="members as a dict"):
        joint([a, b])
    with pytest.raises(TypeError, match="list of member names"):
        joint({"a": a, "b": b}, shared={"x": "ab"})
    with pytest.raises(TypeError, match="list of member names"):
        joint({"a": a, "b": b}, shared={"x": None})
    with pytest.raises(ValueError, match="no members listed"):
        joint({"a": a, "b": b}, shared={"x": []})
    with pytest.raises(TypeError, match="shared must be a dict"):
        joint({"a": a, "b": b}, shared=["x"])
    with pytest.raises(TypeError, match="member names must be strings"):
        joint({1: a, "b": b})
    # a generated name colliding with a shared one says where both came from
    with pytest.raises(ValueError, match=r"produced twice.*shared parameter 'a.y'.*member 'a' parameter 'y'"):
        joint({"a": a, "b": b}, shared={"a.y": {"b": "z"}})
    with pytest.raises(ValueError, match="loc must be finite"):
        ParamSpec("x", loc=float("nan"))


def test_shared_specs_agree_as_resolved():
    """None on a finite box is Uniform there; separately built priors with
    NaN placeholders are the same prior (dataclass == sees through those
    on some Python versions only)."""
    from anvil.priors import LogUniform, Uniform

    def one(prior, lo=0.1, hi=2.0):
        return TransformedLogDensity(lambda v: -0.5 * mx.sum(v * v, axis=-1),
                                     Transform([ParamSpec("x", lo, hi, prior=prior)]))
    assert joint({"a": one(None), "b": one(Uniform())}, shared={"x": ["a", "b"]}).dim == 1
    assert joint({"a": one(LogUniform()), "b": one(LogUniform())}, shared={"x": ["a", "b"]}).dim == 1
    assert joint({"a": one(None, 0, 2), "b": one(None, 0.0, 2.0)}, shared={"x": ["a", "b"]}).dim == 1
    with pytest.raises(ValueError, match="must agree"):
        joint({"a": one(Uniform()), "b": one(LogUniform())}, shared={"x": ["a", "b"]})
    with pytest.raises(ValueError, match="must agree"):
        joint({"a": one(Gaussian(1.0, 0.5)), "b": one(Gaussian(1.0, 0.6))}, shared={"x": ["a", "b"]})


def test_twelve_chunked_members_compile_as_one_graph():
    """A chained a + b + ... over the members let MLX fuse every member's
    tail into one Metal kernel, which it then refused from about 8
    fixed_point members of several chunks each -- members that compile
    alone. The run fell back to eager, 4x slower."""
    from anvil.precision import ChunkedGaussianLogLike, PrecisionPolicy

    def member(seed):
        rng = np.random.default_rng(seed)
        x = np.linspace(-1, 1, 7000)
        y = 0.3 + 1.2 * x + rng.normal(0, 1, x.size)

        def model(v, xx):
            return v[:, 0:1] + v[:, 1:2] * xx + v[:, 2:3] * mx.sin(v[:, 3:4] * xx)
        lik = ChunkedGaussianLogLike(model, x, y, np.ones_like(x),
                                     PrecisionPolicy(reduction="fixed_point", chunk_size=1000))
        return TransformedLogDensity(lik, Transform([ParamSpec(n, -5.0, 5.0) for n in "abcd"]))
    mem = {f"m{k}": member(k) for k in range(12)}
    jt = joint(mem, shared={"a": list(mem), "b": list(mem)})
    u0 = np.random.default_rng(1).normal(0, 0.1, (128, jt.dim))
    k = anvil.EnsembleKernel(jt, seed=1)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        anvil.run(k, jt, u0, n_warmup=0, n_samples=3, seed=1)
    assert not k.compile_refused
