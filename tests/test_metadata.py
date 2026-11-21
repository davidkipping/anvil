"""Saved results record how they were made (anvil.metadata): a run's file
holds its draws, every run() argument, the kernel's and target's settings
and the versions; a flow's file holds its training record, report and
training draws; evidence results hold their call's arguments. Every file is
plain .npz or JSON."""

import json
import math
import warnings

import mlx.core as mx
import numpy as np
import pytest

import anvil
from anvil import ParamSpec, Transform, TransformedLogDensity
from anvil.flow import Flow
from anvil.metadata import jsonable
from anvil.priors import Gaussian

SY = 0.1


def _banana_ll(v):
    return -0.5 * ((v[:, 1] - v[:, 0] ** 2) / SY) ** 2


def _banana():
    specs = [ParamSpec("x", prior=Gaussian(1.0, 1.0)),
             ParamSpec("y", prior=Gaussian(0.0, 10.0))]
    return TransformedLogDensity(_banana_ll, Transform(specs))


def _start(n=32, seed=0):
    rng = np.random.default_rng(seed)
    return np.stack([1 + 0.1 * rng.standard_normal(n), 1 + 0.1 * rng.standard_normal(n)],
                    axis=1).astype(np.float32)


@pytest.fixture(scope="module")
def small_flow():
    rng = np.random.default_rng(0)
    x = 1 + rng.standard_normal(20_000)
    d = np.stack([x, x ** 2 + SY * rng.standard_normal(20_000)], axis=1).astype(np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return anvil.flow.train(_banana(), draws=d, rounds=1, n_chains=64, round_samples=200,
                                n_layers=4, hidden=32, fit_kwargs=dict(max_steps=300))


def test_a_run_saves_its_draws_and_how_it_was_made(tmp_path):
    t = _banana()
    k = anvil.EnsembleKernel(t, moves=[(anvil.StretchMove(a=2.5), 3.0), (anvil.DEMove(), 1.0)])
    res = anvil.run(k, t, _start(), n_warmup=20, n_samples=30, thin=2, seed=7)
    res.save(tmp_path / "run")                            # no extension: written as given
    back = anvil.load_results(tmp_path / "run")
    np.testing.assert_array_equal(back.get_chain(), res.get_chain())
    np.testing.assert_array_equal(back.get_log_prob(flat=True), res.get_log_prob(flat=True))
    assert back.names == ["x", "y"]
    m = back.metadata
    assert m["kind"] == "anvil.run" and m["environment"]["anvil"] == anvil.__version__
    assert m["environment"]["mlx"] and m["environment"]["created_utc"]
    assert m["arguments"]["seed"] == 7 and m["arguments"]["thin"] == 2
    assert m["resolved"]["n_chains"] == 32 and m["resolved"]["iterations_consumed"] == 80
    moves = m["kernel"]["moves"]
    assert [mv["class"] for mv in moves] == ["StretchMove", "DEMove"]
    assert moves[0]["a"] == 2.5 and moves[0]["weight"] == 0.75
    p = m["target"]["parameters"]
    assert p[0]["prior"] == {"class": "Gaussian", "mu": 1.0, "sigma": 1.0}
    assert p[0]["lo"] == -math.inf and m["target"]["model_log_prob"].endswith("_banana_ll")
    # plain .npz plus JSON: readable without anvil
    with np.load(tmp_path / "run", allow_pickle=False) as z:
        assert json.loads(str(z["metadata"]))["arguments"]["seed"] == 7
        assert z["chain"].shape == (30, 32, 2)


def test_chees_settings_and_a_resume_are_recorded():
    t = _banana()
    first = anvil.run(anvil.ChEESHMC(t, target_accept=0.7), t, _start(), n_warmup=30,
                      n_samples=10, seed=1)
    kern = first.metadata["kernel"]
    assert kern["class"] == "ChEESHMC" and kern["target_accept"] == 0.7
    more = anvil.run(anvil.ChEESHMC(t, target_accept=0.7), t, resume=first, n_samples=10)
    assert more.metadata["arguments"]["resume"] is True
    assert more.metadata["resolved"]["resumed_from_iteration"] == 40


def test_a_flow_saves_its_training_report_and_draws(small_flow, tmp_path):
    f = small_flow
    f.save(tmp_path / "f.npz")
    g = Flow.load(tmp_path / "f.npz")
    assert g.fingerprint == f.fingerprint
    np.testing.assert_array_equal(g.pool, f.pool)
    assert g.report.gain_nats == f.report.gain_nats and g.report.warnings == f.report.warnings
    assert g.training["function"] == "anvil.flow.train"
    assert g.training["arguments"]["rounds"] == 1
    assert g.training["final_fit"]["arguments"]["max_steps"] == 300
    assert g.training["target"]["parameters"][1]["name"] == "y"
    f.save(tmp_path / "nopool.npz", pool=False)
    assert Flow.load(tmp_path / "nopool.npz").pool is None
    # a file from before 0.14.0, without metadata, still loads
    with np.load(tmp_path / "f.npz") as z:
        old = {k: z[k] for k in z.files if k not in ("metadata", "pool")}
    np.savez(tmp_path / "old.npz", **old)
    h = Flow.load(tmp_path / "old.npz")
    assert h.fingerprint == f.fingerprint and h.report is None and h.training is None


def test_a_run_in_flow_coordinates_names_its_flow(small_flow, tmp_path):
    f = small_flow
    ft = f.wrap(_banana())
    res = anvil.run(anvil.EnsembleKernel(ft), ft, f.start(32), n_warmup=10, n_samples=10)
    res.save(tmp_path / "run.npz")
    back = anvil.load_results(tmp_path / "run.npz")
    tgt = back.metadata["target"]
    assert tgt["class"] == "FlowTarget" and tgt["flow"]["fingerprint"] == f.fingerprint
    assert tgt["flow"]["trained_by"]["function"] == "anvil.flow.train"
    assert back.metadata["target_tag"] == f"flow:{f.fingerprint}"
    assert back.names == ["x", "y"]


def test_evidence_results_record_their_call(small_flow, tmp_path):
    t = _banana()
    r = anvil.smc(t, 1024, seed=2)
    assert r.metadata["kind"] == "anvil.smc" and r.metadata["arguments"]["n_particles"] == 1024
    r.save(tmp_path / "smc.npz")
    back = anvil.SMCResult.load(tmp_path / "smc.npz")
    assert back.log_z == r.log_z and math.isnan(back.log_z_se)
    np.testing.assert_array_equal(back.u, r.u)
    from anvil.metadata import dumps, loads
    assert back.metadata == loads(dumps(r.metadata))
    ev = anvil.flow_evidence(t, small_flow, n=4096, seed=3)
    assert ev.metadata["arguments"]["flow"]["fingerprint"] == small_flow.fingerprint
    ev.save(tmp_path / "ev.json")
    again = anvil.EvidenceResult.load(tmp_path / "ev.json")
    assert again.log_z == ev.log_z and again.metadata["arguments"]["n"] == 4096
    # draws handed in are summarised, not copied into the metadata
    b = anvil.bridge_evidence(t, r)
    assert b.metadata["arguments"]["draws"]["class"] == "SMCResult"
    assert b.metadata["arguments"]["draws"]["kind"] == "anvil.smc"


def test_jsonable_keeps_non_finite_values_and_summarises_arrays():
    from anvil.metadata import summary
    assert jsonable({"a": math.inf, "b": float("nan"), "c": np.float32(1.5)}) == \
        {"a": "inf", "b": "nan", "c": 1.5}
    assert summary(np.zeros((1000, 3))) == {"array": [1000, 3], "dtype": "float64"}
    assert summary(mx.zeros((4,))) == {"array": [4], "dtype": "mlx.core.float32"}


def test_load_results_refuses_another_file(tmp_path):
    np.savez(tmp_path / "x.npz", a=np.zeros(2))
    with pytest.raises(ValueError, match="not a run"):
        anvil.load_results(tmp_path / "x.npz")


def test_a_broken_describe_does_not_fail_a_run():
    class Odd(anvil.FunctionLogDensity):
        def describe(self):
            raise RuntimeError("no")
    t = Odd(lambda u: -0.5 * mx.sum(u * u, axis=1), 2)
    res = anvil.run(anvil.EnsembleKernel(t), t, _start(), n_warmup=5, n_samples=5)
    assert res.metadata["target"]["describe_error"] == "RuntimeError: no"


# -- review fixes (0.14.1) -----------------------------------------------------------

def test_a_callable_model_holding_data_is_recorded_by_name(tmp_path):
    # a dataclass with __call__ (or an nn.Module, a dict) used to have its
    # fields -- its data -- written into the metadata in full
    import dataclasses

    @dataclasses.dataclass
    class Model:
        y: mx.array

        def __call__(self, v):
            return -0.5 * mx.sum(v * v, axis=1) + 0.0 * mx.sum(self.y)
    model = Model(mx.zeros((100_000,)))
    specs = [ParamSpec("a", prior=Gaussian(0.0, 1.0)), ParamSpec("b", prior=Gaussian(0.0, 1.0))]
    t = TransformedLogDensity(model, Transform(specs))
    res = anvil.run(anvil.EnsembleKernel(t), t, _start(), n_warmup=2, n_samples=2)
    assert isinstance(res.metadata["target"]["model_log_prob"], str)
    res.save(tmp_path / "r.npz")
    assert (tmp_path / "r.npz").stat().st_size < 200_000
    assert jsonable(np.zeros(1000)) == {"array": [1000], "dtype": "float64"}


def test_non_finite_numbers_survive_a_save(small_flow, tmp_path):
    import copy
    f = copy.copy(small_flow)
    f.report = dataclasses_replace(small_flow.report, gain_nats=math.nan,
                                   smc={"log_z": -math.inf, "stalled": 0, "n_stages": 1})
    f.save(tmp_path / "f.npz")
    g = Flow.load(tmp_path / "f.npz")
    assert math.isnan(g.report.gain_nats) and g.report.smc["log_z"] == -math.inf
    str(g.report)                                         # used to raise on a string
    ev = anvil.EvidenceResult(1.0, math.nan, "x", details={"a": math.inf, "t": (1.0, -math.inf)})
    ev.save(tmp_path / "e.json")
    back = anvil.EvidenceResult.load(tmp_path / "e.json")
    assert back.details["a"] == math.inf and back.details["t"][1] == -math.inf
    with pytest.raises(ValueError, match="not an EvidenceResult"):
        anvil.EvidenceResult.load(tmp_path / "f.npz")


def dataclasses_replace(obj, **kw):
    import dataclasses
    return dataclasses.replace(obj, **kw)


def test_saved_final_positions_are_in_the_targets_coordinates(small_flow, tmp_path):
    f = small_flow
    ft = f.wrap(_banana())
    res = anvil.run(anvil.EnsembleKernel(ft), ft, f.start(32), n_warmup=10, n_samples=10)
    res.save(tmp_path / "r.npz")
    back = anvil.load_results(tmp_path / "r.npz")
    np.testing.assert_allclose(back.final_u, res.get_chain()[-1], rtol=1e-5, atol=1e-5)
    np.testing.assert_array_equal(back.final_z, np.array(res.final_state["u"]))
    plain = anvil.run(anvil.EnsembleKernel(_banana()), _banana(), _start(), n_warmup=2,
                      n_samples=2)
    plain.save(tmp_path / "p.npz")
    assert anvil.load_results(tmp_path / "p.npz").final_z is None


def test_the_start_and_a_compile_refusal_are_recorded():
    t = _banana()
    u0 = _start()
    a = anvil.run(anvil.EnsembleKernel(t), t, u0, n_warmup=2, n_samples=2)
    b = anvil.run(anvil.EnsembleKernel(t), t, u0.copy(), n_warmup=2, n_samples=2)
    c = anvil.run(anvil.EnsembleKernel(t), t, _start(seed=1), n_warmup=2, n_samples=2)
    assert a.metadata["arguments"]["u0"]["shape"] == [32, 2]
    assert a.metadata["arguments"]["u0"]["sha256"] == b.metadata["arguments"]["u0"]["sha256"]
    assert a.metadata["arguments"]["u0"]["sha256"] != c.metadata["arguments"]["u0"]["sha256"]
    assert a.metadata["resolved"]["compile_refused"] is False
