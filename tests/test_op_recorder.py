"""AITER_RECORD bundle writing and argument rebuilding.

``aiter/op_recorder.py`` and ``op_tests/replay/replay_fused_moe.py`` are loaded
by path (without importing the aiter package), so these tests run on any host
with torch, including CPU-only ones. The GPU replay itself is not covered here.
"""

import dataclasses
import enum
import importlib.util
import inspect
import json
import os
import pathlib
import sys

import pytest
import torch

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load(name, relpath):
    spec = importlib.util.spec_from_file_location(name, _REPO_ROOT / relpath)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


rec = _load("aiter_op_recorder_under_test", "aiter/op_recorder.py")
replay = _load("replay_fused_moe_under_test", "op_tests/replay/replay_fused_moe.py")


class Act(enum.Enum):
    Silu = 0
    Swiglu = 1


# Registered under a stable importable name so the enum can be resolved back.
sys.modules.setdefault("op_recorder_test_enums", sys.modules[__name__])
Act.__module__ = "op_recorder_test_enums"


@dataclasses.dataclass
class Scatter:
    handle: int
    source_map: torch.Tensor


def fake_moe(
    hidden_states,
    w1,
    topk_weight,
    topk_ids,
    expert_mask=None,
    activation=Act.Silu,
    dtype=None,
    block_size_M=None,
    stage2_scatter=None,
    fail=False,
):
    rec.note_dispatch({"op": "fake", "kernels": [f"k_M{hidden_states.shape[0]}"]})
    if fail:
        raise RuntimeError("boom")
    return hidden_states * 2


@pytest.fixture
def recorder(tmp_path, monkeypatch):
    """A recorded fake_moe writing under tmp_path, with fresh module state."""

    def make(**env):
        config = tmp_path / "tuned.csv"
        config.write_text("token,kernel\n16,asm\n")
        spec = rec.OpSpec(
            name="fake_moe",
            payload_args=("topk_weight",),
            session_info=lambda: {"patched": {}},
            config_files=lambda: {"AITER_CONFIG_FAKE": str(config)},
        )
        environ = {"AITER_RECORD": str(tmp_path / "rec"), **env}
        return rec.record_op(spec, environ=environ)(fake_moe)

    monkeypatch.setattr(rec, "_WRITER", None)
    monkeypatch.setattr(rec, "_HOOK_ACTIVE", False)
    yield make
    if rec._WRITER is not None:
        rec._WRITER.close()


def _inputs(m=4, experts=8, topk=2, dim=16):
    w1 = torch.empty(experts, 2 * dim, dim, dtype=torch.bfloat16)
    w1.is_shuffled = True
    w1.gate_up_interleave = False
    return {
        "hidden_states": torch.randn(m, dim, dtype=torch.bfloat16),
        "w1": w1,
        "topk_weight": torch.rand(m, topk, dtype=torch.float32),
        "topk_ids": torch.randint(0, experts, (m, topk), dtype=torch.int32),
    }


def _bundle(tmp_path):
    (bundle,) = (tmp_path / "rec").iterdir()
    return bundle


def test_settings_from_env():
    assert rec.RecordSettings.from_env({}) is None
    s = rec.RecordSettings.from_env(
        {
            "AITER_RECORD": "/x",
            "AITER_RECORD_OPS": "fused_moe, mla",
            "AITER_RECORD_SKIP_CALLS": "5",
            "AITER_RECORD_MAX_CALLS": "7",
            "AITER_RECORD_EVERY": "0",
        }
    )
    assert s.root == "/x" and s.ops == {"fused_moe", "mla"}
    assert (s.skip_calls, s.max_calls, s.every) == (5, 7, 1)
    assert s.wants("mla") and not s.wants("gemm")
    assert (
        rec.RecordSettings.from_env(
            {"AITER_RECORD": "/x", "AITER_RECORD_OPS": "all"}
        ).ops
        is None
    )


def test_disabled_returns_function_unchanged():
    spec = rec.OpSpec(name="fake_moe")
    assert rec.record_op(spec, environ={})(fake_moe) is fake_moe
    off = {"AITER_RECORD": "/x", "AITER_RECORD_OPS": "other"}
    assert rec.record_op(spec, environ=off)(fake_moe) is fake_moe


def test_record_roundtrip(recorder, tmp_path):
    op = recorder()
    # vLLM and SGLang feature-detect fused_moe kwargs via inspect.signature.
    assert inspect.signature(op) == inspect.signature(fake_moe)
    inputs = _inputs()
    out = op(**inputs, activation=Act.Swiglu, dtype=torch.bfloat16)
    assert torch.equal(out, inputs["hidden_states"] * 2)
    rec._WRITER.close()

    bundle = _bundle(tmp_path)
    session = rec.load_session(bundle)
    assert session["format_version"] == rec.FORMAT_VERSION
    assert "AITER_RECORD" not in session["env"]
    op_info = session["ops"]["fake_moe"]
    assert op_info["patched"] == {}
    copied = bundle / op_info["configs"]["AITER_CONFIG_FAKE"]["file"]
    assert copied.read_text() == "token,kernel\n16,asm\n"

    (call,) = list(rec.iter_calls(bundle))
    assert call["op"] == "fake_moe" and call["capturing"] is False
    assert call["dispatch"] == [{"op": "fake", "kernels": ["k_M4"]}]
    args = call["args"]
    assert set(args) == {
        "hidden_states",
        "w1",
        "topk_weight",
        "topk_ids",
        "expert_mask",
        "activation",
        "dtype",
        "block_size_M",
        "stage2_scatter",
        "fail",
    }
    w1 = args["w1"]["__tensor__"]
    assert w1["shape"] == [8, 32, 16] and w1["dtype"] == "bfloat16"
    assert w1["attrs"] == {"is_shuffled": True, "gate_up_interleave": False}
    assert args["w1"]["payload"] is None  # float weights: metadata only
    assert args["hidden_states"]["payload"] is None
    assert (
        args["activation"]["__enum__"] == "Act"
        and args["activation"]["name"] == "Swiglu"
    )
    assert args["dtype"] == {"__dtype__": "bfloat16"}
    assert args["expert_mask"] is None

    payloads = rec.load_payloads(bundle)
    assert torch.equal(payloads[args["topk_ids"]["payload"]], inputs["topk_ids"])
    assert torch.equal(payloads[args["topk_weight"]["payload"]], inputs["topk_weight"])


def test_sampling_and_payload_dedup(recorder, tmp_path):
    op = recorder(
        AITER_RECORD_SKIP_CALLS="2", AITER_RECORD_EVERY="2", AITER_RECORD_MAX_CALLS="3"
    )
    inputs = _inputs()
    for _ in range(12):
        op(**inputs)
    rec._WRITER.close()
    bundle = _bundle(tmp_path)
    calls = list(rec.iter_calls(bundle))
    # calls 0,1 skipped; then every 2nd: 2,4,6; capped at 3
    assert [c["op_seq"] for c in calls] == [2, 4, 6]
    assert len({c["args"]["topk_ids"]["payload"] for c in calls}) == 1
    assert len(rec.load_payloads(bundle)) == 2  # topk_ids + topk_weight, stored once
    counters = rec.load_session(bundle)["counters"]["fake_moe"]
    assert counters["seen"] == 12 and counters["recorded"] == 3


def test_error_is_recorded_and_raised(recorder, tmp_path):
    op = recorder()
    with pytest.raises(RuntimeError, match="boom"):
        op(**_inputs(), fail=True)
    rec._WRITER.close()
    (call,) = list(rec.iter_calls(_bundle(tmp_path)))
    assert call["error"] == "RuntimeError: boom"


def test_unsupported_objects_are_flagged(recorder, tmp_path):
    op = recorder()
    scatter = Scatter(handle=3, source_map=torch.zeros(4, dtype=torch.int32))
    op(**_inputs(), stage2_scatter=scatter)
    rec._WRITER.close()
    (call,) = list(rec.iter_calls(_bundle(tmp_path)))
    problems = rec.find_unsupported(call["args"])
    assert len(problems) == 1 and problems[0].startswith("stage2_scatter")


def test_capturing_calls_skip_payloads(recorder, tmp_path, monkeypatch):
    monkeypatch.setattr(rec, "_is_capturing", lambda: True)
    op = recorder()
    op(**_inputs())
    rec._WRITER.close()
    (call,) = list(rec.iter_calls(_bundle(tmp_path)))
    assert call["capturing"] is True
    assert call["args"]["topk_ids"]["payload"] is None


def test_collect_dispatch_without_recording(monkeypatch):
    monkeypatch.setattr(rec, "_HOOK_ACTIVE", False)
    assert not rec.dispatch_hook_active()
    with rec.collect_dispatch() as seen:
        assert rec.dispatch_hook_active()
        rec.note_dispatch({"kernels": ["a"]})
    assert seen == [{"kernels": ["a"]}]
    assert not rec.dispatch_hook_active()


def _tensor_enc(shape, stride, dtype, ptr, offset=0, attrs=None, payload=None):
    return {
        "__tensor__": {
            "shape": shape,
            "stride": stride,
            "storage_offset": offset,
            "dtype": dtype,
            "data_ptr": ptr,
            "attrs": attrs or {},
        },
        "payload": payload,
    }


def test_argument_factory_rebuilds_layout_and_values():
    ids = torch.tensor([[1, 2], [3, 0]], dtype=torch.int32)
    factory = rec.ArgumentFactory({"p1": ids}, device="cpu")
    w_enc = _tensor_enc(
        [2, 8, 4], [32, 4, 1], "float4_e2m1fn_x2", 4096, attrs={"is_shuffled": True}
    )
    built = factory.build_args(
        {
            "w1": w_enc,
            "w1_scale": _tensor_enc([2, 8], [8, 1], "float8_e8m0fnu", 8192),
            "w2_scale": _tensor_enc([2, 8], [8, 1], "uint8", 9000),
            "hidden_states": _tensor_enc([2, 4], [1, 2], "bfloat16", 12288, offset=3),
            "topk_ids": _tensor_enc([2, 2], [2, 1], "int32", 0, payload="p1"),
            "activation": {
                "__enum__": "Act",
                "module": "op_recorder_test_enums",
                "name": "Swiglu",
            },
            "dtype": {"__dtype__": "bfloat16"},
            "pair": {"__tuple__": [1, {"__dtype__": "float32"}]},
            "block_size_M": None,
        }
    )
    w1 = built["w1"]
    assert w1.dtype == torch.float4_e2m1fn_x2 and w1.shape == (2, 8, 4)
    assert w1.is_shuffled is True
    assert torch.all(built["w1_scale"].view(torch.uint8) == 127)
    assert torch.all(built["w2_scale"] == 127)
    hs = built["hidden_states"]
    assert hs.stride() == (1, 2) and hs.storage_offset() == 3
    assert torch.equal(built["topk_ids"], ids)
    assert built["activation"] is Act.Swiglu
    assert built["dtype"] is torch.bfloat16
    assert built["pair"] == (1, torch.float32)
    assert built["block_size_M"] is None

    # Same recorded buffer -> same replay buffer; another layer -> new buffer.
    assert factory.build(w_enc, "w1") is w1
    other_layer = _tensor_enc([2, 8, 4], [32, 4, 1], "float4_e2m1fn_x2", 1 << 20)
    assert factory.build(other_layer, "w1") is not w1
    shared = rec.ArgumentFactory({}, device="cpu", ignore_ptr_for=("w1",))
    assert shared.build(w_enc, "w1") is shared.build(other_layer, "w1")


def test_argument_factory_rejects_unsupported():
    factory = rec.ArgumentFactory({}, device="cpu")
    with pytest.raises(ValueError, match="stage2_scatter"):
        factory.build({"__object__": "x.Scatter", "fields": {}}, "stage2_scatter")
    with pytest.raises(ValueError, match="missing"):
        factory.build(_tensor_enc([1], [1], "int32", 0, payload="nope"), "topk_ids")


def test_replay_padding_and_buckets():
    assert replay.target_m(12, [1, 2, 4, 8, 16, 24]) == 16
    assert replay.target_m(16, [1, 2, 4, 8, 16, 24]) == 16
    assert replay.target_m(100, [1, 2, 4, 8, 16, 24]) == 100
    assert replay.target_m(5, []) == 5

    ids = torch.tensor([[0, 1], [2, 3], [4, 5]], dtype=torch.int32)
    hs = torch.randn(3, 8)
    hs.marker = "kept"
    w1 = torch.randn(3, 8)  # first dim == M by coincidence: not per-token
    kwargs = {"hidden_states": hs, "topk_ids": ids, "w1": w1, "output": None}
    replay.pad_per_token(kwargs, 3, 5, torch)
    assert kwargs["hidden_states"].shape == (5, 8)
    assert kwargs["hidden_states"].marker == "kept"
    assert kwargs["topk_ids"].tolist() == [[0, 1], [2, 3], [4, 5], [0, 1], [2, 3]]
    assert kwargs["w1"] is w1 and kwargs["output"] is None


def test_replay_summary():
    def row(bucket, m, us, match, kernels="asm"):
        return {
            "status": "ok",
            "bucket": bucket,
            "replay_m": m,
            "median_us": us,
            "kernels_match": match,
            "replayed_kernels": kernels,
        }

    rows = [
        row(16, 12, 10.0, True),
        row(16, 16, 20.0, False),
        row(8, 8, 5.0, ""),
        {"status": "skipped:capturing"},
    ]
    summary = {s["bucket"]: s for s in replay.summarize(rows)}
    assert summary[16]["calls"] == 2 and summary[16]["share_pct"] == 66.7
    assert summary[16]["m_values"] == "12:1 16:1"
    assert summary[16]["kernels_match_pct"] == 50.0
    assert summary[16]["median_us"] == 15.0 and summary[16]["total_us"] == 30.0
    assert summary[8]["kernels_match_pct"] == ""


def test_replay_restores_environment(tmp_path, monkeypatch):
    bundle = tmp_path / "b"
    (bundle / "configs").mkdir(parents=True)
    session = {
        "env": {
            "AITER_USE_NT": "1",
            "AITER_RECORD_OPS": "fused_moe",
            "AITER_JIT_DIR": "/definitely/not/here",
        },
        "ops": {
            "fused_moe": {"configs": {"AITER_CONFIG_FMOE": {"file": "configs/f.csv"}}}
        },
    }
    for key in ("AITER_USE_NT", "AITER_CONFIG_FMOE", "AITER_JIT_DIR"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("AITER_RECORD", "/should/be/removed")
    args = replay.parse_args([str(bundle)])
    notes = replay.restore_environment(session, bundle, args)
    assert os.environ["AITER_USE_NT"] == "1"
    assert os.environ["AITER_CONFIG_FMOE"] == str((bundle / "configs/f.csv").resolve())
    assert "AITER_RECORD" not in os.environ and "AITER_RECORD_OPS" not in os.environ
    assert "AITER_JIT_DIR" not in os.environ
    assert any("AITER_JIT_DIR" in n for n in notes)
    json.dumps(notes)
