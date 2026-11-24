# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU-only tests for the routed ``sp_qkv_fused`` v1 contract and native reference.

Provider discovery uses a synthesized ``.dist-info`` on ``sys.path`` plus the
stock-main interface's ``_reset_for_testing()``; the native reference runs on
ordinary torch CPU tensors (fp8 is dequantised before matmul).
"""

from __future__ import annotations

import importlib
import textwrap
import uuid

import pytest

torch = pytest.importorskip("torch")

import aiter  # noqa: F401 - the public package must expose the lazy op
import aiter.ops.gemm_fusion.sp_qkv.op as op_module
from aiter import plugin
from aiter.ops.gemm_fusion.sp_qkv.op import sp_qkv_fused

_POS = (
    "q",
    "x",
    "x_scale",
    "w",
    "w_scale",
    "q_norm_w",
    "k_norm_w",
    "cos_cache",
    "sin_cache",
    "positions",
    "slot_mapping",
    "k_cache",
    "v_cache",
    "k_scale",
    "v_scale",
)


def _reset():
    # Current stock main snapshots import-time routes and restores them; the
    # sp_qkv route is registered at module import, so no reload is needed (and
    # reloading would now raise "already routed").
    plugin._reset_for_testing()
    global sp_qkv_fused
    sp_qkv_fused = op_module.sp_qkv_fused


@pytest.fixture(autouse=True)
def _clean_plugin(monkeypatch):
    monkeypatch.delenv("AITER_PLUGINS", raising=False)
    _reset()
    yield
    monkeypatch.delenv("AITER_PLUGINS", raising=False)
    _reset()


def _install_fake_provider(tmp_path, monkeypatch, *, behavior="serve"):
    token = uuid.uuid4().hex
    module_name = f"fake_aiter_provider_{token}"
    provider_name = f"fake-provider-{token}"
    declared_version = 2 if behavior == "wrong-version" else 1
    module = tmp_path / f"{module_name}.py"
    module.write_text(
        textwrap.dedent(f"""
            from aiter.plugin import KernelCandidate

            CALLED = []

            class Provider:
                name = {provider_name!r}
                api_versions = frozenset({{1}})
                op_versions = {{"sp_qkv_fused": frozenset({{{declared_version}}})}}

                def select(self, op, args, kwargs):
                    if op.name != "sp_qkv_fused" or op.version != 1:
                        return None
                    if {behavior!r} == "decline":
                        return None
                    CALLED.append((len(args), dict(kwargs)))
                    return KernelCandidate(
                        provider=self.name,
                        kernel_id="fake.kernel",
                        artifact_digest="sha256:fake",
                        qualification="fake-provider test",
                        run=lambda *a, **k: None,
                    )

            provider = Provider()
            """),
        encoding="utf-8",
    )
    dist = tmp_path / f"{module_name}-0.1.0.dist-info"
    dist.mkdir()
    (dist / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: {module_name}\nVersion: 0.1.0\n",
        encoding="utf-8",
    )
    (dist / "entry_points.txt").write_text(
        f"[aiter.plugins]\n{provider_name} = {module_name}:provider\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    _reset()
    loaded = importlib.import_module(module_name)
    return provider_name, loaded


def _preshuffle(logical):
    rows, k = logical.shape
    return (
        logical.view(rows // 16, 16, k // 32, 2, 16)
        .permute(0, 2, 3, 1, 4)
        .contiguous()
        .view(rows, k)
    )


def _case(mode="dense", m=2, pages=4):
    n = 9216 if mode == "dense" else 9856
    t = 4 * m
    page = 128 if mode == "dense" else 16
    logical = torch.zeros((n, 6144), dtype=torch.float32)
    for head in range(64):
        logical[head * 128, 0] = 1.0
    for rank in range(4):
        logical[8192 + rank * 128, 0] = 2.0
        logical[8704 + rank * 128, 0] = 3.0
    if mode in ("indexer", "reuse"):
        for rank in range(4):
            logical[9216 + rank * 128, 0] = 4.0
        logical[9728, 0] = 5.0
    w = _preshuffle(logical).to(torch.float8_e4m3fn)
    x = torch.zeros((m, 6144), dtype=torch.float8_e4m3fn)
    x[:, 0] = 2.0
    case = {
        "q": torch.zeros((t, 16, 128), dtype=torch.bfloat16),
        "x": x,
        "x_scale": torch.ones((m, 1), dtype=torch.float32),
        "w": w,
        "w_scale": torch.ones(n, dtype=torch.float32),
        "q_norm_w": torch.ones(128, dtype=torch.float32),
        "k_norm_w": torch.ones(128, dtype=torch.float32),
        "cos_cache": torch.ones((4, 32), dtype=torch.bfloat16),
        "sin_cache": torch.zeros((4, 32), dtype=torch.bfloat16),
        "positions": torch.zeros(t, dtype=torch.int64),
        "slot_mapping": torch.arange(t, dtype=torch.int64),
        "k_cache": torch.zeros((pages, 1, 8, page, 16), dtype=torch.float8_e4m3fn),
        "v_cache": torch.zeros(
            (pages, 1, max(1, page // 16), 128, 16), dtype=torch.float8_e4m3fn
        ),
        "k_scale": torch.zeros((pages, 1, page), dtype=torch.float32),
        "v_scale": torch.zeros((pages, 1, page), dtype=torch.float32),
        "mode": mode,
    }
    if mode == "indexer":
        case.update(
            index_q=torch.zeros((t, 1, 128), dtype=torch.bfloat16),
            index_q_norm_w=torch.ones(128, dtype=torch.float32),
            index_k_norm_w=torch.ones(128, dtype=torch.float32),
            index_cache=torch.zeros((pages * page * 128,), dtype=torch.float8_e4m3fn),
        )
    else:
        case.update(
            k=torch.zeros((t, 1, 128), dtype=torch.bfloat16),
            v=torch.zeros((t, 1, 128), dtype=torch.bfloat16),
        )
    return case


def _call(case):
    kwargs = {
        "mode": case["mode"],
        "k": case.get("k"),
        "v": case.get("v"),
        "index_q": case.get("index_q"),
        "index_q_norm_w": case.get("index_q_norm_w"),
        "index_k_norm_w": case.get("index_k_norm_w"),
        "index_cache": case.get("index_cache"),
        "eps": 1e-6,
    }
    return tuple(case[name] for name in _POS), kwargs


def test_routed_op_registers_against_the_main_interface():
    assert plugin.API_VERSION == 1
    assert plugin.ENTRY_POINT_GROUP == "aiter.plugins"
    routed = plugin.status()["routed_ops"]
    assert "sp_qkv_fused" in routed
    assert routed["sp_qkv_fused"]["version"] == 1


def test_no_provider_runs_the_native_reference():
    case = _case()
    before = case["q"].clone()
    result = sp_qkv_fused(*_call(case)[0], **_call(case)[1])
    assert result is None
    assert not torch.equal(case["q"], before), "q must be written in place"
    assert case["q"].dtype == torch.bfloat16
    assert case["k_cache"].float().abs().sum() > 0
    assert case["v_cache"].float().abs().sum() > 0


def test_native_reference_writes_index_q_and_index_cache():
    case = _case("indexer")
    sp_qkv_fused(*_call(case)[0], **_call(case)[1])
    assert case["q"].float().abs().sum() > 0
    assert case["index_q"].float().abs().sum() > 0
    assert case["index_cache"].float().abs().sum() > 0


def test_native_reference_skips_cache_writes_for_padding_rows():
    case = _case()
    case["slot_mapping"].fill_(-1)
    sp_qkv_fused(*_call(case)[0], **_call(case)[1])
    assert case["k_cache"].float().abs().sum() == 0
    assert case["v_cache"].float().abs().sum() == 0
    assert case["q"].float().abs().sum() > 0, "q is written for every row"


def test_native_reference_is_deterministic():
    case = _case()
    sp_qkv_fused(*_call(case)[0], **_call(case)[1])
    first_q = case["q"].clone()
    first_k = case["k_cache"].clone()
    sp_qkv_fused(*_call(case)[0], **_call(case)[1])
    assert torch.equal(case["q"], first_q)
    assert torch.equal(case["k_cache"], first_k)


def test_fake_provider_serves_and_native_is_not_called(tmp_path, monkeypatch):
    provider_name, loaded = _install_fake_provider(tmp_path, monkeypatch)
    plugin.configure(source="cpu-test", enable=[provider_name])
    case = _case()
    before_q = case["q"].clone()
    sp_qkv_fused(*_call(case)[0], **_call(case)[1])
    assert len(loaded.CALLED) == 1
    assert torch.equal(case["q"], before_q), "provider path must not run native"
    decision = plugin.last_decision()
    assert decision is not None and decision.route == "provider"


def test_declining_provider_falls_back_to_native(tmp_path, monkeypatch):
    provider_name, loaded = _install_fake_provider(
        tmp_path, monkeypatch, behavior="decline"
    )
    plugin.configure(source="cpu-test", enable=[provider_name])
    case = _case()
    sp_qkv_fused(*_call(case)[0], **_call(case)[1])
    assert loaded.CALLED == []
    assert case["q"].float().abs().sum() > 0


def test_kill_switch_forces_native(tmp_path, monkeypatch):
    provider_name, loaded = _install_fake_provider(tmp_path, monkeypatch)
    plugin.configure(source="cpu-test", enable=[provider_name])
    monkeypatch.setenv("AITER_PLUGINS", "0")
    plugin.reload_config()
    case = _case()
    sp_qkv_fused(*_call(case)[0], **_call(case)[1])
    assert loaded.CALLED == []
    assert case["q"].float().abs().sum() > 0
    assert plugin.status()["kill_switch"] is True


def test_contract_version_mismatch_provider_is_rejected(tmp_path, monkeypatch):
    provider_name, _loaded = _install_fake_provider(
        tmp_path, monkeypatch, behavior="wrong-version"
    )
    plugin.configure(source="cpu-test", enable=[provider_name])
    case = _case()
    args, kwargs = _call(case)
    explanation = plugin.explain("sp_qkv_fused", *args, **kwargs)
    assert provider_name in explanation.declines
    assert "version" in explanation.declines[provider_name].lower()
    sp_qkv_fused(*args, **kwargs)
    assert case["q"].float().abs().sum() > 0


def test_invalid_mode_is_a_value_error_before_dispatch():
    case = _case()
    case["mode"] = "bogus"
    with pytest.raises(ValueError, match="mode must be one of"):
        sp_qkv_fused(*_call(case)[0], **_call(case)[1])


def test_native_index_helpers_use_the_input_device():
    # CPU tensors cannot reproduce a cuda/cpu index mismatch; a strict fake for
    # torch.arange makes "device=" mandatory, so the old CPU-index bug fails here.
    class StrictArange:
        def arange(self, *args, **kwargs):
            assert "device" in kwargs, "index tensor must carry an explicit device"
            return torch.arange(*args, **kwargs)

    gemm = torch.empty((2, 9216), device="meta")
    q, k, v, iq, ik = op_module._qv_rows(gemm, "dense", 2, StrictArange())
    assert q.device.type == "meta" and q.shape == (8 * 16, 128)
    assert k.device.type == "meta" and k.shape == (8, 1, 128)
    assert v.device.type == "meta" and v.shape == (8, 1, 128)
    assert iq is None and ik is None


def test_qv_rows_uses_the_local_rank_shard_for_every_source_block():
    # The old mapping used block s -> shard s; the fused kernel gives every source
    # token block the destination (local) rank's head shard.  Distinct shard weights
    # make the two mappings distinguishable from CPU tensors.
    m = 2
    gemm = torch.zeros((m, 9216))
    for shard in range(4):
        for head in range(16):
            gemm[:, (shard * 16 + head) * 128] = shard * 100 + head
        gemm[:, 8192 + shard * 128] = 1000 + shard
        gemm[:, 8704 + shard * 128] = 2000 + shard
    q, k, v, iq, ik = op_module._qv_rows(gemm, "dense", m, torch, rank=1)
    blocks = q.reshape(4, m, 16, 128)
    assert all(torch.equal(blocks[0], blocks[i]) for i in (1, 2, 3))
    assert all(
        torch.equal(blocks[0][:, h, 0], torch.full((m,), 100.0 + h)) for h in range(16)
    )
    assert torch.equal(k.reshape(4, m, 128)[:, :, 0], torch.full((4, m), 1001.0))
    assert torch.equal(v.reshape(4, m, 128)[:, :, 0], torch.full((4, m), 2001.0))
    assert iq is None and ik is None


def test_qv_rows_index_tensors_use_the_local_rank_shard():
    m = 2
    gemm = torch.zeros((m, 9856))
    for shard in range(4):
        gemm[:, 9216 + shard * 128] = 3000 + shard
    gemm[:, 9728] = 4000.0
    _, _, _, iq, ik = op_module._qv_rows(gemm, "indexer", m, torch, rank=2)
    assert all(
        torch.equal(iq.reshape(4, m, 128)[0], iq.reshape(4, m, 128)[i])
        for i in (1, 2, 3)
    )
    assert torch.equal(iq.reshape(4, m, 128)[:, :, 0], torch.full((4, m), 3002.0))
    assert torch.equal(ik[:, 0], torch.full((4 * m,), 4000.0))


def test_shape_or_dtype_validation_errors():
    case = _case()
    case["q"] = case["q"].to(torch.float32)
    with pytest.raises(ValueError, match="q must be"):
        sp_qkv_fused(*_call(case)[0], **_call(case)[1])


def test_unsupported_native_case_raises_not_implemented():
    case = _case()
    args, kwargs = _call(case)
    kwargs["group"] = type("G", (), {"size": lambda self: 8})()
    with pytest.raises(NotImplementedError, match="SP world 4"):
        sp_qkv_fused(*args, **kwargs)
