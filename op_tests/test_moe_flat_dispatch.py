# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import ast
import importlib
import inspect
import itertools
from pathlib import Path

import pandas as pd
import pytest
import torch
import torch.nn.functional as F

import aiter
from aiter import ActivationType, QuantType, dtypes
from aiter.aot.flydsl import mxfp4_moe as mxfp4_aot
from aiter.jit import core
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl import moe_direct_m1 as direct_m1
from aiter.ops.flydsl.kernels import mxfp4_gemm1
from aiter.ops.flydsl.mxfp4_moe_capability import MoeCall, metadata_kernel_name
from aiter.ops.shuffle import shuffle_weight
from aiter.utility import fp4_utils

fused_moe = importlib.import_module("aiter.fused_moe")

BM1 = "flydsl_mxmoe_g1_a4w4_16x256x256_NT_F16IN"
BM2 = "flydsl_moe2_layout_afp4_wfp4_bf16_t16x128x128_atomic_sbm16"
D1 = "flydsl_moe1_direct_m1_afp4_wfp4_bf16_t32x32x256_w4_kw2_fp4"
D2 = "flydsl_moe2_direct_m1_layout_afp4_wfp4_bf16_t32x128x128_atomic_sbm32"
HIDDEN, INTER, EXPERTS, TOPK = 1024, 384, 8, 4
BETA, LINEAR_BETA = 3.5, 17.0


def _row(kind: str, token: int | None = None) -> dict:
    direct = kind == "direct"
    row = {
        "gfx": "gfx950",
        "cu_num": 256,
        "token": 1 if direct else (2 if token is None else token),
        "model_dim": HIDDEN,
        "inter_dim": INTER,
        "expert": EXPERTS,
        "topk": TOPK,
        "act_type": "ActivationType.Situv2",
        "dtype": "torch.bfloat16",
        "q_dtype_a": "torch.float4_e2m1fn_x2",
        "q_dtype_w": "torch.float4_e2m1fn_x2",
        "q_type": "QuantType.per_1x32",
        "use_g1u1": 1,
        "doweight_stage1": 0,
        "block_m": 32 if direct else 16,
        "ksplit": 0,
        "run_1stage": 0,
        "flat": int(direct),
        "kernelName1": D1 if direct else BM1,
        "kernelName2": D2 if direct else BM2,
        "us": 1.0,
    }
    assert "path" not in row
    return row


def _metadata_kernels(metadata) -> tuple[str, str]:
    return tuple(metadata_kernel_name(metadata, stage) for stage in (1, 2))


def _moe_call(x, w1, w2, routed, ids, s1, s2, **overrides) -> MoeCall:
    defaults = {
        "hidden_states": x,
        "w1": w1,
        "w2": w2,
        "topk_weight": routed,
        "topk_ids": ids,
        "w1_scale": s1,
        "w2_scale": s2,
        "dtype": torch.bfloat16,
        "q_dtype_a": dtypes.fp4x2,
        "q_dtype_w": dtypes.fp4x2,
        "quant_type": QuantType.per_1x32,
        "activation": ActivationType.Situv2,
        "gate_mode": fused_moe.GateMode.SEPARATED,
        "isG1U1": True,
        "doweight_stage1": False,
        "expert_mask": None,
        "num_local_tokens": None,
        "bias1": None,
        "bias2": None,
        "a1_scale": None,
        "a2_scale": None,
        "stage2_scatter": None,
        "hidden_pad": 0,
        "intermediate_pad": 0,
        "block_size_M": None,
        "beta": BETA,
        "linear_beta": LINEAR_BETA,
    }
    return MoeCall(**{**defaults, **overrides})


def _clear_config_caches() -> None:
    fused_moe.cfg_2stages = None
    fused_moe.cfg_2stages_by_file.clear()
    fused_moe.get_2stage_cfgs.cache_clear()
    type(core.AITER_CONFIGS).get_config_file.cache_clear()


def _lookup(
    monkeypatch,
    csv_path,
    token: int,
    *,
    activation=ActivationType.Situv2,
    disable_direct=False,
    disable_inline=False,
):
    monkeypatch.setenv("AITER_CONFIG_FMOE", str(csv_path))
    monkeypatch.setattr(fused_moe, "get_cu_num", lambda: 256)
    monkeypatch.setattr(fused_moe, "get_gfx_runtime", lambda: "gfx950")
    monkeypatch.setattr(fused_moe, "get_gfx", lambda: "gfx950")
    _clear_config_caches()
    return fused_moe.get_2stage_cfgs(
        token,
        HIDDEN,
        INTER,
        EXPERTS,
        TOPK,
        torch.bfloat16,
        dtypes.fp4x2,
        dtypes.fp4x2,
        QuantType.per_1x32,
        True,
        activation,
        False,
        0,
        0,
        True,
        "separated",
        _disable_direct_m1=disable_direct,
        _disable_inline_sort=disable_inline,
    )


def test_old_schema_flat_one_stage_and_two_stage_metadata(monkeypatch, tmp_path):
    config = tmp_path / "fmoe.csv"

    legacy_flat = _row("bm16")
    legacy_flat.update(
        act_type="ActivationType.Silu",
        run_1stage=1,
        flat=1,
        block_m=32,
        kernelName1="legacy_flat_asm_kernel",
        kernelName2="",
    )
    pd.DataFrame([legacy_flat]).to_csv(config, index=False)
    metadata = _lookup(monkeypatch, config, 2, activation=ActivationType.Silu)
    assert metadata.run_1stage and metadata.flat

    pd.DataFrame([_row("bm16")]).to_csv(config, index=False)
    metadata = _lookup(monkeypatch, config, 2)
    assert not metadata.run_1stage and not metadata.flat
    assert metadata.block_m == 16 and metadata.output_aux
    assert fused_moe._is_mxfp4_inline_sort(metadata)

    old_row = _row("bm16")
    old_row.pop("flat")
    pd.DataFrame([old_row]).to_csv(config, index=False)
    assert not _lookup(monkeypatch, config, 2).flat

    pd.DataFrame([_row("direct")]).to_csv(config, index=False)
    metadata = _lookup(monkeypatch, config, 1)
    assert not metadata.run_1stage and metadata.flat
    assert _metadata_kernels(metadata) == (D1, D2)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("flat", 0),
        ("run_1stage", 1),
        ("kernelName1", "flydsl_moe1_direct_m1_unknown"),
        ("kernelName2", BM2),
    ],
)
def test_invalid_direct_config_falls_back(monkeypatch, tmp_path, field, value):
    row = _row("direct")
    row[field] = value
    config = tmp_path / "invalid.csv"
    pd.DataFrame([row]).to_csv(config, index=False)
    metadata = _lookup(monkeypatch, config, 1)
    assert not metadata.flat
    assert not direct_m1.is_direct_kernel_pair(*_metadata_kernels(metadata))


def test_bm16_inline_config_can_fall_back(monkeypatch, tmp_path):
    config = tmp_path / "bm16.csv"
    pd.DataFrame([_row("bm16")]).to_csv(config, index=False)
    tuned = _lookup(monkeypatch, config, 2)
    fallback = _lookup(monkeypatch, config, 2, disable_inline=True)
    assert fused_moe._is_mxfp4_inline_sort(tuned)
    assert not fallback.output_aux
    assert not fused_moe._is_mxfp4_inline_sort(fallback)


def test_production_csv_stage_field_semantics():
    config = (
        Path(core.AITER_ROOT_DIR)
        / "aiter"
        / "configs"
        / "model_configs"
        / "kimik3_a4w4_tuned_fmoe.csv"
    )
    rows = pd.read_csv(config)
    rows = rows[rows["token"].isin([1, 2, 4, 8, 16])]
    assert len(rows) == 5
    assert all(abs(row.us - (row.us1 + row.us2)) < 1e-6 for row in rows.itertuples())
    assert all(str(value).endswith("%") for value in rows["err1"])
    assert all(str(value).endswith("%") for value in rows["err2"])


def test_bm16_runtime_capability_rejects_ep_and_missing_aux(
    monkeypatch, tmp_path, weights
):
    _, _, w1, w2, s1, s2 = weights
    config = tmp_path / "bm16-runtime.csv"
    pd.DataFrame([_row("bm16")]).to_csv(config, index=False)
    metadata = _lookup(monkeypatch, config, 2)
    x = torch.zeros(2, HIDDEN, dtype=torch.bfloat16, device=w1.device)
    ids = torch.zeros(2, TOPK, dtype=torch.int32, device=w1.device)
    routed = torch.zeros(2, TOPK, device=w1.device)
    call = _moe_call(
        x,
        w1,
        w2,
        routed,
        ids,
        s1,
        s2,
        expert_mask=torch.ones(EXPERTS, dtype=torch.int32, device=w1.device),
    )
    assert fused_moe._mxfp4_inline_sort_runtime_capability(metadata, call) == (
        False,
        "expert-parallel masking or local-token metadata",
    )
    call = _moe_call(x, w1, w2, routed, ids, s1, s2)
    supported, reason = fused_moe._mxfp4_inline_sort_runtime_capability(metadata, call)
    assert not supported and "aux" in reason


def test_bm16_ep_config_falls_back_with_correct_output(monkeypatch, tmp_path, weights):
    ref1, ref2, w1, w2, s1, s2 = weights
    row = _row("bm16")
    # EP lookup strips the conventionally appended fake route from topk.
    row["topk"] = TOPK - 1
    config = tmp_path / "bm16-ep.csv"
    pd.DataFrame([row]).to_csv(config, index=False)
    monkeypatch.setenv("AITER_CONFIG_FMOE", str(config))
    monkeypatch.setenv("AITER_SITUV2_A4W4", "1")
    _clear_config_caches()
    x = torch.randn(2, HIDDEN, device=w1.device, dtype=torch.bfloat16) / 10
    ids = torch.arange(TOPK, device=w1.device, dtype=torch.int32).repeat(2, 1)
    routed = torch.full((2, TOPK), 1 / TOPK, device=w1.device)
    expert_mask = torch.ones(EXPERTS, dtype=torch.int32, device=w1.device)

    result = fused_moe.fused_moe(
        x,
        w1,
        w2,
        routed,
        ids,
        expert_mask=expert_mask,
        activation=ActivationType.Situv2,
        quant_type=QuantType.per_1x32,
        w1_scale=s1,
        w2_scale=s2,
        beta=BETA,
        linear_beta=LINEAR_BETA,
        gate_mode="separated",
    )
    expected = _reference(x, ids, routed, ref1, ref2)
    assert torch.isfinite(result).all()
    assert (
        F.cosine_similarity(result.float().flatten(), expected.flatten(), dim=0) > 0.9
    )


def test_direct_config_contract_and_aot(monkeypatch, tmp_path):
    row = _row("direct")
    assert direct_m1.cfg_is_supported(row) == (True, "")
    from aiter.ops.flydsl.moe_kernels import get_flydsl_kernel_params

    assert get_flydsl_kernel_params(direct_m1._base_name(D1, 1))["k_wave"] == 2
    assert [job["stage"] for job in direct_m1.aot_jobs(row)] == [1, 2]

    config = tmp_path / "aot.csv"
    pd.DataFrame([row]).to_csv(config, index=False)
    jobs = mxfp4_aot.parse_csv(str(config))
    assert len(jobs) == 2 and all(job["direct_m1"] for job in jobs)
    assert all(
        mxfp4_aot.compile_one_config(**job)["compile_time"] is not None for job in jobs
    )

    pd.DataFrame([_row("bm16")]).to_csv(config, index=False)
    jobs = mxfp4_aot.parse_csv(str(config))
    assert sum(job["stage"] == 1 for job in jobs) == 1
    assert sum(job["stage"] == 2 for job in jobs) == 2
    assert next(job for job in jobs if job["stage"] == 1)["activation"] == "situv2"
    assert all(
        mxfp4_aot.compile_one_config(**job)["compile_time"] is not None for job in jobs
    )

    calls = []
    monkeypatch.setattr(direct_m1, "compile_aot_job", lambda **job: calls.append(job))
    job = direct_m1.aot_jobs(row)[0]
    assert mxfp4_aot.compile_one_config(**job)["compile_time"] is not None
    assert calls == [job]


def test_silu_abi_and_cache_key_are_unchanged():
    tree = ast.parse(inspect.getsource(mxfp4_gemm1.compile_gemm1_a4w4_port))
    launchers = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "launch_gemm1"
    ]
    assert sorted(len(node.args.args) for node in launchers) == [13, 15]
    base = {
        "stage": 1,
        "BM": 16,
        "use_nt": True,
        "inline_quant": True,
        "D_HIDDEN": HIDDEN,
        "D_INTER": INTER,
        "NE": EXPERTS,
        "topk": TOPK,
        "xcd_swizzle": 0,
        "activation": "silu",
    }
    assert mxfp4_aot._job_key(base) == (
        1,
        16,
        True,
        True,
        HIDDEN,
        INTER,
        EXPERTS,
        TOPK,
        0,
    )
    assert mxfp4_aot._job_key(base) != mxfp4_aot._job_key(
        {**base, "activation": "situv2"}
    )


def _padded_row_bound(routes: int, experts: int, block_m: int) -> int:
    active = min(routes, experts)
    return block_m * (active + (routes - active) // block_m)


def test_padded_row_bound_is_exact_for_small_partitions():
    for routes, experts, block_m in itertools.product(range(1, 8), range(1, 4), (2, 4)):
        maximum = max(
            sum((count + block_m - 1) // block_m * block_m for count in counts)
            for counts in itertools.product(range(routes + 1), repeat=experts)
            if sum(counts) <= routes
        )
        assert maximum == _padded_row_bound(routes, experts, block_m)


def _launch_quant_sort(*, bounded: bool):
    m, topk, experts, block_m, cols = 1, 4, 8, 16, 256
    capacity = m * topk + experts * block_m - topk
    sorted_ids = torch.full((capacity,), m, dtype=torch.int32, device="cuda")
    for route in range(topk):
        sorted_ids[route * block_m] = route << 24
    num_valid = torch.tensor([topk * block_m, m], dtype=torch.int32, device="cuda")
    x = torch.randn(m, cols, dtype=torch.bfloat16, device="cuda")
    out = torch.full((m, cols // 2), 0xA5, dtype=torch.uint8, device="cuda").view(
        dtypes.fp4x2
    )
    scale_cols = ((cols // 32 + 7) // 8) * 8
    scales = torch.full(
        (((capacity + 31) // 32) * 32, scale_cols),
        0xA5,
        dtype=torch.uint8,
        device="cuda",
    ).view(dtypes.fp8_e8m0)
    args = (out, scales, x, sorted_ids, num_valid, m, block_m)
    if bounded:
        aiter.fused_dynamic_mx_quant_moe_sort_hip_bounded(
            *args, m * topk, experts, 32, None
        )
    else:
        aiter.fused_dynamic_mx_quant_moe_sort_hip(*args, 32, None)
    return out, scales


def test_bounded_quant_matches_legacy():
    if not torch.cuda.is_available():
        pytest.skip("requires a HIP device")
    torch.manual_seed(7)
    legacy = tuple(t.clone() for t in _launch_quant_sort(bounded=False))
    torch.manual_seed(7)
    bounded = _launch_quant_sort(bounded=True)
    torch.cuda.synchronize()
    assert all(
        torch.equal(expected, actual)
        for expected, actual in zip(legacy, bounded, strict=True)
    )


@pytest.fixture(scope="module")
def weights():
    if not torch.cuda.is_available() or get_gfx() != "gfx950":
        pytest.skip("requires gfx950")
    torch.manual_seed(23)
    quant = aiter.get_torch_quant(QuantType.per_1x32)

    ref1 = (
        torch.randn(EXPERTS, 2 * INTER, HIDDEN, device="cuda", dtype=torch.bfloat16)
        / 10
    )
    q1, raw_s1 = quant(ref1, quant_dtype=dtypes.fp4x2)
    w1 = shuffle_weight(q1.view(EXPERTS, 2 * INTER, HIDDEN // 2), (16, 16))
    w1.is_shuffled = True
    s1 = fp4_utils.e8m0_shuffle(raw_s1)

    ref2 = torch.randn(EXPERTS, HIDDEN, INTER, device="cuda", dtype=torch.bfloat16) / 10
    q2, raw_s2 = quant(ref2, quant_dtype=dtypes.fp4x2)
    w2 = shuffle_weight(q2.view(EXPERTS, HIDDEN, INTER // 2), (16, 16))
    w2.is_shuffled = True
    s2 = fp4_utils.e8m0_shuffle(raw_s2)
    return ref1, ref2, w1, w2, s1, s2


def _reference(x, ids, routed, w1, w2):
    out = torch.zeros(x.shape[0], HIDDEN, device=x.device)
    for token in range(x.shape[0]):
        for slot in range(TOPK):
            expert = int(ids[token, slot])
            if not 0 <= expert < w1.shape[0]:
                continue
            gate, up = (x[token].float() @ w1[expert].float().T).chunk(2)
            act = (
                BETA
                * torch.tanh(gate / BETA)
                * torch.sigmoid(gate)
                * LINEAR_BETA
                * torch.tanh(up / LINEAR_BETA)
            )
            out[token] += routed[token, slot] * (act @ w2[expert].float().T)
    return out


def _call_situv2(x, w1, w2, routed, ids, s1, s2):
    return fused_moe.fused_moe(
        x,
        w1,
        w2,
        routed,
        ids,
        activation=ActivationType.Situv2,
        quant_type=QuantType.per_1x32,
        w1_scale=s1,
        w2_scale=s2,
        beta=BETA,
        linear_beta=LINEAR_BETA,
        gate_mode="separated",
    )


@pytest.mark.parametrize(("kind", "m"), [("direct", 1), ("bm16", 2)])
def test_m1_and_bm16_graph_correctness(monkeypatch, tmp_path, weights, kind, m):
    ref1, ref2, w1, w2, s1, s2 = weights
    x = torch.randn(m, HIDDEN, device="cuda", dtype=torch.bfloat16) / 10
    ids = torch.arange(TOPK, device="cuda", dtype=torch.int32).repeat(m, 1)
    routed = torch.full((m, TOPK), 1 / TOPK, device="cuda")
    config = tmp_path / f"{kind}.csv"
    pd.DataFrame([_row(kind, m)]).to_csv(config, index=False)
    monkeypatch.setenv("AITER_CONFIG_FMOE", str(config))
    monkeypatch.setenv("AITER_SITUV2_A4W4", "1")
    _clear_config_caches()

    def call():
        return _call_situv2(x, w1, w2, routed, ids, s1, s2)

    result = call()
    reference = _reference(x, ids, routed, ref1, ref2)
    assert (
        F.cosine_similarity(result.float().flatten(), reference.flatten(), dim=0) > 0.9
    )

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = call()
    graph.replay()
    assert (
        F.cosine_similarity(result.float().flatten(), reference.flatten(), dim=0) > 0.9
    )
    routed.zero_()
    graph.replay()
    assert torch.count_nonzero(result) == 0


@pytest.mark.parametrize("bad_id", [-1, EXPERTS, EXPERTS + 1])
def test_direct_invalid_expert_ids_are_zero_routes(
    monkeypatch, tmp_path, weights, bad_id
):
    ref1, ref2, w1, w2, s1, s2 = weights
    x = torch.randn(1, HIDDEN, device=w1.device, dtype=torch.bfloat16) / 10
    ids = torch.tensor(
        [[bad_id, 0, bad_id, bad_id]], device=w1.device, dtype=torch.int32
    )
    routed = torch.full((1, TOPK), 1 / TOPK, device=w1.device)
    config = tmp_path / f"direct-invalid-{bad_id}.csv"
    pd.DataFrame([_row("direct")]).to_csv(config, index=False)
    monkeypatch.setenv("AITER_CONFIG_FMOE", str(config))
    monkeypatch.setenv("AITER_SITUV2_A4W4", "1")
    _clear_config_caches()

    result = _call_situv2(x, w1, w2, routed, ids, s1, s2)
    expected = _reference(x, ids, routed, ref1, ref2)
    assert torch.isfinite(result).all()
    assert (
        F.cosine_similarity(result.float().flatten(), expected.flatten(), dim=0) > 0.9
    )

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = _call_situv2(x, w1, w2, routed, ids, s1, s2)
    for _ in range(100):
        graph.replay()
    torch.cuda.synchronize(w1.device)
    assert torch.isfinite(result).all()


def test_direct_uses_tensor_device_stream(monkeypatch, tmp_path, weights):
    if torch.cuda.device_count() < 2:
        pytest.skip("requires two HIP devices")
    ref1, ref2, w1, w2, s1, s2 = weights
    x = torch.randn(1, HIDDEN, device=w1.device, dtype=torch.bfloat16) / 10
    ids = torch.arange(TOPK, device=w1.device, dtype=torch.int32).view(1, -1)
    routed = torch.full((1, TOPK), 1 / TOPK, device=w1.device)
    config = tmp_path / "direct-device.csv"
    pd.DataFrame([_row("direct")]).to_csv(config, index=False)
    monkeypatch.setenv("AITER_CONFIG_FMOE", str(config))
    monkeypatch.setenv("AITER_SITUV2_A4W4", "1")
    _clear_config_caches()

    original_device = torch.cuda.current_device()
    other_device = 1 if w1.device.index != 1 else 0
    stream = torch.cuda.Stream(device=w1.device)
    try:
        torch.cuda.set_device(other_device)
        with torch.cuda.stream(stream):
            result = _call_situv2(x, w1, w2, routed, ids, s1, s2)
        stream.synchronize()
    finally:
        torch.cuda.set_device(original_device)
    expected = _reference(x, ids, routed, ref1, ref2)
    assert (
        F.cosine_similarity(result.float().flatten(), expected.flatten(), dim=0) > 0.9
    )


def test_other_dtype_path_keeps_legacy_silu_correct(monkeypatch, weights):
    ref1, ref2, _, _, _, _ = weights
    silu_w1 = ref1[:, :INTER].contiguous()
    m = 16
    x = torch.randn(m, HIDDEN, device=ref1.device, dtype=torch.bfloat16) / 10
    ids = torch.arange(TOPK, device=ref1.device, dtype=torch.int32).repeat(m, 1)
    routed = torch.zeros((m, TOPK), device=ref1.device)
    monkeypatch.setenv("AITER_BYPASS_TUNE_CONFIG", "1")
    _clear_config_caches()

    result = fused_moe.fused_moe(
        x,
        silu_w1,
        ref2,
        routed,
        ids,
        activation=ActivationType.Silu,
        quant_type=QuantType.No,
    )
    assert torch.isfinite(result).all()
    assert torch.count_nonzero(result) == 0
