# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Expert-parallel (expert_mask) coverage for the MXMOE a4w4 GEMM1 path.

An EP rank keys the tuned lookup on its local expert count, which is not one of
the generated MXFP4 aux shapes. The MXMOE path must still run there: the masked
Opus aux sort emits local expert ids, so no NE-specialized aux kernel is used.
"""

import contextlib
import importlib
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

import aiter
from aiter import ActivationType, QuantType, dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.moe_common import GateMode

fused_moe = importlib.import_module("aiter.fused_moe")

MODEL_DIM = 4096
INTER_DIM = 256
ROUTED_EXPERTS = 32
EP = 4
EP_RANK = 1
LOCAL_ROUTED = ROUTED_EXPERTS // EP
# One fused shared expert per rank, as ATOM passes GLM-5 under EP.
LOCAL_EXPERTS = LOCAL_ROUTED + 1
SHARED_ID = ROUTED_EXPERTS
ROUTED_TOPK = 8
TOPK = ROUTED_TOPK + 1

HEADER = (
    "gfx,cu_num,token,model_dim,inter_dim,expert,topk,act_type,dtype,q_dtype_a,"
    "q_dtype_w,q_type,use_g1u1,doweight_stage1,block_m,ksplit,us1,kernelName1,"
    "err1,us2,kernelName2,err2,us,run_1stage,xbf16,flat,tflops,bw,_tag"
)

CASES = {
    "bm16_atomic": (
        "flydsl_mxmoe_g1_a4w4_16x256x256_f16in_nt",
        "flydsl_moe2_layout_afp4_wfp4_bf16_t16x128x256_atomic_persist_nt_sbm16",
    ),
    "bm16_reduce": (
        "flydsl_mxmoe_g1_a4w4_16x256x256_f16in_nt",
        "flydsl_moe2_layout_afp4_wfp4_bf16_t16x128x256_reduce_persist_nt_sbm16",
    ),
    "bm64_reduce": (
        "flydsl_mxmoe_g1_a4w4_64x128x256_nt_xcd4",
        "flydsl_moe2_layout_afp4_wfp4_bf16_t64x128x256_reduce_sbm64",
    ),
    "bm128_scatter": (
        "flydsl_mxmoe_g1_a4w4_128x256x256",
        "flydsl_moe2_layout_afp4_wfp4_bf16_t128x256x256_scatter_sbm128",
    ),
}


def _row(
    gfx,
    cu_num,
    token,
    expert,
    topk,
    kn1,
    kn2,
    *,
    model_dim=MODEL_DIM,
    inter_dim=INTER_DIM,
    act="Silu",
    q_dtype_a="float4_e2m1fn_x2",
):
    bm = fused_moe._parse_mxfp4_g1_kname(kn1)["BM"]
    return (
        f"{gfx},{cu_num},{token},{model_dim},{inter_dim},{expert},{topk},"
        f"ActivationType.{act},torch.bfloat16,torch.{q_dtype_a},"
        "torch.float4_e2m1fn_x2,QuantType.per_1x32,1,0,"
        f"{bm},0,0,{kn1},0,0,{kn2},0,0,0,0,0,0,0,"
    )


@pytest.fixture
def tuned_rows(monkeypatch, tmp_path):
    """Install tuned rows (as CSV lines) as the only fused_moe config."""

    def install(rows):
        path = tmp_path / "tuned_fmoe.csv"
        path.write_text("\n".join([HEADER, *rows]) + "\n")
        monkeypatch.setenv("AITER_CONFIG_FMOE", str(path))
        monkeypatch.setattr(fused_moe, "cfg_2stages", None)
        fused_moe.AITER_CONFIGS.get_config_file.cache_clear()
        fused_moe.get_2stage_cfgs.cache_clear()

    yield install
    fused_moe.get_2stage_cfgs.cache_clear()
    fused_moe.AITER_CONFIGS.get_config_file.cache_clear()


def _dispatch(topk=TOPK, is_ep=True):
    return fused_moe.get_2stage_cfgs(
        16,
        MODEL_DIM,
        INTER_DIM,
        LOCAL_EXPERTS,
        topk,
        dtypes.bf16,
        dtypes.fp4x2,
        dtypes.fp4x2,
        QuantType.per_1x32,
        True,
        ActivationType.Silu,
        False,
        0,
        0,
        is_shuffled=True,
        opus_weights_shuffled=True,
        is_ep=is_ep,
    )


def _is_mxmoe(meta, kn1, kn2):
    return (
        meta.stage1.func is fused_moe._mxfp4_a4w4_stage1_fw
        and meta.stage1.keywords["kernelName1"] == kn1
        and meta.stage2.keywords["kernelName2"] == kn2
    )


@pytest.fixture
def gfx950_dispatch(monkeypatch, tuned_rows):
    monkeypatch.setattr(fused_moe, "get_cu_num", lambda: 256)
    monkeypatch.setattr(fused_moe, "get_gfx_runtime", lambda: "gfx950")
    return lambda rows: tuned_rows(
        [_row("gfx950", 256, 16, LOCAL_EXPERTS, *r) for r in rows]
    )


@pytest.mark.parametrize("case", CASES)
def test_ep_keeps_mxmoe_row_for_local_expert_count(gfx950_dispatch, case):
    kn1, kn2 = CASES[case]
    gfx950_dispatch([(TOPK, kn1, kn2)])
    assert _is_mxmoe(_dispatch(), kn1, kn2)


def test_ep_rejects_native_mxmoe_gemm2(gfx950_dispatch):
    kn1 = "flydsl_mxmoe_g1_a4w4_64x128x256_nt_xcd4"
    kn2 = "flydsl_mxmoe_g2_a4w4_64x256x256_atomic"
    gfx950_dispatch([(TOPK, kn1, kn2)])
    assert _dispatch().stage1.func is not fused_moe._mxfp4_a4w4_stage1_fw


@pytest.mark.parametrize(
    ("case", "expect_mxmoe"),
    [("bm128_scatter", False), ("bm16_atomic", True)],
)
def test_ep_only_scatter_needs_a_scatter_reduce_instance(
    gfx950_dispatch, case, expect_mxmoe
):
    # No (MODEL_DIM, topk=7) scatter_reduce instance is generated; only the
    # scatter epilog launches it.
    assert not any(
        h == MODEL_DIM and t == 7
        for _, h, _, t in aiter.ops.moe_mxfp4_aux.MXFP4_MOE_SUPPORTED_SHAPES
    )
    kn1, kn2 = CASES[case]
    gfx950_dispatch([(7, kn1, kn2)])
    assert _is_mxmoe(_dispatch(topk=7), kn1, kn2) is expect_mxmoe


def _mxfp4(w):
    torch_quant = aiter.get_torch_quant(QuantType.per_1x32)
    w_qt, w_scale = torch_quant(w, quant_dtype=dtypes.fp4x2)
    return w_qt.view(w.shape[0], w.shape[1], w.shape[2] // 2), w_scale


def _shuffled(w_qt, w_scale):
    from aiter.ops.shuffle import shuffle_weight
    from aiter.utility import fp4_utils

    w = shuffle_weight(w_qt, layout=(16, 16))
    w.is_shuffled = True
    return w, fp4_utils.e8m0_shuffle(w_scale)


def _routing(rows):
    """Global routed ids spanning every rank plus the shared-expert column."""
    routed = torch.stack(
        [torch.randperm(ROUTED_EXPERTS)[:ROUTED_TOPK] for _ in range(rows)]
    ).to(torch.int32)
    shared = torch.full((rows, 1), SHARED_ID, dtype=torch.int32)
    ids = torch.cat([routed, shared], dim=1).cuda()
    weights = torch.rand((rows, TOPK), dtype=torch.float32, device="cuda")
    return ids, weights


@contextlib.contextmanager
def _nan_filled_empty():
    # Deterministic mode makes torch.empty NaN-fill new buffers
    # (torch.utils.deterministic.fill_uninitialized_memory), so a route slot a
    # kernel leaves unwritten cannot pass on a lucky zero page.
    prev = torch.are_deterministic_algorithms_enabled()
    prev_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True, warn_only=True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(prev, warn_only=prev_warn_only)


def _local_ids(ids):
    lo = EP_RANK * LOCAL_ROUTED
    local = torch.where((ids >= lo) & (ids < lo + LOCAL_ROUTED), ids - lo, -1)
    return torch.where(ids == SHARED_ID, LOCAL_ROUTED, local)


def _reference(x, w1, w2, ids, weights):
    """A4W4 torch reference over the local experts (remote routes are -1)."""
    from aiter.ops.quant import per_1x32_f4_quant

    (w1_qt, w1_scale), (w2_qt, w2_scale) = w1, w2
    local = _local_ids(ids)
    a1, a1_scale = per_1x32_f4_quant(x, quant_dtype=dtypes.fp4x2, shuffle=False)
    a2 = fused_moe.torch_moe_stage1(
        a1,
        w1_qt,
        w2_qt,
        weights,
        local,
        dtype=dtypes.bf16,
        activation=ActivationType.Silu,
        quant_type=QuantType.per_1x32,
        a1_scale=a1_scale,
        w1_scale=w1_scale,
    )
    rows = x.shape[0]
    a2, a2_scale = per_1x32_f4_quant(
        a2.view(rows * TOPK, INTER_DIM), quant_dtype=dtypes.fp4x2, shuffle=False
    )
    return fused_moe.torch_moe_stage2(
        a2.view(rows, TOPK, -1),
        w1_qt,
        w2_qt,
        weights,
        local,
        dtype=dtypes.bf16,
        quant_type=QuantType.per_1x32,
        w2_scale=w2_scale,
        a2_scale=a2_scale,
        doweight=True,
    )


@pytest.mark.skipif(get_gfx() != "gfx950", reason="MXMOE a4w4 requires gfx950")
@pytest.mark.parametrize("ep_mode", ["real", "fake"])
@pytest.mark.parametrize("betas", [(), ("--beta", "2.0", "--linear-beta", "12.0")])
def test_ep_cli_a4w4_situv2_reference(tuned_rows, ep_mode, betas):
    root = Path(__file__).resolve().parents[1]
    kn1, kn2 = CASES["bm16_atomic"]
    kn1 += "_situv2"
    tuned_rows(
        [
            _row(
                get_gfx(),
                fused_moe.get_cu_num(),
                32,
                LOCAL_ROUTED,
                TOPK if ep_mode == "real" else ROUTED_TOPK,
                kn1,
                kn2,
                act="Situv2",
            )
        ]
    )
    env = {
        **os.environ,
        "PYTHONPATH": str(root),
        "AITER_META_DIR": str(root),
        "AITER_LOG_LEVEL": "INFO",
        "AITER_BYPASS_TUNE_CONFIG": "0",
        "AITER_ONLINE_TUNE": "0",
        "AITER_SMI_MONITOR": "0",
        # The CLI must select A4W4 even if it inherits an A8W4 request.
        "AITER_SITUV2_A8W4": "1",
        "AITER_SITUV2_A4W4": "0",
    }
    result = subprocess.run(
        [
            sys.executable,
            "op_tests/test_moe_ep.py",
            "-t",
            "g1u1_a4w4_mxfp4",
            "-a",
            "situv2",
            "-hd",
            str(MODEL_DIM),
            "-id",
            str(INTER_DIM),
            "-e",
            str(ROUTED_EXPERTS),
            "-k",
            str(ROUTED_TOPK),
            "-ep",
            str(EP),
            "-m",
            "32",
            "--ep-mode",
            ep_mode,
            *betas,
        ],
        cwd=root,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=300,
        check=False,
    )
    assert result.returncode == 0, result.stdout
    assert any(
        "[fused_moe] using 2stage (" in line and kn1 in line and kn2 in line
        for line in result.stdout.splitlines()
    ), result.stdout
    assert "logits_diff=" in result.stdout and "PASSED" in result.stdout, result.stdout


@pytest.mark.skipif(get_gfx() != "gfx950", reason="MXMOE a4w4 requires gfx950")
@pytest.mark.parametrize("dead_tail", [False, True], ids=["dense", "dead_tail"])
@pytest.mark.parametrize("case", CASES)
def test_ep_mxmoe_matches_quantized_reference(monkeypatch, tuned_rows, case, dead_tail):
    kn1, kn2 = CASES[case]
    valid = 24 if fused_moe._parse_mxfp4_g1_kname(kn1)["BM"] == 16 else 200
    # Under MORI dispatch only the first num_local_tokens rows are live; the
    # tail is stale arena data that must not reach the live rows.
    rows = valid + (valid // 3 if dead_tail else 0)
    tuned_rows(
        [
            _row(
                fused_moe.get_gfx_runtime(),
                fused_moe.get_cu_num(),
                fused_moe.get_padded_M(rows),
                LOCAL_EXPERTS,
                TOPK,
                kn1,
                kn2,
            )
        ]
    )

    torch.manual_seed(0)
    w1 = _mxfp4(
        torch.randn((LOCAL_EXPERTS, 2 * INTER_DIM, MODEL_DIM), device="cuda") * 0.1
    )
    w2 = _mxfp4(torch.randn((LOCAL_EXPERTS, MODEL_DIM, INTER_DIM), device="cuda") * 0.1)
    x = torch.randn((rows, MODEL_DIM), dtype=dtypes.bf16, device="cuda")
    ids, weights = _routing(rows)
    expert_mask = torch.zeros(ROUTED_EXPERTS + 1, dtype=torch.int32, device="cuda")
    expert_mask[EP_RANK * LOCAL_ROUTED : (EP_RANK + 1) * LOCAL_ROUTED] = 1
    expert_mask[SHARED_ID] = 1

    calls = []
    monkeypatch.setattr(fused_moe, "kernel_bench_callable", calls)
    stage2 = {}

    def nan_filled_stage2(ordinary_stage2, stage2_args, stage2_kwargs):
        # Stage1 allocates fp4x2 buffers, which torch cannot NaN-fill.
        stage2["func"] = ordinary_stage2
        with _nan_filled_empty():
            return ordinary_stage2(*stage2_args, **stage2_kwargs)

    w1_shuf, w1_scale = _shuffled(*w1)
    w2_shuf, w2_scale = _shuffled(*w2)
    out = fused_moe._fused_moe_impl(
        x,
        w1_shuf,
        w2_shuf,
        weights,
        ids,
        expert_mask=expert_mask,
        activation=ActivationType.Silu,
        quant_type=QuantType.per_1x32,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        num_local_tokens=(
            torch.tensor([valid], dtype=torch.int32, device="cuda")
            if dead_tail
            else None
        ),
        gate_mode=GateMode.SEPARATED.value,
        _stage2_override=nan_filled_stage2,
    )
    stage1 = dict(calls)["stage1"]
    assert stage1.func is fused_moe._mxfp4_a4w4_stage1_fw
    assert stage1.keywords["kernelName1"] == kn1
    assert stage2["func"].keywords["kernelName2"] == kn2

    ref = _reference(x[:valid], w1, w2, ids[:valid], weights[:valid]).float()
    got = out[:valid].float()
    assert torch.isfinite(got).all()
    # A4W4 rounding alone reaches ~0.14 on rows served by one local expert; a
    # dropped or foreign route moves a row by a whole expert's share.
    row_err = (got - ref).norm(dim=1) / ref.norm(dim=1)
    assert row_err.max().item() < 0.25, row_err.max().item()


# A8W4 SiTUv2 shaped like Kimi K3 under EP: no shared expert, and the extra
# routing column always points at the masked expert.
A8W4_MODEL_DIM = 3584
A8W4_INTER_DIM = 3072
A8W4_MASKED_ID = ROUTED_EXPERTS
A8W4_TOPK = ROUTED_TOPK + 1

A8W4_CASES = {
    "bm32_atomic": (
        "flydsl_mxmoe_g1_a8w4_32x256x256_fp8out_situv2",
        "flydsl_moe2_layout_afp8_wfp4_bf16_t32x256x256_atomic_nt_sbm32",
    ),
    "bm32_reduce": (
        "flydsl_mxmoe_g1_a8w4_32x256x256_fp8out_situv2",
        "flydsl_moe2_layout_afp8_wfp4_bf16_t32x256x128_reduce_sbm32",
    ),
    "bm64_atomic": (
        "flydsl_mxmoe_g1_a8w4_64x256x256_fp8out_situv2",
        "flydsl_moe2_layout_afp8_wfp4_bf16_t64x256x128_atomic_nt_sbm64",
    ),
    "bm64_reduce": (
        "flydsl_mxmoe_g1_a8w4_64x256x256_fp8out_situv2",
        "flydsl_moe2_layout_afp8_wfp4_bf16_t64x256x128_reduce_nt_sbm64",
    ),
    "bm128_atomic": (
        "flydsl_mxmoe_g1_a8w4_128x256x256_fp8out_situv2",
        "flydsl_moe2_layout_afp8_wfp4_bf16_t128x256x128_atomic_nt_sbm128",
    ),
    "bm128_reduce": (
        "flydsl_mxmoe_g1_a8w4_128x256x256_fp8out_situv2",
        "flydsl_moe2_layout_afp8_wfp4_bf16_t128x256x128_reduce_nt_sbm128",
    ),
}


def _a8w4_routing(rows):
    """Global routed ids with at least one local route per row, plus the masked column.

    MORI only delivers a token to a rank that owns one of its routes.
    """
    lo = EP_RANK * LOCAL_ROUTED
    local = lo + torch.randint(LOCAL_ROUTED, (rows, 1))
    others = torch.stack(
        [torch.randperm(ROUTED_EXPERTS - 1)[: ROUTED_TOPK - 1] for _ in range(rows)]
    )
    others = others + (others >= local).long()
    masked = torch.full((rows, 1), A8W4_MASKED_ID)
    ids = torch.cat([local, others, masked], dim=1).to(torch.int32).cuda()
    weights = torch.rand((rows, A8W4_TOPK), dtype=torch.float32, device="cuda")
    return ids, weights


def _a8w4_reference(x, w1, w2, ids, weights):
    """A8W4 SiTUv2 torch reference over the local experts (remote routes are -1)."""
    from aiter.ops.flydsl.moe_common import (
        DEFAULT_SITUV2_BETA,
        DEFAULT_SITUV2_LINEAR_BETA,
    )
    from aiter.ops.quant import per_1x32_f8_scale_f8_quant

    def quant(t):
        return per_1x32_f8_scale_f8_quant(
            t, quant_dtype=dtypes.fp8, scale_type=dtypes.fp8_e8m0
        )

    (w1_qt, w1_scale), (w2_qt, w2_scale) = w1, w2
    lo = EP_RANK * LOCAL_ROUTED
    local = torch.where((ids >= lo) & (ids < lo + LOCAL_ROUTED), ids - lo, -1)
    a1, a1_scale = quant(x)
    a2 = fused_moe.torch_moe_stage1(
        a1,
        w1_qt,
        w2_qt,
        weights,
        local,
        dtype=dtypes.bf16,
        activation=ActivationType.Situv2,
        quant_type=QuantType.per_1x32,
        a1_scale=a1_scale,
        w1_scale=w1_scale,
        situ_beta=DEFAULT_SITUV2_BETA,
        situ_linear_beta=DEFAULT_SITUV2_LINEAR_BETA,
    )
    rows = x.shape[0]
    a2, a2_scale = quant(a2.contiguous().view(rows * A8W4_TOPK, A8W4_INTER_DIM))
    return fused_moe.torch_moe_stage2(
        a2.view(rows, A8W4_TOPK, -1),
        w1_qt,
        w2_qt,
        weights,
        local,
        dtype=dtypes.bf16,
        quant_type=QuantType.per_1x32,
        w2_scale=w2_scale,
        a2_scale=a2_scale,
        doweight=True,
    )


@pytest.mark.skipif(get_gfx() != "gfx950", reason="MXMOE a8w4 requires gfx950")
@pytest.mark.parametrize("dead_tail", [False, True], ids=["dense", "dead_tail"])
@pytest.mark.parametrize("case", A8W4_CASES)
def test_ep_mxmoe_a8w4_matches_quantized_reference(
    monkeypatch, tuned_rows, case, dead_tail
):
    from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight_a16w4

    kn1, kn2 = A8W4_CASES[case]
    valid = 200
    rows = valid + (valid // 3 if dead_tail else 0)
    tuned_rows(
        [
            _row(
                fused_moe.get_gfx_runtime(),
                fused_moe.get_cu_num(),
                fused_moe.get_padded_M(rows),
                LOCAL_ROUTED,
                A8W4_TOPK,
                kn1,
                kn2,
                model_dim=A8W4_MODEL_DIM,
                inter_dim=A8W4_INTER_DIM,
                act="Situv2",
                q_dtype_a="float8_e4m3fn",
            )
        ]
    )
    # SiTUv2 otherwise quantizes activations to bf16 (a16w4).
    monkeypatch.setenv("AITER_SITUV2_A8W4", "1")

    torch.manual_seed(0)
    w1 = _mxfp4(
        torch.randn((LOCAL_ROUTED, 2 * A8W4_INTER_DIM, A8W4_MODEL_DIM), device="cuda")
        * 0.1
    )
    w2 = _mxfp4(
        torch.randn((LOCAL_ROUTED, A8W4_MODEL_DIM, A8W4_INTER_DIM), device="cuda") * 0.1
    )
    x = torch.randn((rows, A8W4_MODEL_DIM), dtype=dtypes.bf16, device="cuda")
    ids, weights = _a8w4_routing(rows)
    expert_mask = torch.zeros(ROUTED_EXPERTS + 1, dtype=torch.int32, device="cuda")
    expert_mask[EP_RANK * LOCAL_ROUTED : (EP_RANK + 1) * LOCAL_ROUTED] = 1

    calls = []
    monkeypatch.setattr(fused_moe, "kernel_bench_callable", calls)
    stage2 = {}

    def nan_filled_stage2(ordinary_stage2, stage2_args, stage2_kwargs):
        stage2["func"] = ordinary_stage2
        with _nan_filled_empty():
            return ordinary_stage2(*stage2_args, **stage2_kwargs)

    out = fused_moe._fused_moe_impl(
        x,
        shuffle_weight_a16w4(w1[0], 16, True),
        shuffle_weight_a16w4(w2[0], 16, False),
        weights,
        ids,
        expert_mask=expert_mask,
        activation=ActivationType.Situv2,
        quant_type=QuantType.per_1x32,
        w1_scale=shuffle_scale_a16w4(w1[1], LOCAL_ROUTED, True),
        w2_scale=shuffle_scale_a16w4(w2[1], LOCAL_ROUTED, False),
        num_local_tokens=(
            torch.tensor([valid], dtype=torch.int32, device="cuda")
            if dead_tail
            else None
        ),
        gate_mode=GateMode.INTERLEAVE.value,
        _stage2_override=nan_filled_stage2,
    )
    stage1 = dict(calls)["stage1"]
    assert stage1.func is fused_moe._mxfp4_a4w4_stage1_fw
    assert stage1.keywords["kernelName1"] == kn1
    assert stage2["func"].keywords["kernelName2"] == kn2

    ref = _a8w4_reference(x[:valid], w1, w2, ids[:valid], weights[:valid]).float()
    got = out[:valid].float()
    assert torch.isfinite(got).all()
    # FP8 rounding alone stays near 0.005; a dropped or foreign route moves a
    # row by a whole expert's share.
    row_err = (got - ref).norm(dim=1) / ref.norm(dim=1)
    assert row_err.max().item() < 0.02, row_err.max().item()
