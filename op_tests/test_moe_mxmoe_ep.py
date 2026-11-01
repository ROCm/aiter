# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Expert-parallel (expert_mask) coverage for the MXMOE a4w4 GEMM1 path.

An EP rank keys the tuned lookup on its local expert count, which is not one of
the generated MXFP4 aux shapes. The MXMOE path must still run there: the masked
Opus aux sort emits local expert ids, so no NE-specialized aux kernel is used.
"""

import contextlib
import importlib

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


def _row(gfx, cu_num, token, expert, topk, kn1, kn2):
    bm = fused_moe._parse_mxfp4_g1_kname(kn1)["BM"]
    return (
        f"{gfx},{cu_num},{token},{MODEL_DIM},{INTER_DIM},{expert},{topk},"
        "ActivationType.Silu,torch.bfloat16,torch.float4_e2m1fn_x2,"
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
