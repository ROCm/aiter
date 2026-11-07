# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import csv
from pathlib import Path

import pytest
import torch

import aiter
from aiter import dtypes
from aiter.fused_moe import fused_moe, get_2stage_cfgs, get_padded_M
from aiter.jit.utils.chip_info import get_cu_num, get_gfx
from aiter.ops.enum import ActivationType, QuantType
from aiter.ops.flydsl.moe_common import GateMode
from aiter.ops.moe_sorting_opus import moe_sorting_opus_get_workspace_size
from aiter.ops.shuffle import shuffle_weight

_REPO_ROOT = Path(__file__).resolve().parents[1]
_BF16_G1U1_FMOE_CONFIG = (
    _REPO_ROOT
    / "aiter/configs/model_configs/qwen3_6_35b_a3b_bf16_tuned_fmoe_gfx1201.csv"
)
_OPUS_SORTING_SOURCE = _REPO_ROOT / "csrc/py_itfs_cu/moe_sorting_opus_kernels.cu"


def _bf16_g1u1_rows():
    with _BF16_G1U1_FMOE_CONFIG.open(newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def _make_balanced_topk(token: int, topk: int, expert: int):
    rows = torch.arange(token, device="cuda", dtype=torch.int32)[:, None]
    slots = torch.arange(topk, device="cuda", dtype=torch.int32)[None, :]
    topk_ids = (rows * topk + slots) % expert
    topk_weights = torch.rand((token, topk), dtype=torch.float32, device="cuda")
    topk_weights /= topk_weights.sum(dim=-1, keepdim=True)
    return topk_ids.contiguous(), topk_weights.contiguous()


def test_bf16_g1u1_fmoe_rows_use_runtime_cu_convention():
    rows = _bf16_g1u1_rows()
    assert rows
    assert {row["gfx"] for row in rows} == {"gfx1201"}
    assert {row["cu_num"] for row in rows} == {"64"}


def test_bf16_g1u1_fmoe_rows_cover_padded_m_4096():
    rows = _bf16_g1u1_rows()
    tokens_by_inter_dim = {}
    for row in rows:
        assert row["model_dim"] == "2048"
        assert row["expert"] == "256"
        assert row["topk"] == "8"
        assert row["act_type"] == "ActivationType.Silu"
        assert row["dtype"] == "torch.bfloat16"
        assert row["q_dtype_a"] == "torch.bfloat16"
        assert row["q_dtype_w"] == "torch.bfloat16"
        assert row["q_type"] == "QuantType.No"
        assert row["use_g1u1"] == "1"
        assert row["doweight_stage1"] == "0"
        tokens_by_inter_dim.setdefault(int(row["inter_dim"]), set()).add(
            int(row["token"])
        )

    assert tokens_by_inter_dim[128] == {32, 64, 128, 256, 512, 1024, 2048, 4096}
    assert tokens_by_inter_dim[256] == {32, 64, 128, 256, 512, 1024, 2048}

    row_4096 = [
        row for row in rows if row["inter_dim"] == "128" and row["token"] == "4096"
    ]
    assert len(row_4096) == 1
    assert row_4096[0]["kernelName1"]
    assert row_4096[0]["kernelName2"]


def test_moe_sorting_opus_dpp_scan_scopes_gfx1201_replacement():
    source = _OPUS_SORTING_SOURCE.read_text()
    start = source.index("__device__ __forceinline__ int dpp_inclusive_scan_wave")
    end = source.index("template <int NumCtas>", start)
    helper = source[start:end]

    assert "#if defined(__gfx1201__)" in helper
    assert helper.index("__builtin_amdgcn_ds_bpermute") > helper.index(
        "#if defined(__gfx1201__)"
    )
    assert helper.index("0x142") > helper.index("#else")
    assert helper.index("0x143") > helper.index("#else")
    assert "__builtin_amdgcn_ds_bpermute" in helper


@pytest.mark.skipif(
    get_gfx() != "gfx1201",
    reason="gfx1201 CK2Stage BF16 runtime coverage",
)
def test_bf16_g1u1_ck2stage_4096_bucket_runs():
    torch.manual_seed(4815)
    token = 2050
    model_dim = 2048
    inter_dim = 128
    expert = 256
    topk = 8

    assert get_cu_num() == 64
    assert get_padded_M(token) == 4096
    assert moe_sorting_opus_get_workspace_size(token, expert, topk, 0, 0) >= 0

    metadata = get_2stage_cfgs(
        get_padded_M(token),
        model_dim,
        inter_dim,
        expert,
        topk,
        dtypes.bf16,
        dtypes.bf16,
        dtypes.bf16,
        QuantType.No,
        True,
        ActivationType.Silu,
        False,
        0,
        0,
        True,
        GateMode.SEPARATED.value,
        opus_weights_shuffled=True,
        input_dtype=dtypes.bf16,
    )
    assert metadata.block_m == 64
    assert metadata.stage1.keywords["kernelName"].startswith("moe_ck2stages_gemm1_")
    assert metadata.stage2.keywords["kernelName"].startswith("moe_ck2stages_gemm2_")

    hidden_states = torch.randn((token, model_dim), dtype=dtypes.bf16, device="cuda")
    w1 = torch.randn(
        (expert, inter_dim * 2, model_dim), dtype=dtypes.bf16, device="cuda"
    )
    w2 = torch.randn((expert, model_dim, inter_dim), dtype=dtypes.bf16, device="cuda")
    topk_ids, topk_weight = _make_balanced_topk(token, topk, expert)

    actual = fused_moe(
        hidden_states,
        shuffle_weight(w1, layout=(16, 16)),
        shuffle_weight(w2, layout=(16, 16)),
        topk_weight,
        topk_ids,
        activation=aiter.ActivationType.Silu,
        quant_type=aiter.QuantType.No,
        dtype=dtypes.bf16,
        gate_mode=GateMode.SEPARATED.value,
    )

    assert actual.shape == (token, model_dim)
    assert actual.dtype == dtypes.bf16
    assert torch.isfinite(actual).all()
