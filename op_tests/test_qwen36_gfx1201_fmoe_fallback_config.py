# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import csv
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[1]
_QWEN36_GFX1201_BF16_FMOE = (
    _REPO_ROOT
    / "aiter/configs/model_configs/qwen3_6_35b_a3b_bf16_tuned_fmoe_gfx1201.csv"
)
_OPUS_SORTING_SOURCE = _REPO_ROOT / "csrc/py_itfs_cu/moe_sorting_opus_kernels.cu"


def _qwen36_rows():
    with _QWEN36_GFX1201_BF16_FMOE.open(newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def test_qwen36_gfx1201_bf16_fmoe_rows_use_runtime_cu_convention():
    rows = _qwen36_rows()
    assert rows
    assert {row["gfx"] for row in rows} == {"gfx1201"}
    assert {row["cu_num"] for row in rows} == {"64"}


def test_qwen36_gfx1201_bf16_fmoe_rows_cover_padded_m_4096():
    rows = _qwen36_rows()
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
        row
        for row in rows
        if row["inter_dim"] == "128" and row["token"] == "4096"
    ]
    assert len(row_4096) == 1
    assert row_4096[0]["kernelName1"]
    assert row_4096[0]["kernelName2"]


def test_moe_sorting_opus_dpp_scan_avoids_gfx1201_row_broadcast_controls():
    source = _OPUS_SORTING_SOURCE.read_text()
    assert "0x142" not in source
    assert "0x143" not in source
    assert "__builtin_amdgcn_ds_bpermute" in source
