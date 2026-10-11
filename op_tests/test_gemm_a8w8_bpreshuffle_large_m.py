# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

from aiter import dtypes
from aiter.ops import gemm_op_a8w8 as gemm_mod

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="a8w8 bpreshuffle large-M tests require a CUDA/HIP device",
)

TUNED = "bpreshuffle-test.csv"
KERNEL = "flydsl_bpreshuflle_128x256x128_F8_F8_B16_1x2x4x2_default"


def _row():
    return {"libtype": "flydsl", "splitK": 0, "kernelName": KERNEL, "kernelId": 727}


@pytest.fixture
def tuned_table(monkeypatch):
    gemm_mod._GEMM_QUANT_TYPE_CACHE[TUNED] = {
        ("gfx950", 256, 32768, 3072, 512, "torch.float8_e4m3fn"): _row(),
        ("gfx950", 256, 16, 3072, 512, "torch.float8_e4m3fn"): _row(),
    }
    gemm_mod._GEMM_QUANT_TYPE_HAS_GFX[TUNED] = True
    gemm_mod._largest_bpreshuffle_rows.cache_clear()
    gemm_mod._REUSED_BPRESUFFLE_SHAPES.clear()
    monkeypatch.setattr(gemm_mod, "get_gfx", lambda: "gfx950")
    monkeypatch.setattr(gemm_mod, "get_cu_num", lambda: 256)
    monkeypatch.setattr(
        type(gemm_mod.AITER_CONFIGS),
        "AITER_CONFIG_GEMM_A8W8_BPRESHUFFLE_FILE",
        property(lambda self: TUNED),
    )
    monkeypatch.setattr(
        gemm_mod,
        "get_GEMM_config_with_quant_type",
        lambda *args, **kwargs: None,
    )
    yield
    gemm_mod._GEMM_QUANT_TYPE_CACHE.pop(TUNED, None)
    gemm_mod._largest_bpreshuffle_rows.cache_clear()


def test_m_above_table_reuses_largest_row_and_chunks(tuned_table, monkeypatch):
    seen = []

    def fake_flydsl(XQ, WQ, x_scale, w_scale, Y, config):
        seen.append((XQ.shape[0], x_scale.shape[0], config["kernelName"]))
        return Y

    monkeypatch.setattr(gemm_mod, "gemm_a8w8_bpreshuffle_flydsl", fake_flydsl)
    rows = 70000
    xq = torch.zeros((rows, 512), device="cuda", dtype=dtypes.fp8)
    wq = torch.zeros((3072, 512), device="cuda", dtype=dtypes.fp8)
    x_scale = torch.ones((rows, 1), device="cuda", dtype=torch.float32)
    w_scale = torch.ones((3072, 1), device="cuda", dtype=torch.float32)

    out = gemm_mod.gemm_a8w8_bpreshuffle(xq, wq, x_scale, w_scale)

    assert out.shape == (rows, 3072)
    assert seen == [(65536, 65536, KERNEL), (4464, 4464, KERNEL)]


def test_gap_below_largest_tuned_m_stays_on_default_kernel(tuned_table, monkeypatch):
    seen = []
    monkeypatch.setattr(
        gemm_mod,
        "gemm_a8w8_bpreshuffle_ck",
        lambda *args: seen.append("ck") or args[4],
    )
    monkeypatch.setattr(
        gemm_mod,
        "gemm_a8w8_bpreshuffle_flydsl",
        lambda *args: seen.append("flydsl") or args[4],
    )
    xq = torch.zeros((1000, 512), device="cuda", dtype=dtypes.fp8)
    wq = torch.zeros((3072, 512), device="cuda", dtype=dtypes.fp8)
    x_scale = torch.ones((1000, 1), device="cuda", dtype=torch.float32)
    w_scale = torch.ones((3072, 1), device="cuda", dtype=torch.float32)

    gemm_mod.gemm_a8w8_bpreshuffle(xq, wq, x_scale, w_scale)

    assert seen == ["ck"]
