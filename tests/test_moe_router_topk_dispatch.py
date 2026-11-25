# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

from aiter.jit.utils import chip_info
from aiter.ops import gemm_op_common
from aiter.ops.flydsl import moe_front_gfx1250 as front


def _on(monkeypatch, gfx, cu_num=256):
    monkeypatch.setattr(chip_info, "get_gfx_runtime", lambda: gfx)
    monkeypatch.setattr(chip_info, "get_cu_num", lambda: cu_num)
    monkeypatch.setattr(gemm_op_common, "get_padded_m", lambda m, n, k, gl: m)
    front.router_topk_config.cache_clear()
    front.router_gemm_split_k.cache_clear()


def test_router_topk_dispatch(monkeypatch):
    _on(monkeypatch, "gfx1250")
    assert front.router_topk_config(1, 128, 6144, 4) is not None
    # tgemm.mm + topk_gating stays: an unfused tuned row, group counts, another arch
    assert front.router_topk_config(320, 128, 6144, 4) is None
    assert front.router_topk_config(1, 128, 6144, 4, with_counts=True) is None
    _on(monkeypatch, "gfx950")
    assert front.router_topk_config(1, 128, 6144, 4) is None
