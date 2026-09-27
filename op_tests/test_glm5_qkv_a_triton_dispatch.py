# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Dispatch coverage for the GLM-5 qkv_a_proj gfx950 tuning rows."""

import json
from pathlib import Path

import pytest
import torch

from aiter import dtypes
from aiter.ops import gemm_op_a8w8 as gemm_mod
from aiter.ops.triton.gemm.basic import gemm_a8w8_blockscale as triton_mod
from aiter.ops.triton.utils import gemm_config_utils as triton_config_mod

TUNED_FILE = str(
    Path(gemm_mod.__file__).resolve().parents[1]
    / "configs/model_configs/glm5_a8w8_blockscale_bpreshuffle_tuned_gemm.csv"
)
TRITON_CONFIG = (
    Path(triton_mod.__file__).resolve().parents[2]
    / "configs/gfx950/triton/gemm/gemm_a8w8_blockscale_preshuffled"
    / "GEMM-A8W8_BLOCKSCALE_PRESHUFFLED-N=2624-K=6144.json"
)
GFX = "gfx950"
CU_NUM = 256
N = 2624
K = 6144


def _clear_config_cache():
    gemm_mod.get_CKGEMM_config.cache_clear()
    gemm_mod._CKGEMM_CONFIG_CACHE.pop(TUNED_FILE, None)
    gemm_mod._CKGEMM_HAS_GFX.pop(TUNED_FILE, None)
    triton_config_mod._get_gemm_config_cached.cache_clear()


@pytest.fixture(autouse=True)
def _gfx950_config(monkeypatch):
    monkeypatch.setattr(gemm_mod, "get_gfx", lambda: GFX)
    monkeypatch.setattr(gemm_mod, "get_cu_num", lambda: CU_NUM)
    monkeypatch.setattr(triton_config_mod.arch_info, "get_arch", lambda: GFX)
    _clear_config_cache()
    yield
    _clear_config_cache()


@pytest.mark.parametrize(
    ("m", "expected_row", "json_bucket"),
    [
        (1, 1, "M_LEQ_4"),
        (2, 16, "M_LEQ_4"),
        (4, 4, "M_LEQ_4"),
        (8, 8, "M_LEQ_8"),
        (16, 16, "M_LEQ_16"),
        (32, 32, "M_LEQ_32"),
        (33, 64, "M_LEQ_64"),
        (48, 64, "M_LEQ_64"),
        (64, 64, "M_LEQ_64"),
    ],
)
def test_glm5_qkv_a_decode_routes_to_expected_triton_tier(
    monkeypatch, m, expected_row, json_bucket
):
    expected = gemm_mod.get_CKGEMM_config(expected_row, N, K, TUNED_FILE)
    assert expected["libtype"] == "triton"
    expected_json = json.loads(TRITON_CONFIG.read_text())[json_bucket]

    selected_csv = {}
    selected = {}
    get_config = gemm_mod.get_CKGEMM_config

    def capture_csv(*args, **kwargs):
        config = get_config(*args, **kwargs)
        selected_csv.update(config)
        return config

    def fake_triton(*args, backend=None, **kwargs):
        config, is_tuned = triton_mod._get_config(
            m, N, K, shuffle=True, backend=backend or "triton"
        )
        selected.update(config)
        selected["is_tuned"] = is_tuned
        return kwargs["y"]

    monkeypatch.setattr(gemm_mod, "get_CKGEMM_config", capture_csv)
    monkeypatch.setattr(
        triton_mod,
        "gemm_a8w8_blockscale_preshuffle",
        fake_triton,
    )
    monkeypatch.setattr(
        gemm_mod,
        "gemm_a8w8_blockscale_bpreshuffle_ck",
        lambda *args, **kwargs: pytest.fail("decode shape routed to CK"),
    )

    xq, wq, x_scale, w_scale = _make_meta_inputs(m)
    gemm_mod.gemm_a8w8_blockscale_bpreshuffle(xq, wq, x_scale, w_scale)

    assert selected_csv == expected
    assert selected == expected_json | {"is_tuned": True}


def _make_meta_inputs(m):
    xq = torch.empty((m, K), dtype=dtypes.fp8, device="meta")
    wq = torch.empty((N, K), dtype=dtypes.fp8, device="meta")
    x_scale = torch.empty((K // 128, m), dtype=torch.float32, device="meta")
    w_scale = torch.empty((N // 128, K // 128), dtype=torch.float32, device="meta")
    return xq, wq, x_scale, w_scale


def test_glm5_qkv_a_prefill_control_stays_on_ck(monkeypatch):
    reached = []
    monkeypatch.setattr(
        gemm_mod,
        "gemm_a8w8_blockscale_bpreshuffle_ck",
        lambda *args, **kwargs: reached.append("ck") or args[4],
    )
    monkeypatch.setattr(
        triton_mod,
        "gemm_a8w8_blockscale_preshuffle",
        lambda *args, **kwargs: pytest.fail("control shape routed to Triton"),
    )

    xq, wq, x_scale, w_scale = _make_meta_inputs(128)
    gemm_mod.gemm_a8w8_blockscale_bpreshuffle(xq, wq, x_scale, w_scale)
    assert reached == ["ck"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
