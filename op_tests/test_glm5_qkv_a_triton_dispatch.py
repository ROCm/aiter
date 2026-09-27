# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Dispatch coverage for the GLM-5 qkv_a_proj gfx950 tuning rows."""

from pathlib import Path

import pytest

from aiter.ops import gemm_op_a8w8 as gemm_mod

TUNED_FILE = str(
    Path(gemm_mod.__file__).resolve().parents[1]
    / "configs/model_configs/glm5_a8w8_blockscale_bpreshuffle_tuned_gemm.csv"
)
GFX = "gfx950"
CU_NUM = 256
N = 2624
K = 6144


def _clear_config_cache():
    gemm_mod.get_CKGEMM_config.cache_clear()
    gemm_mod._CKGEMM_CONFIG_CACHE.pop(TUNED_FILE, None)
    gemm_mod._CKGEMM_HAS_GFX.pop(TUNED_FILE, None)


@pytest.fixture(autouse=True)
def _gfx950_config(monkeypatch):
    monkeypatch.setattr(gemm_mod, "get_gfx", lambda: GFX)
    monkeypatch.setattr(gemm_mod, "get_cu_num", lambda: CU_NUM)
    _clear_config_cache()
    yield
    _clear_config_cache()


@pytest.mark.parametrize(
    ("m", "expected_row"),
    [
        (1, 1),
        (2, 16),
        (4, 4),
        (8, 8),
        (16, 16),
        (32, 32),
        (33, 64),
        (48, 64),
        (64, 64),
    ],
)
def test_glm5_qkv_a_decode_routes_to_expected_triton_tier(m, expected_row):
    config = gemm_mod.get_CKGEMM_config(m, N, K, TUNED_FILE)
    expected = gemm_mod._CKGEMM_CONFIG_CACHE[TUNED_FILE][
        (GFX, CU_NUM, expected_row, N, K)
    ]

    assert config == expected
    assert config["libtype"] == "triton"


def test_glm5_qkv_a_prefill_control_stays_on_ck():
    config = gemm_mod.get_CKGEMM_config(128, N, K, TUNED_FILE)
    assert config["libtype"] == "ck"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
