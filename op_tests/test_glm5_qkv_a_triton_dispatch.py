# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Dispatch coverage for the GLM-5 qkv_a_proj gfx950 tuning rows."""

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
    ("m", "expected_row", "block_m", "block_n", "num_ksplit", "num_warps"),
    [
        (1, 1, 16, 16, 8, 1),
        (2, 16, 16, 16, 8, 1),
        (4, 4, 16, 16, 8, 1),
        (8, 8, 16, 32, 8, 2),
        (16, 16, 16, 16, 8, 1),
        (32, 32, 32, 32, 8, 2),
        (33, 64, 64, 32, 4, 4),
        (48, 64, 64, 32, 4, 4),
        (64, 64, 64, 32, 4, 4),
    ],
)
def test_glm5_qkv_a_decode_routes_to_expected_triton_tier(
    monkeypatch, m, expected_row, block_m, block_n, num_ksplit, num_warps
):
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

    expected = gemm_mod._CKGEMM_CONFIG_CACHE[TUNED_FILE][
        (GFX, CU_NUM, expected_row, N, K)
    ]
    assert selected_csv == expected
    assert expected["libtype"] == "triton"
    assert selected == {
        "BLOCK_SIZE_M": block_m,
        "BLOCK_SIZE_N": block_n,
        "BLOCK_SIZE_K": 128,
        "GROUP_SIZE_M": 1,
        "num_warps": num_warps,
        "num_stages": 3,
        "waves_per_eu": 2,
        "matrix_instr_nonkdim": 16,
        "cache_modifier": ".cg",
        "NUM_KSPLIT": num_ksplit,
        "is_tuned": True,
    }


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
