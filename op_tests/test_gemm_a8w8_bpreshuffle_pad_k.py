# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

from aiter import dtypes
from aiter.ops import gemm_op_a8w8 as gemm_mod
from aiter.ops.shuffle import shuffle_weight

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="a8w8 bpreshuffle pad-K tests require a CUDA/HIP device",
)


def test_preshuffle_flat_buffers_stay_below_4gib():
    from aiter.ops.flydsl.gemm_kernels import (
        _check_preshuffle_flat_buffer_capacity,
    )

    # The output is just under 4 GiB; masking the padded M rows keeps this legal.
    _check_preshuffle_flat_buffer_capacity(8161, 262192, 512, 1, 1, 2)

    with pytest.raises(RuntimeError, match="output buffer.*fewer than 4 GiB"):
        _check_preshuffle_flat_buffer_capacity(8191, 262192, 512, 1, 1, 2)
    with pytest.raises(RuntimeError, match="A buffer.*fewer than 4 GiB"):
        _check_preshuffle_flat_buffer_capacity(65536, 16, 65536, 1, 1, 2)
    with pytest.raises(RuntimeError, match="B buffer.*fewer than 4 GiB"):
        _check_preshuffle_flat_buffer_capacity(16, 65536, 65536, 1, 1, 2)


@pytest.mark.parametrize("m, backend", [(65536, "flydsl"), (65537, "ck")])
def test_flydsl_bpreshuffle_row_falls_back_to_ck_above_m_max(monkeypatch, m, backend):
    from aiter.ops.flydsl import gemm_kernels

    # Kimi-K3's 6144x7168 M=8192 row, which padded-M lookup returns for every
    # M > 8192. Meta tensors stand in for the 470 MB activation.
    row = {
        "libtype": "flydsl",
        "splitK": 0,
        "kernelName": "flydsl_bpreshuflle_128x256x128_F8_F8_B16_1x2x4x2_default",
    }
    n, k = 6144, 7168
    xq = torch.empty((m, k), device="meta", dtype=dtypes.fp8)
    wq = torch.empty((n, k), device="meta", dtype=dtypes.fp8)
    x_scale = torch.empty((m, 1), device="meta", dtype=torch.float32)
    w_scale = torch.empty((n, 1), device="meta", dtype=torch.float32)
    out = torch.empty((m, n), device="meta", dtype=torch.bfloat16)
    seen = []

    def fake_flydsl(*args, **kwargs):
        seen.append("flydsl")

    def fake_ck(XQ, WQ, x_scale, w_scale, Y, splitK=0):
        seen.append("ck")
        return Y

    # gfx1250 routes every kernel name to its WMMA backend before the M-limit check.
    monkeypatch.setattr(gemm_mod, "get_gfx", lambda: "gfx950")
    monkeypatch.setattr(gemm_kernels, "flydsl_preshuffle_gemm_a8", fake_flydsl)
    monkeypatch.setattr(gemm_mod, "gemm_a8w8_bpreshuffle_ck", fake_ck)

    got = gemm_mod.gemm_a8w8_bpreshuffle_flydsl(xq, wq, x_scale, w_scale, out, row)

    assert gemm_kernels.PRESHUFFLE_M_MAX == 65536
    assert seen == [backend]
    assert got is out


def test_two_wave_vgpr_estimate_uses_128_threads():
    from aiter.ops.flydsl.gemm_tune.flydsl_gemm_a8w8_bpreshuffle_common import (
        _estimate_max_wpe,
    )

    # Both tiles hold eight accumulator VGPRs per thread: 32x32 / 128 threads
    # for the 2-wave path and 32x64 / 256 threads for the standard 4-wave path.
    assert _estimate_max_wpe(32, 32, 512, total_vgpr=12) == 1
    assert _estimate_max_wpe(32, 64, 512, total_vgpr=12) == 1


def test_shuffle_weight_pad_k_to_pads_last_dim():
    weight = torch.zeros((16, 96), device="cuda", dtype=dtypes.fp8)

    shuffled = shuffle_weight(weight, layout=(16, 16), pad_k_to=128)

    assert shuffled.shape == (16, 128)
    assert shuffled.is_shuffled
    assert shuffled.aiter_original_k == 96
    assert shuffled.aiter_padded_k == 128


def test_gemm_a8w8_bpreshuffle_uses_logical_k_for_ck_config(monkeypatch):
    xq = torch.zeros((2, 96), device="cuda", dtype=dtypes.fp8)
    wq = torch.zeros((16, 128), device="cuda", dtype=dtypes.fp8)
    x_scale = torch.ones((2, 1), device="cuda", dtype=torch.float32)
    w_scale = torch.ones((16, 1), device="cuda", dtype=torch.float32)
    seen = {}

    def fake_config(m, n, k, q_dtype_w, tuned_file):
        seen["config_shape"] = (m, n, k)
        return {"libtype": "ck", "splitK": 0}

    def fake_ck(XQ, WQ, x_scale, w_scale, Y, splitK):
        seen["x_shape"] = tuple(XQ.shape)
        seen["w_shape"] = tuple(WQ.shape)
        return Y

    monkeypatch.setattr(gemm_mod, "get_GEMM_config_with_quant_type", fake_config)
    monkeypatch.setattr(gemm_mod, "gemm_a8w8_bpreshuffle_ck", fake_ck)

    out = gemm_mod.gemm_a8w8_bpreshuffle(xq, wq, x_scale, w_scale, dtype=torch.bfloat16)

    assert out.shape == (2, 16)
    assert out.dtype == torch.bfloat16
    assert seen["config_shape"] == (2, 16, 96)
    assert seen["x_shape"] == (2, 96)
    assert seen["w_shape"] == (16, 128)


def test_gemm_a8w8_bpreshuffle_uses_logical_k_for_cktile_config(monkeypatch):
    xq = torch.zeros((2, 96), device="cuda", dtype=dtypes.fp8)
    wq = torch.zeros((16, 128), device="cuda", dtype=dtypes.fp8)
    x_scale = torch.ones((2, 1), device="cuda", dtype=torch.float32)
    w_scale = torch.ones((16, 1), device="cuda", dtype=torch.float32)
    seen = {}

    def fake_config(m, n, k, q_dtype_w, tuned_file):
        seen["config_shape"] = (m, n, k)
        return {"libtype": "cktile", "splitK": 0}

    def fake_cktile(XQ, WQ, x_scale, w_scale, Y, splitK):
        seen["x_shape"] = tuple(XQ.shape)
        seen["w_shape"] = tuple(WQ.shape)
        return Y

    monkeypatch.setattr(gemm_mod, "get_GEMM_config_with_quant_type", fake_config)
    monkeypatch.setattr(gemm_mod, "gemm_a8w8_bpreshuffle_cktile", fake_cktile)

    out = gemm_mod.gemm_a8w8_bpreshuffle(xq, wq, x_scale, w_scale, dtype=torch.bfloat16)

    assert out.shape == (2, 16)
    assert out.dtype == torch.bfloat16
    assert seen["config_shape"] == (2, 16, 96)
    assert seen["x_shape"] == (2, 96)
    assert seen["w_shape"] == (16, 128)


def test_gemm_a8w8_bpreshuffle_falls_back_to_padded_k_config(monkeypatch):
    xq = torch.zeros((2, 96), device="cuda", dtype=dtypes.fp8)
    wq = torch.zeros((16, 128), device="cuda", dtype=dtypes.fp8)
    x_scale = torch.ones((2, 1), device="cuda", dtype=torch.float32)
    w_scale = torch.ones((16, 1), device="cuda", dtype=torch.float32)
    seen = {"config_shapes": []}

    def fake_config(m, n, k, q_dtype_w, tuned_file):
        seen["config_shapes"].append((m, n, k))
        if k == 128:
            return {"libtype": "cktile", "splitK": 0}
        return None

    def fake_cktile(XQ, WQ, x_scale, w_scale, Y, splitK):
        seen["x_shape"] = tuple(XQ.shape)
        seen["w_shape"] = tuple(WQ.shape)
        return Y

    monkeypatch.setattr(gemm_mod, "get_GEMM_config_with_quant_type", fake_config)
    monkeypatch.setattr(gemm_mod, "gemm_a8w8_bpreshuffle_cktile", fake_cktile)

    out = gemm_mod.gemm_a8w8_bpreshuffle(xq, wq, x_scale, w_scale, dtype=torch.bfloat16)

    assert out.shape == (2, 16)
    assert out.dtype == torch.bfloat16
    assert seen["config_shapes"] == [(2, 16, 96), (2, 16, 128)]
    assert seen["x_shape"] == (2, 96)
    assert seen["w_shape"] == (16, 128)


def test_gemm_a8w8_bpreshuffle_uses_cktile_for_untuned_padded_k(monkeypatch):
    xq = torch.zeros((2, 96), device="cuda", dtype=dtypes.fp8)
    wq = torch.zeros((16, 128), device="cuda", dtype=dtypes.fp8)
    x_scale = torch.ones((2, 1), device="cuda", dtype=torch.float32)
    w_scale = torch.ones((16, 1), device="cuda", dtype=torch.float32)
    seen = {"config_shapes": []}

    def fake_config(m, n, k, q_dtype_w, tuned_file):
        seen["config_shapes"].append((m, n, k))

    def fake_cktile(XQ, WQ, x_scale, w_scale, Y, splitK):
        seen["x_shape"] = tuple(XQ.shape)
        seen["w_shape"] = tuple(WQ.shape)
        return Y

    def fake_ck(*args, **kwargs):
        raise AssertionError("padded-K untuned fallback should use CKTile")

    monkeypatch.setattr(gemm_mod, "get_GEMM_config_with_quant_type", fake_config)
    monkeypatch.setattr(gemm_mod, "gemm_a8w8_bpreshuffle_cktile", fake_cktile)
    monkeypatch.setattr(gemm_mod, "gemm_a8w8_bpreshuffle_ck", fake_ck)

    out = gemm_mod.gemm_a8w8_bpreshuffle(xq, wq, x_scale, w_scale, dtype=torch.bfloat16)

    assert out.shape == (2, 16)
    assert out.dtype == torch.bfloat16
    assert seen["config_shapes"] == [(2, 16, 96), (2, 16, 128)]
    assert seen["x_shape"] == (2, 96)
    assert seen["w_shape"] == (16, 128)


def test_gemm_a8w8_bpreshuffle_pads_activation_for_flydsl(monkeypatch):
    xq = torch.zeros((2, 96), device="cuda", dtype=dtypes.fp8)
    wq = torch.zeros((16, 128), device="cuda", dtype=dtypes.fp8)
    x_scale = torch.ones((2, 1), device="cuda", dtype=torch.float32)
    w_scale = torch.ones((16, 1), device="cuda", dtype=torch.float32)
    seen = {}

    def fake_config(m, n, k, q_dtype_w, tuned_file):
        seen["config_shape"] = (m, n, k)
        return {"libtype": "flydsl", "splitK": 0, "kernelName": "fake"}

    def fake_flydsl(XQ, WQ, x_scale, w_scale, Y, config):
        seen["x_shape"] = tuple(XQ.shape)
        seen["tail_is_zero"] = bool((XQ[:, 96:].to(torch.float32) == 0).all())
        return Y

    monkeypatch.setattr(gemm_mod, "get_GEMM_config_with_quant_type", fake_config)
    monkeypatch.setattr(gemm_mod, "gemm_a8w8_bpreshuffle_flydsl", fake_flydsl)

    out = gemm_mod.gemm_a8w8_bpreshuffle(xq, wq, x_scale, w_scale, dtype=torch.bfloat16)

    assert out.shape == (2, 16)
    assert out.dtype == torch.bfloat16
    assert seen["config_shape"] == (2, 16, 96)
    assert seen["x_shape"] == (2, 128)
    assert seen["tail_is_zero"]


def test_gemm_a8w8_bpreshuffle_rejects_short_weight_k():
    xq = torch.zeros((2, 128), device="cuda", dtype=dtypes.fp8)
    wq = torch.zeros((16, 96), device="cuda", dtype=dtypes.fp8)
    x_scale = torch.ones((2, 1), device="cuda", dtype=torch.float32)
    w_scale = torch.ones((16, 1), device="cuda", dtype=torch.float32)

    with pytest.raises(RuntimeError, match="WQ K >= XQ K"):
        gemm_mod.gemm_a8w8_bpreshuffle(xq, wq, x_scale, w_scale, dtype=torch.bfloat16)
