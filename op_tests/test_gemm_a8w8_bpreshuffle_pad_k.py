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
    monkeypatch.setattr(gemm_mod, "is_flydsl_available", lambda: True)
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
