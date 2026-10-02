# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch
import triton

from aiter.ops.triton.gemm.batched.batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant import (
    batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant,
)
from aiter.ops.triton.utils.types import get_fp8_dtypes, str_to_torch_dtype

e5m2_type, e4m3_type = get_fp8_dtypes()


def generate_batched_gemm_a16w8_inputs(
    B: int,
    M: int,
    N: int,
    K: int,
    dtype: torch.dtype | str,
    has_bias: bool,
    output: bool,
    layout: str = "TN",
    transpose_bm: bool = False,
):
    """
    Returns:
        - x: shape (B, M, K)
        - weight: shape (B, N, K)
        - x_scale: shape (B, M, 1)
        - w_scale: shape (B, 1, N)
    """
    torch.manual_seed(0)
    if isinstance(dtype, str):
        dtype = str_to_torch_dtype[dtype]
    if layout[0] == "T":
        x = (torch.rand((B, M, K), dtype=torch.float16, device="cuda") / 10).to(
            torch.bfloat16
        )
    else:
        x = (
            (torch.rand((B, K, M), dtype=torch.float16, device="cuda") / 10)
            .to(torch.bfloat16)
            .permute(0, 2, 1)
        )

    if layout[1] == "N":
        weight = (torch.rand((B, N, K), dtype=torch.float16, device="cuda") / 10).to(
            e4m3_type
        )
    else:
        weight = (
            (torch.rand((B, N, K), dtype=torch.float16, device="cuda") / 10)
            .to(e4m3_type)
            .permute(0, 2, 1)
        )

    w_scale = torch.rand([1], dtype=torch.float32, device="cuda")[0]
    if has_bias:
        bias = torch.rand([B, 1, N], dtype=dtype).cuda() * 10
    else:
        bias = None

    y = None
    if output:
        if transpose_bm:
            y = torch.empty((M, B, N), dtype=dtype, device=x.device)
        else:
            y = torch.empty((B, M, N), dtype=dtype, device=x.device)

    return x, weight, w_scale, bias, y


def prepare_batched_gemm_input(x, transpose_bm_in=False, input_pad=0, pad_value=0):
    """Prepare the actual input layout, optionally retaining padded row strides."""
    assert input_pad >= 0
    if transpose_bm_in:
        x = x.transpose(0, 1).contiguous()
    if input_pad:
        k = x.shape[-1]
        storage = x.new_full((*x.shape[:-1], k + input_pad), pad_value)
        storage[..., :k].copy_(x)
        x = storage[..., :k]
    return x


def run_torch(x, weight, w_scale, bias=None, dtype=torch.bfloat16, transpose_bm=True):
    B = x.size(0)
    M = x.size(1)
    N = weight.size(1)
    out = torch.empty(B, M, N, dtype=torch.bfloat16, device="cuda")
    w_bf16 = weight.to(torch.bfloat16) * w_scale.to(torch.bfloat16)
    out = torch.bmm(x, w_bf16.transpose(1, 2))
    if bias is not None:
        out = out + bias
    if transpose_bm:
        out = out.transpose(0, 1)
    return out.to(dtype)


def run_triton(
    x,
    weight,
    w_scale,
    group_size=128,
    bias=None,
    dtype=torch.bfloat16,
    y=None,
    transpose_bm=False,
):
    return batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant(
        x,
        weight,
        w_scale,
        group_size=group_size,
        bias=bias,
        dtype=dtype,
        YQ=y,
        transpose_bm=transpose_bm,
    )


@pytest.fixture
def gfx950_lookup(monkeypatch):
    from aiter.ops.triton.utils import config_utils, gemm_config_utils
    from aiter.ops.triton.utils._triton import arch_info

    monkeypatch.setattr(arch_info, "get_arch", lambda: "gfx950")
    config_utils.load_config_json.cache_clear()
    gemm_config_utils._get_gemm_config_cached.cache_clear()
    yield
    config_utils.load_config_json.cache_clear()
    gemm_config_utils._get_gemm_config_cached.cache_clear()


@pytest.mark.parametrize(
    "n, k, m, bm, bn",
    [
        (512, 192, 32, 32, 128),
        (512, 192, 33, 16, 64),
        (512, 192, 64, 16, 64),
        (512, 192, 65, 32, 128),
        (512, 192, 128, 32, 128),
        (512, 192, 129, 64, 256),
        (512, 192, 257, 32, 128),
        (256, 512, 32, 32, 128),
        (256, 512, 33, 16, 64),
        (256, 512, 64, 16, 64),
        (256, 512, 128, 16, 64),
        (256, 512, 129, 32, 128),
        (256, 512, 256, 32, 128),
        (256, 512, 257, 32, 128),
    ],
)
def test_config_boundaries(n, k, m, bm, bn, gfx950_lookup):
    from aiter.ops.triton._triton_kernels.gemm.batched.batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant import (
        _get_config,
    )

    config, tuned = _get_config(m, n, k)
    assert (config["BLOCK_SIZE_M"], config["BLOCK_SIZE_N"]) == (bm, bn)
    assert tuned == (m != 257)
    if m == 33:
        original = config.copy()
        config["BLOCK_SIZE_M"] = -1
        assert _get_config(m, n, k)[0] == original
        neighbor, tuned = _get_config(m, n, k + 1)
        default, default_tuned = _get_config(m, n + 1, k)
        assert not (tuned or default_tuned) and neighbor == default


def test_explicit_config_bypasses_lookup(monkeypatch, gfx950_lookup):
    op = batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant
    config, _ = op.__globals__["_get_config"](64, 256, 512)
    launches = []

    class Kernel:
        def __getitem__(self, grid):
            return lambda *args, **kwargs: launches.append(kwargs)

    monkeypatch.setitem(
        op.__globals__,
        "_get_config",
        lambda *_: pytest.fail("Explicit config must bypass lookup"),
    )
    monkeypatch.setitem(
        op.__globals__,
        "_batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant_kernel",
        Kernel(),
    )
    x = torch.zeros((1, 64, 512), dtype=torch.bfloat16)
    w = torch.zeros((1, 256, 512), dtype=e4m3_type)
    y = torch.empty((1, 64, 256), dtype=torch.bfloat16)
    assert op(x, w, torch.ones(()), YQ=y, config=config) is y
    assert len(launches) == 1 and all(
        launches[0][key] == value for key, value in config.items()
    )


def group_quantized_reference(x, weight, w_scale, group_size=128):
    out = torch.zeros(
        (*x.shape[:2], weight.shape[1]), dtype=torch.float32, device=x.device
    )
    limit = torch.finfo(weight.dtype).max
    for start in range(0, x.shape[-1], group_size):
        a = x[..., start : start + group_size].float()
        scale = a.abs().amax(-1, keepdim=True).clamp_min(1e-10) * (1.0 / limit)
        quantized = (a * scale.reciprocal()).clamp(-limit, limit).to(weight.dtype)
        out += (
            torch.bmm(
                quantized.float(),
                weight[..., start : start + group_size].float().transpose(1, 2),
            )
            * scale
        )
    return out * w_scale


@pytest.mark.parametrize("m, n, k", [(65, 512, 192), (129, 256, 512)])
def test_token_first_query_and_value_layout(m, n, k):
    transpose_output = k == 512
    x, weight, scale, _, y = generate_batched_gemm_a16w8_inputs(
        8, m, n, k, "bf16", False, transpose_output, transpose_bm=transpose_output
    )
    x = (x - 0.05).contiguous()
    inputs = prepare_batched_gemm_input(x, True, 64 if k == 192 else 0, pad_value=1024)
    if k == 192:
        assert inputs.stride() == (8 * 256, 256, 1) and not inputs.is_contiguous()
    op = batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant
    actual = op(
        inputs, weight, scale, YQ=y, transpose_bm=transpose_output, transpose_bm_in=True
    )
    contiguous = op(x, weight, scale, transpose_bm=transpose_output)
    torch.testing.assert_close(actual, contiguous, atol=0, rtol=0)
    if y is not None:
        assert actual is y
        actual = actual.transpose(0, 1)
    torch.testing.assert_close(
        actual.float(),
        group_quantized_reference(x, weight, scale),
        atol=0.02,
        rtol=0.02,
    )


def get_x_vals():

    x_vals = [(1024 * v, 1024 * v, 1024 * v) for v in range(1, 9)]
    x_vals += [
        (1, 1280, 8192),
        (32, 1280, 8192),
        (64, 1280, 8192),
        (128, 1280, 8192),
        (192, 1280, 8192),
        (256, 1280, 8192),
        (320, 1280, 8192),
        (512, 1280, 8192),
        (1024, 1280, 8192),
        (2048, 1280, 8192),
        (4096, 1280, 8192),
        (8192, 1280, 8192),
        (16384, 1280, 8192),
        (1, 8192, 1024),
        (32, 8192, 1024),
        (64, 8192, 1024),
        (128, 8192, 1024),
        (192, 8192, 1024),
        (256, 8192, 1024),
        (320, 8192, 1024),
        (512, 8192, 1024),
        (1024, 8192, 1024),
        (2048, 8192, 1024),
        (4096, 8192, 1024),
        (8192, 8192, 1024),
        (16384, 8192, 1024),
    ]
    x_vals += [(v**2, 128, 512) for v in range(7)]
    x_vals += [(v**2, 512, 128) for v in range(7)]
    x_vals += [
        (64, 512, 192),
        (128, 512, 192),
        (64, 256, 512),
        (128, 256, 512),
        (256, 256, 512),
    ]
    x_vals += [(1, 128, 1)]  # minimal case
    return x_vals


@pytest.mark.parametrize(
    "dtype, b, m, n, k, group_size, has_bias, output, transpose_bm",
    [
        (dtype, b, *shape, group_size, has_bias, output, transpose_bm)
        for output in [True, False]
        for dtype in ["bf16"]
        for b in [16]
        for shape in get_x_vals()
        for group_size in [128]
        for has_bias in [True, False]
        for transpose_bm in [True, False]
    ],
)
def test_batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant(
    dtype, b, m, n, k, group_size, has_bias, output, transpose_bm
):
    torch.cuda.empty_cache()  # Helps avoid hangs in large tests

    dtype = str_to_torch_dtype[dtype]
    x, weight, w_scale, bias, y = generate_batched_gemm_a16w8_inputs(
        b, m, n, k, dtype, has_bias, output, transpose_bm=transpose_bm
    )
    a = run_torch(x, weight, w_scale, bias, dtype, transpose_bm)
    b = run_triton(
        x,
        weight,
        w_scale,
        group_size=group_size,
        bias=bias,
        dtype=dtype,
        y=y,
        transpose_bm=transpose_bm,
    )

    triton.testing.assert_close(a, b, atol=0.1, rtol=0.1)
