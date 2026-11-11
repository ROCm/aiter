# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import importlib.util
import os

import pytest
import torch

from aiter.ops.triton.gemm.basic.gemm_a16w4 import gemm_a16w4
from aiter.ops.triton.utils._triton.arch_info import get_arch
from aiter.ops.triton.utils.shuffle import (
    awq_to_gemm_a16w4,
    gptq_to_gemm_a16w4,
    pack_gemm_a16w4_weight,
)

# Archs with gemm_a16w4 configs.
SUPPORTED_ARCHS = ("gfx942", "gfx1100")

# (N, K) of the linear layers of Llama-3-8B (also Mistral-7B) and Qwen2.5-7B:
# fused qkv, o, fused gate_up and down projections.
MODEL_SHAPES = [
    (6144, 4096),
    (4096, 4096),
    (28672, 4096),
    (4096, 14336),
    (4608, 3584),
    (3584, 3584),
    (37888, 3584),
    (3584, 18944),
]

# Output tolerances (atol, rtol) for outputs with a standard deviation of about 1.
TOLERANCES = {torch.float16: (2e-2, 1e-2), torch.bfloat16: (5e-2, 2e-2)}


def _skip_if_unsupported_arch():
    if get_arch() not in SUPPORTED_ARCHS:
        pytest.skip(f"gemm_a16w4 has no config for {get_arch()}")


def generate_gemm_a16w4_inputs(
    M: int,
    N: int,
    K: int,
    group_size: int = 128,
    dtype=torch.bfloat16,
    has_zeros: bool = False,
    act_order: bool = False,
    bias: bool = False,
    output: bool = False,
):
    """
    Returns (x, w, w_scales, w_zeros, x_perm, bias, y, q). q is the (N, K)
    uint4 weight that w packs. group_size -1 gives per-channel scales.
    """
    assert group_size == -1 or K % group_size == 0, (
        f"K={K} is not a multiple of group_size={group_size}: "
        "use group_size=-1 (per-channel scales)"
    )
    torch.manual_seed(0)
    num_groups = 1 if group_size == -1 else K // group_size

    x = torch.randn((M, K), dtype=dtype, device="cuda")
    q = torch.randint(0, 16, (N, K), dtype=torch.uint8, device="cuda")
    # Scales that give outputs with a standard deviation of about 1.
    w_scales = (
        torch.rand((N, num_groups), dtype=torch.float32, device="cuda") + 0.5
    ) / (4.6 * K**0.5)
    w_scales = w_scales.to(dtype)
    w_zeros = None
    if has_zeros:
        w_zeros = torch.randint(
            0, 16, (N, num_groups), dtype=torch.uint8, device="cuda"
        )
    x_perm = None
    if act_order:
        x_perm = torch.randperm(K, device="cuda").to(torch.int32)
    bias_tensor = None
    if bias:
        bias_tensor = torch.randn((N,), dtype=dtype, device="cuda")
    y = None
    if output:
        y = torch.empty((M, N), dtype=dtype, device="cuda")

    w = pack_gemm_a16w4_weight(q)
    return x, w, w_scales, w_zeros, x_perm, bias_tensor, y, q


def run_torch(x, q, w_scales, w_zeros=None, x_perm=None, bias=None, dtype=None):
    """Dequantize the weight in fp32, then multiply in fp32."""
    K = q.shape[1]
    group_size = K // w_scales.shape[1]
    scales = w_scales.float().repeat_interleave(group_size, dim=1)
    zeros = 8.0 if w_zeros is None else w_zeros.float()
    if w_zeros is not None:
        zeros = zeros.repeat_interleave(group_size, dim=1)
    w = (q.float() - zeros) * scales
    x = x.float()
    if x_perm is not None:
        x = x[:, x_perm.long()]
    y = x @ w.T
    if bias is not None:
        y = y + bias.float()
    return y.to(dtype or x.dtype)


def _check(out, ref, dtype):
    atol, rtol = TOLERANCES[dtype]
    torch.testing.assert_close(out, ref, atol=atol, rtol=rtol)


def get_x_vals():
    x_vals = [(1, 64, 128), (3, 72, 256), (17, 200, 384), (130, 520, 1024)]
    x_vals += [
        (M, N, K)
        for N, K in MODEL_SHAPES
        for M in (1, 2, 4, 8, 16, 32, 64, 128, 256, 1024, 8192)
    ]
    return x_vals


@pytest.mark.parametrize("M, N, K", get_x_vals())
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_gemm_a16w4(M: int, N: int, K: int, dtype):
    _skip_if_unsupported_arch()
    x, w, w_scales, _, _, _, _, q = generate_gemm_a16w4_inputs(
        M, N, K, group_size=128, dtype=dtype
    )

    out = gemm_a16w4(x, w, w_scales)

    _check(out, run_torch(x, q, w_scales, dtype=dtype), dtype)


@pytest.mark.parametrize(
    "M, N, K",
    [(1, 4096, 4096), (33, 4608, 3584), (300, 1024, 2048), (4096, 1024, 2048)],
)
@pytest.mark.parametrize("group_size", [-1, 32, 64, 128])
@pytest.mark.parametrize("has_zeros", [False, True])
@pytest.mark.parametrize("act_order", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_gemm_a16w4_quant_modes(M, N, K, group_size, has_zeros, act_order, dtype):
    _skip_if_unsupported_arch()
    x, w, w_scales, w_zeros, x_perm, _, _, q = generate_gemm_a16w4_inputs(
        M,
        N,
        K,
        group_size=group_size,
        dtype=dtype,
        has_zeros=has_zeros,
        act_order=act_order,
    )

    out = gemm_a16w4(x, w, w_scales, w_zeros=w_zeros, x_perm=x_perm)

    ref = run_torch(x, q, w_scales, w_zeros, x_perm, dtype=dtype)
    _check(out, ref, dtype)


@pytest.mark.parametrize(
    "M, N, K", [(1, 4096, 4096), (64, 3584, 18944), (2048, 640, 768)]
)
@pytest.mark.parametrize("scales_dtype", [torch.float32, None])
@pytest.mark.parametrize("out_dtype", [torch.float32, torch.bfloat16])
def test_gemm_a16w4_bias_and_output(M, N, K, scales_dtype, out_dtype):
    _skip_if_unsupported_arch()
    dtype = torch.bfloat16
    x, w, w_scales, w_zeros, _, bias, _, q = generate_gemm_a16w4_inputs(
        M, N, K, dtype=dtype, has_zeros=True, bias=True
    )
    if scales_dtype is not None:
        w_scales = w_scales.to(scales_dtype)
    y = torch.empty((M, N), dtype=out_dtype, device="cuda")

    out = gemm_a16w4(x, w, w_scales, w_zeros=w_zeros, bias=bias, dtype=out_dtype, y=y)

    assert out.data_ptr() == y.data_ptr()
    ref = run_torch(x, q, w_scales, w_zeros, bias=bias, dtype=out_dtype)
    _check(out, ref, dtype)


def test_gemm_a16w4_strided_perm_and_bias():
    _skip_if_unsupported_arch()
    M, N, K = 33, 1024, 2048
    dtype = torch.bfloat16
    x, w, w_scales, w_zeros, x_perm, bias, _, q = generate_gemm_a16w4_inputs(
        M, N, K, dtype=dtype, has_zeros=True, act_order=True, bias=True
    )
    # Every other element of a larger tensor: same values, stride 2.
    x_perm_strided = torch.stack((x_perm, torch.zeros_like(x_perm)), dim=1)[:, 0]
    bias_strided = torch.stack((bias, torch.zeros_like(bias)), dim=1)[:, 0]

    out = gemm_a16w4(
        x, w, w_scales, w_zeros=w_zeros, x_perm=x_perm_strided, bias=bias_strided
    )

    _check(out, run_torch(x, q, w_scales, w_zeros, x_perm, bias, dtype=dtype), dtype)


def _load_tuning_kernels(monkeypatch):
    """The kernels.py of the tuning scripts, which imports _utils from its own directory."""
    from aiter.ops.triton.utils._triton import arch_info

    tuning_dir = os.path.join(os.path.dirname(arch_info.__file__), "tuning")
    monkeypatch.syspath_prepend(tuning_dir)
    spec = importlib.util.spec_from_file_location(
        "aiter_tuning_kernels", os.path.join(tuning_dir, "kernels.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# K = 320 and 200 are not multiples of 128: the tuning entry uses per-channel scales.
@pytest.mark.parametrize("M, N, K", [(1, 64, 320), (33, 72, 200), (16, 256, 512)])
def test_gemm_a16w4_tuning_entry(M, N, K, monkeypatch):
    _skip_if_unsupported_arch()
    spec = _load_tuning_kernels(monkeypatch).KERNELS["gemm_a16w4"]
    call, (x, w, w_scales, y), _ = spec.setup({"M": M, "N": N, "K": K}, "triton")

    out = call(None, x, w, w_scales, y)

    q = torch.stack((w & 0xF, w >> 4), dim=-1).reshape(N, K)
    _check(out, run_torch(x, q, w_scales, dtype=x.dtype), x.dtype)


def _pack_int32(values: torch.Tensor) -> torch.Tensor:
    """Pack uint4 values (..., 8 * C) to int32 words (..., C), value 8c + i to nibble i."""
    values = values.to(torch.int64).reshape(*values.shape[:-1], -1, 8)
    shifts = torch.arange(0, 32, 4, device=values.device)
    words = (values << shifts).sum(-1)
    return torch.where(words >= 2**31, words - 2**32, words).to(torch.int32)


@pytest.mark.parametrize("M, N, K", [(1, 4096, 4096), (96, 4608, 3584)])
@pytest.mark.parametrize("group_size", [32, 128])
@pytest.mark.parametrize("sym", [True, False])
@pytest.mark.parametrize("desc_act", [False, True])
@pytest.mark.parametrize("checkpoint_format", ["gptq", "gptq_v2"])
def test_gptq_to_gemm_a16w4(M, N, K, group_size, sym, desc_act, checkpoint_format):
    _skip_if_unsupported_arch()
    if sym and checkpoint_format == "gptq_v2":
        pytest.skip("gptq_v2 only changes the zero points")
    torch.manual_seed(0)
    dtype = torch.float16
    num_groups = K // group_size
    x = torch.randn((M, K), dtype=dtype, device="cuda")
    q = torch.randint(0, 16, (K, N), device="cuda")
    scales = (
        (torch.rand((num_groups, N), dtype=torch.float32, device="cuda") + 0.5)
        / (4.6 * K**0.5)
    ).to(dtype)
    zeros = torch.randint(0, 16, (num_groups, N), device="cuda")
    if sym:
        zeros.fill_(8)
    if desc_act:
        g_idx = torch.argsort(torch.randperm(K, device="cuda")) // group_size
    else:
        g_idx = torch.arange(K, device="cuda") // group_size
    g_idx = g_idx.to(torch.int32)

    # Checkpoint tensors, packed as GPTQ stores them.
    qweight = _pack_int32(q.T).T.contiguous()
    stored_zeros = zeros - 1 if checkpoint_format == "gptq" else zeros
    qzeros = _pack_int32(stored_zeros & 0xF)

    w, w_scales, w_zeros, x_perm = gptq_to_gemm_a16w4(
        qweight, scales, qzeros, g_idx, checkpoint_format=checkpoint_format
    )
    assert (w_zeros is None) == sym
    assert (x_perm is None) == (not desc_act)
    out = gemm_a16w4(x, w, w_scales, w_zeros=w_zeros, x_perm=x_perm)

    # Textbook GPTQ dequantization of the checkpoint tensors.
    g = g_idx.long()
    weight = (q.float() - zeros[g].float()) * scales[g].float()
    ref = (x.float() @ weight).to(dtype)
    _check(out, ref, dtype)


@pytest.mark.parametrize("M, N, K", [(1, 4096, 4096), (96, 4608, 3584)])
@pytest.mark.parametrize("group_size", [64, 128])
def test_awq_to_gemm_a16w4(M, N, K, group_size):
    _skip_if_unsupported_arch()
    torch.manual_seed(0)
    dtype = torch.float16
    num_groups = K // group_size
    x = torch.randn((M, K), dtype=dtype, device="cuda")
    q = torch.randint(0, 16, (K, N), device="cuda")
    scales = (
        (torch.rand((num_groups, N), dtype=torch.float32, device="cuda") + 0.5)
        / (4.6 * K**0.5)
    ).to(dtype)
    zeros = torch.randint(0, 16, (num_groups, N), device="cuda")

    # Checkpoint tensors, packed as AWQ stores them (nibble i holds column order[i]).
    order = [0, 2, 4, 6, 1, 3, 5, 7]
    qweight = _pack_int32(q.view(K, N // 8, 8)[:, :, order].reshape(K, N))
    qzeros = _pack_int32(zeros.view(num_groups, N // 8, 8)[:, :, order].reshape(-1, N))

    w, w_scales, w_zeros = awq_to_gemm_a16w4(qweight, scales, qzeros)
    out = gemm_a16w4(x, w, w_scales, w_zeros=w_zeros)

    g = torch.arange(K, device="cuda") // group_size
    weight = (q.float() - zeros[g].float()) * scales[g].float()
    ref = (x.float() @ weight).to(dtype)
    _check(out, ref, dtype)
