# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for FlyDSL HGEMM precision regressions.

Usage:
    pytest -sv aiter/ops/flydsl/test_flydsl_hgemm.py
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from torch.profiler import profile, ProfilerActivity
from dataclasses import dataclass

from aiter.ops.flydsl.utils import is_flydsl_available

if not torch.cuda.is_available():
    pytest.skip("ROCm not available. Skipping GPU tests.", allow_module_level=True)
if not is_flydsl_available():
    pytest.skip(
        "flydsl is not installed. Skipping FlyDSL HGEMM tests.", allow_module_level=True
    )

from flydsl.runtime.device import get_rocm_arch

try:
    from aiter.ops.flydsl.kernels.hgemm_wmma_gfx950 import hgemm
except ImportError as exc:
    pytest.skip(
        f"Unable to import FlyDSL HGEMM kernels: {exc}", allow_module_level=True
    )

torch.set_default_device("cuda")


ROTARY_INPUTS_TARGET_BYTES = 8 * 1024**3


@dataclass
class _TestArgs:
    dtype: torch.dtype | str
    m: int
    n: int
    k: int
    TILE_M: int
    TILE_N: int
    TILE_K: int
    STAGES: int
    SPLIT_K: int
    BLOCK_M_WARPS: int
    BLOCK_N_WARPS: int
    BLOCK_K_WARPS: int
    HAS_BIAS: bool
    GROUP_M: int
    USE_HALF_TILE_INTERLEAVED: bool


def get_torch_fp8_dtype():
    arch = str(get_rocm_arch())
    if ("gfx95" in arch or "gfx12" in arch) and hasattr(torch, "float8_e4m3fn"):
        return torch.float8_e4m3fn
    if hasattr(torch, "float8_e4m3fnuz"):
        return torch.float8_e4m3fnuz
    if hasattr(torch, "float8_e4m3fn"):
        return torch.float8_e4m3fn
    raise RuntimeError("This PyTorch build does not expose an E4M3 FP8 dtype")


def quantize_ptpc_fp8(x: torch.Tensor):
    fp8_dtype = get_torch_fp8_dtype()
    fp8_max = float(torch.finfo(fp8_dtype).max)
    scale = x.float().abs().amax(dim=1, keepdim=True) / fp8_max
    scale[scale == 0] = 1
    x_fp8 = (x.float() / scale).to(fp8_dtype)
    scale = torch.nan_to_num(scale, nan=1.0, posinf=1.0, neginf=1.0).squeeze(1)
    return x_fp8, scale.float()


def create_inputs(args: _TestArgs):
    if args.dtype == "fp8_ptpc":
        a_f32 = torch.empty((args.m, args.k), dtype=torch.float32, device="cuda")
        b_f32 = torch.empty((args.n, args.k), dtype=torch.float32, device="cuda")
        a_f32.uniform_(-1, 1)
        b_f32.uniform_(-1, 1)
        a, scale_a = quantize_ptpc_fp8(a_f32)
        b, scale_b = quantize_ptpc_fp8(b_f32)
        if args.HAS_BIAS:
            bias = torch.empty((args.n,), dtype=torch.bfloat16, device="cuda")
            bias.uniform_(10, 20)
        else:
            bias = None
        return (a, b, scale_a, scale_b, bias)
    a = torch.empty((args.m, args.k), dtype=args.dtype, device="cuda")
    a.uniform_(-1, 1)
    b = torch.empty((args.n, args.k), dtype=args.dtype, device="cuda")
    b.uniform_(-1, 1)
    if args.HAS_BIAS:
        bias = torch.empty((args.n,), dtype=args.dtype, device="cuda")
        bias.uniform_(10, 20)
    else:
        bias = None
    return (a, b, bias)


def create_outputs(args: _TestArgs):
    if args.dtype == "fp8_ptpc":
        c = torch.randn((args.m, args.n), dtype=torch.bfloat16, device="cuda")
    else:
        c = torch.randn((args.m, args.n), dtype=args.dtype, device="cuda")
    return (c,)


def ref_func(*args):
    if len(args) == 6:
        a, b, scale_a, scale_b, bias, c = args
        a_f32 = a.float() * scale_a[:, None]
        b_f32 = b.float() * scale_b[:, None]
        if bias is not None:
            ref = torch.addmm(bias.float(), a_f32, b_f32.t())
        else:
            ref = torch.mm(a_f32, b_f32.t())
        c.copy_(ref.to(c.dtype))
    else:
        a, b, bias, c = args
        F.linear(a, b, out=c, bias=bias)


def func(*args):
    if len(args) == 7:
        a, b, scale_a, scale_b, bias, c, kwargs = args
        hgemm(a, b, c, bias=bias, scale_a=scale_a, scale_b=scale_b, user_kwargs=kwargs)
    else:
        a, b, bias, c, kwargs = args
        hgemm(a, b, c, bias=bias, user_kwargs=kwargs)


def tensor_nbytes(tensors: torch.Tensor):
    return sum(t.numel() * t.element_size() for t in tensors)


def get_rotary_inputs(sample_inputs: torch.Tensor, sample_outputs: torch.Tensor):
    global ROTARY_INPUTS_TARGET_BYTES
    slot_bytes = 2 * (tensor_nbytes(sample_inputs) + tensor_nbytes(sample_outputs))
    rotary_inputs = ROTARY_INPUTS_TARGET_BYTES // slot_bytes
    return max(1, int(rotary_inputs))


def check_acc(args: _TestArgs):
    kwargs = {
        "TILE_M": args.TILE_M,
        "TILE_N": args.TILE_N,
        "TILE_K": args.TILE_K,
        "STAGES": args.STAGES,
        "SPLIT_K": args.SPLIT_K,
        "BLOCK_M_WARPS": args.BLOCK_M_WARPS,
        "BLOCK_N_WARPS": args.BLOCK_N_WARPS,
        "BLOCK_K_WARPS": args.BLOCK_K_WARPS,
        "GROUP_M": args.GROUP_M,
        "USE_HALF_TILE_INTERLEAVED": args.USE_HALF_TILE_INTERLEAVED,
    }
    inputs = create_inputs(args)
    outputs = create_outputs(args)
    ref_outputs = create_outputs(args)
    inouts = inputs + outputs
    ref_inouts = inputs + ref_outputs
    maxdiff_out_ = []

    def get_tol(args):
        k_scale = (args.k / 8192) ** 0.5
        k_scale *= args.SPLIT_K * args.BLOCK_K_WARPS
        if args.dtype == "fp8_ptpc":
            return 2e-1 * k_scale, 2e-1
        if args.dtype is torch.bfloat16:
            return 2e-1 * k_scale, 2e-1
        return 5e-2 * k_scale, 5e-2

    atol, rtol = get_tol(args)
    for _ in range(5):
        func(*(inouts + (kwargs,)))
        ref_func(*ref_inouts)
        for output, ref_output in zip(outputs, ref_outputs):
            maxdiff_out = (output - ref_output).abs().max().item()
            maxdiff_out_.append(maxdiff_out)
            print(maxdiff_out, flush=True)
            torch.testing.assert_close(
                output,
                ref_output,
                atol=atol,
                rtol=rtol,
                check_dtype=True,
            )
    print(f"\n{args}\nmaxdiff_out:{maxdiff_out_}")


def benchmark(args: _TestArgs, warmup: int = 500, niters: int = 600):
    kwargs = {
        "TILE_M": args.TILE_M,
        "TILE_N": args.TILE_N,
        "TILE_K": args.TILE_K,
        "STAGES": args.STAGES,
        "SPLIT_K": args.SPLIT_K,
        "BLOCK_M_WARPS": args.BLOCK_M_WARPS,
        "BLOCK_N_WARPS": args.BLOCK_N_WARPS,
        "BLOCK_K_WARPS": args.BLOCK_K_WARPS,
        "GROUP_M": args.GROUP_M,
        "USE_HALF_TILE_INTERLEAVED": args.USE_HALF_TILE_INTERLEAVED,
    }
    sample_inputs = create_inputs(args)
    sample_outputs = create_outputs(args)
    rotary_inputs = get_rotary_inputs(sample_inputs, sample_outputs)
    inputs = [sample_inputs] + [create_inputs(args) for _ in range(rotary_inputs - 1)]
    ref_inputs = [create_inputs(args) for _ in range(rotary_inputs)]
    outputs = [sample_outputs] + [
        create_outputs(args) for _ in range(rotary_inputs - 1)
    ]
    ref_outputs = [create_outputs(args) for _ in range(rotary_inputs)]
    global ROTARY_INPUTS_TARGET_BYTES
    print(
        f"rotary_inputs:{rotary_inputs}, target_bytes:{ROTARY_INPUTS_TARGET_BYTES}, "
        f"warmup:{warmup}, niters:{niters}"
    )

    def run_ref(idx):
        ref_func(*(ref_inputs[idx] + ref_outputs[idx]))

    def run_flydsl(idx):
        func(*(inputs[idx] + outputs[idx] + (kwargs,)))

    print("===================== [INTERLEAVED] =====================")
    for i in range(warmup):
        idx = i % rotary_inputs
        if i % 2 == 0:
            run_ref(idx)
            run_flydsl(idx)
        else:
            run_flydsl(idx)
            run_ref(idx)
        torch.cuda.synchronize()

    with profile(
        activities=[ProfilerActivity.CUDA],
    ) as prof:
        for i in range(warmup, niters):
            idx = i % rotary_inputs
            if i % 2 == 0:
                run_ref(idx)
                run_flydsl(idx)
            else:
                run_flydsl(idx)
                run_ref(idx)
            torch.cuda.synchronize()
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1))


@pytest.mark.parametrize("dtype", ["fp16", "bf16", "fp8_ptpc"])
@pytest.mark.parametrize(
    "m, n, k, TILE_M, TILE_N, TILE_K, STAGES, SPLIT_K, BLOCK_M_WARPS, BLOCK_N_WARPS, BLOCK_K_WARPS, HAS_BIAS, GROUP_M, USE_HALF_TILE_INTERLEAVED",
    [
        # k 8192
        (8192, 8192, 8192, 256, 256, 64, 2, 1, 2, 4, 1, True, 0, False),
        (8192, 8192, 8192, 256, 256, 64, 2, 1, 2, 4, 1, True, 0, True),
        (8192, 8192, 8192, 256, 256, 64, 2, 1, 2, 4, 1, True, 4, False),
        (8192, 8192, 8192, 256, 256, 64, 2, 1, 2, 4, 1, True, 4, True),
        (8192, 8192, 8192, 256, 256, 64, 2, 1, 2, 4, 1, False, 0, False),
        (8192, 8192, 8192, 256, 256, 64, 2, 1, 2, 4, 1, False, 0, True),
        (8192, 8192, 8192, 256, 256, 64, 2, 1, 2, 4, 1, False, 4, False),
        (8192, 8192, 8192, 256, 256, 64, 2, 1, 2, 4, 1, False, 4, True),
        # k 8224
        (8192, 8192, 8224, 256, 256, 64, 2, 1, 2, 4, 1, True, 0, False),
        (8192, 8192, 8224, 256, 256, 64, 2, 1, 2, 4, 1, True, 0, True),
        (8192, 8192, 8224, 256, 256, 64, 2, 1, 2, 4, 1, True, 4, False),
        (8192, 8192, 8224, 256, 256, 64, 2, 1, 2, 4, 1, True, 4, True),
        (8192, 8192, 8224, 256, 256, 64, 2, 1, 2, 4, 1, False, 0, False),
        (8192, 8192, 8224, 256, 256, 64, 2, 1, 2, 4, 1, False, 0, True),
        (8192, 8192, 8224, 256, 256, 64, 2, 1, 2, 4, 1, False, 4, False),
        (8192, 8192, 8224, 256, 256, 64, 2, 1, 2, 4, 1, False, 4, True),
        # k 8256
        (8192, 8192, 8256, 256, 256, 64, 2, 1, 2, 4, 1, True, 0, False),
        (8192, 8192, 8256, 256, 256, 64, 2, 1, 2, 4, 1, True, 0, True),
        (8192, 8192, 8256, 256, 256, 64, 2, 1, 2, 4, 1, True, 4, False),
        (8192, 8192, 8256, 256, 256, 64, 2, 1, 2, 4, 1, True, 4, True),
        (8192, 8192, 8256, 256, 256, 64, 2, 1, 2, 4, 1, False, 0, False),
        (8192, 8192, 8256, 256, 256, 64, 2, 1, 2, 4, 1, False, 0, True),
        (8192, 8192, 8256, 256, 256, 64, 2, 1, 2, 4, 1, False, 4, False),
        (8192, 8192, 8256, 256, 256, 64, 2, 1, 2, 4, 1, False, 4, True),
        # k 8288
        (8192, 8192, 8288, 256, 256, 64, 2, 1, 2, 4, 1, True, 0, False),
        (8192, 8192, 8288, 256, 256, 64, 2, 1, 2, 4, 1, True, 0, True),
        (8192, 8192, 8288, 256, 256, 64, 2, 1, 2, 4, 1, True, 4, False),
        (8192, 8192, 8288, 256, 256, 64, 2, 1, 2, 4, 1, True, 4, True),
        (8192, 8192, 8288, 256, 256, 64, 2, 1, 2, 4, 1, False, 0, False),
        (8192, 8192, 8288, 256, 256, 64, 2, 1, 2, 4, 1, False, 0, True),
        (8192, 8192, 8288, 256, 256, 64, 2, 1, 2, 4, 1, False, 4, False),
        (8192, 8192, 8288, 256, 256, 64, 2, 1, 2, 4, 1, False, 4, True),
        # k 8320
        (8192, 8192, 8320, 256, 256, 64, 2, 1, 2, 4, 1, True, 0, False),
        (8192, 8192, 8320, 256, 256, 64, 2, 1, 2, 4, 1, True, 0, True),
        (8192, 8192, 8320, 256, 256, 64, 2, 1, 2, 4, 1, True, 4, False),
        (8192, 8192, 8320, 256, 256, 64, 2, 1, 2, 4, 1, True, 4, True),
        (8192, 8192, 8320, 256, 256, 64, 2, 1, 2, 4, 1, False, 0, False),
        (8192, 8192, 8320, 256, 256, 64, 2, 1, 2, 4, 1, False, 0, True),
        (8192, 8192, 8320, 256, 256, 64, 2, 1, 2, 4, 1, False, 4, False),
        (8192, 8192, 8320, 256, 256, 64, 2, 1, 2, 4, 1, False, 4, True),
        # k 8352
        (8192, 8192, 8352, 256, 256, 64, 2, 1, 2, 4, 1, True, 0, False),
        (8192, 8192, 8352, 256, 256, 64, 2, 1, 2, 4, 1, True, 0, True),
        (8192, 8192, 8352, 256, 256, 64, 2, 1, 2, 4, 1, True, 4, False),
        (8192, 8192, 8352, 256, 256, 64, 2, 1, 2, 4, 1, True, 4, True),
        (8192, 8192, 8352, 256, 256, 64, 2, 1, 2, 4, 1, False, 0, False),
        (8192, 8192, 8352, 256, 256, 64, 2, 1, 2, 4, 1, False, 0, True),
        (8192, 8192, 8352, 256, 256, 64, 2, 1, 2, 4, 1, False, 4, False),
        (8192, 8192, 8352, 256, 256, 64, 2, 1, 2, 4, 1, False, 4, True),
        # k 8384
        (8192, 8192, 8384, 256, 256, 64, 2, 1, 2, 4, 1, True, 0, False),
        (8192, 8192, 8384, 256, 256, 64, 2, 1, 2, 4, 1, True, 0, True),
        (8192, 8192, 8384, 256, 256, 64, 2, 1, 2, 4, 1, True, 4, False),
        (8192, 8192, 8384, 256, 256, 64, 2, 1, 2, 4, 1, True, 4, True),
        (8192, 8192, 8384, 256, 256, 64, 2, 1, 2, 4, 1, False, 0, False),
        (8192, 8192, 8384, 256, 256, 64, 2, 1, 2, 4, 1, False, 0, True),
        (8192, 8192, 8384, 256, 256, 64, 2, 1, 2, 4, 1, False, 4, False),
        (8192, 8192, 8384, 256, 256, 64, 2, 1, 2, 4, 1, False, 4, True),
        # k 8416
        (8160, 8160, 8416, 256, 256, 64, 2, 1, 2, 4, 1, True, 0, False),
        (8160, 8160, 8416, 256, 256, 64, 2, 1, 2, 4, 1, True, 0, True),
        (8160, 8192, 8416, 256, 256, 64, 2, 1, 2, 4, 1, True, 4, False),
        (8160, 8192, 8416, 256, 256, 64, 2, 1, 2, 4, 1, True, 4, True),
        (8160, 8192, 8416, 256, 256, 64, 2, 1, 2, 4, 1, False, 0, False),
        (8160, 8192, 8416, 256, 256, 64, 2, 1, 2, 4, 1, False, 0, True),
        (8160, 8160, 8416, 256, 256, 64, 2, 1, 2, 4, 1, False, 4, False),
        (8160, 8160, 8416, 256, 256, 64, 2, 1, 2, 4, 1, False, 4, True),
        # need pad
        (8160, 8160, 8160, 256, 256, 64, 2, 1, 2, 4, 1, True, 0, False),
        (8160, 8160, 8160, 256, 256, 64, 2, 1, 2, 4, 1, True, 0, True),
        (8160, 8160, 8160, 256, 256, 64, 2, 1, 2, 4, 1, True, 4, False),
        (8160, 8160, 8160, 256, 256, 64, 2, 1, 2, 4, 1, True, 4, True),
        (8160, 8160, 8160, 256, 256, 64, 2, 1, 2, 4, 1, False, 0, False),
        (8160, 8160, 8160, 256, 256, 64, 2, 1, 2, 4, 1, False, 0, True),
        (8160, 8160, 8160, 256, 256, 64, 2, 1, 2, 4, 1, False, 4, False),
        (8160, 8160, 8160, 256, 256, 64, 2, 1, 2, 4, 1, False, 4, True),
    ],
)
def test_hgemm_acc_main_loop(
    dtype: str,
    m: int,
    n: int,
    k: int,
    TILE_M: int,
    TILE_N: int,
    TILE_K: int,
    STAGES: int,
    SPLIT_K: int,
    BLOCK_M_WARPS: int,
    BLOCK_N_WARPS: int,
    BLOCK_K_WARPS: int,
    HAS_BIAS: bool,
    GROUP_M: int,
    USE_HALF_TILE_INTERLEAVED: bool,
):
    if dtype == "fp8_ptpc":
        TILE_K = 128
    else:
        dtype = torch.bfloat16 if "bf16" in dtype else torch.half
    args = _TestArgs(
        dtype,
        m,
        n,
        k,
        TILE_M,
        TILE_N,
        TILE_K,
        STAGES,
        SPLIT_K,
        BLOCK_M_WARPS,
        BLOCK_N_WARPS,
        BLOCK_K_WARPS,
        HAS_BIAS,
        GROUP_M,
        USE_HALF_TILE_INTERLEAVED,
    )
    check_acc(args)


@pytest.mark.parametrize("dtype", ["fp16", "bf16", "fp8_ptpc"])
@pytest.mark.parametrize(
    "m, n, k, TILE_M, TILE_N, TILE_K, STAGES, SPLIT_K, BLOCK_M_WARPS, BLOCK_N_WARPS, BLOCK_K_WARPS, HAS_BIAS, GROUP_M, USE_HALF_TILE_INTERLEAVED",
    [
        (32, 384, 7168, 32, 64, 64, 5, 8, 2, 2, 1, True, 0, False),
        (32, 384, 7168, 32, 64, 64, 5, 8, 2, 2, 1, False, 0, False),
        (32, 384, 7168, 32, 64, 64, 5, 8, 2, 2, 1, True, 4, False),
        (32, 384, 7168, 32, 64, 64, 5, 8, 2, 2, 1, False, 4, False),
    ],
)
def test_hgemm_acc_ft_stage_split_k(
    dtype: str,
    m: int,
    n: int,
    k: int,
    TILE_M: int,
    TILE_N: int,
    TILE_K: int,
    STAGES: int,
    SPLIT_K: int,
    BLOCK_M_WARPS: int,
    BLOCK_N_WARPS: int,
    BLOCK_K_WARPS: int,
    HAS_BIAS: bool,
    GROUP_M: int,
    USE_HALF_TILE_INTERLEAVED: bool,
):
    if dtype == "fp8_ptpc":
        TILE_K = 128
    else:
        dtype = torch.bfloat16 if "bf16" in dtype else torch.half
    args = _TestArgs(
        dtype,
        m,
        n,
        k,
        TILE_M,
        TILE_N,
        TILE_K,
        STAGES,
        SPLIT_K,
        BLOCK_M_WARPS,
        BLOCK_N_WARPS,
        BLOCK_K_WARPS,
        HAS_BIAS,
        GROUP_M,
        USE_HALF_TILE_INTERLEAVED,
    )
    check_acc(args)


@pytest.mark.parametrize("dtype", ["fp16", "bf16", "fp8_ptpc"])
@pytest.mark.parametrize(
    "m, n, k, TILE_M, TILE_N, TILE_K, STAGES, SPLIT_K, BLOCK_M_WARPS, BLOCK_N_WARPS, BLOCK_K_WARPS, HAS_BIAS, GROUP_M, USE_HALF_TILE_INTERLEAVED",
    [
        (64, 384, 7168, 64, 64, 64, 2, 8, 2, 2, 1, True, 0, True),
        (64, 384, 7168, 64, 64, 64, 2, 8, 2, 2, 1, False, 0, True),
        (64, 384, 7168, 64, 64, 64, 2, 8, 2, 2, 1, True, 4, True),
        (64, 384, 7168, 64, 64, 64, 2, 8, 2, 2, 1, False, 4, True),
        (2048, 2048, 2048, 128, 128, 64, 2, 4, 2, 2, 1, True, 0, True),
        (2048, 2048, 2048, 128, 128, 64, 2, 4, 2, 2, 1, False, 0, True),
        (2048, 2048, 2048, 128, 128, 64, 2, 4, 2, 2, 1, True, 4, True),
        (2048, 2048, 2048, 128, 128, 64, 2, 4, 2, 2, 1, False, 4, True),
    ],
)
def test_hgemm_acc_ht_split_k(
    dtype: str,
    m: int,
    n: int,
    k: int,
    TILE_M: int,
    TILE_N: int,
    TILE_K: int,
    STAGES: int,
    SPLIT_K: int,
    BLOCK_M_WARPS: int,
    BLOCK_N_WARPS: int,
    BLOCK_K_WARPS: int,
    HAS_BIAS: bool,
    GROUP_M: int,
    USE_HALF_TILE_INTERLEAVED: bool,
):
    if dtype == "fp8_ptpc":
        TILE_K = 128
    else:
        dtype = torch.bfloat16 if "bf16" in dtype else torch.half
    args = _TestArgs(
        dtype,
        m,
        n,
        k,
        TILE_M,
        TILE_N,
        TILE_K,
        STAGES,
        SPLIT_K,
        BLOCK_M_WARPS,
        BLOCK_N_WARPS,
        BLOCK_K_WARPS,
        HAS_BIAS,
        GROUP_M,
        USE_HALF_TILE_INTERLEAVED,
    )
    check_acc(args)


@pytest.mark.parametrize("dtype", ["fp16", "bf16", "fp8_ptpc"])
@pytest.mark.parametrize(
    "m, n, k, TILE_M, TILE_N, TILE_K, STAGES, SPLIT_K, BLOCK_M_WARPS, BLOCK_N_WARPS, BLOCK_K_WARPS, HAS_BIAS, GROUP_M, USE_HALF_TILE_INTERLEAVED",
    [
        (3, 5120, 2880, 64, 64, 64, 5, 3, 2, 2, 1, True, 0, False),
        (3, 5120, 2880, 64, 64, 64, 5, 3, 2, 2, 1, False, 0, False),
        (3, 5120, 2880, 64, 64, 64, 2, 3, 2, 2, 1, True, 0, True),
        (3, 5120, 2880, 64, 64, 64, 2, 3, 2, 2, 1, False, 0, True),
        (3, 5120, 2880, 64, 64, 64, 2, 3, 2, 2, 1, True, 4, True),
    ],
)
def test_hgemm_acc_small_m(
    dtype: str,
    m: int,
    n: int,
    k: int,
    TILE_M: int,
    TILE_N: int,
    TILE_K: int,
    STAGES: int,
    SPLIT_K: int,
    BLOCK_M_WARPS: int,
    BLOCK_N_WARPS: int,
    BLOCK_K_WARPS: int,
    HAS_BIAS: bool,
    GROUP_M: int,
    USE_HALF_TILE_INTERLEAVED: bool,
):
    if dtype == "fp8_ptpc":
        TILE_K = 128
    else:
        dtype = torch.bfloat16 if "bf16" in dtype else torch.half
    args = _TestArgs(
        dtype,
        m,
        n,
        k,
        TILE_M,
        TILE_N,
        TILE_K,
        STAGES,
        SPLIT_K,
        BLOCK_M_WARPS,
        BLOCK_N_WARPS,
        BLOCK_K_WARPS,
        HAS_BIAS,
        GROUP_M,
        USE_HALF_TILE_INTERLEAVED,
    )
    check_acc(args)


@pytest.mark.parametrize("dtype", ["fp16", "bf16", "fp8_ptpc"])
@pytest.mark.parametrize(
    "m, n, k, TILE_M, TILE_N, TILE_K, STAGES, SPLIT_K, BLOCK_M_WARPS, BLOCK_N_WARPS, BLOCK_K_WARPS, HAS_BIAS, GROUP_M, USE_HALF_TILE_INTERLEAVED",
    [
        # ft
        (3, 16, 16, 64, 64, 64, 2, 3, 2, 2, 1, False, 0, False),
        (3, 16, 16, 64, 64, 64, 2, 3, 2, 2, 1, True, 0, False),
        (3, 16, 48, 64, 64, 64, 2, 3, 2, 2, 1, True, 0, False),
        (3, 16, 80, 64, 64, 64, 2, 3, 2, 2, 1, True, 0, False),
        (3, 16, 144, 64, 64, 64, 2, 3, 2, 2, 1, True, 0, False),
        (3, 16, 144, 64, 64, 64, 3, 3, 2, 2, 1, True, 0, False),
        (3, 16, 16, 64, 64, 64, 4, 3, 2, 2, 1, True, 0, False),
        # ht
        (3, 16, 16, 64, 64, 64, 2, 3, 2, 2, 1, False, 0, True),
        (3, 16, 16, 64, 64, 64, 2, 3, 2, 2, 1, True, 0, True),
        (3, 16, 48, 64, 64, 64, 2, 3, 2, 2, 1, True, 0, True),
        (3, 16, 80, 64, 64, 64, 2, 3, 2, 2, 1, True, 0, True),
        (3, 16, 144, 64, 64, 64, 2, 3, 2, 2, 1, True, 0, True),
        (3, 16, 144, 64, 64, 64, 2, 3, 2, 2, 1, False, 4, True),
    ],
)
def test_hgemm_acc_small_mnk(
    dtype: str,
    m: int,
    n: int,
    k: int,
    TILE_M: int,
    TILE_N: int,
    TILE_K: int,
    STAGES: int,
    SPLIT_K: int,
    BLOCK_M_WARPS: int,
    BLOCK_N_WARPS: int,
    BLOCK_K_WARPS: int,
    HAS_BIAS: bool,
    GROUP_M: int,
    USE_HALF_TILE_INTERLEAVED: bool,
):
    if dtype == "fp8_ptpc":
        TILE_K = 128
    else:
        dtype = torch.bfloat16 if "bf16" in dtype else torch.half
    args = _TestArgs(
        dtype,
        m,
        n,
        k,
        TILE_M,
        TILE_N,
        TILE_K,
        STAGES,
        SPLIT_K,
        BLOCK_M_WARPS,
        BLOCK_N_WARPS,
        BLOCK_K_WARPS,
        HAS_BIAS,
        GROUP_M,
        USE_HALF_TILE_INTERLEAVED,
    )
    check_acc(args)


@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize(
    "m, n, k, TILE_M, TILE_N, TILE_K, STAGES, SPLIT_K, BLOCK_M_WARPS, BLOCK_N_WARPS, BLOCK_K_WARPS, HAS_BIAS, GROUP_M, USE_HALF_TILE_INTERLEAVED",
    [
        (800, 384, 7168, 32, 64, 128, 6, 1, 1, 2, 2, True, 0, False),
        (800, 384, 7168, 32, 64, 128, 6, 2, 1, 2, 2, True, 0, False),
        (800, 384, 7168, 32, 64, 128, 6, 1, 1, 2, 2, False, 0, False),
        (800, 384, 7168, 32, 64, 128, 6, 2, 1, 2, 2, False, 0, False),
        (800, 384, 7168, 32, 64, 128, 6, 2, 1, 2, 2, False, 4, False),
    ],
)
def test_hgemm_acc_ft_slice_k(
    dtype: str,
    m: int,
    n: int,
    k: int,
    TILE_M: int,
    TILE_N: int,
    TILE_K: int,
    STAGES: int,
    SPLIT_K: int,
    BLOCK_M_WARPS: int,
    BLOCK_N_WARPS: int,
    BLOCK_K_WARPS: int,
    HAS_BIAS: bool,
    GROUP_M: int,
    USE_HALF_TILE_INTERLEAVED: bool,
):
    if dtype == "fp8_ptpc":
        TILE_K = 128
    else:
        dtype = torch.bfloat16 if "bf16" in dtype else torch.half
    args = _TestArgs(
        dtype,
        m,
        n,
        k,
        TILE_M,
        TILE_N,
        TILE_K,
        STAGES,
        SPLIT_K,
        BLOCK_M_WARPS,
        BLOCK_N_WARPS,
        BLOCK_K_WARPS,
        HAS_BIAS,
        GROUP_M,
        USE_HALF_TILE_INTERLEAVED,
    )
    check_acc(args)


# =========================================== benchmark ===========================================


@pytest.mark.parametrize("dtype", ["bf16"])
@pytest.mark.parametrize(
    "m, n, k, TILE_M, TILE_N, TILE_K, STAGES, SPLIT_K, BLOCK_M_WARPS, BLOCK_N_WARPS, BLOCK_K_WARPS, HAS_BIAS, GROUP_M, USE_HALF_TILE_INTERLEAVED",
    [
        (16384, 16384, 16384, 256, 256, 64, 2, 1, 2, 4, 1, True, 4, True),
        (8192, 8192, 8192, 256, 256, 64, 2, 1, 2, 4, 1, True, 0, True),
        (8160, 8160, 8160, 256, 256, 64, 2, 1, 2, 4, 1, True, 0, True),
        (4096, 4096, 8192, 256, 256, 64, 2, 1, 2, 4, 1, True, 4, True),
        (4096, 4096, 4096, 256, 256, 64, 2, 1, 2, 4, 1, True, 0, True),
        (2048, 2048, 2048, 128, 128, 64, 5, 1, 4, 4, 1, True, 0, False),
    ],
)
def test_hgemm_benchmark_smoke(
    dtype: str,
    m: int,
    n: int,
    k: int,
    TILE_M: int,
    TILE_N: int,
    TILE_K: int,
    STAGES: int,
    SPLIT_K: int,
    BLOCK_M_WARPS: int,
    BLOCK_N_WARPS: int,
    BLOCK_K_WARPS: int,
    HAS_BIAS: bool,
    GROUP_M: int,
    USE_HALF_TILE_INTERLEAVED: bool,
):
    dtype = torch.bfloat16 if "bf16" in dtype else torch.half
    args = _TestArgs(
        dtype,
        m,
        n,
        k,
        TILE_M,
        TILE_N,
        TILE_K,
        STAGES,
        SPLIT_K,
        BLOCK_M_WARPS,
        BLOCK_N_WARPS,
        BLOCK_K_WARPS,
        HAS_BIAS,
        GROUP_M,
        USE_HALF_TILE_INTERLEAVED,
    )
    benchmark(args)
