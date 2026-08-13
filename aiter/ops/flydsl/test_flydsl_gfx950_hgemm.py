# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Focused integration tests for the layout-based gfx950 HGEMM."""

from __future__ import annotations

import pytest
import torch

from aiter.ops.flydsl.utils import is_flydsl_available


def _gfx950_flydsl_available() -> bool:
    if not torch.cuda.is_available() or not is_flydsl_available():
        return False
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    return str(getattr(props, "gcnArchName", "")).split(":")[0] == "gfx950"


if not _gfx950_flydsl_available():
    pytest.skip(
        "requires FlyDSL on a gfx950 ROCm device",
        allow_module_level=True,
    )


from aiter.ops.flydsl.gemm_kernels import (
    flydsl_gfx950_hgemm,
    flydsl_gfx950_kernel_name,
    get_flydsl_hgemm_kernel_params,
)


def test_gfx950_kernel_name_round_trip():
    name = flydsl_gfx950_kernel_name(
        dtype="bf16",
        out_dtype="f32",
        tile_m=64,
        tile_n=128,
        tile_k=64,
        stages=2,
        split_k=2,
        m_waves=2,
        n_waves=4,
        k_waves=1,
        has_bias=True,
        group_m=4,
        use_half_tile_interleaved=True,
    )

    config = get_flydsl_hgemm_kernel_params(name)

    assert config == {
        "kernel_family": "gfx950_a16w16",
        "dtype": "bf16",
        "out_dtype": "f32",
        "tile_m": 64,
        "tile_n": 128,
        "tile_k": 64,
        "stages": 2,
        "split_k": 2,
        "block_m_warps": 2,
        "block_n_warps": 4,
        "block_k_warps": 1,
        "has_bias": True,
        "group_m": 4,
        "policy": "hti",
        "use_half_tile_interleaved": True,
        "target_gfx": "gfx950",
        "async_copy": True,
        "b_to_lds": True,
        "b_preshuffle": False,
        "c_to_lds": False,
    }


@pytest.mark.parametrize("policy", ["ft", "hti"])
@pytest.mark.parametrize("split_k", [1, 2])
@pytest.mark.parametrize("has_bias", [False, True])
def test_gfx950_hgemm_matches_torch(
    policy: str,
    split_k: int,
    has_bias: bool,
):
    torch.manual_seed(20260813)
    m = n = 64
    k = 256
    a = torch.randn((m, k), dtype=torch.bfloat16, device="cuda")
    weight = torch.randn((n, k), dtype=torch.bfloat16, device="cuda")
    bias = torch.randn((n,), dtype=torch.bfloat16, device="cuda") if has_bias else None

    out = flydsl_gfx950_hgemm(
        a,
        weight,
        bias=bias,
        tile_m=64,
        tile_n=64,
        tile_k=64,
        split_k=split_k,
        block_m_warps=2,
        block_n_warps=2,
        block_k_warps=1,
        stages=2,
        group_m=0,
        policy=policy,
    )
    ref = torch.nn.functional.linear(a.float(), weight.float())
    if bias is not None:
        ref = ref + bias.float()
    ref = ref.to(a.dtype)

    torch.testing.assert_close(out, ref, atol=0.2, rtol=0.2)


@pytest.mark.parametrize("policy", ["ft", "hti"])
def test_gfx950_hgemm_explicit_stream_preallocated_tail(policy: str):
    torch.manual_seed(20260813)
    m, n, k = 65, 72, 256
    a = torch.randn((m, k), dtype=torch.bfloat16, device="cuda")
    weight = torch.randn((n, k), dtype=torch.bfloat16, device="cuda")
    bias = torch.randn((n,), dtype=torch.bfloat16, device="cuda")
    out = torch.empty((m, n), dtype=a.dtype, device=a.device)
    stream = torch.cuda.Stream(device=a.device)
    torch.cuda.synchronize(a.device)

    result = flydsl_gfx950_hgemm(
        a,
        weight,
        out=out,
        bias=bias,
        tile_m=64,
        tile_n=64,
        tile_k=64,
        split_k=2,
        block_m_warps=2,
        block_n_warps=2,
        block_k_warps=1,
        stages=2,
        group_m=4,
        policy=policy,
        stream=stream,
    )
    stream.synchronize()
    ref = torch.nn.functional.linear(a.float(), weight.float(), bias.float()).to(
        a.dtype
    )

    assert result.data_ptr() == out.data_ptr()
    torch.testing.assert_close(out, ref, atol=0.2, rtol=0.2)


def test_gfx950_hgemm_fp32_output():
    torch.manual_seed(20260813)
    m = n = 64
    k = 256
    a = torch.randn((m, k), dtype=torch.float16, device="cuda")
    weight = torch.randn((n, k), dtype=torch.float16, device="cuda")

    out = flydsl_gfx950_hgemm(
        a,
        weight,
        tile_m=64,
        tile_n=64,
        tile_k=64,
        block_m_warps=2,
        block_n_warps=2,
        policy="ft",
        out_dtype=torch.float32,
    )
    ref = torch.nn.functional.linear(a.float(), weight.float())

    assert out.dtype == torch.float32
    torch.testing.assert_close(out, ref, atol=0.2, rtol=0.05)
