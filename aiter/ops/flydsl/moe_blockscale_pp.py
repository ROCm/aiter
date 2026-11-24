# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Launchers for the persistent 8-wave block-FP8 MoE GEMMs (gfx950).

Block FP8 here means E4M3 values with power-of-two scales: 1x128 per activation
row block, 128x128 per weight block. The kernels take the scales as E8M0
exponent bytes, which is lossless for power-of-two fp32 scales
(``e8m0_from_pow2``).
"""

from __future__ import annotations

import functools

import torch
from torch import Tensor

BLOCK_M = 256


def e8m0_from_pow2(scale: Tensor) -> Tensor:
    """Exponent bytes of power-of-two fp32 scales (any shape); raises if one is not a power of two."""
    s = scale.float().contiguous()
    bits = s.view(torch.int32)
    if bool(((bits & 0x807FFFFF) != 0).any()):
        raise ValueError("block scales must be positive powers of two")
    return ((bits >> 23) & 0xFF).to(torch.uint8)


@functools.lru_cache(maxsize=64)
def _compile(K: int, N: int, gather: bool, gate_up_inter: int, waves_per_eu: int):
    from .kernels.moe_blockscale_pp.gemm import compile_moe_bs_gemm_8w

    return compile_moe_bs_gemm_8w(
        K=K,
        N=N,
        gather=gather,
        gate_up_inter=gate_up_inter,
        waves_per_eu=waves_per_eu,
    )


def _u8(t: Tensor) -> Tensor:
    return (t if t.dtype == torch.uint8 else t.view(torch.uint8)).reshape(-1)


def moe_bs_gemm(
    a: Tensor,
    a_scale: Tensor,
    w: Tensor,
    w_scale: Tensor,
    out: Tensor,
    sorted_ids: Tensor,
    sorted_expert_ids: Tensor,
    num_valid_ids: Tensor,
    *,
    gather: bool,
    gate_up_inter: int = 0,
    waves_per_eu: int = 2,
    grid: int | None = None,
) -> Tensor:
    """Grouped GEMM ``out[r] = a[row(r)] @ w[expert(r)].T`` over ``moe_sorting``'s 256-row tiles.

    a: fp8 ``[T, K]`` token-major (``gather=True``, row r reads token
    ``sorted_ids[r] & 0xFFFFFF``) with e8m0 ``a_scale [T, K // 128]``; or fp8
    ``[M_pad, K]`` already in sorted order (``gather=False``) with e8m0
    ``a_scale [K // 128, M_pad]``. w: fp8 ``[E, N, K]`` in
    ``shuffle_weight(w, (16, 16))`` layout, ``w_scale`` e8m0 ``[E, N // 128, K // 128]``.
    out: bf16 ``[M_pad, N]``, ``M_pad = sorted_ids.numel()``.
    """
    from .kernels.tensor_shim import _run_compiled, current_stream_arg, ptr_arg

    E, N, K = w.shape
    m_pad = sorted_ids.numel()
    assert m_pad % BLOCK_M == 0
    assert (
        out.shape == (m_pad, N) and out.dtype == torch.bfloat16 and out.is_contiguous()
    )
    assert a.shape[1] == K
    if gather:
        assert tuple(a_scale.shape) == (a.shape[0], K // 128)
    else:
        assert a.shape[0] == m_pad and tuple(a_scale.shape) == (K // 128, m_pad)
    assert tuple(w_scale.shape) == (E, N // 128, K // 128)
    if grid is None:
        grid = torch.cuda.get_device_properties(a.device).multi_processor_count
    assert grid % 8 == 0
    exe = _compile(K, N, bool(gather), int(gate_up_inter), int(waves_per_eu))
    _run_compiled(
        exe,
        _u8(a),
        ptr_arg(w),
        out.view(-1),
        _u8(a_scale),
        _u8(w_scale),
        sorted_ids,
        sorted_expert_ids,
        num_valid_ids,
        m_pad,
        grid,
        current_stream_arg(a.device),
        specialization_key=a.device,
    )
    return out
