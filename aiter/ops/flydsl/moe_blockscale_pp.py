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
def _compile(K, N, gather, gate_up_inter, epilogue, topk, waves_per_eu):
    from .kernels.moe_blockscale_pp.gemm import compile_moe_bs_gemm_8w

    return compile_moe_bs_gemm_8w(
        K=K,
        N=N,
        gather=gather,
        gate_up_inter=gate_up_inter,
        epilogue=epilogue,
        topk=topk,
        waves_per_eu=waves_per_eu,
    )


@functools.lru_cache(maxsize=16)
def _dummy(device: torch.device, dtype: torch.dtype) -> Tensor:
    return torch.zeros(16, dtype=dtype, device=device)


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
    epilogue: str = "bf16",
    out_scale: Tensor | None = None,
    sorted_weights: Tensor | None = None,
    topk: int = 0,
    swiglu_limit: float = 0.0,
    waves_per_eu: int = 2,
    grid: int | None = None,
) -> Tensor:
    """Grouped GEMM ``a[row(r)] @ w[expert(r)].T`` over ``moe_sorting``'s 256-row tiles.

    a: fp8 ``[T, K]`` token-major (``gather=True``, row r reads token
    ``sorted_ids[r] & 0xFFFFFF``) with e8m0 ``a_scale [T, K // 128]``; or fp8
    ``[M_pad, K]`` already in sorted order (``gather=False``) with e8m0
    ``a_scale [K // 128, M_pad]``. w: fp8 ``[E, N, K]`` in
    ``shuffle_weight(w, (16, 16))`` layout, ``w_scale`` e8m0 ``[E, N // 128, K // 128]``.
    ``M_pad = sorted_ids.numel()``. ``epilogue`` (see ``compile_moe_bs_gemm_8w``):

    * ``"bf16"``: out bf16 ``[M_pad, N]``.
    * ``"swiglu_fp8"``: out fp8 ``[M_pad, N // 2]``, out_scale e8m0
      ``[N // 256, M_pad]``; needs ``gate_up_inter = N // 2``.
    * ``"slot_bf16"``: out bf16 ``[T * topk, N]``, weighted by ``sorted_weights``.
    """
    from .kernels.tensor_shim import _run_compiled, current_stream_arg, ptr_arg

    E, N, K = w.shape
    m_pad = sorted_ids.numel()
    dev = a.device
    assert m_pad % BLOCK_M == 0
    assert a.shape[1] == K and out.is_contiguous()
    if gather:
        assert tuple(a_scale.shape) == (a.shape[0], K // 128)
    else:
        assert a.shape[0] == m_pad and tuple(a_scale.shape) == (K // 128, m_pad)
    assert tuple(w_scale.shape) == (E, N // 128, K // 128)
    if epilogue == "bf16":
        assert out.shape == (m_pad, N) and out.dtype == torch.bfloat16
    elif epilogue == "swiglu_fp8":
        assert gate_up_inter == N // 2 and out.shape == (m_pad, gate_up_inter)
        assert out.element_size() == 1 and out_scale is not None
        assert tuple(out_scale.shape) == (gate_up_inter // 128, m_pad)
    else:
        assert epilogue == "slot_bf16" and topk > 0 and sorted_weights is not None
        assert out.dtype == torch.bfloat16 and out.shape[1] == N
        assert out.shape[0] % topk == 0
    if grid is None:
        grid = torch.cuda.get_device_properties(dev).multi_processor_count
    assert grid % 8 == 0
    # The kernel reads "no clamp" as an infinite limit.
    limit = float(swiglu_limit) if swiglu_limit > 0 else 3.0e38
    exe = _compile(
        K,
        N,
        bool(gather),
        int(gate_up_inter),
        epilogue,
        int(topk) if epilogue == "slot_bf16" else 0,
        int(waves_per_eu),
    )
    _run_compiled(
        exe,
        _u8(a),
        ptr_arg(w),
        out.view(-1) if out.dtype == torch.bfloat16 else _u8(out),
        _u8(a_scale),
        _u8(w_scale),
        sorted_ids,
        sorted_expert_ids,
        num_valid_ids,
        _u8(out_scale) if out_scale is not None else _dummy(dev, torch.uint8),
        (
            sorted_weights.view(torch.int32)
            if sorted_weights is not None
            else _dummy(dev, torch.int32)
        ),
        m_pad,
        limit,
        grid,
        current_stream_arg(dev),
        specialization_key=dev,
    )
    return out


def moe_bs_forward(
    x: Tensor,
    topk_ids: Tensor,
    topk_weights: Tensor,
    w13: Tensor,
    w13_scale: Tensor,
    w2: Tensor,
    w2_scale: Tensor,
    *,
    out: Tensor | None = None,
    swiglu_limit: float = 0.0,
    waves_per_eu: int = 2,
) -> Tensor:
    """Block-FP8 MoE layer forward: ``sum_k w_k * W2[e_k] act(W13[e_k] x)``.

    x bf16 ``[T, D]``; topk_ids / topk_weights ``[T, topk]`` (ids outside
    ``[0, E)`` contribute nothing); w13 fp8 ``[E, 2I, D]`` (gate rows, then up
    rows) and w2 fp8 ``[E, D, I]`` in ``shuffle_weight(w, (16, 16))`` layout;
    w13_scale / w2_scale their 128x128 block scales as e8m0 exponent bytes
    (``e8m0_from_pow2`` of power-of-two fp32 scales). act = ``silu(min(g, L)) *
    clamp(u, -L, L)`` for ``swiglu_limit`` L > 0, else ``silu(g) * u``.
    Activations are quantized to fp8 with power-of-two 1x128 scales before
    each GEMM. Returns bf16 ``[T, D]``.
    """
    from aiter import dtypes
    from aiter.fused_moe import moe_sorting
    from aiter.ops.quant import per_group_quant_hip

    from .moe_kernels import _run_moe_reduction

    T, D = x.shape
    E, I2, _ = w13.shape
    inter = I2 // 2
    topk = topk_ids.shape[1]
    if out is None:
        out = torch.empty_like(x)
    if T == 0:
        return out
    ids = topk_ids.to(torch.int32)
    # Out-of-range experts become dropped routes (-1): the sort leaves them
    # out and the reduction skips their never-written slot rows.
    ids = torch.where((ids >= 0) & (ids < E), ids, -1)
    sorted_ids, sorted_w, sorted_eids, num_valid, _ = moe_sorting(
        ids, topk_weights.float(), E, D, x.dtype, BLOCK_M, accumulate=False
    )
    m_pad = sorted_ids.numel()
    xq, xs = per_group_quant_hip(
        x, quant_dtype=dtypes.fp8, group_size=128, scale_type=dtypes.fp8_e8m0
    )
    a2 = torch.empty(m_pad, inter, dtype=dtypes.fp8, device=x.device)
    a2s = torch.empty(inter // 128, m_pad, dtype=torch.uint8, device=x.device)
    moe_bs_gemm(
        xq,
        xs,
        w13,
        w13_scale,
        a2,
        sorted_ids,
        sorted_eids,
        num_valid,
        gather=True,
        gate_up_inter=inter,
        epilogue="swiglu_fp8",
        out_scale=a2s,
        swiglu_limit=swiglu_limit,
        waves_per_eu=waves_per_eu,
    )
    slots = torch.empty(T * topk, D, dtype=torch.bfloat16, device=x.device)
    moe_bs_gemm(
        a2,
        a2s,
        w2,
        w2_scale,
        slots,
        sorted_ids,
        sorted_eids,
        num_valid,
        gather=False,
        epilogue="slot_bf16",
        sorted_weights=sorted_w,
        topk=topk,
        waves_per_eu=waves_per_eu,
    )
    _run_moe_reduction(slots, out, T, topk, D, topk_ids=ids, skip_dropped=True)
    return out
