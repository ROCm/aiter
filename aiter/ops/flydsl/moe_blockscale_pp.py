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
import weakref

import torch
from torch import Tensor

BLOCK_M = 256
# Stage 2's [T * topk, D] bf16 slot buffer is addressed through 32-bit buffer
# ranges: moe_bs_forward runs larger T in token chunks no bigger than the
# largest validated buffer (T 32768, top-k 9, D 4096).
_MAX_SLOT_BYTES = 32768 * 9 * 4096 * 2


def e8m0_from_pow2(scale: Tensor) -> Tensor:
    """Exponent bytes of power-of-two fp32 scales (any shape); raises if one is not a power of two."""
    s = scale.float().contiguous()
    bits = s.view(torch.int32)
    if bool(((bits & 0x807FFFFF) != 0).any()):
        raise ValueError("block scales must be positive powers of two")
    return ((bits >> 23) & 0xFF).to(torch.uint8)


@functools.lru_cache(maxsize=64)
def _compile(K, N, gather, topk, swiglu_clamp, half_skip):
    # Call positionally: lru_cache keys keyword and positional calls apart.
    from .kernels.moe_blockscale_pp.gemm import compile_moe_bs_gemm_8w

    return compile_moe_bs_gemm_8w(
        K=K,
        N=N,
        gather=gather,
        topk=topk,
        swiglu_clamp=swiglu_clamp,
        half_skip=half_skip,
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
    out_scale: Tensor | None = None,
    topk: int = 0,
    swiglu_limit: float = 0.0,
    half_skip: bool = False,
) -> Tensor:
    """One stage, ``a[row(r)] @ w[expert(r)].T``, over ``moe_sorting``'s 256-row tiles.

    w: fp8 ``[E, N, K]`` in ``shuffle_weight(w, (16, 16))`` layout, ``w_scale``
    e8m0 ``[E, N // 128, K // 128]``; ``M_pad = sorted_ids.numel()``.

    * ``gather=True`` (stage 1): a fp8 ``[T, K]`` token-major (row r reads
      token ``sorted_ids[r] & 0xFFFFFF``), ``a_scale`` e8m0 ``[T, K // 128]``;
      ``w`` gate rows then up rows. out fp8 ``[M_pad, N // 2]`` = requantized
      SwiGLU, ``out_scale`` e8m0 ``[N // 256, M_pad]``. ``half_skip`` skips a
      wave's row-half-1 MFMAs when its 64 rows are all padding.
    * ``gather=False`` (stage 2): a fp8 ``[M_pad, K]`` in sorted order,
      ``a_scale`` e8m0 ``[K // 128, M_pad]``. out bf16 ``[T * topk, N]``,
      unweighted: the top-k reduction applies the routing weights.
    """
    from .kernels.tensor_shim import _run_compiled, current_stream_arg, ptr_arg

    E, N, K = w.shape
    m_pad = sorted_ids.numel()
    dev = a.device
    assert m_pad % BLOCK_M == 0
    assert a.shape[1] == K and out.is_contiguous()
    assert tuple(w_scale.shape) == (E, N // 128, K // 128)
    if gather:
        assert tuple(a_scale.shape) == (a.shape[0], K // 128)
        assert out.shape == (m_pad, N // 2) and out.element_size() == 1
        assert out_scale is not None
        assert tuple(out_scale.shape) == (N // 256, m_pad)
    else:
        assert a.shape[0] == m_pad and tuple(a_scale.shape) == (K // 128, m_pad)
        assert topk > 0 and out.dtype == torch.bfloat16 and out.shape[1] == N
        assert out.shape[0] % topk == 0
    grid = torch.cuda.get_device_properties(dev).multi_processor_count
    assert grid % 8 == 0
    exe = _compile(
        K,
        N,
        bool(gather),
        0 if gather else int(topk),
        not gather or swiglu_limit > 0,  # swiglu_clamp: off for no limit
        gather and bool(half_skip),
    )
    _run_compiled(
        exe,
        _u8(a),
        ptr_arg(w),
        _u8(out) if gather else out.view(-1),
        _u8(a_scale),
        _u8(w_scale),
        sorted_ids,
        sorted_expert_ids,
        num_valid_ids,
        _u8(out_scale) if gather else _dummy(dev, torch.uint8),
        m_pad,
        # The kernel reads "no clamp" as an infinite limit.
        float(swiglu_limit) if swiglu_limit > 0 else 3.0e38,
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
) -> Tensor:
    """Block-FP8 MoE layer forward: ``sum_k w_k * W2[e_k] act(W13[e_k] x)``.

    x bf16 ``[T, D]``; topk_ids / topk_weights ``[T, topk]`` (ids outside
    ``[0, E)`` contribute nothing; a token's ids must be distinct, as from a
    top-k router: ``moe_sorting`` keeps one route of a repeated expert); w13
    fp8 ``[E, 2I, D]`` (gate rows, then up rows) and w2 fp8 ``[E, D, I]`` in ``shuffle_weight(w, (16, 16))`` layout;
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
    chunk = max(1, _MAX_SLOT_BYTES // (topk * D * 2))
    if T > chunk:
        step = -(-T // -(-T // chunk))
        for t0 in range(0, T, step):
            rows = slice(t0, t0 + step)
            moe_bs_forward(
                x[rows],
                topk_ids[rows],
                topk_weights[rows],
                w13,
                w13_scale,
                w2,
                w2_scale,
                out=out[rows],
                swiglu_limit=swiglu_limit,
            )
        return out
    # The sort and the reduction both skip ids outside [0, E): no masking pass.
    ids = topk_ids if topk_ids.dtype == torch.int32 else topk_ids.to(torch.int32)
    sorted_ids, _, sorted_eids, num_valid, _ = moe_sorting(
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
        out_scale=a2s,
        swiglu_limit=swiglu_limit,
        # HK's HALF_SKIP rule: experts average at most 144 of a tile's 256
        # rows (stage 1 -7% at T = 4096; at T = 16384 the branches cost +4%).
        half_skip=T * topk <= 144 * E,
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
        topk=topk,
    )
    # Applies the routing weights (stage 2 does not). Nontemporal loads pay
    # once the slot rows outgrow the 256 MB MALL (T=16384: 239 -> 204 us); at
    # 268 MB (T=4096, top-k 8) they cost 5 us.
    _run_moe_reduction(
        slots,
        out,
        T,
        topk,
        D,
        topk_ids=ids,
        topk_weights=topk_weights,
        skip_dropped=True,
        num_experts=E,
        streaming_loads=slots.numel() * slots.element_size() > (280 << 20),
    )
    return out


# id(scale) -> (weak reference to it, its _version, e8m0 bytes). Keyed by id:
# a WeakKeyDictionary compares keys with ==, which tensors do elementwise.
_E8M0_CACHE: dict[int, tuple[weakref.ref, int, Tensor]] = {}


def _e8m0_cached(scale: Tensor) -> Tensor:
    """E8M0 bytes of power-of-two fp32 block scales, converted once per tensor version.

    Outside graph capture the first conversion checks that every scale is a
    power of two; a warm-up call before capture fills the cache.
    """
    if scale.dtype == torch.uint8:
        return scale
    hit = _E8M0_CACHE.get(id(scale))
    if hit is not None and hit[0]() is scale and hit[1] == scale._version:
        return hit[2]
    if torch.cuda.is_current_stream_capturing():
        bits = scale.float().contiguous().view(torch.int32)
        e8m0 = ((bits >> 23) & 0xFF).to(torch.uint8)
    else:
        e8m0 = e8m0_from_pow2(scale)
    key = id(scale)
    _E8M0_CACHE[key] = (
        weakref.ref(scale, lambda _, key=key: _E8M0_CACHE.pop(key, None)),
        scale._version,
        e8m0,
    )
    return e8m0


def run_moe_bs_pp_impl(request, config_string: str) -> Tensor:
    """``fused_moe`` whole-forward backend for ``impl__flydsl_bspp__default`` tuned rows.

    gfx950 per_1x128 block-FP8 MoE with power-of-two scales: fp8 w1
    ``[E, 2I, D]`` (gate rows, then up rows) and w2 ``[E, D, I]`` in
    ``shuffle_weight(w, (16, 16))`` layout marked ``is_shuffled``, fp32
    ``[E, N / 128, K / 128]`` block scales, bf16 activations, SiLU with an
    optional ``swiglu_limit``. Anything else raises NotImplementedError.
    """
    from aiter import ActivationType, QuantType
    from aiter.jit.utils.chip_info import get_gfx_runtime

    if config_string != "default":
        raise NotImplementedError(f"flydsl_bspp: unknown config {config_string!r}")
    gate_mode = getattr(request.gate_mode, "value", request.gate_mode)
    w1, w2 = request.w1, request.w2
    unsupported = None
    if get_gfx_runtime() != "gfx950":
        unsupported = f"arch {get_gfx_runtime()}"
    elif QuantType(request.quant_type) != QuantType.per_1x128:
        unsupported = f"quant type {request.quant_type}"
    elif ActivationType(request.activation) != ActivationType.Silu:
        unsupported = f"activation {request.activation}"
    elif w1.dtype != torch.float8_e4m3fn or w2.dtype != torch.float8_e4m3fn:
        unsupported = "weights other than fp8 e4m3"
    elif not (getattr(w1, "is_shuffled", False) and getattr(w2, "is_shuffled", False)):
        unsupported = "weights not marked is_shuffled"
    elif w1.shape[1] != 2 * w2.shape[2] or w1.shape[2] != w2.shape[1]:
        unsupported = f"weight shapes {tuple(w1.shape)} / {tuple(w2.shape)}"
    elif request.expert_mask is not None:
        unsupported = "expert parallelism (expert_mask)"
    elif request.bias1 is not None or request.bias2 is not None:
        unsupported = "per-expert bias"
    elif request.doweight_stage1:
        unsupported = "doweight_stage1=True"
    elif request.a1_scale is not None or request.a2_scale is not None:
        unsupported = "prequantized activations"
    elif request.hidden_pad or request.intermediate_pad:
        unsupported = "hidden/intermediate padding"
    elif gate_mode not in (None, "separated"):
        unsupported = f"gate mode {gate_mode!r}"
    elif request.hidden_states.dtype != torch.bfloat16 or request.dtype not in (
        None,
        torch.bfloat16,
    ):
        unsupported = "activations / output other than bf16"
    elif request.ksplit:
        unsupported = "split-K"
    if unsupported is not None:
        raise NotImplementedError(f"flydsl_bspp whole-forward backend: {unsupported}")
    limit = request.swiglu_limit
    return moe_bs_forward(
        request.hidden_states,
        request.topk_ids,
        request.topk_weight,
        w1,
        _e8m0_cached(request.w1_scale),
        w2,
        _e8m0_cached(request.w2_scale),
        swiglu_limit=float(limit) if limit else 0.0,
    )
