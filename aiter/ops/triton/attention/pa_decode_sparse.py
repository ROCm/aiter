# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Sparse paged-decode attention over a unified KV pool with per-token paged
indices. See ``_triton_kernels/attention/pa_decode_sparse.py`` for the
kernels' caller contract.

This module exposes ``pa_decode_sparse`` — a 3D split-K + widened-BLOCK_H
+ pipelined-K-loop variant suitable for sparse decode (e.g. V4 top-k gather)
where each token's K range is an unordered subset of a unified KV pool.

``pa_decode_sparse_shuffled`` is the same split-K decode over a *paged* SHUFFLE
K/V cache pair selected by a per-token block table — block-sparse GQA decode
(MiniMax-M3), where the top-k selection has already been compacted into a dense
page list plus an exact context length.

On gfx950 (CDNA4) DeepSeek-V4 sparse-MLA decode has a dedicated gluon
implementation (bottom of this module): ``pa_decode_sparse`` routes all formats
to the merged ``_pa_decode_sparse_gfx950_gluon`` driver -- packed fp8_ds_mla /
bf16 block cache (3D; optional SWA+top-k two-loop via ``extra_*``) and the
uniform fp8 / bf16 pool (2D).
"""

import math

import torch
import triton

from aiter.ops.triton._gluon_kernels.gfx950.attention.pa_decode_sparse import (
    _pa_decode_sparse as _pa_decode_sparse_gfx950,
)
from aiter.ops.triton._gluon_kernels.gfx950.attention.pa_decode_sparse import (
    _pa_decode_sparse_reduce as _pa_decode_sparse_reduce_gfx950,
)
from aiter.ops.triton._gluon_kernels.gfx1250.attention.pa_decode_sparse import (
    _pa_decode_sparse as gluon_pa_decode_sparse,
)
from aiter.ops.triton._gluon_kernels.gfx1250.attention.pa_decode_sparse import (
    _pa_decode_sparse_reduce as gluon_pa_decode_sparse_reduce,
)
from aiter.ops.triton._triton_kernels.attention.pa_decode_sparse import (
    _pa_decode_sparse as triton_pa_decode_sparse,
)
from aiter.ops.triton._triton_kernels.attention.pa_decode_sparse import (
    _pa_decode_sparse_reduce as triton_pa_decode_sparse_reduce,
)
from aiter.ops.triton._triton_kernels.attention.pa_decode_sparse import (
    _pa_decode_sparse_shuffled as triton_pa_decode_sparse_shuffled,
)
from aiter.ops.triton.utils._triton import arch_info
from aiter.ops.triton.utils.common_utils import max_addressable_bytes
from aiter.ops.triton.utils.device_info import get_num_sms
from aiter.ops.triton.utils.logger import AiterTritonLogger

DEVICE_ARCH = arch_info.get_arch()

_LOGGER = AiterTritonLogger()


_FP8_GROUP_SIZE = 64
_FP8_DTYPE = torch.float8_e4m3fnuz


def pa_decode_sparse(
    q: torch.Tensor,
    unified_kv: torch.Tensor,
    kv_indices: torch.Tensor,
    kv_indptr: torch.Tensor,
    attn_sink: torch.Tensor,
    softmax_scale: float,
    kv_scales: torch.Tensor | None = None,
    block_h: int | None = None,
    kv_splits: int | None = None,
    has_invalid: bool | None = True,
    skip_reduce: bool | None = False,
    USE_EXP2: bool | None = None,
    *,
    extra_cache: torch.Tensor | None = None,
    extra_indices: torch.Tensor | None = None,
    extra_indptr: torch.Tensor | None = None,
) -> torch.Tensor:
    """Sparse paged-decode attention with split-K + widened BLOCK_H.

    Args:
        q: ``[N, H, D]`` decode queries, bf16/fp16.
        unified_kv: ``[total_pages, D]`` shared KV pool (page_size=1), same dtype as ``q``.
        kv_indices: ``[total_indices]`` int32 — per-token slot lists, flat.
            Per-token entries live in ``kv_indices[kv_indptr[t] : kv_indptr[t+1]]``.
            ``-1`` entries are skipped (sentinel for unused tail).
        kv_indptr: ``[N+1]`` int32 — true prefix sum.
        attn_sink: ``[H]`` per-head learnable softmax-denom bias (fp32).
        softmax_scale: scalar softmax scale.
        block_h: override ``BLOCK_H`` for the split kernel. Default picks
            ``next_pow2(min(H, 64))``, rounded up to the AMD MFMA min tile (16).
        kv_splits: override ``KV_SPLITS`` for the split-K grid axis. Default
            auto-infers to fill ~512 total CTAs while capping below the number
            of K-blocks, then rounds up to a power of 2.
        num_stages: software-pipeline depth of the K loop (default 2).
        skip_reduce: when the split-K path is active (``kv_splits > 1``), return
            the pre-reduce ``(acc_partial, m_partial, l_partial)`` partials
            instead of launching the reduce kernel. Has no effect when
            ``kv_splits == 1`` (the single-CTA path already produces the final
            ``out`` directly). Useful for profiling the main kernel in
            isolation and for callers that fold the reduce into a downstream op.
        extra_cache/extra_indices/extra_indptr: gfx950 packed-only — the SWA+top-k
            two-loop's second (top-k) cache + index set; must be None otherwise.

    On gfx950 the DSv4 gluon driver handles this: a 3D ``unified_kv`` selects the
    packed fp8_ds_mla / bf16 block cache (``extra_*`` = the two-loop), a 2D one the
    uniform pool (``kv_scales`` present = fp8). ``kv_splits``/``skip_reduce`` are
    honored; ``block_h`` and fp16 ``q`` fall through to the triton path.

    Returns:
        ``[N, H, D]`` attention output, same dtype as ``q``. When
        ``skip_reduce`` is set and ``kv_splits > 1`` instead returns the tuple
        ``(acc_partial, m_partial, l_partial)`` with shapes
        ``([N, KV_SPLITS, H_padded, D], [N, KV_SPLITS, H_padded],
        [N, KV_SPLITS, H_padded])`` (all fp32).

    Optimizations targeted:
      (1) Wider ``BLOCK_H`` so all heads of a token are handled by one CTA →
          eliminates MLA-style KV re-fetch across head-block programs.
      (2) ``num_stages`` on the K loop pipelines KV gather behind the dot.
      (3) Split the K dimension across CTAs via a third grid axis →
          fixes grid undersubscription on long-context decode.
    """
    if not q.is_cuda:
        raise RuntimeError("pa_decode_sparse requires CUDA/HIP tensors")
    if q.dtype not in (torch.bfloat16, torch.float16):
        raise RuntimeError(f"pa_decode_sparse expects fp16/bf16 q, got {q.dtype}")

    # gfx950: route to the merged DSv4 sparse-MLA gluon driver. Format is inferred
    # from the cache: 3D -> packed fp8_ds_mla / bf16 block cache (optional SWA+top-k
    # two-loop via extra_*); 2D -> uniform pool (OCP fp8 + fp32 kv_scales, or bf16).
    # kv_splits and skip_reduce are honored here; block_h and fp16 q fall through to
    # the triton path below (the gluon kernel is bf16-only: bf16 LDS + bf16 MFMA).
    if DEVICE_ARCH == "gfx950" and block_h is None and q.dtype == torch.bfloat16:
        if unified_kv.ndim == 3:
            _ok = kv_scales is None and (
                unified_kv.dtype == torch.uint8 or unified_kv.dtype == q.dtype
            )
        else:
            _fp8 = unified_kv.dtype in (
                torch.float8_e4m3fn,
                torch.float8_e4m3fnuz,
                torch.uint8,
            )
            _ok = (kv_scales is not None and _fp8) or (
                kv_scales is None and unified_kv.dtype == q.dtype
            )
        # fnuz vs OCP e4m3 (2D fp8 only) selects the in-kernel dequant bias.
        fp8_fnuz = unified_kv.ndim == 2 and unified_kv.dtype == torch.float8_e4m3fnuz
        if _ok:
            cache = (
                unified_kv.view(torch.uint8)
                if (unified_kv.ndim == 2 and kv_scales is not None)
                else unified_kv
            )
            return _pa_decode_sparse_gfx950_gluon(
                q,
                cache,
                kv_scales,
                kv_indices,
                kv_indptr,
                softmax_scale,
                attn_sink,
                extra_cache=extra_cache,
                extra_indices=extra_indices,
                extra_indptr=extra_indptr,
                kv_splits=kv_splits,
                skip_reduce=skip_reduce,
                has_invalid=bool(has_invalid),
                fp8_fnuz=fp8_fnuz,
            )

    assert (
        extra_cache is None and extra_indices is None and extra_indptr is None
    ), "extra_cache/extra_indices/extra_indptr are gfx950 packed-only"

    quant_kv = kv_scales is not None
    if quant_kv:
        assert unified_kv.dtype == _FP8_DTYPE, (
            f"kv_scales supplied but unified_kv is {unified_kv.dtype}, "
            f"expected {_FP8_DTYPE}"
        )
        assert (
            kv_scales.dtype == torch.float32
        ), f"kv_scales must be fp32, got {kv_scales.dtype}"
        D_check = unified_kv.shape[-1]
        assert (
            D_check % _FP8_GROUP_SIZE == 0
        ), f"D={D_check} must be divisible by GROUP_SIZE={_FP8_GROUP_SIZE}"
        expected_g = D_check // _FP8_GROUP_SIZE
        assert kv_scales.shape == (unified_kv.shape[0], expected_g), (
            f"kv_scales shape {tuple(kv_scales.shape)} does not match "
            f"expected ({unified_kv.shape[0]}, {expected_g})"
        )
        assert kv_scales.is_contiguous()
    else:
        if unified_kv.dtype != q.dtype:
            raise RuntimeError(
                f"unified_kv dtype mismatch: kv={unified_kv.dtype}, q={q.dtype}"
            )

    T, H, D = q.shape
    _LOGGER.info(
        f"PA_DECODE_SPARSE T={T} H={H} D={D} " f"total_indices={kv_indices.shape[0]}"
    )

    out = torch.empty_like(q)
    assert kv_indices.dtype == torch.int32 and kv_indices.is_contiguous()
    assert kv_indptr.dtype == torch.int32 and kv_indptr.is_contiguous()

    use_gluon = DEVICE_ARCH == "gfx1250"

    if block_h is None:
        # Default: one CTA per token (kills the H/BLOCK_H KV duplication).
        # If H is too large to fit a single tile, halve until it does.
        if use_gluon:
            if H >= 128:
                block_h = 128
            elif H >= 64:
                if T >= 2048:
                    block_h = 64
                elif T >= 32:
                    block_h = 32
                else:
                    block_h = 16
            elif H >= 32:
                if T >= 256:
                    block_h = 32
                else:
                    block_h = 16
            else:
                block_h = triton.next_power_of_2(H)
        else:
            block_h = triton.next_power_of_2(min(H, 16))
    else:
        block_h = triton.next_power_of_2(block_h)
    block_h = max(block_h, 16)  # AMD MFMA min tile

    n_head_blocks = triton.cdiv(H, block_h)
    h_padded = n_head_blocks * block_h
    block_d = triton.next_power_of_2(D)
    assert block_d == D

    # gfx1250 stages slots through LDS via TDM async_load, which hides the
    # larger per-tile KV gather latency -> BLOCK_K=32 is fastest there. Other
    # arches use the synchronous slot path, where 32 exposes memory latency.
    if use_gluon:
        block_k = 16
        waves_per_eu = 1
        if block_h == 128:
            block_k = 32
            attn_num_warps = 8
            max_num_wg = 256
            waves_per_eu = 2
        elif block_h == 64:
            attn_num_warps = 4
            max_num_wg = 256
        elif block_h == 32:
            attn_num_warps = 2
            max_num_wg = 512
        else:
            attn_num_warps = 1
            max_num_wg = 1024
    else:
        block_k = 16 if D >= 256 else 32
        attn_num_warps = 4
        max_num_wg = 256
        waves_per_eu = 1
    num_stages = 2
    # gluon reduce with BLOCK_H=1 keeps KV_SPLITS and BLOCK_H entirely
    # in-thread; a single warp suffices and avoids shared-memory layout
    # mismatches between 2D (m/l) and 3D (acc) loads.
    reduce_num_warps = 1 if use_gluon else 4
    reduce_waves_per_eu = 4 if use_gluon else 1
    USE_EXP2 = True

    # Infer KV_SPLITS from inputs when caller doesn't override.
    # Fill ~512 total CTAs (MI300X has 304 CUs) while never splitting K into
    # more pieces than there are K-blocks. Rounded up to a power of 2 so the
    # reduce kernel's tl.arange(0, KV_SPLITS) compiles; over-splitting past
    # max_kv_splits is handled by the kernel (empty splits early-return and
    # the reduce masks their stale partial-buffer slots).
    # print(f"{kv_indices.shape[0]=}")
    if kv_splits is None:
        max_kv_len = kv_indices.shape[0]
        max_kv_splits = max(1, triton.cdiv(max_kv_len, block_k))
        kv_splits = max(1, max_num_wg // max(1, T * n_head_blocks))
        kv_splits = min(max_kv_splits, kv_splits)
        kv_splits = triton.next_power_of_2(kv_splits)

    if use_gluon:
        _lds_budget = arch_info._LDS_CAP_BYTES.get(DEVICE_ARCH)
        _lds_cap = max(1, _lds_budget // (block_d * 4))
        kv_splits = min(kv_splits, 1 << (_lds_cap.bit_length() - 1))
        if kv_splits > 8:
            reduce_num_warps = 4
            reduce_waves_per_eu = 1

    if kv_splits == 1:
        m_partial = l_partial = acc_partial = out  # unused inside the kernel
        mp_strides = (0, 0, 0)
        lp_strides = (0, 0, 0)
        ap_strides = (0, 0, 0, 0)
    else:
        m_partial = torch.empty(
            (T, kv_splits, h_padded), dtype=torch.float32, device=q.device
        )
        l_partial = torch.empty_like(m_partial)
        acc_partial = torch.empty(
            (T, kv_splits, h_padded, D), dtype=torch.float32, device=q.device
        )
        mp_strides = m_partial.stride()
        lp_strides = l_partial.stride()
        ap_strides = acc_partial.stride()

    if quant_kv:
        kv_scales_arg = kv_scales
        ks_stride_n_arg = kv_scales.stride(0)
        num_groups_arg = D // _FP8_GROUP_SIZE
    else:
        kv_scales_arg = q.new_empty(1, dtype=torch.float32)
        ks_stride_n_arg = 1
        num_groups_arg = 1

    if use_gluon:
        impl = gluon_pa_decode_sparse
        reduce_impl = gluon_pa_decode_sparse_reduce
    else:
        impl = triton_pa_decode_sparse
        reduce_impl = triton_pa_decode_sparse_reduce

    grid_attn = (T, n_head_blocks, kv_splits)
    impl[grid_attn](
        q,
        unified_kv,
        kv_scales_arg,
        kv_indices,
        kv_indptr,
        m_partial,
        l_partial,
        acc_partial,
        attn_sink,
        out,
        unified_kv.shape[0],
        q.stride(0),
        q.stride(1),
        q.stride(2),
        unified_kv.stride(0),
        unified_kv.stride(1),
        ks_stride_n_arg,
        mp_strides[0],
        mp_strides[1],
        mp_strides[2],
        lp_strides[0],
        lp_strides[1],
        lp_strides[2],
        ap_strides[0],
        ap_strides[1],
        ap_strides[2],
        ap_strides[3],
        out.stride(0),
        out.stride(1),
        out.stride(2),
        H,
        D,
        kv_splits,
        float(softmax_scale),
        BLOCK_H=block_h,
        BLOCK_D=block_d,
        BLOCK_K=block_k,
        HAS_INVALID=has_invalid,
        QUANT_KV=quant_kv,
        GROUP_SIZE=_FP8_GROUP_SIZE,
        NUM_GROUPS=num_groups_arg,
        USE_EXP2=USE_EXP2,
        num_warps=attn_num_warps,
        num_stages=num_stages,
        waves_per_eu=waves_per_eu,
    )

    if kv_splits == 1:
        return out

    if skip_reduce:
        # Hand back the pre-reduce partials; the caller (or a downstream op)
        # is responsible for the log-sum-exp combine + sink fold.
        return acc_partial, m_partial, l_partial

    # One reduce CTA per head. For small per-rank H (TP=8 → H ∈ {8, 16}) this
    # multiplies the reduce-side CTA count by H, replacing the previous single
    # under-occupied CTA per token with a small fan-out that hides launch
    # latency. tl.arange(0, 1) is a valid power-of-2 range.
    block_h_reduce = 1
    grid_reduce = (T, triton.cdiv(H, block_h_reduce))

    reduce_impl[grid_reduce](
        m_partial,
        l_partial,
        acc_partial,
        attn_sink,
        kv_indptr,
        out,
        m_partial.stride(0),
        m_partial.stride(1),
        m_partial.stride(2),
        l_partial.stride(0),
        l_partial.stride(1),
        l_partial.stride(2),
        acc_partial.stride(0),
        acc_partial.stride(1),
        acc_partial.stride(2),
        acc_partial.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        H,
        D,
        kv_splits,
        BLOCK_H=block_h_reduce,
        BLOCK_D=block_d,
        BLOCK_K=block_k,
        USE_EXP2=USE_EXP2,
        num_warps=reduce_num_warps,
        waves_per_eu=reduce_waves_per_eu,
    )
    return out


def _as_int32_contiguous_1d(x: torch.Tensor) -> torch.Tensor:
    if x.dtype == torch.int32 and x.ndim == 1 and x.is_contiguous():
        return x
    return x.to(torch.int32).contiguous()


def _decode_num_splits(
    num_queries, heads_blocks, avg_main=0.0, avg_extra=0.0, block_k=64
):
    """Pick the split-K count by minimizing a cost model of the decode work:

        cost(s) = waves(s) * iters(s)  +  GAMMA * s  +  DELTA * fill(s)
        s = # splits
    Tuned on gfx950 for DSv4 decode (H=16, D=512, BLOCK_K=64); split count is
    capped at 16.
    """
    cu = max(1, get_num_sms())
    base = max(1, num_queries * heads_blocks)
    GAMMA, DELTA, FILL_CU = 0.32, 2.0, 0.75
    thr = FILL_CU * cu
    best_splits, best_cost = 1, None
    for splits in range(1, 17):
        m_it = math.ceil(math.ceil(avg_main / splits) / block_k) if avg_main > 0 else 0
        e_it = (
            math.ceil(math.ceil(avg_extra / splits) / block_k) if avg_extra > 0 else 0
        )
        waves = (base * splits + cu - 1) // cu
        fill = max(0.0, 1.0 - base * splits / thr) / splits
        cost = waves * (m_it + e_it) + GAMMA * splits + DELTA * fill
        if best_cost is None or cost < best_cost - 1e-9:
            best_splits, best_cost = splits, cost
    return best_splits


def _pa_decode_sparse_gfx950_gluon(
    q,
    cache,
    cache_scales,
    indices,
    indptr,
    scale,
    attn_sink,
    extra_cache=None,
    extra_indices=None,
    extra_indptr=None,
    kv_splits=None,
    skip_reduce=False,
    has_invalid=False,
    fp8_fnuz=False,
):
    """Merged gfx950 gluon DSv4 sparse-MLA decode driver. Format from ``cache.ndim``:
    3D [nb, block, ...] -> packed fp8_ds_mla (uint8: 448 NoPE fp8 e4m3 OCP +
                           embedded UE8M0 per-64 scale + 64 RoPE bf16) or a bf16
                           block cache; pass ``extra_*`` for the SWA+top-k two-loop,
                           else a single segment.
    2D [pages, D]       -> uniform pool: fp8 (uint8) + ``cache_scales``
                           [pages, D//64] fp32, or bf16 (``cache_scales`` None).
    """
    assert q.ndim == 3, f"expected q=[b,h,d], got {q.shape}"
    assert DEVICE_ARCH == "gfx950", "gluon DSv4 decode kernel is gfx950-only"

    # Tuned launch config (gfx950 / MI355), inlined. BLOCK_M = heads per MFMA M-tile;
    # BLOCK_K = KV tile; num_warps = BLOCK_K // 16 (warps tile the dot-N, MFMA N=16).
    BLOCK_M, BLOCK_K, MFMA_K, waves_per_eu = 16, 64, 16, 0
    num_warps = BLOCK_K // 16
    NOPE_DIM, ROPE_DIM = 448, 64
    MAX_BYTES = 2**31 - 1

    num_queries, num_heads, head_dim = q.shape
    indices = _as_int32_contiguous_1d(indices)
    indptr = _as_int32_contiguous_1d(indptr)
    has_sink = attn_sink is not None
    attn_sink = (
        attn_sink.contiguous().to(torch.float32)
        if has_sink
        else torch.empty(1, device=q.device, dtype=torch.float32)
    )

    if cache.ndim == 2:
        # uniform pool: one fp8 gather over the whole head + separate fp32 scales,
        # or bf16. page_size=1 -> block_idx=slot, pos=0; scales ride the bf16 ptr.
        UNIFORM = True
        main_is_fp8 = cache.dtype == torch.uint8
        if main_is_fp8:
            assert cache_scales is not None and cache_scales.dtype == torch.float32
            main_bf16 = cache_scales.contiguous()
        else:
            main_bf16 = cache
        # if HAS_EXTRA=False, reuse main tensors as unread placeholders.
        extra_cache, extra_bf16, extra_indices, extra_indptr = (
            cache,
            main_bf16,
            indices,
            indptr,
        )
        extra_is_fp8 = main_is_fp8
        has_extra = False
        main_block, extra_block = 1, 1
        nope_dim = head_dim
        main_num_rows = extra_num_rows = cache.shape[0]
        cache_bytes = max_addressable_bytes(cache)
        avg_main = indices.numel() / max(1, num_queries)  # one segment; no extra
        avg_extra = 0.0
    else:
        # packed fp8_ds_mla [nb, block, 584] (embedded scale) or bf16 block cache.
        UNIFORM = False
        main_is_fp8 = cache.dtype == torch.uint8
        main_bf16 = cache.view(torch.bfloat16) if main_is_fp8 else cache
        has_extra = (
            extra_cache is not None
            and extra_indices is not None
            and extra_indptr is not None
        )
        if has_extra:
            extra_indices = _as_int32_contiguous_1d(extra_indices)
            extra_indptr = _as_int32_contiguous_1d(extra_indptr)
        else:
            extra_cache, extra_indices, extra_indptr = cache, indices, indptr
        extra_is_fp8 = extra_cache.dtype == torch.uint8
        extra_bf16 = extra_cache.view(torch.bfloat16) if extra_is_fp8 else extra_cache
        main_block, extra_block = cache.shape[1], extra_cache.shape[1]
        nope_dim = NOPE_DIM
        main_num_rows = cache.shape[0] * cache.shape[1]
        extra_num_rows = extra_cache.shape[0] * extra_cache.shape[1]
        cache_bytes = max(
            max_addressable_bytes(cache), max_addressable_bytes(extra_cache)
        )
        avg_main = indices.numel() / max(1, num_queries)
        avg_extra = extra_indices.numel() / max(1, num_queries) if has_extra else 0.0

    use_buffer_load = cache_bytes < MAX_BYTES
    HEAD_ALIGNED = num_heads % BLOCK_M == 0
    heads_blocks = (num_heads + BLOCK_M - 1) // BLOCK_M
    out = torch.empty_like(q, dtype=torch.bfloat16)

    if kv_splits is not None:
        num_splits = max(1, int(kv_splits))
    else:
        num_splits = _decode_num_splits(
            num_queries, heads_blocks, avg_main, avg_extra, BLOCK_K
        )

    if num_splits > 1:
        part_m = torch.empty(
            (num_queries, num_splits, num_heads), dtype=torch.float32, device=q.device
        )
        part_l = torch.empty_like(part_m)
        part_acc = torch.empty(
            (num_queries, num_splits, num_heads, head_dim),
            dtype=torch.float32,
            device=q.device,
        )
        pm_stride0, pm_stride_s = part_m.stride(0), part_m.stride(1)
        pa_stride0, pa_stride_s, pa_stride_h = (
            part_acc.stride(0),
            part_acc.stride(1),
            part_acc.stride(2),
        )
    else:
        part_m = part_l = part_acc = out  # unused placeholders (never dereferenced)
        pm_stride0 = pm_stride_s = pa_stride0 = pa_stride_s = pa_stride_h = 0

    grid = (num_queries, num_splits, heads_blocks)
    _pa_decode_sparse_gfx950[grid](
        q,
        cache,
        main_bf16,
        indices,
        indptr,
        extra_cache,
        extra_bf16,
        extra_indices,
        extra_indptr,
        attn_sink,
        out,
        part_m,
        part_l,
        part_acc,
        scale,
        q.stride(0),
        q.stride(1),
        out.stride(0),
        out.stride(1),
        cache.stride(0),
        extra_cache.stride(0),
        main_num_rows,
        extra_num_rows,
        pm_stride0,
        pm_stride_s,
        pa_stride0,
        pa_stride_s,
        pa_stride_h,
        num_heads,
        HAS_EXTRA=has_extra,
        HAS_SINK=has_sink,
        MAIN_IS_FP8=main_is_fp8,
        EXTRA_IS_FP8=extra_is_fp8,
        MAIN_BLOCK_SIZE=main_block,
        EXTRA_BLOCK_SIZE=extra_block,
        NOPE_DIM=nope_dim,
        ROPE_DIM=ROPE_DIM,
        HEAD_SIZE=head_dim,
        BLOCK_M=BLOCK_M,
        BLOCK_K=BLOCK_K,
        NUM_SPLITS=num_splits,
        HEAD_ALIGNED=HEAD_ALIGNED,
        MFMA_K=MFMA_K,
        UNIFORM=UNIFORM,
        USE_BUFFER_LOAD=use_buffer_load,
        HAS_INVALID=has_invalid,
        FP8_FNUZ=fp8_fnuz,
        num_warps=num_warps,
        waves_per_eu=waves_per_eu,
    )

    if num_splits == 1:
        return out
    if skip_reduce:
        return part_acc, part_m, part_l

    rgrid = (num_queries, heads_blocks)
    _pa_decode_sparse_reduce_gfx950[rgrid](
        part_m,
        part_l,
        part_acc,
        attn_sink,
        out,
        out.stride(0),
        out.stride(1),
        pm_stride0,
        pm_stride_s,
        pa_stride0,
        pa_stride_s,
        pa_stride_h,
        num_heads,
        HAS_SINK=has_sink,
        HEAD_SIZE=head_dim,
        BLOCK_M=BLOCK_M,
        NUM_SPLITS=num_splits,
        HEAD_ALIGNED=HEAD_ALIGNED,
        num_warps=4,
    )
    return out


# ---------------------------------------------------------------------------
# Block-sparse paged decode over a SHUFFLE K/V cache pair (MiniMax-M3).
# ---------------------------------------------------------------------------

_FP8_CACHE_DTYPES = (
    torch.float8_e4m3fn,
    torch.float8_e4m3fnuz,
    torch.float8_e5m2,
)

# Launch config, tuned on gfx1250 (MI455) for MiniMax-M3 decode (q [N, 16, 128],
# page_size 16) by sweeping N in 1..1024 x ctx in 1k..4k. The two cache dtypes
# hit different limits, so the tile is keyed on the cache element size:
#
#   bf16 (2B): BLOCK_K = 8 pages = 128 keys, ONE warp. Wide enough to amortise
#     the block-table gather and the loop, and at a single warp Triton keeps the
#     K/V tiles in registers (4KB LDS). Two warps make it stage them through LDS
#     instead -- 64KB per CTA, which caps CU residency on a bandwidth-bound
#     kernel. Worth ~1.5x, and it wins at every batch size.
#   fp8 (1B): the fp8 -> bf16 widen in front of the WMMA forces that LDS staging
#     whatever the warp count, so the trade flips and splits by batch. Small
#     batch has CUs to spare, so the wide 8-page tile over two warps amortises
#     best (2.0x at N=4); past ~32 CTAs the LDS traffic is the constraint and
#     the 4-page single-warp tile wins instead (up to 1.5x over the wide one).
#
# waves_per_eu stays 1: at (num_warps=2, waves_per_eu=2) the fp8 path's LDS
# staging races and returns non-deterministic output -- a ~15-ULP drift that
# looks like ordinary fp8 error, not garbage. Do not raise it without
# re-running test_pa_decode_sparse_shuffled_deterministic.
_SHUFFLED_BF16_CFG = (8, 1, 1)  # (pages_per_tile, num_warps, waves_per_eu)
_SHUFFLED_FP8_SMALL_BATCH_CFG = (8, 2, 1)
_SHUFFLED_FP8_CFG = (4, 1, 1)
_SHUFFLED_FP8_SMALL_BATCH_CTAS = 16
_SHUFFLED_NUM_STAGES = 2
# Split-K target: total CTAs to aim for before capping at the tile count.
# 1024 == 4 CTAs/CU on MI455. Splitting further keeps paying for the partial
# buffers and the reduce while the main kernel is already CU-saturated.
_SHUFFLED_CTA_TARGET = 1024
# Reduce launch config: one warp, a few heads per CTA. BLOCK_H=1 with 4 warps --
# what the unified-pool driver above uses -- costs 10x here (72us vs 10us at
# N=512, KV_SPLITS=2): 4 warps over a [KV_SPLITS, 1, D] tile spread a tiny
# reduction across warps and pay for the cross-warp combine. One warp over 4
# heads keeps it in-wave.
_SHUFFLED_REDUCE_BLOCK_H = 4
_SHUFFLED_REDUCE_NUM_WARPS = 1
_SHUFFLED_REDUCE_WAVES_PER_EU = 4
# ...but the CTA holds the whole [KV_SPLITS, BLOCK_H, D] partial tile in
# registers, so the head block has to shrink as the split count grows or the
# reduce spills (90us+ at KV_SPLITS=32, BLOCK_H=4). Cap the tile at 32 fp32 per
# lane per D-element.
_SHUFFLED_REDUCE_MAX_TILE = 32


def pa_decode_sparse_shuffled(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    context_lens: torch.Tensor,
    softmax_scale: float,
    out: torch.Tensor | None = None,
    k_scale: torch.Tensor | None = None,
    v_scale: torch.Tensor | None = None,
    attn_sink: torch.Tensor | None = None,
    kv_splits: int | None = None,
    block_h: int | None = None,
    pages_per_tile: int | None = None,
) -> torch.Tensor:
    """Block-sparse paged decode attention over a SHUFFLE (asm-layout) KV cache.

    The sparse counterpart of ``pa_decode_sparse`` for models whose selected
    keys arrive as a *compacted page list* rather than a flat token-index list:
    the caller has already turned the top-k block selection into a dense
    ``block_table`` row plus an exact ``context_lens`` entry per query token, so
    the first ``context_lens[n]`` keys reachable through row ``n`` are exactly
    the ones to attend to. That is MiniMax-M3 block-sparse decode, and it is
    also the plain paged-decode contract, so a dense block table works too.

    Args:
        q: ``[N, H, D]`` decode queries, bf16/fp16. One row per (token, kv-head)
            when the caller collapses GQA that way; ``H`` is then the query group.
        k_cache: ``[P, 1, D//X, PAGE, X]`` SHUFFLE K cache, same dtype as ``q``
            (or fp8 with ``k_scale``). ``X`` is the layout's packing factor
            (16 // itemsize: 8 for bf16, 16 for fp8).
        v_cache: ``[P, 1, PAGE//X, D, X]`` SHUFFLE V cache, matching ``k_cache``.
        block_table: ``[N, MAX_PAGES]`` int32 page ids into ``k_cache``/``v_cache``.
            Rows are read densely from slot 0; entries past ``context_lens`` are
            never dereferenced, so padding slots may hold any in-bounds page id.
        context_lens: ``[N]`` int32 valid key count per query token.
        softmax_scale: scalar softmax scale.
        out: optional ``[N, H, D]`` destination, same dtype as ``q``.
        k_scale/v_scale: ``[P, 1, PAGE]`` fp32 *per-token* dequant scales for an
            fp8 cache (both or neither). K's fold onto the score columns, V's
            onto the softmax probabilities.
        attn_sink: optional ``[H]`` fp32 per-head softmax-denominator bias.
        kv_splits: override the split-K grid axis. Default fills ~2048 CTAs
            without splitting past the tile count, rounded up to a power of 2.
        block_h: override ``BLOCK_H``. Default ``next_pow2(H)`` at the AMD MFMA
            minimum of 16.
        pages_per_tile: override the KV tile width in pages (``BLOCK_K ==
            pages_per_tile * PAGE``).

    Returns:
        ``[N, H, D]`` attention output, same dtype as ``q``.
    """
    if not q.is_cuda:
        raise RuntimeError("pa_decode_sparse_shuffled requires CUDA/HIP tensors")
    if q.dtype not in (torch.bfloat16, torch.float16):
        raise RuntimeError(
            f"pa_decode_sparse_shuffled expects fp16/bf16 q, got {q.dtype}"
        )
    if q.ndim != 3:
        raise RuntimeError(f"expected q=[N, H, D], got {tuple(q.shape)}")
    if k_cache.ndim != 5 or v_cache.ndim != 5:
        raise RuntimeError(
            "expected 5D SHUFFLE caches K[P,1,D//X,PAGE,X] / V[P,1,PAGE//X,D,X], "
            f"got K{tuple(k_cache.shape)} V{tuple(v_cache.shape)}"
        )
    if k_cache.shape[1] != 1 or v_cache.shape[1] != 1:
        raise RuntimeError(
            "pa_decode_sparse_shuffled wants the kv-head axis collapsed into the "
            "page id (pass a [P*Hkv, 1, ...] view); got "
            f"K{tuple(k_cache.shape)}"
        )

    N, H, D = q.shape
    X = k_cache.shape[4]
    page_size = k_cache.shape[3]
    if k_cache.shape[2] * X != D:
        raise RuntimeError(
            f"K cache head_dim {k_cache.shape[2] * X} does not match q head_dim {D}"
        )
    if page_size % X != 0:
        raise RuntimeError(f"page_size {page_size} must be divisible by X={X}")
    if tuple(v_cache.shape) != (v_cache.shape[0], 1, page_size // X, D, X):
        raise RuntimeError(
            f"V cache {tuple(v_cache.shape)} does not match K cache "
            f"{tuple(k_cache.shape)} (expected [P,1,{page_size // X},{D},{X}])"
        )
    # The X axis is what makes the gather vectorise; a non-unit stride there
    # would silently drop the kernel back to per-element loads.
    if k_cache.stride(4) != 1 or v_cache.stride(4) != 1:
        raise RuntimeError("SHUFFLE caches must be contiguous in their last axis")
    if block_table.dtype != torch.int32 or block_table.ndim != 2:
        raise RuntimeError(
            f"block_table must be 2D int32, got {block_table.ndim}D "
            f"{block_table.dtype}"
        )
    if context_lens.dtype != torch.int32:
        raise RuntimeError(f"context_lens must be int32, got {context_lens.dtype}")
    if block_table.shape[0] != N or context_lens.shape[0] != N:
        raise RuntimeError(
            f"block_table/context_lens must have {N} rows, got "
            f"{block_table.shape[0]}/{context_lens.shape[0]}"
        )

    quant_kv = k_scale is not None or v_scale is not None
    if quant_kv:
        if k_scale is None or v_scale is None:
            raise RuntimeError("k_scale and v_scale must be given together")
        if k_scale.dtype != torch.float32 or v_scale.dtype != torch.float32:
            raise RuntimeError(
                f"per-token KV scales must be fp32, got {k_scale.dtype}/"
                f"{v_scale.dtype}"
            )
        if k_scale.shape[-1] != page_size or v_scale.shape[-1] != page_size:
            raise RuntimeError(
                f"per-token KV scales must be [..., {page_size}], got "
                f"{tuple(k_scale.shape)}/{tuple(v_scale.shape)}"
            )
        if k_scale.stride(-1) != 1 or v_scale.stride(-1) != 1:
            raise RuntimeError("per-token KV scales must be page-contiguous")
        if k_cache.dtype not in _FP8_CACHE_DTYPES or v_cache.dtype != k_cache.dtype:
            raise RuntimeError(
                "per-token KV scales imply an fp8 cache; got "
                f"K={k_cache.dtype}, V={v_cache.dtype}"
            )
        s_stride_p = k_scale.stride(0)
        if v_scale.stride(0) != s_stride_p:
            raise RuntimeError("k_scale and v_scale must share a page stride")
    elif k_cache.dtype != q.dtype:
        raise RuntimeError(
            f"kv cache dtype mismatch: kv={k_cache.dtype}, q={q.dtype} "
            "(pass k_scale/v_scale for an fp8 cache)"
        )

    _LOGGER.info(
        f"PA_DECODE_SPARSE_SHUFFLED N={N} H={H} D={D} page={page_size} X={X} "
        f"max_pages={block_table.shape[1]} quant={quant_kv}"
    )

    if out is None:
        out = torch.empty_like(q)

    block_h = triton.next_power_of_2(block_h if block_h is not None else H)
    block_h = max(block_h, 16)  # AMD MFMA min tile
    n_head_blocks = triton.cdiv(H, block_h)
    h_padded = n_head_blocks * block_h
    block_d = triton.next_power_of_2(D)
    if block_d != D:
        raise RuntimeError(f"head_dim {D} must be a power of two")

    if k_cache.element_size() == 2:
        default_ppt, num_warps, waves_per_eu = _SHUFFLED_BF16_CFG
    elif N * n_head_blocks <= _SHUFFLED_FP8_SMALL_BATCH_CTAS:
        default_ppt, num_warps, waves_per_eu = _SHUFFLED_FP8_SMALL_BATCH_CFG
    else:
        default_ppt, num_warps, waves_per_eu = _SHUFFLED_FP8_CFG
    ppt = pages_per_tile if pages_per_tile is not None else default_ppt
    ppt = max(1, triton.next_power_of_2(ppt))
    block_k = ppt * page_size

    max_pages = block_table.shape[1]
    if kv_splits is None:
        # Split only as far as it takes to fill the machine: the split kernel is
        # already CU-bound once N * head_blocks reaches the target, and every
        # extra split adds an [N, splits, H, D] fp32 round trip. ``context_lens``
        # lives on the device (reading it would cost a per-layer D2H sync), so
        # the tile count is bounded by the block table width instead. Splits past
        # a token's real tile count early-return and the reduce masks their
        # slots, so over-splitting a short context is wasteful but correct.
        max_kv_splits = max(1, triton.cdiv(max_pages, ppt))
        kv_splits = max(1, _SHUFFLED_CTA_TARGET // max(1, N * n_head_blocks))
        kv_splits = triton.next_power_of_2(min(max_kv_splits, kv_splits))
    kv_splits = max(1, int(kv_splits))

    if kv_splits == 1:
        m_partial = l_partial = acc_partial = out  # unused inside the kernel
        mp_strides = (0, 0, 0)
        lp_strides = (0, 0, 0)
        ap_strides = (0, 0, 0, 0)
    else:
        m_partial = torch.empty(
            (N, kv_splits, h_padded), dtype=torch.float32, device=q.device
        )
        l_partial = torch.empty_like(m_partial)
        acc_partial = torch.empty(
            (N, kv_splits, h_padded, D), dtype=torch.float32, device=q.device
        )
        mp_strides = m_partial.stride()
        lp_strides = l_partial.stride()
        ap_strides = acc_partial.stride()

    has_sink = attn_sink is not None
    if has_sink:
        attn_sink = attn_sink.to(torch.float32)
    else:
        attn_sink = q.new_empty(1, dtype=torch.float32)
    if quant_kv:
        k_scale_arg, v_scale_arg = k_scale, v_scale
    else:
        k_scale_arg = v_scale_arg = q.new_empty(1, dtype=torch.float32)
        s_stride_p = 1

    triton_pa_decode_sparse_shuffled[(N, n_head_blocks, kv_splits)](
        q,
        k_cache,
        v_cache,
        k_scale_arg,
        v_scale_arg,
        block_table,
        context_lens,
        m_partial,
        l_partial,
        acc_partial,
        attn_sink,
        out,
        max_pages,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k_cache.stride(0),
        k_cache.stride(2),
        k_cache.stride(3),
        v_cache.stride(0),
        v_cache.stride(2),
        v_cache.stride(3),
        s_stride_p,
        block_table.stride(0),
        mp_strides[0],
        mp_strides[1],
        mp_strides[2],
        lp_strides[0],
        lp_strides[1],
        lp_strides[2],
        ap_strides[0],
        ap_strides[1],
        ap_strides[2],
        ap_strides[3],
        out.stride(0),
        out.stride(1),
        out.stride(2),
        H,
        D,
        kv_splits,
        float(softmax_scale),
        BLOCK_H=block_h,
        BLOCK_D=block_d,
        PAGE_SIZE=page_size,
        PAGES_PER_TILE=ppt,
        X=X,
        HAS_SINK=has_sink,
        QUANT_KV=quant_kv,
        USE_EXP2=True,
        num_warps=num_warps,
        num_stages=_SHUFFLED_NUM_STAGES,
        waves_per_eu=waves_per_eu,
    )

    if kv_splits == 1:
        return out

    block_h_reduce = min(
        _SHUFFLED_REDUCE_BLOCK_H,
        triton.next_power_of_2(H),
        max(1, _SHUFFLED_REDUCE_MAX_TILE // kv_splits),
    )
    triton_pa_decode_sparse_reduce[(N, triton.cdiv(H, block_h_reduce))](
        m_partial,
        l_partial,
        acc_partial,
        attn_sink,
        context_lens,
        out,
        m_partial.stride(0),
        m_partial.stride(1),
        m_partial.stride(2),
        l_partial.stride(0),
        l_partial.stride(1),
        l_partial.stride(2),
        acc_partial.stride(0),
        acc_partial.stride(1),
        acc_partial.stride(2),
        acc_partial.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        H,
        D,
        kv_splits,
        BLOCK_H=block_h_reduce,
        BLOCK_D=block_d,
        BLOCK_K=block_k,
        USE_EXP2=True,
        USE_CTX_LENS=True,
        HAS_SINK=has_sink,
        num_warps=_SHUFFLED_REDUCE_NUM_WARPS,
        waves_per_eu=_SHUFFLED_REDUCE_WAVES_PER_EU,
    )
    return out
