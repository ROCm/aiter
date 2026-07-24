# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Sparse paged-decode attention over a unified KV pool with per-token paged
indices. See ``_triton_kernels/attention/pa_decode_sparse.py`` for the
kernels' caller contract.

This module exposes ``pa_decode_sparse`` — a 3D split-K + widened-BLOCK_H
+ pipelined-K-loop variant suitable for sparse decode (e.g. V4 top-k gather)
where each token's K range is an unordered subset of a unified KV pool.

On gfx950 (CDNA4) DeepSeek-V4 sparse-MLA decode has a dedicated gluon
implementation (bottom of this module): ``pa_decode_sparse`` routes all formats
to the merged ``_pa_decode_sparse_gfx950_gluon`` driver -- packed fp8_ds_mla /
bf16 block cache (3D; optional SWA+top-k two-loop via ``extra_*``) and the
uniform fp8 / bf16 pool (2D). The vLLM DSv4 backend's packed entry is kept as a
thin shim (``_rocm_sparse_attn_decode_ragged_triton``).
"""

from typing import Optional

import torch
import triton

from aiter.ops.triton._triton_kernels.attention.pa_decode_sparse import (
    _pa_decode_sparse as triton_pa_decode_sparse,
    _pa_decode_sparse_reduce as triton_pa_decode_sparse_reduce,
)
from aiter.ops.triton.utils._triton import arch_info
from aiter.ops.triton.utils.logger import AiterTritonLogger
from aiter.ops.triton._gluon_kernels.gfx1250.attention.pa_decode_sparse import (
    _pa_decode_sparse as gluon_pa_decode_sparse,
    _pa_decode_sparse_reduce as gluon_pa_decode_sparse_reduce,
)

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
    kv_scales: Optional[torch.Tensor] = None,
    block_h: Optional[int] = None,
    kv_splits: Optional[int] = None,
    has_invalid: Optional[bool] = True,
    skip_reduce: Optional[bool] = False,
    USE_EXP2: Optional[bool] = None,
    *,
    extra_cache: Optional[torch.Tensor] = None,
    extra_indices: Optional[torch.Tensor] = None,
    extra_indptr: Optional[torch.Tensor] = None,
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
    honored; ``block_h`` and fnuz-fp8 fall through to the triton path.

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
    # kv_splits and skip_reduce are honored here; block_h and fnuz-fp8 fall through to
    # the triton path below.
    if DEVICE_ARCH == "gfx950" and block_h is None:
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
    # kv_indices = kv_indices.to(torch.int32).contiguous()
    # kv_indptr = kv_indptr.to(torch.int32).contiguous()

    if block_h is None:
        # Default: one CTA per token (kills the H/BLOCK_H KV duplication).
        # If H is too large to fit a single tile, halve until it does.
        block_h = triton.next_power_of_2(min(H, 16))
    else:
        block_h = triton.next_power_of_2(block_h)
    block_h = max(block_h, 16)  # AMD MFMA min tile

    n_head_blocks = triton.cdiv(H, block_h)
    h_padded = n_head_blocks * block_h
    block_d = triton.next_power_of_2(D)
    assert block_d == D

    use_gluon = DEVICE_ARCH == "gfx1250"

    # gfx1250 stages slots through LDS via TDM async_load, which hides the
    # larger per-tile KV gather latency -> BLOCK_K=32 is fastest there. Other
    # arches use the synchronous slot path, where 32 exposes memory latency.
    if use_gluon:
        block_k = 16
        attn_num_warps = 1
        max_num_wg = 1024
    else:
        block_k = 16 if D >= 256 else 32
        attn_num_warps = 4
        max_num_wg = 256
    num_stages = 2
    waves_per_eu = 1
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


def _decode_cu_count() -> int:
    try:
        return torch.cuda.get_device_properties(0).multi_processor_count
    except Exception:
        return 256


def _decode_num_splits(num_queries, heads_blocks, avg_len=0.0, block_k=64):
    base = max(1, num_queries * heads_blocks)
    cu = max(1, _decode_cu_count())
    mu = 0.04
    best_splits, best_cost = 1, None
    for splits in range(1, 17):
        waves = (base * splits + cu - 1) // cu
        cost = waves * (1.0 / splits + mu)
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
    ``kv_splits`` overrides the split-K count. ``skip_reduce`` (only takes effect when
    the chosen split count > 1) returns the pre-reduce partials
    ``(part_acc, part_m, part_l)`` -- shapes ``([N, S, H, D], [N, S, H], [N, S, H])``
    fp32; ``m`` is the row-max in the base-2 exponent domain (row-max * softmax_scale
    * log2e) and ``l``/``acc`` are per-split un-normalized -- same convention as the
    triton skip_reduce partials -- instead of the final ``[N, H, D]`` output.
    ``has_invalid`` (default False): when True, -1 sentinels anywhere in a token's
    index range are clamped in-bounds for the gather and masked out of the softmax.
    ``fp8_fnuz`` (uniform-pool fp8 only): fp8 e4m3 flavor -- False = OCP (bias 7),
    True = fnuz (bias 8); selects the in-kernel dequant. Packed fp8_ds_mla is OCP.
    """
    assert q.ndim == 3, f"expected q=[b,h,d], got {q.shape}"
    assert DEVICE_ARCH == "gfx950", "gluon DSv4 decode kernel is gfx950-only"
    from aiter.ops.triton._gluon_kernels.gfx950.attention.pa_decode_sparse import (
        _pa_decode_sparse,
        _pa_decode_sparse_reduce,
    )

    # Tuned launch config (gfx950 / MI355), inlined. BLOCK_M = heads per MFMA M-tile;
    # BLOCK_K = KV tile; num_warps = BLOCK_K // 16 (warps tile the dot-N, MFMA N=16).
    BLOCK_M, BLOCK_K, MFMA_K, waves_per_eu = 16, 64, 16, 0
    num_warps = BLOCK_K // 16
    NOPE_DIM, ROPE_DIM = 448, 64
    MAX_BYTES = 2**31 - 1  # buffer_load 32-bit offset cap; larger -> gl.load int64

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
        cache_bytes = cache.nelement() * cache.element_size()
        avg_len = indices.numel() / max(1, num_queries)
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
            cache.nelement() * cache.element_size(),
            extra_cache.nelement() * extra_cache.element_size(),
        )
        avg_len = (indices.numel() + extra_indices.numel()) / max(1, num_queries)

    use_buffer_load = cache_bytes <= MAX_BYTES
    HEAD_ALIGNED = num_heads % BLOCK_M == 0
    heads_blocks = (num_heads + BLOCK_M - 1) // BLOCK_M
    out = torch.empty_like(q, dtype=torch.bfloat16)

    if kv_splits is not None:
        num_splits = max(1, int(kv_splits))
    else:
        num_splits = _decode_num_splits(num_queries, heads_blocks, avg_len, BLOCK_K)
        # Cap at the tile count: extra splits just fragment short token lists into
        # sub-tile masked chunks (matters for tiny top-k / small merged ctx).
        num_splits = min(num_splits, max(1, int((avg_len + BLOCK_K - 1) // BLOCK_K)))

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
    _pa_decode_sparse[grid](
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
    _pa_decode_sparse_reduce[rgrid](
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
