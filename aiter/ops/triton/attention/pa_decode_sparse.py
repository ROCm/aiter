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
from aiter.ops.triton.utils._triton import arch_info
from aiter.ops.triton.utils.common_utils import max_addressable_bytes
from aiter.ops.triton.utils.device_info import get_num_sms
from aiter.ops.triton.utils.logger import AiterTritonLogger

DEVICE_ARCH = arch_info.get_arch()

_LOGGER = AiterTritonLogger()


_FP8_GROUP_SIZE = 64
_FP8_DTYPE = torch.float8_e4m3fnuz


def _check_out(out, q, dtype):
    """Caller-supplied output buffer, or a fresh one. Writing the caller's buffer
    directly saves a full [T, H, D] device copy per call."""
    if out is None:
        return torch.empty_like(q, dtype=dtype)
    assert out.shape == q.shape, f"out shape {tuple(out.shape)} != q {tuple(q.shape)}"
    assert out.dtype == dtype, f"out dtype {out.dtype} != {dtype}"
    assert out.device == q.device
    return out


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
    out: torch.Tensor | None = None,
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
        out: optional ``[N, H, D]`` destination. Supplied -> written in place and
            returned, which saves the caller a full-size device copy.
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
                out=out,
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

    out = _check_out(out, q, q.dtype)
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
    _rw = _os.environ.get("AITER_PA_DECODE_REDUCE_WARPS")
    if _rw and use_gluon:
        reduce_num_warps = int(_rw)
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

    # Collapse to the FEWEST splits that keeps both the wave count and the
    # per-split BLOCK_K iteration count. The cost model above treats extra
    # splits as nearly free while ``waves`` stays 1 (GAMMA is small and it has no
    # reduce term), so at small per-query work it over-splits: at C=64/H=16 with
    # main=128, extra=8 it picked 3, where 2 does the same 2 iterations in the
    # same single wave but launches 1/3 fewer CTAs and reduces over 2 partials
    # instead of 3 (measured 16.2 -> 14.4 us on gfx950/MI355X).
    #
    # Ported from vLLM's _decode_gfx950_num_splits (PR #52212), which fixed the
    # same over-splitting in the in-tree triton decode.
    def _iters(s):
        m = math.ceil(math.ceil(avg_main / s) / block_k) if avg_main > 0 else 0
        e = math.ceil(math.ceil(avg_extra / s) / block_k) if avg_extra > 0 else 0
        return m + e

    def _waves(s):
        return (base * s + cu - 1) // cu

    if best_splits > 1:
        target_waves, target_iters = _waves(best_splits), _iters(best_splits)
        for splits in range(1, best_splits):
            if _waves(splits) == target_waves and _iters(splits) == target_iters:
                return splits
    return best_splits


def _decode_num_splits_occ(num_queries, heads_blocks, avg_main, avg_extra, block_k):
    """Split-K count for the gfx950 gluon kernel: fill the machine, but never
    split a segment finer than one BLOCK_K tile.

    Two facts set this. (a) The kernel is 256 unified VGPRs and 68,608 B of LDS,
    so a CU holds two workgroups -- the launch wants ~2*CU programs before extra
    splits stop paying. (b) Splitting a segment past its tile count does not divide
    the work, it *multiplies* it: every split then owns a partial tile, and a
    partial tile costs a full masked gather for a fraction of the tokens. So the
    split count is capped by the larger segment's tile count, and the main segment
    separately stops at its own (see MAIN_SPLITS in the kernel).

    Measured at C=64 H=16 BLOCK_K=64 (attn = kernel + reduce, us), this rule picks
    the sweep optimum at extra in {8, 64, 272, 512, 1024, 2048} and lands within
    1.4% at extra=128:

        extra      8    64   128   272   512  1024  2048
        picked S   2     2     2     5     8     8     8
        best S     2     2     4     5     8     8     8
        old model  2     2     2     3     4     4     4
        old  us  14.0  14.6  17.9  21.3  22.0  29.8  45.6
        new  us  14.0  14.4  17.4  19.5  20.9  26.9  38.1
    """
    num_sms = get_num_sms()
    base_wg = max(1, num_queries * heads_blocks)
    cta_cap = max(1, (2 * num_sms) // base_wg)
    main_tiles = max(1, math.ceil(avg_main / block_k)) if avg_main > 0 else 0
    extra_tiles = max(1, math.ceil(avg_extra / block_k)) if avg_extra > 0 else 0
    tiles = max(1, main_tiles, extra_tiles)
    if base_wg >= num_sms:
        # Already at least one workgroup per CU without splitting, so a split buys
        # only the second occupancy slot -- worth an extra round trip of the
        # [queries, heads, D] f32 partials plus a reduce launch only when each
        # split still owns a real run of tiles. At C=256 extra=8 (2 tiles) forcing
        # a split costs 31%; at extra=2048 (32 tiles) it saves 19%.
        return max(1, min(cta_cap, tiles // 4))
    # Deliberately NOT refined toward split counts that divide `tiles` evenly.
    # Minimizing the executed tile count S*ceil(tiles/S) is only right for a
    # uniform batch -- it is worth ~5% on the 1-loop seq=1152 shape -- but `tiles`
    # here is a batch AVERAGE, and wall clock is set by the longest query. On a
    # 16x-ragged batch that refinement picks 5 splits where 8 is 22% faster.
    return max(1, min(cta_cap, tiles))


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
    out=None,
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
    # AITER_PA_DECODE_BLOCK_K: experiment override for the KV tile width.
    # num_warps stays BLOCK_K//16 (warps tile the dot-N, MFMA N=16).
    import os as _os
    _bk = _os.environ.get("AITER_PA_DECODE_BLOCK_K")
    if _bk:
        BLOCK_K = int(_bk)
    # AITER_PA_DECODE_BLOCK_M: heads per MFMA M-tile. 16 is both the MFMA M and the
    # DSv4 head count; larger pads and quadruples the accumulator (see RESULTS.md).
    _bm = _os.environ.get("AITER_PA_DECODE_BLOCK_M")
    if _bm:
        BLOCK_M = int(_bm)
    # AITER_PA_DECODE_MFMA_K: 32 selects CDNA4's double-rate v_mfma_f32_16x16x32_bf16
    # (16 does 16x16x16, i.e. half the K per instruction).
    _mk = _os.environ.get("AITER_PA_DECODE_MFMA_K")
    if _mk:
        MFMA_K = int(_mk)
    num_warps = max(1, BLOCK_K // 16)
    # AITER_PA_DECODE_TW1: threads the NoPE gather spends on the head dim (16 B
    # each). 32 issues a whole 512 B token row per instruction instead of 8 x 128 B
    # quarters, which is what a scattered top-k gather wants; the cost is a longer
    # per-lane slot vector. Measured best at 32 for every extra >= 128 (-10% at
    # extra=128) and within noise at extra=8.
    gather_tw1 = int(_os.environ.get("AITER_PA_DECODE_TW1", "32"))
    # AITER_PA_DECODE_LDS_PAD: bf16 elements of padding after each kv_smem row.
    lds_pad = int(_os.environ.get("AITER_PA_DECODE_LDS_PAD", "8"))
    # AITER_PA_DECODE_NOPE_CHUNK: width of one fp8 dequant piece (0/unset = one shot).
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

    # Largest power-of-2 (<=16) dividing BOTH page strides. Lets the kernel assert
    # 16B-aligned row bases so a contiguous row gather can vectorize to dwordx4.
    _s0 = int(cache.stride(0)); _s1 = int(extra_cache.stride(0))
    cs0_align = 1
    for _a in (16, 8, 4, 2):
        if _s0 % _a == 0 and _s1 % _a == 0:
            cs0_align = _a
            break
    # AITER_PA_DECODE_CS0_ALIGN: override for re-measuring the hint (1 = off). As of
    # the row/column gather split the hint is codegen-neutral -- the column offsets are
    # compile-time constants, so vectorization no longer depends on knowing cs0's
    # divisibility, and the emitted load mix is identical with it on and off.
    _ca = _os.environ.get("AITER_PA_DECODE_CS0_ALIGN")
    if _ca:
        cs0_align = int(_ca)
    # Gate each cache on its OWN span: one oversized cache must not disable the
    # fast path for the other. (cache_bytes, the max of the two, is kept for the
    # arch/format asserts below.)
    main_use_buffer_load = max_addressable_bytes(cache) < MAX_BYTES
    extra_use_buffer_load = max_addressable_bytes(extra_cache) < MAX_BYTES
    # The index lists are one int32 per gathered token, orders of magnitude under
    # the 2 GB buffer offset limit even for a full batch, so they keep the fast path
    # regardless of how big the KV caches are.
    idx_use_buffer_load = (
        _os.environ.get("AITER_PA_DECODE_IDX_BUF", "1") == "1"
        and max_addressable_bytes(indices) < MAX_BYTES
        and max_addressable_bytes(extra_indices) < MAX_BYTES
    )
    use_buffer_load = main_use_buffer_load and extra_use_buffer_load
    # AITER_PA_DECODE_FORCE_GLOBAL_LOAD emulates a >2GB production cache without
    # allocating one: 1 = both caches, main = main only, extra = extra only.
    _fg = _os.environ.get("AITER_PA_DECODE_FORCE_GLOBAL_LOAD", "")
    if _fg == "1":
        main_use_buffer_load = extra_use_buffer_load = False
    elif _fg == "main":
        main_use_buffer_load = False
    elif _fg == "extra":
        extra_use_buffer_load = False
    use_buffer_load = main_use_buffer_load and extra_use_buffer_load
    HEAD_ALIGNED = num_heads % BLOCK_M == 0
    heads_blocks = (num_heads + BLOCK_M - 1) // BLOCK_M
    out = _check_out(out, q, torch.bfloat16)

    # AITER_PA_DECODE_SPLITS: experiment override for the split-K count.
    _sp = _os.environ.get("AITER_PA_DECODE_SPLITS")

    def _pick_splits(hb):
        if _sp:
            return max(1, int(_sp))
        if kv_splits is not None:
            return max(1, int(kv_splits))
        if _os.environ.get("AITER_PA_DECODE_SPLIT_POLICY", "occ") == "occ":
            return _decode_num_splits_occ(
                num_queries, hb, avg_main, avg_extra, BLOCK_K
            )
        return _decode_num_splits(num_queries, hb, avg_main, avg_extra, BLOCK_K)

    num_splits = _pick_splits(heads_blocks)

    # Short top-k is latency- and coverage-bound rather than throughput-bound: at
    # <= 2 KV tiles per query the whole launch is a few dozen workgroups on a
    # 256-CU part, so the win is MORE programs, not fatter ones. Halving BLOCK_M
    # doubles heads_blocks, which is real parallelism -- unlike a finer split-K,
    # which hands every split a partial tile (1.14-1.36x worse when forced). Both
    # gates matter: measured 2-5% better at every concurrency on top-k 8/64, and
    # 1.06-1.22x WORSE on top-k 272/1024, where the halved MFMA M dominates.
    _tiles = max(
        math.ceil(avg_main / BLOCK_K) if avg_main > 0 else 1,
        math.ceil(avg_extra / BLOCK_K) if avg_extra > 0 else 1,
    )
    if (
        _bm is None
        and BLOCK_M == 16
        and num_heads % 8 == 0
        and _tiles <= 2
        and num_queries * heads_blocks * num_splits <= get_num_sms()
    ):
        _hb8 = (num_heads + 7) // 8
        _ns8 = _pick_splits(_hb8)
        # Doubling heads_blocks can push the launch past one workgroup per CU, and
        # the split policy then answers with fewer splits. At C=128 short top-k that
        # is 2 -> 1: split-K disappears and the shape goes 0.94x -> 1.11x. Take the
        # halving only when it costs neither a split nor the occupancy property.
        if _ns8 >= num_splits and num_queries * _hb8 * _ns8 <= get_num_sms():
            BLOCK_M = 8
            HEAD_ALIGNED = num_heads % BLOCK_M == 0
            heads_blocks = _hb8
            num_splits = _ns8

    if num_splits > 1:
        part_m = torch.empty(
            (num_queries, num_splits, num_heads), dtype=torch.float32, device=q.device
        )
        part_l = torch.empty_like(part_m)
        # Split-K accumulator partials in bf16. They are ~31% of the kernel's HBM
        # traffic and ATT puts 8.5% of all stall cycles on the 68 buffer_store of
        # them, so halving both is worth 6-23% depending on split count. The
        # mantissa bits it costs are far inside the error the fp8 KV format already
        # carries: against an fp64 reference, max|delta|/max|ref| goes 3.517e-2 ->
        # 3.567e-2 at C128A, and the bf16-vs-f32 delta (4-6e-3) is ~6x smaller than
        # the error already present. Not used on the skip_reduce path, which hands
        # part_acc back to the caller and must keep its dtype.
        _pab = (
            _os.environ.get("AITER_PA_DECODE_PART_BF16", "1") == "1"
            and not skip_reduce
        )
        part_acc = torch.empty(
            (num_queries, num_splits, num_heads, head_dim),
            dtype=torch.bfloat16 if _pab else torch.float32,
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

    # 128 is the narrowest piece the gather layout can split to for free (below
    # that a split falls inside a thread's 16-byte run, which Triton rejects), and
    # the split is only trivial when the gather's warps tile dim 0.
    # Row-wide gather chunks rows (4 pieces of BLOCK_K/4) instead of 128-wide
    # column pieces; both leave a quarter of the tile's f32 expansion live.
    # Split the axis that still has per-lane register repeats: columns when the
    # gather leaves >=2 column repeats (TW1=8), rows otherwise.
    col_reps = head_dim // (gather_tw1 * 16)
    chunk_axis = 1 if col_reps >= 4 else 0
    _nc = _os.environ.get("AITER_PA_DECODE_NOPE_CHUNK")
    if chunk_axis == 0:
        nope_chunk = int(_nc) if _nc else max(1, BLOCK_K // 4)
    else:
        nope_chunk = int(_nc) if _nc else min(128, head_dim)

    # waves_per_eu=2 caps the allocator at 256 unified VGPRs, the 2 waves/SIMD
    # threshold on gfx950's 512-VGPR file. Asked for on both load paths.
    #
    # This used to be gated on buffer_load, because a cache past buffer_load's 2 GB
    # offset gathers through 64-bit addresses and used to land at ~321 VGPRs, where
    # the cap bought nothing and cost ~150 scratch stores (+40% at C4A). Narrowing
    # the scale gather brought that path to 283, close enough that the cap now
    # reaches 256 with 17 spill slots instead of 148 -- worth 0.745x at extra=272
    # and 0.833x at extra=1024 on the 64-bit path. Re-measure this gate whenever the
    # register picture moves.
    if chunk_axis == 0 or nope_chunk < head_dim:
        waves_per_eu = 2
    # num_warps is 4 -- one wave per SIMD per workgroup -- so the second occupancy
    # slot exists only if a CU gets a second workgroup. When the entire launch is at
    # most one workgroup per CU it cannot, and the 2-waves/EU register cap is then
    # pure loss: it spills ~29 dwords across the per-tile softmax row-max reduction
    # and buys no overlap. Measured on the main kernel at C in {4,16,32,64,128} x
    # top-k in {64,272,1024}: 1 wave/EU + ASM_DEQ is 0.927-0.953x at every shape with
    # launch <= CU, and 1.19-1.43x *worse* at every shape above it.
    one_wg_per_cu = (
        use_buffer_load
        and num_queries * heads_blocks * num_splits <= get_num_sms()
    )
    if one_wg_per_cu:
        waves_per_eu = 1
    _wp = _os.environ.get("AITER_PA_DECODE_WPEU")
    if _wp:
        waves_per_eu = int(_wp)

    # Cap the main segment's split count at whole BLOCK_K tiles: past that every
    # split's main range is a partial tile, which costs a full masked gather for
    # half the tokens. extra keeps all num_splits programs.
    main_splits = num_splits
    if has_extra and avg_main > 0:
        main_splits = max(1, min(num_splits, int(math.ceil(avg_main / BLOCK_K))))
    _ms = _os.environ.get("AITER_PA_DECODE_SPLITS_MAIN")
    if _ms:
        main_splits = max(1, min(num_splits, int(_ms)))

    # AITER_PA_DECODE_ADAPTIVE: per-query split count decided in-kernel. Only
    # meaningful when there is more than one split to give back.
    adaptive_splits = (
        num_splits > 1
        and _os.environ.get("AITER_PA_DECODE_ADAPTIVE", "1") == "1"
    )

    # AITER_PA_DECODE_ASM_DEQ: fuse the fp8 x E8M0 dequant into
    # v_cvt_scalef32_pk_bf16_fp8 via inline asm. gfx950 only, packed OCP fp8 only
    # (UNIFORM scales are f32 and FNUZ is a different encoding), and bit-identical
    # there because the scale is a power of two.
    # Default follows the occupancy decision: the fused convert is a shorter
    # dependent chain but a wider live range, so it pays only where the extra
    # registers are free (see one_wg_per_cu). 0/1 forces it either way.
    asm_deq = (
        _os.environ.get("AITER_PA_DECODE_ASM_DEQ", "1" if one_wg_per_cu else "0") == "1"
        and not UNIFORM
        and not fp8_fnuz
        and main_is_fp8
    )

    # AITER_PA_DECODE_GRID_ORDER: which launch axis varies fastest. Dim 0 is the
    # fastest-varying and XCD assignment is round-robin over the linear workgroup id,
    # so this decides what shares an XCD's L2. "qsh" is the historical order.
    _go = _os.environ.get("AITER_PA_DECODE_GRID_ORDER", "qsh")
    _ax = {"q": num_queries, "s": num_splits, "h": heads_blocks}
    assert sorted(_go) == ["h", "q", "s"], f"bad AITER_PA_DECODE_GRID_ORDER {_go!r}"
    # AITER_PA_DECODE_UNI_TILE: run the partial last tile through the same body as the
    # full ones instead of peeling a masked copy of it. Gluon inlines, so the peeled
    # copy is a second ~1000-instruction gather+dequant+MFMA body whose register demand
    # spills the tile loop: worth 0.900-0.907x on the buffer path and 0.943-0.972x on
    # the 64-bit path at >=2 tiles/split, scaling with tiles per split. 0 restores the
    # peeled tail.
    uni_tile = _os.environ.get("AITER_PA_DECODE_UNI_TILE", "1") == "1"

    grid = tuple(_ax[c] for c in _go)
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
        CS0_ALIGN=cs0_align,
        NOPE_DIM=nope_dim,
        ROPE_DIM=ROPE_DIM,
        HEAD_SIZE=head_dim,
        BLOCK_M=BLOCK_M,
        BLOCK_K=BLOCK_K,
        NUM_SPLITS=num_splits,
        HEAD_ALIGNED=HEAD_ALIGNED,
        MFMA_K=MFMA_K,
        UNIFORM=UNIFORM,
        GATHER_TW1=gather_tw1,
        LDS_PAD=lds_pad,
        NOPE_CHUNK=nope_chunk,
        CHUNK_AXIS=chunk_axis,
        PART_STORE_CACHE=_os.environ.get('AITER_PA_DECODE_PART_ST', ''),
        GRID_ORDER=_go,
        UNI_TILE=uni_tile,
        MAIN_SPLITS=main_splits,
        ADAPTIVE_SPLITS=adaptive_splits,
        ASM_DEQ=asm_deq,
        MAIN_USE_BUFFER_LOAD=main_use_buffer_load,
        EXTRA_USE_BUFFER_LOAD=extra_use_buffer_load,
        IDX_BUFFER_LOAD=idx_use_buffer_load,
        HAS_INVALID=has_invalid,
        FP8_FNUZ=fp8_fnuz,
        num_warps=num_warps,
        waves_per_eu=waves_per_eu,
    )

    if num_splits == 1:
        return out
    if skip_reduce:
        return part_acc, part_m, part_l

    # The combine is pure bandwidth over [num_queries, num_heads, head_dim] f32,
    # so size its head tile for workgroup count, not for the attention kernel's
    # BLOCK_M: at BLOCK_M=num_heads there is one workgroup per query (64 on a
    # 256-CU part). One head per workgroup gives num_queries*num_heads of them and
    # enough concurrent waves to hide the partial-load latency.
    red_block_m = 1
    _rbm = _os.environ.get("AITER_PA_DECODE_REDUCE_BLOCK_M")
    if _rbm:
        red_block_m = int(_rbm)
    red_warps = 1
    _rw = _os.environ.get("AITER_PA_DECODE_REDUCE_WARPS")
    if _rw:
        red_warps = int(_rw)
    rgrid = (num_queries, (num_heads + red_block_m - 1) // red_block_m)
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
        BLOCK_M=red_block_m,
        NUM_SPLITS=num_splits,
        HEAD_ALIGNED=num_heads % red_block_m == 0,
        ADAPTIVE_SPLITS=adaptive_splits,
        PART_LOAD_CACHE=_os.environ.get("AITER_PA_DECODE_PART_LD", ""),
        num_warps=red_warps,
    )
    return out
