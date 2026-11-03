# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Sparse paged-decode attention over a unified or split KV pool with per-token paged
indices.

TODO: add details once API has settled
"""

import functools
import math

import torch
import triton
from packaging.version import Version

from aiter.ops.triton._gluon_kernels.gfx950.attention.sparse_mla import (
    _sparse_mla as _sparse_mla_gfx950,
)
from aiter.ops.triton._gluon_kernels.gfx950.attention.sparse_mla import (
    _sparse_mla_reduce as _sparse_mla_reduce_gfx950,
)
from aiter.ops.triton._gluon_kernels.gfx1250.attention.pa_decode_sparse import (
    _pa_decode_sparse as gluon_pa_decode_sparse,
)
from aiter.ops.triton._gluon_kernels.gfx1250.attention.pa_decode_sparse import (
    _pa_decode_sparse_fp8 as gluon_pa_decode_sparse_fp8,
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
from aiter.ops.triton.utils.device_info import get_num_sms, get_num_xcds
from aiter.ops.triton.utils.logger import AiterTritonLogger

DEVICE_ARCH = arch_info.get_arch()

# Fused fp8 x E8M0 -> bf16 upcast check (gluon cdna4.scaled_upcast)
try:
    from triton.experimental.gluon.language.amd import cdna4 as _cdna4

    _HAS_SCALED_UPCAST = hasattr(_cdna4, "scaled_upcast")
except ImportError:
    _HAS_SCALED_UPCAST = False

# Below Triton 3.8 the peeled 64-bit split-K kernel spills ~110 VGPRs to scratch.
_TRITON_GE_38 = Version(triton.__version__) >= Version("3.8.0")

_LOGGER = AiterTritonLogger()


_FP8_GROUP_SIZE = 64

# Launches of at least this many rows get the prefill config. Row count is the
# only signal, and decode is slower with it, so this sits above decode batch sizes.
_PREFILL_MIN_ROWS = 2048

# Split-K cap
_MAX_SPLITS = 64


def _check_out(out, q, dtype):
    """Caller-supplied output buffer, or a fresh one. Writing the caller's buffer
    directly saves a full [T, H, D] device copy per call."""
    if out is None:
        return torch.empty_like(q, dtype=dtype)
    assert out.shape == q.shape, f"out shape {tuple(out.shape)} != q {tuple(q.shape)}"
    assert out.dtype == dtype, f"out dtype {out.dtype} != {dtype}"
    assert out.device == q.device
    return out


# DSv4 head geometry and the gfx1250 640 B record. The byte layouts themselves
# are documented once, in the "Formats" section of pa_decode_sparse below.
_V4_DIM_NOPE = 448
_V4_DIM_ROPE = 64
_V4_DIM_QK = _V4_DIM_NOPE + _V4_DIM_ROPE  # 512, the logical head dim
_V4_PACKED_FP8_DTYPES = (torch.float8_e4m3fn, torch.uint8)
_V4_REC_ALIGNED = 640  # 5 * 128: every token starts 128 B aligned for TDM
_V4_ROPE_IN_REC = 512  # the RoPE row follows the 512 B packed row


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
    use_lru_cache_partials: bool = False,
    *,
    extra_cache: torch.Tensor | None = None,
    extra_indices: torch.Tensor | None = None,
    extra_indptr: torch.Tensor | None = None,
    unified_kv_rope: torch.Tensor | None = None,
    q_rope: torch.Tensor | None = None,
    num_warps: int | None = None,
    ctas_h: int = 1,
    q_tdm: bool = True,
    out: torch.Tensor | None = None,
    inv_rope_positions: torch.Tensor | None = None,
    inv_rope_cos_sin_cache: torch.Tensor | None = None,
    out_mxfp8: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Sparse paged-decode attention with split-K + widened BLOCK_H.

    Formats
    -------
    What each arch / precision accepts. The path is chosen from the arch and
    the cache's shape and dtype. DSv4 has D = 512 = 448 NoPE + 64 RoPE.
    ``N`` = query rows, ``P`` = pool rows, ``nb`` = pages, ``block`` = page
    size (a power of two on every paged cache).

    gfx1250, fp8 (a8w8) -- the 640 B "aligned" record, one per token:

        bytes [  0, 448)  NoPE, 448 x fp8 e4m3 (OCP)
        bytes [448, 462)  14 E8M0 scales: one per 64-element group, each
                          written twice (s0,s0,s1,s1,...,s6,s6), because the
                          scaled WMMA takes one scale per 32 elements
        bytes [462, 512)  50 B pad
        bytes [512, 640)  RoPE, 64 x bf16, never quantized

      KV   ``unified_kv`` ``[nb, block, 640]`` uint8 / fp8 -- vLLM's gfx1250
           paged cache. It is sliced into its [0, 512) and [512, 640) halves
           as views (no copy). Or pass the halves yourself: ``unified_kv``
           ``[P, 512]`` fp8 + ``unified_kv_rope`` ``[P, 64]`` bf16 (ATOM's
           "2buff" pool; byte-identical to what the gfx1250 MLA-v4 asm decode
           kernel ``mla_decode_fwd_v4_nm`` reads).
      Q    ``q`` ``[N, H, 512]`` fp8, packed like bytes [0, 512) of the record,
           plus ``q_rope`` ``[N, H, 64]`` bf16; see
           ``fused_deepseek_v4_mxfp8_quant_q_pack``. bf16 Q is not accepted.
      ``kv_scales`` must be None (the scales are inside the row). ``extra_*``
      is supported (same record). Output ``[N, H, 512]`` bf16. The
      ``inv_rope_positions`` / ``out_mxfp8`` epilogue is not implemented here
      and is rejected.

    gfx1250, bf16:
      KV   ``unified_kv`` ``[P, D]`` flat pool (page size 1), same dtype as Q.
      Q    ``q`` ``[N, H, D]`` bf16/fp16, D a power of two.
      No ``kv_scales``, no ``extra_*``.

    gfx950, fp8 (a16w8) -- vLLM's default ``fp8_ds_mla`` "trailer" cache:
      KV   ``unified_kv`` ``[nb, block, 584]`` uint8. Each page holds
           ``block`` x 576 B of token data -- [0, 448) NoPE fp8 e4m3 |
           [448, 576) RoPE 64 x bf16 -- followed by a trailer of ``block`` x
           8 B scales (7 UE8M0, one per 64-element group, + 1 pad). 584 is the
           per-token size; tokens themselves sit at a 576 B stride.
           Also accepted: a flat ``[P, D]`` fp8 e4m3 / uint8 pool with
           ``kv_scales`` ``[P, D // 64]`` fp32 (single stream only).
      Q    ``q`` ``[N, H, 512]`` bf16 only (the kernel has no Q scale input).
      ``extra_*`` is supported on the 584 B cache. The ``inv_rope_positions``
      / ``out_mxfp8`` output epilogue is supported.

    gfx950, bf16:
      KV   ``unified_kv`` ``[nb, block, 512]`` bf16 paged cache (``extra_*``
           supported), or a flat ``[P, D]`` bf16 pool.
      Q    ``q`` ``[N, H, D]`` bf16.

    Anything else -- other arches, fp16 Q or an explicit ``block_h`` on
    gfx950 -- takes the Triton kernel: bf16/fp16 flat ``[P, D]`` pool, KV in
    Q's dtype, no ``kv_scales``, no ``extra_*``.

    The two fp8 records are NOT interchangeable: gfx1250 keeps the scales
    inside each 640 B row, gfx950 keeps them in a per-page trailer behind 576 B
    rows. Feeding gfx950 the 640 B record is not detected and reads the pad as
    exponents (all-NaN output).

    Args:
        q: ``[N, H, D]`` decode queries; dtype per arch, see Formats.
        unified_kv: the KV cache; shape and dtype per arch, see Formats.
        kv_indices: ``[total_indices]`` int32 — per-token slot lists, flat.
            Per-token entries live in ``kv_indices[kv_indptr[t] : kv_indptr[t+1]]``.
            ``-1`` entries are skipped (sentinel for unused tail).
        kv_indptr: ``[N+1]`` int32 — true prefix sum.
        attn_sink: ``[H]`` per-head learnable softmax-denom bias (fp32).
        softmax_scale: scalar softmax scale.
        kv_scales: gfx950 only -- ``[P, D // 64]`` fp32 scales for a flat fp8
            pool. Must be None everywhere else.
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
        use_lru_cache_partials: reuse the split-K partial buffers across calls
            rather than allocating them per call. They run to tens of MB per
            layer, so a per-step forward churns gigabytes through the
            allocator, and the caller's runtime cannot account for them.
            Cached entries are never freed -- a captured CUDA graph records
            their addresses. Ignored under ``skip_reduce``, where the caller
            owns the partials.
        extra_cache/extra_indices/extra_indptr: a second cache and ragged index
            set attended over in the same pass. vLLM's decode splits its keys
            this way and these follow its naming: the FIRST (positional) cache
            is its ``main_cache`` = ``swa_k_cache``, the sliding-window keys,
            and this one is its ``extra_cache`` = ``kv_cache``, the compressed
            keys the sparse top-k selects. The two hold the identical record.
            Supported on the gfx950 paged caches and the gfx1250 fp8 path;
            must be None otherwise. See Formats.
        unified_kv_rope: gfx1250 only -- ``[P, 64]`` bf16 RoPE plane. Supplying
            it selects the gfx1250 fp8 path with ``unified_kv`` as the
            ``[P, 512]`` packed half (see Formats); ``q_rope`` is then
            required.
        q_tdm: DEFAULT ON. Load Q by TDM descriptor into LDS instead of masked
            buffer_loads. Worth 90.4 -> 71.7us (kv_len 384) and
            58.9 -> 43.5us (kv_len 136), and it takes VGPR spills 146 -> 0 on
            its own, because Q stops materialising [BLOCK_H, D] register tiles
            and lands in LDS already in dot_q_layout. Set False to A/B it.
        ctas_h: EXPERIMENTAL, default 1 (off). Launch ``ctas_h`` CTAs as one
            workgroup cluster, splitting ``block_h`` across them. All CTAs of a
            cluster serve the same token and want the same KV rows, so the KV
            fetch is multicast once per cluster instead of re-fetched per
            head-block. Only pays off when the kernel is spilling: it helped
            the MX kernel a lot (130.4 -> 83.7us) and the un-TDM'd stage kernel
            at short kv_len, but HURTS the stage kernel once ``q_tdm`` has
            removed its spills (72.2 -> 97.9us).
            UNVERIFIED: the cluster barriers around the multicast refills were
            reordered (see the fp8 kernel's main-loop note) but not re-tested,
            and ``cluster.arrive``/``wait`` need every CTA of a cluster
            co-resident; the large per-CTA LDS of the ``q_tdm`` path may prevent
            that and hang. Re-verify before enabling.
        num_warps: override the warp count. The default ties warps to
            ``block_h`` so the per-thread accumulator is
            ``block_h * D / (num_warps * 32)`` = 256 VGPRs at every block_h;
            overriding it is the only way to shrink the accumulator's register
            footprint.
        q_rope: ``[N, H, 64]`` bf16 RoPE plane for Q, required on the gfx1250
            fp8 path, where ``q`` is the packed fp8 ``[N, H, 512]`` Q.
        inv_rope_positions/inv_rope_cos_sin_cache/out_mxfp8: gfx950 gluon only —
            the output epilogue (see _pa_decode_sparse_gfx950_gluon). Rejected
            everywhere else.

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
    # Every path takes the second stream whole or not at all; a partial set
    # would otherwise be dropped without a word.
    _extra = (extra_cache, extra_indices, extra_indptr)
    if any(x is not None for x in _extra) and any(x is None for x in _extra):
        raise ValueError(
            "extra_cache, extra_indices and extra_indptr go together: pass all "
            "three or none"
        )

    # gfx1250 fp8: a [nb, block, 640] cache is split into the two views the
    # kernel reads (see Formats). Gated on the arch: gfx950 has no 640 B path,
    # and without the gate this would also claim gfx950 calls and raise
    # "gfx1250-only". An explicit unified_kv_rope on gfx950 still reaches that
    # raise, which is right -- the caller asked for this path by name.
    if (
        DEVICE_ARCH == "gfx1250"
        and unified_kv_rope is None
        and unified_kv.dim() == 3
        and unified_kv.shape[-1] == _V4_REC_ALIGNED
        and unified_kv.dtype in _V4_PACKED_FP8_DTYPES
    ):
        unified_kv, unified_kv_rope = _v4_split_aligned(unified_kv)
        if extra_cache is not None:
            extra_cache, extra_cache_rope = _v4_split_aligned(extra_cache)
        else:
            extra_cache_rope = None
    else:
        extra_cache_rope = None

    v4_2buff = unified_kv_rope is not None
    if v4_2buff:
        if (
            inv_rope_positions is not None
            or inv_rope_cos_sin_cache is not None
            or out_mxfp8 is not None
        ):
            raise NotImplementedError(
                "inv_rope_positions / inv_rope_cos_sin_cache / out_mxfp8 (the "
                "output epilogue) are gfx950 gluon only; the gfx1250 fp8 path "
                "returns bf16 without the inverse RoPE"
            )
        return _pa_decode_sparse_gfx1250_fp8(
            q,
            unified_kv,
            unified_kv_rope,
            kv_indices,
            kv_indptr,
            attn_sink,
            softmax_scale,
            q_rope=q_rope,
            kv_scales=kv_scales,
            num_warps=num_warps,
            ctas_h=ctas_h,
            q_tdm=q_tdm,
            block_h=block_h,
            kv_splits=kv_splits,
            has_invalid=has_invalid,
            skip_reduce=skip_reduce,
            use_lru_cache_partials=use_lru_cache_partials,
            out=out,
            extra_cache=extra_cache,
            extra_cache_rope=extra_cache_rope,
            extra_indices=extra_indices,
            extra_indptr=extra_indptr,
        )
    if q_rope is not None:
        raise RuntimeError("q_rope requires unified_kv_rope (the DSv4 2buff path)")

    if q.dtype not in (torch.bfloat16, torch.float16):
        raise RuntimeError(f"pa_decode_sparse expects fp16/bf16 q, got {q.dtype}")
    _LOGGER.info(
        "PA_DECODE_SPARSE: q=%s unified_kv=%s %s kv_indices=%s",
        tuple(q.shape),
        tuple(unified_kv.shape),
        unified_kv.dtype,
        tuple(kv_indices.shape),
    )

    # gfx950: route to the merged DSv4 sparse-MLA gluon driver. Format is inferred
    # from the cache: 3D -> packed fp8_dsv4_mla / bf16 block cache (optional SWA+top-k
    # two-loop via extra_*); 2D -> uniform pool (OCP fp8 + fp32 kv_scales, or bf16).
    # kv_splits and skip_reduce are honored here; block_h and fp16 q fall through to
    # the triton path below (the gluon kernel is bf16-only: bf16 LDS + bf16 MFMA).
    if DEVICE_ARCH == "gfx950" and block_h is None and q.dtype == torch.bfloat16:
        # gfx950 (CDNA4) reads OCP e4m3 natively. fnuz is the gfx942 encoding,
        # so it never appears here and falls through to the triton path below.
        if unified_kv.ndim == 3:
            # packed / bf16 block cache: it carries its own scales, if any
            _ok = kv_scales is None and unified_kv.dtype in (torch.uint8, q.dtype)
        elif kv_scales is not None:
            _ok = unified_kv.dtype in (torch.float8_e4m3fn, torch.uint8)
        else:
            _ok = unified_kv.dtype == q.dtype
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
                out=out,
                inv_rope_positions=inv_rope_positions,
                inv_rope_cos_sin_cache=inv_rope_cos_sin_cache,
                out_mxfp8=out_mxfp8,
            )

    assert extra_cache is None and extra_indices is None and extra_indptr is None, (
        "extra_cache/extra_indices/extra_indptr need a gfx950 paged cache or "
        "the gfx1250 fp8 path"
    )
    assert (
        inv_rope_positions is None and out_mxfp8 is None
    ), "the output epilogue is gfx950 gluon-only"
    if kv_scales is not None:
        raise RuntimeError(
            "kv_scales (a flat fp8 pool with fp32 scales) is gfx950-only, with "
            "bf16 q and no block_h; on gfx1250 use the 640 B fp8 record"
        )
    if unified_kv.dtype != q.dtype:
        raise RuntimeError(
            f"unified_kv dtype mismatch: kv={unified_kv.dtype}, q={q.dtype}"
        )

    T, H, D = q.shape

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

    # Infer KV_SPLITS from inputs when caller doesn't override.
    # Fill ~512 total CTAs (MI300X has 304 CUs) while never splitting K into
    # more pieces than there are K-blocks. Rounded up to a power of 2 so the
    # reduce kernel's tl.arange(0, KV_SPLITS) compiles; over-splitting past
    # max_kv_splits is handled by the kernel (empty splits early-return and
    # the reduce masks their stale partial-buffer slots).
    # print(f"{kv_indices.shape[0]=}")
    if kv_splits is None:
        # PER TOKEN: the split is per token, so the ceiling is the tile count
        # ONE token has. kv_indices.shape[0] is every token's indices together,
        # so the ceiling never bound and splits were made with no work in them
        # -- and a dead split still costs the reduce a slab row.
        avg_kv_len = max(1, kv_indices.shape[0] // max(1, T))
        max_kv_splits = max(1, triton.cdiv(avg_kv_len, block_k))
        kv_splits = max(1, max_num_wg // max(1, T * n_head_blocks))
        kv_splits = min(max_kv_splits, kv_splits)
        # DOWN to a power of two: rounding up puts back what the ceiling removed.
        kv_splits = 1 << (max(1, kv_splits).bit_length() - 1)

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
        HAS_EXTRA=False,
        num_warps=reduce_num_warps,
        waves_per_eu=reduce_waves_per_eu,
    )
    return out


def _as_int32_contiguous_1d(x: torch.Tensor) -> torch.Tensor:
    if x.dtype == torch.int32 and x.ndim == 1 and x.is_contiguous():
        return x
    return x.to(torch.int32).contiguous()


def _decode_num_splits_occ(num_queries, heads_blocks, avg_main, avg_extra, block_k):
    """Split-K count for the gfx950 gluon kernel: fill the machine, but never
    split a segment finer than one BLOCK_K tile or past _MAX_SPLITS.
    """
    num_sms = get_num_sms()
    base_wg = max(1, num_queries * heads_blocks)
    cta_cap = max(1, (2 * num_sms) // base_wg)
    main_tiles = max(1, math.ceil(avg_main / block_k)) if avg_main > 0 else 0
    extra_tiles = max(1, math.ceil(avg_extra / block_k)) if avg_extra > 0 else 0
    tiles = max(1, main_tiles, extra_tiles)
    if tiles <= 2:
        # Splitting two tiles saves less than the reduce launch costs.
        return 1
    if base_wg >= num_sms:
        # Already at least one workgroup per CU without splitting
        return max(1, min(cta_cap, tiles // 4))
    return max(1, min(cta_cap, tiles, _MAX_SPLITS))


def _last_round_fills(programs, num_sms):
    """The grid's last round of programs is full or more than half full."""
    left = programs % num_sms
    return left == 0 or 2 * left > num_sms


def _rounds_suit_32(num_heads, num_queries, num_splits, tiles):
    """The 32-head grid ends on a full or more-than-half-full round (up to four
    rounds), or its programs have four or more tiles (past four rounds)."""
    num_sms = get_num_sms()
    programs = num_queries * (num_heads // 32) * num_splits
    if programs <= 4 * num_sms:
        return _last_round_fills(programs, num_sms)
    return tiles >= 4


def _dsv4_block_m(num_heads, num_queries, num_splits, row_tiles, has_extra):
    """Heads per program (16, 32 or 64) for fp8_dsv4_mla at 32 or 64 heads: 64
    for top-k launches without split-K from one full round of rows (up to four
    rounds, the last more than half full); else 32 past one tile per program."""
    num_sms = get_num_sms()
    rows_fill = num_queries >= num_sms and (
        _last_round_fills(num_queries, num_sms) or num_queries > 4 * num_sms
    )
    if num_heads == 64 and has_extra and num_splits == 1 and rows_fill:
        return 64
    tiles = row_tiles / num_splits
    if tiles > 1 and _rounds_suit_32(num_heads, num_queries, num_splits, tiles):
        return 32
    return 16


def _staged_block_m(num_heads, num_queries, num_splits, row_tiles):
    """Heads per program (16, 32 or 64) at 32 or 64 heads for the staged walks
    (per-tensor fp8; bf16 with the rope inside): 64 without split-K once
    2 x rows > CUs; else 32 from two tiles per program."""
    num_sms = get_num_sms()
    if num_heads == 64 and num_splits == 1 and 2 * num_queries > num_sms:
        return 64
    tiles = row_tiles / num_splits
    if tiles >= 2 and _rounds_suit_32(num_heads, num_queries, num_splits, tiles):
        return 32
    return 16


def _launch_splits(num_splits):
    """Split programs to launch: past 2, rounded up to a multiple of 4, so the
    reduce (unrolled over the launched count) compiles for few counts. The extra
    programs exit early with empty partials."""
    return num_splits if num_splits <= 2 else -(-num_splits // 4) * 4


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
    inv_rope_positions=None,
    inv_rope_cos_sin_cache=None,
    out_mxfp8=None,
):
    """Merged gfx950 gluon DSv4 sparse-MLA decode driver. Format from cache.ndim:
    3D [nb, block, 584] -> packed fp8_dsv4_mla (uint8: 448 NoPE fp8 e4m3 OCP +
                           embedded UE8M0 per-64 scale + 64 RoPE bf16) or a bf16
                           block cache; pass extra_* for the SWA+top-k two-loop,
                           else a single segment.
    2D [pages, D]       -> uniform pool: fp8 (uint8) + cache_scales
                           [pages, D//64] fp32, or bf16 (cache_scales None).

    The output epilogue applies what vLLM runs on the rows before wo_a:
    inv_rope_positions [N] with inv_rope_cos_sin_cache [P, 64] f32 (cos | sin)
    rotate the trailing 64 lanes of each row back (inverse GPT-J RoPE), and
    out_mxfp8 = (data [N, H * D] e4m3, scale [N, H * D // 32] uint8 E8M0)
    replaces out with its MXFP8 quantization; data viewed as [N, H, D] is
    returned.
    """
    assert q.ndim == 3, f"expected q=[b,h,d], got {q.shape}"
    assert DEVICE_ARCH == "gfx950", "gluon DSv4 decode kernel is gfx950-only"

    # Tuned launch config (gfx950 / MI355). BLOCK_M = heads per MFMA M-tile, and 16
    # is both the MFMA M and the DSv4 head count; BLOCK_K = KV tile; num_warps =
    # BLOCK_K // 16, because warps tile the dot-N and MFMA N = 16.
    BLOCK_M, BLOCK_K = 16, 64
    num_warps = max(1, BLOCK_K // 16)
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
        FLAT_POOL = True
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
        avg_main = indices.numel() / max(1, num_queries)  # one segment; no extra
        avg_extra = 0.0
    else:
        # packed fp8_dsv4_mla [nb, block, 584] (UE8M0 block trailer) or bf16 block
        # cache. NB: vLLM's fp8_ds_mla also names the 656 B V3.2 layout, which is
        # fp8_dsv32_mla and reaches the kernel through sparse_mla.py instead.
        FLAT_POOL = False
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
        avg_main = indices.numel() / max(1, num_queries)
        avg_extra = extra_indices.numel() / max(1, num_queries) if has_extra else 0.0

    # Kernel-side cache-format tags (kernel shared with sparse_mla.py).
    if FLAT_POOL:
        main_fmt = "fp8_g64" if main_is_fp8 else "bf16"
        extra_fmt = main_fmt
    else:
        main_fmt = "fp8_dsv4_mla" if main_is_fp8 else "bf16"
        extra_fmt = "fp8_dsv4_mla" if extra_is_fp8 else "bf16"
        # The fp8_dsv4_mla gather splits slot ids into (page, row) by shifts.
        for fmt, block in ((main_fmt, main_block), (extra_fmt, extra_block)):
            assert (
                fmt != "fp8_dsv4_mla" or block & (block - 1) == 0
            ), f"fp8_dsv4_mla page size must be a power of two, got {block}"

    # Alignment hint for the page strides so row gathers can vectorize: the largest
    # power of 2 (<= 16) dividing both.
    s0, s1 = int(cache.stride(0)), int(extra_cache.stride(0))
    cs0_align = 1
    for a in (16, 8, 4, 2):
        if s0 % a == 0 and s1 % a == 0:
            cs0_align = a
            break

    # Gate each cache on its own span: buffer_load carries a 32-bit offset, and one
    # oversized cache must not drop the fast path for the other. The index lists are
    # one int32 per gathered token, far under the limit even for a full batch.
    main_use_buffer_load = max_addressable_bytes(cache) < MAX_BYTES
    extra_use_buffer_load = max_addressable_bytes(extra_cache) < MAX_BYTES
    idx_use_buffer_load = (
        max_addressable_bytes(indices) < MAX_BYTES
        and max_addressable_bytes(extra_indices) < MAX_BYTES
    )
    use_buffer_load = main_use_buffer_load and extra_use_buffer_load
    packed_fp8 = (
        not FLAT_POOL
        and main_fmt == "fp8_dsv4_mla"
        and (not has_extra or extra_fmt == "fp8_dsv4_mla")
    )
    prefill = num_queries >= _PREFILL_MIN_ROWS
    row_tiles = max(avg_main, avg_extra) / BLOCK_K
    # Split count from the 16-head grid, for every program size.
    if kv_splits is not None:
        num_splits = max(1, int(kv_splits))
    else:
        num_splits = _decode_num_splits_occ(
            num_queries,
            (num_heads + BLOCK_M - 1) // BLOCK_M,
            avg_main,
            avg_extra,
            BLOCK_K,
        )
    staged_bf16 = (
        not FLAT_POOL and main_fmt == "bf16" and (not has_extra or extra_fmt == "bf16")
    )
    # 32- and 64-head programs (8 warps) stage each key tile once for all heads.
    if num_heads in (32, 64):
        if packed_fp8:
            BLOCK_M = _dsv4_block_m(
                num_heads, num_queries, num_splits, row_tiles, has_extra
            )
        elif staged_bf16:
            BLOCK_M = _staged_block_m(num_heads, num_queries, num_splits, row_tiles)
    if BLOCK_M > 16:
        num_warps = 8
    HEAD_ALIGNED = num_heads % BLOCK_M == 0
    heads_blocks = (num_heads + BLOCK_M - 1) // BLOCK_M
    inv_rope = inv_rope_positions is not None
    assert inv_rope == (
        inv_rope_cos_sin_cache is not None
    ), "inv_rope_positions and inv_rope_cos_sin_cache go together"
    assert not (
        skip_reduce and (inv_rope or out_mxfp8 is not None)
    ), "the output epilogue runs in the reduce, so skip_reduce cannot be set"
    if inv_rope:
        assert inv_rope_positions.shape == (num_queries,)
        assert inv_rope_positions.stride(0) == 1
        assert inv_rope_cos_sin_cache.dtype == torch.float32
        # [P, 64]: the kernel steps rows by stride(0)
        assert inv_rope_cos_sin_cache.ndim == 2
        assert inv_rope_cos_sin_cache.shape[1] == ROPE_DIM
        assert inv_rope_cos_sin_cache.stride(1) == 1
        assert inv_rope_positions.device == q.device
        assert inv_rope_cos_sin_cache.device == q.device
    if out_mxfp8 is not None:
        assert out is None, "out and out_mxfp8 are mutually exclusive"
        out_data, out_scale = out_mxfp8
        assert out_data.dtype == torch.float8_e4m3fn and out_scale.dtype == torch.uint8
        assert out_data.shape == (num_queries, num_heads * head_dim)
        assert out_scale.shape == (num_queries, num_heads * head_dim // 32)
        assert out_data.stride(-1) == 1 and out_scale.stride(-1) == 1
        assert out_data.device == q.device and out_scale.device == q.device
        out = out_data.view(num_queries, num_heads, head_dim)
    else:
        out_scale = None
        out = _check_out(out, q, torch.bfloat16)
    epilogue = {
        "pos_ptr": inv_rope_positions,
        "cos_sin_ptr": inv_rope_cos_sin_cache,
        "cs_stride": inv_rope_cos_sin_cache.stride(0) if inv_rope else 0,
        "out_scale_ptr": out_scale,
        "os_stride0": out_scale.stride(0) if out_scale is not None else 0,
        "INV_ROPE": inv_rope,
        "OUT_MXFP8": out_scale is not None,
    }

    # Q is read once per query without split-K, and re-read by every split
    q_cache = ".cg" if num_splits == 1 else ""
    # skip_reduce hands the partials to the caller, so only our own reduce pads.
    grid_splits = num_splits if skip_reduce else _launch_splits(num_splits)

    if num_splits > 1:
        part_m = torch.empty(
            (num_queries, grid_splits, num_heads), dtype=torch.float32, device=q.device
        )
        part_l = torch.empty_like(part_m)
        # bf16 partials halve both the split-K HBM traffic (~31% of the kernel's
        # bytes)
        part_acc = torch.empty(
            (num_queries, grid_splits, num_heads, head_dim),
            dtype=torch.float32 if skip_reduce else torch.bfloat16,
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

    # Dequant chunking. The gather layout puts 32 of a wave's 64 lanes along a row
    # (16 B each, 512 B) and the other 32 on a second row; col_reps is how many
    # such spans each lane holds, which is what makes a column split a free
    # register rename.
    col_reps = head_dim // 512
    chunk_axis = 1 if col_reps >= 4 else 0
    nope_chunk = max(1, BLOCK_K // 4) if chunk_axis == 0 else min(128, head_dim)

    prefill_kw = {}
    if num_queries >= _PREFILL_MIN_ROWS:
        # Prefill rows re-read each other's KV rows, so cache the gather instead of .cg.
        prefill_kw["GATHER_CACHE"] = ""
        if main_fmt == extra_fmt == "fp8_dsv4_mla":
            prefill_kw["SLOT_U32"] = max(s0, s1) < (1 << 24)
            nope_chunk = max(1, BLOCK_K // 8)

    waves_per_eu = 2
    one_wg_per_cu = (
        use_buffer_load and num_queries * heads_blocks * num_splits <= get_num_sms()
    )
    if one_wg_per_cu:
        waves_per_eu = 1

    # Decode on buffer loads is faster peeled; prefill is faster unpeeled.
    # On the 64-bit gathers unpeeled is faster unless split-K leaves each program
    # a few tiles, and below Triton 3.8 the peeled kernel spills, so always unpeel.
    short_splits = num_splits > 1 and row_tiles <= 4 * num_splits
    if use_buffer_load:
        unpeel = num_queries >= _PREFILL_MIN_ROWS
    else:
        unpeel = not (_TRITON_GE_38 and short_splits)

    main_splits = num_splits
    if has_extra and avg_main > 0:
        main_splits = max(1, min(num_splits, math.ceil(avg_main / BLOCK_K)))

    # Per-query split count decided in-kernel; only meaningful when there is more
    # than one split to give back.
    adaptive_splits = num_splits > 1

    # dsv4 dequant: the scaled upcast if this Triton has it, else inline asm.
    deq = "upcast" if _HAS_SCALED_UPCAST else "asm"

    # The 16-lane row gather needs row-axis dequant chunks of 4+ rows per warp.
    if packed_fp8 and chunk_axis == 0:
        nope_chunk = max(nope_chunk, 4 * num_warps)
    if packed_fp8:
        # Rows that share KV rows reuse them through the cache, so skip .cg.
        prefill_kw["GATHER_CACHE"] = ""

    # One XCD (and L2) for programs that read the same KV rows: a row's head blocks,
    # or neighbouring rows of bf16's 32/64-head programs without split-K. Not for
    # SWA-only prefill or launches padded past the split count.
    rows_share = heads_blocks > 1 or (staged_bf16 and BLOCK_M > 16 and num_splits == 1)
    xcd_remap = (
        get_num_xcds()
        if rows_share and grid_splits == num_splits and (has_extra or not prefill)
        else 0
    )

    # Grid dim 0 varies fastest and XCD assignment is round-robin over the linear
    # workgroup id, so the axis order decides what shares an XCD's L2.
    grid = (num_queries, grid_splits, heads_blocks)
    _sparse_mla_gfx950[grid](
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
        # f32 scale pointers (separated-rope formats only); None is elided,
        # keeping the DSv4 kernarg layout unchanged.
        None,
        None,
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
        MAIN_FMT=main_fmt,
        EXTRA_FMT=extra_fmt,
        MAIN_BLOCK_SIZE=main_block,
        EXTRA_BLOCK_SIZE=extra_block,
        CS0_ALIGN=cs0_align,
        NOPE_DIM=nope_dim,
        ROPE_DIM=ROPE_DIM,
        HEAD_SIZE=head_dim,
        ROPE_SEPARATE=False,
        BLOCK_M=BLOCK_M,
        BLOCK_K=BLOCK_K,
        num_splits=num_splits,
        SPLIT_K=num_splits > 1,
        HEAD_ALIGNED=HEAD_ALIGNED,
        NOPE_CHUNK=nope_chunk,
        CHUNK_AXIS=chunk_axis,
        PART_STORE_CACHE="",
        Q_CACHE=q_cache,
        GRID_ORDER="qsh",
        # The partial last tile rides the full-tile body. Gluon inlines, so a peeled
        # masked copy would be a second gather+dequant+MFMA body, and its register
        # demand spills the tile loop.
        UNI_TILE=True,
        main_num_splits=main_splits,
        ADAPTIVE_SPLITS=adaptive_splits,
        DEQ=deq,
        MAIN_USE_BUFFER_LOAD=main_use_buffer_load,
        EXTRA_USE_BUFFER_LOAD=extra_use_buffer_load,
        IDX_BUFFER_LOAD=idx_use_buffer_load,
        HAS_INVALID=has_invalid,
        UNPEEL=unpeel,
        XCD_REMAP=xcd_remap,
        # Gather a tile ahead only for fp8_dsv4_mla's 32-head programs below prefill
        # size.
        DSV4_PREFETCH=packed_fp8 and BLOCK_M == 32 and not prefill,
        num_warps=num_warps,
        waves_per_eu=waves_per_eu,
        **prefill_kw,
        # The epilogue runs where the output is written.
        **(epilogue if num_splits == 1 else {}),
    )

    if num_splits == 1:
        return out
    if skip_reduce:
        return part_acc, part_m, part_l

    # One head per reduce workgroup
    rgrid = (num_queries, num_heads)
    _sparse_mla_reduce_gfx950[rgrid](
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
        BLOCK_M=1,
        NUM_SPLITS=grid_splits,
        HEAD_ALIGNED=True,
        ADAPTIVE_SPLITS=adaptive_splits,
        ROPE_DIM=ROPE_DIM,
        **epilogue,
        # A 2-split tile spans two warps; more warps would hold duplicate lanes.
        num_warps=min(4, grid_splits),
    )
    return out


def _v4_split_aligned(cache: torch.Tensor):
    """An aligned paged cache -> the two buffers the 2buff kernel reads.

    ``[nb, block, 640]`` uint8 becomes ``[nb, block, 512]`` (the 2buff row:
    NoPE, its duplicated scales and pad) and ``[nb, block, 64]`` bf16 (the RoPE
    row). Both are VIEWS -- they keep the 640-byte record stride, which the
    descriptors take as the row stride, so nothing is copied or repacked.
    """
    if cache.shape[-1] != _V4_REC_ALIGNED:
        raise RuntimeError(
            f"aligned cache records must be {_V4_REC_ALIGNED} B, got "
            f"{cache.shape[-1]}"
        )
    u8 = cache.view(torch.uint8)
    # The data half must arrive TYPED: the kernel feeds it to the scaled MMA as
    # e4m3, and a uint8 tile is upcast down a different path entirely. Both
    # slices keep the 640-byte record stride, so nothing is copied.
    return (
        u8[..., :_V4_DIM_QK].view(torch.float8_e4m3fn),
        u8[..., _V4_ROPE_IN_REC:].view(torch.bfloat16),
    )


def _v4_2buff_geometry(kv: torch.Tensor, rope: torch.Tensor, name: str):
    """(rows, blk_rows, block_size, num_slots) for a 2buff pool.

    A flat ``[P, 512]`` pool is page size 1 with one row per page, so the slot
    IS the row. A paged ``[nb, block, 512]`` view -- what a vLLM aligned cache
    slices to -- is contiguous inside a page but jumps by the pooled block
    stride across pages, so the kernel needs both the page size and the number
    of record-rows that stride spans.
    """
    # The TDM descriptors take the inner stride as 1.
    if kv.stride(-1) != 1 or rope.stride(-1) != 1:
        raise RuntimeError(
            f"{name} and its rope plane must be contiguous along the last dim, "
            f"got strides {kv.stride()} / {rope.stride()}"
        )
    if kv.dim() == 2:
        if kv.shape[1] != _V4_DIM_QK:
            raise RuntimeError(
                f"{name} must be [P, {_V4_DIM_QK}], got {tuple(kv.shape)}"
            )
        if rope.shape[0] != kv.shape[0]:
            raise RuntimeError(f"{name} rope rows {rope.shape[0]} != {kv.shape[0]}")
        return kv.shape[0], 1, 1, kv.shape[0]

    if kv.dim() != 3 or kv.shape[2] != _V4_DIM_QK:
        raise RuntimeError(
            f"{name} must be [P, {_V4_DIM_QK}] or [nb, block, {_V4_DIM_QK}], "
            f"got {tuple(kv.shape)}"
        )
    nb, block_size, _ = kv.shape
    row_stride = kv.stride(1)
    blk_stride = kv.stride(0)
    if rope.stride(1) != row_stride // 2:
        raise RuntimeError(
            f"{name}: rope row stride {rope.stride(1)} must be half the kv row "
            f"stride {row_stride} -- both halves live in one record"
        )
    if blk_stride % row_stride:
        raise RuntimeError(
            f"{name} block stride {blk_stride} must be a multiple of the record "
            f"stride {row_stride}"
        )
    if block_size & (block_size - 1):
        raise RuntimeError(f"{name} page size {block_size} must be a power of two")
    rows = (nb - 1) * (blk_stride // row_stride) + block_size
    if rows > 2**31 - 1:
        raise RuntimeError(
            f"{name} spans {rows:,} descriptor rows, past the int32 bound"
        )
    return rows, blk_stride // row_stride, block_size, nb * block_size


@functools.cache
def _v4_decode_partials(t_cap, kv_splits, h_padded, d, device_index):
    """Persistent fp32 partials for the split-K decode path.

    maxsize=None so entries are never evicted and the storage is never freed:
    a captured CUDA graph holds these addresses, and handing that memory back
    to the allocator would let a later allocation land underneath a replay.
    Sized at a power-of-two token count so the number of distinct buffers stays
    small; callers slice to their actual T.
    """
    dev = torch.device("cuda", device_index)
    m = torch.empty((t_cap, kv_splits, h_padded), dtype=torch.float32, device=dev)
    l = torch.empty_like(m)
    acc = torch.empty((t_cap, kv_splits, h_padded, d), dtype=torch.float32, device=dev)
    return m, l, acc


def _pa_decode_sparse_gfx1250_fp8(
    q: torch.Tensor,
    unified_kv: torch.Tensor,
    unified_kv_rope: torch.Tensor,
    kv_indices: torch.Tensor,
    kv_indptr: torch.Tensor,
    attn_sink: torch.Tensor,
    softmax_scale: float,
    q_rope: torch.Tensor | None = None,
    kv_scales: torch.Tensor | None = None,
    extra_cache: torch.Tensor | None = None,
    extra_cache_rope: torch.Tensor | None = None,
    extra_indices: torch.Tensor | None = None,
    extra_indptr: torch.Tensor | None = None,
    num_warps: int | None = None,
    ctas_h: int = 1,
    q_tdm: bool = True,
    block_h: int | None = None,
    kv_splits: int | None = None,
    has_invalid: bool = True,
    skip_reduce: bool = False,
    use_lru_cache_partials: bool = False,
    out: torch.Tensor | None = None,
):
    """gfx1250 fp8 driver. Q / KV layouts: "gfx1250, fp8" in the Formats
    section of ``pa_decode_sparse``.

    Returns ``[N, H, 512]`` bf16 (MLA's V is the whole row: dequantized NoPE
    followed by the bf16 RoPE half).

    TODO: the fused output epilogue (``inv_rope_positions`` /
    ``inv_rope_cos_sin_cache`` / ``out_mxfp8``) -- inverse GPT-J RoPE over the
    trailing rope lanes and the optional MXFP8 quant. ``pa_decode_sparse``
    rejects those arguments on this path until it exists. See
    ``_pa_decode_sparse_gfx950_gluon`` for the reference implementation and
    ``test_pa_decode_sparse_inv_rope_mxfp8_epilogue`` for the expected numerics.
    """
    if DEVICE_ARCH != "gfx1250":
        raise RuntimeError(
            f"the DSv4 2buff packed-fp8 path is gfx1250-only, got {DEVICE_ARCH}"
        )
    if kv_scales is not None:
        raise RuntimeError(
            "kv_scales must be None on the 2buff path: the E8M0 group scales are "
            "inline in bytes [448, 462) of every packed row"
        )
    assert unified_kv.dtype in _V4_PACKED_FP8_DTYPES, (
        f"unified_kv must be {_V4_PACKED_FP8_DTYPES} (OCP e4m3 / raw bytes), "
        f"got {unified_kv.dtype}"
    )
    assert unified_kv_rope.dtype == torch.bfloat16
    assert unified_kv_rope.shape[-1] == _V4_DIM_ROPE, (
        f"unified_kv_rope must be [..., {_V4_DIM_ROPE}] bf16, got "
        f"{tuple(unified_kv_rope.shape)}"
    )
    main_rows, main_blk_rows, main_block_size, main_slots = _v4_2buff_geometry(
        unified_kv, unified_kv_rope, "unified_kv"
    )

    has_extra = extra_cache is not None
    if has_extra:
        assert extra_cache_rope is not None, "extra_cache needs extra_cache_rope"
        assert extra_indices is not None and extra_indptr is not None
        assert extra_cache.dtype in _V4_PACKED_FP8_DTYPES
        assert extra_cache_rope.dtype == torch.bfloat16
        (
            extra_rows,
            extra_blk_rows,
            extra_block_size,
            extra_slots,
        ) = _v4_2buff_geometry(extra_cache, extra_cache_rope, "extra_cache")
        # One pair of row strides (kv_stride_n / kvr_stride_n) serves both
        # streams' descriptors, so the extra stream must share the main's.
        if extra_cache.stride(-2) != unified_kv.stride(-2) or extra_cache_rope.stride(
            -2
        ) != unified_kv_rope.stride(-2):
            raise RuntimeError(
                "extra_cache must have the same row strides as unified_kv: "
                f"{extra_cache.stride(-2)}/{extra_cache_rope.stride(-2)} vs "
                f"{unified_kv.stride(-2)}/{unified_kv_rope.stride(-2)}"
            )
        assert extra_indices.dtype == torch.int32 and extra_indices.is_contiguous()
        assert extra_indptr.dtype == torch.int32 and extra_indptr.is_contiguous()
    else:
        # Aliases, not real work: the kernel never reads them under HAS_EXTRA
        # = False, but the launch still has to typecheck.
        extra_cache = unified_kv
        extra_cache_rope = unified_kv_rope
        extra_indices = kv_indices
        extra_indptr = kv_indptr
        extra_rows, extra_blk_rows, extra_block_size = main_rows, 1, 1
        extra_slots = main_slots

    if q_rope is None:
        raise RuntimeError(
            "the 2buff path needs the packed fp8 Q: pass q_rope alongside q"
        )
    T, H, D = q.shape
    assert D == _V4_DIM_QK, f"2buff path is fixed to D={_V4_DIM_QK}, got {D}"
    assert (
        q.dtype in _V4_PACKED_FP8_DTYPES
    ), f"q must be the packed fp8 [N, H, {_V4_DIM_QK}] tensor, got {q.dtype}"
    assert q_rope.shape == (T, H, _V4_DIM_ROPE) and q_rope.dtype == torch.bfloat16
    assert q_rope.is_contiguous()

    assert kv_indices.dtype == torch.int32 and kv_indices.is_contiguous()
    assert kv_indptr.dtype == torch.int32 and kv_indptr.is_contiguous()

    _LOGGER.info(
        f"PA_DECODE_SPARSE_V4_2BUFF T={T} H={H} D={D} "
        f"total_indices={kv_indices.shape[0]}"
    )

    out = _check_out(out, q, torch.bfloat16)

    # The kernel reads the packed pool as raw bytes: the E8M0 scale bytes live
    # inside the rows, and an fp8-typed tile would reinterpret them as e4m3
    # (byte 0x7F == scale 2^0 is an e4m3 NaN). The NoPE bytes are bitcast back
    # to e4m3 in-kernel.
    kv_u8 = unified_kv.view(torch.uint8)
    extra_u8 = extra_cache.view(torch.uint8)
    q_u8 = q.view(torch.uint8)

    # Same BLOCK_H / BLOCK_K / warp heuristics as the bf16 gluon path.
    if block_h is None:
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
            # [TEMPORARY] pinned to 32. BLOCK_H=16 is miscompiled by upstream
            # Triton on the single-stream KV_SPLITS==1 path -- the epilogue's
            # tdm.async_store reads an LDS tile the wave has not finished
            # writing, giving ~1e38 garbage (plan section 27 / 30.4). BLOCK_H=32
            # is clean on both compilers, so pin it until the compiler fix
            # lands, then restore: block_h = 32 if T >= 256 else 16
            # The other H that land on 16 are covered after kv_splits is known.
            block_h = 32
        else:
            block_h = triton.next_power_of_2(H)
    else:
        block_h = triton.next_power_of_2(block_h)
    block_h = max(block_h, 16)

    n_head_blocks = triton.cdiv(H, block_h)
    h_padded = n_head_blocks * block_h
    block_d = D

    block_k = 16
    waves_per_eu = 1
    if block_h == 128:
        # A 64-row KV tile: now that the Q operand is streamed from LDS one K
        # step at a time instead of held in registers, the wider tile no longer
        # overflows the register file, and the fatter iteration amortises the
        # accumulator rescale and the barriers over twice the work -- 64.4us vs
        # 69.9us at kv_len=384, T=512, H=128.
        block_k = 64
        attn_num_warps = 8
        max_num_wg = 256
        # The bf16 path asks for 2 waves/EU here; the dequant pushes this
        # kernel's register demand past that, so requesting it only costs
        # spills (measured ~4% on T=512, H=128, kv_len=384).
        waves_per_eu = 1
    elif block_h == 64:
        attn_num_warps = 4
        max_num_wg = 256
    elif block_h == 32:
        attn_num_warps = 2
        max_num_wg = 512
    else:
        attn_num_warps = 1
        max_num_wg = 1024
    if ctas_h > 1:
        # block_h is the CLUSTER tile: each CTA owns block_h // ctas_h heads,
        # and warps follow the per-CTA tile (16 heads -> 1 warp). The per-CTA
        # tile may not fall below the 16-row WMMA instruction tile -- without
        # this the compiler dies with a bare "PassManager::run failed".
        assert not (block_h % ctas_h), f"block_h {block_h} % ctas_h {ctas_h}"
        assert block_h // ctas_h >= 16, (
            f"ctas_h={ctas_h} would give {block_h // ctas_h} heads per CTA for "
            f"block_h={block_h}; the WMMA tile is 16 rows, so ctas_h must be "
            f"<= block_h // 16 ({block_h // 16})"
        )
        attn_num_warps = max(1, (block_h // ctas_h) // 16)
    if num_warps is not None:
        attn_num_warps = num_warps
    # Q_IN_VGPR: keep the loop-invariant Q tile resident instead of re-reading
    # it from LDS every K step. Gated on the warp count because that sets the
    # VGPR budget -- num_warps 8 caps at 512/SIMD and the BLOCK_H=128 kernel
    # already measures ~494, while num_warps <= 4 caps at 1024 against 737
    # (BLOCK_H=16) / 604 (BLOCK_H=32), leaving room for the ~64 VGPR Q tile.
    reduce_num_warps = 1
    reduce_waves_per_eu = 4

    kv_splits_requested = kv_splits
    if kv_splits is None:
        # PER TOKEN: the split is per token, so the ceiling is the tile count
        # ONE token has. kv_indices.shape[0] is every token's indices together,
        # so the ceiling never bound and splits were made with no work in them
        # -- and a dead split still costs the reduce a slab row.
        total_idx = kv_indices.shape[0] + (extra_indices.shape[0] if has_extra else 0)
        avg_kv_len = max(1, total_idx // max(1, T))
        max_kv_splits = max(1, triton.cdiv(avg_kv_len, block_k))
        kv_splits = max(1, max_num_wg // max(1, T * n_head_blocks))
        kv_splits = min(max_kv_splits, kv_splits)
        # DOWN to a power of two: rounding up puts back what the ceiling removed.
        kv_splits = 1 << (max(1, kv_splits).bit_length() - 1)
        # CAP. Occupancy is only half the story: every split also writes and
        # then re-reads T * H_padded * D fp32 of partials, so past the point
        # where the attention kernel stops speeding up, the splits cost more
        # than they buy -- and they slow the ATTENTION down too, not just the
        # reduce. Measured, main + reduce, kv_len 384:
        #
        #   H=32  T=16   32 splits 17.26us -> 8 splits 13.63us   (main 10.89
        #                -> 10.41, reduce 6.37 -> 3.23)
        #   H=128 T=16   16 splits 52.15us -> 8 splits 43.54us
        #   H=32/128, T >= 64: the heuristic already picks <= 4, unchanged.
        #
        # Only small T can reach past 8, because the occupancy term is
        # max_num_wg // (T * n_head_blocks).
        kv_splits = min(kv_splits, 8)

    _lds_budget = arch_info._LDS_CAP_BYTES.get(DEVICE_ARCH)
    _lds_cap = max(1, _lds_budget // (block_d * 4))
    kv_splits = min(kv_splits, 1 << (_lds_cap.bit_length() - 1))
    # KV_SPLITS == 1 takes a different output path: the accumulators are
    # reassembled in LDS and pushed out with tdm.async_store, rather than the
    # reduce writing `out` from the partials. That path is CORRECT -- an
    # earlier comment here called it a race, which it is not. It is
    # miscompiled by upstream Triton 3.8.0 at BLOCK_H=16 only (the
    # `s_wait_dscnt` before `tensor_store_from_lds` is dropped). The pin above
    # only covers 32 <= H < 64; H < 32 (16 at TP8) and H in [64, 128) at T < 32
    # still pick BLOCK_H=16, and at large T the heuristic then gives one split.
    # Two splits move that output onto the reduce instead -- an empty second
    # split just exits. The internal build is correct at every BLOCK_H; drop
    # this with the pin once the compiler fix lands.
    if block_h == 16 and kv_splits == 1:
        if kv_splits_requested is not None:
            raise ValueError(
                "kv_splits=1 with BLOCK_H=16 hits a Triton 3.8.0 miscompile of "
                "the single-split output store on gfx1250; use kv_splits >= 2 "
                "or block_h >= 32"
            )
        kv_splits = 2
    if kv_splits > 8:
        reduce_num_warps = 4
        reduce_waves_per_eu = 1

    if kv_splits == 1:
        m_partial = l_partial = acc_partial = out  # unused inside the kernel
        mp_strides = (0, 0, 0)
        lp_strides = (0, 0, 0)
        ap_strides = (0, 0, 0, 0)
    else:
        # skip_reduce hands the partials back to the caller, who then owns
        # them; a shared buffer would be overwritten under it next call.
        _cache_ok = (
            use_lru_cache_partials and not skip_reduce and q.device.type == "cuda"
        )
        if _cache_ok:
            m_full, l_full, acc_full = _v4_decode_partials(
                triton.next_power_of_2(T), kv_splits, h_padded, D, q.device.index
            )
            m_partial = m_full[:T]
            l_partial = l_full[:T]
            acc_partial = acc_full[:T]
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

    # Route the fp32 acc partials through LDS + tdm.async_store instead of
    # buffer_store, but only at small T. Measured main-kernel time, kv_len 384:
    #
    #        H=32                      H=128
    #   T=16  10.31 -> 8.69  (-1.62)   38.42 -> 39.02 (+0.60)
    #   T=64  15.00 -> 15.08           80.93 -> 84.52 (+3.59)
    #   T=256 31.02 -> 30.36 (-0.66)  203.23 -> 207.45 (+4.22)
    #
    # It pays only where there is little else to overlap the staging with; at
    # larger T the LDS round trip costs more than the wider store buys.
    #
    # NEVER at BLOCK_H=16: upstream Triton 3.8.0 drops the `s_wait_dscnt`
    # before `tensor_store_from_lds`, so the TDM engine reads an LDS tile the
    # wave has not finished writing. That bug is not specific to the
    # KV_SPLITS==1 epilogue -- this store hits it too (4 op-test failures at
    # H=16 when it was unguarded). See plan section 27.
    _tdm_partials = T <= 16 and block_h != 16
    grid_attn = (T, n_head_blocks, kv_splits)
    gluon_pa_decode_sparse_fp8[grid_attn](
        q,
        q_u8,
        q_rope,
        unified_kv,
        kv_u8,
        unified_kv_rope,
        kv_indices,
        kv_indptr,
        extra_cache,
        extra_u8,
        extra_cache_rope,
        extra_indices,
        extra_indptr,
        m_partial,
        l_partial,
        acc_partial,
        attn_sink,
        out,
        main_rows,
        extra_rows,
        main_blk_rows,
        extra_blk_rows,
        main_slots,
        extra_slots,
        q.stride(0),
        q.stride(1),
        q_rope.stride(0),
        q_rope.stride(1),
        unified_kv.stride(-2),
        unified_kv_rope.stride(-2),
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
        NOPE_DIM=_V4_DIM_NOPE,
        ROPE_DIM=_V4_DIM_ROPE,
        GROUP_SIZE=_FP8_GROUP_SIZE,
        Q_IN_VGPR=(attn_num_warps <= 4),
        HAS_INVALID=bool(has_invalid),
        Q_TDM=bool(q_tdm),
        CTAS_H=ctas_h,
        MAIN_BLOCK_SIZE=main_block_size,
        EXTRA_BLOCK_SIZE=extra_block_size,
        HAS_EXTRA=has_extra,
        TDM_PARTIALS=_tdm_partials,
        num_warps=attn_num_warps,
        num_stages=2,
        waves_per_eu=waves_per_eu,
        num_ctas=ctas_h,
    )

    if kv_splits == 1:
        return out

    if skip_reduce:
        return acc_partial, m_partial, l_partial

    block_h_reduce = 1
    grid_reduce = (T, triton.cdiv(H, block_h_reduce))
    gluon_pa_decode_sparse_reduce[grid_reduce](
        m_partial,
        l_partial,
        acc_partial,
        attn_sink,
        kv_indptr,
        extra_indptr,
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
        # The reduce counts a token's live segments in TILES, and with two
        # streams that count is main_tiles + extra_tiles -- it has to see the
        # extra stream's indptr or it scrubs the wrong segments.
        HAS_EXTRA=has_extra,
        num_warps=reduce_num_warps,
        waves_per_eu=reduce_waves_per_eu,
    )
    return out
