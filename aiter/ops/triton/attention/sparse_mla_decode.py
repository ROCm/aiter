# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Sparse MLA decode (gfx950 gluon): ``kv_lora_rank`` latent + appended
decoupled rope, token-granular top-k gather.

Separated-rope sibling of ``pa_decode_sparse`` — both drive the same gluon
kernel, but the calling structures differ enough to warrant separate APIs:
DSv4 (rope inside the row, K == V, optional SWA + top-k two-loop, attention
sink) stays on ``pa_decode_sparse``; MLA-lineage models (single top-k segment,
no sink, QK over ``kv_lora_rank + qk_rope_head_dim``, V = the latent) use this.

Follows the ``aiter.mla.mla_decode_fwd`` calling convention as vLLM's
``ROCMAiterMLASparseImpl`` uses it, with two deviations: no head padding
(H < 16 runs natively at BLOCK_M = next_pow2(H)) and q stays bf16 (no q-quant
kernel; <1% accuracy effect).
"""

import math

import torch

from aiter.ops.triton._gluon_kernels.gfx950.attention.pa_decode_sparse import (
    _pa_decode_sparse as _pa_decode_sparse_gfx950,
)
from aiter.ops.triton._gluon_kernels.gfx950.attention.pa_decode_sparse import (
    _pa_decode_sparse_reduce as _pa_decode_sparse_reduce_gfx950,
)
from aiter.ops.triton.attention.pa_decode_sparse import (
    _as_int32_contiguous_1d,
)
from aiter.ops.triton.utils._triton import arch_info
from aiter.ops.triton.utils.common_utils import max_addressable_bytes
from aiter.ops.triton.utils.device_info import get_num_sms
from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER = AiterTritonLogger()

# buffer_load carries a signed 32-bit offset; caches past this span gather
# through 64-bit addresses (a production MLA pool crosses it at ~3.7M tokens).
MAX_BYTES = 2**31 - 1

# Tuned launch config (gfx950 / MI355). num_warps = BLOCK_K // 16 (warps tile
# the dot-N, MFMA N=16); GATHER_TW1=32 requests a whole 512 B token row per
# gather instruction.
_BLOCK_K = 64
_MFMA_K = 16
_GATHER_TW1 = 32
_LDS_PAD = 8


def _check_out(out, q, kv_lora_rank):
    """Caller-supplied output buffer, or a fresh one. A buffer wider than
    kv_lora_rank is accepted and written in its leading columns."""
    C, H = q.shape[0], q.shape[1]
    if out is None:
        return torch.empty(
            (C, H, kv_lora_rank), dtype=torch.bfloat16, device=q.device
        )
    assert out.shape[0] == C and out.shape[1] == H, (
        f"out shape {tuple(out.shape)} != [{C}, {H}, >= {kv_lora_rank}]"
    )
    assert out.shape[2] >= kv_lora_rank and out.stride(2) == 1
    assert out.dtype == torch.bfloat16 and out.device == q.device
    return out


def _infer_cache_format(kv, d_qk, kv_lora_rank, qk_rope_head_dim, kv_scale):
    """-> (fmt, cache, alt_ptr, scl_ptr, block_size, fp8_fnuz). Pointer roles
    follow the kernel's Seg contract (see its docstring)."""
    if kv.ndim == 4:  # [slots, 1, 1, R], the asm mla_decode_fwd view
        assert kv.shape[1] == 1 and kv.shape[2] == 1
        kv = kv.reshape(kv.shape[0], kv.shape[3])
    elif kv.ndim == 3 and kv.shape[2] == d_qk:
        # vLLM paged cache [nb, block_size, R]: indices are global slot ids and
        # rows are contiguous, so the flat view is stride-identical.
        assert kv.stride(2) == 1 and kv.stride(1) == kv.shape[2] * kv.stride(2)
        kv = kv.reshape(-1, d_qk)

    fp8_dtypes = (torch.uint8, torch.float8_e4m3fn, torch.float8_e4m3fnuz)
    if kv.ndim == 2 and kv.shape[1] == d_qk:
        if kv.dtype == torch.bfloat16:
            return "bf16", kv, kv, None, 1, False
        assert kv.dtype in fp8_dtypes, f"unsupported flat cache dtype {kv.dtype}"
        assert kv_scale is not None, (
            "flat fp8 cache needs the per-tensor kv_scale (layer._k_scale)"
        )
        assert kv_scale.dtype == torch.float32
        u8 = kv.view(torch.uint8)
        fnuz = kv.dtype == torch.float8_e4m3fnuz
        return "tensor", u8, u8, kv_scale.reshape(1), 1, fnuz
    if (
        kv.ndim == 3
        and kv.element_size() == 1
        and kv.shape[2] == kv_lora_rank + 4 * (kv_lora_rank // 128) + 2 * qk_rope_head_dim
    ):
        # vLLM fp8_ds_mla: 512 fp8 | 4 f32 per-128 scales | 64 bf16 rope = 656 B
        u8 = kv if kv.dtype == torch.uint8 else kv.view(torch.uint8)
        assert u8.stride(2) == 1 and u8.stride(1) == u8.shape[2], (
            "fp8_ds_mla rows must be contiguous 656-byte records"
        )
        return (
            "dsmla",
            u8,
            u8.view(torch.bfloat16),
            u8.view(torch.float32),
            u8.shape[1],
            False,
        )
    raise ValueError(f"unrecognized sparse-MLA kv cache: {tuple(kv.shape)} {kv.dtype}")


def _mla_num_splits(num_queries: int, heads_blocks: int, avg_topk: float) -> int:
    """Split-K count for the sparse-MLA decode.

    Separate from the DSv4 policy in _decode_num_splits_occ, which is tuned for a
    two-segment SWA + top-k launch and over-splits here: at C=1 topk=2048 it asks
    for one split per tile, so 32 programs each pay the full prologue and the
    reduce walks 32 partials, costing 1.23-1.28x.

    Below one workgroup per CU, split to fill the machine but never past 8. Past
    that, splits stop buying parallelism and start paying prologue and reduce
    traffic; 8 is optimal or within 10% everywhere measured. At or above one
    workgroup per CU the launch is already full, so a split only buys the second
    occupancy slot and needs ~16 tiles to earn its extra partial round trip. At
    C=256 that decides one split versus two, and topk=512 wants none.
    """
    num_sms = get_num_sms()
    base_wg = max(1, num_queries * heads_blocks)
    cta_cap = max(1, (2 * num_sms) // base_wg)
    tiles = max(1, math.ceil(avg_topk / _BLOCK_K))
    if base_wg >= num_sms:
        return max(1, min(cta_cap, tiles // 16))
    return max(1, min(cta_cap, tiles, 8))


_E4M3_MAX = 448.0


def _resolve_dot_precision(dot_precision: str, fmt: str, fp8_fnuz: bool) -> bool:
    """True for the fp8 matrix-core path.

    Every rejection below is the same constraint: the fp8 path needs the cache
    scale to be one positive scalar, because that is what folds outside the tile
    loop (K-side into qk_scale, V-side onto the accumulator). A scale that varies
    along a dot's contraction axis cannot be folded out at all.
    """
    if dot_precision not in ("bf16", "fp8"):
        raise ValueError(
            f"dot_precision must be 'bf16' or 'fp8', got {dot_precision!r}"
        )
    if dot_precision == "bf16":
        return False
    if fmt == "dsmla":
        raise ValueError(
            "dot_precision='fp8' does not support the fp8_ds_mla cache. Its "
            "scale varies per 128 elements along the head dim, the QK "
            "contraction axis, so it cannot fold outside the tile loop, and "
            "scaled MFMA cannot express the PV side either (one scale per 32 "
            "contraction elements, where this needs one per token). "
            "Use dot_precision='bf16'."
        )
    if fmt == "bf16":
        raise ValueError(
            "dot_precision='fp8' needs an fp8 cache; this one is bf16, and "
            "quantizing the pool per call would cost more than the dots save."
        )
    if fp8_fnuz:
        raise ValueError(
            "dot_precision='fp8' is OCP e4m3 only; this cache is fnuz, a "
            "different encoding from what the matrix core reads."
        )
    return True


def _lds_pad_for(fp8_dots: bool) -> int:
    """kv_smem row padding, in elements of the staged type.

    The pad shifts which banks a transposed K read lands on, which is a byte
    property, so an fp8 plane needs twice the elements to move the row pitch by
    the same bytes. Carrying bf16's value over to fp8 costs 1.08x at C=64
    topk=2048; 16, 32 and 64 all land at 0.86-0.89x.
    """
    return _LDS_PAD * 2 if fp8_dots else _LDS_PAD


def sparse_mla_decode_fwd(
    q: torch.Tensor,
    kv_buffer: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    softmax_scale: float,
    kv_scale: torch.Tensor | None = None,
    kv_lora_rank: int = 512,
    qk_rope_head_dim: int = 64,
    kv_splits: int | None = None,
    skip_reduce: bool = False,
    has_invalid: bool = False,
    dot_precision: str = "bf16",
    q_scale: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Sparse (top-k gathered) MLA decode, separated-rope geometry.

    Args:
        q: ``[C, H, kv_lora_rank + qk_rope_head_dim]`` bf16 decode queries.
        kv_buffer: the KV pool — ``[nb, block, R]`` / ``[slots, 1, 1, R]`` /
            ``[slots, R]`` in bf16, the same shapes in fp8 (+ scalar
            ``kv_scale``), or ``[nb, block, 656]`` uint8 (vLLM ``fp8_ds_mla``).
        kv_indptr: ``[C + 1]`` int32 prefix sum of per-query index counts.
        kv_indices: flat int32 GLOBAL slot ids into the pool.
        softmax_scale: the layer's softmax scale.
        kv_scale: ``[1]`` f32 per-tensor cache scale (``layer._k_scale``);
            required for the flat fp8 format.
        kv_splits: split-K override; default follows the occupancy policy.
        skip_reduce: with split-K active, return ``(part_acc, part_m, part_l)``
            instead of launching the combine.
        has_invalid: index stream carries -1 sentinels (masked out). Default
            False: vLLM's index converter emits none (it maps -1 to slot 0).
        dot_precision: what the QK and PV matrix-core ops run in.

            "bf16" (default): the KV tile is dequantized to bf16 on its way
                into LDS and both dots are bf16. Works with every cache format.
            "fp8": the cache's own code points go to the fp8 matrix core with no
                dequant, and the per-tensor scale folds outside the tile loop.
                0.75-0.96x of "bf16", but q and p both round-trip through e4m3,
                so accuracy lands at roughly the asm kernel's level. Needs the
                flat per-tensor OCP fp8 cache; see _resolve_dot_precision.

            q is adapted to the choice. bf16 q is quantized here when "fp8" is
            asked for, fp8 q is passed straight through, and fp8 q under "bf16"
            dots is widened in-kernel, which is exact.
        q_scale: scalar f32. Required when q arrives already fp8 (the scale it
            was quantized with; the aiter asm convention, where vLLM passes
            the layer's calibrated ``_q_scale``). Optional when q is bf16 and
            dot_precision="fp8": given, it is used as the static quantization
            scale, matching production's scaled_fp8_quant(q, layer._q_scale);
            omitted, a per-tensor amax is taken here.
        out: optional ``[C, H, >= kv_lora_rank]`` bf16 destination.

    Returns:
        ``[C, H, kv_lora_rank]`` bf16 attention output (the latent V).
    """
    assert q.ndim == 3, f"expected q=[b,h,d], got {q.shape}"
    assert arch_info.get_arch() == "gfx950", "sparse_mla_decode_fwd is gfx950-only"
    q_is_fp8 = q.dtype == torch.float8_e4m3fn
    if q.dtype not in (torch.bfloat16, torch.float8_e4m3fn):
        raise ValueError(
            f"q must be bf16 or float8_e4m3fn, got {q.dtype}"
            + (" (fnuz is a different encoding from what the matrix core reads)"
               if "fnuz" in str(q.dtype) else "")
        )
    if q_is_fp8:
        # Caller-quantized q, the asm calling convention: one scaled_fp8_quant
        # over [C, H*d_qk] with the layer's scale. The kernel folds the scale
        # into qk_scale, so nothing per-tile changes.
        if q_scale is None:
            raise ValueError("fp8 q needs q_scale (the scale it was quantized with)")
        if q_scale.numel() != 1:
            raise ValueError(f"q_scale must be a scalar, got {tuple(q_scale.shape)}")
        q_scale = q_scale.reshape(1).to(torch.float32).contiguous()
    num_queries, num_heads, d_qk = q.shape
    assert d_qk == kv_lora_rank + qk_rope_head_dim, (
        f"q last dim {d_qk} != {kv_lora_rank} + {qk_rope_head_dim}"
    )
    _LOGGER.info(
        f"SPARSE_MLA_DECODE C={num_queries} H={num_heads} d_qk={d_qk} "
        f"nnz={kv_indices.shape[0]}"
    )

    fmt, cache, alt, scl, block_size, fp8_fnuz = _infer_cache_format(
        kv_buffer, d_qk, kv_lora_rank, qk_rope_head_dim, kv_scale
    )
    fp8_dots = _resolve_dot_precision(dot_precision, fmt, fp8_fnuz)
    if fp8_dots and not q_is_fp8:
        # Quantize the way production does, one scaled_fp8_quant over
        # [C, H*d_qk]. A caller-supplied q_scale is the layer's calibrated
        # static scale; without one, take a per-tensor amax.
        if q_scale is None:
            q_scale = (
                (q.detach().float().abs().amax() / _E4M3_MAX)
                .clamp_min(torch.finfo(torch.float32).tiny)
                .reshape(1)
            )
        q = (
            (q.float() / q_scale).clamp(-_E4M3_MAX, _E4M3_MAX)
            .to(torch.float8_e4m3fn)
            .contiguous()
        )
        q_is_fp8 = True
    elif q_scale is not None and not q_is_fp8:
        raise ValueError(
            "q_scale was given but q is bf16 and dot_precision='bf16', so "
            "nothing would use it"
        )
    kv_indices = _as_int32_contiguous_1d(kv_indices)
    kv_indptr = _as_int32_contiguous_1d(kv_indptr)
    assert kv_indptr.numel() == num_queries + 1
    # No attention sink in MLA models; the kernel still wants a live pointer
    # for the compile-time-elided argument slot.
    attn_sink = torch.empty(1, device=q.device, dtype=torch.float32)

    # H < 16 runs natively at BLOCK_M = next_pow2(H) instead of padding heads.
    block_m = 16 if num_heads >= 16 else max(8, 1 << (num_heads - 1).bit_length())
    num_warps = _BLOCK_K // 16

    num_rows = cache.shape[0] * block_size if cache.ndim >= 2 else cache.shape[0]
    avg_topk = kv_indices.numel() / max(1, num_queries)

    # Alignment hint for cs0 so row gathers can vectorize.
    cs0_align = 1
    for a in (16, 8, 4, 2):
        if int(cache.stride(0)) % a == 0:
            cs0_align = a
            break

    use_buffer_load = max_addressable_bytes(cache) < MAX_BYTES
    idx_use_buffer_load = max_addressable_bytes(kv_indices) < MAX_BYTES

    head_aligned = num_heads % block_m == 0
    heads_blocks = (num_heads + block_m - 1) // block_m
    out = _check_out(out, q, kv_lora_rank)

    if kv_splits is not None:
        num_splits = max(1, int(kv_splits))
    else:
        num_splits = _mla_num_splits(num_queries, heads_blocks, avg_topk)

    if num_splits > 1:
        part_m = torch.empty(
            (num_queries, num_splits, num_heads), dtype=torch.float32, device=q.device
        )
        part_l = torch.empty_like(part_m)
        # bf16 partials halve the split-K HBM traffic; skip_reduce hands the
        # partials back to the caller and keeps f32.
        part_acc = torch.empty(
            (num_queries, num_splits, num_heads, kv_lora_rank),
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

    # Dequant chunking: split the axis that keeps per-lane register repeats
    # (rows at GATHER_TW1=32); waves_per_eu=2 caps the allocator at 256 VGPRs
    # for the second occupancy slot, except when the launch cannot fill it.
    col_reps = kv_lora_rank // (_GATHER_TW1 * 16)
    chunk_axis = 1 if col_reps >= 4 else 0
    nope_chunk = max(1, _BLOCK_K // 4) if chunk_axis == 0 else min(128, kv_lora_rank)
    waves_per_eu = 2
    if use_buffer_load and num_queries * heads_blocks * num_splits <= get_num_sms():
        waves_per_eu = 1

    grid = (num_queries, num_splits, heads_blocks)
    _pa_decode_sparse_gfx950[grid](
        q,
        cache,
        alt,
        kv_indices,
        kv_indptr,
        cache,       # extra_* segment: unread placeholders (HAS_EXTRA=False)
        alt,
        kv_indices,
        kv_indptr,
        attn_sink,
        out,
        part_m,
        part_l,
        part_acc,
        scl,         # f32 side-channel: k_scale ("tensor") / f32 view ("dsmla")
        scl,
        float(softmax_scale),
        q.stride(0),
        q.stride(1),
        out.stride(0),
        out.stride(1),
        cache.stride(0),
        cache.stride(0),
        num_rows,
        num_rows,
        pm_stride0,
        pm_stride_s,
        pa_stride0,
        pa_stride_s,
        pa_stride_h,
        num_heads,
        HAS_EXTRA=False,
        HAS_SINK=False,
        MAIN_FMT=fmt,
        EXTRA_FMT=fmt,
        MAIN_BLOCK_SIZE=block_size,
        EXTRA_BLOCK_SIZE=block_size,
        CS0_ALIGN=cs0_align,
        NOPE_DIM=kv_lora_rank,
        ROPE_DIM=qk_rope_head_dim,
        HEAD_SIZE=kv_lora_rank,
        ROPE_SEPARATE=True,
        BLOCK_M=block_m,
        BLOCK_K=_BLOCK_K,
        NUM_SPLITS=num_splits,
        HEAD_ALIGNED=head_aligned,
        MFMA_K=_MFMA_K,
        GATHER_TW1=_GATHER_TW1,
        LDS_PAD=_lds_pad_for(fp8_dots),
        NOPE_CHUNK=nope_chunk,
        CHUNK_AXIS=chunk_axis,
        PART_STORE_CACHE="",
        UNI_TILE=True,
        GRID_ORDER="qsh",
        MAIN_SPLITS=num_splits,
        ADAPTIVE_SPLITS=num_splits > 1,
        ASM_DEQ=False,
        MAIN_USE_BUFFER_LOAD=use_buffer_load,
        EXTRA_USE_BUFFER_LOAD=use_buffer_load,
        IDX_BUFFER_LOAD=idx_use_buffer_load,
        HAS_INVALID=has_invalid,
        FP8_FNUZ=fp8_fnuz,
        FP8_MFMA=fp8_dots,
        q_scl_ptr=q_scale,
        Q_FP8=q_is_fp8,
        num_warps=num_warps,
        waves_per_eu=waves_per_eu,
    )

    if num_splits == 1:
        return out
    if skip_reduce:
        return part_acc, part_m, part_l

    # One head per reduce workgroup: the combine is pure bandwidth, so size the
    # grid for coverage, not for the attention kernel's tile.
    rgrid = (num_queries, num_heads)
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
        HAS_SINK=False,
        HEAD_SIZE=kv_lora_rank,
        BLOCK_M=1,
        NUM_SPLITS=num_splits,
        HEAD_ALIGNED=True,
        ADAPTIVE_SPLITS=num_splits > 1,
        num_warps=1,
    )
    return out
