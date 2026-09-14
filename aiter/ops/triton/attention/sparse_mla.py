# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import math

import torch

from aiter.ops.triton._gluon_kernels.gfx950.attention.sparse_mla import (
    _sparse_mla as _sparse_mla_gfx950,
)
from aiter.ops.triton._gluon_kernels.gfx950.attention.sparse_mla import (
    _sparse_mla_reduce as _sparse_mla_reduce_gfx950,
)
from aiter.ops.triton.attention.pa_decode_sparse import (
    _as_int32_contiguous_1d,
)
from aiter.ops.triton.utils._triton import arch_info
from aiter.ops.triton.utils.common_utils import max_addressable_bytes
from aiter.ops.triton.utils.device_info import get_num_sms
from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER = AiterTritonLogger()


# Tuned launch config (gfx950 / MI355). num_warps = BLOCK_K // 16 (warps tile
# the dot-N, MFMA N=16); GATHER_TW1=32 requests a whole 512 B token row per
# gather instruction.
_BLOCK_K = 64
_MFMA_K = 16
_GATHER_TW1 = 32


def _check_out(out, q, kv_lora_rank):
    """Caller-supplied output buffer, or a fresh one. A buffer wider than
    kv_lora_rank is accepted and written in its leading columns."""
    C, H = q.shape[0], q.shape[1]
    if out is None:
        return torch.empty((C, H, kv_lora_rank), dtype=torch.bfloat16, device=q.device)
    assert (
        out.shape[0] == C and out.shape[1] == H
    ), f"out shape {tuple(out.shape)} != [{C}, {H}, >= {kv_lora_rank}]"
    assert out.shape[2] >= kv_lora_rank and out.stride(2) == 1
    assert out.dtype == torch.bfloat16 and out.device == q.device
    return out


def _infer_cache_format(kv, d_qk, kv_lora_rank, qk_rope_head_dim, kv_scale):
    """-> (fmt, cache, alt_ptr, scl_ptr, block_size). Pointer roles
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
            return "bf16", kv, kv, None, 1
        assert kv.dtype in fp8_dtypes, f"unsupported flat cache dtype {kv.dtype}"
        assert (
            kv_scale is not None
        ), "flat fp8 cache needs the per-tensor kv_scale (layer._k_scale)"
        assert kv_scale.dtype == torch.float32
        u8 = kv.view(torch.uint8)
        # gfx950 reads OCP e4m3 natively; fnuz is the gfx942 encoding.
        assert (
            kv.dtype != torch.float8_e4m3fnuz
        ), "gfx950 reads OCP e4m3; float8_e4m3fnuz is the gfx942 encoding"
        return "fp8_scalar", u8, u8, kv_scale.reshape(1), 1
    if (
        kv.ndim == 3
        and kv.element_size() == 1
        and kv.shape[2]
        == kv_lora_rank + 4 * (kv_lora_rank // 128) + 2 * qk_rope_head_dim
    ):
        # fp8_dsv32_mla: 512 fp8 | 4 f32 per-128 scales | 64 bf16 rope = 656 B.
        # This is vLLM's fp8_ds_mla on V3.2 / Kimi-K3; the same vLLM name also
        # covers V4's 584 B layout (fp8_dsv4_mla), hence the explicit generation.
        u8 = kv if kv.dtype == torch.uint8 else kv.view(torch.uint8)
        assert (
            u8.stride(2) == 1 and u8.stride(1) == u8.shape[2]
        ), "fp8_dsv32_mla rows must be contiguous 656-byte records"
        return (
            "fp8_dsv32_mla",
            u8,
            u8.view(torch.bfloat16),
            u8.view(torch.float32),
            u8.shape[1],
        )
    raise ValueError(f"unrecognized sparse-MLA kv cache: {tuple(kv.shape)} {kv.dtype}")


def _mla_num_splits(num_queries: int, heads_blocks: int, avg_topk: float) -> int:
    """Split-K count for the sparse-MLA decode.

    Below one workgroup per CU, split to fill the machine but never past 8.
    """
    num_sms = get_num_sms()
    base_wg = max(1, num_queries * heads_blocks)
    cta_cap = max(1, (2 * num_sms) // base_wg)
    tiles = max(1, math.ceil(avg_topk / _BLOCK_K))
    if base_wg >= num_sms:
        return max(1, min(cta_cap, tiles // 16))
    return max(1, min(cta_cap, tiles, 8))


# Direct-to-LDS tile pipeline; wins wherever the launch gives a CU more than one
# workgroup to hide the copy behind.
_ASYNC_LDS_DEFAULT = True
_ASYNC_ROPE_VEC = 16  # bytes/lane in the rope copy


def _async_launch_config(
    fp8_dots: bool,
    has_invalid: bool,
    num_queries: int,
    heads_blocks: int,
    num_splits: int,
    avg_topk: float,
    use_buffer_load: bool,
    uni_tile: bool = True,
    has_extra: bool = False,
) -> tuple[bool, int, int]:
    """-> (ASYNC_LDS, BLOCK_K, waves_per_eu) for this launch.

    BLOCK_K follows what the grid supplies, not the token count: given enough
    workgroups to fill four waves/SIMD (prefill) the small tile takes that
    occupancy; otherwise (decode, where split-K caps the grid near two
    workgroups/CU) the large tile wins instead, by halving the cross-warp softmax
    exchange rate per token.
    """
    enabled = _ASYNC_LDS_DEFAULT
    enabled = enabled and fp8_dots and uni_tile and not has_invalid and not has_extra
    workgroups = num_queries * heads_blocks * max(1, num_splits)
    num_sms = get_num_sms()
    if enabled and workgroups >= 4 * num_sms:
        return True, 64, 4
    waves_per_eu = 2
    if use_buffer_load and workgroups <= num_sms:
        waves_per_eu = 1
    return enabled, (128 if enabled else _BLOCK_K), waves_per_eu


def _resolve_dot_precision(dot_precision: str, fmt: str) -> bool:
    if dot_precision not in ("bf16", "fp8"):
        raise ValueError(
            f"dot_precision must be 'bf16' or 'fp8', got {dot_precision!r}"
        )
    if dot_precision == "bf16":
        return False
    if fmt == "fp8_dsv32_mla":
        raise ValueError(
            "dot_precision='fp8' does not support the fp8_dsv32_mla cache."
            "Use dot_precision='bf16'."
        )
    if fmt == "bf16":
        raise ValueError("dot_precision='fp8' needs an fp8 cache.")
    return True


def _is_packed_mla_record(q, kv, kv_lora_rank):
    """True for the dsv3.2 / Kimi-K3 record: rank fp8 | f32 per-128 | bf16 rope.

    The rope width is d_qk - kv_lora_rank (what the MLA geometry assert
    enforces). dsv4's 584 B record does not satisfy the formula for any rope
    width, so the two packed layouts are told apart by size alone. Any one-byte
    dtype counts, matching _infer_cache_format.
    """
    if kv.ndim != 3 or kv.element_size() != 1:
        return False
    rope = q.shape[-1] - kv_lora_rank
    return (
        rope > 0 and kv.shape[2] == kv_lora_rank + 4 * (kv_lora_rank // 128) + 2 * rope
    )


def _is_paged_cache(q, kv, kv_lora_rank, kv_scale=None):
    """True for a cache only the paged driver reads: dsv4's 584 B record, a bf16
    block pool, or the uniform fp8 pool (fp8_g64: a 2-D pool with a per-64
    [pages, D // 64] scale vector). [nb, block, d_qk] is the flat MLA
    pool and a 2-D pool with a scalar kv_scale is fp8_scalar; neither is
    this."""
    if kv.ndim == 2:
        return kv_scale is not None and kv_scale.numel() != 1
    if kv.ndim != 3 or kv.shape[2] == q.shape[-1]:
        return False
    return not _is_packed_mla_record(q, kv, kv_lora_rank)


def _forward_paged(
    q,
    kv,
    kv_indptr,
    kv_indices,
    softmax_scale,
    kv_scale,
    kv_splits,
    skip_reduce,
    has_invalid,
    dot_precision,
    q_scale,
    out,
    return_lse,
    attn_sink,
    extra_kv,
    extra_indptr,
    extra_indices,
):
    """dsv4 and the SWA+top-k two-loop, until the two launchers merge."""
    from aiter.ops.triton.attention.pa_decode_sparse import pa_decode_sparse

    unsupported = [
        name
        for name, asked in (
            ("dot_precision='fp8'", dot_precision != "bf16"),
            ("q_scale", q_scale is not None),
            ("return_lse", return_lse),
        )
        if asked
    ]
    if unsupported:
        raise ValueError(
            f"{', '.join(unsupported)} is not supported for dsv4 / two-loop "
            "caches yet."
        )
    res = pa_decode_sparse(
        q,
        kv,
        kv_indices,
        kv_indptr,
        attn_sink,
        softmax_scale,
        kv_scales=kv_scale,
        kv_splits=kv_splits,
        has_invalid=has_invalid,
        skip_reduce=skip_reduce,
        extra_cache=extra_kv,
        extra_indices=extra_indices,
        extra_indptr=extra_indptr,
        out=out,
    )
    return res if isinstance(res, tuple) else (res, None)


def sparse_mla_fwd(
    q: torch.Tensor,
    kv_buffer: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    softmax_scale: float,
    *,
    kv_scale: torch.Tensor | None = None,
    kv_lora_rank: int = 512,
    qk_rope_head_dim: int = 64,
    kv_splits: int | None = None,
    skip_reduce: bool = False,
    has_invalid: bool = False,
    dot_precision: str = "bf16",
    q_scale: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    return_lse: bool = False,
    attn_sink: torch.Tensor | None = None,
    extra_kv: torch.Tensor | None = None,
    extra_indptr: torch.Tensor | None = None,
    extra_indices: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Sparse (top-k gathered) MLA attention.

    Everything after softmax_scale is keyword-only: kv_indptr and kv_indices are
    both flat int32, so a positional mix-up would type-check and return garbage.

    Supported KV cache formats. The format is inferred from kv_buffer's shape,
    dtype and kv_scale; each row gives what the caller has to pass. R is the
    QK width, kv_lora_rank + qk_rope_head_dim.

        format          kv_buffer                       kv_scale          geometry args
        bf16            [slots, R] / [nb, block, R] /   None              as the model
                        [slots, 1, 1, R], bf16
        fp8_scalar      same shapes, fp8 or uint8       [1] f32 k_scale   as the model
                        (GLM-5.x; the only format that
                        can run dot_precision="fp8")
        fp8_g64         [slots, D] fp8 or uint8         [slots, D//64]    kv_lora_rank=D,
                        (uniform pool, rope inside)     f32               qk_rope_head_dim=0
        fp8_dsv32_mla   [nb, block, 656] uint8          None              512 / 64
                        (DeepSeek-V3.2, Kimi-K3; vLLM
                        fp8_ds_mla, 512 fp8 | 4 f32
                        per-128 | 64 bf16 rope)
        fp8_dsv4_mla    [nb, block, 584] uint8          None              ignored
                        (DeepSeek-V4; vLLM fp8_ds_mla,
                        448 fp8 | 64 bf16 rope + 8 B
                        UE8M0 trailer per block)

    Geometry, set by qk_rope_head_dim: separated rope (DeepSeek-V3.2,
    GLM-5.1/5.2), where the query is the latent plus an appended rope and V is
    the latent; rope-free (GLM-5.3-Flash), qk_rope_head_dim=0, where the query
    is the latent alone. DeepSeek-V4 keeps its rope inside the 512-wide row
    (448 nope + 64 rope) with V the whole row; that layout is fixed by the cache
    format, so the geometry args are not read for it.
    A bf16 pool carries no rope information, so a rope-inside or rope-free
    model on a bf16 cache must pass qk_rope_head_dim=0.

    Args:
        q: [C, H, kv_lora_rank + qk_rope_head_dim] queries, one row per
            query token (prefill and decode alike).
        qk_rope_head_dim: width of the appended rope. 0 means rope-free, and
            the QK contraction becomes a single dot over kv_lora_rank.
        kv_buffer: the KV pool, one of the formats in the table above.
        kv_indptr: [C + 1] int32 prefix sum of per-query index counts.
        kv_indices: flat int32 GLOBAL slot ids into the pool.
        softmax_scale: the layer's softmax scale.
        kv_scale: f32 cache scale, shape per the table; the packed formats
            carry their own scales and take None.
        kv_splits: split-K override; default follows the occupancy policy.
        skip_reduce: with split-K active, return (part_acc, part_m, part_l)
            instead of launching the combine.
        has_invalid: index stream carries -1 sentinels (masked out). Default
            False: on the fp8-dots path True disables the direct-to-LDS pipeline
            (see the ASYNC_LDS static assert), so only set it when the stream
            really carries -1.
        attn_sink: optional [H] f32 per-head sink: exp(sink) joins the softmax
            denominator (and the LSE) as a virtual key. None means no sink,
            which is not the same as a zero sink.
        extra_kv, extra_indptr, extra_indices: a second segment attended in the
            same pass, for the SWA-window + top-k two-loop. All three together.
        dot_precision: what the QK and PV matrix-core ops run in.

            "bf16" (default): the KV tile is dequantized to bf16 on its way
                into LDS and both dots are bf16. Works with every cache format.
            "fp8": the cache's own code points go to the fp8 matrix core with no
                dequant, and the per-tensor scale folds outside the tile loop.

            q is adapted to the choice. bf16 q is quantized in the kernel
            prologue, one scale per (query, head-block) tile; fp8 q is passed
            straight through, and fp8 q under "bf16" dots is widened in-kernel,
            which is exact.
        q_scale: scalar f32, required when q arrives already fp8 (the scale it
            was quantized with; the aiter asm convention). It describes q's
            encoding, so bf16 q must not carry one.
        out: optional [C, H, >= kv_lora_rank] bf16 destination.
        return_lse: also return the natural-log log-sum-exp, [C, H] f32, for
            merging partials across context-parallel ranks. A fully masked row
            reports -inf.

    Returns:
        (out, lse), out is
        [C, H, kv_lora_rank] bf16 (the latent V), lse is None unless return_lse.
    """
    assert q.ndim == 3, f"expected q=[b,h,d], got {q.shape}"
    paged_args = (extra_kv, extra_indptr, extra_indices)
    if any(a is not None for a in paged_args) or _is_paged_cache(
        q, kv_buffer, kv_lora_rank, kv_scale
    ):
        return _forward_paged(
            q,
            kv_buffer,
            kv_indptr,
            kv_indices,
            softmax_scale,
            kv_scale,
            kv_splits,
            skip_reduce,
            has_invalid,
            dot_precision,
            q_scale,
            out,
            return_lse,
            attn_sink,
            extra_kv,
            extra_indptr,
            extra_indices,
        )
    assert arch_info.get_arch() == "gfx950", "sparse_mla_fwd is gfx950-only"
    q_is_fp8 = q.dtype == torch.float8_e4m3fn
    if q.dtype not in (torch.bfloat16, torch.float8_e4m3fn):
        raise ValueError(
            f"q must be bf16 or float8_e4m3fn, got {q.dtype}"
            + (
                " (fnuz is a different encoding from what the matrix core reads)"
                if "fnuz" in str(q.dtype)
                else ""
            )
        )
    if q_is_fp8:
        # Caller-quantized q, one scaled_fp8_quant over [C, H*d_qk]
        if q_scale is None:
            raise ValueError("fp8 q needs q_scale (the scale it was quantized with)")
        q_scale = q_scale.reshape(1).to(torch.float32).contiguous()
    num_queries, num_heads, d_qk = q.shape
    assert (
        d_qk == kv_lora_rank + qk_rope_head_dim
    ), f"q last dim {d_qk} != {kv_lora_rank} + {qk_rope_head_dim}"
    assert qk_rope_head_dim >= 0, "qk_rope_head_dim cannot be negative"
    _LOGGER.info(
        f"SPARSE_MLA C={num_queries} H={num_heads} d_qk={d_qk} "
        f"nnz={kv_indices.shape[0]}"
    )

    fmt, cache, alt, scl, block_size = _infer_cache_format(
        kv_buffer, d_qk, kv_lora_rank, qk_rope_head_dim, kv_scale
    )
    fp8_dots = _resolve_dot_precision(dot_precision, fmt)
    if q_scale is not None and not q_is_fp8:
        raise ValueError(
            "q_scale describes an fp8 q. With bf16 q the kernel quantizes per "
            "(query, head-block) tile, so drop q_scale, or pass q already "
            "quantized."
        )
    kv_indices = _as_int32_contiguous_1d(kv_indices)
    kv_indptr = _as_int32_contiguous_1d(kv_indptr)
    assert kv_indptr.numel() == num_queries + 1
    has_sink = attn_sink is not None
    if has_sink:
        if attn_sink.numel() != num_heads:
            raise ValueError(
                f"attn_sink must have one entry per head ({num_heads}), got "
                f"{tuple(attn_sink.shape)}"
            )
        attn_sink = attn_sink.reshape(-1).to(torch.float32).contiguous()
    else:
        # The kernel still wants a live pointer for the compile-time-elided slot.
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
    # buffer_load carries a signed 32-bit offset
    MAX_BYTES = 2**31 - 1
    use_buffer_load = max_addressable_bytes(cache) < MAX_BYTES
    idx_use_buffer_load = max_addressable_bytes(kv_indices) < MAX_BYTES

    head_aligned = num_heads % block_m == 0
    heads_blocks = (num_heads + block_m - 1) // block_m
    out = _check_out(out, q, kv_lora_rank)
    if return_lse and skip_reduce:
        raise ValueError("return_lse needs the combine, so skip_reduce cannot be set")
    # A live pointer either way, like attn_sink below.
    lse = (
        torch.empty((num_queries, num_heads), dtype=torch.float32, device=q.device)
        if return_lse
        else torch.empty(1, dtype=torch.float32, device=q.device)
    )

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

    # Dequant chunking
    col_reps = kv_lora_rank // (_GATHER_TW1 * 16)
    chunk_axis = 1 if col_reps >= 4 else 0
    nope_chunk = max(1, _BLOCK_K // 4) if chunk_axis == 0 else min(128, kv_lora_rank)
    async_lds_on, block_k, waves_per_eu = _async_launch_config(
        fp8_dots,
        has_invalid,
        num_queries,
        heads_blocks,
        num_splits,
        avg_topk,
        use_buffer_load,
        uni_tile=True,
        has_extra=False,
    )
    # num_warps stays 4: warps tile the dot's N (MFMA N=16), so a larger BLOCK_K just
    # gives each warp two N tiles.

    # Q is read once per query without split-K, and re-read by every split
    q_cache = ".cg" if num_splits == 1 else ""
    grid = (num_queries, num_splits, heads_blocks)
    _sparse_mla_gfx950[grid](
        q,
        cache,
        alt,
        kv_indices,
        kv_indptr,
        cache,  # extra_* segment: unread placeholders (HAS_EXTRA=False)
        alt,
        kv_indices,
        kv_indptr,
        attn_sink,
        out,
        part_m,
        part_l,
        part_acc,
        scl,  # f32 side-channel: k_scale ("fp8_scalar") / f32 view ("fp8_dsv32_mla")
        scl,  # extra segment's slot, unread (HAS_EXTRA=False)
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
        HAS_SINK=has_sink,
        MAIN_FMT=fmt,
        EXTRA_FMT=fmt,
        MAIN_BLOCK_SIZE=block_size,
        EXTRA_BLOCK_SIZE=block_size,
        CS0_ALIGN=cs0_align,
        NOPE_DIM=kv_lora_rank,
        ROPE_DIM=qk_rope_head_dim,
        HEAD_SIZE=kv_lora_rank,
        ROPE_SEPARATE=qk_rope_head_dim > 0,
        BLOCK_M=block_m,
        BLOCK_K=block_k,
        NUM_SPLITS=num_splits,
        HEAD_ALIGNED=head_aligned,
        MFMA_K=_MFMA_K,
        GATHER_TW1=_GATHER_TW1,
        NOPE_CHUNK=nope_chunk,
        CHUNK_AXIS=chunk_axis,
        PART_STORE_CACHE="",
        UNI_TILE=True,
        GRID_ORDER="qsh",
        Q_CACHE=q_cache,
        MAIN_SPLITS=num_splits,
        ADAPTIVE_SPLITS=num_splits > 1,
        DEQ="none",
        MAIN_USE_BUFFER_LOAD=use_buffer_load,
        EXTRA_USE_BUFFER_LOAD=use_buffer_load,
        IDX_BUFFER_LOAD=idx_use_buffer_load,
        HAS_INVALID=has_invalid,
        FP8_MFMA=fp8_dots,
        ASYNC_LDS=async_lds_on,
        ROPE_VEC=_ASYNC_ROPE_VEC,
        GATHER_CACHE="",
        q_scl_ptr=q_scale,
        Q_FP8=q_is_fp8,
        lse_ptr=lse,
        HAS_LSE=return_lse,
        num_warps=num_warps,
        waves_per_eu=waves_per_eu,
    )

    if num_splits == 1:
        return out, (lse if return_lse else None)
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
        HEAD_SIZE=kv_lora_rank,
        BLOCK_M=1,
        NUM_SPLITS=num_splits,
        HEAD_ALIGNED=True,
        ADAPTIVE_SPLITS=num_splits > 1,
        lse_ptr=lse,
        HAS_LSE=return_lse,
        num_warps=1,
    )
    return out, (lse if return_lse else None)
