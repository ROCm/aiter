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


def _cache_pointers(fmt, kv, d_qk, kv_scale):
    """-> (cache, alt_ptr, scl_ptr, block_size) for an MLA-route format. Pointer
    roles follow the kernel's Seg contract (see its docstring)."""
    if kv.ndim == 4:  # [slots, 1, 1, R], the asm mla_decode_fwd view
        kv = kv.reshape(kv.shape[0], kv.shape[3])
    elif kv.ndim == 3 and kv.shape[2] == d_qk:
        # vLLM paged cache [nb, block_size, R]: indices are global slot ids and
        # rows are contiguous, so the flat view is stride-identical.
        if not (kv.stride(2) == 1 and kv.stride(1) == kv.shape[2]):
            raise ValueError("a [nb, block, R] pool must have contiguous rows")
        kv = kv.reshape(-1, d_qk)
    if fmt == "bf16":
        return kv, kv, None, 1
    if fmt == "fp8_scalar":
        u8 = kv.view(torch.uint8)
        return u8, u8, kv_scale.reshape(1), 1
    # fp8_dsv32_mla: 512 fp8 | 4 f32 per-128 scales | 64 bf16 rope = 656 B.
    u8 = kv if kv.dtype == torch.uint8 else kv.view(torch.uint8)
    if not (u8.stride(2) == 1 and u8.stride(1) == u8.shape[2]):
        raise ValueError("fp8_dsv32_mla rows must be contiguous records")
    return u8, u8.view(torch.bfloat16), u8.view(torch.float32), u8.shape[1]


def _mla_num_splits(
    num_queries: int, heads_blocks: int, avg_topk: float, block_k: int = 64
) -> int:
    """Split-K count for the sparse-MLA decode.

    Below one workgroup per CU, split to fill the machine but never past 8.
    """
    num_sms = get_num_sms()
    base_wg = max(1, num_queries * heads_blocks)
    cta_cap = max(1, (2 * num_sms) // base_wg)
    tiles = max(1, math.ceil(avg_topk / block_k))
    if base_wg >= num_sms:
        return max(1, min(cta_cap, tiles // 16))
    return max(1, min(cta_cap, tiles, 8))


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
    block_k: int = 64,
) -> tuple[bool, int, int]:
    """-> (ASYNC_LDS, BLOCK_K, waves_per_eu) for this launch.

    BLOCK_K follows what the grid supplies, not the token count: given enough
    workgroups to fill four waves/SIMD (prefill) the small tile takes that
    occupancy; otherwise (decode, where split-K caps the grid near two
    workgroups/CU) the large tile wins instead, by halving the cross-warp softmax
    exchange rate per token.
    """
    ASYNC_LDS_DEFAULT = True
    enabled = (
        ASYNC_LDS_DEFAULT
        and fp8_dots
        and uni_tile
        and not has_invalid
        and not has_extra
    )
    workgroups = num_queries * heads_blocks * max(1, num_splits)
    num_sms = get_num_sms()
    if enabled and workgroups >= 4 * num_sms:
        return True, 64, 4
    waves_per_eu = 2
    if use_buffer_load and workgroups <= num_sms:
        waves_per_eu = 1
    return enabled, (128 if enabled else block_k), waves_per_eu


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


_DSV4_ROW = 448 + 2 * 64 + 8  # 584 B: fp8 nope | bf16 rope | 8 B UE8M0 trailer


def _dsv32_row(kv_lora_rank, qk_rope_head_dim):
    return kv_lora_rank + 4 * (kv_lora_rank // 128) + 2 * qk_rope_head_dim


def _classify_flat(kv, width, slots, kv_scale, what):
    """Flat pool rows are one QK row per slot; dtype and kv_scale pick the tag."""
    if kv.dtype == torch.bfloat16:
        return "bf16"  # kv_scale, if any, is ignored
    if kv.dtype == torch.float8_e4m3fnuz:
        raise ValueError(
            f"{what}: float8_e4m3fnuz is the gfx942 encoding; gfx950 reads OCP e4m3"
        )
    if kv.element_size() != 1:
        raise ValueError(f"{what}: unsupported cache dtype {kv.dtype}")
    if kv_scale is None:
        raise ValueError(
            f"{what}: a flat fp8 cache needs kv_scale, [1] f32 (fp8_scalar) or "
            f"[slots, D // 64] f32 (fp8_g64)"
        )
    if kv_scale.dtype != torch.float32:
        raise ValueError(f"{what}: kv_scale must be f32, got {kv_scale.dtype}")
    if kv_scale.numel() == 1:
        return "fp8_scalar"
    if width % 64 or tuple(kv_scale.shape) != (slots, width // 64):
        raise ValueError(
            f"{what}: kv_scale {tuple(kv_scale.shape)} is neither [1] (fp8_scalar) "
            f"nor [slots, D // 64] = [{slots}, {width // 64}] (fp8_g64)"
        )
    return "fp8_g64"


def _classify_cache(q, kv, kv_lora_rank, qk_rope_head_dim, kv_scale, what="kv_buffer"):
    """kv -> one of the kernel's format tags, or raise.

    Record width, dtype and the kv_scale shape are the only things that tell the
    formats apart, so whatever they do not pin down is rejected here instead of
    being decoded as the wrong layout. Host metadata only: safe under graph
    capture.
    """
    d_qk = q.shape[-1]
    if kv.device != q.device:
        raise ValueError(f"{what} is on {kv.device}, q is on {q.device}")
    if kv.ndim == 4 and (kv.shape[1] != 1 or kv.shape[2] != 1):
        raise ValueError(
            f"{what}: a 4-D cache must be [slots, 1, 1, R], got {tuple(kv.shape)}"
        )
    if kv.ndim in (2, 4):
        width = kv.shape[-1]
        if width != d_qk:
            raise ValueError(
                f"{what}: flat cache rows are {width} wide but q is {d_qk}; a flat "
                f"pool stores one QK row (kv_lora_rank + qk_rope_head_dim) per slot"
            )
        return _classify_flat(kv, width, kv.shape[0], kv_scale, what)
    if kv.ndim == 3:
        nb, block, width = kv.shape
        if width == d_qk:
            # [nb, block, R]: a flat pool stored in blocks. Same rules, except the
            # per-64 scale vector is only defined for the 2-D pool.
            fmt = _classify_flat(kv, width, nb * block, kv_scale, what)
            if fmt == "fp8_g64":
                raise ValueError(f"{what}: fp8_g64 needs a 2-D [slots, D] pool")
            return fmt
        if kv_scale is not None:
            raise ValueError(
                f"{what}: packed caches carry their own scales; kv_scale must be None"
            )
        one_byte = kv.element_size() == 1
        if one_byte and width == _DSV4_ROW:
            return "fp8_dsv4_mla"
        dsv32 = _dsv32_row(kv_lora_rank, qk_rope_head_dim)
        if one_byte and qk_rope_head_dim > 0 and width == dsv32:
            return "fp8_dsv32_mla"
        raise ValueError(
            f"{what}: unrecognized 3-D cache, {width}-wide {kv.dtype} records for q "
            f"width {d_qk}. Expected {d_qk} (bf16 or fp8_scalar block pool), "
            f"{_DSV4_ROW} uint8 (fp8_dsv4_mla) or {dsv32} uint8 (fp8_dsv32_mla with "
            f"kv_lora_rank={kv_lora_rank}, qk_rope_head_dim={qk_rope_head_dim})"
        )
    raise ValueError(f"{what} must be 2-D, 3-D or 4-D, got {kv.ndim}-D")


def _check_geometry(fmt, d_qk, kv_lora_rank, qk_rope_head_dim):
    if kv_lora_rank <= 0 or qk_rope_head_dim < 0:
        raise ValueError(
            f"kv_lora_rank must be > 0 and qk_rope_head_dim >= 0, got "
            f"{kv_lora_rank} / {qk_rope_head_dim}"
        )
    if fmt == "fp8_dsv4_mla":
        if d_qk != 512:
            raise ValueError(
                f"fp8_dsv4_mla needs q width 512 (448 nope + 64 rope), got {d_qk}"
            )
        return
    if fmt == "fp8_g64":
        return  # the whole row is the head; geometry args are not read
    if d_qk != kv_lora_rank + qk_rope_head_dim:
        hint = (
            " A bf16 or fp8_scalar cache carries no rope information: pass "
            "qk_rope_head_dim=0 for rope-inside-the-row or rope-free models."
            if fmt in ("bf16", "fp8_scalar")
            else ""
        )
        raise ValueError(
            f"q width {d_qk} != kv_lora_rank {kv_lora_rank} + qk_rope_head_dim "
            f"{qk_rope_head_dim}.{hint}"
        )
    if fmt == "fp8_dsv32_mla" and kv_lora_rank % 128:
        raise ValueError(
            f"fp8_dsv32_mla stores one scale per 128 latent columns; kv_lora_rank "
            f"{kv_lora_rank} is not a multiple of 128"
        )


def _check_index_stream(indptr, indices, num_queries, device, what=""):
    """Both are flat integer tensors, so only the length of indptr can catch a
    swapped pair. No device reads."""
    for name, t in ((f"{what}kv_indptr", indptr), (f"{what}kv_indices", indices)):
        if not torch.is_tensor(t):
            raise ValueError(f"{name} must be a tensor, got {type(t).__name__}")
        if t.dtype not in (torch.int32, torch.int64):
            raise ValueError(f"{name} must be int32 (int64 accepted), got {t.dtype}")
        if t.ndim != 1:
            raise ValueError(f"{name} must be 1-D, got {tuple(t.shape)}")
        if t.device != device:
            raise ValueError(f"{name} is on {t.device}, q is on {device}")
    if indptr.numel() != num_queries + 1:
        raise ValueError(
            f"{what}kv_indptr must have C + 1 = {num_queries + 1} entries, got "
            f"{indptr.numel()}. kv_indptr and kv_indices are both flat int32; "
            "check their order."
        )


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

        format         kv_buffer                     kv_scale        geometry args
        bf16           [slots, R], [nb, block, R],   None            as the model
                       [slots, 1, 1, R]; bf16
        fp8_scalar     same shapes, fp8 or uint8     [1] f32         as the model
                       (GLM-5.x; the only format
                       that runs dot_precision="fp8")
        fp8_g64        [slots, D] fp8 or uint8       [slots, D//64]  kv_lora_rank=D,
                       (uniform pool, rope inside)   f32             qk_rope_head_dim=0
        fp8_dsv32_mla  [nb, block, 656] uint8        None            512 / 64
                       (DeepSeek-V3.2, Kimi-K3; vLLM
                       fp8_ds_mla: 512 fp8 | 4 f32
                       per-128 | 64 bf16 rope)
        fp8_dsv4_mla   [nb, block, 584] uint8        None            ignored
                       (DeepSeek-V4; vLLM fp8_ds_mla:
                       448 fp8 | 64 bf16 rope + 8 B
                       UE8M0 trailer per block)

    Geometry, set by qk_rope_head_dim: separated rope (DeepSeek-V3.2,
    GLM-5.1/5.2), where the query is the latent plus an appended rope and V is
    the latent; rope-free (GLM-5.3-Flash), qk_rope_head_dim=0, where the query
    is the latent alone. DeepSeek-V4 keeps its rope inside the 512-wide row
    (448 nope + 64 rope) with V the whole row; that layout is fixed by the cache
    format, so the geometry args (kv_lora_rank and qk_rope_head_dim) are not
    read for it. A bf16 pool carries no rope information, so a rope-inside or
    rope-free model on a bf16 cache must pass qk_rope_head_dim=0.

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
            False.
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
                tensor scale fp8 kv cache only.

            q is adapted to the choice. bf16 q is quantized in the kernel
            prologue, one scale per (query, head-block) tile; fp8 q is passed
            straight through, and fp8 q under "bf16" dots is widened in-kernel.

        q_scale: scalar f32, required when q arrives already fp8 (the scale it
            was quantized with; the aiter asm convention). Ignored for bf16 q,
            which the kernel quantizes itself when dot_precision="fp8".
        out: optional [C, H, >= kv_lora_rank] bf16 destination.
        return_lse: also return the natural-log log-sum-exp, [C, H] f32, for
            merging partials across context-parallel ranks. A fully masked row
            reports -inf.

    Returns:
        (out, lse), out is
        [C, H, kv_lora_rank] bf16 (the latent V), lse is None unless return_lse.
    """
    if q.ndim != 3:
        raise ValueError(f"expected q=[C, H, d_qk], got {tuple(q.shape)}")
    num_queries, num_heads, d_qk = q.shape
    fmt = _classify_cache(q, kv_buffer, kv_lora_rank, qk_rope_head_dim, kv_scale)
    _LOGGER.info(
        f"SPARSE_MLA_FWD: q={tuple(q.shape)} kv_buffer={tuple(kv_buffer.shape)} "
        f"{kv_buffer.dtype} kv_indices={tuple(kv_indices.shape)} fmt={fmt}"
    )
    _check_geometry(fmt, d_qk, kv_lora_rank, qk_rope_head_dim)
    _check_index_stream(kv_indptr, kv_indices, num_queries, q.device)
    if attn_sink is not None and attn_sink.numel() != num_heads:
        raise ValueError(
            f"attn_sink must have one entry per head ({num_heads}), got "
            f"{tuple(attn_sink.shape)}"
        )
    extra = (extra_kv, extra_indptr, extra_indices)
    has_extra = all(a is not None for a in extra)
    if any(a is not None for a in extra) and not has_extra:
        raise ValueError("extra_kv, extra_indptr and extra_indices go together")
    if has_extra:
        # The SWA-window + top-k two-loop: both segments are block caches the
        # dsv4 decoder reads. A flat main pool would silently drop the extra one.
        if kv_buffer.ndim != 3 or fmt not in ("fp8_dsv4_mla", "bf16"):
            raise ValueError(
                "the two-loop needs a 3-D block cache (fp8_dsv4_mla or bf16) as "
                f"kv_buffer, got {fmt}"
            )
        extra_fmt = _classify_cache(
            q, extra_kv, kv_lora_rank, qk_rope_head_dim, None, what="extra_kv"
        )
        if extra_kv.ndim != 3 or extra_fmt not in ("fp8_dsv4_mla", "bf16"):
            raise ValueError(
                f"extra_kv must be a 3-D fp8_dsv4_mla or bf16 block cache, got "
                f"{extra_fmt}"
            )
        _check_index_stream(
            extra_indptr, extra_indices, num_queries, q.device, what="extra_"
        )
    if fmt in ("fp8_dsv4_mla", "fp8_g64") or has_extra:
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
        # Caller-quantized q, tensor scale
        if q_scale is None:
            raise ValueError("fp8 q needs q_scale (the scale it was quantized with)")
        if q_scale.numel() != 1:
            raise ValueError(
                f"q_scale must be a single per-tensor value, got "
                f"{tuple(q_scale.shape)}; per-head q scales are not supported"
            )
        q_scale = q_scale.reshape(1).to(torch.float32).contiguous()
    cache, alt, scl, block_size = _cache_pointers(fmt, kv_buffer, d_qk, kv_scale)
    fp8_dots = _resolve_dot_precision(dot_precision, fmt)
    if not q_is_fp8:
        # q_scale describes an fp8 q's encoding. With bf16 q the kernel quantizes
        # per (query, head-block) tile when the dots are fp8, so a caller-supplied
        # scale has nothing to apply to; callers pass layer._q_scale regardless.
        q_scale = None
    kv_indices = _as_int32_contiguous_1d(kv_indices)
    kv_indptr = _as_int32_contiguous_1d(kv_indptr)
    has_sink = attn_sink is not None
    if has_sink:
        attn_sink = attn_sink.reshape(-1).to(torch.float32).contiguous()
    else:
        # The kernel still wants a live pointer for the compile-time-elided slot.
        attn_sink = torch.empty(1, device=q.device, dtype=torch.float32)

    # Tuned launch config (gfx950 / MI355). H < 16 runs natively at
    # BLOCK_M = next_pow2(H) instead of padding heads
    block_m = 16 if num_heads >= 16 else max(8, 1 << (num_heads - 1).bit_length())
    block_k = 64
    num_warps = block_k // 16

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
        num_splits = _mla_num_splits(num_queries, heads_blocks, avg_topk, block_k)

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

    # Dequant chunking. The gather layout puts 32 of a wave's 64 lanes along a row
    # (16 B each, 512 B) and the other 32 on a second row; col_reps is how many
    # such spans each lane holds, which is what makes a column split a free
    # register rename.
    col_reps = kv_lora_rank // 512
    chunk_axis = 1 if col_reps >= 4 else 0
    nope_chunk = max(1, block_k // 4) if chunk_axis == 0 else min(128, kv_lora_rank)
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
        block_k=block_k,
    )

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
