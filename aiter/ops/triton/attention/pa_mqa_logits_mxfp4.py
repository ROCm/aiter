# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# Launcher, scheduler and cache layout helpers for the paged MXFP4 MQA-logits
# kernel.

import functools

import torch
import triton
from aiter.ops.triton.utils._triton import arch_info
from aiter.ops.triton.utils.device_info import get_num_sms

from aiter.ops.triton._gluon_kernels.gfx950.attention.pa_mqa_logits_mxfp4 import (
    _pa_mqa_logits_mxfp4_kernel,
    _pa_mqa_logits_mxfp4_sched_kernel,
)

SCALE_GROUP = 32
K_WIDTH = 16
# Past this the chunk is compute bound
COMPUTE_CHUNK = 512
# Narrow enough to change config for spec. decoding
SPEC_ROWS = 8
# no need for dynamic scheduling for lower conc.
MIN_DYNAMIC_BATCH = 4
# this is the most performant page size
IDEAL_PAGE_SIZE = 64

def mfma_nonk_dim(num_heads: int, head_size: int) -> int:
    # 32x32x64 leaves one head bit across lanes where 16x16x128 leaves two, so
    # the head sum needs one cross-lane step instead of two.
    return 32 if (head_size <= 64 or num_heads >= 32) else 16


def kv_tile(page_size: int) -> int:
    # Capped by the page so a tile never spans two pages
    return min(IDEAL_PAGE_SIZE, page_size)

def cache_format(num_heads: int, head_size: int, page_size: int) -> dict:
    """Everything needed to write a preshuffled cache and to read it back."""
    npt = mfma_nonk_dim(num_heads, head_size)
    bkv = kv_tile(page_size)
    if page_size % npt:
        raise ValueError(f"page_size {page_size} must be a multiple of the MFMA "
                         f"N ({npt}) or a shuffle group straddles two pages")
    if page_size % bkv:
        raise ValueError(f"page_size {page_size} must be a multiple of BLOCK_KV "
                         f"({bkv})")
    return dict(n_per_tile=npt, d_per_tile=K_WIDTH, block_kv=bkv)


def preshuffle_values(x: torch.Tensor, n_per_tile: int,
                      d_per_tile: int = K_WIDTH) -> torch.Tensor:
    """[P, page, D//2] uint8 into dot-operand order, within each page."""
    p, rows, d = x.shape
    return (x.reshape(p, rows // n_per_tile, n_per_tile, d // d_per_tile, d_per_tile)
            .permute(0, 1, 3, 2, 4).contiguous().reshape(p, rows, d))


def unshuffle_values(x: torch.Tensor, n_per_tile: int,
                     d_per_tile: int = K_WIDTH) -> torch.Tensor:
    p, rows, d = x.shape
    return (x.reshape(p, rows // n_per_tile, d // d_per_tile, n_per_tile, d_per_tile)
            .permute(0, 1, 3, 2, 4).contiguous().reshape(p, rows, d))


def preshuffle_scales(x: torch.Tensor, n_per_tile: int) -> torch.Tensor:
    """[P, page, D//32] e8m0 into the order the scale load reads.

    Group is one MFMA tile, lane's run over the scale axis innermost -- both
    from num_heads and head_size, so the order carries no BLOCK_KV and no warp
    count.
    """
    WARP_SIZE = 64
    p, rows, ns = x.shape
    s_lo = WARP_SIZE // n_per_tile
    s_hi = ns // s_lo
    return (x.reshape(p, rows // n_per_tile, n_per_tile, s_hi, s_lo)
            .permute(0, 1, 4, 2, 3).contiguous().reshape(p, rows, ns))


def unshuffle_scales(x: torch.Tensor, n_per_tile: int) -> torch.Tensor:
    WARP_SIZE = 64
    p, rows, ns = x.shape
    s_lo = WARP_SIZE // n_per_tile
    s_hi = ns // s_lo
    return (x.reshape(p, rows // n_per_tile, s_lo, n_per_tile, s_hi)
            .permute(0, 1, 3, 4, 2).contiguous().reshape(p, rows, ns))


def preshuffle_cache(values: torch.Tensor, scales: torch.Tensor,
                     num_heads: int, head_size: int):
    """Natural order into the stored order. A cache-prep kernel should emit
    these bytes directly; this is the reference for what that means."""
    f = cache_format(num_heads, head_size, values.shape[1])
    return (preshuffle_values(values, f["n_per_tile"], f["d_per_tile"]),
            preshuffle_scales(scales, f["n_per_tile"]))


def pack_cache(values: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """([P, page, D//2], [P, page, D//32]) -> [P, page, 1, D//2 + D//32].

    Values then scales, as the fp8 cache does. Each region is separately
    contiguous, which gives the kernel two constexpr page strides.
    """
    num_pages, page_size, head_bytes = values.shape
    num_scales = scales.shape[2]
    idim = head_bytes + num_scales
    out = torch.empty(num_pages, page_size * idim, dtype=torch.uint8,
                      device=values.device)
    out[:, :page_size * head_bytes] = values.reshape(num_pages, -1)
    out[:, page_size * head_bytes:] = scales.reshape(num_pages, -1)
    return out.view(num_pages, page_size, 1, idim)


def _split_cache(kv_cache: torch.Tensor, head_size: int):
    num_pages = kv_cache.shape[0]
    flat = kv_cache.reshape(num_pages, -1)
    page_bytes = flat.shape[1]
    idim = head_size // 2 + head_size // SCALE_GROUP
    page_size = page_bytes // idim
    head_bytes = head_size // 2
    return (flat[:, :page_size * head_bytes], flat[:, page_size * head_bytes:],
            page_size, page_bytes)


# heuristics


def _spec_block_m(next_n: int, num_heads: int) -> int:
    """Rows per workgroup for a speculative chunk.

    Search for best block m:
        Lower values have better occupancy, but worse kv cache reread
        Higher values have worse occupancy, but better kv cache rereaf
    """
    cap = 7 if num_heads <= 32 else (3 if next_n <= 6 else 2)
    return min(range(1, cap + 1),
               key=lambda b: (-(-next_n // b), b * (-(-next_n // b))))


def plan_block_m(num_heads: int, next_n: int) -> int:
    """BLOCK_M. Exported so a caller sizing operands cannot drift from it."""
    if next_n >= COMPUTE_CHUNK:
        # wide chunk
        return 3 if num_heads <= 32 else 1
    if 2 <= next_n <= SPEC_ROWS:
        return _spec_block_m(next_n, num_heads)
    if num_heads > 32 and next_n >= 2:
        return min(2, next_n)
    return 1 if next_n < 2 else min(2, next_n)


def _kv_splits(batch, row_blocks, max_model_len, block_kv, target_wgs,
               min_tiles, max_tiles):
    """How many workgroups to put on one row block's KV walk.

    A split partitions the output, so there is no reduction to pay for. Three
    terms: fill the machine, keep any workgroup under max_tiles so the tail does
    not set the step time, and never split below min_tiles.
    """
    tile_q = max(1, batch * row_blocks)
    n_tiles = max(1, (max_model_len + block_kv - 1) // block_kv)
    by_occupancy = (target_wgs + tile_q - 1) // tile_q
    by_balance = (n_tiles + max_tiles - 1) // max_tiles
    by_length = max(1, n_tiles // max(1, min_tiles))
    return max(1, min(max(by_occupancy, by_balance), by_length))





@functools.lru_cache(maxsize=256)
def _select_config(num_heads, head_size, next_n, page_size, preshuffle,
                   clean_logits):
    n_per_tile = mfma_nonk_dim(num_heads, head_size)
    compute_chunk = next_n >= COMPUTE_CHUNK
    spec_rows = num_heads <= 32 and 2 <= next_n <= SPEC_ROWS
    wide_decode = num_heads > 32 and next_n == 1

    if preshuffle:
        # Register path. One warp: with no KV tile in LDS a second warp has no
        # producer/consumer to help with, only its barriers.
        block_m = min(plan_block_m(num_heads, next_n), next_n)
        cfg = dict(
            num_warps=1,
            num_buffers=1,
            waves_per_eu=3 if (compute_chunk or wide_decode) else 2,
            depth=2 if wide_decode else 1,
            unroll=2 if ((compute_chunk or spec_rows) and block_m <= 6) else 1,
            fold_asm=1 if (compute_chunk or spec_rows or num_heads > 32) else 0)
    else:
        # LDS path, for an unshuffled cache. A second warp buys issue rate for a
        # barrier per tile, worth it only where one warp cannot hold the fold.
        block_m = min(2 if (num_heads <= 32 and next_n >= 2) else 1, next_n)
        if next_n == 1:
            waves_per_eu = 2
        elif num_heads <= 32 and next_n <= SPEC_ROWS:
            waves_per_eu = 3
        else:
            waves_per_eu = 4
        cfg = dict(
            num_warps=2 if (num_heads > 32 and next_n > 1) else 1,
            num_buffers=2,
            waves_per_eu=waves_per_eu,
            depth=1,
            # One row, which is what lets UNROLL be 2.
            unroll=2 if block_m == 1 else 1,
            fold_asm=1 if num_heads <= 32 else 0)

    cfg.update(
        block_m=block_m,
        row_blocks=(next_n + block_m - 1) // block_m,
        block_kv=kv_tile(page_size),
        n_per_tile=n_per_tile,
        # Workgroups to fill the machine: 4 SIMDs per CU
        target_wgs=4 * get_num_sms() * cfg["waves_per_eu"],
        # Hoists the block-table read an iteration ahead. Off at 64 heads,
        # where the extra live value costs more than it buys.
        page_pipe=1 if num_heads <= 32 else 0,
        m_chunk=n_per_tile if (num_heads > n_per_tile and n_per_tile == 32) else 0,
        num_chains=1,
        relaxed_store=0 if clean_logits else 1,
        # Follows FOLD_ASM rather than standing alone: without it the SLP
        # vectorizer pairs the adds into v_pk_add_f32, which has no abs modifier.
        relu_add=cfg["fold_asm"],
        min_tiles_per_split=4,
        max_tiles_per_split=64)
    return cfg


def select_config(num_heads, head_size, next_n, page_size, preshuffle=1,
                  clean_logits=True):
    """The config for one shape, cached"""
    return dict(_select_config(num_heads, head_size, next_n, page_size,
                               int(bool(preshuffle)), bool(clean_logits)))


def build_schedule(context_lens, next_n, num_heads, head_size,
                   page_size=IDEAL_PAGE_SIZE, preshuffle=1, out=None):
    """Descriptors for one launch, or None if the shape does not fit.

    A descriptor is (sequence, row block, slice index, slice count), relative
    rather than absolute, so the kernel converts it against its own tile count.
    """
    plan = select_config(num_heads, head_size, next_n, page_size, preshuffle)
    row_blocks, block_m = plan["row_blocks"], plan["block_m"]
    block_kv, target_wgs = plan["block_kv"], plan["target_wgs"]
    batch = int(context_lens.numel())
    work = batch * row_blocks
    # Too few sequences to have an imbalance worth the scheduler launch.
    if batch < MIN_DYNAMIC_BATCH:
        return None
    # enough work already
    if work > target_wgs:
        return None
    align_w = max(16, 1 << (work - 1).bit_length())
    if out is None or out.numel() < target_wgs * 4:
        out = torch.empty(target_wgs * 4, dtype=torch.int32,
                          device=context_lens.device)
    # Slots one scheduler program describes, it will try to create equal work per WG
    # while generating enough WGs
    SCHED_BLOCK_P = 4
    _pa_mqa_logits_mxfp4_sched_kernel[(triton.cdiv(target_wgs, SCHED_BLOCK_P),)](
        context_lens,
        out,
        batch,
        next_n,
        target_wgs,
        BLOCK_M=block_m,
        BLOCK_KV=block_kv,
        ROW_BLOCKS=row_blocks,
        ALIGN_W=align_w,
        BLOCK_P=SCHED_BLOCK_P,
        num_warps=4,
    )
    # The launcher takes the grid from the length, so the length is the contract
    return out[:target_wgs * 4]


def _check_schedule(schedule, device):
    if not isinstance(schedule, torch.Tensor):
        raise TypeError("schedule must be an int32 tensor from build_schedule, "
                        f"got {type(schedule).__name__}")
    schedule = schedule.reshape(-1)
    if schedule.dtype != torch.int32:
        raise TypeError(f"schedule must be int32, got {schedule.dtype}")
    if not schedule.is_contiguous():
        raise ValueError("schedule must be contiguous")
    if schedule.device != device:
        raise ValueError(f"schedule is on {schedule.device}, operands are on {device}")
    if schedule.numel() == 0 or schedule.numel() % 4:
        raise ValueError("schedule must be a whole number of 4-word descriptors, "
                         f"got {schedule.numel()} words")
    return schedule


def paged_mxfp4_mqa_logits(
    q: torch.Tensor,
    q_scales: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_table: torch.Tensor,
    max_model_len: int,
    kv_scale_cache: torch.Tensor | None = None,
    out_logits: torch.Tensor | None = None,
    clean_logits: bool = True,
    preshuffle: int = 1,
    dynamic: int = 0,
    schedule: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    This function computes the logits to be used by a topk function for sparse
    attention, from an MXFP4 query against a paged MXFP4 KV cache.

    q:              [B, NEXT_N, NUM_HEADS, HEAD_SIZE//2], dtype uint8, packed e2m1
    q_scales:       [B, NEXT_N, NUM_HEADS, HEAD_SIZE//32], dtype uint8, e8m0
    kv_cache:       [NUM_PAGES, PAGE_SIZE, 1, HEAD_SIZE//2 + HEAD_SIZE//32], dtype
                    uint8, each page holding its e2m1 values then its e8m0 scales
    weights:        [B * NEXT_N, NUM_HEADS], dtype float32
    context_lens:   [B], dtype int32
    block_table:    [B, MAX_BLOCKS], dtype int32, page index per sequence position
    max_model_len:  int, column count of the logits
    kv_scale_cache: [NUM_PAGES, PAGE_SIZE, 1, HEAD_SIZE//32], dtype uint8, when the
                    scales are kept apart from the values rather than packed after
                    them. kv_cache is then the values alone
    out_logits:     [B * NEXT_N, max_model_len], dtype float32, preallocated output.
                    Must arrive filled with -inf when clean_logits is True
    clean_logits:   bool. If True, positions row i does not attend to are explicitly
                    written as -inf. If False those positions are unspecified
    preshuffle:     bool. The cache is stored in dot-operand order (see
                    preshuffle_cache), which reads it straight into the matrix core.
                    If False it is read token-major and staged through LDS
    dynamic:        bool. Build the work schedule on the device, for a batch whose
                    sequences differ in length
    schedule:       [NUM_CTAS, 4], dtype int32, work descriptors from
                    build_schedule. Takes precedence over dynamic

    Returns:
    logits:         [B * NEXT_N, max_model_len], dtype float32

    """
    # Gluon kernel for gfx950 only for now
    assert arch_info.get_arch() == "gfx950", "gfx950 only"

    batch, next_n, num_heads, head_bytes = q.shape
    head_size = head_bytes * 2
    num_scales = head_size // SCALE_GROUP
    assert num_heads & (num_heads - 1) == 0, "head count must be a power of 2"
    assert head_size & (head_size - 1) == 0, "head size must be a power of 2"
    assert q.dtype == torch.uint8 and q_scales.dtype == torch.uint8
    assert q_scales.shape == (batch, next_n, num_heads, num_scales)
    assert q.stride()[2:] == (head_bytes, 1), "q must be row-contiguous over (H, D)"
    assert q_scales.stride()[2:] == (num_scales, 1)
    assert block_table.dtype == torch.int32 and block_table.stride(1) == 1
    assert block_table.shape[0] == batch
    assert context_lens.dtype == torch.int32
    assert weights.shape == (batch * next_n, num_heads) and weights.stride(1) == 1

    # page_size comes from the cache rather than the caller: both layouts pin it
    # exactly, and a second source could disagree with the bytes.
    if kv_scale_cache is None:
        values, scales, page_size, page_bytes = _split_cache(kv_cache, head_size)
        kv_page_stride = kvs_page_stride = page_bytes
        num_pages = kv_cache.shape[0]
    else:
        values, scales = kv_cache, kv_scale_cache
        num_pages = values.shape[0]
        page_size = values.reshape(num_pages, -1).shape[1] // head_bytes
        kv_page_stride = page_size * head_bytes
        kvs_page_stride = page_size * num_scales
    assert values.dtype == torch.uint8 and scales.dtype == torch.uint8
    assert kv_page_stride % 16 == 0, (
        f"value page stride ({kv_page_stride} B) must be 16-byte aligned or the "
        "loads cannot be vectorised")

    preshuffle = 1 if preshuffle else 0
    if preshuffle:
        cache_format(num_heads, head_size, page_size)  # validates the geometry

    if out_logits is None:
        shape = (batch * next_n, max_model_len)
        logits = (torch.full(shape, float("-inf"), dtype=torch.float32,
                             device=q.device) if clean_logits
                  else torch.empty(shape, dtype=torch.float32, device=q.device))
    else:
        logits = out_logits
        assert logits.shape == (batch * next_n, max_model_len)
    # A buffer store addresses the row through a 32-bit record count.
    assert max_model_len * 4 < 2 ** 31, (
        f"max_model_len {max_model_len} exceeds what a buffer store can address")

    cfg = select_config(num_heads, head_size, next_n, page_size, preshuffle,
                        clean_logits)
    block_m, row_blocks = cfg["block_m"], cfg["row_blocks"]
    block_kv, n_per_tile = cfg["block_kv"], cfg["n_per_tile"]
    target_wgs = cfg["target_wgs"]
    # page_size comes from the cache, so it can be a size BLOCK_KV does not fit.
    assert page_size % block_kv == 0, (
        f"BLOCK_KV {block_kv} must divide page_size {page_size} or a tile spans "
        "two pages, which are not adjacent")
    assert block_kv % n_per_tile == 0, (
        f"BLOCK_KV {block_kv} must be a multiple of the MFMA N ({n_per_tile})")

    use_buffer_load = bool(preshuffle) or (
        num_pages * max(kv_page_stride, kvs_page_stride) < 2 ** 31)

    # The two that need the batch, which select_config does not see.
    cfg["num_kv_splits"] = _kv_splits(
        batch, row_blocks, max_model_len, block_kv, target_wgs,
        cfg["min_tiles_per_split"], cfg["max_tiles_per_split"])
    # More than one row block per sequence means the second re-reads the same
    # pages, so the lines are worth keeping in L1
    cfg["kv_reread"] = 1 if row_blocks > 1 else 0

    num_kv_splits = cfg["num_kv_splits"]
    if schedule is None and dynamic:
        schedule = build_schedule(context_lens, next_n, num_heads, head_size,
                                  page_size, preshuffle)
    use_dynamic = schedule is not None
    if use_dynamic:
        schedule = _check_schedule(schedule, context_lens.device)
        # schedule length is the grid
        grid = (schedule.numel() // 4, 1, 1)
    else:
        grid = (row_blocks, batch, num_kv_splits)
        # Placeholder ptr
        schedule = context_lens

    _pa_mqa_logits_mxfp4_kernel[grid](
        Q_ptr=q,
        q_scales_ptr=q_scales,
        KV_ptr=values,
        kv_scales_ptr=scales,
        weights_ptr=weights,
        context_lens_ptr=context_lens,
        block_table_ptr=block_table,
        sched_ptr=schedule,
        logits_ptr=logits,
        next_n=next_n,
        num_kv_splits=num_kv_splits,
        stride_q_b=q.stride(0),
        stride_q_n=q.stride(1),
        stride_qs_b=q_scales.stride(0),
        stride_qs_n=q_scales.stride(1),
        stride_w_s=weights.stride(0),
        stride_logits_s=logits.stride(0),
        stride_logits_k=logits.stride(1),
        stride_blk_b=block_table.stride(0),
        max_blocks=block_table.shape[1],
        NUM_HEADS=num_heads,
        HEAD_SIZE=head_size,
        PAGE_SIZE=page_size,
        KV_PAGE_STRIDE=kv_page_stride,
        KVS_PAGE_STRIDE=kvs_page_stride,
        BLOCK_KV=block_kv,
        BLOCK_M=block_m,
        NUM_WARPS=cfg["num_warps"],
        NUM_BUFFERS=cfg["num_buffers"],
        DEPTH=cfg["depth"],
        UNROLL=cfg["unroll"],
        PAGE_PIPE=cfg["page_pipe"],
        M_CHUNK=cfg["m_chunk"],
        NUM_CHAINS=cfg["num_chains"],
        FOLD_ASM=cfg["fold_asm"],
        RELU_ADD=cfg["relu_add"],
        RELAXED_STORE=cfg["relaxed_store"],
        PRESHUFFLE=preshuffle,
        USE_BUFFER_LOAD=use_buffer_load,
        MFMA_NONK_DIM=n_per_tile,
        HAS_KV_SPLIT=1 if (num_kv_splits > 1 or use_dynamic) else 0,
        KV_REREAD=cfg["kv_reread"],
        DYNAMIC=int(use_dynamic),
        num_warps=cfg["num_warps"],
        waves_per_eu=cfg["waves_per_eu"],
    )
    return logits
