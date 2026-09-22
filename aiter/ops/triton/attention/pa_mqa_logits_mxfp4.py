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

# The preshuffled cache's e8m0 byte order. 1 is what ships: one MFMA tile per
# group with the scale axis innermost, read as one wide run. 0 is the narrow
# order, kept only so a gather can be measured against a cache it was not
# written for.
SCALE_MODE_WIDE = 1

# DeepSeek-V4.1's two-level indexer groups the context in 8-token candidate
# blocks, which is what a fused block score is a maximum over.
CANDIDATE_BLOCK = 8

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


def preshuffle_scales(x: torch.Tensor, n_per_tile: int,
                      scale_mode: int = SCALE_MODE_WIDE) -> torch.Tensor:
    """[P, page, D//32] e8m0 into the order the scale load reads.

    Mode 1, the default, groups one MFMA tile and puts the lane's run over the
    scale axis innermost -- both from num_heads and head_size, so the order
    carries no BLOCK_KV and no warp count -- and a whole tile is then one wide
    run. Mode 0 puts the token axis innermost instead, which costs the dense
    reader a byte-wide load per group and buys nothing; it exists so a candidate
    gather can be priced against a storage order that is not the shipping one.

    Both orders keep a token's scales at a constant stride inside a group, so
    both are addressable by a gather finer than the group.
    """
    # Anything not 0 falls through to mode 1 here while the kernel reads
    # anything not 1 as mode 0, so an out-of-range mode silently disagrees.
    assert scale_mode in (0, 1), "scale_mode must be 0 or 1"
    WARP_SIZE = 64
    p, rows, ns = x.shape
    if scale_mode == 0:
        return (x.reshape(p, rows // n_per_tile, n_per_tile, ns)
                .permute(0, 1, 3, 2).contiguous().reshape(p, rows, ns))
    s_lo = WARP_SIZE // n_per_tile
    s_hi = ns // s_lo
    return (x.reshape(p, rows // n_per_tile, n_per_tile, s_hi, s_lo)
            .permute(0, 1, 4, 2, 3).contiguous().reshape(p, rows, ns))


def unshuffle_scales(x: torch.Tensor, n_per_tile: int,
                     scale_mode: int = SCALE_MODE_WIDE) -> torch.Tensor:
    assert scale_mode in (0, 1), "scale_mode must be 0 or 1"
    WARP_SIZE = 64
    p, rows, ns = x.shape
    if scale_mode == 0:
        return (x.reshape(p, rows // n_per_tile, ns, n_per_tile)
                .permute(0, 1, 3, 2).contiguous().reshape(p, rows, ns))
    s_lo = WARP_SIZE // n_per_tile
    s_hi = ns // s_lo
    return (x.reshape(p, rows // n_per_tile, s_lo, n_per_tile, s_hi)
            .permute(0, 1, 3, 4, 2).contiguous().reshape(p, rows, ns))


def preshuffle_cache(values: torch.Tensor, scales: torch.Tensor,
                     num_heads: int, head_size: int,
                     scale_mode: int = SCALE_MODE_WIDE):
    """Natural order into the stored order. A cache-prep kernel should emit
    these bytes directly; this is the reference for what that means."""
    f = cache_format(num_heads, head_size, values.shape[1])
    return (preshuffle_values(values, f["n_per_tile"], f["d_per_tile"]),
            preshuffle_scales(scales, f["n_per_tile"], scale_mode))


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
    block_kv = kv_tile(page_size)
    # A page wider than the tile leaves a live in-page offset between the
    # block-table read and the KV address; a page equal to it folds that away.
    # Two rules below carry a term for it, and both are gated so the common
    # page_size == BLOCK_KV path is untouched.
    split_page = page_size > block_kv

    if preshuffle:
        # Register path. One warp: with no KV tile in LDS a second warp has no
        # producer/consumer to help with, only its barriers.
        block_m = min(plan_block_m(num_heads, next_n), next_n)
        cfg = dict(
            num_warps=1,
            num_buffers=1,
            waves_per_eu=3 if (compute_chunk or wide_decode) else 2,
            # A second KV tile in flight. Worth its registers where there
            # are registers to spare, and on a split page it also covers the
            # longer address chain -- 1.11x on decode there, a spill and 1.3x
            # the other way on a wide chunk, so decode only.
            depth=2 if (wide_decode or (next_n == 1 and split_page)) else 1,
            # Two KV tiles per body. Speculative decode has the registers for
            # it; a wide chunk does not -- at 64 heads it lands one over
            # waves_per_eu = 3's budget and spills seven words, which costs
            # 1.17-1.37x, and at 32 heads the second tile buys nothing.
            unroll=2 if (spec_rows and block_m <= 6) else 1,
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
        block_kv=block_kv,
        n_per_tile=n_per_tile,
        # Workgroups to fill the machine: 4 SIMDs per CU
        target_wgs=4 * get_num_sms() * cfg["waves_per_eu"],
        # Hoists the block-table read a tile past the KV prefetch. Off at 64
        # heads, where the extra live value costs more than it buys -- unless
        # the page is split, where it pays on decode and prefill but not on
        # speculative decode, which has rows of fold to hide the read behind.
        page_pipe=1 if (num_heads <= 32
                        or (split_page and not 2 <= next_n <= SPEC_ROWS)) else 0,
        m_chunk=n_per_tile if (num_heads > n_per_tile and n_per_tile == 32) else 0,
        num_chains=1,
        # Hoists the candidate-list read a tile past the KV prefetch, the way
        # page_pipe hoists the block-table read. Only meaningful under `gather`.
        gather_pipe=1,
        relaxed_store=0 if clean_logits else 1,
        # Follows FOLD_ASM rather than standing alone: without it the SLP
        # vectorizer pairs the adds into v_pk_add_f32, which has no abs modifier.
        relu_add=cfg["fold_asm"],
        min_tiles_per_split=4,
        # The balance term's cap on a workgroup's tiles. Decode wants it loose:
        # one row block per sequence leaves the occupancy term in charge, and
        # at a long context 64 splits past what occupancy asked for, which
        # costs 8-10% at 128K-365K. Above one query row the row blocks already
        # fill the machine, the occupancy term is small, and the same value
        # halves the split count and costs 1-14%.
        max_tiles_per_split=128 if next_n == 1 else 64)
    return cfg


def select_config(num_heads, head_size, next_n, page_size, preshuffle=1,
                  clean_logits=True):
    """The config for one shape, cached"""
    return dict(_select_config(num_heads, head_size, next_n, page_size,
                               int(bool(preshuffle)), bool(clean_logits)))


def build_schedule(context_lens, next_n, num_heads, head_size,
                   page_size=IDEAL_PAGE_SIZE, preshuffle=1, out=None,
                   cu_ends=None, gather=0):
    """Descriptors for one launch, or None if the shape does not fit.

    A descriptor is (sequence, row block, slice index, slice count), relative
    rather than absolute, so the kernel converts it against its own tile count.

    cu_ends only affects balance here. Pass the same gather flag the launch
    uses, or the slot counts are read as key positions.
    """
    plan = select_config(num_heads, head_size, next_n, page_size, preshuffle)
    row_blocks, block_m = plan["row_blocks"], plan["block_m"]
    if gather:
        row_blocks, block_m = next_n, 1
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
        cu_ends,
        out,
        batch,
        next_n,
        target_wgs,
        BLOCK_M=block_m,
        BLOCK_KV=block_kv,
        ROW_BLOCKS=row_blocks,
        ALIGN_W=align_w,
        BLOCK_P=SCHED_BLOCK_P,
        HAS_CU_ENDS=1 if cu_ends is not None else 0,
        GATHER=int(gather),
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
    scale_mode: int = SCALE_MODE_WIDE,
    dynamic: int = 0,
    schedule: torch.Tensor | None = None,
    cu_ends: torch.Tensor | None = None,
    gather: dict | None = None,
    block_scores: torch.Tensor | None = None,
    block_scores_only: bool = False,
    pin_newest: bool = False,
    candidate_block_size: int = CANDIDATE_BLOCK,
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
    cu_ends:        [B * NEXT_N], dtype int32, optional. Exclusive per-row key
                    bound, indexed like weights, defaulting to
                    context_lens[b] - NEXT_N + n + 1; pass it under compression
                    or context parallelism. build_schedule needs the same tensor.
                    Under gather it counts the row's valid candidate slots
                    instead, the list having been causally filtered already
    gather:         what build_gather returns. The walk then runs over a
                    host-resolved candidate list rather than the context, and
                    output column j holds candidate slot j. context_lens and
                    block_table are unread on this path
    block_scores:   [B * NEXT_N, ceil(max_model_len / candidate_block_size)],
                    dtype float32, optional. The two-level indexer's stage-A
                    block scores, fused into this walk: block b of row i gets
                    the maximum of that row's logits over columns
                    [b*C, (b+1)*C), with the columns the row cannot attend to
                    counted as -inf. Must arrive filled with -inf; blocks past
                    the row's walk are left untouched. Emitted *unpinned* --
                    upstream's +inf on the block holding the row's newest key
                    is one element per row and belongs in a caller-side
                    scatter, not in a compare per block inside the walk. Not
                    available under gather: the producer walks densely
    pin_newest:     bool. Force the block holding each row's newest key to +inf, so
                    recent context is always a candidate whatever it scored. The
                    reference model does this; leave it off to score honestly
    block_scores_only: bool. Emit the block maxima *instead of* the logits
                    rather than beside them: the logits store and its offset
                    and predicate arithmetic leave the walk, and no
                    [B * NEXT_N, max_model_len] tensor is allocated or
                    written. The maxima are bit-identical to what the same
                    launch produces with this off -- only the store is gone.
                    Requires block_scores. See below for who wants it, what it
                    buys, and what it hands back
    candidate_block_size: int, columns per candidate block. Must divide
                    BLOCK_KV so no block straddles a tile or a KV split

    Returns:
    logits:         [B * NEXT_N, max_model_len], dtype float32 -- or, under
                    block_scores_only, the caller's block_scores tensor

    On block_scores_only
    --------------------
    It has one consumer: pass 1 of the two-pass producer. No indexer layer can
    use it alone, because every layer needs its own top-k and that needs
    logits. The producer layer runs this, ranks the maxima into a candidate
    pool, and then runs a second `gather` launch over that pool to recover its
    own top-k -- which is exact, not approximate: a top-2048-of-8 pool
    provably contains the layer's own top-512.

    What it buys is the tensor that never exists. At 512 rows x 365K context
    the dense fp32 logits are 748 MB neither allocated nor stored, and that
    re-bases vLLM's prefill sub-chunker: it sizes a sub-chunk as
    M = budget / 4 / N, and what a layer stores is the only thing that sets N,
    so emitting [M, ctx/C] instead of [M, ctx] gives C times more rows per
    sub-chunk -- 8x at C = 8, which is 23 launches down to 3 at 365K.

    out_logits and clean_logits are rejected rather than ignored, both being
    statements about a logits tensor this path does not have.

    The return value is a deliberate ABI choice, not a fallout: the function
    hands back the caller's `block_scores` tensor. It is the only output there
    is, and returning it keeps `out = paged_mxfp4_mqa_logits(...)` meaning
    "the thing this launch produced" in all four modes. The alternative --
    returning None so that the sole output is unambiguously the out-param the
    caller already holds -- was rejected because it makes the call site's shape
    depend on a keyword flag.

    """
    # Gluon kernel for gfx950 only for now
    assert arch_info.get_arch() == "gfx950", "gfx950 only"
    assert scale_mode in (0, 1), "scale_mode must be 0 or 1"

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
    if cu_ends is not None:
        assert cu_ends.dtype == torch.int32, "cu_ends must be int32"
        assert cu_ends.shape == (batch * next_n,) and cu_ends.stride(0) == 1, (
            "cu_ends must be a contiguous [B * NEXT_N] vector, indexed like weights")

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

    # The fused stage-A reduce, as three values of one knob rather than two
    # flags with a forbidden corner: 0 off, 1 the maxima beside the logits,
    # 2 the maxima instead of them. The shape checks that need BLOCK_KV are
    # below; what is resolved here is whether a logits tensor exists at all.
    bscore_on = 0 if block_scores is None else (2 if block_scores_only else 1)
    assert not pin_newest or bscore_on, "pin_newest writes a block score"
    if block_scores_only:
        assert block_scores is not None, (
            "block_scores_only needs the block_scores tensor it writes; it is "
            "the launch's only output")
        # Both of these are statements about a logits tensor, and on this path
        # there is not one. Rejected rather than ignored: a caller that passed
        # out_logits expecting it filled would get silence.
        assert out_logits is None, (
            "out_logits is inapplicable under block_scores_only: no logits are "
            "written")
        assert clean_logits, (
            "clean_logits is inapplicable under block_scores_only: it asks how "
            "to write the positions of a logits tensor this walk never writes")

    if bscore_on == 2:
        # The point of the mode. At 512 rows x 365K context this is 748 MB
        # neither allocated nor written, and it is what lets the caller's
        # sub-chunker size itself on ceil(ctx / C) columns instead of ctx.
        logits = None
    elif out_logits is None:
        shape = (batch * next_n, max_model_len)
        logits = (torch.full(shape, float("-inf"), dtype=torch.float32,
                             device=q.device) if clean_logits
                  else torch.empty(shape, dtype=torch.float32, device=q.device))
    else:
        logits = out_logits
        assert logits.shape == (batch * next_n, max_model_len)
    if logits is not None:
        # A buffer store addresses the row through a 32-bit record count.
        assert max_model_len * 4 < 2 ** 31, (
            f"max_model_len {max_model_len} exceeds what a buffer store can "
            "address")

    # A candidate gather. `gather` is what build_gather returns: two int32
    # [B*next_n, blocks] tensors of already-resolved offsets -- page address
    # plus the block's own offset inside its page. The kernel adds one of those
    # per column and nothing else; there is no block-table read on this path.
    # The walk length is the row's valid slot count, which arrives as cu_ends.
    gather_on = 1 if gather is not None else 0
    gather_block = int(gather["block"]) if gather_on else 8
    if gather_on:
        g_voff, g_soff = gather["voff"], gather["soff"]
        assert g_voff.dtype == torch.int32 and g_soff.dtype == torch.int32
        assert g_voff.shape == g_soff.shape, (g_voff.shape, g_soff.shape)
        assert g_voff.stride(1) == 1 and g_soff.stride(1) == 1
        assert g_voff.stride(0) == g_soff.stride(0)
        assert g_voff.shape[0] == batch * next_n, g_voff.shape
        assert cu_ends is not None, (
            "the gather takes its walk length from cu_ends, read as the row's "
            "count of valid candidate slots")

    cfg = select_config(num_heads, head_size, next_n, page_size, preshuffle,
                        clean_logits)
    if gather_on:
        # One query row per workgroup: above one row the walk is over the
        # rows' union and each row's store column comes from a per-block slot,
        # none of which the candidate addressing carries.
        #
        # The dense walk's two pipeline knobs invert here, both for the same
        # reason: BLOCK_M is 1 rather than 3, and the candidate list carries
        # the whole page address.
        #
        # DEPTH keeps only its split-page term, and only below 64 heads. The
        # wide-decode term pays the dense walk because a second tile in flight
        # covers the block-table read, and the gather does no such read -- so
        # there it only costs registers, and at 64 heads it costs enough of
        # them to push the scale load back to bytes. Pinning it to 1 at 64
        # heads is 1.01-1.06x on its own and 1.19-1.29x once that spill is
        # counted. A page wider than the tile is the one place a second tile
        # still pays, 1.03x at decode concurrency 128, measured rather than
        # explained by the candidate addressing.
        #
        # UNROLL goes the other way on a wide chunk: one query row leaves the
        # registers the dense path at BLOCK_M = 3 does not have, and four
        # tiles of loads in flight is worth 1.03-1.06x with no spill. Decode
        # keeps what the dense rule gave it -- the walk is short enough there
        # that the peeled remainder costs more than the extra loads buy.
        split_page = page_size > cfg["block_kv"]
        cfg = dict(cfg, block_m=1, row_blocks=next_n,
                   depth=2 if (next_n == 1 and split_page and num_heads <= 32)
                   else 1)
        if preshuffle and next_n >= COMPUTE_CHUNK:
            cfg["unroll"] = 4
    block_m, row_blocks = cfg["block_m"], cfg["row_blocks"]
    block_kv, n_per_tile = cfg["block_kv"], cfg["n_per_tile"]
    target_wgs = cfg["target_wgs"]
    # page_size comes from the cache, so it can be a size BLOCK_KV does not fit.
    assert page_size % block_kv == 0, (
        f"BLOCK_KV {block_kv} must divide page_size {page_size} or a tile spans "
        "two pages, which are not adjacent")
    assert block_kv % n_per_tile == 0, (
        f"BLOCK_KV {block_kv} must be a multiple of the MFMA N ({n_per_tile})")

    # The fused stage-A reduce. C divides BLOCK_KV, so a tile owns a whole
    # number of candidate blocks and a KV split -- cut at BLOCK_KV granularity
    # -- never splits one, which is what makes the block max purely local.
    cand_block = int(candidate_block_size)
    if bscore_on:
        assert gather is None, (
            "block maxima are for the dense producer; the consumers gather")
        assert block_scores.dtype == torch.float32
        assert cand_block <= block_kv and block_kv % cand_block == 0, (
            f"candidate_block_size {cand_block} must divide BLOCK_KV "
            f"{block_kv} or a block straddles a tile")
        n_blocks = (max_model_len + cand_block - 1) // cand_block
        assert block_scores.shape[0] == batch * next_n, block_scores.shape
        assert block_scores.shape[1] >= n_blocks, (
            f"block_scores is {block_scores.shape[1]} blocks wide, needs "
            f"{n_blocks} for max_model_len {max_model_len}")
        assert block_scores.stride(1) == 1, "block_scores rows must be contiguous"
        assert block_scores.shape[1] * 4 < 2 ** 31, (
            "block_scores row exceeds what a buffer store can address")

    use_buffer_load = bool(preshuffle) or (
        num_pages * max(kv_page_stride, kvs_page_stride) < 2 ** 31)
    if gather_on:
        assert block_kv % gather_block == 0 and page_size % gather_block == 0
        assert gather_block <= n_per_tile, (
            "a candidate block must sit inside one shuffle group")
        # The whole address is in the offsets here, so a cache past 2 GiB drops
        # to 64-bit the way every other offset-addressed path does. The value
        # list is in k_width units, so only the scale stream is capped.
        use_buffer_load = num_pages * max(kv_page_stride, kvs_page_stride) < 2 ** 31

    # The two that need the batch, which select_config does not see.
    cfg["num_kv_splits"] = _kv_splits(
        batch, row_blocks, max_model_len, block_kv, target_wgs,
        cfg["min_tiles_per_split"], cfg["max_tiles_per_split"])
    # More than one row block per sequence means the second re-reads the same
    # pages, so the lines are worth keeping in L1
    cfg["kv_reread"] = 1 if row_blocks > 1 else 0

    num_kv_splits = cfg["num_kv_splits"]
    # A gather launch does not build one. Every row walks the same
    # ceil(slots / BLOCK_KV) tiles, so there is no spread for a slice plan to
    # even out, and the build costs about half a launch. A dynamic=1 set for the
    # step must not drag a consumer layer in. An explicit schedule is still
    # honoured, and build_schedule takes the same gather flag so its tile count
    # comes out of the slot counts.
    if schedule is None and dynamic and not gather_on:
        schedule = build_schedule(context_lens, next_n, num_heads, head_size,
                                  page_size, preshuffle, cu_ends=cu_ends)
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
        # None specializes to a constexpr, so an unused argument leaves no trace.
        cu_ends_ptr=cu_ends,
        block_table_ptr=block_table,
        sched_ptr=schedule,
        gather_v_ptr=g_voff if gather_on else schedule,
        gather_s_ptr=g_soff if gather_on else schedule,
        # None specializes to a constexpr, so neither reaches the kernarg
        # segment with the reduce off.
        block_scores_ptr=block_scores if bscore_on else None,
        logits_ptr=logits,
        next_n=next_n,
        num_kv_splits=num_kv_splits,
        stride_q_b=q.stride(0),
        stride_q_n=q.stride(1),
        stride_qs_b=q_scales.stride(0),
        stride_qs_n=q_scales.stride(1),
        stride_w_s=weights.stride(0),
        stride_logits_s=logits.stride(0) if logits is not None else 0,
        stride_logits_k=logits.stride(1) if logits is not None else 0,
        stride_blk_b=block_table.stride(0),
        stride_gather_r=g_voff.stride(0) if gather_on else 0,
        stride_bs_s=block_scores.stride(0) if bscore_on else None,
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
        SCALE_MODE=int(scale_mode),
        USE_BUFFER_LOAD=use_buffer_load,
        MFMA_NONK_DIM=n_per_tile,
        HAS_KV_SPLIT=1 if (num_kv_splits > 1 or use_dynamic) else 0,
        KV_REREAD=cfg["kv_reread"],
        DYNAMIC=int(use_dynamic),
        HAS_CU_ENDS=1 if cu_ends is not None else 0,
        GATHER=gather_on,
        GATHER_BLOCK=gather_block,
        GATHER_PIPE=cfg["gather_pipe"] if gather_on else 0,
        BSCORE=bscore_on,
        BSCORE_BLOCK=cand_block if bscore_on else CANDIDATE_BLOCK,
        PIN_NEWEST=int(pin_newest),
        num_warps=cfg["num_warps"],
        waves_per_eu=cfg["waves_per_eu"],
    )
    # An ABI choice, argued in the docstring: block_scores is the only output
    # this mode has, so it is what comes back.
    return block_scores if bscore_on == 2 else logits
