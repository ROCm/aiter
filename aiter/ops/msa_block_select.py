# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import math

from aiter.jit.core import compile_ops
from aiter.jit.utils.chip_info import get_cu_num


@compile_ops(
    "module_msa_sparse_attention",
    fc_name="pa_sparse_block_score_decode",
    ffi_type="ctypes",
)
def _score_decode_raw(
    q_idx: int,
    key_cache_idx: int,
    score: int,
    block_table: int,
    seq_lens: int,
    num_reqs: int,
    num_q_tiles: int,
    block_table_stride: int,
    score_head_stride: int,
    score_num_stride: int,
    num_chunks: int,
    init_blocks: int,
    local_blocks: int,
    query_len: int,
    rank: int,
    world: int,
    block_size: int,
    head_dim: int,
    num_idx_heads: int,
    num_waves: int,
    kv_fmt: int,
    page_stride: int,
    q_tiles: int,
    req_xor_mask: int,
    fill_shift: int,
) -> None: ...


@compile_ops(
    "module_msa_sparse_attention",
    fc_name="pa_sparse_block_score_prefill",
    ffi_type="ctypes",
)
def _score_prefill_raw(
    q_idx: int,
    key_cache_idx: int,
    score: int,
    block_table: int,
    cu_seqlens_q: int,
    seq_lens: int,
    num_reqs: int,
    num_q_groups: int,
    block_table_stride: int,
    score_head_stride: int,
    score_num_stride: int,
    num_chunks: int,
    chunk_blocks: int,
    init_blocks: int,
    local_blocks: int,
    block_size: int,
    head_dim: int,
    num_idx_heads: int,
    num_waves: int,
    q_tiles: int,
) -> None: ...


@compile_ops(
    "module_msa_sparse_attention",
    fc_name="pa_sparse_block_score_prefill_nvfp4",
    ffi_type="ctypes",
)
def _score_prefill_nvfp4_raw(
    q_idx: int,
    key_cache_idx: int,
    score: int,
    block_table: int,
    cu_seqlens_q: int,
    seq_lens: int,
    num_reqs: int,
    num_q_groups: int,
    block_table_stride: int,
    score_head_stride: int,
    score_num_stride: int,
    num_chunks: int,
    chunk_blocks: int,
    init_blocks: int,
    local_blocks: int,
    page_stride: int,
    num_idx_heads: int,
    wave_tiles: int,
    store_pages: int,
    group_xor: int,
    req_rot: int,
    xcds: int,
    req_chunks: int,
    grid_chunks: int,
) -> None: ...


@compile_ops(
    "module_msa_sparse_attention", fc_name="pa_sparse_block_topk", ffi_type="ctypes"
)
def _topk_raw(
    score: int,
    topk_idx: int,
    seq_lens: int,
    num_valid_pages: int,
    block_table: int,
    row_req_id: int,
    kv_lens: int,
    sparse_bt: int,
    sparse_ctx: int,
    num_idx_heads: int,
    total_q: int,
    score_head_stride: int,
    score_num_stride: int,
    topk_head_stride: int,
    topk_num_stride: int,
    block_table_stride: int,
    sparse_bt_stride: int,
    num_kv_heads: int,
    query_len: int,
    block_size: int,
    topk: int,
    slots: int,
    num_waves: int,
    pages_per_block: int,
) -> None: ...


@compile_ops(
    "module_msa_sparse_attention",
    fc_name="pa_sparse_block_topk_cp",
    ffi_type="ctypes",
)
def _topk_cp_raw(
    score: int,
    peer_cand_ptrs: int,
    cp_gen: int,
    topk_idx: int,
    seq_lens: int,
    block_table: int,
    sparse_bt: int,
    sparse_ctx: int,
    num_idx_heads: int,
    total_q: int,
    cand_rows: int,
    score_head_stride: int,
    score_num_stride: int,
    topk_head_stride: int,
    topk_num_stride: int,
    block_table_stride: int,
    sparse_bt_stride: int,
    num_kv_heads: int,
    query_len: int,
    init_blocks: int,
    local_blocks: int,
    rank: int,
    world: int,
    block_size: int,
    topk: int,
    slots: int,
    num_waves: int,
) -> None: ...


def _ptr(t) -> int:
    """Device address of a tensor; 0 is the null the kernels test for."""
    return 0 if t is None else t.data_ptr()


WAVE_SIZE = 64

# Waves one top-k workgroup can put on a single row.
TOPK_MAX_WAVES = 8

# Slots -- 64-wide strips of the score row -- one lane can hold. Sets the
# largest context the top-k pass will compile for.
TOPK_MAX_SLOTS = 128

# Slots x rows past which splitting a row across waves stops paying, because the
# rows alone already fill the machine.
TOPK_WAVE_BUDGET = 2048

# Candidates one selector workgroup holds, i.e. kTopkMaxCand. Caps the merge:
# every shard's whole top-k has to fit at once.
TOPK_MAX_CAND = 256

SCORE_WAVES = 4

SCORE_MAX_WORKGROUPS = 16384

# Workgroups the score split aims for before it stops folding waves into one.
# Only consulted on a partitioned block axis; see _score_split.
SCORE_MIN_WORKGROUPS = 2048

# nvfp4 decode grid: at most this many workgroups, unless that leaves a
# workgroup more than _SCORE_DECODE_MAX_WG_BLOCKS blocks of the bound.
_SCORE_DECODE_MAX_WORKGROUPS = 4096
_SCORE_DECODE_MAX_WG_BLOCKS = 16

# fill_shift that keeps the bound's split (see _decode_fill_shift).
_DECODE_FILL_OFF = 31

# Columns of the MFMA, each holding one (token, head) pair.
SCORE_MFMA_COLS = 16

# Query tiles one nvfp4 decode wave scores per read of the keys.
_SCORE_DECODE_MAX_Q_TILES = 2

# Ceiling on the pages one chunk of the ragged kernel's block axis covers. Only a
# ceiling: the size itself is derived below, since a fixed 16 collapses the grid
# on a short context. Past this the chunk stops being worth entering as a unit and
# the causal imbalance inside it stops being broken up.
SCORE_PREFILL_MAX_CHUNK_BLOCKS = 32

# Query groups the block axis wants per page of context before it stops splitting.
# The chunk size falls out of this: a group's cost runs with its last token's
# causal reach, so the block axis is what breaks the long ones up, and how far it
# has to go depends on how many groups are already there to fill the machine.
# Tracked the measured optimum from 1k to 64k across batch 1 to 32.
SCORE_PREFILL_GROUPS_PER_BLOCK = 64

SCORE_MAX_Q_TILES = 4

SCORE_PREFILL_MAX_Q_TILES = 4

# Index key formats (kKvFp8 / kKvNvfp4). A uint8 key cache holds nvfp4 pages.
KV_FMT_FP8 = 0
KV_FMT_NVFP4 = 1

# Block size and head dim of nvfp4 pages.
NVFP4_BLOCK_SIZE = 128
NVFP4_HEAD_DIM = 128

NVFP4_INDEX_HEADS = (1, 2, 4)

# nvfp4 prefill plan (_score_prefill_nvfp4_plan). A workgroup uses 38 KiB of
# LDS, so at most 4 fit on one CU.
_NVFP4_PREFILL_TARGET_WG_PER_CU = 2.75
_NVFP4_PREFILL_MAX_CHUNK_BLOCKS = 8
_NVFP4_PREFILL_BATCH_WG_PER_CU = 32
_NVFP4_PREFILL_BATCH_CHUNKS = 128
_NVFP4_PREFILL_WIDE_MIN_KEYS = 32768
_NVFP4_PREFILL_WIDE_MIN_WORK = 80 << 20
_NVFP4_PREFILL_WIDE_MIN_BATCH_WORK = 480 << 20
_NVFP4_PREFILL_NT2_MAX_WORK = 48 << 20
_NVFP4_PREFILL_XOR_ROT_MIN_REQS = 3
_NVFP4_PREFILL_XOR_ROT_MAX_QUERY_LEN = 8192
# One request: the XCD order needs, per row of (max query groups,
# min prefix / query len), a long enough prefix.
_NVFP4_PREFILL_XCD_RULES = ((14, 4), (20, 8))


def nvfp4_page_bytes(block_size: int, head_dim: int) -> int:
    """Bytes of one nvfp4 page: packed e2m1 values, then one e4m3 scale per
    16 values."""
    return block_size * head_dim // 2 + block_size * head_dim // 16


def _num_xcds() -> int:
    # Imported here: aiter.ops.triton is not loaded at aiter import time.
    from aiter.ops.triton.utils.device_info import get_num_xcds

    return get_num_xcds()


def _decode_q_tiles(num_tiles: int, num_reqs: int) -> int:
    """Query tiles per nvfp4 decode wave. One request keeps one tile: the
    longer wave costs more at short context than it saves at long context."""
    if num_reqs < 2:
        return 1
    return min(_SCORE_DECODE_MAX_Q_TILES, num_tiles)


def _pow2_floor(n: int) -> int:
    return 1 << (n.bit_length() - 1) if n >= 1 else 1


def _pow2_ceil(n: int) -> int:
    return 1 << (n - 1).bit_length() if n >= 1 else 1


def _topk_waves(slots: int, rows: int) -> int:
    """Waves to split one top-k row across.

    Splitting shortens each thread's share of the row but adds it to the barriers
    between narrowing passes, so it pays only while there are slots left to
    divide and the rows alone have not already filled the machine. Halving is the
    floor because a wave that ends up with one slot has nothing left to scan and
    pays the barriers for nothing.
    """
    return _pow2_floor(
        min(slots // 2, TOPK_WAVE_BUDGET // max(1, rows), TOPK_MAX_WAVES)
    )


def _shard_topk_waves(slots: int) -> int:
    """Waves to split one shard's row across.

    Every slot there are, where the whole-row selector stops at half of them and
    then divides by the rows. Neither of its two bounds carries over: a wave
    here selects over its own slots in registers rather than through the shared
    histogram, so one left holding a single slot still pays nothing for the
    privilege, and the rows cannot be traded against it because a shard row is
    ``1/world`` as long and short of filling the machine on its own.

    The row bound is the one that costs: past 1024 rows it pins the whole-row
    rule to two waves at every width, which measures 1.32x slow on a 256-block
    shard and 1.34x on a 512-block one. Dropping the halving is worth a further
    1.27x at 256 blocks and costs 0.1us at 512, the only width where taking
    every slot is not the fastest choice.
    """
    return _pow2_floor(min(slots, TOPK_MAX_WAVES))


def _score_split(max_blk: int, num_reqs: int, sharded: bool = False) -> tuple[int, int]:
    """Chunks along the block axis and waves per workgroup for the score pass.

    The pass is a pure stream of the index key cache, so what it wants is one
    wave per block in flight: every wave holds one block's loads open and there
    is nothing to reuse between them. Splitting to exactly that point tracked
    the measured optimum across the batch and context range, and notably does
    not depend on the batch -- rows contribute parallelism of their own, but
    starving the block axis to compensate only lengthens each wave's serial run.
    Both are launch dimensions, so both are fixed at capture.

    The batch does come back in at the ceiling. ``max_blk`` follows the caller's
    upper bound on the context rather than the one being served, so on a batch
    whose rows sit far below that bound the split runs past every block there is
    and the surplus workgroups only read a length and exit. Since the grid is
    num_reqs x num_chunks, the ceiling belongs on the product.

    ``sharded`` says ``max_blk`` is one rank's shard rather than the whole
    context, which is the one case the wave floor gets in the way. The waves a
    request needs is num_reqs x max_blk whichever way the two are split, so the
    only thing the split moves is how many CUs they land on -- a workgroup's
    waves all share one. At 1/P of the blocks the default 4 leaves a quarter of
    the workgroups it would otherwise have, and below a few thousand that is
    the difference between covering the machine and sitting on a corner of it,
    so fold the waves back down until the grid is wide enough again. Left off
    on the whole axis, where the measured tuning above already holds.
    """
    waves = min(SCORE_WAVES, _pow2_floor(max(1, max_blk)))
    if sharded:
        target = max(1, SCORE_MIN_WORKGROUPS // max(1, num_reqs))
        waves = min(waves, _pow2_floor(max(1, max_blk // target)))
    chunks = max(1, max_blk // waves)
    return max(1, min(chunks, SCORE_MAX_WORKGROUPS // max(1, num_reqs))), waves


def _score_split_decode_nvfp4(
    max_blk: int, num_reqs: int, sharded: bool = False
) -> tuple[int, int]:
    """_score_split capped at _SCORE_DECODE_MAX_WORKGROUPS workgroups, but at
    most _SCORE_DECODE_MAX_WG_BLOCKS blocks of the bound per workgroup."""
    chunks, waves = _score_split(max_blk, num_reqs, sharded)
    cap = max(
        _SCORE_DECODE_MAX_WORKGROUPS // max(1, num_reqs),
        math.ceil(max_blk / _SCORE_DECODE_MAX_WG_BLOCKS),
    )
    return max(1, min(chunks, cap)), waves


def _decode_fill_shift(num_reqs: int) -> int:
    """log2(CUs per request), rounded up.

    A short context gets one block per chunk from the bound-sized split, so
    most waves idle. decode_prologue widens such chunks to nblk >> fill_shift
    blocks, at most one per wave, which keeps about one workgroup per CU.
    """
    return (-(-get_cu_num() // max(1, num_reqs)) - 1).bit_length()


def _decode_req_xor_mask(num_reqs: int) -> int:
    """XOR mask of the request index (kDecodeReqXorShift). It spreads a long
    request's chunks over the XCDs; mask + 1 divides num_reqs."""
    return min(num_reqs & -num_reqs, _pow2_floor(_num_xcds())) - 1


def _tokens_per_tile(num_idx_heads: int) -> int:
    """Query tokens one ragged-pass workgroup covers. The MFMA's columns hold one
    (token, head) pair each, so this is what the heads leave behind."""
    return max(1, SCORE_MFMA_COLS // num_idx_heads)


def _resolve_q_tiles_prefill(num_tiles: int, num_reqs: int, q_waves: int) -> int:
    """Query tiles per *wave* for the ragged pass's own kernel.

    Takes the cap whenever the tiles are there to fill it: the page is staged
    once for the whole workgroup, so folding costs registers and nothing else,
    and the block-axis split puts the parallelism back. What it will not do is
    fold past the tiles that exist, since a wave handed none exits and takes its
    share of the workgroup's span with it.
    """
    return _pow2_floor(min(SCORE_PREFILL_MAX_Q_TILES, max(1, num_tiles // q_waves)))


def _score_split_prefill(max_blk: int, num_groups: int) -> tuple[int, int]:
    """Chunk count and size along the block axis for the ragged kernel.

    Unlike the shared kernel this picks the chunk's *size* and lets the count
    follow, because a fixed size is what makes chunk z the same pages for every
    query group that reaches it. Cut into a fixed number of pieces instead, each
    group's pieces land somewhere slightly different, nothing lines up, and every
    L2 ends up holding a different part of a working set that would fit several
    times over in any one of them.

    What the size cannot be is a constant. Held at 16 it is the query groups that
    decide the grid, and a short context has few of them: 2k over 8 requests comes
    out at one chunk and 128 workgroups, which is an eighth of the machine and
    measured 3.8x off its own best point. So derive it from the groups there are,
    leaving the size the same for all of them and only the count varying with the
    context. Falls back to the fixed ceiling once the groups alone are plentiful,
    which is where the constant was measured in the first place.
    """
    blocks = _pow2_floor(max(1, num_groups // SCORE_PREFILL_GROUPS_PER_BLOCK))
    blocks = min(blocks, SCORE_PREFILL_MAX_CHUNK_BLOCKS, max(1, max_blk))

    # The same grid ceiling the uniform pass keeps, reached from the other side:
    # the size is the knob here, so widen it until num_groups x num_chunks fits.
    # Widening rather than dropping chunks is what holds the size the same for
    # every group, which is the alignment the paragraph above rests on.
    cap = max(1, SCORE_MAX_WORKGROUPS // max(1, num_groups))
    blocks = max(blocks, _pow2_ceil(math.ceil(max_blk / cap)))
    return max(1, math.ceil(max_blk / blocks)), blocks


def _score_split_prefill_nvfp4(max_blk: int, num_groups: int) -> tuple[int, int]:
    """Chunk count and size along the block axis for the nvfp4 prefill kernel:
    the largest power-of-two chunk up to _NVFP4_PREFILL_MAX_CHUNK_BLOCKS that
    keeps _prefill_nvfp4_target_workgroups() workgroups."""
    target = _prefill_nvfp4_target_workgroups()
    blocks = 1
    while (
        blocks < _NVFP4_PREFILL_MAX_CHUNK_BLOCKS
        and num_groups * math.ceil(max_blk / (2 * blocks)) >= target
    ):
        blocks *= 2
    cap = max(1, SCORE_MAX_WORKGROUPS // max(1, num_groups))
    blocks = max(blocks, _pow2_ceil(math.ceil(max_blk / cap)))
    return max(1, math.ceil(max_blk / blocks)), blocks


def _prefill_nvfp4_group_rows(wave_tiles: int) -> int:
    """Query rows of one nvfp4 prefill workgroup (kGroupTokens at H=1): 4 waves
    of wave_tiles MFMA tiles."""
    return SCORE_WAVES * SCORE_MFMA_COLS * wave_tiles


def _prefill_nvfp4_target_workgroups() -> int:
    return int(get_cu_num() * _NVFP4_PREFILL_TARGET_WG_PER_CU)


def _prefill_nvfp4_req_chunks(num_groups: int) -> int:
    """Chunks per request that reach the target workgroup count."""
    return -(-_prefill_nvfp4_target_workgroups() // max(1, num_groups))


def _prefill_nvfp4_grid_chunks(max_blk: int, num_groups: int, num_chunks: int) -> int:
    """Block-axis grid size of the nvfp4 prefill kernel with several requests.

    The kernel repeats the chunk rule on each request's own length, so a short
    request takes at most 2 * (req_chunks - 1) smaller chunks. The grid is
    capped at _NVFP4_PREFILL_BATCH_WG_PER_CU workgroups per CU, but keeps at
    least _NVFP4_PREFILL_BATCH_CHUNKS chunks.
    """
    batch_wg = get_cu_num() * _NVFP4_PREFILL_BATCH_WG_PER_CU
    cap = max(_NVFP4_PREFILL_BATCH_CHUNKS, batch_wg // max(1, num_groups))
    small = 2 * (_prefill_nvfp4_req_chunks(num_groups) - 1)
    return max(min(num_chunks, cap), min(max_blk, small))


def _prefill_nvfp4_order(
    num_reqs: int,
    num_q_groups: int,
    chunk_blocks: int,
    max_blk: int,
    max_query_len: int,
    total_q: int,
    wave_tiles: int,
) -> tuple[int, int, int, int]:
    """``(store_pages, group_xor, req_rot, xcds)`` of the nvfp4 prefill kernel
    (see nvfp4_prefill_args).

    When groups x requests is a multiple of the XCD count, the plain order puts
    every chunk of a request on one XCD; the rotated and XOR orders spread them.
    """
    xcds = _num_xcds()
    if num_reqs == 1:
        if wave_tiles != 8:
            return 1, 0, 0, 0
        prefix = max_blk * NVFP4_BLOCK_SIZE - max_query_len
        if any(
            num_q_groups <= groups and prefix >= ratio * max_query_len
            for groups, ratio in _NVFP4_PREFILL_XCD_RULES
        ):
            return 2, 0, 0, xcds
        return (2 if chunk_blocks >= 2 else 1), 0, 0, 0
    if (num_q_groups * num_reqs) % xcds:
        return 1, 0, 0, 0
    if total_q >= num_reqs * max_query_len:
        return 1, 0, int(wave_tiles == 8), 0
    xor_groups = min(num_q_groups & -num_q_groups, _pow2_floor(xcds))
    if (
        num_reqs >= _NVFP4_PREFILL_XOR_ROT_MIN_REQS
        and max_query_len <= _NVFP4_PREFILL_XOR_ROT_MAX_QUERY_LEN
        and (
            (wave_tiles == 8 and xor_groups >= 2)
            or (wave_tiles == 4 and xor_groups >= 4)
        )
    ):
        return 1, xor_groups - 1, 1, 0
    return 1, 0, 1, 0


def _prefill_nvfp4_wave_tiles(
    num_reqs: int, max_query_len: int, max_blk: int, total_q: int
) -> int:
    """Query tiles of 16 rows per wave of the nvfp4 prefill kernel: 8, 4 or 2.

    8 tiles stage each page once for 512 rows and pay off only with long keys
    and enough query rows; otherwise 2 tiles while the query fits one 2-tile
    group or up to _NVFP4_PREFILL_NT2_MAX_WORK, and 4 above it.
    """
    keys = max_blk * NVFP4_BLOCK_SIZE
    work = max_query_len * keys
    short = max_query_len <= _prefill_nvfp4_group_rows(2)
    if not short and keys >= _NVFP4_PREFILL_WIDE_MIN_KEYS:
        if num_reqs == 1:
            wide = work >= _NVFP4_PREFILL_WIDE_MIN_WORK
        else:
            groups = math.ceil(max_query_len / _prefill_nvfp4_group_rows(8))
            wide = total_q >= num_reqs * max_query_len or (
                (groups * num_reqs) % _num_xcds() == 0
                and total_q * keys >= _NVFP4_PREFILL_WIDE_MIN_BATCH_WORK
            )
        if wide:
            return 8
    if short or work <= _NVFP4_PREFILL_NT2_MAX_WORK:
        return 2
    return 4


def _score_prefill_nvfp4_plan(
    num_reqs: int,
    max_query_len: int,
    max_blk: int,
    total_q: int,
    num_heads: int = 1,
) -> tuple[int, int, int, int, tuple[int, int, int, int]]:
    """``(wave_tiles, num_q_groups, num_chunks, chunk_blocks, order)`` of the
    nvfp4 prefill kernel, order from _prefill_nvfp4_order. Each (token, head)
    pair is one query row."""
    max_query_len *= num_heads
    total_q *= num_heads
    wave_tiles = _prefill_nvfp4_wave_tiles(num_reqs, max_query_len, max_blk, total_q)
    wave_tiles = max(wave_tiles, num_heads)
    num_q_groups = math.ceil(max_query_len / _prefill_nvfp4_group_rows(wave_tiles))
    num_chunks, chunk_blocks = _score_split_prefill_nvfp4(
        max_blk, num_q_groups * num_reqs
    )
    order = _prefill_nvfp4_order(
        num_reqs,
        num_q_groups,
        chunk_blocks,
        max_blk,
        max_query_len,
        total_q,
        wave_tiles,
    )
    return wave_tiles, num_q_groups, num_chunks, chunk_blocks, order


def _check_score_tensors(q_idx, key_cache_idx, score):
    """Shared dtype/layout contract of both scoring passes.

    fp8 keys are ``[num_pages, block_size, head_dim]`` in q's dtype. A uint8
    key cache is ``[num_pages, >= page_bytes]``: one nvfp4 page of
    NVFP4_BLOCK_SIZE tokens at the start of each row.

    Returns ``(kv_fmt, total_q, num_idx_heads, head_dim, block_size,
    page_stride)``.
    """
    import torch

    total_q, num_idx_heads, head_dim = q_idx.shape
    if key_cache_idx.dtype == torch.uint8:
        kv_fmt, block_size = KV_FMT_NVFP4, NVFP4_BLOCK_SIZE
        # Keys are decoded to OCP e4m3.
        if q_idx.dtype != torch.float8_e4m3fn:
            raise ValueError(f"nvfp4 keys need float8_e4m3fn q_idx, got {q_idx.dtype}")
        if head_dim != NVFP4_HEAD_DIM:
            raise ValueError(f"nvfp4 keys need head_dim {NVFP4_HEAD_DIM}")
        page_stride = key_cache_idx.stride(0)
        page_bytes = nvfp4_page_bytes(block_size, head_dim)
        if key_cache_idx.dim() != 2 or key_cache_idx.size(1) < page_bytes:
            raise ValueError(
                f"nvfp4 key_cache_idx must be [num_pages, >= {page_bytes}], "
                f"got {tuple(key_cache_idx.shape)}"
            )
    else:
        kv_fmt, block_size, page_stride = KV_FMT_FP8, key_cache_idx.size(1), 0
        if q_idx.dtype not in (torch.float8_e4m3fn, torch.float8_e4m3fnuz):
            raise ValueError(f"q_idx must be fp8 e4m3, got {q_idx.dtype}")
        if key_cache_idx.dtype != q_idx.dtype:
            raise ValueError(
                f"dtype mismatch: key_cache_idx={key_cache_idx.dtype}, "
                f"q_idx={q_idx.dtype}"
            )
        if key_cache_idx.size(2) != head_dim:
            raise ValueError("key_cache_idx head dim must match q_idx")
    if score.dtype != torch.float32:
        raise ValueError(f"score must be fp32, got {score.dtype}")
    if not (q_idx.is_contiguous() and key_cache_idx.is_contiguous()):
        raise ValueError("q_idx and key_cache_idx must be contiguous")
    if score.stride(2) != 1:
        raise ValueError("score must be contiguous along the block axis")
    if score.size(0) != num_idx_heads or score.size(1) != total_q:
        raise ValueError("score must be [num_idx_heads, total_q, S]")
    return kv_fmt, total_q, num_idx_heads, head_dim, block_size, page_stride


def _launch_score_decode(
    q_idx,
    key_cache_idx,
    score,
    block_table,
    seq_lens,
    num_reqs: int,
    num_q_tiles: int,
    num_chunks: int,
    num_waves: int,
    init_blocks: int,
    local_blocks: int,
    query_len: int,
    rank: int,
    world: int,
    num_idx_heads: int,
    head_dim: int,
    block_size: int,
    kv_fmt: int = KV_FMT_FP8,
    page_stride: int = 0,
    q_tiles: int = 1,
    req_xor_mask: int = 0,
    fill_shift: int = _DECODE_FILL_OFF,
):
    """Launch the uniform-row score kernel.

    Rows are laid out from ``query_len`` alone, so there is no ``cu_seqlens_q``
    to pass.

    ``num_q_tiles`` counts the groups of ``q_tiles`` tiles the grid covers; the
    kernel folds ``q_tiles`` tiles per wave, so a grid sized for a different
    count would leave the tail of the query axis unscored.
    """
    _score_decode_raw(
        _ptr(q_idx),
        _ptr(key_cache_idx),
        _ptr(score),
        _ptr(block_table),
        _ptr(seq_lens),
        num_reqs,
        num_q_tiles,
        block_table.stride(0),
        score.stride(0),
        score.stride(1),
        num_chunks,
        init_blocks,
        local_blocks,
        query_len,
        rank,
        world,
        block_size,
        head_dim,
        num_idx_heads,
        num_waves,
        kv_fmt,
        page_stride,
        q_tiles,
        req_xor_mask,
        fill_shift,
    )
    return score


def _launch_score_prefill(
    q_idx,
    key_cache_idx,
    score,
    block_table,
    cu_seqlens_q,
    seq_lens,
    num_reqs: int,
    num_q_groups: int,
    num_chunks: int,
    chunk_blocks: int,
    num_waves: int,
    init_blocks: int,
    local_blocks: int,
    num_idx_heads: int,
    head_dim: int,
    block_size: int,
    q_tiles: int,
):
    """Launch the ragged-row score kernel.

    ``num_q_groups`` counts groups of ``q_tiles * num_waves`` tiles, so all three
    have to come from the same resolution: a variant built for one and launched
    with a grid sized for another would leave the tail of the query axis
    unscored.

    ``chunk_blocks`` is the pages one chunk covers and ``num_chunks`` is how many
    of them the longest reach needs, so the two multiply out to that reach rather
    than dividing it.
    """
    _score_prefill_raw(
        _ptr(q_idx),
        _ptr(key_cache_idx),
        _ptr(score),
        _ptr(block_table),
        _ptr(cu_seqlens_q),
        _ptr(seq_lens),
        num_reqs,
        num_q_groups,
        block_table.stride(0),
        score.stride(0),
        score.stride(1),
        num_chunks,
        chunk_blocks,
        init_blocks,
        local_blocks,
        block_size,
        head_dim,
        num_idx_heads,
        num_waves,
        q_tiles,
    )
    return score


def pa_sparse_block_score_decode(
    q_idx,
    key_cache_idx,
    score,
    block_table,
    seq_lens,
    init_blocks: int = 0,
    local_blocks: int = 0,
    query_len: int = 1,
    max_seq_len: int = 0,
    rank: int = 0,
    world: int = 1,
):
    """Score every block of the index key cache against the query.

    The specialization for rows short enough that a request's whole query fits
    one MFMA tile, which is what decode and speculative decode look like. That
    leaves nothing to fold along the query axis, so the waves split the block
    range instead and prefetch pages into registers rather than staging them in
    LDS. ``pa_sparse_block_score_prefill`` computes the same scores without the
    length limit; this one is the faster path when the limit holds.

    With nvfp4 keys a wave scores up to ``_SCORE_DECODE_MAX_Q_TILES`` tiles per
    read of the keys, and longer rows add workgroups along the query axis.

    Args:
        q_idx: ``[num_reqs * query_len, num_idx_heads, head_dim]`` fp8 e4m3, contiguous.
        key_cache_idx: ``[num_pages, block_size, head_dim]`` fp8 e4m3, contiguous,
            or uint8 nvfp4 pages (see ``_check_score_tensors``).
        score: ``[num_idx_heads, num_reqs * query_len, S]`` fp32, written in place.
            Blocks past ``cdiv(seq_len, block_size)`` are left untouched, so
            pre-fill with ``-inf``. ``S`` indexes this rank's shard, so it spans
            ``cdiv(max_block, world)`` rather than ``max_block``.
        block_table: ``[num_reqs, >= max_block]`` int32.
        seq_lens: ``[num_reqs]`` int32.
        init_blocks / local_blocks: leading / trailing blocks forced into the
            selection via sentinel scores. Resolved against the global block id,
            so a shard holds whichever of them the round-robin gave it and
            nothing has to be re-forced after the scores are exchanged.
        query_len: query tokens per request; with fp8 keys
            ``num_idx_heads * query_len`` must fit the MFMA's 16 columns.
        max_seq_len: upper bound on the context, required. The chunk and wave
            counts the grid is built from come from this and never from
            ``seq_lens``: reading the live lengths would move the grid between
            steps and break cudagraph capture. A bound well above the context
            actually served is safe and costs only what the grid ceiling in
            ``_score_split`` does not already trim.
        rank / world: the round-robin block shard to score, ``block = local *
            world + rank``. The default scores every block. Sharding divides the
            index-cache read by ``world`` -- the term that grows with context --
            at the cost of needing the candidates exchanged afterwards; see
            ``pa_sparse_block_topk_cp``.
            Every causal bound and forced-block test stays on the global id, so
            the scores a shard produces are the ones a whole-axis pass would
            have put in those positions.
    """
    kv_fmt, total_q, num_idx_heads, head_dim, block_size, page_stride = (
        _check_score_tensors(q_idx, key_cache_idx, score)
    )
    if total_q % query_len != 0:
        raise ValueError(
            f"q_idx rows {total_q} not a multiple of query_len {query_len}"
        )
    num_reqs = total_q // query_len
    if kv_fmt == KV_FMT_NVFP4:
        if num_idx_heads not in NVFP4_INDEX_HEADS:
            raise ValueError(
                f"nvfp4 is built for {NVFP4_INDEX_HEADS} index heads, "
                f"got {num_idx_heads}"
            )
    elif num_idx_heads * query_len > SCORE_MFMA_COLS:
        raise ValueError(
            f"num_idx_heads * query_len = {num_idx_heads * query_len} exceeds the "
            f"{SCORE_MFMA_COLS} MFMA columns"
        )
    if seq_lens.size(0) != num_reqs or block_table.size(0) != num_reqs:
        raise ValueError("seq_lens and block_table need one entry per request")
    if num_reqs == 0:
        return score

    if max_seq_len < 1:
        raise ValueError(
            "pass max_seq_len so the launch dimensions are fixed at capture"
        )
    if world < 1 or not 0 <= rank < world:
        raise ValueError(f"rank {rank} is not a shard of a world of {world}")

    max_blk = math.ceil(max_seq_len / block_size)
    local_blk = math.ceil(max_blk / world)
    if score.size(2) < local_blk:
        raise ValueError(
            f"score's block axis is {score.size(2)} but this rank's shard of "
            f"max_seq_len={max_seq_len} spans {local_blk} blocks"
        )
    num_tiles = math.ceil(query_len / _tokens_per_tile(num_idx_heads))
    split, q_tiles, req_xor_mask, fill_shift = _score_split, 1, 0, _DECODE_FILL_OFF
    if kv_fmt == KV_FMT_NVFP4:
        split = _score_split_decode_nvfp4
        q_tiles = _decode_q_tiles(num_tiles, num_reqs)
        req_xor_mask = _decode_req_xor_mask(num_reqs)
        fill_shift = _decode_fill_shift(num_reqs)
    # The split divides this rank's own blocks, not the whole context's.
    num_chunks, num_waves = split(local_blk, num_reqs, sharded=world > 1)

    return _launch_score_decode(
        q_idx,
        key_cache_idx,
        score,
        block_table,
        seq_lens,
        num_reqs=num_reqs,
        num_q_tiles=math.ceil(num_tiles / q_tiles),
        num_chunks=num_chunks,
        num_waves=num_waves,
        init_blocks=init_blocks,
        local_blocks=local_blocks,
        query_len=query_len,
        rank=rank,
        world=world,
        num_idx_heads=num_idx_heads,
        head_dim=head_dim,
        block_size=block_size,
        kv_fmt=kv_fmt,
        page_stride=page_stride,
        q_tiles=q_tiles,
        req_xor_mask=req_xor_mask,
        fill_shift=fill_shift,
    )


def pa_sparse_block_score_prefill(
    q_idx,
    key_cache_idx,
    score,
    block_table,
    cu_seqlens_q,
    seq_lens,
    init_blocks: int = 0,
    local_blocks: int = 0,
    max_query_len: int = 0,
    max_seq_len: int = 0,
):
    """Score every block of the index key cache against ragged query rows.

    The general form, and the only one prefill can use: query lengths come from
    ``cu_seqlens_q`` rather than being uniform, so a request's rows are covered by
    several workgroups and only ``num_idx_heads`` (not
    ``num_idx_heads * query_len``) has to fit the MFMA's 16 columns. Scores,
    causal masking and init/local sentinels are identical to
    ``pa_sparse_block_score_decode``, which is the faster path on the shapes
    that fit its length limit.

    Every wave of the workgroup is on the query axis, so a workgroup spans
    ``q_tiles * num_waves`` tiles of tokens and stages the block range once for
    all of them rather than each wave reading it again.

    nvfp4 keys go to pa_sparse_block_score_prefill_nvfp4.

    Args:
        q_idx: ``[total_q, num_idx_heads, head_dim]`` fp8 e4m3, contiguous.
        key_cache_idx: ``[num_pages, block_size, head_dim]`` fp8 e4m3, contiguous,
            or uint8 nvfp4 pages (see ``_check_score_tensors``).
        score: ``[num_idx_heads, total_q, S]`` fp32, written in place. Blocks a
            row cannot see are left untouched, so pre-fill with ``-inf`` unless
            the consumer bounds each row itself (``pa_sparse_block_topk`` does).
        block_table: ``[num_reqs, >= max_block]`` int32.
        cu_seqlens_q: ``[num_reqs + 1]`` int32, query start offsets rebased to 0.
        seq_lens: ``[num_reqs]`` int32, total KV length per request. A row is
            aligned to the end of its request, so query token ``t`` of a request
            with ``qlen`` rows sees ``seq_len - qlen + t + 1`` tokens.
        init_blocks / local_blocks: leading / trailing blocks forced into the
            selection via sentinel scores.
        max_query_len: launch-time upper bound on a request's query length, which
            fixes the query-tile count and, with the request count, how many tiles
            a workgroup folds into one pass over the block range. Never read from
            ``cu_seqlens_q``, so the launch dimensions stay fixed at capture.
        max_seq_len: upper bound on the context, required. The block-axis split
            comes from this and never from ``seq_lens``, so the launch dimensions
            stay fixed at capture. A bound well above the context actually served
            is safe and costs only what the grid ceiling in
            ``_score_split_prefill`` does not already trim.
    """
    kv_fmt, total_q, num_idx_heads, head_dim, block_size, page_stride = (
        _check_score_tensors(q_idx, key_cache_idx, score)
    )
    if kv_fmt == KV_FMT_NVFP4:
        if num_idx_heads not in NVFP4_INDEX_HEADS:
            raise ValueError(
                f"nvfp4 is built for {NVFP4_INDEX_HEADS} index heads, "
                f"got {num_idx_heads}"
            )
    elif num_idx_heads > SCORE_MFMA_COLS:
        raise ValueError(
            f"num_idx_heads {num_idx_heads} exceeds the {SCORE_MFMA_COLS} MFMA columns"
        )
    num_reqs = cu_seqlens_q.size(0) - 1
    if num_reqs < 0:
        raise ValueError("cu_seqlens_q must hold num_reqs + 1 entries")
    if seq_lens.size(0) != num_reqs or block_table.size(0) != num_reqs:
        raise ValueError("seq_lens and block_table need one entry per request")
    if num_reqs == 0 or total_q == 0:
        return score
    if max_query_len < 1:
        raise ValueError(
            "pass max_query_len so the query-tile count is fixed at capture "
            "rather than read from cu_seqlens_q"
        )

    if max_seq_len < 1:
        raise ValueError(
            "pass max_seq_len so the block-axis split is fixed at capture "
            "rather than read from seq_lens"
        )

    if kv_fmt == KV_FMT_NVFP4:
        max_blk = math.ceil(max_seq_len / block_size)
        wave_tiles, num_q_groups, num_chunks, chunk_blocks, order = (
            _score_prefill_nvfp4_plan(
                num_reqs, max_query_len, max_blk, total_q, num_idx_heads
            )
        )
        store_pages, group_xor, req_rot, xcds = order
        req_chunks, grid_chunks = 0, num_chunks
        if num_reqs > 1:
            groups = num_q_groups * num_reqs
            req_chunks = _prefill_nvfp4_req_chunks(groups)
            grid_chunks = _prefill_nvfp4_grid_chunks(max_blk, groups, num_chunks)
        _score_prefill_nvfp4_raw(
            _ptr(q_idx),
            _ptr(key_cache_idx),
            _ptr(score),
            _ptr(block_table),
            _ptr(cu_seqlens_q),
            _ptr(seq_lens),
            num_reqs,
            num_q_groups,
            block_table.stride(0),
            score.stride(0),
            score.stride(1),
            num_chunks,
            chunk_blocks,
            init_blocks,
            local_blocks,
            page_stride,
            num_idx_heads,
            wave_tiles,
            store_pages,
            group_xor,
            req_rot,
            xcds,
            req_chunks,
            grid_chunks,
        )
        return score

    num_tiles = math.ceil(max_query_len / _tokens_per_tile(num_idx_heads))
    num_waves = SCORE_WAVES

    # Tiles per wave, and the waves are every wave of the workgroup: the page is
    # staged once for all of them, so folding costs registers and nothing else
    # and the block-axis split puts the parallelism back. A workgroup wider than
    # the tiles there are is not a problem -- the waves past the end come out
    # with no live tiles, which masks their loads, their MFMAs and their stores
    # while they still carry their share of the page.
    q_tiles = _resolve_q_tiles_prefill(num_tiles, num_reqs, num_waves)
    num_q_groups = math.ceil(num_tiles / (q_tiles * num_waves))

    max_blk = math.ceil(max_seq_len / block_size)
    num_chunks, chunk_blocks = _score_split_prefill(max_blk, num_q_groups * num_reqs)

    return _launch_score_prefill(
        q_idx,
        key_cache_idx,
        score,
        block_table,
        cu_seqlens_q,
        seq_lens,
        num_reqs=num_reqs,
        num_q_groups=num_q_groups,
        num_chunks=num_chunks,
        chunk_blocks=chunk_blocks,
        num_waves=num_waves,
        init_blocks=init_blocks,
        local_blocks=local_blocks,
        num_idx_heads=num_idx_heads,
        head_dim=head_dim,
        block_size=block_size,
        q_tiles=q_tiles,
    )


def pa_sparse_block_topk(
    score,
    topk_idx,
    block_table,
    seq_lens,
    max_seq_len: int,
    block_size: int,
    query_len: int = 1,
    num_waves: int = 0,
    num_valid_pages=None,
    sparse_bt=None,
    sparse_ctx=None,
    row_req_id=None,
    kv_lens=None,
    num_kv_heads: int = 1,
    pages_per_block: int = 8,
):
    """Keep the top-k blocks of every score row and emit their page table.

    The selection and the table it expands into come out of the same workgroup:
    the winners are already in its LDS, so writing the pages there costs one wave
    and saves the consumer a second pass over the selection. That is why the
    table is not optional.

    Args:
        score: ``[num_idx_heads, total_q, S]`` fp32. Lanes read whole 64-wide
            strips and each holds a power-of-two count of them, so ``S`` must
            reach ``pow2_ceil(cdiv(max_block, 64)) * 64``, which is past
            ``max_block`` itself for all but a few block counts.
        topk_idx: ``[num_idx_heads, total_q, TopK]`` int32, written in place.
            Rows with fewer than ``TopK`` blocks are padded with -1. Only the
            ``TopK`` axis has to be contiguous, so this can be a row slice of a
            wider persistent buffer.
        seq_lens: ``[num_reqs]`` int32. Unused when ``num_valid_pages`` is given.
        max_seq_len: launch-time upper bound, so the slot count is fixed at
            capture rather than read from ``seq_lens``.
        block_size: tokens per block, must match the scoring pass.
        query_len: query tokens per request, uniform across the batch. Unused when
            ``num_valid_pages`` is given.
        num_waves: waves a workgroup splits one row's slots across; 0 picks from
            the slot and row counts.
        num_valid_pages: ``[total_q]`` int32, the causal block count of each row.
            Ragged rows cannot be recovered from ``seq_lens`` and a single
            ``query_len``, so the prefill pass passes the counts directly. Must
            agree with the scoring pass's causal masking block for block.
        block_table: ``[num_reqs, max_blocks]`` int32, logical block to logical
            page, which is wvs hat the emitted table is resolved through.
        sparse_bt: ``[total_q * num_kv_heads, TopK * pages_per_block]`` int32,
            written in place: the selected blocks' pages packed towards slot 0
            with the partial tail block last, zeros after. Physical page ids
            fold (page, kv head) with the head minor. Allocated here when left
            out, which is the shape a caller would allocate anyway; pass a
            buffer to keep the address fixed across cudagraph replays.
        sparse_ctx: ``[total_q * num_kv_heads]`` int32, tokens the emitted pages
            hold, i.e. the row's own bound on ``sparse_bt``. Allocated with
            ``sparse_bt``.
        row_req_id: ``[total_q]`` int32, the request each row belongs to.
            Required with ``num_valid_pages``, whose rows are ragged.
        kv_lens: ``[total_q]`` int32, causal token count of each row. Required
            with ``num_valid_pages``: the tail block's token count does not
            follow from a block count.
        num_kv_heads: kv heads the table is emitted for. Index heads normally
            map 1:1 onto them; a single index head's selection is shared by all
            of them instead.
        pages_per_block: physical pages one selected block expands into.

    Returns:
        ``(topk_idx, sparse_bt, sparse_ctx)``, the same tensors that were passed
        in wherever they were passed in.
    """
    import torch

    if score.dtype != torch.float32:
        raise ValueError(f"score must be fp32, got {score.dtype}")
    if topk_idx.dtype != torch.int32:
        raise ValueError(f"topk_idx must be int32, got {topk_idx.dtype}")
    if score.stride(2) != 1:
        raise ValueError("score must be contiguous along the block axis")
    if topk_idx.stride(2) != 1:
        raise ValueError("topk_idx must be contiguous along the topk axis")
    num_idx_heads, total_q, _ = score.shape
    topk = topk_idx.size(2)
    if topk > WAVE_SIZE:
        raise ValueError(f"topk {topk} exceeds one wave ({WAVE_SIZE})")
    if topk_idx.size(0) != num_idx_heads or topk_idx.size(1) != total_q:
        raise ValueError("topk_idx must match score's leading dims")
    if num_valid_pages is not None:
        if num_valid_pages.dtype != torch.int32:
            raise ValueError(
                f"num_valid_pages must be int32, got {num_valid_pages.dtype}"
            )
        if num_valid_pages.numel() != total_q or num_valid_pages.stride(0) != 1:
            raise ValueError("num_valid_pages must be a contiguous [total_q] vector")
    else:
        if total_q % query_len != 0:
            raise ValueError(
                f"score rows {total_q} not a multiple of query_len {query_len}"
            )
        if seq_lens.size(0) != total_q // query_len:
            raise ValueError("seq_lens needs one entry per request")

    if num_kv_heads < 1 or num_kv_heads % num_idx_heads:
        raise ValueError(
            f"num_kv_heads {num_kv_heads} must be a positive multiple of the "
            f"{num_idx_heads} index head(s)"
        )
    rows = total_q * num_kv_heads
    if sparse_bt is None:
        sparse_bt = torch.empty(
            (rows, topk * pages_per_block), dtype=torch.int32, device=score.device
        )
    if sparse_ctx is None:
        sparse_ctx = torch.empty(rows, dtype=torch.int32, device=score.device)
    for name, t in (
        ("block_table", block_table),
        ("sparse_bt", sparse_bt),
        ("sparse_ctx", sparse_ctx),
    ):
        if t.dtype != torch.int32:
            raise ValueError(f"{name} must be int32, got {t.dtype}")
    if sparse_bt.shape != (rows, topk * pages_per_block):
        raise ValueError(
            f"sparse_bt must be ["
            f"{rows}, {topk * pages_per_block}], got {list(sparse_bt.shape)}"
        )
    if sparse_bt.stride(1) != 1:
        raise ValueError("sparse_bt must be contiguous along the page axis")
    if sparse_ctx.numel() != rows or sparse_ctx.stride(0) != 1:
        raise ValueError(f"sparse_ctx must be a contiguous [{rows}] vector")
    if num_valid_pages is not None and (row_req_id is None or kv_lens is None):
        raise ValueError(
            "ragged rows need row_req_id and kv_lens to place their tail block"
        )
    for name, t in (("row_req_id", row_req_id), ("kv_lens", kv_lens)):
        if t is None:
            continue
        if t.dtype != torch.int32:
            raise ValueError(f"{name} must be int32, got {t.dtype}")
        if t.numel() != total_q or t.stride(0) != 1:
            raise ValueError(f"{name} must be a contiguous [{total_q}] vector")

    max_blk = math.ceil(max_seq_len / block_size)

    # Lanes read whole 64-wide strips, so the slot count -- and with it the
    # padding the score buffer needs -- rounds up to a power of two.
    slots = _pow2_ceil(math.ceil(max_blk / WAVE_SIZE))
    if slots > TOPK_MAX_SLOTS:
        raise ValueError(
            f"max_seq_len={max_seq_len} needs {slots} slots, more than the "
            f"compiled maximum of {TOPK_MAX_SLOTS}"
        )
    if score.size(2) < slots * WAVE_SIZE:
        raise ValueError(
            f"score's block axis is {score.size(2)} but must be padded to "
            f"{slots * WAVE_SIZE} to cover max_seq_len={max_seq_len}"
        )
    if total_q == 0:
        return topk_idx, sparse_bt, sparse_ctx

    if num_waves == 0:
        num_waves = _topk_waves(slots, num_idx_heads * total_q)
    elif slots % num_waves:
        raise ValueError(f"num_waves {num_waves} must divide slots {slots}")
    elif num_waves > TOPK_MAX_WAVES:
        raise ValueError(
            f"num_waves {num_waves} exceeds the {TOPK_MAX_WAVES} a workgroup is "
            f"built to split a row across"
        )

    _topk_raw(
        _ptr(score),
        _ptr(topk_idx),
        _ptr(seq_lens),
        _ptr(num_valid_pages),
        _ptr(block_table),
        _ptr(row_req_id),
        _ptr(kv_lens),
        _ptr(sparse_bt),
        _ptr(sparse_ctx),
        num_idx_heads,
        total_q,
        score.stride(0),
        score.stride(1),
        topk_idx.stride(0),
        topk_idx.stride(1),
        block_table.stride(0),
        sparse_bt.stride(0),
        num_kv_heads,
        query_len,
        block_size,
        topk,
        slots,
        num_waves,
        pages_per_block,
    )
    return topk_idx, sparse_bt, sparse_ctx


def _topk_slots(max_blk: int, pass_name: str) -> int:
    """Wave-wide strips of a score row one lane holds, and the cap on them.

    Lanes read whole strips with no tail guard and each holds a power-of-two
    count, so this is also the padding the score buffer needs.
    """
    slots = _pow2_ceil(math.ceil(max_blk / WAVE_SIZE))
    if slots > TOPK_MAX_SLOTS:
        raise ValueError(
            f"{pass_name}: {max_blk} blocks per row needs {slots} slots, more "
            f"than the compiled maximum of {TOPK_MAX_SLOTS}"
        )
    return slots


# Blocks the fused selector's generation counter is sized for -- rows x index
# heads -- and the ranks it can reach. Mirrors kCpMaxBlocks / kCpMaxRanks in
# pa_sparse_block_select_kernels.cuh.
CP_MAX_BLOCKS = 256
CP_MAX_RANKS = 8
# Generations the candidate buffer holds at once. Two is what lets a rank run
# a whole call ahead of a peer without landing on addresses that peer is still
# reading, which is what removes any wait before the nominate.
CP_BUFFERS = 2
# Block indices a candidate's id half can carry, the top byte of it being the
# generation tag.
CP_MAX_BLOCK_ID = 1 << 24
# Candidates the fused merge holds, which is one wave's lanes: it rides the
# shard pass's launch rather than owning one, so it is not widened per world.
TOPK_CP_MAX_CAND = WAVE_SIZE


def topk_cp_candidate_numel(
    world: int, owned_heads: int, total_q: int, topk: int
) -> int:
    """int64 elements one rank's candidate buffer needs.

    Laid out ``[2, world, owned_heads, total_q, topk]``: the shard axis is what
    the peers fill, so every rank writes a different slice of the same buffer
    and the merge reads it with nothing to gather, and the leading axis is the
    generation parity that keeps consecutive calls off each other's addresses.
    """
    return CP_BUFFERS * world * owned_heads * total_q * topk


def pa_sparse_block_topk_cp(
    score,
    peer_cand_ptrs,
    cp_gen,
    topk_idx,
    seq_lens,
    block_table,
    sparse_bt,
    sparse_ctx,
    max_seq_len: int,
    block_size: int,
    cand_rows: int = 0,
    query_len: int = 1,
    init_blocks: int = 0,
    local_blocks: int = 0,
    num_kv_heads: int = 1,
    pages_per_block: int = 8,
    rank: int = 0,
    world: int = 1,
    num_waves: int = 0,
):
    """Nominate, exchange and merge a context partition in one launch.

    Against ``pa_sparse_block_topk``'s single pass: that one owns a whole
    score row and can finish, a partitioned rank owns ``1/world`` of every row
    and can only nominate, so a selection needs a nominate, an exchange and a
    merge. This is all three in one launch.

    What the fusion removes is a collective between the halves: measured at
    four ranks, an all-gather of the candidates costs the same whether it
    carries 4KB or 256KB, so what it spends is launch and synchronisation
    rather than the wire, and shrinking the payload never helped. Here each
    rank writes its candidates straight into the peer that owns those heads
    over the IPC mapping the ranks already share.

    Nothing signals that write. A candidate carries the generation that
    produced it in the spare top byte of its block index, so it is its own
    arrival flag: the merge spins on the word it was going to read anyway,
    unblocks per lane rather than per block, and overlaps an early row's merge
    with a peer still nominating a late one. The buffer holds two generations
    so that the nominate never waits either -- the addresses a call is about to
    write were last read two calls ago, which stream ordering plus the merge's
    own wait already prove is finished.

    Each rank nominates the full ``topk`` of its shard and never
    ``topk/world``, since the global winners may lie entirely inside one shard.
    That is what makes the merge exact rather than approximate: a block in the
    global top-k necessarily won its own shard's, so re-selecting over what
    arrived reproduces what a rank scoring every block would have picked. The
    forced init and local blocks are re-pinned in the merge so the pin survives
    the round trip through a score.

    A block takes a stride of rows and covers every head of them, so the merge
    of a (row, owned head) needs exactly what the same block index produced on
    each peer. That is what keeps the wait off a grid-wide barrier, and with it
    off a cooperative launch.

    Every rank must make the same sequence of calls with the same ``total_q``,
    ``query_len``, ``num_idx_heads``, ``cand_rows`` and ``world``. Generations
    are counted per block index, so a rank that skipped a call compares tags
    against a different call's. A mismatch does not produce wrong output, it
    hangs.

    One caller must also hold ``cand_rows`` and ``num_idx_heads`` fixed across
    its own calls, even though ``total_q`` may vary. Those two fix the buffer's
    layout and the grid's shape, and between them they are what makes an
    address mean one (shard, head, row, nominee) for the life of the buffer. If
    the layout moved with the batch, a short call would leave a candidate at an
    address a longer one reads as a different row -- and a generation tag that
    happened to match would be taken for fresh. That one *does* produce wrong
    output rather than hanging, so it is checked where it can be.

    Args:
        score: ``[num_idx_heads, total_q, S]`` fp32, this rank's shard as
            ``pa_sparse_block_score_decode`` wrote it, where ``num_idx_heads``
            is the model's whole count. ``S`` must reach
            ``pow2_ceil(cdiv(cdiv(max_block, world), 64)) * 64``.
        peer_cand_ptrs: ``[world]`` int64 CPU tensor of device addresses, entry
            ``r`` being rank ``r``'s candidate buffer. Each must hold
            ``topk_cp_candidate_numel(...)`` int64 and be reachable from this
            rank, i.e. mapped by IPC. Entry ``rank`` is this rank's own. Zero
            it once at allocation: a never-written word reads as generation 0,
            which is what tells the merge it has not arrived yet.
        cp_gen: ``[CP_MAX_BLOCKS]`` int32 device tensor, this rank's own and
            not peer-visible, zeroed once at allocation and then owned by this
            op. It counts the calls each block index has made, which is what
            makes the exchange replay-safe under a cudagraph -- a flag that got
            cleared would race a peer that had not yet read it, whereas a
            counter only ever reads as old. So it must not be cleared between
            calls and must not be shared with another caller. It is tied to one
            ``num_idx_heads``, since block indices are laid out by it, but
            ``total_q`` may vary freely -- a short batch leaves the tail
            counters untouched.
        topk_idx: ``[num_idx_heads // world, total_q, topk]`` int32, written in
            place -- the heads this rank owns, not the heads it scored. Rows
            with fewer than ``topk`` blocks are padded with -1.
        seq_lens: ``[num_reqs]`` int32.
        block_table: ``[num_reqs, max_blocks]`` int32.
        sparse_bt: ``[total_q * num_kv_heads, topk * pages_per_block]``
            int32, and ``sparse_ctx``: ``[total_q * num_kv_heads]`` int32. Same
            layout and numbering ``pa_sparse_block_topk`` emits, so a mixed
            batch can share one buffer whichever pass wrote which rows.
        max_seq_len: launch-time upper bound, so the slot count is fixed at
            capture rather than read from ``seq_lens``.
        block_size: tokens per block, must match the scoring pass.
        cand_rows: rows the candidate buffers were allocated for, which is what
            they are strided by. Must be the widest ``total_q`` this caller
            will ever pass, and the same on every rank and every call; 0 means
            ``total_q``, which is only safe when the batch never varies.
        query_len: query tokens per request, uniform across the batch.
        init_blocks / local_blocks: the counts the scoring pass was given.
        num_kv_heads: kv heads the table is emitted for, which must equal the
            index heads this rank owns.
        pages_per_block: physical pages one selected block expands into.
        rank / world: the round-robin block shard, which must be the one the
            scoring pass was given.
        num_waves: waves a workgroup splits one row's slots across; 0 picks
            from the slot count.

    Returns:
        ``(topk_idx, sparse_bt, sparse_ctx)``, the tensors that were passed in.
    """
    import torch

    if score.dtype != torch.float32:
        raise ValueError(f"score must be fp32, got {score.dtype}")
    if topk_idx.dtype != torch.int32:
        raise ValueError(f"topk_idx must be int32, got {topk_idx.dtype}")
    if score.stride(2) != 1:
        raise ValueError("score must be contiguous along the block axis")
    if topk_idx.stride(2) != 1:
        raise ValueError("topk_idx must be contiguous along the topk axis")

    num_idx_heads, total_q, _ = score.shape
    topk = topk_idx.size(2)
    if world < 1 or not 0 <= rank < world:
        raise ValueError(f"rank {rank} is not a shard of a world of {world}")
    if world > CP_MAX_RANKS:
        raise ValueError(f"world {world} exceeds the {CP_MAX_RANKS} the peer map holds")
    if num_idx_heads % world:
        raise ValueError(
            f"{num_idx_heads} index heads do not divide over a world of {world}"
        )
    owned = num_idx_heads // world
    if topk > WAVE_SIZE:
        raise ValueError(f"topk {topk} exceeds one wave ({WAVE_SIZE})")
    if world * topk > TOPK_CP_MAX_CAND:
        raise ValueError(
            f"world {world} x topk {topk} exceeds the {TOPK_CP_MAX_CAND} "
            f"candidates the fused merge holds"
        )
    if topk_idx.size(0) != owned or topk_idx.size(1) != total_q:
        raise ValueError(
            f"topk_idx must be [{owned}, {total_q}, topk] -- the heads this "
            f"rank owns, not the {num_idx_heads} it scores"
        )
    # One index head per kv head: the page table is laid out against this
    # rank's kv heads, and the merge keeps exactly the heads it owns.
    if num_kv_heads != owned:
        raise ValueError(
            f"the merge emits a page table per kv head, so its {owned} owned "
            f"index head(s) must equal the {num_kv_heads} kv head(s)"
        )
    if query_len < 1 or total_q % query_len:
        raise ValueError(
            f"score rows {total_q} not a multiple of query_len {query_len}"
        )
    if seq_lens.size(0) != total_q // query_len:
        raise ValueError("seq_lens needs one entry per request")

    if peer_cand_ptrs.dtype != torch.int64:
        raise ValueError(f"peer_cand_ptrs must be int64, got {peer_cand_ptrs.dtype}")
    if peer_cand_ptrs.device.type != "cpu":
        raise ValueError(
            "peer_cand_ptrs is read on the host, so it must be a CPU tensor"
        )
    if peer_cand_ptrs.numel() < world:
        raise ValueError(
            f"peer_cand_ptrs needs one address per rank, got {peer_cand_ptrs.numel()}"
        )
    if cp_gen.dtype != torch.int32:
        raise ValueError(f"cp_gen must be int32, got {cp_gen.dtype}")
    if cp_gen.device.type == "cpu" or cp_gen.numel() < CP_MAX_BLOCKS:
        raise ValueError(
            f"cp_gen must be a device tensor of at least {CP_MAX_BLOCKS} counters, "
            f"got {cp_gen.numel()} on {cp_gen.device}"
        )
    if cand_rows == 0:
        cand_rows = total_q
    if cand_rows < total_q:
        raise ValueError(
            f"cand_rows {cand_rows} is smaller than the {total_q} rows this call "
            f"has; the candidate buffer is strided by its allocation"
        )

    rows = total_q * num_kv_heads
    for name, t in (
        ("seq_lens", seq_lens),
        ("block_table", block_table),
        ("sparse_bt", sparse_bt),
        ("sparse_ctx", sparse_ctx),
    ):
        if t.dtype != torch.int32:
            raise ValueError(f"{name} must be int32, got {t.dtype}")
    if sparse_bt.shape != (rows, topk * pages_per_block):
        raise ValueError(
            f"sparse_bt must be [{rows}, {topk * pages_per_block}], "
            f"got {list(sparse_bt.shape)}"
        )
    if sparse_bt.stride(1) != 1:
        raise ValueError("sparse_bt must be contiguous along the page axis")
    if sparse_ctx.numel() != rows or sparse_ctx.stride(0) != 1:
        raise ValueError(f"sparse_ctx must be a contiguous [{rows}] vector")

    max_blk = math.ceil(max_seq_len / block_size)
    # The generation tag takes the id half's top byte, so a block index has to
    # fit what is left. At a 128-token block that is a 2G-token context.
    if max_blk >= CP_MAX_BLOCK_ID:
        raise ValueError(
            f"max_seq_len {max_seq_len} is {max_blk} blocks, more than the "
            f"{CP_MAX_BLOCK_ID} a candidate's id half holds beside its tag"
        )
    local_blk = math.ceil(max_blk / world)
    slots = _topk_slots(local_blk, "pa_sparse_block_topk_cp")
    if score.size(2) < slots * WAVE_SIZE:
        raise ValueError(
            f"score's block axis is {score.size(2)} but must be padded to "
            f"{slots * WAVE_SIZE} to cover this rank's {local_blk} blocks"
        )
    if total_q == 0:
        return topk_idx, sparse_bt, sparse_ctx

    if num_waves == 0:
        num_waves = _shard_topk_waves(slots)
    elif slots % num_waves:
        raise ValueError(f"num_waves {num_waves} must divide slots {slots}")
    elif num_waves > TOPK_MAX_WAVES:
        raise ValueError(
            f"num_waves {num_waves} exceeds the {TOPK_MAX_WAVES} a workgroup is "
            f"built to split a row across"
        )

    _topk_cp_raw(
        _ptr(score),
        _ptr(peer_cand_ptrs),
        _ptr(cp_gen),
        _ptr(topk_idx),
        _ptr(seq_lens),
        _ptr(block_table),
        _ptr(sparse_bt),
        _ptr(sparse_ctx),
        num_idx_heads,
        total_q,
        cand_rows,
        score.stride(0),
        score.stride(1),
        topk_idx.stride(0),
        topk_idx.stride(1),
        block_table.stride(0),
        sparse_bt.stride(0),
        num_kv_heads,
        query_len,
        init_blocks,
        local_blocks,
        rank,
        world,
        block_size,
        topk,
        slots,
        num_waves,
    )
    return topk_idx, sparse_bt, sparse_ctx
