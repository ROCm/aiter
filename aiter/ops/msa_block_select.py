# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import math

from aiter.jit.core import compile_ops


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
    block_size: int,
    head_dim: int,
    num_idx_heads: int,
    num_waves: int,
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

SCORE_WAVES = 4

SCORE_MAX_WORKGROUPS = 16384

# Columns of the MFMA, each holding one (token, head) pair.
SCORE_MFMA_COLS = 16

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


def _score_split(max_blk: int, num_reqs: int) -> tuple[int, int]:
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
    """
    waves = min(SCORE_WAVES, _pow2_floor(max(1, max_blk)))
    chunks = max(1, max_blk // waves)
    return max(1, min(chunks, SCORE_MAX_WORKGROUPS // max(1, num_reqs))), waves


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


def _check_score_tensors(q_idx, key_cache_idx, score):
    """Shared dtype/layout contract of both scoring passes.

    Returns ``(total_q, num_idx_heads, head_dim, block_size)``.
    """
    import torch

    if q_idx.dtype not in (torch.float8_e4m3fn, torch.float8_e4m3fnuz):
        raise ValueError(f"q_idx must be fp8 e4m3, got {q_idx.dtype}")
    if key_cache_idx.dtype != q_idx.dtype:
        raise ValueError(
            f"dtype mismatch: key_cache_idx={key_cache_idx.dtype}, q_idx={q_idx.dtype}"
        )
    if score.dtype != torch.float32:
        raise ValueError(f"score must be fp32, got {score.dtype}")
    if not (q_idx.is_contiguous() and key_cache_idx.is_contiguous()):
        raise ValueError("q_idx and key_cache_idx must be contiguous")
    if score.stride(2) != 1:
        raise ValueError("score must be contiguous along the block axis")

    total_q, num_idx_heads, head_dim = q_idx.shape
    block_size = key_cache_idx.size(1)
    if key_cache_idx.size(2) != head_dim:
        raise ValueError("key_cache_idx head dim must match q_idx")
    if score.size(0) != num_idx_heads or score.size(1) != total_q:
        raise ValueError("score must be [num_idx_heads, total_q, S]")
    return total_q, num_idx_heads, head_dim, block_size


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
    num_idx_heads: int,
    head_dim: int,
    block_size: int,
):
    """Launch the uniform-row score kernel.

    Rows are laid out from ``query_len`` alone, so there is no ``cu_seqlens_q``
    to pass.

    ``num_q_tiles`` counts the tiles the grid covers; the kernel folds one tile
    per wave, so a grid sized for a different count would leave the tail of the
    query axis unscored.
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
        block_size,
        head_dim,
        num_idx_heads,
        num_waves,
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
):
    """Score every block of the index key cache against the query.

    The specialization for rows short enough that a request's whole query fits
    one MFMA tile, which is what decode and speculative decode look like. That
    leaves nothing to fold along the query axis, so the waves split the block
    range instead and prefetch pages into registers rather than staging them in
    LDS. ``pa_sparse_block_score_prefill`` computes the same scores without the
    length limit; this one is the faster path when the limit holds.

    Args:
        q_idx: ``[num_reqs * query_len, num_idx_heads, head_dim]`` fp8 e4m3, contiguous.
        key_cache_idx: ``[num_pages, block_size, head_dim]`` fp8 e4m3, contiguous.
        score: ``[num_idx_heads, num_reqs * query_len, S]`` fp32, written in place.
            Blocks past ``cdiv(seq_len, block_size)`` are left untouched, so
            pre-fill with ``-inf``.
        block_table: ``[num_reqs, >= max_block]`` int32.
        seq_lens: ``[num_reqs]`` int32.
        init_blocks / local_blocks: leading / trailing blocks forced into the
            selection via sentinel scores.
        query_len: query tokens per request; ``num_idx_heads * query_len`` must
            fit the MFMA's 16 columns.
        max_seq_len: upper bound on the context, required. The chunk and wave
            counts the grid is built from come from this and never from
            ``seq_lens``: reading the live lengths would move the grid between
            steps and break cudagraph capture. A bound well above the context
            actually served is safe and costs only what the grid ceiling in
            ``_score_split`` does not already trim.
    """
    total_q, num_idx_heads, head_dim, block_size = _check_score_tensors(
        q_idx, key_cache_idx, score
    )
    if total_q % query_len != 0:
        raise ValueError(
            f"q_idx rows {total_q} not a multiple of query_len {query_len}"
        )
    num_reqs = total_q // query_len
    if num_idx_heads * query_len > SCORE_MFMA_COLS:
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
    num_chunks, num_waves = _score_split(math.ceil(max_seq_len / block_size), num_reqs)

    return _launch_score_decode(
        q_idx,
        key_cache_idx,
        score,
        block_table,
        seq_lens,
        num_reqs=num_reqs,
        num_q_tiles=math.ceil(query_len / _tokens_per_tile(num_idx_heads)),
        num_chunks=num_chunks,
        num_waves=num_waves,
        init_blocks=init_blocks,
        local_blocks=local_blocks,
        query_len=query_len,
        num_idx_heads=num_idx_heads,
        head_dim=head_dim,
        block_size=block_size,
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

    Args:
        q_idx: ``[total_q, num_idx_heads, head_dim]`` fp8 e4m3, contiguous.
        key_cache_idx: ``[num_pages, block_size, head_dim]`` fp8 e4m3, contiguous.
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
    total_q, num_idx_heads, head_dim, block_size = _check_score_tensors(
        q_idx, key_cache_idx, score
    )
    if num_idx_heads > SCORE_MFMA_COLS:
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
