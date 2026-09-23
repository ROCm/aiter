import triton
import triton.language as tl

from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr


@triton.jit(repr=make_kernel_repr("_pa_ps_tile_scan", ["BLOCK_SIZE", "num_warps"]))
def _pa_ps_tile_scan(
    context_lengths,
    tile_prefix,
    tile_totals,
    NUM_SEQS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    block = tl.program_id(0)
    sequence = block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    context = tl.load(context_lengths + sequence, sequence < NUM_SEQS, other=0)
    tiles = (context.to(tl.int64) + 255) // 256
    tl.store(tile_prefix + sequence, tl.cumsum(tiles, 0), sequence < NUM_SEQS)
    tl.store(tile_totals + block, tl.sum(tiles, 0))


@triton.jit(
    repr=make_kernel_repr(
        "_pa_ps_sequence_scan",
        [
            "NUM_SEQS",
            "NUM_GROUPS",
            "MAX_PARTS",
            "WORK_OVERHEAD",
            "BLOCK_SIZE",
            "num_warps",
        ],
    )
)
def _pa_ps_sequence_scan(
    context_lengths,
    tile_prefix,
    tile_totals,
    sequence_info,
    chunk_prefix,
    NUM_SEQS: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    MAX_PARTS: tl.constexpr,
    WORK_OVERHEAD: tl.constexpr,
    NUM_CHUNKS: tl.constexpr,
    BLOCK_CHUNKS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    block = tl.program_id(0)
    local = tl.arange(0, BLOCK_SIZE)
    sequence = block * BLOCK_SIZE + local
    valid = sequence < NUM_SEQS
    if NUM_CHUNKS == 1:
        context = tl.load(context_lengths + sequence, valid, other=0)
        tiles = (context.to(tl.int64) + 255) // 256
        total_tiles = tl.sum(tiles, 0)
    else:
        prefix = tl.load(tile_prefix + sequence, valid, other=0)
        previous = tl.load(tile_prefix + sequence - 1, valid & (local > 0), other=0)
        tiles = tl.where(valid, prefix - previous, 0)
        chunk = tl.arange(0, BLOCK_CHUNKS)
        total_tiles = tl.sum(
            tl.load(tile_totals + chunk, chunk < NUM_CHUNKS, other=0), 0
        )
    total_cost = tl.maximum(total_tiles + NUM_SEQS * WORK_OVERHEAD, 1)
    counts = tl.minimum((tiles * NUM_GROUPS + total_cost - 1) // total_cost, MAX_PARTS)
    counts = tl.where(valid, tl.maximum(tl.minimum(counts, tiles), 1), 0)
    partials = tl.where(counts > 1, counts, 0)
    costs = tiles + counts * WORK_OVERHEAD
    work_prefix = tl.cumsum(counts, 0)
    partial_prefix = tl.cumsum(partials, 0)
    cost_prefix = tl.cumsum(costs, 0)
    tl.store(sequence_info + sequence * 4, counts, valid)
    tl.store(sequence_info + sequence * 4 + 1, work_prefix, valid)
    tl.store(sequence_info + sequence * 4 + 2, partial_prefix, valid)
    tl.store(sequence_info + sequence * 4 + 3, cost_prefix, valid)
    tl.store(chunk_prefix + block * 3, tl.sum(counts, 0))
    tl.store(chunk_prefix + block * 3 + 1, tl.sum(partials, 0))
    tl.store(chunk_prefix + block * 3 + 2, tl.sum(costs, 0))


@triton.jit(repr=make_kernel_repr("_pa_ps_chunk_scan", ["BLOCK_CHUNKS", "num_warps"]))
def _pa_ps_chunk_scan(
    chunk_prefix, NUM_CHUNKS: tl.constexpr, BLOCK_CHUNKS: tl.constexpr
):
    chunk = tl.arange(0, BLOCK_CHUNKS)
    for field in tl.static_range(3):
        values = tl.load(chunk_prefix + chunk * 3 + field, chunk < NUM_CHUNKS, other=0)
        tl.store(
            chunk_prefix + chunk * 3 + field, tl.cumsum(values, 0), chunk < NUM_CHUNKS
        )


@triton.jit
def _pa_ps_sequence_prefix(
    sequence_info, chunk_prefix, sequence, field: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    chunk = sequence // BLOCK_SIZE
    local = tl.load(sequence_info + sequence * 4 + field + 1)
    previous = tl.load(chunk_prefix + (chunk - 1) * 3 + field, chunk > 0, other=0)
    return local + previous


@triton.jit(
    repr=make_kernel_repr(
        "_pa_ps_write_metadata",
        ["NUM_HEADS", "GQA", "MAX_QLEN", "PAGE_SIZE", "BLOCK_PARTS", "num_warps"],
    )
)
def _pa_ps_write_metadata(
    context_lengths,
    qo_indptr,
    kv_indptr,
    sequence_info,
    chunk_prefix,
    work_info,
    reduce_indptr,
    reduce_final_map,
    reduce_partial_map,
    NUM_CHUNKS: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    GQA: tl.constexpr,
    MAX_QLEN: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_PARTS: tl.constexpr,
):
    sequence = tl.program_id(0)
    head = tl.program_id(1)
    count = tl.load(sequence_info + sequence * 4)
    work_end = _pa_ps_sequence_prefix(
        sequence_info, chunk_prefix, sequence, 0, BLOCK_SIZE
    )
    partial_end = _pa_ps_sequence_prefix(
        sequence_info, chunk_prefix, sequence, 1, BLOCK_SIZE
    )
    work_start = work_end - count
    partial_start = partial_end - tl.where(count > 1, count, 0)
    total_work = tl.load(chunk_prefix + (NUM_CHUNKS - 1) * 3)
    context = tl.load(context_lengths + sequence).to(tl.int64)
    tiles = (context + 255) // 256
    pages = (context + PAGE_SIZE - 1) // PAGE_SIZE
    page_start = tl.load(kv_indptr + sequence).to(tl.int64)
    query_start = tl.load(qo_indptr + sequence)
    query_end = tl.load(qo_indptr + sequence + 1)
    part = tl.arange(0, BLOCK_PARTS).to(tl.int64)
    active = part < count
    begin = part * tiles // count
    end = (part + 1) * tiles // count
    slot = head * total_work + work_start + part
    partial = tl.where(count > 1, (partial_start + part) * MAX_QLEN, -1)
    tl.store(work_info + slot * 8, sequence, active)
    tl.store(work_info + slot * 8 + 1, partial, active)
    tl.store(work_info + slot * 8 + 2, query_start, active)
    tl.store(work_info + slot * 8 + 3, query_end, active)
    tl.store(
        work_info + slot * 8 + 4,
        page_start + tl.minimum(begin * (256 // PAGE_SIZE), pages),
        active,
    )
    tl.store(
        work_info + slot * 8 + 5,
        page_start + tl.minimum(end * (256 // PAGE_SIZE), pages),
        active,
    )
    tl.store(work_info + slot * 8 + 6, 0, active)
    tl.store(work_info + slot * 8 + 7, ((head + 1) * GQA << 16) | (head * GQA), active)
    if head == 0:
        if sequence == 0:
            tl.store(reduce_indptr, 0)
        tl.store(reduce_indptr + sequence + 1, partial_end)
        tl.store(reduce_final_map + sequence * 2, query_start)
        tl.store(reduce_final_map + sequence * 2 + 1, query_end)
        tl.store(
            reduce_partial_map + partial_start + part, partial, active & (count > 1)
        )


@triton.jit(
    repr=make_kernel_repr(
        "_pa_ps_schedule",
        ["NUM_SEQS", "NUM_CU", "NUM_GROUPS", "WORK_OVERHEAD", "BLOCK_CU", "num_warps"],
    )
)
def _pa_ps_schedule(
    context_lengths,
    sequence_info,
    chunk_prefix,
    work_metadata_ptrs,
    work_indptr,
    work_info,
    NUM_SEQS: tl.constexpr,
    NUM_CU: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    NUM_CHUNKS: tl.constexpr,
    WORK_OVERHEAD: tl.constexpr,
    LOG_SEQS: tl.constexpr,
    LOG_PARTS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_CU: tl.constexpr,
):
    group = tl.arange(0, BLOCK_CU)
    head = group // NUM_GROUPS
    local_group = group % NUM_GROUPS
    total_work = tl.load(chunk_prefix + (NUM_CHUNKS - 1) * 3)
    total_cost = tl.load(chunk_prefix + (NUM_CHUNKS - 1) * 3 + 2)
    target = local_group.to(tl.int64) * total_cost * 2
    lower = tl.full((BLOCK_CU,), 0, tl.int32)
    upper = tl.full((BLOCK_CU,), NUM_SEQS - 1, tl.int32)
    for step in range(LOG_SEQS):
        middle = (lower + upper) // 2
        prefix = _pa_ps_sequence_prefix(
            sequence_info, chunk_prefix, middle, 2, BLOCK_SIZE
        )
        before = prefix * (2 * NUM_GROUPS) < target
        lower = tl.where(before, middle + 1, lower)
        upper = tl.where(before, upper, middle)
    sequence = lower
    count = tl.load(sequence_info + sequence * 4)
    context = tl.load(context_lengths + sequence).to(tl.int64)
    tiles = (context + 255) // 256
    cost_end = _pa_ps_sequence_prefix(
        sequence_info, chunk_prefix, sequence, 2, BLOCK_SIZE
    )
    cost_start = cost_end - tiles - count * WORK_OVERHEAD
    lower = tl.full((BLOCK_CU,), 0, tl.int64)
    upper = count
    for step in range(LOG_PARTS):
        middle = (lower + upper) // 2
        begin = middle * tiles // count
        end = (middle + 1) * tiles // count
        midpoint = 2 * cost_start + begin + end + (2 * middle + 1) * WORK_OVERHEAD
        before = midpoint * NUM_GROUPS < target
        lower = tl.where(before, middle + 1, lower)
        upper = tl.where(before, upper, middle)
    work_end = _pa_ps_sequence_prefix(
        sequence_info, chunk_prefix, sequence, 0, BLOCK_SIZE
    )
    tl.store(
        work_indptr + group,
        head * total_work + work_end - count + lower,
        group <= NUM_CU,
    )
    tl.store(work_metadata_ptrs, work_indptr.to(tl.uint64))
    tl.store(work_metadata_ptrs + 1, work_info.to(tl.uint64))
