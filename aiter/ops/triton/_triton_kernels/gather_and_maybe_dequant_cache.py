import triton
import triton.language as tl


@triton.jit
def _gather_and_maybe_dequant_cache(
    src_cache_ptr,      # [NUM_BLOCKS, BLOCK_SIZE, *ENTRIES]
    dst_ptr,            # [TOT_TOKENS, *ENTRIES]
    block_table_ptr,    # [BATCH, BLOCK_INDICES]
    cu_seq_lens_ptr,    # [BATCH+1]
    seq_starts_ptr,     # Optional [BATCH]
    scale,              # Optional scale for FP8
    cache_block_stride,
    cache_entry_stride,
    dst_entry_stride,
    block_table_stride,
    BLOCK_SIZE: tl.constexpr,
    ENTRY_SIZE: tl.constexpr,
    ENTRY_SIZE_POW2: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,  # Threads per block
    USE_FP8: tl.constexpr,
    HAS_SEQ_STARTS: tl.constexpr,
):

    # Grid dimensions: (batch_size, num_splits)
    bid = tl.program_id(0)
    split_id = tl.program_id(1)

    # Load sequence boundaries
    seq_start = tl.load(cu_seq_lens_ptr + bid)
    seq_end = tl.load(cu_seq_lens_ptr + bid + 1)
    seq_len = seq_end - seq_start

    # Calculate blocks for this sequence
    tot_blocks = tl.cdiv(seq_len, BLOCK_SIZE)
    split_blocks = tl.cdiv(tot_blocks, NUM_SPLITS)

    split_start = split_id * split_blocks
    split_end = tl.minimum((split_id + 1) * split_blocks, tot_blocks)

    # Early exit if this split has no work
    if split_start >= tot_blocks:
        return

    is_last_split = (split_end == tot_blocks)

    # Handle partial block
    full_blocks_end = split_end
    partial_block_size = seq_len % BLOCK_SIZE
    if is_last_split and partial_block_size > 0:
        full_blocks_end = split_end - 1

    # Get block table offset
    batch_offset = bid * block_table_stride
    offset = 0
    if HAS_SEQ_STARTS:
        offset = tl.load(seq_starts_ptr + bid) // BLOCK_SIZE

    block_table_base = block_table_ptr + batch_offset + offset

    # Adjust dst pointer
    dst_base = dst_ptr + seq_start * dst_entry_stride

    # Process full blocks
    for block in range(split_start, full_blocks_end):
        # Load block ID
        block_id = tl.load(block_table_base + block)

        # Calculate pointers
        src_block_base = src_cache_ptr + block_id * cache_block_stride
        dst_block_base = dst_base + block * BLOCK_SIZE * dst_entry_stride

        # Copy each entry in the block
        for eid in range(BLOCK_SIZE):
            src_entry_ptr = src_block_base + eid * cache_entry_stride
            dst_entry_ptr = dst_block_base + eid * dst_entry_stride

            entry_offsets = tl.arange(0, ENTRY_SIZE_POW2)

            if USE_FP8:
                entry = tl.load(src_entry_ptr + entry_offsets, mask=entry_offsets < ENTRY_SIZE).to(tl.float32)
                entry = entry * scale
                tl.store(dst_entry_ptr + entry_offsets, entry.to(dst_ptr.dtype.element_ty), mask=entry_offsets < ENTRY_SIZE)
            else:
                entry = tl.load(src_entry_ptr + entry_offsets, mask=entry_offsets < ENTRY_SIZE)
                tl.store(dst_entry_ptr + entry_offsets, entry, mask=entry_offsets < ENTRY_SIZE)

    # Process partial block
    if is_last_split and partial_block_size > 0:
        block_id = tl.load(block_table_base + full_blocks_end)

        src_block_base = src_cache_ptr + block_id * cache_block_stride
        dst_block_base = dst_base + full_blocks_end * BLOCK_SIZE * dst_entry_stride

        for eid in range(partial_block_size):
            src_entry_ptr = src_block_base + eid * cache_entry_stride
            dst_entry_ptr = dst_block_base + eid * dst_entry_stride

            entry_offsets = tl.arange(0, ENTRY_SIZE_POW2)

            if USE_FP8:
                entry = tl.load(src_entry_ptr + entry_offsets, mask=entry_offsets < ENTRY_SIZE).to(tl.float32)
                entry = entry * scale
                tl.store(dst_entry_ptr + entry_offsets, entry.to(dst_ptr.dtype.element_ty), mask=entry_offsets < ENTRY_SIZE)
            else:
                entry = tl.load(src_entry_ptr + entry_offsets, mask=entry_offsets < ENTRY_SIZE)
                tl.store(dst_entry_ptr + entry_offsets, entry, mask=entry_offsets < ENTRY_SIZE)
