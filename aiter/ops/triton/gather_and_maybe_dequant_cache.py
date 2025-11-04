import torch
import triton
from aiter.ops.triton._triton_kernels.gather_and_maybe_dequant_cache import (
    _gather_and_maybe_dequant_cache
)


def gather_and_maybe_dequant_cache(
    src_cache: torch.Tensor,     # [NUM_BLOCKS, BLOCK_SIZE, *ENTRIES]
    dst: torch.Tensor,           # [TOT_TOKENS, *ENTRIES]
    block_table: torch.Tensor,   # [BATCH, BLOCK_INDICES]
    cu_seq_lens: torch.Tensor,   # [BATCH+1]
    batch_size: int,
    kv_cache_dtype: str = "auto",
    scale: torch.Tensor = None,
    seq_starts: torch.Tensor = None,
    num_splits: int = None,
):
    block_size = src_cache.size(1)
    entry_size = src_cache.flatten(2, -1).size(2)

    # Strides
    block_table_stride = block_table.stride(0)
    cache_block_stride = src_cache.stride(0)
    cache_entry_stride = src_cache.stride(1)
    dst_entry_stride = dst.stride(0)

    # Determine num_splits
    if num_splits is None:
        if batch_size > 128:
            num_splits = 2
        elif batch_size > 64:
            num_splits = 4
        else:
            num_splits = 256 // batch_size

    USE_FP8 = kv_cache_dtype != "auto"
    HAS_SEQ_STARTS = seq_starts is not None

    _gather_and_maybe_dequant_cache[(batch_size, num_splits,)](
        src_cache,
        dst,
        block_table,
        cu_seq_lens,
        seq_starts if HAS_SEQ_STARTS else src_cache,  # Dummy if None
        scale if USE_FP8 else src_cache,  # Dummy if None
        cache_block_stride=cache_block_stride,
        cache_entry_stride=cache_entry_stride,
        dst_entry_stride=dst_entry_stride,
        block_table_stride=block_table_stride,
        BLOCK_SIZE=block_size,
        ENTRY_SIZE=entry_size,
        ENTRY_SIZE_POW2=triton.next_power_of_2(entry_size),
        NUM_SPLITS=num_splits,
        BLOCK_SIZE_M=1024,
        USE_FP8=USE_FP8,
        HAS_SEQ_STARTS=HAS_SEQ_STARTS,
    )
