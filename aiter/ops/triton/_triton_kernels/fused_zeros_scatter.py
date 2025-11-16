import functools
import json
import os
import triton
import triton.language as tl
from ..utils._triton.pid_preprocessing import pid_grid, remap_xcd
from ..utils._triton import arch_info
from ..utils.core import AITER_TRITON_CONFIGS_PATH

@triton.jit
def _scatter_with_zero_kernel(
    a, indices, values, n_indices, n_a, num_cols,
    BLOCK_SIZE: tl.constexpr = 1024
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # --- Step 1: zero out a ---
    zero_mask = offsets < n_a
    tl.store(a + offsets, 0, mask=zero_mask)

    # --- Step 2: scatter indices into a ---
    # For scatter with dim=0: element at column i goes to row indices[i]
    # In flattened space: flat_idx = indices[i] * num_cols + i
    scatter_mask = offsets < n_indices
    row_idx = tl.load(indices + offsets, mask=scatter_mask, other=0)
    val = tl.load(values + offsets, mask=scatter_mask, other=0)
    flat_idx = row_idx * num_cols + offsets
    tl.store(a + flat_idx, val, mask=scatter_mask)
