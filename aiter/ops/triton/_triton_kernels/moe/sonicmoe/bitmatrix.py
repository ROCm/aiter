import triton
import triton.language as tl

from aiter.ops.triton._triton_kernels.moe.moe_routing.utils import keyed_add
from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr

_bitmatrix_stage1_repr = make_kernel_repr(
    "sonicmoe_bitmatrix_stage1", ["E", "BLOCK_M", "BLOCK_N"]
)
_bitmatrix_stage2_repr = make_kernel_repr(
    "sonicmoe_bitmatrix_stage2", ["K_POW2", "K", "TOKENS_PER_BLOCK"]
)


# Adapted from https://github.com/triton-lang/triton/blob/434aecbe933af6a8d49595d4197bfc3df7618748/python/triton_kernels/triton_kernels/tensor_details/bitmatrix.py#L44
@triton.jit(repr=_bitmatrix_stage1_repr)
def _bitmatrix_metadata_compute_stage1(
    expert_freq_ptr,
    expert_freq_offs_ptr,
    E: tl.constexpr,
    partial_sum_ptr,
    n_tiles,
    TK,
    BLOCK_M: tl.constexpr,  # chunk size for iterating over tiles per expert
    BLOCK_N: tl.constexpr,  # chunk size for iterating over experts in cumsum
):
    pid = tl.program_id(0)
    if pid < E:
        # Convert each expert's tile counts to exclusive prefixes.
        expert_partial_sum_ptr = partial_sum_ptr + pid * n_tiles
        curr_sum = 0
        for start in range(0, n_tiles, BLOCK_M):
            offs = start + tl.arange(0, BLOCK_M)
            tile_counts = tl.load(
                expert_partial_sum_ptr + offs, mask=offs < n_tiles, other=0
            )
            excl_cumsum = tl.cumsum(tile_counts, 0) - tile_counts + curr_sum
            curr_sum += tl.sum(tile_counts, 0)
            tl.store(expert_partial_sum_ptr + offs, excl_cumsum, mask=offs < n_tiles)
    elif pid == E:
        # Compute each expert's global output start.
        curr_sum = 0
        for start in tl.static_range(0, E, BLOCK_N):
            offs = start + tl.arange(0, BLOCK_N)
            expert_freq = tl.load(expert_freq_ptr + offs, mask=offs < E, other=0)
            excl_cumsum = tl.cumsum(expert_freq, 0) - expert_freq + curr_sum
            curr_sum += tl.sum(expert_freq, 0)
            tl.store(expert_freq_offs_ptr + offs, excl_cumsum, mask=offs < E)
    elif pid == E + 1:
        tl.store(expert_freq_offs_ptr + E, TK)


# Adapted from https://github.com/triton-lang/triton/blob/434aecbe933af6a8d49595d4197bfc3df7618748/python/triton_kernels/triton_kernels/tensor_details/bitmatrix.py#L44
@triton.jit(repr=_bitmatrix_stage2_repr)
def _bitmatrix_metadata_compute_stage2(
    s_scatter_idx_ptr,
    s_reverse_scatter_idx_ptr,
    x_gather_idx_ptr,
    topk_indices_ptr,
    T,
    partial_sum_ptr,
    n_tiles,
    expert_offs_ptr,
    K_POW2: tl.constexpr,  # padded K, == BLOCK_SIZE / BLOCK
    K: tl.constexpr,  # actual experts per token
    TOKENS_PER_BLOCK: tl.constexpr,  # tokens per tile
):
    # Sort one tile by expert and produce forward and inverse permutations.
    BLOCK_SIZE: tl.constexpr = TOKENS_PER_BLOCK * K_POW2
    IS_POW2_K: tl.constexpr = K == K_POW2  # fast path: no padding waste
    tl.static_assert(BLOCK_SIZE <= 32768)

    pid_m = tl.program_id(0)
    offs_local = tl.arange(
        0, BLOCK_SIZE
    )  # position within this tile's flat [BLOCK*K_POW2] space
    offs_global = pid_m * BLOCK_SIZE + offs_local
    mask = offs_global < T * K_POW2

    # Power-of-two K permits flat loads without padding gaps.
    if IS_POW2_K:
        expert = tl.load(topk_indices_ptr + offs_global, mask=mask, other=-1).to(
            tl.uint32
        )
    else:
        token_i_local = offs_local // K_POW2
        k_slot = offs_local % K_POW2
        token_i_global = pid_m * TOKENS_PER_BLOCK + token_i_local
        load_mask = mask & (k_slot < K)
        safe_k = tl.minimum(k_slot, K - 1)
        expert = tl.load(
            topk_indices_ptr + token_i_global * K + safe_k,
            mask=load_mask,
            other=-1,
        ).to(tl.uint32)

    # Pack expert and local offset into uint32 for a stable local sort.
    kv_pairs = tl.sort(((expert << 16) | offs_local).to(tl.uint32), 0)
    expert = kv_pairs >> 16
    mask = expert != 0xFFFF  # exclude padding/OOB slots

    scan_input = (kv_pairs & 0xFFFF0000) | 0x00000001
    inclusive_run_lengths = tl.associative_scan(scan_input, 0, keyed_add)
    within_expert_rank = (
        inclusive_run_lengths - 1
    ) & 0xFFFF  # exclusive = inclusive - 1

    s_reverse_scatter_idx = tl.load(
        partial_sum_ptr + pid_m + expert * n_tiles, mask=mask
    )
    s_reverse_scatter_idx += tl.load(expert_offs_ptr + expert, mask=mask)
    s_reverse_scatter_idx += within_expert_rank

    if IS_POW2_K:
        presort_offs = kv_pairs & 0xFFFF
        entry_idx = pid_m * BLOCK_SIZE + presort_offs
        tl.store(
            s_reverse_scatter_idx_ptr + entry_idx, s_reverse_scatter_idx, mask=mask
        )
        tl.store(s_scatter_idx_ptr + s_reverse_scatter_idx, entry_idx, mask=mask)
        tl.store(
            x_gather_idx_ptr + s_reverse_scatter_idx, entry_idx // K_POW2, mask=mask
        )
    else:
        presort_offs = kv_pairs & 0xFFFF
        token_i_global_s = pid_m * TOKENS_PER_BLOCK + presort_offs // K_POW2
        entry_idx = token_i_global_s * K + presort_offs % K_POW2
        tl.store(
            s_reverse_scatter_idx_ptr + entry_idx, s_reverse_scatter_idx, mask=mask
        )
        tl.store(s_scatter_idx_ptr + s_reverse_scatter_idx, entry_idx, mask=mask)
        tl.store(x_gather_idx_ptr + s_reverse_scatter_idx, token_i_global_s, mask=mask)
