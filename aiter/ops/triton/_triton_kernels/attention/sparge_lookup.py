import triton
import triton.language as tl
from aiter.ops.triton._triton_kernels.attention.fav3_sage_attention import (
    map_dims,
)
import torch

@triton.jit
def mean_pooling_kernel(
    Input_ptr,
    stride_z,
    stride_h,
    stride_s,
    stride_d,
    Mean_ptr,
    stride_mz,
    stride_mh,
    stride_ms,
    stride_md,
    Simiarlity_ptr,
    stride_sz,
    stride_sh,
    stride_ss,
    seqlen,
    D: tl.constexpr, 
    BLOCK_SIZE: tl.constexpr, # BLOCK_M or BLOCK_N
):
    # 1. IDENTIFY BLOCK
    start_s = tl.program_id(0)
    off_h = tl.program_id(1)
    off_z = tl.program_id(2)
    offs_d = tl.arange(0, D)
    
    offs_m = start_s * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    remained_seq = seqlen - start_s * BLOCK_SIZE 
    inpout_ptr_mask = offs_m[:, None] < seqlen

    input_offset = (
        Input_ptr + off_z * stride_z + off_h * stride_h
    )
    input_ptrs = input_offset + offs_m[:, None] * stride_s + offs_d[None, :] * stride_d
    input_tile = tl.load(input_ptrs, mask=inpout_ptr_mask, other=0.0) # Shape [BLOCK_Q, D]

    # 2. SELF-SIMILARITY CHECK (Single Pass Variance)
    # Variance = E[X^2] - (E[X])^2
    sum = tl.sum(input_tile, axis=0) # [D]
    
    actual_block_size = tl.minimum(BLOCK_SIZE, remained_seq)
    mean = sum / actual_block_size 

    input_tile = input_tile.to(tl.float32)

    x_norm = tl.sqrt(tl.sum(input_tile * input_tile, axis=1, keep_dims=True))
    x = (input_tile / x_norm).to(tl.float16)  # norm at D dim
    
    grams = tl.dot(x, tl.trans(x))
    sum_value = tl.sum(grams).to(tl.float32)
    cur_sim = (sum_value / (actual_block_size * actual_block_size))

    mean_offset = off_z * stride_mz + off_h * stride_mh + start_s * stride_ms + offs_d * stride_md
    mean_ptr = Mean_ptr + mean_offset
    tl.store(mean_ptr, mean)

    simiarlity_offset = off_z * stride_sz + off_h * stride_sh + start_s * stride_ss
    simiarlity_ptr = Simiarlity_ptr + simiarlity_offset
    tl.store(simiarlity_ptr, cur_sim)


# Theta is the threshold for simiarlity
def sparge_preprocess(q, k, BLOCK_M, BLOCK_N, theta, tau, layout):
    """
    Preprocesses query (Q) and key (K) tensors for the Sparge attention mechanism by computing block-level statistics and generating a sparsity mask.
    This function performs mean pooling on Q and K blocks using Triton kernels, computes block-to-block similarity scores, and determines which Key blocks should be attended to by each Query block based on thresholding ($\theta$) and cumulative probability distribution functions (TopCdf with $\tau$).
    Args:
        q (torch.Tensor): The query tensor. Shape depends on `layout`.
        k (torch.Tensor): The key tensor. Shape depends on `layout`.
        BLOCK_M (int): Block size for the query sequence dimension (used in tiling).
        BLOCK_N (int): Block size for the key sequence dimension (used in tiling).
        theta (float): A threshold value for similarity scores. Blocks with similarity scores below this threshold are marked as "sparse" (or fully attended to, depending on the implementation logic of the mask `M`). Specifically, if a block's internal similarity metric is less than `theta`, the corresponding row/column in the mask `M` is set to True.
        tau (float): The cumulative probability threshold (0 < tau <= 1.0) used for the TopCdf selection strategy.
        layout (str): The data layout format. Either "bshd" (Batch, Seq, Head, Dim) or "bhsd" (Batch, Head, Seq, Dim).
    Returns:
        tuple: A tuple containing:
            - q_mean (torch.Tensor): The mean-pooled representation of Q blocks.
            - q_similarity (torch.Tensor): The similarity/importance scores for Q blocks.
            - k_mean (torch.Tensor): The mean-pooled representation of K blocks.
            - k_similarity (torch.Tensor): The similarity/importance scores for K blocks.
    TopCdf Explanation:
        The "TopCdf" (Top Cumulative Distribution Function) logic selects a dynamic number of Key blocks for each Query block based on their contribution to the total attention mass.
        1. **Similarity:** It computes the approximate attention score between block means: $S = \text{softmax}(Q_{mean} @ K_{mean}^T)$.
        2. **Sorting:** For each query block, the key blocks are sorted by these scores in descending order.
        3. **CumSum:** A cumulative sum (CDF) is calculated over the sorted probabilities.
        4. **Selection:** The algorithm selects the top-k blocks whose cumulative probability sums up to `tau`. This ensures that the selected sparse blocks preserve at least `tau` amount of the estimated attention probability mass.
    """
    bshd_map = [0, 1, 2, 3] if layout == "bshd" else [0, 2, 1, 3]
    batch, seqlen_q, num_q_heads, head_dim = map_dims(q.shape, bshd_map)
    batch, seqlen_k, num_k_heads, head_dim = map_dims(k.shape, bshd_map)

    # Prepare output tensors
    num_blocks_q = triton.cdiv(seqlen_q, BLOCK_M)
    num_blocks_k = triton.cdiv(seqlen_k, BLOCK_N)
    
    q_mean = torch.zeros((batch, num_q_heads, num_blocks_q, head_dim), device=q.device, dtype=q.dtype)
    q_similarity = torch.zeros((batch, num_q_heads, num_blocks_q), device=q.device, dtype=torch.float32)
    
    k_mean = torch.zeros((batch, num_k_heads, num_blocks_k, head_dim), device=k.device, dtype=k.dtype)
    k_similarity = torch.zeros((batch, num_k_heads, num_blocks_k), device=k.device, dtype=torch.float32)
    
    # Get strides
    if layout == "bshd":
        q_stride_z, q_stride_s, q_stride_h, q_stride_d = q.stride()
        k_stride_z, k_stride_s, k_stride_h, k_stride_d = k.stride()
    else:  # bhsd
        q_stride_z, q_stride_h, q_stride_s, q_stride_d = q.stride()
        k_stride_z, k_stride_h, k_stride_s, k_stride_d = k.stride()
    
    qm_stride_z, qm_stride_h, qm_stride_s, qm_stride_d = q_mean.stride()
    qs_stride_z, qs_stride_h, qs_stride_s = q_similarity.stride()
    
    km_stride_z, km_stride_h, km_stride_s, km_stride_d = k_mean.stride()
    ks_stride_z, ks_stride_h, ks_stride_s = k_similarity.stride()
    
    # Launch kernel for Q
    grid_q = (num_blocks_q, num_q_heads, batch)
    mean_pooling_kernel[grid_q](
        q, q_stride_z, q_stride_h, q_stride_s, q_stride_d,
        q_mean, qm_stride_z, qm_stride_h, qm_stride_s, qm_stride_d,
        q_similarity, qs_stride_z, qs_stride_h, qs_stride_s,
        seqlen_q,
        D=head_dim,
        BLOCK_SIZE=BLOCK_M,
    )
    print(q_similarity)
    
    # Launch kernel for K
    grid_k = (num_blocks_k, num_k_heads, batch)
    mean_pooling_kernel[grid_k](
        k, k_stride_z, k_stride_h, k_stride_s, k_stride_d,
        k_mean, km_stride_z, km_stride_h, km_stride_s, km_stride_d,
        k_similarity, ks_stride_z, ks_stride_h, ks_stride_s,
        seqlen_k,
        D=head_dim,
        BLOCK_SIZE=BLOCK_N,
    )
    
    # Compute similarity: Q_mean @ K_mean^T
    similarity = torch.matmul(q_mean, k_mean.transpose(-2, -1))  # [batch, num_q_heads, num_blocks_q, num_blocks_k]

    # S[:, j] = -inf if skj < theta
    mask = k_similarity.unsqueeze(-2) < theta
    similarity = similarity.masked_fill(mask, float("-inf"))

    # Apply softmax
    similarity = torch.softmax(similarity, dim=-1)

    print(similarity)
    # TODO check impl
    # M[i, :] = TopCdf(Pˆ[i], τ )
    # Sort similarity scores in descending order to compute CDF
    sorted_probs, sorted_indices = torch.sort(similarity, descending=True, dim=-1)
    cdf = torch.cumsum(sorted_probs, dim=-1)
    
    # Identify which blocks are needed to reach cumulative probability tau
    # true means we keep it. We keep the first k elements where sum < tau, plus one more to cross the threshold.
    # The mask needs to be mapped back to original indices.
    # The logic here: find the cut-off index where CDF >= tau.
    cutoff_mask = cdf < tau
    # Because we check < tau, the first element that makes it >= tau is False. 
    # We want to include that "crossing" element. 
    # Shift mask to the right to select the one that crosses the threshold.
    cutoff_mask = torch.cat([torch.ones_like(cutoff_mask[..., :1], dtype=torch.bool), cutoff_mask[..., :-1]], dim=-1)
    
    # Create the block mask M. Initialize with zeros (all False).
    M = torch.zeros_like(similarity, dtype=torch.bool)
    # Scatter the True values back to the original positions based on sorted_indices
    M.scatter_(dim=-1, index=sorted_indices, src=cutoff_mask)

    # M[i, :] = 1, If sqi < θ ; 
    q_mask = q_similarity < theta
    # Broadcast q_mask to match M's shape [batch, num_q_heads, num_blocks_q, num_blocks_k]
    # q_similarity is [batch, num_q_heads, num_blocks_q]
    M = M | q_mask.unsqueeze(-1)

    # TODO shape of mask
    # M[:, j] = 1, If skj < θ ;
    k_mask = k_similarity < theta
    # k_similarity is [batch, num_k_heads, num_blocks_k]
    # Note: Assuming num_q_heads == num_k_heads or broadcastable for GQA/MQA handled by torch
    M = M | k_mask.unsqueeze(-2)

    return M