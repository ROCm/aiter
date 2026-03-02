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
    sq_sum = tl.sum(input_tile * input_tile, axis=0) # [D]
    
    actual_block_size = tl.mean(BLOCK_SIZE, remained_seq)
    mean = sum / actual_block_size 
    var = (sq_sum / actual_block_size) - (mean * mean)
    
    mean_offset = off_z * stride_mz + off_h * stride_mh + start_s * stride_ms + tl.arange(0, BLOCK_SIZE) * stride_md
    mean_ptr = Mean_ptr + mean_offset
    tl.store(mean_ptr, mean)

    simiarlity_offset = off_z * stride_sz + off_h * stride_sh + start_s * stride_ss
    simiarlity_ptr = Simiarlity_ptr + simiarlity_offset
    tl.store(simiarlity_ptr, var)


def sparge_preprocess(q, k, BLOCK_M, BLOCK_N, layout):
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

    # Apply softmax
    similarity = torch.softmax(similarity, dim=-1)

    return q_mean, q_similarity, k_mean, k_similarity