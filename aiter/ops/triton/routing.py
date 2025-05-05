from functools import partial

import torch
import triton
import triton.language as tl

def get_config_heuristic(M, K, N):
    """
    Return the best Triton configuration based on input dimensions.

    Args:
        M: Batch dimension
        K: Hidden dimension
        N: Number of experts (16 or 128)
        TOPK: Top-k value (default: 1)

    Returns:
        triton.Config: Configuration for the Triton kernel
    """
    # Determine M bucket (small: <2048, medium: 2048-4095, large: 4096-8191, very_large: 8192+)
    m_bucket = (
        "very_large"
        if M >= 8192
        else "large"
        if M >= 4096
        else "medium"
        if M >= 2048
        else "small"
    )

    # Create parameter configuration using nested dictionaries
    configs = {
        # Format: {N: {m_bucket: (BLOCK_M, BLOCK_K, num_warps, num_stages, waves_per_eu, kpack)}}
        16: {
            "small": (16, 256, 4, 2, 3, 1),
            "medium": (16, 256, 4, 2, 3, 1),
            "large": (16, 256, 4, 2, 3, 2),
            "very_large": (32, 256, 4, 2, 0, 1),
        },
        128: {
            "small": (16, 256, 8, 1, 0, 1),
            "medium": (16, 256, 8, 1, 0, 2),
            "large": (16, 256, 8, 1, 2, 2),
            "very_large": (32, 128, 8, 2, 2, 2),
        },
    }

    # Get configuration parameters
    BLOCK_M, BLOCK_K, num_warps, num_stages, waves_per_eu, kpack = configs[N][m_bucket]

    # Return Triton configuration
    return triton.Config(
        {
            "BLOCK_M": BLOCK_M,
            "BLOCK_K": BLOCK_K,
            "matrix_instr_nonkdim": 16,  # Always 16
            "waves_per_eu": waves_per_eu,
            "kpack": kpack,
        },
        num_warps=num_warps,
        num_stages=num_stages,
        num_ctas=1,
    )


# @triton.autotune(
#     configs=[
#         triton.Config(
#             {
#                 "BLOCK_M": bm,
#                 "BLOCK_K": bk,
#                 "matrix_instr_nonkdim": matrix_instr_nonkdim,
#                 "waves_per_eu": waves_per_eu,
#                 "kpack": kpack,
#             },
#             num_warps=num_warps,
#             num_stages=num_stages,
#         )
#         for bm in [16, 32, 64]  # [32, 64, 128, 256]
#         for bk in [64, 128, 256]  # [32, 64, 128, 256]
#         for num_warps in [4, 8]  # [4, 8]
#         for matrix_instr_nonkdim in [16]
#         for waves_per_eu in [0, 2, 3]  # [0, 2, 3]
#         for kpack in [1, 2]  # [1, 2]
#         for num_stages in [1, 2]  # [1, 2]
#     ],
#     key=["M", "N", "K"],
# )
@triton.jit
def _routing_sigmoid_top1_kernel(
    X_ptr,
    W_ptr,
    topk_ids_ptr,
    topk_weights_ptr,
    M,
    N,
    K,
    stride_xm,
    stride_xk,
    stride_wk,
    stride_wn,
    stride_topk_ids_m,
    stride_topk_ids_n,
    stride_topk_weights_m,
    stride_topk_weights_n,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    TOPK: tl.constexpr,
    FUSED_SHARED_EXPERTS: tl.constexpr,
):
    # Program ID corresponds to the block index in M dimension
    pid_m = tl.program_id(axis=0)

    # Offsets for the current block
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    _TOPK: tl.constexpr = TOPK + 1 if FUSED_SHARED_EXPERTS else TOPK

    offs_topk = tl.arange(0, _TOPK)

    # Masks for bounds checking
    mask_m = offs_m < M
    mask_n = offs_n < N

    # Initialize accumulator for matmul (will be in float32 due to default acc_type)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension in chunks of BLOCK_K
    for k in range(0, K, BLOCK_K):
        # Compute pointers for A and B
        offs_k_iter = k + offs_k
        mask_k = offs_k_iter < K

        X_ptrs = X_ptr + (
            # pyre-ignore
            offs_m[:, None] * stride_xm + offs_k_iter[None, :] * stride_xk
        )
        W_ptrs = W_ptr + (
            offs_k_iter[:, None] * stride_wk + offs_n[None, :] * stride_wn
        )

        # Load A and B tiles
        # pyre-ignore
        x = tl.load(X_ptrs, mask=(mask_m[:, None] & mask_k[None, :]), other=0.0)
        w = tl.load(W_ptrs, mask=(mask_k[:, None] & mask_n[None, :]), other=0.0)

        # Compute partial matmul for the current block using FP16 inputs and FP32 accumulation
        acc = tl.dot(x, w, acc=acc)

    acc = tl.sigmoid(acc)
    # Get topk results
    topk_ids = tl.argmax(acc, axis=1, tie_break_left=True)  # Shape: (BLOCK_M,)
    topk_weights = tl.max(acc, axis=1)  # Shape: (BLOCK_M,)

    # Create buffers for results
    topk_ids_buffer = tl.zeros((BLOCK_M, _TOPK), dtype=tl.int32)
    topk_weights_buffer = tl.zeros((BLOCK_M, _TOPK), dtype=tl.float32)

    if FUSED_SHARED_EXPERTS:
        # Set the first column with broadcasting
        topk_ids_buffer = tl.where(
            (offs_topk[None, :] < _TOPK - 1), topk_ids[:, None], N
        )
        topk_weights_buffer = tl.where(
            (offs_topk[None, :] < _TOPK - 1), topk_weights[:, None], 1.0
        )
    else:
        topk_ids_buffer = topk_ids[:, None]
        topk_weights_buffer = topk_weights[:, None]

    topk_ids_ptrs = (
        topk_ids_ptr
        + offs_m[:, None] * stride_topk_ids_m
        + offs_topk[None, :] * stride_topk_ids_n
    )

    topk_weights_ptrs = (
        topk_weights_ptr
        + offs_m[:, None] * stride_topk_weights_m
        + offs_topk[None, :] * stride_topk_weights_n
    )

    tl.store(topk_ids_ptrs, topk_ids_buffer)
    tl.store(topk_weights_ptrs, topk_weights_buffer)


def routing_sigmoid_top1(x, w, topk, fused_shared_experts=False):
    # assert x.dtype == torch.bfloat16
    # assert w.dtype == torch.bfloat16
    x = x.view(-1, x.shape[-1])

    assert topk == 1

    # M: batch_size x seq_len, K: hidden_dim, N: num_experts
    M, K = x.shape
    Kb, N = w.shape
    assert K == Kb

    _topk = topk
    if fused_shared_experts:
        _topk += 1

    # Output tensor
    topk_ids = torch.empty((M, _topk), device=x.device, dtype=torch.int32)
    topk_weights = torch.empty((M, _topk), device=x.device, dtype=torch.float32)

    heuristc_config = get_config_heuristic(M, K, N)

    # Grid size
    def grid(META):
        return (triton.cdiv(M, META["BLOCK_M"]),)

    _routing_sigmoid_top1_kernel[grid](
        x,
        w,
        topk_ids,
        topk_weights,
        M,
        N,
        K,
        x.stride(0),
        x.stride(1),
        w.stride(0),
        w.stride(1),
        topk_ids.stride(0),
        topk_ids.stride(1),
        topk_weights.stride(0),
        topk_weights.stride(1),
        BLOCK_N=N,  # Set BLOCK_N to N (16)
        TOPK=topk,
        FUSED_SHARED_EXPERTS=fused_shared_experts,
        num_warps=heuristc_config.num_warps,
        num_stages=heuristc_config.num_stages,
        num_ctas=heuristc_config.num_ctas,
        **heuristc_config.kwargs,
    )

    return topk_ids, topk_weights


def torch_routing_sigmoid_top1(
    x, w, topk, fused_shared_experts=False, dummy_ids=None, dummy_weights=None
):
    scores = torch.matmul(x, w)  # [M, N]

    scores = torch.sigmoid(scores.to(torch.float32))  # [M, N]

    assert topk == 1

    topk_weights, topk_ids = torch.topk(scores, topk, dim=1)  # [M, topk]

    topk_ids = topk_ids.to(torch.int32)
    topk_weights = topk_weights.to(torch.float32)

    if fused_shared_experts:
        topk_ids = torch.cat(
            [
                topk_ids,
                dummy_ids,
            ],
            dim=1,
        )
        topk_weights = torch.cat(
            [topk_weights, dummy_weights],
            dim=1,
        )

    return topk_ids, topk_weights