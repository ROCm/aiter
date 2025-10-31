# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import torch
import pytest
from typing import Dict
import triton

from aiter.ops.triton.moe_op import (
    fused_moe as triton_moe,
    moe_set_use_persistent_kernel as triton_moe_set_use_persistent_kernel,
)
from aiter.ops.triton.moe_op_e2e import e2e_moe as triton_e2e_moe
from aiter.ops.triton.moe_op_silu_fused import (
    fused_moe_silu as triton_moe_silu,
    moe_set_use_persistent_kernel as triton_moe_silu_set_use_persistent_kernel,
)
from aiter.ops.triton.moe_op_gelu import (
    fused_moe_gelu as triton_moe_gelu,
    moe_set_use_persistent_kernel as triton_moe_gelu_set_use_persistent_kernel,
)
from aiter.ops.triton.utils.moe_config_utils import (
    get_optimal_moe_config_func,
    get_optimal_moe_e2e_config_func,
)
import aiter.ops.triton.utils._triton.arch_info as arch_info

from aiter.ops.triton.utils.types import torch_to_triton_dtype

DEBUG_MODE = False


def torch_silu_and_mul_ref(input):
    """
    Performs the SiLU activation on the first half of the input tensor and
    multiplies it element-wise with the second half.
    Args:
        input (torch.Tensor): Input tensor of shape [..., 2 * d].
        param (float): Parameter for the SiLU activation function.
    Returns:
        torch.Tensor: Output tensor of shape [..., d].
    """
    dtype = input.dtype
    d = input.size(-1) // 2
    A, B = input[:, :d], input[:, d:]

    silu_A = A / (1.0 + torch.exp(-A.float()))

    output = silu_A * B

    return output.to(dtype)


def torch_moe_ref(
    a,
    b,
    c,
    a_scale,
    b_scale,
    b_zp,
    group_size,
    topk_ids,
    topk_weights,
    routed_weight,
    dtype,
    fp8_w8a8,
    int8_w8a16,
    int4_w4a16,
    gelu=False,
    blockshape=None,
):
    use_block_scale = (blockshape is not None) and (len(blockshape) == 2)
    if use_block_scale:
        blockshape_n, blockshape_k = blockshape

    if fp8_w8a8 and a_scale is None:
        if use_block_scale:
            a, _, a_scale = quantize_fp8_a(a, blockshape_k=blockshape_k)
        else:
            a, _, a_scale = quantize_fp8(a)

    M, top_k, N_half = c.shape
    _, K = a.shape
    N = N_half * 2

    if int4_w4a16:
        b = torch.repeat_interleave(b, repeats=2, dim=2)  # Expand to (E, N, K)
        b_shifter = ((torch.arange(0, K, device=b.device) % 2) * 4)[None, None, :]
        b = (b >> b_shifter) & 0xF
        b_scale = torch.repeat_interleave(
            b_scale, repeats=group_size, dim=2
        )  # (E, N, K)
        if b_zp is not None:
            b_zp = torch.repeat_interleave(
                b_zp, repeats=2, dim=1
            )  # (E,N//2,K//group_size) -> (E, N, K // group_size)
            b_zp = torch.repeat_interleave(
                b_zp, repeats=group_size, dim=2
            )  # (E,N,K//group_size) -> (E, N, K)
            b_zp_shifter = ((torch.arange(0, N, device=b.device) % 2) * 4)[
                None, :, None
            ]
            b_zp = (b_zp >> b_zp_shifter) & 0xF
            b = (b - b_zp) * b_scale
        else:
            b = (b - 8) * b_scale

    # Repeat a -> (M, top_k, K)
    a_expanded = a.unsqueeze(1).repeat(1, top_k, 1)
    # (M, top_k, N, K)
    if fp8_w8a8:
        if use_block_scale:
            b_indexed = b[topk_ids]
        else:
            b_indexed = b.half()[topk_ids]
    else:
        b_indexed = b[topk_ids]

    if fp8_w8a8 and use_block_scale:
        # Emulate the fp8 blockscale matmul
        sK = K // blockshape_k
        # sN = N // blockshape_n
        a_expanded = a_expanded.to(torch.float32).reshape(
            -1, sK, blockshape_k
        )  # (M*top_k, sK, blockshape_k)
        b_indexed = b_indexed.to(torch.float32).reshape(
            -1, N, sK, blockshape_k
        )  # (M*top_k, sN, blockshape_n, sK, blockshape_k)
        # keep sK dimension in result for scaling
        c_fp8 = torch.einsum("Mab, MNab -> MaN", a_expanded, b_indexed)
        # (M*top_k, sN, blockshape_n, sK)
        a_scale_expanded = (
            a_scale.unsqueeze(1)
            .repeat(1, top_k, 1)
            .reshape(-1, sK)
            .unsqueeze(-1)  # (M*top_k, sK, 1)
        )
        b_scale_expanded = (
            b_scale[topk_ids.flatten()]
            .permute(0, 2, 1)
            .repeat_interleave(blockshape_n, dim=2)
        )[..., :N]
        # (M*top_k, sK, N)
        # combined scales
        scale_matrix = a_scale_expanded * b_scale_expanded
        # (M*top_k, sK, N)
        # apply scales, then reduce over sK
        
        # temp = c_fp8 * scale_matrix
        c = (c_fp8 * scale_matrix).sum(dim=1).reshape(-1, N)
        # (M*top_k, N)
    else:
        c = torch.einsum("mek,menk->men", a_expanded.to(dtype), b_indexed.to(dtype))

    if routed_weight:
        c *= topk_weights.unsqueeze(-1)

    if not routed_weight and gelu:
        c = 0.5 * c * (1.0 + torch.tanh(0.7978845608 * (c + 0.044715 * c * c * c)))

    if fp8_w8a8 and not use_block_scale:
        c = c * b_scale[topk_ids].unsqueeze(-1)
        c = c * a_scale
        c = c.to(dtype)

    if int8_w8a16:
        c = c * b_scale[topk_ids].unsqueeze(-1)
        c = c.to(dtype)

    return c


def _moe_align_block_size(
    topk_ids: torch.Tensor,
    num_experts: int,
    top_k: int,
    block_size: int,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_pad: torch.Tensor,
) -> None:
    M, top_k = topk_ids.shape

    expert_to_tokens = [[] for _ in range(num_experts)]
    # For each token, for each selected expert, we append (token_id, expert)
    for token_id in range(M):
        for j in range(top_k):
            e_id = topk_ids[token_id, j].item()
            expert_to_tokens[e_id].append(token_id * top_k + j)

    # Reorder tokens block by block, padding if needed
    reordered_token_ids = []
    reordered_expert_ids = []

    for e_id in range(num_experts):
        tokens_for_expert = expert_to_tokens[e_id]
        num_tokens = len(tokens_for_expert)

        n_blocks = (num_tokens + block_size - 1) // block_size
        # If not a multiple of block_size, pad up to the next multiple
        padded_size = n_blocks * block_size

        # Reorder all actual tokens for expert e_id
        reordered_token_ids.extend(tokens_for_expert)
        # reordered_expert_ids.extend([e_id]*num_tokens)
        reordered_expert_ids.extend([e_id] * n_blocks)

        # Pad with dummy token_id = topk_ids.numel()
        if padded_size > num_tokens:
            pad_count = padded_size - num_tokens
            reordered_token_ids.extend([topk_ids.numel()] * pad_count)

    token_length = len(reordered_token_ids)
    expert_length = len(reordered_expert_ids)

    sorted_token_ids[:token_length] = torch.tensor(
        reordered_token_ids,
        dtype=sorted_token_ids.dtype,
        device=sorted_token_ids.device,
    )
    expert_ids[:expert_length] = torch.tensor(
        reordered_expert_ids, dtype=expert_ids.dtype, device=expert_ids.device
    )

    # Fill remainder with topk_ids.numel() if these arrays are bigger than total_length
    if token_length < sorted_token_ids.numel():
        sorted_token_ids[token_length:] = topk_ids.numel()
    if expert_length < expert_ids.numel():
        expert_ids[expert_length:] = topk_ids.numel()

    num_tokens_post_pad.fill_(token_length)


def torch_moe_align_block_size_ref(
    topk_ids: torch.Tensor, block_size: int, num_experts: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Aligns the token distribution across experts to be compatible with block size for matrix multiplication.

    Parameters:
    - topk_ids: A tensor of shape [total_tokens, top_k] representing the top-k expert indices for each token.
    - block_size: The block size used in block matrix multiplication.
    - num_experts: The total number of experts.

    Returns:
    - sorted_token_ids: A tensor containing the sorted token indices according to their allocated expert.
    - expert_ids: A tensor indicating the assigned expert index for each block.
    - num_tokens_post_padded: The total number of tokens after padding, ensuring divisibility by block_size.

    This function pads the number of tokens that each expert needs to process so that it is divisible by block_size.
    Padding ensures that during block matrix multiplication, the dimensions align correctly.

    Example:
    Given topk_ids = [[2, 3, 4], [1, 2, 4], [1, 3, 4], [1, 2, 3]], block_size = 4, and num_experts = 4:
    - We initially have 12 tokens (after repeating 'top_k' times) and 4 experts, with each expert needing to process 3 tokens.
    - As block_size is 4, we pad 1 token for each expert.
    - First, flatten topk_ids to [2, 3, 4, 1, 2, 4, 1, 3, 4, 1, 2, 3].
    - Then append padding tokens [12, 12, 12, 12] for each block.
    - After sorting by expert index, we obtain token_ids [3, 6, 9, 12, 0, 4, 10, 12, 1, 7, 11, 12, 2, 5, 8, 12].
        Tokens 12 are non-existent (padding) and are ignored in the subsequent matrix multiplication.
    - The padding ensures that the total number of tokens is now divisible by block_size for proper block matrix operations.
    """
    top_k = topk_ids.shape[1]
    sorted_ids = torch.empty(
        (topk_ids.numel() + num_experts * (block_size - 1),),
        dtype=torch.int32,
        device=topk_ids.device,
    )
    expert_ids = torch.empty(
        (topk_ids.numel() + num_experts,), dtype=torch.int32, device=topk_ids.device
    )
    sorted_ids.fill_(topk_ids.numel())
    num_tokens_post_pad = torch.empty((1), dtype=torch.int32, device=topk_ids.device)
    _moe_align_block_size(
        topk_ids,
        num_experts,
        top_k,
        block_size,
        sorted_ids,
        expert_ids,
        num_tokens_post_pad,
    )

    return sorted_ids, expert_ids, num_tokens_post_pad


def torch_moe_gemm2(
    a,
    b,
    c,
    b_scale,
    topk_ids,
    topk_weights,
    routed_weight,
    dtype,
    fp8_w8a8,
    blockshape=None,
):
    use_block_scale = (blockshape is not None) and (len(blockshape) == 2)
    if use_block_scale:
        blockshape_n, blockshape_k = blockshape

    out_dtype = c.dtype

    E, K, N_HALF = b.shape

    if fp8_w8a8 and use_block_scale:
        # dequantize w2
        b_scale = b_scale.repeat_interleave(blockshape_k, dim=1).repeat_interleave(blockshape_n, dim=2)[:, :K, :N_HALF]  # (E, K, N_HALF)
        b = b.to(torch.float32) * b_scale.to(torch.float32)

    b_indexed = b[topk_ids]

    out = torch.einsum(
        "men,mekn->mek", a.to(torch.bfloat16), b_indexed.to(torch.bfloat16)
    )

    if routed_weight:
        out = out * topk_weights.unsqueeze(-1)

    return out.to(out_dtype)


def quantize_fp8(
    tensor: torch.Tensor, dim=(), blockshape=None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    use_block_scale = blockshape is not None and len(blockshape) == 2
    use_dim_quantization = dim is not None and len(dim) == 2

    if use_block_scale:
        blockshape_n, blockshape_k = blockshape[0], blockshape[1]
        use_block_scale = use_block_scale and ((blockshape_n) and (blockshape_k))

    assert not (use_block_scale and use_dim_quantization)

    dev = arch_info.get_device()
    if dev == "MI350X":
        fp8_type = torch.float8_e4m3fn
    else:
        fp8_type = torch.float8_e4m3fnuz

    # If using block-scale quantization, reshape the tensor to merge block dimensions
    if use_block_scale:
        e, n, k = tensor.shape
        # Reshape to merge the second and third dimensions (n direction)
        # and the fourth and fifth dimensions (k direction)
        tensor = tensor.reshape(
            e, n // blockshape_n, blockshape_n, k // blockshape_k, blockshape_k
        )
        tensor = tensor.permute(0, 2, 4, 1, 3)
        tensor = tensor.reshape(
            e, blockshape_n * blockshape_k, n // blockshape_n, k // blockshape_k
        )
        max_vals = torch.max(tensor, 1, keepdim=True).values
    else:
        quantize_dim = [i for i in range(tensor.dim()) if i not in dim]
        max_vals = tensor.abs().amax(dim=quantize_dim, keepdim=True)
    max_repr_val = torch.finfo(fp8_type).max
    max_vals[max_vals == 0] = 1e-8  # Avoid division by zero

    # Compute scale factors for each channel
    scale: torch.Tensor = max_repr_val / max_vals.to(torch.float32)

    # Quantize the tensor
    tensor = tensor * scale
    tensor.clamp_(-max_repr_val, max_repr_val)
    tensor_quantized = tensor.to(fp8_type)

    if use_block_scale:
        tensor_quantized = tensor_quantized.reshape(
            e, blockshape_n, blockshape_k, n // blockshape_n, k // blockshape_k
        )
        tensor_quantized = tensor_quantized.permute(0, 3, 1, 4, 2)
        tensor_quantized = tensor_quantized.reshape(e, n, k)
        scale = scale.reshape(e, n // blockshape_n, k // blockshape_k)
    else:
        scale = scale.squeeze(dim=quantize_dim)

    return tensor_quantized, scale, 1 / scale


def quantize_fp8_a(
    a: torch.Tensor, blockshape_k=1
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    M, K = a.shape
    dev = arch_info.get_device()
    if dev == "MI350X":
        fp8_type = torch.float8_e4m3fn
    else:
        fp8_type = torch.float8_e4m3fnuz

    a = a.reshape(M, K // blockshape_k, blockshape_k)
    a = a.permute(2, 1, 0)
    max_vals = torch.max(a, 0, keepdim=True).values
    max_repr_val = torch.finfo(fp8_type).max
    max_vals[max_vals == 0] = 1e-8  # Avoid division by zero

    # Compute scale factors for each channel
    scale: torch.Tensor = max_repr_val / max_vals.to(torch.float32)

    # Quantize the tensor
    a = a * scale
    a.clamp_(-max_repr_val, max_repr_val)
    tensor_quantized = a.to(fp8_type)
    tensor_quantized = tensor_quantized.permute(2, 1, 0)
    tensor_quantized = tensor_quantized.reshape(M, K)

    scale = scale.reshape(M, K // blockshape_k)

    return tensor_quantized, scale, 1 / scale


def quantize_int8(
    tensor: torch.Tensor, dim=()
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    quantize_dim = [i for i in range(tensor.dim()) if i not in dim]
    max_vals = tensor.abs().amax(dim=quantize_dim, keepdim=True)
    max_repr_val = torch.iinfo(torch.int8).max
    max_vals[max_vals == 0] = 1e-8  # Avoid division by zero

    # Compute scale factors for each channel
    scale: torch.Tensor = max_repr_val / max_vals.to(torch.float32)

    # Quantize the tensor
    tensor = tensor * scale
    tensor.clamp_(-max_repr_val, max_repr_val)
    tensor = tensor.round_()
    tensor_quantized = tensor.to(torch.int8)

    scale = scale.squeeze(dim=quantize_dim)

    return tensor_quantized, scale, 1 / scale


def quantize_int4(
    tensor: torch.Tensor, group_size: int, has_zp: bool
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    # reshape tensor
    k, n = tensor.shape
    tensor = tensor.reshape(-1, group_size, n)
    tensor = tensor.permute(1, 0, 2)

    max_val = torch.max(tensor, 0, keepdim=True).values
    min_val = torch.min(tensor, 0, keepdim=True).values

    # Asymmetric quantization
    zp = None
    if has_zp:
        max_q_val = 15
        min_q_val = 0  # Min maps to 0
        scale = (max_val - min_val).clamp(min=1e-5) / (max_q_val)
        zp = torch.round(torch.abs(min_val / scale)).clamp(min_q_val, max_q_val).int()
    # Symmetric quantization
    else:
        max_q_val = 7
        min_q_val = -7
        scale = max_val / max_q_val

    # quantize and clamp
    tensor_q = torch.round(tensor / scale).int() + (zp if has_zp else 0)
    tensor_q = torch.clamp(tensor, min_q_val, max_q_val)

    # restore shapes
    tensor_q = tensor_q.reshape((group_size, -1, n))
    tensor_q = tensor_q.permute(1, 0, 2)
    tensor_q = tensor_q.reshape((k, n)).contiguous()

    # scale
    scale = scale.reshape((-1, n)).contiguous()

    # zp
    if zp is not None:
        zp = zp.reshape((-1, n)).contiguous()
        zp = zp.to(device=tensor.device)

    return tensor_q, scale, zp


def input_helper(
    M: int,
    N: int,
    K: int,
    top_k: int,
    E: int,
    routed_weight: bool,
    dtype,
    fp8_w8a8: bool,
    int8_w8a16: bool,
    blockshape=None,
):
    assert not (fp8_w8a8 and int8_w8a16)

    a = torch.randn((M, K), dtype=dtype, device="cuda")
    b = torch.rand((E, N, K), dtype=dtype, device="cuda")
    a_scale = None
    b_scale = None

    if fp8_w8a8:
        b, _, b_scale = quantize_fp8(b, dim=(0,), blockshape=blockshape)
        if blockshape is not None:
            blockshape_k = blockshape[1]
            a, _, a_scale = quantize_fp8_a(a, blockshape_k)

    if int8_w8a16:
        b, _, b_scale = quantize_int8(b, dim=(0,))

    b_zp = False

    c = torch.zeros((M, top_k, N), dtype=dtype, device="cuda")
    c_silu = torch.zeros((M * top_k, N // 2), dtype=dtype, device="cuda")

    values = torch.randn(M, E, dtype=dtype, device="cuda")

    softmax_vals = torch.softmax(values, dim=1)
    topk_weights, topk_ids = torch.topk(softmax_vals, k=top_k, dim=1)

    moe_config_func = get_optimal_moe_config_func(
        dtype, use_int8_w8a16=int8_w8a16, use_fp8_w8a8=fp8_w8a8
    )

    config = moe_config_func(M)

    sorted_token_ids, expert_ids, num_tokens_post_padded = (
        torch_moe_align_block_size_ref(topk_ids, config["BLOCK_SIZE_M"], E)
    )

    return (
        a,
        b,
        c,
        c_silu,
        b_zp,
        a_scale,
        b_scale,
        topk_weights,
        topk_ids,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        config,
    )


def input_helper_int4_w4a16(
    M: int,
    N: int,
    K: int,
    top_k: int,
    E: int,
    routed_weight: bool,
    dtype: torch.dtype,
    group_size: int,
    has_zp: bool,
):

    a = torch.randn((M, K), dtype=dtype, device="cuda")
    b = torch.rand((E, N, K), dtype=dtype, device="cuda")

    b_q = torch.empty((E, N, K // 2), dtype=torch.uint8, device="cuda")
    b_scale = torch.empty((E, N, K // group_size), dtype=dtype, device="cuda")
    if has_zp:
        b_zp = torch.empty(
            (E, N // 2, K // group_size), dtype=torch.uint8, device="cuda"
        )
    else:
        b_zp = None

    for e in range(E):
        q, scale, zp = quantize_int4(b[e].T, group_size=group_size, has_zp=has_zp)
        q = q.T
        q = (
            q[:, 1::2] * 16 + q[:, ::2]
        )  # Note, 2<<4=16. For bf16, etc, torch doesn't have shift.
        b_q[e] = q
        b_scale[e] = scale.T
        if has_zp:
            zp = zp.T.contiguous().to(torch.uint8)
            zp = (
                zp[1::2, :] << 4 | zp[::2, :]
            )  # Note, 2<<4=16. For bf16, etc, torch doesn't have shift.
            b_zp[e] = zp

    b = b_q

    c = torch.zeros((M, top_k, N), dtype=dtype, device="cuda")
    c_silu = torch.zeros((M * top_k, N // 2), dtype=dtype, device="cuda")

    values = torch.randn(M, E, dtype=dtype, device="cuda")

    softmax_vals = torch.softmax(values, dim=1)
    topk_weights, topk_ids = torch.topk(softmax_vals, k=top_k, dim=1)

    moe_config_func = get_optimal_moe_config_func(dtype, use_int4_w4a16=True)

    config = moe_config_func(M)
    sorted_token_ids, expert_ids, num_tokens_post_padded = (
        torch_moe_align_block_size_ref(topk_ids, config["BLOCK_SIZE_M"], E)
    )

    return (
        a,
        b,
        c,
        c_silu,
        b_zp,
        b_scale,
        topk_weights,
        topk_ids,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        config,
    )


def input_helper_e2e(
    M: int,
    N: int,
    K: int,
    top_k: int,
    E: int,
    dtype,
    fp8_w8a8: bool,
    blockshape=None,
    tp=1,
):

    a = torch.randn((M, K), dtype=dtype, device="cuda")
    w1 = torch.rand((E, N, K), dtype=dtype, device="cuda")
    w2 = torch.rand((E, K, N // 2), dtype=dtype, device="cuda")
    a_scale = None
    w1_scale = None
    w2_scale = None

    if fp8_w8a8:
        w1, _, w1_scale = quantize_fp8(w1, dim=(0,), blockshape=blockshape)
        w2, _, w2_scale = quantize_fp8(w2, dim=(0,), blockshape=blockshape)
        if blockshape is not None:
            blockshape_k = blockshape[1]
            a, _, a_scale = quantize_fp8_a(a, blockshape_k)

    c = torch.zeros((M, top_k, K), dtype=dtype, device="cuda")

    values = torch.randn(M, E, dtype=dtype, device="cuda")

    softmax_vals = torch.softmax(values, dim=1)
    topk_weights, topk_ids = torch.topk(softmax_vals, k=top_k, dim=1)

    moe_config_func = get_optimal_moe_e2e_config_func(
        N // tp, dtype, use_fp8_w8a8=fp8_w8a8
    )

    config = moe_config_func(M)

    sorted_token_ids, expert_ids, num_tokens_post_padded = (
        torch_moe_align_block_size_ref(topk_ids, config["BLOCK_SIZE_M"], E)
    )

    return (
        a,
        w1,
        w2,
        c,
        a_scale,
        w1_scale,
        w2_scale,
        topk_weights,
        topk_ids,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        config,
    )


# Note: TODO These 2 result in accuracy issues (64, 14336, 4096, 2, 8), (1, 1024, 16384, 1, 2)
@pytest.mark.parametrize(
    "M, N, K, top_k, E",
    [
        (32, 512, 7168, 8, 512),
        (64, 14336, 4096, 2, 8),
        (16, 14336, 1, 2, 4),
        (4, 4, 8, 1, 2),
        (1, 14336, 128, 2, 4),
        (3, 14336, 128, 2, 4),
        (16, 14336, 128, 1, 4),
        (16, 14336, 128, 1, 1),
        (64, 7186, 128, 2, 8),
        (64, 3584, 128, 2, 8),
        (64, 1792, 128, 2, 8),
        (64, 64, 128, 2, 8),
        (1, 1024, 16384, 1, 2),
    ],
)
@pytest.mark.parametrize("routed_weight", [False, True])
# @pytest.mark.parametrize('fp8_w8a8, int8_w8a16', [(False, False), (True, False), (False, True)]) #TODO: Accuracy issues with fp8
@pytest.mark.parametrize("fp8_w8a8, int8_w8a16", [(False, False)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("persistent", [False, True])
@pytest.mark.parametrize("silu_fused", [False, True])
def test_fused_moe(
    M: int,
    N: int,
    K: int,
    top_k: int,
    E: int,
    routed_weight: bool,
    fp8_w8a8: bool,
    int8_w8a16: bool,
    persistent: bool,
    silu_fused: bool,
    dtype,
):
    torch.manual_seed(20)
    torch.set_printoptions(threshold=100000)
    if persistent:
        (
            triton_moe_silu_set_use_persistent_kernel(True)
            if silu_fused
            else triton_moe_set_use_persistent_kernel(True)
        )
    else:
        (
            triton_moe_silu_set_use_persistent_kernel(False)
            if silu_fused
            else triton_moe_set_use_persistent_kernel(False)
        )

    (
        a,
        b,
        triton_out,
        triton_out_silu,
        b_zp,
        a_scale,
        b_scale,
        topk_weights,
        topk_ids,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        config,
    ) = input_helper(
        M,
        N,
        K,
        top_k,
        E,
        routed_weight=routed_weight,
        dtype=dtype,
        fp8_w8a8=fp8_w8a8,
        int8_w8a16=int8_w8a16,
    )

    if DEBUG_MODE:
        print(f"M={M}, N={N}, K={K}, top_K={top_k}, E={E}")
        print(f"config={config}")
        print(f"a.shape={a.shape} a={a}")
        print(f"b.shape={b.shape} b={b}")
        print(f"sorted_token_ids.shape={sorted_token_ids.shape}")
        print(f"sorted_token_ids={sorted_token_ids}")
        print(f"expert_ids.shape={expert_ids.shape}")
        print(f"expert_ids={expert_ids}")
        print(f"num_tokens_post_padded={num_tokens_post_padded}")
    _triton_moe = triton_moe_silu if silu_fused else triton_moe

    _triton_moe(
        a,
        b,
        triton_out_silu if silu_fused else triton_out,
        a_scale,
        b_scale,
        b_zp,
        topk_weights,
        topk_ids,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        routed_weight,
        top_k,
        torch_to_triton_dtype[dtype],
        fp8_w8a8,
        int8_w8a16,
        False,
        config=config,
    )

    torch_out = torch.empty_like(triton_out)
    torch_out = torch_moe_ref(
        a,
        b,
        torch_out,
        a_scale,
        b_scale,
        None,
        0,
        topk_ids,
        topk_weights,
        routed_weight,
        dtype,
        fp8_w8a8,
        int8_w8a16,
        False,
    )
    if silu_fused:
        torch_out_silu = torch_silu_and_mul_ref(torch_out.view(-1, N))

    if DEBUG_MODE:
        print(f"triton_out={triton_out}")
        print(f"torch_out={torch_out}")
    # Validate correctness
    if silu_fused:
        torch.testing.assert_close(
            triton_out_silu, torch_out_silu, atol=1e-1, rtol=1e-1
        )
    else:
        torch.testing.assert_close(triton_out, torch_out, atol=1e-1, rtol=1e-1)


@pytest.mark.parametrize(
    "M, N, K, top_k, E",
    [(1, 64, 128, 1, 2), (1, 64, 128, 2, 4), (4, 32, 64, 4, 16), (8, 96, 256, 2, 16)],
)
@pytest.mark.parametrize("routed_weight", [False, True])
@pytest.mark.parametrize("group_size", [8, 16, 32, 64])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("has_zp", [False, True])
@pytest.mark.parametrize("persistent", [False, True])
@pytest.mark.parametrize("silu_fused", [False, True])
def test_fused_moe_int4_w4a16(
    M: int,
    N: int,
    K: int,
    top_k: int,
    E: int,
    routed_weight: bool,
    dtype: torch.dtype,
    group_size: int,
    has_zp: bool,
    persistent: bool,
    silu_fused: bool,
):

    if (
        M == 1
        and N == 64
        and K == 128
        and top_k == 1
        and E == 2
        and group_size == 8
        and routed_weight
        and not persistent
        and has_zp
        and not silu_fused
    ):
        pytest.skip("Results in accuracy failure because of Triton compiler change")

    torch.manual_seed(20)
    (
        a,
        b,
        triton_out,
        triton_out_silu,
        b_zp,
        b_scale,
        topk_weights,
        topk_ids,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        config,
    ) = input_helper_int4_w4a16(
        M,
        N,
        K,
        top_k,
        E,
        routed_weight=routed_weight,
        dtype=dtype,
        group_size=group_size,
        has_zp=has_zp,
    )

    if persistent:
        (
            triton_moe_silu_set_use_persistent_kernel(True)
            if silu_fused
            else triton_moe_set_use_persistent_kernel(True)
        )
    else:
        (
            triton_moe_silu_set_use_persistent_kernel(False)
            if silu_fused
            else triton_moe_set_use_persistent_kernel(False)
        )

    _triton_moe = triton_moe_silu if silu_fused else triton_moe
    _triton_moe(
        a,
        b,
        triton_out_silu if silu_fused else triton_out,
        None,
        b_scale,
        b_zp,
        topk_weights,
        topk_ids,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        routed_weight,
        top_k,
        torch_to_triton_dtype[dtype],
        use_fp8_w8a8=False,
        use_int8_w8a16=False,
        use_int4_w4a16=True,
        block_shape=(0, group_size),
        config=config,
    )

    torch_out = torch.empty_like(triton_out)
    torch_out = torch_moe_ref(
        a,
        b,
        torch_out,
        None,
        b_scale,
        b_zp,
        group_size,
        topk_ids,
        topk_weights,
        routed_weight,
        dtype,
        False,
        False,
        True,
    )
    if silu_fused:
        torch_out_silu = torch_silu_and_mul_ref(torch_out.view(-1, N))

    if silu_fused:
        torch.testing.assert_close(
            triton_out_silu, torch_out_silu, atol=2e-1, rtol=2e-1
        )
    else:
        torch.testing.assert_close(triton_out, torch_out, atol=2e-1, rtol=2e-1)


# Note: TODO These 2 result in accuracy issues (64, 14336, 4096, 2, 8), (1, 1024, 16384, 1, 2)
@pytest.mark.parametrize(
    "M, N, K, top_k, E",
    [
        (64, 14336, 4096, 2, 8),
        (16, 14336, 1, 2, 4),
        (4, 4, 8, 1, 2),
        (1, 14336, 128, 2, 4),
        (3, 14336, 128, 2, 4),
        (16, 14336, 128, 1, 4),
        (16, 14336, 128, 1, 1),
        (64, 7186, 128, 2, 8),
        (64, 3584, 128, 2, 8),
        (64, 1792, 128, 2, 8),
        (64, 64, 128, 2, 8),
        (1, 1024, 16384, 1, 2),
    ],
)
@pytest.mark.parametrize("routed_weight", [False, True])
# @pytest.mark.parametrize('fp8_w8a8, int8_w8a16', [(False, False), (True, False), (False, True)]) #TODO: Accuracy issues with fp8
@pytest.mark.parametrize("fp8_w8a8, int8_w8a16", [(False, False)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("persistent", [False, True])
def test_fused_moe_gelu(
    M: int,
    N: int,
    K: int,
    top_k: int,
    E: int,
    routed_weight: bool,
    fp8_w8a8: bool,
    int8_w8a16: bool,
    persistent: bool,
    dtype,
):
    torch.manual_seed(20)
    torch.set_printoptions(threshold=100000)
    if persistent:
        triton_moe_gelu_set_use_persistent_kernel(True)
    else:
        triton_moe_gelu_set_use_persistent_kernel(False)

    (
        a,
        b,
        triton_out,
        triton_out_silu,
        b_zp,
        a_scale,
        b_scale,
        topk_weights,
        topk_ids,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        config,
    ) = input_helper(
        M,
        N,
        K,
        top_k,
        E,
        routed_weight=routed_weight,
        dtype=dtype,
        fp8_w8a8=fp8_w8a8,
        int8_w8a16=int8_w8a16,
    )

    if DEBUG_MODE:
        print(f"M={M}, N={N}, K={K}, top_K={top_k}, E={E}")
        print(f"config={config}")
        print(f"a.shape={a.shape} a={a}")
        print(f"b.shape={b.shape} b={b}")
        print(f"sorted_token_ids.shape={sorted_token_ids.shape}")
        print(f"sorted_token_ids={sorted_token_ids}")
        print(f"expert_ids.shape={expert_ids.shape}")
        print(f"expert_ids={expert_ids}")
        print(f"num_tokens_post_padded={num_tokens_post_padded}")
    triton_moe_gelu(
        a,
        b,
        triton_out,
        a_scale,
        b_scale,
        topk_weights,
        topk_ids,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        routed_weight,
        top_k,
        torch_to_triton_dtype[dtype],
        fp8_w8a8,
        int8_w8a16,
        config=config,
    )

    torch_out = torch.empty_like(triton_out)
    torch_out = torch_moe_ref(
        a,
        b,
        torch_out,
        a_scale,
        b_scale,
        None,
        0,
        topk_ids,
        topk_weights,
        routed_weight,
        dtype,
        fp8_w8a8,
        int8_w8a16,
        False,
        gelu=True,
    )

    if DEBUG_MODE:
        print(f"triton_out={triton_out}")
        print(f"torch_out={torch_out}")
    # Validate correctness
    torch.testing.assert_close(triton_out, torch_out, atol=1e-1, rtol=1e-1)


@pytest.mark.parametrize(
    "M, N, K, top_k, E",
    [
        (3, 512, 2048, 10, 512),  # qwen3next
        (333, 512, 2048, 10, 512),
        (1033, 512, 2048, 10, 512),
        (3, 768, 2048, 8, 128),  # qwen3
        (333, 768, 2048, 8, 128),
        (1033, 768, 2048, 8, 128),
        (1033, 1024, 2048, 8, 128),
        (3, 2048, 4096, 2, 8),  # mixtral-7B
        (33, 768, 2048, 8, 128),  # qwen3
        (333, 512, 2048, 10, 512),  # qwen3next
        (33, 8192, 5120, 1, 128),  # llama4-maverick
        (33, 1536, 4096, 8, 128),  # qwen3
        # TODO: add other shapes
    ],
)
@pytest.mark.parametrize("routed_weight", [False])
@pytest.mark.parametrize("fp8_w8a8", [True, False])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("blockshape_n, blockshape_k", [(128, 128)])
@pytest.mark.parametrize("tp", [1, 8])
def test_moe_e2e(
    M: int,
    N: int,
    K: int,
    top_k: int,
    E: int,
    routed_weight: bool,
    fp8_w8a8: bool,
    blockshape_n: int,
    blockshape_k: int,
    dtype,
    tp,
):
    torch.manual_seed(20)
    torch.set_printoptions(threshold=100000)
    blockshape = (blockshape_n, blockshape_k)
    (
        a,
        w1,
        w2,
        triton_out,
        a_scale,
        w1_scale,
        w2_scale,
        topk_weights,
        topk_ids,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        config,
    ) = input_helper_e2e(
        M,
        N,
        K,
        top_k,
        E,
        dtype=dtype,
        fp8_w8a8=fp8_w8a8,
        blockshape=blockshape,
        tp=tp,
    )

    # tensor parallel slicing
    if tp > 1:
        w1 = w1[:, : N // tp, :].contiguous()
        w2 = w2[:, :, : (N // 2 // tp)].contiguous()
        if fp8_w8a8:
            num_w1_scales_per_gpu = triton.cdiv(N // tp, blockshape[0])
            w1_scale = w1_scale[:, :num_w1_scales_per_gpu].contiguous()
            num_w2_scales_per_gpu = triton.cdiv((N // 2) // tp, blockshape[0])
            w2_scale = w2_scale[:, :, :num_w2_scales_per_gpu].contiguous()
        N = N // tp  # for later reshape

    if DEBUG_MODE:
        print(f"M={M}, N={N}, K={K}, top_K={top_k}, E={E}")
        print(f"config={config}")
        print(f"a.shape={a.shape} a={a}")
        print(f"w1.shape={w1.shape} w1={w1}")
        print(f"w2.shape={w2.shape} w2={w2}")
        print(f"sorted_token_ids.shape={sorted_token_ids.shape}")
        print(f"sorted_token_ids={sorted_token_ids}")
        print(f"expert_ids.shape={expert_ids.shape}")
        print(f"expert_ids={expert_ids}")
        print(f"num_tokens_post_padded={num_tokens_post_padded}")

    # onekernel solution
    triton_out, triton_intermediate = triton_e2e_moe(
        a,
        w1,
        w2,
        triton_out,
        a_scale,
        w1_scale,
        w2_scale,
        topk_weights,
        sorted_token_ids,
        topk_ids,
        expert_ids,
        num_tokens_post_padded,
        routed_weight,
        top_k,
        fp8_w8a8,
        blockshape,
        config,
        return_intermediate=True,
    )

    # validate correctness by comparing to the outputs of two torch gemms
    torch_intermediate = torch.zeros_like(triton_intermediate)
    torch_intermediate = torch_moe_ref(
        a,
        w1,
        torch_intermediate,
        a_scale,
        w1_scale,
        None,
        None,
        topk_ids,
        topk_weights,
        routed_weight,
        dtype,
        fp8_w8a8,
        False,
        False,
        gelu=False,
        blockshape=blockshape,
    )
    torch_intermediate = torch_intermediate.to(torch.float32)
    torch_intermediate = torch_silu_and_mul_ref(torch_intermediate.view(-1, N))
    torch_intermediate = torch_intermediate.view(M, top_k, N // 2)

    torch_out = torch.empty_like(triton_out)
    torch_out = torch_moe_gemm2(
        triton_intermediate,  # (acceptable) precision errors from the first gemm accumulate here
        # torch_intermediate,
        w2,
        torch_out,
        w2_scale,
        topk_ids,
        topk_weights,
        routed_weight,
        dtype,
        fp8_w8a8,
        blockshape=blockshape,
    )

    torch.testing.assert_close(
        triton_intermediate, torch_intermediate, atol=1e-1, rtol=1e-1
    )
    # Print both outputs in scientific notation for consistency
    torch.testing.assert_close(triton_out, torch_out, atol=1e-1, rtol=1e-1)


if __name__ == "__main__":
    test_moe_e2e(32, 64, 64, 1, 1, False, True, 128, 128, torch.bfloat16, 1)