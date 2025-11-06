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
from aiter.ops.triton.quant import dynamic_per_tensor_quant_fp8_i8

from aiter.ops.triton.utils.types import torch_to_triton_dtype
from op_tests.triton_tests.test_moe import (
    torch_moe_ref,
    torch_moe_align_block_size_ref,
    torch_silu_and_mul_ref,
    quantize_fp8,
    quantize_fp8_a,
)


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

    E, K, N = b.shape

    # num_scales_along_n = 1
    if fp8_w8a8 and use_block_scale:
        num_scales_along_n = b_scale.shape[2]
        b_scale = b_scale.repeat_interleave(blockshape_k, dim=1)[:, :K]
        # only need to dequantize before dot if the inner dimension spans multiple scales
        if num_scales_along_n > 1:
            b_scale = b_scale.repeat_interleave(blockshape_n, dim=2)[:, :, :N]  # (E, K, N_HALF)
            b = b.to(torch.float32) * b_scale.to(torch.float32)

    b_indexed = b[topk_ids]
    
    out = torch.einsum(
        "men,mekn->mek", a.to(torch.bfloat16), b_indexed.to(torch.bfloat16)
    )

    if fp8_w8a8 and (not use_block_scale or num_scales_along_n == 1):
        if not use_block_scale:
            out = out * b_scale[topk_ids].unsqueeze(-1)
        else:
            out = out * b_scale[topk_ids].squeeze(-1)

    if routed_weight:
        out = out * topk_weights.unsqueeze(-1)

    return out.to(out_dtype)


def input_helper_e2e(
    M: int,
    N: int,
    K: int,
    top_k: int,
    E: int,
    dtype,
    fp8_w8a8: bool,
    blockshape=None,
    pertoken=False,
    tp=1,
):
    
    if pertoken:
        assert fp8_w8a8, "per-token quantization only supported for fp8_w8a8"
        assert blockshape is None or blockshape[-1] == K, "per-token quantization requires weight quantization blockshape for K to be None or K"
    
    
    a = torch.randn((M, K), dtype=dtype, device="cuda")
    w1 = torch.rand((E, N*2, K), dtype=dtype, device="cuda")
    w2 = torch.rand((E, K, N), dtype=dtype, device="cuda")
    a_scale = None
    w1_scale = None
    w2_scale = None

    if fp8_w8a8:
        # scales for tensor E, N, K: E, N//blockshape_n, K//blockshape_k
        w1, _, w1_scale = quantize_fp8(w1, dim=(0,), blockshape=blockshape)
        # scales for tensor E, K, N//2: E, K//blockshape_k, N//2//blockshape_n 
        blockshape_stage2 = None
        if blockshape is not None:
            blockshape_stage2 = (blockshape[1], blockshape[0])
        w2, _, w2_scale = quantize_fp8(w2, dim=(0,), blockshape=blockshape_stage2)
        
        # a_scale takes shape
        if pertoken: # (M, 1)
            a, _, a_scale = quantize_fp8_a(a, a.shape[-1])
        elif blockshape is not None: # (M, K // blockshape_k)
            blockshape_k = blockshape[-1]
            a, _, a_scale = quantize_fp8_a(a, blockshape_k)
        else: # (1,)
            a_scale = torch.zeros(1, device=a.device, dtype=torch.float32)
            output = torch.zeros(a.shape, device=a.device, dtype=w1.dtype)
            a, a_scale = dynamic_per_tensor_quant_fp8_i8(output, a, a_scale)

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
    ],
)
@pytest.mark.parametrize("routed_weight", [False])
@pytest.mark.parametrize("fp8_w8a8", [False])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("blockshape_n, blockshape_k", [(None, None)])
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
    tp
):
    pertoken = False
    torch.manual_seed(20)
    torch.set_printoptions(threshold=100000)
    blockshape = None
    pertoken = pertoken and fp8_w8a8
    if blockshape_n is not None and blockshape_k is not None:
        blockshape = (blockshape_n, blockshape_k)
    # adjust blockshape for per-token quantization
    if pertoken and blockshape is not None:
        blockshape = (blockshape_n, K)
    
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
        pertoken=pertoken,
    )
    # tensor parallel slicing
    if tp > 1:
        w1 = w1[:, : N*2 // tp, :].contiguous()
        w2 = w2[:, :, : (N // tp)].contiguous()
        if fp8_w8a8 and blockshape is not None:
            num_w1_scales_per_gpu = triton.cdiv(N*2 // tp, blockshape[0])
            w1_scale = w1_scale[:, :num_w1_scales_per_gpu].contiguous()
            num_w2_scales_per_gpu = triton.cdiv(N // tp, blockshape[0])
            w2_scale = w2_scale[:, :, :num_w2_scales_per_gpu].contiguous()
        N = N // tp  # for later reshape

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
        pertoken_quant_a=pertoken,
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
    torch_intermediate = torch_silu_and_mul_ref(torch_intermediate.view(-1, N*2))
    torch_intermediate = torch_intermediate.view(M, top_k, N)

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

    torch.testing.assert_close(triton_intermediate, torch_intermediate, atol=1e-1, rtol=1e-1)
    # Print both outputs in scientific notation for consistency
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
    ],
)
@pytest.mark.parametrize("routed_weight", [False])
@pytest.mark.parametrize("fp8_w8a8", [True])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("blockshape_n, blockshape_k", [(None, None)])
@pytest.mark.parametrize("tp", [1, 8])
@pytest.mark.parametrize("pertoken", [False])
def test_moe_e2e_fp8(
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
    pertoken
):
    torch.manual_seed(20)
    # torch.set_printoptions(threshold=100000)
    blockshape = None
    pertoken = pertoken and fp8_w8a8
    if blockshape_n is not None and blockshape_k is not None:
        blockshape = (blockshape_n, blockshape_k)
    # adjust blockshape for per-token quantization
    if pertoken and blockshape is not None:
        blockshape = (blockshape_n, K)
    
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
        pertoken=pertoken,
    )
    # tensor parallel slicing
    if tp > 1:
        w1 = w1[:, : N * 2 // tp, :].contiguous()
        w2 = w2[:, :, : N // tp].contiguous()
        if fp8_w8a8 and blockshape is not None:
            num_w1_scales_per_gpu = triton.cdiv(N * 2 // tp, blockshape[0])
            w1_scale = w1_scale[:, :num_w1_scales_per_gpu].contiguous()
            num_w2_scales_per_gpu = triton.cdiv(N // tp, blockshape[0])
            w2_scale = w2_scale[:, :, :num_w2_scales_per_gpu].contiguous()
        N = N // tp  # for later reshape

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
        pertoken_quant_a=pertoken,
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
    torch_intermediate = torch_silu_and_mul_ref(torch_intermediate.view(-1, N*2))
    torch_intermediate = torch_intermediate.view(M, top_k, N)
    
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
    torch.testing.assert_close(triton_intermediate, torch_intermediate, atol=1e-1, rtol=1e-1)
    # Print both outputs in scientific notation for consistency
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
    ],
)
@pytest.mark.parametrize("routed_weight", [False])
@pytest.mark.parametrize("fp8_w8a8", [True])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("blockshape_n, blockshape_k", [(128, 128)])
@pytest.mark.parametrize("tp", [1,8])
@pytest.mark.parametrize("pertoken", [False])
def test_moe_e2e_fp8_blockscale(
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
    pertoken
):
    torch.manual_seed(20)
    # torch.set_printoptions(threshold=100000)
    blockshape = None
    pertoken = pertoken and fp8_w8a8
    if blockshape_n is not None and blockshape_k is not None:
        blockshape = (blockshape_n, blockshape_k)
    # adjust blockshape for per-token quantization
    if pertoken and blockshape is not None:
        blockshape = (blockshape_n, K)
    
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
        pertoken=pertoken,
    )
    # tensor parallel slicing
    if tp > 1:
        w1 = w1[:, : N * 2 // tp, :].contiguous()
        w2 = w2[:, :, : N // tp].contiguous()
        if fp8_w8a8 and blockshape is not None:
            num_w1_scales_per_gpu = triton.cdiv(N * 2 // tp, blockshape[0])
            w1_scale = w1_scale[:, :num_w1_scales_per_gpu].contiguous()
            num_w2_scales_per_gpu = triton.cdiv(N // tp, blockshape[0])
            w2_scale = w2_scale[:, :, :num_w2_scales_per_gpu].contiguous()
        N = N // tp  # for later reshape

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
        pertoken_quant_a=pertoken,
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
    torch_intermediate = torch_silu_and_mul_ref(torch_intermediate.view(-1, N*2))
    torch_intermediate = torch_intermediate.view(M, top_k, N)
    
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
    torch.testing.assert_close(triton_intermediate, torch_intermediate, atol=1e-1, rtol=1e-1)
    # Print both outputs in scientific notation for consistency
    torch.testing.assert_close(triton_out, torch_out, atol=1e-1, rtol=1e-1)


if __name__ == "__main__":
    test_moe_e2e(
        3, 2048, 4096, 2, 8,
        routed_weight=False,
        fp8_w8a8=False,
        blockshape_n=None,
        blockshape_k=None,
        dtype=torch.bfloat16,
        tp=8,
    )