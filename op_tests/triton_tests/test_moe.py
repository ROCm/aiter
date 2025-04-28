import torch
import triton.language as tl
import pytest
from typing import Dict
from aiter.ops.triton.moe_op import fused_moe as triton_moe
from aiter.ops.triton.moe_op import moe_set_use_persistent_kernel
from aiter.ops.triton.moe_op_gelu import fused_moe_gelu as triton_moe_gelu
from aiter.ops.triton.moe_op_silu_fused import fused_moe_silu as triton_moe_silu
from op_tests.op_benchmarks.triton.utils.common import (
    str_to_torch_dtype,
    torch_to_tl_dtype,
)
from op_tests.op_benchmarks.triton.utils.moe import (
    generate_moe_alignment,
    generate_moe_input,
    get_optimal_moe_config,
)

from .utils.fused_moe_ref import torch_moe


DEBUG_MODE = False


def silu_and_mul(input):
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


# TODO These 2 result in accuracy issues (64, 14336, 4096, 2, 8), (1, 1024, 16384, 1, 2)
@pytest.mark.parametrize(
    "M, N, K, top_k, E",
    [  # (64, 14336, 4096, 2, 8), # TODO: this causes OOM in torch_moe b_index = b[topk_ids]
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
@pytest.mark.parametrize("input_dtype, aux_dtype", [(None, "fp16")])
@pytest.mark.parametrize("persistent", [False, True])
@pytest.mark.parametrize("silu_fused", [False, True])
def test_fused_moe(
    M: int,
    N: int,
    K: int,
    top_k: int,
    E: int,
    routed_weight: bool,
    persistent: bool,
    silu_fused: bool,
    input_dtype,
    aux_dtype,
):
    moe_set_use_persistent_kernel(persistent)
    group_size = has_zp = None
    fp8_w8a8 = input_dtype == "fp8_w8a8"
    int8_w8a16 = input_dtype == "int8_w8a16"
    a, b, triton_out, triton_out_silu, b_zp, a_scale, b_scale = generate_moe_input(
        M, N, K, top_k, E, group_size, has_zp, input_dtype, aux_dtype
    )
    config = get_optimal_moe_config(input_dtype, aux_dtype, M)
    topk_weights, topk_ids, sorted_token_ids, expert_ids, num_tokens_post_padded = (
        generate_moe_alignment(M, E, top_k, config["BLOCK_SIZE_M"])
    )
    aux_dtype = str_to_torch_dtype[aux_dtype]

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
        config,
        torch_to_tl_dtype[aux_dtype],
        fp8_w8a8,
        int8_w8a16,
        False,
    )

    torch_out = torch.empty_like(triton_out)
    torch_out = torch_moe(
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
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        aux_dtype,
        fp8_w8a8,
        int8_w8a16,
        False,
    )
    if silu_fused:
        torch_out_silu = silu_and_mul(torch_out.view(-1, N))

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
@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize("has_zp", [False, True])
@pytest.mark.parametrize("persistent", [False])
# @pytest.mark.parametrize('persistent',[False, True]) #Persistent results in accuracy issues
# @pytest.mark.parametrize('silu_fused', [False, True])
@pytest.mark.parametrize("silu_fused", [False])  # Silu results in accuracy issues
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
    moe_set_use_persistent_kernel(persistent)
    input_dtype = "int4_w4a16"
    a, b, triton_out, triton_out_silu, b_zp, a_scale, b_scale = generate_moe_input(
        M, N, K, top_k, E, group_size, has_zp, input_dtype=input_dtype, aux_dtype=dtype
    )
    config = get_optimal_moe_config(input_dtype, dtype, M)
    topk_weights, topk_ids, sorted_token_ids, expert_ids, num_tokens_post_padded = (
        generate_moe_alignment(M, E, top_k, config["BLOCK_SIZE_M"])
    )
    dtype = str_to_torch_dtype[dtype]

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
        config,
        torch_to_tl_dtype[dtype],
        use_fp8_w8a8=False,
        use_int8_w8a16=False,
        use_int4_w4a16=True,
        block_shape=(0, group_size),
    )

    torch_out = torch.empty_like(triton_out)
    torch_out = torch_moe(
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
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        dtype,
        False,
        False,
        True,
    )
    if silu_fused:
        torch_out_silu = silu_and_mul(torch_out.view(-1, N))

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
# @pytest.mark.parametrize('fp8_w8a8, int8_w8a16', [(False, False)])
@pytest.mark.parametrize("input_dtype, aux_dtype", [(None, "fp16"), (None, "bf16")])
@pytest.mark.parametrize("persistent", [False, True])
def test_moe_fused_gelu(
    M: int,
    N: int,
    K: int,
    top_k: int,
    E: int,
    routed_weight: bool,
    persistent: bool,
    input_dtype,
    aux_dtype,
):
    moe_set_use_persistent_kernel(persistent)

    group_size = has_zp = None
    fp8_w8a8 = input_dtype == "fp8_w8a8"
    int8_w8a16 = input_dtype == "int8_w8a16"
    a, b, triton_out, triton_out_silu, b_zp, a_scale, b_scale = generate_moe_input(
        M, N, K, top_k, E, group_size, has_zp, input_dtype, aux_dtype
    )
    config = get_optimal_moe_config(input_dtype, aux_dtype, M)
    topk_weights, topk_ids, sorted_token_ids, expert_ids, num_tokens_post_padded = (
        generate_moe_alignment(M, E, top_k, config["BLOCK_SIZE_M"])
    )
    aux_dtype = str_to_torch_dtype[aux_dtype]

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
        config,
        torch_to_tl_dtype[aux_dtype],
        fp8_w8a8,
        int8_w8a16,
    )

    torch_out = torch.empty_like(triton_out)
    torch_out = torch_moe(
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
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        aux_dtype,
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
