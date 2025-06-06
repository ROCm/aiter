import pytest
import torch
import triton
import triton.language as tl
import aiter

from aiter import ActivationType, get_torch_quant

from aiter.fused_moe import fused_topk, moe_sorting

# from aiter.quant import get_triton_quant
from aiter.utility.fp4_utils import moe_mxfp4_sort

from aiter.test_common import (
    checkAllclose,
    benchmark,
    run_perftest,
)

from aiter.fused_moe_bf16_asm import ck_moe_2stages
from aiter import dtypes

from aiter.ops.triton.quant import dynamic_mxfp4_quant
from aiter import fp4_utils

# from aiter.utility.fp4_utils import moe_mxfp4_sort

DEBUG_MODE = False


def ck_moe_stage1(
    hidden_states,
    w1,  # [E, inter_dim*2, model_dim]
    w2,  # [E, model_dim, inter_dim]
    sorted_token_ids,  # [max_num_tokens_padded]
    sorted_expert_ids,  # [max_num_m_blocks]
    num_valid_ids,  # [1]
    w1_scale,
    a1_scale,
    dtype,
    topk,
    block_size=64,
    Activation=ActivationType.Gelu,
    sorted_weights=None,  # [max_num_tokens_padded]
):
    token_num = hidden_states.shape[0]
    D = w1.shape[1] // 2
    # max_num_tokens_padded = sorted_expert_ids.shape[0]*block_size
    if Activation == ActivationType.Silu:
        act_op = 1
    else:
        act_op = 0

    if w1.dtype is torch.uint32:
        D = D * 8

    out = torch.empty((token_num, topk, D), dtype=dtype, device=hidden_states.device)

    aiter.ck_moe_stage1(
        hidden_states,
        w1,
        w2,
        sorted_token_ids,
        sorted_expert_ids,
        num_valid_ids,
        out,
        topk,
        w1_scale,
        a1_scale,
        block_size,
        sorted_weights,
        act_op,
        3,
    )

    return out


def ck_moe_stage2(
    hidden_states,
    w1,  # [E, inter_dim*2, model_dim]
    w2,  # [E, model_dim, inter_dim]
    sorted_token_ids,  # [max_num_tokens_padded]
    sorted_expert_ids,  # [max_num_m_blocks]
    num_valid_ids,  # [1]
    w2_scale,
    a2_scale,
    dtype,
    topk,
    block_size=64,
    sorted_weights=None,  # [max_num_tokens_padded]
):
    token_num = hidden_states.shape[0]
    D = w2.shape[1]
    # max_num_tokens_padded = sorted_expert_ids.shape[0]*block_size

    out = torch.zeros(
        (token_num, D),
        dtype=dtype,
        device=hidden_states.device,
    )
    aiter.ck_moe_stage2(
        hidden_states,
        w1,
        w2,
        sorted_token_ids,
        sorted_expert_ids,
        num_valid_ids,
        out,
        topk,
        w2_scale,
        a2_scale,
        block_size,
        sorted_weights,
        3,
    )
    return out


def torch_moe_stage1(
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
    sorted_token_ids,
    expert_ids,
    num_tokens_post_padded,
    activation,
    dtype,
):
    M, top_k, N = c.shape
    _, K = a.shape

    # Repeat a -> (M, top_k, K)
    a_expanded = a.unsqueeze(1).repeat(1, top_k, 1)
    # (M, top_k, N, K)
    # b_indexed = b[topk_ids]

    c = torch.zeros(
        (M, top_k, N * 2),
        dtype=torch.float,
        device=a.device,
    )
    for E_id in range(b.shape[0]):
        mask = topk_ids == E_id
        if mask.sum():
            sub_tokens = a_expanded[mask]
            act_input = sub_tokens @ (b[E_id].transpose(0, 1))
            if routed_weight:
                act_input = act_input * topk_weights[mask].view(-1, 1)
            c[mask] = act_input

    use_g1u1 = b.shape[1] == (2 * N)
    torch_act = aiter.get_torch_act(activation)
    c = c.to(torch.float)
    if use_g1u1:
        gate, up = c.split([N, N], dim=-1)
        c = torch_act(gate) * up
    else:
        c = torch_act(c)

    return c.to(dtype)


def torch_moe_stage2(
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
    sorted_token_ids,
    expert_ids,
    num_tokens_post_padded,
    dtype,
):
    M, N = c.shape
    _, topk, K = a.shape
    # Repeat a -> (M, top_k, K)
    a_expanded = a
    # (M, top_k, N, K)
    # b_indexed = b[topk_ids]

    c = torch.zeros(
        (M, topk, N),
        dtype=torch.float,
        device=a.device,
    )
    for E_id in range(b.shape[0]):
        mask = topk_ids == E_id
        if mask.sum():
            sub_tokens = a[mask]
            act_input = sub_tokens @ (b[E_id].transpose(0, 1))
            c[mask] = act_input
    if routed_weight:
        c = c * topk_weights.view(M, -1, 1)

    return c.sum(1).to(dtype)


def get_cdna_version():
    """
    Gets the AMD architecture version, i.e. CDNA3 or CDNA4, currently
    only supports 3 (gfx942) or 4 (gfx950). Returns -1 if it is not AMD
    hardware or unsupported architecture
    """
    target = triton.runtime.driver.active.get_current_target()
    if target.backend != "hip":
        return -1
    if target.arch == "gfx942":
        return 3
    if target.arch == "gfx950":
        return 4
    return -1


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


def moe_align_block_size(
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


def torch_dynamic_mxfp4_quant(
    x: torch.Tensor, scaling_mode: str = "even"
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize a tensor to MX FP4 format based of AMD Quark Spec.

    Math equivalent:
        blockscale_e8m0 = 2^(floor(log2(rounding(max_abs(x_block)))-max_exp))
        x_block_fp4 = x_block / blockscale_e8m0
        where max_exp = 2 for fp4_e2m1.

    Args:
        x: The input tensor, typically fp16 or bf16.
        scaling_mode: The method to calculate MX block scaling.
            - "even" (default): `even_round`.
    Returns:
        A tuple of (x_fp4, blockscale_e8m0).
    """
    # Create padded x. Needed because mxfp4 works with block of 64 elements
    MXFP4_QUANT_BLOCK_SIZE = 32
    x_shape = x.shape
    if x.shape[-1] % MXFP4_QUANT_BLOCK_SIZE != 0:
        shape = list(x_shape)
        shape = shape[:-1] + [
            ((shape[-1] - 1 + MXFP4_QUANT_BLOCK_SIZE) // MXFP4_QUANT_BLOCK_SIZE)
            * MXFP4_QUANT_BLOCK_SIZE
        ]
        shape = tuple(shape)
        x_padded = torch.zeros((shape), device=x.device, dtype=x.dtype)
        x_padded[..., : x.shape[-1]] = x
    else:
        x_padded = x

    # Calculate scale
    x_padded = x_padded.reshape(
        -1, x_padded.shape[-1] // MXFP4_QUANT_BLOCK_SIZE, MXFP4_QUANT_BLOCK_SIZE
    ).to(torch.float32)
    # print(f"x_padded.shape={x_padded.shape}")
    amax, _ = torch.max(torch.abs(x_padded), dim=-1)
    amax = amax.view(torch.int32)
    amax = (amax + 0x200000) & 0xFF800000
    amax = amax.view(torch.float32)
    scale_e8m0_unbiased = torch.log2(amax).floor() - 2
    scale_e8m0_unbiased = torch.clamp(scale_e8m0_unbiased, min=-127, max=127)
    quant_scale = torch.exp2(-scale_e8m0_unbiased)

    # Compute quantized x
    qx = x_padded * quant_scale.unsqueeze(-1)

    # blockscale_e8m0
    bs_e8m0 = scale_e8m0_unbiased.to(torch.uint8) + 127

    # Convert to mxfp4 format
    #
    # Note: This code is adapted from Triton Bench numerics mxfp4 code
    #
    # Note: MXFP4  S:1-bit, E:2-bit, M:1-bit
    #   Zeros: S000 -> +/-0
    #   Denormal Numbers: S001 -> +/- 0.5
    #   Normal Numbers:
    #           S010 -> +/- 1.0
    #           S011 -> +/- 1.5
    #           S100 -> +/- 2.0
    #           S101 -> +/- 3.0
    #           S110 -> +/- 4.0
    #           S111 -> +/- 6.0
    # Convert quantized fp32 tensor to int32 before converting to mxfp4 format
    qx = qx.view(torch.int32)

    # Extract sign, exponents and mantissa fields from int32
    s = qx & 0x80000000
    e = (qx >> 23) & 0xFF
    m = qx & 0x7FFFFF

    E8_BIAS = 127
    E2_BIAS = 1

    # Denormal numbers
    # If exponent is less than 127, then it's a denormal number
    # See above, for denormal number mantissa is always 1 and we set bit 1 of mantissa
    adjusted_exponents = E8_BIAS - e - 1
    m = torch.where(e < E8_BIAS, (0x400000 | (m >> 1)) >> adjusted_exponents, m)

    # For normal numbers, bias is changed from 127 to 1, and for subnormals, we keep exponent as 0.
    # Note: E8_BIAS - E2_BIAS = 126, so for normals we subtract that.
    e = torch.where(e > E8_BIAS - E2_BIAS, e, E8_BIAS - E2_BIAS) - (E8_BIAS - E2_BIAS)

    # Combine sign, exponent, and mantissa, while saturating
    # rounding nearest with tie breaking up by adding +1 to one bit right of the LSB, then shift right
    combined_val = (((e << 2) | (m >> 21)) + 1) >> 1
    e2m1_tmp = torch.where(combined_val < 0x7, combined_val, 0x7)
    e2m1_value = (((s >> 28) & 0xF) | e2m1_tmp).to(torch.uint8)

    # Pack 2 4-bit values into 8-bit
    x_mxfp4 = e2m1_value[..., ::2] | (e2m1_value[..., 1::2] << 4)

    # Recover last dimension's shape
    x_mxfp4 = torch.flatten(x_mxfp4, -2, -1)

    # Remove padded values
    if x.shape[-1] % MXFP4_QUANT_BLOCK_SIZE != 0:
        x_mxfp4 = x_mxfp4[..., : x.shape[-1] // 2]

    # Reshape back to original
    mxfp4_shape = list(x_shape)
    mxfp4_shape = tuple(mxfp4_shape[:-1] + [mxfp4_shape[-1] // 2])
    x_mxfp4 = x_mxfp4.reshape(mxfp4_shape)
    bs_e8m0_shape = list(x_shape)
    bs_e8m0_shape = tuple(
        bs_e8m0_shape[:-1] + [bs_e8m0_shape[-1] // MXFP4_QUANT_BLOCK_SIZE]
    )
    bs_e8m0 = bs_e8m0.reshape(bs_e8m0_shape)

    return x_mxfp4, bs_e8m0


def mxfp4_to_f32(x):
    # 2 because we pack fp4 in uint8.
    x = x.repeat_interleave(2, dim=-1)
    x[..., ::2] = x[..., ::2] & 0xF
    x[..., 1::2] = x[..., 1::2] >> 4
    mxfp4_list = [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ]
    mxfp4_in_f32 = torch.tensor(mxfp4_list, dtype=torch.float32, device="cuda")
    return mxfp4_in_f32[x.long()]


def e8m0_to_f32(x):
    x_f32 = 2 ** ((x - 127).to(torch.float32))
    x_f32[x == 255] = float("nan")
    return x_f32


def torch_mxfp4_to_fp32(x, x_scales):
    # First convert the x to f32.
    x_f32 = mxfp4_to_f32(x)
    print(
        f"x.shape={x.shape} x_f32.shape={x_f32.shape} x_scales.shape={x_scales.shape}"
    )

    # Next convert the e8m0 scale to f32.
    x_scales = x_scales.repeat_interleave(32, dim=-1)
    x_scales_f32 = fp4_utils.e8m0_to_f32(x_scales)
    x_f32 = x_f32 * x_scales_f32

    return x_f32


def alloc_rand(shape, device, dtype, requires_grad=True):
    if dtype.itemsize == 1:
        tmp = 2 ** -(torch.randint(4, 8, shape, device=device, dtype=torch.float16))
        return tmp.to(dtype).requires_grad_(requires_grad)
    return torch.randn(shape, device=device, dtype=dtype, requires_grad=requires_grad)


# OCP mixed-format fp4 (mxfp4) has two elements packed in one uint8
str_to_torch_dtype = {
    "fp32": torch.float32,
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp8_e4m3": (
        torch.float8_e4m3fn if get_cdna_version() == 4 else torch.float8_e4m3fnuz
    ),
    "fp8_e5m2": torch.float8_e5m2 if get_cdna_version() == 4 else torch.float8_e5m2fnuz,
    "mxfp4_e2m1": torch.uint8,
}

torch_to_tl_dtype = {
    torch.float32: tl.float32,
    torch.bfloat16: tl.bfloat16,
    torch.float16: tl.float16,
    torch.float8_e4m3fn: tl.float8e4nv,
    torch.float8_e4m3fnuz: tl.float8e4b8,
    torch.float8_e5m2: tl.float8e5,
    torch.float8_e5m2fnuz: tl.float8e5b16,
    torch.uint8: tl.uint8,
}


# Note: Eventually all these combinations will be supported
# Hardware native OCP
# ("fp8_e5m2", "mxfp4_e2m1"),
# ("mxfp4_e2m1", "mxfp4_e2m1"),
# Software emulation that upcasts mxfp4 to fp16
# ("fp16", "mxfp4_e2m1"),
# ("bf16", "mxfp4_e2m1"),
@pytest.mark.parametrize(
    "M, N, K, E, top_k",
    [
        (1024, 6144, 4096, 8, 1),
        # (64, 64, 128, 8, 2),
        # (16, 256, 256, 128, 4),
        (1000, 704, 800, 3, 1),
        # (1000, 704, 800, 8, 2),
        # (64, 14336, 4096, 8, 2),
        # (16, 14336, 128, 8, 2),  # not working either
        # (16, 14336, 4096, 4, 1),
        # (1, 14336, 128, 4, 2),
        # (3, 14336, 128, 4, 2),
        # (16, 14336, 128, 1, 1),
        # (64, 7186, 128, 8, 2),
        # (64, 3584, 128, 8, 2),
        # (64, 1792, 128, 8, 2),
        # (64, 64, 128, 8, 2),
        # (1, 1024, 16384, 2, 1),
    ],
)
@pytest.mark.parametrize(
    "a_dtype_str, b_dtype_str",
    [
        # Hardware native OCP
        ("mxfp4_e2m1", "mxfp4_e2m1"),  # TODO Add support for other types
    ],
)
@pytest.mark.parametrize("routed_weight", [False, True])
@pytest.mark.parametrize("swizzle_mx_scale", [False])  # TODO Add support for swizzle
def test_fused_moe(
    tokens: int,
    model_dim: int,
    inter_dim: int,
    top_k: int,
    E: int,
    a_dtype_str: str,
    b_dtype_str: str,
    routed_weight: bool,
    swizzle_mx_scale: bool,
):
    block_size_m = 128
    if triton.runtime.driver.active.get_current_target().arch not in ("gfx950"):
        pytest.skip("MXFP4 not supported on this architecture")
    aiter_quant = aiter.get_torch_quant(aiter.QuantType.per_1x32)
    is_a_mixed_input = a_dtype_str.startswith("mx")
    is_b_mixed_input = b_dtype_str.startswith("mx")
    a_dtype = str_to_torch_dtype[a_dtype_str]
    c_dtype = torch.bfloat16 if is_a_mixed_input else a_dtype
    fp16_dtype = torch.float16 if a_dtype_str == "fp16" else torch.bfloat16
    a_tri_ = alloc_rand(
        (tokens, model_dim), dtype=c_dtype, device="cuda", requires_grad=False
    )
    b1_tri_ = alloc_rand(
        (E, inter_dim * 2, model_dim), dtype=c_dtype, device="cuda", requires_grad=False
    )
    b2_tri_ = alloc_rand(
        (E, model_dim, inter_dim), dtype=c_dtype, device="cuda", requires_grad=False
    )

    # a_tri = alloc_rand((1, 1), dtype=fp16_dtype, device="cuda", requires_grad=False)
    # a_tri = a_tri.repeat(tokens, inter_dim)
    # b_tri = alloc_rand((1, 1, 1), dtype=fp16_dtype, device="cuda", requires_grad=False)
    # b_tri = b_tri.repeat(E, model_dim, inter_dim)

    c1_tri = torch.zeros(
        (tokens, top_k, inter_dim), dtype=c_dtype, device="cuda", requires_grad=False
    )

    c2_tri = torch.zeros(
        (tokens, model_dim), dtype=c_dtype, device="cuda", requires_grad=False
    )
    a_scale = torch.tensor([1.00], dtype=torch.float32, device="cuda")
    b_scale = torch.tensor([1.00] * E, dtype=torch.float32, device="cuda")
    # Reference inputs
    a_ref, b1_ref, b2_ref, c1_ref, c2_ref = (
        a_tri_.clone(),
        b1_tri_.clone(),
        b2_tri_.clone(),
        c1_tri.clone(),
        c2_tri.clone(),
    )

    # Try fixed config for now
    config = {
        "BLOCK_SIZE_M": 128,
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_K": 128,
        "GROUP_SIZE_M": 4,
        "num_warps": 8,
        "num_stages": 2,
        "waves_per_eu": 0,
        "matrix_instr_nonkdim": 16,
        "kpack": 1,
    }

    values = torch.randn((tokens, E), dtype=c_dtype, device="cuda")
    # softmax_vals = torch.softmax(values, dim=1)
    topk_weights, topk_ids = fused_topk(a_tri_, values, top_k, True)

    sorted_token_ids, sorted_weights, expert_ids, num_tokens_post_padded, moe_buf = (
        moe_sorting(topk_ids, topk_weights, E, model_dim, a_tri_.dtype, block_size_m)
    )
    # print(f"{topk_ids=}, {topk_weights=}, {E=}, {model_dim=}, {a_tri_.dtype=}, {block_size_m=}")
    # expert_ids = torch.zeros_like(expert_ids)
    # sorted_token_ids = torch.load('./sorted_token_ids.pt')
    # sorted_token_ids = torch.arange(sorted_token_ids.shape[-1])
    # print(f"{sorted_token_ids=}")
    # print(f"{expert_ids=}")
    # print(f"{num_tokens_post_padded=}")
    # print(f"{sorted_token_ids[num_tokens_post_padded - 1]=}, {sorted_token_ids[num_tokens_post_padded]=} ")
    # sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
    #     topk_ids, config["BLOCK_SIZE_M"], E
    # )
    # for i in range(num_tokens_post_padded):
    #     if sorted_token_ids[i] >= 1024:
    #         sorted_token_ids[i] = sorted_token_ids[i] % 1024

    # for i in range(num_tokens_post_padded):
    #     print(f"{i=}:{sorted_token_ids[i]=}")
    # Downcast a tensor to mxfp4 and upcast back for reference
    # Downcast b tensor to mxfp4 and upcast back for reference
    if is_b_mixed_input:
        # b_ref = b_tri

        # swizzle_axis = 1 if swizzle_mx_scale else None  # TODO Add Swizzle support
        b1_tri, b1_mx_scales = aiter_quant(b1_tri_.view(-1, model_dim), shuffle=False)
        b2_tri, b2_mx_scales = aiter_quant(b2_tri_.view(-1, inter_dim), shuffle=False)

        b1_tri = b1_tri.view(E, inter_dim * 2, -1).cuda()
        b1_mx_scales = b1_mx_scales.view(E, inter_dim * 2, -1).cuda()

        b2_tri = b2_tri.view(E, model_dim, -1).cuda()
        b2_mx_scales = b2_mx_scales.view(E, model_dim, -1).cuda()
        # b_tri = b_tri.repeat(1 ,2, 1)
        # b_mx_scales = b_mx_scales.repeat(1, 2, 1)
        # b1_mx_scales = torch.ones_like(b1_mx_scales) * 127
        # b2_mx_scales = torch.ones_like(b2_mx_scales) * 127
        # TODO Add Upcast support
        # b_ref = torch_upcast_from_mxfp(
        #    b_tri, b_mx_scales, fp16_dtype, axis=2, swizzle_axis=swizzle_axis
        # )
        b1_ref = torch_mxfp4_to_fp32(b1_tri, b1_mx_scales)
        b2_ref = torch_mxfp4_to_fp32(b2_tri, b2_mx_scales)
        print(
            f"b1_ref.shape={b1_ref.shape} b1_tri.shape={b1_tri.shape} b1_tri., b1_mx_scales.shape={b1_mx_scales.shape}"
        )
        print(
            f"b2_ref.shape={b2_ref.shape} b2_tri.shape={b2_tri.shape} b2_tri., b2_mx_scales.shape={b2_mx_scales.shape}"
        )

    # b1_tri = shuffle_weight(b1_tri, layout=(16, 16))
    # b2_tri = shuffle_weight(b2_tri, layout=(16, 16))
    if is_a_mixed_input:
        # a_ref = a_tri

        # swizzle_axis = 0 if swizzle_mx_scale else None  # TODO Add Swizzle support
        # from aiter import get_torch_quant
        # aiter_quant = get_torch_quant(aiter.QuantType.per_1x32)
        a_tri, a_mx_scales = dynamic_mxfp4_quant(a_tri_)
        # TODO Add Upcast support
        # a_ref = torch_upcast_from_mxfp(
        #    a_tri, a_mx_scales, fp16_dtype, axis=1, swizzle_axis=swizzle_axisv
        # )
        a_ref = torch_mxfp4_to_fp32(a_tri, a_mx_scales)
    else:
        a_ref = a_ref.to(fp16_dtype)

    # Torch
    b_zp = None
    group_size = 0
    # a_scale and b_scale not used actually
    c1_ref = torch_moe_stage1(
        a_ref,
        b1_ref,
        c1_ref,
        a_scale,
        b_scale,
        b_zp,
        group_size,
        topk_ids,
        topk_weights,
        False,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        ActivationType.Silu,
        dtype=fp16_dtype,
    )

    a_mx_scales = moe_mxfp4_sort(
        a_mx_scales,
        sorted_ids=sorted_token_ids,
        num_valid_ids=num_tokens_post_padded,
        token_num=tokens,
        block_size=block_size_m,
    )
    # print(f"{sorted_token_ids=}, {num_tokens_post_padded=}, {tokens=}, {block_size_m=}")
    b1_tri, b1_mx_scales = aiter_quant(b1_tri_.view(-1, model_dim), shuffle=True)

    b1_tri = b1_tri.view(E, inter_dim * 2, -1).cuda()
    b1_mx_scales = b1_mx_scales.view(E, inter_dim * 2, -1).cuda()

    out1_ck, us = run_perftest(
        ck_moe_stage1,
        a_tri,
        b1_tri,
        b2_tri,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        b1_mx_scales,
        a_mx_scales,
        fp16_dtype,
        top_k,
        block_size_m,
        ActivationType.Silu,
        sorted_weights=None,
    )
    checkAllclose(
        c1_ref,
        out1_ck,
        msg=f"[perf]  ck_moe_stage1:{us:>8.2f} us, {(tokens * model_dim * inter_dim * 2 * top_k * 2 + tokens * model_dim * inter_dim * 2 * top_k * 2 / 32)/us/1000/1000:>8.2f} tflops......(quant)",
    )

    if is_a_mixed_input:
        # a_ref = a_tri

        # swizzle_axis = 0 if swizzle_mx_scale else None  # TODO Add Swizzle support
        aiter_quant = get_torch_quant(aiter.QuantType.per_1x32)
        a2_tri, a2_mx_scales = dynamic_mxfp4_quant(c1_ref.view(-1, inter_dim))
        a2_tri = a2_tri.view(tokens, top_k, -1)
        a2_mx_scales = a2_mx_scales.view(tokens, top_k, -1)
        # a2_mx_scales = torch.ones_like(a2_mx_scales) * 127
        # TODO Add Upcast support
        # a_ref = torch_upcast_from_mxfp(
        #    a_tri, a_mx_scales, fp16_dtype, axis=1, swizzle_axis=swizzle_axisv
        # )
        a2_ref = torch_mxfp4_to_fp32(a2_tri, a2_mx_scales)
    else:
        a2_ref = a2_ref.to(fp16_dtype)
        a2_mx_scales = None

    c2_ref = torch_moe_stage2(
        a2_ref,
        b2_ref,
        c2_ref,
        a_scale,
        b_scale,
        b_zp,
        group_size,
        topk_ids,
        topk_weights,
        True,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        # ActivationType.Silu,
        dtype=fp16_dtype,
    )

    a2_mx_scales = moe_mxfp4_sort(
        a2_mx_scales,
        sorted_ids=sorted_token_ids,
        num_valid_ids=num_tokens_post_padded,
        token_num=tokens,
        block_size=block_size_m,
    )
    # aiter_quant = aiter.get_triton_quant(aiter.QuantType.per_1x32)
    b2_tri2, b2_mx_scales2 = aiter_quant(b2_tri_.view(-1, inter_dim), shuffle=True)
    b2_tri2 = b2_tri2.view(E, model_dim, -1)
    b2_mx_scales2 = b2_mx_scales2.view(E, model_dim, -1)

    out2_ck, us = run_perftest(
        ck_moe_stage2,
        a2_tri,
        b2_tri2,
        b2_tri2,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        b2_mx_scales2,
        a2_mx_scales,
        fp16_dtype,
        top_k,
        block_size_m,
        sorted_weights=sorted_weights,
    )

    checkAllclose(
        c2_ref,
        out2_ck,
        msg=f"[perf]  ck_moe_stage2:{us:>8.2f} us, {((2 * tokens * model_dim * inter_dim * top_k) + (2 * tokens * model_dim * inter_dim * top_k / 32)) /us/1000/1000:>8.2f} tflops......(quant)",
    )
    out_ck, us = run_perftest(
        ck_moe_2stages,
        a_tri_,
        b1_tri,
        b2_tri2,
        topk_weights,
        topk_ids,
        quant_type=aiter.QuantType.per_1x32,
        fc1_scale=b1_mx_scales,  # [expert(local_expert:EP), inter_dim, 1]
        fc2_scale=b2_mx_scales2,  # [expert(local_expert:EP), model_dim, 1]
        block_size=block_size_m,
        activation=ActivationType.Silu,
        doweight_stage1=False,
    )

    checkAllclose(
        c2_ref,
        out_ck,
        msg=f"ck_moe_2stages:{us:>8.2f} us, {(2 * tokens * model_dim * inter_dim * top_k * 3 + 2 * tokens * model_dim * inter_dim * top_k * 3 / 32)/us/1000/1000:>8.2f} tflops......(quant)",
        # rtol=5e-2, atol=5e-2
    )


test_fused_moe(8192, 6144, 4096, 2, 8, "mxfp4_e2m1", "mxfp4_e2m1", False, False)
# test_fused_moe(512, 2048, 2048, 2, 4, "mxfp4_e2m1", "mxfp4_e2m1", False, False)
