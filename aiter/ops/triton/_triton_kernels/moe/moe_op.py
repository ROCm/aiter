# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from aiter.ops.triton.utils._triton.pid_preprocessing import pid_grid, remap_xcd
from aiter.ops.triton.utils._triton.moe_common import _write_zeros_to_output, _gluon_write_zeros_to_output
from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr

# Source:
# MoE Kernel adapted from VLLM


_fused_moe_kernel_gptq_awq_repr = make_kernel_repr(
    "_fused_moe_kernel_gptq_awq",
    [
        "N",
        "K",
        "group_size",
        "BLOCK_SIZE_M",
        "BLOCK_SIZE_N",
        "BLOCK_SIZE_K",
        "GROUP_SIZE_M",
        "EVEN_K",
        "MUL_ROUTED_WEIGHT",
        "top_k",
        "compute_type",
        "has_zp",
        "use_int4_w4a16",
        "use_int8_w8a16",
        "NUM_XCDS",
    ],
)

_fused_moe_persistent_kernel_gptq_awq_repr = make_kernel_repr(
    "_fused_moe_persistent_kernel_gptq_awq",
    [
        "N",
        "K",
        "group_size",
        "BLOCK_SIZE_M",
        "BLOCK_SIZE_N",
        "BLOCK_SIZE_K",
        "GROUP_SIZE_M",
        "EVEN_K",
        "NUM_SMS",
        "MUL_ROUTED_WEIGHT",
        "top_k",
        "compute_type",
        "has_zp",
        "use_int4_w4a16",
        "use_int8_w8a16",
        "NUM_XCDS",
    ],
)

_fused_moe_kernel_repr = make_kernel_repr(
    "_fused_moe_kernel",
    [
        "group_n",
        "group_k",
        "BLOCK_SIZE_M",
        "BLOCK_SIZE_N",
        "BLOCK_SIZE_K",
        "GROUP_SIZE_M",
        "EVEN_K",
        "MUL_ROUTED_WEIGHT",
        "top_k",
        "compute_type",
        "use_fp8_w8a8",
        "use_int8_w8a16",
        "NUM_XCDS",
    ],
)

_fused_moe_persistent_kernel_repr = make_kernel_repr(
    "_fused_moe_persistent_kernel",
    [
        "group_n",
        "group_k",
        "BLOCK_SIZE_M",
        "BLOCK_SIZE_N",
        "BLOCK_SIZE_K",
        "GROUP_SIZE_M",
        "EVEN_K",
        "NUM_SMS",
        "MUL_ROUTED_WEIGHT",
        "top_k",
        "compute_type",
        "use_fp8_w8a8",
        "use_int8_w8a16",
        "NUM_XCDS",
    ],
)


@triton.heuristics(
    {
        "EVEN_K": lambda args: args["K"] % args["BLOCK_SIZE_K"] == 0,
    }
)
@triton.jit(repr=_fused_moe_kernel_gptq_awq_repr)
def _fused_moe_kernel_gptq_awq(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    b_scale_ptr,
    b_zp_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    # Matrix dimensions
    N: tl.constexpr,
    K: tl.constexpr,
    EM,
    num_valid_tokens,
    # The stride variables represent how much to increase the ptr by when
    # moving by 1 element in a particular dimension. E.g. `stride_am` is
    # how much to increase `a_ptr` by to get the element one row down
    # (A has M rows).
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_bse,
    stride_bsk,
    stride_bsn,
    stride_bze,
    stride_bzk,
    stride_bzn,
    group_size: tl.constexpr,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    EVEN_K: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
    has_zp: tl.constexpr,
    use_int4_w4a16: tl.constexpr,
    use_int8_w8a16: tl.constexpr,
    NUM_XCDS: tl.constexpr,
):
    """
    Implements the fused computation for a Mixture of Experts (MOE) using
    token and expert matrices.

    Key Parameters:
    - A: The input tensor representing tokens with shape (*, K), where '*' can
        be any shape representing batches and K is the feature dimension of
        each token.
    - B: The stacked MOE weight tensor with shape (E, N, K), where E is
        the number of experts, K is the input feature dimension, and N is
        the output feature dimension.
    - C: The output cache tensor with shape (M, topk, N), where M is the
        total number of tokens post padding, topk is the number of times
        each token is repeated, and N is the output feature dimension.
    - sorted_token_ids: A tensor containing the sorted indices of tokens,
        repeated topk times and arranged by the expert index they are
        assigned to.
    - expert_ids: A tensor containing the indices of the expert for each
        block. It determines which expert matrix from B should be used for
        each block in A.
    This kernel performs the multiplication of a token by its corresponding
    expert matrix as determined by `expert_ids`. The sorting of
    `sorted_token_ids` by expert index and padding ensures divisibility by
    BLOCK_SIZE_M, which is necessary to maintain consistency in block matrix
    multiplication across different blocks processed by the same expert.
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    pid = tl.program_id(axis=0)
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)

    num_pid_m = tl.cdiv(num_tokens_post_padded, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    GRID_MN = num_pid_n * num_pid_m
    if pid < GRID_MN:
        pid = remap_xcd(pid, GRID_MN, NUM_XCDS)
    else:
        return
    pid_m, pid_n = pid_grid(pid, num_pid_m, num_pid_n, GROUP_SIZE_M)

    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
    token_mask = offs_token < num_valid_tokens

    off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    if off_experts == -1:
        # -----------------------------------------------------------
        # Write back zeros to the output when the expert is not
        # in the current expert parallel rank.
        _write_zeros_to_output(
            c_ptr,
            stride_cm,
            stride_cn,
            pid_n,
            N,
            offs_token,
            token_mask,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            compute_type,
        )
        return

    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (
        offs_token[:, None] // top_k * stride_am + offs_k[None, :] * stride_ak
    )

    if use_int4_w4a16:
        b_ptrs = (
            b_ptr
            + off_experts * stride_be
            + (offs_k[:, None] // 2) * stride_bk
            + offs_bn[None, :] * stride_bn
        )
        b_shifter = (offs_k[:, None] % 2) * 4
    elif use_int8_w8a16:
        b_ptrs = (
            b_ptr
            + off_experts * stride_be
            + offs_k[:, None] * stride_bk
            + offs_bn[None, :] * stride_bn
        )

    if not has_zp and use_int4_w4a16:
        b_zp_num = 8
    if not has_zp and use_int8_w8a16:
        b_zp_num = 128
    elif has_zp and use_int4_w4a16:
        b_zp_shifter = (offs_bn[None, :] % 2) * 4

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the
        # K dimension.

        if EVEN_K:
            k_mask = None
            k_other = None
            a = tl.load(
                a_ptrs,
                mask=token_mask[:, None],
                other=0.0,
            )
        else:
            k_mask = offs_k[:, None] < K - k * BLOCK_SIZE_K
            k_other = 0.0
            a = tl.load(
                a_ptrs,
                mask=token_mask[:, None] & (offs_k[None, :] < K - k * BLOCK_SIZE_K),
                other=0.0,
            )

        b = tl.load(b_ptrs, mask=k_mask, other=k_other)

        if use_int4_w4a16:
            b = (b >> b_shifter) & 0xF

        b_scale_ptrs = (
            b_scale_ptr
            + off_experts * stride_bse
            + offs_bn[None, :] * stride_bsn
            + ((offs_k[:, None] + BLOCK_SIZE_K * k) // group_size) * stride_bsk
        )
        b_scale = tl.load(b_scale_ptrs, mask=k_mask, other=k_other)
        b_scale = b_scale.to(tl.float32)

        if has_zp and use_int4_w4a16:
            offs_k_true = (offs_k[:, None] + BLOCK_SIZE_K * k) // group_size
            b_zp_ptrs = (
                b_zp_ptr
                + off_experts * stride_bze
                + (offs_bn[None, :] // 2) * stride_bzn
                + offs_k_true * stride_bzk
            )
            b_zp = tl.load(b_zp_ptrs, mask=k_mask, other=k_other)
            b_zp = (b_zp >> b_zp_shifter) & 0xF
            b_zp = b_zp.to(tl.float32)
        elif has_zp and use_int8_w8a16:
            offs_k_true = (offs_k[:, None] + BLOCK_SIZE_K * k) // group_size
            b_zp_ptrs = (
                b_zp_ptr
                + off_experts * stride_bze
                + offs_bn[None, :] * stride_bzn
                + offs_k_true * stride_bzk
            )
            b_zp = tl.load(b_zp_ptrs, mask=k_mask, other=k_other)
            b_zp = b_zp.to(tl.float32)

        # We accumulate along the K dimension.
        if has_zp:
            b = ((b.to(tl.float32) - b_zp) * b_scale).to(compute_type)
        else:
            b = ((b.to(tl.float32) - b_zp_num) * b_scale).to(compute_type)
        accumulator = tl.dot(a, b, acc=accumulator)

        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        if use_int4_w4a16:
            b_ptrs += (BLOCK_SIZE_K // 2) * stride_bk
        else:
            b_ptrs += BLOCK_SIZE_K * stride_bk

    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0)
        accumulator = accumulator * moe_weight[:, None]

    accumulator = accumulator.to(compute_type)
    # -----------------------------------------------------------
    # Write back the block of the output
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


# @triton.heuristics(
#     {
#         "EVEN_K": lambda args: args["K"] % args["BLOCK_SIZE_K"] == 0,
#     }
# )
@gluon.jit(repr=_fused_moe_kernel_repr)
def _gluon_fused_moe_unroll_k_kernel(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    a_scale_ptr,
    b_scale_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    # Matrix dimensions
    N,
    # K,
    num_valid_tokens,
    # The stride variables represent how much to increase the ptr by when
    # moving by 1 element in a particular dimension. E.g. `stride_am` is
    # how much to increase `a_ptr` by to get the element one row down
    # (A has M rows).
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_asm,
    stride_ask,
    stride_bse,
    stride_bsk,
    stride_bsn,
    # Block size for block-wise quantization
    group_n: gl.constexpr,
    group_k: gl.constexpr,
    # Meta-parameters
    BLOCK_SIZE_M: gl.constexpr,
    BLOCK_SIZE_N: gl.constexpr,
    BLOCK_SIZE_K: gl.constexpr,
    GROUP_SIZE_M: gl.constexpr,
    MUL_ROUTED_WEIGHT: gl.constexpr,
    top_k: gl.constexpr,
    compute_type: gl.constexpr,
    use_fp8_w8a8: gl.constexpr,
    use_int8_w8a16: gl.constexpr,
    NUM_XCDS: gl.constexpr,
    # Unroll K loop
    K: gl.constexpr,
    num_warps: gl.constexpr,
):
    """
    Implements the fused computation for a Mixture of Experts (MOE) using
    token and expert matrices.

    Key Parameters:
    - A: The input tensor representing tokens with shape (*, K), where '*' can
        be any shape representing batches and K is the feature dimension of
        each token.
    - B: The stacked MOE weight tensor with shape (E, N, K), where E is
        the number of experts, K is the input feature dimension, and N is
        the output feature dimension.
    - C: The output cache tensor with shape (M, topk, N), where M is the
        total number of tokens post padding, topk is the number of times
        each token is repeated, and N is the output feature dimension.
    - sorted_token_ids: A tensor containing the sorted indices of tokens,
        repeated topk times and arranged by the expert index they are
        assigned to.
    - expert_ids: A tensor containing the indices of the expert for each
        block. It determines which expert matrix from B should be used for
        each block in A.
    This kernel performs the multiplication of a token by its corresponding
    expert matrix as determined by `expert_ids`. The sorting of
    `sorted_token_ids` by expert index and padding ensures divisibility by
    BLOCK_SIZE_M, which is necessary to maintain consistency in block matrix
    multiplication across different blocks processed by the same expert.
    """

    SUB_BLOCK_SIZE_N: gl.constexpr = 64
    EVEN_K: gl.constexpr = K % BLOCK_SIZE_K == 0

    blocked_a: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 16],
        threads_per_warp=[32, 2],
        warps_per_cta=[num_warps, 1],
        order=[1, 0],
    )

    blocked_b: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[16, 1],
        threads_per_warp=[4, 16],
        warps_per_cta=[1, num_warps],
        order=[0, 1],
    )

    if use_fp8_w8a8:
        mfma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
            version=3,
            instr_shape=[32, 32, 16],
            transposed=True,
            warps_per_cta=[num_warps, 1],
        )
    else:
        mfma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
            version=3,
            instr_shape=[16, 16, 16],
            transposed=True,
            warps_per_cta=[num_warps, 1],
        )

    mfma_a_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=mfma_layout, k_width=16
    )

    mfma_b_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=mfma_layout, k_width=16
    )

    blocked_d: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 8],
        threads_per_warp=[16, 4],
        warps_per_cta=[num_warps, 1],
        order=[1, 0],
    )

    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    pid = gl.program_id(axis=0)
    num_tokens_post_padded = gl.load(num_tokens_post_padded_ptr)

    num_pid_m = gl.cdiv(num_tokens_post_padded, BLOCK_SIZE_M)
    num_pid_n = gl.cdiv(N, BLOCK_SIZE_N)

    GRID_MN = num_pid_n * num_pid_m
    if pid < GRID_MN:
        # TODO: gluon
        pid = remap_xcd(pid, GRID_MN, NUM_XCDS)
    else:
        return
    # TODO: gluon
    pid_m, pid_n = pid_grid(pid, num_pid_m, num_pid_n, GROUP_SIZE_M)

    # TODO: i64
    offs_token_id = pid_m * BLOCK_SIZE_M + gl.arange(0, BLOCK_SIZE_M, layout=gl.SliceLayout(1, blocked_a))
    offs_token = gl.amd.cdna3.buffer_load(sorted_token_ids_ptr, offs_token_id)
    token_mask = offs_token < num_valid_tokens

    # TODO: i64
    off_experts = gl.load(expert_ids_ptr + pid_m)
    if off_experts == -1:
        # -----------------------------------------------------------
        # Write back zeros to the output when the expert is not
        # in the current expert parallel rank.
        _gluon_write_zeros_to_output(
            c_ptr,
            stride_cm,
            stride_cn,
            pid_n,
            N,
            offs_token,
            token_mask,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            compute_type,
        )
        return

    offs_bk = gl.arange(0, BLOCK_SIZE_K, layout=gl.SliceLayout(1, blocked_b))
    offs_ak = gl.arange(0, BLOCK_SIZE_K, layout=gl.SliceLayout(0, blocked_a))

    #####################################
    # preload scales
    #####################################

    if use_fp8_w8a8:
        if group_k > 0 and group_n > 0:
            offs_ks = 0
            a_scale0 = gl.amd.cdna3.buffer_load(a_scale_ptr + offs_ks * stride_ask, (offs_token // top_k) * stride_asm, mask=token_mask, other=0.0)
            offs_ks += (BLOCK_SIZE_K // group_k)
            a_scale1 = gl.amd.cdna3.buffer_load(a_scale_ptr + offs_ks * stride_ask, (offs_token // top_k) * stride_asm, mask=token_mask, other=0.0)
            offs_ks += (BLOCK_SIZE_K // group_k)
            a_scale2 = gl.amd.cdna3.buffer_load(a_scale_ptr + offs_ks * stride_ask, (offs_token // top_k) * stride_asm, mask=token_mask, other=0.0)
        else:
            a_scale = gl.load(a_scale_ptr)
            b_scale = gl.load(b_scale_ptr + off_experts)

    #####################################
    # pre-load a
    #####################################
    if EVEN_K:
        a0 = gl.amd.cdna3.buffer_load(a_ptr,
                                      offs_token[:, None] // top_k * stride_am + offs_ak[None, :] * stride_ak,
                                      mask=token_mask[:, None],
                                      other=0.0
        )
        a_ptr += BLOCK_SIZE_K * stride_ak
        a1 = gl.amd.cdna3.buffer_load(a_ptr,
                                      offs_token[:, None] // top_k * stride_am + offs_ak[None, :] * stride_ak,
                                      mask=token_mask[:, None],
                                      other=0.0
        )
        a_ptr += BLOCK_SIZE_K * stride_ak
        a2 = gl.amd.cdna3.buffer_load(a_ptr,
                                      offs_token[:, None] // top_k * stride_am + offs_ak[None, :] * stride_ak,
                                      mask=token_mask[:, None],
                                      other=0.0
        )
        a_ptr += BLOCK_SIZE_K * stride_ak
    else:
        k = 0
        a0 = gl.amd.cdna3.buffer_load(a_ptr,
                                      offs_token[:, None] // top_k * stride_am + offs_ak[None, :] * stride_ak,
                                      mask=token_mask[:, None] & (offs_ak[None, :] < K - k),
                                      other=0.0
        )
        a_ptr += BLOCK_SIZE_K * stride_ak
        k += BLOCK_SIZE_K
        a1 = gl.amd.cdna3.buffer_load(a_ptr,
                                      offs_token[:, None] // top_k * stride_am + offs_ak[None, :] * stride_ak,
                                      mask=token_mask[:, None] & (offs_ak[None, :] < K - k),
                                      other=0.0
        )
        a_ptr += BLOCK_SIZE_K * stride_ak
        k += BLOCK_SIZE_K
        a2 = gl.amd.cdna3.buffer_load(a_ptr,
                                      offs_token[:, None] // top_k * stride_am + offs_ak[None, :] * stride_ak,
                                      mask=token_mask[:, None] & (offs_ak[None, :] < K - k),
                                      other=0.0
        )
        a_ptr += BLOCK_SIZE_K * stride_ak
    a0_converted = gl.convert_layout(a0, mfma_a_layout)
    a1_converted = gl.convert_layout(a1, mfma_a_layout)
    a2_converted = gl.convert_layout(a2, mfma_a_layout)

    #####################################
    # loop n
    #####################################

    for n_start in range(0, BLOCK_SIZE_N, SUB_BLOCK_SIZE_N):
        # TODO: i64
        offs_bn = (pid_n * BLOCK_SIZE_N + n_start + gl.arange(0, SUB_BLOCK_SIZE_N, layout=gl.SliceLayout(0, blocked_b))) % N

        #####################################
        # load b scale
        #####################################
        if use_int8_w8a16:
            b_scale = gl.amd.cdna3.buffer_load(b_scale_ptr + off_experts * stride_bse, offs_bn * stride_bsn)

        # -----------------------------------------------------------
        # Iterate to compute a block of the C matrix.
        # We accumulate into a `[BLOCK_SIZE_M, SUB_BLOCK_SIZE_N]` block
        # of fp32 values for higher accuracy.
        # `accumulator` will be converted back to fp16 after the loop.
        accumulator = gl.zeros((BLOCK_SIZE_M, SUB_BLOCK_SIZE_N), dtype=gl.float32, layout=mfma_layout)

        #####################################
        # load b
        #####################################

        if EVEN_K:
            # b0
            b0_ptr = b_ptr
            b0 = gl.amd.cdna3.buffer_load(b0_ptr + off_experts * stride_be,
                                          offs_bk[:, None] * stride_bk + offs_bn[None, :] * stride_bn
            )
            # b1
            b1_ptr = b0_ptr + BLOCK_SIZE_K * stride_bk
            b1 = gl.amd.cdna3.buffer_load(b1_ptr + off_experts * stride_be,
                                          offs_bk[:, None] * stride_bk + offs_bn[None, :] * stride_bn
            )
            # b2
            b2_ptr = b1_ptr + BLOCK_SIZE_K * stride_bk
            b2 = gl.amd.cdna3.buffer_load(b2_ptr + off_experts * stride_be,
                                          offs_bk[:, None] * stride_bk + offs_bn[None, :] * stride_bn
            )
        else:
            # b0
            k = 0
            b0_ptr = b_ptr
            b0 = gl.amd.cdna3.buffer_load(b0_ptr + off_experts * stride_be,
                                          offs_bk[:, None] * stride_bk + offs_bn[None, :] * stride_bn,
                                          mask=offs_bk[:, None] < K - k,
                                          other=0.0
            )
            # b1
            k += BLOCK_SIZE_K
            b1_ptr = b0_ptr + BLOCK_SIZE_K * stride_bk
            b1 = gl.amd.cdna3.buffer_load(b1_ptr + off_experts * stride_be,
                                          offs_bk[:, None] * stride_bk + offs_bn[None, :] * stride_bn,
                                          mask=offs_bk[:, None] < K - k,
                                          other=0.0
            )
            # b2
            k += BLOCK_SIZE_K
            b2_ptr = b1_ptr + BLOCK_SIZE_K * stride_bk
            b2 = gl.amd.cdna3.buffer_load(b2_ptr + off_experts * stride_be,
                                          offs_bk[:, None] * stride_bk + offs_bn[None, :] * stride_bn,
                                          mask=offs_bk[:, None] < K - k,
                                          other=0.0
            )

        b0_converted = gl.convert_layout(b0, mfma_b_layout)
        b1_converted = gl.convert_layout(b1, mfma_b_layout)
        b2_converted = gl.convert_layout(b2, mfma_b_layout)

        #####################################
        # We accumulate along the K dimension.
        #####################################
        if use_int8_w8a16:
            accumulator = gl.amd.cdna3.mfma(a0_converted, b0_converted.to(compute_type), accumulator)
            accumulator = gl.amd.cdna3.mfma(a1_converted, b1_converted.to(compute_type), accumulator)
            accumulator = gl.amd.cdna3.mfma(a2_converted, b2_converted.to(compute_type), accumulator)
        elif use_fp8_w8a8:
            if group_k > 0 and group_n > 0:
                offs_ks = 0
                b_scale0 = gl.amd.cdna3.buffer_load(b_scale_ptr + offs_ks * stride_bsk + off_experts * stride_bse, offs_bn * stride_bsn)
                offs_ks += (BLOCK_SIZE_K // group_k)
                b_scale1 = gl.amd.cdna3.buffer_load(b_scale_ptr + offs_ks * stride_bsk + off_experts * stride_bse, offs_bn * stride_bsn)
                offs_ks += (BLOCK_SIZE_K // group_k)
                b_scale2 = gl.amd.cdna3.buffer_load(b_scale_ptr + offs_ks * stride_bsk + off_experts * stride_bse, offs_bn * stride_bsn)
                
                a0_scale_converted = gl.convert_layout(a_scale0, gl.SliceLayout(1, mfma_layout))
                a1_scale_converted = gl.convert_layout(a_scale1, gl.SliceLayout(1, mfma_layout))
                a2_scale_converted = gl.convert_layout(a_scale2, gl.SliceLayout(1, mfma_layout))
                b0_scale_converted = gl.convert_layout(b_scale0, gl.SliceLayout(0, mfma_layout))
                b1_scale_converted = gl.convert_layout(b_scale1, gl.SliceLayout(0, mfma_layout))
                b2_scale_converted = gl.convert_layout(b_scale2, gl.SliceLayout(0, mfma_layout))

                accumulator = gl.amd.cdna3.mfma(a0_converted, b0_converted, accumulator) * a0_scale_converted[:, None] * b0_scale_converted[None, :]
                accumulator = gl.amd.cdna3.mfma(a1_converted, b1_converted, accumulator) * a1_scale_converted[:, None] * b1_scale_converted[None, :]
                accumulator = gl.amd.cdna3.mfma(a2_converted, b2_converted, accumulator) * a2_scale_converted[:, None] * b2_scale_converted[None, :]
            else:
                accumulator = gl.amd.cdna3.mfma(a0_converted, b0_converted, accumulator)
                accumulator = gl.amd.cdna3.mfma(a1_converted, b1_converted, accumulator)
                accumulator = gl.amd.cdna3.mfma(a2_converted, b2_converted, accumulator)
        else:
            accumulator = gl.amd.cdna3.mfma(a0_converted, b0_converted, accumulator)
            accumulator = gl.amd.cdna3.mfma(a1_converted, b1_converted, accumulator)
            accumulator = gl.amd.cdna3.mfma(a2_converted, b2_converted, accumulator)

        if MUL_ROUTED_WEIGHT:
            moe_weight = gl.amd.cdna3.buffer_load(topk_weights_ptr, offs_token, mask=token_mask, other=0)
            moe_weight_converted = gl.convert_layout(moe_weight, gl.SliceLayout(1, mfma_layout))
            accumulator *= moe_weight_converted[:, None]
        if use_int8_w8a16:
            b_scale_converted = gl.convert_layout(b_scale, gl.SliceLayout(0, mfma_layout))
            accumulator = (accumulator * b_scale_converted[None, :]).to(compute_type)
        elif use_fp8_w8a8:
            if group_k > 0 and group_n > 0:
                accumulator = accumulator.to(compute_type)
            else:
                accumulator = (accumulator * a_scale * b_scale).to(compute_type)
        else:
            accumulator = accumulator.to(compute_type)

        # -----------------------------------------------------------
        # Write back the block of the output
        offs_cn = pid_n * BLOCK_SIZE_N + n_start + gl.arange(0, SUB_BLOCK_SIZE_N, gl.SliceLayout(0, blocked_d))
        offs_token_converted = gl.convert_layout(offs_token, gl.SliceLayout(1, blocked_d))
        token_mask_converted = gl.convert_layout(token_mask, gl.SliceLayout(1, blocked_d))
        c_mask = token_mask_converted[:, None] & (offs_cn[None, :] < N)
        gl.amd.cdna3.buffer_store(gl.convert_layout(accumulator, blocked_d),
                                  c_ptr, stride_cm * offs_token_converted[:, None] + stride_cn * offs_cn[None, :],
                                  mask=c_mask)

@triton.heuristics(
    {
        "EVEN_K": lambda args: args["K"] % args["BLOCK_SIZE_K"] == 0,
    }
)
@triton.jit(repr=_fused_moe_persistent_kernel_gptq_awq_repr)
def _fused_moe_persistent_kernel_gptq_awq(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    b_scale_ptr,
    b_zp_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    # Matrix dimensions
    N: tl.constexpr,
    K: tl.constexpr,
    num_valid_tokens,
    # The stride variables represent how much to increase the ptr by when
    # moving by 1 element in a particular dimension. E.g. `stride_am` is
    # how much to increase `a_ptr` by to get the element one row down
    # (A has M rows).
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_bse,
    stride_bsk,
    stride_bsn,
    stride_bze,
    stride_bzk,
    stride_bzn,
    group_size: tl.constexpr,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    EVEN_K: tl.constexpr,
    NUM_SMS: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
    has_zp: tl.constexpr,
    use_int4_w4a16: tl.constexpr,
    use_int8_w8a16: tl.constexpr,
    NUM_XCDS: tl.constexpr,
):
    """
    Implements the fused computation for a Mixture of Experts (MOE) using
    token and expert matrices.

    Key Parameters:
    - A: The input tensor representing tokens with shape (*, K), where '*' can
        be any shape representing batches and K is the feature dimension of
        each token.
    - B: The stacked MOE weight tensor with shape (E, N, K), where E is
        the number of experts, K is the input feature dimension, and N is
        the output feature dimension.
    - C: The output cache tensor with shape (M, topk, N), where M is the
        total number of tokens post padding, topk is the number of times
        each token is repeated, and N is the output feature dimension.
    - sorted_token_ids: A tensor containing the sorted indices of tokens,
        repeated topk times and arranged by the expert index they are
        assigned to.
    - expert_ids: A tensor containing the indices of the expert for each
        block. It determines which expert matrix from B should be used for
        each block in A.
    This kernel performs the multiplication of a token by its corresponding
    expert matrix as determined by `expert_ids`. The sorting of
    `sorted_token_ids` by expert index and padding ensures divisibility by
    BLOCK_SIZE_M, which is necessary to maintain consistency in block matrix
    multiplication across different blocks processed by the same expert.
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    start_pid = tl.program_id(axis=0)
    # Load tile-invariant runtime constant
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)

    num_pid_m = tl.cdiv(num_tokens_post_padded, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    tile_id = start_pid

    offs_k = tl.arange(0, BLOCK_SIZE_K)

    num_tiles = num_pid_m * num_pid_n
    # Compute how many tiles are outside the padding region
    num_valid_tiles = tl.cdiv((num_tiles - tile_id), NUM_SMS)
    for _ in range(0, num_valid_tiles):
        tile_id_remapped = remap_xcd(tile_id, num_tiles, NUM_XCDS)
        pid_m, pid_n = pid_grid(tile_id_remapped, num_pid_m, num_pid_n, GROUP_SIZE_M)

        # Compute the mask
        offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
        offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
        token_mask = offs_token < num_valid_tokens
        off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)

        # Compute the A pointer
        a_ptrs = a_ptr + (
            offs_token[:, None] // top_k * stride_am + offs_k[None, :] * stride_ak
        )
        offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N

        if use_int4_w4a16:
            b_ptrs = (
                b_ptr
                + off_experts * stride_be
                + (offs_k[:, None] // 2) * stride_bk
                + offs_bn[None, :] * stride_bn
            )
            b_shifter = (offs_k[:, None] % 2) * 4
        elif use_int8_w8a16:
            b_ptrs = (
                b_ptr
                + off_experts * stride_be
                + offs_k[:, None] * stride_bk
                + offs_bn[None, :] * stride_bn
            )

        if not has_zp and use_int4_w4a16:
            b_zp_num = 8
        if not has_zp and use_int8_w8a16:
            b_zp_num = 128
        elif has_zp and use_int4_w4a16:
            b_zp_shifter = (offs_bn[None, :] % 2) * 4

        # -----------------------------------------------------------
        # Iterate to compute a block of the C matrix.
        # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
        # of fp32 values for higher accuracy.
        # `accumulator` will be converted back to fp16 after the loop.
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
            # Load the next block of A and B, generate a mask by checking the
            # K dimension.

            if EVEN_K:
                k_mask = None
                k_other = None
                a = tl.load(
                    a_ptrs,
                    mask=token_mask[:, None],
                    other=0.0,
                )
            else:
                k_mask = offs_k[:, None] < K - k * BLOCK_SIZE_K
                k_other = 0.0
                a = tl.load(
                    a_ptrs,
                    mask=token_mask[:, None] & (offs_k[None, :] < K - k * BLOCK_SIZE_K),
                    other=0.0,
                )
            b = tl.load(b_ptrs, mask=k_mask, other=k_other)

            if use_int4_w4a16:
                b = (b >> b_shifter) & 0xF

            b_scale_ptrs = (
                b_scale_ptr
                + off_experts * stride_bse
                + offs_bn[None, :] * stride_bsn
                + ((offs_k[:, None] + BLOCK_SIZE_K * k) // group_size) * stride_bsk
            )
            b_scale = tl.load(b_scale_ptrs, mask=k_mask, other=k_other)
            b_scale = b_scale.to(tl.float32)

            if has_zp and use_int4_w4a16:
                offs_k_true = (offs_k[:, None] + BLOCK_SIZE_K * k) // group_size
                b_zp_ptrs = (
                    b_zp_ptr
                    + off_experts * stride_bze
                    + (offs_bn[None, :] // 2) * stride_bzn
                    + offs_k_true * stride_bzk
                )
                b_zp = tl.load(b_zp_ptrs, mask=k_mask, other=k_other)
                b_zp = (b_zp >> b_zp_shifter) & 0xF
                b_zp = b_zp.to(tl.float32)
            elif has_zp and use_int8_w8a16:
                offs_k_true = (offs_k[:, None] + BLOCK_SIZE_K * k) // group_size
                b_zp_ptrs = (
                    b_zp_ptr
                    + off_experts * stride_bze
                    + offs_bn[None, :] * stride_bzn
                    + offs_k_true * stride_bzk
                )
                b_zp = tl.load(b_zp_ptrs, mask=k_mask, other=k_other)
                b_zp = b_zp.to(tl.float32)

            # We accumulate along the K dimension.
            if has_zp:
                b = ((b.to(tl.float32) - b_zp) * b_scale).to(compute_type)
            else:
                b = ((b.to(tl.float32) - b_zp_num) * b_scale).to(compute_type)
            accumulator = tl.dot(a, b, acc=accumulator)

            # Advance the ptrs to the next K block.
            a_ptrs += BLOCK_SIZE_K * stride_ak
            if use_int4_w4a16:
                b_ptrs += (BLOCK_SIZE_K // 2) * stride_bk
            else:
                b_ptrs += BLOCK_SIZE_K * stride_bk

        if MUL_ROUTED_WEIGHT:
            moe_weight = tl.load(
                topk_weights_ptr + offs_token, mask=token_mask, other=0
            )
            accumulator = accumulator * moe_weight[:, None]

        accumulator = accumulator.to(compute_type)
        # -----------------------------------------------------------
        # Write back the block of the output
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
        c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
        tl.store(c_ptrs, accumulator, mask=c_mask)

        tile_id += NUM_SMS


@triton.heuristics(
    {
        "EVEN_K": lambda args: args["K"] % args["BLOCK_SIZE_K"] == 0,
    }
)
@triton.jit(repr=_fused_moe_kernel_repr)
def _fused_moe_kernel(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    a_scale_ptr,
    b_scale_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    # Matrix dimensions
    N,
    K,
    num_valid_tokens,
    # The stride variables represent how much to increase the ptr by when
    # moving by 1 element in a particular dimension. E.g. `stride_am` is
    # how much to increase `a_ptr` by to get the element one row down
    # (A has M rows).
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_asm,
    stride_ask,
    stride_bse,
    stride_bsk,
    stride_bsn,
    # Block size for block-wise quantization
    group_n: tl.constexpr,
    group_k: tl.constexpr,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    EVEN_K: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
    use_fp8_w8a8: tl.constexpr,
    use_int8_w8a16: tl.constexpr,
    NUM_XCDS: tl.constexpr,
):
    """
    Implements the fused computation for a Mixture of Experts (MOE) using
    token and expert matrices.

    Key Parameters:
    - A: The input tensor representing tokens with shape (*, K), where '*' can
        be any shape representing batches and K is the feature dimension of
        each token.
    - B: The stacked MOE weight tensor with shape (E, N, K), where E is
        the number of experts, K is the input feature dimension, and N is
        the output feature dimension.
    - C: The output cache tensor with shape (M, topk, N), where M is the
        total number of tokens post padding, topk is the number of times
        each token is repeated, and N is the output feature dimension.
    - sorted_token_ids: A tensor containing the sorted indices of tokens,
        repeated topk times and arranged by the expert index they are
        assigned to.
    - expert_ids: A tensor containing the indices of the expert for each
        block. It determines which expert matrix from B should be used for
        each block in A.
    This kernel performs the multiplication of a token by its corresponding
    expert matrix as determined by `expert_ids`. The sorting of
    `sorted_token_ids` by expert index and padding ensures divisibility by
    BLOCK_SIZE_M, which is necessary to maintain consistency in block matrix
    multiplication across different blocks processed by the same expert.
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    pid = tl.program_id(axis=0)
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)

    num_pid_m = tl.cdiv(num_tokens_post_padded, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    GRID_MN = num_pid_n * num_pid_m
    if pid < GRID_MN:
        pid = remap_xcd(pid, GRID_MN, NUM_XCDS)
    else:
        return
    pid_m, pid_n = pid_grid(pid, num_pid_m, num_pid_n, GROUP_SIZE_M)

    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
    token_mask = offs_token < num_valid_tokens

    off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    if off_experts == -1:
        # -----------------------------------------------------------
        # Write back zeros to the output when the expert is not
        # in the current expert parallel rank.
        _write_zeros_to_output(
            c_ptr,
            stride_cm,
            stride_cn,
            pid_n,
            N,
            offs_token,
            token_mask,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            compute_type,
        )
        return

    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (
        offs_token[:, None] // top_k * stride_am + offs_k[None, :] * stride_ak
    )

    b_ptrs = (
        b_ptr
        + off_experts * stride_be
        + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    )
    if use_int8_w8a16:
        b_scale_ptrs = (
            b_scale_ptr + off_experts * stride_bse + offs_bn[None, :] * stride_bsn
        )
        b_scale = tl.load(b_scale_ptrs)

    if use_fp8_w8a8:
        if group_k > 0 and group_n > 0:
            a_scale_ptrs = a_scale_ptr + (offs_token // top_k) * stride_asm
            offs_bsn = offs_bn // group_n
            b_scale_ptrs = (
                b_scale_ptr + off_experts * stride_bse + offs_bsn * stride_bsn
            )
        else:
            a_scale = tl.load(a_scale_ptr)
            b_scale = tl.load(b_scale_ptr + off_experts)

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the
        # K dimension.

        if EVEN_K:
            a = tl.load(a_ptrs, mask=token_mask[:, None], other=0.0)
            b = tl.load(b_ptrs)
        else:
            a = tl.load(
                a_ptrs,
                mask=token_mask[:, None] & (offs_k[None, :] < K - k * BLOCK_SIZE_K),
                other=0.0,
            )
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        # We accumulate along the K dimension.
        if use_int8_w8a16:
            accumulator = tl.dot(a, b.to(compute_type), acc=accumulator)
        elif use_fp8_w8a8:
            if group_k > 0 and group_n > 0:
                k_start = k * BLOCK_SIZE_K
                offs_ks = k_start // group_k
                a_scale = tl.load(
                    a_scale_ptrs + offs_ks * stride_ask, mask=token_mask, other=0.0
                )
                b_scale = tl.load(b_scale_ptrs + offs_ks * stride_bsk)

                accumulator += tl.dot(a, b) * a_scale[:, None] * b_scale[None, :]
            else:
                accumulator = tl.dot(a, b, acc=accumulator)
        else:
            accumulator += tl.dot(a, b)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0)
        accumulator = accumulator * moe_weight[:, None]
    if use_int8_w8a16:
        accumulator = (accumulator * b_scale).to(compute_type)
    elif use_fp8_w8a8:
        if group_k > 0 and group_n > 0:
            accumulator = accumulator.to(compute_type)
        else:
            accumulator = (accumulator * a_scale * b_scale).to(compute_type)
    else:
        accumulator = accumulator.to(compute_type)
    # -----------------------------------------------------------
    # Write back the block of the output
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask, cache_modifier=".wt")


@triton.heuristics(
    {
        "EVEN_K": lambda args: args["K"] % args["BLOCK_SIZE_K"] == 0,
    }
)
@triton.jit(repr=_fused_moe_persistent_kernel_repr)
def _fused_moe_persistent_kernel(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    a_scale_ptr,
    b_scale_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    # Matrix dimensions
    N,
    K,
    num_valid_tokens,
    # The stride variables represent how much to increase the ptr by when
    # moving by 1 element in a particular dimension. E.g. `stride_am` is
    # how much to increase `a_ptr` by to get the element one row down
    # (A has M rows).
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_asm,
    stride_ask,
    stride_bse,
    stride_bsk,
    stride_bsn,
    # Block size for block-wise quantization
    group_n: tl.constexpr,
    group_k: tl.constexpr,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    EVEN_K: tl.constexpr,
    NUM_SMS: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
    use_fp8_w8a8: tl.constexpr,
    use_int8_w8a16: tl.constexpr,
    NUM_XCDS: tl.constexpr,
):
    """
    Implements the fused computation for a Mixture of Experts (MOE) using
    token and expert matrices.
    This is the persistent version of the fused_moe kernel.

    Key Parameters:
    - A: The input tensor representing tokens with shape (*, K), where '*' can
        be any shape representing batches and K is the feature dimension of
        each token.
    - B: The stacked MOE weight tensor with shape (E, N, K), where E is
        the number of experts, K is the input feature dimension, and N is
        the output feature dimension.
    - C: The output cache tensor with shape (M, topk, N), where M is the
        total number of tokens post padding, topk is the number of times
        each token is repeated, and N is the output feature dimension.
    - sorted_token_ids: A tensor containing the sorted indices of tokens,
        repeated topk times and arranged by the expert index they are
        assigned to.
    - expert_ids: A tensor containing the indices of the expert for each
        block. It determines which expert matrix from B should be used for
        each block in A.
    This kernel performs the multiplication of a token by its corresponding
    expert matrix as determined by `expert_ids`. The sorting of
    `sorted_token_ids` by expert index and padding ensures divisibility by
    BLOCK_SIZE_M, which is necessary to maintain consistency in block matrix
    multiplication across different blocks processed by the same expert.
    """
    # -----------------------------------------------------------
    # Simply compute how many iterations each persistent block needs to do
    start_pid = tl.program_id(axis=0)

    # Load tile-invariant runtime constant
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)

    num_pid_m = tl.cdiv(num_tokens_post_padded, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    tile_id = start_pid

    offs_k = tl.arange(0, BLOCK_SIZE_K)

    num_tiles = num_pid_m * num_pid_n

    # Compute how many tiles are outside the padding region
    num_valid_tiles = tl.cdiv((num_tiles - tile_id), NUM_SMS)

    for _ in range(0, num_valid_tiles):
        tile_id_remapped = remap_xcd(tile_id, num_tiles, NUM_XCDS)
        pid_m, pid_n = pid_grid(tile_id_remapped, num_pid_m, num_pid_n, GROUP_SIZE_M)

        offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
        offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
        token_mask = offs_token < num_valid_tokens

        off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
        if off_experts == -1:
            # -----------------------------------------------------------
            # Write back zeros to the output when the expert is not
            # in the current expert parallel rank.
            _write_zeros_to_output(
                c_ptr,
                stride_cm,
                stride_cn,
                pid_n,
                N,
                offs_token,
                token_mask,
                BLOCK_SIZE_M,
                BLOCK_SIZE_N,
                compute_type,
            )
        else:
            # Compute the A pointer
            a_ptrs = a_ptr + (
                offs_token[:, None] // top_k * stride_am + offs_k[None, :] * stride_ak
            )
            # Compute the B pointer
            offs_bn = (
                pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
            ) % N
            b_ptrs = (
                b_ptr
                + off_experts * stride_be
                + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
            )

            if use_int8_w8a16:
                b_scale_ptrs = (
                    b_scale_ptr
                    + off_experts * stride_bse
                    + offs_bn[None, :] * stride_bsn
                )
                b_scale = tl.load(b_scale_ptrs)

            if use_fp8_w8a8:
                if group_k > 0 and group_n > 0:
                    a_scale_ptrs = a_scale_ptr + (offs_token // top_k) * stride_asm
                    offs_bsn = offs_bn // group_n
                    b_scale_ptrs = (
                        b_scale_ptr + off_experts * stride_bse + offs_bsn * stride_bsn
                    )
                else:
                    a_scale = tl.load(a_scale_ptr)
                    b_scale = tl.load(b_scale_ptr + off_experts)

            accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

            for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
                # Load the next block of A and B, generate a mask by checking the
                # K dimension.
                if EVEN_K:
                    a = tl.load(a_ptrs, mask=token_mask[:, None], other=0.0)
                    b = tl.load(b_ptrs)
                else:
                    a = tl.load(
                        a_ptrs,
                        mask=token_mask[:, None]
                        & (offs_k[None, :] < K - k * BLOCK_SIZE_K),
                        other=0.0,
                    )
                    b = tl.load(
                        b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0
                    )
                # We accumulate along the K dimension.
                if use_int8_w8a16:
                    accumulator = tl.dot(a, b.to(compute_type), acc=accumulator)
                elif use_fp8_w8a8:
                    if group_k > 0 and group_n > 0:
                        k_start = k * BLOCK_SIZE_K
                        offs_ks = k_start // group_k
                        a_scale = tl.load(
                            a_scale_ptrs + offs_ks * stride_ask,
                            mask=token_mask,
                            other=0.0,
                        )
                        b_scale = tl.load(b_scale_ptrs + offs_ks * stride_bsk)
                        accumulator += (
                            tl.dot(a, b) * a_scale[:, None] * b_scale[None, :]
                        )
                    else:
                        accumulator = tl.dot(a, b, acc=accumulator)
                else:
                    accumulator += tl.dot(a, b)
                # Advance the ptrs to the next K block.
                a_ptrs += BLOCK_SIZE_K * stride_ak
                b_ptrs += BLOCK_SIZE_K * stride_bk

            if MUL_ROUTED_WEIGHT:
                moe_weight = tl.load(
                    topk_weights_ptr + offs_token, mask=token_mask, other=0
                )
                accumulator = accumulator * moe_weight[:, None]

            if use_int8_w8a16:
                accumulator = (accumulator * b_scale).to(compute_type)
            elif use_fp8_w8a8:
                if group_k > 0 and group_n > 0:
                    accumulator = accumulator.to(compute_type)
                else:
                    accumulator = (accumulator * a_scale * b_scale).to(compute_type)
            else:
                accumulator = accumulator.to(compute_type)
            # -----------------------------------------------------------
            # Write back the block of the output
            offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            c_ptrs = (
                c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
            )
            c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
            tl.store(c_ptrs, accumulator, mask=c_mask)

            # advance tile_id
            tile_id += NUM_SMS
