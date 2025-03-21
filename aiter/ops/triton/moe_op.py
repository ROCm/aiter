import torch
import triton
import triton.language as tl
from typing import Any, Dict, Optional, List

from aiter.ops.triton.quant import dynamic_per_tensor_fp8_quant

#Source:
#MoE Kernel adapted from VLLM 

_PADDING_SIZE = 0 #TODO add support to set this

_MOE_A_QUANT_FUNC = dynamic_per_tensor_fp8_quant

def moe_set_padding_size(size: int):
    """
    Override padding size
    """
    global _PADDING_SIZE
    _PADDING_SIZE = size

def moe_set_quant_func(func):
    """
    Override 'A' matrix ie activations quantization function.
    Default function does dynamic quantization.
    """
    global _MOE_A_QUANT_FUNC
    _MOE_A_QUANT_FUNC = func


@triton.jit
def _write_zeros_to_output(c_ptr, stride_cm, stride_cn, pid_n, N, offs_token,
                          token_mask, BLOCK_SIZE_M, BLOCK_SIZE_N,
                          compute_type):
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=compute_type)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[
        None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


@triton.heuristics({
'GRID_MN':
    lambda args: triton.cdiv(args['EM'], args['BLOCK_SIZE_M']) * triton.cdiv(args['N'], args['BLOCK_SIZE_N'])
})
@triton.jit
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
        block_k_diviable: tl.constexpr,
        group_size: tl.constexpr,
        # Meta-parameters
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr,
        MUL_ROUTED_WEIGHT: tl.constexpr,
        top_k: tl.constexpr,
        compute_type: tl.constexpr,
        has_zp: tl.constexpr,
        use_int4_w4a16: tl.constexpr,
        use_int8_w8a16: tl.constexpr,
        GRID_MN: tl.constexpr,
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
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    NUM_XCDS: tl.constexpr = 8

    ## pid remapping on xcds
    # Number of pids per XCD in the new arrangement
    pids_per_xcd = (GRID_MN + NUM_XCDS - 1) // NUM_XCDS
    # When GRID_MN cannot divide NUM_XCDS, some xcds will have
    # pids_per_xcd pids, the other will have pids_per_xcd - 1 pids.
    # We calculate the number of xcds that have pids_per_xcd pids as
    # tall_xcds
    tall_xcds = GRID_MN % NUM_XCDS
    tall_xcds = NUM_XCDS if tall_xcds == 0 else tall_xcds
    # Compute current XCD and local pid within the XCD
    xcd = pid % NUM_XCDS
    local_pid = pid // NUM_XCDS
    # Calculate new pid based on the new grouping
    # Note that we need to consider the following two cases:
    # 1. the current pid is on a tall xcd
    # 2. the current pid is on a short xcd
    if xcd < tall_xcds:
        pid = xcd * pids_per_xcd + local_pid
    else:
        pid = tall_xcds * pids_per_xcd + (xcd - tall_xcds) * (pids_per_xcd - 1) + local_pid

    if GROUP_SIZE_M == 1:
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n
    else:
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + (pid % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return
    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(
        tl.int64)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
    token_mask = offs_token < num_valid_tokens

    off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    if off_experts == -1:
        # -----------------------------------------------------------
        # Write back zeros to the output when the expert is not
        # in the current expert parallel rank.
        _write_zeros_to_output(c_ptr, stride_cm, stride_cn, pid_n, N,
                              offs_token, token_mask, BLOCK_SIZE_M,
                              BLOCK_SIZE_N, compute_type)
        return

    offs_bn = (pid_n * BLOCK_SIZE_N +
               tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_token[:, None] // top_k * stride_am +
                      offs_k[None, :] * stride_ak)

    if use_int4_w4a16:
        b_ptrs = b_ptr + off_experts * stride_be + \
            (offs_k[:, None] // 2) * stride_bk + offs_bn[None, :] * \
                stride_bn
        b_shifter = (offs_k[:, None] % 2) * 4
    elif use_int8_w8a16:
        b_ptrs = b_ptr + off_experts * stride_be + \
            offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn

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

        if not block_k_diviable:
            k_mask = offs_k[:, None] < K - k * BLOCK_SIZE_K
            k_other = 0.0
        else:
            k_mask = None
            k_other = None

        a = tl.load(a_ptrs,
                    mask=token_mask[:, None] &
                    (offs_k[None, :] < K - k * BLOCK_SIZE_K),
                    other=0.0)
        b = tl.load(b_ptrs)
        if use_int4_w4a16:
            b = (b >> b_shifter) & 0xF

        b_scale_ptrs = b_scale_ptr + off_experts * stride_bse + \
            offs_bn[None, :] * stride_bsn + \
            ((offs_k[:, None] + BLOCK_SIZE_K * k) // group_size) * \
                stride_bsk
        b_scale = tl.load(b_scale_ptrs, mask=k_mask, other=k_other)
        b_scale = b_scale.to(tl.float32)

        if has_zp and use_int4_w4a16:
            offs_k_true = (offs_k[:, None] + BLOCK_SIZE_K * k) // group_size
            b_zp_ptrs = b_zp_ptr + off_experts * stride_bze + \
                (offs_bn[None, :] // 2) * stride_bzn + \
                offs_k_true * stride_bzk
            b_zp = tl.load(b_zp_ptrs, mask=k_mask, other=k_other)
            b_zp = ((b_zp >> b_zp_shifter) & 0xF)
            b_zp = b_zp.to(tl.float32)
        elif has_zp and use_int8_w8a16:
            offs_k_true = (offs_k[:, None] + BLOCK_SIZE_K * k) // group_size
            b_zp_ptrs = b_zp_ptr + off_experts * stride_bze + \
                offs_bn[None, :] * stride_bzn + \
                offs_k_true * stride_bzk
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
        moe_weight = tl.load(topk_weights_ptr + offs_token,
                             mask=token_mask,
                             other=0)
        accumulator = accumulator * moe_weight[:, None]

    accumulator = accumulator.to(compute_type)
    # -----------------------------------------------------------
    # Write back the block of the output
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[
        None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


@triton.heuristics({
'GRID_MN':
    lambda args: triton.cdiv(args['EM'], args['BLOCK_SIZE_M']) * triton.cdiv(args['N'], args['BLOCK_SIZE_N'])
})
@triton.jit
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
        MUL_ROUTED_WEIGHT: tl.constexpr,
        top_k: tl.constexpr,
        compute_type: tl.constexpr,
        use_fp8_w8a8: tl.constexpr,
        use_int8_w8a16: tl.constexpr,
        GRID_MN: tl.constexpr,
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
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    NUM_XCDS: tl.constexpr = 8

    ## pid remapping on xcds
    # Number of pids per XCD in the new arrangement
    pids_per_xcd = (GRID_MN + NUM_XCDS - 1) // NUM_XCDS
    # When GRID_MN cannot divide NUM_XCDS, some xcds will have
    # pids_per_xcd pids, the other will have pids_per_xcd - 1 pids.
    # We calculate the number of xcds that have pids_per_xcd pids as
    # tall_xcds
    tall_xcds = GRID_MN % NUM_XCDS
    tall_xcds = NUM_XCDS if tall_xcds == 0 else tall_xcds
    # Compute current XCD and local pid within the XCD
    xcd = pid % NUM_XCDS
    local_pid = pid // NUM_XCDS
    # Calculate new pid based on the new grouping
    # Note that we need to consider the following two cases:
    # 1. the current pid is on a tall xcd
    # 2. the current pid is on a short xcd
    if xcd < tall_xcds:
        pid = xcd * pids_per_xcd + local_pid
    else:
        pid = tall_xcds * pids_per_xcd + (xcd - tall_xcds) * (pids_per_xcd - 1) + local_pid

    if GROUP_SIZE_M == 1:
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n
    else:
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + (pid % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m
    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return
    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(
        tl.int64)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
    token_mask = offs_token < num_valid_tokens

    off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    if off_experts == -1:
        # -----------------------------------------------------------
        # Write back zeros to the output when the expert is not
        # in the current expert parallel rank.
        _write_zeros_to_output(c_ptr, stride_cm, stride_cn, pid_n, N,
                              offs_token, token_mask, BLOCK_SIZE_M,
                              BLOCK_SIZE_N, compute_type)
        return

    offs_bn = (pid_n * BLOCK_SIZE_N +
               tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_token[:, None] // top_k * stride_am +
                      offs_k[None, :] * stride_ak)

    b_ptrs = b_ptr + off_experts * stride_be + (offs_k[:, None] * stride_bk +
                                                offs_bn[None, :] * stride_bn)
    if use_int8_w8a16:
        b_scale_ptrs = b_scale_ptr + off_experts * stride_bse + offs_bn[
            None, :] * stride_bsn
        b_scale = tl.load(b_scale_ptrs)

    if use_fp8_w8a8:
        if group_k > 0 and group_n > 0:
            a_scale_ptrs = a_scale_ptr + (offs_token // top_k) * stride_asm
            offs_bsn = offs_bn // group_n
            b_scale_ptrs = (b_scale_ptr + off_experts * stride_bse +
                            offs_bsn * stride_bsn)
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
        a = tl.load(a_ptrs,
                    mask=token_mask[:, None] &
                    (offs_k[None, :] < K - k * BLOCK_SIZE_K),
                    other=0.0)
        b = tl.load(b_ptrs,
                    mask=offs_k[:, None] < K - k * BLOCK_SIZE_K,
                    other=0.0)
        # We accumulate along the K dimension.
        if use_int8_w8a16:
            accumulator = tl.dot(a, b.to(compute_type), acc=accumulator)
        elif use_fp8_w8a8:
            if group_k > 0 and group_n > 0:
                k_start = k * BLOCK_SIZE_K
                offs_ks = k_start // group_k
                a_scale = tl.load(a_scale_ptrs + offs_ks * stride_ask,
                                  mask=token_mask,
                                  other=0.0)
                b_scale = tl.load(b_scale_ptrs + offs_ks * stride_bsk)

                accumulator += tl.dot(a, b) * a_scale[:,
                                                      None] * b_scale[None, :]
            else:
                accumulator = tl.dot(a, b, acc=accumulator)
        else:
            accumulator += tl.dot(a, b)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token,
                             mask=token_mask,
                             other=0)
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
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[
        None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)



def fused_moe(A: torch.Tensor,
                            B: torch.Tensor,
                            C: torch.Tensor,
                            A_scale: Optional[torch.Tensor],
                            B_scale: Optional[torch.Tensor],
                            B_zp: Optional[torch.Tensor],
                            topk_weights: torch.Tensor,
                            topk_ids: torch.Tensor,
                            sorted_token_ids: torch.Tensor,
                            expert_ids: torch.Tensor,
                            num_tokens_post_padded: torch.Tensor,
                            mul_routed_weight: bool,
                            top_k: int,
                            config: Dict[str, Any],
                            compute_type: tl.dtype,
                            use_fp8_w8a8: bool,
                            use_int8_w8a16: bool,
                            use_int4_w4a16: bool,
                            block_shape: Optional[List[int]] = None) -> None:
    """
    #TODO: Add doc
    """
    assert topk_weights.stride(1) == 1
    assert sorted_token_ids.stride(0) == 1

    if use_fp8_w8a8:
        assert B_scale is not None
        if block_shape is None:
            output = torch.zeros(A.shape, device=A.device, dtype=torch.float8_e4m3fnuz)
            A_scale = torch.zeros(1, device=A.device, dtype=torch.float32)
            A, A_scale = _MOE_A_QUANT_FUNC(output, A, A_scale)
        else:
            #TODO: Add support for per token group quantization
            assert len(block_shape) == 2
            block_n, block_k = block_shape[0], block_shape[1]
            #A, A_scale = per_token_group_quant_fp8(A, block_k)
            assert triton.cdiv(A.shape[-1], block_k) == A_scale.shape[-1]
            assert triton.cdiv(B.shape[-2], block_n) == B_scale.shape[-2]
            assert triton.cdiv(B.shape[-1], block_k) == B_scale.shape[-1]
    elif use_int8_w8a16 or use_int4_w4a16:
        assert B_scale is not None
        assert block_shape is None or block_shape[0] == 0
    else:
        assert A_scale is None
        assert B_scale is None

    EM = sorted_token_ids.shape[0]
    if A.shape[0] < config["BLOCK_SIZE_M"]:
        # optimize for small batch_size.
        # We assume that top_ids of each token is unique, so
        # so num_valid_experts <= batch_size <= BLOCK_SIZE_M,
        # and we can skip some invalid blocks.
        EM = min(sorted_token_ids.shape[0],
                 A.shape[0] * top_k * config['BLOCK_SIZE_M'])
    grid = lambda META: (triton.cdiv(EM, META['BLOCK_SIZE_M']) * triton.cdiv(
        B.shape[1], META['BLOCK_SIZE_N']), )

    if (use_int8_w8a16 or use_int4_w4a16) and \
            block_shape is not None and block_shape[1] > 0:
        assert B_scale is not None and B_scale.ndim == 3
        assert B_zp is None or B_zp.ndim == 3

        _fused_moe_kernel_gptq_awq[grid](
            A,
            B,
            C,
            B_scale,
            B_zp,
            topk_weights,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            B.shape[1],
            A.shape[1],
            EM,
            topk_ids.numel(),
            A.stride(0),
            A.stride(1),
            B.stride(0),
            B.stride(2),
            B.stride(1),
            C.stride(1),
            C.stride(2),
            B_scale.stride(0),
            B_scale.stride(2),
            B_scale.stride(1),
            B_zp.stride(0) if B_zp is not None else 0,
            B_zp.stride(2) if B_zp is not None else 0,
            B_zp.stride(1) if B_zp is not None else 0,
            block_k_diviable=A.shape[1] % config["BLOCK_SIZE_K"] == 0,
            group_size=block_shape[1],
            MUL_ROUTED_WEIGHT=mul_routed_weight,
            top_k=top_k,
            compute_type=compute_type,
            has_zp=B_zp is not None,
            use_int4_w4a16=use_int4_w4a16,
            use_int8_w8a16=use_int8_w8a16,
            **config,
        )

    else:
        _fused_moe_kernel[grid](
            A,
            B,
            C,
            A_scale,
            B_scale,
            topk_weights,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            B.shape[1],
            A.shape[1] - _PADDING_SIZE,
            EM,
            topk_ids.numel(),
            A.stride(0),
            A.stride(1),
            B.stride(0),
            B.stride(2),
            B.stride(1),
            C.stride(1),
            C.stride(2),
            A_scale.stride(0)
            if A_scale is not None and A_scale.ndim == 2 else 0,
            A_scale.stride(1)
            if A_scale is not None and A_scale.ndim == 2 else 0,
            B_scale.stride(0)
            if B_scale is not None and B_scale.ndim >= 2 else 0,
            B_scale.stride(2)
            if B_scale is not None and B_scale.ndim == 3 else 0,
            B_scale.stride(1)
            if B_scale is not None and B_scale.ndim >= 2 else 0,
            0 if block_shape is None else block_shape[0],
            0 if block_shape is None else block_shape[1],
            MUL_ROUTED_WEIGHT=mul_routed_weight,
            top_k=top_k,
            compute_type=compute_type,
            use_fp8_w8a8=use_fp8_w8a8,
            use_int8_w8a16=use_int8_w8a16,
            **config,
        )
