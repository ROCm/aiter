import os

import torch
import triton
import triton.language as tl

from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr
from aiter.ops.triton.utils.sonicmoe_config_utils import (
    get_grouped_gemm_dw_config,
    get_grouped_gemm_fwd_config,
    split_launch_config,
)

_grouped_gemm_repr = make_kernel_repr(
    "_grouped_gemm_kernel",
    [
        "N",
        "K",
        "E",
        "SCALE_BLOCK_SIZE",
        "BLOCKWISE_FP8",
        "BLOCK_M",
        "BLOCK_N",
        "BLOCK_K",
        "GROUP_SIZE_M",
        "HAS_BIAS",
        "HAS_GATHER_IDX",
        "HAS_SCATTER_IDX",
    ],
)
_grouped_gemm_dw_repr = make_kernel_repr(
    "_grouped_gemm_dw_kernel",
    [
        "N",
        "K",
        "E",
        "SCALE_BLOCK_SIZE",
        "BLOCKWISE_FP8",
        "BLOCK_K",
        "BLOCK_N",
        "BLOCK_T",
        "HAS_GATHER_IDX",
    ],
)


def _local_tensor(tensor: torch.Tensor | None) -> torch.Tensor | None:
    if tensor is not None and hasattr(tensor, "to_local"):
        return tensor.to_local()
    return tensor


def _use_qwen3_tuned_configs() -> bool:
    return os.environ.get("SONIC_MOE_USE_QWEN3_TUNED_GEMM", "0") == "1"


@triton.jit(repr=_grouped_gemm_repr)
def _grouped_gemm_kernel(
    A_ptr,
    B_ptr,
    A_scale_ptr,
    B_scale_ptr,
    C_ptr,
    cu_seqlens_ptr,
    bias_ptr,
    A_idx_ptr,
    scatter_idx_ptr,
    stride_ak,
    stride_am,
    stride_be,
    stride_bk,
    stride_bn,
    stride_asm,
    stride_ask,
    stride_bse,
    stride_bsk,
    stride_bsn,
    stride_cm,
    stride_cn,
    stride_bias_e,
    stride_bias_n,
    N: tl.constexpr,
    K: tl.constexpr,
    E: tl.constexpr,
    SCALE_BLOCK_SIZE: tl.constexpr,
    BLOCKWISE_FP8: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_GATHER_IDX: tl.constexpr,
    HAS_SCATTER_IDX: tl.constexpr,
):
    pid = tl.program_id(0)

    cumulative_blocks = 0
    expert_id = 0
    expert_start = 0
    expert_end = 0

    for e in range(E):
        s = tl.load(cu_seqlens_ptr + e).to(tl.int32)
        f = tl.load(cu_seqlens_ptr + e + 1).to(tl.int32)
        m_e = f - s
        blocks_m_e = tl.cdiv(m_e, BLOCK_M)
        blocks_this_expert = blocks_m_e * tl.cdiv(N, BLOCK_N)
        if pid >= cumulative_blocks and pid < cumulative_blocks + blocks_this_expert:
            expert_id = e
            expert_start = s
            expert_end = f
        cumulative_blocks += blocks_this_expert

    # Launching an upper bound avoids copying cu_seqlens to the CPU just to
    # calculate the exact grid size.
    if pid >= cumulative_blocks:
        return

    local_pid = pid
    for e in range(E):
        if e < expert_id:
            s = tl.load(cu_seqlens_ptr + e).to(tl.int32)
            f = tl.load(cu_seqlens_ptr + e + 1).to(tl.int32)
            m_e = f - s
            local_pid -= tl.cdiv(m_e, BLOCK_M) * tl.cdiv(N, BLOCK_N)

    M_expert = expert_end - expert_start
    num_pid_m = tl.cdiv(M_expert, BLOCK_M)
    num_pid_n: tl.constexpr = tl.cdiv(N, BLOCK_N)

    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = local_pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (local_pid % num_pid_in_group) % group_size_m
    pid_n = (local_pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    m_mask = offs_m < M_expert
    global_m = expert_start + offs_m

    if HAS_GATHER_IDX:
        a_row_idx = tl.load(A_idx_ptr + global_m, mask=m_mask, other=0).to(tl.int64)
    else:
        a_row_idx = global_m.to(tl.int64)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    expert_id_i64 = expert_id.to(tl.int64)
    a_dtype = A_ptr.dtype.element_ty

    for k_start in range(0, K, BLOCK_K):
        k_offs = k_start + offs_k
        k_mask = k_offs < K
        a = tl.load(
            A_ptr
            + a_row_idx[:, None] * stride_ak
            + k_offs[None, :].to(tl.int64) * stride_am,
            mask=m_mask[:, None] & k_mask[None, :],
            other=0.0,
        ).to(a_dtype)
        b = tl.load(
            B_ptr
            + expert_id_i64 * stride_be
            + k_offs[:, None].to(tl.int64) * stride_bk
            + offs_n[None, :].to(tl.int64) * stride_bn,
            mask=k_mask[:, None] & (offs_n[None, :] < N),
            other=0.0,
        ).to(a_dtype)
        dot = tl.dot(a, b)
        if BLOCKWISE_FP8:
            scale_k = k_start // SCALE_BLOCK_SIZE
            a_scale = tl.load(
                A_scale_ptr + a_row_idx * stride_asm + scale_k * stride_ask,
                mask=m_mask,
                other=0.0,
            )
            b_scale = tl.load(
                B_scale_ptr
                + expert_id_i64 * stride_bse
                + scale_k * stride_bsk
                + (offs_n // SCALE_BLOCK_SIZE).to(tl.int64) * stride_bsn,
                mask=offs_n < N,
                other=0.0,
            )
            dot *= a_scale[:, None] * b_scale[None, :]
        acc += dot

    if HAS_BIAS:
        bias_vals = tl.load(
            bias_ptr
            + expert_id_i64 * stride_bias_e
            + offs_n.to(tl.int64) * stride_bias_n,
            mask=offs_n < N,
            other=0.0,
        )
        acc += bias_vals[None, :]

    c = acc.to(C_ptr.dtype.element_ty)

    if HAS_SCATTER_IDX:
        c_row_idx = tl.load(scatter_idx_ptr + global_m, mask=m_mask, other=0).to(
            tl.int64
        )
    else:
        c_row_idx = global_m.to(tl.int64)

    c_ptrs = (
        C_ptr
        + c_row_idx[:, None] * stride_cm
        + offs_n[None, :].to(tl.int64) * stride_cn
    )
    c_mask = m_mask[:, None] & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


@triton.jit(repr=_grouped_gemm_dw_repr)
def _grouped_gemm_dw_kernel(
    A_ptr,
    B_ptr,
    A_scale_ptr,
    B_scale_ptr,
    C_ptr,
    cu_seqlens_ptr,
    A_idx_ptr,
    stride_ak,
    stride_am,
    stride_bm,
    stride_bn,
    stride_ast,
    stride_ask,
    stride_bst,
    stride_bsn,
    stride_ce,
    stride_ck,
    stride_cn,
    N: tl.constexpr,
    K: tl.constexpr,
    E: tl.constexpr,
    SCALE_BLOCK_SIZE: tl.constexpr,
    BLOCKWISE_FP8: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_T: tl.constexpr,
    HAS_GATHER_IDX: tl.constexpr,
):
    pid = tl.program_id(0)
    num_k_blocks: tl.constexpr = tl.cdiv(K, BLOCK_K)
    num_n_blocks: tl.constexpr = tl.cdiv(N, BLOCK_N)
    blocks_per_expert: tl.constexpr = num_k_blocks * num_n_blocks

    expert_id = pid // blocks_per_expert
    local_pid = pid % blocks_per_expert
    pid_k = local_pid // num_n_blocks
    pid_n = local_pid % num_n_blocks

    expert_start = tl.load(cu_seqlens_ptr + expert_id).to(tl.int32)
    expert_end = tl.load(cu_seqlens_ptr + expert_id + 1).to(tl.int32)
    M_expert = expert_end - expert_start
    scale_expert_start = 0
    if BLOCKWISE_FP8:
        for e in range(E):
            if e < expert_id:
                e_start = tl.load(cu_seqlens_ptr + e).to(tl.int32)
                e_end = tl.load(cu_seqlens_ptr + e + 1).to(tl.int32)
                scale_expert_start += tl.cdiv(e_end - e_start, SCALE_BLOCK_SIZE)

    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_t = tl.arange(0, BLOCK_T)

    k_mask = offs_k < K
    n_mask = offs_n < N

    acc = tl.zeros((BLOCK_K, BLOCK_N), dtype=tl.float32)
    a_dtype = A_ptr.dtype.element_ty

    for t_start in range(0, M_expert, BLOCK_T):
        t_offs = t_start + offs_t
        t_mask = t_offs < M_expert
        global_t = expert_start + t_offs

        if HAS_GATHER_IDX:
            a_row_idx = tl.load(A_idx_ptr + global_t, mask=t_mask, other=0).to(tl.int64)
        else:
            a_row_idx = global_t.to(tl.int64)

        a = tl.load(
            A_ptr
            + offs_k[:, None].to(tl.int64) * stride_am
            + a_row_idx[None, :] * stride_ak,
            mask=k_mask[:, None] & t_mask[None, :],
            other=0.0,
        ).to(a_dtype)

        b = tl.load(
            B_ptr
            + global_t[:, None].to(tl.int64) * stride_bm
            + offs_n[None, :].to(tl.int64) * stride_bn,
            mask=t_mask[:, None] & n_mask[None, :],
            other=0.0,
        ).to(a_dtype)

        # Load A directly as [K, T]. Keeping the reduction dimension contiguous
        # in the dot operands avoids the very slow FP8 lowering of tl.trans(a).
        dot = tl.dot(a, b)
        if BLOCKWISE_FP8:
            scale_t = scale_expert_start + t_start // SCALE_BLOCK_SIZE
            a_scale = tl.load(
                A_scale_ptr + scale_t * stride_ast + offs_k.to(tl.int64) * stride_ask,
                mask=k_mask,
                other=0.0,
            )
            b_scale = tl.load(
                B_scale_ptr + scale_t * stride_bst + offs_n.to(tl.int64) * stride_bsn,
                mask=n_mask,
                other=0.0,
            )
            dot *= a_scale[:, None] * b_scale[None, :]
        acc += dot

    c = acc.to(C_ptr.dtype.element_ty)
    expert_id_i64 = expert_id.to(tl.int64)
    c_ptrs = (
        C_ptr
        + expert_id_i64 * stride_ce
        + offs_k[:, None].to(tl.int64) * stride_ck
        + offs_n[None, :].to(tl.int64) * stride_cn
    )
    c_mask = k_mask[:, None] & n_mask[None, :]
    tl.store(c_ptrs, c, mask=c_mask)


def grouped_gemm(
    A: torch.Tensor,
    B: torch.Tensor,
    cu_seqlens: torch.Tensor,
    out: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    A_idx: torch.Tensor | None = None,
    scatter_idx: torch.Tensor | None = None,
    A_is_transposed: bool = False,
    B_is_transposed: bool = False,
    A_scale: torch.Tensor | None = None,
    B_scale: torch.Tensor | None = None,
    block_size: int = 128,
    out_dtype: torch.dtype | None = None,
):
    """Run grouped GEMM, optionally with 1x128 activation and 128x128 weight scales."""
    if A_is_transposed:
        if B_is_transposed:
            raise ValueError("a grouped wgrad does not support a transposed B")
        if bias is not None:
            raise ValueError("bias is invalid for a grouped wgrad")
        if scatter_idx is not None:
            raise ValueError("scatter_idx is invalid for a grouped wgrad")

    local_out = _local_tensor(out)
    local_b = _local_tensor(B)
    triton_b = local_b.transpose(1, 2) if B_is_transposed else local_b
    local_b_scale = _local_tensor(B_scale)
    triton_b_scale = (
        local_b_scale.transpose(1, 2)
        if B_is_transposed and local_b_scale is not None
        else local_b_scale
    )
    result = _grouped_gemm_triton(
        _local_tensor(A),
        triton_b,
        _local_tensor(cu_seqlens),
        local_out,
        _local_tensor(bias),
        _local_tensor(A_idx),
        _local_tensor(scatter_idx),
        A_is_transposed,
        _local_tensor(A_scale),
        triton_b_scale,
        block_size,
        out_dtype,
    )
    return out if out is not None else result


def _grouped_gemm_triton(
    A: torch.Tensor,
    B: torch.Tensor,
    cu_seqlens: torch.Tensor,
    out: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    A_idx: torch.Tensor | None = None,
    scatter_idx: torch.Tensor | None = None,
    A_is_transposed: bool = False,
    A_scale: torch.Tensor | None = None,
    B_scale: torch.Tensor | None = None,
    block_size: int = 128,
    out_dtype: torch.dtype | None = None,
):
    if A_is_transposed and B.dim() == 2:
        return _grouped_gemm_dw(
            A, B, cu_seqlens, out, A_idx, A_scale, B_scale, block_size, out_dtype
        )

    E = B.shape[0]
    K_dim = B.shape[1]
    N = B.shape[2]

    TK = A.shape[0] if A_idx is None else A_idx.numel()

    if out is None:
        out = torch.empty(
            TK,
            N,
            dtype=out_dtype if out_dtype is not None else A.dtype,
            device=A.device,
        )

    blockwise_fp8 = A_scale is not None
    if blockwise_fp8:
        if B_scale is None:
            raise ValueError("B_scale is required when A_scale is provided")
        expected_a_scale = (A.shape[0], triton.cdiv(K_dim, block_size))
        expected_b_scale = (
            E,
            triton.cdiv(K_dim, block_size),
            triton.cdiv(N, block_size),
        )
        if tuple(A_scale.shape) != expected_a_scale:
            raise ValueError(
                f"A_scale must have shape {expected_a_scale}, got {tuple(A_scale.shape)}"
            )
        if tuple(B_scale.shape) != expected_b_scale:
            raise ValueError(
                f"B_scale must have shape {expected_b_scale}, got {tuple(B_scale.shape)}"
            )

    def grid(META):
        max_m_blocks = triton.cdiv(TK, META["BLOCK_M"]) + E - 1
        return (max_m_blocks * triton.cdiv(N, META["BLOCK_N"]),)

    launch_args = (
        A,
        B,
        A_scale if A_scale is not None else A,
        B_scale if B_scale is not None else B,
        out,
        cu_seqlens,
        bias if bias is not None else A,
        A_idx if A_idx is not None else cu_seqlens,
        scatter_idx if scatter_idx is not None else cu_seqlens,
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(1),
        B.stride(2),
        A_scale.stride(0) if A_scale is not None else 0,
        A_scale.stride(1) if A_scale is not None else 0,
        B_scale.stride(0) if B_scale is not None else 0,
        B_scale.stride(1) if B_scale is not None else 0,
        B_scale.stride(2) if B_scale is not None else 0,
        out.stride(0),
        out.stride(1),
        bias.stride(0) if bias is not None else 0,
        bias.stride(1) if bias is not None else 0,
    )
    launch_meta = {
        "N": N,
        "K": K_dim,
        "E": E,
        "SCALE_BLOCK_SIZE": block_size,
        "BLOCKWISE_FP8": blockwise_fp8,
        "HAS_BIAS": (bias is not None),
        "HAS_GATHER_IDX": (A_idx is not None),
        "HAS_SCATTER_IDX": (scatter_idx is not None),
    }
    fwd_cfg = get_grouped_gemm_fwd_config(
        N, K_dim, E, A_idx is not None, _use_qwen3_tuned_configs()
    )
    constexprs, launch = split_launch_config(fwd_cfg)
    _grouped_gemm_kernel[grid](*launch_args, **launch_meta, **constexprs, **launch)
    return out


def _grouped_gemm_dw(
    A: torch.Tensor,
    B: torch.Tensor,
    cu_seqlens: torch.Tensor,
    out: torch.Tensor | None,
    A_idx: torch.Tensor | None,
    A_scale: torch.Tensor | None = None,
    B_scale: torch.Tensor | None = None,
    block_size: int = 128,
    out_dtype: torch.dtype | None = None,
):
    K_dim = A.shape[1]
    N = B.shape[1]
    E = cu_seqlens.shape[0] - 1

    if out is None:
        out = torch.empty(
            E,
            K_dim,
            N,
            dtype=out_dtype if out_dtype is not None else A.dtype,
            device=A.device,
        )

    blockwise_fp8 = A_scale is not None
    if blockwise_fp8:
        if B_scale is None:
            raise ValueError("B_scale is required when A_scale is provided")
        min_scale_rows = triton.cdiv(A.shape[0], block_size)
        if (
            A_scale.dim() != 2
            or A_scale.shape[0] < min_scale_rows
            or A_scale.shape[1] != K_dim
        ):
            raise ValueError(
                f"A_scale must have shape [>={min_scale_rows}, {K_dim}], "
                f"got {tuple(A_scale.shape)}"
            )
        if (
            B_scale.dim() != 2
            or B_scale.shape[0] < min_scale_rows
            or B_scale.shape[1] != N
        ):
            raise ValueError(
                f"B_scale must have shape [>={min_scale_rows}, {N}], "
                f"got {tuple(B_scale.shape)}"
            )
        if A_idx is not None:
            raise ValueError("blockwise FP8 grouped wgrad does not support A_idx")

    def grid(META):
        num_k_blocks = triton.cdiv(K_dim, META["BLOCK_K"])
        num_n_blocks = triton.cdiv(N, META["BLOCK_N"])
        return (E * num_k_blocks * num_n_blocks,)

    launch_args = (
        A,
        B,
        A_scale if A_scale is not None else A,
        B_scale if B_scale is not None else B,
        out,
        cu_seqlens,
        A_idx if A_idx is not None else cu_seqlens,
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(1),
        A_scale.stride(0) if A_scale is not None else 0,
        A_scale.stride(1) if A_scale is not None else 0,
        B_scale.stride(0) if B_scale is not None else 0,
        B_scale.stride(1) if B_scale is not None else 0,
        out.stride(0),
        out.stride(1),
        out.stride(2),
    )
    launch_meta = {
        "N": N,
        "K": K_dim,
        "E": E,
        "SCALE_BLOCK_SIZE": block_size,
        "BLOCKWISE_FP8": blockwise_fp8,
        "HAS_GATHER_IDX": A_idx is not None,
    }
    dw_cfg = get_grouped_gemm_dw_config(
        N, K_dim, E, A_idx is not None, _use_qwen3_tuned_configs()
    )
    constexprs, launch = split_launch_config(dw_cfg)
    _grouped_gemm_dw_kernel[grid](*launch_args, **launch_meta, **constexprs, **launch)
    return out
