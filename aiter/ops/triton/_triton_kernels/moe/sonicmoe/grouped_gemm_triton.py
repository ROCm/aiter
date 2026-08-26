import torch
import triton
import triton.language as tl

from aiter.ops.triton.utils.sonicmoe_config_utils import (
    get_grouped_gemm_dw_config,
    get_grouped_gemm_fwd_config,
    split_launch_config,
)


def _get_fwd_autotune_configs():
    configs = []
    for BLOCK_M in [32, 64, 128]:
        for BLOCK_N in [32, 64, 128]:
            for BLOCK_K in [32, 64]:
                for num_warps in [4, 8]:
                    for num_stages in [2, 4]:
                        if BLOCK_M * BLOCK_N <= 16384 and BLOCK_M * BLOCK_K <= 8192:
                            configs.append(
                                triton.Config(
                                    {
                                        "BLOCK_M": BLOCK_M,
                                        "BLOCK_N": BLOCK_N,
                                        "BLOCK_K": BLOCK_K,
                                    },
                                    num_warps=num_warps,
                                    num_stages=num_stages,
                                )
                            )
    return configs


def _prune_fwd_configs(configs, nargs, **kw):
    K = kw.get("K", nargs.get("K", 9999))
    N = kw.get("N", nargs.get("N", 9999))
    pruned = []
    for c in configs:
        bk = c.kwargs["BLOCK_K"]
        bn = c.kwargs["BLOCK_N"]
        if bk <= triton.next_power_of_2(K) and bn <= triton.next_power_of_2(N):
            pruned.append(c)
    return pruned if pruned else configs


@triton.jit
def _grouped_gemm_kernel(
    A_ptr,
    B_ptr,
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
    stride_cm,
    stride_cn,
    stride_bias_e,
    stride_bias_n,
    N: tl.constexpr,
    K: tl.constexpr,
    E: tl.constexpr,
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
        acc += tl.dot(a, b)

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


_grouped_gemm_kernel_autotuned = triton.autotune(
    configs=_get_fwd_autotune_configs(),
    key=["N", "K", "E"],
    prune_configs_by={"early_config_prune": _prune_fwd_configs},
)(_grouped_gemm_kernel)


def _get_dw_autotune_configs():
    configs = []
    for BLOCK_K in [32, 64, 128]:
        for BLOCK_N in [32, 64, 128]:
            for BLOCK_T in [16, 32, 64]:
                for num_warps in [4, 8]:
                    if BLOCK_K * BLOCK_N <= 16384 and BLOCK_T * BLOCK_K <= 8192:
                        configs.append(
                            triton.Config(
                                {
                                    "BLOCK_K": BLOCK_K,
                                    "BLOCK_N": BLOCK_N,
                                    "BLOCK_T": BLOCK_T,
                                },
                                num_warps=num_warps,
                                num_stages=2,
                            )
                        )
    return configs


def _prune_dw_configs(configs, nargs, **kw):
    K = kw.get("K", nargs.get("K", 9999))
    N = kw.get("N", nargs.get("N", 9999))
    pruned = []
    for c in configs:
        bk = c.kwargs["BLOCK_K"]
        bn = c.kwargs["BLOCK_N"]
        if bk <= triton.next_power_of_2(K) and bn <= triton.next_power_of_2(N):
            pruned.append(c)
    return pruned if pruned else configs


@triton.jit
def _grouped_gemm_dw_kernel(
    A_ptr,
    B_ptr,
    C_ptr,
    cu_seqlens_ptr,
    A_idx_ptr,
    stride_ak,
    stride_am,
    stride_bm,
    stride_bn,
    stride_ce,
    stride_ck,
    stride_cn,
    N: tl.constexpr,
    K: tl.constexpr,
    E: tl.constexpr,
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
            + a_row_idx[:, None] * stride_ak
            + offs_k[None, :].to(tl.int64) * stride_am,
            mask=t_mask[:, None] & k_mask[None, :],
            other=0.0,
        ).to(a_dtype)

        b = tl.load(
            B_ptr
            + global_t[:, None].to(tl.int64) * stride_bm
            + offs_n[None, :].to(tl.int64) * stride_bn,
            mask=t_mask[:, None] & n_mask[None, :],
            other=0.0,
        ).to(a_dtype)

        acc += tl.dot(tl.trans(a), b)

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


_grouped_gemm_dw_kernel_autotuned = triton.autotune(
    configs=_get_dw_autotune_configs(),
    key=["N", "K", "E"],
    prune_configs_by={"early_config_prune": _prune_dw_configs},
)(_grouped_gemm_dw_kernel)


def _compute_grid_fwd(cu_seqlens_cpu, N, E, BLOCK_M, BLOCK_N):
    total_blocks = 0
    for e in range(E):
        m_e = cu_seqlens_cpu[e + 1].item() - cu_seqlens_cpu[e].item()
        total_blocks += triton.cdiv(m_e, BLOCK_M) * triton.cdiv(N, BLOCK_N)
    return total_blocks


def grouped_gemm(
    A: torch.Tensor,
    B: torch.Tensor,
    cu_seqlens: torch.Tensor,
    out: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    A_idx: torch.Tensor | None = None,
    scatter_idx: torch.Tensor | None = None,
    A_is_transposed: bool = False,
):
    if A_is_transposed and B.dim() == 2:
        return _grouped_gemm_dw(A, B, cu_seqlens, out, A_idx)

    E = B.shape[0]
    K_dim = B.shape[1]
    N = B.shape[2]

    TK = A.shape[0] if A_idx is None else cu_seqlens[-1].item()

    if out is None:
        out = torch.empty(TK, N, dtype=A.dtype, device=A.device)

    cu_seqlens_cpu = cu_seqlens.cpu()

    def grid(META):
        return (
            _compute_grid_fwd(cu_seqlens_cpu, N, E, META["BLOCK_M"], META["BLOCK_N"]),
        )

    fwd_cfg = get_grouped_gemm_fwd_config(N, K_dim, E)
    common = (
        A,
        B,
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
        out.stride(0),
        out.stride(1),
        bias.stride(0) if bias is not None else 0,
        bias.stride(1) if bias is not None else 0,
    )
    kwargs = {
        "N": N,
        "K": K_dim,
        "E": E,
        "HAS_BIAS": (bias is not None),
        "HAS_GATHER_IDX": (A_idx is not None),
        "HAS_SCATTER_IDX": (scatter_idx is not None),
    }
    if fwd_cfg is not None:
        constexprs, launch = split_launch_config(fwd_cfg)
        _grouped_gemm_kernel[grid](*common, **kwargs, **constexprs, **launch)
    else:
        _grouped_gemm_kernel_autotuned[grid](*common, **kwargs, GROUP_SIZE_M=8)
    return out


def _grouped_gemm_dw(
    A: torch.Tensor,
    B: torch.Tensor,
    cu_seqlens: torch.Tensor,
    out: torch.Tensor | None,
    A_idx: torch.Tensor | None,
):
    K_dim = A.shape[1]
    N = B.shape[1]
    E = cu_seqlens.shape[0] - 1

    if out is None:
        out = torch.empty(E, K_dim, N, dtype=A.dtype, device=A.device)

    def grid(META):
        num_k_blocks = triton.cdiv(K_dim, META["BLOCK_K"])
        num_n_blocks = triton.cdiv(N, META["BLOCK_N"])
        return (E * num_k_blocks * num_n_blocks,)

    common = (
        A,
        B,
        out,
        cu_seqlens,
        A_idx if A_idx is not None else cu_seqlens,
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(1),
        out.stride(0),
        out.stride(1),
        out.stride(2),
    )
    kwargs = {
        "N": N,
        "K": K_dim,
        "E": E,
        "HAS_GATHER_IDX": (A_idx is not None),
    }
    dw_cfg = get_grouped_gemm_dw_config(N, K_dim, E)
    if dw_cfg is not None:
        constexprs, launch = split_launch_config(dw_cfg)
        _grouped_gemm_dw_kernel[grid](*common, **kwargs, **constexprs, **launch)
    else:
        _grouped_gemm_dw_kernel_autotuned[grid](*common, **kwargs)
    return out
