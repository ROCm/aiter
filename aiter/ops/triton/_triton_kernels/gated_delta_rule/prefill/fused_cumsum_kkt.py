import torch
import triton
import triton.language as tl

from ..gated_delta_rule_utils import autotune_cache_kwargs, IS_AMD
from ..utils import prepare_chunk_indices
from ..utils.op import exp


@triton.jit
def safe_exp(x):
    return tl.exp(tl.where(x <= 0, x, float("-inf")))


@triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens"] is not None})
@triton.jit(do_not_specialize=["T"])
def _fused_cumsum_kkt_kernel(
    g_ptr,
    k_ptr,
    beta_ptr,
    g_cumsum_ptr,
    A_ptr,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H

    if IS_VARLEN:
        i_n = tl.load(chunk_indices + i_t * 2).to(tl.int32)
        i_t_local = tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos = tl.load(cu_seqlens + i_n).to(tl.int32)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T_seq = eos - bos
        i_t = i_t_local
    else:
        bos = i_b * T
        T_seq = T

    o_t = tl.arange(0, BT)

    p_g = tl.make_block_ptr(
        g_ptr + bos * H + i_h, (T_seq,), (H,), (i_t * BT,), (BT,), (0,)
    )
    b_g = tl.load(p_g, boundary_check=(0,)).to(tl.float32)
    b_g_cumsum = tl.cumsum(b_g, axis=0)
    p_g_out = tl.make_block_ptr(
        g_cumsum_ptr + bos * H + i_h, (T_seq,), (H,), (i_t * BT,), (BT,), (0,)
    )
    tl.store(p_g_out, b_g_cumsum.to(p_g_out.dtype.element_ty), boundary_check=(0,))

    p_beta = tl.make_block_ptr(
        beta_ptr + bos * H + i_h, (T_seq,), (H,), (i_t * BT,), (BT,), (0,)
    )
    b_beta = tl.load(p_beta, boundary_check=(0,)).to(tl.float32)

    p_k = tl.make_block_ptr(
        k_ptr + (bos * Hg + i_h // (H // Hg)) * K,
        (T_seq, K),
        (Hg * K, 1),
        (i_t * BT, 0),
        (BT, K),
        (1, 0),
    )
    b_k = tl.load(p_k, boundary_check=(0, 1)).to(tl.float32)

    b_A = tl.dot(b_k, tl.trans(b_k))
    b_g_diff = b_g_cumsum[:, None] - b_g_cumsum[None, :]
    b_A = b_A * safe_exp(b_g_diff) * b_beta[:, None]
    b_A = tl.where(o_t[:, None] > o_t[None, :], b_A, 0.0)

    p_A = tl.make_block_ptr(
        A_ptr + (bos * H + i_h) * BT,
        (T_seq, BT),
        (BT * H, 1),
        (i_t * BT, 0),
        (BT, BT),
        (1, 0),
    )
    tl.store(p_A, b_A.to(A_ptr.dtype.element_ty), boundary_check=(0, 1))


def fused_cumsum_kkt(
    g: torch.Tensor,
    k: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int = 64,
    cu_seqlens: torch.Tensor | None = None,
):
    """
    Fused cumsum + KKT.

    Args:
        g: [B, T, H]
        k: [B, T, Hg, K]
        beta: [B, T, H]

    Returns:
        g_cumsum: [B, T, H]
        A: [B, T, H, chunk_size], strictly lower triangular
    """
    B, T, H = g.shape
    Hg, K = k.shape[2], k.shape[3]

    if cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
        NT = len(chunk_indices)
    else:
        chunk_indices = None
        NT = triton.cdiv(T, chunk_size)

    g_cumsum = torch.empty(B, T, H, device=g.device, dtype=torch.float32)
    A = torch.empty(B, T, H, chunk_size, device=k.device, dtype=torch.float32)

    _fused_cumsum_kkt_kernel[(NT, B * H)](
        g,
        k,
        beta,
        g_cumsum,
        A,
        cu_seqlens,
        chunk_indices,
        T,
        H,
        Hg,
        K,
        chunk_size,
        num_warps=4,
        num_stages=3,
    )
    return g_cumsum, A


@triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens"] is not None})
@triton.autotune(
    configs=[
        triton.Config({"BK": BK}, num_warps=nw, num_stages=ns)
        for BK in [32, 64]
        for nw in [2, 4]
        for ns in ([2, 3] if IS_AMD else [2, 3, 4])
    ],
    key=["H", "K", "BT", "IS_VARLEN"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T"])
def fused_chunk_local_cumsum_scaled_dot_kkt_fwd_kernel(
    g,
    k,
    beta,
    g_cumsum_out,
    A_out,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = (
            tl.load(chunk_indices + i_t * 2).to(tl.int32),
            tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32),
        )
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int32),
            tl.load(cu_seqlens + i_n + 1).to(tl.int32),
        )
        T = eos - bos
    else:
        bos = i_b * T

    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T

    p_g = tl.make_block_ptr(
        g + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,)
    )
    b_g = tl.load(p_g, boundary_check=(0,)).to(tl.float32)
    b_g_cumsum = tl.cumsum(b_g, axis=0)

    p_go = tl.make_block_ptr(
        g_cumsum_out + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,)
    )
    tl.store(p_go, b_g_cumsum.to(p_go.dtype.element_ty), boundary_check=(0,))

    p_beta = tl.make_block_ptr(
        beta + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,)
    )
    b_beta = tl.load(p_beta, boundary_check=(0,))

    b_A = tl.zeros([BT, BT], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        p_k = tl.make_block_ptr(
            k + (bos * Hg + i_h // (H // Hg)) * K,
            (T, K),
            (Hg * K, 1),
            (i_t * BT, i_k * BK),
            (BT, BK),
            (1, 0),
        )
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_kb = b_k * b_beta[:, None]
        b_A += tl.dot(b_kb.to(b_k.dtype), tl.trans(b_k))

    b_g_diff = b_g_cumsum[:, None] - b_g_cumsum[None, :]
    m_A = (o_t[:, None] > o_t[None, :]) & (m_t[:, None] & m_t)
    b_A = tl.where(m_A, b_A * exp(b_g_diff), 0.0)

    p_A = tl.make_block_ptr(
        A_out + (bos * H + i_h) * BT,
        (T, BT),
        (BT * H, 1),
        (i_t * BT, 0),
        (BT, BT),
        (1, 0),
    )
    tl.store(p_A, b_A.to(A_out.dtype.element_ty), boundary_check=(0, 1))


def fused_chunk_local_cumsum_scaled_dot_kkt_fwd(
    k: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    g_output_dtype: torch.dtype = torch.float32,
    A_output_dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Fused cumsum + scaled dot KKT (optimized, with autotuning).

    Args:
        k: [B, T, Hg, K]
        beta: [B, T, H]
        g: [B, T, H], raw forget gate increments
        cu_seqlens: [N+1]
        chunk_size: int (must be 64)
        g_output_dtype: dtype for g_cumsum (default fp32)
        A_output_dtype: dtype for A_raw (default fp32)

    Returns:
        g_cumsum: [B, T, H]
        A_raw: [B, T, H, 64]
    """
    B, T, Hg, K = k.shape
    H = beta.shape[-1]
    BT = chunk_size

    if cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
        NT = len(chunk_indices)
    else:
        chunk_indices = None
        NT = triton.cdiv(T, BT)

    g_cumsum_out = torch.empty(B, T, H, device=g.device, dtype=g_output_dtype)
    A_out = torch.empty(B, T, H, BT, device=k.device, dtype=A_output_dtype)

    fused_chunk_local_cumsum_scaled_dot_kkt_fwd_kernel[(NT, B * H)](
        g,
        k,
        beta,
        g_cumsum_out,
        A_out,
        cu_seqlens,
        chunk_indices,
        T=T,
        H=H,
        Hg=Hg,
        K=K,
        BT=BT,
    )
    return g_cumsum_out, A_out
