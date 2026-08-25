import torch
import triton
import triton.language as tl

from .enums import LIBRARY_NAME
from .reduction_over_k_gather import token_gather_and_sum_varlen_K_triton


@torch.library.custom_op(f"{LIBRARY_NAME}::_router_forward_rocm", mutates_args={"o"})
def _router_forward(
    y: torch.Tensor,
    o: torch.Tensor,
    topk_scores: torch.Tensor,
    s_reverse_scatter_idx: torch.Tensor,
    num_activated_expert_per_token_offset: torch.Tensor,
    varlen_K_max: int,
    H: int,
    is_varlen_K: bool,
) -> None:
    token_gather_and_sum_varlen_K_triton(
        y,
        topk_scores,
        o,
        s_reverse_scatter_idx,
        num_activated_expert_per_token_offset,
        o.size(0),
        varlen_K_max,
        H,
        is_varlen_K,
    )


@torch.library.custom_op(
    f"{LIBRARY_NAME}::_softmax_topk_fwd_rocm",
    mutates_args={"topk_router_score", "topk_router_indices"},
)
def _topk_softmax_fwd(
    router_logits: torch.Tensor,
    topk_router_score: torch.Tensor,
    topk_router_indices: torch.Tensor,
    E: int,
    K: int,
    is_softmax_over_topk: bool,
    norm_topk_probs: bool,
) -> None:
    if is_softmax_over_topk:
        topk_results = router_logits.topk(K, dim=-1)
        vals = topk_results.values.softmax(dim=-1, dtype=torch.float32)
        topk_router_score.copy_(vals.to(topk_router_score.dtype))
        topk_router_indices.copy_(topk_results.indices.to(topk_router_indices.dtype))
    else:
        probs = router_logits.softmax(dim=-1, dtype=torch.float32)
        topk_results = probs.topk(K, dim=-1)
        vals = topk_results.values
        if norm_topk_probs:
            vals = vals / vals.sum(dim=-1, keepdim=True)
        topk_router_score.copy_(vals.to(topk_router_score.dtype))
        topk_router_indices.copy_(topk_results.indices.to(topk_router_indices.dtype))


@triton.jit
def _softmax_over_topk_bwd_kernel(
    dlogits_ptr,
    dlogits_full_ptr,
    score_ptr,
    dscore_ptr,
    idx_ptr,
    stride_dm: tl.constexpr,
    stride_dn: tl.constexpr,
    stride_sm: tl.constexpr,
    stride_sn: tl.constexpr,
    stride_gm: tl.constexpr,
    stride_gk: tl.constexpr,
    stride_im: tl.constexpr,
    stride_ik: tl.constexpr,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
    dlogits_is_none: tl.constexpr,
):
    row = tl.program_id(axis=0)
    k_offs = tl.arange(0, BLOCK_K)
    k_mask = k_offs < K

    idx = tl.load(
        idx_ptr + row * stride_im + k_offs * stride_ik, mask=k_mask, other=0
    ).to(tl.int32)
    s_sel = tl.load(
        score_ptr + row * stride_sm + k_offs * stride_sn, mask=k_mask, other=0
    ).to(tl.float32)
    g_sel = tl.load(
        dscore_ptr + row * stride_gm + k_offs * stride_gk, mask=k_mask, other=0
    ).to(tl.float32)

    dot = tl.sum(g_sel * s_sel, axis=0)
    add_vals = s_sel * (g_sel - dot)

    indices = row * stride_dm + idx * stride_dn
    if not dlogits_is_none:
        add_vals += tl.load(dlogits_ptr + indices, mask=k_mask)
    tl.store(dlogits_full_ptr + indices, add_vals, mask=k_mask)


@triton.jit
def _topk_over_softmax_bwd_kernel(
    logits_ptr,
    dlogits_ptr,
    dscore_ptr,
    idx_ptr,
    score_ptr,
    stride_lm: tl.constexpr,
    stride_le: tl.constexpr,
    stride_dm: tl.constexpr,
    stride_dn: tl.constexpr,
    stride_sm: tl.constexpr,
    stride_sn: tl.constexpr,
    stride_im: tl.constexpr,
    stride_ik: tl.constexpr,
    stride_scm: tl.constexpr,
    stride_scn: tl.constexpr,
    E: tl.constexpr,
    K: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_K: tl.constexpr,
    norm_topk_probs: tl.constexpr,
):
    row = tl.program_id(axis=0)

    e_offs = tl.arange(0, BLOCK_E)
    e_mask = e_offs < E
    logits = tl.load(
        logits_ptr + row * stride_lm + e_offs * stride_le,
        mask=e_mask,
        other=-float("inf"),
    ).to(tl.float32)
    row_max = tl.max(logits, axis=0)
    exp_vals = tl.exp(logits - row_max)
    row_sum = tl.sum(exp_vals, axis=0)
    p = exp_vals / row_sum

    k_offs = tl.arange(0, BLOCK_K)
    k_mask = k_offs < K
    idx = tl.load(
        idx_ptr + row * stride_im + k_offs * stride_ik, mask=k_mask, other=0
    ).to(tl.int32)
    g_sel = tl.load(
        dscore_ptr + row * stride_sm + k_offs * stride_sn, mask=k_mask, other=0
    ).to(tl.float32)

    sel_logits = tl.load(
        logits_ptr + row * stride_lm + idx * stride_le, mask=k_mask, other=-float("inf")
    ).to(tl.float32)
    p_sel = tl.exp(sel_logits - row_max) / row_sum

    if norm_topk_probs:
        scores = tl.load(
            score_ptr + row * stride_scm + k_offs * stride_scn, mask=k_mask, other=0
        ).to(tl.float32)
        dot_s = tl.sum(g_sel * scores, axis=0)
        S = tl.sum(p_sel, axis=0)
        dp_sel = (g_sel - dot_s) / S
    else:
        dp_sel = g_sel

    dot = tl.sum(dp_sel * p_sel, axis=0)

    dp = tl.zeros([BLOCK_E], dtype=tl.float32)
    for k_iter in tl.static_range(K):
        cur_dp = tl.sum(tl.where(k_offs == k_iter, dp_sel, 0.0))
        cur_idx = tl.sum(tl.where(k_offs == k_iter, idx, 0))
        dp = tl.where(e_offs == cur_idx, cur_dp, dp)

    dlogits = p * (dp - dot)
    tl.store(dlogits_ptr + row * stride_dm + e_offs * stride_dn, dlogits, mask=e_mask)


@torch.library.custom_op(
    f"{LIBRARY_NAME}::_topk_softmax_bwd_rocm", mutates_args={"dlogits_full"}
)
def _topk_softmax_bwd(
    router_logits: torch.Tensor,
    dlogits_full: torch.Tensor,
    dlogits: torch.Tensor | None,
    dtopk_score: torch.Tensor,
    topk_router_score: torch.Tensor,
    topk_router_indices: torch.Tensor,
    E: int,
    K: int,
    is_softmax_over_topk: bool = True,
    norm_topk_probs: bool = False,
) -> None:
    T = dtopk_score.shape[0]

    if is_softmax_over_topk:
        _softmax_over_topk_bwd_kernel[T,](
            dlogits,
            dlogits_full,
            topk_router_score,
            dtopk_score,
            topk_router_indices,
            dlogits_full.stride(0),
            dlogits_full.stride(1),
            topk_router_score.stride(0),
            topk_router_score.stride(1),
            dtopk_score.stride(0),
            dtopk_score.stride(1),
            topk_router_indices.stride(0),
            topk_router_indices.stride(1),
            K,
            triton.next_power_of_2(K),
            (dlogits is None),
        )
    else:
        _topk_over_softmax_bwd_kernel[T,](
            router_logits,
            dlogits_full,
            dtopk_score,
            topk_router_indices,
            topk_router_score,
            router_logits.stride(0),
            router_logits.stride(1),
            dlogits_full.stride(0),
            dlogits_full.stride(1),
            dtopk_score.stride(0),
            dtopk_score.stride(1),
            topk_router_indices.stride(0),
            topk_router_indices.stride(1),
            topk_router_score.stride(0),
            topk_router_score.stride(1),
            E,
            K,
            triton.next_power_of_2(E),
            triton.next_power_of_2(K),
            norm_topk_probs,
        )
