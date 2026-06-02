from __future__ import annotations

import torch
import triton
import triton.language as tl

from aiter.ops.triton.moe.moe_routing.bitmatrix import Bitmatrix


@triton.jit
def _grouped_topk(
    X,                      # router logits [n_rows, n_expts_tot] (bf16/fp32)
    stride_xm,
    ExpertGroup,            # int32 [n_expts_tot] expert→group_id
    Yv,                     # [n_rows, N_EXPTS_ACT_PAD] selected weights
    Yi,                     # [n_rows, N_EXPTS_ACT_PAD] selected expert ids (int16)
    stride_ym,
    Bits,                   # bitmatrix data
    stride_rm,
    stride_rn,
    n_rows,
    n_expts_tot,
    S,                      # bitmatrix scratchpad — must memset to 0
    BLOCK_S: tl.constexpr,
    s_blocks,
    SP,                     # bitmatrix partials — must memset to 0
    BLOCK_SP: tl.constexpr,
    sp_blocks,
    sp_size,
    BLOCK_M: tl.constexpr,
    N_EXPTS_PAD: tl.constexpr,
    BLOCK_N: tl.constexpr,
    N_EXPTS_ACT: tl.constexpr,
    N_EXPTS_ACT_PAD: tl.constexpr,
    NUM_EXPERT_GROUP: tl.constexpr,
    TOPK_GROUP: tl.constexpr,
    Bias=None,
    SCORE_MODE: tl.constexpr = "softmax",
    HAS_BIAS: tl.constexpr = False,
    APPLY_RENORM: tl.constexpr = False,
    ROUTED_SCALING: tl.constexpr = 1.0,
    N_SHARED: tl.constexpr = 0,
    SHARED_SCORE: tl.constexpr = 1.0,
):
    pid = tl.program_id(0)

    # -- Memset bitmatrix scratchpads (same idiom as _topk / _hash_routing).
    if pid < s_blocks:
        tl.store(
            S + BLOCK_S * pid + tl.arange(0, BLOCK_S),
            tl.zeros([BLOCK_S], tl.int32),
        )
    elif pid < s_blocks + sp_blocks:
        offs = BLOCK_SP * (pid - s_blocks) + tl.arange(0, BLOCK_SP)
        tl.store(SP + offs, tl.zeros([BLOCK_SP], tl.int32), mask=offs < sp_size)

    if pid * BLOCK_M >= n_rows:
        return

    tl.static_assert(BLOCK_N % 32 == 0)
    tl.static_assert(
        N_EXPTS_PAD == BLOCK_N,
        "DeepSeek-class envelope: BLOCK_N must equal N_EXPTS_PAD (single-block).",
    )

    x_dtype: tl.constexpr = X.dtype.element_ty

    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < n_rows
    offs_n = tl.arange(0, BLOCK_N)
    mask_n = offs_n < n_expts_tot

    # -- 1. Load logits.
    X_ptrs = X + offs_m[:, None] * stride_xm + offs_n[None, :]
    x = tl.load(X_ptrs, mask=mask_m[:, None] & mask_n[None, :], other=0.0)

    # -- 2. Score transform.
    if SCORE_MODE == "softmax":
        # Numerically-stable row softmax with masked-out lanes set to -inf.
        x_f = tl.where(mask_n[None, :], x.to(tl.float32), float("-inf"))
        x_max = tl.max(x_f, axis=1, keep_dims=True)
        x_e = tl.exp(x_f - x_max)
        x_e = tl.where(mask_n[None, :], x_e, 0.0)
        scores = x_e / (tl.sum(x_e, axis=1, keep_dims=True) + 1e-30)
    elif SCORE_MODE == "sigmoid":
        scores = 1.0 / (1.0 + tl.exp(-x.to(tl.float32)))
    elif SCORE_MODE == "sqrtsoftplus":
        x_f = x.to(tl.float32)
        sp = tl.maximum(x_f, 0.0) + tl.log(1.0 + tl.exp(-tl.abs(x_f)))
        scores = tl.sqrt(sp)
    else:
        scores = x.to(tl.float32)

    # Pad-lane safety: invalid columns must lose every comparison.
    scores = tl.where(mask_n[None, :], scores, float("-inf"))

    # -- 3. Bias-augmented choice scores. Weights are gathered later from the
    #       untouched ``scores`` (matches biased_grouped_topk_torch +
    #       FusedMoE.select_experts sigmoid path: select on s+b, return s).
    if HAS_BIAS:
        b = tl.load(Bias + offs_n, mask=mask_n, other=0.0).to(tl.float32)
        scores_for_choice = scores + b[None, :]
    else:
        scores_for_choice = scores

    # -- 4. Per-group reduction over arbitrary expert→group mapping.
    gid = tl.load(ExpertGroup + offs_n, mask=mask_n, other=0).to(tl.int32)
    g_arange = tl.arange(0, NUM_EXPERT_GROUP)
    gid_eq = gid[:, None] == g_arange[None, :]   # [BLOCK_N, NUM_EXPERT_GROUP]

    # 3-D one-hot expand: [BLOCK_M, BLOCK_N, NUM_EXPERT_GROUP], with -inf
    # outside each group's column.
    sfc_3d = scores_for_choice[:, :, None].broadcast_to(
        BLOCK_M, BLOCK_N, NUM_EXPERT_GROUP
    )
    expanded = tl.where(gid_eq[None, :, :], sfc_3d, float("-inf"))
    group_max1 = tl.max(expanded, axis=1)        # [BLOCK_M, NUM_EXPERT_GROUP]

    if HAS_BIAS:
        # Top-2-sum-per-group. To find the second-largest score per group
        # without tl.argmax-on-3D, suppress the per-group max by exact-equality
        # match (ties on float scores are negligible in DeepSeek workloads).
        gm1_per_e = tl.sum(
            gid_eq[None, :, :].to(tl.float32) * group_max1[:, None, :],
            axis=2,
        )                                         # [BLOCK_M, BLOCK_N]
        suppressed = tl.where(
            scores_for_choice == gm1_per_e, float("-inf"), scores_for_choice
        )
        sup_3d = suppressed[:, :, None].broadcast_to(
            BLOCK_M, BLOCK_N, NUM_EXPERT_GROUP
        )
        expanded2 = tl.where(gid_eq[None, :, :], sup_3d, float("-inf"))
        group_max2 = tl.max(expanded2, axis=1)
        group_scores = group_max1 + group_max2
    else:
        group_scores = group_max1

    # -- 5. Top ``TOPK_GROUP`` groups via repeated argmax (NUM_EXPERT_GROUP
    #       is small; static-range unroll).
    group_mask_i = tl.zeros([BLOCK_M, NUM_EXPERT_GROUP], dtype=tl.int32)
    gs = group_scores
    for _gj in tl.static_range(TOPK_GROUP):
        am_g = tl.argmax(gs, axis=1).to(tl.int32)            # [BLOCK_M]
        sel_g = (g_arange[None, :] == am_g[:, None])         # [BLOCK_M, NUM_EXPERT_GROUP]
        group_mask_i = group_mask_i | sel_g.to(tl.int32)
        gs = tl.where(sel_g, float("-inf"), gs)

    # -- 6. Per-(token, expert) keep-mask via group-id lookup, then suppress
    #       experts in non-selected groups on the bias-augmented scores.
    expert_keep = tl.sum(
        gid_eq[None, :, :].to(tl.int32) * group_mask_i[:, None, :],
        axis=2,
    ) > 0                                                    # [BLOCK_M, BLOCK_N]
    sfc_masked = tl.where(expert_keep, scores_for_choice, float("-inf"))

    # -- 7. Per-expert top-``N_EXPTS_ACT`` via repeated argmax. Padded slots
    #       (N_EXPTS_ACT_PAD > N_EXPTS_ACT) are kept in the y_indices/y_values
    #       buffers but masked off on the writeback / bitmatrix-pack.
    n_arange = tl.arange(0, BLOCK_N)
    y_indices = tl.zeros([BLOCK_M, N_EXPTS_ACT_PAD], dtype=tl.int32)
    sfc_iter = sfc_masked
    for kj in tl.static_range(N_EXPTS_ACT):
        am_k = tl.argmax(sfc_iter, axis=1).to(tl.int32)      # [BLOCK_M]
        slot_eq = (tl.arange(0, N_EXPTS_ACT_PAD) == kj)[None, :]
        y_indices = tl.where(slot_eq, am_k[:, None], y_indices)
        sfc_iter = tl.where(
            n_arange[None, :] == am_k[:, None], float("-inf"), sfc_iter
        )

    # -- 8. Gather UNBIASED weights at selected indices.
    pos_eq = (
        n_arange[None, None, :] == y_indices[:, :, None]
    )                                                        # [BLOCK_M, K_PAD, BLOCK_N]
    scores_3d = scores[:, None, :].broadcast_to(BLOCK_M, N_EXPTS_ACT_PAD, BLOCK_N)
    y_weights = tl.sum(tl.where(pos_eq, scores_3d, 0.0), axis=2)  # [BLOCK_M, K_PAD]

    # Routed-slot mask: the first N_EXPTS_ACT slots hold the grouped-topk
    # selection (shared experts, if any, occupy the next N_SHARED slots and
    # must be excluded from the routed renorm denominator).
    k_arange = tl.arange(0, N_EXPTS_ACT_PAD)
    routed_mask = k_arange[None, :] < N_EXPTS_ACT

    # -- 9. Renorm + scale over the ROUTED slots only (mirrors _topk's
    #       APPLY_RENORM / ROUTED_SCALING and the noaux_tc semantics where the
    #       always-on shared expert is appended unscaled after renorm).
    if APPLY_RENORM:
        y_f = tl.where(routed_mask, y_weights, 0.0)
        s = tl.sum(y_f, axis=1, keep_dims=True)
        y_weights = y_f / (s + 1e-20) * ROUTED_SCALING
    elif ROUTED_SCALING != 1.0:
        y_weights = y_weights * ROUTED_SCALING

    # -- 9b. Append fused shared expert(s): always-on, fixed id n_expts_tot+i
    #        and fixed weight SHARED_SCORE (matches init_aiter_topK_meta_data /
    #        rocm_aiter_grouped_topk). Placed AFTER renorm so the shared weight
    #        is not folded into the routed normalization.
    if N_SHARED > 0:
        shared_slot = (k_arange[None, :] >= N_EXPTS_ACT) & (
            k_arange[None, :] < N_EXPTS_ACT + N_SHARED
        )
        shared_idx = (n_expts_tot + k_arange - N_EXPTS_ACT)[None, :].to(tl.int32)
        y_indices = tl.where(shared_slot, shared_idx, y_indices)
        y_weights = tl.where(shared_slot, SHARED_SCORE, y_weights)
        real_mask = k_arange[None, :] < (N_EXPTS_ACT + N_SHARED)
    else:
        real_mask = routed_mask

    y_values_out = y_weights.to(x_dtype)

    # -- 10. Writeback selected weights / indices.
    Yv_ptrs = Yv + offs_m[:, None] * stride_ym + k_arange[None, :]
    Yi_ptrs = Yi + offs_m[:, None] * stride_ym + k_arange[None, :]
    write_mask = mask_m[:, None] & real_mask
    tl.store(Yv_ptrs, y_values_out, mask=write_mask)
    tl.store(Yi_ptrs, y_indices, mask=write_mask)

    # -- 11. Pack into bitmatrix (mirrors _topk's tail).
    safe_idx = tl.where(real_mask, y_indices, 0).to(tl.uint32)
    y_div = safe_idx // 32
    y_rem = safe_idx % 32
    bm_iters: tl.constexpr = N_EXPTS_PAD // BLOCK_N          # = 1 (single-block)
    for i in range(bm_iters):
        offs_r_n = tl.arange(0, BLOCK_N // 32) + i * (BLOCK_N // 32)
        y2 = tl.where(
            (y_div[:, :, None] == offs_r_n[None, None, :]) & real_mask[:, :, None],
            (1 << y_rem)[:, :, None],
            0,
        )
        r = tl.reduce_or(y2, axis=1)
        BitsPtrs = (
            Bits + offs_m[:, None] * stride_rm + offs_r_n[None, :] * stride_rn
        )
        tl.store(BitsPtrs, r, mask=mask_m[:, None])

