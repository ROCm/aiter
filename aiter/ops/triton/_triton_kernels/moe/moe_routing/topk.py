import triton
import triton.language as tl


@triton.jit
def get_topmask_and_fullmask(x):
    tl.static_assert(
        x.dtype.is_int_unsigned(), "floating-point value must be passed as bits"
    )
    tm: tl.constexpr = 1 << (-1 + x.dtype.primitive_bitwidth)
    fm: tl.constexpr = (1 << x.dtype.primitive_bitwidth) - 1
    tm_arr = tl.full(x.shape, tm, dtype=x.dtype)
    fm_arr = tl.full(x.shape, fm, dtype=x.dtype)
    return tm_arr, fm_arr


@triton.jit
def fpval_to_key(x):
    tm, fm = get_topmask_and_fullmask(x)
    return x ^ tl.where((x & tm) != 0, fm, tm)


@triton.jit
def key_to_fpval(x):
    tm, fm = get_topmask_and_fullmask(x)
    return x ^ tl.where((x & tm) == 0, fm, tm)


@triton.jit
def _apply_score_mode(x, SCORE_MODE: tl.constexpr):
    """Pre-transform raw logits before topk selection.

    SCORE_MODE values:
      - "softmax": no pre-transform (caller may apply softmax to selected
        values via APPLY_SOFTMAX flag).
      - "sqrtsoftplus": x → sqrt(softplus(x)) using numerically stable
        softplus(x) = max(x, 0) + log(1 + exp(-|x|)). Matches
        torch.nn.functional.softplus for the DeepSeek-V4 sqrtsoftplus router.
    """
    if SCORE_MODE == "sqrtsoftplus":
        x_f = x.to(tl.float32)
        softplus_x = tl.maximum(x_f, 0.0) + tl.log(1.0 + tl.exp(-tl.abs(x_f)))
        return tl.sqrt(softplus_x).to(x.dtype)
    # "softmax" (and default): identity
    return x


@triton.jit
def streaming_topk(
    X,
    stride_xm,
    n_expts_tot,
    offs_m,
    mask_m,
    N_EXPTS_PAD: tl.constexpr,
    N_EXPTS_ACT: tl.constexpr,
    N_EXPTS_ACT_PAD: tl.constexpr,
    BLOCK_N: tl.constexpr,
    Bias=None,
    SCORE_MODE: tl.constexpr = "softmax",
    HAS_BIAS: tl.constexpr = False,
):
    x_nbits: tl.constexpr = X.dtype.element_ty.primitive_bitwidth
    x_utype: tl.constexpr = tl.dtype(f"uint{x_nbits}")
    if x_nbits < 16:
        y_nbits: tl.constexpr = 32
    else:
        y_nbits: tl.constexpr = x_nbits * 2
    x_ultype: tl.constexpr = tl.dtype(f"uint{y_nbits}")
    x_dtype: tl.constexpr = X.dtype.element_ty

    loop_iterations: tl.constexpr = N_EXPTS_PAD // BLOCK_N - 1
    offs_x_n = loop_iterations * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_x_n[None, :] < n_expts_tot

    # First iteration (peeled, may have masked lanes). For SCORE_MODE="softmax"
    # (legacy), keep the exact original sequence (load with -inf placeholder,
    # no transform). For SCORE_MODE="sqrtsoftplus", load with 0 placeholder,
    # apply transform + optional bias, then explicitly mask invalid lanes to
    # -inf — because the transform of -inf is NOT -inf (sqrt(softplus(-inf))
    # = 0) and would incorrectly win topk against small valid scores.
    X_ptrs = X + offs_m[:, None] * stride_xm + offs_x_n[None, :]
    if SCORE_MODE == "softmax":
        x = tl.load(X_ptrs, mask=(mask_m & mask_n), other=float("-inf"))
    else:
        x = tl.load(X_ptrs, mask=(mask_m & mask_n), other=0.0)
        x = _apply_score_mode(x, SCORE_MODE)
        if HAS_BIAS:
            bias_col_mask = offs_x_n < n_expts_tot
            b = tl.load(Bias + offs_x_n, mask=bias_col_mask, other=0.0)
            x = x + b[None, :].to(x_dtype)
        x = tl.where(mask_m & mask_n, x, float("-inf"))
    x = fpval_to_key(x.to(x_utype, bitcast=True))
    x = (x.to(x_ultype) << 16) | offs_x_n[None, :]
    acc = tl.topk(x, N_EXPTS_ACT_PAD, dim=1)

    # subsequent iterations: full blocks within n_expts_tot, no col mask
    for _i in (tl.static_range if loop_iterations <= 4 else range)(loop_iterations):
        acc = tl.bitonic_merge(acc)  # ensure sorted ascending for the merge
        X_ptrs -= BLOCK_N
        offs_x_n -= BLOCK_N
        if SCORE_MODE == "softmax":
            x = tl.load(X_ptrs, mask=mask_m, other=float("-inf"))
        else:
            x = tl.load(X_ptrs, mask=mask_m, other=0.0)
            x = _apply_score_mode(x, SCORE_MODE)
            if HAS_BIAS:
                b = tl.load(Bias + offs_x_n)
                x = x + b[None, :].to(x_dtype)
            x = tl.where(mask_m, x, float("-inf"))
        x = fpval_to_key(x.to(x_utype, bitcast=True))
        x = (x.to(x_ultype) << 16) | offs_x_n[None, :]
        acc = tl.maximum(acc, tl.topk(x, N_EXPTS_ACT_PAD, dim=1))

    # Pre-existing bug fix: after the streaming merge loop, acc is not
    # guaranteed to be sorted by value (tl.maximum of an ASC and the new
    # tl.topk output is bitonic, not sorted). The mask `arange < K` only
    # works if acc is already sorted descending by value with the top-K at
    # positions 0..K-1. For K_PAD > K (e.g. K=6 with K_PAD=8), this drops
    # arbitrary entries, including real top-K entries. Fix: sort by value
    # ascending first, then mask the smallest (K_PAD - K) positions.
    if N_EXPTS_ACT != N_EXPTS_ACT_PAD:
        acc = tl.sort(acc, dim=1)
    # rotate expert index into upper 16 bits:
    # 0000vvvvvvvviiii --> iiii0000vvvvvvvv
    acc = (acc << (y_nbits - 16)) | (acc >> 16)
    if N_EXPTS_ACT != N_EXPTS_ACT_PAD:
        mask_expts_act = tl.arange(0, N_EXPTS_ACT_PAD)[None, :] >= (
            N_EXPTS_ACT_PAD - N_EXPTS_ACT
        )
        acc = tl.where(mask_expts_act, acc, N_EXPTS_PAD << (y_nbits - 16))
    # sort in ascending order of expert (descending order of key)
    acc = tl.sort(acc, dim=1)
    # iiii0000vvvvvvvv --> 0000iiii:
    y_indices = (acc >> (y_nbits - 16)).to(tl.uint32)
    # iiii0000vvvvvvvv --> vvvvvvvv:
    y_values_raw = acc.to(x_utype)
    y_values = key_to_fpval(y_values_raw).to(x_dtype, bitcast=True)
    if N_EXPTS_ACT != N_EXPTS_ACT_PAD:
        y_values = tl.where(y_indices == N_EXPTS_PAD, float("-inf"), y_values)

    return y_values, y_indices


@triton.jit
def _topk(
    X,
    stride_xm,  # inputs
    Yv,
    Yi,
    stride_ym,  # topk values/indices
    Bits,
    stride_rm,
    stride_rn,  # bitmatrix
    n_rows,
    n_expts_tot,  # shape
    S,
    BLOCK_S: tl.constexpr,
    s_blocks,  # thing to memset
    SP,
    BLOCK_SP: tl.constexpr,
    sp_blocks,
    sp_size,
    APPLY_SOFTMAX: tl.constexpr,  # constant
    BLOCK_M: tl.constexpr,
    N_EXPTS_PAD: tl.constexpr,
    N_EXPTS_ACT: tl.constexpr,
    N_EXPTS_ACT_PAD: tl.constexpr,
    BLOCK_N: tl.constexpr,
    Bias=None,
    SCORE_MODE: tl.constexpr = "softmax",
    HAS_BIAS: tl.constexpr = False,
    APPLY_RENORM: tl.constexpr = False,
    ROUTED_SCALING: tl.constexpr = 1.0,
):
    # Backward-compat sanity. APPLY_SOFTMAX = post-selection softmax over the
    # K selected logits (legacy behavior). It only makes sense when no
    # pre-transform is applied; for SCORE_MODE="sqrtsoftplus" the caller is
    # expected to use APPLY_RENORM + ROUTED_SCALING instead.
    tl.static_assert(
        (not APPLY_SOFTMAX) or (SCORE_MODE == "softmax"),
        "APPLY_SOFTMAX is only valid when SCORE_MODE='softmax'",
    )

    pid = tl.program_id(0)
    if isinstance(n_rows, tl.tensor) and n_rows.dtype.is_ptr():
        n_rows = tl.load(n_rows)

    if pid < s_blocks:
        tl.store(
            S + BLOCK_S * pid + tl.arange(0, BLOCK_S), tl.zeros([BLOCK_S], tl.int32)
        )
    elif pid < s_blocks + sp_blocks:
        offs = BLOCK_SP * (pid - s_blocks) + tl.arange(0, BLOCK_SP)
        tl.store(SP + offs, tl.zeros([BLOCK_SP], tl.int32), mask=offs < sp_size)

    if pid * BLOCK_M >= n_rows:
        return

    tl.static_assert(BLOCK_N % 32 == 0)
    tl.static_assert(N_EXPTS_PAD % BLOCK_N == 0)
    x_dtype: tl.constexpr = X.dtype.element_ty

    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_y_n = tl.arange(0, N_EXPTS_ACT_PAD)
    mask_m = offs_m[:, None] < n_rows
    y_values, y_indices = streaming_topk(
        X,
        stride_xm,
        n_expts_tot,
        offs_m,
        mask_m,
        N_EXPTS_PAD,
        N_EXPTS_ACT,
        N_EXPTS_ACT_PAD,
        BLOCK_N,
        Bias=Bias,
        SCORE_MODE=SCORE_MODE,
        HAS_BIAS=HAS_BIAS,
    )

    # Real-entry mask: padded (sentinel) entries are flagged by
    # y_indices == N_EXPTS_PAD inside streaming_topk and have y_values = -inf.
    # Post-selection ops (bias-subtract, renorm, scaling) must operate on
    # real entries only — otherwise -inf poisons the renorm sum, and the
    # sentinel y_indices (== N_EXPTS_PAD) would OOB the Bias array.
    real_mask = (
        y_indices != N_EXPTS_PAD if N_EXPTS_ACT != N_EXPTS_ACT_PAD else (y_indices >= 0)
    )

    # For SCORE_MODE="sqrtsoftplus" with HAS_BIAS, the y_values returned by
    # streaming_topk are biased scores (sqrt(softplus(x)) + bias) — used for
    # selection. We want the unbiased sqrt(softplus(x)) values as the gathered
    # weights (the "noaux_tc" pattern from V4). Subtract bias[y_indices].
    if SCORE_MODE == "sqrtsoftplus" and HAS_BIAS:
        safe_idx = tl.where(real_mask, y_indices, 0).to(tl.int32)
        b_at_idx = tl.load(Bias + safe_idx)
        y_unbiased = y_values.to(tl.float32) - b_at_idx
        y_values = tl.where(real_mask, y_unbiased, 0.0).to(x_dtype)

    # normalize selected values
    if APPLY_SOFTMAX:
        y_values = tl.softmax(y_values.to(tl.float32), dim=1, keep_dims=True).to(
            x_dtype
        )
    elif APPLY_RENORM:
        y_f = tl.where(real_mask, y_values.to(tl.float32), 0.0)
        s = tl.sum(y_f, axis=1, keep_dims=True)
        y_values = (y_f / (s + 1e-20) * ROUTED_SCALING).to(x_dtype)
    elif ROUTED_SCALING != 1.0:
        y_values = (y_values.to(tl.float32) * ROUTED_SCALING).to(x_dtype)

    # write back
    Yv_ptrs = Yv + offs_m[:, None] * stride_ym + offs_y_n[None, :]
    if N_EXPTS_ACT != N_EXPTS_ACT_PAD:
        mask_n = offs_y_n[None, :] < N_EXPTS_ACT
        mask = mask_m & mask_n
    else:
        mask = mask_m
    tl.store(Yv_ptrs, y_values, mask=mask)
    Yi_ptrs = Yi + offs_m[:, None] * stride_ym + offs_y_n[None, :]
    tl.store(Yi_ptrs, y_indices, mask=mask)

    # pack into bitmatrix
    y_div = y_indices // 32
    y_rem = y_indices % 32
    loop_iterations = N_EXPTS_PAD // BLOCK_N
    for i in range(loop_iterations):
        offs_r_n = tl.arange(0, BLOCK_N // 32) + i * (BLOCK_N // 32)
        y2 = tl.where(
            y_div[:, :, None] == offs_r_n[None, None, :], (1 << y_rem)[:, :, None], 0
        )
        r = tl.reduce_or(y2, axis=1)
        BitsPtrs = Bits + offs_m[:, None] * stride_rm + offs_r_n[None, :] * stride_rn
        tl.store(BitsPtrs, r, mask=mask_m)


@triton.jit
def _hash_routing(
    InputIds,  # int32 [n_rows] — token-id per row
    Tid2Eid,  # int32 [vocab_size, K] — per-token-id top-K expert table
    stride_t2e_v,  # row stride of Tid2Eid (= K)
    X,  # [n_rows, n_expts_tot] router logits (bf16/fp32)
    stride_xm,
    Yv,  # output expt_scal [n_rows, N_EXPTS_ACT_PAD]
    Yi,  # output expt_indx [n_rows, N_EXPTS_ACT_PAD] (int16)
    stride_ym,
    Bits,  # bitmatrix data
    stride_rm,
    stride_rn,
    n_rows,
    n_expts_tot,
    S,  # bitmatrix scratchpad (must memset to 0)
    BLOCK_S: tl.constexpr,
    s_blocks,
    SP,  # bitmatrix partials (must memset to 0)
    BLOCK_SP: tl.constexpr,
    sp_blocks,
    sp_size,
    BLOCK_M: tl.constexpr,
    N_EXPTS_PAD: tl.constexpr,  # padded n_expts_tot (power of 2 ≥ n_expts_tot)
    N_EXPTS_ACT: tl.constexpr,
    N_EXPTS_ACT_PAD: tl.constexpr,  # next pow2 of K
    BLOCK_N: tl.constexpr,
    SCORE_MODE: tl.constexpr = "sqrtsoftplus",
    APPLY_RENORM: tl.constexpr = True,
    ROUTED_SCALING: tl.constexpr = 1.0,
):
    """Fused hash routing for DeepSeek-V4 hash layers.

    Replaces _hash_topk (Python: tid2eid lookup + softplus + sqrt + gather +
    renorm + scale) AND fused_routing_from_topk (3-kernel counting sort) AND
    bitmatrix construction with ONE Triton kernel. Output contract matches
    `_topk` so downstream `sort_tokens_fused` consumes it unchanged.

    Pipeline per row:
      1. expt_indx = Tid2Eid[input_id, :K]  # tid2eid lookup
      2. raw_scores = sqrt(softplus(X[row, :]))  # apply score transform
      3. expt_scal = raw_scores[expt_indx]  # gather K weights
      4. (optional) renorm: expt_scal /= expt_scal.sum() ; clamp_min(1e-20)
      5. expt_scal *= routed_scaling_factor
      6. Pack expt_indx into bitmatrix.

    No topk selection — expt_indx is fully determined by Tid2Eid lookup.
    """
    pid = tl.program_id(0)

    # Memset bitmatrix scratchpads (mirror _topk pattern)
    if pid < s_blocks:
        tl.store(
            S + BLOCK_S * pid + tl.arange(0, BLOCK_S), tl.zeros([BLOCK_S], tl.int32)
        )
    elif pid < s_blocks + sp_blocks:
        offs = BLOCK_SP * (pid - s_blocks) + tl.arange(0, BLOCK_SP)
        tl.store(SP + offs, tl.zeros([BLOCK_SP], tl.int32), mask=offs < sp_size)

    if pid * BLOCK_M >= n_rows:
        return

    tl.static_assert(BLOCK_N % 32 == 0)
    tl.static_assert(N_EXPTS_PAD % BLOCK_N == 0)
    x_dtype: tl.constexpr = X.dtype.element_ty

    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < n_rows

    # 1. Load input_ids for this BLOCK_M, then tid2eid[input_ids[i], :K]
    input_ids = tl.load(InputIds + offs_m, mask=mask_m, other=0).to(tl.int32)
    offs_k = tl.arange(0, N_EXPTS_ACT_PAD)
    mask_k = (
        offs_k < N_EXPTS_ACT
        if N_EXPTS_ACT != N_EXPTS_ACT_PAD
        else tl.full([N_EXPTS_ACT_PAD], 1, tl.int1)
    )
    # Gather Tid2Eid[input_ids[m], k] for m in BLOCK_M, k in K
    t2e_offs = input_ids[:, None] * stride_t2e_v + offs_k[None, :]
    expt_indx = tl.load(
        Tid2Eid + t2e_offs,
        mask=mask_m[:, None] & mask_k[None, :],
        other=0,
    ).to(tl.int32)

    # 2-3. Apply score transform to full row + gather at expt_indx.
    # Streaming load of X row in BLOCK_N chunks; accumulate scores per expert
    # at expt_indx (each row has K << n_expts_tot, so we test which chunk
    # holds each expert index).
    y_scores = tl.zeros([BLOCK_M, N_EXPTS_ACT_PAD], dtype=tl.float32)
    loop_iterations: tl.constexpr = N_EXPTS_PAD // BLOCK_N
    for i in range(loop_iterations):
        offs_x_n = i * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_x_n < n_expts_tot
        X_ptrs = X + offs_m[:, None] * stride_xm + offs_x_n[None, :]
        x = tl.load(X_ptrs, mask=mask_m[:, None] & mask_n[None, :], other=0.0).to(
            tl.float32
        )
        # sqrt(softplus(x)) — numerically stable
        if SCORE_MODE == "sqrtsoftplus":
            softplus_x = tl.maximum(x, 0.0) + tl.log(1.0 + tl.exp(-tl.abs(x)))
            scores = tl.sqrt(softplus_x)
        else:
            scores = x
        # For each (m, k): if expt_indx[m, k] is in [i*BLOCK_N, (i+1)*BLOCK_N), pick scores[m, expt_indx[m, k] - i*BLOCK_N]
        # Implement via expand: scores[m, n] vs expt_indx[m, k] mapping.
        # gate_mask[m, k, n] = (expt_indx[m, k] == offs_x_n[n])
        match = expt_indx[:, :, None] == offs_x_n[None, None, :]
        # picked[m, k] = sum_n match[m, k, n] * scores[m, n]
        scores_3d = scores[:, None, :].broadcast_to(BLOCK_M, N_EXPTS_ACT_PAD, BLOCK_N)
        picked = tl.where(match, scores_3d, 0.0)
        y_scores += tl.sum(picked, axis=2)

    # Real-entry mask (handles N_EXPTS_ACT_PAD > N_EXPTS_ACT padding lanes)
    real_mask = mask_k[None, :]
    y_f = tl.where(mask_m[:, None] & real_mask, y_scores, 0.0)

    # 4-5. Renorm + scale
    if APPLY_RENORM:
        s = tl.sum(y_f, axis=1, keep_dims=True)
        y_f = y_f / (s + 1e-20) * ROUTED_SCALING
    elif ROUTED_SCALING != 1.0:
        y_f = y_f * ROUTED_SCALING

    y_values = y_f.to(x_dtype)

    # Write outputs
    Yv_ptrs = Yv + offs_m[:, None] * stride_ym + offs_k[None, :]
    Yi_ptrs = Yi + offs_m[:, None] * stride_ym + offs_k[None, :]
    write_mask = mask_m[:, None] & real_mask
    tl.store(Yv_ptrs, y_values, mask=write_mask)
    tl.store(Yi_ptrs, expt_indx, mask=write_mask)

    # Pack into bitmatrix (mirror _topk pattern; sentinels (indx=0 in padded
    # lanes) safely OR with a real bit since y_values=0 in those lanes won't
    # affect downstream — but to be safe, mask out padded lanes from packing).
    safe_indx = tl.where(real_mask, expt_indx, 0).to(tl.int32)
    y_div = safe_indx // 32
    y_rem = safe_indx % 32
    bm_iters: tl.constexpr = N_EXPTS_PAD // BLOCK_N
    for i in range(bm_iters):
        offs_r_n = tl.arange(0, BLOCK_N // 32) + i * (BLOCK_N // 32)
        # Only contribute bits from real lanes
        y2 = tl.where(
            (y_div[:, :, None] == offs_r_n[None, None, :]) & real_mask[:, :, None],
            (1 << y_rem)[:, :, None],
            0,
        )
        r = tl.reduce_or(y2, axis=1)
        BitsPtrs = Bits + offs_m[:, None] * stride_rm + offs_r_n[None, :] * stride_rn
        tl.store(BitsPtrs, r, mask=mask_m[:, None])
