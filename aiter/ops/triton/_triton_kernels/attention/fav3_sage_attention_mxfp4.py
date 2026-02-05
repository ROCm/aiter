import torch
import triton
import triton.language as tl
from aiter.ops.triton._triton_kernels.flash_attn_triton_amd.common import (
    compute_alibi_block,
)
from aiter.ops.triton.utils._triton.pid_preprocessing import pid_grid_3d
from aiter.ops.triton._triton_kernels.attention.fav3_sage_attention import (
    compute_block_masking,
    compute_alibi_block,
    map_dims,
)
from aiter.ops.triton.moe.quant_moe import downcast_to_mxfp


@triton.jit
def _sage_fwd_no_mask_mxfp4(
    acc,
    l_i,
    m_i,
    q,
    k_base_ptrs,
    v_base_ptrs,
    delta_s_base_ptrs,
    bias_base_ptrs,
    stride_kn,
    stride_vk,
    stride_bn,
    stride_sn,
    stride_sm,
    start_m,
    seqlen_k,
    seqlen_q,
    dropout_p,
    philox_seed,
    philox_offset_base,
    sd_mask,
    stride_sz,
    stride_sh,
    off_z,
    off_h_q,
    offs_m,
    offs_d_k,
    offs_d_v,
    block_min,
    block_max,
    alibi_slope,
    q_descale,
    k_descale_base_ptrs,
    stride_ksn,
    SM_SCALE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    PRE_LOAD_V: tl.constexpr,
    ENABLE_DROPOUT: tl.constexpr,
    PADDED_HEAD_QK: tl.constexpr,
    PADDED_HEAD_V: tl.constexpr,
    ACTUAL_BLOCK_DMODEL_QK: tl.constexpr,
    ACTUAL_BLOCK_DMODEL_V: tl.constexpr,
    USE_ALIBI: tl.constexpr,
    USE_EXP2: tl.constexpr,
    RETURN_SCORES: tl.constexpr,
    Q_DTYPE_STR: tl.constexpr,
    K_DTYPE_STR: tl.constexpr,
):
    # loop over k, v, and update accumulator
    for start_n in range(block_min, block_max, BLOCK_N):
        # get ptrs
        k_ptrs = k_base_ptrs + start_n * stride_kn
        v_ptrs = v_base_ptrs + start_n * stride_vk
        k_descale_ptrs = k_descale_base_ptrs + start_n * stride_ksn
        delta_s_ptrs = delta_s_base_ptrs + start_n

        kv_offs_n = start_n + tl.arange(0, BLOCK_N)
        # Load K
        if PADDED_HEAD_QK:
            k_mask = offs_d_k[:, None] < ACTUAL_BLOCK_DMODEL_QK
            k = tl.load(k_ptrs, mask=k_mask, other=0.0)
        else:
            k = tl.load(k_ptrs)

        k_descale = tl.load(k_descale_ptrs)
        delta_s = tl.load(delta_s_ptrs)

        # Optionally preload V
        if PRE_LOAD_V:
            if PADDED_HEAD_V:
                v_mask = offs_d_v[None, :] < ACTUAL_BLOCK_DMODEL_V
                v = tl.load(v_ptrs, mask=v_mask, other=0.0)
            else:
                v = tl.load(v_ptrs)

        # setup qk accumlator
        qk = delta_s[None, :].broadcast_to((BLOCK_M, BLOCK_N))

        # TODO SM_SCALE
        # -- compute qk ----
        qk = (
            tl.dot_scaled(
                q,
                q_descale,
                Q_DTYPE_STR,
                k,
                k_descale,
                K_DTYPE_STR,
                fast_math=True,
                acc=qk,
            )
            * SM_SCALE
        )
        # qk_scaled = qk * SM_SCALE
        qk_scaled = qk
        if USE_ALIBI:
            # compute the global position of each token within the sequence
            q_offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
            alibi_block = compute_alibi_block(
                alibi_slope, seqlen_q, seqlen_k, q_offs_m, kv_offs_n
            )
            qk_scaled += alibi_block

        # compute qk mask
        qk_mask = (offs_m[:, None] < seqlen_q) & (kv_offs_n[None, :] < seqlen_k)

        # compute bias
        if bias_base_ptrs is not None:
            bias_ptrs = bias_base_ptrs + start_n * stride_bn
            bias = tl.load(bias_ptrs, mask=qk_mask, other=0.0)
            qk_scaled += bias

        # get max scores so far
        m_ij = tl.maximum(m_i, tl.max(qk_scaled, 1))

        # scale and subtract max
        q_shifted = tl.where(
            m_ij[:, None] == float("-inf"), float("-inf"), qk_scaled - m_ij[:, None]
        )

        # Compute scaled QK and softmax probabilities
        if USE_EXP2:
            # p = tl.math.exp2(q_shifted * RCP_LN2)
            p = tl.math.exp2(q_shifted)
        else:
            p = tl.math.exp(q_shifted)

        # CAVEAT: Must update l_ij before applying dropout
        l_ij = tl.sum(p, 1)
        if ENABLE_DROPOUT:
            # Compute pointers for this block
            philox_base = philox_offset_base + off_z * stride_sz + off_h_q * stride_sh
            philox_ptrs = (
                philox_base
                + offs_m[:, None] * stride_sm
                + kv_offs_n[None, :] * stride_sn
            )

            # compute dropout mask
            rng_output = tl.rand(philox_seed, philox_ptrs)
            dropout_mask = rng_output > dropout_p

            # return scores with negative values for dropped vals (only if RETURN_SCORES is True)
            if RETURN_SCORES:
                sd_mask_value = tl.where(dropout_mask, p, -p)
                sd_mask_base = sd_mask + off_z * stride_sz + off_h_q * stride_sh
                sd_mask_ptrs = (
                    sd_mask_base
                    + offs_m[:, None] * stride_sm
                    + kv_offs_n[None, :] * stride_sn
                )

                # Compute mask for sd_mask storage
                sd_store_mask = (offs_m[:, None] < seqlen_q) & (
                    kv_offs_n[None, :] < seqlen_k
                )
                tl.store(sd_mask_ptrs, sd_mask_value, mask=sd_store_mask)

            # apply dropout mask in place
            p = tl.where(dropout_mask, p, 0.0)
        elif RETURN_SCORES:
            # NOTE: the returned score is not the same as the reference because we need to adjust as we find new maxes per block. We are not doing that
            sd_mask_base = sd_mask + off_z * stride_sz + off_h_q * stride_sh
            sd_mask_ptrs = (
                sd_mask_base
                + offs_m[:, None] * stride_sm
                + kv_offs_n[None, :] * stride_sn
            )

            # Compute mask for sd_mask storage
            sd_store_mask = (offs_m[:, None] < seqlen_q) & (
                kv_offs_n[None, :] < seqlen_k
            )
            tl.store(sd_mask_ptrs, p, mask=sd_store_mask)

        # -- update output accumulator --
        # alpha is an adjustment factor for acc and li as we loop and find new maxes
        # store the diff in maxes to adjust acc and li as we discover new maxes
        m_diff = tl.where(m_ij == float("-inf"), float("-inf"), m_i - m_ij)
        if USE_EXP2:
            # alpha = tl.math.exp2(m_diff * RCP_LN2)
            alpha = tl.math.exp2(m_diff)
        else:
            alpha = tl.math.exp(m_diff)
        acc = acc * alpha[:, None]
        if not PRE_LOAD_V:
            if PADDED_HEAD_V:
                v_mask = offs_d_v[None, :] < ACTUAL_BLOCK_DMODEL_V
                v = tl.load(v_ptrs, mask=v_mask, other=0.0)
            else:
                v = tl.load(v_ptrs)

        # -- update m_i and l_i
        l_i = l_i * alpha + l_ij
        m_i = m_ij

        acc += tl.dot((p).to(v.type.element_ty), v, out_dtype=tl.float32)

    return acc, l_i, m_i


@triton.jit
def _sage_fwd_mask_mxfp4(
    acc,
    l_i,
    m_i,
    q,
    k_base_ptrs,
    v_base_ptrs,
    delta_s_base_ptrs,
    bias_base_ptrs,
    stride_kn,
    stride_vk,
    stride_bn,
    stride_sn,
    stride_sm,
    start_m,
    seqlen_k,
    seqlen_q,
    dropout_p,
    philox_seed,
    philox_offset_base,
    sd_mask,
    stride_sz,
    stride_sh,
    off_z,
    off_h_q,
    offs_m,
    offs_n,
    offs_d_k,
    offs_d_v,
    block_min,
    block_max,
    n_extra_tokens,
    alibi_slope,
    q_descale,
    k_descale_base_ptrs,
    stride_ksn,
    SM_SCALE: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    PRE_LOAD_V: tl.constexpr,
    ENABLE_DROPOUT: tl.constexpr,
    PADDED_HEAD_QK: tl.constexpr,
    PADDED_HEAD_V: tl.constexpr,
    ACTUAL_BLOCK_DMODEL_QK: tl.constexpr,
    ACTUAL_BLOCK_DMODEL_V: tl.constexpr,
    USE_ALIBI: tl.constexpr,
    USE_EXP2: tl.constexpr,
    RETURN_SCORES: tl.constexpr,
    USE_SLIDING_WINDOW: tl.constexpr,
    WINDOW_SIZE_LEFT: tl.constexpr,
    WINDOW_SIZE_RIGHT: tl.constexpr,
    Q_DTYPE_STR: tl.constexpr,
    K_DTYPE_STR: tl.constexpr,
):
    # seqlen diff
    seqlen_delta_qk = seqlen_k - seqlen_q
    # loop over k, v, and update accumulator
    for start_n in range(block_min, block_max, BLOCK_N):
        # get ptrs
        k_ptrs = k_base_ptrs + start_n * stride_kn
        v_ptrs = v_base_ptrs + start_n * stride_vk
        k_descale_ptrs = k_descale_base_ptrs + start_n * stride_ksn
        delta_s_ptrs = delta_s_base_ptrs + start_n

        # For padded blocks, we will overrun the tensor size if
        # we load all BLOCK_N. For others, the blocks are all within range.
        kv_offs_n = start_n + tl.arange(0, BLOCK_N)
        k_mask = kv_offs_n[None, :] < seqlen_k
        v_mask = kv_offs_n[:, None] < seqlen_k
        if PADDED_HEAD_QK:
            k_mask = k_mask & (offs_d_k[:, None] < ACTUAL_BLOCK_DMODEL_QK)
        if PADDED_HEAD_V:
            v_mask = v_mask & (offs_d_v[None, :] < ACTUAL_BLOCK_DMODEL_V)

        # load k and if preload_v then v
        k = tl.load(k_ptrs, mask=k_mask, other=0.0)
        k_descale = tl.load(
            k_descale_ptrs, mask=kv_offs_n[:, None] < seqlen_k, other=0.0
        )
        delta_s = tl.load(delta_s_ptrs, kv_offs_n < seqlen_k, other=0.0)

        if PRE_LOAD_V:
            v = tl.load(v_ptrs, mask=v_mask, other=0.0)

        # setup qk accumlator
        qk = delta_s[None, :].broadcast_to((BLOCK_M, BLOCK_N))

        # We start from end of seqlen_k so only the first iteration would need
        # to be checked for padding if it is not a multiple of block_n
        # TODO: This can be optimized to only be true for the padded block.
        # If this is the last block / iteration, we want to
        # mask if the sequence length is not a multiple of block size
        # a solution is to always do BLOCK_M // BLOCK_N + 1 steps if not is_modulo_mn.
        # last step might get wasted but that is okay. check if this masking works For
        # that case.
        if (n_extra_tokens != 0) and (start_n + BLOCK_N == block_max):
            boundary_m = tl.full([BLOCK_M], seqlen_k, dtype=tl.int32)
            size_n = start_n + offs_n[None, :]
            mask = size_n < boundary_m[:, None]
            qk = tl.where(mask, qk, float("-inf"))

        # -- compute qk ----
        qk = (
            tl.dot_scaled(
                q,
                q_descale,
                Q_DTYPE_STR,
                k,
                k_descale,
                K_DTYPE_STR,
                fast_math=True,
                acc=qk,
            )
            * SM_SCALE
        )
        # qk_scaled = qk * SM_SCALE
        qk_scaled = qk

        if USE_ALIBI:
            # compute the global position of each token within the sequence
            q_offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
            alibi_block = compute_alibi_block(
                alibi_slope, seqlen_q, seqlen_k, q_offs_m, kv_offs_n
            )
            qk_scaled += alibi_block

        if USE_SLIDING_WINDOW:
            if IS_CAUSAL:
                # ========== CAUSAL SLIDING WINDOW MASKING ==========
                # For causal sliding window, we need to apply both constraints:
                # 1. Causal: col_idx <= row_idx + (seqlen_k - seqlen_q)
                # 2. Sliding window: row_idx - window_left <= col_idx <= row_idx + window_right

                # Get positions
                row_idx = offs_m  # Query positions
                col_idx = kv_offs_n  # Key positions

                # Expand for broadcasting
                row_idx_expanded = row_idx[:, None]  # [BLOCK_M, 1]
                col_idx_expanded = col_idx[None, :]  # [1, BLOCK_N]

                # Apply causal constraint: can only attend to positions before or at the diagonal
                causal_offset = seqlen_k - seqlen_q
                causal_mask = col_idx_expanded > (row_idx_expanded + causal_offset)

                # Apply sliding window constraint
                if WINDOW_SIZE_LEFT < 0:
                    # Only right window constraint
                    window_mask = col_idx_expanded > (
                        row_idx_expanded + causal_offset + WINDOW_SIZE_RIGHT
                    )
                else:
                    # Both left and right window constraints
                    # Adjust window bounds by causal offset
                    left_bound = row_idx_expanded + causal_offset - WINDOW_SIZE_LEFT
                    right_bound = row_idx_expanded + causal_offset + WINDOW_SIZE_RIGHT

                    # Can't attend to positions outside the window
                    window_mask = (col_idx_expanded < left_bound) | (
                        col_idx_expanded > right_bound
                    )

                # Final mask is the union of both constraints (True = cannot attend)
                mask = causal_mask | window_mask

                # Apply mask
                qk_scaled = tl.where(mask, float("-inf"), qk_scaled)
            else:
                # ========== NON-CAUSAL SLIDING WINDOW MASKING ==========
                # Exactly matching reference construct_local_mask:
                # row_idx = query positions, col_idx = key positions
                # sk = seqlen_k, sq = seqlen_q

                # Get positions
                row_idx = offs_m  # Query positions
                col_idx = kv_offs_n  # Key positions

                # sk and sq from reference (no padding masks in this test)
                sk = seqlen_k
                sq = seqlen_q

                # Expand for broadcasting
                row_idx_expanded = row_idx[:, None]  # [BLOCK_M, 1]
                col_idx_expanded = col_idx[None, :]  # [1, BLOCK_N]

                # Reference logic for mask computation
                if WINDOW_SIZE_LEFT < 0:
                    # Reference: return col_idx > row_idx + sk - sq + window_size[1]
                    mask = col_idx_expanded > (
                        row_idx_expanded + sk - sq + WINDOW_SIZE_RIGHT
                    )
                else:
                    # Reference:
                    # sk = torch.full_like(col_idx, seqlen_k) if key_padding_mask is None else sk
                    # return torch.logical_or(
                    #     col_idx > torch.minimum(row_idx + sk - sq + window_size[1], sk),
                    #     col_idx < row_idx + sk - sq - window_size[0],
                    # )
                    # Create sk tensor with proper shape for broadcasting
                    # sk represents the key sequence length, which should be compared per column
                    sk_full = tl.full((1, BLOCK_N), sk, dtype=tl.int32)

                    # Compute boundaries
                    right_bound_val = row_idx_expanded + sk - sq + WINDOW_SIZE_RIGHT
                    right_bound = tl.minimum(right_bound_val, sk_full)
                    left_bound = row_idx_expanded + sk - sq - WINDOW_SIZE_LEFT

                    # Mask where True = cannot attend (matching reference)
                    mask = (col_idx_expanded > right_bound) | (
                        col_idx_expanded < left_bound
                    )

                # Apply mask (set to -inf where mask is True)
                qk_scaled = tl.where(mask, float("-inf"), qk_scaled)
        else:
            if IS_CAUSAL:
                causal_boundary = start_n + offs_n - seqlen_delta_qk
                causal_mask = offs_m[:, None] >= causal_boundary[None, :]
                qk_scaled = tl.where(causal_mask, qk_scaled, float("-inf"))

        # compute qk mask
        qk_mask = (offs_m[:, None] < seqlen_q) & (kv_offs_n[None, :] < seqlen_k)

        # compute bias
        if bias_base_ptrs is not None:
            bias_ptrs = bias_base_ptrs + start_n * stride_bn
            bias = tl.load(bias_ptrs, mask=qk_mask, other=0.0)
            qk_scaled += bias

        # get max scores so far
        m_ij = tl.maximum(m_i, tl.max(qk_scaled, 1))

        # scale and subtract max
        # IMPORTANT: Handle the case where all values are -inf
        # When m_ij = -inf and qk_scaled = -inf, subtraction gives NaN
        # We need to handle this explicitly
        if USE_SLIDING_WINDOW:
            # Check if this block has any valid values (m_ij != -inf)
            # For rows where everything is -inf, set q_shifted to -inf (not NaN)
            q_shifted = tl.where(
                m_ij[:, None] == float("-inf"), float("-inf"), qk_scaled - m_ij[:, None]
            )
        else:
            q_shifted = qk_scaled - m_ij[:, None]

        # Compute scaled QK and softmax probabilities
        if USE_EXP2:
            # p = tl.math.exp2(q_shifted * RCP_LN2)
            p = tl.math.exp2(q_shifted)
        else:
            p = tl.math.exp(q_shifted)

        # CAVEAT: Must update l_ij before applying dropout
        l_ij = tl.sum(p, 1)
        if ENABLE_DROPOUT:
            # Compute pointers for this block
            philox_base = philox_offset_base + off_z * stride_sz + off_h_q * stride_sh
            philox_ptrs = (
                philox_base
                + offs_m[:, None] * stride_sm
                + kv_offs_n[None, :] * stride_sn
            )

            # compute dropout mask
            rng_output = tl.rand(philox_seed, philox_ptrs)
            dropout_mask = rng_output > dropout_p

            # return scores with negative values for dropped vals (only if RETURN_SCORES is True)
            if RETURN_SCORES:
                sd_mask_value = tl.where(dropout_mask, p, -p)
                sd_mask_base = sd_mask + off_z * stride_sz + off_h_q * stride_sh
                sd_mask_ptrs = (
                    sd_mask_base
                    + offs_m[:, None] * stride_sm
                    + kv_offs_n[None, :] * stride_sn
                )

                # Compute mask for sd_mask storage - include bounds check
                sd_store_mask = (offs_m[:, None] < seqlen_q) & (
                    kv_offs_n[None, :] < seqlen_k
                )

                # Add causal mask if applicable to prevent writing to invalid positions
                if IS_CAUSAL:
                    seqlen_delta_qk = seqlen_k - seqlen_q
                    causal_constraint = kv_offs_n[None, :] <= (
                        offs_m[:, None] + seqlen_delta_qk
                    )
                    sd_store_mask = sd_store_mask & causal_constraint

                # Add sliding window mask if applicable
                if USE_SLIDING_WINDOW:
                    seqlen_delta_qk = seqlen_k - seqlen_q
                    if WINDOW_SIZE_LEFT < 0:
                        # Only right window constraint
                        window_constraint = kv_offs_n[None, :] <= (
                            offs_m[:, None] + seqlen_delta_qk + WINDOW_SIZE_RIGHT
                        )
                    else:
                        # Both left and right window constraints
                        left_bound = (
                            offs_m[:, None] + seqlen_delta_qk - WINDOW_SIZE_LEFT
                        )
                        right_bound = (
                            offs_m[:, None] + seqlen_delta_qk + WINDOW_SIZE_RIGHT
                        )
                        window_constraint = (kv_offs_n[None, :] >= left_bound) & (
                            kv_offs_n[None, :] <= right_bound
                        )
                    sd_store_mask = sd_store_mask & window_constraint

                tl.store(sd_mask_ptrs, sd_mask_value, mask=sd_store_mask)

            # apply dropout mask in place
            p = tl.where(dropout_mask, p, 0.0)
        elif RETURN_SCORES:
            # NOTE: the returned score is not the same as the reference because we need to adjust as we find new maxes per block. We are not doing that
            sd_mask_base = sd_mask + off_z * stride_sz + off_h_q * stride_sh
            sd_mask_ptrs = (
                sd_mask_base
                + offs_m[:, None] * stride_sm
                + kv_offs_n[None, :] * stride_sn
            )

            # Compute mask for sd_mask storage - include bounds check
            sd_store_mask = (offs_m[:, None] < seqlen_q) & (
                kv_offs_n[None, :] < seqlen_k
            )

            # Add causal mask if applicable
            if IS_CAUSAL:
                seqlen_delta_qk = seqlen_k - seqlen_q
                causal_constraint = kv_offs_n[None, :] <= (
                    offs_m[:, None] + seqlen_delta_qk
                )
                sd_store_mask = sd_store_mask & causal_constraint

            # Add sliding window mask if applicable
            if USE_SLIDING_WINDOW:
                seqlen_delta_qk = seqlen_k - seqlen_q
                if WINDOW_SIZE_LEFT < 0:
                    # Only right window constraint
                    window_constraint = kv_offs_n[None, :] <= (
                        offs_m[:, None] + seqlen_delta_qk + WINDOW_SIZE_RIGHT
                    )
                else:
                    # Both left and right window constraints
                    left_bound = offs_m[:, None] + seqlen_delta_qk - WINDOW_SIZE_LEFT
                    right_bound = offs_m[:, None] + seqlen_delta_qk + WINDOW_SIZE_RIGHT
                    window_constraint = (kv_offs_n[None, :] >= left_bound) & (
                        kv_offs_n[None, :] <= right_bound
                    )
                sd_store_mask = sd_store_mask & window_constraint

            tl.store(sd_mask_ptrs, p, mask=sd_store_mask)

        # -- update output accumulator --
        # alpha is an adjustment factor for acc and li as we loop and find new maxes
        # store the diff in maxes to adjust acc and li as we discover new maxes
        m_diff = tl.where(m_ij == float("-inf"), float("-inf"), m_i - m_ij)
        if USE_EXP2:
            # alpha = tl.math.exp2(m_diff * RCP_LN2)
            alpha = tl.math.exp2(m_diff)
        else:
            alpha = tl.math.exp(m_diff)
        acc = acc * alpha[:, None]
        if not PRE_LOAD_V:
            v = tl.load(v_ptrs, mask=v_mask, other=0.0)

        # -- update m_i and l_i
        l_i = l_i * alpha + l_ij
        m_i = m_ij
        acc += tl.dot((p).to(v.type.element_ty), v, out_dtype=tl.float32)

    return acc, l_i, m_i


@triton.jit
def sage_fwd_mxfp4(
    Q,
    K,
    V,
    Delta_S,  # [b, h, seq_q_blk, seq_k]
    bias,
    Q_Descale,
    K_Descale,
    V_Descale,
    stride_qsz,
    stride_qsh,
    stride_qsm,
    stride_ksz,
    stride_ksh,
    stride_ksn,
    stride_vsz,
    stride_vsh,
    stride_dsz,
    stride_dsh,
    stride_dsq,
    LSE,
    Out,
    SD_MASK,
    ALIBI_SLOPES,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,
    stride_vz,
    stride_vh,
    stride_vk,
    stride_vn,
    stride_oz,
    stride_oh,
    stride_om,
    stride_on,
    stride_bz,
    stride_bh,
    stride_bm,
    stride_bn,
    stride_az,
    stride_ah,
    stride_sz,
    stride_sh,
    stride_sm,
    stride_sn,
    stride_lse_z,
    stride_lse_h,
    stride_lse_m,
    cu_seqlens_q,
    cu_seqlens_k,
    seqused_q,
    seqused_k,  # Add seqused parameters
    dropout_p,
    philox_seed,
    philox_offset_base,
    Q_DTYPE_STR: tl.constexpr,
    K_DTYPE_STR: tl.constexpr,
    RETURN_LSE: tl.constexpr,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    ACTUAL_BLOCK_DMODEL_QK: tl.constexpr,
    ACTUAL_BLOCK_DMODEL_V: tl.constexpr,
    MAX_SEQLENS_Q: tl.constexpr,
    MAX_SEQLENS_K: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    SM_SCALE: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    USE_SLIDING_WINDOW: tl.constexpr,
    WINDOW_SIZE_LEFT: tl.constexpr,
    WINDOW_SIZE_RIGHT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL_QK: tl.constexpr,
    BLOCK_DMODEL_V: tl.constexpr,
    BLOCK_N: tl.constexpr,
    PRE_LOAD_V: tl.constexpr,
    USE_BIAS: tl.constexpr,
    ENABLE_DROPOUT: tl.constexpr,
    RETURN_SCORES: tl.constexpr,
    USE_ALIBI: tl.constexpr,
    USE_EXP2: tl.constexpr,
    USE_SEQUSED: tl.constexpr,
):
    # set params
    Q_HEAD_DIM_DIVISOR: tl.constexpr = 2 if Q_DTYPE_STR == "e2m1" else 1
    K_HEAD_DIM_DIVISOR: tl.constexpr = 2 if K_DTYPE_STR == "e2m1" else 1
    SCALE_GROUP_SIZE: tl.constexpr = 32
    ACCUMULATOR_TYPE: tl.constexpr = tl.float32  # for q*k product

    BLOCK_DMODEL_Q: tl.constexpr = BLOCK_DMODEL_QK // Q_HEAD_DIM_DIVISOR
    BLOCK_DMODEL_K: tl.constexpr = BLOCK_DMODEL_QK // K_HEAD_DIM_DIVISOR
    BLOCK_DMODEL_QK_S: tl.constexpr = BLOCK_DMODEL_QK // SCALE_GROUP_SIZE

    # compute offsets
    start_m = tl.program_id(0)
    off_h_q = tl.program_id(1)
    off_z = tl.program_id(2)
    # If MQA / GQA, set the K and V head offsets appropriately.
    GROUP_SIZE: tl.constexpr = HQ // HK
    if GROUP_SIZE != 1:
        off_h_k = off_h_q // GROUP_SIZE
    else:
        off_h_k = off_h_q
    # Determine if we need to mask the heads
    PADDED_HEAD_QK: tl.constexpr = ACTUAL_BLOCK_DMODEL_QK != BLOCK_DMODEL_QK
    PADDED_HEAD_V: tl.constexpr = ACTUAL_BLOCK_DMODEL_V != BLOCK_DMODEL_V

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d_q = tl.arange(0, BLOCK_DMODEL_Q)  # we fit 2 fp4 elements per int8
    offs_d_k = tl.arange(0, BLOCK_DMODEL_K)  # we fit 2 fp4 elements per int8
    offs_d_qk_s = tl.arange(0, BLOCK_DMODEL_QK_S)
    offs_d_v = tl.arange(0, BLOCK_DMODEL_V)
    tl.multiple_of(offs_m, BLOCK_M),
    # N dimension
    offs_n = tl.arange(0, BLOCK_N)
    tl.multiple_of(offs_n, BLOCK_N),

    # D dimensions (MOST IMPORTANT)
    offs_d_q = tl.max_contiguous(
        tl.multiple_of(offs_d_q, BLOCK_DMODEL_Q), BLOCK_DMODEL_Q
    )
    offs_d_k = tl.max_contiguous(
        tl.multiple_of(offs_d_q, BLOCK_DMODEL_K), BLOCK_DMODEL_K
    )
    offs_d_v = tl.max_contiguous(
        tl.multiple_of(offs_d_v, BLOCK_DMODEL_V), BLOCK_DMODEL_V
    )

    # handle seqlen
    if IS_VARLEN:
        cu_seqlens_q_start = tl.load(cu_seqlens_q + off_z)
        cu_seqlens_q_end = tl.load(cu_seqlens_q + off_z + 1)

        # If seqused is provided, use it to limit the actual sequence length
        if USE_SEQUSED:
            actual_seqlen_q = (
                tl.load(seqused_q + off_z)
                if seqused_q is not None
                else cu_seqlens_q_end - cu_seqlens_q_start
            )
            seqlen_q = tl.minimum(
                actual_seqlen_q, cu_seqlens_q_end - cu_seqlens_q_start
            )
        else:
            seqlen_q = cu_seqlens_q_end - cu_seqlens_q_start

        # we have a one-size-fits-all grid in id(0). Some seqlens might be too small for all start_m so for those we return early.
        if start_m * BLOCK_M > seqlen_q:
            return
        cu_seqlens_k_start = tl.load(cu_seqlens_k + off_z)
        cu_seqlens_k_end = tl.load(cu_seqlens_k + off_z + 1)

        # If seqused is provided, use it to limit the actual sequence length for keys
        if USE_SEQUSED:
            actual_seqlen_k = (
                tl.load(seqused_k + off_z)
                if seqused_k is not None
                else cu_seqlens_k_end - cu_seqlens_k_start
            )
            seqlen_k = tl.minimum(
                actual_seqlen_k, cu_seqlens_k_end - cu_seqlens_k_start
            )
        else:
            seqlen_k = cu_seqlens_k_end - cu_seqlens_k_start
    else:
        cu_seqlens_q_start = 0
        cu_seqlens_k_start = 0
        seqlen_q = MAX_SEQLENS_Q
        seqlen_k = MAX_SEQLENS_K

    # figure out masking pattern
    (
        n_front_skip_blocks,
        n_front_masked_blocks,
        n_full_blocks,
        n_back_masked_blocks,
        n_extra_tokens,
    ) = compute_block_masking(
        seqlen_k,
        seqlen_q,
        start_m,
        IS_CAUSAL,
        USE_SLIDING_WINDOW,
        WINDOW_SIZE_LEFT,
        WINDOW_SIZE_RIGHT,
        BLOCK_M,
        BLOCK_N,
    )

    # ============================================================
    #          PROGRAM EARLY EXIT (All K Blocks Skipped)
    # ============================================================
    total_visible_blocks = n_front_masked_blocks + n_full_blocks + n_back_masked_blocks
    if total_visible_blocks == 0:
        """
        No K blocks visible - write zeros and exit.
        """
        # Write zeros to output
        o_offset = (
            Out
            + off_z * stride_oz
            + off_h_q * stride_oh
            + cu_seqlens_q_start * stride_om
        )
        o_ptrs = o_offset + offs_m[:, None] * stride_om + offs_d_v[None, :] * stride_on
        o_mask = offs_m[:, None] < seqlen_q
        if PADDED_HEAD_V:
            o_mask = o_mask & (offs_d_v[None, :] < ACTUAL_BLOCK_DMODEL_V)
        tl.store(
            o_ptrs,
            tl.zeros([BLOCK_M, BLOCK_DMODEL_V], dtype=Out.type.element_ty),
            mask=o_mask,
        )

        # Write zeros to LSE
        if RETURN_LSE:
            l_ptrs = (
                LSE
                + off_z * stride_lse_z
                + off_h_q * stride_lse_h
                + cu_seqlens_q_start * stride_lse_m
                + offs_m * stride_lse_m
            )
            tl.store(
                l_ptrs, tl.zeros([BLOCK_M], dtype=tl.float32), mask=offs_m < seqlen_q
            )
        return

    # ============================================================
    #         NORMAL PROCESSING (Some K Blocks Visible)
    # ============================================================
    """
    This program has visible K blocks to process.
    We'll use two calls to handle different block types efficiently.
    """

    # Initialize for processing
    # Compute pointers for all the tensors used in this kernel.
    q_offset = (
        Q + off_z * stride_qz + off_h_q * stride_qh + cu_seqlens_q_start * stride_qm
    )
    q_ptrs = q_offset + offs_m[:, None] * stride_qm + offs_d_q[None, :] * stride_qk
    k_offset = (
        K + off_z * stride_kz + off_h_k * stride_kh + cu_seqlens_k_start * stride_kn
    )
    k_ptrs = k_offset + offs_d_k[:, None] * stride_kk + offs_n[None, :] * stride_kn
    v_offset = (
        V + off_z * stride_vz + off_h_k * stride_vh + cu_seqlens_k_start * stride_vk
    )
    v_ptrs = v_offset + offs_n[:, None] * stride_vk + offs_d_v[None, :] * stride_vn
    q_descale_offset = (
        Q_Descale
        + off_z * stride_qsz
        + off_h_q * stride_qsh
        + (start_m + cu_seqlens_q_start) * stride_qsm
    )
    q_descale_ptrs = (
        q_descale_offset + offs_m[:, None] * stride_qsm + offs_d_qk_s[None, :]
    )
    k_descale_offset = (
        K_Descale
        + off_z * stride_ksz
        + off_h_k * stride_ksh
        + cu_seqlens_k_start * stride_ksn
    )
    k_descale_ptrs = (
        k_descale_offset + offs_n[:, None] * stride_ksn + offs_d_qk_s[None, :]
    )
    v_descale_ptr = V_Descale + off_z * stride_vsz + off_h_k * stride_vsh + offs_d_v
    delta_s_offset = Delta_S + off_z * stride_dsz + off_h_k * stride_dsh + start_m * stride_dsq
    delta_s_ptrs = delta_s_offset + offs_n

    q_descale = tl.load(
        q_descale_ptrs, mask=offs_m[:, None] < seqlen_q, other=0.0
    )  # MHA: use q head index
    if USE_BIAS:
        # Note: this might get large enough to overflow on some configs
        bias_offset = off_h_q * stride_bh
        bias_ptrs = (
            bias
            + bias_offset
            + offs_m[:, None] * stride_bm
            + offs_n[None, :] * stride_bn
        )
    else:
        bias_ptrs = None

    if USE_ALIBI:
        a_offset = off_z * stride_az + off_h_q * stride_ah
        alibi_slope = tl.load(ALIBI_SLOPES + a_offset)
    else:
        alibi_slope = None

    # initialize pointer to m and l
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=ACCUMULATOR_TYPE)
    l_i = tl.full([BLOCK_M], 1.0, dtype=ACCUMULATOR_TYPE)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL_V], dtype=ACCUMULATOR_TYPE)

    # Q is loaded once at the beginning and shared by all N blocks.
    q_ptrs_mask = offs_m[:, None] < seqlen_q
    if PADDED_HEAD_QK:
        q_ptrs_mask = q_ptrs_mask & (
            offs_d_q[None, :] < ACTUAL_BLOCK_DMODEL_QK // Q_HEAD_DIM_DIVISOR
        )
    q = tl.load(q_ptrs, mask=q_ptrs_mask, other=0.0)

    # ========== Process MASKED K Blocks in the front ==========
    # NOTE: we use USE_SLIDING_WINDOW as guard because the compiler will crash other wise. front masking is only for sliding window so that is fine.
    if n_front_masked_blocks > 0 and USE_SLIDING_WINDOW:
        block_min = n_front_skip_blocks * BLOCK_N
        block_max = (n_front_skip_blocks + n_front_masked_blocks) * BLOCK_N
        acc, l_i, m_i = _sage_fwd_mask_mxfp4(
            acc,
            l_i,
            m_i,
            q,
            k_ptrs,
            v_ptrs,
            delta_s_ptrs,
            bias_ptrs,
            stride_kn,
            stride_vk,
            stride_bn,
            stride_sn,
            stride_sm,
            start_m,
            seqlen_k,
            seqlen_q,
            dropout_p,
            philox_seed,
            philox_offset_base,
            SD_MASK,
            stride_sz,
            stride_sh,
            off_z,
            off_h_q,
            offs_m,
            offs_n,
            offs_d_k,
            offs_d_v,
            block_min,  # Start of front masked blocks
            block_max,  # End of front masked blocks
            0,  # n_extra_tokens (0 for front blocks, only relevant for last block)
            alibi_slope,
            q_descale,
            k_descale_ptrs,
            stride_ksn,
            SM_SCALE,
            IS_CAUSAL,
            BLOCK_M,
            BLOCK_N,
            PRE_LOAD_V,
            ENABLE_DROPOUT,
            PADDED_HEAD_QK,
            PADDED_HEAD_V,
            ACTUAL_BLOCK_DMODEL_QK,
            ACTUAL_BLOCK_DMODEL_V,
            USE_ALIBI=USE_ALIBI,
            USE_EXP2=USE_EXP2,
            RETURN_SCORES=RETURN_SCORES,
            USE_SLIDING_WINDOW=USE_SLIDING_WINDOW,
            WINDOW_SIZE_LEFT=WINDOW_SIZE_LEFT,
            WINDOW_SIZE_RIGHT=WINDOW_SIZE_RIGHT,
            Q_DTYPE_STR=Q_DTYPE_STR,
            K_DTYPE_STR=K_DTYPE_STR,
        )

    # ========== Process FULL K Blocks (Fast Path) ==========
    if n_full_blocks > 0:
        block_min = (n_front_skip_blocks + n_front_masked_blocks) * BLOCK_N
        block_max = (
            n_front_skip_blocks + n_front_masked_blocks + n_full_blocks
        ) * BLOCK_N

        acc, l_i, m_i = _sage_fwd_no_mask_mxfp4(
            acc,
            l_i,
            m_i,
            q,
            k_ptrs,
            v_ptrs,
            delta_s_ptrs,
            bias_ptrs,
            stride_kn,
            stride_vk,
            stride_bn,
            stride_sn,
            stride_sm,
            start_m,
            seqlen_k,
            seqlen_q,
            dropout_p,
            philox_seed,
            philox_offset_base,
            SD_MASK,
            stride_sz,
            stride_sh,
            off_z,
            off_h_q,
            offs_m,
            offs_d_k,
            offs_d_v,
            block_min,  # Start of range: 0
            block_max,  # End of range: n_full_blocks * BLOCK_N
            alibi_slope,
            q_descale,
            k_descale_ptrs,
            stride_ksn,
            SM_SCALE,
            BLOCK_M,
            BLOCK_N,
            PRE_LOAD_V,
            ENABLE_DROPOUT,
            PADDED_HEAD_QK,
            PADDED_HEAD_V,
            ACTUAL_BLOCK_DMODEL_QK,
            ACTUAL_BLOCK_DMODEL_V,
            USE_ALIBI=USE_ALIBI,
            USE_EXP2=USE_EXP2,
            RETURN_SCORES=RETURN_SCORES,
            Q_DTYPE_STR=Q_DTYPE_STR,
            K_DTYPE_STR=K_DTYPE_STR,
        )

    # ========== Process MASKED K Blocks in the back ==========
    if n_back_masked_blocks > 0:
        block_min = (
            n_front_skip_blocks + n_front_masked_blocks + n_full_blocks
        ) * BLOCK_N
        block_max = (
            n_front_skip_blocks
            + n_front_masked_blocks
            + n_full_blocks
            + n_back_masked_blocks
        ) * BLOCK_N

        acc, l_i, m_i = _sage_fwd_mask_mxfp4(
            acc,
            l_i,
            m_i,
            q,
            k_ptrs,
            v_ptrs,
            delta_s_ptrs,
            bias_ptrs,
            stride_kn,
            stride_vk,
            stride_bn,
            stride_sn,
            stride_sm,
            start_m,
            seqlen_k,
            seqlen_q,
            dropout_p,
            philox_seed,
            philox_offset_base,
            SD_MASK,
            stride_sz,
            stride_sh,
            off_z,
            off_h_q,
            offs_m,
            offs_n,
            offs_d_k,
            offs_d_v,
            block_min,  # Start of range: n_full_blocks * BLOCK_N
            block_max,  # End of range: n_visible_k_blocks * BLOCK_N
            n_extra_tokens,  # Padding tokens in last block
            alibi_slope,
            q_descale,
            k_descale_ptrs,
            stride_ksn,
            SM_SCALE,
            IS_CAUSAL,  # Use actual causal flag
            BLOCK_M,
            BLOCK_N,
            PRE_LOAD_V,
            ENABLE_DROPOUT,
            PADDED_HEAD_QK,
            PADDED_HEAD_V,
            ACTUAL_BLOCK_DMODEL_QK,
            ACTUAL_BLOCK_DMODEL_V,
            USE_ALIBI=USE_ALIBI,
            USE_EXP2=USE_EXP2,
            RETURN_SCORES=RETURN_SCORES,
            USE_SLIDING_WINDOW=USE_SLIDING_WINDOW,
            WINDOW_SIZE_LEFT=WINDOW_SIZE_LEFT,
            WINDOW_SIZE_RIGHT=WINDOW_SIZE_RIGHT,
            Q_DTYPE_STR=Q_DTYPE_STR,
            K_DTYPE_STR=K_DTYPE_STR,
        )

    # ============================================================
    #                        EPILOGUE
    # ============================================================
    # This helps the compiler do Newton Raphson on l_i vs on acc which is much larger.
    # Instead of directly computing 1/l_i which can be inf,
    # we check for the invalid case first
    if USE_SLIDING_WINDOW:
        # For rows where m_i is still -inf, no keys were valid
        # Set l_i to 1.0 to avoid division by zero (acc is already 0)
        invalid_mask = m_i == float("-inf")
        l_i_safe = tl.where(invalid_mask, 1.0, l_i)
        l_recip = 1 / l_i_safe[:, None]
    else:
        invalid_mask = None
        l_recip = 1 / l_i[:, None]

    v_descale = tl.load(
        v_descale_ptr,
        mask=offs_d_v < ACTUAL_BLOCK_DMODEL_V,
        other=0.0,
    )

    acc = acc * l_recip * v_descale
    if ENABLE_DROPOUT:
        dropout_scale = 1 / (1 - dropout_p)
        acc = acc * dropout_scale

    # compute log-sum-exp
    if RETURN_LSE:
        if USE_EXP2:
            # RCP_LN2: tl.constexpr = 1.4426950408889634
            LN2: tl.constexpr = 0.6931471824645996
            # compute log-sum-exp in base 2 units
            # mi_base2 = m_i * RCP_LN2
            mi_base2 = m_i
            # For invalid rows, log(l_i) would be -inf, but we want LSE to be -inf
            # So we handle this case explicitly
            if USE_SLIDING_WINDOW:
                log_l_i = tl.where(invalid_mask, 0.0, tl.math.log2(l_i))
                softmax_lse = mi_base2 + log_l_i
                # Ensure invalid rows have LSE = -inf
                softmax_lse = tl.where(invalid_mask, float("-inf"), softmax_lse)
            else:
                softmax_lse = mi_base2 + tl.math.log2(l_i)
            # convert back to natural units
            softmax_lse *= LN2
        else:
            if USE_SLIDING_WINDOW:
                log_l_i = tl.where(invalid_mask, 0.0, tl.math.log(l_i))
                softmax_lse = m_i + log_l_i
                softmax_lse = tl.where(invalid_mask, float("-inf"), softmax_lse)
            else:
                softmax_lse = m_i + tl.math.log(l_i)

    # handle masking edge cases
    if USE_SLIDING_WINDOW:
        if IS_CAUSAL:
            pass
        else:
            pass
    else:
        if IS_CAUSAL:
            # When seqlen_q > seqlen_k, some rows are completely above the causal diagonal
            # These rows have all -inf attention scores, resulting in NaN after softmax
            # e.g.
            # Q length: 6, K length: 4
            # Causal mask (X = can attend, . = cannot):
            #    K0 K1 K2 K3
            # Q0   .  .  .  .  <- All masked, would give NaN
            # Q1   .  .  .  .  <- All masked, would give NaN
            # Q2   X  .  .  .  <- First valid row
            # Q3   X  X  .  .
            # Q4   X  X  X  .
            # Q5   X  X  X  X
            causal_start_idx = seqlen_q - seqlen_k
            start_m_idx = start_m * BLOCK_M

            # Create mask for rows that need zeroing
            row_indices = start_m_idx + tl.arange(0, BLOCK_M)
            causal_mask = row_indices < causal_start_idx

            # Zero out both acc and LSE for these rows
            if causal_start_idx > start_m_idx:
                end_m_idx = (start_m + 1) * BLOCK_M
                if causal_start_idx < end_m_idx:
                    # This block contains the boundary - need to mask acc
                    out_mask_boundary = tl.full(
                        (BLOCK_DMODEL_V,), causal_start_idx, dtype=tl.int32
                    )
                    out_ptrs_mask = row_indices[:, None] >= out_mask_boundary[None, :]
                    z = 0.0
                    acc = tl.where(out_ptrs_mask, acc, z.to(acc.type.element_ty))

            # Zero out LSE for rows above diagonal
            if RETURN_LSE:
                softmax_lse = tl.where(causal_mask, 0.0, softmax_lse)

    # write back LSE(Log Sum Exponents), the log of the normalization constant
    if RETURN_LSE:
        l_offset = (
            LSE
            + off_z * stride_lse_z
            + off_h_q * stride_lse_h
            + cu_seqlens_q_start * stride_lse_m
        )
        l_ptrs = l_offset + offs_m * stride_lse_m

    # If seqlen_q not multiple of BLOCK_M, we need to mask out the last few rows.
    # This is only true for the last Q block. For others, overflow_size will be -ve
    end_m_idx = (start_m + 1) * BLOCK_M
    overflow_size = end_m_idx - seqlen_q
    if RETURN_LSE:
        if overflow_size > 0:
            boundary = tl.full((BLOCK_M,), BLOCK_M - overflow_size, dtype=tl.int32)
            l_ptrs_mask = tl.arange(0, BLOCK_M) < boundary
            tl.store(l_ptrs, softmax_lse, mask=l_ptrs_mask)
        else:
            tl.store(l_ptrs, softmax_lse)

    # write back O
    o_offset = (
        Out + off_z * stride_oz + off_h_q * stride_oh + cu_seqlens_q_start * stride_om
    )
    o_ptrs = o_offset + offs_m[:, None] * stride_om + offs_d_v[None, :] * stride_on
    o_ptrs_mask = tl.full([BLOCK_M, BLOCK_DMODEL_V], 1, dtype=tl.int1)
    if overflow_size > 0:
        o_ptrs_mask = o_ptrs_mask & (offs_m[:, None] < seqlen_q)
    if PADDED_HEAD_V:
        o_ptrs_mask = o_ptrs_mask & (offs_d_v[None, :] < ACTUAL_BLOCK_DMODEL_V)

    tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=o_ptrs_mask)


def sage_quant_mxfp4(
    q,
    k,
    v,
    FP8_TYPE,
    FP8_MAX,
    BLKQ=128,
    BLKK=64,
    sm_scale=None,
    layout="bshd",
    smooth_k=True,
):
    v_fp8 = torch.empty_like(v, dtype=FP8_TYPE, device=v.device)

    if layout == "bhsd":
        b, h_qo, qo_len, head_dim = q.shape
        _, h_kv, kv_len, _ = k.shape

        stride_bz_k, stride_h_k, stride_seq_k = k.stride(0), k.stride(1), k.stride(2)

    elif layout == "bshd":
        b, qo_len, h_qo, head_dim = q.shape
        _, kv_len, h_kv, _ = k.shape

        stride_bz_k, stride_h_k, stride_seq_k = k.stride(0), k.stride(2), k.stride(1)
    else:
        raise ValueError(f"Unknown tensor layout: {layout}")
    Q_NUM_BLKS = (qo_len + BLKQ - 1) // BLKQ
    K_NUM_BLKS = (kv_len + BLKK - 1) // BLKK

    # Apply K tensor smoothing following SageAttention approach
    v_scale = v.abs().amax(dim=1 if layout == "bshd" else 2).to(torch.float32) / FP8_MAX

    v_task_count = b * h_kv * K_NUM_BLKS
    grid = (v_task_count,)
    q, k, delta_s = rotation_smooth_qk(q, k, BLKQ, layout=layout)
    sage_quant_v_kernel[grid](
        v,
        v_fp8,
        v_scale,
        stride_bz_k,
        stride_h_k,
        stride_seq_k,
        v_scale.stride(0),
        v_scale.stride(1),
        b,
        h_kv,
        K_NUM_BLKS,
        kv_len,
        D=head_dim,
        BLK_K=BLKK,
        num_stages=3,
        num_warps=8,
    )

    q_fp4, q_scale = downcast_to_mxfp(q, torch.uint8, axis=-1)
    k_fp4, k_scale = downcast_to_mxfp(k, torch.uint8, axis=-1)

    return q_fp4, q_scale, k_fp4, k_scale, v_fp8, v_scale, delta_s


@triton.jit
def sage_quant_v_kernel(
    V_Input,
    V_Output,
    V_Scale,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_vsz,
    stride_vsh,
    BATCH,
    K_HEAD,
    K_NUM_BLKS,
    SEQLEN_K,
    D: tl.constexpr,
    BLK_K: tl.constexpr,
):
    pid = tl.program_id(0)

    offs_blk_k = tl.arange(0, BLK_K)
    offs_d = tl.arange(0, D)

    # V
    off_blk, off_h, off_b = pid_grid_3d(pid, K_NUM_BLKS, K_HEAD, BATCH)
    offs_kn = off_blk * BLK_K + offs_blk_k

    v_offs = (
        off_b * stride_kz
        + off_h * stride_kh
        + offs_kn[:, None] * stride_kn
        + offs_d[None, :]
    )

    v_input_ptrs = V_Input + v_offs
    v_output_ptrs = V_Output + v_offs

    # just apply the per channel v_scales that have been computed outside
    v_scale_ptrs = V_Scale + off_b * stride_vsz + off_h * stride_vsh + offs_d[None, :]
    v = tl.load(v_input_ptrs, mask=offs_kn[:, None] < SEQLEN_K, other=0.0)
    v = v.to(tl.float32)
    v_scales = tl.load(v_scale_ptrs)
    v_quant = v / v_scales
    v_quant = v_quant.to(v_output_ptrs.dtype.element_ty)
    tl.store(v_output_ptrs, v_quant, mask=offs_kn[:, None] < SEQLEN_K)


@triton.jit
def _rot_q_kernel(
    Q,
    Q_rot,
    Q_mean,
    R,  # Hadamard matrix
    stride_qb,
    stride_qh,
    stride_qm,
    stride_qd,
    stride_mb,
    stride_mh,
    stride_mm,
    stride_md,
    n_heads,
    seq_len,
    d_model,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,  # BLOCK_D is 32
):
    # Grid: (batch * n_heads, seq_len // BLOCK_M, d_model // BLOCK_D)
    pid_bh = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_d = tl.program_id(2)

    pid_h = pid_bh % n_heads
    pid_b = pid_bh // n_heads

    # Offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)

    # Load Q block and R (Hadamard)
    # Q block shape: [BLOCK_M, BLOCK_D]
    q_ptr = (
        Q
        + pid_b * stride_qb
        + pid_h * stride_qh
        + offs_m[:, None] * stride_qm
        + offs_d[None, :] * stride_qd
    )
    r_ptr = (
        R + tl.arange(0, BLOCK_D)[:, None] * BLOCK_D + tl.arange(0, BLOCK_D)[None, :]
    )
    q_tile = tl.load(
        q_ptr, mask=(offs_m[:, None] < seq_len) & (offs_d[None, :] < d_model), other=0.0
    )
    r_mat = tl.load(r_ptr)  # 32x32

    # Rotate: Q_rot = Q @ R
    q_rot_tile = tl.dot(q_tile, r_mat)

    # Store rotated Q
    rot_ptr = (
        Q_rot
        + pid_b * stride_qb
        + pid_h * stride_qh
        + offs_m[:, None] * stride_qm
        + offs_d[None, :] * stride_qd
    )
    tl.store(
        rot_ptr,
        q_rot_tile,
        mask=(offs_m[:, None] < seq_len) & (offs_d[None, :] < d_model),
    )

    # Calculate mean for the block (reduction over d within the BLOCK_M)
    # q_mean shape: [B, H, Q_NUM_BLKS, D]
    m_row_sum = tl.sum(q_rot_tile, axis=0)  # Sum over BLOCK_M -> shape [BLOCK_D]

    # Store mean (Atomic add or structured store)
    # For simplicity in this layout, we store the block-sum
    # and divide by BLOCK_M in the host or final step
    mean_ptr = Q_mean + pid_b * stride_mb + pid_h * stride_mh + pid_m * stride_mm + offs_d * stride_md
    tl.store(mean_ptr, m_row_sum / BLOCK_M)


@triton.jit
def _rot_k_only_kernel(
    K,
    K_rot,
    R,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_kd,
    n_heads,
    seq_k,
    d_model,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_d = tl.program_id(2)

    pid_h = pid_bh % n_heads
    pid_b = pid_bh // n_heads

    offs_n = pid_n * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)

    # Load K block and R
    k_ptr = (
        K
        + pid_b * stride_kb
        + pid_h * stride_kh
        + offs_n[:, None] * stride_kn
        + offs_d[None, :] * stride_kd
    )
    r_ptr = (
        R + tl.arange(0, BLOCK_D)[:, None] * BLOCK_D + tl.arange(0, BLOCK_D)[None, :]
    )

    k_tile = tl.load(
        k_ptr, mask=(offs_n[:, None] < seq_k) & (offs_d[None, :] < d_model), other=0.0
    )
    r_mat = tl.load(r_ptr)

    # Rotate K
    k_rot_tile = tl.dot(k_tile, r_mat)

    # Store
    rot_ptr = (
        K_rot
        + pid_b * stride_kb
        + pid_h * stride_kh
        + offs_n[:, None] * stride_kn
        + offs_d[None, :] * stride_kd
    )
    tl.store(
        rot_ptr,
        k_rot_tile,
        mask=(offs_n[:, None] < seq_k) & (offs_d[None, :] < d_model),
    )


@triton.jit
def _compute_delta_s_kernel(
    Q_mean,
    K_rot,
    Delta_S,
    stride_mb,
    stride_mh,
    stride_mm,
    stride_md,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_kd,
    stride_sb,
    stride_sh,
    stride_sm,
    stride_sn,
    n_heads,
    seq_k,
    d_model,
    BLOCK_N: tl.constexpr,  # Number of K-tokens to process
):
    pid_bh = tl.program_id(0)
    pid_m_q = tl.program_id(1)  # The Q-block index
    pid_n_k = tl.program_id(2)  # The K-block index

    pid_h = pid_bh % n_heads
    pid_b = pid_bh // n_heads

    offs_n = pid_n_k * BLOCK_N + tl.arange(0, BLOCK_N)

    # Accumulate dot product across the whole d_model
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)

    # Loop over d_model in steps of 32 (our block_size)
    for d_offset in range(0, d_model, 32):
        offs_d = d_offset + tl.arange(0, 32)

        # Load Q_mean segment: [32]
        qm_ptr = Q_mean + pid_b * stride_mb + pid_h * stride_mh + pid_m_q * stride_mm + offs_d * stride_md
        qm_val = tl.load(qm_ptr)

        # Load K_rot segment: [BLOCK_N, 32]
        kn_ptr = (
            K_rot
            + pid_b * stride_kb
            + pid_h * stride_kh
            + offs_n[:, None] * stride_kn
            + offs_d[None, :] * stride_kd
        )
        kn_val = tl.load(kn_ptr, mask=offs_n[:, None] < seq_k, other=0.0)

        # Compute dot product for this d-segment
        acc += tl.sum(qm_val[None, :] * kn_val, axis=1)

    # Store to Delta_S [B, H, Q_BLKS, seq_k]
    s_ptr = Delta_S + pid_b * stride_sb + pid_h * stride_sh + pid_m_q * stride_sm + offs_n * stride_sn
    tl.store(s_ptr, acc, mask=offs_n < seq_k)


def create_hadamard_matrix(block_size, device="cuda", dtype=torch.float32):
    """
    Create an orthogonal Hadamard matrix of size block_size x block_size.
    Uses Sylvester's recursive construction and normalizes to be orthogonal.

    Args:
        block_size: Size of the matrix (must be a power of 2)

    Returns:
        Orthogonal Hadamard matrix of shape (block_size, block_size)
        Satisfies: H @ H.T = I (identity matrix)

    Example:
        H_2 = [[1,  1],
               [1, -1]] / sqrt(2)

        H_4 = [[1,  1,  1,  1],
               [1, -1,  1, -1],
               [1,  1, -1, -1],
               [1, -1, -1,  1]] / 2
    """
    assert (block_size & (block_size - 1)) == 0, "block_size must be power of 2"
    assert block_size > 0, "block_size must be positive"

    # Base case: H_1 = [1]
    if block_size == 1:
        return torch.ones(1, 1, device=device, dtype=dtype)

    # Recursive construction: H_{2n} = [H_n   H_n  ]
    #                                   [H_n  -H_n ]
    H_half = create_hadamard_matrix(block_size // 2, device=device, dtype=dtype)

    # Build the full matrix (unnormalized)
    H = torch.zeros(block_size, block_size, device=device, dtype=dtype)
    half = block_size // 2
    H[:half, :half] = H_half
    H[:half, half:] = H_half
    H[half:, :half] = H_half
    H[half:, half:] = -H_half

    # Normalize to make it orthogonal: H @ H.T = I
    # The unnormalized matrix satisfies H_unnorm @ H_unnorm.T = block_size * I
    # So divide by sqrt(block_size) to get orthogonal matrix
    # H = H / (2.0 ** 0.5)  # Divide by sqrt(2) since we doubled the size

    return H


def rotation_smooth_qk(q, k, BLOCK_SIZE_M=256, block_size=32, layout="bhsd"):
    # Generate Hadamard Matrix R (Rank 32)
    R = create_hadamard_matrix(block_size, dtype=q.dtype) / (block_size**0.5)
    bshd = [0, 1, 2, 3] if layout == "bshd" else [0, 2, 1, 3]

    # shapes
    b, s_q, h_q, d = map_dims(q.shape, bshd)
    _, s_k, h_k, _ = map_dims(k.shape, bshd)

    Q_rot = torch.empty_like(q)
    K_rot = torch.empty_like(k)

    Q_NUM_BLKS = (s_q + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    K_NUM_BLKS = (s_k + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M

    # TODO check the dtypes for scales
    q_mean = torch.empty((b, h_q, Q_NUM_BLKS, d), dtype=torch.float32, device=q.device)
    delta_s = torch.empty(
        (b, h_q, Q_NUM_BLKS, s_k), dtype=torch.float32, device=q.device
    )

    stride_qb, stride_qm, stride_qh, stride_qd = map_dims(q.stride(), bshd)
    stride_kb, stride_kn, stride_kh, stride_kd = map_dims(k.stride(), bshd)

    # Launch Q Kernel
    grid_q = (b * h_q, Q_NUM_BLKS, d // block_size)
    _rot_q_kernel[grid_q](
        q,
        Q_rot,
        q_mean,
        R,
        stride_qb,
        stride_qh,
        stride_qm,
        stride_qd,
        q_mean.stride(0),
        q_mean.stride(1),
        q_mean.stride(2),
        q_mean.stride(3),
        h_q,
        s_q,
        d,
        BLOCK_M=BLOCK_SIZE_M,
        BLOCK_D=block_size,
    )

    # 2. Rotate K (Only once!)
    grid_k = (b * h_k, K_NUM_BLKS, d // block_size)
    _rot_k_only_kernel[grid_k](
        k,
        K_rot,
        R,
        stride_kb,
        stride_kh,
        stride_kn,
        stride_kd,
        h_k,
        s_k,
        d,
        BLOCK_M=BLOCK_SIZE_M,
        BLOCK_D=block_size,
    )

    # smooth k after rotation
    k = k - k.mean(dim=1 if layout == "bshd" else 2, keepdim=True)

    # 3. Compute Smoothing Delta S
    # Grid: Each Q-block x Each K-block
    grid_delta = (b * h_k, Q_NUM_BLKS, K_NUM_BLKS)
    _compute_delta_s_kernel[grid_delta](
        q_mean,
        K_rot,
        delta_s,
        q_mean.stride(0),
        q_mean.stride(1),
        q_mean.stride(2),
        q_mean.stride(3),
        stride_kb,
        stride_kh,
        stride_kn,
        stride_kd,
        delta_s.stride(0),
        delta_s.stride(1),
        delta_s.stride(2),
        delta_s.stride(3),
        h_k,
        s_k,
        d,
        BLOCK_N=BLOCK_SIZE_M,
    )

    return Q_rot, K_rot, delta_s
