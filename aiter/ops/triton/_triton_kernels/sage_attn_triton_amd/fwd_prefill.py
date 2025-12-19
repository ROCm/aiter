import os
import warnings
import torch
import triton
import triton.language as tl
from typing import Literal, Optional
from .utils import (
    DEBUG,
    AUTOTUNE,
    FP8_AUTO_DESCALE,
    compute_alibi_block,
    get_arch,
    map_dims,
)
from aiter.ops.triton.utils._triton.pid_preprocessing import pid_grid_3d

# 0 for per block quantization, 1 for per channel quantization
V_QUANT_SCHEME = int(os.environ.get("V_QUANT_SCHEME", "1"))


def get_fwd_configs(autotune: bool):
    assert not autotune, "Autotuning is not supported."
    return {
        "BLOCK_M": 256,
        "BLOCK_N": 128,
        "waves_per_eu": 2,
        "PRE_LOAD_V": True,
        "num_stages": 1,
        "num_warps": 8,
    }

    configs = []
    keys = [
        "IS_CAUSAL",
        "dropout_p",
        "MAX_SEQLENS_Q",
        "MAX_SEQLENS_K",
        "ACTUAL_BLOCK_DMODEL_QK",
        "ACTUAL_BLOCK_DMODEL_V",
        "IS_VARLEN",
        "HQ",
        "HK",
    ]

    # get best config for the architecture
    if not autotune:
        arch = get_arch()
        if arch == "gfx950":
            configs.append(
                triton.Config(
                    {
                        "BLOCK_M": 128,
                        "BLOCK_N": 128,
                        "waves_per_eu": 2,
                        "PRE_LOAD_V": False,
                    },
                    num_stages=1,
                    num_warps=4,
                )
            )
        elif arch == "gfx942":
            configs.append(
                triton.Config(
                    {
                        "BLOCK_M": 128,
                        "BLOCK_N": 64,
                        "waves_per_eu": 2,
                        "PRE_LOAD_V": True,
                    },
                    num_warps=4,
                    num_stages=1,
                )
            )
            # if get_cu_count() < 304:
            #    configs.extend(
            #        [
            #            # best fp8 config
            #            triton.Config(
            #                {
            #                    "BLOCK_M": 128,
            #                    "BLOCK_N": 64,
            #                    "waves_per_eu": 2,
            #                    "PRE_LOAD_V": False,
            #                },
            #                num_stages=1,
            #                num_warps=4,
            #            ),
            #            ## best f16 config
            #            #triton.Config(
            #            #    {
            #            #        "BLOCK_M": 128,
            #            #        "BLOCK_N": 32,
            #            #        "waves_per_eu": 2,
            #            #        "PRE_LOAD_V": False,
            #            #    },
            #            #    num_stages=2,
            #            #    num_warps=4,
            #            #),
            #        ]
            #    )
            # else:
            #    configs.append(
            #        triton.Config(
            #            {
            #                "BLOCK_M": 128,
            #                "BLOCK_N": 64,
            #                "waves_per_eu": 2,
            #                "PRE_LOAD_V": False,
            #            },
            #            num_stages=1,
            #            num_warps=4,
            #        )
            #    )
        elif arch in (
            "gfx1030",
            "gfx1100",
            "gfx1101",
            "gfx1102",
            "gfx1200",
            "gfx1201",
        ):  # RDNA architectures
            configs.append(
                triton.Config(
                    {
                        "BLOCK_M": 32,
                        "BLOCK_N": 32,
                        "waves_per_eu": 2,
                        "PRE_LOAD_V": False,
                    },
                    num_stages=1,
                    num_warps=2,
                )
            )
        else:
            configs.append(
                triton.Config(
                    {
                        "BLOCK_M": 64,
                        "BLOCK_N": 64,
                        "waves_per_eu": 2,
                        "PRE_LOAD_V": False,
                    },
                    num_stages=1,
                    num_warps=4,
                )
            )

        return configs, keys

    # ===================== Autotune Sweep =====================
    BLOCK_M_OPTIONS = [128, 64, 32]
    BLOCK_N_OPTIONS = [128, 64, 32]
    NUM_WARPS_OPTIONS = [2, 4, 8]
    NUM_STAGES_OPTIONS = [1, 2]
    WAVES_PER_EU_OPTIONS = [4, 2, 1]
    PRE_LOAD_V_OPTIONS = [False]
    for bm in BLOCK_M_OPTIONS:
        for bn in BLOCK_N_OPTIONS:
            for waves in WAVES_PER_EU_OPTIONS:
                for nw in NUM_WARPS_OPTIONS:
                    for ns in NUM_STAGES_OPTIONS:
                        for preload_v in PRE_LOAD_V_OPTIONS:
                            configs.append(
                                triton.Config(
                                    {
                                        "BLOCK_M": bm,
                                        "BLOCK_N": bn,
                                        "waves_per_eu": waves,
                                        "PRE_LOAD_V": preload_v,
                                    },
                                    num_stages=ns,
                                    num_warps=nw,
                                )
                            )

    return configs, keys


# fwd_prefill_autotune_configs, fwd_prefill_autotune_keys = get_fwd_configs(AUTOTUNE)


@triton.jit
def _attn_fwd_no_mask(
    acc,
    l_i,
    m_i,
    q,
    k_base_ptrs,
    v_base_ptrs,
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
    offs_d_qk,
    offs_d_v,
    block_min,
    block_max,
    alibi_slope,
    q_descale,
    k_descale_base_ptr,
    v_descale_base_ptr,
    FP8_MAX,
    stride_k_descale_blk,
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
    V_QUANT_SCHEME: tl.constexpr,
    ACCUMULATOR_TYPE,
):
    if USE_EXP2:
        RCP_LN2: tl.constexpr = 1.4426950408889634

    k_descale_ptr = k_descale_base_ptr
    if V_QUANT_SCHEME == 0:
        v_descale_ptr = v_descale_base_ptr

    # loop over k, v, and update accumulator
    for start_n in range(block_min, block_max, BLOCK_N):
        # get ptrs
        k_ptrs = k_base_ptrs + start_n * stride_kn
        v_ptrs = v_base_ptrs + start_n * stride_vk

        kv_offs_n = start_n + tl.arange(0, BLOCK_N)
        # Load K
        if PADDED_HEAD_QK:
            k_mask = offs_d_qk[:, None] < ACTUAL_BLOCK_DMODEL_QK
            k = tl.load(k_ptrs, mask=k_mask, other=0.0)
        else:
            k = tl.load(k_ptrs)

        k_descale = tl.load(k_descale_ptr)
        k_descale_ptr += stride_k_descale_blk
        if V_QUANT_SCHEME == 0:
            v_descale = tl.load(v_descale_ptr)
            v_descale_ptr += stride_k_descale_blk

        # Optionally preload V
        if PRE_LOAD_V:
            if PADDED_HEAD_V:
                v_mask = offs_d_v[None, :] < ACTUAL_BLOCK_DMODEL_V
                v = tl.load(v_ptrs, mask=v_mask, other=0.0)
            else:
                v = tl.load(v_ptrs)

        # setup qk accumlator
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=ACCUMULATOR_TYPE)

        # -- compute qk ----
        qk += tl.dot(q, k) * (q_descale * k_descale)
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

        if V_QUANT_SCHEME == 0:
            acc += (
                tl.dot((p * FP8_MAX).to(v.type.element_ty), v, out_dtype=tl.float32)
                * v_descale
            )
        else:
            acc += tl.dot((p * FP8_MAX).to(v.type.element_ty), v, out_dtype=tl.float32)

    return acc, l_i, m_i


@triton.jit
def _attn_fwd_mask(
    acc,
    l_i,
    m_i,
    q,
    k_base_ptrs,
    v_base_ptrs,
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
    offs_d_qk,
    offs_d_v,
    block_min,
    block_max,
    n_extra_tokens,
    alibi_slope,
    q_descale,
    k_descale_base_ptr,
    v_descale_base_ptr,
    FP8_MAX,
    stride_k_descale_blk,
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
    V_QUANT_SCHEME: tl.constexpr,
    ACCUMULATOR_TYPE,
):
    if USE_EXP2:
        RCP_LN2: tl.constexpr = 1.4426950408889634

    # seqlen diff
    seqlen_delta_qk = seqlen_k - seqlen_q

    k_descale_ptr = k_descale_base_ptr
    if V_QUANT_SCHEME == 0:
        v_descale_ptr = v_descale_base_ptr

    # loop over k, v, and update accumulator
    for start_n in range(block_min, block_max, BLOCK_N):
        # get ptrs
        k_ptrs = k_base_ptrs + start_n * stride_kn
        v_ptrs = v_base_ptrs + start_n * stride_vk

        # For padded blocks, we will overrun the tensor size if
        # we load all BLOCK_N. For others, the blocks are all within range.
        kv_offs_n = start_n + tl.arange(0, BLOCK_N)
        k_mask = kv_offs_n[None, :] < seqlen_k
        v_mask = kv_offs_n[:, None] < seqlen_k
        if PADDED_HEAD_QK:
            k_mask = k_mask & (offs_d_qk[:, None] < ACTUAL_BLOCK_DMODEL_QK)
        if PADDED_HEAD_V:
            v_mask = v_mask & (offs_d_v[None, :] < ACTUAL_BLOCK_DMODEL_V)

        # load k and if preload_v then v
        k = tl.load(k_ptrs, mask=k_mask, other=0.0)
        k_descale = tl.load(k_descale_ptr)
        k_descale_ptr += stride_k_descale_blk
        if V_QUANT_SCHEME == 0:
            v_descale = tl.load(v_descale_ptr)
            v_descale_ptr += stride_k_descale_blk

        if PRE_LOAD_V:
            v = tl.load(v_ptrs, mask=v_mask, other=0.0)

        # setup qk accumlator
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=ACCUMULATOR_TYPE)

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
        qk += tl.dot(q, k) * (q_descale * k_descale)
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
        if V_QUANT_SCHEME == 0:
            acc += (
                tl.dot((p * FP8_MAX).to(v.type.element_ty), v, out_dtype=tl.float32)
                * v_descale
            )
        else:
            acc += tl.dot((p * FP8_MAX).to(v.type.element_ty), v, out_dtype=tl.float32)

    return acc, l_i, m_i


@triton.jit
def compute_window_bounds(
    q_start,
    q_end,
    diag,
    seqlen_k,
    WINDOW_SIZE_LEFT: tl.constexpr,
    WINDOW_SIZE_RIGHT: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    """Calculate the window boundaries for a query block."""
    # Left boundary
    if WINDOW_SIZE_LEFT < 0:
        left_min = 0
        left_max = 0
    else:
        left_min = tl.maximum(0, q_start + diag - WINDOW_SIZE_LEFT)
        left_max = tl.maximum(0, q_end + diag - WINDOW_SIZE_LEFT)

    # Right boundary
    if IS_CAUSAL:
        # Causal cap: col ≤ row + diag
        right_min = tl.minimum(seqlen_k - 1, q_start + diag)
        right_max = tl.minimum(seqlen_k - 1, q_end + diag)
    else:
        if WINDOW_SIZE_RIGHT < 0:
            right_min = tl.minimum(seqlen_k - 1, q_start + diag + WINDOW_SIZE_RIGHT)
            right_max = tl.minimum(seqlen_k - 1, q_end + diag + WINDOW_SIZE_RIGHT)
        else:
            # Non-causal doesn't have the diagonal constraint
            right_min = tl.minimum(seqlen_k - 1, q_start + diag + WINDOW_SIZE_RIGHT)
            right_max = tl.minimum(seqlen_k - 1, q_end + diag + WINDOW_SIZE_RIGHT)

    return left_min, left_max, right_min, right_max


@triton.jit
def classify_window_blocks(
    left_min, left_max, right_min, right_max, BLOCK_N: tl.constexpr
):
    """Classify blocks based on window boundaries."""
    # First and last blocks that have ANY overlap with window
    first_block = left_min // BLOCK_N
    last_block = right_max // BLOCK_N

    # First block that is FULLY visible for all rows in Q block
    full_left_block = left_max // BLOCK_N + (left_max % BLOCK_N != 0)
    clipped_left = tl.minimum(full_left_block, last_block + 1)

    # Last block that is FULLY visible for all rows in Q block
    last_full_block_candidate = right_min // BLOCK_N
    if (last_full_block_candidate + 1) * BLOCK_N - 1 > right_min:
        last_full_block_candidate -= 1
    full_right_block = tl.maximum(last_full_block_candidate, clipped_left - 1)

    # Calculate counts
    n_front_skip_blocks = first_block
    n_front_masked_blocks = tl.maximum(0, clipped_left - first_block)
    n_full_blocks = tl.maximum(0, full_right_block - clipped_left + 1)
    n_back_masked_blocks = tl.maximum(0, last_block - full_right_block)

    return (
        n_front_skip_blocks,
        n_front_masked_blocks,
        n_full_blocks,
        n_back_masked_blocks,
        clipped_left,
    )  # Return clipped_left for padded block handling


@triton.jit
def handle_padded_last_block(
    n_extra_tokens,
    last_block,
    total_k_blocks,
    clipped_left,
    n_front_masked_blocks,
    n_full_blocks,
    n_back_masked_blocks,
):
    """Ensure a padded last K-block is never classified as 'full'.

    We move the padded last block (if visible) into the back-masked bucket.
    If it's already back-masked, we do nothing.  If it was counted in the
    front-masked range, we decrement front-masked; if it was counted as full,
    we decrement full.  Then we increment back-masked.
    """
    padded_last_k = (n_extra_tokens != 0) & (last_block == total_k_blocks - 1)

    if padded_last_k:
        # current 'full' range right edge
        full_right_block = clipped_left + n_full_blocks - 1

        # If last_block is already beyond full_right_block, it's already in back-masked → nothing to do
        last_already_back_masked = last_block > full_right_block
        if not last_already_back_masked:
            # If the window starts past last_block, it was counted in front-masked
            if clipped_left > last_block:
                n_front_masked_blocks = tl.maximum(0, n_front_masked_blocks - 1)
            else:
                # Otherwise it was counted 'full' → move it out of full
                n_full_blocks = tl.maximum(0, n_full_blocks - 1)
            # In both cases we need one more back-masked block
            n_back_masked_blocks = n_back_masked_blocks + 1

    return n_front_masked_blocks, n_full_blocks, n_back_masked_blocks


@triton.jit
def compute_padding_info(seqlen_k, BLOCK_N: tl.constexpr):
    """Calculate padding information for the last K block."""
    # check if we will need to do masking due either BLOCK_N being bigger than seqlen_k or seqlen_k not being a factor of BLOCK_N
    # n_extra_tokens = 10 % 4 = 2
    # This means the last K block has 2 valid tokens and 2 padding positions
    # K blocks visualization:
    #         Block 0         Block 1         Block 2 (last)
    #         K0 K1 K2 K3    K4 K5 K6 K7     K8 K9 ?? ??
    #         ↑---------↑    ↑---------↑     ↑---↑ ↑---↑
    #         full block     full block      valid  pad
    if seqlen_k < BLOCK_N:
        n_extra_tokens = BLOCK_N - seqlen_k
    elif seqlen_k % BLOCK_N:
        n_extra_tokens = seqlen_k % BLOCK_N
    else:
        n_extra_tokens = 0
    return n_extra_tokens


@triton.jit
def compute_block_masking(
    seqlen_k,
    seqlen_q,
    start_m,
    IS_CAUSAL: tl.constexpr,
    USE_SLIDING_WINDOW: tl.constexpr,
    WINDOW_SIZE_LEFT: tl.constexpr,
    WINDOW_SIZE_RIGHT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Classify K blocks for attention computation with sliding window support.

    Returns:
        - n_front_skip_blocks: Blocks completely before the window
        - n_front_masked_blocks: Blocks partially overlapping window front
        - n_full_blocks: Blocks completely inside the window
        - n_back_masked_blocks: Blocks partially overlapping window back
        - n_extra_tokens: Padding tokens in last K block
    """

    # common
    q_start = start_m * BLOCK_M
    q_end = tl.minimum((start_m + 1) * BLOCK_M - 1, seqlen_q - 1)
    diag = seqlen_k - seqlen_q
    total_k_blocks = tl.cdiv(seqlen_k, BLOCK_N)
    n_extra_tokens = compute_padding_info(seqlen_k, BLOCK_N)

    if USE_SLIDING_WINDOW:
        # get window bounds
        left_min, left_max, right_min, right_max = compute_window_bounds(
            q_start,
            q_end,
            diag,
            seqlen_k,
            WINDOW_SIZE_LEFT,
            WINDOW_SIZE_RIGHT,
            IS_CAUSAL,
        )

        # window vanishes → early exit
        if right_max < left_min:
            return 0, 0, 0, 0, n_extra_tokens

        # classify blocks
        (
            n_front_skip_blocks,
            n_front_masked_blocks,
            n_full_blocks,
            n_back_masked_blocks,
            clipped_left,
        ) = classify_window_blocks(left_min, left_max, right_min, right_max, BLOCK_N)

        # handle padded last block if needed
        if n_extra_tokens != 0:
            last_block = right_max // BLOCK_N
            n_front_masked_blocks, n_full_blocks, n_back_masked_blocks = (
                handle_padded_last_block(
                    n_extra_tokens,
                    last_block,
                    total_k_blocks,
                    clipped_left,
                    n_front_masked_blocks,
                    n_full_blocks,
                    n_back_masked_blocks,
                )
            )
        return (
            n_front_skip_blocks,
            n_front_masked_blocks,
            n_full_blocks,
            n_back_masked_blocks,
            n_extra_tokens,
        )
    else:
        if IS_CAUSAL:
            # ========== CAUSAL MODE: Classify K Blocks ==========
            # Calculate causal boundary for this Q block
            #          [K0 K1 K2 K3] [K4 K5 K6 K7] [K8 K9 ?? ??]
            # Q0-Q3:   [ 1  0  0  0] [ 0  0  0  0] [ 0  0 -- --]  ← Q0
            #          [ 1  1  0  0] [ 0  0  0  0] [ 0  0 -- --]  ← Q1
            #          [ 1  1  1  0] [ 0  0  0  0] [ 0  0 -- --]  ← Q2
            #          [ 1  1  1  1] [ 1  1  0  0] [ 0  0 -- --]  ← Q3
            #                            ↑ can see up to K5
            #
            # Q4-Q7:   [ 1  1  1  1] [ 1  1  1  0] [ 0  0 -- --]  ← Q4
            #          [ 1  1  1  1] [ 1  1  1  1] [ 0  0 -- --]  ← Q5
            #          [ 1  1  1  1] [ 1  1  1  1] [ 1  0 -- --]  ← Q6
            #          [ 1  1  1  1] [ 1  1  1  1] [ 1  1 -- --]  ← Q7

            # ------------------------------------------------------------
            # 1. figure out, in tokens, the right-most K position
            #    this Q-block may attend to
            # ------------------------------------------------------------
            k_max_token = q_end + diag  # last visible K index

            # this Q-block is entirely above the diagonal ⇒ nothing to do
            if k_max_token < 0:
                return 0, 0, 0, 0, n_extra_tokens

            k_max_token = tl.minimum(k_max_token, seqlen_k - 1)

            # ------------------------------------------------------------
            # 2. translate token indices into K-block indices
            # ------------------------------------------------------------
            last_visible_k_block = k_max_token // BLOCK_N
            n_visible_k_blocks = tl.minimum(last_visible_k_block + 1, total_k_blocks)

            # ------------------------------------------------------------
            # 3. classify those visible blocks
            #    – we *never* skip or mask blocks in front, because causal
            #      attention always starts at K0
            #    – the back side can require several masked blocks:
            #         • intersection of the causal diagonal with K-grid
            #           (at most  ⌈BLOCK_M / BLOCK_N⌉ blocks)
            #         • plus one extra block if this Q-block stops in the
            #           middle of a K-block or the last K-block is padded
            # ------------------------------------------------------------
            padded_last_k = n_extra_tokens != 0
            is_modulo_mn = (not padded_last_k) & (seqlen_q % BLOCK_M == 0)

            n_back_masked_blocks = BLOCK_M // BLOCK_N + tl.where(is_modulo_mn, 0, 1)
            n_back_masked_blocks = tl.minimum(n_back_masked_blocks, n_visible_k_blocks)

            n_front_skip_blocks = 0  # causal never skips the left side
            n_front_masked_blocks = 0  # ditto
            n_full_blocks = n_visible_k_blocks - n_back_masked_blocks
        else:
            # ========== NON-CAUSAL MODE ==========
            # Without causal mask, all positions can attend to all positions
            # Only need to handle the padding in the last block
            #          [K0 K1 K2 K3] [K4 K5 K6 K7] [K8 K9 ?? ??]
            # Q0-Q3:   [ 1  1  1  1] [ 1  1  1  1] [ 1  1 -∞ -∞]
            #          [ 1  1  1  1] [ 1  1  1  1] [ 1  1 -∞ -∞]
            #          [ 1  1  1  1] [ 1  1  1  1] [ 1  1 -∞ -∞]
            #          [ 1  1  1  1] [ 1  1  1  1] [ 1  1 -∞ -∞]
            #
            # Q4-Q7:   [ 1  1  1  1] [ 1  1  1  1] [ 1  1 -∞ -∞]
            #          [ 1  1  1  1] [ 1  1  1  1] [ 1  1 -∞ -∞]
            #          [ 1  1  1  1] [ 1  1  1  1] [ 1  1 -∞ -∞]
            #          [ 1  1  1  1] [ 1  1  1  1] [ 1  1 -∞ -∞]

            n_front_skip_blocks = 0  # never skips the left side
            n_front_masked_blocks = 0  # ditto
            if n_extra_tokens != 0:
                n_back_masked_blocks = 1  # Last block needs padding mask
                n_full_blocks = total_k_blocks - 1
            else:
                n_back_masked_blocks = 0  # All blocks are aligned
                n_full_blocks = total_k_blocks

        return (
            n_front_skip_blocks,
            n_front_masked_blocks,
            n_full_blocks,
            n_back_masked_blocks,
            n_extra_tokens,
        )


# @triton.autotune(
#     configs=fwd_prefill_autotune_configs,
#     key=fwd_prefill_autotune_keys,
#     use_cuda_graph=True,
# )
@triton.jit
def attn_fwd(
    Q,
    K,
    V,
    bias,
    Q_Descale,
    K_Descale,
    V_Descale,
    FP8_MAX,
    stride_q_descale_z,
    stride_q_descale_h,
    stride_q_descale_blk,
    stride_k_descale_z,
    stride_k_descale_h,
    stride_k_descale_blk,
    stride_v_descale_z,
    stride_v_descale_h,
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
    V_QUANT_SCHEME: tl.constexpr,
):
    # set params
    ACCUMULATOR_TYPE = tl.float32  # for q*k product

    # compute offsets
    off_z = tl.program_id(0)
    off_h_q = tl.program_id(1)
    start_m = tl.program_id(2)
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
    offs_d_qk = tl.arange(0, BLOCK_DMODEL_QK)
    offs_d_v = tl.arange(0, BLOCK_DMODEL_V)

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

    # Load scale factors
    # For MQA/GQA (GROUP_SIZE != 1), q_descale uses the same indexing as k/v (off_h_k)
    # For MHA (GROUP_SIZE == 1), q_descale uses off_h_q (same as off_h_k)
    # TODO: consider GROUP_SIZE != 1
    q_descale = tl.load(
        Q_Descale
        + off_z * stride_q_descale_z
        + off_h_q * stride_q_descale_h
        + start_m * stride_q_descale_blk
    )  # MHA: use q head index

    k_descale_offset = off_z * stride_k_descale_z + off_h_k * stride_k_descale_h
    if V_QUANT_SCHEME == 1:
        v_descale = tl.load(
            V_Descale
            + off_z * stride_v_descale_z
            + off_h_k * stride_v_descale_h
            + offs_d_v,
            mask=offs_d_v < ACTUAL_BLOCK_DMODEL_V,
            other=0.0,
        )

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
    q_ptrs = q_offset + offs_m[:, None] * stride_qm + offs_d_qk[None, :] * stride_qk
    k_offset = (
        K + off_z * stride_kz + off_h_k * stride_kh + cu_seqlens_k_start * stride_kn
    )
    k_ptrs = k_offset + offs_d_qk[:, None] * stride_kk + offs_n[None, :] * stride_kn
    v_offset = (
        V + off_z * stride_vz + off_h_k * stride_vh + cu_seqlens_k_start * stride_vk
    )
    v_ptrs = v_offset + offs_n[:, None] * stride_vk + offs_d_v[None, :] * stride_vn
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
        q_ptrs_mask = q_ptrs_mask & (offs_d_qk[None, :] < ACTUAL_BLOCK_DMODEL_QK)
    q = tl.load(q_ptrs, mask=q_ptrs_mask, other=0.0)

    # ========== Process MASKED K Blocks in the front ==========
    # NOTE: we use USE_SLIDING_WINDOW as guard because the compiler will crash other wise. front masking is only for sliding window so that is fine.
    if n_front_masked_blocks > 0 and USE_SLIDING_WINDOW:
        block_min = n_front_skip_blocks * BLOCK_N
        block_max = (n_front_skip_blocks + n_front_masked_blocks) * BLOCK_N

        k_descale_ptr = (
            K_Descale + k_descale_offset + n_front_skip_blocks * stride_k_descale_blk
        )
        v_descale_ptr = (
            V_Descale + k_descale_offset + n_front_skip_blocks * stride_k_descale_blk
        )

        acc, l_i, m_i = _attn_fwd_mask(
            acc,
            l_i,
            m_i,
            q,
            k_ptrs,
            v_ptrs,
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
            offs_d_qk,
            offs_d_v,
            block_min,  # Start of front masked blocks
            block_max,  # End of front masked blocks
            0,  # n_extra_tokens (0 for front blocks, only relevant for last block)
            alibi_slope,
            q_descale,
            k_descale_ptr,
            v_descale_ptr if V_QUANT_SCHEME == 0 else None,
            FP8_MAX,
            stride_k_descale_blk,
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
            ACCUMULATOR_TYPE=ACCUMULATOR_TYPE,
            V_QUANT_SCHEME=V_QUANT_SCHEME,
        )

    # ========== Process FULL K Blocks (Fast Path) ==========
    if n_full_blocks > 0:
        block_min = (n_front_skip_blocks + n_front_masked_blocks) * BLOCK_N
        block_max = (
            n_front_skip_blocks + n_front_masked_blocks + n_full_blocks
        ) * BLOCK_N

        k_descale_ptr = (
            K_Descale
            + k_descale_offset
            + (n_front_skip_blocks + n_front_masked_blocks) * stride_k_descale_blk
        )
        v_descale_ptr = (
            V_Descale
            + k_descale_offset
            + (n_front_skip_blocks + n_front_masked_blocks) * stride_k_descale_blk
        )

        acc, l_i, m_i = _attn_fwd_no_mask(
            acc,
            l_i,
            m_i,
            q,
            k_ptrs,
            v_ptrs,
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
            offs_d_qk,
            offs_d_v,
            block_min,  # Start of range: 0
            block_max,  # End of range: n_full_blocks * BLOCK_N
            alibi_slope,
            q_descale,
            k_descale_ptr,
            v_descale_ptr if V_QUANT_SCHEME == 0 else None,
            FP8_MAX,
            stride_k_descale_blk,
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
            ACCUMULATOR_TYPE=ACCUMULATOR_TYPE,
            V_QUANT_SCHEME=V_QUANT_SCHEME,
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

        k_descale_ptr = (
            K_Descale
            + k_descale_offset
            + (n_front_skip_blocks + n_front_masked_blocks + n_full_blocks)
            * stride_k_descale_blk
        )
        v_descale_ptr = (
            V_Descale
            + k_descale_offset
            + (n_front_skip_blocks + n_front_masked_blocks + n_full_blocks)
            * stride_k_descale_blk
        )

        acc, l_i, m_i = _attn_fwd_mask(
            acc,
            l_i,
            m_i,
            q,
            k_ptrs,
            v_ptrs,
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
            offs_d_qk,
            offs_d_v,
            block_min,  # Start of range: n_full_blocks * BLOCK_N
            block_max,  # End of range: n_visible_k_blocks * BLOCK_N
            n_extra_tokens,  # Padding tokens in last block
            alibi_slope,
            q_descale,
            k_descale_ptr,
            v_descale_ptr if V_QUANT_SCHEME == 0 else None,
            FP8_MAX,
            stride_k_descale_blk,
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
            ACCUMULATOR_TYPE=ACCUMULATOR_TYPE,
            V_QUANT_SCHEME=V_QUANT_SCHEME,
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
    if V_QUANT_SCHEME == 0:
        acc = (acc * l_recip) / FP8_MAX
    else:
        acc = (acc * l_recip * v_descale) / FP8_MAX
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


def fav3_sage_triton_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    softmax_lse: Optional[torch.Tensor],
    sd_mask: Optional[torch.Tensor],
    sm_scale: float,
    alibi_slopes: Optional[torch.Tensor],
    causal: bool,
    window_size_left: int,
    window_size_right: int,
    bias: Optional[torch.Tensor],
    layout: Literal["bshd", "bhsd", "thd"],
    # varlen
    cu_seqlens_q: Optional[torch.Tensor],
    cu_seqlens_k: Optional[torch.Tensor],
    max_seqlens_q: int,
    max_seqlens_k: int,
    # dropout
    dropout_p: float,
    philox_seed: Optional[int],
    philox_offset: Optional[int],
    # misc
    return_scores: bool,
    use_exp2: bool,
    # fp8
    q_descale: Optional[torch.Tensor],
    k_descale: Optional[torch.Tensor],
    v_descale: Optional[torch.Tensor],
    FP8_MAX: float,
    # seqused for FA v3
    seqused_q: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    # rotary (optional)
    rotary_cos: Optional[torch.Tensor] = None,
    rotary_sin: Optional[torch.Tensor] = None,
    rotary_interleaved: bool = False,
    seqlens_rotary: Optional[torch.Tensor] = None,
):
    # get params, strides and shape
    IS_VARLEN = layout == "thd"

    # common assertions
    assert (
        0.0 <= dropout_p <= 1.0
    ), f"dropout_p must be between 0 and 1, got {dropout_p}"
    assert (
        q.device == k.device == v.device == o.device
    ), f"All tensors must be on the same device. Got: q={q.device}, k={k.device}, v={v.device}, o={o.device}"
    assert (
        q.dtype == k.dtype == torch.int8
    ), f"q, k must have the dtype torch.int8. Got {q.dtype} for q and {k.dtype} for k"
    current_device = torch.cuda.current_device()
    assert (
        q.is_cuda and q.device.index == current_device
    ), f"Device mismatch: Kernel will launch on cuda:{current_device}, but tensors are on {q.device}"

    # get shapes and strides
    if IS_VARLEN:
        # shape
        total_seqlen_q, nheads_q, head_size_q = q.shape
        total_seqlen_k, nheads_k, head_size_k = k.shape
        total_seqlen_v, nheads_v, head_size_v = v.shape

        # assert shapes
        assert (
            cu_seqlens_q is not None
        ), "cu_seqlens_q must be provided for varlen layout"
        assert (
            cu_seqlens_k is not None
        ), "cu_seqlens_k must be provided for varlen layout"
        assert (
            max_seqlens_q is not None and max_seqlens_q > 0
        ), "max_seqlens_q must be provided and positive for varlen layout"
        assert (
            max_seqlens_k is not None and max_seqlens_k > 0
        ), "max_seqlens_k must be provided and positive for varlen layout"

        # assert head dimensions
        assert (
            head_size_q == head_size_k
        ), f"head sizes must match: q={head_size_q}, k={head_size_k}"
        assert (
            nheads_k == nheads_v
        ), f"k and v must have same number of heads: k={nheads_k}, v={nheads_v}"
        assert (
            nheads_q % nheads_k == 0
        ), f"nheads_q {nheads_q} must be divisible by nheads_k {nheads_k} for GQA/MQA"

        # assert output shapes
        assert o.shape == (
            total_seqlen_q,
            nheads_q,
            head_size_v,
        ), f"o shape {o.shape} != expected {(total_seqlen_q, nheads_q, head_size_v)}"

        # assert cu_seqlens
        assert (
            cu_seqlens_q.dtype == torch.int32
        ), f"cu_seqlens_q must be int32, got {cu_seqlens_q.dtype}"
        assert (
            cu_seqlens_k.dtype == torch.int32
        ), f"cu_seqlens_k must be int32, got {cu_seqlens_k.dtype}"
        assert cu_seqlens_q[0] == 0, "cu_seqlens_q must start with 0"
        assert cu_seqlens_k[0] == 0, "cu_seqlens_k must start with 0"
        assert (
            cu_seqlens_q[-1] == total_seqlen_q
        ), f"cu_seqlens_q[-1] {cu_seqlens_q[-1]} != total_seqlen_q {total_seqlen_q}"
        assert (
            cu_seqlens_k[-1] == total_seqlen_k
        ), f"cu_seqlens_k[-1] {cu_seqlens_k[-1]} != total_seqlen_k {total_seqlen_k}"

        # set vars
        batch = len(cu_seqlens_q) - 1
        head_size_qk = head_size_q

        # Assert softmax_lse tensor is large enough
        if softmax_lse is not None:
            assert (
                softmax_lse.shape[0] >= nheads_q
            ), f"softmax_lse.shape[0]={softmax_lse.shape[0]} must be >= nheads_q={nheads_q}"
            assert (
                softmax_lse.shape[1] >= total_seqlen_q
            ), f"softmax_lse.shape[1]={softmax_lse.shape[1]} must be >= total_seqlen_q={total_seqlen_q}"
            assert (
                softmax_lse.dtype == torch.float32
            ), f"softmax_lse must be float32, got {softmax_lse.dtype}"
            assert (
                softmax_lse.device == q.device
            ), f"softmax_lse must be on same device as q"

        # strides
        stride_qb, stride_qh, stride_qm, stride_qd = (
            0,
            q.stride(1),
            q.stride(0),
            q.stride(2),
        )
        stride_kb, stride_kh, stride_kn, stride_kd = (
            0,
            k.stride(1),
            k.stride(0),
            k.stride(2),
        )
        stride_vb, stride_vh, stride_vn, stride_vd = (
            0,
            v.stride(1),
            v.stride(0),
            v.stride(2),
        )
        stride_ob, stride_oh, stride_om, stride_od = (
            0,
            o.stride(1),
            o.stride(0),
            o.stride(2),
        )
        stride_lse_z, stride_lse_h, stride_lse_m = (
            (
                0,
                softmax_lse.stride(0),
                softmax_lse.stride(1),
            )
            if softmax_lse is not None
            else (0, 0, 0)
        )
    else:
        bshd = [0, 1, 2, 3] if layout == "bshd" else [0, 2, 1, 3]
        # shapes
        batch_q, seqlen_q, nheads_q, head_size_q = map_dims(q.shape, bshd)
        batch_k, seqlen_k, nheads_k, head_size_k = map_dims(k.shape, bshd)
        batch_v, seqlen_v, nheads_v, head_size_v = map_dims(v.shape, bshd)

        # assert batch dimensions
        assert (
            batch_q == batch_k == batch_v
        ), f"batch sizes must match: q={batch_q}, k={batch_k}, v={batch_v}"

        # assert head dimensions
        assert (
            head_size_q == head_size_k
        ), f"head sizes must match: q={head_size_q}, k={head_size_k}"
        assert (
            nheads_k == nheads_v
        ), f"k and v must have same number of heads: k={nheads_k}, v={nheads_v}"
        assert (
            nheads_q % nheads_k == 0
        ), f"nheads_q {nheads_q} must be divisible by nheads_k {nheads_k} for GQA/MQA"

        # assert sequence lengths
        assert (
            seqlen_k == seqlen_v
        ), f"k and v sequence lengths must match: k={seqlen_k}, v={seqlen_v}"

        # assert output shapes
        assert o.shape == (
            q.shape[0],
            q.shape[1],
            q.shape[2],
            v.shape[-1],
        ), f"o shape {o.shape} != expected {(batch_q, seqlen_q, nheads_q, head_size_v)}"

        # set vars
        batch = batch_q
        head_size_qk = head_size_q
        max_seqlens_q = seqlen_q
        max_seqlens_k = seqlen_k

        # Assert softmax_lse tensor is large enough
        if softmax_lse is not None:
            assert (
                softmax_lse.shape[0] >= batch
            ), f"softmax_lse.shape[0]={softmax_lse.shape[0]} must be >= batch={batch}"
            assert (
                softmax_lse.shape[1] >= nheads_q
            ), f"softmax_lse.shape[1]={softmax_lse.shape[1]} must be >= nheads_q={nheads_q}"
            assert (
                softmax_lse.shape[2] >= seqlen_q
            ), f"softmax_lse.shape[2]={softmax_lse.shape[2]} must be >= seqlen_q={seqlen_q}"
            assert (
                softmax_lse.dtype == torch.float32
            ), f"softmax_lse must be float32, got {softmax_lse.dtype}"
            assert (
                softmax_lse.device == q.device
            ), f"softmax_lse must be on same device as q"

        # strides
        stride_qb, stride_qm, stride_qh, stride_qd = map_dims(q.stride(), bshd)
        stride_kb, stride_kn, stride_kh, stride_kd = map_dims(k.stride(), bshd)
        stride_vb, stride_vn, stride_vh, stride_vd = map_dims(v.stride(), bshd)
        stride_ob, stride_om, stride_oh, stride_od = map_dims(o.stride(), bshd)
        stride_lse_z, stride_lse_h, stride_lse_m = (
            softmax_lse.stride() if softmax_lse is not None else (0, 0, 0)
        )

    # apply rotary embeddings
    if rotary_cos is not None and rotary_sin is not None:
        raise NotImplementedError("Rotary embeddings prefill are not implemented yet.")
    else:
        assert o.dtype in [
            torch.float16,
            torch.bfloat16,
            torch.float32,
        ], f"Output tensor o must be fp16, bf16, or fp32 when using fp8, got {o.dtype}"

    stride_q_descale_z, stride_q_descale_h, stride_q_descale_blk = q_descale.stride()
    stride_k_descale_z, stride_k_descale_h, stride_k_descale_blk = k_descale.stride()

    if V_QUANT_SCHEME == 1:
        stride_v_descale_z, stride_v_descale_h, _ = v_descale.stride()
    else:
        stride_v_descale_z = stride_v_descale_h = 0

    # check features
    use_sliding_window = window_size_left != -1 or window_size_right != -1
    use_alibi, (stride_az, stride_ah) = (
        (True, alibi_slopes.stride()) if alibi_slopes is not None else (False, (0, 0))
    )
    # NOTE: a large bias tensor leads to overflow during pointer arithmetic
    if bias is not None:
        assert bias.numel() < 2**31

    # Get closest power of 2 over or equal to 32 for both QK and V dimensions
    padded_d_model_qk = 1 << (head_size_qk - 1).bit_length()
    padded_d_model_v = 1 << (head_size_v - 1).bit_length()
    # Smallest head_dim supported is 16. If smaller, the tile in the
    # kernel is padded - there is no padding in memory for any dims.
    padded_d_model_qk = max(padded_d_model_qk, 16)
    padded_d_model_v = max(padded_d_model_v, 16)

    # sd_mask assertions and strides
    if sd_mask is not None:
        assert dropout_p > 0.0 or return_scores, "sd_mask provided but not used"
        assert (
            sd_mask is not None
        ), "sd_mask must be provided when return_scores=True or dropout_p > 0"
        # Assert sd_mask tensor is large enough
        assert (
            sd_mask.shape[0] >= batch
        ), f"sd_mask.shape[0]={sd_mask.shape[0]} must be >= batch={batch}"
        assert (
            sd_mask.shape[1] >= nheads_q
        ), f"sd_mask.shape[1]={sd_mask.shape[1]} must be >= nheads_q={nheads_q}"
        assert (
            sd_mask.shape[2] >= max_seqlens_q
        ), f"sd_mask.shape[2]={sd_mask.shape[2]} must be >= max_seqlens_q={max_seqlens_q}"
        assert (
            sd_mask.shape[3] >= max_seqlens_k
        ), f"sd_mask.shape[3]={sd_mask.shape[3]} must be >= max_seqlens_k={max_seqlens_k}"
        assert sd_mask.device == q.device, f"sd_mask must be on same device as q"

        stride_sz, stride_sh, stride_sm, stride_sn = (
            sd_mask.stride(0),
            sd_mask.stride(1),
            sd_mask.stride(2),
            sd_mask.stride(3),
        )
    else:
        stride_sz, stride_sh, stride_sm, stride_sn = (0, 0, 0, 0)

    if bias is not None:
        stride_bz, stride_bh, stride_bm, stride_bn = (
            bias.stride(0),
            bias.stride(1),
            bias.stride(2),
            bias.stride(3),
        )
    else:
        stride_bz, stride_bh, stride_bm, stride_bn = (0, 0, 0, 0)

    return_lse = True if softmax_lse is not None else False

    # launch kernel
    grid = lambda META: (batch, nheads_q, triton.cdiv(max_seqlens_q, META["BLOCK_M"]))
    config = get_fwd_configs(False)
    attn_fwd[grid](
        q,
        k,
        v,
        bias,
        q_descale,
        k_descale,
        v_descale,
        FP8_MAX,
        stride_q_descale_z,
        stride_q_descale_h,
        stride_q_descale_blk,
        stride_k_descale_z,
        stride_k_descale_h,
        stride_k_descale_blk,
        stride_v_descale_z,
        stride_v_descale_h,
        softmax_lse,
        o,
        sd_mask,
        alibi_slopes,
        stride_qb,
        stride_qh,
        stride_qm,
        stride_qd,
        stride_kb,
        stride_kh,
        stride_kn,
        stride_kd,
        stride_vb,
        stride_vh,
        stride_vn,
        stride_vd,
        stride_ob,
        stride_oh,
        stride_om,
        stride_od,
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
        seqused_k,  # Pass seqused tensors
        V_QUANT_SCHEME=V_QUANT_SCHEME,
        dropout_p=dropout_p,
        philox_seed=philox_seed,
        philox_offset_base=philox_offset,
        RETURN_LSE=return_lse,
        HQ=nheads_q,
        HK=nheads_k,
        ACTUAL_BLOCK_DMODEL_QK=head_size_qk,
        ACTUAL_BLOCK_DMODEL_V=head_size_v,
        MAX_SEQLENS_Q=max_seqlens_q,
        MAX_SEQLENS_K=max_seqlens_k,
        SM_SCALE=sm_scale,
        IS_CAUSAL=causal,
        USE_SLIDING_WINDOW=use_sliding_window,
        WINDOW_SIZE_LEFT=window_size_left,
        WINDOW_SIZE_RIGHT=window_size_right,
        IS_VARLEN=IS_VARLEN,
        BLOCK_DMODEL_QK=padded_d_model_qk,
        BLOCK_DMODEL_V=padded_d_model_v,
        USE_BIAS=False if bias is None else True,
        USE_ALIBI=use_alibi,
        ENABLE_DROPOUT=dropout_p > 0.0,
        USE_EXP2=use_exp2,
        RETURN_SCORES=return_scores,
        USE_SEQUSED=(seqused_q is not None or seqused_k is not None),
        **config,
    )


def compute_k_smoothing_factors(
    k: torch.Tensor, tensor_layout: str = "NHD"
) -> torch.Tensor:
    """
    Compute per-channel smoothing factors for K tensor following SageAttention approach.

    This computes the mean across the sequence dimension for each channel (head_dim)
    to reduce outliers before quantization, improving INT8 accuracy.

    Args:
        k: Key tensor with shape (B, kv_len, H, head_dim) for NHD layout
           or (B, H, kv_len, head_dim) for HND layout
        tensor_layout: Either "NHD" or "HND"

    Returns:
        k_smooth: Smoothing factors with shape matching k, computed as per-channel mean
    """
    if tensor_layout == "NHD":
        # k shape: [B, kv_len, H, head_dim]
        # Compute mean across sequence dimension (dim=1), keep dims for broadcasting
        k_mean = k.mean(dim=1, keepdim=True)  # [B, 1, H, head_dim]
    elif tensor_layout == "HND":
        # k shape: [B, H, kv_len, head_dim]
        # Compute mean across sequence dimension (dim=2), keep dims for broadcasting
        k_mean = k.mean(dim=2, keepdim=True)  # [B, H, 1, head_dim]
    else:
        raise ValueError(f"Unknown tensor layout: {tensor_layout}")

    return k_mean


def sage_quant(
    q,
    k,
    v,
    FP8_TYPE,
    FP8_MAX,
    km=None,
    BLKQ=128,
    BLKK=64,
    sm_scale=None,
    tensor_layout="NHD",
    smooth_k=True,
):
    """
    Quantize Q and K tensors to INT8 with per-block scaling.

    Args:
        q: Query tensor
        k: Key tensor
        km: Optional pre-computed K smoothing factors (if None and smooth_k=True, will be computed)
        BLKQ: Block size for Q quantization
        BLKK: Block size for K quantization
        sm_scale: Softmax scale factor (defaults to head_dim^-0.5)
        tensor_layout: Either "NHD" or "HND"
        smooth_k: Whether to apply SageAttention-style smoothing to K tensor (default: True)

    Returns:
        q_int8: Quantized Q tensor
        q_scale: Per-block scales for Q
        k_int8: Quantized K tensor
        k_scale: Per-block scales for K
        k_smooth: K smoothing factors applied (or None if smooth_k=False)
    """
    q_int8 = torch.empty(q.shape, dtype=torch.int8, device=q.device)
    k_int8 = torch.empty(k.shape, dtype=torch.int8, device=k.device)
    v_fp8 = torch.empty(v.shape, dtype=FP8_TYPE, device=v.device)

    # Apply K tensor smoothing following SageAttention approach
    k_smooth = None
    if smooth_k:
        if km is None:
            # TOOD maybe we turn this into a kernel?
            km = compute_k_smoothing_factors(k, tensor_layout)
            k_smooth = km
        k = k - km

    if tensor_layout == "HND":
        b, h_qo, qo_len, head_dim = q.shape
        _, h_kv, kv_len, _ = k.shape

        stride_bz_q, stride_h_q, stride_seq_q = q.stride(0), q.stride(1), q.stride(2)
        stride_bz_k, stride_h_k, stride_seq_k = k.stride(0), k.stride(1), k.stride(2)

    elif tensor_layout == "NHD":
        b, qo_len, h_qo, head_dim = q.shape
        _, kv_len, h_kv, _ = k.shape

        stride_bz_q, stride_h_q, stride_seq_q = q.stride(0), q.stride(2), q.stride(1)
        stride_bz_k, stride_h_k, stride_seq_k = k.stride(0), k.stride(2), k.stride(1)
    else:
        raise ValueError(f"Unknown tensor layout: {tensor_layout}")
    Q_NUM_BLKS = (qo_len + BLKQ - 1) // BLKQ
    K_NUM_BLKS = (kv_len + BLKK - 1) // BLKK

    q_scale = torch.empty((b, h_qo, Q_NUM_BLKS), device=q.device, dtype=torch.float32)
    k_scale = torch.empty((b, h_kv, K_NUM_BLKS), device=q.device, dtype=torch.float32)
    if V_QUANT_SCHEME == 0:
        v_scale = torch.empty(
            (b, h_kv, K_NUM_BLKS), device=v.device, dtype=torch.float32
        )
    else:
        v_scale = torch.empty((b, h_kv, head_dim), device=v.device, dtype=torch.float32)

    if sm_scale is None:
        sm_scale = head_dim**-0.5

    q_task_count = b * h_qo * Q_NUM_BLKS
    k_task_count = b * h_kv * K_NUM_BLKS
    if V_QUANT_SCHEME == 0:
        v_task_count = b * h_kv * K_NUM_BLKS
    else:
        v_task_count = b * h_kv * head_dim

    grid = (q_task_count + k_task_count + v_task_count,)

    # call sage_quant_kernel
    sage_quant_kernel[grid](
        q,
        q_int8,
        q_scale,
        k,
        k_int8,
        k_scale,
        v,
        v_fp8,
        v_scale,
        stride_bz_q,
        stride_h_q,
        stride_seq_q,
        stride_bz_k,
        stride_h_k,
        stride_seq_k,
        q_scale.stride(0),
        q_scale.stride(1),
        k_scale.stride(0),
        k_scale.stride(1),
        v_scale.stride(0),
        v_scale.stride(1),
        (sm_scale * 1.4426950408889634),
        q_task_count,
        k_task_count,
        b,
        h_qo,
        h_kv,
        Q_NUM_BLKS,
        K_NUM_BLKS,
        qo_len,
        kv_len,
        triton.next_power_of_2(kv_len),
        FP8_MAX=FP8_MAX,
        INT8_MAX=torch.iinfo(q_int8.dtype).max,
        D=head_dim,
        BLK_Q=BLKQ,
        BLK_K=BLKK,
        V_QUANT_SCHEME=V_QUANT_SCHEME,
    )

    return q_int8, q_scale, k_int8, k_scale, v_fp8, v_scale, k_smooth


@triton.jit
def sage_quant_kernel(
    Q_Input,
    Q_Output,
    Q_Scale,
    K_Input,
    K_Output,
    K_Scale,
    V_Input,
    V_Output,
    V_Scale,
    stride_qz,
    stride_qh,
    stride_qn,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_qsz,
    stride_qsh,
    stride_ksz,
    stride_ksh,
    stride_vsz,
    stride_vsh,
    sm_scale,
    q_task_count,
    k_task_count,
    BATCH,
    Q_HEAD,
    K_HEAD,
    Q_NUM_BLKS,
    K_NUM_BLKS,
    SEQLEN_Q,
    SEQLEN_K,
    SEQLEN_K_PADDED: tl.constexpr,
    FP8_MAX: tl.constexpr,
    INT8_MAX: tl.constexpr,
    D: tl.constexpr,
    BLK_Q: tl.constexpr,
    BLK_K: tl.constexpr,
    V_QUANT_SCHEME: tl.constexpr,
):
    pid = tl.program_id(0)

    offs_blk_q = tl.arange(0, BLK_Q)
    offs_blk_k = tl.arange(0, BLK_K)
    offs_d = tl.arange(0, D)

    if pid < q_task_count:
        # here we do Q
        off_blk, off_h, off_b = pid_grid_3d(pid, Q_NUM_BLKS, Q_HEAD, BATCH)
        offs_qn = off_blk * BLK_Q + offs_blk_q

        q_offs = (
            off_b * stride_qz
            + off_h * stride_qh
            + offs_qn[:, None] * stride_qn
            + offs_d[None, :]
        )

        q_input_ptrs = Q_Input + q_offs
        q_output_ptrs = Q_Output + q_offs
        q_scale_ptrs = Q_Scale + off_b * stride_qsz + off_h * stride_qsh + off_blk

        _general_quant_kernel(
            q_input_ptrs,
            q_output_ptrs,
            q_scale_ptrs,
            INT8_MAX,
            offs_qn[:, None] < SEQLEN_Q,
            sm_scale,
        )
    elif pid >= q_task_count and pid < q_task_count + k_task_count:
        # here we do K
        _pid = pid - q_task_count
        off_blk, off_h, off_b = pid_grid_3d(_pid, K_NUM_BLKS, K_HEAD, BATCH)

        offs_kn = off_blk * BLK_K + offs_blk_k

        k_offs = (
            off_b * stride_kz
            + off_h * stride_kh
            + offs_kn[:, None] * stride_kn
            + offs_d[None, :]
        )

        k_input_ptrs = K_Input + k_offs
        k_output_ptrs = K_Output + k_offs
        k_scale_ptrs = K_Scale + off_b * stride_ksz + off_h * stride_ksh + off_blk

        _general_quant_kernel(
            k_input_ptrs,
            k_output_ptrs,
            k_scale_ptrs,
            INT8_MAX,
            offs_kn[:, None] < SEQLEN_K,
        )
    else:
        # V
        _pid = pid - (q_task_count + k_task_count)
        if V_QUANT_SCHEME == 0:
            off_blk, off_h, off_b = pid_grid_3d(_pid, K_NUM_BLKS, K_HEAD, BATCH)
            offs_kn = off_blk * BLK_K + offs_blk_k

            v_offs = (
                off_b * stride_kz
                + off_h * stride_kh
                + offs_kn[:, None] * stride_kn
                + offs_d[None, :]
            )

            v_input_ptrs = V_Input + v_offs
            v_output_ptrs = V_Output + v_offs
            v_scale_ptrs = V_Scale + off_b * stride_vsz + off_h * stride_vsh + off_blk
            _general_quant_kernel(
                v_input_ptrs,
                v_output_ptrs,
                v_scale_ptrs,
                FP8_MAX,
                mask=None,
            )
        else:
            off_d, off_h, off_b = pid_grid_3d(_pid, D, K_HEAD, BATCH)
            offs_k = tl.arange(0, SEQLEN_K_PADDED)

            v_offs = off_b * stride_kz + off_h * stride_kh + offs_k * stride_kn + off_d

            v_input_ptrs = V_Input + v_offs
            v_output_ptrs = V_Output + v_offs

            v_scale_ptrs = V_Scale + off_b * stride_vsz + off_h * stride_vsh + off_d
            _general_quant_kernel(
                v_input_ptrs, v_output_ptrs, v_scale_ptrs, FP8_MAX, offs_k < SEQLEN_K
            )


@triton.jit
def _general_quant_kernel(
    input_ptrs, output_ptrs, scale_ptrs, DTYPE_MAX, mask, sm_scale=None
):
    x = tl.load(input_ptrs, mask=mask, other=0.0)
    x = x.to(tl.float32)
    if sm_scale is not None:
        x *= sm_scale
    scale = tl.max(tl.abs(x)) / DTYPE_MAX
    x_quant = x / scale
    if output_ptrs.dtype.element_ty == tl.int8:
        x_quant += 0.5 * tl.where(x_quant >= 0, 1, -1)
    x_quant = x_quant.to(output_ptrs.dtype.element_ty)
    tl.store(output_ptrs, x_quant, mask=mask)
    tl.store(scale_ptrs, scale)



def quantize_v_fp8(
    v: torch.Tensor, FP8_MAX: float, BLKK: int, tensor_layout: str = "NHD"
):
    # call the corresponding quantization function with the V_QUANT_SCHEME

    if V_QUANT_SCHEME == 0:
        return _v_per_block_fp8(v, FP8_MAX, BLKK, tensor_layout=tensor_layout)
    elif V_QUANT_SCHEME == 1:
        return _v_per_channel_fp8(v, FP8_MAX, tensor_layout=tensor_layout)


def _v_per_block_fp8(v, FP8_MAX, BLKK=64, tensor_layout="NHD"):
    """
    Quantize V tensor to FP8 per-block scales.

    Args:
        v: Input V tensor of shape (B, N, H_kv, D_head
              or (B, H_kv, N, D_head) depending on tensor_layout.
        FP8_MAX: Maximum representable value in FP8 format.
        BLKK: Block size for quantization.
        tensor_layout: Layout of the tensor, either "HND" or "NHD".

    Returns:
        v_fp8: Quantized V tensor in FP8 format.
        v_scale: Scale factors for each block.
    """
    v_fp8 = torch.empty(v.shape, dtype=torch.torch.float8_e4m3fnuz, device=v.device)

    # Apply K tensor smoothing following SageAttention approach

    if tensor_layout == "HND":
        b, h_kv, kv_len, head_dim = v.shape

        stride_bz_k, stride_h_k, stride_seq_k = v.stride(0), v.stride(1), v.stride(2)
        stride_bz_ko, stride_h_ko, stride_seq_ko = (
            v_fp8.stride(0),
            v_fp8.stride(1),
            v_fp8.stride(2),
        )
    elif tensor_layout == "NHD":
        b, kv_len, h_kv, head_dim = v.shape

        stride_bz_k, stride_h_k, stride_seq_k = v.stride(0), v.stride(2), v.stride(1)
        stride_bz_ko, stride_h_ko, stride_seq_ko = (
            v_fp8.stride(0),
            v_fp8.stride(2),
            v_fp8.stride(1),
        )
    else:
        raise ValueError(f"Unknown tensor layout: {tensor_layout}")

    v_scale = torch.empty(
        (b, h_kv, (kv_len + BLKK - 1) // BLKK), device=v.device, dtype=torch.float32
    )

    grid = ((kv_len + BLKK - 1) // BLKK, h_kv, b)
    quant_per_block_fp8_kernel[grid](
        v,
        v_fp8,
        v_scale,
        kv_len,
        stride_bz_k,
        stride_h_k,
        stride_seq_k,
        stride_bz_ko,
        stride_h_ko,
        stride_seq_ko,
        v_scale.stride(0),
        v_scale.stride(1),
        FP8_MAX=FP8_MAX,
        C=head_dim,
        BLK=BLKK,
    )

    return v_fp8, v_scale


@triton.jit
def quant_per_block_fp8_kernel(
    Input,
    Output,
    Scale,
    L,
    stride_iz,
    stride_ih,
    stride_in,
    stride_oz,
    stride_oh,
    stride_on,
    stride_sz,
    stride_sh,
    FP8_MAX: tl.constexpr,
    C: tl.constexpr,
    BLK: tl.constexpr,
):
    off_blk = tl.program_id(0)
    off_h = tl.program_id(1)
    off_b = tl.program_id(2)

    offs_n = off_blk * BLK + tl.arange(0, BLK)
    offs_k = tl.arange(0, C)

    input_ptrs = (
        Input
        + off_b * stride_iz
        + off_h * stride_ih
        + offs_n[:, None] * stride_in
        + offs_k[None, :]
    )
    output_ptrs = (
        Output
        + off_b * stride_oz
        + off_h * stride_oh
        + offs_n[:, None] * stride_on
        + offs_k[None, :]
    )
    scale_ptrs = Scale + off_b * stride_sz + off_h * stride_sh + off_blk

    x = tl.load(input_ptrs, mask=offs_n[:, None] < L)
    x = x.to(tl.float32)
    scale = tl.max(tl.abs(x)) / FP8_MAX
    x_fp8 = x / scale
    # x_fp8 += 0.5 * tl.where(x_fp8 >= 0, 1, -1)
    x_fp8 = x_fp8.to(Scale.type.element_ty)
    tl.store(output_ptrs, x_fp8, mask=offs_n[:, None] < L)
    tl.store(scale_ptrs, scale)


def _v_per_channel_fp8(v, FP8_MAX, tensor_layout="NHD"):
    """
    Quantize V tensor to FP8 per-channel scales.

    Args:
        v: Input V tensor of shape (B, N, H_kv, D_head
                or (B, H_kv, N, D_head) depending on tensor_layout.
        FP8_MAX: Maximum representable value in FP8 format.
        BLKK: Block size for quantization.
        tensor_layout: Layout of the tensor, either "HND" or "NHD".

    Returns:
        v_fp8: Quantized V tensor in FP8 format.
        v_scale: Scale factors for each channel.
    """
    v_fp8 = torch.empty(v.shape, dtype=torch.torch.float8_e4m3fnuz, device=v.device)

    # Apply K tensor smoothing following SageAttention approach

    if tensor_layout == "HND":
        b, h_kv, kv_len, head_dim = v.shape

        stride_bz_k, stride_h_k, stride_seq_k = v.stride(0), v.stride(1), v.stride(2)
        stride_bz_ko, stride_h_ko, stride_seq_ko = (
            v_fp8.stride(0),
            v_fp8.stride(1),
            v_fp8.stride(2),
        )
    elif tensor_layout == "NHD":
        b, kv_len, h_kv, head_dim = v.shape

        stride_bz_k, stride_h_k, stride_seq_k = v.stride(0), v.stride(2), v.stride(1)
        stride_bz_ko, stride_h_ko, stride_seq_ko = (
            v_fp8.stride(0),
            v_fp8.stride(2),
            v_fp8.stride(1),
        )
    else:
        raise ValueError(f"Unknown tensor layout: {tensor_layout}")

    v_scale = torch.empty((b, h_kv, head_dim), device=v.device, dtype=torch.float32)

    grid = (head_dim, h_kv, b)
    quant_per_channel_fp8_kernel[grid](
        v,
        v_fp8,
        v_scale,
        kv_len,
        triton.next_power_of_2(kv_len),
        stride_bz_k,
        stride_h_k,
        stride_seq_k,
        stride_bz_ko,
        stride_h_ko,
        stride_seq_ko,
        v_scale.stride(0),
        v_scale.stride(1),
        FP8_MAX=FP8_MAX,
    )

    return v_fp8, v_scale


@triton.jit
def quant_per_channel_fp8_kernel(
    Input,
    Output,
    Scale,
    L,
    L_PADDED: tl.constexpr,
    stride_iz,
    stride_ih,
    stride_in,
    stride_oz,
    stride_oh,
    stride_on,
    stride_sz,
    stride_sh,
    FP8_MAX: tl.constexpr,
):

    off_d = tl.program_id(0)
    off_h = tl.program_id(1)
    off_b = tl.program_id(2)

    offs_l = tl.arange(0, L_PADDED)

    input_ptrs = (
        Input + off_b * stride_iz + off_h * stride_ih + offs_l * stride_in + off_d
    )
    output_ptrs = (
        Output + off_b * stride_oz + off_h * stride_oh + offs_l * stride_on + off_d
    )
    scale_ptrs = Scale + off_b * stride_sz + off_h * stride_sh + off_d

    x = tl.load(input_ptrs, mask=offs_l < L)
    x = x.to(tl.float32)
    scale = tl.max(tl.abs(x)) / FP8_MAX
    x_fp8 = x / scale
    # x_fp8 += 0.5 * tl.where(x_fp8 >= 0, 1, -1)
    x_fp8 = x_fp8.to(Scale.type.element_ty)
    tl.store(output_ptrs, x_fp8, mask=offs_l < L)
    tl.store(scale_ptrs, scale)