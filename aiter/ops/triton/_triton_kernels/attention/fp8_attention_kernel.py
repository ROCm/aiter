# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Triton implementation of Flash Attention v2 with FP8 block-scaling support.

Based on the Flash Attention v2 algorithm from Tri Dao
(https://tridao.me/publications/flash2/flash2.pdf)
Credits: OpenAI kernel team, AMD ML Frameworks Triton team

Features supported:

1) Fwd with causal masking
2) Any sequence lengths without padding (currently fwd kernel only)
3) Support for different sequence lengths for q and k
4) Nested tensor API currently does not support dropout or bias.

Not currently supported:

1) Non power of two head dims

"""

import os

import triton
import triton.language as tl

# Seed the RNG so we get reproducible results for testing.
philox_seed: tl.constexpr = 0x1BF52
philox_offset: tl.constexpr = 0x1D4B42

AUTOTUNE = os.environ.get("AITER_TRITON_AMD_AUTOTUNE", "0").lower() in (
    "1",
    "true",
    "yes",
)
DEBUG = os.environ.get("AITER_ATTENTION_TRITON_AMD_DEBUG", "0").lower() in (
    "1",
    "true",
    "yes",
)
PERF = os.environ.get("AITER_ATTENTION_TRITON_AMD_PERF", "0").lower() in (
    "1",
    "true",
    "yes",
)
USE_FP8E5M2_BWD = os.environ.get("AITER_ATTENTION_USE_FP8E5M2_BWD", "0").lower() in (
    "1",
    "true",
    "yes",
)

FIXED_BLOCK_M = 64
FIXED_BLOCK_N = 64


def get_shape_from_layout(
    q,
    k,
    v,
    layout,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    max_seqlen_q=None,
    max_seqlen_k=None,
):
    if layout == "bhsd":
        batch_q, nheads_q, max_seqlen_q, head_size_q = q.shape
        batch_k, nheads_k, max_seqlen_k, head_size_k = k.shape
        _batch_v, _nheads_v, _max_seqlen_v, head_size_v = v.shape
    elif layout == "bshd":
        batch_q, max_seqlen_q, nheads_q, head_size_q = q.shape
        batch_k, max_seqlen_k, nheads_k, head_size_k = k.shape
        _batch_v, _max_seqlen_v, _nheads_v, head_size_v = v.shape
    elif layout == "thd":
        batch_q, nheads_q, head_size_q = (
            len(cu_seqlens_q) - 1,
            q.shape[1],
            q.shape[2],
        )
        batch_k, nheads_k, head_size_k = (
            len(cu_seqlens_k) - 1,
            k.shape[1],
            k.shape[2],
        )
        _batch_v, _max_seqlen_v, _nheads_v, head_size_v = (
            len(cu_seqlens_k) - 1,
            max_seqlen_k,
            v.shape[1],
            v.shape[2],
        )
    else:
        assert False, "Got unsupported layout."

    # assert
    assert batch_q == batch_k
    assert head_size_q == head_size_k

    return (
        batch_q,
        nheads_q,
        nheads_k,
        head_size_q,
        head_size_v,
        max_seqlen_q,
        max_seqlen_k,
    )


def get_strides_from_layout(q, layout):
    if layout == "thd":
        q_strides = (0, q.stride(1), q.stride(0), q.stride(2))
    elif layout == "bhsd":
        q_strides = (q.stride(0), q.stride(1), q.stride(2), q.stride(3))
    elif layout == "bshd":
        q_strides = (q.stride(0), q.stride(2), q.stride(1), q.stride(3))
    else:
        assert False, "Got unsupported layout."
    return q_strides


def get_padded_headsize(size):
    # Get closest power of 2 over or equal to 32.
    padded_d_model = 1 << (size - 1).bit_length()
    # Smallest head_dim supported is 16. If smaller, the tile in the
    # kernel is padded - there is no padding in memory for any dims.
    padded_d_model = max(padded_d_model, 16)
    return padded_d_model


def get_input_shapes():
    cases = [(max(1, 2 ** (16 - i)), 1, 2**i, 16, 1, 128) for i in range(8, 18)] + [
        (max(1, 2 ** (16 - i)), 1, 2**i, 16, 2, 128) for i in range(8, 18)
    ]
    return cases


def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"


def is_cdna():
    return is_hip() and triton.runtime.driver.active.get_current_target().arch in (
        "gfx950",
        "gfx940",
        "gfx941",
        "gfx942",
        "gfx90a",
        "gfx908",
    )


def is_cdna4():
    target = triton.runtime.driver.active.get_current_target()
    return target is not None and target.backend == "hip" and target.arch == "gfx950"


def is_rdna():
    return is_hip() and triton.runtime.driver.active.get_current_target().arch in (
        "gfx1030",
        "gfx1100",
        "gfx1101",
        "gfx1102",
        "gfx1200",
        "gfx1201",
    )


def get_tl_f8_bwd_dtype():
    if USE_FP8E5M2_BWD:
        return tl.float8e5b16 if is_hip() and not is_cdna4() else tl.float8e5
    return tl.float8e4b8 if is_hip() and not is_cdna4() else tl.float8e4nv


@triton.jit
def cdiv_fn(x, y):
    return (x + y - 1) // y


@triton.jit
def dropout_offsets(philox_seed, philox_offset, dropout_p, m, n, stride):
    ms = tl.arange(0, m)
    ns = tl.arange(0, n)
    return philox_offset + ms[:, None] * stride + ns[None, :]


@triton.jit
def dropout_rng(philox_seed, philox_offset, dropout_p, m, n, stride):
    rng_offsets = dropout_offsets(
        philox_seed, philox_offset, dropout_p, m, n, stride
    ).to(tl.uint32)
    # TODO: use tl.randint for better performance
    return tl.rand(philox_seed, rng_offsets)


@triton.jit
def dropout_mask(philox_seed, philox_offset, dropout_p, m, n, stride):
    rng_output = dropout_rng(philox_seed, philox_offset, dropout_p, m, n, stride)
    rng_keep = rng_output > dropout_p
    return rng_keep


# Convenience function to load with optional boundary checks.
# "First" is the major dim, "second" is the minor dim.
@triton.jit
def load_fn(ptrs, offset_first, offset_second, boundary_first, boundary_second):
    if offset_first is not None and offset_second is not None:
        mask = (offset_first[:, None] < boundary_first) & (
            offset_second[None, :] < boundary_second
        )
        tensor = tl.load(ptrs, mask=mask, other=0.0)
    elif offset_first is not None:
        mask = offset_first[:, None] < boundary_first
        tensor = tl.load(ptrs, mask=mask, other=0.0)
    elif offset_second is not None:
        mask = offset_second[None, :] < boundary_second
        tensor = tl.load(ptrs, mask=mask, other=0.0)
    else:
        tensor = tl.load(ptrs)
    return tensor


@triton.jit
def compute_alibi_block(
    alibi_slope, seqlen_q, seqlen_k, offs_m, offs_n, transpose=False
):
    # when seqlen_k and seqlen_q are different we want the diagonal to stick to the bottom right of the attention matrix
    # for casual mask we want something like this where (1 is kept and 0 is masked)
    # seqlen_q = 2 and seqlen_k = 5
    #   1 1 1 1 0
    #   1 1 1 1 1
    # seqlen_q = 5 and seqlen_k = 2
    #        0 0
    #        0 0
    #        0 0
    #        1 0
    #        1 1
    # for alibi the diagonal is 0 indicating no penalty for attending to that spot and increasing penalty for attending further from the diagonal
    # e.g. alibi_slope = 1, seqlen_q = 2, seqlen_k = 5, offs_m = [0, 1, 2, 3], offs_n = [0, 1, 2, 3, 4], transpose = False
    # 1. offs_m[:,None] = [[0],
    #                       [1],
    # 2. offs_m[:,None] + seqlen_k = [[5],
    #                                  [6],
    # 3. offs_m[:,None] + seqlen_k - seqlen_q = [[3],
    #                                             [4],
    # 4. offs_m[:,None] + seqlen_k - seqlen_q - offs_n[None,:] = [[3], - [[0, 1, 2, 3, 4]] =  [[ 3, 2, 1, 0,-1],
    #                                                            [4],                           [ 4, 3, 2, 1, 0]]
    # 5. -1 * alibi_slope * tl.abs(relative_pos_block) = [[ -3, -2, -1, 0,-1],
    #                                                     [ -4, -3, -2, -1, 0]],
    relative_pos_block = offs_m[:, None] + seqlen_k - seqlen_q - offs_n[None, :]
    alibi_block = -1 * alibi_slope * tl.abs(relative_pos_block)
    if transpose:
        return alibi_block.T
    else:
        return alibi_block


@triton.jit
def _attn_fwd_inner(
    acc,
    l_i,
    m_i,
    q,
    q_descale,
    k_descale,
    p_scale: tl.constexpr,
    USE_FP8: tl.constexpr,
    k_ptrs,
    v_ptrs,
    bias_ptrs,
    stride_kn,
    stride_vk,
    stride_bn,
    start_m,
    actual_seqlen_k,
    actual_seqlen_q,
    dropout_p,
    philox_seed,
    batch_philox_offset,
    exp_scores_ptrs,
    block_min,
    block_max,
    offs_n_causal,
    masked_blocks,
    n_extra_tokens,
    alibi_slope,
    score_ptrs,
    scores_scaled_shifted_ptrs,
    IS_CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL_QK: tl.constexpr,
    BLOCK_DMODEL_V: tl.constexpr,
    BLOCK_N: tl.constexpr,
    OFFS_M: tl.constexpr,
    OFFS_N: tl.constexpr,
    PRE_LOAD_V: tl.constexpr,
    MASK_STEPS: tl.constexpr,
    ENABLE_DROPOUT: tl.constexpr,
    PADDED_HEAD_QK: tl.constexpr,
    PADDED_HEAD_V: tl.constexpr,
    ACTUAL_BLOCK_DMODEL_QK: tl.constexpr,
    ACTUAL_BLOCK_DMODEL_V: tl.constexpr,
    SM_SCALE: tl.constexpr,
    USE_EXP2: tl.constexpr,
    RETURN_SCORES: tl.constexpr,
):
    if USE_EXP2:
        RCP_LN2: tl.constexpr = 1.4426950408889634

    # loop over k, v, and update accumulator
    for start_n in range(block_min, block_max, BLOCK_N):
        # For padded blocks, we will overrun the tensor size if
        # we load all BLOCK_N. For others, the blocks are all within range.
        if USE_FP8:
            idx_block_n = tl.full([1], start_n // BLOCK_N, dtype=tl.int32)
            blk_k_descale = k_descale.gather(index=idx_block_n, axis=0)
        else:
            blk_k_descale = 1.0

        if MASK_STEPS:
            k_offs_n = start_n + tl.arange(0, BLOCK_N)
        else:
            k_offs_n = None
        k_offs_k = None if not PADDED_HEAD_QK else tl.arange(0, BLOCK_DMODEL_QK)
        k = load_fn(k_ptrs, k_offs_k, k_offs_n, ACTUAL_BLOCK_DMODEL_QK, actual_seqlen_k)
        v_offs_k = None if not PADDED_HEAD_V else tl.arange(0, BLOCK_DMODEL_V)
        if PRE_LOAD_V:
            # We can use the same offsets as k, just with dims transposed.
            v = load_fn(
                v_ptrs, k_offs_n, v_offs_k, actual_seqlen_k, ACTUAL_BLOCK_DMODEL_V
            )
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        # We start from end of seqlen_k so only the first iteration would need
        # to be checked for padding if it is not a multiple of block_n
        # TODO: This can be optimized to only be true for the padded block.
        if MASK_STEPS and (start_n + BLOCK_N == block_max) and (n_extra_tokens != 0):
            boundary_m = tl.full([BLOCK_M], actual_seqlen_k, dtype=tl.int32)
            size_n = start_n + OFFS_N[None, :]
            mask = size_n < boundary_m[:, None]
            qk = tl.where(mask, qk, float("-inf"))

        # -- compute qk ----
        qk += tl.dot(q, k, out_dtype=tl.float32, allow_tf32=False)

        if USE_FP8:
            qk_scaled = qk * (q_descale * blk_k_descale * SM_SCALE)
        else:
            qk_scaled = qk * SM_SCALE

        if RETURN_SCORES:
            score_mask = (OFFS_M[:, None] < actual_seqlen_q) & (
                (start_n + tl.arange(0, BLOCK_N))[None, :] < actual_seqlen_k
            )
            tl.store(score_ptrs, qk_scaled, mask=score_mask)

        if IS_CAUSAL:
            causal_boundary = start_n + offs_n_causal
            causal_mask = OFFS_M[:, None] >= causal_boundary[None, :]
            qk_scaled = tl.where(causal_mask, qk_scaled, float("-inf"))
        if bias_ptrs is not None:
            bias_offs_n = start_n + tl.arange(0, BLOCK_N) if MASK_STEPS else None
            bias = load_fn(
                bias_ptrs, OFFS_M, bias_offs_n, actual_seqlen_q, actual_seqlen_k
            )
            qk_scaled += bias

        if alibi_slope is not None:
            # Compute the global position of each token within the sequence
            global_m_positions = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
            global_n_positions = start_n + tl.arange(0, BLOCK_N)
            alibi_block = compute_alibi_block(
                alibi_slope,
                actual_seqlen_q,
                actual_seqlen_k,
                global_m_positions,
                global_n_positions,
            )
            qk_scaled += alibi_block
        # get max scores so far
        m_ij = tl.maximum(m_i, tl.max(qk_scaled, 1))

        # scale and subtract max
        q_shifted = qk_scaled - m_ij[:, None]
        if RETURN_SCORES:
            # NOTE: the returned score is not the same as the reference because we need to adjust as we find new maxes per block. We are not doing that
            scores_scaled_shifted_mask = (OFFS_M[:, None] < actual_seqlen_q) & (
                (start_n + tl.arange(0, BLOCK_N))[None, :] < actual_seqlen_k
            )
            tl.store(
                scores_scaled_shifted_ptrs, q_shifted, mask=scores_scaled_shifted_mask
            )

        # Compute scaled QK and softmax probabilities
        if USE_EXP2:
            p = tl.math.exp2(q_shifted * RCP_LN2)
        else:
            p = tl.math.exp(q_shifted)

        # CAVEAT: Must update l_ij before applying dropout
        l_ij = tl.sum(p, 1)
        if ENABLE_DROPOUT:
            philox_offset = (
                batch_philox_offset
                + start_m * BLOCK_M * actual_seqlen_k
                + start_n
                - BLOCK_N
            )
            keep = dropout_mask(
                philox_seed, philox_offset, dropout_p, BLOCK_M, BLOCK_N, actual_seqlen_k
            )
            if RETURN_SCORES:
                # NOTE: the returned score is not the same as the reference because we need to adjust as we find new maxes per block. We are not doing that
                exp_score_mask = (OFFS_M[:, None] < actual_seqlen_q) & (
                    (start_n + tl.arange(0, BLOCK_N))[None, :] < actual_seqlen_k
                )
                tl.store(exp_scores_ptrs, tl.where(keep, p, -p), mask=exp_score_mask)
            p = tl.where(keep, p, 0.0)
        elif RETURN_SCORES:
            # NOTE: the returned score is not the same as the reference because we need to adjust as we find new maxes per block. We are not doing that
            exp_score_mask = (OFFS_M[:, None] < actual_seqlen_q) & (
                (start_n + tl.arange(0, BLOCK_N))[None, :] < actual_seqlen_k
            )
            tl.store(exp_scores_ptrs, p, mask=exp_score_mask)

        # -- update output accumulator --
        # alpha is an adjustment factor for acc and li as we loop and find new maxes
        # store the diff in maxes to adjust acc and li as we discover new maxes
        m_diff = m_i - m_ij
        if USE_EXP2:
            alpha = tl.math.exp2(m_diff * RCP_LN2)
        else:
            alpha = tl.math.exp(m_diff)
        acc = acc * alpha[:, None]
        if not PRE_LOAD_V:
            v = load_fn(
                v_ptrs, k_offs_n, v_offs_k, actual_seqlen_k, ACTUAL_BLOCK_DMODEL_V
            )
        # -- update m_i and l_i
        l_i = l_i * alpha + l_ij
        # update m_i and l_i
        m_i = m_ij
        if USE_FP8:
            p *= p_scale

        acc = tl.dot(p.to(v.dtype), v, acc=acc, allow_tf32=False, out_dtype=tl.float32)

        k_ptrs += BLOCK_N * stride_kn
        v_ptrs += BLOCK_N * stride_vk
        if bias_ptrs is not None:
            bias_ptrs += BLOCK_N * stride_bn
        if RETURN_SCORES:
            score_ptrs += BLOCK_N
            scores_scaled_shifted_ptrs += BLOCK_N
            exp_scores_ptrs += BLOCK_N
    return acc, l_i, m_i


def get_autotune_fwd_configs():
    return [
        triton.Config(
            {
                "PRE_LOAD_V": False,
            },
            num_stages=1,
            num_warps=4,
        ),
    ], [
        "IS_CAUSAL",
        "dropout_p",
        "MAX_SEQLENS_Q",
        "MAX_SEQLENS_K",
        "ACTUAL_BLOCK_DMODEL_QK",
        "ACTUAL_BLOCK_DMODEL_V",
        "VARLEN",
        "HQ",
        "HK",
        "USE_FP8",
    ]


autotune_fwd_configs, autotune_fwd_keys = get_autotune_fwd_configs()


@triton.autotune(
    configs=autotune_fwd_configs,
    key=autotune_fwd_keys,
)
@triton.jit
def attn_fwd(
    Q,
    K,
    V,
    bias,
    p_scale: tl.constexpr,
    q_descale_ptr,
    k_descale_ptr,
    v_scale_ptr,
    USE_FP8: tl.constexpr,
    SM_SCALE: tl.constexpr,
    LSE,
    Out,
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
    stride_qdescale_z: tl.constexpr,
    stride_qdescale_h: tl.constexpr,
    stride_qdescale_m: tl.constexpr,
    stride_kdescale_z: tl.constexpr,
    stride_kdescale_h: tl.constexpr,
    stride_kdescale_m: tl.constexpr,
    padded_kscale_block_num: tl.constexpr,
    cu_seqlens_q,
    cu_seqlens_k,
    dropout_p,
    philox_seed,
    philox_offset_base,
    scores,
    scores_scaled_shifted,
    exp_scores,
    alibi_slopes,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    ACTUAL_BLOCK_DMODEL_QK: tl.constexpr,
    ACTUAL_BLOCK_DMODEL_V: tl.constexpr,
    MAX_SEQLENS_Q: tl.constexpr,
    MAX_SEQLENS_K: tl.constexpr,
    VARLEN: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
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
):
    start_m = tl.program_id(0)
    off_h_q = tl.program_id(1)
    off_z = tl.program_id(2)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d_qk = tl.arange(0, BLOCK_DMODEL_QK)
    offs_d_v = tl.arange(0, BLOCK_DMODEL_V)

    # If MQA / GQA, set the K and V head offsets appropriately.
    GROUP_SIZE: tl.constexpr = HQ // HK
    if GROUP_SIZE != 1:
        off_h_k = off_h_q // GROUP_SIZE
    else:
        off_h_k = off_h_q

    PADDED_HEAD_QK: tl.constexpr = ACTUAL_BLOCK_DMODEL_QK != BLOCK_DMODEL_QK
    PADDED_HEAD_V: tl.constexpr = ACTUAL_BLOCK_DMODEL_V != BLOCK_DMODEL_V

    # we assume q and k has the same length
    if USE_FP8:
        actual_kscale_block_num = stride_kdescale_h
        kscale_mask = tl.arange(0, padded_kscale_block_num) < actual_kscale_block_num
        k_descale_offset = (
            k_descale_ptr
            + stride_kdescale_z * off_z
            + stride_kdescale_h * off_h_k
            + tl.arange(0, padded_kscale_block_num)
        )
        q_descale_offset = (
            q_descale_ptr
            + stride_qdescale_z * off_z
            + stride_qdescale_h * off_h_q
            + start_m
        )  #  + stride_qdescale_m * cu_seqlens_q

        k_descale = tl.load(k_descale_offset, mask=kscale_mask, other=1.0)
        q_descale = tl.load(q_descale_offset)
        v_scale = tl.load(v_scale_ptr)
    else:
        k_descale = 1.0
        q_descale = 1.0
        v_scale = 1.0

    acc_descale = 1.0 / (p_scale * v_scale)

    if VARLEN:
        cu_seqlens_q_start = tl.load(cu_seqlens_q + off_z)
        cu_seqlens_q_end = tl.load(cu_seqlens_q + off_z + 1)
        # print("cu_seqlens_q_start:", cu_seqlens_q_start)

        seqlen_q = cu_seqlens_q_end - cu_seqlens_q_start
        # We have a one-size-fits-all grid in id(0). Some seqlens might be too
        # small for all start_m so for those we return early.
        if start_m * BLOCK_M > seqlen_q:
            return
        cu_seqlens_k_start = tl.load(cu_seqlens_k + off_z)
        cu_seqlens_k_end = tl.load(cu_seqlens_k + off_z + 1)
        seqlen_k = cu_seqlens_k_end - cu_seqlens_k_start
    else:
        cu_seqlens_q_start = 0
        cu_seqlens_k_start = 0
        seqlen_q = MAX_SEQLENS_Q
        seqlen_k = MAX_SEQLENS_K

    # Now we compute whether we need to exit early due to causal masking.
    # This is because for seqlen_q > seqlen_k, M rows of the attn scores
    # are completely masked, resulting in 0s written to the output, and
    # inf written to LSE. We don't need to do any GEMMs in this case.
    # This block of code determines what N is, and if this WG is operating
    # on those M rows.
    n_blocks = cdiv_fn(seqlen_k, BLOCK_N)
    if IS_CAUSAL:
        # If seqlen_q == seqlen_k, the attn scores are a square matrix.
        # If seqlen_q != seqlen_k, attn scores are rectangular which means
        # the causal mask boundary is bottom right aligned, and ends at either
        # the top edge (seqlen_q < seqlen_k) or left edge.
        # This captures the decrease in n_blocks if we have a rectangular attn matrix
        n_blocks_seqlen = cdiv_fn(
            (start_m + 1) * BLOCK_M + seqlen_k - seqlen_q, BLOCK_N
        )
        # This is what adjusts the block_max for the current WG, only
        # if IS_CAUSAL. Otherwise we want to always iterate through all n_blocks
        n_blocks = min(n_blocks, n_blocks_seqlen)
        # If we have no blocks after adjusting for seqlen deltas, this WG is part of
        # the blocks that are all 0. We exit early.
        if n_blocks <= 0:
            o_offset = (
                Out
                + off_z * stride_oz
                + off_h_q * stride_oh
                + cu_seqlens_q_start * stride_om
            )
            o_ptrs = (
                o_offset + offs_m[:, None] * stride_om + offs_d_v[None, :] * stride_on
            )
            acc = tl.zeros([BLOCK_M, BLOCK_DMODEL_V], dtype=Out.type.element_ty)
            o_ptrs_mask = offs_m[:, None] < seqlen_q
            if PADDED_HEAD_V:
                o_ptrs_mask = o_ptrs_mask & (offs_d_v[None, :] < ACTUAL_BLOCK_DMODEL_V)
            # We still need to write 0s to the result
            tl.store(o_ptrs, acc, mask=o_ptrs_mask)
            # The tensor allocated for L is based on MAX_SEQLENS_Q as that is
            # statically known.
            l_offset = (
                LSE
                + off_z * stride_lse_z
                + off_h_q * stride_lse_h
                + cu_seqlens_q_start * stride_lse_m
            )
            l_ptrs = l_offset + offs_m * stride_lse_m

            l = tl.full([BLOCK_M], value=0.0, dtype=tl.float32)

            # mask_m_offsets = start_m + tl.arange(0, BLOCK_M)
            # lse_mask = mask_m_offsets < causal_start_idx
            # softmax_lse = tl.where(lse_mask, 0.0, softmax_lse)
            l_ptrs_mask = offs_m < MAX_SEQLENS_Q
            tl.store(l_ptrs, l, mask=l_ptrs_mask)
            # TODO: Should dropout and return encoded softmax be handled here too?
            return

    n_extra_tokens = 0
    # print("n_extra_tokens:", n_extra_tokens)
    # print("seqlen_k:", seqlen_k)
    # print("BLOCK_N:", BLOCK_N)
    # return
    if seqlen_k < BLOCK_N:
        n_extra_tokens = BLOCK_N - seqlen_k
    elif seqlen_k % BLOCK_N:
        n_extra_tokens = seqlen_k % BLOCK_N

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
        alibi_slope = tl.load(alibi_slopes + a_offset)
    else:
        alibi_slope = None

    if RETURN_SCORES:
        scores_offset = (
            scores
            + off_z * stride_sz
            + off_h_q * stride_sh
            + cu_seqlens_q_start * stride_sm
        )
        score_ptrs = (
            scores_offset + offs_m[:, None] * stride_sm + offs_n[None, :] * stride_sn
        )

        scores_scaled_shifted_offset = (
            scores_scaled_shifted
            + off_z * stride_sz
            + off_h_q * stride_sh
            + cu_seqlens_q_start * stride_sm
        )
        scores_scaled_shifted_ptrs = (
            scores_scaled_shifted_offset
            + offs_m[:, None] * stride_sm
            + offs_n[None, :] * stride_sn
        )

        exp_scores_offset = (
            exp_scores
            + off_z * stride_sz
            + off_h_q * stride_sh
            + cu_seqlens_q_start * stride_sm
        )
        exp_scores_ptrs = (
            exp_scores_offset
            + offs_m[:, None] * stride_sm
            + offs_n[None, :] * stride_sn
        )
    else:
        score_ptrs = None
        scores_scaled_shifted_ptrs = None
        exp_scores_ptrs = None

    if ENABLE_DROPOUT:
        off_hz = off_z * HQ + off_h_q
        batch_philox_offset = philox_offset_base + off_hz * seqlen_q * seqlen_k
    else:
        batch_philox_offset = 0
    # initialize pointer to m and l
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL_V], dtype=tl.float32)
    # Q is loaded once at the beginning and shared by all N blocks.
    q_ptrs_mask = offs_m[:, None] < seqlen_q
    if PADDED_HEAD_QK:
        q_ptrs_mask = q_ptrs_mask & (offs_d_qk[None, :] < ACTUAL_BLOCK_DMODEL_QK)

    q = tl.load(q_ptrs, mask=q_ptrs_mask, other=0.0)

    # Here we compute how many full and masked blocks we have.
    padded_block_k = n_extra_tokens != 0
    is_modulo_mn = not padded_block_k and (seqlen_q % BLOCK_M == 0)
    if IS_CAUSAL:
        # There are always at least BLOCK_M // BLOCK_N masked blocks.
        # Additionally there might be one more due to dissimilar seqlens.
        masked_blocks = BLOCK_M // BLOCK_N + (not is_modulo_mn)
    else:
        # Padding on Q does not need to be masked in the FA loop.
        masked_blocks = padded_block_k
    # if IS_CAUSAL, not is_modulo_mn does not always result in an additional block.
    # In this case we might exceed n_blocks so pick the min.

    masked_blocks = min(masked_blocks, n_blocks)
    n_full_blocks = n_blocks - masked_blocks
    block_min = 0
    block_max = n_blocks * BLOCK_N
    # Compute for full blocks. Here we set causal to false regardless of its actual
    # value because there is no masking. Similarly we do not need padding.

    if n_full_blocks > 0:
        block_max = (n_blocks - masked_blocks) * BLOCK_N
        acc, l_i, m_i = _attn_fwd_inner(
            acc,
            l_i,
            m_i,
            q,
            q_descale,
            k_descale,
            p_scale,
            USE_FP8,
            k_ptrs,
            v_ptrs,
            bias_ptrs,
            stride_kn,
            stride_vk,
            stride_bn,
            start_m,
            seqlen_k,
            seqlen_q,
            dropout_p,
            philox_seed,
            batch_philox_offset,
            exp_scores_ptrs,
            # _, _, offs_n_causal, masked_blocks, n_extra_tokens, _
            block_min,
            block_max,
            0,
            0,
            0,
            alibi_slope,
            score_ptrs,
            scores_scaled_shifted_ptrs,
            # IS_CAUSAL, ....
            False,
            BLOCK_M,
            BLOCK_DMODEL_QK,
            BLOCK_DMODEL_V,
            BLOCK_N,
            offs_m,
            offs_n,
            # _, MASK_STEPS, ...
            PRE_LOAD_V,
            False,
            ENABLE_DROPOUT,
            PADDED_HEAD_QK,
            PADDED_HEAD_V,
            ACTUAL_BLOCK_DMODEL_QK,
            ACTUAL_BLOCK_DMODEL_V,
            SM_SCALE,
            USE_EXP2=USE_EXP2,
            RETURN_SCORES=RETURN_SCORES,
        )
        block_min = block_max
        block_max = n_blocks * BLOCK_N

    # Remaining blocks, if any, are full / not masked.
    if masked_blocks > 0:
        if IS_CAUSAL:
            offs_n_causal = offs_n + (seqlen_q - seqlen_k)
        else:
            offs_n_causal = 0
        k_ptrs += n_full_blocks * BLOCK_N * stride_kn
        v_ptrs += n_full_blocks * BLOCK_N * stride_vk
        if USE_BIAS:
            bias_ptrs += n_full_blocks * BLOCK_N * stride_bn
        if RETURN_SCORES:
            score_ptrs += n_full_blocks * BLOCK_N
            scores_scaled_shifted_ptrs += n_full_blocks * BLOCK_N
            exp_scores_ptrs += n_full_blocks * BLOCK_N

        acc, l_i, m_i = _attn_fwd_inner(
            acc,
            l_i,
            m_i,
            q,
            q_descale,
            k_descale,
            p_scale,
            USE_FP8,
            k_ptrs,
            v_ptrs,
            bias_ptrs,
            stride_kn,
            stride_vk,
            stride_bn,
            start_m,
            seqlen_k,
            seqlen_q,
            dropout_p,
            philox_seed,
            batch_philox_offset,
            exp_scores_ptrs,
            block_min,
            block_max,
            offs_n_causal,
            masked_blocks,
            n_extra_tokens,
            alibi_slope,
            score_ptrs,
            scores_scaled_shifted_ptrs,
            IS_CAUSAL,
            BLOCK_M,
            BLOCK_DMODEL_QK,
            BLOCK_DMODEL_V,
            BLOCK_N,
            offs_m,
            offs_n,
            # _, MASK_STEPS, ...
            PRE_LOAD_V,
            True,
            ENABLE_DROPOUT,
            PADDED_HEAD_QK,
            PADDED_HEAD_V,
            ACTUAL_BLOCK_DMODEL_QK,
            ACTUAL_BLOCK_DMODEL_V,
            SM_SCALE,
            USE_EXP2=USE_EXP2,
            RETURN_SCORES=RETURN_SCORES,
        )
    if USE_FP8:
        # FP8 -> FP32
        acc *= acc_descale

    # epilogue
    # This helps the compiler do Newton Raphson on l_i vs on acc which is much larger.
    l_recip = 1 / l_i[:, None]
    acc = acc * l_recip
    if ENABLE_DROPOUT:
        acc = acc / (1 - dropout_p)
    # If seqlen_q > seqlen_k but the delta is not a multiple of BLOCK_M,
    # then we have one block with a row of all NaNs which come from computing
    # softmax over a row of all -infs (-inf - inf = NaN). We check for that here
    # and store 0s where there are NaNs as these rows should've been zeroed out.
    end_m_idx = (start_m + 1) * BLOCK_M
    start_m_idx = start_m * BLOCK_M
    causal_start_idx = seqlen_q - seqlen_k

    acc = acc.to(Out.type.element_ty)
    if IS_CAUSAL and causal_start_idx > start_m_idx and causal_start_idx < end_m_idx:
        out_mask_boundary = tl.full((BLOCK_DMODEL_V,), causal_start_idx, dtype=tl.int32)
        mask_m_offsets = start_m_idx + tl.arange(0, BLOCK_M)
        out_ptrs_mask = mask_m_offsets[:, None] >= out_mask_boundary[None, :]
        z = 0.0
        acc = tl.where(out_ptrs_mask, acc, z.to(acc.dtype))

    # write back LSE(Log Sum Exponents), the log of the normalization constant
    l_offset = (
        LSE
        + off_z * stride_lse_z
        + off_h_q * stride_lse_h
        + cu_seqlens_q_start * stride_lse_m
    )
    # leave an extra BLOCK_M for delta in backward
    # so we can load them together
    offs_l_m = start_m * BLOCK_M * 2 + tl.arange(0, BLOCK_M)
    l_ptrs = l_offset + offs_l_m * stride_lse_m
    if USE_EXP2:
        RCP_LN2: tl.constexpr = 1.4426950408889634
        LN2: tl.constexpr = 0.6931471824645996
        # compute log-sum-exp in base 2 units
        mi_base2 = m_i * RCP_LN2
        softmax_lse = mi_base2 + tl.math.log2(l_i)
        # convert back to natural units
        softmax_lse *= LN2
    else:
        softmax_lse = m_i + tl.math.log(l_i)

    if IS_CAUSAL:
        # zero out nans caused by -infs when doing causal
        lse_mask = (start_m_idx + tl.arange(0, BLOCK_M)) < causal_start_idx
        softmax_lse = tl.where(lse_mask, 0.0, softmax_lse)

    # If seqlen_q not multiple of BLOCK_M, we need to mask out the last few rows.
    # This is only true for the last M block. For others, overflow_size will be -ve
    overflow_size = end_m_idx - seqlen_q
    if overflow_size > 0:
        boundary = tl.full((BLOCK_M,), BLOCK_M - overflow_size, dtype=tl.int32)
        l_ptrs_mask = tl.arange(0, BLOCK_M) < boundary
        tl.store(
            l_ptrs, softmax_lse, mask=l_ptrs_mask
        )  # the log of the normalization constant
    else:
        tl.store(l_ptrs, softmax_lse)  # the log of the normalization constant

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
    tl.store(o_ptrs, acc.to(Out.type.element_ty), mask=o_ptrs_mask)


def get_padded_head_dim(head_size: int):
    # Get closest power of 2 over or equal to 32.
    padded_d_model = 1 << (head_size - 1).bit_length()
    # Smallest head_dim supported is 16. If smaller, the tile in the
    # kernel is padded - there is no padding in memory for any dims.
    padded_d_model = max(padded_d_model, 16)
    return padded_d_model


@triton.jit
def compute_fp8_scaling_factors(x, fp8_max: tl.constexpr):
    x_amax = tl.max(tl.abs(x))
    scale_x = fp8_max / (x_amax + 1e-7)
    return scale_x


@triton.jit
def _bwd_preprocess_use_o(
    Out,
    DO,
    DO_FP8,
    do_scale_ptr,
    Delta,
    USE_FP8: tl.constexpr,
    stride_oz,
    stride_oh,
    stride_om,
    stride_ok,
    stride_doz,
    stride_doh,
    stride_dom,
    stride_dok,
    stride_deltaz,
    stride_deltah,
    stride_deltam,
    stride_doscalez,
    stride_doscaleh,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL_V: tl.constexpr,
    ACTUAL_BLOCK_DMODEL_V: tl.constexpr,
    N_CTX_Q: tl.constexpr,
    Z: tl.constexpr,
    HQ: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    F8_BWD_DTYPE: tl.constexpr,
    F8_BWD_MAX: tl.constexpr,
):
    """
    load o, do and compute delta
    """
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    # Compute batch and head indices
    off_z = pid_bh // HQ
    off_h = pid_bh % HQ

    if IS_VARLEN:
        # Compute sequence lengths for the current batch
        q_start = tl.load(cu_seqlens_q + off_z)
        q_end = tl.load(cu_seqlens_q + off_z + 1)
        k_start = tl.load(cu_seqlens_k + off_z)
        k_end = tl.load(cu_seqlens_k + off_z + 1)

        # Compute actual sequence lengths
        N_CTX_Q = q_end - q_start
        k_end - k_start
    else:
        q_start = 0
        k_start = 0
        N_CTX_Q = max_seqlen_q

    off_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    off_d_v = tl.arange(0, BLOCK_DMODEL_V)

    mask_m = off_m < N_CTX_Q
    mask_d_v = off_d_v < ACTUAL_BLOCK_DMODEL_V
    mask_o = mask_m[:, None] & mask_d_v[None, :]

    # compute offsets
    o_offset = Out + off_z * stride_oz + off_h * stride_oh + q_start * stride_om
    do_offset = DO + off_z * stride_doz + off_h * stride_doh + q_start * stride_dom

    # compute pointers
    out_ptrs = o_offset + off_m[:, None] * stride_om + off_d_v[None, :] * stride_ok
    do_ptrs = do_offset + off_m[:, None] * stride_dom + off_d_v[None, :] * stride_dok

    # load
    o = tl.load(out_ptrs, mask=mask_o, other=0.0).to(tl.float32)
    do = tl.load(do_ptrs, mask=mask_o, other=0.0).to(tl.float32)

    # compute delta
    delta = tl.sum(o * do, axis=1)

    # write-back delta
    off_d_m = pid_m * BLOCK_M * 2 + tl.arange(BLOCK_M, 2 * BLOCK_M)
    delta_offset = (
        Delta + off_z * stride_deltaz + off_h * stride_deltah + q_start * stride_deltam
    )
    delta_ptrs = delta_offset + off_d_m * stride_deltam
    tl.store(delta_ptrs, delta, mask=mask_m)

    if USE_FP8:
        do_scale = compute_fp8_scaling_factors(do, F8_BWD_MAX)
        do_fp8 = (do * do_scale).to(F8_BWD_DTYPE)

        do_fp8_offset = (
            DO_FP8 + off_z * stride_doz + off_h * stride_doh + q_start * stride_dom
        )
        do_fp8_ptrs = (
            do_fp8_offset + off_m[:, None] * stride_dom + off_d_v[None, :] * stride_dok
        )

        tl.store(do_fp8_ptrs, do_fp8, mask=mask_o)

        do_scale_offset = (
            do_scale_ptr + off_z * stride_doscalez + off_h * stride_doscaleh + pid_m
        )
        tl.store(do_scale_offset, 1.0 / do_scale)


def get_autotune_bwd_configs():
    return [
        triton.Config(
            {},
            num_stages=1,
            num_warps=4,
        ),
    ], [
        "BLOCK_DMODEL",
        "ACTUAL_BLOCK_DMODEL_QK",
        "ACTUAL_BLOCK_DMODEL_V",
        "SEQUENCE_PARALLEL",
        "CAUSAL",
        "USE_FP8",
    ]


autotune_bwd_configs, autotune_bwd_keys = get_autotune_bwd_configs()


@triton.autotune(
    configs=autotune_bwd_configs,
    key=autotune_bwd_keys,
)
@triton.jit
def _bwd_kernel_dkdv(
    Q,
    K,
    V,
    sm_scale: tl.constexpr,
    q_scale_ptr,
    k_descale_ptr,
    v_scale_ptr,
    p_scale: tl.constexpr,
    do_descale_ptr,
    Out,
    DO,
    DQ,
    DK,
    DV,
    LD,
    stride_dq_all,
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
    stride_vn,
    stride_vk,
    stride_doz,
    stride_doh,
    stride_dom,
    stride_dok,
    stride_ldz,
    stride_ldh,
    stride_ldm,
    stride_doscalez: tl.constexpr,
    stride_doscaleh: tl.constexpr,
    stride_doscalem: tl.constexpr,
    stride_qscalez: tl.constexpr,
    stride_qscaleh: tl.constexpr,
    stride_qscalem: tl.constexpr,
    stride_kscalez: tl.constexpr,
    stride_kscaleh: tl.constexpr,
    stride_kscalem: tl.constexpr,
    padded_doscale_block_num: tl.constexpr,
    padded_qscale_block_num: tl.constexpr,
    padded_kscale_block_num: tl.constexpr,
    Z,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    num_block_m: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL_QK: tl.constexpr,
    BLOCK_DMODEL_V: tl.constexpr,
    ACTUAL_BLOCK_DMODEL_QK: tl.constexpr,
    ACTUAL_BLOCK_DMODEL_V: tl.constexpr,
    SEQUENCE_PARALLEL: tl.constexpr,
    CAUSAL: tl.constexpr,
    USE_EXP2: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_FP8: tl.constexpr,
    log_p_scale: tl.constexpr,
    F8_FWD_MAX: tl.constexpr,
):
    # program ids
    off_hz = tl.program_id(0)
    start_n = tl.program_id(1)

    off_z = off_hz // HK
    off_h_k = off_hz % HK

    GROUP_SIZE: tl.constexpr = HQ // HK
    if GROUP_SIZE != 1:
        off_h_q = off_h_k * GROUP_SIZE
    else:
        off_h_q = off_h_k

    if IS_VARLEN:
        # Compute sequence lengths for the current batch
        q_start = tl.load(cu_seqlens_q + off_z)
        q_end = tl.load(cu_seqlens_q + off_z + 1)
        k_start = tl.load(cu_seqlens_k + off_z)
        k_end = tl.load(cu_seqlens_k + off_z + 1)

        # Compute actual sequence lengths
        N_CTX_Q = q_end - q_start
        N_CTX_K = k_end - k_start
    else:
        q_start = 0
        k_start = 0
        N_CTX_Q = max_seqlen_q
        N_CTX_K = max_seqlen_k

    # input tensor offsets
    q_offset = Q + off_z * stride_qz + off_h_q * stride_qh + q_start * stride_qm
    k_offset = K + off_z * stride_kz + off_h_k * stride_kh + k_start * stride_kn
    v_offset = V + off_z * stride_vz + off_h_k * stride_vh + k_start * stride_vn
    do_offset = DO + off_z * stride_doz + off_h_q * stride_doh + q_start * stride_dom
    ld_offset = LD + off_z * stride_ldz + off_h_q * stride_ldh + q_start * stride_ldm

    # output tensor offsets
    # sume dk and dv
    dk_offset = DK + off_z * stride_kz + off_h_k * stride_kh + k_start * stride_kn
    dv_offset = DV + off_z * stride_vz + off_h_k * stride_vh + k_start * stride_vn

    if USE_FP8:
        # we keep v in per-tensor scaling
        # while q, k, do in per-block scaling
        v_scale = tl.load(v_scale_ptr)  # + tl.arange(0, num_block_n)
        q_descale_offset = (
            q_scale_ptr
            + off_z * stride_qscalez
            + off_h_q * stride_qscaleh
            + q_start * stride_qscalem
        )
        do_descale_offset = (
            do_descale_ptr
            + off_z * stride_doscalez
            + off_h_q * stride_doscaleh
            + q_start * stride_doscalem
        )
    else:
        v_scale = 1.0
        q_descale_offset = None
        do_descale_offset = None

    if CAUSAL:
        causal_boundary = start_n * BLOCK_N - BLOCK_M
        lo = (causal_boundary + 1) // BLOCK_M * BLOCK_M

    else:
        causal_boundary = 0
        lo = 0

    offs_d_qk = tl.arange(0, BLOCK_DMODEL_QK)
    offs_d_v = tl.arange(0, BLOCK_DMODEL_V)
    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_n = offs_n < N_CTX_K
    mask_d_qk = offs_d_qk < ACTUAL_BLOCK_DMODEL_QK
    mask_d_v = offs_d_v < ACTUAL_BLOCK_DMODEL_V
    k_mask = mask_n[:, None] & mask_d_qk[None, :]
    v_mask = mask_n[:, None] & mask_d_v[None, :]

    k_ptrs = k_offset + offs_n[:, None] * stride_kn + offs_d_qk[None, :] * stride_kk
    v_ptrs = v_offset + offs_n[:, None] * stride_vn + offs_d_v[None, :] * stride_vk

    k = tl.load(k_ptrs, mask=k_mask, other=0.0)
    v = tl.load(v_ptrs, mask=v_mask, other=0.0)

    k = tl.trans(k)
    v = tl.trans(v)

    dk = tl.zeros([BLOCK_N, BLOCK_DMODEL_QK], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, BLOCK_DMODEL_V], dtype=tl.float32)

    if USE_FP8:
        k_descale_offset = (
            k_descale_ptr
            + off_z * stride_kscalez
            + off_h_k * stride_kscaleh
            + start_n * stride_kscalem
        )
        blk_k_descale = tl.load(k_descale_offset)
    else:
        blk_k_descale = 1.0

    for group_idx in range(GROUP_SIZE):
        dk, dv = _attn_bwd_dkdv(
            k,
            v,
            dk,
            dv,
            offs_d_qk,
            offs_d_v,
            offs_n,
            mask_d_qk,
            mask_d_v,
            q_offset,
            do_offset,
            stride_qm,
            stride_qk,
            stride_dom,
            stride_dok,
            ld_offset,
            stride_ldm,
            BLOCK_M,
            BLOCK_N,
            q_descale_offset,
            do_descale_offset,
            blk_k_descale,
            v_scale,
            p_scale,
            sm_scale,
            log_p_scale,
            lo,
            num_block_m,
            causal_boundary,
            USE_FP8,
            USE_EXP2,
            F8_FWD_MAX,
            N_CTX_Q,
            N_CTX_K,
            CAUSAL,
        )

        q_offset += stride_qh
        do_offset += stride_qh
        ld_offset += stride_ldh
        if USE_FP8:
            q_descale_offset += stride_qscaleh
            do_descale_offset += stride_doscaleh

    if USE_FP8:
        dv_descale = 1.0 / (p_scale)
        dv *= dv_descale

    dk *= sm_scale / p_scale

    dk_ptrs = dk_offset + offs_n[:, None] * stride_kn + offs_d_qk[None, :] * stride_kk
    tl.store(dk_ptrs, dk, mask=k_mask)

    dv_ptrs = dv_offset + offs_n[:, None] * stride_vn + offs_d_v[None, :] * stride_vk
    tl.store(dv_ptrs, dv, mask=v_mask)


@triton.jit
def _attn_bwd_dkdv(
    k,
    v,
    dk,
    dv,
    offs_d_qk,
    offs_d_v,
    offs_n,
    mask_d_qk,
    mask_d_v,
    q_offset,
    do_offset,
    stride_qm,
    stride_qk,
    stride_dom,
    stride_dok,
    ld_offset,
    stride_ldm,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    q_descale_offset,
    do_descale_offset,
    k_descale,
    v_scale,
    p_scale: tl.constexpr,
    sm_scale: tl.constexpr,
    log_p_scale: tl.constexpr,
    lo: tl.constexpr,
    num_block_m: tl.constexpr,
    causal_boundary: tl.constexpr,
    USE_FP8: tl.constexpr,
    USE_EXP2: tl.constexpr,
    F8_FWD_MAX: tl.constexpr,
    N_CTX_Q: tl.constexpr,
    N_CTX_K: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    idx_block_m = lo // BLOCK_M - 1

    # loop over rows
    for start_m in range(lo, num_block_m * BLOCK_M, BLOCK_M):
        # can_skip_causal_block = start_m < causal_boundary
        offs_m = start_m + tl.arange(0, BLOCK_M)
        q_ptrs = q_offset + offs_m[:, None] * stride_qm + offs_d_qk[None, :] * stride_qk
        do_ptrs = (
            do_offset + offs_m[:, None] * stride_dom + offs_d_v[None, :] * stride_dok
        )

        mask_m = offs_m < N_CTX_Q
        q_mask = mask_m[:, None] & mask_d_qk[None, :]
        do_mask = mask_m[:, None] & mask_d_v[None, :]

        if USE_FP8:
            idx_block_m += 1
            blk_q_descale = tl.load(q_descale_offset + idx_block_m)
            blk_do_descale = tl.load(do_descale_offset + idx_block_m)
        else:
            blk_q_descale = 1.0
            blk_do_descale = 1.0

        q = tl.load(q_ptrs, mask=q_mask, other=0.0)
        qk = tl.dot(q, k, out_dtype=tl.float32, allow_tf32=False)

        if USE_FP8:
            # can fuse with sm_scale
            qk_descale = blk_q_descale * k_descale
            qk = (
                qk * qk_descale
            )  # we fused sm_scale into blk_q_descale so we do not need one more mul here

        if CAUSAL:
            # if not can_skip_causal_block:
            col_offset = N_CTX_Q - N_CTX_K
            causal_mask = offs_m[:, None] >= (col_offset + offs_n[None, :])
            qk = tl.where(causal_mask, qk, float("-inf"))

        l_ptrs = ld_offset + (2 * start_m + tl.arange(0, 2 * BLOCK_M)) * stride_ldm
        mask_ldm = tl.ravel(tl.join(mask_m, mask_m))
        lds = tl.load(l_ptrs, mask=mask_ldm, other=0.0)
        l_i = tl.gather(lds, index=tl.arange(0, BLOCK_M), axis=0)

        # compute p
        if USE_EXP2:
            RCP_LN2: tl.constexpr = 1.4426950408889634
            qk *= sm_scale * RCP_LN2
            l_i *= RCP_LN2
            p = tl.math.exp2(qk - l_i[:, None] + log_p_scale * RCP_LN2)
        else:
            qk *= sm_scale
            p = tl.math.exp(qk - l_i[:, None] + log_p_scale)

        do = tl.load(do_ptrs, mask=do_mask, other=0.0)

        # compute dp
        dp = tl.dot(do, v, out_dtype=tl.float32, allow_tf32=False)

        if USE_FP8:
            dp_descale = blk_do_descale / v_scale
            dp = dp * dp_descale

        Di = tl.gather(lds, index=tl.arange(BLOCK_M, 2 * BLOCK_M), axis=0)
        ds = p * (dp - Di[:, None])

        if USE_FP8:
            ds_scale = compute_fp8_scaling_factors(ds, F8_FWD_MAX)
            ds = ds * ds_scale

        ds = ds.to(q.dtype)

        # compute dv
        _dv = tl.dot(
            tl.trans(p.to(k.dtype)), do, out_dtype=tl.float32, allow_tf32=False
        )

        if USE_FP8:
            dv = tl.fma(_dv, blk_do_descale, dv)
        else:
            dv += _dv

        # compute dk = dot(ds.T, q)
        _dk = tl.dot(tl.trans(ds), q, out_dtype=tl.float32, allow_tf32=False)

        if USE_FP8:
            dk_descale = blk_q_descale / ds_scale
            dk = tl.fma(_dk, dk_descale, dk)
        else:
            dk += _dk

    return dk, dv


@triton.autotune(
    configs=autotune_bwd_configs,
    key=autotune_bwd_keys,
)
@triton.jit
def _bwd_kernel_dq(
    Q,
    K,
    V,
    sm_scale: tl.constexpr,
    q_scale_ptr,
    k_descale_ptr,
    v_scale_ptr,
    p_scale: tl.constexpr,
    do_descale_ptr,
    Out,
    DO,
    DQ,
    DK,
    DV,
    LD,
    stride_dq_all,
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
    stride_vn,
    stride_vk,
    stride_doz,
    stride_doh,
    stride_dom,
    stride_dok,
    stride_ldz,
    stride_ldh,
    stride_ldm,
    stride_doscalez: tl.constexpr,
    stride_doscaleh: tl.constexpr,
    stride_doscalem: tl.constexpr,
    stride_qscalez: tl.constexpr,
    stride_qscaleh: tl.constexpr,
    stride_qscalem: tl.constexpr,
    stride_kscalez: tl.constexpr,
    stride_kscaleh: tl.constexpr,
    stride_kscalem: tl.constexpr,
    padded_doscale_block_num: tl.constexpr,
    padded_qscale_block_num: tl.constexpr,
    padded_kscale_block_num: tl.constexpr,
    Z,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    num_block_m: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL_QK: tl.constexpr,
    BLOCK_DMODEL_V: tl.constexpr,
    ACTUAL_BLOCK_DMODEL_QK: tl.constexpr,
    ACTUAL_BLOCK_DMODEL_V: tl.constexpr,
    SEQUENCE_PARALLEL: tl.constexpr,
    CAUSAL: tl.constexpr,
    USE_EXP2: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_FP8: tl.constexpr,
    log_p_scale: tl.constexpr,
    F8_FWD_MAX: tl.constexpr,
):
    # program ids
    off_hz = tl.program_id(0)
    start_m = tl.program_id(1)

    num_block_n = tl.cdiv(max_seqlen_k, BLOCK_N)

    off_z = off_hz // HQ
    off_h_q = off_hz % HQ

    GROUP_SIZE: tl.constexpr = HQ // HK
    if GROUP_SIZE != 1:
        off_h_k = off_h_q // GROUP_SIZE
    else:
        off_h_k = off_h_q

    if IS_VARLEN:
        # Compute sequence lengths for the current batch
        q_start = tl.load(cu_seqlens_q + off_z)
        q_end = tl.load(cu_seqlens_q + off_z + 1)
        k_start = tl.load(cu_seqlens_k + off_z)
        k_end = tl.load(cu_seqlens_k + off_z + 1)

        # Compute actual sequence lengths
        N_CTX_Q = q_end - q_start
        N_CTX_K = k_end - k_start
    else:
        q_start = 0
        k_start = 0
        N_CTX_Q = max_seqlen_q
        N_CTX_K = max_seqlen_k

    # input tensor offsets
    q_offset = Q + off_z * stride_qz + off_h_q * stride_qh + q_start * stride_qm
    k_offset = K + off_z * stride_kz + off_h_k * stride_kh + k_start * stride_kn
    v_offset = V + off_z * stride_vz + off_h_k * stride_vh + k_start * stride_vn
    do_offset = DO + off_z * stride_doz + off_h_q * stride_doh + q_start * stride_dom
    ld_offset = LD + off_z * stride_ldz + off_h_q * stride_ldh + q_start * stride_ldm

    # output tensor offsets
    dq_offset = DQ + off_z * stride_qz + off_h_q * stride_qh + q_start * stride_qm

    if USE_FP8:
        # we keep v in per-tensor scaling
        # while q, k, do in per-block scaling
        v_scale = tl.load(v_scale_ptr)  # + tl.arange(0, num_block_n)
        acutal_kscale_block_num = stride_kscaleh

        kscale_mask = tl.arange(0, padded_kscale_block_num) < acutal_kscale_block_num
        k_descale_offset = (
            k_descale_ptr
            + off_z * stride_kscalez
            + off_h_k * stride_kscaleh
            + tl.arange(0, padded_kscale_block_num)
        )  #  + q_start * stride_qm
        k_descale = tl.load(k_descale_offset, mask=kscale_mask, other=1.0)
    else:
        k_descale = 1.0
        v_scale = 1.0

    if CAUSAL:
        causal_boundary = start_m * BLOCK_M - BLOCK_N
        hi = (
            tl.minimum(BLOCK_M // BLOCK_N * (start_m + 1), num_block_n)
        ) * BLOCK_N  # start_n == start_m

    else:
        causal_boundary = 0
        hi = num_block_n * BLOCK_N

    offs_d_qk = tl.arange(0, BLOCK_DMODEL_QK)
    offs_d_v = tl.arange(0, BLOCK_DMODEL_V)

    # compute dq
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    dq = tl.zeros([BLOCK_M, BLOCK_DMODEL_QK], dtype=tl.float32)
    q_ptrs = q_offset + offs_m[:, None] * stride_qm + offs_d_qk[None, :] * stride_qk
    do_ptrs = do_offset + offs_m[:, None] * stride_dom + offs_d_v[None, :] * stride_dok

    mask_m = offs_m < N_CTX_Q
    mask_d_qk = offs_d_qk < ACTUAL_BLOCK_DMODEL_QK
    mask_d_v = offs_d_v < ACTUAL_BLOCK_DMODEL_V
    mask_q = mask_m[:, None] & mask_d_qk[None, :]
    mask_do = mask_m[:, None] & mask_d_v[None, :]

    q = tl.load(q_ptrs, mask=mask_q, other=0.0)
    do = tl.load(do_ptrs, mask=mask_do, other=0.0)

    if USE_FP8:
        q_descale_offset = (
            q_scale_ptr
            + off_z * stride_qscalez
            + off_h_q * stride_qscaleh
            + start_m * stride_qscalem
        )
        blk_q_descale = tl.load(q_descale_offset)

        do_descale_offset = (
            do_descale_ptr
            + off_z * stride_doscalez
            + off_h_q * stride_doscaleh
            + start_m * stride_doscalem
        )
        blk_do_descale = tl.load(do_descale_offset)
    else:
        blk_q_descale = 1.0
        blk_do_descale = 1.0

    l_ptrs = (
        ld_offset + (2 * start_m * BLOCK_M + tl.arange(0, 2 * BLOCK_M)) * stride_ldm
    )
    mask_ldm = tl.ravel(tl.join(mask_m, mask_m))
    lds = tl.load(l_ptrs, mask=mask_ldm, other=0.0)

    dq = _attn_bwd_dq(
        dq,
        q,
        offs_d_qk,
        offs_d_v,
        offs_m,
        lds,
        do,
        mask_d_qk,
        mask_d_v,
        k_offset,
        v_offset,
        stride_kn,
        stride_kk,
        stride_vn,
        stride_vk,
        BLOCK_M,
        BLOCK_N,
        blk_q_descale,
        k_descale,
        blk_do_descale,
        v_scale,
        p_scale,
        sm_scale,
        log_p_scale,
        hi,
        num_block_m,
        causal_boundary,
        USE_FP8,
        USE_EXP2,
        F8_FWD_MAX,
        N_CTX_Q,
        N_CTX_K,
        CAUSAL,
    )

    dq_ptrs = dq_offset + offs_m[:, None] * stride_qm + offs_d_qk[None, :] * stride_qk
    dq *= sm_scale / p_scale
    tl.store(dq_ptrs, dq, mask=mask_q)


@triton.jit
def _attn_bwd_dq(
    dq,
    q,
    offs_d_qk,
    offs_d_v,
    offs_m,
    lds,
    do,
    mask_d_qk,
    mask_d_v,
    k_offset,
    v_offset,
    stride_kn,
    stride_kk,
    stride_vn,
    stride_vk,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    q_descale,
    k_descale,
    do_descale,
    v_scale,
    p_scale: tl.constexpr,
    sm_scale: tl.constexpr,
    log_p_scale: tl.constexpr,
    hi: tl.constexpr,
    num_block_n: tl.constexpr,
    causal_boundary: tl.constexpr,
    USE_FP8: tl.constexpr,
    USE_EXP2: tl.constexpr,
    F8_FWD_MAX: tl.constexpr,
    N_CTX_Q: tl.constexpr,
    N_CTX_K: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    idx_block_n = tl.full([1], -1, dtype=tl.int32)
    l_i = tl.gather(lds, index=tl.arange(0, BLOCK_M), axis=0)
    Di = tl.gather(lds, index=tl.arange(BLOCK_M, 2 * BLOCK_M), axis=0)

    RCP_LN2: tl.constexpr = 1.4426950408889634
    if USE_EXP2:
        l_i *= RCP_LN2

    # loop over rows
    for start_n in range(0, hi, BLOCK_N):
        # can_skip_causal_block = start_n < causal_boundary
        offs_n = start_n + tl.arange(0, BLOCK_N)

        mask_n = offs_n < N_CTX_K
        mask_k = mask_n[:, None] & mask_d_qk[None, :]
        mask_v = mask_n[:, None] & mask_d_v[None, :]

        # k_ptrs += stride_kn * BLOCK_N
        # v_ptrs += stride_kn * BLOCK_N
        k_ptrs = k_offset + offs_n[:, None] * stride_kn + offs_d_qk[None, :] * stride_kk
        v_ptrs = v_offset + offs_n[:, None] * stride_vn + offs_d_v[None, :] * stride_vk

        k = tl.load(k_ptrs, mask=mask_k, other=0.0)
        v = tl.load(v_ptrs, mask=mask_v, other=0.0)

        kt = tl.trans(k)
        qk = tl.dot(q, kt, out_dtype=tl.float32, allow_tf32=False)

        if USE_FP8:
            # can fuse with sm_scale
            idx_block_n += 1
            blk_k_descale = tl.gather(k_descale, index=idx_block_n, axis=0)
            qk_descale = q_descale * blk_k_descale
            qk = (
                qk * qk_descale
            )  # we fused sm_scale into blk_q_descale so we do not need one more mul here

        if CAUSAL:
            # if not can_skip_causal_block:
            col_offset = N_CTX_Q - N_CTX_K
            causal_mask = offs_m[:, None] >= (col_offset + offs_n[None, :])
            qk = tl.where(causal_mask, qk, float("-inf"))

        # compute p
        if USE_EXP2:
            qk *= sm_scale * RCP_LN2
            p = tl.math.exp2(qk - l_i[:, None] + log_p_scale * RCP_LN2)
        else:
            qk *= sm_scale
            p = tl.math.exp(qk - l_i[:, None] + log_p_scale)

        # compute dp
        dp = tl.dot(do, tl.trans(v), out_dtype=tl.float32, allow_tf32=False)

        if USE_FP8:
            dp_descale = do_descale / v_scale
            dp = dp * dp_descale

        ds = p * (dp - Di[:, None])

        if USE_FP8:
            ds_scale = compute_fp8_scaling_factors(ds, F8_FWD_MAX)
            ds = ds * ds_scale

        ds = ds.to(q.dtype)
        _dq = tl.dot(ds, k, out_dtype=tl.float32, allow_tf32=False)

        if USE_FP8:
            dq_descale = blk_k_descale / ds_scale  # ds_scale # 1. / k_scale
            _dq = _dq * dq_descale

        dq += _dq
    return dq
