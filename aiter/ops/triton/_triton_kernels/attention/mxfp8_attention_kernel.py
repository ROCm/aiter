"""
Fused Attention
===============

This is a Triton implementation of the Flash Attention v2 algorithm from Tri Dao
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

from aiter.ops.triton._triton_kernels.quant.quant_mxfp8 import (
    _calculate_scales,
    _pack_fp8,
    _unpack_fp8,
)
from aiter.ops.triton.utils._triton.arch_info import is_cdna4

if is_cdna4():
    QUANT_SIZE: tl.constexpr = 32
else:
    QUANT_SIZE: tl.constexpr = 16

# Seed the RNG so we get reproducible results for testing.
philox_seed: tl.constexpr = 0x1BF52
philox_offset: tl.constexpr = 0x1D4B42

AUTOTUNE = os.environ.get("AITER_TRITON_AMD_AUTOTUNE", "0").lower() in (
    "1",
    "true",
    "yes",
)


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
    q_scale,
    k_scale_ptr,
    v_scale_ptr,
    p_scale: tl.constexpr,
    use_mxfp8: tl.constexpr,
    k_ptrs,
    v_ptrs,
    bias_ptrs,
    stride_kn,
    stride_vk,
    stride_bn,
    stride_kdescale_m,
    stride_vdescale_m,
    scales_num_block_m: tl.constexpr,  # number of scale block per BLOCK_M
    scales_num_block_n: tl.constexpr,  # number of scale block per BLOCK_N
    scales_num_block_d_qk: tl.constexpr,  # number of scale block per BLOCK_D_QK
    scales_num_block_d_v: tl.constexpr,  # number of scale block per BLOCK_D_V
    SCALE_NUM_PER_QUANT_BLK: tl.constexpr,
    SCALE_NUM_PER_D_QK: tl.constexpr,
    SCALE_NUM_PER_D_V: tl.constexpr,
    SCALE_NUM_PER_N: tl.constexpr,
    SCALE_NUM_PER_M: tl.constexpr,
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
    QUANT_BLOCK_SIZE: tl.constexpr,
    FLOAT_DTYPE: tl.constexpr,
    QUANT_SIZE: tl.constexpr,
):
    if USE_EXP2:
        RCP_LN2: tl.constexpr = 1.4426950408889634

    if FLOAT_DTYPE == "e4m3":
        USE_ASM: tl.constexpr = True
    else:
        USE_ASM: tl.constexpr = False

    if scales_num_block_n == 1 and scales_num_block_d_qk == 1:
        pass
    else:
        pass

    if scales_num_block_d_v == 1 and scales_num_block_n == 1:
        pass
    else:
        pass

    if use_mxfp8:
        p_scale_t = tl.cast(p_scale, tl.uint8)
        p_scale_b = tl.zeros([BLOCK_M, SCALE_NUM_PER_N], dtype=tl.uint8) + p_scale_t

        p_scale_f = tl.cast(p_scale, tl.uint32)
        p_scale_f = (p_scale_f << 23).to(tl.float32, bitcast=True)
    else:
        p_scale_b = p_scale

    # loop over k, v, and update accumulator
    for start_n in range(block_min, block_max, BLOCK_N):
        if MASK_STEPS:
            k_offs_n = start_n + tl.arange(0, BLOCK_N)
        else:
            k_offs_n = None
        v_offs_k = None if not PADDED_HEAD_V else tl.arange(0, BLOCK_DMODEL_V)

        # For padded blocks, we will overrun the tensor size if
        # we load all BLOCK_N. For others, the blocks are all within range.
        if use_mxfp8:
            blk_v_scale = tl.load(v_scale_ptr)
            blk_k_scale = tl.load(k_scale_ptr)

        else:
            blk_k_scale = 1.0
            blk_v_scale = 1.0

        k_offs_k = None if not PADDED_HEAD_QK else tl.arange(0, BLOCK_DMODEL_QK)
        k = load_fn(k_ptrs, k_offs_k, k_offs_n, ACTUAL_BLOCK_DMODEL_QK, actual_seqlen_k)

        if PRE_LOAD_V:
            # We can use the same offsets as k, just with dims transposed.
            v = load_fn(
                v_ptrs, k_offs_n, v_offs_k, actual_seqlen_k, ACTUAL_BLOCK_DMODEL_V
            )

        # -- compute qk ----
        if use_mxfp8:
            if SCALE_NUM_PER_D_QK % 2 == 0:
                qk = tl.dot_scaled(
                    q,
                    q_scale,
                    FLOAT_DTYPE,
                    k,
                    blk_k_scale,
                    FLOAT_DTYPE,
                    out_dtype=tl.float32,
                )
            else:
                q_descaled = _unpack_fp8(
                    q,
                    q_scale,
                    tl.float32,
                    BLOCK_M,
                    BLOCK_DMODEL_QK,
                    QUANT_BLOCK_SIZE,
                    True,
                    USE_ASM,
                )
                k_descaled = _unpack_fp8(
                    k,
                    blk_k_scale,
                    tl.float32,
                    BLOCK_DMODEL_QK,
                    BLOCK_N,
                    QUANT_BLOCK_SIZE,
                    True,
                    USE_ASM,
                )
                qk = tl.dot(
                    q_descaled, k_descaled, out_dtype=tl.float32, allow_tf32=False
                )

        else:
            qk = tl.dot(q, k, out_dtype=tl.float32, allow_tf32=False)

        # We start from end of seqlen_k so only the first iteration would need
        # to be checked for padding if it is not a multiple of block_n
        # TODO: This can be optimized to only be true for the padded block.
        if MASK_STEPS and (start_n + BLOCK_N == block_max) and (n_extra_tokens != 0):
            boundary_m = tl.full([BLOCK_M], actual_seqlen_k, dtype=tl.int32)
            size_n = start_n + OFFS_N[None, :]
            mask = size_n < boundary_m[:, None]
            qk = tl.where(mask, qk, float("-inf"))

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

        if not PRE_LOAD_V:
            v = load_fn(
                v_ptrs, k_offs_n, v_offs_k, actual_seqlen_k, ACTUAL_BLOCK_DMODEL_V
            )

        if use_mxfp8:
            if SCALE_NUM_PER_N % 2 == 0:
                pv = tl.dot_scaled(
                    (p).to(q.dtype),
                    p_scale_b.to(tl.uint8),
                    FLOAT_DTYPE,
                    v.to(q.dtype),
                    blk_v_scale.to(tl.uint8),
                    FLOAT_DTYPE,
                    out_dtype=tl.float32,
                )

            else:
                v_descaled = _unpack_fp8(
                    v,
                    blk_v_scale,
                    tl.float32,
                    BLOCK_N,
                    BLOCK_DMODEL_V,
                    QUANT_BLOCK_SIZE,
                    True,
                    USE_ASM,
                )
                pv = tl.dot(
                    p * p_scale_f, v_descaled, allow_tf32=False, out_dtype=tl.float32
                )

        else:
            pv = tl.dot(p.to(v.dtype), v, allow_tf32=False, out_dtype=tl.float32)

        # -- update output accumulator --
        # alpha is an adjustment factor for acc and li as we loop and find new maxes
        # store the diff in maxes to adjust acc and li as we discover new maxes
        m_diff = m_i - m_ij
        if USE_EXP2:
            alpha = tl.math.exp2(m_diff * RCP_LN2)
        else:
            alpha = tl.math.exp(m_diff)
        acc *= alpha[:, None]

        # -- update m_i and l_i
        l_i = l_i * alpha + l_ij
        # update m_i and l_i
        m_i = m_ij

        acc += pv

        k_ptrs += BLOCK_N * stride_kn
        v_ptrs += BLOCK_N * stride_vk
        if use_mxfp8:
            k_scale_ptr += scales_num_block_n * stride_kdescale_m
            v_scale_ptr += scales_num_block_n * stride_vdescale_m
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
        "VARLEN",
        "HQ",
        "HK",
        "use_mxfp8",
    ]


autotune_fwd_configs, autotune_fwd_keys = get_autotune_fwd_configs()


@triton.autotune(
    configs=autotune_fwd_configs,
    key=autotune_fwd_keys,
)
@triton.jit
def attn_fwd_mxfp8(
    Q,
    K,
    V,
    bias,
    p_scale: tl.constexpr,
    q_scale_ptr,
    k_scale_ptr,
    v_scale_ptr,
    use_mxfp8: tl.constexpr,
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
    stride_qdescale_z,
    stride_qdescale_h,
    stride_qdescale_m,
    stride_qdescale_d,
    stride_kdescale_z,
    stride_kdescale_h,
    stride_kdescale_m,
    stride_kdescale_d,
    stride_vdescale_z,
    stride_vdescale_h,
    stride_vdescale_m,
    stride_vdescale_d,
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
    QUANT_BLOCK_SIZE: tl.constexpr,
    QUANT_SIZE: tl.constexpr,
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
    FLOAT_DTYPE: tl.constexpr = "e4m3"
    if Q.dtype.element_ty == tl.float8e5:
        FLOAT_DTYPE: tl.constexpr = "e5m2"

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

    # we assume q and k has the same length
    if use_mxfp8:
        scales_num_block_m: tl.constexpr = BLOCK_M // QUANT_BLOCK_SIZE
        scales_num_block_n: tl.constexpr = BLOCK_N // QUANT_BLOCK_SIZE
        scales_num_block_d_qk: tl.constexpr = BLOCK_DMODEL_QK // QUANT_BLOCK_SIZE
        scales_num_block_d_v: tl.constexpr = BLOCK_DMODEL_V // QUANT_BLOCK_SIZE

        scale_offs_m = tl.arange(0, scales_num_block_m)
        scale_offs_n = tl.arange(0, scales_num_block_n)
        scale_offs_d_qk = tl.arange(0, scales_num_block_d_qk)
        scale_offs_d_v = tl.arange(0, scales_num_block_d_v)
    else:
        scales_num_block_m = 1
        scales_num_block_n = 1
        scales_num_block_d_qk = 1
        scales_num_block_d_v = 1

    # scale number per block in this warp tile for mxfp
    SCALE_NUM_PER_QUANT_BLK: tl.constexpr = QUANT_BLOCK_SIZE // QUANT_SIZE
    # scale number per D QK in this warp tile for mxfp
    SCALE_NUM_PER_D_QK: tl.constexpr = scales_num_block_d_qk * SCALE_NUM_PER_QUANT_BLK
    # scale number per D V in this warp tile for mxfp
    SCALE_NUM_PER_D_V: tl.constexpr = scales_num_block_d_v * SCALE_NUM_PER_QUANT_BLK
    # scale number per N in this warp tile for mxfp
    SCALE_NUM_PER_N: tl.constexpr = scales_num_block_n * SCALE_NUM_PER_QUANT_BLK
    # scale number per N in this warp tile for mxfp
    SCALE_NUM_PER_M: tl.constexpr = scales_num_block_m * SCALE_NUM_PER_QUANT_BLK

    if use_mxfp8:
        scale_offs_m_b = tl.arange(0, BLOCK_M)
        scale_offs_n_b = tl.arange(0, BLOCK_N)
        tl.arange(0, BLOCK_DMODEL_QK)
        scale_offs_d_v_b = tl.arange(0, BLOCK_DMODEL_V)

        tl.arange(0, SCALE_NUM_PER_M)
        scale_offs_n_q = tl.arange(0, SCALE_NUM_PER_N)
        scale_offs_d_qk_q = tl.arange(0, SCALE_NUM_PER_D_QK)
        tl.arange(0, SCALE_NUM_PER_D_V)

        if SCALE_NUM_PER_D_QK % 2 == 0:
            k_scale_ptr_base = (
                k_scale_ptr
                + stride_kdescale_z * off_z
                + stride_kdescale_h * off_h_k
                + cu_seqlens_k_start // QUANT_BLOCK_SIZE * stride_kdescale_m
                + scale_offs_n_b[:, None] // QUANT_BLOCK_SIZE * stride_kdescale_m
                + scale_offs_d_qk_q[None, :]
                // SCALE_NUM_PER_QUANT_BLK
                * stride_kdescale_d
            )
            q_scale_offset = (
                q_scale_ptr
                + stride_qdescale_z * off_z
                + stride_qdescale_h * off_h_q
                + cu_seqlens_q_start // QUANT_BLOCK_SIZE * stride_qdescale_m
                + start_m * scales_num_block_m * stride_qdescale_m
                + scale_offs_m_b[:, None] // QUANT_BLOCK_SIZE * stride_qdescale_m
                + scale_offs_d_qk_q[None, :]
                // SCALE_NUM_PER_QUANT_BLK
                * stride_qdescale_d
            )

        else:
            k_scale_ptr_base = (
                k_scale_ptr
                + stride_kdescale_z * off_z
                + stride_kdescale_h * off_h_k
                + cu_seqlens_k_start // QUANT_BLOCK_SIZE * stride_kdescale_m
                + scale_offs_d_qk[:, None] * stride_kdescale_d
                + scale_offs_n[None, :] * stride_kdescale_m
            )
            q_scale_offset = (
                q_scale_ptr
                + stride_qdescale_z * off_z
                + stride_qdescale_h * off_h_q
                + cu_seqlens_q_start // QUANT_BLOCK_SIZE * stride_qdescale_m
                + start_m * scales_num_block_m * stride_qdescale_m
                + scale_offs_m[:, None] * stride_qdescale_m
                + scale_offs_d_qk[None, :] * stride_qdescale_d
            )

        if SCALE_NUM_PER_N % 2 == 0:
            v_scale_ptr_base = (
                v_scale_ptr
                + stride_vdescale_z * off_z
                + stride_vdescale_h * off_h_k
                + cu_seqlens_k_start // QUANT_BLOCK_SIZE * stride_vdescale_m
                + scale_offs_d_v_b[:, None] // QUANT_BLOCK_SIZE * stride_vdescale_d
                + scale_offs_n_q[None, :] // SCALE_NUM_PER_QUANT_BLK * stride_vdescale_m
            )
        else:
            v_scale_ptr_base = (
                v_scale_ptr
                + stride_vdescale_z * off_z
                + stride_vdescale_h * off_h_k
                + cu_seqlens_k_start // QUANT_BLOCK_SIZE * stride_vdescale_m
                + scale_offs_n[:, None] * stride_vdescale_m
                + scale_offs_d_v[None, :] * stride_vdescale_d
            )
        q_scale = tl.load(q_scale_offset)
    else:
        q_scale = 1.0
        k_scale_ptr_base = None
        v_scale_ptr_base = None

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
        # Number of K blocks that intersect the causal boundary within a single Q block.
        # This is ceil(BLOCK_M / BLOCK_N). Additionally, there might be one more due to padding.
        masked_blocks = cdiv_fn(BLOCK_M, BLOCK_N) + (not is_modulo_mn)
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
            q_scale,
            k_scale_ptr_base,
            v_scale_ptr_base,
            p_scale,
            use_mxfp8,
            k_ptrs,
            v_ptrs,
            bias_ptrs,
            stride_kn,
            stride_vk,
            stride_bn,
            stride_kdescale_m,
            stride_vdescale_m,
            scales_num_block_m,
            scales_num_block_n,
            scales_num_block_d_qk,
            scales_num_block_d_v,
            SCALE_NUM_PER_QUANT_BLK,
            SCALE_NUM_PER_D_QK,
            SCALE_NUM_PER_D_V,
            SCALE_NUM_PER_N,
            SCALE_NUM_PER_M,
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
            QUANT_BLOCK_SIZE=QUANT_BLOCK_SIZE,
            FLOAT_DTYPE=FLOAT_DTYPE,
            QUANT_SIZE=QUANT_SIZE,
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
        if use_mxfp8:
            k_scale_ptr_base += n_full_blocks * scales_num_block_n * stride_kdescale_m
            v_scale_ptr_base += n_full_blocks * scales_num_block_n * stride_vdescale_m
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
            q_scale,
            k_scale_ptr_base,
            v_scale_ptr_base,
            p_scale,
            use_mxfp8,
            k_ptrs,
            v_ptrs,
            bias_ptrs,
            stride_kn,
            stride_vk,
            stride_bn,
            stride_kdescale_m,
            stride_vdescale_m,
            scales_num_block_m,
            scales_num_block_n,
            scales_num_block_d_qk,
            scales_num_block_d_v,
            SCALE_NUM_PER_QUANT_BLK,
            SCALE_NUM_PER_D_QK,
            SCALE_NUM_PER_D_V,
            SCALE_NUM_PER_N,
            SCALE_NUM_PER_M,
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
            QUANT_BLOCK_SIZE=QUANT_BLOCK_SIZE,
            FLOAT_DTYPE=FLOAT_DTYPE,
            QUANT_SIZE=QUANT_SIZE,
        )

    if use_mxfp8:
        p_scale_t = tl.cast(p_scale, tl.uint32)
        p_scale_t = (p_scale_t << 23).to(tl.float32, bitcast=True)
        # FP8 -> FP32
        acc = acc / p_scale_t

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
    offs_l_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
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
def _bwd_preprocess_use_o_mxfp8(
    out_ptr,
    do_ptr,
    do_fp8_ptr,
    do_scale_ptr,
    delta_ptr,
    use_mxfp8: tl.constexpr,
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
    stride_doscalem,
    stride_doscaled,
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
    QUANT_BLOCK_SIZE: tl.constexpr,
):
    """
    load o, do and compute delta
    """
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    if F8_BWD_DTYPE == tl.float8e4nv:
        USE_ASM: tl.constexpr = True
    else:
        USE_ASM: tl.constexpr = False

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
    o_offset = out_ptr + off_z * stride_oz + off_h * stride_oh + q_start * stride_om
    do_offset = do_ptr + off_z * stride_doz + off_h * stride_doh + q_start * stride_dom

    # compute pointers
    out_ptrs = o_offset + off_m[:, None] * stride_om + off_d_v[None, :] * stride_ok
    do_ptrs = do_offset + off_m[:, None] * stride_dom + off_d_v[None, :] * stride_dok

    # load
    o = tl.load(out_ptrs, mask=mask_o, other=0.0).to(tl.float32)
    do = tl.load(do_ptrs, mask=mask_o, other=0.0).to(tl.float32)

    # compute delta
    delta = tl.sum(o * do, axis=1)

    # write-back delta
    off_d_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    delta_offset = (
        delta_ptr
        + off_z * stride_deltaz
        + off_h * stride_deltah
        + q_start * stride_deltam
    )
    delta_ptrs = delta_offset + off_d_m * stride_deltam
    tl.store(delta_ptrs, delta, mask=mask_m)

    if use_mxfp8:
        do_scale = _calculate_scales(
            do, BLOCK_M, BLOCK_DMODEL_V, QUANT_BLOCK_SIZE, True, F8_BWD_DTYPE
        )

        do_fp8 = _pack_fp8(
            do,
            do_scale,
            None,
            None,
            BLOCK_M,
            BLOCK_DMODEL_V,
            QUANT_BLOCK_SIZE,
            True,
            False,
            USE_ASM,
            F8_BWD_DTYPE,
        )

        do_fp8_offset = (
            do_fp8_ptr + off_z * stride_doz + off_h * stride_doh + q_start * stride_dom
        )
        do_fp8_ptrs = (
            do_fp8_offset + off_m[:, None] * stride_dom + off_d_v[None, :] * stride_dok
        )

        tl.store(do_fp8_ptrs, do_fp8, mask=mask_o)

        off_do_scale_m = tl.arange(0, BLOCK_M // QUANT_BLOCK_SIZE)
        off_do_scale_d = tl.arange(0, BLOCK_DMODEL_V // QUANT_BLOCK_SIZE)
        do_scale_offset = (
            do_scale_ptr
            + off_z * stride_doscalez
            + off_h * stride_doscaleh
            + pid_m * BLOCK_M // QUANT_BLOCK_SIZE * stride_doscalem
            + off_do_scale_m[:, None] * stride_doscalem
            + off_do_scale_d[None, :] * stride_doscaled
        )

        tl.store(do_scale_offset, do_scale)


def get_autotune_bwd_configs():
    return [
        triton.Config(
            {},
            num_stages=1,
            num_warps=4,
        ),
    ], ["BLOCK_DMODEL", "CAUSAL", "use_mxfp8"]


autotune_bwd_configs, autotune_bwd_keys = get_autotune_bwd_configs()


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
    stride_qdescalem,
    stride_dodescalem,
    l_offset,
    d_offset,
    stride_ldm,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL_QK: tl.constexpr,
    BLOCK_DMODEL_V: tl.constexpr,
    scales_num_block_m: tl.constexpr,  # number of scale block per BLOCK_M
    scales_num_block_n: tl.constexpr,  # number of scale block per BLOCK_N
    scales_num_block_d_qk: tl.constexpr,  # number of scale block per BLOCK_D_QK
    scales_num_block_d_v: tl.constexpr,  # number of scale block per BLOCK_D_V
    SCALE_NUM_PER_QUANT_BLK: tl.constexpr,
    SCALE_NUM_PER_D_QK: tl.constexpr,
    SCALE_NUM_PER_D_V: tl.constexpr,
    SCALE_NUM_PER_N: tl.constexpr,
    SCALE_NUM_PER_M: tl.constexpr,
    q_scale_ptr_2d_base,
    q_scale_ptr_1d_ds_base,
    q_scale_ptr_1d_ds_base_T,
    do_scale_ptr_2d_base,
    do_scale_ptr_1d_ds_base,
    do_scale_ptr_1d_ds_base_T,
    k_scale,
    v_scale,
    sm_scale: tl.constexpr,
    p_scale: tl.constexpr,
    log_p_scale: tl.constexpr,
    lo: tl.constexpr,
    num_block_m: tl.constexpr,
    causal_boundary: tl.constexpr,
    use_mxfp8: tl.constexpr,
    USE_EXP2: tl.constexpr,
    N_CTX_Q: tl.constexpr,
    N_CTX_K: tl.constexpr,
    CAUSAL: tl.constexpr,
    F8_BWD_DTYPE: tl.constexpr,
    QUANT_BLOCK_SIZE: tl.constexpr,
    FLOAT_DTYPE: tl.constexpr,
    QUANT_SIZE: tl.constexpr,
):
    if FLOAT_DTYPE == "e4m3":
        USE_ASM: tl.constexpr = True
    else:
        USE_ASM: tl.constexpr = False

    if F8_BWD_DTYPE == tl.float8e4nv:
        DO_USE_ASM: tl.constexpr = True
        DO_DTYPE: tl.constexpr = "e4m3"
    else:
        DO_USE_ASM: tl.constexpr = False
        DO_DTYPE: tl.constexpr = "e5m2"

    if scales_num_block_m == 1 and scales_num_block_d_qk == 1:
        TRANS_Q_SCALE_BLK: tl.constexpr = False
    else:
        TRANS_Q_SCALE_BLK: tl.constexpr = True

    if use_mxfp8:
        k_scale = k_scale.to(tl.uint8)
        v_scale = v_scale.to(tl.uint8)
        p_scale_b = tl.zeros([BLOCK_N, SCALE_NUM_PER_M], dtype=tl.uint8) + 127
        p_scale_b = p_scale_b.to(tl.uint8)
    else:
        p_scale_b = 0.0

    # loop over rows
    for start_m in range(lo, num_block_m * BLOCK_M, BLOCK_M):
        # can_skip_causal_block = start_m < causal_boundary
        offs_m = start_m + tl.arange(0, BLOCK_M)
        q_ptrs = q_offset + offs_m[:, None] * stride_qm + offs_d_qk[None, :] * stride_qk
        do_ptrs = (
            do_offset + offs_m[:, None] * stride_dom + offs_d_v[None, :] * stride_dok
        )

        # Important: keep parentheses; `&` has higher precedence than comparisons in Triton.
        mask_m = (offs_m >= 0) & (offs_m < N_CTX_Q)
        q_mask = mask_m[:, None] & mask_d_qk[None, :]
        do_mask = mask_m[:, None] & mask_d_v[None, :]

        if use_mxfp8:
            blk_q_scale_1d_ds = tl.load(q_scale_ptr_1d_ds_base)
            if TRANS_Q_SCALE_BLK:
                blk_q_scale_1d_ds_T = tl.load(q_scale_ptr_1d_ds_base_T)
            else:
                blk_q_scale_1d_ds_T = blk_q_scale_1d_ds
            blk_q_scale_2d = tl.load(q_scale_ptr_2d_base)

            blk_do_scale_1d_ds = tl.load(do_scale_ptr_1d_ds_base)
            if TRANS_Q_SCALE_BLK:
                blk_do_scale_1d_ds_T = tl.load(do_scale_ptr_1d_ds_base_T)
            else:
                blk_do_scale_1d_ds_T = blk_do_scale_1d_ds
            blk_do_scale_2d = tl.load(do_scale_ptr_2d_base)

        else:
            blk_q_scale_1d_ds = 1.0
            blk_q_scale_1d_ds_T = 1.0
            blk_q_scale_2d = 1.0
            blk_do_scale_1d_ds = 1.0
            blk_do_scale_1d_ds_T = 1.0
            blk_do_scale_2d = 1.0

        q = tl.load(q_ptrs, mask=q_mask, other=0.0)

        if use_mxfp8:
            if (SCALE_NUM_PER_D_QK) % 2 == 0:
                qk = tl.dot_scaled(
                    q,
                    blk_q_scale_1d_ds,
                    FLOAT_DTYPE,
                    k,
                    k_scale,
                    FLOAT_DTYPE,
                    out_dtype=tl.float32,
                )
            else:
                # can fuse with sm_scale
                q_descaled = _unpack_fp8(
                    q,
                    blk_q_scale_2d,
                    tl.float32,
                    BLOCK_M,
                    BLOCK_DMODEL_QK,
                    QUANT_BLOCK_SIZE,
                    True,
                    USE_ASM,
                )
                k_descaled = _unpack_fp8(
                    k,
                    k_scale,
                    tl.float32,
                    BLOCK_DMODEL_QK,
                    BLOCK_N,
                    QUANT_BLOCK_SIZE,
                    True,
                    USE_ASM,
                )
                qk = tl.dot(
                    q_descaled, k_descaled, out_dtype=tl.float32, allow_tf32=False
                )
        else:
            qk = tl.dot(q, k, out_dtype=tl.float32, allow_tf32=False)

        if CAUSAL:
            # if not can_skip_causal_block:
            col_offset = N_CTX_Q - N_CTX_K
            causal_mask = offs_m[:, None] >= (col_offset + offs_n[None, :])
            qk = tl.where(causal_mask, qk, float("-inf"))

        l_ptrs = l_offset + offs_m * stride_ldm
        l_i = tl.load(l_ptrs, mask=mask_m, other=0.0)

        # compute p
        if USE_EXP2:
            RCP_LN2: tl.constexpr = 1.4426950408889634
            qk *= sm_scale * RCP_LN2
            l_i *= RCP_LN2
            p = tl.math.exp2(qk - l_i[:, None] + p_scale - 127)
        else:
            qk *= sm_scale
            p = tl.math.exp(qk - l_i[:, None] + log_p_scale)

        do = tl.load(do_ptrs, mask=do_mask, other=0.0)

        # compute dp
        if use_mxfp8:
            if (SCALE_NUM_PER_D_V) % 2 == 0:
                # tranfer non to zero
                do = do.to(tl.float32).to(F8_BWD_DTYPE)
                dp = tl.dot_scaled(
                    do,
                    blk_do_scale_1d_ds.to(tl.uint8),
                    DO_DTYPE,
                    v,
                    v_scale.to(tl.uint8),
                    FLOAT_DTYPE,
                    out_dtype=tl.float32,
                )
            else:
                do_descaled = _unpack_fp8(
                    do,
                    blk_do_scale_2d,
                    tl.float32,
                    BLOCK_M,
                    BLOCK_DMODEL_V,
                    QUANT_BLOCK_SIZE,
                    True,
                    DO_USE_ASM,
                )
                v_descaled = _unpack_fp8(
                    v,
                    v_scale,
                    tl.float32,
                    BLOCK_DMODEL_V,
                    BLOCK_N,
                    QUANT_BLOCK_SIZE,
                    True,
                    USE_ASM,
                )
                dp = tl.dot(
                    do_descaled, v_descaled, allow_tf32=False, out_dtype=tl.float32
                )
        else:
            dp = tl.dot(do, v, out_dtype=tl.float32, allow_tf32=False)

        d_ptrs = d_offset + offs_m * stride_ldm
        Di = tl.load(d_ptrs, mask=mask_m, other=0.0)
        ds = p * (dp - Di[:, None])

        if use_mxfp8:
            if (SCALE_NUM_PER_M) % 2 == 0:
                dv = tl.dot_scaled(
                    tl.trans(p).to(q.dtype),
                    p_scale_b,
                    FLOAT_DTYPE,
                    do,
                    blk_do_scale_1d_ds_T,
                    DO_DTYPE,
                    dv,
                    out_dtype=tl.float32,
                )
            else:
                if (SCALE_NUM_PER_D_V) % 2 == 0:
                    do_descaled = _unpack_fp8(
                        do,
                        blk_do_scale_2d,
                        tl.float32,
                        BLOCK_M,
                        BLOCK_DMODEL_V,
                        QUANT_BLOCK_SIZE,
                        True,
                        DO_USE_ASM,
                    )
                dv += tl.dot(
                    tl.trans(p), do_descaled, out_dtype=tl.float32, allow_tf32=False
                )

        else:
            # compute dv
            dv += tl.dot(
                tl.trans(p.to(k.dtype)), do, out_dtype=tl.float32, allow_tf32=False
            )

        # compute dk = dot(ds.T, q)
        if use_mxfp8:
            if (SCALE_NUM_PER_M) % 2 == 0:
                ds_T = ds.T
                ds_scale = _calculate_scales(
                    ds_T, BLOCK_N, BLOCK_M, QUANT_SIZE, False, q.dtype
                )

                ds_scalsed = _pack_fp8(
                    ds_T,
                    ds_scale,
                    None,
                    None,
                    BLOCK_N,
                    BLOCK_M,
                    QUANT_SIZE,
                    False,
                    False,
                    USE_ASM,
                    q.dtype,
                )
                dk = tl.dot_scaled(
                    ds_scalsed,
                    ds_scale,
                    FLOAT_DTYPE,
                    q,
                    blk_q_scale_1d_ds_T,
                    FLOAT_DTYPE,
                    dk,
                    out_dtype=tl.float32,
                )
            else:
                if (SCALE_NUM_PER_D_QK) % 2 == 0:
                    q_descaled = _unpack_fp8(
                        q,
                        blk_q_scale_2d,
                        tl.float32,
                        BLOCK_M,
                        BLOCK_DMODEL_QK,
                        QUANT_BLOCK_SIZE,
                        True,
                        USE_ASM,
                    )
                _dk = tl.dot(
                    tl.trans(ds), q_descaled, out_dtype=tl.float32, allow_tf32=False
                )
                dk += _dk
        else:
            _dk = tl.dot(
                tl.trans(ds).to(q.dtype), q, out_dtype=tl.float32, allow_tf32=False
            )
            dk += _dk

        if use_mxfp8:
            q_scale_ptr_2d_base += scales_num_block_m * stride_qdescalem
            q_scale_ptr_1d_ds_base += scales_num_block_m * stride_qdescalem
            q_scale_ptr_1d_ds_base_T += scales_num_block_m * stride_qdescalem
            do_scale_ptr_1d_ds_base += scales_num_block_m * stride_dodescalem
            do_scale_ptr_1d_ds_base_T += scales_num_block_m * stride_dodescalem
            do_scale_ptr_2d_base += scales_num_block_m * stride_dodescalem

    return dk, dv


@triton.autotune(
    configs=autotune_bwd_configs,
    key=autotune_bwd_keys,
)
@triton.jit
def _bwd_kernel_dkdv_mxfp8(
    Q,
    K,
    V,
    p_scale: tl.constexpr,
    log_p_scale: tl.constexpr,
    sm_scale: tl.constexpr,
    q_scale_ptr,
    k_scale_ptr,
    v_scale_ptr,
    do_scale_ptr,
    Out,
    DO,
    DQ,
    DK,
    DV,
    LSE,
    Delta,
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
    stride_dodescalez,
    stride_dodescaleh,
    stride_dodescalem,
    stride_dodescaled,
    stride_qdescalez,
    stride_qdescaleh,
    stride_qdescalem,
    stride_qdescaled,
    stride_kdescalez,
    stride_kdescaleh,
    stride_kdescalem,
    stride_kdescaled,
    stride_vdescalez,
    stride_vdescaleh,
    stride_vdescalem,
    stride_vdescaled,
    Z: tl.constexpr,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q: tl.constexpr,
    max_seqlen_k: tl.constexpr,
    num_block_m: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL_QK: tl.constexpr,
    BLOCK_DMODEL_V: tl.constexpr,
    ACTUAL_BLOCK_DMODEL_QK: tl.constexpr,
    ACTUAL_BLOCK_DMODEL_V: tl.constexpr,
    CAUSAL: tl.constexpr,
    USE_EXP2: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    use_mxfp8: tl.constexpr,
    F8_BWD_DTYPE: tl.constexpr,
    QUANT_BLOCK_SIZE: tl.constexpr,
    QUANT_SIZE: tl.constexpr,
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
    FLOAT_DTYPE: tl.constexpr = "e4m3"
    if Q.dtype.element_ty == tl.float8e5:
        FLOAT_DTYPE: tl.constexpr = "e5m2"

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
    adj_delta = off_z * stride_ldz + off_h_q * stride_ldh + q_start * stride_ldm
    l_offset = LSE + adj_delta
    d_offset = Delta + adj_delta

    # output tensor offsets
    # sume dk and dv
    dk_offset = DK + off_z * stride_kz + off_h_k * stride_kh + k_start * stride_kn
    dv_offset = DV + off_z * stride_vz + off_h_k * stride_vh + k_start * stride_vn

    if CAUSAL:
        causal_boundary = start_n * BLOCK_N - BLOCK_M
        lo = (causal_boundary + BLOCK_M) // BLOCK_M * BLOCK_M
    else:
        causal_boundary = 0
        lo = 0

    if use_mxfp8:
        scales_num_block_m: tl.constexpr = BLOCK_M // QUANT_BLOCK_SIZE
        scales_num_block_n: tl.constexpr = BLOCK_N // QUANT_BLOCK_SIZE
        scales_num_block_d_qk: tl.constexpr = BLOCK_DMODEL_QK // QUANT_BLOCK_SIZE
        scales_num_block_d_v: tl.constexpr = BLOCK_DMODEL_V // QUANT_BLOCK_SIZE

        scale_offs_m = tl.arange(0, scales_num_block_m)
        scale_offs_n = tl.arange(0, scales_num_block_n)
        scale_offs_d_qk = tl.arange(0, scales_num_block_d_qk)
        scale_offs_d_v = tl.arange(0, scales_num_block_d_v)
    else:
        scales_num_block_m = 1
        scales_num_block_n = 1
        scales_num_block_d_qk = 1
        scales_num_block_d_v = 1

        scale_offs_m = tl.arange(0, 1)
        scale_offs_n = tl.arange(0, 1)
        scale_offs_d_qk = tl.arange(0, 1)
        scale_offs_d_v = tl.arange(0, 1)

    # scale number per block in this warp tile for mxfp
    SCALE_NUM_PER_QUANT_BLK: tl.constexpr = QUANT_BLOCK_SIZE // QUANT_SIZE
    # scale number per D QK in this warp tile for mxfp
    SCALE_NUM_PER_D_QK: tl.constexpr = scales_num_block_d_qk * SCALE_NUM_PER_QUANT_BLK
    # scale number per D V in this warp tile for mxfp
    SCALE_NUM_PER_D_V: tl.constexpr = scales_num_block_d_v * SCALE_NUM_PER_QUANT_BLK
    # scale number per N in this warp tile for mxfp
    SCALE_NUM_PER_N: tl.constexpr = scales_num_block_n * SCALE_NUM_PER_QUANT_BLK
    # scale number per N in this warp tile for mxfp
    SCALE_NUM_PER_M: tl.constexpr = scales_num_block_m * SCALE_NUM_PER_QUANT_BLK

    if use_mxfp8:
        scale_offs_m_b = tl.arange(0, BLOCK_M)
        scale_offs_n_b = tl.arange(0, BLOCK_N)
        scale_offs_d_qk_b = tl.arange(0, BLOCK_DMODEL_QK)
        scale_offs_d_v_b = tl.arange(0, BLOCK_DMODEL_V)

        scale_offs_m_q = tl.arange(0, SCALE_NUM_PER_M)
        tl.arange(0, SCALE_NUM_PER_N)
        scale_offs_d_qk_q = tl.arange(0, SCALE_NUM_PER_D_QK)
        scale_offs_d_v_q = tl.arange(0, SCALE_NUM_PER_D_V)

        q_scale_ptr_offs = (
            q_scale_ptr
            + stride_qdescalez * off_z
            + stride_qdescaleh * off_h_q
            + q_start // QUANT_BLOCK_SIZE * stride_qdescalem
        )
        q_scale_ptr_2d_base = (
            q_scale_ptr_offs
            + scale_offs_m[:, None] * stride_qdescalem
            + scale_offs_d_qk[None, :] * stride_qdescaled
        )
        q_scale_ptr_1d_ds_base = (
            q_scale_ptr_offs
            + scale_offs_m_b[:, None] // QUANT_BLOCK_SIZE * stride_qdescalem
            + scale_offs_d_qk_q[None, :] // SCALE_NUM_PER_QUANT_BLK * stride_qdescaled
        )
        q_scale_ptr_1d_ds_base_T = (
            q_scale_ptr_offs
            + scale_offs_d_qk_b[:, None] // QUANT_BLOCK_SIZE * stride_qdescaled
            + scale_offs_m_q[None, :] // SCALE_NUM_PER_QUANT_BLK * stride_qdescalem
        )

        do_scale_ptr_offs = (
            do_scale_ptr
            + stride_dodescalez * off_z
            + stride_dodescaleh * off_h_q
            + q_start // QUANT_BLOCK_SIZE * stride_dodescalem
        )
        do_scale_ptr_1d_ds_base = (
            do_scale_ptr_offs
            + scale_offs_m_b[:, None] // QUANT_BLOCK_SIZE * stride_dodescalem
            + scale_offs_d_v_q[None, :] // SCALE_NUM_PER_QUANT_BLK * stride_dodescaled
        )
        do_scale_ptr_1d_ds_base_T = (
            do_scale_ptr_offs
            + scale_offs_d_v_b[:, None] // QUANT_BLOCK_SIZE * stride_dodescaled
            + scale_offs_m_q[None, :] // SCALE_NUM_PER_QUANT_BLK * stride_dodescalem
        )
        do_scale_ptr_2d_base = (
            do_scale_ptr_offs
            + scale_offs_m[:, None] * stride_dodescalem
            + scale_offs_d_v[None, :] * stride_dodescaled
        )

        q_scale_ptr_2d_base += (lo // QUANT_BLOCK_SIZE) * stride_qdescalem
        q_scale_ptr_1d_ds_base += (lo // QUANT_BLOCK_SIZE) * stride_qdescalem
        q_scale_ptr_1d_ds_base_T += (lo // QUANT_BLOCK_SIZE) * stride_qdescalem
        do_scale_ptr_1d_ds_base += (lo // QUANT_BLOCK_SIZE) * stride_dodescalem
        do_scale_ptr_1d_ds_base_T += (lo // QUANT_BLOCK_SIZE) * stride_dodescalem
        do_scale_ptr_2d_base += (lo // QUANT_BLOCK_SIZE) * stride_dodescalem
    else:
        q_scale_ptr_2d_base = q_scale_ptr
        q_scale_ptr_1d_ds_base = q_scale_ptr
        q_scale_ptr_1d_ds_base_T = q_scale_ptr
        do_scale_ptr_1d_ds_base = do_scale_ptr
        do_scale_ptr_1d_ds_base_T = do_scale_ptr
        do_scale_ptr_2d_base = do_scale_ptr

    offs_d_qk = tl.arange(0, BLOCK_DMODEL_QK)
    offs_d_v = tl.arange(0, BLOCK_DMODEL_V)
    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_n = offs_n < N_CTX_K
    mask_d_qk = offs_d_qk < ACTUAL_BLOCK_DMODEL_QK
    mask_d_v = offs_d_v < ACTUAL_BLOCK_DMODEL_V
    k_mask = mask_n[None, :] & mask_d_qk[:, None]
    v_mask = mask_n[None, :] & mask_d_v[:, None]

    k_ptrs = k_offset + offs_d_qk[:, None] * stride_kk + offs_n[None, :] * stride_kn
    v_ptrs = v_offset + offs_d_v[:, None] * stride_vk + offs_n[None, :] * stride_vn

    k = tl.load(k_ptrs, mask=k_mask, other=0.0)
    v = tl.load(v_ptrs, mask=v_mask, other=0.0)

    dk = tl.zeros([BLOCK_N, BLOCK_DMODEL_QK], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, BLOCK_DMODEL_V], dtype=tl.float32)

    if use_mxfp8:
        k_scale_offset_base = (
            k_scale_ptr
            + off_z * stride_kdescalez
            + off_h_k * stride_kdescaleh
            + k_start // QUANT_BLOCK_SIZE * stride_kdescalem
            + start_n * stride_kdescalem * scales_num_block_n
        )
        v_scale_offset_base = (
            v_scale_ptr
            + stride_vdescalez * off_z
            + stride_vdescaleh * off_h_k
            + k_start // QUANT_BLOCK_SIZE * stride_vdescalem
            + start_n * stride_vdescalem * scales_num_block_n
        )

        if (SCALE_NUM_PER_D_QK) % 2 == 0:
            k_scale_offset = (
                k_scale_offset_base
                + scale_offs_n_b[:, None] // QUANT_BLOCK_SIZE * stride_kdescalem
                + scale_offs_d_qk_q[None, :]
                // SCALE_NUM_PER_QUANT_BLK
                * stride_kdescaled
            )
        else:
            k_scale_offset = (
                k_scale_offset_base
                + scale_offs_d_qk[:, None] * stride_kdescaled
                + scale_offs_n[None, :] * stride_kdescalem
            )

        if (SCALE_NUM_PER_D_V) % 2 == 0:
            v_scale_offset = (
                v_scale_offset_base
                + scale_offs_n_b[:, None] // QUANT_BLOCK_SIZE * stride_vdescalem
                + scale_offs_d_v_q[None, :]
                // SCALE_NUM_PER_QUANT_BLK
                * stride_vdescaled
            )
        else:
            v_scale_offset = (
                v_scale_offset_base
                + scale_offs_d_v[:, None] * stride_vdescaled
                + scale_offs_n[None, :] * stride_vdescalem
            )

        blk_k_scale = tl.load(k_scale_offset)
        blk_v_scale = tl.load(v_scale_offset)
    else:
        blk_k_scale = 1.0
        blk_v_scale = 1.0

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
            stride_qdescalem,
            stride_dodescalem,
            l_offset,
            d_offset,
            stride_ldm,
            BLOCK_M,
            BLOCK_N,
            BLOCK_DMODEL_QK,
            BLOCK_DMODEL_V,
            scales_num_block_m,
            scales_num_block_n,
            scales_num_block_d_qk,
            scales_num_block_d_v,
            SCALE_NUM_PER_QUANT_BLK,
            SCALE_NUM_PER_D_QK,
            SCALE_NUM_PER_D_V,
            SCALE_NUM_PER_N,
            SCALE_NUM_PER_M,
            q_scale_ptr_2d_base,
            q_scale_ptr_1d_ds_base,
            q_scale_ptr_1d_ds_base_T,
            do_scale_ptr_2d_base,
            do_scale_ptr_1d_ds_base,
            do_scale_ptr_1d_ds_base_T,
            blk_k_scale,
            blk_v_scale,
            sm_scale,
            p_scale,
            log_p_scale,
            lo,
            num_block_m,
            causal_boundary,
            use_mxfp8,
            USE_EXP2,
            N_CTX_Q,
            N_CTX_K,
            CAUSAL,
            F8_BWD_DTYPE,
            QUANT_BLOCK_SIZE,
            FLOAT_DTYPE,
            QUANT_SIZE,
        )

        q_offset += stride_qh
        do_offset += stride_doh
        l_offset += stride_ldh
        d_offset += stride_ldh
        if use_mxfp8:
            q_scale_ptr_2d_base += stride_qdescaleh
            q_scale_ptr_1d_ds_base += stride_qdescaleh
            q_scale_ptr_1d_ds_base_T += stride_qdescaleh
            do_scale_ptr_1d_ds_base_T += stride_dodescaleh
            do_scale_ptr_2d_base += stride_dodescaleh
            do_scale_ptr_1d_ds_base += stride_dodescaleh

    if use_mxfp8:
        p_scale_t = tl.cast(p_scale, tl.uint32)
        p_scale_t = (p_scale_t << 23).to(tl.float32, bitcast=True)
        # FP8 -> FP32
        dv /= p_scale_t
    else:
        p_scale_t = 1.0

    dk *= sm_scale / p_scale_t

    dk_mask = mask_n[:, None] & mask_d_qk[None, :]
    dk_ptrs = dk_offset + offs_n[:, None] * stride_kn + offs_d_qk[None, :] * stride_kk
    tl.store(dk_ptrs, dk.to(DK.type.element_ty), mask=dk_mask)

    dv_mask = mask_n[:, None] & mask_d_v[None, :]
    dv_ptrs = dv_offset + offs_n[:, None] * stride_vn + offs_d_v[None, :] * stride_vk
    tl.store(dv_ptrs, dv.to(DV.type.element_ty), mask=dv_mask)


@triton.jit
def _attn_bwd_dq(
    dq,
    q,
    offs_d_qk,
    offs_d_v,
    offs_m,
    l_i,
    Di,
    do,
    mask_d_qk,
    mask_d_v,
    k_offset,
    v_offset,
    stride_kn,
    stride_kk,
    stride_vn,
    stride_vk,
    stride_kdescalem,
    stride_vdescalem,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL_QK: tl.constexpr,
    BLOCK_DMODEL_V: tl.constexpr,
    scales_num_block_m: tl.constexpr,  # number of scale block per BLOCK_M
    scales_num_block_n: tl.constexpr,  # number of scale block per BLOCK_N
    scales_num_block_d_qk: tl.constexpr,  # number of scale block per BLOCK_D_QK
    scales_num_block_d_v: tl.constexpr,  # number of scale block per BLOCK_D_V
    q_scale,
    k_scale_ptr_2d_base,
    k_scale_ptr_1d_ds_base,
    k_scale_ptr_1d_ds_base_T,
    do_scale,
    v_scale_ptr,
    sm_scale: tl.constexpr,
    p_scale: tl.constexpr,
    log_p_scale: tl.constexpr,
    hi: tl.constexpr,
    num_block_n: tl.constexpr,
    causal_boundary: tl.constexpr,
    use_mxfp8: tl.constexpr,
    USE_EXP2: tl.constexpr,
    F8_BWD_DTYPE: tl.constexpr,
    N_CTX_Q: tl.constexpr,
    N_CTX_K: tl.constexpr,
    CAUSAL: tl.constexpr,
    QUANT_BLOCK_SIZE: tl.constexpr,
    FLOAT_DTYPE: tl.constexpr,
    QUANT_SIZE: tl.constexpr,
):
    tl.full([1], -1, dtype=tl.int32)

    RCP_LN2: tl.constexpr = 1.4426950408889634
    if USE_EXP2:
        l_i *= RCP_LN2

    if FLOAT_DTYPE == "e4m3":
        USE_ASM: tl.constexpr = True
    else:
        USE_ASM: tl.constexpr = False

    if F8_BWD_DTYPE == tl.float8e4nv:
        DO_USE_ASM: tl.constexpr = True
        DO_DTYPE: tl.constexpr = "e4m3"
    else:
        DO_USE_ASM: tl.constexpr = False
        DO_DTYPE: tl.constexpr = "e5m2"

    if use_mxfp8:
        do_scale = do_scale.to(tl.uint8)

    # scale number per block in this warp tile for mxfp
    SCALE_NUM_PER_QUANT_BLK: tl.constexpr = QUANT_BLOCK_SIZE // QUANT_SIZE
    # scale number per D QK in this warp tile for mxfp
    SCALE_NUM_PER_D_QK: tl.constexpr = scales_num_block_d_qk * SCALE_NUM_PER_QUANT_BLK
    # scale number per D V in this warp tile for mxfp
    SCALE_NUM_PER_D_V: tl.constexpr = scales_num_block_d_v * SCALE_NUM_PER_QUANT_BLK
    # scale number per N in this warp tile for mxfp
    SCALE_NUM_PER_N: tl.constexpr = scales_num_block_n * SCALE_NUM_PER_QUANT_BLK
    # scale number per N in this warp tile for mxfp
    scales_num_block_m * SCALE_NUM_PER_QUANT_BLK

    # if block size can be divided by quant block size, the scale wont be tranposed.
    if scales_num_block_n == 1 and scales_num_block_d_v == 1:
        pass
    else:
        pass

    if scales_num_block_n == 1 and scales_num_block_d_qk == 1:
        pass
    else:
        pass

    # loop over rows
    for start_n in range(0, hi, BLOCK_N):
        # can_skip_causal_block = start_n < causal_boundary
        offs_n = start_n + tl.arange(0, BLOCK_N)

        mask_n = offs_n < N_CTX_K
        mask_k = mask_n[:, None] & mask_d_qk[None, :]
        mask_v = mask_n[None, :] & mask_d_v[:, None]

        k_ptrs = k_offset + offs_n[:, None] * stride_kn + offs_d_qk[None, :] * stride_kk
        v_ptrs = v_offset + offs_d_v[:, None] * stride_vk + offs_n[None, :] * stride_vn

        k = tl.load(k_ptrs, mask=mask_k, other=0.0)
        v = tl.load(v_ptrs, mask=mask_v, other=0.0)

        if use_mxfp8:
            if (SCALE_NUM_PER_D_QK) % 2 == 0:
                blk_k_scale = tl.load(k_scale_ptr_1d_ds_base)
                qk = tl.dot_scaled(
                    q,
                    q_scale.to(tl.uint8),
                    FLOAT_DTYPE,
                    tl.trans(k),
                    blk_k_scale.to(tl.uint8),
                    FLOAT_DTYPE,
                    out_dtype=tl.float32,
                )
            else:
                blk_k_scale = tl.load(k_scale_ptr_2d_base)
                q_descaled = _unpack_fp8(
                    q,
                    q_scale,
                    tl.float32,
                    BLOCK_M,
                    BLOCK_DMODEL_QK,
                    QUANT_BLOCK_SIZE,
                    True,
                    USE_ASM,
                )
                k_descaled = _unpack_fp8(
                    k,
                    blk_k_scale,
                    tl.float32,
                    BLOCK_N,
                    BLOCK_DMODEL_QK,
                    QUANT_BLOCK_SIZE,
                    True,
                    USE_ASM,
                )
                qk = tl.dot(
                    q_descaled,
                    tl.trans(k_descaled),
                    out_dtype=tl.float32,
                    allow_tf32=False,
                )

        else:
            kt = tl.trans(k)
            qk = tl.dot(q, kt, out_dtype=tl.float32, allow_tf32=False)

        if CAUSAL:
            # if not can_skip_causal_block:
            col_offset = N_CTX_Q - N_CTX_K
            causal_mask = offs_m[:, None] >= (col_offset + offs_n[None, :])
            qk = tl.where(causal_mask, qk, float("-inf"))

        # compute p
        if USE_EXP2:
            qk *= sm_scale * RCP_LN2
            p = tl.math.exp2(qk - l_i[:, None] + p_scale - 127)
        else:
            qk *= sm_scale
            p = tl.math.exp(qk - l_i[:, None] + log_p_scale)

        # compute dp
        if use_mxfp8:
            blk_v_scale = tl.load(v_scale_ptr)
            if (SCALE_NUM_PER_D_V) % 2 == 0:
                do = do.to(tl.float32).to(F8_BWD_DTYPE)
                dp = tl.dot_scaled(
                    do,
                    do_scale.to(tl.uint8),
                    DO_DTYPE,
                    v,
                    blk_v_scale.to(tl.uint8),
                    FLOAT_DTYPE,
                    out_dtype=tl.float32,
                )
            else:
                do_descaled = _unpack_fp8(
                    do,
                    do_scale,
                    tl.float32,
                    BLOCK_M,
                    BLOCK_DMODEL_V,
                    QUANT_BLOCK_SIZE,
                    True,
                    DO_USE_ASM,
                )
                v_descaled = _unpack_fp8(
                    v,
                    blk_v_scale,
                    tl.float32,
                    BLOCK_DMODEL_V,
                    BLOCK_N,
                    QUANT_BLOCK_SIZE,
                    True,
                    USE_ASM,
                )
                dp = tl.dot(
                    do_descaled, v_descaled, allow_tf32=False, out_dtype=tl.float32
                )
        else:
            dp = tl.dot(do, v, out_dtype=tl.float32, allow_tf32=False)

        ds = p * (dp - Di[:, None])

        if use_mxfp8:
            if (SCALE_NUM_PER_N) % 2 == 0:
                blk_k_scale = tl.load(k_scale_ptr_1d_ds_base_T)
                ds_scale = _calculate_scales(
                    ds, BLOCK_M, BLOCK_N, QUANT_SIZE, False, q.dtype
                )
                ds = _pack_fp8(
                    ds,
                    ds_scale,
                    None,
                    None,
                    BLOCK_M,
                    BLOCK_N,
                    QUANT_SIZE,
                    False,
                    False,
                    USE_ASM,
                    q.dtype,
                )

                dq = tl.dot_scaled(
                    ds,
                    ds_scale,
                    FLOAT_DTYPE,
                    k,
                    blk_k_scale.to(tl.uint8),
                    FLOAT_DTYPE,
                    dq.to(tl.float32),
                    out_dtype=tl.float32,
                )

            else:
                if (SCALE_NUM_PER_D_QK) % 2 == 0:
                    blk_k_scale = tl.load(k_scale_ptr_2d_base)
                    k_descaled = _unpack_fp8(
                        k,
                        blk_k_scale,
                        tl.float32,
                        BLOCK_N,
                        BLOCK_DMODEL_QK,
                        QUANT_BLOCK_SIZE,
                        True,
                        USE_ASM,
                    )

                _dq = tl.dot(ds, k_descaled, out_dtype=tl.float32, allow_tf32=False)
                dq += _dq

            k_scale_ptr_2d_base += stride_kdescalem * scales_num_block_n
            k_scale_ptr_1d_ds_base += stride_kdescalem * scales_num_block_n
            k_scale_ptr_1d_ds_base_T += stride_kdescalem * scales_num_block_n
            v_scale_ptr += stride_vdescalem * scales_num_block_n
        else:
            ds = ds.to(q.dtype)
            _dq = tl.dot(ds, k, out_dtype=tl.float32, allow_tf32=False)
            dq += _dq

        k_ptrs += stride_kn * BLOCK_N
        v_ptrs += stride_vn * BLOCK_N

    return dq


@triton.autotune(
    configs=autotune_bwd_configs,
    key=autotune_bwd_keys,
)
@triton.jit
def _bwd_kernel_dq_mxfp8(
    Q,
    K,
    V,
    sm_scale: tl.constexpr,
    p_scale: tl.constexpr,
    log_p_scale: tl.constexpr,
    q_scale_ptr,
    k_scale_ptr,
    v_scale_ptr,
    do_scale_ptr,
    Out,
    DO,
    DQ,
    DK,
    DV,
    LSE,
    Delta,
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
    stride_dodescalez,
    stride_dodescaleh,
    stride_dodescalem,
    stride_dodescaled,
    stride_qdescalez,
    stride_qdescaleh,
    stride_qdescalem,
    stride_qdescaled,
    stride_kdescalez,
    stride_kdescaleh,
    stride_kdescalem,
    stride_kdescaled,
    stride_vdescalez,
    stride_vdescaleh,
    stride_vdescalem,
    stride_vdescaled,
    Z: tl.constexpr,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q: tl.constexpr,
    max_seqlen_k: tl.constexpr,
    num_block_m: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL_QK: tl.constexpr,
    BLOCK_DMODEL_V: tl.constexpr,
    ACTUAL_BLOCK_DMODEL_QK: tl.constexpr,
    ACTUAL_BLOCK_DMODEL_V: tl.constexpr,
    CAUSAL: tl.constexpr,
    USE_EXP2: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    use_mxfp8: tl.constexpr,
    F8_BWD_DTYPE: tl.constexpr,
    QUANT_BLOCK_SIZE: tl.constexpr,
    QUANT_SIZE: tl.constexpr,
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

    FLOAT_DTYPE: tl.constexpr = "e4m3"
    if Q.dtype.element_ty == tl.float8e5:
        FLOAT_DTYPE: tl.constexpr = "e5m2"

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
    adj_delta = off_z * stride_ldz + off_h_q * stride_ldh + q_start * stride_ldm
    l_offset = LSE + adj_delta
    d_offset = Delta + adj_delta

    # output tensor offsets
    dq_offset = DQ + off_z * stride_qz + off_h_q * stride_qh + q_start * stride_qm

    if use_mxfp8:
        scales_num_block_m: tl.constexpr = BLOCK_M // QUANT_BLOCK_SIZE
        scales_num_block_n: tl.constexpr = BLOCK_N // QUANT_BLOCK_SIZE
        scales_num_block_d_qk: tl.constexpr = BLOCK_DMODEL_QK // QUANT_BLOCK_SIZE
        scales_num_block_d_v: tl.constexpr = BLOCK_DMODEL_V // QUANT_BLOCK_SIZE

        scale_offs_m = tl.arange(0, scales_num_block_m)
        scale_offs_n = tl.arange(0, scales_num_block_n)
        scale_offs_d_qk = tl.arange(0, scales_num_block_d_qk)
        scale_offs_d_v = tl.arange(0, scales_num_block_d_v)
    else:
        scales_num_block_m = 1
        scales_num_block_n = 1
        scales_num_block_d_qk = 1
        scales_num_block_d_v = 1

        scale_offs_m = tl.arange(0, 1)
        scale_offs_n = tl.arange(0, 1)
        scale_offs_d_qk = tl.arange(0, 1)
        scale_offs_d_v = tl.arange(0, 1)

    # scale number per block in this warp tile for mxfp
    SCALE_NUM_PER_QUANT_BLK: tl.constexpr = QUANT_BLOCK_SIZE // QUANT_SIZE
    # scale number per D QK in this warp tile for mxfp
    SCALE_NUM_PER_D_QK: tl.constexpr = scales_num_block_d_qk * SCALE_NUM_PER_QUANT_BLK
    # scale number per D V in this warp tile for mxfp
    SCALE_NUM_PER_D_V: tl.constexpr = scales_num_block_d_v * SCALE_NUM_PER_QUANT_BLK
    # scale number per N in this warp tile for mxfp
    SCALE_NUM_PER_N: tl.constexpr = scales_num_block_n * SCALE_NUM_PER_QUANT_BLK
    # scale number per N in this warp tile for mxfp
    SCALE_NUM_PER_M: tl.constexpr = scales_num_block_m * SCALE_NUM_PER_QUANT_BLK

    if use_mxfp8:
        scale_offs_m_b = tl.arange(0, BLOCK_M)
        scale_offs_n_b = tl.arange(0, BLOCK_N)
        scale_offs_d_qk_b = tl.arange(0, BLOCK_DMODEL_QK)
        tl.arange(0, BLOCK_DMODEL_V)

        tl.arange(0, SCALE_NUM_PER_M)
        scale_offs_n_q = tl.arange(0, SCALE_NUM_PER_N)
        scale_offs_d_qk_q = tl.arange(0, SCALE_NUM_PER_D_QK)
        scale_offs_d_v_q = tl.arange(0, SCALE_NUM_PER_D_V)

        k_scale_ptr_offs = (
            k_scale_ptr
            + stride_kdescalez * off_z
            + stride_kdescaleh * off_h_k
            + k_start // QUANT_BLOCK_SIZE * stride_kdescalem
        )
        k_scale_ptr_2d_base = (
            k_scale_ptr_offs
            + scale_offs_n[:, None] * stride_kdescalem
            + scale_offs_d_qk[None, :] * stride_kdescaled
        )
        k_scale_ptr_1d_ds_base = (
            k_scale_ptr_offs
            + scale_offs_n_b[:, None] // QUANT_BLOCK_SIZE * stride_kdescalem
            + scale_offs_d_qk_q[None, :] // SCALE_NUM_PER_QUANT_BLK * stride_kdescaled
        )
        k_scale_ptr_1d_ds_base_T = (
            k_scale_ptr_offs
            + scale_offs_d_qk_b[:, None] // QUANT_BLOCK_SIZE * stride_kdescaled
            + scale_offs_n_q[None, :] // SCALE_NUM_PER_QUANT_BLK * stride_kdescalem
        )

        v_scale_ptr_offs = (
            v_scale_ptr
            + stride_vdescalez * off_z
            + stride_vdescaleh * off_h_k
            + k_start // QUANT_BLOCK_SIZE * stride_vdescalem
        )
        if (SCALE_NUM_PER_D_V) % 2 == 0:
            v_scale_ptr_base = (
                v_scale_ptr_offs
                + scale_offs_n_b[:, None] // QUANT_BLOCK_SIZE * stride_vdescalem
                + scale_offs_d_v_q[None, :]
                // SCALE_NUM_PER_QUANT_BLK
                * stride_vdescaled
            )
        else:
            v_scale_ptr_base = (
                v_scale_ptr_offs
                + scale_offs_d_v[:, None] * stride_vdescaled
                + scale_offs_n[None, :] * stride_vdescalem
            )
    else:
        k_scale_ptr_2d_base = k_scale_ptr
        k_scale_ptr_1d_ds_base = k_scale_ptr
        k_scale_ptr_1d_ds_base_T = k_scale_ptr
        v_scale_ptr_base = v_scale_ptr

    if CAUSAL:
        causal_boundary = start_m * BLOCK_M - BLOCK_N
        # For a given Q block (start_m), keys up to (start_m+1)*BLOCK_M can contribute under causal masking.
        # Convert that limit to a K-block boundary in multiples of BLOCK_N.
        hi = (
            tl.minimum(cdiv_fn((start_m + 1) * BLOCK_M, BLOCK_N), num_block_n)
        ) * BLOCK_N

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

    if use_mxfp8:
        q_scale_offset_base = (
            q_scale_ptr
            + off_z * stride_qdescalez
            + off_h_q * stride_qdescaleh
            + q_start * scales_num_block_m * stride_qdescalem
            + start_m * stride_qdescalem * scales_num_block_m
        )
        if (SCALE_NUM_PER_D_QK) % 2 == 0:
            q_scale_offset = (
                q_scale_offset_base
                + scale_offs_m_b[:, None] // QUANT_BLOCK_SIZE * stride_qdescalem
                + scale_offs_d_qk_q[None, :]
                // SCALE_NUM_PER_QUANT_BLK
                * stride_qdescaled
            )
        else:
            q_scale_offset = (
                q_scale_offset_base
                + scale_offs_m[:, None] * stride_qdescalem
                + scale_offs_d_qk[None, :] * stride_qdescaled
            )
        blk_q_scale = tl.load(q_scale_offset)

        do_scale_offset_base = (
            do_scale_ptr
            + off_z * stride_dodescalez
            + off_h_q * stride_dodescaleh
            + q_start * scales_num_block_m * stride_dodescalem
            + start_m * stride_dodescalem * scales_num_block_m
        )
        if (SCALE_NUM_PER_D_V) % 2 == 0:
            do_scale_offset = (
                do_scale_offset_base
                + scale_offs_m_b[:, None] // QUANT_BLOCK_SIZE * stride_dodescalem
                + scale_offs_d_v_q[None, :]
                // SCALE_NUM_PER_QUANT_BLK
                * stride_dodescaled
            )
        else:
            do_scale_offset = (
                do_scale_offset_base
                + scale_offs_m[:, None] * stride_dodescalem
                + scale_offs_d_v[None, :] * stride_dodescaled
            )

        blk_do_scale = tl.load(do_scale_offset)
    else:
        blk_q_scale = 1.0
        blk_do_scale = 1.0

    l_ptrs = l_offset + offs_m * stride_ldm
    l_i = tl.load(l_ptrs, mask=mask_m, other=0.0)
    d_ptrs = d_offset + offs_m * stride_ldm
    Di = tl.load(d_ptrs, mask=mask_m, other=0.0)  # D stored in fp32

    dq = _attn_bwd_dq(
        dq,
        q,
        offs_d_qk,
        offs_d_v,
        offs_m,
        l_i,
        Di,
        do,
        mask_d_qk,
        mask_d_v,
        k_offset,
        v_offset,
        stride_kn,
        stride_kk,
        stride_vn,
        stride_vk,
        stride_kdescalem,
        stride_vdescalem,
        BLOCK_M,
        BLOCK_N,
        BLOCK_DMODEL_QK,
        BLOCK_DMODEL_V,
        scales_num_block_m,
        scales_num_block_n,
        scales_num_block_d_qk,
        scales_num_block_d_v,
        blk_q_scale,
        k_scale_ptr_2d_base,
        k_scale_ptr_1d_ds_base,
        k_scale_ptr_1d_ds_base_T,
        blk_do_scale,
        v_scale_ptr_base,
        sm_scale,
        p_scale,
        log_p_scale,
        hi,
        num_block_n,
        causal_boundary,
        use_mxfp8,
        USE_EXP2,
        F8_BWD_DTYPE,
        N_CTX_Q,
        N_CTX_K,
        CAUSAL,
        QUANT_BLOCK_SIZE,
        FLOAT_DTYPE,
        QUANT_SIZE,
    )

    if use_mxfp8:
        p_scale_t = tl.cast(p_scale, tl.uint32)
        p_scale_t = (p_scale_t << 23).to(tl.float32, bitcast=True)
    else:
        p_scale_t = 1.0
    dq_ptrs = dq_offset + offs_m[:, None] * stride_qm + offs_d_qk[None, :] * stride_qk
    dq *= sm_scale / p_scale_t
    tl.store(dq_ptrs, dq.to(DQ.type.element_ty), mask=mask_q)
