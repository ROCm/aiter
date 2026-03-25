# The kernels in this file are adapted from vLLM:
# https://github.com/vllm-project/vllm/blob/main/vllm/attention/ops/triton_unified_attention.py
import triton
import triton.language as tl
import torch
from aiter.ops.triton.utils.types import e4m3_dtype

float8_info = torch.finfo(e4m3_dtype)


@triton.jit
def fast_exp(x):
    RCP_LN2: tl.constexpr = 1.4426950408889634
    return tl.math.exp2(x * RCP_LN2)


@triton.jit
def cdiv_fn(x, y):
    return (x + y - 1) // y


@triton.jit
def apply_softcap(S, x):
    Sdiv = S / x
    p1 = tl.math.exp2(Sdiv)
    p2 = tl.math.exp2(-Sdiv)
    return x * (p1 - p2) / (p1 + p2)


@triton.jit
def find_seq_idx(
    query_start_len_ptr,
    target_idx,
    num_seqs,
    BLOCK_Q: tl.constexpr,
    use_q_block_mode: tl.constexpr,
):
    left: tl.int32 = 0
    right = num_seqs
    while left < right:
        mid = (left + right) // 2
        val = tl.load(query_start_len_ptr + mid)
        mid_val = val // BLOCK_Q + mid if use_q_block_mode else val

        if mid_val <= target_idx:
            left = mid + 1
        else:
            right = mid

    return left - 1


@triton.jit
def q_load(
    query_ptr,
    query_offset_0,
    query_offset_1,
    offs_d,
    query_stride_0,
    query_stride_1,
    query_mask_0,
    query_mask_1,
    dim_mask,
    Q_cache_modifier: tl.constexpr,
):
    query_offset = (
        query_offset_0[:, None] * query_stride_0
        + query_offset_1[:, None] * query_stride_1
        + offs_d[None, :]
    )
    return tl.load(
        query_ptr + query_offset,
        mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
        other=0.0,
        cache_modifier=Q_cache_modifier,
    )


@triton.jit
def kv_load(
    kv_cache_ptr,
    physical_block_idx,
    kv_head_idx,
    offs_d,
    seq_offset,
    stride_k_cache_0,
    stride_k_cache_1,
    stride_k_cache_2,
    stride_k_cache_3,
    dim_mask,
    tile_mask,
    BLOCK_SIZE: tl.constexpr,
    KV_cache_modifier: tl.constexpr,
):
    kv_offset = (
        physical_block_idx * stride_k_cache_0
        + kv_head_idx * stride_k_cache_2
        + offs_d * stride_k_cache_3
        + (seq_offset % BLOCK_SIZE) * stride_k_cache_1
    )

    return tl.load(
        kv_cache_ptr + kv_offset,
        mask=dim_mask & tile_mask,
        other=0.0,
        cache_modifier=KV_cache_modifier,
    )


@triton.jit
def qk_dot(
    Q,
    K,
    qk_scale,  #
    q_descale,
    k_descale,
    SAGE_MXFP4: tl.constexpr = False,
):
    if SAGE_MXFP4:
        return tl.dot_scaled(Q, q_descale, "e2m1", K, k_descale, "e2m1", fast_math=True)
    else:
        return qk_scale * tl.dot(Q, K)


@triton.jit
def kernel_unified_attention_2d(
    output_ptr,  # [num_tokens, num_query_heads, head_size]
    query_ptr,  # [num_tokens, num_query_heads, head_size]
    key_cache_ptr,  # [num_blks, blk_size, num_kv_heads, head_size]
    value_cache_ptr,  # [num_blks, blk_size, num_kv_heads, head_size]
    sink_ptr,  # [num_query_heads]
    block_tables_ptr,  # [num_seqs, max_num_blocks_per_seq]
    seq_lens_ptr,  # [num_seqs]
    alibi_slopes_ptr,  # [num_query_heads]
    qq_bias_ptr,  # [num_query_tokens, num_query_tokens]
    scale: tl.constexpr,  # float32
    q_scale,  # [num_tokens, num_query_heads, head_size // 32] (sage_mxfp4), None (no quantization) or scalar (per tensor fp8)
    k_scale,  # [num_blks, blk_size, num_kv_heads, head_size // 32] (sage_mxfp4), None (no quantization) or scalar (per tensor fp8)
    v_scale,  # [num_kv_heads, head_size] (sage_mxfp4), None (no quantization) or scalar (per tensor fp8)
    out_scale,  # float32
    softcap,  # float32
    num_query_heads: tl.constexpr,  # int
    num_queries_per_kv: tl.constexpr,  # int
    block_table_stride: tl.int64,  # int
    query_stride_0: tl.int64,  # int
    query_stride_1: tl.int64,  # int, should be equal to head_size
    query_scale_stride_0: tl.int64,
    query_scale_stride_1: tl.int64,
    query_scale_stride_2: tl.int64,
    output_stride_0: tl.int64,  # int
    output_stride_1: tl.int64,  # int, should be equal to head_size
    qq_bias_stride_0: tl.int64,  # int
    BLOCK_SIZE: tl.constexpr,  # int
    TILE_SIZE: tl.constexpr,  # int must be power of 2
    HEAD_SIZE: tl.constexpr,  # int
    HEAD_SIZE_PADDED: tl.constexpr,  # int, must be power of 2
    HEAD_SIZE_V: tl.constexpr,  # int
    HEAD_SIZE_V_PADDED: tl.constexpr,  # int, must be power of 2
    ROPE_SIZE: tl.constexpr,  # int
    ROPE_SIZE_PADDED: tl.constexpr,  # int, must be power of 2
    USE_ALIBI_SLOPES: tl.constexpr,  # bool
    USE_QQ_BIAS: tl.constexpr,  # bool
    USE_SOFTCAP: tl.constexpr,  # bool
    USE_SINKS: tl.constexpr,  # bool
    SLIDING_WINDOW: tl.constexpr,  # int
    stride_k_cache_0: tl.int64,  # int
    stride_k_cache_1: tl.int64,  # int
    stride_k_cache_2: tl.int64,  # int
    stride_k_cache_3: tl.constexpr,  # int
    stride_v_cache_0: tl.int64,  # int
    stride_v_cache_1: tl.int64,  # int
    stride_v_cache_2: tl.int64,  # int
    stride_v_cache_3: tl.constexpr,  # int
    stride_k_cache_scale_0: tl.int64,  # int
    stride_k_cache_scale_1: tl.int64,  # int
    stride_k_cache_scale_2: tl.int64,  # int
    stride_k_cache_scale_3: tl.int64,  # int
    stride_v_cache_scale_0: tl.int64,  # int
    stride_v_cache_scale_1: tl.int64,  # int
    query_start_len_ptr,  # [num_seqs+1]
    key_start_len_ptr,  # [num_seqs+1]
    BLOCK_Q: tl.constexpr,  # int
    num_seqs: tl.int32,
    BLOCK_M: tl.constexpr,  # int
    OUTPUT_FP8: tl.constexpr,  # bool
    FP8_MIN: tl.constexpr = float8_info.min,
    FP8_MAX: tl.constexpr = float8_info.max,
    ALL_DECODE: tl.constexpr = False,  # bool
    SAGE_MXFP4: tl.constexpr = False,  # bool
    KV_LAYOUT_IS_THD: tl.constexpr = True,  # bool
    PRELOAD_V: tl.constexpr = False,  # bool
    HAS_ROPE: tl.constexpr = False,  # bool
    IS_CAUSAL: tl.constexpr = True,  # bool
):
    kv_head_idx = tl.program_id(0)
    q_block_global_idx = tl.program_id(1)

    # needed to use exp2 (exp2 -> exp conversion)
    RCP_LN2 = 1.4426950408889634

    if not SAGE_MXFP4:
        _q_ds = tl.load(q_scale) if q_scale is not None else 1.0
        _k_ds = tl.load(k_scale) if k_scale is not None else 1.0
        sm_scale = scale if scale is not None else 1.0
        qk_scale = sm_scale * RCP_LN2 * _q_ds * _k_ds
    else:
        qk_scale = None

    seq_idx = find_seq_idx(
        query_start_len_ptr, q_block_global_idx, num_seqs, BLOCK_Q, True
    )

    q_block_start_idx = tl.load(query_start_len_ptr + seq_idx) // BLOCK_Q + seq_idx

    q_block_local_idx = q_block_global_idx - q_block_start_idx

    cur_batch_in_all_start_index = tl.load(query_start_len_ptr + seq_idx)
    cur_batch_in_all_stop_index = tl.load(query_start_len_ptr + seq_idx + 1)

    cur_batch_query_len = cur_batch_in_all_stop_index - cur_batch_in_all_start_index

    if q_block_local_idx * BLOCK_Q >= cur_batch_query_len:
        return

    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_SIZE_PADDED)
    offs_dv = tl.arange(0, HEAD_SIZE_V_PADDED)
    if SAGE_MXFP4:  # per token mxfp4
        offs_d_qk = tl.arange(0, HEAD_SIZE_PADDED // 2)
        offs_d_scale = tl.arange(0, HEAD_SIZE_PADDED // 32)
    else:
        offs_d_qk = tl.arange(0, HEAD_SIZE_PADDED)
        offs_d_scale = None

    if HAS_ROPE:
        if SAGE_MXFP4:  # per token mxfp4
            offs_rope = tl.arange(HEAD_SIZE // 2, (HEAD_SIZE + ROPE_SIZE_PADDED) // 2)
            offs_rope_scale = tl.arange(
                HEAD_SIZE // 32, (HEAD_SIZE + ROPE_SIZE_PADDED) // 32
            )
        else:
            offs_rope = tl.arange(HEAD_SIZE, HEAD_SIZE + ROPE_SIZE_PADDED)
            offs_rope_scale = None

    offs_t = tl.arange(0, TILE_SIZE)
    query_pos = q_block_local_idx * BLOCK_Q + offs_m // num_queries_per_kv

    query_offset_0 = cur_batch_in_all_start_index + query_pos
    query_offset_1 = kv_head_idx * num_queries_per_kv + offs_m % num_queries_per_kv

    # TODO: should be actually possible to remove this as HEAD_SIZE is always pow2. the non pow2 part is dealt with rope
    if HEAD_SIZE_PADDED != HEAD_SIZE:
        dim_mask = offs_d < HEAD_SIZE
        if SAGE_MXFP4:
            dim_mask_qk = offs_d_qk < (HEAD_SIZE // 2)
            scale_dim_mask = offs_d_scale < (HEAD_SIZE // 32)
        else:
            dim_mask_qk = dim_mask
            scale_dim_mask = dim_mask
    else:
        dim_mask = tl.full((1,), 1, dtype=tl.int1)
        dim_mask_qk = dim_mask
        scale_dim_mask = dim_mask

    if HEAD_SIZE_V_PADDED != HEAD_SIZE_V:
        dim_mask_v = offs_dv < HEAD_SIZE_V
    else:
        dim_mask_v = tl.full((1,), 1, dtype=tl.int1)

    query_mask_0 = query_pos < cur_batch_query_len
    query_mask_1 = query_offset_1 < num_query_heads

    if ALL_DECODE or BLOCK_M >= num_query_heads:
        Q_cache_modifier: tl.constexpr = ".cg"
    else:
        Q_cache_modifier: tl.constexpr = ""

    # Q : (BLOCK_M, HEAD_SIZE_PADDED)
    Q = q_load(
        query_ptr,
        query_offset_0,
        query_offset_1,
        offs_d_qk,
        query_stride_0,
        query_stride_1,
        query_mask_0,
        query_mask_1,
        dim_mask_qk,
        Q_cache_modifier,
    )

    q_scale_loaded = None
    if SAGE_MXFP4:
        q_scale_loaded = q_load(
            q_scale,
            query_offset_0,
            query_offset_1,
            offs_d_scale,
            query_scale_stride_0,
            query_scale_stride_1,
            query_mask_0,
            query_mask_1,
            scale_dim_mask,
            "",
        )

    if HAS_ROPE:
        if ROPE_SIZE_PADDED != ROPE_SIZE:
            if SAGE_MXFP4:
                rope_dim_mask = offs_rope < ((HEAD_SIZE + ROPE_SIZE) // 2)
                rope_scale_dim_mask = offs_rope_scale < ((HEAD_SIZE + ROPE_SIZE) // 32)
            else:
                rope_dim_mask = offs_rope < (HEAD_SIZE + ROPE_SIZE)
        else:
            rope_dim_mask = tl.full((1,), 1, dtype=tl.int1)
            rope_scale_dim_mask = tl.full((1,), 1, dtype=tl.int1)

        Q_rope = q_load(
            query_ptr,
            query_offset_0,
            query_offset_1,
            offs_rope,
            query_stride_0,
            query_stride_1,
            query_mask_0,
            query_mask_1,
            rope_dim_mask,
            "",
        )

        if SAGE_MXFP4:
            q_rope_scale_loaded = q_load(
                q_scale,
                query_offset_0,
                query_offset_1,
                offs_rope_scale,
                query_scale_stride_0,
                query_scale_stride_1,
                query_mask_0,
                query_mask_1,
                rope_scale_dim_mask,
                "",
            )

    if not USE_SINKS:
        M = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    else:
        # Prescale with RCP_LN2, needed for exp2
        M = (
            tl.load(
                sink_ptr + query_offset_1,
                mask=query_mask_1,
                other=float("-inf"),
            ).to(dtype=tl.float32)
            * RCP_LN2
        )

    L = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_SIZE_PADDED], dtype=tl.float32)

    # sequence len for this particular sequence
    seq_len = tl.load(seq_lens_ptr + seq_idx)

    # context length for this particular sequences
    context_len = seq_len - cur_batch_query_len

    # alibi slope for this head
    if USE_ALIBI_SLOPES:
        alibi_slope = tl.load(
            alibi_slopes_ptr + query_offset_1, mask=query_mask_1, other=0.0
        )

    # query-query attention bias
    if USE_QQ_BIAS:
        qq_bias_row_ptrs = (
            qq_bias_ptr + query_pos[:, None] * qq_bias_stride_0
        )  # shape: [BLOCK_M]

    
    if IS_CAUSAL:
        # compute the length of the longest sequence prefix spanned by any
        # query token in the current q_block (q_block_local_idx)
        max_seq_prefix_len = (
            context_len
            + q_block_local_idx * BLOCK_Q
            + (BLOCK_M - 1) // num_queries_per_kv
            + 1
        )

        # adjust for potential padding in the last q_block by considering the
        # actual sequence length
        max_seq_prefix_len = tl.minimum(max_seq_prefix_len, seq_len)
    else:
        max_seq_prefix_len = seq_len

    # calculate the number of tiles that need to be processed to
    # cover the longest sequence prefix (due to causal masking, tiles beyond
    # this prefix can be skipped)
    num_tiles = cdiv_fn(max_seq_prefix_len, TILE_SIZE)

    # ---- Sliding-window tile pruning --------------------
    # Default: keep previous global behavior
    tile_start = 0
    tile_end = num_tiles
    if SLIDING_WINDOW > 0:
        # Query rows covered by this Q-block
        qpos_lo = q_block_local_idx * BLOCK_Q
        qpos_hi = tl.minimum(
            qpos_lo + (BLOCK_M - 1) // num_queries_per_kv,
            cur_batch_query_len - 1,
        )
        # For sliding window, each query position q can only attend to
        # keys in the range [q_abs - SLIDING_WINDOW + 1, q_abs]
        # where q_abs = context_len + q
        # The union of allowed key positions for this Q-block is:
        # [context_len + qpos_lo - SLIDING_WINDOW + 1, context_len + qpos_hi]
        first_allowed_key = context_len + qpos_lo - SLIDING_WINDOW + 1
        last_allowed_key = context_len + qpos_hi
        # Convert to tile indices and clamp
        tile_start = tl.maximum(0, first_allowed_key // TILE_SIZE)
        tile_end = tl.minimum((last_allowed_key // TILE_SIZE) + 1, num_tiles)

    KV_cache_modifier: tl.constexpr = ".cg" if ALL_DECODE else ""
    # iterate through tiles (now limited to the sliding window range)

    
    if KV_LAYOUT_IS_THD: # No need to page inside the loop as its a contiguous layout.
        base_offset = tl.load(
            key_start_len_ptr + seq_idx
        ).to(tl.int64)
    else:
        block_table_offset = seq_idx * block_table_stride

    for j in range(tile_start, tile_end):
        seq_offset = j * TILE_SIZE + offs_t
        # to reduce the masking effect when not needed
        if TILE_SIZE == BLOCK_SIZE:
            tile_mask = tl.full((1,), 1, dtype=tl.int1)
        else:
            tile_mask = seq_offset < max_seq_prefix_len

        if not KV_LAYOUT_IS_THD: # Need to page inside the loop as its not a contiguous layout.
            base_offset = tl.load(
                block_tables_ptr + block_table_offset + seq_offset // BLOCK_SIZE
            ).to(tl.int64)

        # K : (HEAD_SIZE, TILE_SIZE)
        K = kv_load(
            key_cache_ptr,
            base_offset[None, :],
            kv_head_idx,
            offs_d_qk[:, None],
            seq_offset[None, :],
            stride_k_cache_0,
            stride_k_cache_1,
            stride_k_cache_2,
            stride_k_cache_3,
            dim_mask_qk[:, None],
            tile_mask[None, :],
            BLOCK_SIZE,
            KV_cache_modifier,
        )

        S = tl.zeros([BLOCK_M, TILE_SIZE], dtype=tl.float32)

        k_scale_loaded = None
        if SAGE_MXFP4:
            k_scale_loaded = kv_load(
                k_scale,
                base_offset[:, None],
                kv_head_idx,
                offs_d_scale[None, :],
                seq_offset[:, None],
                stride_k_cache_scale_0,
                stride_k_cache_scale_1,
                stride_k_cache_scale_2,
                stride_k_cache_scale_3,
                scale_dim_mask[None, :],
                tile_mask[:, None],
                BLOCK_SIZE,
                "",
            )

        K = K.to(Q.dtype)

        if HAS_ROPE:
            K_rope = kv_load(
                key_cache_ptr,
                base_offset[None, :],
                kv_head_idx,
                offs_rope[:, None],
                seq_offset[None, :],
                stride_k_cache_0,
                stride_k_cache_1,
                stride_k_cache_2,
                stride_k_cache_3,
                rope_dim_mask[:, None],
                tile_mask[None, :],
                BLOCK_SIZE,
                "",
            )
            if SAGE_MXFP4:
                # Do not transpose rhs mxfp4 scale!
                k_rope_scale_loaded = kv_load(
                    k_scale,
                    base_offset[:, None],
                    kv_head_idx,
                    offs_rope_scale[None, :],
                    seq_offset[:, None],
                    stride_k_cache_scale_0,
                    stride_k_cache_scale_1,
                    stride_k_cache_scale_2,
                    stride_k_cache_scale_3,
                    rope_scale_dim_mask[None, :],
                    tile_mask[:, None],
                    BLOCK_SIZE,
                    "",
                )
                S += tl.dot_scaled(
                    Q_rope,
                    q_rope_scale_loaded,
                    "e2m1",
                    K_rope,
                    k_rope_scale_loaded,
                    "e2m1",
                    fast_math=True,
                )
            else:
                S += qk_scale * tl.dot(Q_rope, K_rope)

        if PRELOAD_V:
            # V : (TILE_SIZE, HEAD_SIZE)
            V = kv_load(
                value_cache_ptr,
                base_offset[:, None],
                kv_head_idx,
                offs_dv[None, :],
                seq_offset[:, None],
                stride_v_cache_0,
                stride_v_cache_1,
                stride_v_cache_2,
                stride_v_cache_3,
                dim_mask_v[None, :],
                tile_mask[:, None],
                BLOCK_SIZE,
                KV_cache_modifier,
            )

        S += qk_dot(
            Q,
            K,
            qk_scale,
            q_scale_loaded,
            k_scale_loaded,
            SAGE_MXFP4,
        )

        if USE_SOFTCAP:
            # softcap here uses exp2 and consumes RCP_LN2 conversion.
            # multiply by RCP_LN2 again to be used in later exp2
            S = apply_softcap(S, softcap) * RCP_LN2
        
        if IS_CAUSAL:
            seq_mask = seq_offset[None, :] < context_len + query_pos[:, None] + 1
        else:
            seq_mask = seq_offset[None, :] < seq_len

        S = tl.where(
            query_mask_1[:, None] & query_mask_0[:, None] & seq_mask, S, float("-inf")
        )

        if SLIDING_WINDOW > 0:
            S = tl.where(
                (context_len + query_pos[:, None] - seq_offset) < SLIDING_WINDOW,
                S,
                float("-inf"),
            )

        if USE_ALIBI_SLOPES:
            # prescale w. RCP_LN2 for later exp2
            S += alibi_slope[:, None] * (seq_offset - context_len) * RCP_LN2

        if USE_QQ_BIAS:
            # compute key positions relative to query section
            key_rel_pos = seq_offset - context_len  # shape: [BLOCK_SIZE]
            # load bias only for keys that correspond to queries
            is_query_key = key_rel_pos >= 0 and key_rel_pos < qq_bias_stride_0
            qq_bias = tl.load(
                qq_bias_row_ptrs + key_rel_pos[None, :],
                mask=is_query_key[None, :],  # avoid OOB for context keys
                other=0.0,
            )
            # prescale w. RCP_LN2 for later exp2
            S += qq_bias * RCP_LN2

        # compute running maximum
        # m_j : (BLOCK_M,)
        m_j = tl.maximum(M, tl.max(S, axis=1))

        # For sliding window there's a chance the max is -inf due to masking of
        # the entire row. In this case we need to set m_j 0 to avoid NaN
        m_j = tl.where(m_j > float("-inf"), m_j, 0.0)

        # P : (BLOCK_M, TILE_SIZE)
        P = tl.math.exp2(S - m_j[:, None])

        # l_j : (BLOCK_M,)
        l_j = tl.sum(P, axis=1)

        # alpha : (BLOCK_M, )
        alpha = tl.math.exp2(M - m_j)

        # acc : (BLOCK_M, HEAD_SIZE_PADDED)
        acc = acc * alpha[:, None]

        if not PRELOAD_V:
            # V : (TILE_SIZE, HEAD_SIZE)
            V = kv_load(
                value_cache_ptr,
                base_offset[:, None],
                kv_head_idx,
                offs_dv[None, :],
                seq_offset[:, None],
                stride_v_cache_0,
                stride_v_cache_1,
                stride_v_cache_2,
                stride_v_cache_3,
                dim_mask_v[None, :],
                tile_mask[:, None],
                BLOCK_SIZE,
                KV_cache_modifier,
            )
        # update constants
        L = L * alpha + l_j
        M = m_j

        # acc : (BLOCK_M, HEAD_SIZE_PADDED)
        acc += tl.dot(P.to(V.dtype), V)

    # epilogue
    # This helps the compiler do Newton Raphson on l_i vs on acc which is much larger.
    one_over_L = 1.0 / L[:, None]
    acc = acc * one_over_L

    if SAGE_MXFP4:
        v_descale_ptr = (
            v_scale
            + kv_head_idx * stride_v_cache_scale_0
            + offs_d * stride_v_cache_scale_1
        )
        v_scale_loaded = tl.load(v_descale_ptr, mask=dim_mask, other=0.0)
        acc *= v_scale_loaded[None, :]
    elif v_scale is not None:
        v_scale_loaded = tl.load(v_scale)
        acc *= v_scale_loaded

    if OUTPUT_FP8:
        acc = acc * tl.load(out_scale)
        acc = tl.clamp(acc, FP8_MIN, FP8_MAX)

    output_offset = (
        query_offset_0[:, None] * output_stride_0
        + query_offset_1[:, None] * output_stride_1
        + offs_dv[None, :]
    )

    tl.store(
        output_ptr + output_offset,
        acc,
        mask=dim_mask_v[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
    )


@triton.jit
def kernel_unified_attention_3d(
    segm_output_ptr,  # [num_tokens, num_query_heads, num_segments, head_size]
    segm_max_ptr,  # [num_tokens, num_query_heads, num_segments]
    segm_expsum_ptr,  # [num_tokens, num_query_heads, num_segments]
    query_ptr,  # [num_tokens, num_query_heads, head_size]
    key_cache_ptr,  # [num_blks, blk_size, num_kv_heads, head_size]
    value_cache_ptr,  # [num_blks, blk_size, num_kv_heads, head_size]
    sink_ptr,  # [num_query_heads]
    block_tables_ptr,  # [num_seqs, max_num_blocks_per_seq]
    seq_lens_ptr,  # [num_seqs]
    alibi_slopes_ptr,  # [num_query_heads]
    qq_bias_ptr,  # [num_query_tokens, num_query_tokens]
    scale,  # float32
    q_scale,  # [num_tokens, num_query_heads, head_size // 32] (sage_mxfp4), None (no quantization) or scalar (per tensor fp8)
    k_scale,  # [num_blks, blk_size, num_kv_heads, head_size // 32] (sage_mxfp4), None (no quantization) or scalar (per tensor fp8)
    softcap,  # float32
    num_query_heads: tl.constexpr,  # int
    num_queries_per_kv: tl.constexpr,  # int
    block_table_stride: tl.int64,  # int
    query_stride_0: tl.int64,  # int
    query_stride_1: tl.int64,  # int, should be equal to head_size
    query_scale_stride_0: tl.int64,
    query_scale_stride_1: tl.int64,
    query_scale_stride_2: tl.int64,
    qq_bias_stride_0: tl.int64,  # int
    BLOCK_SIZE: tl.constexpr,  # int
    TILE_SIZE: tl.constexpr,  # int, must be power of 2
    HEAD_SIZE: tl.constexpr,  # int
    HEAD_SIZE_PADDED: tl.constexpr,  # int, must be power of 2
    HEAD_SIZE_V: tl.constexpr,  # int
    HEAD_SIZE_V_PADDED: tl.constexpr,  # int, must be power of 2
    USE_ALIBI_SLOPES: tl.constexpr,  # bool
    USE_QQ_BIAS: tl.constexpr,  # bool
    USE_SOFTCAP: tl.constexpr,  # bool
    USE_SINKS: tl.constexpr,  # bool
    SLIDING_WINDOW: tl.constexpr,  # int
    stride_k_cache_0: tl.int64,  # int
    stride_k_cache_1: tl.int64,  # int
    stride_k_cache_2: tl.int64,  # int
    stride_k_cache_3: tl.constexpr,  # int
    stride_v_cache_0: tl.int64,  # int
    stride_v_cache_1: tl.int64,  # int
    stride_v_cache_2: tl.int64,  # int
    stride_v_cache_3: tl.constexpr,  # int
    stride_k_cache_scale_0: tl.int64,  # int
    stride_k_cache_scale_1: tl.int64,  # int
    stride_k_cache_scale_2: tl.int64,  # int
    stride_k_cache_scale_3: tl.int64,  # int
    query_start_len_ptr,  # [num_seqs+1]
    BLOCK_Q: tl.constexpr,  # int
    num_seqs: tl.int32,
    BLOCK_M: tl.constexpr,  # int
    NUM_SEGMENTS_PER_SEQ: tl.constexpr,  # int
    ALL_DECODE: tl.constexpr = False,  # bool
    SAGE_MXFP4: tl.constexpr = False,  # bool
    IS_PAGING: tl.constexpr = True,  # bool
    PRELOAD_V: tl.constexpr = True,  # bool
    HAS_ROPE: tl.constexpr = False,  # bool
    ROPE_SIZE: tl.constexpr = 0,  # int
    ROPE_SIZE_PADDED: tl.constexpr = 0,  # int, must be power of 2
    IS_CAUSAL: tl.constexpr = True,  # bool
):
    q_block_global_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)
    segm_idx = tl.program_id(2)

    # needed to use exp2 (exp2 -> exp conversion)
    RCP_LN2 = 1.4426950408889634
    if not SAGE_MXFP4:
        _q_ds = tl.load(q_scale) if q_scale is not None else 1.0
        _k_ds = tl.load(k_scale) if k_scale is not None else 1.0
        sm_scale = scale if scale is not None else 1.0
        qk_scale = sm_scale * RCP_LN2 * _q_ds * _k_ds
    else:
        qk_scale = None

    seq_idx = find_seq_idx(
        query_start_len_ptr, q_block_global_idx, num_seqs, BLOCK_Q, True
    )

    q_block_start_idx = tl.load(query_start_len_ptr + seq_idx) // BLOCK_Q + seq_idx

    q_block_local_idx = q_block_global_idx - q_block_start_idx

    cur_batch_in_all_start_index = tl.load(query_start_len_ptr + seq_idx)
    cur_batch_in_all_stop_index = tl.load(query_start_len_ptr + seq_idx + 1)

    cur_batch_query_len = cur_batch_in_all_stop_index - cur_batch_in_all_start_index

    if q_block_local_idx * BLOCK_Q >= cur_batch_query_len:
        return

    # sequence len for this particular sequence
    seq_len = tl.load(seq_lens_ptr + seq_idx)

    # number of segments for this particular sequence
    num_segments = NUM_SEGMENTS_PER_SEQ
    tiles_per_segment = cdiv_fn(seq_len, num_segments * TILE_SIZE)

    if segm_idx * tiles_per_segment * TILE_SIZE >= seq_len:
        return

    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_SIZE_PADDED)
    offs_dv = tl.arange(0, HEAD_SIZE_V_PADDED)

    # MXFP4
    if SAGE_MXFP4:
        offs_d_qk = tl.arange(0, HEAD_SIZE_PADDED // 2)
        offs_d_scale = tl.arange(0, HEAD_SIZE_PADDED // 32)
    else:
        offs_d_qk = tl.arange(0, HEAD_SIZE_PADDED)
        offs_d_scale = None

    if HAS_ROPE:
        if SAGE_MXFP4:
            offs_rope = tl.arange(HEAD_SIZE // 2, (HEAD_SIZE + ROPE_SIZE_PADDED) // 2)
            offs_rope_scale = tl.arange(
                HEAD_SIZE // 32, (HEAD_SIZE + ROPE_SIZE_PADDED) // 32
            )
        else:
            offs_rope = tl.arange(HEAD_SIZE, HEAD_SIZE + ROPE_SIZE_PADDED)
            offs_rope_scale = None

    offs_t = tl.arange(0, TILE_SIZE)
    query_pos = q_block_local_idx * BLOCK_Q + offs_m // num_queries_per_kv

    query_offset_0 = cur_batch_in_all_start_index + query_pos
    query_offset_1 = kv_head_idx * num_queries_per_kv + offs_m % num_queries_per_kv

    if HEAD_SIZE_PADDED != HEAD_SIZE:
        dim_mask = offs_d < HEAD_SIZE
        if SAGE_MXFP4:
            dim_mask_qk = offs_d_qk < (HEAD_SIZE // 2)
            scale_dim_mask = offs_d_scale < (HEAD_SIZE // 32)
        else:
            dim_mask_qk = dim_mask
            scale_dim_mask = dim_mask
    else:
        dim_mask = tl.full((1,), 1, dtype=tl.int1)
        dim_mask_qk = dim_mask
        scale_dim_mask = dim_mask

    if HEAD_SIZE_V_PADDED != HEAD_SIZE_V:
        dim_mask_v = offs_dv < HEAD_SIZE_V
    else:
        dim_mask_v = tl.full((1,), 1, dtype=tl.int1)
    
    
    query_mask_0 = query_pos < cur_batch_query_len
    query_mask_1 = query_offset_1 < num_query_heads

    # Q : (BLOCK_M, HEAD_SIZE_PADDED)
    Q = q_load(
        query_ptr,
        query_offset_0,
        query_offset_1,
        offs_d_qk,
        query_stride_0,
        query_stride_1,
        query_mask_0,
        query_mask_1,
        dim_mask_qk,
        "",
    )

    q_scale_loaded = None
    if SAGE_MXFP4:
        q_scale_loaded = q_load(
            q_scale,
            query_offset_0,
            query_offset_1,
            offs_d_scale,
            query_scale_stride_0,
            query_scale_stride_1,
            query_mask_0,
            query_mask_1,
            scale_dim_mask,
            "",
        )

    if HAS_ROPE:
        if ROPE_SIZE_PADDED != ROPE_SIZE:
            if SAGE_MXFP4:
                rope_dim_mask = offs_rope < ((HEAD_SIZE + ROPE_SIZE) // 2)
                rope_scale_dim_mask = offs_rope_scale < ((HEAD_SIZE + ROPE_SIZE) // 32)
            else:
                rope_dim_mask = offs_rope < (HEAD_SIZE + ROPE_SIZE)
        else:
            rope_dim_mask = tl.full((1,), 1, dtype=tl.int1)
            rope_scale_dim_mask = tl.full((1,), 1, dtype=tl.int1)

        Q_rope = q_load(
            query_ptr,
            query_offset_0,
            query_offset_1,
            offs_rope,
            query_stride_0,
            query_stride_1,
            query_mask_0,
            query_mask_1,
            rope_dim_mask,
            "",
        )

        if SAGE_MXFP4:
            q_rope_scale_loaded = q_load(
                q_scale,
                query_offset_0,
                query_offset_1,
                offs_rope_scale,
                query_scale_stride_0,
                query_scale_stride_1,
                query_mask_0,
                query_mask_1,
                rope_scale_dim_mask,
                "",
            )

    block_table_offset = seq_idx * block_table_stride

    if USE_SINKS:
        if segm_idx == 0:
            # Prescale with RCP_LN2, needed for exp2
            M = (
                tl.load(
                    sink_ptr + query_offset_1,
                    mask=query_mask_1,
                    other=float("-inf"),
                ).to(dtype=tl.float32)
                * RCP_LN2
            )
        else:
            M = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    else:
        M = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)

    L = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_SIZE_PADDED], dtype=tl.float32)

    # context length for this particular sequences
    context_len = seq_len - cur_batch_query_len

    # alibi slope for this head
    if USE_ALIBI_SLOPES:
        alibi_slope = tl.load(
            alibi_slopes_ptr + query_offset_1, mask=query_mask_1, other=0.0
        )

    # query-query attention bias
    if USE_QQ_BIAS:
        qq_bias_row_ptrs = (
            qq_bias_ptr + query_pos[:, None] * qq_bias_stride_0
        )  # shape: [BLOCK_M]

    if IS_CAUSAL:
        # compute the length of the longest sequence prefix spanned by any
        # query token in the current q_block (q_block_local_idx)
        max_seq_prefix_len = (
            context_len
            + q_block_local_idx * BLOCK_Q
            + (BLOCK_M - 1) // num_queries_per_kv
            + 1
        )

        # adjust for potential padding in the last q_block by considering the
        # actual sequence length
        max_seq_prefix_len = tl.minimum(max_seq_prefix_len, seq_len)
    else:
        max_seq_prefix_len = seq_len

    # calculate the number of tiles that need to be processed to
    # cover the longest sequence prefix (due to causal masking, tiles beyond
    # this prefix can be skipped)
    num_tiles = cdiv_fn(max_seq_prefix_len, TILE_SIZE)

    KV_cache_modifier: tl.constexpr = ".cg" if ALL_DECODE else ""
    if not IS_PAGING:
        seq_offset = segm_idx * tiles_per_segment * TILE_SIZE + offs_t
        physical_block_idx = tl.load(
            block_tables_ptr + block_table_offset + seq_offset // BLOCK_SIZE
        ).to(tl.int64)

    # iterate through tiles within current segment
    for j in range(
        segm_idx * tiles_per_segment,
        min((segm_idx + 1) * tiles_per_segment, num_tiles),
    ):
        seq_offset = j * TILE_SIZE + offs_t
        if TILE_SIZE == BLOCK_SIZE:
            tile_mask = tl.full((1,), 1, dtype=tl.int1)
        else:
            tile_mask = seq_offset < max_seq_prefix_len

        if IS_PAGING:
            physical_block_idx = tl.load(
                block_tables_ptr + block_table_offset + seq_offset // BLOCK_SIZE
            ).to(tl.int64)

        S = tl.zeros([BLOCK_M, TILE_SIZE], dtype=tl.float32)

        # K : (HEAD_SIZE, TILE_SIZE)
        K = kv_load(
            key_cache_ptr,
            physical_block_idx[None, :],
            kv_head_idx,
            offs_d_qk[:, None],
            seq_offset[None, :],
            stride_k_cache_0,
            stride_k_cache_1,
            stride_k_cache_2,
            stride_k_cache_3,
            dim_mask_qk[:, None],
            tile_mask[None, :],
            BLOCK_SIZE,
            KV_cache_modifier,
        )

        k_scale_loaded = None
        if SAGE_MXFP4:
            k_scale_loaded = kv_load(
                k_scale,
                physical_block_idx[:, None],
                kv_head_idx,
                offs_d_scale[None, :],
                seq_offset[:, None],
                stride_k_cache_scale_0,
                stride_k_cache_scale_1,
                stride_k_cache_scale_2,
                stride_k_cache_scale_3,
                scale_dim_mask[None, :],
                tile_mask[:, None],
                BLOCK_SIZE,
                "",
            )

        K = K.to(Q.dtype)

        if HAS_ROPE:
            K_rope = kv_load(
                key_cache_ptr,
                physical_block_idx[None, :],
                kv_head_idx,
                offs_rope[:, None],
                seq_offset[None, :],
                stride_k_cache_0,
                stride_k_cache_1,
                stride_k_cache_2,
                stride_k_cache_3,
                rope_dim_mask[:, None],
                tile_mask[None, :],
                BLOCK_SIZE,
                "",
            )
            if SAGE_MXFP4:
                # Do not transpose rhs mxfp4 scale!
                k_rope_scale_loaded = kv_load(
                    k_scale,
                    physical_block_idx[:, None],
                    kv_head_idx,
                    offs_rope_scale[None, :],
                    seq_offset[:, None],
                    stride_k_cache_scale_0,
                    stride_k_cache_scale_1,
                    stride_k_cache_scale_2,
                    stride_k_cache_scale_3,
                    rope_scale_dim_mask[None, :],
                    tile_mask[:, None],
                    BLOCK_SIZE,
                    "",
                )
                S += tl.dot_scaled(
                    Q_rope,
                    q_rope_scale_loaded,
                    "e2m1",
                    K_rope,
                    k_rope_scale_loaded,
                    "e2m1",
                    fast_math=True,
                )
            else:
                S += qk_scale * tl.dot(Q_rope, K_rope)

        if PRELOAD_V:
            # V : (TILE_SIZE, HEAD_SIZE)
            V = kv_load(
                value_cache_ptr,
                physical_block_idx[:, None],
                kv_head_idx,
                offs_dv[None, :],
                seq_offset[:, None],
                stride_v_cache_0,
                stride_v_cache_1,
                stride_v_cache_2,
                stride_v_cache_3,
                dim_mask_v[None, :],
                tile_mask[:, None],
                BLOCK_SIZE,
                KV_cache_modifier,
            )

        S += qk_dot(
            Q,
            K,
            qk_scale,
            q_scale_loaded,
            k_scale_loaded,
            SAGE_MXFP4,
        )

        if USE_SOFTCAP:
            # softcap here uses exp2 and consumes RCP_LN2 conversion.
            # multiply by RCP_LN2 again to be used in later exp2
            S = apply_softcap(S, softcap) * RCP_LN2
        
        if IS_CAUSAL:
            seq_mask = seq_offset[None, :] < context_len + query_pos[:, None] + 1
        else:
            seq_mask = seq_offset[None, :] < seq_len

        S = tl.where(
            query_mask_1[:, None] & query_mask_0[:, None] & seq_mask, S, float("-inf")
        )

        if SLIDING_WINDOW > 0:
            S = tl.where(
                (context_len + query_pos[:, None] - seq_offset) < SLIDING_WINDOW,
                S,
                float("-inf"),
            )

        if USE_ALIBI_SLOPES:
            # prescale w. RCP_LN2 for later exp2
            S += alibi_slope[:, None] * (seq_offset - context_len) * RCP_LN2

        if USE_QQ_BIAS:
            # compute key positions relative to query section
            key_rel_pos = seq_offset - context_len  # shape: [BLOCK_SIZE]
            # load bias only for keys that correspond to queries
            is_query_key = key_rel_pos >= 0 and key_rel_pos < qq_bias_stride_0
            qq_bias = tl.load(
                qq_bias_row_ptrs + key_rel_pos[None, :],
                mask=is_query_key[None, :],  # avoid OOB for context keys
                other=0.0,
            )
            # prescale w. RCP_LN2 for later exp2
            S += qq_bias * RCP_LN2

        # compute running maximum
        # m_j : (BLOCK_M,)
        m_j = tl.maximum(M, tl.max(S, axis=1))

        # For sliding window there's a chance the max is -inf due to masking of
        # the entire row. In this case we need to set m_j 0 to avoid NaN
        m_j = tl.where(m_j > float("-inf"), m_j, 0.0)

        # P : (BLOCK_M, TILE_SIZE,)
        P = tl.math.exp2(S - m_j[:, None])

        # l_j : (BLOCK_M,)
        l_j = tl.sum(P, axis=1)

        # alpha : (BLOCK_M, )
        alpha = tl.math.exp2(M - m_j)

        # acc : (BLOCK_M, HEAD_SIZE_PADDED)
        acc = acc * alpha[:, None]

        if not PRELOAD_V:
            # V : (TILE_SIZE, HEAD_SIZE)
            V = kv_load(
                value_cache_ptr,
                physical_block_idx[:, None],
                kv_head_idx,
                offs_dv[None, :],
                seq_offset[:, None],
                stride_v_cache_0,
                stride_v_cache_1,
                stride_v_cache_2,
                stride_v_cache_3,
                dim_mask_v[None, :],
                tile_mask[:, None],
                BLOCK_SIZE,
                KV_cache_modifier,
            )
        # update constants
        L = L * alpha + l_j
        M = m_j

        # acc : (BLOCK_M, HEAD_SIZE_PADDED)
        acc += tl.dot(P.to(V.dtype), V)

    segm_output_offset = (
        query_offset_0[:, None].to(tl.int64)
        * (num_query_heads * NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_V_PADDED)
        + query_offset_1[:, None] * (NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_V_PADDED)
        + segm_idx * HEAD_SIZE_V_PADDED
        + tl.arange(0, HEAD_SIZE_V_PADDED)[None, :]
    )
    tl.store(
        segm_output_ptr + segm_output_offset,
        acc,
        mask=dim_mask_v[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
    )
    segm_offset = (
        query_offset_0.to(tl.int64) * (num_query_heads * NUM_SEGMENTS_PER_SEQ)
        + query_offset_1 * NUM_SEGMENTS_PER_SEQ
        + segm_idx
    )
    tl.store(segm_max_ptr + segm_offset, M, mask=query_mask_0 & query_mask_1)
    tl.store(segm_expsum_ptr + segm_offset, L, mask=query_mask_0 & query_mask_1)


@triton.jit
def reduce_segments(
    output_ptr,  # [num_tokens, num_query_heads, head_size]
    v_scale,  # [num_kv_heads, head_size] if sage or sage_mxfp4 else non or scalar
    segm_output_ptr,
    # [num_tokens, num_query_heads, max_num_segments, head_size]
    segm_max_ptr,  # [num_tokens, num_query_heads, max_num_segments]
    segm_expsum_ptr,  # [num_tokens, num_query_heads, max_num_segments]
    seq_lens_ptr,  # [num_seqs]
    num_seqs,  # int
    num_query_heads: tl.constexpr,  # int
    num_queries_per_kv: tl.constexpr,  # int
    out_scale_inv,  # float32
    output_stride_0: tl.int64,  # int
    output_stride_1: tl.int64,  # int, should be equal to head_size
    stride_v_cache_scale_0: tl.int64,  # int
    stride_v_cache_scale_1: tl.int64,  # int
    block_table_stride: tl.int64,  # int
    TILE_SIZE: tl.constexpr,  # int
    HEAD_SIZE: tl.constexpr,  # int, must be power of 2
    HEAD_SIZE_PADDED: tl.constexpr,  # int, must be power of 2
    query_start_len_ptr,  # [num_seqs+1]
    BLOCK_Q: tl.constexpr,  # int
    NUM_SEGMENTS_PER_SEQ: tl.constexpr,  # int
    OUTPUT_FP8: tl.constexpr,  # bool
    SAGE_MXFP4: tl.constexpr = False,  # bool
    FP8_MIN: tl.constexpr = float8_info.min,
    FP8_MAX: tl.constexpr = float8_info.max,
):
    query_token_idx = tl.program_id(0)
    query_head_idx = tl.program_id(1)
    offs_d = tl.arange(0, HEAD_SIZE_PADDED)

    seq_idx = find_seq_idx(
        query_start_len_ptr, query_token_idx, num_seqs, BLOCK_Q, False
    )

    # sequence len for this particular sequence
    seq_len = tl.load(seq_lens_ptr + seq_idx)

    # number of segments for this particular sequence
    num_segments = NUM_SEGMENTS_PER_SEQ
    tiles_per_segment = cdiv_fn(seq_len, num_segments * TILE_SIZE)

    # create masks for subsequent loads
    act_num_segments = cdiv_fn(seq_len, tiles_per_segment * TILE_SIZE)
    segm_mask = tl.arange(0, NUM_SEGMENTS_PER_SEQ) < tl.full(
        [NUM_SEGMENTS_PER_SEQ], act_num_segments, dtype=tl.int32
    )

    if HEAD_SIZE_PADDED != HEAD_SIZE:
        dim_mask = offs_d < HEAD_SIZE
    else:
        dim_mask = tl.full((1,), 1, dtype=tl.int1)

    # load segment maxima
    segm_offset = (
        query_token_idx.to(tl.int64) * (num_query_heads * NUM_SEGMENTS_PER_SEQ)
        + query_head_idx * NUM_SEGMENTS_PER_SEQ
        + tl.arange(0, NUM_SEGMENTS_PER_SEQ)
    )
    segm_max = tl.load(segm_max_ptr + segm_offset, mask=segm_mask, other=float("-inf"))
    overall_max = tl.max(segm_max)

    # load and rescale segment exp sums
    segm_expsum = tl.load(segm_expsum_ptr + segm_offset, mask=segm_mask, other=0.0)
    segm_expsum = segm_expsum * tl.math.exp2(segm_max - overall_max)
    overall_expsum = tl.sum(segm_expsum)

    # load, rescale, and add segment attention outputs
    segm_output_offset = (
        query_token_idx.to(tl.int64)
        * (num_query_heads * NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED)
        + query_head_idx * (NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED)
        + tl.arange(0, NUM_SEGMENTS_PER_SEQ)[:, None] * HEAD_SIZE_PADDED
        + tl.arange(0, HEAD_SIZE_PADDED)[None, :]
    )
    segm_output = tl.load(
        segm_output_ptr + segm_output_offset,
        mask=segm_mask[:, None] & dim_mask[None, :],
        other=0.0,
    )
    segm_output *= tl.math.exp2(segm_max - overall_max)[:, None]
    acc_sum = tl.sum(segm_output, axis=0)
    # safely divide by overall_expsum, returning 0.0 if overall_expsum is 0
    acc = tl.where(overall_expsum == 0.0, 0.0, acc_sum / overall_expsum)

    if SAGE_MXFP4:
        v_descale_ptr = (
            v_scale
            + query_head_idx // num_queries_per_kv * stride_v_cache_scale_0
            + offs_d * stride_v_cache_scale_1
        )
        v_scale_loaded = tl.load(v_descale_ptr, mask=dim_mask, other=0.0)
        acc *= v_scale_loaded
    elif v_scale is not None:
        v_scale_loaded = tl.load(v_scale)
        acc *= v_scale_loaded

    if OUTPUT_FP8:
        acc = acc * tl.load(out_scale_inv)
        acc = tl.clamp(acc, FP8_MIN, FP8_MAX)

    # write result
    output_offset = (
        query_token_idx * output_stride_0
        + query_head_idx * output_stride_1
        + tl.arange(0, HEAD_SIZE_PADDED)
    )
    tl.store(output_ptr + output_offset, acc, mask=dim_mask)
