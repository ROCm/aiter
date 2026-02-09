# The kernels in this file are adapted from vLLM:
# https://github.com/vllm-project/vllm/blob/main/vllm/attention/ops/triton_unified_attention.py
import triton
import triton.language as tl
import torch
from aiter.ops.triton.utils.types import e4m3_dtype
from triton.experimental import gluon
import triton.experimental.gluon.language as ttgl

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


@gluon.jit
def _find_seq_idx(
    query_start_len_ptr,
    target_idx,
    num_seqs,
    BLOCK_Q: ttgl.constexpr,
    use_q_block_mode: ttgl.constexpr,
):
    left: ttgl.int32 = 0
    right = num_seqs
    while left < right:
        mid = (left + right) // 2
        val = ttgl.load(query_start_len_ptr + mid)
        mid_val = val // BLOCK_Q + mid if use_q_block_mode else val

        if mid_val <= target_idx:
            left = mid + 1
        else:
            right = mid

    return left - 1


@gluon.jit
def _get_q_metadata(
    query_start_len_ptr,
    seq_idx,
    q_block_global_idx,
    BLOCK_Q: ttgl.constexpr,
):
    q_block_start_idx = ttgl.load(query_start_len_ptr + seq_idx) // BLOCK_Q + seq_idx

    q_block_local_idx = q_block_global_idx - q_block_start_idx

    cur_batch_in_all_start_index = ttgl.load(query_start_len_ptr + seq_idx)
    cur_batch_in_all_stop_index = ttgl.load(query_start_len_ptr + seq_idx + 1)

    cur_batch_query_len = cur_batch_in_all_stop_index - cur_batch_in_all_start_index

    return q_block_local_idx, cur_batch_query_len, cur_batch_in_all_start_index


@gluon.jit
def _get_seq_metadata(
    seq_lens_ptr,
    seq_idx,
    TILE_SIZE: ttgl.constexpr,
    NUM_SEGMENTS_PER_SEQ: ttgl.constexpr,
):
    # sequence len for this particular sequence
    seq_len = ttgl.load(seq_lens_ptr + seq_idx)

    # number of segments for this particular sequence
    num_segments = NUM_SEGMENTS_PER_SEQ
    tiles_per_segment = cdiv_fn(seq_len, num_segments * TILE_SIZE)

    return seq_len, tiles_per_segment


@gluon.jit
def _allocate_L_M_acc(
    sink_ptr,
    segm_idx,
    query_offset_1,
    query_mask_1,
    RCP_LN2,
    BLOCK_M: ttgl.constexpr,
    Q_BLOCKED_LAYOUT: ttgl.constexpr,
    HEAD_SIZE_PADDED: ttgl.constexpr,
    PV_WMMA_LAYOUT: ttgl.constexpr,
    USE_SINKS: ttgl.constexpr,
):

    # M : shape = (BLOCK_M, ), layout = ttgl.SliceLayout(1, Q_BLOCKED_LAYOUT)
    if USE_SINKS:
        if segm_idx == 0:
            # Prescale with RCP_LN2, needed for exp2
            M = (
                ttgl.amd.cdna4.buffer_load(
                    ptr=sink_ptr,
                    offsets=query_offset_1.to(ttgl.int32),
                    mask=query_mask_1,
                    other=float("-inf"),
                ).to(dtype=ttgl.float32)
                * RCP_LN2
            )
        else:
            M = ttgl.full(
                [BLOCK_M],
                float("-inf"),
                dtype=tl.float32,
                layout=ttgl.SliceLayout(1, Q_BLOCKED_LAYOUT),
            )
    else:
        M = ttgl.full(
            [BLOCK_M],
            float("-inf"),
            dtype=tl.float32,
            layout=ttgl.SliceLayout(1, Q_BLOCKED_LAYOUT),
        )

    # L : shape = (BLOCK_M, ), layout = ttgl.SliceLayout(1, Q_BLOCKED_LAYOUT)
    L = ttgl.full(
        [BLOCK_M], 1.0, dtype=tl.float32, layout=ttgl.SliceLayout(1, Q_BLOCKED_LAYOUT)
    )
    # acc : shape = (BLOCK_M, HEAD_SIZE_PADDED), layout = PV_WMMA_LAYOUT
    acc = ttgl.zeros(
        [BLOCK_M, HEAD_SIZE_PADDED], dtype=tl.float32, layout=PV_WMMA_LAYOUT
    )

    return L, M, acc


@gluon.jit
def _perform_QK_wmma_and_update_L_M(
    Q,
    K,
    L,
    M,
    acc,
    qq_bias_row_ptrs,
    seq_offset,
    query_mask_1,
    query_mask_0,
    context_len,
    query_pos,
    alibi_slope,
    qq_bias_stride_0,
    qk_scale,
    softcap,
    RCP_LN2,
    BLOCK_M: ttgl.constexpr,
    TILE_SIZE: ttgl.constexpr,
    USE_SOFTCAP: ttgl.constexpr,
    SLIDING_WINDOW: ttgl.constexpr,
    USE_ALIBI_SLOPES: ttgl.constexpr,
    USE_QQ_BIAS: ttgl.constexpr,
    Q_BLOCKED_LAYOUT: ttgl.constexpr,
    QK_WMMA_LAYOUT: ttgl.constexpr,
    PV_WMMA_LAYOUT: ttgl.constexpr,
):
    # S : shape = (BLOCK_M, TILE_SIZE), layout = QK_WMMA_LAYOUT
    S = ttgl.zeros([BLOCK_M, TILE_SIZE], dtype=tl.float32, layout=QK_WMMA_LAYOUT)
    # qk_scale = scale * RCP_LN2 (log_2 e) so that we can use exp2 later
    S = qk_scale * ttgl.amd.gfx1250.wmma(Q, K, S)
    # S : shape = (BLOCK_M, TILE_SIZE), layout = Q_BLOCKED_LAYOUT
    S = ttgl.convert_layout(S, layout=Q_BLOCKED_LAYOUT)

    if USE_SOFTCAP:
        # softcap here uses exp2 and consumes RCP_LN2 conversion.
        # multiply by RCP_LN2 again to be used in later exp2
        S = apply_softcap(S, softcap) * RCP_LN2

    seq_mask = seq_offset[None, :] < context_len + query_pos[:, None] + 1

    S = ttgl.where(
        query_mask_1[:, None] & query_mask_0[:, None] & seq_mask, S, float("-inf")
    )

    if SLIDING_WINDOW > 0:
        S = ttgl.where(
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
        qq_bias = ttgl.load(
            qq_bias_row_ptrs + key_rel_pos[None, :],
            mask=is_query_key[None, :],  # avoid OOB for context keys
            other=0.0,
        )
        # prescale w. RCP_LN2 for later exp2
        S += qq_bias * RCP_LN2

    # compute running maximum
    # m_j : shape = (BLOCK_M, ), layout = ttgl.SliceLayout(1, Q_BLOCKED_LAYOUT)
    m_j = ttgl.maximum(M, ttgl.max(S, axis=1))

    # For sliding window there's a chance the max is -inf due to masking of
    # the entire row. In this case we need to set m_j 0 to avoid NaN
    m_j = ttgl.where(m_j > float("-inf"), m_j, 0.0)

    # P : shape = (BLOCK_M, TILE_SIZE), layout = Q_BLOCKED_LAYOUT
    P = ttgl.exp2(S - m_j[:, None])

    # l_j : shape = (BLOCK_M, ), layout = ttgl.SliceLayout(1, Q_BLOCKED_LAYOUT)
    l_j = ttgl.sum(P, axis=1)

    # alpha : shape = (BLOCK_M, ), layout = ttgl.SliceLayout(1, Q_BLOCKED_LAYOUT)
    alpha = ttgl.exp2(M - m_j)

    # acc : shape = (BLOCK_M, HEAD_SIZE_PADDED), layout = PV_WMMA_LAYOUT
    acc = acc * ttgl.convert_layout(alpha[:, None], layout=PV_WMMA_LAYOUT)

    # update constants
    # L : shape = (BLOCK_M, ), layout = ttgl.SliceLayout(1, Q_BLOCKED_LAYOUT)
    # M : shape = (BLOCK_M, ), layout = ttgl.SliceLayout(1, Q_BLOCKED_LAYOUT)
    L = L * alpha + l_j
    M = m_j

    return P, L, M, acc


@gluon.jit
def _perform_PV_wmma(
    P,
    V,
    acc,
    P_DOT_LAYOUT: ttgl.constexpr,
):
    P = P.to(V.dtype)
    P = ttgl.convert_layout(P, layout=P_DOT_LAYOUT)
    # P : shape = (BLOCK_M, TILE_SIZE), layout = P_DOT_LAYOUT
    # V : shape = (TILE_SIZE, HEAD_SIZE), layout = V_DOT_LAYOUT
    # acc : shape = (BLOCK_M, HEAD_SIZE_PADDED), layout = PV_WMMA_LAYOUT
    acc = ttgl.amd.gfx1250.wmma(P, V, acc)
    return acc


@gluon.jit
def _tdm_get_kv_offsets(
    j,
    kv_head_idx,
    block_tables_ptr,
    block_table_offset,
    stride_k_cache_0: ttgl.int64,
    stride_k_cache_2: ttgl.int64,
    stride_v_cache_0: ttgl.int64,
    stride_v_cache_2: ttgl.int64,
):
    physical_block_idx = ttgl.load(block_tables_ptr + block_table_offset + j)

    offs_k_t_starts = (
        physical_block_idx * stride_k_cache_0 + kv_head_idx * stride_k_cache_2
    ).to(tl.int32)
    offs_v_t_starts = (
        physical_block_idx * stride_v_cache_0 + kv_head_idx * stride_v_cache_2
    ).to(tl.int32)

    return j + 1, offs_k_t_starts, offs_v_t_starts


@gluon.jit
def _tdm_async_load_to_lds(
    j,
    src,
    offsets,
    dest,
    pred_i32,
    num_stages: ttgl.constexpr,
):
    # ttgl.amd.gfx1250.tdm.prefetch(
    #     src=k_desc,
    #     offsets=[
    #         0,
    #         offs_kv_t_starts,
    #     ],
    #     pred=pred.to(ttgl.int1)
    # )
    ttgl.amd.gfx1250.tdm.async_load(
        src=src,
        offsets=offsets,
        dest=dest.index(j % num_stages),
        pred=pred_i32,
    )

    return j + 1


@gluon.jit
def _tdm_request_from_lds(
    j,
    kv_scale,
    Q_dtype,
    smem,
    asycn_wait: ttgl.constexpr,
    layout: ttgl.constexpr,
    transpose: ttgl.constexpr,
    num_ctas: ttgl.constexpr,
    num_stages: ttgl.constexpr,
):
    if num_ctas > 1:
        ttgl.amd.gfx1250.cluster.arrive()
    ttgl.amd.gfx1250.tdm.async_wait(asycn_wait)
    if num_ctas > 1:
        ttgl.amd.gfx1250.cluster.wait()
    if transpose:
        X = smem.index(j % num_stages).permute([1, 0]).load(layout=layout)
    else:
        X = smem.index(j % num_stages).load(layout=layout)

    if X.dtype.is_fp8() and not Q_dtype.is_fp8():
        X = (X.to(ttgl.float32) * ttgl.load(kv_scale)).to(Q_dtype)

    return j + 1, X


@gluon.jit
def _tdm_create_tensor_descriptors_and_allocate_lds(
    q_ptr,
    k_ptr,
    v_ptr,
    stride_q_m: ttgl.int64,  # int
    stride_q_d: ttgl.constexpr,  # int
    stride_k_t: ttgl.int64,  # int
    stride_k_d: ttgl.constexpr,  # int
    stride_v_t: ttgl.int64,  # int
    stride_v_d: ttgl.constexpr,  # int
    q_shared_layout: ttgl.constexpr,
    k_shared_layout: ttgl.constexpr,
    v_shared_layout: ttgl.constexpr,
    M: ttgl.constexpr,
    T: ttgl.constexpr,
    BLOCK_M: ttgl.constexpr,
    HEAD_SIZE: ttgl.constexpr,
    TILE_SIZE: ttgl.constexpr,
    HEAD_SIZE_PADDED: ttgl.constexpr,
    num_stages: ttgl.constexpr,
):
    ttgl.static_assert(stride_q_d == 1, "stride_q_d must be 1")
    ttgl.static_assert(stride_k_d == 1, "stride_k_d must be 1")
    ttgl.static_assert(stride_v_d == 1, "stride_v_d must be 1")
    # q_desc = ttgl.amd.gfx1250.tdm.make_tensor_descriptor(
    #     base=q_ptr,
    #     shape=(M, HEAD_SIZE),
    #     strides=(stride_q_m, stride_q_d),
    #     block_shape=(BLOCK_M, HEAD_SIZE_PADDED),
    #     layout=q_shared_layout,
    # )

    k_desc = ttgl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=k_ptr,
        shape=(TILE_SIZE, T * HEAD_SIZE),
        strides=(stride_k_t, stride_k_d),
        block_shape=(TILE_SIZE, HEAD_SIZE_PADDED),
        layout=k_shared_layout,
    )

    v_desc = ttgl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=v_ptr,
        shape=(TILE_SIZE, T * HEAD_SIZE),
        strides=(stride_v_t, stride_v_d),
        block_shape=(TILE_SIZE, HEAD_SIZE_PADDED),
        layout=v_shared_layout,
    )

    # smem_Q = ttgl.allocate_shared_memory(
    #     q_desc.dtype,
    #     shape=q_desc.block_shape,
    #     layout=q_shared_layout,
    # )
    smem_Q = ttgl.allocate_shared_memory(
        q_ptr.type.element_ty,
        shape=[BLOCK_M, HEAD_SIZE_PADDED],
        layout=q_shared_layout,
    )
    smem_K = ttgl.allocate_shared_memory(
        k_desc.dtype,
        shape=[num_stages] + k_desc.block_shape,
        layout=k_desc.layout,
    )
    smem_V = ttgl.allocate_shared_memory(
        v_desc.dtype,
        shape=[num_stages] + v_desc.block_shape,
        layout=v_desc.layout,
    )

    return k_desc, v_desc, smem_Q, smem_K, smem_V


# from triton._C.libtriton.gluon_ir import make_cga_layout
@gluon.jit
def gluon_kernel_unified_attention_3d_tdm_pipelined(
    segm_output_ptr,
    # [num_tokens, num_query_heads, num_segments, head_size]
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
    k_scale,  # float32
    v_scale,  # float32
    softcap,  # float32
    num_tokens,  # int
    num_query_heads: ttgl.constexpr,  # int
    num_queries_per_kv: ttgl.constexpr,  # int
    block_table_stride: ttgl.int64,  # int
    query_stride_0: ttgl.int64,  # int
    query_stride_1: ttgl.int64,  # int, should be equal to head_size
    qq_bias_stride_0: ttgl.int64,  # int
    NUM_BLOCKS: ttgl.constexpr,  # int
    BLOCK_SIZE: ttgl.constexpr,  # int
    TILE_SIZE: ttgl.constexpr,  # int, must be power of 2
    HEAD_SIZE: ttgl.constexpr,  # int
    HEAD_SIZE_PADDED: ttgl.constexpr,  # int, must be power of 2
    USE_ALIBI_SLOPES: ttgl.constexpr,  # bool
    USE_QQ_BIAS: ttgl.constexpr,  # bool
    USE_SOFTCAP: ttgl.constexpr,  # bool
    USE_SINKS: ttgl.constexpr,  # bool
    SLIDING_WINDOW: ttgl.constexpr,  # int
    stride_k_cache_0: ttgl.int64,  # int
    stride_k_cache_1: ttgl.int64,  # int
    stride_k_cache_2: ttgl.int64,  # int
    stride_k_cache_3: ttgl.constexpr,  # int
    stride_v_cache_0: ttgl.int64,  # int
    stride_v_cache_1: ttgl.int64,  # int
    stride_v_cache_2: ttgl.int64,  # int
    stride_v_cache_3: ttgl.constexpr,  # int
    query_start_len_ptr,  # [num_seqs+1]
    BLOCK_Q: ttgl.constexpr,  # int
    num_seqs: ttgl.int32,
    BLOCK_M: ttgl.constexpr,  # int
    NUM_SEGMENTS_PER_SEQ: ttgl.constexpr,  # int
    num_warps: ttgl.constexpr,  # int
    num_stages: ttgl.constexpr,  # int
    QK_WMMA_LAYOUT: ttgl.constexpr,
    PV_WMMA_LAYOUT: ttgl.constexpr,
    Q_DOT_LAYOUT: ttgl.constexpr,
    K_DOT_LAYOUT: ttgl.constexpr,
    P_DOT_LAYOUT: ttgl.constexpr,
    V_DOT_LAYOUT: ttgl.constexpr,
    Q_SHARED_LAYOUT: ttgl.constexpr,
    K_SHARED_LAYOUT: ttgl.constexpr,
    V_SHARED_LAYOUT: ttgl.constexpr,
    Q_BLOCKED_LAYOUT: ttgl.constexpr,
    K_BLOCKED_LAYOUT: ttgl.constexpr,
    num_ctas: ttgl.constexpr = 1,  # int
    ALL_DECODE: ttgl.constexpr = False,  # bool
):
    q_block_global_idx = ttgl.program_id(0)
    kv_head_idx = ttgl.program_id(1)
    segm_idx = ttgl.program_id(2)
    # num_ctas: ttgl.constexpr = ttgl.num_ctas()
    pred = 1
    pred_i32 = pred.to(ttgl.int32) if hasattr(pred, "to") else pred

    assert TILE_SIZE == BLOCK_SIZE, "TILE_SIZE must be identical to BLOCK_SIZE"

    # needed to use exp2 (exp2 -> exp conversion)
    RCP_LN2 = 1.4426950408889634
    qk_scale = scale * RCP_LN2

    seq_idx = _find_seq_idx(
        query_start_len_ptr, q_block_global_idx, num_seqs, BLOCK_Q, True
    )

    q_block_local_idx, cur_batch_query_len, cur_batch_in_all_start_index = (
        _get_q_metadata(
            query_start_len_ptr,
            seq_idx,
            q_block_global_idx,
            BLOCK_Q,
        )
    )

    if q_block_local_idx * BLOCK_Q >= cur_batch_query_len:
        return

    seq_len, tiles_per_segment = _get_seq_metadata(
        seq_lens_ptr,
        seq_idx,
        TILE_SIZE,
        NUM_SEGMENTS_PER_SEQ,
    )

    if segm_idx * tiles_per_segment * TILE_SIZE >= seq_len:
        return

    # block table offset for this particular sequence
    block_table_offset = seq_idx * block_table_stride

    # context length for this particular sequence
    context_len = seq_len - cur_batch_query_len

    offs_q_m = ttgl.arange(0, BLOCK_M, layout=ttgl.SliceLayout(1, Q_BLOCKED_LAYOUT))
    offs_q_d = ttgl.arange(
        0, HEAD_SIZE_PADDED, layout=ttgl.SliceLayout(0, Q_BLOCKED_LAYOUT)
    )

    query_pos = q_block_local_idx * BLOCK_Q + offs_q_m // num_queries_per_kv

    query_offset_0 = cur_batch_in_all_start_index + query_pos
    query_offset_1 = kv_head_idx * num_queries_per_kv + offs_q_m % num_queries_per_kv
    query_offset = (
        query_offset_0[:, None] * query_stride_0
        + query_offset_1[:, None] * query_stride_1
        + offs_q_d[None, :]
    )

    if HEAD_SIZE_PADDED != HEAD_SIZE:
        dim_mask = offs_q_d < HEAD_SIZE
    else:
        dim_mask = ttgl.full((1,), 1, dtype=tl.int1)

    query_mask_0 = query_pos < cur_batch_query_len
    query_mask_1 = query_offset_1 < num_query_heads

    total_num_q_tokens = num_tokens * num_query_heads
    total_num_kv_tokens = (
        NUM_BLOCKS * BLOCK_SIZE * (num_query_heads // num_queries_per_kv)
    )

    k_desc, v_desc, smem_Q, smem_K, smem_V = (
        _tdm_create_tensor_descriptors_and_allocate_lds(
            query_ptr,
            key_cache_ptr,
            value_cache_ptr,
            query_stride_1,
            1,
            stride_k_cache_1,  # stride_k_cache_1 = HEAD_SIZE * num_kv_heads
            stride_k_cache_3,
            stride_v_cache_1,
            stride_v_cache_3,
            Q_SHARED_LAYOUT,
            K_SHARED_LAYOUT,
            V_SHARED_LAYOUT,
            total_num_q_tokens,
            total_num_kv_tokens,
            BLOCK_M,
            HEAD_SIZE,
            TILE_SIZE,
            HEAD_SIZE_PADDED,
            num_stages=num_stages,
        )
    )

    # Q_load : shape = (BLOCK_M, HEAD_SIZE_PADDED), layout = Q_BLOCKED_LAYOUT
    Q_load = ttgl.amd.cdna4.buffer_load(
        ptr=query_ptr,
        offsets=query_offset.to(ttgl.int32),
        mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
        other=0.0,
    )
    smem_Q.store(Q_load)
    # Q : shape = (BLOCK_M, HEAD_SIZE_PADDED), layout = Q_DOT_LAYOUT
    Q = smem_Q.load(layout=Q_DOT_LAYOUT)

    L, M, acc = _allocate_L_M_acc(
        sink_ptr,
        segm_idx,
        query_offset_1,
        query_mask_1,
        RCP_LN2,
        BLOCK_M,
        Q_BLOCKED_LAYOUT,
        HEAD_SIZE_PADDED,
        PV_WMMA_LAYOUT,
        USE_SINKS,
    )

    # alibi slope for this head
    alibi_slope = None
    if USE_ALIBI_SLOPES:
        alibi_slope = tl.load(
            alibi_slopes_ptr + query_offset_1, mask=query_mask_1, other=0.0
        )

    # query-query attention bias
    qq_bias_row_ptrs = None
    if USE_QQ_BIAS:
        qq_bias_row_ptrs = (
            qq_bias_ptr + query_pos[:, None] * qq_bias_stride_0
        )  # shape: [BLOCK_M]

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
    max_seq_prefix_len = ttgl.minimum(max_seq_prefix_len, seq_len)

    # calculate the number of tiles that need to be processed to
    # cover the longest sequence prefix (due to causal masking, tiles beyond
    # this prefix can be skipped)
    num_tiles = cdiv_fn(max_seq_prefix_len, TILE_SIZE)

    # KV_cache_modifier: ttgl.constexpr = ".cg" if ALL_DECODE else ""

    k_producer = 0
    k_consumer = 0
    v_producer = 0
    v_consumer = 0
    j_producer = segm_idx * tiles_per_segment
    j_consumer = segm_idx * tiles_per_segment
    seq_offset = j_consumer * TILE_SIZE + ttgl.arange(
        0, TILE_SIZE, layout=ttgl.SliceLayout(0, Q_BLOCKED_LAYOUT)
    )

    for _ in range(num_stages - 1):
        j_producer, offs_k_t_starts, offs_v_t_starts = _tdm_get_kv_offsets(
            j_producer,
            kv_head_idx,
            block_tables_ptr,
            block_table_offset,
            stride_k_cache_0,
            stride_k_cache_2,
            stride_v_cache_0,
            stride_v_cache_2,
        )
        k_producer = _tdm_async_load_to_lds(
            k_producer,
            src=k_desc,
            offsets=[
                0,
                offs_k_t_starts,
            ],
            dest=smem_K,
            pred_i32=pred_i32,
            num_stages=num_stages,
        )
        v_producer = _tdm_async_load_to_lds(
            v_producer,
            src=v_desc,
            offsets=[
                0,
                offs_v_t_starts,
            ],
            dest=smem_V,
            pred_i32=pred_i32,
            num_stages=num_stages,
        )

    # iterate through tiles within current segment
    for _ in range(tiles_per_segment - (num_stages - 1)):
        if j_producer < num_tiles:
            j_producer, offs_k_t_starts, offs_v_t_starts = _tdm_get_kv_offsets(
                j_producer,
                kv_head_idx,
                block_tables_ptr,
                block_table_offset,
                stride_k_cache_0,
                stride_k_cache_2,
                stride_v_cache_0,
                stride_v_cache_2,
            )
            k_producer = _tdm_async_load_to_lds(
                k_producer,
                src=k_desc,
                offsets=[
                    0,
                    offs_k_t_starts,
                ],
                dest=smem_K,
                pred_i32=pred_i32,
                num_stages=num_stages,
            )
            v_producer = _tdm_async_load_to_lds(
                v_producer,
                src=v_desc,
                offsets=[
                    0,
                    offs_v_t_starts,
                ],
                dest=smem_V,
                pred_i32=pred_i32,
                num_stages=num_stages,
            )

        if j_consumer < num_tiles:
            # K : shape = (HEAD_SIZE_PADDED, TILE_SIZE), layout = K_DOT_LAYOUT
            k_consumer, K = _tdm_request_from_lds(
                k_consumer,
                k_scale,
                Q.dtype,
                smem_K,
                asycn_wait=(num_stages - 1) * 2 + 1,
                layout=K_DOT_LAYOUT,
                transpose=True,
                num_ctas=num_ctas,
                num_stages=num_stages,
            )

            # P : shape = (BLOCK_M, TILE_SIZE), layout = Q_BLOCKED_LAYOUT
            # L : shape = (BLOCK_M, ), layout = ttgl.SliceLayout(1, Q_BLOCKED_LAYOUT)
            # M : shape = (BLOCK_M, ), layout = ttgl.SliceLayout(1, Q_BLOCKED_LAYOUT)
            # acc : shape = (BLOCK_M, HEAD_SIZE_PADDED), layout = PV_WMMA_LAYOUT
            P, L, M, acc = _perform_QK_wmma_and_update_L_M(
                Q,
                K,
                L,
                M,
                acc,
                qq_bias_row_ptrs,
                seq_offset,
                query_mask_1,
                query_mask_0,
                context_len,
                query_pos,
                alibi_slope,
                qq_bias_stride_0,
                qk_scale,
                softcap,
                RCP_LN2,
                BLOCK_M,
                TILE_SIZE,
                USE_SOFTCAP,
                SLIDING_WINDOW,
                USE_ALIBI_SLOPES,
                USE_QQ_BIAS,
                Q_BLOCKED_LAYOUT,
                QK_WMMA_LAYOUT,
                PV_WMMA_LAYOUT,
            )

            # V : shape = (TILE_SIZE, HEAD_SIZE_PADDED), layout = V_DOT_LAYOUT
            v_consumer, V = _tdm_request_from_lds(
                v_consumer,
                v_scale,
                Q.dtype,
                smem_V,
                asycn_wait=(num_stages - 1) * 2,
                layout=V_DOT_LAYOUT,
                transpose=False,
                num_ctas=num_ctas,
                num_stages=num_stages,
            )

            # acc : shape = (BLOCK_M, HEAD_SIZE_PADDED), layout = PV_WMMA_LAYOUT
            acc = _perform_PV_wmma(P, V, acc, P_DOT_LAYOUT)

            j_consumer = j_consumer + 1
            seq_offset += TILE_SIZE

    for _ in range(num_stages - 1):
        if j_consumer < num_tiles:
            # K : shape = (HEAD_SIZE_PADDED, TILE_SIZE), layout = K_DOT_LAYOUT
            k_consumer, K = _tdm_request_from_lds(
                k_consumer,
                k_scale,
                Q.dtype,
                smem_K,
                asycn_wait=(num_stages - 1) * 2 + 1,
                layout=K_DOT_LAYOUT,
                transpose=True,
                num_ctas=num_ctas,
                num_stages=num_stages,
            )

            # P : shape = (BLOCK_M, TILE_SIZE), layout = Q_BLOCKED_LAYOUT
            # L : shape = (BLOCK_M, ), layout = ttgl.SliceLayout(1, Q_BLOCKED_LAYOUT)
            # M : shape = (BLOCK_M, ), layout = ttgl.SliceLayout(1, Q_BLOCKED_LAYOUT)
            # acc : shape = (BLOCK_M, HEAD_SIZE_PADDED), layout = PV_WMMA_LAYOUT
            P, L, M, acc = _perform_QK_wmma_and_update_L_M(
                Q,
                K,
                L,
                M,
                acc,
                qq_bias_row_ptrs,
                seq_offset,
                query_mask_1,
                query_mask_0,
                context_len,
                query_pos,
                alibi_slope,
                qq_bias_stride_0,
                qk_scale,
                softcap,
                RCP_LN2,
                BLOCK_M,
                TILE_SIZE,
                USE_SOFTCAP,
                SLIDING_WINDOW,
                USE_ALIBI_SLOPES,
                USE_QQ_BIAS,
                Q_BLOCKED_LAYOUT,
                QK_WMMA_LAYOUT,
                PV_WMMA_LAYOUT,
            )

            # V : shape = (TILE_SIZE, HEAD_SIZE_PADDED), layout = V_DOT_LAYOUT
            v_consumer, V = _tdm_request_from_lds(
                v_consumer,
                v_scale,
                Q.dtype,
                smem_V,
                asycn_wait=(num_stages - 1) * 2,
                layout=V_DOT_LAYOUT,
                transpose=False,
                num_ctas=num_ctas,
                num_stages=num_stages,
            )

            # acc : shape = (BLOCK_M, HEAD_SIZE_PADDED), layout = PV_WMMA_LAYOUT
            acc = _perform_PV_wmma(P, V, acc, P_DOT_LAYOUT)

            j_consumer = j_consumer + 1
            seq_offset += TILE_SIZE

    # store segm_output
    # acc : shape = (BLOCK_M, HEAD_SIZE_PADDED), layout = Q_BLOCKED_LAYOUT
    acc = ttgl.convert_layout(acc, layout=Q_BLOCKED_LAYOUT)
    segm_output_offset = (
        query_offset_0[:, None]
        * (num_query_heads * NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED)
        + query_offset_1[:, None] * (NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED)
        + segm_idx * HEAD_SIZE_PADDED
        + offs_q_d[None, :]
    )
    ttgl.amd.cdna4.buffer_store(
        stored_value=acc,
        ptr=segm_output_ptr,
        offsets=segm_output_offset,
        mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
    )

    # store segm_max and segm_expsum
    # L : shape = (BLOCK_M, ), layout = ttgl.SliceLayout(1, Q_BLOCKED_LAYOUT)
    # M : shape = (BLOCK_M, ), layout = ttgl.SliceLayout(1, Q_BLOCKED_LAYOUT)
    segm_offset = (
        query_offset_0 * (num_query_heads * NUM_SEGMENTS_PER_SEQ)
        + query_offset_1 * NUM_SEGMENTS_PER_SEQ
        + segm_idx
    )
    ttgl.amd.cdna4.buffer_store(
        stored_value=M,
        ptr=segm_max_ptr,
        offsets=segm_offset,
        mask=query_mask_0 & query_mask_1,
    )
    ttgl.amd.cdna4.buffer_store(
        stored_value=L,
        ptr=segm_expsum_ptr,
        offsets=segm_offset,
        mask=query_mask_0 & query_mask_1,
    )


@gluon.jit
def _async_load_to_lds(
    producer,
    dest,
    ptr,
    offsets,
    mask,
    num_stages: ttgl.constexpr,
    cache_modifier: ttgl.constexpr,
):
    ttgl.amd.cdna4.async_copy.global_load_to_shared(
        dest=dest.index(producer % num_stages),
        ptr=ptr + offsets,
        mask=mask,
        cache_modifier=cache_modifier,
    )
    # ttgl.amd.cdna4.async_copy.buffer_load_to_shared(
    #     dest=dest.index(producer % num_stages),
    #     ptr=ptr,
    #     offsets=offsets.to(ttgl.int32),
    #     mask=mask,
    #     cache_modifier=cache_modifier,
    # )
    ttgl.amd.cdna4.async_copy.commit_group()
    return producer + 1


@gluon.jit
def _request_from_lds(
    consumer,
    kv_scale,
    Q_dtype,
    smem,
    layout,
    wait_group,
    num_stages: ttgl.constexpr,
):
    ttgl.amd.cdna4.async_copy.wait_group(wait_group)
    KV = smem.index(consumer % num_stages).load(layout=layout)
    if KV.dtype.is_fp8() and not Q_dtype.is_fp8():
        KV = (KV.to(ttgl.float32) * ttgl.load(kv_scale)).to(Q_dtype)
    return KV, consumer + 1


@gluon.jit
def _get_kv_offsets(
    j,
    kv_head_idx,
    block_tables_ptr,
    block_table_offset,
    offs_k_t,
    offs_k_d,
    offs_v_t,
    offs_v_d,
    max_seq_prefix_len,
    stride_k_cache_0: ttgl.int64,
    stride_k_cache_1: ttgl.int64,
    stride_k_cache_2: ttgl.int64,
    stride_k_cache_3: ttgl.constexpr,
    stride_v_cache_0: ttgl.int64,
    stride_v_cache_1: ttgl.int64,
    stride_v_cache_2: ttgl.int64,
    stride_v_cache_3: ttgl.constexpr,
    K_BLOCKED_LAYOUT: ttgl.constexpr,
    Q_BLOCKED_LAYOUT: ttgl.constexpr,
    TILE_SIZE: ttgl.constexpr,
    BLOCK_SIZE: ttgl.constexpr,
):
    # seq_k_offset : shape = (TILE_SIZE, ), layout = ttgl.SliceLayout(0, K_BLOCKED_LAYOUT)
    # seq_v_offset : shape = (TILE_SIZE, ), layout = ttgl.SliceLayout(1, Q_BLOCKED_LAYOUT)
    seq_k_offset = j * TILE_SIZE + offs_k_t
    seq_v_offset = j * TILE_SIZE + offs_v_t

    if TILE_SIZE == BLOCK_SIZE:
        tile_k_mask = ttgl.full(
            (1,), 1, dtype=tl.int1, layout=ttgl.SliceLayout(0, K_BLOCKED_LAYOUT)
        )
        tile_v_mask = ttgl.full(
            (1,), 1, dtype=tl.int1, layout=ttgl.SliceLayout(1, Q_BLOCKED_LAYOUT)
        )
    else:
        tile_k_mask = seq_k_offset < max_seq_prefix_len
        tile_v_mask = seq_v_offset < max_seq_prefix_len

    physical_block_idx_k = ttgl.amd.cdna4.buffer_load(
        ptr=block_tables_ptr,
        offsets=(block_table_offset + seq_k_offset // BLOCK_SIZE).to(ttgl.int32),
    ).to(tl.int64)

    physical_block_idx_v = ttgl.amd.cdna4.buffer_load(
        ptr=block_tables_ptr,
        offsets=(block_table_offset + seq_v_offset // BLOCK_SIZE).to(ttgl.int32),
    ).to(tl.int64)

    v_offset = (
        physical_block_idx_v[:, None] * stride_v_cache_0
        + kv_head_idx * stride_v_cache_2
        + offs_v_d[None, :] * stride_v_cache_3
        + (seq_v_offset % BLOCK_SIZE)[:, None] * stride_v_cache_1
    )

    k_offset = (
        physical_block_idx_k[None, :] * stride_k_cache_0
        + kv_head_idx * stride_k_cache_2
        + offs_k_d[:, None] * stride_k_cache_3
        + (seq_k_offset % BLOCK_SIZE)[None, :] * stride_k_cache_1
    )

    return j + 1, k_offset, v_offset, tile_k_mask, tile_v_mask


@gluon.jit
def gluon_kernel_unified_attention_3d_pipelined(
    segm_output_ptr,
    # [num_tokens, num_query_heads, num_segments, head_size]
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
    k_scale,  # float32
    v_scale,  # float32
    softcap,  # float32
    num_tokens,  # int
    num_query_heads: ttgl.constexpr,  # int
    num_queries_per_kv: ttgl.constexpr,  # int
    block_table_stride: ttgl.int64,  # int
    query_stride_0: ttgl.int64,  # int
    query_stride_1: ttgl.int64,  # int, should be equal to head_size
    qq_bias_stride_0: ttgl.int64,  # int
    NUM_BLOCKS: ttgl.constexpr,  # int
    BLOCK_SIZE: ttgl.constexpr,  # int
    TILE_SIZE: ttgl.constexpr,  # int, must be power of 2
    HEAD_SIZE: ttgl.constexpr,  # int
    HEAD_SIZE_PADDED: ttgl.constexpr,  # int, must be power of 2
    USE_ALIBI_SLOPES: ttgl.constexpr,  # bool
    USE_QQ_BIAS: ttgl.constexpr,  # bool
    USE_SOFTCAP: ttgl.constexpr,  # bool
    USE_SINKS: ttgl.constexpr,  # bool
    SLIDING_WINDOW: ttgl.constexpr,  # int
    stride_k_cache_0: ttgl.int64,  # int
    stride_k_cache_1: ttgl.int64,  # int
    stride_k_cache_2: ttgl.int64,  # int
    stride_k_cache_3: ttgl.constexpr,  # int
    stride_v_cache_0: ttgl.int64,  # int
    stride_v_cache_1: ttgl.int64,  # int
    stride_v_cache_2: ttgl.int64,  # int
    stride_v_cache_3: ttgl.constexpr,  # int
    query_start_len_ptr,  # [num_seqs+1]
    BLOCK_Q: ttgl.constexpr,  # int
    num_seqs: ttgl.int32,
    BLOCK_M: ttgl.constexpr,  # int
    NUM_SEGMENTS_PER_SEQ: ttgl.constexpr,  # int
    num_warps: ttgl.constexpr,  # int
    num_stages: ttgl.constexpr,  # int
    QK_WMMA_LAYOUT: ttgl.constexpr,
    PV_WMMA_LAYOUT: ttgl.constexpr,
    Q_DOT_LAYOUT: ttgl.constexpr,
    K_DOT_LAYOUT: ttgl.constexpr,
    P_DOT_LAYOUT: ttgl.constexpr,
    V_DOT_LAYOUT: ttgl.constexpr,
    Q_SHARED_LAYOUT: ttgl.constexpr,
    K_SHARED_LAYOUT: ttgl.constexpr,
    V_SHARED_LAYOUT: ttgl.constexpr,
    Q_BLOCKED_LAYOUT: ttgl.constexpr,
    K_BLOCKED_LAYOUT: ttgl.constexpr,
    ALL_DECODE: ttgl.constexpr = False,  # bool
):
    q_block_global_idx = ttgl.program_id(0)
    kv_head_idx = ttgl.program_id(1)
    segm_idx = ttgl.program_id(2)
    num_ctas: ttgl.constexpr = ttgl.num_ctas()
    pred = 1
    pred_i32 = pred.to(ttgl.int32) if hasattr(pred, "to") else pred

    # needed to use exp2 (exp2 -> exp conversion)
    RCP_LN2 = 1.4426950408889634
    qk_scale = scale * RCP_LN2

    seq_idx = _find_seq_idx(
        query_start_len_ptr, q_block_global_idx, num_seqs, BLOCK_Q, True
    )

    q_block_local_idx, cur_batch_query_len, cur_batch_in_all_start_index = (
        _get_q_metadata(
            query_start_len_ptr,
            seq_idx,
            q_block_global_idx,
            BLOCK_Q,
        )
    )

    if q_block_local_idx * BLOCK_Q >= cur_batch_query_len:
        return

    seq_len, tiles_per_segment = _get_seq_metadata(
        seq_lens_ptr,
        seq_idx,
        TILE_SIZE,
        NUM_SEGMENTS_PER_SEQ,
    )

    if segm_idx * tiles_per_segment * TILE_SIZE >= seq_len:
        return

    # block table offset for this particular sequence
    block_table_offset = seq_idx * block_table_stride

    # context length for this particular sequence
    context_len = seq_len - cur_batch_query_len

    smem_Q = ttgl.allocate_shared_memory(
        query_ptr.type.element_ty, [BLOCK_M, HEAD_SIZE_PADDED], layout=Q_SHARED_LAYOUT
    )
    smem_K = ttgl.allocate_shared_memory(
        key_cache_ptr.type.element_ty,
        [num_stages, HEAD_SIZE_PADDED, TILE_SIZE],
        layout=K_SHARED_LAYOUT,
    )
    smem_V = ttgl.allocate_shared_memory(
        value_cache_ptr.type.element_ty,
        [num_stages, TILE_SIZE, HEAD_SIZE_PADDED],
        layout=V_SHARED_LAYOUT,
    )

    offs_q_m = ttgl.arange(0, BLOCK_M, layout=ttgl.SliceLayout(1, Q_BLOCKED_LAYOUT))
    offs_q_d = ttgl.arange(
        0, HEAD_SIZE_PADDED, layout=ttgl.SliceLayout(0, Q_BLOCKED_LAYOUT)
    )

    offs_k_t = ttgl.arange(0, TILE_SIZE, layout=ttgl.SliceLayout(0, K_BLOCKED_LAYOUT))
    offs_k_d = ttgl.arange(
        0, HEAD_SIZE_PADDED, layout=ttgl.SliceLayout(1, K_BLOCKED_LAYOUT)
    )

    offs_v_t = ttgl.arange(0, TILE_SIZE, layout=ttgl.SliceLayout(1, Q_BLOCKED_LAYOUT))
    offs_v_d = ttgl.arange(
        0, HEAD_SIZE_PADDED, layout=ttgl.SliceLayout(0, Q_BLOCKED_LAYOUT)
    )

    query_pos = q_block_local_idx * BLOCK_Q + offs_q_m // num_queries_per_kv

    query_offset_0 = cur_batch_in_all_start_index + query_pos
    query_offset_1 = kv_head_idx * num_queries_per_kv + offs_q_m % num_queries_per_kv
    query_offset = (
        query_offset_0[:, None] * query_stride_0
        + query_offset_1[:, None] * query_stride_1
        + offs_q_d[None, :]
    )

    if HEAD_SIZE_PADDED != HEAD_SIZE:
        dim_mask = offs_q_d < HEAD_SIZE
    else:
        dim_mask = ttgl.full((1,), 1, dtype=tl.int1)

    query_mask_0 = query_pos < cur_batch_query_len
    query_mask_1 = query_offset_1 < num_query_heads

    # Q_load : shape = (BLOCK_M, HEAD_SIZE_PADDED), layout = Q_BLOCKED_LAYOUT
    Q_load = ttgl.amd.cdna4.buffer_load(
        ptr=query_ptr,
        offsets=query_offset.to(ttgl.int32),
        mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
        other=0.0,
    )
    smem_Q.store(Q_load)
    # Q : shape = (BLOCK_M, HEAD_SIZE_PADDED), layout = Q_DOT_LAYOUT
    Q = smem_Q.load(layout=Q_DOT_LAYOUT)

    L, M, acc = _allocate_L_M_acc(
        sink_ptr,
        segm_idx,
        query_offset_1,
        query_mask_1,
        RCP_LN2,
        BLOCK_M,
        Q_BLOCKED_LAYOUT,
        HEAD_SIZE_PADDED,
        PV_WMMA_LAYOUT,
        USE_SINKS,
    )

    # alibi slope for this head
    alibi_slope = None
    if USE_ALIBI_SLOPES:
        alibi_slope = tl.load(
            alibi_slopes_ptr + query_offset_1, mask=query_mask_1, other=0.0
        )

    # query-query attention bias
    qq_bias_row_ptrs = None
    if USE_QQ_BIAS:
        qq_bias_row_ptrs = (
            qq_bias_ptr + query_pos[:, None] * qq_bias_stride_0
        )  # shape: [BLOCK_M]

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
    max_seq_prefix_len = ttgl.minimum(max_seq_prefix_len, seq_len)

    # calculate the number of tiles that need to be processed to
    # cover the longest sequence prefix (due to causal masking, tiles beyond
    # this prefix can be skipped)
    num_tiles = cdiv_fn(max_seq_prefix_len, TILE_SIZE)

    KV_cache_modifier: ttgl.constexpr = ".cg" if ALL_DECODE else ""

    k_producer = 0
    k_consumer = 0
    v_producer = 0
    v_consumer = 0
    j_producer = segm_idx * tiles_per_segment
    j_consumer = segm_idx * tiles_per_segment
    seq_offset = j_consumer * TILE_SIZE + ttgl.arange(
        0, TILE_SIZE, layout=ttgl.SliceLayout(0, Q_BLOCKED_LAYOUT)
    )

    for _ in range(num_stages - 1):
        j_producer, k_offset, v_offset, tile_k_mask, tile_v_mask = _get_kv_offsets(
            j_producer,
            kv_head_idx,
            block_tables_ptr,
            block_table_offset,
            offs_k_t,
            offs_k_d,
            offs_v_t,
            offs_v_d,
            max_seq_prefix_len,
            stride_k_cache_0,
            stride_k_cache_1,
            stride_k_cache_2,
            stride_k_cache_3,
            stride_v_cache_0,
            stride_v_cache_1,
            stride_v_cache_2,
            stride_v_cache_3,
            K_BLOCKED_LAYOUT,
            Q_BLOCKED_LAYOUT,
            TILE_SIZE,
            BLOCK_SIZE,
        )
        k_producer = _async_load_to_lds(
            k_producer,
            dest=smem_K,
            ptr=key_cache_ptr,
            offsets=k_offset,
            mask=dim_mask[:, None] & tile_k_mask[None, :],
            num_stages=num_stages,
            cache_modifier=KV_cache_modifier,
        )
        v_producer = _async_load_to_lds(
            v_producer,
            dest=smem_V,
            ptr=value_cache_ptr,
            offsets=v_offset,
            mask=dim_mask[None, :] & tile_v_mask[:, None],
            num_stages=num_stages,
            cache_modifier=KV_cache_modifier,
        )

    # iterate through tiles within current segment
    for _ in range(tiles_per_segment - (num_stages - 1)):
        if j_producer < num_tiles:
            j_producer, k_offset, v_offset, tile_k_mask, tile_v_mask = _get_kv_offsets(
                j_producer,
                kv_head_idx,
                block_tables_ptr,
                block_table_offset,
                offs_k_t,
                offs_k_d,
                offs_v_t,
                offs_v_d,
                max_seq_prefix_len,
                stride_k_cache_0,
                stride_k_cache_1,
                stride_k_cache_2,
                stride_k_cache_3,
                stride_v_cache_0,
                stride_v_cache_1,
                stride_v_cache_2,
                stride_v_cache_3,
                K_BLOCKED_LAYOUT,
                Q_BLOCKED_LAYOUT,
                TILE_SIZE,
                BLOCK_SIZE,
            )

            # K_load : shape = (HEAD_SIZE_PADDED, TILE_SIZE), layout = K_BLOCKED_LAYOUT
            k_producer = _async_load_to_lds(
                k_producer,
                dest=smem_K,
                ptr=key_cache_ptr,
                offsets=k_offset,
                mask=dim_mask[:, None] & tile_k_mask[None, :],
                num_stages=num_stages,
                cache_modifier=KV_cache_modifier,
            )

            # V_load : shape = (TILE_SIZE, HEAD_SIZE_PADDED), layout = Q_BLOCKED_LAYOUT
            v_producer = _async_load_to_lds(
                v_producer,
                dest=smem_V,
                ptr=value_cache_ptr,
                offsets=v_offset,
                mask=dim_mask[None, :] & tile_v_mask[:, None],
                num_stages=num_stages,
                cache_modifier=KV_cache_modifier,
            )

        if j_consumer < num_tiles:
            # K : shape = (HEAD_SIZE_PADDED, TILE_SIZE), layout = K_DOT_LAYOUT
            K, k_consumer = _request_from_lds(
                k_consumer,
                k_scale,
                Q.dtype,
                smem_K,
                layout=K_DOT_LAYOUT,
                wait_group=(num_stages - 1) * 2 + 1,
                num_stages=num_stages,
            )

            # P : shape = (BLOCK_M, TILE_SIZE), layout = Q_BLOCKED_LAYOUT
            # L : shape = (BLOCK_M, ), layout = ttgl.SliceLayout(1, Q_BLOCKED_LAYOUT)
            # M : shape = (BLOCK_M, ), layout = ttgl.SliceLayout(1, Q_BLOCKED_LAYOUT)
            # acc : shape = (BLOCK_M, HEAD_SIZE_PADDED), layout = PV_WMMA_LAYOUT
            P, L, M, acc = _perform_QK_wmma_and_update_L_M(
                Q,
                K,
                L,
                M,
                acc,
                qq_bias_row_ptrs,
                seq_offset,
                query_mask_1,
                query_mask_0,
                context_len,
                query_pos,
                alibi_slope,
                qq_bias_stride_0,
                qk_scale,
                softcap,
                RCP_LN2,
                BLOCK_M,
                TILE_SIZE,
                USE_SOFTCAP,
                SLIDING_WINDOW,
                USE_ALIBI_SLOPES,
                USE_QQ_BIAS,
                Q_BLOCKED_LAYOUT,
                QK_WMMA_LAYOUT,
                PV_WMMA_LAYOUT,
            )

            # V : shape = (TILE_SIZE, HEAD_SIZE_PADDED), layout = V_DOT_LAYOUT
            V, v_consumer = _request_from_lds(
                v_consumer,
                v_scale,
                Q.dtype,
                smem_V,
                layout=V_DOT_LAYOUT,
                wait_group=(num_stages - 1) * 2,
                num_stages=num_stages,
            )

            # acc : shape = (BLOCK_M, HEAD_SIZE_PADDED), layout = PV_WMMA_LAYOUT
            acc = _perform_PV_wmma(P, V, acc, P_DOT_LAYOUT)

            j_consumer = j_consumer + 1
            seq_offset += TILE_SIZE

    for _ in range(num_stages - 1):
        if j_consumer < num_tiles:
            # K : shape = (HEAD_SIZE_PADDED, TILE_SIZE), layout = K_DOT_LAYOUT
            K, k_consumer = _request_from_lds(
                k_consumer,
                k_scale,
                Q.dtype,
                smem_K,
                layout=K_DOT_LAYOUT,
                wait_group=(num_stages - 1) * 2 + 1,
                num_stages=num_stages,
            )

            # P : shape = (BLOCK_M, TILE_SIZE), layout = Q_BLOCKED_LAYOUT
            # L : shape = (BLOCK_M, ), layout = ttgl.SliceLayout(1, Q_BLOCKED_LAYOUT)
            # M : shape = (BLOCK_M, ), layout = ttgl.SliceLayout(1, Q_BLOCKED_LAYOUT)
            # acc : shape = (BLOCK_M, HEAD_SIZE_PADDED), layout = PV_WMMA_LAYOUT
            P, L, M, acc = _perform_QK_wmma_and_update_L_M(
                Q,
                K,
                L,
                M,
                acc,
                qq_bias_row_ptrs,
                seq_offset,
                query_mask_1,
                query_mask_0,
                context_len,
                query_pos,
                alibi_slope,
                qq_bias_stride_0,
                qk_scale,
                softcap,
                RCP_LN2,
                BLOCK_M,
                TILE_SIZE,
                USE_SOFTCAP,
                SLIDING_WINDOW,
                USE_ALIBI_SLOPES,
                USE_QQ_BIAS,
                Q_BLOCKED_LAYOUT,
                QK_WMMA_LAYOUT,
                PV_WMMA_LAYOUT,
            )

            # V : shape = (TILE_SIZE, HEAD_SIZE_PADDED), layout = V_DOT_LAYOUT
            V, v_consumer = _request_from_lds(
                v_consumer,
                v_scale,
                Q.dtype,
                smem_V,
                layout=V_DOT_LAYOUT,
                wait_group=(num_stages - 1) * 2,
                num_stages=num_stages,
            )

            # acc : shape = (BLOCK_M, HEAD_SIZE_PADDED), layout = PV_WMMA_LAYOUT
            acc = _perform_PV_wmma(P, V, acc, P_DOT_LAYOUT)

            j_consumer = j_consumer + 1
            seq_offset += TILE_SIZE

    # store segm_output
    # acc : shape = (BLOCK_M, HEAD_SIZE_PADDED), layout = Q_BLOCKED_LAYOUT
    acc = ttgl.convert_layout(acc, layout=Q_BLOCKED_LAYOUT)
    segm_output_offset = (
        query_offset_0[:, None]
        * (num_query_heads * NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED)
        + query_offset_1[:, None] * (NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED)
        + segm_idx * HEAD_SIZE_PADDED
        + offs_q_d[None, :]
    )
    ttgl.amd.cdna4.buffer_store(
        stored_value=acc,
        ptr=segm_output_ptr,
        offsets=segm_output_offset,
        mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
    )

    # store segm_max and segm_expsum
    # L : shape = (BLOCK_M, ), layout = ttgl.SliceLayout(1, Q_BLOCKED_LAYOUT)
    # M : shape = (BLOCK_M, ), layout = ttgl.SliceLayout(1, Q_BLOCKED_LAYOUT)
    segm_offset = (
        query_offset_0 * (num_query_heads * NUM_SEGMENTS_PER_SEQ)
        + query_offset_1 * NUM_SEGMENTS_PER_SEQ
        + segm_idx
    )
    ttgl.amd.cdna4.buffer_store(
        stored_value=M,
        ptr=segm_max_ptr,
        offsets=segm_offset,
        mask=query_mask_0 & query_mask_1,
    )
    ttgl.amd.cdna4.buffer_store(
        stored_value=L,
        ptr=segm_expsum_ptr,
        offsets=segm_offset,
        mask=query_mask_0 & query_mask_1,
    )


@gluon.jit
def _buffer_load_to_reg_shuffle_via_lds(
    x_scale,
    Q_dtype,
    smem,
    ptr,
    offsets,
    mask,
    layout: ttgl.constexpr,
    cache_modifier: ttgl.constexpr,
):
    X = ttgl.amd.cdna4.buffer_load(
        ptr=ptr,
        offsets=offsets.to(ttgl.int32),
        mask=mask,
        other=0.0,
        cache=cache_modifier,
    )
    if X.dtype.is_fp8() and not Q_dtype.is_fp8():
        X = (X.to(ttgl.float32) * ttgl.load(x_scale)).to(Q_dtype)
    smem.store(X)
    # K : shape = (HEAD_SIZE_PADDED, TILE_SIZE), layout = K_DOT_LAYOUT
    return smem.load(layout=layout)


@gluon.jit
def gluon_kernel_unified_attention_3d(
    segm_output_ptr,
    # [num_tokens, num_query_heads, num_segments, head_size]
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
    k_scale,  # float32
    v_scale,  # float32
    softcap,  # float32
    num_tokens,  # int
    num_query_heads: ttgl.constexpr,  # int
    num_queries_per_kv: ttgl.constexpr,  # int
    block_table_stride: ttgl.int64,  # int
    query_stride_0: ttgl.int64,  # int
    query_stride_1: ttgl.int64,  # int, should be equal to head_size
    qq_bias_stride_0: ttgl.int64,  # int
    NUM_BLOCKS: ttgl.constexpr,  # int
    BLOCK_SIZE: ttgl.constexpr,  # int
    TILE_SIZE: ttgl.constexpr,  # int, must be power of 2
    HEAD_SIZE: ttgl.constexpr,  # int
    HEAD_SIZE_PADDED: ttgl.constexpr,  # int, must be power of 2
    USE_ALIBI_SLOPES: ttgl.constexpr,  # bool
    USE_QQ_BIAS: ttgl.constexpr,  # bool
    USE_SOFTCAP: ttgl.constexpr,  # bool
    USE_SINKS: ttgl.constexpr,  # bool
    SLIDING_WINDOW: ttgl.constexpr,  # int
    stride_k_cache_0: ttgl.int64,  # int
    stride_k_cache_1: ttgl.int64,  # int
    stride_k_cache_2: ttgl.int64,  # int
    stride_k_cache_3: ttgl.constexpr,  # int
    stride_v_cache_0: ttgl.int64,  # int
    stride_v_cache_1: ttgl.int64,  # int
    stride_v_cache_2: ttgl.int64,  # int
    stride_v_cache_3: ttgl.constexpr,  # int
    query_start_len_ptr,  # [num_seqs+1]
    BLOCK_Q: ttgl.constexpr,  # int
    num_seqs: ttgl.int32,
    BLOCK_M: ttgl.constexpr,  # int
    NUM_SEGMENTS_PER_SEQ: ttgl.constexpr,  # int
    num_warps: ttgl.constexpr,  # int
    num_stages: ttgl.constexpr,  # int
    QK_WMMA_LAYOUT: ttgl.constexpr,
    PV_WMMA_LAYOUT: ttgl.constexpr,
    Q_DOT_LAYOUT: ttgl.constexpr,
    K_DOT_LAYOUT: ttgl.constexpr,
    P_DOT_LAYOUT: ttgl.constexpr,
    V_DOT_LAYOUT: ttgl.constexpr,
    Q_SHARED_LAYOUT: ttgl.constexpr,
    K_SHARED_LAYOUT: ttgl.constexpr,
    V_SHARED_LAYOUT: ttgl.constexpr,
    Q_BLOCKED_LAYOUT: ttgl.constexpr,
    K_BLOCKED_LAYOUT: ttgl.constexpr,
    ALL_DECODE: ttgl.constexpr = False,  # bool
):
    q_block_global_idx = ttgl.program_id(0)
    kv_head_idx = ttgl.program_id(1)
    segm_idx = ttgl.program_id(2)

    # needed to use exp2 (exp2 -> exp conversion)
    RCP_LN2 = 1.4426950408889634
    qk_scale = scale * RCP_LN2

    seq_idx = _find_seq_idx(
        query_start_len_ptr, q_block_global_idx, num_seqs, BLOCK_Q, True
    )

    q_block_local_idx, cur_batch_query_len, cur_batch_in_all_start_index = (
        _get_q_metadata(
            query_start_len_ptr,
            seq_idx,
            q_block_global_idx,
            BLOCK_Q,
        )
    )

    if q_block_local_idx * BLOCK_Q >= cur_batch_query_len:
        return

    seq_len, tiles_per_segment = _get_seq_metadata(
        seq_lens_ptr,
        seq_idx,
        TILE_SIZE,
        NUM_SEGMENTS_PER_SEQ,
    )

    if segm_idx * tiles_per_segment * TILE_SIZE >= seq_len:
        return

    # block table offset for this particular sequence
    block_table_offset = seq_idx * block_table_stride

    # context length for this particular sequence
    context_len = seq_len - cur_batch_query_len

    smem_Q = ttgl.allocate_shared_memory(
        query_ptr.type.element_ty, [BLOCK_M, HEAD_SIZE_PADDED], layout=Q_SHARED_LAYOUT
    )
    smem_K = ttgl.allocate_shared_memory(
        key_cache_ptr.type.element_ty,
        [HEAD_SIZE_PADDED, TILE_SIZE],
        layout=K_SHARED_LAYOUT,
    )
    smem_V = ttgl.allocate_shared_memory(
        value_cache_ptr.type.element_ty,
        [TILE_SIZE, HEAD_SIZE_PADDED],
        layout=V_SHARED_LAYOUT,
    )

    offs_q_m = ttgl.arange(0, BLOCK_M, layout=ttgl.SliceLayout(1, Q_BLOCKED_LAYOUT))
    offs_q_d = ttgl.arange(
        0, HEAD_SIZE_PADDED, layout=ttgl.SliceLayout(0, Q_BLOCKED_LAYOUT)
    )

    offs_k_t = ttgl.arange(0, TILE_SIZE, layout=ttgl.SliceLayout(0, K_BLOCKED_LAYOUT))
    offs_k_d = ttgl.arange(
        0, HEAD_SIZE_PADDED, layout=ttgl.SliceLayout(1, K_BLOCKED_LAYOUT)
    )

    offs_v_t = ttgl.arange(0, TILE_SIZE, layout=ttgl.SliceLayout(1, Q_BLOCKED_LAYOUT))
    offs_v_d = ttgl.arange(
        0, HEAD_SIZE_PADDED, layout=ttgl.SliceLayout(0, Q_BLOCKED_LAYOUT)
    )

    query_pos = q_block_local_idx * BLOCK_Q + offs_q_m // num_queries_per_kv

    query_offset_0 = cur_batch_in_all_start_index + query_pos
    query_offset_1 = kv_head_idx * num_queries_per_kv + offs_q_m % num_queries_per_kv
    query_offset = (
        query_offset_0[:, None] * query_stride_0
        + query_offset_1[:, None] * query_stride_1
        + offs_q_d[None, :]
    )

    if HEAD_SIZE_PADDED != HEAD_SIZE:
        dim_mask = offs_q_d < HEAD_SIZE
    else:
        dim_mask = ttgl.full((1,), 1, dtype=tl.int1)

    query_mask_0 = query_pos < cur_batch_query_len
    query_mask_1 = query_offset_1 < num_query_heads

    # Q_load : shape = (BLOCK_M, HEAD_SIZE_PADDED), layout = Q_BLOCKED_LAYOUT
    Q_load = ttgl.amd.cdna4.buffer_load(
        ptr=query_ptr,
        offsets=query_offset.to(ttgl.int32),
        mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
        other=0.0,
    )
    smem_Q.store(Q_load)
    # Q : shape = (BLOCK_M, HEAD_SIZE_PADDED), layout = Q_DOT_LAYOUT
    Q = smem_Q.load(layout=Q_DOT_LAYOUT)

    L, M, acc = _allocate_L_M_acc(
        sink_ptr,
        segm_idx,
        query_offset_1,
        query_mask_1,
        RCP_LN2,
        BLOCK_M,
        Q_BLOCKED_LAYOUT,
        HEAD_SIZE_PADDED,
        PV_WMMA_LAYOUT,
        USE_SINKS,
    )

    # alibi slope for this head
    alibi_slope = None
    if USE_ALIBI_SLOPES:
        alibi_slope = tl.load(
            alibi_slopes_ptr + query_offset_1, mask=query_mask_1, other=0.0
        )

    # query-query attention bias
    qq_bias_row_ptrs = None
    if USE_QQ_BIAS:
        qq_bias_row_ptrs = (
            qq_bias_ptr + query_pos[:, None] * qq_bias_stride_0
        )  # shape: [BLOCK_M]

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

    # calculate the number of tiles that need to be processed to
    # cover the longest sequence prefix (due to causal masking, tiles beyond
    # this prefix can be skipped)
    num_tiles = cdiv_fn(max_seq_prefix_len, TILE_SIZE)

    KV_cache_modifier: tl.constexpr = ".cg" if ALL_DECODE else ""
    # iterate through tiles within current segment
    for j in range(
        segm_idx * tiles_per_segment,
        min((segm_idx + 1) * tiles_per_segment, num_tiles),
    ):
        # seq_k_offset : shape = (TILE_SIZE, ), layout = ttgl.SliceLayout(0, K_BLOCKED_LAYOUT)
        # seq_v_offset : shape = (TILE_SIZE, ), layout = ttgl.SliceLayout(1, Q_BLOCKED_LAYOUT)
        seq_k_offset = j * TILE_SIZE + offs_k_t
        seq_v_offset = j * TILE_SIZE + offs_v_t

        if TILE_SIZE == BLOCK_SIZE:
            tile_k_mask = ttgl.full(
                (1,), 1, dtype=tl.int1, layout=ttgl.SliceLayout(0, K_BLOCKED_LAYOUT)
            )
            tile_v_mask = ttgl.full(
                (1,), 1, dtype=tl.int1, layout=ttgl.SliceLayout(1, Q_BLOCKED_LAYOUT)
            )
        else:
            tile_k_mask = seq_k_offset < max_seq_prefix_len
            tile_v_mask = seq_v_offset < max_seq_prefix_len

        physical_block_idx_k = ttgl.amd.cdna4.buffer_load(
            ptr=block_tables_ptr,
            offsets=(block_table_offset + seq_k_offset // BLOCK_SIZE).to(ttgl.int32),
        ).to(tl.int64)

        # # At ASM level, the following two options of getting physical_block_idx_v are identical even though they are different at IR level:
        # # 1. buffer_load directly with ttgl.SliceLayout(1, Q_BLOCKED_LAYOUT) layout (see below):
        # physical_block_idx_v = ttgl.amd.cdna4.buffer_load(
        #     ptr=block_tables_ptr,
        #     offsets=(block_table_offset + seq_v_offset // BLOCK_SIZE).to(ttgl.int32),
        # ).to(tl.int64)
        #
        # # 2. convert_layout from physical_block_idx_k, i.e., ttgl.SliceLayout(0, K_BLOCKED_LAYOUT) -> ttgl.SliceLayout(1, Q_BLOCKED_LAYOUT) (see below):
        physical_block_idx_v = ttgl.convert_layout(
            physical_block_idx_k, layout=ttgl.SliceLayout(1, Q_BLOCKED_LAYOUT)
        )

        v_offset = (
            physical_block_idx_v[:, None] * stride_v_cache_0
            + kv_head_idx * stride_v_cache_2
            + offs_v_d[None, :] * stride_v_cache_3
            + (seq_v_offset % BLOCK_SIZE)[:, None] * stride_v_cache_1
        )

        k_offset = (
            physical_block_idx_k[None, :] * stride_k_cache_0
            + kv_head_idx * stride_k_cache_2
            + offs_k_d[:, None] * stride_k_cache_3
            + (seq_k_offset % BLOCK_SIZE)[None, :] * stride_k_cache_1
        )

        # K_load : shape = (HEAD_SIZE_PADDED, TILE_SIZE), layout = K_BLOCKED_LAYOUT
        K = _buffer_load_to_reg_shuffle_via_lds(
            k_scale,
            Q.dtype,
            smem_K,
            key_cache_ptr,
            k_offset.to(ttgl.int32),
            dim_mask[:, None] & tile_k_mask[None, :],
            K_DOT_LAYOUT,
            KV_cache_modifier,
        )

        # V_load : shape = (TILE_SIZE, HEAD_SIZE_PADDED), layout = Q_BLOCKED_LAYOUT
        V = _buffer_load_to_reg_shuffle_via_lds(
            v_scale,
            Q.dtype,
            smem_V,
            value_cache_ptr,
            v_offset.to(ttgl.int32),
            dim_mask[None, :] & tile_v_mask[:, None],
            V_DOT_LAYOUT,
            KV_cache_modifier,
        )

        seq_offset = ttgl.convert_layout(
            seq_v_offset, layout=ttgl.SliceLayout(0, Q_BLOCKED_LAYOUT)
        )

        P, L, M, acc = _perform_QK_wmma_and_update_L_M(
            Q,
            K,
            L,
            M,
            acc,
            qq_bias_row_ptrs,
            seq_offset,
            query_mask_1,
            query_mask_0,
            context_len,
            query_pos,
            alibi_slope,
            qq_bias_stride_0,
            qk_scale,
            softcap,
            RCP_LN2,
            BLOCK_M,
            TILE_SIZE,
            USE_SOFTCAP,
            SLIDING_WINDOW,
            USE_ALIBI_SLOPES,
            USE_QQ_BIAS,
            Q_BLOCKED_LAYOUT,
            QK_WMMA_LAYOUT,
            PV_WMMA_LAYOUT,
        )

        # acc : shape = (BLOCK_M, HEAD_SIZE_PADDED), layout = PV_WMMA_LAYOUT
        acc = _perform_PV_wmma(P, V, acc, P_DOT_LAYOUT)

    # store segm_output
    # acc : shape = (BLOCK_M, HEAD_SIZE_PADDED), layout = Q_BLOCKED_LAYOUT
    acc = ttgl.convert_layout(acc, layout=Q_BLOCKED_LAYOUT)
    segm_output_offset = (
        query_offset_0[:, None]
        * (num_query_heads * NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED)
        + query_offset_1[:, None] * (NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED)
        + segm_idx * HEAD_SIZE_PADDED
        + offs_q_d[None, :]
    )
    ttgl.amd.cdna4.buffer_store(
        stored_value=acc,
        ptr=segm_output_ptr,
        offsets=segm_output_offset,
        mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
    )

    # store segm_max and segm_expsum
    # L : shape = (BLOCK_M, ), layout = ttgl.SliceLayout(1, Q_BLOCKED_LAYOUT)
    # M : shape = (BLOCK_M, ), layout = ttgl.SliceLayout(1, Q_BLOCKED_LAYOUT)
    segm_offset = (
        query_offset_0 * (num_query_heads * NUM_SEGMENTS_PER_SEQ)
        + query_offset_1 * NUM_SEGMENTS_PER_SEQ
        + segm_idx
    )
    ttgl.amd.cdna4.buffer_store(
        stored_value=M,
        ptr=segm_max_ptr,
        offsets=segm_offset,
        mask=query_mask_0 & query_mask_1,
    )
    ttgl.amd.cdna4.buffer_store(
        stored_value=L,
        ptr=segm_expsum_ptr,
        offsets=segm_offset,
        mask=query_mask_0 & query_mask_1,
    )


@triton.jit
def gluon_reduce_segments(
    output_ptr,  # [num_tokens, num_query_heads, head_size]
    segm_output_ptr,
    # [num_tokens, num_query_heads, max_num_segments, head_size]
    segm_max_ptr,  # [num_tokens, num_query_heads, max_num_segments]
    segm_expsum_ptr,  # [num_tokens, num_query_heads, max_num_segments]
    seq_lens_ptr,  # [num_seqs]
    num_seqs,  # int
    num_query_heads: tl.constexpr,  # int
    out_scale_inv,  # float32
    output_stride_0: tl.int64,  # int
    output_stride_1: tl.int64,  # int, should be equal to head_size
    block_table_stride: tl.int64,  # int
    TILE_SIZE: tl.constexpr,  # int
    HEAD_SIZE: tl.constexpr,  # int, must be power of 2
    HEAD_SIZE_PADDED: tl.constexpr,  # int, must be power of 2
    query_start_len_ptr,  # [num_seqs+1]
    BLOCK_Q: tl.constexpr,  # int
    NUM_SEGMENTS_PER_SEQ: tl.constexpr,  # int
    USE_FP8: tl.constexpr,  # bool
    FP8_MIN: tl.constexpr = float8_info.min,
    FP8_MAX: tl.constexpr = float8_info.max,
):
    query_token_idx = tl.program_id(0)
    query_head_idx = tl.program_id(1)

    seq_idx = _find_seq_idx(
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

    if USE_FP8:
        acc = acc * tl.load(out_scale_inv)
        acc = tl.clamp(acc, FP8_MIN, FP8_MAX)

    # write result
    output_offset = (
        query_token_idx * output_stride_0
        + query_head_idx * output_stride_1
        + tl.arange(0, HEAD_SIZE_PADDED)
    )
    tl.store(output_ptr + output_offset, acc, mask=dim_mask)
