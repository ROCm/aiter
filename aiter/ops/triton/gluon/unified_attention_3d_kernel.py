# The kernels in this file are adapted from vLLM:
# https://github.com/vllm-project/vllm/blob/main/vllm/attention/ops/triton_unified_attention.py
from re import T
from shlex import join
import triton
import triton.language as tl
import torch
from aiter.ops.triton.utils.types import e4m3_dtype
from triton.experimental import gluon
import triton.experimental.gluon.language as gl
import aiter.ops.triton.utils._triton.arch_info as arch_info
from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr

# from triton._C.libtriton.gluon_ir import make_cga_layout

DEVICE_ARCH = arch_info.get_arch()
MMA_operation: gl.constexpr = (
    gl.amd.gfx1250.wmma
    if gl.constexpr(DEVICE_ARCH in ("gfx1250",))
    else gl.amd.cdna4.mfma
)

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
    BLOCK_Q: gl.constexpr,
    use_q_block_mode: gl.constexpr,
):
    left: gl.int32 = 0
    right = num_seqs
    while left < right:
        mid = (left + right) // 2
        val = gl.load(query_start_len_ptr + mid)
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
    BLOCK_Q: gl.constexpr,
):
    q_block_start_idx = gl.load(query_start_len_ptr + seq_idx) // BLOCK_Q + seq_idx

    q_block_local_idx = q_block_global_idx - q_block_start_idx

    cur_batch_in_all_start_index = gl.load(query_start_len_ptr + seq_idx)
    cur_batch_in_all_stop_index = gl.load(query_start_len_ptr + seq_idx + 1)

    cur_batch_query_len = cur_batch_in_all_stop_index - cur_batch_in_all_start_index

    return q_block_local_idx, cur_batch_query_len, cur_batch_in_all_start_index


@gluon.jit
def _get_seq_metadata(
    seq_lens_ptr,
    seq_idx,
    TILE_SIZE: gl.constexpr,
    NUM_SEGMENTS_PER_SEQ: gl.constexpr,
):
    # sequence len for this particular sequence
    seq_len = gl.load(seq_lens_ptr + seq_idx)

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
    BLOCK_M: gl.constexpr,
    HEAD_SIZE_PADDED: gl.constexpr,
    QK_WMMA_LAYOUT: gl.constexpr,
    PV_WMMA_LAYOUT: gl.constexpr,
    USE_SINKS: gl.constexpr,
):

    # M : shape = (BLOCK_M, ), layout = gl.SliceLayout(1, Q_LOAD_LAYOUT)
    if USE_SINKS:
        if segm_idx == 0:
            # Prescale with RCP_LN2, needed for exp2
            M = (
                gl.amd.cdna4.buffer_load(
                    ptr=sink_ptr,
                    offsets=query_offset_1.to(gl.int32),
                    mask=query_mask_1,
                    other=float("-inf"),
                ).to(dtype=gl.float32)
                * RCP_LN2
            )
        else:
            M = gl.full(
                [BLOCK_M],
                float("-inf"),
                dtype=tl.float32,
                layout=gl.SliceLayout(1, QK_WMMA_LAYOUT),
            )
    else:
        M = gl.full(
            [BLOCK_M],
            float("-inf"),
            dtype=tl.float32,
            layout=gl.SliceLayout(1, QK_WMMA_LAYOUT),
        )

    # L : shape = (BLOCK_M, ), layout = gl.SliceLayout(1, Q_LOAD_LAYOUT)
    L = gl.full(
        [BLOCK_M], 1.0, dtype=tl.float32, layout=gl.SliceLayout(1, QK_WMMA_LAYOUT)
    )
    # acc : shape = (BLOCK_M, HEAD_SIZE_PADDED), layout = PV_WMMA_LAYOUT
    acc = gl.zeros([BLOCK_M, HEAD_SIZE_PADDED], dtype=tl.float32, layout=PV_WMMA_LAYOUT)

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
    query_mask,
    query_pos,
    context_len,
    alibi_slope,
    qq_bias_stride_0,
    qk_scale,
    softcap,
    RCP_LN2,
    BLOCK_M: gl.constexpr,
    TILE_SIZE: gl.constexpr,
    USE_SOFTCAP: gl.constexpr,
    SLIDING_WINDOW: gl.constexpr,
    USE_ALIBI_SLOPES: gl.constexpr,
    USE_QQ_BIAS: gl.constexpr,
    Q_LOAD_LAYOUT: gl.constexpr,
    QK_WMMA_LAYOUT: gl.constexpr,
    PV_WMMA_LAYOUT: gl.constexpr,
):
    # S : shape = (BLOCK_M, TILE_SIZE), layout = QK_WMMA_LAYOUT
    S = gl.zeros([BLOCK_M, TILE_SIZE], dtype=tl.float32, layout=QK_WMMA_LAYOUT)
    # qk_scale = scale * RCP_LN2 (log_2 e) so that we can use exp2 later
    S = qk_scale * MMA_operation(Q, K, S)
    # S : shape = (BLOCK_M, TILE_SIZE), layout = Q_LOAD_LAYOUT
    # S = gl.convert_layout(S, layout=Q_LOAD_LAYOUT)

    if USE_SOFTCAP:
        # softcap here uses exp2 and consumes RCP_LN2 conversion.
        # multiply by RCP_LN2 again to be used in later exp2
        S = apply_softcap(S, softcap) * RCP_LN2

    seq_mask = seq_offset[None, :] < context_len + query_pos[:, None] + 1

    S = gl.where(query_mask & seq_mask, S, float("-inf"))

    if SLIDING_WINDOW > 0:
        S = gl.where(
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
        qq_bias = gl.load(
            qq_bias_row_ptrs + key_rel_pos[None, :],
            mask=is_query_key[None, :],  # avoid OOB for context keys
            other=0.0,
        )
        # prescale w. RCP_LN2 for later exp2
        S += qq_bias * RCP_LN2

    # compute running maximum
    # m_j : shape = (BLOCK_M, ), layout = gl.SliceLayout(1, Q_LOAD_LAYOUT)
    m_j = gl.maximum(M, gl.max(S, axis=1))

    # For sliding window there's a chance the max is -inf due to masking of
    # the entire row. In this case we need to set m_j 0 to avoid NaN
    m_j = gl.where(m_j > float("-inf"), m_j, 0.0)

    # P : shape = (BLOCK_M, TILE_SIZE), layout = Q_LOAD_LAYOUT
    P = gl.exp2(S - m_j[:, None])

    # l_j : shape = (BLOCK_M, ), layout = gl.SliceLayout(1, Q_LOAD_LAYOUT)
    l_j = gl.sum(P, axis=1)

    # alpha : shape = (BLOCK_M, ), layout = gl.SliceLayout(1, Q_LOAD_LAYOUT)
    alpha = gl.exp2(M - m_j)

    # acc : shape = (BLOCK_M, HEAD_SIZE_PADDED), layout = PV_WMMA_LAYOUT
    acc = acc * gl.convert_layout(alpha[:, None], layout=PV_WMMA_LAYOUT)

    # update constants
    # L : shape = (BLOCK_M, ), layout = gl.SliceLayout(1, Q_LOAD_LAYOUT)
    # M : shape = (BLOCK_M, ), layout = gl.SliceLayout(1, Q_LOAD_LAYOUT)
    L = L * alpha + l_j
    M = m_j

    return P, L, M, acc


@gluon.jit
def _perform_PV_wmma(
    P,
    V,
    acc,
    P_DOT_LAYOUT: gl.constexpr,
):
    P = P.to(V.dtype)
    P = gl.convert_layout(P, layout=P_DOT_LAYOUT)
    # P : shape = (BLOCK_M, TILE_SIZE), layout = P_DOT_LAYOUT
    # V : shape = (TILE_SIZE, HEAD_SIZE), layout = V_DOT_LAYOUT
    # acc : shape = (BLOCK_M, HEAD_SIZE_PADDED), layout = PV_WMMA_LAYOUT
    acc = MMA_operation(P, V, acc)
    return acc


@gluon.jit
def _tdm_async_gather_load_to_lds(
    j,
    desc,
    src_row_indices,
    src_col_offset,
    dst,
    num_stages: gl.constexpr,
):
    gl.amd.gfx1250.tdm.async_gather(
        desc=desc,
        src_row_indices=src_row_indices,
        src_col_offset=0,
        dst=dst.index(j % num_stages),
    )

    return j + 1


@gluon.jit
def _tdm_gather_request_from_lds(
    j,
    kv_scale,
    Q_dtype,
    smem,
    asycn_wait: gl.constexpr,
    layout: gl.constexpr,
    transpose: gl.constexpr,
    num_ctas: gl.constexpr,
    num_stages: gl.constexpr,
    TILE_SIZE: gl.constexpr,
    HEAD_SIZE_PADDED: gl.constexpr,
):
    if num_ctas > 1:
        gl.amd.gfx1250.cluster.arrive()
    gl.amd.gfx1250.tdm.async_wait(asycn_wait)
    if num_ctas > 1:
        gl.amd.gfx1250.cluster.wait()
    if transpose:
        X = (
            smem.index(j % num_stages)
            .reshape([TILE_SIZE, HEAD_SIZE_PADDED])
            .permute([1, 0])
            .load(layout=layout)
        )
    else:
        X = (
            smem.index(j % num_stages)
            .reshape([TILE_SIZE, HEAD_SIZE_PADDED])
            .load(layout=layout)
        )

    if X.dtype.is_fp8() and not Q_dtype.is_fp8():
        X = (X.to(gl.float32) * gl.load(kv_scale)).to(Q_dtype)

    return j + 1, X


@gluon.jit
def _tdm_gather_get_kv_offsets(
    j,
    offs_j,
    kv_head_idx,
    block_tables_sorted_ptr,
    block_table_offset,
    stride_k_cache_h: gl.int64,
    stride_v_cache_h: gl.int64,
    NUM_BLOCKS_GATHER_PER_TILE: gl.constexpr,
):
    physical_block_idx = gl.load(
        block_tables_sorted_ptr
        + block_table_offset
        + j * NUM_BLOCKS_GATHER_PER_TILE
        + offs_j
    )

    offs_k_gather_idx = (physical_block_idx * stride_k_cache_h + kv_head_idx).to(
        tl.int32
    )
    offs_v_gather_idx = (physical_block_idx * stride_v_cache_h + kv_head_idx).to(
        tl.int32
    )

    return j + 1, offs_k_gather_idx, offs_v_gather_idx


@gluon.jit
def _tdm_gather_create_tensor_descriptors_and_allocate_lds(
    q_ptr,
    k_ptr,
    v_ptr,
    NUM_BLOCKS,
    stride_q_m: gl.int64,  # int
    stride_q_d: gl.constexpr,  # int
    stride_k_t: gl.int64,  # int
    stride_k_d: gl.constexpr,  # int
    stride_v_t: gl.int64,  # int
    stride_v_d: gl.constexpr,  # int
    q_shared_layout: gl.constexpr,
    k_shared_layout: gl.constexpr,
    v_shared_layout: gl.constexpr,
    NUM_KV_HEADS: gl.constexpr,
    BLOCK_M: gl.constexpr,
    HEAD_SIZE: gl.constexpr,
    BLOCK_SIZE: gl.constexpr,
    TILE_SIZE: gl.constexpr,
    HEAD_SIZE_PADDED: gl.constexpr,
    NUM_BLOCKS_GATHER_PER_TILE: gl.constexpr,
    num_stages: gl.constexpr,
):
    gl.static_assert(stride_k_d == 1, "stride_k_d must be 1")
    gl.static_assert(stride_v_d == 1, "stride_v_d must be 1")
    # gl.static_assert(stride_k_t == BLOCK_SIZE * HEAD_SIZE, "stride_k_t must be BLOCK_SIZE * HEAD_SIZE")
    # gl.static_assert(stride_v_t == BLOCK_SIZE * HEAD_SIZE, "stride_v_t must be BLOCK_SIZE * HEAD_SIZE")

    k_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=k_ptr,
        shape=(NUM_BLOCKS * NUM_KV_HEADS, BLOCK_SIZE * HEAD_SIZE),
        strides=(stride_k_t, stride_k_d),
        block_shape=(NUM_BLOCKS_GATHER_PER_TILE, BLOCK_SIZE * HEAD_SIZE_PADDED),
        layout=k_shared_layout,
    )

    v_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=v_ptr,
        shape=(NUM_BLOCKS * NUM_KV_HEADS, BLOCK_SIZE * HEAD_SIZE),
        strides=(stride_v_t, stride_v_d),
        block_shape=(NUM_BLOCKS_GATHER_PER_TILE, BLOCK_SIZE * HEAD_SIZE_PADDED),
        layout=v_shared_layout,
    )

    smem_Q = gl.allocate_shared_memory(
        q_ptr.type.element_ty,
        shape=[BLOCK_M, HEAD_SIZE_PADDED],
        layout=q_shared_layout,
    )
    smem_K = gl.allocate_shared_memory(
        k_desc.dtype,
        shape=[num_stages] + k_desc.block_shape,
        layout=k_desc.layout,
    )
    smem_V = gl.allocate_shared_memory(
        v_desc.dtype,
        shape=[num_stages] + v_desc.block_shape,
        layout=v_desc.layout,
    )

    return k_desc, v_desc, smem_Q, smem_K, smem_V


gluon_kernel_unified_attention_3d_tdm_gather_repr = make_kernel_repr(
    "gluon_kernel_unified_attention_3d_tdm_gather",
    [
        "num_query_heads",
        "num_queries_per_kv",
        "BLOCK_SIZE",
        "TILE_SIZE",
        "HEAD_SIZE",
        "num_warps",
        "num_stages",
        "cache_modifier",
    ],
)


@gluon.jit(repr=gluon_kernel_unified_attention_3d_tdm_gather_repr)
def gluon_kernel_unified_attention_3d_tdm_gather(
    segm_output_ptr,
    # [num_tokens, num_query_heads, num_segments, head_size]
    segm_max_ptr,  # [num_tokens, num_query_heads, num_segments]
    segm_expsum_ptr,  # [num_tokens, num_query_heads, num_segments]
    query_ptr,  # [num_tokens, num_query_heads, head_size]
    key_cache_ptr,  # [num_blks, num_kv_heads, blk_size, head_size]
    value_cache_ptr,  # [num_blks, num_kv_heads, blk_size, head_size]
    sink_ptr,  # [num_query_heads]
    block_tables_ptr,  # [num_seqs, max_num_blocks_per_seq]
    # block_table_sorted_indices_ptr,  # [num_seqs, max_num_blocks_per_seq]
    seq_lens_ptr,  # [num_seqs]
    alibi_slopes_ptr,  # [num_query_heads]
    qq_bias_ptr,  # [num_query_tokens, num_query_tokens]
    scale,  # float32
    k_scale,  # float32
    v_scale,  # float32
    softcap,  # float32
    num_tokens,  # int
    NUM_BLOCKS,  # int
    num_query_heads: gl.constexpr,  # int
    num_queries_per_kv: gl.constexpr,  # int
    block_table_stride: gl.int64,  # int
    # block_table_sorted_indices_stride: gl.int64,  # int
    query_stride_0: gl.int64,  # int
    query_stride_1: gl.int64,  # int, should be equal to head_size
    qq_bias_stride_0: gl.int64,  # int
    BLOCK_SIZE: gl.constexpr,  # int
    TILE_SIZE: gl.constexpr,  # int, must be power of 2
    HEAD_SIZE: gl.constexpr,  # int
    HEAD_SIZE_PADDED: gl.constexpr,  # int, must be power of 2
    USE_ALIBI_SLOPES: gl.constexpr,  # bool
    USE_QQ_BIAS: gl.constexpr,  # bool
    USE_SOFTCAP: gl.constexpr,  # bool
    USE_SINKS: gl.constexpr,  # bool
    SLIDING_WINDOW: gl.constexpr,  # int
    stride_k_cache_0: gl.int64,  # int
    stride_k_cache_1: gl.int64,  # int
    stride_k_cache_2: gl.int64,  # int
    stride_k_cache_3: gl.constexpr,  # int
    stride_v_cache_0: gl.int64,  # int
    stride_v_cache_1: gl.int64,  # int
    stride_v_cache_2: gl.int64,  # int
    stride_v_cache_3: gl.constexpr,  # int
    query_start_len_ptr,  # [num_seqs+1]
    BLOCK_Q: gl.constexpr,  # int
    num_seqs: gl.int32,
    BLOCK_M: gl.constexpr,  # int
    NUM_SEGMENTS_PER_SEQ: gl.constexpr,  # int
    num_warps: gl.constexpr,  # int
    num_stages: gl.constexpr,  # int
    QK_WMMA_LAYOUT: gl.constexpr,
    PV_WMMA_LAYOUT: gl.constexpr,
    Q_DOT_LAYOUT: gl.constexpr,
    K_DOT_LAYOUT: gl.constexpr,
    P_DOT_LAYOUT: gl.constexpr,
    V_DOT_LAYOUT: gl.constexpr,
    Q_SHARED_LAYOUT: gl.constexpr,
    K_SHARED_LAYOUT: gl.constexpr,
    V_SHARED_LAYOUT: gl.constexpr,
    Q_LOAD_LAYOUT: gl.constexpr,
    K_LOAD_LAYOUT: gl.constexpr,
    V_LOAD_LAYOUT: gl.constexpr,
    num_ctas: gl.constexpr = 1,  # int
    ALL_DECODE: gl.constexpr = False,  # bool
    SHUFFLED_KV_CACHE: gl.constexpr = False,  # bool
):
    q_block_global_idx = gl.program_id(0)
    kv_head_idx = gl.program_id(1)
    segm_idx = gl.program_id(2)
    # num_ctas: gl.constexpr = gl.num_ctas()
    pred = 1
    pred_i32 = pred.to(gl.int32) if hasattr(pred, "to") else pred

    gl.static_assert(
        TILE_SIZE % BLOCK_SIZE == 0, "TILE_SIZE must be multiple of BLOCK_SIZE"
    )
    NUM_BLOCKS_GATHER_PER_TILE: gl.constexpr = TILE_SIZE // BLOCK_SIZE
    gl.static_assert(
        NUM_BLOCKS_GATHER_PER_TILE == 4 or NUM_BLOCKS_GATHER_PER_TILE == 8,
        "NUM_BLOCKS_GATHER_PER_TILE must be 4 or 8",
    )

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

    offs_q_m = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, Q_LOAD_LAYOUT))
    offs_q_d = gl.arange(0, HEAD_SIZE_PADDED, layout=gl.SliceLayout(0, Q_LOAD_LAYOUT))

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
        dim_mask = gl.full((1,), 1, dtype=tl.int1)

    query_mask_0 = query_pos < cur_batch_query_len
    query_mask_1 = query_offset_1 < num_query_heads

    NUM_KV_HEADS: gl.constexpr = num_query_heads // num_queries_per_kv

    k_desc, v_desc, smem_Q, smem_K, smem_V = (
        _tdm_gather_create_tensor_descriptors_and_allocate_lds(
            query_ptr,
            key_cache_ptr,
            value_cache_ptr,
            NUM_BLOCKS,
            query_stride_1,
            1,
            stride_k_cache_1,  # stride_k_cache_1 = BLOCK_SIZE * HEAD_SIZE
            stride_k_cache_3,
            stride_v_cache_1,  # stride_v_cache_1 = BLOCK_SIZE * HEAD_SIZE
            stride_v_cache_3,
            Q_SHARED_LAYOUT,
            K_SHARED_LAYOUT,
            V_SHARED_LAYOUT,
            NUM_KV_HEADS,
            BLOCK_M,
            HEAD_SIZE,
            BLOCK_SIZE,
            TILE_SIZE,
            HEAD_SIZE_PADDED,
            NUM_BLOCKS_GATHER_PER_TILE,
            num_stages=num_stages,
        )
    )

    # Q_load : shape = (BLOCK_M, HEAD_SIZE_PADDED), layout = Q_LOAD_LAYOUT
    Q_load = gl.amd.cdna4.buffer_load(
        ptr=query_ptr,
        offsets=query_offset.to(gl.int32),
        mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
        other=0.0,
    )
    smem_Q.store(Q_load)
    # Q : shape = (BLOCK_M, HEAD_SIZE_PADDED), layout = Q_DOT_LAYOUT
    Q = smem_Q.load(layout=Q_DOT_LAYOUT)

    offs_q_m_qk = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, QK_WMMA_LAYOUT))
    query_pos_qk = q_block_local_idx * BLOCK_Q + offs_q_m_qk // num_queries_per_kv
    query_offset_1_qk = (
        kv_head_idx * num_queries_per_kv + offs_q_m_qk % num_queries_per_kv
    )
    query_mask_0_qk = query_pos_qk < cur_batch_query_len
    query_mask_1_qk = query_offset_1_qk < num_query_heads
    query_mask_qk = query_mask_1_qk[:, None] & query_mask_0_qk[:, None]

    L, M, acc = _allocate_L_M_acc(
        sink_ptr,
        segm_idx,
        query_offset_1_qk,
        query_mask_1_qk,
        RCP_LN2,
        BLOCK_M,
        HEAD_SIZE_PADDED,
        QK_WMMA_LAYOUT,
        PV_WMMA_LAYOUT,
        USE_SINKS,
    )

    # alibi slope for this head
    alibi_slope = None
    if USE_ALIBI_SLOPES:
        alibi_slope = tl.load(
            alibi_slopes_ptr + query_offset_1_qk, mask=query_mask_1, other=0.0
        )

    # query-query attention bias
    qq_bias_row_ptrs = None
    if USE_QQ_BIAS:
        qq_bias_row_ptrs = (
            qq_bias_ptr + query_pos_qk[:, None] * qq_bias_stride_0
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
    max_seq_prefix_len = gl.minimum(max_seq_prefix_len, seq_len)

    # calculate the number of tiles that need to be processed to
    # cover the longest sequence prefix (due to causal masking, tiles beyond
    # this prefix can be skipped)
    num_tiles = cdiv_fn(max_seq_prefix_len, TILE_SIZE)

    # KV_cache_modifier: gl.constexpr = ".cg" if ALL_DECODE else ""

    k_from_hbm = 0
    k_from_lds = 0
    v_from_hbm = 0
    v_from_lds = 0
    GATHER_LOAD_LAYOUT: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[NUM_BLOCKS_GATHER_PER_TILE],
        threads_per_warp=[32],
        warps_per_cta=[num_warps],
        order=[0],
    )
    offs_j = gl.arange(0, NUM_BLOCKS_GATHER_PER_TILE, layout=GATHER_LOAD_LAYOUT)
    j_from_hbm = segm_idx * tiles_per_segment
    j_from_lds = segm_idx * tiles_per_segment
    seq_offset = j_from_lds * TILE_SIZE + gl.arange(
        0, TILE_SIZE, layout=gl.SliceLayout(0, QK_WMMA_LAYOUT)
    )

    for _ in range(num_stages - 1):
        j_from_hbm, offs_k_gather_idx, offs_v_gather_idx = _tdm_gather_get_kv_offsets(
            j_from_hbm,
            offs_j,
            kv_head_idx,
            block_tables_ptr,
            block_table_offset,
            stride_k_cache_0 // stride_k_cache_1,  # = NUM_KV_HEADS
            stride_v_cache_0 // stride_v_cache_1,  # = NUM_KV_HEADS
            NUM_BLOCKS_GATHER_PER_TILE,
        )
        k_from_hbm = _tdm_async_gather_load_to_lds(
            k_from_hbm,
            desc=k_desc,
            src_row_indices=offs_k_gather_idx,
            src_col_offset=0,
            dst=smem_K,
            num_stages=num_stages,
        )
        v_from_hbm = _tdm_async_gather_load_to_lds(
            v_from_hbm,
            desc=v_desc,
            src_row_indices=offs_v_gather_idx,
            src_col_offset=0,
            dst=smem_V,
            num_stages=num_stages,
        )

    # iterate through tiles within current segment
    # for _ in range(tiles_per_segment - (num_stages - 1)):
    for _ in range(
        segm_idx * tiles_per_segment,
        min((segm_idx + 1) * tiles_per_segment, num_tiles) - (num_stages - 1),
    ):
        j_from_hbm, offs_k_gather_idx, offs_v_gather_idx = _tdm_gather_get_kv_offsets(
            j_from_hbm,
            offs_j,
            kv_head_idx,
            block_tables_ptr,
            block_table_offset,
            stride_k_cache_0 // stride_k_cache_1,  # = NUM_KV_HEADS
            stride_v_cache_0 // stride_v_cache_1,  # = NUM_KV_HEADS
            NUM_BLOCKS_GATHER_PER_TILE,
        )
        k_from_hbm = _tdm_async_gather_load_to_lds(
            k_from_hbm,
            desc=k_desc,
            src_row_indices=offs_k_gather_idx,
            src_col_offset=0,
            dst=smem_K,
            num_stages=num_stages,
        )
        v_from_hbm = _tdm_async_gather_load_to_lds(
            v_from_hbm,
            desc=v_desc,
            src_row_indices=offs_v_gather_idx,
            src_col_offset=0,
            dst=smem_V,
            num_stages=num_stages,
        )

        # K : shape = (HEAD_SIZE_PADDED, TILE_SIZE), layout = K_DOT_LAYOUT
        k_from_lds, K = _tdm_gather_request_from_lds(
            k_from_lds,
            k_scale,
            Q.dtype,
            smem_K,
            asycn_wait=(num_stages - 1) * 2 + 1,
            layout=K_DOT_LAYOUT,
            transpose=True,
            num_ctas=num_ctas,
            num_stages=num_stages,
            TILE_SIZE=TILE_SIZE,
            HEAD_SIZE_PADDED=HEAD_SIZE_PADDED,
        )

        # P : shape = (BLOCK_M, TILE_SIZE), layout = Q_LOAD_LAYOUT
        # L : shape = (BLOCK_M, ), layout = gl.SliceLayout(1, Q_LOAD_LAYOUT)
        # M : shape = (BLOCK_M, ), layout = gl.SliceLayout(1, Q_LOAD_LAYOUT)
        # acc : shape = (BLOCK_M, HEAD_SIZE_PADDED), layout = PV_WMMA_LAYOUT
        P, L, M, acc = _perform_QK_wmma_and_update_L_M(
            Q,
            K,
            L,
            M,
            acc,
            qq_bias_row_ptrs,
            seq_offset,
            query_mask_qk,
            query_pos_qk,
            context_len,
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
            Q_LOAD_LAYOUT,
            QK_WMMA_LAYOUT,
            PV_WMMA_LAYOUT,
        )

        # V : shape = (TILE_SIZE, HEAD_SIZE_PADDED), layout = V_DOT_LAYOUT
        v_from_lds, V = _tdm_gather_request_from_lds(
            v_from_lds,
            v_scale,
            Q.dtype,
            smem_V,
            asycn_wait=(num_stages - 1) * 2,
            layout=V_DOT_LAYOUT,
            transpose=False,
            num_ctas=num_ctas,
            num_stages=num_stages,
            TILE_SIZE=TILE_SIZE,
            HEAD_SIZE_PADDED=HEAD_SIZE_PADDED,
        )

        # acc : shape = (BLOCK_M, HEAD_SIZE_PADDED), layout = PV_WMMA_LAYOUT
        acc = _perform_PV_wmma(P, V, acc, P_DOT_LAYOUT)

        j_from_lds = j_from_lds + 1
        seq_offset += TILE_SIZE

    for _ in range(num_stages - 1):
        # K : shape = (HEAD_SIZE_PADDED, TILE_SIZE), layout = K_DOT_LAYOUT
        k_from_lds, K = _tdm_gather_request_from_lds(
            k_from_lds,
            k_scale,
            Q.dtype,
            smem_K,
            asycn_wait=(num_stages - 2) * 2 + 1,
            layout=K_DOT_LAYOUT,
            transpose=True,
            num_ctas=num_ctas,
            num_stages=num_stages,
            TILE_SIZE=TILE_SIZE,
            HEAD_SIZE_PADDED=HEAD_SIZE_PADDED,
        )

        # P : shape = (BLOCK_M, TILE_SIZE), layout = Q_LOAD_LAYOUT
        # L : shape = (BLOCK_M, ), layout = gl.SliceLayout(1, Q_LOAD_LAYOUT)
        # M : shape = (BLOCK_M, ), layout = gl.SliceLayout(1, Q_LOAD_LAYOUT)
        # acc : shape = (BLOCK_M, HEAD_SIZE_PADDED), layout = PV_WMMA_LAYOUT
        P, L, M, acc = _perform_QK_wmma_and_update_L_M(
            Q,
            K,
            L,
            M,
            acc,
            qq_bias_row_ptrs,
            seq_offset,
            query_mask_qk,
            query_pos_qk,
            context_len,
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
            Q_LOAD_LAYOUT,
            QK_WMMA_LAYOUT,
            PV_WMMA_LAYOUT,
        )

        # V : shape = (TILE_SIZE, HEAD_SIZE_PADDED), layout = V_DOT_LAYOUT
        v_from_lds, V = _tdm_gather_request_from_lds(
            v_from_lds,
            v_scale,
            Q.dtype,
            smem_V,
            asycn_wait=(num_stages - 2) * 2,
            layout=V_DOT_LAYOUT,
            transpose=False,
            num_ctas=num_ctas,
            num_stages=num_stages,
            TILE_SIZE=TILE_SIZE,
            HEAD_SIZE_PADDED=HEAD_SIZE_PADDED,
        )

        # acc : shape = (BLOCK_M, HEAD_SIZE_PADDED), layout = PV_WMMA_LAYOUT
        acc = _perform_PV_wmma(P, V, acc, P_DOT_LAYOUT)

        j_from_lds = j_from_lds + 1
        seq_offset += TILE_SIZE

    # store segm_output
    # acc : shape = (BLOCK_M, HEAD_SIZE_PADDED), layout = Q_LOAD_LAYOUT
    acc = gl.convert_layout(acc, layout=Q_LOAD_LAYOUT)
    segm_output_offset = (
        query_offset_0[:, None]
        * (num_query_heads * NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED)
        + query_offset_1[:, None] * (NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED)
        + segm_idx * HEAD_SIZE_PADDED
        + offs_q_d[None, :]
    )
    gl.amd.cdna4.buffer_store(
        stored_value=acc,
        ptr=segm_output_ptr,
        offsets=segm_output_offset,
        mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
    )

    # store segm_max and segm_expsum
    # L : shape = (BLOCK_M, ), layout = gl.SliceLayout(1, QK_WMMA_LAYOUT)
    # M : shape = (BLOCK_M, ), layout = gl.SliceLayout(1, QK_WMMA_LAYOUT)
    segm_offset = (
        query_offset_0 * (num_query_heads * NUM_SEGMENTS_PER_SEQ)
        + query_offset_1 * NUM_SEGMENTS_PER_SEQ
        + segm_idx
    )
    L = gl.convert_layout(L, layout=gl.SliceLayout(1, Q_LOAD_LAYOUT))
    M = gl.convert_layout(M, layout=gl.SliceLayout(1, Q_LOAD_LAYOUT))
    gl.amd.cdna4.buffer_store(
        stored_value=M,
        ptr=segm_max_ptr,
        offsets=segm_offset,
        mask=query_mask_0 & query_mask_1,
    )
    gl.amd.cdna4.buffer_store(
        stored_value=L,
        ptr=segm_expsum_ptr,
        offsets=segm_offset,
        mask=query_mask_0 & query_mask_1,
    )


@gluon.jit
def _tdm_get_kv_offsets(
    j,
    kv_head_idx,
    block_tables_ptr,
    block_table_offset,
    stride_k_cache_t: gl.int64,
    stride_k_cache_d: gl.int64,
    stride_v_cache_t: gl.int64,
    stride_v_cache_d: gl.int64,
):
    physical_block_idx = gl.load(block_tables_ptr + block_table_offset + j)

    offs_k_t = (physical_block_idx * stride_k_cache_t).to(tl.int32)
    offs_k_d = (kv_head_idx * stride_k_cache_d).to(tl.int32)

    offs_v_t = (physical_block_idx * stride_v_cache_t).to(tl.int32)
    offs_v_d = (kv_head_idx * stride_v_cache_d).to(tl.int32)

    return j + 1, offs_k_t, offs_k_d, offs_v_t, offs_v_d


@gluon.jit
def _tdm_async_load_to_lds(
    j,
    src,
    offsets,
    dest,
    pred_i32,
    num_stages: gl.constexpr,
):
    # gl.amd.gfx1250.tdm.prefetch(
    #     src=k_desc,
    #     offsets=[
    #         0,
    #         offs_kv_t_starts,
    #     ],
    #     pred=pred.to(gl.int1)
    # )
    gl.amd.gfx1250.tdm.async_load(
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
    asycn_wait: gl.constexpr,
    layout: gl.constexpr,
    transpose: gl.constexpr,
    num_ctas: gl.constexpr,
    num_stages: gl.constexpr,
):
    if num_ctas > 1:
        gl.amd.gfx1250.cluster.arrive()
    gl.amd.gfx1250.tdm.async_wait(asycn_wait)
    if num_ctas > 1:
        gl.amd.gfx1250.cluster.wait()
    if transpose:
        X = smem.index(j % num_stages).permute([1, 0]).load(layout=layout)
    else:
        X = smem.index(j % num_stages).load(layout=layout)

    if X.dtype.is_fp8() and not Q_dtype.is_fp8():
        X = (X.to(gl.float32) * gl.load(kv_scale)).to(Q_dtype)

    return j + 1, X


@gluon.jit
def _tdm_create_tensor_descriptors_and_allocate_lds(
    q_ptr,
    k_ptr,
    v_ptr,
    NUM_BLOCKS,
    stride_q_m: gl.int64,  # int
    stride_q_d: gl.constexpr,  # int
    stride_k_t: gl.int64,  # int
    stride_k_d: gl.constexpr,  # int
    stride_v_t: gl.int64,  # int
    stride_v_d: gl.constexpr,  # int
    q_shared_layout: gl.constexpr,
    k_shared_layout: gl.constexpr,
    v_shared_layout: gl.constexpr,
    NUM_KV_HEADS: gl.constexpr,
    BLOCK_M: gl.constexpr,
    HEAD_SIZE: gl.constexpr,
    TILE_SIZE: gl.constexpr,
    HEAD_SIZE_PADDED: gl.constexpr,
    num_stages: gl.constexpr,
):
    gl.static_assert(stride_q_d == 1, "stride_q_d must be 1")
    gl.static_assert(stride_k_d == 1, "stride_k_d must be 1")
    gl.static_assert(stride_v_d == 1, "stride_v_d must be 1")
    # q_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
    #     base=q_ptr,
    #     shape=(M, HEAD_SIZE),
    #     strides=(stride_q_m, stride_q_d),
    #     block_shape=(BLOCK_M, HEAD_SIZE_PADDED),
    #     layout=q_shared_layout,
    # )

    k_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=k_ptr,
        shape=(NUM_BLOCKS * TILE_SIZE, NUM_KV_HEADS * HEAD_SIZE),
        strides=(stride_k_t, stride_k_d),
        block_shape=(TILE_SIZE, HEAD_SIZE_PADDED),
        layout=k_shared_layout,
    )

    v_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=v_ptr,
        shape=(NUM_BLOCKS * TILE_SIZE, NUM_KV_HEADS * HEAD_SIZE),
        strides=(stride_v_t, stride_v_d),
        block_shape=(TILE_SIZE, HEAD_SIZE_PADDED),
        layout=v_shared_layout,
    )

    # smem_Q = gl.allocate_shared_memory(
    #     q_desc.dtype,
    #     shape=q_desc.block_shape,
    #     layout=q_shared_layout,
    # )
    smem_Q = gl.allocate_shared_memory(
        q_ptr.type.element_ty,
        shape=[BLOCK_M, HEAD_SIZE_PADDED],
        layout=q_shared_layout,
    )
    smem_K = gl.allocate_shared_memory(
        k_desc.dtype,
        shape=[num_stages] + k_desc.block_shape,
        layout=k_desc.layout,
    )
    smem_V = gl.allocate_shared_memory(
        v_desc.dtype,
        shape=[num_stages] + v_desc.block_shape,
        layout=v_desc.layout,
    )

    return k_desc, v_desc, smem_Q, smem_K, smem_V


gluon_kernel_unified_attention_3d_tdm_repr = make_kernel_repr(
    "gluon_kernel_unified_attention_3d_tdm",
    [
        "num_query_heads",
        "num_queries_per_kv",
        "BLOCK_SIZE",
        "TILE_SIZE",
        "HEAD_SIZE",
        "num_warps",
        "num_stages",
        "cache_modifier",
    ],
)


@gluon.jit(repr=gluon_kernel_unified_attention_3d_tdm_repr)
def gluon_kernel_unified_attention_3d_tdm(
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
    NUM_BLOCKS,  # int
    num_query_heads: gl.constexpr,  # int
    num_queries_per_kv: gl.constexpr,  # int
    block_table_stride: gl.int64,  # int
    query_stride_0: gl.int64,  # int
    query_stride_1: gl.int64,  # int, should be equal to head_size
    qq_bias_stride_0: gl.int64,  # int
    BLOCK_SIZE: gl.constexpr,  # int
    TILE_SIZE: gl.constexpr,  # int, must be power of 2
    HEAD_SIZE: gl.constexpr,  # int
    HEAD_SIZE_PADDED: gl.constexpr,  # int, must be power of 2
    USE_ALIBI_SLOPES: gl.constexpr,  # bool
    USE_QQ_BIAS: gl.constexpr,  # bool
    USE_SOFTCAP: gl.constexpr,  # bool
    USE_SINKS: gl.constexpr,  # bool
    SLIDING_WINDOW: gl.constexpr,  # int
    stride_k_cache_0: gl.int64,  # int
    stride_k_cache_1: gl.int64,  # int
    stride_k_cache_2: gl.int64,  # int
    stride_k_cache_3: gl.constexpr,  # int
    stride_v_cache_0: gl.int64,  # int
    stride_v_cache_1: gl.int64,  # int
    stride_v_cache_2: gl.int64,  # int
    stride_v_cache_3: gl.constexpr,  # int
    query_start_len_ptr,  # [num_seqs+1]
    BLOCK_Q: gl.constexpr,  # int
    num_seqs: gl.int32,
    BLOCK_M: gl.constexpr,  # int
    NUM_SEGMENTS_PER_SEQ: gl.constexpr,  # int
    num_warps: gl.constexpr,  # int
    num_stages: gl.constexpr,  # int
    QK_WMMA_LAYOUT: gl.constexpr,
    PV_WMMA_LAYOUT: gl.constexpr,
    Q_DOT_LAYOUT: gl.constexpr,
    K_DOT_LAYOUT: gl.constexpr,
    P_DOT_LAYOUT: gl.constexpr,
    V_DOT_LAYOUT: gl.constexpr,
    Q_SHARED_LAYOUT: gl.constexpr,
    K_SHARED_LAYOUT: gl.constexpr,
    V_SHARED_LAYOUT: gl.constexpr,
    Q_LOAD_LAYOUT: gl.constexpr,
    K_LOAD_LAYOUT: gl.constexpr,
    V_LOAD_LAYOUT: gl.constexpr,
    num_ctas: gl.constexpr = 1,  # int
    ALL_DECODE: gl.constexpr = False,  # bool
    SHUFFLED_KV_CACHE: gl.constexpr = False,  # bool
):
    q_block_global_idx = gl.program_id(0)
    kv_head_idx = gl.program_id(1)
    segm_idx = gl.program_id(2)
    # num_ctas: gl.constexpr = gl.num_ctas()
    pred = 1
    pred_i32 = pred.to(gl.int32) if hasattr(pred, "to") else pred

    gl.static_assert(
        TILE_SIZE == BLOCK_SIZE, "TILE_SIZE must be the same as BLOCK_SIZE"
    )

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

    offs_q_m = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, Q_LOAD_LAYOUT))
    offs_q_d = gl.arange(0, HEAD_SIZE_PADDED, layout=gl.SliceLayout(0, Q_LOAD_LAYOUT))

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
        dim_mask = gl.full((1,), 1, dtype=tl.int1)

    query_mask_0 = query_pos < cur_batch_query_len
    query_mask_1 = query_offset_1 < num_query_heads

    NUM_KV_HEADS: gl.constexpr = num_query_heads // num_queries_per_kv

    k_desc, v_desc, smem_Q, smem_K, smem_V = (
        _tdm_create_tensor_descriptors_and_allocate_lds(
            query_ptr,
            key_cache_ptr,
            value_cache_ptr,
            NUM_BLOCKS,
            query_stride_1,
            1,
            stride_k_cache_1,  # stride_k_cache_1 = HEAD_SIZE * num_kv_heads
            stride_k_cache_3,
            stride_v_cache_1,
            stride_v_cache_3,
            Q_SHARED_LAYOUT,
            K_SHARED_LAYOUT,
            V_SHARED_LAYOUT,
            NUM_KV_HEADS,
            BLOCK_M,
            HEAD_SIZE,
            TILE_SIZE,
            HEAD_SIZE_PADDED,
            num_stages=num_stages,
        )
    )

    # Q_load : shape = (BLOCK_M, HEAD_SIZE_PADDED), layout = Q_LOAD_LAYOUT
    Q_load = gl.amd.cdna4.buffer_load(
        ptr=query_ptr,
        offsets=query_offset.to(gl.int32),
        mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
        other=0.0,
    )
    smem_Q.store(Q_load)
    # Q : shape = (BLOCK_M, HEAD_SIZE_PADDED), layout = Q_DOT_LAYOUT
    Q = smem_Q.load(layout=Q_DOT_LAYOUT)

    offs_q_m_qk = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, QK_WMMA_LAYOUT))
    query_pos_qk = q_block_local_idx * BLOCK_Q + offs_q_m_qk // num_queries_per_kv
    query_offset_1_qk = (
        kv_head_idx * num_queries_per_kv + offs_q_m_qk % num_queries_per_kv
    )
    query_mask_0_qk = query_pos_qk < cur_batch_query_len
    query_mask_1_qk = query_offset_1_qk < num_query_heads
    query_mask_qk = query_mask_1_qk[:, None] & query_mask_0_qk[:, None]

    L, M, acc = _allocate_L_M_acc(
        sink_ptr,
        segm_idx,
        query_offset_1_qk,
        query_mask_1_qk,
        RCP_LN2,
        BLOCK_M,
        HEAD_SIZE_PADDED,
        QK_WMMA_LAYOUT,
        PV_WMMA_LAYOUT,
        USE_SINKS,
    )

    # alibi slope for this head
    alibi_slope = None
    if USE_ALIBI_SLOPES:
        alibi_slope = tl.load(
            alibi_slopes_ptr + query_offset_1_qk, mask=query_mask_1, other=0.0
        )

    # query-query attention bias
    qq_bias_row_ptrs = None
    if USE_QQ_BIAS:
        qq_bias_row_ptrs = (
            qq_bias_ptr + query_pos_qk[:, None] * qq_bias_stride_0
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
    max_seq_prefix_len = gl.minimum(max_seq_prefix_len, seq_len)

    # calculate the number of tiles that need to be processed to
    # cover the longest sequence prefix (due to causal masking, tiles beyond
    # this prefix can be skipped)
    num_tiles = cdiv_fn(max_seq_prefix_len, TILE_SIZE)

    # KV_cache_modifier: gl.constexpr = ".cg" if ALL_DECODE else ""

    k_from_hbm = 0
    k_from_lds = 0
    v_from_hbm = 0
    v_from_lds = 0
    j_from_hbm = segm_idx * tiles_per_segment
    j_from_lds = segm_idx * tiles_per_segment
    seq_offset = j_from_lds * TILE_SIZE + gl.arange(
        0, TILE_SIZE, layout=gl.SliceLayout(0, QK_WMMA_LAYOUT)
    )

    for _ in range(num_stages - 1):
        j_from_hbm, offs_k_t, offs_k_d, offs_v_t, offs_v_d = _tdm_get_kv_offsets(
            j_from_hbm,
            kv_head_idx,
            block_tables_ptr,
            block_table_offset,
            stride_k_cache_0 // stride_k_cache_1,  # = BLOCK_SIZE
            stride_k_cache_2,
            stride_v_cache_0 // stride_v_cache_1,  # = BLOCK_SIZE
            stride_v_cache_2,
        )
        k_from_hbm = _tdm_async_load_to_lds(
            k_from_hbm,
            src=k_desc,
            offsets=[offs_k_t, offs_k_d],
            dest=smem_K,
            pred_i32=pred_i32,
            num_stages=num_stages,
        )
        v_from_hbm = _tdm_async_load_to_lds(
            v_from_hbm,
            src=v_desc,
            offsets=[offs_v_t, offs_v_d],
            dest=smem_V,
            pred_i32=pred_i32,
            num_stages=num_stages,
        )

    # iterate through tiles within current segment
    # for _ in range(tiles_per_segment - (num_stages - 1)):
    for _ in range(
        segm_idx * tiles_per_segment,
        min((segm_idx + 1) * tiles_per_segment, num_tiles) - (num_stages - 1),
    ):
        j_from_hbm, offs_k_t, offs_k_d, offs_v_t, offs_v_d = _tdm_get_kv_offsets(
            j_from_hbm,
            kv_head_idx,
            block_tables_ptr,
            block_table_offset,
            stride_k_cache_0 // stride_k_cache_1,  # = BLOCK_SIZE
            stride_k_cache_2,
            stride_v_cache_0 // stride_v_cache_1,  # = BLOCK_SIZE
            stride_v_cache_2,
        )
        k_from_hbm = _tdm_async_load_to_lds(
            k_from_hbm,
            src=k_desc,
            offsets=[offs_k_t, offs_k_d],
            dest=smem_K,
            pred_i32=pred_i32,
            num_stages=num_stages,
        )
        v_from_hbm = _tdm_async_load_to_lds(
            v_from_hbm,
            src=v_desc,
            offsets=[offs_v_t, offs_v_d],
            dest=smem_V,
            pred_i32=pred_i32,
            num_stages=num_stages,
        )

        # K : shape = (HEAD_SIZE_PADDED, TILE_SIZE), layout = K_DOT_LAYOUT
        k_from_lds, K = _tdm_request_from_lds(
            k_from_lds,
            k_scale,
            Q.dtype,
            smem_K,
            asycn_wait=(num_stages - 1) * 2 + 1,
            layout=K_DOT_LAYOUT,
            transpose=True,
            num_ctas=num_ctas,
            num_stages=num_stages,
        )

        # P : shape = (BLOCK_M, TILE_SIZE), layout = Q_LOAD_LAYOUT
        # L : shape = (BLOCK_M, ), layout = gl.SliceLayout(1, Q_LOAD_LAYOUT)
        # M : shape = (BLOCK_M, ), layout = gl.SliceLayout(1, Q_LOAD_LAYOUT)
        # acc : shape = (BLOCK_M, HEAD_SIZE_PADDED), layout = PV_WMMA_LAYOUT
        P, L, M, acc = _perform_QK_wmma_and_update_L_M(
            Q,
            K,
            L,
            M,
            acc,
            qq_bias_row_ptrs,
            seq_offset,
            query_mask_qk,
            query_pos_qk,
            context_len,
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
            Q_LOAD_LAYOUT,
            QK_WMMA_LAYOUT,
            PV_WMMA_LAYOUT,
        )

        # V : shape = (TILE_SIZE, HEAD_SIZE_PADDED), layout = V_DOT_LAYOUT
        v_from_lds, V = _tdm_request_from_lds(
            v_from_lds,
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

        j_from_lds = j_from_lds + 1
        seq_offset += TILE_SIZE

    for _ in range(num_stages - 1):
        # K : shape = (HEAD_SIZE_PADDED, TILE_SIZE), layout = K_DOT_LAYOUT
        k_from_lds, K = _tdm_request_from_lds(
            k_from_lds,
            k_scale,
            Q.dtype,
            smem_K,
            asycn_wait=(num_stages - 2) * 2 + 1,
            layout=K_DOT_LAYOUT,
            transpose=True,
            num_ctas=num_ctas,
            num_stages=num_stages,
        )

        # P : shape = (BLOCK_M, TILE_SIZE), layout = Q_LOAD_LAYOUT
        # L : shape = (BLOCK_M, ), layout = gl.SliceLayout(1, Q_LOAD_LAYOUT)
        # M : shape = (BLOCK_M, ), layout = gl.SliceLayout(1, Q_LOAD_LAYOUT)
        # acc : shape = (BLOCK_M, HEAD_SIZE_PADDED), layout = PV_WMMA_LAYOUT
        P, L, M, acc = _perform_QK_wmma_and_update_L_M(
            Q,
            K,
            L,
            M,
            acc,
            qq_bias_row_ptrs,
            seq_offset,
            query_mask_qk,
            query_pos_qk,
            context_len,
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
            Q_LOAD_LAYOUT,
            QK_WMMA_LAYOUT,
            PV_WMMA_LAYOUT,
        )

        # V : shape = (TILE_SIZE, HEAD_SIZE_PADDED), layout = V_DOT_LAYOUT
        v_from_lds, V = _tdm_request_from_lds(
            v_from_lds,
            v_scale,
            Q.dtype,
            smem_V,
            asycn_wait=(num_stages - 2) * 2,
            layout=V_DOT_LAYOUT,
            transpose=False,
            num_ctas=num_ctas,
            num_stages=num_stages,
        )

        # acc : shape = (BLOCK_M, HEAD_SIZE_PADDED), layout = PV_WMMA_LAYOUT
        acc = _perform_PV_wmma(P, V, acc, P_DOT_LAYOUT)

        j_from_lds = j_from_lds + 1
        seq_offset += TILE_SIZE

    # store segm_output
    # acc : shape = (BLOCK_M, HEAD_SIZE_PADDED), layout = Q_LOAD_LAYOUT
    acc = gl.convert_layout(acc, layout=Q_LOAD_LAYOUT)
    segm_output_offset = (
        query_offset_0[:, None]
        * (num_query_heads * NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED)
        + query_offset_1[:, None] * (NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED)
        + segm_idx * HEAD_SIZE_PADDED
        + offs_q_d[None, :]
    )
    gl.amd.cdna4.buffer_store(
        stored_value=acc,
        ptr=segm_output_ptr,
        offsets=segm_output_offset,
        mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
    )

    # store segm_max and segm_expsum
    # L : shape = (BLOCK_M, ), layout = gl.SliceLayout(1, QK_WMMA_LAYOUT)
    # M : shape = (BLOCK_M, ), layout = gl.SliceLayout(1, QK_WMMA_LAYOUT)
    segm_offset = (
        query_offset_0 * (num_query_heads * NUM_SEGMENTS_PER_SEQ)
        + query_offset_1 * NUM_SEGMENTS_PER_SEQ
        + segm_idx
    )
    L = gl.convert_layout(L, layout=gl.SliceLayout(1, Q_LOAD_LAYOUT))
    M = gl.convert_layout(M, layout=gl.SliceLayout(1, Q_LOAD_LAYOUT))
    gl.amd.cdna4.buffer_store(
        stored_value=M,
        ptr=segm_max_ptr,
        offsets=segm_offset,
        mask=query_mask_0 & query_mask_1,
    )
    gl.amd.cdna4.buffer_store(
        stored_value=L,
        ptr=segm_expsum_ptr,
        offsets=segm_offset,
        mask=query_mask_0 & query_mask_1,
    )


@gluon.jit
def _async_load_to_lds(
    from_hbm,
    dest,
    ptr,
    offsets,
    mask,
    num_stages: gl.constexpr,
    cache_modifier: gl.constexpr,
    use_buffer_load: gl.constexpr = False,
):
    if use_buffer_load:
        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            dest=dest.index(from_hbm % num_stages),
            ptr=ptr,
            offsets=offsets.to(gl.int32),
            mask=mask,
            cache_modifier=cache_modifier,
        )
    else:
        gl.amd.cdna4.async_copy.global_load_to_shared(
            dest=dest.index(from_hbm % num_stages),
            ptr=ptr + offsets,
            mask=mask,
            cache_modifier=cache_modifier,
        )
    gl.amd.cdna4.async_copy.commit_group()
    return from_hbm + 1


@gluon.jit
def _request_from_lds(
    from_lds,
    kv_scale,
    Q_dtype,
    smem,
    layout,
    wait_group,
    num_stages: gl.constexpr,
):
    gl.amd.cdna4.async_copy.wait_group(wait_group)
    KV = gl.amd.cdna4.async_copy.load_shared_relaxed(
        smem.index(from_lds % num_stages), layout=layout
    )
    if KV.dtype.is_fp8() and not Q_dtype.is_fp8():
        KV = (KV.to(gl.float32) * gl.load(kv_scale)).to(Q_dtype)
    return KV, from_lds + 1


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
    stride_k_cache_0: gl.int64,
    stride_k_cache_1: gl.int64,
    stride_k_cache_2: gl.int64,
    stride_k_cache_3: gl.constexpr,
    stride_v_cache_0: gl.int64,
    stride_v_cache_1: gl.int64,
    stride_v_cache_2: gl.int64,
    stride_v_cache_3: gl.constexpr,
    K_LOAD_LAYOUT: gl.constexpr,
    V_LOAD_LAYOUT: gl.constexpr,
    TILE_SIZE: gl.constexpr,
    BLOCK_SIZE: gl.constexpr,
):
    # seq_k_offset : shape = (TILE_SIZE, ), layout = gl.SliceLayout(0, K_LOAD_LAYOUT)
    # seq_v_offset : shape = (TILE_SIZE, ), layout = gl.SliceLayout(1, V_LOAD_LAYOUT)
    seq_k_offset = j * TILE_SIZE + offs_k_t
    seq_v_offset = j * TILE_SIZE + offs_v_t

    if TILE_SIZE == BLOCK_SIZE:
        tile_k_mask = gl.full(
            (1,), 1, dtype=tl.int1, layout=gl.SliceLayout(0, K_LOAD_LAYOUT)
        )
        tile_v_mask = gl.full(
            (1,), 1, dtype=tl.int1, layout=gl.SliceLayout(1, V_LOAD_LAYOUT)
        )
    else:
        tile_k_mask = seq_k_offset < max_seq_prefix_len
        tile_v_mask = seq_v_offset < max_seq_prefix_len

    physical_block_idx_k = gl.amd.cdna4.buffer_load(
        ptr=block_tables_ptr,
        offsets=(block_table_offset + seq_k_offset // BLOCK_SIZE).to(gl.int32),
    ).to(tl.int64)

    physical_block_idx_v = gl.amd.cdna4.buffer_load(
        ptr=block_tables_ptr,
        offsets=(block_table_offset + seq_v_offset // BLOCK_SIZE).to(gl.int32),
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


gluon_kernel_unified_attention_3d_async_repr = make_kernel_repr(
    "gluon_kernel_unified_attention_3d_async",
    [
        "num_query_heads",
        "num_queries_per_kv",
        "BLOCK_SIZE",
        "TILE_SIZE",
        "HEAD_SIZE",
        "num_warps",
        "num_stages",
        "cache_modifier",
    ],
)


@gluon.jit(repr=gluon_kernel_unified_attention_3d_async_repr)
def gluon_kernel_unified_attention_3d_async(
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
    NUM_BLOCKS,  # int
    num_query_heads: gl.constexpr,  # int
    num_queries_per_kv: gl.constexpr,  # int
    block_table_stride: gl.int64,  # int
    query_stride_0: gl.int64,  # int
    query_stride_1: gl.int64,  # int, should be equal to head_size
    qq_bias_stride_0: gl.int64,  # int
    BLOCK_SIZE: gl.constexpr,  # int
    TILE_SIZE: gl.constexpr,  # int, must be power of 2
    HEAD_SIZE: gl.constexpr,  # int
    HEAD_SIZE_PADDED: gl.constexpr,  # int, must be power of 2
    USE_ALIBI_SLOPES: gl.constexpr,  # bool
    USE_QQ_BIAS: gl.constexpr,  # bool
    USE_SOFTCAP: gl.constexpr,  # bool
    USE_SINKS: gl.constexpr,  # bool
    SLIDING_WINDOW: gl.constexpr,  # int
    stride_k_cache_0: gl.int64,  # int
    stride_k_cache_1: gl.int64,  # int
    stride_k_cache_2: gl.int64,  # int
    stride_k_cache_3: gl.constexpr,  # int
    stride_v_cache_0: gl.int64,  # int
    stride_v_cache_1: gl.int64,  # int
    stride_v_cache_2: gl.int64,  # int
    stride_v_cache_3: gl.constexpr,  # int
    query_start_len_ptr,  # [num_seqs+1]
    BLOCK_Q: gl.constexpr,  # int
    num_seqs: gl.int32,
    BLOCK_M: gl.constexpr,  # int
    NUM_SEGMENTS_PER_SEQ: gl.constexpr,  # int
    num_warps: gl.constexpr,  # int
    num_stages: gl.constexpr,  # int
    QK_WMMA_LAYOUT: gl.constexpr,
    PV_WMMA_LAYOUT: gl.constexpr,
    Q_DOT_LAYOUT: gl.constexpr,
    K_DOT_LAYOUT: gl.constexpr,
    P_DOT_LAYOUT: gl.constexpr,
    V_DOT_LAYOUT: gl.constexpr,
    Q_SHARED_LAYOUT: gl.constexpr,
    K_SHARED_LAYOUT: gl.constexpr,
    V_SHARED_LAYOUT: gl.constexpr,
    Q_LOAD_LAYOUT: gl.constexpr,
    K_LOAD_LAYOUT: gl.constexpr,
    V_LOAD_LAYOUT: gl.constexpr,
    use_buffer_load: gl.constexpr = True,
    ALL_DECODE: gl.constexpr = False,  # bool
    SHUFFLED_KV_CACHE: gl.constexpr = False,  # bool
):
    q_block_global_idx = gl.program_id(0)
    kv_head_idx = gl.program_id(1)
    segm_idx = gl.program_id(2)
    num_ctas: gl.constexpr = gl.num_ctas()
    pred = 1
    pred_i32 = pred.to(gl.int32) if hasattr(pred, "to") else pred

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

    smem_Q = gl.allocate_shared_memory(
        query_ptr.type.element_ty, [BLOCK_M, HEAD_SIZE_PADDED], layout=Q_SHARED_LAYOUT
    )
    smem_K = gl.allocate_shared_memory(
        key_cache_ptr.type.element_ty,
        [num_stages, HEAD_SIZE_PADDED, TILE_SIZE],
        layout=K_SHARED_LAYOUT,
    )
    smem_V = gl.allocate_shared_memory(
        value_cache_ptr.type.element_ty,
        [num_stages, TILE_SIZE, HEAD_SIZE_PADDED],
        layout=V_SHARED_LAYOUT,
    )

    offs_q_m = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, Q_LOAD_LAYOUT))
    offs_q_d = gl.arange(0, HEAD_SIZE_PADDED, layout=gl.SliceLayout(0, Q_LOAD_LAYOUT))

    offs_k_t = gl.arange(0, TILE_SIZE, layout=gl.SliceLayout(0, K_LOAD_LAYOUT))
    offs_k_d = gl.arange(0, HEAD_SIZE_PADDED, layout=gl.SliceLayout(1, K_LOAD_LAYOUT))

    offs_v_t = gl.arange(0, TILE_SIZE, layout=gl.SliceLayout(1, V_LOAD_LAYOUT))
    offs_v_d = gl.arange(0, HEAD_SIZE_PADDED, layout=gl.SliceLayout(0, V_LOAD_LAYOUT))

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
        dim_mask = gl.full((1,), 1, dtype=tl.int1)

    query_mask_0 = query_pos < cur_batch_query_len
    query_mask_1 = query_offset_1 < num_query_heads

    # Q_load : shape = (BLOCK_M, HEAD_SIZE_PADDED), layout = Q_LOAD_LAYOUT
    Q_load = gl.amd.cdna4.buffer_load(
        ptr=query_ptr,
        offsets=query_offset.to(gl.int32),
        mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
        other=0.0,
    )
    smem_Q.store(Q_load)
    # Q : shape = (BLOCK_M, HEAD_SIZE_PADDED), layout = Q_DOT_LAYOUT
    Q = smem_Q.load(layout=Q_DOT_LAYOUT)

    offs_q_m_qk = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, QK_WMMA_LAYOUT))
    query_pos_qk = q_block_local_idx * BLOCK_Q + offs_q_m_qk // num_queries_per_kv
    query_offset_1_qk = (
        kv_head_idx * num_queries_per_kv + offs_q_m_qk % num_queries_per_kv
    )
    query_mask_0_qk = query_pos_qk < cur_batch_query_len
    query_mask_1_qk = query_offset_1_qk < num_query_heads
    query_mask_qk = query_mask_1_qk[:, None] & query_mask_0_qk[:, None]

    L, M, acc = _allocate_L_M_acc(
        sink_ptr,
        segm_idx,
        query_offset_1_qk,
        query_mask_1_qk,
        RCP_LN2,
        BLOCK_M,
        HEAD_SIZE_PADDED,
        QK_WMMA_LAYOUT,
        PV_WMMA_LAYOUT,
        USE_SINKS,
    )

    # alibi slope for this head
    alibi_slope = None
    if USE_ALIBI_SLOPES:
        alibi_slope = tl.load(
            alibi_slopes_ptr + query_offset_1_qk, mask=query_mask_1, other=0.0
        )

    # query-query attention bias
    qq_bias_row_ptrs = None
    if USE_QQ_BIAS:
        qq_bias_row_ptrs = (
            qq_bias_ptr + query_pos_qk[:, None] * qq_bias_stride_0
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
    max_seq_prefix_len = gl.minimum(max_seq_prefix_len, seq_len)

    # calculate the number of tiles that need to be processed to
    # cover the longest sequence prefix (due to causal masking, tiles beyond
    # this prefix can be skipped)
    num_tiles = cdiv_fn(max_seq_prefix_len, TILE_SIZE)

    KV_cache_modifier: gl.constexpr = ".cg" if ALL_DECODE else ""

    k_from_hbm = 0
    k_from_lds = 0
    v_from_hbm = 0
    v_from_lds = 0
    j_from_hbm = segm_idx * tiles_per_segment
    seq_offset = j_from_hbm * TILE_SIZE + gl.arange(
        0, TILE_SIZE, layout=gl.SliceLayout(0, QK_WMMA_LAYOUT)
    )

    for _ in range(num_stages - 1):
        j_from_hbm, k_offset, v_offset, tile_k_mask, tile_v_mask = _get_kv_offsets(
            j_from_hbm,
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
            K_LOAD_LAYOUT,
            V_LOAD_LAYOUT,
            TILE_SIZE,
            BLOCK_SIZE,
        )
        k_from_hbm = _async_load_to_lds(
            k_from_hbm,
            dest=smem_K,
            ptr=key_cache_ptr,
            offsets=k_offset,
            mask=dim_mask[:, None] & tile_k_mask[None, :],
            num_stages=num_stages,
            cache_modifier=KV_cache_modifier,
            use_buffer_load=use_buffer_load,
        )
        v_from_hbm = _async_load_to_lds(
            v_from_hbm,
            dest=smem_V,
            ptr=value_cache_ptr,
            offsets=v_offset,
            mask=dim_mask[None, :] & tile_v_mask[:, None],
            num_stages=num_stages,
            cache_modifier=KV_cache_modifier,
            use_buffer_load=use_buffer_load,
        )

    # iterate through tiles within current segment
    for _ in range(
        segm_idx * tiles_per_segment,
        min((segm_idx + 1) * tiles_per_segment, num_tiles) - (num_stages - 1),
    ):
        j_from_hbm, k_offset, v_offset, tile_k_mask, tile_v_mask = _get_kv_offsets(
            j_from_hbm,
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
            K_LOAD_LAYOUT,
            V_LOAD_LAYOUT,
            TILE_SIZE,
            BLOCK_SIZE,
        )

        K, k_from_lds = _request_from_lds(
            k_from_lds,
            k_scale,
            Q.dtype,
            smem_K,
            layout=K_DOT_LAYOUT,
            wait_group=(num_stages - 2) * 2 + 1,
            num_stages=num_stages,
        )

        # K_load : shape = (HEAD_SIZE_PADDED, TILE_SIZE), layout = K_LOAD_LAYOUT
        k_from_hbm = _async_load_to_lds(
            k_from_hbm,
            dest=smem_K,
            ptr=key_cache_ptr,
            offsets=k_offset,
            mask=dim_mask[:, None] & tile_k_mask[None, :],
            num_stages=num_stages,
            cache_modifier=KV_cache_modifier,
            use_buffer_load=use_buffer_load,
        )

        # V_load : shape = (TILE_SIZE, HEAD_SIZE_PADDED), layout = Q_LOAD_LAYOUT
        v_from_hbm = _async_load_to_lds(
            v_from_hbm,
            dest=smem_V,
            ptr=value_cache_ptr,
            offsets=v_offset,
            mask=dim_mask[None, :] & tile_v_mask[:, None],
            num_stages=num_stages,
            cache_modifier=KV_cache_modifier,
            use_buffer_load=use_buffer_load,
        )

        # P : shape = (BLOCK_M, TILE_SIZE), layout = Q_LOAD_LAYOUT
        # L : shape = (BLOCK_M, ), layout = gl.SliceLayout(1, Q_LOAD_LAYOUT)
        # M : shape = (BLOCK_M, ), layout = gl.SliceLayout(1, Q_LOAD_LAYOUT)
        # acc : shape = (BLOCK_M, HEAD_SIZE_PADDED), layout = PV_WMMA_LAYOUT
        P, L, M, acc = _perform_QK_wmma_and_update_L_M(
            Q,
            K,
            L,
            M,
            acc,
            qq_bias_row_ptrs,
            seq_offset,
            query_mask_qk,
            query_pos_qk,
            context_len,
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
            Q_LOAD_LAYOUT,
            QK_WMMA_LAYOUT,
            PV_WMMA_LAYOUT,
        )

        # V : shape = (TILE_SIZE, HEAD_SIZE_PADDED), layout = V_DOT_LAYOUT
        V, v_from_lds = _request_from_lds(
            v_from_lds,
            v_scale,
            Q.dtype,
            smem_V,
            layout=V_DOT_LAYOUT,
            wait_group=(num_stages - 1) * 2,
            num_stages=num_stages,
        )

        # acc : shape = (BLOCK_M, HEAD_SIZE_PADDED), layout = PV_WMMA_LAYOUT
        acc = _perform_PV_wmma(P, V, acc, P_DOT_LAYOUT)

        seq_offset += TILE_SIZE

    for _ in range(num_stages - 1):
        # K : shape = (HEAD_SIZE_PADDED, TILE_SIZE), layout = K_DOT_LAYOUT
        K, k_from_lds = _request_from_lds(
            k_from_lds,
            k_scale,
            Q.dtype,
            smem_K,
            layout=K_DOT_LAYOUT,
            wait_group=(num_stages - 2) * 2
            + 1,  # there is no async_copy in the epilogue, hence num_stages - 2
            # wait_group=0,
            num_stages=num_stages,
        )

        # P : shape = (BLOCK_M, TILE_SIZE), layout = Q_LOAD_LAYOUT
        # L : shape = (BLOCK_M, ), layout = gl.SliceLayout(1, Q_LOAD_LAYOUT)
        # M : shape = (BLOCK_M, ), layout = gl.SliceLayout(1, Q_LOAD_LAYOUT)
        # acc : shape = (BLOCK_M, HEAD_SIZE_PADDED), layout = PV_WMMA_LAYOUT
        P, L, M, acc = _perform_QK_wmma_and_update_L_M(
            Q,
            K,
            L,
            M,
            acc,
            qq_bias_row_ptrs,
            seq_offset,
            query_mask_qk,
            query_pos_qk,
            context_len,
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
            Q_LOAD_LAYOUT,
            QK_WMMA_LAYOUT,
            PV_WMMA_LAYOUT,
        )

        # V : shape = (TILE_SIZE, HEAD_SIZE_PADDED), layout = V_DOT_LAYOUT
        V, v_from_lds = _request_from_lds(
            v_from_lds,
            v_scale,
            Q.dtype,
            smem_V,
            layout=V_DOT_LAYOUT,
            wait_group=(num_stages - 2)
            * 2,  # there is no async_copy in the epilogue, hence num_stages - 2
            # wait_group=0,
            num_stages=num_stages,
        )

        # acc : shape = (BLOCK_M, HEAD_SIZE_PADDED), layout = PV_WMMA_LAYOUT
        acc = _perform_PV_wmma(P, V, acc, P_DOT_LAYOUT)

        seq_offset += TILE_SIZE

    # store segm_output
    # acc : shape = (BLOCK_M, HEAD_SIZE_PADDED), layout = Q_LOAD_LAYOUT
    acc = gl.convert_layout(acc, layout=Q_LOAD_LAYOUT)
    segm_output_offset = (
        query_offset_0[:, None]
        * (num_query_heads * NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED)
        + query_offset_1[:, None] * (NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED)
        + segm_idx * HEAD_SIZE_PADDED
        + offs_q_d[None, :]
    )
    gl.amd.cdna4.buffer_store(
        stored_value=acc,
        ptr=segm_output_ptr,
        offsets=segm_output_offset,
        mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
    )

    # store segm_max and segm_expsum
    # L : shape = (BLOCK_M, ), layout = gl.SliceLayout(1, QK_WMMA_LAYOUT)
    # M : shape = (BLOCK_M, ), layout = gl.SliceLayout(1, QK_WMMA_LAYOUT)
    segm_offset = (
        query_offset_0 * (num_query_heads * NUM_SEGMENTS_PER_SEQ)
        + query_offset_1 * NUM_SEGMENTS_PER_SEQ
        + segm_idx
    )
    L = gl.convert_layout(L, layout=gl.SliceLayout(1, Q_LOAD_LAYOUT))
    M = gl.convert_layout(M, layout=gl.SliceLayout(1, Q_LOAD_LAYOUT))
    gl.amd.cdna4.buffer_store(
        stored_value=M,
        ptr=segm_max_ptr,
        offsets=segm_offset,
        mask=query_mask_0 & query_mask_1,
    )
    gl.amd.cdna4.buffer_store(
        stored_value=L,
        ptr=segm_expsum_ptr,
        offsets=segm_offset,
        mask=query_mask_0 & query_mask_1,
    )


@gluon.jit
def _unshuffle_kv_cache(
    X,
    BLOCK_SIZE_N: gl.constexpr,
    BLOCK_SIZE_INNER_DIM: gl.constexpr,
):
    return (
        X.reshape(
            1,
            BLOCK_SIZE_N // 16,
            BLOCK_SIZE_INNER_DIM // 16,
            2,
            16,
            8,
        )
        .permute(0, 1, 4, 2, 3, 5)
        .reshape(BLOCK_SIZE_N, BLOCK_SIZE_INNER_DIM)
        .trans(1, 0)
    )


@gluon.jit
def _buffer_load_to_reg(
    x_scale,
    Q_dtype,
    ptr,
    offsets,
    mask,
    other,
    cache_modifier: gl.constexpr,
    SHUFFLED_KV_CACHE: gl.constexpr = False,
):
    X = gl.amd.cdna4.buffer_load(
        ptr=ptr,
        offsets=offsets.to(gl.int32),
        mask=mask,
        other=other,
        cache=cache_modifier,
    )
    if X.dtype.is_fp8() and not Q_dtype.is_fp8():
        X = (X.to(gl.float32) * gl.load(x_scale)).to(Q_dtype)
    return X


gluon_kernel_unified_attention_3d_repr = make_kernel_repr(
    "gluon_kernel_unified_attention_3d",
    [
        "num_query_heads",
        "num_queries_per_kv",
        "BLOCK_SIZE",
        "TILE_SIZE",
        "HEAD_SIZE",
        "num_warps",
        "num_stages",
        "ALL_DECODE",
    ],
)


@gluon.jit(repr=gluon_kernel_unified_attention_3d_repr)
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
    num_query_heads: gl.constexpr,  # int
    num_queries_per_kv: gl.constexpr,  # int
    block_table_stride: gl.int64,  # int
    query_stride_0: gl.int64,  # int
    query_stride_1: gl.int64,  # int, should be equal to head_size
    qq_bias_stride_0: gl.int64,  # int
    NUM_BLOCKS: gl.constexpr,  # int
    BLOCK_SIZE: gl.constexpr,  # int
    TILE_SIZE: gl.constexpr,  # int, must be power of 2
    HEAD_SIZE: gl.constexpr,  # int
    HEAD_SIZE_PADDED: gl.constexpr,  # int, must be power of 2
    USE_ALIBI_SLOPES: gl.constexpr,  # bool
    USE_QQ_BIAS: gl.constexpr,  # bool
    USE_SOFTCAP: gl.constexpr,  # bool
    USE_SINKS: gl.constexpr,  # bool
    SLIDING_WINDOW: gl.constexpr,  # int
    stride_k_cache_0: gl.int64,  # int
    stride_k_cache_1: gl.int64,  # int
    stride_k_cache_2: gl.int64,  # int
    stride_k_cache_3: gl.constexpr,  # int
    stride_v_cache_0: gl.int64,  # int
    stride_v_cache_1: gl.int64,  # int
    stride_v_cache_2: gl.int64,  # int
    stride_v_cache_3: gl.constexpr,  # int
    query_start_len_ptr,  # [num_seqs+1]
    BLOCK_Q: gl.constexpr,  # int
    num_seqs: gl.int32,
    BLOCK_M: gl.constexpr,  # int
    NUM_SEGMENTS_PER_SEQ: gl.constexpr,  # int
    num_warps: gl.constexpr,  # int
    num_stages: gl.constexpr,  # int
    QK_WMMA_LAYOUT: gl.constexpr,
    PV_WMMA_LAYOUT: gl.constexpr,
    Q_DOT_LAYOUT: gl.constexpr,
    K_DOT_LAYOUT: gl.constexpr,
    P_DOT_LAYOUT: gl.constexpr,
    V_DOT_LAYOUT: gl.constexpr,
    Q_SHARED_LAYOUT: gl.constexpr,
    K_SHARED_LAYOUT: gl.constexpr,
    V_SHARED_LAYOUT: gl.constexpr,
    Q_LOAD_LAYOUT: gl.constexpr,
    K_LOAD_LAYOUT: gl.constexpr,
    V_LOAD_LAYOUT: gl.constexpr,
    ALL_DECODE: gl.constexpr = False,  # bool
    SHUFFLED_KV_CACHE: gl.constexpr = False,  # bool
):
    q_block_global_idx = gl.program_id(0)
    kv_head_idx = gl.program_id(1)
    segm_idx = gl.program_id(2)

    if SHUFFLED_KV_CACHE:
        gl.static_assert(
            TILE_SIZE == BLOCK_SIZE,
            "TILE_SIZE must be equal to BLOCK_SIZE if SHUFFLED_KV_CACHE is True",
        )
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

    smem_Q = gl.allocate_shared_memory(
        query_ptr.type.element_ty, [BLOCK_M, HEAD_SIZE_PADDED], layout=Q_SHARED_LAYOUT
    )
    smem_K = None
    smem_V = None
    if SHUFFLED_KV_CACHE:
        # pass
        smem_K = gl.allocate_shared_memory(
            key_cache_ptr.type.element_ty,
            [TILE_SIZE // 16, HEAD_SIZE_PADDED * 16],
            layout=K_SHARED_LAYOUT,
        )
        smem_V = gl.allocate_shared_memory(
            value_cache_ptr.type.element_ty,
            [HEAD_SIZE_PADDED // 16, TILE_SIZE * 16],
            layout=V_SHARED_LAYOUT,
        )
    else:
        smem_K = gl.allocate_shared_memory(
            key_cache_ptr.type.element_ty,
            [HEAD_SIZE_PADDED, TILE_SIZE],
            layout=K_SHARED_LAYOUT,
        )
        smem_V = gl.allocate_shared_memory(
            value_cache_ptr.type.element_ty,
            [TILE_SIZE, HEAD_SIZE_PADDED],
            layout=V_SHARED_LAYOUT,
        )

    offs_q_m = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, Q_LOAD_LAYOUT))
    offs_q_d = gl.arange(0, HEAD_SIZE_PADDED, layout=gl.SliceLayout(0, Q_LOAD_LAYOUT))

    if SHUFFLED_KV_CACHE:
        offs_k_t = gl.arange(
            0, TILE_SIZE // 16, layout=gl.SliceLayout(1, K_LOAD_LAYOUT)
        )
        offs_k_d = gl.arange(
            0, HEAD_SIZE_PADDED * 16, layout=gl.SliceLayout(0, K_LOAD_LAYOUT)
        )
        offs_v_t = gl.arange(0, TILE_SIZE * 16, layout=gl.SliceLayout(0, V_LOAD_LAYOUT))
        offs_v_d = gl.arange(
            0, HEAD_SIZE_PADDED // 16, layout=gl.SliceLayout(1, V_LOAD_LAYOUT)
        )
    else:
        offs_k_t = gl.arange(0, TILE_SIZE, layout=gl.SliceLayout(0, K_LOAD_LAYOUT))
        offs_k_d = gl.arange(
            0, HEAD_SIZE_PADDED, layout=gl.SliceLayout(1, K_LOAD_LAYOUT)
        )
        offs_v_t = gl.arange(0, TILE_SIZE, layout=gl.SliceLayout(1, V_LOAD_LAYOUT))
        offs_v_d = gl.arange(
            0, HEAD_SIZE_PADDED, layout=gl.SliceLayout(0, V_LOAD_LAYOUT)
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
        dim_mask = gl.full((1,), 1, dtype=tl.int1)

    query_mask_0 = query_pos < cur_batch_query_len
    query_mask_1 = query_offset_1 < num_query_heads

    # Q_load : shape = (BLOCK_M, HEAD_SIZE_PADDED), layout = Q_LOAD_LAYOUT
    Q_load = gl.amd.cdna4.buffer_load(
        ptr=query_ptr,
        offsets=query_offset.to(gl.int32),
        mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
        other=0.0,
    )
    smem_Q.store(Q_load)
    # Q : shape = (BLOCK_M, HEAD_SIZE_PADDED), layout = Q_DOT_LAYOUT
    Q = smem_Q.load(layout=Q_DOT_LAYOUT)

    offs_q_m_qk = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, QK_WMMA_LAYOUT))
    offs_q_d_qk = gl.arange(
        0, HEAD_SIZE_PADDED, layout=gl.SliceLayout(0, QK_WMMA_LAYOUT)
    )
    query_pos_qk = q_block_local_idx * BLOCK_Q + offs_q_m_qk // num_queries_per_kv
    query_offset_0_qk = cur_batch_in_all_start_index + query_pos_qk
    query_offset_1_qk = (
        kv_head_idx * num_queries_per_kv + offs_q_m_qk % num_queries_per_kv
    )
    query_mask_0_qk = query_pos_qk < cur_batch_query_len
    query_mask_1_qk = query_offset_1_qk < num_query_heads
    query_mask_qk = query_mask_1_qk[:, None] & query_mask_0_qk[:, None]
    offs_seq_t = gl.arange(0, TILE_SIZE, layout=gl.SliceLayout(0, QK_WMMA_LAYOUT))

    L, M, acc = _allocate_L_M_acc(
        sink_ptr,
        segm_idx,
        query_offset_1_qk,
        query_mask_1_qk,
        RCP_LN2,
        BLOCK_M,
        HEAD_SIZE_PADDED,
        QK_WMMA_LAYOUT,
        PV_WMMA_LAYOUT,
        USE_SINKS,
    )

    # alibi slope for this head
    alibi_slope = None
    if USE_ALIBI_SLOPES:
        alibi_slope = tl.load(
            alibi_slopes_ptr + query_offset_1_qk, mask=query_mask_1, other=0.0
        )

    # query-query attention bias
    qq_bias_row_ptrs = None
    if USE_QQ_BIAS:
        qq_bias_row_ptrs = (
            qq_bias_ptr + query_pos_qk[:, None] * qq_bias_stride_0
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
        # seq_k_offset : shape = (TILE_SIZE if not SHUFFLED_KV_CACHE else TILE_SIZE // 16, ), layout = gl.SliceLayout(0 if not SHUFFLED_KV_CACHE else 1, K_LOAD_LAYOUT)
        # seq_v_offset : shape = (TILE_SIZE, ), layout = gl.SliceLayout(1, Q_LOAD_LAYOUT)
        # seq_offset : shape = (TILE_SIZE, ), layout = gl.SliceLayout(0, QK_WMMA_LAYOUT)
        seq_offset = j * TILE_SIZE + offs_seq_t

        k_mask = None
        v_mask = None
        other = None
        if SHUFFLED_KV_CACHE:
            # seq_k_offset = j * TILE_SIZE + offs_k_t * 16
            # seq_v_offset = j * TILE_SIZE + offs_v_t // 16
            # physical_block_idx_k = gl.amd.cdna4.buffer_load(
            #     ptr=block_tables_ptr,
            #     offsets=(block_table_offset + seq_k_offset // BLOCK_SIZE).to(gl.int32),
            # ).to(tl.int64)

            # physical_block_idx_v = gl.amd.cdna4.buffer_load(
            #     ptr=block_tables_ptr,
            #     offsets=(block_table_offset + seq_v_offset // BLOCK_SIZE).to(gl.int32),
            # ).to(tl.int64)
            physical_block_idx = gl.load(block_tables_ptr + block_table_offset + j).to(
                tl.int64
            )

            k_offset = (
                # physical_block_idx_k[:, None] * stride_k_cache_0
                physical_block_idx * stride_k_cache_0
                + kv_head_idx * stride_k_cache_1
                + offs_k_t[:, None] * stride_k_cache_2
                + offs_k_d[None, :] * stride_k_cache_3
            )
            v_offset = (
                # physical_block_idx_v[None, :] * stride_v_cache_0
                physical_block_idx * stride_v_cache_0
                + kv_head_idx * stride_v_cache_1
                + offs_v_t[None, :] * stride_v_cache_3
                + offs_v_d[:, None] * stride_v_cache_2
            )
        else:
            seq_k_offset = j * TILE_SIZE + offs_k_t
            seq_v_offset = j * TILE_SIZE + offs_v_t

            if TILE_SIZE == BLOCK_SIZE:
                tile_k_mask = gl.full(
                    (1,), 1, dtype=tl.int1, layout=gl.SliceLayout(0, K_LOAD_LAYOUT)
                )
                tile_v_mask = gl.full(
                    (1,), 1, dtype=tl.int1, layout=gl.SliceLayout(1, Q_LOAD_LAYOUT)
                )
            else:
                tile_k_mask = seq_k_offset < max_seq_prefix_len
                tile_v_mask = seq_v_offset < max_seq_prefix_len

            physical_block_idx_k = gl.amd.cdna4.buffer_load(
                ptr=block_tables_ptr,
                offsets=(block_table_offset + seq_k_offset // BLOCK_SIZE).to(gl.int32),
            ).to(tl.int64)

            physical_block_idx_v = gl.amd.cdna4.buffer_load(
                ptr=block_tables_ptr,
                offsets=(block_table_offset + seq_v_offset // BLOCK_SIZE).to(gl.int32),
            ).to(tl.int64)

            k_offset = (
                physical_block_idx_k[None, :] * stride_k_cache_0
                + kv_head_idx * stride_k_cache_2
                + offs_k_d[:, None] * stride_k_cache_3
                + (seq_k_offset % BLOCK_SIZE)[None, :] * stride_k_cache_1
            )
            k_mask = dim_mask[:, None] & tile_k_mask[None, :]
            v_offset = (
                physical_block_idx_v[:, None] * stride_v_cache_0
                + kv_head_idx * stride_v_cache_2
                + offs_v_d[None, :] * stride_v_cache_3
                + (seq_v_offset % BLOCK_SIZE)[:, None] * stride_v_cache_1
            )
            v_mask = dim_mask[None, :] & tile_v_mask[:, None]
            other = 0.0

        # K_load : shape = (HEAD_SIZE_PADDED, TILE_SIZE), layout = K_LOAD_LAYOUT
        K = _buffer_load_to_reg(
            k_scale,
            Q.dtype,
            key_cache_ptr,
            k_offset.to(gl.int32),
            k_mask,
            other,
            KV_cache_modifier,
            SHUFFLED_KV_CACHE,
        )
        if SHUFFLED_KV_CACHE:
            # smem_K.store(K)
            pass
        else:
            smem_K.store(K)

        # V_load : shape = (TILE_SIZE, HEAD_SIZE_PADDED), layout = Q_LOAD_LAYOUT
        V = _buffer_load_to_reg(
            v_scale,
            Q.dtype,
            value_cache_ptr,
            v_offset.to(gl.int32),
            v_mask,
            other,
            KV_cache_modifier,
        )
        if SHUFFLED_KV_CACHE:
            # smem_V.store(V)
            pass
        else:
            smem_V.store(V)

        if SHUFFLED_KV_CACHE:
            # K = smem_K.load(layout=K_LOAD_LAYOUT)
            K = _unshuffle_kv_cache(K, TILE_SIZE, HEAD_SIZE_PADDED)
            K = gl.convert_layout(value=K, layout=K_DOT_LAYOUT, assert_trivial=True)
        else:
            K = smem_K.load(layout=K_DOT_LAYOUT)
        P, L, M, acc = _perform_QK_wmma_and_update_L_M(
            Q,
            K,
            L,
            M,
            acc,
            qq_bias_row_ptrs,
            seq_offset,
            query_mask_qk,
            query_pos_qk,
            context_len,
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
            Q_LOAD_LAYOUT,
            QK_WMMA_LAYOUT,
            PV_WMMA_LAYOUT,
        )

        if SHUFFLED_KV_CACHE:
            # V = smem_V.load(layout=V_LOAD_LAYOUT)
            V = _unshuffle_kv_cache(V, HEAD_SIZE_PADDED, TILE_SIZE)
            V = gl.convert_layout(value=V, layout=V_DOT_LAYOUT, assert_trivial=True)
        else:
            V = smem_V.load(layout=V_DOT_LAYOUT)
        # acc : shape = (BLOCK_M, HEAD_SIZE_PADDED), layout = PV_WMMA_LAYOUT
        acc = _perform_PV_wmma(P, V, acc, P_DOT_LAYOUT)

    # store segm_output
    # acc : shape = (BLOCK_M, HEAD_SIZE_PADDED), layout = PV_WMMA_LAYOUT
    offs_q_m_pv = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, PV_WMMA_LAYOUT))
    offs_q_d_pv = gl.arange(
        0, HEAD_SIZE_PADDED, layout=gl.SliceLayout(0, PV_WMMA_LAYOUT)
    )
    query_pos_pv = q_block_local_idx * BLOCK_Q + offs_q_m_pv // num_queries_per_kv
    query_offset_0_pv = cur_batch_in_all_start_index + query_pos_pv
    query_offset_1_pv = (
        kv_head_idx * num_queries_per_kv + offs_q_m_pv % num_queries_per_kv
    )
    query_mask_0_pv = query_pos_pv < cur_batch_query_len
    query_mask_1_pv = query_offset_1_pv < num_query_heads
    query_mask_pv = query_mask_1_pv[:, None] & query_mask_0_pv[:, None]
    segm_output_offset = (
        query_offset_0_pv[:, None]
        * (num_query_heads * NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED)
        + query_offset_1_pv[:, None] * (NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED)
        + segm_idx * HEAD_SIZE_PADDED
        + offs_q_d_pv[None, :]
    )
    gl.amd.cdna4.buffer_store(
        stored_value=acc,
        ptr=segm_output_ptr,
        offsets=segm_output_offset,
        mask=dim_mask[None, :] & query_mask_pv,
    )

    # store segm_max and segm_expsum
    # L : shape = (BLOCK_M, ), layout = gl.SliceLayout(1, QK_WMMA_LAYOUT)
    # M : shape = (BLOCK_M, ), layout = gl.SliceLayout(1, QK_WMMA_LAYOUT)
    segm_offset = (
        query_offset_0_qk * (num_query_heads * NUM_SEGMENTS_PER_SEQ)
        + query_offset_1_qk * NUM_SEGMENTS_PER_SEQ
        + segm_idx
    )
    gl.amd.cdna4.buffer_store(
        stored_value=M,
        ptr=segm_max_ptr,
        offsets=segm_offset,
        mask=query_mask_0_qk & query_mask_1_qk,
    )
    gl.amd.cdna4.buffer_store(
        stored_value=L,
        ptr=segm_expsum_ptr,
        offsets=segm_offset,
        mask=query_mask_0_qk & query_mask_1_qk,
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
    acc = acc.to(output_ptr.type.element_ty)
    tl.store(output_ptr + output_offset, acc, mask=dim_mask)
