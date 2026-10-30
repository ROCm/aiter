# The kernels in this file are adapted from vLLM:
# https://github.com/vllm-project/vllm/blob/main/vllm/attention/ops/triton_unified_attention.py
import torch
import triton
import triton.language as tl

from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr
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


# Length-aware KV split (LENGTH_AWARE_SPLIT), for batches with one query block
# per sequence (decode and short verify queries; the launcher checks max_seqlen_q
# against BLOCK_Q). A CUDA graph captures the grid with a loose max_seqlen_k, so
# the grid cannot see the real lengths. Instead the batch's KV tokens are cut into
# segments of
#     segment_tokens = cdiv(sum(seq_len), split_budget)
# and a sequence takes seq_len // segment_tokens of them, at least one and at
# most NUM_SEGMENTS_PER_SEQ. The counts sum to at most split_budget + num_seqs,
# so the attention grid is a flat list of that many (sequence, segment) work
# items per KV head; the reduce recomputes the same count to merge exactly the
# segments a sequence used.
# The KV heads of one token share cache lines, so the grid deals each run of
# ITEM_RUN items out once per KV head: workgroups go round-robin over the XCDs,
# and a run that is a multiple of the XCD count puts every head of an item on
# the same XCD (one L2).


@triton.jit
def batch_segment_tokens(lens, seq_lens_ptr, num_seqs, split_budget, SEQ_BLOCK):
    """segment_tokens for the batch; lens holds the first SEQ_BLOCK lengths.

    int32 is enough: the launcher keeps num_seqs * max_seqlen_k + split_budget
    below 2**31.
    """
    total = tl.sum(lens)
    for s0 in range(SEQ_BLOCK, num_seqs, SEQ_BLOCK):
        total += tl.sum(load_seq_lens(seq_lens_ptr, num_seqs, s0, SEQ_BLOCK))
    return tl.maximum(cdiv_fn(total, split_budget), 1)


@triton.jit
def seq_num_segments(seq_len, segment_tokens, NUM_SEGMENTS_PER_SEQ):
    return tl.minimum(tl.maximum(seq_len // segment_tokens, 1), NUM_SEGMENTS_PER_SEQ)


@triton.jit
def load_seq_lens(seq_lens_ptr, num_seqs, s0, SEQ_BLOCK: tl.constexpr):
    offs = s0 + tl.arange(0, SEQ_BLOCK)
    # callers pass int32 or int64 lengths; the split math runs in int32
    return tl.load(seq_lens_ptr + offs, mask=offs < num_seqs, other=0).to(tl.int32)


@triton.jit
def find_work_item(
    item,
    seq_lens_ptr,
    num_seqs,
    split_budget,
    NUM_SEGMENTS_PER_SEQ: tl.constexpr,
    SEQ_BLOCK: tl.constexpr,
):
    """Map a flat work item to (sequence, segment, its KV length, segment_tokens).

    The sequence index is num_seqs for an item past the batch's last segment.
    """
    lens = load_seq_lens(seq_lens_ptr, num_seqs, 0, SEQ_BLOCK)
    segment_tokens = batch_segment_tokens(
        lens, seq_lens_ptr, num_seqs, split_budget, SEQ_BLOCK
    )
    # one reduction per chunk: [sequences before | items before] in the high
    # word (both below 2**16 for the batches the launcher admits), the owning
    # sequence's KV length in the low word
    found = tl.zeros([], dtype=tl.int64)
    items = 0
    for s0 in range(0, num_seqs, SEQ_BLOCK):
        if s0 > 0:
            lens = load_seq_lens(seq_lens_ptr, num_seqs, s0, SEQ_BLOCK)
        valid = s0 + tl.arange(0, SEQ_BLOCK) < num_seqs
        segs = seq_num_segments(lens, segment_tokens, NUM_SEGMENTS_PER_SEQ)
        segs = tl.where(valid, segs, 0)
        end = items + tl.cumsum(segs, 0)
        before = end <= item
        owner = (end > item) & (end - segs <= item) & valid
        word = tl.where(before, (segs + 65536).to(tl.int64) << 32, 0)
        found += tl.sum(word + tl.where(owner, lens, 0).to(tl.int64))
        if s0 + SEQ_BLOCK < num_seqs:
            items += tl.sum(segs)
    high = (found >> 32).to(tl.int32)
    return high >> 16, item - (high & 65535), found.to(tl.int32), segment_tokens


_kernel_unified_attention_2d_repr = make_kernel_repr(
    "kernel_unified_attention_2d",
    [
        "num_query_heads",
        "num_queries_per_kv",
        "BLOCK_SIZE",
        "TILE_SIZE",
        "HEAD_SIZE",
        "HEAD_SIZE_PADDED",
        "V_HEAD_SIZE",
        "V_HEAD_SIZE_PADDED",
        "USE_ALIBI_SLOPES",
        "USE_QQ_BIAS",
        "USE_SOFTCAP",
        "USE_SINKS",
        "SLIDING_WINDOW",
        "BLOCK_Q",
        "BLOCK_M",
        "ALL_DECODE",
        "SHUFFLED_KV_CACHE",
        "SPLIT_UNMASKED_LOOP",
        "K_WIDTH",
    ],
)


@triton.jit(repr=_kernel_unified_attention_2d_repr)
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
    q_descale_ptr,  # float32
    k_descale_ptr,  # float32
    v_descale_ptr,  # float32
    out_scale_ptr,  # float32
    softcap,  # float32
    num_query_heads: tl.constexpr,  # int
    num_queries_per_kv: tl.constexpr,  # int
    block_table_stride: tl.int64,  # int
    query_stride_0: tl.int64,  # int
    query_stride_1: tl.int64,  # int, should be equal to head_size
    output_stride_0: tl.int64,  # int
    output_stride_1: tl.int64,  # int, should be equal to head_size
    qq_bias_stride_0: tl.int64,  # int
    BLOCK_SIZE: tl.constexpr,  # int
    TILE_SIZE: tl.constexpr,  # int must be power of 2
    HEAD_SIZE: tl.constexpr,  # int
    HEAD_SIZE_PADDED: tl.constexpr,  # int, must be power of 2
    V_HEAD_SIZE: tl.constexpr,  # int, value head size (may differ from HEAD_SIZE)
    V_HEAD_SIZE_PADDED: tl.constexpr,  # int, must be power of 2
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
    query_start_len_ptr,  # [num_seqs+1]
    BLOCK_Q: tl.constexpr,  # int
    num_seqs: tl.int32,
    BLOCK_M: tl.constexpr,  # int
    FP8_MIN: tl.constexpr = float8_info.min,
    FP8_MAX: tl.constexpr = float8_info.max,
    ALL_DECODE: tl.constexpr = False,  # bool
    SHUFFLED_KV_CACHE: tl.constexpr = False,  # bool
    SPLIT_UNMASKED_LOOP: tl.constexpr = False,  # bool
    K_WIDTH: tl.constexpr = 0,  # int
):
    # SPLIT_UNMASKED_LOOP does not support SHUFFLED_KV_CACHE or SLIDING_WINDOW.
    tl.static_assert(
        not (SPLIT_UNMASKED_LOOP and SHUFFLED_KV_CACHE),
        "SPLIT_UNMASKED_LOOP does not support SHUFFLED_KV_CACHE",
    )
    tl.static_assert(
        not (SPLIT_UNMASKED_LOOP and SLIDING_WINDOW > 0),
        "SPLIT_UNMASKED_LOOP does not support sliding-window attention",
    )

    kv_head_idx = tl.program_id(0)
    q_block_global_idx = tl.program_id(1)

    # needed to use exp2 (exp2 -> exp conversion)
    RCP_LN2 = 1.4426950408889634
    qk_scale = scale * RCP_LN2

    if ALL_DECODE:
        seq_idx = q_block_global_idx
        q_block_local_idx: tl.int32 = 0
        cur_batch_query_len: tl.int32 = 1
        cur_batch_in_all_start_index: tl.int32 = q_block_global_idx
    else:
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
    offs_vd = tl.arange(0, V_HEAD_SIZE_PADDED)
    offs_t = tl.arange(0, TILE_SIZE)
    query_pos = q_block_local_idx * BLOCK_Q + offs_m // num_queries_per_kv

    offs_shfl = None
    if SHUFFLED_KV_CACHE:
        offs_shfl = tl.arange(0, TILE_SIZE * HEAD_SIZE_PADDED)

    query_offset_0 = cur_batch_in_all_start_index + query_pos
    query_offset_1 = kv_head_idx * num_queries_per_kv + offs_m % num_queries_per_kv
    query_offset = (
        query_offset_0[:, None] * query_stride_0
        + query_offset_1[:, None] * query_stride_1
        + offs_d[None, :]
    )

    if HEAD_SIZE_PADDED != HEAD_SIZE:
        dim_mask = offs_d < HEAD_SIZE
    else:
        dim_mask = tl.full((1,), 1, dtype=tl.int1)
    if V_HEAD_SIZE_PADDED != V_HEAD_SIZE:
        v_dim_mask = offs_vd < V_HEAD_SIZE
    else:
        v_dim_mask = tl.full((1,), 1, dtype=tl.int1)
    query_mask_0 = query_pos < cur_batch_query_len
    query_mask_1 = query_offset_1 < num_query_heads

    if ALL_DECODE or BLOCK_M >= num_query_heads:
        Q_cache_modifier: tl.constexpr = ".cg"
    else:
        Q_cache_modifier: tl.constexpr = ""
    # Q : (BLOCK_M, HEAD_SIZE_PADDED)
    Q = tl.load(
        query_ptr + query_offset,
        mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
        other=0.0,
        cache_modifier=Q_cache_modifier,
    )

    block_table_offset = seq_idx * block_table_stride

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
    acc = tl.zeros([BLOCK_M, V_HEAD_SIZE_PADDED], dtype=tl.float32)

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
    if q_descale_ptr is not None:
        q_descale = tl.load(q_descale_ptr)
        qk_scale = qk_scale * q_descale
    else:
        q_descale = None
    if k_descale_ptr is not None and v_descale_ptr is not None:
        k_descale = tl.load(k_descale_ptr)
        v_descale = tl.load(v_descale_ptr)
        qk_scale = qk_scale * k_descale
    else:
        k_descale = None
        v_descale = None
    KV_cache_modifier: tl.constexpr = ".cg" if ALL_DECODE else ""

    masked_tile_start = tile_start
    if SPLIT_UNMASKED_LOOP:
        min_query_key_limit = context_len + q_block_local_idx * BLOCK_Q + 1
        unmasked_tile_end = tl.minimum(min_query_key_limit // TILE_SIZE, tile_end)
        unmasked_tile_end = tl.maximum(unmasked_tile_end, tile_start)

        for j in range(tile_start, unmasked_tile_end):
            seq_offset = j * TILE_SIZE + offs_t

            physical_block_idx = tl.load(
                block_tables_ptr + block_table_offset + seq_offset // BLOCK_SIZE
            ).to(tl.int64)
            v_offset = (
                physical_block_idx[:, None] * stride_v_cache_0
                + kv_head_idx * stride_v_cache_2
                + offs_vd[None, :] * stride_v_cache_3
                + (seq_offset % BLOCK_SIZE)[:, None] * stride_v_cache_1
            )
            k_offset = (
                physical_block_idx[None, :] * stride_k_cache_0
                + kv_head_idx * stride_k_cache_2
                + offs_d[:, None] * stride_k_cache_3
                + (seq_offset % BLOCK_SIZE)[None, :] * stride_k_cache_1
            )

            K_load = tl.load(
                key_cache_ptr + k_offset,
                mask=dim_mask[:, None],
                other=0.0,
                cache_modifier=KV_cache_modifier,
            )
            K = K_load.to(Q.dtype)

            V_load = tl.load(
                value_cache_ptr + v_offset,
                mask=v_dim_mask[None, :],
                other=0.0,
                cache_modifier=KV_cache_modifier,
            )
            V = V_load.to(Q.dtype)

            S = qk_scale * tl.dot(Q, K)

            if USE_SOFTCAP:
                S = apply_softcap(S, softcap) * RCP_LN2

            if USE_ALIBI_SLOPES:
                S += alibi_slope[:, None] * (seq_offset - context_len) * RCP_LN2

            if USE_QQ_BIAS:
                key_rel_pos = seq_offset - context_len
                is_query_key = key_rel_pos >= 0 and key_rel_pos < qq_bias_stride_0
                qq_bias = tl.load(
                    qq_bias_row_ptrs + key_rel_pos[None, :],
                    mask=is_query_key[None, :],
                    other=0.0,
                )
                S += qq_bias * RCP_LN2

            m_j = tl.maximum(M, tl.max(S, axis=1))
            m_j = tl.where(m_j > float("-inf"), m_j, 0.0)
            P = tl.math.exp2(S - m_j[:, None])
            l_j = tl.sum(P, axis=1)
            alpha = tl.math.exp2(M - m_j)
            acc = acc * alpha[:, None]
            L = L * alpha + l_j
            M = m_j
            acc = tl.dot(P.to(V.dtype), V, acc=acc)

        masked_tile_start = unmasked_tile_end

    for j in range(masked_tile_start, tile_end):
        seq_offset = j * TILE_SIZE + offs_t
        # to reduce the masking effect when not needed
        if TILE_SIZE == BLOCK_SIZE:
            tile_mask = tl.full((1,), 1, dtype=tl.int1)
        else:
            tile_mask = seq_offset < max_seq_prefix_len

        k_mask = None
        v_mask = None
        other = None
        if SHUFFLED_KV_CACHE:
            physical_block_idx_shfl = tl.load(
                block_tables_ptr + block_table_offset + j
            ).to(tl.int64)
            k_offset = (
                physical_block_idx_shfl * stride_k_cache_0
                + kv_head_idx * stride_k_cache_1
                + offs_shfl
            )

            v_offset = (
                physical_block_idx_shfl * stride_v_cache_0
                + kv_head_idx * stride_v_cache_1
                + offs_shfl
            )
        else:
            physical_block_idx = tl.load(
                block_tables_ptr + block_table_offset + seq_offset // BLOCK_SIZE
            ).to(tl.int64)

            v_offset = (
                physical_block_idx[:, None] * stride_v_cache_0
                + kv_head_idx * stride_v_cache_2
                + offs_vd[None, :] * stride_v_cache_3
                + (seq_offset % BLOCK_SIZE)[:, None] * stride_v_cache_1
            )
            v_mask = v_dim_mask[None, :] & tile_mask[:, None]

            k_offset = (
                physical_block_idx[None, :] * stride_k_cache_0
                + kv_head_idx * stride_k_cache_2
                + offs_d[:, None] * stride_k_cache_3
                + (seq_offset % BLOCK_SIZE)[None, :] * stride_k_cache_1
            )
            k_mask = dim_mask[:, None] & tile_mask[None, :]
            other = 0.0

        # K : (HEAD_SIZE, TILE_SIZE)
        K_load = tl.load(
            key_cache_ptr + k_offset,
            mask=k_mask,
            other=other,
            cache_modifier=KV_cache_modifier,
        )

        K = K_load.to(Q.dtype)
        if SHUFFLED_KV_CACHE:
            K = (
                K.reshape(
                    HEAD_SIZE_PADDED // K_WIDTH,
                    TILE_SIZE,
                    K_WIDTH,
                )
                .permute(1, 0, 2)
                .reshape(TILE_SIZE, HEAD_SIZE_PADDED)
                .trans(1, 0)
            )

        # V : (TILE_SIZE, HEAD_SIZE)
        V_load = tl.load(
            value_cache_ptr + v_offset,
            mask=v_mask,
            other=other,
            cache_modifier=KV_cache_modifier,
        )

        V = V_load.to(Q.dtype)
        if SHUFFLED_KV_CACHE:
            V = (
                V.reshape(
                    TILE_SIZE // K_WIDTH,
                    HEAD_SIZE_PADDED,
                    K_WIDTH,
                )
                .permute(0, 2, 1)
                .reshape(TILE_SIZE, HEAD_SIZE_PADDED)
            )

        # S : (BLOCK_M, TILE_SIZE)
        # qk_scale = scale * RCP_LN2 (log_2 e) so that we can use exp2 later
        S = qk_scale * tl.dot(Q, K)

        if USE_SOFTCAP:
            # softcap here uses exp2 and consumes RCP_LN2 conversion.
            # multiply by RCP_LN2 again to be used in later exp2
            S = apply_softcap(S, softcap) * RCP_LN2
        seq_mask = seq_offset[None, :] < context_len + query_pos[:, None] + 1

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

        # update constants
        L = L * alpha + l_j
        M = m_j

        # acc : (BLOCK_M, HEAD_SIZE_PADDED)
        acc = tl.dot(P.to(V.dtype), V, acc=acc)

    # epilogue
    # This helps the compiler do Newton Raphson on l_i vs on acc which is much larger.
    if v_descale is not None:
        one_over_L = v_descale / L[:, None]
    else:
        one_over_L = 1.0 / L[:, None]
    acc = acc * one_over_L
    if out_scale_ptr is not None:
        acc = acc / tl.load(out_scale_ptr)

    if output_ptr.type.element_ty.is_fp8():
        acc = tl.clamp(acc, FP8_MIN, FP8_MAX)

    output_offset = (
        query_offset_0[:, None] * output_stride_0
        + query_offset_1[:, None] * output_stride_1
        + offs_vd[None, :]
    )

    tl.store(
        output_ptr + output_offset,
        acc,
        mask=v_dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
    )


kernel_unified_attention_3d_repr = make_kernel_repr(
    "kernel_unified_attention_3d",
    [
        "num_query_heads",
        "num_queries_per_kv",
        "BLOCK_SIZE",
        "TILE_SIZE",
        "HEAD_SIZE",
        "V_HEAD_SIZE",
        "NUM_SEGMENTS_PER_SEQ",
        "num_warps",
        "waves_per_eu",
        "num_stages",
        "ALL_DECODE",
        "SHUFFLED_KV_CACHE",
        "IS_Q_FP8",
        "IS_KV_FP8",
    ],
)


@triton.jit(repr=kernel_unified_attention_3d_repr)
def kernel_unified_attention_3d(
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
    q_descale_ptr,  # float32
    k_descale_ptr,  # float32
    v_descale_ptr,  # float32
    out_scale_ptr,  # float32
    softcap,  # float32
    num_query_heads: tl.constexpr,  # int
    num_queries_per_kv: tl.constexpr,  # int
    block_table_stride: tl.int64,  # int
    query_stride_0: tl.int64,  # int
    query_stride_1: tl.int64,  # int, should be equal to head_size
    qq_bias_stride_0: tl.int64,  # int
    BLOCK_SIZE: tl.constexpr,  # int
    TILE_SIZE: tl.constexpr,  # int, must be power of 2
    HEAD_SIZE: tl.constexpr,  # int
    HEAD_SIZE_PADDED: tl.constexpr,  # int, must be power of 2
    V_HEAD_SIZE: tl.constexpr,  # int, value head size (may differ from HEAD_SIZE)
    V_HEAD_SIZE_PADDED: tl.constexpr,  # int, must be power of 2
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
    query_start_len_ptr,  # [num_seqs+1]
    BLOCK_Q: tl.constexpr,  # int
    num_seqs: tl.int32,
    BLOCK_M: tl.constexpr,  # int
    num_warps: tl.constexpr,  # int
    waves_per_eu: tl.constexpr,  # int
    num_stages: tl.constexpr,  # int
    NUM_SEGMENTS_PER_SEQ: tl.constexpr,  # int
    ALL_DECODE: tl.constexpr = False,  # bool
    SHUFFLED_KV_CACHE: tl.constexpr = False,  # bool
    K_WIDTH: tl.constexpr = 0,  # int
    IS_Q_FP8: tl.constexpr = False,  # bool
    IS_KV_FP8: tl.constexpr = False,  # bool
    split_budget=1,  # int, segments per KV head the batch shares (LENGTH_AWARE_SPLIT)
    LENGTH_AWARE_SPLIT: tl.constexpr = False,  # bool, grid axis 0 = work items x heads
    SEQ_BLOCK: tl.constexpr = 256,  # int, sequences per step over seq_lens
    ITEM_RUN: tl.constexpr = 1,  # int, work items per KV head in a row (LENGTH_AWARE_SPLIT)
    num_tokens=1,  # int, query tokens: the partials' slot stride (LENGTH_AWARE_SPLIT)
):
    q_block_global_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)
    segm_idx = tl.program_id(2)

    # needed to use exp2 (exp2 -> exp conversion)
    RCP_LN2 = 1.4426950408889634
    qk_scale = scale * RCP_LN2

    if LENGTH_AWARE_SPLIT:
        # one q block per sequence; axis 0 enumerates (work item, KV head) pairs
        NUM_KV_HEADS: tl.constexpr = num_query_heads // num_queries_per_kv
        kv_head_idx = (q_block_global_idx // ITEM_RUN) % NUM_KV_HEADS
        item = (q_block_global_idx // (ITEM_RUN * NUM_KV_HEADS)) * ITEM_RUN + (
            q_block_global_idx % ITEM_RUN
        )
        seq_idx, segm_idx, seq_len, segment_tokens = find_work_item(
            item, seq_lens_ptr, num_seqs, split_budget, NUM_SEGMENTS_PER_SEQ, SEQ_BLOCK
        )
        if seq_idx >= num_seqs:
            return
        q_block_local_idx: tl.int32 = 0
        if ALL_DECODE:
            cur_batch_query_len: tl.int32 = 1
            cur_batch_in_all_start_index = seq_idx
        else:
            cur_batch_in_all_start_index = tl.load(query_start_len_ptr + seq_idx)
            cur_batch_query_len = (
                tl.load(query_start_len_ptr + seq_idx + 1)
                - cur_batch_in_all_start_index
            )
            if cur_batch_query_len == 0:
                return
    elif ALL_DECODE:
        seq_idx = q_block_global_idx
        q_block_local_idx: tl.int32 = 0
        cur_batch_query_len: tl.int32 = 1
        cur_batch_in_all_start_index: tl.int32 = q_block_global_idx
    else:
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
    if not LENGTH_AWARE_SPLIT:
        seq_len = tl.load(seq_lens_ptr + seq_idx)

    # number of segments for this particular sequence
    if LENGTH_AWARE_SPLIT:
        num_segments = seq_num_segments(seq_len, segment_tokens, NUM_SEGMENTS_PER_SEQ)
    else:
        num_segments = NUM_SEGMENTS_PER_SEQ
    tiles_per_segment = cdiv_fn(seq_len, num_segments * TILE_SIZE)

    if segm_idx * tiles_per_segment * TILE_SIZE >= seq_len:
        return

    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_SIZE_PADDED)
    offs_vd = tl.arange(0, V_HEAD_SIZE_PADDED)
    offs_t = tl.arange(0, TILE_SIZE)

    offs_shfl = None
    if SHUFFLED_KV_CACHE:
        offs_shfl = tl.arange(0, TILE_SIZE * HEAD_SIZE_PADDED)

    query_pos = q_block_local_idx * BLOCK_Q + offs_m // num_queries_per_kv

    query_offset_0 = cur_batch_in_all_start_index + query_pos
    query_offset_1 = kv_head_idx * num_queries_per_kv + offs_m % num_queries_per_kv
    query_offset = (
        query_offset_0[:, None] * query_stride_0
        + query_offset_1[:, None] * query_stride_1
        + offs_d[None, :]
    )

    if HEAD_SIZE_PADDED != HEAD_SIZE:
        dim_mask = offs_d < HEAD_SIZE
    else:
        dim_mask = tl.full((1,), 1, dtype=tl.int1)
    if V_HEAD_SIZE_PADDED != V_HEAD_SIZE:
        v_dim_mask = offs_vd < V_HEAD_SIZE
    else:
        v_dim_mask = tl.full((1,), 1, dtype=tl.int1)
    query_mask_0 = query_pos < cur_batch_query_len
    query_mask_1 = query_offset_1 < num_query_heads

    # Q : (BLOCK_M, HEAD_SIZE_PADDED)
    Q = tl.load(
        query_ptr + query_offset,
        mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
        other=0.0,
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
    acc = tl.zeros([BLOCK_M, V_HEAD_SIZE_PADDED], dtype=tl.float32)

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
    if q_descale_ptr is not None:
        q_descale = tl.load(q_descale_ptr)
        qk_scale = qk_scale * q_descale
    else:
        q_descale = None

    if k_descale_ptr is not None:
        k_scale = tl.load(k_descale_ptr)
        qk_scale = qk_scale * k_scale
    else:
        k_scale = None

    out_factor: tl.float32 = 1.0
    if v_descale_ptr is not None:
        out_factor = tl.load(v_descale_ptr)

    if out_scale_ptr is not None:
        out_factor = out_factor / tl.load(out_scale_ptr)

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

        k_mask = None
        v_mask = None
        other = None
        if SHUFFLED_KV_CACHE:
            physical_block_idx_shfl = tl.load(
                block_tables_ptr + block_table_offset + j
            ).to(tl.int64)
            k_offset = (
                physical_block_idx_shfl * stride_k_cache_0
                + kv_head_idx * stride_k_cache_1
                + offs_shfl
            )

            v_offset = (
                physical_block_idx_shfl * stride_v_cache_0
                + kv_head_idx * stride_v_cache_1
                + offs_shfl
            )
        else:
            physical_block_idx = tl.load(
                block_tables_ptr + block_table_offset + seq_offset // BLOCK_SIZE
            ).to(tl.int64)

            v_offset = (
                physical_block_idx[:, None] * stride_v_cache_0
                + kv_head_idx * stride_v_cache_2
                + offs_vd[None, :] * stride_v_cache_3
                + (seq_offset % BLOCK_SIZE)[:, None] * stride_v_cache_1
            )
            v_mask = v_dim_mask[None, :] & tile_mask[:, None]

            k_offset = (
                physical_block_idx[None, :] * stride_k_cache_0
                + kv_head_idx * stride_k_cache_2
                + offs_d[:, None] * stride_k_cache_3
                + (seq_offset % BLOCK_SIZE)[None, :] * stride_k_cache_1
            )
            k_mask = dim_mask[:, None] & tile_mask[None, :]
            other = 0.0

        # K : (HEAD_SIZE, TILE_SIZE)
        K_load = tl.load(
            key_cache_ptr + k_offset,
            mask=k_mask,
            other=other,
            cache_modifier=KV_cache_modifier,
        )

        K = K_load.to(Q.dtype)
        if SHUFFLED_KV_CACHE:
            K = (
                K.reshape(
                    HEAD_SIZE_PADDED // K_WIDTH,
                    TILE_SIZE,
                    K_WIDTH,
                )
                .permute(1, 0, 2)
                .reshape(TILE_SIZE, HEAD_SIZE_PADDED)
                .trans(1, 0)
            )

        # V : (TILE_SIZE, HEAD_SIZE)
        V_load = tl.load(
            value_cache_ptr + v_offset,
            mask=v_mask,
            other=other,
            cache_modifier=KV_cache_modifier,
        )

        V = V_load.to(Q.dtype)
        if SHUFFLED_KV_CACHE:
            V = (
                V.reshape(
                    TILE_SIZE // K_WIDTH,
                    HEAD_SIZE_PADDED,
                    K_WIDTH,
                )
                .permute(0, 2, 1)
                .reshape(TILE_SIZE, HEAD_SIZE_PADDED)
            )

        seq_mask = seq_offset[None, :] < context_len + query_pos[:, None] + 1

        # S : (BLOCK_M, TILE_SIZE)
        # qk_scale = scale * RCP_LN2 (log_2 e) so that we can use exp2 later
        S = qk_scale * tl.dot(Q, K)

        if USE_SOFTCAP:
            # softcap here uses exp2 and consumes RCP_LN2 conversion.
            # multiply by RCP_LN2 again to be used in later exp2
            S = apply_softcap(S, softcap) * RCP_LN2

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

        # update constants
        L = L * alpha + l_j
        M = m_j

        # acc : (BLOCK_M, HEAD_SIZE_PADDED)
        acc = tl.dot(P.to(V.dtype), V, acc=acc)

    acc = acc * out_factor
    if NUM_SEGMENTS_PER_SEQ == 1:
        one_over_L = 1.0 / L[:, None]
        acc = acc * one_over_L

    if LENGTH_AWARE_SPLIT:
        # slot-major partials [segment, token, head]: the slots a batch uses fill
        # one dense region instead of the head of every token's slot block
        segm_offset = (
            segm_idx.to(tl.int64) * num_tokens + query_offset_0
        ) * num_query_heads + query_offset_1
        segm_output_offset = (
            segm_offset[:, None] * V_HEAD_SIZE_PADDED + offs_vd[None, :]
        )
    else:
        segm_output_offset = (
            query_offset_0[:, None].to(tl.int64)
            * (num_query_heads * NUM_SEGMENTS_PER_SEQ * V_HEAD_SIZE_PADDED)
            + query_offset_1[:, None] * (NUM_SEGMENTS_PER_SEQ * V_HEAD_SIZE_PADDED)
            + segm_idx * V_HEAD_SIZE_PADDED
            + offs_vd[None, :]
        )
    tl.store(
        segm_output_ptr + segm_output_offset,
        acc,
        mask=v_dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
    )
    if NUM_SEGMENTS_PER_SEQ > 1:
        if not LENGTH_AWARE_SPLIT:
            segm_offset = (
                query_offset_0.to(tl.int64) * (num_query_heads * NUM_SEGMENTS_PER_SEQ)
                + query_offset_1 * NUM_SEGMENTS_PER_SEQ
                + segm_idx
            )
        tl.store(segm_max_ptr + segm_offset, M, mask=query_mask_0 & query_mask_1)
        tl.store(segm_expsum_ptr + segm_offset, L, mask=query_mask_0 & query_mask_1)


_reduce_segments_repr = make_kernel_repr(
    "reduce_segments",
    [
        "num_query_heads",
        "TILE_SIZE",
        "HEAD_SIZE",
        "V_HEAD_SIZE",
        "NUM_SEGMENTS_PER_SEQ",
        "SEGMENT_BLOCK",
        "BLOCK_D",
    ],
)


@triton.jit(repr=_reduce_segments_repr)
def reduce_segments(
    output_ptr,  # [num_tokens, num_query_heads, head_size]
    segm_output_ptr,
    # [num_tokens, num_query_heads, max_num_segments, head_size]
    segm_max_ptr,  # [num_tokens, num_query_heads, max_num_segments]
    segm_expsum_ptr,  # [num_tokens, num_query_heads, max_num_segments]
    seq_lens_ptr,  # [num_seqs]
    num_seqs,  # int
    num_query_heads: tl.constexpr,  # int
    out_scale_ptr,  # float32
    output_stride_0: tl.int64,  # int
    output_stride_1: tl.int64,  # int, should be equal to head_size
    block_table_stride: tl.int64,  # int
    TILE_SIZE: tl.constexpr,  # int
    HEAD_SIZE: tl.constexpr,  # int, must be power of 2
    HEAD_SIZE_PADDED: tl.constexpr,  # int, must be power of 2
    V_HEAD_SIZE: tl.constexpr,  # int, value head size (may differ from HEAD_SIZE)
    V_HEAD_SIZE_PADDED: tl.constexpr,  # int, must be power of 2
    query_start_len_ptr,  # [num_seqs+1]
    BLOCK_Q: tl.constexpr,  # int
    NUM_SEGMENTS_PER_SEQ: tl.constexpr,  # int
    FP8_MIN: tl.constexpr = float8_info.min,
    FP8_MAX: tl.constexpr = float8_info.max,
    SEGMENT_BLOCK: tl.constexpr = 0,  # segments per merge step, 0 = all at once
    BLOCK_D: tl.constexpr = 0,  # head-dim slice per program (grid axis 2), 0 = all
    split_budget=1,  # int, as passed to kernel_unified_attention_3d
    LENGTH_AWARE_SPLIT: tl.constexpr = False,  # bool, see seq_num_segments
    SEQ_BLOCK: tl.constexpr = 256,  # int, sequences per step over seq_lens
    LENGTH_AWARE_SEGMENT_BLOCK: tl.constexpr = 16,  # int, used slots merged per step
):
    query_token_idx = tl.program_id(0)
    query_head_idx = tl.program_id(1)

    out_scale = None
    if out_scale_ptr is not None:
        out_scale = 1 / tl.load(out_scale_ptr)

    if LENGTH_AWARE_SPLIT:
        # independent of this program's sequence: issue it first
        segment_tokens = batch_segment_tokens(
            load_seq_lens(seq_lens_ptr, num_seqs, 0, SEQ_BLOCK),
            seq_lens_ptr,
            num_seqs,
            split_budget,
            SEQ_BLOCK,
        )

    seq_idx = find_seq_idx(
        query_start_len_ptr, query_token_idx, num_seqs, BLOCK_Q, False
    )

    # sequence len for this particular sequence
    seq_len = tl.load(seq_lens_ptr + seq_idx)

    # number of segments for this particular sequence
    if LENGTH_AWARE_SPLIT:
        num_segments = seq_num_segments(seq_len, segment_tokens, NUM_SEGMENTS_PER_SEQ)
    else:
        num_segments = NUM_SEGMENTS_PER_SEQ
    tiles_per_segment = cdiv_fn(seq_len, num_segments * TILE_SIZE)

    # create masks for subsequent loads
    act_num_segments = cdiv_fn(seq_len, tiles_per_segment * TILE_SIZE)
    single_tile: tl.constexpr = (
        SEGMENT_BLOCK == 0 or SEGMENT_BLOCK >= NUM_SEGMENTS_PER_SEQ
    )
    if single_tile and not LENGTH_AWARE_SPLIT:
        segm_mask = tl.arange(0, NUM_SEGMENTS_PER_SEQ) < tl.full(
            [NUM_SEGMENTS_PER_SEQ], act_num_segments, dtype=tl.int32
        )

    # BLOCK_D splits the head dim across grid axis 2, so a small batch still
    # launches enough programs to keep the reduce bandwidth- not latency-bound.
    if BLOCK_D == 0:
        D_TILE: tl.constexpr = V_HEAD_SIZE_PADDED
        offs_vd = tl.arange(0, V_HEAD_SIZE_PADDED)
    else:
        D_TILE: tl.constexpr = BLOCK_D
        offs_vd = tl.program_id(2) * BLOCK_D + tl.arange(0, BLOCK_D)
    if V_HEAD_SIZE_PADDED != V_HEAD_SIZE:
        v_dim_mask = offs_vd < V_HEAD_SIZE
    else:
        v_dim_mask = tl.full((1,), 1, dtype=tl.int1)

    if LENGTH_AWARE_SPLIT:
        # slot-major partials (see kernel_unified_attention_3d), merged a block of
        # used slots at a time, so the cost follows the split, not the slot count
        segm_row = query_token_idx.to(tl.int64) * num_query_heads + query_head_idx
        slot_stride = tl.num_programs(0).to(tl.int64) * num_query_heads
        offs_s = tl.arange(0, LENGTH_AWARE_SEGMENT_BLOCK)
        overall_max = float("-inf")
        overall_expsum = 0.0
        acc_sum = tl.zeros([D_TILE], dtype=tl.float32)
        for s0 in range(0, act_num_segments, LENGTH_AWARE_SEGMENT_BLOCK):
            segs = s0 + offs_s
            segm_mask = segs < act_num_segments
            segm_offset = segs.to(tl.int64) * slot_stride + segm_row
            segm_max = tl.load(
                segm_max_ptr + segm_offset, mask=segm_mask, other=float("-inf")
            )
            segm_expsum = tl.load(
                segm_expsum_ptr + segm_offset, mask=segm_mask, other=0.0
            )
            segm_output = tl.load(
                segm_output_ptr
                + segm_offset[:, None] * V_HEAD_SIZE_PADDED
                + offs_vd[None, :],
                mask=segm_mask[:, None] & v_dim_mask[None, :],
                other=0.0,
            )
            block_max = tl.maximum(overall_max, tl.max(segm_max))
            block_max = tl.where(block_max > float("-inf"), block_max, 0.0)
            rescale = tl.math.exp2(overall_max - block_max)
            weight = tl.math.exp2(segm_max - block_max)
            overall_expsum = overall_expsum * rescale + tl.sum(segm_expsum * weight)
            acc_sum = acc_sum * rescale + tl.sum(segm_output * weight[:, None], axis=0)
            overall_max = block_max
    elif single_tile:
        # load segment maxima
        segm_offset = (
            query_token_idx.to(tl.int64) * (num_query_heads * NUM_SEGMENTS_PER_SEQ)
            + query_head_idx * NUM_SEGMENTS_PER_SEQ
            + tl.arange(0, NUM_SEGMENTS_PER_SEQ)
        )
        segm_max = tl.load(
            segm_max_ptr + segm_offset, mask=segm_mask, other=float("-inf")
        )
        overall_max = tl.max(segm_max)

        # load and rescale segment exp sums
        segm_expsum = tl.load(segm_expsum_ptr + segm_offset, mask=segm_mask, other=0.0)
        segm_expsum = segm_expsum * tl.math.exp2(segm_max - overall_max)
        overall_expsum = tl.sum(segm_expsum)

        # load, rescale, and add segment attention outputs
        segm_output_offset = (
            query_token_idx.to(tl.int64)
            * (num_query_heads * NUM_SEGMENTS_PER_SEQ * V_HEAD_SIZE_PADDED)
            + query_head_idx * (NUM_SEGMENTS_PER_SEQ * V_HEAD_SIZE_PADDED)
            + tl.arange(0, NUM_SEGMENTS_PER_SEQ)[:, None] * V_HEAD_SIZE_PADDED
            + offs_vd[None, :]
        )
        segm_output = tl.load(
            segm_output_ptr + segm_output_offset,
            mask=segm_mask[:, None] & v_dim_mask[None, :],
            other=0.0,
        )
        segm_output *= tl.math.exp2(segm_max - overall_max)[:, None]
        acc_sum = tl.sum(segm_output, axis=0)
    else:
        segm_base = (
            query_token_idx.to(tl.int64) * (num_query_heads * NUM_SEGMENTS_PER_SEQ)
            + query_head_idx * NUM_SEGMENTS_PER_SEQ
        )
        segm_output_base = query_token_idx.to(tl.int64) * (
            num_query_heads * NUM_SEGMENTS_PER_SEQ * V_HEAD_SIZE_PADDED
        ) + query_head_idx * (NUM_SEGMENTS_PER_SEQ * V_HEAD_SIZE_PADDED)
        # Pass 1: the global max and exp-sum from the [S] statistics. Pass 2 then
        # weights each output tile by a fixed factor, so its loads carry no
        # loop-carried rescale and can be issued back to back.
        offs_all = tl.arange(0, NUM_SEGMENTS_PER_SEQ)
        all_mask = offs_all < act_num_segments
        segm_max_all = tl.load(
            segm_max_ptr + segm_base + offs_all, mask=all_mask, other=float("-inf")
        )
        overall_max = tl.max(segm_max_all)
        segm_expsum_all = tl.load(
            segm_expsum_ptr + segm_base + offs_all, mask=all_mask, other=0.0
        )
        overall_expsum = tl.sum(
            segm_expsum_all * tl.math.exp2(segm_max_all - overall_max)
        )

        offs_s = tl.arange(0, SEGMENT_BLOCK)
        acc_sum = tl.zeros([D_TILE], dtype=tl.float32)
        for s0 in range(0, act_num_segments, SEGMENT_BLOCK):
            segs = s0 + offs_s
            segm_mask = segs < act_num_segments
            weight = tl.math.exp2(
                tl.load(
                    segm_max_ptr + segm_base + segs,
                    mask=segm_mask,
                    other=float("-inf"),
                )
                - overall_max
            )
            segm_output = tl.load(
                segm_output_ptr
                + segm_output_base
                + segs[:, None] * V_HEAD_SIZE_PADDED
                + offs_vd[None, :],
                mask=segm_mask[:, None] & v_dim_mask[None, :],
                other=0.0,
            )
            acc_sum += tl.sum(segm_output * weight[:, None], axis=0)
    # safely divide by overall_expsum, returning 0.0 if overall_expsum is 0
    acc = tl.where(overall_expsum == 0.0, 0.0, acc_sum / overall_expsum)

    if out_scale_ptr is not None:
        acc = acc * out_scale

    if output_ptr.type.element_ty.is_fp8():
        acc = tl.clamp(acc, FP8_MIN, FP8_MAX)

    # write result
    output_offset = (
        query_token_idx * output_stride_0 + query_head_idx * output_stride_1 + offs_vd
    )
    tl.store(
        output_ptr + output_offset,
        acc.to(output_ptr.type.element_ty),
        mask=v_dim_mask,
    )
