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


_kernel_unified_attention_2d_repr = make_kernel_repr(
    "kernel_unified_attention_2d",
    [
        "num_query_heads",
        "num_queries_per_kv",
        "BLOCK_SIZE",
        "TILE_SIZE",
        "HEAD_SIZE",
        "HEAD_SIZE_PADDED",
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
        # Block skipping. Both change codegen, so they belong in the name; the
        # threshold itself does not, because it is a runtime scalar.
        "ENABLE_BLOCK_SKIP",
        "PRELOAD_V",
        "DESCENDING_Q",
        "DYNAMIC_SCHED",
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
    # Block skipping: drop a K/V tile whose scores are all far enough below the
    # running maximum that their softmax weights cannot matter. Off by default,
    # and with ENABLE_BLOCK_SKIP=False every use below folds away at compile
    # time, so the dense path emits identical code.
    ENABLE_BLOCK_SKIP: tl.constexpr = False,  # bool
    PRELOAD_V: tl.constexpr = True,  # bool
    # log2(threshold), a RUNTIME scalar on purpose. As a constexpr it would key
    # the compilation cache, so sweeping thresholds would recompile per value.
    log2_threshold=0.0,  # float32
    # Scheduling, enabled together with block skipping and only then. Skipping
    # makes per-head cost depend on the data, and the grid pins each KV head to
    # one chiplet, so without these the launch waits on whichever head happened
    # to skip least. They are pure work assignment: identical arithmetic, and
    # the output is bitwise unchanged.
    DESCENDING_Q: tl.constexpr = False,  # bool
    DYNAMIC_SCHED: tl.constexpr = False,  # bool
    sched_counter_ptr=None,
    num_q_blocks=0,
    # Optional instrumentation: [tiles visited, tiles elided]. Off unless the
    # caller passes a buffer. The useful threshold is workload-dependent, so
    # this is how a caller checks that theirs elides anything at all.
    COUNT_SKIPS: tl.constexpr = False,  # bool
    skip_counter_ptr=None,
):
    tl.static_assert(
        not (SPLIT_UNMASKED_LOOP and SHUFFLED_KV_CACHE),
        "SPLIT_UNMASKED_LOOP is incompatible with SHUFFLED_KV_CACHE",
    )
    tl.static_assert(
        not COUNT_SKIPS or ENABLE_BLOCK_SKIP,
        "COUNT_SKIPS requires ENABLE_BLOCK_SKIP: with skipping off there is "
        "nothing to count",
    )

    if DYNAMIC_SCHED:
        # Claim a work item from a shared counter instead of deriving it from
        # this workgroup's id.
        #
        # The grid is (num_kv_heads, total_num_q_blocks) and Triton linearises
        # dim 0 fastest, so workgroup w = kv_head + num_kv_heads * q_block. The
        # hardware places workgroup w on chiplet w % 8, which at 8 KV heads on 8
        # chiplets means chiplet == kv_head exactly. Every workgroup for a head
        # is then stuck on one chiplet, and head-dependent skipping becomes
        # chiplet imbalance that the launch has to wait out.
        #
        # Head-major on purpose: consecutive tickets stay within one KV head for
        # num_q_blocks draws, so the workgroups resident at any instant are
        # nearly all on the same head and keep its K/V hot. The decode order is
        # not arbitrary -- issuing tickets q-major instead measures slower.
        #
        # The grid still launches exactly one workgroup per work item, so
        # tickets 0..N-1 are each drawn exactly once and no bounds check is
        # needed here.
        ticket = tl.atomic_add(sched_counter_ptr, 1, sem="relaxed", scope="gpu")
        kv_head_idx = ticket // num_q_blocks
        q_block_raw = ticket % num_q_blocks
    else:
        kv_head_idx = tl.program_id(0)
        q_block_raw = tl.program_id(1)

    if DESCENDING_Q:
        # Longest job first. Under causal masking query block i walks key tiles
        # 0..i, so work grows with the index, and the default order issues the
        # heaviest blocks LAST -- the worst case for a greedy dispatcher, which
        # ends up with one long block still running after everything else has
        # drained. Reversing costs nothing at compile time.
        q_block_global_idx = num_q_blocks - 1 - q_block_raw
    else:
        q_block_global_idx = q_block_raw

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

    if COUNT_SKIPS:
        # Accumulated in registers across BOTH tile loops and flushed once, at
        # the end. Doing the atomics per tile would be ~500x more of them onto
        # two addresses, which contends badly enough to distort the very timing
        # this instrumentation exists to explain.
        n_tiles_seen = 0
        n_tiles_elided = 0

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
                + offs_d[None, :] * stride_v_cache_3
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

            if PRELOAD_V:
                V_load = tl.load(
                    value_cache_ptr + v_offset,
                    mask=dim_mask[None, :],
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

            s_max = tl.max(S, axis=1)
            m_j = tl.maximum(M, s_max)

            if ENABLE_BLOCK_SKIP:
                # The decision is per TILE, not per row: a tile is dropped only
                # when every row agrees it is negligible. A tile that survives
                # the vote is then computed exactly as dense.
                #
                # M is the running maximum BEFORE this tile. Comparing against
                # the folded m_j is equivalent for thresholds below 1.0 but
                # degenerates to `0 < log2(threshold)` above it, skipping every
                # tile including the one holding the maximum.
                skip = (s_max - M) < log2_threshold
                all_skip = tl.sum(skip.to(tl.int32)) == BLOCK_M
                if COUNT_SKIPS:
                    n_tiles_seen += 1
                    n_tiles_elided += all_skip.to(tl.int32)
            else:
                all_skip = False

            m_j = tl.where(m_j > float("-inf"), m_j, 0.0)

            if not (ENABLE_BLOCK_SKIP and all_skip):
                P = tl.math.exp2(S - m_j[:, None])
                alpha = tl.math.exp2(M - m_j)
                l_j = tl.sum(P, axis=1)
                acc = acc * alpha[:, None]
                L = L * alpha + l_j
                M = m_j
                if not PRELOAD_V:
                    V_load = tl.load(
                        value_cache_ptr + v_offset,
                        mask=dim_mask[None, :],
                        other=0.0,
                        cache_modifier=KV_cache_modifier,
                    )
                    V = V_load.to(Q.dtype)
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
                + offs_d[None, :] * stride_v_cache_3
                + (seq_offset % BLOCK_SIZE)[:, None] * stride_v_cache_1
            )
            v_mask = dim_mask[None, :] & tile_mask[:, None]

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
        # Hoisted by default. With block skipping the load moves below the skip
        # decision, so a tile that is elided never touches V at all -- that is
        # where the bandwidth saving comes from, not from the elided matmul.
        if PRELOAD_V:
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
        s_max = tl.max(S, axis=1)
        m_j = tl.maximum(M, s_max)

        if ENABLE_BLOCK_SKIP:
            # Per-row votes, but a per-TILE decision: the tile is dropped only
            # if every row agrees its scores here are negligible against the
            # running maximum. A tile that survives the vote computes exactly
            # as dense below.
            #
            # M is the maximum BEFORE this tile. Comparing against the folded
            # m_j is equivalent below a threshold of 1.0, but above it reduces
            # to `0 < log2(threshold)` -- true for every row, so even the tile
            # holding the maximum is dropped.
            skip = (s_max - M) < log2_threshold
            all_skip = tl.sum(skip.to(tl.int32)) == BLOCK_M
            if COUNT_SKIPS:
                n_tiles_seen += 1
                n_tiles_elided += all_skip.to(tl.int32)
        else:
            all_skip = False

        # For sliding window there's a chance the max is -inf due to masking of
        # the entire row. In this case we need to set m_j 0 to avoid NaN
        m_j = tl.where(m_j > float("-inf"), m_j, 0.0)

        if not (ENABLE_BLOCK_SKIP and all_skip):
            # P : (BLOCK_M, TILE_SIZE)
            P = tl.math.exp2(S - m_j[:, None])

            # alpha : (BLOCK_M, )
            alpha = tl.math.exp2(M - m_j)

            # l_j : (BLOCK_M,)
            l_j = tl.sum(P, axis=1)

            # acc : (BLOCK_M, HEAD_SIZE_PADDED)
            acc = acc * alpha[:, None]

            # update constants
            L = L * alpha + l_j
            M = m_j

            if not PRELOAD_V:
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

            # acc : (BLOCK_M, HEAD_SIZE_PADDED)
            acc = tl.dot(P.to(V.dtype), V, acc=acc)

    if COUNT_SKIPS:
        # One flush per program, covering both loops. int32 is ample: this
        # counts TILES, and overflow would need ~2.1e9 of them.
        tl.atomic_add(skip_counter_ptr + 0, n_tiles_seen)
        tl.atomic_add(skip_counter_ptr + 1, n_tiles_elided)

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
        + offs_d[None, :]
    )

    tl.store(
        output_ptr + output_offset,
        acc,
        mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
    )


kernel_unified_attention_3d_repr = make_kernel_repr(
    "kernel_unified_attention_3d",
    [
        "num_query_heads",
        "num_queries_per_kv",
        "BLOCK_SIZE",
        "TILE_SIZE",
        "HEAD_SIZE",
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
):
    q_block_global_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)
    segm_idx = tl.program_id(2)

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

    # sequence len for this particular sequence
    seq_len = tl.load(seq_lens_ptr + seq_idx)

    # number of segments for this particular sequence
    num_segments = NUM_SEGMENTS_PER_SEQ
    tiles_per_segment = cdiv_fn(seq_len, num_segments * TILE_SIZE)

    if segm_idx * tiles_per_segment * TILE_SIZE >= seq_len:
        return

    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_SIZE_PADDED)
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
                + offs_d[None, :] * stride_v_cache_3
                + (seq_offset % BLOCK_SIZE)[:, None] * stride_v_cache_1
            )
            v_mask = dim_mask[None, :] & tile_mask[:, None]

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

    segm_output_offset = (
        query_offset_0[:, None].to(tl.int64)
        * (num_query_heads * NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED)
        + query_offset_1[:, None] * (NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED)
        + segm_idx * HEAD_SIZE_PADDED
        + tl.arange(0, HEAD_SIZE_PADDED)[None, :]
    )
    tl.store(
        segm_output_ptr + segm_output_offset,
        acc,
        mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
    )
    if NUM_SEGMENTS_PER_SEQ > 1:
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
        "NUM_SEGMENTS_PER_SEQ",
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
    query_start_len_ptr,  # [num_seqs+1]
    BLOCK_Q: tl.constexpr,  # int
    NUM_SEGMENTS_PER_SEQ: tl.constexpr,  # int
    FP8_MIN: tl.constexpr = float8_info.min,
    FP8_MAX: tl.constexpr = float8_info.max,
):
    query_token_idx = tl.program_id(0)
    query_head_idx = tl.program_id(1)

    out_scale = None
    if out_scale_ptr is not None:
        out_scale = 1 / tl.load(out_scale_ptr)

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
        offs_d = tl.arange(0, HEAD_SIZE_PADDED)
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

    if out_scale_ptr is not None:
        acc = acc * out_scale

    if output_ptr.type.element_ty.is_fp8():
        acc = tl.clamp(acc, FP8_MIN, FP8_MAX)

    # write result
    output_offset = (
        query_token_idx * output_stride_0
        + query_head_idx * output_stride_1
        + tl.arange(0, HEAD_SIZE_PADDED)
    )
    tl.store(
        output_ptr + output_offset, acc.to(output_ptr.type.element_ty), mask=dim_mask
    )
