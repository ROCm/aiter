
@gluon.jit
def _paged_attn_decode_v2_w_dot_kernel_reshape_loop_qk_gluon(
    exp_sums_ptr,       # [num_seqs, num_kv_heads, max_parts, q_grp_sz]
    max_logits_ptr,     # [num_seqs, num_kv_heads, max_parts, q_grp_sz]
    logits_ptr,         # [num_seqs, num_kv_heads, max_parts, q_grp_sz, head_sz]
    q_ptr,              # [num_seqs, num_kv_heads * query_grp_sz, head_sz]
    k_cache_ptr,        # [num_blks, num_kv_heads, head_sz/x, kv_blk_sz, x]
    v_cache_ptr,        # [num_blks, num_kv_heads, head_sz, kv_blk_sz]
    sink_ptr,  # [num_query_heads]
    blk_tables_ptrs,    # [num_seqs, max_num_blks_per_seq]
    seq_lens_ptr,       # [num_seqs]
    scale,
    k_scale,
    v_scale,
    alibi_slopes,
    stride_max_logits_s,
    stride_max_logits_nh,
    stride_max_logits_p,
    stride_logits_s,
    stride_logits_nh,
    stride_logits_p,
    stride_logits_g,
    stride_q_s,
    stride_q_nh,
    stride_k_b,
    stride_k_nh,
    stride_k_hz,
    stride_k_bz,
    stride_v_b,
    stride_v_nh,
    stride_v_hz,
    stride_bt_s,
    compute_type: gl.constexpr,
    HEAD_SZ: gl.constexpr,
    HEAD_SZ_POW2: gl.constexpr,
    QUERY_GRP_SZ: gl.constexpr,
    QUERY_GRP_SZ_POW2: gl.constexpr,
    KV_BLK_SZ: gl.constexpr,
    KV_BLK_SZ_POW2: gl.constexpr,
    SEQ_PARTITION_SZ: gl.constexpr,
    USE_SINKS: gl.constexpr,
    SLIDING_WINDOW: gl.constexpr,
):

    seq_idx = gl.program_id(0)
    kv_head_idx = gl.program_id(1)
    num_query_heads = gl.num_programs(1) * QUERY_GRP_SZ

    log2e: gl.constexpr = 1.4426950408889634
    CONTIGUOUS_KV_ELEMS_16B_LOAD: gl.constexpr = 8
    K_HEAD_SZ_POW2_SPLIT: gl.constexpr = HEAD_SZ_POW2 // CONTIGUOUS_KV_ELEMS_16B_LOAD

    seq_len = gl.load(seq_lens_ptr + seq_idx)
    num_blocks = gl.cdiv(seq_len, SEQ_PARTITION_SZ)
    # seq_start_idx = seq_part_idx * SEQ_PARTITION_SZ
    # if seq_start_idx >= seq_len:
    #     return

    # seq_end_idx = gl.minimum(seq_start_idx + SEQ_PARTITION_SZ, seq_len)
    MAX_NUM_KV_BLKS: gl.constexpr = (SEQ_PARTITION_SZ + KV_BLK_SZ - 1) // KV_BLK_SZ

    # # 1 x QUERY_GRP_SZ_POW2 x HEAD_SZ_POW2
    # # 1 x 8(mdim) x 128(kdim)
    # blocked_q: gl.constexpr = gl.BlockedLayout(
    #     size_per_thread =[1, 1, 4],
    #     threads_per_warp=[1, 8, 8],
    #     warps_per_cta   =[1, 1, 4],
    #     order           =[2, 1, 0],
    # )
    # QUERY_GRP_SZ_POW2 x HEAD_SZ_POW2
    # 16(mdim) x 128(kdim)
    blocked_q0: gl.constexpr = gl.BlockedLayout(
        size_per_thread =[2, 4],
        threads_per_warp=[8, 8],
        warps_per_cta   =[1, 4],
        order           =[1, 0],
    )
    blocked_q: gl.constexpr = gl.BlockedLayout(
        size_per_thread =[1, 8],
        threads_per_warp=[4, 16],
        warps_per_cta   =[4, 1],
        order           =[1, 0],
    )
    shared_a_layout: gl.constexpr = gl.SwizzledSharedLayout(8, 1, 16, order=[1, 0])
    # MAX_NUM_KV_BLKS x K_HEAD_SZ_POW2_SPLIT x KV_BLK_SZ_POW2 x CONTIGUOUS_KV_ELEMS_16B_LOAD
    # 16 x 16 x 16 x 8
    # blocked_k: gl.constexpr = gl.BlockedLayout(
    #     size_per_thread =[4, 2, 2, 8],
    #     threads_per_warp=[1, 8, 8, 1],
    #     warps_per_cta   =[4, 1, 1, 1],
    #     order           =[3, 2, 1, 0],
    # )
    blocked_k: gl.constexpr = gl.BlockedLayout(
        size_per_thread =[1, 1, 1, 8],
        threads_per_warp=[1, 4, 16, 1],
        warps_per_cta   =[4, 1, 1, 1],
        order           =[3, 2, 1, 0],
    )
    # blocked_k: gl.constexpr = gl.DistributedLinearLayout( # 128x256
    #     reg_bases=((0,0,0,1), (0,0,0,2), (0,0,0,4), (0,1,0,0), (0,8,0,0), (4,0,0,0), (8,0,0,0)), # 16 x 8
    #     lane_bases=((0,0,1,0), (0,0,2,0), (0,0,4,0), (0,0,8,0), (0,2,0,0), (0,4,0,0)), # 64
    #     warp_bases=((1,0,0,0), (2,0,0,0)), # 4
    #     block_bases=[], # 8
    #     shape=[16, 16, 16, 8],
    # )

    # transposed: indicates the result tensor is transposed so that each thread holds consecutive elements
    # in the same row instead of column, which is good for chained dot and global write.
    qk_mfma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
        # version=3, instr_shape=[16, 16], transposed=False, warps_per_cta=[4, 1]
        version=3, instr_shape=[16, 16, 16], transposed=True, warps_per_cta=[1, 4]
    )
    qk_lhs_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=qk_mfma_layout, k_width=8
    )
    qk_rhs_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=qk_mfma_layout, k_width=8
    )

    pv_mfma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=3, instr_shape=[16, 16, 16], transposed=False, warps_per_cta=[1, 4]
    )
    pv_lhs_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=pv_mfma_layout, k_width=16
    )
    pv_rhs_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=pv_mfma_layout, k_width=16
    )

    # # blocked_q dim1
    # query_grp_sz_layout: gl.constexpr = gl.SliceLayout(0, gl.SliceLayout(2, blocked_q))
    # # blocked_q dim2
    # head_sz_layout: gl.constexpr = gl.SliceLayout(0, gl.SliceLayout(1, blocked_q))
    # blocked_q dim0
    query_grp_sz_layout: gl.constexpr = gl.SliceLayout(1, blocked_q)
    # blocked_q dim1
    head_sz_layout: gl.constexpr = gl.SliceLayout(0, blocked_q)

    # blocked_k dim0
    blk_id_layout: gl.constexpr = gl.SliceLayout(1, gl.SliceLayout(2, gl.SliceLayout(3, blocked_k)))
    # blocked_k dim1
    head_sz_div_layout: gl.constexpr = gl.SliceLayout(0, gl.SliceLayout(2, gl.SliceLayout(3, blocked_k)))
    # blocked_k dim2
    blk_layout: gl.constexpr = gl.SliceLayout(0, gl.SliceLayout(1, gl.SliceLayout(3, blocked_k)))
    # blocked_k dim3
    contiguous_kv_elems_layout: gl.constexpr = gl.SliceLayout(0, gl.SliceLayout(1, gl.SliceLayout(2, blocked_k)))

    q_grp_offs = gl.arange(0, QUERY_GRP_SZ_POW2, layout=query_grp_sz_layout)
    head_sz_offs = gl.arange(0, HEAD_SZ_POW2, layout=head_sz_layout)
    head_sz_div_offs = gl.arange(0, K_HEAD_SZ_POW2_SPLIT, layout=head_sz_div_layout)
    blk_offs = gl.arange(0, KV_BLK_SZ_POW2, layout=blk_layout)
    contiguous_kv_elems_offs = gl.arange(0, CONTIGUOUS_KV_ELEMS_16B_LOAD, layout=contiguous_kv_elems_layout)

    
    qk_row_offs = gl.arange(0, QUERY_GRP_SZ_POW2, layout=gl.SliceLayout(1, qk_mfma_layout))
    

    # load alibi slopes[QUERY_GRP_SZ_POW2]
    if alibi_slopes is None:
        alibi_slope = gl.zeros([QUERY_GRP_SZ_POW2], dtype=gl.float32)
    else:
        # alibi_slope = gl.load(
        #     alibi_slopes + kv_head_idx * QUERY_GRP_SZ + q_grp_offs,
        #     mask=q_grp_offs < QUERY_GRP_SZ,
        #     other=0.0,
        # )
        alibi_slope = gl.amd.cdna3.buffer_load(ptr=alibi_slopes + kv_head_idx * QUERY_GRP_SZ, offsets=qk_row_offs, mask=qk_row_offs < QUERY_GRP_SZ)

    # load all kv blocks in one time
    blk_ids = gl.arange(0, MAX_NUM_KV_BLKS, layout=blk_id_layout)
    
    blk_tables_start_ptr = blk_tables_ptrs + seq_idx * stride_bt_s
    # kv_blk_nums = gl.load(blk_tables_start_ptr + kv_blk_start + masked_blk_ids)
    

    # # load q[1, QUERY_GRP_SZ_POW2, HEAD_SZ_POW2]
    # q_offs = (
    #     seq_idx * stride_q_s
    #     + (kv_head_idx * QUERY_GRP_SZ + q_grp_offs[None, :, None]) * stride_q_nh
    #     + head_sz_offs[None, None, :]
    # )
    # q_mask = (q_grp_offs[None, :, None] < QUERY_GRP_SZ) & (head_sz_offs[None, None, :] < HEAD_SZ)
    # # q = gl.load(q_ptr + q_offs, mask=q_mask, other=0.0)
    # q = gl.amd.cdna3.buffer_load(ptr=q_ptr, offsets=q_offs, mask=q_mask)
    # q = (q * scale).to(compute_type)

    # load q[QUERY_GRP_SZ_POW2, HEAD_SZ_POW2]
    q_offs = (
        seq_idx * stride_q_s
        + (kv_head_idx * QUERY_GRP_SZ + q_grp_offs[:, None]) * stride_q_nh
        + head_sz_offs[None, :]
    )
    q_mask = (q_grp_offs[:, None] < QUERY_GRP_SZ) & (head_sz_offs[None, :] < HEAD_SZ)
    # q = gl.load(q_ptr + q_offs, mask=q_mask, other=0.0)
    q = gl.amd.cdna3.buffer_load(ptr=q_ptr, offsets=q_offs, mask=q_mask)
    # q = gl.amd.cdna3.buffer_load(ptr=q_ptr, offsets=q_offs)
    # q = (q * scale).to(compute_type)
    q_shared = gl.allocate_shared_memory(q.dtype, q.shape, shared_a_layout, q)
    # # (MAX_NUM_KV_BLKS, K_HEAD_SZ_POW2_SPLIT, KV_BLK_SZ_POW2, CONTIGUOUS_KV_ELEMS_16B_LOAD) --> (MAX_NUM_KV_BLKS, K_HEAD_SZ_POW2_SPLIT, CONTIGUOUS_KV_ELEMS_16B_LOAD, KV_BLK_SZ_POW2)
    # kt_temp = tl.permute(k, [0, 1, 3, 2])
    # # k[MAX_NUM_KV_BLKS, MAX_NUM_KV_BLKS * KV_BLK_SZ_POW2]
    # kt = tl.reshape(kt_temp, [MAX_NUM_KV_BLKS, HEAD_SZ_POW2, KV_BLK_SZ_POW2])

    # # qk[MAX_NUM_KV_BLKS, QUERY_GRP_SZ_POW2, KV_BLK_SZ_POW2]
    # qk = gl.dot(q, kt, out_dtype=gl.float32)
    # qk[QUERY_GRP_SZ_POW2, MAX_NUM_KV_BLKS * KV_BLK_SZ_POW2]
    # qk = gl.dot(q, k, out_dtype=gl.float32)
    accumulator = gl.zeros((QUERY_GRP_SZ_POW2, MAX_NUM_KV_BLKS * KV_BLK_SZ_POW2), dtype=gl.float32, layout=qk_mfma_layout)
    if USE_SINKS:
    #     M = gl.full([QUERY_GRP_SZ_POW2], float("-inf"), dtype=tl.float32)
    # else:
        M = gl.load(
            sink_ptr + (kv_head_idx * QUERY_GRP_SZ + q_grp_offs),
            mask=(kv_head_idx * QUERY_GRP_SZ + q_grp_offs) < num_query_heads,
            other=0,
        )
        

    # qc = gl.convert_layout(q, layout=qk_lhs_layout)
    qc = q_shared.load(qk_lhs_layout)
    for i in range(num_blocks):
        seq_start_idx = i * SEQ_PARTITION_SZ
        seq_end_idx = seq_start_idx + SEQ_PARTITION_SZ
        kv_blk_start = seq_part_idx * MAX_NUM_KV_BLKS
        qk_col_offs = seq_start_idx + gl.arange(0, SEQ_PARTITION_SZ, layout=gl.SliceLayout(0, qk_mfma_layout))
        kv_blk_nums = gl.amd.cdna3.buffer_load(ptr=blk_tables_start_ptr + kv_blk_start, offsets=masked_blk_ids)
        # k_blk_offs[MAX_NUM_KV_BLKS, K_HEAD_SZ_POW2_SPLIT, KV_BLK_SZ_POW2, CONTIGUOUS_KV_ELEMS_16B_LOAD]
        k_blk_offs = (
            kv_blk_nums[:, None, None, None] * stride_k_b
            + kv_head_idx * stride_k_nh
            + head_sz_div_offs[None, :, None, None] * stride_k_hz
            + blk_offs[None, None, :, None] * CONTIGUOUS_KV_ELEMS_16B_LOAD
            + contiguous_kv_elems_offs[None, None, None, :]
        )
        # blk_seq_offs = ((kv_blk_start + blk_ids[:, None]) * KV_BLK_SZ + blk_offs[None, :])
        # blk_seq_offs[MAX_NUM_KV_BLKS, 1, KV_BLK_SZ_POW2, 1]
        blk_seq_offs = ((kv_blk_start + blk_ids[:, None, None, None]) * KV_BLK_SZ + blk_offs[None, None, :, None])

        k_mask = (
            # (blk_seq_offs[:, None, :, None] < seq_len) &
            (blk_seq_offs < seq_len) &
            (blk_offs[None, None, :, None] < KV_BLK_SZ) &
            (head_sz_div_offs[None, :, None, None] < (HEAD_SZ // CONTIGUOUS_KV_ELEMS_16B_LOAD))
        )

        # k[MAX_NUM_KV_BLKS, K_HEAD_SZ_POW2_SPLIT, KV_BLK_SZ_POW2, CONTIGUOUS_KV_ELEMS_16B_LOAD]
        # k_0 = gl.load(k_cache_ptr + k_blk_offs)
        k_0 = gl.amd.cdna3.buffer_load(ptr=k_cache_ptr, offsets=k_blk_offs)
        k_scale_val = gl.load(k_scale) if k_0.dtype.is_fp8() else 1.0
        # k = k_0.to(gl.float32) * 
        k = k_0.to(compute_type)
        # (MAX_NUM_KV_BLKS, K_HEAD_SZ_POW2_SPLIT, KV_BLK_SZ_POW2, CONTIGUOUS_KV_ELEMS_16B_LOAD) --> (K_HEAD_SZ_POW2_SPLIT, CONTIGUOUS_KV_ELEMS_16B_LOAD, MAX_NUM_KV_BLKS, KV_BLK_SZ_POW2)
        kt_temp = tl.permute(k, [1, 3, 0, 2])
        # k[HEAD_SZ_POW2, MAX_NUM_KV_BLKS * KV_BLK_SZ_POW2]
        kt = tl.reshape(kt_temp, [HEAD_SZ_POW2, MAX_NUM_KV_BLKS * KV_BLK_SZ_POW2])

        num_kv_blks = gl.cdiv(seq_end_idx - seq_start_idx, KV_BLK_SZ)
        masked_blk_ids = gl.where(blk_ids < num_kv_blks, blk_ids, 0)
        

        kc = gl.convert_layout(kt, layout=qk_rhs_layout)
        qk = gl.amd.cdna3.mfma(qc, kc, accumulator) * scale * k_scale_val
    # qk = qk.to(compute_type)

    # blk_seq_flatten_offs = gl.reshape(blk_seq_offs, [MAX_NUM_KV_BLKS * KV_BLK_SZ_POW2])
    # if alibi_slopes is not None:
    #     qk += (alibi_slope[:, None] * (blk_seq_flatten_offs - seq_len + 1)[None, :]).to(gl.float32)
    # qk = gl.where(
    #     (q_grp_offs[:, None] < QUERY_GRP_SZ) & (blk_seq_flatten_offs[None, :] < seq_len),
    #     qk,
    #     float("-inf"),
    # )
        if SLIDING_WINDOW > 0:
            qk = gl.where(qk_col_offs[None, :] > seq_len-SLIDING_WINDOW, qk, -100000.0)

        if alibi_slopes is not None:
            qk += (alibi_slope[:, None] * (qk_col_offs - seq_len + 1)[None, :]).to(gl.float32)

        qk = gl.where(
            (qk_row_offs[:, None] < QUERY_GRP_SZ) & (qk_col_offs[None, :] < seq_len),
            qk,
            float("-inf"),
        )

        # gl.static_print(qk.shape, qk_row_offs.shape, qk_col_offs.shape)
        max_logit_new = gl.maximum(M, gl.max(qk, axis=1))
        max_logit_new = gl.where(max_logit_new > float("-inf"), max_logit_new, 0.0)
    
        p = tl.math.exp2((qk - max_logit_new[:, None]) * log2e)
        exp_sum = gl.sum(p, axis=1)
        p = p.to(compute_type)

        m_l_base_offs = gl.arange(0, QUERY_GRP_SZ_POW2, layout=gl.SliceLayout(1, qk_mfma_layout))
        m_l_offs = (
            seq_idx * stride_max_logits_s
            + kv_head_idx * stride_max_logits_nh
            + seq_part_idx * stride_max_logits_p
            + m_l_base_offs
        )
        m_l_grp_mask = m_l_base_offs < QUERY_GRP_SZ
    # gl.amd.cdna3.buffer_store(stored_value=max_logit_new, ptr=max_logits_ptr, offsets=m_l_offs, mask=m_l_grp_mask)
    # gl.amd.cdna3.buffer_store(stored_value=exp_sum, ptr=exp_sums_ptr, offsets=m_l_offs, mask=m_l_grp_mask)


    # # shape_info = (num_seqs, num_kv_heads, max_num_partitions, query_grp_sz)
    # # stride_logits_s,
    # # stride_logits_nh,
    # # stride_logits_p,
    # # stride_logits_g,
    # o_grp_offs = gl.arange(0, QUERY_GRP_SZ_POW2, layout=gl.SliceLayout(1, qk_mfma_layout))
    # o_head_sz_offs = gl.arange(0, 256, layout=gl.SliceLayout(0, qk_mfma_layout))
    # o_mask = (o_grp_offs[:, None] < QUERY_GRP_SZ) & (kv_blk_start + o_head_sz_offs[None, :] < seq_len)
    # logits_offs = seq_idx * stride_logits_s
    # logits_offs += kv_head_idx * stride_logits_nh
    # logits_offs += (
    #     seq_part_idx * stride_logits_p
    #     + o_grp_offs[:, None] * stride_logits_g
    #     + o_head_sz_offs[None, :]
    # )
    # gl.amd.cdna3.buffer_store(stored_value=p, ptr=logits_ptr, offsets=logits_offs, mask=o_mask)
    # # gl.amd.cdna3.buffer_store(stored_value=qk.to(compute_type), ptr=logits_ptr, offsets=logits_offs, mask=o_mask)
    # # gl.amd.cdna3.buffer_store(stored_value=qk.to(compute_type), ptr=logits_ptr, offsets=logits_offs)




    # # v_blk_offs[MAX_NUM_KV_BLKS, HEAD_SZ_POW2, KV_BLK_SZ_POW2]
    # v_blk_offs = (
    #     kv_blk_nums[:, None, None] * stride_v_b
    #     + kv_head_idx * stride_v_nh
    #     + head_sz_offs[None, :, None] * stride_v_hz
    #     + blk_offs[None, None, :]
    # )
    # v_mask = (
    #     (blk_seq_offs[:, None, :] < seq_len) &
    #     (blk_offs[None, None, :] < KV_BLK_SZ) &
    #     (head_sz_offs[None, :, None] < HEAD_SZ)
    # )

    # MAX_NUM_KV_BLKS x HEAD_SZ_POW2 x KV_BLK_SZ_POW2
    # 16(kdim0) x 128(ndim) x 16(kdim1)
    blocked_v_layout0: gl.constexpr = gl.BlockedLayout(
        size_per_thread =[4, 4,  8],
        threads_per_warp=[1, 32, 2],
        warps_per_cta   =[4, 1,  1],
        order           =[2, 1,  0],
    )
    blocked_v_layout: gl.constexpr = gl.DistributedLinearLayout( # 256x128
        reg_bases=((0,0,1), (0,0,2), (0,0,4), (0,0,8), (4,0,0), (8,0,0), (0,64,0)), # 16 x 8
        lane_bases=((0,1,0), (0,2,0), (0,4,0), (0,8,0), (1,0,0), (2,0,0)), # 64
        warp_bases=((0,16,0), (0,32,0)), # 4
        block_bases=[], # 8
        shape=[16, 128, 16],
    )
    v_dim0_offs = gl.arange(0, MAX_NUM_KV_BLKS, layout=gl.SliceLayout(1, gl.SliceLayout(2, blocked_v_layout)))
    v_dim1_offs = gl.arange(0, HEAD_SZ_POW2, layout=gl.SliceLayout(0, gl.SliceLayout(2, blocked_v_layout)))
    v_dim2_offs = gl.arange(0, KV_BLK_SZ_POW2, layout=gl.SliceLayout(0, gl.SliceLayout(1, blocked_v_layout)))

    # # load all kv blocks in one time
    # blk_ids = gl.arange(0, MAX_NUM_KV_BLKS, layout=blk_id_layout)
    # masked_blk_ids = gl.where(blk_ids < num_kv_blks, blk_ids, 0)
    # blk_tables_start_ptr = blk_tables_ptrs + seq_idx * stride_bt_s
    # # kv_blk_nums = gl.load(blk_tables_start_ptr + kv_blk_start + masked_blk_ids)
    # kv_blk_nums = gl.amd.cdna3.buffer_load(ptr=blk_tables_start_ptr + kv_blk_start, offsets=masked_blk_ids)

    kv_blk_nums2 = gl.convert_layout(kv_blk_nums, layout=gl.SliceLayout(1, gl.SliceLayout(2, blocked_v_layout)))
    # v_blk_offs[MAX_NUM_KV_BLKS, HEAD_SZ_POW2, KV_BLK_SZ_POW2]
    v_blk_offs = (
        kv_blk_nums2[:, None, None] * stride_v_b
        + kv_head_idx * stride_v_nh
        + v_dim1_offs[None, :, None] * stride_v_hz
        + v_dim2_offs[None, None, :]
    )
    v_len_offs = kv_blk_start + v_dim0_offs[:, None, None] * KV_BLK_SZ + v_dim2_offs[None, None, :]
    v_mask = (
        (v_len_offs < seq_len) &
        (v_dim2_offs[None, None, :] < KV_BLK_SZ) &
        (v_dim1_offs[None, :, None] < HEAD_SZ)
    )

    # v[MAX_NUM_KV_BLKS, HEAD_SZ_POW2, KV_BLK_SZ_POW2]
    # v_0 = gl.load(v_cache_ptr + v_blk_offs)
    
    v_0 = gl.amd.cdna3.buffer_load(ptr=v_cache_ptr, offsets=v_blk_offs)
    v_scale_val = gl.load(v_scale) if v_0.dtype.is_fp8() else 1.0
    # v = v_0.to(gl.float32) * v_scale_val
    v = v_0.to(compute_type)
    # [MAX_NUM_KV_BLKS, HEAD_SZ_POW2, KV_BLK_SZ_POW2] --> [MAX_NUM_KV_BLKS, KV_BLK_SZ_POW2, HEAD_SZ_POW2]
    v = gl.permute(v, [0, 2, 1])
    # v[MAX_NUM_KV_BLKS * KV_BLK_SZ_POW2, HEAD_SZ_POW2]
    v = gl.reshape(v, [MAX_NUM_KV_BLKS * KV_BLK_SZ_POW2, HEAD_SZ_POW2])

    m_l_base_offs = gl.arange(0, QUERY_GRP_SZ_POW2, layout=gl.SliceLayout(1, qk_mfma_layout))
    m_l_offs = (
        seq_idx * stride_max_logits_s
        + kv_head_idx * stride_max_logits_nh
        + seq_part_idx * stride_max_logits_p
        + m_l_base_offs
    )
    m_l_grp_mask = m_l_base_offs < QUERY_GRP_SZ

    # q_grp_offs2 = tl.arange(0, QUERY_GRP_SZ_POW2)
    # max_logits_offs = (
    #     seq_idx * stride_max_logits_s
    #     + kv_head_idx * stride_max_logits_nh
    #     + seq_part_idx * stride_max_logits_p
    #     + q_grp_offs2
    # )
    # m_grp_mask = q_grp_offs2 < QUERY_GRP_SZ
    # tl.store(max_logits_ptr + max_logits_offs, max_logit_new, mask=m_grp_mask)
    # tl.store(exp_sums_ptr + max_logits_offs, exp_sum, mask=m_grp_mask)
    gl.amd.cdna3.buffer_store(stored_value=max_logit_new, ptr=max_logits_ptr, offsets=m_l_offs, mask=m_l_grp_mask)
    gl.amd.cdna3.buffer_store(stored_value=exp_sum, ptr=exp_sums_ptr, offsets=m_l_offs, mask=m_l_grp_mask)

    # acc[QUERY_GRP_SZ_POW2, HEAD_SZ_POW2]
    # acc = gl.dot(p, v, out_dtype=gl.float32)
    # acc = acc / exp_sum[:, None]
    accumulator2 = gl.zeros((QUERY_GRP_SZ_POW2, HEAD_SZ_POW2), dtype=gl.float32, layout=pv_mfma_layout)

    pc = gl.convert_layout(p, layout=pv_lhs_layout)
    vc = gl.convert_layout(v, layout=pv_rhs_layout)

    acc = gl.amd.cdna3.mfma(pc, vc, accumulator2) * v_scale_val
    exp_sum = gl.convert_layout(exp_sum[:, None], layout=pv_mfma_layout)
    if USE_SINKS:
        M = tl.math.exp2((gl.convert_layout(M[:, None], layout=qk_mfma_layout) - max_logit_new[:, None]) * log2e)
        exp_sum += gl.convert_layout(M, layout=pv_mfma_layout)
    exp_sum = tl.broadcast_to(exp_sum, QUERY_GRP_SZ_POW2, HEAD_SZ_POW2)
    # exp_sum = tl.broadcast_to(exp_sum[:, None], QUERY_GRP_SZ_POW2, HEAD_SZ_POW2)
    # exp_sum = gl.convert_layout(exp_sum, layout=pv_mfma_layout)
    acc = acc / exp_sum
    acc = acc.to(compute_type)

    # end up computation
    o_grp_offs = gl.arange(0, QUERY_GRP_SZ_POW2, layout=gl.SliceLayout(1, pv_mfma_layout))
    o_head_sz_offs = gl.arange(0, HEAD_SZ_POW2, layout=gl.SliceLayout(0, pv_mfma_layout))
    o_mask = (o_grp_offs[:, None] < QUERY_GRP_SZ) & (o_head_sz_offs[None, :] < HEAD_SZ)
    logits_offs = seq_idx * stride_logits_s
    logits_offs += kv_head_idx * stride_logits_nh
    logits_offs += (
        seq_part_idx * stride_logits_p
        + o_grp_offs[:, None] * stride_logits_g
        + o_head_sz_offs[None, :]
    )
    # gl.store(logits_ptr + logits_offs, acc, mask=q_mask)
    gl.amd.cdna3.buffer_store(stored_value=acc, ptr=logits_ptr, offsets=logits_offs, mask=o_mask)