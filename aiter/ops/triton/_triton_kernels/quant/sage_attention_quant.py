import triton
import triton.language as tl


from aiter.ops.triton.utils._triton.pid_preprocessing import pid_grid_3d




###################### Unified quantization kernels ###############

from aiter.ops.triton._triton_kernels.attention.unified_attention import find_seq_idx

@triton.jit
def perblock_quantize_q_kernel(
    query_ptr,
    Q_q,
    Q_descale,
    query_start_len_ptr,
    num_seqs,
    num_query_heads,
    num_queries_per_kv,
    query_stride_0,
    query_stride_1,
    scale_stride_0,
    scale_stride_1,
    BLOCK_Q: tl.constexpr,
    BLOCK_M: tl.constexpr,
    HEAD_SIZE_PADDED: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    sm_scale: tl.constexpr,
    DTYPE_MAX: tl.constexpr,
):

    kv_head_idx = tl.program_id(0)
    q_block_global_idx = tl.program_id(1)

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

    Q_cache_modifier: tl.constexpr = ""
    # Q : (BLOCK_M, HEAD_SIZE_PADDED)
    Q = tl.load(
        query_ptr + query_offset,
        mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
        other=0.0,
        cache_modifier=Q_cache_modifier,
    )

    Q = Q.to(tl.float32)
    if sm_scale is not None:
        Q *= sm_scale
    scale = tl.max(tl.abs(Q)) / DTYPE_MAX
    Q_quant = Q / scale
    Q_quant = Q_quant.to(Q_q.dtype.element_ty)

    # scale (total num blocks, hk)
    scale_ptrs = Q_descale + q_block_global_idx * scale_stride_0 + kv_head_idx * scale_stride_1
    tl.store(scale_ptrs, scale)
    tl.store(
        Q_q + query_offset, Q_quant,
        mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None])


@triton.jit
def perblock_quantize_kernel(
    query_ptr,
    Q_q,
    Q_descale,
    max_seqlen,
    query_start_len_ptr,
    stride_m,
    scale_stride_0,
    scale_stride_1,
    BLOCK_M: tl.constexpr,
    D: tl.constexpr,
    sm_scale: tl.constexpr,
    DTYPE_MAX: tl.constexpr,
):
    
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_h = tl.program_id(2)
    
    if query_start_len_ptr is not None:
        cur_batch_in_all_start_index = tl.load(query_start_len_ptr + pid_b).to(tl.int64)
        cur_batch_in_all_stop_index = tl.load(query_start_len_ptr + pid_b + 1).to(tl.int64)
        seqlen = cur_batch_in_all_stop_index - cur_batch_in_all_start_index
    else:
        seqlen = max_seqlen

    if pid_m * BLOCK_M >= seqlen:
        return

    num_pid_m = tl.cdiv(max_seqlen, BLOCK_M)
    global_blk_idx = pid_b * num_pid_m + pid_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    offs_d = tl.arange(0, D)[None,:]
    mask = offs_m < seqlen
    tile_offsets = offs_m * stride_m + offs_d
    tile_ptrs = query_ptr + tile_offsets
    Q = tl.load(
        tile_ptrs,
        mask=mask,
        other=0.0,
    )

    Q = Q.to(tl.float32)
    if sm_scale is not None:
        Q *= sm_scale
    scale = tl.max(tl.abs(Q)) / DTYPE_MAX
    Q_quant = Q / scale
    Q_quant = Q_quant.to(Q_q.dtype.element_ty)

    # scale (total num blocks)
    scale_ptrs = Q_descale + global_blk_idx * scale_stride_0 + pid_h * scale_stride_1
    tl.store(scale_ptrs, scale)
    
    qtile_ptrs = Q_q + tile_offsets
    tl.store(
        qtile_ptrs, Q_quant,
        mask=mask)


@triton.jit
def _rotate_mxfp_quantize_k_kernel(
    K,
    K_q,
    K_descale,
    R,  # Hadamard matrix
    stride_t,
    stride_tq,
    stride_ts,
    num_tokens,
    hadamard_rotation: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_R: tl.constexpr,  # rotation block size
    D: tl.constexpr,  # D is 128
    sm_scale: tl.constexpr,
    SCALE_GROUP_SIZE: tl.constexpr = 32,
):

    pid = tl.program_id(0).to(tl.int64)

    # Offsets
    offs_t = pid * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    mask = (offs_t < num_tokens)

    offs_d = tl.arange(0, D)
    offs_dq = tl.arange(0, D // 2)
    offs_ds = tl.arange(0, D // SCALE_GROUP_SIZE)

    # set pointers to either Q or K tensor, descale, quantized output
    # Q block shape: [BLOCK_M, D]
    tensor_offset = K + (
        offs_t * stride_t
        + offs_d[None, :]
    )
    descale_offset = K_descale + (
        offs_t * stride_ts
        + offs_ds[None, :]
    )  # we group 32 values together for quantization

    quant_tensor_offset = K_q + (
        offs_t * stride_tq
        + offs_dq[None, :]
    )

    qk_ptr = tensor_offset
    qk_descale_ptr = descale_offset
    qk_quant_ptr = quant_tensor_offset

    qk_tile = tl.load(
        qk_ptr, mask=mask & (offs_d[None, :] < D), other=0.0
    )  # (BLOCK_M, D)

    if hadamard_rotation:
        r_ptr = (
            R
            + tl.arange(0, BLOCK_R)[:, None] * BLOCK_R
            + tl.arange(0, BLOCK_R)[None, :]
        )
        r_mat = tl.load(r_ptr)  # BLOCK_R x BLOCK_R

        shape0: tl.constexpr = BLOCK_M * D // BLOCK_R

        # Rotate: Q_rot = Q @ R
        qk_rot_tile = tl.dot(qk_tile.reshape((shape0, BLOCK_R)).to(r_mat.dtype), r_mat)
        qk_rot_tile = qk_rot_tile.reshape((BLOCK_M, D))
    else:
        qk_rot_tile = qk_tile.to(tl.float32)

    if sm_scale is not None:
        qk_rot_tile *= sm_scale
    
    qk_quant_tile, qk_descale = _compute_mx_quant_and_scale_rne(
        qk_rot_tile, mask, tl.uint8
    )

    tl.store(qk_descale_ptr, qk_descale, mask=mask)

    tl.store(
        qk_quant_ptr,
        qk_quant_tile,
        mask=mask,
    )



# @triton.jit
# def perchannel_quantize_v_kernel(
#     V_Input,
#     V_Output,
#     V_Descale,
#     stride_t,
#     stride_h,
#     stride_sh,
#     num_tokens,
#     D: tl.constexpr,
#     BLOCK_M: tl.constexpr,
# ):
#     pid = tl.program_id(0).to(tl.int64)
#     pid_h = tl.program_id(1).to(tl.int64)

#     offs_token = pid * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
#     mask = offs_token < num_tokens
#     offs_d = tl.arange(0, D)

#     v_offs = (
#         offs_token * stride_t
#         + pid_h * stride_h
#         + offs_d[None, :]
#     )

#     v_input_ptrs = V_Input + v_offs
#     v_output_ptrs = V_Output + v_offs

#     # just apply the per channel v_scales that have been computed outside
#     v_descale_ptrs = V_Descale + pid_h * stride_sh + offs_d
#     v = tl.load(v_input_ptrs, mask=mask, other=0.0)
#     v_descales = tl.load(v_descale_ptrs)
#     v_quant = v / v_descales
#     v_quant = v_quant.to(v_output_ptrs.dtype.element_ty)
#     tl.store(v_output_ptrs, v_quant, mask=mask)



################# Sage V2 quantization kernels ####################


@triton.jit
def _compute_mx_quant_and_scale_rne(
    src_tensor,
    valid_src_mask,
    mx_tensor_dtype: tl.constexpr,
):
    """
    Compute MX quantization with RNE (Round to Nearest Even) rounding for the scale.

    RNE is applied when converting max_abs to E8M0 format (nearest power of 2).
    This is equivalent to computing: scale = 2^(clip(floor(log2(RNE(max_abs(x)))), -127, 127) - 2)
    where RNE rounds to the nearest power of 2, with ties going to even exponent.
    """
    is_fp8: tl.constexpr = (
        mx_tensor_dtype == tl.float8e4nv or mx_tensor_dtype == tl.float8e5
    )
    BLOCK_SIZE_OUT_DIM: tl.constexpr = src_tensor.shape[0]
    BLOCK_SIZE_QUANT_DIM: tl.constexpr = src_tensor.shape[1]
    BLOCK_SIZE_QUANT_MX_SCALE: tl.constexpr = src_tensor.shape[1] // 32

    # Explicit cast to fp32 since most ops are not supported on bfloat16
    f32_tensor = src_tensor.to(tl.float32)
    abs_tensor = tl.abs(f32_tensor)
    abs_tensor = tl.where(
        valid_src_mask, abs_tensor, -1.0
    )  # Don't consider padding tensors in scale computation
    abs_tensor = tl.reshape(
        abs_tensor, [BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_MX_SCALE, 32]
    )
    max_val = tl.max(abs_tensor, axis=2, keep_dims=True)

    # RNE (Round to Nearest Even) rounding when converting max_abs to E8M0 format
    # E8M0 stores only exponent (no mantissa), so we round to nearest power of 2
    # Extract exponent and mantissa from float32
    max_val_bits = max_val.to(tl.uint32, bitcast=True)
    exponent = (max_val_bits >> 23) & 0xFF
    mantissa = max_val_bits & 0x7FFFFF

    # RNE to nearest power of 2:
    # For value 2^n * (1 + m/2^23), the threshold is at m = 0.5 * 2^23 = 0x400000
    # - If mantissa < 0x400000: round to 2^n (keep exponent)
    # - If mantissa > 0x400000: round to 2^(n+1) (increment exponent)
    # - If mantissa == 0x400000: tie case, round to even exponent (RNE)

    # Determine if we should round up
    should_round_up = (mantissa > 0x400000) | (
        (mantissa == 0x400000) & ((exponent & 1) == 1)
    )

    rounded_exponent = tl.where(should_round_up, exponent + 1, exponent)

    # Subtract 2 from exponent (divide by 4) to get final scale exponent
    # Clamp to valid E8M0 range [-127, 127] (exponent 0-254 in biased representation)
    scale_exponent = rounded_exponent - 2
    scale_exponent = tl.maximum(scale_exponent, 0)
    scale_exponent = tl.minimum(scale_exponent, 254)

    # Construct the scale as a power of 2
    dequant_scale_exponent = (scale_exponent << 23) & 0x7F800000
    dequant_scale = dequant_scale_exponent.to(tl.float32, bitcast=True)
    quant_scale = tl.where(dequant_scale == 0, 0, 1.0 / dequant_scale)

    f32_tensor = tl.reshape(
        f32_tensor, [BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_MX_SCALE, 32]
    )
    quant_tensor = f32_tensor * quant_scale

    # Reshape the tensors after scaling
    quant_tensor = quant_tensor.reshape([BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_DIM])
    # Set the invalid portions of the tensor to 0
    quant_tensor = tl.where(valid_src_mask, quant_tensor, 0)
    dequant_scale_exponent = dequant_scale_exponent.reshape(
        [BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_MX_SCALE]
    )

    # Extract the exponent part of the scales and store the result
    dequant_scale_exponent = (dequant_scale_exponent >> 23).to(tl.uint8)

    # Convert the tensors to the mx format
    if is_fp8:
        out_tensor = quant_tensor.to(mx_tensor_dtype)
    else:
        quant_tensor = quant_tensor.to(tl.uint32, bitcast=True)
        signs = quant_tensor & 0x80000000
        exponents = (quant_tensor >> 23) & 0xFF
        mantissas = quant_tensor & 0x7FFFFF

        # 0.25 <= x < 0.75 maps to 0.5, a denormal number
        E8_BIAS = 127
        E2_BIAS = 1
        # Move implicit bit 1 at the beginning to mantissa for denormals
        adjusted_exponents = tl.core.sub(
            E8_BIAS, exponents + 1, sanitize_overflow=False
        )
        mantissas = tl.where(
            exponents < E8_BIAS,
            (0x400000 | (mantissas >> 1)) >> adjusted_exponents,
            mantissas,
        )

        # For normal numbers, we change the bias from 127 to 1, and for subnormals, we keep exponent as 0.
        exponents = tl.maximum(exponents, E8_BIAS - E2_BIAS) - (E8_BIAS - E2_BIAS)

        # Combine sign, exponent, and mantissa, while saturating
        # rounding nearest with tie breaking up by adding +1 to one bit right of the LSB, then shift right
        e2m1_tmp = tl.minimum((((exponents << 2) | (mantissas >> 21)) + 1) >> 1, 0x7)
        e2m1_value = ((signs >> 28) | e2m1_tmp).to(tl.uint8)

        e2m1_value = tl.reshape(
            e2m1_value, [BLOCK_SIZE_OUT_DIM, BLOCK_SIZE_QUANT_DIM // 2, 2]
        )
        evens, odds = tl.split(e2m1_value)
        out_tensor = evens | (odds << 4)

    return out_tensor, dequant_scale_exponent


@triton.jit
def sage_quant_v_kernel(
    V_Input,
    V_Output,
    V_Scale,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kd,
    stride_vsz,
    stride_vsh,
    BATCH,
    K_HEAD,
    K_NUM_BLKS,
    SEQLEN_K,
    D: tl.constexpr,
    BLK_K: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)

    offs_blk_k = tl.arange(0, BLK_K)
    offs_d = tl.arange(0, D)

    # V
    off_blk, off_h, off_b = pid_grid_3d(pid, K_NUM_BLKS, K_HEAD, BATCH)
    offs_kn = off_blk * BLK_K + offs_blk_k

    v_offs = (
        off_b * stride_kz
        + off_h * stride_kh
        + offs_kn[:, None] * stride_kn
        + offs_d[None, :] * stride_kd
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
def _rotate_quantize_q_kernel(
    Q,
    Q_q,
    Q_descale,
    Q_mean,
    R,  # Hadamard matrix
    sm_scale: tl.constexpr,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_qd,
    stride_qqb,
    stride_qqm,
    stride_qqh,
    stride_qqd,
    stride_qsb,
    stride_qsm,
    stride_qsh,
    stride_qsd,
    stride_mb,
    stride_mh,
    stride_mm,
    stride_md,
    batch,
    heads_q,
    seqlen_q,
    d_model,
    q_smoothing: tl.constexpr,
    hadamard_rotation: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_R: tl.constexpr,  # rotation block size
    D: tl.constexpr,  # D is 128
):
    SCALE_GROUP_SIZE: tl.constexpr = 32
    pid = tl.program_id(0).to(tl.int64)
    pid_b = pid % batch
    pid_h = pid // batch % heads_q
    pid_m = pid // (batch * heads_q)

    # Offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)

    offs_dq = tl.arange(0, D // 2)
    offs_ds = tl.arange(0, D // SCALE_GROUP_SIZE)

    # set pointers to either Q or K tensor, descale, quantized output
    # Q block shape: [BLOCK_M, D]
    tensor_offset = Q + (
        pid_b * stride_qb
        + pid_h * stride_qh
        + offs_m[:, None] * stride_qm
        + offs_d[None, :] * stride_qd
    )
    descale_offset = Q_descale + (
        pid_b * stride_qsb
        + pid_h * stride_qsh
        + offs_m[:, None] * stride_qsm
        + offs_ds[None, :] * stride_qsd
    )  # we group 32 values together for quantization
    # Store rotated and quantized Q
    quant_tensor_offset = Q_q + (
        pid_b * stride_qqb
        + pid_h * stride_qqh
        + offs_m[:, None] * stride_qqm
        + offs_dq[None, :] * stride_qqd
    )
    seqlen = seqlen_q
    qk_ptr = tensor_offset
    qk_descale_ptr = descale_offset
    qk_quant_ptr = quant_tensor_offset

    qk_tile = tl.load(
        qk_ptr, mask=(offs_m[:, None] < seqlen) & (offs_d[None, :] < d_model), other=0.0
    )  # (BLOCK_M, D)
    original_dtype = qk_tile.dtype
    if q_smoothing:
        ACTUAL_BLOCK_M = tl.minimum(BLOCK_M, seqlen - pid_m * BLOCK_M)
        m_row_mean = (
            tl.sum(qk_tile, axis=0) / ACTUAL_BLOCK_M
        )  # Sum over BLOCK_M -> shape [D]
        qk_tile -= m_row_mean[None, :]
        qk_tile = qk_tile.to(original_dtype)
        mean_ptr = (
            Q_mean
            + pid_b * stride_mb
            + pid_h * stride_mh
            + pid_m * stride_mm
            + offs_d * stride_md
        )
        tl.store(mean_ptr, m_row_mean * sm_scale)

    if hadamard_rotation:
        r_ptr = (
            R
            + tl.arange(0, BLOCK_R)[:, None] * BLOCK_R
            + tl.arange(0, BLOCK_R)[None, :]
        )
        r_mat = tl.load(r_ptr)  # BLOCK_R x BLOCK_R

        shape0: tl.constexpr = BLOCK_M * D // BLOCK_R

        # Rotate: Q_rot = Q @ R
        qk_rot_tile = tl.dot(qk_tile.reshape((shape0, BLOCK_R)).to(r_mat.dtype), r_mat)
        qk_rot_tile = qk_rot_tile.reshape((BLOCK_M, D))
    else:
        qk_rot_tile = qk_tile.to(tl.float32)

    qk_rot_tile *= sm_scale

    qk_quant_tile, qk_descale = _compute_mx_quant_and_scale_rne(
        qk_rot_tile, offs_m[:, None] < seqlen, tl.uint8
    )

    tl.store(qk_descale_ptr, qk_descale, mask=(offs_m[:, None] < seqlen))

    tl.store(
        qk_quant_ptr,
        qk_quant_tile,
        mask=(offs_m[:, None] < seqlen),
    )


@triton.jit
def _rotate_quantize_k_kernel(
    Q,
    Q_q,
    Q_descale,
    Q_mean,
    K,
    K_q,
    K_descale,
    R,  # Hadamard matrix
    sm_scale: tl.constexpr,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_qd,
    stride_qqb,
    stride_qqm,
    stride_qqh,
    stride_qqd,
    stride_qsb,
    stride_qsm,
    stride_qsh,
    stride_qsd,
    stride_mb,
    stride_mh,
    stride_mm,
    stride_md,
    stride_kb,
    stride_kh,
    stride_km,
    stride_kd,
    stride_kqb,
    stride_kqn,
    stride_kqh,
    stride_kqd,
    stride_ksb,
    stride_ksn,
    stride_ksh,
    stride_ksd,
    batch,
    heads_q,
    heads_k,
    seqlen_q,
    seqlen_k,
    d_model,
    q_smoothing: tl.constexpr,
    hadamard_rotation: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_R: tl.constexpr,  # rotation block size
    D: tl.constexpr,  # D is 128
):
    SCALE_GROUP_SIZE: tl.constexpr = 32

    pid = tl.program_id(0).to(tl.int64)

    pid_b = pid % batch
    pid_h = pid // batch % heads_k
    pid_m = pid // (batch * heads_k)

    # Offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)

    offs_dq = tl.arange(0, D // 2)
    offs_ds = tl.arange(0, D // SCALE_GROUP_SIZE)

    # set pointers to either Q or K tensor, descale, quantized output
    # Q block shape: [BLOCK_M, D]
    tensor_offset = K + (
        pid_b * stride_kb
        + pid_h * stride_kh
        + offs_m[:, None] * stride_km
        + offs_d[None, :] * stride_kd
    )
    descale_offset = K_descale + (
        pid_b * stride_ksb
        + pid_h * stride_ksh
        + offs_m[:, None] * stride_ksn
        + offs_ds[None, :] * stride_ksd
    )  # we group 32 values together for quantization

    quant_tensor_offset = K_q + (
        pid_b * stride_kqb
        + pid_h * stride_kqh
        + offs_m[:, None] * stride_kqn
        + offs_dq[None, :] * stride_kqd
    )
    seqlen = seqlen_k

    qk_ptr = tensor_offset
    qk_descale_ptr = descale_offset
    qk_quant_ptr = quant_tensor_offset

    qk_tile = tl.load(
        qk_ptr, mask=(offs_m[:, None] < seqlen) & (offs_d[None, :] < d_model), other=0.0
    )  # (BLOCK_M, D)

    if hadamard_rotation:
        r_ptr = (
            R
            + tl.arange(0, BLOCK_R)[:, None] * BLOCK_R
            + tl.arange(0, BLOCK_R)[None, :]
        )
        r_mat = tl.load(r_ptr)  # BLOCK_R x BLOCK_R

        shape0: tl.constexpr = BLOCK_M * D // BLOCK_R

        # Rotate: Q_rot = Q @ R
        qk_rot_tile = tl.dot(qk_tile.reshape((shape0, BLOCK_R)).to(r_mat.dtype), r_mat)
        qk_rot_tile = qk_rot_tile.reshape((BLOCK_M, D))
    else:
        qk_rot_tile = qk_tile.to(tl.float32)

    qk_quant_tile, qk_descale = _compute_mx_quant_and_scale_rne(
        qk_rot_tile, offs_m[:, None] < seqlen, tl.uint8
    )

    tl.store(qk_descale_ptr, qk_descale, mask=(offs_m[:, None] < seqlen))

    tl.store(
        qk_quant_ptr,
        qk_quant_tile,
        mask=(offs_m[:, None] < seqlen),
    )


@triton.jit
def _rotate_quantize_qk_kernel(
    Q,
    Q_q,
    Q_descale,
    Q_mean,
    K,
    K_q,
    K_descale,
    R,  # Hadamard matrix
    sm_scale: tl.constexpr,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_qd,
    stride_qqb,
    stride_qqm,
    stride_qqh,
    stride_qqd,
    stride_qsb,
    stride_qsm,
    stride_qsh,
    stride_qsd,
    stride_mb,
    stride_mh,
    stride_mm,
    stride_md,
    stride_kb,
    stride_kh,
    stride_km,
    stride_kd,
    stride_kqb,
    stride_kqn,
    stride_kqh,
    stride_kqd,
    stride_ksb,
    stride_ksn,
    stride_ksh,
    stride_ksd,
    batch,
    heads_q,
    heads_k,
    seqlen_q,
    seqlen_k,
    d_model,
    q_smoothing: tl.constexpr,
    hadamard_rotation: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_R: tl.constexpr,  # rotation block size
    D: tl.constexpr,  # D is 128
):
    SCALE_GROUP_SIZE: tl.constexpr = 32

    q_pids = batch * heads_q * tl.cdiv(seqlen_q, BLOCK_M)
    pid = tl.program_id(0).to(tl.int64)
    is_q_pid = pid < q_pids

    if is_q_pid:
        pid_b = pid % batch
        pid_h = pid // batch % heads_q
        pid_m = pid // (batch * heads_q)
    else:  # is k pid
        pid -= q_pids
        pid_b = pid % batch
        pid_h = pid // batch % heads_k
        pid_m = pid // (batch * heads_k)

    # Offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)

    offs_dq = tl.arange(0, D // 2)
    offs_ds = tl.arange(0, D // SCALE_GROUP_SIZE)

    # set pointers to either Q or K tensor, descale, quantized output
    # Q block shape: [BLOCK_M, D]
    if is_q_pid:
        tensor_offset = Q + (
            pid_b * stride_qb
            + pid_h * stride_qh
            + offs_m[:, None] * stride_qm
            + offs_d[None, :] * stride_qd
        )
        descale_offset = Q_descale + (
            pid_b * stride_qsb
            + pid_h * stride_qsh
            + offs_m[:, None] * stride_qsm
            + offs_ds[None, :] * stride_qsd
        )  # we group 32 values together for quantization
        # Store rotated and quantized Q
        quant_tensor_offset = Q_q + (
            pid_b * stride_qqb
            + pid_h * stride_qqh
            + offs_m[:, None] * stride_qqm
            + offs_dq[None, :] * stride_qqd
        )
        seqlen = seqlen_q
    else:
        tensor_offset = K + (
            pid_b * stride_kb
            + pid_h * stride_kh
            + offs_m[:, None] * stride_km
            + offs_d[None, :] * stride_kd
        )
        descale_offset = K_descale + (
            pid_b * stride_ksb
            + pid_h * stride_ksh
            + offs_m[:, None] * stride_ksn
            + offs_ds[None, :] * stride_ksd
        )  # we group 32 values together for quantization

        quant_tensor_offset = K_q + (
            pid_b * stride_kqb
            + pid_h * stride_kqh
            + offs_m[:, None] * stride_kqn
            + offs_dq[None, :] * stride_kqd
        )
        seqlen = seqlen_k

    qk_ptr = tensor_offset
    qk_descale_ptr = descale_offset
    qk_quant_ptr = quant_tensor_offset

    qk_tile = tl.load(
        qk_ptr, mask=(offs_m[:, None] < seqlen) & (offs_d[None, :] < d_model), other=0.0
    )  # (BLOCK_M, D)
    original_dtype = qk_tile.dtype

    if is_q_pid:
        if q_smoothing:
            ACTUAL_BLOCK_M = tl.minimum(BLOCK_M, seqlen - pid_m * BLOCK_M)
            m_row_mean = (
                tl.sum(qk_tile, axis=0) / ACTUAL_BLOCK_M
            )  # Sum over BLOCK_M -> shape [D]
            qk_tile -= m_row_mean[None, :]
            qk_tile = qk_tile.to(original_dtype)
            mean_ptr = (
                Q_mean
                + pid_b * stride_mb
                + pid_h * stride_mh
                + pid_m * stride_mm
                + offs_d * stride_md
            )
            tl.store(mean_ptr, m_row_mean * sm_scale)

    if hadamard_rotation:
        r_ptr = (
            R
            + tl.arange(0, BLOCK_R)[:, None] * BLOCK_R
            + tl.arange(0, BLOCK_R)[None, :]
        )
        r_mat = tl.load(r_ptr)  # BLOCK_R x BLOCK_R

        shape0: tl.constexpr = BLOCK_M * D // BLOCK_R

        # Rotate: Q_rot = Q @ R
        qk_rot_tile = tl.dot(qk_tile.reshape((shape0, BLOCK_R)).to(r_mat.dtype), r_mat)
        qk_rot_tile = qk_rot_tile.reshape((BLOCK_M, D))
    else:
        qk_rot_tile = qk_tile.to(tl.float32)

    if is_q_pid:
        qk_rot_tile *= sm_scale

    qk_quant_tile, qk_descale = _compute_mx_quant_and_scale_rne(
        qk_rot_tile, offs_m[:, None] < seqlen, tl.uint8
    )

    tl.store(qk_descale_ptr, qk_descale, mask=(offs_m[:, None] < seqlen))

    tl.store(
        qk_quant_ptr,
        qk_quant_tile,
        mask=(offs_m[:, None] < seqlen),
    )


@triton.jit
def _rot_q_kernel(
    Q,
    Q_rot,
    Q_mean,
    R,  # Hadamard matrix
    sm_scale: tl.constexpr,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_qd,
    stride_qob,
    stride_qoh,
    stride_qom,
    stride_qod,
    stride_mb,
    stride_mh,
    stride_mm,
    stride_md,
    stride_rm,
    stride_rd,
    n_heads,
    seq_len,
    d_model,
    q_smoothing: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,  # BLOCK_D is 32
):
    # Grid: (batch * n_heads, seq_len // BLOCK_M, d_model // BLOCK_D)
    pid_bh = tl.program_id(0).to(tl.int64)
    pid_m = tl.program_id(1).to(tl.int64)
    pid_d = tl.program_id(2).to(tl.int64)

    pid_h = pid_bh % n_heads
    pid_b = pid_bh // n_heads

    # Offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)

    # Load Q block and R (Hadamard)
    # Q block shape: [BLOCK_M, BLOCK_D]
    q_ptr = (
        Q
        + (pid_b * stride_qb)
        + pid_h * stride_qh
        + offs_m[:, None] * stride_qm
        + offs_d[None, :] * stride_qd
    )
    r_ptr = (
        R
        + tl.arange(0, BLOCK_D)[:, None] * stride_rm
        + tl.arange(0, BLOCK_D)[None, :] * stride_rd
    )
    q_tile = tl.load(
        q_ptr, mask=(offs_m[:, None] < seq_len) & (offs_d[None, :] < d_model), other=0.0
    )
    r_mat = tl.load(r_ptr)  # 32x32

    # Rotate: Q_rot = Q @ R
    q_rot_tile = tl.dot(q_tile.to(r_mat.dtype), r_mat)
    if sm_scale is not None:
        q_rot_tile *= sm_scale

    # Store rotated Q
    rot_ptr = (
        Q_rot
        + (pid_b * stride_qob)
        + pid_h * stride_qoh
        + offs_m[:, None] * stride_qom
        + offs_d[None, :] * stride_qod
    )

    # Calculate mean for the block (reduction over d within the BLOCK_M)
    # q_mean shape: [B, H, Q_NUM_BLKS, D]
    if q_smoothing:
        m_row_mean = (
            tl.sum(q_rot_tile, axis=0) / BLOCK_M
        )  # Sum over BLOCK_M -> shape [BLOCK_D]

        q_rot_tile -= m_row_mean[None, :]
        # Store mean (Atomic add or structured store)
        # For simplicity in this layout, we store the block-sum
        # and divide by BLOCK_M in the host or final step
        mean_ptr = (
            Q_mean
            + (pid_b * stride_mb)
            + pid_h * stride_mh
            + pid_m * stride_mm
            + offs_d * stride_md
        )
        tl.store(mean_ptr, m_row_mean)

    tl.store(
        rot_ptr,
        q_rot_tile,
        mask=(offs_m[:, None] < seq_len) & (offs_d[None, :] < d_model),
    )


@triton.jit
def _rot_k_only_kernel(
    K,
    K_rot,
    R,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_kd,
    stride_kob,
    stride_koh,
    stride_kon,
    stride_kod,
    stride_rm,
    stride_rd,
    n_heads,
    seq_k,
    d_model,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_bh = tl.program_id(0).to(tl.int64)
    pid_n = tl.program_id(1).to(tl.int64)
    pid_d = tl.program_id(2).to(tl.int64)

    pid_h = pid_bh % n_heads
    pid_b = pid_bh // n_heads

    offs_n = pid_n * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)

    # Load K block and R
    k_ptr = (
        K
        + (pid_b * stride_kb)
        + (pid_h * stride_kh)
        + offs_n[:, None] * stride_kn
        + offs_d[None, :] * stride_kd
    )
    r_ptr = (
        R
        + tl.arange(0, BLOCK_D)[:, None] * stride_rm
        + tl.arange(0, BLOCK_D)[None, :] * stride_rd
    )

    k_tile = tl.load(
        k_ptr, mask=(offs_n[:, None] < seq_k) & (offs_d[None, :] < d_model), other=0.0
    )
    r_mat = tl.load(r_ptr)

    # Rotate K
    k_rot_tile = tl.dot(k_tile.to(r_mat.dtype), r_mat)

    # Store
    rot_ptr = (
        K_rot
        + (pid_b * stride_kob)
        + pid_h * stride_koh
        + offs_n[:, None] * stride_kon
        + offs_d[None, :] * stride_kod
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
    n_heads_q,
    n_heads_k,
    seq_k,
    d_model,
    BLOCK_N: tl.constexpr,  # Number of K-tokens to process
):
    pid_bh = tl.program_id(0).to(tl.int64)
    pid_m_q = tl.program_id(1).to(tl.int64)  # The Q-block index
    pid_n_k = tl.program_id(2).to(tl.int64)  # The K-block index

    pid_hq = pid_bh % n_heads_q
    pid_b = pid_bh // n_heads_q

    pid_hk = pid_hq // (n_heads_q // n_heads_k)

    offs_n = pid_n_k * BLOCK_N + tl.arange(0, BLOCK_N)

    # Accumulate dot product across the whole d_model
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)

    # Loop over d_model in steps of 32 (our block_size)
    for d_offset in range(0, d_model, 32):
        offs_d = d_offset + tl.arange(0, 32)

        # Load Q_mean segment: [32]
        qm_ptr = (
            Q_mean
            + pid_b * stride_mb
            + pid_hq * stride_mh
            + pid_m_q * stride_mm
            + offs_d * stride_md
        )
        qm_val = tl.load(qm_ptr)

        # Load K_rot segment: [BLOCK_N, 32]
        kn_ptr = (
            K_rot
            + pid_b * stride_kb
            + pid_hk * stride_kh
            + offs_n[:, None] * stride_kn
            + offs_d[None, :] * stride_kd
        )
        kn_val = tl.load(kn_ptr, mask=offs_n[:, None] < seq_k, other=0.0)

        # Compute dot product for this d-segment
        acc += tl.sum(qm_val[None, :] * kn_val, axis=1)

    # Store to Delta_S [B, H, Q_BLKS, seq_k]
    s_ptr = (
        Delta_S
        + pid_b * stride_sb
        + pid_hq * stride_sh
        + pid_m_q * stride_sm
        + offs_n * stride_sn
    )
    tl.store(s_ptr, acc, mask=offs_n < seq_k)


################# Sage V1 quantization kernels ####################


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
):
    pid = tl.program_id(0).to(tl.int64)

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
            sm_scale=sm_scale,
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

        # just apply the per channel v_scales that have been computed outside
        v_scale_ptrs = (
            V_Scale + off_b * stride_vsz + off_h * stride_vsh + offs_d[None, :]
        )
        v = tl.load(v_input_ptrs, mask=offs_kn[:, None] < SEQLEN_K, other=0.0)
        v = v.to(tl.float32)
        v_scales = tl.load(v_scale_ptrs)
        v_quant = v / v_scales
        v_quant = v_quant.to(v_output_ptrs.dtype.element_ty)
        tl.store(v_output_ptrs, v_quant, mask=offs_kn[:, None] < SEQLEN_K)


@triton.jit
def _general_quant_kernel(
    input_ptrs, output_ptrs, scale_ptrs, DTYPE_MAX, mask, sm_scale=None
):
    if mask is not None:
        x = tl.load(input_ptrs, mask=mask, other=0.0)
    else:
        x = tl.load(input_ptrs)
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
