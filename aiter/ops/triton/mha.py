import torch
import triton
import triton.language as tl

from typing import Optional, Tuple

@triton.jit
def cdiv_fn(x, y):
    return (x + y - 1) // y

@triton.jit
def load_fn(ptrs, offset_first, offset_second, boundary_first, boundary_second):
    if offset_first is not None and offset_second is not None:
        mask = (offset_first[:, None] < boundary_first) & \
               (offset_second[None, :] < boundary_second)
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
def compute_alibi_block(alibi_slope, seqlen_q, seqlen_k, offs_m, offs_n, transpose=False):
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
    k_ptrs,
    v_ptrs,
    stride_kn,
    stride_vk,
    stride_sn,
    start_m,
    seqlen_k,
    seqlen_q, 
    dropout_p,
    sd_mask_ptrs,
    dropout_mask_ptrs,
    block_min,
    block_max,
    offs_n_causal,
    masked_blocks,
    n_extra_tokens,
    alibi_slope,
    OFFS_M: tl.constexpr,
    OFFS_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DMODEL_POW2: tl.constexpr,
    SM_SCALE: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    MASK_STEPS: tl.constexpr,
    ENABLE_DROPOUT: tl.constexpr,
    RETURN_SCORES: tl.constexpr,
    PADDED_HEAD: tl.constexpr
):
    RCP_LN2: tl.constexpr = 1.4426950408889634

    # loop over k, v, and update accumulator
    #tl.device_print("q_in",q)
    #tl.device_print("block_min", block_min)
    #tl.device_print("block_max", block_max)
    #tl.device_print("seqlen_k",seqlen_k)
    #tl.device_print("seqlen_q",seqlen_q)
    for start_n in range(block_min, block_max, BLOCK_N):
        # For padded blocks, we will overrun the tensor size if
        # we load all BLOCK_N. For others, the blocks are all within range.
        if MASK_STEPS:
            k_offs_n = start_n + tl.arange(0, BLOCK_N)
        else:
            k_offs_n = None
        k_offs_k = None if not PADDED_HEAD else tl.arange(0, BLOCK_DMODEL_POW2)
        k = load_fn(k_ptrs, k_offs_k, k_offs_n, BLOCK_DMODEL, seqlen_k)
        #tl.device_print("k",k)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        # We start from end of seqlen_k so only the first iteration would need
        # to be checked for padding if it is not a multiple of block_n
        # TODO: This can be optimized to only be true for the padded block.
        if MASK_STEPS:
            # If this is the last block / iteration, we want to
            # mask if the sequence length is not a multiple of block size
            # a solution is to always do BLOCK_M // BLOCK_N + 1 steps if not is_modulo_mn.
            # last step might get wasted but that is okay. check if this masking works For
            # that case.
            if (start_n + BLOCK_N == block_max) and (n_extra_tokens != 0):
                boundary_m = tl.full([BLOCK_M], seqlen_k, dtype=tl.int32)
                size_n = start_n + OFFS_N[None, :]
                mask = size_n < boundary_m[:, None]
                qk = tl.where(mask, qk, float("-inf"))

        # compute masks
        q_mask = (OFFS_M[:, None] < seqlen_q)
        k_mask = ((start_n + tl.arange(0, BLOCK_N))[None, :] < seqlen_k)
        p_mask = q_mask & k_mask

        # -- compute qk ----
        #if IS_FP8 : #TODO
        #    qk += (tl.dot(q, k) * descale_q * descale_k)
        #else:
        qk += tl.dot(q, k)
        qk_scaled =  qk * SM_SCALE
        #print("qk",qk)
        if IS_CAUSAL:
            causal_boundary = start_n + offs_n_causal
            causal_mask = OFFS_M[:, None] >= causal_boundary[None, :]
            qk_scaled = tl.where(causal_mask, qk_scaled, float("-inf"))

        if alibi_slope is not None:
            # Compute the global position of each token within the sequence
            global_m_positions = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
            global_n_positions = start_n + tl.arange(0, BLOCK_N)
            alibi_block = compute_alibi_block(alibi_slope, seqlen_q, seqlen_k, global_m_positions,
                                              global_n_positions)
            qk_scaled += alibi_block
        # get max scores so far
        m_ij = tl.maximum(m_i, tl.max(qk_scaled, 1))

        # scale and subtract max
        q_shifted = qk_scaled - m_ij[:, None]
        
        # Compute scaled QK and softmax probabilities
        p = tl.math.exp2(q_shifted * RCP_LN2)

        # CAVEAT: Must update l_ij before applying dropout
        l_ij = tl.sum(p, 1)
        #if ENABLE_DROPOUT:
        #    rng_output = tl.rand(philox_seed, philox_ptrs)  # TODO: use tl.randint for better performance
        #    dropout_mask = rng_output > dropout_p
        #    tl.store(dropout_mask_ptrs, dropout_mask, mask=p_mask)

        #    # return scores with negative values for dropped vals
        #    sd_mask = tl.where(dropout_mask, p, -p)
        #    tl.store(sd_mask_ptrs, sd_mask, mask=p_mask)

        #    # apply dropout mask in place
        #    p = tl.where(dropout_mask, p, 0.0)
        #elif RETURN_SCORES:
        #    # NOTE: the returned score is not the same as the reference because we need to adjust as we find new maxes per block. We are not doing that
        #    tl.store(sd_mask_ptrs, p, mask=p_mask)
        
        # -- update output accumulator --
        # alpha is an adjustment factor for acc and li as we loop and find new maxes
        # store the diff in maxes to adjust acc and li as we discover new maxes
        m_diff = m_i - m_ij
        alpha = tl.math.exp2(m_diff * RCP_LN2)
        acc = acc * alpha[:, None]
        v = load_fn(v_ptrs, k_offs_n, k_offs_k, seqlen_k, BLOCK_DMODEL)
        #print("v",v)
        # -- update m_i and l_i
        l_i = l_i * alpha + l_ij
        # update m_i and l_i
        m_i = m_ij

        #if IS_FP8:
        #    scale_p, descale_p = compute_fp8_scaling_factors(p, FP8_MAX)
        #    acc += (tl.dot((p * scale_p).to(v.type.element_ty), v) * descale_p * descale_v)
        #else:
        acc += tl.dot(p.to(v.type.element_ty), v)

        #print("acc",acc)
        k_ptrs += BLOCK_N * stride_kn
        v_ptrs += BLOCK_N * stride_vk
        if RETURN_SCORES:
            sd_mask_ptrs += BLOCK_N * stride_sn
        
        #if ENABLE_DROPOUT:
        #    dropout_mask_ptrs += BLOCK_N * stride_sn
        #    philox_ptrs += BLOCK_N * stride_sn
    
    return acc, l_i, m_i


@triton.jit
def _attn_fwd(q_ptr: torch.Tensor, 
            k_ptr: torch.Tensor, 
            v_ptr: torch.Tensor,
            out_ptr: torch.Tensor,
            alibi_slopes_ptr: torch.Tensor,
            s_dmask_ptr: torch.Tensor,
            dropout_mask_ptr: torch.Tensor,
            softmax_lse_ptr: torch.Tensor,
            stride_qz, stride_qh, stride_qm, stride_qk,
            stride_kz, stride_kh, stride_kn, stride_kk,
            stride_vz, stride_vh, stride_vn, stride_vk,
            stride_oz, stride_oh, stride_om, stride_on,
            stride_alibi_z, stride_alibi_h,
            stride_sd_z, stride_sd_h, stride_sd_m, stride_sd_n,
            stride_lse_z, stride_lse_h, stride_lse_m,
            sm_scale,
            dropout_p,
            SEQLEN_Q: tl.constexpr,
            SEQLEN_K: tl.constexpr,
            IS_CAUSAL: tl.constexpr,
            NUM_Q_HEADS: tl.constexpr,
            NUM_K_HEADS: tl.constexpr,
            BLOCK_M: tl.constexpr,
            BLOCK_N: tl.constexpr,
            BLOCK_DMODEL: tl.constexpr,
            BLOCK_DMODEL_POW2: tl.constexpr,
            RETURN_SCORES: tl.constexpr
):
    #tl.device_print("BLOCK_DMODEL",BLOCK_DMODEL)
    #calculate offsets
    start_m = tl.program_id(0) #seqlen_q
    off_q_head = tl.program_id(1)  #num_q_heads
    off_z = tl.program_id(2) #batch

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL_POW2)

    n_blocks = cdiv_fn(SEQLEN_K, BLOCK_N)

    # Now we compute whether we need to exit early due to causal masking.
    # This is because for seqlen_q > seqlen_k, M rows of the attn scores
    # are completely masked, resulting in 0s written to the output, and
    # inf written to LSE. We don't need to do any GEMMs in this case.
    # This block of code determines what N is, and if this WG is operating
    # on those M rows.
    if (IS_CAUSAL):
        # If seqlen_q == seqlen_k, the attn scores are a square matrix.
        # If seqlen_q != seqlen_k, attn scores are rectangular which means
        # the causal mask boundary is bottom right aligned, and ends at either
        # the top edge (seqlen_q < seqlen_k) or left edge.

        # This captures the decrease in n_blocks if we have a rectangular attn matrix
        n_blocks_seqlen = cdiv_fn((start_m + 1) * BLOCK_M + seqlen_k - seqlen_q, BLOCK_N)

        # This is what adjusts the block_max for the current WG, only
        # if IS_CAUSAL. Otherwise we want to always iterate through all n_blocks
        n_blocks = min(n_blocks, n_blocks_seqlen)

        # If we have no blocks after adjusting for seqlen deltas, this WG is part of
        # the blocks that are all 0. We exit early.
        if n_blocks <= 0:
            offs_out = (off_z * stride_oz + 
                        off_q_head * stride_oh + 
                        offs_m[:, None] * stride_om + 
                        offs_d[None, :] * stride_on)
            acc = tl.zeros([BLOCK_M, BLOCK_DMODEL_POW2], dtype=out_ptr.type.element_ty)
            out_mask = (offs_m[:, None] < SEQLEN_Q) & (offs_d < BLOCK_DMODEL)
            tl.store(out_ptr + offs_out, acc, mask=out_mask)

            offs_lse = off_z * stride_lse_z + off_q_head * stride_lse_h + offs_m*stride_lse_m
            lse_mask = offs_m < SEQLEN_Q 
            lse = tl.full([BLOCK_M], value=0.0, dtype=tl.float32)
            tl.store(softmax_lse_ptr + offs_softmax_lse, lse, mask=lse_mask)

            # TODO: Should dropout and return encoded softmax be handled here too?
            return

    grp_sz:tl.constexpr = NUM_Q_HEADS // NUM_K_HEADS 
    if grp_sz != 1: #Grouped Query Attention
        off_k_head = off_q_head / grp_sz 
    else: 
        off_k_head = off_q_head

    #padded_head, n_extra_tokens. Do below

    #q,k,v offsets
    q_offs = (off_z * stride_qz + 
                off_q_head * stride_qh +
                offs_m[:, None] * stride_qm + offs_d[None, :]*stride_qk
    )
    q_ptrs = q_ptr + q_offs

    k_offs = (off_z * stride_kz + 
                off_k_head * stride_kh +
                offs_d[:, None] * stride_kk + offs_n[None, :]*stride_kn
    )
    k_ptrs = k_ptr + k_offs

    v_offs = (off_z * stride_vz + 
                off_k_head * stride_vh +
                offs_n[:, None] * stride_vn + offs_d[None, :]*stride_vk
    )
    v_ptrs = v_ptr + v_offs

    #alibi slopes
    if alibi_slopes_ptr is not None:
        alibi_offs = off_z * stride_alibi_z + off_q_head * stride_alibi_h
        alibi_slope = tl.load(alibi_slopes + alibi_offs)
    else:
        alibi_slope = None

    #s_dmask
    if s_dmask_ptr is not None:
        s_dmask_offs =  (off_z * stride_sd_z + 
                        off_q_head * stride_sd_h + 
                        offs_m[:, None] * stride_sd_m +
                        offs_n[None, :] * stride_sd_n
        )
        s_dmask_ptrs = s_dmask_ptr + s_dmask_offs
    else:
        s_dmask_ptrs = None

    #dropout #TODO
    if dropout_mask_ptr is not None:
        dropout_mask_offs =  (off_z * stride_sd_z + 
                        off_q_head * stride_sd_h + 
                        offs_m[:, None] * stride_sd_m +
                        offs_n[None, :] * stride_sd_n
        )
        dropout_mask_ptrs = dropout_mask_ptr + dropout_mask_offs
    else:
        dropout_mask_ptrs = None

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL_POW2], dtype=tl.float32)
    if (BLOCK_DMODEL == BLOCK_DMODEL_POW2):
        q_mask = (offs_m[:, None] < SEQLEN_Q) 
    else:
        q_mask = (offs_m[:, None] < SEQLEN_Q) & (offs_d[None, :] < BLOCK_DMODEL)
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)
    #tl.device_print("q",q)

    if SEQLEN_K < BLOCK_N:
        n_extra_tokens = BLOCK_N - SEQLEN_K 
    elif SEQLEN_K % BLOCK_N:
        n_extra_tokens = SEQLEN_K % BLOCK_N
    
    #if CAUSAL, then determine masked_blocks and full blocks
    # Here we compute how many full and masked blocks we have.
    padded_block_k = n_extra_tokens != 0
    is_modulo_mn = not padded_block_k and (SEQLEN_Q % BLOCK_M == 0)
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
    #tl.device_print("masked_blocks", masked_blocks)
    #tl.device_print("n_full_blocks", n_full_blocks)
    if n_full_blocks > 0:
        block_max = (n_blocks - masked_blocks) * BLOCK_N
        acc, l_i, m_i = _attn_fwd_inner(acc, 
                                        l_i, 
                                        m_i, 
                                        q, 
                                        k_ptrs, 
                                        v_ptrs, 
                                        stride_kn, 
                                        stride_vk, 
                                        stride_sd_n,
                                        start_m, 
                                        SEQLEN_K, 
                                        SEQLEN_Q, 
                                        dropout_p, 
                                        s_dmask_ptrs, dropout_mask_ptrs,
                                        block_min, block_max, 0, 0, 0, alibi_slope, 
                                        offs_m, offs_n, BLOCK_M, BLOCK_N, BLOCK_DMODEL,BLOCK_DMODEL_POW2,
                                        sm_scale, IS_CAUSAL, MASK_STEPS=False, ENABLE_DROPOUT=dropout_p > 0, 
                                        RETURN_SCORES=RETURN_SCORES, PADDED_HEAD=BLOCK_DMODEL!=BLOCK_DMODEL_POW2)
        block_min = block_max
        block_max = n_blocks * BLOCK_N

      # Remaining blocks, if any, are full / not masked.
    if (masked_blocks > 0):
        if IS_CAUSAL:
            offs_n_causal = offs_n + (seqlen_q - seqlen_k)
        else:
            offs_n_causal = 0
        k_ptrs += n_full_blocks * BLOCK_N * stride_kn
        v_ptrs += n_full_blocks * BLOCK_N * stride_vk
        if RETURN_SCORES:
            s_dmask_ptrs += n_full_blocks * BLOCK_N * stride_sd_n
        #if ENABLE_DROPOUT:
        #    dropout_mask_ptrs += n_full_blocks * BLOCK_N * stride_sd_n
        acc, l_i, m_i = _attn_fwd_inner(acc, 
                                        l_i, 
                                        m_i, 
                                        q, 
                                        k_ptrs, 
                                        v_ptrs, 
                                        stride_kn, stride_vk, stride_sd_n,
                                        start_m, SEQLEN_Q, SEQLEN_K, 
                                        dropout_p, 
                                        s_dmask_ptrs, dropout_mask_ptrs, 
                                        block_min, block_max, offs_n_causal, masked_blocks, n_extra_tokens, alibi_slope, 
                                        offs_m, offs_n, BLOCK_M, BLOCK_N, BLOCK_DMODEL,BLOCK_DMODEL_POW2,
                                        sm_scale, IS_CAUSAL, MASK_STEPS=True, ENABLE_DROPOUT=dropout_p > 0, 
                                        RETURN_SCORES=RETURN_SCORES, PADDED_HEAD=BLOCK_DMODEL!=BLOCK_DMODEL_POW2)
    # epilogue
    # This helps the compiler do Newton Raphson on l_i vs on acc which is much larger.
    l_recip = 1 / l_i[:, None]
    acc = acc * l_recip
    #tl.device_print("acc", acc)
    #if ENABLE_DROPOUT:
    #    dropout_scale = 1 / (1 - dropout_p)
    #    acc = acc * dropout_scale
    # If seqlen_q > seqlen_k but the delta is not a multiple of BLOCK_M,
    # then we have one block with a row of all NaNs which come from computing
    # softmax over a row of all -infs (-inf - inf = NaN). We check for that here
    # and store 0s where there are NaNs as these rows should've been zeroed out.
    end_m_idx = (start_m + 1) * BLOCK_M
    start_m_idx = start_m * BLOCK_M
    causal_start_idx = SEQLEN_Q - SEQLEN_K 
    if IS_CAUSAL:
        if causal_start_idx > start_m_idx and causal_start_idx < end_m_idx:
            out_mask_boundary = tl.full((BLOCK_DMODEL, ), causal_start_idx, dtype=tl.int32)
            mask_m_offsets = start_m_idx + tl.arange(0, BLOCK_M)
            out_ptrs_mask = mask_m_offsets[:, None] >= out_mask_boundary[None, :]
            z = 0.0
            acc = tl.where(out_ptrs_mask, acc, z.to(acc.type.element_ty))

    # write back LSE(Log Sum Exponents), the log of the normalization constant
    overflow_size = end_m_idx - SEQLEN_Q 
    if softmax_lse_ptr is not None: 
        RCP_LN2: tl.constexpr = 1.4426950408889634
        LN2: tl.constexpr = 0.6931471824645996
        # compute log-sum-exp in base 2 units
        mi_base2 = m_i * RCP_LN2
        softmax_lse = mi_base2 + tl.math.log2(l_i)
        # convert back to natural units
        softmax_lse *= LN2
    
        if IS_CAUSAL:
            # zero out nans caused by -infs when doing causal
            lse_causal_mask = (start_m_idx + tl.arange(0, BLOCK_M)) < causal_start_idx
            softmax_lse = tl.where(lse_causal_mask, 0.0, softmax_lse)

        # If seqlen_q not multiple of BLOCK_M, we need to mask out the last few rows.
        # This is only true for the last M block. For others, overflow_size will be -ve
        offs_lse = off_z * stride_lse_z + off_q_head * stride_lse_h + offs_m*stride_lse_m
        if overflow_size > 0:
            boundary = tl.full((BLOCK_M, ), BLOCK_M - overflow_size, dtype=tl.int32)
            lse_mask = tl.arange(0, BLOCK_M) < boundary
            tl.store(softmax_lse_ptr + offs_lse, softmax_lse, mask=lse_mask) # the log of the normalization constant
        else:
            tl.store(softmax_lse_ptr + offs_lse, softmax_lse) # the log of the normalization constant

    # write back O
    offs_out = (off_z * stride_oz + 
                off_q_head * stride_oh + 
                offs_m[:, None] * stride_om + 
                offs_d[None, :] * stride_on) 
    out_mask = tl.full([BLOCK_M, BLOCK_DMODEL_POW2], 1, dtype=tl.int1)
    if overflow_size > 0:
        out_mask = out_mask & (offs_m[:, None] < SEQLEN_Q)
    if BLOCK_DMODEL != BLOCK_DMODEL_POW2:
        out_mask = out_mask & (offs_d[None, :] < BLOCK_DMODEL)
    
    tl.store(out_ptr + offs_out, acc.to(out_ptr.dtype.element_ty), mask=out_mask)

def _flash_attn_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dropout_p: float,
    softmax_scale: float,
    causal: bool,
    window_size_left: int,
    window_size_right: int,
    alibi_slopes: Optional[torch.Tensor],
    return_lse: bool,
    return_softmax: bool
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    #Layout for q,k,v is bshd ie [batch, seq_len, num_head, head_dim] 
    batch, seqlen_q, num_q_heads, head_sz = q.shape
    seqlen_k = k.shape[1]
    num_k_heads = k.shape[2]
    
    #padding for head_dim. Power of 2 or 16
    BLOCK_DMODEL_POW2 = triton.next_power_of_2(head_sz)
    BLOCK_DMODEL_POW2 = max(BLOCK_DMODEL_POW2, 16)

    #softmax_lse [batch, num_q_heads, seqlen_q]
    if return_lse:
        softmax_lse = torch.zeros((batch, num_q_heads, seqlen_q), device=q.device, dtype=torch.float32)
    else:
        softmax_lse = None

    #exp_scores [batch, num_q_heads, seqlen_q, seqlen_k]
    if return_softmax:
        s_dmask = torch.zeros((batch, num_q_heads, seqlen_q, seqlen_k), device=q.device, dtype=torch.float32)
        dropout_mask = torch.zeros((batch, num_q_heads, seqlen_q, seqlen_k), device=q.device, dtype=torch.float32)
    else:
        s_dmask = None
        dropout_mask = None

    o = torch.zeros((batch, num_q_heads, seqlen_q, head_sz), device=q.device, dtype=q.dtype)
    BLOCK_M = 64 #TODO. Add config/tuning support
    BLOCK_N = 64 #TODO Add config/tuning support
    grid = lambda META:(triton.cdiv(seqlen_q, META['BLOCK_M']), num_q_heads, batch)
    _attn_fwd[grid](q,
                    k,
                    v,
                    o,
                    alibi_slopes,
                    s_dmask,
                    dropout_mask,
                    softmax_lse,
                    q.stride(0), #z, batch
                    q.stride(2), #h, num_q_heads
                    q.stride(1), #m, seqlen_q
                    q.stride(3), #k, head_sz
                    k.stride(0), #z, batch
                    k.stride(2), #h, num_k_heads
                    k.stride(1), #m, seqlen_k
                    k.stride(3), #k, head_sz
                    v.stride(0), #z, batch
                    v.stride(2), #h, num_k_heads
                    v.stride(1), #n, seqlen_k
                    v.stride(3), #k, head_sz,
                    o.stride(0), #z, batch
                    o.stride(2), #h, num_q_heads
                    o.stride(1), #n, seqlen_q
                    o.stride(3), #k, head_sz,
                    alibi_slopes.stride(0) if alibi_slopes is not None else 0,
                    alibi_slopes.stride(1) if alibi_slopes is not None else 0,
                    s_dmask.stride(0) if s_dmask is not None else 0,
                    s_dmask.stride(1) if s_dmask is not None else 0,
                    s_dmask.stride(2) if s_dmask is not None else 0,
                    s_dmask.stride(3) if s_dmask is not None else 0,
                    softmax_lse.stride(0) if softmax_lse is not None else 0,
                    softmax_lse.stride(1) if softmax_lse is not None else 0,
                    softmax_lse.stride(2) if softmax_lse is not None else 0,
                    softmax_scale, 
                    dropout_p,
                    SEQLEN_Q=seqlen_q,
                    SEQLEN_K=seqlen_k,
                    IS_CAUSAL=causal,
                    NUM_Q_HEADS=num_q_heads,
                    NUM_K_HEADS=num_k_heads,
                    BLOCK_M=BLOCK_M,
                    BLOCK_N=BLOCK_N,
                    BLOCK_DMODEL=head_sz,
                    BLOCK_DMODEL_POW2=BLOCK_DMODEL_POW2,
                    RETURN_SCORES=return_softmax
    )


    return o, softmax_lse, s_dmask 


class FlashAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        alibi_slopes,
        deterministic,
        return_lse,
        return_softmax,
        is_grad_enabled, #TODO add bkwd support
    ):
        is_grad = is_grad_enabled and any(
            x.requires_grad for x in [q,k,v]
        )
        if softmax_scale is None:
            softmax_scale = q.shape[1] ** (-0.5)
        head_size_og = q.size(3)
        if head_size_og % 8 != 0:
            q = torch.nn.functional.pad(q, [0, 8 - head_size_og % 8])
            k = torch.nn.functional.pad(k, [0, 8 - head_size_og % 8])
            v = torch.nn.functional.pad(v, [0, 8 - head_size_og % 8])
        out_padded, softmax_lse, S_dmask = _flash_attn_forward(
            q,
            k,
            v,
            dropout_p,
            softmax_scale,
            causal=causal,
            window_size_left=window_size[0],
            window_size_right=window_size[1],
            alibi_slopes=alibi_slopes,
            return_lse=return_lse,
            return_softmax=return_softmax and dropout_p > 0,
        )
        '''
        if is_grad:
            ctx.save_for_backward(q, k, v, out_padded, softmax_lse, rng_state)
            ctx.dropout_p = dropout_p
            ctx.softmax_scale = softmax_scale
            ctx.causal = causal
            ctx.window_size = window_size
            ctx.alibi_slopes = alibi_slopes
            ctx.deterministic = deterministic
        '''
        out = out_padded[..., :head_size_og]
        result = [out]
        if return_lse:
            result.append(softmax_lse)
        if return_softmax:
            result.append(S_dmask)

        return tuple(result)

def flash_attn_func(
    q,
    k,
    v,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1,-1),
    alibi_slopes=None,
    deterministic=True,
    return_lse=False,
    return_attn_probs=False,
):
    """dropout_p should be set to 0.0 during evaluation
    Supports multi-query and grouped-query attention (MQA/GQA) by passing in KV with fewer heads
    than Q. Note that the number of heads in Q must be divisible by the number of heads in KV.
    For example, if Q has 6 heads and K, V have 2 heads, head 0, 1, 2 of Q will attention to head
    0 of K, V, and head 3, 4, 5 of Q will attention to head 1 of K, V.

    If causal=True, the causal mask is aligned to the bottom right corner of the attention matrix.
    For example, if seqlen_q = 2 and seqlen_k = 5, the causal mask (1 = keep, 0 = masked out) is:
        1 1 1 1 0
        1 1 1 1 1
    If seqlen_q = 5 and seqlen_k = 2, the causal mask is:
        0 0
        0 0
        0 0
        1 0
        1 1
    If the row of the mask is all zero, the output will be zero.

    If window_size != (-1, -1), implements sliding window local attention. Query at position i
    will only attend to keys between
    [i + seqlen_k - seqlen_q - window_size[0], i + seqlen_k - seqlen_q + window_size[1]] inclusive.

    Arguments:
        q: (batch_size, seqlen, nheads, headdim)
        k: (batch_size, seqlen, nheads_k, headdim)
        v: (batch_size, seqlen, nheads_k, headdim)
        dropout_p: float. Dropout probability.
        softmax_scale: float. The scaling of QK^T before applying softmax.
            Default to 1 / sqrt(headdim).
        causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
        window_size: (left, right). If not (-1, -1), implements sliding window local attention.
        alibi_slopes: (nheads,) or (batch_size, nheads), fp32. A bias of
            (-alibi_slope * |i + seqlen_k - seqlen_q - j|)
            is added to the attention score of query i and key j.
        deterministic: bool. Whether to use the deterministic implementation of the backward pass,
            which is slightly slower and uses more memory. The forward pass is always deterministic.
        return_attn_probs: bool. Whether to return the attention probabilities. This option is for
           testing only. The returned probabilities are not guaranteed to be correct
           (they might not have the right scaling).
    Return:
        out: (batch_size, seqlen, nheads, headdim).
        softmax_lse [optional, if return_lse=True]: (batch_size, nheads, seqlen). The
            logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax
            normalization factor).
        S_dmask [optional, if return_attn_probs=True]: (batch_size, nheads, seqlen, seqlen).
            The output of softmax (possibly with different scaling). It also encodes the dropout
            pattern (negative means that location was dropped, nonnegative means it was kept).
    """
    print(f"window_size={window_size}")
    return FlashAttnFunc.apply(
        q,
        k,
        v,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        alibi_slopes,
        deterministic,
        return_lse,
        return_attn_probs,
        torch.is_grad_enabled()
    )

