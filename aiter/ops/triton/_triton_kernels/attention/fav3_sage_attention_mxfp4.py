import torch
import triton
import triton.language as tl
from aiter.ops.triton._triton_kernels.flash_attn_triton_amd.common import (
    compute_alibi_block,
)
from aiter.ops.triton.utils._triton.pid_preprocessing import pid_grid_3d, remap_xcd
from aiter.ops.triton._triton_kernels.attention.fav3_sage_attention import (
    compute_alibi_block,
    map_dims,
)
from aiter.ops.triton.moe.quant_moe import downcast_to_mxfp
from aiter.ops.triton._triton_kernels.quant.downcast_to_mxfp_rne import (
    downcast_to_mxfp_rne,
)


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
def _sage_fwd_mxfp4_inner(
    acc,
    l_i,
    m_i,
    q,
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
    encoded_sm_ptrs,
    block_min,
    block_max,
    offs_n_causal,
    masked_blocks,
    n_extra_tokens,
    alibi_slope,
    q_descale,
    k_descale_base_ptrs,
    stride_ksn,
    IS_CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    OFFS_M: tl.constexpr,
    OFFS_N: tl.constexpr,
    PRE_LOAD_V: tl.constexpr,
    MASK_STEPS: tl.constexpr,
    ENABLE_DROPOUT: tl.constexpr,
    RETURN_ENCODED_SOFTMAX: tl.constexpr,
    PADDED_HEAD: tl.constexpr,
    ACTUAL_BLOCK_DMODEL: tl.constexpr,
    ENABLE_PIPELINING: tl.constexpr,
):
    k_descale_ptrs = k_descale_base_ptrs

    # loop over k, v, and update accumulator
    num_stages: tl.constexpr = (
        None if ENABLE_PIPELINING else 1
    )  # Set num_stages==1 if we want to disable pipelining
    for start_n in tl.range(block_min, block_max, BLOCK_N, num_stages=num_stages):
        # For padded blocks, we will overrun the tensor size if
        # we load all BLOCK_N. For others, the blocks are all within range.
        if MASK_STEPS:
            k_offs_n = start_n + tl.arange(0, BLOCK_N)
        else:
            k_offs_n = None
        k_offs_k = None if not PADDED_HEAD else tl.arange(0, BLOCK_DMODEL)
        k = load_fn(k_ptrs, k_offs_k, k_offs_n, ACTUAL_BLOCK_DMODEL, actual_seqlen_k)
        # load k_descale
        # k_descale_ptrs = k_descale_base_ptrs + start_n * stride_ksn
        k_descale = tl.load(k_descale_ptrs)
        k_descale_ptrs += BLOCK_N * stride_ksn

        if PRE_LOAD_V:
            # We can use the same offsets as k, just with dims transposed.
            v = load_fn(
                v_ptrs, k_offs_n, k_offs_k, actual_seqlen_k, ACTUAL_BLOCK_DMODEL
            )
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
                boundary_m = tl.full([BLOCK_M], actual_seqlen_k, dtype=tl.int32)
                size_n = start_n + OFFS_N[None, :]
                mask = size_n < boundary_m[:, None]
                qk = tl.where(mask, qk, float("-inf"))
        if IS_CAUSAL:
            causal_boundary = start_n + offs_n_causal
            causal_mask = OFFS_M[:, None] >= causal_boundary[None, :]
            qk = tl.where(causal_mask, qk, float("-inf"))
        # -- compute qk in mxfp4 ----
        qk += tl.dot_scaled(q, q_descale, "e2m1", k, k_descale, "e2m1", fast_math=True)

        if bias_ptrs is not None:
            bias_offs_n = start_n + tl.arange(0, BLOCK_N) if MASK_STEPS else None
            bias = load_fn(
                bias_ptrs,
                start_m * BLOCK_M,
                bias_offs_n,
                actual_seqlen_q,
                actual_seqlen_k,
            )
            # If bias is added after multiplying qk with sm_scale,
            # our optimization to use 2^x instead of e^x results in an additional
            # scale factor of log2(e) which we must also multiply the bias with.
            qk += bias  # (bias * 1.44269504089 / QK_SCALE)

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
            qk += alibi_block  # scale factor of log2(e)

        # softmax
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        m_ij_scaled = m_ij 
        qk = qk - m_ij_scaled[:, None]
        p = tl.math.exp2(qk)

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
            if RETURN_ENCODED_SOFTMAX:
                tl.store(
                    encoded_sm_ptrs,
                    tl.where(keep, p, -p).to(encoded_sm_ptrs.type.element_ty),
                )
            p = tl.where(keep, p, 0.0)
        elif RETURN_ENCODED_SOFTMAX:
            tl.store(encoded_sm_ptrs, p.to(encoded_sm_ptrs.type.element_ty))
        # -- update output accumulator --
        alpha = tl.math.exp2(m_i - m_ij_scaled)
        acc = acc * alpha[:, None]
        if not PRE_LOAD_V:
            v = load_fn(
                v_ptrs, k_offs_n, k_offs_k, actual_seqlen_k, ACTUAL_BLOCK_DMODEL
            )
        # -- update m_i and l_i
        l_i = l_i * alpha + l_ij
        # update m_i and l_i
        m_i = m_ij

        # PV is done in fp8
        acc += tl.dot(p.to(v.type.element_ty), v)

        k_ptrs += BLOCK_N * stride_kn
        v_ptrs += BLOCK_N * stride_vk
        if bias_ptrs is not None:
            bias_ptrs += BLOCK_N * stride_bn
        if RETURN_ENCODED_SOFTMAX:
            encoded_sm_ptrs += BLOCK_N
    return acc, l_i, m_i


@triton.jit
def sage_fwd_mxfp4(
    Q,
    K,
    V,
    bias,
    SM_SCALE: tl.constexpr,
    L,
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
    stride_qsz,
    stride_qsh,
    stride_qsm,
    stride_qsk,
    stride_ksz,
    stride_ksh,
    stride_ksn,
    stride_ksk,
    stride_vsz,
    stride_vsh,
    stride_vsn,
    Q_descale,
    K_descale,
    V_descale,
    cu_seqlens_q,
    cu_seqlens_k,
    dropout_p,
    philox_seed,
    PERSISTENT: tl.constexpr,
    PERSISTENT_DYNAMIC: tl.constexpr,
    atomic_counter,
    NUM_CU: tl.constexpr,
    GRID_CU_MULTIP: tl.constexpr,
    B: tl.constexpr,
    philox_offset_base,
    encoded_softmax,
    alibi_slopes,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    ACTUAL_BLOCK_DMODEL: tl.constexpr,
    MAX_SEQLENS_Q: tl.constexpr,
    MAX_SEQLENS_K: tl.constexpr,
    VARLEN: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    PRE_LOAD_V: tl.constexpr,
    USE_BIAS: tl.constexpr,
    ENABLE_DROPOUT: tl.constexpr,
    RETURN_ENCODED_SOFTMAX: tl.constexpr,
    USE_ALIBI: tl.constexpr,
    RETURN_LSE: tl.constexpr,
):

    tl.assume(stride_qz >= 0)
    tl.assume(stride_qh >= 0)
    tl.assume(stride_qm >= 0)
    tl.assume(stride_qk >= 0)
    tl.assume(stride_kz >= 0)
    tl.assume(stride_kh >= 0)
    tl.assume(stride_kn >= 0)
    tl.assume(stride_kk >= 0)
    tl.assume(stride_bz >= 0)
    tl.assume(stride_bh >= 0)
    tl.assume(stride_bm >= 0)
    tl.assume(stride_bn >= 0)
    tl.assume(stride_vz >= 0)
    tl.assume(stride_vh >= 0)
    tl.assume(stride_vk >= 0)
    tl.assume(stride_vn >= 0)
    tl.assume(stride_oz >= 0)
    tl.assume(stride_oh >= 0)
    tl.assume(stride_om >= 0)
    tl.assume(stride_on >= 0)
    tl.assume(stride_qsz >= 0)
    tl.assume(stride_qsh >= 0)
    tl.assume(stride_qsm >= 0)
    tl.assume(stride_qsk >= 0)
    tl.assume(stride_ksz >= 0)
    tl.assume(stride_ksh >= 0)
    tl.assume(stride_ksn >= 0)
    tl.assume(stride_ksk >= 0)
    tl.assume(stride_vsz >= 0)
    tl.assume(stride_vsh >= 0)
    tl.assume(stride_vsn >= 0)

    if PERSISTENT:  # if persistent, kernel loops over multiple tiles
        NUM_WG = NUM_CU * GRID_CU_MULTIP  # number of workgroups launched
        num_tiles_per_head = tl.cdiv(
            MAX_SEQLENS_Q, BLOCK_M
        )  # the number of work units (tiles) of a single head
        num_tiles_per_sample = num_tiles_per_head * HQ  # times the number of heads
        num_tiles_total = num_tiles_per_sample * B  # times the number of samples
        if PERSISTENT_DYNAMIC:
            tile_id = atomic_counter.atomic_add(
                1
            )  # retuns the value BEFORE the atomic operation
        else:
            tile_id = tl.program_id(0)
    else:  # standard, kernel processes only one tile
        tile_id = 0
        num_tiles_total = 1

    while tile_id < num_tiles_total:  # loops more than once only if PERSISTENT
        if PERSISTENT:
            # tile id basically tells us the Q block we are handling
            off_z = tile_id // num_tiles_per_sample  # at which batch sample are we
            off_h_q = (
                tile_id % num_tiles_per_sample // num_tiles_per_head
            )  # at which head are we inside the sample
            start_m = (
                tile_id % num_tiles_per_sample % num_tiles_per_head
            )  # at which tile are we inside the head
        else:
            off_h_q = tl.program_id(0)
            # off_h_q = remap_xcd(off_h_q, HQ)
            start_m = tl.program_id(1)
            off_z = tl.program_id(2)

        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, BLOCK_DMODEL // 2)
        offs_dv = tl.arange(0, BLOCK_DMODEL)
        offs_ds = tl.arange(0, BLOCK_DMODEL // 32)

        continue_condition = (
            True  # as we can't have return statements inside while loop in Triton
        )

        if VARLEN:
            cu_seqlens_q_start = tl.load(cu_seqlens_q + off_z)
            cu_seqlens_q_end = tl.load(cu_seqlens_q + off_z + 1)
            seqlen_q = cu_seqlens_q_end - cu_seqlens_q_start
            # We have a one-size-fits-all grid in id(0). Some seqlens might be too
            # small for all start_m so for those we return early.
            if start_m * BLOCK_M > seqlen_q:
                continue_condition = False
                # return
            cu_seqlens_k_start = tl.load(cu_seqlens_k + off_z)
            cu_seqlens_k_end = tl.load(cu_seqlens_k + off_z + 1)
            seqlen_k = cu_seqlens_k_end - cu_seqlens_k_start
        else:
            cu_seqlens_q_start = 0
            cu_seqlens_k_start = 0
            seqlen_q = MAX_SEQLENS_Q
            seqlen_k = MAX_SEQLENS_K

        if continue_condition:
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
                        o_offset
                        + offs_m[:, None] * stride_om
                        + offs_dv[None, :] * stride_on
                    )
                    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=Out.type.element_ty)
                    o_ptrs_mask = (offs_m[:, None] < seqlen_q).broadcast_to(
                        [BLOCK_M, BLOCK_DMODEL]
                    )
                    # We still need to write 0s to the result
                    tl.store(o_ptrs, acc, mask=o_ptrs_mask)
                    # The tensor allocated for L is based on MAX_SEQLENS_Q as that is
                    # statically known.
                    l_ptrs = (
                        L
                        + off_z * HQ * MAX_SEQLENS_Q
                        + off_h_q * MAX_SEQLENS_Q
                        + offs_m
                    )
                    # We store inf to LSE, not -inf because in the bwd pass, we subtract this
                    # from qk which makes it -inf, such that exp(qk - inf) = 0 for these masked blocks.
                    l = tl.full([BLOCK_M], value=float("inf"), dtype=tl.float32)
                    l_ptrs_mask = offs_m < MAX_SEQLENS_Q
                    tl.store(l_ptrs, l, mask=l_ptrs_mask)
                    # TODO: Should dropout and return encoded softmax be handled here too?
                    continue_condition = False
                    # return

            if continue_condition:
                # If MQA / GQA, set the K and V head offsets appropriately.
                GROUP_SIZE: tl.constexpr = HQ // HK
                if GROUP_SIZE != 1:
                    off_h_k = off_h_q // GROUP_SIZE
                else:
                    off_h_k = off_h_q

                n_extra_tokens = 0
                if seqlen_k < BLOCK_N:
                    n_extra_tokens = BLOCK_N - seqlen_k
                elif seqlen_k % BLOCK_N:
                    n_extra_tokens = seqlen_k % BLOCK_N
                PADDED_HEAD: tl.constexpr = ACTUAL_BLOCK_DMODEL != BLOCK_DMODEL

                # Compute pointers for all the tensors used in this kernel.
                q_offset = (
                    Q
                    + off_z * stride_qz
                    + off_h_q * stride_qh
                    + cu_seqlens_q_start * stride_qm
                )
                q_ptrs = (
                    q_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
                )
                k_offset = (
                    K
                    + off_z * stride_kz
                    + off_h_k * stride_kh
                    + cu_seqlens_k_start * stride_kn
                )
                k_ptrs = (
                    k_offset + offs_d[:, None] * stride_kk + offs_n[None, :] * stride_kn
                )
                v_offset = (
                    V
                    + off_z * stride_vz
                    + off_h_k * stride_vh
                    + cu_seqlens_k_start * stride_vk
                )
                v_ptrs = (
                    v_offset
                    + offs_n[:, None] * stride_vk
                    + offs_dv[None, :] * stride_vn
                )
                # Compute pointers for all the scale tensors used in this kernel.
                q_descale_offset = (
                    Q_descale
                    + off_z * stride_qsz
                    + off_h_q * stride_qsh
                    + cu_seqlens_q_start * stride_qsm
                )
                # this is too hacky... correct would be to use offs_ds but the offs_d seems to perform better and not hurt the accuracy
                # TODO: find out why offs_d even works and why it increases perf
                q_descale_ptrs = (
                    q_descale_offset
                    + offs_m[:, None] * stride_qsm
                    + offs_ds[None, :] * stride_qsk
                )

                k_descale_offset = (
                    K_descale
                    + off_z * stride_ksz
                    + off_h_k * stride_ksh
                    + cu_seqlens_k_start * stride_ksn
                )
                k_descale_ptrs = (
                    k_descale_offset
                    + offs_ds[None, :] * stride_ksk
                    + offs_n[:, None] * stride_ksn
                )  # Do not transpose for dot scaled!

                v_descale_offset = V_descale + off_z * stride_vsz + off_h_k * stride_vsh
                v_descale_ptrs = v_descale_offset + offs_dv[None, :] * stride_vsn

                if USE_BIAS:
                    # Note: this might get large enough to overflow on some configs
                    bias_offset = off_z * stride_bz + off_h_q * stride_bh
                    bias_ptrs = (
                        bias
                        + bias_offset
                        + start_m * stride_bm
                        + offs_n[None, :] * stride_bn
                    )
                else:
                    bias_ptrs = None

                if USE_ALIBI:
                    a_offset = off_z * stride_az + off_h_q * stride_ah
                    alibi_slope = tl.load(alibi_slopes + a_offset)
                else:
                    alibi_slope = None

                if ENABLE_DROPOUT:
                    off_hz = off_z * HQ + off_h_q
                    batch_philox_offset = (
                        philox_offset_base + off_hz * seqlen_q * seqlen_k
                    )
                else:
                    batch_philox_offset = 0
                # We can ask to return the dropout mask without actually doing any dropout. In
                # this case, we return an invalid pointer so indicate the mask is not valid.
                if RETURN_ENCODED_SOFTMAX:
                    encoded_sm_base = encoded_softmax + off_h_q * seqlen_q * seqlen_k
                    encoded_sm_ptrs = (
                        encoded_sm_base + offs_m[:, None] * seqlen_k + offs_n[None, :]
                    )
                else:
                    encoded_sm_ptrs = None
                # initialize pointer to m and l
                m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
                l_i = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
                acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
                # scale sm_scale by log_2(e) and use 2^x in the loop as we do not
                # have native e^x support in HW.
                # QK_SCALE: tl.constexpr = SM_SCALE * 1.44269504089
                # Q is loaded once at the beginning and shared by all N blocks.
                q_ptrs_mask = offs_m[:, None] < seqlen_q
                if PADDED_HEAD:
                    q_ptrs_mask = q_ptrs_mask & (
                        offs_d[None, :] < ACTUAL_BLOCK_DMODEL // 2
                    )
                q = tl.load(q_ptrs, mask=q_ptrs_mask, other=0.0)
                q_descale = tl.load(q_descale_ptrs)

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
                    acc, l_i, m_i = _sage_fwd_mxfp4_inner(
                        acc,
                        l_i,
                        m_i,
                        q,
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
                        encoded_sm_ptrs,
                        # _, _, offs_n_causal, masked_blocks, n_extra_tokens, _
                        block_min,
                        block_max,
                        0,
                        0,
                        0,
                        alibi_slope,
                        q_descale,
                        k_descale_ptrs,
                        stride_ksn,
                        # IS_CAUSAL, ....
                        False,
                        BLOCK_M,
                        BLOCK_DMODEL,
                        BLOCK_N,
                        offs_m,
                        offs_n,
                        # _, MASK_STEPS, ...
                        PRE_LOAD_V,
                        False,
                        ENABLE_DROPOUT,
                        RETURN_ENCODED_SOFTMAX,
                        PADDED_HEAD,
                        ACTUAL_BLOCK_DMODEL,
                        True,
                    )
                    block_min = block_max
                    block_max = n_blocks * BLOCK_N

                tl.debug_barrier()
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
                    if RETURN_ENCODED_SOFTMAX:
                        encoded_sm_ptrs += n_full_blocks * BLOCK_N

                    k_descale_ptrs += n_full_blocks * BLOCK_N * stride_ksn

                    acc, l_i, m_i = _sage_fwd_mxfp4_inner(
                        acc,
                        l_i,
                        m_i,
                        q,
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
                        encoded_sm_ptrs,
                        block_min,
                        block_max,
                        offs_n_causal,
                        masked_blocks,
                        n_extra_tokens,
                        alibi_slope,
                        q_descale,
                        k_descale_ptrs,
                        stride_ksn,
                        IS_CAUSAL,
                        BLOCK_M,
                        BLOCK_DMODEL,
                        BLOCK_N,
                        offs_m,
                        offs_n,
                        # _, MASK_STEPS, ...
                        PRE_LOAD_V,
                        True,
                        ENABLE_DROPOUT,
                        RETURN_ENCODED_SOFTMAX,
                        PADDED_HEAD,
                        ACTUAL_BLOCK_DMODEL,
                        False,
                    )

                # apply per channel v_descale outside the loop
                v_descale = tl.load(v_descale_ptrs)
                acc *= v_descale

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

                if IS_CAUSAL:
                    if causal_start_idx > start_m_idx and causal_start_idx < end_m_idx:
                        out_mask_boundary = tl.full(
                            (BLOCK_DMODEL,), causal_start_idx, dtype=tl.int32
                        )
                        mask_m_offsets = start_m_idx + tl.arange(0, BLOCK_M)
                        out_ptrs_mask = (
                            mask_m_offsets[:, None] >= out_mask_boundary[None, :]
                        )
                        z = 0.0
                        acc = tl.where(out_ptrs_mask, acc, z.to(acc.type.element_ty))
                # write back LSE
                overflow_size = end_m_idx - seqlen_q
                if RETURN_LSE:
                    l_ptrs = (
                        L + off_z * HQ * MAX_SEQLENS_Q + off_h_q * MAX_SEQLENS_Q + offs_m
                    )
                    # If seqlen_q not multiple of BLOCK_M, we need to mask out the last few rows.
                    # This is only true for the last M block. For others, overflow_size will be -ve
                    if overflow_size > 0:
                        boundary = tl.full((BLOCK_M, ), BLOCK_M - overflow_size, dtype=tl.int32)
                        l_ptrs_mask = tl.arange(0, BLOCK_M) < boundary
                        tl.store(l_ptrs, m_i + tl.math.log2(l_i), mask=l_ptrs_mask)
                    else:
                        tl.store(l_ptrs, m_i + tl.math.log2(l_i))

                # write back O
                o_offset = (
                    Out
                    + off_z * stride_oz
                    + off_h_q * stride_oh
                    + cu_seqlens_q_start * stride_om
                )
                o_ptrs = (
                    o_offset
                    + offs_m[:, None] * stride_om
                    + offs_dv[None, :] * stride_on
                )
                o_ptrs_mask = tl.full([BLOCK_M, BLOCK_DMODEL], 1, dtype=tl.int1)
                if overflow_size > 0:
                    o_ptrs_mask = o_ptrs_mask & (offs_m[:, None] < seqlen_q)
                if PADDED_HEAD:
                    o_ptrs_mask = o_ptrs_mask & (offs_dv[None, :] < ACTUAL_BLOCK_DMODEL)
                tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=o_ptrs_mask)

        if PERSISTENT:
            if PERSISTENT_DYNAMIC:
                tile_id = atomic_counter.atomic_add(1)
            else:
                tile_id += NUM_WG
        else:
            tile_id = num_tiles_total  # break after single tile


def sage_quant_mxfp4(
    q,
    k,
    v,
    FP8_TYPE,
    FP8_MAX,
    BLKQ=128,
    BLKK=64,
    sm_scale=None,
    q_smoothing=False,
    layout="bshd",
    USE_RNE=False,
):
    v_fp8 = torch.empty_like(v, dtype=FP8_TYPE, device=v.device)

    if layout == "bhsd":
        b, h_qo, qo_len, head_dim = q.shape
        _, h_kv, kv_len, _ = k.shape

        stride_bz_k, stride_h_k, stride_seq_k = k.stride(0), k.stride(1), k.stride(2)

    elif layout == "bshd":
        b, qo_len, h_qo, head_dim = q.shape
        _, kv_len, h_kv, _ = k.shape

        stride_bz_k, stride_h_k, stride_seq_k = k.stride(0), k.stride(2), k.stride(1)
    else:
        raise ValueError(f"Unknown tensor layout: {layout}")
    K_NUM_BLKS = (kv_len + BLKK - 1) // BLKK

    # Apply K tensor smoothing following SageAttention approach
    v_scale = v.abs().amax(dim=1 if layout == "bshd" else 2).to(torch.float32) / FP8_MAX

    v_task_count = b * h_kv * K_NUM_BLKS
    grid = (v_task_count,)

    padded_head_dim = max(16, 1 << (head_dim - 1).bit_length())

    if sm_scale is None:
        sm_scale = head_dim**-0.5

    q, k, delta_s = rotation_smooth_qk(
        q,
        k,
        BLKQ,
        block_size=padded_head_dim,
        q_smoothing=q_smoothing,
        layout=layout,
        sm_scale=(sm_scale * 1.4426950408889634),
    )

    sage_quant_v_kernel[grid](
        v,
        v_fp8,
        v_scale,
        stride_bz_k,
        stride_h_k,
        stride_seq_k,
        v_scale.stride(0),
        v_scale.stride(1),
        b,
        h_kv,
        K_NUM_BLKS,
        kv_len,
        D=head_dim,
        BLK_K=BLKK,
        num_stages=3,
        num_warps=8,
    )

    downcast_func = downcast_to_mxfp_rne if USE_RNE else downcast_to_mxfp

    q_fp4, q_scale = downcast_func(q, torch.uint8, axis=-1)
    k_fp4, k_scale = downcast_func(k, torch.uint8, axis=-1)

    return q_fp4, q_scale, k_fp4, k_scale, v_fp8, v_scale, delta_s


@triton.jit
def sage_quant_v_kernel(
    V_Input,
    V_Output,
    V_Scale,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_vsz,
    stride_vsh,
    BATCH,
    K_HEAD,
    K_NUM_BLKS,
    SEQLEN_K,
    D: tl.constexpr,
    BLK_K: tl.constexpr,
):
    pid = tl.program_id(0)

    offs_blk_k = tl.arange(0, BLK_K)
    offs_d = tl.arange(0, D)

    # V
    off_blk, off_h, off_b = pid_grid_3d(pid, K_NUM_BLKS, K_HEAD, BATCH)
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
    v_scale_ptrs = V_Scale + off_b * stride_vsz + off_h * stride_vsh + offs_d[None, :]
    v = tl.load(v_input_ptrs, mask=offs_kn[:, None] < SEQLEN_K, other=0.0)
    v = v.to(tl.float32)
    v_scales = tl.load(v_scale_ptrs)
    v_quant = v / v_scales
    v_quant = v_quant.to(v_output_ptrs.dtype.element_ty)
    tl.store(v_output_ptrs, v_quant, mask=offs_kn[:, None] < SEQLEN_K)


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
    stride_mb,
    stride_mh,
    stride_mm,
    stride_md,
    n_heads,
    seq_len,
    d_model,
    q_smoothing: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,  # BLOCK_D is 32
):
    # Grid: (batch * n_heads, seq_len // BLOCK_M, d_model // BLOCK_D)
    pid_bh = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_d = tl.program_id(2)

    pid_h = pid_bh % n_heads
    pid_b = pid_bh // n_heads

    # Offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)

    # Load Q block and R (Hadamard)
    # Q block shape: [BLOCK_M, BLOCK_D]
    q_ptr = (
        Q
        + pid_b * stride_qb
        + pid_h * stride_qh
        + offs_m[:, None] * stride_qm
        + offs_d[None, :] * stride_qd
    )
    r_ptr = (
        R + tl.arange(0, BLOCK_D)[:, None] * BLOCK_D + tl.arange(0, BLOCK_D)[None, :]
    )
    q_tile = tl.load(
        q_ptr, mask=(offs_m[:, None] < seq_len) & (offs_d[None, :] < d_model), other=0.0
    )
    r_mat = tl.load(r_ptr)  # 32x32

    # Rotate: Q_rot = Q @ R
    q_rot_tile = tl.dot(q_tile, r_mat)
    if sm_scale is not None:
        q_rot_tile *= sm_scale

    # Store rotated Q
    rot_ptr = (
        Q_rot
        + pid_b * stride_qb
        + pid_h * stride_qh
        + offs_m[:, None] * stride_qm
        + offs_d[None, :] * stride_qd
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
            + pid_b * stride_mb
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
    n_heads,
    seq_k,
    d_model,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_d = tl.program_id(2)

    pid_h = pid_bh % n_heads
    pid_b = pid_bh // n_heads

    offs_n = pid_n * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)

    # Load K block and R
    k_ptr = (
        K
        + pid_b * stride_kb
        + pid_h * stride_kh
        + offs_n[:, None] * stride_kn
        + offs_d[None, :] * stride_kd
    )
    r_ptr = (
        R + tl.arange(0, BLOCK_D)[:, None] * BLOCK_D + tl.arange(0, BLOCK_D)[None, :]
    )

    k_tile = tl.load(
        k_ptr, mask=(offs_n[:, None] < seq_k) & (offs_d[None, :] < d_model), other=0.0
    )
    r_mat = tl.load(r_ptr)

    # Rotate K
    k_rot_tile = tl.dot(k_tile, r_mat)

    # Store
    rot_ptr = (
        K_rot
        + pid_b * stride_kb
        + pid_h * stride_kh
        + offs_n[:, None] * stride_kn
        + offs_d[None, :] * stride_kd
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
    n_heads,
    seq_k,
    d_model,
    BLOCK_N: tl.constexpr,  # Number of K-tokens to process
):
    pid_bh = tl.program_id(0)
    pid_m_q = tl.program_id(1)  # The Q-block index
    pid_n_k = tl.program_id(2)  # The K-block index

    pid_h = pid_bh % n_heads
    pid_b = pid_bh // n_heads

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
            + pid_h * stride_mh
            + pid_m_q * stride_mm
            + offs_d * stride_md
        )
        qm_val = tl.load(qm_ptr)

        # Load K_rot segment: [BLOCK_N, 32]
        kn_ptr = (
            K_rot
            + pid_b * stride_kb
            + pid_h * stride_kh
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
        + pid_h * stride_sh
        + pid_m_q * stride_sm
        + offs_n * stride_sn
    )
    tl.store(s_ptr, acc, mask=offs_n < seq_k)


def create_hadamard_matrix(block_size, device="cuda", dtype=torch.float32):
    """
    Create an orthogonal Hadamard matrix of size block_size x block_size.
    Uses Sylvester's recursive construction and normalizes to be orthogonal.

    Args:
        block_size: Size of the matrix (must be a power of 2)

    Returns:
        Orthogonal Hadamard matrix of shape (block_size, block_size)
        Satisfies: H @ H.T = I (identity matrix)

    Example:
        H_2 = [[1,  1],
               [1, -1]] / sqrt(2)

        H_4 = [[1,  1,  1,  1],
               [1, -1,  1, -1],
               [1,  1, -1, -1],
               [1, -1, -1,  1]] / 2
    """
    assert (block_size & (block_size - 1)) == 0, "block_size must be power of 2"
    assert block_size > 0, "block_size must be positive"

    # Base case: H_1 = [1]
    if block_size == 1:
        return torch.ones(1, 1, device=device, dtype=dtype)

    # Recursive construction: H_{2n} = [H_n   H_n  ]
    #                                   [H_n  -H_n ]
    H_half = create_hadamard_matrix(block_size // 2, device=device, dtype=dtype)

    # Build the full matrix (unnormalized)
    H = torch.zeros(block_size, block_size, device=device, dtype=dtype)
    half = block_size // 2
    H[:half, :half] = H_half
    H[:half, half:] = H_half
    H[half:, :half] = H_half
    H[half:, half:] = -H_half

    # Normalize to make it orthogonal: H @ H.T = I
    # The unnormalized matrix satisfies H_unnorm @ H_unnorm.T = block_size * I
    # So divide by sqrt(block_size) to get orthogonal matrix
    # H = H / (2.0 ** 0.5)  # Divide by sqrt(2) since we doubled the size

    return H


def rotation_smooth_qk(
    q,
    k,
    BLOCK_SIZE_M=256,
    block_size=32,
    q_smoothing=False,
    sm_scale=None,
    layout="bhsd",
):
    # Generate Hadamard Matrix R (Rank 32)
    # TODO we might want to manually define this matrix
    R = create_hadamard_matrix(block_size, dtype=q.dtype) / (block_size**0.5)
    bshd = [0, 1, 2, 3] if layout == "bshd" else [0, 2, 1, 3]

    # shapes
    b, s_q, h_q, d = map_dims(q.shape, bshd)
    _, s_k, h_k, _ = map_dims(k.shape, bshd)

    Q_rot = torch.empty_like(q)
    K_rot = torch.empty_like(k)

    Q_NUM_BLKS = (s_q + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    K_NUM_BLKS = (s_k + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M

    # TODO check the dtypes for scales
    if q_smoothing:
        q_mean = torch.empty(
            (b, h_q, Q_NUM_BLKS, d), dtype=torch.float32, device=q.device
        )
        delta_s = torch.empty(
            (b, h_q, Q_NUM_BLKS, s_k), dtype=torch.float32, device=q.device
        )
    else:
        q_mean = None
        delta_s = None

    stride_qb, stride_qm, stride_qh, stride_qd = map_dims(q.stride(), bshd)
    stride_kb, stride_kn, stride_kh, stride_kd = map_dims(k.stride(), bshd)

    # Launch Q Kernel
    grid_q = (b * h_q, Q_NUM_BLKS, d // block_size)
    _rot_q_kernel[grid_q](
        q,
        Q_rot,
        q_mean,
        R,
        sm_scale,
        stride_qb,
        stride_qh,
        stride_qm,
        stride_qd,
        q_mean.stride(0) if q_smoothing else None,
        q_mean.stride(1) if q_smoothing else None,
        q_mean.stride(2) if q_smoothing else None,
        q_mean.stride(3) if q_smoothing else None,
        h_q,
        s_q,
        d,
        q_smoothing=q_smoothing,
        BLOCK_M=BLOCK_SIZE_M,
        BLOCK_D=block_size,
    )

    # 2. Rotate K (Only once!)
    grid_k = (b * h_k, K_NUM_BLKS, d // block_size)
    _rot_k_only_kernel[grid_k](
        k,
        K_rot,
        R,
        stride_kb,
        stride_kh,
        stride_kn,
        stride_kd,
        h_k,
        s_k,
        d,
        BLOCK_M=BLOCK_SIZE_M,
        BLOCK_D=block_size,
    )

    # smooth k after rotation
    k = k - k.mean(dim=1 if layout == "bshd" else 2, keepdim=True)

    if q_smoothing:
        # 3. Compute Smoothing Delta S
        # Grid: Each Q-block x Each K-block
        grid_delta = (b * h_k, Q_NUM_BLKS, K_NUM_BLKS)
        _compute_delta_s_kernel[grid_delta](
            q_mean,
            K_rot,
            delta_s,
            q_mean.stride(0),
            q_mean.stride(1),
            q_mean.stride(2),
            q_mean.stride(3),
            stride_kb,
            stride_kh,
            stride_kn,
            stride_kd,
            delta_s.stride(0),
            delta_s.stride(1),
            delta_s.stride(2),
            delta_s.stride(3),
            h_k,
            s_k,
            d,
            BLOCK_N=BLOCK_SIZE_M,
        )

    return Q_rot, K_rot, delta_s
