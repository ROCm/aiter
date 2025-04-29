"""
Lean Attention
===============
This is a Triton implementation of the Lean Attention algorithm from https://arxiv.org/abs/2405.10480
Lean Attention adopts streamK style tiling strategy, which efficiently utilize all available CUs in the system.
Lean Attention is for both decode and prefill attention of transformer based models.

It currently supports ragged batching decode and prefill attention with causal=1

TO be added features:
- Add GQA support
- batch_size > 1 for prefill/causal=1
- Misc
    - N_CTX with non-integer number of BLOCK_N (pad zeros or add mask)
    -
"""

import pytest
import torch
import sys

import triton
import triton.language as tl

def persistent_lean_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    Mp: torch.Tensor,
    Lp: torch.Tensor,
    Op: torch.Tensor,
    locks: torch.Tensor,
    batch_num_block_n: torch.Tensor,
    total_programs: int,
    BLOCK_M: int,
    BLOCK_N: int,
    causal: bool,
    # d: int,
    batch_size: int,
    sm_scale: torch.float16,
    num_warps : int,
    waves_per_eu: int,
):
    # shape constraints
    HEAD_DIM_Q, HEAD_DIM_K, HEAD_DIM_V = q.shape[-1], k.shape[-1], v.shape[-1]
    assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V, "Incompatible Q/K/V Hidden Dimensions"
    assert HEAD_DIM_K in {16, 32, 64, 128, 256}

    N_CTX_Q = q.shape[1] // batch_size
    N_CTX_K = k.shape[1]  # This is the sum of all ctx_n in a batch
    H = q.shape[0]

    qk_scale = sm_scale * 1.44269504

    num_m_blocks, high_load_wgs, max_tiles_per_wg, tiles_per_head, total_programs, num_splits, even_split = (
        get_num_splits_and_buffer_sizes(causal, N_CTX_Q, N_CTX_K, H, H, HEAD_DIM_Q, BLOCK_M, BLOCK_N, total_programs)
    )

    grid = (total_programs, 1, 1)

    o = torch.empty_like(q, dtype=v.dtype)

    la_fwd = la_persistent[grid](
        q,
        k,
        v,
        qk_scale,
        Mp,
        Lp,
        Op,
        o,
        batch_num_block_n,
        locks,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        o.stride(0),
        o.stride(1),
        o.stride(2),
        Op.stride(0),
        Op.stride(1),
        Op.stride(2),
        HEAD_DIM=HEAD_DIM_K,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        batch_size=batch_size,
        causal=causal,
        num_m_blocks=num_m_blocks,
        # leanAttention params
        high_load_wgs=high_load_wgs,
        max_tiles_per_wg=max_tiles_per_wg,
        tiles_per_head=tiles_per_head,
        num_splits=num_splits,
        waves_per_eu=waves_per_eu,
        num_warps=waves_per_eu,
    )

    return o


def get_num_splits_and_buffer_sizes(
    causal, max_seqlen_q, max_seqlen_k, num_heads, num_heads_k, head_size, BLOCK_M, BLOCK_N, num_SMs
):
    ##### Lean Atteion: Calculate Splits and Tile Sizes #####
    ## based on onnxruntime/contrib_ops/cuda/bert/lean_attention
    num_m_blocks = (max_seqlen_q + BLOCK_M - 1) // BLOCK_M
    num_n_blocks = (max_seqlen_k + BLOCK_N - 1) // BLOCK_N

    # TODO: Support Grouped-Query Attention
    max_seqlen_q = max_seqlen_q * num_heads // num_heads_k

    #print(f"block_m: {BLOCK_M}, block_n: {BLOCK_N} ")
    #print(f"num_m_block: {num_m_blocks}, num_n_block: {num_n_blocks} ")
    #print(f"max_seqlen_q: {max_seqlen_q}, max_seqlen_k: {max_seqlen_k}")
    #print(f"num_heads: {num_heads}, num_heads_k: {num_heads_k} ")

    if max_seqlen_q == 1:
        causal = False

    tiles_per_head = 0
    if causal:
        # Prefill - Causal
        for i in range(0, num_m_blocks):
            tiles_per_head += (((i + 1) * BLOCK_M) + BLOCK_N - 1) // BLOCK_N
    else:
        # Decode or Not Causal
        tiles_per_head = num_m_blocks * num_n_blocks

    total_tiles = tiles_per_head * num_heads_k  # Total tiles across all heads

    # StreamK Lean has as many threadblocks as SMs
    # This should be a function of tile size and number of scratchpad space
    # LeanAttention assign 3 CTAs per SM (bounded by LDS size)
    lean_griddimz = num_SMs  # CTA launch grid
    # if (total_tiles <= 2 * 2 * num_SMs):
    #    lean_griddimz = min((total_tiles + 1) / 2, (32 * total_tiles + num_n_blocks - 1) / num_n_blocks)
    # else:
    #    lean_griddimz = min(2 * num_SMs, 32 * num_heads_k * batch_size * num_m_blocks)

    # Max number lean tiles per task block (CTA)
    max_tiles_per_tb = (total_tiles + lean_griddimz - 1) // lean_griddimz

    # Find max number of splits
    num_splits = 0
    even_split = False
    if total_tiles % lean_griddimz == 0:
        even_split = True
        num_splits = 1 + ((num_n_blocks + max_tiles_per_tb - 2) // (max_tiles_per_tb))
    else:
        even_split = False
        num_splits = 1 + ((num_n_blocks + max_tiles_per_tb - 3) // (max_tiles_per_tb - 1))

    # high_load_tbs is the remainder of total_tile / num_cta
    high_load_tbs = total_tiles - ((max_tiles_per_tb - 1) * lean_griddimz)

    return num_m_blocks, high_load_tbs, max_tiles_per_tb, tiles_per_head, lean_griddimz, num_splits, even_split

@triton.jit
def find_group(x):
    group_id = 0
    total_blocks = 0
    while total_blocks + (group_id + 1) <= x:
        total_blocks += (group_id + 1)
        group_id += 1
    group_size = group_id + 1
    return group_id, group_size, total_blocks

@triton.jit
def la_persistent(
    Q,
    K,
    V,
    qk_scale,
    Mp,
    Lp,
    Op,
    Out,
    batch_num_block_n,
    locks,
    stride_qh,
    stride_qm,
    stride_qk,
    stride_kh,
    stride_kn,
    stride_kk,
    stride_vh,
    stride_vn,
    stride_vk,
    stride_oh,
    stride_om,
    stride_on,
    stride_oph,
    stride_opm,
    stride_opn,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    batch_size: tl.constexpr,
    causal: tl.constexpr,
    num_m_blocks: tl.constexpr,
    # leanAttention params
    high_load_wgs: tl.constexpr,
    max_tiles_per_wg: tl.constexpr,
    tiles_per_head: tl.constexpr,
    num_splits: tl.constexpr,
):
    current_pid = tl.program_id(0)

    if current_pid < high_load_wgs:
        iter = max_tiles_per_wg * current_pid
        cta_end_tile_gid = iter + max_tiles_per_wg
    else:
        iter = (max_tiles_per_wg - 1) * (current_pid - high_load_wgs) + high_load_wgs * max_tiles_per_wg
        cta_end_tile_gid = iter + (max_tiles_per_wg - 1)

    # Loop context length
    while iter < cta_end_tile_gid:
        # Calculate index of current head output tile
        # The tiles_per_head is the numner of BLOCK_N in the K/V sequence
        tile_head_idx = iter // tiles_per_head

        # To generate an otuput tile, a loop over [tile_iter, tile_iter_end) lean tiles is needed
        # [tile_iter, tile_iter_end) are in the form of global tile id
        if causal:
            #TODO: causal with batch!=1
            per_head_tile_idx, per_head_tile_size, total_blocks = find_group(iter - (tile_head_idx * tiles_per_head))
            tile_iter = tile_head_idx * tiles_per_head + total_blocks
            tile_iter_end = tile_iter + per_head_tile_size
            tile_idx = tile_head_idx * num_m_blocks + per_head_tile_idx
        else:
            tile_idx = tile_head_idx * batch_size
            tile_iter = tile_head_idx * tiles_per_head
            if batch_size == 1:
                req_size = tiles_per_head
            else:
                req_size = tl.load(batch_num_block_n)
            tile_iter_end = tile_iter + req_size
            for b in range(1, batch_size):
                next_req_size = tl.load(batch_num_block_n + b)
                local_head_iter = iter % tiles_per_head
                if (local_head_iter < next_req_size) and (local_head_iter >= req_size):
                    tile_iter = tile_iter + req_size
                    tile_idx = tile_idx + b
                    tile_iter_end = tile_iter + (next_req_size - req_size)
                req_size = next_req_size
        # Local lean tile ID within a loop of an output tile
        local_iter = iter - tile_iter
        local_iter_end = tl.minimum(tile_iter_end, cta_end_tile_gid) - tile_iter

        if iter == tile_iter:
            host_block = True
        else:
            host_block = False
        # finishing_block: the output tile is finished within this block
        if cta_end_tile_gid >= tile_iter_end:
            finishing_block = True
        else:
            finishing_block = False
      
         
        apply_causal = False
        if causal and finishing_block:
            apply_causal = True
        
        kv_block_shape = (local_iter_end - local_iter) * BLOCK_N
        
        if causal:
            kv_offset = tile_head_idx * stride_kh + local_iter * BLOCK_N * stride_kn
        else:
            kv_offset = iter * BLOCK_N * stride_kn
        K_base = K + kv_offset
        V_base = V + kv_offset

        if causal:
            #TODO: batch>1, seqlen_q != seqlen_k
            q_offset = tile_idx * BLOCK_M * stride_qm
            Q_base = Q + q_offset
        else:
            Q_base = Q + tile_idx * (stride_qh // batch_size)

        m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
        acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

 
        acc, l_i, m_i = _attn_lean_tile(
                acc,
                l_i,
                m_i,
                Q_base,
                stride_qm,
                stride_qk,
                kv_block_shape,
                K_base,
                V_base,
                stride_kn,
                stride_kk,
                stride_vn,
                stride_vk,
                qk_scale,
                BLOCK_M,
                BLOCK_N,
                HEAD_DIM,
                apply_causal,
                tile_idx,
                local_iter,
                local_iter_end,
            )
        
        # initialize pointer to m and l
        m_cta = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
        l_cta = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
        acc_cta = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

        # lean output tile epilogue
        offs_m = tl.arange(0, BLOCK_M)
        offs_k = tl.arange(0, HEAD_DIM)

        if not host_block:
            # Update pointers of partial results M[cta], L[cta], O[cta]
            mp_ptrs = Mp + current_pid * BLOCK_M + offs_m
            lp_ptrs = Lp + current_pid * BLOCK_M + offs_m
            op_ptrs = Op + current_pid * stride_oph + offs_m[:, None] * stride_opm + offs_k[None, :] * stride_opn

            tl.store(mp_ptrs, m_i, cache_modifier=".wt")
            tl.store(lp_ptrs, l_i, cache_modifier=".wt")
            tl.store(op_ptrs, acc, cache_modifier=".wt")
            tl.debug_barrier()
            #tl.store(locks + current_pid, 1, cache_modifier=".wt")
            #According to streamK gemm, store + cache_modifier won't work universally
            # atomic_xchg is better solution but a less performant variant
            tl.atomic_xchg(locks + current_pid, 1)

        if host_block and finishing_block:
            # A host block that is also a finishing block completes all the LeanTile iterations for its output tile
            # in a single CTA and so can directly store its results from LeanTile() in global memory without any reduction
            if causal:
                #TODO: assume batch=1 for now
                o_h_offs = Out + tile_idx * BLOCK_M * stride_qm
            else:
                o_h_offs = Out + tile_idx * (stride_oh // batch_size)
            o_ptrs = o_h_offs + offs_m[:, None] * stride_om + offs_k[None, :] * stride_on
            acc = acc / l_i[:, None]
            tl.store(o_ptrs, acc.to(Out.type.element_ty))

        if host_block and not finishing_block:
            # if not finishing_block: # another CTA is processing the end of the output tile and store partial results
            if causal:
                #TODO: assume batch=1 for now
                o_h_offs = Out + tile_idx * BLOCK_M * stride_qm
            else:
                o_h_offs = Out + tile_idx * (stride_oh // batch_size)
            o_ptrs = o_h_offs + offs_m[:, None] * stride_om + offs_k[None, :] * stride_on

            last_cta = current_pid + 1
            temp_end_gid = cta_end_tile_gid
            split = 1
            while (split < num_splits) and (temp_end_gid < tile_iter_end):
                if last_cta < high_load_wgs:
                    if (tile_iter_end - temp_end_gid) < max_tiles_per_wg:
                        temp_end_gid += tile_iter_end - temp_end_gid
                    else:
                        temp_end_gid += max_tiles_per_wg
                else:
                    if (tile_iter_end - temp_end_gid) < (max_tiles_per_wg - 1):
                        temp_end_gid += tile_iter_end - temp_end_gid
                    else:
                        temp_end_gid += max_tiles_per_wg - 1

                last_cta += 1
                split += 1
            # Next, load nonHost partial restult
            for cta in range((current_pid + 1), last_cta):
                #According to treamK gemm, atomic_cas is universal solution but less performant
                while tl.atomic_cas(locks + cta, 1, 1) != 1:
                #while tl.load(locks + cta, cache_modifier=".cv", volatile=True) != 1:
                    pass

                # Partial results are stored in [nonHost, Host-nonFinishing] layout
                offs_mplp = cta * BLOCK_M + tl.arange(0, BLOCK_M)
                mp_ptrs = Mp + offs_mplp
                lp_ptrs = Lp + offs_mplp
                op_h_offs = Op + cta * stride_oph
                op_ptrs = op_h_offs + offs_m[:, None] * stride_opm + offs_k[None, :] * stride_opn
                m_cta = tl.load(mp_ptrs)
                l_cta = tl.load(lp_ptrs)
                acc_cta = tl.load(op_ptrs)

                # m_i is the host CTA's m, m_cta is other nonHost CTA's m
                m_new = tl.maximum(m_cta, m_i)
                alpha = tl.math.exp2(m_cta - m_new)
                alpha1 = tl.math.exp2(m_i - m_new)
                l_new = alpha * l_cta + alpha1 * l_i
                acc = acc_cta * alpha[:, None] + acc * alpha1[:, None]
                # update m, l
                m_i = m_new
                l_i = l_new
            # host non-finishing CTA write final result to memory
            acc = acc / l_i[:, None]
            tl.store(o_ptrs, acc.to(Out.type.element_ty))

        # update iter
        iter = iter + (local_iter_end - local_iter)


@triton.jit
def _attn_lean_tile(
    acc,
    l_i,
    m_i,
    Q_base,
    stride_qm,
    stride_qk,
    kv_block_shape,
    K_base,
    V_base,
    stride_kn,
    stride_kk,
    stride_vn,
    stride_vk,
    qk_scale: tl.constexpr,  #
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    causal: tl.constexpr,
    tile_idx,
    local_iter,
    local_iter_end,
):  #
    Q_block_ptr = tl.make_block_ptr(
        base=Q_base,
        shape=(BLOCK_M, HEAD_DIM),
        strides=(stride_qm, stride_qk),
        offsets=(0, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )

    q = tl.load(Q_block_ptr)

    K_block_ptr = tl.make_block_ptr(
        base=K_base,
        shape=(HEAD_DIM, kv_block_shape),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(HEAD_DIM, BLOCK_N),
        order=(0, 1),  # K parent tensor shape [Z, H, CTX, HEAD_DIM]
    )
    V_block_ptr = tl.make_block_ptr(
        base=V_base,
        shape=(kv_block_shape, HEAD_DIM),
        strides=(stride_vn, stride_vk),
        offsets=(0, 0),
        block_shape=(BLOCK_N, HEAD_DIM),
        order=(1, 0),
    )


    #offs_m = tile_idx * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    for iter in range(local_iter, local_iter_end):
        # -- compute qk ----
        k = tl.load(K_block_ptr, cache_modifier=".cg")
        qk = tl.dot(q, k)
        qk = qk * qk_scale

        if iter==(local_iter_end-1) and causal:
        #if causal:
            #offs_n = iter * BLOCK_N + tl.arange(0, BLOCK_N)
            mask = offs_m[:, None] >= offs_n[None, :]
            #Apply the causal mask
            qk = tl.where(mask, qk, float("-inf"))
        #else:
        #    m_ij = tl.maximum(m_i, tl.max(qk, 1))
        #    alpha = tl.math.exp2(m_i - m_ij)
        #    qk = qk - m_ij[:, None]
        '''
        if causal:
            mask = offs_m[:, None] >= (offs_n[None, :])
            qk = qk * qk_scale + tl.where(mask, 0, -1.0e6)
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk = qk - m_ij[:, None]
        else:
            m_ij = tl.maximum(m_i, tl.max(qk, 1)*qk_scale)
            qk = qk * qk_scale - m_ij[:, None]
        '''
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk = qk - m_ij[:, None]
        p = tl.math.exp2(qk)  # p.shape = [BLOCK_M, BLOCK_N]
        # -- update output accumulator --
        alpha = tl.math.exp2(m_i - m_ij)
        #alpha = tl.math.exp(m_i - m_ij)
        acc = acc * alpha[:, None]  # Scale each row of acc by the corresponding elements in alpha
        v = tl.load(V_block_ptr, cache_modifier=".cg")  # v.shape = [BLOCK_N, HEAD_DIM]
        # v = tl.load(tl.multiple_of(V_block_ptr, (1,16)), cache_modifier=".cg")
        acc += tl.dot(p.to(v.dtype), v)  # acc.shape = [BLOCK_M, HEAD_DIM]
        # -- update l_i
        l_ij = tl.sum(p, 1)  # rowsum(p)
        l_i = l_i * alpha + l_ij
        # update m_i
        m_i = m_ij.to(m_i.dtype)
        # update k/v pointer
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
    
    return acc, l_i, m_i

