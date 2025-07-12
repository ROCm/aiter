# SPDX-License-Identifier: MIT
# Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

import torch
import sys


# Support tensor in [B, Seqlen, H, d] format. Taking tensors in [B*Seqlen, H, d] as inputs
def persistent_lean_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    Mp: torch.Tensor,
    Lp: torch.Tensor,
    Op: torch.Tensor,  # (total_programs, n_ctx_q, d)
    locks: torch.Tensor,
    batch_num_block_n: torch.Tensor,
    total_programs: int,
    BLOCK_M: int,
    BLOCK_N: int,
    causal: bool,
    batch_size: int,
    sm_scale: torch.float16,
):
    # shape constraints
    HEAD_DIM_Q, HEAD_DIM_K, HEAD_DIM_V = q.shape[-1], k.shape[-1], v.shape[-1]
    assert (
        HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
    ), "Incompatible Q/K/V Hidden Dimensions"
    assert HEAD_DIM_K in {16, 32, 64, 128, 256}

    N_CTX_Q = q.shape[0] // batch_size
    N_CTX_K = k.shape[0]  # This is the sum of all ctx_n in a batch
    H = q.shape[1]

    BLOCK_RATIO = BLOCK_M // BLOCK_N
    print(f"BLOCK_RATIO={BLOCK_RATIO}")

    qk_scale = sm_scale * 1.44269504

    (
        num_m_blocks,
        num_n_blocks,
        high_load_wgs,
        max_tiles_per_wg,
        tiles_per_head,
        total_programs,
        num_splits,
        even_split,
    ) = get_num_splits_and_buffer_sizes(
        causal,
        batch_size,
        N_CTX_Q,
        N_CTX_K,
        H,
        H,
        HEAD_DIM_Q,
        BLOCK_M,
        BLOCK_N,
        total_programs,
    )
    print(
        f"high_load_wgs={high_load_wgs}, max_tiles_per_wg={max_tiles_per_wg}, tiles_per_head={tiles_per_head}"
    )
    print(
        f"total_programs={total_programs}, num_splits={num_splits}, even_split={even_split}"
    )
    print(f"num_m_blocks={num_m_blocks}, num_n_blocks={num_n_blocks}")

    # grid = (total_programs, 1, 1)

    o = torch.empty_like(q, dtype=v.dtype)

    print(
        f"q.stride(0)={q.stride(0)}, q.stride(1)={q.stride(1)}, q.stride(2)={q.stride(2)}"
    )
    print(
        f"k.stride(0)={k.stride(0)}, k.stride(1)={k.stride(1)}, k.stride(2)={k.stride(2)}"
    )

    for pid in range(total_programs):
        la_persistent(
            pid,
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
            q.stride(0),  # N_CTX_Q
            q.stride(1),  # H
            q.stride(2),  # Head_Dim
            k.stride(0),
            k.stride(1),
            k.stride(2),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            o.stride(0),
            o.stride(1),
            o.stride(2),
            Op.stride(0),  # total_programs
            Op.stride(1),  # n_ctx_q
            Op.stride(2),  # head_dim
            HEAD_DIM=HEAD_DIM_K,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_RATIO=BLOCK_RATIO,
            batch_size=batch_size,
            causal=causal,
            num_m_blocks=num_m_blocks,
            num_n_blocks=num_n_blocks,
            # leanAttention params
            high_load_wgs=high_load_wgs,
            max_tiles_per_wg=max_tiles_per_wg,
            tiles_per_head=tiles_per_head,
            num_splits=num_splits,
        )


def get_num_splits_and_buffer_sizes(
    causal,
    batch_size,
    max_seqlen_q,
    max_seqlen_k,
    num_heads,
    num_heads_k,
    head_size,
    BLOCK_M,
    BLOCK_N,
    num_SMs,
):
    ##### Lean Atteion: Calculate Splits and Tile Sizes #####
    ## based on onnxruntime/contrib_ops/cuda/bert/lean_attention
    num_m_blocks = (max_seqlen_q + BLOCK_M - 1) // BLOCK_M
    num_n_blocks = (max_seqlen_k + BLOCK_N - 1) // BLOCK_N

    # TODO: Support Grouped-Query Attention
    max_seqlen_q = max_seqlen_q * num_heads // num_heads_k

    print(f"block_m: {BLOCK_M}, block_n: {BLOCK_N} ")
    print(f"num_m_block: {num_m_blocks}, num_n_block: {num_n_blocks} ")
    print(f"max_seqlen_q: {max_seqlen_q}, max_seqlen_k: {max_seqlen_k}")
    print(f"num_heads: {num_heads}, num_heads_k: {num_heads_k} ")

    if max_seqlen_q == 1:
        causal = False

    tiles_per_head = 0
    if causal:
        # Prefill - Causal
        for i in range(0, num_m_blocks):
            tiles_per_head += (((i + 1) * BLOCK_M) + BLOCK_N - 1) // BLOCK_N
            print(f"tiles_per_head={tiles_per_head}")
        # Does not support ragged batch for causal.
        tiles_per_head = tiles_per_head * batch_size
        print(f"batch_size={batch_size}, tiles_per_head={tiles_per_head}")
    else:
        # Decode or Not Causal
        tiles_per_head = num_m_blocks * num_n_blocks

    total_tiles = tiles_per_head * num_heads_k  # Total tiles across all heads
    print(f"total_tiles={total_tiles}")
    # StreamK Lean has as many threadblocks as SMs
    # This should be a function of tile size and number of scratchpad space
    # LeanAttention assign 3 CTAs per SM (bounded by LDS size)
    lean_griddimz = num_SMs  # CTA launch grid

    # if (total_tiles <= 2 * 2 * num_SMs):
    #    lean_griddimz = min((total_tiles + 1) / 2, (32 * total_tiles + num_n_blocks - 1) / num_n_blocks)
    # else:
    #    lean_griddimz = min(2 * num_SMs, 32 * num_heads_k * batch_size * num_m_blocks)

    # Max number lean tiles per task block (CTA)
    # print(f"total_tiles={total_tiles}")
    max_tiles_per_tb = (total_tiles + lean_griddimz - 1) // lean_griddimz
    # print(f"lean_griddimz={lean_griddimz}, max_tiles_per_tb={max_tiles_per_tb}")

    # Find max number of splits
    num_splits = 0
    even_split = False
    if total_tiles % lean_griddimz == 0:
        even_split = True
        num_splits = 1 + ((num_n_blocks + max_tiles_per_tb - 2) // (max_tiles_per_tb))
    else:
        even_split = False
        num_splits = 1 + (
            (num_n_blocks + max_tiles_per_tb - 3) // (max_tiles_per_tb - 1)
        )

    # high_load_tbs is the remainder of total_tile / num_cta
    high_load_tbs = total_tiles - ((max_tiles_per_tb - 1) * lean_griddimz)

    # Needed for causal. This is (per batch n_ctx) // BLOCK_N
    num_n_blocks = num_n_blocks // batch_size

    return (
        num_m_blocks,
        num_n_blocks,
        high_load_tbs,
        max_tiles_per_tb,
        tiles_per_head,
        lean_griddimz,
        num_splits,
        even_split,
    )


def find_group(x, BLOCK_RATIO):
    group_id = 0
    total_blocks = 0
    while total_blocks + (group_id + 1) * BLOCK_RATIO <= x:
        total_blocks += (group_id + 1) * BLOCK_RATIO
        group_id += 1
        print(f"find_group(): x={x}, group_id={group_id}, total_blocks={total_blocks}")
    group_size = (group_id + 1) * BLOCK_RATIO
    return group_id, group_size, total_blocks


def la_persistent(
    pid,
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
    stride_qm,  # n_ctx_q
    stride_qh,  # Head
    stride_qk,  # head_dim
    stride_kn,
    stride_kh,
    stride_kk,
    stride_vn,
    stride_vh,
    stride_vk,
    stride_om,  # n_ctx_q
    stride_oh,  # Head
    stride_on,  # head_dim
    stride_oph,  # total_programs
    stride_opm,  # n_ctx_q
    stride_opn,  # head_dim
    HEAD_DIM,
    BLOCK_M,
    BLOCK_N,
    BLOCK_RATIO,
    batch_size,
    causal,
    num_m_blocks,
    num_n_blocks,
    # leanAttention params
    high_load_wgs,
    max_tiles_per_wg,
    tiles_per_head,
    num_splits,
):
    current_pid = pid

    if current_pid < high_load_wgs:
        iter = max_tiles_per_wg * current_pid
        cta_end_tile_gid = iter + max_tiles_per_wg
    else:
        iter = (max_tiles_per_wg - 1) * (
            current_pid - high_load_wgs
        ) + high_load_wgs * max_tiles_per_wg
        cta_end_tile_gid = iter + (max_tiles_per_wg - 1)
    print(
        f"current_pid={current_pid}, iter={iter}, cta_end_tile_gid={cta_end_tile_gid}"
    )

    # Loop context length
    while iter < cta_end_tile_gid:
        # Calculate index of current head output tile
        # The tiles_per_head is the sum of # BLOCK_N in K/V sequence of all batches
        tile_head_idx = iter // tiles_per_head
        print(f"    tile_head_idx={tile_head_idx}")
        # To generate an otuput tile, a loop over [tile_iter, tile_iter_end) lean tiles is needed
        # [tile_iter, tile_iter_end) are in the form of global tile id
        if causal:
            tile_batch_idx = (iter % tiles_per_head) // (tiles_per_head // batch_size)
            # Does not support ragged batching. All requests in the batch have the same context length (per_head_tile_size)
            # tiles_per_head: total sum of # BLOCK_N in K/V sequence of all batches
            # per_head_tile_size: per head # BLOCK_N of each output tile
            per_head_tile_idx, per_head_tile_size, total_blocks = find_group(
                iter
                - (tile_head_idx * tiles_per_head)
                - (tile_batch_idx * (tiles_per_head // batch_size)),
                BLOCK_RATIO,
            )
            tile_iter = (
                tile_head_idx * tiles_per_head
                + (tile_batch_idx * (tiles_per_head // batch_size))
                + total_blocks
            )
            tile_iter_end = tile_iter + (per_head_tile_size)
            tile_idx = (
                tile_head_idx * batch_size + tile_batch_idx
            ) * num_m_blocks + per_head_tile_idx
            print(f"    causal: per_head_tile_idx={per_head_tile_idx}")
            print(f"    causal: per_head_tile_size={per_head_tile_size},")
            print(f"    causal: total_blocks={total_blocks}")
            print(f"    causal: tile_batch_idx={tile_batch_idx}")
        else:
            tile_idx = (
                tile_head_idx * batch_size
            )  # Output tile idx, 1 output tile per head per batch
            tile_iter = tile_head_idx * tiles_per_head
            if batch_size == 1:
                req_size = tiles_per_head
            else:
                # req_size = tl.load(batch_num_block_n)
                req_size = batch_num_block_n[0]
            tile_iter_end = tile_iter + req_size
            for b in range(1, batch_size):
                # next_req_size = tl.load(batch_num_block_n + b)
                next_req_size = batch_num_block_n[b]
                local_head_iter = iter % tiles_per_head
                if (local_head_iter < next_req_size) and (local_head_iter >= req_size):
                    tile_iter = tile_iter + req_size
                    tile_idx = tile_idx + b
                    tile_iter_end = tile_iter + (next_req_size - req_size)
                req_size = next_req_size
        print(
            f"    tile_idx={tile_idx}, tile_iter={tile_iter}, tile_iter_end={tile_iter_end}"
        )
        # Local lean tile ID within a loop of an output tile
        local_iter = iter - tile_iter
        # local_iter_end = tl.minimum(tile_iter_end, cta_end_tile_gid) - tile_iter
        local_iter_end = min(tile_iter_end, cta_end_tile_gid) - tile_iter
        print(f"    local_iter={local_iter}, local_iter_end={local_iter_end}")

        if iter == tile_iter:
            host_block = True
        else:
            host_block = False
        # finishing_block: the output tile is finished within this block
        if cta_end_tile_gid >= tile_iter_end:
            finishing_block = True
        else:
            finishing_block = False
        print(f"    host_block={host_block}, finishing_block={finishing_block}")
        offs_m = torch.arange(0, BLOCK_M)
        offs_n = torch.arange(0, BLOCK_N)
        offs_k = torch.arange(0, HEAD_DIM)

        if causal:
            b_seq_size = tile_batch_idx * num_n_blocks
        else:
            tile_batch_idx = tile_idx % batch_size
            b_seq_size = 0
            if tile_batch_idx > 0:
                b_seq_size = 1
                # b_seq_size = tl.load(
                #    batch_num_block_n + tile_batch_idx - 1
                # )  # Previous batch size

        k_offs = (
            (b_seq_size + local_iter) * BLOCK_N * stride_kn
            + tile_head_idx * stride_kh
            + offs_n[None, :] * stride_kn
            + offs_k[:, None] * stride_kk
        )
        v_offs = (
            (b_seq_size + local_iter) * BLOCK_N * stride_vn
            + tile_head_idx * stride_vh
            + offs_n[:, None] * stride_vn
            + offs_k[None, :] * stride_vk
        )
        print(
            f"    b_seq_size={b_seq_size}, k_offs.shape={k_offs.shape}, k_offs={k_offs}"
        )
        print(
            f"    b_seq_size={b_seq_size}, v_offs.shape={v_offs.shape}, v_offs={v_offs}"
        )
        # k_ptrs = K + k_offs
        # k_ptrs = tl.multiple_of(k_ptrs,(16,1))
        # v_ptrs = V + v_offs
        # v_ptrs = tl.multiple_of(v_ptrs,(1,16))

        if causal:
            q_idx = per_head_tile_idx + tile_batch_idx * num_m_blocks
        else:
            q_idx = tile_batch_idx
        q_offs = (
            q_idx * BLOCK_M * stride_qm
            + tile_head_idx * stride_qh
            + offs_m[:, None] * stride_qm
            + offs_k[None, :] * stride_qk
        )
        print(f"    q_idx={q_idx}, q_offs.shape={q_offs.shape}, q_offs={q_offs}")
        o_h_offs = (
            q_idx * BLOCK_M * stride_om
            + tile_head_idx * stride_oh
            + offs_m[:, None] * stride_om
            + offs_k[None, :] * stride_on
        )
        # print(f"    q_idx={q_idx}, o_offs.shape={o_h_offs.shape}, o_offs={o_h_offs}")
        # q_ptrs = Q + q_offs
        # q_ptrs = tl.multiple_of(q_ptrs,(1,16))

        # m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
        # l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
        # acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

        # q = tl.load(q_ptrs)
        offs_m = torch.arange(BLOCK_M)
        # OFFSM = q_idx * BLOCK_M + offs_m
        offs_n = torch.arange(BLOCK_N)
        for l_iter in range(local_iter, local_iter_end):
            """
            if causal:
                if (tile_iter_end - tile_iter) - l_iter <= BLOCK_RATIO:
                    OFFSN = (l_iter + tile_batch_idx * num_n_blocks) * BLOCK_N + offs_n
                    #mask = offs_m[:, None] >= offs_n[None, :]
                    mask = OFFSM[:, None] >= OFFSN[None, :]
                    #torch.set_printoptions(threshold=10000)
            """
            if causal and (BLOCK_RATIO > 1):
                if l_iter == (tile_iter_end - tile_iter) - 2:
                    mask = offs_m[:, None] >= offs_n[None, :]
                    torch.set_printoptions(threshold=10_000)
                    print(f"    Inner loop: l_iter={l_iter}, mask.shape={mask.shape}")
                    torch.set_printoptions(threshold=10_000)
                    print(f"    mask = {mask}")
                    # qk = tl.where(mask, qk, float("-inf"))
                if l_iter == (tile_iter_end - tile_iter) - 1:
                    mask = (offs_m[:, None] >= BLOCK_N) & (
                        offs_n[None, :] <= (offs_m[:, None] - BLOCK_N)
                    )
                    # mask = offs_m[:, None] >= offs_n[None, :]
                    # qk = tl.where(mask, qk, float("-inf"))
                    torch.set_printoptions(threshold=10_000)
                    print(f"    Inner loop: l_iter={l_iter}, mask.shape={mask.shape}")
                    torch.set_printoptions(threshold=10_000)
                    print(f"    mask = {mask}")
            if causal and (BLOCK_RATIO == 1):
                # if (l_iter == (tile_iter_end - tile_iter) - 1):
                if (iter + (l_iter - local_iter)) == (tile_iter_end - 1):
                    mask = offs_m[:, None] >= offs_n[None, :]
                    # qk = tl.where(mask, qk, float("-inf"))

            # if (l_iter == (tile_iter_end - tile_iter) - 1) and causal:
            #    mask = (offs_m[:, None] >= BLOCK_N) & (offs_n[None, :] <= (offs_m[:, None]-BLOCK_N))
            #    print(f"    Inner loop: l_iter={l_iter}, mask = {mask}")

        #    print(f"    Inner Loop: l_iter={l_iter}")
        print(f"    Inner loop: {local_iter} to {local_iter_end}")

        # lean output tile epilogue
        if not host_block:
            # Update pointers of partial results Mp[cta], Lp[cta], Op[cta]
            mp_ptrs = Mp + current_pid * BLOCK_M + offs_m
            lp_ptrs = Lp + current_pid * BLOCK_M + offs_m
            op_ptrs = (
                Op
                + current_pid * stride_oph  # stride_oph is total_program dimension
                + offs_m[:, None] * stride_opm
                + offs_k[None, :] * stride_opn
            )
            print(" Non host block write partial result")
            print(f"mp_ptrs.shape={mp_ptrs.shape}")
            print(f"mp_ptrs={mp_ptrs}")
            print(f"op_ptrs={op_ptrs}")
            # print(f"Mp.shape={Mp.shape}, Mp={Mp}")

            # tl.store(mp_ptrs, m_i, cache_modifier=".wt")
            # tl.store(lp_ptrs, l_i, cache_modifier=".wt")
            # tl.store(op_ptrs, acc, cache_modifier=".wt")
            # tl.debug_barrier()
            # tl.store(locks + current_pid, 1, cache_modifier=".wt")
            # According to streamK gemm, store + cache_modifier won't work universally
            # atomic_xchg is better solution but a less performant variant
            # tl.atomic_xchg(locks + current_pid, 1)

        else:  # host block
            # A host block that is also a finishing block completes all the LeanTile iterations for its output tile
            # in a single CTA and so can directly store its results from LeanTile() in global memory without any reduction

            o_h_offs = (
                q_idx * BLOCK_M * stride_om
                + tile_head_idx * stride_oh
                + offs_m[:, None] * stride_om
                + offs_k[None, :] * stride_on
            )
            print(f"o_h_offs={o_h_offs}")
            # o_ptrs = Out + o_h_offs
            if not finishing_block:
                # if host not finishing_block: # another CTA is processing the end of the output tile and store partial results
                """
                if causal:
                    q_idx = per_head_tile_idx + tile_batch_idx * num_m_blocks
                else:
                    q_idx = tile_batch_idx

                o_h_offs = (
                    q_idx * BLOCK_M * stride_om
                    + tile_head_idx * stride_oh
                    + offs_m[:, None] * stride_om
                    + offs_k[None, :] * stride_on
                )
                o_ptrs = Out + o_h_offs
                """

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
                    print(
                        f"    Host-NonFinishing block cta{cta} loop {current_pid + 1} to {last_cta}"
                    )

                    # Partial results are stored in [nonHost, Host-nonFinishing] layout
                    offs_mplp = cta * BLOCK_M + offs_m
                    mp_ptrs = Mp + offs_mplp
                    lp_ptrs = Lp + offs_mplp
                    op_h_offs = (
                        cta * stride_oph
                        + offs_m[:, None] * stride_opm
                        + offs_k[None, :] * stride_opn
                    )
                    print(f"    Host-NonFinishing block offs_mplp={offs_mplp}")
                    print(f"    Host-NonFinishing block mp_ptrs={mp_ptrs}")
                    print(f"    Host-NonFinishing block lp_ptrs={lp_ptrs}")
                    print(f"    Host-NonFinishing block op_h_offs={op_h_offs}")
                    # op_ptrs = Op + op_h_offs

        # update iter
        iter = iter + (local_iter_end - local_iter)


def main():
    batch = 1
    causal = True
    h = 1
    n_ctx_q = 512
    n_ctx = [512]
    d = 128
    total_programs = 4

    init_dtype = torch.float16
    BLOCK_M = 128
    BLOCK_N = 64
    assert batch == len(n_ctx)

    try:
        sum_n_ctx = sum(int(n) for n in n_ctx)
    except ValueError:
        print(f"N_CTX contains non-numeric values: {n_ctx}")

    print(f"causal={causal}, batch={batch}")
    # N_CTX is a list of context lengthes for all the req in a batch
    # First, calculate #BLOCK_N for each context length "list_num_block_n"
    # Second, Convert it to a list of assumulative lengthes "list_sum_block_n"
    # Third, convert list to a tensor "batch_num_block_n"
    for s in n_ctx:
        list_num_block_n = [
            (int(str(s).strip()) + BLOCK_N - 1) // BLOCK_N for s in n_ctx
        ]
    len_sum = 0
    list_sum_block_n = []
    for i in range(batch):
        len_sum += list_num_block_n[i]
        list_sum_block_n.append(len_sum)
    batch_num_block_n = torch.tensor(list_sum_block_n, dtype=torch.int32)

    sm_scale = 0.5

    # Allocate Tensors
    q = torch.empty((n_ctx_q * batch, h, d), dtype=init_dtype).normal_(
        mean=0.0, std=0.5
    )
    k = torch.empty((sum_n_ctx, h, d), dtype=init_dtype).normal_(mean=0.0, std=0.5)
    v = torch.empty((sum_n_ctx, h, d), dtype=init_dtype).normal_(mean=0.0, std=0.5)

    # LeanAttention Specific Parameters
    # Mp = torch.empty((total_programs, n_ctx_q), device=q.device, dtype=torch.float32)
    # Lp = torch.empty((total_programs, n_ctx_q), device=q.device, dtype=torch.float32)
    # Op = torch.empty((total_programs, n_ctx_q, d), device=q.device, dtype=torch.float32)
    Mp = torch.empty((total_programs, BLOCK_M), device=q.device, dtype=torch.float32)
    Lp = torch.empty((total_programs, BLOCK_M), device=q.device, dtype=torch.float32)
    Op = torch.empty((total_programs, BLOCK_M, d), device=q.device, dtype=torch.float32)

    locks = torch.zeros((total_programs,), device=q.device, dtype=torch.int32)

    # Triton LeanAttention output
    persistent_lean_attention(
        q,
        k,
        v,
        Mp,
        Lp,
        Op,
        locks,
        batch_num_block_n,
        total_programs,
        BLOCK_M,
        BLOCK_N,
        causal,
        batch,
        sm_scale,
    )


if __name__ == "__main__":
    sys.exit(main())
#     benchmark_params = BenchmarkArgs()
#     args = benchmark_params.parse_args()
#     bench_streamk(args.m, args.n, args.k, args.total_programs_streamk, str_to_dtype(args.in_dtype), str_to_dtype(args.out_dtype), args.BLK_M, args.BLK_N, args.BLK_K, args.gsize_m)
