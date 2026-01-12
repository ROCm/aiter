import torch

import importlib.util
from pathlib import Path
from aiter.ops.triton._triton_kernels.quant import (
    pod_persistent,
)
from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER = AiterTritonLogger()

file_path = Path("./aiter/ops/triton/lean_atten.py").resolve()
module_name = "la_persistent"
spec = importlib.util.spec_from_file_location(module_name, file_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def pod_attention(
    cu_ctr: torch.Tensor,
    # Decode
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
    # causal: bool,
    batch_size: int,
    sm_scale: torch.float16,
    num_warps,
    waves_per_eu,
    # Prefill
    q_pf: torch.Tensor,
    k_pf: torch.Tensor,
    v_pf: torch.Tensor,
    Mp_pf: torch.Tensor,
    Lp_pf: torch.Tensor,
    Op_pf: torch.Tensor,
    locks_pf: torch.Tensor,
    batch_num_block_n_pf: torch.Tensor,
    BLOCK_M_pf: int,
    BLOCK_N_pf: int,
    # causal_pf: bool,
    batch_size_pf: int,
    prefill_ratio: int,
    decode_ratio: int,
):
    """
    POD (Prefill-On-Decode) fused attention for simultaneous prefill and decode execution.
    Launches persistent kernels that execute both operations concurrently on different CUs
    for improved hardware utilization.

    Args:
        cu_ctr (torch.Tensor): CU (Compute Unit) counter for workload distribution.
        q (torch.Tensor): Decode query with shape (batch_size * 1, num_heads, head_dim).
        k (torch.Tensor): Decode key with shape (total_tokens, num_heads, head_dim).
        v (torch.Tensor): Decode value with shape (total_tokens, num_heads, head_dim).
        Mp (torch.Tensor): Decode partial max buffer with shape (total_programs, BLOCK_M).
        Lp (torch.Tensor): Decode partial sum buffer with shape (total_programs, BLOCK_M).
        Op (torch.Tensor): Decode partial output buffer with shape (total_programs, seq_len, head_dim).
        locks (torch.Tensor): Decode synchronization locks.
        batch_num_block_n (torch.Tensor): Decode cumulative BLOCK_N counts per batch.
        total_programs (int): Total number of thread blocks (CTAs) to launch. Should be 2x the
            number of CUs (one for prefill, one for decode per CU).
        BLOCK_M (int): Decode query tile size.
        BLOCK_N (int): Decode key tile size.
        batch_size (int): Decode batch size.
        sm_scale (torch.float16): Softmax scale, typically 1/sqrt(head_dim).
        num_warps (int): Number of warps per CTA.
        waves_per_eu (int): Number of waves per execution unit.
        q_pf (torch.Tensor): Prefill query with shape (batch_size_pf * seq_len_pf, num_heads, head_dim).
        k_pf (torch.Tensor): Prefill key with shape (total_tokens_pf, num_heads, head_dim).
        v_pf (torch.Tensor): Prefill value with shape (total_tokens_pf, num_heads, head_dim).
        Mp_pf (torch.Tensor): Prefill partial max buffer.
        Lp_pf (torch.Tensor): Prefill partial sum buffer.
        Op_pf (torch.Tensor): Prefill partial output buffer.
        locks_pf (torch.Tensor): Prefill synchronization locks.
        batch_num_block_n_pf (torch.Tensor): Prefill cumulative BLOCK_N counts per batch.
        BLOCK_M_pf (int): Prefill query tile size.
        BLOCK_N_pf (int): Prefill key tile size.
        batch_size_pf (int): Prefill batch size.
        prefill_ratio (int): Ratio of workload assigned to prefill workgroups.
        decode_ratio (int): Ratio of workload assigned to decode workgroups.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: (decode_output, prefill_output) with shapes
            matching respective query tensors.
    """
    _LOGGER.info(
        f"POD_ATTENTION: q={tuple(q.shape)} k={tuple(k.shape)} v={tuple(v.shape)}"
    )
    # shape constraints
    HEAD_DIM_Q, HEAD_DIM_K, HEAD_DIM_V = q.shape[-1], k.shape[-1], v.shape[-1]
    assert (
        HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
    ), "Incompatible Q/K/V Hidden Dimensions"
    assert HEAD_DIM_K in {16, 32, 64, 128, 256}

    # Calculate Decode Params
    N_CTX_Q = q.shape[0] // batch_size
    N_CTX_K = k.shape[0]  # This is the sum of all ctx_n in a batch
    H = q.shape[1]

    qk_scale = sm_scale * 1.44269504

    # We assume the kernel functions fused by pod attention are persistent kernel functions
    # For gfx942, we launch total 608 WGs. Each CU will get 2 WG --- one WG will be doing decode and one prefill
    # For different decode:prefill ratios, assign (decode+prefill)*304 number of WGs
    total_wgs = total_programs // 2

    (
        num_m_blocks,
        num_n_blocks,
        high_load_wgs,
        max_tiles_per_wg,
        tiles_per_head,
        num_splits,
        even_split,
    ) = get_num_splits_and_buffer_sizes(
        False,  # causal
        batch_size,
        N_CTX_Q,
        N_CTX_K,
        H,
        H,
        BLOCK_M,
        BLOCK_N,
        total_wgs,
    )
    # print(" Decode LA params")
    # print(f" num_m_blocks={num_m_blocks}, high_load_wgs={high_load_wgs}, max_tiles_per_wg={max_tiles_per_wg}")
    # print(f" tiles_per_head={tiles_per_head}, total_wgs={total_wgs}")

    o = torch.empty_like(q, dtype=v.dtype)

    # Calculate Prefill Params
    N_CTX_Q_pf = q_pf.shape[0] // batch_size
    N_CTX_K_pf = k_pf.shape[0]  # This is the sum of all ctx_n in a batch

    # MASKED_BLOCKS is used for prefill/causal for BLOCK_M > BLOCK_N
    # For gfx942, BLOCK_M=128, BLOCK_N=64 is better for performance
    MASKED_BLOCKS = BLOCK_M_pf // BLOCK_N_pf

    # if causal_pf:
    # Only support BLOCK_M is multiple of BLOCK_N
    # TODO: add other scenarios
    assert BLOCK_M_pf % BLOCK_N_pf == 0

    #    num_m_blocks_pf, high_load_wgs_pf, max_tiles_per_wg_pf, tiles_per_head_pf, num_splits_pf, even_split_pf = (
    #        get_num_splits_and_buffer_sizes(causal_pf, N_CTX_Q_pf, N_CTX_K_pf, H, H, HEAD_DIM_Q, BLOCK_M_pf, BLOCK_N_pf, total_programs)
    #    )
    (
        num_m_blocks_pf,
        num_n_blocks_pf,
        high_load_wgs_pf,
        max_tiles_per_wg_pf,
        tiles_per_head_pf,
        num_splits_pf,
        even_split_pf,
    ) = get_num_splits_and_buffer_sizes(
        True,  # causal,
        batch_size_pf,
        N_CTX_Q_pf,
        N_CTX_K_pf,
        H,
        H,
        BLOCK_M_pf,
        BLOCK_N_pf,
        total_wgs,
    )
    print("\n Prefill LA params")
    print(
        f" num_m_blocks={num_m_blocks_pf}, high_load_wgs={high_load_wgs_pf}, max_tiles_per_wg={max_tiles_per_wg_pf}"
    )
    print(f" tiles_per_head={tiles_per_head_pf}, total_wgs={total_wgs}")
    print(
        f" BLOCK_M_pf={BLOCK_M_pf}, BLOCK_N_pf={BLOCK_N_pf}, MASKED_BLOCKS={MASKED_BLOCKS}"
    )
    print(
        f" batch_size_pf={batch_size_pf}, num_m_blocks_pf={num_m_blocks_pf}, num_n_blocks_pf={num_n_blocks_pf}"
    )

    print(f" Launching {total_programs} of kernels")

    grid = (total_programs, 1, 1)

    o_pf = torch.empty_like(q_pf, dtype=v_pf.dtype)

    # TODO: need to tune
    max_output_tile_cnt = 16

    pod_kernel = pod_persistent[grid](
        cu_ctr,
        # Decode positional arguments
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
        q.stride(2),  # HEAD_DIM
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
        Op.stride(1),  # N_CTX_Q
        Op.stride(2),  # HEAD_DIM
        # Prefill positional arguments
        q_pf,
        k_pf,
        v_pf,
        Mp_pf,
        Lp_pf,
        Op_pf,
        o_pf,
        batch_num_block_n_pf,
        locks_pf,
        q_pf.stride(0),
        q_pf.stride(1),
        q_pf.stride(2),
        k_pf.stride(0),
        k_pf.stride(1),
        k_pf.stride(2),
        v_pf.stride(0),
        v_pf.stride(1),
        v_pf.stride(2),
        o_pf.stride(0),
        o_pf.stride(1),
        o_pf.stride(2),
        Op_pf.stride(0),
        Op_pf.stride(1),
        Op_pf.stride(2),
        # Decode keyword argument
        HEAD_DIM=HEAD_DIM_K,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        batch_size=batch_size,
        num_m_blocks=num_m_blocks,
        num_n_blocks=num_n_blocks,
        # leanAttention params
        high_load_wgs=high_load_wgs,
        max_tiles_per_wg=max_tiles_per_wg,
        tiles_per_head=tiles_per_head,
        num_splits=num_splits,
        waves_per_eu=waves_per_eu,
        num_warps=num_warps,
        # Prefill keyword argument
        # HEAD_DIM=HEAD_DIM_K,
        BLOCK_M_pf=BLOCK_M_pf,
        BLOCK_N_pf=BLOCK_N_pf,
        MASKED_BLOCKS=MASKED_BLOCKS,
        batch_size_pf=batch_size_pf,
        # causal_pf=causal_pf,
        num_m_blocks_pf=num_m_blocks_pf,
        num_n_blocks_pf=num_n_blocks_pf,
        # leanAttention params
        high_load_wgs_pf=high_load_wgs_pf,
        max_tiles_per_wg_pf=max_tiles_per_wg_pf,
        tiles_per_head_pf=tiles_per_head_pf,
        num_splits_pf=num_splits_pf,
        prefill_ratio=prefill_ratio,
        decode_ratio=decode_ratio,
        max_output_tile_cnt=max_output_tile_cnt,
    )
    # torch.cuda.synchronize()
    print(
        f"pod kernel {pod_kernel.n_regs} registers used, {pod_kernel.n_spills} spills"
    )

    return o, o_pf


def get_num_splits_and_buffer_sizes(
    causal,
    batch_size,
    max_seqlen_q,
    max_seqlen_k,
    num_heads,
    num_heads_k,
    BLOCK_M,
    BLOCK_N,
    num_SMs,
):
    """
    Calculates workload distribution parameters for POD attention stream-K scheduling.
    Similar to Lean Attention scheduling but adapted for POD's dual prefill/decode execution.

    Args:
        causal (bool): Causal masking mode.
        batch_size (int): Batch size.
        max_seqlen_q (int): Maximum query sequence length.
        max_seqlen_k (int): Maximum key sequence length.
        num_heads (int): Number of query heads.
        num_heads_k (int): Number of key/value heads.
        BLOCK_M (int): Query tile size.
        BLOCK_N (int): Key tile size.
        num_SMs (int): Number of streaming multiprocessors (CTAs available).

    Returns:
        Tuple: (num_m_blocks, num_n_blocks, high_load_tbs, max_tiles_per_tb,
            tiles_per_head, num_splits, even_split).
    """
    ##### Lean Attention: Calculate Splits and Tile Sizes #####
    ## based on onnxruntime/contrib_ops/cuda/bert/lean_attention
    num_m_blocks = (max_seqlen_q + BLOCK_M - 1) // BLOCK_M
    num_n_blocks = (max_seqlen_k + BLOCK_N - 1) // BLOCK_N

    # TODO: Support Grouped-Query Attention
    max_seqlen_q = max_seqlen_q * num_heads // num_heads_k

    # print(f"block_m: {BLOCK_M}, block_n: {BLOCK_N} ")
    # print(f"num_m_block: {num_m_blocks}, num_n_block: {num_n_blocks} ")
    # print(f"max_seqlen_q: {max_seqlen_q}, max_seqlen_k: {max_seqlen_k}")
    # print(f"num_heads: {num_heads}, num_heads_k: {num_heads_k} ")
    # print(f"num_SMs: {num_SMs}")

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
    # LeanAttention assign 2 tiles per CTA and 2 CTAs per SM
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
        num_splits = 1 + (
            (num_n_blocks + max_tiles_per_tb - 3) // (max_tiles_per_tb - 1)
        )

    # high_load_tbs is the remainder of total_tile / num_cta
    high_load_tbs = total_tiles - ((max_tiles_per_tb - 1) * lean_griddimz)

    # Needed for causal. This is (per batch n_ctx) // BLOCK_N
    num_n_blocks = num_n_blocks // batch_size

    # print(f"total_tiles={total_tiles}, max_tiles_per_tb={max_tiles_per_tb}, high_load_tbs={high_load_tbs}")
    return (
        num_m_blocks,
        num_n_blocks,
        high_load_tbs,
        max_tiles_per_tb,
        tiles_per_head,
        num_splits,
        even_split,
    )
