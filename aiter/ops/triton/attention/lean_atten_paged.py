"""
Lean Attention + Paged Attention
================================
This is a Triton implementation of the Lean Attention algorithm from https://arxiv.org/abs/2405.10480, enhanced
with Paged Attention from https://arxiv.org/abs/2309.06180, for the decode phase.
Lean Attention adopts streamK style tiling strategy, which efficiently utilize all available CUs in the system.

It currently supports ragged batching decode

TO be added features:
- Add GQA support
- Misc
    - N_CTX with non-integer number of BLOCK_N (pad zeros or add mask)
    -
"""

import torch

import triton
import triton.language as tl
from aiter.ops.triton._triton_kernels.attention.lean_atten_paged import (
    la_persistent_paged,
)
from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER = AiterTritonLogger()


def persistent_lean_attention_paged(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kv_block_tables: torch.Tensor,
    Mp: torch.Tensor,
    Lp: torch.Tensor,
    Op: torch.Tensor,
    locks: torch.Tensor,
    batch_num_block_n: torch.Tensor,
    total_programs: int,
    BLOCK_M: int,
    BLOCK_N: int,
    # d: int,
    batch_size: int,
    sm_scale: torch.float16,
    num_warps: int,
    waves_per_eu: int,
):
    _LOGGER.info(
        f"LEAN_ATTEN_PAGED: q={tuple(q.shape)}  k={tuple(k.shape)}  v={tuple(v.shape)} Mp={tuple(Mp.shape)} Lp={tuple(Lp.shape)}  Op={tuple(Op.shape)}"
    )
    # shape constraints
    HEAD_DIM_Q, HEAD_DIM_K, HEAD_DIM_V = q.shape[-1], k.shape[-1], v.shape[-1]
    assert (
        HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
    ), "Incompatible Q/K/V Hidden Dimensions"
    assert HEAD_DIM_K in {16, 32, 64, 128, 256}

    N_CTX_Q = q.shape[1] // batch_size
    N_CTX_K = k.shape[1]  # This is the sum of all ctx_n in a batch
    H = q.shape[0]

    qk_scale = sm_scale * 1.44269504

    (
        num_m_blocks,
        high_load_wgs,
        max_tiles_per_wg,
        tiles_per_head,
        total_programs,
        num_splits,
        even_split,
    ) = get_num_splits_and_buffer_sizes(
        N_CTX_Q, N_CTX_K, H, H, HEAD_DIM_Q, BLOCK_M, BLOCK_N, total_programs
    )

    kv_shape = k.shape[1] // BLOCK_N + (1 if k.shape[1] % BLOCK_N != 0 else 0)

    grid = (total_programs, 1, 1)

    o = torch.empty_like(q, dtype=v.dtype)

    la_persistent_paged[grid](
        q,
        k,
        v,
        qk_scale,
        Mp,
        Lp,
        Op,
        o,
        kv_block_tables,
        kv_shape,
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

    max_seqlen_q = max_seqlen_q * num_heads // num_heads_k

    tiles_per_head = 0
    tiles_per_head = num_m_blocks * num_n_blocks

    total_tiles = tiles_per_head * num_heads_k  # Total tiles across all heads

    # StreamK Lean has as many threadblocks as SMs
    # This should be a function of tile size and number of scratchpad space
    # LeanAttention assign 3 CTAs per SM (bounded by LDS size)
    lean_griddimz = num_SMs  # CTA launch grid

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

    return (
        num_m_blocks,
        high_load_tbs,
        max_tiles_per_tb,
        tiles_per_head,
        lean_griddimz,
        num_splits,
        even_split,
    )
