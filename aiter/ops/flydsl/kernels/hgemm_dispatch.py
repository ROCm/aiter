from __future__ import annotations

from .hgemm_wmma_gfx950 import compile_hgemm_wmma_kernel


def compile_flydsl_hgemm_kernel(
    dtype: str,
    tile_m: int = 128,
    tile_n: int = 128,
    tile_k: int = 64,
    stages: int = 2,
    split_k: int = 1,
    block_m_warps: int = 2,
    block_n_warps: int = 2,
    block_k_warps: int = 1,
    has_bias: bool = True,
    has_k_tail: bool = True,
    group_m: int = 0,
    policy: str = "ft",
):
    assert policy in ["ft", "ht"]
    assert dtype in ["fp16", "bf16", "fp8_ptpc"]
    if policy == "ft":
        kernel = compile_hgemm_wmma_kernel(
            DTYPE=dtype,
            TILE_M=tile_m,
            TILE_N=tile_n,
            TILE_K=tile_k,
            STAGES=stages,
            SPLIT_K=split_k,
            BLOCK_M_WARPS=block_m_warps,
            BLOCK_N_WARPS=block_n_warps,
            BLOCK_K_WARPS=block_k_warps,
            HAS_BIAS=has_bias,
            HAS_K_TAIL=has_k_tail,
            GROUP_M=group_m,
            USE_HALF_TILE_INTERLEAVED=False,
        )
    elif policy == "ht":
        kernel = compile_hgemm_wmma_kernel(
            DTYPE=dtype,
            TILE_M=tile_m,
            TILE_N=tile_n,
            TILE_K=tile_k,
            STAGES=stages,
            SPLIT_K=split_k,
            BLOCK_M_WARPS=block_m_warps,
            BLOCK_N_WARPS=block_n_warps,
            BLOCK_K_WARPS=block_k_warps,
            HAS_BIAS=has_bias,
            HAS_K_TAIL=has_k_tail,
            GROUP_M=group_m,
            USE_HALF_TILE_INTERLEAVED=True,
        )
    return kernel
