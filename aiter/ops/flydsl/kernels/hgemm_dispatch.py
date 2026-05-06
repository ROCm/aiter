from __future__ import annotations

from typing import Optional

from .small_m_hgemm import compile_small_m_hgemm_kernel
from .splitk_hgemm import compile_hgemm_kernel

KERNEL_FAMILY_HGEMM = "hgemm"
KERNEL_FAMILY_SMALL_M = "small_m"


def compile_flydsl_hgemm_kernel(
    dtype: str,
    n: int,
    k: int,
    *,
    kernel_family: Optional[str] = None,
    tile_m: int = 128,
    tile_n: int = 128,
    tile_k: int = 64,
    pack_n: int = 1,
    split_k: int = 1,
    block_m_warps: int = 1,
    block_n_warps: int = 4,
    n_tile_repeat: int = 1,
    persistent_n_tiles: int = 1,
    waves_per_eu: int = 0,
    b_to_lds_unroll: int = 0,
    stages: int = 2,
    async_copy: bool = False,
    b_to_lds: bool = False,
    b_preshuffle: bool = False,
    c_to_lds: bool = False,
    has_bias: bool = False,
):
    """Build one FlyDSL HGEMM-family kernel from a unified config surface."""

    del pack_n, stages, async_copy, c_to_lds

    if kernel_family in (None, KERNEL_FAMILY_HGEMM):
        if b_preshuffle:
            raise ValueError(
                "Generic FlyDSL HGEMM does not support `b_preshuffle=True`"
            )
        return compile_hgemm_kernel(
            dtype,
            n,
            k,
            TILE_M=tile_m,
            TILE_N=tile_n,
            TILE_K=tile_k,
            SPLIT_K=split_k,
            BLOCK_M_WARPS=block_m_warps,
            BLOCK_N_WARPS=block_n_warps,
            B_TO_LDS=b_to_lds,
            HAS_BIAS=has_bias,
        )

    if kernel_family == KERNEL_FAMILY_SMALL_M:
        return compile_small_m_hgemm_kernel(
            dtype,
            n,
            k,
            TILE_N=tile_n,
            TILE_K=tile_k,
            SPLIT_K=split_k,
            BLOCK_N_WARPS=block_n_warps,
            N_TILE_REPEAT=n_tile_repeat,
            PERSISTENT_N_TILES=persistent_n_tiles,
            WAVES_PER_EU_HINT=waves_per_eu,
            B_TO_LDS_UNROLL=b_to_lds_unroll,
            B_TO_LDS=b_to_lds,
            HAS_BIAS=has_bias,
        )

    raise ValueError(
        f"Unsupported kernel_family={kernel_family!r}; expected "
        f"{KERNEL_FAMILY_HGEMM!r} or {KERNEL_FAMILY_SMALL_M!r}"
    )
