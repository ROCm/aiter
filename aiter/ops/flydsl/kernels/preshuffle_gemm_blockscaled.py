"""FP8 blockscaled GEMM: C(M,N) = A(M,K) @ B(N,K).T with DeepGEMM packed-UE8M0 scales.

Preshuffled B layout (N//16, K//64, 4, 16, 16) for gfx950 / MI355X.

Public entry points:
  compile_preshuffle_gemm_blockscaled(M, N, K, tile_m, tile_n, k_unroll, ...)
      Underlying factory. Returns a launcher with ABI
          launch(arg_c, arg_a, arg_b, arg_sa, arg_sb, i32_m, stream)
      where:
        arg_c : bf16 (M, N)
        arg_a : fp8  e4m3 (M, K)            — caller pre-quantizes
        arg_b : fp8  e4m3 (N//16, K//64, 4, 16, 16) preshuffled
        arg_sa: int32 (M, K//512)            — packed UE8M0, 1×128 along K
        arg_sb: int32 (N//128, K//512)       — packed UE8M0, 128×128 blocks
        i32_m : runtime M (currently unused except for assert; M is compile-time)

  compile_preshuffle_gemm_blockscaled_auto(M, N, K, ...)
      Autotune-dispatched. Looks up best tiles in _AUTOTUNE_WINNERS; falls back
      to defaults on miss.

This kernel is a port of fp8_einsum.py (the 4D-einsum kernel that hits ~1896 TF
peak on gfx950). The port:
  - drops the head dim H (this is a 2D GEMM, not an einsum-per-head)
  - renames B→M, D→N, R→K
  - replaces the row-major B loader with a preshuffled-tile loader
  - drops the qz (D-128 group fp8 quant) epilogue — bf16 output only

The packed-UE8M0 scale ABI matches DeepGEMM SM100: 4 UE8M0 exponent bytes
(one per consecutive 128-K-col group) are packed into one i32, little-endian.
"""

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import buffer_ops, const_expr, gpu, range_constexpr, rocdl
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr

from .mfma_epilogues import mfma_epilog
from .mfma_preshuffle_pipeline import (
    swizzle_xor16,
    tile_chunk_coord_i32,
    xcd_remap_bx_by,
)


# ──────────────────────────────────────────────────────────────────────────
# Autotune winners
# ──────────────────────────────────────────────────────────────────────────
# Hand-populated from sweeps run by preshuffle_gemm_blockscaled_perf/autotune.py.
# Key: (M, N, K) → dict(tile_m, tile_n, k_unroll, num_warps, num_stages,
#                       xcd_supergroup_m, waves_per_eu)
# Initially empty — callers fall back to defaults on miss.
_AUTOTUNE_WINNERS: dict[tuple[int, int, int], dict] = {}


def _autotune_lookup(M: int, N: int, K: int):
    """Find the best tuning dict for the given (M, N, K).

    Resolution order:
      1. Exact (M, N, K) match → return dict.
      2. Miss → return None (caller uses defaults).
    """
    if (M, N, K) in _AUTOTUNE_WINNERS:
        return _AUTOTUNE_WINNERS[(M, N, K)]
    return None
