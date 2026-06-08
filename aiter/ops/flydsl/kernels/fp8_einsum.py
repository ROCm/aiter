# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Clean, submission-ready FP8 blockscaled batched einsum for gfx950 / MI355X.

Computes  Z[b,h,d] = sum_r X[b,h,r] * Y[h,d,r]   (einsum 'bhr,hdr->bhd')
in FP8 e4m3 with DeepGEMM-style packed-UE8M0 per-128-K block scales.

This is the "reg-B" path: the proven-correct and peak-performant pipeline.
  - A (X tile)  : gmem -> LDS via raw_ptr_buffer_load_lds (compiler intrinsic,
                  no register transit; swizzle-the-source for bank-conflict-free
                  ds_read). Double-buffered ping/pong across K-tiles.
  - B (Y tile)  : gmem -> registers via load_b_tile (carried across iters).
  - sx (X scale): staged once per WG to a small LDS slab, then read per K-tile.
  - sy (Y scale): gmem -> registers per K-tile (tiny; packed UE8M0).
  - Scale fed DIRECTLY to the MFMA (packed UE8M0 via the opsel byte index).
  - Workgroup barrier sits AFTER compute_tile (WAR protection on the ping/pong
    A LDS slab — the next K-tile's DMA must not overwrite a slab still being
    read by a slower wave).

Supports: bf16 output (default) and in-kernel D-128 group fp8 quant output
(quant_output=True), each with optional split-K accumulation (split_k>1).

Public entry points (all in this module):
  compile_fp8_einsum_clean_ue8m0(*, H, D, R, tile_m, tile_n, tile_k,
                                 block_swizzle_n=0, quant_output=False,
                                 quant_transpose_scale=False, split_k=1)
      Underlying factory. Returns a launcher for either bf16 output (default)
      or in-kernel D-128 group fp8 quant output (quant_output=True). split_k>1
      returns a host wrapper that accumulates fp32 partials then casts to bf16.

  compile_fp8_einsum_clean_ue8m0_qz(*, H, D, R, ..., transpose_scale=False)
      Convenience wrapper — quant_output=True. The qz launcher signature adds
      `arg_sz` (fp32 D-128 group scale) between `arg_z` and `arg_x`.

  compile_fp8_einsum_clean_ue8m0_qz_splitk(*, H, D, R, ..., split_k)
      qz + split-K convenience wrapper.

  compile_fp8_einsum_clean_ue8m0_auto(*, H, D, R, B=None, ...)
  compile_fp8_einsum_clean_ue8m0_qz_auto(*, H, D, R, B=None, ...)
      Autotune-dispatched: looks up the best
      (tile_m, tile_n, tile_k, bsw, split_k) from the per-shape table at the top
      of this file and forwards to the underlying factory. Use these by default
      unless you need to override the tile.

  fp8_einsum(x, y, sx, sy, *, out_dtype=..., ...)
      User-facing entry: validates, compiles+caches, allocates output, launches.

ABI (DeepGEMM SM100 packed-UE8M0):
  arg_z : bf16 (B, H, D)                  (qz: fp8 e4m3 (B, H, D) + arg_sz)
  arg_x : fp8 e4m3 (B, H, R)             — caller pre-quantizes
  arg_y : fp8 e4m3 preshuffled per head, layout (n0, k0, klane=4, nlane=16, kpack=16)
  arg_sx: int32 (B, H, R // 512)         — 4 packed UE8M0 bytes per i32 along K
  arg_sy: int32 (H, D // 128, R // 512)  — same packing along K
  i32_b : runtime batch size
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
# Autotune winners (per-shape best tiles)
# ──────────────────────────────────────────────────────────────────────────
# Hand-populated from the autotune sweeps in
#   /tmp/autotune_qz_h16.log     (qz + bf16 at H=16 D=1024 R=4096)
#   /tmp/autotune_qz_peak5.log   (qz at H=8 D=R=8192, 5 peak shapes)
#   /tmp/autotune_bf_peak3.py    (bf16 at H=8 D=R=8192, B ∈ {128,512,2048})
#
# Key:   (H, D, R, B) → (tile_m, tile_n, tile_k, block_swizzle_n)
# Lookup is done by `_autotune_lookup(...)` below; missing-B entries fall
# back to the nearest tabulated B for the same (H, D, R) shape.
#
# Regenerate by running the autotune scripts above and pasting winners here.
# These tiles have been validated for correctness + were time-best on
_AUTOTUNE_WINNERS_BF16 = {
    # (H,  D,    R,    B    ) -> (tm, tn, tk, bsw, split_k)
    # Retuned 2026-06-08 by rocprof COLD-CACHE kernel time (autotune_ready.py
    # --cold: 512MB cache flush between reps, median duration). Cold-cache
    # reflects real workloads where other kernels evict the inputs from L2.
    # NOTE: supersedes a 2026-06-07 table whose small-B split_k>=2 winners
    # (e.g. H2 B64 sk4@6.48us) were NOT reproducible under a fresh cold measure
    # — they re-measure ~12-16us and split_k=1 actually wins at small B. The
    # stale numbers had skewed the qz1/qz2 crossover (it borrows these configs
    # for the qz2 GEMM). split_k>1 now wins only at a few tiny B.
    # H=16 D=1024 R=4096 (TP=1)
    (16, 1024, 4096, 1): (32, 64, 256, 0, 1),
    (16, 1024, 4096, 2): (32, 64, 256, 0, 1),
    (16, 1024, 4096, 4): (32, 64, 256, 0, 1),
    (16, 1024, 4096, 8): (32, 64, 256, 0, 1),
    (16, 1024, 4096, 16): (32, 64, 256, 0, 1),
    (16, 1024, 4096, 32): (32, 64, 256, 0, 1),
    (16, 1024, 4096, 64): (32, 64, 256, 4, 1),
    (16, 1024, 4096, 128): (64, 64, 256, 4, 1),
    (16, 1024, 4096, 256): (128, 64, 256, 4, 1),
    (16, 1024, 4096, 512): (128, 128, 256, 4, 1),
    (16, 1024, 4096, 1024): (128, 128, 128, 4, 1),
    (16, 1024, 4096, 4096): (128, 128, 256, 4, 1),
    (16, 1024, 4096, 8192): (128, 128, 256, 4, 1),
    (16, 1024, 4096, 16384): (128, 128, 256, 4, 1),
    (16, 1024, 4096, 32768): (128, 128, 256, 4, 1),
    # H=8 D=1024 R=4096 (TP=2)
    (8, 1024, 4096, 1): (32, 64, 256, 0, 1),
    (8, 1024, 4096, 2): (32, 64, 512, 0, 1),
    (8, 1024, 4096, 4): (32, 64, 512, 0, 1),
    (8, 1024, 4096, 8): (32, 64, 256, 4, 1),
    (8, 1024, 4096, 16): (32, 64, 256, 4, 1),
    (8, 1024, 4096, 32): (32, 64, 256, 4, 1),
    (8, 1024, 4096, 64): (32, 64, 256, 4, 1),
    (8, 1024, 4096, 128): (64, 64, 256, 4, 1),
    (8, 1024, 4096, 256): (64, 64, 256, 4, 1),
    (8, 1024, 4096, 512): (128, 64, 256, 4, 1),
    (8, 1024, 4096, 1024): (128, 128, 128, 4, 1),
    (8, 1024, 4096, 4096): (128, 128, 256, 4, 1),
    (8, 1024, 4096, 8192): (128, 128, 256, 4, 1),
    (8, 1024, 4096, 16384): (128, 128, 256, 4, 1),
    # H=4 D=1024 R=4096 (TP=4)
    (4, 1024, 4096, 1): (32, 64, 512, 4, 4),
    (4, 1024, 4096, 2): (32, 64, 512, 0, 4),
    (4, 1024, 4096, 4): (32, 64, 512, 0, 1),
    (4, 1024, 4096, 8): (32, 64, 512, 0, 1),
    (4, 1024, 4096, 16): (32, 64, 512, 0, 1),
    (4, 1024, 4096, 32): (32, 64, 512, 4, 1),
    (4, 1024, 4096, 64): (32, 64, 512, 4, 1),
    (4, 1024, 4096, 128): (32, 64, 512, 4, 1),
    (4, 1024, 4096, 256): (32, 64, 256, 4, 1),
    (4, 1024, 4096, 512): (64, 64, 256, 4, 1),
    (4, 1024, 4096, 1024): (128, 64, 256, 4, 1),
    (4, 1024, 4096, 4096): (128, 128, 256, 4, 1),
    (4, 1024, 4096, 8192): (128, 128, 256, 4, 1),
    (4, 1024, 4096, 16384): (128, 128, 256, 4, 1),
    # H=2 D=1024 R=4096 (TP=8)
    (2, 1024, 4096, 1): (64, 64, 256, 4, 8),
    (2, 1024, 4096, 2): (32, 64, 512, 0, 1),
    (2, 1024, 4096, 4): (64, 64, 512, 0, 4),
    (2, 1024, 4096, 8): (32, 64, 512, 4, 4),
    (2, 1024, 4096, 16): (32, 64, 512, 0, 1),
    (2, 1024, 4096, 32): (32, 64, 512, 0, 1),
    (2, 1024, 4096, 64): (32, 64, 512, 0, 1),
    (2, 1024, 4096, 128): (32, 64, 512, 4, 1),
    (2, 1024, 4096, 256): (32, 64, 512, 4, 1),
    (2, 1024, 4096, 512): (32, 64, 512, 4, 1),
    (2, 1024, 4096, 1024): (64, 128, 256, 4, 1),
    (2, 1024, 4096, 4096): (128, 128, 256, 0, 1),
    (2, 1024, 4096, 8192): (128, 128, 256, 4, 1),
    (2, 1024, 4096, 16384): (128, 128, 256, 8, 1),
    # H=8 D=8192 R=8192
    (8, 8192, 8192, 128): (128, 128, 256, 0, 1),
    (8, 8192, 8192, 512): (128, 128, 256, 0, 1),
    (8, 8192, 8192, 2048): (128, 128, 256, 0, 1),
}

_AUTOTUNE_WINNERS_QZ = {
    # (H,  D,    R,    B    ) -> (tm, tn, tk, bsw)   FP8 (qz) output path.
    # Retuned 2026-06-08 by rocprof COLD-CACHE kernel time (autotune_ready.py),
    # after the qz2 path dropped its host contiguous()+eltwise+D2D overhead.
    # QZ: tile_n in {128,256}, split_k=1 (per-WG amax cannot reduce across
    # split-K). bsw may be nonzero (4th field); split_k defaults to 1 in lookup.
    # H=16 D=1024 R=4096 (TP=1)
    # H=16 D=1024 R=4096 (TP=1)
    (16, 1024, 4096, 1): (32, 128, 256, 0),
    (16, 1024, 4096, 2): (32, 128, 256, 0),
    (16, 1024, 4096, 4): (32, 128, 256, 0),
    (16, 1024, 4096, 8): (32, 128, 256, 0),
    (16, 1024, 4096, 16): (32, 128, 256, 0),
    (16, 1024, 4096, 32): (32, 128, 256, 0),
    (16, 1024, 4096, 64): (32, 128, 256, 0),
    (16, 1024, 4096, 128): (32, 128, 256, 2),
    (16, 1024, 4096, 256): (64, 256, 512, 2),
    (16, 1024, 4096, 512): (128, 256, 256, 2),
    (16, 1024, 4096, 1024): (256, 256, 128, 4),
    (16, 1024, 4096, 4096): (256, 256, 128, 4),
    (16, 1024, 4096, 8192): (64, 256, 128, 2),
    (16, 1024, 4096, 16384): (64, 256, 128, 2),
    (16, 1024, 4096, 32768): (64, 256, 128, 2),
    # H=8 D=1024 R=4096 (TP=2)
    (8, 1024, 4096, 1): (32, 128, 256, 0),
    (8, 1024, 4096, 2): (32, 128, 256, 0),
    (8, 1024, 4096, 4): (32, 128, 256, 0),
    (8, 1024, 4096, 8): (32, 128, 256, 0),
    (8, 1024, 4096, 16): (32, 128, 256, 0),
    (8, 1024, 4096, 32): (32, 128, 256, 0),
    (8, 1024, 4096, 64): (32, 128, 256, 0),
    (8, 1024, 4096, 128): (32, 128, 256, 2),
    (8, 1024, 4096, 256): (32, 128, 256, 2),
    (8, 1024, 4096, 512): (64, 128, 256, 2),
    (8, 1024, 4096, 1024): (128, 256, 256, 2),
    (8, 1024, 4096, 4096): (256, 256, 128, 4),
    (8, 1024, 4096, 8192): (64, 256, 128, 4),
    (8, 1024, 4096, 16384): (64, 256, 128, 2),
    # H=4 D=1024 R=4096 (TP=4)
    (4, 1024, 4096, 1): (32, 128, 256, 0),
    (4, 1024, 4096, 2): (32, 128, 256, 0),
    (4, 1024, 4096, 4): (32, 128, 256, 0),
    (4, 1024, 4096, 8): (32, 128, 256, 8),
    (4, 1024, 4096, 16): (32, 128, 256, 0),
    (4, 1024, 4096, 32): (32, 128, 256, 0),
    (4, 1024, 4096, 64): (32, 128, 256, 8),
    (4, 1024, 4096, 128): (32, 128, 256, 2),
    (4, 1024, 4096, 256): (32, 128, 256, 2),
    (4, 1024, 4096, 512): (32, 128, 256, 2),
    (4, 1024, 4096, 1024): (64, 256, 256, 0),
    (4, 1024, 4096, 4096): (64, 256, 128, 2),
    (4, 1024, 4096, 8192): (64, 256, 128, 2),
    (4, 1024, 4096, 16384): (128, 256, 256, 4),
    # H=2 D=1024 R=4096 (TP=8)
    (2, 1024, 4096, 1): (32, 128, 256, 8),
    (2, 1024, 4096, 2): (32, 128, 256, 0),
    (2, 1024, 4096, 4): (32, 128, 256, 0),
    (2, 1024, 4096, 8): (32, 128, 256, 0),
    (2, 1024, 4096, 16): (32, 128, 256, 0),
    (2, 1024, 4096, 32): (32, 128, 256, 0),
    (2, 1024, 4096, 64): (32, 128, 256, 0),
    (2, 1024, 4096, 128): (32, 128, 256, 0),
    (2, 1024, 4096, 256): (32, 128, 256, 0),
    (2, 1024, 4096, 512): (32, 128, 256, 4),
    (2, 1024, 4096, 1024): (32, 128, 256, 4),
    (2, 1024, 4096, 4096): (64, 256, 128, 0),
    (2, 1024, 4096, 8192): (64, 256, 128, 2),
    (2, 1024, 4096, 16384): (128, 128, 256, 8),
    # H=8 D=8192 R=8192
    (8, 8192, 8192, 128): (128, 256, 256, 2),
    (8, 8192, 8192, 512): (128, 128, 128, 0),
    (8, 8192, 8192, 2048): (256, 256, 128, 2),
}


def _autotune_lookup(table, H, D, R, B):
    """Find the best (tm, tn, tk, bsw) for the given (H, D, R, B).

    Resolution order:
      1. Exact (H, D, R, B) match.
      2. If (H, D, R) has any tuned B, pick the closest tabulated B
         (preferring next-higher; ties broken by absolute distance).
      3. Otherwise raise ValueError listing the tuned shapes.
    """
    if B is not None and (H, D, R, B) in table:
        return table[(H, D, R, B)], "exact"

    same_shape_bs = sorted(b for (h, d, r, b) in table if (h, d, r) == (H, D, R))
    if not same_shape_bs:
        tuned_shapes = sorted({(h, d, r) for (h, d, r, _) in table})
        raise ValueError(
            f"No autotune winner for (H={H}, D={D}, R={R}). "
            f"Tuned shapes: {tuned_shapes}. "
            f"Either tune this shape and add it to the table, "
            f"or call the non-auto factory directly with explicit tiles."
        )

    if B is None:
        # Caller didn't specify B — pick the largest tuned B for this shape
        # (assumes large-B is the dominant use case, which is also where
        # tile choice matters most).
        chosen_b = same_shape_bs[-1]
    else:
        # Prefer the smallest tabulated B that is >= the requested B
        # (better to use a tile tuned for a larger workload than a smaller
        # one — large tiles amortize better when over-tiled). If none,
        # take the closest tabulated B (always <= the requested B at this
        # point).
        higher = [b for b in same_shape_bs if b >= B]
        chosen_b = higher[0] if higher else same_shape_bs[-1]

    note = f"nearest-B={chosen_b}" if chosen_b != B else "exact"
    return table[(H, D, R, chosen_b)], note


def _unpack_winner(entry):
    """Normalize a winners-table value to (tm, tn, tk, bsw, split_k).

    Backward-compatible: 4-tuples (tm,tn,tk,bsw) default split_k=1; 5-tuples
    carry the tuned split_k; legacy 6-tuples (tm,tn,tk,bsw,sk,ns) drop the
    obsolete num_stages field.
    """
    if len(entry) == 6:
        tm, tn, tk, bsw, sk, _ns = entry
        return (tm, tn, tk, bsw, sk)
    if len(entry) == 5:
        return entry
    if len(entry) == 4:
        tm, tn, tk, bsw = entry
        return (tm, tn, tk, bsw, 1)
    raise ValueError(
        f"autotune winner entry must be a 4-, 5-, or 6-tuple, got {entry!r}"
    )


def compile_fp8_einsum_clean_ue8m0_auto(
    *,
    H: int,
    D: int,
    R: int,
    B: int | None = None,
    quant_output: bool = False,
    quant_transpose_scale: bool = False,
):
    """Autotune-dispatch entry point for the bf16 / qz fp8 einsum kernel.

    Looks up the best per-shape tile in `_AUTOTUNE_WINNERS_{BF16,QZ}` for
    the given (H, D, R, B) and calls the underlying factory with those
    tiles. Returns the same launcher object as the non-auto factory.

    Args:
      H, D, R: einsum compile-time shape.
      B: runtime batch size hint (used for tile selection only — the
        kernel itself still consumes B at launch time as i32_b). Pass
        `None` if B is genuinely unknown at compile time; the lookup
        will pick the tile tuned for the largest tabulated B in this
        shape (best for the compute-bound regime, may be suboptimal
        for small B).
      quant_output: when True, dispatch to the qz (D-128 group fp8
        quant epilogue) variant. The autotune table for qz is separate
        from bf16 (different winners at some B).
      quant_transpose_scale: forwarded to qz factory; see its docstring.

    Raises:
      ValueError: if (H, D, R) is not in the autotune table at all.

    Example:
      # Default bf16 output, B=1024 known
      kernel = compile_fp8_einsum_clean_ue8m0_auto(H=16, D=1024, R=4096, B=1024)
      # qz output, B unknown at compile time
      kernel = compile_fp8_einsum_clean_ue8m0_auto(
          H=16, D=1024, R=4096, quant_output=True, quant_transpose_scale=True)
    """
    table = _AUTOTUNE_WINNERS_QZ if quant_output else _AUTOTUNE_WINNERS_BF16
    entry, _ = _autotune_lookup(table, H, D, R, B)
    tm, tn, tk, bsw, sk = _unpack_winner(entry)
    # split_k applies to the bf16 path only (qz is split_k=1; its in-kernel
    # amax epilogue can't span split-K partitions — the QZ split-K win is
    # handled separately by the 2-pass qz_splitk path).
    if quant_output:
        sk = 1
    return compile_fp8_einsum_clean_ue8m0(
        H=H,
        D=D,
        R=R,
        tile_m=tm,
        tile_n=tn,
        tile_k=tk,
        block_swizzle_n=bsw,
        quant_output=quant_output,
        quant_transpose_scale=quant_transpose_scale,
        split_k=sk,
    )


# QZ method selection: shapes where the 2-pass QZ (qz2 = bf16 GEMM +
# per_group_quant, exactly TWO kernels) beats the single-pass in-kernel QZ
# (qz1) by rocprof kernel time. qz2 wins in the occupancy-starved regime (small
# B, low H); qz1 wins once the GEMM fills the GPU. Missing (H,D,R,B) -> use qz1
# (single-pass). Measured 2026-06-08 on MI355X (card 0) by qz_crossover.py,
# rocprof cold cache, with the qz2 path down to 2 kernels (no contiguous/
# eltwise/D2D intermediates).
#
# Value encoding:
#   _QZ_BF16_SK1        -> qz2 wins here, but at split_k==1, whose GEMM config
#                          IS the bf16 winner. We do NOT duplicate the tile —
#                          dispatch resolves it from _AUTOTUNE_WINNERS_BF16 at
#                          lookup time, so the qz2 split_k==1 GEMM tile stays
#                          auto-synced with the bf16 table on every re-tune.
#   (tm, tn, tk, sk>1)  -> a genuinely split-K config, tuned/stored explicitly
#                          (the only QZ-specific configs that need their own
#                          entry; split_k==1 always just reuses bf16).
#
# Dispatch rules: (1) qz2 must beat qz1 by >= 5% (sub-margin wins are within
# cold-cache noise / a weak qz1 tile, not a real 2-pass advantage); (2)
# monotonic-crossover guard — once qz1 wins at some B for an (H,D,R), every
# larger B uses qz1 too. Result: a clean contiguous small-B region. qz2 wins to
# B<=256 at H=2, B<=128 at H=4, and B<=2 at H=8; H=16 and large D,R -> qz1.
_QZ_BF16_SK1 = "bf16_sk1"  # marker: use the bf16 winner tile at split_k=1
_QZ_SPLITK_WINNERS: dict[tuple[int, int, int, int], tuple | str] = {
    # (H,    D,    R,    B ) -> (tm, tn, tk, split_k>1)  OR  _QZ_BF16_SK1.
    (2, 1024, 4096, 1): (64, 64, 256, 8),
    (2, 1024, 4096, 2): _QZ_BF16_SK1,
    (2, 1024, 4096, 4): (64, 64, 512, 4),
    (2, 1024, 4096, 8): (32, 64, 512, 4),
    (2, 1024, 4096, 16): _QZ_BF16_SK1,
    (2, 1024, 4096, 32): _QZ_BF16_SK1,
    (2, 1024, 4096, 64): _QZ_BF16_SK1,
    (2, 1024, 4096, 128): _QZ_BF16_SK1,
    (2, 1024, 4096, 256): _QZ_BF16_SK1,
    (4, 1024, 4096, 1): (32, 64, 512, 4),
    (4, 1024, 4096, 2): (32, 64, 512, 4),
    (4, 1024, 4096, 4): _QZ_BF16_SK1,
    (4, 1024, 4096, 8): _QZ_BF16_SK1,
    (4, 1024, 4096, 16): _QZ_BF16_SK1,
    (4, 1024, 4096, 32): _QZ_BF16_SK1,
    (4, 1024, 4096, 64): _QZ_BF16_SK1,
    (4, 1024, 4096, 128): _QZ_BF16_SK1,
    (8, 1024, 4096, 1): _QZ_BF16_SK1,
    (8, 1024, 4096, 2): _QZ_BF16_SK1,
}


def _qz_splitk_lookup(H, D, R, B):
    """Return the qz2 GEMM config (tm, tn, tk, split_k) for (H,D,R,B), or None
    to use qz1.

    Exact match only for B (the qz1/qz2 crossover is B-sensitive); for an
    untuned B we conservatively fall through to qz1 (single-pass, always valid).

    Marker resolution: a value of `_QZ_BF16_SK1` means "qz2 wins here, but with
    split_k==1, whose GEMM config is exactly the bf16 winner" — so we resolve
    the tile from `_AUTOTUNE_WINNERS_BF16` at lookup time instead of duplicating
    it. This keeps the qz2 split_k==1 GEMM tile auto-synced with the bf16 table
    (only the genuinely-split-K, split_k>1 configs are stored explicitly here).
    """
    val = _QZ_SPLITK_WINNERS.get((H, D, R, B))
    if val is None:
        return None
    if val is _QZ_BF16_SK1:
        # Reuse the bf16 winner's tile at split_k=1 (always valid, exact B).
        bf = _AUTOTUNE_WINNERS_BF16.get((H, D, R, B))
        if bf is None:
            return None  # no bf16 entry -> fall through to qz1 (safe)
        tm, tn, tk, _bsw, _sk = _unpack_winner(bf)
        return (tm, tn, tk, 1)
    return val


def compile_fp8_einsum_clean_ue8m0_qz_auto(
    *,
    H: int,
    D: int,
    R: int,
    B: int | None = None,
    transpose_scale: bool = False,
):
    """Autotune-dispatch QZ (fp8-output) variant.

    Picks per-shape between:
      - qz2 (2-pass split-K: split-K GEMM -> fp32/bf16 -> per_group_quant_hip)
        when (H,D,R,B) is in `_QZ_SPLITK_WINNERS` (occupancy-starved small-B).
      - qz1 (single-pass in-kernel quant) otherwise (the default, always valid).

    Returns a RETURNS-tensors launcher (no caller output buffers, no copies):
      launch(x, y, sx, sy, B, stream) -> (z_fp8, sz)
    qz2 returns per_group_quant_hip's tensors directly; qz1 allocates z/sz
    internally and returns them (the in-kernel quant writes them in place).
    """
    sk_cfg = _qz_splitk_lookup(H, D, R, B) if B is not None else None
    if sk_cfg is not None:
        # qz2 2-pass split-K — already returns (z_fp8, sz).
        tm, tn, tk, sk = sk_cfg
        return compile_fp8_einsum_clean_ue8m0_qz_splitk(
            H=H,
            D=D,
            R=R,
            tile_m=tm,
            tile_n=tn,
            tile_k=tk,
            split_k=sk,
            transpose_scale=transpose_scale,
        )
    # qz1 single-pass: wrap the buffer-ABI launcher to allocate + return.
    import torch as _torch

    _qz1 = compile_fp8_einsum_clean_ue8m0_auto(
        H=H,
        D=D,
        R=R,
        B=B,
        quant_output=True,
        quant_transpose_scale=transpose_scale,
    )

    def _qz1_return(arg_x, arg_y, arg_sx, arg_sy, i32_b, stream):
        Bb = int(i32_b)
        dev = arg_x.device
        z = _torch.empty((Bb, H, D), dtype=_torch.float8_e4m3fn, device=dev)
        if transpose_scale:
            # (H, D//128, B) — matches the HIP per_group_quant transpose layout
            # (the qz2 path), so qz1/qz2 return identical transposed scales.
            sz = _torch.empty((H, D // 128, Bb), dtype=_torch.float32, device=dev)
        else:
            sz = _torch.empty((Bb, H, D // 128), dtype=_torch.float32, device=dev)
        _qz1(z, sz, arg_x, arg_y, arg_sx, arg_sy, i32_b, stream)
        return z, sz

    return _qz1_return


# ──────────────────────────────────────────────────────────────────────────
# Unified user-facing interface
# ──────────────────────────────────────────────────────────────────────────
# Compiled-kernel cache keyed by (H, D, R, B, out_dtype, transpose_scale).
# Each unique config triggers flydsl JIT once; subsequent calls are dispatch-only.
_FP8_EINSUM_KERNEL_CACHE: dict = {}


def fp8_einsum(
    x_fp8: torch.Tensor,
    y_pre: torch.Tensor,
    sx: torch.Tensor,
    sy: torch.Tensor,
    *,
    out_dtype: torch.dtype = torch.bfloat16,
    transpose_scale: bool = False,
    stream: torch.cuda.Stream | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Compute Z[b, h, d] = einsum('bhr, hdr -> bhd', X, Y) with packed-UE8M0
    scales, dispatching to the autotune-selected kernel for this shape.

    This is the recommended user-facing entry point: it picks bf16 or fp8
    output based on ``out_dtype``, allocates the output tensor(s) internally
    using the same device as the inputs, looks up the autotune winner for
    (H, D, R, B), compiles the kernel on first call (cached), and launches.

    Args:
      x_fp8: fp8 e4m3 (B, H, R) — pre-quantized X (caller responsibility).
      y_pre: fp8 e4m3 preshuffled per head (output of
        ``aiter.ops.shuffle.shuffle_weight(y_fp8, layout=(16, 32))``).
      sx:    int32 (B, H, R // 512) — 4 packed UE8M0 bytes per i32 along K.
      sy:    int32 (H, D // 128, R // 512) — same packing along K.
      out_dtype: ``torch.bfloat16`` (default) → bf16 output, no scale.
                 ``torch.float8_e4m3fn`` → fp8 output + per-(B, H, D/128)
                                           fp32 scale.
      transpose_scale: only meaningful when ``out_dtype == torch.float8_e4m3fn``.
                       When True the scale tensor is laid out as
                       ``(H, D // 128, B)`` instead of the default
                       ``(B, H, D // 128)``. Memory is ``(H, D//128, B)``
                       contiguous, matching the HIP per_group_quant transpose
                       layout (fed a ``(B, H*D)`` view) so the qz1 and qz2 paths
                       return identical transposed scales.
      stream: CUDA stream; defaults to the current stream.

    Returns:
      ``(z, sz)`` tuple. For bf16 output, ``sz`` is ``None``. For fp8 output,
      ``sz`` is the fp32 scale tensor in the layout selected by
      ``transpose_scale``.

    Raises:
      ValueError: for unsupported ``out_dtype``, or if ``transpose_scale=True``
        is passed with a non-fp8 ``out_dtype``, or if the input shapes are
        inconsistent / not in the autotune table.

    Example:
      # bf16 output (default)
      z, _ = fp8_einsum(x_fp8, y_pre, sx, sy)

      # fp8 output with transposed-scale layout
      z_fp8, sz = fp8_einsum(
          x_fp8, y_pre, sx, sy,
          out_dtype=torch.float8_e4m3fn, transpose_scale=True,
      )
    """
    # ── Validate dtype combination ──
    is_qz = out_dtype == torch.float8_e4m3fn
    if not is_qz and out_dtype != torch.bfloat16:
        raise ValueError(
            f"fp8_einsum: out_dtype must be torch.bfloat16 or "
            f"torch.float8_e4m3fn, got {out_dtype}."
        )
    if transpose_scale and not is_qz:
        raise ValueError(
            "fp8_einsum: transpose_scale=True requires "
            "out_dtype=torch.float8_e4m3fn (it controls the SCALE tensor "
            "layout, which only exists in the fp8 output path)."
        )

    # ── Infer shape from inputs ──
    if x_fp8.ndim != 3:
        raise ValueError(
            f"fp8_einsum: x_fp8 must be (B, H, R), got shape {tuple(x_fp8.shape)}."
        )
    B, H, R = x_fp8.shape
    if sy.ndim != 3:
        raise ValueError(
            f"fp8_einsum: sy must be (H, D/128, R/512), got "
            f"shape {tuple(sy.shape)}."
        )
    sy_H, d128, r512 = sy.shape
    if sy_H != H:
        raise ValueError(
            f"fp8_einsum: sy.shape[0]={sy_H} must equal x_fp8.shape[1]={H}."
        )
    if r512 != R // 512:
        raise ValueError(
            f"fp8_einsum: sy.shape[2]={r512} must equal R // 512 = {R // 512}."
        )
    D = d128 * 128

    # ── Compile (cached) ──
    cache_key = (H, D, R, B, "fp8" if is_qz else "bf16", bool(transpose_scale))
    kernel = _FP8_EINSUM_KERNEL_CACHE.get(cache_key)
    if kernel is None:
        if is_qz:
            kernel = compile_fp8_einsum_clean_ue8m0_qz_auto(
                H=H,
                D=D,
                R=R,
                B=B,
                transpose_scale=transpose_scale,
            )
        else:
            kernel = compile_fp8_einsum_clean_ue8m0_auto(H=H, D=D, R=R, B=B)
        _FP8_EINSUM_KERNEL_CACHE[cache_key] = kernel

    # ── Launch ──
    if stream is None:
        stream = torch.cuda.current_stream()
    device = x_fp8.device
    if is_qz:
        # QZ dispatch returns (z_fp8, sz) directly (qz1 allocs internally; qz2
        # returns per_group_quant_hip's tensors) — no caller buffers, no copy.
        z, sz = kernel(x_fp8, y_pre, sx, sy, B, stream)
    else:
        z = torch.empty(B, H, D, dtype=torch.bfloat16, device=device)
        sz = None
        kernel(z, x_fp8, y_pre, sx, sy, B, stream)
    return z, sz


def compile_fp8_einsum_clean_ue8m0(
    *,
    H: int,
    D: int,
    R: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    block_swizzle_n: int = 0,
    quant_output: bool = False,
    quant_transpose_scale: bool = False,
    split_k: int = 1,
):
    """Compile DeepGEMM-style clean fp8 einsum (v2 pipeline).

    Args:
      H, D, R: einsum compile-time shape.
      tile_m/n/k: tile sizes. tile_k must be a multiple of 128. tile_n must be 128.
      block_swizzle_n: L2 supergroup swizzle (0 disables).
      quant_output: when True, swaps the bf16 output path for an in-kernel
        D-128 group fp8 quant epilogue. Adds one extra `arg_sz`
        (fp32 (B, H, D//128)) between `arg_z` and `arg_x`. Default False
        preserves the original bf16 ABI bit-exact. The convenience wrapper
        `compile_fp8_einsum_clean_ue8m0_qz()` simply calls this factory
        with `quant_output=True`.
      quant_transpose_scale: when True with `quant_output=True`, the scale
        tensor `arg_sz` is laid out as `(H, D // 128, B)` fp32 rather than the
        default `(B, H, D//128)`. Memory is `(H, D//128, B)` contiguous. This
        matches the HIP per_group_quant transpose layout (fed a `(B, H*D)`
        view), so the qz1 (in-kernel) and qz2 (2-pass split-K) paths return the
        SAME transposed scale for any shape. Requires `quant_output=True`.

    Returns: launcher with signature
      bf16 mode: launch(z_bf16, x_fp8, y_pre, sx_i32, sy_i32, B, stream)
      qz   mode: launch(z_fp8, sz_fp32, x_fp8, y_pre, sx_i32, sy_i32, B, stream)

    Note: lds_stage param dropped — only ping/pong is supported in v2.
    B is gmem->reg ("reg-B"); A/sx staged to LDS via raw_ptr_buffer_load_lds;
    sy gmem->reg per K-tile. This is the proven-correct, peak path.
    """
    if quant_output:
        # Cross-WG amax not supported in v1: each WG owns 1 or 2 D-128 blocks.
        if tile_n not in (128, 256):
            raise ValueError(
                f"quant_output=True requires tile_n in {{128, 256}} "
                f"(per-WG D-128 reduce). Got tile_n={tile_n}."
            )
    if quant_transpose_scale and not quant_output:
        raise ValueError("quant_transpose_scale=True requires quant_output=True")
    if tile_k % 128 != 0:
        raise ValueError(f"tile_k must be a multiple of 128, got {tile_k}")
    if tile_m < 16 or (tile_m % 16) != 0:
        raise ValueError(f"tile_m must be positive multiple of 16, got {tile_m}")
    # tile_n granularity: full N-128-block (mult of 128) OR 64 (shares a
    # 128-block scale with its sibling 64-tile). The 4-wave×16-N MFMA split
    # needs tile_n >= 64 (n_per_wave = tile_n/4 >= 16). qz requires {128,256}.
    if tile_n == 64:
        if quant_output:
            raise ValueError("tile_n=64 not supported for quant_output (qz).")
    elif tile_n % 128 != 0:
        raise ValueError(
            f"tile_n must be 64 or a multiple of 128 (N-128-block / shared "
            f"scale granularity). Got tile_n={tile_n}."
        )
    if tile_n < 64:
        raise ValueError(f"tile_n must be >= 64, got {tile_n}")
    if R % tile_k != 0:
        raise ValueError(f"R={R} must be divisible by tile_k={tile_k}")
    if D % tile_n != 0:
        raise ValueError(f"D={D} must be divisible by tile_n={tile_n}")
    # ── split-K validation ──────────────────────────────────────
    # split_k partitions the R/tile_k K-tile loop across gridZ; each CTA
    # computes a partial sum over its K-slice and atomicAdds into an fp32
    # workspace, which a tiny cast pass then writes to bf16 Z. split_k=1 is
    # the original single-CTA-walks-all-K path (bit-exact, zero overhead).
    _num_k_tiles = R // tile_k
    if split_k < 1:
        raise ValueError(f"split_k must be >= 1, got {split_k}")
    # Uneven split-K: K is partitioned in 512-element blocks distributed evenly
    # across split_k CTAs (need NOT divide the tile count). Requirements:
    #   - at least one 512-block per CTA: split_k <= R//512.
    #   - a 512-block must be a whole number of tiles: (tile_k//128) | 4
    #     (i.e. tile_k <= 512). This keeps slices 512-aligned so the MFMA opsel
    #     byte stays compile-time constant.
    if split_k > 1:
        _n512_chk = R // 512
        if split_k > _n512_chk:
            raise ValueError(
                f"split_k={split_k} exceeds R//512={_n512_chk} (cannot give "
                f"every CTA a 512-block). R={R}."
            )
        if 4 % (tile_k // 128) != 0:
            raise ValueError(
                f"split_k requires tile_k <= 512 ((tile_k//128) must divide 4); "
                f"got tile_k={tile_k}."
            )
        # UNEVEN split-K (split_k does not divide R//512) is UNSUPPORTED: the
        # phantom-guard path that handles ragged slices has a pipeline race at
        # small (1-512-block) slices that intermittently drops partials. Even
        # split_k (a divisor of R//512) is the only validated/correct path, and
        # every autotuned winner uses an even split_k (2/4/8). Reject uneven to
        # guarantee no silent wrong results.
        if _n512_chk % split_k != 0:
            raise ValueError(
                f"uneven split_k unsupported: split_k={split_k} must divide "
                f"R//512={_n512_chk} (use an even divisor, e.g. "
                f"{[d for d in range(2, _n512_chk + 1) if _n512_chk % d == 0]})."
            )
    if split_k > 1 and quant_output:
        raise ValueError(
            "split_k>1 is only implemented for the bf16-output path "
            "(quant_output=False) in v1."
        )
    # ── pipeline depth ──────────────────────────────────────────
    # Fixed 2-stage prefetch:
    #   split_k==1 -> hand-unrolled ping/pong driver.
    #   split_k>1  -> 2-stage rotating-buffer driver with the uneven-split-K
    #                 phantom-guard. (The old deep N-stage `num_stages>2`
    #                 capability was removed — autotune never picked it.)
    # The smallest CTA must walk >= 2 K-tiles to prefetch. With uneven split-K
    # the smallest slice has floor(R//512 / split_k) 512-blocks.
    _tiles_per_512block_v = 4 // (tile_k // 128)
    if split_k > 1:
        _min_512_v = (R // 512) // split_k  # floor = smallest CTA's 512-blocks
        _min_tiles_per_cta = _min_512_v * _tiles_per_512block_v
    else:
        _min_tiles_per_cta = _num_k_tiles
    if _min_tiles_per_cta < 2:
        raise ValueError(
            f"smallest per-CTA K-tile count={_min_tiles_per_cta} < 2 "
            f"(uneven split_k={split_k}); the 2-stage prefetch needs >=2. "
            f"Reduce split_k."
        )
    if R % 512 != 0:
        raise ValueError(
            f"R={R} must be divisible by 512 (packed UE8M0 needs 4 K-128 "
            f"blocks per i32)."
        )
    gpu_arch = get_hip_arch()
    if not str(gpu_arch).startswith("gfx95"):
        raise RuntimeError(f"clean ue8m0 kernel targets gfx950 only, got {gpu_arch}")

    elem_bytes_a = 1  # fp8
    elem_bytes_b = 1  # fp8

    _qz_suffix = "_qz" if quant_output else ""
    if quant_transpose_scale:
        _qz_suffix += "_tsz"
    KERNEL_NAME = (
        f"fp8_einsum_clean_ue8m0{_qz_suffix}"
        f"_H{H}_D{D}_R{R}"
        f"_t{tile_m}x{tile_n}x{tile_k}"
    )
    if block_swizzle_n > 0:
        KERNEL_NAME += f"_bsw{block_swizzle_n}"
        _gy_static = D // tile_n
        if _gy_static % block_swizzle_n != 0:
            raise ValueError(
                f"block_swizzle_n={block_swizzle_n} must divide gy="
                f"D/tile_n={_gy_static}."
            )
    # Encode split_k so distinct configs get distinct kernel symbols (needed
    # to attribute rocprof per-config kernel durations).
    if split_k > 1:
        KERNEL_NAME += f"_sk{split_k}"

    # Threading
    total_threads = 256
    wave_size = 64
    num_waves = total_threads // wave_size  # 4

    # A bytes
    tile_k_bytes_a = tile_k * elem_bytes_a
    bytes_a_per_tile = tile_m * tile_k_bytes_a
    if bytes_a_per_tile % total_threads != 0:
        raise ValueError(
            f"tile_m*tile_k must be divisible by {total_threads}: "
            f"tile_m={tile_m}, tile_k={tile_k}"
        )
    bytes_per_thread_a = bytes_a_per_tile // total_threads
    # A async-DMA chunk width per lane. raw_ptr_buffer_load_lds only legalizes
    # 16 B (dwordx4) and 4 B (dword) on this backend (the 8 B / s64 form fails
    # "Do not know how to expand"). So prefer 16 B for peak DMA efficiency, and
    # fall back to 4 B chunks (more, smaller DMAs) when bytes_per_thread_a is
    # not a multiple of 16 — e.g. tile_m=16, tile_k=128 -> 8 B/thread = 2x4 B.
    # This unlocks the smallest tile_m for tiny-B shapes.
    if bytes_per_thread_a % 16 == 0:
        a_load_bytes = 16
    elif bytes_per_thread_a % 4 == 0:
        a_load_bytes = 4
    else:
        raise ValueError(
            f"bytes_per_thread_a={bytes_per_thread_a} must be a multiple of "
            f"4 (tile_m*tile_k/{total_threads}; got tile_m={tile_m}, "
            f"tile_k={tile_k})."
        )
    num_a_loads = bytes_per_thread_a // a_load_bytes

    # B bytes (preshuffled fp8). tile_k_bytes_b also sets the A-LDS stride below.
    tile_k_bytes_b = tile_k * elem_bytes_b
    bytes_b_per_tile = tile_n * tile_k_bytes_b
    bytes_per_thread_b = bytes_b_per_tile // total_threads
    b_load_bytes = 16
    # Per-thread B vmem load count. The K-loop's pre-compute barrier keeps the
    # next tile's (num_a_loads + num_b_loads) A+B loads in flight rather than
    # fully draining HBM. See _bar_keep in the driver.
    num_b_loads = bytes_per_thread_b // b_load_bytes

    # LDS allocators. Two physical allocators (smem0/smem1) for bank
    # separation; the 2 stages map to ping(smem1)/pong(smem0).
    allocator_pong = SmemAllocator(None, arch=gpu_arch, global_sym_name="smem0")
    allocator_ping = SmemAllocator(None, arch=gpu_arch, global_sym_name="smem1")
    lds_stage = 2

    def _alloc_per_stage(byte_size, stages):
        offsets, allocs = [], []
        for s in range(stages):
            alloc = allocator_pong if (s % 2) == 0 else allocator_ping
            off = alloc._align(alloc.ptr, 16)
            alloc.ptr = off + byte_size
            offsets.append(off)
            allocs.append(alloc)
        return offsets, allocs

    # A fp8 LDS — ping/pong per stage.
    lds_a_fp8_stride_bytes = tile_k_bytes_b
    lds_a_fp8_bytes_per_stage = tile_m * lds_a_fp8_stride_bytes
    lds_a_fp8_alloc_offsets, lds_a_fp8_allocs = _alloc_per_stage(
        lds_a_fp8_bytes_per_stage, lds_stage
    )
    lds_a_fp8_k_blocks16 = lds_a_fp8_stride_bytes // 16

    # B fp8 is kept in registers (gmem->regs via load_b_tile) — "reg-B". This is
    # the proven-correct, peak path; no B LDS slab.

    # sy (Y scales) are NOT staged in LDS — loaded gmem->reg per K-tile in
    # prefetch_sy_tile (tiny). gmem layout (H, D//128, R//512) i32; see
    # _sy_per_n128 / _sy_per_head below.

    # sx LDS slab — persistent per WG, loaded once in prologue.
    # sx is `(B, H, R//512)` i32; per WG we need rows bx_m..bx_m+tile_m-1 of
    # the per-token-per-K-128 packed scale. Slab layout (row-major within WG):
    #   slot(row_in_tile, k_packed_idx) = row_in_tile * (R/512) + k_packed_idx
    # Bytes: tile_m * (R/512) * 4. At tm=128 R=8192: 128*16*4 = 8 KB per WG.
    _sx_lds_per_row = R // 512  # i32 entries per row
    _sx_lds_count = tile_m * _sx_lds_per_row  # total i32 entries
    _sx_lds_bytes = _sx_lds_count * 4
    # We distribute the load across all 256 threads; require perfect
    # divisibility for a clean coalesced load with no bounds check.
    if _sx_lds_count % total_threads != 0:
        raise ValueError(
            f"sx LDS load distribution requires tile_m * (R/512) divisible "
            f"by {total_threads}, but tile_m={tile_m} × R/512={R//512} = "
            f"{_sx_lds_count}, remainder {_sx_lds_count % total_threads}. "
            f"Increase R or tile_m so the product is a multiple of {total_threads}."
        )
    sx_lds_offset = allocator_pong._align(allocator_pong.ptr, 16)
    allocator_pong.ptr = sx_lds_offset + max(16, _sx_lds_bytes)

    # ── qz-mode cross-wave amax LDS slab ─────────────────────────────────
    # Slot layout: [row_in_tile (tile_m)][d128 (tile_n//128)][wave_per_d128]
    # where row_in_tile = mi*16 + lane_div_16*4 + ii (the per-lane row).
    #
    # row_in_tile is the unique row key — lanes computing different rows
    # MUST write to different slots, otherwise lanes holding OOB rows
    # race-write 0 against lanes holding valid rows (visible when
    # B < tile_m: only rows with lane_div_16==0 are valid, but lanes
    # with lane_div_16>0 still write to the same slot if the key
    # omits lane_div_16).
    #
    # waves_per_d128: tile_n==128 → all 4 waves share the single D-128 block;
    #                 tile_n==256 → 2 adjacent waves share each of 2 blocks.
    #
    # Per-WG entries: tile_m * (tile_n//128) * waves_per_d128.
    # At tm=128 tn=128: 128*1*4 = 512 fp32 = 2 KB. At tm=128 tn=256: 2 KB.
    # At tm=64  tn=128: 64*1*4  = 256 fp32 = 1 KB.
    if quant_output:
        _qz_d128_per_tile = tile_n // 128
        _qz_waves_per_d128 = 4 if tile_n == 128 else 2
        _qz_amax_count = tile_m * _qz_d128_per_tile * _qz_waves_per_d128
        _qz_amax_bytes = _qz_amax_count * 4
        qz_amax_lds_offset = allocator_pong._align(allocator_pong.ptr, 16)
        allocator_pong.ptr = qz_amax_lds_offset + max(16, _qz_amax_bytes)
    else:
        _qz_d128_per_tile = 0
        _qz_waves_per_d128 = 0
        _qz_amax_count = 0
        qz_amax_lds_offset = 0

    # sy (Y scales) are loaded gmem->reg per K-tile (prefetch_sy_tile); no LDS
    # slab. sx (X scales) are staged once to an LDS slab in the prologue.

    # CDNA4 / gfx950 (MI355X): 160 KB LDS per CU.
    # (CDNA3 / MI300X was 64 KB; the older value here was a stale copy.)
    LDS_PER_CU_BYTES = 163840
    _total_lds_bytes = allocator_pong.ptr + allocator_ping.ptr
    if _total_lds_bytes > LDS_PER_CU_BYTES:
        raise ValueError(
            f"LDS budget overflow: {_total_lds_bytes} bytes per workgroup, "
            f"limit is {LDS_PER_CU_BYTES} (gfx950 = 160 KB/CU). Try smaller tile."
        )

    # MFMA layout numbers
    m_repeat = tile_m // 16
    n_per_wave = tile_n // num_waves
    num_acc_n = n_per_wave // 16
    k_unroll = tile_k_bytes_a // elem_bytes_a // 32

    # B preshuffle layout (per-head)
    k_bytes_b_per_head = R * elem_bytes_b
    n0_val = D // 16
    k0_val = k_bytes_b_per_head // 64
    kpack_elems = 16
    _stride_nlane = kpack_elems
    _stride_klane = 16 * _stride_nlane
    _stride_k0 = 4 * _stride_klane
    _stride_n0 = k0_val * _stride_k0
    b_elems_per_head = n0_val * _stride_n0

    # K=128 quant group constants
    k_per_quant_group = 128
    mfmas_per_group = k_per_quant_group // 32  # = 4
    groups_per_tile = tile_k // k_per_quant_group

    # sx packed layout: (B, H, R // 512) i32
    _sx_per_head = (R // 128) // 4
    # sy packed layout: (H, D // 128, R // 512) i32
    _sy_per_n128 = R // 128 // 4
    _sy_per_head = (D // 128) * _sy_per_n128

    # ────────────────────────────────────────────────────────────────────────
    # The kernel body is identical between bf16 and qz modes except for:
    #   - the signature (qz adds arg_sz between arg_z and arg_x)
    #   - the _z_nrec computation (qz: fp8 byte; bf16: bf16 byte)
    #   - the store_output epilogue (qz: amax+pack_fp8; bf16: bf16 cast+store)
    # We hoist the shared body into `_kernel_body(arg_z, arg_sz, arg_x, ...)`.
    # In bf16 mode `arg_sz` is None and the qz epilogue is skipped.
    def _kernel_body(
        arg_z,
        arg_sz,
        arg_x,
        arg_y,
        arg_sx,
        arg_sy,
        i32_b,
        arg_signal=None,
        arg_semaphore=None,
    ):
        Vec = fx.Vector
        fp8_dtype = fx.Float8E4M3FN

        c_b = fx.Index(i32_b)

        tx = gpu.thread_id("x")
        bx_raw = gpu.block_id("x")
        by_raw = gpu.block_id("y")

        # split-K: gridZ indexes the K-partition this CTA owns. K is partitioned
        # in units of 512-element blocks (= 4 K-128-groups = _tiles_per_512block
        # tiles), distributed as evenly as possible across split_k CTAs so split_k
        # need NOT divide the tile count (uneven slices: the first `rem` CTAs get
        # one extra 512-block). Every slice starts on a 512 boundary, so the MFMA
        # opsel byte (k_block_global % 4) stays compile-time constant (it depends
        # only on the relative tile index walked in the loop).
        #
        #   n512        = R // 512                      (total 512-blocks)
        #   base, rem   = divmod(n512, split_k)
        #   my512       = base + (1 if kz < rem else 0) (this CTA's 512-blocks)
        #   start512    = kz*base + min(kz, rem)        (this CTA's first block)
        #   tiles/512blk= 4 // (tile_k//128)            (must be integer)
        # kt0          = start512 * tiles_per_512block  (runtime, this CTA's base)
        # _max_tiles   = ceil(n512/split_k)*tiles_per_512block  (compile-time
        #                loop bound; CTAs with fewer tiles predicate the tail off)
        _n512 = R // 512
        _k128_per_tile = tile_k // 128
        if (4 % _k128_per_tile) != 0:
            # a 512-block must be a whole number of tiles for clean slicing
            raise ValueError(
                f"split_k requires tile_k <= 512 (tile_k//128 must divide 4); "
                f"got tile_k={tile_k}."
            )
        _tiles_per_512block = 4 // _k128_per_tile
        _base512, _rem512 = divmod(_n512, split_k)
        import math as _math

        _max_512 = _math.ceil(_n512 / split_k) if split_k > 1 else _n512
        _max_tiles = _max_512 * _tiles_per_512block
        if const_expr(split_k > 1):
            kz = gpu.block_id("z")
            # my512 = base + (kz < rem ? 1 : 0); start512 = kz*base + min(kz,rem)
            _extra = fx.arith.select(
                fx.arith.cmpi(
                    fx.arith.CmpIPredicate.slt,
                    fx.Int32(kz).ir_value(),
                    fx.Int32(_rem512).ir_value(),
                ),
                fx.Int32(1).ir_value(),
                fx.Int32(0).ir_value(),
            )
            _extra = fx.Index(fx.Int32(_extra))
            _min_kz_rem = fx.arith.select(
                fx.arith.cmpi(
                    fx.arith.CmpIPredicate.slt,
                    fx.Int32(kz).ir_value(),
                    fx.Int32(_rem512).ir_value(),
                ),
                fx.Int32(kz).ir_value(),
                fx.Int32(_rem512).ir_value(),
            )
            _min_kz_rem = fx.Index(fx.Int32(_min_kz_rem))
            _my512 = fx.Index(_base512) + _extra
            _start512 = kz * fx.Index(_base512) + _min_kz_rem
            kt0 = _start512 * fx.Index(_tiles_per_512block)
            _my_tiles = _my512 * fx.Index(_tiles_per_512block)  # runtime trip count
            # scale-slot offset in packed-i32 (R/512) units = start512 (1 i32 per
            # 512-block) — exact, no rounding (slice is 512-aligned).
            kt0_k128_packed = _start512
        else:
            kz = fx.Index(0)
            kt0 = fx.Index(0)
            kt0_k128_packed = fx.Index(0)
            _my_tiles = fx.Index(_max_tiles)

        gx_per_h = (c_b + (tile_m - 1)) // tile_m
        gx = gx_per_h * fx.Index(H)

        if const_expr(block_swizzle_n > 0):
            _c_m_eff = gx * fx.Index(tile_m)
            bx, by = xcd_remap_bx_by(
                bx_raw,
                by_raw,
                _c_m_eff,
                tile_m=tile_m,
                tile_n=tile_n,
                N=D,
                xcd_swizzle=block_swizzle_n,
            )
        else:
            bx = bx_raw
            by = by_raw

        bx_h = bx // gx_per_h
        bx_m_idx = bx % gx_per_h
        bx_m = bx_m_idx * tile_m
        by_n = by * tile_n

        # split-K: per-output-tile index into the signal/semaphore scratch.
        # Uses POST-swizzle (bx, by) — the true output-tile identity — which is
        # independent of kz, so all `split_k` K-slice CTAs of one tile agree on
        # the same index. gy = D // tile_n; total tiles = gx * gy.
        if const_expr(split_k > 1):
            _gy_tiles = fx.Index(D // tile_n)
            signal_idx = bx * _gy_tiles + by  # fx.Index, output-tile linear id

        # ── LDS pointers ───────────────────────────────────────────────
        base_ptr_pong = allocator_pong.get_base()
        base_ptr_ping = allocator_ping.get_base()
        lds_a_fp8_stages = []
        for _off, _alloc in zip(lds_a_fp8_alloc_offsets, lds_a_fp8_allocs):
            _base = base_ptr_pong if _alloc is allocator_pong else base_ptr_ping
            _ptr = SmemPtr(
                _base,
                _off,
                fp8_dtype.ir_type,
                shape=(lds_a_fp8_bytes_per_stage,),
            )
            lds_a_fp8_stages.append(_ptr.get())
        lds_a_pong = lds_a_fp8_stages[0]
        lds_a_ping = lds_a_fp8_stages[1]

        # B fp8 is carried in registers via load_b_tile (gmem→regs); no LDS.
        # sy (Y scales) are loaded gmem→reg per K-tile (prefetch_sy_tile); no LDS.

        # sx LDS slab — persistent, tile_m * R/512 i32 entries.
        _sx_lds_ptr = SmemPtr(
            base_ptr_pong,
            sx_lds_offset,
            fx.Int32.ir_type,
            shape=(max(4, _sx_lds_count),),
        )
        lds_sx = _sx_lds_ptr.get()

        # qz amax LDS slab.
        if quant_output:
            _qz_amax_lds_ptr = SmemPtr(
                base_ptr_pong,
                qz_amax_lds_offset,
                fx.Float32.ir_type,
                shape=(max(4, _qz_amax_count),),
            )
            lds_qz_amax = _qz_amax_lds_ptr.get()
        else:
            lds_qz_amax = None

        # ── Buffer resources ──────────────────────────────────────────
        _x_nrec = fx.Int64(c_b * (H * R))
        x_rsrc = buffer_ops.create_buffer_resource(
            arg_x, max_size=False, num_records_bytes=_x_nrec
        )
        y_rsrc = buffer_ops.create_buffer_resource(arg_y, max_size=True)
        _sx_nrec = fx.Int64(c_b * (H * (R // 512) * 4))
        sx_rsrc = buffer_ops.create_buffer_resource(
            arg_sx, max_size=False, num_records_bytes=_sx_nrec
        )
        sy_rsrc = buffer_ops.create_buffer_resource(arg_sy, max_size=True)
        # qz: fp8 byte output (1 B/elem). bf16 (incl. split-K): 2 B/elem. split-K
        # reduces DIRECTLY in bf16 into arg_z via global_atomic_pk_add_bf16 — the
        # first K-slice CTA device-zeros each output tile, then all K-slice CTAs
        # packed-bf16-atomic-add their partials. No fp32 workspace, no host copy.
        if quant_output:
            _z_bytes_per_elem = 1
        else:
            _z_bytes_per_elem = 2
        _z_nrec = fx.Int64(c_b * (H * D * _z_bytes_per_elem))
        z_rsrc = buffer_ops.create_buffer_resource(
            arg_z, max_size=False, num_records_bytes=_z_nrec
        )
        if quant_output:
            _sz_nrec = fx.Int64(c_b * (H * (D // 128) * 4))
            sz_rsrc = buffer_ops.create_buffer_resource(
                arg_sz, max_size=False, num_records_bytes=_sz_nrec
            )
        else:
            sz_rsrc = None

        # ── Wave/lane decomposition ──────────────────────────────────
        layout_wave_lane = fx.make_layout((num_waves, wave_size), (wave_size, 1))
        coord_wave_lane = fx.idx2crd(tx, layout_wave_lane)
        wave_id = fx.get(coord_wave_lane, 0)
        lane_id = fx.get(coord_wave_lane, 1)

        layout_lane16 = fx.make_layout((4, 16), (16, 1))
        coord_lane16 = fx.idx2crd(lane_id, layout_lane16)
        lane_div_16 = fx.get(coord_lane16, 0)
        lane_mod_16 = fx.get(coord_lane16, 1)

        # N-tile column indexing per wave (wave owns N-cols
        # [wave_id*n_per_wave, +n_per_wave)).
        n_tile_base = wave_id * n_per_wave
        n_blk_list = []
        n_intra_list = []
        for i in range_constexpr(num_acc_n):
            global_n_in_head = by_n + n_tile_base + (i * 16) + lane_mod_16
            n_blk_list.append(global_n_in_head // 16)
            n_intra_list.append(global_n_in_head % 16)

        _scale_base_packed = kt0_k128_packed

        _b_stride_n0_c = fx.Index(_stride_n0)
        _b_stride_k0_c = fx.Index(_stride_k0)
        _b_stride_klane_c = fx.Index(_stride_klane)
        _b_stride_nlane_c = fx.Index(_stride_nlane)

        y_head_byte_off = bx_h * fx.Index(b_elems_per_head)

        stride_b_x = H * R
        stride_b_z = H * D

        # ── A async copy: gmem → LDS direct (swizzle-the-source) ──────
        # Chunk width in dwords (4 for 16 B, 2 for 8 B). tx_i32_base spaces
        # each lane by chunk_i32 dwords so lanes tile contiguous dword runs.
        _a_chunk_i32 = a_load_bytes // 4
        tile_k_dwords = tile_k // 4
        c4 = fx.Index(4)
        tx_i32_base = tx * fx.Index(_a_chunk_i32)
        layout_a_tile_div4 = fx.make_layout((tile_m, tile_k_dwords), (tile_k_dwords, 1))

        def a_tile_chunk_coord(i):
            return tile_chunk_coord_i32(
                fx.arith,
                tx_i32_base=tx_i32_base,
                i=i,
                total_threads=total_threads,
                layout_tile_div4=layout_a_tile_div4,
                chunk_i32=_a_chunk_i32,
            )

        x_head_elem_off = bx_h * fx.Index(R)
        _lds_a_fp8_k_dim_c = fx.Index(lds_a_fp8_stride_bytes)
        _lds_a_fp8_k_blocks16_c = fx.Index(lds_a_fp8_k_blocks16)

        # async-A constants (adaptive: 16 B = dwordx4, or 8 B = dwordx2)
        _a_async_load_bytes = a_load_bytes
        _a_wave_bytes_per_chunk = wave_size * _a_async_load_bytes
        _a_chunk_stride_bytes = total_threads * _a_async_load_bytes

        def load_a_tile_to_lds_async(base_k_elem, lds_buffer):
            """Issue async DMA: A gmem → LDS using swizzle-the-source.

            Per lane, per chunk i:
              row, col_dword = a_tile_chunk_coord(i)
              col_byte = col_dword * 4
              col_swz_bytes = swizzle_xor16(row, col_byte, k_blocks16)
              global byte = row_global*(H*R) + bx_h*R + base_k + col_swz_bytes
              LDS dst = wave_base + i*total_threads*16   (HW adds lane*16)

            Caller must `s_waitcnt(lgkmcnt=0)` + barrier before reading back.
            """
            from flydsl._mlir.dialects import memref as memref_dialect

            wave_byte_off = rocdl.readfirstlane(
                fx.Int64.ir_type,
                fx.Int64(wave_id * fx.Index(_a_wave_bytes_per_chunk)),
            )
            lds_base = memref_dialect.extract_aligned_pointer_as_index(lds_buffer)
            lds_ptr_base = buffer_ops.create_llvm_ptr(
                fx.Int64(lds_base),
                address_space=3,
            )
            lds_ptr_wave = buffer_ops.get_element_ptr(
                lds_ptr_base,
                wave_byte_off,
            )

            for i in range_constexpr(num_a_loads):
                row_a_local, col_dword_local = a_tile_chunk_coord(i)
                col_byte_local = col_dword_local * c4
                col_byte_swz = swizzle_xor16(
                    row_a_local,
                    col_byte_local,
                    _lds_a_fp8_k_blocks16_c,
                )
                row_a_global = bx_m + row_a_local
                idx_elem = (
                    row_a_global * fx.Index(stride_b_x)
                    + x_head_elem_off
                    + (base_k_elem + col_byte_swz)
                )
                global_offset_i32 = fx.Int32(idx_elem)

                lds_dst = lds_ptr_wave
                if const_expr(i > 0):
                    lds_dst = buffer_ops.get_element_ptr(
                        lds_ptr_wave,
                        static_byte_offset=i * _a_chunk_stride_bytes,
                    )
                rocdl.raw_ptr_buffer_load_lds(
                    x_rsrc,
                    lds_dst,
                    fx.Int32(_a_async_load_bytes),
                    global_offset_i32,
                    fx.Int32(0),
                    fx.Int32(0),
                    fx.Int32(1),
                )

        # ── B global load (preshuffled, per-head) — gmem→regs path ─────
        def load_b_tile(base_k_elem):
            """Load (tile_n, tile_k) fp8 B for current head & N-tile.
            Returns packs_flat[ku=0..k_unroll-1] of per-ni i64 lists.
            """
            k0_base = base_k_elem // 64
            packs_flat = [[] for _ in range(k_unroll)]
            for ni in range_constexpr(num_acc_n):
                n_base_byte = (
                    n_blk_list[ni] * _b_stride_n0_c
                    + lane_div_16 * _b_stride_klane_c
                    + n_intra_list[ni] * _b_stride_nlane_c
                ) + y_head_byte_off
                for ku64 in range_constexpr(k_unroll // 2):
                    k0 = k0_base + ku64
                    idx_byte = n_base_byte + k0 * _b_stride_k0_c
                    idx_dword = idx_byte // 4
                    b_vec4 = buffer_ops.buffer_load(
                        y_rsrc,
                        fx.Int32(idx_dword),
                        vec_width=4,
                        dtype=fx.Int32,
                    )
                    b16 = Vec(b_vec4).bitcast(fp8_dtype)
                    b_i64x2 = Vec(b16).bitcast(fx.Int64)
                    packs_flat[ku64 * 2].append(b_i64x2[0].ir_value())
                    packs_flat[ku64 * 2 + 1].append(b_i64x2[1].ir_value())
            return packs_flat

        # ── LDS A read primitive (16 fp8 bytes → 2 i64) ───────────
        def lds_load_a_packs_k64(row, col_base_bytes, lds_buffer):
            col_swz = swizzle_xor16(row, col_base_bytes, _lds_a_fp8_k_blocks16_c)
            idx = row * _lds_a_fp8_k_dim_c + col_swz
            v16 = Vec.load(Vec.make_type(16, fp8_dtype), lds_buffer, [idx])
            v2 = Vec(v16).bitcast(fx.Int64)
            return v2[0].ir_value(), v2[1].ir_value()

        # ── qz: in-wave 16-lane fmax butterfly (used by quant epilogue) ──
        # Implementation copy of fp8_einsum.intra_16_max_f32 — same routine.
        _i32_ir = fx.Int32.ir_type
        _f32_ir = fx.Float32.ir_type

        def intra_16_max_f32(local_f32, lane_id_in_wave):
            """Max-reduce across the 16 lanes that share lane_id // 16
            within the current wave. `lane_id_in_wave` is the per-lane i32
            value of (tx % 64). Result: all 16 lanes in the row-fragment
            hold the row-fragment fmax.
            """
            from flydsl._mlir.dialects import rocdl as _rocdl_low

            val_i32 = fx.arith.bitcast(_i32_ir, local_f32.ir_value())
            for xor_n in (1, 2, 4, 8):
                src_lane = lane_id_in_wave ^ fx.Int32(xor_n)
                src_byte = src_lane * fx.Int32(4)
                gather = _rocdl_low.ds_bpermute(
                    res=_i32_ir,
                    index=src_byte.ir_value(),
                    src=val_i32,
                )
                gather_f = fx.arith.bitcast(_f32_ir, gather)
                cur_f = fx.arith.bitcast(_f32_ir, val_i32)
                mx = fx.arith.maximumf(cur_f, gather_f)
                val_i32 = fx.arith.bitcast(_i32_ir, mx)
            return fx.Float32(fx.arith.bitcast(_f32_ir, val_i32))

        # ── Scale prefetch helpers (carry across iters) ───────────
        # For each K-tile we need m_repeat*groups_per_tile sx i32s per lane
        # (one per mi-row per K-128 group) and num_acc_n*groups_per_tile sy
        # i32s per lane (one per ni-col-block per K-128 group).
        #
        # sx i32 packs 4 K-128 groups per dword. For groups within the same
        # 4-group aligned K=512 region, the SAME i32 is reused (op_sel picks
        # the right byte). We exploit this by indexing the packed i32, not
        # the unpacked byte.

        # ── sx LDS load-once-per-WG ──────────────────────────────────
        # sx is `(B, H, R//512)` i32. For this WG we need rows
        # bx_m..bx_m+tile_m-1 × all R/512 K-packed-groups. Slab layout:
        #   slot(row_in_tile, k_packed_idx) = row_in_tile * (R/512) + k_packed_idx
        # Total i32s per WG: tile_m * (R/512). At tm=128 R=8192: 2048 (8 KB).
        #
        # Distribute across all 256 threads. We require perfect divisibility
        # so every thread loads the same number of slots with no bounds
        # check. This holds for tile_m * (R/512) % 256 == 0:
        #   tm=128, R≥1024 ✓; tm=64, R≥2048 ✓; tm=32, R≥4096 ✓.
        _sx_slots_per_thread = _sx_lds_count // total_threads

        def load_sx_to_lds():
            """Issue all sx loads + store into LDS. One-time per WG.
            Caller must barrier before any read from `lds_sx`.
            Each thread handles `_sx_slots_per_thread` slots strided by
            total_threads (so threads 0..255 cover slots 0..255 in chunk 0,
            then 256..511 in chunk 1, etc.) — coalesced gmem access.
            """
            for i in range_constexpr(_sx_slots_per_thread):
                slot_id = tx + fx.Index(i * total_threads)
                row_in_tile = slot_id // fx.Index(_sx_lds_per_row)
                k_packed_idx = slot_id % fx.Index(_sx_lds_per_row)
                row_a_global = bx_m + row_in_tile
                sx_gmem_idx = (
                    row_a_global * fx.Index(H * _sx_per_head)
                    + bx_h * fx.Index(_sx_per_head)
                    + k_packed_idx
                )
                sx_val = buffer_ops.buffer_load(
                    sx_rsrc,
                    fx.Int32(sx_gmem_idx),
                    vec_width=1,
                    dtype=fx.Int32,
                )
                v1 = Vec.from_elements([sx_val], fx.Int32)
                v1.store(lds_sx, [slot_id])

        # sy (Y scales) are loaded gmem->reg per K-tile directly in
        # prefetch_sy_tile below — NOT staged in LDS (tiny: num_acc_n*
        # groups_per_tile i32/lane). gmem layout: (H, D//128, R//512) i32.
        n_blocks_per_tile = max(1, tile_n // 128)

        def prefetch_sy_tile(kt):
            """Load this K-tile's sy (Y) scales gmem->reg, FLAT layout. Each entry
            is ONE ue8m0 byte BROADCAST to all 4 byte lanes of the i32.

            Loads (tile_n//128) * groups_per_tile i32 — one per (N-128 block in
            the CTA-tile, K-128 group). sy is PACKED ue8m0 (one i32 = 4 K-128
            groups, 1 byte each). A single ue8m0 byte is the scale for the whole
            128(N)×128(K) block, so we extract the K-correct byte (k_global % 4,
            CONSTEXPR) and broadcast it to all 4 lanes (byte * 0x01010101). Then
            the MFMA reads the correct scale from ANY lane → opselB is don't-care
            (we pass 0).

            Flat index:  sy_flat[nb + g * (tile_n//128)]   nb in [0,tile_n//128)
            CTA-tile's first N-128 block = by*tile_n//128; block nb at +nb.
            Distinct packed-i32 (k_global//4) are loaded once and reused.
            """
            tile_n_block0 = by * fx.Index(tile_n) // fx.Index(128)
            _loaded = {}  # (nb, k_packed) -> raw packed i32 (loaded once)
            sy_flat = []
            for g in range_constexpr(groups_per_tile):
                k_block_global_int = kt * (tile_k // 128) + g
                k_packed_idx = k_block_global_int // 4
                byte_in_i32 = k_block_global_int % 4  # constexpr
                for nb in range_constexpr(n_blocks_per_tile):
                    key = (nb, k_packed_idx)
                    if key not in _loaded:
                        if True:
                            sy_gmem_idx = (
                                bx_h * fx.Index(_sy_per_head)
                                + (tile_n_block0 + fx.Index(nb))
                                * fx.Index(_sy_per_n128)
                                + fx.Index(k_packed_idx)
                                + _scale_base_packed
                            )
                            _loaded[key] = buffer_ops.buffer_load(
                                sy_rsrc,
                                fx.Int32(sy_gmem_idx),
                                vec_width=1,
                                dtype=fx.Int32,
                            )
                    packed = _loaded[key]
                    # K-correct ue8m0 byte -> bits[7:0] (constexpr shift+mask)
                    if const_expr(byte_in_i32 == 0):
                        one_byte = fx.arith.andi(packed, fx.Int32(0xFF).ir_value())
                    else:
                        one_byte = fx.arith.andi(
                            fx.arith.shrui(
                                packed, fx.Int32(byte_in_i32 * 8).ir_value()
                            ),
                            fx.Int32(0xFF).ir_value(),
                        )
                    # broadcast that byte to all 4 lanes: byte * 0x01010101
                    sy_bcast = fx.arith.muli(one_byte, fx.Int32(0x01010101).ir_value())
                    sy_flat.append(sy_bcast)
            return sy_flat  # sy_flat[nb + g*(tile_n//128)], ue8m0 in every lane

        # ── MFMA + scale ──────────────────────────────────────────
        mfma_res_ty = Vec.make_type(4, fx.Float32)
        mfma_fp8_k128 = rocdl.mfma_scale_f32_16x16x128_f8f6f4

        def pack_i64x4_to_i32x8(x0, x1, x2, x3):
            return Vec.from_elements(
                [x0, x1, x2, x3],
                fx.Int64,
            ).bitcast(fx.Int32)

        # ── A0 prefetch helper ────────────────────────────────────
        # Returns (a0_i64, a1_i64) — the first K=64 chunk of A for (mi=0, g=0)
        # at lane's row/col. Pre-loaded into regs after barrier so the
        # next-iter compute_tile's first MFMA skips its ds_read.
        def lds_a0_prefetch(lds_buffer):
            fp8_row = lane_mod_16  # mi=0
            fp8_col_bytes = lane_div_16 * 16  # g=0
            return lds_load_a_packs_k64(fp8_row, fp8_col_bytes, lds_buffer)

        def lds_load_sa(mi, k_packed_idx):
            """Load one sx (A-scale) packed i32 from the LDS slab for this lane's
            row (bx_m + mi*16 + lane_mod_16) and K packed-i32 `k_packed_idx`.
            Called inside compute_tile's mi-loop right after the A LDS fetch.
            slot = row_in_tile * (R/512) + k_packed_idx (+ split-K base).
            The MFMA opsel byte (k_global % 4) picks the right ue8m0 (sx packed).

            NOTE (ATT-tested): an inline-asm `ds_read_b32` here (to hide the read
            from the compiler's alias analysis vs the opaque inline-asm B `... lds`
            DMA) does NOT remove the conservative vmcnt(0): the compiler simply
            relocates the guard to the next VISIBLE LDS read (the A-tile b128
            reads) — the vmcnt(0)->ds_read count stayed 8, total stall rose
            ~6.86M->7.46M, and perf dropped (1128->1040 TF) because the opaque
            read breaks the sx/MFMA double-buffer overlap. The guard is
            structural to having any opaque `... lds` DMA inflight, so we keep
            the plain compiler ds_read (better scheduling) and accept the vmcnt.
            """
            slot = (
                (lane_mod_16 + fx.Index(mi * 16)) * fx.Index(_sx_lds_per_row)
                + fx.Index(k_packed_idx)
                + _scale_base_packed
            )
            return Vec.load(Vec.make_type(1, fx.Int32), lds_sx, [slot])[0].ir_value()

        # ── compute_tile (scaled MFMA, no promote) ───────────────
        def compute_tile(
            accs_in,
            b_tile_in,
            kt,
            fp8_lds_buffer,
            sy_per_ni,
            a0_prefetch=None,
        ):
            """Compute one K-tile's MFMAs — 1896 TF reference (B-in-regs).

            b_tile_in is `packs_flat[ku][ni]` of i64 IR values from
            `load_b_tile(base_k_elem)` (gmem→regs). The B reg-list lives
            across compute_tile calls via the K-loop driver's b_pong/b_ping.
            """
            current_accs = list(accs_in)
            rocdl.iglp_opt(2)

            # sy_per_ni is the FLAT sy list from prefetch_sy_tile, indexed
            # sy_flat[nb + g*(tile_n//128)]. nb = ni's tile-local N-128 block =
            # (wave_id*n_per_wave + ni*16)//128. At tn<=256 a wave fits one
            # block so nb is the SAME for all ni in a wave; pick it once (runtime
            # via wave_id). n_blocks_per_tile is 1 (tn128) or 2 (tn256).
            sy_flat = sy_per_ni
            if const_expr(n_blocks_per_tile == 1):
                _wave_nb = None
            else:
                _wave_nb = fx.Int32(wave_id * fx.Index(n_per_wave) // fx.Index(128))

            def _sy_at(g):
                # return sy_flat[_wave_nb + g*n_blocks_per_tile]
                if const_expr(n_blocks_per_tile == 1):
                    return sy_flat[g]  # nb==0
                base = g * n_blocks_per_tile
                sel = sy_flat[base + 0]
                for nb in range_constexpr(1, n_blocks_per_tile):
                    is_nb = fx.arith.cmpi(
                        fx.arith.CmpIPredicate.eq,
                        _wave_nb.ir_value(),
                        fx.Int32(nb).ir_value(),
                    )
                    sel = fx.arith.select(is_nb, sy_flat[base + nb], sel)
                return sel

            a_128_list = [None, None]
            sx_i32_list = [None, None]
            for g in range_constexpr(groups_per_tile):
                k_block_global_int = kt * (tile_k // 128) + g
                byte_in_i32 = k_block_global_int % 4  # for sx opsel (sx packed)
                # sy: byte already extracted in prefetch_sy_tile -> bits[7:0];
                # _sy_at picks this wave's N-128 block from the flat list.
                sy_i32_g = _sy_at(g)

                fp8_group_col_base = g * 128

                # B packs for this g are 4 contiguous ku slots from b_tile_in.
                ku_base = g * mfmas_per_group
                b_pa = b_tile_in[ku_base + 0]
                b_pb = b_tile_in[ku_base + 1]
                b_pc = b_tile_in[ku_base + 2]
                b_pd = b_tile_in[ku_base + 3]

                # sx LDS slot for this (mi,g): row*(R/512) + k_packed (+ splitk
                # base). k_packed/byte are constexpr (kt,g). Loaded inside the
                # mi loop right after the A-from-LDS fetch (below).
                k_packed_idx = k_block_global_int // 4
                fp8_col_bytes_0 = fp8_group_col_base + lane_div_16 * 16
                fp8_col_bytes_1 = fp8_col_bytes_0 + 64
                # (mi=0, g=0) first pack-pair (a0,a1) may be PREFETCHED by the
                # driver (lds_a0_prefetch) after the barrier, so the first MFMA
                # skips its ds_read. a0_prefetch carries exactly (a0,a1) for
                # (mi=0, g=0, col=fp8_col_bytes_0); (a2,a3) is still loaded here.
                if const_expr(g == 0) and a0_prefetch is not None:
                    a0, a1 = a0_prefetch
                else:
                    a0, a1 = lds_load_a_packs_k64(
                        lane_mod_16,
                        fp8_col_bytes_0,
                        fp8_lds_buffer,
                    )
                a2, a3 = lds_load_a_packs_k64(
                    lane_mod_16,
                    fp8_col_bytes_1,
                    fp8_lds_buffer,
                )
                a_128_list[0] = pack_i64x4_to_i32x8(a0, a1, a2, a3)
                sx_i32_list[0] = lds_load_sa(0, k_packed_idx)
                for mi in range_constexpr(m_repeat):
                    mi_next = mi + 1
                    fp8_row = lane_mod_16 + (mi_next * 16)

                    a0, a1 = lds_load_a_packs_k64(
                        fp8_row,
                        fp8_col_bytes_0,
                        fp8_lds_buffer,
                    )
                    a2, a3 = lds_load_a_packs_k64(
                        fp8_row,
                        fp8_col_bytes_1,
                        fp8_lds_buffer,
                    )
                    a_128_list[mi_next % 2] = pack_i64x4_to_i32x8(a0, a1, a2, a3)
                    sx_i32_list[mi_next % 2] = lds_load_sa(mi_next, k_packed_idx)

                    for ni in range_constexpr(num_acc_n):
                        b128 = pack_i64x4_to_i32x8(
                            b_pa[ni],
                            b_pb[ni],
                            b_pc[ni],
                            b_pd[ni],
                        )
                        sy_i32 = sy_i32_g
                        acc_idx = mi * num_acc_n + ni
                        current_accs[acc_idx] = mfma_fp8_k128(
                            mfma_res_ty,
                            [
                                a_128_list[mi % 2],
                                b128,
                                current_accs[acc_idx],
                                0,
                                0,  # cbsz, blgp
                                byte_in_i32,
                                sx_i32_list[
                                    mi % 2
                                ],  # opsel_a, scale_a (sx still packed)
                                0,
                                sy_i32,
                            ],  # opsel_b=0: sy byte pre-extracted to bits[7:0]
                        )

            return current_accs

        # ── Output store ────────────────────────────────────────
        z_head_elem_off = bx_h * fx.Index(D)

        def store_output_bf16(final_accs):
            def body_row(*, mi, ii, row_in_tile, row):
                col_base_n = by_n + n_tile_base + lane_mod_16
                idx_base = row * fx.Index(stride_b_z) + z_head_elem_off + col_base_n
                for ni in range_constexpr(num_acc_n):
                    acc_idx = mi * num_acc_n + ni
                    acc = final_accs[acc_idx]
                    val = Vec(acc)[ii]
                    val_bf16 = fx.BFloat16(val)
                    idx_out = idx_base + (ni * 16)
                    buffer_ops.buffer_store(val_bf16, z_rsrc, idx_out)

            mfma_epilog(
                use_cshuffle=False,
                arith=fx.arith,
                range_constexpr=range_constexpr,
                m_repeat=m_repeat,
                lane_div_16=lane_div_16,
                bx_m=bx_m,
                body_row=body_row,
            )

        def store_output_splitk(final_accs):
            # split-K reduction, DEVICE-ZEROED, reduce DIRECTLY in bf16 into arg_z
            # (the real bf16 output) — no fp32 workspace, no host copy. Pattern
            # ported from splitk_hgemm.py (zero_c / split_k_barrier / packed-bf16
            # atomic write-back). Sequence per output tile (bx,by):
            #   1. zero_c(): the kz==0 CTA zeros this tile in arg_z, then raises
            #      signal[tile]; every other kz CTA spin-waits on signal so it
            #      never atomic-adds into not-yet-zeroed memory.
            #   2. all kz CTAs packed-bf16-atomic-add their partials into arg_z.
            #   3. the last arriver (semaphore counts to split_k-1) resets
            #      signal[tile]=semaphore[tile]=0 so the scratch is reusable.
            from flydsl._mlir.dialects import llvm as _llvm
            from flydsl._mlir.dialects import scf as _scf
            from flydsl._mlir import ir as _ir
            from flydsl._mlir.dialects import arith as _arith

            def _ptr1(base_tensor, elem_idx, elem_bytes):
                # Build an !llvm.ptr<1> (global) at base_tensor + elem_idx*bytes.
                # extract_base_index returns a raw index ir.Value; wrap it so the
                # byte-offset add happens in flydsl, then hand the index to
                # create_llvm_ptr (it casts index->i64->ptr internally).
                base = fx.Index(buffer_ops.extract_base_index(base_tensor))
                addr = base + fx.Index(elem_idx) * fx.Index(elem_bytes)
                p = buffer_ops.create_llvm_ptr(addr.ir_value(), address_space=1)
                return p._value if hasattr(p, "_value") else p

            _tid0 = _arith.cmpi(
                _arith.CmpIPredicate.eq,
                fx.Int32(tx).ir_value(),
                fx.Int32(0).ir_value(),
            )

            def zero_c():
                # kz==0 zeros this output tile in arg_z (bf16), then raises signal.
                cond_kz0 = _arith.cmpi(
                    _arith.CmpIPredicate.eq,
                    fx.Int32(kz).ir_value(),
                    fx.Int32(0).ir_value(),
                )
                if_kz0 = _scf.IfOp(cond_kz0, results_=[], has_else=False)
                with _ir.InsertionPoint(if_kz0.then_block):
                    z0 = fx.BFloat16(0.0)

                    def body_zero(*, mi, ii, row_in_tile, row):
                        col_base_n = by_n + n_tile_base + lane_mod_16
                        idx_base = (
                            row * fx.Index(stride_b_z) + z_head_elem_off + col_base_n
                        )
                        for ni in range_constexpr(num_acc_n):
                            idx_out = idx_base + (ni * 16)
                            buffer_ops.buffer_store(z0, z_rsrc, idx_out)

                    mfma_epilog(
                        use_cshuffle=False,
                        arith=fx.arith,
                        range_constexpr=range_constexpr,
                        m_repeat=m_repeat,
                        lane_div_16=lane_div_16,
                        bx_m=bx_m,
                        body_row=body_zero,
                    )
                    gpu.barrier()
                    # tid==0 publishes signal=1 (cache-bypassing store, sc0 sc1).
                    if_t0 = _scf.IfOp(_tid0, results_=[], has_else=False)
                    with _ir.InsertionPoint(if_t0.then_block):
                        sig_ptr = _ptr1(arg_signal, signal_idx, 4)
                        _llvm.InlineAsmOp(
                            None,
                            [sig_ptr, _arith.constant(fx.Int32.ir_type, 1)],
                            "global_store_dword $0, $1, off sc0 sc1",
                            "v,v",
                            has_side_effects=True,
                        )
                        _scf.YieldOp([])
                    gpu.barrier()
                    _scf.YieldOp([])

            def split_k_barrier():
                # Every CTA: spin until signal[tile]!=0 (tile is zeroed), then
                # count arrivals; the last arriver clears signal+semaphore.
                if_t0 = _scf.IfOp(_tid0, results_=[], has_else=False)
                with _ir.InsertionPoint(if_t0.then_block):
                    init0 = _arith.constant(fx.Int32.ir_type, 0)
                    w = _scf.WhileOp([fx.Int32.ir_type], [init0])
                    before = _ir.Block.create_at_start(w.before, [fx.Int32.ir_type])
                    after = _ir.Block.create_at_start(w.after, [fx.Int32.ir_type])
                    with _ir.InsertionPoint(before):
                        cur = before.arguments[0]
                        need = _arith.CmpIOp(
                            _arith.CmpIPredicate.eq,
                            cur,
                            _arith.constant(fx.Int32.ir_type, 0),
                        ).result
                        _scf.ConditionOp(need, [cur])
                    with _ir.InsertionPoint(after):
                        sig_ptr = _ptr1(arg_signal, signal_idx, 4)
                        data = _llvm.InlineAsmOp(
                            fx.Int32.ir_type,
                            [sig_ptr],
                            "global_load_dword $0, $1, off sc1",
                            "=v,v",
                            has_side_effects=True,
                        ).result
                        rocdl.s_waitcnt(0)
                        _scf.YieldOp([data])
                    _scf.YieldOp([])
                rocdl.sched_barrier(0)
                gpu.barrier()
                # last arriver resets scratch (so signal/semaphore are reusable).
                if_t0b = _scf.IfOp(_tid0, results_=[], has_else=False)
                with _ir.InsertionPoint(if_t0b.then_block):
                    sem_ptr = _ptr1(arg_semaphore, signal_idx, 4)
                    arrive = _llvm.AtomicRMWOp(
                        _llvm.AtomicBinOp.add,
                        sem_ptr,
                        _arith.constant(fx.Int32.ir_type, 1),
                        _llvm.AtomicOrdering.monotonic,
                        syncscope="agent",
                        alignment=4,
                    ).result
                    cond_last = _arith.cmpi(
                        _arith.CmpIPredicate.eq,
                        arrive,
                        fx.Int32(split_k - 1).ir_value(),
                    )
                    if_last = _scf.IfOp(cond_last, results_=[], has_else=False)
                    with _ir.InsertionPoint(if_last.then_block):
                        sem_ptr2 = _ptr1(arg_semaphore, signal_idx, 4)
                        sig_ptr2 = _ptr1(arg_signal, signal_idx, 4)
                        _llvm.InlineAsmOp(
                            None,
                            [sem_ptr2, _arith.constant(fx.Int32.ir_type, 0)],
                            "global_store_dword $0, $1, off sc0 sc1",
                            "v,v",
                            has_side_effects=True,
                        )
                        _llvm.InlineAsmOp(
                            None,
                            [sig_ptr2, _arith.constant(fx.Int32.ir_type, 0)],
                            "global_store_dword $0, $1, off sc0 sc1",
                            "v,v",
                            has_side_effects=True,
                        )
                        _scf.YieldOp([])
                    _scf.YieldOp([])
                gpu.barrier()

            zero_c()
            split_k_barrier()

            def body_row(*, mi, ii, row_in_tile, row):
                col_base_n = by_n + n_tile_base + lane_mod_16
                idx_base = row * fx.Index(stride_b_z) + z_head_elem_off + col_base_n
                for ni in range_constexpr(num_acc_n):
                    acc_idx = mi * num_acc_n + ni
                    val = Vec(final_accs[acc_idx])[ii]  # fp32 partial
                    v_bf16 = fx.BFloat16(val)
                    idx_out = idx_base + (ni * 16)
                    # Zero-partner packed bf16 atomic: each owned element sits at
                    # an even/odd bf16 slot (col stride 16 is even -> parity tracks
                    # lane_mod_16 parity). The lane writes [v,0] into the lo half
                    # if idx_out is even, else [0,v] into the hi half, targeting
                    # the shared 32-bit dword (idx_out//2)*4. The two lanes that
                    # share a dword each do a full packed 32-bit atomic RMW adding
                    # their half + 0 -> compose correctly. Lowers to the native
                    # gfx950 buffer_atomic_pk_add_bf16 (probe-verified).
                    z0 = fx.BFloat16(0.0)
                    is_even = _arith.cmpi(
                        _arith.CmpIPredicate.eq,
                        (fx.Index(idx_out) % fx.Index(2)).ir_value(),
                        fx.Index(0).ir_value(),
                    )
                    lo = fx.BFloat16(
                        _arith.select(is_even, v_bf16.ir_value(), z0.ir_value())
                    )
                    hi = fx.BFloat16(
                        _arith.select(is_even, z0.ir_value(), v_bf16.ir_value())
                    )
                    pair = Vec.from_elements([lo, hi], fx.BFloat16)
                    dword_byte = (fx.Index(idx_out) // fx.Index(2)) * fx.Index(4)
                    rocdl.raw_ptr_buffer_atomic_fadd(
                        pair.ir_value(),
                        z_rsrc,
                        fx.Int32(dword_byte).ir_value(),
                        fx.Int32(0).ir_value(),
                        fx.Int32(0).ir_value(),
                    )

            mfma_epilog(
                use_cshuffle=False,
                arith=fx.arith,
                range_constexpr=range_constexpr,
                m_repeat=m_repeat,
                lane_div_16=lane_div_16,
                bx_m=bx_m,
                body_row=body_row,
            )

        # ── qz epilogue ─────────────────────────────────────────
        # Algorithm:
        #   For each (mi, ii):
        #     For each d128_in_tile in [0, tile_n//128):
        #       1. local_amax over (num_acc_n / d128_per_tile) accs this lane owns
        #          in this d128 block (= 2 accs at tn=128; = 2 accs/block at tn=256).
        #       2. in-wave 16-lane reduce via intra_16_max_f32 (xor butterfly)
        #       3. (if waves_per_d128 > 1) one lane per (mi,ii,d128,wave) writes its
        #          partial to LDS slab, barrier, all lanes in d128's waves read all
        #          waves_per_d128 partials and fmax them.
        #       4. s = max(amax, 1e-30) * (1/448);  inv_s = 1/s
        #       5. scaled = acc * inv_s,  pack 8 f32 → i64 of fp8 (HW saturating RNE)
        #          → 1 i64 store per (row, d128_in_tile, lane) at the same col layout
        #            as the bf16 path. NOTE: 8 f32 = 4 cols × 2 ii rows of one lane in
        #            one d128 block, BUT pack_8 expects 8 contiguous fp32 within ONE
        #            row across 8 lanes — we do a different approach below.
        #
        # IMPORTANT detail: in the MFMA layout each lane owns 4 ROWS × num_acc_n COLS
        # (ii ∈ [0,4) gives 4 rows; ni gives cols of stride 16). The bf16 epilogue
        # writes 1 bf16 per (lane, ii, ni). For fp8 with a per-(row, d128) scale,
        # we must reduce amax PER ROW across cols within d128, then quantize and
        # store PER (lane, ii, ni-in-block) as a single fp8 byte.
        # We don't use pack_8_fp32_to_i64 — we use scalar `cvt_pk_fp8_f32` with
        # opsel byte selection to write a single fp8 byte per store via 1-byte
        # buffer_store (or alternatively, gather 4 ii into one 4-byte i32 per
        # (lane, ni-in-block) and one buffer_store_dword). v1 = scalar 1-B store
        # for simplicity; we'll optimise later if measured overhead is high.
        c_zero_f32 = fx.Float32(0.0)
        c_inv_448 = fx.Float32(1.0 / 448.0)
        c_amax_clamp_eps = fx.Float32(1e-30)
        # cvt_pk_fp8_f32 does not saturate — values > 448 produce NaN.
        # Clamp scaled accs to [-448, 448] before the cvt.
        c_pos_448 = fx.Float32(448.0)
        c_neg_448 = fx.Float32(-448.0)
        lane_id_i32 = fx.Int32(lane_id) if quant_output else None

        def store_output_qz(final_accs):
            """D-128 group fp8 quant epilogue.

            Per WG: tile_m rows × n_blocks_per_tile = tile_n/128 D-128 blocks.
            Reductions:
              tile_n == 128: ALL 4 waves contribute to the SAME single D-128 block
              tile_n == 256: waves 0+1 → block 0,  waves 2+3 → block 1
            """
            # Helpers for slot layout in lds_qz_amax.
            # slot(mi, ii, d128, wave_in_grp) =
            #   ((mi*4 + ii) * _qz_d128_per_tile + d128) * _qz_waves_per_d128
            #   + wave_in_grp
            _waves_pd = _qz_waves_per_d128  # python int
            _d128_pt = _qz_d128_per_tile  # python int
            # Per-tn config: which d128 block (within tile) a given wave belongs
            # to, and which slot (0..waves_per_d128-1) inside that block.
            #   tn==128: wave 0..3 → d128_owner=0, wave_in_grp=wave_id
            #   tn==256: waves 0+1 → d128_owner=0 (slot=wave_id);
            #            waves 2+3 → d128_owner=1 (slot=wave_id-2)
            if tile_n == 128:
                d128_owner_v = fx.Index(0)
                wave_in_grp_v = wave_id
            else:  # tile_n == 256
                d128_owner_v = wave_id // fx.Index(2)
                wave_in_grp_v = wave_id % fx.Index(2)

            # Which (ni) indices fall into which d128 block?
            # ni gives column ni*16 within the WAVE. Wave starts at col
            # wave_id * n_per_wave within tile. d128 boundary at col 128.
            # ni-in-d128 = ni for tn==128 (all num_acc_n=2 ni's are in block 0);
            # for tn==256 wave is in exactly one of 2 blocks, so all num_acc_n=4
            # ni's are in that one block.
            # → In both cases, every ni a lane holds is in ONE d128 block
            #   (the one owned by its wave). num_acc_n_per_d128 = num_acc_n.

            # === Phase 1: write per-(mi,ii,d128_owner,wave_in_grp) partial ===
            # We compute the amax for this lane's (mi, ii) over its
            # num_acc_n accs (= 16 cols × num_acc_n columns of its wave),
            # then 16-lane in-wave reduce, then one writer per wave.
            for mi in range_constexpr(m_repeat):
                for ii in range_constexpr(4):
                    # local_amax over this lane's num_acc_n accs at (mi, ii).
                    local_amax = c_zero_f32
                    for ni in range_constexpr(num_acc_n):
                        acc_idx = mi * num_acc_n + ni
                        v = Vec(final_accs[acc_idx])[ii]
                        neg_v = c_zero_f32 - v
                        abs_v = fx.Float32(
                            fx.arith.maximumf(v.ir_value(), neg_v.ir_value())
                        )
                        local_amax = fx.Float32(
                            fx.arith.maximumf(local_amax.ir_value(), abs_v.ir_value())
                        )
                    # In-wave 16-lane reduce (across lane_mod_16 → 16 cols).
                    row_amax = intra_16_max_f32(local_amax, lane_id_i32)
                    # Write 1 partial per (row_in_tile, d128_owner, wave_in_grp).
                    # row_in_tile = mi*16 + lane_div_16*4 + ii is the per-lane
                    # row index — unique per row, so lanes with different
                    # lane_div_16 never race on the same slot.
                    row_in_tile_v = (
                        fx.Index(mi * 16) + lane_div_16 * fx.Index(4) + fx.Index(ii)
                    )
                    base_slot_v = (
                        row_in_tile_v * fx.Index(_d128_pt) + d128_owner_v
                    ) * fx.Index(_waves_pd) + wave_in_grp_v
                    # All 16 lanes in the row-fragment hold the same
                    # row_amax value after intra_16_max_f32 (replicated).
                    # They all write the same bytes to the same LDS slot
                    # — idempotent, no race. We previously gated this on
                    # lane_mod_16==0 via scf.if, which caused ~5 μs of
                    # epilogue overhead per WG (16 scf.if invocations ×
                    # exec-divergence cost). Dropping the predicate is
                    # safe because the writes are bitwise identical.
                    #
                    # Earlier we saw non-determinism at (mi=1, ii=3) B=32
                    # — that turned out to be a missing s_waitcnt between
                    # back-to-back intra_16_max_f32 calls. The bpermute
                    # internally uses LDS; consecutive bpermutes need
                    # lgkmcnt=0 between them so each one reads its own
                    # input, not the prior call's pending value.
                    rocdl.s_waitcnt(0xC07F)  # lgkmcnt=0 only
                    v1 = Vec.from_elements([row_amax], fx.Float32)
                    v1.store(lds_qz_amax, [base_slot_v])

            gpu.barrier()

            # === Phase 2: each lane reads its d128's waves_per_d128 partials
            #     and fmax-reduces to get the cross-wave amax for (mi, ii). ===
            # Then computes s, inv_s, scales, and stores fp8 + scale.
            # Cache cross-wave amax per (mi, ii) — same for all (ni) in this wave.
            for mi in range_constexpr(m_repeat):
                # Compute row global: row = bx_m + mi*16 + lane_div_16*4 + ii
                row_base_in_tile = fx.Index(mi * 16) + lane_div_16 * fx.Index(4)
                for ii in range_constexpr(4):
                    row_in_tile = row_base_in_tile + fx.Index(ii)
                    row = bx_m + row_in_tile

                    # Read waves_per_d128 partials for this
                    # (row_in_tile, d128_owner). Matches Phase 1 layout.
                    row_in_tile_v2 = (
                        fx.Index(mi * 16) + lane_div_16 * fx.Index(4) + fx.Index(ii)
                    )
                    base_slot_v2 = (
                        row_in_tile_v2 * fx.Index(_d128_pt) + d128_owner_v
                    ) * fx.Index(_waves_pd)
                    cross_amax = c_zero_f32
                    for w in range_constexpr(_waves_pd):
                        slot = base_slot_v2 + fx.Index(w)
                        v = Vec.load(
                            Vec.make_type(1, fx.Float32),
                            lds_qz_amax,
                            [slot],
                        )
                        peer = fx.Float32(v[0])
                        cross_amax = fx.Float32(
                            fx.arith.maximumf(cross_amax.ir_value(), peer.ir_value())
                        )
                    # Clamp + scale.
                    clamped = fx.Float32(
                        fx.arith.maximumf(
                            cross_amax.ir_value(), c_amax_clamp_eps.ir_value()
                        )
                    )
                    s_fp32 = clamped * c_inv_448
                    # inv_s = 1 / s_fp32. Reciprocal of f32 scalar.
                    inv_s = fx.Float32(1.0) / s_fp32

                    # ── Quantize & store fp8 ──
                    # Column layout matches bf16 epilogue:
                    #   col_global = by_n + n_tile_base + lane_mod_16 + ni*16
                    # 1-byte stores per (lane, ii, ni). cvt_pk_fp8_f32 needs
                    # 2 fp32 → emits 2 fp8 bytes packed in a u16 slot of a
                    # working i32. We use it with src1=src0 and opsel=0 to
                    # output one byte (low byte of u16); then bitcast i32 →
                    # bytes and store the low byte.
                    # Simpler: use pack_8_fp32_to_i64 once per d128 if we can
                    # batch 8 cols × 1 ii into one store. But the layout
                    # gives one lane only num_acc_n cols (2 at tn=128, 4 at
                    # tn=256), so we can't fill 8 lanes worth here.
                    # → V1: emit num_acc_n × 1-byte stores per (lane, ii).
                    col_base_n = by_n + n_tile_base + lane_mod_16

                    # gmem index in fp8 BYTES (1 elem = 1 byte):
                    z_idx_base = (
                        row * fx.Index(stride_b_z) + z_head_elem_off + col_base_n
                    )
                    for ni in range_constexpr(num_acc_n):
                        acc_idx = mi * num_acc_n + ni
                        v = Vec(final_accs[acc_idx])[ii]
                        scaled_raw = v * inv_s
                        # cvt_pk_fp8_f32 does NOT saturate: inputs > 448
                        # produce NaN in the output. We must clamp to
                        # [-448, 448] explicitly before the cvt.
                        # min(max(scaled, -448), 448).
                        scaled_lo = fx.Float32(
                            fx.arith.maximumf(
                                scaled_raw.ir_value(),
                                c_neg_448.ir_value(),
                            )
                        )
                        scaled = fx.Float32(
                            fx.arith.minimumf(
                                scaled_lo.ir_value(),
                                c_pos_448.ir_value(),
                            )
                        )
                        # cvt_pk_fp8_f32 packs 2 fp32 → 2 fp8 bytes into the
                        # low half of an i32. We set src1=src0, opsel=0 →
                        # both fp8 bytes hold the same value; we extract the
                        # low byte via TruncI(i32→i8) and store 1 byte.
                        c0_i32 = fx.Int32(0)
                        packed_w = rocdl.cvt_pk_fp8_f32(
                            fx.Int32.ir_type,
                            scaled.ir_value(),
                            scaled.ir_value(),
                            c0_i32.ir_value(),
                            0,
                        )
                        # Truncate i32 → i8 (low byte = the fp8 we want).
                        from flydsl._mlir.dialects import arith as _arith_d
                        from flydsl._mlir import ir as _ir

                        i8_ty = _ir.IntegerType.get_signless(8)
                        byte_val_i8 = _arith_d.TruncIOp(i8_ty, packed_w).result
                        z_idx_out = z_idx_base + fx.Index(ni * 16)
                        buffer_ops.buffer_store(
                            byte_val_i8,
                            z_rsrc,
                            z_idx_out,
                        )

                    # ── Store scale ──
                    # One writer per (row, d128_global): pick lane_mod_16 == 0
                    # AND the first (ii==0) of the row-frag. But the scale is
                    # per (row, d128) — one scalar. We have 16 lanes × 4 ii
                    # × waves_per_d128 all holding the SAME s_fp32 here.
                    # Simplest: every lane writes the same value to the same
                    # gmem slot (idempotent). Same trade-off as Phase 1.
                    # d128_global = (by_n + n_tile_base) / 128 + ...
                    #   For tn==128: each wave has same d128_owner=0 within
                    #   tile, and by * tile_n / 128 = by since tn=128.
                    #   For tn==256: by*2 + d128_owner.
                    if tile_n == 128:
                        d128_global = by  # 1 d128 per tile, == by
                    else:  # tile_n == 256
                        d128_global = by * fx.Index(2) + d128_owner_v
                    if quant_transpose_scale:
                        # Layout (H, D//128, B): memory (H, D//128, B) contiguous.
                        # This MATCHES the HIP per_group_quant transpose layout
                        # used by the qz2 2-pass path (which feeds HIP a (B, H*D)
                        # view -> bytes (h*(D//128)+d128, b)), so qz1 and qz2
                        # return the SAME transposed scale for any (H, D, R, B).
                        # sz_idx = h*(D//128)*B + d128*B + b.
                        sz_idx = (
                            bx_h * fx.Index(D // 128) * c_b + d128_global * c_b + row
                        )
                    else:
                        # Layout (B, H, D//128) — sz_idx = b*(H*D//128) + h*(D//128) + d128
                        sz_idx = (
                            row * fx.Index(H * (D // 128))
                            + bx_h * fx.Index(D // 128)
                            + d128_global
                        )
                    # Predicate: skip OOB rows (row >= B). When B < tile_m,
                    # padding rows would otherwise compute an amax=0 (because
                    # OOB sx loads return 0 → 0 acc) and write a clamp-floor
                    # scale to an in-bounds-but-WRONG slot of sz_rsrc (the
                    # linear index wraps in the (B, H, D//128) layout for
                    # small B), overwriting valid scales for other d128
                    # blocks of real rows.
                    #
                    # The fp8 byte stores are already protected by z_rsrc's
                    # record-count bound (_z_nrec = c_b*H*D bytes; OOB row
                    # writes are silently dropped). Scale stores need this
                    # explicit predicate.
                    from flydsl._mlir.dialects import scf as _scf_d
                    from flydsl._mlir.dialects.arith import (
                        CmpIPredicate as _CmpIPred,
                    )
                    from flydsl._mlir import ir as _ir_d

                    _row_i32 = fx.Int32(row)
                    _b_i32 = fx.Int32(i32_b)
                    _row_in_range = fx.arith.cmpi(
                        _CmpIPred.ult,
                        _row_i32.ir_value(),
                        _b_i32.ir_value(),
                    )
                    _if_row = _scf_d.IfOp(_row_in_range)
                    with _ir_d.InsertionPoint(_if_row.then_block):
                        buffer_ops.buffer_store(s_fp32, sz_rsrc, sz_idx)
                        _scf_d.YieldOp([])

        # Picks the right epilogue based on mode.
        if quant_output:
            store_output = store_output_qz
        elif const_expr(split_k > 1):
            store_output = store_output_splitk
        else:
            store_output = store_output_bf16

        # ── K-loop driver: ONE unified 2-stage ping/pong driver ─────────
        # This single driver serves BOTH split_k==1 and split_k>1. The two used
        # to be separate (a hand-unrolled ping/pong for split_k==1 and a generic
        # rotating-buffer loop for split_k>1); they shared the load/compute/a0
        # sequence, so they are merged here. The only per-path differences are
        # selected by const_expr(split_k > 1):
        #
        #   * BARRIER placement. gpu.barrier() is replaced by rocdl.s_barrier()
        #     throughout this driver — we let the backend SIInsertWaitcnts pass
        #     insert vmcnt/lgkmcnt rather than hardcoding the conservative
        #     vmcnt(0) lgkmcnt(0) fence that gpu.barrier() implies.
        #       - split_k==1: ONE barrier AFTER each compute_tile (publishes the
        #         already-issued NEXT tile's A-DMA + WAR-protects the slab the
        #         next load will overwrite). This is the proven schedule — a
        #         single-barrier-per-iter variant regressed 1.6-5.4%.
        #       - split_k>1 : additionally a PRE-compute barrier (publish THIS
        #         tile's A-LDS before reading it), matching the former split-K
        #         driver's two-barriers-per-tile partial-drain schedule.
        #
        #   * PHANTOM GUARD for uneven split-K. _clamp_kt keeps phantom (out-of-
        #     slice) tile loads in-bounds; _guarded() masks their compute via a
        #     select on (kt < _my_tiles). Both compile to no-ops when _even
        #     (always true for split_k==1 and for even split-K), so split_k==1 is
        #     byte-for-byte the old hand-unrolled driver plus the s_barrier swap.
        #
        #   * K addressing. _k_off() already adds the runtime slice base kt0 when
        #     split_k>1 and is identity when ==1, so it is used uniformly.

        # ── workgroup barrier ──────────────────────────────────────
        # We emit `s_waitcnt vmcnt(N)` + `s_barrier` via inline asm (the same
        # idiom as splitk_hgemm.__barrier), letting the compiler manage lgkmcnt.
        # Using bare rocdl.s_barrier() (no explicit vmcnt) NaNs at tk=256/bsw=0:
        # the backend under-drains the gmem→LDS A-DMA before the cross-wave read.
        #
        # The barrier must publish the slab the NEXT compute reads (its A-DMA must
        # retire) but should NOT fully drain HBM — the NEXT tile's A loads, issued
        # just before this barrier, can stay in flight. From the ISA, the compiler
        # itself re-establishes exactly `vmcnt(num_a_loads)` right after our
        # barrier (the next tile's A-DMA count), then does the LDS reads. So
        # draining to `vmcnt(num_a_loads)` retires the slab-publish A-DMA (older,
        # in-order vmcnt) while keeping the next tile's A-DMA pipelined — instead
        # of the full `vmcnt(0)` drain that stalls the load pipeline.
        #
        from flydsl._mlir.dialects import llvm as _llvm_bar

        # next tile's A+B load count to keep in flight at the pre-compute barrier.
        _bar_keep_vmcnt = num_a_loads + num_b_loads

        def _waitcnt_barrier(vmcnt):
            _llvm_bar.InlineAsmOp(
                None,
                [],
                f"s_waitcnt vmcnt({vmcnt})\n\ts_barrier",
                "",
                has_side_effects=True,
            )

        # _even: True for split_k==1 and for K-slices that divide evenly. When
        # True, _clamp_kt / _guarded compile out (no phantom tail).
        _even = (split_k == 1) or ((R // 512) % split_k == 0)

        def _clamp_kt(kt):
            # Relative tile index for the A/B/scale LOADS. Even slices: identity,
            # returning the RAW Python int `kt` so _k_off folds the offset at
            # compile time (same as the old hand-unrolled path). Uneven: clamp
            # phantom tiles (kt >= my_tiles) to the last real tile (runtime
            # fx.Index) so they load valid in-slice data (no OOB); their compute
            # contribution is masked by _guarded below.
            if const_expr(_even):
                return kt
            in_range = fx.arith.cmpi(
                fx.arith.CmpIPredicate.slt,
                fx.Int32(fx.Index(kt)).ir_value(),
                fx.Int32(_my_tiles).ir_value(),
            )
            last = _my_tiles - fx.Index(1)
            return fx.Index(
                fx.Int32(
                    fx.arith.select(
                        in_range,
                        fx.Int32(fx.Index(kt)).ir_value(),
                        fx.Int32(last).ir_value(),
                    )
                )
            )

        def _guarded(new_accs, old_accs, kt):
            # Accept new_accs only if tile kt is real (kt < my_tiles), else keep
            # old (phantom tiles contribute nothing). No-op when _even.
            if const_expr(_even):
                return new_accs
            real = fx.arith.cmpi(
                fx.arith.CmpIPredicate.slt,
                fx.Int32(fx.Index(kt)).ir_value(),
                fx.Int32(_my_tiles).ir_value(),
            )
            return [
                Vec(fx.arith.select(real, Vec(na).ir_value(), Vec(oa).ir_value()))
                for na, oa in zip(new_accs, old_accs)
            ]

        Vec_init = fx.Vector.filled(4, 0.0, fx.Float32)
        accs = [Vec_init] * (num_acc_n * m_repeat)
        # Per-CTA K-tile loop length.
        #   split_k==1: full K = R/tile_k.
        #   split_k>1 : the guarded loop runs _max_tiles iterations (compile-
        #     time); each CTA does its runtime _my_tiles and predicates the rest
        #     (uneven slices). _my_tiles == _max_tiles for even slices.
        if const_expr(split_k > 1):
            num_tiles = _max_tiles
        else:
            num_tiles = _num_k_tiles

        def _k_off(base_elem):
            # base_elem is rel_kt*tile_k — either a Python int (even/split_k==1,
            # constexpr-foldable) or an fx.Index (uneven split-K clamped index).
            # fx.Index(x) accepts both (int -> constant; Index -> passthrough).
            if const_expr(split_k > 1):
                return kt0 * fx.Index(tile_k) + fx.Index(base_elem)
            return fx.Index(base_elem)

        load_sx_to_lds()
        # sy is loaded gmem->reg per K-tile in prefetch_sy_tile (no LDS slab).

        # ── Unified ping/pong driver (split_k==1 AND split_k>1) ──────────
        # B is gmem→regs (sync), A is gmem→LDS (async), double-buffered
        # ping/pong. Per tile (see _step):
        #   load_a_sync(next) ; load_b(next)
        #   s_waitcnt vmcnt(num_a+num_b) ; barrier()   # keep next A+B in flight,
        #       drain older -> publishes the CURRENT slab
        #   fetch_sy(cur) ; fetch_a0(cur)              # sx is read in compute_tile
        #   compute_tile(cur)
        #   barrier()                                  # WAR: cur read before reuse
        # The SAME schedule serves both split_k==1 and split_k>1. sy + a0 are
        # fetched for the CURRENT tile right before compute (not prefetched
        # ahead). _load_ab() applies the phantom clamp (no-op when even) so
        # phantom split-K tiles never read OOB; _guarded() masks their compute.
        # compute_tile's `kt` is the constexpr relative tile index in all cases
        # (absolute K folded via kt0/_scale_base_packed outside).

        def _load_ab(lds_slab, kt):
            # Issue ONLY the A gmem→LDS DMA + B gmem→reg for relative tile `kt`.
            # sx is read from LDS inside compute_tile (prefetch_sx_tile is a
            # no-op); sy + a0 are fetched for the CURRENT tile right before
            # compute (see _step). _clamp_kt keeps phantom split-K loads in-bounds.
            ck = _clamp_kt(kt)
            a_off = _k_off(ck * tile_k)
            load_a_tile_to_lds_async(a_off, lds_slab)
            return load_b_tile(a_off)

        # Per-tile pipeline, exactly:
        #   load_a_sync(next) ; load_b(next)
        #   s_waitcnt vmcnt(num_a + num_b) ; s_barrier   # keep next A+B in
        #       flight, drain older -> publishes CURRENT slab
        #   fetch_sx(cur) [no-op] ; fetch_sy(cur) ; fetch_a0(cur)
        #   compute_tile(cur)
        #   s_barrier                                    # WAR only (bare)
        # The pre-compute barrier publishes the CURRENT slab (loaded one step
        # earlier); the NEXT tile's A+B, just issued, stay pipelined because the
        # vmcnt keep doesn't drain them. The trailing bare s_barrier WAR-protects
        # the current slab before the next iteration's load overwrites it (no
        # vmem fence needed there).
        def _step(accs, cur, cur_lds, b_cur, nxt, nxt_lds):
            if nxt is not None:
                b_nxt = _load_ab(nxt_lds, nxt)
                _waitcnt_barrier(
                    _bar_keep_vmcnt
                )  # vmcnt(num_a+num_b): keep next A+B, publish cur
            else:
                b_nxt = None
                _waitcnt_barrier(
                    0
                )  # last tile: drain to publish cur (no next loads to keep)
            sy_cur = prefetch_sy_tile(cur)
            a0_cur = lds_a0_prefetch(cur_lds)
            accs = _guarded(
                compute_tile(accs, b_cur, cur, cur_lds, sy_cur, a0_prefetch=a0_cur),
                accs,
                cur,
            )
            rocdl.s_barrier()
            return accs, b_nxt

        # Prologue: issue tile-0's A+B into pong (publish happens in _step's
        # pre-compute barrier).
        b_cur = _load_ab(lds_a_pong, 0)
        for kt in range_constexpr(0, num_tiles):
            cur_lds = lds_a_pong if (kt % 2 == 0) else lds_a_ping
            nxt_lds = lds_a_ping if (kt % 2 == 0) else lds_a_pong
            nxt = (kt + 1) if (kt + 1) < num_tiles else None
            accs, b_cur = _step(accs, kt, cur_lds, b_cur, nxt, nxt_lds)

        store_output(accs)

    # ── Kernel + host launcher ───────────────────────────────
    # ── ONE unified kernel + jit, with thin per-variant Python launchers ──
    # All three output modes (bf16, qz, bf16 split-K) trace the SAME _kernel_body
    # and use the SAME grid, differing only in which extra tensors they consume:
    #   qz       -> arg_sz   (D-128 group scale, written by the in-kernel quant)
    #   split-K  -> arg_signal + arg_semaphore (device-zero/atomic-reduce scratch)
    # flydsl's @flyc.kernel builds the func signature from the runtime arg VALUES
    # and REJECTS None in a tensor slot, so rather than three signatures we
    # declare ONE kernel taking the full superset of 8 tensors and pass tiny
    # DUMMY tensors for the slots a given variant doesn't use. The dummies are
    # never dereferenced: _kernel_body is called with Python `None` for the
    # unused slots (quant_output / split_k are compile-time constants here), so
    # the corresponding branches are not traced. The external launcher ABI is
    # unchanged — the thin wrappers below keep the exact positional signatures
    # callers already use.

    import torch as _torch

    def _finalize_allocators():
        allocator_pong.finalized = False
        allocator_ping.finalized = False
        ctx = CompilationContext.get_current()
        from flydsl._mlir import ir

        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator_pong.finalize()
            allocator_ping.finalize()

    def _launch_grid(i32_b):
        gx_per_h = (i32_b + (tile_m - 1)) // tile_m
        # gridZ = split_k: each Z-CTA reduces one K-slice via atomicAdd.
        return (gx_per_h * H, D // tile_n, split_k)

    @flyc.kernel
    def kernel_gemm(
        arg_z: fx.Tensor,
        arg_sz: fx.Tensor,
        arg_x: fx.Tensor,
        arg_y: fx.Tensor,
        arg_sx: fx.Tensor,
        arg_sy: fx.Tensor,
        arg_signal: fx.Tensor,
        arg_semaphore: fx.Tensor,
        i32_b: fx.Int32,
    ):
        # Pass None (not the dummy tensor) for slots this config doesn't use, so
        # _kernel_body's trace-time `is None` gates fire and the dummy is never
        # traced/dereferenced. quant_output / split_k are closure constants.
        _kernel_body(
            arg_z,
            arg_sz if quant_output else None,
            arg_x,
            arg_y,
            arg_sx,
            arg_sy,
            i32_b,
            arg_signal if split_k > 1 else None,
            arg_semaphore if split_k > 1 else None,
        )

    @flyc.jit
    def launch_gemm(
        arg_z: fx.Tensor,
        arg_sz: fx.Tensor,
        arg_x: fx.Tensor,
        arg_y: fx.Tensor,
        arg_sx: fx.Tensor,
        arg_sy: fx.Tensor,
        arg_signal: fx.Tensor,
        arg_semaphore: fx.Tensor,
        i32_b: fx.Int32,
        stream: fx.Stream,
    ):
        _finalize_allocators()
        kernel_gemm._func.__name__ = KERNEL_NAME
        launcher = kernel_gemm(
            arg_z,
            arg_sz,
            arg_x,
            arg_y,
            arg_sx,
            arg_sy,
            arg_signal,
            arg_semaphore,
            i32_b,
        )
        launcher.launch(
            grid=_launch_grid(i32_b), block=(total_threads, 1, 1), stream=stream
        )

    # 1-element read-only dummy per device for unused kernel arg slots (never
    # written/read by the kernel; only satisfies flydsl's no-None rule).
    _dummy_cache: dict = {}

    def _dummy(device):
        d = _dummy_cache.get(device.index)
        if d is None:
            _dummy_cache[device.index] = d = _torch.zeros(
                1, dtype=_torch.float32, device=device
            )
        return d

    if quant_output:
        # qz: launch(z_fp8, sz_fp32, x, y, sx, sy, B, stream). signal/semaphore
        # unused -> dummies.
        def qz_launch(arg_z, arg_sz, arg_x, arg_y, arg_sx, arg_sy, i32_b, stream):
            stream = getattr(stream, "cuda_stream", stream)
            dummy = _dummy(arg_z.device)
            launch_gemm(
                arg_z,
                arg_sz,
                arg_x,
                arg_y,
                arg_sx,
                arg_sy,
                dummy,
                dummy,
                i32_b,
                int(stream),
            )

        return qz_launch

    if split_k > 1:
        # bf16 split-K: device-zeroed, packed-bf16-atomic reduction directly into
        # the bf16 output arg_z (NO fp32 workspace / host zero / copy). The first
        # K-slice CTA zeros each output tile, a signal/semaphore spin-wait barrier
        # orders the zero before the atomics, and the last arriver resets the
        # scratch so it is reusable across calls. arg_sz unused -> dummy.
        # ABI unchanged: split_k_launch(z_bf16, x, y, sx, sy, B, stream).

        # Persistent per-tile scratch (signal + semaphore), zeroed ONCE and
        # self-reset by the last CTA each launch -> reusable. Keyed by
        # (device, num_tiles). num_tiles MUST equal the device grid gx*gy.
        _splitk_scratch: dict = {}

        def _get_scratch(device, num_tiles):
            key = (device.index, num_tiles)
            t = _splitk_scratch.get(key)
            if t is None:
                signal = _torch.zeros(num_tiles, dtype=_torch.int32, device=device)
                semaphore = _torch.zeros(num_tiles, dtype=_torch.int32, device=device)
                _splitk_scratch[key] = t = (signal, semaphore)
            return t

        def split_k_launch(arg_z, arg_x, arg_y, arg_sx, arg_sy, i32_b, stream):
            B = int(i32_b)
            # Accept a raw HIP stream int or a torch.cuda.Stream object.
            stream = getattr(stream, "cuda_stream", stream)
            # Per-tile scratch count = device grid gx*gy =
            #   (ceil(B/tile_m)*H) * (D//tile_n). Keep this formula IN SYNC with
            #   _launch_grid / the device signal_idx, or the indexing corrupts.
            gx_per_h = (B + (tile_m - 1)) // tile_m
            num_tiles = gx_per_h * H * (D // tile_n)
            signal, semaphore = _get_scratch(arg_z.device, num_tiles)
            # Device self-zeros the output + scratch ordering is fully on-device,
            # so we just launch on the caller's stream (no side stream / events).
            launch_gemm(
                arg_z,
                _dummy(arg_z.device),
                arg_x,
                arg_y,
                arg_sx,
                arg_sy,
                signal,
                semaphore,
                i32_b,
                int(stream),
            )

        return split_k_launch

    # bf16, split_k == 1: launch(z_bf16, x, y, sx, sy, B, stream). sz + scratch
    # all unused -> dummies.
    def bf16_launch(arg_z, arg_x, arg_y, arg_sx, arg_sy, i32_b, stream):
        stream = getattr(stream, "cuda_stream", stream)
        dummy = _dummy(arg_z.device)
        launch_gemm(
            arg_z,
            dummy,
            arg_x,
            arg_y,
            arg_sx,
            arg_sy,
            dummy,
            dummy,
            i32_b,
            int(stream),
        )

    return bf16_launch


def compile_fp8_einsum_clean_ue8m0_qz_splitk(
    *,
    H: int,
    D: int,
    R: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    split_k: int,
    block_swizzle_n: int = 0,
    transpose_scale: bool = False,
):
    """2-PASS split-K fp8 (QZ) output.

    Pass 1: the bf16-split-K GEMM (quant_output=False, split_k>1) writes the
      COMPLETE fp32 output into a workspace via gridZ atomic reduction. This
      gives QZ the same CTA multiplication / occupancy that bf16 split-K gets
      (and allows tile_n<128 — the per-D128 amax is deferred to pass 2).
    Pass 2: a torch quant pass reads the complete fp32 workspace, computes the
      per-(B,H,D//128) amax, scale = amax/448, and casts to fp8 e4m3 + sz.
      Because it sees the FULL summed output, the amax is correct (unlike the
      in-kernel QZ epilogue, which can't span split-K partitions).

    ABI (same as compile_fp8_einsum_clean_ue8m0_qz):
      launch(z_fp8, sz_fp32, x_fp8, y_pre, sx_i32, sy_i32, B, stream)
      sz layout: (B, H, D//128), or (H, D//128, B) if transpose_scale.
    """
    import torch as _torch
    from aiter.ops.quant import per_group_quant_hip as _pgq
    from aiter.utility import dtypes as _dtypes

    # Pass-1 GEMM: the bf16-output split-K kernel (gridZ atomic -> fp32
    # workspace internally -> cast to bf16). Gives QZ the same CTA
    # multiplication bf16 split-K gets, and allows tile_n<128 (amax deferred).
    _gemm = compile_fp8_einsum_clean_ue8m0(
        H=H,
        D=D,
        R=R,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        block_swizzle_n=block_swizzle_n,
        quant_output=False,
        split_k=split_k,
    )

    def launch(arg_x, arg_y, arg_sx, arg_sy, i32_b, stream):
        """2-pass split-K QZ. RETURNS (z_fp8, sz) — no output buffers / copies.
        ABI: (x_fp8, y_pre, sx_i32, sy_i32, B, stream) -> (z_fp8, sz).
        """
        B = int(i32_b)
        dev = arg_x.device
        # pass 1: split-K GEMM -> COMPLETE bf16 output, written DIRECTLY into zb
        # via device-zeroed packed-bf16 atomic reduction (no fp32 workspace, no
        # host zero/cast/copy). `_gemm` (split_k_launch) launches on the caller's
        # `stream`; `_pgq` below runs on the current torch stream (== caller's),
        # so pass 2 is naturally ordered after pass 1 — no extra fence needed.
        zb = _torch.empty((B, H, D), dtype=_torch.bfloat16, device=dev)
        _gemm(zb, arg_x, arg_y, arg_sx, arg_sy, i32_b, stream)
        # pass 2: fused per-D128 fp8 group-quant (aiter HIP kernel, 1 launch);
        # allocates+returns z_fp8 & sz directly (no copy). Sees the complete
        # output -> per-D128 amax is correct.
        #
        # per_group_quant_hip's shuffle_scale (transpose_scale) writes the scale
        # TRANSPOSED as flat (groups, M) for a 2D (M, N) input, where groups =
        # N//128 (group axis OUTER) and M is the leading axis. To get the target
        # (H, D//128, B) layout with NO host permute/.contiguous() copy, feed the
        # 2D view (B, H*D): then M = B and the group axis spans H*(D//128) with H
        # OUTER (D is zb's contiguous inner axis), so the flat byte order is
        # (h*(D//128) + d128, b) == (H, D//128, B) exactly. qz1's in-kernel
        # transpose epilogue writes the SAME (H, D//128, B) layout, so both
        # paths agree.
        if transpose_scale:
            z_fp8_2d, sz = _pgq(
                zb.view(B, H * D),
                quant_dtype=_dtypes.fp8,
                group_size=128,
                transpose_scale=True,
            )
            # HIP returns sz.shape=(B, H*D//128) but the BYTES are
            # (H*(D//128), B); reinterpret to the true (H, D//128, B) layout. z
            # is unaffected by the scale shuffle — reshape it back to (B, H, D).
            z_fp8 = z_fp8_2d.view(B, H, D)
            sz = sz.view(H, D // 128, B)
        else:
            z_fp8, sz = _pgq(
                zb.view(B * H, D),
                quant_dtype=_dtypes.fp8,
                group_size=128,
                transpose_scale=False,
            )
            z_fp8 = z_fp8.view(B, H, D)
            sz = sz.view(B, H, D // 128)
        return z_fp8, sz

    return launch


def compile_fp8_einsum_clean_ue8m0_qz(
    *,
    H: int,
    D: int,
    R: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    block_swizzle_n: int = 0,
    transpose_scale: bool = False,
):
    """Compile fp8 einsum + D-128 group fp8 quant epilogue.

    Output ABI:
      arg_z : fp8 e4m3 (B, H, D)            (was bf16 in the non-qz variant)
      arg_sz: fp32     scale tensor — layout depends on `transpose_scale`:
                       False (default): (B, H, D // 128)
                       True:            (H, D // 128, B)
      arg_x : fp8 e4m3 (B, H, R)
      arg_y : fp8 e4m3 preshuffled (per-head packed)
      arg_sx: int32 (B, H, R // 512)        — packed UE8M0
      arg_sy: int32 (H, D // 128, R // 512) — packed UE8M0
      i32_b : runtime batch size

    Same K-loop, same MFMA pipeline, same scale prefetch as the bf16 variant.
    Epilogue: per-(row × D-128-block) amax (in-wave xor butterfly + cross-wave
    LDS slab) → fp32 scale `s = amax / 448 (>= 1e-30)` → manual clamp to
    [-448, 448] + RNE fp8 conversion via `cvt_pk_fp8_f32` (the HW
    instruction does NOT saturate — it produces NaN above 448).

    Args:
      transpose_scale: when True, scale is stored in `(H, D // 128, B)` layout
        instead of the default `(B, H, D // 128)`. Mirrors the
        `transpose_scale` option of `aiter.ops.quant.per_group_quant_hip` (fed
        a `(B, H*D)` view) — useful when the scale will be re-consumed by a
        downstream kernel that wants the per-head D-group axes on the outside.

    Restriction (v1): tile_n in {128, 256}. tile_n=128 → 1 D-128 block per WG
    (all 4 waves contribute). tile_n=256 → 2 D-128 blocks (2 waves each).
    Other restrictions identical to the bf16 variant.
    """
    return compile_fp8_einsum_clean_ue8m0(
        H=H,
        D=D,
        R=R,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        block_swizzle_n=block_swizzle_n,
        quant_output=True,
        quant_transpose_scale=transpose_scale,
    )
