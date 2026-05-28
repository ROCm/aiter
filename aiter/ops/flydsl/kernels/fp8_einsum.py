"""DeepGEMM-style clean FP8 einsum for gfx950 — v2 pipeline (preshuffle-style).

Reference shape: H=8 D=8192 R=8192 B=2048, tm=128 tn=128 tk=256, bsw=4,
sx+sy in LDS, B in regs via pingpong → ~1896 TF peak.

Public entry points (all in this module):
  compile_fp8_einsum_clean_ue8m0(*, H, D, R, tile_m, tile_n, tile_k,
                                 block_swizzle_n=0, quant_output=False,
                                 quant_transpose_scale=False)
      Underlying factory. Returns a launcher for either bf16 output (default)
      or in-kernel D-128 group fp8 quant output (quant_output=True).

  compile_fp8_einsum_clean_ue8m0_qz(*, H, D, R, ..., transpose_scale=False)
      Convenience wrapper — calls the underlying factory with
      quant_output=True. The qz launcher signature adds `arg_sz`
      (fp32 D-128 group scale) between `arg_z` and `arg_x`.

  compile_fp8_einsum_clean_ue8m0_auto(*, H, D, R, B=None, ...)
  compile_fp8_einsum_clean_ue8m0_qz_auto(*, H, D, R, B=None, ...)
      Autotune-dispatched: looks up the best (tile_m, tile_n, tile_k, bsw)
      from the per-shape table at the top of this file and forwards to the
      underlying factory. Use these by default unless you need to override
      the tile.

The experimental ni-rotated variant lives in fp8_einsum_clean.py as
`compile_fp8_einsum_clean_ue8m0_ni_rotated` (not used by autotune dispatch).

ABI (DeepGEMM SM100 packed-UE8M0):
  arg_z : bf16 (B, H, D)
  arg_x : fp8 e4m3 (B, H, R)             — caller pre-quantizes
  arg_y : fp8 e4m3 preshuffled per head, layout (n0, k0, klane=4, nlane=16, kpack=16)
  arg_sx: int32 (B, H, R // 512)         — 4 packed UE8M0 bytes per i32 along K
  arg_sy: int32 (H, D // 128, R // 512)  — same packing along K
  i32_b : runtime batch size

V2 pipeline (preshuffle-style async pingpong, scale-in-MFMA):
  - async-A: raw_ptr_buffer_load_lds with swizzle-the-source (no reg transit)
  - B, sx_i32, sy_i32 all carried in registers across iters
  - Scale fed DIRECTLY to MFMA (packed UE8M0 via opsel byte index)
  - a0_prefetch: first A pack of NEXT tile pre-loaded from LDS to regs before
    next-iter MFMA fires (hides 1 ds_read from the inner-loop critical path)
  - Unrolled pingpong: each loop iter handles 2 K-tiles
  - Single waitcnt per half: vmcnt(N_b+N_sx+N_sy)+lgkmcnt(0) AFTER compute_tile
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
    # (H,  D,    R,    B    ) -> (tm, tn, tk, bsw)
    # ─────────────────────────────────────────────────────────────────────
    # V4-Pro wo_a grouped LoRA: (H=G=n_local_groups, D=o_lora_rank=1024,
    # R=d_per_group=4096). G = 16 / tp_size, so H sweeps {2, 4, 8, 16}
    # covers TP=8, 4, 2, 1 respectively. Tuned 2026-05-27 on MI355X.
    # ─────────────────────────────────────────────────────────────────────
    # H=16 D=1024 R=4096 dynamic-B sweep  (TP=1)
    (16, 1024, 4096, 1): (32, 128, 256, 2),
    (16, 1024, 4096, 2): (32, 128, 256, 8),
    (16, 1024, 4096, 4): (32, 128, 256, 0),
    (16, 1024, 4096, 8): (32, 128, 256, 2),
    (16, 1024, 4096, 16): (32, 128, 256, 0),
    (16, 1024, 4096, 32): (32, 128, 256, 8),
    (16, 1024, 4096, 64): (64, 128, 256, 0),
    (16, 1024, 4096, 128): (64, 128, 256, 0),
    (16, 1024, 4096, 256): (128, 128, 256, 0),
    (16, 1024, 4096, 512): (64, 256, 128, 2),
    (16, 1024, 4096, 1024): (64, 256, 128, 4),
    (16, 1024, 4096, 4096): (128, 128, 128, 8),
    (16, 1024, 4096, 8192): (128, 128, 128, 4),
    (16, 1024, 4096, 16384): (128, 128, 128, 4),
    (16, 1024, 4096, 32768): (128, 128, 128, 4),
    # H=8 D=1024 R=4096  (TP=2)
    (8, 1024, 4096, 1): (32, 128, 256, 8),       # 5 TF
    (8, 1024, 4096, 2): (32, 128, 256, 0),       # 10 TF
    (8, 1024, 4096, 4): (32, 128, 256, 0),       # 19 TF
    (8, 1024, 4096, 8): (64, 128, 512, 4),       # 38 TF
    (8, 1024, 4096, 16): (64, 128, 256, 4),      # 76 TF
    (8, 1024, 4096, 32): (64, 128, 512, 0),      # 153 TF
    (8, 1024, 4096, 64): (32, 128, 512, 0),      # 303 TF
    (8, 1024, 4096, 128): (64, 128, 512, 8),     # 581 TF
    (8, 1024, 4096, 256): (64, 128, 512, 2),     # 940 TF
    (8, 1024, 4096, 512): (64, 256, 128, 2),     # 1336 TF
    (8, 1024, 4096, 1024): (128, 128, 256, 4),   # 1807 TF
    (8, 1024, 4096, 4096): (128, 128, 128, 8),   # 1795 TF
    (8, 1024, 4096, 8192): (128, 128, 128, 4),   # 1683 TF
    (8, 1024, 4096, 16384): (128, 128, 128, 8),  # 1806 TF
    # H=4 D=1024 R=4096  (TP=4)
    (4, 1024, 4096, 1): (32, 128, 128, 2),       # 2 TF
    (4, 1024, 4096, 2): (32, 128, 256, 8),       # 5 TF
    (4, 1024, 4096, 4): (32, 128, 128, 2),       # 9 TF
    (4, 1024, 4096, 8): (32, 128, 256, 8),       # 19 TF
    (4, 1024, 4096, 16): (32, 128, 512, 0),      # 38 TF
    (4, 1024, 4096, 32): (32, 128, 128, 2),      # 76 TF
    (4, 1024, 4096, 64): (32, 128, 256, 4),      # 153 TF
    (4, 1024, 4096, 128): (64, 128, 256, 0),     # 304 TF
    (4, 1024, 4096, 256): (32, 128, 512, 4),     # 609 TF
    (4, 1024, 4096, 512): (64, 128, 512, 4),     # 1036 TF
    (4, 1024, 4096, 1024): (64, 128, 256, 4),    # 1417 TF
    (4, 1024, 4096, 4096): (64, 256, 128, 2),    # 1759 TF
    (4, 1024, 4096, 8192): (128, 128, 128, 4),   # 1815 TF
    (4, 1024, 4096, 16384): (128, 128, 128, 4),  # 1681 TF
    # H=2 D=1024 R=4096  (TP=8)
    (2, 1024, 4096, 1): (32, 128, 128, 2),       # 1 TF
    (2, 1024, 4096, 2): (32, 128, 512, 2),       # 2 TF
    (2, 1024, 4096, 4): (32, 128, 128, 8),       # 5 TF
    (2, 1024, 4096, 8): (64, 128, 128, 0),       # 10 TF
    (2, 1024, 4096, 16): (32, 128, 128, 2),      # 19 TF
    (2, 1024, 4096, 32): (32, 128, 128, 2),      # 38 TF
    (2, 1024, 4096, 64): (32, 128, 128, 8),      # 75 TF
    (2, 1024, 4096, 128): (32, 128, 512, 8),     # 153 TF
    (2, 1024, 4096, 256): (32, 128, 512, 4),     # 308 TF
    (2, 1024, 4096, 512): (32, 128, 256, 4),     # 614 TF
    (2, 1024, 4096, 1024): (64, 128, 512, 4),    # 1048 TF
    (2, 1024, 4096, 4096): (128, 128, 128, 0),   # 1832 TF
    (2, 1024, 4096, 8192): (64, 256, 128, 4),    # 1823 TF
    (2, 1024, 4096, 16384): (128, 128, 256, 8),  # 1771 TF
    # H=8 D=R=8192 peak shapes (not a V4 shape; DSV3 prefill)
    (8, 8192, 8192, 128): (128, 128, 256, 0),
    (8, 8192, 8192, 512): (128, 256, 128, 0),
    (8, 8192, 8192, 2048): (128, 128, 256, 2),
}

_AUTOTUNE_WINNERS_QZ = {
    # ─────────────────────────────────────────────────────────────────────
    # V4-Pro wo_a (FP8 output, transpose_scale=True). Same per-shape tiles
    # as the BF16 sweep — the qz epilogue only adds a per-D128 amax + cast
    # at output time, which doesn't change the K-loop tile economics.
    # qz constraints: tile_n in {128, 256}; entries with tile_n=512 below
    # would fail to compile in qz mode and are excluded.
    # ─────────────────────────────────────────────────────────────────────
    # H=16 D=1024 R=4096 dynamic-B sweep  (TP=1)
    (16, 1024, 4096, 1): (32, 128, 256, 2),
    (16, 1024, 4096, 2): (32, 128, 256, 8),
    (16, 1024, 4096, 4): (32, 128, 256, 0),
    (16, 1024, 4096, 8): (32, 128, 256, 2),
    (16, 1024, 4096, 16): (32, 128, 256, 0),
    (16, 1024, 4096, 32): (32, 128, 256, 8),
    (16, 1024, 4096, 64): (64, 128, 256, 0),
    (16, 1024, 4096, 128): (64, 128, 256, 0),
    (16, 1024, 4096, 256): (128, 128, 256, 0),
    (16, 1024, 4096, 512): (64, 256, 128, 2),
    (16, 1024, 4096, 1024): (64, 256, 128, 4),
    (16, 1024, 4096, 4096): (128, 128, 128, 8),
    (16, 1024, 4096, 8192): (128, 128, 128, 4),
    (16, 1024, 4096, 16384): (128, 128, 128, 4),
    (16, 1024, 4096, 32768): (128, 128, 128, 4),
    # H=8 D=1024 R=4096  (TP=2). Tiles match _AUTOTUNE_WINNERS_BF16 with
    # tile_n constrained to {128, 256} for qz.
    (8, 1024, 4096, 1): (32, 128, 256, 8),
    (8, 1024, 4096, 2): (32, 128, 256, 0),
    (8, 1024, 4096, 4): (32, 128, 256, 0),
    (8, 1024, 4096, 8): (64, 128, 256, 4),
    (8, 1024, 4096, 16): (64, 128, 256, 4),
    (8, 1024, 4096, 32): (64, 128, 256, 0),
    (8, 1024, 4096, 64): (32, 128, 256, 0),
    (8, 1024, 4096, 128): (64, 128, 256, 8),
    (8, 1024, 4096, 256): (64, 128, 256, 2),
    (8, 1024, 4096, 512): (64, 256, 128, 2),
    (8, 1024, 4096, 1024): (128, 128, 256, 4),
    (8, 1024, 4096, 4096): (128, 128, 128, 8),
    (8, 1024, 4096, 8192): (128, 128, 128, 4),
    (8, 1024, 4096, 16384): (128, 128, 128, 8),
    # H=4 D=1024 R=4096  (TP=4)
    (4, 1024, 4096, 1): (32, 128, 128, 2),
    (4, 1024, 4096, 2): (32, 128, 256, 8),
    (4, 1024, 4096, 4): (32, 128, 128, 2),
    (4, 1024, 4096, 8): (32, 128, 256, 8),
    (4, 1024, 4096, 16): (32, 128, 256, 0),
    (4, 1024, 4096, 32): (32, 128, 128, 2),
    (4, 1024, 4096, 64): (32, 128, 256, 4),
    (4, 1024, 4096, 128): (64, 128, 256, 0),
    (4, 1024, 4096, 256): (32, 128, 256, 4),
    (4, 1024, 4096, 512): (64, 128, 256, 4),
    (4, 1024, 4096, 1024): (64, 128, 256, 4),
    (4, 1024, 4096, 4096): (64, 256, 128, 2),
    (4, 1024, 4096, 8192): (128, 128, 128, 4),
    (4, 1024, 4096, 16384): (128, 128, 128, 4),
    # H=2 D=1024 R=4096  (TP=8)
    (2, 1024, 4096, 1): (32, 128, 128, 2),
    (2, 1024, 4096, 2): (32, 128, 256, 2),
    (2, 1024, 4096, 4): (32, 128, 128, 8),
    (2, 1024, 4096, 8): (64, 128, 128, 0),
    (2, 1024, 4096, 16): (32, 128, 128, 2),
    (2, 1024, 4096, 32): (32, 128, 128, 2),
    (2, 1024, 4096, 64): (32, 128, 128, 8),
    (2, 1024, 4096, 128): (32, 128, 256, 8),
    (2, 1024, 4096, 256): (32, 128, 256, 4),
    (2, 1024, 4096, 512): (32, 128, 256, 4),
    (2, 1024, 4096, 1024): (64, 128, 256, 4),
    (2, 1024, 4096, 4096): (128, 128, 128, 0),
    (2, 1024, 4096, 8192): (64, 256, 128, 4),
    (2, 1024, 4096, 16384): (128, 128, 256, 8),
    # H=8 D=R=8192 peak shapes
    (8, 8192, 8192, 128): (128, 128, 128, 0),
    (8, 8192, 8192, 512): (128, 128, 256, 0),
    (8, 8192, 8192, 2048): (128, 128, 256, 4),
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
    (tm, tn, tk, bsw), _ = _autotune_lookup(table, H, D, R, B)
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
    )


def compile_fp8_einsum_clean_ue8m0_qz_auto(
    *,
    H: int,
    D: int,
    R: int,
    B: int | None = None,
    transpose_scale: bool = False,
):
    """Convenience wrapper: autotune-dispatch qz variant.

    Equivalent to `compile_fp8_einsum_clean_ue8m0_auto(..., quant_output=True,
    quant_transpose_scale=transpose_scale)`.
    """
    return compile_fp8_einsum_clean_ue8m0_auto(
        H=H,
        D=D,
        R=R,
        B=B,
        quant_output=True,
        quant_transpose_scale=transpose_scale,
    )


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
                       ``(D // 128, B, H)`` instead of the default
                       ``(B, H, D // 128)``. Mirrors
                       ``aiter.ops.quant.per_group_quant_hip``'s option.
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

    # ── Allocate outputs ──
    device = x_fp8.device
    if is_qz:
        z = torch.empty(B, H, D, dtype=torch.float8_e4m3fn, device=device)
        if transpose_scale:
            sz = torch.empty(D // 128, B, H, dtype=torch.float32, device=device)
        else:
            sz = torch.empty(B, H, D // 128, dtype=torch.float32, device=device)
    else:
        z = torch.empty(B, H, D, dtype=torch.bfloat16, device=device)
        sz = None

    # ── Launch ──
    if stream is None:
        stream = torch.cuda.current_stream()
    if is_qz:
        kernel(z, sz, x_fp8, y_pre, sx, sy, B, stream)
    else:
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
        tensor `arg_sz` is laid out as `(D // 128, B, H)` fp32 (the D-128
        group axis is leading) rather than the default `(B, H, D//128)`.
        Mirrors `aiter.ops.quant.per_group_quant_hip`'s `transpose_scale`
        option — useful when the scale will feed a downstream kernel that
        expects the scale-group axis on the outside. Requires `quant_output=True`.

    Returns: launcher with signature
      bf16 mode: launch(z_bf16, x_fp8, y_pre, sx_i32, sy_i32, B, stream)
      qz   mode: launch(z_fp8, sz_fp32, x_fp8, y_pre, sx_i32, sy_i32, B, stream)

    Note: lds_stage param dropped — only ping/pong is supported in v2.
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
    if tile_n % 128 != 0:
        raise ValueError(
            f"tile_n must be a multiple of 128 (N-128-block granularity). "
            f"Got tile_n={tile_n}."
        )
    if tile_n < 128:
        raise ValueError(f"tile_n must be >= 128, got {tile_n}")
    if R % tile_k != 0:
        raise ValueError(f"R={R} must be divisible by tile_k={tile_k}")
    if D % tile_n != 0:
        raise ValueError(f"D={D} must be divisible by tile_n={tile_n}")
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
    a_load_bytes = 16
    if bytes_per_thread_a % a_load_bytes != 0:
        raise ValueError(
            f"bytes_per_thread_a={bytes_per_thread_a} must be divisible by 16"
        )
    num_a_loads = bytes_per_thread_a // a_load_bytes

    # B bytes (preshuffled fp8)
    tile_k_bytes_b = tile_k * elem_bytes_b
    bytes_b_per_tile = tile_n * tile_k_bytes_b
    bytes_per_thread_b = bytes_b_per_tile // total_threads
    b_load_bytes = 16
    num_b_loads = bytes_per_thread_b // b_load_bytes

    # LDS allocators (always ping/pong)
    allocator_pong = SmemAllocator(None, arch=gpu_arch, global_sym_name="smem0")
    allocator_ping = SmemAllocator(None, arch=gpu_arch, global_sym_name="smem1")
    lds_stage = 2  # hardcoded — v2 always ping/pong

    def _alloc_per_stage(byte_size, stages):
        offsets, allocs = [], []
        for s in range(stages):
            alloc = allocator_pong if s == 0 else allocator_ping
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

    # B fp8 is kept in registers (gmem→regs via load_b_tile); no LDS slab.

    # Number of N-128-blocks per WG (= tile_n / 128). At tile_n=128 this is
    # 1 (the original assumption); at tile_n=256, 2; etc.
    n_blocks_per_tile = tile_n // 128

    # sy LDS slab — persistent per WG, loaded once in prologue.
    # sy is `(H, D//128, R//512)` i32; per WG we pick 1 head + n_blocks_per_tile
    # N-128-blocks. Layout (row-major):
    #   slot(nb_idx, k_packed_idx) = nb_idx * (R/512) + k_packed_idx
    # Bytes: n_blocks_per_tile * (R/512) * 4. At tn=128 R=8192: 1*16*4 = 64.
    # At tn=256 R=8192: 2*16*4 = 128.
    _sy_lds_per_nb = R // 512  # i32 entries per N-128-block
    _sy_lds_count = n_blocks_per_tile * _sy_lds_per_nb
    _sy_lds_bytes = _sy_lds_count * 4
    sy_lds_offset = allocator_pong._align(allocator_pong.ptr, 16)
    allocator_pong.ptr = sy_lds_offset + max(16, _sy_lds_bytes)

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
    ):
        Vec = fx.Vector
        fp8_dtype = fx.Float8E4M3FN

        c_b = fx.Index(i32_b)

        tx = gpu.thread_id("x")
        bx_raw = gpu.block_id("x")
        by_raw = gpu.block_id("y")

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

        # sy LDS slab — persistent, R/512 i32 entries (max R/512 = 16 at R=8192).
        _sy_lds_ptr = SmemPtr(
            base_ptr_pong,
            sy_lds_offset,
            fx.Int32.ir_type,
            shape=(max(4, _sy_lds_count),),
        )
        lds_sy = _sy_lds_ptr.get()

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
        # qz: fp8 byte output (1 B/elem). bf16: 2 B/elem.
        _z_bytes_per_elem = 1 if quant_output else 2
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
        layout_wave_lane = fx.make_layout((4, wave_size), (64, 1))
        coord_wave_lane = fx.idx2crd(tx, layout_wave_lane)
        wave_id = fx.get(coord_wave_lane, 0)
        lane_id = fx.get(coord_wave_lane, 1)

        layout_lane16 = fx.make_layout((4, 16), (16, 1))
        coord_lane16 = fx.idx2crd(lane_id, layout_lane16)
        lane_div_16 = fx.get(coord_lane16, 0)
        lane_mod_16 = fx.get(coord_lane16, 1)

        # N-tile column indexing per wave
        n_tile_base = wave_id * n_per_wave
        n_blk_list = []
        n_intra_list = []
        for i in range_constexpr(num_acc_n):
            global_n_in_head = by_n + n_tile_base + (i * 16) + lane_mod_16
            n_blk_list.append(global_n_in_head // 16)
            n_intra_list.append(global_n_in_head % 16)

        _b_stride_n0_c = fx.Index(_stride_n0)
        _b_stride_k0_c = fx.Index(_stride_k0)
        _b_stride_klane_c = fx.Index(_stride_klane)
        _b_stride_nlane_c = fx.Index(_stride_nlane)

        y_head_byte_off = bx_h * fx.Index(b_elems_per_head)

        stride_b_x = H * R
        stride_b_z = H * D

        # ── A async copy: gmem → LDS direct (swizzle-the-source) ──────
        tile_k_dwords = tile_k // 4
        c4 = fx.Index(4)
        tx_i32_base = tx * c4
        layout_a_tile_div4 = fx.make_layout((tile_m, tile_k_dwords), (tile_k_dwords, 1))

        def a_tile_chunk_coord(i):
            return tile_chunk_coord_i32(
                fx.arith,
                tx_i32_base=tx_i32_base,
                i=i,
                total_threads=total_threads,
                layout_tile_div4=layout_a_tile_div4,
            )

        x_head_elem_off = bx_h * fx.Index(R)
        _lds_a_fp8_k_dim_c = fx.Index(lds_a_fp8_stride_bytes)
        _lds_a_fp8_k_blocks16_c = fx.Index(lds_a_fp8_k_blocks16)

        # async-A constants
        _a_async_load_bytes = 16  # vec4 i32 per lane per chunk
        _a_wave_bytes_per_chunk = wave_size * _a_async_load_bytes  # 1024
        _a_chunk_stride_bytes = total_threads * _a_async_load_bytes  # 4096

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

        def prefetch_sx_tile(kt):
            """Return list[m_repeat][groups_per_tile] of i32 IR values
            read from the persistent LDS sx slab.

            Per (mi, g): lane G needs sx for row `bx_m + mi*16 + lane_mod_16`
            and K-128-group `kt*(tile_k/128) + g`. LDS slot:
                slot = (mi*16 + lane_mod_16) * (R/512) + (k_block_global // 4)
            byte_in_i32 = k_block_global % 4 (consumed by compute_tile's opsel).
            """
            sx_per_mi = []
            for mi in range_constexpr(m_repeat):
                row_in_tile = lane_mod_16 + fx.Index(mi * 16)
                sx_for_mi = []
                for g in range_constexpr(groups_per_tile):
                    k_block_global_int = kt * (tile_k // 128) + g
                    k_packed_idx = k_block_global_int // 4
                    slot = row_in_tile * fx.Index(_sx_lds_per_row) + fx.Index(
                        k_packed_idx
                    )
                    v = Vec.load(
                        Vec.make_type(1, fx.Int32),
                        lds_sx,
                        [slot],
                    )
                    sx_for_mi.append(v[0].ir_value())
                sx_per_mi.append(sx_for_mi)
            return sx_per_mi

        # ── sy LDS load-once-per-WG ──────────────────────────────────
        # sy is `(H, D//128, R//512)` i32. For this WG we need one head
        # (bx_h) and n_blocks_per_tile = tile_n/128 consecutive N-128-blocks
        # starting at by * n_blocks_per_tile. Slab layout:
        #   slot(nb_idx, k_packed_idx) = nb_idx * (R/512) + k_packed_idx
        # Total i32s: n_blocks_per_tile * (R/512). At tn=128 R=8192: 16.
        # At tn=256 R=8192: 32. Tiny; load once.
        #
        # Distribution: thread tx loads slot tx (if tx < _sy_lds_count),
        # decomposed to (nb_idx = tx // (R/512), k_packed_idx = tx % (R/512)).
        # For shapes where _sy_lds_count <= total_threads, threads with
        # tx >= _sy_lds_count just no-op (don't store).
        def load_sy_to_lds():
            """Issue all sy loads + store into LDS. One-time per WG.
            Caller must barrier before any read from `lds_sy`.
            """
            for slot_id in range_constexpr(_sy_lds_count):
                nb_idx = slot_id // _sy_lds_per_nb
                k_packed_idx = slot_id % _sy_lds_per_nb
                # WG-global N-128-block = by * n_blocks_per_tile + nb_idx
                n_block_global = by * fx.Index(n_blocks_per_tile) + fx.Index(nb_idx)
                sy_gmem_idx = (
                    bx_h * fx.Index(_sy_per_head)
                    + n_block_global * fx.Index(_sy_per_n128)
                    + fx.Index(k_packed_idx)
                )
                sy_val = buffer_ops.buffer_load(
                    sy_rsrc,
                    fx.Int32(sy_gmem_idx),
                    vec_width=1,
                    dtype=fx.Int32,
                )
                v1 = Vec.from_elements([sy_val], fx.Int32)
                v1.store(lds_sy, [fx.Index(slot_id)])

        def prefetch_sy_tile(kt):
            """Return list[num_acc_n][groups_per_tile] of i32 IR values
            read from the persistent LDS sy slab.

            Per (ni, g):
              n_col_in_tile = wave_id * n_per_wave + ni * 16 + lane_mod_16
              n_block_idx_local = n_col_in_tile // 128   (relative to this WG)
              k_block_global_int = kt * (tile_k//128) + g
              k_packed_idx = k_block_global_int // 4
              slot = n_block_idx_local * (R/512) + k_packed_idx
              byte_in_i32 = k_block_global_int % 4  (consumed by compute_tile)

            Note: lane_mod_16 is < 16 < 128, so within one (ni), all lanes
            have the same n_block_idx_local. We compute it once per (ni).
            """
            sy_per_ni = []
            for ni in range_constexpr(num_acc_n):
                # n_block_idx_local for this ni (wave-uniform within ni).
                # = (wave_id * n_per_wave + ni * 16) // 128
                # since lane_mod_16 < 16 < 128 doesn't shift the // 128 result.
                n_col_in_tile_wave_uniform = wave_id * fx.Index(n_per_wave) + fx.Index(
                    ni * 16
                )
                n_block_idx_local = n_col_in_tile_wave_uniform // fx.Index(128)
                sy_for_ni = []
                for g in range_constexpr(groups_per_tile):
                    k_block_global_int = kt * (tile_k // 128) + g
                    k_packed_idx = k_block_global_int // 4
                    slot = n_block_idx_local * fx.Index(_sy_lds_per_nb) + fx.Index(
                        k_packed_idx
                    )
                    v = Vec.load(
                        Vec.make_type(1, fx.Int32),
                        lds_sy,
                        [slot],
                    )
                    sy_for_ni.append(v[0].ir_value())
                sy_per_ni.append(sy_for_ni)
            return sy_per_ni

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

        # ── compute_tile (scaled MFMA, no promote) ───────────────
        # kt is the compile-time K-tile index for this call.
        # sx_per_mi, sy_per_ni are pre-loaded scale regs (carried across iters).
        # a0_prefetch is the optional (a0, a1) tuple for (mi=0, g=0, ku=0..1)
        # — when supplied, the first MFMA's A skips ds_read.
        def compute_tile(
            accs_in,
            b_tile_in,
            kt,
            fp8_lds_buffer,
            sx_per_mi,
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

            for g in range_constexpr(groups_per_tile):
                k_block_global_int = kt * (tile_k // 128) + g
                byte_in_i32 = k_block_global_int % 4

                fp8_group_col_base = g * 128

                # B packs for this g are 4 contiguous ku slots from b_tile_in.
                ku_base = g * mfmas_per_group
                b_pa = b_tile_in[ku_base + 0]
                b_pb = b_tile_in[ku_base + 1]
                b_pc = b_tile_in[ku_base + 2]
                b_pd = b_tile_in[ku_base + 3]

                for mi in range_constexpr(m_repeat):
                    fp8_row = lane_mod_16 + (mi * 16)

                    fp8_col_bytes_0 = fp8_group_col_base + lane_div_16 * 16
                    fp8_col_bytes_1 = fp8_col_bytes_0 + 64

                    if const_expr((a0_prefetch is not None) and (g == 0) and (mi == 0)):
                        a0, a1 = a0_prefetch
                    else:
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
                    a128 = pack_i64x4_to_i32x8(a0, a1, a2, a3)

                    sx_i32 = sx_per_mi[mi][g]

                    for ni in range_constexpr(num_acc_n):
                        b128 = pack_i64x4_to_i32x8(
                            b_pa[ni],
                            b_pb[ni],
                            b_pc[ni],
                            b_pd[ni],
                        )
                        sy_i32 = sy_per_ni[ni][g]
                        acc_idx = mi * num_acc_n + ni
                        current_accs[acc_idx] = mfma_fp8_k128(
                            mfma_res_ty,
                            [
                                a128,
                                b128,
                                current_accs[acc_idx],
                                0,
                                0,  # cbsz, blgp
                                byte_in_i32,
                                sx_i32,  # opsel_a, scale_a
                                byte_in_i32,
                                sy_i32,
                            ],  # opsel_b, scale_b
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
                        # Layout (D//128, B, H) — sz_idx = d128*(B*H) + b*H + h
                        sz_idx = (
                            d128_global * (c_b * fx.Index(H)) + row * fx.Index(H) + bx_h
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
        else:
            store_output = store_output_bf16

        # ── K-loop driver: unrolled pingpong (preshuffle-style) ──
        # We mirror preshuffle_gemm's structure:
        #   prologue: load tile-0 into pong, prefetch a0_pong from pong
        #   loop iter k in 0..num_tiles-2 (step 2):
        #     ── half-pong→ping ──
        #     issue async A tile (k+1) → ping
        #     b_ping, sx_ping, sy_ping = load (k+1)
        #     compute_tile on (b_pong, lds_pong, sx_pong, sy_pong, a0_pong)
        #     s_waitcnt(vmcnt=N_keep, lgkmcnt=0); barrier
        #     a0_ping = prefetch first A pack from ping
        #     ── half-ping→pong ──
        #     issue async A tile (k+2) → pong
        #     b_pong, sx_pong, sy_pong = load (k+2)
        #     compute_tile on (b_ping, lds_ping, sx_ping, sy_ping, a0_ping)
        #     s_waitcnt(...); barrier
        #     a0_pong = prefetch first A pack from pong
        #   epilog: depends on parity of num_tiles
        #
        # Note: We also tried a single-barrier-per-iter variant (load both
        # pong+ping in prologue, drain both DMAs with one barrier). It
        # regressed 1.6-5.4% — ATT showed the extra barrier was actually
        # cheaper than the increased compute/MFMA stall + gmem-load stall
        # caused by the larger barrier wait window.
        # Archived in att_1barrier_regression/.
        # n_vmem_keep counts B sync gmem loads (in vmcnt queue) that
        # should remain in flight when we hit the post-compute barrier.
        n_vmem_keep = num_b_loads

        # s_waitcnt encoding: vmcnt=N_keep, expcnt=max, lgkmcnt=0.
        def _waitcnt_vmcnt_lgkm0(vmc):
            vm_lo = vmc & 0xF
            vm_hi = (vmc >> 4) & 0x3
            return vm_lo | (7 << 4) | (0 << 8) | (vm_hi << 14)

        _waitcnt_imm = _waitcnt_vmcnt_lgkm0(n_vmem_keep)

        Vec_init = fx.Vector.filled(4, 0.0, fx.Float32)
        accs = [Vec_init] * (num_acc_n * m_repeat)
        num_tiles = R // tile_k

        # 1896 TF reference: B is gmem→regs (sync), A is gmem→LDS (async).
        # Per-iter pattern:
        #   issue async A (next k) → ping
        #   sync-load B (next k) → b_ping regs
        #   compute_tile on (b_pong regs, lds_a_pong)
        #   s_waitcnt(vmcnt=N_b, lgkmcnt=0) + barrier
        load_sx_to_lds()
        load_sy_to_lds()
        load_a_tile_to_lds_async(fx.Index(0), lds_a_pong)
        b_pong = load_b_tile(fx.Index(0))
        gpu.barrier()
        sx_pong = prefetch_sx_tile(0)
        sy_pong = prefetch_sy_tile(0)
        a0_pong = lds_a0_prefetch(lds_a_pong)

        if const_expr(num_tiles == 1):
            accs = compute_tile(
                accs,
                b_pong,
                0,
                lds_a_pong,
                sx_pong,
                sy_pong,
                a0_prefetch=a0_pong,
            )
        else:
            if const_expr(num_tiles % 2 == 1):
                _loop_end_excl = num_tiles - 1
            else:
                _loop_end_excl = num_tiles - 2

            for kt_base in range_constexpr(0, _loop_end_excl, 2):
                next_k1 = fx.Index((kt_base + 1) * tile_k)
                load_a_tile_to_lds_async(next_k1, lds_a_ping)
                b_ping = load_b_tile(next_k1)
                sx_ping = prefetch_sx_tile(kt_base + 1)
                sy_ping = prefetch_sy_tile(kt_base + 1)

                accs = compute_tile(
                    accs,
                    b_pong,
                    kt_base,
                    lds_a_pong,
                    sx_pong,
                    sy_pong,
                    a0_prefetch=a0_pong,
                )

                rocdl.s_waitcnt(_waitcnt_imm)
                gpu.barrier()
                a0_ping = lds_a0_prefetch(lds_a_ping)

                next_k2 = fx.Index((kt_base + 2) * tile_k)
                load_a_tile_to_lds_async(next_k2, lds_a_pong)
                b_pong = load_b_tile(next_k2)
                sx_pong = prefetch_sx_tile(kt_base + 2)
                sy_pong = prefetch_sy_tile(kt_base + 2)

                accs = compute_tile(
                    accs,
                    b_ping,
                    kt_base + 1,
                    lds_a_ping,
                    sx_ping,
                    sy_ping,
                    a0_prefetch=a0_ping,
                )

                rocdl.s_waitcnt(_waitcnt_imm)
                gpu.barrier()
                a0_pong = lds_a0_prefetch(lds_a_pong)

            if const_expr(num_tiles % 2 == 1):
                accs = compute_tile(
                    accs,
                    b_pong,
                    _loop_end_excl,
                    lds_a_pong,
                    sx_pong,
                    sy_pong,
                    a0_prefetch=a0_pong,
                )
            else:
                next_k1 = fx.Index((_loop_end_excl + 1) * tile_k)
                load_a_tile_to_lds_async(next_k1, lds_a_ping)
                b_ping = load_b_tile(next_k1)
                sx_ping = prefetch_sx_tile(_loop_end_excl + 1)
                sy_ping = prefetch_sy_tile(_loop_end_excl + 1)

                accs = compute_tile(
                    accs,
                    b_pong,
                    _loop_end_excl,
                    lds_a_pong,
                    sx_pong,
                    sy_pong,
                    a0_prefetch=a0_pong,
                )

                rocdl.s_waitcnt(_waitcnt_imm)
                gpu.barrier()
                a0_ping = lds_a0_prefetch(lds_a_ping)

                accs = compute_tile(
                    accs,
                    b_ping,
                    _loop_end_excl + 1,
                    lds_a_ping,
                    sx_ping,
                    sy_ping,
                    a0_prefetch=a0_ping,
                )

        store_output(accs)

    # ── Kernel + host launcher ───────────────────────────────
    # The qz path threads arg_sz as a second tensor right after arg_z; the
    # bf16 path omits it. We declare both kernels statically (flydsl's
    # @flyc.kernel inspects the function signature at decoration time), then
    # wrap each in a launch helper. The bodies are identical apart from the
    # `arg_sz` argument plumbed through to `_kernel_body`.

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
        return (gx_per_h * H, D // tile_n, 1)

    if quant_output:

        @flyc.kernel
        def kernel_gemm(
            arg_z: fx.Tensor,
            arg_sz: fx.Tensor,
            arg_x: fx.Tensor,
            arg_y: fx.Tensor,
            arg_sx: fx.Tensor,
            arg_sy: fx.Tensor,
            i32_b: fx.Int32,
        ):
            _kernel_body(arg_z, arg_sz, arg_x, arg_y, arg_sx, arg_sy, i32_b)

        @flyc.jit
        def launch_gemm(
            arg_z: fx.Tensor,
            arg_sz: fx.Tensor,
            arg_x: fx.Tensor,
            arg_y: fx.Tensor,
            arg_sx: fx.Tensor,
            arg_sy: fx.Tensor,
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
                i32_b,
            )
            launcher.launch(grid=_launch_grid(i32_b), block=(256, 1, 1), stream=stream)

    else:

        @flyc.kernel
        def kernel_gemm(
            arg_z: fx.Tensor,
            arg_x: fx.Tensor,
            arg_y: fx.Tensor,
            arg_sx: fx.Tensor,
            arg_sy: fx.Tensor,
            i32_b: fx.Int32,
        ):
            _kernel_body(arg_z, None, arg_x, arg_y, arg_sx, arg_sy, i32_b)

        @flyc.jit
        def launch_gemm(
            arg_z: fx.Tensor,
            arg_x: fx.Tensor,
            arg_y: fx.Tensor,
            arg_sx: fx.Tensor,
            arg_sy: fx.Tensor,
            i32_b: fx.Int32,
            stream: fx.Stream,
        ):
            _finalize_allocators()
            kernel_gemm._func.__name__ = KERNEL_NAME
            launcher = kernel_gemm(arg_z, arg_x, arg_y, arg_sx, arg_sy, i32_b)
            launcher.launch(grid=_launch_grid(i32_b), block=(256, 1, 1), stream=stream)

    return launch_gemm


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
                       True:            (D // 128, B, H)
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
      transpose_scale: when True, scale is stored in `(D // 128, B, H)` layout
        instead of the default `(B, H, D // 128)`. Mirrors the
        `transpose_scale` option of `aiter.ops.quant.per_group_quant_hip` —
        useful when the scale will be re-consumed by a downstream kernel
        that wants the D-group axis on the outside.

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
