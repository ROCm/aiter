# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Split-K preshuffle GEMM: the ``preshuffle_gemm`` pipeline with the K loop cut into
``split_k`` independent slices, each workgroup emitting its fp32 partial tile into a
workspace. A separate reduce pass (``preshuffle_gemm_splitk_reduce``) sums the slices.

Small-M decode shapes launch far too few workgroups for the single-pass kernel
(M=4, N=2048 fills ~32 of 256 CUs); splitting K is what fills the grid.

Accumulation stays fp32 end to end — the partials are fp32, the reduce sums in fp32,
and only the reduce's final store downcasts. The per-row/per-col fp8 scales are applied
here rather than in the reduce because scaling is linear and commutes with the sum,
which lets the epilogue stay byte-for-byte the base kernel's.

``scale_mode`` picks how the fp8 scales are applied: ``"epilogue"`` (per-row x per-col),
``"blockscale"`` (arbitrary fp32 per 128-K block, dequantized by an in-loop fma), or
``"mx128"`` (E8M0 per 128-K block, fed straight to the scaled MFMA's scale operands).
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import const_expr, gpu, math, range_constexpr, rocdl
from flydsl.expr.typing import (
    BFloat16,
    Float4E2M1FN,
    Float8E4M3FN,
    Float8E4M3FNUZ,
    Float16,
    Float32,
    Int8,
    Int32,
    T,
)
from flydsl.expr.typing import Vector as Vec
from flydsl.runtime.device import get_rocm_arch

from aiter.ops.flydsl.kernels import buffer_ops, vector

from .mfma_preshuffle_pipeline import xcd_remap_bx_by

# Blockscale's in-loop dequant runs 2*acc_size scalar VALU ops per k-step as an
# elementwise mul+add. Set False to fall back to that form for an A/B.
USE_PACKED_FMA = True

# (dsrd_preload, dvmem_preload) per (tile_m, tile_n, tile_k).
_TILE_PRELOAD_TABLE = {
    # ── tile_m = 16 ──
    (16, 16, 256): (4, 8),
    (16, 16, 512): (8, 16),
    (16, 32, 256): (4, 6),
    (16, 32, 512): (8, 12),
    (16, 64, 256): (2, 2),
    (16, 64, 512): (4, 4),
    (16, 128, 256): (2, 2),
    (16, 128, 512): (2, 2),
    (16, 192, 256): (2, 2),
    (16, 256, 256): (2, 2),
    (16, 256, 512): (2, 2),
    (16, 512, 256): (2, 2),
    # ── tile_m = 32 ──
    (32, 64, 128): (6, 6),
    (32, 64, 256): (6, 6),
    (32, 64, 512): (2, 2),
    (32, 128, 128): (6, 6),
    (32, 128, 256): (6, 6),
    (32, 192, 128): (6, 6),
    (32, 192, 256): (6, 6),
    (32, 256, 128): (6, 6),
    (32, 256, 256): (6, 6),
    # ── tile_m = 48 ──
    (48, 64, 128): (8, 8),
    (48, 64, 256): (2, 2),
    (48, 128, 256): (6, 6),
    (48, 192, 256): (6, 6),
    (48, 256, 256): (6, 6),
    # ── tile_m = 64 ──
    (64, 64, 128): (4, 4),
    (64, 64, 256): (4, 4),
    (64, 128, 128): (8, 8),
    (64, 128, 256): (8, 8),
    (64, 192, 128): (8, 8),
    (64, 192, 256): (8, 8),
    (64, 256, 64): (8, 8),
    (64, 256, 128): (8, 8),
    (64, 256, 256): (8, 8),
    # ── tile_m = 80 ──
    (80, 64, 256): (4, 4),
    (80, 128, 256): (8, 8),
    (80, 192, 256): (8, 8),
    (80, 256, 256): (8, 8),
    # ── tile_m = 96 ──
    (96, 64, 128): (6, 6),
    (96, 64, 256): (6, 6),
    (96, 128, 128): (8, 8),
    (96, 128, 256): (6, 6),
    (96, 192, 128): (8, 8),
    (96, 192, 256): (8, 8),
    (96, 256, 128): (8, 8),
    (96, 256, 256): (8, 8),
    # ── tile_m = 112 ──
    (112, 64, 256): (8, 8),
    (112, 128, 256): (4, 4),
    (112, 192, 256): (8, 8),
    (112, 256, 256): (8, 8),
    # ── tile_m = 128 ──
    (128, 64, 128): (6, 6),
    (128, 64, 256): (8, 8),
    (128, 128, 64): (4, 4),
    (128, 128, 128): (8, 8),
    (128, 128, 256): (4, 4),
    (128, 192, 128): (8, 8),
    (128, 192, 256): (8, 8),
    (128, 256, 128): (6, 6),
    (128, 256, 256): (4, 4),
    # ── tile_m = 160 ──
    (160, 192, 128): (8, 8),
    # ── tile_m = 192 ──
    (192, 64, 128): (6, 6),
    (192, 128, 128): (6, 6),
    # ── tile_m = 224 ──
    (224, 64, 128): (4, 4),
    (224, 128, 128): (6, 6),
    (224, 192, 128): (6, 6),
    # ── tile_m = 256 ──
    (256, 64, 128): (4, 4),
    (256, 128, 128): (6, 6),
    (256, 192, 128): (6, 6),
    (256, 256, 128): (4, 4),
}

_TILE_PRELOAD_DEFAULT = (0, 0)


def _get_preload(tile_m, tile_n, tile_k):
    """Look up (dsrd_preload, dvmem_preload) from the tile table."""
    return _TILE_PRELOAD_TABLE.get(
        (int(tile_m), int(tile_n), int(tile_k)), _TILE_PRELOAD_DEFAULT
    )


@functools.lru_cache(maxsize=1024)
def compile_preshuffle_gemm_splitk(
    *,
    N: int,
    K: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    split_k: int,
    in_dtype: str = "fp8",
    out_dtype: str = "bf16",
    epilogue: str = "none",  # "none", "bias", "bias_relu", "bias_silu", "bias_gelu"
    waves_per_eu: int | None = None,
    enable_scheduler: bool = True,
    use_async_copy: bool = False,
    xcd_swizzle: int = 0,
    lds_stage: int = 2,
    scale_mode: str = "epilogue",  # "epilogue" (per-row x per-col), "blockscale", "mx128"
    scale_block_k: int = 128,
    use_m_bounded_store: bool = False,
    direct_out: bool = False,
    stage_a_scales: bool = False,
):
    """Compile the split-K preshuffle GEMM partial pass (fp8/int8/fp16/bf16).

    Signature: fn(workspace, A, B, scale_a, scale_b, bias, M, N, stream). ``workspace``
    is fp32, laid out as ``(split_k, ceil(M/tile_m)*tile_m, N)`` — the M extent is
    padded to the tile so a block's partial tile index is ``bid_z * gx + bid_x`` with
    no runtime pointer arithmetic. Rows past M hold garbage; the reduce ignores them.

    ``direct_out`` (split_k==1 only) replaces that partial store with a downcast store
    straight into the final ``out_dtype`` output, which makes the reduce pass — whose
    only remaining job would be a one-slice copy — unnecessary. The first argument is
    then the output tensor itself, not a workspace.
    """
    if in_dtype not in ("fp8", "int8", "fp16", "bf16", "fp4"):
        raise ValueError(f"in_dtype must be fp8/int8/fp16/bf16/fp4, got {in_dtype!r}")
    if tile_n < 16:
        raise ValueError(f"tile_n must be at least 16; got tile_n={tile_n}")
    if tile_k <= 0 or K % tile_k != 0:
        raise ValueError(
            f"tile_k must be a positive divisor of K; got tile_k={tile_k}, K={K}"
        )
    if epilogue != "none":
        # bias and the activations are not linear in the K sum, so they cannot be
        # applied per-partial; they would have to move into the reduce pass.
        raise ValueError(f"split-K supports only epilogue='none', got {epilogue!r}")
    if lds_stage not in (1, 2):
        raise ValueError(f"lds_stage must be 1 or 2, got {lds_stage}")
    if scale_mode not in ("epilogue", "blockscale", "mx128", "mxfp4"):
        raise ValueError(
            f"scale_mode must be epilogue/blockscale/mx128/mxfp4, got {scale_mode!r}"
        )
    if split_k < 1 or split_k > (K // tile_k):
        # Splits are cut on the tile boundary — B is addressed through the logical
        # (N, K) layout, so a slice starting mid-tile would straddle the preshuffle
        # permutation block. split_k need not divide the tile count (the tail is
        # padded, see `num_tiles` below), but there must be a tile per split.
        raise ValueError(
            f"split_k must be in [1, K/tile_k={K // tile_k}], got {split_k}"
        )
    if direct_out:
        if split_k != 1:
            raise ValueError(
                f"direct_out writes the final output in one pass, so it needs "
                f"split_k=1; got {split_k}"
            )
        # The output descriptor is bounded to M rows, so a ragged block's padding
        # rows are dropped there — the accumulator-order predicate is redundant.
        use_m_bounded_store = False
    _has_epilogue = epilogue != "none"
    _has_bias = epilogue in ("bias", "bias_relu", "bias_silu", "bias_gelu")
    _has_relu = epilogue == "bias_relu"
    _has_silu = epilogue == "bias_silu"
    _has_gelu = epilogue == "bias_gelu"

    is_fp8 = in_dtype == "fp8"
    is_int8 = in_dtype == "int8"
    is_f16 = in_dtype == "fp16"
    is_bf16 = in_dtype == "bf16"
    is_fp4 = in_dtype == "fp4"
    is_f16_or_bf16 = is_f16 or is_bf16
    is_8bit = is_fp8 or is_int8
    # fp4 is sub-byte, so every tile/LDS/copy extent is derived from elem_bits and
    # only collapses to whole bytes at the granularities the hardware moves (>=16B).
    elem_bits = 4 if is_fp4 else (8 if is_8bit else 16)
    elem_bytes = elem_bits // 8  # 0 for fp4 — use nbytes() for byte arithmetic

    def nbytes(n_elems):
        return n_elems * elem_bits // 8

    # The async gmem->LDS DMA (buffer_load_lds 128b) only lowers for 8-bit inputs.
    if use_async_copy and not is_8bit:
        raise ValueError("use_async_copy is only supported for 8-bit inputs (fp8/int8)")

    gpu_arch = get_rocm_arch()
    is_gfx942 = str(gpu_arch).startswith("gfx942")
    is_gfx950 = str(gpu_arch).startswith("gfx950")
    use_mfma_scale_128 = (is_fp8 or is_fp4) and is_gfx950 and (tile_k % 128 == 0)
    use_mfma_k32 = is_f16_or_bf16 and is_gfx950

    is_blockscale = scale_mode == "blockscale"
    # mx128: same [K/128, M] / [N/128, K/128] scale geometry as blockscale, but the
    # scales are E8M0 bytes, which the 16x16x128 scaled MFMA consumes natively — so
    # the dequant is the hardware scale operand rather than an in-loop fp32 fma.
    # mxfp4: 4-bit operands with a 32-K E8M0 block, so one 128-K MFMA spans four
    # scale bytes per operand — supplied by the four lane groups rather than by a
    # single broadcast byte as in mx128.
    is_mx32 = scale_mode == "mxfp4"
    is_mx = scale_mode == "mx128" or is_mx32
    if is_mx32 != is_fp4:
        raise ValueError(
            f"scale_mode='mxfp4' and in_dtype='fp4' imply each other; got "
            f"scale_mode={scale_mode!r}, in_dtype={in_dtype!r}"
        )
    # gfx942 has no 16x16x128 scaled MFMA; blockscale dequants in software regardless,
    # so cover each 128-K scale block with 4x standard 16x16x32 fp8 MFMA accumulated
    # into frag_blk (the existing K=64 tiled_mma issues two k-steps per block) and reuse
    # the same in-loop fp32 dequant. mx128/mxfp4 have no gfx942 path (they need the
    # hardware scale operand) and stay gfx950-only.
    use_blockscale_gfx942 = (
        is_fp8 and is_gfx942 and is_blockscale and (tile_k % 128 == 0)
    )
    if is_blockscale or is_mx:
        # The in-loop dequant fuses whole scale blocks per MMA k-step, so the MFMA K
        # depth must be a multiple of the scale block and each tile hold whole blocks.
        if not ((is_gfx950 and use_mfma_scale_128) or use_blockscale_gfx942):
            raise ValueError(
                f"scale_mode={scale_mode!r} requires fp8/fp4 with tile_k%128==0 on "
                f"gfx950 (16x16x128 scaled MFMA) or, for blockscale, gfx942 "
                f"(4x16x16x32 software dequant); got in_dtype={in_dtype!r}, "
                f"arch={gpu_arch}, tile_k={tile_k}"
            )
        want_block_k = 32 if is_mx32 else 128
        if scale_block_k != want_block_k:
            raise ValueError(
                f"scale_mode={scale_mode!r} supports scale_block_k={want_block_k} "
                f"only, got {scale_block_k}"
            )
        if is_mx32 and (tile_k % 256 != 0 or K % 256 != 0):
            # The shuffled E8M0 buffer is addressed in 256-K chunks (8 blocks of 32).
            raise ValueError(
                f"scale_mode='mxfp4' needs tile_k and K to be multiples of 256; "
                f"got tile_k={tile_k}, K={K}"
            )
        if tile_k % scale_block_k != 0 or K % scale_block_k != 0:
            raise ValueError(
                f"tile_k={tile_k} and K={K} must be multiples of scale_block_k={scale_block_k}"
            )
    scale_k = K // scale_block_k
    blocks_per_tile = tile_k // scale_block_k
    a_scale_tile_elems = blocks_per_tile * tile_m
    effective_stage_a_scales = (
        stage_a_scales
        and is_blockscale
        and use_async_copy
        and a_scale_tile_elems % 64 == 0
        and not use_blockscale_gfx942
    )

    if is_f16_or_bf16:
        layout_elem = Float16 if is_f16 else BFloat16
    elif is_int8:
        layout_elem = Int8
    elif is_fp4:
        layout_elem = Float4E2M1FN
    else:
        layout_elem = Float8E4M3FN if is_gfx950 else Float8E4M3FNUZ
    # Without direct_out, out_dtype is carried for the reduce pass only and this pass
    # stores fp32 partials; with it, it is this pass's own store type.
    if out_dtype not in ("bf16", "fp16"):
        raise ValueError(f"out_dtype must be bf16/fp16, got {out_dtype!r}")
    out_elem_cls = BFloat16 if out_dtype == "bf16" else Float16

    # Tile geometry (tile_K_perm = K-elements grouped per MMA k-step)
    tile_K_perm = 128 if use_mfma_scale_128 else (64 if is_8bit else 32)
    k_iters = tile_k // tile_K_perm
    # Every workgroup walks the same (ceil'd) tile count so the loop structure stays
    # compile-time uniform; when split_k does not divide the tile count the trailing
    # tiles of the last workgroup(s) fall past K and are zeroed out at load time.
    num_k_tiles = K // tile_k
    num_tiles = (num_k_tiles + split_k - 1) // split_k
    is_ragged = num_k_tiles % split_k != 0
    m_repeat = tile_m // 16
    # tile_n < 64 needs fewer waves so num_acc_n stays >= 1; clamp at 4 to keep
    # large tiles at the historical 256-thread block.
    num_waves = 4 if tile_n >= 64 else tile_n // 16
    n_per_wave = tile_n // num_waves
    num_acc_n = n_per_wave // 16
    acc_size = m_repeat * num_acc_n * 4

    total_threads = num_waves * 64
    a_load_bytes = 16
    bytes_per_thread_a = nbytes(tile_m * tile_k) // total_threads
    num_a_loads = bytes_per_thread_a // a_load_bytes
    num_b_loads = nbytes(tile_n * tile_k) // total_threads // 16
    num_ds_load = nbytes(tile_m * tile_k) // 64 // 16  # A LDS reads per wave
    num_gmem_loads = num_a_loads + num_b_loads
    if is_fp4:
        # The G2S copy gives every thread one 16B chunk of a K row, so the tile must
        # be at least total_threads*16 bytes or the thread grid spills past tile_m and
        # the trailing rows never get written. fp4 halves the tile bytes, which puts
        # otherwise-legal (tile_m, tile_k) pairs under the floor.
        thrs_m = total_threads // (tile_k // (a_load_bytes * 8 // elem_bits))
        if thrs_m > tile_m:
            raise ValueError(
                f"tile_m={tile_m}, tile_k={tile_k} is too small for fp4: the A tile is "
                f"{nbytes(tile_m * tile_k)}B but the {total_threads}-thread G2S copy "
                f"needs >= {total_threads * a_load_bytes}B (raise tile_k or tile_m)"
            )
    if (is_8bit or is_fp4) and is_gfx950:
        dsrd_preload, dvmem_preload = _get_preload(tile_m, tile_n, tile_k)
    else:
        dsrd_preload, dvmem_preload = (0, 0)

    a_lds_elems = tile_m * tile_k * (2 if is_fp4 else 1)

    @fx.struct
    class SharedStorage:
        a0: fx.Array[layout_elem, a_lds_elems, 16]
        if lds_stage == 2:
            a1: fx.Array[layout_elem, a_lds_elems, 16]
        if effective_stage_a_scales:
            scale_a0: fx.Array[Float32, a_scale_tile_elems, 16]
            if lds_stage == 2:
                scale_a1: fx.Array[Float32, a_scale_tile_elems, 16]

    # ── Kernel ────────────────────────────────────────────────────────
    @flyc.kernel
    def kernel_gemm(
        arg_c: fx.Tensor,
        arg_a: fx.Tensor,
        arg_b: fx.Tensor,
        arg_scale_a: fx.Tensor,
        arg_scale_b: fx.Tensor,
        arg_bias: fx.Tensor,
        i32_m: fx.Int32,
        i32_n: fx.Int32,
        tiled_mma_arg: fx.TiledMma,
        tiled_copy_g2s: fx.TiledCopy,
    ):
        tid = fx.thread_idx.x
        bid_x, bid_y, bid_z = fx.block_idx
        # First K tile owned by this split. `num_tiles` is already per-split.
        k_off = Int32(bid_z) * Int32(num_tiles)

        if const_expr(xcd_swizzle > 0):
            _bx, _by = xcd_remap_bx_by(
                gpu.block_id("x"),
                gpu.block_id("y"),
                fx.Index(i32_m),
                tile_m=tile_m,
                tile_n=tile_n,
                N=N,
                xcd_swizzle=xcd_swizzle,
            )
            bid_x, bid_y = Int32(_bx), Int32(_by)

        if const_expr(use_mfma_scale_128):
            _scale_atom = fx.make_mma_atom(
                fx.rocdl.cdna4.MFMA_Scale(16, 16, 128, layout_elem)
            )

            def _tile_scale_atom(atom):
                return fx.make_tiled_mma(
                    atom,
                    fx.make_layout((1, num_waves, 1), (0, 1, 0)),
                    fx.make_tile(None, None, fx.make_layout((32, 4), (1, 32))),
                )

            tiled_mma = _tile_scale_atom(_scale_atom)
        else:
            tiled_mma = tiled_mma_arg

        # Bound A (read) and C (store) to the actual M extent so blocks covering
        # rows past M (ragged M) drop their OOB loads/stores at the descriptor
        # instead of faulting / writing past the allocation. B and scale_b/bias
        # are per-N (exact multiple) and stay max_size; scale_a is per-row (M) and
        # is bounded the same way below (see its epilogue load).
        gA = fx.rocdl.make_buffer_tensor(
            arg_a,
            max_size=False,
            num_records_bytes=fx.Int64(i32_m) * fx.Int64(nbytes(K)),
        )
        # Bound B to its real extent: a padded K tile addresses past the last N block,
        # which the descriptor must drop rather than fault on.
        gB = fx.rocdl.make_buffer_tensor(
            arg_b,
            max_size=False,
            num_records_bytes=fx.Int64(N) * fx.Int64(nbytes(K)),
        )
        # Workspace rows are M padded up to whole tiles, so the OOB bound is the whole
        # allocation rather than i32_m: a ragged block writes junk into its own padding
        # rows instead of having the descriptor drop the stores.
        gx_blocks = (Int32(i32_m) + Int32(tile_m - 1)) // Int32(tile_m)
        if const_expr(direct_out):
            # Final output: M-bounded (a ragged block's padding rows fall off the
            # descriptor) and 2 bytes per element, with no split-slice dimension.
            gC = fx.rocdl.make_buffer_tensor(
                arg_c,
                max_size=False,
                num_records_bytes=fx.Int64(i32_m) * fx.Int64(N) * fx.Int64(2),
            )
        else:
            gC = fx.rocdl.make_buffer_tensor(
                arg_c,
                max_size=False,
                num_records_bytes=fx.Int64(gx_blocks)
                * fx.Int64(split_k * tile_m)
                * fx.Int64(N)
                * fx.Int64(4),
            )

        tA = fx.flat_divide(gA, fx.make_tile(tile_m, tile_k))[None, None, bid_x, None]
        tB = fx.flat_divide(gB, fx.make_tile(tile_n, tile_k))[None, None, bid_y, None]
        if const_expr(direct_out):
            tC = fx.flat_divide(gC, fx.make_tile(tile_m, tile_n))[
                None, None, Int32(bid_x), bid_y
            ]
        else:
            tC = fx.flat_divide(gC, fx.make_tile(tile_m, tile_n))[
                None, None, Int32(bid_z) * gx_blocks + Int32(bid_x), bid_y
            ]

        buf_copy = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), layout_elem)
        uni_copy = fx.make_copy_atom(fx.UniversalCopy128b(), layout_elem)

        # Per-thread slices
        thr_mma = tiled_mma.thr_slice(tid)
        thr_g2s = tiled_copy_g2s.get_slice(tid)
        thr_s2r = fx.make_tiled_copy_A(buf_copy, tiled_mma).get_slice(tid)
        thr_g2r_B = fx.make_tiled_copy_B(buf_copy, tiled_mma).get_slice(tid)

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        if const_expr(is_8bit or is_fp4):
            k_blocks16 = nbytes(tile_k) // 16
            if k_blocks16 <= 0 or (k_blocks16 & (k_blocks16 - 1)) != 0:
                raise ValueError(
                    f"Unsupported tile_k for narrow-type LDS swizzle: tile_k={tile_k}, elem_bits={elem_bits} (k_blocks16={k_blocks16}); "
                    "expected tile_k bytes to be a positive multiple of 16 with (bytes/16) a power of two."
                )
            swz_bits = k_blocks16.bit_length() - 1  # log2
            # base = log2(elements per 16B): 4 for 8-bit, 5 for fp4.
            swz_base = (128 // elem_bits).bit_length() - 1
            swz = fx.SwizzleType.get(swz_bits, swz_base, swz_bits)
        else:
            swz = fx.SwizzleType.get(3, 3, 3)

        def _make_sA(arr):
            return fx.make_view(
                arr.ptr,
                fx.make_composed_layout(
                    fx.static(swz),
                    fx.make_ordered_layout((tile_m, tile_k), (1, 0)),
                ),
            )

        if const_expr(lds_stage == 1):
            sA_stages = [_make_sA(lds.a0)]
        else:
            sA_stages = [_make_sA(lds.a0), _make_sA(lds.a1)]

        if const_expr(effective_stage_a_scales):
            scale_a_lds_arrays = [lds.scale_a0]
            scale_a_lds_stages = [
                lds.scale_a0.view(fx.make_layout(a_scale_tile_elems, 1))
            ]
            scale_a_lds_i32_stages = [
                fx.make_view(
                    fx.recast_iter(Int32, lds.scale_a0.ptr),
                    fx.make_layout(a_scale_tile_elems, 1),
                )
            ]
            if const_expr(lds_stage == 2):
                scale_a_lds_arrays.append(lds.scale_a1)
                scale_a_lds_stages.append(
                    lds.scale_a1.view(fx.make_layout(a_scale_tile_elems, 1))
                )
                scale_a_lds_i32_stages.append(
                    fx.make_view(
                        fx.recast_iter(Int32, lds.scale_a1.ptr),
                        fx.make_layout(a_scale_tile_elems, 1),
                    )
                )

        # Partitions
        pA_g = thr_g2s.partition_S(tA)
        pA_s_stages = [thr_g2s.partition_D(s) for s in sA_stages]
        pA_s2r_stages = [thr_s2r.partition_S(s) for s in sA_stages]
        pB_g = thr_g2r_B.partition_S(tB)

        # Fragments — 2 separate B fragments (split double buffer for VGPR lifetime)
        frag_copy_A = fx.make_fragment_like(pA_s_stages[0][None, None, None])
        frag_A = thr_mma.make_fragment_A(sA_stages[0])
        frag_B_single_layout = thr_mma.partition_B(tB).layout(None, None, None, 0)
        frag_B_stages = [
            fx.make_fragment_like(frag_B_single_layout, layout_elem.ir_type)
            for _ in range(2)
        ]
        frag_C = thr_mma.make_fragment_C(tC)
        frag_A_retile = thr_s2r.retile(frag_A)
        frag_B_retile_stages = [thr_g2r_B.retile(b) for b in frag_B_stages]
        # Partials are fp32: 32b stores, and the "out" fragment is accumulator-shaped.
        # direct_out downcasts instead, so its store is 16b wide.
        store_elem = out_elem_cls if direct_out else Float32
        buf_copy_out = fx.make_copy_atom(
            fx.rocdl.BufferCopy16b() if direct_out else fx.rocdl.BufferCopy32b(),
            store_elem,
        )
        thr_r2g_C = fx.make_tiled_copy_C(buf_copy_out, tiled_mma).get_slice(tid)
        pC_g = thr_r2g_C.partition_S(tC)
        frag_C_out = fx.make_fragment_like(frag_C, store_elem.ir_type)
        frag_C_retile = thr_r2g_C.retile(frag_C_out)
        if const_expr(is_blockscale):
            # Per-128-K-block raw MMA result; frag_C holds the running scaled sum.
            frag_blk = fx.make_fragment_like(frag_C, Float32.ir_type)
        if const_expr(is_mx):

            def _mx_one(sub):
                """Re-add the unit M/N and K repeat modes an indexed sub-view drops.

                The scaled atom is issued per (mi, ni) so it can carry that sub-block's
                own E8M0 words, but rmem->vector SSA promotion only matches gemm
                operands that still carry the fragment's full rank — a rank-collapsed
                ``[None, mi, k]`` view poisons the pass.
                """
                lay = sub.layout
                return fx.make_view(
                    fx.get_iter(sub),
                    fx.make_layout(
                        fx.make_shape(fx.get_shape(lay), 1, 1),
                        fx.make_stride(fx.get_stride(lay), 0, 0),
                    ),
                )

        # ── Async gmem->LDS DMA (buffer_load_lds) for the A tile ──
        if const_expr(use_async_copy):
            dma_atom = fx.make_copy_atom(fx.rocdl.BufferCopyLDS128b(), 128)
            # Bound to the real M extent (like the sync gA) so ragged-M blocks DMA-read
            # OOB rows as 0 instead of faulting past the allocation.
            gA_flat = fx.rocdl.make_buffer_tensor(
                fx.Tensor(
                    fx.make_view(fx.get_iter(arg_a), fx.make_layout(65536 * K, 1))
                ),
                max_size=False,
                num_records_bytes=fx.Int64(i32_m) * fx.Int64(K) * fx.Int64(elem_bytes),
            )
            gA_div = fx.logical_divide(gA_flat, fx.make_layout(1, 1))
            if const_expr(effective_stage_a_scales):
                scale_a_dma_atom = fx.make_copy_atom(
                    fx.rocdl.BufferCopyLDS32b(), Int32
                )
                g_scale_a = fx.rocdl.make_buffer_tensor(
                    fx.Tensor(
                        fx.make_view(
                            fx.recast_iter(Int32, fx.get_iter(arg_scale_a)),
                            fx.make_layout(65536 * scale_k, 1),
                        )
                    ),
                    max_size=False,
                    num_records_bytes=fx.Int64(scale_k)
                    * fx.Int64(i32_m)
                    * fx.Int64(4),
                )
                g_scale_a_flat = fx.make_view(
                    fx.get_iter(g_scale_a), fx.make_layout(65536 * scale_k, 1)
                )
                g_scale_a_tiles = fx.logical_divide(
                    g_scale_a_flat, fx.make_layout(1, 1)
                )
                s_scale_a_tiles = [
                    fx.logical_divide(s, fx.make_layout(1, 1))
                    for s in scale_a_lds_i32_stages
                ]
            sA_i8_ptr = [fx.recast_iter(Int8, lds.a0.ptr)]
            if const_expr(lds_stage == 2):
                sA_i8_ptr.append(fx.recast_iter(Int8, lds.a1.ptr))
            bx_m = bid_x * tile_m
            wave_id = tid // 64
            step_bytes = total_threads * a_load_bytes
            wave_stride_bytes = 64 * a_load_bytes
            k_blocks16_dma = (tile_k * elem_bytes) // 16
            elems_per_16b = 16 // elem_bytes

            def dma_a_to_lds(k_tile_val, stage):
                wave_off = rocdl.readfirstlane(
                    fx.Int32.ir_type, wave_id * wave_stride_bytes
                )
                lds_ptr = fx.add_offset(sA_i8_ptr[stage], wave_off)
                base_k = k_tile_val * tile_k
                for i in range_constexpr(num_a_loads):
                    if const_expr(i > 0):
                        lds_ptr = fx.add_offset(lds_ptr, step_bytes)
                    pos_bytes = i * total_threads * a_load_bytes + tid * a_load_bytes
                    elem_idx = pos_bytes // elem_bytes
                    m = elem_idx // tile_k
                    k = elem_idx % tile_k
                    k_swz = k ^ ((m % k_blocks16_dma) * elems_per_16b)
                    gmem_byte = ((bx_m + m) * K + base_k + k_swz) * elem_bytes
                    dst = fx.make_view(lds_ptr, fx.make_layout(1, 1))
                    src = fx.slice(gA_div, (None, fx.Int32(gmem_byte)))
                    fx.copy(dma_atom, src, dst)

            def dma_scale_a_to_lds(k_tile_val, stage):
                if const_expr(effective_stage_a_scales):
                    global_kb0 = k_tile_val * Int32(blocks_per_tile)
                    lane = tid % Int32(64)
                    for it in range_constexpr(a_scale_tile_elems // 64):
                        slot = Int32(it * 64) + lane
                        sb = slot // Int32(tile_m)
                        row = slot % Int32(tile_m)
                        src_idx = (global_kb0 + sb) * Int32(i32_m) + bid_x * Int32(
                            tile_m
                        ) + row
                        fx.copy(
                            scale_a_dma_atom,
                            fx.slice(g_scale_a_tiles, (None, src_idx)),
                            fx.slice(
                                s_scale_a_tiles[stage], (None, Int32(it * 64))
                            ),
                        )

        # ── Scheduling hints ───────────
        def build_scheduler(numer: int, denom: int):
            if const_expr(denom <= 0):
                return []
            if const_expr(numer <= 0):
                return [0] * denom
            out = []
            prev = 0
            for i in range_constexpr(denom):
                cur = ((i + 1) * numer + (denom - 1)) // denom
                out.append(cur - prev)
                prev = cur
            return out

        def hot_loop_scheduler():
            mfma_group = num_acc_n

            if const_expr(is_gfx942):
                mfma_total = (k_iters * 2) * m_repeat * mfma_group
                mfma_per_iter = 2 * mfma_group
                sche_iters = 0 if mfma_per_iter == 0 else (mfma_total // mfma_per_iter)

                rocdl.sched_dsrd(2)
                rocdl.sched_mfma(1)
                if const_expr(tile_m == 16):
                    rocdl.sched_vmem(1)
                rocdl.sched_mfma(1)
                if const_expr(tile_m == 16):
                    rocdl.sched_vmem(1)

                if const_expr(num_acc_n < 4):
                    rocdl.sched_dsrd(1)
                    rocdl.sched_mfma(1)
                    if const_expr(tile_m == 16):
                        rocdl.sched_vmem(1)
                    rocdl.sched_dsrd(1)
                    rocdl.sched_mfma(1)
                    if const_expr(tile_m == 16):
                        rocdl.sched_vmem(1)
                    rocdl.sched_mfma(1)

                dswr_tail = num_a_loads
                dstr_advance = 2
                if const_expr(dswr_tail > sche_iters):
                    dswr_tail = sche_iters
                dswr_start = max(sche_iters - dswr_tail - dstr_advance, 0)

                for sche_i in range_constexpr(sche_iters):
                    rocdl.sched_vmem(1)
                    rocdl.sched_mfma(mfma_group)
                    rocdl.sched_dsrd(1)
                    rocdl.sched_mfma(mfma_group)
                    if const_expr(sche_i >= dswr_start - 1):
                        rocdl.sched_dswr(1)
            else:
                if const_expr(use_mfma_scale_128):
                    element_k_per_mfma = 128
                else:
                    element_k_per_mfma = 32
                num_mfma_per_tile_k = tile_k // element_k_per_mfma
                mfma_total = num_mfma_per_tile_k * m_repeat * mfma_group
                dswr_tail = num_a_loads
                dstr_advance = 2
                if const_expr(dswr_tail > mfma_total):
                    dswr_tail = mfma_total
                dsrd_preload_eff = min(int(dsrd_preload), num_ds_load)
                dvmem_preload_eff = min(int(dvmem_preload), num_gmem_loads)
                vmem_remaining = num_gmem_loads - dvmem_preload_eff
                dsrd_remaining = num_ds_load - dsrd_preload_eff
                if const_expr(vmem_remaining > 0 and vmem_remaining < mfma_total):
                    vmem_schedule = build_scheduler(vmem_remaining, vmem_remaining) + [
                        0
                    ] * (mfma_total - vmem_remaining)
                else:
                    vmem_schedule = build_scheduler(vmem_remaining, mfma_total)
                dsrd_schedule = build_scheduler(dsrd_remaining, mfma_total)
                dswr_start = max(mfma_total - dswr_tail - dstr_advance, 0)
                last_dsrd_mfma_idx = -1
                for sched_idx in range_constexpr(mfma_total):
                    if const_expr(dsrd_schedule[sched_idx]):
                        last_dsrd_mfma_idx = sched_idx
                dswr_start = max(dswr_start, last_dsrd_mfma_idx + 1)
                idx_ds_read = dsrd_preload_eff
                idx_gmem_load = dvmem_preload_eff
                idx_ds_write = 0
                if const_expr(dvmem_preload_eff):
                    rocdl.sched_vmem(dvmem_preload_eff)
                if const_expr(dsrd_preload_eff):
                    rocdl.sched_dsrd(dsrd_preload_eff)
                for mfma_idx in range_constexpr(mfma_total):
                    rocdl.sched_mfma(1)
                    n_dsrd = dsrd_schedule[mfma_idx]
                    if const_expr(n_dsrd and (idx_ds_read < num_ds_load)):
                        if const_expr(idx_ds_read + n_dsrd > num_ds_load):
                            n_dsrd = num_ds_load - idx_ds_read
                        if const_expr(n_dsrd):
                            rocdl.sched_dsrd(n_dsrd)
                            idx_ds_read += n_dsrd
                    n_vmem = vmem_schedule[mfma_idx]
                    if const_expr(n_vmem and (idx_gmem_load < num_gmem_loads)):
                        if const_expr(idx_gmem_load + n_vmem > num_gmem_loads):
                            n_vmem = num_gmem_loads - idx_gmem_load
                        if const_expr(n_vmem):
                            rocdl.sched_vmem(n_vmem)
                            idx_gmem_load += n_vmem
                    if const_expr(
                        (not use_async_copy)
                        and (idx_ds_write < dswr_tail)
                        and (mfma_idx >= dswr_start)
                    ):
                        rocdl.sched_dswr(1)
                        idx_ds_write += 1
                if const_expr((not use_async_copy) and idx_ds_write < num_a_loads):
                    rocdl.sched_dswr(num_a_loads - idx_ds_write)

            rocdl.sched_barrier(0)

        # ── In-loop 128-block dequant (scale_mode="blockscale") ───
        if const_expr(is_blockscale):
            bs_bx_m = bid_x * tile_m
            bs_by_n = bid_y * tile_n
            bs_wave = gpu.thread_id("x") // 64
            bs_lane_div_16 = (gpu.thread_id("x") % 64) // 16
            # scale_a is [K/128, M] fp32 (transposed); scale_b is [N/128, K/128] fp32.
            bs_scale_a_rsrc = buffer_ops.create_buffer_resource(
                arg_scale_a,
                max_size=False,
                num_records_bytes=fx.Int64(scale_k) * fx.Int64(i32_m) * fx.Int64(4),
            )
            bs_scale_b_rsrc = buffer_ops.create_buffer_resource(
                arg_scale_b,
                max_size=False,
                num_records_bytes=fx.Int64((N // scale_block_k) * scale_k * 4),
            )

            def block_scale_vec(kb, a_stage, local_sb):
                """scale_a[kb,row] * scale_b[col/128,kb], laid out like the accumulator.

                The N-block index is lane-uniform: a wave's 16-column group starts on a
                multiple of 16, so all 16 lanes fall inside one 128-column scale block.
                """
                s_b = [
                    buffer_ops.buffer_load(
                        bs_scale_b_rsrc,
                        fx.Int32(
                            ((bs_by_n + (ni * num_waves + bs_wave) * 16) // 128)
                            * scale_k
                            + kb
                        ),
                        vec_width=1,
                        dtype=T.f32,
                    )
                    for ni in range_constexpr(num_acc_n)
                ]
                if const_expr(effective_stage_a_scales):
                    s_a = []
                    for mi in range_constexpr(m_repeat):
                        lds_off = (
                            local_sb * tile_m
                            + mi * 16
                            + bs_lane_div_16 * 4
                        )
                        ptr_off = fx.add_offset(
                            scale_a_lds_arrays[a_stage].ptr, lds_off
                        )
                        i8_iter = fx.recast_iter(fx.Uint8, ptr_off)
                        s_a.append(
                            Vec(
                                fx.make_view(i8_iter, fx.make_layout(16, 1)).load()
                            ).bitcast(Float32)
                        )
                else:
                    s_a = [
                        Vec(
                            buffer_ops.buffer_load(
                                bs_scale_a_rsrc,
                                fx.Int32(kb) * Int32(i32_m)
                                + fx.Int32(bs_bx_m + mi * 16 + bs_lane_div_16 * 4),
                                vec_width=4,
                                dtype=T.f32,
                            )
                        ).bitcast(fx.Float32)
                        for mi in range_constexpr(m_repeat)
                    ]
                elems = []
                for p in range_constexpr(acc_size):
                    ni = p // (m_repeat * 4)
                    mi = (p // 4) % m_repeat
                    ii = p % 4
                    elems.append(fx.Float32(s_a[mi][ii] * s_b[ni]))
                return vector.from_elements(T.vec(acc_size, Float32.ir_type), elems)

            def scaled_acc(acc, blk, sc):
                """acc + blk * sc over the whole accumulator.

                math.fma lowers to v_pk_fma_f32 over the full width; the flat
                index map keeps each consecutive pair inside one (mi, ni)
                sub-block and 128-K scale block, so packing is safe. Keeping
                the op compiler-visible lets GCNHazardRecognizer schedule the
                MFMA->VALU hazard instead of a fixed nop count.
                """
                if const_expr(not USE_PACKED_FMA):
                    return Vec(acc) + Vec(blk) * Vec(sc)
                return math.fma(blk, sc, acc)

        # ── In-loop E8M0 hardware scale (scale_mode="mxfp4") ──────
        if const_expr(is_mx32):
            # 32-K blocks: one 128-K MFMA needs four scale bytes per operand, and the
            # hardware sources them from the four lane groups (lane//16 = sub-block,
            # lane%16 = row/col). The buffers are the shuffle_scale_w4_cdna4 layout,
            # [G/32, K/256, KLane=4, NLane=16, KPack=2, NPack=2] bytes over the
            # 16-row/16-col group axis G, so a lane's dword is
            # ((g1*K1 + chunk)*64 + lane) and the byte within it is
            # kpack*2 + npack. opsel stays 0 — the byte is selected by address and
            # masked into bits [7:0], which keeps the atom identical to mx128's.
            mx32_lane = gpu.thread_id("x") % 64
            mx32_wave = gpu.thread_id("x") // 64
            mx32_K1 = K // 256
            mx32_a_grp = [bid_x * m_repeat + mi for mi in range_constexpr(m_repeat)]
            mx32_b_grp = [
                bid_y * (tile_n // 16) + ni * num_waves + mx32_wave
                for ni in range_constexpr(num_acc_n)
            ]
            # The shuffled A-scale groups 32 rows, so its real extent is M rounded up
            # to that group — bounding at M would clip the tail group's dwords to 0.
            mx32_sa_rsrc = buffer_ops.create_buffer_resource(
                arg_scale_a,
                max_size=False,
                num_records_bytes=fx.Int64((Int32(i32_m) + Int32(31)) // Int32(32))
                * fx.Int64(32 * scale_k),
            )
            mx32_sb_rsrc = buffer_ops.create_buffer_resource(
                arg_scale_b,
                max_size=False,
                num_records_bytes=fx.Int64(N * scale_k),
            )

            def mx32_byte(rsrc, grp, chunk, sub_byte):
                g1 = Int32(grp) // Int32(2)
                npack = Int32(grp) % Int32(2)
                dword = (g1 * Int32(mx32_K1) + chunk) * Int32(64) + Int32(mx32_lane)
                return Int32(
                    buffer_ops.buffer_load(
                        rsrc,
                        dword * Int32(4) + sub_byte + npack,
                        vec_width=1,
                        dtype=T.i8,
                    )
                ) & Int32(0xFF)

            def mx_scale_words(khs):
                """(scale_a[mi], scale_b[ni]) E8M0 bytes for global 128-K step ``khs``."""
                chunk = Int32(khs) // Int32(2)
                sub_byte = (Int32(khs) % Int32(2)) * Int32(2)  # kpack * 2
                sa = [
                    mx32_byte(mx32_sa_rsrc, mx32_a_grp[mi], chunk, sub_byte)
                    for mi in range_constexpr(m_repeat)
                ]
                sb = [
                    mx32_byte(mx32_sb_rsrc, mx32_b_grp[ni], chunk, sub_byte)
                    for ni in range_constexpr(num_acc_n)
                ]
                return sa, sb

        # ── In-loop E8M0 hardware scale (scale_mode="mx128") ──────
        elif const_expr(is_mx):
            mx_bx_m = bid_x * tile_m
            # A/B operand lane map for 16x16x128: lane holds row/col lane%16, so the
            # per-row A scale is lane-varying and the per-128-column B scale is
            # wave-uniform (one wave owns a single 16-column group).
            mx_row = gpu.thread_id("x") % 64 % 16
            mx_wave = gpu.thread_id("x") // 64
            # Same 16-column group map as block_scale_vec: wave ni's group starts at
            # (ni * num_waves + wave) * 16, and all 16 of its columns share one
            # 128-column scale block, so the index stays lane-uniform.
            mx_n_blks = [
                (bid_y * tile_n + (ni * num_waves + mx_wave) * 16) // 128
                for ni in range_constexpr(num_acc_n)
            ]
            mx_sa_rsrc = buffer_ops.create_buffer_resource(
                arg_scale_a,
                max_size=False,
                num_records_bytes=fx.Int64(scale_k) * fx.Int64(i32_m),
            )
            mx_sb_rsrc = buffer_ops.create_buffer_resource(
                arg_scale_b,
                max_size=False,
                num_records_bytes=fx.Int64((N // scale_block_k) * scale_k),
            )

            def mx_scale_words(kb):
                """Per-sub-tile (scale_a[mi], scale_b[ni]) i32 words, one E8M0 byte each.

                opsel defaults to 0 on the atom, so the byte must sit in bits [7:0];
                the mask makes the widening unsigned (exponents >= 127 are the common
                case and a sign-extended byte would corrupt them).
                """
                mask = Int32(0xFF)
                sa = [
                    Int32(
                        buffer_ops.buffer_load(
                            mx_sa_rsrc,
                            fx.Int32(kb) * Int32(i32_m)
                            + fx.Int32(mx_bx_m + mi * 16 + mx_row),
                            vec_width=1,
                            dtype=T.i8,
                        )
                    )
                    & mask
                    for mi in range_constexpr(m_repeat)
                ]
                sb = [
                    Int32(
                        buffer_ops.buffer_load(
                            mx_sb_rsrc,
                            fx.Int32(mx_n_blks[ni]) * Int32(scale_k) + fx.Int32(kb),
                            vec_width=1,
                            dtype=T.i8,
                        )
                    )
                    & mask
                    for ni in range_constexpr(num_acc_n)
                ]
                return sa, sb

        # ── B tile load, with the ragged-tail guard ───────────────
        def load_B(stage, k_global):
            """Load B tile ``k_global`` into ``stage``, zeroing it if the tile is padding.

            A padded tile is skipped by making its B operand zero rather than by
            branching around the load/MMA: the contribution is identically zero, and
            the prefetch/barrier structure of the software pipeline stays intact.
            A is left unguarded — it is read through an M-bounded descriptor, so a
            padded tile pulls either the next row's finite values or descriptor zeros,
            both of which vanish against the zeroed B. The predicate is
            workgroup-uniform (it depends only on bid_z), so no lane diverges.
            """
            fx.copy(
                buf_copy,
                pB_g[None, None, None, k_global],
                frag_B_retile_stages[stage],
            )
            if const_expr(is_ragged):
                b_i32 = Vec(frag_B_stages[stage].load()).bitcast(Int32)
                keep = (Int32(k_global) < Int32(num_k_tiles)).select(
                    Int32(-1), Int32(0)
                )
                b_i32 = b_i32 & Vec.filled(b_i32.shape, keep, Int32)
                frag_B_stages[stage].store(b_i32.bitcast(layout_elem))

        # ── Pipeline stage (double-buffered B via split fragments) ─
        def mma_kloop(a_stage, cur_frag_B, k_tile=None):
            for ki in range_constexpr(k_iters):
                fx.copy(
                    uni_copy,
                    pA_s2r_stages[a_stage][None, None, ki],
                    frag_A_retile[None, None, ki],
                )
                k_coord = ki if (use_mfma_scale_128 or use_mfma_k32) else (None, ki)
                if const_expr(use_blockscale_gfx942):
                    # No 16x16x128 MFMA on gfx942: cover one 128-K scale block with two
                    # K=64 k-steps (4x 16x16x32 fp8 MFMA) accumulated into frag_blk,
                    # then one in-loop dequant. k_iters == 2*blocks_per_tile here, so a
                    # scale block spans the even/odd ki pair (ki//2 is its block index).
                    if const_expr(ki % 2 == 0):
                        frag_blk.store(acc_zero)
                    fx.gemm(
                        tiled_mma,
                        frag_blk,
                        frag_A[None, None, k_coord],
                        cur_frag_B[None, None, k_coord],
                        frag_blk,
                    )
                    if const_expr(ki % 2 == 1):
                        sc = block_scale_vec(
                            k_tile * Int32(blocks_per_tile) + Int32(ki // 2),
                            a_stage,
                            ki // 2,
                        )
                        frag_C.store(scaled_acc(frag_C.load(), frag_blk.load(), sc))
                elif const_expr(is_blockscale):
                    # Each k-step is exactly one 128-K scale block: MMA it raw into a
                    # zeroed scratch fragment, then fma its scaled value into frag_C.
                    frag_blk.store(acc_zero)
                    fx.gemm(
                        tiled_mma,
                        frag_blk,
                        frag_A[None, None, k_coord],
                        cur_frag_B[None, None, k_coord],
                        frag_blk,
                    )
                    sc = block_scale_vec(
                        k_tile * Int32(blocks_per_tile) + Int32(ki), a_stage, ki
                    )
                    frag_C.store(scaled_acc(frag_C.load(), frag_blk.load(), sc))
                elif const_expr(is_mx):
                    # One k-step spans exactly one 128-K scale block, so a single
                    # E8M0 byte per operand covers the whole MFMA: the accumulator
                    # comes out dequantized with no epilogue fixup.
                    # Global 128-K MFMA step: mx128 has one scale block per step and
                    # mxfp4 four, so the step index (not the block index) is the key.
                    sa, sb = mx_scale_words(k_tile * Int32(k_iters) + Int32(ki))
                    # Scale words are atom state, so one gemm can only carry one (mi, ni)
                    # pair of them: issue the sub-blocks explicitly instead of letting
                    # the TiledMma expand them under a single shared scale. Slices drop
                    # the repeat modes, so each call is one 16x16x128 scaled MFMA
                    # accumulating in place on its own frag_C sub-fragment.
                    for ni in range_constexpr(num_acc_n):
                        for mi in range_constexpr(m_repeat):
                            scaled_mma = _scale_atom.set_value(
                                {"scale_a": sa[mi], "scale_b": sb[ni]}
                            )
                            acc_one = _mx_one(frag_C[None, mi, ni])
                            fx.gemm(
                                _tile_scale_atom(scaled_mma),
                                acc_one,
                                _mx_one(frag_A[None, mi, k_coord]),
                                _mx_one(cur_frag_B[None, ni, k_coord]),
                                acc_one,
                            )
                else:
                    fx.gemm(
                        tiled_mma,
                        frag_C,
                        frag_A[None, None, k_coord],
                        cur_frag_B[None, None, k_coord],
                        frag_C,
                    )

        def pipeline_2stage(
            read_stage, cur_k_val=None, next_k_val=None, read_next=True
        ):
            # next_k_val is the split-local tile index; k_off makes it global.
            # cur_k_val is the GLOBAL index of the tile resident in read_stage — needed
            # only by blockscale's in-loop dequant to pick the right 128-K scale block.
            if next_k_val is not None:
                next_k_val = k_off + next_k_val
            write_stage = read_stage ^ 1
            a_read = read_stage
            a_write = write_stage
            cur_frag_B = frag_B_stages[read_stage]
            do_next = read_next and next_k_val is not None
            if const_expr(use_async_copy):
                if const_expr(do_next):
                    dma_scale_a_to_lds(next_k_val, a_write)
                    dma_a_to_lds(next_k_val, a_write)
                    load_B(write_stage, next_k_val)
                mma_kloop(a_read, cur_frag_B, cur_k_val)
                if const_expr(enable_scheduler):
                    hot_loop_scheduler()
                if const_expr(do_next):
                    rocdl.s_waitcnt(num_b_loads)
                gpu.barrier()
                return
            if const_expr(do_next):
                fx.copy(buf_copy, pA_g[None, None, None, next_k_val], frag_copy_A)
                load_B(write_stage, next_k_val)
            mma_kloop(a_read, cur_frag_B, cur_k_val)
            if const_expr(do_next):
                fx.copy(uni_copy, frag_copy_A, pA_s_stages[a_write][None, None, None])
            if const_expr(enable_scheduler):
                hot_loop_scheduler()
            if const_expr(do_next):
                gpu.barrier()

        # ── Prologue ──────────────────────────────────────────────
        acc_zero = (
            Vec.filled(acc_size, 0, Int32)
            if const_expr(is_int8)
            else Vec.filled(acc_size, 0.0, Float32)
        )
        if const_expr(use_async_copy):
            dma_scale_a_to_lds(k_off, 0)
            dma_a_to_lds(k_off, 0)
            load_B(0, k_off)
            frag_C.store(acc_zero)
            rocdl.s_waitcnt(num_b_loads)
            gpu.barrier()
        else:
            fx.copy(buf_copy, pA_g[None, None, None, k_off], frag_copy_A)
            load_B(0, k_off)
            frag_C.store(acc_zero)
            fx.copy(uni_copy, frag_copy_A, pA_s_stages[0][None, None, None])
            gpu.barrier()
        rocdl.sched_barrier(0)

        # ── Main tile loop ────────────────────────────────────────────
        if const_expr(lds_stage == 1 and num_tiles > 1):
            frag_Bc = frag_B_stages[0]
            for iv, state in range(0, num_tiles - 1, 1, init=[frag_C.load()]):
                frag_C.store(state[0])
                k_next = k_off + fx.Int32(iv + 1)
                mma_kloop(0, frag_Bc, k_off + fx.Int32(iv))
                if const_expr(not use_async_copy):
                    fx.copy(buf_copy, pA_g[None, None, None, k_next], frag_copy_A)
                load_B(0, k_next)
                gpu.barrier()  # single buffer: all reads done before overwrite
                if const_expr(use_async_copy):
                    dma_scale_a_to_lds(k_next, 0)
                    dma_a_to_lds(k_next, 0)
                else:
                    fx.copy(uni_copy, frag_copy_A, pA_s_stages[0][None, None, None])
                if const_expr(enable_scheduler):
                    hot_loop_scheduler()
                if const_expr(use_async_copy):
                    rocdl.s_waitcnt(num_b_loads)
                gpu.barrier()
                results = yield [frag_C.load()]
            frag_C.store(results)
        elif const_expr(lds_stage == 2):
            # 2-tile/iter ping-pong: middle loop runs 2 tiles/iter
            is_odd_tiles = (num_tiles % 2) == 1
            tail = 1 if is_odd_tiles else 2
            loop_end = (num_tiles - tail) // 2

            def two_tiles(k_base):
                pipeline_2stage(
                    read_stage=0,
                    cur_k_val=k_off + k_base,
                    next_k_val=k_base + fx.Int32(1),
                )
                pipeline_2stage(
                    read_stage=1,
                    cur_k_val=k_off + k_base + fx.Int32(1),
                    next_k_val=k_base + fx.Int32(2),
                )

            if const_expr(loop_end > 0 and use_async_copy):
                for iv in range_constexpr(loop_end):
                    two_tiles(fx.Int32(iv * 2))
            elif const_expr(loop_end > 0):
                for iv, state in range(0, loop_end, 1, init=[frag_C.load()]):
                    frag_C.store(state[0])
                    two_tiles(fx.Int32(iv * 2))
                    results = yield [frag_C.load()]
                frag_C.store(results)
            k_tail0 = num_tiles - tail  # first tile handled by the peeled tail
            for j in range_constexpr(tail - 1):
                pipeline_2stage(
                    read_stage=(k_tail0 + j) % 2,
                    cur_k_val=k_off + fx.Int32(k_tail0 + j),
                    next_k_val=fx.Int32(k_tail0 + j + 1),
                )

        # ── Epilogue-operand preloads (scale_a / scale_b / bias) ─────────
        bx_m = bid_x * tile_m
        by_n = bid_y * tile_n
        wave_id = gpu.thread_id("x") // 64
        lane_id = gpu.thread_id("x") % 64
        lane_div_16 = lane_id // 16
        lane_mod_16 = lane_id % 16

        def load_epi_operands():
            s_a = s_b = bias = None
            if const_expr(is_8bit):
                # Per-row(scale_a) × per-col(scale_b) scaling, applied in the epilogue.
                scale_b_rsrc = buffer_ops.create_buffer_resource(
                    arg_scale_b, max_size=True
                )
                s_b = [
                    buffer_ops.buffer_load(
                        scale_b_rsrc,
                        fx.Int32(by_n + (ni * num_waves + wave_id) * 16 + lane_mod_16),
                        vec_width=1,
                        dtype=T.f32,
                    )
                    for ni in range_constexpr(num_acc_n)
                ]
                scale_a_rsrc = buffer_ops.create_buffer_resource(
                    arg_scale_a,
                    max_size=False,
                    num_records_bytes=fx.Int64(i32_m) * fx.Int64(4),
                )
                s_a = [
                    Vec(
                        buffer_ops.buffer_load(
                            scale_a_rsrc,
                            fx.Int32(bx_m + mi * 16 + lane_div_16 * 4),
                            vec_width=4,
                            dtype=T.f32,
                        )
                    ).bitcast(fx.Float32)
                    for mi in range_constexpr(m_repeat)
                ]
            if const_expr(_has_bias):
                # Per-column bias (out_dtype), one scalar per N-block, shared across rows.
                bias_rsrc = buffer_ops.create_buffer_resource(arg_bias, max_size=True)
                bias_elem_ty = T.bf16 if out_dtype == "bf16" else T.f16
                bias = [
                    fx.Float32(
                        buffer_ops.buffer_load(
                            bias_rsrc,
                            fx.Int32(
                                by_n + (ni * num_waves + wave_id) * 16 + lane_mod_16
                            ),
                            vec_width=1,
                            dtype=bias_elem_ty,
                        )
                    )
                    for ni in range_constexpr(num_acc_n)
                ]
            return s_a, s_b, bias

        # small enough accumulator to keep operands live over the MMA; blockscale has
        # no epilogue operands to preload (dequant already happened in the K loop).
        overlap_epi_load = acc_size <= 64 and not (is_blockscale or is_mx)
        s_a_vals = s_b_vals = bias_vals = None
        if const_expr(overlap_epi_load):
            s_a_vals, s_b_vals, bias_vals = load_epi_operands()

        # Final MMA stage — overlaps the epilogue-operand loads when issued above.
        if const_expr(lds_stage == 1):
            mma_kloop(0, frag_B_stages[0], k_off + fx.Int32(num_tiles - 1))
        else:
            pipeline_2stage(
                read_stage=(num_tiles - 1) % 2,
                cur_k_val=k_off + fx.Int32(num_tiles - 1),
                read_next=False,
            )

        # ── Epilogue ─────────────────────────────────────────────
        # Optional M-bounded store: a ragged block otherwise writes its padding rows
        # (up to tile_m-1 of them) into the workspace, which the reduce never reads.
        # The predicate is built in accumulator order and retiled alongside the data,
        # so it cannot drift from the copy atom's per-thread ordering.
        pred_C_retile = None
        if const_expr(use_m_bounded_store):
            pred_C = fx.make_fragment_like(frag_C_out, dtype=fx.Boolean)
            for p in range_constexpr(acc_size):
                mi = (p // 4) % m_repeat
                ii = p % 4
                pred_C[p] = Int32(bx_m + mi * 16 + lane_div_16 * 4 + ii) < Int32(i32_m)
            pred_C_retile = thr_r2g_C.retile(pred_C)

        if const_expr(is_blockscale or is_mx or (not is_8bit and not _has_epilogue)):
            if const_expr(direct_out):
                frag_C_out.store(Vec(frag_C.load()).to(out_elem_cls))
            elif const_expr(is_blockscale or is_mx):
                frag_C_out.store(frag_C.load())  # already fp32 and already scaled
            else:
                frag_C_out.store(Vec(frag_C.load()).to(Float32))
            fx.copy(buf_copy_out, frag_C_retile, pC_g, pred=pred_C_retile)
        else:
            if const_expr(not overlap_epi_load):
                s_a_vals, s_b_vals, bias_vals = load_epi_operands()

            def apply_activation(val_s):
                # ReLU/SiLU/GeLU : maximumf for relu; exp+rcp for silu;
                # tanh-approx gelu expanded through a non-positive exponent (no overflow).
                if const_expr(_has_relu):
                    return fx.Float32(val_s).maximumf(fx.Float32(0.0))
                if const_expr(_has_silu):
                    exp_neg = math.exp(val_s * fx.Float32(-1.0))
                    return val_s * (fx.Float32(1.0) / (fx.Float32(1.0) + exp_neg))
                if const_expr(_has_gelu):
                    half_f32 = fx.Float32(0.5)
                    one_f32 = fx.Float32(1.0)
                    zero_f32 = fx.Float32(0.0)
                    two_f32 = fx.Float32(2.0)
                    x3 = val_s * val_s * val_s
                    y = fx.Float32(0.7978845608) * (val_s + fx.Float32(0.044715) * x3)
                    abs_y = fx.Float32(y).maximumf(zero_f32 - y)
                    e_neg2abs = math.exp(fx.Float32(-2.0) * abs_y)
                    denom = one_f32 + e_neg2abs
                    numerator = (y > zero_f32).select(two_f32, two_f32 * e_neg2abs)
                    return half_f32 * val_s * (numerator * (one_f32 / denom))
                return val_s

            acc_vec = Vec(frag_C.load())
            out_elems = []
            for p in range_constexpr(acc_size):
                ni = p // (m_repeat * 4)
                mi = (p // 4) % m_repeat
                ii = p % 4
                val = acc_vec[p]
                if const_expr(is_int8):
                    val = val.to(Float32)
                if const_expr(is_8bit):
                    val_s = (val * s_a_vals[mi][ii]) * s_b_vals[ni]
                else:
                    val_s = val
                if const_expr(_has_bias):
                    val_s = val_s + bias_vals[ni]
                val_s = apply_activation(val_s)
                out_elems.append(fx.Float32(val_s))

            out_vec = vector.from_elements(T.vec(acc_size, Float32.ir_type), out_elems)
            if const_expr(direct_out):
                out_vec = Vec(out_vec).to(out_elem_cls)
            frag_C_out.store(out_vec)
            fx.copy(buf_copy_out, frag_C_retile, pC_g, pred=pred_C_retile)

    # ── Host launcher ─────────────────────────────────────────────
    @flyc.jit
    def launch_gemm(
        arg_c: fx.Tensor,
        arg_a: fx.Tensor,
        arg_b: fx.Tensor,
        arg_scale_a: fx.Tensor,
        arg_scale_b: fx.Tensor,
        arg_bias: fx.Tensor,
        i32_m: fx.Int32,
        i32_n: fx.Int32,
        stream: fx.Stream,
    ):
        CompilationContext.get_current()

        # MMA atom — layout_elem carries the dtype (Float16/BFloat16/Float8E4M3FN/etc)
        if const_expr(use_mfma_k32):
            mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 32, layout_elem))
            k_perm = fx.make_layout((8, 4), (1, 8))
        elif const_expr(is_f16_or_bf16):
            mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 16, layout_elem))
            k_perm = fx.make_layout((4, 4, 2), (1, 8, 4))
        elif const_expr(is_int8):
            mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 32, layout_elem, Int32))
            k_perm = fx.make_layout((8, 4, 2), (1, 16, 8))
        elif const_expr(is_fp4):
            # fp4 has no narrow MFMA: the scaled 16x16x128 atom is the only one, and
            # it is what the kernel rebuilds in-kernel with the E8M0 operands bound.
            mma_atom = fx.make_mma_atom(
                fx.rocdl.cdna4.MFMA_Scale(16, 16, 128, layout_elem)
            )
            k_perm = fx.make_layout((32, 4), (1, 32))
        else:
            # fp8: narrow atom here; the scale (16x16x128) tiled_mma is rebuilt in-kernel
            mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 32, layout_elem))
            k_perm = fx.make_layout((8, 4, 2), (1, 16, 8))

        tiled_mma = fx.make_tiled_mma(
            mma_atom,
            fx.make_layout((1, num_waves, 1), (0, 1, 0)),
            fx.make_tile(None, None, k_perm),
        )

        # G2S tiled copy
        val_per_thr = a_load_bytes * 8 // elem_bits
        thrs_k = tile_k // val_per_thr
        thrs_m = total_threads // thrs_k
        tiled_copy_g2s = fx.make_tiled_copy(
            fx.make_copy_atom(fx.UniversalCopy128b(), layout_elem),
            fx.make_layout(
                ((thrs_k, thrs_m), (1, val_per_thr)),
                ((thrs_m * val_per_thr, 1), (1, thrs_m)),
            ),
            fx.make_tile(thrs_m, tile_k),
        )

        # fp4 arrives as packed bytes; retype the iterators so the layout algebra
        # counts 4-bit elements (every extent below is in elements, not bytes).
        if const_expr(is_fp4):
            a_iter = fx.recast_iter(layout_elem, fx.get_iter(arg_a))
            b_iter = fx.recast_iter(layout_elem, fx.get_iter(arg_b))
        else:
            a_iter = fx.get_iter(arg_a)
            b_iter = fx.get_iter(arg_b)

        # Preshuffle B layout (2D hierarchical)
        kp_bytes = 16
        kp_elems = kp_bytes * 8 // elem_bits
        k_bytes_b = nbytes(K)
        n0 = N // 16
        k0 = k_bytes_b // 64
        s_nlane = kp_elems
        s_klane = 16 * s_nlane
        s_k0 = 4 * s_klane
        s_n0 = k0 * s_k0
        preshuffle_B = fx.Tensor(
            fx.make_view(
                b_iter,
                fx.make_layout(
                    ((16, n0), (kp_elems, 4, k0)), ((s_nlane, s_n0), (1, s_klane, s_k0))
                ),
            )
        )

        # Reshape A and C to 2D
        M_max = 65536
        arg_a_2d = fx.Tensor(fx.make_view(a_iter, fx.make_layout((M_max, K), (K, 1))))
        arg_c_2d = fx.Tensor(
            fx.make_view(fx.get_iter(arg_c), fx.make_layout((M_max, N), (N, 1)))
        )

        gx = (i32_m + (tile_m - 1)) // tile_m
        gy = i32_n // tile_n

        kernel_gemm(
            arg_c_2d,
            arg_a_2d,
            preshuffle_B,
            arg_scale_a,
            arg_scale_b,
            arg_bias,
            i32_m,
            i32_n,
            tiled_mma,
            tiled_copy_g2s,
            value_attrs={"rocdl.waves_per_eu": waves_per_eu},
        ).launch(
            grid=(gx, gy, split_k),
            block=(total_threads, 1, 1),
            stream=stream,
        )

    if const_expr(is_f16_or_bf16 and num_acc_n <= 2):
        launch_gemm.compile_hints["llvm_options"] = {"enable-post-misched": False}

    return launch_gemm
