"""FP8 blockscaled GEMM: C(M,N) = A(M,K) @ B(N,K).T with DeepGEMM packed-UE8M0 scales.

Preshuffled B layout (N//16, K//64, 4, 16, 16) for gfx950 / MI355X.

Public entry points:
  compile_preshuffle_gemm_blockscaled(M, N, K, tile_m, tile_n, tile_k, ...)
      Underlying factory. Returns a launcher with ABI
          launch(arg_c, arg_a, arg_b, arg_sa, arg_sb, i32_m, stream)
      where:
        arg_c : bf16 (M, N)
        arg_a : fp8  e4m3 (M, K)                — caller pre-quantizes
        arg_b : fp8  e4m3 (N//16, K//64, 4, 16, 16) preshuffled
        arg_sa: int32 (M, K//512)               — packed UE8M0, 1×128 along K
        arg_sb: int32 (N//128, K//512)          — packed UE8M0, 128×128 blocks
        i32_m : runtime M (used for OOB protection in scale stores)

  compile_preshuffle_gemm_blockscaled_auto(M, N, K, ...)
      Autotune-dispatched. Looks up best tiles in _AUTOTUNE_WINNERS; falls back
      to defaults on miss.

This kernel is a port of fp8_einsum.py (which hits ~1896 TF peak on gfx950).
The port:
  - drops the head dim H (this is a 2D GEMM, not an einsum-per-head)
  - renames B→M, D→N, R→K
  - drops the qz epilogue (bf16 output only)
  - drops B reg-pass and replaces with a preshuffled-tile loader (the einsum's
    B preshuffle is per-head; ours is whole-tensor so the n0 stride matches).
"""

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir as mlir_ir
from flydsl._mlir.dialects import scf as scf_dialect
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
# Key: (M, N, K) → (tile_m, tile_n, tile_k, block_swizzle_n)
# Missing shapes fall back to defaults in compile_preshuffle_gemm_blockscaled_auto.
#
# Tuned on MI355X (gfx950, 2500 TF FP8 peak).
# Methodology: full grid over tile_m ∈ {64,128,256}, tile_n ∈ {128,256},
# tile_k ∈ {128,256,512,1024}, bsw ∈ {0,1,2,3,4,6,8,12,16,24,32,48,64,96}
# (~336 configs). Robust timing: min over 5 repeats of 60 iters each with
# 15-iter warmup, HW-event timing. The robust harness disambiguates ~3%
# noise that the older single-shot n_iters=40 harness left in the rankings.
# Entries are (tile_m, tile_n, tile_k, block_swizzle_n) or
# (tile_m, tile_n, tile_k, block_swizzle_n, mfma_batch). The 4-tuple form
# defaults mfma_batch to 2 (the universal-best for the fp32_post_mfma path);
# add a 5th element to override per-shape (e.g. tile_m=64 shapes like 6).
_AUTOTUNE_WINNERS: dict[tuple[int, int, int], tuple] = {
    # (M,     N,     K   ) -> (tm,  tn,  tk,  bsw[, mfma_batch])
    (32768, 12288, 2048): (128, 128, 128, 2),         # 0.794 ms  -> 2077 TF  (83.1%)
    (32768,  4096, 2048): (128, 128, 128, 8),         # 0.250 ms  -> 2195 TF  (87.8%)
}


def _autotune_lookup(M: int, N: int, K: int):
    """Find the best config tuple for the given (M, N, K).

    Returns a 4-tuple (tm, tn, tk, bsw) or 5-tuple (..., mfma_batch), or None
    on miss (caller uses defaults).
    """
    if (M, N, K) in _AUTOTUNE_WINNERS:
        return _AUTOTUNE_WINNERS[(M, N, K)]
    return None


def compile_preshuffle_gemm_blockscaled(
    *,
    M: int,
    N: int,
    K: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    block_swizzle_n: int = 0,
    scale_format: str = "ue8m0",
    scale_layout: str = "row",
    mfma_batch: int = 2,
):
    """Compile the bf16-output fp8 blockscaled GEMM.

    Args:
      M, N, K: GEMM shape (M is compile-time for the autotune lookup; the
        kernel ALSO consumes a runtime M via `i32_m`).
      tile_m/n/k: tile sizes. tile_k must be a multiple of 128. tile_n must
        be a multiple of 128, and >= 128.
      block_swizzle_n: L2 supergroup swizzle (0 disables).
      scale_format: "ue8m0" (default), "fp32", or "fp32_post_mfma".
        - "ue8m0": arg_sa/arg_sb are i32 tensors of pre-packed UE8M0 bytes
            (4 bytes per i32 along K). Shapes (row): arg_sa (M, K//512),
            arg_sb (N//128, K//512). Scale fed to MFMA via opsel byte.
            Fastest path; caller responsible for packing.
        - "fp32": arg_sa/arg_sb are fp32 tensors. Shapes (row):
            arg_sa (M, K//128), arg_sb (N//128, K//128). Each fp32 is
            ROUNDED to nearest power-of-2 (UE8M0) in the load-to-LDS
            prologue, then fed to MFMA as opsel byte. Hot path bit-
            identical to "ue8m0". Numerical accuracy LOSSY vs CK because
            scales are quantized to powers of 2 (~10% median rel error
            on random fp32 scales). Use only when scales are already
            pow-2 (MXFP8/DeepGEMM-style checkpoints).
        - "fp32_post_mfma": arg_sa/arg_sb are fp32 tensors. Shapes (row):
            arg_sa (M, K//128), arg_sb (N//128, K//128). True FP32 block-
            scale GEMM: MFMA runs UNSCALED (scale operand fixed to 1.0
            = UE8M0 byte 127), then per K-128 partial sum is multiplied
            by `sa[m,g] * sb[n,g]` (fp32 ×, fp32 +) into the outer acc.
            Numerically equivalent to CK/cuBLAS-Hopper FP8 GEMM. Slower
            than "ue8m0"/"fp32" by ~5-15% (extra fp32 muladd per group).
            Required when caller's scales are arbitrary fp32 values
            (the existing CK blockscale tuner's data flow).
      scale_layout: "row" (default, M-major) or "col" (K-major).
        Affects only arg_sa's gmem layout (arg_sb is small and always row-major).
        - "row": arg_sa shape (M, K//512) ue8m0 or (M, K//128) fp32; the
            standard layout where rows of M are contiguous along K.
        - "col": arg_sa shape (K//512, M) ue8m0 or (K//128, M) fp32; the
            production layout used by ATOM when preshuffleB=True (per
            `aiter.ops.quant.per_token_quant_hip(transpose_scale=True)`).
            Loaders use a different thread distribution so col-major gmem
            access still coalesces. Same LDS slab layout regardless;
            prefetch_sa_tile/compute_tile unchanged.
      mfma_batch: software-pipeline depth for the scaled-K128 MFMA + post-MFMA
        FMA chain (fp32_post_mfma path only; harmless no-op otherwise). The
        compute loop issues `mfma_batch` independent MFMAs into distinct scratch
        accumulators, then drains their FMAs, repeating. A larger window hides
        more of the 33.5-cyc MFMA latency (latency/throughput = 8.37× on
        gfx950) but holds more scratch VGPRs. Default 2 (universal-best across
        shapes); tile_m=64 shapes can prefer ~6. Clamped to m_repeat×num_acc_n.

    Returns: launcher with signature
      launch(arg_c, arg_a, arg_b, arg_sa, arg_sb, i32_m, stream)
    """
    if scale_format not in ("ue8m0", "fp32", "fp32_post_mfma"):
        raise ValueError(
            f"scale_format must be 'ue8m0', 'fp32', or 'fp32_post_mfma', "
            f"got {scale_format!r}"
        )
    # post-MFMA path always reads fp32 scales but rejects ue8m0 input ABI
    # (the LDS slab is fp32-typed; cannot reuse the i32-packed loader path).
    if scale_format == "fp32_post_mfma" and scale_layout == "col":
        # col-major fp32 path requires 4 scalar loads per slot (no vec-4) +
        # we'd need to extend that to the per-K-128 fp32 LDS slab. Skip v1.
        raise ValueError(
            "scale_format='fp32_post_mfma' currently supports scale_layout='row' "
            "only; col-major support requires additional gmem-stride work."
        )
    if scale_layout not in ("row", "col"):
        raise ValueError(
            f"scale_layout must be 'row' or 'col', got {scale_layout!r}"
        )
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
    if K % tile_k != 0:
        raise ValueError(f"K={K} must be divisible by tile_k={tile_k}")
    if N % tile_n != 0:
        raise ValueError(f"N={N} must be divisible by tile_n={tile_n}")
    if K % 512 != 0:
        raise ValueError(
            f"K={K} must be divisible by 512 (packed UE8M0 needs 4 K-128 "
            f"blocks per i32)."
        )
    if mfma_batch < 1:
        raise ValueError(f"mfma_batch must be >= 1, got {mfma_batch}")

    gpu_arch = get_hip_arch()
    if not str(gpu_arch).startswith("gfx95"):
        raise RuntimeError(
            f"preshuffle_gemm_blockscaled targets gfx950 only, got {gpu_arch}"
        )

    elem_bytes_a = 1  # fp8
    elem_bytes_b = 1  # fp8

    KERNEL_NAME = (
        f"preshuffle_gemm_blockscaled"
        f"_M{M}_N{N}_K{K}"
        f"_t{tile_m}x{tile_n}x{tile_k}"
    )
    if block_swizzle_n > 0:
        KERNEL_NAME += f"_bsw{block_swizzle_n}"
        _gy_static = N // tile_n
        if _gy_static % block_swizzle_n != 0:
            raise ValueError(
                f"block_swizzle_n={block_swizzle_n} must divide gy="
                f"N/tile_n={_gy_static}."
            )
    # mfma_batch only affects the fp32_post_mfma compute loop; include it in
    # the kernel name (when non-default) so distinct values get distinct
    # compiled kernels / cache entries / ASM dumps.
    if scale_format == "fp32_post_mfma" and mfma_batch != 2:
        KERNEL_NAME += f"_mb{mfma_batch}"

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
    lds_stage = 2

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

    # B fp8 not in LDS (kept in registers via load_b_tile_preshuffled).
    lds_b_fp8_bytes_per_stage = 0
    lds_b_fp8_alloc_offsets, lds_b_fp8_allocs = [], []

    n_blocks_per_tile = tile_n // 128

    # sb is loaded to registers (WG-uniform; tiny). Entry granularity depends
    # on mode:
    #   - ue8m0/fp32:    1 packed-i32 per K-512 region (4 K-128 groups packed)
    #   - fp32_post_mfma: 1 fp32 per K-128 group
    # Layout in regs (Python list, constexpr-indexed at use site):
    #   sb_reg_table[nb_idx * _sb_lds_per_nb + k_idx]
    if scale_format == "fp32_post_mfma":
        _sb_lds_per_nb = K // 128              # fp32 entries per N-128-block
    else:
        _sb_lds_per_nb = K // 512              # packed-i32 entries per N-128-block
    _sb_lds_count = n_blocks_per_tile * _sb_lds_per_nb

    # sa LDS slab — persistent per WG, loaded once in prologue.
    # In ue8m0/fp32 modes the slab holds packed-i32 (4 bytes per K-512 region).
    # In fp32_post_mfma mode the slab holds the PRE-MULTIPLIED product
    # sxsy[n_block, k_g, row] = sa[row, k_g] × sb[n_block, k_g] (fp32). Each
    # wave only READS its own n_block slice in the K-loop (n_block is wave-
    # uniform → wave_id × n_per_wave // 128). The K-loop's ds_read count is
    # identical to the un-multiplied baseline (1 ds_read per (mi, g) per wave),
    # but the per-K-tile sxsy_vecs FMA precompute is eliminated and the
    # sxsy_vecs register holding (num_acc_n × 4 VGPRs/lane) is freed.
    #   - ue8m0/fp32:    slot(row, k_pkt) = row*(K/512) + k_pkt; bytes = tm*(K/512)*4
    #   - fp32_post_mfma slot(n_block, k_g, row) = n_block*(K/128)*tm
    #                                              + k_g*tm + row
    #                    bytes = n_blocks_per_tile * tm * (K/128) * 4
    if scale_format == "fp32_post_mfma":
        _sa_lds_per_row = K // 128                # fp32 entries per row
        _sa_lds_count = n_blocks_per_tile * tile_m * _sa_lds_per_row
    else:
        _sa_lds_per_row = K // 512                # packed-i32 entries per row
        _sa_lds_count = tile_m * _sa_lds_per_row
    _sa_lds_bytes = _sa_lds_count * 4              # i32 or fp32: both 4 B/entry
    if _sa_lds_count % total_threads != 0:
        raise ValueError(
            f"sa LDS load distribution requires tile_m * sa_per_row divisible "
            f"by {total_threads}, but tile_m={tile_m} × per_row={_sa_lds_per_row} "
            f"= {_sa_lds_count}, remainder {_sa_lds_count % total_threads}. "
            f"Increase K or tile_m so the product is a multiple of {total_threads}."
        )
    sa_lds_offset = allocator_pong._align(allocator_pong.ptr, 16)
    allocator_pong.ptr = sa_lds_offset + max(16, _sa_lds_bytes)

    # CDNA4 / gfx950 (MI355X): 160 KB LDS per CU.
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

    # B preshuffle layout: (N//16, K//64, 4, 16, 16)
    # Stride decomposition (in fp8 bytes):
    #   stride_nlane = 16   (innermost 16 = kpack)
    #   stride_klane = 16 * 16 = 256
    #   stride_k0    = 4 * stride_klane = 1024
    #   stride_n0    = (K/64) * stride_k0
    k_bytes_b_total = K * elem_bytes_b
    n0_val = N // 16
    k0_val = k_bytes_b_total // 64
    kpack_elems = 16
    _stride_nlane = kpack_elems
    _stride_klane = 16 * _stride_nlane
    _stride_k0 = 4 * _stride_klane
    _stride_n0 = k0_val * _stride_k0

    # K=128 quant group constants
    k_per_quant_group = 128
    mfmas_per_group = k_per_quant_group // 32  # = 4
    groups_per_tile = tile_k // k_per_quant_group

    # sa packed: (M, K//512) i32  — column-stride K/512 i32s per row
    # sb packed: (N//128, K//512) i32
    _sb_per_n128 = K // 512

    def _kernel_body(
        arg_c, arg_a, arg_b, arg_sa, arg_sb, i32_m,
    ):
        Vec = fx.Vector
        fp8_dtype = fx.Float8E4M3FN

        c_m = fx.Index(i32_m)

        tx = gpu.thread_id("x")
        bx_raw = gpu.block_id("x")
        by_raw = gpu.block_id("y")

        # Single GEMM (no head dim). grid_x covers M-tiles, grid_y covers N-tiles.
        gx = (c_m + (tile_m - 1)) // tile_m

        if const_expr(block_swizzle_n > 0):
            _c_m_eff = gx * fx.Index(tile_m)
            bx, by = xcd_remap_bx_by(
                bx_raw, by_raw, _c_m_eff,
                tile_m=tile_m, tile_n=tile_n, N=N,
                xcd_swizzle=block_swizzle_n,
            )
        else:
            bx = bx_raw
            by = by_raw

        bx_m = bx * tile_m
        by_n = by * tile_n

        # ── LDS pointers ───────────────────────────────────────────────
        base_ptr_pong = allocator_pong.get_base()
        base_ptr_ping = allocator_ping.get_base()
        lds_a_fp8_stages = []
        for _off, _alloc in zip(lds_a_fp8_alloc_offsets, lds_a_fp8_allocs):
            _base = base_ptr_pong if _alloc is allocator_pong else base_ptr_ping
            _ptr = SmemPtr(
                _base, _off, fp8_dtype.ir_type,
                shape=(lds_a_fp8_bytes_per_stage,),
            )
            lds_a_fp8_stages.append(_ptr.get())
        lds_a_pong = lds_a_fp8_stages[0]
        lds_a_ping = lds_a_fp8_stages[1]

        # sb lives in regs, not LDS — see load_sb_to_regs() / prefetch_sb_tile() below.

        # sa LDS slab — i32 in ue8m0/fp32 modes (packed UE8M0 bytes),
        # fp32 in fp32_post_mfma mode (one fp32 per K-128 group per row).
        if scale_format == "fp32_post_mfma":
            _sa_lds_ir_type = fx.Float32.ir_type
        else:
            _sa_lds_ir_type = fx.Int32.ir_type
        _sa_lds_ptr = SmemPtr(
            base_ptr_pong, sa_lds_offset, _sa_lds_ir_type,
            shape=(max(4, _sa_lds_count),),
        )
        lds_sa = _sa_lds_ptr.get()

        # ── Buffer resources ──────────────────────────────────────────
        # A: M*K fp8 bytes (record-count in bytes for fp8 = elem count).
        _a_nrec = fx.Int64(c_m * fx.Index(K))
        a_rsrc = buffer_ops.create_buffer_resource(
            arg_a, max_size=False, num_records_bytes=_a_nrec,
        )
        # B: preshuffled total bytes = N*K fp8 bytes. Use max_size for simplicity
        # (whole tensor; OOB protection irrelevant since B is compile-time shape).
        b_rsrc = buffer_ops.create_buffer_resource(arg_b, max_size=True)

        # SA: bytes-per-row depends on scale format.
        #   ue8m0: (K/512) i32 per row → (K/512)*4 bytes per row.
        #   fp32:  (K/128) fp32 per row → (K/128)*4 bytes per row.
        if scale_format == "ue8m0":
            _sa_row_bytes = (K // 512) * 4
        else:  # "fp32"
            _sa_row_bytes = (K // 128) * 4
        _sa_nrec = fx.Int64(c_m * fx.Index(_sa_row_bytes))
        sa_rsrc = buffer_ops.create_buffer_resource(
            arg_sa, max_size=False, num_records_bytes=_sa_nrec,
        )
        # SB: whole-tensor bound is fine; N is compile-time.
        sb_rsrc = buffer_ops.create_buffer_resource(arg_sb, max_size=True)

        # C: M * N bf16 (2 bytes).
        _c_nrec = fx.Int64(c_m * fx.Index(N * 2))
        c_rsrc = buffer_ops.create_buffer_resource(
            arg_c, max_size=False, num_records_bytes=_c_nrec,
        )

        # ── Wave/lane decomposition ──────────────────────────────────
        layout_wave_lane = fx.make_layout((4, wave_size), (64, 1))
        coord_wave_lane = fx.idx2crd(tx, layout_wave_lane)
        wave_id = fx.get(coord_wave_lane, 0)
        lane_id = fx.get(coord_wave_lane, 1)

        layout_lane16 = fx.make_layout((4, 16), (16, 1))
        coord_lane16 = fx.idx2crd(lane_id, layout_lane16)
        lane_div_16 = fx.get(coord_lane16, 0)
        lane_mod_16 = fx.get(coord_lane16, 1)

        n_tile_base = wave_id * n_per_wave
        n_blk_list = []
        n_intra_list = []
        for i in range_constexpr(num_acc_n):
            global_n_in_tensor = by_n + n_tile_base + (i * 16) + lane_mod_16
            n_blk_list.append(global_n_in_tensor // 16)
            n_intra_list.append(global_n_in_tensor % 16)

        _b_stride_n0_c = fx.Index(_stride_n0)
        _b_stride_k0_c = fx.Index(_stride_k0)
        _b_stride_klane_c = fx.Index(_stride_klane)
        _b_stride_nlane_c = fx.Index(_stride_nlane)

        # ── A async copy: gmem → LDS direct (swizzle-the-source) ──────
        tile_k_dwords = tile_k // 4
        c4 = fx.Index(4)
        tx_i32_base = tx * c4
        layout_a_tile_div4 = fx.make_layout(
            (tile_m, tile_k_dwords), (tile_k_dwords, 1)
        )

        def a_tile_chunk_coord(i):
            return tile_chunk_coord_i32(
                fx.arith,
                tx_i32_base=tx_i32_base,
                i=i,
                total_threads=total_threads,
                layout_tile_div4=layout_a_tile_div4,
            )

        _lds_a_fp8_k_dim_c = fx.Index(lds_a_fp8_stride_bytes)
        _lds_a_fp8_k_blocks16_c = fx.Index(lds_a_fp8_k_blocks16)

        _a_async_load_bytes = 16
        _a_wave_bytes_per_chunk = wave_size * _a_async_load_bytes
        _a_chunk_stride_bytes = total_threads * _a_async_load_bytes

        # Stride for A row: K elements/row × 1 byte/elem = K bytes/row.
        stride_a_row_bytes = K  # bytes per row of A in gmem

        def load_a_tile_to_lds_async(base_k_elem, lds_buffer):
            """Async DMA A gmem → LDS (swizzle-the-source)."""
            from flydsl._mlir.dialects import memref as memref_dialect

            wave_byte_off = rocdl.readfirstlane(
                fx.Int64.ir_type,
                fx.Int64(wave_id * fx.Index(_a_wave_bytes_per_chunk)),
            )
            lds_base = memref_dialect.extract_aligned_pointer_as_index(
                lds_buffer
            )
            lds_ptr_base = buffer_ops.create_llvm_ptr(
                fx.Int64(lds_base), address_space=3,
            )
            lds_ptr_wave = buffer_ops.get_element_ptr(
                lds_ptr_base, wave_byte_off,
            )

            for i in range_constexpr(num_a_loads):
                row_a_local, col_dword_local = a_tile_chunk_coord(i)
                col_byte_local = col_dword_local * c4
                col_byte_swz = swizzle_xor16(
                    row_a_local, col_byte_local, _lds_a_fp8_k_blocks16_c,
                )
                row_a_global = bx_m + row_a_local
                # No head dim: gmem byte index = row*K + base_k + col
                idx_elem = (
                    row_a_global * fx.Index(stride_a_row_bytes)
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
                    a_rsrc, lds_dst,
                    fx.Int32(_a_async_load_bytes),
                    global_offset_i32,
                    fx.Int32(0),
                    fx.Int32(0),
                    fx.Int32(1),
                )

        num_b_loads_per_lane = num_acc_n * (k_unroll // 2)

        # ── B global load (preshuffled, whole-tensor) — gmem→regs path ─────
        # B layout (fp8 bytes): (N//16, K//64, 4, 16, 16). Same per-lane chunk
        # math as the einsum kernel; only the head_byte_off is removed.
        def load_b_tile_preshuffled(base_k_elem):
            """Load (tile_n, tile_k) fp8 B for current N-tile.
            Returns packs_flat[ku=0..k_unroll-1] of per-ni i64 lists.
            """
            k0_base = base_k_elem // 64
            packs_flat = [[] for _ in range(k_unroll)]
            for ni in range_constexpr(num_acc_n):
                n_base_byte = (
                    n_blk_list[ni] * _b_stride_n0_c
                    + lane_div_16 * _b_stride_klane_c
                    + n_intra_list[ni] * _b_stride_nlane_c
                )
                for ku64 in range_constexpr(k_unroll // 2):
                    k0 = k0_base + ku64
                    idx_byte = n_base_byte + k0 * _b_stride_k0_c
                    idx_dword = idx_byte // 4
                    b_vec4 = buffer_ops.buffer_load(
                        b_rsrc, fx.Int32(idx_dword),
                        vec_width=4, dtype=fx.Int32,
                    )
                    b16 = Vec(b_vec4).bitcast(fp8_dtype)
                    b_i64x2 = Vec(b16).bitcast(fx.Int64)
                    packs_flat[ku64 * 2].append(b_i64x2[0].ir_value())
                    packs_flat[ku64 * 2 + 1].append(b_i64x2[1].ir_value())
            return packs_flat

        # ── LDS A read primitive ───────────────────────────────────
        def lds_load_a_packs_k64(row, col_base_bytes, lds_buffer):
            col_swz = swizzle_xor16(row, col_base_bytes, _lds_a_fp8_k_blocks16_c)
            idx = row * _lds_a_fp8_k_dim_c + col_swz
            v16 = Vec.load(
                Vec.make_type(16, fp8_dtype), lds_buffer, [idx]
            )
            v2 = Vec(v16).bitcast(fx.Int64)
            return v2[0].ir_value(), v2[1].ir_value()

        # ── sa load-once-per-WG ──────────────────────────────────
        _sa_slots_per_thread = _sa_lds_count // total_threads

        # fp32 → UE8M0 byte conversion (matches host-side _fp32_to_ue8m0_byte).
        # exp_field = ((bits + 0x400000) & 0xFF800000) >> 23   ∈ [0, 255]
        _c_round_bias = fx.Int32(0x400000)
        _c_exp_mask = fx.Int32(0xFF800000)
        _c_exp_shift = fx.Int32(23)
        _c_byte_mask = fx.Int32(0xFF)
        _c_shift_8 = fx.Int32(8)
        _c_shift_16 = fx.Int32(16)
        _c_shift_24 = fx.Int32(24)

        def _fp32_to_ue8m0_byte_i32(fp32_val):
            """Returns an i32 in [0, 255] — the UE8M0 exponent byte.

            Accepts either a wrapper with .ir_value() (e.g. from Vec[i]) or
            a raw ArithValue (e.g. from buffer_load(vec_width=1)).
            """
            ir = fp32_val.ir_value() if hasattr(fp32_val, "ir_value") else fp32_val
            bits = fx.Int32(fx.arith.bitcast(fx.Int32.ir_type, ir))
            rounded = (bits + _c_round_bias) & _c_exp_mask
            return (rounded >> _c_exp_shift) & _c_byte_mask

        def _pack_4_bytes_to_i32(b0, b1, b2, b3):
            """Little-endian pack: byte0 in low 8 bits, byte3 in high 8 bits."""
            return (
                (b0 & _c_byte_mask)
                | ((b1 & _c_byte_mask) << _c_shift_8)
                | ((b2 & _c_byte_mask) << _c_shift_16)
                | ((b3 & _c_byte_mask) << _c_shift_24)
            )

        def load_sa_to_lds(sb_regs=None):
            """Issue all sa loads + store into LDS. One-time per WG.

            Three scale formats × two scale layouts:
              - ue8m0: gmem i32 (M, K/512) packed UE8M0 → LDS i32, 1 dword/slot
              - fp32:  gmem fp32 (M, K/128) → in-kernel UE8M0 byte → LDS i32
              - fp32_post_mfma: gmem fp32 (M, K/128) × sb[nb, k_g] (from
                                sb_regs) → LDS fp32 (PRE-MULTIPLIED sxsy slab)

            Slab dimensions:
              - ue8m0/fp32:    slot_id = row*(K/512) + k_pkt (packed-byte index)
              - fp32_post_mfma slot_id = n_block*(K/128)*tile_m
                                         + k_g*tile_m + row

            CRITICAL: structured as TWO phases — phase 1 issues ALL gmem loads,
            phase 2 does ALL LDS stores. The naive load/store-per-iter pattern
            forces the compiler to insert s_waitcnt vmcnt(0) BEFORE each store
            (the LDS store reads the just-loaded value, so a fence is required).
            That serializes loads/stores and stalls ~1.1M cyc on our shape (~7%
            of total). With load-all-then-store-all, all loads issue concurrently
            and one waitcnt at the start of phase 2 drains them in parallel.
            """
            # ── fp32_post_mfma branch: constexpr (n_block, k_g) outer loops
            # so sb_regs lookup is a compile-time index → zero v_cndmask. Per
            # constexpr (nb_c, kg_c), distribute `tile_m` rows across the first
            # `tile_m` threads (others idle). Each active thread does
            # `n_blocks_per_tile × (K/128)` outer iters total.
            if const_expr(scale_format == "fp32_post_mfma"):
                assert sb_regs is not None, (
                    "fp32_post_mfma load_sa_to_lds requires sb_regs"
                )
                # Gate on thread-in-row-range; uses scf.IfOp because tx is
                # runtime. The 'else' branch is empty (idle threads).
                in_range = fx.arith.cmpi(
                    fx.arith.CmpIPredicate.slt,
                    fx.Int32(tx).ir_value(),
                    fx.Int32(tile_m).ir_value(),
                )
                if_op = scf_dialect.IfOp(
                    in_range, results_=[], has_else=False,
                )
                with mlir_ir.InsertionPoint(if_op.then_block):
                    row_local = tx  # in [0, tile_m)
                    row_global = bx_m + row_local
                    # Phase 1: issue all gmem loads, deferred multiply.
                    pending_fp = []  # list of (slot, sa_val_ir, sb_const_ir)
                    for nb_c in range_constexpr(n_blocks_per_tile):
                        for kg_c in range_constexpr(K // 128):
                            # Constexpr-indexed sb scalar (zero v_cndmask).
                            sb_const = sb_regs[nb_c * (K // 128) + kg_c]
                            sa_gmem_idx = (
                                row_global * fx.Index(K // 128)
                                + fx.Index(kg_c)
                            )
                            sa_fp32 = buffer_ops.buffer_load(
                                sa_rsrc, fx.Int32(sa_gmem_idx),
                                vec_width=1, dtype=fx.Float32,
                            )
                            slot = (
                                fx.Index(nb_c * (K // 128) * tile_m)
                                + fx.Index(kg_c * tile_m)
                                + row_local
                            )
                            pending_fp.append((slot, sa_fp32, sb_const))
                    # Phase 2: all stores (sxsy = sa × sb), compiler emits
                    # one waitcnt at the start draining all loads in parallel.
                    for slot, sa_val, sb_const in pending_fp:
                        sxsy_val = sa_val * sb_const
                        v1 = Vec.from_elements([sxsy_val], fx.Float32)
                        v1.store(lds_sa, [slot])
                    scf_dialect.YieldOp([])
                return

            # Phase 1: issue ALL buffer_loads, accumulate (slot_id, value) pairs.
            pending = []  # list of (slot_id, value_to_store, dtype)
            for i in range_constexpr(_sa_slots_per_thread):
                slot_id = tx + fx.Index(i * total_threads)
                row_in_tile = slot_id // fx.Index(_sa_lds_per_row)
                k_packed_idx = slot_id % fx.Index(_sa_lds_per_row)
                row_a_global = bx_m + row_in_tile

                if const_expr(scale_format == "ue8m0"):
                    if const_expr(scale_layout == "row"):
                        sa_gmem_idx = (
                            row_a_global * fx.Index(_sa_lds_per_row)
                            + k_packed_idx
                        )
                    else:  # "col"
                        sa_gmem_idx = (
                            k_packed_idx * c_m
                            + row_a_global
                        )
                    sa_val = buffer_ops.buffer_load(
                        sa_rsrc, fx.Int32(sa_gmem_idx),
                        vec_width=1, dtype=fx.Int32,
                    )
                    pending.append((slot_id, sa_val, fx.Int32))
                else:  # "fp32" (in-kernel ue8m0 conversion)
                    if const_expr(scale_layout == "row"):
                        fp32_base = (
                            row_a_global * fx.Index(K // 128)
                            + k_packed_idx * fx.Index(4)
                        )
                        fp32_vec = buffer_ops.buffer_load(
                            sa_rsrc, fx.Int32(fp32_base),
                            vec_width=4, dtype=fx.Float32,
                        )
                        # Defer the byte conversion + pack to phase 2 — but the
                        # conversion is pure ALU (no memory dep) so it doesn't
                        # force a waitcnt. The fp32_vec uses the loaded values
                        # but only via VALU ops; the LDS store is the trigger.
                        # Compute now (no LDS store yet), pack to sa_val.
                        fp32v = Vec(fp32_vec)
                        b0 = _fp32_to_ue8m0_byte_i32(fp32v[0])
                        b1 = _fp32_to_ue8m0_byte_i32(fp32v[1])
                        b2 = _fp32_to_ue8m0_byte_i32(fp32v[2])
                        b3 = _fp32_to_ue8m0_byte_i32(fp32v[3])
                    else:  # "col"
                        bytes_per_k = c_m
                        k_base = k_packed_idx * fx.Index(4)
                        fp32s = []
                        for j in range_constexpr(4):
                            idx = (k_base + fx.Index(j)) * bytes_per_k + row_a_global
                            fp32_val = buffer_ops.buffer_load(
                                sa_rsrc, fx.Int32(idx),
                                vec_width=1, dtype=fx.Float32,
                            )
                            fp32s.append(fp32_val)
                        b0 = _fp32_to_ue8m0_byte_i32(fp32s[0])
                        b1 = _fp32_to_ue8m0_byte_i32(fp32s[1])
                        b2 = _fp32_to_ue8m0_byte_i32(fp32s[2])
                        b3 = _fp32_to_ue8m0_byte_i32(fp32s[3])
                    sa_val = _pack_4_bytes_to_i32(b0, b1, b2, b3)
                    pending.append((slot_id, sa_val, fx.Int32))

            # Phase 2: issue ALL LDS stores. Compiler will insert one waitcnt
            # at the start covering all in-flight loads concurrently.
            for slot_id, value, dtype in pending:
                v1 = Vec.from_elements([value], dtype)
                v1.store(lds_sa, [slot_id])

        def prefetch_sa_tile(kt):
            """Return list[m_repeat][groups_per_tile] of i32 IR values
            from the persistent LDS sa slab (ue8m0/fp32 modes).
            """
            sa_per_mi = []
            for mi in range_constexpr(m_repeat):
                row_in_tile = lane_mod_16 + fx.Index(mi * 16)
                sa_for_mi = []
                for g in range_constexpr(groups_per_tile):
                    k_block_global_int = kt * (tile_k // 128) + g
                    k_packed_idx = k_block_global_int // 4
                    slot = (
                        row_in_tile * fx.Index(_sa_lds_per_row)
                        + fx.Index(k_packed_idx)
                    )
                    v = Vec.load(
                        Vec.make_type(1, fx.Int32),
                        lds_sa, [slot],
                    )
                    sa_for_mi.append(v[0].ir_value())
                sa_per_mi.append(sa_for_mi)
            return sa_per_mi

        def prefetch_sa_tile_fp32_post_mfma(kt):
            """Return list[m_repeat][groups_per_tile] of Vec(4, fp32) values
            from the PRE-MULTIPLIED sxsy LDS slab.

            Slab layout: slot = n_block*(K/128)*tile_m + k_g*tile_m + row.

            n_block is WAVE-UNIFORM (= wave_id × n_per_wave // 128). Each
            wave reads only its own slab slice — same ds_read count as the
            un-multiplied baseline (1 per (mi, g)). The nb_offset is wave-
            uniform → held in SGPR, no per-lane address compute.
            """
            sa_per_mi_g = []
            if const_expr(n_blocks_per_tile == 1):
                nb_offset = fx.Index(0)
            else:
                # Wave-uniform: same across all 64 lanes of a wave.
                n_col_wave = wave_id * fx.Index(n_per_wave)
                n_block_wave = n_col_wave // fx.Index(128)
                nb_offset = n_block_wave * fx.Index((K // 128) * tile_m)
            for mi in range_constexpr(m_repeat):
                row_base_in_tile = fx.Index(mi * 16) + lane_div_16 * fx.Index(4)
                sa_for_mi = []
                for g in range_constexpr(groups_per_tile):
                    k_g = kt * (tile_k // 128) + g
                    base_slot = (
                        nb_offset
                        + fx.Index(k_g) * fx.Index(tile_m)
                        + row_base_in_tile
                    )
                    sa_vec = Vec.load(
                        Vec.make_type(4, fx.Float32),
                        lds_sa, [base_slot],
                    )
                    sa_for_mi.append(sa_vec)
                sa_per_mi_g.append(sa_for_mi)
            return sa_per_mi_g

        # ── sb load-once-per-WG into REGISTERS (not LDS) ──────────
        # sb is WG-uniform and tiny (n_blocks_per_tile * K/512 = 4-32 i32 total).
        # Loading to LDS + reading back per K-tile is pure overhead. Instead
        # we load all sb i32s into a Python list of IR values at prologue time;
        # `prefetch_sb_tile` then does Python-list indexing — no LDS round-trip,
        # no per-K-tile gmem load. `by` is workgroup-uniform so the gmem
        # addresses are wave-uniform and the compiler is free to use SALU paths.
        def load_sb_to_regs():
            """Load all sb scales for this WG into a Python list of IR values.

            Returns: list[_sb_lds_count] of i32 OR fp32 IR values:
                     - ue8m0/fp32:    i32 (packed UE8M0 byte across 4 K-128 groups)
                     - fp32_post_mfma: fp32 (1 per K-128 group)
                     Indexed by slot_id = nb_idx * _sb_lds_per_nb + k_idx
                     (consistent with the per-mode _sb_lds_per_nb).
            """
            sb_regs = []
            for slot_id in range_constexpr(_sb_lds_count):
                nb_idx = slot_id // _sb_lds_per_nb
                k_packed_idx = slot_id % _sb_lds_per_nb
                n_block_global = by * fx.Index(n_blocks_per_tile) + fx.Index(nb_idx)

                if const_expr(scale_format == "fp32_post_mfma"):
                    # gmem (N/128, K/128) fp32, 1 fp32 per K-128 group per N-128 block.
                    fp32_idx = (
                        n_block_global * fx.Index(K // 128)
                        + fx.Index(k_packed_idx)
                    )
                    sb_val = buffer_ops.buffer_load(
                        sb_rsrc, fx.Int32(fp32_idx),
                        vec_width=1, dtype=fx.Float32,
                    )
                elif const_expr(scale_format == "ue8m0"):
                    sb_gmem_idx = (
                        n_block_global * fx.Index(_sb_per_n128)
                        + fx.Index(k_packed_idx)
                    )
                    sb_val = buffer_ops.buffer_load(
                        sb_rsrc, fx.Int32(sb_gmem_idx),
                        vec_width=1, dtype=fx.Int32,
                    )
                else:  # "fp32" (in-kernel ue8m0 conversion)
                    fp32_base = (
                        n_block_global * fx.Index(K // 128)
                        + fx.Index(k_packed_idx * 4)
                    )
                    fp32_vec = buffer_ops.buffer_load(
                        sb_rsrc, fx.Int32(fp32_base),
                        vec_width=4, dtype=fx.Float32,
                    )
                    fp32v = Vec(fp32_vec)
                    b0 = _fp32_to_ue8m0_byte_i32(fp32v[0])
                    b1 = _fp32_to_ue8m0_byte_i32(fp32v[1])
                    b2 = _fp32_to_ue8m0_byte_i32(fp32v[2])
                    b3 = _fp32_to_ue8m0_byte_i32(fp32v[3])
                    sb_val = _pack_4_bytes_to_i32(b0, b1, b2, b3)

                # Append the raw IR value so prefetch_sb_tile and compute_tile
                # consume it the same way they consumed the old LDS-load result.
                if hasattr(sb_val, "ir_value"):
                    sb_regs.append(sb_val.ir_value())
                else:
                    sb_regs.append(sb_val)
            return sb_regs

        def prefetch_sb_tile(kt, sb_regs):
            """Return list[num_acc_n][groups_per_tile] of i32 IR values.

            Pure Python-list indexing into the pre-loaded sb_regs list — no
            LDS read, no gmem load. For tile_n=128 (n_blocks_per_tile=1) the
            slot is constexpr; for tile_n=256 (=2 blocks) we use arith.select
            to pick between the two blocks based on wave_id (waves 0,1 → block 0;
            waves 2,3 → block 1).
            """
            sb_per_ni = []
            for ni in range_constexpr(num_acc_n):
                # n_block_idx_local = (wave_id*n_per_wave + ni*16) // 128
                # (lane_mod_16 < 16 < 128, so lane-uniform)
                n_col_in_tile_wave_uniform = (
                    wave_id * fx.Index(n_per_wave) + fx.Index(ni * 16)
                )
                n_block_idx_local = (
                    n_col_in_tile_wave_uniform // fx.Index(128)
                )
                sb_for_ni = []
                for g in range_constexpr(groups_per_tile):
                    k_block_global_int = kt * (tile_k // 128) + g
                    k_packed_idx = k_block_global_int // 4
                    if const_expr(n_blocks_per_tile == 1):
                        # tile_n=128: 1 block per tile → slot is just k_packed_idx.
                        sb_for_ni.append(sb_regs[k_packed_idx])
                    else:
                        # tile_n=256: 2 blocks per tile. Select between block 0
                        # (slot = k_packed_idx) and block 1 (slot = _sb_lds_per_nb
                        # + k_packed_idx) on the wave-uniform n_block_idx_local.
                        val0 = sb_regs[k_packed_idx]
                        val1 = sb_regs[_sb_lds_per_nb + k_packed_idx]
                        cond = fx.arith.cmpi(
                            fx.arith.CmpIPredicate.eq,
                            fx.Int32(n_block_idx_local).ir_value(),
                            fx.Int32(1).ir_value(),
                        )
                        sb_for_ni.append(fx.arith.select(cond, val1, val0))
                sb_per_ni.append(sb_for_ni)
            return sb_per_ni

        def prefetch_sb_tile_fp32_post_mfma(kt, sb_regs):
            """Return list[num_acc_n][groups_per_tile] of fp32 IR values.

            sb_regs holds 1 fp32 per K-128 group per N-128 block, indexed by
            slot_id = nb_idx * (K/128) + k_g. Per-(ni, g) we pick the slot.
            """
            sb_per_ni = []
            for ni in range_constexpr(num_acc_n):
                n_col_in_tile_wave_uniform = (
                    wave_id * fx.Index(n_per_wave) + fx.Index(ni * 16)
                )
                n_block_idx_local = (
                    n_col_in_tile_wave_uniform // fx.Index(128)
                )
                sb_for_ni = []
                for g in range_constexpr(groups_per_tile):
                    k_g = kt * (tile_k // 128) + g
                    if const_expr(n_blocks_per_tile == 1):
                        sb_for_ni.append(sb_regs[k_g])
                    else:
                        val0 = sb_regs[k_g]
                        val1 = sb_regs[_sb_lds_per_nb + k_g]
                        cond = fx.arith.cmpi(
                            fx.arith.CmpIPredicate.eq,
                            fx.Int32(n_block_idx_local).ir_value(),
                            fx.Int32(1).ir_value(),
                        )
                        sb_for_ni.append(fx.arith.select(cond, val1, val0))
                sb_per_ni.append(sb_for_ni)
            return sb_per_ni

        # ── MFMA + scale ──────────────────────────────────────────
        mfma_res_ty = Vec.make_type(4, fx.Float32)
        mfma_fp8_k128 = rocdl.mfma_scale_f32_16x16x128_f8f6f4

        def pack_i64x4_to_i32x8(x0, x1, x2, x3):
            return Vec.from_elements(
                [x0, x1, x2, x3], fx.Int64,
            ).bitcast(fx.Int32)

        def lds_a0_prefetch(lds_buffer):
            fp8_row = lane_mod_16
            fp8_col_bytes = lane_div_16 * 16
            return lds_load_a_packs_k64(fp8_row, fp8_col_bytes, lds_buffer)

        def compute_tile(
            accs_in, b_tile_in, kt, fp8_lds_buffer,
            sa_per_mi, sb_per_ni, a0_prefetch=None,
        ):
            """Compute one K-tile's MFMAs (ue8m0 / fp32-in-kernel-round modes).

            MFMA-fused scale: scale operands fed directly to mfma_scale_*.
            sa_per_mi[mi][g] = i32 (packed UE8M0 bytes), opsel via byte_in_i32.
            sb_per_ni[ni][g] = i32 (same).
            """
            current_accs = list(accs_in)
            rocdl.iglp_opt(2)

            for g in range_constexpr(groups_per_tile):
                k_block_global_int = kt * (tile_k // 128) + g
                byte_in_i32 = k_block_global_int % 4

                fp8_group_col_base = g * 128
                ku_base = g * mfmas_per_group
                b_pa = b_tile_in[ku_base + 0]
                b_pb = b_tile_in[ku_base + 1]
                b_pc = b_tile_in[ku_base + 2]
                b_pd = b_tile_in[ku_base + 3]

                for mi in range_constexpr(m_repeat):
                    fp8_row = lane_mod_16 + (mi * 16)
                    fp8_col_bytes_0 = fp8_group_col_base + lane_div_16 * 16
                    fp8_col_bytes_1 = fp8_col_bytes_0 + 64

                    if const_expr(
                        (a0_prefetch is not None) and (g == 0) and (mi == 0)
                    ):
                        a0, a1 = a0_prefetch
                    else:
                        a0, a1 = lds_load_a_packs_k64(
                            fp8_row, fp8_col_bytes_0, fp8_lds_buffer,
                        )
                    a2, a3 = lds_load_a_packs_k64(
                        fp8_row, fp8_col_bytes_1, fp8_lds_buffer,
                    )
                    a128 = pack_i64x4_to_i32x8(a0, a1, a2, a3)
                    sa_i32 = sa_per_mi[mi][g]

                    for ni in range_constexpr(num_acc_n):
                        b128 = pack_i64x4_to_i32x8(
                            b_pa[ni], b_pb[ni], b_pc[ni], b_pd[ni],
                        )
                        sb_i32 = sb_per_ni[ni][g]
                        acc_idx = mi * num_acc_n + ni
                        current_accs[acc_idx] = mfma_fp8_k128(
                            mfma_res_ty,
                            [a128, b128, current_accs[acc_idx],
                             0, 0,
                             byte_in_i32, sa_i32,
                             byte_in_i32, sb_i32],
                        )

            return current_accs

        # ── compute_tile for fp32_post_mfma mode ────────────────────────
        # Identity scale (UE8M0 byte 127 = 2^0 = 1.0). Same opsel byte for all
        # 4 packed bytes, so a single 0x7F7F7F7F constant works for any g.
        _MFMA_K128_NO_SCALE = 0x7F7F7F7F

        def compute_tile_fp32_post_mfma(
            accs_in, b_tile_in, kt, fp8_lds_buffer,
            sa_per_mi_g, sb_per_ni, a0_prefetch=None,
        ):
            """Compute one K-tile's MFMAs with TRUE FP32 scale (post-MFMA Vec FMA).

            For each (mi, ni, g):
              1. Scratch MFMA: mfma_scale(...) with scale=1.0 → Vec(4,fp32) per (mi,ni)
              2. Compute sxsy_vec[ii] = sa_vec[mi,g][ii] * sb[ni,g]  (Vec × scalar)
              3. acc[mi,ni] = acc[mi,ni] + scratch[ni] * sxsy_vec     (VEC FMA)

            Key optimization vs naive per-ii loop:
              - sa_per_mi_g returns Vec(4, fp32) per (mi, g) — packs the 4 ii's
                values into one Vec at prefetch time.
              - The post-MFMA update is a single Vec(4) × Vec(4) + Vec(4) per
                (mi, ni, g) — 4 fp32 FMAs in vector form, NO immutable Vec
                rebuild loop. Compiler emits 4 v_fmac_f32 directly.
              - sb is a scalar broadcast across the 4 lanes of the Vec (one
                scalar-to-vector splat then vec-mul, or compiler may fold it).
            """
            current_accs = list(accs_in)
            rocdl.iglp_opt(2)

            for g in range_constexpr(groups_per_tile):
                fp8_group_col_base = g * 128
                ku_base = g * mfmas_per_group
                b_pa = b_tile_in[ku_base + 0]
                b_pb = b_tile_in[ku_base + 1]
                b_pc = b_tile_in[ku_base + 2]
                b_pd = b_tile_in[ku_base + 3]

                # ── PARTIAL HOIST: per-g a-LDS loads ─────────────────
                # Issue all ds_reads for THIS group's A operands UP FRONT
                # (across all mi), before any MFMA in this group. Puts
                # MFMA-needed LDS reads at the head of the lgkm FIFO so
                # LLVM emits relaxed lgkmcnt(N) instead of lgkmcnt(0).
                #
                # GATED on num_acc_n <= 2: at small num_acc_n (= tile_n
                # / 64), the MFMA inner loop's register footprint
                # (scratch_accs + sxsy_vecs scale linearly with
                # num_acc_n) is small enough that the +8 VGPRs from
                # hoisted a128 doesn't push us over the spill/occupancy
                # cliff. At num_acc_n >= 4 (tile_n >= 256), measured
                # regressions of −3.7% to −15.1% at large-M shapes.
                #
                # Measured wins (tile_n=128, num_acc_n=2):
                #   1024x1024x2048:    +1.7%
                #   4096x4096x8192:    +6.0%
                #   1024x4096x8192:    +3.1%
                #   4096x1024x2048:    +7.8%
                #   32768x12288x2048:  +4.0%
                #   32768x2048x4096:   +1.5%
                #
                # Implementation note: we don't hoist across groups
                # (full hoist measured +108 VGPRs / −23% at skinny-M
                # deep-K shapes).
                if const_expr(num_acc_n <= 2):
                    a128_for_g = []
                    for mi in range_constexpr(m_repeat):
                        fp8_row = lane_mod_16 + (mi * 16)
                        fp8_col_bytes_0 = fp8_group_col_base + lane_div_16 * 16
                        fp8_col_bytes_1 = fp8_col_bytes_0 + 64
                        if const_expr(
                            (a0_prefetch is not None) and (g == 0) and (mi == 0)
                        ):
                            a0, a1 = a0_prefetch
                        else:
                            a0, a1 = lds_load_a_packs_k64(
                                fp8_row, fp8_col_bytes_0, fp8_lds_buffer,
                            )
                        a2, a3 = lds_load_a_packs_k64(
                            fp8_row, fp8_col_bytes_1, fp8_lds_buffer,
                        )
                        a128_for_g.append(pack_i64x4_to_i32x8(a0, a1, a2, a3))

                # ── Resolve all a128[mi] and b128[ni] up front ──────
                a128_list = []
                for mi in range_constexpr(m_repeat):
                    if const_expr(num_acc_n <= 2):
                        a128_list.append(a128_for_g[mi])
                    else:
                        # HOIST OFF: load inline (large num_acc_n → too much VGPR pressure)
                        fp8_row = lane_mod_16 + (mi * 16)
                        fp8_col_bytes_0 = fp8_group_col_base + lane_div_16 * 16
                        fp8_col_bytes_1 = fp8_col_bytes_0 + 64
                        if const_expr(
                            (a0_prefetch is not None) and (g == 0) and (mi == 0)
                        ):
                            a0, a1 = a0_prefetch
                        else:
                            a0, a1 = lds_load_a_packs_k64(
                                fp8_row, fp8_col_bytes_0, fp8_lds_buffer,
                            )
                        a2, a3 = lds_load_a_packs_k64(
                            fp8_row, fp8_col_bytes_1, fp8_lds_buffer,
                        )
                        a128_list.append(pack_i64x4_to_i32x8(a0, a1, a2, a3))

                b128_list = []
                for ni in range_constexpr(num_acc_n):
                    b128_list.append(pack_i64x4_to_i32x8(
                        b_pa[ni], b_pb[ni], b_pc[ni], b_pd[ni],
                    ))

                # ── BATCHED MFMA + DEFERRED FMA (pipeline-depth window) ──
                # The scaled K128 MFMA is latency=33.5cyc / throughput=4cyc on
                # gfx950 (8.37× pipeline depth — needs ≥8 INDEPENDENT MFMAs in
                # flight to saturate). The old (mi){MFMA;FMA(prev)} nesting kept
                # the FMA only 1 MFMA behind its producer → the FMA stalled on
                # the 33.5-cyc MFMA latency, forcing s_nop padding.
                #
                # Fix: process the m_repeat×num_acc_n (mi,ni) accumulators in
                # BATCHES of MFMA_BATCH independent scratch accs. Within a batch
                # we issue all MFMAs first (filling the pipeline to depth), then
                # drain that batch's FMAs (which read accs written BATCH MFMAs
                # ago — past the latency window). Batching caps live scratch at
                # MFMA_BATCH Vec(4) instead of the full 32, avoiding the VGPR
                # blowup (full flatten hit 476 VGPRs / −18%).
                #
                # mfma_batch: issue `mfma_batch` independent MFMAs into distinct
                # scratch accs, then drain their FMAs, repeat. This N-deep window
                # hides the 33.5-cyc scaled-K128 MFMA latency (latency/throughput
                # = 8.37× on gfx950) so the consumer FMA doesn't stall, WITHOUT
                # the VGPR blowup of holding all m_repeat×num_acc_n scratch accs
                # at once (full flatten hit 476 VGPRs / −18%).
                #
                # Sweet spot is shape-dependent (swept {2,4,6,8,16}):
                #   shape1 (tile_m=64): batch=6 best (+1.9%); batch≥8 cliffs.
                #   shape2 (tile_m=128): batch=2 best (+3.4%).
                # Default mfma_batch=2 wins on BOTH with no regressions; the
                # autotuner can pick a larger value per (M,N,K).
                # Clamp to the total accumulator count (m_repeat×num_acc_n).
                _flat = [(mi, ni) for mi in range(m_repeat)
                         for ni in range(num_acc_n)]
                MFMA_BATCH = min(mfma_batch, len(_flat))
                for _b0 in range_constexpr(0, len(_flat), MFMA_BATCH):
                    _batch = _flat[_b0:_b0 + MFMA_BATCH]
                    _scratch = {}
                    # Issue all MFMAs in this batch (independent accumulators).
                    for (mi, ni) in _batch:
                        _scratch[(mi, ni)] = mfma_fp8_k128(
                            mfma_res_ty,
                            [a128_list[mi], b128_list[ni],
                             Vec.filled(4, 0.0, fx.Float32),
                             0, 0,
                             0, _MFMA_K128_NO_SCALE,
                             0, _MFMA_K128_NO_SCALE],
                        )
                    # Drain this batch's FMAs (producers are ≥BATCH MFMAs back).
                    for (mi, ni) in _batch:
                        idx = mi * num_acc_n + ni
                        sxsy_vec = sa_per_mi_g[mi][g]
                        current_accs[idx] = Vec(fx.vector.fma(
                            _scratch[(mi, ni)],
                            sxsy_vec,
                            current_accs[idx],
                        ))

            return current_accs

        # ── Output store (bf16) ───────────────────────────────────
        # No head dim: C is (M, N); stride per row = N elements = N bf16 cells.
        def store_output_bf16(final_accs):
            def body_row(*, mi, ii, row_in_tile, row):
                col_base_n = by_n + n_tile_base + lane_mod_16
                idx_base = row * fx.Index(N) + col_base_n
                for ni in range_constexpr(num_acc_n):
                    acc_idx = mi * num_acc_n + ni
                    acc = final_accs[acc_idx]
                    val = Vec(acc)[ii]
                    val_bf16 = fx.BFloat16(val)
                    idx_out = idx_base + (ni * 16)
                    buffer_ops.buffer_store(val_bf16, c_rsrc, idx_out)

            mfma_epilog(
                use_cshuffle=False,
                arith=fx.arith,
                range_constexpr=range_constexpr,
                m_repeat=m_repeat,
                lane_div_16=lane_div_16,
                bx_m=bx_m,
                body_row=body_row,
            )

        # ── K-loop driver: unrolled pingpong ──
        n_vmem_keep = num_b_loads

        def _waitcnt_vmcnt_lgkm0(vmc):
            vm_lo = vmc & 0xF
            vm_hi = (vmc >> 4) & 0x3
            return vm_lo | (7 << 4) | (0 << 8) | (vm_hi << 14)

        _waitcnt_imm = _waitcnt_vmcnt_lgkm0(n_vmem_keep)

        Vec_init = fx.Vector.filled(4, 0.0, fx.Float32)
        accs = [Vec_init] * (num_acc_n * m_repeat)
        num_tiles = K // tile_k

        # Dispatch prefetch + compute based on scale_format. The K-loop body
        # below is mode-agnostic — it uses `_prefetch_sa`, `_prefetch_sb`,
        # `_compute_tile` which are bound to the right per-mode implementation.
        if const_expr(scale_format == "fp32_post_mfma"):
            _prefetch_sa = prefetch_sa_tile_fp32_post_mfma
            _prefetch_sb = lambda kt, regs: prefetch_sb_tile_fp32_post_mfma(kt, regs)
            _compute_tile = compute_tile_fp32_post_mfma
        else:
            _prefetch_sa = prefetch_sa_tile
            _prefetch_sb = lambda kt, regs: prefetch_sb_tile(kt, regs)
            _compute_tile = compute_tile

        # NB: for fp32_post_mfma, load_sb_to_regs() must run BEFORE
        # load_sa_to_lds() so the latter can multiply sa × sb in-flight.
        sb_regs = load_sb_to_regs()
        load_sa_to_lds(sb_regs)
        load_a_tile_to_lds_async(fx.Index(0), lds_a_pong)
        b_pong = load_b_tile_preshuffled(fx.Index(0))
        gpu.barrier()
        sa_pong = _prefetch_sa(0)
        sb_pong = _prefetch_sb(0, sb_regs)
        a0_pong = lds_a0_prefetch(lds_a_pong)

        if const_expr(num_tiles == 1):
            accs = _compute_tile(
                accs, b_pong, 0, lds_a_pong,
                sa_pong, sb_pong, a0_prefetch=a0_pong,
            )
        else:
            if const_expr(num_tiles % 2 == 1):
                _loop_end_excl = num_tiles - 1
            else:
                _loop_end_excl = num_tiles - 2

            for kt_base in range_constexpr(0, _loop_end_excl, 2):
                next_k1 = fx.Index((kt_base + 1) * tile_k)
                load_a_tile_to_lds_async(next_k1, lds_a_ping)
                b_ping = load_b_tile_preshuffled(next_k1)
                sa_ping = _prefetch_sa(kt_base + 1)
                sb_ping = _prefetch_sb(kt_base + 1, sb_regs)

                accs = _compute_tile(
                    accs, b_pong, kt_base, lds_a_pong,
                    sa_pong, sb_pong, a0_prefetch=a0_pong,
                )

                rocdl.s_waitcnt(_waitcnt_imm)
                gpu.barrier()
                a0_ping = lds_a0_prefetch(lds_a_ping)

                next_k2 = fx.Index((kt_base + 2) * tile_k)
                load_a_tile_to_lds_async(next_k2, lds_a_pong)
                b_pong = load_b_tile_preshuffled(next_k2)
                sa_pong = _prefetch_sa(kt_base + 2)
                sb_pong = _prefetch_sb(kt_base + 2, sb_regs)

                accs = _compute_tile(
                    accs, b_ping, kt_base + 1, lds_a_ping,
                    sa_ping, sb_ping, a0_prefetch=a0_ping,
                )

                rocdl.s_waitcnt(_waitcnt_imm)
                gpu.barrier()
                a0_pong = lds_a0_prefetch(lds_a_pong)

            if const_expr(num_tiles % 2 == 1):
                accs = _compute_tile(
                    accs, b_pong, _loop_end_excl, lds_a_pong,
                    sa_pong, sb_pong, a0_prefetch=a0_pong,
                )
            else:
                next_k1 = fx.Index((_loop_end_excl + 1) * tile_k)
                load_a_tile_to_lds_async(next_k1, lds_a_ping)
                b_ping = load_b_tile_preshuffled(next_k1)
                sa_ping = _prefetch_sa(_loop_end_excl + 1)
                sb_ping = _prefetch_sb(_loop_end_excl + 1, sb_regs)

                accs = _compute_tile(
                    accs, b_pong, _loop_end_excl, lds_a_pong,
                    sa_pong, sb_pong, a0_prefetch=a0_pong,
                )

                rocdl.s_waitcnt(_waitcnt_imm)
                gpu.barrier()
                a0_ping = lds_a0_prefetch(lds_a_ping)

                accs = _compute_tile(
                    accs, b_ping, _loop_end_excl + 1, lds_a_ping,
                    sa_ping, sb_ping, a0_prefetch=a0_ping,
                )

        store_output_bf16(accs)

    @flyc.kernel
    def kernel_gemm(
        arg_c: fx.Tensor,
        arg_a: fx.Tensor,
        arg_b: fx.Tensor,
        arg_sa: fx.Tensor,
        arg_sb: fx.Tensor,
        i32_m: fx.Int32,
    ):
        _kernel_body(arg_c, arg_a, arg_b, arg_sa, arg_sb, i32_m)

    @flyc.jit
    def launch_gemm(
        arg_c: fx.Tensor,
        arg_a: fx.Tensor,
        arg_b: fx.Tensor,
        arg_sa: fx.Tensor,
        arg_sb: fx.Tensor,
        i32_m: fx.Int32,
        stream: fx.Stream,
    ):
        allocator_pong.finalized = False
        allocator_ping.finalized = False
        ctx = CompilationContext.get_current()
        from flydsl._mlir import ir
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator_pong.finalize()
            allocator_ping.finalize()
        gx = (i32_m + (tile_m - 1)) // tile_m
        gy = N // tile_n
        kernel_gemm._func.__name__ = KERNEL_NAME
        launcher = kernel_gemm(arg_c, arg_a, arg_b, arg_sa, arg_sb, i32_m)
        launcher.launch(
            grid=(gx, gy, 1),
            block=(256, 1, 1),
            stream=stream,
        )

    return launch_gemm


def compile_preshuffle_gemm_blockscaled_auto(
    *,
    M: int,
    N: int,
    K: int,
    scale_format: str = "ue8m0",
):
    """Autotune-dispatched entry point.

    Looks up (M, N, K) in _AUTOTUNE_WINNERS; on miss, falls back to defaults
    (tile_m=128, tile_n=128, tile_k=128, block_swizzle_n=0). The same tile
    winners are used for both `ue8m0` and `fp32` scale formats — the hot
    path (compute_tile, MFMA) is bit-identical between modes; only the
    prologue loaders differ, and their cost is amortized over all K-tiles.
    """
    winner = _autotune_lookup(M, N, K)
    if winner is None:
        tm, tn, tk, bsw, mb = 128, 128, 128, 0, 2
    elif len(winner) == 5:
        tm, tn, tk, bsw, mb = winner
    else:
        tm, tn, tk, bsw = winner
        mb = 2
    return compile_preshuffle_gemm_blockscaled(
        M=M, N=N, K=K,
        tile_m=tm, tile_n=tn, tile_k=tk,
        block_swizzle_n=bsw,
        scale_format=scale_format,
        mfma_batch=mb,
    )


def compile_preshuffle_gemm_blockscaled_fp32(
    *,
    M: int,
    N: int,
    K: int,
    tile_m: int = 128,
    tile_n: int = 128,
    tile_k: int = 128,
    block_swizzle_n: int = 0,
    mfma_batch: int = 2,
):
    """Convenience wrapper: fp32-scale variant of the blockscaled GEMM.

    Equivalent to `compile_preshuffle_gemm_blockscaled(..., scale_format="fp32")`.

    Scale ABI differs from the UE8M0 default:
      arg_sa: fp32 (M, K//128)         — one fp32 scale per K-128 group of A
      arg_sb: fp32 (N//128, K//128)    — one fp32 scale per (128×128) block of B

    Each fp32 is converted to a UE8M0 byte (rounded to nearest power-of-2
    exponent) in the load-to-LDS prologue. Conversion uses:
        exp_field = ((bits + 0x400000) & 0xFF800000) >> 23   ∈ [0, 255]
    Same rounding rule as `aiter.ops.quant.per_token_quant_hip` and as the
    host-side `_fp32_to_ue8m0_byte` helper in the test file.

    Accuracy: identical to passing the corresponding pre-rounded UE8M0
    bytes — the conversion is lossless within the UE8M0 (power-of-2 only)
    representation.
    """
    return compile_preshuffle_gemm_blockscaled(
        M=M, N=N, K=K,
        tile_m=tile_m, tile_n=tile_n, tile_k=tile_k,
        block_swizzle_n=block_swizzle_n,
        scale_format="fp32",
        mfma_batch=mfma_batch,
    )
