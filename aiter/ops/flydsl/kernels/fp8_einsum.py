"""FP8 einsum kernel: `bhr,hdr->bhd` with online bf16->fp8 activation quant.

DeepGEMM-equivalent fp8_einsum for AMD gfx950, with two scale modes selectable
at compile time:
  - "fp32"  : SM90-style FP32 scales, scratch acc + register multiply.
  - "ue8m0" : SM100-style packed UE8M0 scales, hardware-fused scaled MFMA.

See `fp8_einsum_design.md` for the full design and accuracy contract.

Step 1 in the build sequence: structural skeleton + bf16 A LDS pipeline +
(h, m_tile) block-id grouping. The compute_tile in this file is a placeholder
that uses a constant scale s_x = 1/448 so input range must be clamped to
[-448, 448] for the bf16 reference to match. Real scale logic lands in step 3.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import buffer_ops, const_expr, gpu, range_constexpr, rocdl
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr

from .mfma_epilogues import mfma_epilog
from .mfma_preshuffle_pipeline import (
    buffer_copy_gmem16_dwordx4,
    swizzle_xor16,
    tile_chunk_coord_i32,
)


_VALID_SCALE_MODES = ("fp32", "ue8m0")


def compile_fp8_einsum_bhr_hdr_bhd(
    *,
    H: int,
    D: int,
    R: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    scale_mode: str = "fp32",
    lds_stage: int = 1,
    waves_per_eu=None,
    use_async_copy: bool = False,
    xcd_swizzle: int = 0,
):
    """Compile `Z[b,h,d] = einsum('bhr,hdr->bhd', X_bf16, Y_fp8)` for gfx950.

    Compile-time constants: H, D, R, tile_m/n/k, scale_mode.
    Runtime parameter:      B (batch size).

    Kernel signature (launch_fn):
        launch_fn(arg_z, arg_x, arg_y, arg_sy, i32_b, stream)
          arg_z : bf16 (B, H, D)
          arg_x : bf16 (B, H, R)
          arg_y : fp8  e4m3, preshuffled per head, layout (n0,k0,4,16,16)
          arg_sy: fp32 (H, D//128, R//128)         if scale_mode="fp32"
                  int32 (H, D//128, R//(128*4))    if scale_mode="ue8m0"
          i32_b : runtime batch size
    """
    if scale_mode not in _VALID_SCALE_MODES:
        raise ValueError(
            f"scale_mode must be one of {_VALID_SCALE_MODES}, got {scale_mode!r}"
        )
    if scale_mode == "ue8m0":
        # UE8M0 mode requires `mfma_scale_f32_16x16x128_f8f6f4` with per-lane
        # packed UE8M0 scale operands whose row/K-block mapping has not been
        # validated on real gfx950 hardware in this codebase yet. The fp32
        # path produces bit-aligned results vs DeepGEMM `use_ue8m0=False` and
        # is the focus of v1. Step 4 (ue8m0) is left as a stub with the design
        # doc as spec — wire it up once hardware-in-the-loop testing is available.
        raise NotImplementedError(
            "scale_mode='ue8m0' lands in step 4 of the build sequence; only "
            "scale_mode='fp32' is implemented in v1. See fp8_einsum_design.md."
        )

    gpu_arch = get_hip_arch()
    if not str(gpu_arch).startswith("gfx95"):
        raise RuntimeError(
            f"fp8_einsum kernel currently targets gfx950 only, got {gpu_arch}"
        )

    # ── Validate compile-time shape/tile constraints ─────────────────────────
    if tile_k % 128 != 0:
        raise ValueError(
            f"tile_k must be a multiple of 128 (the quant group size), "
            f"got tile_k={tile_k}"
        )
    if tile_m < 16 or (tile_m % 16) != 0:
        raise ValueError(f"tile_m must be a positive multiple of 16, got {tile_m}")
    if tile_n < 64 or (tile_n % 64) != 0:
        raise ValueError(f"tile_n must be a positive multiple of 64, got {tile_n}")
    if R % tile_k != 0:
        raise ValueError(f"R={R} must be divisible by tile_k={tile_k}")
    if D % tile_n != 0:
        raise ValueError(f"D={D} must be divisible by tile_n={tile_n}")
    if R % 128 != 0:
        raise ValueError(f"R={R} must be a multiple of 128 (quant group)")
    if D % 128 != 0:
        raise ValueError(f"D={D} must be a multiple of 128 (B-side block)")

    # A is bf16, B is preshuffled fp8 e4m3.
    elem_bytes_a = 2  # bf16
    elem_bytes_b = 1  # fp8 e4m3

    KERNEL_NAME = (
        f"fp8_einsum_bhr_hdr_bhd"
        f"_H{H}_D{D}_R{R}"
        f"_t{tile_m}x{tile_n}x{tile_k}"
        f"_sm_{scale_mode}"
        f"_lds{lds_stage}"
    )
    if xcd_swizzle > 0:
        KERNEL_NAME += f"_xcd{xcd_swizzle}"

    # ── Threading + per-tile byte budgets ────────────────────────────────────
    total_threads = 256
    wave_size = 64
    num_waves = total_threads // wave_size  # 4

    tile_k_bytes_a = tile_k * elem_bytes_a  # bf16
    tile_k_bytes_b = tile_k * elem_bytes_b  # fp8

    bytes_a_per_tile = tile_m * tile_k_bytes_a
    if bytes_a_per_tile % total_threads != 0:
        raise ValueError(
            f"tile_m*tile_k*2 must be divisible by {total_threads}: "
            f"tile_m={tile_m}, tile_k={tile_k}"
        )
    bytes_per_thread_a = bytes_a_per_tile // total_threads
    a_load_bytes = 16
    if bytes_per_thread_a % a_load_bytes != 0:
        raise ValueError(
            f"bytes_per_thread_a ({bytes_per_thread_a}) must be divisible by 16"
        )
    num_a_loads = bytes_per_thread_a // a_load_bytes

    bytes_b_per_tile = tile_n * tile_k_bytes_b
    bytes_per_thread_b = bytes_b_per_tile // total_threads
    b_load_bytes = 16
    num_b_loads = bytes_per_thread_b // b_load_bytes

    # ── LDS sizing (A only; B is fetched into registers) ─────────────────────
    Vec = fx.Vector
    allocator = SmemAllocator(None, arch=gpu_arch, global_sym_name="smem0")
    lds_stride_bytes = tile_k_bytes_a  # row stride = K in bytes
    lds_tile_bytes = tile_m * lds_stride_bytes  # bf16 A tile
    # For step 1 use lds_stage=1 (single buffer). Ping-pong wires in later.
    if lds_stage != 1:
        raise NotImplementedError(
            "lds_stage=2 ping-pong pipeline lands in a later build step; "
            "use lds_stage=1 for now."
        )
    lds_total_elems_a = lds_tile_bytes // elem_bytes_a  # bf16 elems
    lds_alloc_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_alloc_offset + lds_total_elems_a * elem_bytes_a

    # ── Compile-time MFMA layout numbers ─────────────────────────────────────
    m_repeat = tile_m // 16
    n_per_wave = tile_n // num_waves
    num_acc_n = n_per_wave // 16
    # K=64 bytes worth of A per MFMA chunk (= 32 fp8 elements after quant; matches
    # the template's K=64-byte unit even though MFMA itself does K=32 fp8).
    k_unroll = tile_k_bytes_a // elem_bytes_a // 32  # number of K=32 fp8 chunks
    # k_unroll above = tile_k / 32 = total MFMA K-steps (32 fp8 elems per MFMA).

    # ── B preshuffle layout (per-head; same as preshuffle_gemm) ──────────────
    # Per head, B[h, :, :] has shape (D, R) in logical layout, preshuffled into
    # (n0, k0, klane=4, nlane=16, kpack=16) with strides as computed below.
    k_bytes_b_per_head = R * elem_bytes_b  # bf16-equiv K bytes = R since fp8
    n0_val = D // 16
    k0_val = k_bytes_b_per_head // 64  # k0_val = R // 64
    kpack_elems = 16  # bytes per pack (16 fp8)
    _stride_nlane = kpack_elems  # = 16
    _stride_klane = 16 * _stride_nlane  # = 256
    _stride_k0 = 4 * _stride_klane  # = 1024
    _stride_n0 = k0_val * _stride_k0  # per-head stride
    # Per-head element count of B (fp8 elements):
    b_elems_per_head = n0_val * _stride_n0  # == D * R

    return _build_compiled(
        H=H, D=D, R=R,
        tile_m=tile_m, tile_n=tile_n, tile_k=tile_k,
        scale_mode=scale_mode,
        gpu_arch=gpu_arch,
        KERNEL_NAME=KERNEL_NAME,
        elem_bytes_a=elem_bytes_a, elem_bytes_b=elem_bytes_b,
        total_threads=total_threads, wave_size=wave_size, num_waves=num_waves,
        num_a_loads=num_a_loads, a_load_bytes=a_load_bytes,
        num_b_loads=num_b_loads, b_load_bytes=b_load_bytes,
        bytes_per_thread_a=bytes_per_thread_a,
        bytes_per_thread_b=bytes_per_thread_b,
        allocator=allocator,
        lds_alloc_offset=lds_alloc_offset,
        lds_total_elems_a=lds_total_elems_a,
        lds_stride_bytes=lds_stride_bytes,
        m_repeat=m_repeat, n_per_wave=n_per_wave, num_acc_n=num_acc_n,
        k_unroll=k_unroll,
        n0_val=n0_val, k0_val=k0_val, kpack_elems=kpack_elems,
        _stride_n0=_stride_n0, _stride_k0=_stride_k0,
        _stride_klane=_stride_klane, _stride_nlane=_stride_nlane,
        b_elems_per_head=b_elems_per_head,
        waves_per_eu=waves_per_eu,
    )


def _build_compiled(
    *,
    H, D, R, tile_m, tile_n, tile_k, scale_mode,
    gpu_arch, KERNEL_NAME,
    elem_bytes_a, elem_bytes_b,
    total_threads, wave_size, num_waves,
    num_a_loads, a_load_bytes,
    num_b_loads, b_load_bytes,
    bytes_per_thread_a, bytes_per_thread_b,
    allocator, lds_alloc_offset, lds_total_elems_a, lds_stride_bytes,
    m_repeat, n_per_wave, num_acc_n, k_unroll,
    n0_val, k0_val, kpack_elems,
    _stride_n0, _stride_k0, _stride_klane, _stride_nlane,
    b_elems_per_head,
    waves_per_eu,
):
    Vec = fx.Vector
    fp8_dtype = fx.Float8E4M3FN  # gfx950 uses E4M3FN (not FNUZ)

    # Number of M-tile rows per `h` group:
    #   gx_per_h = ceil(B, tile_m)  — but B is runtime, so compute at runtime.

    # Compile-time strides (bytes / elems as appropriate)
    stride_h_x_elems = R            # bf16 elems between heads of X within a batch
    stride_b_x_elems = H * R        # bf16 elems between batches of X
    stride_b_z_elems = H * D        # bf16 elems between batches of Z

    @flyc.kernel
    def kernel_gemm(
        arg_z: fx.Tensor,
        arg_x: fx.Tensor,
        arg_y: fx.Tensor,
        arg_sy: fx.Tensor,
        i32_b: fx.Int32,
    ):
        c_b = fx.Index(i32_b)

        # ── Thread / block decomposition ─────────────────────────────────────
        tx = gpu.thread_id("x")
        bx = gpu.block_id("x")   # encodes (h, m_tile)
        by = gpu.block_id("y")   # n-tile

        # gx_per_h = ceil(B, tile_m) (runtime)
        gx_per_h = (c_b + (tile_m - 1)) // tile_m
        bx_h = bx // gx_per_h           # 0 .. H-1
        bx_m_idx = bx % gx_per_h        # m-tile index within this h
        bx_m = bx_m_idx * tile_m        # row offset in M (=B)
        by_n = by * tile_n              # column offset in N (=D)

        # ── LDS ──────────────────────────────────────────────────────────────
        base_ptr = allocator.get_base()
        lds_a_ptr = SmemPtr(
            base_ptr, lds_alloc_offset,
            fx.BFloat16.ir_type, shape=(lds_total_elems_a,),
        )
        lds_a = lds_a_ptr.get()

        # ── Buffer resources ────────────────────────────────────────────────
        # X: bf16 (B, H, R), num_records = B*H*R bf16 elems = B*H*R*2 bytes.
        _x_nrec_bytes = fx.Int64(c_b * (H * R * elem_bytes_a))
        x_rsrc = buffer_ops.create_buffer_resource(
            arg_x, max_size=False, num_records_bytes=_x_nrec_bytes
        )
        # Z: bf16 (B, H, D), num_records = B*H*D*2 bytes.
        _z_nrec_bytes = fx.Int64(c_b * (H * D * elem_bytes_a))
        z_rsrc = buffer_ops.create_buffer_resource(
            arg_z, max_size=False, num_records_bytes=_z_nrec_bytes
        )
        # Y: max_size=True — actual size is known statically (H*D*R fp8 bytes).
        y_rsrc = buffer_ops.create_buffer_resource(arg_y, max_size=True)
        # sy: max_size=True.
        sy_rsrc = buffer_ops.create_buffer_resource(arg_sy, max_size=True)

        # ── Wave / lane decomposition (matches MFMA 16x16 layout) ───────────
        layout_wave_lane = fx.make_layout((4, wave_size), (64, 1))
        coord_wave_lane = fx.idx2crd(tx, layout_wave_lane)
        wave_id = fx.get(coord_wave_lane, 0)
        lane_id = fx.get(coord_wave_lane, 1)

        layout_lane16 = fx.make_layout((4, 16), (16, 1))
        coord_lane16 = fx.idx2crd(lane_id, layout_lane16)
        lane_div_16 = fx.get(coord_lane16, 0)
        lane_mod_16 = fx.get(coord_lane16, 1)

        row_a_lds = lane_mod_16
        # bf16: 8 elems per 16B; lane_div_16 ∈ {0..3} → byte offsets {0,16,32,48}
        col_offset_base_bytes = lane_div_16 * 16

        # ── N-tile column indexing (per wave) ───────────────────────────────
        n_tile_base = wave_id * n_per_wave
        n_blk_list = []
        n_intra_list = []
        for i in range_constexpr(num_acc_n):
            global_n_in_head = by_n + n_tile_base + (i * 16) + lane_mod_16
            n_blk_list.append(global_n_in_head // 16)
            n_intra_list.append(global_n_in_head % 16)

        # ── B preshuffle stride consts ──────────────────────────────────────
        _b_stride_n0_c = fx.Index(_stride_n0)
        _b_stride_k0_c = fx.Index(_stride_k0)
        _b_stride_klane_c = fx.Index(_stride_klane)
        _b_stride_nlane_c = fx.Index(_stride_nlane)

        # Per-head base byte offset in Y for this WG's h.
        # All four waves in a WG share the same `h`, so this is a uniform
        # value across the WG.
        y_head_byte_off = bx_h * fx.Index(b_elems_per_head)  # fp8: bytes==elems

        # ── A global→LDS pipeline (sync, bf16, simple stage-1) ──────────────
        # bf16: each 16B load = 8 bf16 elems. k chunks per lane in 4-dword units.
        tile_k_dwords = (tile_k * elem_bytes_a) // 4   # K bytes / 4 → dwords
        k_blocks16 = fx.Index((tile_k * elem_bytes_a) // 16)
        layout_a_tile_div4 = fx.make_layout(
            (tile_m, tile_k_dwords), (tile_k_dwords, 1)
        )
        c4 = fx.Index(4)
        tx_i32_base = tx * c4

        def a_tile_chunk_coord_i32(i: int):
            return tile_chunk_coord_i32(
                fx.arith,
                tx_i32_base=tx_i32_base,
                i=i,
                total_threads=total_threads,
                layout_tile_div4=layout_a_tile_div4,
            )

        # X global element stride per row (in bf16 elems):
        #   row = (b, h) within a head → stride = H*R (because layout is (B,H,R))
        # But within a workgroup, h is fixed (bx_h), and rows are the B rows for
        # that h. Adjacent rows in M = adjacent b values → byte stride = H*R*2.
        # Pointer base for this WG's A tile: arg_x + bx_h * R (elems).
        # Per-thread row offset: bx_m + row_a_local (in M = B).
        # Element index = (bx_m + row_a_local) * (H*R)  +  bx_h * R  +  k_elem.

        x_head_elem_off = bx_h * fx.Index(R)

        def load_a_tile_to_lds(base_k_elem):
            """Load (tile_m, tile_k) bf16 from X into LDS, sync.

            base_k_elem: starting K offset in bf16 elements within the head.
            """
            for i in range_constexpr(num_a_loads):
                row_a_local, col_dword_local = a_tile_chunk_coord_i32(i)
                col_byte_local = col_dword_local * c4
                col_elem_local = col_byte_local // 2  # bf16: 2 bytes/elem

                row_a_global = bx_m + row_a_local                  # in B
                # Element index into X (bf16 elems from base) — the helper
                # converts to bytes internally based on elem_bytes.
                idx_elem = (
                    row_a_global * fx.Index(stride_b_x_elems)
                    + x_head_elem_off
                    + (base_k_elem + col_elem_local)
                )

                # Issue 16B load → 8 bf16 elems
                a_16B = buffer_copy_gmem16_dwordx4(
                    buffer_ops, fx.vector,
                    elem_type=fx.BFloat16.ir_type,
                    idx_i32=fx.Int32(idx_elem),
                    rsrc=x_rsrc,
                    vec_elems=8,
                    elem_bytes=elem_bytes_a,
                )
                # Store into LDS with swizzle (same scheme as template)
                col_swz_bytes = swizzle_xor16(row_a_local, col_byte_local, k_blocks16)
                col_swz_elems = col_swz_bytes // 2
                # LDS index in bf16 elems:
                idx_lds = row_a_local * fx.Index(lds_stride_bytes // 2) + col_swz_elems
                v8 = Vec(a_16B).bitcast(fx.BFloat16)
                v8.store(lds_a, [idx_lds])

        # ── B global load (preshuffled, per-head) ───────────────────────────
        # k_bytes_b advances within the head; head base is added once.
        def load_b_tile(base_k_elem):
            """Load (tile_n, tile_k) fp8 B for current head & N-tile.

            Returns a list-of-(packs0,packs1) per K=64 chunk, matching the
            template's structure so future swap to the ping-pong driver is
            straightforward.
            base_k_elem: starting K offset (fp8 elems == bytes) within head.
            """
            k0_base = base_k_elem // 64
            # One MFMA per K=32 chunk, k_unroll MFMAs per tile, packed in pairs
            # of (b0,b1) per K=64 chunk. Return a flat list of k_unroll
            # packs-per-N-block — packs[ku] is the per-N i64 list for MFMA ku.
            packs_flat = [[] for _ in range(k_unroll)]

            for ni in range_constexpr(num_acc_n):
                # Base in Y in i32 dwords (we load as vec4 of i32).
                # n_base in fp8 bytes:
                n_base_byte = (
                    n_blk_list[ni] * _b_stride_n0_c
                    + lane_div_16 * _b_stride_klane_c
                    + n_intra_list[ni] * _b_stride_nlane_c
                ) + y_head_byte_off
                for ku64 in range_constexpr(k_unroll // 2):
                    k0 = fx.Index(k0_base) + fx.Index(ku64)
                    idx_byte = n_base_byte + k0 * _b_stride_k0_c
                    idx_dword = idx_byte // 4
                    # 16B = 16 fp8 elems = one (klane=4, kpack=16) entry,
                    # contributing 2 i64 packs covering K=64.
                    b_vec4_i32 = buffer_ops.buffer_load(
                        y_rsrc, fx.Int32(idx_dword),
                        vec_width=4, dtype=fx.Int32,
                    )
                    b16 = Vec(b_vec4_i32).bitcast(fp8_dtype)
                    b_i64x2 = Vec(b16).bitcast(fx.Int64)
                    b0_i64 = b_i64x2[0].ir_value()
                    b1_i64 = b_i64x2[1].ir_value()
                    packs_flat[ku64 * 2].append(b0_i64)
                    packs_flat[ku64 * 2 + 1].append(b1_i64)

            return packs_flat

        # ── A LDS load (per K=64 byte chunk) ────────────────────────────────
        _lds_k_dim_elems = fx.Index(tile_k)  # bf16 elems per row in LDS

        def lds_load_bf16_8elems(curr_row_a_lds, col_base_bytes):
            """Load 16B (= 8 bf16 elems) from LDS at given (row, col-byte)."""
            col_base_swz_bytes = swizzle_xor16(
                curr_row_a_lds, col_base_bytes, k_blocks16
            )
            col_base_swz_elems = col_base_swz_bytes // 2
            idx = curr_row_a_lds * _lds_k_dim_elems + col_base_swz_elems
            return Vec.load(Vec.make_type(8, fx.BFloat16), lds_a, [idx])

        # ── compute_tile primitives ──────────────────────────────────────────
        mfma_res_ty = Vec.make_type(4, fx.Float32)
        mfma_fp8 = rocdl.mfma_f32_16x16x32_fp8_fp8
        i32_ir = fx.Int32.ir_type
        f32_ir = fx.Float32.ir_type
        c0_i32 = fx.Int32(0)

        # K=32 per MFMA, 4 MFMAs per K=128 quant group.
        k_per_quant_group = 128
        mfmas_per_group = k_per_quant_group // 32        # = 4
        groups_per_tile = tile_k // k_per_quant_group    # ≥ 1
        # Per-lane K-elements within one 128-K group: 128 / 4 lane_groups = 32.
        per_lane_k_per_group = k_per_quant_group // 4    # = 32

        def pack_8_fp32_to_i64(scaled_fp32_v8):
            """8 scaled fp32 → 1 i64 (= 2 i32 dwords of 4 fp8 each)."""
            w0 = c0_i32
            w0 = rocdl.cvt_pk_fp8_f32(
                i32_ir, scaled_fp32_v8[0], scaled_fp32_v8[1], w0, 0,
            )
            w0 = rocdl.cvt_pk_fp8_f32(
                i32_ir, scaled_fp32_v8[2], scaled_fp32_v8[3], w0, 1,
            )
            w1 = c0_i32
            w1 = rocdl.cvt_pk_fp8_f32(
                i32_ir, scaled_fp32_v8[4], scaled_fp32_v8[5], w1, 0,
            )
            w1 = rocdl.cvt_pk_fp8_f32(
                i32_ir, scaled_fp32_v8[6], scaled_fp32_v8[7], w1, 1,
            )
            return Vec.from_elements(
                [w0, w1], fx.Int32,
            ).bitcast(fx.Int64)[0].ir_value()

        def cross_lane_amax(local_amax_f32):
            """2-step DPP butterfly: max across lane^16 then lane^32.

            Both swaps use `row_xmask` DPP control (rows are 16 lanes wide):
              row_xmask:1 → dpp_ctrl=0x101 → lanes swap across rows {0↔1, 2↔3}
              row_xmask:2 → dpp_ctrl=0x102 → lanes swap across rows {0↔2, 1↔3}
            Combined, all 4 lane_div_16 groups (rows 0..3 of the wave) hold
            the same max for each lane_mod_16 column.
            """
            from flydsl._mlir.dialects import rocdl as _rocdl_low
            local_i32 = fx.arith.bitcast(i32_ir, local_amax_f32.ir_value())
            # Step 1: swap with lane ^ 16
            sw1_i32 = _rocdl_low.update_dpp(
                res=i32_ir, old=local_i32, src=local_i32,
                dpp_ctrl=0x101, row_mask=0xF, bank_mask=0xF, bound_ctrl=True,
            )
            sw1_f32 = fx.arith.bitcast(f32_ir, sw1_i32)
            mid = fx.arith.maximumf(local_amax_f32.ir_value(), sw1_f32)
            # Step 2: swap with lane ^ 32
            sw2_i32 = _rocdl_low.update_dpp(
                res=i32_ir, old=fx.arith.bitcast(i32_ir, mid),
                src=fx.arith.bitcast(i32_ir, mid),
                dpp_ctrl=0x102, row_mask=0xF, bank_mask=0xF, bound_ctrl=True,
            )
            sw2_f32 = fx.arith.bitcast(f32_ir, sw2_i32)
            final = fx.arith.maximumf(mid, sw2_f32)
            return fx.Float32(final)

        def ds_bpermute_f32(src_lane_index_i32, val_f32):
            """ds_bpermute on fp32: each lane reads val from src_lane.

            src_lane_index_i32 is the per-lane SOURCE lane index (Int32),
            but ds_bpermute expects byte offset = lane_idx * 4.
            """
            from flydsl._mlir.dialects import rocdl as _rocdl_low
            idx_bytes = src_lane_index_i32 * fx.Int32(4)
            src_i32 = fx.arith.bitcast(i32_ir, val_f32.ir_value())
            permuted_i32 = _rocdl_low.ds_bpermute(
                res=i32_ir, index=idx_bytes.ir_value(), src=src_i32,
            )
            return fx.Float32(fx.arith.bitcast(f32_ir, permuted_i32))

        # Per-output-row source-lane mapping. For each ii in {0,1,2,3}, the
        # lane holding s_x for output row (mi*16 + lane_div_16*4 + ii) is
        # the input-quant lane with lane_mod_16 == lane_div_16*4 + ii and
        # lane_div_16 == 0. That lane's lane_id = lane_div_16*4 + ii.
        # (Same for any mi — s_x is recomputed per mi.)
        # We use the lane in lane_div_16=0 because all 4 lane_div_16 lanes
        # hold the same final s_x post-DPP; any of them is a valid source.
        def src_lane_for_ii(ii_const):
            # ii is a python int compile-time; source lane = lane_div_16*4 + ii
            return fx.Int32(lane_div_16 * 4 + ii_const)

        # Float32 constants reused inside the loop.
        c_inv_448 = fx.Float32(1.0 / 448.0)
        c_amax_clamp = fx.Float32(1.0e-4)
        c_zero_f32 = fx.Float32(0.0)
        c_one_f32 = fx.Float32(1.0)

        # sy strides (fp32, layout (H, D//128, R//128))
        _sy_per_head = (D // 128) * (R // 128)
        _sy_per_n128 = R // 128

        def compute_tile_fp32(global_accs_in, b_tile_in, base_k_elem_in_head):
            """Run one tile_k worth of compute with fp32-mode online quant.

            base_k_elem_in_head: starting K elem (in bf16 elems) of this tile
            within the head. Used to index arg_sy.
            """
            current_accs = list(global_accs_in)

            for g in range_constexpr(groups_per_tile):
                # ku range within this group: [g*4, g*4+4) for mfmas_per_group=4
                # Per-lane bf16 fragments for the whole 128-K group: 32 bf16.
                # We load 4 chunks of 8 bf16 (one per MFMA).
                a_fp32_chunks = []  # list of 4 Vec<8xfp32>, one per MFMA
                for ku_in_g in range_constexpr(mfmas_per_group):
                    ku_global = g * mfmas_per_group + ku_in_g
                    col_base_bytes = col_offset_base_bytes + (ku_global * 64)
                    # Note: this load is per-mi; we defer it until inside the
                    # mi loop below since `a_fp32_chunks` is per-mi.
                    a_fp32_chunks.append(col_base_bytes)
                # ^ Above: `a_fp32_chunks` here actually holds col_base_bytes
                # per ku for use inside mi loop. Renaming for clarity below.
                ku_col_bases = a_fp32_chunks

                # Compute s_y for each ni for this K-128 group.
                # arg_sy layout: (H, D//128, R//128) fp32.
                # Per lane n_block_g = (by_n + n_tile_base + ni*16 + lane_mod_16) // 128.
                # k_block_g = (base_k_elem_in_head + g*128) // 128.
                k_block_g = (base_k_elem_in_head + (g * k_per_quant_group)) // 128
                sy_per_ni = []
                for ni in range_constexpr(num_acc_n):
                    n_col_global = (
                        by_n + n_tile_base + (ni * 16) + lane_mod_16
                    )
                    n_block_g = n_col_global // 128
                    sy_idx = (
                        bx_h * fx.Index(_sy_per_head)
                        + n_block_g * fx.Index(_sy_per_n128)
                        + k_block_g
                    )
                    sy_val = buffer_ops.buffer_load(
                        sy_rsrc, fx.Int32(sy_idx),
                        vec_width=1, dtype=fx.Float32,
                    )
                    sy_per_ni.append(fx.Float32(sy_val))

                for mi in range_constexpr(m_repeat):
                    curr_row_a_lds = row_a_lds + (mi * 16)
                    # 4 LDS loads + fp32 promote, compute per-lane amax over
                    # the 32 elements held by this lane for input row L.
                    bf16_chunks = []
                    fp32_chunks = []
                    local_amax = c_zero_f32
                    for ku_in_g in range_constexpr(mfmas_per_group):
                        col_b = ku_col_bases[ku_in_g]
                        a_bf16 = lds_load_bf16_8elems(curr_row_a_lds, col_b)
                        a_fp32 = a_bf16.to(fx.Float32)
                        bf16_chunks.append(a_bf16)
                        fp32_chunks.append(a_fp32)
                        # amax over the 8 elements (abs).
                        for i in range_constexpr(8):
                            v = a_fp32[i]
                            # |v| via mulf with sign mask or just abs op.
                            # arith.absf may not be wrapped — use max(v, -v).
                            neg_v = c_zero_f32 - v
                            abs_v = fx.Float32(
                                fx.arith.maximumf(v.ir_value(), neg_v.ir_value())
                            )
                            local_amax = fx.Float32(
                                fx.arith.maximumf(
                                    local_amax.ir_value(), abs_v.ir_value()
                                )
                            )
                    # Cross-lane reduce: each lane gets the row's amax.
                    row_amax = cross_lane_amax(local_amax)
                    # s_x = max(amax, 1e-4) / 448 ; inv_s = 1/s_x
                    clamped = fx.Float32(
                        fx.arith.maximumf(
                            row_amax.ir_value(), c_amax_clamp.ir_value()
                        )
                    )
                    s_x = clamped * c_inv_448
                    inv_s = c_one_f32 / s_x

                    # Scratch acc (reset per K-128 group).
                    scratch_init = Vec.filled(4, 0.0, fx.Float32)
                    scratch_accs = [scratch_init] * num_acc_n

                    # Quant + 4 MFMAs into scratch.
                    for ku_in_g in range_constexpr(mfmas_per_group):
                        a_fp32_v8 = fp32_chunks[ku_in_g]
                        a_scaled = Vec.from_elements(
                            [a_fp32_v8[i] * inv_s for i in range(8)],
                            fx.Float32,
                        )
                        a_i64 = pack_8_fp32_to_i64(a_scaled)
                        ku_global = g * mfmas_per_group + ku_in_g
                        b_packs = b_tile_in[ku_global]
                        for ni in range_constexpr(num_acc_n):
                            scratch_accs[ni] = mfma_fp8(
                                mfma_res_ty,
                                [a_i64, b_packs[ni], scratch_accs[ni], 0, 0, 0],
                            )

                    # Promote scratch -> global: per-ii, fetch s_x for output
                    # row via ds_bpermute then multiply by (s_x * s_y) and add.
                    for ii in range_constexpr(4):
                        src_lane_i32 = src_lane_for_ii(ii)
                        s_x_for_out = ds_bpermute_f32(src_lane_i32, s_x)
                        for ni in range_constexpr(num_acc_n):
                            acc_idx = mi * num_acc_n + ni
                            sxsy = s_x_for_out * sy_per_ni[ni]
                            scratch_val = scratch_accs[ni][ii] * sxsy
                            new_val = current_accs[acc_idx][ii] + scratch_val
                            # Store back into the Vec — reconstruct via from_elements.
                            elems = [
                                current_accs[acc_idx][j] if j != ii else new_val
                                for j in range(4)
                            ]
                            current_accs[acc_idx] = Vec.from_elements(
                                elems, fx.Float32,
                            )
            return current_accs

        # ── Output store: direct, per-(B, H, D) layout ──────────────────────
        z_head_elem_off = bx_h * fx.Index(D)

        def store_output(final_accs):
            def body_row(*, mi, ii, row_in_tile, row):
                # row is the M-row global (= batch index b)
                col_base_n = by_n + n_tile_base + lane_mod_16
                # Z global element index:
                idx_base = (
                    row * fx.Index(stride_b_z_elems)
                    + z_head_elem_off
                    + col_base_n
                )
                for ni in range_constexpr(num_acc_n):
                    acc_idx = mi * num_acc_n + ni
                    acc = final_accs[acc_idx]
                    val = Vec(acc)[ii]
                    # compute_tile_fp32 already applied (s_x * s_y); store as-is.
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

        # ── Main K-loop (sequential, no double-buffer for step 1) ───────────
        acc_init = Vec.filled(4, 0.0, fx.Float32)
        accs = [acc_init] * (num_acc_n * m_repeat)

        num_tiles = R // tile_k

        # Prime: load A[0] into LDS, then loop.
        load_a_tile_to_lds(fx.Index(0))
        gpu.barrier()

        for kt in range_constexpr(num_tiles):
            base_k_elem = fx.Index(kt * tile_k)
            b_tile = load_b_tile(base_k_elem)
            accs = compute_tile_fp32(accs, b_tile, base_k_elem)
            if const_expr(kt + 1 < num_tiles):
                gpu.barrier()  # ensure prior LDS reads done before overwrite
                load_a_tile_to_lds(fx.Index((kt + 1) * tile_k))
                gpu.barrier()

        store_output(accs)

    # ── Host launcher ───────────────────────────────────────────────────────
    @flyc.jit
    def launch_gemm(
        arg_z: fx.Tensor,
        arg_x: fx.Tensor,
        arg_y: fx.Tensor,
        arg_sy: fx.Tensor,
        i32_b: fx.Int32,
        stream: fx.Stream,
    ):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        from flydsl._mlir import ir

        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()

        gx_per_h = (i32_b + (tile_m - 1)) // tile_m
        gx = gx_per_h * H
        gy = D // tile_n

        kernel_gemm._func.__name__ = KERNEL_NAME
        launcher = kernel_gemm(arg_z, arg_x, arg_y, arg_sy, i32_b)
        if waves_per_eu is not None:
            _wpe = int(waves_per_eu)
            if _wpe >= 1:
                for op in ctx.gpu_module_body.operations:
                    if hasattr(op, "attributes") and op.OPERATION_NAME == "gpu.func":
                        op.attributes["rocdl.waves_per_eu"] = ir.IntegerAttr.get(
                            fx.Int32.ir_type, _wpe
                        )

        launcher.launch(
            grid=(gx, gy, 1),
            block=(256, 1, 1),
            stream=stream,
        )

    return launch_gemm
