"""Preshuffle GEMM kernel using the @flyc.kernel API."""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.kernel_function import CompilationContext

from flydsl.expr import range_constexpr
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr

from flydsl._mlir import ir

from flydsl.expr import arith, vector
from flydsl.expr import gpu
from flydsl.expr import buffer_ops, rocdl


from flydsl.expr.typing import T
from flydsl.compiler.protocol import fly_values

from .layout_utils import crd2idx, idx2crd, get as layout_get

from .mfma_preshuffle_pipeline import (
    buffer_copy_gmem16_dwordx4,
    lds_load_pack_k32,
    lds_store_16b_xor16,
    make_preshuffle_b_layout,
    load_b_pack_k32,
    tile_chunk_coord_i32,
    swizzle_xor16,
    _buffer_load_vec,
)
from .mfma_epilogues import mfma_epilog


def compile_preshuffle_gemm_a8(
    *,
    M: int = 0,
    N: int = 0,
    K: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    in_dtype: str = "fp8",
    out_dtype: str = "fp16",
    lds_stage: int = 2,
    use_cshuffle_epilog: bool = False,
    waves_per_eu: int = None,
    use_async_copy: bool = False,
):
    """Compile the preshuffle GEMM kernel using the @flyc.kernel API.

    Returns a JitFunction that auto-compiles and executes when called.
    Signature:  launch_fn(arg_c, arg_a, arg_b, arg_scale_a, arg_scale_b, M, N, stream)

    Compile-time constants: K, tile_m/n/k, in_dtype, out_dtype (determine loop structure).
    Runtime parameters: M, N (passed as i32 kernel args).

    Args:
        out_dtype: Output element type, "fp16" or "bf16" (default: "fp16").
        waves_per_eu: Occupancy hint (None = default, 1-4 = limit occupancy).
        use_async_copy: Use async DMA for A tile global-to-LDS transfer.
    """
    if in_dtype not in ("fp8", "int8", "int4", "fp16", "bf16", "fp4"):
        raise ValueError(
            "in_dtype must be one of ('fp8','int8','int4','fp16','bf16','fp4'), "
            f"got {in_dtype!r}"
        )
    if out_dtype not in ("fp16", "bf16"):
        raise ValueError(
            f"out_dtype must be 'fp16' or 'bf16', got {out_dtype!r}"
        )
    _out_is_bf16 = out_dtype == "bf16"
    is_fp4 = in_dtype == "fp4"
    is_int4 = in_dtype == "int4"
    is_int8 = (in_dtype == "int8") or is_int4
    is_f16 = in_dtype == "fp16"
    is_bf16 = in_dtype == "bf16"
    is_f16_or_bf16 = is_f16 or is_bf16
    elem_bytes = 1 if (in_dtype in ("fp8", "int8", "int4", "fp4")) else 2
    a_elem_vec_pack = 2 if is_fp4 else 1
    b_elem_vec_pack = 2 if is_fp4 else 1

    tile_k_bytes = int(tile_k) * int(elem_bytes)

    if (tile_k_bytes % 64) != 0:
        raise ValueError(
            f"tile_k_bytes must be divisible by 64, got tile_k_bytes={tile_k_bytes} "
            f"(tile_k={tile_k}, elem_bytes={elem_bytes})"
        )

    _min_k_unroll = tile_k_bytes // a_elem_vec_pack // 64
    if is_fp4 and _min_k_unroll < 2:
        raise ValueError(
            f"FP4 requires tile_k >= {64 * 2 * a_elem_vec_pack} "
            f"(mfma_scale_f32_16x16x128 needs k_unroll >= 2), "
            f"got tile_k={tile_k} (k_unroll={_min_k_unroll})"
        )

    mfma_i32_k32 = None
    if is_int8:
        mfma_i32_k32 = getattr(rocdl, "mfma_i32_16x16x32i8", None) or getattr(
            rocdl, "mfma_i32_16x16x32_i8", None
        )
        if mfma_i32_k32 is None:
            raise AttributeError(
                "INT8 K32 MFMA op not found: expected `rocdl.mfma_i32_16x16x32i8` "
                "(or `rocdl.mfma_i32_16x16x32_i8`)."
            )

    gpu_arch = get_hip_arch()

    allocator_pong = SmemAllocator(None, arch=gpu_arch, global_sym_name="smem0")
    allocator_ping = SmemAllocator(None, arch=gpu_arch, global_sym_name="smem1")

    total_threads = 256
    bytes_a_per_tile = int(tile_m) * int(tile_k) * int(elem_bytes) // a_elem_vec_pack
    if bytes_a_per_tile % total_threads != 0:
        raise ValueError(
            "tile_m*tile_k*elem_bytes/a_elem_vec_pack must be divisible by "
            f"{total_threads}: tile_m={tile_m}, tile_k={tile_k}, elem_bytes={elem_bytes}, pack={a_elem_vec_pack}"
        )
    bytes_per_thread_a = bytes_a_per_tile // total_threads

    a_load_bytes = 16
    if bytes_per_thread_a % a_load_bytes != 0:
        raise ValueError(
            f"bytes_per_thread_a ({bytes_per_thread_a}) must be divisible by {a_load_bytes}"
        )
    a_async_load_bytes = 4 if gpu_arch == "gfx942" else 16
    a_async_load_dword = a_async_load_bytes // 4

    bytes_b_per_tile = int(tile_n) * int(tile_k) * int(elem_bytes) // b_elem_vec_pack
    bytes_per_thread_b = bytes_b_per_tile // total_threads
    b_load_bytes = 16
    num_b_loads = bytes_per_thread_b // b_load_bytes

    wave_size = 64
    num_a_lds_load = bytes_a_per_tile // wave_size // a_load_bytes
    num_a_async_loads = bytes_per_thread_a // a_async_load_bytes

    _is_gfx950 = str(gpu_arch).startswith("gfx950")
    _is_gfx942 = str(gpu_arch).startswith("gfx942")

    lds_stride_bytes = tile_k_bytes

    def _elem_type():
        if is_f16:
            return T.f16
        if is_bf16:
            return T.bf16
        if is_fp4:
            return T.i8
        return T.i8 if is_int8 else T.f8

    def _vec16_type():
        if is_f16:
            return T.f16x8
        if is_bf16:
            return T.bf16x8
        if is_fp4:
            return T.i8x16
        return T.i8x16 if is_int8 else T.f8x16

    def _mfma_pack_ty():
        if is_f16:
            return T.f16x4
        if is_bf16:
            return T.i16x4
        return T.i64

    def _out_elem():
        return T.bf16 if _out_is_bf16 else T.f16

    epilog_tag = "cshuffle" if use_cshuffle_epilog else "direct"
    module_name = f"mfma_preshuffle_{lds_stage}stages_{in_dtype}_{out_dtype}_{epilog_tag}".replace("-", "_")

    # ── LDS sizing (pure Python, no MLIR ops) ────────────────────────────────
    lds_tile_bytes = int(tile_m) * int(lds_stride_bytes) // a_elem_vec_pack
    lds_out_bytes = 2 * int(tile_m) * int(tile_n) if use_cshuffle_epilog else 0

    if int(lds_stage) == 2:
        assert lds_out_bytes % 2 == 0, "lds_out_bytes should be multiple of 2"
        buffer_size_bytes = max(lds_tile_bytes, lds_out_bytes // lds_stage)
        buffer_size_elems = buffer_size_bytes if elem_bytes == 1 else (buffer_size_bytes // 2)

        lds_pong_offset = allocator_pong._align(allocator_pong.ptr, 16)
        allocator_pong.ptr = lds_pong_offset + buffer_size_elems * elem_bytes

        lds_ping_offset = allocator_ping._align(allocator_ping.ptr, 16)
        allocator_ping.ptr = lds_ping_offset + buffer_size_elems * elem_bytes
    else:
        lds_total_bytes = max(lds_tile_bytes, lds_out_bytes)
        lds_total_elems = lds_total_bytes if elem_bytes == 1 else (lds_total_bytes // 2)

        lds_alloc_offset = allocator_pong._align(allocator_pong.ptr, 16)
        allocator_pong.ptr = lds_alloc_offset + lds_total_elems * elem_bytes

    # ── Kernel function ────────────────────────────────────────────────────
    @flyc.kernel
    def kernel_gemm(
        arg_c: fx.Tensor,
        arg_a: fx.Tensor,
        arg_b: fx.Tensor,
        arg_scale_a: fx.Tensor,
        arg_scale_b: fx.Tensor,
        i32_m: fx.Int32,
        i32_n: fx.Int32,
    ):
        c_m = arith.index_cast(T.index, i32_m)
        c_n = arith.index_cast(T.index, i32_n)

        # ---- Types ----
        acc_init = (
            arith.constant_vector(0, T.i32x4)
            if is_int8
            else arith.constant_vector(0.0, T.f32x4)
        )

        # ---- Layouts ----
        _k_div4_factor = (K * elem_bytes) // 4 // a_elem_vec_pack

        kpack_bytes = 8 if is_int4 else 16
        kpack_elems = kpack_bytes if elem_bytes == 1 else kpack_bytes // elem_bytes
        k_bytes_b = K * elem_bytes // b_elem_vec_pack
        k_bytes = K * elem_bytes
        n0_val = N // 16
        k0_val = k_bytes_b // 64
        _stride_nlane = kpack_elems
        _stride_klane = 16 * _stride_nlane
        _stride_k0 = 4 * _stride_klane
        _stride_n0 = k0_val * _stride_k0
        layout_b = fx.make_layout(
            (n0_val, k0_val, 4, 16, kpack_elems),
            (_stride_n0, _stride_k0, _stride_klane, _stride_nlane, 1),
        )

        lds_k_dim = tile_k // a_elem_vec_pack
        shape_lds = fx.make_shape(tile_m, lds_k_dim)
        stride_lds = fx.make_stride(lds_k_dim, 1)
        layout_lds = fx.make_layout(shape_lds, stride_lds)

        k_blocks16 = arith.index(tile_k_bytes // a_elem_vec_pack // 16)

        tx = gpu.thread_id("x")
        bx = gpu.block_id("x")
        by = gpu.block_id("y")

        # ---- LDS (separate ping/pong buffers for no-alias guarantee) ----
        base_ptr_pong = allocator_pong.get_base()
        base_ptr_ping = allocator_ping.get_base()

        if lds_stage == 2:
            lds_a_pong = SmemPtr(
                base_ptr_pong, lds_pong_offset, _elem_type(), shape=(tile_m * tile_k,)
            ).get()
            lds_a_ping = SmemPtr(
                base_ptr_ping, lds_ping_offset, _elem_type(), shape=(tile_m * tile_k,)
            ).get()

            if use_cshuffle_epilog:
                lds_out = SmemPtr(
                    base_ptr_pong, lds_pong_offset, _out_elem(), shape=(tile_m * tile_n,)
                ).get()
            else:
                lds_out = None
        else:
            lds_a_ptr = SmemPtr(
                base_ptr_pong, lds_alloc_offset, _elem_type(), shape=(lds_total_elems,)
            )
            lds_a_pong = lds_a_ptr.get()
            lds_a_ping = lds_a_pong
            lds_out = (
                SmemPtr(base_ptr_pong, lds_alloc_offset, _out_elem(), shape=(tile_m * tile_n,)).get()
                if use_cshuffle_epilog
                else None
            )

        # ---- Buffer resources (runtime byte sizes for OOB protection) ----
        _a_nrec = arith.index_cast(T.i64, c_m * (K * elem_bytes // a_elem_vec_pack))
        _c_nrec = arith.index_cast(T.i64, c_m * c_n * 2)
        a_rsrc = buffer_ops.create_buffer_resource(arg_a, max_size=False,
                                                   num_records_bytes=_a_nrec)
        c_rsrc = buffer_ops.create_buffer_resource(arg_c, max_size=False,
                                                   num_records_bytes=_c_nrec)
        _needs_per_token_scale = not is_f16_or_bf16 and not is_fp4
        scale_a_rsrc = None if (is_f16_or_bf16) else buffer_ops.create_buffer_resource(
            arg_scale_a, max_size=False)
        b_rsrc = buffer_ops.create_buffer_resource(arg_b, max_size=True)
        scale_b_rsrc = None if (is_f16_or_bf16) else buffer_ops.create_buffer_resource(
            arg_scale_b, max_size=True)

        bx_m = bx * tile_m
        by_n = by * tile_n

        # ---- Wave / lane decomposition ----
        wave_size = 64
        layout_wave_lane = fx.make_layout((4, wave_size), (64, 1))
        coord_wave_lane = idx2crd(tx, layout_wave_lane)
        wave_id = layout_get(coord_wave_lane, 0)
        lane_id = layout_get(coord_wave_lane, 1)

        layout_lane16 = fx.make_layout((4, 16), (16, 1))
        coord_lane16 = idx2crd(lane_id, layout_lane16)
        lane_div_16 = layout_get(coord_lane16, 0)
        lane_mod_16 = layout_get(coord_lane16, 1)

        row_a_lds = lane_mod_16
        kpack_elems = 16 if elem_bytes == 1 else 8
        col_offset_base = lane_div_16 * kpack_elems
        col_offset_base_bytes = (
            col_offset_base if elem_bytes == 1 else col_offset_base * elem_bytes
        )

        m_repeat = tile_m // 16
        k_unroll = tile_k_bytes // a_elem_vec_pack // 64

        num_waves = 4
        n_per_wave = tile_n // num_waves
        num_acc_n = n_per_wave // 16

        n_tile_base = wave_id * n_per_wave

        n_intra_list = []
        n_blk_list = []
        for i in range_constexpr(num_acc_n):
            global_n = by_n + n_tile_base + (i * 16) + lane_mod_16
            n_blk_list.append(global_n // 16)
            n_intra_list.append(global_n % 16)

        # ── B load helpers ────────────────────────────────────────────────
        def load_b_pack(base_k, ki_step, ni):
            return load_b_pack_k32(
                buffer_ops, arith, vector,
                arg_b=arg_b, b_rsrc=b_rsrc, layout_b=layout_b,
                base_k=base_k, ki_step=ki_step,
                n_blk=n_blk_list[ni], n_intra=n_intra_list[ni],
                lane_div_16=lane_div_16,
                elem_type=_elem_type(), kpack_bytes=kpack_bytes,
                elem_bytes=elem_bytes, unpack_int4=is_int4,
            )

        c64_b = 64
        c0_idx = 0

        _b_stride_n0_c = fx.Index(_stride_n0)
        _b_stride_k0_c = fx.Index(_stride_k0)
        _b_stride_klane_c = fx.Index(_stride_klane)
        _b_stride_nlane_c = fx.Index(_stride_nlane)

        _b_dword_stride_n0 = _stride_n0 // 4
        _b_dword_stride_k0 = _stride_k0 // 4
        _b_dword_stride_klane = _stride_klane // 4
        _b_dword_stride_nlane = _stride_nlane // 4

        _b_n_full_dword_list = []
        for _ni in range_constexpr(num_acc_n):
            _n_dword = (n_blk_list[_ni] * fx.Index(_b_dword_stride_n0)
                        + n_intra_list[_ni] * fx.Index(_b_dword_stride_nlane)
                        + lane_div_16 * fx.Index(_b_dword_stride_klane))
            _b_n_full_dword_list.append(_n_dword)

        _b_dword_stride_k0_c = fx.Index(_b_dword_stride_k0)
        _c64_elem = arith.index(64 // elem_bytes * b_elem_vec_pack) if (64 // elem_bytes * b_elem_vec_pack) != 64 else fx.Index(64)

        def _extract_b_packs(b16):
            b_i64x2 = vector.bitcast(T.i64x2, b16)
            b0_i64 = vector.extract(b_i64x2, static_position=[0], dynamic_position=[])
            b1_i64 = vector.extract(b_i64x2, static_position=[1], dynamic_position=[])
            if not is_f16_or_bf16:
                return b0_i64, b1_i64
            b0_v1 = vector.from_elements(T.vec(1, T.i64), [b0_i64])
            b1_v1 = vector.from_elements(T.vec(1, T.i64), [b1_i64])
            if is_f16:
                return vector.bitcast(T.f16x4, b0_v1), vector.bitcast(T.f16x4, b1_v1)
            return vector.bitcast(T.i16x4, b0_v1), vector.bitcast(T.i16x4, b1_v1)

        def _load_b_single(k_dword_offset, ni):
            """Load one 16B B vector using pre-computed k dword offset."""
            dword_idx = _b_n_full_dword_list[ni] + k_dword_offset
            dword_idx_i32 = arith.index_cast(T.i32, dword_idx)
            b_vec4 = buffer_ops.buffer_load(b_rsrc, dword_idx_i32, vec_width=4, dtype=T.i32)
            vec_elems = 16 if elem_bytes == 1 else 8
            b16 = vector.bitcast(T.vec(int(vec_elems), _elem_type()), b_vec4)
            return _extract_b_packs(b16)

        def load_b_packs_k64(base_k, ku: int, ni: int):
            if is_int4:
                ki0 = (ku * 2) + 0
                ki1 = (ku * 2) + 1
                return load_b_pack(base_k, ki0, ni), load_b_pack(base_k, ki1, ni)

            base_k_bytes = base_k * elem_bytes
            k0 = base_k_bytes // c64_b + ku
            idx_pack = n_blk_list[ni] * _b_stride_n0_c + k0 * _b_stride_k0_c + lane_div_16 * _b_stride_klane_c + n_intra_list[ni] * _b_stride_nlane_c
            vec_elems = 16 if elem_bytes == 1 else 8
            b16 = _buffer_load_vec(
                buffer_ops, vector, b_rsrc, idx_pack,
                elem_type=_elem_type(), vec_elems=vec_elems,
                elem_bytes=elem_bytes, offset_in_bytes=(elem_bytes == 1),
            )
            return _extract_b_packs(b16)

        def load_b_tile(base_k):
            if not is_int4 and not is_f16_or_bf16:
                base_k_bytes = base_k * elem_bytes
                k0_base = base_k_bytes // c64_b
                b_tile = []
                for ku in range_constexpr(k_unroll):
                    k0 = k0_base + ku
                    k_dword = k0 * _b_dword_stride_k0_c
                    packs0 = []
                    packs1 = []
                    for ni in range_constexpr(num_acc_n):
                        b0, b1 = _load_b_single(k_dword, ni)
                        packs0.append(b0)
                        packs1.append(b1)
                    b_tile.append((packs0, packs1))
                return b_tile

            b_tile = []
            for ku in range_constexpr(k_unroll):
                packs0 = []
                packs1 = []
                for ni in range_constexpr(num_acc_n):
                    b0, b1 = load_b_packs_k64(base_k, ku, ni)
                    packs0.append(b0)
                    packs1.append(b1)
                b_tile.append((packs0, packs1))
            return b_tile

        # ── A LDS load/store helpers (now take lds_buffer memref directly) ──
        lds_base_zero = fx.Index(0)

        _lds_k_dim_c = fx.Index(lds_k_dim)

        def lds_load_16b(curr_row_a_lds, col_base, lds_buffer):
            col_base_swz_bytes = swizzle_xor16(curr_row_a_lds, col_base, k_blocks16)
            col_base_swz = col_base_swz_bytes if elem_bytes == 1 else (col_base_swz_bytes // 2)
            idx_a16 = curr_row_a_lds * _lds_k_dim_c + col_base_swz
            return vector.load_op(_vec16_type(), lds_buffer, [idx_a16])

        def lds_load_packs_k64(curr_row_a_lds, col_base, lds_buffer):
            loaded_a16 = lds_load_16b(curr_row_a_lds, col_base, lds_buffer)
            a_i64x2 = vector.bitcast(T.i64x2, loaded_a16)
            a0_i64 = vector.extract(a_i64x2, static_position=[0], dynamic_position=[])
            a1_i64 = vector.extract(a_i64x2, static_position=[1], dynamic_position=[])

            if not is_f16_or_bf16:
                return a0_i64, a1_i64

            a0_v1 = vector.from_elements(T.vec(1, T.i64), [a0_i64])
            a1_v1 = vector.from_elements(T.vec(1, T.i64), [a1_i64])
            if is_f16:
                return vector.bitcast(T.f16x4, a0_v1), vector.bitcast(T.f16x4, a1_v1)
            return vector.bitcast(T.i16x4, a0_v1), vector.bitcast(T.i16x4, a1_v1)

        # ── A global→reg load ─────────────────────────────────────────────
        num_a_loads = bytes_per_thread_a // a_load_bytes
        tile_k_dwords = (tile_k * 2) // 4 if elem_bytes == 2 else tile_k // 4 // a_elem_vec_pack
        layout_a_tile_div4 = fx.make_layout((tile_m, tile_k_dwords), (tile_k_dwords, 1))
        c4 = fx.Index(4)
        tx_i32_base = tx * c4

        def load_a_16(idx_elem):
            return buffer_copy_gmem16_dwordx4(
                buffer_ops, vector,
                elem_type=_elem_type(),
                idx_i32=idx_elem,
                rsrc=a_rsrc, vec_elems=(16 if elem_bytes == 1 else 8),
                elem_bytes=elem_bytes,
            )

        def a_tile_chunk_coord_i32(i: int):
            return tile_chunk_coord_i32(
                arith, tx_i32_base=tx_i32_base, i=i,
                total_threads=total_threads, layout_tile_div4=layout_a_tile_div4,
            )

        def load_a_tile(base_k_div4):
            parts = []
            for i in range_constexpr(num_a_loads):
                row_a_local, col_a_local_i32 = a_tile_chunk_coord_i32(i)
                row_a_global = bx_m + row_a_local
                idx_i32 = row_a_global * _k_div4_factor + (base_k_div4 + col_a_local_i32)
                idx_elem = idx_i32 if elem_bytes == 1 else idx_i32 * 2
                a_16B = load_a_16(idx_elem)
                parts.append(vector.bitcast(T.i32x4, a_16B))
            return parts

        def store_a_tile_to_lds(vec_a_parts, lds_buffer):
            for i in range_constexpr(num_a_loads):
                row_a_local, col_a_local_i32 = a_tile_chunk_coord_i32(i)
                col_local_bytes = col_a_local_i32 * c4
                col_swz_bytes = swizzle_xor16(row_a_local, col_local_bytes, k_blocks16)
                col_swz = col_swz_bytes if elem_bytes == 1 else col_swz_bytes // 2
                idx0 = row_a_local * _lds_k_dim_c + col_swz + lds_base_zero
                v16 = vector.bitcast(_vec16_type(), vec_a_parts[i])
                vector.store(v16, lds_buffer, [idx0])

        # ── A DMA async: direct global→LDS transfer ─────────────────────
        num_a_async_loads = bytes_per_thread_a // a_async_load_bytes
        tx_i32_async_base = tx * a_async_load_dword
        k_bytes_factor = K * elem_bytes
        layout_a_bytes = fx.make_layout((tile_m, k_bytes_factor), (k_bytes_factor, 1))

        def a_tile_chunk_coord_i32_async(i: int):
            return tile_chunk_coord_i32(
                arith, tx_i32_base=tx_i32_async_base, i=i,
                total_threads=total_threads, layout_tile_div4=layout_a_tile_div4,
                chunk_i32=a_async_load_dword,
            )

        def dma_a_tile_to_lds(base_k_div4, lds_buffer):
            from flydsl._mlir.dialects import llvm, memref as memref_dialect

            dma_bytes = a_async_load_bytes

            for i in range_constexpr(num_a_async_loads):
                row_a_local, col_a_local_i32 = a_tile_chunk_coord_i32_async(i)
                col_a_local_sw = swizzle_xor16(row_a_local, col_a_local_i32 * c4, k_blocks16)
                row_a_global = bx_m + row_a_local
                global_byte_idx = row_a_global * k_bytes_factor + (base_k_div4 * c4 + col_a_local_sw)
                global_offset = arith.index_cast(T.i32, global_byte_idx)

                if i == 0:
                    lds_addr = memref_dialect.extract_aligned_pointer_as_index(lds_buffer) + wave_id * wave_size * dma_bytes
                    lds_ptr_i64_lane0 = rocdl.readfirstlane(T.i64, arith.index_cast(T.i64, lds_addr))
                else:
                    lds_ptr_i64_lane0 += total_threads * dma_bytes
                lds_ptr_type = ir.Type.parse("!llvm.ptr<3>")
                lds_ptr = llvm.inttoptr(lds_ptr_type, lds_ptr_i64_lane0)

                size_i32 = arith.constant(dma_bytes, type=T.i32)
                soffset = arith.constant(0, type=T.i32)
                offset_imm = arith.constant(0, type=T.i32)
                aux = arith.constant(1, type=T.i32)

                rocdl.raw_ptr_buffer_load_lds(
                    a_rsrc,
                    lds_ptr,
                    size_i32,
                    global_offset,
                    soffset,
                    offset_imm,
                    aux,
                )

        def prefetch_a_to_lds(base_k, lds_buffer):
            base_k_div4 = base_k // 4
            dma_a_tile_to_lds(base_k_div4, lds_buffer)

        def prefetch_a_tile(base_k):
            base_k_bytes = base_k * elem_bytes // a_elem_vec_pack
            base_k_div4 = base_k_bytes // 4
            return load_a_tile(base_k_div4)

        def prefetch_b_tile(base_k):
            base_k_packed = base_k // b_elem_vec_pack if b_elem_vec_pack > 1 else base_k
            return load_b_tile(base_k_packed)

        def prefetch_ab_tile(base_k):
            a_regs = prefetch_a_tile(base_k)
            b_regs = prefetch_b_tile(base_k)
            return a_regs, b_regs

        # ── FP4 scale pre-fetch (outside compute_tile for latency hiding) ──
        if is_fp4:
            from flydsl._mlir.dialects import fly as _fly_d, llvm as _llvm_d

            _fp4_pack_M_outer = 2
            _fp4_pack_N_outer = 2
            _fp4_pack_K_outer = 2
            _K1_outer = K // (32 * 4 * _fp4_pack_K_outer)
            _k_unroll_packed_outer = k_unroll // _fp4_pack_K_outer
            _m_repeat_packed_outer = m_repeat // _fp4_pack_M_outer
            _num_acc_n_packed_outer = num_acc_n // _fp4_pack_N_outer
            _fp4_scale_k_stride = tile_k // (32 * 4 * _fp4_pack_K_outer)
            _fp4_use_scheduler = (tile_m >= 64)

            _ptr_type = ir.Type.parse('!llvm.ptr<1>')
            _i32_type = T.i32
            _i64_type = T.i64
            _scale_a_raw = fly_values(arg_scale_a)[0]
            _scale_b_raw = fly_values(arg_scale_b)[0]
            _scale_a_ptr = _fly_d.extract_aligned_pointer_as_index(_ptr_type, _scale_a_raw)
            _scale_b_ptr = _fly_d.extract_aligned_pointer_as_index(_ptr_type, _scale_b_raw)
            _scale_a_ptr_int = _llvm_d.ptrtoint(_i64_type, _scale_a_ptr)
            _scale_b_ptr_int = _llvm_d.ptrtoint(_i64_type, _scale_b_ptr)

            _scale_lane_byte_off = arith.index_cast(T.i32, (lane_div_16 * fx.Index(16) + lane_mod_16) * fx.Index(4))
            _scale_lane_byte_i64 = _llvm_d.ZExtOp(_i64_type, _scale_lane_byte_off).result

            _scale_a_base_ptrs = []
            for mi in range_constexpr(_m_repeat_packed_outer):
                mni_a = fx.Index(mi) + bx_m // arith.index(_fp4_pack_M_outer * 16)
                mni_byte_off = _llvm_d.ZExtOp(_i64_type, arith.index_cast(T.i32, mni_a * arith.index(_K1_outer * 64 * 4))).result
                base = _llvm_d.AddOp(_scale_a_ptr_int, mni_byte_off, _llvm_d.IntegerOverflowFlags(0)).result
                base = _llvm_d.AddOp(base, _scale_lane_byte_i64, _llvm_d.IntegerOverflowFlags(0)).result
                _scale_a_base_ptrs.append(_llvm_d.inttoptr(_ptr_type, base))

            _scale_b_base_ptrs = []
            for ni in range_constexpr(_num_acc_n_packed_outer):
                mni_b = fx.Index(ni) + (by_n + n_tile_base) // arith.index(_fp4_pack_N_outer * 16)
                mni_byte_off = _llvm_d.ZExtOp(_i64_type, arith.index_cast(T.i32, mni_b * arith.index(_K1_outer * 64 * 4))).result
                base = _llvm_d.AddOp(_scale_b_ptr_int, mni_byte_off, _llvm_d.IntegerOverflowFlags(0)).result
                base = _llvm_d.AddOp(base, _scale_lane_byte_i64, _llvm_d.IntegerOverflowFlags(0)).result
                _scale_b_base_ptrs.append(_llvm_d.inttoptr(_ptr_type, base))

            _stride_k0_bytes = 64 * 4

            def load_fp4_scales(base_k_scale_idx):
                a_scales, b_scales = [], []
                for ku in range_constexpr(_k_unroll_packed_outer):
                    ku_byte_offset = arith.index_cast(T.i32, (base_k_scale_idx + fx.Index(ku)) * fx.Index(_stride_k0_bytes))
                    ku_byte_i64 = _llvm_d.ZExtOp(_i64_type, ku_byte_offset).result
                    for mi in range_constexpr(_m_repeat_packed_outer):
                        addr_i64 = _llvm_d.AddOp(_llvm_d.ptrtoint(_i64_type, _scale_a_base_ptrs[mi]), ku_byte_i64, _llvm_d.IntegerOverflowFlags(0)).result
                        addr_ptr = _llvm_d.inttoptr(_ptr_type, addr_i64)
                        a_scales.append(_llvm_d.load(_i32_type, addr_ptr, alignment=4))
                    for ni in range_constexpr(_num_acc_n_packed_outer):
                        addr_i64 = _llvm_d.AddOp(_llvm_d.ptrtoint(_i64_type, _scale_b_base_ptrs[ni]), ku_byte_i64, _llvm_d.IntegerOverflowFlags(0)).result
                        addr_ptr = _llvm_d.inttoptr(_ptr_type, addr_i64)
                        b_scales.append(_llvm_d.load(_i32_type, addr_ptr, alignment=4))
                return a_scales, b_scales

        # ── Compute tile (MFMA) ───────────────────────────────────────────
        def compute_tile(accs_in, b_tile_in, lds_buffer, *, is_last_tile=False, a0_prefetch=None, fp4_scales=None):
            scales_pf = {}
            if is_last_tile and (not is_f16_or_bf16):
                s_b_vals = []
                for ni in range_constexpr(num_acc_n):
                    col_g = by_n + n_tile_base + (ni * 16) + lane_mod_16
                    s_b_vals.append(
                        buffer_ops.buffer_load(scale_b_rsrc, col_g, vec_width=1, dtype=T.f32)
                    )
                scales_pf["s_b_vals"] = s_b_vals
                scales_pf["s_a_vecs"] = []
                row_off_base = lane_div_16 * 4
                for mi in range_constexpr(m_repeat):
                    row_base_m = bx_m + (mi * 16)
                    row_g_base = row_base_m + row_off_base
                    s_a_vec = buffer_ops.buffer_load(
                        scale_a_rsrc, row_g_base, vec_width=4, dtype=T.f32
                    )
                    scales_pf["s_a_vecs"].append(vector.bitcast(T.f32x4, s_a_vec))

            current_accs_list = list(accs_in)

            use_mfma_scale_128 = (
                str(gpu_arch).startswith("gfx95")
                and (not is_int8) and (not is_int4) and (not is_f16_or_bf16)
            )
            if use_mfma_scale_128:
                if (int(tile_k) % 128) != 0:
                    raise ValueError(
                        f"tile_k must be divisible by 128 for mfma_scale_x128, got tile_k={tile_k}"
                    )
                mfma_res_ty = T.f32x4
                vec4_i64 = T.vec(4, T.i64)
                vec8_i32 = T.vec(8, T.i32)
                c0_i64 = arith.constant(0, type=T.i64)

                _fp4_cbsz = 4 if is_fp4 else 0
                _fp4_blgp = 4 if is_fp4 else 0
                _fp4_pack_M = 2 if is_fp4 else 1
                _fp4_pack_N = 2 if is_fp4 else 1
                _fp4_pack_K = 2 if is_fp4 else 1
                _quant_block_size = 32
                _K1 = K // (_quant_block_size * 4 * _fp4_pack_K) if is_fp4 else 1
                _k_unroll_packed = k_unroll // _fp4_pack_K
                _m_repeat_packed = m_repeat // _fp4_pack_M
                _num_acc_n_packed = num_acc_n // _fp4_pack_N

                def pack_i64x4_to_i32x8(x0, x1, x2, x3):
                    v4 = vector.from_elements(vec4_i64, [x0, x1, x2, x3])
                    return vector.bitcast(vec8_i32, v4)

                if is_fp4:
                    _fp4_a_sc, _fp4_b_sc = fp4_scales if fp4_scales else ([], [])
                    for ku128 in range_constexpr(_k_unroll_packed):
                        for mi_p in range_constexpr(_m_repeat_packed):
                            a_scale_val = _fp4_a_sc[ku128 * _m_repeat_packed + mi_p]
                            for ni_p in range_constexpr(_num_acc_n_packed):
                                b_scale_val = _fp4_b_sc[ku128 * _num_acc_n_packed + ni_p]
                                for ikxdl in range_constexpr(_fp4_pack_K):
                                    k_idx = ku128 * _fp4_pack_K + ikxdl
                                    b_packs0, b_packs1 = b_tile_in[k_idx]
                                    col_base = col_offset_base_bytes + arith.index((k_idx * 128) // a_elem_vec_pack)
                                    for imxdl in range_constexpr(_fp4_pack_M):
                                        mi_idx = mi_p * _fp4_pack_M + imxdl
                                        curr_row_a_lds = row_a_lds + (mi_idx * 16)
                                        if (a0_prefetch is not None) and (k_idx == 0) and (mi_idx == 0):
                                            a0, a1 = a0_prefetch
                                        else:
                                            a0, a1 = lds_load_packs_k64(curr_row_a_lds, col_base, lds_buffer)
                                        a128 = pack_i64x4_to_i32x8(a0, a1, c0_i64, c0_i64)
                                        for inxdl in range_constexpr(_fp4_pack_N):
                                            ni_idx = ni_p * _fp4_pack_N + inxdl
                                            b0 = b_packs0[ni_idx]
                                            b1 = b_packs1[ni_idx]
                                            b128 = pack_i64x4_to_i32x8(b0, b1, c0_i64, c0_i64)
                                            acc_idx = mi_idx * num_acc_n + ni_idx
                                            if not _fp4_use_scheduler:
                                                rocdl.sched_barrier(0)
                                            current_accs_list[acc_idx] = rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                                                mfma_res_ty,
                                                [a128, b128, current_accs_list[acc_idx],
                                                 _fp4_cbsz, _fp4_blgp,
                                                 ikxdl * _fp4_pack_M + imxdl, a_scale_val,
                                                 ikxdl * _fp4_pack_N + inxdl, b_scale_val],
                                            )
                else:
                    for ku128 in range_constexpr(k_unroll // 2):
                        ku0 = ku128 * 2
                        ku1 = ku0 + 1
                        b0_packs0, b0_packs1 = b_tile_in[ku0]
                        b1_packs0, b1_packs1 = b_tile_in[ku1]
                        col_base0 = col_offset_base_bytes + (ku0 * 64)
                        col_base1 = col_offset_base_bytes + (ku1 * 64)

                        for mi in range_constexpr(m_repeat):
                            curr_row_a_lds = row_a_lds + (mi * 16)
                            if (a0_prefetch is not None) and (ku0 == 0) and (mi == 0):
                                a0, a1 = a0_prefetch
                            else:
                                a0, a1 = lds_load_packs_k64(curr_row_a_lds, col_base0, lds_buffer)
                            a2, a3 = lds_load_packs_k64(curr_row_a_lds, col_base1, lds_buffer)
                            a128 = pack_i64x4_to_i32x8(a0, a1, a2, a3)

                            for ni in range_constexpr(num_acc_n):
                                b128 = pack_i64x4_to_i32x8(
                                    b0_packs0[ni], b0_packs1[ni],
                                    b1_packs0[ni], b1_packs1[ni],
                                )
                                acc_idx = mi * num_acc_n + ni
                                current_accs_list[acc_idx] = rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                                    mfma_res_ty,
                                    [a128, b128, current_accs_list[acc_idx],
                                     0, 0, 0, 0x7F7F7F7F, 0, 0x7F7F7F7F],
                                )
                return current_accs_list, scales_pf

            mfma_res_ty = T.i32x4 if is_int8 else T.f32x4
            if is_int8:
                mfma_fn = mfma_i32_k32
            elif is_f16:
                mfma_fn = rocdl.mfma_f32_16x16x16f16
            elif is_bf16:
                mfma_fn = rocdl.mfma_f32_16x16x16bf16_1k
            else:
                mfma_fn = rocdl.mfma_f32_16x16x32_fp8_fp8

            def mfma_step(acc_in, a, b):
                return mfma_fn(mfma_res_ty, [a, b, acc_in, 0, 0, 0])

            def mfma_k64_bytes(acc_in, a0, a1, b0, b1):
                acc_mid = mfma_step(acc_in, a0, b0)
                return mfma_step(acc_mid, a1, b1)

            for ku in range_constexpr(k_unroll):
                b_packs0, b_packs1 = b_tile_in[ku]
                ki64 = ku * 64
                col_base = col_offset_base_bytes + ki64
                for mi in range_constexpr(m_repeat):
                    curr_row_a_lds = row_a_lds + (mi * 16)
                    if (a0_prefetch is not None) and (ku == 0) and (mi == 0):
                        a0, a1 = a0_prefetch
                    else:
                        a0, a1 = lds_load_packs_k64(curr_row_a_lds, col_base, lds_buffer)
                    for ni in range_constexpr(num_acc_n):
                        acc_idx = mi * num_acc_n + ni
                        current_accs_list[acc_idx] = mfma_k64_bytes(
                            current_accs_list[acc_idx], a0, a1,
                            b_packs0[ni], b_packs1[ni],
                        )
            return current_accs_list, scales_pf

        # ── Epilogue (store output) ───────────────────────────────────────
        vec1_f16 = T.vec(1, _out_elem())
        vec2_f16 = T.vec(2, _out_elem())
        vec1_i16 = T.vec(1, T.i16)
        vec2_i16 = T.i16x2
        vec1_i32 = T.vec(1, T.i32)

        def store_output(final_accs, scales):
            if is_f16_or_bf16 or is_fp4:
                s_b_vals = None
                s_a_vecs = None
            else:
                s_b_vals = scales["s_b_vals"]
                s_a_vecs = scales["s_a_vecs"]

            if use_cshuffle_epilog:
                if lds_out is None:
                    raise RuntimeError("use_cshuffle_epilog=True but lds_out is not allocated.")
                gpu.barrier()

                def write_row_to_lds(*, mi, ii, row_in_tile, row, row_base_lds,
                                     col_base_local, num_acc_n, lds_out):
                    if _needs_per_token_scale:
                        s_a_vec4 = s_a_vecs[mi]
                        s_a = vector.extract(s_a_vec4, static_position=[ii], dynamic_position=[])
                    for ni in range_constexpr(num_acc_n):
                        col_local = col_base_local + (ni * 16)
                        acc_idx = mi * num_acc_n + ni
                        acc = final_accs[acc_idx]
                        val = vector.extract(acc, static_position=[ii], dynamic_position=[])
                        if is_int8:
                            val = arith.sitofp(T.f32, val)
                        if is_f16_or_bf16 or is_fp4:
                            val_s = val
                        elif _needs_per_token_scale:
                            val_s = (val * s_a) * s_b_vals[ni]
                        else:
                            val_s = val
                        v16 = arith.trunc_f(_out_elem(), val_s)

                        lds_idx = row_base_lds + col_local
                        v1 = vector.from_elements(vec1_f16, [v16])
                        vector.store(v1, lds_out, [lds_idx], alignment=2)

                def store_pair(*, row_local, row, row_ctx, col_pair0, col_g0, frag):
                    idx_out = row * c_n + col_g0
                    byte_off = idx_out * 2
                    e_vec = 4 if (int(tile_n) % (32 * 4)) == 0 else 2
                    if e_vec == 4:
                        frag_i32x2 = vector.bitcast(T.vec(2, T.i32), frag)
                        buffer_ops.buffer_store(frag_i32x2, c_rsrc, byte_off, offset_is_bytes=True)
                    else:
                        frag_i32x1 = vector.bitcast(T.vec(1, T.i32), frag)
                        frag_i32 = vector.extract(frag_i32x1, static_position=[0], dynamic_position=[])
                        buffer_ops.buffer_store(frag_i32, c_rsrc, byte_off, offset_is_bytes=True)

                e_vec = 4 if (int(tile_n) % (32 * 4)) == 0 else 2
                mfma_epilog(
                    use_cshuffle=True, arith=arith, vector=vector, gpu=gpu,
                    range_constexpr=range_constexpr, tile_m=tile_m, tile_n=tile_n,
                    e_vec=e_vec, m_repeat=m_repeat, num_acc_n=num_acc_n, tx=tx,
                    lane_div_16=lane_div_16, lane_mod_16=lane_mod_16,
                    bx_m=bx_m, by_n=by_n, n_tile_base=n_tile_base, lds_out=lds_out,
                    write_row_to_lds=write_row_to_lds, store_pair=store_pair,
                    frag_elem_type=_out_elem(),
                )
                return

            def body_row(*, mi, ii, row_in_tile, row):
                if _needs_per_token_scale:
                    s_a_vec4 = s_a_vecs[mi]
                    s_a = vector.extract(s_a_vec4, static_position=[ii], dynamic_position=[])
                col_base = by_n + n_tile_base + lane_mod_16
                idx_base = row * c_n + col_base
                for ni in range_constexpr(num_acc_n):
                    acc_idx = mi * num_acc_n + ni
                    acc = final_accs[acc_idx]
                    val = vector.extract(acc, static_position=[ii], dynamic_position=[])
                    if is_int8:
                        val = arith.sitofp(T.f32, val)
                    if is_f16_or_bf16 or is_fp4:
                        val_s = val
                    elif _needs_per_token_scale:
                        val_s = (val * s_a) * s_b_vals[ni]
                    else:
                        val_s = val
                    val_f16 = arith.trunc_f(_out_elem(), val_s)
                    idx_out = idx_base + (ni * 16)
                    buffer_ops.buffer_store(val_f16, c_rsrc, idx_out)

            mfma_epilog(
                use_cshuffle=False, arith=arith, range_constexpr=range_constexpr,
                m_repeat=m_repeat, lane_div_16=lane_div_16, bx_m=bx_m, body_row=body_row,
            )

        # ── Scheduling hints ──────────────────────────────────────────────
        rocdl.sched_barrier(0)

        def hot_loop_scheduler():
            if is_fp4:
                if not _fp4_use_scheduler:
                    return
                mfma_group = _fp4_pack_N_outer
                mfma_total = _k_unroll_packed_outer * _m_repeat_packed_outer * _num_acc_n_packed_outer * _fp4_pack_K_outer * _fp4_pack_M_outer * _fp4_pack_N_outer
                mfma_per_iter = 2 * mfma_group
                sche_iters = 0 if mfma_per_iter == 0 else (mfma_total // mfma_per_iter)
                rocdl.sched_dsrd(2)
                rocdl.sched_mfma(1)
                rocdl.sched_mfma(1)
                dswr_tail = num_a_loads
                if dswr_tail > sche_iters:
                    dswr_tail = sche_iters
                dswr_start = sche_iters - dswr_tail
                for sche_i in range_constexpr(sche_iters):
                    rocdl.sched_vmem(1)
                    rocdl.sched_mfma(mfma_group)
                    rocdl.sched_dsrd(1)
                    rocdl.sched_mfma(mfma_group)
                    if sche_i >= dswr_start - 1:
                        rocdl.sched_dswr(1)
                rocdl.sched_barrier(0)
                return

            import math as _math

            def _build_scheduler(numer: int, denom: int):
                if denom <= 0:
                    return []
                if numer <= 0:
                    return [0] * denom
                out = []
                prev = 0
                for i in range_constexpr(denom):
                    cur = ((i + 1) * numer + (denom - 1)) // denom
                    out.append(cur - prev)
                    prev = cur
                return out

            if _is_gfx942 or (not use_async_copy):
                mfma_group = num_acc_n
                mfma_total = (k_unroll * 2) * m_repeat * mfma_group
                mfma_per_iter = 2 * mfma_group
                sche_iters = 0 if mfma_per_iter == 0 else (mfma_total // mfma_per_iter)
                rocdl.sched_dsrd(2)
                rocdl.sched_mfma(1)
                if tile_m == 16:
                    rocdl.sched_vmem(1)
                rocdl.sched_mfma(1)
                if tile_m == 16:
                    rocdl.sched_vmem(1)
                if num_acc_n < 4:
                    rocdl.sched_dsrd(1)
                    rocdl.sched_mfma(1)
                    if tile_m == 16:
                        rocdl.sched_vmem(1)
                    rocdl.sched_dsrd(1)
                    rocdl.sched_mfma(1)
                    if tile_m == 16:
                        rocdl.sched_vmem(1)
                    rocdl.sched_mfma(1)
                dswr_tail = num_a_loads
                dstr_advance = 2
                if dswr_tail > sche_iters:
                    dswr_tail = sche_iters
                dswr_start = max(sche_iters - dswr_tail - dstr_advance, 0)
                for sche_i in range_constexpr(sche_iters):
                    rocdl.sched_vmem(1)
                    rocdl.sched_mfma(mfma_group)
                    rocdl.sched_dsrd(1)
                    rocdl.sched_mfma(mfma_group)
                    if sche_i >= dswr_start - 1:
                        rocdl.sched_dswr(1)
            else:
                mfma_group = num_acc_n
                bytes_k_per_mfma = 128 if _is_gfx950 else 32
                num_mfma_per_tile_k = tile_k * elem_bytes // bytes_k_per_mfma
                mfma_total = num_mfma_per_tile_k * m_repeat * mfma_group
                num_ds_load = num_a_lds_load
                dswr_tail = num_a_loads
                dstr_advance = 2
                if dswr_tail > mfma_total:
                    dswr_tail = mfma_total
                num_gmem_loads = num_b_loads + num_a_async_loads
                dswr_start = max(mfma_total - dswr_tail - dstr_advance, 0)
                dsrd_preload = 2
                if dsrd_preload > num_ds_load:
                    dsrd_preload = num_ds_load
                dsrd_schedule = _build_scheduler(num_ds_load - dsrd_preload, mfma_total)
                vmem_schedule = _build_scheduler(num_gmem_loads, mfma_total)

                idx_ds_read = dsrd_preload
                idx_gmem_load = 0
                if dsrd_preload:
                    rocdl.sched_dsrd(dsrd_preload)
                for mfma_idx in range_constexpr(mfma_total):
                    rocdl.sched_mfma(1)
                    n_dsrd = dsrd_schedule[mfma_idx]
                    if n_dsrd and (idx_ds_read < num_ds_load):
                        if idx_ds_read + n_dsrd > num_ds_load:
                            n_dsrd = num_ds_load - idx_ds_read
                        if n_dsrd:
                            rocdl.sched_dsrd(n_dsrd)
                            idx_ds_read += n_dsrd

                    n_vmem = vmem_schedule[mfma_idx]
                    if n_vmem and (idx_gmem_load < num_gmem_loads):
                        if idx_gmem_load + n_vmem > num_gmem_loads:
                            n_vmem = num_gmem_loads - idx_gmem_load
                        if n_vmem:
                            rocdl.sched_vmem(n_vmem)
                            idx_gmem_load += n_vmem

            rocdl.sched_barrier(0)

        # ── Main pipeline ─────────────────────────────────────────────────
        def _flatten_b_tile(bt):
            flat = []
            for packs0, packs1 in bt:
                flat.extend(packs0)
                flat.extend(packs1)
            return flat

        def _unflatten_b_tile(flat):
            bt = []
            idx = 0
            for _ in range_constexpr(k_unroll):
                p0 = [flat[idx + ni] for ni in range_constexpr(num_acc_n)]
                idx += num_acc_n
                p1 = [flat[idx + ni] for ni in range_constexpr(num_acc_n)]
                idx += num_acc_n
                bt.append((p0, p1))
            return bt

        n_accs = num_acc_n * m_repeat
        n_btile = k_unroll * 2 * num_acc_n
        n_a0pf = 2

        def _pack_state(accs_l, bt_flat, a0pf):
            return list(accs_l) + list(bt_flat) + [a0pf[0], a0pf[1]]

        def _unpack_state(vals):
            accs_l = list(vals[:n_accs])
            bt_flat = list(vals[n_accs:n_accs + n_btile])
            a0pf = (vals[n_accs + n_btile], vals[n_accs + n_btile + 1])
            return accs_l, bt_flat, a0pf

        def _build_pingpong_body(k_iv, inner_state):
            accs_in, bt_flat_in, a0pf_in = _unpack_state(inner_state)
            b_tile_pong_in = _unflatten_b_tile(bt_flat_in)

            next_k1 = k_iv + tile_k
            _sc_pong = load_fp4_scales(k_iv // fx.Index(tile_k) * fx.Index(_fp4_scale_k_stride)) if is_fp4 else None
            if use_async_copy:
                prefetch_a_to_lds(next_k1, lds_a_ping)
            else:
                store_a_tile_to_lds(prefetch_a_tile(next_k1), lds_a_ping)
            b_tile_ping = prefetch_b_tile(next_k1)
            accs_in, _ = compute_tile(accs_in, b_tile_pong_in, lds_a_pong,
                                      a0_prefetch=a0pf_in, fp4_scales=_sc_pong)
            hot_loop_scheduler()
            rocdl.s_waitcnt(num_b_loads)
            gpu.barrier()
            a0_prefetch_ping = prefetch_a0_pack(lds_a_ping)

            next_k2 = k_iv + (tile_k * 2)
            _sc_ping = load_fp4_scales((k_iv + fx.Index(tile_k)) // fx.Index(tile_k) * fx.Index(_fp4_scale_k_stride)) if is_fp4 else None
            if use_async_copy:
                prefetch_a_to_lds(next_k2, lds_a_pong)
            else:
                store_a_tile_to_lds(prefetch_a_tile(next_k2), lds_a_pong)
            b_tile_pong_new = prefetch_b_tile(next_k2)
            accs_in, _ = compute_tile(accs_in, b_tile_ping, lds_a_ping,
                                      a0_prefetch=a0_prefetch_ping, fp4_scales=_sc_ping)
            hot_loop_scheduler()
            rocdl.s_waitcnt(num_b_loads)
            gpu.barrier()
            a0_prefetch_pong_new = prefetch_a0_pack(lds_a_pong)

            return _pack_state(accs_in, _flatten_b_tile(b_tile_pong_new),
                               a0_prefetch_pong_new)

        if lds_stage == 2:
            def prefetch_a0_pack(lds_buffer):
                return lds_load_packs_k64(row_a_lds, col_offset_base_bytes, lds_buffer)

            k0 = fx.Index(0)
            b_tile0 = prefetch_b_tile(k0)
            if use_async_copy:
                prefetch_a_to_lds(k0, lds_a_pong)
            else:
                store_a_tile_to_lds(prefetch_a_tile(k0), lds_a_pong)
            gpu.barrier()
            accs = [acc_init] * n_accs
            a0_prefetch_pong = prefetch_a0_pack(lds_a_pong)

            num_tiles = K // tile_k
            if (num_tiles % 2) == 1:
                c_k_main = K - tile_k
                init_state = _pack_state(accs, _flatten_b_tile(b_tile0),
                                         a0_prefetch_pong)
                for iv, inner in range(0, c_k_main, tile_k * 2, init=init_state):
                    results = yield _build_pingpong_body(iv, inner)
                accs, bt_flat, a0pf = _unpack_state(results)
                b_tile_pong_final = _unflatten_b_tile(bt_flat)
                _sc_last = load_fp4_scales(arith.index(((K // tile_k) - 1) * _fp4_scale_k_stride)) if is_fp4 else None
                final_accs, scales = compute_tile(
                    accs, b_tile_pong_final, lds_a_pong,
                    is_last_tile=not is_fp4, a0_prefetch=a0pf, fp4_scales=_sc_last,
                )
            else:
                c_k_stop = K - (tile_k * 3)
                init_state = _pack_state(accs, _flatten_b_tile(b_tile0),
                                         a0_prefetch_pong)
                for iv, inner in range(0, c_k_stop, tile_k * 2, init=init_state):
                    results = yield _build_pingpong_body(iv, inner)
                accs, bt_flat, a0pf = _unpack_state(results)
                b_tile_pong_ep = _unflatten_b_tile(bt_flat)

                last_k = arith.index(K - tile_k)
                b_tile_ping = prefetch_b_tile(last_k)
                if use_async_copy:
                    prefetch_a_to_lds(last_k, lds_a_ping)
                else:
                    a_regs_ping = prefetch_a_tile(last_k)
                _sc_ep = load_fp4_scales(arith.index(((K // tile_k) - 2) * _fp4_scale_k_stride)) if is_fp4 else None
                accs, _ = compute_tile(accs, b_tile_pong_ep, lds_a_pong,
                                       a0_prefetch=a0pf, fp4_scales=_sc_ep)
                if not use_async_copy:
                    store_a_tile_to_lds(a_regs_ping, lds_a_ping)
                hot_loop_scheduler()
                rocdl.s_waitcnt(num_b_loads)
                gpu.barrier()
                a0_prefetch_ping = prefetch_a0_pack(lds_a_ping)
                _sc_last2 = load_fp4_scales(arith.index(((K // tile_k) - 1) * _fp4_scale_k_stride)) if is_fp4 else None
                final_accs, scales = compute_tile(
                    accs, b_tile_ping, lds_a_ping,
                    is_last_tile=not is_fp4, a0_prefetch=a0_prefetch_ping, fp4_scales=_sc_last2,
                )
            store_output(final_accs, scales)
        else:
            a_regs0, b_tile0 = prefetch_ab_tile(fx.Index(0))
            store_a_tile_to_lds(a_regs0, lds_a_pong)
            gpu.barrier()
            accs = [acc_init] * n_accs
            bt_flat0 = _flatten_b_tile(b_tile0)

            init_state = list(accs) + list(bt_flat0)
            for iv, state in range(0, K - tile_k, tile_k, init=init_state):
                accs_in = list(state[:n_accs])
                bt_flat_in = list(state[n_accs:])
                b_tile_in = _unflatten_b_tile(bt_flat_in)

                next_k = iv + tile_k
                a_next, b_next = prefetch_ab_tile(next_k)
                _fp4_sc = load_fp4_scales(iv // fx.Index(tile_k) * fx.Index(_fp4_scale_k_stride)) if is_fp4 else None
                accs_in, _ = compute_tile(accs_in, b_tile_in, lds_a_pong, fp4_scales=_fp4_sc)
                gpu.barrier()
                store_a_tile_to_lds(a_next, lds_a_pong)
                hot_loop_scheduler()
                rocdl.s_waitcnt(num_b_loads)
                gpu.barrier()
                results = yield list(accs_in) + _flatten_b_tile(b_next)

            accs_final = list(results[:n_accs])
            bt_final = _unflatten_b_tile(list(results[n_accs:]))
            _last_fp4_sc = load_fp4_scales(arith.index((K - tile_k) // tile_k * _fp4_scale_k_stride)) if is_fp4 else None
            final_accs, scales = compute_tile(
                accs_final, bt_final, lds_a_pong, is_last_tile=not is_fp4, fp4_scales=_last_fp4_sc
            )
            store_output(final_accs, scales)

    # ── Host launcher ──────────────────────────────────────────────────────
    _cache_tag = (in_dtype, out_dtype, K, lds_stage, use_cshuffle_epilog)

    @flyc.jit
    def launch_gemm(
        arg_c: fx.Tensor,
        arg_a: fx.Tensor,
        arg_b: fx.Tensor,
        arg_scale_a: fx.Tensor,
        arg_scale_b: fx.Tensor,
        i32_m: fx.Int32,
        i32_n: fx.Int32,
        stream: fx.Stream,
    ):
        _ = _cache_tag
        allocator_pong.finalized = False
        allocator_ping.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator_pong.finalize()
            allocator_ping.finalize()

        gx = (i32_m + (tile_m - 1)) // tile_m
        gy = i32_n // tile_n

        launcher = kernel_gemm(arg_c, arg_a, arg_b, arg_scale_a, arg_scale_b, i32_m, i32_n)
        if waves_per_eu is not None:
            _wpe = int(waves_per_eu)
            if _wpe >= 1:
                for op in ctx.gpu_module_body.operations:
                    if hasattr(op, 'attributes') and op.OPERATION_NAME == "gpu.func":
                        op.attributes["rocdl.waves_per_eu"] = ir.IntegerAttr.get(
                            T.i32, _wpe)
        launcher.launch(
            grid=(gx, gy, 1),
            block=(256, 1, 1),
            stream=stream,
        )

    return launch_gemm


def compile_preshuffle_gemm_w4(
    *,
    M: int = 0,
    N: int = 0,
    K: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    a_dtype: str = "fp4",
    b_dtype: str = "fp4",
    out_dtype: str = "bf16",
    lds_stage: int = 2,
    use_cshuffle_epilog: bool = False,
):
    """MXFP4 preshuffle GEMM — delegates to compile_preshuffle_gemm_a8 with fp4 config."""
    if a_dtype == "fp8":
        raise NotImplementedError("fp8-A not yet supported with MXFP4 kernel (op_sel_a overflow)")
    if str(get_hip_arch()) != "gfx950":
        raise RuntimeError(f"FP4 GEMM requires gfx950, got {get_hip_arch()}")

    inner = compile_preshuffle_gemm_a8(
        M=M, N=N, K=K,
        tile_m=tile_m, tile_n=tile_n, tile_k=tile_k,
        in_dtype="fp4",
        out_dtype=out_dtype,
        lds_stage=1,
        use_cshuffle_epilog=use_cshuffle_epilog,
    )
    return inner


__all__ = ["compile_preshuffle_gemm_a8", "compile_preshuffle_gemm_w4"]
