"""Attn bwd kernel using the @flyc.kernel API."""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.kernel_function import CompilationContext

from flydsl.expr import range_constexpr
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr

from flydsl._mlir import ir
from flydsl._mlir.dialects import scf

from flydsl.expr import arith, vector, math as fx_math, const_expr
from flydsl.expr import gpu
from flydsl.expr import buffer_ops, rocdl
from flydsl.expr.typing import T

from aiter.ops.flydsl.kernels.mfma_preshuffle_pipeline import (
    buffer_copy_gmem16_dwordx4,
    tile_chunk_coord_i32,
    swizzle_xor16,
)


def lds_transpose_load(lds_memref, elem_offset):
    """Transpose-load from LDS memref via ds_read_tr8_b64 (gfx950).

    Args:
        lds_memref:  LDS memref value (address-space 3), typically from
                     ``SmemPtr.get()`` or ``get_op_result_or_value(...)``.
        elem_offset: Per-lane linearized element offset into the memref
                     (ArithValue / ir.Value of index type / Python int).

    Returns:
        Loaded and transposed vector ``ir.Value``.
    """
    from flydsl._mlir.dialects import llvm, memref
    from flydsl.expr.arith import _to_raw
    from flydsl.expr.utils.arith import ArithValue as AV

    lds_ptr_ty = ir.Type.parse("!llvm.ptr<3>")
    raw_memref = arith.unwrap(lds_memref)
    lds_base = memref.extract_aligned_pointer_as_index(raw_memref)

    byte_off = AV(arith.unwrap(elem_offset, index=True))
    total_byte_idx = AV(lds_base) + byte_off
    addr_i32 = _to_raw(arith.index_cast(T.i32, total_byte_idx))
    ptr_val = llvm.inttoptr(lds_ptr_ty, addr_i32)

    result_type = T.i32x2
    result = llvm.call_intrinsic(
        result_type, "llvm.amdgcn.ds.read.tr8.b64", [ptr_val], [], []
    )
    return result


def compile_attn_bwd_mxfp8_gfx950(
    *,
    num_heads_q: int,
    num_heads_kv: int,
    seqlen: int,
    head_dim: int,
    tile_m: int,
    tile_n: int,
    tile_head: int,
    sm_scale: float,
    causal: bool = False,
    waves_per_eu: int = None,
):
    """Compile the attention backward mx8 kernel using the @flyc.kernel API.

    Returns a JitFunction that auto-compiles and executes when called.
    Compile-time constants: seqlen, head_dim, tile_m/n/head
    Runtime parameters: batch

    """

    elem_bytes = 1
    tile_head_mx = tile_head // 32
    tile_m_mx = tile_m // 32
    tile_n_mx = tile_n // 32
    gqa_size = num_heads_q // num_heads_kv
    seqlen_rounded = ((seqlen + tile_m - 1) // tile_m) * tile_m

    gpu_arch = get_hip_arch()

    allocator_pong = SmemAllocator(None, arch=gpu_arch, global_sym_name="smem0")
    allocator_ping = SmemAllocator(None, arch=gpu_arch, global_sym_name="smem1")
    allocator_k = SmemAllocator(None, arch=gpu_arch, global_sym_name="smem_k")
    allocator_v = SmemAllocator(None, arch=gpu_arch, global_sym_name="smem_v")
    allocator_v_scale = SmemAllocator(
        None, arch=gpu_arch, global_sym_name="smem_v_scale"
    )
    allocator_ppt_shuffle = SmemAllocator(
        None, arch=gpu_arch, global_sym_name="smem_ppt_shuffle"
    )
    allocator_ppt_scale_shuffle = SmemAllocator(
        None, arch=gpu_arch, global_sym_name="smem_ppt_scale_shuffle"
    )
    allocator_dst_shuffle = SmemAllocator(
        None, arch=gpu_arch, global_sym_name="smem_dst_shuffle"
    )
    allocator_dst_scale_shuffle = SmemAllocator(
        None, arch=gpu_arch, global_sym_name="smem_dst_scale_shuffle"
    )
    allocator_ds_shuffle = SmemAllocator(
        None, arch=gpu_arch, global_sym_name="smem_ds_shuffle"
    )
    allocator_ds_scale_shuffle = SmemAllocator(
        None, arch=gpu_arch, global_sym_name="smem_ds_scale_shuffle"
    )

    wave_size = 64
    total_threads = 256

    bytes_per_tile_qo = int(tile_m) * int(tile_head)
    bytes_per_thread_qo = bytes_per_tile_qo // total_threads
    qo_load_bytes = 16

    bytes_per_tile_kv = int(tile_n) * int(tile_head)
    bytes_per_thread_kv = bytes_per_tile_kv // total_threads
    kv_load_bytes = 16

    bytes_per_tile_qo_scale = (int(tile_m) * int(tile_head)) // 32
    bytes_per_thread_qo_scale = max(1, bytes_per_tile_qo_scale // total_threads)

    bytes_per_tile_kv_scale = (int(tile_n) * int(tile_head)) // 32
    bytes_per_thread_kv_scale = max(1, bytes_per_tile_kv_scale // total_threads)

    def _elem_type():
        return T.f8

    def _vec16_type():
        return T.f8x16

    # ── LDS sizing (pure Python, no MLIR ops) ────────────────────────────────
    lds_qo_tile_bytes = int(tile_m) * int(tile_head)
    lds_k_tile_bytes = int(tile_n) * int(tile_head)
    lds_v_tile_bytes = int(tile_n) * int(tile_head)
    lds_v_scale_tile_bytes = int(tile_n) * int(tile_head_mx)
    lds_ppt_tile_bytes = int(tile_n) * int(tile_m)
    lds_ppt_scale_tile_bytes = int(tile_n) * int(tile_m_mx)
    lds_dst_tile_bytes = int(tile_n) * int(tile_m)
    lds_dst_scale_tile_bytes = int(tile_n) * int(tile_m_mx)
    lds_ds_tile_bytes = int(tile_m) * int(tile_n)
    lds_ds_scale_tile_bytes = int(tile_m) * int(tile_n_mx)

    buffer_size_bytes = lds_qo_tile_bytes * 2  # + lds_qo_scale_tile_bytes * 4

    lds_pong_offset = allocator_pong._align(allocator_pong.ptr, 16)
    allocator_pong.ptr = lds_pong_offset + buffer_size_bytes
    lds_q_pong_offset = lds_pong_offset
    lds_do_pong_offset = lds_q_pong_offset + lds_qo_tile_bytes

    lds_ping_offset = allocator_ping._align(allocator_ping.ptr, 16)
    allocator_ping.ptr = lds_ping_offset + buffer_size_bytes
    lds_q_ping_offset = lds_ping_offset
    lds_do_ping_offset = lds_q_ping_offset + lds_qo_tile_bytes

    lds_k_offset = allocator_k._align(allocator_k.ptr, 16)
    allocator_k.ptr = lds_k_offset + lds_k_tile_bytes

    lds_v_offset = allocator_v._align(allocator_v.ptr, 16)
    allocator_v.ptr = lds_v_offset + lds_v_tile_bytes

    lds_v_scale_offset = allocator_v_scale._align(allocator_v_scale.ptr, 16)
    allocator_v_scale.ptr = lds_v_scale_offset + lds_v_scale_tile_bytes

    lds_ppt_shuffle_offset = allocator_ppt_shuffle._align(allocator_ppt_shuffle.ptr, 16)
    allocator_ppt_shuffle.ptr = lds_ppt_shuffle_offset + lds_ppt_tile_bytes

    lds_ppt_scale_shuffle_offset = allocator_ppt_scale_shuffle._align(
        allocator_ppt_scale_shuffle.ptr, 16
    )
    allocator_ppt_scale_shuffle.ptr = (
        lds_ppt_scale_shuffle_offset + lds_ppt_scale_tile_bytes
    )

    lds_dst_shuffle_offset = allocator_dst_shuffle._align(allocator_dst_shuffle.ptr, 16)
    allocator_dst_shuffle.ptr = lds_dst_shuffle_offset + lds_dst_tile_bytes

    lds_dst_scale_shuffle_offset = allocator_dst_scale_shuffle._align(
        allocator_dst_scale_shuffle.ptr, 16
    )
    allocator_dst_scale_shuffle.ptr = (
        lds_dst_scale_shuffle_offset + lds_dst_scale_tile_bytes
    )

    lds_ds_shuffle_offset = allocator_ds_shuffle._align(allocator_ds_shuffle.ptr, 16)
    allocator_ds_shuffle.ptr = lds_ds_shuffle_offset + lds_ds_tile_bytes

    lds_ds_scale_shuffle_offset = allocator_ds_scale_shuffle._align(
        allocator_ds_scale_shuffle.ptr, 16
    )
    allocator_ds_scale_shuffle.ptr = (
        lds_ds_scale_shuffle_offset + lds_ds_scale_tile_bytes
    )

    # ── Kernel function ────────────────────────────────────────────────────
    @flyc.kernel
    def kernel_attn_bwd(
        arg_dq: fx.Tensor,
        arg_dk: fx.Tensor,
        arg_dv: fx.Tensor,
        arg_q: fx.Tensor,
        arg_q_scale: fx.Tensor,
        arg_k: fx.Tensor,
        arg_k_scale: fx.Tensor,
        arg_v: fx.Tensor,
        arg_v_scale: fx.Tensor,
        arg_do: fx.Tensor,
        arg_do_scale: fx.Tensor,
        arg_M: fx.Tensor,
        arg_D: fx.Tensor,
        batch: fx.Int32,
        stride_qo_batch: fx.Int32,
        stride_kv_batch: fx.Int32,
        stride_MD_batch: fx.Int32,
        stride_qkvo_nheads: fx.Int32,
        stride_MD_nheads: fx.Int32,
        stride_q_scale_batch: fx.Int32,
        stride_q_scale_nheads: fx.Int32,
        stride_k_scale_batch: fx.Int32,
        stride_k_scale_nheads: fx.Int32,
        stride_v_scale_batch: fx.Int32,
        stride_v_scale_nheads: fx.Int32,
        stride_do_scale_batch: fx.Int32,
        stride_do_scale_nheads: fx.Int32,
    ):

        # ---- Types ----
        zero_f = arith.constant(0.0, type=T.f32)
        acc_init = arith.constant_vector(0.0, T.f32x4)
        log2e = arith.constant(1.4426950408889634, type=T.f32)
        c_sm_scale = arith.constant(sm_scale, type=T.f32)
        fp8_max_rcp = arith.constant(1.0 / 448.0, type=T.f32)

        tx = gpu.thread_id("x")
        bx = gpu.block_id("x")
        by = gpu.block_id("y")
        bz = gpu.block_id("z")
        batch_id = bz
        head_q = bx
        head_kv = head_q // gqa_size

        # ---- LDS (separate ping/pong buffers) ----
        base_ptr_pong = allocator_pong.get_base()
        base_ptr_ping = allocator_ping.get_base()
        base_ptr_k = allocator_k.get_base()
        base_ptr_v = allocator_v.get_base()
        base_ptr_v_scale = allocator_v_scale.get_base()
        base_ptr_ppt_shuffle = allocator_ppt_shuffle.get_base()
        base_ptr_ppt_scale_shuffle = allocator_ppt_scale_shuffle.get_base()
        base_ptr_dst_shuffle = allocator_dst_shuffle.get_base()
        base_ptr_dst_scale_shuffle = allocator_dst_scale_shuffle.get_base()
        base_ptr_ds_shuffle = allocator_ds_shuffle.get_base()
        base_ptr_ds_scale_shuffle = allocator_ds_scale_shuffle.get_base()

        lds_q_pong = SmemPtr(
            base_ptr_pong,
            lds_q_pong_offset,
            T.f8,
            shape=(tile_m * tile_head,),
        ).get()
        lds_q_ping = SmemPtr(
            base_ptr_ping,
            lds_q_ping_offset,
            T.f8,
            shape=(tile_m * tile_head,),
        ).get()
        lds_do_pong = SmemPtr(
            base_ptr_pong,
            lds_do_pong_offset,
            T.f8,
            shape=(tile_m * tile_head,),
        ).get()
        lds_do_ping = SmemPtr(
            base_ptr_ping,
            lds_do_ping_offset,
            T.f8,
            shape=(tile_m * tile_head,),
        ).get()
        lds_k = SmemPtr(
            base_ptr_k,
            lds_k_offset,
            T.f8,
            shape=(tile_n * tile_head,),
        ).get()
        lds_v = SmemPtr(
            base_ptr_v, lds_v_offset, T.f8, shape=(tile_n * tile_head,)
        ).get()
        lds_v_scale = SmemPtr(
            base_ptr_v_scale, lds_v_scale_offset, T.i8, shape=(tile_n * tile_head_mx,)
        ).get()
        lds_ppt_shuffle = SmemPtr(
            base_ptr_ppt_shuffle, lds_ppt_shuffle_offset, T.f8, shape=(tile_n * tile_m,)
        ).get()
        lds_ppt_scale_shuffle = SmemPtr(
            base_ptr_ppt_scale_shuffle,
            lds_ppt_scale_shuffle_offset,
            T.i8,
            shape=(tile_n * tile_m_mx,),
        ).get()
        lds_dst_shuffle = SmemPtr(
            base_ptr_dst_shuffle, lds_dst_shuffle_offset, T.f8, shape=(tile_n * tile_m,)
        ).get()
        lds_dst_scale_shuffle = SmemPtr(
            base_ptr_dst_scale_shuffle,
            lds_dst_scale_shuffle_offset,
            T.i8,
            shape=(tile_n * tile_m_mx,),
        ).get()
        lds_ds_shuffle = SmemPtr(
            base_ptr_ds_shuffle, lds_ds_shuffle_offset, T.f8, shape=(tile_m * tile_n,)
        ).get()
        lds_ds_scale_shuffle = SmemPtr(
            base_ptr_ds_scale_shuffle,
            lds_ds_scale_shuffle_offset,
            T.i8,
            shape=(tile_m * tile_n_mx,),
        ).get()

        offset_qo_nheads = batch_id * fx.Index(stride_qo_batch) + head_q * fx.Index(
            stride_qkvo_nheads
        )
        offset_dq_nheads = offset_qo_nheads * 4
        offset_kv_nheads = batch_id * fx.Index(stride_kv_batch) + head_kv * fx.Index(
            stride_qkvo_nheads
        )
        offset_dkdv_nheads = offset_kv_nheads * 4
        offset_q_scale_nheads = batch_id * fx.Index(
            stride_q_scale_batch
        ) + head_q * fx.Index(stride_q_scale_nheads)
        offset_k_scale_nheads = batch_id * fx.Index(
            stride_k_scale_batch
        ) + head_kv * fx.Index(stride_k_scale_nheads)
        offset_v_scale_nheads = batch_id * fx.Index(
            stride_v_scale_batch
        ) + head_kv * fx.Index(stride_v_scale_nheads)
        offset_do_scale_nheads = batch_id * fx.Index(
            stride_do_scale_batch
        ) + head_q * fx.Index(stride_do_scale_nheads)
        offset_MD_nheads = (
            batch_id * fx.Index(stride_MD_batch) + head_q * fx.Index(stride_MD_nheads)
        ) * 4

        # ---- Buffer resources (runtime byte sizes for OOB protection) ----
        head_dim_mx = head_dim // 32
        seqlen_mx = seqlen // 32
        global_buffer_size_tensor = fx.Index(seqlen * head_dim)
        global_buffer_size_scale = fx.Index(seqlen * head_dim_mx)
        global_buffer_size_scale_2d = fx.Index(seqlen_mx * head_dim_mx)
        q_nrec = arith.index_cast(T.i64, global_buffer_size_tensor)
        q_scale_nrec = arith.index_cast(T.i64, global_buffer_size_scale_2d)
        k_nrec = arith.index_cast(T.i64, global_buffer_size_tensor)
        k_scale_nrec = arith.index_cast(T.i64, global_buffer_size_scale_2d)
        v_nrec = arith.index_cast(T.i64, global_buffer_size_tensor)
        v_scale_nrec = arith.index_cast(T.i64, global_buffer_size_scale)
        do_nrec = arith.index_cast(T.i64, global_buffer_size_tensor)
        do_scale_nrec = arith.index_cast(T.i64, global_buffer_size_scale_2d)
        output_nrec = arith.index_cast(T.i64, global_buffer_size_tensor * 4)
        MD_nrec = arith.index_cast(T.i64, fx.Index(seqlen * 4))

        q_rsrc = buffer_ops.create_buffer_resource(
            arg_q,
            max_size=False,
            num_records_bytes=q_nrec,
            base_byte_offset=offset_qo_nheads,
        )
        q_scale_rsrc = buffer_ops.create_buffer_resource(
            arg_q_scale,
            max_size=False,
            num_records_bytes=q_scale_nrec,
            base_byte_offset=offset_q_scale_nheads,
        )
        k_rsrc = buffer_ops.create_buffer_resource(
            arg_k,
            max_size=False,
            num_records_bytes=k_nrec,
            base_byte_offset=offset_kv_nheads,
        )
        k_scale_rsrc = buffer_ops.create_buffer_resource(
            arg_k_scale,
            max_size=False,
            num_records_bytes=k_scale_nrec,
            base_byte_offset=offset_k_scale_nheads,
        )
        v_rsrc = buffer_ops.create_buffer_resource(
            arg_v,
            max_size=False,
            num_records_bytes=v_nrec,
            base_byte_offset=offset_kv_nheads,
        )
        v_scale_rsrc = buffer_ops.create_buffer_resource(
            arg_v_scale,
            max_size=False,
            num_records_bytes=v_scale_nrec,
            base_byte_offset=offset_v_scale_nheads,
        )
        do_rsrc = buffer_ops.create_buffer_resource(
            arg_do,
            max_size=False,
            num_records_bytes=do_nrec,
            base_byte_offset=offset_qo_nheads,
        )
        do_scale_rsrc = buffer_ops.create_buffer_resource(
            arg_do_scale,
            max_size=False,
            num_records_bytes=do_scale_nrec,
            base_byte_offset=offset_do_scale_nheads,
        )
        dq_rsrc = buffer_ops.create_buffer_resource(
            arg_dq,
            max_size=False,
            num_records_bytes=output_nrec,
            base_byte_offset=offset_dq_nheads,
        )
        dk_rsrc = buffer_ops.create_buffer_resource(
            arg_dk,
            max_size=False,
            num_records_bytes=output_nrec,
            base_byte_offset=offset_dkdv_nheads,
        )
        dv_rsrc = buffer_ops.create_buffer_resource(
            arg_dv,
            max_size=False,
            num_records_bytes=output_nrec,
            base_byte_offset=offset_dkdv_nheads,
        )
        M_rsrc = buffer_ops.create_buffer_resource(
            arg_M,
            max_size=False,
            num_records_bytes=MD_nrec,
            base_byte_offset=offset_MD_nheads,
        )
        D_rsrc = buffer_ops.create_buffer_resource(
            arg_D,
            max_size=False,
            num_records_bytes=MD_nrec,
            base_byte_offset=offset_MD_nheads,
        )

        global_offset_n = by * tile_n
        global_offset_n_mx = global_offset_n // 32

        # ---- Wave / lane decomposition ----
        layout_wave_lane = fx.make_layout((4, wave_size), (64, 1))
        coord_wave_lane = fx.idx2crd(tx, layout_wave_lane)
        wave_id = fx.get(coord_wave_lane, 0)
        lane_id = fx.get(coord_wave_lane, 1)

        layout_lane16 = fx.make_layout((4, 16), (16, 1))
        coord_lane16 = fx.idx2crd(lane_id, layout_lane16)
        lane_div_16 = fx.get(coord_lane16, 0)
        lane_mod_16 = fx.get(coord_lane16, 1)

        layout_lane2 = fx.make_layout((8, 2), (2, 1))
        coord_lane2 = fx.idx2crd(lane_mod_16, layout_lane2)
        lane_div_2 = fx.get(coord_lane2, 0)
        lane_mod_2 = fx.get(coord_lane2, 1)

        # wave partitioning for qk, p, dp, ds
        ps_m_num_waves = 2
        ps_n_num_waves = 2
        ps_wave_layout = fx.make_layout(
            (ps_m_num_waves, ps_n_num_waves), (ps_n_num_waves, 1)
        )
        ps_coord = fx.idx2crd(wave_id, ps_wave_layout)
        ps_m_wave_id = fx.get(ps_coord, 0)
        ps_n_wave_id = fx.get(ps_coord, 1)
        ps_m_per_wave = tile_m // ps_m_num_waves
        ps_m_mx_per_wave = tile_m_mx // ps_m_num_waves
        ps_m_num_subtiles = ps_m_per_wave // 16
        ps_n_per_wave = tile_n // ps_n_num_waves
        ps_n_mx_per_wave = tile_n_mx // ps_n_num_waves
        ps_n_num_subtiles = ps_n_per_wave // 16
        ps_n_accs = ps_n_num_subtiles * ps_m_num_subtiles

        # wave partitioning for dv gemm
        dv_n_num_waves = 2
        dv_head_num_waves = 2
        dv_wave_layout = fx.make_layout(
            (dv_n_num_waves, dv_head_num_waves), (dv_head_num_waves, 1)
        )
        dv_coord = fx.idx2crd(wave_id, dv_wave_layout)
        dv_n_wave_id = fx.get(dv_coord, 0)
        dv_head_wave_id = fx.get(dv_coord, 1)
        dv_n_per_wave = tile_n // dv_n_num_waves
        dv_n_num_subtiles = dv_n_per_wave // 16
        dv_head_per_wave = tile_head // dv_head_num_waves
        dv_head_mx_per_wave = tile_head_mx // dv_head_num_waves
        dv_head_num_subtiles = dv_head_per_wave // 16
        dv_n_accs = dv_n_num_subtiles * dv_head_num_subtiles

        # wave partitioning for dk gemm
        dk_n_num_waves = 2
        dk_head_num_waves = 2
        dk_wave_layout = fx.make_layout(
            (dk_n_num_waves, dk_head_num_waves), (dk_head_num_waves, 1)
        )
        dk_coord = fx.idx2crd(wave_id, dk_wave_layout)
        dk_n_wave_id = fx.get(dk_coord, 0)
        dk_head_wave_id = fx.get(dk_coord, 1)
        dk_n_per_wave = tile_n // dk_n_num_waves
        dk_num_subtiles_n = dk_n_per_wave // 16
        dk_head_per_wave = tile_head // dk_head_num_waves
        dk_head_mx_per_wave = tile_head_mx // dk_head_num_waves
        dk_num_subtiles_head = dk_head_per_wave // 16
        dk_n_accs = dk_num_subtiles_n * dk_num_subtiles_head

        # wave partitioning for dq gemm
        dq_m_num_waves = 2
        dq_head_num_waves = 2
        dq_wave_layout = fx.make_layout(
            (dq_m_num_waves, dq_head_num_waves), (dq_head_num_waves, 1)
        )
        dq_coord = fx.idx2crd(wave_id, dq_wave_layout)
        dq_m_wave_id = fx.get(dq_coord, 0)
        dq_head_wave_id = fx.get(dq_coord, 1)
        dq_m_per_wave = tile_m // dq_m_num_waves
        dq_num_subtiles_m = dq_m_per_wave // 16
        dq_head_per_wave = tile_head // dq_head_num_waves
        dq_head_mx_per_wave = tile_head_mx // dq_head_num_waves
        dq_num_subtiles_head = dq_head_per_wave // 16
        dq_n_accs = dq_num_subtiles_m * dq_num_subtiles_head

        # ── A LDS load helpers ──

        def lds_load_16b(curr_row_lds, col_base, lds_stride, lds_buffer, swizzle=16):
            if swizzle == 16:
                col_base = swizzle_xor16(curr_row_lds, col_base, lds_stride // swizzle)
            idx = curr_row_lds * lds_stride + col_base
            return vector.load_op(_vec16_type(), lds_buffer, [idx])

        def lds_load_8b_transposed(
            curr_row_lds, col_base, lds_stride, lds_buffer, swizzle=16
        ):
            if swizzle == 16:
                col_base = swizzle_xor16(curr_row_lds, col_base, lds_stride // swizzle)
            col_base = col_base + lane_mod_2 * 8
            idx = curr_row_lds * lds_stride + col_base
            return lds_transpose_load(lds_buffer, idx)

        def lds_load_packs_k64(
            curr_row_lds, col_base, lds_stride, lds_buffer, swizzle=16
        ):
            vec = lds_load_16b(curr_row_lds, col_base, lds_stride, lds_buffer, swizzle)
            vec = vector.bitcast(T.i64x2, vec)
            val0_i64 = vector.extract(vec, static_position=[0], dynamic_position=[])
            val1_i64 = vector.extract(vec, static_position=[1], dynamic_position=[])
            return val0_i64, val1_i64

        def lds_load_packs_k32_transposed(
            curr_row_lds, col_base, lds_stride, lds_buffer, swizzle=16
        ):
            vec = lds_load_8b_transposed(
                curr_row_lds, col_base, lds_stride, lds_buffer, swizzle
            )
            vec = vector.bitcast(T.vec(1, T.i64), vec)
            val_i64 = vector.extract(vec, static_position=[0], dynamic_position=[])
            return val_i64

        def lds_scale_load(row, col, lds_stride, lds_buffer):
            idx = row * lds_stride + col
            vec = vector.load_op(T.vec(1, T.i8), lds_buffer, [idx])
            val = vector.extract(vec, static_position=[0], dynamic_position=[])
            val = val.extui(T.i32)
            return val

        # ── A global→reg load ─────────────────────────────────────────────
        head_dim_div4 = head_dim // 4
        tile_m_div16 = tile_m // 16
        tile_head_div16 = arith.index(tile_head // 16)
        num_qo_loads = bytes_per_thread_qo // qo_load_bytes
        num_kv_loads = bytes_per_thread_kv // kv_load_bytes
        tile_head_dwords = tile_head // 4
        layout_qo_tile_div4 = fx.make_layout(
            (tile_m, tile_head_dwords), (tile_head_dwords, 1)
        )
        layout_kv_tile_div4 = fx.make_layout(
            (tile_n, tile_head_dwords), (tile_head_dwords, 1)
        )
        c4 = fx.Index(4)
        tx_i32_base = tx * c4

        def load_q_16(idx_elem):
            return buffer_copy_gmem16_dwordx4(
                buffer_ops,
                vector,
                elem_type=_elem_type(),
                idx_i32=idx_elem,
                rsrc=q_rsrc,
                vec_elems=16,
                elem_bytes=elem_bytes,
            )

        def load_k_16(idx_elem):
            return buffer_copy_gmem16_dwordx4(
                buffer_ops,
                vector,
                elem_type=_elem_type(),
                idx_i32=idx_elem,
                rsrc=k_rsrc,
                vec_elems=16,
                elem_bytes=elem_bytes,
            )

        def load_v_16(idx_elem):
            return buffer_copy_gmem16_dwordx4(
                buffer_ops,
                vector,
                elem_type=_elem_type(),
                idx_i32=idx_elem,
                rsrc=v_rsrc,
                vec_elems=16,
                elem_bytes=elem_bytes,
            )

        def load_do_16(idx_elem):
            return buffer_copy_gmem16_dwordx4(
                buffer_ops,
                vector,
                elem_type=_elem_type(),
                idx_i32=idx_elem,
                rsrc=do_rsrc,
                vec_elems=16,
                elem_bytes=elem_bytes,
            )

        def qo_tile_chunk_coord_i32(i: int):
            return tile_chunk_coord_i32(
                arith,
                tx_i32_base=tx_i32_base,
                i=i,
                total_threads=total_threads,
                layout_tile_div4=layout_qo_tile_div4,
            )

        def kv_tile_chunk_coord_i32(i: int):
            return tile_chunk_coord_i32(
                arith,
                tx_i32_base=tx_i32_base,
                i=i,
                total_threads=total_threads,
                layout_tile_div4=layout_kv_tile_div4,
            )

        def prefetch_q_tile(offset_m):
            parts = []
            for i in range_constexpr(num_qo_loads):
                row_q_local, col_q_local_i32 = qo_tile_chunk_coord_i32(i)
                row_q_global = offset_m + row_q_local
                idx_elem = row_q_global * head_dim_div4 + col_q_local_i32
                q_16B = load_q_16(idx_elem)
                parts.append(vector.bitcast(T.i32x4, q_16B))
            return parts

        def prefetch_k_tile():
            parts = []
            for i in range_constexpr(num_kv_loads):
                row_k_local, col_k_local_i32 = kv_tile_chunk_coord_i32(i)
                row_k_global = global_offset_n + row_k_local
                idx_elem = row_k_global * head_dim_div4 + col_k_local_i32
                k_16B = load_k_16(idx_elem)
                parts.append(vector.bitcast(T.i32x4, k_16B))
            return parts

        def prefetch_v_tile():
            parts = []
            for i in range_constexpr(num_kv_loads):
                row_v_local, col_v_local_i32 = kv_tile_chunk_coord_i32(i)
                row_v_global = global_offset_n + row_v_local
                idx_elem = row_v_global * head_dim_div4 + col_v_local_i32
                v_16B = load_v_16(idx_elem)
                parts.append(vector.bitcast(T.i32x4, v_16B))
            return parts

        def prefetch_do_tile(offset_m):
            parts = []
            for i in range_constexpr(num_qo_loads):
                row_do_local, col_do_local_i32 = qo_tile_chunk_coord_i32(i)
                row_do_global = offset_m + row_do_local
                idx_elem = row_do_global * head_dim_div4 + col_do_local_i32
                do_16B = load_do_16(idx_elem)
                parts.append(vector.bitcast(T.i32x4, do_16B))
            return parts

        def prefetch_q_scale_head_2d_tile(offset_m):
            parts = []
            for i in range_constexpr(ps_m_num_subtiles // 2):
                global_row = offset_m + ps_m_wave_id * ps_m_mx_per_wave + i
                global_col = lane_div_16 % tile_head_mx
                global_idx = global_row * head_dim_mx + global_col
                vec = buffer_ops.buffer_load(
                    q_scale_rsrc, global_idx, vec_width=1, dtype=T.i8
                )
                vec = vec.extui(T.i32)
                parts.append(vec)
            return parts

        def prefetch_q_scale_m_2d_tile(offset_m):
            parts = []
            for i in range_constexpr(dk_num_subtiles_head // 2):
                global_row = offset_m + lane_div_16 % tile_m_mx
                global_col = dk_head_wave_id * dk_head_mx_per_wave + i
                global_idx = global_row * head_dim_mx + global_col
                vec = buffer_ops.buffer_load(
                    q_scale_rsrc, global_idx, vec_width=1, dtype=T.i8
                )
                vec = vec.extui(T.i32)
                parts.append(vec)
            return parts

        def prefetch_k_scale_head_2d_tile():
            parts = []
            for i in range_constexpr(ps_n_num_subtiles // 2):
                global_row = global_offset_n_mx + ps_n_wave_id * ps_n_mx_per_wave + i
                global_col = lane_div_16 % tile_head_mx
                global_idx = global_row * head_dim_mx + global_col
                vec = buffer_ops.buffer_load(
                    k_scale_rsrc, global_idx, vec_width=1, dtype=T.i8
                )
                vec = vec.extui(T.i32)
                parts.append(vec)
            return parts

        def prefetch_k_scale_n_2d_tile():
            parts = []
            for i in range_constexpr(dq_num_subtiles_head // 2):
                global_row = global_offset_n_mx + lane_div_16 % tile_n_mx
                global_col = dq_head_wave_id * dq_head_mx_per_wave + i
                global_idx = global_row * head_dim_mx + global_col
                vec = buffer_ops.buffer_load(
                    k_scale_rsrc, global_idx, vec_width=1, dtype=T.i8
                )
                vec = vec.extui(T.i32)
                parts.append(vec)
            return parts

        def prefetch_v_scale_tile():
            vec_width = bytes_per_thread_kv_scale
            if const_expr(vec_width == 1):
                if const_expr(bytes_per_tile_kv_scale < total_threads):
                    idx_elem = (
                        global_offset_n * head_dim_mx + tx % bytes_per_tile_kv_scale
                    )
                else:
                    idx_elem = global_offset_n * head_dim_mx + tx
                vec = buffer_ops.buffer_load(
                    v_scale_rsrc, idx_elem, vec_width=1, dtype=T.i8
                )
                vec = vector.from_elements(T.vec(1, T.i8), [vec])
            else:  # vec_width=2
                idx_elem = (global_offset_n * head_dim_mx + tx * vec_width) // 2
                vec = buffer_ops.buffer_load(
                    v_scale_rsrc, idx_elem, vec_width=1, dtype=T.i16
                )
                vec = vector.from_elements(T.vec(1, T.i16), [vec])
                vec = vector.bitcast(T.i8x2, vec)
            return vec

        def prefetch_do_scale_head_2d_tile(offset_m):
            parts = []
            for i in range_constexpr(ps_m_num_subtiles // 2):
                global_row = offset_m + ps_m_wave_id * ps_m_mx_per_wave + i
                global_col = lane_div_16 % tile_head_mx
                global_idx = global_row * head_dim_mx + global_col
                vec = buffer_ops.buffer_load(
                    do_scale_rsrc, global_idx, vec_width=1, dtype=T.i8
                )
                vec = vec.extui(T.i32)
                parts.append(vec)
            return parts

        def prefetch_do_scale_m_2d_tile(offset_m):
            parts = []
            for i in range_constexpr(dv_head_num_subtiles // 2):
                global_row = offset_m + lane_div_16 % tile_m_mx
                global_col = dv_head_wave_id * dv_head_mx_per_wave + i
                global_idx = global_row * head_dim_mx + global_col
                vec = buffer_ops.buffer_load(
                    do_scale_rsrc, global_idx, vec_width=1, dtype=T.i8
                )
                vec = vec.extui(T.i32)
                parts.append(vec)
            return parts

        def store_q_tile_to_lds(vec_q_parts, lds_buffer):
            for i in range_constexpr(num_qo_loads):
                row_q_local, col_q_local_i32 = qo_tile_chunk_coord_i32(i)
                col_local_bytes = col_q_local_i32 * c4
                col_swz_bytes = swizzle_xor16(
                    row_q_local, col_local_bytes, tile_head_div16
                )
                col_swz = col_swz_bytes
                idx0 = row_q_local * tile_head + col_swz
                v16 = vector.bitcast(_vec16_type(), vec_q_parts[i])
                vector.store(v16, lds_buffer, [idx0])

        def store_k_tile_to_lds(vec_k_parts, lds_buffer):
            for i in range_constexpr(num_kv_loads):
                row_k_local, col_k_local_i32 = kv_tile_chunk_coord_i32(i)
                col_local_bytes = col_k_local_i32 * c4
                col_swz_bytes = swizzle_xor16(
                    row_k_local, col_local_bytes, tile_head_div16
                )
                col_swz = col_swz_bytes
                idx0 = row_k_local * tile_head + col_swz
                v16 = vector.bitcast(_vec16_type(), vec_k_parts[i])
                vector.store(v16, lds_buffer, [idx0])

        def store_v_tile_to_lds(vec_v_parts, lds_buffer):
            for i in range_constexpr(num_kv_loads):
                row_v_local, col_v_local_i32 = kv_tile_chunk_coord_i32(i)
                col_local_bytes = col_v_local_i32 * c4
                col_swz_bytes = swizzle_xor16(
                    row_v_local, col_local_bytes, tile_head_div16
                )
                col_swz = col_swz_bytes
                idx0 = row_v_local * tile_head + col_swz
                v16 = vector.bitcast(_vec16_type(), vec_v_parts[i])
                vector.store(v16, lds_buffer, [idx0])

        def store_do_tile_to_lds(vec_do_parts, lds_buffer):
            for i in range_constexpr(num_qo_loads):
                row_do_local, col_do_local_i32 = qo_tile_chunk_coord_i32(i)
                col_local_bytes = col_do_local_i32 * c4
                col_swz_bytes = swizzle_xor16(
                    row_do_local, col_local_bytes, tile_head_div16
                )
                col_swz = col_swz_bytes
                idx0 = row_do_local * tile_head + col_swz
                v16 = vector.bitcast(_vec16_type(), vec_do_parts[i])
                vector.store(v16, lds_buffer, [idx0])

        def store_v_scale_tile_to_lds(vec_scale, lds_buffer):
            vec_width = bytes_per_thread_kv_scale
            idx = tx * vec_width
            if total_threads > bytes_per_tile_kv_scale:
                idx = idx % bytes_per_tile_kv_scale
            vector.store(vec_scale, lds_buffer, [idx])

        # ── Compute tile (MFMA) ───────────────────────────────────────────

        def pack_i64x4_to_i32x8(x0, x1, x2, x3):
            vec4_i64 = T.vec(4, T.i64)
            vec8_i32 = T.vec(8, T.i32)
            v4 = vector.from_elements(vec4_i64, [x0, x1, x2, x3])
            return vector.bitcast(vec8_i32, v4)

        def compute_qk(lds_a_buffer, a_scales, lds_b_buffer, b_scales):
            # (m, head) @ (head, n) = (m, n)

            current_accs_list = [acc_init] * ps_n_accs
            mfma_res_ty = T.f32x4

            ku0 = 0
            ku1 = 1
            lds_col0 = ku0 * 64 + lane_div_16 * 16
            lds_col1 = ku1 * 64 + lane_div_16 * 16
            lds_scale_col = lane_div_16
            if const_expr(tile_head == 64):
                lds_scale_col = lds_scale_col % 2

            for mi in range_constexpr(ps_m_num_subtiles):
                lds_a_row = ps_m_wave_id * ps_m_per_wave + mi * 16 + lane_mod_16
                a0, a1 = lds_load_packs_k64(
                    lds_a_row, lds_col0, tile_head, lds_a_buffer
                )
                if const_expr(tile_head == 128):
                    a2, a3 = lds_load_packs_k64(
                        lds_a_row, lds_col1, tile_head, lds_a_buffer
                    )
                else:
                    a2 = a3 = fx.Int64(0)
                a128 = pack_i64x4_to_i32x8(a0, a1, a2, a3)

                lds_a_scale_row = lds_a_row
                # a_scale = lds_scale_load(
                #    lds_a_scale_row, lds_scale_col, tile_head_mx, lds_a_scale_buffer
                # )
                a_scale = a_scales[mi // 2]

                for ni in range_constexpr(ps_n_num_subtiles):
                    lds_b_row = ps_n_wave_id * ps_n_per_wave + ni * 16 + lane_mod_16
                    b0, b1 = lds_load_packs_k64(
                        lds_b_row, lds_col0, tile_head, lds_b_buffer
                    )
                    if const_expr(tile_head == 128):
                        b2, b3 = lds_load_packs_k64(
                            lds_b_row, lds_col1, tile_head, lds_b_buffer
                        )
                    else:
                        b2 = b3 = fx.Int64(0)
                    b128 = pack_i64x4_to_i32x8(b0, b1, b2, b3)

                    b_scale = b_scales[ni // 2]

                    acc_idx = mi * ps_n_num_subtiles + ni
                    current_accs_list[acc_idx] = rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                        mfma_res_ty,
                        [
                            a128,
                            b128,
                            current_accs_list[acc_idx],
                            0,
                            0,
                            0,
                            a_scale,
                            0,
                            b_scale,
                        ],
                    )
            return current_accs_list

        def softmax(accs_in, offset_m):
            # inputs are tile_m x tile_n shape

            accs_out = [acc_init] * ps_n_accs

            for mi in range_constexpr(ps_m_num_subtiles):
                global_m_norm_idx = (
                    offset_m + ps_m_wave_id * ps_m_per_wave + mi * 16 + lane_div_16 * 4
                )
                m_norm_vector = buffer_ops.buffer_load(
                    M_rsrc, global_m_norm_idx, vec_width=4
                )

                for ni in range_constexpr(ps_n_num_subtiles):

                    acc_idx = mi * ps_n_num_subtiles + ni
                    acc = accs_in[acc_idx]

                    vals_f32 = []
                    for ii in range_constexpr(4):
                        val_f32 = vector.extract(
                            acc, static_position=[ii], dynamic_position=[]
                        )
                        m_norm = vector.extract(
                            m_norm_vector, static_position=[ii], dynamic_position=[]
                        )
                        val_f32 = val_f32 * c_sm_scale
                        val_f32 = val_f32 - m_norm
                        val_f32 = val_f32 * log2e
                        val_f32 = rocdl.exp2(T.f32, val_f32)
                        if causal:
                            global_m = (
                                offset_m
                                + ps_m_wave_id * ps_m_per_wave
                                + mi * 16
                                + lane_div_16 * 4
                                + ii
                            )
                            global_n = (
                                global_offset_n
                                + ps_n_wave_id * ps_n_per_wave
                                + ni * 16
                                + lane_mod_16
                            )
                            needs_mask = arith.cmpi(
                                arith.CmpIPredicate.ugt, global_n, global_m
                            )
                            mask_if = scf.IfOp(needs_mask, [T.f32], has_else=True)
                            with ir.InsertionPoint(mask_if.then_block):
                                scf.YieldOp([arith.constant(0.0, type=T.f32)])
                            with ir.InsertionPoint(mask_if.else_block):
                                scf.YieldOp([val_f32])
                            val_f32 = mask_if.results[0]
                        vals_f32.append(val_f32)
                    vals_f32_vector = vector.from_elements(T.f32x4, vals_f32)
                    accs_out[acc_idx] = vals_f32_vector

            return accs_out

        def compute_dv(
            accs_in, lds_a_buffer, lds_a_scale_buffer, lds_b_buffer, b_scales
        ):
            current_accs_list = list(accs_in)
            mfma_res_ty = T.f32x4
            num_subtiles_reduction = max(1, tile_m // 128)
            for ku128 in range_constexpr(num_subtiles_reduction):
                ku0 = ku128 * 2
                ku1 = ku0 + 1

                lds_a_col0 = (
                    ku0 * 64 + lane_div_16 * 16
                )  # 16 elements packed per lane, 64 per wave
                lds_a_col1 = ku1 * 64 + lane_div_16 * 16

                lds_b_row0 = ku0 * 64 + lane_div_16 * 16 + lane_div_2
                lds_b_row1 = ku0 * 64 + lane_div_16 * 16 + 8 + lane_div_2
                lds_b_row2 = ku1 * 64 + lane_div_16 * 16 + lane_div_2
                lds_b_row3 = ku1 * 64 + lane_div_16 * 16 + 8 + lane_div_2

                lds_a_scale_col = lane_div_16
                lds_b_scale_row = lane_div_16
                if const_expr(tile_m == 64):
                    lds_a_scale_col = lds_a_scale_col % 2
                    lds_b_scale_row = lds_b_scale_row % 2

                for ni in range_constexpr(dv_n_num_subtiles):
                    lds_a_row = dv_n_wave_id * dv_n_per_wave + ni * 16 + lane_mod_16
                    a0, a1 = lds_load_packs_k64(
                        lds_a_row, lds_a_col0, tile_m, lds_a_buffer
                    )
                    if const_expr(tile_m == 128):
                        a2, a3 = lds_load_packs_k64(
                            lds_a_row, lds_a_col1, tile_m, lds_a_buffer
                        )
                    else:
                        a2 = a3 = fx.Int64(0)
                    a128 = pack_i64x4_to_i32x8(a0, a1, a2, a3)
                    a_scale = lds_scale_load(
                        lds_a_row, lds_a_scale_col, tile_m_mx, lds_a_scale_buffer
                    )

                    for hi in range_constexpr(dv_head_num_subtiles):
                        lds_b_col = dv_head_wave_id * dv_head_per_wave + hi * 16
                        b0 = lds_load_packs_k32_transposed(
                            lds_b_row0, lds_b_col, tile_head, lds_b_buffer
                        )
                        b1 = lds_load_packs_k32_transposed(
                            lds_b_row1, lds_b_col, tile_head, lds_b_buffer
                        )
                        if const_expr(tile_m == 128):
                            b2 = lds_load_packs_k32_transposed(
                                lds_b_row2, lds_b_col, tile_head, lds_b_buffer
                            )
                            b3 = lds_load_packs_k32_transposed(
                                lds_b_row3, lds_b_col, tile_head, lds_b_buffer
                            )
                        else:
                            b2 = fx.Int64(0)
                            b3 = fx.Int64(0)
                        b128 = pack_i64x4_to_i32x8(b0, b1, b2, b3)

                        b_scale = b_scales[hi // 2]

                        acc_idx = ni * dv_head_num_subtiles + hi
                        current_accs_list[acc_idx] = (
                            rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                                mfma_res_ty,
                                [
                                    a128,
                                    b128,
                                    current_accs_list[acc_idx],
                                    0,
                                    0,
                                    0,
                                    a_scale,
                                    0,
                                    b_scale,
                                ],
                            )
                        )
            return current_accs_list

        def compute_dp(lds_a_buffer, a_scales, lds_b_buffer, lds_b_scale_buffer):
            current_accs_list = [acc_init] * ps_n_accs
            mfma_res_ty = T.f32x4
            ku0 = 0
            ku1 = 1
            lds_col0 = ku0 * 64 + lane_div_16 * 16
            lds_col1 = ku1 * 64 + lane_div_16 * 16
            lds_scale_col = lane_div_16
            if const_expr(tile_head == 64):
                lds_scale_col = lds_scale_col % 2

            for mi in range_constexpr(ps_m_num_subtiles):
                lds_a_row = ps_m_wave_id * ps_m_per_wave + mi * 16 + lane_mod_16
                a0, a1 = lds_load_packs_k64(
                    lds_a_row, lds_col0, tile_head, lds_a_buffer
                )
                if const_expr(tile_head == 128):
                    a2, a3 = lds_load_packs_k64(
                        lds_a_row, lds_col1, tile_head, lds_a_buffer
                    )
                else:
                    a2 = a3 = fx.Int64(0)
                a128 = pack_i64x4_to_i32x8(a0, a1, a2, a3)

                a_scale = a_scales[mi // 2]

                for ni in range_constexpr(ps_n_num_subtiles):
                    lds_b_row = ps_n_wave_id * ps_n_per_wave + ni * 16 + lane_mod_16
                    b0, b1 = lds_load_packs_k64(
                        lds_b_row, lds_col0, tile_head, lds_b_buffer
                    )
                    if const_expr(tile_head == 128):
                        b2, b3 = lds_load_packs_k64(
                            lds_b_row, lds_col1, tile_head, lds_b_buffer
                        )
                    else:
                        b2 = b3 = fx.Int64(0)
                    b128 = pack_i64x4_to_i32x8(b0, b1, b2, b3)
                    b_scale = lds_scale_load(
                        lds_b_row, lds_scale_col, tile_head_mx, lds_b_scale_buffer
                    )

                    acc_idx = mi * ps_n_num_subtiles + ni
                    current_accs_list[acc_idx] = rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                        mfma_res_ty,
                        [
                            a128,
                            b128,
                            current_accs_list[acc_idx],
                            0,
                            0,
                            0,
                            a_scale,
                            0,
                            b_scale,
                        ],
                    )
            return current_accs_list

        def compute_ds(dp_accs, p_accs, offset_m):
            # inputs are tile_m x tile_n shape

            accs_out = [acc_init] * ps_n_accs

            for mi in range_constexpr(ps_m_num_subtiles):

                global_D_idx = (
                    offset_m + ps_m_wave_id * ps_m_per_wave + mi * 16 + lane_div_16 * 4
                )
                D_vector = buffer_ops.buffer_load(D_rsrc, global_D_idx, vec_width=4)

                for ni in range_constexpr(ps_n_num_subtiles):

                    acc_idx = mi * ps_n_num_subtiles + ni
                    dp_f32x4 = dp_accs[acc_idx]
                    p_f32x4 = p_accs[acc_idx]

                    vals_f32 = []
                    for ii in range_constexpr(4):
                        dp_f32 = vector.extract(
                            dp_f32x4, static_position=[ii], dynamic_position=[]
                        )
                        p_f32 = vector.extract(
                            p_f32x4, static_position=[ii], dynamic_position=[]
                        )
                        D = vector.extract(
                            D_vector, static_position=[ii], dynamic_position=[]
                        )
                        ds_f32 = p_f32 * (dp_f32 - D)
                        vals_f32.append(ds_f32)

                    vals_f32_vector = vector.from_elements(T.f32x4, vals_f32)
                    accs_out[acc_idx] = vals_f32_vector

            return accs_out

        def wave_reduce_max_4threads(x):
            width_i32 = arith.constant(64, type=T.i32)
            w = x
            for sh in [32, 16]:
                off = arith.constant(sh, type=T.i32)
                peer = w.shuffle_xor(off, width_i32)
                w = w.maximumf(peer)
            return w

        def mxquant_m_and_store_to_lds(accs_in, lds_buffer, lds_buffer_scale):
            # inputs are tile_m x tile_n shape

            for mi in range_constexpr(ps_m_num_subtiles // 2):
                for ni in range_constexpr(ps_n_num_subtiles):

                    acc_idx0 = (mi * 2) * ps_n_num_subtiles + ni
                    acc_idx1 = (mi * 2 + 1) * ps_n_num_subtiles + ni
                    acc0 = accs_in[acc_idx0]
                    acc1 = accs_in[acc_idx1]

                    vals_subtile0 = []
                    vals_subtile1 = []
                    vals_abs = []
                    for ii in range_constexpr(4):
                        val0 = vector.extract(
                            acc0, static_position=[ii], dynamic_position=[]
                        )
                        vals_subtile0.append(val0)
                        val1 = vector.extract(
                            acc1, static_position=[ii], dynamic_position=[]
                        )
                        vals_subtile1.append(val1)
                        val0_abs = fx_math.absf(val0)
                        val1_abs = fx_math.absf(val1)
                        vals_abs.append(val0_abs)
                        vals_abs.append(val1_abs)

                    vals_abs_vector = vector.from_elements(T.vec(8, T.f32), vals_abs)
                    val_max = vector.reduction(T.f32, "maxnumf", vals_abs_vector)
                    val_max = wave_reduce_max_4threads(val_max)
                    val_max = val_max * fp8_max_rcp
                    val_max = arith.bitcast(T.i32, val_max)
                    val_max = val_max + arith.constant(0x007FFFFF, type=T.i32)
                    val_max = val_max & arith.constant(0x7F800000, type=T.i32)
                    val_max_f32 = arith.bitcast(T.f32, val_max)
                    val_max_rcp = arith.select(
                        val_max_f32 == zero_f, zero_f, rocdl.rcp(T.f32, val_max_f32)
                    )
                    scale = val_max >> 23
                    scale = arith.trunci(T.i8, scale)
                    scale_vector = vector.from_elements(T.vec(1, T.i8), [scale])

                    for ii in range_constexpr(4):
                        vals_subtile0[ii] = vals_subtile0[ii] * val_max_rcp
                        vals_subtile1[ii] = vals_subtile1[ii] * val_max_rcp

                    val_f8_packed_i32 = rocdl.cvt_pk_fp8_f32(
                        T.i32, vals_subtile0[0], vals_subtile0[1], fx.Int32(0), False
                    )
                    val_f8_packed_i32 = rocdl.cvt_pk_fp8_f32(
                        T.i32,
                        vals_subtile0[2],
                        vals_subtile0[3],
                        val_f8_packed_i32,
                        True,
                    )
                    val_f8_packed_i32_vector = vector.from_elements(
                        T.vec(1, T.i32), [val_f8_packed_i32]
                    )
                    val_f8x4_subtile0 = vector.bitcast(T.f8x4, val_f8_packed_i32_vector)

                    val_f8_packed_i32 = rocdl.cvt_pk_fp8_f32(
                        T.i32, vals_subtile1[0], vals_subtile1[1], fx.Int32(0), False
                    )
                    val_f8_packed_i32 = rocdl.cvt_pk_fp8_f32(
                        T.i32,
                        vals_subtile1[2],
                        vals_subtile1[3],
                        val_f8_packed_i32,
                        True,
                    )
                    val_f8_packed_i32_vector = vector.from_elements(
                        T.vec(1, T.i32), [val_f8_packed_i32]
                    )
                    val_f8x4_subtile1 = vector.bitcast(T.f8x4, val_f8_packed_i32_vector)

                    lds_row = ps_n_wave_id * ps_n_per_wave + ni * 16 + lane_mod_16
                    lds_col_base0 = (
                        ps_m_wave_id * ps_m_per_wave + (mi * 2) * 16
                    )  # + lane_div_16 * 4
                    lds_col_base1 = (
                        ps_m_wave_id * ps_m_per_wave + (mi * 2 + 1) * 16
                    )  # + lane_div_16 * 4
                    lds_col0 = swizzle_xor16(lds_row, lds_col_base0, tile_m_div16)
                    lds_col1 = swizzle_xor16(lds_row, lds_col_base1, tile_m_div16)
                    lds_col0 = lds_col0 + lane_div_16 * 4
                    lds_col1 = lds_col1 + lane_div_16 * 4
                    lds_scale_col = ps_m_wave_id * ps_m_mx_per_wave + mi
                    lds_idx0 = lds_row * tile_m + lds_col0
                    lds_idx1 = lds_row * tile_m + lds_col1
                    lds_scale_idx = lds_row * tile_m_mx + lds_scale_col

                    vector.store(val_f8x4_subtile0, lds_buffer, [lds_idx0])
                    vector.store(val_f8x4_subtile1, lds_buffer, [lds_idx1])
                    vector.store(scale_vector, lds_buffer_scale, [lds_scale_idx])

        def wave_reduce_max_16threads(x):
            width_i32 = arith.constant(64, type=T.i32)
            w = x
            for sh in [8, 4, 2, 1]:
                off = arith.constant(sh, type=T.i32)
                peer = w.shuffle_xor(off, width_i32)
                w = w.maximumf(peer)
            return w

        def mxquant_n_and_store_to_lds(accs_in, lds_buffer, lds_buffer_scale):
            # inputs are tile_m x tile_n shape

            for mi in range_constexpr(ps_m_num_subtiles):
                for ni in range_constexpr(ps_n_num_subtiles // 2):

                    acc_idx0 = mi * ps_n_num_subtiles + ni * 2
                    acc_idx1 = mi * ps_n_num_subtiles + ni * 2 + 1
                    acc0 = accs_in[acc_idx0]
                    acc1 = accs_in[acc_idx1]

                    vals_subtile0 = []
                    vals_subtile1 = []
                    scales = []
                    for ii in range_constexpr(4):
                        val0 = vector.extract(
                            acc0, static_position=[ii], dynamic_position=[]
                        )
                        val1 = vector.extract(
                            acc1, static_position=[ii], dynamic_position=[]
                        )
                        val0_abs = fx_math.absf(val0)
                        val1_abs = fx_math.absf(val1)
                        val_max = arith.maximumf(val0_abs, val1_abs)
                        val_max = wave_reduce_max_16threads(val_max)
                        val_max = val_max * fp8_max_rcp
                        val_max = arith.bitcast(T.i32, val_max)
                        val_max = val_max + arith.constant(0x007FFFFF, type=T.i32)
                        val_max = val_max & arith.constant(0x7F800000, type=T.i32)
                        val_max_f32 = arith.bitcast(T.f32, val_max)
                        val_max_rcp = arith.select(
                            val_max_f32 == zero_f, zero_f, rocdl.rcp(T.f32, val_max_f32)
                        )
                        val0_quant = val0 * val_max_rcp
                        vals_subtile0.append(val0_quant)
                        val1_quant = val1 * val_max_rcp
                        vals_subtile1.append(val1_quant)
                        scale = val_max >> 23
                        scale = arith.trunci(T.i8, scale)
                        scales.append(scale)

                    val_f8_packed_i32 = rocdl.cvt_pk_fp8_f32(
                        T.i32, vals_subtile0[0], vals_subtile0[1], fx.Int32(0), False
                    )
                    val_f8_packed_i32 = rocdl.cvt_pk_fp8_f32(
                        T.i32,
                        vals_subtile0[2],
                        vals_subtile0[3],
                        val_f8_packed_i32,
                        True,
                    )
                    val_f8_packed_i32_vector = vector.from_elements(
                        T.vec(1, T.i32), [val_f8_packed_i32]
                    )
                    val_f8x4_subtile0 = vector.bitcast(T.f8x4, val_f8_packed_i32_vector)

                    val_f8_packed_i32 = rocdl.cvt_pk_fp8_f32(
                        T.i32, vals_subtile1[0], vals_subtile1[1], fx.Int32(0), False
                    )
                    val_f8_packed_i32 = rocdl.cvt_pk_fp8_f32(
                        T.i32,
                        vals_subtile1[2],
                        vals_subtile1[3],
                        val_f8_packed_i32,
                        True,
                    )
                    val_f8_packed_i32_vector = vector.from_elements(
                        T.vec(1, T.i32), [val_f8_packed_i32]
                    )
                    val_f8x4_subtile1 = vector.bitcast(T.f8x4, val_f8_packed_i32_vector)

                    lds_row0 = (
                        ps_n_wave_id * ps_n_per_wave + (ni * 2) * 16 + lane_mod_16
                    )
                    lds_row1 = (
                        ps_n_wave_id * ps_n_per_wave + (ni * 2 + 1) * 16 + lane_mod_16
                    )
                    lds_col_base = (
                        ps_m_wave_id * ps_m_per_wave + mi * 16
                    )  # + lane_div_16 * 4
                    lds_col0 = swizzle_xor16(lds_row0, lds_col_base, tile_m_div16)
                    lds_col1 = swizzle_xor16(lds_row1, lds_col_base, tile_m_div16)
                    lds_col0 = lds_col0 + lane_div_16 * 4
                    lds_col1 = lds_col1 + lane_div_16 * 4
                    lds_idx0 = lds_row0 * tile_m + lds_col0
                    lds_idx1 = lds_row1 * tile_m + lds_col1
                    vector.store(val_f8x4_subtile0, lds_buffer, [lds_idx0])
                    vector.store(val_f8x4_subtile1, lds_buffer, [lds_idx1])

                    for ii in range_constexpr(4):
                        lds_scale_row = (
                            ps_m_wave_id * ps_m_per_wave
                            + mi * 16
                            + lane_div_16 * 4
                            + ii
                        )
                        lds_scale_col = ps_n_wave_id * ps_n_mx_per_wave + ni
                        lds_scale_idx = lds_scale_row * tile_n_mx + lds_scale_col

                        scale_vector = vector.from_elements(
                            T.vec(1, T.i8), [scales[ii]]
                        )
                        vector.store(scale_vector, lds_buffer_scale, [lds_scale_idx])

        def compute_dk(
            accs_in, lds_a_buffer, lds_a_scale_buffer, lds_b_buffer, b_scales
        ):
            current_accs_list = list(accs_in)
            mfma_res_ty = T.f32x4
            num_subtiles_reduction = max(1, tile_m // 128)
            for ku128 in range_constexpr(num_subtiles_reduction):
                ku0 = ku128 * 2
                ku1 = ku0 + 1

                lds_a_col0 = ku0 * 64 + lane_div_16 * 16
                lds_a_col1 = ku1 * 64 + lane_div_16 * 16

                lds_b_row0 = ku0 * 64 + lane_div_16 * 16 + lane_div_2
                lds_b_row1 = ku0 * 64 + lane_div_16 * 16 + 8 + lane_div_2
                lds_b_row2 = ku1 * 64 + lane_div_16 * 16 + lane_div_2
                lds_b_row3 = ku1 * 64 + lane_div_16 * 16 + 8 + lane_div_2

                lds_a_scale_col = lane_div_16
                lds_b_scale_row = lane_div_16
                if tile_m == 64:
                    lds_a_scale_col = lds_a_scale_col % 2
                    lds_b_scale_row = lds_b_scale_row % 2

                for ni in range_constexpr(dk_num_subtiles_n):
                    lds_a_row = dk_n_wave_id * dk_n_per_wave + ni * 16 + lane_mod_16
                    a0, a1 = lds_load_packs_k64(
                        lds_a_row, lds_a_col0, tile_m, lds_a_buffer
                    )
                    if const_expr(tile_m == 128):
                        a2, a3 = lds_load_packs_k64(
                            lds_a_row, lds_a_col1, tile_m, lds_a_buffer
                        )
                    else:
                        a2 = fx.Int64(0)
                        a3 = fx.Int64(0)
                    a128 = pack_i64x4_to_i32x8(a0, a1, a2, a3)
                    a_scale = lds_scale_load(
                        lds_a_row, lds_a_scale_col, tile_m_mx, lds_a_scale_buffer
                    )

                    for hi in range_constexpr(dk_num_subtiles_head):
                        lds_b_col = dk_head_wave_id * dk_head_per_wave + hi * 16
                        b0 = lds_load_packs_k32_transposed(
                            lds_b_row0, lds_b_col, tile_head, lds_b_buffer
                        )
                        b1 = lds_load_packs_k32_transposed(
                            lds_b_row1, lds_b_col, tile_head, lds_b_buffer
                        )
                        if const_expr(tile_m == 128):
                            b2 = lds_load_packs_k32_transposed(
                                lds_b_row2, lds_b_col, tile_head, lds_b_buffer
                            )
                            b3 = lds_load_packs_k32_transposed(
                                lds_b_row3, lds_b_col, tile_head, lds_b_buffer
                            )
                        else:
                            b2 = fx.Int64(0)
                            b3 = fx.Int64(0)
                        b128 = pack_i64x4_to_i32x8(b0, b1, b2, b3)

                        b_scale = b_scales[hi // 2]

                        acc_idx = ni * dk_num_subtiles_head + hi
                        current_accs_list[acc_idx] = (
                            rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                                mfma_res_ty,
                                [
                                    a128,
                                    b128,
                                    current_accs_list[acc_idx],
                                    0,
                                    0,
                                    0,
                                    a_scale,
                                    0,
                                    b_scale,
                                ],
                            )
                        )
            return current_accs_list

        def compute_dq(lds_a_buffer, lds_a_scale_buffer, lds_b_buffer, b_scales):
            # (m, n) @ (n, head) = (m, head)

            current_accs_list = [acc_init] * dq_n_accs
            mfma_res_ty = T.f32x4

            num_subtiles_reduction = max(1, tile_n // 128)
            for ku128 in range_constexpr(num_subtiles_reduction):
                ku0 = ku128 * 2
                ku1 = ku0 + 1

                lds_a_row0 = ku0 * 64 + lane_div_16 * 16 + lane_div_2
                lds_a_row1 = ku0 * 64 + lane_div_16 * 16 + 8 + lane_div_2
                lds_a_row2 = ku1 * 64 + lane_div_16 * 16 + lane_div_2
                lds_a_row3 = ku1 * 64 + lane_div_16 * 16 + 8 + lane_div_2

                lds_b_row0 = ku0 * 64 + lane_div_16 * 16 + lane_div_2
                lds_b_row1 = ku0 * 64 + lane_div_16 * 16 + 8 + lane_div_2
                lds_b_row2 = ku1 * 64 + lane_div_16 * 16 + lane_div_2
                lds_b_row3 = ku1 * 64 + lane_div_16 * 16 + 8 + lane_div_2

                lds_a_scale_col = lane_div_16
                # lds_b_scale_row = lane_div_16
                if const_expr(tile_n == 64):
                    lds_a_scale_col = lds_a_scale_col % 2
                    # lds_b_scale_row = lds_b_scale_row % 2

                for mi in range_constexpr(dq_num_subtiles_m):
                    lds_a_col = (
                        dq_m_wave_id * dq_m_per_wave + mi * 16
                    )  # + lane_mod_2 * 8
                    a0 = lds_load_packs_k32_transposed(
                        lds_a_row0, lds_a_col, tile_m, lds_a_buffer
                    )
                    a1 = lds_load_packs_k32_transposed(
                        lds_a_row1, lds_a_col, tile_m, lds_a_buffer
                    )
                    if const_expr(tile_n == 128):
                        a2 = lds_load_packs_k32_transposed(
                            lds_a_row2, lds_a_col, tile_m, lds_a_buffer
                        )
                        a3 = lds_load_packs_k32_transposed(
                            lds_a_row3, lds_a_col, tile_m, lds_a_buffer
                        )
                    else:
                        a2 = fx.Int64(0)
                        a3 = fx.Int64(0)

                    a128 = pack_i64x4_to_i32x8(a0, a1, a2, a3)
                    lds_a_scale_row = (
                        dq_m_wave_id * dq_m_per_wave + mi * 16 + lane_mod_16
                    )
                    a_scale = lds_scale_load(
                        lds_a_scale_row, lds_a_scale_col, tile_n_mx, lds_a_scale_buffer
                    )

                    for hi in range_constexpr(dq_num_subtiles_head):
                        lds_b_col = dq_head_wave_id * dq_head_per_wave + hi * 16
                        b0 = lds_load_packs_k32_transposed(
                            lds_b_row0, lds_b_col, tile_head, lds_b_buffer
                        )
                        b1 = lds_load_packs_k32_transposed(
                            lds_b_row1, lds_b_col, tile_head, lds_b_buffer
                        )
                        if const_expr(tile_n == 128):
                            b2 = lds_load_packs_k32_transposed(
                                lds_b_row2, lds_b_col, tile_head, lds_b_buffer
                            )
                            b3 = lds_load_packs_k32_transposed(
                                lds_b_row3, lds_b_col, tile_head, lds_b_buffer
                            )
                        else:
                            b2 = fx.Int64(0)
                            b3 = fx.Int64(0)
                        b128 = pack_i64x4_to_i32x8(b0, b1, b2, b3)

                        b_scale = b_scales[hi // 2]

                        acc_idx = mi * dq_num_subtiles_head + hi
                        current_accs_list[acc_idx] = (
                            rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                                mfma_res_ty,
                                [
                                    a128,
                                    b128,
                                    current_accs_list[acc_idx],
                                    0,
                                    0,
                                    0,
                                    a_scale,
                                    0,
                                    b_scale,
                                ],
                            )
                        )
            return current_accs_list

        def store_dq_atomic(final_accs, offset_m):
            for mi in range_constexpr(dq_num_subtiles_m):
                for hi in range_constexpr(dq_num_subtiles_head):
                    for ii in range_constexpr(4):
                        global_row = (
                            offset_m
                            + dq_m_wave_id * dq_m_per_wave
                            + mi * 16
                            + lane_div_16 * 4
                            + ii
                        )
                        global_col = (
                            dq_head_wave_id * dq_head_per_wave + hi * 16 + lane_mod_16
                        )
                        global_idx = global_row * head_dim + global_col
                        global_idx_bytes = global_idx * 4

                        acc_idx = mi * dq_num_subtiles_head + hi
                        acc = final_accs[acc_idx]
                        val_f32 = vector.extract(
                            acc, static_position=[ii], dynamic_position=[]
                        )
                        val_f32 = val_f32 * c_sm_scale
                        rocdl.raw_ptr_buffer_atomic_fadd(
                            val_f32,
                            dq_rsrc,
                            fx.Int32(global_idx_bytes),
                            fx.Int32(0),
                            fx.Int32(0),
                        )
                        # buffer_ops.buffer_store(val_f32, dq_rsrc, global_idx)

        def store_dk_atomic(final_accs):
            for ni in range_constexpr(dk_num_subtiles_n):
                for hi in range_constexpr(dk_num_subtiles_head):
                    acc_idx = ni * dk_num_subtiles_head + hi
                    acc = final_accs[acc_idx]
                    for ii in range_constexpr(4):

                        global_row = (
                            global_offset_n
                            + dk_n_wave_id * dk_n_per_wave
                            + ni * 16
                            + lane_div_16 * 4
                            + ii
                        )
                        global_col = (
                            dk_head_wave_id * dk_head_per_wave + hi * 16 + lane_mod_16
                        )
                        global_idx = global_row * head_dim + global_col

                        acc_idx = ni * dk_num_subtiles_head + hi
                        acc = final_accs[acc_idx]
                        val_f32 = vector.extract(
                            acc, static_position=[ii], dynamic_position=[]
                        )
                        val_f32 = val_f32 * c_sm_scale
                        if const_expr(gqa_size == 1):
                            buffer_ops.buffer_store(val_f32, dk_rsrc, global_idx)
                        else:
                            global_idx_bytes = global_idx * 4
                            rocdl.raw_ptr_buffer_atomic_fadd(
                                val_f32,
                                dk_rsrc,
                                fx.Int32(global_idx_bytes),
                                fx.Int32(0),
                                fx.Int32(0),
                            )

        def store_dv_atomic(final_accs):
            for ni in range_constexpr(dv_n_num_subtiles):
                for hi in range_constexpr(dv_head_num_subtiles):
                    acc_idx = ni * dv_head_num_subtiles + hi
                    acc = final_accs[acc_idx]
                    for ii in range_constexpr(4):

                        global_row = (
                            global_offset_n
                            + dv_n_wave_id * dv_n_per_wave
                            + ni * 16
                            + lane_div_16 * 4
                            + ii
                        )
                        global_col = (
                            dv_head_wave_id * dv_head_per_wave + hi * 16 + lane_mod_16
                        )
                        global_idx = global_row * head_dim + global_col

                        acc_idx = ni * dv_head_num_subtiles + hi
                        acc = final_accs[acc_idx]
                        val_f32 = vector.extract(
                            acc, static_position=[ii], dynamic_position=[]
                        )
                        if const_expr(gqa_size == 1):
                            buffer_ops.buffer_store(val_f32, dv_rsrc, global_idx)
                        else:
                            global_idx_bytes = global_idx * 4
                            rocdl.raw_ptr_buffer_atomic_fadd(
                                val_f32,
                                dv_rsrc,
                                fx.Int32(global_idx_bytes),
                                fx.Int32(0),
                                fx.Int32(0),
                            )

        # ── Scheduling hints ──────────────────────────────────────────────
        rocdl.sched_barrier(0)

        def hot_loop_scheduler():
            rocdl.sched_barrier(0)
            return

        # ── Main pipeline ─────────────────────────────────────────────────

        def _pack_state(dk, dv, q_scales_head, q_scales_m, do_scales_head, do_scales_m):
            return (
                list(dk)
                + list(dv)
                + list(q_scales_head)
                + list(q_scales_m)
                + list(do_scales_head)
                + list(do_scales_m)
            )

        def _unpack_state(vals):
            dk = list(vals[:dk_n_accs])
            dv = list(vals[dk_n_accs : dk_n_accs + dv_n_accs])
            q_scales_head = list(
                vals[
                    dk_n_accs
                    + dv_n_accs : dk_n_accs
                    + dv_n_accs
                    + ps_m_num_subtiles // 2
                ]
            )
            q_scales_m = list(
                vals[
                    dk_n_accs
                    + dv_n_accs
                    + ps_m_num_subtiles // 2 : dk_n_accs
                    + dv_n_accs
                    + ps_m_num_subtiles // 2
                    + dk_num_subtiles_head // 2
                ]
            )
            do_scales_head = list(
                vals[
                    dk_n_accs
                    + dv_n_accs
                    + ps_m_num_subtiles // 2
                    + dk_num_subtiles_head // 2 : dk_n_accs
                    + dv_n_accs
                    + ps_m_num_subtiles // 2
                    + dk_num_subtiles_head // 2
                    + ps_m_num_subtiles // 2
                ]
            )
            do_scales_m = list(
                vals[
                    dk_n_accs
                    + dv_n_accs
                    + ps_m_num_subtiles // 2
                    + dk_num_subtiles_head // 2
                    + ps_m_num_subtiles // 2 :
                ]
            )
            return dk, dv, q_scales_head, q_scales_m, do_scales_head, do_scales_m

        def pingpong(offset_m, inner_state):
            (
                dk,
                dv,
                q_scales_head_pong,
                q_scales_m_pong,
                do_scales_head_pong,
                do_scales_m_pong,
            ) = _unpack_state(inner_state)

            next_offset_m = offset_m + tile_m
            next_offset_m_mx = next_offset_m // 32
            store_q_tile_to_lds(prefetch_q_tile(next_offset_m), lds_q_ping)
            q_scales_head_ping = prefetch_q_scale_head_2d_tile(next_offset_m_mx)
            q_scales_m_ping = prefetch_q_scale_m_2d_tile(next_offset_m_mx)
            store_do_tile_to_lds(prefetch_do_tile(next_offset_m), lds_do_ping)
            do_scales_head_ping = prefetch_do_scale_head_2d_tile(next_offset_m_mx)
            do_scales_m_ping = prefetch_do_scale_m_2d_tile(next_offset_m_mx)

            qk = compute_qk(
                lds_q_pong,
                q_scales_head_pong,
                lds_k,
                k_scales_head,
            )
            p = softmax(qk, offset_m)
            mxquant_m_and_store_to_lds(p, lds_ppt_shuffle, lds_ppt_scale_shuffle)
            gpu.barrier()
            dv = compute_dv(
                dv,
                lds_ppt_shuffle,
                lds_ppt_scale_shuffle,
                lds_do_pong,
                do_scales_m_pong,
            )
            dp = compute_dp(lds_do_pong, do_scales_head_pong, lds_v, lds_v_scale)
            ds = compute_ds(dp, p, offset_m)
            mxquant_m_and_store_to_lds(ds, lds_dst_shuffle, lds_dst_scale_shuffle)
            mxquant_n_and_store_to_lds(ds, lds_ds_shuffle, lds_ds_scale_shuffle)
            gpu.barrier()
            dk = compute_dk(
                dk,
                lds_dst_shuffle,
                lds_dst_scale_shuffle,
                lds_q_pong,
                q_scales_m_pong,
            )
            dq = compute_dq(lds_ds_shuffle, lds_ds_scale_shuffle, lds_k, k_scales_n)
            store_dq_atomic(dq, offset_m)
            hot_loop_scheduler()
            gpu.barrier()

            next_offset_m = offset_m + (tile_m * 2)
            next_offset_m_mx = next_offset_m // 32
            store_q_tile_to_lds(prefetch_q_tile(next_offset_m), lds_q_pong)
            q_scales_head_pong = prefetch_q_scale_head_2d_tile(next_offset_m_mx)
            q_scales_m_pong = prefetch_q_scale_m_2d_tile(next_offset_m_mx)
            store_do_tile_to_lds(prefetch_do_tile(next_offset_m), lds_do_pong)
            do_scales_head_pong = prefetch_do_scale_head_2d_tile(next_offset_m_mx)
            do_scales_m_pong = prefetch_do_scale_m_2d_tile(next_offset_m_mx)

            qk = compute_qk(
                lds_q_ping,
                q_scales_head_ping,
                lds_k,
                k_scales_head,
            )
            p = softmax(qk, offset_m + tile_m)
            mxquant_m_and_store_to_lds(p, lds_ppt_shuffle, lds_ppt_scale_shuffle)
            gpu.barrier()
            dv = compute_dv(
                dv,
                lds_ppt_shuffle,
                lds_ppt_scale_shuffle,
                lds_do_ping,
                do_scales_m_ping,
            )
            dp = compute_dp(lds_do_ping, do_scales_head_ping, lds_v, lds_v_scale)
            ds = compute_ds(dp, p, offset_m + tile_m)
            mxquant_m_and_store_to_lds(ds, lds_dst_shuffle, lds_dst_scale_shuffle)
            mxquant_n_and_store_to_lds(ds, lds_ds_shuffle, lds_ds_scale_shuffle)
            gpu.barrier()
            dk = compute_dk(
                dk,
                lds_dst_shuffle,
                lds_dst_scale_shuffle,
                lds_q_ping,
                q_scales_m_ping,
            )
            dq = compute_dq(lds_ds_shuffle, lds_ds_scale_shuffle, lds_k, k_scales_n)
            store_dq_atomic(dq, offset_m + tile_m)
            hot_loop_scheduler()
            gpu.barrier()

            return _pack_state(
                dk,
                dv,
                q_scales_head_pong,
                q_scales_m_pong,
                do_scales_head_pong,
                do_scales_m_pong,
            )

        if const_expr(causal):
            start_m = (global_offset_n // (tile_m * 2)) * (tile_m * 2)
        else:
            start_m = fx.Index(0)
        start_m_mx = start_m // 32

        store_q_tile_to_lds(prefetch_q_tile(start_m), lds_q_pong)
        q_scales_head_pong = prefetch_q_scale_head_2d_tile(start_m_mx)
        q_scales_m_pong = prefetch_q_scale_m_2d_tile(start_m_mx)
        store_k_tile_to_lds(prefetch_k_tile(), lds_k)
        k_scales_head = prefetch_k_scale_head_2d_tile()
        k_scales_n = prefetch_k_scale_n_2d_tile()
        store_v_tile_to_lds(prefetch_v_tile(), lds_v)
        store_v_scale_tile_to_lds(prefetch_v_scale_tile(), lds_v_scale)
        store_do_tile_to_lds(prefetch_do_tile(start_m), lds_do_pong)
        do_scales_head_pong = prefetch_do_scale_head_2d_tile(start_m_mx)
        do_scales_m_pong = prefetch_do_scale_m_2d_tile(start_m_mx)
        gpu.barrier()
        dk = [acc_init] * dk_n_accs
        dv = [acc_init] * dv_n_accs

        num_tiles_loop = seqlen_rounded // tile_m
        if const_expr((num_tiles_loop % 2) == 1):
            upper_bound = seqlen_rounded - tile_m
            init_state = _pack_state(
                dk,
                dv,
                q_scales_head_pong,
                q_scales_m_pong,
                do_scales_head_pong,
                do_scales_m_pong,
            )
            for iv, inner in range(start_m, upper_bound, tile_m * 2, init=init_state):
                results = yield pingpong(iv, inner)
            (
                dk,
                dv,
                q_scales_head_pong,
                q_scales_m_pong,
                do_scales_head_pong,
                do_scales_m_pong,
            ) = _unpack_state(results)

            curr_m = arith.index(seqlen_rounded - tile_m)
            qk = compute_qk(
                lds_q_pong,
                q_scales_head_pong,
                lds_k,
                k_scales_head,
            )
            p = softmax(qk, curr_m)
            mxquant_m_and_store_to_lds(p, lds_ppt_shuffle, lds_ppt_scale_shuffle)
            gpu.barrier()
            dv = compute_dv(
                dv,
                lds_ppt_shuffle,
                lds_ppt_scale_shuffle,
                lds_do_pong,
                do_scales_m_pong,
            )
            dp = compute_dp(lds_do_pong, do_scales_head_pong, lds_v, lds_v_scale)
            ds = compute_ds(dp, p, curr_m)
            mxquant_m_and_store_to_lds(ds, lds_dst_shuffle, lds_dst_scale_shuffle)
            mxquant_n_and_store_to_lds(ds, lds_ds_shuffle, lds_ds_scale_shuffle)
            gpu.barrier()
            dk = compute_dk(
                dk,
                lds_dst_shuffle,
                lds_dst_scale_shuffle,
                lds_q_pong,
                q_scales_m_pong,
            )
            dq = compute_dq(lds_ds_shuffle, lds_ds_scale_shuffle, lds_k, k_scales_n)
            store_dq_atomic(dq, curr_m)
        else:
            upper_bound = seqlen_rounded - (tile_m * 2)
            init_state = _pack_state(
                dk,
                dv,
                q_scales_head_pong,
                q_scales_m_pong,
                do_scales_head_pong,
                do_scales_m_pong,
            )
            for iv, inner in range(start_m, upper_bound, tile_m * 2, init=init_state):
                results = yield pingpong(iv, inner)
            (
                dk,
                dv,
                q_scales_head_pong,
                q_scales_m_pong,
                do_scales_head_pong,
                do_scales_m_pong,
            ) = _unpack_state(results)

            curr_m = arith.index(seqlen_rounded - tile_m * 2)
            last_m = arith.index(seqlen_rounded - tile_m)
            last_m_mx = last_m // 32
            store_q_tile_to_lds(prefetch_q_tile(last_m), lds_q_ping)
            q_scales_head_ping = prefetch_q_scale_head_2d_tile(last_m_mx)
            q_scales_m_ping = prefetch_q_scale_m_2d_tile(last_m_mx)
            store_do_tile_to_lds(prefetch_do_tile(last_m), lds_do_ping)
            do_scales_head_ping = prefetch_do_scale_head_2d_tile(last_m_mx)
            do_scales_m_ping = prefetch_do_scale_m_2d_tile(last_m_mx)

            qk = compute_qk(
                lds_q_pong,
                q_scales_head_pong,
                lds_k,
                k_scales_head,
            )
            p = softmax(qk, curr_m)
            mxquant_m_and_store_to_lds(p, lds_ppt_shuffle, lds_ppt_scale_shuffle)
            gpu.barrier()
            dv = compute_dv(
                dv,
                lds_ppt_shuffle,
                lds_ppt_scale_shuffle,
                lds_do_pong,
                do_scales_m_pong,
            )
            dp = compute_dp(lds_do_pong, do_scales_head_pong, lds_v, lds_v_scale)
            ds = compute_ds(dp, p, curr_m)
            mxquant_m_and_store_to_lds(ds, lds_dst_shuffle, lds_dst_scale_shuffle)
            mxquant_n_and_store_to_lds(ds, lds_ds_shuffle, lds_ds_scale_shuffle)
            gpu.barrier()
            dk = compute_dk(
                dk,
                lds_dst_shuffle,
                lds_dst_scale_shuffle,
                lds_q_pong,
                q_scales_m_pong,
            )
            dq = compute_dq(lds_ds_shuffle, lds_ds_scale_shuffle, lds_k, k_scales_n)
            store_dq_atomic(dq, curr_m)

            hot_loop_scheduler()
            gpu.barrier()

            curr_m = last_m
            qk = compute_qk(
                lds_q_ping,
                q_scales_head_ping,
                lds_k,
                k_scales_head,
            )
            p = softmax(qk, curr_m)
            mxquant_m_and_store_to_lds(p, lds_ppt_shuffle, lds_ppt_scale_shuffle)
            gpu.barrier()
            dv = compute_dv(
                dv,
                lds_ppt_shuffle,
                lds_ppt_scale_shuffle,
                lds_do_ping,
                do_scales_m_ping,
            )
            dp = compute_dp(lds_do_ping, do_scales_head_ping, lds_v, lds_v_scale)
            ds = compute_ds(dp, p, curr_m)
            mxquant_m_and_store_to_lds(ds, lds_dst_shuffle, lds_dst_scale_shuffle)
            mxquant_n_and_store_to_lds(ds, lds_ds_shuffle, lds_ds_scale_shuffle)
            gpu.barrier()
            dk = compute_dk(
                dk,
                lds_dst_shuffle,
                lds_dst_scale_shuffle,
                lds_q_ping,
                q_scales_m_ping,
            )
            dq = compute_dq(lds_ds_shuffle, lds_ds_scale_shuffle, lds_k, k_scales_n)
            store_dq_atomic(dq, curr_m)

        store_dk_atomic(dk)
        store_dv_atomic(dv)

    # ── Host launcher ──────────────────────────────────────────────────────
    _cache_tag = (tile_m, tile_n, head_dim)

    @flyc.jit
    def launch_attn_bwd(
        arg_dq: fx.Tensor,
        arg_dk: fx.Tensor,
        arg_dv: fx.Tensor,
        arg_q: fx.Tensor,
        arg_q_scale: fx.Tensor,
        arg_k: fx.Tensor,
        arg_k_scale: fx.Tensor,
        arg_v: fx.Tensor,
        arg_v_scale: fx.Tensor,
        arg_do_quant_head: fx.Tensor,
        arg_do_scale: fx.Tensor,
        arg_M: fx.Tensor,
        arg_D: fx.Tensor,
        batch: fx.Int32,
        stride_qo_batch: fx.Int32,
        stride_kv_batch: fx.Int32,
        stride_MD_batch: fx.Int32,
        stride_qkvo_nheads: fx.Int32,
        stride_MD_nheads: fx.Int32,
        stride_q_scale_batch: fx.Int32,
        stride_q_scale_nheads: fx.Int32,
        stride_k_scale_batch: fx.Int32,
        stride_k_scale_nheads: fx.Int32,
        stride_v_scale_batch: fx.Int32,
        stride_v_scale_nheads: fx.Int32,
        stride_do_scale_batch: fx.Int32,
        stride_do_scale_nheads: fx.Int32,
        stream: fx.Stream,
    ):
        _ = _cache_tag
        allocator_pong.finalized = False
        allocator_ping.finalized = False
        allocator_k.finalized = False
        allocator_v.finalized = False
        allocator_v_scale.finalized = False
        allocator_ppt_shuffle.finalized = False
        allocator_ppt_scale_shuffle.finalized = False
        allocator_dst_shuffle.finalized = False
        allocator_dst_scale_shuffle.finalized = False
        allocator_ds_shuffle.finalized = False
        allocator_ds_scale_shuffle.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator_pong.finalize()
            allocator_ping.finalize()
            allocator_k.finalize()
            allocator_v.finalize()
            allocator_v_scale.finalize()
            allocator_ppt_shuffle.finalize()
            allocator_ppt_scale_shuffle.finalize()
            allocator_dst_shuffle.finalize()
            allocator_dst_scale_shuffle.finalize()
            allocator_ds_shuffle.finalize()
            allocator_ds_scale_shuffle.finalize()

        gx = num_heads_q
        gy = (seqlen + tile_n - 1) // tile_n
        gz = batch

        launcher = kernel_attn_bwd(
            arg_dq,
            arg_dk,
            arg_dv,
            arg_q,
            arg_q_scale,
            arg_k,
            arg_k_scale,
            arg_v,
            arg_v_scale,
            arg_do_quant_head,
            arg_do_scale,
            arg_M,
            arg_D,
            batch,
            stride_qo_batch,
            stride_kv_batch,
            stride_MD_batch,
            stride_qkvo_nheads,
            stride_MD_nheads,
            stride_q_scale_batch,
            stride_q_scale_nheads,
            stride_k_scale_batch,
            stride_k_scale_nheads,
            stride_v_scale_batch,
            stride_v_scale_nheads,
            stride_do_scale_batch,
            stride_do_scale_nheads,
        )
        if waves_per_eu is not None:
            _wpe = int(waves_per_eu)
            if _wpe >= 1:
                for op in ctx.gpu_module_body.operations:
                    if hasattr(op, "attributes") and op.OPERATION_NAME == "gpu.func":
                        op.attributes["rocdl.waves_per_eu"] = ir.IntegerAttr.get(
                            T.i32, _wpe
                        )
        launcher.launch(
            grid=(gx, gy, gz),
            block=(256, 1, 1),
            stream=stream,
        )

    return launch_attn_bwd


__all__ = ["compile_attn_bwd_mxfp8_gfx950"]
