"""MoE GEMM stage2 small-shape kernel (FP4, BLOCKSIZE=64).

Optimized for small token counts (1, 2, 4, 8) with inter_dim=256 or 512.
No K-loop: all K tiles are statically unrolled at compile time.
BLOCKSIZE=64 (1 wave) minimizes launch overhead.
Output via AtomicAdd bf16x2 with topk_weight multiplication.
"""

import functools
from contextlib import contextmanager

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.kernel_function import CompilationContext

from flydsl.expr import range_constexpr
from flydsl.runtime.device import get_rocm_arch as get_hip_arch

from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr

from flydsl._mlir import ir
from flydsl.expr.typing import T

from flydsl.expr import arith, gpu, buffer_ops, vector, rocdl
from flydsl._mlir.dialects import llvm, scf, memref
from flydsl._mlir.dialects.arith import CmpIPredicate

from .mfma_preshuffle_pipeline import (
    _buffer_load_vec,
    make_preshuffle_b_layout,
    make_preshuffle_scale_layout,
    tile_chunk_coord_i32,
)
from .mfma_epilogues import c_shuffle_epilog
from .layout_utils import crd2idx, idx2crd, get as layout_get


@functools.lru_cache(maxsize=None)
def compile_mixed_moe_gemm2_small(
    *,
    model_dim: int,
    inter_dim: int,
    experts: int,
    topk: int,
    tile_m: int = 32,
    tile_n: int = 32,
    swap_ab: bool = False,
):
    """Compile stage2 small-shape kernel for FP4 MoE.

    Fixed: a_dtype=fp4, b_dtype=fp4, out_dtype=bf16, BLOCKSIZE=64, AtomicAdd.
    inter_dim must be 256 or 512 (1 or 2 K tiles, statically unrolled).

    swap_ab: if True, swap A/B operands in MFMA so the fragment layout gives
    4 consecutive model_dim elements per lane, enabling a direct epilog without
    CShuffle LDS round-trip.
    """
    tile_k = 256
    total_threads = 64
    a_elem_bytes = 1
    b_elem_bytes = 1
    a_elem_vec_pack = 2
    cbsz = 4
    blgp = 4
    pack_M = 2
    pack_N = 2
    pack_K = 2
    elem_bytes = 1
    kpack_bytes = 16
    out_elem_bytes = 2

    num_k_tiles = inter_dim // tile_k
    if num_k_tiles not in (1, 2):
        raise ValueError(
            f"inter_dim must be 256 or 512 for small kernel, got {inter_dim}"
        )

    gpu_arch = get_hip_arch()
    allocator = SmemAllocator(None, arch=gpu_arch)

    tile_k_bytes = tile_k * a_elem_bytes
    m_repeat = tile_m // 16
    num_acc_n = tile_n // 16
    k_unroll = tile_k_bytes // 128
    k_unroll_packed = k_unroll // pack_K
    m_repeat_packed = m_repeat // pack_M
    num_acc_n_packed = num_acc_n // pack_N

    tile_k_packed = tile_k // a_elem_vec_pack
    lds_stride = tile_k_packed  # 128, actual packed row width (no XOR16 swizzle)
    lds_x_bytes = 2 * tile_m * lds_stride * a_elem_bytes
    lds_out_bytes = 2 * tile_m * tile_n
    lds_tid_bytes = tile_m * 4
    lds_total_bytes = max(lds_x_bytes, lds_out_bytes) + lds_tid_bytes
    lds_total_elems = lds_total_bytes

    lds_alloc_bytes = lds_total_elems * a_elem_bytes
    lds_alloc_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_alloc_offset + lds_alloc_bytes

    bytes_x_per_tile = tile_m * tile_k_packed * a_elem_bytes
    bytes_per_thread_x = bytes_x_per_tile // total_threads
    x_load_bytes = 16
    num_x_loads = bytes_per_thread_x // x_load_bytes
    chunk_i32 = x_load_bytes // 4
    tile_k_dwords = (tile_k * a_elem_bytes) // (4 * a_elem_vec_pack)

    _epilog_tag = "swapab" if swap_ab else "cshuffle"
    module_name = (
        f"mfma_moe2s_afp4_wfp4_bf16"
        f"_t{tile_m}x{tile_n}x{tile_k}"
        f"_id{inter_dim}_{_epilog_tag}_v11"
    ).replace("-", "_")

    if True:

        @flyc.kernel
        def moe_gemm2_small(
            arg_out: fx.Tensor,
            arg_x: fx.Tensor,
            arg_w: fx.Tensor,
            arg_scale_x: fx.Tensor,
            arg_scale_w: fx.Tensor,
            arg_sorted_token_ids: fx.Tensor,
            arg_expert_ids: fx.Tensor,
            arg_sorted_weights: fx.Tensor,
            arg_num_valid_ids: fx.Tensor,
            arg_bias: fx.Tensor,
            i32_tokens_in: fx.Int32,
            i32_n_in: fx.Int32,
            i32_k_in: fx.Int32,
            i32_size_expert_ids_in: fx.Int32,
        ):
            arg_out = arg_out.value
            arg_x = arg_x.value
            arg_w = arg_w.value
            arg_scale_x = arg_scale_x.value
            arg_scale_w = arg_scale_w.value
            arg_sorted_token_ids = arg_sorted_token_ids.value
            arg_expert_ids = arg_expert_ids.value
            arg_sorted_weights = arg_sorted_weights.value
            arg_num_valid_ids = arg_num_valid_ids.value
            arg_bias = arg_bias.value

            tokens_in = arith.index_cast(ir.IndexType.get(), i32_tokens_in.ir_value())
            n_in = arith.index_cast(ir.IndexType.get(), i32_n_in.ir_value())
            k_in = arith.index_cast(ir.IndexType.get(), i32_k_in.ir_value())
            size_expert_ids_in = arith.index_cast(
                ir.IndexType.get(), i32_size_expert_ids_in.ir_value()
            )

            tx = gpu.thread_id("x")
            by = gpu.block_id("x")
            bx = gpu.block_id("y")
            by_n = by * arith.constant(tile_n, index=True)
            bx_m = bx * arith.constant(tile_m, index=True)

            x_elem = T.f8
            f32 = T.f32
            i32 = T.i32
            i64 = T.i64
            vec4_f32 = T.vec(4, f32)
            vec16_x = T.vec(16, x_elem)
            vec2_i64 = T.vec(2, i64)

            acc_init = arith.constant_vector(0.0, vec4_f32)
            topk_idx = arith.constant(topk, index=True)
            topk_i32_v = arith.constant(topk)
            tokens_i32 = arith.index_cast(T.i32, tokens_in)
            mask24_i32 = arith.constant(0xFFFFFF)

            numids_rsrc = buffer_ops.create_buffer_resource(
                arg_num_valid_ids,
                max_size=False,
                num_records_bytes=arith.constant(4, type=T.i32),
            )
            num_valid_i32 = buffer_ops.buffer_load(
                numids_rsrc, arith.constant(0, index=True), vec_width=1, dtype=T.i32
            )
            eid_nbytes_idx = size_expert_ids_in * arith.constant(4, index=True)
            eid_nbytes_i32 = arith.index_cast(T.i32, eid_nbytes_idx)
            expert_rsrc = buffer_ops.create_buffer_resource(
                arg_expert_ids, max_size=False, num_records_bytes=eid_nbytes_i32
            )
            expert_i32 = buffer_ops.buffer_load(
                expert_rsrc, bx, vec_width=1, dtype=T.i32
            )

            rocdl.sched_barrier(0)

            # B layout: [E * model_dim, inter_dim] preshuffled
            c_n_total = arith.constant(experts * model_dim, index=True)
            b_layout = make_preshuffle_b_layout(
                arith,
                c_n=c_n_total,
                c_k=k_in // pack_K,
                kpack_bytes=kpack_bytes,
                elem_bytes=b_elem_bytes,
            )
            layout_b = b_layout.layout_b

            sorted_m = size_expert_ids_in * arith.constant(tile_m, index=True)
            c_inter_dim = arith.constant(inter_dim, index=True)
            layout_a_scale = make_preshuffle_scale_layout(
                arith, c_mn=sorted_m, c_k=c_inter_dim
            )
            layout_b_scale = make_preshuffle_scale_layout(
                arith, c_mn=c_n_total, c_k=c_inter_dim
            )

            shape_lds = fx.make_shape(tile_m, lds_stride)
            stride_lds = fx.make_stride(lds_stride, 1)
            layout_lds = fx.make_layout(shape_lds, stride_lds)

            layout_lane16 = fx.make_layout((4, 16), stride=(16, 1))

            base_ptr = allocator.get_base()
            lds_x_ptr = SmemPtr(
                base_ptr, lds_alloc_offset, T.f8, shape=(lds_total_elems,)
            )
            lds_x = lds_x_ptr.get()
            lds_out = SmemPtr(
                base_ptr,
                lds_x_ptr.byte_offset,
                T.bf16,
                shape=(tile_m * tile_n,),
            ).get()
            _lds_tid_off = max(lds_x_bytes, lds_out_bytes)
            lds_tid = SmemPtr(
                base_ptr, lds_x_ptr.byte_offset + _lds_tid_off, T.i32, shape=(tile_m,)
            ).get()

            c_a_pack = arith.constant(a_elem_vec_pack, index=True)
            c_elem_bytes = arith.constant(a_elem_bytes, index=True)

            x_row_count = tokens_in * topk_idx
            x_nbytes_idx = (x_row_count * k_in * c_elem_bytes) / c_a_pack
            x_nbytes_i32 = arith.index_cast(T.i32, x_nbytes_idx)
            x_rsrc = buffer_ops.create_buffer_resource(
                arg_x, max_size=False, num_records_bytes=x_nbytes_i32
            )

            w_rsrc = buffer_ops.create_buffer_resource(arg_w, max_size=False)

            out_nbytes_idx = (
                tokens_in * n_in * arith.constant(out_elem_bytes, index=True)
            )
            out_nbytes_i32 = arith.index_cast(T.i32, out_nbytes_idx)
            out_rsrc = buffer_ops.create_buffer_resource(
                arg_out, max_size=False, num_records_bytes=out_nbytes_i32
            )

            c32 = arith.constant(32, index=True)
            kblk = k_in / c32
            sx_nbytes_idx = sorted_m * kblk
            sx_nbytes_i32 = arith.index_cast(T.i32, sx_nbytes_idx)
            sx_rsrc = buffer_ops.create_buffer_resource(
                arg_scale_x, max_size=False, num_records_bytes=sx_nbytes_i32
            )

            mn_w = arith.constant(experts * model_dim, index=True)
            sw_nbytes_idx = mn_w * kblk
            sw_nbytes_i32 = arith.index_cast(T.i32, sw_nbytes_idx)
            sw_rsrc = buffer_ops.create_buffer_resource(
                arg_scale_w, max_size=False, num_records_bytes=sw_nbytes_i32
            )

            sorted_nbytes_idx = size_expert_ids_in * arith.constant(
                tile_m * 4, index=True
            )
            sorted_nbytes_i32 = arith.index_cast(T.i32, sorted_nbytes_idx)
            sorted_rsrc = buffer_ops.create_buffer_resource(
                arg_sorted_token_ids,
                max_size=False,
                num_records_bytes=sorted_nbytes_i32,
            )
            sorted_w_rsrc = buffer_ops.create_buffer_resource(
                arg_sorted_weights, max_size=False, num_records_bytes=sorted_nbytes_i32
            )

            rocdl.sched_barrier(0)

            bx_m_i32 = arith.index_cast(T.i32, bx_m)
            blk_valid = arith.cmpi(CmpIPredicate.ult, bx_m_i32, num_valid_i32)

            expert_idx = arith.index_cast(ir.IndexType.get(), expert_i32)
            exp_valid = arith.cmpi(
                CmpIPredicate.ult, expert_i32, arith.constant(experts, type=T.i32)
            )

            def _moe_gemm2_body():
                expert_off_idx = expert_idx * arith.constant(model_dim, index=True)

                # ---- X (A) loading setup ----
                c_k_div4 = (
                    (k_in / c_a_pack) * arith.constant(a_elem_bytes, index=True)
                ) / arith.index(4)

                layout_x_tile_div4 = fx.make_layout(
                    (tile_m, tile_k_dwords), stride=(tile_k_dwords, 1)
                )
                c_chunk_i32 = arith.constant(chunk_i32, index=True)
                tx_i32_base = tx * c_chunk_i32

                def x_tile_chunk_coord_i32(i: int):
                    return tile_chunk_coord_i32(
                        arith,
                        tx_i32_base=tx_i32_base,
                        i=i,
                        total_threads=total_threads,
                        layout_tile_div4=layout_x_tile_div4,
                        chunk_i32=chunk_i32,
                    )

                x_row_base_div4 = []
                x_col_local_i32 = []

                for i in range_constexpr(num_x_loads):
                    row_local, col_local_i32 = x_tile_chunk_coord_i32(i)
                    x_col_local_i32.append(col_local_i32)

                    sorted_row_i = bx_m + row_local
                    fused_i = buffer_ops.buffer_load(
                        sorted_rsrc, sorted_row_i, vec_width=1, dtype=T.i32
                    )
                    t_i32 = arith.andi(fused_i, mask24_i32)
                    s_i32 = arith.shrui(fused_i, arith.constant(24))
                    _fv1 = vector.from_elements(T.vec(1, T.i32), [fused_i])
                    vector.store(_fv1, lds_tid, [row_local])
                    t_valid = arith.cmpi(CmpIPredicate.ult, t_i32, tokens_i32)
                    s_valid = arith.cmpi(CmpIPredicate.ult, s_i32, topk_i32_v)
                    ts_valid = arith.andi(t_valid, s_valid)
                    ts_i32 = t_i32 * topk_i32_v + s_i32
                    ts_safe = arith.select(ts_valid, ts_i32, arith.constant(0))
                    ts_idx = arith.index_cast(ir.IndexType.get(), ts_safe)
                    x_row_base_div4.append(ts_idx * c_k_div4)

                # Wave/lane decomposition (1 wave, 64 threads)
                coord_l16 = idx2crd(tx, layout_lane16)
                lane_div_16 = layout_get(coord_l16, 0)
                lane_mod_16 = layout_get(coord_l16, 1)
                row_a_lds = lane_mod_16
                col_offset_base = lane_div_16 * arith.constant(16, index=True)

                n_tile_base = arith.constant(0, index=True)

                # B weight N-tile precompute
                n_blk_list = []
                n_intra_list = []
                c_n0_static = experts * model_dim // 16
                layout_n_blk_intra = fx.make_layout(
                    (c_n0_static, 16), stride=(16, 1)
                )

                for i in range_constexpr(num_acc_n):
                    c_offset = arith.constant(i * 16, index=True)
                    global_n = expert_off_idx + by_n + c_offset + lane_mod_16
                    coord = idx2crd(global_n, layout_n_blk_intra)
                    n_blk_list.append(layout_get(coord, 0))
                    n_intra_list.append(layout_get(coord, 1))

                # ---- B loading ----
                def load_b_packs_k64(base_k, ku, n_blk, n_intra):
                    c64 = arith.constant(64, index=True)
                    base_k_bytes = base_k * arith.constant(b_elem_bytes, index=True)
                    k0 = base_k_bytes // c64 + arith.constant(ku, index=True)
                    k1 = lane_div_16
                    coord_pack = (
                        n_blk, k0, k1, n_intra, arith.constant(0, index=True)
                    )
                    idx_pack = crd2idx(coord_pack, layout_b)
                    vec_elems = kpack_bytes // b_elem_bytes
                    b16 = _buffer_load_vec(
                        buffer_ops,
                        vector,
                        w_rsrc,
                        idx_pack,
                        elem_type=T.i8,
                        vec_elems=vec_elems,
                        elem_bytes=b_elem_bytes,
                        offset_in_bytes=(b_elem_bytes == 1),
                    )
                    b_i64x2 = vector.bitcast(vec2_i64, b16)
                    b0 = vector.extract(
                        b_i64x2, static_position=[0], dynamic_position=[]
                    )
                    b1 = vector.extract(
                        b_i64x2, static_position=[1], dynamic_position=[]
                    )
                    return b0, b1

                def load_b_tile(base_k):
                    b_tile = []
                    for ku in range_constexpr(k_unroll):
                        packs0, packs1 = [], []
                        for ni in range_constexpr(num_acc_n):
                            b0, b1 = load_b_packs_k64(
                                base_k, ku, n_blk_list[ni], n_intra_list[ni]
                            )
                            packs0.append(b0)
                            packs1.append(b1)
                        b_tile.append((packs0, packs1))
                    return b_tile

                # ---- Scale loading ----
                def load_scale(rsrc, scale_info, ku, mni):
                    k_lane = lane_div_16
                    n_lane = lane_mod_16
                    idx_pack = (
                        mni * scale_info.stride_n0
                        + ku * scale_info.stride_k0
                        + k_lane * scale_info.stride_klane
                        + n_lane
                    )
                    s = buffer_ops.buffer_load(
                        rsrc, idx_pack, vec_width=1, dtype=T.i32
                    )
                    return vector.from_elements(T.vec(1, T.i32), [s])

                def load_b_scale_tile(base_k_scale):
                    b_scale = []
                    for ku in range_constexpr(k_unroll_packed):
                        for ni in range_constexpr(num_acc_n_packed):
                            col_offset_idx = arith.constant(
                                ni * 16 * pack_N, index=True
                            )
                            col_base = by_n + col_offset_idx
                            mni = (expert_off_idx + col_base) // arith.constant(
                                32, index=True
                            )
                            b_scale.append(
                                load_scale(
                                    sw_rsrc, layout_b_scale, ku + base_k_scale, mni
                                )
                            )
                    return b_scale

                def load_a_scale_tile(base_k_scale):
                    a_scale_tile = []
                    for ku in range_constexpr(k_unroll_packed):
                        for mi in range_constexpr(m_repeat_packed):
                            scale = load_scale(
                                sx_rsrc,
                                layout_a_scale,
                                ku + base_k_scale,
                                mi + bx_m // pack_M // 16,
                            )
                            a_scale_tile.append(scale)
                    return a_scale_tile

                # ---- Async DMA: global → LDS ----
                _dma_bytes = 16

                def dma_x_tile_to_lds(base_k, lds_base):
                    c4_idx = arith.index(4)
                    base_k_div4 = (
                        (base_k / c_a_pack)
                        * arith.constant(a_elem_bytes, index=True)
                    ) / arith.index(4)

                    lds_ptr_i64 = None
                    for i in range_constexpr(num_x_loads):
                        global_dw = (
                            x_row_base_div4[i]
                            + base_k_div4
                            + x_col_local_i32[i]
                        )
                        global_offset = arith.index_cast(
                            T.i32, global_dw * c4_idx
                        )

                        if i == 0:
                            lds_addr = (
                                memref.extract_aligned_pointer_as_index(
                                    lds_x
                                )
                                + lds_base
                            )
                            lds_ptr_i64 = rocdl.readfirstlane(
                                T.i64,
                                arith.index_cast(T.i64, lds_addr),
                            )
                        else:
                            lds_ptr_i64 = (
                                lds_ptr_i64
                                + arith.constant(
                                    total_threads * _dma_bytes,
                                    type=T.i64,
                                )
                            )

                        lds_ptr_type = ir.Type.parse("!llvm.ptr<3>")
                        lds_ptr = llvm.inttoptr(
                            lds_ptr_type, lds_ptr_i64
                        )
                        rocdl.raw_ptr_buffer_load_lds(
                            x_rsrc,
                            lds_ptr,
                            arith.constant(_dma_bytes, type=T.i32),
                            global_offset,
                            arith.constant(0, type=T.i32),
                            arith.constant(0, type=T.i32),
                            arith.constant(0, type=T.i32),
                        )

                def lds_load_packs_k64(curr_row_a_lds, col_base, lds_base):
                    idx_a16 = (
                        crd2idx([curr_row_a_lds, col_base], layout_lds)
                        + lds_base
                    )
                    loaded_a16 = vector.load_op(vec16_x, lds_x, [idx_a16])
                    a_i64x2 = vector.bitcast(vec2_i64, loaded_a16)
                    a0 = vector.extract(
                        a_i64x2, static_position=[0], dynamic_position=[]
                    )
                    a1 = vector.extract(
                        a_i64x2, static_position=[1], dynamic_position=[]
                    )
                    return a0, a1

                # ---- MFMA compute (single GEMM, statically unrolled) ----
                def compute_tile(acc_in, b_tile_in, lds_base, a_scale, b_scale):
                    acc_list = list(acc_in)
                    mfma_res_ty = vec4_f32
                    c0_i64 = arith.constant(0, type=T.i64)
                    vec4_i64_ty = T.vec(4, T.i64)
                    vec8_i32_ty = T.vec(8, T.i32)

                    def pack_i64x4_to_i32x8(x0, x1, x2, x3):
                        v4 = vector.from_elements(vec4_i64_ty, [x0, x1, x2, x3])
                        return vector.bitcast(vec8_i32_ty, v4)

                    for ku128 in range_constexpr(k_unroll_packed):
                        for mi in range_constexpr(m_repeat_packed):
                            a_sc = a_scale[ku128 * m_repeat_packed + mi]
                            a_scale_val = vector.extract(
                                a_sc, static_position=[0], dynamic_position=[]
                            )
                            for ni in range_constexpr(num_acc_n_packed):
                                b_sc = b_scale[ku128 * num_acc_n_packed + ni]
                                bs_val = vector.extract(
                                    b_sc, static_position=[0], dynamic_position=[]
                                )
                                for ikxdl in range_constexpr(pack_K):
                                    k_idx = ku128 * pack_K + ikxdl
                                    bp0, bp1 = b_tile_in[k_idx]
                                    col_base = (
                                        col_offset_base
                                        + (k_idx * 128) // a_elem_vec_pack
                                    )
                                    for imxdl in range_constexpr(pack_M):
                                        mi_idx = mi * pack_M + imxdl
                                        mi_val = arith.constant(
                                            mi_idx * 16, index=True
                                        )
                                        curr_row = row_a_lds + mi_val
                                        a0, a1 = lds_load_packs_k64(
                                            curr_row, col_base, lds_base
                                        )
                                        a128 = pack_i64x4_to_i32x8(
                                            a0, a1, c0_i64, c0_i64
                                        )
                                        for inxdl in range_constexpr(pack_N):
                                            ni_idx = ni * pack_N + inxdl
                                            acc_idx = mi_idx * num_acc_n + ni_idx
                                            b0 = bp0[ni_idx]
                                            b1 = bp1[ni_idx]
                                            b128 = pack_i64x4_to_i32x8(
                                                b0, b1, c0_i64, c0_i64
                                            )
                                            rocdl.sched_barrier(0)
                                            if swap_ab:
                                                acc_list[acc_idx] = (
                                                    rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                                                        mfma_res_ty,
                                                        [
                                                            b128,
                                                            a128,
                                                            acc_list[acc_idx],
                                                            blgp,
                                                            cbsz,
                                                            ikxdl * pack_N + inxdl,
                                                            bs_val,
                                                            ikxdl * pack_M + imxdl,
                                                            a_scale_val,
                                                        ],
                                                    )
                                                )
                                            else:
                                                acc_list[acc_idx] = (
                                                    rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                                                        mfma_res_ty,
                                                        [
                                                            a128,
                                                            b128,
                                                            acc_list[acc_idx],
                                                            cbsz,
                                                            blgp,
                                                            ikxdl * pack_M + imxdl,
                                                            a_scale_val,
                                                            ikxdl * pack_N + inxdl,
                                                            bs_val,
                                                        ],
                                                    )
                                                )
                    return acc_list

                # ---- Pipeline: async DMA + compute (no K-loop) ----
                lds_tile_elems = arith.constant(tile_m * lds_stride, index=True)
                lds_base_0 = arith.index(0)
                lds_base_1 = lds_tile_elems

                k0 = arith.index(0)
                dma_x_tile_to_lds(k0, lds_base_0)
                a_scale_0 = load_a_scale_tile(0)
                b_scale_0 = load_b_scale_tile(0)
                b_tile_0 = load_b_tile(k0)

                acc = [acc_init] * (num_acc_n * m_repeat)

                if num_k_tiles == 1:
                    rocdl.s_waitcnt(0xC07F)
                    acc = compute_tile(
                        acc, b_tile_0, lds_base_0, a_scale_0, b_scale_0
                    )
                else:
                    k1_a = arith.index(tile_k)
                    k1_b = arith.index(tile_k // 2)
                    dma_x_tile_to_lds(k1_a, lds_base_1)
                    a_scale_1 = load_a_scale_tile(1)
                    b_scale_1 = load_b_scale_tile(1)
                    b_tile_1 = load_b_tile(k1_b)

                    rocdl.s_waitcnt(0xC07F)
                    acc = compute_tile(
                        acc, b_tile_0, lds_base_0, a_scale_0, b_scale_0
                    )
                    acc = compute_tile(
                        acc, b_tile_1, lds_base_1, a_scale_1, b_scale_1
                    )

                # ---- Epilogue ----
                from flydsl._mlir.dialects import fly as _fly

                _llvm_ptr_ty = ir.Type.parse("!llvm.ptr")
                out_base_ptr = _fly.extract_aligned_pointer_as_index(
                    _llvm_ptr_ty, arg_out
                )
                out_base_i64 = llvm.ptrtoint(T.i64, out_base_ptr)
                out_base_idx = arith.index_cast(ir.IndexType.get(), out_base_i64)

                def _idx_to_llvm_ptr(idx_val, addr_space=1):
                    idx_v = (
                        idx_val._value if hasattr(idx_val, "_value") else idx_val
                    )
                    i64_v = arith.index_cast(T.i64, idx_v)
                    i64_raw = (
                        i64_v._value if hasattr(i64_v, "_value") else i64_v
                    )
                    ptr_ty = ir.Type.parse(f"!llvm.ptr<{addr_space}>")
                    return llvm.inttoptr(ptr_ty, i64_raw)

                # ---- CShuffle epilog (default) ----
                _e_vec = 2
                _cshuffle_nlane = min(32, tile_n // _e_vec)

                def write_row_to_lds(
                    *,
                    mi,
                    ii,
                    row_in_tile,
                    row,
                    row_base_lds,
                    col_base_local,
                    num_acc_n,
                    lds_out,
                ):
                    tw = buffer_ops.buffer_load(
                        sorted_w_rsrc, row, vec_width=1, dtype=f32
                    )
                    for ni in range_constexpr(num_acc_n):
                        col_local = col_base_local + (ni * 16)
                        acc_idx = mi * num_acc_n + ni
                        v = vector.extract(
                            acc[acc_idx],
                            static_position=[ii],
                            dynamic_position=[],
                        )
                        v = v * tw
                        v_out = arith.trunc_f(T.bf16, v)
                        lds_idx = row_base_lds + col_local
                        v1 = vector.from_elements(
                            T.vec(1, T.bf16), [v_out]
                        )
                        vector.store(
                            v1, lds_out, [lds_idx], alignment=2
                        )

                def precompute_row(*, row_local, row):
                    fused2 = memref.load(lds_tid, [row_local])
                    row_i32 = arith.index_cast(T.i32, row)
                    row_valid0 = arith.cmpi(
                        CmpIPredicate.ult, row_i32, num_valid_i32
                    )
                    t = fused2 & mask24_i32
                    s = fused2 >> 24
                    t_ok = arith.cmpi(
                        CmpIPredicate.ult, t, tokens_i32
                    )
                    s_ok = arith.cmpi(
                        CmpIPredicate.ult, s, topk_i32_v
                    )
                    row_valid = arith.andi(
                        row_valid0, arith.andi(t_ok, s_ok)
                    )
                    t_idx = arith.index_cast(ir.IndexType.get(), t)
                    row_byte_base = (
                        out_base_idx
                        + t_idx
                        * arith.constant(
                            model_dim * out_elem_bytes, index=True
                        )
                    )
                    return ((fused2, row_byte_base), row_valid)

                def store_pair(
                    *, row_local, row, row_ctx, col_pair0, col_g0, frag
                ):
                    _fused, row_byte_base = row_ctx
                    byte_off_col = col_g0 * arith.constant(
                        out_elem_bytes, index=True
                    )
                    ptr_addr_idx = row_byte_base + byte_off_col
                    out_ptr_v = _idx_to_llvm_ptr(ptr_addr_idx)
                    frag_v = (
                        frag._value
                        if hasattr(frag, "_value")
                        else frag
                    )
                    llvm.AtomicRMWOp(
                        llvm.AtomicBinOp.fadd,
                        out_ptr_v,
                        frag_v,
                        llvm.AtomicOrdering.monotonic,
                        syncscope="agent",
                        alignment=_e_vec * out_elem_bytes,
                    )

                c_shuffle_epilog(
                    arith=arith,
                    vector=vector,
                    gpu=gpu,
                    scf=scf,
                    range_constexpr=range_constexpr,
                    tile_m=tile_m,
                    tile_n=tile_n,
                    e_vec=_e_vec,
                    cshuffle_nlane=_cshuffle_nlane,
                    block_size=total_threads,
                    m_repeat=m_repeat,
                    num_acc_n=num_acc_n,
                    tx=tx,
                    lane_div_16=lane_div_16,
                    lane_mod_16=lane_mod_16,
                    bx_m=bx_m,
                    by_n=by_n,
                    n_tile_base=n_tile_base,
                    lds_out=lds_out,
                    frag_elem_type=ir.BF16Type.get(),
                    write_row_to_lds=write_row_to_lds,
                    precompute_row=precompute_row,
                    store_pair=store_pair,
                )

            _if_blk = scf.IfOp(blk_valid)
            with ir.InsertionPoint(_if_blk.then_block):
                _ifexpert_of = scf.IfOp(exp_valid)
                with ir.InsertionPoint(_ifexpert_of.then_block):
                    _moe_gemm2_body()
                    scf.YieldOp([])
                scf.YieldOp([])

    _cache_tag = (module_name, model_dim, inter_dim, experts, topk, tile_m, tile_n, swap_ab)

    @flyc.jit
    def launch_moe_gemm2_small(
        arg_out: fx.Tensor,
        arg_x: fx.Tensor,
        arg_w: fx.Tensor,
        arg_scale_x: fx.Tensor,
        arg_scale_w: fx.Tensor,
        arg_sorted_token_ids: fx.Tensor,
        arg_expert_ids: fx.Tensor,
        arg_sorted_weights: fx.Tensor,
        arg_num_valid_ids: fx.Tensor,
        arg_bias: fx.Tensor,
        i32_tokens_in: fx.Int32,
        i32_n_in: fx.Int32,
        i32_k_in: fx.Int32,
        i32_size_expert_ids_in: fx.Int32,
        stream: fx.Stream,
    ):
        _ = _cache_tag
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()

        n_in = arith.index_cast(ir.IndexType.get(), i32_n_in.ir_value())
        gx = n_in / arith.constant(tile_n, index=True)
        gy = arith.index_cast(
            ir.IndexType.get(), i32_size_expert_ids_in.ir_value()
        )

        moe_gemm2_small(
            arg_out,
            arg_x,
            arg_w,
            arg_scale_x,
            arg_scale_w,
            arg_sorted_token_ids,
            arg_expert_ids,
            arg_sorted_weights,
            arg_num_valid_ids,
            arg_bias,
            i32_tokens_in,
            i32_n_in,
            i32_k_in,
            i32_size_expert_ids_in,
        ).launch(
            grid=(gx, gy, 1), block=(total_threads, 1, 1), stream=stream
        )

    return launch_moe_gemm2_small
