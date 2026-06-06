# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# Fresh FlyDSL drop-in replacements for the mxfp4 a4w4 MoE GEMM kernels:
#   - mxfp4_moe_gemm1_a4w4  (gate/up + SiLU*mul + per-32 fp4 requant)   [TODO]
#   - mxfp4_moe_gemm2_a4w4  (down proj + atomic topk reduce)            [in progress]
#
# The FlyDSL gemm1/gemm2 here match the HIP kernels' function AND interface
# exactly (true drop-in); every *other* MoE kernel (sort / quant / sort_scales /
# scatter_reduce) is reused from the mxfp4 (mx_fn) HIP path unchanged.
#
# Reference contract (csrc/kernels/mxfp4_moe/gemm_a4w4/gemm2_a4w4.cuh, gfx950,
# Kimi-K2.5: NE=385, K=512, N_OUT=7168, TOPK=9; BM in {16,32} atomic):
#   A_q (inter)      [max_sorted, K/2=256]      fp4x2  (SORTED order; row=sorted_pos)
#   A_scale          make_preshuffle tile       e8m0
#   B_q (w2)         [E, N_OUT, K/2]            fp4x2  (a16w4 preshuffle, gate_up=False)
#   B_scale          [E*N_OUT, K/32]           e8m0   (a16w4 scale preshuffle)
#   sorted_token_ids [max_sorted]               i32    (token=id&0xFFFFFF)
#   sorted_weights   [max_sorted]               f32
#   out_bf16         [M, N_OUT]                  bf16   (atomic acc*weight add; pre-zeroed)
#   grid=(cumsum/BM)*(N_OUT/256), block=256 (4 waves), K_TILES=2.
#
# a16w4 B-load (gemm2_a4w4.cuh:316/230): base = (e*N_OUT + n_block*256 +
#   wave_n*64 + j*16) * (K/2);  v_voff = (lane/16)*256 + (lane%16)*16 + K_C*2048;
#   two b128 loads at imm {0, 1024}.
# Scale (kBS_*) == FlyDSL make_preshuffle_scale_layout (reused).

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import range_constexpr, const_expr
from flydsl.expr import arith, gpu, buffer_ops, vector, rocdl
from flydsl.expr.typing import T
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm, scf, memref
from flydsl._mlir.dialects.arith import CmpIPredicate
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr

from .mfma_preshuffle_pipeline import (
    make_preshuffle_scale_layout,
    swizzle_xor16,
    lds_store_16b_xor16,
    buffer_copy_gmem16_dwordx4,
    tile_chunk_coord_i32,
)
from .layout_utils import crd2idx, idx2crd, get as layout_get


def make_preshuffle_scale_bases(*, n_out, k, experts):
    """Per-expert / per-N-block scale stride constants (== HIP kBS_* / kAS_*,
    == FlyDSL make_preshuffle_scale_layout). All in dword units.
      kBS_c_n1 = N_OUT/16/2 ;  kBS_c_k1 = (K/32)/4/2
      stride_k0_dw = 64 ;  stride_n0_dw = c_k1*64
      per_expert_dw = c_n1 * stride_n0_dw
      as_per_chunk_dw = ((K/32)/4/2) * 64
    """
    c_n1 = n_out // 16 // 2
    c_k1 = (k // 32) // 4 // 2
    stride_k0_dw = 64
    stride_n0_dw = c_k1 * 64
    per_expert_dw = c_n1 * stride_n0_dw
    as_per_chunk_dw = ((k // 32) // 4 // 2) * 64
    return dict(
        c_n1=c_n1,
        c_k1=c_k1,
        stride_k0_dw=stride_k0_dw,
        stride_n0_dw=stride_n0_dw,
        per_expert_dw=per_expert_dw,
        as_per_chunk_dw=as_per_chunk_dw,
    )


def _ptr_buffer_resource(arg_ptr, num_records_bytes):
    """Build an AMD buffer resource (rsrc) from a raw pointer + byte size.

    Matches mixed_moe_gemm_2stage.py:_ptr_buffer_resource.
    """
    addr = fx.ptrtoint(arg_ptr)
    addr_i64 = arith.index_cast(T.i64, addr)
    return buffer_ops.create_buffer_resource_from_addr(
        addr_i64, num_records_bytes=num_records_bytes
    )


@functools.lru_cache(maxsize=None)
def compile_mxfp4_gemm2_a4w4(
    *,
    experts: int,
    model_dim: int,   # N_OUT (down-proj output) = 7168
    inter_dim: int,   # K (down-proj contraction) = 512
    topk: int,
    BM: int,          # 16 or 32 (atomic)
    b_nt: int = 2,
):
    """Compile a fresh FlyDSL gemm2 that is a drop-in for mxfp4_moe_gemm2_a4w4
    (atomic epilogue, BM in {16,32}). Returns the @flyc.jit launcher.

    Simplest-correct first version: per (m_block, n_block) tile, load A (BM x K
    fp4, sorted-direct) into LDS, loop K (K/256 tiles), fp4 scaled-MFMA into
    accm, then cshuffle->LDS and atomic acc*weight into out[token].
    """
    assert BM in (16, 32), "atomic gemm2 supports BM in {16,32} here"
    assert inter_dim == 512, "gemm2 contract: K==512"
    assert model_dim % 256 == 0

    K = inter_dim
    N_OUT = model_dim
    K_HALF = K // 2
    BN = 256
    BK = 256
    K_TILES = K // BK            # 2
    num_n_blocks = N_OUT // BN   # 28
    num_waves = 4
    total_threads = num_waves * 64
    pack_M = BM // 16            # 2 (BM=32) or 1 (BM=16)
    pack_K = 2
    a_elem_vec_pack = 2          # fp4: 2 per byte
    elem_bytes = 1
    cbsz = 4                     # fp4 sub-format
    blgp = 4

    # scale layout (== HIP kBS_*); reused for A_scale and B_scale.
    _scale_pack_m = 2
    _scale_pack_n = 2

    gpu_arch = get_arch()
    allocator = SmemAllocator(None, arch=gpu_arch, global_sym_name="smem_g2")

    # LDS sizing (Python ints; layouts/types built inside kernel where the MLIR
    # context is active).
    lds_stride = BK              # fp4 elems per LDS row (one K-tile, no pad)
    k_blocks16 = (BK * elem_bytes) // 16
    _a_lds_bytes = BM * lds_stride          # fp4-bytes for one A K-tile
    _acc_lds_elems = BM * BN                # f32 cshuffle buffer
    _lds_a_offset = 0
    _lds_acc_offset = ((_a_lds_bytes + 15) // 16) * 16
    allocator.ptr = _lds_acc_offset + _acc_lds_elems * 4

    # A-load tiling (a_elem_bytes==1 fp4 convention, 16B dwordx4 loads).
    x_load_bytes = 16
    num_x_loads = (BM * BK * elem_bytes) // total_threads // x_load_bytes
    chunk_i32 = x_load_bytes // 4
    tile_k_dwords = (BK * elem_bytes) // (4 * a_elem_vec_pack)
    c_k_div4 = (K // a_elem_vec_pack) * elem_bytes // 4   # A_q row stride (dwords)

    # scale (make_preshuffle == HIP kBS_*/kAS_*) constants
    _sc = make_preshuffle_scale_bases(n_out=N_OUT, k=K, experts=experts)
    kBS_per_expert_dw = _sc["per_expert_dw"]
    kBS_stride_n0_dw = _sc["stride_n0_dw"]
    kAS_per_chunk_dw = _sc["as_per_chunk_dw"]
    kSubBlocks = 1 if BM < 32 else BM // 32   # 1 for BM in {16,32}
    _mni_n0 = BN // 16 // 2                    # 8
    _mni_wn = BN // 64 // 2                    # 2
    _as_chunk_div = BM                         # atomic: chunk_base = m_row/BM

    module_name = f"mxfp4_g2_a4w4_E{experts}_K{K}_N{N_OUT}_TOPK{topk}_BM{BM}_v0"

    @flyc.kernel
    def gemm2_kernel(
        arg_out: fx.Pointer,         # out_bf16 [M, N_OUT]
        arg_a: fx.Pointer,           # A_q (inter) [max_sorted, K/2] fp4
        arg_b: fx.Pointer,           # B_q (w2)    [E, N_OUT, K/2] fp4 a16w4
        arg_a_scale: fx.Pointer,     # A_scale (make_preshuffle)
        arg_b_scale: fx.Pointer,     # B_scale (make_preshuffle)
        arg_sorted_token_ids: fx.Pointer,
        arg_sorted_expert_ids: fx.Pointer,
        arg_sorted_weights: fx.Pointer,
        arg_num_valid_ids: fx.Pointer,   # cumsum_tensor [1]
        i32_tokens: fx.Int32,            # M (logical tokens)
        i32_max_sorted: fx.Int32,
    ):
        # types/layouts (MLIR context active here)
        vec16_x = T.vec(16, T.i8)
        vec2_i64 = T.vec(2, T.i64)
        vec4_i64 = T.vec(4, T.i64)
        vec8_i32 = T.vec(8, T.i32)
        vec4_f32 = T.vec(4, T.f32)
        vec2_i32 = T.vec(2, T.i32)
        i32 = T.i32
        f32 = T.f32
        layout_lds = fx.make_layout((BM, lds_stride), (lds_stride, 1))

        tx = gpu.thread_id("x")
        m_block = gpu.block_id("x")
        n_block = gpu.block_id("y")

        # wave / lane decomposition (4 waves x 64 lanes).
        coord_wl = idx2crd(tx, fx.make_layout((num_waves, 64), (64, 1)))
        wave = layout_get(coord_wl, 0)
        lane = layout_get(coord_wl, 1)
        coord_l16 = idx2crd(lane, fx.make_layout((4, 16), (16, 1)))
        lane_div_16 = layout_get(coord_l16, 0)
        lane_mod_16 = layout_get(coord_l16, 1)
        wave_n = wave

        c_tokens = arith.index_cast(ir.IndexType.get(), i32_tokens.ir_value())
        c_max_sorted = arith.index_cast(
            ir.IndexType.get(), i32_max_sorted.ir_value()
        )

        # --- buffer resources (byte sizes) ---
        out_nbytes = arith.index_cast(
            T.i32, c_tokens * arith.constant(N_OUT * 2, index=True)
        )
        out_rsrc = _ptr_buffer_resource(arg_out, out_nbytes)
        a_nbytes = arith.index_cast(
            T.i32, c_max_sorted * arith.constant(K_HALF, index=True)
        )
        a_rsrc = _ptr_buffer_resource(arg_a, a_nbytes)
        b_rsrc = _ptr_buffer_resource(
            arg_b, arith.constant(experts * N_OUT * K_HALF, type=T.i32)
        )
        eid_rsrc = _ptr_buffer_resource(
            arg_sorted_expert_ids,
            arith.index_cast(
                T.i32, (c_max_sorted / arith.constant(BM, index=True))
                * arith.constant(4, index=True),
            ),
        )
        numids_rsrc = _ptr_buffer_resource(
            arg_num_valid_ids, arith.constant(4, type=T.i32)
        )
        sti_rsrc = _ptr_buffer_resource(
            arg_sorted_token_ids,
            arith.index_cast(T.i32, c_max_sorted * arith.constant(4, index=True)),
        )
        sw_rsrc = _ptr_buffer_resource(
            arg_sorted_weights,
            arith.index_cast(T.i32, c_max_sorted * arith.constant(4, index=True)),
        )

        num_valid = buffer_ops.buffer_load(
            numids_rsrc, arith.constant(0, index=True), vec_width=1, dtype=T.i32
        )
        m_row = m_block * arith.constant(BM, index=True)
        m_row_i32 = arith.index_cast(T.i32, m_row)
        blk_valid = arith.cmpi(CmpIPredicate.ult, m_row_i32, num_valid)
        expert_i32 = buffer_ops.buffer_load(
            eid_rsrc, m_block, vec_width=1, dtype=T.i32
        )

        # LDS: A tile (fp4 bytes) + cshuffle acc (f32).
        lds_base = allocator.get_base()
        lds_a = SmemPtr(lds_base, _lds_a_offset, T.i8, shape=(_a_lds_bytes,)).get()
        lds_acc = SmemPtr(
            lds_base, _lds_acc_offset, T.f32, shape=(_acc_lds_elems,)
        ).get()

        # ---- A-load (build/test increment 2): A_q is sorted inter, read
        # directly by sorted_pos = m_row + row_local (NO token gather). One
        # K-tile (BK fp4) at a time into lds_a with xor16 swizzle. ----
        tx_i32_base = tx * arith.constant(chunk_i32, index=True)
        layout_x_tile = fx.make_layout((BM, tile_k_dwords), (tile_k_dwords, 1))
        x_row_local = []
        x_col_local = []
        x_row_base_div4 = []
        for i in range_constexpr(num_x_loads):
            _rl, _cl = tile_chunk_coord_i32(
                arith,
                tx_i32_base=tx_i32_base,
                i=i,
                total_threads=total_threads,
                layout_tile_div4=layout_x_tile,
                chunk_i32=chunk_i32,
            )
            x_row_local.append(_rl)
            x_col_local.append(_cl)
            _row = m_row + _rl
            x_row_base_div4.append(_row * arith.constant(c_k_div4, index=True))

        def load_a_tile(base_k):
            base_k_div4 = (
                (base_k / arith.constant(a_elem_vec_pack, index=True))
                * arith.constant(elem_bytes, index=True)
            ) / arith.index(4)
            parts = []
            for i in range_constexpr(num_x_loads):
                idx = x_row_base_div4[i] + base_k_div4 + x_col_local[i]
                v = buffer_copy_gmem16_dwordx4(
                    buffer_ops,
                    vector,
                    elem_type=T.i8,
                    idx_i32=idx,
                    rsrc=a_rsrc,
                    vec_elems=16,
                )
                parts.append(vector.bitcast(T.vec(4, i32), v))
            return parts

        def store_a_tile(parts):
            for i in range_constexpr(num_x_loads):
                lds_store_16b_xor16(
                    arith,
                    vector,
                    lds_memref=lds_a,
                    vec16_ty=vec16_x,
                    layout_lds=layout_lds,
                    row_local=x_row_local[i],
                    col_local_i32=x_col_local[i],
                    tx_c4=arith.index(4),
                    k_blocks16=k_blocks16,
                    lds_base=arith.index(0),
                    vec_part_i32x4=parts[i],
                    elem_bytes=elem_bytes,
                )

        _a_parts0 = load_a_tile(arith.index(0))
        store_a_tile(_a_parts0)
        gpu.barrier()

        # ---- a16w4 B-load (build/test increment 3): exact HIP issue_b_load_j.
        #   base = (e*N_OUT + n_block*256 + wave_n*64 + j*16) * K_HALF   [bytes]
        #   v_voff = (lane/16)*256 + (lane%16)*16 + k_c*2048             [bytes]
        #   two b128 (16B) loads per j at +0 and +1024 bytes (= +0/+256 dwords)
        # -> b_tile[j] = [(b0,b1)_mfmaK0, (b0,b1)_mfmaK1]; each (b0,b1) is the
        #    lane's 32 fp4 for one MFMA-K=128.
        e_idx = arith.index_cast(ir.IndexType.get(), expert_i32)
        _c256 = arith.constant(256, index=True)
        _c64 = arith.constant(64, index=True)
        _c16 = arith.constant(16, index=True)
        _c2048 = arith.constant(2048, index=True)
        _cKH = arith.constant(K_HALF, index=True)
        _cN = arith.constant(N_OUT, index=True)

        def load_b_tile(k_c):
            b_tile = []
            for j in range_constexpr(4):
                n_off = (
                    n_block * _c256
                    + wave_n * _c64
                    + arith.constant(j * 16, index=True)
                )
                base_bytes = (e_idx * _cN + n_off) * _cKH
                v_voff = (
                    lane_div_16 * _c256 + lane_mod_16 * _c16 + k_c * _c2048
                )
                base_dw = (base_bytes + v_voff) / arith.index(4)
                b_j = []
                for sub in range_constexpr(2):
                    off = base_dw + arith.constant(sub * 256, index=True)
                    v16 = buffer_ops.buffer_load(
                        b_rsrc, off, vec_width=4, dtype=i32, cache_modifier=b_nt
                    )
                    i64x2 = vector.bitcast(vec2_i64, v16)
                    b0 = vector.extract(
                        i64x2, static_position=[0], dynamic_position=[]
                    )
                    b1 = vector.extract(
                        i64x2, static_position=[1], dynamic_position=[]
                    )
                    b_j.append((b0, b1))
                b_tile.append(b_j)
            return b_tile

        _b_tile0 = load_b_tile(arith.index(0))

        # ---- scale load (build/test increment 4): make_preshuffle A/B scale.
        #   a_scale dword off = (m_row/_as_chunk_div + sub)*kAS_per_chunk_dw
        #                     + (lane/16*16 + lane%16) + ku*64
        #   b_scale dword off = e*kBS_per_expert_dw
        #                     + (mni_base+mw)*kBS_stride_n0_dw
        #                     + (lane/16*16 + lane%16) + ku*64
        a_scale_nbytes = arith.index_cast(
            T.i32,
            (c_max_sorted / arith.constant(_as_chunk_div, index=True)
             + arith.constant(2, index=True))
            * arith.constant(kAS_per_chunk_dw * 4, index=True),
        )
        a_scale_rsrc = _ptr_buffer_resource(arg_a_scale, a_scale_nbytes)
        b_scale_rsrc = _ptr_buffer_resource(
            arg_b_scale,
            arith.constant(experts * kBS_per_expert_dw * 4, type=T.i32),
        )
        _scale_lane_dw = lane_div_16 * _c16 + lane_mod_16   # (lane/16*16+lane%16)
        _chunk_base = m_row / arith.constant(_as_chunk_div, index=True)
        _mni_base = (
            n_block * arith.constant(_mni_n0, index=True)
            + wave_n * arith.constant(_mni_wn, index=True)
        )

        def load_scales(ku):
            # returns (a_scale_dword, [b_scale_dword_mw0, b_scale_dword_mw1])
            ku_off = ku * arith.constant(64, index=True)
            a_base = (
                _chunk_base * arith.constant(kAS_per_chunk_dw, index=True)
                + _scale_lane_dw + ku_off
            )
            a_sc = buffer_ops.buffer_load(
                a_scale_rsrc, a_base, vec_width=1, dtype=T.i32, cache_modifier=0
            )
            b_scs = []
            for mw in range_constexpr(2):
                b_base = (
                    e_idx * arith.constant(kBS_per_expert_dw, index=True)
                    + (_mni_base + arith.constant(mw, index=True))
                    * arith.constant(kBS_stride_n0_dw, index=True)
                    + _scale_lane_dw + ku_off
                )
                b_scs.append(
                    buffer_ops.buffer_load(
                        b_scale_rsrc, b_base, vec_width=1, dtype=T.i32,
                        cache_modifier=0,
                    )
                )
            return a_sc, b_scs

        _a_sc0, _b_scs0 = load_scales(arith.index(0))

        # ---- fp4 scaled-MFMA (build/test increment 5) ----
        # MFMA 16x16x128 f8f6f4: a128/b128 = pack_i64x4_to_i32x8(b0,b1,0,0);
        # op_sel selects the e8m0 byte in the make_preshuffle scale dword.
        # (op_sel numeric correctness to be validated vs HIP cos.)
        row_a_lds = lane_mod_16
        col_offset_base = lane_div_16 * _c16
        m_repeat = BM // 16                  # 2 (BM=32)
        m_repeat_packed = m_repeat // pack_M  # 1
        num_acc_n = BN // num_waves // 16     # 4
        num_acc_n_packed = num_acc_n // 2     # 2 (pack_N=2)
        pack_N = 2

        def _pack_i64x4(x0, x1):
            c0 = arith.constant(0, type=T.i64)
            v4 = vector.from_elements(vec4_i64, [x0, x1, c0, c0])
            return vector.bitcast(vec8_i32, v4)

        def lds_load_packs_k64(curr_row, col_base):
            col_swz = swizzle_xor16(curr_row, col_base, k_blocks16)
            idx_a = crd2idx([curr_row, col_swz], layout_lds)
            loaded = vector.load_op(vec16_x, lds_a, [idx_a])
            a_i64x2 = vector.bitcast(vec2_i64, loaded)
            a0 = vector.extract(a_i64x2, static_position=[0], dynamic_position=[])
            a1 = vector.extract(a_i64x2, static_position=[1], dynamic_position=[])
            return a0, a1

        acc_init = arith.constant_vector(0.0, vec4_f32)
        accm = [acc_init] * (m_repeat * num_acc_n)   # mi_idx*num_acc_n + ni_idx

        def mfma_ktile(b_tile, a_sc, b_scs):
            for k_idx in range_constexpr(2):       # 2 MFMA-K (128) per K-tile
                ikxdl = k_idx
                col_base = col_offset_base + arith.constant(
                    (k_idx * 128) // a_elem_vec_pack, index=True
                )
                a_scale_val = a_sc
                for imxdl in range_constexpr(pack_M):
                    mi_idx = imxdl                  # m_repeat_packed=1
                    curr_row = row_a_lds + arith.constant(mi_idx * 16, index=True)
                    a0, a1 = lds_load_packs_k64(curr_row, col_base)
                    a128 = _pack_i64x4(a0, a1)
                    # A-scale op_sel (HIP CBID) = k*_scale_pack_m + i_within.
                    # The stride is the FIXED m-half count in the make_preshuffle
                    # scale dword (_scale_pack_m=2), NOT pack_M (which is 1 at
                    # BM=16, giving the wrong byte for the 2nd MFMA-K).
                    op_sel_a = ikxdl * _scale_pack_m + imxdl
                    for ni in range_constexpr(num_acc_n_packed):
                        b_scale_val = b_scs[ni]
                        for inxdl in range_constexpr(pack_N):
                            ni_idx = ni * pack_N + inxdl
                            b0, b1 = b_tile[ni_idx][k_idx]
                            b128 = _pack_i64x4(b0, b1)
                            op_sel_b = ikxdl * pack_N + inxdl
                            acc_idx = mi_idx * num_acc_n + ni_idx
                            accm[acc_idx] = rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                                vec4_f32,
                                [
                                    a128, b128, accm[acc_idx],
                                    cbsz, blgp,
                                    op_sel_a, a_scale_val,
                                    op_sel_b, b_scale_val,
                                ],
                            )

        mfma_ktile(_b_tile0, _a_sc0, _b_scs0)

        # ---- K-tile 1 (build/test increment 5b) ----
        gpu.barrier()
        _a_parts1 = load_a_tile(arith.constant(BK, index=True))
        store_a_tile(_a_parts1)
        gpu.barrier()
        _b_tile1 = load_b_tile(arith.index(1))
        _a_sc1, _b_scs1 = load_scales(arith.index(1))
        mfma_ktile(_b_tile1, _a_sc1, _b_scs1)

        # ---- cshuffle + atomic epilogue (build/test increment 6) ----
        # accm[mi*4+J][v]  ->  lds_acc[(mi*16 + lane/16*4 + v)*BN + col],
        #   col = wave_n*64 + J*16 + lane%16
        gpu.barrier()
        c_BN = arith.constant(BN, index=True)
        for i in range_constexpr(m_repeat):
            for J in range_constexpr(num_acc_n):
                col = (
                    wave_n * _c64
                    + arith.constant(J * 16, index=True)
                    + lane_mod_16
                )
                row_base = arith.constant(i * 16, index=True) + lane_div_16 * arith.constant(
                    4, index=True
                )
                acc_v = accm[i * num_acc_n + J]
                for v in range_constexpr(4):
                    lds_idx = (
                        (row_base + arith.constant(v, index=True)) * c_BN + col
                    )
                    val = vector.extract(
                        acc_v, static_position=[v], dynamic_position=[]
                    )
                    v1 = vector.from_elements(T.vec(1, f32), [val])
                    vector.store(v1, lds_acc, [lds_idx], alignment=4)
        gpu.barrier()

        # read by (m_lane=tid/32, n_lane=tid%32), atomic acc*weight -> out[token]
        _c32 = arith.constant(32, index=True)
        m_lane = tx / _c32
        n_lane = tx % _c32
        col_start = n_lane * arith.constant(2, index=True)
        out_base_idx = arith.index_cast(ir.IndexType.get(), fx.ptrtoint(arg_out))
        c_tokens_i32 = arith.index_cast(T.i32, c_tokens)
        _mask24 = arith.constant(0xFFFFFF, type=T.i32)
        M_REPS = BM // 8
        bf16x2 = T.vec(2, T.bf16)
        for mr in range_constexpr(M_REPS):
            row_in_block = arith.constant(mr * 8, index=True) + m_lane
            sorted_pos = m_row + row_in_block
            packed = buffer_ops.buffer_load(
                sti_rsrc, sorted_pos, vec_width=1, dtype=T.i32
            )
            token = arith.andi(packed, _mask24)
            # guard padding rows: sorted_pos < cumsum (HIP per-row bound) AND
            # token < M.  Launching max_sorted/BM blocks means trailing rows are
            # padding (uninitialized A_q); skipping them avoids garbage atomics.
            sorted_pos_i32 = arith.index_cast(T.i32, sorted_pos)
            pos_ok = arith.cmpi(CmpIPredicate.ult, sorted_pos_i32, num_valid)
            tok_lt_m = arith.cmpi(CmpIPredicate.ult, token, c_tokens_i32)
            tok_ok = arith.andi(pos_ok, tok_lt_m)
            _if_tok = scf.IfOp(tok_ok)
            with ir.InsertionPoint(_if_tok.then_block):
                token_idx = arith.index_cast(ir.IndexType.get(), token)
                weight = buffer_ops.buffer_load(
                    sw_rsrc, sorted_pos, vec_width=1, dtype=f32
                )
                row_out_base = (
                    token_idx * arith.constant(N_OUT, index=True)
                    + n_block * _c256
                    + col_start
                )
                for s in range_constexpr(4):
                    lds_row_off = row_in_block * c_BN + col_start + arith.constant(
                        s * 64, index=True
                    )
                    e0 = vector.load_op(
                        T.vec(1, f32), lds_acc, [lds_row_off]
                    )
                    e1 = vector.load_op(
                        T.vec(1, f32),
                        lds_acc,
                        [lds_row_off + arith.constant(1, index=True)],
                    )
                    f0 = vector.extract(e0, static_position=[0], dynamic_position=[])
                    f1 = vector.extract(e1, static_position=[0], dynamic_position=[])
                    f0w = arith.mulf(f0, weight)
                    f1w = arith.mulf(f1, weight)
                    b0 = arith.trunc_f(T.bf16, f0w)
                    b1 = arith.trunc_f(T.bf16, f1w)
                    pk = vector.from_elements(bf16x2, [b0, b1])
                    out_addr_idx = (
                        out_base_idx
                        + (row_out_base + arith.constant(s * 64, index=True))
                        * arith.constant(2, index=True)
                    )
                    out_addr_i64 = arith.index_cast(T.i64, out_addr_idx)
                    out_ptr = llvm.inttoptr(
                        ir.Type.parse("!llvm.ptr<1>"), out_addr_i64
                    )
                    pk_raw = pk._value if hasattr(pk, "_value") else pk
                    llvm.AtomicRMWOp(
                        llvm.AtomicBinOp.fadd,
                        out_ptr,
                        pk_raw,
                        llvm.AtomicOrdering.monotonic,
                        syncscope="agent",
                        alignment=4,
                    )
                scf.YieldOp([])

    @flyc.jit
    def launch_gemm2(
        arg_out: fx.Pointer,
        arg_a: fx.Pointer,
        arg_b: fx.Pointer,
        arg_a_scale: fx.Pointer,
        arg_b_scale: fx.Pointer,
        arg_sorted_token_ids: fx.Pointer,
        arg_sorted_expert_ids: fx.Pointer,
        arg_sorted_weights: fx.Pointer,
        arg_num_valid_ids: fx.Pointer,
        i32_tokens: fx.Int32,
        i32_max_sorted: fx.Int32,
        i32_total_m_blocks: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()
        gx = arith.index_cast(ir.IndexType.get(), i32_total_m_blocks.ir_value())
        gemm2_kernel(
            arg_out, arg_a, arg_b, arg_a_scale, arg_b_scale,
            arg_sorted_token_ids, arg_sorted_expert_ids, arg_sorted_weights,
            arg_num_valid_ids, i32_tokens, i32_max_sorted,
        ).launch(
            grid=(gx, num_n_blocks, 1),
            block=(total_threads, 1, 1),
            smem=65536,
            stream=stream,
        )

    return launch_gemm2


def _f32_to_e2m1(x_f32):
    """Software f32 -> fp4(e2m1) 4-bit int (matches HIP cvt + fp4_utils:
    saturate / denormal / round-to-nearest-even)."""
    T_i32 = T.i32
    qx = x_f32.bitcast(T_i32)
    s = qx & arith.constant(0x80000000, type=T_i32)
    qx_abs = qx & arith.constant(0x7FFFFFFF, type=T_i32)
    c_3F8 = arith.constant(0x3F800000, type=T_i32)
    c_40C = arith.constant(0x40C00000, type=T_i32)
    c_4A8 = arith.constant(0x4A800000, type=T_i32)
    dmask = arith.cmpi(CmpIPredicate.ult, qx_abs, c_3F8)
    nmask = arith.andi(
        arith.cmpi(CmpIPredicate.ult, qx_abs, c_40C),
        arith.cmpi(CmpIPredicate.uge, qx_abs, c_3F8),
    )
    dn = qx_abs.bitcast(T.f32) + c_4A8.bitcast(T.f32)
    dnx = dn.bitcast(T_i32) - c_4A8
    modd = (qx_abs >> arith.constant(22, type=T_i32)) & arith.constant(1, type=T_i32)
    nx = (qx_abs + arith.constant(0xC11FFFFF, type=T_i32) + modd) >> arith.constant(
        22, type=T_i32
    )
    e2 = arith.select(nmask, nx, arith.constant(0x7, type=T_i32))
    e2 = arith.select(dmask, dnx, e2)
    return (s >> arith.constant(28, type=T_i32)) | e2


@functools.lru_cache(maxsize=None)
def compile_mxfp4_gemm1_a4w4(
    *,
    experts: int,
    model_dim: int,   # D_HIDDEN (gemm1 contraction K) = 7168
    inter_dim: int,   # D_INTER  (gemm1 output inter)  = 512
    topk: int,
    BM: int,          # 16 (inline) or 32
    use_nt: bool = True,
    inline_quant: bool = False,
):
    """Fresh FlyDSL gemm1 drop-in for mxfp4_moe_gemm1_a4w4 (gate/up a16w4 +
    SiLU*mul + per-32 fp4 requant).  K=D_HIDDEN, N_OUT=2*D_INTER (gate+up
    interleaved), output inter = D_INTER fp4 (sorted) + make_preshuffle scale.

    Simplest-correct first: NT path (pre-quant A_q), simple K-loop, software
    f32->fp4 requant (HW cvt.scalef32.pk_fp4 isn't exposed in rocdl).
    """
    assert BM in (16, 32), "gemm1 here supports BM in {16,32}"

    K = model_dim                # 7168 (contraction)
    N_OUT = 2 * inter_dim        # 1024 (gate+up)
    N_INTER = inter_dim          # 512  (output inter)
    K_HALF = K // 2
    BN = 256
    BK = 256
    K_TILES = K // BK            # 28
    num_n_blocks = N_OUT // BN   # 4
    num_waves = 4
    total_threads = num_waves * 64
    pack_M = BM // 16
    a_elem_vec_pack = 2
    elem_bytes = 1
    cbsz = 4
    blgp = 4
    _scale_pack_m = 2
    b_nt = 2

    # B-scale (gate/up a16w4) — kBS_* over N_OUT=2*D_INTER.
    _bsc = make_preshuffle_scale_bases(n_out=N_OUT, k=K, experts=experts)
    kBS_per_expert_dw = _bsc["per_expert_dw"]
    kBS_stride_n0_dw = _bsc["stride_n0_dw"]
    kBS_stride_k0_dw = _bsc["stride_k0_dw"]   # 64
    # A-scale (activation, K=D_HIDDEN) chunk = 28 K-tiles.
    kAS_c_k1 = (K // 32) // 4 // 2
    kAS_per_chunk_dw = 1 * kAS_c_k1 * 64
    _as_chunk_div = 32                          # gemm1 A-scale: chunk = m_row/32
    _mni_n0 = BN // 16 // 2
    _mni_wn = BN // 64 // 2

    # output inter requant make_preshuffle (kAS over N_INTER).
    kOS_c_k1 = (N_INTER // 32) // 4 // 2
    kOS_per_chunk_dw = 1 * kOS_c_k1 * 64

    gpu_arch = get_arch()
    allocator = SmemAllocator(None, arch=gpu_arch, global_sym_name="smem_g1")

    lds_stride = BK
    k_blocks16 = (BK * elem_bytes) // 16
    _a_lds_bytes = BM * lds_stride
    _acc_lds_elems = BM * BN
    _lds_a_offset = 0
    _lds_acc_offset = ((_a_lds_bytes + 15) // 16) * 16
    allocator.ptr = _lds_acc_offset + _acc_lds_elems * 4

    # A-load tiling (gather rows via m_indices; per-K-tile, 16B dwordx4 loads).
    x_load_bytes = 16
    num_x_loads = (BM * BK * elem_bytes) // total_threads // x_load_bytes
    chunk_i32 = x_load_bytes // 4
    tile_k_dwords = (BK * elem_bytes) // (4 * a_elem_vec_pack)
    c_k_div4 = (K // a_elem_vec_pack) * elem_bytes // 4   # A_q row stride (dwords)

    module_name = (
        f"mxfp4_g1_a4w4_E{experts}_K{K}_N{N_OUT}_TOPK{topk}_BM{BM}"
        f"_{'IQ' if inline_quant else 'NT'}_v0"
    )

    @flyc.kernel
    def gemm1_kernel(
        arg_a: fx.Pointer,           # A_q [max_sorted, K/2] fp4 (pre-quant)
        arg_a_scale: fx.Pointer,     # A_scale (make_preshuffle, sorted)
        arg_b: fx.Pointer,           # B_ps_q (w12) [E, N_OUT, K/2] fp4 a16w4
        arg_b_scale: fx.Pointer,     # B_ps_scale (make_preshuffle)
        arg_sorted_expert_ids: fx.Pointer,
        arg_cumsum: fx.Pointer,
        arg_m_indices: fx.Pointer,
        arg_aq_out: fx.Pointer,      # inter_sorted_quant [max_sorted, N_INTER/2]
        arg_ascale_out: fx.Pointer,  # inter_sorted_shuffled_scale
        arg_hidden: fx.Pointer,      # hidden_states [M, K] bf16 (inline quant)
        i32_tokens: fx.Int32,
        i32_max_sorted: fx.Int32,
    ):
        i32 = T.i32
        f32 = T.f32
        vec16_x = T.vec(16, T.i8)
        vec2_i64 = T.vec(2, T.i64)
        vec4_i64 = T.vec(4, T.i64)
        vec8_i32 = T.vec(8, T.i32)
        vec4_f32 = T.vec(4, T.f32)
        layout_lds = fx.make_layout((BM, lds_stride), (lds_stride, 1))

        tx = gpu.thread_id("x")
        m_block = gpu.block_id("x")
        n_block = gpu.block_id("y")
        coord_wl = idx2crd(tx, fx.make_layout((num_waves, 64), (64, 1)))
        wave = layout_get(coord_wl, 0)
        lane = layout_get(coord_wl, 1)
        coord_l16 = idx2crd(lane, fx.make_layout((4, 16), (16, 1)))
        lane_div_16 = layout_get(coord_l16, 0)
        lane_mod_16 = layout_get(coord_l16, 1)
        wave_n = wave

        c_tokens = arith.index_cast(ir.IndexType.get(), i32_tokens.ir_value())
        c_max_sorted = arith.index_cast(
            ir.IndexType.get(), i32_max_sorted.ir_value()
        )

        # A_q is per-token a_quant [M, K/2] (NOT sorted): bound by M so that
        # padding rows whose gathered token >= M clamp to 0 (OOB buffer load).
        a_nbytes = arith.index_cast(
            T.i32, c_tokens * arith.constant(K_HALF, index=True)
        )
        a_rsrc = _ptr_buffer_resource(arg_a, a_nbytes)
        b_rsrc = _ptr_buffer_resource(
            arg_b, arith.constant(experts * N_OUT * K_HALF, type=T.i32)
        )
        aqout_nbytes = arith.index_cast(
            T.i32, c_max_sorted * arith.constant(N_INTER // 2, index=True)
        )
        aqout_rsrc = _ptr_buffer_resource(arg_aq_out, aqout_nbytes)
        eid_rsrc = _ptr_buffer_resource(
            arg_sorted_expert_ids,
            arith.index_cast(
                T.i32, (c_max_sorted / arith.constant(BM, index=True))
                * arith.constant(4, index=True),
            ),
        )

        expert_i32 = buffer_ops.buffer_load(
            eid_rsrc, m_block, vec_width=1, dtype=T.i32
        )
        m_row = m_block * arith.constant(BM, index=True)

        lds_base = allocator.get_base()
        lds_a = SmemPtr(lds_base, _lds_a_offset, T.i8, shape=(_a_lds_bytes,)).get()
        lds_acc = SmemPtr(
            lds_base, _lds_acc_offset, T.f32, shape=(_acc_lds_elems,)
        ).get()

        # ---- gemm1 GEMM (increment 2): A-load(gather) + a16w4 gate/up B-load
        # + A/B scale + K=K_TILES fp4 MFMA -> accm[kMChunks][4]. ----
        m_idx_rsrc = _ptr_buffer_resource(
            arg_m_indices,
            arith.index_cast(T.i32, c_max_sorted * arith.constant(4, index=True)),
        )
        a_scale_nbytes = arith.index_cast(
            T.i32,
            (c_max_sorted / arith.constant(32, index=True)
             + arith.constant(2, index=True))
            * arith.constant(kAS_per_chunk_dw * 4, index=True),
        )
        a_scale_rsrc = _ptr_buffer_resource(arg_a_scale, a_scale_nbytes)
        b_scale_rsrc = _ptr_buffer_resource(
            arg_b_scale,
            arith.constant(experts * kBS_per_expert_dw * 4, type=T.i32),
        )
        e_idx = arith.index_cast(ir.IndexType.get(), expert_i32)
        _c256 = arith.constant(256, index=True)
        _c64 = arith.constant(64, index=True)
        _c16 = arith.constant(16, index=True)
        _c2048 = arith.constant(2048, index=True)
        _cKH = arith.constant(K_HALF, index=True)
        _cN = arith.constant(N_OUT, index=True)

        # A-load tiling + per-row token gather (m_indices[m_row + row_local]).
        tx_i32_base = tx * arith.constant(chunk_i32, index=True)
        layout_x_tile = fx.make_layout((BM, tile_k_dwords), (tile_k_dwords, 1))
        x_row_local = []
        x_col_local = []
        x_row_base_div4 = []
        for i in range_constexpr(num_x_loads):
            _rl, _cl = tile_chunk_coord_i32(
                arith, tx_i32_base=tx_i32_base, i=i,
                total_threads=total_threads, layout_tile_div4=layout_x_tile,
                chunk_i32=chunk_i32,
            )
            x_row_local.append(_rl)
            x_col_local.append(_cl)
            _tok = buffer_ops.buffer_load(
                m_idx_rsrc, m_row + _rl, vec_width=1, dtype=T.i32
            )
            _tok_idx = arith.index_cast(ir.IndexType.get(), _tok)
            x_row_base_div4.append(_tok_idx * arith.constant(c_k_div4, index=True))

        def load_a_tile(kt):
            base_k_div4 = arith.constant(kt * tile_k_dwords, index=True)
            parts = []
            for i in range_constexpr(num_x_loads):
                idx = x_row_base_div4[i] + base_k_div4 + x_col_local[i]
                v = buffer_copy_gmem16_dwordx4(
                    buffer_ops, vector, elem_type=T.i8, idx_i32=idx,
                    rsrc=a_rsrc, vec_elems=16,
                )
                parts.append(vector.bitcast(T.vec(4, i32), v))
            return parts

        def store_a_tile(parts):
            for i in range_constexpr(num_x_loads):
                lds_store_16b_xor16(
                    arith, vector, lds_memref=lds_a, vec16_ty=vec16_x,
                    layout_lds=layout_lds, row_local=x_row_local[i],
                    col_local_i32=x_col_local[i], tx_c4=arith.index(4),
                    k_blocks16=k_blocks16, lds_base=arith.index(0),
                    vec_part_i32x4=parts[i], elem_bytes=elem_bytes,
                )

        def load_b_tile(kt):
            b_tile = []
            for j in range_constexpr(4):
                n_off = (
                    n_block * _c256 + wave_n * _c64
                    + arith.constant(j * 16, index=True)
                )
                base_bytes = (e_idx * _cN + n_off) * _cKH
                v_voff = (
                    lane_div_16 * _c256 + lane_mod_16 * _c16
                    + arith.constant(kt * 2048, index=True)
                )
                base_dw = (base_bytes + v_voff) / arith.index(4)
                b_j = []
                for sub in range_constexpr(2):
                    off = base_dw + arith.constant(sub * 256, index=True)
                    v16 = buffer_ops.buffer_load(
                        b_rsrc, off, vec_width=4, dtype=i32, cache_modifier=b_nt
                    )
                    i64x2 = vector.bitcast(vec2_i64, v16)
                    b0 = vector.extract(i64x2, static_position=[0], dynamic_position=[])
                    b1 = vector.extract(i64x2, static_position=[1], dynamic_position=[])
                    b_j.append((b0, b1))
                b_tile.append(b_j)
            return b_tile

        _scale_lane_dw = lane_div_16 * _c16 + lane_mod_16
        _chunk_base = m_row / arith.constant(32, index=True)
        _mni_base = (
            n_block * arith.constant(_mni_n0, index=True)
            + wave_n * arith.constant(_mni_wn, index=True)
        )

        def load_a_scale(kt):
            a_base = (
                _chunk_base * arith.constant(kAS_per_chunk_dw, index=True)
                + arith.constant(kt * 64, index=True) + _scale_lane_dw
            )
            return buffer_ops.buffer_load(
                a_scale_rsrc, a_base, vec_width=1, dtype=T.i32, cache_modifier=0
            )

        def load_b_scale(kt):
            b_scs = []
            for mw in range_constexpr(2):
                b_base = (
                    e_idx * arith.constant(kBS_per_expert_dw, index=True)
                    + (_mni_base + arith.constant(mw, index=True))
                    * arith.constant(kBS_stride_n0_dw, index=True)
                    + arith.constant(kt * 64, index=True) + _scale_lane_dw
                )
                b_scs.append(buffer_ops.buffer_load(
                    b_scale_rsrc, b_base, vec_width=1, dtype=T.i32, cache_modifier=0
                ))
            return b_scs

        # MFMA setup
        row_a_lds = lane_mod_16
        col_offset_base = lane_div_16 * _c16
        m_repeat = BM // 16
        num_acc_n = BN // num_waves // 16     # 4
        num_acc_n_packed = num_acc_n // 2     # 2
        pack_N = 2

        def _pack_i64x4(x0, x1):
            c0 = arith.constant(0, type=T.i64)
            v4 = vector.from_elements(vec4_i64, [x0, x1, c0, c0])
            return vector.bitcast(vec8_i32, v4)

        def lds_load_packs_k64(curr_row, col_base):
            col_swz = swizzle_xor16(curr_row, col_base, k_blocks16)
            idx_a = crd2idx([curr_row, col_swz], layout_lds)
            loaded = vector.load_op(vec16_x, lds_a, [idx_a])
            a_i64x2 = vector.bitcast(vec2_i64, loaded)
            a0 = vector.extract(a_i64x2, static_position=[0], dynamic_position=[])
            a1 = vector.extract(a_i64x2, static_position=[1], dynamic_position=[])
            return a0, a1

        acc_init = arith.constant_vector(0.0, vec4_f32)
        accm = [acc_init] * (m_repeat * num_acc_n)

        def mfma_ktile(b_tile, a_sc, b_scs):
            for k_idx in range_constexpr(2):
                ikxdl = k_idx
                col_base = col_offset_base + arith.constant(
                    (k_idx * 128) // a_elem_vec_pack, index=True
                )
                for imxdl in range_constexpr(pack_M):
                    mi_idx = imxdl
                    curr_row = row_a_lds + arith.constant(mi_idx * 16, index=True)
                    a0, a1 = lds_load_packs_k64(curr_row, col_base)
                    a128 = _pack_i64x4(a0, a1)
                    op_sel_a = ikxdl * _scale_pack_m + imxdl
                    for ni in range_constexpr(num_acc_n_packed):
                        b_scale_val = b_scs[ni]
                        for inxdl in range_constexpr(pack_N):
                            ni_idx = ni * pack_N + inxdl
                            b0, b1 = b_tile[ni_idx][k_idx]
                            b128 = _pack_i64x4(b0, b1)
                            op_sel_b = ikxdl * pack_N + inxdl
                            acc_idx = mi_idx * num_acc_n + ni_idx
                            accm[acc_idx] = rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                                vec4_f32,
                                [a128, b128, accm[acc_idx], cbsz, blgp,
                                 op_sel_a, a_sc, op_sel_b, b_scale_val],
                            )

        for kt in range_constexpr(K_TILES):
            if const_expr(kt > 0):
                gpu.barrier()
            store_a_tile(load_a_tile(kt))
            gpu.barrier()
            mfma_ktile(load_b_tile(kt), load_a_scale(kt), load_b_scale(kt))

        # ---- SiLU*mul + fp4 requant epilogue (increment 3) ----
        ascale_out_rsrc = _ptr_buffer_resource(
            arg_ascale_out,
            arith.index_cast(
                T.i32,
                (c_max_sorted / arith.constant(32, index=True)
                 + arith.constant(2, index=True))
                * arith.constant(kOS_per_chunk_dw * 4, index=True),
            ),
        )
        gpu.barrier()
        c_BN = arith.constant(BN, index=True)
        _c128 = arith.constant(128, index=True)
        _c32idx = arith.constant(32, index=True)
        # cshuffle: accm[i][J] -> lds_acc; J even=gate (col), J odd=up (128+col)
        for i in range_constexpr(m_repeat):
            row_base = arith.constant(i * 16, index=True) + lane_div_16 * arith.constant(
                4, index=True
            )
            for J in range_constexpr(num_acc_n):
                is_up = (J % 2) == 1
                J_local = J // 2
                col_local = (
                    wave_n * _c32idx
                    + arith.constant(J_local * 16, index=True)
                    + lane_mod_16
                )
                lds_col = (col_local + _c128) if is_up else col_local
                acc_v = accm[i * num_acc_n + J]
                for v in range_constexpr(4):
                    lds_idx = (row_base + arith.constant(v, index=True)) * c_BN + lds_col
                    val = vector.extract(acc_v, static_position=[v], dynamic_position=[])
                    vv = vector.from_elements(T.vec(1, f32), [val])
                    vector.store(vv, lds_acc, [lds_idx], alignment=4)
        gpu.barrier()

        # read gate/up, SiLU*mul, per-32 amax(DPP) -> fp4 + e8m0
        m_lane = tx / _c16
        n_lane = tx % _c16
        wave_grp = n_lane / arith.index(4)
        kk = n_lane % arith.index(4)
        M_REPS = BM // 16
        _neg_log2e = arith.constant(-1.4426950408889634, type=f32)
        _one_f = arith.constant(1.0, type=f32)
        _c7fff = arith.constant(0x7FFFFFFF, type=i32)
        _c2e21 = arith.constant(0x200000, type=i32)
        _quarter = arith.constant(0.25, type=f32)
        _c254i = arith.constant(254, type=i32)
        K_G2_HALF = N_INTER // 2
        scales_per_mr = []
        for mr in range_constexpr(M_REPS):
            row_local = arith.constant(mr * 16, index=True) + m_lane
            gate_col0 = wave_grp * _c32idx + kk * arith.index(8)
            results = []
            for e in range_constexpr(8):
                gcol = gate_col0 + arith.constant(e, index=True)
                g = vector.extract(
                    vector.load_op(T.vec(1, f32), lds_acc, [row_local * c_BN + gcol]),
                    static_position=[0], dynamic_position=[],
                )
                u = vector.extract(
                    vector.load_op(
                        T.vec(1, f32), lds_acc, [row_local * c_BN + gcol + _c128]
                    ),
                    static_position=[0], dynamic_position=[],
                )
                t = g * _neg_log2e
                emu = llvm.call_intrinsic(f32, "llvm.amdgcn.exp2.f32", [t], [], [])
                sig = llvm.call_intrinsic(
                    f32, "llvm.amdgcn.rcp.f32", [_one_f + emu], [], []
                )
                results.append((g * sig) * u)
            # amax over 8 + quad (kk) cross-lane reduction
            local_max = (results[0].bitcast(i32) & _c7fff).bitcast(f32)
            for e in range_constexpr(1, 8):
                local_max = arith.maximumf(
                    local_max, (results[e].bitcast(i32) & _c7fff).bitcast(f32)
                )
            local_max = arith.maximumf(local_max, local_max.shuffle_xor(1, 64))
            local_max = arith.maximumf(local_max, local_max.shuffle_xor(2, 64))
            quant_scale = ((local_max.bitcast(i32) + _c2e21).bitcast(f32)) * _quarter
            sb_raw = quant_scale.bitcast(i32) >> arith.constant(23, type=i32)
            scales_per_mr.append(arith.minsi(sb_raw, _c254i))
            packed = None
            for e in range_constexpr(8):
                fp4 = _f32_to_e2m1(arith.divf(results[e], quant_scale))
                if const_expr(e % 2 == 0):
                    byte_b = fp4
                else:
                    byte_b = byte_b | (fp4 << arith.constant(4, type=i32))
                    sh = byte_b << arith.constant((e // 2) * 8, type=i32)
                    packed = sh if packed is None else (packed | sh)
            byte_pos = (
                n_block * arith.constant(64, index=True)
                + wave_grp * _c16 + kk * arith.index(4)
            )
            out_row = m_row + row_local
            aq_off = out_row * arith.constant(K_G2_HALF, index=True) + byte_pos
            buffer_ops.buffer_store(
                packed, aqout_rsrc, aq_off, offset_is_bytes=True
            )

        # scale store (kk==0): make_preshuffle inter scale
        kk_i32 = arith.index_cast(i32, kk)
        is_w = arith.cmpi(CmpIPredicate.eq, kk_i32, arith.constant(0, type=i32))
        _ifw = scf.IfOp(is_w)
        with ir.InsertionPoint(_ifw.then_block):
            ku = n_block / arith.index(2)
            ikxdl = n_block % arith.index(2)
            ikxdl_b = arith.index_cast(i32, ikxdl)
            dword_off = (
                m_block * arith.constant(kOS_per_chunk_dw, index=True)
                + ku * _c64 + wave_grp * _c16 + m_lane
            )
            sc_byte = dword_off * arith.index(4) + arith.index_cast(
                ir.IndexType.get(), ikxdl_b * arith.constant(2, type=i32)
            )
            if const_expr(BM == 16):
                s0 = arith.TruncIOp(T.i8, scales_per_mr[0])
                buffer_ops.buffer_store(
                    s0, ascale_out_rsrc, sc_byte, offset_is_bytes=True
                )
            else:
                pair = arith.TruncIOp(
                    T.i16,
                    scales_per_mr[0]
                    | (scales_per_mr[1] << arith.constant(8, type=i32)),
                )
                buffer_ops.buffer_store(
                    pair, ascale_out_rsrc, sc_byte, offset_is_bytes=True
                )
            scf.YieldOp([])

    @flyc.jit
    def launch_gemm1(
        arg_a: fx.Pointer,
        arg_a_scale: fx.Pointer,
        arg_b: fx.Pointer,
        arg_b_scale: fx.Pointer,
        arg_sorted_expert_ids: fx.Pointer,
        arg_cumsum: fx.Pointer,
        arg_m_indices: fx.Pointer,
        arg_aq_out: fx.Pointer,
        arg_ascale_out: fx.Pointer,
        arg_hidden: fx.Pointer,
        i32_tokens: fx.Int32,
        i32_max_sorted: fx.Int32,
        i32_total_m_blocks: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()
        gx = arith.index_cast(ir.IndexType.get(), i32_total_m_blocks.ir_value())
        gemm1_kernel(
            arg_a, arg_a_scale, arg_b, arg_b_scale,
            arg_sorted_expert_ids, arg_cumsum, arg_m_indices,
            arg_aq_out, arg_ascale_out, arg_hidden,
            i32_tokens, i32_max_sorted,
        ).launch(
            grid=(gx, num_n_blocks, 1),
            block=(total_threads, 1, 1),
            smem=65536,
            stream=stream,
        )

    return launch_gemm1


def get_arch():
    from flydsl.runtime.device import get_rocm_arch
    return get_rocm_arch()
