"""A16W4 (BF16 activation, MXFP4 weight) MoE GEMM stage1/stage2 kernels.

Uses ``mfma_f32_16x16x32_bf16`` with BF16 activations and FP4 E2M1 weights
dequantized to BF16 in-register.  E8M0 block scales (one per 32 K elements)
are applied during dequantization.

Host-side prerequisites:
  - Weights: ``shuffle_weight_a16w4(w, NLane=16, gate_up=...)``
  - Scales:  ``shuffle_scale_a16w4(scale, E, gate_up=...)``
"""

import logging
import os
import functools
from contextlib import contextmanager

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith
from flydsl.expr import gpu, buffer_ops, vector, rocdl
from flydsl.expr import range_constexpr
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr

try:
    from flydsl.runtime.device import (
        supports_bf16_global_atomics,
        bf16_global_atomics_arch_description,
    )
except ImportError:
    def supports_bf16_global_atomics(arch: str) -> bool:
        return str(arch).startswith(("gfx94", "gfx95", "gfx12"))

    def bf16_global_atomics_arch_description() -> str:
        return "gfx94+/gfx95+/gfx12+"

from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm, scf, memref
from flydsl._mlir.dialects import fly as _fly
from flydsl.expr.typing import T

from .mfma_preshuffle_pipeline import (
    _buffer_load_vec,
    buffer_copy_gmem16_dwordx4,
    lds_store_16b_xor16,
    lds_store_8b_xor16,
    make_preshuffle_b_layout,
    make_preshuffle_scale_layout,
    load_b_raw_mxfp4,
    load_b_raw_mxfp4_dwordx4,
    unpack_b_mxfp4_bf16,
    tile_chunk_coord_i32,
    swizzle_xor16,
)
from .mfma_epilogues import c_shuffle_epilog, default_epilog, mfma_epilog
from .layout_utils import crd2idx, idx2crd, get as layout_get


@contextmanager
def _if_then(if_op):
    with ir.InsertionPoint(if_op.then_block):
        try:
            yield if_op.then_block
        finally:
            blk = if_op.then_block
            if (not blk.operations) or not isinstance(blk.operations[-1], scf.YieldOp):
                scf.YieldOp([])


@contextmanager
def _if_else(if_op):
    if getattr(if_op, "else_block", None) is None:
        raise RuntimeError("IfOp has no else block")
    with ir.InsertionPoint(if_op.else_block):
        try:
            yield if_op.else_block
        finally:
            blk = if_op.else_block
            if (not blk.operations) or not isinstance(blk.operations[-1], scf.YieldOp):
                scf.YieldOp([])


def _decode_e8m0_byte_to_f32(byte_i8, arith_mod):
    """Convert a single E8M0 byte (i8) to f32 = 2^(e - 127)."""
    c23 = arith_mod.constant(23, type=T.i32)
    byte_u32 = arith_mod.extui(T.i32, byte_i8)
    scale_bits = arith_mod.shli(byte_u32, c23)
    return arith_mod.bitcast(T.f32, scale_bits)


def _barrier(vmcnt=63, lgkmcnt=63):
    """Emit s_waitcnt + s_barrier via inline asm.

    Bypasses LLVM SIInsertWaitcnts which would insert a conservative
    s_waitcnt vmcnt(0) lgkmcnt(0) before every S_BARRIER MI.
    """
    parts = []
    needs_waitcnt = vmcnt < 63 or lgkmcnt < 63
    parts.append("s_barrier")
    if needs_waitcnt:
        wc = []
        if vmcnt < 63:
            wc.append(f"vmcnt({vmcnt})")
        if lgkmcnt < 63:
            wc.append(f"lgkmcnt({lgkmcnt})")
        parts.append("s_waitcnt " + " ".join(wc))
    llvm.InlineAsmOp(
        res=None,
        operands_=[],
        asm_string="\n".join(parts),
        constraints="",
        has_side_effects=True,
        is_align_stack=False,
    )

# ---------------------------------------------------------------------------
# Stage 2: Down-projection GEMM (MXFP4 weights, BF16 activations)
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=1024)
def compile_a16w4_moe_gemm2(
    *,
    model_dim: int,
    inter_dim: int,
    experts: int,
    topk: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    doweight_stage2: bool,
    out_dtype: str = "bf16",
    use_cshuffle_epilog: bool | None = None,
    enable_bias: bool = False,
    model_dim_pad: int = 0,
    inter_dim_pad: int = 0,
    accumulate: bool = True,
    waves_per_eu: int = 0,
    k_batch: int = 1,
):
    """Compile stage2 A16W4 MXFP4 kernel and return the compiled executable.

    A2 is bf16.  W is MXFP4 (FP4 E2M1) with E8M0 block scales, pre-shuffled
    by ``shuffle_weight_a16w4`` and ``shuffle_scale_a16w4``.

    enable_bias: add per-column f32 bias after GEMM accumulation.
    model_dim_pad / inter_dim_pad: padding semantics (see stage1).
    """
    gpu_arch = get_hip_arch()
    allocator_pong = SmemAllocator(None, arch=gpu_arch, global_sym_name="smem0")
    allocator_ping = SmemAllocator(None, arch=gpu_arch, global_sym_name="smem1")

    elem_bytes = 2  # bf16 activations
    kpack_bytes = 16  # MXFP4 preshuffle

    _mfma_k32_raw = getattr(rocdl, "mfma_f32_16x16x32_bf16_", None)
    if _mfma_k32_raw is None:
        raise AttributeError(
            "BF16 K32 MFMA op not found: expected `rocdl.mfma_f32_16x16x32_bf16_`"
        )
    _split_mfma = rocdl._split_mfma_operands

    def mfma_f32_bf16_k32(result_type, operands, *, loc=None, ip=None):
        a, b, c, cbsz, abid, blgp = _split_mfma(operands, loc=loc)
        return _mfma_k32_raw(result_type, a, b, c, cbsz, abid, blgp, loc=loc, ip=ip)

    tile_k_bytes = int(tile_k) * int(elem_bytes)
    if (tile_k_bytes % 64) != 0:
        raise ValueError(
            f"tile_k_bytes must be divisible by 64, got tile_k_bytes={tile_k_bytes}"
        )

    out_s = str(out_dtype).strip().lower()
    out_is_f32 = out_s in ("f32", "fp32", "float")
    out_is_bf16 = out_s in ("bf16", "bfloat16")
    if (not bool(accumulate)) and out_is_f32:
        raise ValueError(
            "compile_a16w4_moe_gemm2(accumulate=False) only supports out_dtype in {'f16','bf16'}"
        )

    DYN = ir.ShapedType.get_dynamic_size()
    # FP4: 2 nibbles per byte → half the byte count.
    size_w = (experts * model_dim * inter_dim) // 2

    total_threads = 256
    bytes_x_per_tile = int(tile_m) * int(tile_k) * int(elem_bytes)
    if bytes_x_per_tile % total_threads != 0:
        raise ValueError(
            "tile_m*tile_k*elem_bytes must be divisible by "
            f"{total_threads}: tile_m={tile_m}, tile_k={tile_k}"
        )
    bytes_per_thread_x = bytes_x_per_tile // total_threads

    _ck_lds128 = os.environ.get("FLIR_CK_LDS128", "1") in (
        "1", "true", "True", "YES", "yes",
    )
    pad_k = 0 if _ck_lds128 else 8
    lds_stride = tile_k + pad_k

    if out_is_bf16:
        if not supports_bf16_global_atomics(gpu_arch):
            raise ValueError(
                f"out_dtype='bf16' requires bf16 global atomics, got arch={gpu_arch!r}"
            )

    if out_is_f32:
        _use_cshuffle_epilog = (
            False if use_cshuffle_epilog is None else bool(use_cshuffle_epilog)
        )
        if _use_cshuffle_epilog:
            raise ValueError("out_dtype='f32' does not support CShuffle epilogue.")
    else:
        if use_cshuffle_epilog is None:
            _use_cshuffle_epilog = os.environ.get("FLIR_MOE_STAGE2_CSHUFFLE", "1") in (
                "1", "true", "True", "YES", "yes",
            )
        else:
            _use_cshuffle_epilog = bool(use_cshuffle_epilog)
        if not _use_cshuffle_epilog:
            raise ValueError(
                "stage2 f16 output currently requires CShuffle epilogue."
            )

    def out_elem():
        ty = T.f32 if out_is_f32 else (T.bf16 if out_is_bf16 else T.f16)
        return ty() if callable(ty) else ty

    _cshuffle_nlane = 32
    if bool(accumulate):
        _e_vec = 2
    else:
        _e_vec = 8 if int(tile_n) % (_cshuffle_nlane * 8) == 0 else 2
        _cshuffle_stride = _cshuffle_nlane * _e_vec
        if int(tile_n) % _cshuffle_stride != 0:
            raise ValueError(
                f"tile_n={tile_n} must be divisible by {_cshuffle_stride} when accumulate=False"
            )

    _single_x_bytes = int(tile_m) * int(lds_stride) * int(elem_bytes)
    _single_x_elems = _single_x_bytes // int(elem_bytes)
    lds_out_bytes = (
        2 * int(tile_m) * int(tile_n) if _use_cshuffle_epilog else 0
    )
    _pong_buffer_bytes = max(_single_x_bytes, lds_out_bytes)
    _ping_buffer_bytes = _single_x_bytes

    lds_pong_offset = allocator_pong._align(allocator_pong.ptr, 16)
    allocator_pong.ptr = lds_pong_offset + _pong_buffer_bytes

    _sorted_info_elems = int(tile_m)
    _sorted_info_bytes = _sorted_info_elems * 4
    lds_sorted_info_offset = allocator_pong._align(allocator_pong.ptr, 4)
    allocator_pong.ptr = lds_sorted_info_offset + _sorted_info_bytes

    lds_ping_offset = allocator_ping._align(allocator_ping.ptr, 16)
    allocator_ping.ptr = lds_ping_offset + _ping_buffer_bytes

    if waves_per_eu >= 1:
        _total_cu_lds = 160 * 1024
        _min_lds = _total_cu_lds // (waves_per_eu + 1) + 1
        _pong_sz = allocator_pong._align(allocator_pong.ptr, 128)
        _ping_sz = allocator_ping._align(allocator_ping.ptr, 128)
        _cur_lds = _pong_sz + _ping_sz
        if _cur_lds < _min_lds:
            allocator_ping.ptr += _min_lds - _cur_lds

    _k_batch = int(k_batch)
    if _k_batch > 1:
        if inter_dim % (_k_batch * tile_k) != 0:
            raise ValueError(
                f"inter_dim={inter_dim} must be divisible by k_batch*tile_k="
                f"{_k_batch * tile_k}"
            )
    _k_dim = inter_dim // _k_batch
    _total_tiles_check = _k_dim // tile_k
    if _total_tiles_check < 2 or _total_tiles_check % 2 != 0:
        raise ValueError(
            f"k_batch={_k_batch}: _k_dim/tile_k={_total_tiles_check} must be "
            f"even and >= 2 for the ping-pong pipeline"
        )

    _wpe_tag = f"_wpe{waves_per_eu}" if waves_per_eu >= 1 else ""
    _kb_tag = f"_kb{_k_batch}" if _k_batch > 1 else ""
    _bias_tag = "_bias" if enable_bias else ""
    _pad_tag = f"_mp{model_dim_pad}_ip{inter_dim_pad}" if (model_dim_pad or inter_dim_pad) else ""
    module_name = (
        f"mfma_a16w4_moe2_mxfp4_{out_s}"
        f"_t{tile_m}x{tile_n}x{tile_k}"
        f"{_wpe_tag}{_kb_tag}{_bias_tag}{_pad_tag}_abi1"
    ).replace("-", "_")

    if True:

        @flyc.kernel
        def moe_gemm2(
            arg_out: fx.Tensor,
            arg_x: fx.Tensor,
            arg_w: fx.Tensor,
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
            tokens_in = arith.index_cast(T.index, i32_tokens_in.ir_value())
            n_in = arith.ArithValue(arith.index_cast(T.index, i32_n_in.ir_value()))
            k_in = arith.index_cast(T.index, i32_k_in.ir_value())
            size_expert_ids_in = arith.index_cast(
                T.index, i32_size_expert_ids_in.ir_value()
            )
            k_i32_v = i32_k_in.ir_value()

            x_elem = T.bf16
            w_elem = T.i8  # packed FP4 stored as bytes
            f16 = T.f16
            f32 = T.f32
            i32 = T.i32
            i64 = T.i64
            vec4_f32 = T.vec(4, f32)
            vec1_i32 = T.vec(1, i32)
            vec4_i16 = T.vec(4, T.i16)
            vec8_bf16 = T.vec(8, x_elem)
            vec16_bf16 = T.vec(16, x_elem) if elem_bytes == 2 else T.vec(8, x_elem)
            vec1_i64 = T.vec(1, i64)
            vec2_i64 = T.vec(2, i64)

            acc_init = arith.constant_vector(0.0, vec4_f32)

            # A2 layout
            topk_idx = arith.index(topk)
            m_in = tokens_in * topk_idx
            m_i32_v = arith.index_cast(i32, m_in)
            layout_x = fx.make_layout((m_i32_v, k_i32_v), stride=(k_i32_v, 1))

            # B preshuffle layout: MXFP4 kpack=16, c_k = k_in // 2
            c_n_total = arith.index(experts * model_dim)
            c2 = arith.index(2)
            c_k_packed = k_in // c2  # FP4: 2 nibbles per byte
            b_layout = make_preshuffle_b_layout(
                arith,
                c_n=c_n_total,
                c_k=c_k_packed,
                kpack_bytes=kpack_bytes,
                elem_bytes=1,
            )
            layout_b = b_layout.layout_b

            # Scale layout
            layout_b_scale = make_preshuffle_scale_layout(
                arith,
                c_mn=c_n_total,
                c_k=k_in,
                mn_pack=2,
                k_pack=2,
                elem_bytes=4,
                scale_block_size=32,
            )

            shape_lds = fx.make_shape(tile_m, tile_k)
            stride_lds = fx.make_stride(lds_stride, 1)
            layout_lds = fx.make_layout(shape_lds, stride_lds)

            tx = gpu.thread_id("x")
            by = gpu.block_id("x")  # tile along model_dim
            bx = gpu.block_id("y")  # tile along sorted M

            k_blocks16 = arith.index(tile_k_bytes // 16)
            layout_tx_wave_lane = fx.make_layout((4, 64), stride=(64, 1))
            layout_lane16 = fx.make_layout((4, 16), stride=(16, 1))

            base_ptr_pong = allocator_pong.get_base()
            base_ptr_ping = allocator_ping.get_base()
            lds_x_pong = SmemPtr(
                base_ptr_pong, lds_pong_offset, T.bf16,
                shape=(_single_x_elems,),
            ).get()
            lds_x_ping = SmemPtr(
                base_ptr_ping, lds_ping_offset, T.bf16,
                shape=(_single_x_elems,),
            ).get()
            lds_out = (
                SmemPtr(
                    base_ptr_pong, lds_pong_offset,
                    (T.bf16 if out_is_bf16 else T.f16),
                    shape=(tile_m * tile_n,),
                ).get()
                if _use_cshuffle_epilog
                else None
            )
            lds_sorted_cache = SmemPtr(
                base_ptr_pong, lds_sorted_info_offset, T.i32,
                shape=(_sorted_info_elems,),
            ).get()

            c_topk = arith.index(topk)

            x_nbytes_idx = (tokens_in * c_topk) * k_in * arith.index(int(elem_bytes))
            x_rsrc = buffer_ops.create_buffer_resource(
                arg_x, max_size=False, num_records_bytes=x_nbytes_idx
            )
            w_rsrc = buffer_ops.create_buffer_resource(arg_w, max_size=False)
            sw_rsrc = buffer_ops.create_buffer_resource(arg_scale_w, max_size=False)

            out_elem_bytes = 4 if out_is_f32 else 2
            out_nbytes_idx = tokens_in * n_in * arith.index(out_elem_bytes)
            if not bool(accumulate):
                out_nbytes_idx = (
                    tokens_in * arith.index(topk) * n_in * arith.index(out_elem_bytes)
                )
            out_rsrc = buffer_ops.create_buffer_resource(
                arg_out, max_size=False, num_records_bytes=out_nbytes_idx
            )

            sorted_nbytes_idx = (
                size_expert_ids_in * arith.index(tile_m) * arith.index(4)
            )
            sorted_rsrc = buffer_ops.create_buffer_resource(
                arg_sorted_token_ids,
                max_size=False,
                num_records_bytes=sorted_nbytes_idx,
            )
            sorted_w_rsrc = buffer_ops.create_buffer_resource(
                arg_sorted_weights, max_size=False, num_records_bytes=sorted_nbytes_idx
            )

            eid_nbytes_idx = size_expert_ids_in * arith.index(4)
            expert_rsrc = buffer_ops.create_buffer_resource(
                arg_expert_ids, max_size=False, num_records_bytes=eid_nbytes_idx
            )
            bx_m = bx * arith.index(tile_m)

            numids_rsrc = buffer_ops.create_buffer_resource(
                arg_num_valid_ids, max_size=False, num_records_bytes=arith.index(4),
            )
            bias_rsrc = (
                buffer_ops.create_buffer_resource(arg_bias, max_size=False)
                if enable_bias
                else None
            )
            num_valid_i32 = buffer_ops.buffer_load(
                numids_rsrc, arith.index(0), vec_width=1, dtype=i32
            )
            bx_m_i32 = arith.index_cast(i32, bx_m)
            blk_valid = arith.cmpi(arith.CmpIPredicate.ult, bx_m_i32, num_valid_i32)

            def _moe_gemm2_then_body():
                expert_i32 = buffer_ops.buffer_load(
                    expert_rsrc, bx, vec_width=1, dtype=i32
                )
                expert_idx = arith.index_cast(T.index, expert_i32)
                n_idx = arith.index(model_dim)
                expert_off_idx = expert_idx * n_idx

                if bytes_per_thread_x >= 16 and bytes_per_thread_x % 16 == 0:
                    x_load_bytes = 16
                elif bytes_per_thread_x >= 8 and bytes_per_thread_x % 8 == 0:
                    x_load_bytes = 8
                elif bytes_per_thread_x >= 4 and bytes_per_thread_x % 4 == 0:
                    x_load_bytes = 4
                else:
                    raise ValueError(
                        f"bytes_per_thread_x ({bytes_per_thread_x}) must be "
                        f"divisible by 4"
                    )
                num_x_loads = bytes_per_thread_x // x_load_bytes
                chunk_i32 = x_load_bytes // 4
                x_vec_elems = x_load_bytes // elem_bytes
                x_vec_i32_ty = T.vec(chunk_i32, i32) if chunk_i32 > 1 else T.vec(1, i32)
                x_vec_x_ty = T.vec(x_vec_elems, x_elem)
                vec16_x = T.vec(8, x_elem)

                c_k_div4 = (k_in * arith.index(int(elem_bytes))) // arith.index(4)
                c_k_div4_i32 = arith.index_cast(i32, c_k_div4)
                layout_x_div4 = fx.make_layout(
                    (m_i32_v, c_k_div4_i32), stride=(c_k_div4_i32, 1)
                )
                tile_k_dwords = (int(tile_k) * int(elem_bytes)) // 4
                layout_x_tile_div4 = fx.make_layout(
                    (tile_m, tile_k_dwords), stride=(tile_k_dwords, 1)
                )
                c_chunk_i32 = arith.index(chunk_i32)
                tx_i32_base = tx * c_chunk_i32

                topk_i32 = arith.constant(topk, type=T.i32)
                mask24 = arith.constant(0xFFFFFF, type=T.i32)
                tokens_i32 = arith.index_cast(i32, tokens_in)

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
                x_row_local = []
                for i in range_constexpr(num_x_loads):
                    row_local, col_local_i32 = x_tile_chunk_coord_i32(i)
                    x_row_local.append(row_local)
                    x_col_local_i32.append(col_local_i32)

                    sorted_row_i = bx_m + row_local
                    fused_i = buffer_ops.buffer_load(
                        sorted_rsrc, sorted_row_i, vec_width=1, dtype=i32
                    )
                    _fused_v1 = vector.from_elements(vec1_i32, [fused_i])
                    vector.store(_fused_v1, lds_sorted_cache, [row_local])
                    t_i32 = fused_i & mask24
                    s_i32 = arith.shrui(fused_i, arith.constant(24, type=T.i32))
                    t_valid = arith.cmpi(arith.CmpIPredicate.ult, t_i32, tokens_i32)
                    s_valid = arith.cmpi(arith.CmpIPredicate.ult, s_i32, topk_i32)
                    ts_valid = t_valid & s_valid
                    t_safe = arith.select(
                        ts_valid, t_i32, arith.constant(0, type=T.i32)
                    )
                    s_safe = arith.select(
                        ts_valid, s_i32, arith.constant(0, type=T.i32)
                    )
                    row_ts_i32 = t_safe * topk_i32 + s_safe
                    row_ts_idx = arith.index_cast(T.index, row_ts_i32)
                    x_row_base_div4.append(row_ts_idx * c_k_div4)

                def load_x_tile(base_k):
                    base_k_div4 = (
                        base_k * arith.index(int(elem_bytes))
                    ) // arith.index(4)
                    parts = []
                    for i in range_constexpr(num_x_loads):
                        idx_i32 = x_row_base_div4[i] + base_k_div4 + x_col_local_i32[i]
                        idx_elem = idx_i32 * arith.index(2)
                        x_vec = _buffer_load_vec(
                            buffer_ops, vector, x_rsrc, idx_elem,
                            elem_type=x_elem,
                            vec_elems=x_vec_elems,
                            elem_bytes=elem_bytes,
                            offset_in_bytes=False,
                        )
                        parts.append(vector.bitcast(x_vec_i32_ty, x_vec))
                    return parts

                coord_wl = idx2crd(tx, layout_tx_wave_lane)
                wave_id = layout_get(coord_wl, 0)
                lane_id = layout_get(coord_wl, 1)
                coord_l16 = idx2crd(lane_id, layout_lane16)
                lane_div_16 = layout_get(coord_l16, 0)
                lane_mod_16 = layout_get(coord_l16, 1)

                _dma_bytes = 16
                _wave_size = 64

                def dma_x_tile_to_lds(base_k, lds_buffer):
                    """Async DMA: global -> LDS via buffer_load_lds, no VGPR."""
                    c4_idx = arith.index(4)
                    base_k_div4 = (
                        base_k * arith.index(int(elem_bytes))
                    ) // arith.index(4)

                    lds_ptr_i64 = None
                    for i in range_constexpr(num_x_loads):
                        row_local_i = x_row_local[i]
                        col_local_i32_i = x_col_local_i32[i]
                        col_local_sw = swizzle_xor16(
                            row_local_i, col_local_i32_i * c4_idx, k_blocks16
                        )
                        row_k_dw = x_row_base_div4[i] + base_k_div4
                        global_byte_idx = row_k_dw * c4_idx + col_local_sw
                        global_offset = arith.index_cast(i32, global_byte_idx)

                        if i == 0:
                            lds_addr = memref.extract_aligned_pointer_as_index(
                                lds_buffer
                            ) + wave_id * arith.constant(
                                _wave_size * _dma_bytes, index=True
                            )
                            lds_ptr_i64 = rocdl.readfirstlane(
                                i64, arith.index_cast(i64, lds_addr)
                            )
                        else:
                            lds_ptr_i64 = lds_ptr_i64 + arith.constant(
                                total_threads * _dma_bytes, type=i64
                            )

                        lds_ptr_type = ir.Type.parse("!llvm.ptr<3>")
                        lds_ptr = llvm.inttoptr(lds_ptr_type, lds_ptr_i64)

                        rocdl.raw_ptr_buffer_load_lds(
                            x_rsrc,
                            lds_ptr,
                            arith.constant(_dma_bytes, type=i32),
                            global_offset,
                            arith.constant(0, type=i32),
                            arith.constant(0, type=i32),
                            arith.constant(0, type=i32),
                        )

                def prefetch_x_to_lds(base_k, lds_buffer):
                    dma_x_tile_to_lds(base_k, lds_buffer)

                row_a_lds = lane_mod_16
                _a_sublane_stride = 64   # 32 bf16 * 2 bytes
                _a_ku_stride_bytes = 16  # 8 bf16 * 2 bytes
                col_offset_base_bytes = lane_div_16 * arith.index(_a_sublane_stride)

                by_n = by * arith.index(tile_n)
                num_waves = 4
                n_per_wave = tile_n // num_waves
                num_acc_n = n_per_wave // 16
                c_n_per_wave = arith.index(n_per_wave)
                wave_mod_4 = wave_id % arith.index(4)
                n_tile_base = wave_mod_4 * c_n_per_wave

                n_intra_list = []
                n_blk_list = []
                col_g_list = []
                c_n0_static = experts * model_dim // 16
                layout_n_blk_intra = fx.make_layout((c_n0_static, 16), stride=(16, 1))
                for ni in range_constexpr(num_acc_n):
                    offset = arith.index(ni * 16)
                    col_g = by_n + n_tile_base + offset + lane_mod_16
                    col_g_list.append(col_g)

                    row_w = expert_off_idx + col_g
                    coord_w = idx2crd(row_w, layout_n_blk_intra)
                    n_blk_list.append(layout_get(coord_w, 0))
                    n_intra_list.append(layout_get(coord_w, 1))

                m_repeat = tile_m // 16
                k_unroll = tile_k_bytes // 64

                _pad_k_elems = (inter_dim_pad % tile_k) if (_k_batch == 1 and inter_dim_pad > 0) else 0
                _pad_ku_skip = _pad_k_elems // 32
                _tail_ku = k_unroll - _pad_ku_skip
                _tail_k0_count = (_tail_ku + 3) // 4 if _pad_ku_skip > 0 else None

                # ---- Scale index helpers ----
                # mni for each ni: (expert_off + by_n + n_tile_base + ni*16) // 32
                scale_mni_list = []
                scale_n_pack_list = []
                for ni in range_constexpr(num_acc_n):
                    n_global = expert_off_idx + by_n + n_tile_base + arith.index(ni * 16)
                    scale_mni_list.append(n_global // arith.index(32))
                    n_block_16 = n_global // arith.index(16)
                    scale_n_pack_list.append(n_block_16 % arith.index(2))

                def _load_scale_i32(scale_ku_idx, ni, scale_klane=None):
                    """Load one packed i32 from the scale buffer."""
                    _klane = scale_klane if scale_klane is not None else lane_div_16
                    idx = (scale_mni_list[ni] * layout_b_scale.stride_n0
                           + scale_ku_idx * layout_b_scale.stride_k0
                           + _klane * layout_b_scale.stride_klane
                           + lane_mod_16)
                    return buffer_ops.buffer_load(
                        sw_rsrc, idx, vec_width=1, dtype=i32
                    )

                def _extract_e8m0_f32_dynamic(packed_i32, byte_pos_idx):
                    """Extract E8M0 byte at runtime byte_pos and decode to f32."""
                    shift = arith.index_cast(i32, byte_pos_idx) * arith.constant(8, type=i32)
                    byte_i32 = arith.shrui(packed_i32, shift) & arith.constant(0xFF, type=i32)
                    scale_bits = arith.shli(byte_i32, arith.constant(23, type=i32))
                    return arith.bitcast(f32, scale_bits)

                # ---- B Load (dwordx4) + Scale for MXFP4 ----
                def _get_scale_f32(base_k, ku, ni, scale_cache):
                    """CK addressing for scale: adj_ku = base_k//32 + (ku//4)*4 + lane_div_16."""
                    _k0_blk = ku // 4
                    adj_ku = (base_k // arith.index(32)
                              + arith.index(_k0_blk * 4)
                              + lane_div_16)
                    scale_klane_rt = lane_div_16
                    k_pack_sub_rt = (adj_ku // arith.index(4)) % arith.index(2)
                    s_ku = adj_ku // arith.index(8)

                    cache_key = (_k0_blk, ni)
                    if cache_key not in scale_cache:
                        scale_cache[cache_key] = _load_scale_i32(
                            s_ku, ni, scale_klane=scale_klane_rt
                        )
                    packed = scale_cache[cache_key]
                    n_pack_sub_val = scale_n_pack_list[ni]
                    byte_pos_even = k_pack_sub_rt * arith.index(2)
                    byte_pos_odd = byte_pos_even + arith.index(1)
                    scale_even = _extract_e8m0_f32_dynamic(packed, byte_pos_even)
                    scale_odd = _extract_e8m0_f32_dynamic(packed, byte_pos_odd)
                    n_pack_is_zero = arith.cmpi(
                        arith.CmpIPredicate.eq,
                        arith.index_cast(i32, n_pack_sub_val),
                        arith.constant(0, type=i32),
                    )
                    return arith.select(n_pack_is_zero, scale_even, scale_odd)

                _k_per_dwordx4 = 128
                _k0_count = tile_k // _k_per_dwordx4

                def load_b_raw(base_k, k0_limit=_k0_count):
                    """Load raw FP4 data via dwordx4. Returns raw_v4[k0_idx][ni]."""
                    raw_all = []
                    for k0_idx in range_constexpr(k0_limit):
                        raw_k0 = []
                        k_off = base_k + arith.index(k0_idx * _k_per_dwordx4)
                        for ni in range_constexpr(num_acc_n):
                            v4 = load_b_raw_mxfp4_dwordx4(
                                buffer_ops, arith, vector,
                                arg_b=arg_w,
                                b_rsrc=w_rsrc,
                                layout_b=layout_b,
                                base_k=k_off,
                                n_blk=n_blk_list[ni],
                                n_intra=n_intra_list[ni],
                                lane_div_16=lane_div_16,
                                elem_type=w_elem,
                                kpack_bytes=kpack_bytes,
                                cache_modifier=2,
                            )
                            raw_k0.append(v4)
                        raw_all.append(raw_k0)
                    return raw_all

                def load_b_scale_raw(base_k, k0_limit=_k0_count):
                    """Issue scale buffer_loads only (no extraction).
                    Returns (packed_dict, kps_dict):
                      packed_dict: {(k0_blk, ni): packed_i32}
                      kps_dict: {k0_blk: k_pack_sub_rt}
                    """
                    packed_dict = {}
                    kps_dict = {}
                    for k0_blk in range_constexpr(k0_limit):
                        adj_ku = (base_k // arith.index(32)
                                  + arith.index(k0_blk * 4)
                                  + lane_div_16)
                        scale_klane_rt = lane_div_16
                        kps_dict[k0_blk] = (adj_ku // arith.index(4)) % arith.index(2)
                        s_ku = adj_ku // arith.index(8)
                        for ni in range_constexpr(num_acc_n):
                            packed_dict[(k0_blk, ni)] = _load_scale_i32(
                                s_ku, ni, scale_klane=scale_klane_rt
                            )
                    return packed_dict, kps_dict

                def extract_b_scales(packed_dict, kps_dict, ku_limit=k_unroll):
                    """Extract f32 scales from pre-loaded packed i32.
                    Returns scales[ku][ni] = f32.
                    """
                    scales = []
                    for ku in range_constexpr(ku_limit):
                        scales_ku = []
                        _k0_blk = ku // 4
                        k_pack_sub_rt = kps_dict[_k0_blk]
                        for ni in range_constexpr(num_acc_n):
                            packed = packed_dict[(_k0_blk, ni)]
                            n_pack_sub_val = scale_n_pack_list[ni]
                            byte_pos_even = k_pack_sub_rt * arith.index(2)
                            byte_pos_odd = byte_pos_even + arith.index(1)
                            scale_even = _extract_e8m0_f32_dynamic(packed, byte_pos_even)
                            scale_odd = _extract_e8m0_f32_dynamic(packed, byte_pos_odd)
                            n_pack_is_zero = arith.cmpi(
                                arith.CmpIPredicate.eq,
                                arith.index_cast(i32, n_pack_sub_val),
                                arith.constant(0, type=i32),
                            )
                            sf = arith.select(n_pack_is_zero, scale_even, scale_odd)
                            scales_ku.append(sf)
                        scales.append(scales_ku)
                    return scales

                def lds_load_packs_k64(curr_row_a_lds, col_base_bytes, lds_buffer):
                    col_base_swz_bytes = swizzle_xor16(
                        curr_row_a_lds, col_base_bytes, k_blocks16
                    )
                    col_base_swz = col_base_swz_bytes // arith.index(int(elem_bytes))
                    idx_a16 = crd2idx((curr_row_a_lds, col_base_swz), layout_lds)
                    loaded_a16 = vector.load_op(vec16_x, lds_buffer, [idx_a16])
                    a_i64x2 = vector.bitcast(vec2_i64, loaded_a16)
                    a0 = vector.extract(a_i64x2, static_position=[0], dynamic_position=[])
                    a1 = vector.extract(a_i64x2, static_position=[1], dynamic_position=[])
                    return a0, a1

                def _a_col_bytes_for_ku(ku_val):
                    """CK-style A col address: L*64 + (ku%4)*16 + (ku//4)*256."""
                    _k0_blk = ku_val // 4
                    _ku_in = ku_val % 4
                    return col_offset_base_bytes + arith.index(
                        _ku_in * _a_ku_stride_bytes + _k0_blk * 256
                    )

                _total_a_slots = k_unroll * m_repeat

                def preload_a_from_lds(lds_buffer, ku_limit=k_unroll):
                    """Load all A tiles for ku_limit × m_repeat from LDS into VGPRs."""
                    a_tiles = [None] * (ku_limit * m_repeat)
                    for ku in range_constexpr(ku_limit):
                        for mi in range_constexpr(m_repeat):
                            col = _a_col_bytes_for_ku(ku)
                            row = row_a_lds + arith.index(mi * 16)
                            a_tiles[ku * m_repeat + mi] = lds_load_packs_k64(
                                row, col, lds_buffer
                            )
                    return a_tiles

                def _mfma_k32(acc_in, a0, a1, b0, b1):
                    a_v2 = vector.from_elements(vec2_i64, [a0, a1])
                    a_v8 = vector.bitcast(vec8_bf16, a_v2)
                    b_v2 = vector.from_elements(vec2_i64, [b0, b1])
                    b_v8 = vector.bitcast(vec8_bf16, b_v2)
                    return mfma_f32_bf16_k32(vec4_f32, [a_v8, b_v8, acc_in, 0, 0, 0])

                def compute_tile(
                    acc_in, b_v4, b_scales, a_tiles_cur,
                    *, ku_count=k_unroll, prefetch_epilogue: bool = False,
                ):
                    """Compute GEMM tile with preloaded A (pure compute, no ds_read).

                    Returns: (acc_list, epilogue_pf).
                    """
                    acc_list = list(acc_in)

                    epilogue_pf = None
                    if prefetch_epilogue:
                        tw_pf = None
                        if doweight_stage2:
                            tw_pf = []
                            lane_div_16_mul4_pf = lane_div_16 * arith.index(4)
                            ii_idx_list_pf = [arith.index(ii) for ii in range(4)]
                            for mi in range_constexpr(m_repeat):
                                mi_base_pf = arith.index(mi * 16)
                                for ii in range_constexpr(4):
                                    row_off_pf = lane_div_16_mul4_pf + ii_idx_list_pf[ii]
                                    row_in_tile_pf = mi_base_pf + row_off_pf
                                    sorted_row_pf = bx_m + row_in_tile_pf
                                    tw_pf.append(
                                        buffer_ops.buffer_load(
                                            sorted_w_rsrc, sorted_row_pf,
                                            vec_width=1, dtype=f32,
                                        )
                                    )
                        epilogue_pf = (None, tw_pf)

                    for ni in range_constexpr(num_acc_n):
                        for ku in range_constexpr(ku_count):
                            _k0_idx = ku // 4
                            _ku_in_k0 = ku % 4

                            b_raw_ku = vector.extract(
                                b_v4[_k0_idx][ni],
                                static_position=[_ku_in_k0],
                                dynamic_position=[],
                            )
                            bb0, bb1 = unpack_b_mxfp4_bf16(
                                b_raw_ku, arith, vector,
                                scale_f32=b_scales[ku][ni],
                            )

                            for mi in range_constexpr(m_repeat):
                                _flat = ku * m_repeat + mi
                                a0, a1 = a_tiles_cur[_flat]

                                acc_idx = mi * num_acc_n + ni
                                acc_list[acc_idx] = _mfma_k32(
                                    acc_list[acc_idx], a0, a1, bb0, bb1,
                                )

                    return acc_list, epilogue_pf

                rocdl.sched_barrier(0)

                def hot_loop_scheduler():
                    """CK-style scheduler: interleave MFMA, DS_READ, VMEM_READ."""
                    _dsread_per_wg = 1
                    _mfma_per_wg = 1
                    _NIterPerWarp = num_acc_n
                    _mfma_perM_perK = _NIterPerWarp * _mfma_per_wg

                    _HalfMIter = (m_repeat + 1) // 2

                    _Aload_num_perK = _dsread_per_wg * m_repeat
                    _Aload_rep = max((_Aload_num_perK + m_repeat - 1) // m_repeat, 1)
                    _Bload_num_perK = num_acc_n
                    _Bload_rep = max((_Bload_num_perK + _HalfMIter - 1) // _HalfMIter, 1)

                    for _ku in range_constexpr(k_unroll):
                        for _mi in range_constexpr(m_repeat):
                            _dsread_perM = _dsread_per_wg
                            _load_perM = 0

                            if _mi < _HalfMIter:
                                _load_perM = (
                                    (_Aload_rep if (_Aload_num_perK - (m_repeat - 1 - _mi) * _Aload_rep) > 0 else 0)
                                    + (_Bload_rep if (_Bload_num_perK - (_HalfMIter - 1 - _mi) * _Bload_rep) > 0 else 0)
                                )
                            else:
                                _load_perM = (
                                    _Aload_rep if (_Aload_num_perK - (m_repeat - 1 - _mi) * _Aload_rep) > 0 else 0
                                )

                            _sum_data = _dsread_perM + _load_perM
                            _round_data = max((_sum_data + _mfma_perM_perK - 1) // _mfma_perM_perK, 1)

                            _inst_order = []
                            _max_data = max(_load_perM, _dsread_perM)
                            for _j in range_constexpr(_max_data):
                                if _load_perM > _j:
                                    _inst_order.append(2)
                                if _dsread_perM > _j:
                                    _inst_order.append(3)
                            _pad_len = _mfma_perM_perK * _round_data - len(_inst_order)
                            _inst_order.extend([0] * _pad_len)

                            for _nj in range_constexpr(_mfma_perM_perK):
                                if _nj == 0:
                                    _inst_idx = 0
                                elif _nj == 1:
                                    _inst_idx = _mfma_perM_perK - 2 if _mfma_perM_perK > 2 else 1
                                elif _nj == 2:
                                    _inst_idx = _mfma_perM_perK - 1
                                else:
                                    _inst_idx = _mfma_perM_perK - _nj

                                rocdl.sched_mfma(1)

                                for _r in range_constexpr(_round_data):
                                    if _r % 2 == 0:
                                        _oi = _inst_idx + _r * _mfma_perM_perK
                                    else:
                                        _oi = (_r + 1) * _mfma_perM_perK - 1 - _inst_idx
                                    if _oi < len(_inst_order):
                                        if _inst_order[_oi] == 2:
                                            rocdl.sched_vmem(1)
                                        elif _inst_order[_oi] == 3:
                                            rocdl.sched_dsrd(1)

                    if _Aload_num_perK == 0:
                        rocdl.sched_vmem(1)
                    rocdl.sched_barrier(0)

                # ---- K-batch offset ----
                if _k_batch > 1:
                    bz = gpu.block_id("z")
                    k_base = bz * arith.index(_k_dim)
                else:
                    k_base = arith.index(0)

                # ---- CK-style pipeline: HEAD (scale prefetch) ----
                k0 = k_base
                prefetch_x_to_lds(k0, lds_x_pong)
                rocdl.sched_barrier(0)

                sc_raw_cur, kps_cur = load_b_scale_raw(k0)
                b_v4_cur = load_b_raw(k0)
                rocdl.sched_barrier(0)

                _k1 = k_base + arith.index(tile_k)
                prefetch_x_to_lds(_k1, lds_x_ping)
                rocdl.sched_barrier(0)

                acc = [acc_init] * (num_acc_n * m_repeat)

                rocdl.s_waitcnt(0)
                gpu.barrier()
                rocdl.sched_barrier(0)
                a_cur = preload_a_from_lds(lds_x_pong)
                b_sc_cur = extract_b_scales(sc_raw_cur, kps_cur)
                gpu.barrier()
                rocdl.sched_barrier(0)

                total_tiles = int(_k_dim) // int(tile_k)
                pair_iters = max((total_tiles - 2) // 2, 0)

                for pair_i in range_constexpr(pair_iters):
                    k_iv = k_base + arith.index(pair_i * (tile_k * 2))

                    # ---- Half 2i: scale prefetch → B_raw → compute → extract → barrier ----
                    rocdl.sched_barrier(0)
                    _k_a2 = k_iv + arith.index(tile_k * 2)
                    prefetch_x_to_lds(_k_a2, lds_x_pong)
                    rocdl.sched_barrier(0)
                    _k_b1 = k_iv + arith.index(tile_k)
                    sc_raw_nxt, kps_nxt = load_b_scale_raw(_k_b1)
                    rocdl.sched_barrier(0)

                    b_v4_nxt = load_b_raw(_k_b1)

                    rocdl.sched_barrier(0)
                    acc, _ = compute_tile(
                        acc, b_v4_cur, b_sc_cur, a_cur,
                    )
                    a_next = preload_a_from_lds(lds_x_ping)
                    rocdl.sched_barrier(0)
                    b_sc_nxt = extract_b_scales(sc_raw_nxt, kps_nxt)

                    rocdl.sched_barrier(0)
                    _barrier(lgkmcnt=2)
                    rocdl.sched_barrier(0)
                    a_cur = a_next

                    # ---- Half 2i+1: scale prefetch → B_raw → compute → extract → barrier ----
                    _k_a3 = k_iv + arith.index(tile_k * 3)
                    prefetch_x_to_lds(_k_a3, lds_x_ping)
                    rocdl.sched_barrier(0)

                    _k_b2 = k_iv + arith.index(tile_k * 2)
                    sc_raw_cur2, kps_cur2 = load_b_scale_raw(_k_b2)
                    b_v4_cur2 = load_b_raw(_k_b2)

                    rocdl.sched_barrier(0)
                    acc, _ = compute_tile(
                        acc, b_v4_nxt, b_sc_nxt, a_cur,
                    )
                    a_next = preload_a_from_lds(lds_x_pong)
                    b_sc_cur2 = extract_b_scales(sc_raw_cur2, kps_cur2)

                    rocdl.sched_barrier(0)
                    _barrier(lgkmcnt=2)
                    rocdl.sched_barrier(0)
                    b_v4_cur, b_sc_cur = b_v4_cur2, b_sc_cur2
                    a_cur = a_next

                # ---- TAIL: last 2 tiles (scale prefetch) ----
                k_tail1 = k_base + arith.index(_k_dim) - arith.index(tile_k)
                if _pad_ku_skip > 0:
                    sc_raw_tail, kps_tail = load_b_scale_raw(k_tail1, k0_limit=_tail_k0_count)
                    b_v4_tail = load_b_raw(k_tail1, k0_limit=_tail_k0_count)
                else:
                    sc_raw_tail, kps_tail = load_b_scale_raw(k_tail1)
                    b_v4_tail = load_b_raw(k_tail1)

                acc, _ = compute_tile(
                    acc, b_v4_cur, b_sc_cur, a_cur,
                )
                if _pad_ku_skip > 0:
                    a_next = preload_a_from_lds(lds_x_ping, ku_limit=_tail_ku)
                    b_sc_tail = extract_b_scales(sc_raw_tail, kps_tail, ku_limit=_tail_ku)
                else:
                    a_next = preload_a_from_lds(lds_x_ping)
                    b_sc_tail = extract_b_scales(sc_raw_tail, kps_tail)

                hot_loop_scheduler()
                rocdl.s_waitcnt(0)
                a_cur = a_next

                acc, epilogue_pf = compute_tile(
                    acc, b_v4_tail, b_sc_tail, a_cur,
                    ku_count=_tail_ku if _pad_ku_skip > 0 else k_unroll,
                    prefetch_epilogue=True,
                )

                # ---- Bias: add to raw accumulators ----
                if enable_bias:
                    _bias_vals = []
                    for _ni in range_constexpr(num_acc_n):
                        _bn = by_n + n_tile_base + arith.index(_ni * 16) + lane_mod_16
                        _bias_vals.append(
                            buffer_ops.buffer_load(
                                bias_rsrc, expert_off_idx + _bn,
                                vec_width=1, dtype=f32
                            )
                        )
                    for _mi in range_constexpr(m_repeat):
                        for _ni in range_constexpr(num_acc_n):
                            _aidx = _mi * num_acc_n + _ni
                            _bsplat = vector.splat(vec4_f32, _bias_vals[_ni])
                            acc[_aidx] = arith.addf(acc[_aidx], _bsplat)

                # ---- Epilogue ----
                expert_off = expert_off_idx
                mask24_i32 = arith.constant(0xFFFFFF, type=T.i32)
                model_i32 = arith.constant(model_dim, type=T.i32)
                topk_i32_v = topk_i32

                zero_i32 = arith.constant(0, type=T.i32)
                c2_i32 = arith.constant(2, type=T.i32)
                mask_even_i32 = arith.constant(0xFFFFFFFE, type=T.i32)
                e_vec = _e_vec

                sw_pf = None
                tw_pf = None
                if epilogue_pf is not None:
                    sw_pf, tw_pf = epilogue_pf

                # No per-channel weight scale for MXFP4 (scales already applied in dequant).
                sw_vals = [arith.constant(1.0, type=T.f32)] * num_acc_n

                if out_is_f32:
                    c4_i32 = arith.constant(4, type=T.i32)

                    def atomic_add_f32(val_f32, byte_off_i32):
                        rocdl.raw_ptr_buffer_atomic_fadd(
                            val_f32, out_rsrc, byte_off_i32, zero_i32, zero_i32,
                        )

                    def _stage2_row_atomic(*, mi: int, ii: int, row_in_tile, row):
                        _fv1 = vector.load_op(vec1_i32, lds_sorted_cache, [row_in_tile])
                        fused2 = vector.extract(_fv1, static_position=[0], dynamic_position=[])
                        t2 = fused2 & mask24_i32
                        s2 = arith.shrui(fused2, arith.constant(24, type=T.i32))
                        t_ok = arith.cmpi(arith.CmpIPredicate.ult, t2, tokens_i32)
                        s_ok = arith.cmpi(arith.CmpIPredicate.ult, s2, topk_i32_v)
                        ts_ok = t_ok & s_ok
                        t2_safe = arith.select(ts_ok, t2, arith.constant(0, type=T.i32))
                        s2_safe = arith.select(ts_ok, s2, arith.constant(0, type=T.i32))
                        sx = arith.select(
                            ts_ok,
                            arith.constant(1.0, type=T.f32),
                            arith.constant(0.0, type=T.f32),
                        )
                        if doweight_stage2:
                            tw_idx = (mi * 4) + ii
                            if tw_pf is not None:
                                tw = arith.select(
                                    ts_ok, tw_pf[tw_idx],
                                    arith.constant(0.0, type=T.f32),
                                )
                            else:
                                tw = arith.select(
                                    ts_ok,
                                    buffer_ops.buffer_load(
                                        sorted_w_rsrc, row, vec_width=1, dtype=f32
                                    ),
                                    arith.constant(0.0, type=T.f32),
                                )
                        idx0 = t2_safe * model_i32

                        for ni in range_constexpr(num_acc_n):
                            col_g = col_g_list[ni]
                            acc_idx = mi * num_acc_n + ni
                            v = vector.extract(
                                acc[acc_idx], static_position=[ii], dynamic_position=[]
                            )
                            v = v * sx
                            if doweight_stage2:
                                v = v * tw
                            col_i32 = arith.index_cast(i32, col_g)
                            idx_elem = idx0 + col_i32
                            byte_off = idx_elem * c4_i32
                            atomic_add_f32(v, byte_off)

                    default_epilog(
                        arith=arith,
                        range_constexpr=range_constexpr,
                        m_repeat=m_repeat,
                        lane_div_16=lane_div_16,
                        bx_m=bx_m,
                        body_row=_stage2_row_atomic,
                    )
                else:
                    if lds_out is None:
                        raise RuntimeError("CShuffle epilogue requires lds_out.")

                    out_base_idx = None
                    if out_is_bf16:
                        _llvm_ptr_ty = ir.Type.parse("!llvm.ptr")
                        _out_base_ptr = _fly.extract_aligned_pointer_as_index(
                            _llvm_ptr_ty, arg_out
                        )
                        out_base_idx = arith.index_cast(
                            T.index, llvm.ptrtoint(T.i64, _out_base_ptr)
                        )

                    def write_row_to_lds(
                        *, mi: int, ii: int, row_in_tile, row,
                        row_base_lds, col_base_local, num_acc_n: int, lds_out,
                    ):
                        _fv1 = vector.load_op(vec1_i32, lds_sorted_cache, [row_in_tile])
                        fused2 = vector.extract(_fv1, static_position=[0], dynamic_position=[])
                        t2 = fused2 & mask24_i32
                        s2 = fused2 >> 24
                        t_ok = arith.cmpi(arith.CmpIPredicate.ult, t2, tokens_i32)
                        s_ok = arith.cmpi(arith.CmpIPredicate.ult, s2, topk_i32_v)
                        ts_ok = t_ok & s_ok
                        sx = arith.constant(1.0, type=T.f32)

                        if doweight_stage2:
                            tw_idx = (mi * 4) + ii
                            if tw_pf is not None:
                                tw = tw_pf[tw_idx]
                            else:
                                tw = buffer_ops.buffer_load(
                                    sorted_w_rsrc, row, vec_width=1, dtype=f32
                                )

                        for ni in range_constexpr(num_acc_n):
                            col_local = col_base_local + (ni * 16)
                            acc_idx = mi * num_acc_n + ni
                            v = vector.extract(
                                acc[acc_idx], static_position=[ii], dynamic_position=[]
                            )
                            v = v * sx
                            if doweight_stage2:
                                v = v * tw
                            v_out = arith.trunc_f(out_elem(), v)
                            lds_idx = row_base_lds + col_local
                            vec1_out = T.vec(1, out_elem())
                            v1 = vector.from_elements(vec1_out, [v_out])
                            vector.store(v1, lds_out, [lds_idx], alignment=2)

                    def precompute_row(*, row_local, row):
                        _fv1 = vector.load_op(vec1_i32, lds_sorted_cache, [row_local])
                        fused2 = vector.extract(_fv1, static_position=[0], dynamic_position=[])
                        row_i32 = arith.index_cast(i32, row)
                        row_valid0 = arith.cmpi(
                            arith.CmpIPredicate.ult, row_i32, num_valid_i32
                        )
                        t = fused2 & mask24_i32
                        s = fused2 >> 24
                        t_ok = arith.cmpi(arith.CmpIPredicate.ult, t, tokens_i32)
                        s_ok = arith.cmpi(arith.CmpIPredicate.ult, s, topk_i32_v)
                        row_valid = row_valid0 & t_ok & s_ok
                        return (fused2, row_valid)

                    def atomic_add_f16x2(val_f16x2, byte_off_i32):
                        rocdl.raw_ptr_buffer_atomic_fadd(
                            val_f16x2, out_rsrc, byte_off_i32, zero_i32, zero_i32,
                        )

                    def store_pair(*, row_local, row, row_ctx, col_pair0, col_g0, frag):
                        fused = row_ctx
                        t = fused & mask24_i32
                        s = fused >> 24
                        idx0 = t * model_i32
                        if not bool(accumulate):
                            ts = t * topk_i32_v + s
                            idx0 = ts * model_i32
                        col_i32 = arith.index_cast(i32, col_g0)
                        idx_elem = idx0 + col_i32
                        idx_elem_even = idx_elem & mask_even_i32
                        if out_is_bf16:
                            if bool(accumulate):
                                byte_off = idx_elem_even * c2_i32
                                byte_off_idx = arith.index_cast(T.index, byte_off)
                                ptr_addr_idx = out_base_idx + byte_off_idx
                                out_ptr = buffer_ops.create_llvm_ptr(
                                    ptr_addr_idx, address_space=1
                                )
                                out_ptr_v = (
                                    out_ptr._value if hasattr(out_ptr, "_value") else out_ptr
                                )
                                frag_v = frag._value if hasattr(frag, "_value") else frag
                                llvm.AtomicRMWOp(
                                    llvm.AtomicBinOp.fadd, out_ptr_v, frag_v,
                                    llvm.AtomicOrdering.monotonic,
                                    syncscope="agent", alignment=4,
                                )
                            else:
                                buffer_ops.buffer_store(frag, out_rsrc, idx_elem_even)
                        else:
                            byte_off = idx_elem_even * c2_i32
                            if bool(accumulate):
                                atomic_add_f16x2(frag, byte_off)
                            else:
                                buffer_ops.buffer_store(frag, out_rsrc, idx_elem_even)

                    c_shuffle_epilog(
                        arith=arith, vector=vector, gpu=gpu, scf=scf,
                        range_constexpr=range_constexpr,
                        tile_m=tile_m, tile_n=tile_n, e_vec=e_vec,
                        m_repeat=m_repeat, num_acc_n=num_acc_n,
                        tx=tx, lane_div_16=lane_div_16, lane_mod_16=lane_mod_16,
                        bx_m=bx_m, by_n=by_n, n_tile_base=n_tile_base,
                        lds_out=lds_out,
                        frag_elem_type=(T.bf16 if out_is_bf16 else T.f16),
                        write_row_to_lds=write_row_to_lds,
                        precompute_row=precompute_row,
                        store_pair=store_pair,
                    )

            _if_blk = scf.IfOp(blk_valid)
            with _if_then(_if_blk):
                _moe_gemm2_then_body()

    _cache_tag = (
        module_name, out_dtype, tile_m, tile_n, tile_k,
        doweight_stage2, accumulate, use_cshuffle_epilog,
        enable_bias, model_dim_pad, inter_dim_pad,
        waves_per_eu, _k_batch,
    )

    @flyc.jit
    def launch_moe_gemm2(
        arg_out: fx.Tensor,
        arg_x: fx.Tensor,
        arg_w: fx.Tensor,
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
        allocator_pong.finalized = False
        allocator_ping.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator_pong.finalize()
            allocator_ping.finalize()

        n_in = arith.ArithValue(arith.index_cast(T.index, i32_n_in.ir_value()))
        model_dim_pad_idx = arith.index(model_dim_pad)
        tile_n_index = arith.index(tile_n)
        size_expert_ids_in = arith.index_cast(
            T.index, i32_size_expert_ids_in.ir_value()
        )
        gx = (n_in - model_dim_pad_idx + tile_n_index - arith.index(1)) // tile_n_index
        gy = size_expert_ids_in

        moe_gemm2(
            arg_out, arg_x, arg_w, arg_scale_w,
            arg_sorted_token_ids, arg_expert_ids, arg_sorted_weights,
            arg_num_valid_ids, arg_bias,
            i32_tokens_in, i32_n_in, i32_k_in, i32_size_expert_ids_in,
        ).launch(
            grid=(gx, gy, _k_batch),
            block=(256, 1, 1),
            stream=stream,
        )

    return launch_moe_gemm2


# ---------------------------------------------------------------------------
# Stage 1: Gate+Up GEMM (MXFP4, gate-up-interleave mode)
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=1024)
def compile_a16w4_moe_gemm1(
    *,
    model_dim: int,
    inter_dim: int,
    experts: int,
    topk: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    doweight_stage1: bool,
    out_dtype: str = "bf16",
    act: str = "silu",
    use_cshuffle_epilog: bool | None = None,
    enable_bias: bool = False,
    model_dim_pad: int = 0,
    inter_dim_pad: int = 0,
    gate_only: bool = False,
    gate_up_interleave: bool = False,
    k_batch: int = 1,
    waves_per_eu: int = 0,
    split_k_intra: int = 1,
):
    """Compile stage1 A16W4 MXFP4 kernel.

    Three modes:
    - **separated** (default): each WG computes both gate and up for the
      same output columns via dual B streams.  Grid X = inter_in / (2*tile_n).
    - **gate_only** (requires k_batch > 1): single B stream, no activation
      fusion.  Grid X = inter_in / tile_n, Z = k_batch.
    - **gate_up_interleave**: weights interleaved along N.  Single B stream,
      adjacent acc slots are gate/up pairs.  Epilogue fuses silu(gate)*up.
      Grid X = inter_in / tile_n.

    A is bf16.  W is MXFP4 with E8M0 scales, pre-shuffled by
    ``shuffle_weight_a16w4`` and ``shuffle_scale_a16w4``.
    """
    if gate_only and gate_up_interleave:
        raise ValueError("gate_only and gate_up_interleave are mutually exclusive")

    _is_splitk = k_batch > 1
    if gate_only and not _is_splitk:
        raise ValueError("gate_only requires k_batch > 1 (split-K)")

    _single_b = gate_only or gate_up_interleave
    accumulate = _is_splitk
    gpu_arch = get_hip_arch()
    allocator_pong = SmemAllocator(None, arch=gpu_arch, global_sym_name="smem0")
    allocator_ping = SmemAllocator(None, arch=gpu_arch, global_sym_name="smem1")

    elem_bytes = 2
    kpack_bytes = 16

    # Padding semantics: model_dim and inter_dim INCLUDE padding.
    #   model_dim = model_dim_true + model_dim_pad   (K direction)
    #   inter_dim = inter_dim_true + inter_dim_pad   (N direction)
    # Tensor sizes use the padded dimensions (inter_dim, model_dim).
    # Padding only affects grid computation (launcher) and is transparent to
    # the GEMM body – the grid simply does not launch tiles for padding columns.
    _inter_dim_valid = inter_dim - inter_dim_pad

    if _is_splitk:
        _k_per_batch = model_dim // k_batch
    else:
        _k_per_batch = model_dim
    _k_dim = _k_per_batch

    _mfma_k32_raw = getattr(rocdl, "mfma_f32_16x16x32_bf16_", None)
    if _mfma_k32_raw is None:
        raise AttributeError(
            "BF16 K32 MFMA op not found: expected `rocdl.mfma_f32_16x16x32_bf16_`"
        )
    _split_mfma = rocdl._split_mfma_operands

    def mfma_f32_bf16_k32(result_type, operands, *, loc=None, ip=None):
        a, b, c, cbsz, abid, blgp = _split_mfma(operands, loc=loc)
        return _mfma_k32_raw(result_type, a, b, c, cbsz, abid, blgp, loc=loc, ip=ip)

    tile_k_bytes = int(tile_k) * int(elem_bytes)
    if (tile_k_bytes % 64) != 0:
        raise ValueError(
            f"tile_k_bytes must be divisible by 64, got tile_k_bytes={tile_k_bytes}"
        )

    if out_dtype not in ("f16", "bf16"):
        raise ValueError(f"out_dtype must be 'f16' or 'bf16', got {out_dtype!r}")
    out_mlir = lambda: (
        (lambda ty: ty() if callable(ty) else ty)(
            T.f16 if out_dtype == "f16" else T.bf16
        )
    )

    DYN = ir.ShapedType.get_dynamic_size()
    size_w = (experts * (2 * inter_dim) * model_dim) // 2

    total_threads = 256
    bytes_x_per_tile = int(tile_m) * int(tile_k) * int(elem_bytes)
    if bytes_x_per_tile % total_threads != 0:
        raise ValueError(
            "tile_m*tile_k*elem_bytes must be divisible by "
            f"{total_threads}: tile_m={tile_m}, tile_k={tile_k}"
        )
    bytes_per_thread_x = bytes_x_per_tile // total_threads

    _ck_lds128 = os.environ.get("FLIR_CK_LDS128", "1") in (
        "1", "true", "True", "YES", "yes",
    )
    pad_k = 0 if _ck_lds128 else 8
    lds_stride = tile_k + pad_k

    if use_cshuffle_epilog is None:
        use_cshuffle_epilog = os.environ.get("FLIR_MOE_STAGE1_CSHUFFLE", "1") in (
            "1", "true", "True", "YES", "yes",
        )
    use_cshuffle_epilog = bool(use_cshuffle_epilog)
    if out_dtype not in ("f16", "bf16") and use_cshuffle_epilog:
        raise ValueError(
            "stage1 cshuffle epilog supports only f16/bf16 output"
        )

    _use_cshuffle_epilog = bool(use_cshuffle_epilog)
    _split_k_intra = split_k_intra

    if _split_k_intra > 1:
        _use_cshuffle_epilog = False
        _waves_per_group = 4 // _split_k_intra
        _n_per_wave_check = tile_n // _waves_per_group
        if _n_per_wave_check < 16:
            raise ValueError(
                f"split_k_intra={_split_k_intra} with tile_n={tile_n}: "
                f"n_per_wave={_n_per_wave_check} < 16 (MFMA minimum)"
            )

    # GUI cross-wave fusion: needed when num_acc_n < 2 per wave
    # (standard pair fusion requires gate+up in same wave, i.e. num_acc_n >= 2)
    _n_per_wave_eff = tile_n // ((4 // _split_k_intra) if _split_k_intra > 1 else 4)
    _gui_xwave_fuse = (
        gate_up_interleave and not _is_splitk
        and (_n_per_wave_eff // 16) < 2
    )
    if _gui_xwave_fuse:
        _use_cshuffle_epilog = False

    # Auto-disable cshuffle when tile_m doesn't meet CShuffleMLane constraint
    if _use_cshuffle_epilog:
        _eff_out_n = (tile_n // 2) if (gate_up_interleave and not _is_splitk) else tile_n
        _cs_nlane_chk = min(32, _eff_out_n // 4)
        if _cs_nlane_chk > 0:
            _cs_mlane_chk = 256 // _cs_nlane_chk
            if tile_m % _cs_mlane_chk != 0:
                _use_cshuffle_epilog = False

    _mode_tag = "gui" if gate_up_interleave else ("go" if gate_only else "sep")
    epilog_tag = "cshuffle" if _use_cshuffle_epilog else "direct"

    _wpe_tag = f"_wpe{waves_per_eu}" if waves_per_eu >= 1 else ""
    _ski_tag = f"_ski{_split_k_intra}" if _split_k_intra > 1 else ""
    _bias_tag = "_bias" if enable_bias else ""
    _act_tag = f"_{act}" if act != "silu" else ""
    _pad_tag = f"_mp{model_dim_pad}_ip{inter_dim_pad}" if (model_dim_pad or inter_dim_pad) else ""
    module_name = (
        f"mfma_a16w4_moe1_mxfp4_{out_dtype}_{_mode_tag}_{epilog_tag}"
        f"_t{tile_m}x{tile_n}x{tile_k}_kb{k_batch}"
        f"{_wpe_tag}{_ski_tag}{_bias_tag}{_act_tag}{_pad_tag}_abi1"
    ).replace("-", "_")

    # For interleave+non-splitk cshuffle, the epilogue output tile_n is halved
    _gui_out_tile_n = tile_n // 2 if (gate_up_interleave and not _is_splitk) else tile_n
    _single_x_bytes = int(tile_m) * int(lds_stride) * int(elem_bytes)
    _single_x_elems = _single_x_bytes // int(elem_bytes)
    lds_out_bytes = 2 * int(tile_m) * int(_gui_out_tile_n) if _use_cshuffle_epilog else 0

    # Ping-pong: pong buffer holds max(input, output), ping buffer holds input only
    _pong_buffer_bytes = max(_single_x_bytes, lds_out_bytes)
    _ping_buffer_bytes = _single_x_bytes

    lds_pong_offset = allocator_pong._align(allocator_pong.ptr, 16)
    allocator_pong.ptr = lds_pong_offset + _pong_buffer_bytes

    lds_ping_offset = allocator_ping._align(allocator_ping.ptr, 16)
    allocator_ping.ptr = lds_ping_offset + _ping_buffer_bytes

    if waves_per_eu >= 1:
        _total_cu_lds = 160 * 1024
        _min_lds = _total_cu_lds // (waves_per_eu + 1) + 1
        _pong_sz = allocator_pong._align(allocator_pong.ptr, 128)
        _ping_sz = allocator_ping._align(allocator_ping.ptr, 128)
        _cur_lds = _pong_sz + _ping_sz
        if _cur_lds < _min_lds:
            allocator_ping.ptr += _min_lds - _cur_lds

    if True:

        @flyc.kernel
        def moe_gemm1(
            arg_out: fx.Tensor,
            arg_x: fx.Tensor,
            arg_w: fx.Tensor,
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
            tokens_in = arith.index_cast(T.index, i32_tokens_in.ir_value())
            inter_in = arith.ArithValue(
                arith.index_cast(T.index, i32_n_in.ir_value())
            )
            k_in = arith.index_cast(T.index, i32_k_in.ir_value())
            size_expert_ids_in = arith.index_cast(
                T.index, i32_size_expert_ids_in.ir_value()
            )
            k_i32_v = i32_k_in.ir_value()

            x_elem = T.bf16
            w_elem = T.i8
            f16 = T.f16
            f32 = T.f32
            i32 = T.i32
            i64 = T.i64
            vec4_f32 = T.vec(4, f32)
            vec1_f16 = T.vec(1, f16)
            vec4_i16 = T.vec(4, T.i16)
            vec8_bf16 = T.vec(8, x_elem)
            vec16_x = T.vec(8, x_elem)
            vec1_i64 = T.vec(1, i64)
            vec2_i64 = T.vec(2, i64)

            def _silu_elem(g):
                neg_log2e = arith.constant(-1.4426950408889634, type=f32)
                t = g * neg_log2e
                emu = llvm.call_intrinsic(f32, "llvm.amdgcn.exp2.f32", [t], [], [])
                one = arith.constant(1.0, type=f32)
                den = one + emu
                sig = llvm.call_intrinsic(f32, "llvm.amdgcn.rcp.f32", [den], [], [])
                return g * sig

            silu = _silu_elem

            def _silu_mul_vec4(gate_v4, up_v4):
                result_elems = []
                for ei in range_constexpr(4):
                    g = vector.extract(
                        gate_v4, static_position=[ei], dynamic_position=[]
                    )
                    u = vector.extract(
                        up_v4, static_position=[ei], dynamic_position=[]
                    )
                    result_elems.append(_silu_elem(g) * u)
                return vector.from_elements(vec4_f32, result_elems)

            def _swiglu_mul_vec4(gate_v4, up_v4):
                result_elems = []
                _alpha = arith.constant(1.702, type=f32)
                _limit = arith.constant(7.0, type=f32)
                _neg_limit = arith.constant(-7.0, type=f32)
                _one = arith.constant(1.0, type=f32)
                _neg_log2e = arith.constant(-1.4426950408889634, type=f32)
                for ei in range_constexpr(4):
                    g = vector.extract(
                        gate_v4, static_position=[ei], dynamic_position=[]
                    )
                    u = vector.extract(
                        up_v4, static_position=[ei], dynamic_position=[]
                    )
                    g = arith.minimumf(g, _limit)
                    u = arith.minimumf(u, _limit)
                    u = arith.maximumf(u, _neg_limit)
                    t = g * _alpha * _neg_log2e
                    emu = llvm.call_intrinsic(
                        f32, "llvm.amdgcn.exp2.f32", [t], [], []
                    )
                    den = _one + emu
                    sig = llvm.call_intrinsic(
                        f32, "llvm.amdgcn.rcp.f32", [den], [], []
                    )
                    result_elems.append(g * sig * (u + _one))
                return vector.from_elements(vec4_f32, result_elems)

            def _act_vec4(gate_v4, up_v4):
                if act == "swiglu":
                    return _swiglu_mul_vec4(gate_v4, up_v4)
                else:
                    return _silu_mul_vec4(gate_v4, up_v4)

            def _act_elem(g_e, u_e):
                if act == "swiglu":
                    _alpha = arith.constant(1.702, type=f32)
                    _limit = arith.constant(7.0, type=f32)
                    _neg_limit = arith.constant(-7.0, type=f32)
                    _one = arith.constant(1.0, type=f32)
                    _neg_log2e = arith.constant(-1.4426950408889634, type=f32)
                    g_e = arith.minimumf(g_e, _limit)
                    u_e = arith.minimumf(u_e, _limit)
                    u_e = arith.maximumf(u_e, _neg_limit)
                    t = g_e * _alpha * _neg_log2e
                    emu = llvm.call_intrinsic(
                        f32, "llvm.amdgcn.exp2.f32", [t], [], []
                    )
                    den = _one + emu
                    sig = llvm.call_intrinsic(
                        f32, "llvm.amdgcn.rcp.f32", [den], [], []
                    )
                    return g_e * sig * (u_e + _one)
                else:
                    return _silu_elem(g_e) * u_e

            acc_init = arith.constant_vector(0.0, vec4_f32)

            layout_x = fx.make_layout(
                (arith.index_cast(i32, tokens_in), k_i32_v), stride=(k_i32_v, 1)
            )

            # Gate+up interleaved: N_total = experts * 2 * inter_dim
            c_n_total = arith.index(experts * (2 * inter_dim))
            c2 = arith.index(2)
            c_k_packed = k_in // c2
            b_layout = make_preshuffle_b_layout(
                arith,
                c_n=c_n_total,
                c_k=c_k_packed,
                kpack_bytes=kpack_bytes,
                elem_bytes=1,
            )
            layout_b = b_layout.layout_b

            layout_b_scale = make_preshuffle_scale_layout(
                arith,
                c_mn=c_n_total,
                c_k=k_in,
                mn_pack=2,
                k_pack=2,
                elem_bytes=4,
                scale_block_size=32,
            )

            shape_lds = fx.make_shape(tile_m, tile_k)
            stride_lds = fx.make_stride(lds_stride, 1)
            layout_lds = fx.make_layout(shape_lds, stride_lds)

            tx = gpu.thread_id("x")

            by = gpu.block_id("x")
            bx = gpu.block_id("y")

            bx_m = bx * arith.index(tile_m)
            numids_rsrc = buffer_ops.create_buffer_resource(
                arg_num_valid_ids, max_size=False,
                num_records_bytes=arith.constant(4, type=i32),
            )
            num_valid_i32 = buffer_ops.buffer_load(
                numids_rsrc, arith.constant(0, index=True), vec_width=1, dtype=i32
            )
            num_valid_idx = arith.index_cast(T.index, num_valid_i32)
            bx_m_i32 = arith.index_cast(i32, bx_m)
            blk_valid = arith.cmpi(arith.CmpIPredicate.ult, bx_m_i32, num_valid_i32)

            k_blocks16 = arith.index(tile_k_bytes // 16)
            layout_tx_wave_lane = fx.make_layout((4, 64), stride=(64, 1))
            layout_lane16 = fx.make_layout((4, 16), stride=(16, 1))

            _if_blk = scf.IfOp(blk_valid)
            with _if_then(_if_blk):
                base_ptr_pong = allocator_pong.get_base()
                base_ptr_ping = allocator_ping.get_base()
                lds_x_pong = SmemPtr(
                    base_ptr_pong, lds_pong_offset, T.bf16,
                    shape=(_single_x_elems,),
                ).get()
                lds_x_ping = SmemPtr(
                    base_ptr_ping, lds_ping_offset, T.bf16,
                    shape=(_single_x_elems,),
                ).get()
                lds_out = (
                    SmemPtr(
                        base_ptr_pong, lds_pong_offset, out_mlir(),
                        shape=(tile_m * _gui_out_tile_n,),
                    ).get()
                    if _use_cshuffle_epilog
                    else None
                )

                c_topk = arith.index(topk)
                x_nbytes_idx = tokens_in * k_in * arith.index(int(elem_bytes))
                x_rsrc = buffer_ops.create_buffer_resource(
                    arg_x, max_size=False, num_records_bytes=x_nbytes_idx
                )
                w_rsrc = buffer_ops.create_buffer_resource(arg_w, max_size=False)
                sw_rsrc = buffer_ops.create_buffer_resource(arg_scale_w, max_size=False)

                # Split-K uses f32 atomics → out_elem_bytes = 4
                out_elem_bytes = 4 if _is_splitk else 2
                out_nbytes_idx = (
                    tokens_in * c_topk * inter_in * arith.index(out_elem_bytes)
                )
                out_rsrc = buffer_ops.create_buffer_resource(
                    arg_out, max_size=False, num_records_bytes=out_nbytes_idx
                )

                sorted_rsrc = buffer_ops.create_buffer_resource(
                    arg_sorted_token_ids, max_size=False
                )
                sorted_w_rsrc = buffer_ops.create_buffer_resource(
                    arg_sorted_weights, max_size=False
                )
                expert_rsrc = buffer_ops.create_buffer_resource(
                    arg_expert_ids, max_size=False,
                    num_records_bytes=(size_expert_ids_in * arith.index(4)),
                )

                expert_i32 = buffer_ops.buffer_load(
                    expert_rsrc, bx, vec_width=1, dtype=i32
                )
                exp_valid = arith.cmpi(
                    arith.CmpIPredicate.ult,
                    expert_i32,
                    arith.constant(experts, type=i32),
                )
                expert_idx = arith.index_cast(T.index, expert_i32)
                inter2_idx = arith.index(2 * inter_dim)
                expert_off_idx = expert_idx * inter2_idx

                bias_rsrc = (
                    buffer_ops.create_buffer_resource(arg_bias, max_size=False)
                    if enable_bias
                    else None
                )

                if bytes_per_thread_x >= 16 and bytes_per_thread_x % 16 == 0:
                    x_load_bytes = 16
                elif bytes_per_thread_x >= 8 and bytes_per_thread_x % 8 == 0:
                    x_load_bytes = 8
                elif bytes_per_thread_x >= 4 and bytes_per_thread_x % 4 == 0:
                    x_load_bytes = 4
                else:
                    raise ValueError(
                        f"bytes_per_thread_x ({bytes_per_thread_x}) must be "
                        f"divisible by 4"
                    )
                num_x_loads = bytes_per_thread_x // x_load_bytes
                chunk_i32 = x_load_bytes // 4
                x_vec_elems = x_load_bytes // elem_bytes
                x_vec_i32_ty = T.vec(chunk_i32, i32) if chunk_i32 > 1 else T.vec(1, i32)
                x_vec_x_ty = T.vec(x_vec_elems, x_elem)

                c_k_div4 = (k_in * arith.index(int(elem_bytes))) // arith.index(4)
                c_k_div4_i32 = arith.index_cast(i32, c_k_div4)
                tile_k_dwords = (int(tile_k) * int(elem_bytes)) // 4
                layout_x_tile_div4 = fx.make_layout(
                    (tile_m, tile_k_dwords), stride=(tile_k_dwords, 1)
                )
                c_chunk_i32 = arith.index(chunk_i32)
                tx_i32_base = tx * c_chunk_i32
                mask24 = arith.constant(0xFFFFFF, type=T.i32)
                tokens_i32 = arith.index_cast(i32, tokens_in)
                topk_i32 = arith.constant(topk, type=T.i32)

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
                x_row_local = []
                for i in range_constexpr(num_x_loads):
                    row_local, col_local_i32 = x_tile_chunk_coord_i32(i)
                    x_row_local.append(row_local)
                    x_col_local_i32.append(col_local_i32)

                    sorted_row_i = bx_m + row_local
                    fused_i = buffer_ops.buffer_load(
                        sorted_rsrc, sorted_row_i, vec_width=1, dtype=i32
                    )
                    t_i32 = fused_i & mask24
                    s_i32 = arith.shrui(fused_i, arith.constant(24, type=i32))
                    t_valid = arith.cmpi(arith.CmpIPredicate.ult, t_i32, tokens_i32)
                    s_valid = arith.cmpi(arith.CmpIPredicate.ult, s_i32, topk_i32)
                    ts_valid = arith.andi(t_valid, s_valid)
                    t_safe = arith.select(
                        ts_valid,
                        arith.index_cast(T.index, t_i32),
                        arith.index(0),
                    )
                    x_row_base_div4.append(t_safe * c_k_div4)

                def load_x_tile(base_k):
                    base_k_div4 = (
                        base_k * arith.index(int(elem_bytes))
                    ) // arith.index(4)
                    parts = []
                    for i in range_constexpr(num_x_loads):
                        idx_i32 = x_row_base_div4[i] + base_k_div4 + x_col_local_i32[i]
                        idx_elem = idx_i32 * arith.index(2)
                        x_vec = _buffer_load_vec(
                            buffer_ops, vector, x_rsrc, idx_elem,
                            elem_type=x_elem,
                            vec_elems=x_vec_elems,
                            elem_bytes=elem_bytes,
                            offset_in_bytes=False,
                        )
                        parts.append(vector.bitcast(x_vec_i32_ty, x_vec))
                    return parts

                coord_wl = idx2crd(tx, layout_tx_wave_lane)
                wave_id = layout_get(coord_wl, 0)
                lane_id = layout_get(coord_wl, 1)

                _dma_bytes = 16
                _wave_size = 64

                def dma_x_tile_to_lds(base_k, lds_buffer):
                    """Async DMA: global -> LDS via buffer_load_lds, no VGPR."""
                    c4_idx = arith.index(4)
                    base_k_div4 = (
                        base_k * arith.index(int(elem_bytes))
                    ) // arith.index(4)

                    lds_ptr_i64 = None
                    for i in range_constexpr(num_x_loads):
                        row_local_i = x_row_local[i]
                        col_local_i32_i = x_col_local_i32[i]
                        col_local_sw = swizzle_xor16(
                            row_local_i, col_local_i32_i * c4_idx, k_blocks16
                        )
                        row_k_dw = x_row_base_div4[i] + base_k_div4
                        global_byte_idx = row_k_dw * c4_idx + col_local_sw
                        global_offset = arith.index_cast(i32, global_byte_idx)

                        if i == 0:
                            lds_addr = memref.extract_aligned_pointer_as_index(
                                lds_buffer
                            ) + wave_id * arith.constant(
                                _wave_size * _dma_bytes, index=True
                            )
                            lds_ptr_i64 = rocdl.readfirstlane(
                                i64, arith.index_cast(i64, lds_addr)
                            )
                        else:
                            lds_ptr_i64 = lds_ptr_i64 + arith.constant(
                                total_threads * _dma_bytes, type=i64
                            )

                        lds_ptr_type = ir.Type.parse("!llvm.ptr<3>")
                        lds_ptr = llvm.inttoptr(lds_ptr_type, lds_ptr_i64)

                        rocdl.raw_ptr_buffer_load_lds(
                            x_rsrc,
                            lds_ptr,
                            arith.constant(_dma_bytes, type=i32),
                            global_offset,
                            arith.constant(0, type=i32),
                            arith.constant(0, type=i32),
                            arith.constant(0, type=i32),
                        )

                def prefetch_x_to_lds(base_k, lds_buffer):
                    dma_x_tile_to_lds(base_k, lds_buffer)

                coord_l16 = idx2crd(lane_id, layout_lane16)
                lane_div_16 = layout_get(coord_l16, 0)
                lane_mod_16 = layout_get(coord_l16, 1)

                row_a_lds = lane_mod_16
                # CK-style A addressing: sub-lane L covers K[L*32..L*32+31]
                # within each k0 block (128 K elements).
                # col_offset_base = lane_div_16 * 32 bf16 = lane_div_16 * 64 bytes
                # stride per ku step = 8 bf16 = 16 bytes
                # For k0 boundaries: ku=4 wraps to next k0 block (+256 bytes)
                _a_sublane_stride = 64   # 32 bf16 * 2 bytes
                _a_ku_stride_bytes = 16  # 8 bf16 * 2 bytes
                col_offset_base_bytes = lane_div_16 * arith.index(_a_sublane_stride)

                by_n = by * arith.index(tile_n)
                _waves_per_group = 4 // _split_k_intra
                n_per_wave = tile_n // _waves_per_group
                num_acc_n = n_per_wave // 16
                c_n_per_wave = arith.index(n_per_wave)
                if _split_k_intra > 1:
                    _wave_group = wave_id // arith.index(_waves_per_group)
                    _wave_in_group = wave_id % arith.index(_waves_per_group)
                    n_tile_base = _wave_in_group * c_n_per_wave
                    # A LDS: each wave group reads different K half
                    _k_half_bytes = (tile_k // _split_k_intra) * elem_bytes
                    col_offset_base_bytes = (
                        col_offset_base_bytes
                        + _wave_group * arith.index(_k_half_bytes)
                    )
                else:
                    _wave_group = None
                    wave_mod_4 = wave_id % arith.index(4)
                    n_tile_base = wave_mod_4 * c_n_per_wave

                n_intra_gate = []
                n_blk_gate = []
                col_g_list = []
                inter_idx = arith.index(inter_dim)
                c_n0_static = experts * (2 * inter_dim) // 16
                layout_n_blk_intra = fx.make_layout((c_n0_static, 16), stride=(16, 1))

                if not _single_b:
                    # Separated mode: gate and up are distinct N regions.
                    n_intra_up = []
                    n_blk_up = []
                    for ni in range_constexpr(num_acc_n):
                        offset = arith.index(ni * 16)
                        col_g = by_n + n_tile_base + offset + lane_mod_16
                        col_g_list.append(col_g)

                        row_gate = expert_off_idx + col_g
                        row_up = row_gate + inter_idx

                        coord_gate = idx2crd(row_gate, layout_n_blk_intra)
                        n_blk_gate.append(layout_get(coord_gate, 0))
                        n_intra_gate.append(layout_get(coord_gate, 1))

                        coord_up = idx2crd(row_up, layout_n_blk_intra)
                        n_blk_up.append(layout_get(coord_up, 0))
                        n_intra_up.append(layout_get(coord_up, 1))
                else:
                    # gate_only / gate_up_interleave: single B stream
                    n_intra_up = None
                    n_blk_up = None
                    for ni in range_constexpr(num_acc_n):
                        offset = arith.index(ni * 16)
                        global_n = by_n + n_tile_base + offset + lane_mod_16
                        gate_row_w = expert_off_idx + global_n

                        coord_gate = idx2crd(gate_row_w, layout_n_blk_intra)
                        n_blk_gate.append(layout_get(coord_gate, 0))
                        n_intra_gate.append(layout_get(coord_gate, 1))

                    if gate_up_interleave and not _is_splitk:
                        if _gui_xwave_fuse:
                            if _split_k_intra > 1:
                                # Split_k xwave: all waves share same output cols
                                _gui_col_g = (
                                    by_n // arith.index(2) + lane_mod_16
                                )
                            else:
                                # 4-wave xwave: per-pair output cols
                                # pair (0,1)→cols[0:15], pair (2,3)→cols[16:31]
                                _xw_pair_off = (
                                    (wave_id // arith.index(2))
                                    * c_n_per_wave
                                )
                                _gui_col_g = (
                                    by_n // arith.index(2)
                                    + _xw_pair_off
                                    + lane_mod_16
                                )
                            col_g_list.append(_gui_col_g)
                        else:
                            # Standard pair fusion: pairs of N subtiles → one output col
                            pack_N = 2
                            _gui_num_acc_n_out = num_acc_n // pack_N
                            for _gui_i in range_constexpr(_gui_num_acc_n_out):
                                _gui_offset = arith.index(_gui_i * 16)
                                _gui_col_g = (
                                    (by_n + n_tile_base)
                                    // arith.index(2)
                                    + _gui_offset
                                    + lane_mod_16
                                )
                                col_g_list.append(_gui_col_g)
                    else:
                        # gate_only or interleave+splitk: output covers full N
                        for ni in range_constexpr(num_acc_n):
                            offset = arith.index(ni * 16)
                            col_g = by_n + n_tile_base + offset + lane_mod_16
                            col_g_list.append(col_g)

                m_repeat = tile_m // 16
                k_unroll = (tile_k_bytes // 64) // _split_k_intra

                _pad_k_elems = (model_dim_pad % tile_k) if (not _is_splitk and _split_k_intra == 1 and model_dim_pad > 0) else 0
                _pad_ku_skip = _pad_k_elems // 32
                _tail_ku = k_unroll - _pad_ku_skip
                _tail_k0_count = (_tail_ku + 3) // 4 if _pad_ku_skip > 0 else None

                # Each dwordx4 covers 128 K elements; per-group count
                _k_per_dwordx4 = 128
                _k0_count = (tile_k // _k_per_dwordx4) // _split_k_intra

                # Scale mni for gate (and up if separated)
                scale_mni_gate = []
                scale_n_pack_gate = []
                for ni in range_constexpr(num_acc_n):
                    n_gate = expert_off_idx + by_n + n_tile_base + arith.index(ni * 16)
                    if not _single_b:
                        n_gate_phys = n_gate
                    else:
                        n_gate_phys = n_gate
                    scale_mni_gate.append(n_gate_phys // arith.index(32))
                    scale_n_pack_gate.append(
                        (n_gate_phys // arith.index(16)) % arith.index(2)
                    )

                if not _single_b:
                    scale_mni_up = []
                    scale_n_pack_up = []
                    for ni in range_constexpr(num_acc_n):
                        n_up = (
                            expert_off_idx + by_n + n_tile_base
                            + arith.index(ni * 16) + inter_idx
                        )
                        scale_mni_up.append(n_up // arith.index(32))
                        scale_n_pack_up.append(
                            (n_up // arith.index(16)) % arith.index(2)
                        )
                else:
                    scale_mni_up = None
                    scale_n_pack_up = None

                def _load_scale_i32(scale_ku_idx, mni_val, scale_klane=None):
                    _klane = scale_klane if scale_klane is not None else lane_div_16
                    idx = (mni_val * layout_b_scale.stride_n0
                           + scale_ku_idx * layout_b_scale.stride_k0
                           + _klane * layout_b_scale.stride_klane
                           + lane_mod_16)
                    return buffer_ops.buffer_load(sw_rsrc, idx, vec_width=1, dtype=i32)

                def _extract_e8m0_f32_dynamic(packed_i32, byte_pos_idx):
                    """Extract E8M0 byte at runtime byte_pos and decode to f32."""
                    shift = arith.index_cast(i32, byte_pos_idx) * arith.constant(8, type=i32)
                    byte_i32 = arith.shrui(packed_i32, shift) & arith.constant(0xFF, type=i32)
                    scale_bits = arith.shli(byte_i32, arith.constant(23, type=i32))
                    return arith.bitcast(f32, scale_bits)

                def _get_scale_f32(base_k, ku, ni, mni_list, n_pack_list, scale_cache):
                    # CK addressing: adj_ku = base_k//32 + (ku//4)*4 + lane_div_16
                    # scale_klane = lane_div_16, k_pack_sub = (ku//4) % 2
                    _k0_blk = ku // 4
                    adj_ku = (base_k // arith.index(32)
                              + arith.index(_k0_blk * 4)
                              + lane_div_16)
                    scale_klane_rt = lane_div_16
                    k_pack_sub_rt = (adj_ku // arith.index(4)) % arith.index(2)
                    s_ku = adj_ku // arith.index(8)

                    cache_key = (_k0_blk, ni, id(mni_list))
                    if cache_key not in scale_cache:
                        scale_cache[cache_key] = _load_scale_i32(
                            s_ku, mni_list[ni], scale_klane=scale_klane_rt
                        )
                    packed = scale_cache[cache_key]
                    n_pack_sub_val = n_pack_list[ni]
                    byte_pos_even = k_pack_sub_rt * arith.index(2)
                    byte_pos_odd = byte_pos_even + arith.index(1)
                    scale_even = _extract_e8m0_f32_dynamic(packed, byte_pos_even)
                    scale_odd = _extract_e8m0_f32_dynamic(packed, byte_pos_odd)
                    n_pack_is_zero = arith.cmpi(
                        arith.CmpIPredicate.eq,
                        arith.index_cast(i32, n_pack_sub_val),
                        arith.constant(0, type=i32),
                    )
                    return arith.select(n_pack_is_zero, scale_even, scale_odd)

                def load_b_raw(base_k, blk_list, intra_list, k0_limit=_k0_count):
                    """Load raw FP4 data via dwordx4. Returns raw_v4[k0_idx][ni] = vec4_i32."""
                    raw_all = []
                    for k0_idx in range_constexpr(k0_limit):
                        raw_k0 = []
                        k_off = base_k + arith.index(k0_idx * _k_per_dwordx4)
                        for ni in range_constexpr(num_acc_n):
                            v4 = load_b_raw_mxfp4_dwordx4(
                                buffer_ops, arith, vector,
                                arg_b=arg_w,
                                b_rsrc=w_rsrc,
                                layout_b=layout_b,
                                base_k=k_off,
                                n_blk=blk_list[ni],
                                n_intra=intra_list[ni],
                                lane_div_16=lane_div_16,
                                elem_type=w_elem,
                                kpack_bytes=kpack_bytes,
                                cache_modifier=2,
                            )
                            raw_k0.append(v4)
                        raw_all.append(raw_k0)
                    return raw_all

                def load_b_scale(base_k, mni_list, n_pack_list, ku_limit=k_unroll):
                    """Load scales for all ku × ni. Returns scales[ku][ni] = f32."""
                    scale_cache = {}
                    scales = []
                    for ku in range_constexpr(ku_limit):
                        scales_ku = []
                        for ni in range_constexpr(num_acc_n):
                            sf = _get_scale_f32(
                                base_k, ku, ni, mni_list, n_pack_list, scale_cache
                            )
                            scales_ku.append(sf)
                        scales.append(scales_ku)
                    return scales

                def load_all_b_raw(base_k, k0_limit=_k0_count, ku_limit=k_unroll):
                    """Load raw B (dwordx4) + scales for gate (and up if separated)."""
                    g_scales = load_b_scale(base_k, scale_mni_gate, scale_n_pack_gate, ku_limit=ku_limit)
                    u_scales = None
                    if not _single_b:
                        u_scales = load_b_scale(base_k, scale_mni_up, scale_n_pack_up, ku_limit=ku_limit)

                    g_v4 = load_b_raw(base_k, n_blk_gate, n_intra_gate, k0_limit=k0_limit)
                    u_v4 = None
                    if not _single_b:
                        u_v4 = load_b_raw(base_k, n_blk_up, n_intra_up, k0_limit=k0_limit)
                    return g_v4, g_scales, u_v4, u_scales

                def store_x_tile_to_lds(vec_x_in_parts, lds_buffer):
                    _lds_base_zero = arith.index(0)
                    for i in range_constexpr(num_x_loads):
                        row_local = x_row_local[i]
                        col_local_i32 = x_col_local_i32[i]
                        if x_load_bytes >= 16:
                            lds_store_16b_xor16(
                                arith, vector,
                                lds_memref=lds_buffer,
                                vec16_ty=vec16_x,
                                layout_lds=layout_lds,
                                row_local=row_local,
                                col_local_i32=col_local_i32,
                                tx_c4=arith.index(4),
                                k_blocks16=k_blocks16,
                                lds_base=_lds_base_zero,
                                vec_part_i32x4=vec_x_in_parts[i],
                                elem_bytes=elem_bytes,
                            )
                        else:
                            lds_store_8b_xor16(
                                arith, vector,
                                lds_memref=lds_buffer,
                                vec8_ty=x_vec_x_ty,
                                layout_lds=layout_lds,
                                row_local=row_local,
                                col_local_i32=col_local_i32,
                                tx_c4=arith.index(4),
                                k_blocks16=k_blocks16,
                                lds_base=_lds_base_zero,
                                vec_part_i32x2=vec_x_in_parts[i],
                                elem_bytes=elem_bytes,
                            )

                def lds_load_packs_k64(curr_row_a_lds, col_base_bytes, lds_buffer):
                    col_base_swz_bytes = swizzle_xor16(
                        curr_row_a_lds, col_base_bytes, k_blocks16
                    )
                    col_base_swz = col_base_swz_bytes // arith.index(int(elem_bytes))
                    idx_a16 = crd2idx((curr_row_a_lds, col_base_swz), layout_lds)
                    loaded_a16 = vector.load_op(vec16_x, lds_buffer, [idx_a16])
                    a_i64x2 = vector.bitcast(vec2_i64, loaded_a16)
                    a0 = vector.extract(a_i64x2, static_position=[0], dynamic_position=[])
                    a1 = vector.extract(a_i64x2, static_position=[1], dynamic_position=[])
                    return a0, a1

                def _a_col_bytes_for_ku(ku_val):
                    """CK-style A col address: L*64 + (ku%4)*16 + (ku//4)*256."""
                    _k0_blk = ku_val // 4
                    _ku_in = ku_val % 4
                    return col_offset_base_bytes + arith.index(
                        _ku_in * _a_ku_stride_bytes + _k0_blk * 256
                    )

                _can_full_preload = (m_repeat == 1)
                _m_preload = (k_unroll * m_repeat) if _can_full_preload else min(2, k_unroll * m_repeat)
                _total_a_slots = k_unroll * m_repeat

                def preload_a_from_lds(lds_buffer):
                    """Load first _m_preload A tiles from LDS into VGPRs."""
                    a_tiles = [None] * _m_preload
                    for pl in range_constexpr(_m_preload):
                        _pl_mi = pl % m_repeat
                        _pl_ku = pl // m_repeat
                        col = _a_col_bytes_for_ku(_pl_ku)
                        row = row_a_lds + arith.index(_pl_mi * 16)
                        a_tiles[pl] = lds_load_packs_k64(row, col, lds_buffer)
                    return a_tiles

                def _mfma_k32(acc_in, a0, a1, b0, b1):
                    a_v2 = vector.from_elements(vec2_i64, [a0, a1])
                    a_v8 = vector.bitcast(vec8_bf16, a_v2)
                    b_v2 = vector.from_elements(vec2_i64, [b0, b1])
                    b_v8 = vector.bitcast(vec8_bf16, b_v2)
                    return mfma_f32_bf16_k32(vec4_f32, [a_v8, b_v8, acc_in, 0, 0, 0])

                def compute_tile(
                    acc_gate_in, acc_up_in,
                    g_v4, g_scales, u_v4, u_scales,
                    a_preloaded, cur_lds_buffer, next_lds_buffer,
                    ku_count=k_unroll,
                ):
                    """Compute GEMM tile with preloaded A.

                    Full preload (m_repeat=1): ni→ku, all A in VGPRs.
                    Partial preload (m_repeat>1): ku→mi→ni, m_preload=2
                    pipeline, reload remaining A from cur_lds_buffer.

                    ku_count: number of k_unroll iterations to execute
                    (< k_unroll for the last tile when K-padding is active).

                    Returns: (gate_list, up_list, a_tiles_next).
                    a_tiles_next has _m_preload tiles from next_lds_buffer.
                    """
                    gate_list = list(acc_gate_in)
                    up_list = list(acc_up_in) if acc_up_in is not None else None
                    a_tiles_next = [None] * _m_preload

                    if _can_full_preload:
                        # --- Full preload: ni → ku → mi ---
                        _is_last_ni = num_acc_n - 1

                        for ni in range_constexpr(num_acc_n):
                            for ku in range_constexpr(ku_count):
                                _k0_idx = ku // 4
                                _ku_in_k0 = ku % 4

                                g_raw_ku = vector.extract(
                                    g_v4[_k0_idx][ni],
                                    static_position=[_ku_in_k0],
                                    dynamic_position=[],
                                )
                                gb0, gb1 = unpack_b_mxfp4_bf16(
                                    g_raw_ku, arith, vector,
                                    scale_f32=g_scales[ku][ni],
                                )
                                if up_list is not None:
                                    u_raw_ku = vector.extract(
                                        u_v4[_k0_idx][ni],
                                        static_position=[_ku_in_k0],
                                        dynamic_position=[],
                                    )
                                    ub0, ub1 = unpack_b_mxfp4_bf16(
                                        u_raw_ku, arith, vector,
                                        scale_f32=u_scales[ku][ni],
                                    )

                                for mi in range_constexpr(m_repeat):
                                    _flat = ku * m_repeat + mi
                                    a0, a1 = a_preloaded[_flat]
                                    acc_idx = mi * num_acc_n + ni
                                    gate_list[acc_idx] = _mfma_k32(
                                        gate_list[acc_idx], a0, a1, gb0, gb1,
                                    )
                                    if up_list is not None:
                                        up_list[acc_idx] = _mfma_k32(
                                            up_list[acc_idx], a0, a1, ub0, ub1,
                                        )

                                if next_lds_buffer is not None and ni == _is_last_ni:
                                    for mi in range_constexpr(m_repeat):
                                        _flat = ku * m_repeat + mi
                                        _nxt_col = _a_col_bytes_for_ku(ku)
                                        _nxt_row = row_a_lds + arith.index(mi * 16)
                                        a_tiles_next[_flat] = lds_load_packs_k64(
                                            _nxt_row, _nxt_col, next_lds_buffer
                                        )
                    else:
                        # --- bm=32: full load from cur_lds inside compute ---
                        _local_a_slots = ku_count * m_repeat
                        all_a = [None] * _local_a_slots
                        for ku in range_constexpr(ku_count):
                            for mi in range_constexpr(m_repeat):
                                _col = _a_col_bytes_for_ku(ku)
                                _row = row_a_lds + arith.index(mi * 16)
                                all_a[ku * m_repeat + mi] = lds_load_packs_k64(
                                    _row, _col, cur_lds_buffer
                                )

                        for ni in range_constexpr(num_acc_n):
                            for ku in range_constexpr(ku_count):
                                _k0_idx = ku // 4
                                _ku_in_k0 = ku % 4

                                g_raw_ku = vector.extract(
                                    g_v4[_k0_idx][ni],
                                    static_position=[_ku_in_k0],
                                    dynamic_position=[],
                                )
                                gb0, gb1 = unpack_b_mxfp4_bf16(
                                    g_raw_ku, arith, vector,
                                    scale_f32=g_scales[ku][ni],
                                )
                                if up_list is not None:
                                    u_raw_ku = vector.extract(
                                        u_v4[_k0_idx][ni],
                                        static_position=[_ku_in_k0],
                                        dynamic_position=[],
                                    )
                                    ub0, ub1 = unpack_b_mxfp4_bf16(
                                        u_raw_ku, arith, vector,
                                        scale_f32=u_scales[ku][ni],
                                    )

                                for mi in range_constexpr(m_repeat):
                                    _flat = ku * m_repeat + mi
                                    a0, a1 = all_a[_flat]
                                    acc_idx = mi * num_acc_n + ni
                                    gate_list[acc_idx] = _mfma_k32(
                                        gate_list[acc_idx], a0, a1, gb0, gb1,
                                    )
                                    if up_list is not None:
                                        up_list[acc_idx] = _mfma_k32(
                                            up_list[acc_idx], a0, a1, ub0, ub1,
                                        )

                        # Load next round's preload from next_lds
                        if next_lds_buffer is not None:
                            for pl in range_constexpr(_m_preload):
                                _pl_mi = pl % m_repeat
                                _pl_ku = pl // m_repeat
                                _nxt_col = _a_col_bytes_for_ku(_pl_ku)
                                _nxt_row = row_a_lds + arith.index(_pl_mi * 16)
                                a_tiles_next[pl] = lds_load_packs_k64(
                                    _nxt_row, _nxt_col, next_lds_buffer
                                )

                    return gate_list, up_list, a_tiles_next

                rocdl.sched_barrier(0)

                _b_stream_mult = 1 if _single_b else 2

                def hot_loop_scheduler():
                    """CK-style scheduler: interleave MFMA, DS_READ, VMEM_READ."""
                    # Constants matching CK:
                    # dsread_per_wg = WG_M * WG_K * sizeof(bf16) / 64 / VectorLoadSize(16) = 16*32*2/64/16 = 1
                    _dsread_per_wg = 1
                    _mfma_per_wg = 1
                    _NIterPerWarp = num_acc_n * _b_stream_mult  # 2 or 4
                    _mfma_perM_perK = _NIterPerWarp * _mfma_per_wg

                    _HalfMIter = (m_repeat + 1) // 2

                    _Aload_num_perK = _dsread_per_wg * m_repeat  # num buffer_load_lds per K iter
                    _Aload_rep = max((_Aload_num_perK + m_repeat - 1) // m_repeat, 1)
                    _Bload_num_perK = num_acc_n * _b_stream_mult  # B loads per K iter
                    _Bload_rep = max((_Bload_num_perK + _HalfMIter - 1) // _HalfMIter, 1)

                    for _ku in range_constexpr(k_unroll):
                        for _mi in range_constexpr(m_repeat):
                            _dsread_perM = _dsread_per_wg
                            _load_perM = 0

                            if _mi < _HalfMIter:
                                _load_perM = (
                                    (_Aload_rep if (_Aload_num_perK - (m_repeat - 1 - _mi) * _Aload_rep) > 0 else 0)
                                    + (_Bload_rep if (_Bload_num_perK - (_HalfMIter - 1 - _mi) * _Bload_rep) > 0 else 0)
                                )
                            else:
                                _load_perM = (
                                    _Aload_rep if (_Aload_num_perK - (m_repeat - 1 - _mi) * _Aload_rep) > 0 else 0
                                )

                            _sum_data = _dsread_perM + _load_perM
                            _round_data = max((_sum_data + _mfma_perM_perK - 1) // _mfma_perM_perK, 1)

                            # Build instruction order: 2=VMEM, 3=DS_READ
                            _inst_order = []
                            _max_data = max(_load_perM, _dsread_perM)
                            for _j in range_constexpr(_max_data):
                                if _load_perM > _j:
                                    _inst_order.append(2)
                                if _dsread_perM > _j:
                                    _inst_order.append(3)
                            # Pad to mfma_perM_perK * round_data
                            _pad_len = _mfma_perM_perK * _round_data - len(_inst_order)
                            _inst_order.extend([0] * _pad_len)

                            for _nj in range_constexpr(_mfma_perM_perK):
                                if _nj == 0:
                                    _inst_idx = 0
                                elif _nj == 1:
                                    _inst_idx = _mfma_perM_perK - 2 if _mfma_perM_perK > 2 else 1
                                elif _nj == 2:
                                    _inst_idx = _mfma_perM_perK - 1
                                else:
                                    _inst_idx = _mfma_perM_perK - _nj

                                rocdl.sched_mfma(1)

                                for _r in range_constexpr(_round_data):
                                    if _r % 2 == 0:
                                        _oi = _inst_idx + _r * _mfma_perM_perK
                                    else:
                                        _oi = (_r + 1) * _mfma_perM_perK - 1 - _inst_idx
                                    if _oi < len(_inst_order):
                                        if _inst_order[_oi] == 2:
                                            rocdl.sched_vmem(1)
                                        elif _inst_order[_oi] == 3:
                                            rocdl.sched_dsrd(1)

                    if _Aload_num_perK == 0:
                        rocdl.sched_vmem(1)
                    rocdl.sched_barrier(0)

                def _s1_barrier(vmcnt=63, lgkmcnt=63):
                    """s_waitcnt + s_barrier via inline asm (bypasses LLVM)."""
                    parts = []
                    needs_waitcnt = vmcnt < 63 or lgkmcnt < 63
                    if needs_waitcnt:
                        wc = []
                        if vmcnt < 63:
                            wc.append(f"vmcnt({vmcnt})")
                        if lgkmcnt < 63:
                            wc.append(f"lgkmcnt({lgkmcnt})")
                        parts.append("s_waitcnt " + " ".join(wc))
                    parts.append("s_barrier")
                    llvm.InlineAsmOp(
                        res=None,
                        operands_=[],
                        asm_string="\n".join(parts),
                        constraints="",
                        has_side_effects=True,
                        is_align_stack=False,
                    )

                # ---- CK-style constants ----
                _vmcnt_before_barrier = num_x_loads

                # Split-K: base K offset
                if _is_splitk:
                    bz = gpu.block_id("z")
                    k_base = bz * arith.index(_k_dim)
                else:
                    k_base = arith.index(0)

                # Intra-WG split-K: B offset per wave group
                if _split_k_intra > 1:
                    _k_half = tile_k // _split_k_intra
                    _wg_k_off = _wave_group * arith.index(_k_half)
                else:
                    _wg_k_off = arith.index(0)

                # ---- CK-style pipeline: HEAD ----
                # DMA A[0] → pong (full tile_k, shared by all wave groups)
                k0 = k_base
                prefetch_x_to_lds(k0, lds_x_pong)
                rocdl.sched_barrier(0)

                # Load B[0] (raw + scale, per-group K range)
                g_raw_ping, g_sc_ping, u_raw_ping, u_sc_ping = load_all_b_raw(
                    k0 + _wg_k_off
                )

                rocdl.sched_barrier(0)

                # DMA A[1] → ping
                _k1 = k_base + arith.index(tile_k)

                prefetch_x_to_lds(_k1, lds_x_ping)
                rocdl.sched_barrier(0)

                # Init C
                acc_gate = [acc_init] * (num_acc_n * m_repeat)
                acc_up = [acc_init] * (num_acc_n * m_repeat) if not _single_b else None

                # Wait for all DMA + barrier to sync all threads
                rocdl.s_waitcnt(0)
                gpu.barrier()
                rocdl.sched_barrier(0)

                # Preload A[0] from pong LDS → VGPRs (safe: all threads' DMA done)
                a_cur = preload_a_from_lds(lds_x_pong)
                rocdl.sched_barrier(0)

                total_tiles = int(_k_dim) // int(tile_k)
                pair_iters = max((total_tiles - 2) // 2, 0)

                for pair_i in range_constexpr(pair_iters):
                    k_iv = k_base + arith.index(pair_i * (tile_k * 2))

                    # ---- Half 2i ----
                    _k_b1 = k_iv + arith.index(tile_k)
                    g_raw_pong, g_sc_pong, u_raw_pong, u_sc_pong = load_all_b_raw(
                        _k_b1 + _wg_k_off
                    )

                    acc_gate, acc_up, a_next = compute_tile(
                        acc_gate, acc_up,
                        g_raw_ping, g_sc_ping, u_raw_ping, u_sc_ping,
                        a_cur, lds_x_pong, lds_x_ping,
                    )

                    _k_a2 = k_iv + arith.index(tile_k * 2)
                    prefetch_x_to_lds(_k_a2, lds_x_pong)
                    rocdl.s_waitcnt(0)
                    gpu.barrier()
                    rocdl.sched_barrier(0)
                    a_cur = a_next

                    # ---- Half 2i+1 ----
                    _k_b2 = k_iv + arith.index(tile_k * 2)
                    g_raw_ping, g_sc_ping, u_raw_ping, u_sc_ping = load_all_b_raw(
                        _k_b2 + _wg_k_off
                    )

                    acc_gate, acc_up, a_next = compute_tile(
                        acc_gate, acc_up,
                        g_raw_pong, g_sc_pong, u_raw_pong, u_sc_pong,
                        a_cur, lds_x_ping, lds_x_pong,
                    )

                    _k_a3 = k_iv + arith.index(tile_k * 3)
                    prefetch_x_to_lds(_k_a3, lds_x_ping)
                    rocdl.s_waitcnt(0)
                    gpu.barrier()
                    rocdl.sched_barrier(0)
                    a_cur = a_next

                # ---- TAIL: last 2 tiles ----
                # Load B for tail-1 (partial if K-padding active)
                k_tail1 = k_base + arith.index(_k_dim) - arith.index(tile_k)
                if _pad_ku_skip > 0:
                    g_raw_pong, g_sc_pong, u_raw_pong, u_sc_pong = load_all_b_raw(
                        k_tail1 + _wg_k_off,
                        k0_limit=_tail_k0_count,
                        ku_limit=_tail_ku,
                    )
                else:
                    g_raw_pong, g_sc_pong, u_raw_pong, u_sc_pong = load_all_b_raw(
                        k_tail1 + _wg_k_off
                    )

                # GEMM tail-0: use a_cur (from pong), load a_next from ping
                acc_gate, acc_up, a_next = compute_tile(
                    acc_gate, acc_up,
                    g_raw_ping, g_sc_ping, u_raw_ping, u_sc_ping,
                    a_cur, lds_x_pong, lds_x_ping,
                )
                hot_loop_scheduler()
                rocdl.s_waitcnt(0)
                # _s1_barrier()

                # GEMM tail-1: use a_next (from ping), no next round
                # When K-padding is active, only compute the valid portion
                acc_gate, acc_up, _ = compute_tile(
                    acc_gate, acc_up,
                    g_raw_pong, g_sc_pong, u_raw_pong, u_sc_pong,
                    a_next, lds_x_ping, None,
                    ku_count=_tail_ku if _pad_ku_skip > 0 else k_unroll,
                )

                # ---- Intra-WG split-K reduce via LDS ----
                if _split_k_intra > 1:
                    _num_accs = num_acc_n * m_repeat
                    _has_up = not _single_b
                    _streams = 2 if _has_up else 1
                    # 4 f32 per vec4_f32 acc × _num_accs × _streams per thread
                    _f32_per_thread = 4 * _num_accs * _streams
                    _reduce_stride = arith.index(_f32_per_thread)
                    # 4 waves × 64 threads × _f32_per_thread
                    _reduce_f32_total = 4 * 64 * _f32_per_thread

                    reduce_lds = SmemPtr(
                        base_ptr_pong, lds_pong_offset, T.f32,
                        shape=(_reduce_f32_total,),
                    ).get()

                    tx_local = lane_id
                    _lds_base = (
                        wave_id * arith.index(64) * _reduce_stride
                        + tx_local * _reduce_stride
                    )

                    for _ai in range_constexpr(_num_accs):
                        _off = _lds_base + arith.index(_ai * 4)
                        vector.store(acc_gate[_ai], reduce_lds, [_off])
                    if _has_up:
                        for _ai in range_constexpr(_num_accs):
                            _off = _lds_base + arith.index(_num_accs * 4 + _ai * 4)
                            vector.store(acc_up[_ai], reduce_lds, [_off])

                    gpu.barrier()

                    _partner_wave = (
                        (wave_id + arith.index(_waves_per_group))
                        % arith.index(4)
                    )
                    _partner_base = (
                        _partner_wave * arith.index(64) * _reduce_stride
                        + tx_local * _reduce_stride
                    )

                    for _ai in range_constexpr(_num_accs):
                        _off = _partner_base + arith.index(_ai * 4)
                        other_acc = vector.load_op(vec4_f32, reduce_lds, [_off])
                        acc_gate[_ai] = arith.addf(acc_gate[_ai], other_acc)
                    if _has_up:
                        for _ai in range_constexpr(_num_accs):
                            _off = _partner_base + arith.index(
                                _num_accs * 4 + _ai * 4
                            )
                            other_up = vector.load_op(vec4_f32, reduce_lds, [_off])
                            acc_up[_ai] = arith.addf(acc_up[_ai], other_up)

                    # Cross-wave gate-up fusion for GUI + split_k + num_acc_n=1
                    if _gui_xwave_fuse:
                        gpu.barrier()
                        for _ai in range_constexpr(_num_accs):
                            _off = _lds_base + arith.index(_ai * 4)
                            vector.store(acc_gate[_ai], reduce_lds, [_off])
                        gpu.barrier()

                        # Read partner within same wave group (0↔1)
                        _xw_partner_in_grp = (
                            arith.index(1) - _wave_in_group
                        )
                        _xw_partner_wave = (
                            _wave_group * arith.index(_waves_per_group)
                            + _xw_partner_in_grp
                        )
                        _xw_partner_base = (
                            _xw_partner_wave * arith.index(64) * _reduce_stride
                            + tx_local * _reduce_stride
                        )

                        # silu(gate) * up — gate is wave_in_group=0, up is wave_in_group=1
                        _is_gate_wave = arith.cmpi(
                            arith.CmpIPredicate.eq,
                            arith.index_cast(i32, _wave_in_group),
                            arith.constant(0, type=T.i32),
                        )
                        for _ai in range_constexpr(_num_accs):
                            _xw_off = _xw_partner_base + arith.index(_ai * 4)
                            _xw_partner_acc = vector.load_op(
                                vec4_f32, reduce_lds, [_xw_off],
                            )
                            _fused_elems = [None] * 4
                            for _e in range_constexpr(4):
                                own_e = vector.extract(
                                    acc_gate[_ai],
                                    static_position=[_e], dynamic_position=[],
                                )
                                par_e = vector.extract(
                                    _xw_partner_acc,
                                    static_position=[_e], dynamic_position=[],
                                )
                                g_e = arith.select(_is_gate_wave, own_e, par_e)
                                u_e = arith.select(_is_gate_wave, par_e, own_e)
                                _fused_elems[_e] = _act_elem(g_e, u_e)
                            acc_gate[_ai] = vector.from_elements(
                                vec4_f32, _fused_elems
                            )

                # ---- Cross-wave gate-up fusion (4-wave, no split_k) ----
                if _gui_xwave_fuse and _split_k_intra <= 1:
                    _xw_num_accs = num_acc_n * m_repeat
                    _xw_f32_per_thr = 4 * _xw_num_accs
                    _xw_stride = arith.index(_xw_f32_per_thr)
                    _xw_total = 4 * 64 * _xw_f32_per_thr

                    xw_lds = SmemPtr(
                        base_ptr_pong, lds_pong_offset, T.f32,
                        shape=(_xw_total,),
                    ).get()

                    _xw_tx = lane_id
                    _xw_base = (
                        wave_id * arith.index(64) * _xw_stride
                        + _xw_tx * _xw_stride
                    )

                    for _ai in range_constexpr(_xw_num_accs):
                        _off = _xw_base + arith.index(_ai * 4)
                        vector.store(acc_gate[_ai], xw_lds, [_off])

                    gpu.barrier()

                    # Partner: wave_id XOR 1 → pairs (0,1) and (2,3)
                    _xw_wid_i32 = arith.index_cast(i32, wave_id)
                    _xw_pid_i32 = arith.xori(
                        _xw_wid_i32, arith.constant(1, type=T.i32),
                    )
                    _xw_partner = arith.index_cast(T.index, _xw_pid_i32)
                    _xw_pbase = (
                        _xw_partner * arith.index(64) * _xw_stride
                        + _xw_tx * _xw_stride
                    )

                    # even wave_id = gate, odd = up
                    _xw_is_gate = arith.cmpi(
                        arith.CmpIPredicate.eq,
                        arith.andi(
                            _xw_wid_i32, arith.constant(1, type=T.i32),
                        ),
                        arith.constant(0, type=T.i32),
                    )

                    for _ai in range_constexpr(_xw_num_accs):
                        _xw_off = _xw_pbase + arith.index(_ai * 4)
                        _xw_pacc = vector.load_op(
                            vec4_f32, xw_lds, [_xw_off],
                        )
                        _fused = [None] * 4
                        for _e in range_constexpr(4):
                            own_e = vector.extract(
                                acc_gate[_ai],
                                static_position=[_e], dynamic_position=[],
                            )
                            par_e = vector.extract(
                                _xw_pacc,
                                static_position=[_e], dynamic_position=[],
                            )
                            g_e = arith.select(_xw_is_gate, own_e, par_e)
                            u_e = arith.select(_xw_is_gate, par_e, own_e)
                            _fused[_e] = _act_elem(g_e, u_e)
                        acc_gate[_ai] = vector.from_elements(
                            vec4_f32, _fused,
                        )

                # ---- Bias: add to raw accumulators before activation ----
                if enable_bias and not _is_splitk:
                    _bias_gate_vals = []
                    for _ni in range_constexpr(num_acc_n):
                        _bn = by_n + n_tile_base + arith.index(_ni * 16) + lane_mod_16
                        _bias_gate_vals.append(
                            buffer_ops.buffer_load(
                                bias_rsrc, expert_off_idx + _bn,
                                vec_width=1, dtype=f32
                            )
                        )
                    for _mi in range_constexpr(m_repeat):
                        for _ni in range_constexpr(num_acc_n):
                            _aidx = _mi * num_acc_n + _ni
                            _bsplat = vector.splat(vec4_f32, _bias_gate_vals[_ni])
                            acc_gate[_aidx] = arith.addf(acc_gate[_aidx], _bsplat)

                    if not (gate_only or gate_up_interleave):
                        _bias_up_vals = []
                        for _ni in range_constexpr(num_acc_n):
                            _bn = by_n + n_tile_base + arith.index(_ni * 16) + lane_mod_16
                            _bias_up_vals.append(
                                buffer_ops.buffer_load(
                                    bias_rsrc, expert_off_idx + inter_idx + _bn,
                                    vec_width=1, dtype=f32
                                )
                            )
                        for _mi in range_constexpr(m_repeat):
                            for _ni in range_constexpr(num_acc_n):
                                _aidx = _mi * num_acc_n + _ni
                                _bsplat = vector.splat(vec4_f32, _bias_up_vals[_ni])
                                acc_up[_aidx] = arith.addf(acc_up[_aidx], _bsplat)

                # ---- Epilogue ----
                expert_off = expert_off_idx
                bx_m0 = bx_m
                tokens_i32_v = tokens_i32
                topk_i32_v = topk_i32
                inter_i32_v = arith.constant(inter_dim, type=T.i32)
                inter2_i32_v = arith.constant(inter_dim * 2, type=T.i32)
                mask24_i32 = arith.constant(0xFFFFFF, type=T.i32)

                # Fuse activation for non-split-K paths
                if gate_up_interleave and not _is_splitk and _gui_xwave_fuse:
                    acc = acc_gate
                    _eff_num_acc_n = 1
                    _eff_tile_n = _gui_out_tile_n
                elif gate_up_interleave and not _is_splitk:
                    pack_N_e = 2
                    _gui_out_n = num_acc_n // pack_N_e
                    acc = [None] * (_gui_out_n * m_repeat)
                    for _mi in range_constexpr(m_repeat):
                        for _ni in range_constexpr(_gui_out_n):
                            _g_idx = _mi * num_acc_n + _ni * pack_N_e
                            _u_idx = _g_idx + 1
                            _out_idx = _mi * _gui_out_n + _ni
                            acc[_out_idx] = _act_vec4(
                                acc_gate[_g_idx], acc_gate[_u_idx]
                            )
                    _eff_num_acc_n = _gui_out_n
                    _eff_tile_n = _gui_out_tile_n
                elif not _is_splitk:
                    acc = [None] * (num_acc_n * m_repeat)
                    for _mi in range_constexpr(m_repeat):
                        for _ni in range_constexpr(num_acc_n):
                            _aidx = _mi * num_acc_n + _ni
                            acc[_aidx] = _act_vec4(acc_gate[_aidx], acc_up[_aidx])
                    _eff_num_acc_n = num_acc_n
                    _eff_tile_n = tile_n
                else:
                    acc = acc_gate
                    _eff_num_acc_n = num_acc_n
                    _eff_tile_n = tile_n

                col_i32_list = []
                for ni in range_constexpr(len(col_g_list)):
                    col_i32_list.append(arith.index_cast(i32, col_g_list[ni]))

                # Row stride for output indexing
                if _is_splitk:
                    _out_row_stride_i32 = inter2_i32_v
                else:
                    _out_row_stride_i32 = inter_i32_v

                _sk_n_offset = [0]

                zero_i32_sk = arith.constant(0, type=T.i32)
                c4_i32_sk = arith.constant(4, type=T.i32)

                def _atomic_add_f32(val_f32, byte_off_i32):
                    rocdl.raw_ptr_buffer_atomic_fadd(
                        val_f32, out_rsrc, byte_off_i32, zero_i32_sk, zero_i32_sk,
                    )

                if _is_splitk:
                    if gate_only or gate_up_interleave:
                        # Single-pass atomic: no activation fusion
                        def _splitk_store_row(*, mi: int, ii: int, row_in_tile, row):
                            fused2 = buffer_ops.buffer_load(
                                sorted_rsrc, row, vec_width=1, dtype=i32
                            )
                            t2 = fused2 & mask24_i32
                            s2 = arith.shrui(fused2, arith.constant(24, type=T.i32))
                            row_i32 = arith.index_cast(i32, row)
                            row_ok = arith.cmpi(arith.CmpIPredicate.ult, row_i32, num_valid_i32)
                            t_ok = arith.cmpi(arith.CmpIPredicate.ult, t2, tokens_i32_v)
                            s_ok = arith.cmpi(arith.CmpIPredicate.ult, s2, topk_i32_v)
                            all_ok = arith.andi(row_ok, arith.andi(t_ok, s_ok))
                            sx = arith.select(
                                all_ok,
                                arith.constant(1.0, type=T.f32),
                                arith.constant(0.0, type=T.f32),
                            )
                            t2_safe = arith.select(all_ok, t2, arith.constant(0, type=T.i32))
                            s2_safe = arith.select(all_ok, s2, arith.constant(0, type=T.i32))
                            idx0 = (t2_safe * topk_i32_v + s2_safe) * _out_row_stride_i32

                            for ni in range_constexpr(_eff_num_acc_n):
                                col_i32 = col_i32_list[ni]
                                acc_idx = mi * _eff_num_acc_n + ni
                                val = vector.extract(
                                    acc[acc_idx],
                                    static_position=[ii],
                                    dynamic_position=[],
                                )
                                val = val * sx
                                idx_elem = idx0 + col_i32
                                byte_off = idx_elem * c4_i32_sk
                                _atomic_add_f32(val, byte_off)

                        mfma_epilog(
                            use_cshuffle=False,
                            arith=arith,
                            range_constexpr=range_constexpr,
                            m_repeat=m_repeat,
                            lane_div_16=lane_div_16,
                            bx_m=bx_m,
                            body_row=_splitk_store_row,
                        )
                    else:
                        # Separated split-K: two-pass atomic (gate then up)
                        def _splitk_sep_store_row(
                            *, mi: int, ii: int, row_in_tile, row
                        ):
                            fused2 = buffer_ops.buffer_load(
                                sorted_rsrc, row, vec_width=1, dtype=i32
                            )
                            t2 = fused2 & mask24_i32
                            s2 = arith.shrui(fused2, arith.constant(24, type=T.i32))
                            row_i32 = arith.index_cast(i32, row)
                            row_ok = arith.cmpi(arith.CmpIPredicate.ult, row_i32, num_valid_i32)
                            t_ok = arith.cmpi(arith.CmpIPredicate.ult, t2, tokens_i32_v)
                            s_ok = arith.cmpi(arith.CmpIPredicate.ult, s2, topk_i32_v)
                            all_ok = arith.andi(row_ok, arith.andi(t_ok, s_ok))
                            sx = arith.select(
                                all_ok,
                                arith.constant(1.0, type=T.f32),
                                arith.constant(0.0, type=T.f32),
                            )
                            t2_safe = arith.select(all_ok, t2, arith.constant(0, type=T.i32))
                            s2_safe = arith.select(all_ok, s2, arith.constant(0, type=T.i32))
                            idx0 = (
                                (t2_safe * topk_i32_v + s2_safe) * _out_row_stride_i32
                                + arith.constant(_sk_n_offset[0], type=T.i32)
                            )

                            for ni in range_constexpr(num_acc_n):
                                col_i32 = col_i32_list[ni]
                                acc_idx = mi * num_acc_n + ni
                                val = vector.extract(
                                    acc[acc_idx],
                                    static_position=[ii],
                                    dynamic_position=[],
                                )
                                val = val * sx
                                idx_elem = idx0 + col_i32
                                byte_off = idx_elem * c4_i32_sk
                                _atomic_add_f32(val, byte_off)

                        # Pass 1: gate (offset 0)
                        acc = acc_gate
                        _sk_n_offset[0] = 0
                        mfma_epilog(
                            use_cshuffle=False,
                            arith=arith,
                            range_constexpr=range_constexpr,
                            m_repeat=m_repeat,
                            lane_div_16=lane_div_16,
                            bx_m=bx_m,
                            body_row=_splitk_sep_store_row,
                        )
                        gpu.barrier()
                        # Pass 2: up (offset inter_dim)
                        acc = acc_up
                        _sk_n_offset[0] = inter_dim
                        mfma_epilog(
                            use_cshuffle=False,
                            arith=arith,
                            range_constexpr=range_constexpr,
                            m_repeat=m_repeat,
                            lane_div_16=lane_div_16,
                            bx_m=bx_m,
                            body_row=_splitk_sep_store_row,
                        )
                elif _use_cshuffle_epilog:
                    if lds_out is None:
                        raise RuntimeError("CShuffle requires lds_out.")

                    def write_row_to_lds(
                        *, mi: int, ii: int, row_in_tile, row,
                        row_base_lds, col_base_local, num_acc_n: int, lds_out,
                    ):
                        fused2 = buffer_ops.buffer_load(
                            sorted_rsrc, row, vec_width=1, dtype=i32
                        )
                        t2 = fused2 & mask24_i32
                        s2 = fused2 >> arith.constant(24, type=T.i32)
                        t_valid = arith.cmpi(arith.CmpIPredicate.ult, t2, tokens_i32_v)

                        if doweight_stage1:
                            tw = buffer_ops.buffer_load(
                                sorted_w_rsrc, row, vec_width=1, dtype=f32
                            )

                        _out_ty = out_mlir()
                        _vec1_out = T.vec(1, _out_ty)
                        for ni in range_constexpr(_eff_num_acc_n):
                            col_local = col_base_local + (ni * 16)
                            acc_idx = mi * _eff_num_acc_n + ni
                            y = vector.extract(
                                acc[acc_idx],
                                static_position=[ii], dynamic_position=[],
                            )
                            if doweight_stage1:
                                y = y * tw
                            y16 = arith.trunc_f(_out_ty, y)
                            lds_idx = row_base_lds + col_local
                            v1 = vector.from_elements(_vec1_out, [y16])
                            vector.store(v1, lds_out, [lds_idx], alignment=2)

                    def precompute_row(*, row_local, row):
                        fused2 = buffer_ops.buffer_load(
                            sorted_rsrc, row, vec_width=1, dtype=i32
                        )
                        row_i32 = arith.index_cast(i32, row)
                        row_valid0 = arith.cmpi(
                            arith.CmpIPredicate.ult, row_i32, num_valid_i32
                        )
                        t2 = fused2 & mask24_i32
                        s2 = fused2 >> 24
                        t_ok = arith.cmpi(arith.CmpIPredicate.ult, t2, tokens_i32_v)
                        s_ok = arith.cmpi(arith.CmpIPredicate.ult, s2, topk_i32_v)
                        row_valid = arith.andi(row_valid0, arith.andi(t_ok, s_ok))
                        row_byte_base = (t2 * topk_i32_v + s2) * inter_i32_v
                        return (row_byte_base, row_valid)

                    def store_pair(*, row_local, row, row_ctx, col_pair0, col_g0, frag):
                        fused2 = buffer_ops.buffer_load(
                            sorted_rsrc, row, vec_width=1, dtype=i32
                        )
                        t2 = fused2 & mask24_i32
                        t_valid = arith.cmpi(arith.CmpIPredicate.ult, t2, tokens_i32_v)
                        _if_valid = scf.IfOp(t_valid)
                        with _if_then(_if_valid):
                            idx0 = row_ctx
                            col_i32 = arith.index_cast(i32, col_g0)
                            idx_out = idx0 + col_i32
                            buffer_ops.buffer_store(frag, out_rsrc, idx_out)

                    _cs_by_n = by_n
                    _cs_n_tile_base = n_tile_base
                    if gate_up_interleave:
                        _cs_by_n = by_n // arith.index(2)
                        _cs_n_tile_base = n_tile_base // arith.index(2)

                    _cs_nlane = min(32, _eff_tile_n // 4)
                    mfma_epilog(
                        use_cshuffle=True,
                        arith=arith, vector=vector, gpu=gpu, scf=scf,
                        range_constexpr=range_constexpr,
                        tile_m=tile_m, tile_n=_eff_tile_n, e_vec=4,
                        cshuffle_nlane=_cs_nlane,
                        m_repeat=m_repeat, num_acc_n=_eff_num_acc_n,
                        tx=tx, lane_div_16=lane_div_16, lane_mod_16=lane_mod_16,
                        bx_m=bx_m, by_n=_cs_by_n, n_tile_base=_cs_n_tile_base,
                        lds_out=lds_out,
                        frag_elem_type=out_mlir(),
                        write_row_to_lds=write_row_to_lds,
                        precompute_row=precompute_row,
                        store_pair=store_pair,
                    )
                else:
                    # Direct epilogue (non-split-K, non-cshuffle)
                    def _stage1_store_row(*, mi: int, ii: int, row_in_tile, row):
                        fused2 = buffer_ops.buffer_load(
                            sorted_rsrc, row, vec_width=1, dtype=i32
                        )
                        t2 = fused2 & mask24_i32
                        s2 = fused2 >> 24
                        row_i32 = arith.index_cast(i32, row)
                        row_valid0 = arith.cmpi(arith.CmpIPredicate.ult, row_i32, num_valid_i32)
                        t_ok = arith.cmpi(arith.CmpIPredicate.ult, t2, tokens_i32_v)
                        s_ok = arith.cmpi(arith.CmpIPredicate.ult, s2, topk_i32_v)
                        row_valid = arith.andi(row_valid0, arith.andi(t_ok, s_ok))

                        idx0 = (t2 * topk_i32_v + s2) * inter_i32_v

                        if doweight_stage1:
                            tw = buffer_ops.buffer_load(
                                sorted_w_rsrc, row, vec_width=1, dtype=f32
                            )

                        _if_valid = scf.IfOp(row_valid)
                        with _if_then(_if_valid):
                            for ni in range_constexpr(_eff_num_acc_n):
                                col_i32 = col_i32_list[ni]
                                acc_idx = mi * _eff_num_acc_n + ni
                                y = vector.extract(
                                    acc[acc_idx],
                                    static_position=[ii], dynamic_position=[],
                                )
                                if doweight_stage1:
                                    y = y * tw
                                y = arith.trunc_f(out_mlir(), y)
                                idx_out0 = idx0 + col_i32
                                buffer_ops.buffer_store(y, out_rsrc, idx_out0)

                    mfma_epilog(
                        use_cshuffle=False,
                        arith=arith,
                        range_constexpr=range_constexpr,
                        m_repeat=m_repeat,
                        lane_div_16=lane_div_16,
                        bx_m=bx_m,
                        body_row=_stage1_store_row,
                    )

    _cache_tag = (
        module_name, out_dtype, tile_m, tile_n, tile_k,
        doweight_stage1, _use_cshuffle_epilog, act,
        enable_bias, model_dim_pad, inter_dim_pad,
        gate_only, gate_up_interleave, k_batch,
        waves_per_eu, _split_k_intra,
    )

    @flyc.jit
    def launch_moe_gemm1(
        arg_out: fx.Tensor,
        arg_x: fx.Tensor,
        arg_w: fx.Tensor,
        arg_scale_w: fx.Tensor,
        arg_sorted_token_ids: fx.Tensor,
        arg_expert_ids: fx.Tensor,
        arg_sorted_weights: fx.Tensor,
        arg_num_valid_ids: fx.Tensor,
        arg_bias: fx.Tensor,
        i32_tokens_in: fx.Int32,
        i32_inter_in: fx.Int32,
        i32_k_in: fx.Int32,
        i32_size_expert_ids_in: fx.Int32,
        stream: fx.Stream,
    ):
        _ = _cache_tag
        allocator_pong.finalized = False
        allocator_ping.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator_pong.finalize()
            allocator_ping.finalize()

        inter_in = arith.index_cast(T.index, i32_inter_in.ir_value())
        tile_n_index = arith.index(tile_n)
        inter_dim_pad_total = arith.index(2 * inter_dim_pad)
        if gate_only or gate_up_interleave:
            gx = (inter_in - inter_dim_pad_total + tile_n_index - arith.index(1)) // tile_n_index
        else:
            _two_tn = arith.index(2 * tile_n)
            gx = (inter_in - inter_dim_pad_total + _two_tn - arith.index(1)) // _two_tn

        size_expert_ids_in = arith.index_cast(
            T.index, i32_size_expert_ids_in.ir_value()
        )
        grid = (gx, size_expert_ids_in, k_batch)

        moe_gemm1(
            arg_out, arg_x, arg_w, arg_scale_w,
            arg_sorted_token_ids, arg_expert_ids, arg_sorted_weights,
            arg_num_valid_ids, arg_bias,
            i32_tokens_in, i32_inter_in, i32_k_in, i32_size_expert_ids_in,
        ).launch(
            grid=grid,
            block=(256, 1, 1),
            stream=stream,
        )

    return launch_moe_gemm1
