"""MoE GEMM stage1/stage2 kernel implementations (FLIR MFMA FP8/FP16).

This module intentionally contains the **kernel builder code** for:
- `moe_gemm1` (stage1)
- `moe_gemm2` (stage2)

It is extracted from `tests/kernels/test_moe_gemm.py` so that:
- `kernels/` holds the implementation
- `tests/` holds correctness/perf harnesses
"""

import os
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
    buffer_copy_gmem16_dwordx4,
    lds_store_16b_xor16,
    lds_store_8b_xor16,
    lds_store_4b_xor16,
    make_preshuffle_b_layout,
    make_preshuffle_scale_layout,
    tile_chunk_coord_i32,
    swizzle_xor16,
)
from .mfma_epilogues import c_shuffle_epilog
from .layout_utils import crd2idx, idx2crd, get as layout_get

import functools


@contextmanager
def _if_then(if_op):
    """Compat helper for SCF IfOp then-region across old/new Python APIs."""
    with ir.InsertionPoint(if_op.then_block):
        try:
            yield if_op.then_block
        finally:
            blk = if_op.then_block
            if (not blk.operations) or not isinstance(blk.operations[-1], scf.YieldOp):
                scf.YieldOp([])


@functools.lru_cache(maxsize=None)
def compile_mixed_moe_gemm1(
    *,
    model_dim: int,
    inter_dim: int,
    experts: int,
    topk: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    doweight_stage1: bool,
    a_dtype: str = "fp8",
    b_dtype: str = "fp4",
    out_dtype: str = "f16",
    act: str = "silu",
    use_cshuffle_epilog: bool | None = None,
    enable_bias: bool = False,
    model_dim_pad: int = 0,
    inter_dim_pad: int = 0,
    persist_m: int = 1,
):
    """Compile stage1 kernel (gate+up with silu) based on stage2 structure.

    GEMM: silu(X @ W_gate.T) * (X @ W_up.T) -> [tokens*topk, inter_dim]
    Direct store (no atomic).
    Note: persist_m=1 (no persistence) is optimal for stage1 because K=model_dim
    is large, so each CTA is already compute-heavy. persist_m>1 serializes M blocks
    that the GPU can process in parallel.
    """
    gpu_arch = get_hip_arch()
    allocator = SmemAllocator(None, arch=gpu_arch)
    _state = {}

    if a_dtype not in ("fp8", "fp16", "int8", "fp4"):
        raise ValueError(
            f"a_dtype must be one of ('fp8','fp16','int8','fp4'), got {a_dtype!r}"
        )
    if b_dtype not in ("fp8", "fp16", "int8", "int4", "fp4"):
        raise ValueError(
            f"b_dtype must be one of ('fp8','fp16','int8','int4','fp4'), got {b_dtype!r}"
        )

    is_f16_a = a_dtype == "fp16"
    is_f16_b = b_dtype == "fp16"
    is_f16 = is_f16_a or is_f16_b
    is_f8_a = a_dtype == "fp8"
    is_f4_a = a_dtype == "fp4"
    is_f4_b = b_dtype == "fp4"

    pack_M = 2
    pack_N = 2
    pack_K = 2
    elem_bytes = 1
    a_elem_bytes = 2 if is_f16_a else 1
    b_elem_bytes = 1
    tile_k_bytes = int(tile_k) * int(a_elem_bytes)
    a_elem_vec_pack = 2 if is_f4_a else 1
    cbsz = 0 if is_f8_a else 4
    blgp = 4

    if (tile_k_bytes % 64) != 0:
        raise ValueError(f"tile_k_bytes must be divisible by 64, got {tile_k_bytes}")

    out_s = str(out_dtype).strip().lower()
    out_is_f32 = out_s in ("f32", "fp32", "float")
    out_is_bf16 = out_s in ("bf16", "bfloat16")
    is_int4 = b_dtype == "int4"
    is_int8 = False

    def _x_elem_type():
        if is_f4_b:
            return T.f8 if is_f8_a else T.i8
        return T.f16 if is_f16_a else (T.i8 if is_int8 else T.f8)

    def _w_elem_type():
        if is_f4_b:
            return T.i8
        return T.f16 if is_f16_b else (T.i8 if is_int8 else T.f8)

    def out_elem():
        return T.f32 if out_is_f32 else (T.bf16 if out_is_bf16 else T.f16)

    # Stage1 gate-only: output = [tokens*topk, inter_dim], direct store (accumulate=False)
    # Weight layout: [E * 2*inter_dim, model_dim] pre-shuffled; gate = first inter_dim rows per expert
    # GEMM: X[tokens, model_dim] @ W_gate[inter_dim, model_dim].T -> [tokens*topk, inter_dim]
    accumulate = False

    DYN = ir.ShapedType.get_dynamic_size()
    size_w = experts * (2 * inter_dim) * model_dim
    total_threads = 256
    bytes_x_per_tile = int(tile_m) * int(tile_k) * int(a_elem_bytes)
    if bytes_x_per_tile % total_threads != 0:
        raise ValueError(
            f"tile_m*tile_k*elem_bytes must be divisible by {total_threads}"
        )
    bytes_per_thread_x = bytes_x_per_tile // total_threads

    _use_lds128 = os.environ.get("FLIR_CK_LDS128", "1") in (
        "1",
        "true",
        "True",
        "YES",
        "yes",
    )
    pad_k = 0 if _use_lds128 else 8
    lds_stride = tile_k + pad_k

    if use_cshuffle_epilog is None:
        _use_cshuffle_epilog = os.environ.get("FLIR_MOE_STAGE1_CSHUFFLE", "1") in (
            "1",
            "true",
            "True",
            "YES",
            "yes",
        )
    else:
        _use_cshuffle_epilog = bool(use_cshuffle_epilog)

    module_name = (
        f"mfma_moe1_silu_mul_a{a_dtype}_w{b_dtype}_{out_s}"
        f"_t{tile_m}x{tile_n}x{tile_k}_pm{persist_m}_v15"
    ).replace("-", "_")

    # -- LDS sizing --
    lds_x_bytes = 2 * int(tile_m) * int(lds_stride) * int(a_elem_bytes)
    lds_out_bytes = 2 * int(tile_m) * int(tile_n) if _use_cshuffle_epilog else 0
    lds_tid_bytes = int(tile_m) * 4
    lds_total_bytes = max(lds_x_bytes, lds_out_bytes) + lds_tid_bytes
    lds_total_elems = lds_total_bytes if a_elem_bytes == 1 else (lds_total_bytes // 2)

    def x_lds_elem():
        return T.f16 if is_f16_a else (T.i8 if is_int8 else T.f8)

    lds_alloc_bytes = int(lds_total_elems) * int(a_elem_bytes)
    lds_alloc_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_alloc_offset + lds_alloc_bytes

    kpack_bytes = 8 if is_int4 else 16
    out_elem_bytes = 4 if out_is_f32 else 2

    if True:

        @flyc.kernel
        def moe_gemm1(
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

            x_elem = T.f16 if is_f16_a else (T.i8 if is_int8 else T.f8)
            w_elem = T.f16 if is_f16_b else (T.i8 if is_int8 else T.f8)
            f16 = T.f16
            f32 = T.f32
            i32 = T.i32
            i64 = T.i64
            vec4_f32 = T.vec(4, f32)
            vec4_i32 = T.vec(4, i32)
            vec16_elems = 16 if a_elem_bytes == 1 else 8
            vec8_elems = 8 if a_elem_bytes == 1 else 4
            vec4_elems = 4 if a_elem_bytes == 1 else 2
            vec16_x = T.vec(vec16_elems, x_elem)
            vec8_x = T.vec(vec8_elems, x_elem)
            vec1_i64 = T.vec(1, i64)
            vec2_i64 = T.vec(2, i64)

            acc_init = arith.constant_vector(0.0, vec4_f32)

            # --- Stage1 dimension mapping ---
            # X: [tokens, model_dim] -- M = sorted tokens, K = model_dim
            # W: [E*2*inter_dim, model_dim] gate portion -- N = inter_dim
            # Out: [tokens*topk, inter_dim]
            topk_idx = arith.constant(topk, index=True)
            # X layout: row = token_id, col = model_dim
            tokens_i32_v = i32_tokens_in.ir_value()
            k_i32_v = i32_k_in.ir_value()

            # B preshuffle layout: [E*2*inter_dim, model_dim]
            # Gate rows for expert e: [e*2*inter_dim, e*2*inter_dim + inter_dim)
            c_n_total = arith.constant(experts * (2 * inter_dim), index=True)
            b_layout = make_preshuffle_b_layout(
                arith,
                c_n=c_n_total,
                c_k=k_in // pack_K,
                kpack_bytes=kpack_bytes,
                elem_bytes=b_elem_bytes,
            )
            layout_b = b_layout.layout_b

            # A-scale: [sorted_size, K/32] -- pre-scattered by caller into sorted layout
            # Same as stage2: indexed by sorted_row position, not by token_id.
            sorted_m = size_expert_ids_in * arith.constant(tile_m, index=True)
            layout_a_scale = make_preshuffle_scale_layout(
                arith, c_mn=sorted_m, c_k=arith.constant(model_dim, index=True)
            )
            # B-scale: [E*2*inter_dim, K/32]
            layout_b_scale = make_preshuffle_scale_layout(
                arith, c_mn=c_n_total, c_k=arith.constant(model_dim, index=True)
            )

            shape_lds = fx.make_shape(tile_m, tile_k)
            stride_lds = fx.make_stride(lds_stride, 1)
            layout_lds = fx.make_layout(shape_lds, stride_lds)

            tx = gpu.thread_id("x")
            by = gpu.block_id("x")  # tile along inter_dim (N)
            bx_persist = gpu.block_id("y")  # persistent WG index
            by_n = by * arith.constant(tile_n, index=True)

            k_blocks16 = arith.constant(tile_k_bytes // 16, index=True)
            layout_tx_wave_lane = fx.make_layout((4, 64), stride=(64, 1))
            layout_lane16 = fx.make_layout((4, 16), stride=(16, 1))

            base_ptr = allocator.get_base()
            lds_x_ptr = SmemPtr(
                base_ptr, lds_alloc_offset, x_lds_elem(), shape=(lds_total_elems,)
            )
            lds_x = lds_x_ptr.get()
            lds_out = (
                SmemPtr(
                    base_ptr,
                    lds_x_ptr.byte_offset,
                    (T.bf16 if out_is_bf16 else T.f16),
                    shape=(tile_m * tile_n,),
                ).get()
                if _use_cshuffle_epilog
                else None
            )
            _lds_x_b = 2 * int(tile_m) * int(lds_stride) * int(a_elem_bytes)
            _lds_out_b = 2 * int(tile_m) * int(tile_n) if _use_cshuffle_epilog else 0
            _lds_tid_off = max(_lds_x_b, _lds_out_b)
            lds_tid = SmemPtr(
                base_ptr, lds_x_ptr.byte_offset + _lds_tid_off, T.i32, shape=(tile_m,)
            ).get()

            # Buffer resources
            c_a_pack = arith.constant(int(a_elem_vec_pack), index=True)
            c_elem_bytes = arith.constant(int(a_elem_bytes), index=True)

            # X: [tokens, model_dim]
            x_nbytes_idx = (tokens_in * k_in * c_elem_bytes) / c_a_pack
            x_nbytes_i32 = arith.index_cast(T.i32, x_nbytes_idx)
            x_rsrc = buffer_ops.create_buffer_resource(
                arg_x, max_size=False, num_records_bytes=x_nbytes_i32
            )

            w_rsrc = buffer_ops.create_buffer_resource(arg_w, max_size=False)

            # Out: [tokens*topk, inter_dim]
            out_nbytes_idx = (
                tokens_in
                * arith.index(topk)
                * n_in
                * arith.constant(out_elem_bytes, index=True)
            )
            out_nbytes_i32 = arith.index_cast(T.i32, out_nbytes_idx)
            out_rsrc = buffer_ops.create_buffer_resource(
                arg_out, max_size=False, num_records_bytes=out_nbytes_i32
            )

            numids_rsrc = buffer_ops.create_buffer_resource(
                arg_num_valid_ids,
                max_size=False,
                num_records_bytes=arith.constant(4, type=T.i32),
            )
            num_valid_i32 = buffer_ops.buffer_load(
                numids_rsrc, arith.constant(0, index=True), vec_width=1, dtype=T.i32
            )
            num_valid_idx = arith.index_cast(ir.IndexType.get(), num_valid_i32)

            if is_f16_a:
                sx_rsrc = None
            else:
                # A scale: [sorted_size, model_dim/32] pre-scattered by caller
                c32 = arith.constant(32, index=True)
                kblk = k_in / c32
                sx_nbytes_idx = sorted_m * kblk
                sx_nbytes_i32 = arith.index_cast(T.i32, sx_nbytes_idx)
                sx_rsrc = buffer_ops.create_buffer_resource(
                    arg_scale_x, max_size=False, num_records_bytes=sx_nbytes_i32
                )

            if is_f16_b:
                sw_rsrc = None
            else:
                c32 = arith.constant(32, index=True)
                kblk_w = k_in / c32
                mn_w = arith.constant(experts * (2 * inter_dim), index=True)
                sw_nbytes_idx = mn_w * kblk_w
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

            eid_nbytes_idx = size_expert_ids_in * arith.constant(4, index=True)
            eid_nbytes_i32 = arith.index_cast(T.i32, eid_nbytes_idx)
            expert_rsrc = buffer_ops.create_buffer_resource(
                arg_expert_ids, max_size=False, num_records_bytes=eid_nbytes_i32
            )

            # ---- persist_m loop (same pattern as stage2) ----
            _PERSIST_M = persist_m
            _c0_p = arith.constant(0, index=True)
            _c1_p = arith.constant(1, index=True)
            _c_pm = arith.constant(_PERSIST_M, index=True)
            _for_persist = scf.ForOp(_c0_p, _c_pm, _c1_p)
            _for_ip = ir.InsertionPoint(_for_persist.body)
            _for_ip.__enter__()
            _mi_p = _for_persist.induction_variable
            bx = bx_persist * _c_pm + _mi_p
            bx_m = bx * arith.constant(tile_m, index=True)

            # Block validity
            bx_m_i32 = arith.index_cast(T.i32, bx_m)
            blk_valid = arith.cmpi(CmpIPredicate.ult, bx_m_i32, num_valid_i32)
            expert_i32 = buffer_ops.buffer_load(
                expert_rsrc, bx, vec_width=1, dtype=T.i32
            )
            expert_idx = arith.index_cast(ir.IndexType.get(), expert_i32)
            exp_valid = arith.cmpi(
                CmpIPredicate.ult, expert_i32, arith.constant(experts, type=T.i32)
            )

            def _moe_gemm1_body():
                # Gate expert offset: first inter_dim rows of each expert's 2*inter_dim block
                expert_off_idx = expert_idx * arith.constant(2 * inter_dim, index=True)

                # X loading -- KEY DIFFERENCE from stage2: X row = token_id only
                x_load_bytes = 16
                num_x_loads = bytes_per_thread_x // x_load_bytes
                chunk_i32 = x_load_bytes // 4
                vec4_i32_t = T.vec(4, i32)

                c_k_div4 = (
                    (k_in / c_a_pack) * arith.constant(int(a_elem_bytes), index=True)
                ) / arith.index(4)
                c_k_div4_i32 = arith.index_cast(T.i32, c_k_div4)
                tile_k_dwords = (int(tile_k) * int(a_elem_bytes)) // (
                    4 * int(a_elem_vec_pack)
                )
                layout_x_tile_div4 = fx.make_layout(
                    (tile_m, tile_k_dwords), stride=(tile_k_dwords, 1)
                )
                c_chunk_i32 = arith.constant(chunk_i32, index=True)
                tx_i32_base = tx * c_chunk_i32

                topk_i32 = arith.constant(topk)
                mask24 = arith.constant(0xFFFFFF)
                tokens_i32 = arith.index_cast(T.i32, tokens_in)

                def x_tile_chunk_coord_i32(i: int):
                    return tile_chunk_coord_i32(
                        arith,
                        tx_i32_base=tx_i32_base,
                        i=i,
                        total_threads=total_threads,
                        layout_tile_div4=layout_x_tile_div4,
                        chunk_i32=chunk_i32,
                    )

                def load_x(idx_i32):
                    idx_elem = (
                        idx_i32 if a_elem_bytes == 1 else (idx_i32 * arith.index(2))
                    )
                    return buffer_copy_gmem16_dwordx4(
                        buffer_ops,
                        vector,
                        elem_type=x_elem,
                        idx_i32=idx_elem,
                        rsrc=x_rsrc,
                        vec_elems=vec16_elems,
                    )

                # Decode sorted token ids -- stage1: X row = token_id (not t*topk+s)
                x_row_base_div4 = []
                x_col_local_i32 = []
                x_row_local = []
                # Also store token_id and slot_id for output indexing
                x_token_idx = []
                x_slot_idx = []
                x_row_valid = []

                for i in range_constexpr(num_x_loads):
                    row_local, col_local_i32 = x_tile_chunk_coord_i32(i)
                    x_row_local.append(row_local)
                    x_col_local_i32.append(col_local_i32)

                    sorted_row_i = bx_m + row_local
                    fused_i = buffer_ops.buffer_load(
                        sorted_rsrc, sorted_row_i, vec_width=1, dtype=T.i32
                    )
                    t_i32 = arith.andi(fused_i, mask24)
                    s_i32 = arith.shrui(fused_i, arith.constant(24))
                    t_valid = arith.cmpi(CmpIPredicate.ult, t_i32, tokens_i32)
                    s_valid = arith.cmpi(CmpIPredicate.ult, s_i32, topk_i32)
                    ts_valid = arith.andi(t_valid, s_valid)
                    t_safe = arith.select(ts_valid, t_i32, arith.constant(0))

                    # KEY: X row base uses token_id only (not t*topk+s)
                    t_idx = arith.index_cast(ir.IndexType.get(), t_safe)
                    x_row_base_div4.append(t_idx * c_k_div4)

                def load_x_tile(base_k):
                    base_k_div4 = (
                        (base_k / c_a_pack)
                        * arith.constant(int(a_elem_bytes), index=True)
                    ) / arith.index(4)
                    parts = []
                    for i in range_constexpr(num_x_loads):
                        idx_i32 = x_row_base_div4[i] + base_k_div4 + x_col_local_i32[i]
                        x_vec = load_x(idx_i32)
                        parts.append(vector.bitcast(T.vec(4, i32), x_vec))
                    return parts

                # Wave/lane decomposition (identical to stage2)
                coord_wl = idx2crd(tx, layout_tx_wave_lane)
                wave_id = layout_get(coord_wl, 0)
                lane_id = layout_get(coord_wl, 1)
                coord_l16 = idx2crd(lane_id, layout_lane16)
                lane_div_16 = layout_get(coord_l16, 0)
                lane_mod_16 = layout_get(coord_l16, 1)
                row_a_lds = lane_mod_16
                col_offset_base = lane_div_16 * arith.constant(16, index=True)

                num_waves = 4
                n_per_wave = tile_n // num_waves
                num_acc_n = n_per_wave // 16
                c_n_per_wave = arith.constant(n_per_wave, index=True)
                wave_mod_4 = wave_id % arith.constant(4, index=True)
                n_tile_base = wave_mod_4 * c_n_per_wave

                # N-tile precompute for gate AND up weights
                gate_n_intra_list = []
                gate_n_blk_list = []
                up_n_intra_list = []
                up_n_blk_list = []
                col_g_list = []
                c_n0_static = experts * (2 * inter_dim) // 16
                layout_n_blk_intra = fx.make_layout((c_n0_static, 16), stride=(16, 1))
                inter_idx = arith.constant(inter_dim, index=True)

                for i in range_constexpr(num_acc_n):
                    offset = i * 16
                    c_offset = arith.constant(offset, index=True)
                    col_g = by_n + n_tile_base + c_offset + lane_mod_16
                    col_g_list.append(col_g)

                    global_n = by_n + n_tile_base + c_offset + lane_mod_16
                    # Gate: rows [expert_off, expert_off + inter_dim)
                    gate_row_w = expert_off_idx + global_n
                    gate_coord = idx2crd(gate_row_w, layout_n_blk_intra)
                    gate_n_blk_list.append(layout_get(gate_coord, 0))
                    gate_n_intra_list.append(layout_get(gate_coord, 1))
                    # Up: rows [expert_off + inter_dim, expert_off + 2*inter_dim)
                    up_row_w = gate_row_w + inter_idx
                    up_coord = idx2crd(up_row_w, layout_n_blk_intra)
                    up_n_blk_list.append(layout_get(up_coord, 0))
                    up_n_intra_list.append(layout_get(up_coord, 1))

                m_repeat = tile_m // 16
                k_unroll = tile_k_bytes // 128
                k_unroll_packed = k_unroll // pack_K
                m_repeat_packed = m_repeat // pack_M
                num_acc_n_packed = num_acc_n // pack_N

                # B load for gate and up separately
                def load_b_packs_k64(base_k, ku: int, n_blk, n_intra):
                    c64 = arith.constant(64, index=True)
                    base_k_bytes = base_k * arith.constant(
                        int(b_elem_bytes), index=True
                    )
                    k0 = base_k_bytes // c64 + arith.constant(ku, index=True)
                    k1 = lane_div_16
                    coord_pack = (n_blk, k0, k1, n_intra, arith.constant(0, index=True))
                    idx_pack = crd2idx(coord_pack, layout_b)
                    vec_elems = kpack_bytes // int(b_elem_bytes)
                    b16 = _buffer_load_vec(
                        buffer_ops,
                        vector,
                        w_rsrc,
                        idx_pack,
                        elem_type=_w_elem_type(),
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
                    """Load both gate and up B tiles. Returns (gate_b_tile, up_b_tile)."""
                    gate_b_tile = []
                    up_b_tile = []
                    for ku in range_constexpr(k_unroll):
                        g_packs0, g_packs1 = [], []
                        u_packs0, u_packs1 = [], []
                        for ni in range_constexpr(num_acc_n):
                            gb0, gb1 = load_b_packs_k64(
                                base_k, ku, gate_n_blk_list[ni], gate_n_intra_list[ni]
                            )
                            g_packs0.append(gb0)
                            g_packs1.append(gb1)
                            ub0, ub1 = load_b_packs_k64(
                                base_k, ku, up_n_blk_list[ni], up_n_intra_list[ni]
                            )
                            u_packs0.append(ub0)
                            u_packs1.append(ub1)
                        gate_b_tile.append((g_packs0, g_packs1))
                        up_b_tile.append((u_packs0, u_packs1))
                    return gate_b_tile, up_b_tile

                # Scale load (identical to stage2)
                def load_scale(arg_scale, rsrc, scale_info, ku, mni):
                    k_lane = lane_div_16
                    n_lane = lane_mod_16
                    idx_pack = (
                        mni * scale_info.stride_n0
                        + ku * scale_info.stride_k0
                        + k_lane * scale_info.stride_klane
                        + n_lane
                    )
                    s = buffer_ops.buffer_load(rsrc, idx_pack, vec_width=1, dtype=T.i32)
                    return vector.from_elements(T.vec(1, T.i32), [s])

                def load_b_scale_tile(base_k):
                    """Load gate and up B scales. Returns (gate_b_scale, up_b_scale)."""
                    gate_b_scale = []
                    up_b_scale = []
                    for ku in range_constexpr(k_unroll_packed):
                        for ni in range_constexpr(num_acc_n_packed):
                            col_offset = ni * 16 * pack_N
                            col_offset_idx = arith.constant(col_offset, index=True)
                            col_base = by_n + n_tile_base + col_offset_idx
                            gate_mni = (expert_off_idx + col_base) // arith.constant(
                                32, index=True
                            )
                            up_mni = (
                                expert_off_idx + inter_idx + col_base
                            ) // arith.constant(32, index=True)
                            gate_b_scale.append(
                                load_scale(
                                    arg_scale_w,
                                    sw_rsrc,
                                    layout_b_scale,
                                    ku + base_k,
                                    gate_mni,
                                )
                            )
                            up_b_scale.append(
                                load_scale(
                                    arg_scale_w,
                                    sw_rsrc,
                                    layout_b_scale,
                                    ku + base_k,
                                    up_mni,
                                )
                            )
                    return gate_b_scale, up_b_scale

                def load_a_scale_tile(base_k):
                    a_scale_tile = []
                    for ku in range_constexpr(k_unroll_packed):
                        for mi in range_constexpr(m_repeat_packed):
                            scale = load_scale(
                                arg_scale_x,
                                sx_rsrc,
                                layout_a_scale,
                                ku + base_k,
                                mi + bx_m // pack_M // 16,
                            )
                            a_scale_tile.append(scale)
                    return a_scale_tile

                def prefetch_ab_scale_tile(base_k):
                    gate_bs, up_bs = load_b_scale_tile(base_k)
                    return [load_a_scale_tile(base_k), gate_bs, up_bs]

                vec4_x_lds = T.vec(vec4_elems, x_elem)

                def store_x_tile_to_lds(vec_x_in_parts, lds_base):
                    for i in range_constexpr(num_x_loads):
                        row_local = x_row_local[i]
                        col_local_i32 = x_col_local_i32[i]
                        if x_load_bytes == 16:
                            lds_store_16b_xor16(
                                arith,
                                vector,
                                lds_memref=lds_x,
                                vec16_ty=vec16_x,
                                layout_lds=layout_lds,
                                row_local=row_local,
                                col_local_i32=col_local_i32,
                                tx_c4=arith.index(4),
                                k_blocks16=k_blocks16,
                                lds_base=lds_base,
                                vec_part_i32x4=vec_x_in_parts[i],
                                elem_bytes=elem_bytes,
                            )

                def lds_load_packs_k64(curr_row_a_lds, col_base, lds_base):
                    col_base_swz_bytes = swizzle_xor16(
                        curr_row_a_lds, col_base, k_blocks16
                    )
                    col_base_swz = (
                        col_base_swz_bytes
                        if elem_bytes == 1
                        else (col_base_swz_bytes / arith.index(2))
                    )
                    idx_a16 = (
                        crd2idx([curr_row_a_lds, col_base_swz], layout_lds) + lds_base
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

                # Compute tile: gate + up MFMA interleaved, same A data, different B data.
                # Two accumulator sets; after all K tiles, acc = acc_gate + acc_up (f32 add).
                def compute_tile(
                    acc_gate_in,
                    acc_up_in,
                    gate_b_tile_in,
                    up_b_tile_in,
                    lds_base,
                    a_scale=None,
                    gate_b_scale=None,
                    up_b_scale=None,
                    *,
                    prefetch_epilogue=False,
                    a0_prefetch=None,
                ):
                    gate_list = list(acc_gate_in)
                    up_list = list(acc_up_in)
                    mfma_res_ty = vec4_f32
                    epilogue_pf = None
                    if prefetch_epilogue and doweight_stage1:
                        tw_pf = []
                        lane_div_16_mul4_pf = lane_div_16 * arith.index(4)
                        ii_idx_list_pf = [
                            arith.constant(ii, index=True) for ii in range(4)
                        ]
                        for mi in range_constexpr(m_repeat):
                            mi_base_pf = arith.constant(mi * 16, index=True)
                            for ii in range_constexpr(4):
                                row_off_pf = lane_div_16_mul4_pf + ii_idx_list_pf[ii]
                                sorted_row_pf = bx_m + mi_base_pf + row_off_pf
                                tw_pf.append(
                                    buffer_ops.buffer_load(
                                        sorted_w_rsrc,
                                        sorted_row_pf,
                                        vec_width=1,
                                        dtype=f32,
                                    )
                                )
                        epilogue_pf = (None, tw_pf, None)

                    c0_i64 = arith.constant(0, type=T.i64)
                    vec4_i64 = T.vec(4, T.i64)
                    vec8_i32 = T.vec(8, T.i32)

                    def pack_i64x4_to_i32x8(x0, x1, x2, x3):
                        v4 = vector.from_elements(vec4_i64, [x0, x1, x2, x3])
                        return vector.bitcast(vec8_i32, v4)

                    for ku128 in range_constexpr(k_unroll_packed):
                        for mi in range_constexpr(m_repeat_packed):
                            a_scale_i32 = a_scale[ku128 * m_repeat_packed + mi]
                            a_scale_val = vector.extract(
                                a_scale_i32, static_position=[0], dynamic_position=[]
                            )
                            for ni in range_constexpr(num_acc_n_packed):
                                gate_bs_i32 = gate_b_scale[
                                    ku128 * num_acc_n_packed + ni
                                ]
                                gate_bs_val = vector.extract(
                                    gate_bs_i32,
                                    static_position=[0],
                                    dynamic_position=[],
                                )
                                up_bs_i32 = up_b_scale[ku128 * num_acc_n_packed + ni]
                                up_bs_val = vector.extract(
                                    up_bs_i32, static_position=[0], dynamic_position=[]
                                )
                                for ikxdl in range_constexpr(pack_K):
                                    k_idx = ku128 * pack_K + ikxdl
                                    gate_bp0, gate_bp1 = gate_b_tile_in[k_idx]
                                    up_bp0, up_bp1 = up_b_tile_in[k_idx]
                                    col_base = (
                                        col_offset_base
                                        + (k_idx * 128) // a_elem_vec_pack
                                    )
                                    for imxdl in range_constexpr(pack_M):
                                        mi_idx = mi * pack_M + imxdl
                                        mi_val = arith.constant(mi_idx * 16, index=True)
                                        curr_row_a_lds = row_a_lds + mi_val
                                        if (
                                            (a0_prefetch is not None)
                                            and (k_idx == 0)
                                            and (mi_idx == 0)
                                        ):
                                            a0, a1 = a0_prefetch
                                        else:
                                            a0, a1 = lds_load_packs_k64(
                                                curr_row_a_lds, col_base, lds_base
                                            )
                                        if is_f8_a:
                                            a2, a3 = lds_load_packs_k64(
                                                curr_row_a_lds, col_base + 64, lds_base
                                            )
                                            a128 = pack_i64x4_to_i32x8(a0, a1, a2, a3)
                                        else:
                                            a128 = pack_i64x4_to_i32x8(
                                                a0, a1, c0_i64, c0_i64
                                            )
                                        for inxdl in range_constexpr(pack_N):
                                            ni_idx = ni * pack_N + inxdl
                                            acc_idx = mi_idx * num_acc_n + ni_idx
                                            # Gate MFMA
                                            gb0 = gate_bp0[ni_idx]
                                            gb1 = gate_bp1[ni_idx]
                                            gb128 = pack_i64x4_to_i32x8(
                                                gb0, gb1, c0_i64, c0_i64
                                            )
                                            rocdl.sched_barrier(0)
                                            gate_list[acc_idx] = (
                                                rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                                                    mfma_res_ty,
                                                    [
                                                        a128,
                                                        gb128,
                                                        gate_list[acc_idx],
                                                        cbsz,
                                                        blgp,
                                                        ikxdl * pack_M + imxdl,
                                                        a_scale_val,
                                                        ikxdl * pack_N + inxdl,
                                                        gate_bs_val,
                                                    ],
                                                )
                                            )
                                            # Up MFMA
                                            ub0 = up_bp0[ni_idx]
                                            ub1 = up_bp1[ni_idx]
                                            ub128 = pack_i64x4_to_i32x8(
                                                ub0, ub1, c0_i64, c0_i64
                                            )
                                            rocdl.sched_barrier(0)
                                            up_list[acc_idx] = (
                                                rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                                                    mfma_res_ty,
                                                    [
                                                        a128,
                                                        ub128,
                                                        up_list[acc_idx],
                                                        cbsz,
                                                        blgp,
                                                        ikxdl * pack_M + imxdl,
                                                        a_scale_val,
                                                        ikxdl * pack_N + inxdl,
                                                        up_bs_val,
                                                    ],
                                                )
                                            )
                    return gate_list, up_list, epilogue_pf

                # Pipeline (identical to stage2 structure)
                lds_tile_elems = arith.constant(tile_m * lds_stride, index=True)
                lds_base_cur = arith.index(0)
                lds_base_nxt = lds_tile_elems
                rocdl.sched_barrier(0)

                k0 = arith.index(0)
                gate_w0, up_w0 = load_b_tile(k0)
                a_scale_pong, gate_bs_pong, up_bs_pong = prefetch_ab_scale_tile(
                    k0 // pack_K // 128
                )
                x_regs0 = load_x_tile(k0)
                store_x_tile_to_lds(x_regs0, lds_base_cur)
                _c_tile_m_idx = arith.constant(tile_m, index=True)
                _tid_in_range = arith.cmpi(CmpIPredicate.ult, tx, _c_tile_m_idx)
                _if_tid = scf.IfOp(_tid_in_range)
                with ir.InsertionPoint(_if_tid.then_block):
                    _tid_row = bx_m + tx
                    _tid_val = buffer_ops.buffer_load(
                        sorted_rsrc, _tid_row, vec_width=1, dtype=T.i32
                    )
                    _tid_vec1 = vector.from_elements(T.vec(1, T.i32), [_tid_val])
                    vector.store(_tid_vec1, lds_tid, [tx])
                    scf.YieldOp([])
                gpu.barrier()

                acc_gate = [acc_init] * num_acc_n * m_repeat
                acc_up = [acc_init] * num_acc_n * m_repeat
                lds_base_pong = lds_base_cur
                lds_base_ping = lds_base_nxt
                a0_prefetch_pong = lds_load_packs_k64(
                    row_a_lds, col_offset_base, lds_base_pong
                )

                num_k_tiles_py = int(model_dim) // int(tile_k)
                odd_k_tiles = (num_k_tiles_py % 2) == 1
                tail_tiles = 1 if odd_k_tiles else 2
                k_main2_py = (num_k_tiles_py - tail_tiles) * int(tile_k)
                if k_main2_py < 0:
                    k_main2_py = 0

                c2_tile_k = arith.constant(tile_k * 2, index=True)
                gate_w_pong = gate_w0
                up_w_pong = up_w0

                def _sched_hints_stage1_gate_up():
                    """Stage1 hot-loop scheduler adapted from the gate/up gufusion pipeline.

                    The original hot loop doubles the B-side VMEM and MFMA streams:
                    - gate B load + up B load
                    - gate B-scale load + up B-scale load
                    - gate MFMA + up MFMA

                    The scheduler API here is less expressive than the original
                    `__builtin_amdgcn_sched_group_barrier`, so we encode the same
                    idea with a compact heuristic:
                    - always double MFMA groups (`num_acc_n * 2`)
                    - use 2 VMEM groups only when the N tile is wide enough to
                      sustain the extra B-side traffic (`num_acc_n >= 4`)
                    - otherwise keep 1 VMEM group to avoid over-throttling the
                      smaller `tile_n=128` kernels
                    """
                    mfma_group = num_acc_n * 2
                    mfma_total = (k_unroll * 2) * m_repeat * mfma_group
                    mfma_per_iter = 2 * mfma_group
                    sche_iters = (
                        0 if mfma_per_iter == 0 else (mfma_total // mfma_per_iter)
                    )

                    # Approximate the doubled B-side prefetch pressure.
                    vmem_groups = 2 if int(num_acc_n) >= 4 else 1

                    rocdl.sched_dsrd(2)
                    rocdl.sched_mfma(2)
                    rocdl.sched_dsrd(1)
                    rocdl.sched_mfma(1)
                    rocdl.sched_dsrd(1)
                    rocdl.sched_mfma(1)

                    dswr_tail = num_x_loads
                    if dswr_tail > sche_iters:
                        dswr_tail = sche_iters
                    dswr_start = sche_iters - dswr_tail

                    for sche_i in range_constexpr(sche_iters):
                        rocdl.sched_vmem(vmem_groups)
                        rocdl.sched_mfma(mfma_group)
                        rocdl.sched_dsrd(1)
                        rocdl.sched_mfma(mfma_group)
                        if sche_i >= dswr_start - 1:
                            rocdl.sched_dswr(1)
                    rocdl.sched_barrier(0)

                if k_main2_py > 0:
                    for k_iv_py in range_constexpr(0, k_main2_py, tile_k * 2):
                        k_iv = k_iv_py
                        next_k1 = k_iv + tile_k
                        x_regs_ping = load_x_tile(next_k1)
                        gate_w_ping, up_w_ping = load_b_tile(next_k1 // 2)
                        a_scale_ping, gate_bs_ping, up_bs_ping = prefetch_ab_scale_tile(
                            next_k1 // pack_K // 128
                        )
                        acc_gate, acc_up, _ = compute_tile(
                            acc_gate,
                            acc_up,
                            gate_w_pong,
                            up_w_pong,
                            lds_base_pong,
                            a_scale_pong,
                            gate_bs_pong,
                            up_bs_pong,
                            a0_prefetch=a0_prefetch_pong,
                        )
                        store_x_tile_to_lds(x_regs_ping, lds_base_ping)
                        _sched_hints_stage1_gate_up()
                        gpu.barrier()
                        a0_prefetch_ping = lds_load_packs_k64(
                            row_a_lds, col_offset_base, lds_base_ping
                        )
                        next_k2 = k_iv + c2_tile_k
                        x_regs_pong = load_x_tile(next_k2)
                        gate_w_pong, up_w_pong = load_b_tile(next_k2 // 2)
                        a_scale_pong, gate_bs_pong, up_bs_pong = prefetch_ab_scale_tile(
                            next_k2 // pack_K // 128
                        )
                        acc_gate, acc_up, _ = compute_tile(
                            acc_gate,
                            acc_up,
                            gate_w_ping,
                            up_w_ping,
                            lds_base_ping,
                            a_scale_ping,
                            gate_bs_ping,
                            up_bs_ping,
                            a0_prefetch=a0_prefetch_ping,
                        )
                        store_x_tile_to_lds(x_regs_pong, lds_base_pong)
                        _sched_hints_stage1_gate_up()
                        gpu.barrier()
                        a0_prefetch_pong = lds_load_packs_k64(
                            row_a_lds, col_offset_base, lds_base_pong
                        )

                if odd_k_tiles:
                    acc_gate, acc_up, epilogue_pf = compute_tile(
                        acc_gate,
                        acc_up,
                        gate_w_pong,
                        up_w_pong,
                        lds_base_pong,
                        a_scale_pong,
                        gate_bs_pong,
                        up_bs_pong,
                        a0_prefetch=a0_prefetch_pong,
                        prefetch_epilogue=True,
                    )
                else:
                    k_tail1 = (k_in + tile_k - 1) // tile_k * tile_k - tile_k
                    x_regs_ping = load_x_tile(k_tail1)
                    gate_w_ping, up_w_ping = load_b_tile(k_tail1 // 2)
                    a_scale_ping, gate_bs_ping, up_bs_ping = prefetch_ab_scale_tile(
                        k_tail1 // pack_K // 128
                    )
                    acc_gate, acc_up, _ = compute_tile(
                        acc_gate,
                        acc_up,
                        gate_w_pong,
                        up_w_pong,
                        lds_base_pong,
                        a_scale_pong,
                        gate_bs_pong,
                        up_bs_pong,
                        a0_prefetch=a0_prefetch_pong,
                    )
                    store_x_tile_to_lds(x_regs_ping, lds_base_ping)
                    gpu.barrier()
                    a0_prefetch_ping = lds_load_packs_k64(
                        row_a_lds, col_offset_base, lds_base_ping
                    )
                    acc_gate, acc_up, epilogue_pf = compute_tile(
                        acc_gate,
                        acc_up,
                        gate_w_ping,
                        up_w_ping,
                        lds_base_ping,
                        a_scale_ping,
                        gate_bs_ping,
                        up_bs_ping,
                        a0_prefetch=a0_prefetch_ping,
                        prefetch_epilogue=True,
                    )

                # silu(gate) * up in f32 before epilogue
                # silu(x) = x * sigmoid(x); use HW fast path: exp2, rcp
                def _silu_mul_vec4(gate_v4, up_v4):
                    """Element-wise silu(gate) * up on vec4_f32."""
                    result_elems = []
                    for ei in range_constexpr(4):
                        g = vector.extract(
                            gate_v4, static_position=[ei], dynamic_position=[]
                        )
                        u = vector.extract(
                            up_v4, static_position=[ei], dynamic_position=[]
                        )
                        neg_log2e = arith.constant(-1.4426950408889634, type=f32)
                        t = g * neg_log2e
                        emu = llvm.call_intrinsic(
                            f32, "llvm.amdgcn.exp2.f32", [t], [], []
                        )
                        one = arith.constant(1.0, type=f32)
                        den = one + emu
                        sig = llvm.call_intrinsic(
                            f32, "llvm.amdgcn.rcp.f32", [den], [], []
                        )
                        result_elems.append(g * sig * u)
                    return vector.from_elements(vec4_f32, result_elems)

                acc = [None] * (int(num_acc_n) * int(m_repeat))
                for _mi in range_constexpr(m_repeat):
                    for _ni in range_constexpr(num_acc_n):
                        _aidx = _mi * num_acc_n + _ni
                        acc[_aidx] = _silu_mul_vec4(acc_gate[_aidx], acc_up[_aidx])

                # ---- Epilogue: CShuffle + direct store (accumulate=False) ----
                # Output: out[(t*topk+s) * inter_dim + col] = silu(gate) * up
                tw_pf = None
                if epilogue_pf is not None:
                    _, tw_pf, _ = epilogue_pf

                mask24_i32 = arith.constant(0xFFFFFF)
                topk_i32_v = topk_i32
                tokens_i32_v = tokens_i32

                from flydsl._mlir.dialects import fly as _fly

                _llvm_ptr_ty = ir.Type.parse("!llvm.ptr")
                out_base_ptr = _fly.extract_aligned_pointer_as_index(
                    _llvm_ptr_ty, arg_out
                )
                out_base_i64 = llvm.ptrtoint(T.i64, out_base_ptr)
                out_base_idx = arith.index_cast(ir.IndexType.get(), out_base_i64)

                if lds_out is None:
                    raise RuntimeError("CShuffle epilogue requires lds_out")

                def write_row_to_lds(
                    *,
                    mi: int,
                    ii: int,
                    row_in_tile,
                    row,
                    row_base_lds,
                    col_base_local,
                    num_acc_n: int,
                    lds_out,
                ):
                    if doweight_stage1:
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
                        if doweight_stage1:
                            v = v * tw
                        v_out = arith.trunc_f(out_elem(), v)
                        lds_idx = row_base_lds + col_local
                        vec1_out = T.vec(1, out_elem())
                        v1 = vector.from_elements(vec1_out, [v_out])
                        vector.store(v1, lds_out, [lds_idx], alignment=2)

                def precompute_row(*, row_local, row):
                    fused2 = memref.load(lds_tid, [row_local])
                    row_i32 = arith.index_cast(T.i32, row)
                    row_valid0 = arith.cmpi(CmpIPredicate.ult, row_i32, num_valid_i32)
                    t = fused2 & mask24_i32
                    s = fused2 >> 24
                    t_ok = arith.cmpi(CmpIPredicate.ult, t, tokens_i32_v)
                    s_ok = arith.cmpi(CmpIPredicate.ult, s, topk_i32_v)
                    row_valid = arith.andi(row_valid0, arith.andi(t_ok, s_ok))
                    t_idx = arith.index_cast(ir.IndexType.get(), t)
                    s_idx = arith.index_cast(ir.IndexType.get(), s)
                    ts_idx = t_idx * arith.constant(topk, index=True) + s_idx
                    # Direct store: out base = (t*topk+s) * inter_dim * elem_bytes
                    row_byte_base = out_base_idx + ts_idx * arith.constant(
                        inter_dim * out_elem_bytes, index=True
                    )
                    return ((fused2, row_byte_base), row_valid)

                def _idx_to_llvm_ptr(idx_val, addr_space=1):
                    idx_v = idx_val._value if hasattr(idx_val, "_value") else idx_val
                    i64_v = arith.index_cast(T.i64, idx_v)
                    i64_raw = i64_v._value if hasattr(i64_v, "_value") else i64_v
                    ptr_ty = ir.Type.parse(f"!llvm.ptr<{addr_space}>")
                    return llvm.inttoptr(ptr_ty, i64_raw)

                _e_vec = min(tile_n // 32, 8)

                def store_pair(*, row_local, row, row_ctx, col_pair0, col_g0, frag):
                    fused, row_byte_base = row_ctx
                    col_idx = col_g0
                    byte_off_col = col_idx * arith.constant(out_elem_bytes, index=True)
                    ptr_addr_idx = row_byte_base + byte_off_col
                    out_ptr_v = _idx_to_llvm_ptr(ptr_addr_idx)
                    frag_v = frag._value if hasattr(frag, "_value") else frag
                    llvm.StoreOp(
                        frag_v,
                        out_ptr_v,
                        alignment=_e_vec * out_elem_bytes,
                        nontemporal=True,
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
                    m_repeat=m_repeat,
                    num_acc_n=num_acc_n,
                    tx=tx,
                    lane_div_16=lane_div_16,
                    lane_mod_16=lane_mod_16,
                    bx_m=bx_m,
                    by_n=by_n,
                    n_tile_base=n_tile_base,
                    lds_out=lds_out,
                    frag_elem_type=(
                        ir.BF16Type.get() if out_is_bf16 else ir.F16Type.get()
                    ),
                    write_row_to_lds=write_row_to_lds,
                    precompute_row=precompute_row,
                    store_pair=store_pair,
                )

            _if_blk = scf.IfOp(blk_valid)
            with ir.InsertionPoint(_if_blk.then_block):
                _ifexpert_of = scf.IfOp(exp_valid)
                with ir.InsertionPoint(_ifexpert_of.then_block):
                    _moe_gemm1_body()
                    scf.YieldOp([])
                scf.YieldOp([])

            gpu.barrier()
            scf.YieldOp([])
            _for_ip.__exit__(None, None, None)

    # -- Host launcher --
    _cache_tag = (
        module_name,
        a_dtype,
        b_dtype,
        out_dtype,
        tile_m,
        tile_n,
        tile_k,
        doweight_stage1,
        act,
        enable_bias,
        model_dim_pad,
        inter_dim_pad,
        use_cshuffle_epilog,
        persist_m,
    )

    @flyc.jit
    def launch_mixed_moe_gemm1(
        arg_out: fx.Tensor,
        arg_x: fx.Tensor,
        arg_w: fx.Tensor,
        arg_scale_x: fx.Tensor,
        arg_scale_w: fx.Tensor,
        arg_sorted_token_ids: fx.Tensor,
        arg_expert_ids: fx.Tensor,
        arg_sorted_weights: fx.Tensor,
        arg_max_token_ids: fx.Tensor,
        arg_bias: fx.Tensor,
        i32_tokens_in: fx.Int32,
        i32_inter_in: fx.Int32,
        i32_k_in: fx.Int32,
        i32_size_expert_ids_in: fx.Int32,
        stream: fx.Stream,
    ):
        _ = _cache_tag
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()

        inter_in = arith.index_cast(ir.IndexType.get(), i32_inter_in.ir_value())
        gx = (
            inter_in
            / arith.constant(2, index=True)
            / arith.constant(tile_n, index=True)
        )
        # gy = size_expert_ids_in (caller passes valid block count, not full sorted_expert_ids.shape[0])
        _c_pm_l = arith.constant(persist_m, index=True)
        gy = (
            arith.index_cast(ir.IndexType.get(), i32_size_expert_ids_in.ir_value())
            + _c_pm_l
            - arith.constant(1, index=True)
        ) / _c_pm_l

        moe_gemm1(
            arg_out,
            arg_x,
            arg_w,
            arg_scale_x,
            arg_scale_w,
            arg_sorted_token_ids,
            arg_expert_ids,
            arg_sorted_weights,
            arg_max_token_ids,
            arg_bias,
            i32_tokens_in,
            i32_inter_in,
            i32_k_in,
            i32_size_expert_ids_in,
        ).launch(grid=(gx, gy, 1), block=(256, 1, 1), stream=stream)

    return launch_mixed_moe_gemm1


@functools.lru_cache(maxsize=None)
def compile_mixed_moe_gemm2(
    *,
    model_dim: int,
    inter_dim: int,
    experts: int,
    topk: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    doweight_stage2: bool,
    a_dtype: str = "fp8",
    b_dtype: str = "fp4",
    out_dtype: str = "f16",
    use_cshuffle_epilog: bool | None = None,
    # Optional experiment: write per-(token,slot) output (no atomics) into an output shaped
    # [tokens*topk, model_dim] (or [tokens, topk, model_dim] flattened), then reduce over topk outside.
    # This can reduce atomic contention for small tokens at the cost of extra bandwidth / reduction.
    accumulate: bool = True,
    enable_bias: bool = False,
    model_dim_pad: int = 0,
    inter_dim_pad: int = 0,
    persist_m: int = 4,
):
    """Compile stage2 kernel (`moe_gemm2`) and return the compiled executable.

    a_dtype:
      - "fp8": A2 is fp8
      - "fp16": A2 is fp16 (caller uses tile_k halved vs fp8 to match MFMA K halving)
      - "int8": A2 is int8
      - "fp4": A2 is fp4

    b_dtype:
      - "fp8": W is fp8
      - "fp16": W is fp16 (caller uses tile_k halved vs fp8 to match MFMA K halving)
      - "int8": W is int8
      - "int4": W4A8 path: A2 is int8, W is packed int4 (2 values per byte) unpacked to int8 in-kernel
      - "fp4": W is fp4

    Stage2 output supports:
      - out_dtype="f16": fp16 half2 atomics (fast, can overflow to +/-inf for bf16 workloads)
      - out_dtype="f32": fp32 scalar atomics (slower, but avoids fp16 atomic overflow)

    `use_cshuffle_epilog` controls whether we use the LDS CShuffle epilogue before
    global atomics (recommended for performance).
    """
    gpu_arch = get_hip_arch()
    allocator = SmemAllocator(None, arch=gpu_arch)
    _state = {}

    if a_dtype not in ("fp8", "fp16", "int8", "fp4"):
        raise ValueError(
            f"a_dtype must be one of ('fp8','fp16','int8','fp4'), got {a_dtype!r}"
        )
    if b_dtype not in ("fp8", "fp16", "int8", "int4", "fp4"):
        raise ValueError(
            f"b_dtype must be one of ('fp8','fp16','int8','int4','fp4'), got {b_dtype!r}"
        )

    is_f16_a = a_dtype == "fp16"
    is_f16_b = b_dtype == "fp16"
    is_f16 = is_f16_a or is_f16_b

    is_f8_a = a_dtype == "fp8"
    is_f4_a = a_dtype == "fp4"
    is_f4_b = b_dtype == "fp4"

    pack_M = 2
    pack_N = 2
    pack_K = 2

    elem_bytes = 1

    a_elem_bytes = 2 if is_f16_a else 1
    b_elem_bytes = 1
    tile_k_bytes = int(tile_k) * int(a_elem_bytes)

    a_elem_vec_pack = 2 if is_f4_a else 1
    cbsz = 0 if is_f8_a else 4
    blgp = 4

    # K64-byte micro-step: always 64 bytes per `ku`. For fp16, this is 32 elements (2xK16 MFMA).
    if (tile_k_bytes % 64) != 0:
        raise ValueError(
            f"tile_k_bytes must be divisible by 64, got tile_k_bytes={tile_k_bytes} "
            f"(tile_k={tile_k}, elem_bytes={a_elem_bytes})"
        )

    out_s = str(out_dtype).strip().lower()
    if out_s not in ("f16", "fp16", "half", "bf16", "bfloat16", "f32", "fp32", "float"):
        raise ValueError(
            f"out_dtype must be 'f16', 'bf16', or 'f32', got {out_dtype!r}"
        )
    out_is_f32 = out_s in ("f32", "fp32", "float")
    out_is_bf16 = out_s in ("bf16", "bfloat16")
    if (not bool(accumulate)) and out_is_f32:
        raise ValueError(
            "compile_moe_gemm2(accumulate=False) only supports out_dtype in {'f16','bf16'}"
        )
    is_int4 = b_dtype == "int4"
    # INT4 here means W4A8: A2 is int8, W is packed int4 and unpacked to int8 in-kernel.
    is_int8 = False

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

    def _x_elem_type():
        if is_f4_b:
            return T.f8 if is_f8_a else T.i8
        return T.f16 if is_f16_a else (T.i8 if is_int8 else T.f8)

    def _w_elem_type():
        if is_f4_b:
            return T.i8
        return T.f16 if is_f16_b else (T.i8 if is_int8 else T.f8)

    def _scale_elem_type():
        return T.i32

    DYN = ir.ShapedType.get_dynamic_size()
    size_out = DYN
    size_x = DYN
    size_sorted = DYN
    size_expert_ids_shape = DYN
    size_scale_x = DYN
    # W is packed int4 for W4A8: 2 values per byte.
    size_w = (
        (experts * model_dim * inter_dim) // 2
        if is_int4
        else (experts * model_dim * inter_dim)
    )

    total_threads = 256
    bytes_x_per_tile = int(tile_m) * int(tile_k) * int(a_elem_bytes)
    if bytes_x_per_tile % total_threads != 0:
        raise ValueError(
            "tile_m*tile_k*elem_bytes must be divisible by "
            f"{total_threads}: tile_m={tile_m}, tile_k={tile_k}, elem_bytes={a_elem_bytes}"
        )
    bytes_per_thread_x = bytes_x_per_tile // total_threads

    _use_lds128 = os.environ.get("FLIR_CK_LDS128", "1") in (
        "1",
        "true",
        "True",
        "YES",
        "yes",
    )
    pad_k = 0 if _use_lds128 else 8
    lds_stride = tile_k + pad_k

    if out_is_f32:
        # Match origin/dev_a16w4: f32 output uses scalar atomics and does NOT use the CShuffle epilogue.
        _use_cshuffle_epilog = (
            False if use_cshuffle_epilog is None else bool(use_cshuffle_epilog)
        )
        if _use_cshuffle_epilog:
            raise ValueError(
                "out_dtype='f32' does not support CShuffle epilogue (set use_cshuffle_epilog=False)."
            )
    else:
        if use_cshuffle_epilog is None:
            _use_cshuffle_epilog = os.environ.get("FLIR_MOE_STAGE2_CSHUFFLE", "1") in (
                "1",
                "true",
                "True",
                "YES",
                "yes",
            )
        else:
            _use_cshuffle_epilog = bool(use_cshuffle_epilog)
        if not _use_cshuffle_epilog:
            raise ValueError(
                "stage2 f16 output currently requires CShuffle epilogue (FLIR_MOE_STAGE2_CSHUFFLE=1)."
            )

    # NOTE: Keep this as a callable so we don't require an MLIR Context at Python-time.
    def out_elem():
        return T.f32 if out_is_f32 else (T.bf16 if out_is_bf16 else T.f16)

    epilog_tag = "cshuffle"
    # IMPORTANT: include tiling in the module name to avoid accidentally reusing a compiled
    # binary for a different (tile_m, tile_n, tile_k) configuration.
    # See stage1 note: include ABI tag to prevent binary reuse across signature changes.
    # IMPORTANT: module name participates in the compiler cache key.
    # Dynamic-shape variant: safe to reuse across (tokens/sorted_size/size_expert_ids) at runtime.
    # Keep a distinct ABI tag so the compile cache never mixes with historical signatures.
    module_name = (
        f"mfma_moe2_a{a_dtype}_w{b_dtype}_{out_s}_{epilog_tag}"
        f"_t{tile_m}x{tile_n}x{tile_k}"
        f"_vscale_fix3_pm{persist_m}"
    ).replace("-", "_")
    # -- LDS sizing (pure Python; no MLIR Context needed) ---------------------
    # Reuse a single allocation for both:
    # - ping-pong A2 tiles (2 * tile_m * lds_stride * elem_bytes bytes)
    # - epilogue CShuffle tile (tile_m * tile_n f16 -> 2 * tile_m * tile_n bytes)
    lds_x_bytes = 2 * int(tile_m) * int(lds_stride) * int(a_elem_bytes)
    lds_out_bytes = (
        2 * int(tile_m) * int(tile_n) if _use_cshuffle_epilog else 0
    )  # f16 bytes
    lds_tid_bytes = int(tile_m) * 4
    lds_total_bytes = max(lds_x_bytes, lds_out_bytes) + lds_tid_bytes
    lds_total_elems = lds_total_bytes if a_elem_bytes == 1 else (lds_total_bytes // 2)

    def x_lds_elem():
        return T.f16 if is_f16_a else (T.i8 if is_int8 else T.f8)

    lds_alloc_bytes = int(lds_total_elems) * int(a_elem_bytes)
    lds_alloc_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_alloc_offset + lds_alloc_bytes

    if True:

        @flyc.kernel
        def moe_gemm2(
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
            x_elem = T.f16 if is_f16_a else (T.i8 if is_int8 else T.f8)
            # For int4, weights are stored as packed bytes (i8) and unpacked to i8 packs.
            w_elem = T.f16 if is_f16_b else (T.i8 if is_int8 else T.f8)
            f16 = T.f16
            f32 = T.f32
            i32 = T.i32
            i64 = T.i64
            vec4_f32 = T.vec(4, f32)
            vec4_i32 = T.vec(4, i32)
            vec1_f16 = T.vec(1, f16)
            vec2_f16 = T.vec(2, f16)
            vec4_f16 = T.vec(4, f16)
            vec16_elems = 16 if a_elem_bytes == 1 else 8
            vec8_elems = 8 if a_elem_bytes == 1 else 4
            vec4_elems = 4 if a_elem_bytes == 1 else 2
            vec8_x = T.vec(vec8_elems, x_elem)
            vec16_x = T.vec(vec16_elems, x_elem)
            vec1_i64 = T.vec(1, i64)
            vec2_i64 = T.vec(2, i64)

            acc_init = (
                arith.constant_vector(0, vec4_i32)
                if is_int8
                else arith.constant_vector(0.0, vec4_f32)
            )

            # A2 layout (flatten token-slot -> M; use i32 for fly.make_shape).
            topk_idx = arith.constant(topk, index=True)
            m_in = tokens_in * topk_idx
            # fly.make_shape requires i32/i64, not index
            m_i32_v = arith.index_cast(T.i32, m_in)
            k_i32_v = i32_k_in.ir_value()
            layout_x = fx.make_layout((m_i32_v, k_i32_v), stride=(k_i32_v, 1))

            # B preshuffle layout: [experts*model_dim, inter_dim]
            c_n_total = arith.constant(experts * model_dim, index=True)
            kpack_bytes = 8 if is_int4 else 16
            b_layout = make_preshuffle_b_layout(
                arith,
                c_n=c_n_total,
                c_k=k_in // pack_K,
                kpack_bytes=kpack_bytes,
                elem_bytes=b_elem_bytes,
            )
            layout_b = b_layout.layout_b
            c_k0 = (k_in * arith.constant(int(a_elem_bytes), index=True)) / arith.index(
                64
            )

            def check_c_n_valid_gate(base_n):
                return arith.cmpi(CmpIPredicate.ult, base_n, model_dim - model_dim_pad)

            def check_c_k_valid_gate(base_k):
                return arith.cmpi(CmpIPredicate.ult, base_k, inter_dim - inter_dim_pad)

            # A&B's scale preshuffle layout
            # For fp4, k_in is already packed (inter_dim // a_elem_vec_pack), so we need original inter_dim
            c_k_orig = arith.constant(inter_dim, index=True)
            layout_a_scale = make_preshuffle_scale_layout(
                arith, c_mn=m_in, c_k=c_k_orig
            )
            layout_b_scale = make_preshuffle_scale_layout(
                arith, c_mn=c_n_total, c_k=c_k_orig
            )

            shape_lds = fx.make_shape(tile_m, tile_k)
            stride_lds = fx.make_stride(lds_stride, 1)
            layout_lds = fx.make_layout(shape_lds, stride_lds)

            tx = gpu.thread_id("x")
            # Align with Aiter launch mapping:
            # - blockIdx.x -> N dimension (tile along model_dim)
            # - blockIdx.y -> expert-block id / M dimension (tile along sorted M)
            by = gpu.block_id("x")  # tile along model_dim
            bx_persist = gpu.block_id("y")  # persistent WG index

            # XOR16 swizzle parameter (in bytes; constant, power-of-two in our configs).
            k_blocks16 = arith.constant(tile_k_bytes // 16, index=True)
            layout_tx_wave_lane = fx.make_layout((4, 64), stride=(64, 1))
            layout_lane16 = fx.make_layout((4, 16), stride=(16, 1))

            base_ptr = allocator.get_base()
            lds_x_ptr = SmemPtr(
                base_ptr,
                lds_alloc_offset,
                x_lds_elem(),
                shape=(lds_total_elems,),
            )
            lds_x = lds_x_ptr.get()
            # Alias the same underlying LDS bytes as f16/bf16 for epilogue shuffle.
            lds_out = (
                SmemPtr(
                    base_ptr,
                    lds_x_ptr.byte_offset,
                    (T.bf16 if out_is_bf16 else T.f16),
                    shape=(tile_m * tile_n,),
                ).get()
                if _use_cshuffle_epilog
                else None
            )

            # lds_tid: alias LDS after max(x, out) for sorted_idx preload
            _lds_x_b = 2 * int(tile_m) * int(lds_stride) * int(a_elem_bytes)
            _lds_out_b = 2 * int(tile_m) * int(tile_n) if _use_cshuffle_epilog else 0
            _lds_tid_off = max(_lds_x_b, _lds_out_b)
            lds_tid = SmemPtr(
                base_ptr, lds_x_ptr.byte_offset + _lds_tid_off, T.i32, shape=(tile_m,)
            ).get()

            # Buffer resources.
            # For dynamic memrefs, `max_size=False` cannot infer the logical size from the memref *type*,
            # so we should pass `num_records_bytes` explicitly for stable hardware OOB behavior.
            c_topk = arith.constant(topk, index=True)

            # X(A2): buffer size in bytes, accounting for FP4 packing (2 elements per byte).
            # fp8/int8: 1 byte per element  -> bytes = tokens*topk * K
            # fp4:      2 elements per byte -> bytes = tokens*topk * K / 2
            c_a_pack = arith.constant(int(a_elem_vec_pack), index=True)
            c_elem_bytes = arith.constant(int(a_elem_bytes), index=True)
            x_nbytes_idx = ((tokens_in * c_topk) * k_in * c_elem_bytes) / c_a_pack
            x_nbytes_i32 = arith.index_cast(T.i32, x_nbytes_idx)
            x_rsrc = buffer_ops.create_buffer_resource(
                arg_x, max_size=False, num_records_bytes=x_nbytes_i32
            )

            w_rsrc = buffer_ops.create_buffer_resource(arg_w, max_size=False)

            # OUT: [tokens, model_dim] -> clamp to descriptor max (i32 bytes) to avoid overflow on huge tokens.
            out_elem_bytes = 4 if out_is_f32 else 2
            out_nbytes_idx = (
                tokens_in * n_in * arith.constant(out_elem_bytes, index=True)
            )
            if not bool(accumulate):
                out_nbytes_idx = (
                    tokens_in
                    * arith.index(topk)
                    * n_in
                    * arith.constant(out_elem_bytes, index=True)
                )
            out_nbytes_i32 = arith.index_cast(T.i32, out_nbytes_idx)
            out_rsrc = buffer_ops.create_buffer_resource(
                arg_out, max_size=False, num_records_bytes=out_nbytes_i32
            )

            # num_valid_ids (sorted padded MN) for scale sizing / guards.
            numids_rsrc = buffer_ops.create_buffer_resource(
                arg_num_valid_ids,
                max_size=False,
                num_records_bytes=arith.constant(4, type=T.i32),
            )
            num_valid_i32 = buffer_ops.buffer_load(
                numids_rsrc, arith.constant(0, index=True), vec_width=1, dtype=T.i32
            )
            num_valid_idx = arith.index_cast(ir.IndexType.get(), num_valid_i32)

            # fp16 path ignores scales completely (implicit scale=1.0).
            if is_f16_a:
                sx_rsrc = None
            else:
                if is_f4_a:
                    # A2 microscale: e8m0 in sorted layout [sorted_size, K/32].
                    # Caller must pre-scatter a2_scale via moe_mxfp4_sort.
                    c32 = arith.constant(32, index=True)
                    kblk = k_in / c32
                    sx_nbytes_idx = num_valid_idx * kblk
                    sx_nbytes_i32 = arith.index_cast(T.i32, sx_nbytes_idx)
                    sx_rsrc = buffer_ops.create_buffer_resource(
                        arg_scale_x, max_size=False, num_records_bytes=sx_nbytes_i32
                    )
                else:
                    # scale_x (A2 scale): [tokens*topk] f32 -> bytes = tokens*topk*4
                    sx_nbytes_idx = (tokens_in * c_topk) * arith.constant(4, index=True)
                    sx_nbytes_i32 = arith.index_cast(T.i32, sx_nbytes_idx)
                    sx_rsrc = buffer_ops.create_buffer_resource(
                        arg_scale_x, max_size=False, num_records_bytes=sx_nbytes_i32
                    )

            if is_f16_b:
                sw_rsrc = None
            else:
                # Weight microscale buffer (packed i32 holding e8m0 bytes).
                # Use an exact descriptor size so hardware OOB checking works.
                c32 = arith.constant(32, index=True)
                kblk_w = k_in / c32  # K/32
                mn_w = arith.constant(experts * model_dim, index=True)
                sw_nbytes_idx = mn_w * kblk_w  # bytes (e8m0)
                sw_nbytes_i32 = arith.index_cast(T.i32, sw_nbytes_idx)
                sw_rsrc = buffer_ops.create_buffer_resource(
                    arg_scale_w, max_size=False, num_records_bytes=sw_nbytes_i32
                )

            # sorted_token_ids / sorted_weights: [blocks*tile_m] (padded length)
            sorted_nbytes_idx = (
                size_expert_ids_in
                * arith.constant(tile_m, index=True)
                * arith.constant(4, index=True)
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

            # expert ids: [blocks] i32 -> bytes = size_expert_ids_in*4
            eid_nbytes_idx = size_expert_ids_in * arith.constant(4, index=True)
            eid_nbytes_i32 = arith.index_cast(T.i32, eid_nbytes_idx)
            expert_rsrc = buffer_ops.create_buffer_resource(
                arg_expert_ids, max_size=False, num_records_bytes=eid_nbytes_i32
            )
            bias_rsrc = (
                buffer_ops.create_buffer_resource(arg_bias, max_size=False)
                if enable_bias
                else None
            )

            # ---- persist_m loop ----
            _PERSIST_M = persist_m
            _c0_p = arith.constant(0, index=True)
            _c1_p = arith.constant(1, index=True)
            _c_pm = arith.constant(_PERSIST_M, index=True)
            _for_persist = scf.ForOp(_c0_p, _c_pm, _c1_p)
            _for_ip = ir.InsertionPoint(_for_persist.body)
            _for_ip.__enter__()
            _mi_p = _for_persist.induction_variable
            bx = bx_persist * _c_pm + _mi_p
            bx_m = bx * arith.constant(tile_m, index=True)

            # Early-exit guard: skip garbage expert blocks beyond `num_valid_ids`.
            bx_m_i32 = arith.index_cast(T.i32, bx_m)
            blk_valid = arith.cmpi(CmpIPredicate.ult, bx_m_i32, num_valid_i32)

            expert_i32 = buffer_ops.buffer_load(
                expert_rsrc, bx, vec_width=1, dtype=T.i32
            )
            expert_idx = arith.index_cast(ir.IndexType.get(), expert_i32)
            exp_valid = arith.cmpi(
                CmpIPredicate.ult, expert_i32, arith.constant(experts, type=T.i32)
            )

            def _moe_gemm2_then_body():
                # Expert id for this M tile.
                n_idx = arith.constant(model_dim, index=True)
                expert_off_idx = expert_idx * n_idx  # index

                # ---- X gmem->reg prefetch (match preshuffle GEMM mapping) ----
                # Prefer 16B buffer-load (dwordx4). If the per-thread byte count isn't divisible by
                # 16, fall back to 8B (dwordx2) or 4B (dword) loads. For fp16 we require 16B.
                if is_f16_a:
                    if bytes_per_thread_x % 16 != 0:
                        raise ValueError(
                            f"[fp16] bytes_per_thread_x ({bytes_per_thread_x}) must be divisible by 16"
                        )
                    x_load_bytes = 16
                else:
                    if bytes_per_thread_x % 16 == 0:
                        x_load_bytes = 16
                    elif bytes_per_thread_x % 8 == 0:
                        x_load_bytes = 8
                    elif bytes_per_thread_x % 4 == 0:
                        x_load_bytes = 4
                    else:
                        raise ValueError(
                            f"bytes_per_thread_x ({bytes_per_thread_x}) must be divisible by 4 to use the dword-indexed load mapping."
                        )
                num_x_loads = bytes_per_thread_x // x_load_bytes
                chunk_i32 = x_load_bytes // 4  # dwords per chunk (1/2/4)
                vec4_i32 = T.vec(4, i32)

                c_k_div4 = (
                    (k_in / c_a_pack) * arith.constant(int(a_elem_bytes), index=True)
                ) / arith.index(4)
                c_k_div4_i32 = arith.index_cast(T.i32, c_k_div4)
                layout_x_div4 = fx.make_layout(
                    (m_i32_v, c_k_div4_i32), stride=(c_k_div4_i32, 1)
                )
                tile_k_dwords = (int(tile_k) * int(a_elem_bytes)) // (
                    4 * int(a_elem_vec_pack)
                )
                layout_x_tile_div4 = fx.make_layout(
                    (tile_m, tile_k_dwords), stride=(tile_k_dwords, 1)
                )
                c_chunk_i32 = arith.constant(chunk_i32, index=True)
                tx_i32_base = tx * c_chunk_i32

                topk_i32 = arith.constant(topk)
                mask24 = arith.constant(0xFFFFFF)
                # Sentinel clamp uses `tokens` as the upper bound: t_valid = (t < tokens).
                tokens_i32 = arith.index_cast(T.i32, tokens_in)

                def x_tile_chunk_coord_i32(i: int):
                    return tile_chunk_coord_i32(
                        arith,
                        tx_i32_base=tx_i32_base,
                        i=i,
                        total_threads=total_threads,
                        layout_tile_div4=layout_x_tile_div4,
                        chunk_i32=chunk_i32,
                    )

                vec1_i32 = T.vec(1, i32)
                vec2_i32 = T.vec(2, i32)
                vec4_x = T.vec(4, x_elem)
                x_load_vec_elems = (
                    x_load_bytes if a_elem_bytes == 1 else x_load_bytes // a_elem_bytes
                )

                def load_x(idx_i32):
                    """Load `x_load_bytes` bytes from X (gmem) into regs.

                    For 16B, keep the fast dwordx4 path. For 8B/4B, use byte offsets.
                    """
                    if x_load_bytes == 16:
                        idx_elem = (
                            idx_i32 if a_elem_bytes == 1 else (idx_i32 * arith.index(2))
                        )
                        return buffer_copy_gmem16_dwordx4(
                            buffer_ops,
                            vector,
                            elem_type=x_elem,
                            idx_i32=idx_elem,
                            rsrc=x_rsrc,
                            vec_elems=vec16_elems,
                        )
                    # 8B/4B: convert dword index to byte offset and use offset_in_bytes path.
                    idx_bytes = idx_i32 * arith.index(4)
                    return _buffer_load_vec(
                        buffer_ops,
                        vector,
                        x_rsrc,
                        idx_bytes,
                        elem_type=x_elem,
                        vec_elems=x_load_vec_elems,
                        elem_bytes=a_elem_bytes,
                        offset_in_bytes=True,
                    )

                # decode routed token once (per thread's M-slice) and build a base offset.
                x_row_base_div4 = []
                x_col_local_i32 = []
                x_row_local = []
                for i in range_constexpr(num_x_loads):
                    row_local, col_local_i32 = x_tile_chunk_coord_i32(i)
                    x_row_local.append(row_local)
                    x_col_local_i32.append(col_local_i32)

                    sorted_row_i = bx_m + row_local
                    fused_i = buffer_ops.buffer_load(
                        sorted_rsrc, sorted_row_i, vec_width=1, dtype=T.i32
                    )
                    t_i32 = arith.andi(fused_i, mask24)
                    s_i32 = arith.shrui(fused_i, arith.constant(24))
                    # Keep `blk_valid` only; remove per-row token validity checks.

                    t_valid = arith.cmpi(CmpIPredicate.ult, t_i32, tokens_i32)
                    s_valid = arith.cmpi(CmpIPredicate.ult, s_i32, topk_i32)
                    ts_valid = arith.andi(t_valid, s_valid)
                    t_safe = arith.select(ts_valid, t_i32, arith.constant(0))
                    s_safe = arith.select(ts_valid, s_i32, arith.constant(0))
                    row_ts_i32 = t_safe * topk_i32 + s_safe
                    row_ts_idx = arith.index_cast(ir.IndexType.get(), row_ts_i32)

                    # Base row offset in dword units: row_ts_idx * (k_in/4)
                    x_row_base_div4.append(row_ts_idx * c_k_div4)

                def load_x_tile(base_k):
                    base_k_div4 = (
                        (base_k / c_a_pack)
                        * arith.constant(int(a_elem_bytes), index=True)
                    ) / arith.index(4)
                    parts = []
                    for i in range_constexpr(num_x_loads):
                        idx_i32 = x_row_base_div4[i] + base_k_div4 + x_col_local_i32[i]
                        x_vec = load_x(idx_i32)

                        if x_load_bytes == 16:
                            parts.append(vector.bitcast(vec4_i32, x_vec))
                        elif x_load_bytes == 8:
                            parts.append(vector.bitcast(vec2_i32, x_vec))
                        else:
                            parts.append(vector.bitcast(vec1_i32, x_vec))
                    return parts

                # tx -> wave/lane (GEMM-style decomposition).
                coord_wl = idx2crd(tx, layout_tx_wave_lane)
                wave_id = layout_get(coord_wl, 0)
                lane_id = layout_get(coord_wl, 1)
                coord_l16 = idx2crd(lane_id, layout_lane16)
                lane_div_16 = layout_get(coord_l16, 0)
                lane_mod_16 = layout_get(coord_l16, 1)

                row_a_lds = lane_mod_16

                col_offset_base = lane_div_16 * arith.constant(16, index=True)

                # Dynamic N tiling within block.
                by_n = by * arith.constant(tile_n, index=True)
                num_waves = 4
                n_per_wave = tile_n // num_waves
                num_acc_n = n_per_wave // 16
                c_n_per_wave = arith.constant(n_per_wave, index=True)
                wave_mod_4 = wave_id % arith.constant(4, index=True)
                n_tile_base = wave_mod_4 * c_n_per_wave

                # Precompute (n_blk, n_intra) for B, and col indices for output.
                n_intra_list = []
                n_blk_list = []
                col_g_list = []
                c_n0 = c_n_total / arith.index(16)
                c_n0_static = experts * model_dim // 16
                layout_n_blk_intra = fx.make_layout((c_n0_static, 16), stride=(16, 1))

                for i in range_constexpr(num_acc_n):
                    offset = i * 16

                    col_g = by_n + n_tile_base
                    col_g = col_g // 2 + offset
                    col_g = col_g + lane_mod_16
                    col_g_list.append(col_g)

                    c_offset = arith.constant(offset, index=True)
                    global_n = by_n + n_tile_base + c_offset + lane_mod_16
                    row_w = expert_off_idx + global_n
                    coord_n = idx2crd(row_w, layout_n_blk_intra)
                    n_blk_list.append(layout_get(coord_n, 0))
                    n_intra_list.append(layout_get(coord_n, 1))

                m_repeat = tile_m // 16
                k_unroll = tile_k_bytes // 128  # K64-byte micro-step (2x MFMA)

                # fp4 pack
                k_unroll_packed = k_unroll // pack_K
                m_repeat_packed = m_repeat // pack_M
                num_acc_n_packed = num_acc_n // pack_N

                # --- B Load Logic (K64) - shared layout with preshuffle GEMM ---
                def load_b_packs_k64(base_k, ku: int, ni: int):
                    """Load one K64-byte B micro-step: single 16B load, split into 2x i64."""
                    c64 = arith.constant(64, index=True)
                    base_k_bytes = base_k * arith.constant(
                        int(b_elem_bytes), index=True
                    )
                    k0_base = base_k_bytes // c64
                    k0 = k0_base + arith.constant(ku, index=True)
                    k1 = lane_div_16
                    coord_pack = (
                        n_blk_list[ni],
                        k0,
                        k1,
                        n_intra_list[ni],
                        arith.constant(0, index=True),
                    )
                    idx_pack = crd2idx(coord_pack, layout_b)

                    vec_elems = kpack_bytes // int(b_elem_bytes)
                    b16 = _buffer_load_vec(
                        buffer_ops,
                        vector,
                        w_rsrc,
                        idx_pack,
                        elem_type=_w_elem_type(),
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
                        packs0 = []
                        packs1 = []
                        for ni in range_constexpr(num_acc_n):
                            b0, b1 = load_b_packs_k64(base_k, ku, ni)
                            packs0.append(b0)
                            packs1.append(b1)
                        b_tile.append((packs0, packs1))
                    return b_tile

                def load_scale(arg_scale, rsrc, scale_info, ku, mni):
                    k_lane = lane_div_16
                    n_lane = lane_mod_16
                    # Direct arith crd2idx: idx = mni*stride_n0 + ku*stride_k0 + k_lane*stride_klane + n_lane
                    idx_pack = (
                        mni * scale_info.stride_n0
                        + ku * scale_info.stride_k0
                        + k_lane * scale_info.stride_klane
                        + n_lane
                    )
                    s = buffer_ops.buffer_load(rsrc, idx_pack, vec_width=1, dtype=T.i32)
                    return vector.from_elements(T.vec(1, T.i32), [s])

                def load_b_scale_tile(base_k):
                    b_scale_tile = []
                    for ku in range_constexpr(k_unroll_packed):
                        for ni in range_constexpr(num_acc_n_packed):
                            scale = load_scale(
                                arg_scale_w,
                                sw_rsrc,
                                layout_b_scale,
                                ku + base_k,
                                ni
                                + (expert_off_idx + by_n + n_tile_base) // pack_N // 16,
                            )
                            b_scale_tile.append(scale)
                    return b_scale_tile

                def load_a_scale_tile(base_k):
                    a_scale_tile = []
                    for ku in range_constexpr(k_unroll_packed):
                        for mi in range_constexpr(m_repeat_packed):
                            scale = load_scale(
                                arg_scale_x,
                                sx_rsrc,
                                layout_a_scale,
                                ku + base_k,
                                mi + bx_m // pack_M // 16,
                            )
                            a_scale_tile.append(scale)
                    return a_scale_tile

                def prefetch_ab_scale_tile(base_k):
                    return [load_a_scale_tile(base_k), load_b_scale_tile(base_k)]

                vec8_x = T.vec(vec8_elems, x_elem)
                vec4_x_lds = T.vec(vec4_elems, x_elem)

                # ---- Pipeline helpers: store X tile to LDS with ping-pong base ----
                def store_x_tile_to_lds(vec_x_in_parts, lds_base):
                    for i in range_constexpr(num_x_loads):
                        row_local = x_row_local[i]
                        col_local_i32 = x_col_local_i32[i]
                        if x_load_bytes == 16:
                            lds_store_16b_xor16(
                                arith,
                                vector,
                                lds_memref=lds_x,
                                vec16_ty=vec16_x,
                                layout_lds=layout_lds,
                                row_local=row_local,
                                col_local_i32=col_local_i32,
                                tx_c4=arith.index(4),
                                k_blocks16=k_blocks16,
                                lds_base=lds_base,
                                vec_part_i32x4=vec_x_in_parts[i],
                                elem_bytes=elem_bytes,
                            )
                        elif x_load_bytes == 8:
                            lds_store_8b_xor16(
                                arith,
                                vector,
                                lds_memref=lds_x,
                                vec8_ty=vec8_x,
                                layout_lds=layout_lds,
                                row_local=row_local,
                                col_local_i32=col_local_i32,
                                tx_c4=arith.index(4),
                                k_blocks16=k_blocks16,
                                lds_base=lds_base,
                                vec_part_i32x2=vec_x_in_parts[i],
                                elem_bytes=elem_bytes,
                            )
                        else:  # x_load_bytes == 4
                            lds_store_4b_xor16(
                                arith,
                                vector,
                                lds_memref=lds_x,
                                vec4_ty=vec4_x_lds,
                                layout_lds=layout_lds,
                                row_local=row_local,
                                col_local_i32=col_local_i32,
                                tx_c4=arith.index(4),
                                k_blocks16=k_blocks16,
                                lds_base=lds_base,
                                vec_part_i32x1=vec_x_in_parts[i],
                                elem_bytes=elem_bytes,
                            )

                # --- A LDS load helper for K64 (load 16B once, extract 2x i64 halves) ---
                def lds_load_packs_k64(curr_row_a_lds, col_base, lds_base):
                    # Swizzle in bytes, then convert to element offset for memref indexing.
                    col_base_swz_bytes = swizzle_xor16(
                        curr_row_a_lds, col_base, k_blocks16
                    )
                    col_base_swz = (
                        col_base_swz_bytes
                        if elem_bytes == 1
                        else (col_base_swz_bytes / arith.index(2))
                    )
                    # Pass as list so layout_utils.crd2idx uses static arith path
                    idx_a16 = crd2idx([curr_row_a_lds, col_base_swz], layout_lds)
                    idx_a16 = idx_a16 + lds_base
                    loaded_a16 = vector.load_op(vec16_x, lds_x, [idx_a16])
                    a_i64x2 = vector.bitcast(vec2_i64, loaded_a16)
                    a0 = vector.extract(
                        a_i64x2, static_position=[0], dynamic_position=[]
                    )
                    a1 = vector.extract(
                        a_i64x2, static_position=[1], dynamic_position=[]
                    )
                    return a0, a1

                def compute_tile(
                    acc_in,
                    b_tile_in,
                    lds_base,
                    a_scale=None,
                    b_scale=None,
                    *,
                    prefetch_epilogue: bool = False,
                    a0_prefetch=None,
                ):
                    acc_list = list(acc_in)
                    mfma_res_ty = vec4_i32 if is_int8 else vec4_f32

                    epilogue_pf = None
                    bias = None
                    if prefetch_epilogue:
                        if enable_bias:
                            bias = []
                            for ni in range_constexpr(num_acc_n):
                                global_n = by_n + n_tile_base + ni * 16 + lane_mod_16
                                bias_offset = expert_off_idx + global_n
                                bias.append(
                                    buffer_ops.buffer_load(
                                        bias_rsrc, bias_offset, vec_width=1, dtype=f32
                                    )
                                )
                        tw_pf = None
                        if doweight_stage2:
                            tw_pf = []
                            lane_div_16_mul4_pf = lane_div_16 * arith.index(4)
                            ii_idx_list_pf = [
                                arith.constant(ii, index=True) for ii in range(4)
                            ]
                            for mi in range_constexpr(m_repeat):
                                mi_base_pf = arith.constant(mi * 16, index=True)
                                for ii in range_constexpr(4):
                                    row_off_pf = (
                                        lane_div_16_mul4_pf + ii_idx_list_pf[ii]
                                    )
                                    row_in_tile_pf = mi_base_pf + row_off_pf
                                    sorted_row_pf = bx_m + row_in_tile_pf
                                    tw_pf.append(
                                        buffer_ops.buffer_load(
                                            sorted_w_rsrc,
                                            sorted_row_pf,
                                            vec_width=1,
                                            dtype=f32,
                                        )
                                    )
                        epilogue_pf = (None, tw_pf, bias)

                    c0_i64 = arith.constant(0, type=T.i64)
                    vec4_i64 = T.vec(4, T.i64)
                    vec8_i32 = T.vec(8, T.i32)

                    def pack_i64x4_to_i32x8(x0, x1, x2, x3):
                        v4 = vector.from_elements(vec4_i64, [x0, x1, x2, x3])
                        return vector.bitcast(vec8_i32, v4)

                    # fp4 path
                    for ku128 in range_constexpr(k_unroll_packed):
                        for mi in range_constexpr(m_repeat_packed):
                            a_scale_i32 = a_scale[ku128 * m_repeat_packed + mi]
                            a_scale_val = vector.extract(
                                a_scale_i32, static_position=[0], dynamic_position=[]
                            )
                            for ni in range_constexpr(num_acc_n_packed):
                                b_scale_i32 = b_scale[ku128 * num_acc_n_packed + ni]
                                b_scale_val = vector.extract(
                                    b_scale_i32,
                                    static_position=[0],
                                    dynamic_position=[],
                                )
                                for ikxdl in range_constexpr(pack_K):
                                    k_idx = ku128 * pack_K + ikxdl

                                    b_packs0, b_packs1 = b_tile_in[k_idx]

                                    col_base = (
                                        col_offset_base
                                        + (k_idx * 128) // a_elem_vec_pack
                                    )

                                    for imxdl in range_constexpr(pack_M):
                                        col_base0 = col_base
                                        mi_idx = mi * pack_M + imxdl
                                        mi_val = arith.constant(mi_idx * 16, index=True)
                                        curr_row_a_lds = row_a_lds + mi_val

                                        if (
                                            (a0_prefetch is not None)
                                            and (k_idx == 0)
                                            and (mi_idx == 0)
                                        ):
                                            a0, a1 = a0_prefetch
                                        else:
                                            a0, a1 = lds_load_packs_k64(
                                                curr_row_a_lds, col_base0, lds_base
                                            )

                                        if is_f8_a:
                                            col_base1 = col_base + 64
                                            a2, a3 = lds_load_packs_k64(
                                                curr_row_a_lds, col_base1, lds_base
                                            )
                                            a128 = pack_i64x4_to_i32x8(a0, a1, a2, a3)
                                        else:
                                            a128 = pack_i64x4_to_i32x8(
                                                a0, a1, c0_i64, c0_i64
                                            )

                                        for inxdl in range_constexpr(pack_N):
                                            ni_idx = ni * pack_N + inxdl

                                            b0 = b_packs0[ni_idx]
                                            b1 = b_packs1[ni_idx]
                                            b128 = pack_i64x4_to_i32x8(
                                                b0, b1, c0_i64, c0_i64
                                            )

                                            acc_idx = mi_idx * num_acc_n + ni_idx
                                            rocdl.sched_barrier(0)
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
                                                        b_scale_val,
                                                    ],
                                                )
                                            )

                    return acc_list, epilogue_pf

                # ---------------- 2-stage pipeline (ping-pong LDS + B tile prefetch) ----------------
                lds_tile_elems = arith.constant(tile_m * lds_stride, index=True)
                lds_base_cur = arith.index(0)
                lds_base_nxt = lds_tile_elems

                rocdl.sched_barrier(0)

                def hot_loop_scheduler():
                    # - MFMA group size per "slot": num_acc_n
                    # - Total MFMA per tile: (2*K32 per K64) * k_unroll * m_repeat * num_acc_n
                    # - We emit (mfma_group + dsrd + mfma_group) per scheduler iteration.
                    mfma_group = num_acc_n
                    mfma_total = (k_unroll * 2) * m_repeat * mfma_group
                    mfma_per_iter = 2 * mfma_group
                    sche_iters = (
                        0 if mfma_per_iter == 0 else (mfma_total // mfma_per_iter)
                    )

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

                    # DS-write hints near the end: match total A LDS-store micro-ops per thread.
                    dswr_tail = num_x_loads
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

                # Prologue -- B-first.
                k0 = arith.index(0)
                b_cur = load_b_tile(k0)
                a_scale_pong, b_scale_pong = prefetch_ab_scale_tile(k0 // pack_K // 128)
                x_regs0 = load_x_tile(k0)
                store_x_tile_to_lds(x_regs0, lds_base_cur)
                # Preload sorted_idx into lds_tid for epilogue precompute_row
                _c_tile_m_idx = arith.constant(tile_m, index=True)
                _tid_in_range = arith.cmpi(CmpIPredicate.ult, tx, _c_tile_m_idx)
                _if_tid = scf.IfOp(_tid_in_range)
                with ir.InsertionPoint(_if_tid.then_block):
                    _tid_row = bx_m + tx
                    _tid_val = buffer_ops.buffer_load(
                        sorted_rsrc, _tid_row, vec_width=1, dtype=T.i32
                    )
                    _tid_vec1 = vector.from_elements(T.vec(1, T.i32), [_tid_val])
                    vector.store(_tid_vec1, lds_tid, [tx])
                    scf.YieldOp([])
                gpu.barrier()

                acc = [acc_init] * num_acc_n * m_repeat
                lds_base_pong = lds_base_cur
                lds_base_ping = lds_base_nxt

                # Cross-tile A0 LDS prefetch (default-on): prefetch the first A-pack (K64) for the
                # tile we are about to compute from LDS, to overlap with upcoming VMEM.
                a0_prefetch_pong = lds_load_packs_k64(
                    row_a_lds, col_offset_base, lds_base_pong
                )
                # a0_prefetch_pong = lds_load_packs_k64(0, 0, lds_base_pong)

                # Main loop: process K tiles in 2-tile ping-pong steps.
                #
                # IMPORTANT: for odd number of K tiles, leave **1** tail tile; for even, leave **2**.
                # Otherwise the 2-tile tail below would double-count the last tile when num_tiles is odd
                # (e.g. inter_dim=192, tile_k=64 -> 3 tiles).
                num_k_tiles_py = int(inter_dim) // int(tile_k)
                odd_k_tiles = (num_k_tiles_py % 2) == 1
                tail_tiles = 1 if odd_k_tiles else 2
                k_main2_py = (num_k_tiles_py - tail_tiles) * int(tile_k)
                if k_main2_py < 0:
                    k_main2_py = 0

                c2_tile_k = arith.constant(tile_k * 2, index=True)
                b_pong = b_cur
                # Only emit the scf.for when there are actually iterations to run.
                # When k_main2_py == 0 the loop body is empty; emitting an scf.for
                # would create a region whose internal SSA values cannot be used
                # by the post-loop tail code.
                if k_main2_py > 0:
                    for k_iv_py in range_constexpr(0, k_main2_py, tile_k * 2):
                        k_iv = k_iv_py
                        next_k1 = k_iv + tile_k
                        x_regs_ping = load_x_tile(next_k1)
                        b_ping = load_b_tile(next_k1 // 2)
                        a_scale_ping, b_scale_ping = prefetch_ab_scale_tile(
                            next_k1 // pack_K // 128
                        )

                        acc, _ = compute_tile(
                            acc,
                            b_pong,
                            lds_base_pong,
                            a_scale_pong,
                            b_scale_pong,
                            a0_prefetch=a0_prefetch_pong,
                        )
                        store_x_tile_to_lds(x_regs_ping, lds_base_ping)
                        # hot_loop_scheduler()
                        gpu.barrier()

                        # Cross-tile prefetch for the ping tile we are about to compute.
                        a0_prefetch_ping = lds_load_packs_k64(
                            row_a_lds, col_offset_base, lds_base_ping
                        )

                        next_k2 = k_iv + c2_tile_k
                        x_regs_pong = load_x_tile(next_k2)
                        b_pong = load_b_tile(next_k2 // 2)
                        a_scale_pong, b_scale_pong = prefetch_ab_scale_tile(
                            next_k2 // pack_K // 128
                        )

                        acc, _ = compute_tile(
                            acc,
                            b_ping,
                            lds_base_ping,
                            a_scale_ping,
                            b_scale_ping,
                            a0_prefetch=a0_prefetch_ping,
                        )
                        store_x_tile_to_lds(x_regs_pong, lds_base_pong)
                        # hot_loop_scheduler()
                        gpu.barrier()

                        # Cross-tile prefetch for the next pong tile.
                        a0_prefetch_pong = lds_load_packs_k64(
                            row_a_lds, col_offset_base, lds_base_pong
                        )

                if odd_k_tiles:
                    # Tail: single remaining tile (already in `b_cur` / `lds_base_pong`).
                    acc, epilogue_pf = compute_tile(
                        acc,
                        b_pong,
                        lds_base_pong,
                        a_scale_pong,
                        b_scale_pong,
                        a0_prefetch=a0_prefetch_pong,
                        prefetch_epilogue=True,
                    )

                else:
                    # Tail: 2 remaining tiles.
                    k_tail1 = (k_in + tile_k - 1) // tile_k * tile_k - tile_k
                    x_regs_ping = load_x_tile(k_tail1)
                    b_ping = load_b_tile(k_tail1 // 2)
                    a_scale_ping, b_scale_ping = prefetch_ab_scale_tile(
                        k_tail1 // pack_K // 128
                    )

                    acc, _ = compute_tile(
                        acc,
                        b_pong,
                        lds_base_pong,
                        a_scale_pong,
                        b_scale_pong,
                        a0_prefetch=a0_prefetch_pong,
                    )

                    store_x_tile_to_lds(x_regs_ping, lds_base_ping)
                    # hot_loop_scheduler()
                    gpu.barrier()

                    # Epilogue tile with sw prefetch.
                    a0_prefetch_ping = lds_load_packs_k64(
                        row_a_lds, col_offset_base, lds_base_ping
                    )
                    acc, epilogue_pf = compute_tile(
                        acc,
                        b_ping,
                        lds_base_ping,
                        a_scale_ping,
                        b_scale_ping,
                        a0_prefetch=a0_prefetch_ping,
                        prefetch_epilogue=True,
                    )

                # ---------------- Epilogue: LDS CShuffle + atomic half2 (x2) ----------------
                # Reuse the shared helper so GEMM / MoE kernels share the exact same CShuffle skeleton.

                sw_pf = None
                tw_pf = None
                bias_pf = None
                if epilogue_pf is not None:
                    sw_pf, tw_pf, bias_pf = epilogue_pf

                mask24_i32 = arith.constant(0xFFFFFF)
                topk_i32_v = topk_i32

                zero_i32 = arith.constant(0)

                def atomic_add_f16x2(val_f16x2, byte_off_i32):
                    rocdl.raw_ptr_buffer_atomic_fadd(
                        val_f16x2,
                        out_rsrc,
                        byte_off_i32,
                        zero_i32,
                        zero_i32,
                    )

                # Weight scales for the N tile (col_g depends on lane/wave/by but not on (t,s)).
                if lds_out is None:
                    raise RuntimeError(
                        "FLIR_MOE_STAGE2_CSHUFFLE=1 but lds_out is not allocated/aliased."
                    )

                # Precompute the output base address (i64 index) for ALL paths.
                # Both accumulate=True (global atomic) and accumulate=False (global store)
                # need 64-bit addressing to avoid i32 offset overflow when
                # tokens * model_dim * elem_bytes > INT32_MAX (~150K tokens for model_dim=7168).
                from flydsl._mlir.dialects import fly as _fly

                _llvm_ptr_ty = ir.Type.parse("!llvm.ptr")
                out_base_ptr = _fly.extract_aligned_pointer_as_index(
                    _llvm_ptr_ty, arg_out
                )
                out_base_i64 = llvm.ptrtoint(T.i64, out_base_ptr)
                out_base_idx = arith.index_cast(ir.IndexType.get(), out_base_i64)

                def write_row_to_lds(
                    *,
                    mi: int,
                    ii: int,
                    row_in_tile,
                    row,
                    row_base_lds,
                    col_base_local,
                    num_acc_n: int,
                    lds_out,
                ):
                    # Match origin/dev_a16w4: rely on sentinel padded rows + hardware OOB behavior.
                    fused2 = buffer_ops.buffer_load(
                        sorted_rsrc, row, vec_width=1, dtype=T.i32
                    )
                    t2 = fused2 & mask24_i32
                    s2 = fused2 >> 24

                    t_ok = arith.cmpi(CmpIPredicate.ult, t2, tokens_i32)
                    s_ok = arith.cmpi(CmpIPredicate.ult, s2, topk_i32_v)
                    ts_ok = arith.andi(t_ok, s_ok)
                    t2_safe = arith.select(ts_ok, t2, arith.constant(0))
                    s2_safe = arith.select(ts_ok, s2, arith.constant(0))
                    t2_safe * topk_i32_v + s2_safe

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
                        if is_int8:
                            v = arith.sitofp(f32, v)
                        if enable_bias:
                            v = v + bias_pf[ni]

                        if doweight_stage2:
                            v = v * tw
                        v_out = arith.trunc_f(out_elem(), v)

                        lds_idx = row_base_lds + col_local
                        vec1_out = T.vec(1, out_elem())
                        v1 = vector.from_elements(vec1_out, [v_out])

                        vector.store(v1, lds_out, [lds_idx], alignment=2)

                def precompute_row(*, row_local, row):
                    # Use lds_tid (sorted_idx preloaded to LDS) instead of buffer_load
                    # to avoid extra VMEM round-trips in the epilogue.
                    fused2 = memref.load(lds_tid, [row_local])
                    row_i32 = arith.index_cast(T.i32, row)
                    row_valid0 = arith.cmpi(CmpIPredicate.ult, row_i32, num_valid_i32)
                    t = fused2 & mask24_i32
                    s = fused2 >> 24
                    t_ok = arith.cmpi(CmpIPredicate.ult, t, tokens_i32)
                    s_ok = arith.cmpi(CmpIPredicate.ult, s, topk_i32_v)
                    row_valid = arith.andi(row_valid0, arith.andi(t_ok, s_ok))
                    t_idx = arith.index_cast(ir.IndexType.get(), t)
                    s_idx = arith.index_cast(ir.IndexType.get(), s)
                    ts_idx = t_idx * arith.constant(topk, index=True) + s_idx
                    if accumulate:
                        row_byte_base = out_base_idx + t_idx * arith.constant(
                            model_dim * out_elem_bytes, index=True
                        )
                    else:
                        row_byte_base = out_base_idx + ts_idx * arith.constant(
                            model_dim * out_elem_bytes, index=True
                        )
                    return ((fused2, row_byte_base), row_valid)

                def _idx_to_llvm_ptr(idx_val, addr_space=1):
                    """Convert an index-typed byte address to !llvm.ptr<addr_space>."""
                    idx_v = idx_val._value if hasattr(idx_val, "_value") else idx_val
                    i64_v = arith.index_cast(T.i64, idx_v)
                    i64_raw = i64_v._value if hasattr(i64_v, "_value") else i64_v
                    ptr_ty = ir.Type.parse(f"!llvm.ptr<{addr_space}>")
                    return llvm.inttoptr(ptr_ty, i64_raw)

                def store_pair(*, row_local, row, row_ctx, col_pair0, col_g0, frag):
                    fused, row_byte_base = row_ctx
                    if not bool(accumulate):
                        # ---- 64-bit global store path (avoids i32 offset overflow) ----
                        col_idx = col_g0
                        byte_off_col = col_idx * arith.constant(
                            out_elem_bytes, index=True
                        )
                        ptr_addr_idx = row_byte_base + byte_off_col
                        out_ptr_v = _idx_to_llvm_ptr(ptr_addr_idx)
                        frag_v = frag._value if hasattr(frag, "_value") else frag
                        llvm.StoreOp(
                            frag_v,
                            out_ptr_v,
                            alignment=_e_vec * out_elem_bytes,
                            nontemporal=True,
                        )
                    else:
                        # ---- accumulate=True: 64-bit global atomic path ----
                        col_idx = col_g0
                        byte_off_col = col_idx * arith.constant(
                            out_elem_bytes, index=True
                        )
                        ptr_addr_idx = row_byte_base + byte_off_col
                        out_ptr_v = _idx_to_llvm_ptr(ptr_addr_idx)
                        frag_v = frag._value if hasattr(frag, "_value") else frag
                        llvm.AtomicRMWOp(
                            llvm.AtomicBinOp.fadd,
                            out_ptr_v,
                            frag_v,
                            llvm.AtomicOrdering.monotonic,
                            syncscope="agent",
                            alignment=_e_vec * out_elem_bytes,
                        )

                _e_vec = 2 if accumulate else min(tile_n // 32, 8)
                c_shuffle_epilog(
                    arith=arith,
                    vector=vector,
                    gpu=gpu,
                    scf=scf,
                    range_constexpr=range_constexpr,
                    tile_m=tile_m,
                    tile_n=tile_n,
                    e_vec=_e_vec,
                    m_repeat=m_repeat,
                    num_acc_n=num_acc_n,
                    tx=tx,
                    lane_div_16=lane_div_16,
                    lane_mod_16=lane_mod_16,
                    bx_m=bx_m,
                    by_n=by_n,
                    n_tile_base=n_tile_base,
                    lds_out=lds_out,
                    frag_elem_type=(
                        ir.BF16Type.get() if out_is_bf16 else ir.F16Type.get()
                    ),
                    write_row_to_lds=write_row_to_lds,
                    precompute_row=precompute_row,
                    store_pair=store_pair,
                )

            _if_blk = scf.IfOp(blk_valid)
            with ir.InsertionPoint(_if_blk.then_block):
                _ifexpert_of = scf.IfOp(exp_valid)
                with ir.InsertionPoint(_ifexpert_of.then_block):
                    _moe_gemm2_then_body()
                    scf.YieldOp([])
                scf.YieldOp([])

            gpu.barrier()
            scf.YieldOp([])
            _for_ip.__exit__(None, None, None)

    # -- Host launcher (flyc.jit + .launch) --------------------------------
    _cache_tag = (
        module_name,
        a_dtype,
        b_dtype,
        out_dtype,
        tile_m,
        tile_n,
        tile_k,
        doweight_stage2,
        accumulate,
        enable_bias,
        model_dim_pad,
        inter_dim_pad,
        use_cshuffle_epilog,
        persist_m,
    )

    @flyc.jit
    def launch_mixed_moe_gemm2(
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
        _c_pm_l = arith.constant(persist_m, index=True)
        gy = (
            arith.index_cast(ir.IndexType.get(), i32_size_expert_ids_in.ir_value())
            + _c_pm_l
            - arith.constant(1, index=True)
        ) / _c_pm_l

        moe_gemm2(
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
            grid=(gx, gy, 1),
            block=(256, 1, 1),
            stream=stream,
        )

    return launch_mixed_moe_gemm2
