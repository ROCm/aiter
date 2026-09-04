# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Varlen FlashAttention-2 forward for gfx950, hd=72 bf16.

CDNA wave64. Host Q/K/V stay width 72; LDS/PV tiles still pad D to 96
because mfma 32x32x16 cannot use a 72-wide K/N. QK only runs ceil(72/16)=5
K-steps (the last pad-16 is skipped). O stores mask D>=72.

GEMM1/GEMM2 use mfma 32x32x16 bf16: QK is mfma(K, Q)
and PV is mfma(V, P). K rows are bit-swizzled (swap bits 2 and 3 of
lane%32) so QK C[0:8]/C[8:16] is already the 32x32 B fragment and P
needs no shuffle_xor pack. Softmax still reduces the partner half-wave
with shuffle_xor 32. Each thread owns one Q row (lane%32).

V B/A-frags use two gfx950 ds_read_tr16_b64 (Double, LaneGroupSize 32).
V HBM is token-major buffer_load_to_lds. Q HBM overlaps the first K DMA.
After QK, next K is issued immediately so softmax/pack overlap V completion
and next-K DMA; PV waits for V (and that K) before ds_read_tr. V LDS is
single-buffered (~21.5 KB, K width 72 / V width 96) so three workgroups
fit in 64 KB LDS. After PV, the next V tile is DMA'd (lgkmcnt+barrier
first) so later tiles overlap V with the whole next QK. PV prefetches
the next k-step's first V A-frag during the last D-chunk MFMA. Softmax
VALU and MFMA regions use s_setprio 1 so they win issue slots against
other waves' memory ops.
"""

import functools
import math as host_math

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm as _llvm
from flydsl._mlir.dialects import scf as _scf
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, const_expr, gpu, range_constexpr, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec
from flydsl.expr.utils.arith import ArithValue
from flydsl.expr.utils.arith import _to_raw as _raw

from aiter.ops.flydsl.kernels import buffer_ops, vector

from ..kernels_common import dtype_to_elem_type
from ..tensor_shim import _run_compiled, ptr_arg

KERNEL_NAME = "flash_attn_varlen_d72_gfx950"
HEAD_DIM = 72
_LOG2E = host_math.log2(host_math.e)
_WARP_SIZE = 64
_NUM_WAVES = 4
_BLOCK_SIZE = _WARP_SIZE * _NUM_WAVES
_MFMA_M = 32
_MFMA_N = 32
_ROWS_PER_WAVE = 32
_MFMA_K = 16


def _llvm_value(value):
    if hasattr(value, "ir_value") and not isinstance(value, ir.Value):
        return value.ir_value()
    return value


def _llvm_ptr_ty():
    return ir.Type.parse("!llvm.ptr")


def _pointer_to_llvm_ptr(ptr) -> ir.Value:
    ptr_i64 = arith.index_cast(T.i64, fx.ptrtoint(ptr))
    return _llvm.IntToPtrOp(_llvm_ptr_ty(), ptr_i64).result


def _lds_to_llvm_ptr_as3(ptr) -> ir.Value:
    addr_i64 = ArithValue(_raw(fx.ptrtoint(ptr))).extui(T.i64)
    return buffer_ops.create_llvm_ptr(addr_i64, address_space=3)


def _pointer_load(result_type: ir.Type, ptr: ir.Value) -> ir.Value:
    return _llvm.LoadOp(result_type, _llvm_value(ptr)).result


def _pointer_store(value: ir.Value, ptr: ir.Value):
    return _llvm.StoreOp(_llvm_value(value), _llvm_value(ptr))


def _patch_reusable_slot_specs():
    import ctypes

    from flydsl.expr.numeric import Float32

    if not hasattr(Float32, "_reusable_slot_spec"):

        @classmethod
        def _f32_slot_spec(cls, arg):
            return ctypes.c_float, lambda a: a.value if hasattr(a, "value") else a

        Float32._reusable_slot_spec = _f32_slot_spec
        Float32._reusable_ctype = ctypes.c_float


def build_fmha_fwd_d72_module(
    *,
    block_m=128,
    block_n=64,
    head_dim_pad=96,
    qk_mfma_k=16,
    prefetch_kv=False,
    waves_per_eu=2,
    num_k_bufs=1,
    daz=True,
    unsafe_fp_math=True,
    fast_fp_math=True,
    post_misched=False,
    dma_hd_only=False,
    next_k_after_qk=True,
    vwait_keep_k=False,
    lds_stride_pad=0,
    k_xor=False,
):
    """Build the gfx950 hd=72 varlen FA2 launcher (32x32x16 + ds_read_tr)."""
    del prefetch_kv
    del qk_mfma_k
    assert block_m % _ROWS_PER_WAVE == 0, "block_m must be a multiple of 32 Q rows/wave"
    assert block_m >= _ROWS_PER_WAVE
    assert block_n % _MFMA_N == 0
    assert block_n >= _MFMA_N
    assert head_dim_pad % _MFMA_K == 0
    assert head_dim_pad % _MFMA_N == 0
    assert head_dim_pad >= HEAD_DIM
    assert int(num_k_bufs) in (1, 2)
    assert int(lds_stride_pad) >= 0
    assert int(lds_stride_pad) % 8 == 0
    if vwait_keep_k:
        assert next_k_after_qk, "vwait_keep_k requires next_k_after_qk"

    BLOCK_M = int(block_m)
    BLOCK_N = int(block_n)
    HEAD_DIM_PAD = int(head_dim_pad)
    WARP_SIZE = _WARP_SIZE
    NUM_WAVES = BLOCK_M // _ROWS_PER_WAVE
    BLOCK_SIZE = WARP_SIZE * NUM_WAVES
    MFMA_M = _MFMA_M
    MFMA_N = _MFMA_N
    MFMA_K = _MFMA_K
    ROWS_PER_WAVE = _ROWS_PER_WAVE
    M_REPEAT = ROWS_PER_WAVE // MFMA_M
    N_REPEAT = BLOCK_N // MFMA_N
    # hd=72 only needs ceil(72/16)=5 QK K-steps; the last pad-16 is Q=0.
    K_STEPS_QK_COMPUTE = (HEAD_DIM + MFMA_K - 1) // MFMA_K
    D_CHUNK = MFMA_N
    D_CHUNKS = HEAD_DIM_PAD // D_CHUNK
    C_ELEMS = 16
    NUM_O_ACCS = M_REPEAT * D_CHUNKS
    PV_K_STEPS = BLOCK_N // MFMA_K
    VEC = 8
    K_XOR = bool(k_xor)
    # K LDS is width 72 (hd only). V stays 96 so ds_read_tr D-chunk 64-95
    # is 32-aligned. ~21.5 KB total fits 3 workgroups in 64 KB LDS.
    # Column xor can land past 72 (e.g. 64 xor 40 = 104); pad those rows to 128.
    K_STRIDE = (128 if K_XOR else HEAD_DIM) + int(lds_stride_pad)
    V_STRIDE = HEAD_DIM_PAD
    VECS_PER_ROW_K = HEAD_DIM // VEC
    VECS_PER_ROW_V = HEAD_DIM_PAD // VEC
    NUM_K_VECS = BLOCK_N * VECS_PER_ROW_K
    NUM_V_VECS = BLOCK_N * VECS_PER_ROW_V
    NUM_KV_ITERS = (NUM_K_VECS + BLOCK_SIZE - 1) // BLOCK_SIZE
    NUM_V_ITERS = (NUM_V_VECS + BLOCK_SIZE - 1) // BLOCK_SIZE
    LDS_K_TILE = BLOCK_N * K_STRIDE
    LDS_V_TILE = BLOCK_N * V_STRIDE
    NUM_K_BUFS = int(num_k_bufs)
    NUM_V_BUFS = 1
    NEXT_K_AFTER_QK = bool(next_k_after_qk)
    VWAIT_KEEP_K = bool(vwait_keep_k)

    @fx.struct
    class SharedStorage:
        k: fx.Array[fx.BFloat16, NUM_K_BUFS * LDS_K_TILE, 16]
        v: fx.Array[fx.BFloat16, NUM_V_BUFS * LDS_V_TILE, 16]

    kernel_sym = (
        f"{KERNEL_NAME}_bm{BLOCK_M}_bn{BLOCK_N}_pad{HEAD_DIM_PAD}"
        f"_m32_swk_tr_kb{NUM_K_BUFS}_vb{NUM_V_BUFS}_wpe{int(waves_per_eu)}_pm{int(post_misched)}"
        f"_hd{int(bool(dma_hd_only))}_nk{int(NEXT_K_AFTER_QK)}"
        f"_vk{int(VWAIT_KEEP_K)}_sp{int(lds_stride_pad)}_xor{int(K_XOR)}_v43"
    )

    @flyc.kernel(known_block_size=[BLOCK_SIZE, 1, 1], name=kernel_sym)
    def fmha_fwd_d72_kernel(
        Q: fx.Pointer,
        K: fx.Pointer,
        V: fx.Pointer,
        O: fx.Pointer,
        cu_seqlens_q: fx.Pointer,
        cu_seqlens_k: fx.Pointer,
        softmax_scale_log2e: fx.Float32,
        stride_q_seq: fx.Int32,
        stride_k_seq: fx.Int32,
        stride_v_seq: fx.Int32,
        stride_o_seq: fx.Int32,
        stride_q_head: fx.Int32,
        stride_k_head: fx.Int32,
        stride_v_head: fx.Int32,
        stride_o_head: fx.Int32,
        max_seqlen_q: fx.Int32,
        tensor_bytes: fx.Int32,
    ):
        elem_type = dtype_to_elem_type("bf16")
        i32_type = ir.IntegerType.get_signless(32)
        fm_fast = arith.FastMathFlags.fast

        def _fadd(a, b):
            return arith.addf(_raw(a), _raw(b), fastmath=fm_fast)

        def _fsub(a, b):
            return arith.subf(_raw(a), _raw(b), fastmath=fm_fast)

        def _fmul(a, b):
            return arith.mulf(_raw(a), _raw(b), fastmath=fm_fast)

        def _fmax(a, b):
            return arith.MaxNumFOp(_raw(a), _raw(b), fastmath=fm_fast).result

        def _fmax3(a, b, c):
            return _llvm.inline_asm(
                ir.F32Type.get(),
                [_raw(a), _raw(b), _raw(c)],
                "v_max3_f32 $0, $1, $2, $3",
                "=v,v,v,v",
                has_side_effects=False,
            )

        v8bf16_type = Vec.make_type(8, fx.BFloat16)
        c_zero_v16f32 = Vec.filled(16, 0.0, fx.Float32)
        c_zero_v8bf16 = Vec.filled(8, 0.0, fx.BFloat16)
        c_neg_inf = fx.Float32(float("-inf"))
        c_zero_f = fx.Float32(0.0)
        c_one_f = fx.Float32(1.0)
        c_sm = softmax_scale_log2e
        width_i32 = fx.Int32(WARP_SIZE)
        _ = NUM_WAVES
        _ = max_seqlen_q

        def mfma32(a_frag, b_frag, c_frag):
            return rocdl.mfma_f32_32x32x16_bf16(
                T.vec(16, T.f32), [a_frag, b_frag, c_frag, 0, 0, 0]
            )

        q_ptr = _pointer_to_llvm_ptr(Q)
        k_ptr = _pointer_to_llvm_ptr(K)
        v_ptr = _pointer_to_llvm_ptr(V)
        o_ptr = _pointer_to_llvm_ptr(O)
        cuq_ptr = _pointer_to_llvm_ptr(cu_seqlens_q)
        cuk_ptr = _pointer_to_llvm_ptr(cu_seqlens_k)
        _ = q_ptr
        _ = k_ptr
        _ = v_ptr
        nbytes_i64 = ArithValue(_raw(tensor_bytes)).extui(T.i64)
        q_rsrc = buffer_ops.create_buffer_resource_from_addr(
            arith.index_cast(T.i64, fx.ptrtoint(Q)),
            num_records_bytes=nbytes_i64,
        )
        k_rsrc = buffer_ops.create_buffer_resource_from_addr(
            arith.index_cast(T.i64, fx.ptrtoint(K)),
            num_records_bytes=nbytes_i64,
        )
        v_rsrc = buffer_ops.create_buffer_resource_from_addr(
            arith.index_cast(T.i64, fx.ptrtoint(V)),
            num_records_bytes=nbytes_i64,
        )

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        lds_k = lds.k.ptr
        lds_v = lds.v.ptr

        tid = fx.Index(gpu.thread_idx.x)
        wave_id = tid // WARP_SIZE
        lane = tid % WARP_SIZE
        lane32 = lane % 32
        kgrp = lane // 32
        # Swap bits 2 and 3 of lane%32 so QK C[0:8] is consecutive KV (SwizzleB).
        _lane32_i = fx.Int32(lane32)
        _t = arith.andi(
            arith.xori(
                arith.shrui(_raw(_lane32_i), _raw(fx.Int32(2))),
                arith.shrui(_raw(_lane32_i), _raw(fx.Int32(3))),
            ),
            _raw(fx.Int32(1)),
        )
        lane_sw = fx.Index(
            arith.xori(
                arith.xori(_raw(_lane32_i), arith.shli(_t, _raw(fx.Int32(2)))),
                arith.shli(_t, _raw(fx.Int32(3))),
            )
        )

        head_idx = fx.Index(gpu.block_idx.x)
        q_tile_idx = fx.Index(gpu.block_idx.y)
        batch_idx = fx.Index(gpu.block_idx.z)

        def _load_i32(ptr, idx):
            gep = buffer_ops.get_element_ptr(ptr, fx.Int64(idx), elem_type=i32_type)
            return _pointer_load(i32_type, gep)

        q_start_tok = fx.Index(_load_i32(cuq_ptr, batch_idx))
        q_end_tok = fx.Index(_load_i32(cuq_ptr, batch_idx + fx.Index(1)))
        k_start_tok = fx.Index(_load_i32(cuk_ptr, batch_idx))
        k_end_tok = fx.Index(_load_i32(cuk_ptr, batch_idx + fx.Index(1)))
        actual_q_len = q_end_tok - q_start_tok
        actual_kv_len = k_end_tok - k_start_tok
        q_tile_start = q_tile_idx * BLOCK_M

        def _store_vec(ptr, elem_idx, val):
            gep = buffer_ops.get_element_ptr(
                ptr, fx.Int64(elem_idx), elem_type=elem_type
            )
            _pointer_store(val, gep)

        sq = fx.Index(stride_q_seq)
        sk = fx.Index(stride_k_seq)
        sv = fx.Index(stride_v_seq)
        so = fx.Index(stride_o_seq)
        hq = fx.Index(stride_q_head)
        hk = fx.Index(stride_k_head)
        hv = fx.Index(stride_v_head)
        ho = fx.Index(stride_o_head)

        def q_elem(local_row, col):
            return (q_start_tok + local_row) * sq + head_idx * hq + col

        def k_elem(local_row, col):
            return (k_start_tok + local_row) * sk + head_idx * hk + col

        def v_elem(local_row, col):
            return (k_start_tok + local_row) * sv + head_idx * hv + col

        def o_elem(local_row, col):
            return (q_start_tok + local_row) * so + head_idx * ho + col

        def _wait_lgkm_barrier():
            _llvm.inline_asm(
                None,
                [],
                "s_waitcnt vmcnt(0) lgkmcnt(0)\ns_barrier",
                "",
                has_side_effects=True,
            )

        def _setprio(pri):
            _llvm.inline_asm(
                None,
                [],
                f"s_setprio {int(pri)}",
                "",
                has_side_effects=True,
            )

        def _wait_vmem_barrier():
            _llvm.inline_asm(
                None,
                [],
                "s_waitcnt vmcnt(0)\ns_barrier",
                "",
                has_side_effects=True,
            )

        def _barrier():
            _llvm.inline_asm(
                None,
                [],
                "s_barrier",
                "",
                has_side_effects=True,
            )

        def _wait_lgkmcnt(n):
            _llvm.inline_asm(
                None,
                [],
                f"s_waitcnt lgkmcnt({int(n)})",
                "",
                has_side_effects=True,
            )

        def coop_dma_kv_iter(
            rsrc, lds_base, tile_start, it, elem_fn, stride, skip_tok=False
        ):
            linear = tid + fx.Index(it * BLOCK_SIZE)
            row = linear // VECS_PER_ROW_V
            col = (linear % VECS_PER_ROW_V) * VEC
            kv_row = tile_start + row
            col_valid = arith.cmpi(
                arith.CmpIPredicate.ult, _raw(col), _raw(fx.Index(HEAD_DIM))
            )
            do_load = col_valid
            if const_expr(not skip_tok):
                tok_valid = arith.cmpi(
                    arith.CmpIPredicate.ult, _raw(kv_row), _raw(actual_kv_len)
                )
                do_load = arith.andi(tok_valid, col_valid)
                safe_row = ArithValue(tok_valid).select(kv_row, fx.Index(0))
            else:
                safe_row = kv_row
            if const_expr(NUM_V_VECS % BLOCK_SIZE != 0):
                in_range = arith.cmpi(
                    arith.CmpIPredicate.ult, _raw(linear), _raw(fx.Index(NUM_V_VECS))
                )
                do_load = arith.andi(do_load, in_range)
            byte_off = fx.Int32(elem_fn(safe_row, col)) * fx.Int32(2)
            voff = ArithValue(do_load).select(byte_off, fx.Int32(0x7FFFFFFF))
            lds_col = fx.Int32(col)
            lds_idx = fx.Int32(row) * fx.Int32(stride) + lds_col
            lds_ptr = _lds_to_llvm_ptr_as3(lds_base + lds_idx)
            rocdl.buffer_load_to_lds(rsrc, lds_ptr, voff, size_bytes=16)

        def coop_dma_k_iter(lds_base, tile_start, it, skip_tok=False):
            linear = tid + fx.Index(it * BLOCK_SIZE)
            row = linear // VECS_PER_ROW_K
            col = (linear % VECS_PER_ROW_K) * VEC
            kv_row = tile_start + row
            last_partial = const_expr(
                NUM_K_VECS % BLOCK_SIZE != 0 and it == NUM_KV_ITERS - 1
            )
            if const_expr(skip_tok):
                byte_off = fx.Int32(k_elem(kv_row, col)) * fx.Int32(2)
                voff = byte_off
            else:
                tok_valid = arith.cmpi(
                    arith.CmpIPredicate.ult, _raw(kv_row), _raw(actual_kv_len)
                )
                safe_row = ArithValue(tok_valid).select(kv_row, fx.Index(0))
                byte_off = fx.Int32(k_elem(safe_row, col)) * fx.Int32(2)
                voff = ArithValue(tok_valid).select(byte_off, fx.Int32(0x7FFFFFFF))
            lds_col = fx.Int32(col)
            if K_XOR:
                lds_col = fx.Int32(
                    arith.xori(
                        _raw(lds_col),
                        arith.shli(
                            arith.andi(_raw(fx.Int32(row)), _raw(fx.Int32(7))),
                            _raw(fx.Int32(3)),
                        ),
                    )
                )
            lds_idx = fx.Int32(row) * fx.Int32(K_STRIDE) + lds_col
            lds_ptr = _lds_to_llvm_ptr_as3(lds_base + lds_idx)
            if const_expr(last_partial):
                in_range = arith.cmpi(
                    arith.CmpIPredicate.ult, _raw(linear), _raw(fx.Index(NUM_K_VECS))
                )
                if in_range:
                    rocdl.buffer_load_to_lds(k_rsrc, lds_ptr, voff, size_bytes=16)
            else:
                rocdl.buffer_load_to_lds(k_rsrc, lds_ptr, voff, size_bytes=16)

        def coop_dma_k(tile_start, buf, skip_tok=False):
            base = lds_k
            if const_expr(NUM_K_BUFS == 2):
                base = lds_k + fx.Int32(buf) * fx.Int32(LDS_K_TILE)
            for it in range_constexpr(NUM_KV_ITERS):
                coop_dma_k_iter(base, tile_start, it, skip_tok)

        def coop_dma_v_iter(tile_start, it, v_lds, skip_tok=False):
            coop_dma_kv_iter(v_rsrc, v_lds, tile_start, it, v_elem, V_STRIDE, skip_tok)

        def coop_dma_v(tile_start, v_lds, skip_tok=False):
            for it in range_constexpr(NUM_V_ITERS):
                coop_dma_v_iter(tile_start, it, v_lds, skip_tok)

        def load_q_frag(ks, q_row, q_valid):
            d_col = fx.Index(ks * MFMA_K) + kgrp * fx.Index(8)
            col_valid = arith.cmpi(
                arith.CmpIPredicate.ult, _raw(d_col), _raw(fx.Index(HEAD_DIM))
            )
            do_load = arith.andi(_raw(q_valid), col_valid)
            safe_row = ArithValue(q_valid).select(q_row, fx.Index(0))
            g_idx = q_elem(safe_row, d_col)
            byte_off = fx.Int32(g_idx) * fx.Int32(2)
            voff = ArithValue(do_load).select(byte_off, fx.Int32(0x7FFFFFFF))
            c0 = _raw(fx.Int32(0))
            raw = rocdl.raw_ptr_buffer_load(
                ir.VectorType.get([4], i32_type),
                q_rsrc,
                _raw(voff),
                c0,
                c0,
            )
            loaded = vector.bitcast(v8bf16_type, raw)
            return ArithValue(do_load).select(loaded, c_zero_v8bf16.ir_value())

        def load_k_frag(ni, ks, k_lds):
            d_col = fx.Int32(ks * MFMA_K) + fx.Int32(kgrp) * fx.Int32(8)
            k_row = fx.Index(ni * MFMA_N) + lane_sw
            if K_XOR:
                d_col = fx.Int32(
                    arith.xori(
                        _raw(d_col),
                        arith.shli(
                            arith.andi(_raw(fx.Int32(k_row)), _raw(fx.Int32(7))),
                            _raw(fx.Int32(3)),
                        ),
                    )
                )
                lds_idx = fx.Int32(k_row) * fx.Int32(K_STRIDE) + d_col
                return fx.ptr_load(k_lds + lds_idx, result_type=v8bf16_type)
            if const_expr(ks * MFMA_K + 8 >= HEAD_DIM):
                col_ok = arith.cmpi(
                    arith.CmpIPredicate.ult, _raw(d_col), _raw(fx.Int32(HEAD_DIM))
                )
                safe_col = ArithValue(col_ok).select(d_col, fx.Int32(0))
                lds_idx = fx.Int32(k_row) * fx.Int32(K_STRIDE) + safe_col
                loaded = fx.ptr_load(k_lds + lds_idx, result_type=v8bf16_type)
                return ArithValue(col_ok).select(loaded, c_zero_v8bf16.ir_value())
            lds_idx = fx.Int32(k_row) * fx.Int32(K_STRIDE) + d_col
            return fx.ptr_load(k_lds + lds_idx, result_type=v8bf16_type)

        def load_v_a_frag(kstep, dc, v_lds):
            # Two ds_read_tr16_b64 (Double): 16-wide consecutive quads, D-half
            # from lane%32//16 so M=lane%32. K = kgrp*8 + acc*4 + 0..3.
            def _tr(acc):
                d_col = (
                    fx.Int32(dc * D_CHUNK)
                    + (fx.Int32(lane32) // fx.Int32(16)) * fx.Int32(16)
                    + (fx.Int32(lane) % fx.Int32(4)) * fx.Int32(4)
                )
                k_row = (
                    fx.Int32(kstep * MFMA_K)
                    + fx.Int32(kgrp) * fx.Int32(8)
                    + fx.Int32(acc) * fx.Int32(4)
                    + (fx.Int32(lane) % fx.Int32(16)) // fx.Int32(4)
                )
                elem_idx = k_row * fx.Int32(V_STRIDE) + d_col
                ptr = _lds_to_llvm_ptr_as3(v_lds + elem_idx)
                return rocdl.ds_read_tr16_b64(T.vec(4, T.bf16), ptr).result

            lo = Vec(_tr(0))
            hi = Vec(_tr(1))
            return Vec.from_elements(
                [lo[0], lo[1], lo[2], lo[3], hi[0], hi[1], hi[2], hi[3]],
                fx.BFloat16,
            ).ir_value()

        def reduce_max(x):
            return _fmax(x, fx.Float32(x).shuffle_xor(fx.Int32(32), width_i32))

        def reduce_sum(x):
            return _fadd(x, fx.Float32(x).shuffle_xor(fx.Int32(32), width_i32))

        def _max16(vals, acc_max):
            m0 = _fmax3(vals[0], vals[1], vals[2])
            m1 = _fmax3(vals[3], vals[4], vals[5])
            m2 = _fmax3(vals[6], vals[7], vals[8])
            m3 = _fmax3(vals[9], vals[10], vals[11])
            m4 = _fmax3(vals[12], vals[13], vals[14])
            acc_max = _fmax3(acc_max, vals[15], _fmax3(m0, m1, m2))
            return _fmax3(acc_max, m3, m4)

        def _p_to_pack8(ps):
            words = [
                rocdl.cvt_pk_bf16_f32(_raw(ps[2 * i]), _raw(ps[2 * i + 1]))
                for i in range_constexpr(4)
            ]
            packed_i32 = Vec.from_elements(words, fx.Int32).ir_value()
            return vector.bitcast(v8bf16_type, packed_i32)

        def _c_kv(i):
            # After K bit-swap: C[i] is KV = (i//8)*16 + kgrp*8 + (i%8).
            return (i // 8) * 16 + (i % 8)

        def _c_d(i):
            # PV C is unswizzled: D = 8*(i//4) + kgrp*4 + (i%4).
            return (8 * (i // 4)) % 32 + (i % 4)

        def pack_p_bfrag(s_own, k_half):
            ps = []
            base = 0 if const_expr(k_half == 0) else 8
            psum = c_zero_f
            for i in range_constexpr(8):
                p = s_own[base + i]
                ps.append(p)
                psum = _fadd(psum, p)
            return _p_to_pack8(ps), psum

        q_row = q_tile_start + wave_id * ROWS_PER_WAVE + lane32
        q_valid = arith.cmpi(arith.CmpIPredicate.ult, _raw(q_row), _raw(actual_q_len))
        first_full = arith.cmpi(
            arith.CmpIPredicate.uge, _raw(actual_kv_len), _raw(fx.Index(BLOCK_N))
        )
        # Issue first K DMA before Q HBM loads so they overlap.
        if first_full:
            coop_dma_k(fx.Index(0), fx.Int32(0), skip_tok=True)
        else:
            coop_dma_k(fx.Index(0), fx.Int32(0), skip_tok=False)
        q_frags = []
        for ks in range_constexpr(K_STEPS_QK_COMPUTE):
            q_frags.append(load_q_frag(ks, q_row, q_valid))

        init_args = [_raw(c_neg_inf), _raw(c_zero_f)]
        for _ in range_constexpr(NUM_O_ACCS):
            init_args.append(_raw(c_zero_v16f32))

        loop_results = init_args
        for kv_block_start, inner_iter_args in range(
            0, actual_kv_len, BLOCK_N, init=init_args
        ):
            m_running = inner_iter_args[0]
            l_running = inner_iter_args[1]
            o_accs = [inner_iter_args[2 + i] for i in range_constexpr(NUM_O_ACCS)]

            v_lds = lds_v
            kv_i = fx.Int32(kv_block_start) // fx.Int32(BLOCK_N)
            if NUM_V_BUFS == 2:
                v_lds = lds_v + (kv_i % fx.Int32(2)) * fx.Int32(LDS_V_TILE)
            is_first = arith.cmpi(
                arith.CmpIPredicate.eq, _raw(kv_block_start), _raw(fx.Index(0))
            )
            if is_first:
                _wait_lgkm_barrier()
            else:
                _barrier()
            k_lds = lds_k
            if const_expr(NUM_K_BUFS == 2):
                k_lds = lds_k + (kv_i % fx.Int32(2)) * fx.Int32(LDS_K_TILE)

            s_accs = [_raw(c_zero_v16f32) for _ in range(N_REPEAT)]
            _setprio(1)
            for ks in range_constexpr(K_STEPS_QK_COMPUTE):
                if const_expr(ks < NUM_V_ITERS) and is_first:
                    if first_full:
                        coop_dma_v_iter(kv_block_start, ks, v_lds, skip_tok=True)
                    else:
                        coop_dma_v_iter(kv_block_start, ks, v_lds, skip_tok=False)
                    rocdl.sched_vmem(1)
                q_pack = q_frags[ks]
                k_pack = load_k_frag(0, ks, k_lds)
                for ni in range_constexpr(N_REPEAT):
                    if const_expr(ni + 1 < N_REPEAT):
                        k_nxt = load_k_frag(ni + 1, ks, k_lds)
                        rocdl.sched_barrier(0)
                        _wait_lgkmcnt(N_REPEAT - ni - 1)
                    else:
                        rocdl.sched_barrier(0)
                        _wait_lgkmcnt(0)
                    s_accs[ni] = mfma32(k_pack, q_pack, s_accs[ni])
                    if const_expr(ni + 1 < N_REPEAT):
                        k_pack = k_nxt
                rocdl.sched_mfma(N_REPEAT)
            _setprio(0)
            if const_expr(NUM_V_ITERS > K_STEPS_QK_COMPUTE) and is_first:
                for it in range_constexpr(NUM_V_ITERS - K_STEPS_QK_COMPUTE):
                    if first_full:
                        coop_dma_v_iter(
                            kv_block_start,
                            K_STEPS_QK_COMPUTE + it,
                            v_lds,
                            skip_tok=True,
                        )
                    else:
                        coop_dma_v_iter(
                            kv_block_start,
                            K_STEPS_QK_COMPUTE + it,
                            v_lds,
                            skip_tok=False,
                        )

            next_start = kv_block_start + fx.Index(BLOCK_N)
            more_k = arith.cmpi(
                arith.CmpIPredicate.ult, _raw(next_start), _raw(actual_kv_len)
            )
            next_full = arith.cmpi(
                arith.CmpIPredicate.ule,
                _raw(next_start + fx.Index(BLOCK_N)),
                _raw(actual_kv_len),
            )
            nxt_buf = fx.Int32(0)
            if const_expr(NUM_K_BUFS == 2):
                kv_i = fx.Int32(kv_block_start) // fx.Int32(BLOCK_N)
                nxt_buf = fx.Int32(1) - (kv_i % fx.Int32(2))
            if const_expr(NEXT_K_AFTER_QK) and more_k:
                if next_full:
                    coop_dma_k(next_start, nxt_buf, skip_tok=True)
                else:
                    coop_dma_k(next_start, nxt_buf, skip_tok=False)

            kv_start_i32 = fx.Int32(kv_block_start)
            kv_len_i32 = fx.Int32(actual_kv_len)
            partial_tile = arith.cmpi(
                arith.CmpIPredicate.ugt, _raw(next_start), _raw(actual_kv_len)
            )

            def _mask_acc_ir(acc_ir, ni, kv_start, kv_len):
                acc = Vec(acc_ir)
                kv_base = (
                    kv_start + fx.Int32(ni * MFMA_N) + fx.Int32(kgrp) * fx.Int32(8)
                )
                elems = []
                for i in range_constexpr(C_ELEMS):
                    kv_col = kv_base + fx.Int32(_c_kv(i))
                    in_kv = arith.cmpi(
                        arith.CmpIPredicate.slt, _raw(kv_col), _raw(kv_len)
                    )
                    elems.append(ArithValue(in_kv).select(acc[i], c_neg_inf))
                return Vec.from_elements(elems, fx.Float32).ir_value()

            v16f32_ty = ir.VectorType.get([C_ELEMS], ir.F32Type.get())
            mask_if = _scf.IfOp(
                _raw(partial_tile),
                results_=[v16f32_ty] * N_REPEAT,
                has_else=True,
            )
            with ir.InsertionPoint(mask_if.then_block):
                _scf.YieldOp(
                    [
                        _mask_acc_ir(s_accs[ni], ni, kv_start_i32, kv_len_i32)
                        for ni in range_constexpr(N_REPEAT)
                    ]
                )
            with ir.InsertionPoint(mask_if.else_block):
                _scf.YieldOp([_raw(s_accs[ni]) for ni in range_constexpr(N_REPEAT)])

            s_by_ni = []
            local_max = c_neg_inf
            _setprio(1)
            for ni in range_constexpr(N_REPEAT):
                acc = Vec(mask_if.results[ni])
                s_raw = [acc[i] for i in range_constexpr(C_ELEMS)]
                s_by_ni.append(s_raw)
                local_max = _max16(s_raw, local_max)

            row_max = reduce_max(local_max)
            m_new = _fmax(m_running, row_max)
            corr = rocdl.exp2(
                ir.F32Type.get(),
                _raw(_fmul(_fsub(m_running, m_new), c_sm)),
            )
            neg_scaled_max = _fsub(c_zero_f, _fmul(c_sm, m_new))
            corr_vec = Vec.from_elements(
                [corr] * C_ELEMS,
                fx.Float32,
            )
            for dc in range_constexpr(D_CHUNKS):
                o_accs[dc] = (Vec(o_accs[dc]) * corr_vec).ir_value()

            for ni in range_constexpr(N_REPEAT):
                for i in range_constexpr(C_ELEMS):
                    s_by_ni[ni][i] = rocdl.exp2(
                        ir.F32Type.get(),
                        _raw(fmath.fma(s_by_ni[ni][i], _raw(c_sm), neg_scaled_max)),
                    )

            p_frags = []
            tile_sum = c_zero_f
            for ni in range_constexpr(N_REPEAT):
                for kh in range_constexpr(2):
                    pk, ps = pack_p_bfrag(s_by_ni[ni], kh)
                    p_frags.append(pk)
                    tile_sum = _fadd(tile_sum, ps)

            _setprio(0)
            if const_expr(NEXT_K_AFTER_QK):
                _wait_vmem_barrier()
            else:
                _wait_lgkm_barrier()
                if more_k:
                    coop_dma_k(next_start, nxt_buf)

            _setprio(1)
            v_pack = load_v_a_frag(0, 0, v_lds)
            rocdl.sched_dsrd(1)
            for kstep in range_constexpr(PV_K_STEPS):
                p_pack = p_frags[kstep]
                for dc in range_constexpr(D_CHUNKS):
                    if const_expr(dc + 1 < D_CHUNKS):
                        v_nxt = load_v_a_frag(kstep, dc + 1, v_lds)
                        rocdl.sched_dsrd(1)
                        rocdl.sched_barrier(0)
                        _wait_lgkmcnt(2)
                    elif const_expr(kstep + 1 < PV_K_STEPS):
                        v_nxt = load_v_a_frag(kstep + 1, 0, v_lds)
                        rocdl.sched_dsrd(1)
                        rocdl.sched_barrier(0)
                        _wait_lgkmcnt(2)
                    else:
                        rocdl.sched_barrier(0)
                        _wait_lgkmcnt(0)
                    o_accs[dc] = mfma32(v_pack, p_pack, o_accs[dc])
                    if const_expr((dc + 1 < D_CHUNKS) or (kstep + 1 < PV_K_STEPS)):
                        v_pack = v_nxt
                rocdl.sched_mfma(D_CHUNKS)

            _setprio(0)
            l_new = _fadd(_fmul(corr, l_running), reduce_sum(tile_sum))
            if more_k:
                _wait_lgkm_barrier()
                if next_full:
                    coop_dma_v(next_start, v_lds, skip_tok=True)
                else:
                    coop_dma_v(next_start, v_lds, skip_tok=False)
            loop_results = yield [m_new, l_new] + list(o_accs)

        o_finals = [loop_results[2 + i] for i in range_constexpr(NUM_O_ACCS)]
        l_final = loop_results[1]
        l_pos = arith.cmpf(arith.CmpFPredicate.OGT, _raw(l_final), _raw(c_zero_f))
        inv_l = arith.divf(_raw(c_one_f), _raw(l_final), fastmath=fm_fast)
        inv_l = ArithValue(l_pos).select(inv_l, c_zero_f)
        if q_valid:
            for dc in range_constexpr(D_CHUNKS):
                acc = Vec(o_finals[dc])
                n_grp = 4 if const_expr((dc + 1) * D_CHUNK <= HEAD_DIM) else 1
                for g in range_constexpr(n_grp):
                    i0 = g * 4
                    packed = Vec.from_elements(
                        [
                            fx.Float32(_fmul(acc[i0 + j], inv_l)).to(fx.BFloat16)
                            for j in range_constexpr(4)
                        ],
                        fx.BFloat16,
                    )
                    d_abs = (
                        fx.Int32(dc * D_CHUNK)
                        + fx.Int32(kgrp) * fx.Int32(4)
                        + fx.Int32(_c_d(i0))
                    )
                    _store_vec(o_ptr, o_elem(q_row, fx.Index(d_abs)), packed.ir_value())

    @flyc.jit
    def launch_fmha_fwd_d72(
        Q: fx.Pointer,
        K: fx.Pointer,
        V: fx.Pointer,
        O: fx.Pointer,
        cu_seqlens_q: fx.Pointer,
        cu_seqlens_k: fx.Pointer,
        softmax_scale_log2e: fx.Float32,
        stride_q_seq: fx.Int32,
        stride_k_seq: fx.Int32,
        stride_v_seq: fx.Int32,
        stride_o_seq: fx.Int32,
        stride_q_head: fx.Int32,
        stride_k_head: fx.Int32,
        stride_v_head: fx.Int32,
        stride_o_head: fx.Int32,
        max_seqlen_q: fx.Int32,
        batch_size: fx.Int32,
        num_heads: fx.Int32,
        tensor_bytes: fx.Int32,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        ctx = CompilationContext.get_current()
        _ = kernel_sym
        num_q_tiles = (fx.Index(max_seqlen_q) + fx.Index(BLOCK_M - 1)) // fx.Index(
            BLOCK_M
        )
        grid_x = arith.index_cast(T.index, num_heads)
        grid_y = arith.index_cast(T.index, num_q_tiles)
        grid_z = arith.index_cast(T.index, batch_size)
        launcher = fmha_fwd_d72_kernel(
            Q,
            K,
            V,
            O,
            cu_seqlens_q,
            cu_seqlens_k,
            softmax_scale_log2e,
            stride_q_seq,
            stride_k_seq,
            stride_v_seq,
            stride_o_seq,
            stride_q_head,
            stride_k_head,
            stride_v_head,
            stride_o_head,
            max_seqlen_q,
            tensor_bytes,
        )
        if const_expr(waves_per_eu is not None):
            _wpe = int(waves_per_eu)
            if const_expr(_wpe >= 1):
                for op in ctx.gpu_module_body.operations:
                    if const_expr(getattr(op, "OPERATION_NAME", None) == "gpu.func"):
                        op.attributes["rocdl.waves_per_eu"] = ir.IntegerAttr.get(
                            T.i32, _wpe
                        )
        passthrough_entries = []
        if const_expr(daz):
            passthrough_entries.append(
                ir.ArrayAttr.get(
                    [
                        ir.StringAttr.get("denormal-fp-math-f32"),
                        ir.StringAttr.get("preserve-sign,preserve-sign"),
                    ]
                )
            )
            passthrough_entries.append(
                ir.ArrayAttr.get(
                    [
                        ir.StringAttr.get("no-nans-fp-math"),
                        ir.StringAttr.get("true"),
                    ]
                )
            )
            if const_expr(unsafe_fp_math):
                passthrough_entries.append(
                    ir.ArrayAttr.get(
                        [
                            ir.StringAttr.get("unsafe-fp-math"),
                            ir.StringAttr.get("true"),
                        ]
                    )
                )
        for op in ctx.gpu_module_body.operations:
            if const_expr(getattr(op, "OPERATION_NAME", None) == "gpu.func"):
                op.attributes["passthrough"] = ir.ArrayAttr.get(passthrough_entries)
        launcher.launch(
            grid=(grid_x, grid_y, grid_z),
            block=(BLOCK_SIZE, 1, 1),
            stream=stream,
        )

    launch_fmha_fwd_d72.compile_hints = {
        "fast_fp_math": fast_fp_math,
        "unsafe_fp_math": unsafe_fp_math,
        "llvm_options": {
            "enable-post-misched": bool(post_misched),
            "lsr-drop-solution": True,
        },
    }

    def _launch(*args, stream=None, **kwargs):
        if stream is None:
            stream = kwargs.pop("stream", None)
        if stream is not None and not isinstance(stream, fx.Stream):
            stream = fx.Stream(stream)
        if stream is None:
            stream = fx.Stream(None)
        _run_compiled(launch_fmha_fwd_d72, *args, stream)

    return _launch


@functools.lru_cache(maxsize=64)
def _get_launcher(
    block_m: int,
    block_n: int,
    head_dim_pad: int,
    qk_mfma_k: int,
    prefetch_kv: bool,
    waves_per_eu: int,
    num_k_bufs: int,
    post_misched: bool,
    dma_hd_only: bool,
    next_k_after_qk: bool,
    vwait_keep_k: bool,
    lds_stride_pad: int,
    k_xor: bool,
):
    _patch_reusable_slot_specs()
    return build_fmha_fwd_d72_module(
        block_m=block_m,
        block_n=block_n,
        head_dim_pad=head_dim_pad,
        qk_mfma_k=qk_mfma_k,
        prefetch_kv=prefetch_kv,
        waves_per_eu=waves_per_eu,
        num_k_bufs=num_k_bufs,
        post_misched=post_misched,
        dma_hd_only=dma_hd_only,
        next_k_after_qk=next_k_after_qk,
        vwait_keep_k=vwait_keep_k,
        lds_stride_pad=lds_stride_pad,
        k_xor=k_xor,
    )


def flash_attn_varlen_d72_gfx950(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale=None,
    causal=False,
    out=None,
    return_lse=False,
    *,
    block_m: int = 128,
    block_n: int = 64,
    head_dim_pad: int = 96,
    qk_mfma_k: int = 16,
    prefetch_kv: bool = False,
    waves_per_eu: int = 2,
    num_k_bufs: int = 1,
    post_misched: bool = False,
    dma_hd_only: bool = False,
    next_k_after_qk: bool = True,
    vwait_keep_k: bool = False,
    lds_stride_pad: int = 0,
    k_xor: bool = False,
):
    """Run gfx950 hd=72 varlen FA2. q/k/v are THD bf16 with last dim 72."""
    del max_seqlen_k
    if causal:
        raise ValueError("flash_attn_varlen_d72_gfx950 v0 is non-causal only")
    if return_lse:
        raise ValueError("flash_attn_varlen_d72_gfx950 v0 does not return LSE")
    assert q.dtype == torch.bfloat16
    assert k.dtype == torch.bfloat16
    assert v.dtype == torch.bfloat16
    assert q.shape[-1] == HEAD_DIM
    assert k.shape[-1] == HEAD_DIM
    assert v.shape[-1] == HEAD_DIM
    if q.shape[1] != k.shape[1] or q.shape[1] != v.shape[1]:
        raise ValueError("v0 does not support GQA")

    nheads_q = q.shape[1]
    batch = int(cu_seqlens_q.shape[0] - 1)
    if softmax_scale is None:
        softmax_scale = 1.0 / (HEAD_DIM**0.5)
    scale_log2e = float(softmax_scale) * _LOG2E
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    cu_seqlens_q = cu_seqlens_q.contiguous()
    cu_seqlens_k = cu_seqlens_k.contiguous()
    if out is None:
        out = torch.empty_like(q)
    else:
        out = out.contiguous()

    launch = _get_launcher(
        int(block_m),
        int(block_n),
        int(head_dim_pad),
        int(qk_mfma_k),
        bool(prefetch_kv),
        int(waves_per_eu),
        int(num_k_bufs),
        bool(post_misched),
        bool(dma_hd_only),
        bool(next_k_after_qk),
        bool(vwait_keep_k),
        int(lds_stride_pad),
        bool(k_xor),
    )
    launch(
        ptr_arg(q),
        ptr_arg(k),
        ptr_arg(v),
        ptr_arg(out),
        ptr_arg(cu_seqlens_q),
        ptr_arg(cu_seqlens_k),
        fx.Float32(scale_log2e),
        int(q.stride(0)),
        int(k.stride(0)),
        int(v.stride(0)),
        int(out.stride(0)),
        int(q.stride(1)),
        int(k.stride(1)),
        int(v.stride(1)),
        int(out.stride(1)),
        int(max_seqlen_q),
        int(batch),
        int(nheads_q),
        int(max(q.numel(), k.numel(), v.numel()) * q.element_size()),
        stream=torch.cuda.current_stream(),
    )
    return out
