# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""One-pass MoE gather-reduce (weighted) epilogue kernel (FlyDSL).

Background
----------
After the per-expert stage2 GEMM, MoE output lives in a grouped layout
``grouped_out (E, max_m, model_dim)``: expert ``e`` holds its routed tokens in
rows ``[0, counts[e])``.  The final epilogue scatters those rows back to the
flat per-token output, multiplying by the route weight and summing the ``topk``
contributions of each token::

    moe_out[t] = sum_k  w(t,k) * grouped_out[expert(t,k), pos(t,k)]

The Python reference does this as a per-expert loop of ``index_add_`` (scatter).
This kernel reformulates it as a **gather-reduce**: one thread-block per output
token gathers that token's ``topk`` source rows (via a precomputed inverse index
map), weights them, and sums them in registers in a single pass.  No atomics, so
the result is deterministic and order-independent like ``index_add_``.

Layout / grid
-------------
Inputs (all on device):
  grouped_out_flat : (E*max_m, model_dim)  bf16/f16   -- grouped_out viewed flat
  topids_to_rows         : (token_num, topk)     i32        -- flat source row per (t,k)
  gather_w         : (token_num, topk)     f32        -- weight per (t,k)
  out              : (token_num, model_dim) bf16/f16

Grid  : (token_num, 1, 1)   -- one block per output token
Block : (BLOCK_THREADS, 1, 1)

Each thread owns 4 consecutive dwords (16 B = 8 elements) and issues
``buffer_load``/``buffer_store`` at ``vec_width=4``.  Wider transactions raise
per-request bytes and cut the in-flight loads needed to saturate HBM.  When the
row's dword count is not a multiple of 4 the trailing group falls back to a
per-lane scalar tail (mirroring ``compile_moe_reduction`` in
``moe_gemm_2stage.py``), so any even ``model_dim`` is supported.  Unused (t,k)
slots are filled with row 0 and weight 0 by the host wrapper, so they contribute
nothing and need no branch.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, range_constexpr, vector
from flydsl.expr.typing import T, Int32
from flydsl.expr.arith import ArithValue, CmpIPredicate
from flydsl.compiler.kernel_function import CompilationContext

from flydsl._mlir import ir
from flydsl._mlir.dialects import scf
from flydsl.expr import buffer_ops

from aiter.ops.flydsl.kernels.tensor_shim import ptr_rsrc, MOE_KERNARG_PRELOAD_COUNT

BLOCK_THREADS = 256


def _unpack_pair_to_f32(raw_dw, out_dtype, *, f32, i32):
    """Unpack a dword holding 2 packed bf16/f16 elements into (lo_f32, hi_f32)."""
    mask16 = arith.constant(0xFFFF, type=i32)
    lo16 = raw_dw & mask16
    hi16 = (raw_dw >> arith.constant(16, type=i32)) & mask16
    if out_dtype == "bf16":
        # bf16 -> f32 is just the bf16 bits in the upper half of the f32.
        lo = arith.bitcast(f32, lo16 << arith.constant(16, type=i32))
        hi = arith.bitcast(f32, hi16 << arith.constant(16, type=i32))
    else:  # f16
        lo = arith.extf(f32, arith.bitcast(T.f16, arith.trunci(T.i16, lo16)))
        hi = arith.extf(f32, arith.bitcast(T.f16, arith.trunci(T.i16, hi16)))
    return ArithValue(lo), ArithValue(hi)


def _pack_pair_from_f32(acc_lo, acc_hi, out_dtype, *, i32):
    """Truncate two f32 accumulators to bf16/f16 and pack into one dword."""
    odt = T.bf16 if out_dtype == "bf16" else T.f16
    lo_i16 = arith.bitcast(T.i16, arith.trunc_f(odt, acc_lo))
    hi_i16 = arith.bitcast(T.i16, arith.trunc_f(odt, acc_hi))
    lo_i32 = arith.extui(i32, lo_i16)
    hi_i32 = arith.extui(i32, hi_i16)
    return lo_i32 | (hi_i32 << arith.constant(16, type=i32))


def build_moe_gather_reduce_module(
    model_dim: int,
    topk: int,
    out_dtype: str = "bf16",
    split_k: int = 1,
    vec_dwords: int = 2,
):
    """Return a JIT launcher for the one-pass MoE gather-reduce epilogue.

    Each thread owns ``vec_dwords`` consecutive dwords and loads/stores at
    ``vec_width=vec_dwords``.  Rows whose dword count is not a multiple of VEC process the
    trailing partial group through a per-lane scalar tail, so any even
    ``model_dim`` is supported.

    Parameters
    ----------
    model_dim : int   output columns (must be even; 2 elems per dword)
    topk      : int   number of expert contributions summed per token
    out_dtype : str   "bf16" or "f16" (input and output share this dtype)
    split_k   : int   number of split-K slices in grouped_out_flat
    vec_dwords: int   dwords per thread (2 or 4)
    """
    assert model_dim % 2 == 0, "model_dim must be even (2 elems per dword)"
    assert out_dtype in ("bf16", "f16")
    if vec_dwords not in (2, 4):
        raise ValueError(f"vec_dwords must be 2 or 4, got {vec_dwords}")
    # Smaller per-thread groups increase column-grid parallelism for tiny token
    # batches (e.g. token=1 decode) and reduce per-CTA split_k/topk work.
    VEC = int(vec_dwords)
    out_dwords = model_dim // 2  # dwords per output row
    row_dwords = model_dim // 2  # dwords per grouped_out_flat row (same)
    DWORDS_PER_ITER = BLOCK_THREADS * VEC  # dwords advanced per loop iter
    n_iters = (out_dwords + DWORDS_PER_ITER - 1) // DWORDS_PER_ITER

    module_name = (
        f"moe_gather_reduce_{out_dtype}_d{model_dim}_tk{topk}_sk{split_k}_v{VEC}"
    )

    @flyc.kernel(name=module_name)
    def moe_gather_reduce_kernel(
        grouped_out_flat: fx.Pointer,
        topids_to_rows: fx.Pointer,
        gather_w: fx.Pointer,
        out: fx.Pointer,
        num_tokens: Int32,
        slice_stride_dw: Int32,
    ):
        bid = fx.block_idx.x
        tid = fx.thread_idx.x

        f32 = T.f32
        i32 = T.i32
        vec_i32_ty = T.vec(VEC, i32)
        w_dt = T.bf16 if out_dtype == "bf16" else T.f16  # weight native dtype

        out_dwords_i32 = arith.constant(out_dwords, type=i32)
        topk_i32 = arith.constant(topk, type=i32)
        row_dwords_i32 = arith.constant(row_dwords, type=i32)
        vec_i32 = arith.constant(VEC, type=i32)
        num_tokens_i32 = ArithValue(num_tokens)
        bid_i32 = ArithValue(bid)
        slice_stride_dw_i32 = ArithValue(slice_stride_dw)

        tok_valid = arith.cmpi(CmpIPredicate.ult, bid_i32, num_tokens_i32)
        _if_tok = scf.IfOp(tok_valid)
        with ir.InsertionPoint(_if_tok.then_block):
            in_rsrc = ptr_rsrc(grouped_out_flat)
            rows_rsrc = ptr_rsrc(topids_to_rows)
            w_rsrc = ptr_rsrc(gather_w)
            out_rsrc = ptr_rsrc(out)
            thread_id = ArithValue(tid)
            iter_idx_i32 = ArithValue(fx.block_idx.y)

            # Base dword offset of this token's row in topids_to_rows / gather_w
            # (both are (token_num, topk), 1 dword per element).
            map_base = bid_i32 * topk_i32
            out_row_dw_base = bid_i32 * out_dwords_i32

            def _load_row_weight(k):
                """Load (source grouped row, route weight as f32) for slot k."""
                map_off = map_base + arith.constant(k, type=i32)
                row_i32 = ArithValue(
                    buffer_ops.buffer_load(rows_rsrc, map_off, vec_width=1, dtype=i32)
                )
                # weight is bf16/f16 (its native dtype); extend to f32
                w_f32 = ArithValue(
                    arith.extf(
                        f32,
                        buffer_ops.buffer_load(
                            w_rsrc, map_off, vec_width=1, dtype=w_dt
                        ),
                    )
                )
                return row_i32, w_f32

            dw_base = thread_id * vec_i32 + iter_idx_i32 * arith.constant(
                DWORDS_PER_ITER, type=i32
            )
            dw_valid = arith.cmpi(CmpIPredicate.ult, dw_base, out_dwords_i32)
            _if_dw = scf.IfOp(dw_valid)
            with ir.InsertionPoint(_if_dw.then_block):
                full_valid = arith.cmpi(
                    CmpIPredicate.ule, dw_base + vec_i32, out_dwords_i32
                )
                _if_full = scf.IfOp(full_valid, has_else=True)
                with ir.InsertionPoint(_if_full.then_block):
                    acc = [
                        ArithValue(arith.constant(0.0, type=f32))
                        for _ in range(2 * VEC)
                    ]

                    for k in range_constexpr(topk):
                        row_i32, w_f32 = _load_row_weight(k)
                        src_dw_base = row_i32 * row_dwords_i32 + dw_base
                        red = [
                            ArithValue(arith.constant(0.0, type=f32))
                            for _ in range(2 * VEC)
                        ]
                        for sk in range_constexpr(split_k):
                            sk_off = arith.constant(sk, type=i32) * slice_stride_dw_i32
                            raw_vec = buffer_ops.buffer_load(
                                in_rsrc,
                                src_dw_base + sk_off,
                                vec_width=VEC,
                                dtype=i32,
                            )
                            for lane in range_constexpr(VEC):
                                raw_dw = ArithValue(
                                    vector.extract(
                                        raw_vec,
                                        static_position=[lane],
                                        dynamic_position=[],
                                    )
                                )
                                lo_f32, hi_f32 = _unpack_pair_to_f32(
                                    raw_dw, out_dtype, f32=f32, i32=i32
                                )
                                red[2 * lane] = red[2 * lane] + lo_f32
                                red[2 * lane + 1] = red[2 * lane + 1] + hi_f32
                        for lane in range_constexpr(VEC):
                            acc[2 * lane] = acc[2 * lane] + w_f32 * red[2 * lane]
                            acc[2 * lane + 1] = (
                                acc[2 * lane + 1] + w_f32 * red[2 * lane + 1]
                            )

                    packed = [
                        _pack_pair_from_f32(
                            acc[2 * lane], acc[2 * lane + 1], out_dtype, i32=i32
                        )
                        for lane in range(VEC)
                    ]
                    out_vec = vector.from_elements(vec_i32_ty, packed)
                    buffer_ops.buffer_store(
                        out_vec, out_rsrc, out_row_dw_base + dw_base
                    )
                    scf.YieldOp([])
                with ir.InsertionPoint(_if_full.else_block):
                    for lane in range_constexpr(VEC):
                        dw_idx = dw_base + arith.constant(lane, type=i32)
                        lane_valid = arith.cmpi(
                            CmpIPredicate.ult, dw_idx, out_dwords_i32
                        )
                        _if_lane = scf.IfOp(lane_valid)
                        with ir.InsertionPoint(_if_lane.then_block):
                            acc_lo = ArithValue(arith.constant(0.0, type=f32))
                            acc_hi = ArithValue(arith.constant(0.0, type=f32))
                            for k in range_constexpr(topk):
                                row_i32, w_f32 = _load_row_weight(k)
                                src_dw_base = row_i32 * row_dwords_i32 + dw_idx
                                red_lo = ArithValue(arith.constant(0.0, type=f32))
                                red_hi = ArithValue(arith.constant(0.0, type=f32))
                                for sk in range_constexpr(split_k):
                                    sk_off = (
                                        arith.constant(sk, type=i32)
                                        * slice_stride_dw_i32
                                    )
                                    raw_dw = ArithValue(
                                        buffer_ops.buffer_load(
                                            in_rsrc,
                                            src_dw_base + sk_off,
                                            vec_width=1,
                                            dtype=i32,
                                        )
                                    )
                                    lo_f32, hi_f32 = _unpack_pair_to_f32(
                                        raw_dw, out_dtype, f32=f32, i32=i32
                                    )
                                    red_lo = red_lo + lo_f32
                                    red_hi = red_hi + hi_f32
                                acc_lo = acc_lo + w_f32 * red_lo
                                acc_hi = acc_hi + w_f32 * red_hi

                            packed = _pack_pair_from_f32(
                                acc_lo, acc_hi, out_dtype, i32=i32
                            )
                            buffer_ops.buffer_store(
                                packed, out_rsrc, out_row_dw_base + dw_idx
                            )
                            scf.YieldOp([])
                    scf.YieldOp([])
                scf.YieldOp([])
            scf.YieldOp([])

    @flyc.jit
    def launch_moe_gather_reduce(
        grouped_out_flat: fx.Pointer,
        topids_to_rows: fx.Pointer,
        gather_w: fx.Pointer,
        out: fx.Pointer,
        num_tokens: fx.Int32,
        slice_stride_dw: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            pass

        idx_tokens = arith.index_cast(T.index, num_tokens)
        launcher = moe_gather_reduce_kernel(
            grouped_out_flat,
            topids_to_rows,
            gather_w,
            out,
            num_tokens,
            slice_stride_dw,
        )
        launcher.launch(
            grid=(idx_tokens, n_iters, 1),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    launch_moe_gather_reduce.compile_hints = {
        "llvm_options": {
            "amdgpu-kernarg-preload": True,
            "amdgpu-kernarg-preload-count": MOE_KERNARG_PRELOAD_COUNT,
        },
    }
    return launch_moe_gather_reduce
