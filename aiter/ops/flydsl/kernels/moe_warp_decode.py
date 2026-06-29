# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL warp-decode MoE gate_up kernel (SILOTIGER-667).

One GPU wavefront (64 lanes) per output scalar inter[token_b, expert_k, neuron_j].
No MFMA.  Designed for decode batch sizes B=1..4 where MFMA tiles are >=75% empty.

Grid:  (B * TOPK * INTER,)   — one block per neuron
Block: (64,)                 — one wavefront

Two accumulation paths (selected at compile time via use_dot2):
  use_dot2=False  BF16 pairs widened to FP32 via bitshift, scalar FMA.
                  Correct on gfx942 and gfx950.  Use for correctness gating.
  use_dot2=True   v_dot2_f32_bf16 inline asm + s_nop 2 (dependent chain form).
                  gfx950 (CDNA4) only.  Use for performance benchmarking.

Public API
----------
compile_wd_moe_gate_up(**kwargs) -> JitFunction
    Returns a @flyc.jit launcher.  Call it with raw pointers produced by
    flyc.from_c_void_p(fx.Uint8, tensor.data_ptr()), not with raw tensors.

    Signature:
        launch(_ptr(inter_out), _ptr(x), _ptr(w_gate), _ptr(w_up),
               _ptr(router_ids), B, topk, inter, hidden, experts, stream)

    Tensor layouts (all row-major, contiguous):
        inter_out : [B*TOPK, INTER]   bf16  output
        x         : [B, HIDDEN]       bf16  activations
        w_gate    : [E*INTER, HIDDEN] bf16  gate weight rows
        w_up      : [E*INTER, HIDDEN] bf16  up weight rows
        router_ids: [B*TOPK]          i32   expert index per (token, topk-slot)
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, buffer_ops, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.utils.arith import ArithValue
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm, scf
from flydsl._mlir.dialects.arith import CmpIPredicate

_WAVE_SIZE = 64


# ---------------------------------------------------------------------------
# Numeric helpers
# ---------------------------------------------------------------------------


def _bf16_pair_to_f32(packed_i32, which: int):
    """Unpack one BF16 from a packed-pair i32 word and widen to FP32.

    BF16 shares the same exponent/mantissa layout as the upper 16 bits of FP32,
    so the conversion is a bitshift with no value change.

    which=0: low  16 bits (bits [15:0])
    which=1: high 16 bits (bits [31:16])
    """
    i32, f32 = T.i32, T.f32
    if which == 0:
        lo = arith.andi(packed_i32, arith.constant(0xFFFF, type=i32))
        shifted = arith.shli(lo, arith.constant(16, type=i32))
    else:
        shifted = arith.andi(packed_i32, arith.constant(0xFFFF0000, type=i32))
    return arith.bitcast(f32, shifted)


def _dot2_dep(acc_f32, a_i32, b_i32):
    """v_dot2_f32_bf16 in dependent-chain form (gfx950 only).

    dst += a[lo]*b[lo] + a[hi]*b[hi]  (FP32 accumulate, 2 MACs/lane/cycle).
    s_nop 2 covers the write->read hazard on the dependent accumulator.
    """
    result = llvm.inline_asm(
        T.f32,
        [acc_f32, a_i32, b_i32],
        "v_dot2_f32_bf16 $0, $2, $3, $1",
        "=v,v,v,v",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    rocdl.s_nop(2)
    return result


def _butterfly_reduce(val_f32):
    """6-step XOR butterfly reduction across 64 lanes -> all lanes hold the sum."""
    av = ArithValue(val_f32)
    for stage in range(6):  # offsets 1, 2, 4, 8, 16, 32
        peer = av.shuffle_xor(1 << stage, _WAVE_SIZE)
        av = ArithValue(arith.addf(av, ArithValue(peer)))
    return av  # ArithValue is an ir.Value subclass; usable directly


# ---------------------------------------------------------------------------
# Kernel + launcher
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=None)
def compile_wd_moe_gate_up(
    *,
    hidden: int,
    inter: int,
    experts: int,
    topk: int,
    k_vector: int = 8,
    use_dot2: bool = False,
):
    """Compile warp-decode gate_up and return a @flyc.jit launch wrapper.

    Parameters
    ----------
    hidden    : HIDDEN dimension (contraction axis, e.g. 7168 for DeepSeek-V3).
    inter     : INTER dimension (output neurons per expert, e.g. 2048).
    experts   : number of experts E.
    topk      : top-K experts selected per token.
    k_vector  : BF16 elements owned by each lane per K-step (default 8).
                Must divide hidden evenly when multiplied by 64 (wave size).
    use_dot2  : if True, emit v_dot2_f32_bf16 (gfx950 only);
                if False, use FP32 scalar path (gfx942 + gfx950).
    """
    if hidden % (_WAVE_SIZE * k_vector) != 0:
        raise ValueError(
            f"hidden={hidden} must be divisible by WAVE_SIZE*k_vector={_WAVE_SIZE * k_vector}"
        )

    dot2_tag = "_dot2" if use_dot2 else "_f32"
    module_name = (
        f"wd_gate_up_h{hidden}_i{inter}_e{experts}_topk{topk}_kv{k_vector}{dot2_tag}"
    )

    _k_pairs = k_vector // 2  # BF16 pairs per lane per K-step
    _k_step = _WAVE_SIZE * k_vector  # BF16 elements consumed per loop iteration
    _n_k_steps = hidden // _k_step

    @flyc.kernel(name=module_name, known_block_size=[_WAVE_SIZE, 1, 1])
    def _kernel(
        arg_inter_out: fx.Pointer,
        arg_x: fx.Pointer,
        arg_w_gate: fx.Pointer,
        arg_w_up: fx.Pointer,
        arg_router_ids: fx.Pointer,
        i32_B: fx.Int32,
        i32_TOPK: fx.Int32,
        i32_INTER: fx.Int32,
        i32_HIDDEN: fx.Int32,
        i32_E: fx.Int32,
    ):
        f32 = T.f32
        i32 = T.i32
        i64 = T.i64
        bf16 = T.bf16

        B_i32 = i32_B.ir_value()
        TOPK_i32 = i32_TOPK.ir_value()
        INTER_i32 = i32_INTER.ir_value()
        HIDDEN_i32 = i32_HIDDEN.ir_value()

        # ── Buffer resources ─────────────────────────────────────────────────
        def _rsrc(ptr, nbytes_i32):
            addr64 = arith.index_cast(i64, fx.ptrtoint(ptr))
            return buffer_ops.create_buffer_resource_from_addr(
                addr64, num_records_bytes=nbytes_i32
            )

        max_slots = B_i32 * TOPK_i32  # B*TOPK, fits i32 at decode sizes
        x_rsrc = _rsrc(arg_x, max_slots * HIDDEN_i32 * arith.constant(2, type=i32))
        rid_rsrc = _rsrc(arg_router_ids, max_slots * arith.constant(4, type=i32))
        out_rsrc = _rsrc(
            arg_inter_out, max_slots * INTER_i32 * arith.constant(2, type=i32)
        )
        # Weight resources are created per-row below (row byte offset can exceed i32).

        # ── Block → (neuron_j, expert_k, token_b) ───────────────────────────
        lane_i32 = arith.index_cast(i32, gpu.thread_id("x"))
        blk_i32 = arith.index_cast(i32, gpu.block_id("x"))

        neuron_j = arith.remui(blk_i32, INTER_i32)
        blk_div = arith.divui(blk_i32, INTER_i32)
        expert_k = arith.remui(blk_div, TOPK_i32)
        token_b = arith.divui(blk_div, TOPK_i32)

        expert_e = buffer_ops.buffer_load(
            rid_rsrc, token_b * TOPK_i32 + expert_k, vec_width=1, dtype=i32
        )

        # ── Per-row weight resources (i64 address to avoid i32 overflow) ────
        # For large shapes (e.g. DeepSeek-V3 E=256, INTER=2048, HIDDEN=7168),
        # (expert_e * INTER + neuron_j) * HIDDEN can exceed 2^31.  Compute the
        # row byte address in i64, bump the base pointer, then use a small i32
        # intra-row offset (max = HIDDEN*2 bytes = 14336 for DeepSeek-V3).
        HIDDEN_i64 = arith.extsi(i64, HIDDEN_i32)
        INTER_i64 = arith.extsi(i64, INTER_i32)
        expert_i64 = arith.extsi(i64, expert_e)
        neuron_i64 = arith.extsi(i64, neuron_j)
        w_row_byte_off = (
            (expert_i64 * INTER_i64 + neuron_i64)
            * HIDDEN_i64
            * arith.constant(2, type=i64)
        )

        row_nb = HIDDEN_i32 * arith.constant(2, type=i32)  # bytes in one row, fits i32
        wg_base = arith.addi(
            arith.index_cast(i64, fx.ptrtoint(arg_w_gate)), w_row_byte_off
        )
        wu_base = arith.addi(
            arith.index_cast(i64, fx.ptrtoint(arg_w_up)), w_row_byte_off
        )
        wg_row_rsrc = buffer_ops.create_buffer_resource_from_addr(
            wg_base, num_records_bytes=row_nb
        )
        wu_row_rsrc = buffer_ops.create_buffer_resource_from_addr(
            wu_base, num_records_bytes=row_nb
        )

        # ── K-loop ───────────────────────────────────────────────────────────
        lane_kV = lane_i32 * arith.constant(k_vector, type=i32)
        c_k_step = arith.constant(_k_step, type=i32)
        c_two = arith.constant(2, type=i32)
        x_row_base = token_b * HIDDEN_i32  # i32 BF16 element offset into x

        for_op = scf.ForOp(
            arith.constant(0, index=True),
            arith.constant(_n_k_steps, index=True),
            arith.constant(1, index=True),
            iter_args=[arith.constant(0.0, type=f32), arith.constant(0.0, type=f32)],
        )
        with ir.InsertionPoint(for_op.body):
            step_i32 = arith.index_cast(i32, for_op.induction_variable)
            g_acc = for_op.inner_iter_args[0]
            u_acc = for_op.inner_iter_args[1]

            k_base = step_i32 * c_k_step
            lane_k = k_base + lane_kV

            g_cur, u_cur = g_acc, u_acc
            for p in range_constexpr(_k_pairs):
                p2 = arith.constant(p * 2, type=i32)
                w_off = (lane_k + p2) // c_two  # i32 word offset within row
                x_off = (x_row_base + lane_k + p2) // c_two

                x_word = buffer_ops.buffer_load(x_rsrc, x_off, vec_width=1, dtype=i32)
                g_word = buffer_ops.buffer_load(
                    wg_row_rsrc, w_off, vec_width=1, dtype=i32
                )
                u_word = buffer_ops.buffer_load(
                    wu_row_rsrc, w_off, vec_width=1, dtype=i32
                )

                if use_dot2:
                    g_cur = _dot2_dep(g_cur, x_word, g_word)
                    u_cur = _dot2_dep(u_cur, x_word, u_word)
                else:
                    x0 = _bf16_pair_to_f32(x_word, 0)
                    x1 = _bf16_pair_to_f32(x_word, 1)
                    g_cur = arith.addf(
                        g_cur,
                        arith.addf(
                            arith.mulf(x0, _bf16_pair_to_f32(g_word, 0)),
                            arith.mulf(x1, _bf16_pair_to_f32(g_word, 1)),
                        ),
                    )
                    u_cur = arith.addf(
                        u_cur,
                        arith.addf(
                            arith.mulf(x0, _bf16_pair_to_f32(u_word, 0)),
                            arith.mulf(x1, _bf16_pair_to_f32(u_word, 1)),
                        ),
                    )
            scf.YieldOp([g_cur, u_cur])

        gate_sum = _butterfly_reduce(for_op.results[0])
        up_sum = _butterfly_reduce(for_op.results[1])

        # ── Epilogue: lane 0 writes silu(gate) * up as BF16 ─────────────────
        lane_zero = arith.cmpi(CmpIPredicate.eq, lane_i32, arith.constant(0, type=i32))
        if_op = scf.IfOp(lane_zero)
        with ir.InsertionPoint(if_op.then_block):
            # silu(x) = x * sigmoid(x);  fast: exp2(-log2(e) * x) for the exponent
            t = arith.mulf(gate_sum, arith.constant(-1.4426950408889634, type=f32))
            sig = rocdl.rcp(
                f32, arith.addf(arith.constant(1.0, type=f32), rocdl.exp2(f32, t))
            )
            out_f32 = arith.mulf(arith.mulf(gate_sum, sig), up_sum)
            out_elem = (token_b * TOPK_i32 + expert_k) * INTER_i32 + neuron_j
            buffer_ops.buffer_store(arith.truncf(bf16, out_f32), out_rsrc, out_elem)
            scf.YieldOp([])

    # ── @flyc.jit launch wrapper ─────────────────────────────────────────────
    _k = _kernel

    @flyc.jit
    def _launch(
        arg_inter_out: fx.Pointer,
        arg_x: fx.Pointer,
        arg_w_gate: fx.Pointer,
        arg_w_up: fx.Pointer,
        arg_router_ids: fx.Pointer,
        i32_B: fx.Int32,
        i32_TOPK: fx.Int32,
        i32_INTER: fx.Int32,
        i32_HIDDEN: fx.Int32,
        i32_E: fx.Int32,
        stream: fx.Stream,
    ):
        idx = ir.IndexType.get()
        grid_x = (
            arith.index_cast(idx, i32_B.ir_value())
            * arith.index_cast(idx, i32_TOPK.ir_value())
            * arith.index_cast(idx, i32_INTER.ir_value())
        )
        _k(
            arg_inter_out,
            arg_x,
            arg_w_gate,
            arg_w_up,
            arg_router_ids,
            i32_B,
            i32_TOPK,
            i32_INTER,
            i32_HIDDEN,
            i32_E,
        ).launch(grid=(grid_x, 1, 1), block=(_WAVE_SIZE, 1, 1), stream=stream)

    return _launch
