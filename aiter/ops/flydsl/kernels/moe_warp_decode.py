# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL warp-decode MoE gate_up kernel (SILOTIGER-667).

One GPU wavefront (64 lanes) per output scalar inter[token_b, expert_k, neuron_j].
No MFMA.  Designed for decode batch sizes B=1..4 where MFMA tiles are >=75% empty.

Grid:  (B * TOPK * INTER,)   — one block per neuron
Block: (64,)                 — one wavefront

Weight dtype variants (selected at compile time via w_dtype):
  w_dtype="bf16"  BF16 x BF16 weights.  Scaffold path; runs on gfx942 + gfx950.
  w_dtype="fp8"   BF16 x FP8 weights.   Matches CK gate_bf16_d2.  gfx950 only.
                  Uses v_cvt_scalef32_pk_bf16_fp8 + v_dot2_f32_bf16 (use_dot2=True
                  enforced).  PerTensor weight scale passed as f32 at launch.

Accumulation path (w_dtype="bf16" only):
  use_dot2=False  BF16->FP32 via bitshift + scalar FMA.  gfx942 + gfx950.
  use_dot2=True   v_dot2_f32_bf16 inline asm.  gfx950 only.

Public API
----------
compile_wd_moe_gate_up(**kwargs) -> JitFunction
    Returns a @flyc.jit launcher.

    Launch signature (same for all variants):
        launch(_ptr(inter_out), _ptr(x), _ptr(w_gate), _ptr(w_up),
               _ptr(router_ids), B, topk, inter, hidden, experts,
               w_scale,   # float: PerTensor weight scale (use 1.0 for bf16 path)
               stream)

    Tensor layouts (all row-major, contiguous):
        inter_out : [B*TOPK, INTER]   bf16
        x         : [B, HIDDEN]       bf16
        w_gate    : [E*INTER, HIDDEN] bf16 or fp8 (OCP E4M3)
        w_up      : [E*INTER, HIDDEN] bf16 or fp8 (OCP E4M3)
        router_ids: [B*TOPK]          i32
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
    """Unpack one BF16 from a packed-pair i32 and widen to FP32 via bitshift.

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


def _fp8x2_to_bf16(fp8_word_i32, scale_f32, sel: int):
    """Convert one packed FP8 pair to packed BF16 (gfx950 only).

    v_cvt_scalef32_pk_bf16_fp8 dst, src, scale [op_sel:[1,0,0]]
      src   : i32 holding 4 packed FP8 (OCP E4M3) values
      scale : f32 multiplier applied during conversion
      sel=0 : convert fp8[0], fp8[1]  (low  pair, bytes [1:0]) — default, no op_sel
      sel=1 : convert fp8[2], fp8[3]  (high pair, bytes [3:2]) — op_sel:[1,0,0]
      dst   : i32 holding 2 packed BF16 results
    """
    # byte_sel is encoded via the op_sel modifier (VOP3 field), not as an operand.
    asm = (
        "v_cvt_scalef32_pk_bf16_fp8 $0, $1, $2"
        if sel == 0
        else "v_cvt_scalef32_pk_bf16_fp8 $0, $1, $2 op_sel:[1,0,0]"
    )
    return llvm.inline_asm(
        T.i32,
        [fp8_word_i32, scale_f32],
        asm,
        "=v,v,v",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


def _vmcnt0():
    """Wait for all pending VMEM load operations to complete.

    Must be inserted between the load phase and compute phase of a batched
    load/compute pattern.  The AMDGPU SIInsertWaitcnts LLVM pass does not
    analyse inline asm register operands, so callers of _dot2_batched must
    manually emit this barrier after issuing all loads.
    """
    llvm.InlineAsmOp(
        res=None,
        operands_=[],
        asm_string="s_waitcnt vmcnt(0)",
        constraints="",
        has_side_effects=True,
        is_align_stack=False,
    )


def _dot2_batched(acc_f32, a_i32, b_i32):
    """v_dot2_f32_bf16 for use AFTER an explicit s_waitcnt vmcnt(0).

    No embedded wait (loads are already guaranteed ready by the caller's
    _vmcnt0() call).  No embedded s_nop — designed for use with independent
    accumulators; caller must emit one rocdl.s_nop(2) drain before reading
    any accumulator produced by this function.

    Replaces the old _dot2_dep which embedded both a per-call s_waitcnt
    (serialising loads) and a per-call s_nop (unnecessary with independent
    accumulators).  The batched pattern issues all buffer_loads → _vmcnt0()
    → all _dot2_batched() → one rocdl.s_nop(2) → sum accumulators.
    This allows the GPU memory controller to have _k_pairs * 3 VMEM ops
    in flight simultaneously instead of just 2, recovering ~2× HBM utilisation.
    """
    return llvm.inline_asm(
        T.f32,
        [acc_f32, a_i32, b_i32],
        "v_dot2_f32_bf16 $0, $2, $3, $1",
        "=v,v,v,v",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


def _dot2_dep(acc_f32, a_i32, b_i32):
    """v_dot2_f32_bf16 in legacy dependent-chain form.

    Kept for the down_reduce kernel which uses a scf.ForOp carried
    accumulator with loads interleaved across iterations.  For the gate_up
    inner loop use _dot2_batched + _vmcnt0 instead.
    """
    result = llvm.inline_asm(
        T.f32,
        [acc_f32, a_i32, b_i32],
        "s_waitcnt vmcnt(0)\nv_dot2_f32_bf16 $0, $2, $3, $1",
        "=v,v,v,v",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    rocdl.s_nop(2)
    return result


def _butterfly_reduce(val_f32):
    """6-step XOR butterfly: all 64 lanes end up with the total sum."""
    av = ArithValue(val_f32)
    for stage in range(6):
        peer = av.shuffle_xor(1 << stage, _WAVE_SIZE)
        av = ArithValue(arith.addf(av, ArithValue(peer)))
    return av


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
    w_dtype: str = "bf16",
    use_dot2: bool = False,
):
    """Compile warp-decode gate_up and return a @flyc.jit launch wrapper.

    Parameters
    ----------
    hidden    : HIDDEN dimension (contraction axis).
    inter     : INTER dimension (output neurons per expert).
    experts   : E, number of experts.
    topk      : top-K experts per token.
    k_vector  : elements per lane per K-step.  Must satisfy:
                hidden % (WAVE_SIZE * k_vector) == 0.
                Default 8 (one 128-bit load for BF16 or FP8).
    w_dtype   : "bf16" (default) or "fp8" (OCP E4M3, gfx950 only).
                FP8 enforces use_dot2=True automatically.
    use_dot2  : if True, use v_dot2_f32_bf16 (gfx950 only).
                if False, use scalar FP32 path (gfx942 + gfx950).
                Ignored (forced True) when w_dtype="fp8".
    """
    if w_dtype not in ("bf16", "fp8"):
        raise ValueError(f"w_dtype must be 'bf16' or 'fp8', got {w_dtype!r}")
    if hidden % (_WAVE_SIZE * k_vector) != 0:
        raise ValueError(
            f"hidden={hidden} must be divisible by WAVE_SIZE*k_vector"
            f"={_WAVE_SIZE * k_vector}"
        )

    # FP8 weights always require dot2 (v_cvt_scalef32_pk_bf16_fp8 is gfx950-only)
    _use_dot2 = True if w_dtype == "fp8" else use_dot2
    _w_bytes = 1 if w_dtype == "fp8" else 2  # bytes per weight element

    # For FP8: k_vector elements = k_vector bytes = k_vector//4 i32 words of weight.
    # For BF16: k_vector elements = k_vector*2 bytes = k_vector//2 i32 words of weight.
    # Activation always BF16: k_vector elements = k_vector//2 i32 words.
    _k_pairs = k_vector // 2  # activation i32 pairs per lane per K-step
    # FP8: each weight i32 covers 4 elements → k_vector//4 words → each word feeds 2 pairs
    _k_fp8_words = k_vector // 4  # only used for fp8 path
    _k_step = _WAVE_SIZE * k_vector  # K-elements per loop iteration
    _n_k_steps = hidden // _k_step

    w_tag = "fp8" if w_dtype == "fp8" else ("d2" if _use_dot2 else "f32")
    module_name = (
        f"wd_gate_up_h{hidden}_i{inter}_e{experts}_topk{topk}" f"_kv{k_vector}_w{w_tag}"
    )

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
        f32_w_scale: fx.Float32,  # PerTensor weight scale (1.0 for bf16 path)
    ):
        f32 = T.f32
        i32 = T.i32
        i64 = T.i64
        bf16 = T.bf16

        B_i32 = i32_B.ir_value()
        TOPK_i32 = i32_TOPK.ir_value()
        INTER_i32 = i32_INTER.ir_value()
        HIDDEN_i32 = i32_HIDDEN.ir_value()
        w_scale = f32_w_scale.ir_value()

        # ── Buffer resources ─────────────────────────────────────────────────
        def _rsrc(ptr, nbytes_i32):
            addr64 = arith.index_cast(i64, fx.ptrtoint(ptr))
            return buffer_ops.create_buffer_resource_from_addr(
                addr64, num_records_bytes=nbytes_i32
            )

        max_slots = B_i32 * TOPK_i32
        x_rsrc = _rsrc(arg_x, max_slots * HIDDEN_i32 * arith.constant(2, type=i32))
        rid_rsrc = _rsrc(arg_router_ids, max_slots * arith.constant(4, type=i32))
        out_rsrc = _rsrc(
            arg_inter_out, max_slots * INTER_i32 * arith.constant(2, type=i32)
        )

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

        # ── Per-row weight resources (i64 to avoid i32 overflow) ─────────────
        # For DeepSeek-V3: (255*2048+2047)*7168 = 3.76B > INT32_MAX.
        HIDDEN_i64 = arith.extsi(i64, HIDDEN_i32)
        INTER_i64 = arith.extsi(i64, INTER_i32)
        expert_i64 = arith.extsi(i64, expert_e)
        neuron_i64 = arith.extsi(i64, neuron_j)
        # row byte offset = (expert*INTER + neuron) * HIDDEN * bytes_per_elem
        w_row_byte_off = (
            (expert_i64 * INTER_i64 + neuron_i64)
            * HIDDEN_i64
            * arith.constant(_w_bytes, type=i64)
        )
        # intra-row range: HIDDEN * bytes_per_elem (fits i32; max 14336 for DeepSeek-V3 FP8)
        row_nb = HIDDEN_i32 * arith.constant(_w_bytes, type=i32)

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
        c_four = arith.constant(4, type=i32)
        x_row_base = token_b * HIDDEN_i32  # BF16 element offset

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

            k_base = step_i32 * c_k_step  # K-element start of this tile
            lane_k = k_base + lane_kV  # this lane's element start

            # Batched load/compute pattern: issue ALL loads → one vmcnt → compute.
            # Profiling showed per-pair s_waitcnt vmcnt(0) (the old _dot2_dep) kept
            # HBM utilisation at only 44% vs CK's 97%.  With all _k_pairs loads
            # in-flight simultaneously, the HBM controller can pipeline requests.
            #
            # ── Phase 1: issue all buffer loads ──────────────────────────────────
            # Pre-initialise all list containers so they are always in scope.
            # FlyDSL's AST rewriter visits all branches of if/elif/else for type
            # inference, so every variable referenced in any branch must be defined
            # before the first conditional.
            g_fp8_words: list = []
            u_fp8_words: list = []
            x_words_fp8: list = []
            x_words_bf16: list = []
            g_words_bf16: list = []
            u_words_bf16: list = []

            if w_dtype == "fp8":
                for wi in range_constexpr(_k_fp8_words):
                    wi4 = arith.constant(wi * 4, type=i32)
                    w_i32_off = (lane_k + wi4) // c_four
                    g_fp8_words.append(
                        buffer_ops.buffer_load(
                            wg_row_rsrc, w_i32_off, vec_width=1, dtype=i32
                        )
                    )
                    u_fp8_words.append(
                        buffer_ops.buffer_load(
                            wu_row_rsrc, w_i32_off, vec_width=1, dtype=i32
                        )
                    )
                    for sel in range_constexpr(2):
                        pair_elem = arith.constant((wi * 2 + sel) * 2, type=i32)
                        x_off = (x_row_base + lane_k + pair_elem) // c_two
                        x_words_fp8.append(
                            buffer_ops.buffer_load(
                                x_rsrc, x_off, vec_width=1, dtype=i32
                            )
                        )
            else:
                x_words_bf16 = []
                g_words_bf16 = []
                u_words_bf16 = []
                for p in range_constexpr(_k_pairs):
                    p2 = arith.constant(p * 2, type=i32)
                    x_words_bf16.append(
                        buffer_ops.buffer_load(
                            x_rsrc,
                            (x_row_base + lane_k + p2) // c_two,
                            vec_width=1,
                            dtype=i32,
                        )
                    )
                    if _use_dot2:
                        g_words_bf16.append(
                            buffer_ops.buffer_load(
                                wg_row_rsrc,
                                (lane_k + p2) // c_two,
                                vec_width=1,
                                dtype=i32,
                            )
                        )
                        u_words_bf16.append(
                            buffer_ops.buffer_load(
                                wu_row_rsrc,
                                (lane_k + p2) // c_two,
                                vec_width=1,
                                dtype=i32,
                            )
                        )

            # ── Phase 2: one vmcnt to wait for all loads ──────────────────────────
            if _use_dot2:
                _vmcnt0()

            # ── Phase 3: compute with independent accumulators ───────────────────
            # Each of the _k_pairs accumulator slots (g_slots[i], u_slots[i]) is
            # independent — no write→read hazard between them, so no s_nop between
            # consecutive dot2 calls.  One drain s_nop 2 covers all before summing.
            zero_f32 = arith.constant(0.0, type=f32)
            # Pre-initialise g_cur/u_cur so the AST rewriter can resolve them
            # from scf.YieldOp even when visiting branches where they're not set.
            g_cur = g_acc
            u_cur = u_acc

            if w_dtype == "fp8":
                g_slots = []
                u_slots = []
                for wi in range_constexpr(_k_fp8_words):
                    for sel in range_constexpr(2):
                        idx = wi * 2 + sel
                        x_word = x_words_fp8[idx]
                        g_bf16 = _fp8x2_to_bf16(g_fp8_words[wi], w_scale, sel)
                        u_bf16 = _fp8x2_to_bf16(u_fp8_words[wi], w_scale, sel)
                        g_slots.append(_dot2_batched(zero_f32, x_word, g_bf16))
                        u_slots.append(_dot2_batched(zero_f32, x_word, u_bf16))
                rocdl.s_nop(2)  # one drain covers all slots
                g_partial = g_slots[0]
                u_partial = u_slots[0]
                for i in range(1, _k_pairs):
                    g_partial = arith.addf(g_partial, g_slots[i])
                    u_partial = arith.addf(u_partial, u_slots[i])
                g_cur = arith.addf(g_acc, g_partial)
                u_cur = arith.addf(u_acc, u_partial)

            elif _use_dot2:
                g_slots = []
                u_slots = []
                for p in range_constexpr(_k_pairs):
                    g_slots.append(
                        _dot2_batched(zero_f32, x_words_bf16[p], g_words_bf16[p])
                    )
                    u_slots.append(
                        _dot2_batched(zero_f32, x_words_bf16[p], u_words_bf16[p])
                    )
                rocdl.s_nop(2)  # one drain covers all slots
                g_partial = g_slots[0]
                u_partial = u_slots[0]
                for i in range(1, _k_pairs):
                    g_partial = arith.addf(g_partial, g_slots[i])
                    u_partial = arith.addf(u_partial, u_slots[i])
                g_cur = arith.addf(g_acc, g_partial)
                u_cur = arith.addf(u_acc, u_partial)

            else:
                # FP32 scalar path: loads are handled by LLVM's SIInsertWaitcnts,
                # no explicit vmcnt needed.
                g_cur = g_acc
                u_cur = u_acc
                for p in range_constexpr(_k_pairs):
                    x_word = x_words_bf16[p]
                    g_word = buffer_ops.buffer_load(
                        wg_row_rsrc,
                        (lane_k + arith.constant(p * 2, type=i32)) // c_two,
                        vec_width=1,
                        dtype=i32,
                    )
                    u_word = buffer_ops.buffer_load(
                        wu_row_rsrc,
                        (lane_k + arith.constant(p * 2, type=i32)) // c_two,
                        vec_width=1,
                        dtype=i32,
                    )
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
        f32_w_scale: fx.Float32,
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
            f32_w_scale,
        ).launch(grid=(grid_x, 1, 1), block=(_WAVE_SIZE, 1, 1), stream=stream)

    return _launch


# ---------------------------------------------------------------------------
# Stage 2: down_reduce
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=None)
def compile_wd_moe_down_reduce(
    *,
    hidden: int,
    inter: int,
    experts: int,
    topk: int,
    k_vector: int = 8,
    use_dot2: bool = False,
):
    """Compile warp-decode down_reduce and return a @flyc.jit launch wrapper.

    One wavefront (64 lanes) per output channel Y[token_b, out_j].
    Contracts over INTER (inner) then TOPK (expert slots, outer).
    Epilogue: FP32 atomic-add into y_out (caller must zero-init y_out).

    Parameters
    ----------
    hidden   : HIDDEN dimension (output size, e.g. 7168 for DeepSeek-V3).
    inter    : INTER dimension (contraction axis, e.g. 2048).
    experts  : E, number of experts.
    topk     : top-K experts per token (compile-time; enables range_constexpr).
    k_vector : BF16 elements per lane per K-step (default 8).
    use_dot2 : if True, use v_dot2_f32_bf16 (gfx950 only);
               if False, use FP32 scalar path (gfx942 + gfx950).

    Launch signature:
        exe(_ptr(y_out), _ptr(inter_states), _ptr(w_down), _ptr(router_ids),
            _ptr(router_wts), B, topk, inter, hidden, experts, stream)

    Tensor layouts (all row-major, contiguous):
        y_out        : [B, HIDDEN]       f32  output (zero-init before launch!)
        inter_states : [B*TOPK, INTER]   bf16 (gate_up output)
        w_down       : [E*HIDDEN, INTER] bf16 weight rows
        router_ids   : [B*TOPK]          i32  expert index per slot
        router_wts   : [B*TOPK]          f32  router weight per slot
    """
    if hidden % (_WAVE_SIZE * k_vector) != 0:
        raise ValueError(
            f"hidden={hidden} must be divisible by WAVE_SIZE*k_vector"
            f"={_WAVE_SIZE * k_vector}"
        )
    if inter % (_WAVE_SIZE * k_vector) != 0:
        raise ValueError(
            f"inter={inter} must be divisible by WAVE_SIZE*k_vector"
            f"={_WAVE_SIZE * k_vector}"
        )

    dot2_tag = "_dot2" if use_dot2 else "_f32"
    module_name = (
        f"wd_down_h{hidden}_i{inter}_e{experts}_topk{topk}_kv{k_vector}{dot2_tag}"
    )

    _k_pairs = k_vector // 2
    _k_step = _WAVE_SIZE * k_vector  # INTER elements per loop iteration
    _n_k_steps = inter // _k_step

    @flyc.kernel(name=module_name, known_block_size=[_WAVE_SIZE, 1, 1])
    def _kernel(
        arg_y_out: fx.Pointer,  # [B, HIDDEN] f32, zero-init
        arg_inter: fx.Pointer,  # [B*TOPK, INTER] bf16
        arg_w_down: fx.Pointer,  # [E*HIDDEN, INTER] bf16
        arg_router_ids: fx.Pointer,  # [B*TOPK] i32
        arg_router_wts: fx.Pointer,  # [B*TOPK] f32
        i32_B: fx.Int32,
        i32_TOPK: fx.Int32,
        i32_INTER: fx.Int32,
        i32_HIDDEN: fx.Int32,
        i32_E: fx.Int32,
    ):
        f32 = T.f32
        i32 = T.i32
        i64 = T.i64

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

        max_slots = B_i32 * TOPK_i32
        inter_rsrc = _rsrc(
            arg_inter, max_slots * INTER_i32 * arith.constant(2, type=i32)
        )
        rid_rsrc = _rsrc(arg_router_ids, max_slots * arith.constant(4, type=i32))
        rwt_rsrc = _rsrc(arg_router_wts, max_slots * arith.constant(4, type=i32))
        # y_out: [B, HIDDEN] f32 — use -1 sentinel (unbounded) since B*HIDDEN can be large
        y_rsrc = _rsrc(arg_y_out, arith.constant(-1, type=i32))
        # w_down: per-row resources built per slot below (E*HIDDEN*INTER can exceed i32)

        # ── Block → (out_j, token_b) ─────────────────────────────────────────
        lane_i32 = arith.index_cast(i32, gpu.thread_id("x"))
        blk_i32 = arith.index_cast(i32, gpu.block_id("x"))

        out_j = arith.remui(blk_i32, HIDDEN_i32)
        token_b = arith.divui(blk_i32, HIDDEN_i32)

        # ── Outer loop: TOPK slots ────────────────────────────────────────────
        # Each slot: router_wt * sum_over_INTER(inter[slot] · W_down[expert, out_j])
        # Accumulate weighted slot sums into total_acc.
        # Both the k-step loop (over INTER) and the pair loop are compile-time
        # unrolled with range_constexpr to keep everything in one flat basic block
        # per slot — this avoids v_dot2_f32_bf16 write latency crossing loop
        # back-edges and the resulting correctness hazard.
        lane_kV = lane_i32 * arith.constant(k_vector, type=i32)
        c_two = arith.constant(2, type=i32)
        zero_f32 = arith.constant(0.0, type=f32)
        zero_i32 = arith.constant(0, type=i32)

        HIDDEN_i64 = arith.extsi(i64, HIDDEN_i32)
        INTER_i64 = arith.extsi(i64, INTER_i32)
        out_j_i64 = arith.extsi(i64, out_j)
        row_nb = INTER_i32 * arith.constant(2, type=i32)

        # Outer for: slot = 0..TOPK  (carries total_acc as f32)
        outer_for = scf.ForOp(
            arith.constant(0, index=True),
            arith.index_cast(ir.IndexType.get(), TOPK_i32),
            arith.constant(1, index=True),
            iter_args=[zero_f32],
        )
        with ir.InsertionPoint(outer_for.body):
            slot_idx = arith.index_cast(i32, outer_for.induction_variable)
            total_acc = outer_for.inner_iter_args[0]

            flat_slot = token_b * TOPK_i32 + slot_idx
            expert_e = buffer_ops.buffer_load(
                rid_rsrc, flat_slot, vec_width=1, dtype=i32
            )
            router_wt = buffer_ops.buffer_load(
                rwt_rsrc, flat_slot, vec_width=1, dtype=f32
            )

            # Per-row w_down resource (i64 to avoid i32 overflow for large shapes).
            # DeepSeek-V3: (255*7168+7167)*2048*2 = 3.75 GB > INT32_MAX.
            expert_i64 = arith.extsi(i64, expert_e)
            w_row_byte_off = (
                (expert_i64 * HIDDEN_i64 + out_j_i64)
                * INTER_i64
                * arith.constant(2, type=i64)
            )
            w_base = arith.addi(
                arith.index_cast(i64, fx.ptrtoint(arg_w_down)), w_row_byte_off
            )
            w_row_rsrc = buffer_ops.create_buffer_resource_from_addr(
                w_base, num_records_bytes=row_nb
            )

            inter_row_base = flat_slot * INTER_i32  # bf16 element offset

            # Both loops compile-time unrolled: no loop-carried dot2 hazards.
            cur = zero_f32
            for k_step in range_constexpr(_n_k_steps):
                k_base_c = arith.constant(k_step * _k_step, type=i32)
                lane_k = k_base_c + lane_kV
                for p in range_constexpr(_k_pairs):
                    p2 = arith.constant(p * 2, type=i32)
                    x_off = (inter_row_base + lane_k + p2) // c_two
                    w_off = (lane_k + p2) // c_two
                    x_word = buffer_ops.buffer_load(
                        inter_rsrc, x_off, vec_width=1, dtype=i32
                    )
                    w_word = buffer_ops.buffer_load(
                        w_row_rsrc, w_off, vec_width=1, dtype=i32
                    )
                    if use_dot2:
                        cur = _dot2_dep(cur, x_word, w_word)
                    else:
                        x0 = _bf16_pair_to_f32(x_word, 0)
                        x1 = _bf16_pair_to_f32(x_word, 1)
                        cur = arith.addf(
                            cur,
                            arith.addf(
                                arith.mulf(x0, _bf16_pair_to_f32(w_word, 0)),
                                arith.mulf(x1, _bf16_pair_to_f32(w_word, 1)),
                            ),
                        )

            new_total = arith.addf(total_acc, arith.mulf(router_wt, cur))
            scf.YieldOp([new_total])

        total_sum_all_lanes = _butterfly_reduce(outer_for.results[0])

        # ── Epilogue: lane 0 atomic-adds FP32 result into y_out ──────────────
        lane_zero = arith.cmpi(CmpIPredicate.eq, lane_i32, arith.constant(0, type=i32))
        if_op = scf.IfOp(lane_zero)
        with ir.InsertionPoint(if_op.then_block):
            # y_out element offset (f32 = 4 bytes): token_b * HIDDEN + out_j
            out_elem_i32 = token_b * HIDDEN_i32 + out_j  # element index
            # raw_ptr_buffer_atomic_fadd: atomically y_out[out_elem] += total
            rocdl.raw_ptr_buffer_atomic_fadd(
                total_sum_all_lanes,
                y_rsrc,
                out_elem_i32 * arith.constant(4, type=i32),  # byte offset
                zero_i32,
                zero_i32,
            )
            scf.YieldOp([])

    # ── @flyc.jit launch wrapper ──────────────────────────────────────────────
    _dk = _kernel

    @flyc.jit
    def _launch_down(
        arg_y_out: fx.Pointer,
        arg_inter: fx.Pointer,
        arg_w_down: fx.Pointer,
        arg_router_ids: fx.Pointer,
        arg_router_wts: fx.Pointer,
        i32_B: fx.Int32,
        i32_TOPK: fx.Int32,
        i32_INTER: fx.Int32,
        i32_HIDDEN: fx.Int32,
        i32_E: fx.Int32,
        stream: fx.Stream,
    ):
        idx = ir.IndexType.get()
        # Grid: B * HIDDEN blocks (one wave per output channel)
        grid_x = arith.index_cast(idx, i32_B.ir_value()) * arith.index_cast(
            idx, i32_HIDDEN.ir_value()
        )
        _dk(
            arg_y_out,
            arg_inter,
            arg_w_down,
            arg_router_ids,
            arg_router_wts,
            i32_B,
            i32_TOPK,
            i32_INTER,
            i32_HIDDEN,
            i32_E,
        ).launch(grid=(grid_x, 1, 1), block=(_WAVE_SIZE, 1, 1), stream=stream)

    return _launch_down
