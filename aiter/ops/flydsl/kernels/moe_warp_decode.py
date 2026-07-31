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
from flydsl.expr.gpu import barrier as gpu_barrier, get_dyn_shared
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


def _fp4x2_to_bf16(fp4_word_i32, scale_f32, sel: int):
    """Convert one packed FP4 pair to packed BF16 (gfx950 only).

    v_cvt_scalef32_pk_bf16_fp4 dst, src, scale [op_sel_modifier]
      src   : i32 holding 8 packed FP4 (E2M1) values
      scale : f32 multiplier applied during conversion
      sel=0 → fp4[0], fp4[1]  (byte 0, bits  [7: 0])  — no op_sel
      sel=1 → fp4[2], fp4[3]  (byte 1, bits [15: 8])  — op_sel:[1,0,0]
      sel=2 → fp4[4], fp4[5]  (byte 2, bits [23:16])  — op_sel:[0,1,0]
      sel=3 → fp4[6], fp4[7]  (byte 3, bits [31:24])  — op_sel:[1,1,0]
      dst   : i32 holding 2 packed BF16 results
    """
    _OP_SEL = {0: "", 1: " op_sel:[1,0,0]", 2: " op_sel:[0,1,0]", 3: " op_sel:[1,1,0]"}
    asm = f"v_cvt_scalef32_pk_bf16_fp4 $0, $1, $2{_OP_SEL[sel]}"
    return llvm.inline_asm(
        T.i32,
        [fp4_word_i32, scale_f32],
        asm,
        "=v,v,v",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


def _lgkmcnt0():
    """s_waitcnt lgkmcnt(0): wait for all outstanding LDS (ds_read/ds_write) ops."""
    llvm.inline_asm(
        None, [], "s_waitcnt lgkmcnt(0)", "",
        has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT,
    )


def _ds_write_b32(lds_byte_addr, data_i32):
    """Write one i32 word to LDS at byte address lds_byte_addr."""
    llvm.inline_asm(
        None, [lds_byte_addr, data_i32], "ds_write_b32 $0, $1", "v,v",
        has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT,
    )


def _ds_read_b32(lds_byte_addr):
    """Read one i32 word from LDS at byte address lds_byte_addr."""
    return llvm.inline_asm(
        T.i32, [lds_byte_addr], "ds_read_b32 $0, $1", "=v,v",
        has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT,
    )


def _vmcnt_n(n: int):
    """s_waitcnt vmcnt(N) — wait until N or fewer VMEM ops are outstanding.

    Used between the 'issue current-step loads' and 'compute previous-step'
    phases of the software-prefetch pipeline.  N is the number of loads just
    issued for the current step (which we do NOT want to wait for yet).
    """
    llvm.InlineAsmOp(
        res=None,
        operands_=[],
        asm_string=f"s_waitcnt vmcnt({n})",
        constraints="",
        has_side_effects=True,
        is_align_stack=False,
    )


def _vmcnt0():
    """Wait for ALL pending VMEM loads to complete (vmcnt(0)).

    Used in the epilogue of the prefetch pipeline (after the last k-step's
    loads have been issued, before the final compute pass).
    Also used in the non-prefetch batched pattern.
    """
    _vmcnt_n(0)


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


def _dot2_acc_nonop(acc_f32, a_i32, b_i32):
    """v_dot2_f32_bf16 with TIED accumulation — no s_nop.

    Uses the "0" constraint to tie the accumulator and output to the SAME
    physical register: output ≡ accumulator ≡ vdst.  LLVM cannot alias the
    output with any OTHER live operand (a_i32, b_i32) because the output
    register is already committed to the accumulator.

    This mirrors CK's ``dot2_bf16_packed_raw_nonop`` pattern. The caller must
    emit one ``rocdl.s_nop(2)`` drain after issuing the full batch of dot2s
    (covering the write→read hazard for all accumulators at once).

    Constraint layout:
      =v   → $0  output (written, also used as accumulator via "0" tie)
      v    → $1  a_i32  (read)
      v    → $2  b_i32  (read)
      0    → $3  acc_f32 (tied to $0, i.e. same register as output)
    Asm: v_dot2_f32_bf16 vdst, src0, src1, vdst   (in-place: vdst += dot)
    """
    return llvm.inline_asm(
        T.f32,
        [a_i32, b_i32, acc_f32],
        "v_dot2_f32_bf16 $0, $1, $2, $0",
        "=v,v,v,0",
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
    n_waves: int = 1,
):
    """Compile warp-decode gate_up and return a @flyc.jit launch wrapper.

    Parameters
    ----------
    hidden    : HIDDEN dimension (contraction axis).
    inter     : INTER dimension (output neurons per expert).
    experts   : E, number of experts.
    topk      : top-K experts per token.
    k_vector  : elements per lane per K-step.
    w_dtype   : "bf16" (default), "fp8" (gfx950), or "fp4" (gfx950).
    use_dot2  : if True, use v_dot2_f32_bf16 (gfx950 only).
    n_waves   : waves per block; all waves cooperatively load x[token,:]
                into LDS so each block reads x once (reuse=n_waves×).
                Grid: B × TOPK × (INTER / n_waves).
                Reduces x-bandwidth by n_waves× — key for small-grid shapes
                (e.g. qwen3next B=1: 14× gap → ~3-4× gap expected).
                Requires hidden % (n_waves * WAVE_SIZE * 2) == 0.
    """
    if w_dtype not in ("bf16", "fp8", "fp4"):
        raise ValueError(f"w_dtype must be 'bf16', 'fp8', or 'fp4', got {w_dtype!r}")
    if hidden % (_WAVE_SIZE * k_vector) != 0:
        raise ValueError(
            f"hidden={hidden} must be divisible by WAVE_SIZE*k_vector"
            f"={_WAVE_SIZE * k_vector}"
        )
    if n_waves < 1:
        raise ValueError(f"n_waves must be >= 1, got {n_waves}")
    if n_waves > 1 and inter % n_waves != 0:
        raise ValueError(f"inter={inter} must be divisible by n_waves={n_waves}")
    if n_waves > 1 and hidden % (n_waves * _WAVE_SIZE * 2) != 0:
        raise ValueError(
            f"LDS gate_up: hidden={hidden} must be divisible by n_waves*WAVE_SIZE*2"
            f"={n_waves * _WAVE_SIZE * 2}"
        )

    # FP8/FP4 weights always require dot2 (gfx950-only conversion instructions)
    _use_dot2 = True if w_dtype in ("fp8", "fp4") else use_dot2
    # Bytes per weight element: BF16=2, FP8=1, FP4=0.5 (represented as a fraction)
    # Row buffer size uses integer arithmetic: HIDDEN*2, HIDDEN*1, or HIDDEN//2 bytes.
    _w_bytes = 2 if w_dtype == "bf16" else 1  # placeholder for row_nb computation
    _w_fp4 = w_dtype == "fp4"

    # Activation always BF16: k_vector elements = k_vector//2 i32 pairs per K-step.
    _k_pairs = k_vector // 2
    # FP8: k_vector//4 i32 words per lane (4 FP8 per i32, 2 pairs per word).
    _k_fp8_words = k_vector // 4  # only used for fp8 path
    # FP4: 1 i32 word per lane (8 FP4 per i32, 4 pairs per word).
    # _k_fp4_words = 1       (always 1 for k_vector=8)
    _k_step = _WAVE_SIZE * k_vector  # K-elements per loop iteration
    _n_k_steps = hidden // _k_step

    _block_threads_gu = n_waves * _WAVE_SIZE
    _use_lds_gu = n_waves > 1
    _lds_bytes_gu = hidden * 2 if _use_lds_gu else 0  # one BF16 x-row in LDS
    _inter_per_thread_gu = hidden // _block_threads_gu if _use_lds_gu else 0

    w_tag = w_dtype if w_dtype in ("fp8", "fp4") else ("d2" if _use_dot2 else "f32")
    nw_tag = f"_nw{n_waves}" if n_waves > 1 else ""
    module_name = (
        f"wd_gate_up_h{hidden}_i{inter}_e{experts}_topk{topk}"
        f"_kv{k_vector}_w{w_tag}{nw_tag}"
    )

    @flyc.kernel(name=module_name, known_block_size=[_block_threads_gu, 1, 1])
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
        # For FP4: row is HIDDEN/2 bytes (8 FP4 per i32 → HIDDEN/8 i32 words).
        # Pre-initialise so the AST rewriter resolves both variables in both branches.
        row_nb = arith.constant(0, type=i32)
        w_row_byte_off = arith.constant(0, type=i64)
        if _w_fp4:
            row_nb = HIDDEN_i32 // arith.constant(2, type=i32)
            w_row_byte_off = (
                (expert_i64 * INTER_i64 + neuron_i64)
                * (HIDDEN_i64 // arith.constant(2, type=i64))
            )
        else:
            row_nb = HIDDEN_i32 * arith.constant(_w_bytes, type=i32)
            w_row_byte_off = (
                (expert_i64 * INTER_i64 + neuron_i64)
                * HIDDEN_i64
                * arith.constant(_w_bytes, type=i64)
            )

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

        # ── Software-pipelined prefetch K-loop ───────────────────────────────
        # For dot2 paths (FP8 and BF16+dot2), we overlap VMEM loads with VALU
        # compute using a prologue/epilogue structure:
        #
        #   Prologue:  issue loads for k-step 0
        #   for step in 1..n_k_steps-1:
        #     A) issue loads for step (no wait)       ← NEW VMEM pipeline
        #     B) vmcnt(n_loads_per_step)              ← wait for PREV step only
        #     C) compute PREV step (from iter_args)
        #     yield [g, u, curr_loads...]
        #   Epilogue: vmcnt(0) + compute last step
        #
        # The loads for step k+1 fly in HBM while step k's dot2s execute,
        # recovering the ~30% HBM gap vs CK (which achieves this automatically
        # via LLVM's instruction scheduler on its non-asm intrinsics).
        #
        # Load order within each step:
        #   FP8 path:   [g_fp8_0,g_fp8_1, u_fp8_0,u_fp8_1, x_0,x_1,x_2,x_3]
        #   BF16 dot2:  [x_0,g_0,u_0, x_1,g_1,u_1, x_2,g_2,u_2, x_3,g_3,u_3]
        # (interleaved to allow the GPU to pipeline the address calculations)

        zero_f32 = arith.constant(0.0, type=f32)
        c1_idx = arith.constant(1, index=True)
        # Pre-initialise so the AST rewriter can resolve g_final/u_final from
        # _butterfly_reduce even when visiting the branch where they're not set.
        g_final = zero_f32
        u_final = zero_f32

        c_eight = arith.constant(8, type=i32)

        if w_dtype == "fp4":
            # ── FP4: batched load/compute (same structure as FP8 but 2× denser)
            # One i32 = 8 FP4 values per lane per k_step (vs 2 i32 for FP8).
            # v_cvt_scalef32_pk_bf16_fp4 with sel=0..3 extracts 4 BF16 pairs.
            for_op = scf.ForOp(
                arith.constant(0, index=True),
                arith.constant(_n_k_steps, index=True),
                c1_idx,
                iter_args=[zero_f32, zero_f32],
            )
            with ir.InsertionPoint(for_op.body):
                step_i32 = arith.index_cast(i32, for_op.induction_variable)
                g_acc = for_op.inner_iter_args[0]
                u_acc = for_op.inner_iter_args[1]
                lane_k = step_i32 * c_k_step + lane_kV

                # One FP4 i32 per weight row covers all k_vector=8 elements
                w_fp4_off = lane_k // c_eight
                g_fp4_word = buffer_ops.buffer_load(
                    wg_row_rsrc, w_fp4_off, vec_width=1, dtype=i32
                )
                u_fp4_word = buffer_ops.buffer_load(
                    wu_row_rsrc, w_fp4_off, vec_width=1, dtype=i32
                )
                # 4 x loads (one BF16 i32 pair per sel)
                x_words_fp4: list = []
                for sel in range_constexpr(4):
                    pair_elem = arith.constant(sel * 2, type=i32)
                    x_off = (x_row_base + lane_k + pair_elem) // c_two
                    x_words_fp4.append(
                        buffer_ops.buffer_load(x_rsrc, x_off, vec_width=1, dtype=i32)
                    )

                _vmcnt0()

                g_slots: list = []
                u_slots: list = []
                for sel in range_constexpr(4):
                    xw = x_words_fp4[sel]
                    g_slots.append(
                        _dot2_batched(
                            zero_f32, xw, _fp4x2_to_bf16(g_fp4_word, w_scale, sel)
                        )
                    )
                    u_slots.append(
                        _dot2_batched(
                            zero_f32, xw, _fp4x2_to_bf16(u_fp4_word, w_scale, sel)
                        )
                    )
                rocdl.s_nop(2)
                gp = g_slots[0]
                up = u_slots[0]
                for i in range(1, _k_pairs):
                    gp = arith.addf(gp, g_slots[i])
                    up = arith.addf(up, u_slots[i])
                g_cur = arith.addf(g_acc, gp)
                u_cur = arith.addf(u_acc, up)
                scf.YieldOp([g_cur, u_cur])

            g_final = for_op.results[0]
            u_final = for_op.results[1]

        elif w_dtype == "fp8":
            # ── FP8: batched load/compute (independent accumulators) ──────────
            # A prefetch pipeline with tied-accumulator dot2 was tested and found
            # to be SLOWER (e.g. deepseek B=1: 0.092 ms vs 0.077 ms batched).
            # Root cause: FP8 weights are already small enough that 8 loads per
            # k-step complete before the compute finishes; the 8 extra ForOp
            # iter_args from the prefetch add register pressure that outweighs
            # any latency-hiding benefit.  The batched approach (issue all loads →
            # vmcnt0 → 4 independent dot2s → 1 s_nop drain → sum) already reaches
            # near the HBM roofline and is the right path for FP8.
            for_op = scf.ForOp(
                arith.constant(0, index=True),
                arith.constant(_n_k_steps, index=True),
                c1_idx,
                iter_args=[zero_f32, zero_f32],
            )
            with ir.InsertionPoint(for_op.body):
                step_i32 = arith.index_cast(i32, for_op.induction_variable)
                g_acc = for_op.inner_iter_args[0]
                u_acc = for_op.inner_iter_args[1]
                lane_k = step_i32 * c_k_step + lane_kV

                g_fp8_words: list = []
                u_fp8_words: list = []
                x_words_fp8: list = []
                for wi in range_constexpr(_k_fp8_words):
                    wi4 = arith.constant(wi * 4, type=i32)
                    w_i32_off = (lane_k + wi4) // c_four
                    g_fp8_words.append(
                        buffer_ops.buffer_load(wg_row_rsrc, w_i32_off, vec_width=1, dtype=i32)
                    )
                    u_fp8_words.append(
                        buffer_ops.buffer_load(wu_row_rsrc, w_i32_off, vec_width=1, dtype=i32)
                    )
                    for sel in range_constexpr(2):
                        pair_elem = arith.constant((wi * 2 + sel) * 2, type=i32)
                        x_off = (x_row_base + lane_k + pair_elem) // c_two
                        x_words_fp8.append(
                            buffer_ops.buffer_load(x_rsrc, x_off, vec_width=1, dtype=i32)
                        )

                _vmcnt0()

                g_slots: list = []
                u_slots: list = []
                for wi in range_constexpr(_k_fp8_words):
                    for sel in range_constexpr(2):
                        idx = wi * 2 + sel
                        xw = x_words_fp8[idx]
                        g_slots.append(
                            _dot2_batched(zero_f32, xw, _fp8x2_to_bf16(g_fp8_words[wi], w_scale, sel))
                        )
                        u_slots.append(
                            _dot2_batched(zero_f32, xw, _fp8x2_to_bf16(u_fp8_words[wi], w_scale, sel))
                        )
                rocdl.s_nop(2)
                gp = g_slots[0]
                up = u_slots[0]
                for i in range(1, _k_pairs):
                    gp = arith.addf(gp, g_slots[i])
                    up = arith.addf(up, u_slots[i])
                g_cur = arith.addf(g_acc, gp)
                u_cur = arith.addf(u_acc, up)
                scf.YieldOp([g_cur, u_cur])

            g_final = for_op.results[0]
            u_final = for_op.results[1]

        elif _use_dot2:
            # ── BF16 dot2: software-pipelined prefetch (overlaps VMEM + VALU) ─
            # Issue next k-step's 12 loads while computing current k-step's dot2s.
            # Carries loaded i32 VGPRs as scf.ForOp iter_args across iterations.
            _n_loads = _k_pairs * 3  # 4 × (x + g + u) = 12 loads per step

            def _emit_loads_bf16d2(lane_k_i32):
                """Return 12 i32 loads for one k-step (BF16×BF16 dot2 path)."""
                ld = []
                for p in range_constexpr(_k_pairs):
                    p2c = arith.constant(p * 2, type=i32)
                    ld.append(
                        buffer_ops.buffer_load(
                            x_rsrc,
                            (x_row_base + lane_k_i32 + p2c) // c_two,
                            vec_width=1,
                            dtype=i32,
                        )
                    )
                    ld.append(
                        buffer_ops.buffer_load(
                            wg_row_rsrc,
                            (lane_k_i32 + p2c) // c_two,
                            vec_width=1,
                            dtype=i32,
                        )
                    )
                    ld.append(
                        buffer_ops.buffer_load(
                            wu_row_rsrc,
                            (lane_k_i32 + p2c) // c_two,
                            vec_width=1,
                            dtype=i32,
                        )
                    )
                return ld

            def _compute_bf16d2(prev_ld, g_acc_val, u_acc_val):
                """Compute dot2s from pre-loaded (vmcnt-cleared) BF16×BF16 values."""
                # Layout: [x0,g0,u0, x1,g1,u1, x2,g2,u2, x3,g3,u3]
                g_sl: list = []
                u_sl: list = []
                for p in range_constexpr(_k_pairs):
                    g_sl.append(
                        _dot2_batched(zero_f32, prev_ld[p * 3], prev_ld[p * 3 + 1])
                    )
                    u_sl.append(
                        _dot2_batched(zero_f32, prev_ld[p * 3], prev_ld[p * 3 + 2])
                    )
                rocdl.s_nop(2)
                gp = g_sl[0]
                up = u_sl[0]
                for i in range(1, _k_pairs):
                    gp = arith.addf(gp, g_sl[i])
                    up = arith.addf(up, u_sl[i])
                return arith.addf(g_acc_val, gp), arith.addf(u_acc_val, up)

            # Prologue: issue step-0 loads
            prologue_ld = _emit_loads_bf16d2(lane_kV)

            for_op = scf.ForOp(
                c1_idx,
                arith.constant(_n_k_steps, index=True),
                c1_idx,
                iter_args=[zero_f32, zero_f32] + prologue_ld,
            )
            with ir.InsertionPoint(for_op.body):
                step_i32 = arith.index_cast(i32, for_op.induction_variable)
                g_acc = for_op.inner_iter_args[0]
                u_acc = for_op.inner_iter_args[1]
                prev_ld = list(for_op.inner_iter_args[2:])
                lane_k_curr = step_i32 * c_k_step + lane_kV
                curr_ld = _emit_loads_bf16d2(lane_k_curr)  # A) issue next step
                _vmcnt_n(_n_loads)  # B) wait for prev step
                g_new, u_new = _compute_bf16d2(prev_ld, g_acc, u_acc)  # C) compute
                scf.YieldOp([g_new, u_new] + curr_ld)

            last_ld = list(for_op.results[2:])
            _vmcnt0()
            g_final, u_final = _compute_bf16d2(
                last_ld, for_op.results[0], for_op.results[1]
            )

        else:
            # ── f32 scalar path (gfx942 + gfx950 correctness) ────────────────
            # LLVM's SIInsertWaitcnts handles load ordering automatically for
            # regular VALU ops; no explicit vmcnt needed.
            for_op = scf.ForOp(
                arith.constant(0, index=True),
                arith.constant(_n_k_steps, index=True),
                c1_idx,
                iter_args=[zero_f32, zero_f32],
            )
            with ir.InsertionPoint(for_op.body):
                step_i32 = arith.index_cast(i32, for_op.induction_variable)
                g_acc = for_op.inner_iter_args[0]
                u_acc = for_op.inner_iter_args[1]
                lane_k = step_i32 * c_k_step + lane_kV
                g_cur = g_acc
                u_cur = u_acc
                for p in range_constexpr(_k_pairs):
                    p2 = arith.constant(p * 2, type=i32)
                    x_word = buffer_ops.buffer_load(
                        x_rsrc,
                        (x_row_base + lane_k + p2) // c_two,
                        vec_width=1,
                        dtype=i32,
                    )
                    g_word = buffer_ops.buffer_load(
                        wg_row_rsrc, (lane_k + p2) // c_two, vec_width=1, dtype=i32
                    )
                    u_word = buffer_ops.buffer_load(
                        wu_row_rsrc, (lane_k + p2) // c_two, vec_width=1, dtype=i32
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

            g_final = for_op.results[0]
            u_final = for_op.results[1]

        gate_sum = _butterfly_reduce(g_final)
        up_sum = _butterfly_reduce(u_final)

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
# gate_up split-K (two-phase, arch-agnostic, f32 path only)
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=None)
def compile_wd_moe_gate_up_splitk(
    *,
    hidden: int,
    inter: int,
    experts: int,
    topk: int,
    k_vector: int = 8,
    k_batch: int = 2,
):
    """Compile gate_up split-K phase 1: FP32 partial sums via atomicAdd.

    Splits the HIDDEN contraction across k_batch workgroups.  Each block
    computes dot(x[k_slice], w_gate/w_up[k_slice]) and atomicAdds its
    partial result into FP32 scratch buffers.

    Grid: (B * TOPK * INTER, k_batch, 1).
    Caller must zero-init gate_scratch and up_scratch before launch.
    After this kernel, call compile_wd_moe_gate_finalize() to produce BF16 inter_out.

    Parameters
    ----------
    hidden   : HIDDEN dimension.
    inter    : INTER dimension (output neurons per expert).
    experts  : E, number of experts.
    topk     : top-K experts per token.
    k_vector : elements per lane per K-step (default 8).
    k_batch  : number of split-K workgroups (≥ 2).

    Launch signature:
        exe(_ptr(gate_scratch), _ptr(up_scratch), _ptr(x), _ptr(w_gate), _ptr(w_up),
            _ptr(router_ids), B, topk, inter, hidden, experts, stream)

    Scratch buffers:
        gate_scratch : [B*TOPK*INTER] f32, zero-init before launch
        up_scratch   : [B*TOPK*INTER] f32, zero-init before launch
    """
    if hidden % (k_batch * _WAVE_SIZE * k_vector) != 0:
        raise ValueError(
            f"hidden={hidden} must be divisible by k_batch*WAVE_SIZE*k_vector"
            f"={k_batch * _WAVE_SIZE * k_vector}"
        )
    if k_batch < 2:
        raise ValueError(f"k_batch must be >= 2 for split-K, got {k_batch}")

    module_name = (
        f"wd_gate_up_sk_h{hidden}_i{inter}_e{experts}_topk{topk}"
        f"_kv{k_vector}_kb{k_batch}"
    )
    _k_pairs = k_vector // 2
    _k_step = _WAVE_SIZE * k_vector
    _n_k_steps = hidden // _k_step
    _n_k_steps_per_batch = _n_k_steps // k_batch
    @flyc.kernel(name=module_name, known_block_size=[_WAVE_SIZE, 1, 1])
    def _kernel(
        arg_gate_scratch: fx.Pointer,  # [B*TOPK*INTER] f32, zero-init
        arg_up_scratch: fx.Pointer,  # [B*TOPK*INTER] f32, zero-init
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

        B_i32 = i32_B.ir_value()
        TOPK_i32 = i32_TOPK.ir_value()
        INTER_i32 = i32_INTER.ir_value()
        HIDDEN_i32 = i32_HIDDEN.ir_value()

        def _rsrc(ptr, nbytes_i32):
            addr64 = arith.index_cast(i64, fx.ptrtoint(ptr))
            return buffer_ops.create_buffer_resource_from_addr(
                addr64, num_records_bytes=nbytes_i32
            )

        max_slots = B_i32 * TOPK_i32
        x_rsrc = _rsrc(arg_x, max_slots * HIDDEN_i32 * arith.constant(2, type=i32))
        rid_rsrc = _rsrc(arg_router_ids, max_slots * arith.constant(4, type=i32))
        scratch_nrec = max_slots * INTER_i32 * arith.constant(4, type=i32)  # f32 = 4B
        gsc_rsrc = _rsrc(arg_gate_scratch, scratch_nrec)
        usc_rsrc = _rsrc(arg_up_scratch, scratch_nrec)

        lane_i32 = arith.index_cast(i32, gpu.thread_id("x"))
        blk_i32 = arith.index_cast(i32, gpu.block_id("x"))

        neuron_j = arith.remui(blk_i32, INTER_i32)
        blk_div = arith.divui(blk_i32, INTER_i32)
        expert_k = arith.remui(blk_div, TOPK_i32)
        token_b = arith.divui(blk_div, TOPK_i32)

        expert_e = buffer_ops.buffer_load(
            rid_rsrc, token_b * TOPK_i32 + expert_k, vec_width=1, dtype=i32
        )

        # Per-row weight resources (i64 to avoid i32 overflow)
        HIDDEN_i64 = arith.extsi(i64, HIDDEN_i32)
        INTER_i64 = arith.extsi(i64, INTER_i32)
        expert_i64 = arith.extsi(i64, expert_e)
        neuron_i64 = arith.extsi(i64, neuron_j)
        w_row_byte_off = (
            (expert_i64 * INTER_i64 + neuron_i64)
            * HIDDEN_i64
            * arith.constant(2, type=i64)
        )
        row_nb = HIDDEN_i32 * arith.constant(2, type=i32)
        wg_base = arith.addi(arith.index_cast(i64, fx.ptrtoint(arg_w_gate)), w_row_byte_off)
        wu_base = arith.addi(arith.index_cast(i64, fx.ptrtoint(arg_w_up)), w_row_byte_off)
        wg_row_rsrc = buffer_ops.create_buffer_resource_from_addr(wg_base, num_records_bytes=row_nb)
        wu_row_rsrc = buffer_ops.create_buffer_resource_from_addr(wu_base, num_records_bytes=row_nb)

        # split-K offset
        k_bid_i32 = arith.index_cast(i32, gpu.block_id("y"))
        k_batch_base = k_bid_i32 * arith.constant(_n_k_steps_per_batch * _k_step, type=i32)

        lane_kV = lane_i32 * arith.constant(k_vector, type=i32)
        c_two = arith.constant(2, type=i32)
        x_row_base = token_b * HIDDEN_i32

        # f32 scalar accumulation (arch-agnostic)
        for_op = scf.ForOp(
            arith.constant(0, index=True),
            arith.constant(_n_k_steps_per_batch, index=True),
            arith.constant(1, index=True),
            iter_args=[arith.constant(0.0, type=f32), arith.constant(0.0, type=f32)],
        )
        with ir.InsertionPoint(for_op.body):
            step_i32 = arith.index_cast(i32, for_op.induction_variable)
            g_acc = for_op.inner_iter_args[0]
            u_acc = for_op.inner_iter_args[1]
            lane_k = arith.addi(k_batch_base, step_i32 * arith.constant(_k_step, type=i32)) + lane_kV
            g_cur, u_cur = g_acc, u_acc
            for p in range_constexpr(_k_pairs):
                p2 = arith.constant(p * 2, type=i32)
                x_word = buffer_ops.buffer_load(
                    x_rsrc, (x_row_base + lane_k + p2) // c_two, vec_width=1, dtype=i32
                )
                g_word = buffer_ops.buffer_load(
                    wg_row_rsrc, (lane_k + p2) // c_two, vec_width=1, dtype=i32
                )
                u_word = buffer_ops.buffer_load(
                    wu_row_rsrc, (lane_k + p2) // c_two, vec_width=1, dtype=i32
                )
                x0 = _bf16_pair_to_f32(x_word, 0)
                x1 = _bf16_pair_to_f32(x_word, 1)
                g_cur = arith.addf(g_cur, arith.addf(
                    arith.mulf(x0, _bf16_pair_to_f32(g_word, 0)),
                    arith.mulf(x1, _bf16_pair_to_f32(g_word, 1)),
                ))
                u_cur = arith.addf(u_cur, arith.addf(
                    arith.mulf(x0, _bf16_pair_to_f32(u_word, 0)),
                    arith.mulf(x1, _bf16_pair_to_f32(u_word, 1)),
                ))
            scf.YieldOp([g_cur, u_cur])

        gate_sum = _butterfly_reduce(for_op.results[0])
        up_sum = _butterfly_reduce(for_op.results[1])

        # Epilogue: lane 0 atomicAdds FP32 partials into scratch buffers
        zero_i32 = arith.constant(0, type=i32)
        lane_zero = arith.cmpi(CmpIPredicate.eq, lane_i32, zero_i32)
        if_op = scf.IfOp(lane_zero)
        with ir.InsertionPoint(if_op.then_block):
            out_elem = (token_b * TOPK_i32 + expert_k) * INTER_i32 + neuron_j
            byte_off = out_elem * arith.constant(4, type=i32)
            rocdl.raw_ptr_buffer_atomic_fadd(gate_sum, gsc_rsrc, byte_off, zero_i32, zero_i32)
            rocdl.raw_ptr_buffer_atomic_fadd(up_sum, usc_rsrc, byte_off, zero_i32, zero_i32)
            scf.YieldOp([])

    _sk = _kernel

    @flyc.jit
    def _launch_sk(
        arg_gate_scratch: fx.Pointer,
        arg_up_scratch: fx.Pointer,
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
        grid_y = arith.constant(k_batch, index=True)
        _sk(
            arg_gate_scratch, arg_up_scratch, arg_x, arg_w_gate, arg_w_up,
            arg_router_ids, i32_B, i32_TOPK, i32_INTER, i32_HIDDEN, i32_E,
        ).launch(grid=(grid_x, grid_y, 1), block=(_WAVE_SIZE, 1, 1), stream=stream)

    return _launch_sk


@functools.lru_cache(maxsize=None)
def compile_wd_moe_gate_finalize(*, inter: int, topk: int):
    """Compile gate_up split-K phase 2: silu(gate_scratch)*up_scratch → BF16.

    Reads accumulated FP32 partials from phase 1, applies silu(gate)*up,
    writes BF16 to inter_out.  One block per output neuron.

    Grid: B * TOPK * INTER blocks.
    """
    module_name = f"wd_gate_finalize_i{inter}_topk{topk}"

    @flyc.kernel(name=module_name, known_block_size=[_WAVE_SIZE, 1, 1])
    def _kernel(
        arg_inter_out: fx.Pointer,    # [B*TOPK*INTER] bf16, output
        arg_gate_scratch: fx.Pointer,  # [B*TOPK*INTER] f32, fully accumulated
        arg_up_scratch: fx.Pointer,    # [B*TOPK*INTER] f32, fully accumulated
        i32_B: fx.Int32,
        i32_TOPK: fx.Int32,
        i32_INTER: fx.Int32,
    ):
        f32 = T.f32
        i32 = T.i32
        i64 = T.i64
        bf16 = T.bf16

        B_i32 = i32_B.ir_value()
        TOPK_i32 = i32_TOPK.ir_value()
        INTER_i32 = i32_INTER.ir_value()

        def _rsrc(ptr, nbytes_i32):
            addr64 = arith.index_cast(i64, fx.ptrtoint(ptr))
            return buffer_ops.create_buffer_resource_from_addr(
                addr64, num_records_bytes=nbytes_i32
            )

        max_slots = B_i32 * TOPK_i32
        nelem = max_slots * INTER_i32
        scratch_nrec = nelem * arith.constant(4, type=i32)
        out_nrec = nelem * arith.constant(2, type=i32)
        gsc_rsrc = _rsrc(arg_gate_scratch, scratch_nrec)
        usc_rsrc = _rsrc(arg_up_scratch, scratch_nrec)
        out_rsrc = _rsrc(arg_inter_out, out_nrec)

        lane_i32 = arith.index_cast(i32, gpu.thread_id("x"))
        elem = arith.index_cast(i32, gpu.block_id("x"))  # block_id == neuron index

        # Lane 0 reads gate_scratch[elem] and up_scratch[elem], applies silu, writes BF16
        zero_i32 = arith.constant(0, type=i32)
        lane_zero = arith.cmpi(CmpIPredicate.eq, lane_i32, zero_i32)
        if_op = scf.IfOp(lane_zero)
        with ir.InsertionPoint(if_op.then_block):
            gate_val = buffer_ops.buffer_load(gsc_rsrc, elem, vec_width=1, dtype=f32)
            up_val = buffer_ops.buffer_load(usc_rsrc, elem, vec_width=1, dtype=f32)
            # silu(gate) = gate * sigmoid(gate)
            neg_log2e = arith.constant(-1.4426950408889634, type=f32)
            t = arith.mulf(gate_val, neg_log2e)
            sig = rocdl.rcp(f32, arith.addf(arith.constant(1.0, type=f32), rocdl.exp2(f32, t)))
            silu_g = arith.mulf(gate_val, sig)
            out_f32 = arith.mulf(silu_g, up_val)
            buffer_ops.buffer_store(arith.truncf(bf16, out_f32), out_rsrc, elem)
            scf.YieldOp([])

    _fin = _kernel

    @flyc.jit
    def _launch_fin(
        arg_inter_out: fx.Pointer,
        arg_gate_scratch: fx.Pointer,
        arg_up_scratch: fx.Pointer,
        i32_B: fx.Int32,
        i32_TOPK: fx.Int32,
        i32_INTER: fx.Int32,
        stream: fx.Stream,
    ):
        idx = ir.IndexType.get()
        grid_x = (
            arith.index_cast(idx, i32_B.ir_value())
            * arith.index_cast(idx, i32_TOPK.ir_value())
            * arith.index_cast(idx, i32_INTER.ir_value())
        )
        _fin(
            arg_inter_out, arg_gate_scratch, arg_up_scratch,
            i32_B, i32_TOPK, i32_INTER,
        ).launch(grid=(grid_x, 1, 1), block=(_WAVE_SIZE, 1, 1), stream=stream)

    return _launch_fin


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
    h_per_warp: int = 1,
    k_batch: int = 1,
    w_dtype: str = "bf16",
    n_waves: int = 1,
):
    """Compile warp-decode down_reduce and return a @flyc.jit launch wrapper.

    One wavefront (64 lanes) per h_per_warp output channel(s) of Y[token_b, :].
    Contracts over INTER (inner) then TOPK (expert slots, outer).
    Epilogue: FP32 atomic-add into y_out (caller must zero-init y_out).

    Parameters
    ----------
    hidden      : HIDDEN dimension (output size).
    inter       : INTER dimension (contraction axis).
    experts     : E, number of experts.
    topk        : top-K experts per token (compile-time).
    k_vector    : BF16 elements per lane per K-step (default 8).
    use_dot2    : if True, use v_dot2_f32_bf16 (gfx950 only).
    h_per_warp  : output channels per wave (1 or 2).
    k_batch     : split-K factor (default 1 = no split).
    w_dtype     : "bf16" or "fp8" (FP8 requires use_dot2=True, gfx950 only).

    Launch signature (BF16 weights):
        exe(_ptr(y_out), _ptr(inter_states), _ptr(w_down), _ptr(router_ids),
            _ptr(router_wts), B, topk, inter, hidden, experts, stream)

    Launch signature (FP8 weights):
        exe(_ptr(y_out), _ptr(inter_states), _ptr(w_down), _ptr(router_ids),
            _ptr(router_wts), B, topk, inter, hidden, experts, w_scale, stream)
    """
    if hidden % (_WAVE_SIZE * k_vector) != 0:
        raise ValueError(
            f"hidden={hidden} must be divisible by WAVE_SIZE*k_vector"
            f"={_WAVE_SIZE * k_vector}"
        )
    if inter % (k_batch * _WAVE_SIZE * k_vector) != 0:
        raise ValueError(
            f"inter={inter} must be divisible by k_batch*WAVE_SIZE*k_vector"
            f"={k_batch * _WAVE_SIZE * k_vector}"
        )
    if h_per_warp not in (1, 2):
        raise ValueError(f"h_per_warp must be 1 or 2, got {h_per_warp}")
    if hidden % h_per_warp != 0:
        raise ValueError(
            f"hidden={hidden} must be divisible by h_per_warp={h_per_warp}"
        )
    if k_batch < 1:
        raise ValueError(f"k_batch must be >= 1, got {k_batch}")
    if w_dtype not in ("bf16", "fp8", "fp4"):
        raise ValueError(f"w_dtype must be 'bf16', 'fp8', or 'fp4', got {w_dtype}")
    if w_dtype in ("fp8", "fp4") and not use_dot2:
        raise ValueError(f"w_dtype='{w_dtype}' down_reduce requires use_dot2=True (gfx950)")
    if n_waves < 1:
        raise ValueError(f"n_waves must be >= 1, got {n_waves}")
    if n_waves > 1 and k_batch > 1:
        raise ValueError("LDS mode (n_waves>1) is incompatible with split-K (k_batch>1)")
    # n_waves > 1 is compatible with FP8 weights: LDS caches inter_states
    # (BF16 activations), not the weights. Weight loading is unaffected.
    _block_threads = n_waves * _WAVE_SIZE
    if n_waves > 1 and inter % (_block_threads * 2) != 0:
        raise ValueError(
            f"LDS mode: inter={inter} must be divisible by n_waves*WAVE_SIZE*2={_block_threads * 2}"
        )

    _use_fp8_w = w_dtype == "fp8"
    _use_fp4_w = w_dtype == "fp4"
    _block_k = 32  # FP4 MXFP4 block size (elements per e8m0 scale)
    _k_fp8_words = k_vector // 4  # FP8: 4 elements per i32
    _k_fp4_words = 1               # FP4: 1 i32 = 8 FP4 per lane per k_step
    # LDS inter_states caching only applies to the f32 scalar path.
    # In the dot2 path, x_word reads come from the inner scf.ForOp which
    # currently sources from global memory; adding LDS reads there requires
    # careful lgkmcnt management and is deferred.  For the dot2 path,
    # n_waves > 1 still uses a wider block for better occupancy but skips
    # the cooperative LDS fill (which would otherwise be a no-op overhead).
    _use_lds = n_waves > 1 and not use_dot2
    _lds_bytes = inter * 2 if _use_lds else 0  # BF16 row in LDS

    # Elements of inter each thread loads in the cooperative LDS-fill pass.
    # With block_threads threads and inter elements: each thread loads
    # inter // block_threads elements = inter_per_thread (must be even).
    _inter_per_thread = inter // _block_threads if _use_lds else 0

    dot2_tag = "_dot2" if use_dot2 else "_f32"
    h2_tag = "_h2" if h_per_warp == 2 else ""
    kb_tag = f"_kb{k_batch}" if k_batch > 1 else ""
    w_tag = "_fp8w" if _use_fp8_w else ("_fp4w" if _use_fp4_w else "")
    lds_tag = f"_lds{n_waves}" if _use_lds else ""
    module_name = (
        f"wd_down_h{hidden}_i{inter}_e{experts}_topk{topk}"
        f"_kv{k_vector}{dot2_tag}{h2_tag}{kb_tag}{w_tag}{lds_tag}"
    )

    _k_pairs = k_vector // 2
    _k_step = _WAVE_SIZE * k_vector  # INTER elements per loop iteration
    _n_k_steps = inter // _k_step
    _n_k_steps_per_batch = _n_k_steps // k_batch  # k-steps per split-K workgroup
    # FP4 scale blocks per k_step per lane: each lane covers k_vector=8 FP4 elements;
    # block_k=32 → 8 elements < 32, so each lane sees ONE scale per k_step.
    # The scale index within the weight row = (K*_k_step + lane*k_vector) // block_k.
    _scales_per_row = inter // _block_k  # columns in scale tensor per row

    @flyc.kernel(name=module_name, known_block_size=[_block_threads, 1, 1])
    def _kernel(
        arg_y_out: fx.Pointer,  # [B, HIDDEN] f32, zero-init
        arg_inter: fx.Pointer,  # [B*TOPK, INTER] bf16
        arg_w_down: fx.Pointer,  # [E*HIDDEN, INTER] bf16/fp8/fp4
        arg_w_scale: fx.Pointer,  # [E*HIDDEN, INTER//block_k] e8m0 uint8 (FP4 only; unused otherwise)
        arg_router_ids: fx.Pointer,  # [B*TOPK] i32
        arg_router_wts: fx.Pointer,  # [B*TOPK] f32
        i32_B: fx.Int32,
        i32_TOPK: fx.Int32,
        i32_INTER: fx.Int32,
        i32_HIDDEN: fx.Int32,
        i32_E: fx.Int32,
        f32_w_scale_pt: fx.Float32,  # FP8 per-tensor scale (unused/1.0 for BF16/FP4)
    ):
        f32 = T.f32
        i32 = T.i32
        i64 = T.i64

        B_i32 = i32_B.ir_value()
        TOPK_i32 = i32_TOPK.ir_value()
        INTER_i32 = i32_INTER.ir_value()
        HIDDEN_i32 = i32_HIDDEN.ir_value()
        w_scale_pt = f32_w_scale_pt.ir_value()  # FP8 per-tensor scale

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
        # w_scale: [E*HIDDEN, INTER//block_k] e8m0 uint8 (FP4 only)
        # Total bytes: E * HIDDEN * (INTER // block_k) — but since E*HIDDEN*scales may overflow i32,
        # use -1 sentinel for the scale resource too.
        scale_rsrc = _rsrc(arg_w_scale, arith.constant(-1, type=i32))

        # ── Block → (wave_id, lane_id, out_j_0, token_b) ────────────────────
        # n_waves=1: single-wave blocks; thread_id == lane_id.
        # n_waves>1: multi-wave blocks; block handles n_waves * h_per_warp
        #   output channels and cooperatively loads inter_states into LDS.
        #
        # Grid layout (h_per_block = n_waves * h_per_warp):
        #   grid x = B × (HIDDEN / h_per_block)
        #   out_j_0 = (blk_in_token * h_per_block) + wave_id * h_per_warp
        _h_per_block = n_waves * h_per_warp
        tid_i32 = arith.index_cast(i32, gpu.thread_id("x"))
        blk_i32 = arith.index_cast(i32, gpu.block_id("x"))
        # Wave ID and lane within wave
        wave_id = tid_i32 // arith.constant(_WAVE_SIZE, type=i32)
        lane_i32 = tid_i32 % arith.constant(_WAVE_SIZE, type=i32)

        HIDDEN_blocks = arith.constant(hidden // _h_per_block, type=i32)
        blk_in_token = arith.remui(blk_i32, HIDDEN_blocks)
        token_b = arith.divui(blk_i32, HIDDEN_blocks)
        out_j_0 = (
            blk_in_token * arith.constant(_h_per_block, type=i32)
            + wave_id * arith.constant(h_per_warp, type=i32)
        )

        # LDS base pointer — pre-init so AST rewriter can resolve lds_base_i32
        # in all conditional branches (even the ones it visits but won't execute).
        # get_dyn_shared() returns a pointer in LDS addrspace (32-bit on AMD).
        # fx.ptrtoint on an LDS pointer gives i32 directly; no truncation needed.
        lds_base_i32 = arith.constant(0, type=i32)
        if _use_lds:
            lds_ptr = get_dyn_shared()
            lds_base_i32 = fx.ptrtoint(lds_ptr)

        lane_kV = lane_i32 * arith.constant(k_vector, type=i32)
        c_two = arith.constant(2, type=i32)
        zero_f32 = arith.constant(0.0, type=f32)
        zero_i32 = arith.constant(0, type=i32)

        HIDDEN_i64 = arith.extsi(i64, HIDDEN_i32)
        INTER_i64 = arith.extsi(i64, INTER_i32)
        out_j0_i64 = arith.extsi(i64, out_j_0)
        # Row size: INTER*2 bytes for BF16, INTER bytes for FP8, INTER//2 bytes for FP4.
        _w_elem_bytes = 2 if not (_use_fp8_w or _use_fp4_w) else 1
        row_nb = INTER_i32 * arith.constant(_w_elem_bytes, type=i32)
        if _use_fp4_w:
            row_nb = INTER_i32 // arith.constant(2, type=i32)  # FP4: 2 elements per byte
        w_scale = w_scale_pt  # FP8 per-tensor scale (also used in FP8 down path)

        # Outer for: slot = 0..TOPK
        # Carries: [total_acc_0] for h_per_warp=1,
        #          [total_acc_0, total_acc_1] for h_per_warp=2.
        n_accs = h_per_warp
        initial_accs = [zero_f32] * n_accs

        outer_for = scf.ForOp(
            arith.constant(0, index=True),
            arith.index_cast(ir.IndexType.get(), TOPK_i32),
            arith.constant(1, index=True),
            iter_args=initial_accs,
        )
        with ir.InsertionPoint(outer_for.body):
            slot_idx = arith.index_cast(i32, outer_for.induction_variable)
            accs = list(outer_for.inner_iter_args)

            flat_slot = token_b * TOPK_i32 + slot_idx
            expert_e = buffer_ops.buffer_load(
                rid_rsrc, flat_slot, vec_width=1, dtype=i32
            )
            router_wt = buffer_ops.buffer_load(
                rwt_rsrc, flat_slot, vec_width=1, dtype=f32
            )
            expert_i64 = arith.extsi(i64, expert_e)

            # Build per-row w_down resources for each output channel this wave owns.
            # Byte offset: element_byte = 2 (BF16), 1 (FP8), 0.5 (FP4: INTER//2 bytes).
            w_row_rsrcs: list = []
            scale_row_rsrcs: list = []  # per-output-channel scale rows (FP4 only)
            # Pre-init so AST rewriter sees w_row_byte_off defined in both branches.
            w_row_byte_off = arith.constant(0, type=i64)
            for h in range_constexpr(h_per_warp):
                out_jh_i64 = arith.addi(out_j0_i64, arith.constant(h, type=i64))
                if _use_fp4_w:
                    # FP4: row is INTER//2 bytes
                    w_row_byte_off = (
                        (expert_i64 * HIDDEN_i64 + out_jh_i64)
                        * (INTER_i64 // arith.constant(2, type=i64))
                    )
                else:
                    w_row_byte_off = (
                        (expert_i64 * HIDDEN_i64 + out_jh_i64)
                        * INTER_i64
                        * arith.constant(_w_elem_bytes, type=i64)
                    )
                w_base = arith.addi(
                    arith.index_cast(i64, fx.ptrtoint(arg_w_down)), w_row_byte_off
                )
                w_row_rsrcs.append(
                    buffer_ops.create_buffer_resource_from_addr(
                        w_base, num_records_bytes=row_nb
                    )
                )
                if _use_fp4_w:
                    # Scale row: [INTER // block_k] uint8 values for this output channel
                    sc_row_byte_off = (
                        (expert_i64 * HIDDEN_i64 + out_jh_i64)
                        * arith.constant(_scales_per_row, type=i64)
                    )
                    sc_base = arith.addi(
                        arith.index_cast(i64, fx.ptrtoint(arg_w_scale)), sc_row_byte_off
                    )
                    scale_row_rsrcs.append(
                        buffer_ops.create_buffer_resource_from_addr(
                            sc_base, num_records_bytes=arith.constant(_scales_per_row, type=i32)
                        )
                    )
                else:
                    scale_row_rsrcs.append(scale_rsrc)  # placeholder (unused)

            inter_row_base = flat_slot * INTER_i32

            # ── LDS cooperative load (n_waves > 1 only) ──────────────────────
            # All n_waves*WAVE_SIZE threads collectively load inter_states[slot]
            # into LDS, then barrier so all waves can read from it.
            # Each thread loads _inter_per_thread BF16 elements = _inter_per_thread//2 i32.
            if _use_lds:
                my_base_elem = tid_i32 * arith.constant(_inter_per_thread, type=i32)
                for _i in range_constexpr(_inter_per_thread // 2):
                    # Global source: inter_rsrc at (flat_slot * INTER + my_base + i*2) // 2
                    g_off = (inter_row_base + my_base_elem + arith.constant(_i * 2, type=i32)) // c_two
                    word = buffer_ops.buffer_load(inter_rsrc, g_off, vec_width=1, dtype=i32)
                    # LDS destination: byte offset (my_base_elem + i*2) * 2 bytes_per_BF16
                    # = (my_base_elem + i*2) * 2 bytes = local i32 index * 4 bytes
                    lds_i32_idx = (my_base_elem + arith.constant(_i * 2, type=i32)) // c_two
                    lds_byte_addr = arith.addi(
                        lds_base_i32, arith.muli(lds_i32_idx, arith.constant(4, type=i32))
                    )
                    _ds_write_b32(lds_byte_addr, word)
                gpu_barrier()  # s_barrier: ensure all writes are visible before reads

            # split-K: this block covers k-steps [k_bid*S .. (k_bid+1)*S-1]
            # where S = _n_k_steps_per_batch.  k_bid=0 when k_batch=1 (no split).
            k_bid_i32 = arith.index_cast(i32, gpu.block_id("y"))
            k_batch_base = k_bid_i32 * arith.constant(
                _n_k_steps_per_batch * _k_step, type=i32
            )

            # Inner accumulation over k_steps and k_pairs.
            #
            # dot2 path uses an scf.ForOp for k_steps + BATCHED loads per step.
            # Batched pattern: issue ALL loads → one vmcnt0 → independent dot2s
            # → one s_nop(2) drain → reduce into cur_accs.  This mirrors the
            # gate_up batched path (44%→65% HBM improvement) and eliminates the
            # per-call s_waitcnt + s_nop serialisation that _dot2_dep caused.
            #
            # f32 path: no vmcnt constraints, range_constexpr is fine.
            curs: list = [zero_f32] * h_per_warp
            if use_dot2:
                inner_for = scf.ForOp(
                    arith.constant(0, index=True),
                    arith.constant(_n_k_steps_per_batch, index=True),
                    arith.constant(1, index=True),
                    iter_args=[zero_f32] * h_per_warp,
                )
                with ir.InsertionPoint(inner_for.body):
                    k_step_i32 = arith.index_cast(i32, inner_for.induction_variable)
                    cur_accs: list = list(inner_for.inner_iter_args)
                    k_base_c = k_step_i32 * arith.constant(_k_step, type=i32)
                    lane_k = arith.addi(k_batch_base, k_base_c) + lane_kV

                    # ── Phase A: issue ALL loads for this k-step ─────────────
                    # Pre-init so AST rewriter resolves `slots` from all branches.
                    slots: list = [zero_f32] * h_per_warp
                    if _use_fp4_w:
                        # ── FP4 MXFP4 path: per-block e8m0 scales, 4-acc drain ─
                        # Matches CK's dot0..dot3 + dot2_drain4 pattern.
                        # k_vector=8 FP4 elements → 1 i32 per lane per k_step.
                        # Each lane's 8 elements are covered by ONE scale block
                        # (block_k=32 > k_vector=8).
                        # Scale index = (K*_k_step + lane*k_vector) // block_k
                        #             = K * (_k_step // block_k) + lane * (k_vector // block_k)
                        # With k_vector=8, block_k=32: lane offset = 0 (since 8//32=0).
                        # So scale_idx = K * (_k_step // _block_k) + lane // (_block_k // k_vector)
                        #              = K * 16 + lane // 4
                        c_eight = arith.constant(8, type=i32)
                        c_32 = arith.constant(32, type=i32)
                        fp4_lane_off = lane_k // c_eight  # i32 word index in FP4 row
                        scale_col = k_step_i32 * arith.constant(_k_step // _block_k, type=i32) \
                                    + lane_i32 // arith.constant(_block_k // k_vector, type=i32)

                        # Load FP4 weight words + scales for each h channel
                        w_fp4_loads: list = []  # [h] — i32 FP4 words
                        scale_loads: list = []  # [h] — i8 e8m0 scale bytes
                        for h in range_constexpr(h_per_warp):
                            w_fp4_loads.append(buffer_ops.buffer_load(
                                w_row_rsrcs[h], fp4_lane_off, vec_width=1, dtype=i32
                            ))
                            scale_loads.append(buffer_ops.buffer_load(
                                scale_row_rsrcs[h], scale_col, vec_width=1, dtype=T.i8
                            ))

                        # Load x pairs (BF16) — shared across h
                        x_fp4_loads: list = []  # [sel] — one x per FP4 sel pair
                        for sel in range_constexpr(4):
                            pair_elem = arith.constant(sel * 2, type=i32)
                            x_off = (inter_row_base + lane_k + pair_elem) // c_two
                            x_fp4_loads.append(buffer_ops.buffer_load(
                                inter_rsrc, x_off, vec_width=1, dtype=i32
                            ))

                        _vmcnt0()

                        # Convert e8m0 → f32: scale = 2^(biased_exp - 127) via bitcast
                        # e8m0 byte encodes a biased exponent; shift to bit [30:23] of f32.
                        scale_f32_vals: list = []
                        for h in range_constexpr(h_per_warp):
                            sc_byte = arith.extui(i32, scale_loads[h])
                            sc_f32 = arith.bitcast(f32, arith.shli(sc_byte, arith.constant(23, type=i32)))
                            scale_f32_vals.append(sc_f32)

                        # Pre-convert FP4→BF16 with per-block scale (before dot2 to prevent aliasing)
                        wbf16_fp4: list = []  # [h * 4 + sel]
                        for h in range_constexpr(h_per_warp):
                            for sel in range_constexpr(4):
                                wbf16_fp4.append(_fp4x2_to_bf16(w_fp4_loads[h], scale_f32_vals[h], sel))

                        # 4 independent dot2s per h (CK pattern: no s_nop between, one drain at end)
                        for h in range_constexpr(h_per_warp):
                            partial = zero_f32
                            for sel in range_constexpr(4):
                                xw = x_fp4_loads[sel]
                                partial = arith.addf(
                                    partial, _dot2_batched(zero_f32, xw, wbf16_fp4[h * 4 + sel])
                                )
                            slots[h] = partial
                        rocdl.s_nop(2)  # single drain covers all dot2 write→read hazards
                    elif _use_fp8_w:
                        c_four = arith.constant(4, type=i32)
                        # x: loaded once per (wi, sel) — shared between h=0,h=1
                        # w: loaded once per (wi, h) — separate for each output channel
                        x_fp8_loads: list = []  # [wi * 2 + sel] — one x per pair
                        w_fp8_loads: list = []  # [wi * h_per_warp + h]
                        for wi in range_constexpr(_k_fp8_words):
                            wi4 = arith.constant(wi * 4, type=i32)
                            w_i32_off = (lane_k + wi4) // c_four
                            for h in range_constexpr(h_per_warp):
                                w_fp8_loads.append(buffer_ops.buffer_load(
                                    w_row_rsrcs[h], w_i32_off, vec_width=1, dtype=i32
                                ))
                        for wi in range_constexpr(_k_fp8_words):
                            for sel in range_constexpr(2):
                                pair_elem = arith.constant((wi * 2 + sel) * 2, type=i32)
                                x_off = (inter_row_base + lane_k + pair_elem) // c_two
                                x_fp8_loads.append(buffer_ops.buffer_load(
                                    inter_rsrc, x_off, vec_width=1, dtype=i32
                                ))
                        _vmcnt0()
                        # ── Phase B: pre-convert ALL FP8→BF16, then all dot2s ───
                        # Pre-converting before any dot2 ensures fp8-word VGPRs are
                        # dead by the time dot2 runs.  xw (per wi,sel) is shared
                        # between h=0 and h=1; adjacent dot2 calls keep xw live so
                        # LLVM won't alias the h=0 output with xw's VGPR.
                        # Interleaving fp8_to_bf16 inside the dot2 loop allows LLVM
                        # to hoist them (side_effects=False), breaking xw adjacency.
                        wbf16_all: list = []
                        for wi in range_constexpr(_k_fp8_words):
                            for sel in range_constexpr(2):
                                for h in range_constexpr(h_per_warp):
                                    wbf16_all.append(
                                        _fp8x2_to_bf16(w_fp8_loads[wi * h_per_warp + h], w_scale, sel)
                                    )
                        for wi in range_constexpr(_k_fp8_words):
                            for sel in range_constexpr(2):
                                xw = x_fp8_loads[wi * 2 + sel]
                                for h in range_constexpr(h_per_warp):
                                    idx_bf16 = (wi * 2 + sel) * h_per_warp + h
                                    slots[h] = arith.addf(
                                        slots[h], _dot2_batched(zero_f32, xw, wbf16_all[idx_bf16])
                                    )
                        rocdl.s_nop(2)
                    else:
                        # BF16 weight path: one i32 = 2 BF16 per pair.
                        # x is loaded ONCE per pair (shared between h=0 and h=1)
                        # to avoid LLVM CSE-aliasing: loading x per-h gives the
                        # same address → LLVM CSEs to one SSA → h=0's dot2 output
                        # aliases x's VGPR → h=1 reads wrong data (H2 aliasing bug).
                        # Adjacent h=0,h=1 dot2 calls with shared xw prevent aliasing
                        # since xw stays live until h=1's dot2 (same as _compute_bf16d2).
                        x_loads: list = []   # [p] — one x per pair, shared across h
                        w_loads: list = []   # [p * h_per_warp + h]
                        for p in range_constexpr(_k_pairs):
                            p2 = arith.constant(p * 2, type=i32)
                            x_off = (inter_row_base + lane_k + p2) // c_two
                            w_off = (lane_k + p2) // c_two
                            x_loads.append(buffer_ops.buffer_load(
                                inter_rsrc, x_off, vec_width=1, dtype=i32
                            ))
                            for h in range_constexpr(h_per_warp):
                                w_loads.append(buffer_ops.buffer_load(
                                    w_row_rsrcs[h], w_off, vec_width=1, dtype=i32
                                ))
                        _vmcnt0()
                        # ── Phase B: independent dot2 per (p, h) ─────────────
                        # xw (x_loads[p]) is shared between h=0 and h=1 for each p.
                        # Adjacent calls keep xw live so LLVM won't alias.
                        for p in range_constexpr(_k_pairs):
                            xw = x_loads[p]
                            for h in range_constexpr(h_per_warp):
                                slots[h] = arith.addf(
                                    slots[h],
                                    _dot2_batched(zero_f32, xw, w_loads[p * h_per_warp + h]),
                                )
                        rocdl.s_nop(2)
                    # ── Phase C: reduce k-step slots into running accumulators ─
                    new_accs_inner: list = []
                    for h in range_constexpr(h_per_warp):
                        new_accs_inner.append(arith.addf(cur_accs[h], slots[h]))
                    scf.YieldOp(new_accs_inner)
                curs = list(inner_for.results)
            else:
                for k_step in range_constexpr(_n_k_steps_per_batch):
                    k_base_c = arith.constant(k_step * _k_step, type=i32)
                    lane_k = arith.addi(k_batch_base, k_base_c) + lane_kV
                    for p in range_constexpr(_k_pairs):
                        p2 = arith.constant(p * 2, type=i32)
                        # Pre-init x_word so AST rewriter resolves it from both branches
                        x_word = zero_i32
                        if _use_lds:
                            # Read from LDS: byte address = (lane_k + p2) // 2 * 4
                            lds_i32_idx = (lane_k + p2) // c_two
                            lds_byte_addr = arith.addi(
                                lds_base_i32,
                                arith.muli(lds_i32_idx, arith.constant(4, type=i32)),
                            )
                            x_word = _ds_read_b32(lds_byte_addr)
                            _lgkmcnt0()  # ensure LDS read is complete before use
                        else:
                            x_off = (inter_row_base + lane_k + p2) // c_two
                            x_word = buffer_ops.buffer_load(
                                inter_rsrc, x_off, vec_width=1, dtype=i32
                            )
                        for h in range_constexpr(h_per_warp):
                            w_off = (lane_k + p2) // c_two
                            w_word = buffer_ops.buffer_load(
                                w_row_rsrcs[h], w_off, vec_width=1, dtype=i32
                            )
                            x0 = _bf16_pair_to_f32(x_word, 0)
                            x1 = _bf16_pair_to_f32(x_word, 1)
                            curs[h] = arith.addf(
                                curs[h],
                                arith.addf(
                                    arith.mulf(x0, _bf16_pair_to_f32(w_word, 0)),
                                    arith.mulf(x1, _bf16_pair_to_f32(w_word, 1)),
                                ),
                            )

            new_accs = [
                arith.addf(accs[h], arith.mulf(router_wt, curs[h]))
                for h in range(h_per_warp)
            ]
            scf.YieldOp(new_accs)

        # ── Epilogue: butterfly reduce then lane-0 atomic-add ────────────────
        # _butterfly_reduce uses gpu.shuffle XOR — ALL 64 lanes must execute it.
        # It must be OUTSIDE the lane_zero if_op.
        totals = [_butterfly_reduce(outer_for.results[h]) for h in range(h_per_warp)]

        lane_zero = arith.cmpi(CmpIPredicate.eq, lane_i32, arith.constant(0, type=i32))
        if_op = scf.IfOp(lane_zero)
        with ir.InsertionPoint(if_op.then_block):
            for h in range_constexpr(h_per_warp):
                out_j_h = arith.addi(out_j_0, arith.constant(h, type=i32))
                out_elem_i32 = token_b * HIDDEN_i32 + out_j_h
                rocdl.raw_ptr_buffer_atomic_fadd(
                    totals[h],
                    y_rsrc,
                    out_elem_i32 * arith.constant(4, type=i32),
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
        arg_w_scale: fx.Pointer,   # [E*HIDDEN, INTER//block_k] e8m0 uint8 (FP4); dummy otherwise
        arg_router_ids: fx.Pointer,
        arg_router_wts: fx.Pointer,
        i32_B: fx.Int32,
        i32_TOPK: fx.Int32,
        i32_INTER: fx.Int32,
        i32_HIDDEN: fx.Int32,
        i32_E: fx.Int32,
        f32_w_scale_pt: fx.Float32,  # FP8 per-tensor scale; 1.0 otherwise
        stream: fx.Stream,
    ):
        idx = ir.IndexType.get()
        # Grid x: B × (HIDDEN / h_per_block)  where h_per_block = n_waves * h_per_warp
        # Grid y: k_batch — each y-slice covers INTER/k_batch elements
        # h_per_block = n_waves * h_per_warp; use direct expressions for jit closure compatibility
        grid_x = (
            arith.index_cast(idx, i32_B.ir_value())
            * arith.index_cast(idx, i32_HIDDEN.ir_value())
            // arith.constant(n_waves * h_per_warp, index=True)
        )
        grid_y = arith.constant(k_batch, index=True)
        _dk(
            arg_y_out,
            arg_inter,
            arg_w_down,
            arg_w_scale,
            arg_router_ids,
            arg_router_wts,
            i32_B,
            i32_TOPK,
            i32_INTER,
            i32_HIDDEN,
            i32_E,
            f32_w_scale_pt,
        ).launch(
            grid=(grid_x, grid_y, 1),
            block=(n_waves * _WAVE_SIZE, 1, 1),
            stream=stream,
            smem=inter * 2 if n_waves > 1 else 0,
        )

    return _launch_down
