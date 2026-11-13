# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Optimized gfx950 MiMo varlen attention specialization.

This is an intentionally isolated adaptation of the reviewed external implementation.
The public router only selects it for the global BF16
D192/V128, GQA16, single-KV-head, page-128 direct-output plan with long
contexts.  Keeping the specialization here avoids changing the shared paged
decode kernels used by uniform, sliding-window, and general varlen attention.

out[t, h] = softmax_j(scale * q[t, h] . k[j, h // 16]) @ v[j, h // 16], causal bottom-right per
sequence of a varlen batch. bf16 q/k/v/out, Q/K head dim 192, V/out head dim 128, 16 query heads per
KV head, linear KV cache [blocks, page, kv_heads, dim] with 64- or 128-token pages.

Each sequence's queries split into blocks of _QUERY_BLOCK (16) that run as pseudo-sequences of 16
queries x 16 heads = 256 rows of one KV head. A call launches, in order:
1. the query map: one CTA lists the pseudo-sequences, longest context first (a torch build above
   2**19 pseudo-sequence x sequence pairs);
2. the plan: one CTA splits each pseudo-sequence's 256-token KV blocks into tasks (work_info) and
   records each pseudo-sequence's first task and task count (reduce_info);
3. the ping-pong kernel: persistent CTAs, runtime-CU-count / KV heads per KV head, each running task after task
   in plan order through an atomic ticket (see compile_pa_decode_pingpong); with one task per
   pseudo-sequence a task writes out directly, otherwise packed partials. Per task (_task): load_q and the
   first tiles' kv_tile_loads, then for each 64-token tile i stagger_step runs online_softmax of tile i's S
   (issuing a later tile's load_kv_tile DMAs between its exp2s), pv_accumulate of tile i and qk_scores of
   tile i + 1; last the epilogue's store_output. Waves with only padded rows run idle_wave_loads;
4. the reducer, which merges the partials into out (only without direct output).

Flattened from aiter/ops/flydsl/unified_attention_bf16.py (_unified_attention_varlen) and the
kernels it calls, keeping only what this geometry compiles: the branch's os.environ knobs are fixed
at their defaults, and the ones an optimiser may flip are the constants below. Device queries are
MI355X constants. CPU tensors compile every kernel a GPU call of the same shapes would launch, and
launch nothing.
"""

import contextlib
from functools import cache, lru_cache

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.compiler.protocol import dsl_size_of, extract_to_ir_values
from flydsl.expr import const_expr, gpu, ptrtoint, range_constexpr, rocdl
from flydsl.expr.typing import T

__all__ = ["run", "workspace_requirements"]

# Ping-pong kernel knobs (the branch's PP_* defaults).
# (16-row tiles per wave, row blocks, 16-token chunks per wave): 8 waves of 32 rows, each on whole 64-token tiles.
LAYOUT = (2, 8, 4)
STAGES = 4  # LDS ring depth: four 40 KiB D192/V128 stages fill gfx950's 160 KiB LDS
PRIO = 1  # s_setprio 1 around PV and QK when set
AUX = 0  # aux (cache policy) bits of the K/V DMA loads (2 = nt cost ~6%: query blocks of a sequence reread K/V from L2)
PIN = 2  # 1 pins P in registers after the softmax, 2 also the row sums
YSCHED = 6  # LDS reads kept this many MFMAs ahead in the PV + QK phase; 0 leaves it to LLVM
# YSCHED of direct-output builds under STATIC4 (static LDS addresses leave the VGPRs for two more reads in flight:
# 247 VGPRs; the non-direct builds spill one VGPR at 8, so they keep YSCHED).
YSCHED_DIRECT = 8
# LLVM amdgpu-sched-strategy for the ping-pong kernel ("" keeps the default); iterative-ilp re-schedules for latency.
STRATEGY = "iterative-ilp"
# Each tile's K/V DMA is split over all 8 waves (3 K + 2 V loads each), branch-free (past t_end the table index is
# clamped and the loads read nothing into a free stage); the waits are vmcnt(5), trusting in-order LDS-DMA
# completion per wave, as FlyDSL's fp4_gemm_4wave does. The loads issue in the softmax phase: group 0 fills tile
# it + 2, group 1 tile it + 3 (group 1's PV of tile it - 1 runs beside group 0's softmax).
# Softmax values (0 .. M * CH * 4 - 1) the loads follow in source; LLVM hoists them to the phase head anyway.
DMA_X_AT = (4, 10, 16, 22, 28)
DMA_TIE = 0  # 1: each softmax-phase load waits (by a register tie) for the exp it follows
# 1: lean softmax-phase DMA prep: the block-table entry is read one step ahead and loop-carried (no SMEM wait in
# the phase), the K/V page bases (with the head and the in-page tile offset) are hoisted per task, and the tail
# range is one clamp of a per-task bound.
XLEAN = 2  # 2 also: hoisted per-load LDS bases, (page * tiles-per-page + tile) * tile bytes, raw V# words
# V blocks of PV read at the end of the softmax phase, before the mid barrier (their stage landed with K, before the
# previous QK, and is not rewritten before this PV).
VPRE = 0
ILOOP = 1  # 1: the tile loops run on an i32 induction variable (scalar compare; index lowers to a 64-bit VALU compare)
# 1 (XORBASE): the swizzled LDS read bases are one add per stage plus XORs (v_lds[nb] = v_lds[0] ^ 32 nb,
# k_lds[1] = k_lds[0] ^ 64): the swizzle flips only address bits 4-7 and every stage offset is a multiple of 8 KiB.
XORBASE = 1
# 1 (LAYOUT_SPLIT): the ring is split by operand: the four V stages contiguous in LDS [0, 64 KiB) (stage s at
# 16 KiB s) and the four K stages in [64 KiB, 160 KiB) (stage s at 64 KiB + 24 KiB s). Every V read of every stage
# is then one resident base per nb plus an immediate below 2**16 (16 KiB s + 4 KiB c + < 4 KiB of lane part), and
# K stages {0, 1} / {2, 3} each share one base per K half (immediates <= 43.3 KiB): the static-stage unroll's
# address plan (see stage_bases). 0 keeps the interleaved [K0 V0 K1 V1 ...] 40 KiB stages.
LAYOUT_SPLIT = 1
# 1 (STATIC4, needs LAYOUT_SPLIT): the unmasked tile loop runs 4 tiles per iteration on compile-time stages 0-3 (LDS
# read addresses are resident bases + DS immediates), then a dynamic remainder loop of < 4 tiles.
STATIC4 = 1
STEP_FENCE = 1  # STATIC4: sched_barrier fences around each unrolled step's end barrier
RAWMID = 0  # 1: the mid barrier is a bare s_barrier (+ memory clobber), so the VPRE reads stay in flight across it
# 1: direct bf16 output stores pair 16-column blocks across lane rows with v_permlane16_swap, so each dwordx4
# store covers 64 contiguous bytes per output row (the epilogue's stores were write-request bound). (launch r7)
SWAPST = 1

# Architectural gfx950 constants used by compiled kernels.  Host capacities
# and grids use the runtime device CU count supplied by the caller.
LDS_BYTES = 160 * 1024
NUM_XCDS = 8
WAVE = 64

_HEAD_DIM = 192
_VALUE_DIM = 128
_GQA = 16
_PAGE_SIZES = (64, 128)
_MAX_BATCH = 4096
_QUERY_BLOCK = 16  # queries per varlen pseudo-sequence
# Varlen pseudo-sequences the planner's LDS holds on gfx950 (160 KiB).
_MAX_PSEUDO_SEQS = 40000

LOG2E = 1.4426950408889634
KV_COMPUTE_BLOCK = 256
# Direct plans (one task per pseudo-sequence) split the TAIL_ROWS shortest live pseudo-sequences, the ones
# dispatched last, into up to TAIL_SPLIT tasks each; those write FP32 partials that a reducer over the tail merges.
TAIL_ROWS = 128
TAIL_SPLIT = 4
# A tail row gets one task per TAIL_MIN_PIECE 256-token plan tiles (at least one), so short rows, whose extra
# tasks would cost more prologue and epilogue than they balance, stay whole.
TAIL_MIN_PIECE = 8


# --- buffer and pointer helpers ------------------------------------------------------------------


def buf_base_i64(base):
    """i64 base address of *base*, an fx pointer or an address, so a caller can narrow a descriptor to one row
    (``ptr + row * row_bytes``) without first materialising a pointer for it."""
    raw = extract_to_ir_values(base)[0]
    if str(raw.type).startswith(("!fly.ptr", "!llvm.ptr")):
        return fx.Int64(ptrtoint(base))
    return fx.Int64(base)


def ptr_buf_tensor(ptr, elem=fx.Int32, unit_elems=1, num_records_bytes=None, unit_stride=None):
    """Buffer-resource (V#) view of *ptr* (an fx.Pointer or an i64 address), so ``t[i]`` / ``fx.slice`` index it.

    Keeps the descriptor in SGPRs and a 32-bit voffset per access; indexing a plain typed pointer instead builds a
    64-bit address in VGPRs on every access. ``unit_elems`` > 1 makes it ``(n, unit_elems)``, one wide access per
    ``fx.slice(t, (u, None))``; ``unit_stride`` (default ``unit_elems``) is the element distance between units and
    sets the pointer alignment. A runtime ``num_records_bytes`` gives a hardware OOB check that zero-fills.
    """
    n = 0xFFFFFFFF  # nominal extent: the real bound is the V# num_records field
    unit_stride = unit_elems if unit_stride is None else unit_stride
    layout = fx.make_layout((n,), (1,)) if unit_elems == 1 else fx.make_layout((n, unit_elems), (unit_stride, 1))
    pt = fx.PointerType.get(
        elem.ir_type, address_space=fx.AddressSpace.Global, alignment=unit_stride * (elem.width // 8)
    )
    view = fx.make_view(fx.inttoptr(pt, buf_base_i64(ptr)), layout)
    return fx.rocdl.make_buffer_tensor(view, num_records_bytes=num_records_bytes)


def ptr_arg(t: torch.Tensor, dtype):
    """Wrap a torch.Tensor as an fx.Pointer for kernel launch; a CPU tensor gives a null pointer, so flyc.compile
    only compiles."""
    return flyc.from_c_void_p(dtype, t.data_ptr() if t.is_cuda else 0)


def _run_compiled(exe, *args, live):
    """First live call: ``flyc.compile(exe, *args)`` compiles **and** executes the kernel. Later calls: fast
    dispatch via the cached ``CompiledFunction``. Calls that are not live (CPU tensors, null pointers) only compile."""
    cf = getattr(exe, "_cf", None) if live else None
    if cf is not None:
        cf(*args)
        return
    try:
        cf = flyc.compile(exe, *args)
    except Exception:
        # flyc.compile leaks ir.Context on failure; pop it so a retry takes the right path.
        with contextlib.suppress(Exception):
            while ir.Context.current is not None:
                ir.Context.current.__exit__(None, None, None)
        raise
    if live:
        exe._cf = cf


def rcp_f32(value):
    # v_rcp_f32, not IEEE 1 / x: the output bits follow it (0.5.0 has no stable form).
    return rocdl.rcp(T.f32, value)


def udiv_pow2(value, divisor: int):
    assert divisor > 0 and divisor & (divisor - 1) == 0
    return value >> fx.Int32(divisor.bit_length() - 1)


def urem_pow2(value, divisor: int):
    assert divisor > 0 and divisor & (divisor - 1) == 0
    return value & fx.Int32(divisor - 1)


def shared_ptr(elem_ty):
    """LDS pointer type of elem_ty with its element alignment (a byte-base recast drops it)."""
    return fx.PointerType.get(elem_ty.ir_type, fx.AddressSpace.Shared, dsl_size_of(elem_ty))


# --- pa_decode_pingpong.py: the tile kernel ------------------------------------------------------
#
# One CTA owns a task's query block (16 queries x 16 heads = 256 rows of one KV head) and streams 64-token K/V
# tiles into an LDS ring by async DMA. Wave w owns rows [32 w, 32 w + 32) as M = 2 16-row tiles and all 64 tokens
# of each tile. Waves 0-3 and 4-7 form two groups, one wave of each per SIMD, that run one phase apart (see
# stagger_step). Each wave keeps its own online softmax, and P stays in registers (the QK accumulator layout is
# PV's B operand). Query lengths that do not fill the block pad it; padded rows are computed but never stored.


def _pin(value):
    # Tied empty asm materializes the value here; LLVM otherwise sinks the softmax past the stagger barrier into the
    # MFMA phase (the PIN knob; 6-12% on the ancestor kernel).
    value = fx.Vector(value)
    words = value.bitcast(fx.Int32) if value.dtype.width < 32 else value
    tied = fx.Vector(llvm.inline_asm(words.ir_value().type, [words.ir_value()], "", "=v,0", has_side_effects=True))
    return tied.bitcast(value.dtype) if value.dtype.width < 32 else tied


TILE = 64
# Scores may exceed the row's reference max by 2**RESCALE_LOG2 before P is
# rescaled; BF16 P and f32 l and O hold that range.
RESCALE_LOG2 = 8.0
# Lazy rescale on each lane's tile sum of P instead of a per-tile row max: the reference moves only when
# a lane's sum of P for a row passes 2**SUM_LOG2 (so every P <= 2**SUM_LOG2, which BF16 P and f32 l and
# O hold), and the max is computed only inside that branch.
SUM_LOG2 = 8
SUM_LIMIT = float(2**SUM_LOG2)
NREF_UNSET = float(2.0**126)


def _dep_tie(word, dep):
    """word (an i32 VGPR value) returned through an empty asm that also reads dep, so its users wait for dep."""
    dep_v = fx.Vector.from_elements([fx.Float32(dep)], dtype=fx.Float32).bitcast(fx.Int32)
    dep_i = fx.Int32(dep_v[0]).ir_value()
    return llvm.inline_asm(word.type, [word, dep_i], "", "=v,0,v", has_side_effects=False)


def uniform(v):
    """v through readfirstlane: wave-uniform to LLVM, so it and the address math on it stay in SGPRs. Without it
    every ping-pong build spills 34-51 VGPRs (probe P3); 0.5.0 has no stable readfirstlane."""
    return fx.Int32(fx.rocdl.readfirstlane(T.i32, fx.Int32(v)))


def uniform64(v):
    v = fx.Int64(v)
    lo = uniform(fx.Int32(v & fx.Int64(0xFFFFFFFF)))
    hi = uniform(fx.Int32(v >> 32))
    return (fx.Int64(hi) << 32) | (fx.Int64(lo) & fx.Int64(0xFFFFFFFF))


def permlane16_swap(x, y):
    """v_permlane16_swap of two i32 values: (x, y) with lanes 16-31 of x swapped for lanes 0-15 of y, and lanes
    48-63 of x for lanes 32-47 of y. 0.5.0 has no stable form."""
    sw = fx.rocdl.permlane16_swap(ir.Type.parse("!llvm.struct<(i32, i32)>"), x.ir_value(), y.ir_value(), False, False)
    return fx.Int32(llvm.extractvalue(T.i32, sw, [0])), fx.Int32(llvm.extractvalue(T.i32, sw, [1]))


def opaque(x):
    """x through an empty asm, so LLVM cannot fold a known constant into it."""
    return fx.Int32(llvm.inline_asm(T.i32, [fx.Int32(x).ir_value()], "", "=v,0", has_side_effects=False))


def _aux(value):
    # The aux (cache policy) operand of a raw buffer op: the AUX knob's bits on the LDS DMA, which
    # cdna4.BufferLoadAsyncLDS* hard-codes to 0.
    return ir.IntegerAttr.get(ir.IntegerType.get_signless(32), value)


@cache
def compile_pa_decode_pingpong(block_size, softmax_scale, direct_output):
    """The tile kernel's launch for one page size and scale.

    The launch reads an int32 [num_seqs, 4] query map of (first query row,
    block-table row, valid queries, unused) per work-info sequence; queries at
    or past the valid count repeat the last one and the reducer drops them.

    ``direct_output`` (plans with one task per pseudo-sequence) writes
    normalized rows to a contiguous [tokens, heads, DV] output, so no reducer runs.
    """
    D, DV, GQA, QLEN = _HEAD_DIM, _VALUE_DIM, _GQA, _QUERY_BLOCK
    M, R, CH = LAYOUT
    SLOT_TOKENS = 16 * CH
    S = TILE // SLOT_TOKENS
    NWAVES = R * S
    assert NWAVES == 8 and S == 1, "the staggered schedule runs two groups of one wave per SIMD on whole tiles"
    # It overlaps a stage's DMA with reads of the stage before it.
    assert STAGES >= 3, "a 2-stage ring races the staggered schedule"
    ROWS = QLEN * GQA
    assert R * M * 16 >= ROWS, "layout must cover every query row"
    DIRECT = bool(direct_output)
    YS = YSCHED_DIRECT if (DIRECT and STATIC4) else YSCHED

    K_ROW = D * 2
    V_ROW = DV * 2
    K_CPR = D // 8
    V_CPR = DV // 8
    K_STAGE = TILE * K_ROW
    V_STAGE = TILE * V_ROW
    STAGE = K_STAGE + V_STAGE
    LEAD = STAGES - 1
    KK = D // 32
    NB = DV // 16
    TILES_PER_PLAN = KV_COMPUTE_BLOCK // TILE
    total_bytes = STAGES * STAGE
    if LAYOUT_SPLIT:
        # V stages in [0, STAGES * V_STAGE), K stages above them.
        K_BASE, V_BASE, K_STRIDE, V_STRIDE = STAGES * V_STAGE, 0, K_STAGE, V_STAGE
        assert (STAGES - 1) * V_STAGE + V_STAGE <= 1 << 16, "every V read must be base + a 16-bit DS offset"
    else:
        K_BASE, V_BASE, K_STRIDE, V_STRIDE = 0, K_STAGE, STAGE, STAGE
    # The XOR bases need stage strides and bases with no bits below 8 KiB (the swizzle flips bits 4-7).
    assert all(x % 8192 == 0 for x in (K_BASE, V_BASE, K_STRIDE, V_STRIDE))

    def k_stage_off(stage):
        """LDS byte offset of K stage `stage` (Python int or traced)."""
        return K_BASE + stage * K_STRIDE

    def v_stage_off(stage):
        return V_BASE + stage * V_STRIDE
    scale_log2 = float(softmax_scale) * LOG2E

    @fx.struct
    class SharedStorage:
        buf: fx.Array[fx.Int32, total_bytes // 4, 16]

    # The branch's other knobs stay in the name at their defaults, so the symbol (and climb's ISA hash) is unchanged.
    kernel_name = (
        f"pa_decode_pingpong_kvbf16_d{D}_v{DV}_bs{block_size}_gqa{GQA}_mtp{QLEN}"
        f"_m{M}r{R}c{CH}_s{STAGES}_p{PRIO}_c0_n0_a{AUX}_x1{PIN}y{YS}g0k1i{int(STAGES >= 4)}u0l0f00w000000_fd_varlen"
    )
    if STRATEGY:
        kernel_name += "_" + STRATEGY.replace("-", "")
    kernel_name += "_dma8_p1p4p7q1q2_x" + "_".join(map(str, DMA_X_AT)) + f"t{DMA_TIE}w"
    if XLEAN:
        kernel_name += f"_xl{XLEAN}"
    if VPRE or RAWMID:
        kernel_name += f"_vp{VPRE}r{RAWMID}"
    if ILOOP:
        kernel_name += "_il"
    if XORBASE or LAYOUT_SPLIT:
        kernel_name += f"_xb{XORBASE}ls{LAYOUT_SPLIT}o{STATIC4}f{STEP_FENCE}"
    if DIRECT:
        kernel_name += "_direct"
    kernel_name += "_idlew"
    if SWAPST:
        kernel_name += "_swst"
    # Persistent CTAs take kernel arguments the one-task-per-CTA kernel lacks: never alias its cached binary.
    kernel_name += "_persist"
    # Online-softmax state: [m (M), l (M), O (M * NB)].
    STATE = 2 * M + M * NB

    @flyc.jit
    def _task(
        lds, pmax_ptr: fx.Pointer, psum_ptr: fx.Pointer, pout_ptr: fx.Pointer, query_ptr: fx.Pointer,
        key_cache_ptr: fx.Pointer, value_cache_ptr: fx.Pointer, block_tables_ptr: fx.Pointer,
        query_map_ptr: fx.Pointer, max_blocks_per_seq: fx.Int32, stride_q_row: fx.Int32, stride_q_head: fx.Int32,
        seq: fx.Int32, start: fx.Int32, end: fx.Int32, context: fx.Int32, slot: fx.Int32, work_capacity: fx.Int32,
        out_ptr: fx.Pointer, reduce_ptr: fx.Pointer, n_kv,
    ):
        """One task: query block seq against KV tiles [start, end) * TILES_PER_PLAN, through the LDS ring at lds."""
        kv_h = fx.Int32(fx.block_idx.y)
        tid = fx.Int32(fx.thread_idx.x)
        warp = uniform(tid // WAVE)
        lane = tid % WAVE
        g = lane // 16
        l16 = lane % 16
        # The wave's token slot and row block: ts == 0 (S == 1) and rb == warp (R == 8 waves), but dropping them
        # moved the ISA and raced slower on TIME[0] in both final races (change 9, reverted).
        ts = warp // R
        rb = warp % R
        grp = warp // (NWAVES // 2)  # the wave's group: waves 0-3 and 4-7 run one phase apart
        neg_inf = fx.Float32(float("-inf"))
        zero = fx.Float32(0.0)
        # Query of each of the wave's M tiles; padded tiles reuse the last query.
        queries = [rb * M + m for m in range_constexpr(M)]
        qmap = fx.ptr_load(
            fx.add_offset(fx.recast_iter(fx.Int32, query_map_ptr), seq * 4),
            result_type=fx.Vector.make_type(4, fx.Int32),
        )
        first_row = uniform(fx.Int32(qmap[0]))
        table_seq = uniform(fx.Int32(qmap[1]))
        valid = uniform(fx.Int32(qmap[2]))

        def load_q():
            """Q as QK's B operand: lane (g, l16) holds head l16, dims 32 kk + 8 g, of each of the M queries."""
            q_buf = ptr_buf_tensor(query_ptr, fx.BFloat16)
            q_tiled = fx.logical_divide(q_buf, fx.make_layout(1, 1))
            q_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.BFloat16)
            q_reg = fx.make_rmem_tensor(fx.make_layout(8, 1), fx.BFloat16)
            q_ops = []
            for m in range_constexpr(M):
                q_idx = queries[m]
                q_idx = (q_idx < valid).select(q_idx, valid - 1)
                q_row = first_row + q_idx
                head = kv_h * GQA + l16
                row_elem = q_row * stride_q_row + head * stride_q_head + g * 8
                ops_m = []
                for kk in range_constexpr(KK):
                    fx.copy(q_atom, fx.slice(q_tiled, (None, row_elem + kk * 32)), q_reg)
                    ops_m.append(fx.Vector(fx.memref_load_vec(q_reg)))
                q_ops.append(ops_m)
            return q_ops

        q_ops = load_q()

        k_base = buf_base_i64(key_cache_ptr)
        v_base = buf_base_i64(value_cache_ptr)
        bt = ptr_buf_tensor(block_tables_ptr, fx.Int32)
        k_tok_bytes = n_kv * K_ROW
        v_tok_bytes = n_kv * V_ROW

        t_begin = start * TILES_PER_PLAN
        ctx_tiles = (context + (TILE - 1)) // TILE
        t_end_plan = end * TILES_PER_PLAN
        t_end = (t_end_plan < ctx_tiles).select(t_end_plan, ctx_tiles)
        # Tiles below t_split hold no position past any query's causal bound.
        unmasked = context - (valid - 1)
        unmasked = (unmasked > 0).select(unmasked, 0) // TILE
        t_split = (unmasked < t_end).select(unmasked, t_end)
        t_split = (t_split > t_begin).select(t_split, t_begin)

        # Each wave's DMA share of a tile: K8 K and V8 V loads of 16 bytes a lane, TILE_LOADS in all (its vmcnt).
        K8 = TILE * K_CPR // WAVE // NWAVES
        V8 = TILE * V_CPR // WAVE // NWAVES
        assert K8 * WAVE * NWAVES == TILE * K_CPR and V8 * WAVE * NWAVES == TILE * V_CPR
        TILE_LOADS = K8 + V8

        # A stage holds a token's K (V) row as 16-byte chunks, chunk c at c ^ swizzle(token): the XOR spreads the read
        # lanes over the LDS banks. Rows of 24 chunks are not a power of two, so no SwizzleType.
        def k_swizzle(tok):
            return (tok >> 1) & 7

        def v_swizzle(tok):
            return (tok & 7) << 1

        # DMA copy i of a stage writes LDS bytes [1024 i, 1024 i + 1024); lane l
        # fetches the global chunk the swizzle stores at chunk 64 i + l.
        def dma_offsets(copies, cpr, tok_bytes, swizzle):
            offs = []
            for j in range_constexpr(copies):
                pos = (warp * copies + j) * WAVE + lane
                tok = pos // cpr
                offs.append(tok * tok_bytes + ((pos % cpr) ^ swizzle(tok)) * 16)
            return offs

        k_voff = dma_offsets(K8, K_CPR, k_tok_bytes, k_swizzle)
        v_voff = dma_offsets(V8, V_CPR, v_tok_bytes, v_swizzle)

        page_shift = (block_size // TILE).bit_length() - 1
        k_page_bytes = uniform(n_kv * (block_size * K_ROW))
        v_page_bytes = uniform(n_kv * (block_size * V_ROW))

        def zext(x):
            return fx.Int64(x) & fx.Int64(0xFFFFFFFF)

        TPP = block_size // TILE  # tiles per page
        if const_expr(XLEAN):
            # Per task: the head's row inside a page folded into the K/V bases, one base per in-page tile.
            nkr = uniform(n_kv * K_ROW)
            nvr = uniform(n_kv * V_ROW)
            kb_t = [uniform64(k_base + zext(kv_h * K_ROW + p * TILE * nkr)) for p in range_constexpr(TPP)]
            vb_t = [uniform64(v_base + zext(kv_h * V_ROW + p * TILE * nvr)) for p in range_constexpr(TPP)]
            if const_expr(XLEAN > 1):
                # Each load's LDS offset within a stage (V's includes K_STAGE), so a load's m0 is one add.
                lds8_k = [uniform(fx.Int32(K_BASE + (warp * K8 + j) * (WAVE * 16))) for j in range_constexpr(K8)]
                lds8_v = [uniform(fx.Int32(V_BASE + (warp * V8 + j) * (WAVE * 16))) for j in range_constexpr(V8)]
            # DMA range: rows * n_kv * ROW (a lane's offset is row * n_kv * ROW + chunk, chunk < ROW), rows =
            # clamp(lim - tok0, 0, TILE); lim = min(context, t_end * TILE) also empties tiles past t_end.
            lim = uniform((context < t_end * TILE).select(context, t_end * TILE))
            step_k = uniform(nkr * TILE)
            step_v = uniform(nvr * TILE)

        def raw_rsrc(addr, nrec):
            """V# from its four words, without make_buffer_rsrc's stride mask on the high address word (a
            device address is below 2**48, so bits 48-63, the stride, are already 0)."""
            addr = fx.Int64(addr)
            words = [
                fx.Int32(addr & fx.Int64(0xFFFFFFFF)), fx.Int32(addr >> 32), fx.Int32(nrec),
                fx.Int32((7 << 12) | (4 << 15)),
            ]
            vec = fx.Vector.from_elements(words, dtype=fx.Int32)
            i128 = llvm.bitcast(ir.IntegerType.get_signless(128), vec.ir_value())
            return llvm.inttoptr(ir.Type.parse("!llvm.ptr<8>"), i128)

        def load_phys(tile):
            """The block table's page of tile, clamped to the last tile so the read stays in the sequence's row."""
            tile_c = (tile < t_end).select(tile, t_end - 1)
            # Wave-uniform, so an s_buffer_load into an SGPR.
            return fx.rocdl.s_buffer_load(fx.get_iter(bt), table_seq * max_blocks_per_seq + (tile_c >> page_shift))

        def load_kv_tile(rs, lds_off, voff, dep=None):
            """One LDS DMA of this wave's share of a K or V tile: 16 bytes a lane from voffset voff of rs to LDS
            lds_off. dep: a value the load waits for (by a register tie, which holds nothing else)."""
            dst = fx.to_llvm_ptr(fx.recast_iter(shared_ptr(fx.Int32), fx.add_offset(lds, fx.make_int_tuple(lds_off))))
            voff = fx.Int32(voff).ir_value()
            if const_expr(dep is not None):
                voff = _dep_tie(voff, dep)
            fx.rocdl.raw_ptr_buffer_load_async_lds(
                rs, dst, fx.Int32(16).ir_value(), voff, fx.Int32(0).ir_value(), fx.Int32(0).ir_value(), aux=_aux(AUX)
            )

        def kv_tile_loads_lean(tile, stage, phys):
            """kv_tile_loads with a carried phys and the bases hoisted per task (XLEAN)."""
            if const_expr(XLEAN > 1):
                kb, vb = kb_t[0], vb_t[0]
                # The tile's index among all page tiles: the in-page tile offset rides the page multiply.
                ptile = phys if TPP == 1 else phys * TPP + (tile & (TPP - 1))
            elif const_expr(TPP == 1):
                kb, vb = kb_t[0], vb_t[0]
            else:  # TPP == 2: pages are 64 or 128 tokens
                odd = (tile & 1) != 0
                kb = odd.select(kb_t[1], kb_t[0])
                vb = odd.select(vb_t[1], vb_t[0])
            loads = []
            rows = lim - tile * TILE
            rows = (rows < TILE).select(rows, TILE)

            def operand_loads(base, page_bytes, nkvrow, voffs, copies, lds_off, step_b, lds8, stage_off):
                # Clamped at 0 after the multiply: a min/max pair on rows becomes a VALU v_med3.
                nrec = rows * nkvrow
                nrec = uniform((nrec > 0).select(nrec, 0))
                if const_expr(XLEAN > 1):
                    rs = raw_rsrc(uniform64(base + zext(ptile) * zext(step_b)), nrec)
                    for j in range_constexpr(copies):
                        loads.append(lambda dep=None, j=j: load_kv_tile(rs, stage_off + lds8[j], voffs[j], dep))
                else:
                    addr = uniform64(base + zext(phys) * zext(page_bytes))
                    rs = fx.rocdl.get_buffer_rsrc(fx.get_iter(ptr_buf_tensor(addr, num_records_bytes=fx.Int64(nrec))))
                    for j in range_constexpr(copies):
                        loads.append(
                            lambda dep=None, j=j: load_kv_tile(rs, lds_off + (warp * copies + j) * (WAVE * 16), voffs[j], dep)
                        )

            operand_loads(kb, k_page_bytes, nkr, k_voff, K8, k_stage_off(stage), step_k, lds8_k if XLEAN > 1 else None,
                 stage * K_STRIDE)
            operand_loads(vb, v_page_bytes, nvr, v_voff, V8, v_stage_off(stage), step_v, lds8_v if XLEAN > 1 else None,
                 stage * V_STRIDE)
            return loads

        def kv_tile_loads(tile, stage, ok):
            """This wave's share of tile's K/V DMA into stage, one load_kv_tile call per thunk; the block-table read
            and the descriptors happen now. Past t_end (ok false) the table index is clamped and the loads read
            nothing."""
            phys = load_phys(tile)
            tok0 = tile * TILE
            row_idx = (tok0 & (block_size - 1)) * n_kv + kv_h
            left = context - tok0
            rows_m1 = ((left < TILE).select(left, TILE) - 1) * n_kv
            loads = []

            def operand_loads(base, page_bytes, row, voffs, copies, lds_off):
                addr = uniform64(base + zext(phys) * zext(page_bytes) + zext(row_idx * row))
                nrec = uniform(ok.select(rows_m1 * row + row, 0))
                rs = fx.rocdl.get_buffer_rsrc(fx.get_iter(ptr_buf_tensor(addr, num_records_bytes=fx.Int64(nrec))))
                for j in range_constexpr(copies):
                    loads.append(
                        lambda dep=None, j=j: load_kv_tile(rs, lds_off + (warp * copies + j) * (WAVE * 16), voffs[j], dep)
                    )

            operand_loads(k_base, k_page_bytes, K_ROW, k_voff, K8, k_stage_off(stage))
            operand_loads(v_base, v_page_bytes, V_ROW, v_voff, V8, v_stage_off(stage))
            return loads

        for s in range_constexpr(STAGES - 1):
            # Always issued (empty past t_end), so the prologue's vmcnt counts hold. Group 0 issues tile 2 itself
            # in its first step.
            if const_expr(s == STAGES - 2):
                if grp == 1:
                    for load in kv_tile_loads(fx.Int32(t_begin + s), s, fx.Int32(t_begin + s) < t_end):
                        load()
            else:
                for load in kv_tile_loads(fx.Int32(t_begin + s), s, fx.Int32(t_begin + s) < t_end):
                    load()

        # Loop-invariant LDS read offsets within a stage.
        kq = l16 // 4
        qq = l16 % 4
        # The swizzles see neither the chunk nor bits above 8 chunks of a K row,
        # so those become instruction immediates over 2 K and NB V bases.
        swz = k_swizzle(l16)
        k_tok = ts * SLOT_TOKENS + l16
        k_lds = [K_BASE + k_tok * K_ROW + ((par * 4 + g) ^ swz) * 16 for par in range_constexpr(2)]
        k_imm = [[c * 16 * K_ROW + (kk // 2) * 128 for kk in range_constexpr(KK)] for c in range_constexpr(CH)]
        vswz = v_swizzle(g * 4 + kq)
        v_tok = ts * SLOT_TOKENS + g * 4 + kq
        v_lds = [
            v_tok * V_ROW + ((nb * 2 + qq // 2) ^ vswz) * 16 + (qq % 2) * 8 for nb in range_constexpr(NB)
        ]
        # V_BASE rides the immediates (with XORBASE a base plus a constant would cost LLVM a third add per stage).
        v_imm = [V_BASE + c * 16 * V_ROW for c in range_constexpr(CH)]
        # K stages {2, 3}'s bases (static-stage plan: K stages {0, 1} read from k_lds, {2, 3} from k_lds_hi).
        k_lds_hi = [b + 2 * K_STRIDE for b in k_lds]
        if const_expr(LAYOUT_SPLIT):
            # The K bases sit at or above 64 KiB: a known constant there makes LLVM fold base + offset into values too
            # wide for the 16-bit DS field, one VGPR each (kb lds-buffer-above-64k-loses-offsets; under 0.5.0 +4 to
            # +14 VGPRs and 0.55% slower, probe P1). The asm hides the constant once per task; the mask restores the
            # 16-byte alignment it hides (without it K reads split into ds_read2_b64, an address add each).
            k_lds_hi = [opaque(b) & fx.Int32(-16) for b in k_lds_hi]
            k_lds = [opaque(b) & fx.Int32(-16) for b in k_lds]

        # Raw-score state [m (M), l (M), O (M * NB)]: m is a row-uniform
        # reference, rescaled only when a score exceeds it by RESCALE_LOG2 in
        # the log2 domain; l is per lane.
        # m is held as the negated scaled reference -m_raw * scale_log2 (the exp2 addend); NREF_UNSET
        # marks a row with no reference yet: every finite score then exponentiates to inf, which the
        # tile-sum test below catches.
        init = [fx.Float32(NREF_UNSET)] * M + [zero] * M + [
            fx.full(4, 0.0, fx.Float32) for _ in range_constexpr(M * NB)
        ]
        limit = [context - valid + queries[m] for m in range_constexpr(M)]

        def rescale(st, lmax):
            """Move each row's reference m to its max over the rows' lanes; scale l and O by alpha = exp2(old - new)."""
            out_m = []
            out_l = []
            out_o = []
            for m in range_constexpr(M):
                rmax = lmax[m]
                for sh in (16, 32):
                    rmax = fx.maxnumf(rmax, rmax.shuffle_xor(sh, WAVE))
                ms_old = -st[m]
                ms_new = fx.maxnumf(ms_old, rmax * fx.Float32(scale_log2))
                alpha = fx.Float32(fx.exp2(ms_old - ms_new, fastmath="fast"))
                m_new = -ms_new
                out_m.append(m_new)
                out_l.append(st[M + m] * alpha)
                alpha_b = fx.full(4, alpha, fx.Float32)
                out_o.extend(st[2 * M + m * NB + nb] * alpha_b for nb in range_constexpr(NB))
            return out_m + out_l + out_o

        def stage_bases(stage):
            """The K (2) and V (NB) LDS read bases of a stage."""
            if const_expr(isinstance(stage, int) and LAYOUT_SPLIT):
                # A compile-time stage (LAYOUT_SPLIT): resident bases, the stage rides the DS immediates. V: one
                # base per nb serves every stage; K: one base per (par, stage // 2), 8 V + 4 K loop-invariant VGPRs.
                half = stage // 2
                ko = (stage - 2 * half) * K_STRIDE
                return [b + ko for b in (k_lds_hi if half else k_lds)], [b + stage * V_STRIDE for b in v_lds]
            k_off = stage * K_STRIDE
            v_off = stage * V_STRIDE
            if const_expr(XORBASE):
                # (b + off) ^ c == (b ^ c) + off: off has no bits below 8 KiB, c only bits 5-7 (V) or 6 (K).
                kb0 = k_off + k_lds[0]
                vb0 = v_off + v_lds[0]
                return [kb0, kb0 ^ 64], [vb0 if nb == 0 else vb0 ^ (32 * nb) for nb in range_constexpr(NB)]
            return [k_off + b for b in k_lds], [v_off + b for b in v_lds]

        mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 32, fx.BFloat16))  # v_mfma_f32_16x16x32_bf16

        def frag(value, dtype):
            """A register fragment holding value, as fx.gemm takes its operands; the copy folds away."""
            t = fx.make_rmem_tensor(len(value), dtype)
            t.store(fx.Vector(value))
            return t

        def qk_scores(stage):
            """S = K Q^T for a stage: lane (g, l16) of chunk c holds token
            slot + 16c + 4g + i, row l16."""
            k_stage, _ = stage_bases(stage)
            s_acc = [[None] * CH for _ in range_constexpr(M)]

            def read_k(c):
                return [
                    fx.ptr_load(
                        fx.recast_iter(shared_ptr(fx.Int64), fx.add_offset(lds, fx.make_int_tuple(k_stage[kk % 2] + k_imm[c][kk]))),
                        result_type=fx.Vector.make_type(2, fx.Int64),
                    ).bitcast(fx.BFloat16)
                    for kk in range_constexpr(KK)
                ]

            # Chunk c + 1's K reads issue under chunk c's MFMAs.
            k_ops = read_k(0)
            for c in range_constexpr(CH):
                k_next = read_k(c + 1) if c + 1 < CH else None
                for m in range_constexpr(M):
                    s_acc[m][c] = fx.full(4, 0.0, fx.Float32)
                for kk in range_constexpr(KK):
                    for m in range_constexpr(M):
                        s_frag = frag(s_acc[m][c], fx.Float32)
                        fx.gemm(mma_atom, s_frag, frag(k_ops[kk], fx.BFloat16), frag(q_ops[m][kk], fx.BFloat16), s_frag)
                        s_acc[m][c] = fx.Vector(s_frag.load())
                k_ops = k_next
            return [s_acc[m][c] for m in range_constexpr(M) for c in range_constexpr(CH)]

        def online_softmax(it, s_flat, state, masked, loads):
            """Online softmax of one tile's scores: (state with O rescaled, P). loads: the DMA thunks to issue
            after the exp2 of each DMA_X_AT score."""
            lane_tok = it * TILE + ts * SLOT_TOKENS + g * 4
            scores = []
            for m in range_constexpr(M):
                vals = []
                for c in range_constexpr(CH):
                    for i in range_constexpr(4):
                        v = fx.Float32(s_flat[m * CH + c][i])
                        if const_expr(masked):
                            rel = limit[m] - lane_tok
                            keep = fx.Int32(c * 16 + i) <= rel
                            v = keep.select(v, neg_inf)
                        vals.append(v)
                scores.append(vals)

            def exp_scores(nrefs, hk=None):
                """P against the current reference, and each lane's tile sum per row."""
                p_ops = []
                tsum = []
                for m in range_constexpr(M):
                    p = []
                    for v in scores[m]:
                        e = fx.Float32(fx.exp2(fx.fma(v, fx.Float32(scale_log2), nrefs[m]), fastmath="fast"))
                        p.append(e)
                        if const_expr(hk is not None):
                            for load in hk.get(m * len(scores[m]) + len(p) - 1, []):
                                load(e if DMA_TIE else None)
                        if const_expr(len(p) % 8 == 0):
                            p_ops.append(fx.Vector.from_elements(p[len(p) - 8 :], dtype=fx.Float32).to(fx.BFloat16))
                    # Four interleaved partial chains, then a tree.
                    parts = [p[k] for k in range_constexpr(4)]
                    for k in range_constexpr(4, len(p)):
                        parts[k % 4] = parts[k % 4] + p[k]
                    tsum.append((parts[0] + parts[1]) + (parts[2] + parts[3]))
                return p_ops, tsum

            st = list(state)
            p_ops, tsum = exp_scores(st[:M], loads)
            # Each P <= its lane's tile sum: rescale only when a sum passes 2**SUM_LOG2 (inf when a score
            # overflows exp2 or the row has no reference yet), so the row max is computed only then.
            over = tsum[0] > fx.Float32(SUM_LIMIT)
            for m in range_constexpr(1, M):
                over = over | (tsum[m] > fx.Float32(SUM_LIMIT))
            if fx.Int64(fx.rocdl.ballot(T.i64, over.ir_value())) != fx.Int64(0):
                lmax = []
                for m in range_constexpr(M):
                    mx = scores[m][0]
                    for v in scores[m][1:]:
                        mx = fx.maxnumf(mx, v)
                    lmax.append(mx)
                st = rescale(st, lmax)
                p_ops, tsum = exp_scores(st[:M])
            for m in range_constexpr(M):
                st[M + m] = st[M + m] + tsum[m]
            return st, p_ops

        v_tr_atom = fx.make_copy_atom(fx.rocdl.cdna4.LDSReadTrans(16, 64), fx.BFloat16)  # ds_read_b64_tr_b16
        v_tr_layout = fx.make_layout(4, 1)

        def read_v_chunk(stage, nb, c):
            """V block nb of chunk c, transposed (VPRE's early reads)."""
            _, v_stage = stage_bases(stage)
            v_reg = fx.make_rmem_tensor(v_tr_layout, fx.BFloat16)
            v_ptr = fx.recast_iter(shared_ptr(fx.BFloat16), fx.add_offset(lds, fx.make_int_tuple(v_stage[nb] + v_imm[c])))
            fx.copy(v_tr_atom, fx.make_view(v_ptr, v_tr_layout), v_reg)
            return fx.Vector(v_reg.load())

        def read_v(stage, nb):
            """V block nb of every chunk, transposed: lane j holds dim j of tokens 4g + k, k = 0..3."""
            _, v_stage = stage_bases(stage)
            parts = []
            for c in range_constexpr(CH):
                v_reg = fx.make_rmem_tensor(v_tr_layout, fx.BFloat16)
                v_ptr = fx.recast_iter(shared_ptr(fx.BFloat16), fx.add_offset(lds, fx.make_int_tuple(v_stage[nb] + v_imm[c])))
                fx.copy(v_tr_atom, fx.make_view(v_ptr, v_tr_layout), v_reg)
                parts.append(fx.Vector(v_reg.load()))
            return parts

        def pv_accumulate(stage, p_flat, st, v_pre=()):
            """O += P V for a stage; the transpose read gives lane j dim j of
            tokens 4g + k, k = 0..3, in its 16-lane group."""
            pairs = CH // 2
            o_acc = [[st[2 * M + m * NB + nb] for nb in range_constexpr(NB)] for m in range_constexpr(M)]

            def v_block(nb):
                if const_expr(nb < len(v_pre)):
                    return v_pre[nb]
                return read_v(stage, nb)

            v_cur = v_block(0)
            for nb in range_constexpr(NB):
                parts = v_cur
                v_cur = v_block(nb + 1) if nb + 1 < NB else None
                for m in range_constexpr(M):
                    for j in range_constexpr(pairs):
                        v_op = fx.Vector.from_elements(
                            [parts[2 * j + c][i] for c in range_constexpr(2) for i in range_constexpr(4)],
                            dtype=fx.BFloat16,
                        )
                        o_frag = frag(o_acc[m][nb], fx.Float32)
                        fx.gemm(mma_atom, o_frag, frag(v_op, fx.BFloat16), frag(p_flat[m * pairs + j], fx.BFloat16), o_frag)
                        o_acc[m][nb] = fx.Vector(o_frag.load())
            return st[: 2 * M] + [o_acc[m][nb] for m in range_constexpr(M) for nb in range_constexpr(NB)]

        def schedule_pv_qk():
            """One MFMA wave a SIMD: keep the PV and QK LDS reads YS MFMAs ahead."""
            n_ds = (NB - VPRE + KK) * CH
            n_mfma = M * NB * (CH // 2) + M * KK * CH
            fx.rocdl.sched_dsrd(YS)
            for _ in range_constexpr(n_ds - YS):
                fx.rocdl.sched_mfma(1)
                fx.rocdl.sched_dsrd(1)
            fx.rocdl.sched_mfma(n_mfma - (n_ds - YS))

        def smod(x):
            return x & (STAGES - 1) if STAGES & (STAGES - 1) == 0 else x % STAGES

        def stagger_step(it, carry, masked, srel=None):
            """One tile of the pipeline: online_softmax of tile it (issuing a later tile's DMA), then pv_accumulate
            of tile it and qk_scores of tile it + 1.

            Group 1 runs one phase behind group 0: each phase X (softmax of tile i) pairs with the other group's
            phase Y (PV of tile i, QK of tile i + 1). Tile i + LEAD reuses tile i - 1's stage once group 1's PV of
            it ends; a group waits for its tile two slots before the first QK, when that tile is its only one in
            flight. srel (STATIC4): rel mod STAGES known at trace time, so the read stages are Python ints."""
            rel = it - t_begin
            state = carry[:STATE]
            s_flat = carry[STATE : STATE + M * CH]
            phys_next = None

            # Group 0's softmax runs beside group 1's PV of tile it - 1, so it fills tile it - 2's stage.
            xt = it + (LEAD - 1) + grp
            if const_expr(XLEAN):
                loads = kv_tile_loads_lean(xt, smod(rel + (LEAD - 1) + grp), fx.Int32(carry[STATE + M * CH]))
                # Next step's table entry, read now and carried: no SMEM wait in either phase.
                phys_next = load_phys(xt + 1)
            else:
                loads = kv_tile_loads(xt, smod(rel + (LEAD - 1) + grp), xt < t_end)
            at_score = {}
            for at, load in zip(DMA_X_AT, loads):
                at_score.setdefault(at, []).append(load)
            st, p_flat = online_softmax(it, s_flat, state, masked, at_score)
            if const_expr(PIN):
                p_flat = [_pin(p) for p in p_flat]
            if const_expr(PIN > 1):
                sums = _pin(fx.Vector.from_elements(st[M : 2 * M], dtype=fx.Float32))
                st = st[:M] + [sums[m] for m in range_constexpr(M)] + st[2 * M :]
            v_pre = ()
            if const_expr(VPRE):
                fx.rocdl.sched_barrier(0)
                if const_expr(XLEAN):
                    # The carried table read: retire it before the V reads, so the PV + QK phase's LDS waits
                    # stay graded (a pending SMEM read forces lgkmcnt(0) at the first one).
                    fx.rocdl.s_waitcnt(lgkmcnt=0)
                # One read at a time in use order, so the PV phase's first waits are graded.
                v_pre = []
                for nb in range_constexpr(VPRE):
                    parts = []
                    for c in range_constexpr(CH):
                        parts.append(read_v_chunk(smod(rel) if srel is None else srel % STAGES, nb, c))
                        fx.rocdl.sched_barrier(0)
                    v_pre.append(parts)
                v_pre = tuple(v_pre)
            # Group 0: its share of tile it + 1 has landed (tile it + 2's, just issued, may be in flight).
            # Group 1 needs only the end-of-step wait; this one (its tile it + 2 share, issued a step ago)
            # is free and saves the branch.
            if const_expr(XLEAN and not VPRE):
                # The carried table read landed long ago; clearing it here keeps the PV + QK phase's LDS
                # waits graded (a pending SMEM read forces lgkmcnt(0) at the first one).
                fx.rocdl.s_waitcnt(vmcnt=TILE_LOADS, lgkmcnt=0)
            else:
                fx.rocdl.s_waitcnt(vmcnt=TILE_LOADS)
            if const_expr(RAWMID):
                fx.rocdl.s_barrier()
                # The bare barrier orders no memory: keep LLVM from moving LDS accesses across it.
                llvm.inline_asm(res=None, operands_=[], asm_string="", constraints="~{memory}", has_side_effects=True)
            else:
                gpu.barrier()
            if const_expr(PRIO):
                fx.rocdl.s_setprio(1)
            # The last tile's lookahead QK reads a stale stage; it is unused.
            state = pv_accumulate(smod(rel) if srel is None else srel % STAGES, p_flat, st, v_pre)
            s_next = qk_scores(smod(rel + 1) if srel is None else (srel + 1) % STAGES)
            if const_expr(YS):
                schedule_pv_qk()
            if const_expr(PRIO):
                fx.rocdl.s_setprio(0)
            # Group 1: its share of tile it + 2 has landed (tile it + 3's, just issued, may be in flight).
            # Group 0 needs only the mid-step wait; this one is free and saves the branch.
            fx.rocdl.s_waitcnt(vmcnt=TILE_LOADS)
            if const_expr(srel is not None and STEP_FENCE):
                # Unrolled steps share a block: keep the next step's DMA prep (SALU) out of this MFMA phase's tail.
                fx.rocdl.sched_barrier(0)
            gpu.barrier()
            if const_expr(srel is not None and STEP_FENCE):
                fx.rocdl.sched_barrier(0)
            if const_expr(phys_next is not None):
                return list(state) + list(s_next) + [phys_next]
            return list(state) + list(s_next)

        def epilogue(final):
            """Normalise O by l and store it: straight to out (DIRECT) or as FP32 or bf16 partials with m and l."""
            m_fin = [(final[m] < fx.Float32(NREF_UNSET)).select(-final[m], neg_inf) for m in range_constexpr(M)]
            l_fin = []
            for m in range_constexpr(M):
                lr = final[M + m]
                for sh in (16, 32):
                    lr = lr + lr.shuffle_xor(sh, WAVE)
                l_fin.append(lr)
            o_fin = [[final[2 * M + m * NB + nb] for nb in range_constexpr(NB)] for m in range_constexpr(M)]

            partial = kv_h * work_capacity + slot
            # Per-partial resources: padded rows land past them and are dropped.
            # Direct plans: a row with one task stores normalized bf16 rows straight to out; a split tail row stores
            # FP32 partials plus m and l. Both stores issue; the unused path's resources are empty, so they drop.
            out_bytes = 4 if DIRECT else 2
            part_on = None
            if const_expr(DIRECT):
                rinfo = fx.ptr_load(
                    fx.add_offset(fx.recast_iter(fx.Int32, reduce_ptr), seq * 2),
                    result_type=fx.Vector.make_type(2, fx.Int32),
                )
                split_row = uniform(fx.Int32(rinfo[1])) > 1
                part_on = split_row

            def part_bytes(n):
                return n if part_on is None else fx.Int64(part_on.select(fx.Int32(n), fx.Int32(0)))

            # Buffer views of this task's partial O (FP32 for direct plans, else bf16), out, m and l.
            pout = ptr_buf_tensor(
                uniform64(buf_base_i64(pout_ptr) + fx.Int64(partial) * (ROWS * DV * out_bytes)),
                fx.Float32 if DIRECT else fx.BFloat16,
                num_records_bytes=part_bytes(ROWS * DV * out_bytes),
            )
            if const_expr(DIRECT):
                # The sequence's valid query rows of out; padded rows drop.
                row_bytes = n_kv * (GQA * DV * 2)
                dout = ptr_buf_tensor(
                    uniform64(buf_base_i64(out_ptr) + fx.Int64(first_row) * fx.Int64(row_bytes)), fx.BFloat16,
                    num_records_bytes=fx.Int64(uniform(split_row.select(fx.Int32(0), valid * row_bytes))),
                )
                dout = fx.logical_divide(dout, fx.make_layout(1, 1))
            pmax = ptr_buf_tensor(
                uniform64(buf_base_i64(pmax_ptr) + fx.Int64(partial) * (ROWS * 4)), fx.Float32,
                num_records_bytes=part_bytes(ROWS * 4),
            )
            psum = ptr_buf_tensor(
                uniform64(buf_base_i64(psum_ptr) + fx.Int64(partial) * (ROWS * 4)), fx.Float32,
                num_records_bytes=part_bytes(ROWS * 4),
            )
            pout = fx.logical_divide(pout, fx.make_layout(1, 1))
            # Stores of 4 FP32, 4 bf16 and (SWAPST) 8 bf16 a lane.
            st_f32x4 = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.Float32)
            st_bf16x4 = fx.make_copy_atom(fx.rocdl.BufferCopy64b(), fx.BFloat16)
            st_bf16x8 = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.BFloat16)

            def store_output(fp32_part):
                # fp32_part (direct plans): this task is a piece of a split row, so it stores FP32 O plus m and l;
                # otherwise it stores normalized bf16 rows to out. Only the taken path's stores issue.
                for m in range_constexpr(M):
                    mm = m_fin[m]
                    safe = (mm > neg_inf).select(mm, zero)
                    w = fx.Float32(fx.exp2(mm - safe, fastmath="fast"))
                    lt = l_fin[m] * w
                    inv = (lt > zero).select(fx.Float32(rcp_f32(lt)), zero)
                    wb = fx.full(4, w * inv, fx.Float32)
                    row = queries[m] * GQA + l16
                    if const_expr(DIRECT):
                        # Row of query q, head h in out [tokens, KV heads * GQA, DV].
                        drow = (queries[m] * n_kv + kv_h) * GQA + l16
                    if const_expr(DIRECT and not fp32_part and SWAPST):
                        # Line-friendlier direct stores: pack each 16-column block to bf16 (2 dwords a lane), then
                        # one v_permlane16_swap per dword pair of blocks (2k, 2k + 1) gives every lane 8 adjacent
                        # columns, so each dwordx4 store writes 64 contiguous bytes of each of its 16 rows (was
                        # 32 B a row per dwordx2: twice the write requests for the same bytes).
                        packs = [(o_fin[m][nb] * wb).to(fx.BFloat16).bitcast(fx.Int32) for nb in range_constexpr(NB)]
                        col = (g % 2) * 16 + (g // 2) * 8
                        for k in range_constexpr(NB // 2):
                            xs, ys = [], []
                            for j in range_constexpr(2):
                                x, y = permlane16_swap(packs[2 * k][j], packs[2 * k + 1][j])
                                xs.append(x)
                                ys.append(y)
                            o_reg = fx.make_rmem_tensor(8, fx.BFloat16)
                            o_reg.store(fx.Vector.from_elements(xs + ys, dtype=fx.Int32).bitcast(fx.BFloat16))
                            fx.copy(st_bf16x8, o_reg, fx.slice(dout, (None, drow * DV + 32 * k + col)))
                    for nb in range_constexpr(NB if const_expr(not (DIRECT and not fp32_part and SWAPST)) else 0):
                        o32 = o_fin[m][nb] * wb
                        if const_expr(DIRECT and fp32_part):
                            o_reg = fx.make_rmem_tensor(4, fx.Float32)
                            o_reg.store(o32)
                            fx.copy(st_f32x4, o_reg, fx.slice(pout, (None, row * DV + nb * 16 + g * 4)))
                        else:
                            o_reg = fx.make_rmem_tensor(4, fx.BFloat16)
                            o_reg.store(o32.to(fx.BFloat16))
                            dst, at_row = (dout, drow) if DIRECT else (pout, row)
                            fx.copy(st_bf16x4, o_reg, fx.slice(dst, (None, at_row * DV + nb * 16 + g * 4)))
                    if const_expr(fp32_part or not DIRECT):
                        pmax[row] = fx.Float32(mm * fx.Float32(1.0 / LOG2E))
                        psum[row] = fx.Float32(lt)

            if const_expr(DIRECT):
                if split_row:
                    store_output(True)
                else:
                    store_output(False)
            else:
                store_output(False)

        def idle_wave_loads():
            """A wave whose queries are all padding issues only its DMA share of every tile, at the active waves'
            barriers."""
            if grp == 0:
                fx.rocdl.s_waitcnt(vmcnt=TILE_LOADS)
            else:
                fx.rocdl.s_waitcnt(vmcnt=2 * TILE_LOADS)
            gpu.barrier()
            if grp == 1:
                gpu.barrier()
            if grp == 1:
                fx.rocdl.s_waitcnt(vmcnt=TILE_LOADS)
            gpu.barrier()
            for tile_i in range(t_begin, t_end, 1):
                it_i = fx.Int32(tile_i)
                rel_i = it_i - t_begin
                xt_i = it_i + (LEAD - 1) + grp
                for load in kv_tile_loads(xt_i, smod(rel_i + (LEAD - 1) + grp), xt_i < t_end):
                    load()
                fx.rocdl.s_waitcnt(vmcnt=TILE_LOADS)
                gpu.barrier()
                fx.rocdl.s_waitcnt(vmcnt=TILE_LOADS)
                gpu.barrier()
            if grp == 0:
                gpu.barrier()
            fx.rocdl.s_waitcnt(vmcnt=0)
            gpu.barrier()

        # Waves whose queries are all padding (rows past valid: decode rows have one valid query, so waves 1-7)
        # skip QK, softmax, PV and the stores; they still issue their DMA share of every tile and keep the
        # active waves' barrier count and placement, so the shared K/V ring stays correct.
        active = queries[0] < valid
        if active:
            # Prologue: tile 0 ready for group 0's QK, then group 1 falls one phase behind. Tiles 0-1 (group 0) or
            # 0-2 (group 1) in flight: tile 0 lands first.
            if grp == 0:
                fx.rocdl.s_waitcnt(vmcnt=TILE_LOADS)
            else:
                fx.rocdl.s_waitcnt(vmcnt=2 * TILE_LOADS)
            gpu.barrier()
            if grp == 1:
                gpu.barrier()
            s0 = qk_scores(0)
            if grp == 1:
                fx.rocdl.s_waitcnt(vmcnt=TILE_LOADS)
            gpu.barrier()
            carry = list(init) + list(s0)
            if const_expr(XLEAN):
                carry.append(load_phys(fx.Int32(t_begin + (LEAD - 1) + grp)))
            if const_expr(ILOOP):
                # No init=: FlyDSL builds these scf.for loops on the i32 bounds (scf_for_dispatch).
                if const_expr(STATIC4):
                    assert LAYOUT_SPLIT and STAGES == 4
                    # rel = tile - t_begin is a multiple of 4 at each iteration start, so step s reads stage s.
                    t_four = t_begin + ((t_split - t_begin) & ~3)
                    for tile_i in range(t_begin, t_four, 4):
                        for s4 in range_constexpr(4):
                            carry = stagger_step(fx.Int32(tile_i) + s4, carry, False, s4)
                    for tile_i in range(t_four, t_split, 1):
                        carry = stagger_step(fx.Int32(tile_i), carry, False)
                else:
                    for tile_i in range(t_begin, t_split, 1):
                        carry = stagger_step(fx.Int32(tile_i), carry, False)
                for tile_i in range(t_split, t_end, 1):
                    carry = stagger_step(fx.Int32(tile_i), carry, True)
                final = carry[:STATE]
            else:
                for tile_i, c_in in range(t_begin, t_split, 1, init=carry):
                    c_mid = yield stagger_step(fx.Int32(tile_i), c_in, False)
                for tile_i, c_in in range(t_split, t_end, 1, init=c_mid):
                    c_out = yield stagger_step(fx.Int32(tile_i), c_in, True)
                final = c_out[:STATE]
            if grp == 0:
                gpu.barrier()
            fx.rocdl.s_waitcnt(vmcnt=0)
            gpu.barrier()
            epilogue(final)
        else:
            idle_wave_loads()

        # No closing barrier: the persistent loop's ticket barrier follows, and no LDS is touched since the
        # post-loop barrier.

    @flyc.kernel(name=kernel_name, known_block_size=(NWAVES * WAVE, 1, 1))
    def pa_decode_pingpong_kernel(
        pmax_ptr: fx.Pointer, psum_ptr: fx.Pointer, pout_ptr: fx.Pointer, query_ptr: fx.Pointer,
        key_cache_ptr: fx.Pointer, value_cache_ptr: fx.Pointer, block_tables_ptr: fx.Pointer,
        max_blocks_per_seq: fx.Int32, stride_q_row: fx.Int32, stride_q_head: fx.Int32, work_info_ptr: fx.Pointer,
        work_capacity: fx.Int32, query_map_ptr: fx.Pointer, out_ptr: fx.Pointer, reduce_ptr: fx.Pointer,
        ticket_ptr: fx.Pointer, num_kv_heads: fx.Int32, n_ctas: fx.Int32,
    ):
        # Persistent CTAs: CTA b runs task b, then takes tickets (an atomic counter per KV head, zeroed per call)
        # for tasks n_ctas, n_ctas + 1, ..., so tasks still start in plan order (longest first) on whichever CTA
        # frees up first. The ticket reaches the other waves through the last LDS word of stage 3, which the next
        # task's DMA writes only after its prologue barriers.
        # The LDS ring's byte base: an ir.Value pointer, which scf control flow can carry (SharedStorage cannot).
        lds = fx.recast_iter(fx.Uint8, fx.SharedAllocator().allocate(SharedStorage).peek().buf.ptr)
        lds_word = fx.recast_iter(shared_ptr(fx.Int32), fx.add_offset(lds, fx.make_int_tuple(total_bytes - 4)))
        work = fx.recast_iter(fx.Int32, work_info_ptr)
        tick = fx.add_offset(fx.recast_iter(fx.Int32, ticket_ptr), fx.Int32(fx.block_idx.y))
        # Kernel arguments rather than grid_dim, which calls __ockl_get_num_groups (a real call, here inside the
        # task loop, that pins registers across it).
        n_kv = uniform(num_kv_heads)
        n_ctas = uniform(n_ctas)
        tid0 = fx.Int32(fx.thread_idx.x) == 0
        slot = uniform(fx.block_idx.x)
        while slot < work_capacity:
            task = fx.ptr_load(fx.add_offset(work, slot * 4), result_type=fx.Vector.make_type(4, fx.Int32))
            start = fx.Int32(task[1])
            end = fx.Int32(task[2])
            if start < end:
                _task(
                    lds, pmax_ptr, psum_ptr, pout_ptr, query_ptr, key_cache_ptr, value_cache_ptr, block_tables_ptr,
                    query_map_ptr, max_blocks_per_seq, stride_q_row, stride_q_head, fx.Int32(task[0]), start, end,
                    fx.Int32(task[3]), slot, work_capacity, out_ptr, reduce_ptr, n_kv,
                )
            if tid0:
                t = fx.atomic_add(tick, fx.Int32(1), syncscope=fx.rocdl.SyncScope.Agent)
                fx.ptr_store(fx.Vector.from_elements([fx.Int32(t)], dtype=fx.Int32), lds_word)
            gpu.barrier()
            nxt = fx.ptr_load(lds_word, result_type=fx.Vector.make_type(1, fx.Int32))
            slot = uniform(fx.Int32(nxt[0]) + n_ctas)
            # Every wave has read the ticket before tid 0 can write the next one.
            gpu.barrier()

    @flyc.jit
    def pa_decode_pingpong_launch(
        pmax: fx.Pointer, psum: fx.Pointer, pout: fx.Pointer, query: fx.Pointer, key_cache: fx.Pointer,
        value_cache: fx.Pointer, block_tables: fx.Pointer, out: fx.Pointer, max_blocks_per_seq: fx.Int32,
        num_kv_heads: fx.Int32, stride_q_row: fx.Int32, stride_q_head: fx.Int32, work_info: fx.Pointer,
        work_capacity: fx.Int32, query_map: fx.Pointer, reduce_info: fx.Pointer, ticket: fx.Pointer,
        n_ctas: fx.Int32, stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        _ = kernel_name  # a captured scalar, so FlyDSL's disk cache keys the launch by the name
        pa_decode_pingpong_kernel(
            pmax, psum, pout, query, key_cache, value_cache, block_tables, max_blocks_per_seq, stride_q_row,
            stride_q_head, work_info, work_capacity, query_map, out, reduce_info, ticket, num_kv_heads, n_ctas,
            # Packed FP32 softmax VALU stalls behind the sibling wave's MFMAs; without it 40-47 more SGPR spills and
            # 13x the output error (probe P2). 0.5.0 has no compile hint for target features.
            value_attrs={"llvm.target_features": ir.Attribute.parse('#llvm.target_features<["-packed-fp32-ops"]>')},
        ).launch(grid=(n_ctas, num_kv_heads, 1), block=(NWAVES * WAVE, 1, 1), stream=stream)

    pa_decode_pingpong_launch.compile_hints.update(waves_per_eu=2, fastmath="contract")
    if STRATEGY:
        pa_decode_pingpong_launch.compile_hints["llvm_options"] = {"amdgpu-sched-strategy": STRATEGY}
    return pa_decode_pingpong_launch


# --- pa_decode_plan.py: plan and query-map kernels ----------------------------------------------

# Two warp scans and one reduction use at most 512 bytes at 1024 wave32 threads.
_PLAN_SCAN_LDS_BYTES = 512


def _lds_max_batch(lds_bytes):
    """Largest batch whose task ends and scan storage fit ``lds_bytes``."""
    return (lds_bytes - _PLAN_SCAN_LDS_BYTES) // 16 * 4


def _block_threads(capacity):
    return max(64, min(1024, 1 << (capacity - 1).bit_length()))


@cache
def compile_pa_decode_plan(max_batch, threads, hybrid=False):
    """Compile a batch bucket and block size for dense plans.

    Actual batch, task capacity and partition cap are runtime values. Per
    block: count_tiles, a scan, split_tasks, a scan, record_rows, write_work.
    ``hybrid`` plans give each live row one task, except the last TAIL_ROWS
    live rows at or past row ``window``, which get up to ``max_parts`` tasks
    each; dense plans pass window 0 and ignore it.
    """
    scan_tiles = fx.coop.BlockScan[fx.Int64, threads]
    scan_counts = fx.coop.BlockScan[fx.Int32, threads]
    reduce_rows = fx.coop.BlockReduce[fx.Int32, threads]

    @fx.struct
    class TaskStorage:
        ends: fx.Array[fx.Int32, max_batch, 16]

    def visible_tiles(ctx):
        """256-token KV tiles of a context."""
        return (fx.max(fx.Int64(ctx), fx.Int64(0)) + 255) // 256

    def split_dense(tiles, lower, upper, max_parts):
        """Return a row's task count from its cumulative slot bounds."""
        # Tiles of an int32 length fit Int32, so clamp the partition cap there.
        return fx.min(fx.Int32(fx.min(fx.Int64(tiles > 0) + upper - lower, tiles)), max_parts)

    # Lookup reaches rows below max_batch, so it needs one fewer level than
    # max_batch.bit_length() at power-of-two buckets.
    lookup_bits = (max_batch - 1).bit_length()

    @flyc.kernel(name=f"pa_decode_plan_kernel_b{max_batch}_t{threads}_h{int(hybrid)}_tz", known_block_size=[threads, 1, 1])
    def pa_decode_plan_kernel(
        lengths_ptr: fx.Pointer, work_ptr: fx.Pointer, reduce_ptr: fx.Pointer, batch: fx.Int32, capacity: fx.Int32,
        max_parts: fx.Int32, window: fx.Int32, ticket_ptr: fx.Pointer, num_tickets: fx.Int32,
    ):
        tid = fx.thread_idx.x
        lengths = ptr_buf_tensor(lengths_ptr)
        work = ptr_buf_tensor(work_ptr)
        reduce_info = ptr_buf_tensor(reduce_ptr)
        # Zero the tile kernel's ticket counters (it runs after this kernel on the same stream), so the host
        # need not launch a memset.
        tickets = ptr_buf_tensor(ticket_ptr)
        for t in range(tid, num_tickets, threads):
            tickets[t] = fx.Int32(0)
        smem = fx.SharedAllocator()
        tile_storage = smem.allocate(scan_tiles.SharedStorage).peek()
        count_storage = smem.allocate(scan_counts.SharedStorage).peek()
        row_storage = smem.allocate(reduce_rows.SharedStorage).peek()
        task_ends = smem.allocate(TaskStorage).peek().ends
        # Each thread owns consecutive rows, so scans of per-thread totals
        # preserve sequence order. Buckets that fit the block give each thread
        # at most one row; larger ones walk rows with runtime bounds, so padding
        # rows cost nothing and register use does not grow with the bucket.
        single = max_batch <= threads

        def count_tiles():
            """The thread's (KV tiles, rows with any tile)."""
            if fx.const_expr(single):
                row_ctx = fx.Int32(0)
                if row < batch:
                    row_ctx = lengths[row]
                thread_tiles = visible_tiles(row_ctx)
                thread_rows = fx.Int32(thread_tiles > 0)
            else:
                thread_tiles = fx.Int64(0)
                thread_rows = fx.Int32(0)
                for seq in range(row_begin, row_end):
                    tiles = visible_tiles(lengths[seq])
                    thread_tiles = thread_tiles + tiles
                    thread_rows = thread_rows + fx.Int32(tiles > 0)
            return thread_tiles, thread_rows

        def split(tiles, lower, upper, max_parts, row):
            """A row's task count: hybrid plans split only the tail rows, by TAIL_MIN_PIECE tiles a task."""
            return (
                (tiles > 0).select(
                    (row >= tail_lo).select(
                        fx.Int32(fx.max(fx.min(tiles // TAIL_MIN_PIECE, fx.Int64(max_parts)), fx.Int64(1))),
                        fx.Int32(1),
                    ),
                    fx.Int32(0),
                )
                if hybrid
                else split_dense(tiles, lower, upper, max_parts)
            )

        def split_tasks(task_ends, thread_tiles, cumulative, total, remaining):
            """The thread's task count: the remaining slots in proportion to tiles, each row's in task_ends."""
            if fx.const_expr(single):
                # Empty and padding rows get no tasks; skip their Int64 divisions.
                thread_tasks = fx.Int32(0)
                if thread_tiles > 0:
                    thread_tasks = split(
                        thread_tiles,
                        cumulative * remaining // total,
                        (cumulative + thread_tiles) * remaining // total,
                        max_parts,
                        row,
                    )
            else:
                # Each row's lower bound is the previous row's upper bound.
                lower = cumulative * remaining // total
                thread_tasks = fx.Int32(0)
                for seq in range(row_begin, row_end):
                    tiles = visible_tiles(lengths[seq])
                    cumulative = cumulative + tiles
                    upper = cumulative * remaining // total
                    count = split(tiles, lower, upper, max_parts, seq)
                    lower = upper
                    task_ends[seq] = count
                    thread_tasks = thread_tasks + count
            return thread_tasks

        def record_rows(reduce_info, task_ends, row_start, thread_tasks):
            """Each row's first task and task count in reduce_info, and its task end in task_ends."""
            if fx.const_expr(single):
                if row < batch:
                    reduce_info[row * 2] = row_start
                    reduce_info[row * 2 + 1] = thread_tasks
                    task_ends[row] = row_start + thread_tasks
            else:
                # Rows are thread-private here; convert counts to ends in place.
                for seq in range(row_begin, row_end):
                    count = task_ends[seq]
                    reduce_info[seq * 2] = row_start
                    reduce_info[seq * 2 + 1] = count
                    row_start = row_start + count
                    task_ends[seq] = row_start

        def write_work(work, total_tasks):
            """One work_info record (sequence, first tile, end tile, context) per slot; padding slots get zeros."""
            # Spread packed task writes across all threads, including small batches
            # with many partitions. Upper-bound search skips repeated ends from empty
            # rows and uses one LDS int32 per sequence (16 KiB at the default cap).
            for slot in range(tid, capacity, threads):
                if slot < total_tasks:
                    seq = fx.Int32(-1)
                    for bit in fx.range_constexpr(lookup_bits - 1, -1, -1):
                        candidate = seq + (1 << bit)
                        if candidate < batch:  # noqa: SIM102 -- guard the LDS read
                            if task_ends[candidate] <= slot:
                                seq = candidate
                    seq = seq + 1
                    start = fx.Int32(0)
                    if seq > 0:
                        start = task_ends[seq - 1]
                    count = task_ends[seq] - start
                    ctx = lengths[seq]
                    num_tiles = visible_tiles(ctx)
                    part = fx.Int64(slot - start)
                    work[slot * 4] = seq
                    work[slot * 4 + 1] = fx.Int32(part * num_tiles // count)
                    work[slot * 4 + 2] = fx.Int32((part + 1) * num_tiles // count)
                    work[slot * 4 + 3] = ctx
                else:
                    # Padding is disjoint from active records, even for empty batches.
                    for field in fx.range_constexpr(4):
                        work[slot * 4 + field] = fx.Int32(0)

        if fx.const_expr(single):
            row = fx.Int32(tid)
        else:
            rows = (batch + threads - 1) // threads
            row_begin = fx.min(fx.Int32(tid) * rows, batch)
            row_end = fx.min(row_begin + rows, batch)
        thread_tiles, thread_rows = count_tiles()
        cumulative, total = scan_tiles.exclusive_with_aggregate(thread_tiles, fx.ReductionOp.ADD, storage=tile_storage)
        total = fx.max(total, fx.Int64(1))
        live_rows = reduce_rows(thread_rows, fx.ReductionOp.ADD, storage=row_storage)
        remaining = fx.Int64(capacity - live_rows)
        # Hybrid: live rows are a prefix (sorted longest first); split the last TAIL_ROWS of them.
        tail_lo = fx.max(live_rows - fx.Int32(TAIL_ROWS), window)
        thread_tasks = split_tasks(task_ends, thread_tiles, cumulative, total, remaining)
        row_start, total_tasks = scan_counts.exclusive_with_aggregate(
            thread_tasks, fx.ReductionOp.ADD, storage=count_storage
        )
        # Captured tensors written under a runtime branch would become branch state: pass them in.
        record_rows(reduce_info, task_ends, row_start, thread_tasks)
        fx.gpu.barrier()
        write_work(work, total_tasks)

    @flyc.jit
    def launch(
        lengths: fx.Pointer, work: fx.Pointer, reduce_info: fx.Pointer, batch: fx.Int32, capacity: fx.Int32,
        max_parts: fx.Int32, window: fx.Int32, ticket: fx.Pointer, num_tickets: fx.Int32, stream: fx.Stream,
    ):
        pa_decode_plan_kernel(lengths, work, reduce_info, batch, capacity, max_parts, window, ticket, num_tickets).launch(
            grid=(1, 1, 1), block=(threads, 1, 1), stream=stream
        )

    return launch


@cache
def compile_query_block_map(max_batch):
    """Split varlen queries into blocks of _QUERY_BLOCK in sequence order.

    Slot p of ``query_map`` gets (first query row, sequence, valid queries, 0)
    and ``contexts[p]`` the KV tokens its last query sees; slots past the last
    block get no queries and no context. Batch and capacity are runtime values.

    Live blocks are ordered by context, longest first, ties in sequence order
    (a stable descending sort); each block's position costs one pass over the
    batch.
    """
    threads, query_block = 1024, _QUERY_BLOCK
    scan_blocks = fx.coop.BlockScan[fx.Int32, threads]

    @fx.struct
    class EndStorage:
        ends: fx.Array[fx.Int32, max_batch, 16]
        # KV minus query tokens and KV tokens per sequence, for the ordering.
        offsets: fx.Array[fx.Int32, max_batch, 16]
        lengths: fx.Array[fx.Int32, max_batch, 16]

    lookup_bits = (max_batch - 1).bit_length()

    def blocks_of(cu, seq):
        return (cu[seq + 1] - cu[seq] + (query_block - 1)) // query_block

    @flyc.kernel(known_block_size=[threads, 1, 1])
    def query_block_map_kernel(
        cu_seqlens_ptr: fx.Pointer, lengths_ptr: fx.Pointer, map_ptr: fx.Pointer, contexts_ptr: fx.Pointer,
        batch: fx.Int32, capacity: fx.Int32, num_tokens: fx.Int32,
    ):
        tid = fx.thread_idx.x
        cu = ptr_buf_tensor(cu_seqlens_ptr)
        lengths = ptr_buf_tensor(lengths_ptr)
        query_map = ptr_buf_tensor(map_ptr)
        contexts = ptr_buf_tensor(contexts_ptr)
        smem = fx.SharedAllocator()
        scan_storage = smem.allocate(scan_blocks.SharedStorage).peek()
        storage = smem.allocate(EndStorage).peek()
        block_ends = storage.ends

        def record_blocks(storage):
            """Each sequence's query-block end, KV minus query tokens and KV tokens, into storage."""
            rows = (batch + threads - 1) // threads
            row_begin = fx.min(fx.Int32(tid) * rows, batch)
            row_end = fx.min(row_begin + rows, batch)
            thread_blocks = fx.Int32(0)
            for seq in range(row_begin, row_end):
                thread_blocks = thread_blocks + blocks_of(cu, seq)
            row_start, _ = scan_blocks.exclusive_with_aggregate(thread_blocks, fx.ReductionOp.ADD, storage=scan_storage)
            for seq in range(row_begin, row_end):
                row_start = row_start + blocks_of(cu, seq)
                storage.ends[seq] = row_start
                storage.offsets[seq] = lengths[seq] - (cu[seq + 1] - cu[seq])
                storage.lengths[seq] = lengths[seq]

        def rank_of(seq, block, num_blocks):
            """Blocks sorting before (seq, block): longer, or as long and earlier.

            Ordering keys ignore num_tokens truncation and the clamp at zero:
            block b of n has key offset + b_end with b_end = b + 1 blocks of
            queries, except the last, whose key is the KV length. Keys rise
            strictly within a sequence, so each sequence's count is closed-form.
            """
            offset = storage.offsets[seq]
            key = (block == num_blocks - 1).select(storage.lengths[seq], offset + (block + 1) * query_block)
            rank = fx.Int32(0)
            for other in range(0, batch):  # noqa: PIE808 -- preserve reviewed FlyDSL IR
                prev = block_ends[fx.max(other - 1, 0)]
                blocks = block_ends[other] - (other > 0).select(prev, fx.Int32(0))
                gap = key - storage.offsets[other]
                last = storage.lengths[other]
                # Non-last blocks with keys <= key: b_end in [1, gap // qb].
                inner = fx.max(blocks - 1, 0)
                below = fx.min(fx.max(gap // query_block, 0), inner)
                longer = fx.Int32(last > key) + inner - below
                ties = fx.Int32(last == key) + fx.Int32(
                    (gap >= query_block) & (gap % query_block == 0) & (gap // query_block <= inner)
                )
                live = blocks > 0
                rank = rank + live.select(longer, fx.Int32(0))
                rank = rank + (live & (other < seq)).select(ties, fx.Int32(0))
            return rank

        def write_map(query_map, contexts):
            """Each slot's query block (first row, sequence, valid queries, 0) and context, at its rank."""
            last_row = fx.max(num_tokens - 1, 0)
            total_blocks = block_ends[batch - 1]
            for slot in range(tid, capacity, threads):
                seq = fx.Int32(-1)
                for bit in fx.range_constexpr(lookup_bits - 1, -1, -1):
                    candidate = seq + (1 << bit)
                    if candidate < batch:  # noqa: SIM102 -- guard the LDS read
                        if block_ends[candidate] <= slot:
                            seq = candidate
                live = slot < total_blocks
                seq = fx.min(seq + 1, batch - 1)
                first_block = fx.Int32(0)
                if seq > 0:
                    first_block = block_ends[seq - 1]
                query0 = cu[seq]
                seq_queries = cu[seq + 1] - query0
                first = (slot - first_block) * query_block
                row = query0 + first
                valid = fx.min(fx.min(seq_queries - first, num_tokens - row), query_block)
                valid = live.select(fx.max(valid, 0), 0)
                ctx = fx.max(lengths[seq] - seq_queries + first + valid, 0)
                # Dead slots keep their place after every live block.
                block = slot - first_block
                rank = rank_of(seq, block, block_ends[seq] - first_block)
                dest = live.select(rank, slot)
                query_map[dest * 4] = fx.min(fx.max(row, 0), last_row)
                query_map[dest * 4 + 1] = seq
                query_map[dest * 4 + 2] = valid
                query_map[dest * 4 + 3] = fx.Int32(0)
                contexts[dest] = live.select(ctx, 0)

        # Captured tensors written under a runtime branch would become branch state: pass them in.
        record_blocks(storage)
        fx.gpu.barrier()
        write_map(query_map, contexts)

    @flyc.jit
    def launch(
        cu_seqlens: fx.Pointer, lengths: fx.Pointer, query_map: fx.Pointer, contexts: fx.Pointer, batch: fx.Int32,
        capacity: fx.Int32, num_tokens: fx.Int32, stream: fx.Stream,
    ):
        query_block_map_kernel(cu_seqlens, lengths, query_map, contexts, batch, capacity, num_tokens).launch(
            grid=(1, 1, 1), block=(threads, 1, 1), stream=stream
        )

    return launch


# --- pa_decode_reduce.py: partitioned-softmax reducer -------------------------------------------

# Above this partition bound the CTA reduces in LDS; at most DUAL_REGISTER
# live partitions still take its register branch.
REGISTER_PARTITIONS = 64
DUAL_REGISTER_PARTITIONS = 32
MAX_CTA_WAVES = 16


def _shuffle_offsets(parts):
    width = 1 if parts == 1 else 1 << (parts - 1).bit_length()
    return tuple(o for o in (32, 16, 8, 4, 2, 1) if o < width)


def _fast_div(value, magic):
    """floor(value / d) for magic = ceil(2**30 / d), exact while value * d <= 2**30."""
    return fx.Int32((fx.Int64(value) * fx.Int64(magic)) >> 30)


@lru_cache(maxsize=256)
def compile_pa_decode_ps_reduce(partitions, groups, num_xcds, hybrid=False):
    """Build PA decode's partitioned-softmax reducer for bf16 partials and output, DV 128, GQA 16.

    One CTA reduces one (sequence, KV head, query row). Logits are packed
    [KV heads, capacity, rows, D] rows whose span of NP partitions per sequence
    is at most INT32_MAX bytes: a per-sequence buffer bound zero-fills
    inactive partitions without reading adjacent sequences or NaN padding.
    Each lane takes two adjacent bf16 elements; output stores stay scalar.

    NP up to the wave size builds a register-only CTA: lane p owns partition
    p's statistics and a CTA-uniform count branch selects one unrolled rung.
    Larger bounds build a dual-mode CTA of ``groups`` partition groups.
    Up to 32 live partitions, group 0 runs the register ladder; above, wave 0
    publishes the softmax weights to LDS, each group sums a contiguous chunk
    and group 0 adds the partials.

    ``num_xcds`` > 0 launches a 1-D grid that gives each XCD a contiguous run
    of CTAs; the launch then needs ``rows_magic`` and ``seq_magic`` =
    ceil(2**30 / d) for d = rows and rows * KV heads, valid while
    batch * (rows * KV heads)**2 <= 2**30.

    The launch takes an int32 [batch, 4] map of (first output row,
    table row, valid queries, unused) per sequence: query q of a sequence is
    output row first + q, and rows q >= valid are not stored.
    ``stride_output_bs`` is unused.
    """
    wave = WAVE
    head_size, elems, query_group_size = _VALUE_DIM, 2, _GQA
    group_waves = head_size // elems // wave
    dual = partitions > REGISTER_PARTITIONS
    assert groups & (groups - 1) == 0 and (groups == 1 or dual) and groups * group_waves <= MAX_CTA_WAVES

    low_rungs = high_rungs = dual_rungs = lds_rungs = ()
    if not dual:
        block = [head_size // elems, 1, 1]
        # NP <= 8 is one rung; above, test c > 8 first, then the largest rung
        # in each half.
        low_rungs = (1, 2, 4, 8)
        high_rungs = tuple(r for r in (16, 32) if r < partitions) + (partitions,)
    else:
        block = [wave, group_waves * groups, 1]
        dual_rungs = (1, 2, 4, 8, 16, DUAL_REGISTER_PARTITIONS)
        # Group g owns partitions [g * R, (g + 1) * R) for the selected rung R.
        # Above 32 live partitions the first rung covers up to 64.
        per_group = -(-partitions // groups)
        rung = max(1, 2 * DUAL_REGISTER_PARTITIONS // groups)
        while rung < per_group:
            lds_rungs += (rung,)
            rung *= 2
        lds_rungs += (per_group,)
    # Groups tile whole wave-sized weight chunks, so their slots stay in range.
    weight_chunks = -(-partitions // wave)

    @fx.struct
    class SharedStorage:
        weights: fx.Array[fx.Float32, weight_chunks * wave, 16]
        partials: fx.Array[fx.Float32, max(1, (groups - 1) * head_size), 16]

    kernel_name = f"pa_decode_reduce_kernel_obf16_d{head_size}_np{partitions}_gqa{query_group_size}_pairs"
    if groups > 1:
        kernel_name += f"_g{groups}"
    if num_xcds:
        kernel_name += f"_xcd{num_xcds}"
    kernel_name += "_qmap"
    if hybrid:
        # FP32 partials of a direct plan's split tail rows; rows with one task were stored directly.
        kernel_name += "_hybf32"
    out_bytes = 2

    @flyc.kernel(name=kernel_name, known_block_size=block)
    def pa_decode_reduce_kernel(
        output_ptr: fx.Pointer, exp_sums_ptr: fx.Pointer, max_logits_ptr: fx.Pointer, logits_ptr: fx.Pointer,
        sink_token_ptr: fx.Pointer,  # null: no sinks (kept so the signature matches the branch)
        stride_output_bs: fx.Int32, stride_output_len: fx.Int32, stride_output_kv_head: fx.Int32,
        stride_output_group_size: fx.Int32, stride_exp_sums_head: fx.Int32, stride_exp_sums_part: fx.Int32,
        stride_logits_head: fx.Int32, stride_logits_part: fx.Int32, stride_logits_group: fx.Int32,
        reduce_info_ptr: fx.Pointer, query_rows: fx.Int32, num_kv_heads: fx.Int32, num_ctas: fx.Int32,
        rows_magic: fx.Int32, seq_magic: fx.Int32, query_map_ptr: fx.Pointer,
    ):
        if fx.const_expr(num_xcds > 0):
            # Hardware deals linear workgroup ids round-robin over XCDs. Give
            # each XCD a contiguous run of (sequence, KV head, row) so rows
            # whose statistics share cache lines share an L2.
            linear = fx.block_idx.x
            xcd = urem_pow2(linear, num_xcds)
            per_xcd = udiv_pow2(num_ctas, num_xcds)
            extra = urem_pow2(num_ctas, num_xcds)
            logical = xcd * per_xcd + (xcd < extra).select(xcd, extra) + udiv_pow2(linear, num_xcds)
            batch_idx = _fast_div(logical, seq_magic)
            inner = logical - batch_idx * (query_rows * num_kv_heads)
            kv_head_idx = _fast_div(inner, rows_magic)
            row_idx = inner - kv_head_idx * query_rows
        else:
            batch_idx = fx.block_idx.x
            kv_head_idx = fx.block_idx.y
            row_idx = fx.block_idx.z

        exp_sums = fx.recast_iter(fx.Float32, exp_sums_ptr)
        max_logits = fx.recast_iter(fx.Float32, max_logits_ptr)
        reduce_info = fx.recast_iter(fx.Int32, reduce_info_ptr)

        zero_f = fx.Float32(0.0)
        one_f = fx.Float32(1.0)
        neg_inf = fx.Float32(float("-inf"))
        c_log2e = fx.Float32(LOG2E)
        zero_i = fx.Int32(0)
        c_wave = fx.Int32(wave)
        lane = fx.thread_idx.x & fx.Int32(wave - 1)
        group_idx = urem_pow2(row_idx, query_group_size)
        query_idx = udiv_pow2(row_idx, query_group_size)
        qmap = fx.recast_iter(fx.Int32, query_map_ptr)
        first_row = fx.Int32(qmap[batch_idx * 4])
        valid_rows = fx.Int32(qmap[batch_idx * 4 + 2])
        row_elem = (
            (first_row + query_idx) * stride_output_len
            + kv_head_idx * stride_output_kv_head
            + group_idx * stride_output_group_size
        )
        # Padded query rows get an empty resource: their stores are dropped.
        first_part = fx.Int32(reduce_info[batch_idx * 2])
        part_count = fx.Int32(reduce_info[batch_idx * 2 + 1])
        store_row = query_idx < valid_rows
        if fx.const_expr(hybrid):
            store_row = store_row & (part_count > fx.Int32(1))
        output = ptr_buf_tensor(
            buf_base_i64(output_ptr) + fx.Int64(row_elem) * out_bytes, fx.BFloat16, unit_elems=elems,
            num_records_bytes=fx.Int64(store_row.select(fx.Int32(head_size * out_bytes), zero_i)),
        )
        store_atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.BFloat16)

        # Empty sequences may point past packed storage; clamped loads need
        # slot 0. Every inactive value is masked before arithmetic.
        first_part = (part_count > zero_i).select(first_part, zero_i)
        stats_seq_offset = first_part * stride_exp_sums_part
        # Rebase in i64 before narrowing: packed storage can exceed 2 GiB. A
        # count of 0 gives a zero-sized descriptor that returns zeros.
        logits_item_bytes = 4 if hybrid else 2
        logits_dtype = fx.Float32 if hybrid else fx.BFloat16
        logits_buffer = ptr_buf_tensor(
            buf_base_i64(logits_ptr)
            + (
                fx.Int64(first_part) * fx.Int64(stride_logits_part)
                + fx.Int64(kv_head_idx) * fx.Int64(stride_logits_head)
            )
            * logits_item_bytes,
            logits_dtype,
            unit_elems=elems,
            unit_stride=1,
            num_records_bytes=fx.Int64(part_count) * fx.Int64(stride_logits_part) * logits_item_bytes,
        )

        if fx.const_expr(dual):
            worker = fx.thread_idx.y
            partition_group = udiv_pow2(worker, group_waves)
            unit = urem_pow2(worker, group_waves) * c_wave + lane
        else:
            unit = fx.thread_idx.x

        def _zero():
            return fx.full(2, 0.0, fx.Float32)

        def _load_logits(part, unit):
            """Two adjacent elements of partition part's partial: FP32 (hybrid) or bf16."""
            index = part * stride_logits_part + row_idx * stride_logits_group + unit * fx.Int32(elems)
            fragment = fx.make_rmem_tensor(elems, logits_dtype)
            atom = fx.make_copy_atom((fx.rocdl.BufferCopy64b if hybrid else fx.rocdl.BufferCopy32b)(0), logits_dtype)
            fx.copy(atom, fx.slice(logits_buffer, (index, None)), fragment)
            return fx.Vector(fragment.load())

        def _f32(value):
            return value if hybrid else value.to(fx.Float32)

        def _store(acc, unit):
            fragment = fx.make_rmem_tensor(elems, fx.BFloat16)
            fragment.store(acc.to(fx.BFloat16))
            fx.copy(store_atom, fragment, fx.slice(output, (unit, None)))

        def _lane_stats(chunk):
            # Lane l owns partition l + wave * chunk; part_count <= NP.
            part = lane + fx.Int32(chunk * wave)
            live = part < part_count
            offset = (
                stats_seq_offset
                + kv_head_idx * stride_exp_sums_head
                + live.select(part, zero_i) * stride_exp_sums_part
                + row_idx
            )
            part_sum = live.select(fx.Float32(exp_sums[offset]), zero_f)
            part_max = live.select(fx.Float32(max_logits[offset]), neg_inf)
            return part_sum, part_max

        def _softmax_weights(sums, maxes, offsets):
            lane_max = maxes[0]
            for part_max in maxes[1:]:
                lane_max = lane_max.maximumf(part_max)
            for offset in offsets:
                lane_max = lane_max.maximumf(lane_max.shuffle_xor(fx.Int32(offset), c_wave))
            safe_max = (lane_max > neg_inf).select(lane_max, zero_f)
            scaled = []
            for part_sum, part_max in zip(sums, maxes):
                scale = (part_max > neg_inf).select(fx.exp2((part_max - safe_max) * c_log2e, fastmath="fast"), zero_f)
                scaled.append(part_sum * scale)
            exp_sum = scaled[0]
            for value in scaled[1:]:
                exp_sum = exp_sum + value
            for offset in offsets:
                exp_sum = fx.Float32(exp_sum.addf(exp_sum.shuffle_xor(fx.Int32(offset), c_wave), fastmath="fast"))
            inv_exp_sum = fx.Float32(rcp_f32((exp_sum > zero_f).select(exp_sum, one_f)))
            return [value * inv_exp_sum for value in scaled]

        def _reduce_register(rung, unit):
            part_sum, part_max = _lane_stats(0)
            # Loads stay in the logits dtype until accumulation, so a rung
            # holds half the registers of its FP32 values.
            logits = [_load_logits(fx.Int32(part), unit) for part in fx.range_constexpr(rung)]
            (weight,) = _softmax_weights([part_sum], [part_max], _shuffle_offsets(rung))
            acc = _zero()
            for part in fx.range_constexpr(rung):
                part_weight = fx.gpu.shuffle_idx(weight, part, wave)
                # Preserve increasing-part order and separate multiply/add rounding.
                acc = acc + _f32(logits[part]) * part_weight
            return acc

        def _top_first(rungs, reduce, unit, scale=1):
            # CTA-uniform count: the largest rung is the first test and the
            # smallest the fall-through. Branches carry the same acc type.
            acc = _zero()
            if fx.const_expr(len(rungs) == 1):
                acc = reduce(rungs[0], unit)
            else:
                if part_count > fx.Int32(rungs[-2] * scale):
                    acc = reduce(rungs[-1], unit)
                else:
                    acc = _top_first(rungs[:-1], reduce, unit, scale)
            return acc

        def _bottom_first(rungs, reduce, unit):
            acc = _zero()
            if fx.const_expr(len(rungs) == 1):
                acc = reduce(rungs[0], unit)
            else:
                if part_count <= fx.Int32(rungs[0]):
                    acc = reduce(rungs[0], unit)
                else:
                    acc = _bottom_first(rungs[1:], reduce, unit)
            return acc

        if fx.const_expr(hybrid):
            # Rows with one task were stored by the tile kernel.
            if part_count > fx.Int32(1):
                _store(_reduce_register(partitions, unit), unit)
        elif fx.const_expr(not dual):
            # Select one unrolled rung, so loads past it are skipped rather
            # than masked.
            acc = _zero()
            if fx.const_expr(partitions <= 8):
                acc = _reduce_register(partitions, unit)
            else:
                if part_count > 8:
                    acc = _top_first(high_rungs, _reduce_register, unit)
                else:
                    acc = _top_first(low_rungs, _reduce_register, unit)
            _store(acc, unit)
        else:

            def _publish_weights(weights):
                # Every thread reaches each barrier; the count is CTA-uniform.
                if worker == zero_i:
                    sums, maxes = [], []
                    for chunk in fx.range_constexpr(weight_chunks):
                        part_sum, part_max = _lane_stats(chunk)
                        sums.append(part_sum)
                        maxes.append(part_max)
                    chunk_weights = _softmax_weights(sums, maxes, _shuffle_offsets(wave))
                    for chunk in fx.range_constexpr(weight_chunks):
                        weights[lane + fx.Int32(chunk * wave)] = chunk_weights[chunk]
                fx.gpu.barrier()

            def _group_sum(rung, unit, weights):
                # Inactive slots have zero weight and load zero past the
                # sequence bound, so they add exactly +0.
                group_acc = _zero()
                begin = partition_group * fx.Int32(rung)
                if begin < part_count:
                    for local in fx.range_constexpr(rung):
                        part = begin + fx.Int32(local)
                        weight = fx.Float32(weights[part])
                        group_acc = group_acc + _f32(_load_logits(part, unit)) * weight
                return group_acc

            def _combine(acc, partials):
                if fx.const_expr(groups > 1):
                    # Idle groups publish +0, so the combine needs no range checks.
                    if partition_group > zero_i:
                        partials[partition_group - 1, unit * 2] = acc[0]
                        partials[partition_group - 1, unit * 2 + 1] = acc[1]
                    fx.gpu.barrier()
                    if partition_group == zero_i:
                        for other in fx.range_constexpr(1, groups):
                            acc = acc + fx.Vector.from_elements(
                                [partials[other - 1, unit * 2], partials[other - 1, unit * 2 + 1]], dtype=fx.Float32
                            )
                return acc

            def _reduce_lds():
                storage = fx.SharedAllocator().allocate(SharedStorage).peek()
                weights = storage.weights
                partials = storage.partials.view(
                    fx.make_layout((max(1, groups - 1), head_size), (head_size, 1))
                )
                _publish_weights(weights)

                def _sum(rung, unit):
                    return _group_sum(rung, unit, weights)

                return _combine(_top_first(lds_rungs, _sum, unit, groups), partials)

            if part_count <= fx.Int32(DUAL_REGISTER_PARTITIONS):
                # Few live partitions skip the weight exchange.
                if partition_group == zero_i:
                    ladder_acc = _bottom_first(dual_rungs, _reduce_register, unit)
                    _store(ladder_acc, unit)
            else:
                lds_acc = _reduce_lds()
                if partition_group == zero_i:
                    _store(lds_acc, unit)

    @flyc.jit
    def launch_pa_decode_ps_reduce_varlen(
        output: fx.Pointer, exp_sums: fx.Pointer, max_logits: fx.Pointer, logits: fx.Pointer, sink_token: fx.Pointer,
        stride_output_bs: fx.Int32, stride_output_len: fx.Int32, stride_output_kv_head: fx.Int32,
        stride_output_group_size: fx.Int32, stride_exp_sums_head: fx.Int32, stride_exp_sums_part: fx.Int32,
        stride_logits_head: fx.Int32, stride_logits_part: fx.Int32, stride_logits_group: fx.Int32,
        query_seq_len: fx.Int32, batch_size: fx.Int32, num_kv_heads: fx.Int32, reduce_info: fx.Pointer,
        rows_magic: fx.Int32, seq_magic: fx.Int32, query_map_ptr: fx.Pointer, stream: fx.Stream,
    ):
        query_rows = query_seq_len * fx.Int32(query_group_size)
        num_ctas = batch_size * num_kv_heads * query_rows
        pa_decode_reduce_kernel(
            output, exp_sums, max_logits, logits, sink_token, stride_output_bs, stride_output_len,
            stride_output_kv_head, stride_output_group_size, stride_exp_sums_head, stride_exp_sums_part,
            stride_logits_head, stride_logits_part, stride_logits_group, reduce_info, query_rows, num_kv_heads,
            num_ctas, rows_magic, seq_magic, query_map_ptr,
        ).launch(
            grid=(num_ctas, 1, 1) if num_xcds > 0 else (batch_size, num_kv_heads, query_rows),
            block=tuple(block),
            stream=stream,
        )

    return launch_pa_decode_ps_reduce_varlen


# --- host launches -------------------------------------------------------------------------------


def launch_pa_decode_ps_reduce(
    output,
    exp_sums,
    max_logits,
    logits,
    max_partitions,
    reduce_info,
    query_map,
    stream,
    live,
    num_cu,
    hybrid=False,
):
    """Reduce packed partials [KV heads, capacity, rows, DV] into ``output``; every sequence has at most
    ``max_partitions``. ``query_map`` (int32 [batch, 4]) places each sequence's query rows."""
    batch, kv_heads = output.shape[0], output.shape[2]
    rows = _QUERY_BLOCK * _GQA
    # Exact bounds keep the register ladder short; above a wave a power of two
    # bounds the specialization count. Per-sequence buffer bounds zero-fill
    # partitions beyond the live count, so rounding past the bound is safe.
    partitions = max_partitions
    if partitions > REGISTER_PARTITIONS:
        partitions = 1 << (partitions - 1).bit_length()
    # A per-sequence span <= INT32_MAX keeps every attempted byte offset,
    # including inactive partitions, below 2**32, so out-of-bounds reads
    # cannot wrap back into valid data.
    if hybrid:
        # (address, strides) of FP32 partials whose base is shifted before the allocation.
        logits_addr, logits_strides = logits
        logits_arg = flyc.from_c_void_p(fx.Float32, logits_addr)
        item = 4
    else:
        logits_strides = logits.stride()
        logits_arg = ptr_arg(logits, fx.BFloat16)
        item = logits.element_size()
    assert logits_strides[0] <= 2**31 - 1 and partitions * logits_strides[1] * item <= 2**31 - 1
    ctas = batch * kv_heads * rows
    groups, num_xcds = 1, 0
    if partitions > REGISTER_PARTITIONS:
        # Partition groups fill the CTA while the grid stays within 16 waves
        # per CU. D128 sequences averaging at most 32 partitions keep one.
        group_waves = _VALUE_DIM // WAVE // 2
        budget = MAX_CTA_WAVES * num_cu
        max_groups = MAX_CTA_WAVES // group_waves
        if exp_sums.shape[1] <= DUAL_REGISTER_PARTITIONS * batch:
            max_groups = 1
        while 2 * groups <= max_groups and 2 * groups * group_waves * ctas <= budget:
            groups *= 2
        per_seq = rows * kv_heads
        if ctas * group_waves >= budget and batch * per_seq * per_seq <= 2**30:
            num_xcds = NUM_XCDS
    _run_compiled(
        compile_pa_decode_ps_reduce(partitions, groups, num_xcds, hybrid),
        ptr_arg(output, fx.BFloat16), ptr_arg(exp_sums, fx.Float32), ptr_arg(max_logits, fx.Float32),
        logits_arg, flyc.from_c_void_p(fx.BFloat16, 0),
        0, *output.stride()[1:4], *max_logits.stride()[:2], *logits_strides[:3],
        _QUERY_BLOCK, batch, kv_heads, ptr_arg(reduce_info, fx.Int32),
        -(-(1 << 30) // rows), -(-(1 << 30) // (rows * kv_heads)),  # rows_magic, seq_magic
        ptr_arg(query_map, fx.Int32), stream, live=live,
    )


def build_query_block_map(cu_seqlens_q, seqused_k, num_tokens, query_map, contexts, stream, live):
    """Fill ``query_map`` [capacity, 4] and ``contexts`` [capacity] on the GPU."""
    batch = seqused_k.numel()
    _run_compiled(
        compile_query_block_map(1 << (batch - 1).bit_length()),
        ptr_arg(cu_seqlens_q, fx.Int32), ptr_arg(seqused_k, fx.Int32), ptr_arg(query_map, fx.Int32),
        ptr_arg(contexts, fx.Int32), batch, query_map.shape[0], num_tokens, stream, live=live,
    )


def plan_pa_decode(context_lengths, work_info, reduce_info, max_partitions, stream, live, tail_lo=None, ticket=None):
    """Fill the dense plan (no window): ``work_info`` [capacity, 4] holds sequence,
    begin/end absolute 256-token tile and context length per task; ``reduce_info``
    [batch, 2] each sequence's first packed task and task count."""
    batch = context_lengths.numel()
    capacity = work_info.shape[0]
    assert (batch * 4 + 15) // 16 * 16 + _PLAN_SCAN_LDS_BYTES <= LDS_BYTES, "batch exceeds the planner's LDS"
    assert 1 <= max_partitions and batch <= capacity <= batch * max_partitions
    # A cap below the power-of-two bound has the same lookup depth and row path,
    # so buckets ignore it. Clip the last bucket so rounding up cannot reject a
    # batch that fits LDS.
    max_batch = min(_lds_max_batch(LDS_BYTES), 1 << (batch - 1).bit_length())
    _run_compiled(
        compile_pa_decode_plan(max_batch, _block_threads(capacity), tail_lo is not None),
        ptr_arg(context_lengths, fx.Int32), ptr_arg(work_info, fx.Int32), ptr_arg(reduce_info, fx.Int32),
        batch, capacity, max_partitions, 0 if tail_lo is None else tail_lo, ptr_arg(ticket, fx.Int32),
        ticket.numel(), stream, live=live,
    )


# --- unified_attention_bf16.py: the varlen route -------------------------------------------------


def _pseudo_capacity(num_tokens, num_seqs, block=_QUERY_BLOCK):
    # Sequence i has ceil(q_i / b) <= q_i / b + (b - 1) / b query blocks.
    return -(-num_tokens // block) + num_seqs


def _plan_capacity(num_seqs, num_kv_heads, num_cu):
    # plan_pa_decode's default: a 2-per-CU task budget, at least one per sequence.
    return min(num_seqs * num_cu, max(num_seqs, -(-2 * num_cu // num_kv_heads)))


def _build_query_map(cu_seqlens_q, seqused_k, num_tokens, query_map, context_lengths):
    """Fill pseudo-sequence rows (first query row, sequence, valid queries, 0).

    Each sequence's queries split into blocks of _QUERY_BLOCK; block b of a
    sequence with q queries and k KV tokens sees k - q + _QUERY_BLOCK b + valid
    tokens. Rows past the last block get no queries and no context. Rows are
    sorted by context, longest first, so the longest tasks start first. Queries
    at or past ``num_tokens`` are dropped, so offsets beyond q never address
    memory.
    """
    capacity = query_map.shape[0]
    dev = cu_seqlens_q.device
    query_lengths = (cu_seqlens_q[1:] - cu_seqlens_q[:-1]).long()
    blocks = (query_lengths + (_QUERY_BLOCK - 1)) // _QUERY_BLOCK
    block_ends = torch.cumsum(blocks, 0)
    pseudo = torch.arange(capacity, device=dev)
    seq = torch.searchsorted(block_ends, pseudo, right=True)
    live = seq < blocks.numel()
    seq = seq.clamp_(max=blocks.numel() - 1)
    first = (pseudo - (block_ends[seq] - blocks[seq])) * _QUERY_BLOCK
    row = cu_seqlens_q[seq].long() + first
    valid = torch.minimum(query_lengths[seq] - first, num_tokens - row)
    valid = torch.where(live, valid.clamp(0, _QUERY_BLOCK), 0)
    context = torch.where(live, seqused_k[seq].long() - query_lengths[seq] + first + valid, 0).clamp_(min=0)
    order = torch.argsort(context, descending=True, stable=True)
    rows = torch.stack((row.clamp(0, max(num_tokens - 1, 0)), seq, valid, torch.zeros_like(seq)), 1)
    query_map.copy_(rows[order])
    context_lengths.copy_(context[order])


def _check(q, k, v, out, cu_seqlens_q, seqused_k, block_table, max_seqlen_q, max_seqlen_k):
    # The branch's route checks (unified_attention_bf16._common_decline and _route).
    num_tokens, num_q_heads, head_dim = q.shape
    num_blocks, block_size, num_kv_heads, key_dim = k.shape
    num_seqs = seqused_k.numel()
    assert all(t.dtype == torch.bfloat16 for t in (q, k, v, out)), "bf16 q/k/v/out"
    assert (head_dim, key_dim) == (_HEAD_DIM, _HEAD_DIM), "Q/K head dim 192"
    assert tuple(v.shape) == (num_blocks, block_size, num_kv_heads, _VALUE_DIM)
    assert tuple(out.shape) == (num_tokens, num_q_heads, _VALUE_DIM)
    assert block_size in _PAGE_SIZES, f"page size in {_PAGE_SIZES}"
    assert num_kv_heads >= 1 and num_q_heads == _GQA * num_kv_heads, "16 query heads per KV head"
    assert k.is_contiguous() and v.is_contiguous() and out.is_contiguous()
    assert q.stride(2) == 1, "q must have a contiguous head_dim axis"
    for t in (seqused_k, block_table, cu_seqlens_q):
        assert t.dtype == torch.int32 and t.is_contiguous() and t.device == q.device
    assert 1 <= num_seqs <= _MAX_BATCH, f"1..{_MAX_BATCH} sequences"
    assert block_table.dim() == 2 and block_table.shape[0] == num_seqs
    assert 0 <= int(max_seqlen_k) <= block_table.shape[1] * block_size
    assert cu_seqlens_q.shape == (num_seqs + 1,)
    assert int(max_seqlen_q) >= 1
    assert _pseudo_capacity(num_tokens, num_seqs) <= _MAX_PSEUDO_SEQS, "query tokens exceed the varlen planner"


def _compile_development_attention(
    q,
    k,
    v,
    out,
    cu_seqlens_q,
    seqused_k,
    block_table,
    max_seqlen_q,
    max_seqlen_k,
    softmax_scale,
    num_cu=None,
):
    """Standalone development entry point retained from the reviewed artifact.

    Production routing uses :func:`run` below with managed workspace.  This
    helper intentionally remains useful for offline kernel compilation, but
    allocates its own tensors and must not be used during graph capture.
    """
    _check(q, k, v, out, cu_seqlens_q, seqused_k, block_table, max_seqlen_q, max_seqlen_k)
    live = q.is_cuda
    num_tokens = q.shape[0]
    num_seqs = seqused_k.numel()
    num_kv_heads = k.shape[2]
    rows = _QUERY_BLOCK * _GQA
    pseudo = _pseudo_capacity(num_tokens, num_seqs)
    dev = q.device
    with torch.cuda.device(dev) if live else contextlib.nullcontext():
        stream = torch.cuda.current_stream(dev) if live else fx.Stream(None)
        if live:
            num_cu = torch.cuda.get_device_properties(dev).multi_processor_count
        elif num_cu is None:
            raise ValueError("CPU-only compilation requires an explicit num_cu")
        capacity = _plan_capacity(pseudo, num_kv_heads, num_cu)
        # With a slot per pseudo-sequence the batch already fills the CUs:
        # one task each, written straight to out, and no reducer.
        direct = capacity == pseudo and out.stride() == (q.shape[1] * _VALUE_DIM, _VALUE_DIM, 1)
        # Otherwise spare slots go to the longest rows: a row has at most one
        # task plus the slots beyond the live rows, of which there are at least
        # ceil(tokens / b).
        max_partitions = (
            1
            if direct
            else min(num_cu, capacity - -(-num_tokens // _QUERY_BLOCK) + 1)
        )
        tail_lo = None
        if direct:
            # Split the shortest live rows, dispatched last, into up to TAIL_SPLIT tasks. Live rows are a prefix
            # of at least pseudo - num_seqs rows, so the last `tail` of them sit at or past tail_lo.
            tail = min(TAIL_ROWS, pseudo)
            tail_lo = max(0, pseudo - num_seqs - tail)
            capacity = pseudo + tail * (TAIL_SPLIT - 1)
            max_partitions = TAIL_SPLIT
        # shortcut: a fresh workspace per call (the branch keeps a grow-only one per stream).
        query_map = torch.empty((pseudo, 4), dtype=torch.int32, device=dev)
        context_lengths = torch.empty(pseudo, dtype=torch.int32, device=dev)
        work_info = torch.empty((capacity, 4), dtype=torch.int32, device=dev)
        reduce_info = torch.empty((pseudo, 2), dtype=torch.int32, device=dev)
        # Persistent tile CTAs' task tickets, one counter per KV head.
        # The plan kernel zeroes them.
        ticket = torch.empty(num_kv_heads, dtype=torch.int32, device=dev)
        # The kernel's ordering is one pass over the batch per row; above this
        # it is slower than the torch sort.
        if pseudo * num_seqs <= 1 << 19:
            build_query_block_map(cu_seqlens_q, seqused_k, num_tokens, query_map, context_lengths, stream, live)
        elif live:
            _build_query_map(cu_seqlens_q, seqused_k, num_tokens, query_map, context_lengths)
        plan_pa_decode(context_lengths, work_info, reduce_info, max_partitions, stream, live, tail_lo, ticket)
        null = flyc.from_c_void_p(fx.Float32, 0)
        if direct:
            # FP32 partials: bf16 ones rounded twice exceed the error budget.
            shape = (num_kv_heads, capacity, rows)
            pmax = torch.empty(shape, dtype=torch.float32, device=dev)
            psum = torch.empty(shape, dtype=torch.float32, device=dev)
            # Only slots at or past tail_lo can belong to a split row, so O partials are allocated from there:
            # the pointer is shifted back tail_lo slots (other slots' stores have empty descriptors).
            slot_elems = rows * _VALUE_DIM
            pout = torch.empty(((num_kv_heads * capacity - tail_lo) * slot_elems,), dtype=torch.float32, device=dev)
            pout_addr = pout.data_ptr() - tail_lo * slot_elems * 4 if live else 0
            pout_strides = (capacity * slot_elems, slot_elems, _VALUE_DIM)
            partials = (ptr_arg(pmax, fx.Float32), ptr_arg(psum, fx.Float32), flyc.from_c_void_p(fx.Float32, pout_addr))
        else:
            shape = (num_kv_heads, capacity, rows)
            pmax = torch.empty(shape, dtype=torch.float32, device=dev)
            psum = torch.empty(shape, dtype=torch.float32, device=dev)
            pout = torch.empty((*shape, _VALUE_DIM), dtype=torch.bfloat16, device=dev)
            partials = (ptr_arg(pmax, fx.Float32), ptr_arg(psum, fx.Float32), ptr_arg(pout, fx.BFloat16))
        _run_compiled(
            compile_pa_decode_pingpong(k.shape[1], float(softmax_scale), direct),
            *partials, ptr_arg(q, fx.BFloat16), ptr_arg(k, fx.BFloat16), ptr_arg(v, fx.BFloat16),
            ptr_arg(block_table, fx.Int32), ptr_arg(out, fx.BFloat16) if direct else null,
            int(block_table.shape[1]), num_kv_heads, int(q.stride(0)), int(q.stride(1)),
            ptr_arg(work_info, fx.Int32), capacity, ptr_arg(query_map, fx.Int32), ptr_arg(reduce_info, fx.Int32),
            ptr_arg(ticket, fx.Int32),
            min(capacity, max(1, num_cu // num_kv_heads)),
            stream,
            live=live,
        )
        if direct:
            # Merge the split tail rows only; the reducer skips rows with one task.
            output = out.as_strided(
                (pseudo - tail_lo, 1, num_kv_heads, _GQA, _VALUE_DIM),
                (0, out.stride(0), _GQA * out.stride(1), out.stride(1), 1),
            )
            launch_pa_decode_ps_reduce(
                output, psum, pmax, (pout_addr, pout_strides), TAIL_SPLIT, reduce_info[tail_lo:],
                query_map[tail_lo:], stream, live, num_cu, hybrid=True,
            )
            return out
        # Pseudo-sequence p writes rows query_map[p, 0] + [0, valid) of out; the
        # view supplies the batch and head geometry, and fits short outputs.
        output = out.as_strided(
            (pseudo, 1, num_kv_heads, _GQA, _VALUE_DIM),
            (0, out.stride(0), _GQA * out.stride(1), out.stride(1), 1),
        )
        max_partitions = max(1, min(max_partitions, -(-int(max_seqlen_k) // 256)))
        launch_pa_decode_ps_reduce(
            output,
            psum,
            pmax,
            pout,
            max_partitions,
            reduce_info,
            query_map,
            stream,
            live,
            num_cu,
        )
    return out


def workspace_requirements(num_tokens: int, num_seqs: int, num_cu: int):
    """Return managed-buffer requirements for an eligible direct plan.

    ``None`` means the base planner would partition rows and therefore is not
    the reviewed direct-output specialization.  The returned metadata is derived
    only from host-visible shapes, so the router can reserve it before launch.
    """
    pseudo = _pseudo_capacity(num_tokens, num_seqs)
    if _plan_capacity(pseudo, 1, num_cu) != pseudo:
        return None
    tail = min(TAIL_ROWS, pseudo)
    tail_lo = max(0, pseudo - num_seqs - tail)
    capacity = pseudo + tail * (TAIL_SPLIT - 1)
    rows = _QUERY_BLOCK * _GQA
    slot_elems = rows * _VALUE_DIM
    need = {
        "query_map": (pseudo * 4, torch.int32),
        "context_lengths": (pseudo, torch.int32),
        "work_info": (capacity * 4, torch.int32),
        "reduce_info": (pseudo * 2, torch.int32),
        "ticket": (1, torch.int32),
        "tail_max_logits": (capacity * rows, torch.float32),
        "tail_exp_sums": (capacity * rows, torch.float32),
        # Slots before tail_lo write directly to ``out``.  The tile kernel is
        # passed a virtually shifted base so only split-tail FP32 stores land.
        "tail_output": ((capacity - tail_lo) * slot_elems, torch.float32),
    }
    return need, pseudo, capacity, tail_lo


def run(
    q,
    k,
    v,
    out,
    cu_seqlens_q,
    seqused_k,
    block_table,
    softmax_scale,
    workspace,
    num_cu,
):
    """Launch the specialization using caller-owned, graph-stable workspace."""
    spec = workspace_requirements(
        q.shape[0],
        seqused_k.numel(),
        num_cu,
    )
    if spec is None:
        raise RuntimeError("the optimized varlen kernel requires a direct-output plan")
    need, pseudo, capacity, tail_lo = spec
    for name, (size, dtype) in need.items():
        tensor = workspace[name]
        if tensor.dtype != dtype or tensor.numel() < size:
            raise RuntimeError(f"invalid optimized varlen workspace buffer: {name}")

    rows = _QUERY_BLOCK * _GQA
    slot_elems = rows * _VALUE_DIM
    query_map = workspace["query_map"][: pseudo * 4].view(pseudo, 4)
    contexts = workspace["context_lengths"][:pseudo]
    work_info = workspace["work_info"][: capacity * 4].view(capacity, 4)
    reduce_info = workspace["reduce_info"][: pseudo * 2].view(pseudo, 2)
    ticket = workspace["ticket"][:1]
    shape = (1, capacity, rows)
    pmax = workspace["tail_max_logits"][: capacity * rows].view(shape)
    psum = workspace["tail_exp_sums"][: capacity * rows].view(shape)
    pout = workspace["tail_output"][: (capacity - tail_lo) * slot_elems]

    stream = torch.cuda.current_stream(q.device)
    build_query_block_map(
        cu_seqlens_q,
        seqused_k,
        q.shape[0],
        query_map,
        contexts,
        stream,
        True,
    )
    # This launch also zeros ``ticket``.  It is recorded in graph capture, so
    # every replay starts the persistent queue from the same state.
    plan_pa_decode(
        contexts,
        work_info,
        reduce_info,
        TAIL_SPLIT,
        stream,
        True,
        tail_lo,
        ticket,
    )

    pout_addr = pout.data_ptr() - tail_lo * slot_elems * 4
    pout_strides = (capacity * slot_elems, slot_elems, _VALUE_DIM)
    _run_compiled(
        compile_pa_decode_pingpong(k.shape[1], float(softmax_scale), True),
        ptr_arg(pmax, fx.Float32),
        ptr_arg(psum, fx.Float32),
        flyc.from_c_void_p(fx.Float32, pout_addr),
        ptr_arg(q, fx.BFloat16),
        ptr_arg(k, fx.BFloat16),
        ptr_arg(v, fx.BFloat16),
        ptr_arg(block_table, fx.Int32),
        ptr_arg(out, fx.BFloat16),
        int(block_table.shape[1]),
        1,
        int(q.stride(0)),
        int(q.stride(1)),
        ptr_arg(work_info, fx.Int32),
        capacity,
        ptr_arg(query_map, fx.Int32),
        ptr_arg(reduce_info, fx.Int32),
        ptr_arg(ticket, fx.Int32),
        min(capacity, max(1, num_cu)),
        stream,
        live=True,
    )

    tail_output = out.as_strided(
        (pseudo - tail_lo, 1, 1, _GQA, _VALUE_DIM),
        (0, out.stride(0), _GQA * out.stride(1), out.stride(1), 1),
    )
    launch_pa_decode_ps_reduce(
        tail_output,
        psum,
        pmax,
        (pout_addr, pout_strides),
        TAIL_SPLIT,
        reduce_info[tail_lo:],
        query_map[tail_lo:],
        stream,
        True,
        num_cu,
        hybrid=True,
    )
    return out
