# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""gfx950 varlen attention for large MiMo prefill and mixed batches.

out[t, h] = softmax_j(scale * q[t, h] . k[j, h // 16]) @ v[j, h // 16], causal
bottom-right per sequence. bf16 q/k/v/out, Q/K head dim 192, V/out head dim 128,
16 query heads per KV head, a linear KV cache [blocks, page, kv_heads, dim] of
128-token pages. The router selects it for one KV head on a 256-CU device when
the batch has at least one query block per task slot, so every block runs as
exactly one task and writes ``out`` directly.

Each sequence's queries split into blocks of _QUERY_BLOCK (16): 16 queries x 16
heads = 256 rows of one KV head. A call launches, in order:

1. the query map (``pa_decode_plan.build_query_block_map``): the blocks, longest
   context first;
2. the plan (compile_plan): one task per block, except the TAIL_ROWS shortest,
   dispatched last, which split into up to TAIL_SPLIT tasks; it also zeroes the
   tile kernel's task ticket;
3. the tile kernel (compile_tile_kernel): persistent CTAs that take tasks in plan
   order through an atomic ticket. A task loads Q and the first KV tiles, then
   each 64-token tile i runs the online softmax of tile i (issuing a later tile's
   DMA between its exp2s), PV of tile i and QK of tile i + 1. Unsplit blocks store
   normalized bf16 rows to ``out``; split ones store FP32 partials;
4. the tail reducer (compile_tail_reduce), which merges the split blocks'
   partials into ``out``.
"""

from functools import cache

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.compiler.protocol import dsl_size_of
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr.typing import T

from .kernels.pa_decode_plan import build_query_block_map
from .kernels.tensor_shim import _run_compiled, buf_base_i64, ptr_arg, ptr_buf_tensor

__all__ = ["run", "workspace_requirements"]

# (16-row tiles per wave, waves, 16-token chunks per wave): 8 waves of 32 rows, each
# on whole 64-token tiles.
LAYOUT = (2, 8, 4)
# LDS ring depth: four 40 KiB D192/V128 stages fill gfx950's 160 KiB LDS.
STAGES = 4
# LDS reads kept this many MFMAs ahead in the PV + QK phase.
LDS_READ_LEAD = 8
# Each tile's K/V DMA is split over all 8 waves (3 K + 2 V loads each), issued
# after the exp2 of these softmax values (0 .. M * CH * 4 - 1).
DMA_AFTER_EXP = (4, 10, 16, 22, 28)

LDS_BYTES = 160 * 1024
WAVE = 64
LOG2E = 1.4426950408889634

_HEAD_DIM = 192
_VALUE_DIM = 128
_GQA = 16
_QUERY_BLOCK = 16
# Query blocks the planner's LDS holds.
_MAX_PSEUDO_SEQS = 40000
KV_COMPUTE_BLOCK = 256
# The TAIL_ROWS shortest live blocks, dispatched last, split into up to TAIL_SPLIT
# tasks each, so the last CTAs to finish run short tasks.
TAIL_ROWS = 128
TAIL_SPLIT = 4
# A tail block gets one task per TAIL_MIN_PIECE 256-token plan tiles (at least
# one): extra tasks on short blocks cost more prologue and epilogue than they
# balance.
TAIL_MIN_PIECE = 8
# Planner LDS for its scan and reduction.
_PLAN_SCAN_LDS_BYTES = 512

TILE = 64
# Lazy rescale on each lane's tile sum of P instead of a per-tile row max: the
# reference moves only when a lane's sum of P for a row passes 2**SUM_LOG2 (so
# every P <= 2**SUM_LOG2, which BF16 P and f32 l and O hold), and the max is
# computed only inside that branch.
SUM_LOG2 = 8
SUM_LIMIT = float(2**SUM_LOG2)
NREF_UNSET = float(2.0**126)


def rcp_f32(value):
    # afn 1 / x lowers to v_rcp_f32.
    with fx.arith.fastmath("afn"):
        return fx.Float32(1.0) / fx.Float32(value)


def shared_ptr(elem_ty):
    """LDS pointer type of elem_ty with its element alignment (a byte-base recast drops it)."""
    return fx.PointerType.get(
        elem_ty.ir_type, fx.AddressSpace.Shared, dsl_size_of(elem_ty)
    )


def _pin(value):
    # Tied empty asm materializes the value here; LLVM otherwise sinks the softmax
    # past the stagger barrier into the MFMA phase (6-12% slower). FlyDSL has no
    # stable optimization barrier.
    value = fx.Vector(value)
    words = value.bitcast(fx.Int32) if value.dtype.width < 32 else value
    tied = fx.Vector(
        llvm.inline_asm(
            words.ir_value().type, [words.ir_value()], "", "=v,0", has_side_effects=True
        )
    )
    return tied.bitcast(value.dtype) if value.dtype.width < 32 else tied


def uniform(v):
    """v through readfirstlane: wave-uniform to LLVM, so it and the address math on it
    stay in SGPRs. Without it the kernel spills 34-51 VGPRs; 0.5.0 has no stable
    readfirstlane."""
    return fx.Int32(fx.rocdl.readfirstlane(T.i32, fx.Int32(v)))


def uniform64(v):
    v = fx.Int64(v)
    lo = uniform(fx.Int32(v & fx.Int64(0xFFFFFFFF)))
    hi = uniform(fx.Int32(v >> 32))
    return (fx.Int64(hi) << 32) | (fx.Int64(lo) & fx.Int64(0xFFFFFFFF))


def permlane16_swap(x, y):
    """v_permlane16_swap of two i32 values: (x, y) with lanes 16-31 of x swapped for
    lanes 0-15 of y, and lanes 48-63 of x for lanes 32-47 of y. 0.5.0 has no stable
    form."""
    sw = fx.rocdl.permlane16_swap(
        ir.Type.parse("!llvm.struct<(i32, i32)>"),
        x.ir_value(),
        y.ir_value(),
        False,
        False,
    )
    return fx.Int32(llvm.extractvalue(T.i32, sw, [0])), fx.Int32(
        llvm.extractvalue(T.i32, sw, [1])
    )


def opaque(x):
    """x through an empty asm, so LLVM cannot fold a known constant into it. FlyDSL
    has no stable optimization barrier."""
    return fx.Int32(
        llvm.inline_asm(
            T.i32, [fx.Int32(x).ir_value()], "", "=v,0", has_side_effects=False
        )
    )


# --- tile kernel ----------------------------------------------------------------
#
# One CTA owns a task's query block (256 rows of one KV head) and streams 64-token
# K/V tiles into an LDS ring by async DMA. Wave w owns rows [32 w, 32 w + 32) as
# M = 2 16-row tiles and all 64 tokens of each tile. Waves 0-3 and 4-7 form two
# groups, one wave of each per SIMD, that run one phase apart (see stagger_step).
# Each wave keeps its own online softmax, and P stays in registers (the QK
# accumulator layout is PV's B operand). Query lengths that do not fill the block
# pad it; padded rows are computed but never stored.


@cache
def compile_tile_kernel(block_size, softmax_scale):
    """The tile kernel's launch for one page size and scale.

    The launch reads an int32 [blocks, 4] query map of (first query row,
    block-table row, valid queries, unused) per query block; queries at or past
    the valid count repeat the last one and are not stored.
    """
    D, DV, GQA, QLEN = _HEAD_DIM, _VALUE_DIM, _GQA, _QUERY_BLOCK
    M, R, CH = LAYOUT
    SLOT_TOKENS = 16 * CH
    S = TILE // SLOT_TOKENS
    NWAVES = R * S
    assert NWAVES == 8 and S == 1, "two groups of one wave per SIMD on whole tiles"
    ROWS = QLEN * GQA
    assert R * M * 16 >= ROWS, "layout must cover every query row"

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
    # The ring is split by operand: the four V stages contiguous in [0, 64 KiB)
    # (stage s at 16 KiB s), the four K stages in [64 KiB, 160 KiB) (stage s at
    # 64 KiB + 24 KiB s). Every V read is then one resident base per nb plus an
    # immediate below 2**16, and K stages {0, 1} / {2, 3} each share one base per
    # K half (see stage_bases).
    K_BASE, V_BASE, K_STRIDE, V_STRIDE = STAGES * V_STAGE, 0, K_STAGE, V_STAGE
    assert STAGES * V_STAGE <= 1 << 16, "every V read must be base + a 16-bit DS offset"
    # The XOR bases need stage strides and bases with no bits below 8 KiB (the
    # swizzle flips bits 4-7).
    assert all(x % 8192 == 0 for x in (K_BASE, V_BASE, K_STRIDE, V_STRIDE))
    scale_log2 = float(softmax_scale) * LOG2E

    @fx.struct
    class SharedStorage:
        buf: fx.Array[fx.Int32, total_bytes // 4, 16]

    kernel_name = (
        f"pa_decode_pingpong_persist_kvbf16_d{D}_v{DV}_bs{block_size}_gqa{GQA}"
        f"_mtp{QLEN}_m{M}r{R}c{CH}_s{STAGES}_direct"
    )
    # Online-softmax state: [m (M), l (M), O (M * NB)].
    STATE = 2 * M + M * NB

    @flyc.jit
    def _task(
        lds,
        pmax_ptr: fx.Pointer,
        psum_ptr: fx.Pointer,
        pout_ptr: fx.Pointer,
        query_ptr: fx.Pointer,
        key_cache_ptr: fx.Pointer,
        value_cache_ptr: fx.Pointer,
        block_tables_ptr: fx.Pointer,
        query_map_ptr: fx.Pointer,
        max_blocks_per_seq: fx.Int32,
        stride_q_row: fx.Int32,
        stride_q_head: fx.Int32,
        seq: fx.Int32,
        start: fx.Int32,
        end: fx.Int32,
        context: fx.Int32,
        slot: fx.Int32,
        work_capacity: fx.Int32,
        out_ptr: fx.Pointer,
        reduce_ptr: fx.Pointer,
        n_kv,
    ):
        """One task: query block seq against KV tiles [start, end) * TILES_PER_PLAN."""
        kv_h = fx.Int32(fx.block_idx.y)
        tid = fx.Int32(fx.thread_idx.x)
        warp = uniform(tid // WAVE)
        lane = tid % WAVE
        g = lane // 16
        l16 = lane % 16
        # The wave's token slot and row block. ts is always 0 and rb the wave, but
        # spelling them out keeps LLVM's register allocation of the fast schedule.
        ts = warp // R
        rb = warp % R
        grp = warp // (NWAVES // 2)  # waves 0-3 and 4-7 run one phase apart
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
            """Q as QK's B operand: lane (g, l16) holds head l16, dims 32 kk + 8 g."""
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
                    fx.copy(
                        q_atom, fx.slice(q_tiled, (None, row_elem + kk * 32)), q_reg
                    )
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

        # Each wave's DMA share of a tile: K8 K and V8 V loads of 16 bytes a lane,
        # TILE_LOADS in all (its vmcnt).
        K8 = TILE * K_CPR // WAVE // NWAVES
        V8 = TILE * V_CPR // WAVE // NWAVES
        assert K8 * WAVE * NWAVES == TILE * K_CPR and V8 * WAVE * NWAVES == TILE * V_CPR
        TILE_LOADS = K8 + V8

        # A stage holds a token's K (V) row as 16-byte chunks, chunk c at
        # c ^ swizzle(token): the XOR spreads the read lanes over the LDS banks.
        # Rows of 24 chunks are not a power of two, so no SwizzleType.
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
        # Per task, for the loop's DMA: the head's row inside a page folded into the
        # K/V bases, each load's LDS offset within a stage (so its m0 is one add),
        # and the DMA bound. A lane's offset is row * n_kv * ROW + chunk, so the
        # bound is rows * n_kv * ROW with rows = clamp(lim - tok0, 0, TILE);
        # lim = min(context, t_end * TILE) also empties tiles past t_end.
        nkr = uniform(n_kv * K_ROW)
        nvr = uniform(n_kv * V_ROW)
        kb = uniform64(k_base + zext(kv_h * K_ROW))
        vb = uniform64(v_base + zext(kv_h * V_ROW))
        lds8_k = [
            uniform(fx.Int32(K_BASE + (warp * K8 + j) * (WAVE * 16)))
            for j in range_constexpr(K8)
        ]
        lds8_v = [
            uniform(fx.Int32(V_BASE + (warp * V8 + j) * (WAVE * 16)))
            for j in range_constexpr(V8)
        ]
        lim = uniform((context < t_end * TILE).select(context, t_end * TILE))
        step_k = uniform(nkr * TILE)
        step_v = uniform(nvr * TILE)

        def raw_rsrc(addr, nrec):
            """V# from its four words, without make_buffer_rsrc's stride mask on the
            high address word (a device address is below 2**48, so bits 48-63, the
            stride, are already 0)."""
            addr = fx.Int64(addr)
            words = [
                fx.Int32(addr & fx.Int64(0xFFFFFFFF)),
                fx.Int32(addr >> 32),
                fx.Int32(nrec),
                fx.Int32((7 << 12) | (4 << 15)),
            ]
            vec = fx.Vector.from_elements(words, dtype=fx.Int32)
            i128 = llvm.bitcast(ir.IntegerType.get_signless(128), vec.ir_value())
            return llvm.inttoptr(ir.Type.parse("!llvm.ptr<8>"), i128)

        def load_phys(tile):
            """The block table's page of tile, clamped to the last tile so the read
            stays in the sequence's row."""
            tile_c = (tile < t_end).select(tile, t_end - 1)
            # Wave-uniform, so an s_buffer_load into an SGPR.
            return fx.rocdl.s_buffer_load(
                fx.get_iter(bt), table_seq * max_blocks_per_seq + (tile_c >> page_shift)
            )

        def load_kv_tile(rs, lds_off, voff):
            """One LDS DMA of this wave's share of a K or V tile: 16 bytes a lane from
            voffset voff of rs to LDS lds_off. cdna4.BufferLoadAsyncLDS128b takes a
            tensor view, which would rebuild the V# the caller hoists per task."""
            dst = fx.to_llvm_ptr(
                fx.recast_iter(
                    shared_ptr(fx.Int32), fx.add_offset(lds, fx.make_int_tuple(lds_off))
                )
            )
            fx.rocdl.raw_ptr_buffer_load_async_lds(
                rs,
                dst,
                fx.Int32(16).ir_value(),
                fx.Int32(voff).ir_value(),
                fx.Int32(0).ir_value(),
                fx.Int32(0).ir_value(),
                aux=ir.IntegerAttr.get(ir.IntegerType.get_signless(32), 0),
            )

        def kv_tile_loads_lean(tile, stage, phys):
            """This wave's share of tile's K/V DMA into stage from page phys (read a
            step ahead), one thunk per load."""
            # The tile's index among all page tiles: the in-page tile offset rides
            # the page multiply.
            ptile = phys if TPP == 1 else phys * TPP + (tile & (TPP - 1))
            loads = []
            rows = lim - tile * TILE
            rows = (rows < TILE).select(rows, TILE)

            def operand_loads(base, nkvrow, voffs, copies, step_b, lds8, stage_off):
                # Clamped at 0 after the multiply: a min/max pair on rows becomes a
                # VALU v_med3.
                nrec = rows * nkvrow
                nrec = uniform((nrec > 0).select(nrec, 0))
                rs = raw_rsrc(uniform64(base + zext(ptile) * zext(step_b)), nrec)
                for j in range_constexpr(copies):
                    loads.append(
                        lambda j=j: load_kv_tile(rs, stage_off + lds8[j], voffs[j])
                    )

            operand_loads(kb, nkr, k_voff, K8, step_k, lds8_k, stage * K_STRIDE)
            operand_loads(vb, nvr, v_voff, V8, step_v, lds8_v, stage * V_STRIDE)
            return loads

        def kv_tile_loads(tile, stage, ok):
            """kv_tile_loads_lean reading its own page: past t_end (ok false) the
            table index is clamped and the loads read nothing."""
            phys = load_phys(tile)
            tok0 = tile * TILE
            row_idx = (tok0 & (block_size - 1)) * n_kv + kv_h
            left = context - tok0
            rows_m1 = ((left < TILE).select(left, TILE) - 1) * n_kv
            loads = []

            def operand_loads(base, page_bytes, row, voffs, copies, lds_off):
                addr = uniform64(
                    base + zext(phys) * zext(page_bytes) + zext(row_idx * row)
                )
                nrec = uniform(ok.select(rows_m1 * row + row, 0))
                rs = fx.rocdl.get_buffer_rsrc(
                    fx.get_iter(ptr_buf_tensor(addr, num_records_bytes=fx.Int64(nrec)))
                )
                for j in range_constexpr(copies):
                    loads.append(
                        lambda j=j: load_kv_tile(
                            rs, lds_off + (warp * copies + j) * (WAVE * 16), voffs[j]
                        )
                    )

            operand_loads(
                k_base, k_page_bytes, K_ROW, k_voff, K8, K_BASE + stage * K_STRIDE
            )
            operand_loads(
                v_base, v_page_bytes, V_ROW, v_voff, V8, V_BASE + stage * V_STRIDE
            )
            return loads

        for s in range_constexpr(STAGES - 1):
            # Always issued (empty past t_end), so the prologue's vmcnt counts hold.
            # Group 0 issues tile 2 itself in its first step.
            if const_expr(s == STAGES - 2):
                if grp == 1:
                    for load in kv_tile_loads(
                        fx.Int32(t_begin + s), s, fx.Int32(t_begin + s) < t_end
                    ):
                        load()
            else:
                for load in kv_tile_loads(
                    fx.Int32(t_begin + s), s, fx.Int32(t_begin + s) < t_end
                ):
                    load()

        # Loop-invariant LDS read offsets within a stage.
        kq = l16 // 4
        qq = l16 % 4
        # The swizzles see neither the chunk nor bits above 8 chunks of a K row,
        # so those become instruction immediates over 2 K and NB V bases.
        swz = k_swizzle(l16)
        k_tok = ts * SLOT_TOKENS + l16
        k_lds = [
            K_BASE + k_tok * K_ROW + ((par * 4 + g) ^ swz) * 16
            for par in range_constexpr(2)
        ]
        k_imm = [
            [c * 16 * K_ROW + (kk // 2) * 128 for kk in range_constexpr(KK)]
            for c in range_constexpr(CH)
        ]
        vswz = v_swizzle(g * 4 + kq)
        v_tok = ts * SLOT_TOKENS + g * 4 + kq
        v_lds = [
            v_tok * V_ROW + ((nb * 2 + qq // 2) ^ vswz) * 16 + (qq % 2) * 8
            for nb in range_constexpr(NB)
        ]
        # V_BASE rides the immediates (a base plus a constant would cost LLVM a
        # third add per stage in the XOR bases).
        v_imm = [V_BASE + c * 16 * V_ROW for c in range_constexpr(CH)]
        # K stages {0, 1} read from k_lds, {2, 3} from k_lds_hi.
        k_lds_hi = [b + 2 * K_STRIDE for b in k_lds]
        # The K bases sit at or above 64 KiB: a known constant there makes LLVM fold
        # base + offset into values too wide for the 16-bit DS field, one VGPR each
        # (+4 to +14 VGPRs, 0.5% slower). The asm hides the constant once per task;
        # the mask restores the 16-byte alignment it hides (without it K reads split
        # into ds_read2_b64, an address add each).
        k_lds_hi = [opaque(b) & fx.Int32(-16) for b in k_lds_hi]
        k_lds = [opaque(b) & fx.Int32(-16) for b in k_lds]

        # Raw-score state [m (M), l (M), O (M * NB)]: m is a row-uniform reference,
        # held negated and scaled (-m_raw * scale_log2, the exp2 addend); NREF_UNSET
        # marks a row with no reference yet: every finite score then exponentiates
        # to inf, which the tile-sum test catches. l is per lane.
        init = (
            [fx.Float32(NREF_UNSET)] * M
            + [zero] * M
            + [fx.full(4, 0.0, fx.Float32) for _ in range_constexpr(M * NB)]
        )
        limit = [context - valid + queries[m] for m in range_constexpr(M)]

        def rescale(st, lmax):
            """Move each row's reference m to its max over the row's lanes; scale l
            and O by alpha = exp2(old - new)."""
            out_m = []
            out_l = []
            out_o = []
            for m in range_constexpr(M):
                rmax = lmax[m]
                for sh in (16, 32):
                    rmax = fx.maxnumf(
                        rmax, fx.Float32(fx.gpu.shuffle_xor(rmax, sh, WAVE))
                    )
                ms_old = -st[m]
                ms_new = fx.maxnumf(ms_old, rmax * fx.Float32(scale_log2))
                alpha = fx.Float32(fx.exp2(ms_old - ms_new, fastmath="fast"))
                m_new = -ms_new
                out_m.append(m_new)
                out_l.append(st[M + m] * alpha)
                alpha_b = fx.full(4, alpha, fx.Float32)
                out_o.extend(
                    st[2 * M + m * NB + nb] * alpha_b for nb in range_constexpr(NB)
                )
            return out_m + out_l + out_o

        def stage_bases(stage):
            """The K (2) and V (NB) LDS read bases of a stage."""
            if const_expr(isinstance(stage, int)):
                # A compile-time stage: resident bases, the stage rides the DS
                # immediates. V: one base per nb serves every stage; K: one base per
                # (par, stage // 2), 8 V + 4 K loop-invariant VGPRs.
                half = stage // 2
                ko = (stage - 2 * half) * K_STRIDE
                return [b + ko for b in (k_lds_hi if half else k_lds)], [
                    b + stage * V_STRIDE for b in v_lds
                ]
            # (b + off) ^ c == (b ^ c) + off: off has no bits below 8 KiB, c only
            # bits 5-7 (V) or 6 (K).
            kb0 = stage * K_STRIDE + k_lds[0]
            vb0 = stage * V_STRIDE + v_lds[0]
            return [kb0, kb0 ^ 64], [
                vb0 if nb == 0 else vb0 ^ (32 * nb) for nb in range_constexpr(NB)
            ]

        mma_atom = fx.make_mma_atom(
            fx.rocdl.MFMA(16, 16, 32, fx.BFloat16)
        )  # v_mfma_f32_16x16x32_bf16

        def frag(value, dtype):
            """A register fragment holding value, as fx.gemm takes its operands; the
            copy folds away."""
            t = fx.make_rmem_tensor(len(value), dtype)
            t.store(fx.Vector(value))
            return t

        def qk_scores(stage):
            """S = K Q^T for a stage: lane (g, l16) of chunk c holds token
            16c + 4g + i, row l16."""
            k_stage, _ = stage_bases(stage)
            s_acc = [[None] * CH for _ in range_constexpr(M)]

            def read_k(c):
                return [
                    fx.ptr_load(
                        fx.recast_iter(
                            shared_ptr(fx.Int64),
                            fx.add_offset(
                                lds, fx.make_int_tuple(k_stage[kk % 2] + k_imm[c][kk])
                            ),
                        ),
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
                        fx.gemm(
                            mma_atom,
                            s_frag,
                            frag(k_ops[kk], fx.BFloat16),
                            frag(q_ops[m][kk], fx.BFloat16),
                            s_frag,
                        )
                        s_acc[m][c] = fx.Vector(s_frag.load())
                k_ops = k_next
            return [
                s_acc[m][c] for m in range_constexpr(M) for c in range_constexpr(CH)
            ]

        def online_softmax(it, s_flat, state, masked, loads):
            """Online softmax of one tile's scores: (state with O rescaled, P).
            loads: the DMA thunks to issue after the exp2 of each score index."""
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
                        e = fx.Float32(
                            fx.exp2(
                                fx.fma(v, fx.Float32(scale_log2), nrefs[m]),
                                fastmath="fast",
                            )
                        )
                        p.append(e)
                        if const_expr(hk is not None):
                            for load in hk.get(m * len(scores[m]) + len(p) - 1, []):
                                load()
                        if const_expr(len(p) % 8 == 0):
                            p_ops.append(
                                fx.Vector.from_elements(
                                    p[len(p) - 8 :], dtype=fx.Float32
                                ).to(fx.BFloat16)
                            )
                    # Four interleaved partial chains, then a tree.
                    parts = [p[k] for k in range_constexpr(4)]
                    for k in range_constexpr(4, len(p)):
                        parts[k % 4] = parts[k % 4] + p[k]
                    tsum.append((parts[0] + parts[1]) + (parts[2] + parts[3]))
                return p_ops, tsum

            st = list(state)
            p_ops, tsum = exp_scores(st[:M], loads)
            # Each P <= its lane's tile sum: rescale only when a sum passes
            # 2**SUM_LOG2 (inf when a score overflows exp2 or the row has no
            # reference yet), so the row max is computed only then.
            over = tsum[0] > fx.Float32(SUM_LIMIT)
            for m in range_constexpr(1, M):
                over = over | (tsum[m] > fx.Float32(SUM_LIMIT))
            # One SALU ballot; FlyDSL has no stable one, and a warp_reduce OR would be
            # six cross-lane shuffles every tile.
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

        v_tr_atom = fx.make_copy_atom(
            fx.rocdl.cdna4.LDSReadTrans(16, 64), fx.BFloat16
        )  # ds_read_b64_tr_b16
        v_tr_layout = fx.make_layout(4, 1)

        def read_v(stage, nb):
            """V block nb of every chunk, transposed: lane j holds dim j of tokens
            4g + k, k = 0..3."""
            _, v_stage = stage_bases(stage)
            parts = []
            for c in range_constexpr(CH):
                v_reg = fx.make_rmem_tensor(v_tr_layout, fx.BFloat16)
                v_ptr = fx.recast_iter(
                    shared_ptr(fx.BFloat16),
                    fx.add_offset(lds, fx.make_int_tuple(v_stage[nb] + v_imm[c])),
                )
                fx.copy(v_tr_atom, fx.make_view(v_ptr, v_tr_layout), v_reg)
                parts.append(fx.Vector(v_reg.load()))
            return parts

        def pv_accumulate(stage, p_flat, st):
            """O += P V for a stage."""
            pairs = CH // 2
            o_acc = [
                [st[2 * M + m * NB + nb] for nb in range_constexpr(NB)]
                for m in range_constexpr(M)
            ]
            v_cur = read_v(stage, 0)
            for nb in range_constexpr(NB):
                parts = v_cur
                v_cur = read_v(stage, nb + 1) if nb + 1 < NB else None
                for m in range_constexpr(M):
                    for j in range_constexpr(pairs):
                        v_op = fx.Vector.from_elements(
                            [
                                parts[2 * j + c][i]
                                for c in range_constexpr(2)
                                for i in range_constexpr(4)
                            ],
                            dtype=fx.BFloat16,
                        )
                        o_frag = frag(o_acc[m][nb], fx.Float32)
                        fx.gemm(
                            mma_atom,
                            o_frag,
                            frag(v_op, fx.BFloat16),
                            frag(p_flat[m * pairs + j], fx.BFloat16),
                            o_frag,
                        )
                        o_acc[m][nb] = fx.Vector(o_frag.load())
            return st[: 2 * M] + [
                o_acc[m][nb] for m in range_constexpr(M) for nb in range_constexpr(NB)
            ]

        def schedule_pv_qk():
            """One MFMA wave a SIMD: keep the PV and QK LDS reads LDS_READ_LEAD MFMAs ahead."""
            n_ds = (NB + KK) * CH
            n_mfma = M * NB * (CH // 2) + M * KK * CH
            fx.rocdl.sched_dsrd(LDS_READ_LEAD)
            for _ in range_constexpr(n_ds - LDS_READ_LEAD):
                fx.rocdl.sched_mfma(1)
                fx.rocdl.sched_dsrd(1)
            fx.rocdl.sched_mfma(n_mfma - (n_ds - LDS_READ_LEAD))

        def smod(x):
            return x & (STAGES - 1)

        def stagger_step(it, carry, masked, srel=None):
            """One tile of the pipeline: online_softmax of tile it (issuing a later
            tile's DMA), then pv_accumulate of tile it and qk_scores of tile it + 1.

            Group 1 runs one phase behind group 0: each phase X (softmax of tile i)
            pairs with the other group's phase Y (PV of tile i, QK of tile i + 1).
            Tile i + LEAD reuses tile i - 1's stage once group 1's PV of it ends; a
            group waits for its tile two slots before the first QK, when that tile is
            its only one in flight. srel: rel mod STAGES known at trace time, so the
            read stages are Python ints (the unrolled loop)."""
            rel = it - t_begin
            state = carry[:STATE]
            s_flat = carry[STATE : STATE + M * CH]

            # Group 0's softmax runs beside group 1's PV of tile it - 1, so it fills
            # tile it - 2's stage.
            xt = it + (LEAD - 1) + grp
            loads = kv_tile_loads_lean(
                xt, smod(rel + (LEAD - 1) + grp), fx.Int32(carry[STATE + M * CH])
            )
            # Next step's table entry, read now and carried: no SMEM wait in either phase.
            phys_next = load_phys(xt + 1)
            at_score = {}
            for at, load in zip(DMA_AFTER_EXP, loads):
                at_score.setdefault(at, []).append(load)
            st, p_flat = online_softmax(it, s_flat, state, masked, at_score)
            p_flat = [_pin(p) for p in p_flat]
            sums = _pin(fx.Vector.from_elements(st[M : 2 * M], dtype=fx.Float32))
            st = st[:M] + [sums[m] for m in range_constexpr(M)] + st[2 * M :]
            # Group 0: its share of tile it + 1 has landed (tile it + 2's, just
            # issued, may be in flight). Group 1 needs only the end-of-step wait;
            # this one (its tile it + 2 share, issued a step ago) is free and saves
            # the branch. The carried table read landed long ago; clearing it here
            # keeps the PV + QK phase's LDS waits graded (a pending SMEM read forces
            # lgkmcnt(0) at the first one).
            fx.rocdl.s_waitcnt(vmcnt=TILE_LOADS, lgkmcnt=0)
            gpu.barrier()
            # The wave in its MFMA phase wins issue over the sibling's softmax.
            # 0.5.0 has no stable s_setprio.
            fx.rocdl.s_setprio(1)
            # The last tile's lookahead QK reads a stale stage; it is unused.
            state = pv_accumulate(
                smod(rel) if srel is None else srel % STAGES, p_flat, st
            )
            s_next = qk_scores(smod(rel + 1) if srel is None else (srel + 1) % STAGES)
            schedule_pv_qk()
            fx.rocdl.s_setprio(0)
            # Group 1: its share of tile it + 2 has landed (tile it + 3's, just
            # issued, may be in flight). Group 0 needs only the mid-step wait; this
            # one is free and saves the branch.
            fx.rocdl.s_waitcnt(vmcnt=TILE_LOADS)
            if const_expr(srel is not None):
                # Unrolled steps share a block: keep the next step's DMA prep (SALU)
                # out of this MFMA phase's tail. 0.5.0 has no stable sched_barrier.
                fx.rocdl.sched_barrier(0)
            gpu.barrier()
            if const_expr(srel is not None):
                fx.rocdl.sched_barrier(0)
            return list(state) + list(s_next) + [phys_next]

        def epilogue(final):
            """Normalize O by l and store it: unsplit blocks as bf16 rows of out,
            split ones as FP32 partials with m and l."""
            m_fin = [
                (final[m] < fx.Float32(NREF_UNSET)).select(-final[m], neg_inf)
                for m in range_constexpr(M)
            ]
            l_fin = []
            for m in range_constexpr(M):
                lr = final[M + m]
                for sh in (16, 32):
                    lr = lr + fx.Float32(fx.gpu.shuffle_xor(lr, sh, WAVE))
                l_fin.append(lr)
            o_fin = [
                [final[2 * M + m * NB + nb] for nb in range_constexpr(NB)]
                for m in range_constexpr(M)
            ]

            partial = kv_h * work_capacity + slot
            rinfo = fx.ptr_load(
                fx.add_offset(fx.recast_iter(fx.Int32, reduce_ptr), seq * 2),
                result_type=fx.Vector.make_type(2, fx.Int32),
            )
            split_row = uniform(fx.Int32(rinfo[1])) > 1

            # Per-partial resources: padded rows land past them and are dropped, and
            # the unused path's are empty.
            def part_bytes(n):
                return fx.Int64(split_row.select(fx.Int32(n), fx.Int32(0)))

            pout = ptr_buf_tensor(
                uniform64(buf_base_i64(pout_ptr) + fx.Int64(partial) * (ROWS * DV * 4)),
                fx.Float32,
                num_records_bytes=part_bytes(ROWS * DV * 4),
            )
            # The block's valid query rows of out; padded rows drop.
            row_bytes = n_kv * (GQA * DV * 2)
            dout = ptr_buf_tensor(
                uniform64(
                    buf_base_i64(out_ptr) + fx.Int64(first_row) * fx.Int64(row_bytes)
                ),
                fx.BFloat16,
                num_records_bytes=fx.Int64(
                    uniform(split_row.select(fx.Int32(0), valid * row_bytes))
                ),
            )
            dout = fx.logical_divide(dout, fx.make_layout(1, 1))
            pmax = ptr_buf_tensor(
                uniform64(buf_base_i64(pmax_ptr) + fx.Int64(partial) * (ROWS * 4)),
                fx.Float32,
                num_records_bytes=part_bytes(ROWS * 4),
            )
            psum = ptr_buf_tensor(
                uniform64(buf_base_i64(psum_ptr) + fx.Int64(partial) * (ROWS * 4)),
                fx.Float32,
                num_records_bytes=part_bytes(ROWS * 4),
            )
            pout = fx.logical_divide(pout, fx.make_layout(1, 1))
            st_f32x4 = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.Float32)
            st_bf16x8 = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.BFloat16)

            def store_output(fp32_part):
                for m in range_constexpr(M):
                    mm = m_fin[m]
                    safe = (mm > neg_inf).select(mm, zero)
                    w = fx.Float32(fx.exp2(mm - safe, fastmath="fast"))
                    lt = l_fin[m] * w
                    inv = (lt > zero).select(fx.Float32(rcp_f32(lt)), zero)
                    wb = fx.full(4, w * inv, fx.Float32)
                    row = queries[m] * GQA + l16
                    if const_expr(fp32_part):
                        for nb in range_constexpr(NB):
                            o32 = o_fin[m][nb] * wb
                            o_reg = fx.make_rmem_tensor(4, fx.Float32)
                            o_reg.store(o32)
                            fx.copy(
                                st_f32x4,
                                o_reg,
                                fx.slice(pout, (None, row * DV + nb * 16 + g * 4)),
                            )
                        pmax[row] = fx.Float32(mm * fx.Float32(1.0 / LOG2E))
                        psum[row] = fx.Float32(lt)
                    else:
                        # Row of query q, head h in out [tokens, KV heads * GQA, DV].
                        drow = (queries[m] * n_kv + kv_h) * GQA + l16
                        # Pack each 16-column block to bf16 (2 dwords a lane), then one
                        # v_permlane16_swap per dword pair of blocks (2k, 2k + 1) gives
                        # every lane 8 adjacent columns, so each dwordx4 store writes 64
                        # contiguous bytes of each of its 16 rows (the stores are
                        # write-request bound).
                        packs = [
                            (o_fin[m][nb] * wb).to(fx.BFloat16).bitcast(fx.Int32)
                            for nb in range_constexpr(NB)
                        ]
                        col = (g % 2) * 16 + (g // 2) * 8
                        for k in range_constexpr(NB // 2):
                            xs, ys = [], []
                            for j in range_constexpr(2):
                                x, y = permlane16_swap(
                                    packs[2 * k][j], packs[2 * k + 1][j]
                                )
                                xs.append(x)
                                ys.append(y)
                            o_reg = fx.make_rmem_tensor(8, fx.BFloat16)
                            o_reg.store(
                                fx.Vector.from_elements(
                                    xs + ys, dtype=fx.Int32
                                ).bitcast(fx.BFloat16)
                            )
                            fx.copy(
                                st_bf16x8,
                                o_reg,
                                fx.slice(dout, (None, drow * DV + 32 * k + col)),
                            )

            if split_row:
                store_output(True)
            else:
                store_output(False)

        def idle_wave_loads():
            """A wave whose queries are all padding issues only its DMA share of every
            tile, at the active waves' barriers."""
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
                for load in kv_tile_loads(
                    xt_i, smod(rel_i + (LEAD - 1) + grp), xt_i < t_end
                ):
                    load()
                fx.rocdl.s_waitcnt(vmcnt=TILE_LOADS)
                gpu.barrier()
                fx.rocdl.s_waitcnt(vmcnt=TILE_LOADS)
                gpu.barrier()
            if grp == 0:
                gpu.barrier()
            fx.rocdl.s_waitcnt(vmcnt=0)
            gpu.barrier()

        # Waves whose queries are all padding (decode blocks have one valid query, so
        # waves 1-7) skip QK, softmax, PV and the stores; they still issue their DMA
        # share of every tile and keep the active waves' barrier count and placement,
        # so the shared K/V ring stays correct.
        active = queries[0] < valid
        if active:
            # Prologue: tile 0 ready for group 0's QK, then group 1 falls one phase
            # behind. Tiles 0-1 (group 0) or 0-2 (group 1) in flight: tile 0 lands
            # first.
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
            carry.append(load_phys(fx.Int32(t_begin + (LEAD - 1) + grp)))
            # Loops on i32 bounds: a scalar compare (an index induction variable
            # lowers to a 64-bit VALU compare). The unmasked loop runs 4 tiles per
            # iteration on compile-time stages 0-3 (rel is a multiple of 4 at each
            # iteration start), then a dynamic remainder loop of < 4 tiles.
            t_four = t_begin + ((t_split - t_begin) & ~3)
            for tile_i in range(t_begin, t_four, 4):
                for s4 in range_constexpr(4):
                    carry = stagger_step(fx.Int32(tile_i) + s4, carry, False, s4)
            for tile_i in range(t_four, t_split, 1):
                carry = stagger_step(fx.Int32(tile_i), carry, False)
            for tile_i in range(t_split, t_end, 1):
                carry = stagger_step(fx.Int32(tile_i), carry, True)
            final = carry[:STATE]
            if grp == 0:
                gpu.barrier()
            fx.rocdl.s_waitcnt(vmcnt=0)
            gpu.barrier()
            epilogue(final)
        else:
            idle_wave_loads()

        # No closing barrier: the persistent loop's ticket barrier follows, and no
        # LDS is touched since the post-loop barrier.

    @flyc.kernel(name=kernel_name, known_block_size=(NWAVES * WAVE, 1, 1))
    def pa_decode_pingpong_kernel(
        pmax_ptr: fx.Pointer,
        psum_ptr: fx.Pointer,
        pout_ptr: fx.Pointer,
        query_ptr: fx.Pointer,
        key_cache_ptr: fx.Pointer,
        value_cache_ptr: fx.Pointer,
        block_tables_ptr: fx.Pointer,
        max_blocks_per_seq: fx.Int32,
        stride_q_row: fx.Int32,
        stride_q_head: fx.Int32,
        work_info_ptr: fx.Pointer,
        work_capacity: fx.Int32,
        query_map_ptr: fx.Pointer,
        out_ptr: fx.Pointer,
        reduce_ptr: fx.Pointer,
        ticket_ptr: fx.Pointer,
        num_kv_heads: fx.Int32,
        n_ctas: fx.Int32,
    ):
        # Persistent CTAs: CTA b runs task b, then takes tickets (an atomic counter
        # per KV head, zeroed by the plan) for tasks n_ctas, n_ctas + 1, ..., so
        # tasks still start in plan order (longest first) on whichever CTA frees up
        # first. The ticket reaches the other waves through the last LDS word of
        # stage 3, which the next task's DMA writes only after its prologue barriers.
        # The LDS ring's byte base: an ir.Value pointer, which scf control flow can
        # carry (SharedStorage cannot).
        lds = fx.recast_iter(
            fx.Uint8, fx.SharedAllocator().allocate(SharedStorage).peek().buf.ptr
        )
        lds_word = fx.recast_iter(
            shared_ptr(fx.Int32), fx.add_offset(lds, fx.make_int_tuple(total_bytes - 4))
        )
        work = fx.recast_iter(fx.Int32, work_info_ptr)
        tick = fx.add_offset(
            fx.recast_iter(fx.Int32, ticket_ptr), fx.Int32(fx.block_idx.y)
        )
        # Kernel arguments rather than grid_dim, which calls __ockl_get_num_groups (a
        # real call, here inside the task loop, that pins registers across it).
        n_kv = uniform(num_kv_heads)
        n_ctas = uniform(n_ctas)
        tid0 = fx.Int32(fx.thread_idx.x) == 0
        slot = uniform(fx.block_idx.x)
        while slot < work_capacity:
            task = fx.ptr_load(
                fx.add_offset(work, slot * 4),
                result_type=fx.Vector.make_type(4, fx.Int32),
            )
            start = fx.Int32(task[1])
            end = fx.Int32(task[2])
            if start < end:
                _task(
                    lds,
                    pmax_ptr,
                    psum_ptr,
                    pout_ptr,
                    query_ptr,
                    key_cache_ptr,
                    value_cache_ptr,
                    block_tables_ptr,
                    query_map_ptr,
                    max_blocks_per_seq,
                    stride_q_row,
                    stride_q_head,
                    fx.Int32(task[0]),
                    start,
                    end,
                    fx.Int32(task[3]),
                    slot,
                    work_capacity,
                    out_ptr,
                    reduce_ptr,
                    n_kv,
                )
            if tid0:
                t = fx.atomic_add(tick, fx.Int32(1), syncscope=fx.rocdl.SyncScope.Agent)
                fx.ptr_store(
                    fx.Vector.from_elements([fx.Int32(t)], dtype=fx.Int32), lds_word
                )
            gpu.barrier()
            nxt = fx.ptr_load(lds_word, result_type=fx.Vector.make_type(1, fx.Int32))
            slot = uniform(fx.Int32(nxt[0]) + n_ctas)
            # Every wave has read the ticket before tid 0 can write the next one.
            gpu.barrier()

    @flyc.jit
    def pa_decode_pingpong_launch(
        pmax: fx.Pointer,
        psum: fx.Pointer,
        pout: fx.Pointer,
        query: fx.Pointer,
        key_cache: fx.Pointer,
        value_cache: fx.Pointer,
        block_tables: fx.Pointer,
        out: fx.Pointer,
        max_blocks_per_seq: fx.Int32,
        num_kv_heads: fx.Int32,
        stride_q_row: fx.Int32,
        stride_q_head: fx.Int32,
        work_info: fx.Pointer,
        work_capacity: fx.Int32,
        query_map: fx.Pointer,
        reduce_info: fx.Pointer,
        ticket: fx.Pointer,
        n_ctas: fx.Int32,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        _ = kernel_name
        pa_decode_pingpong_kernel(
            pmax,
            psum,
            pout,
            query,
            key_cache,
            value_cache,
            block_tables,
            max_blocks_per_seq,
            stride_q_row,
            stride_q_head,
            work_info,
            work_capacity,
            query_map,
            out,
            reduce_info,
            ticket,
            num_kv_heads,
            n_ctas,
            # Packed FP32 softmax VALU stalls behind the sibling wave's MFMAs; with
            # it, 40-47 more SGPR spills and 13x the output error. 0.5.0's
            # target-features compile hint has no effect, so the kernel attribute
            # sets it.
            value_attrs={
                "llvm.target_features": ir.Attribute.parse(
                    '#llvm.target_features<["-packed-fp32-ops"]>'
                )
            },
        ).launch(
            grid=(n_ctas, num_kv_heads, 1), block=(NWAVES * WAVE, 1, 1), stream=stream
        )

    pa_decode_pingpong_launch.compile_hints.update(waves_per_eu=2, fastmath="contract")
    # iterative-ilp re-schedules for latency.
    pa_decode_pingpong_launch.compile_hints["llvm_options"] = {
        "amdgpu-sched-strategy": "iterative-ilp"
    }
    return pa_decode_pingpong_launch


# --- plan -----------------------------------------------------------------------


@cache
def compile_plan(max_batch, threads):
    """One task per live block, except the last TAIL_ROWS live blocks at or past row
    ``tail_floor``, which get up to ``max_parts`` tasks each.

    Batch, task capacity and partition cap are runtime values. Blocks arrive sorted
    longest first, so live blocks are a prefix. Per CTA: count_tiles, a reduction,
    split_tasks, a scan, record_rows, write_work.
    """
    scan_counts = fx.coop.BlockScan[fx.Int32, threads]
    reduce_rows = fx.coop.BlockReduce[fx.Int32, threads]

    @fx.struct
    class TaskStorage:
        ends: fx.Array[fx.Int32, max_batch, 16]

    def visible_tiles(ctx):
        """256-token KV tiles of a context."""
        return (fx.max(fx.Int64(ctx), fx.Int64(0)) + 255) // 256

    # Lookup reaches rows below max_batch, so it needs one fewer level than
    # max_batch.bit_length() at power-of-two buckets.
    lookup_bits = (max_batch - 1).bit_length()

    @flyc.kernel(
        name=f"pa_decode_plan_tail_kernel_b{max_batch}_t{threads}",
        known_block_size=[threads, 1, 1],
    )
    def pa_decode_plan_kernel(
        lengths_ptr: fx.Pointer,
        work_ptr: fx.Pointer,
        reduce_ptr: fx.Pointer,
        batch: fx.Int32,
        capacity: fx.Int32,
        max_parts: fx.Int32,
        tail_floor: fx.Int32,
        ticket_ptr: fx.Pointer,
        num_tickets: fx.Int32,
    ):
        tid = fx.thread_idx.x
        lengths = ptr_buf_tensor(lengths_ptr)
        work = ptr_buf_tensor(work_ptr)
        reduce_info = ptr_buf_tensor(reduce_ptr)
        # Zero the tile kernel's ticket counters (it runs after this kernel on the
        # same stream), so the host need not launch a memset.
        tickets = ptr_buf_tensor(ticket_ptr)
        for t in range(tid, num_tickets, threads):
            tickets[t] = fx.Int32(0)
        smem = fx.SharedAllocator()
        count_storage = smem.allocate(scan_counts.SharedStorage).peek()
        row_storage = smem.allocate(reduce_rows.SharedStorage).peek()
        task_ends = smem.allocate(TaskStorage).peek().ends
        # Each thread owns consecutive rows, so scans of per-thread totals preserve
        # sequence order. Buckets that fit the block give each thread at most one
        # row; larger ones walk rows with runtime bounds, so padding rows cost
        # nothing and register use does not grow with the bucket.
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

        def split(tiles, row):
            """A row's task count: tail rows get one task per TAIL_MIN_PIECE tiles."""
            return (tiles > 0).select(
                (row >= tail_lo).select(
                    fx.Int32(
                        fx.max(
                            fx.min(tiles // TAIL_MIN_PIECE, fx.Int64(max_parts)),
                            fx.Int64(1),
                        )
                    ),
                    fx.Int32(1),
                ),
                fx.Int32(0),
            )

        def split_tasks(task_ends, thread_tiles):
            """The thread's task count, each row's in task_ends."""
            if fx.const_expr(single):
                thread_tasks = fx.Int32(0)
                if thread_tiles > 0:
                    thread_tasks = split(thread_tiles, row)
            else:
                thread_tasks = fx.Int32(0)
                for seq in range(row_begin, row_end):
                    count = split(visible_tiles(lengths[seq]), seq)
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
            """One work_info record (sequence, first tile, end tile, context) per slot;
            padding slots get zeros."""
            # Upper-bound search skips repeated ends from empty rows.
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
                    for field in fx.range_constexpr(4):
                        work[slot * 4 + field] = fx.Int32(0)

        if fx.const_expr(single):
            row = fx.Int32(tid)
        else:
            rows = (batch + threads - 1) // threads
            row_begin = fx.min(fx.Int32(tid) * rows, batch)
            row_end = fx.min(row_begin + rows, batch)
        thread_tiles, thread_rows = count_tiles()
        live_rows = reduce_rows(thread_rows, fx.ReductionOp.ADD, storage=row_storage)
        tail_lo = fx.max(live_rows - fx.Int32(TAIL_ROWS), tail_floor)
        thread_tasks = split_tasks(task_ends, thread_tiles)
        row_start, total_tasks = scan_counts.exclusive_with_aggregate(
            thread_tasks, fx.ReductionOp.ADD, storage=count_storage
        )
        # Captured tensors written under a runtime branch would become branch state:
        # pass them in.
        record_rows(reduce_info, task_ends, row_start, thread_tasks)
        fx.gpu.barrier()
        write_work(work, total_tasks)

    @flyc.jit
    def launch(
        lengths: fx.Pointer,
        work: fx.Pointer,
        reduce_info: fx.Pointer,
        batch: fx.Int32,
        capacity: fx.Int32,
        max_parts: fx.Int32,
        tail_floor: fx.Int32,
        ticket: fx.Pointer,
        num_tickets: fx.Int32,
        stream: fx.Stream,
    ):
        pa_decode_plan_kernel(
            lengths,
            work,
            reduce_info,
            batch,
            capacity,
            max_parts,
            tail_floor,
            ticket,
            num_tickets,
        ).launch(grid=(1, 1, 1), block=(threads, 1, 1), stream=stream)

    return launch


# --- tail reducer ---------------------------------------------------------------


@cache
def compile_tail_reduce():
    """Merge the FP32 partials of each split block into its bf16 rows of out.

    One 64-thread CTA per (block, KV head, row); lane p owns partition p's
    statistics, and each lane reduces two adjacent output columns. Blocks with one
    task were stored by the tile kernel and are skipped.
    """
    parts, elems = TAIL_SPLIT, 2
    threads = _VALUE_DIM // elems
    assert threads == WAVE and parts <= WAVE
    offsets = tuple(
        o for o in (32, 16, 8, 4, 2, 1) if o < 1 << (parts - 1).bit_length()
    )

    @flyc.kernel(
        name=f"pa_decode_reduce_tail_kernel_obf16_d{_VALUE_DIM}_np{parts}_gqa{_GQA}",
        known_block_size=[threads, 1, 1],
    )
    def tail_reduce_kernel(
        output_ptr: fx.Pointer,
        exp_sums_ptr: fx.Pointer,
        max_logits_ptr: fx.Pointer,
        logits_ptr: fx.Pointer,
        stride_output_len: fx.Int32,
        stride_output_kv_head: fx.Int32,
        stride_output_group_size: fx.Int32,
        stride_exp_sums_head: fx.Int32,
        stride_exp_sums_part: fx.Int32,
        stride_logits_head: fx.Int32,
        stride_logits_part: fx.Int32,
        stride_logits_group: fx.Int32,
        reduce_info_ptr: fx.Pointer,
        query_map_ptr: fx.Pointer,
    ):
        batch_idx = fx.block_idx.x
        kv_head_idx = fx.block_idx.y
        row_idx = fx.block_idx.z
        unit = fx.thread_idx.x
        exp_sums = fx.recast_iter(fx.Float32, exp_sums_ptr)
        max_logits = fx.recast_iter(fx.Float32, max_logits_ptr)
        reduce_info = fx.recast_iter(fx.Int32, reduce_info_ptr)

        zero_f = fx.Float32(0.0)
        neg_inf = fx.Float32(float("-inf"))
        lane = unit & fx.Int32(WAVE - 1)
        group_idx = row_idx & fx.Int32(_GQA - 1)
        query_idx = row_idx >> fx.Int32(_GQA.bit_length() - 1)
        qmap = fx.recast_iter(fx.Int32, query_map_ptr)
        first_row = fx.Int32(qmap[batch_idx * 4])
        valid_rows = fx.Int32(qmap[batch_idx * 4 + 2])
        row_elem = (
            (first_row + query_idx) * stride_output_len
            + kv_head_idx * stride_output_kv_head
            + group_idx * stride_output_group_size
        )
        first_part = fx.Int32(reduce_info[batch_idx * 2])
        part_count = fx.Int32(reduce_info[batch_idx * 2 + 1])
        # Padded query rows get an empty resource: their stores are dropped.
        store_row = (query_idx < valid_rows) & (part_count > fx.Int32(1))
        output = ptr_buf_tensor(
            buf_base_i64(output_ptr) + fx.Int64(row_elem) * 2,
            fx.BFloat16,
            unit_elems=elems,
            num_records_bytes=fx.Int64(
                store_row.select(fx.Int32(_VALUE_DIM * 2), fx.Int32(0))
            ),
        )
        # Rebase in i64 before narrowing: the partials can exceed 2 GiB. The bound
        # zero-fills partitions past the live count.
        logits = ptr_buf_tensor(
            buf_base_i64(logits_ptr)
            + (
                fx.Int64(first_part) * fx.Int64(stride_logits_part)
                + fx.Int64(kv_head_idx) * fx.Int64(stride_logits_head)
            )
            * 4,
            fx.Float32,
            unit_elems=elems,
            unit_stride=1,
            num_records_bytes=fx.Int64(part_count) * fx.Int64(stride_logits_part) * 4,
        )
        load_atom = fx.make_copy_atom(fx.rocdl.BufferCopy64b(0), fx.Float32)
        store_atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.BFloat16)

        if part_count > fx.Int32(1):
            live = lane < part_count
            offset = (
                first_part * stride_exp_sums_part
                + kv_head_idx * stride_exp_sums_head
                + live.select(lane, fx.Int32(0)) * stride_exp_sums_part
                + row_idx
            )
            part_sum = live.select(fx.Float32(exp_sums[offset]), zero_f)
            part_max = live.select(fx.Float32(max_logits[offset]), neg_inf)
            values = []
            for part in fx.range_constexpr(parts):
                index = (
                    fx.Int32(part) * stride_logits_part
                    + row_idx * stride_logits_group
                    + unit * fx.Int32(elems)
                )
                fragment = fx.make_rmem_tensor(elems, fx.Float32)
                fx.copy(load_atom, fx.slice(logits, (index, None)), fragment)
                values.append(fx.Vector(fragment.load()))
            lane_max = part_max
            for o in offsets:
                lane_max = fx.max(
                    lane_max, fx.Float32(fx.gpu.shuffle_xor(lane_max, o, WAVE))
                )
            safe_max = (lane_max > neg_inf).select(lane_max, zero_f)
            scale = (part_max > neg_inf).select(
                fx.exp2((part_max - safe_max) * fx.Float32(LOG2E), fastmath="fast"),
                zero_f,
            )
            weight = part_sum * scale
            exp_sum = weight
            for o in offsets:
                with fx.arith.fastmath("fast"):
                    exp_sum = exp_sum + fx.Float32(fx.gpu.shuffle_xor(exp_sum, o, WAVE))
            weight = weight * fx.Float32(
                rcp_f32((exp_sum > zero_f).select(exp_sum, fx.Float32(1.0)))
            )
            acc = fx.full(2, 0.0, fx.Float32)
            for part in fx.range_constexpr(parts):
                # Increasing-part order and separate multiply/add rounding.
                acc = acc + values[part] * fx.gpu.shuffle_idx(weight, part, WAVE)
            fragment = fx.make_rmem_tensor(elems, fx.BFloat16)
            fragment.store(acc.to(fx.BFloat16))
            fx.copy(store_atom, fragment, fx.slice(output, (unit, None)))

    @flyc.jit
    def launch(
        output: fx.Pointer,
        exp_sums: fx.Pointer,
        max_logits: fx.Pointer,
        logits: fx.Pointer,
        stride_output_len: fx.Int32,
        stride_output_kv_head: fx.Int32,
        stride_output_group_size: fx.Int32,
        stride_exp_sums_head: fx.Int32,
        stride_exp_sums_part: fx.Int32,
        stride_logits_head: fx.Int32,
        stride_logits_part: fx.Int32,
        stride_logits_group: fx.Int32,
        blocks: fx.Int32,
        num_kv_heads: fx.Int32,
        reduce_info: fx.Pointer,
        query_map: fx.Pointer,
        stream: fx.Stream,
    ):
        tail_reduce_kernel(
            output,
            exp_sums,
            max_logits,
            logits,
            stride_output_len,
            stride_output_kv_head,
            stride_output_group_size,
            stride_exp_sums_head,
            stride_exp_sums_part,
            stride_logits_head,
            stride_logits_part,
            stride_logits_group,
            reduce_info,
            query_map,
        ).launch(
            grid=(blocks, num_kv_heads, _QUERY_BLOCK * _GQA),
            block=(threads, 1, 1),
            stream=stream,
        )

    return launch


# --- host -----------------------------------------------------------------------


def _pseudo_capacity(num_tokens, num_seqs, block=_QUERY_BLOCK):
    # Sequence i has ceil(q_i / b) <= q_i / b + (b - 1) / b query blocks.
    return -(-num_tokens // block) + num_seqs


def _plan_capacity(num_seqs, num_kv_heads, num_cu):
    # The generic planner's default: a 2-per-CU task budget, at least one per sequence.
    return min(num_seqs * num_cu, max(num_seqs, -(-2 * num_cu // num_kv_heads)))


def workspace_requirements(num_tokens: int, num_seqs: int, num_cu: int):
    """Managed-buffer requirements of a call, from host-visible shapes only, so the
    router can reserve them before launch: ``(need, blocks, capacity, tail_lo)``.

    ``None`` means the generic planner would split blocks (fewer query blocks than
    task slots) or the blocks exceed the planner's LDS: not this specialization.
    """
    pseudo = _pseudo_capacity(num_tokens, num_seqs)
    if _plan_capacity(pseudo, 1, num_cu) != pseudo or pseudo > _MAX_PSEUDO_SEQS:
        return None
    # Live blocks are a prefix of at least pseudo - num_seqs rows, so the last
    # `tail` of them sit at or past tail_lo.
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
        # Only slots at or past tail_lo can belong to a split block; the tile kernel
        # takes a base shifted back tail_lo slots.
        "tail_output": ((capacity - tail_lo) * slot_elems, torch.float32),
    }
    return need, pseudo, capacity, tail_lo


def run(
    q, k, v, out, cu_seqlens_q, seqused_k, block_table, softmax_scale, workspace, num_cu
):
    """Launch the specialization on caller-owned, graph-stable workspace."""
    spec = workspace_requirements(q.shape[0], seqused_k.numel(), num_cu)
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
    pmax = workspace["tail_max_logits"][: capacity * rows]
    psum = workspace["tail_exp_sums"][: capacity * rows]
    pout = workspace["tail_output"][: (capacity - tail_lo) * slot_elems]
    pout_addr = pout.data_ptr() - tail_lo * slot_elems * 4

    stream = torch.cuda.current_stream(q.device)
    build_query_block_map(
        cu_seqlens_q,
        seqused_k,
        q.shape[0],
        _QUERY_BLOCK,
        query_map,
        contexts,
        longest_first=True,
    )
    assert (pseudo * 4 + 15) // 16 * 16 + _PLAN_SCAN_LDS_BYTES <= LDS_BYTES
    max_batch = min(
        (LDS_BYTES - _PLAN_SCAN_LDS_BYTES) // 16 * 4, 1 << (pseudo - 1).bit_length()
    )
    # Also zeroes ``ticket``; recorded in graph capture, so every replay starts the
    # persistent queue from the same state.
    _run_compiled(
        compile_plan(max_batch, max(64, min(1024, 1 << (capacity - 1).bit_length()))),
        ptr_arg(contexts, fx.Int32),
        ptr_arg(work_info, fx.Int32),
        ptr_arg(reduce_info, fx.Int32),
        pseudo,
        capacity,
        TAIL_SPLIT,
        tail_lo,
        ptr_arg(ticket, fx.Int32),
        1,
        stream,
    )
    _run_compiled(
        compile_tile_kernel(k.shape[1], float(softmax_scale)),
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
        min(capacity, num_cu),
        stream,
    )
    # Rows of the split blocks (at or past tail_lo) in out.
    blocks = pseudo - tail_lo
    _run_compiled(
        compile_tail_reduce(),
        ptr_arg(out, fx.BFloat16),
        ptr_arg(psum, fx.Float32),
        ptr_arg(pmax, fx.Float32),
        flyc.from_c_void_p(fx.Float32, pout_addr),
        out.stride(0),
        _GQA * out.stride(1),
        out.stride(1),
        capacity * rows,
        rows,
        capacity * slot_elems,
        slot_elems,
        _VALUE_DIM,
        blocks,
        1,
        ptr_arg(reduce_info[tail_lo:], fx.Int32),
        ptr_arg(query_map[tail_lo:], fx.Int32),
        stream,
    )
    return out
