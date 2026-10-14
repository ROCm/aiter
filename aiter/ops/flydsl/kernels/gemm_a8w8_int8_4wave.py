# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""4-wave true-INT8 matmul with row-wise scaling for AMD CDNA4 (gfx950 only).

One wave per SIMD, a 2x2 wave grid over a 256x256 block and a 128x128 tile
per wave, accumulated in AGPRs with ``v_mfma_i32_16x16x64_i8``. The K loop
follows the hipBLASLt FP8 schedule: two LDS buffers, three barriers per K128
tile, and LDS reads / LDS-DMA loads issued between MFMAs.

LDS holds both operands in 16-row blocks per K128 tile. A is stored
[row][k16 chunk] with an XOR swizzle on the chunk, so its global loads cover
whole 128-byte rows. B keeps its ``shuffle_weight(layout=(16, 16))`` block
layout [k16 chunk][row][16 B], so its global loads are contiguous, with the
blocks padded to 2064 bytes. B tile ``j`` of a wave takes rows ``8 * k + j``,
so each lane ends up with eight adjacent output columns and the epilogue
writes whole 256-byte rows, as hipBLASLt does.

gfx950 serves a ds_read_b128 in four phases of 16 lanes: lane groups
``l // 4`` in {0, 3, 5, 6} and {1, 2, 4, 7} of each 32-lane half. Both
layouts give each phase 16 distinct 16-byte bank groups.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr import const_expr, range_constexpr, rocdl
from flydsl.expr.typing import Vector as Vec

from aiter.ops.flydsl.kernels import buffer_ops
from aiter.ops.flydsl.kernels.gemm_a8w8_8wave import (
    S2RLoader,
    StoreC,
    _xcd_swizzle_any,
)
from aiter.ops.flydsl.kernels.kernels_common import ceildiv
from aiter.ops.flydsl.kernels.mfma_preshuffle_pipeline import split_row_major_2d
from aiter.ops.flydsl.kernels.mxfp4_gemm_common import _lds_ptr3

BLOCK_M = 256
BLOCK_N = 256
BLOCK_K = 128
N_WAVES = 4
WAVE_ROWS = 128  # rows (and columns) of C per wave
TILE = 16  # MFMA output tile
MFMA_K = 64
WAVE_TILES = WAVE_ROWS // TILE
K_STEPS = BLOCK_K // MFMA_K
K_GROUPS = 4  # lane groups along K within one operand
ROW_BLOCKS = BLOCK_M // 16  # 16-row blocks per region
A_BLOCK_STRIDE = 16 * BLOCK_K
B_BLOCK_STRIDE = A_BLOCK_STRIDE + 16  # shifts block b by b bank groups
A_REGION_BYTES = ROW_BLOCKS * A_BLOCK_STRIDE
B_REGION_BYTES = ROW_BLOCKS * B_BLOCK_STRIDE
LOADS_PER_REGION = BLOCK_M * BLOCK_K // (N_WAVES * 1024)


def _a_chunk_swizzle(row):
    """XOR key for row ``row`` (0-15) of an A row block.

    Rows ``r`` and ``r ^ 1`` share 64 banks. A phase of the A operand read
    covers all 16 rows, with k16 chunk ``c ^ 1`` for rows 4-11 and ``c`` for
    the others, so keys 0-3 go to rows 4-11 and 4-7 to the rest, distinct
    within each row parity.
    """
    quad = row // 4
    return ((1 ^ (quad % 2) ^ (quad // 2)) << 2) | ((quad // 2) << 1) | ((row % 4) // 2)


def _lane_offsets(lane_id, wave_id, K, preshuffled):
    """Per-lane global offsets for the LDS-DMA steps of one region.

    Step ``s`` of wave ``w`` fills half ``s % 2`` (1 KB) of row block
    ``4 * w + s // 2``: rows ``(s % 2) * 8`` onwards for A, k16 chunks
    ``(s % 2) * 4`` onwards for B. A stores row ``r`` as eight 16-byte chunks
    with chunk ``c`` at slot ``c ^ _a_chunk_swizzle(r)``, so every 8 lanes
    load one 128-byte row.
    """
    offsets = []
    for step in range_constexpr(LOADS_PER_REGION):
        row_block = wave_id * (LOADS_PER_REGION // 2) + step // 2
        if const_expr(preshuffled):
            offsets.append(row_block * (16 * K) + (step % 2) * 1024 + lane_id * 16)
        else:
            row = (step % 2) * 8 + lane_id // 8
            chunk = (lane_id % 8) ^ _a_chunk_swizzle(row)
            offsets.append((row_block * 16 + row) * K + chunk * 16)
    return offsets


class G2SDirect:
    """LDS-DMA loads of one region, four steps per M0 value.

    The instruction offset applies to both the global and the LDS address, so
    steps ``4q .. 4q + 3`` share M0 and differ only in that offset. Each M0
    change costs an ``s_mov``, a hazard ``s_nop`` and a load stall that a
    16-cycle MFMA slot cannot hide.
    """

    STEPS_PER_M0 = 4

    def __init__(self, rsrc, lane_offsets, wave_id, block_stride):
        self.rsrc = rsrc
        piece = [
            (s // 2) * block_stride + (s % 2) * 1024 for s in range(LOADS_PER_REGION)
        ]
        self.m0_off = [
            piece[s - s % self.STEPS_PER_M0] for s in range(LOADS_PER_REGION)
        ]
        self.imm = [piece[s] - self.m0_off[s] for s in range(LOADS_PER_REGION)]
        self.voffsets = [off - imm for off, imm in zip(lane_offsets, self.imm)]
        self.wave_off = wave_id * (LOADS_PER_REGION // 2) * block_stride

    def _load(self, lds_region, soffset, step):
        base = fx.Int32(fx.ptrtoint(lds_region.ptr)) + self.wave_off + self.m0_off[step]
        rocdl.raw_ptr_buffer_load_lds(
            self.rsrc,
            _lds_ptr3(base, fx.Int32(0)),
            fx.Int32(16),
            self.voffsets[step],
            fx.Int32(soffset),
            fx.Int32(self.imm[step]),
            fx.Int32(0),
        )

    def ops(self, lds_region, soffset, steps):
        return [
            lambda step=step: self._load(lds_region, soffset, step) for step in steps
        ]


class StoreCInt8Rows(StoreC):
    """Scale and store the wave's INT32 accumulators as BF16 rows.

    Lane ``l`` of MFMA tile (i, j) holds row ``16 i + 4 (l // 16) + r`` in
    register ``r`` and column ``8 (l % 16) + j``, so register ``r`` of the
    eight j tiles is one 16-byte store and 16 lanes cover a 256-byte row.
    Preshuffled B has N % 16 == 0: a store is fully inside C or past column N.
    """

    def __init__(self, *args):
        super().__init__(*args)
        self.out_atom_8 = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.BFloat16)
        self.reg_bf16_8 = fx.make_rmem_tensor(fx.make_layout(8, 1), fx.BFloat16)

    def _load_b_scale_vec4(self, col):
        fx.copy(
            self.scale_atom_4,
            fx.slice(self.sb_div, (None, fx.Int32(col))),
            self.reg_f32_4,
        )
        return Vec(fx.memref_load_vec(self.reg_f32_4))

    def _store_bf16x8(self, values, c_index):
        fx.memref_store_vec(values, self.reg_bf16_8)
        fx.copy(
            self.out_atom_8,
            self.reg_bf16_8,
            fx.slice(self.c_div, (None, fx.Int32(c_index))),
        )

    def store_tiles(self, accs, row_base, col_base):
        row0 = row_base + (self.lane_id // 16) * 4
        col = col_base + (self.lane_id % 16) * 8
        a_scales = [
            self._load_scale_vec4(row0 + i * TILE) for i in range_constexpr(WAVE_TILES)
        ]
        b_lo, b_hi = self._load_b_scale_vec4(col), self._load_b_scale_vec4(col + 4)
        b_scales = Vec.from_elements(
            [b_lo[j] for j in range(4)] + [b_hi[j] for j in range(4)], fx.Float32
        )
        oob = fx.Int32(self.c_rows * self.c_cols)
        col_valid = col < self.c_cols
        for i in range_constexpr(WAVE_TILES):
            for r in range_constexpr(4):
                acc = Vec.from_elements(
                    [Vec(accs[i][j])[r] for j in range(WAVE_TILES)], fx.Int32
                )
                a_scale = Vec.filled(WAVE_TILES, a_scales[i][r], fx.Float32)
                scaled = (acc.to(fx.Float32) * (a_scale * b_scales)).to(fx.BFloat16)
                c_index = (row0 + i * TILE + r) * self.c_cols + col
                self._store_bf16x8(scaled, col_valid.select(c_index, oob))


def _mfma_i8(a, b, c):
    """``c is None`` starts the accumulation from an inline zero.

    The accumulators are pinned to AGPRs like hipBLASLt's: with the LLVM
    intrinsic, the allocator keeps them in VGPRs and the operands in AGPRs,
    leaving too few VGPRs and adding accumulator copies inside the loop.
    LLVM does not see this as an MFMA, so ``_mfma_drain`` must separate the
    last one from any read of its result outside another MFMA.
    """
    res_t = ir.VectorType.get([4], ir.IntegerType.get_signless(32))
    operands = [Vec(a).ir_value(), Vec(b).ir_value()]
    if c is None:
        asm, constraints = "v_mfma_i32_16x16x64_i8 $0, $1, $2, 0", "=a,v,v"
    else:
        asm, constraints = "v_mfma_i32_16x16x64_i8 $0, $1, $2, $3", "=a,v,v,0"
        operands.append(Vec(c).ir_value())
    return _llvm.inline_asm(
        res=res_t,
        operands_=operands,
        asm_string=asm,
        constraints=constraints,
        has_side_effects=True,
    )


def _mfma_drain(accs):
    """Return ``accs`` readable after the XDL-write -> v_accvgpr_read wait states.

    LLVM copies an accumulator out of its AGPRs right at the defining asm, so
    each one passes through an empty asm placed after the nops.
    """
    _llvm.inline_asm(
        res=None,
        operands_=[],
        asm_string="s_nop 7\ns_nop 7\ns_nop 7",
        constraints="",
        has_side_effects=True,
    )
    return [
        [
            _llvm.inline_asm(
                res=Vec(acc).ir_value().type,
                operands_=[Vec(acc).ir_value()],
                asm_string="",
                constraints="=a,0",
                has_side_effects=True,
            )
            for acc in row
        ]
        for row in accs
    ]


def compile_int8_gemm_4w(*, K: int, xcd_swizzle: int = 8):
    assert K % BLOCK_K == 0 and K >= 2 * BLOCK_K
    K_ITERS = K // BLOCK_K
    B_K_STEP = 16 * BLOCK_K  # bytes per K128 tile inside a preshuffled 16-row strip
    HALF = WAVE_TILES // 2

    # Quadrants of the wave's tile grid by (A half, B half).
    def quadrant(a_half, b_half):
        return [
            (a_half * HALF + ia, b_half * HALF + jb)
            for jb in range(HALF)
            for ia in range(HALF)
        ]

    LL, LH, HL, HH = quadrant(0, 0), quadrant(1, 0), quadrant(0, 1), quadrant(1, 1)
    K_FIRST, K_SECOND = range(K_STEPS // 2), range(K_STEPS // 2, K_STEPS)
    ALL_K = range(K_STEPS)

    def kk_major(pairs, kks):
        return [(i, j, kk) for kk in kks for (i, j) in pairs]

    # MFMA order per phase. A-lo/B-lo operands are read in the previous tile's
    # phase 4, A-hi in phase 1, B-hi in phase 2; all lo operands must be dead
    # before phase 4 overwrites them with the next tile.
    LH_ORDER = kk_major(LH, ALL_K)
    LH_SPLIT = 6 * len(LH_ORDER) // 16
    PHASE_MFMAS = (
        kk_major(LL, K_FIRST),
        kk_major(LL, K_SECOND) + LH_ORDER[:LH_SPLIT],
        LH_ORDER[LH_SPLIT:] + kk_major(HL, ALL_K),
        kk_major(HH, ALL_K),
    )
    assert sum(len(p) for p in PHASE_MFMAS) == WAVE_TILES * WAVE_TILES * K_STEPS

    A_LOADS_P2 = 4
    B_LOADS_P3 = LOADS_PER_REGION - 1
    LOADS_BEFORE_P3_WAIT = LOADS_PER_REGION + B_LOADS_P3

    _kname = (
        f"flydsl_bpreshuffle_4w_{BLOCK_M}x{BLOCK_N}x{BLOCK_K}_I8_I8_B16_"
        f"1x{xcd_swizzle}_k{K}"
    )

    @fx.struct
    class SharedStorage:
        A_lds_0: fx.Array[fx.Int8, A_REGION_BYTES, 16]
        B_lds_0: fx.Array[fx.Int8, B_REGION_BYTES, 16]
        A_lds_1: fx.Array[fx.Int8, A_REGION_BYTES, 16]
        B_lds_1: fx.Array[fx.Int8, B_REGION_BYTES, 16]

    @flyc.kernel(name=_kname, known_block_size=[N_WAVES * 64, 1, 1])
    def kernel_gemm(
        A: fx.Tensor,
        B_T: fx.Tensor,
        C: fx.Tensor,
        A_scale: fx.Tensor,
        B_scale: fx.Tensor,
        c_m: fx.Int32,
        c_n: fx.Int32,
    ):
        n_blocks = ceildiv(c_n, BLOCK_N)

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        a_lds = (lds.A_lds_0, lds.A_lds_1)
        b_lds = (lds.B_lds_0, lds.B_lds_1)

        lane_id = fx.thread_idx.x % 64
        wave_id = fx.thread_idx.x // 64
        wave_m = wave_id // 2
        wave_n = wave_id % 2
        if const_expr(xcd_swizzle > 0):
            block_m, block_n = _xcd_swizzle_any(
                ceildiv(c_m, BLOCK_M), n_blocks, wgm=xcd_swizzle
            )
        else:
            block_m, block_n = split_row_major_2d(fx.block_idx.x, n_blocks)

        a_gl_base = (block_m * BLOCK_M) * K
        b_gl_base = (block_n * BLOCK_N) * K

        a_rsrc = buffer_ops.create_buffer_resource(A, num_records_bytes=c_m * K)
        b_rsrc = buffer_ops.create_buffer_resource(B_T, num_records_bytes=c_n * K)
        a_g2s = G2SDirect(
            a_rsrc, _lane_offsets(lane_id, wave_id, K, False), wave_id, A_BLOCK_STRIDE
        )
        b_g2s = G2SDirect(
            b_rsrc, _lane_offsets(lane_id, wave_id, K, True), wave_id, B_BLOCK_STRIDE
        )
        s2r = S2RLoader(wave_m, WAVE_TILES)

        # Operand (tile, kk): lane l reads k16 chunk kk * K_GROUPS + l // 16 of
        # A row l % 16 of the tile, or of B row 8 * (l % 16) + tile of the wave.
        # It sits at base[kk] + tile * A_BLOCK_STRIDE (A) or + tile * 16 (B).
        row_in_block = lane_id % 16
        k_group = lane_id // 16
        a_lane_base = [
            wave_m * (WAVE_ROWS // 16) * A_BLOCK_STRIDE
            + row_in_block * BLOCK_K
            + ((kk * K_GROUPS + k_group) ^ _a_chunk_swizzle(row_in_block)) * 16
            for kk in range_constexpr(K_STEPS)
        ]
        b_lane_base = [
            (wave_n * (WAVE_ROWS // 16) + row_in_block // 2) * B_BLOCK_STRIDE
            + (kk * K_GROUPS + k_group) * 256
            + (row_in_block % 2) * 128
            for kk in range_constexpr(K_STEPS)
        ]

        # K tile t lives in LDS buffer t % 2; ``buf`` is that parity, static
        # even where t is the loop's runtime value.
        def a_loads(tile_k, buf, steps):
            return a_g2s.ops(a_lds[buf], a_gl_base + tile_k * BLOCK_K, steps)

        def b_loads(tile_k, buf, steps):
            return b_g2s.ops(b_lds[buf], b_gl_base + tile_k * B_K_STEP, steps)

        a_ops = [[None] * K_STEPS for _ in range(WAVE_TILES)]
        b_ops = [[None] * K_STEPS for _ in range(WAVE_TILES)]

        def a_read(buf, i, kk):
            def emit():
                a_ops[i][kk] = s2r.load_one(
                    a_lds[buf], a_lane_base[kk] + i * A_BLOCK_STRIDE
                )

            return emit

        def b_read(buf, j, kk):
            def emit():
                b_ops[j][kk] = s2r.load_one(b_lds[buf], b_lane_base[kk] + j * 16)

            return emit

        accs = [[None] * WAVE_TILES for _ in range(WAVE_TILES)]

        def do_mfma(i, j, kk):
            accs[i][j] = _mfma_i8(a_ops[i][kk], b_ops[j][kk], accs[i][j])

        def spread(ops, first, last):
            """Place ``ops`` evenly after MFMAs ``first`` .. ``last - 1``."""
            n = len(ops)
            return [(first + q * (last - first) // n, op) for q, op in enumerate(ops)]

        def run_phase(mfmas, placed, wait=None):
            """Issue MFMAs with each memory op after its assigned MFMA.

            The four waves reach each phase together, so back-to-back LDS-DMA
            loads from all of them overrun the load path and are spread out.
            ``wait`` (vmcnt, lgkmcnt) goes before the last MFMA so the MFMA
            pipe stays busy while the following barrier resolves.
            """
            for idx, (i, j, kk) in enumerate(mfmas):
                if const_expr(wait is not None and idx == len(mfmas) - 1):
                    rocdl.s_waitcnt(vmcnt=wait[0], lgkmcnt=wait[1])
                    rocdl.sched_barrier(0)
                do_mfma(i, j, kk)
                rocdl.sched_barrier(0)
                for slot, op in placed:
                    if const_expr(slot == idx):
                        op()
                        rocdl.sched_barrier(0)

        def interleave(loads, reads):
            ops, step = [], len(reads) // max(len(loads), 1)
            for q, load in enumerate(loads):
                ops += [load] + reads[q * step : (q + 1) * step]
            return ops + reads[len(loads) * step :]

        def lo_reads(buf):
            return (
                [a_read(buf, i, kk) for kk in K_FIRST for i in range(HALF)]
                + [b_read(buf, j, kk) for kk in K_FIRST for j in range(HALF)]
                + [a_read(buf, i, kk) for kk in K_SECOND for i in range(HALF)]
                + [b_read(buf, j, kk) for kk in K_SECOND for j in range(HALF)]
            )

        def k_tile(t, buf, refill, has_next):
            """One K128 tile; ``refill`` loads tile t + 2 into the freed buffer."""
            a_hi_reads = [
                a_read(buf, HALF + i, kk) for kk in ALL_K for i in range(HALF)
            ]
            b_hi_reads = [
                b_read(buf, HALF + j, kk) for kk in ALL_K for j in range(HALF)
            ]
            steps = range(LOADS_PER_REGION)
            p2_loads, p3_loads, p4_loads = [], [], []
            if const_expr(refill):
                p2_loads = a_loads(t + 2, buf, steps[:A_LOADS_P2])
                p3_loads = a_loads(t + 2, buf, steps[A_LOADS_P2:]) + b_loads(
                    t + 2, buf, steps[:B_LOADS_P3]
                )
                p4_loads = b_loads(t + 2, buf, steps[B_LOADS_P3:])

            run_phase(
                PHASE_MFMAS[0],
                spread(a_hi_reads, 0, len(PHASE_MFMAS[0]) - 2),
                wait=(None, 0),
            )
            rocdl.s_barrier()
            p2_ops = interleave(p2_loads, b_hi_reads)
            run_phase(
                PHASE_MFMAS[1],
                spread(p2_ops, 0, len(PHASE_MFMAS[1]) - 2),
                wait=(None, 0),
            )
            rocdl.s_barrier()
            if const_expr(has_next):
                p3_vmcnt = LOADS_BEFORE_P3_WAIT if refill else 0
                run_phase(
                    PHASE_MFMAS[2],
                    spread(p3_loads, 0, len(PHASE_MFMAS[2]) - 2),
                    wait=(p3_vmcnt, None),
                )
                rocdl.s_barrier()
                p4_ops = interleave(p4_loads, lo_reads(1 - buf))
                run_phase(PHASE_MFMAS[3], spread(p4_ops, 0, len(PHASE_MFMAS[3]) - 2))
            else:
                run_phase(PHASE_MFMAS[2], [])
                run_phase(PHASE_MFMAS[3], [])

        # Values live across a tile boundary: the accumulators and the next
        # tile's lo operands.
        def live_state():
            return (
                [accs[i][j] for i in range(WAVE_TILES) for j in range(WAVE_TILES)]
                + [a_ops[i][kk] for i in range(HALF) for kk in ALL_K]
                + [b_ops[j][kk] for j in range(HALF) for kk in ALL_K]
            )

        def set_state(values):
            values = list(values)
            for i in range_constexpr(WAVE_TILES):
                for j in range_constexpr(WAVE_TILES):
                    accs[i][j] = values.pop(0)
            for i in range_constexpr(HALF):
                for kk in range_constexpr(K_STEPS):
                    a_ops[i][kk] = values.pop(0)
            for j in range_constexpr(HALF):
                for kk in range_constexpr(K_STEPS):
                    b_ops[j][kk] = values.pop(0)

        # Prologue: tiles 0 and 1 in flight, wait for tile 0, read its lo operands.
        for t in range_constexpr(2):
            for op in a_loads(t, t, range(LOADS_PER_REGION)) + b_loads(
                t, t, range(LOADS_PER_REGION)
            ):
                op()
        rocdl.s_waitcnt(vmcnt=2 * LOADS_PER_REGION)
        rocdl.s_barrier()
        for op in lo_reads(0):
            op()

        # Tiles 0 and 1 start the accumulators. A runtime loop then takes the
        # tiles that refill, two per trip so the buffer parity is static; a
        # fully unrolled K loop overflows the instruction cache.
        n_pairs = max(0, (K_ITERS - 4) // 2)
        for t in range_constexpr(2):
            k_tile(t, t, t + 2 < K_ITERS, t + 1 < K_ITERS)
        if const_expr(n_pairs > 0):
            for pair, state in range(0, n_pairs, 1, init=live_state()):
                set_state(state)
                t0 = fx.Int32(pair) * 2 + 2
                k_tile(t0, 0, True, True)
                k_tile(t0 + 1, 1, True, True)
                results = yield live_state()
            set_state(results)
        for t in range_constexpr(2 + 2 * n_pairs, K_ITERS):
            k_tile(t, t % 2, t + 2 < K_ITERS, t + 1 < K_ITERS)

        store_c = StoreCInt8Rows(
            A_scale, B_scale, C, c_m, c_n, None, WAVE_TILES, WAVE_TILES
        )
        store_c.store_tiles(
            _mfma_drain(accs),
            block_m * BLOCK_M + wave_m * WAVE_ROWS,
            block_n * BLOCK_N + wave_n * WAVE_ROWS,
        )

    @flyc.jit
    def launch_gemm(
        A: fx.Tensor,
        B_T: fx.Tensor,
        C: fx.Tensor,
        A_scale: fx.Tensor,
        B_scale: fx.Tensor,
        c_m: fx.Int32,
        c_n: fx.Int32,
        stream: fx.Stream,
    ):
        grid_x = ceildiv(c_m, BLOCK_M) * ceildiv(c_n, BLOCK_N)
        kernel_gemm(
            A,
            B_T,
            C,
            A_scale,
            B_scale,
            c_m,
            c_n,
            value_attrs={
                "rocdl.waves_per_eu": 1,
                "rocdl.flat_work_group_size": f"{N_WAVES * 64},{N_WAVES * 64}",
            },
        ).launch(grid=(grid_x, 1, 1), block=(N_WAVES * 64, 1, 1), stream=stream)

    return launch_gemm
