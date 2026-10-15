# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""4-wave true-INT8 matmul with row-wise scaling for AMD CDNA4 (gfx950 only).

One wave per SIMD, a 2x2 wave grid over a 256x256 block and a 128x128 tile
per wave, accumulated in AGPRs with ``v_mfma_i32_16x16x64_i8``. The K loop
follows the hipBLASLt FP8 schedule: two LDS buffers, three barriers per K128
tile, and LDS reads / LDS-DMA loads issued between MFMAs. The loads are
``global_load_lds``, whose signed offset lets one M0 value cover a wave's
whole share of an operand tile.

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

from aiter.jit.utils.chip_info import get_cdna4_num_xcds
from aiter.ops.flydsl.kernels import buffer_ops
from aiter.ops.flydsl.kernels.gemm_a8w8_8wave import (
    S2RLoader,
    StoreC,
    _xcd_swizzle_any,
)
from aiter.ops.flydsl.kernels.kernels_common import ceildiv
from aiter.ops.flydsl.kernels.mfma_preshuffle_pipeline import split_row_major_2d

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


def _lane_offset(lane_id, wave_id, K, preshuffled, block_row0, n_rows, step):
    """Per-lane global offset for LDS-DMA step ``step`` of one region.

    Step ``s`` of wave ``w`` fills half ``s % 2`` (1 KB) of row block
    ``4 * w + s // 2``: rows ``(s % 2) * 8`` onwards for A, k16 chunks
    ``(s % 2) * 4`` onwards for B. A stores row ``r`` as eight 16-byte chunks
    with chunk ``c`` at slot ``c ^ _a_chunk_swizzle(r)``, so every 8 lanes
    load one 128-byte row.

    The offset is from the operand base, for the block starting at row
    ``block_row0``. Nothing bounds a global load, so rows at or past
    ``n_rows`` (whole 16-row strips for B) read the last valid one instead;
    they only reach C rows and columns that are not stored.
    """
    row_block = wave_id * (LOADS_PER_REGION // 2) + step // 2
    if const_expr(preshuffled):
        strip = block_row0 // 16 + row_block
        strip = (strip < n_rows // 16).select(strip, n_rows // 16 - 1)
        return strip * (16 * K) + (step % 2) * 1024 + lane_id * 16
    row = (step % 2) * 8 + lane_id // 8
    chunk = (lane_id % 8) ^ _a_chunk_swizzle(row)
    g_row = block_row0 + row_block * 16 + row
    g_row = (g_row < n_rows).select(g_row, n_rows - 1)
    return g_row * K + chunk * 16


class G2SGlobal:
    """``global_load_lds`` steps of one region, all on one M0 value.

    The instruction offset applies to both the global and the LDS address and
    is signed 13-bit, so with M0 4 KB into the wave's 8 KB of the region it
    reaches all eight steps; the buffer form's unsigned 12-bit offset needs
    four M0 changes per K tile, this one two.

    The loads are inline asm because LLVM asserts on the intrinsic's negative
    LDS memory-operand offset. LLVM does not count them in vmcnt, so the K
    loop waits for them explicitly, and the epilogue's scale loads, which
    persistent workgroups issue before the next block's, are asm as well.
    M0 is written by ``set_m0`` ops, placed an MFMA slot ahead of the loads
    so the write and its hazard ``s_nop`` stay off the loads' slots. LLVM
    does not see M0 in use, so nothing else in the kernel may use it.
    """

    CENTER = 4096

    def __init__(self, tensor, wave_id, block_stride, lds_pair):
        piece = [
            (s // 2) * block_stride + (s % 2) * 1024 for s in range(LOADS_PER_REGION)
        ]
        self.imm = [p - self.CENTER for p in piece]
        assert all(-4096 <= i <= self.CENTER for i in self.imm)
        # The VGPR offset is unsigned, so the SGPR base carries -CENTER.
        self.base = fx.Int64(fx.ptrtoint(fx.get_iter(tensor))) - self.CENTER
        wave_off = wave_id * (LOADS_PER_REGION // 2) * block_stride + self.CENTER
        self.m0 = [
            rocdl.readfirstlane(
                ir.IntegerType.get_signless(32),
                (fx.Int32(fx.ptrtoint(region.ptr)) + wave_off).ir_value(),
            )
            for region in lds_pair
        ]

    def voffset(self, step, lane_offset):
        """VGPR offset of ``step`` for a lane at ``lane_offset`` from the base."""
        return lane_offset - self.imm[step] + self.CENTER

    def set_m0(self, buf):
        """Point M0 at LDS buffer ``buf`` for the ``ops`` that follow."""
        _llvm.inline_asm(
            res=None,
            operands_=[fx.Int32(self.m0[buf]).ir_value()],
            asm_string="s_mov_b32 m0, $0\ns_nop 0",
            constraints="s",
            has_side_effects=True,
        )

    def _load(self, voffset, k_offset, step):
        _llvm.inline_asm(
            res=None,
            operands_=[
                fx.Int32(voffset).ir_value(),
                (self.base + fx.Int64(k_offset)).ir_value(),
            ],
            asm_string=f"global_load_lds_dwordx4 $0, $1 offset:{self.imm[step]}",
            constraints="v,s",
            has_side_effects=True,
        )

    def ops(self, voffsets, k_offset):
        """One op per step, each loading at ``voffsets[step]``."""
        return [
            lambda step=step: self._load(voffsets[step], k_offset, step)
            for step in range(LOADS_PER_REGION)
        ]


class StoreCInt8Rows(StoreC):
    """Scale and store the wave's INT32 accumulators as BF16 rows.

    Lane ``l`` of MFMA tile (i, j) holds row ``16 i + 4 (l // 16) + r`` in
    register ``r`` and column ``8 (l % 16) + j``, so register ``r`` of the
    eight j tiles is one 16-byte store and 16 lanes cover a 256-byte row.
    Preshuffled B has N % 16 == 0: a store is fully inside C or past column N.
    """

    SCALE_LOADS = WAVE_TILES + 2

    def __init__(self, A_scale, B_scale, C, c_rows, c_cols, *args):
        super().__init__(A_scale, B_scale, C, c_rows, c_cols, *args)
        self.out_atom_8 = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.BFloat16)
        self.reg_bf16_8 = fx.make_rmem_tensor(fx.make_layout(8, 1), fx.BFloat16)
        self.sa_rsrc = buffer_ops.create_buffer_resource(
            A_scale, max_size=False, num_records_bytes=c_rows * 4
        )
        self.sb_rsrc = buffer_ops.create_buffer_resource(
            B_scale, max_size=False, num_records_bytes=c_cols * 4
        )

    @staticmethod
    def _load_f32x4(rsrc, byte_offset):
        return _llvm.inline_asm(
            res=ir.VectorType.get([4], ir.F32Type.get()),
            operands_=[fx.Int32(byte_offset).ir_value(), rsrc],
            asm_string="buffer_load_dwordx4 $0, $1, $2, 0 offen",
            constraints="=v,v,s",
            has_side_effects=True,
        )

    def scale_load_ops(self, row_base, col_base, scales):
        """``SCALE_LOADS`` ops appending the scales for ``store_tiles``.

        The loads are asm, so LLVM leaves them out of its vmcnt waits: an LLVM
        wait for them would also wait for the LDS-DMA loads that follow them,
        which it cannot count. The caller waits before the stores.
        """

        def op(rsrc, element):
            return lambda: scales.append(self._load_f32x4(rsrc, element() * 4))

        def row(i):
            return row_base + (self.lane_id // 16) * 4 + i * TILE

        def col(half):
            return col_base + (self.lane_id % 16) * 8 + half * 4

        return [op(self.sa_rsrc, lambda i=i: row(i)) for i in range(WAVE_TILES)] + [
            op(self.sb_rsrc, lambda h=h: col(h)) for h in (0, 1)
        ]

    def _store_bf16x8(self, values, c_index):
        fx.memref_store_vec(values, self.reg_bf16_8)
        fx.copy(
            self.out_atom_8,
            self.reg_bf16_8,
            fx.slice(self.c_div, (None, fx.Int32(c_index))),
        )

    def store_tiles(self, accs, row_base, col_base, scales):
        """``scales`` comes from ``scale_load_ops`` and must have landed."""
        row0 = row_base + (self.lane_id // 16) * 4
        col = col_base + (self.lane_id % 16) * 8
        a_scales = [Vec(s) for s in scales[:WAVE_TILES]]
        b_lo, b_hi = (Vec(s) for s in scales[WAVE_TILES:])
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


def compile_int8_gemm_4w(
    *,
    K: int,
    xcd_swizzle: int = 8,
    num_cu: int = 0,
    num_xcds: int | None = None,
):
    """``num_cu > 0`` makes the kernel persistent over that many workgroups.

    A persistent workgroup loads the first two K tiles of its next output
    block during the last two of the current one, so the next block's K loop
    starts while the epilogue runs. That needs an even number of K tiles, so
    the next block's first tile lands in LDS buffer 0; otherwise each
    workgroup takes one block.
    """
    assert K % BLOCK_K == 0 and K >= 2 * BLOCK_K
    if num_xcds is None:
        num_xcds = get_cdna4_num_xcds() if xcd_swizzle > 0 else 8
    num_xcds = int(num_xcds)
    if num_xcds <= 0 or num_xcds > 8 or num_xcds & (num_xcds - 1):
        raise ValueError(f"num_xcds must be a power of two in [1, 8], got {num_xcds}")
    K_ITERS = K // BLOCK_K
    B_K_STEP = 16 * BLOCK_K  # bytes per K128 tile inside a preshuffled 16-row strip
    HALF = WAVE_TILES // 2
    persistent = num_cu > 0 and K_ITERS % 2 == 0
    assert not persistent or num_cu % num_xcds == 0

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

    # Tile t + 2's A loads go in phase 2 and its B loads in phase 4, so the
    # waves' loads do not crowd one phase. At phase 3's wait, the A loads are
    # the only ones issued after tile t + 1's.
    P3_VMCNT = LOADS_PER_REGION
    # In a persistent workgroup, the previous block's epilogue (four stores
    # per A tile) also comes between tile 1's loads and tile 0's phase 3 wait.
    P3_VMCNT_TILE0 = P3_VMCNT + 4 * WAVE_TILES if persistent else P3_VMCNT
    # The epilogue's scale loads go in phase 1 of tile K_ITERS - 2, ahead of
    # the next block's LDS-DMA loads, and add to that tile's phase 3 wait.
    # Tile K_ITERS - 1's phase 3 wait, or with no next block tile
    # K_ITERS - 2's wait for everything, then covers them.
    SCALE_LOADS = StoreCInt8Rows.SCALE_LOADS
    assert P3_VMCNT_TILE0 + SCALE_LOADS <= 63
    # Phase 1's last A-hi reads and phase 2's last B-hi reads must land
    # before the phase's barrier. Phase 2's A loads do not count against that
    # wait and run to the end of the phase.
    P1_READS_END = len(PHASE_MFMAS[0]) - 6
    P2_READS_END = len(PHASE_MFMAS[1]) - 6

    _kname = (
        f"flydsl_bpreshuffle_4w_{BLOCK_M}x{BLOCK_N}x{BLOCK_K}_I8_I8_B16_"
        f"1x{xcd_swizzle}_k{K}"
        + (f"_p{num_cu}" if persistent else "")
        + (f"_xcd{num_xcds}" if num_xcds != 8 else "")
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
        m_blocks = ceildiv(c_m, BLOCK_M)
        n_blocks = ceildiv(c_n, BLOCK_N)
        num_blocks = m_blocks * n_blocks

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        a_lds = (lds.A_lds_0, lds.A_lds_1)
        b_lds = (lds.B_lds_0, lds.B_lds_1)

        lane_id = fx.thread_idx.x % 64
        wave_id = fx.thread_idx.x // 64
        wave_m = wave_id // 2
        wave_n = wave_id % 2

        def block_coords(block):
            # A persistent grid is a multiple of num_xcds workgroups or
            # covers all blocks, so block b stays on its workgroup's XCD.
            if const_expr(xcd_swizzle > 0):
                return _xcd_swizzle_any(
                    m_blocks,
                    n_blocks,
                    xcd_swizzle,
                    block,
                    num_xcds,
                )
            return split_row_major_2d(block, n_blocks)

        a_g2s = G2SGlobal(A, wave_id, A_BLOCK_STRIDE, a_lds)
        b_g2s = G2SGlobal(B_T, wave_id, B_BLOCK_STRIDE, b_lds)
        s2r = S2RLoader(wave_m, WAVE_TILES)

        def voffset_ops(coords, voffsets):
            """Ops appending the block's A and B LDS-DMA offsets to ``voffsets``.

            ``coords`` holds (block_m, block_n) by the time the first op runs.
            """

            def op(g2s, preshuffled, step):
                def emit():
                    row0, n_rows = (
                        (coords[1] * BLOCK_N, c_n)
                        if preshuffled
                        else (coords[0] * BLOCK_M, c_m)
                    )
                    lane_offset = _lane_offset(
                        lane_id, wave_id, K, preshuffled, row0, n_rows, step
                    )
                    voffsets[int(preshuffled)].append(g2s.voffset(step, lane_offset))

                return emit

            return [op(a_g2s, False, s) for s in range(LOADS_PER_REGION)] + [
                op(b_g2s, True, s) for s in range(LOADS_PER_REGION)
            ]

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
        # even where t is the loop's runtime value. ``voffsets`` (A, B) picks
        # the output block.
        def tile_loads(voffsets, tile_k):
            """A and B loads of a K tile; M0 must point at its buffer."""
            return (
                a_g2s.ops(voffsets[0], tile_k * BLOCK_K),
                b_g2s.ops(voffsets[1], tile_k * B_K_STEP),
            )

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

        def k_tile(buf, refill, has_next, p3_vmcnt=P3_VMCNT, p1_ops=(), p3_ops=()):
            """One K128 tile.

            ``refill`` (voffsets, tile_k) loads the K tile two ahead into the
            freed buffer; ``has_next`` reads the next one's lo operands.
            ``p1_ops`` go after phase 1's reads, ``p3_ops`` over phase 3.
            """
            a_hi_reads = [
                a_read(buf, HALF + i, kk) for kk in ALL_K for i in range(HALF)
            ]
            b_hi_reads = [
                b_read(buf, HALF + j, kk) for kk in ALL_K for j in range(HALF)
            ]
            p2_loads, p4_loads = [], []
            p1_m0, p3_m0 = [], []
            if const_expr(refill is not None):
                p2_loads, p4_loads = tile_loads(*refill)
                p1_m0 = [(len(PHASE_MFMAS[0]) - 2, lambda: a_g2s.set_m0(buf))]
                p3_m0 = [(len(PHASE_MFMAS[2]) - 2, lambda: b_g2s.set_m0(buf))]

            p1_placed = spread(a_hi_reads, 0, P1_READS_END)
            p1_placed += spread(list(p1_ops), P1_READS_END, len(PHASE_MFMAS[0]) - 1)
            run_phase(PHASE_MFMAS[0], p1_placed + p1_m0, wait=(None, 0))
            rocdl.s_barrier()
            p2_placed = spread(b_hi_reads, 0, P2_READS_END)
            p2_placed += spread(p2_loads, 1, len(PHASE_MFMAS[1]) - 1)
            run_phase(PHASE_MFMAS[1], p2_placed, wait=(None, 0))
            rocdl.s_barrier()
            if const_expr(has_next):
                p3_placed = spread(list(p3_ops), 0, len(PHASE_MFMAS[2]) - 2)
                p3_wait = p3_vmcnt if refill is not None else 0
                run_phase(PHASE_MFMAS[2], p3_placed + p3_m0, wait=(p3_wait, None))
                rocdl.s_barrier()
                p4_ops = interleave(p4_loads, lo_reads(1 - buf))
                run_phase(PHASE_MFMAS[3], spread(p4_ops, 0, len(PHASE_MFMAS[3]) - 2))
            else:
                run_phase(PHASE_MFMAS[2], [])
                run_phase(PHASE_MFMAS[3], [])

        def lo_state():
            return [a_ops[i][kk] for i in range(HALF) for kk in ALL_K] + [
                b_ops[j][kk] for j in range(HALF) for kk in ALL_K
            ]

        def set_lo_state(values):
            for i in range_constexpr(HALF):
                for kk in range_constexpr(K_STEPS):
                    a_ops[i][kk] = values.pop(0)
            for j in range_constexpr(HALF):
                for kk in range_constexpr(K_STEPS):
                    b_ops[j][kk] = values.pop(0)

        # Values live across a K tile boundary: the accumulators and the next
        # tile's lo operands.
        def live_state():
            accs_flat = [
                accs[i][j] for i in range(WAVE_TILES) for j in range(WAVE_TILES)
            ]
            return accs_flat + lo_state()

        def set_state(values):
            values = list(values)
            for i in range_constexpr(WAVE_TILES):
                for j in range_constexpr(WAVE_TILES):
                    accs[i][j] = values.pop(0)
            set_lo_state(values)

        store_c = StoreCInt8Rows(
            A_scale, B_scale, C, c_m, c_n, None, WAVE_TILES, WAVE_TILES
        )

        # Prologue: the first block's K tiles 0 and 1 in flight, wait for
        # tile 0, read its lo operands. Tile 0's phase 3 wait in a persistent
        # workgroup counts on an epilogue before it, so the first block also
        # waits for tile 1 here.
        block0 = fx.Int32(fx.block_idx.x)
        coords = list(block_coords(block0))
        voffsets = ([], [])
        for op in voffset_ops(coords, voffsets):
            op()
        for t in range_constexpr(2):
            a_loads, b_loads = tile_loads(voffsets, t)
            for g2s, loads in ((a_g2s, a_loads), (b_g2s, b_loads)):
                g2s.set_m0(t)
                for op in loads:
                    op()
        rocdl.s_waitcnt(vmcnt=0 if persistent else 2 * LOADS_PER_REGION)
        rocdl.s_barrier()
        for op in lo_reads(0):
            op()

        # Values live across an output block boundary: its coordinates and
        # offsets, and the lo operands of its first K tile.
        def block_state():
            return coords + voffsets[0] + voffsets[1] + lo_state()

        def set_block_state(values):
            values = list(values)
            coords[:] = [values.pop(0) for _ in range(2)]
            for v in voffsets:
                v[:] = [values.pop(0) for _ in range(LOADS_PER_REGION)]
            set_lo_state(values)

        def static_tile(t, next_voffsets, next_ops, scale_ops):
            """K tile ``t`` of a block, outside the runtime K loop.

            The last two refill from the next block at ``next_voffsets``;
            tile 0 spreads ``next_ops`` over phase 3 and tile K_ITERS - 2
            issues ``scale_ops``.
            """
            refill = None
            if const_expr(t + 2 < K_ITERS):
                refill = (voffsets, t + 2)
            elif const_expr(persistent):
                refill = (next_voffsets, t + 2 - K_ITERS)
            p3_vmcnt = P3_VMCNT_TILE0 if t == 0 else P3_VMCNT
            p1_ops = []
            if const_expr(t == K_ITERS - 2):
                p1_ops = scale_ops
                p3_vmcnt += SCALE_LOADS
            has_next = t + 1 < K_ITERS or persistent
            p3_ops = next_ops if t == 0 else ()
            k_tile(t % 2, refill, has_next, p3_vmcnt, p1_ops, p3_ops)

        n_pairs = max(0, (K_ITERS - 4) // 2)
        grid = fx.Int32(fx.grid_dim.x)
        for block, bstate in range(block0, num_blocks, grid, init=block_state()):
            set_block_state(bstate)
            for i in range_constexpr(WAVE_TILES):
                for j in range_constexpr(WAVE_TILES):
                    accs[i][j] = None

            # The next block's coordinates and offsets, computed during tile
            # 0's phase 3 unless tile 0 already loads that block. The last
            # block reloads itself; nothing reads those loads.
            next_coords, next_voffsets, next_ops = [], ([], []), []
            if const_expr(persistent):
                this_block = fx.Int32(block)

                def next_coords_op(this_block=this_block, next_coords=next_coords):
                    nxt = this_block + grid
                    nxt = (nxt < num_blocks).select(nxt, this_block)
                    next_coords.extend(block_coords(nxt))

                next_ops = [next_coords_op] + voffset_ops(next_coords, next_voffsets)
                if const_expr(K_ITERS == 2):
                    for op in next_ops:
                        op()
                    next_ops = []

            row_base = coords[0] * BLOCK_M + wave_m * WAVE_ROWS
            col_base = coords[1] * BLOCK_N + wave_n * WAVE_ROWS
            scales = []
            scale_ops = store_c.scale_load_ops(row_base, col_base, scales)
            tile_args = (next_voffsets, next_ops, scale_ops)

            # Tiles 0 and 1 start the accumulators. A runtime loop then takes
            # the tiles that refill from this block, two per trip so the
            # buffer parity is static; a fully unrolled K loop overflows the
            # instruction cache.
            static_tile(0, *tile_args)
            static_tile(1, *tile_args)
            if const_expr(n_pairs > 0):
                for pair, state in range(0, n_pairs, 1, init=live_state()):
                    set_state(state)
                    t0 = fx.Int32(pair) * 2 + 2
                    k_tile(0, (voffsets, t0 + 2), True)
                    k_tile(1, (voffsets, t0 + 3), True)
                    results = yield live_state()
                set_state(results)
            for t in range_constexpr(2 + 2 * n_pairs, K_ITERS):
                static_tile(t, *tile_args)

            store_c.store_tiles(_mfma_drain(accs), row_base, col_base, scales)
            if const_expr(persistent):
                coords[:] = next_coords
                for v, nv in zip(voffsets, next_voffsets):
                    v[:] = nv
            _ = yield block_state()

        if const_expr(persistent):
            # LDS-DMA must not land in LDS that a later workgroup owns.
            rocdl.s_waitcnt(vmcnt=0)

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
        if const_expr(persistent):
            grid_x = (grid_x > num_cu).select(fx.Int32(num_cu), grid_x)
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
