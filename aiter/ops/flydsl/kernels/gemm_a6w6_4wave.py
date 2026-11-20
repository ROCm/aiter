# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""4-wave MXFP6 (A6W6) GEMM for AMD CDNA4 (gfx950 only): C = A * B^T in BF16.

A and B are FP6 E2M3 with one E8M0 scale per 32 K values, quantized and packed by
``aiter.ops.gemm_op_a6w6.quant_mxfp6_gemm`` (layout ``mxfp6_c0c1_256_padk2``), the
operands of AITER's ASM A6W6 kernels. One wave per SIMD, a 2x2 wave grid over a
256x256 block and a 128x128 tile per wave, accumulated in AGPRs with
``v_mfma_scale_f32_16x16x128_f8f6f4``.

The packed layout is the MFMA's operand layout. A 256-row, 128-K tile is 24576
bytes: 1024 blocks of 32 values of one row (24 bytes), block ``64 rb + l`` holding
row ``16 rb + l % 16`` and K group ``l // 16``, with each block's first 16 bytes in
the tile's first 16 KB and its last 8 in the rest. Lane ``l`` of a wave therefore
reads its operand of row block ``rb`` with one ``ds_read_b128`` and one
``ds_read_b64`` at ``l * 16`` and ``l * 8``, and a tile is copied to LDS as it is.
A tile's scales are 1024 bytes; lane ``l`` of the wave on row half ``h`` finds
the scales of its row in the half's eight row blocks in the eight bytes at
``512 h + 8 l``.

K loop: three LDS stages, each one K128 tile of A and of B and their scales. K tile
``t + 3`` is loaded into tile ``t``'s stage during tile ``t`` (``global_load_lds``),
and tile ``t + 1``'s operands and scales are read into VGPRs, each operand after its
register's last MFMA of tile ``t``. One barrier per tile: before it each wave waits
for tile ``t + 1``'s loads and for its own reads of tile ``t``'s stage. The last
three tiles load nothing, so the packed layout's guard tiles are never read.

The scales go through LDS, as in the ASM ``dmabig`` kernels: a scale load into
VGPRs would be inline asm, which LLVM takes as complete when issued, and the loop
carries its result, which LLVM may copy before the load lands.

The weights are the MFMA's first operand, so lane ``l`` of output tile (i, j)
holds four adjacent columns ``16 j + 4 (l // 16) + r`` of row ``16 i + l % 16``.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir

# The LLVM dialect is used only for inline assembly at instructions FlyDSL 0.5
# cannot represent through its stable surface: M0 ownership, signed-offset LDS DMA,
# the FP6 scaled MFMA into AGPRs, and the accumulator hazard boundary. The raw
# ROCDL op is the epilogue's v_permlane16_swap, which has no fx wrapper.
from flydsl._mlir.dialects import llvm as _llvm
from flydsl._mlir.dialects import rocdl as _rocdl
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import Vector as Vec

from aiter.ops.flydsl.kernels.gemm_a8w8_8wave import _xcd_swizzle_any
from aiter.ops.flydsl.kernels.kernels_common import ceildiv

BLOCK_M = 256
BLOCK_N = 256
BLOCK_K = 128
N_WAVES = 4
WAVE_ROWS = 128
TILE = 16
WAVE_TILES = WAVE_ROWS // TILE
STAGES = 3
TILE_BYTES = BLOCK_M * BLOCK_K * 6 // 8  # one packed 256-row K128 tile
C1_OFFSET = TILE_BYTES * 2 // 3  # the blocks' last 8 bytes start here
SCALE_TILE_BYTES = BLOCK_M * BLOCK_K // 32
PADK = 2  # guard K tiles of the packed layout, past K
LOADS_PER_REGION = TILE_BYTES // (N_WAVES * 64 * 16)  # 16-byte loads per lane
WAVE_REGION_BYTES = LOADS_PER_REGION * 1024
# A stage: the A tile, the B tile, the A scales, the B scales.
B_OFFSET = TILE_BYTES
SCALES_OFFSET = 2 * TILE_BYTES
STAGE_BYTES = 2 * TILE_BYTES + 2 * SCALE_TILE_BYTES
# A tile's VMEM ops: the A and B LDS-DMA loads, then the two scale loads.
TILE_VMEM = 2 * LOADS_PER_REGION + 2
CFG_E2M3 = 2  # cbsz / blgp: FP6 E2M3 for both operands


def _readfirstlane(value):
    return fx.rocdl.readfirstlane(fx.Int32.ir_type, fx.Int32(value).ir_value())


def _set_m0(value):
    # FlyDSL 0.5 has no stable M0 setter with the write's hazard nop. LLVM does not
    # see M0 in use, so nothing else in the kernel may use it.
    _llvm.inline_asm(
        res=None,
        operands_=[fx.Int32(value).ir_value()],
        asm_string="s_mov_b32 m0, $0\ns_nop 0",
        constraints="s",
        has_side_effects=True,
    )


def _lds_dma(instruction, voffset, base, imm):
    # FlyDSL 0.5 cannot encode the signed LDS memory-operand offset; its intrinsic
    # path reaches an LLVM assertion, and with non-negative offsets it needs two M0
    # values per operand tile, which is slower. LLVM does not count these loads in
    # vmcnt, so the K loop waits for them explicitly.
    _llvm.inline_asm(
        res=None,
        operands_=[fx.Int32(voffset).ir_value(), fx.Int64(base).ir_value()],
        asm_string=f"{instruction} $0, $1 offset:{imm}",
        constraints="v,s",
        has_side_effects=True,
    )


class G2SOperand:
    """``global_load_lds_dwordx4`` of a wave's 6 KB share of one operand tile.

    The instruction offset applies to the global and the LDS address and is signed
    13-bit, so with M0 in the middle of the wave's share it reaches all six loads.
    The tile is copied as it is, so both addresses share one offset.
    """

    CENTER = 3072

    def __init__(self, tensor, wave_id, stage_ptrs, region, tile_row_offset, lane_id):
        self.imm = [q * 1024 - self.CENTER for q in range(LOADS_PER_REGION)]
        assert all(-4096 <= i < 4096 for i in self.imm)
        self.base = fx.Int64(fx.ptrtoint(fx.get_iter(tensor)))
        wave_off = wave_id * WAVE_REGION_BYTES + self.CENTER
        self.m0 = [
            _readfirstlane(fx.Int32(fx.ptrtoint(ptr)) + region + wave_off)
            for ptr in stage_ptrs
        ]
        self.voffset = tile_row_offset + wave_off + lane_id * 16

    def ops(self, t, stage):
        """One op per load; the first also points M0 at ``stage``."""

        def op(q):
            if const_expr(q == 0):
                _set_m0(self.m0[stage])
            _lds_dma(
                "global_load_lds_dwordx4",
                self.voffset,
                self.base + fx.Int64(t * TILE_BYTES),
                self.imm[q],
            )

        return [lambda q=q: op(q) for q in range(LOADS_PER_REGION)]


class G2SScales:
    """``global_load_lds_dword`` of a wave's 256 bytes of the A and the B scales.

    One M0 value covers both: the A scales sit 1024 bytes below the B scales, and
    the instruction offset (-1024 or 0) applies to both addresses.
    """

    def __init__(self, a_scale, b_scale, wave_id, stage_ptrs, a_row, b_row, lane_id):
        self.bases = (
            fx.Int64(fx.ptrtoint(fx.get_iter(a_scale))),
            fx.Int64(fx.ptrtoint(fx.get_iter(b_scale))),
        )
        wave_off = wave_id * 256 + SCALE_TILE_BYTES
        self.m0 = [
            _readfirstlane(fx.Int32(fx.ptrtoint(ptr)) + SCALES_OFFSET + wave_off)
            for ptr in stage_ptrs
        ]
        lane_off = wave_id * 256 + lane_id * 4
        self.voffsets = (
            a_row * SCALE_TILE_BYTES + lane_off + SCALE_TILE_BYTES,
            b_row * SCALE_TILE_BYTES + lane_off,
        )

    def ops(self, t, stage):
        def op(k):
            if const_expr(k == 0):
                _set_m0(self.m0[stage])
            _lds_dma(
                "global_load_lds_dword",
                self.voffsets[k],
                self.bases[k] + fx.Int64(t * SCALE_TILE_BYTES),
                -SCALE_TILE_BYTES if k == 0 else 0,
            )

        return [lambda k=k: op(k) for k in (0, 1)]


def _lds_load(ptr, offset, nbytes):
    at = fx.add_offset(ptr, fx.make_int_tuple(offset))
    view = fx.make_view(fx.recast_iter(fx.Uint8, at), fx.make_layout(nbytes, 1))
    return Vec(view.load()).bitcast(fx.Int32)


def _read_operand(ptr, region, row_block, lane_id):
    """Lane ``lane_id``'s 32 FP6 values of row block ``row_block``: six dwords."""
    c0 = _lds_load(ptr, region + row_block * 1024 + lane_id * 16, 16)
    c1 = _lds_load(ptr, region + C1_OFFSET + row_block * 512 + lane_id * 8, 8)
    return Vec.from_elements([c0[0], c0[1], c0[2], c0[3], c1[0], c1[1]], fx.Int32)


def _mfma(b, a, scale_b, scale_a, j, i, c):
    """``c is None`` starts the accumulation from an inline zero.

    The accumulators are pinned to AGPRs as in the INT8 4-wave kernel: with the
    intrinsic the allocator keeps them in VGPRs. Scale byte ``j % 4`` of
    ``scale_b`` belongs to B row block ``j`` and byte ``i % 4`` of ``scale_a`` to A
    row block ``i``; op_sel and op_sel_hi hold the byte's low and high bit.
    """
    sel = f"op_sel:[{j % 2},{i % 2},0] op_sel_hi:[{(j % 4) // 2},{(i % 4) // 2},0]"
    fmt = f"cbsz:{CFG_E2M3} blgp:{CFG_E2M3}"
    operands = [
        Vec(b).ir_value(),
        Vec(a).ir_value(),
        fx.Int32(scale_b).ir_value(),
        fx.Int32(scale_a).ir_value(),
    ]
    if c is None:
        asm = f"v_mfma_scale_f32_16x16x128_f8f6f4 $0, $1, $2, 0, $3, $4 {sel} {fmt}"
        constraints = "=a,v,v,v,v"
    else:
        asm = f"v_mfma_scale_f32_16x16x128_f8f6f4 $0, $1, $2, $5, $3, $4 {sel} {fmt}"
        constraints = "=a,v,v,v,v,0"
        operands.append(Vec(c).ir_value())
    return _llvm.inline_asm(
        res=fx.Vector.make_type(4, fx.Float32),
        operands_=operands,
        asm_string=asm,
        constraints=constraints,
        has_side_effects=True,
    )


def _permlane16_swap(x, y):
    """``v_permlane16_swap_b32``: x's odd 16-lane rows trade places with y's even rows."""
    pair_t = ir.Type.parse("!llvm.struct<(i32, i32)>")
    swapped = _rocdl.permlane16_swap(
        pair_t, fx.Int32(x).ir_value(), fx.Int32(y).ir_value(), False, False
    )
    return tuple(
        fx.Int32(_llvm.extractvalue(fx.Int32.ir_type, swapped, [k])) for k in (0, 1)
    )


def _mfma_drain(accs):
    """Return ``accs`` readable after the XDL-write -> v_accvgpr_read wait states.

    LLVM does not see the asm MFMA as one, so it would read an accumulator right
    after its last MFMA; each passes through an empty asm placed after the nop.
    The callers leave EPI_LAG MFMAs between an accumulator's last MFMA and this.
    """
    _llvm.inline_asm(
        res=None,
        operands_=[],
        asm_string="s_nop 7",
        constraints="",
        has_side_effects=True,
    )
    return [
        _llvm.inline_asm(
            res=Vec(acc).ir_value().type,
            operands_=[Vec(acc).ir_value()],
            asm_string="",
            constraints="=a,0",
            has_side_effects=True,
        )
        for acc in accs
    ]


# The last tile stores output tile pair (i, g) EPI_LAG MFMAs after the pair's last
# MFMA, so the epilogue overlaps the remaining MFMAs, as in the ASM kernels.
EPI_LAG = 2


class Schedule:
    """A K tile's MFMA order and the slots of its barrier, reads and VMEM ops.

    ``mblk_n`` B row blocks at a time are swept over the eight A blocks, the order
    the ASM generators call FlyDSL ``mblk=(8, mblk_n)``: B blocks are dead after
    their sweep, A blocks after the last sweep. The barrier follows MFMA ``bar``;
    each operand is re-read after its last MFMA and the barrier. Four waves issue
    together, so the tile's VMEM ops are spread evenly over the slots after the
    barrier that hold no read, as in the ASM generators.
    """

    def __init__(self, mblk_n=2, bar=None):
        assert mblk_n in (2, 4)
        self.mblk_n = mblk_n
        self.order = [
            (j, i)
            for g in range(WAVE_TILES // mblk_n)
            for i in range(WAVE_TILES)
            for j in range(g * mblk_n, (g + 1) * mblk_n)
        ]
        last = lambda key: max(k for k, cell in enumerate(self.order) if key(cell))
        self.last_b = {j: last(lambda c, j=j: c[0] == j) for j in range(WAVE_TILES)}
        self.last_a = {i: last(lambda c, i=i: c[1] == i) for i in range(WAVE_TILES)}
        first_last = min(min(self.last_b.values()), min(self.last_a.values()))
        self.bar = first_last if bar is None else bar
        assert 0 <= self.bar < len(self.order)
        read_slots = {
            max(s, self.bar) for s in list(self.last_b.values()) + list(self.last_a.values())
        }
        free = [k for k in range(self.bar + 1, len(self.order)) if k not in read_slots]
        assert len(free) >= TILE_VMEM
        self.vmem_slots = [free[k * len(free) // TILE_VMEM] for k in range(TILE_VMEM)]
        self.pair_done = {
            (i, g): last(lambda c, i=i, g=g: c[1] == i and c[0] // 2 == g)
            for i in range(WAVE_TILES)
            for g in range(WAVE_TILES // 2)
        }

    @property
    def tag(self):
        return f"m8{self.mblk_n}b{self.bar}"


def compile_a6w6_gemm_4w(
    *,
    K: int,
    xcd_swizzle: int = 4,
    num_xcds: int = 8,
    mblk_n: int = 2,
    bar: int | None = 8,
):
    """``K`` is the padded contraction dimension, a multiple of 128 of at least 384.

    ``xcd_swizzle`` is the M-tile group of the tile order; ``mblk_n`` and ``bar``
    select the K tile's schedule (``Schedule``).
    """
    assert K % BLOCK_K == 0 and K >= STAGES * BLOCK_K
    sched = Schedule(mblk_n, bar)
    K_TILES = K // BLOCK_K
    NK_PAD = K_TILES + PADK
    # Tiles 0 .. LOADING - 1 load the tile three ahead.
    LOADING = K_TILES - STAGES
    # Tile 0 starts the accumulators; the next loading tiles run in trips of three,
    # so stages are static; the rest are static.
    TRIPS = max(0, (LOADING - 1) // STAGES)
    STATIC_FROM = 1 + TRIPS * STAGES

    _kname = (
        f"flydsl_a6w6_4w_{BLOCK_M}x{BLOCK_N}x{BLOCK_K}_{sched.tag}_1x{xcd_swizzle}_k{K}"
    )

    @fx.struct
    class SharedStorage:
        stage_0: fx.Array[fx.Int8, STAGE_BYTES, 16]
        stage_1: fx.Array[fx.Int8, STAGE_BYTES, 16]
        stage_2: fx.Array[fx.Int8, STAGE_BYTES, 16]

    @flyc.kernel(name=_kname, known_block_size=[N_WAVES * 64, 1, 1])
    def kernel_gemm(
        A: fx.Tensor,
        B: fx.Tensor,
        C: fx.Tensor,
        A_scale: fx.Tensor,
        B_scale: fx.Tensor,
        c_m: fx.Int32,
        c_n: fx.Int32,
    ):
        m_blocks = c_m // BLOCK_M
        n_blocks = c_n // BLOCK_N

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        stage_ptrs = [lds.stage_0.ptr, lds.stage_1.ptr, lds.stage_2.ptr]

        lane_id = fx.thread_idx.x % 64
        wave_id = fx.thread_idx.x // 64
        wave_m = wave_id // 2
        wave_n = wave_id % 2

        block = fx.Int32(fx.block_idx.x)
        if const_expr(xcd_swizzle > 0):
            block_m, block_n = _xcd_swizzle_any(
                m_blocks, n_blocks, xcd_swizzle, block, num_xcds
            )
        else:
            block_m, block_n = block // n_blocks, block % n_blocks
        a_row = fx.Int32(block_m) * NK_PAD
        b_row = fx.Int32(block_n) * NK_PAD

        g2s = (
            G2SOperand(A, wave_id, stage_ptrs, 0, a_row * TILE_BYTES, lane_id),
            G2SOperand(B, wave_id, stage_ptrs, B_OFFSET, b_row * TILE_BYTES, lane_id),
        )
        g2s_scales = G2SScales(
            A_scale, B_scale, wave_id, stage_ptrs, a_row, b_row, lane_id
        )

        def tile_vmem(t, stage):
            """Tile ``t``'s VMEM ops into ``stage``: A, B, then the scales."""
            return g2s[0].ops(t, stage) + g2s[1].ops(t, stage) + g2s_scales.ops(t, stage)

        def read_a(stage, i):
            return _read_operand(
                stage_ptrs[stage], 0, wave_m * WAVE_TILES + i, lane_id
            )

        def read_b(stage, j):
            return _read_operand(
                stage_ptrs[stage], B_OFFSET, wave_n * WAVE_TILES + j, lane_id
            )

        def read_scales(stage):
            ptr = stage_ptrs[stage]
            return (
                _lds_load(ptr, SCALES_OFFSET + wave_m * 512 + lane_id * 8, 8),
                _lds_load(
                    ptr, SCALES_OFFSET + SCALE_TILE_BYTES + wave_n * 512 + lane_id * 8, 8
                ),
            )

        # The epilogue's store context. Lane l of output tile (i, j) holds row
        # 16 i + l % 16 and the four columns 16 j + 4 (l // 16) + r of the wave's
        # tile. Swapping the 16-lane rows of tiles j and j + 1 gives lane group g of
        # the pair the eight columns 16 (j + g % 2) + 8 (g // 2) of its row: one
        # 16-byte store per lane. The output is written once, so the stores are
        # non-temporal, as the ASM kernels' are.
        c_tensor = fx.rocdl.make_buffer_tensor(
            C, max_size=False, num_records_bytes=fx.Int64(c_m) * fx.Int64(c_n) * 2
        )
        c_div = fx.logical_divide(c_tensor, fx.make_layout(1, 1))
        out_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(2), fx.BFloat16)
        out_reg = fx.make_rmem_tensor(fx.make_layout(8, 1), fx.BFloat16)
        group = lane_id // 16
        row0 = fx.Int32(block_m) * BLOCK_M + wave_m * WAVE_ROWS + lane_id % 16
        col0 = (
            fx.Int32(block_n) * BLOCK_N
            + wave_n * WAVE_ROWS
            + (group % 2) * TILE
            + (group // 2) * 8
        )

        def store_pair(accs, i, g):
            """Output tiles (i, 2 g) and (i, 2 g + 1), EPI_LAG MFMAs after their last."""
            lo, hi = _mfma_drain([accs[i][2 * g], accs[i][2 * g + 1]])
            lo = Vec(lo).to(fx.BFloat16).bitcast(fx.Int32)
            hi = Vec(hi).to(fx.BFloat16).bitcast(fx.Int32)
            x0, y0 = _permlane16_swap(lo[0], hi[0])
            x1, y1 = _permlane16_swap(lo[1], hi[1])
            row = Vec.from_elements([x0, x1, y0, y1], fx.Int32).bitcast(fx.BFloat16)
            fx.memref_store_vec(row, out_reg)
            index = (row0 + i * TILE) * c_n + col0 + g * 2 * TILE
            fx.copy(out_atom, out_reg, fx.slice(c_div, (None, fx.Int32(index))))

        def k_tile(
            accs, a_ops, b_ops, scales, stage, has_next, vmem_wait, prefetch, epilogue
        ):
            """One K tile in LDS ``stage``; ``scales`` are its (A, B) scale words.

            ``prefetch`` (tile, stage) spreads that tile's VMEM ops after the barrier;
            ``vmem_wait`` is the vmcnt the barrier waits for; ``has_next`` reads the
            next tile's operands and scales from the next stage. The last tile
            (``epilogue``) has no barrier and stores the output as it completes.
            """
            next_stage = (stage + 1) % STAGES
            new_a = [None] * WAVE_TILES
            new_b = [None] * WAVE_TILES
            new_scales = None
            sa = (Vec(scales[0])[0], Vec(scales[0])[1])
            sb = (Vec(scales[1])[0], Vec(scales[1])[1])
            vmem = [] if prefetch is None else tile_vmem(*prefetch)
            for idx, (j, i) in enumerate(sched.order):
                accs[i][j] = _mfma(
                    b_ops[j], a_ops[i], sb[j // 4], sa[i // 4], j, i, accs[i][j]
                )
                fx.rocdl.sched_barrier(0)
                if const_expr(epilogue):
                    for (pi, pg), done in sched.pair_done.items():
                        if const_expr(done + EPI_LAG == idx):
                            store_pair(accs, pi, pg)
                            fx.rocdl.sched_barrier(0)
                    continue
                if const_expr(idx == sched.bar):
                    fx.rocdl.s_waitcnt(vmcnt=vmem_wait, lgkmcnt=0)
                    fx.rocdl.sched_barrier(0)
                    fx.rocdl.s_barrier()
                    fx.rocdl.sched_barrier(0)
                    if const_expr(has_next):
                        new_scales = read_scales(next_stage)
                        fx.rocdl.sched_barrier(0)
                if const_expr(idx < sched.bar):
                    continue
                if const_expr(len(vmem) > 0 and idx in sched.vmem_slots):
                    vmem.pop(0)()
                    fx.rocdl.sched_barrier(0)
                if const_expr(has_next):
                    # Each read follows its register's last MFMA, and the barrier.
                    for jj in range_constexpr(WAVE_TILES):
                        if const_expr(new_b[jj] is None and sched.last_b[jj] <= idx):
                            new_b[jj] = read_b(next_stage, jj)
                            fx.rocdl.sched_barrier(0)
                    for ii in range_constexpr(WAVE_TILES):
                        if const_expr(new_a[ii] is None and sched.last_a[ii] <= idx):
                            new_a[ii] = read_a(next_stage, ii)
                            fx.rocdl.sched_barrier(0)
            while const_expr(len(vmem) > 0):
                vmem.pop(0)()
                fx.rocdl.sched_barrier(0)
            if const_expr(epilogue):
                for (pi, pg), done in sched.pair_done.items():
                    if const_expr(done + EPI_LAG >= len(sched.order)):
                        store_pair(accs, pi, pg)
                        fx.rocdl.sched_barrier(0)
            return accs, new_a, new_b, new_scales

        def step(accs, a_ops, b_ops, scales, t, stage, loading, has_next, two_after):
            """Tile ``t``; ``loading`` loads tile ``t + 3`` into its stage.

            Before the barrier, tile ``t + 1``'s VMEM ops must have landed; tile
            ``t + 2``'s, issued after them, may still run if there is one.
            """
            return k_tile(
                accs,
                a_ops,
                b_ops,
                scales,
                stage,
                has_next,
                TILE_VMEM if two_after else 0,
                (t + STAGES, stage) if loading else None,
                not has_next,
            )

        def flat_state(accs, a_ops, b_ops, scales):
            return (
                [accs[i][j] for i in range(WAVE_TILES) for j in range(WAVE_TILES)]
                + list(a_ops)
                + list(b_ops)
                + list(scales)
            )

        def unflat_state(values):
            values = list(values)
            accs = [
                [values.pop(0) for _ in range(WAVE_TILES)] for _ in range(WAVE_TILES)
            ]
            a_ops = [values.pop(0) for _ in range(WAVE_TILES)]
            b_ops = [values.pop(0) for _ in range(WAVE_TILES)]
            scales = (values.pop(0), values.pop(0))
            return accs, a_ops, b_ops, scales

        # Prologue: tiles 0, 1 and 2 in flight; wait for tile 0 and read it.
        for t in range_constexpr(STAGES):
            for op in tile_vmem(t, t):
                op()
        fx.rocdl.s_waitcnt(vmcnt=(STAGES - 1) * TILE_VMEM)
        fx.rocdl.s_barrier()
        scales = read_scales(0)
        a_ops = [read_a(0, i) for i in range_constexpr(WAVE_TILES)]
        b_ops = [read_b(0, j) for j in range_constexpr(WAVE_TILES)]
        accs = [[None] * WAVE_TILES for _ in range(WAVE_TILES)]

        def static_step(accs, a_ops, b_ops, scales, t):
            return step(
                accs,
                a_ops,
                b_ops,
                scales,
                t,
                t % STAGES,
                t < LOADING,
                t + 1 < K_TILES,
                t + 2 < K_TILES,
            )

        accs, a_ops, b_ops, scales = static_step(accs, a_ops, b_ops, scales, 0)
        if const_expr(TRIPS > 0):
            for trip, state in range(
                0, TRIPS, 1, init=flat_state(accs, a_ops, b_ops, scales)
            ):
                accs, a_ops, b_ops, scales = unflat_state(state)
                t0 = fx.Int32(trip) * STAGES + 1
                for u in range_constexpr(STAGES):
                    accs, a_ops, b_ops, scales = step(
                        accs,
                        a_ops,
                        b_ops,
                        scales,
                        t0 + u,
                        (1 + u) % STAGES,
                        True,
                        True,
                        True,
                    )
                results = yield flat_state(accs, a_ops, b_ops, scales)
            accs, a_ops, b_ops, scales = unflat_state(results)
        # The last tile stores the output as its accumulators complete.
        for t in range_constexpr(STATIC_FROM, K_TILES):
            accs, a_ops, b_ops, scales = static_step(accs, a_ops, b_ops, scales, t)

    @flyc.jit
    def launch_gemm(
        A: fx.Tensor,
        B: fx.Tensor,
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
            B,
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
