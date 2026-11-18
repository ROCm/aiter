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

K loop: three LDS stages, each one K128 tile of A and of B. K tile ``t + 3`` (its
scales and its LDS-DMA loads) is loaded into tile ``t``'s stage during tile ``t``,
and tile ``t + 1``'s operands are read into VGPRs, each one after its register's
last MFMA of tile ``t``. One barrier per tile: before it each wave waits for tile
``t + 1``'s loads and for its own reads of tile ``t``'s stage. The last three
tiles load nothing, so the packed layout's guard tiles are never read.

The weights are the MFMA's first operand, so lane ``l`` of output tile (i, j)
holds four adjacent columns ``16 j + 4 (l // 16) + r`` of row ``16 i + l % 16``.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx

# The LLVM dialect is used only for inline assembly at instructions FlyDSL 0.5
# cannot represent through its stable surface: M0 ownership, signed-offset LDS DMA,
# the scale loads LLVM must not count, the FP6 scaled MFMA into AGPRs, and the
# accumulator hazard boundary.
from flydsl._mlir.dialects import llvm as _llvm
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
STAGE_BYTES = 2 * TILE_BYTES
# A tile's VMEM ops: two scale loads, then the A and B LDS-DMA loads.
TILE_VMEM = 2 + 2 * LOADS_PER_REGION
CFG_E2M3 = 2  # cbsz / blgp: FP6 E2M3 for both operands


class G2SGlobal:
    """``global_load_lds`` of a wave's share of one operand tile, on one M0 value.

    The instruction offset applies to the global and the LDS address and is
    signed 13-bit, so with M0 in the middle of the wave's 6 KB it reaches all six
    loads. The loads are inline asm because LLVM asserts on the intrinsic's
    negative LDS offset, so LLVM does not count them in vmcnt and the K loop waits
    for them explicitly. LLVM does not see M0 in use either, so nothing else in the
    kernel may use it.
    """

    CENTER = 3072

    def __init__(self, tensor, wave_id, region_ptrs, tile_row_offset, lane_id):
        self.imm = [q * 1024 - self.CENTER for q in range(LOADS_PER_REGION)]
        assert all(-4096 <= i < 4096 for i in self.imm)
        self.base = fx.Int64(fx.ptrtoint(fx.get_iter(tensor)))
        # The tile is copied as it is, so the global and the LDS address share one
        # offset; both carry +CENTER against the signed immediate.
        wave_off = wave_id * WAVE_REGION_BYTES + self.CENTER
        # M0 is written from an SGPR, so the wave-uniform LDS address is read from
        # lane 0.
        self.m0 = [
            fx.rocdl.readfirstlane(
                fx.Int32.ir_type, (fx.Int32(fx.ptrtoint(ptr)) + wave_off).ir_value()
            )
            for ptr in region_ptrs
        ]
        self.voffset = tile_row_offset + wave_off + lane_id * 16

    def set_m0(self, stage):
        # FlyDSL 0.5 has no stable M0 setter with the write's hazard nop.
        _llvm.inline_asm(
            res=None,
            operands_=[fx.Int32(self.m0[stage]).ir_value()],
            asm_string="s_mov_b32 m0, $0\ns_nop 0",
            constraints="s",
            has_side_effects=True,
        )

    def load(self, k_offset, q):
        # FlyDSL 0.5 cannot encode this signed LDS memory-operand offset; its
        # intrinsic path reaches an LLVM assertion.
        _llvm.inline_asm(
            res=None,
            operands_=[
                fx.Int32(self.voffset).ir_value(),
                (self.base + fx.Int64(k_offset)).ir_value(),
            ],
            asm_string=f"global_load_lds_dwordx4 $0, $1 offset:{self.imm[q]}",
            constraints="v,s",
            has_side_effects=True,
        )


def _load_scales(base, voffset, k_offset):
    """The eight scale bytes of the lane's row in its wave's eight row blocks.

    Inline asm so that LLVM's vmcnt waits do not also wait for the LDS-DMA loads
    issued after it, which it cannot count; the K loop waits for it explicitly.
    """
    return _llvm.inline_asm(
        res=fx.Vector.make_type(2, fx.Int32),
        operands_=[fx.Int32(voffset).ir_value(), (base + fx.Int64(k_offset)).ir_value()],
        asm_string="global_load_dwordx2 $0, $1, $2",
        constraints="=v,v,s",
        has_side_effects=True,
    )


def _lds_load(ptr, offset, nbytes):
    at = fx.add_offset(ptr, fx.make_int_tuple(offset))
    view = fx.make_view(fx.recast_iter(fx.Uint8, at), fx.make_layout(nbytes, 1))
    return Vec(view.load()).bitcast(fx.Int32)


def _read_operand(ptr, row_block, lane_id):
    """Lane ``lane_id``'s 32 FP6 values of row block ``row_block``: six dwords."""
    c0 = _lds_load(ptr, row_block * 1024 + lane_id * 16, 16)
    c1 = _lds_load(ptr, C1_OFFSET + row_block * 512 + lane_id * 8, 8)
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


def _mfma_drain(accs):
    """Return ``accs`` readable after the XDL-write -> v_accvgpr_read wait states.

    LLVM does not see the asm MFMA as one, so it would read an accumulator right
    after its last MFMA; each passes through an empty asm placed after the nops.
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


# MFMA order of a K tile: B row blocks in pairs, each pair over the eight A blocks.
# B pair g is dead after its 16 MFMAs, A block i after the last pair's MFMA on it.
ORDER = [
    (j, i)
    for g in range(WAVE_TILES // 2)
    for i in range(WAVE_TILES)
    for j in (2 * g, 2 * g + 1)
]
LAST_USE_B = {j: max(k for k, (jj, _) in enumerate(ORDER) if jj == j) for j in range(8)}
LAST_USE_A = {i: max(k for k, (_, ii) in enumerate(ORDER) if ii == i) for i in range(8)}
# The barrier precedes B pair 0's last MFMA; the next tile's reads follow it.
BARRIER_AT = LAST_USE_B[1]


def compile_a6w6_gemm_4w(*, K: int, xcd_swizzle: int = 8, num_xcds: int = 8):
    """``K`` is the padded contraction dimension, a multiple of 128 of at least 384."""
    assert K % BLOCK_K == 0 and K >= STAGES * BLOCK_K
    K_TILES = K // BLOCK_K
    NK_PAD = K_TILES + PADK
    # Tiles 0 .. LOADING - 1 load the tile three ahead.
    LOADING = K_TILES - STAGES
    # Tile 0 starts the accumulators; the next loading tiles run in trips of three,
    # so stages are static; the rest are static.
    TRIPS = max(0, (LOADING - 1) // STAGES)
    STATIC_FROM = 1 + TRIPS * STAGES

    _kname = f"flydsl_a6w6_4w_{BLOCK_M}x{BLOCK_N}x{BLOCK_K}_1x{xcd_swizzle}_k{K}"

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
        a_ptrs = stage_ptrs
        b_ptrs = [fx.add_offset(p, fx.make_int_tuple(TILE_BYTES)) for p in stage_ptrs]

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
        block_m = fx.Int32(block_m)
        block_n = fx.Int32(block_n)

        a_g2s = G2SGlobal(A, wave_id, a_ptrs, block_m * (NK_PAD * TILE_BYTES), lane_id)
        b_g2s = G2SGlobal(B, wave_id, b_ptrs, block_n * (NK_PAD * TILE_BYTES), lane_id)
        sa_base = fx.Int64(fx.ptrtoint(fx.get_iter(A_scale)))
        sb_base = fx.Int64(fx.ptrtoint(fx.get_iter(B_scale)))
        sa_voff = (
            block_m * (NK_PAD * SCALE_TILE_BYTES) + wave_m * 512 + lane_id * 8
        )
        sb_voff = (
            block_n * (NK_PAD * SCALE_TILE_BYTES) + wave_n * 512 + lane_id * 8
        )

        def tile_vmem(t, stage):
            """Issue tile ``t``'s scale loads; return them and its LDS-DMA ops."""
            scales = (
                _load_scales(sa_base, sa_voff, t * SCALE_TILE_BYTES),
                _load_scales(sb_base, sb_voff, t * SCALE_TILE_BYTES),
            )
            ops = []
            for g2s in (a_g2s, b_g2s):
                ops.append(lambda g2s=g2s: g2s.set_m0(stage))
                for q in range_constexpr(LOADS_PER_REGION):
                    ops.append(lambda g2s=g2s, q=q: g2s.load(t * TILE_BYTES, q))
            return scales, ops

        def read_a(stage, i):
            return _read_operand(a_ptrs[stage], wave_m * WAVE_TILES + i, lane_id)

        def read_b(stage, j):
            return _read_operand(b_ptrs[stage], wave_n * WAVE_TILES + j, lane_id)

        def k_tile(accs, a_ops, b_ops, scales, stage, has_next, vmem_wait, prefetch):
            """One K tile in LDS ``stage``, with ``scales`` = (A, B) words.

            ``prefetch`` (tile, stage) issues that tile's VMEM ops after the barrier;
            ``vmem_wait`` is the vmcnt the barrier waits for; ``has_next`` reads the
            next tile's operands from the next stage.
            """
            next_stage = (stage + 1) % STAGES
            new_a = [None] * WAVE_TILES
            new_b = [None] * WAVE_TILES
            sa = (Vec(scales[0])[0], Vec(scales[0])[1])
            sb = (Vec(scales[1])[0], Vec(scales[1])[1])
            fetched, vmem = None, []
            for idx, (j, i) in enumerate(ORDER):
                if const_expr(idx == BARRIER_AT):
                    fx.rocdl.s_waitcnt(vmcnt=vmem_wait, lgkmcnt=0)
                    fx.rocdl.sched_barrier(0)
                    fx.rocdl.s_barrier()
                    fx.rocdl.sched_barrier(0)
                    if const_expr(prefetch is not None):
                        fetched, vmem = tile_vmem(*prefetch)
                        fx.rocdl.sched_barrier(0)
                accs[i][j] = _mfma(
                    b_ops[j], a_ops[i], sb[j // 4], sa[i // 4], j, i, accs[i][j]
                )
                fx.rocdl.sched_barrier(0)
                if const_expr(idx < BARRIER_AT):
                    continue
                # Spread the LDS-DMA loads over the MFMAs after the barrier.
                if const_expr(len(vmem) > 0 and idx % 2 == 1):
                    vmem.pop(0)()
                    fx.rocdl.sched_barrier(0)
                if const_expr(has_next):
                    # Each read follows its register's last MFMA, and the barrier.
                    for jj in range_constexpr(WAVE_TILES):
                        if const_expr(new_b[jj] is None and LAST_USE_B[jj] <= idx):
                            new_b[jj] = read_b(next_stage, jj)
                            fx.rocdl.sched_barrier(0)
                    for ii in range_constexpr(WAVE_TILES):
                        if const_expr(new_a[ii] is None and LAST_USE_A[ii] <= idx):
                            new_a[ii] = read_a(next_stage, ii)
                            fx.rocdl.sched_barrier(0)
            while const_expr(len(vmem) > 0):
                vmem.pop(0)()
                fx.rocdl.sched_barrier(0)
            return accs, new_a, new_b, fetched

        def step(accs, a_ops, b_ops, scale_queue, t, stage, loading, has_next, two_after):
            """Tile ``t``; ``loading`` loads tile ``t + 3`` into its stage.

            Before the barrier, tile ``t + 1``'s VMEM ops must have landed; tile
            ``t + 2``'s, issued after them, may still run if there is one.
            """
            prefetch = (t + STAGES, stage) if loading else None
            accs, a_ops, b_ops, fetched = k_tile(
                accs,
                a_ops,
                b_ops,
                scale_queue[0],
                stage,
                has_next,
                TILE_VMEM if two_after else 0,
                prefetch,
            )
            scale_queue = scale_queue[1:] + ([fetched] if loading else [])
            return accs, a_ops, b_ops, scale_queue

        def flat_state(accs, a_ops, b_ops, scale_queue):
            return (
                [accs[i][j] for i in range(WAVE_TILES) for j in range(WAVE_TILES)]
                + list(a_ops)
                + list(b_ops)
                + [s for pair in scale_queue for s in pair]
            )

        def unflat_state(values):
            values = list(values)
            accs = [
                [values.pop(0) for _ in range(WAVE_TILES)] for _ in range(WAVE_TILES)
            ]
            a_ops = [values.pop(0) for _ in range(WAVE_TILES)]
            b_ops = [values.pop(0) for _ in range(WAVE_TILES)]
            scale_queue = [(values.pop(0), values.pop(0)) for _ in range(STAGES)]
            return accs, a_ops, b_ops, scale_queue

        # Prologue: tiles 0, 1 and 2 in flight; wait for tile 0 and read it.
        scale_queue = []
        for t in range_constexpr(STAGES):
            scales, ops = tile_vmem(t, t)
            scale_queue.append(scales)
            for op in ops:
                op()
        fx.rocdl.s_waitcnt(vmcnt=(STAGES - 1) * TILE_VMEM)
        fx.rocdl.s_barrier()
        a_ops = [read_a(0, i) for i in range_constexpr(WAVE_TILES)]
        b_ops = [read_b(0, j) for j in range_constexpr(WAVE_TILES)]
        accs = [[None] * WAVE_TILES for _ in range(WAVE_TILES)]

        def static_step(accs, a_ops, b_ops, scale_queue, t):
            return step(
                accs,
                a_ops,
                b_ops,
                scale_queue,
                t,
                t % STAGES,
                t < LOADING,
                t + 1 < K_TILES,
                t + 2 < K_TILES,
            )

        accs, a_ops, b_ops, scale_queue = static_step(
            accs, a_ops, b_ops, scale_queue, 0
        )
        if const_expr(TRIPS > 0):
            for trip, state in range(
                0, TRIPS, 1, init=flat_state(accs, a_ops, b_ops, scale_queue)
            ):
                accs, a_ops, b_ops, scale_queue = unflat_state(state)
                t0 = fx.Int32(trip) * STAGES + 1
                for u in range_constexpr(STAGES):
                    accs, a_ops, b_ops, scale_queue = step(
                        accs,
                        a_ops,
                        b_ops,
                        scale_queue,
                        t0 + u,
                        (1 + u) % STAGES,
                        True,
                        True,
                        True,
                    )
                results = yield flat_state(accs, a_ops, b_ops, scale_queue)
            accs, a_ops, b_ops, scale_queue = unflat_state(results)
        for t in range_constexpr(STATIC_FROM, K_TILES):
            accs, a_ops, b_ops, scale_queue = static_step(
                accs, a_ops, b_ops, scale_queue, t
            )

        # Epilogue: lane l of tile (i, j) holds row 16 i + l % 16 and the four
        # columns 16 j + 4 (l // 16) + r of the wave's tile.
        accs = _mfma_drain(accs)
        c_tensor = fx.rocdl.make_buffer_tensor(
            C, max_size=False, num_records_bytes=fx.Int64(c_m) * fx.Int64(c_n) * 2
        )
        c_div = fx.logical_divide(c_tensor, fx.make_layout(1, 1))
        out_atom = fx.make_copy_atom(fx.rocdl.BufferCopy64b(), fx.BFloat16)
        reg = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.BFloat16)
        row0 = block_m * BLOCK_M + wave_m * WAVE_ROWS + lane_id % 16
        col0 = block_n * BLOCK_N + wave_n * WAVE_ROWS + (lane_id // 16) * 4
        for i in range_constexpr(WAVE_TILES):
            for j in range_constexpr(WAVE_TILES):
                fx.memref_store_vec(Vec(accs[i][j]).to(fx.BFloat16), reg)
                index = (row0 + i * TILE) * c_n + col0 + j * TILE
                fx.copy(out_atom, reg, fx.slice(c_div, (None, fx.Int32(index))))

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
