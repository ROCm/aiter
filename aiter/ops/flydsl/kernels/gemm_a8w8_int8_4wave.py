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

# The LLVM dialect is used only for inline assembly where FlyDSL 0.5's stable
# surface and LLVM's intrinsics fall short: M0 ownership, signed-offset LDS DMA
# and the K64 INT8 MFMA.
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import Vector as Vec

from aiter.jit.utils.chip_info import get_cdna4_num_xcds
from aiter.ops.flydsl.gemm_a8w8_bpreshuffle_4wave import (
    make_4wave_kernel_name,
    validate_4wave_config,
)
from aiter.ops.flydsl.kernels.gemm_a8w8_8wave import S2RLoader, _xcd_swizzle_any
from aiter.ops.flydsl.kernels.gemm_hooks import (
    GemmHooks,
    make_host_steps,
    name_suffix,
    role_block_coords,
    validate_roles,
)
from aiter.ops.flydsl.kernels.kernels_common import (
    ceildiv,
    format_kernel_name,
    per_device_cache,
    set_slp_vectorize_once_used,
)
from aiter.ops.flydsl.kernels.mfma_preshuffle_pipeline import split_row_major_2d
from aiter.ops.flydsl.kernels.qk_prep import PrepSpec
from aiter.ops.flydsl.kernels.qk_prep_epilogue import (
    Int32Source,
    StoreCRows,
    mfma_drain,
)

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


def _a_lds_read_base(lane_id, wave_m, kk):
    """LDS byte offset for A tile zero; tile ``i`` adds ``i*A_BLOCK_STRIDE``."""
    row = lane_id % 16
    k_group = lane_id // 16
    return (
        wave_m * (WAVE_ROWS // 16) * A_BLOCK_STRIDE
        + row * BLOCK_K
        + ((kk * K_GROUPS + k_group) ^ _a_chunk_swizzle(row)) * 16
    )


def _b_lds_read_base(lane_id, wave_n, kk):
    """LDS byte offset for B tile zero; tile ``j`` adds ``j*16``."""
    row = lane_id % 16
    k_group = lane_id // 16
    return (
        (wave_n * (WAVE_ROWS // 16) + row // 2) * B_BLOCK_STRIDE
        + (kk * K_GROUPS + k_group) * 256
        + (row % 2) * 128
    )


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

    There is no stable layout object shared with the LDS read path because the
    LDS destination is implicit in M0 plus the instruction's signed immediate.
    The named strides and ``_a_chunk_swizzle`` are therefore the shared layout
    contract used by this address calculation and the read-base helpers above.
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
    LDS memory-operand offset, and with non-negative offsets it cannot tell
    the stages apart: it drains vmcnt before LDS reads that follow a load.
    LLVM does not count the asm loads in vmcnt, so the K loop waits for them
    explicitly, and the epilogue's scale loads, which persistent workgroups
    issue before the next block's, are asm as well.
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
        # M0 is written from an SGPR, so the wave-uniform LDS address is read
        # from lane 0. FlyDSL 0.5 exports no stable readfirstlane.
        self.m0 = [
            fx.rocdl.readfirstlane(
                fx.Int32.ir_type,
                (fx.Int32(fx.ptrtoint(region.ptr)) + wave_off).ir_value(),
            )
            for region in lds_pair
        ]

    def voffset(self, step, lane_offset):
        """VGPR offset of ``step`` for a lane at ``lane_offset`` from the base."""
        return lane_offset - self.imm[step] + self.CENTER

    def set_m0(self, buf):
        """Point M0 at LDS buffer ``buf`` for the ``ops`` that follow.

        FlyDSL 0.5 has no stable M0 setter. Remove this boundary when a setter
        with an explicit post-write hazard is exported.
        """
        _llvm.inline_asm(
            res=None,
            operands_=[fx.Int32(self.m0[buf]).ir_value()],
            asm_string="s_mov_b32 m0, $0\ns_nop 0",
            constraints="s",
            has_side_effects=True,
        )

    def _load(self, voffset, k_offset, step):
        # On this signed LDS memory-operand offset, FlyDSL 0.5's global_load_lds
        # reaches an LLVM assertion ("Negative MachineMemOperand offset"). Keep
        # the raw instruction until signed-offset global_load_lds is supported.
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


def _mfma_i8(a, b, c):
    """``c is None`` starts the accumulation from an inline zero.

    The accumulators are pinned to AGPRs like hipBLASLt's: with the LLVM
    intrinsic, the allocator keeps them in VGPRs and the operands in AGPRs,
    leaving too few VGPRs, so it spills and adds accumulator copies inside the
    loop. LLVM does not see this as an MFMA, so ``mfma_drain`` must separate
    the last one from any read of its result outside another MFMA. Remove both
    boundaries together once the intrinsic keeps the accumulators in AGPRs.
    """
    res_t = fx.Vector.make_type(4, fx.Int32)
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


@per_device_cache
def compile_int8_gemm_4w(
    *,
    K: int,
    xcd_swizzle: int = 8,
    num_cu: int = 0,
    num_xcds: int | None = None,
    bias: bool = False,
    prep: PrepSpec | None = None,
    amax: bool = False,
    roles: int = 1,
    role_cols: int = 0,
    hooks: GemmHooks | None = None,
    col_amax_offset: int | None = None,
):
    """Compile, once per configuration, the launcher of one 4-wave INT8 kernel.

    ``num_cu > 0`` makes the kernel persistent over that many workgroups.

    ``bias`` adds a per-column FP32 bias in the epilogue. The column operand then holds
    the N column scales followed by the N biases.

    ``prep``, a ``PrepSpec``, applies a Q/K prep in the epilogue; its ``bias`` replaces
    ``bias``. The column operand then also holds the spec's column vectors after the
    scales and biases (``PrepSpec.pack_columns``). The launcher takes the spec's FP32
    tables (``PrepSpec.pack_tables``) and writes FP32 ``[2, M, N // 128]``: the row-factor
    op's per-head partials, then per-head maxima of the stored output. With a row-factor
    op the caller scales each row by the op's factor of the sum of the row's partials. All
    three are flat: FlyDSL indexes a 2-D tensor argument column-major. N is a multiple of
    128, one head per 128 columns.

    ``amax`` (without ``prep``) writes the same ``[2, M, N // 128]`` buffer but only its
    maxima half, for a projection the consumer quantizes as it is, such as V.

    ``roles > 1`` (with ``prep``, ``hooks`` and ``num_cu``) runs a fused weight of ``roles``
    blocks of ``role_cols`` columns, Q/K/V for 3, and stores C planar,
    ``[roles, M, role_cols]``. The prep applies to every role but the last. ``hooks`` are
    ``GemmHooks``, code run inside the GEMM under the contract in ``gemm_hooks``, and
    ``aux`` starts with the block counters. ``col_amax_offset`` makes the last role's
    epilogue also write FP32 ``[ceil(M / 128), role_cols]`` per-column maxima of its stored
    output to ``aux`` plus that many bytes, one row per 128 rows of C; they land before
    the block is counted.

    A persistent workgroup loads the first two K tiles of its next output
    block during the last two of the current one, so the next block's K loop
    starts while the epilogue runs. That needs an even number of K tiles, so
    the next block's first tile lands in LDS buffer 0; otherwise each
    workgroup takes one block.

    The launcher takes ``(A, B_T, C, A_scale, Columns, c_m, c_n, stream)``; with a prep
    or ``amax`` also ``Tables`` and ``Partials`` after ``Columns`` (any tensor serves as
    tables without them), and with ``roles > 1`` also ``aux`` and the hooks' six Int64
    values after ``Partials``. The plain launcher takes no placeholders for them: every
    tensor argument adds host time to each op call.
    """
    if prep is not None and not isinstance(prep, PrepSpec):
        raise TypeError(f"prep must be a PrepSpec or None, got {prep!r}")
    if prep is not None and bias and not prep.bias:
        raise ValueError("bias=True conflicts with a PrepSpec without bias")
    if prep is not None and amax:
        raise ValueError("amax is for an output without a prep, which writes its own")
    if prep is not None:
        bias = prep.bias
    if num_xcds is None:
        num_xcds = get_cdna4_num_xcds() if xcd_swizzle > 0 else 8
    num_xcds = int(num_xcds)
    validate_4wave_config(
        K=K, xcd_swizzle=xcd_swizzle, num_cu=num_cu, num_xcds=num_xcds
    )
    validate_roles(
        roles=roles,
        role_cols=role_cols,
        block_n=BLOCK_N,
        hooks=hooks,
        prep=prep,
        num_cu=num_cu,
        num_xcds=num_xcds,
        xcd_swizzle=xcd_swizzle,
        col_amax_offset=col_amax_offset,
    )
    K_ITERS = K // BLOCK_K
    B_K_STEP = 16 * BLOCK_K  # bytes per K128 tile inside a preshuffled 16-row strip
    HALF = WAVE_TILES // 2
    persistent = num_cu > 0
    if roles > 1:
        after_block, end_role = make_host_steps(hooks)

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
    assert P3_VMCNT_TILE0 + StoreCRows.load_count(Int32Source, bias, prep) <= 63
    # Phase 1's last A-hi reads and phase 2's last B-hi reads must land
    # before the phase's barrier. Phase 2's A loads do not count against that
    # wait and run to the end of the phase.
    P1_READS_END = len(PHASE_MFMAS[0]) - 6
    P2_READS_END = len(PHASE_MFMAS[1]) - 6

    _kname = format_kernel_name(
        make_4wave_kernel_name(
            K=K, xcd_swizzle=xcd_swizzle, num_cu=num_cu, num_xcds=num_xcds
        )
        + ("_bias" if bias else "")
        + ("_prep" + prep.suffix if prep is not None else "")
        + ("_amax" if amax else "")
        + name_suffix(
            roles=roles,
            role_cols=role_cols,
            hooks=hooks,
            col_amax_offset=col_amax_offset,
        )
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
        Columns: fx.Tensor,
        Tables: fx.Tensor,
        Partials: fx.Tensor,
        c_m: fx.Int32,
        c_n: fx.Int32,
        aux: fx.Int64,
        h0: fx.Int64,
        h1: fx.Int64,
        h2: fx.Int64,
        h3: fx.Int64,
        h4: fx.Int64,
        h5: fx.Int64,
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

        def block_coords(block, role=0):
            if const_expr(roles > 1):
                return role_block_coords(
                    block,
                    role,
                    m_blocks=m_blocks,
                    role_cols=role_cols,
                    block_n=BLOCK_N,
                    xcd_swizzle=xcd_swizzle,
                    num_xcds=num_xcds,
                )
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

        def voffset_ops():
            """Return scheduled operations producing A/B LDS-DMA offsets.

            ``apply_scheduled_op`` passes the current (block_m, block_n)
            explicitly when each operation runs.
            """

            def op(g2s, preshuffled, step):
                def emit(coords):
                    row0, n_rows = (
                        (coords[1] * BLOCK_N, c_n)
                        if preshuffled
                        else (coords[0] * BLOCK_M, c_m)
                    )
                    lane_offset = _lane_offset(
                        lane_id, wave_id, K, preshuffled, row0, n_rows, step
                    )
                    return g2s.voffset(step, lane_offset)

                return ("voffset", int(preshuffled), emit)

            return [op(a_g2s, False, s) for s in range(LOADS_PER_REGION)] + [
                op(b_g2s, True, s) for s in range(LOADS_PER_REGION)
            ]

        # Operand (tile, kk): lane l reads k16 chunk kk * K_GROUPS + l // 16 of
        # A row l % 16 of the tile, or of B row 8 * (l % 16) + tile of the wave.
        # It sits at base[kk] + tile * A_BLOCK_STRIDE (A) or + tile * 16 (B).
        a_lane_base = [
            _a_lds_read_base(lane_id, wave_m, kk) for kk in range_constexpr(K_STEPS)
        ]
        b_lane_base = [
            _b_lds_read_base(lane_id, wave_n, kk) for kk in range_constexpr(K_STEPS)
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
                return s2r.load_one(a_lds[buf], a_lane_base[kk] + i * A_BLOCK_STRIDE)

            return ("a", i, kk, emit)

        def b_read(buf, j, kk):
            def emit():
                return s2r.load_one(b_lds[buf], b_lane_base[kk] + j * 16)

            return ("b", j, kk, emit)

        accs = [[None] * WAVE_TILES for _ in range(WAVE_TILES)]

        def apply_read(a_values, b_values, read):
            operand, tile, kk, emit = read
            value = emit()
            if operand == "a":
                a_values[tile][kk] = value
            else:
                b_values[tile][kk] = value
            return a_values, b_values

        def apply_scheduled_op(
            a_values,
            b_values,
            scale_values,
            next_coord_values,
            next_offset_values,
            op,
        ):
            if not isinstance(op, tuple):
                op()
                return (
                    a_values,
                    b_values,
                    scale_values,
                    next_coord_values,
                    next_offset_values,
                )
            kind = op[0]
            if kind in ("a", "b"):
                a_values, b_values = apply_read(a_values, b_values, op)
            elif kind == "scale":
                scale_values.append(op[1]())
            elif kind == "coords":
                next_coord_values = op[1]()
            elif kind == "effect":
                op[1]()
            else:
                _, operand, emit = op
                next_offset_values[operand].append(emit(next_coord_values))
            return (
                a_values,
                b_values,
                scale_values,
                next_coord_values,
                next_offset_values,
            )

        def do_mfma(acc_values, a_values, b_values, i, j, kk):
            acc_values[i][j] = _mfma_i8(
                a_values[i][kk], b_values[j][kk], acc_values[i][j]
            )
            return acc_values

        def spread(ops, first, last):
            """Place ``ops`` evenly after MFMAs ``first`` .. ``last - 1``."""
            n = len(ops)
            return [(first + q * (last - first) // n, op) for q, op in enumerate(ops)]

        def run_phase(
            acc_values,
            a_values,
            b_values,
            scale_values,
            next_coord_values,
            next_offset_values,
            mfmas,
            placed,
            wait=None,
        ):
            """Issue MFMAs with each memory op after its assigned MFMA.

            The four waves reach each phase together, so back-to-back LDS-DMA
            loads from all of them overrun the load path and are spread out.
            ``wait`` (vmcnt, lgkmcnt) goes before the last MFMA so the MFMA
            pipe stays busy while the following barrier resolves.
            ``sched_barrier(0)`` keeps LLVM from moving any op off its slot;
            FlyDSL 0.5 does not export it as a stable API.
            """
            for idx, (i, j, kk) in enumerate(mfmas):
                if const_expr(wait is not None and idx == len(mfmas) - 1):
                    fx.rocdl.s_waitcnt(vmcnt=wait[0], lgkmcnt=wait[1])
                    fx.rocdl.sched_barrier(0)
                acc_values = do_mfma(acc_values, a_values, b_values, i, j, kk)
                fx.rocdl.sched_barrier(0)
                for slot, op in placed:
                    if const_expr(slot == idx):
                        (
                            a_values,
                            b_values,
                            scale_values,
                            next_coord_values,
                            next_offset_values,
                        ) = apply_scheduled_op(
                            a_values,
                            b_values,
                            scale_values,
                            next_coord_values,
                            next_offset_values,
                            op,
                        )
                        fx.rocdl.sched_barrier(0)
            return (
                acc_values,
                a_values,
                b_values,
                scale_values,
                next_coord_values,
                next_offset_values,
            )

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

        def k_tile(
            acc_values,
            a_values,
            b_values,
            scale_values,
            next_coord_values,
            next_offset_values,
            buf,
            refill,
            has_next,
            p3_vmcnt=P3_VMCNT,
            p1_ops=(),
            p3_ops=(),
        ):
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
                p2_loads = [("effect", op) for op in p2_loads]
                p4_loads = [("effect", op) for op in p4_loads]
                p1_m0 = [
                    (
                        len(PHASE_MFMAS[0]) - 2,
                        ("effect", lambda: a_g2s.set_m0(buf)),
                    )
                ]
                p3_m0 = [
                    (
                        len(PHASE_MFMAS[2]) - 2,
                        ("effect", lambda: b_g2s.set_m0(buf)),
                    )
                ]

            p1_placed = spread(a_hi_reads, 0, P1_READS_END)
            p1_placed += spread(list(p1_ops), P1_READS_END, len(PHASE_MFMAS[0]) - 1)
            (
                acc_values,
                a_values,
                b_values,
                scale_values,
                next_coord_values,
                next_offset_values,
            ) = run_phase(
                acc_values,
                a_values,
                b_values,
                scale_values,
                next_coord_values,
                next_offset_values,
                PHASE_MFMAS[0],
                p1_placed + p1_m0,
                wait=(None, 0),
            )
            # The stable fx.barrier is gpu.barrier, which also makes prior
            # memory accesses visible and so would wait for the LDS-DMA loads
            # this schedule keeps in flight across barriers. The bare
            # s_barrier is not a stable FlyDSL 0.5 export.
            fx.rocdl.s_barrier()
            p2_placed = spread(b_hi_reads, 0, P2_READS_END)
            p2_placed += spread(p2_loads, 1, len(PHASE_MFMAS[1]) - 1)
            (
                acc_values,
                a_values,
                b_values,
                scale_values,
                next_coord_values,
                next_offset_values,
            ) = run_phase(
                acc_values,
                a_values,
                b_values,
                scale_values,
                next_coord_values,
                next_offset_values,
                PHASE_MFMAS[1],
                p2_placed,
                wait=(None, 0),
            )
            fx.rocdl.s_barrier()
            if const_expr(has_next):
                p3_placed = spread(list(p3_ops), 0, len(PHASE_MFMAS[2]) - 2)
                p3_wait = p3_vmcnt if refill is not None else 0
                (
                    acc_values,
                    a_values,
                    b_values,
                    scale_values,
                    next_coord_values,
                    next_offset_values,
                ) = run_phase(
                    acc_values,
                    a_values,
                    b_values,
                    scale_values,
                    next_coord_values,
                    next_offset_values,
                    PHASE_MFMAS[2],
                    p3_placed + p3_m0,
                    wait=(p3_wait, None),
                )
                fx.rocdl.s_barrier()
                p4_ops = interleave(p4_loads, lo_reads(1 - buf))
                (
                    acc_values,
                    a_values,
                    b_values,
                    scale_values,
                    next_coord_values,
                    next_offset_values,
                ) = run_phase(
                    acc_values,
                    a_values,
                    b_values,
                    scale_values,
                    next_coord_values,
                    next_offset_values,
                    PHASE_MFMAS[3],
                    spread(p4_ops, 0, len(PHASE_MFMAS[3]) - 2),
                )
            else:
                (
                    acc_values,
                    a_values,
                    b_values,
                    scale_values,
                    next_coord_values,
                    next_offset_values,
                ) = run_phase(
                    acc_values,
                    a_values,
                    b_values,
                    scale_values,
                    next_coord_values,
                    next_offset_values,
                    PHASE_MFMAS[2],
                    [],
                )
                (
                    acc_values,
                    a_values,
                    b_values,
                    scale_values,
                    next_coord_values,
                    next_offset_values,
                ) = run_phase(
                    acc_values,
                    a_values,
                    b_values,
                    scale_values,
                    next_coord_values,
                    next_offset_values,
                    PHASE_MFMAS[3],
                    [],
                )
            return (
                acc_values,
                a_values,
                b_values,
                scale_values,
                next_coord_values,
                next_offset_values,
            )

        def lo_state(a_values, b_values):
            return [a_values[i][kk] for i in range(HALF) for kk in ALL_K] + [
                b_values[j][kk] for j in range(HALF) for kk in ALL_K
            ]

        def unpack_lo_state(values, a_values, b_values):
            values = list(values)
            a_values = [list(row) for row in a_values]
            b_values = [list(row) for row in b_values]
            for i in range_constexpr(HALF):
                for kk in range_constexpr(K_STEPS):
                    a_values[i][kk] = values.pop(0)
            for j in range_constexpr(HALF):
                for kk in range_constexpr(K_STEPS):
                    b_values[j][kk] = values.pop(0)
            return a_values, b_values

        # Values live across a K tile boundary: the accumulators and the next
        # tile's lo operands.
        def live_state(acc_values, a_values, b_values):
            accs_flat = [
                acc_values[i][j] for i in range(WAVE_TILES) for j in range(WAVE_TILES)
            ]
            return accs_flat + lo_state(a_values, b_values)

        def unpack_state(values, a_values, b_values):
            values = list(values)
            acc_values = [[None] * WAVE_TILES for _ in range(WAVE_TILES)]
            for i in range_constexpr(WAVE_TILES):
                for j in range_constexpr(WAVE_TILES):
                    acc_values[i][j] = values.pop(0)
            a_values, b_values = unpack_lo_state(values, a_values, b_values)
            return acc_values, a_values, b_values

        store_c = StoreCRows(
            Int32Source(A_scale, c_m),
            Columns,
            C,
            c_m,
            c_n,
            bias=bias,
            prep=prep,
            tables=Tables,
            partials=Partials,
            amax=amax,
            planar_cols=role_cols if roles > 1 else 0,
        )

        for role in range_constexpr(roles):
            if const_expr(roles > 1):
                block_end = m_blocks * (role_cols // BLOCK_N)
            else:
                block_end = num_blocks
            # Prologue: the first block's K tiles 0 and 1 in flight, wait for
            # tile 0, read its lo operands. Tile 0's phase 3 wait in a persistent
            # workgroup counts on an epilogue before it, so the first block also
            # waits for tile 1 here.
            block0 = fx.Int32(fx.block_idx.x)
            coords = list(block_coords(block0, role))
            voffsets = ([], [])
            for _, operand, emit in voffset_ops():
                voffsets[operand].append(emit(coords))
            for t in range_constexpr(2):
                a_loads, b_loads = tile_loads(voffsets, t)
                for g2s, loads in ((a_g2s, a_loads), (b_g2s, b_loads)):
                    g2s.set_m0(t)
                    for op in loads:
                        op()
            fx.rocdl.s_waitcnt(vmcnt=0 if persistent else 2 * LOADS_PER_REGION)
            fx.rocdl.s_barrier()
            for read in lo_reads(0):
                a_ops, b_ops = apply_read(a_ops, b_ops, read)

            # Values live across an output block boundary: its coordinates and
            # offsets, and the lo operands of its first K tile.
            def block_state(coord_values, offset_values, a_values, b_values):
                return (
                    coord_values
                    + offset_values[0]
                    + offset_values[1]
                    + lo_state(a_values, b_values)
                )

            def unpack_block_state(values, a_values, b_values):
                values = list(values)
                coord_values = [values.pop(0) for _ in range(2)]
                offset_values = (
                    [values.pop(0) for _ in range(LOADS_PER_REGION)],
                    [values.pop(0) for _ in range(LOADS_PER_REGION)],
                )
                a_values, b_values = unpack_lo_state(values, a_values, b_values)
                return coord_values, offset_values, a_values, b_values

            def static_tile(
                acc_values,
                a_values,
                b_values,
                scale_values,
                next_coord_values,
                next_offset_values,
                t,
                next_voffsets,
                next_ops,
                scale_ops,
            ):
                """K tile ``t`` of a block, outside the runtime K loop.

                The last two refill from the next block at ``next_voffsets``;
                tile 0 spreads ``next_ops`` over phase 3 and tile K_ITERS - 2
                issues ``scale_ops``.
                """
                refill = None
                if const_expr(t + 2 < K_ITERS):
                    # The current block's offsets, read when the tile is traced.
                    refill = (voffsets, t + 2)  # noqa: B023
                elif const_expr(persistent):
                    refill = (next_voffsets, t + 2 - K_ITERS)
                p3_vmcnt = P3_VMCNT_TILE0 if t == 0 else P3_VMCNT
                p1_ops = []
                if const_expr(t == K_ITERS - 2):
                    p1_ops = scale_ops
                    p3_vmcnt += len(scale_ops)
                has_next = t + 1 < K_ITERS or persistent
                p3_ops = next_ops if t == 0 else ()
                return k_tile(
                    acc_values,
                    a_values,
                    b_values,
                    scale_values,
                    next_coord_values,
                    next_offset_values,
                    t % 2,
                    refill,
                    has_next,
                    p3_vmcnt,
                    p1_ops,
                    p3_ops,
                )

            n_pairs = max(0, (K_ITERS - 4) // 2)
            if const_expr(roles > 1):
                grid = fx.Int32(num_cu)
            else:
                grid = fx.Int32(fx.grid_dim.x)
            for block, bstate in range(
                block0,
                block_end,
                grid,
                init=block_state(coords, voffsets, a_ops, b_ops),
            ):
                coords, voffsets, a_ops, b_ops = unpack_block_state(
                    bstate, a_ops, b_ops
                )
                accs = [[None] * WAVE_TILES for _ in range(WAVE_TILES)]

                # The next block's coordinates and offsets, computed during tile
                # 0's phase 3 unless tile 0 already loads that block. The last
                # block reloads itself; nothing reads those loads.
                next_coords, next_voffsets, next_ops = [], ([], []), []
                scales = []
                if const_expr(persistent):
                    this_block = fx.Int32(block)

                    def next_coords_op(
                        this_block=this_block, grid=grid, block_end=block_end, role=role
                    ):
                        nxt = this_block + grid
                        nxt = (nxt < block_end).select(nxt, this_block)
                        return list(block_coords(nxt, role))

                    next_ops = [("coords", next_coords_op)] + voffset_ops()
                    if const_expr(K_ITERS == 2):
                        for op in next_ops:
                            (
                                a_ops,
                                b_ops,
                                scales,
                                next_coords,
                                next_voffsets,
                            ) = apply_scheduled_op(
                                a_ops,
                                b_ops,
                                scales,
                                next_coords,
                                next_voffsets,
                                op,
                            )
                        next_ops = []

                # The wave's tile of the block; WaveTile maps its lanes.
                tile = store_c.tile(
                    coords[0] * BLOCK_M + wave_m * WAVE_ROWS,
                    coords[1] * BLOCK_N + wave_n * WAVE_ROWS,
                )
                # In a roles kernel the last role applies no prep and loads none of its
                # column vectors.
                apply_prep = role < roles - 1 if roles > 1 else None
                scale_ops = [("scale", op) for op in store_c.load_ops(tile, apply_prep)]
                # Tiles 0 and 1 start the accumulators. A runtime loop then takes
                # the tiles that refill from this block, two per trip so the
                # buffer parity is static; a fully unrolled K loop overflows the
                # instruction cache.
                (
                    accs,
                    a_ops,
                    b_ops,
                    scales,
                    next_coords,
                    next_voffsets,
                ) = static_tile(
                    accs,
                    a_ops,
                    b_ops,
                    scales,
                    next_coords,
                    next_voffsets,
                    0,
                    next_voffsets,
                    next_ops,
                    scale_ops,
                )
                (
                    accs,
                    a_ops,
                    b_ops,
                    scales,
                    next_coords,
                    next_voffsets,
                ) = static_tile(
                    accs,
                    a_ops,
                    b_ops,
                    scales,
                    next_coords,
                    next_voffsets,
                    1,
                    next_voffsets,
                    next_ops,
                    scale_ops,
                )
                if const_expr(n_pairs > 0):
                    for pair, state in range(
                        0, n_pairs, 1, init=live_state(accs, a_ops, b_ops)
                    ):
                        accs, a_ops, b_ops = unpack_state(state, a_ops, b_ops)
                        t0 = fx.Int32(pair) * 2 + 2
                        (
                            accs,
                            a_ops,
                            b_ops,
                            scales,
                            next_coords,
                            next_voffsets,
                        ) = k_tile(
                            accs,
                            a_ops,
                            b_ops,
                            scales,
                            next_coords,
                            next_voffsets,
                            0,
                            (voffsets, t0 + 2),
                            True,
                        )
                        (
                            accs,
                            a_ops,
                            b_ops,
                            scales,
                            next_coords,
                            next_voffsets,
                        ) = k_tile(
                            accs,
                            a_ops,
                            b_ops,
                            scales,
                            next_coords,
                            next_voffsets,
                            1,
                            (voffsets, t0 + 3),
                            True,
                        )
                        results = yield live_state(accs, a_ops, b_ops)
                    accs, a_ops, b_ops = unpack_state(results, a_ops, b_ops)
                for t in range_constexpr(2 + 2 * n_pairs, K_ITERS):
                    (
                        accs,
                        a_ops,
                        b_ops,
                        scales,
                        next_coords,
                        next_voffsets,
                    ) = static_tile(
                        accs,
                        a_ops,
                        b_ops,
                        scales,
                        next_coords,
                        next_voffsets,
                        t,
                        next_voffsets,
                        next_ops,
                        scale_ops,
                    )

                if const_expr(roles > 1):
                    col_max = None
                    if const_expr(col_amax_offset is not None and role == roles - 1):
                        col_max = aux + col_amax_offset
                    store_c.store_tiles(
                        mfma_drain(accs), tile, scales, apply_prep, col_max=col_max
                    )
                    after_block(
                        role,
                        aux,
                        (h0, h1, h2, h3, h4, h5),
                        c_m,
                        tile.row_base,
                        tile.col_base,
                        block,
                        block0,
                    )
                else:
                    store_c.store_tiles(mfma_drain(accs), tile, scales)
                if const_expr(persistent):
                    coords = list(next_coords)
                    voffsets = (
                        list(next_voffsets[0]),
                        list(next_voffsets[1]),
                    )
                _ = yield block_state(coords, voffsets, a_ops, b_ops)

            if const_expr(persistent):
                # LDS-DMA must not land in LDS that a later workgroup owns.
                fx.rocdl.s_waitcnt(vmcnt=0)
            if const_expr(roles > 1):
                end_role(role, aux, (h0, h1, h2, h3, h4, h5), c_m)

    value_attrs = {
        "rocdl.waves_per_eu": 1,
        "rocdl.flat_work_group_size": f"{N_WAVES * 64},{N_WAVES * 64}",
    }

    def grid_size(c_m, c_n):
        grid_x = ceildiv(c_m, BLOCK_M) * ceildiv(c_n, BLOCK_N)
        if const_expr(persistent):
            grid_x = (grid_x > num_cu).select(fx.Int32(num_cu), grid_x)
        return grid_x

    @flyc.jit
    def launch_gemm(
        A: fx.Tensor,
        B_T: fx.Tensor,
        C: fx.Tensor,
        A_scale: fx.Tensor,
        Columns: fx.Tensor,
        c_m: fx.Int32,
        c_n: fx.Int32,
        stream: fx.Stream,
    ):
        grid_x = grid_size(c_m, c_n)
        # Without a prep or maxima the kernel reads neither tables nor partials.
        kernel_gemm(
            A,
            B_T,
            C,
            A_scale,
            Columns,
            A_scale,
            A_scale,
            c_m,
            c_n,
            fx.Int64(0),
            fx.Int64(0),
            fx.Int64(0),
            fx.Int64(0),
            fx.Int64(0),
            fx.Int64(0),
            fx.Int64(0),
            value_attrs=value_attrs,
        ).launch(grid=(grid_x, 1, 1), block=(N_WAVES * 64, 1, 1), stream=stream)

    @flyc.jit
    def launch_gemm_epilogue(
        A: fx.Tensor,
        B_T: fx.Tensor,
        C: fx.Tensor,
        A_scale: fx.Tensor,
        Columns: fx.Tensor,
        Tables: fx.Tensor,
        Partials: fx.Tensor,
        c_m: fx.Int32,
        c_n: fx.Int32,
        stream: fx.Stream,
    ):
        grid_x = grid_size(c_m, c_n)
        kernel_gemm(
            A,
            B_T,
            C,
            A_scale,
            Columns,
            Tables,
            Partials,
            c_m,
            c_n,
            fx.Int64(0),
            fx.Int64(0),
            fx.Int64(0),
            fx.Int64(0),
            fx.Int64(0),
            fx.Int64(0),
            fx.Int64(0),
            value_attrs=value_attrs,
        ).launch(grid=(grid_x, 1, 1), block=(N_WAVES * 64, 1, 1), stream=stream)

    @flyc.jit
    def launch_gemm_roles(
        A: fx.Tensor,
        B_T: fx.Tensor,
        C: fx.Tensor,
        A_scale: fx.Tensor,
        Columns: fx.Tensor,
        Tables: fx.Tensor,
        Partials: fx.Tensor,
        aux: fx.Int64,
        h0: fx.Int64,
        h1: fx.Int64,
        h2: fx.Int64,
        h3: fx.Int64,
        h4: fx.Int64,
        h5: fx.Int64,
        c_m: fx.Int32,
        c_n: fx.Int32,
        stream: fx.Stream,
    ):
        kernel_gemm(
            A,
            B_T,
            C,
            A_scale,
            Columns,
            Tables,
            Partials,
            c_m,
            c_n,
            aux,
            h0,
            h1,
            h2,
            h3,
            h4,
            h5,
            value_attrs=value_attrs,
        ).launch(grid=(num_cu, 1, 1), block=(N_WAVES * 64, 1, 1), stream=stream)

    if roles > 1:
        launcher = launch_gemm_roles
    elif prep is not None or amax:
        launcher = launch_gemm_epilogue
    else:
        launcher = launch_gemm
    return set_slp_vectorize_once_used(launcher, False)
