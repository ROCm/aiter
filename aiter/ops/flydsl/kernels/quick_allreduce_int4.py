# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""gfx942/gfx950 TP∈{2,4,8} INT4 two-shot all-reduce.

INT4 nibble: [-8,+7], −1/8, 4 B/thread, 1152 B rank-tile. Scale is
group-16 signed E4M3 in the 128 B region. Super-tile ST∈{1,8}; host
uses ST=1 when ``num_tiles ≤`` the occupancy-clamped persistent grid.
Payload HBM is bf16; in-kernel math is packed fp16. Each rank owns
``atoms / world_size`` atoms of a tile (8 GPUs → 1, 4 → 2, 2 → 4); LDS
stays ``atoms * 1152``.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr import arith, const_expr, gpu, range_constexpr, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.typing import Int32, Int64, ReductionOp, Stream, T, as_ir_value

from . import buffer_ops
from .quant_utils import emit_mx_e8m0_scale

WORLD = 8
SUPPORTED_WORLDS = (2, 4, 8)
SUPER_TILES = (1, 8)
DEFAULT_GRID_CAP = 304 * 4

# Shared by host tile math, LDS PackStorage, and the kernel factory.
BLOCK = 256
ATOMS = 8
WAVE = 64
GROUP = 8
TILE_BYTES = BLOCK * ATOMS * 16
RANK_TILE_BYTES = 1152
PACK_I32 = ATOMS * (RANK_TILE_BYTES // 4)

# (world_size, super_tile) → VGPR-limited workgroups per CU.
_RESIDENT_WGS_PER_CU = {
    (2, 1): 3,
    (2, 8): 4,
    (4, 1): 4,
    (4, 8): 5,
    (8, 1): 4,
    (8, 8): 6,
}


def clamp_grid_cap(
    requested: int,
    *,
    arch: str,
    world_size: int,
    super_tile: int,
    cu_count: int,
) -> int:
    if requested < 1 or cu_count < 1:
        raise ValueError("grid_cap and cu_count must be positive")
    if arch not in ("gfx942", "gfx950"):
        raise ValueError(
            f"quick_allreduce_int4 has no residency measurement for {arch!r}"
        )
    try:
        resident = _RESIDENT_WGS_PER_CU[(int(world_size), int(super_tile))]
    except KeyError:
        raise ValueError(
            "quick_allreduce_int4 has no residency measurement for "
            f"{(world_size, super_tile)}"
        ) from None
    return min(int(requested), resident * int(cu_count))


def _f16x2(packed):
    return fx.Vector.from_elements([packed], fx.Int32).bitcast(fx.Float16)


def _i32(vec):
    return vec.bitcast(fx.Int32)[0]


def _splat_f16x2(x):
    return fx.Vector.filled(2, x, fx.Float16)


def _clamp_fp16_overflow():
    """Saturate packed fp16 overflow to ±65504 instead of Inf.

    Packed add/mul/FMA follow MODE bit 23 (FP16_OVFL). Unset, overflow
    becomes Inf and every later FMA in that tile is Inf. Set, it saturates
    to the max finite fp16. INT4 is already a saturating codec, so a rare
    overflow should not poison the all-reduce.

    FlyDSL has no MODE helper; ``llvm.amdgcn.s.setreg`` is the same
    intrinsic ``rocdl.disable_xdl_arb_stall`` uses for a different bit.
    """
    # hwreg(HW_REG_MODE, offset=23, size=2): id | (off<<6) | ((size-1)<<11)
    imm = as_ir_value(fx.Int32(0xDC1))
    val = as_ir_value(fx.Int32(1))
    llvm.call_intrinsic(None, "llvm.amdgcn.s.setreg", [imm, val], [], [])


def _shuffle_f16x2(vec, xor_off):
    return _f16x2(fx.Int32(gpu.shuffle_xor(_i32(vec), xor_off, WAVE)))


def _pair_signed_ext_f16(atom):
    """Signed extremum of 16 fp16 (this thread's 8 + xor-1 neighbor)."""
    p0, p1, p2, p3 = (
        _f16x2(atom[0]),
        _f16x2(atom[1]),
        _f16x2(atom[2]),
        _f16x2(atom[3]),
    )
    wmax = fx.maxnumf(fx.maxnumf(p0, p1), fx.maxnumf(p2, p3))
    wmin = fx.min(fx.min(p0, p1), fx.min(p2, p3))
    wmax = fx.maxnumf(wmax, _shuffle_f16x2(wmax, 1))
    wmin = fx.min(wmin, _shuffle_f16x2(wmin, 1))
    pk = (abs(wmax) > abs(wmin)).select(wmax, wmin)
    lo, hi = pk[0], pk[1]
    return fx.Float32((abs(lo) > abs(hi)).select(lo, hi))


def _atom_bf16_to_f16(atom):
    return fx.Vector(atom).bitcast(fx.BFloat16).to(fx.Float16).bitcast(fx.Int32)


def _atom_f16_to_bf16(atom):
    return fx.Vector(atom).bitcast(fx.Float16).to(fx.BFloat16).bitcast(fx.Int32)


def _f32_to_e4m3(x):
    """Pack f32 to a signed E4M3 byte: 1 sign, 4-bit exp (bias 7), 3-bit mantissa.

    Decode is ``±(1 + m/8) * 2**(e - 7)`` for every ``e``, including 0
    (no OCP denorms). ``0x7F`` / ``0xFF`` are max finite, not NaN.

    ``0x00`` is +0 only. Any other value that would land there (positive
    underflow, or exactly ``2**-7``) is stored as ``0x01``. Negative
    underflow keeps ``0x80`` (``-2**-7``). Overflow clamps both exponent
    and mantissa; saturating the exponent alone would map ``512`` to
    ``256``.
    """
    is_z = x == fx.Float32(0.0)
    sign = (x < fx.Float32(0.0)).select(fx.Int32(0x80), fx.Int32(0))
    bits = abs(x).bitcast(fx.Int32)
    e = (bits.shrui(fx.Int32(23)) & fx.Int32(255)) - fx.Int32(127)
    mant = bits & fx.Int32(0x7FFFFF)
    m3 = (mant + fx.Int32(1 << 19)).shrui(fx.Int32(20))
    carry = m3 == fx.Int32(8)
    e = e + carry.select(fx.Int32(1), fx.Int32(0))
    m3 = carry.select(fx.Int32(0), m3)
    e4 = e + fx.Int32(7)
    overflow = e4 > fx.Int32(15)
    m3 = overflow.select(fx.Int32(7), m3)
    e4 = (e4 < fx.Int32(0)).select(fx.Int32(0), overflow.select(fx.Int32(15), e4))
    byte = sign | (e4 << fx.Int32(3)) | (m3 & fx.Int32(7))
    return is_z.select(fx.Int32(0), (byte == fx.Int32(0)).select(fx.Int32(1), byte))


def _e4m3_to_f32(b):
    is_z = b == fx.Int32(0)
    sign = (b & fx.Int32(0x80)) != fx.Int32(0)
    e4 = b.shrui(fx.Int32(3)) & fx.Int32(15)
    m3 = b & fx.Int32(7)
    mag_bits = ((e4 + fx.Int32(120)) << fx.Int32(23)) | (m3 << fx.Int32(20))
    mag = mag_bits.bitcast(fx.Float32)
    signed = sign.select(-mag, mag)
    return is_z.select(fx.Float32(0.0), signed)


def _pack_e4m3_word(e, lane):
    """Four pair-E4M3 bytes into the i32 scale slot (lanes 0,2,4,6 of GROUP)."""
    base = (lane // GROUP) * GROUP
    e0 = fx.Int32(gpu.shuffle_idx(e, base, WAVE))
    e1 = fx.Int32(gpu.shuffle_idx(e, base + fx.Int32(2), WAVE))
    e2 = fx.Int32(gpu.shuffle_idx(e, base + fx.Int32(4), WAVE))
    e3 = fx.Int32(gpu.shuffle_idx(e, base + fx.Int32(6), WAVE))
    b = fx.Int32(0xFF)
    return (
        (e0 & b)
        | ((e1 & b) << fx.Int32(8))
        | ((e2 & b) << fx.Int32(16))
        | ((e3 & b) << fx.Int32(24))
    )


def _e4m3_decoding_scale(e):
    return _splat_f16x2(_e4m3_to_f32(e) * fx.Float32(-0.125))


def _quant_atom_fp16(atom, enc_pk):
    q = []
    lo = _splat_f16x2(fx.Float16(-8.0))
    hi = _splat_f16x2(fx.Float16(7.0))
    bias = fx.Vector.filled(2, fx.Int16(8), fx.Int16)
    for i in range_constexpr(4):
        w = fx.min(fx.maxnumf(_f16x2(atom[i]) * enc_pk, lo), hi)
        q.append(_i32(fx.roundeven(w).to(fx.Int16) + bias))
    return q[0] | (q[1] << fx.Int32(4)) | (q[2] << fx.Int32(8)) | (q[3] << fx.Int32(12))


def _codec_quant(atom, lane, tid):
    ext = _pair_signed_ext_f16(atom)
    e = _f32_to_e4m3(ext)
    d = _e4m3_to_f32(e) * fx.Float32(-0.125)
    packed = _quant_atom_fp16(
        atom, _splat_f16x2(fx.Float32(1.0) / (d + fx.Float32(1e-7)))
    )
    is_leader = (tid % GROUP) == 0
    return packed, _pack_e4m3_word(e, lane), is_leader


def _codec_dequant(packed, scale, acc=None):
    """Unpack four INT4 nibbles to f16x2, scale, optionally FMA into *acc*.

    ``a * b + c`` does not contract to ``v_pk_fma_f16``; ``fx.fma`` does.
    Two fp16 lanes are independent channels, not a dot into f32.
    """
    out = []
    # nibble | 0x6400 then + (-1032.0) f16x2 reconstructs (q-8).
    mask = fx.Int32(0x000F000F)
    bias_hi = fx.Int32(0x64006400)
    bias_lo = _f16x2(fx.Int32(0xE408E408))
    for i in range_constexpr(4):
        q4 = (packed.shrui(fx.Int32(i * 4)) & mask) | bias_hi
        dq = _f16x2(q4) + bias_lo
        if acc is None:
            out.append(_i32(dq * scale))
        else:
            out.append(_i32(fx.fma(dq, scale, _f16x2(acc[i]))))
    return fx.Vector.from_elements(out, fx.Int32)


def _i32_to_bytes(i32_off):
    return fx.Int64(i32_off) * fx.Int64(4)


def _store_v4i32_nt_global(addr_i64, data):
    """NT-store 16 B through a per-lane global address.

    One instruction here sends 16 B to a different GPU in each lane of a
    4-wide group. A buffer-descriptor store wants the descriptor in scalar
    registers, so LLVM would serialize those lanes (one destination at a
    time). A flat global store takes the address from a vector register,
    so all destinations issue together. ``nontemporal`` skips L2; this is
    payload, not a flag. ``fx.ptr_store`` has no NT flag.
    """
    ptr_ty = ir.Type.parse("!llvm.ptr<1>")
    ptr = llvm.IntToPtrOp(ptr_ty, as_ir_value(addr_i64)).result
    llvm.StoreOp(as_ir_value(data), ptr, alignment=16, nontemporal=True)


def _load_i32_nt(rsrc, elem_off):
    return fx.Int32(
        buffer_ops.buffer_load(
            rsrc, elem_off, vec_width=1, dtype=T.i32, cache_modifier=4  # NT
        )
    )


def _load_i32_uncached(rsrc):
    val = buffer_ops.buffer_load(
        rsrc, 0, vec_width=1, dtype=T.i32, cache_modifier=2  # sc1, bypass L2
    )
    rocdl.s_waitcnt(vmcnt=0)
    return fx.Int32(val)


def _invalidate_l1():
    # gfx942/gfx950 cannot select llvm.amdgcn.buffer.wbinvl1.sc.
    llvm.InlineAsmOp(None, [], "buffer_inv sc1", "", has_side_effects=True)


@fx.struct
class PackStorage:
    pack: fx.Array[fx.Int32, PACK_I32, 16]


# One fp32 partial sum of squares per (wave, row) for the fused RMSNorm.
NORM_RED_F32 = (BLOCK // WAVE) * ATOMS


@fx.struct
class PackNormStorage:
    pack: fx.Array[fx.Int32, PACK_I32, 16]
    red: fx.Array[fx.Float32, NORM_RED_F32, 16]


NORM_LATENCY_WGS_PER_CU = 2


def latency_max_tiles(cu_count: int) -> int:
    """Largest launch the ``norm_latency`` build keeps fully resident."""
    return NORM_LATENCY_WGS_PER_CU * int(cu_count)


def rmsnorm_hidden_supported(hidden: int) -> bool:
    """Fused RMSNorm needs each 4 KiB atom inside one bf16 row of a 32 KiB tile."""
    row_bytes = int(hidden) * 2
    atom_bytes = BLOCK * 16
    return row_bytes % atom_bytes == 0 and TILE_BYTES % row_bytes == 0


MXFP4_GROUP = 32


def mxfp4_scale_shape(rows: int, hidden: int) -> tuple[int, int]:
    """Shuffled e8m0 scale buffer of ``per_1x32_f4_quant_hip(shuffle=True)``."""
    return (int(rows) + 255) // 256 * 256, (int(hidden) // MXFP4_GROUP + 7) // 8 * 8


def make_quick_allreduce_int4_kernel(
    *,
    world_size: int = WORLD,
    super_tile: int = 1,
    grid: int,
    norm_hidden: int | None = None,
    norm_eps: float = 1e-6,
    gemma_norm: bool = False,
    norm_latency: bool = False,
    norm_mxfp4: bool = False,
):
    """Build the INT4 all-reduce kernel.

    With ``norm_hidden`` set, the all-gather epilogue fuses
    ``residual_out = ar + residual`` and
    ``out = rmsnorm(residual_out) * weight`` (``1 + weight`` with
    ``gemma_norm``), the contract of vLLM's ``fused_add_rms_norm``.

    ``norm_mxfp4`` additionally quantizes the bf16 ``out`` to MXFP4 the way
    ``per_1x32_f4_quant_hip(out, shuffle=True)`` does: packed fp4x2
    ``(rows, hidden // 2)`` and shuffled e8m0 scales of
    ``mxfp4_scale_shape(rows, hidden)``, the layout ``gemm_a4w4`` reads.

    ``norm_latency`` (``super_tile=1`` only) trades occupancy for latency:
    two workgroups per CU instead of the persistent-grid residency, so the
    host may only launch it with ``num_tiles <= latency_max_tiles(cu)``.
    """
    fuse_norm = norm_hidden is not None
    if norm_latency and (not fuse_norm or super_tile != 1):
        raise ValueError("norm_latency needs norm_hidden and super_tile=1")
    if norm_mxfp4 and not fuse_norm:
        raise ValueError("norm_mxfp4 needs norm_hidden")
    NORM_LATENCY = bool(norm_latency)
    NORM_MXFP4 = bool(norm_mxfp4)
    if fuse_norm and not rmsnorm_hidden_supported(norm_hidden):
        raise ValueError(
            f"fused RMSNorm supports hidden in (2048, 4096, 8192, 16384), "
            f"got {norm_hidden}"
        )
    if world_size not in SUPPORTED_WORLDS:
        raise ValueError(
            f"world_size must be one of {SUPPORTED_WORLDS}, got {world_size}"
        )
    if ATOMS % world_size != 0:
        raise ValueError(f"ATOMS={ATOMS} is not divisible by world_size={world_size}")
    if super_tile not in SUPER_TILES:
        raise ValueError(f"super_tile must be one of {SUPER_TILES}, got {super_tile!r}")
    if grid < 1:
        raise ValueError(f"grid must be positive, got {grid}")
    PHASES = 2
    PHASE_REDUCE_SCATTER = 0
    PHASE_ALL_GATHER = 1
    RANK_TILE_I32 = RANK_TILE_BYTES // 4
    SCALE_I32_OFF = 256
    PAIR = 2
    WAVES = BLOCK // WAVE
    QUAD_LANES = 4
    QUADS_PER_WAVE = WAVE // QUAD_LANES
    N_SECTORS = RANK_TILE_BYTES // 64
    TILE_I32 = TILE_BYTES // 4
    TILE_FP16 = TILE_BYTES // 2
    LDS_BYTES = ATOMS * RANK_TILE_BYTES
    # Each rank owns this many 16-byte atoms of a 32 KiB tile
    # (8 GPUs → 1, 4 → 2, 2 → 4). LDS still holds all ATOMS atoms.
    rank_atoms = ATOMS // world_size
    # Last-sector pad is ST * rank_atoms * RANK_TILE_I32 after the ST tiles.
    rank_payload_i32 = rank_atoms * RANK_TILE_I32
    release_i32_off = super_tile * rank_payload_i32
    wire_tile_i32 = release_i32_off + 16
    wire_tile_bytes = wire_tile_i32 * 4

    # flags_i32 is also the i32 offset of the wire area, so the flag prefix has
    # to be a whole number of 64 B sectors (16 i32s). At a smaller multiple every
    # rank-tile and release sector straddles two hardware sectors, so the 64 B
    # fanout stores and the last-sector release stop being one sector wide.
    grid_multiple = 16 // (PHASES * world_size)
    if grid % grid_multiple != 0:
        raise ValueError(
            f"grid must be a multiple of {grid_multiple} at "
            f"world_size={world_size} to keep the wire area 64 B aligned, got "
            f"{grid}"
        )
    flags_i32 = PHASES * grid * world_size
    if fuse_norm:
        ATOMS_PER_ROW = (int(norm_hidden) * 2) // (BLOCK * 16)
        ROWS_PER_TILE = ATOMS // ATOMS_PER_ROW
        SCALE_N = mxfp4_scale_shape(1, norm_hidden)[1]
        INV_HIDDEN = 1.0 / float(norm_hidden)
        EPS = float(norm_eps)
        # Atoms whose loads issue together. LLVM cannot hoist a load above
        # the previous atom's store (in place, residual is residual_out), so
        # this is the epilogue's memory-level parallelism; larger groups
        # spill under the pinned residency budget.
        NORM_GROUP = (
            ATOMS if NORM_LATENCY else 2 if (world_size, super_tile) == (8, 1) else 1
        )
        storage_cls = PackNormStorage
    else:
        storage_cls = PackStorage

    @flyc.kernel(known_block_size=[BLOCK, 1, 1])
    def quick_allreduce_int4(
        rank: Int32,
        nbytes: Int64,
        num_tiles: Int32,
        inp_ptr: Int64,
        out_ptr: Int64,
        peer_ptrs: Int64,
        colors_ptr: Int64,
        residual_ptr: Int64,
        residual_out_ptr: Int64,
        weight_ptr: Int64,
        mxfp4_ptr: Int64,
        mxfp4_scale_ptr: Int64,
    ):
        _clamp_fp16_overflow()
        tid = fx.Int32(gpu.thread_id("x"))
        bid = fx.Int32(gpu.block_id("x"))

        thread_layout = fx.make_layout((WAVES, WAVE), (WAVE, 1))
        wave, lane = fx.idx2crd(tid, thread_layout).unpack()
        quad_layout = fx.make_layout((QUADS_PER_WAVE, QUAD_LANES), (QUAD_LANES, 1))
        quad, lane_in_quad = fx.idx2crd(lane, quad_layout).unpack()
        quad_id = wave * fx.Int32(QUADS_PER_WAVE) + quad

        pack_layout = fx.make_layout((ATOMS, RANK_TILE_I32), (RANK_TILE_I32, 1))
        # 64 B NT sectors of one 1152 B rank-tile: (sector, lane-in-quad)
        # -> i32 start of the dwordx4. Isolated NT store stays explicit.
        nt_own_layout = fx.make_layout((N_SECTORS, QUAD_LANES), (16, 4))
        # Remote NT fanout stays a vector-addressed nontemporal store.
        hbm_layout = fx.make_layout(
            (num_tiles, ATOMS, BLOCK * 4),
            (TILE_I32, BLOCK * 4, 1),
        )
        hbm_row_layout = fx.make_layout((1, BLOCK * 4), (BLOCK * 4, 1))
        hbm_copy_atom = fx.make_copy_atom(rocdl.BufferCopy128b(), fx.Int32)
        hbm_copy = fx.make_tiled_copy_tv(
            hbm_copy_atom,
            fx.make_layout((1, BLOCK), (1, 1)),
            fx.make_layout((1, 4), (1, 1)),
        ).get_slice(tid)
        # Four group-16 E4M3 bytes share the i32 slot eight threads already own.
        scale_own_layout = fx.make_layout(
            (BLOCK // GROUP, GROUP // PAIR, PAIR), (GROUP, PAIR, 1)
        )
        scale_slot, pair_in_slot, _lane_in_pair = fx.idx2crd(
            tid, scale_own_layout
        ).unpack()
        # Rank-tile = 18 × 64 B Infinity Fabric sectors: 16 INT4 then 2 E4M3.
        # A workgroup has 64 quads. Lockstep: consecutive quads target
        # consecutive peers of one sector so one NT store hits every GPU.
        # A stripe of 8 sectors needs world_size*8 quads (64 at 8 GPUs);
        # leftover quads sit idle (always on the 2-sector scale tail, and
        # on the INT4 stripes when world_size < 8). Cover 18 as 8+8+2.
        fanout_int4_stripe = fx.make_layout((world_size, 8), (1, world_size))
        fanout_scale_stripe = fx.make_layout((world_size, 2), (1, world_size))
        color_layout = fx.make_layout((grid,), (1,))
        wire_slot_layout = fx.make_layout(
            (PHASES, grid, world_size, super_tile),
            (
                grid * world_size * wire_tile_i32,
                world_size * wire_tile_i32,
                wire_tile_i32,
                rank_payload_i32,
            ),
        )

        lds = fx.SharedAllocator().allocate(storage_cls).peek()
        pack = lds.pack.view(pack_layout)
        smem_ptr = lds.pack.ptr

        peer_rsrc = buffer_ops.create_buffer_resource_from_addr(peer_ptrs)
        peers = [
            buffer_ops.buffer_load(peer_rsrc, i, vec_width=1, dtype=T.i64)
            for i in range(world_size)
        ]
        peer_vec = fx.Vector.from_elements(peers, dtype=fx.Int64)
        self_rsrc = buffer_ops.create_buffer_resource_from_addr(
            fx.Int64(rocdl.readfirstlane(T.i64, as_ir_value(peer_vec[rank])))
        )
        # inp/out are a 3-D i32 tensor consumed by TiledCopy (BufferCopy128b).
        # That API needs a FlyDSL buffer-backed tensor (layout + descriptor),
        # not a raw descriptor. create_buffer_resource_from_addr is the
        # scalar-offset buffer_load/store path used for the peer-pointer
        # table, IPC inbox, and color flags. num_records_bytes is the live
        # tensor size so a partial last tile is out-of-range safe.
        hbm_i32_ptr = fx.PointerType.get(
            T.i32, address_space=fx.AddressSpace.Global, alignment=16
        )

        def _payload_tensor(ptr):
            view = fx.make_view(fx.inttoptr(hbm_i32_ptr, ptr), hbm_layout)
            return rocdl.make_buffer_tensor(
                view, max_size=False, num_records_bytes=nbytes
            )

        in_buf = _payload_tensor(inp_ptr)
        out_buf = _payload_tensor(out_ptr)
        if const_expr(fuse_norm):
            res_rsrc = buffer_ops.create_buffer_resource_from_addr(
                residual_ptr, num_records_bytes=nbytes
            )
            res_out_rsrc = buffer_ops.create_buffer_resource_from_addr(
                residual_out_ptr, num_records_bytes=nbytes
            )
            out_rsrc = buffer_ops.create_buffer_resource_from_addr(
                out_ptr, num_records_bytes=nbytes
            )
            weight_rsrc = buffer_ops.create_buffer_resource_from_addr(
                weight_ptr, num_records_bytes=int(norm_hidden) * 2
            )
            red_ptr = lds.red.ptr
            lane_i32 = tid * fx.Int32(4)
        if const_expr(NORM_MXFP4):
            mxfp4_rsrc = buffer_ops.create_buffer_resource_from_addr(
                mxfp4_ptr, num_records_bytes=nbytes // fx.Int64(4)
            )
            n_rows = nbytes // fx.Int64(int(norm_hidden) * 2)
            mxfp4_scale_rsrc = buffer_ops.create_buffer_resource_from_addr(
                mxfp4_scale_ptr,
                num_records_bytes=(n_rows + fx.Int64(255))
                // fx.Int64(256)
                * fx.Int64(256 * SCALE_N),
            )
            # Lane part of the shuffled scale index for group ``tid // 4`` of
            # an atom (see mx_scale_shuffle_idx); the atom's group base is a
            # multiple of 8, so the row and atom parts fold into soffset.
            g = tid // fx.Int32(MXFP4_GROUP // 8)
            scale_lane = (
                (g // fx.Int32(8)) * fx.Int32(256)
                + (g % fx.Int32(4)) * fx.Int32(64)
                + ((g % fx.Int32(8)) // fx.Int32(4)) * fx.Int32(2)
            )
        color_rsrc = buffer_ops.create_buffer_resource_from_addr(colors_ptr)

        def _pack_off(peer, i32_idx):
            return fx.get_scalar(fx.crd2idx((peer, i32_idx), pack_layout))

        def _sub_tile_i32(phase, src, sub):
            slot = fx.get_scalar(
                fx.crd2idx((fx.Int32(phase), bid, src, sub), wire_slot_layout)
            )
            return fx.Int32(flags_i32) + slot

        def _hbm_atom_row(buf, tile, atom):
            return fx.make_view(
                fx.get_iter(fx.slice(buf, (tile, atom, None))),
                hbm_row_layout,
            )

        def _load_color():
            off = fx.get_scalar(fx.crd2idx((bid,), color_layout))
            return fx.Int32(
                buffer_ops.buffer_load(color_rsrc, off, vec_width=1, dtype=T.i32)
            )

        def _store_color(color):
            off = fx.get_scalar(fx.crd2idx((bid,), color_layout))
            buffer_ops.buffer_store(color, color_rsrc, off)

        def _load_tile_atoms(tile):
            atoms = []
            for atom in range_constexpr(ATOMS):
                src = hbm_copy.partition_S(_hbm_atom_row(in_buf, tile, atom))
                frag = fx.make_fragment_like(src)
                fx.copy(hbm_copy_atom, src, frag)
                atoms.append(_atom_bf16_to_f16(fx.Vector(frag.load())))
            return atoms

        def _store_tile_atoms(tile, atoms):
            for atom in range_constexpr(ATOMS):
                packed = _atom_f16_to_bf16(atoms[atom])
                dst = hbm_copy.partition_D(_hbm_atom_row(out_buf, tile, atom))
                frag = fx.make_fragment_like(dst)
                frag.store(packed)
                fx.copy(hbm_copy_atom, frag, dst)

        def _atom_soff(tile, atom):
            return tile * fx.Int32(TILE_BYTES) + fx.Int32(atom * BLOCK * 16)

        def _load_atom_bf16(rsrc, soff):
            return fx.Vector(
                buffer_ops.buffer_load(
                    rsrc,
                    lane_i32,
                    vec_width=4,
                    dtype=T.i32,
                    mask=fx.Int64(soff) < nbytes,
                    soffset_bytes=soff,
                )
            )

        def _store_atom_bf16(rsrc, soff, packed):
            buffer_ops.buffer_store(
                packed,
                rsrc,
                lane_i32,
                mask=fx.Int64(soff) < nbytes,
                soffset_bytes=soff,
            )

        def _store_mxfp4_atom(tile, atom, yb):
            """MXFP4-quantize one atom of bf16 ``out``.

            The 32-element group spans four consecutive lanes. The round-up
            e8m0 scale and the hardware scaled convert match
            ``per_1x32_f4_quant_hip`` bit for bit.
            """
            amax = abs(yb.to(fx.Float32)).reduce(ReductionOp.MAX)
            for off in range_constexpr(2):
                amax = fx.maxnumf(amax, amax.shuffle_xor(1 << off, WAVE))
            e8m0 = fx.Int32(emit_mx_e8m0_scale(amax))
            scale = as_ir_value((e8m0 << fx.Int32(23)).bitcast(fx.Float32))
            words = yb.bitcast(fx.Int32)
            pk = as_ir_value(fx.Int32(0))
            for j in range_constexpr(4):
                pk = rocdl.cvt_scalef32_pk_fp4_bf16(
                    T.i32,
                    pk,
                    as_ir_value(
                        fx.Vector.from_elements([words[j]], fx.Int32).bitcast(
                            fx.BFloat16
                        )
                    ),
                    scale,
                    j,
                )
            live = fx.Int64(_atom_soff(tile, atom)) < nbytes
            buffer_ops.buffer_store(
                fx.Int32(pk),
                mxfp4_rsrc,
                tid,
                mask=live,
                soffset_bytes=tile * fx.Int32(TILE_BYTES // 4)
                + fx.Int32(atom * BLOCK * 4),
            )
            # The group's four lanes store the same byte.
            x = tile * fx.Int32(ROWS_PER_TILE) + fx.Int32(atom // ATOMS_PER_ROW)
            scale_row = (
                (x // fx.Int32(32)) * fx.Int32(32 * SCALE_N)
                + (x % fx.Int32(16)) * fx.Int32(4)
                + (x % fx.Int32(32)) // fx.Int32(16)
            )
            groups_per_atom = BLOCK * 8 // MXFP4_GROUP
            buffer_ops.buffer_store(
                arith.trunci(T.i8, as_ir_value(e8m0)),
                mxfp4_scale_rsrc,
                scale_lane,
                mask=live,
                soffset_bytes=scale_row
                + fx.Int32((atom % ATOMS_PER_ROW) * groups_per_atom // 8 * 256),
            )

        def _prefetch_residual(tile):
            res = []
            for atom in range_constexpr(ATOMS):
                res.append(_load_atom_bf16(res_rsrc, _atom_soff(tile, atom)))
            return res

        def _norm_store_tile_atoms(tile, sub, res_pref):
            """Fused add + RMSNorm epilogue over one reduced 32 KiB tile.

            Atom *a* holds row ``a // ATOMS_PER_ROW`` of the tile, so the row
            of every register is a compile-time constant. A row is a whole
            number of atoms, so in the partial last tile each atom is either
            live or past the payload; the uniform mask drops the dead ones.

            The atom's byte offset goes in ``soffset`` (SGPR) and every lane
            shares one ``tid * 16`` VGPR offset. Per-atom VGPR offsets would
            be loop-invariant, so LLVM hoists them and keeps them live through
            the reduce-scatter and all-gather phases.

            Like vLLM ``fused_add_rms_norm``, the sum of squares uses the fp32
            sum and the output normalizes the bf16 ``residual_out``.

            The second pass re-reads ``residual_out`` instead of keeping the
            tile in registers across the reduction barrier: each lane reads
            back only the 16 B it stored, so program order makes it visible.

            Each atom's all-gather packet is received inside the first pass.
            Receiving the whole tile first, as the plain store does, keeps
            all eight dequantized atoms live under the residual loads. Each
            row's partial is reduced and parked in LDS as soon as the row is
            complete, and its rstd is formed when the second pass reaches it,
            so one row's state is live at a time (8 rows per tile at 2048).

            The ``norm_latency`` build instead takes the residual prefetched
            before the reduce-scatter (*res_pref*) and keeps the tile in
            registers between the passes.
            """
            sq = fx.Float32(0.0)
            xbs = []
            for g in range_constexpr(0, ATOMS, NORM_GROUP):
                pkts = []
                ress = []
                for atom in range_constexpr(g, g + NORM_GROUP):
                    pkts.append(
                        _recv_quantized(
                            PHASE_ALL_GATHER,
                            fx.Int32(atom // rank_atoms),
                            sub,
                            atom % rank_atoms,
                        )
                    )
                    if const_expr(NORM_LATENCY):
                        ress.append(res_pref[atom])
                    else:
                        ress.append(_load_atom_bf16(res_rsrc, _atom_soff(tile, atom)))
                for i in range_constexpr(NORM_GROUP):
                    atom = g + i
                    packed, scale = pkts[i]
                    ar = (
                        fx.Vector(_codec_dequant(packed, scale))
                        .bitcast(fx.Float16)
                        .to(fx.Float32)
                    )
                    x = ar + ress[i].bitcast(fx.BFloat16).to(fx.Float32)
                    xb = x.to(fx.BFloat16).bitcast(fx.Int32)
                    _store_atom_bf16(res_out_rsrc, _atom_soff(tile, atom), xb)
                    if const_expr(NORM_LATENCY):
                        xbs.append(xb)
                    sq = sq + (x * x).reduce(ReductionOp.ADD)
                    if const_expr((atom + 1) % ATOMS_PER_ROW == 0):
                        row = atom // ATOMS_PER_ROW
                        for off in range_constexpr(6):
                            sq = sq + sq.shuffle_xor(1 << (5 - off), WAVE)
                        if lane == 0:
                            fx.ptr_store(
                                sq,
                                red_ptr + (wave * fx.Int32(ATOMS) + fx.Int32(row)),
                            )
                        sq = fx.Float32(0.0)
            gpu.barrier()

            for g in range_constexpr(0, ATOMS, NORM_GROUP):
                xs = []
                ws = []
                for atom in range_constexpr(g, g + NORM_GROUP):
                    if const_expr(NORM_LATENCY):
                        xs.append(xbs[atom])
                    else:
                        xs.append(_load_atom_bf16(res_out_rsrc, _atom_soff(tile, atom)))
                    # bf16 column (atom % ATOMS_PER_ROW) * 2048 + tid * 8.
                    ws.append(
                        fx.Vector(
                            buffer_ops.buffer_load(
                                weight_rsrc,
                                lane_i32,
                                vec_width=4,
                                dtype=T.i32,
                                soffset_bytes=(atom % ATOMS_PER_ROW) * BLOCK * 16,
                            )
                        )
                    )
                for i in range_constexpr(NORM_GROUP):
                    atom = g + i
                    row = atom // ATOMS_PER_ROW
                    if const_expr(atom % ATOMS_PER_ROW == 0):
                        total = fx.Float32(fx.ptr_load(red_ptr + row))
                        for w in range_constexpr(1, WAVES):
                            total = total + fx.Float32(
                                fx.ptr_load(red_ptr + (w * ATOMS + row))
                            )
                        rstd = fmath.rsqrt(
                            total * fx.Float32(INV_HIDDEN) + fx.Float32(EPS)
                        )
                    x = xs[i].bitcast(fx.BFloat16).to(fx.Float32)
                    w = ws[i].bitcast(fx.BFloat16).to(fx.Float32)
                    if const_expr(gemma_norm):
                        w = w + fx.Vector.filled(8, fx.Float32(1.0), fx.Float32)
                    y = x * w * fx.Vector.filled(8, rstd, fx.Float32)
                    yb = y.to(fx.BFloat16)
                    _store_atom_bf16(
                        out_rsrc, _atom_soff(tile, atom), yb.bitcast(fx.Int32)
                    )
                    if const_expr(NORM_MXFP4):
                        _store_mxfp4_atom(tile, atom, yb)
            # The next tile's partials reuse ``red``.
            gpu.barrier()

        def _epilogue(tile, sub, res_pref=None):
            if const_expr(fuse_norm):
                _norm_store_tile_atoms(tile, sub, res_pref)
            else:
                _store_tile_atoms(tile, _recv_all_gather(sub))

        def _lds_write_packet(slot, packed, scale, is_leader):
            fx.memref_store(packed, pack, (slot, tid))
            if is_leader:
                fx.memref_store(
                    scale, pack, (slot, fx.Int32(SCALE_I32_OFF) + scale_slot)
                )

        def _pack_reduce_scatter(atoms):
            """Quantize each destination's slice of this tile into LDS.

            A 32 KiB tile is 8 atoms; destination *d* owns
            ``atoms[d * rank_atoms : (d+1) * rank_atoms]``. Those packets
            are later NT-stored into *d*'s reduce-scatter inbox.
            """
            for dest in range_constexpr(world_size):
                for k in range_constexpr(rank_atoms):
                    packed, scale, is_leader = _codec_quant(
                        atoms[dest * rank_atoms + k], lane, tid
                    )
                    _lds_write_packet(
                        fx.Int32(dest * rank_atoms + k), packed, scale, is_leader
                    )

        def _pack_all_gather(accs):
            """Quantize the reduced slice and replicate it for every peer.

            After reduce-scatter this rank holds ``rank_atoms`` reduced
            atoms. Copy the same packets into every destination slot so the
            NT fanout can push them into every peer's all-gather inbox.
            """
            for k in range_constexpr(rank_atoms):
                packed, scale, is_leader = _codec_quant(accs[k], lane, tid)
                for dest in range_constexpr(world_size):
                    _lds_write_packet(
                        fx.Int32(dest * rank_atoms + k), packed, scale, is_leader
                    )

        def _fanout_nt(phase, inbox_src, sub):
            """NT-store one rank-tile from LDS to every peer's inbox.

            Three lockstep stripes cover the 18 sectors: INT4 [0, 8), INT4
            [8, 16), E4M3 [16, 18). ``stripe * 8`` is the first sector of
            each stripe (16 for the scale tail).
            """
            for k in range_constexpr(rank_atoms):
                for stripe in range_constexpr(3):
                    is_scale_tail = stripe == 2
                    n_sectors = 2 if is_scale_tail else 8
                    fanout = (
                        fanout_scale_stripe if is_scale_tail else fanout_int4_stripe
                    )
                    n_quads = fx.Int32(world_size * n_sectors)
                    safe = (quad_id < n_quads).select(quad_id, fx.Int32(0))
                    peer, sector_in_stripe = fx.idx2crd(safe, fanout).unpack()
                    sector = fx.Int32(stripe * 8) + sector_in_stripe
                    if quad_id < n_quads:
                        vec_idx = fx.get_scalar(
                            fx.crd2idx((sector, lane_in_quad), nt_own_layout)
                        )
                        pack_peer = peer
                        wire_idx = vec_idx
                        if rank_atoms != 1:
                            pack_peer = peer * fx.Int32(rank_atoms) + fx.Int32(k)
                            wire_idx = vec_idx + fx.Int32(k * RANK_TILE_I32)
                        # 4xi32 NT vector cannot go through the i32 pack view.
                        v4 = fx.ptr_load(
                            smem_ptr + _pack_off(pack_peer, vec_idx),
                            result_type=fx.Vector.make_type(4, fx.Int32),
                        )
                        dest = peer_vec[peer]
                        byte_off = _i32_to_bytes(
                            _sub_tile_i32(phase, inbox_src, sub) + wire_idx
                        )
                        _store_v4i32_nt_global(dest + byte_off, v4)

        def _publish(phase, inbox_src, color):
            """Drain payload NT stores, then write *color* into every peer inbox.

            Last 64 B of this rank's slot (after the ST rank-tiles) is the
            handshake: 16 i32s all equal to *color*. Peers spin on that
            sector in their copy of our slot; seeing *color* means our
            payload is visible.

            ``vmcnt(0)``: this 64-lane wave's NT payload stores are done.
            The workgroup barrier: the other three 64-lane waves issued
            payload too; ``vmcnt`` is per-wave, so without the join a
            wave-0 handshake could race stores still in flight. Neither
            can move after the color store, and neither can be dropped.
            """
            rocdl.s_waitcnt(vmcnt=0)
            gpu.barrier()
            limit = fx.Int32(world_size)
            safe = (quad_id < limit).select(quad_id, fx.Int32(0))
            if quad_id < limit:
                vec_idx = fx.Int32(release_i32_off) + lane_in_quad * fx.Int32(4)
                v4 = fx.Vector.from_elements([color, color, color, color], fx.Int32)
                dest = peer_vec[safe]
                byte_off = _i32_to_bytes(
                    _sub_tile_i32(phase, inbox_src, fx.Int32(0)) + vec_idx
                )
                _store_v4i32_nt_global(dest + byte_off, v4)

        def _wait_flag(flag_rsrc, color):
            current = _load_i32_uncached(flag_rsrc)
            while current != color:
                current = _load_i32_uncached(flag_rsrc)
                _invalidate_l1()

        def _wait_release(phase, color):
            if tid < world_size:
                elem = _sub_tile_i32(phase, tid, fx.Int32(0)) + fx.Int32(
                    release_i32_off
                )
                _wait_flag(
                    buffer_ops.create_buffer_resource_from_addr(
                        peer_vec[rank] + _i32_to_bytes(elem)
                    ),
                    color,
                )
            gpu.barrier()

        def _recv_quantized(phase, src, sub, k=0):
            # Packed dword is at base+tid; scale dword is 1024 B later at a
            # group slot. They are not adjacent, so they cannot share one
            # vector load.
            base = _sub_tile_i32(phase, src, sub)
            if k:
                base = base + fx.Int32(k * RANK_TILE_I32)
            packed = _load_i32_nt(self_rsrc, base + tid)
            word = _load_i32_nt(self_rsrc, base + fx.Int32(SCALE_I32_OFF) + scale_slot)
            e = word.shrui(pair_in_slot * fx.Int32(8)) & fx.Int32(0xFF)
            return packed, _e4m3_decoding_scale(e)

        def _reduce_scattered(sub):
            """Dequant-accumulate every peer's reduce-scatter packet for *sub*."""
            accs = [None] * rank_atoms
            for src in range_constexpr(world_size):
                for k in range_constexpr(rank_atoms):
                    packed, scale = _recv_quantized(
                        PHASE_REDUCE_SCATTER, fx.Int32(src), sub, k
                    )
                    if accs[k] is None:
                        accs[k] = _codec_dequant(packed, scale)
                    else:
                        accs[k] = _codec_dequant(packed, scale, accs[k])
            return accs

        def _recv_all_gather(sub):
            """Dequantize every peer's all-gather packet back into full-tile atoms."""
            gathered = []
            for src in range_constexpr(world_size):
                for k in range_constexpr(rank_atoms):
                    packed, scale = _recv_quantized(
                        PHASE_ALL_GATHER, fx.Int32(src), sub, k
                    )
                    gathered.append(_codec_dequant(packed, scale))
            return gathered

        n_block_tiles = (num_tiles - bid + fx.Int32(grid - 1)) // fx.Int32(grid)
        color = _load_color()
        if super_tile == 1:
            for i in range(fx.Int32(0), n_block_tiles, fx.Int32(1)):
                tile = bid + i * fx.Int32(grid)
                res_pref = None
                if const_expr(NORM_LATENCY):
                    res_pref = _prefetch_residual(tile)
                atoms = _load_tile_atoms(tile)
                _pack_reduce_scatter(atoms)
                gpu.barrier()
                _fanout_nt(PHASE_REDUCE_SCATTER, rank, fx.Int32(0))
                _publish(PHASE_REDUCE_SCATTER, rank, color)

                _wait_release(PHASE_REDUCE_SCATTER, color)
                acc = _reduce_scattered(fx.Int32(0))

                _pack_all_gather(acc)
                gpu.barrier()
                _fanout_nt(PHASE_ALL_GATHER, rank, fx.Int32(0))
                _publish(PHASE_ALL_GATHER, rank, color)

                _wait_release(PHASE_ALL_GATHER, color)
                _epilogue(tile, fx.Int32(0), res_pref)

                color = color + fx.Int32(1)
                if color == fx.Int32(0):  # 0 is unset sentinel
                    color = fx.Int32(1)
        else:
            st_i = fx.Int32(super_tile)
            for i in range(fx.Int32(0), n_block_tiles, st_i):
                remain = n_block_tiles - i
                n_this = (remain < st_i).select(remain, st_i)

                for s in range(fx.Int32(0), n_this, fx.Int32(1)):
                    tile = bid + (i + s) * fx.Int32(grid)
                    atoms = _load_tile_atoms(tile)
                    _pack_reduce_scatter(atoms)
                    gpu.barrier()
                    _fanout_nt(PHASE_REDUCE_SCATTER, rank, s)
                    if (s + fx.Int32(1)) < n_this:
                        # Drain this wave's LDS loads, then join the WG.
                        # world_size<8 leaves waves idle in fanout; without the
                        # barrier they pack the next sub-tile into LDS while
                        # a busy wave still ptr_loads it. lgkmcnt only: NT
                        # payload stays in flight until _publish.
                        rocdl.s_waitcnt(lgkmcnt=0)
                        gpu.barrier()

                _publish(PHASE_REDUCE_SCATTER, rank, color)
                _wait_release(PHASE_REDUCE_SCATTER, color)

                for s in range(fx.Int32(0), n_this, fx.Int32(1)):
                    acc = _reduce_scattered(s)
                    _pack_all_gather(acc)
                    gpu.barrier()
                    _fanout_nt(PHASE_ALL_GATHER, rank, s)
                    if (s + fx.Int32(1)) < n_this:
                        rocdl.s_waitcnt(lgkmcnt=0)
                        gpu.barrier()

                _publish(PHASE_ALL_GATHER, rank, color)
                _wait_release(PHASE_ALL_GATHER, color)

                for s in range(fx.Int32(0), n_this, fx.Int32(1)):
                    tile = bid + (i + s) * fx.Int32(grid)
                    _epilogue(tile, s)

                color = color + fx.Int32(1)
                if color == fx.Int32(0):  # 0 is unset sentinel
                    color = fx.Int32(1)
        if tid == 0:
            _store_color(color)
        gpu.barrier()

    flat_wg = f"{BLOCK},{BLOCK}"
    value_attrs = {"rocdl.flat_work_group_size": flat_wg}
    if NORM_LATENCY:
        value_attrs["rocdl.waves_per_eu"] = NORM_LATENCY_WGS_PER_CU
    elif fuse_norm:
        # The host sizes the persistent grid from _RESIDENT_WGS_PER_CU, which
        # was measured on the plain kernel. One wave per SIMD per workgroup,
        # so pin the fused epilogue to the same VGPR budget.
        value_attrs["rocdl.waves_per_eu"] = _RESIDENT_WGS_PER_CU[
            (world_size, super_tile)
        ]

    @flyc.jit
    def launch_quick_allreduce_int4(
        rank: Int32,
        nbytes: Int64,
        num_tiles: Int32,
        inp_ptr: Int64,
        out_ptr: Int64,
        peer_ptrs: Int64,
        colors_ptr: Int64,
        residual_ptr: Int64,
        residual_out_ptr: Int64,
        weight_ptr: Int64,
        mxfp4_ptr: Int64,
        mxfp4_scale_ptr: Int64,
        grid_x: Int32,
        stream: Stream = Stream(None),  # noqa: B008
    ):
        quick_allreduce_int4(
            rank,
            nbytes,
            num_tiles,
            inp_ptr,
            out_ptr,
            peer_ptrs,
            colors_ptr,
            residual_ptr,
            residual_out_ptr,
            weight_ptr,
            mxfp4_ptr,
            mxfp4_scale_ptr,
            value_attrs=value_attrs,
        ).launch(
            grid=(grid_x, 1, 1),
            block=(BLOCK, 1, 1),
            stream=stream,
        )

    norm_tag = ""
    if fuse_norm:
        norm_tag = (
            f"_rms{norm_hidden}{'g' if gemma_norm else ''}"
            f"{'_mxfp4' if NORM_MXFP4 else ''}"
        )
    launch_quick_allreduce_int4.func.__name__ = (
        f"launch_quick_allreduce_int4_ws{world_size}_st{super_tile}{norm_tag}"
    )
    return {
        "launch": launch_quick_allreduce_int4,
        "flags_bytes": flags_i32 * 4,
        "data_bytes": PHASES * grid * world_size * wire_tile_bytes,
        "lds_bytes": LDS_BYTES + (NORM_RED_F32 * 4 if fuse_norm else 0),
        "tile_bytes": TILE_BYTES,
        "tile_fp16": TILE_FP16,
        "rank_tile_bytes": RANK_TILE_BYTES,
        "wire_tile_bytes": wire_tile_bytes,
        "super_tile": super_tile,
        "world_size": world_size,
        "rank_atoms": rank_atoms,
        "grid": grid,
        "block": BLOCK,
    }
