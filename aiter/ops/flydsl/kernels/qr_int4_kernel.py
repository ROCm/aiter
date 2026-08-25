# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""gfx942/gfx950 TP∈{2,4,8} INT4 two-shot all-reduce.

INT4 nibble: [-8,+7], −1/8, 4 B/thread, 1152 B rank-tile. Scale is
group-16 signed E4M3 in the 128 B region. Super-tile ST∈{1,8}; host
uses ST=1 when ``num_tiles ≤ GRID``. Payload HBM is bf16; in-kernel
math is packed fp16. Each rank owns ``ATOMS / world_size`` atoms of a
tile (8 GPUs → 1, 4 → 2, 2 → 4); LDS stays ``ATOMS * 1152``.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr import gpu, range_constexpr, rocdl
from flydsl.expr.typing import Int32, Int64, Stream, T, as_ir_value

from . import buffer_ops

WORLD = 8  # Default value for world size
SUPPORTED_WORLDS = (2, 4, 8)
BLOCK = 256
ATOMS = 8
TILE_BYTES = BLOCK * ATOMS * 16
TILE_I32 = TILE_BYTES // 4
TILE_FP16 = TILE_BYTES // 2
DEFAULT_GRID_CAP = 304 * 4
PHASES = 2
PHASE_REDUCE_SCATTER = 0
PHASE_ALL_GATHER = 1
RANK_TILE_BYTES = 1152
RANK_TILE_I32 = RANK_TILE_BYTES // 4
SUPER_TILES = (1, 8)
# 1024 B INT4 (256 i32) then 128 B group-16 E4M3 (32 i32). Rank-tile 1152 B.
SCALE_I32_OFF = 256
# Two threads (PAIR) share one E4M3; GROUP threads share the i32 slot.
GROUP = 8
PAIR = 2
WAVE = 64
WAVES = 4
# 4 lanes × 16 B = one 64 B NT sector. Not world_size.
QUAD_LANES = 4
QUADS_PER_WAVE = WAVE // QUAD_LANES
N_SECTORS = RANK_TILE_BYTES // 64
# dest × rank_atoms == ATOMS for every supported world size.
PACK_I32 = ATOMS * RANK_TILE_I32
LDS_BYTES = ATOMS * RANK_TILE_BYTES
# Wire/inbox addresses are byte pointers; tile math is in i32 slots.
I32_BYTES = 4

# gfx942 buffer aux: bit 1 = sc1 (bypass L2), bit 2 = NT.
_CM_SC1 = 2
_CM_NT = 4

# Dequant bit-trick: nibble | 0x6400 then + (-1032.0) as f16x2 reconstructs (q-8).
_K_MASK_000F = 0x000F000F
_K_HALF2_1024 = 0x64006400
_K_HALF2_1032 = 0xE408E408  # -1032.0 fp16x2


def _f16x2(packed):
    return fx.Vector.from_elements([packed], fx.Int32).bitcast(fx.Float16)


def _i32(vec):
    return vec.bitcast(fx.Int32)[0]


def _minnumf(a, b):
    return fx.Vector(fx.arith.minnumf(a, b), a.shape, a.dtype)


def _splat_f16x2(x):
    return fx.Vector.filled(2, x, fx.Float16)


def _clamp_fp16_overflow():
    """Saturate packed fp16 overflow to ±65504 instead of Inf.

    Packed add/mul/FMA follow MODE bit 23 (FP16_OVFL). Unset, overflow
    becomes Inf and every later FMA in that tile is Inf. Set, it saturates
    to the max finite fp16. INT4 is already a saturating codec, so a rare
    overflow should not poison the all-reduce.

    There is no FlyDSL wrapper; ``s_setreg_imm32_b32 0xdc1, 1`` writes
    ``hwreg(HW_REG_MODE, offset=23, size=2)``.
    """
    llvm.InlineAsmOp(None, [], "s_setreg_imm32_b32 0xdc1, 1", "", has_side_effects=True)


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
    wmin = _minnumf(_minnumf(p0, p1), _minnumf(p2, p3))
    wmax = fx.maxnumf(wmax, _shuffle_f16x2(wmax, 1))
    wmin = _minnumf(wmin, _shuffle_f16x2(wmin, 1))
    pk = (abs(wmax) > abs(wmin)).select(wmax, wmin)
    lo, hi = pk[0], pk[1]
    return fx.Float32((abs(lo) > abs(hi)).select(lo, hi))


def _atom_bf16_to_f16(atom):
    return fx.Vector(atom).bitcast(fx.BFloat16).to(fx.Float16).bitcast(fx.Int32)


def _atom_f16_to_bf16(atom):
    return fx.Vector(atom).bitcast(fx.Float16).to(fx.BFloat16).bitcast(fx.Int32)


def _f32_to_e4m3(x):
    """Signed E4M3 of a f32: 1 sign + 4 exp (bias 7, e=0 still implicit 1) + 3 mant.

    Group-16 wire scale. Not IEEE OCP E4M3 denorms: e=0 still encodes
    ``(1+m/8)*2^-7`` so typical INT4 extrema (~0.1) stay in range after
    ×−1/8. Byte 0 is +0.
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
    e4 = (e4 < fx.Int32(0)).select(
        fx.Int32(0), (e4 > fx.Int32(15)).select(fx.Int32(15), e4)
    )
    byte = sign | (e4 << fx.Int32(3)) | (m3 & fx.Int32(7))
    return is_z.select(fx.Int32(0), byte)


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
        w = _minnumf(fx.maxnumf(_f16x2(atom[i]) * enc_pk, lo), hi)
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
    mask = fx.Int32(_K_MASK_000F)
    bias_hi = fx.Int32(_K_HALF2_1024)
    bias_lo = _f16x2(fx.Int32(_K_HALF2_1032))
    for i in range_constexpr(4):
        q4 = (packed.shrui(fx.Int32(i * 4)) & mask) | bias_hi
        dq = _f16x2(q4) + bias_lo
        if acc is None:
            out.append(_i32(dq * scale))
        else:
            out.append(_i32(fx.fma(dq, scale, _f16x2(acc[i]))))
    return fx.Vector.from_elements(out, fx.Int32)


def _i32_to_bytes(i32_off):
    return fx.Int64(i32_off) * fx.Int64(I32_BYTES)


def _to_sgpr_i64(addr):
    """Copy a wave-uniform i64 from vector to scalar registers.

    Buffer loads/stores need the descriptor in scalar registers. After
    ``peers[rank]``, every lane holds the same pointer, but it sits in a
    vector register, so LLVM cannot prove that. It then serializes the
    wave: one lane at a time, copy that lane's pointer to a scalar
    register, mask to that lane, issue the load, repeat. ``readfirstlane``
    copies lane 0's value into a scalar register once so the whole wave
    issues a single buffer op.

    Do this at the inbox descriptor, not on the peer list: fanout stores
    use a different peer per lane, so those addresses must stay in vector
    registers. ``T.i64`` is the result type ``readfirstlane`` requires.
    """
    return fx.Int64(rocdl.readfirstlane(T.i64, as_ir_value(addr)))


def _store_v4i32_nt_global(addr_i64, data):
    """NT-store 16 B through a per-lane global address.

    One instruction here sends 16 B to a different GPU in each lane of a
    4-wide group. A buffer-descriptor store wants the descriptor in scalar
    registers, so LLVM would serialize those lanes (one destination at a
    time). A flat global store takes the address from a vector register,
    so all destinations issue together. ``nt`` skips L2; this is payload,
    not a flag.
    """
    ptr_ty = ir.Type.parse("!llvm.ptr<1>")
    ptr = llvm.IntToPtrOp(ptr_ty, as_ir_value(addr_i64)).result
    llvm.InlineAsmOp(
        None,
        [ptr, as_ir_value(data)],
        "global_store_dwordx4 $0, $1, off nt",
        "v,v",
        has_side_effects=True,
    )


def _load_i32_nt(rsrc, elem_off):
    return fx.Int32(
        buffer_ops.buffer_load(
            rsrc, elem_off, vec_width=1, dtype=T.i32, cache_modifier=_CM_NT
        )
    )


def _load_i32_uncached(rsrc):
    val = buffer_ops.buffer_load(
        rsrc, 0, vec_width=1, dtype=T.i32, cache_modifier=_CM_SC1
    )
    rocdl.s_waitcnt(vmcnt=0)
    return fx.Int32(val)


def _invalidate_l1():
    llvm.InlineAsmOp(None, [], "buffer_inv sc1", "", has_side_effects=True)


@fx.struct
class PackStorage:
    pack: fx.Array[fx.Int32, PACK_I32, 16]


def make_qr_int4_kernel(
    *, world_size: int = WORLD, super_tile: int = 1, grid: int
):
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
    # Each rank owns this many 16-byte atoms of a 32 KiB tile
    # (8 GPUs → 1, 4 → 2, 2 → 4). LDS still holds all ATOMS atoms.
    rank_atoms = ATOMS // world_size
    # Last-sector pad is ST * rank_atoms * RANK_TILE_I32 after the ST tiles.
    rank_payload_i32 = rank_atoms * RANK_TILE_I32
    release_i32_off = super_tile * rank_payload_i32
    wire_tile_i32 = release_i32_off + 16
    wire_tile_bytes = wire_tile_i32 * 4

    flags_i32 = PHASES * grid * world_size

    @flyc.kernel(known_block_size=[BLOCK, 1, 1])
    def qr_int4(
        rank: Int32,
        nbytes: Int64,
        num_tiles: Int32,
        inp_ptr: Int64,
        out_ptr: Int64,
        peer_ptrs: Int64,
        colors_ptr: Int64,
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
        # Remote NT fanout stays explicit global_store_dwordx4 nt.
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

        lds = fx.SharedAllocator().allocate(PackStorage).peek()
        pack = lds.pack.view(pack_layout)
        smem_ptr = lds.pack.ptr

        peer_rsrc = buffer_ops.create_buffer_resource_from_addr(peer_ptrs)
        peers = [
            buffer_ops.buffer_load(peer_rsrc, i, vec_width=1, dtype=T.i64)
            for i in range(world_size)
        ]
        peer_vec = fx.Vector.from_elements(peers, dtype=fx.Int64)
        self_rsrc = buffer_ops.create_buffer_resource_from_addr(
            _to_sgpr_i64(peer_vec[rank])
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
                gathered = _recv_all_gather(fx.Int32(0))
                _store_tile_atoms(tile, gathered)

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
                    gathered = _recv_all_gather(s)
                    tile = bid + (i + s) * fx.Int32(grid)
                    _store_tile_atoms(tile, gathered)

                color = color + fx.Int32(1)
                if color == fx.Int32(0):  # 0 is unset sentinel
                    color = fx.Int32(1)
        if tid == 0:
            _store_color(color)
        gpu.barrier()

    flat_wg = f"{BLOCK},{BLOCK}"

    @flyc.jit
    def launch_qr_int4(
        rank: Int32,
        nbytes: Int64,
        num_tiles: Int32,
        inp_ptr: Int64,
        out_ptr: Int64,
        peer_ptrs: Int64,
        colors_ptr: Int64,
        grid_x: Int32,
        stream: Stream = Stream(None),  # noqa: B008
    ):
        qr_int4(
            rank,
            nbytes,
            num_tiles,
            inp_ptr,
            out_ptr,
            peer_ptrs,
            colors_ptr,
            value_attrs={"rocdl.flat_work_group_size": flat_wg},
        ).launch(
            grid=(grid_x, 1, 1),
            block=(BLOCK, 1, 1),
            stream=stream,
        )

    launch_qr_int4.func.__name__ = f"launch_qr_int4_ws{world_size}_st{super_tile}"
    try:
        qr_int4.func.__name__ = f"qr_int4_ws{world_size}_st{super_tile}"
    except AttributeError:
        pass
    return {
        "launch": launch_qr_int4,
        "flags_bytes": flags_i32 * 4,
        "data_bytes": PHASES * grid * world_size * wire_tile_bytes,
        "lds_bytes": LDS_BYTES,
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
