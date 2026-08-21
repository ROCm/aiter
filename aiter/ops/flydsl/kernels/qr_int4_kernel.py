# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""gfx942 TP∈{2,4,8} INT4 two-shot all-reduce.

INT4 nibble: [-8,+7], −1/8, 4 B/thread, 1152 B rank-tile. Scale is
group-16 signed E4M3 in the 128 B region. Super-tile ST∈{1,8}; host
uses ST=1 when ``num_tiles ≤ GRID``. Payload HBM is bf16; in-kernel
math is packed fp16. HIP ``kRankAtoms = ATOMS / WORLD`` (TP8→1, TP4→2,
TP2→4); LDS stays ``ATOMS * 1152``.
"""

from typing import ClassVar

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm, scf
from flydsl.compiler.ast_rewriter import ASTRewriter
from flydsl.expr import gpu, range_constexpr
from flydsl.expr import rocdl as fly_rocdl
from flydsl.expr.typing import Int32, Int64, Stream, T

from . import buffer_ops

WORLD = 8
SUPPORTED_WORLDS = (2, 4, 8)
BLOCK = 256
ATOMS = 8
TILE_BYTES = BLOCK * ATOMS * 16
TILE_I32 = TILE_BYTES // 4
TILE_FP16 = TILE_BYTES // 2
GRID = 304 * 4
PHASES = 2
PHASE_RS = 0
PHASE_AG = 1
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
# dest × kRankAtoms == ATOMS for every supported WORLD.
PACK_I32 = ATOMS * RANK_TILE_I32
LDS_BYTES = ATOMS * RANK_TILE_BYTES

# gfx942 buffer aux: bit 1 = sc1 (bypass L2), bit 2 = NT.
_CM_SC1 = 2
_CM_NT = 4

# Dequant bit-trick: nibble | 0x6400 then + (-1032.0) as f16x2 reconstructs (q-8).
_K_MASK_000F = 0x000F000F
_K_HALF2_1024 = 0x64006400
_K_HALF2_1032 = 0xE408E408  # -1032.0 fp16x2


def _raw(v):
    return v.ir_value() if hasattr(v, "ir_value") else v


def _f16x2(packed):
    return fx.Vector.from_elements([packed], fx.Int32).bitcast(fx.Float16)


def _i32(vec):
    return vec.bitcast(fx.Int32)[0]


def _minnumf(a, b):
    return fx.Vector(fx.arith.minnumf(a, b), a.shape, a.dtype)


def _splat_f16x2(x):
    return fx.Vector.filled(2, x, fx.Float16)


def _enable_fp16_ovfl():
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
    zero = fx.Float32(0.0)
    is_z = x == zero
    sign = (x < zero).select(fx.Int32(0x80), fx.Int32(0))
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


def _codec_dequant(packed, scale):
    out = []
    mask = fx.Int32(_K_MASK_000F)
    bias_hi = fx.Int32(_K_HALF2_1024)
    bias_lo = _f16x2(fx.Int32(_K_HALF2_1032))
    for i in range_constexpr(4):
        q4 = (packed.shrui(fx.Int32(i * 4)) & mask) | bias_hi
        out.append(_i32((_f16x2(q4) + bias_lo) * scale))
    return fx.Vector.from_elements(out, fx.Int32)


def _codec_dequant_acc(packed, scale, acc):
    """Nibble reconstruct ``(q-8)``, then packed f16 FMA into *acc*.

    ``a * b + c`` does not contract to ``v_pk_fma_f16``; ``fx.fma`` does.
    Two fp16 lanes are independent channels, not a dot into f32.
    """
    out = []
    mask = fx.Int32(_K_MASK_000F)
    bias_hi = fx.Int32(_K_HALF2_1024)
    bias_lo = _f16x2(fx.Int32(_K_HALF2_1032))
    for i in range_constexpr(4):
        q4 = (packed.shrui(fx.Int32(i * 4)) & mask) | bias_hi
        out.append(_i32(fx.fma(_f16x2(q4) + bias_lo, scale, _f16x2(acc[i]))))
    return fx.Vector.from_elements(out, fx.Int32)


def _uniform_i64(addr):
    """Wave-uniform i64 → SGPR pair so ``make_buffer_rsrc`` does not waterfall.

    ``peer_vec[rank]`` is the same pointer in every lane, but it lives in VGPRs
    after the dynamic extract. LLVM then waterfalls every ``buffer_load`` that
    uses that rsrc (``v_readfirstlane`` + exec mask per load). One
    ``readfirstlane`` here moves the descriptor to SGPRs for the whole kernel.
    """
    return fx.Int64(
        fly_rocdl.readfirstlane(ir.IntegerType.get_signless(64), _raw(addr))
    )


def _store_v4i32_nt_global(addr_i64, data):
    """Per-lane NT ``global_store_dwordx4``. Buffer-rsrc stores waterfall when
    lanes target different peers; VGPR addresses keep the quad-fanout lockstep.
    """
    ptr_ty = ir.Type.parse("!llvm.ptr<1>")
    ptr = llvm.IntToPtrOp(ptr_ty, _raw(addr_i64)).result
    llvm.InlineAsmOp(
        None,
        [ptr, _raw(data)],
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
    fly_rocdl.s_waitcnt(vmcnt=0)
    return val


def _invalidate_l1():
    llvm.InlineAsmOp(None, [], "buffer_inv sc1", "", has_side_effects=True)


class _IfOnlyASTRewriter(ASTRewriter):
    """AST rewriter variant that lowers Python if, keeps while untouched."""

    transformers: ClassVar[list] = [
        t for t in ASTRewriter.transformers if t.__name__ != "CanonicalizeWhile"
    ]
    rewrite_globals: ClassVar[dict] = {
        name: value
        for name, value in ASTRewriter.rewrite_globals.items()
        if name not in {"scf_while_gen", "scf_while_init"}
    }


def _dsl_if_only(func):
    return _IfOnlyASTRewriter.transform(func)


@_dsl_if_only
def _wait_flag(flag_rsrc, color):
    i32 = T.i32
    initial = _load_i32_uncached(flag_rsrc)
    wait_loop = scf.WhileOp([i32], [initial])
    wait_cond_block = ir.Block.create_at_start(wait_loop.before, [i32])
    wait_body_block = ir.Block.create_at_start(wait_loop.after, [i32])
    with ir.InsertionPoint(wait_cond_block):
        current = wait_cond_block.arguments[0]
        should_wait = fx.Int32(current) != color
        scf.ConditionOp(_raw(should_wait), [current])
    with ir.InsertionPoint(wait_body_block):
        nxt = _load_i32_uncached(flag_rsrc)
        _invalidate_l1()
        scf.YieldOp([nxt])


@fx.struct
class PackStorage:
    pack: fx.Array[fx.Int32, PACK_I32, 16]


def make_qr_int4_kernel(*, world_size: int = WORLD, super_tile: int = 1):
    if world_size not in SUPPORTED_WORLDS:
        raise ValueError(
            f"world_size must be one of {SUPPORTED_WORLDS}, got {world_size}"
        )
    if ATOMS % world_size != 0:
        raise ValueError(f"ATOMS={ATOMS} is not divisible by world_size={world_size}")
    if super_tile not in SUPER_TILES:
        raise ValueError(f"super_tile must be one of {SUPER_TILES}, got {super_tile!r}")
    # HIP CodecQ4: kRankAtoms = kAtoms / world_size. TP8→1, TP4→2, TP2→4.
    k_rank_atoms = ATOMS // world_size
    # Last-sector pad is ST * kRankAtoms * RANK_TILE_I32 after the ST tiles.
    rank_payload_i32 = k_rank_atoms * RANK_TILE_I32
    release_i32_off = super_tile * rank_payload_i32
    wire_tile_i32 = release_i32_off + 16
    wire_tile_bytes = wire_tile_i32 * 4

    flags_i32 = PHASES * GRID * world_size

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
        _enable_fp16_ovfl()
        tid = fx.Int32(gpu.thread_id("x"))
        bid = fx.Int32(gpu.block_id("x"))

        thread_layout = fx.make_layout((WAVES, WAVE), (WAVE, 1))
        wave, lane = fx.idx2crd(tid, thread_layout).unpack()
        quad_layout = fx.make_layout((QUADS_PER_WAVE, QUAD_LANES), (QUAD_LANES, 1))
        quad, liq = fx.idx2crd(lane, quad_layout).unpack()
        linear = wave * fx.Int32(QUADS_PER_WAVE) + quad

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
        hbm_copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.Int32)
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
        # Lockstep (WORLD, sectors); idle lanes at TP2/4 are intentional.
        fanout_s8 = fx.make_layout((world_size, 8), (1, world_size))
        fanout_s2 = fx.make_layout((world_size, 2), (1, world_size))
        color_layout = fx.make_layout((GRID,), (1,))
        wire_slot_layout = fx.make_layout(
            (PHASES, GRID, world_size, super_tile),
            (
                GRID * world_size * wire_tile_i32,
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
            _uniform_i64(peer_vec[rank])
        )
        hbm_i32_ptr = fx.PointerType.get(
            T.i32, address_space=fx.AddressSpace.Global, alignment=16
        )

        def _hbm_buf(ptr):
            view = fx.make_view(fx.inttoptr(hbm_i32_ptr, ptr), hbm_layout)
            return fx.rocdl.make_buffer_tensor(
                view, max_size=False, num_records_bytes=nbytes
            )

        in_buf = _hbm_buf(inp_ptr)
        out_buf = _hbm_buf(out_ptr)
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
            fly_rocdl.s_waitcnt(vmcnt=0)

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

        def _pack_rs(atoms):
            # HIP: codec.send(..., &tA[r * kRankAtoms]) for each dest r.
            for dest in range_constexpr(world_size):
                for k in range_constexpr(k_rank_atoms):
                    packed, scale, is_leader = _codec_quant(
                        atoms[dest * k_rank_atoms + k], lane, tid
                    )
                    _lds_write_packet(
                        fx.Int32(dest * k_rank_atoms + k), packed, scale, is_leader
                    )

        def _pack_ag(accs):
            for k in range_constexpr(k_rank_atoms):
                packed, scale, is_leader = _codec_quant(accs[k], lane, tid)
                for dest in range_constexpr(world_size):
                    _lds_write_packet(
                        fx.Int32(dest * k_rank_atoms + k), packed, scale, is_leader
                    )

        def _fanout_nt(phase, inbox_src, sub):
            for k in range_constexpr(k_rank_atoms):
                for stripe in range_constexpr(3):
                    n_sec = 2 if stripe == 2 else 8
                    fanout = fanout_s2 if stripe == 2 else fanout_s8
                    limit = fx.Int32(world_size * n_sec)

                    def _emit(
                        lin,
                        limit=limit,
                        fanout=fanout,
                        stripe=stripe,
                        k=k,
                    ):
                        safe = (lin < limit).select(lin, fx.Int32(0))
                        peer, sec_in = fx.idx2crd(safe, fanout).unpack()
                        sector = fx.Int32(stripe * 8) + sec_in
                        if lin < limit:
                            vec_idx = fx.get_scalar(
                                fx.crd2idx((sector, liq), nt_own_layout)
                            )
                            # WORLD=8: kRankAtoms=1 so slot=peer, wire=vec_idx.
                            pack_peer = peer
                            wire_idx = vec_idx
                            if k_rank_atoms != 1:
                                pack_peer = peer * fx.Int32(k_rank_atoms) + fx.Int32(k)
                                wire_idx = vec_idx + fx.Int32(k * RANK_TILE_I32)
                            # 4xi32 NT vector cannot go through the i32 pack view.
                            v4 = fx.ptr_load(
                                smem_ptr + _pack_off(pack_peer, vec_idx),
                                result_type=fx.Vector.make_type(4, fx.Int32),
                            )
                            dest = peer_vec[peer]
                            byte_off = fx.Int64(
                                _sub_tile_i32(phase, inbox_src, sub) + wire_idx
                            ) * fx.Int64(4)
                            _store_v4i32_nt_global(dest + byte_off, v4)

                    _emit(linear)

        def _fanout_release(phase, inbox_src, color):
            """Last 64 B after ST rank-tiles: seq=color on all 16 i32."""
            limit = fx.Int32(world_size)
            safe = (linear < limit).select(linear, fx.Int32(0))
            if linear < limit:
                vec_idx = fx.Int32(release_i32_off) + liq * fx.Int32(4)
                v4 = fx.Vector.from_elements([color, color, color, color], fx.Int32)
                dest = peer_vec[safe]
                byte_off = fx.Int64(
                    _sub_tile_i32(phase, inbox_src, fx.Int32(0)) + vec_idx
                ) * fx.Int64(4)
                _store_v4i32_nt_global(dest + byte_off, v4)

        def _publish(phase, inbox_src, color):
            fly_rocdl.s_waitcnt(vmcnt=0)
            gpu.barrier()
            _fanout_release(phase, inbox_src, color)

        def _wait_release(phase, color):
            if tid < world_size:
                elem = _sub_tile_i32(phase, tid, fx.Int32(0)) + fx.Int32(
                    release_i32_off
                )
                _wait_flag(
                    buffer_ops.create_buffer_resource_from_addr(
                        peer_vec[rank] + fx.Int64(elem) * fx.Int64(4)
                    ),
                    color,
                )
            gpu.barrier()

        def _recv_q4(phase, src, sub, k=0):
            base = _sub_tile_i32(phase, src, sub)
            if k:
                base = base + fx.Int32(k * RANK_TILE_I32)
            packed = _load_i32_nt(self_rsrc, base + tid)
            word = _load_i32_nt(self_rsrc, base + fx.Int32(SCALE_I32_OFF) + scale_slot)
            e = word.shrui(pair_in_slot * fx.Int32(8)) & fx.Int32(0xFF)
            return packed, _e4m3_decoding_scale(e)

        def _reduce_rs(sub):
            accs = [None] * k_rank_atoms
            for src in range_constexpr(world_size):
                for k in range_constexpr(k_rank_atoms):
                    packed, scale = _recv_q4(PHASE_RS, fx.Int32(src), sub, k)
                    if accs[k] is None:
                        accs[k] = _codec_dequant(packed, scale)
                    else:
                        accs[k] = _codec_dequant_acc(packed, scale, accs[k])
            return accs

        def _recv_ag(sub):
            gathered = []
            for src in range_constexpr(world_size):
                for k in range_constexpr(k_rank_atoms):
                    packed, scale = _recv_q4(PHASE_AG, fx.Int32(src), sub, k)
                    gathered.append(_codec_dequant(packed, scale))
            return gathered

        n_block_tiles = (num_tiles - bid + fx.Int32(GRID - 1)) // fx.Int32(GRID)
        zero = fx.Int32(0)
        one = fx.Int32(1)
        color = _load_color()
        if super_tile == 1:
            for i in range(zero, n_block_tiles, one):
                tile = bid + i * fx.Int32(GRID)
                atoms = _load_tile_atoms(tile)
                _pack_rs(atoms)
                gpu.barrier()
                _fanout_nt(PHASE_RS, rank, zero)
                _publish(PHASE_RS, rank, color)

                _wait_release(PHASE_RS, color)
                acc = _reduce_rs(zero)

                _pack_ag(acc)
                gpu.barrier()
                _fanout_nt(PHASE_AG, rank, zero)
                _publish(PHASE_AG, rank, color)

                _wait_release(PHASE_AG, color)
                gathered = _recv_ag(zero)
                _store_tile_atoms(tile, gathered)

                color = color + one
                if color == 0:  # 0 is unset sentinel
                    color = one
        else:
            st_i = fx.Int32(super_tile)
            for i in range(zero, n_block_tiles, st_i):
                remain = n_block_tiles - i
                n_this = (remain < st_i).select(remain, st_i)

                for s in range(zero, n_this, one):
                    tile = bid + (i + s) * fx.Int32(GRID)
                    atoms = _load_tile_atoms(tile)
                    _pack_rs(atoms)
                    gpu.barrier()
                    _fanout_nt(PHASE_RS, rank, s)
                    if (s + one) < n_this:
                        # lgkmcnt(0) only: reuse ATOMS*1152 B LDS while NT stores fly.
                        fly_rocdl.s_waitcnt(lgkmcnt=0)

                _publish(PHASE_RS, rank, color)
                _wait_release(PHASE_RS, color)

                for s in range(zero, n_this, one):
                    acc = _reduce_rs(s)
                    _pack_ag(acc)
                    gpu.barrier()
                    _fanout_nt(PHASE_AG, rank, s)
                    if (s + one) < n_this:
                        # lgkmcnt(0) only: reuse ATOMS*1152 B LDS while NT stores fly.
                        fly_rocdl.s_waitcnt(lgkmcnt=0)

                _publish(PHASE_AG, rank, color)
                _wait_release(PHASE_AG, color)

                for s in range(zero, n_this, one):
                    gathered = _recv_ag(s)
                    tile = bid + (i + s) * fx.Int32(GRID)
                    _store_tile_atoms(tile, gathered)

                color = color + one
                if color == 0:  # 0 is unset sentinel
                    color = one
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
        "data_bytes": PHASES * GRID * world_size * wire_tile_bytes,
        "lds_bytes": LDS_BYTES,
        "tile_bytes": TILE_BYTES,
        "tile_fp16": TILE_FP16,
        "rank_tile_bytes": RANK_TILE_BYTES,
        "wire_tile_bytes": wire_tile_bytes,
        "super_tile": super_tile,
        "world_size": world_size,
        "rank_atoms": k_rank_atoms,
        "grid": GRID,
        "block": BLOCK,
    }
