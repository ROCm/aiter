# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""gfx942/gfx950 TP∈{2,4,8} INT4 **mesh** all-reduce.

Topology of each lap: every rank pushes directly to all ``N-1`` peers, twice. 

INT4 nibble: [-8,+7], −1/8, 4 B/thread, 1152 B rank-tile. Scale is
group-16 signed E4M3 in the 128 B region. Super-tile ST∈{1,8}; host
uses ST=1 when ``num_tiles ≤ GRID``. Payload HBM is bf16; in-kernel
math is packed fp16. Each rank owns ``ATOMS / world_size`` atoms of a
tile (8 GPUs → 1, 4 → 2, 2 → 4); LDS stays ``ATOMS * 1152``.

"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm
from flydsl.expr import gpu, range_constexpr, rocdl
from flydsl.expr.typing import Int32, Int64, Stream, T

from . import buffer_ops
from .qr_int_codec import (
    INT4,
    N_SECTORS,
    RANK_TILE_BYTES,
    RANK_TILE_I32,
    _atom_bf16_to_f16,
    _atom_f16_to_bf16,
    _clamp_fp16_overflow,
    _codec_dequant,
    _codec_quant,
    _scale_from_word,
    scale_slot_of,
    thread_lane,
)
from .qr_int_shared import (
    _INBOX_POLICY,
    ATOMS,
    BLOCK,
    QUAD_LANES,
    QUADS_PER_WAVE,
    SUPPORTED_WORLDS,
    TILE_BYTES,
    TILE_FP16,
    TILE_I32,
    WORLD,
    _acquire_inbox,
    _i32_to_bytes,
    _load_i32_nt,
    _load_i32_uncached,
    _store_v4i32_peer,
    _to_sgpr_i64,
    make_pack_storage,
)

PHASES = 2
PHASE_REDUCE_SCATTER = 0
PHASE_ALL_GATHER = 1
SUPER_TILES = (1, 8)
# dest × rank_atoms == ATOMS for every supported world size.
PACK_I32 = ATOMS * RANK_TILE_I32
LDS_BYTES = ATOMS * RANK_TILE_BYTES


PackStorage = make_pack_storage(PACK_I32)


def make_qr_int4_kernel(
    *,
    world_size: int = WORLD,
    super_tile: int = 1,
    grid: int,
    inbox_memory: str = "uncached",
):
    if world_size not in SUPPORTED_WORLDS:
        raise ValueError(
            f"world_size must be one of {SUPPORTED_WORLDS}, got {world_size}"
        )
    if inbox_memory not in _INBOX_POLICY:
        raise ValueError(
            f"inbox_memory must be one of {tuple(_INBOX_POLICY)}, got {inbox_memory!r}"
        )
    policy = _INBOX_POLICY[inbox_memory]
    payload_policy = policy["payload"]
    flag_policy = policy["flag"]
    release_writeback = policy["writeback"]
    recv_policy = policy["recv"]
    # Strides for the (peer, sector) fanout, resolved here rather than in the
    # kernel body: bindings made inside an `if` do not survive FlyDSL's trace,
    # which is why the body pre-assigns before conditionally overwriting.
    if policy["fanout"] == "peer":
        int4_stride, scale_stride = (8, 1), (2, 1)
    else:
        int4_stride, scale_stride = (1, world_size), (1, world_size)
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
        n_blocks: Int32,
    ):
        _clamp_fp16_overflow()
        tid = fx.Int32(gpu.thread_id("x"))
        bid = fx.Int32(gpu.block_id("x"))

        wave, lane = thread_lane(tid)
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
        scale_slot, pair_in_slot = scale_slot_of(tid)
        # Rank-tile = 18 × 64 B Infinity Fabric sectors: 16 INT4 then 2 E4M3.
        # A workgroup has 64 quads. A stripe of 8 sectors needs world_size*8
        # quads (64 at 8 GPUs); leftover quads sit idle (always on the
        # 2-sector scale tail, and on the INT4 stripes when world_size < 8).
        # Cover 18 as 8+8+2.
        #
        # Which axis runs fastest across consecutive quads is a fabric
        # question. "sector": consecutive quads target consecutive peers of
        # one sector, so a single store instruction hits every GPU -- ideal
        # on xGMI, whose native packet is exactly the 64 B a quad writes.
        # "peer": consecutive quads walk the sectors of one peer, giving each
        # destination a 512 B contiguous run. PCIe wants that -- interleaving
        # destinations every 64 B costs ~1.5x against >=256 B runs (36.25 vs
        # 54.03 GB/s measured on MI350P).
        fanout_int4_stripe = fx.make_layout((world_size, 8), int4_stride)
        fanout_scale_stripe = fx.make_layout((world_size, 2), scale_stride)
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

        def _lds_write_packet(slot, words, scale, is_leader):
            # INT4 only, so one payload word and no 2-bit plane. The ring is
            # where the multi-plane store lives.
            fx.memref_store(words[0], pack, (slot, tid))
            if is_leader:
                fx.memref_store(
                    scale, pack, (slot, fx.Int32(INT4.scale_i32_off) + scale_slot)
                )

        def _pack_reduce_scatter(atoms):
            """Quantize each destination's slice of this tile into LDS.

            A 32 KiB tile is 8 atoms; destination *d* owns
            ``atoms[d * rank_atoms : (d+1) * rank_atoms]``. Those packets
            are later NT-stored into *d*'s reduce-scatter inbox.
            """
            for dest in range_constexpr(world_size):
                for k in range_constexpr(rank_atoms):
                    words, scale, is_leader = _codec_quant(
                        INT4, atoms[dest * rank_atoms + k], lane, tid
                    )
                    _lds_write_packet(
                        fx.Int32(dest * rank_atoms + k), words, scale, is_leader
                    )

        def _pack_all_gather(accs):
            """Quantize the reduced slice and replicate it for every peer.

            After reduce-scatter this rank holds ``rank_atoms`` reduced
            atoms. Copy the same packets into every destination slot so the
            NT fanout can push them into every peer's all-gather inbox.
            """
            for k in range_constexpr(rank_atoms):
                words, scale, is_leader = _codec_quant(INT4, accs[k], lane, tid)
                for dest in range_constexpr(world_size):
                    _lds_write_packet(
                        fx.Int32(dest * rank_atoms + k), words, scale, is_leader
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
                        _store_v4i32_peer(dest + byte_off, v4, payload_policy)

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

            On a cacheable inbox retiring the stores is not enough -- they
            can be sitting in this XCD's L2. ``buffer_wbl2`` after the join
            writes them back, and its own ``vmcnt(0)`` waits for that to
            land before the flag goes out. Every workgroup issues its own:
            L2 is per-XCD, so one workgroup's writeback says nothing about
            a workgroup on another die.
            """
            rocdl.s_waitcnt(vmcnt=0)
            gpu.barrier()
            if release_writeback is not None:
                llvm.InlineAsmOp(None, [], release_writeback, "", has_side_effects=True)
                rocdl.s_waitcnt(vmcnt=0)
            limit = fx.Int32(world_size)
            safe = (quad_id < limit).select(quad_id, fx.Int32(0))
            if quad_id < limit:
                vec_idx = fx.Int32(release_i32_off) + lane_in_quad * fx.Int32(4)
                v4 = fx.Vector.from_elements([color, color, color, color], fx.Int32)
                dest = peer_vec[safe]
                byte_off = _i32_to_bytes(
                    _sub_tile_i32(phase, inbox_src, fx.Int32(0)) + vec_idx
                )
                _store_v4i32_peer(dest + byte_off, v4, flag_policy)

        def _wait_flag(flag_rsrc, color):
            # No fence in the loop body: _load_i32_uncached carries `sc0 sc1`,
            # so a retry cannot be served from a stale line. The acquire the
            # payload reads need is in _wait_release, once, after the join.
            current = _load_i32_uncached(flag_rsrc)
            while current != color:
                current = _load_i32_uncached(flag_rsrc)

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
            # Unconditional and after the join. Only `tid < world_size` spun,
            # so scoping the acquire to the spin would leave the other waves
            # of this workgroup reading the payload with nothing invalidated
            # on their behalf -- and would also skip it entirely in the common
            # case where the flag is already set on the first read.
            _acquire_inbox()

        def _recv_quantized(phase, src, sub, k=0):
            # Packed dword is at base+tid; scale dword is 1024 B later at a
            # group slot. They are not adjacent, so they cannot share one
            # vector load.
            base = _sub_tile_i32(phase, src, sub)
            if k:
                base = base + fx.Int32(k * RANK_TILE_I32)
            packed = _load_i32_nt(self_rsrc, base + tid, recv_policy)
            word = _load_i32_nt(
                self_rsrc, base + fx.Int32(INT4.scale_i32_off) + scale_slot, recv_policy
            )
            return (packed,), _scale_from_word(INT4, word, pair_in_slot)

        def _reduce_scattered(sub):
            """Dequant-accumulate every peer's reduce-scatter packet for *sub*."""
            accs = [None] * rank_atoms
            for src in range_constexpr(world_size):
                for k in range_constexpr(rank_atoms):
                    words, scale = _recv_quantized(
                        PHASE_REDUCE_SCATTER, fx.Int32(src), sub, k
                    )
                    if accs[k] is None:
                        accs[k] = _codec_dequant(INT4, words, scale, tid)
                    else:
                        accs[k] = _codec_dequant(INT4, words, scale, tid, accs[k])
            return accs

        def _recv_all_gather(sub):
            """Dequantize every peer's all-gather packet back into full-tile atoms."""
            gathered = []
            for src in range_constexpr(world_size):
                for k in range_constexpr(rank_atoms):
                    words, scale = _recv_quantized(
                        PHASE_ALL_GATHER, fx.Int32(src), sub, k
                    )
                    gathered.append(_codec_dequant(INT4, words, scale, tid))
            return gathered

        # Stride by the *launched* grid, not the compile-time cap. The host
        # launches fewer blocks than `grid` whenever it wants each block to own
        # several tiles (see QRInt4._grid_x); striding by the cap instead would
        # silently leave every tile above n_blocks unprocessed. `grid` still
        # sizes the wire slots and colour array, so n_blocks <= grid always.
        n_block_tiles = (num_tiles - bid + n_blocks - fx.Int32(1)) // n_blocks
        color = _load_color()
        if super_tile == 1:
            for i in range(fx.Int32(0), n_block_tiles, fx.Int32(1)):
                tile = bid + i * n_blocks
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
                    tile = bid + (i + s) * n_blocks
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
                    tile = bid + (i + s) * n_blocks
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
            grid_x,
            value_attrs={"rocdl.flat_work_group_size": flat_wg},
        ).launch(
            grid=(grid_x, 1, 1),
            block=(BLOCK, 1, 1),
            stream=stream,
        )

    # The inbox memory type changes the emitted store policy, so it has to be
    # part of the symbol name -- two variants that differ only in cache bits
    # must not collide in the JIT cache.
    tag = f"ws{world_size}_st{super_tile}_{inbox_memory}"
    launch_qr_int4.func.__name__ = f"launch_qr_int4_{tag}"
    try:
        qr_int4.func.__name__ = f"qr_int4_{tag}"
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
        "inbox_memory": inbox_memory,
        "payload_policy": payload_policy,
        "flag_policy": flag_policy,
        "release_writeback": release_writeback,
        "rank_atoms": rank_atoms,
        "grid": grid,
        "block": BLOCK,
    }
