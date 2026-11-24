# SPDX-License-Identifier: MIT
# SLICE-MODEL one-shot all-reduce: parametric over NWG workgroups and NWAVES waves/WG,
# INDEPENDENTLY. Each WG owns a contiguous slice of the H atoms; every atom has a FIXED
# inbox slot [color][src_rank][atom] (no grid-stride tile reuse), so back-to-back timing
# is stable and NWG is free (1..num_atoms). Reuses the Lamport push/poll/reduce/rearm of
# one_shot_allreduce and the _StEngine IPC setup verbatim; only the slot indexing and the
# per-WG atom-slice iteration are new.
import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr.typing import T

from aiter.ops.flydsl.allreduce_policy import resolve_link
from aiter.ops.flydsl.comm_collectives_host_shared import _StEngine, resolve_inbox_flags
from aiter.ops.flydsl.kernels.comm.one_shot_allreduce import (
    ATOM_BYTES,
    ATOM_I32,
    INBOX_BUFFERS,
    SENTINEL_FILL_BYTE,
    _canonicalize,
    _has_sentinel,
)
from aiter.ops.flydsl.kernels.comm.transport import (
    _CM_SC0,
    _CM_SC1,
    _INBOX_POLICY,
    _RECV_POLICY,
    _buffer_load,
    _buffer_ptr,
    _buffer_store,
    _color_io,
    _i32_to_bytes,
    _load_peers,
    _poll_v4i32,
    _release_inbox,
    _store_v4i32_peer,
    _to_sgpr_i64,
    atom_bf16_to_f32,
    atom_f32_to_bf16,
)

WAVE = 64


def build_slice_allreduce(world_size, nwg, nwaves, rank, num_atoms, inbox_memory):
    WS = int(world_size)
    NWG = int(nwg)
    NWAVES = int(nwaves)
    self_rank = int(rank)
    A_TOTAL = int(num_atoms)
    BLOCK = NWAVES * WAVE
    base = A_TOTAL // NWG
    rem = A_TOTAL % NWG
    MAX_ATOMS = base + (1 if rem else 0)
    MAXIT = (MAX_ATOMS + BLOCK - 1) // BLOCK
    push_peers = [p for p in range(WS) if p != self_rank]

    policy = _INBOX_POLICY[inbox_memory]
    payload_policy = policy["payload"]
    release_scope = policy["release"]

    COLOR_STRIDE = WS * A_TOTAL * ATOM_I32  # i32 between colours
    SRC_STRIDE = A_TOTAL * ATOM_I32  # i32 between source rows
    data_bytes = INBOX_BUFFERS * WS * A_TOTAL * ATOM_BYTES

    def _wrap3(c):
        over = c >= fx.Int32(INBOX_BUFFERS)
        return over.select(c - fx.Int32(INBOX_BUFFERS), c)

    @flyc.kernel(known_block_size=[BLOCK, 1, 1])
    def kern(
        rank: fx.Int32,
        inp_ptr: fx.Int64,
        out_ptr: fx.Int64,
        peer_ptrs: fx.Int64,
        colors_ptr: fx.Int64,
    ):
        tid = fx.Int32(gpu.thread_id("x"))
        bid = fx.Int32(gpu.block_id("x"))
        # this WG's contiguous atom slice
        extra = (bid < fx.Int32(rem)).select(fx.Int32(1), fx.Int32(0))
        cnt = fx.Int32(base) + extra
        gmin = (bid < fx.Int32(rem)).select(bid, fx.Int32(rem))
        start = bid * fx.Int32(base) + gmin

        peers = _load_peers(peer_ptrs, WS)
        peer_vec = fx.Vector.from_elements(peers, dtype=fx.Int64)
        my_inbox = _buffer_ptr(_to_sgpr_i64(peer_vec[rank]), T.i32, 16)
        in_buf = _buffer_ptr(_to_sgpr_i64(inp_ptr), T.i32, 16)
        out_buf = _buffer_ptr(_to_sgpr_i64(out_ptr), T.i32, 16)
        _load_color, _store_color = _color_io(colors_ptr, bid)
        color = _load_color()
        rearm_c = _wrap3(color + fx.Int32(2))

        def slot(c, src, a):
            return (
                c * fx.Int32(COLOR_STRIDE)
                + src * fx.Int32(SRC_STRIDE)
                + a * fx.Int32(ATOM_I32)
            )

        for it in range_constexpr(MAXIT):
            la = tid + fx.Int32(it * BLOCK)
            valid = la < cnt
            a = start + la
            safe_a = valid.select(a, fx.Int32(0))

            my = _canonicalize(
                _buffer_load(in_buf, safe_a * fx.Int32(ATOM_I32), ATOM_I32, fx.Int32)
            )

            if valid:
                # PUSH my atom into every peer's inbox[color][self_rank][a]
                for peer in push_peers:
                    _store_v4i32_peer(
                        peer_vec[peer]
                        + _i32_to_bytes(slot(color, fx.Int32(self_rank), safe_a)),
                        my,
                        payload_policy,
                    )
                if const_expr(release_scope is not None):
                    _release_inbox(release_scope)

                # POLL inbox[color][src][a] for every peer, spin until no sentinel
                addrs = [
                    peer_vec[rank] + _i32_to_bytes(slot(color, fx.Int32(src), safe_a))
                    for src in push_peers
                ]

                def _pack(vecs):
                    return fx.Vector.from_elements(
                        [v[e] for v in vecs for e in range_constexpr(ATOM_I32)],
                        fx.Int32,
                    )

                got = _pack(
                    [
                        _buffer_load(
                            my_inbox,
                            slot(color, fx.Int32(src), safe_a),
                            ATOM_I32,
                            fx.Int32,
                            _RECV_POLICY,
                        )
                        for src in push_peers
                    ]
                )
                while _has_sentinel(got):
                    got = _pack([_poll_v4i32(addr) for addr in addrs])
                recv = {
                    src: fx.Vector.from_elements(
                        [got[i * ATOM_I32 + e] for e in range_constexpr(ATOM_I32)],
                        fx.Int32,
                    )
                    for i, src in enumerate(push_peers)
                }

                # REDUCE across all ranks in rank order (fp32), mine from registers
                acc = None
                for src in range_constexpr(WS):
                    raw = my if src == self_rank else recv[src]
                    v = atom_bf16_to_f32(raw)
                    acc = v if acc is None else acc + v
                _buffer_store(
                    out_buf, safe_a * fx.Int32(ATOM_I32), atom_f32_to_bf16(acc)
                )

                # REARM colour+2 slots I read, to sentinel, for reuse 2 rounds later
                sentinel = fx.Vector.filled((ATOM_I32,), -1, fx.Int32)
                for src in push_peers:
                    if const_expr(release_scope is not None):
                        _buffer_store(
                            my_inbox,
                            slot(rearm_c, fx.Int32(src), safe_a),
                            sentinel,
                            _CM_SC0 | _CM_SC1,
                        )
                    else:
                        _store_v4i32_peer(
                            peer_vec[rank]
                            + _i32_to_bytes(slot(rearm_c, fx.Int32(src), safe_a)),
                            sentinel,
                            payload_policy,
                        )

        if tid == fx.Int32(0):
            _store_color(_wrap3(color + fx.Int32(1)))

    @flyc.jit
    def launch(
        rank: fx.Int32,
        inp_ptr: fx.Int64,
        out_ptr: fx.Int64,
        peer_ptrs: fx.Int64,
        colors_ptr: fx.Int64,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        kern(rank, inp_ptr, out_ptr, peer_ptrs, colors_ptr).launch(
            grid=(fx.Int32(NWG), 1, 1), block=(BLOCK, 1, 1), stream=stream
        )

    tile_bytes = A_TOTAL * ATOM_BYTES // NWG if NWG <= A_TOTAL else ATOM_BYTES
    return {
        "launch": launch,
        "flags_bytes": 0,
        "data_bytes": data_bytes,
        "lds_bytes": 0,
        "tile_bytes": max(ATOM_BYTES, tile_bytes),
        "wire_tile_bytes": max(ATOM_BYTES, tile_bytes),
        "super_tile": 1,
        "rank_tile_bytes": max(ATOM_BYTES, tile_bytes),
        "tile_fp16": max(ATOM_BYTES, tile_bytes) // 2,
        "atoms": 1,
        "world_size": WS,
        "inbox_memory": inbox_memory,
        "fanout": "peer",
        "rank": self_rank,
        "inbox_fill": SENTINEL_FILL_BYTE,
        "grid": NWG,
        "block": BLOCK,
    }


class SliceAllReduce:
    """Parametric slice-model one-shot all-reduce. One engine per (nwg, nwaves)."""

    def __init__(
        self,
        *,
        group,
        device,
        rank,
        world_size,
        nwg,
        nwaves,
        num_atoms,
        inbox_memory="uncached",
        link=None
    ):
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.device = device
        link = resolve_link(link)
        inbox_flags, resolved = resolve_inbox_flags(inbox_memory, world_size, link)
        self.spec = build_slice_allreduce(
            world_size, nwg, nwaves, rank, num_atoms, resolved
        )
        self._eng = _StEngine(
            spec=self.spec,
            group=group,
            rank=self.rank,
            world_size=self.world_size,
            inbox_flags=inbox_flags,
        )

    def allreduce(self, inp, out, stream=None):
        st = fx.Stream(
            torch.cuda.current_stream().cuda_stream if stream is None else stream
        )
        self.spec["launch"](
            fx.Int32(self.rank),
            fx.Int64(inp.data_ptr()),
            fx.Int64(out.data_ptr()),
            fx.Int64(int(self._eng._gpu_peer_ptrs)),
            fx.Int64(int(self._eng._colors)),
            st,
        )
        return out

    def close(self):
        self._eng.close()
