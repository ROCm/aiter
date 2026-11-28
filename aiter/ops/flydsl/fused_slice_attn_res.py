# SPDX-License-Identifier: MIT
# FUSED slice all-reduce + AttnRes read (FULL op), parametric (nwg, nwaves). One kernel:
# each WG owns a contiguous slice of the H atoms, all-reduces its slice (Lamport, fixed
# per-atom inbox slot), residual-adds prefix_in -> prefix candidate (in registers, no HBM
# round-trip), then the full-op AttnRes read (score+softmax+mix+out_norm) over its slice.
# OVERLAP: the bank loads + the 8 bank-candidate dots (independent of AR) are issued before
# the AR poll spin, to hide the xGMI poll latency under HBM bank traffic. Whether that
# overlap materializes is the thing to MEASURE.
import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr import const_expr, range_constexpr, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.typing import Int32, ReductionOp, T
from flydsl.extension.coop.warp.rocdl import warp_reduce

from aiter.ops.flydsl.allreduce_policy import resolve_link
from aiter.ops.flydsl.comm_collectives_host_shared import _StEngine, resolve_inbox_flags
from aiter.ops.flydsl.kernels import communication_ops_utils as C
from aiter.ops.flydsl.kernels.comm.one_shot_allreduce import (
    ATOM_BYTES,
    ATOM_I32,
    INBOX_BUFFERS,
    SENTINEL_FILL_BYTE,
    _canonicalize,
    _has_sentinel,
)
from aiter.ops.flydsl.kernels.comm.transport import (
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
from aiter.ops.flydsl.kernels.tensor_shim import buf_base_i64

ATOM_ELEMS = 8
WAVE = 64


def _dpp_sum(v):
    return warp_reduce(v, ReductionOp.ADD, width=WAVE)


def _lds_barrier():
    rocdl.s_waitcnt(lgkmcnt=0)
    rocdl.s_barrier()


def build_fused_slice(
    world_size, nvb, nwg, nwaves, rank, hidden, num_tokens, inbox_memory
):
    import os as _os

    DBG_PREFIX = _os.environ.get("FUSED_DBG_PREFIX") == "1"
    WS = int(world_size)
    nvb = int(nvb)
    NWG = int(nwg)
    NWAVES = int(nwaves)
    self_rank = int(rank)
    hidden = int(hidden)
    M = int(num_tokens)
    A_TOTAL = hidden // ATOM_ELEMS
    BLOCK = NWAVES * WAVE
    base = A_TOTAL // NWG
    rem = A_TOTAL % NWG
    MAX_ATOMS = base + (1 if rem else 0)
    MAXIT = (MAX_ATOMS + BLOCK - 1) // BLOCK
    N_CAND = nvb + 1
    INV_H = 1.0 / hidden
    NPC = 2 * N_CAND
    TOTWG = M * NWG
    N_PART = NWAVES * NPC
    N_W = N_CAND + 1
    push_peers = [p for p in range(WS) if p != self_rank]
    policy = _INBOX_POLICY[inbox_memory]
    payload_policy = policy["payload"]
    release_scope = policy["release"]
    # inbox is sized for ALL M*A_TOTAL atoms; the slot carries the token so tokens
    # never collide (M>1 correctness).
    COLOR_STRIDE = WS * (M * A_TOTAL) * ATOM_I32
    SRC_STRIDE = (M * A_TOTAL) * ATOM_I32
    data_bytes = INBOX_BUFFERS * WS * (M * A_TOTAL) * ATOM_BYTES

    def _wrap3(c):
        over = c >= fx.Int32(INBOX_BUFFERS)
        return over.select(c - fx.Int32(INBOX_BUFFERS), c)

    @fx.struct
    class Shared:
        part: fx.Array[fx.Float32, N_PART, 16]
        weight: fx.Array[fx.Float32, max(1, N_W), 16]
        scale: fx.Array[fx.Float32, 1, 16]

    @flyc.kernel(known_block_size=[BLOCK, 1, 1])
    def kern(
        inp_ptr: fx.Tensor,
        prefix_ptr: fx.Tensor,
        bank_ptr: fx.Tensor,
        cw_ptr: fx.Tensor,
        ow_ptr: fx.Tensor,
        out_ptr: fx.Tensor,
        gpart_ptr: fx.Tensor,
        gsq_ptr: fx.Tensor,
        counter_ptr: fx.Tensor,
        peer_ptrs: fx.Int64,
        colors_ptr: fx.Int64,
        rank: fx.Int32,
        score_eps: fx.Float32,
        out_eps: fx.Float32,
        bank_stride_t: Int32,
        bank_stride_b: Int32,
    ):
        tid = fx.Int32(rocdl.workitem_id_x()) if False else fx.thread_idx.x
        bidx = fx.block_idx.x
        g = bidx % fx.Int32(NWG)
        t = bidx // fx.Int32(NWG)
        wave = tid // fx.Int32(WAVE)
        lane = tid % fx.Int32(WAVE)

        extra = (g < fx.Int32(rem)).select(fx.Int32(1), fx.Int32(0))
        cnt = fx.Int32(base) + extra
        gmin = (g < fx.Int32(rem)).select(g, fx.Int32(rem))
        start = g * fx.Int32(base) + gmin

        sh = fx.SharedAllocator().allocate(Shared).peek()
        part = sh.part.ptr
        wbuf = sh.weight.ptr
        scbuf = sh.scale.ptr

        peers = _load_peers(peer_ptrs, WS)
        peer_vec = fx.Vector.from_elements(peers, dtype=fx.Int64)
        my_inbox = _buffer_ptr(_to_sgpr_i64(peer_vec[rank]), T.i32, 16)
        inp_buf = _buffer_ptr(_to_sgpr_i64(buf_base_i64(inp_ptr)), T.i32, 16)
        prefix_buf = _buffer_ptr(_to_sgpr_i64(buf_base_i64(prefix_ptr)), T.i32, 16)
        bank_buf = _buffer_ptr(_to_sgpr_i64(buf_base_i64(bank_ptr)), T.i32, 16)
        cw_buf = _buffer_ptr(_to_sgpr_i64(buf_base_i64(cw_ptr)), T.f32, 16)
        ow_buf = _buffer_ptr(_to_sgpr_i64(buf_base_i64(ow_ptr)), T.f32, 16)
        out_buf = _buffer_ptr(_to_sgpr_i64(buf_base_i64(out_ptr)), T.i32, 16)
        gfp = C._to_ptr_global(buf_base_i64(gpart_ptr), fx.Float32, 4)
        gsqp = C._to_ptr_global(buf_base_i64(gsq_ptr), fx.Float32, 4)
        counter_addr = buf_base_i64(counter_ptr)

        _load_color, _store_color = _color_io(colors_ptr, bidx)
        color = _load_color()
        rearm_c = _wrap3(color + fx.Int32(2))

        def arslot(c, src, a):
            return (
                c * fx.Int32(COLOR_STRIDE)
                + src * fx.Int32(SRC_STRIDE)
                + a * fx.Int32(ATOM_I32)
            )

        bank_base = t * fx.Int32(bank_stride_t)
        acc_dot = [fx.Float32(0.0) for _ in range_constexpr(N_CAND)]
        acc_sq = [fx.Float32(0.0) for _ in range_constexpr(N_CAND)]
        prefix_keep = []
        bank_keep = []
        col_keep = []
        valid_keep = []

        for it in range_constexpr(MAXIT):
            la = tid + fx.Int32(it * BLOCK)
            valid = la < cnt
            atom = start + la
            safe_a = valid.select(atom, fx.Int32(0))
            aglob = (
                t * fx.Int32(A_TOTAL) + safe_a
            )  # global atom index (token*A_TOTAL + atom)
            col_i32 = safe_a * fx.Int32(ATOM_I32)
            col_f32 = safe_a * fx.Int32(ATOM_ELEMS)
            col_keep.append(col_i32)
            valid_keep.append(valid)

            # ---- AR push: my inp atom to every peer's inbox[color][self_rank][a] ----
            my_inp = _canonicalize(
                _buffer_load(
                    inp_buf, t * fx.Int32(hidden // 2) + col_i32, ATOM_I32, fx.Int32
                )
            )
            if valid:
                for peer in push_peers:
                    _store_v4i32_peer(
                        peer_vec[peer]
                        + _i32_to_bytes(arslot(color, fx.Int32(self_rank), aglob)),
                        my_inp,
                        payload_policy,
                    )
                if const_expr(release_scope is not None):
                    _release_inbox(release_scope)

            # ---- OVERLAP window: issue bank loads + bank-candidate dots (no AR dep) ----
            cw_lo = _buffer_load(cw_buf, col_f32, 4, fx.Float32)
            cw_hi = _buffer_load(cw_buf, col_f32 + fx.Int32(4), 4, fx.Float32)
            cw = fx.Vector.from_elements(
                [cw_lo[e] for e in range_constexpr(4)]
                + [cw_hi[e] for e in range_constexpr(4)],
                fx.Float32,
            )
            pin = atom_bf16_to_f32(
                _buffer_load(
                    prefix_buf, t * fx.Int32(hidden // 2) + col_i32, ATOM_I32, fx.Int32
                )
            )
            row_banks = []
            for r in range_constexpr(nvb):
                row_i32 = bank_base + fx.Int32(r) * fx.Int32(bank_stride_b)
                f = atom_bf16_to_f32(
                    _buffer_load(
                        bank_buf, (row_i32 // fx.Int32(2)) + col_i32, ATOM_I32, fx.Int32
                    )
                )
                row_banks.append(f)
                pdot = fx.Float32((f * cw).reduce(ReductionOp.ADD, fastmath="reassoc"))
                psq = fx.Float32((f * f).reduce(ReductionOp.ADD, fastmath="reassoc"))
                acc_dot[r] = acc_dot[r] + valid.select(pdot, fx.Float32(0.0))
                acc_sq[r] = acc_sq[r] + valid.select(psq, fx.Float32(0.0))
            bank_keep.append(row_banks)

            # ---- AR poll + reduce (bank work above overlapped the xGMI spin) ----
            # Unconditional (FlyDSL if/else does not leak a value); invalid lanes mask out
            # of the spin via `valid & has_sentinel` and are dropped from acc below.
            addrs = [
                peer_vec[rank] + _i32_to_bytes(arslot(color, fx.Int32(src), aglob))
                for src in push_peers
            ]

            def _pack(vecs):
                return fx.Vector.from_elements(
                    [v[e] for v in vecs for e in range_constexpr(ATOM_I32)], fx.Int32
                )

            got = _pack(
                [
                    _buffer_load(
                        my_inbox,
                        arslot(color, fx.Int32(src), aglob),
                        ATOM_I32,
                        fx.Int32,
                        _RECV_POLICY,
                    )
                    for src in push_peers
                ]
            )
            while valid & _has_sentinel(got):
                got = _pack([_poll_v4i32(a) for a in addrs])
            recv = {
                src: fx.Vector.from_elements(
                    [got[i * ATOM_I32 + e] for e in range_constexpr(ATOM_I32)], fx.Int32
                )
                for i, src in enumerate(push_peers)
            }
            acc = None
            for src in range_constexpr(WS):
                raw = my_inp if src == self_rank else recv[src]
                v = atom_bf16_to_f32(raw)
                acc = v if acc is None else acc + v
            ar = atom_bf16_to_f32(atom_f32_to_bf16(acc))  # AR output rounds to bf16

            # residual add -> prefix candidate (bf16-rounded, matches op), kept in registers
            prefix = atom_bf16_to_f32(atom_f32_to_bf16(pin + ar))
            prefix_keep.append(prefix)
            pdp = fx.Float32((prefix * cw).reduce(ReductionOp.ADD, fastmath="reassoc"))
            psp = fx.Float32(
                (prefix * prefix).reduce(ReductionOp.ADD, fastmath="reassoc")
            )
            acc_dot[nvb] = acc_dot[nvb] + valid.select(pdp, fx.Float32(0.0))
            acc_sq[nvb] = acc_sq[nvb] + valid.select(psp, fx.Float32(0.0))

            # rearm AR inbox (colour+2) for my atoms
            if valid:
                sentinel = fx.Vector.filled((ATOM_I32,), -1, fx.Int32)
                for src in push_peers:
                    _store_v4i32_peer(
                        peer_vec[rank]
                        + _i32_to_bytes(arslot(rearm_c, fx.Int32(src), aglob)),
                        sentinel,
                        payload_policy,
                    )

        # ---- fold partials -> gpart (own slot) ----
        for cand in range_constexpr(N_CAND):
            dot = _dpp_sum(acc_dot[cand])
            sq = _dpp_sum(acc_sq[cand])
            fx.ptr_store(dot, part + fx.Int64(fx.Int32(cand * 2 * NWAVES) + wave))
            fx.ptr_store(sq, part + fx.Int64(fx.Int32((cand * 2 + 1) * NWAVES) + wave))
        _lds_barrier()
        base_slot = (t * fx.Int32(NWG) + g) * fx.Int32(NPC)
        if wave == fx.Int32(0):
            live = lane < fx.Int32(N_CAND)
            dot_c = fx.Float32(0.0)
            sq_c = fx.Float32(0.0)
            for w in range_constexpr(NWAVES):
                di = lane * fx.Int32(2 * NWAVES) + fx.Int32(w)
                si = lane * fx.Int32(2 * NWAVES) + fx.Int32(NWAVES + w)
                dot_c = dot_c + fx.Float32(part[live.select(di, fx.Int32(0))])
                sq_c = sq_c + fx.Float32(part[live.select(si, fx.Int32(0))])
            if live:
                fx.ptr_store(dot_c, gfp + fx.Int64(base_slot + lane * fx.Int32(2)))
                fx.ptr_store(
                    sq_c, gfp + fx.Int64(base_slot + lane * fx.Int32(2) + fx.Int32(1))
                )

        # ---- GRID BARRIER #1 ----
        if tid == fx.Int32(0):
            C.fence_agent_release()
            a = fx.Int32(C.atomic_add_agent_one_as(counter_addr, 1))
            target = ((a // fx.Int32(TOTWG)) + fx.Int32(1)) * fx.Int32(TOTWG)
            C.spin_until_ge_i32_agent(counter_addr, target)
            C.fence_agent_acquire()
        _lds_barrier()

        if wave == fx.Int32(0):
            live = lane < fx.Int32(N_CAND)
            fdot = fx.Float32(0.0)
            fsq = fx.Float32(0.0)
            tok_base = t * fx.Int32(NWG) * fx.Int32(NPC)
            for s in range_constexpr(NWG):
                off = tok_base + fx.Int32(s * NPC) + lane * fx.Int32(2)
                off = live.select(off, fx.Int32(0))
                fdot = fdot + fx.Float32(fx.ptr_load(gfp + fx.Int64(off)))
                fsq = fsq + fx.Float32(fx.ptr_load(gfp + fx.Int64(off + fx.Int32(1))))
            with fx.fastmath("afn"):
                score = fdot * fmath.rsqrt(fsq * INV_H + score_eps)
            score = live.select(score, fx.Float32(float("-inf")))
            m = score
            for o in (8, 4, 2, 1):
                mv = m.shuffle_xor(o, WAVE)
                m = (mv > m).select(mv, m)
            e = live.select(fmath.exp(score - m), fx.Float32(0.0))
            denom = e
            for o in (8, 4, 2, 1):
                denom = denom + denom.shuffle_xor(o, WAVE)
            wgt = e * (fx.Float32(1.0) / denom)
            idx = live.select(lane, fx.Int32(N_CAND))
            fx.ptr_store(wgt, wbuf + fx.Int64(idx))
        _lds_barrier()

        # ---- mix (prefix from registers) + partial sum(mix^2) ----
        wp = fx.Float32(wbuf[fx.Int32(nvb)])
        mix_keep = []
        psumsq = fx.Float32(0.0)
        for it in range_constexpr(MAXIT):
            mix = prefix_keep[it] * wp
            for r in range_constexpr(nvb):
                mix = mix + bank_keep[it][r] * fx.Float32(wbuf[fx.Int32(r)])
            mix_keep.append(mix)
            pmsq = fx.Float32((mix * mix).reduce(ReductionOp.ADD, fastmath="reassoc"))
            psumsq = psumsq + valid_keep[it].select(pmsq, fx.Float32(0.0))

        l = _dpp_sum(psumsq)
        fx.ptr_store(l, part + fx.Int64(wave))
        _lds_barrier()
        if wave == fx.Int32(0):
            wgsq = fx.Float32(0.0)
            for w in range_constexpr(NWAVES):
                wgsq = wgsq + fx.Float32(part[fx.Int32(w)])
            if lane == fx.Int32(0):
                fx.ptr_store(wgsq, gsqp + fx.Int64(t * fx.Int32(NWG) + g))

        # ---- GRID BARRIER #2 ----
        if tid == fx.Int32(0):
            C.fence_agent_release()
            a2 = fx.Int32(C.atomic_add_agent_one_as(counter_addr, 1))
            target2 = ((a2 // fx.Int32(TOTWG)) + fx.Int32(1)) * fx.Int32(TOTWG)
            C.spin_until_ge_i32_agent(counter_addr, target2)
            C.fence_agent_acquire()
        _lds_barrier()

        if tid == fx.Int32(0):
            tot = fx.Float32(0.0)
            for s in range_constexpr(NWG):
                tot = tot + fx.Float32(
                    fx.ptr_load(gsqp + fx.Int64(t * fx.Int32(NWG) + fx.Int32(s)))
                )
            with fx.fastmath("afn"):
                sc = fx.Float32(1.0) / fmath.sqrt(tot * INV_H + out_eps)
            fx.ptr_store(sc, scbuf + fx.Int64(0))
        _lds_barrier()
        scale = fx.Float32(scbuf[fx.Int32(0)])

        for it in range_constexpr(MAXIT):
            cf = col_keep[it] * fx.Int32(2)
            ow_lo = _buffer_load(ow_buf, cf, 4, fx.Float32)
            ow_hi = _buffer_load(ow_buf, cf + fx.Int32(4), 4, fx.Float32)
            ow = fx.Vector.from_elements(
                [ow_lo[e] for e in range_constexpr(4)]
                + [ow_hi[e] for e in range_constexpr(4)],
                fx.Float32,
            )
            out_f = mix_keep[it] * scale * ow
            stored = prefix_keep[it] if const_expr(DBG_PREFIX) else out_f
            if valid_keep[it]:
                _buffer_store(
                    out_buf,
                    t * fx.Int32(hidden // 2) + col_keep[it],
                    atom_f32_to_bf16(stored),
                )

        # advance this workgroup's inbox colour: the next call writes and polls the buffer this
        # call re-armed two rounds ago, never the one it just read (stale peer data otherwise)
        if tid == fx.Int32(0):
            _store_color(_wrap3(color + fx.Int32(1)))

    @flyc.jit
    def launch(
        inp_ptr,
        prefix_ptr,
        bank_ptr,
        cw_ptr,
        ow_ptr,
        out_ptr,
        gpart_ptr,
        gsq_ptr,
        counter_ptr,
        peer_ptrs: fx.Int64,
        colors_ptr: fx.Int64,
        rank: fx.Int32,
        score_eps: fx.Float32,
        out_eps: fx.Float32,
        bank_stride_t: Int32,
        bank_stride_b: Int32,
        stream: fx.Stream,
    ):
        kern(
            inp_ptr,
            prefix_ptr,
            bank_ptr,
            cw_ptr,
            ow_ptr,
            out_ptr,
            gpart_ptr,
            gsq_ptr,
            counter_ptr,
            peer_ptrs,
            colors_ptr,
            rank,
            score_eps,
            out_eps,
            bank_stride_t,
            bank_stride_b,
        ).launch(grid=(fx.Int64(TOTWG), 1, 1), block=(BLOCK, 1, 1), stream=stream)

    spec = {
        "launch": launch,
        "flags_bytes": 0,
        "data_bytes": data_bytes,
        "lds_bytes": 0,
        "tile_bytes": ATOM_BYTES,
        "wire_tile_bytes": ATOM_BYTES,
        "super_tile": 1,
        "rank_tile_bytes": ATOM_BYTES,
        "tile_fp16": ATOM_BYTES // 2,
        "atoms": 1,
        "world_size": WS,
        "inbox_memory": inbox_memory,
        "fanout": "peer",
        "rank": self_rank,
        "inbox_fill": SENTINEL_FILL_BYTE,
        "grid": TOTWG,
        "block": BLOCK,
    }
    return spec


# Optimal (nwg, nwaves) per token count M, from the full TP8 gfx950 decode sweep
# (one_arm median+graph, best of the complete NWG x NWAVES grid incl. NWG=1,2).
# NWG falls with M (the per-token grid barrier over NWG workgroups gets costly as
# M grows), NWAVES rises to keep the ALU busy. Measured points:
#   M=1,2 -> (7,2)  M=4 -> (4,4)  M=8,16 -> (2,7)  M=32,64 -> (1,7)
# See vault decode_allm_result.txt for the numbers and the win per M.
def select_fused_geometry(num_tokens):
    """(nwg, nwaves) the decode sweep found optimal for ``num_tokens``."""
    M = int(num_tokens)
    if M <= 2:
        return 7, 2
    if M <= 4:
        return 4, 4
    if M <= 16:
        return 2, 7
    return 1, 7


class FusedSliceAttnRes:
    def __init__(
        self,
        *,
        group,
        device,
        rank,
        world_size,
        nvb,
        hidden,
        num_tokens,
        nwg=None,
        nwaves=None,
        inbox_memory="uncached",
        link=None
    ):
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.nvb = int(nvb)
        self.H = int(hidden)
        self.M = int(num_tokens)
        # Auto-pick the swept-optimal geometry unless the caller pins one.
        if nwg is None or nwaves is None:
            g, w = select_fused_geometry(num_tokens)
            nwg = g if nwg is None else nwg
            nwaves = w if nwaves is None else nwaves
        self.nwg = int(nwg)
        self.nwaves = int(nwaves)
        link = resolve_link(link)
        inbox_flags, resolved = resolve_inbox_flags(inbox_memory, world_size, link)
        self.spec = build_fused_slice(
            world_size, nvb, self.nwg, self.nwaves, rank, hidden, num_tokens, resolved
        )
        self._eng = _StEngine(
            spec=self.spec,
            group=group,
            rank=self.rank,
            world_size=self.world_size,
            inbox_flags=inbox_flags,
        )
        dev = device
        self.gpart = torch.zeros(
            self.M * self.nwg * 2 * (self.nvb + 1), dtype=torch.float32, device=dev
        )
        self.gsq = torch.zeros(self.M * self.nwg, dtype=torch.float32, device=dev)
        self.counter = torch.zeros(1, dtype=torch.int32, device=dev)

    def run(self, inp, prefix_in, bank, cw, ow, out, score_eps, out_eps):
        NB = bank.shape[1]
        cwf = cw if cw.dtype == torch.float32 else cw.float()
        owf = ow if ow.dtype == torch.float32 else ow.float()
        st = fx.Stream(torch.cuda.current_stream().cuda_stream)
        self.spec["launch"](
            inp,
            prefix_in,
            bank,
            cwf,
            owf,
            out,
            self.gpart,
            self.gsq,
            self.counter,
            fx.Int64(int(self._eng._gpu_peer_ptrs)),
            fx.Int64(int(self._eng._colors)),
            fx.Int32(self.rank),
            float(score_eps),
            float(out_eps),
            int(NB * self.H),
            int(self.H),
            st,
        )
        return out

    def close(self):
        self._eng.close()
