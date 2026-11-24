# SPDX-License-Identifier: MIT
# Kimi-K3 decode: the TP all-reduce and the attention-residual read in one FlyDSL kernel.
#
#   S       = A1 + ... + A8                         (Lamport one-shot over the ranks, per 16 B atom, fp32 in
#                                                    rank order, one bf16 rounding: the all-reduce's bits)
#   prefix  = bf16(residual + S)                    (the new running residual, written to out[1])
#   B[m,k]  = blocks bank[m, 0..nvb) and prefix     (candidates)
#   score_k = dot(B_k, cw) * rsqrt(mean(B_k^2) + score_eps)
#   w       = softmax(score)
#   mix     = sum_k w_k * B_k
#   out     = bf16(mix * rsqrt(mean(mix^2) + out_eps) * ow)   (written to out[0])
#
# Each token row is split over NWG workgroups. The epilogue (from an attlas run on a one-GPU proxy of this
# op) accumulates the candidates' Gram matrix on the matrix core in the first pass, so mean(mix^2) = w^T G w
# needs no second pass, and the workgroups of a row exchange their partials as epoch-tagged words instead of
# through a grid barrier. The inbox colour advances every call, so no call reads a peer's earlier data.
import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr import range_constexpr, rocdl, const_expr
from flydsl.expr import gpu as fgpu
from flydsl.expr import math as fmath
from flydsl.expr.typing import Int32, ReductionOp, T
from flydsl.extension.coop.warp.rocdl import warp_reduce

from aiter.ops.flydsl.kernels.comm.transport import (
    _buffer_load, _buffer_ptr, _buffer_store, _to_sgpr_i64, _i32_to_bytes,
    _store_v4i32_peer, _poll_v4i32, _load_peers, _color_io, _release_inbox,
    atom_bf16_to_f32, atom_f32_to_bf16, _INBOX_POLICY, _RECV_POLICY,
)
from aiter.ops.flydsl.kernels.comm.one_shot_allreduce import (
    _canonicalize, _has_sentinel, SENTINEL_FILL_BYTE, INBOX_BUFFERS, ATOM_BYTES,
)
from aiter.ops.flydsl.comm_collectives_host_shared import _StEngine, resolve_inbox_flags
from aiter.ops.flydsl.allreduce_policy import resolve_link
from aiter.ops.flydsl.kernels.tensor_shim import buf_base_i64
from aiter.ops.flydsl.kernels import communication_ops_utils as C

ATOM_ELEMS = 8
ATOM_I32 = ATOM_ELEMS // 2
WAVE = 64
SCORE_EPS = 1e-6
OUT_EPS = 1e-6


def geometry(num_tokens):
    """(workgroups per row, waves per workgroup): the best of a full 2D sweep of the TP8 kernel per token count."""
    if num_tokens <= 2:
        return 7, 2
    return 4, 4


def _dpp_sum(v):
    return warp_reduce(v, ReductionOp.ADD, width=WAVE)


def _tag(f, tag):
    bits = fx.Vector.from_elements([f], fx.Float32).bitcast(fx.Int32)[0]
    return fx.Vector.from_elements([bits, tag], fx.Int32).bitcast(fx.Int64)[0]


def _store_tagged(ptr, v):
    fx.generic_store(ptr, v, memory_order=fx.AtomicOrdering.Monotonic, syncscope=fx.rocdl.SyncScope.AgentOneAs)


def _load_tagged(ptr):
    return fx.Int64(fx.generic_load(ptr, volatile=True, memory_order=fx.AtomicOrdering.Monotonic,
                                    syncscope=fx.rocdl.SyncScope.AgentOneAs))


def _load_all(ptrs):
    return fx.Vector.from_elements([_load_tagged(p) for p in ptrs], fx.Int64)


def _count_missing(v, n, epoch):
    vi = v.bitcast(fx.Int32)
    miss = fx.Int32(0)
    for k in range(n):
        miss = miss + (vi[2 * k + 1] == epoch).select(fx.Int32(0), fx.Int32(1))
    return miss


def _values(v, n):
    vf = v.bitcast(fx.Float32)
    return [vf[2 * k] for k in range(n)]


def _lds_barrier():
    rocdl.s_waitcnt(lgkmcnt=0)
    rocdl.s_barrier()


def _finish(F, FD, lane, wbuf, scbuf, N_CAND, NG, HALFD, INV_H):
    """wave 0: lane l holds value l of [dot_0..dot_{N-1}, G in circulant order] in F, and lanes l < N hold G_ll in
    FD; the softmax weights times the output scale rsqrt(w^T G w / H + eps) go to wbuf. No branches.
    w^T G w = (e^T G e) / denom^2, so the exchanges of e overlap the denominator's reduction."""
    live = lane < fx.Int32(N_CAND)
    glive = lane < fx.Int32(NG)
    with fx.fastmath("afn"):
        score = F * fmath.rsqrt(FD * INV_H + SCORE_EPS)
    score = live.select(score, fx.Float32(float("-inf")))
    m = score
    for o in (8, 4, 2, 1):
        mv = m.shuffle_xor(o, WAVE); m = (mv > m).select(mv, m)
    e = live.select(fmath.exp(score - m), fx.Float32(0.0))
    # lane k holds G[i, (i + d) % N] with i = k % N, d = k // N (in F, after the N dots)
    kk = lane - fx.Int32(N_CAND)
    gk = glive.select(lane + fx.Int32(N_CAND), fx.Int32(0))
    fg = fx.Float32(fgpu.shuffle_idx(F, gk, WAVE))
    gi_l = lane % fx.Int32(N_CAND)
    gd_l = lane // fx.Int32(N_CAND)
    gj_l = gi_l + gd_l
    gj_l = (gj_l >= fx.Int32(N_CAND)).select(gj_l - fx.Int32(N_CAND), gj_l)
    gi_l = glive.select(gi_l, fx.Int32(0)); gj_l = glive.select(gj_l, fx.Int32(0))
    ei = fx.Float32(fgpu.shuffle_idx(e, gi_l, WAVE))
    ej = fx.Float32(fgpu.shuffle_idx(e, gj_l, WAVE))
    denom = e
    for o in (8, 4, 2, 1):
        denom = denom + denom.shuffle_xor(o, WAVE)
    coef = ((gd_l == fx.Int32(0)) | (gd_l == fx.Int32(HALFD))).select(fx.Float32(1.0), fx.Float32(2.0))
    term = glive.select(coef * ei * ej * fg, fx.Float32(0.0))
    qe = _dpp_sum(term)
    rd = fx.Float32(1.0) / denom
    with fx.fastmath("afn"):
        sc = fx.Float32(1.0) / fmath.sqrt(qe * rd * rd * INV_H + OUT_EPS)
    idx = live.select(lane, fx.Int32(N_CAND))
    fx.ptr_store(e * (rd * sc), wbuf + fx.Int64(idx))


def build(hidden, nvb, nwg, nwaves, num_tokens, world_size, rank, inbox_memory):
    hidden = int(hidden); nvb = int(nvb); NWG = int(nwg)
    NWAVES = int(nwaves); M = int(num_tokens)
    WS = int(world_size); self_rank = int(rank)
    push_peers = [p for p in range(WS) if p != self_rank]
    policy = _INBOX_POLICY[inbox_memory]
    payload_policy = policy["payload"]
    release_scope = policy["release"]
    A_TOTAL = hidden // ATOM_ELEMS
    BLOCK = NWAVES * WAVE
    base = A_TOTAL // NWG
    rem = A_TOTAL % NWG
    MAX_ATOMS = base + (1 if rem else 0)
    MAXIT = (MAX_ATOMS + BLOCK - 1) // BLOCK
    N_CAND = nvb + 1
    INV_H = 1.0 / hidden
    # Gram entries in circulant order: G[i, (i + d) % N] for d = 0 .. N // 2 (pairs at d = N / 2 appear twice)
    GPAIRS = [(k % N_CAND, (k % N_CAND + k // N_CAND) % N_CAND) for k in range(N_CAND * (N_CAND // 2 + 1))]
    NG = len(GPAIRS)
    ALL_VALID = rem == 0 and base == MAXIT * BLOCK
    HALFD = N_CAND // 2 if N_CAND % 2 == 0 else -1
    NV = N_CAND + NG
    assert NV <= WAVE
    NPC = NV
    TOTWG = M * NWG
    NCOL = N_CAND + 2
    assert NCOL <= 16
    XROW = NCOL * ATOM_I32
    N_PART = NWAVES * 256
    ROW_I32 = hidden // 2
    RANK_I32 = M * ROW_I32
    COLOR_STRIDE = WS * (M * A_TOTAL) * ATOM_I32
    SRC_STRIDE = (M * A_TOTAL) * ATOM_I32
    data_bytes = INBOX_BUFFERS * WS * (M * A_TOTAL) * ATOM_BYTES

    def _wrap3(c):
        over = c >= fx.Int32(INBOX_BUFFERS)
        return over.select(c - fx.Int32(INBOX_BUFFERS), c)

    @fx.struct
    class Shared:
        part: fx.Array[fx.Float32, N_PART, 16]
        xpose: fx.Array[fx.Int32, NWAVES * WAVE * XROW, 16]
        weight: fx.Array[fx.Float32, N_CAND + 1, 16]
        scale: fx.Array[fx.Float32, 1, 16]

    @flyc.kernel(name=f"attnres_fuse_kernel_m{M}_v{nvb}_h{hidden}")
    def attnres_fuse_kernel(inp_ptr: fx.Tensor, res_ptr: fx.Tensor, bank_ptr: fx.Tensor, cw_ptr: fx.Tensor,
                            ow_ptr: fx.Tensor, out_ptr: fx.Tensor, gpart_ptr: fx.Tensor, gsq_ptr: fx.Tensor,
                            counter_ptr: fx.Tensor, peer_ptrs: fx.Int64, colors_ptr: fx.Int64, rank: fx.Int32,
                            bank_stride_t: Int32, bank_stride_b: Int32):
        tid = fx.thread_idx.x
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
        xp = sh.xpose.ptr
        wbuf = sh.weight.ptr
        scbuf = sh.scale.ptr

        inp_buf = _buffer_ptr(_to_sgpr_i64(buf_base_i64(inp_ptr)), T.i32, 16)
        peers = _load_peers(peer_ptrs, WS)
        peer_vec = fx.Vector.from_elements(peers, dtype=fx.Int64)
        my_inbox = _buffer_ptr(_to_sgpr_i64(peer_vec[rank]), T.i32, 16)
        _load_color, _store_color = _color_io(colors_ptr, bidx)
        color = _load_color()
        rearm_c = _wrap3(color + fx.Int32(2))

        def arslot(c, src, a):
            return c * fx.Int32(COLOR_STRIDE) + src * fx.Int32(SRC_STRIDE) + a * fx.Int32(ATOM_I32)
        res_buf = _buffer_ptr(_to_sgpr_i64(buf_base_i64(res_ptr)), T.i32, 16)
        bank_buf = _buffer_ptr(_to_sgpr_i64(buf_base_i64(bank_ptr)), T.i32, 16)
        cw_buf = _buffer_ptr(_to_sgpr_i64(buf_base_i64(cw_ptr)), T.f32, 16)
        ow_buf = _buffer_ptr(_to_sgpr_i64(buf_base_i64(ow_ptr)), T.f32, 16)
        out_buf = _buffer_ptr(_to_sgpr_i64(buf_base_i64(out_ptr)), T.i32, 16)
        counter_addr = buf_base_i64(counter_ptr)
        gfp = C._to_ptr_global(buf_base_i64(gpart_ptr), fx.Int64, 8)
        if const_expr(NWG > 1):
            epoch_raw = fx.Int32(C.load_i32_global_agent(counter_addr))
        gsqp = C._to_ptr_global(buf_base_i64(gsq_ptr), fx.Float32, 4)

        bank_base = t * fx.Int32(bank_stride_t)
        gacc = fx.Vector.from_elements([fx.Float32(0.0)] * 4, fx.Float32)
        pv_keep = []; bank_keep = []; col_keep = []; valid_keep = []

        for it in range_constexpr(MAXIT):
            la = tid + fx.Int32(it * BLOCK)
            valid = la < cnt
            atom = start + la
            safe_atom = valid.select(atom, fx.Int32(0))
            col_i32 = safe_atom * fx.Int32(ATOM_I32)
            col_f32 = safe_atom * fx.Int32(ATOM_ELEMS)
            col_keep.append(col_i32); valid_keep.append(valid)
            row_i32 = t * fx.Int32(ROW_I32) + col_i32

            cw_lo = _buffer_load(cw_buf, col_f32, 4, fx.Float32)
            cw_hi = _buffer_load(cw_buf, col_f32 + fx.Int32(4), 4, fx.Float32)
            cw = fx.Vector.from_elements(
                [cw_lo[e] for e in range_constexpr(4)] + [cw_hi[e] for e in range_constexpr(4)], fx.Float32)

            # the all-reduce over 8 GPUs (Lamport one-shot): push my atom, poll the peers', sum in rank order
            aglob = t * fx.Int32(A_TOTAL) + safe_atom
            my_inp = _canonicalize(_buffer_load(inp_buf, row_i32, ATOM_I32, fx.Int32))
            if valid:
                for peer in push_peers:
                    _store_v4i32_peer(
                        peer_vec[peer] + _i32_to_bytes(arslot(color, fx.Int32(self_rank), aglob)),
                        my_inp, payload_policy)
                if const_expr(release_scope is not None):
                    _release_inbox(release_scope)
            addrs = [peer_vec[rank] + _i32_to_bytes(arslot(color, fx.Int32(src_), aglob)) for src_ in push_peers]

            def _pack(vecs):
                return fx.Vector.from_elements([v[e] for v in vecs for e in range_constexpr(ATOM_I32)], fx.Int32)

            got = _pack([_buffer_load(my_inbox, arslot(color, fx.Int32(src_), aglob), ATOM_I32, fx.Int32,
                                      _RECV_POLICY) for src_ in push_peers])
            while valid & _has_sentinel(got):
                got = _pack([_poll_v4i32(a) for a in addrs])
            recv = {src_: fx.Vector.from_elements([got[i * ATOM_I32 + e] for e in range_constexpr(ATOM_I32)],
                                                  fx.Int32) for i, src_ in enumerate(push_peers)}
            s = None
            for src_ in range_constexpr(WS):
                v_ = atom_bf16_to_f32(my_inp if src_ == self_rank else recv[src_])
                s = v_ if s is None else s + v_
            s_bf = atom_bf16_to_f32(atom_f32_to_bf16(s))
            if valid:
                sentinel = fx.Vector.filled((ATOM_I32,), -1, fx.Int32)
                for src_ in push_peers:
                    _store_v4i32_peer(peer_vec[rank] + _i32_to_bytes(arslot(rearm_c, fx.Int32(src_), aglob)),
                                      sentinel, payload_policy)
            res = atom_bf16_to_f32(_buffer_load(res_buf, row_i32, ATOM_I32, fx.Int32))
            prefix_i32 = atom_f32_to_bf16(res + s_bf)
            if valid:
                _buffer_store(out_buf, fx.Int32(RANK_I32) + row_i32, prefix_i32, 17)
            pv = atom_bf16_to_f32(prefix_i32)
            pv_keep.append(pv)

            row_banks = []; raw_banks = []
            for r in range_constexpr(nvb):
                brow = bank_base + fx.Int32(r) * fx.Int32(bank_stride_b)
                braw = _buffer_load(bank_buf, (brow // fx.Int32(2)) + col_i32, ATOM_I32, fx.Int32)
                raw_banks.append(braw)
                row_banks.append(atom_bf16_to_f32(braw))
            bank_keep.append(row_banks)
            # Gram on the matrix core: X = [B_0 .. B_{N-1}, cw_hi, cw_lo] (bf16 columns, cw split in two bf16),
            # staged through this wave's LDS so lane l feeds column l % 16 of 4 atoms; C += X^T X
            cwh = atom_f32_to_bf16(cw)
            cwl = atom_f32_to_bf16(cw - atom_bf16_to_f32(cwh))
            rawc = raw_banks + [prefix_i32, cwh, cwl]
            xbase = wave * fx.Int32(WAVE * XROW) + lane * fx.Int32(XROW)
            for c_ in range_constexpr(NCOL):
                xv = rawc[c_]
                if const_expr(not ALL_VALID):
                    xv = fx.Vector.from_elements([valid.select(xv[e], fx.Int32(0)) for e in range(ATOM_I32)], fx.Int32)
                fx.ptr_store(xv, xp + fx.Int64(xbase + fx.Int32(c_ * ATOM_I32)))
            mcol = (lane % fx.Int32(16)) % fx.Int32(NCOL)   # columns past NCOL repeat real ones; their C is unused
            rbase = wave * fx.Int32(WAVE * XROW) + (lane // fx.Int32(16)) * fx.Int32(XROW) + mcol * fx.Int32(ATOM_I32)
            for t_ in range_constexpr(WAVE // 4):
                xa = fx.ptr_load(xp + fx.Int64(rbase + fx.Int32(t_ * 4 * XROW)), T.vec(4, T.i32))
                xa = fx.Vector(xa).bitcast(fx.BFloat16)
                gacc = rocdl.mfma_f32_16x16x32_bf16(T.vec(4, T.f32), [xa, xa, gacc, 0, 0, 0])

        # C[row][col], row = 4 * (lane // 16) + r, col = lane % 16 (symmetric)
        gvec = fx.Vector(gacc)
        for r_ in range_constexpr(4):
            gi_ = (fx.Int32(4) * (lane // fx.Int32(16)) + fx.Int32(r_)) * fx.Int32(16) + lane % fx.Int32(16)
            fx.ptr_store(fx.Float32(gvec[r_]), part + fx.Int64(wave * fx.Int32(256) + gi_))
        _lds_barrier()

        base_slot = (t * fx.Int32(NWG) + g) * fx.Int32(NPC)
        if wave == fx.Int32(0):
            live = lane < fx.Int32(NV)
            isdot = lane < fx.Int32(N_CAND)
            kk = lane - fx.Int32(N_CAND)
            ci = kk % fx.Int32(N_CAND); cd = kk // fx.Int32(N_CAND)
            cj = ci + cd
            cj = (cj >= fx.Int32(N_CAND)).select(cj - fx.Int32(N_CAND), cj)
            i1 = isdot.select(lane * fx.Int32(16) + fx.Int32(N_CAND), ci * fx.Int32(16) + cj)
            i1 = live.select(i1, fx.Int32(0))
            i2 = isdot.select(lane * fx.Int32(16) + fx.Int32(N_CAND + 1), fx.Int32(0))
            tot_c = fx.Float32(0.0); tot_d = fx.Float32(0.0)
            i3 = isdot.select(lane * fx.Int32(17), fx.Int32(0))
            for w in range_constexpr(NWAVES):
                tot_c = tot_c + fx.Float32(part[i1 + fx.Int32(w * 256)])
                tot_c = tot_c + isdot.select(fx.Float32(part[i2 + fx.Int32(w * 256)]), fx.Float32(0.0))
                if const_expr(NWG == 1):
                    tot_d = tot_d + fx.Float32(part[i3 + fx.Int32(w * 256)])
            if const_expr(NWG == 1):
                _finish(tot_c, tot_d, lane, wbuf, scbuf, N_CAND, NG, HALFD, INV_H)
            else:
                epoch = epoch_raw // fx.Int32(TOTWG) + fx.Int32(1)
                if live:
                    _store_tagged(gfp + fx.Int64(base_slot + lane), _tag(tot_c, epoch))
                fsum = fx.Float32(0.0); dsum = fx.Float32(0.0)
                tok_base = t * fx.Int32(NWG) * fx.Int32(NPC)
                dlive = lane < fx.Int32(N_CAND)
                ptrs = []
                for s_ in range_constexpr(NWG):
                    off = live.select(tok_base + fx.Int32(s_ * NPC) + lane, tok_base + fx.Int32(s_ * NPC))
                    offd = dlive.select(tok_base + fx.Int32(s_ * NPC + N_CAND) + lane, tok_base + fx.Int32(s_ * NPC))
                    ptrs.append(gfp + fx.Int64(off)); ptrs.append(gfp + fx.Int64(offd))
                pv_all = _load_all(ptrs)
                while _count_missing(pv_all, 2 * NWG, epoch) != fx.Int32(0):
                    pv_all = _load_all(ptrs)
                vals = _values(pv_all, 2 * NWG)
                for s_ in range_constexpr(NWG):
                    fsum = fsum + vals[2 * s_]
                    dsum = dsum + vals[2 * s_ + 1]
                _finish(fsum, dsum, lane, wbuf, scbuf, N_CAND, NG, HALFD, INV_H)

        _lds_barrier()

        wp = fx.Float32(wbuf[fx.Int32(nvb)])
        mix_keep = []
        wv = [fx.Vector.from_elements([fx.Float32(wbuf[fx.Int32(r)])] * ATOM_ELEMS, fx.Float32) for r in range(nvb)]
        for it in range_constexpr(MAXIT):
            mix = pv_keep[it] * wp
            for r in range_constexpr(nvb):
                mix = fmath.fma(bank_keep[it][r], wv[r], mix)
            mix_keep.append(mix)

        for it in range_constexpr(MAXIT):
            col_f32 = col_keep[it] * fx.Int32(2)
            ow_lo = _buffer_load(ow_buf, col_f32, 4, fx.Float32)
            ow_hi = _buffer_load(ow_buf, col_f32 + fx.Int32(4), 4, fx.Float32)
            ow = fx.Vector.from_elements(
                [ow_lo[e] for e in range_constexpr(4)] + [ow_hi[e] for e in range_constexpr(4)], fx.Float32)
            out_f = mix_keep[it] * ow
            if valid_keep[it]:
                _buffer_store(out_buf, t * fx.Int32(ROW_I32) + col_keep[it], atom_f32_to_bf16(out_f), 17)
        if const_expr(NWG > 1):
            if tid == fx.Int32(0):
                C.atomic_add_agent_one_as(counter_addr, 1)
        # next call uses the next inbox colour (the one re-armed two calls ago)
        if tid == fx.Int32(0):
            _store_color(_wrap3(color + fx.Int32(1)))

    @flyc.jit
    def launch(inp_ptr, res_ptr, bank_ptr, cw_ptr, ow_ptr, out_ptr, gpart_ptr, gsq_ptr, counter_ptr,
               peer_ptrs: fx.Int64, colors_ptr: fx.Int64, rank: fx.Int32,
               bank_stride_t: Int32, bank_stride_b: Int32, stream: fx.Stream):
        attnres_fuse_kernel(inp_ptr, res_ptr, bank_ptr, cw_ptr, ow_ptr, out_ptr, gpart_ptr, gsq_ptr,
                            counter_ptr, peer_ptrs, colors_ptr, rank, bank_stride_t, bank_stride_b).launch(
            grid=(fx.Int64(TOTWG), 1, 1), block=(BLOCK, 1, 1), stream=stream)

    return {
        "launch": launch, "flags_bytes": 0, "data_bytes": data_bytes, "lds_bytes": 0,
        "tile_bytes": ATOM_BYTES, "wire_tile_bytes": ATOM_BYTES, "super_tile": 1,
        "rank_tile_bytes": ATOM_BYTES, "tile_fp16": ATOM_BYTES // 2, "atoms": 1,
        "world_size": WS, "inbox_memory": inbox_memory, "fanout": "peer", "rank": self_rank,
        "inbox_fill": SENTINEL_FILL_BYTE, "grid": TOTWG, "block": BLOCK,
    }


class FusedAttnResV2:
    """TP8 all-reduce + AttnRes read in one kernel (attlas champion epilogue). run() writes out[0] = the normed read
    and out[1] = the prefix of an out tensor [2, M, H]."""

    def __init__(self, *, group, device, rank, world_size, nvb, hidden, num_tokens, nwg=None, nwaves=None,
                 inbox_memory="uncached", link=None):
        self.rank = int(rank); self.M = int(num_tokens); self.nvb = int(nvb)
        g, w = geometry(num_tokens)
        self.nwg = int(nwg or g); self.nwaves = int(nwaves or w)
        inbox_flags, resolved = resolve_inbox_flags(inbox_memory, world_size, resolve_link(link))
        self.spec = build(hidden, nvb, self.nwg, self.nwaves, num_tokens, world_size, rank, resolved)
        self._eng = _StEngine(spec=self.spec, group=group, rank=self.rank, world_size=int(world_size),
                              inbox_flags=inbox_flags)
        M, n = self.M, int(nvb) + 1
        self.gpart = torch.zeros(M * self.nwg * (n * (n // 2 + 2)), dtype=torch.int64, device=device)
        self.gsq = torch.zeros(M * self.nwg, dtype=torch.float32, device=device)
        self.counter = torch.zeros(1, dtype=torch.int32, device=device)

    def run(self, inp, residual, bank, cw, ow, out2):
        st = fx.Stream(torch.cuda.current_stream().cuda_stream)
        self.spec["launch"](inp, residual, bank, cw, ow, out2, self.gpart, self.gsq, self.counter,
                            fx.Int64(int(self._eng._gpu_peer_ptrs)), fx.Int64(int(self._eng._colors)),
                            fx.Int32(self.rank), bank.stride(0), bank.stride(1), st)
        return out2

    def close(self):
        self._eng.close()
