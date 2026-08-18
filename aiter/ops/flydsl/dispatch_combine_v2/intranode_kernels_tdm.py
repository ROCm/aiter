# Copyright © Advanced Micro Devices, Inc. All rights reserved.
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""TDM-transported intranode EP dispatch (gfx1250), a FlyDSL port of mori's
``src/ops/dispatch_combine_v2/ep_intranode_1250x.hpp``.

The sibling ``intranode_kernels.make_dispatch`` gives one whole wave to one
(token, expert-slot) route: the wave takes a remote atomic for the recv slot,
then copies the token with 16-byte per-lane stores. That shape is what a machine
without a DMA engine wants. It is the wrong shape for gfx1250, where the Tensor
Data Mover moves a whole row on one descriptor: the copy stops needing lanes at
all, so the per-route remote atomic and the scattered 4-byte metadata stores --
previously hidden under the payload copy -- become the cost.

So this kernel keeps mori's restructuring rather than bolting TDM onto the
existing one. Three things change, and they only pay off together:

  * a route is one LANE, not one wave (``WAVE / topk`` tokens per wave), so a
    wave sees all of a token's routes at once and can dedup same-peer routes
    against each other with a permute instead of a ballot per route;
  * the recv slots for a whole BLOCK are reserved with one remote atomic per
    (block, peer) instead of one per route, off an LDS histogram;
  * idx / weights / srcmap are gathered into a block-local, destination-ordered
    staging array first, so the cross-GPU metadata write is a handful of bulk
    TDM runs instead of thousands of scattered dwords.

Everything downstream (``tok_map`` encoding, the grid barrier, the per-peer
count signal, the self-resetting counters) is bit-compatible with
``make_dispatch``, so combine, replay and the op layer are unchanged.

Not ported: scales, fp4, push-group and replay. They are orthogonal to the
transport and each one needs its own staging region; ``make_dispatch`` still
owns those paths.
"""
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, const_expr, range_constexpr
from aiter.ops.flydsl.kernels.buffer_ops import (
    buffer_load,
    buffer_store,
    create_buffer_resource_from_addr,
)
from flydsl.expr.rocdl import ballot, ds_bpermute, readfirstlane, readlane
from flydsl.expr.typing import Int32, Int64, T

import mori.cco.device.flydsl as cco
from . import flydsl_prims as P
from . import tdm_prims as TDM
from .intranode_kernels import LANE_MASK, LOG2_WAVE, WAVE, _BALLOT_INT

#: gfx1250 LDS per workgroup. The payload tiles are the whole budget.
_LDS_BUDGET = 327680


def _align(v, a):
    return (v + a - 1) // a * a


def tdm_tokens_per_wave(experts_per_token):
    """Tokens one wave covers per iteration: enough lanes for a token's routes.

    ``WAVE / topk`` when the routes tile the wave exactly, so COUNT reads
    ``topk`` consecutive indices per token with every lane busy. Otherwise one
    token per wave with the tail lanes idle -- correct, just wider.
    """
    topk = experts_per_token
    return WAVE // topk if (0 < topk <= WAVE and WAVE % topk == 0) else 1


def tdm_stage_capacity(
    *, npes, experts_per_token, max_tok_per_rank, block_num, warp_num_per_block
):
    """(per-(block,peer) slot capacity, total staging slots).

    A block's routes to one peer cannot outnumber the tokens the block itself
    walked, and the token loop is grid-strided, so the bound is the number of
    whole rounds the block takes times what it covers in a round. Staging is
    indexed ``((block * npes) + peer) * cap + j`` -- contiguous per (block, peer)
    is what lets the metadata leave as bulk runs.

    Still an upper bound under the kernel's runtime quota ``etpi <= tpi``: the
    quota only drops below ``tpi`` when ``warps_total * etpi`` already covers the
    token count, which is one round of ``warp_num_per_block * etpi`` per block,
    and ``rounds`` is floored at 1 so ``cap`` is never under
    ``warp_num_per_block * tpi``.
    """
    tpi = tdm_tokens_per_wave(experts_per_token)
    per_round_grid = block_num * warp_num_per_block * tpi
    rounds = (max_tok_per_rank + per_round_grid - 1) // per_round_grid
    cap = warp_num_per_block * tpi * max(rounds, 1)
    return cap, block_num * npes * cap


def tdm_lds_bytes(*, hidden_dim, hidden_elem_size, warp_num_per_block, npes):
    """LDS the kernel asks for: one payload tile per warp, plus the counters."""
    tile = _align(hidden_dim * hidden_elem_size, 128)
    return warp_num_per_block * tile + _align(3 * npes * 4, 128)


def tdm_max_warps(*, hidden_dim, hidden_elem_size, npes):
    """Widest power-of-two warp count whose payload tiles fit the LDS budget.

    The vector transport holds no per-warp LDS, so a geometry tuned against it
    can name a warp count this one cannot honour: a 7168-wide bf16 tile is 14 KB
    and 32 of them want 448 KB against a 320 KB budget. A caller clamping to this
    keeps the tuned block count -- which is what paces the grid barrier -- and
    gives up only the warp width.
    """
    tile = _align(hidden_dim * hidden_elem_size, 128)
    room = (_LDS_BUDGET - _align(3 * npes * 4, 128)) // tile
    if room < 1:
        raise ValueError(
            f"a single {tile}B payload tile (hidden_dim={hidden_dim}, "
            f"{hidden_elem_size}B elements) does not fit the {_LDS_BUDGET}B LDS "
            f"budget; dispatch_transport='tdm' cannot serve this hidden size"
        )
    warps = 1
    while warps * 2 <= room:
        warps *= 2
    return warps


def make_dispatch_tdm(
    *,
    rank,
    npes,
    experts_per_rank,
    experts_per_token,
    hidden_dim,
    hidden_elem_size,
    max_tok_per_rank,
    max_recv,
    block_num,
    warp_num_per_block,
    off_tok_off,
    off_recv_num,
    off_tis,
    off_out_idx,
    off_out_wts,
    off_out_tok,
    enable_signal=True,
    meta_tdm=True,
):
    """Build the TDM dispatch kernel. Returns a ``@flyc.jit`` launcher.

    Arguments mirror :func:`intranode_kernels.make_dispatch` so the op layer can
    forward the same kwargs, minus the scales / fp4 / push-group / replay knobs
    this transport does not implement. ``meta_tdm=False`` routes the metadata
    through per-lane stores instead of the TDM engine -- same result, and the
    A/B that says whether the bulk path is worth its LDS.
    """
    if WAVE != 32:
        raise ValueError(
            f"TDM dispatch is gfx1250-only (wave32); wave size resolved to {WAVE}"
        )
    topk = experts_per_token
    nbytes = hidden_dim * hidden_elem_size
    if nbytes % 4:
        raise ValueError(f"token payload must be a whole number of dwords, got {nbytes}B")
    if topk > WAVE:
        raise ValueError(f"topk={topk} exceeds the wave size; a token's routes must fit one wave")

    tpi = tdm_tokens_per_wave(topk)
    warps_total = block_num * warp_num_per_block
    block_threads = warp_num_per_block * WAVE
    cap_blk, _stage_slots = tdm_stage_capacity(
        npes=npes,
        experts_per_token=topk,
        max_tok_per_rank=max_tok_per_rank,
        block_num=block_num,
        warp_num_per_block=warp_num_per_block,
    )
    sentinel_val = npes * max_recv

    tile_bytes = _align(nbytes, 128)
    ctl_bytes = _align(3 * npes * 4, 128)
    lds_bytes = warp_num_per_block * tile_bytes + ctl_bytes
    if lds_bytes > _LDS_BUDGET:
        raise ValueError(
            f"TDM dispatch needs {lds_bytes}B of LDS ({warp_num_per_block} warps x "
            f"{tile_bytes}B payload tile) over the {_LDS_BUDGET}B budget; lower "
            f"warp_num_per_block"
        )

    # A metadata batch reuses the warp's payload tile: idx, weights and srcmap
    # for `meta_chunk` destination-ordered tokens, each region 128B-aligned so
    # every TDM row is a full 128 bytes. Chunks are a whole number of rows, so
    # the shapes are compile-time; the ragged remainder falls back to stores.
    # A chunk needs a legal tile for BOTH its `chunk*topk` index/weight run and
    # its much shorter `chunk` srcmap run, and the srcmap is what binds: it needs
    # chunk >= 64 before any dim1 >= 2 leaves a 32-element row. Chunks that only
    # the srcmap rejects would otherwise silently take a 1xN descriptor.
    meta_per_tok = topk * 4 * 2 + 4
    meta_cap = (tile_bytes - 3 * 128) // meta_per_tok
    meta_chunk = 0
    meta_idx_shape = meta_src_shape = None
    for cand in (128, 64):
        if cand > meta_cap:
            continue
        idx_shape = TDM.tdm_run_shape(cand * topk)
        src_shape = TDM.tdm_run_shape(cand)
        if idx_shape and src_shape:
            meta_chunk, meta_idx_shape, meta_src_shape = cand, idx_shape, src_shape
            break
    use_meta_tdm = bool(meta_tdm) and meta_chunk > 0
    m_idx_off = 0
    m_wt_off = _align(meta_chunk * topk * 4, 128)
    m_src_off = m_wt_off + _align(meta_chunk * topk * 4, 128)

    # One warp per peer would leave every warp past the world size idle, so the
    # peers are split across the warps that exist. mori measured the split to be
    # a loss at small batches (the sub-runs get shorter than a TDM row), where
    # the metadata is short enough for one warp per peer to hide anyway.
    peer_split = max(1, warp_num_per_block // npes) if npes else 1
    meta_runs = npes * peer_split

    @flyc.kernel(known_block_size=[block_threads, 1, 1])
    def ep_dispatch_tdm(
        arena: Int64,
        addr_inp_tok: Int64,
        addr_inp_idx: Int64,
        addr_inp_wts: Int64,
        addr_tok_map: Int64,
        addr_dest_pe_ctr: Int64,
        addr_disp_bar: Int64,
        addr_total_recv: Int64,
        addr_stg_idx: Int64,
        addr_stg_wt: Int64,
        addr_stg_src: Int64,
        my_lsa_rank: Int32,
        inp_cur_tok: Int32,
    ):
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        lane = tid & LANE_MASK
        warp = tid >> LOG2_WAVE
        global_warp_id = bid * warp_num_per_block + warp
        window = cco.Window(arena)

        rsrc_inp_idx = create_buffer_resource_from_addr(addr_inp_idx)
        rsrc_inp_wts = create_buffer_resource_from_addr(addr_inp_wts)
        rsrc_tok_map = create_buffer_resource_from_addr(addr_tok_map)
        rsrc_dest_ctr = create_buffer_resource_from_addr(addr_dest_pe_ctr)
        rsrc_disp_bar = create_buffer_resource_from_addr(addr_disp_bar)
        rsrc_stg_idx = create_buffer_resource_from_addr(addr_stg_idx)
        rsrc_stg_wt = create_buffer_resource_from_addr(addr_stg_wt)
        rsrc_stg_src = create_buffer_resource_from_addr(addr_stg_src)

        # LDS: `warp_num_per_block` payload tiles, then the three per-peer
        # counters (committed count / reserved remote base / handout cursor).
        smem = fx.SharedAllocator(static=False)
        tile_ptr = smem.allocate(warp_num_per_block * tile_bytes, 128)._ptr
        ctl_ptr = smem.allocate(ctl_bytes, 128)._ptr
        tile_base_i32 = arith.index_cast(T.i32, fx.index_cast(T.index, fx.ptrtoint(tile_ptr)))
        my_tile = arith.addi(
            tile_base_i32,
            arith.muli(
                readfirstlane(T.i32, warp), arith.constant(tile_bytes, type=T.i32)
            ),
        )
        ctl = fx.Int64(fx.ptrtoint(ctl_ptr))

        def s_n(p):
            return ctl + fx.Int64(p) * fx.Int64(4)

        def s_base(p):
            return ctl + (fx.Int64(p) + fx.Int64(npes)) * fx.Int64(4)

        def s_run(p):
            return ctl + (fx.Int64(p) + fx.Int64(2 * npes)) * fx.Int64(4)

        # Tokens a warp takes per iteration. A fixed quota of `tpi` only fills the
        # grid once there are `warps_total * tpi` tokens to go round: below that
        # every token lands on a low warp id, which is block-major, so most of the
        # grid sends no payload at all and the kernel costs the same at 8 tokens
        # as at 512. Capping the quota at what it takes to cover the grid spreads
        # them. COUNT gives up its full-warp index burst when the cap bites, which
        # is the cheaper half of that trade.
        #
        # The clamp to 1 is load-bearing: `ceil(n / warps_total)` is 0 for n == 0,
        # and a token loop stepping by `warps_total * 0` never advances -- a hang
        # no correctness check can report, because the check hangs with it.
        if const_expr(tpi > 1):
            q_tok = (inp_cur_tok + (warps_total - 1)) // warps_total
            etpi = arith.select(
                q_tok < 1,
                fx.Int32(1),
                arith.select(q_tok < tpi, q_tok, fx.Int32(tpi)),
            )
        else:
            etpi = fx.Int32(1)

        # Lane -> (which of the wave's tokens, which of that token's routes). The
        # grouping follows `etpi`, not `tpi`: left on `tpi` the surplus lanes
        # would keep routing tokens the loops below hand to another warp.
        if const_expr(tpi > 1):
            s_lane = lane // topk
            e_lane = lane - s_lane * topk
        else:
            s_lane = fx.Int32(0)
            e_lane = lane
        lane_act = (s_lane < etpi) & (e_lane < topk)

        if tid < npes:
            P.store_i32_lds(s_n(tid), arith.constant(0))
            P.store_i32_lds(s_run(tid), arith.constant(0))
        fx.barrier()

        # Route resolution, shared by COUNT and FINALIZE. No dynamic `if`: it
        # runs under both phases' loops and selects rather than branches, so
        # inactive lanes stay in bounds instead of being masked off.
        def resolve(tok_base):
            tok = tok_base + s_lane
            act = lane_act & (tok < inp_cur_tok)
            tok_s = arith.select(tok < inp_cur_tok, tok, inp_cur_tok - 1)
            e_s = arith.select(lane_act, e_lane, 0)
            slot_off = tok_s * topk + e_s
            expert = buffer_load(rsrc_inp_idx, slot_off, vec_width=1, dtype=T.i32)
            dest_pe = expert // experts_per_rank
            valid = act & (expert >= 0) & (dest_pe < npes)
            # Dedup: a token routed to several experts on one peer is sent once.
            # The token's routes are `topk` adjacent lanes of this wave, so the
            # duplicate test is a permute per route slot -- lane (s_lane, e)'s
            # dest PE against mine -- and the lowest such lane wins.
            valid_i = arith.select(valid, fx.Int32(1), fx.Int32(0))
            keep = valid
            for e in range_constexpr(topk):
                probe = (s_lane * topk + e) * 4
                other_pe = ds_bpermute(T.i32, probe, dest_pe)
                other_ok = ds_bpermute(T.i32, probe, valid_i)
                dup = (e < e_lane) & (other_ok != 0) & (other_pe == dest_pe)
                keep = keep & ~dup
            # slot_off rather than the weight itself: COUNT has no use for the
            # weight, and the op permits a null weight pointer as long as nothing
            # is published, so the load belongs in FINALIZE.
            return tok, act, expert, slot_off, dest_pe, keep

        # ── COUNT: block-local histogram of routes per destination peer ──
        for tok_base in range(global_warp_id * etpi, inp_cur_tok, warps_total * etpi):
            _tok, _act, _expert, _off, dest_pe, keep = resolve(tok_base)
            if keep:
                P.atomic_add_lds(s_n(dest_pe), arith.constant(1))
        fx.barrier()

        # ── RESERVE: one remote atomic per (block, peer), not per route ──
        if tid < npes:
            n = P.load_i32_lds(s_n(tid))
            base = arith.constant(0)
            if n > 0:
                base = P.atomic_add_global(
                    fx.Int64(window.lsa_ptr(tid, off_tok_off)), n
                )
                P.atomic_add_global(
                    fx.Int64(addr_dest_pe_ctr) + fx.Int64(tid) * fx.Int64(4), n
                )
            P.store_i32_lds(s_base(tid), base)
        fx.barrier()

        # ── FINALIZE: hand out the reserved slots and gather the metadata ──
        # The slot order within a block's run is whatever the LDS cursor hands
        # out; only its contiguity matters, since the metadata is staged in that
        # same order and shipped as one run.
        for tok_base in range(global_warp_id * etpi, inp_cur_tok, warps_total * etpi):
            tok, act, expert, slot_off, dest_pe, keep = resolve(tok_base)
            wt = buffer_load(rsrc_inp_wts, slot_off, vec_width=1, dtype=T.f32)
            j = arith.constant(0)
            if keep:
                j = P.atomic_add_lds(s_run(dest_pe), arith.constant(1))
            base = arith.constant(0)
            if keep:
                base = P.load_i32_lds(s_base(dest_pe))
            dest_tok = base + j
            # `j >= cap_blk` cannot happen for a block that walked at most
            # cap_blk tokens; `dest_tok >= max_recv` can, when the peer is over
            # its configured recv cap. Both drop the route rather than write out
            # of bounds, matching make_dispatch's overflow behaviour.
            pub = keep & (j < cap_blk) & (dest_tok < max_recv)
            if act:
                buffer_store(
                    arith.select(pub, dest_pe * max_recv + dest_tok, sentinel_val),
                    rsrc_tok_map,
                    tok * topk + e_lane,
                )
            slot = (bid * npes + dest_pe) * cap_blk + j
            src_encoded = rank * max_tok_per_rank + tok
            # Stage the metadata destination-ordered. Every lane of the token's
            # group writes element `e_lane` of the published lane's slot, so a
            # route's `topk` indices leave as one contiguous 32-byte run and a
            # token routed to several peers is staged once per peer.
            pub_i = arith.select(pub, fx.Int32(1), fx.Int32(0))
            for e in range_constexpr(topk):
                probe = (s_lane * topk + e) * 4
                pub_e = ds_bpermute(T.i32, probe, pub_i)
                slot_e = ds_bpermute(T.i32, probe, slot)
                if lane_act & (pub_e != 0):
                    buffer_store(expert, rsrc_stg_idx, slot_e * topk + e_lane)
                    buffer_store(
                        arith.bitcast(T.i32, wt), rsrc_stg_wt, slot_e * topk + e_lane
                    )
                    if e_lane == 0:
                        buffer_store(src_encoded, rsrc_stg_src, slot_e)

        # tok_map is written here and re-read by the payload phase, and the
        # staging arrays are written here and read by the metadata phase; both
        # go through global memory, so the stores have to land before either.
        P.waitcnt_stores()
        fx.barrier()

        # ── META: the staged runs leave as bulk cross-GPU writes ──
        for run_id in range(warp, meta_runs, warp_num_per_block):
            peer = run_id // peer_split
            part = run_id - peer * peer_split
            cnt_all = P.load_i32_lds(s_n(peer))
            base_all = P.load_i32_lds(s_base(peer))
            # Split the peer's run across `peer_split` warps, remainder to the
            # low parts so the sub-runs differ by at most one token.
            q = cnt_all // peer_split
            rem = cnt_all - q * peer_split
            my_beg = part * q + arith.select(part < rem, part, rem)
            my_cnt = q + arith.select(part < rem, fx.Int32(1), fx.Int32(0))
            peer_idx = create_buffer_resource_from_addr(
                fx.Int64(window.lsa_ptr(peer, off_out_idx))
            )
            peer_wts = create_buffer_resource_from_addr(
                fx.Int64(window.lsa_ptr(peer, off_out_wts))
            )
            peer_tis = create_buffer_resource_from_addr(
                fx.Int64(window.lsa_ptr(peer, off_tis))
            )
            stg_beg = (bid * npes + peer) * cap_blk + my_beg
            step = meta_chunk if use_meta_tdm else 1
            for cs in range(0, my_cnt, step):
                left = my_cnt - cs
                dst = base_all + my_beg + cs
                src = stg_beg + cs
                if const_expr(use_meta_tdm):
                    if left >= meta_chunk:
                        g_idx = TDM.tdm_group1(*meta_idx_shape, 4)
                        g_src = TDM.tdm_group1(*meta_src_shape, 4)
                        l_idx = arith.addi(my_tile, arith.constant(m_idx_off, type=T.i32))
                        l_wt = arith.addi(my_tile, arith.constant(m_wt_off, type=T.i32))
                        l_src = arith.addi(my_tile, arith.constant(m_src_off, type=T.i32))
                        TDM.tdm_load(
                            TDM.tdm_group0(
                                l_idx,
                                fx.Int64(addr_stg_idx)
                                + fx.Int64(src) * fx.Int64(topk * 4),
                            ),
                            g_idx,
                        )
                        TDM.tdm_load(
                            TDM.tdm_group0(
                                l_wt,
                                fx.Int64(addr_stg_wt)
                                + fx.Int64(src) * fx.Int64(topk * 4),
                            ),
                            g_idx,
                        )
                        TDM.tdm_load(
                            TDM.tdm_group0(
                                l_src,
                                fx.Int64(addr_stg_src) + fx.Int64(src) * fx.Int64(4),
                            ),
                            g_src,
                        )
                        TDM.tdm_wait(0)
                        TDM.tdm_store(
                            TDM.tdm_group0(
                                l_idx,
                                fx.Int64(window.lsa_ptr(peer, off_out_idx))
                                + fx.Int64(dst) * fx.Int64(topk * 4),
                            ),
                            g_idx,
                        )
                        TDM.tdm_store(
                            TDM.tdm_group0(
                                l_wt,
                                fx.Int64(window.lsa_ptr(peer, off_out_wts))
                                + fx.Int64(dst) * fx.Int64(topk * 4),
                            ),
                            g_idx,
                        )
                        TDM.tdm_store(
                            TDM.tdm_group0(
                                l_src,
                                fx.Int64(window.lsa_ptr(peer, off_tis))
                                + fx.Int64(dst) * fx.Int64(4),
                            ),
                            g_src,
                        )
                        TDM.tdm_wait(0)
                    else:
                        # Ragged tail: fewer tokens than a whole TDM tile.
                        for i in range(lane, left * topk, WAVE):
                            buffer_store(
                                buffer_load(
                                    rsrc_stg_idx, src * topk + i, vec_width=1, dtype=T.i32
                                ),
                                peer_idx,
                                dst * topk + i,
                            )
                            buffer_store(
                                buffer_load(
                                    rsrc_stg_wt, src * topk + i, vec_width=1, dtype=T.i32
                                ),
                                peer_wts,
                                dst * topk + i,
                            )
                        for i in range(lane, left, WAVE):
                            buffer_store(
                                buffer_load(
                                    rsrc_stg_src, src + i, vec_width=1, dtype=T.i32
                                ),
                                peer_tis,
                                dst + i,
                            )
                else:
                    for i in range(lane, topk, WAVE):
                        buffer_store(
                            buffer_load(
                                rsrc_stg_idx, src * topk + i, vec_width=1, dtype=T.i32
                            ),
                            peer_idx,
                            dst * topk + i,
                        )
                        buffer_store(
                            buffer_load(
                                rsrc_stg_wt, src * topk + i, vec_width=1, dtype=T.i32
                            ),
                            peer_wts,
                            dst * topk + i,
                        )
                    if lane == 0:
                        buffer_store(
                            buffer_load(rsrc_stg_src, src, vec_width=1, dtype=T.i32),
                            peer_tis,
                            dst,
                        )

        # ── PAYLOAD: one TDM load per token, one TDM store per surviving route ──
        # No barrier before this. The tile a warp is about to overwrite is the
        # one it just drained itself; the cross-warp state (staging, s_base) was
        # published by the barrier after FINALIZE.
        #
        # The token partition has to be FINALIZE's, walked one token at a time.
        # tok_map goes through global memory but the only barrier between the two
        # phases is a workgroup one, so a warp may read back nothing but the
        # entries it wrote itself. A grid-strided `range(global_warp_id, ...)`
        # reads slots other BLOCKS own: at 512 tokens on a 64x8 grid it is warps
        # 0..127 (blocks 0..15) that route, and every one of the remaining 48
        # blocks would send payload off entries still holding the host's -1 fill.
        # -1 passes a `< sentinel` liveness test and decodes to dest_pe 0,
        # dest_tok -1, i.e. a TDM store one whole token BEFORE a peer's recv
        # buffer -- an out-of-bounds fabric write, which is what wedges the
        # engine rather than merely corrupting the result.
        #
        # `sub` is a runtime loop and not `range_constexpr` on purpose: the route
        # loop below already unrolls `topk` descriptor sites, and unrolling this
        # one too would multiply them by `tpi`.
        g_payload = TDM.tdm_group1(hidden_dim, 1, hidden_elem_size)
        probe_off = arith.select(lane < topk, lane, 0)
        for tok_base in range(global_warp_id * etpi, inp_cur_tok, warps_total * etpi):
            for sub in range(0, etpi):
                tok = tok_base + sub
                if tok < inp_cur_tok:
                    flat = buffer_load(
                        rsrc_tok_map, tok * topk + probe_off, vec_width=1, dtype=T.i32
                    )
                    # `flat >= 0` rejects the host's -1 fill as well as the
                    # sentinel, so a slot FINALIZE never published can never name
                    # a route.
                    live = (lane < topk) & (flat >= 0) & (flat < sentinel_val)
                    if ballot(_BALLOT_INT(), live) != 0:
                        TDM.tdm_load(
                            TDM.tdm_group0(
                                my_tile,
                                fx.Int64(addr_inp_tok)
                                + fx.Int64(tok) * fx.Int64(nbytes),
                            ),
                            g_payload,
                        )
                        TDM.tdm_wait(0)
                        live_i = arith.select(live, fx.Int32(1), fx.Int32(0))
                        for l in range_constexpr(topk):
                            live_l = readlane(T.i32, live_i, l)
                            flat_l = readlane(T.i32, flat, l)
                            if live_l != 0:
                                dest_pe = flat_l // max_recv
                                dest_tok = flat_l - dest_pe * max_recv
                                TDM.tdm_store(
                                    TDM.tdm_group0(
                                        my_tile,
                                        fx.Int64(window.lsa_ptr(dest_pe, off_out_tok))
                                        + fx.Int64(dest_tok) * fx.Int64(nbytes),
                                    ),
                                    g_payload,
                                )
                        # The next token reloads into this same tile.
                        TDM.tdm_wait(0)

        if const_expr(enable_signal):
            # Identical to make_dispatch's completion, so the two kernels are
            # interchangeable from combine's point of view.
            if global_warp_id == 0:
                if lane == 0:
                    buffer_store(
                        arith.constant(0),
                        create_buffer_resource_from_addr(addr_total_recv),
                        0,
                    )

            # TDM stores retire on the tensor counter, which storecnt does not
            # track, so the grid barrier needs both drains to cover the payload
            # and the plain stores.
            TDM.tdm_wait(0)
            P.waitcnt_stores()
            fx.barrier()
            if tid == 0:
                P.atomic_add_global(fx.Int64(addr_disp_bar), arith.constant(1))

            local_recv_num = fx.Int64(window.lsa_ptr(my_lsa_rank, off_recv_num))
            for dest_pe in range(lane, npes, WAVE):
                if global_warp_id == 0:
                    P.spin_until_eq_i32(fx.Int64(addr_disp_bar), block_num)
                    buffer_store(arith.constant(0), rsrc_disp_bar, 0)
                    signal_value = (
                        buffer_load(rsrc_dest_ctr, dest_pe, vec_width=1, dtype=T.i32) + 1
                    )
                    peer_recv_num = fx.Int64(window.lsa_ptr(dest_pe, off_recv_num))
                    recv_num_remote_addr = peer_recv_num + fx.Int64(rank) * fx.Int64(4)
                    P.spin_until_eq_i32(recv_num_remote_addr, 0)
                    P.store_i32_system(
                        recv_num_remote_addr, arith.constant(0), signal_value
                    )

            for src_pe in range(lane, npes, WAVE):
                if global_warp_id == 0:
                    recv_num_src_addr = local_recv_num + fx.Int64(src_pe) * fx.Int64(4)
                    signal_value = P.spin_until_gt_i32(recv_num_src_addr, 0)
                    peer_recv_count = signal_value - 1
                    P.store_i32_system(
                        recv_num_src_addr, arith.constant(0), arith.constant(0)
                    )
                    P.atomic_add_global(fx.Int64(addr_total_recv), peer_recv_count)
                    buffer_store(arith.constant(0), rsrc_dest_ctr, src_pe)

            if global_warp_id == 0:
                if lane == 0:
                    local_tok_off = fx.Int64(window.lsa_ptr(my_lsa_rank, off_tok_off))
                    P.store_i32_system(
                        local_tok_off, arith.constant(0), arith.constant(0)
                    )

    @flyc.jit
    def run(
        arena: Int64,
        addr_inp_tok: Int64,
        addr_inp_idx: Int64,
        addr_inp_wts: Int64,
        addr_tok_map: Int64,
        addr_dest_pe_ctr: Int64,
        addr_disp_bar: Int64,
        addr_total_recv: Int64,
        addr_stg_idx: Int64,
        addr_stg_wt: Int64,
        addr_stg_src: Int64,
        my_lsa_rank: Int32,
        inp_cur_tok: Int32,
        stream=fx.Stream(None),
    ):
        ep_dispatch_tdm(
            arena,
            addr_inp_tok,
            addr_inp_idx,
            addr_inp_wts,
            addr_tok_map,
            addr_dest_pe_ctr,
            addr_disp_bar,
            addr_total_recv,
            addr_stg_idx,
            addr_stg_wt,
            addr_stg_src,
            my_lsa_rank,
            inp_cur_tok,
        ).launch(
            grid=(block_num, 1, 1),
            block=[block_threads, 1, 1],
            stream=stream,
        )

    return run
