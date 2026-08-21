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
"""FlyDSL intranode CCO-LSA dispatch kernel for gfx1250 MegaMoE.

Conventions: `rsrc_*` = buffer resource descriptor; `safe_*` = real value on live
lanes / in-bounds fallback (0 or self-rank) on dropped lanes; "sentinel" = tok_map
dropped-slot marker (dest PE == npes); "tis" = recv-slot -> source-token map.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
import mori.cco.device.flydsl as cco
from flydsl.expr import arith, const_expr, range_constexpr
from flydsl.expr.rocdl import (
    ballot,
    readlane,
)
from flydsl.expr.typing import Int32, Int64, T

from aiter.ops.flydsl.kernels import communication_ops_utils as comm_ops
from aiter.ops.flydsl.kernels.buffer_ops import (
    buffer_load,
    buffer_store,
    create_buffer_resource_from_addr,
)

from .config import (
    _LANE_MASK as LANE_MASK,
)
from .config import (
    _LOG2_WAVE_SIZE as LOG2_WAVE,
)
from .config import (
    _WAVE_SIZE as WAVE,
)

_LANE_STRIDE_I32 = WAVE * 4
_DISP_NSTREAMS = 4
_MAIN_STRIDE_I32 = _DISP_NSTREAMS * _LANE_STRIDE_I32
_BUTTERFLY_OFFSETS = tuple(WAVE >> i for i in range(1, LOG2_WAVE + 1))

# Dispatch's cross-device barrier is not shared with combine's: this one gates on
# a grid-wide disp_bar count and then hands each peer its recv_num, while combine
# waits on monotonic per-rank phase slots. Different state, so nothing to factor
# out.


def _make_dispatch(
    *,
    rank,
    npes,
    experts_per_rank,
    experts_per_token,
    hidden_dim,
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
):
    nbytes = hidden_dim * 2
    n_i32 = nbytes // 4
    # sentinel: tok_map dropped-slot marker whose dest_pe (value // max_recv) == npes.
    sentinel_val = npes * max_recv

    @flyc.kernel(known_block_size=[warp_num_per_block * WAVE, 1, 1])
    def ep_dispatch(
        arena: Int64,
        addr_inp_tok: Int64,
        addr_inp_idx: Int64,
        addr_inp_wts: Int64,
        addr_tok_map: Int64,
        addr_dest_pe_ctr: Int64,
        addr_disp_bar: Int64,
        addr_total_recv: Int64,
        my_lsa_rank: Int32,
        inp_cur_tok: Int32,
    ):
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        lane = tid & LANE_MASK
        warp = tid >> LOG2_WAVE
        global_warp_id = bid * warp_num_per_block + warp
        global_warp_num = block_num * warp_num_per_block
        work_limit = inp_cur_tok * experts_per_token

        window = cco.Window(arena)
        rsrc_inp_idx = create_buffer_resource_from_addr(addr_inp_idx)
        rsrc_inp_wts = create_buffer_resource_from_addr(addr_inp_wts)
        rsrc_tok_map = create_buffer_resource_from_addr(addr_tok_map)
        rsrc_dest_ctr = create_buffer_resource_from_addr(addr_dest_pe_ctr)
        rsrc_disp_bar = create_buffer_resource_from_addr(addr_disp_bar)

        # ── Phase 1: P2P-scatter each (src_tok, k_slot) to its dest PE ──
        for work_idx in range(global_warp_id, work_limit, global_warp_num):
            src_tok = work_idx // experts_per_token
            k_slot = work_idx % experts_per_token
            dest_expert = buffer_load(rsrc_inp_idx, work_idx, vec_width=1, dtype=T.i32)
            # Dedup: a token routed to several experts on the SAME dest PE is sent
            # once, by the lowest k_slot. safe_lane keeps the probe in-bounds for
            # lanes >= k_slot.
            safe_lane = arith.select(lane < k_slot, lane, 0)
            lane_expert = buffer_load(
                rsrc_inp_idx,
                src_tok * experts_per_token + safe_lane,
                vec_width=1,
                dtype=T.i32,
            )
            dest_pe = dest_expert // experts_per_rank
            lane_dest_pe = lane_expert // experts_per_rank
            dup_per_lane = arith.select(
                lane_dest_pe == dest_pe, arith.select(lane < k_slot, lane, WAVE), WAVE
            )
            dup_ballot = ballot(T.i32, dup_per_lane < WAVE)
            is_dup = dup_ballot != 0

            dest_tok_lane0 = arith.constant(0)
            if lane == 0:  # noqa: SIM102 - device predicates
                if dup_ballot == 0:
                    peer_tok_off = fx.Int64(window.lsa_ptr(dest_pe, off_tok_off))
                    dest_tok_lane0 = comm_ops.atomic_add_system(
                        peer_tok_off, fx.Int32(1)
                    )
            dest_tok_id = readlane(T.i32, dest_tok_lane0, 0)
            overflow = dest_tok_id >= max_recv
            is_dup_or_overflow = arith.select(is_dup, is_dup, overflow)
            no_dup = dup_ballot == 0
            in_cap = dest_tok_id < max_recv
            do_publish = arith.select(no_dup, in_cap, no_dup)
            tok_map_entry = arith.select(
                is_dup_or_overflow, sentinel_val, dest_pe * max_recv + dest_tok_id
            )
            if lane == 0:
                buffer_store(tok_map_entry, rsrc_tok_map, work_idx)

            if lane == 0:  # noqa: SIM102 - device predicates
                if do_publish:
                    # Publish this recv slot's origin into the dest peer's tis,
                    # which combine routing reads back.
                    src_tok_encoded = rank * max_tok_per_rank + src_tok
                    peer_tis = fx.Int64(window.lsa_ptr(dest_pe, off_tis))
                    buffer_store(
                        src_tok_encoded,
                        create_buffer_resource_from_addr(peer_tis),
                        dest_tok_id,
                    )
                    dest_ctr_addr = fx.Int64(addr_dest_pe_ctr) + fx.Int64(
                        dest_pe
                    ) * fx.Int64(4)
                    comm_ops.atomic_add_system(dest_ctr_addr, fx.Int32(1))

            # Per-lane (weight, expert-idx) scatter (lanes < k).
            if lane < experts_per_token:  # noqa: SIM102 - device predicates
                if do_publish:
                    weight_src_off = src_tok * experts_per_token + lane
                    weight_val = buffer_load(
                        rsrc_inp_wts, weight_src_off, vec_width=1, dtype=T.f32
                    )
                    idx_val = buffer_load(
                        rsrc_inp_idx, weight_src_off, vec_width=1, dtype=T.i32
                    )
                    dest_slot = dest_tok_id * experts_per_token + lane
                    peer_wts = fx.Int64(window.lsa_ptr(dest_pe, off_out_wts))
                    buffer_store(
                        arith.bitcast(T.i32, weight_val),
                        create_buffer_resource_from_addr(peer_wts),
                        dest_slot,
                    )
                    peer_idx = fx.Int64(window.lsa_ptr(dest_pe, off_out_idx))
                    buffer_store(
                        idx_val, create_buffer_resource_from_addr(peer_idx), dest_slot
                    )

            # Token-embedding scatter: each lane owns 4 i32 (16B). _DISP_NSTREAMS
            # vec4 streams for memory-level parallelism, one-stream tail for the
            # remainder; dropped slots set copy_end == lane_i32_off (no-op).
            peer_tok_base = fx.Int64(window.lsa_ptr(dest_pe, off_out_tok))
            remote_tok_addr = peer_tok_base + fx.Int64(dest_tok_id) * fx.Int64(nbytes)
            local_tok_addr = fx.Int64(addr_inp_tok) + fx.Int64(src_tok) * fx.Int64(
                nbytes
            )
            rsrc_src = create_buffer_resource_from_addr(local_tok_addr)
            rsrc_dst = create_buffer_resource_from_addr(remote_tok_addr)
            lane_i32_off = lane * 4
            safe_end_i32 = (n_i32 // _MAIN_STRIDE_I32) * _MAIN_STRIDE_I32
            if const_expr(n_i32 >= _MAIN_STRIDE_I32 and safe_end_i32 > 0):
                copy_end_main = arith.select(
                    is_dup_or_overflow, lane_i32_off, safe_end_i32
                )
                for chunk in range(lane_i32_off, copy_end_main, _MAIN_STRIDE_I32):
                    vecs = [
                        buffer_load(
                            rsrc_src,
                            chunk + k * _LANE_STRIDE_I32,
                            vec_width=4,
                            dtype=T.i32,
                        )
                        for k in range_constexpr(_DISP_NSTREAMS)
                    ]
                    for k in range_constexpr(_DISP_NSTREAMS):
                        buffer_store(vecs[k], rsrc_dst, chunk + k * _LANE_STRIDE_I32)
            if const_expr(safe_end_i32 < n_i32):
                copy_end_tail = arith.select(is_dup_or_overflow, lane_i32_off, n_i32)
                for chunk in range(
                    lane_i32_off + safe_end_i32, copy_end_tail, _LANE_STRIDE_I32
                ):
                    vec_a = buffer_load(rsrc_src, chunk, vec_width=4, dtype=T.i32)
                    buffer_store(vec_a, rsrc_dst, chunk)
            elif const_expr(n_i32 < _MAIN_STRIDE_I32):
                copy_end_small = arith.select(is_dup_or_overflow, lane_i32_off, n_i32)
                for chunk in range(lane_i32_off, copy_end_small, _LANE_STRIDE_I32):
                    vec_a = buffer_load(rsrc_src, chunk, vec_width=4, dtype=T.i32)
                    buffer_store(vec_a, rsrc_dst, chunk)

        # Self-reset total_recv (CUDAGraph-safe; replaces a host-side zero_()).
        # Only global warp 0 touches it, and the waitcnt_all + grid barrier below
        # drains this store before the Phase-3 adds. total_recv is local, so no
        # release fence / L2 writeback is needed.
        if global_warp_id == 0:  # noqa: SIM102 - device predicates
            if lane == 0:
                buffer_store(
                    arith.constant(0),
                    create_buffer_resource_from_addr(addr_total_recv),
                    0,
                )

        # ── Phase 2: grid barrier + per-peer count signal ──
        # gpu.barrier lowers to s_barrier, which syncs wavefronts but (unlike HIP
        # __syncthreads) emits no implicit s_waitcnt, so drain the memory counters
        # first or the stores above may not be visible to peers.
        comm_ops.waitcnt_all()
        fx.barrier()
        if tid == 0:
            comm_ops.atomic_add_system(fx.Int64(addr_disp_bar), arith.constant(1))

        local_recv_num = fx.Int64(window.lsa_ptr(my_lsa_rank, off_recv_num))
        for dest_pe in range(lane, npes, WAVE):
            if global_warp_id == 0:
                comm_ops.spin_until_eq_i32(fx.Int64(addr_disp_bar), block_num)
                buffer_store(arith.constant(0), rsrc_disp_bar, 0)
                signal_value = (
                    buffer_load(rsrc_dest_ctr, dest_pe, vec_width=1, dtype=T.i32) + 1
                )
                peer_recv_num = fx.Int64(window.lsa_ptr(dest_pe, off_recv_num))
                recv_num_remote_addr = peer_recv_num + fx.Int64(rank) * fx.Int64(4)
                comm_ops.spin_until_eq_i32(recv_num_remote_addr, 0)
                comm_ops.store_i32_system(
                    recv_num_remote_addr, arith.constant(0), signal_value
                )

        # ── Phase 3: collect per-source counts into total_recv ──
        for src_pe in range(lane, npes, WAVE):
            if global_warp_id == 0:
                recv_num_src_addr = local_recv_num + fx.Int64(src_pe) * fx.Int64(4)
                signal_value = comm_ops.spin_until_gt_i32(recv_num_src_addr, 0)
                peer_recv_count = signal_value - 1
                comm_ops.store_i32_system(
                    recv_num_src_addr, arith.constant(0), arith.constant(0)
                )
                comm_ops.atomic_add_system(fx.Int64(addr_total_recv), peer_recv_count)
                buffer_store(arith.constant(0), rsrc_dest_ctr, src_pe)

        if global_warp_id == 0:  # noqa: SIM102 - device predicates
            if lane == 0:
                local_tok_off = fx.Int64(window.lsa_ptr(my_lsa_rank, off_tok_off))
                comm_ops.store_i32_system(
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
        my_lsa_rank: Int32,
        inp_cur_tok: Int32,
        stream=fx.Stream(None),  # noqa: B008
    ):
        ep_dispatch(
            arena,
            addr_inp_tok,
            addr_inp_idx,
            addr_inp_wts,
            addr_tok_map,
            addr_dest_pe_ctr,
            addr_disp_bar,
            addr_total_recv,
            my_lsa_rank,
            inp_cur_tok,
        ).launch(
            grid=(block_num, 1, 1),
            block=[warp_num_per_block * WAVE, 1, 1],
            stream=stream,
        )

    return run
