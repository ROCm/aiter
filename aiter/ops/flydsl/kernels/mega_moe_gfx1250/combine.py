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
"""Barrier-wait and top-k reduce kernel for gfx1250 Stage2-fused MegaMoE."""

import flydsl.compiler as flyc
import flydsl.expr as fx
import mori.cco.device.flydsl as cco
from flydsl.expr import arith, const_expr, range_constexpr
from flydsl.expr.typing import Int32, Int64, T

from aiter.ops.flydsl.kernels import communication_ops_utils as comm_ops
from aiter.ops.flydsl.kernels import vector
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

# NOTE: the cross-device xdb barrier is kept inlined per kernel (dispatch Phase 2,
# combine gather/scatter/fused) rather than a shared helper: FlyDSL only AST-rewrites
# dynamic `if` in the @flyc.kernel body, not in called helpers, so a helper's
# `if <lane predicate>:` would raise "cannot evaluate dynamic Boolean" at trace time.


def _V2BF16():
    return T.vec(2, T.bf16)


def _V2F32():
    return T.vec(2, T.f32)


def _V1I32():
    return T.vec(1, T.i32)


def _bf16_accum_funcs():
    def to_accum(i32_scalar):
        return vector.bitcast(
            _V2BF16(), vector.from_elements(_V1I32(), [i32_scalar])
        ).extf(_V2F32())

    def from_accum(acc):
        return vector.extract(
            vector.bitcast(_V1I32(), acc.truncf(_V2BF16())), static_position=[0]
        )

    def zero_accum():
        return to_accum(arith.constant(0))

    return to_accum, from_accum, zero_accum


# Fixed size of the per-block xdb flag counter array for the gather combine entry
# barrier: one i64 per block. 256 == the CU count (the max combine block_num), so
# block_num never exceeds it. Deliberately a hard-coded compile-time constant.
_XDB_FLAG_SLOTS = 256


def _make_combine_fused_sync(
    *,
    rank,
    npes,
    off_xdb_mem,
):
    """Cross-device barrier kernel for the split fused-combine path.

    A single-block kernel that does ONLY Stage A: wait until every peer's gemm2
    P2P writes into our comb_inp are globally visible. Launch this first on the
    stream, then the ``include_sync=False`` reduce kernel (grid fully free).
    Kernel retirement (stream-ordered) is the completion signal the reduce waits
    on -- no in-kernel fence needed.

    One thread per peer pushes the phase and polls that peer's local slot. The
    block is rounded up to whole waves, so rack-scale domains larger than one
    wave are covered without introducing a cross-wave dependency or barrier.
    The kernel's cost is the peer wait, not thread count.

    Because there is exactly one block, the per-block flag bookkeeping collapses
    to a single monotonic counter in xdb_flag[0]; the remaining _XDB_FLAG_SLOTS
    stay untouched. That makes this path EXCLUSIVE: within a run that uses it,
    the per-block (single-kernel) variant must never also run, or it would read
    a stale counter from its own slot and clear its barrier without waiting.
    """

    sync_block_size = ((npes + WAVE - 1) // WAVE) * WAVE

    @flyc.kernel(known_block_size=[sync_block_size, 1, 1])
    def ep_combine_fused_sync(
        arena: Int64,
        addr_xdb_flag: Int64,
        my_lsa_rank: Int32,
    ):
        tid = fx.thread_idx.x
        window = cco.Window(arena)
        rsrc_xdb_flag = create_buffer_resource_from_addr(addr_xdb_flag)
        # single monotonic phase (one block owns the counter; slot 0)
        phase = fx.Int64(buffer_load(rsrc_xdb_flag, 0, vec_width=1, dtype=T.i64))
        # push this call's phase to every peer's shared xdb slot [rank]
        if tid < npes:
            xdb_remote = fx.Int64(window.lsa_ptr(tid, off_xdb_mem)) + fx.Int64(
                rank
            ) * fx.Int64(8)
            comm_ops.store_i64_global_system(xdb_remote, phase)
        # advance the counter for the next call (single writer, no atomic)
        if tid == 0:
            buffer_store(phase + arith.constant(1, type=T.i64), rsrc_xdb_flag, 0)
        # poll each peer's local slot until it reports >= this call's phase.
        if tid < npes:
            xdb_peer_slot = fx.Int64(
                window.lsa_ptr(my_lsa_rank, off_xdb_mem)
            ) + fx.Int64(tid) * fx.Int64(8)
            comm_ops.spin_until_ge_i64(xdb_peer_slot, phase)

    @flyc.jit
    def run(
        arena: Int64,
        addr_xdb_flag: Int64,
        my_lsa_rank: Int32,
        stream=fx.Stream(None),  # noqa: B008
    ):
        ep_combine_fused_sync(arena, addr_xdb_flag, my_lsa_rank).launch(
            grid=(1, 1, 1),
            block=[sync_block_size, 1, 1],
            stream=stream,
        )

    return run


def _make_combine_fused_reduce(
    *,
    rank,
    npes,
    experts_per_token,
    hidden_dim,
    block_num,
    warp_num_per_block,
    off_comb_inp,
    off_xdb_mem,
    slot_stride_nbytes=None,
    include_sync=True,
    _s3_cache=2,
):
    """Combine for the GEMM2-fused scatter path.

    gemm2 has already P2P-written each token's WEIGHTED per-expert result into this
    rank's comb_inp[origin_lid*topk + k] (one contiguous topk-block per token). So
    combine only:
      Stage A  cross-device barrier -- wait until every peer's gemm2 P2P writes are
               globally visible before reading comb_inp.
      Stage B  each origin token: out[t] = sum_{k<topk} comb_inp[t*topk + k]. Weights
               were pre-applied in gemm2, so this is an unweighted sum. The dropless
               full-topk pipeline overwrites every active (token,k) slot each call.

    ``include_sync=True`` (default) keeps the original single-kernel behavior
    (Stage A cross-device barrier + Stage B reduce). ``include_sync=False`` emits
    ONLY Stage B (grid fully free), for the split path where a separate
    ``_make_combine_fused_sync`` 1-block kernel does the barrier first.

    The fused path carries a bf16 wire.
    """
    if include_sync and block_num > _XDB_FLAG_SLOTS:
        # Stage A indexes xdb_flag[bid]; a larger grid would run off the array.
        raise ValueError(
            f"combine block_num {block_num} exceeds xdb_flag_slots "
            f"{_XDB_FLAG_SLOTS}; use the split path (include_sync=False) or "
            f"raise _XDB_FLAG_SLOTS"
        )
    to_acc, from_acc, zero_acc = _bf16_accum_funcs()
    wire_nbytes = hidden_dim * 2
    n_i32 = wire_nbytes // 4  # valid i32 units read per slot (unpadded)
    # Per-slot stride: padded (pow2) for the TDM gather-store path so slot
    # addresses divide the 4GB-aligned per-rank window; defaults to the natural
    # hidden row size. Only the slot ADDRESS strides by this; the read count
    # (n_i32) stays hidden-based so padding tail is never read.
    slot_stride = slot_stride_nbytes if slot_stride_nbytes is not None else wire_nbytes
    topk = experts_per_token

    @flyc.kernel(known_block_size=[warp_num_per_block * WAVE, 1, 1])
    def ep_combine_fused(
        arena: Int64,
        addr_comb_inp: Int64,
        addr_xdb_flag: Int64,
        addr_out: Int64,
        my_lsa_rank: Int32,
        cur_rank_num_token: Int32,
    ):
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        lane = tid & LANE_MASK
        warp = tid >> LOG2_WAVE
        global_warp_id = bid * warp_num_per_block + warp
        global_warp_num = block_num * warp_num_per_block
        grid_thread_id = bid * (warp_num_per_block * WAVE) + tid

        window = cco.Window(arena)
        rsrc_out = create_buffer_resource_from_addr(addr_out)
        rsrc_xdb_flag = create_buffer_resource_from_addr(addr_xdb_flag)

        # ── Stage A: cross-device barrier — wait until all peers' gemm2 P2P writes
        # into our comb_inp are globally visible. Per-block private flag counters,
        # Each block owns xdb_flag[bid] (single writer, no atomic); block 0 pushes
        # this call's flag to every peer's shared
        # xdb slot, then EVERY block independently polls the local slots for its own
        # flag. No shared comb_bar => no atomic contention, no reset, no block-0
        # funnel; the monotonic flag needs no reset. ──
        # Compile-time gate: the split path (include_sync=False) omits Stage A and
        # relies on a separate 1-block _make_combine_fused_sync launched first.
        if include_sync:
            phase = fx.Int64(buffer_load(rsrc_xdb_flag, bid, vec_width=1, dtype=T.i64))
            if grid_thread_id < npes:
                xdb_remote = fx.Int64(
                    window.lsa_ptr(grid_thread_id, off_xdb_mem)
                ) + fx.Int64(rank) * fx.Int64(8)
                comm_ops.store_i64_global_system(xdb_remote, phase)
            # advance this block's private counter for the next call (single writer)
            if tid == 0:
                buffer_store(phase + arith.constant(1, type=T.i64), rsrc_xdb_flag, bid)
            # block 0 fills the unused tail counters [block_num, xdb_flag_slots) so a
            # later call that picks a larger block_num still reads synced counters.
            if const_expr(  # noqa: SIM102 - keep the device and compile-time branches separate.
                _XDB_FLAG_SLOTS > block_num
            ):
                if bid == 0:
                    _tail = _XDB_FLAG_SLOTS - block_num
                    _nthr = warp_num_per_block * WAVE
                    for r in range_constexpr((_tail + _nthr - 1) // _nthr):
                        idx = tid + r * _nthr
                        if idx < _tail:
                            buffer_store(
                                phase + arith.constant(1, type=T.i64),
                                rsrc_xdb_flag,
                                block_num + idx,
                            )
            # every block independently polls the shared local slots (local reads).
            # `>=` not `==`: a faster peer can lap us and overwrite its monotonic push
            # with a higher call count before our late-scheduled blocks read it.
            if tid < npes:
                xdb_peer_slot = fx.Int64(
                    window.lsa_ptr(my_lsa_rank, off_xdb_mem)
                ) + fx.Int64(tid) * fx.Int64(8)
                comm_ops.spin_until_ge_i64(xdb_peer_slot, phase)
            fx.barrier()

        # ── Stage B: local read of comb_inp[tok*topk + k] + unweighted topk sum ──
        comb_inp_base = fx.Int64(addr_comb_inp)
        safe_tok = arith.select(
            cur_rank_num_token == arith.constant(0),
            arith.constant(1),
            cur_rank_num_token,
        )
        warps_per_tok = (
            arith.constant(global_warp_num) + safe_tok - arith.constant(1)
        ) // safe_tok
        units_per_warp = (
            arith.constant(n_i32) + warps_per_tok - arith.constant(1)
        ) // warps_per_tok
        stageb_total = cur_rank_num_token * warps_per_tok
        for stageb_idx in range(global_warp_id, stageb_total, global_warp_num):
            tok_id = stageb_idx // warps_per_tok
            part_id = stageb_idx % warps_per_tok
            unit_base = part_id * units_per_warp
            slot0 = fx.Int64(tok_id) * fx.Int64(topk)  # comb_inp[tok*topk + 0]
            expert_addrs = []
            for k_slot in range_constexpr(topk):
                expert_addrs.append(
                    comb_inp_base + (slot0 + fx.Int64(k_slot)) * fx.Int64(slot_stride)
                )
            rem = arith.constant(n_i32) - unit_base
            eff = arith.select(rem < units_per_warp, rem, units_per_warp)
            out_base = tok_id * n_i32

            def _one(off, expert_addrs=expert_addrs, out_base=out_base):
                acc = zero_acc()
                for k_slot in range_constexpr(topk):
                    v = comm_ops.load_i32_nt(expert_addrs[k_slot], off)
                    acc = acc + to_acc(v)
                buffer_store(from_acc(acc), rsrc_out, out_base + off)

            # One vec4 group keeps VGPR low so the grid can fill all CUs. Deeper
            # unrolling increases VGPR pressure and reduces occupancy on this
            # HBM-bandwidth-bound reduce.
            _UNROLL = 1
            VEC = 4
            STEP_CHUNK = WAVE * VEC  # 128 i32 elems/round across the wave
            STEP_V4 = _UNROLL * STEP_CHUNK  # _UNROLL * 128
            main_end = (eff // arith.constant(STEP_V4)) * arith.constant(STEP_V4)
            for u in range(lane * VEC, main_end, STEP_V4):
                base = unit_base + u
                _pre = []
                for _r in range_constexpr(_UNROLL):
                    _off_r = base + _r * STEP_CHUNK
                    _pre.append(
                        [
                            comm_ops.load_v4i32_nt(expert_addrs[k_slot], _off_r)
                            for k_slot in range_constexpr(topk)
                        ]
                    )
                for _r in range_constexpr(_UNROLL):
                    _off = base + _r * STEP_CHUNK
                    _v8bf = T.vec(8, T.bf16)
                    _v8f = T.vec(8, T.f32)
                    _vacc = arith.constant_vector(0.0, _v8f)
                    for k_slot in range_constexpr(topk):
                        _vacc = _vacc + vector.bitcast(_v8bf, _pre[_r][k_slot]).extf(
                            _v8f
                        )
                    _res = vector.bitcast(T.vec(4, T.i32), _vacc.truncf(_v8bf))
                    buffer_store(_res, rsrc_out, out_base + _off)
            for u in range(main_end + lane, eff, WAVE):
                _one(unit_base + u)

        # No exit barrier: the per-block monotonic flag needs no reset, and the
        # reduce does no post-completion work, so kernel retirement (stream-ordered)
        # is the only completion signal the host needs. This removes both the former
        # comb_bar reset (#494) and its (block_num-1)-way contended atomic_add.

    @flyc.jit
    def run(
        arena: Int64,
        addr_comb_inp: Int64,
        addr_xdb_flag: Int64,
        addr_out: Int64,
        my_lsa_rank: Int32,
        cur_rank_num_token: Int32,
        stream=fx.Stream(None),  # noqa: B008
    ):
        ep_combine_fused(
            arena,
            addr_comb_inp,
            addr_xdb_flag,
            addr_out,
            my_lsa_rank,
            cur_rank_num_token,
        ).launch(
            grid=(block_num, 1, 1),
            block=[warp_num_per_block * WAVE, 1, 1],
            stream=stream,
        )

    return run
