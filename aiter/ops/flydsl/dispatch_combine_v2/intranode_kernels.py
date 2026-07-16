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
"""FlyDSL intranode device kernels for the cco-LSA dispatch/combine op.

All kernels here are single-node (cco-LSA P2P over the flat symmetric VA);
internode (RDMA) kernels would live in a separate internode_kernels.py.

Merged factories: dispatch (+scales/replay), combine (gather + scatter/quant),
StdMoE convert (ConvertDispatchOutput/CombineInput), and local expert count.
Each is a compile-time-parameterised @flyc.jit factory; peer addressing goes
through cco.Window(handle).lsa_ptr(pe, off).

Recurring conventions used throughout:
  * `window.lsa_ptr(pe, off)` -> address of peer `pe`'s copy of arena region `off`.
  * `rsrc_*` = a buffer resource descriptor (create_buffer_resource_from_addr).
  * `safe_*` = a value that is the real one on live lanes but a harmless
    in-bounds fallback (0 / self-rank) on dropped lanes, so invalid (duplicate
    or overflow) slots never issue an out-of-bounds load/store.
  * "sentinel" = the dropped-slot marker in tok_map: a (src_tok, k) whose dest
    encodes PE == npes (>= any real PE), telling combine to skip it.
  * "tis" = the per-peer "recv slot -> source token" reverse map; dispatch
    stores the global source token id (rank*max_tok_per_rank + src_tok) there so
    combine/scatter can route each result back to its origin.
  * "xdb" = cross-device barrier: a monotonically bumped i64 flag each rank
    writes into every peer's xdb_mem slot, then spins until its own slot matches
    — an all-ranks handshake that publishes peers' writes before the next stage.
"""
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, const_expr, range_constexpr, T, vector
from flydsl.expr.buffer_ops import (
    buffer_load,
    buffer_store,
    create_buffer_resource_from_addr,
)
from flydsl.expr.rocdl import (
    ballot,
    readlane,
    ds_bpermute,
    fmed3,
    cvt_pk_f32_fp8,
    cvt_pk_fp8_f32,
    cvt_scalef32_pk_f32_fp4,
    cvt_scalef32_pk_fp4_f32,
    cvt_scale_pk8_f32_fp4,
    cvt_scalef32_pk8_fp4_f32,
)
from flydsl.expr.typing import Int32, Int64

import mori.cco.device.flydsl as cco
from . import flydsl_prims as P

# Wavefront size: gfx9 (MI300/MI350) = 64, gfx12 (MI400/gfx1250) = 32. Detected
# once per process; override with MORI_WAVE_SIZE. get_warp_size(gfx1250) wrongly
# reports 64, so key off the arch string.
import os as _os


def _detect_wave_size():
    v = _os.environ.get("MORI_WAVE_SIZE")
    if v:
        return int(v)
    try:
        from mori.jit.config import detect_gpu_arch

        return 32 if detect_gpu_arch().startswith("gfx12") else 64
    except Exception:
        return 64


WAVE = _detect_wave_size()
LANE_MASK = WAVE - 1
LOG2_WAVE = WAVE.bit_length() - 1
_BALLOT_INT = T.i64 if WAVE == 64 else T.i32
_LANE_STRIDE_I32 = WAVE * 4  # one wave of lanes, vec4 (16B) each
_MAIN_STRIDE_I32 = 2 * _LANE_STRIDE_I32
_BUTTERFLY_OFFSETS = tuple(WAVE >> i for i in range(1, LOG2_WAVE + 1))

# ── dispatch ──────────────────────────────────────────────────────────────


def make_dispatch(
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
    off_out_scales=0,
    scale_dim=0,
    scale_type_size=0,
    enable_signal=True,
    replay=False,
    fp4=False,
):
    # fp4 (e2m1) packs 2 values per byte, so a token is hidden_dim/2 bytes; dispatch
    # is a pure byte mover (no fp4 decode), matching mori v1's plain-fp4 path.
    nbytes = hidden_dim // 2 if fp4 else hidden_dim * hidden_elem_size
    n_i32 = nbytes // 4
    # Dropped-slot marker stored in tok_map (see module docstring "sentinel"):
    # its encoded dest_pe (value // max_recv) equals npes, i.e. no real PE.
    sentinel_val = npes * max_recv
    # Optional per-token scales (e.g. fp4/blockwise quant inputs): forwarded
    # verbatim alongside the token to the dest peer's out_scales (mori parity).
    scale_bytes = scale_dim * scale_type_size
    scale_num_i32 = (scale_bytes + 3) // 4
    enable_scales = scale_bytes > 0

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
        addr_inp_scales: Int64,
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
            dest_expert = buffer_load(
                rsrc_inp_idx, work_idx, vec_width=1, dtype=T.i32()
            )
            # Dedup: one token routed to several experts on the SAME dest PE must
            # be sent only once. Each lane l (< k) inspects this token's l-th
            # expert; if a LOWER lane already targets our dest_pe, this
            # (src_tok, k_slot) is a duplicate and gets dropped. safe_lane keeps
            # the probe in-bounds for lanes >= k_slot.
            safe_lane = arith.select(lane < k_slot, lane, 0)
            lane_expert = buffer_load(
                rsrc_inp_idx,
                src_tok * experts_per_token + safe_lane,
                vec_width=1,
                dtype=T.i32(),
            )
            dest_pe = dest_expert // experts_per_rank
            lane_dest_pe = lane_expert // experts_per_rank
            dup_per_lane = arith.select(
                lane_dest_pe == dest_pe, arith.select(lane < k_slot, lane, WAVE), WAVE
            )
            dup_ballot = ballot(_BALLOT_INT(), dup_per_lane < WAVE)
            is_dup = dup_ballot != 0

            if const_expr(replay):
                # decode dest_tok_id from cached tok_map (skip atomic alloc; same layout)
                cached = buffer_load(rsrc_tok_map, work_idx, vec_width=1, dtype=T.i32())
                is_dup_or_overflow = cached >= sentinel_val
                do_publish = cached < sentinel_val
                dest_tok_id = cached - dest_pe * max_recv
            else:
                dest_tok_lane0 = arith.constant(0)
                if lane == 0:
                    if dup_ballot == 0:
                        peer_tok_off = fx.Int64(window.lsa_ptr(dest_pe, off_tok_off))
                        dest_tok_lane0 = P.atomic_add_global(peer_tok_off, fx.Int32(1))
                dest_tok_id = readlane(T.i32(), dest_tok_lane0, 0)
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

            if lane == 0:
                if do_publish:
                    # publish this recv slot's origin into the dest peer's tis
                    # (recv slot -> global source token id) for combine routing.
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
                    P.atomic_add_global(dest_ctr_addr, fx.Int32(1))

            # Per-lane (weight, expert-idx) scatter (lanes < k).
            if lane < experts_per_token:
                if do_publish:
                    weight_src_off = src_tok * experts_per_token + lane
                    weight_val = buffer_load(
                        rsrc_inp_wts, weight_src_off, vec_width=1, dtype=T.f32()
                    )
                    idx_val = buffer_load(
                        rsrc_inp_idx, weight_src_off, vec_width=1, dtype=T.i32()
                    )
                    dest_slot = dest_tok_id * experts_per_token + lane
                    peer_wts = fx.Int64(window.lsa_ptr(dest_pe, off_out_wts))
                    buffer_store(
                        arith.bitcast(T.i32(), weight_val),
                        create_buffer_resource_from_addr(peer_wts),
                        dest_slot,
                    )
                    peer_idx = fx.Int64(window.lsa_ptr(dest_pe, off_out_idx))
                    buffer_store(
                        idx_val, create_buffer_resource_from_addr(peer_idx), dest_slot
                    )

            # Per-token scales scatter: forward the src token's scale_num_i32 dwords
            # to the dest peer's out_scales[dest_tok_id] (lane-strided to cover
            # scale_dim > one wavefront). Verbatim copy (opaque bytes).
            if const_expr(enable_scales):
                if do_publish:
                    rsrc_inp_scales = create_buffer_resource_from_addr(addr_inp_scales)
                    peer_scales = fx.Int64(window.lsa_ptr(dest_pe, off_out_scales))
                    rsrc_peer_scales = create_buffer_resource_from_addr(peer_scales)
                    for k_off in range(lane, scale_num_i32, WAVE):
                        scale_val = buffer_load(
                            rsrc_inp_scales,
                            src_tok * scale_num_i32 + k_off,
                            vec_width=1,
                            dtype=T.i32(),
                        )
                        buffer_store(
                            scale_val,
                            rsrc_peer_scales,
                            dest_tok_id * scale_num_i32 + k_off,
                        )

            # Token-embedding scatter: each lane owns 4 i32 (16B). Two vec4 streams
            # (chunk and chunk+_LANE_STRIDE_I32, stride _MAIN_STRIDE_I32) for
            # memory-level parallelism; a one-stream tail covers the remainder.
            # Dropped slots (dup/overflow) set copy_end == lane_i32_off → no-op.
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
                    vec_a = buffer_load(rsrc_src, chunk, vec_width=4, dtype=T.i32())
                    vec_b = buffer_load(
                        rsrc_src, chunk + _LANE_STRIDE_I32, vec_width=4, dtype=T.i32()
                    )
                    buffer_store(vec_a, rsrc_dst, chunk)
                    buffer_store(vec_b, rsrc_dst, chunk + _LANE_STRIDE_I32)
            if const_expr(safe_end_i32 < n_i32):
                copy_end_tail = arith.select(is_dup_or_overflow, lane_i32_off, n_i32)
                for chunk in range(
                    lane_i32_off + safe_end_i32, copy_end_tail, _LANE_STRIDE_I32
                ):
                    vec_a = buffer_load(rsrc_src, chunk, vec_width=4, dtype=T.i32())
                    buffer_store(vec_a, rsrc_dst, chunk)
            elif const_expr(n_i32 < _MAIN_STRIDE_I32):
                copy_end_small = arith.select(is_dup_or_overflow, lane_i32_off, n_i32)
                for chunk in range(lane_i32_off, copy_end_small, _LANE_STRIDE_I32):
                    vec_a = buffer_load(rsrc_src, chunk, vec_width=4, dtype=T.i32())
                    buffer_store(vec_a, rsrc_dst, chunk)

        if const_expr(enable_signal):
            # Self-reset total_recv (replaces the host-side total_recv.zero_()):
            # only warp 0 accumulates into it in Phase 3, so warp 0 zeros it here
            # and release-fences before the grid barrier; the Phase-2 acquire then
            # orders the Phase-3 atomic adds after this zero. No other warp touches
            # total_recv, so there is no cross-block race.
            if global_warp_id == 0:
                if lane == 0:
                    buffer_store(
                        arith.constant(0),
                        create_buffer_resource_from_addr(addr_total_recv),
                        0,
                    )
                P.fence_system_release()

            # ── Phase 2: grid barrier + per-peer count signal ──
            fx.barrier()
            if tid == 0:
                P.atomic_add_global(fx.Int64(addr_disp_bar), arith.constant(1))

            local_recv_num = fx.Int64(window.lsa_ptr(my_lsa_rank, off_recv_num))
            for dest_pe in range(lane, npes, WAVE):
                if global_warp_id == 0:
                    P.spin_until_eq_i32(fx.Int64(addr_disp_bar), block_num)
                    P.fence_system_acquire()
                    buffer_store(arith.constant(0), rsrc_disp_bar, 0)
                    signal_value = (
                        buffer_load(rsrc_dest_ctr, dest_pe, vec_width=1, dtype=T.i32())
                        + 1
                    )
                    peer_recv_num = fx.Int64(window.lsa_ptr(dest_pe, off_recv_num))
                    recv_num_remote_addr = peer_recv_num + fx.Int64(rank) * fx.Int64(4)
                    P.spin_until_eq_i32(recv_num_remote_addr, 0)
                    P.store_i32_system(
                        recv_num_remote_addr, arith.constant(0), signal_value
                    )

            # ── Phase 3: collect per-source counts into total_recv ──
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
        addr_inp_scales: Int64,
        my_lsa_rank: Int32,
        inp_cur_tok: Int32,
        stream=fx.Stream(None),
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
            addr_inp_scales,
            my_lsa_rank,
            inp_cur_tok,
        ).launch(
            grid=(block_num, 1, 1),
            block=[warp_num_per_block * WAVE, 1, 1],
            stream=stream,
        )

    return run


# ── combine (gather + scatter/quant) ──────────────────────────────────────


def _V2BF16():
    return T.VectorType.get([2], T.bf16())


def _V2F32():
    return T.VectorType.get([2], T.f32())


def _V4F32():
    return T.VectorType.get([4], T.f32())


def _V8F32():
    return T.VectorType.get([8], T.f32())


def _V1I32():
    return T.VectorType.get([1], T.i32())


def _accum_funcs(hidden_elem_size, fp8_direct_cast=False, fp4=False):
    if fp4:  # fp4 e2m1: i32 = 8 packed fp4 -> v8f32
        if WAVE == 32:
            # gfx1250: one pack-8 instr converts all 8 fp4/i32 at once. The
            # gfx950 pk-2 (cvt_scalef32_pk_*_fp4, src/dst_sel) don't exist here
            # ("Cannot select"); dequant only has the E8M0 scale family
            # (cvt_scale_pk8_f32_fp4), quant has scalef32 (cvt_scalef32_pk8_fp4_f32).
            # Plain fp4 (no microscaling) => scale 1.0, scale_sel 0.
            def to_accum(i32_scalar):
                # dequant uses the E8M0 block-scale (i32); 1.0 == 2^(127-127) => 127.
                return cvt_scale_pk8_f32_fp4(
                    res=_V8F32(),
                    src=i32_scalar,
                    scale=arith.constant(127, type=T.i32()),
                    scale_sel=0,
                )

            def from_accum(acc):
                return cvt_scalef32_pk8_fp4_f32(
                    res=T.i32(), src=acc, scale=arith.constant(1.0, type=T.f32())
                )

            def zero_accum():
                return arith.constant_vector(0.0, _V8F32())

            return to_accum, from_accum, zero_accum

        # NOTE: cvt_scalef32_pk_*_fp4 are gfx950-only (MI350). On gfx942
        # (MI300X) codegen fails "instruction not supported on this GPU".
        # Faithful port of the FlyDSL reference fp4 branch; opt-in (fp4=True).
        def to_accum(i32_scalar):
            one = arith.constant(1.0, type=T.f32())
            pairs = [
                cvt_scalef32_pk_f32_fp4(
                    res=_V2F32(), src=i32_scalar, scale=one, src_sel_index=s
                )
                for s in range(4)
            ]
            lo4 = vector.shuffle(pairs[0], pairs[1], [0, 1, 2, 3])
            hi4 = vector.shuffle(pairs[2], pairs[3], [0, 1, 2, 3])
            return vector.shuffle(lo4, hi4, [0, 1, 2, 3, 4, 5, 6, 7])

        def from_accum(acc):
            one = arith.constant(1.0, type=T.f32())
            old = arith.constant(0, type=T.i32())
            for s in range(4):
                fa = vector.extract(acc, static_position=[s * 2])
                fb = vector.extract(acc, static_position=[s * 2 + 1])
                old = cvt_scalef32_pk_fp4_f32(
                    res=T.i32(),
                    old_vdst=old,
                    src0=fa,
                    src1=fb,
                    scale=one,
                    dst_sel_index=s,
                )
            return old

        def zero_accum():
            return arith.constant_vector(0.0, _V8F32())

        return to_accum, from_accum, zero_accum
    return _accum_funcs_int(hidden_elem_size, fp8_direct_cast)


def _accum_funcs_int(hidden_elem_size, fp8_direct_cast=False):
    """Return (to_accum, from_accum, zero_accum) for one i32 'unit' of the
    transport dtype, mirroring the FlyDSL reference's per-dtype branches.

    Each i32 packs: 2 bf16 (v2f32), 1 f32 (scalar), or 4 fp8 (v4f32).
    """
    if hidden_elem_size == 2:  # bf16: i32 = 2 bf16

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

    elif hidden_elem_size == 4:  # f32: i32 = 1 f32

        def to_accum(i32_scalar):
            return fx.Float32(arith.bitcast(T.f32(), arith.unwrap(i32_scalar)))

        def from_accum(acc):
            return fx.Int32(arith.bitcast(T.i32(), arith.unwrap(acc)))

        def zero_accum():
            return fx.Float32(arith.constant(0.0, type=T.f32()))

    elif hidden_elem_size == 1:  # fp8 (OCP e4m3): i32 = 4 fp8

        def to_accum(i32_scalar):
            lo = cvt_pk_f32_fp8(res=_V2F32(), src=i32_scalar, word_sel=False)
            hi = cvt_pk_f32_fp8(res=_V2F32(), src=i32_scalar, word_sel=True)
            return vector.shuffle(lo, hi, [0, 1, 2, 3])

        def from_accum(acc):
            f0 = vector.extract(acc, static_position=[0])
            f1 = vector.extract(acc, static_position=[1])
            f2 = vector.extract(acc, static_position=[2])
            f3 = vector.extract(acc, static_position=[3])
            if fp8_direct_cast:  # wire fp8 -> external bf16: v4f32 -> v4bf16 -> 2 i32
                v4bf16 = acc.truncf(T.VectorType.get([4], T.bf16()))
                return vector.bitcast(T.VectorType.get([2], T.i32()), v4bf16)
            zero = arith.constant(0, type=T.i32())
            lo = cvt_pk_fp8_f32(
                res=T.i32(), src_a=f0, src_b=f1, old=zero, word_sel=False
            )
            return cvt_pk_fp8_f32(
                res=T.i32(), src_a=f2, src_b=f3, old=lo, word_sel=True
            )

        def zero_accum():
            return arith.constant_vector(0.0, _V4F32())

    else:
        raise ValueError(f"unsupported hidden_elem_size {hidden_elem_size}")
    return to_accum, from_accum, zero_accum


def make_combine(
    *,
    rank,
    npes,
    experts_per_token,
    hidden_dim,
    hidden_elem_size,
    max_tok_per_rank,
    max_recv,
    block_num,
    warp_num_per_block,
    off_out_tok,
    off_xdb_mem,
    off_out_wts=0,
    enable_weights=True,
    fp8_direct_cast=False,
    fp4=False,
    reset_total_recv=True,
    _s3_cache=2,
    _unroll=2,
):
    # Transport dtype = external dtype, except fp8_direct_cast wires fp8 while
    # the output (comb_out) stays bf16 (2 i32 per fp8 i32 unit). fp4: i32 = 8 fp4.
    _to_accum2, _from_accum2, _zero_accum = _accum_funcs(
        hidden_elem_size, fp8_direct_cast, fp4
    )
    nbytes = hidden_dim // 2 if fp4 else hidden_dim * hidden_elem_size
    n_i32 = hidden_dim // 8 if fp4 else nbytes // 4
    # fp8_direct_cast: output stride is bf16 (2 i32 per fp8 unit) vs input fp8.
    out_n_i32 = (hidden_dim * 2) // 4 if fp8_direct_cast else n_i32
    # vec4 gather: 4 i32 per load (global_load_dwordx4) when output is 1:1.
    _use_vec4 = (n_i32 % 4 == 0) and not fp8_direct_cast
    out_step_mult = 2 if fp8_direct_cast else 1

    @flyc.kernel(known_block_size=[warp_num_per_block * WAVE, 1, 1])
    def ep_combine(
        arena: Int64,
        addr_tok_map: Int64,
        addr_comb_bar: Int64,
        addr_xdb_flag: Int64,
        addr_total_recv: Int64,
        addr_out: Int64,
        addr_out_wts: Int64,
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
        rsrc_tok_map = create_buffer_resource_from_addr(addr_tok_map)
        rsrc_comb_bar = create_buffer_resource_from_addr(addr_comb_bar)
        rsrc_total_recv = create_buffer_resource_from_addr(addr_total_recv)
        rsrc_out = create_buffer_resource_from_addr(addr_out)
        xdb_cur_flag = P.load_i64_acquire(fx.Int64(addr_xdb_flag))

        # ── Stage 1: cross-device entry barrier (xdb, see module docstring) ──
        # Gather reads only out_tok from the prior dispatch/expert kernel, so a
        # cross-device barrier suffices (no Stage-1 scatter). The grid barrier
        # (comb_bar) is REQUIRED: it guarantees every block has already read
        # xdb_cur_flag before block 0 increments xdb_flag — otherwise a slow
        # block reads the bumped flag and the per-peer == handshake deadlocks.
        fx.barrier()
        if tid == 0:
            P.atomic_add_global(fx.Int64(addr_comb_bar), arith.constant(1))
        if grid_thread_id < npes:
            P.spin_until_eq_i32(fx.Int64(addr_comb_bar), block_num)
            P.fence_system_acquire()
            buffer_store(arith.constant(0), rsrc_comb_bar, 0)
            xdb_remote = fx.Int64(
                window.lsa_ptr(grid_thread_id, off_xdb_mem)
            ) + fx.Int64(rank) * fx.Int64(8)
            P.store_i64_system(xdb_remote, arith.constant(0), xdb_cur_flag)
        if grid_thread_id == 0:
            P.atomic_add_global(
                fx.Int64(addr_xdb_flag), arith.constant(1, type=T.i64())
            )
        if tid < npes:
            xdb_peer_slot = fx.Int64(
                window.lsa_ptr(my_lsa_rank, off_xdb_mem)
            ) + fx.Int64(tid) * fx.Int64(8)
            P.spin_until_eq_i64(xdb_peer_slot, xdb_cur_flag)
            P.fence_system_acquire()
        fx.barrier()
        P.fence_system_acquire()  # ALL threads: peers' out_tok visible
        if const_expr(reset_total_recv):
            if tid == 0:
                buffer_store(arith.constant(0), rsrc_total_recv, 0)

        rsrc_out_wts = create_buffer_resource_from_addr(addr_out_wts)

        # ── Stage 2: warp-partitioned remote gather + f32 accumulate ──
        # Register-light i32 (2 bf16, v2f32) reads + `_unroll`-way unroll: each
        # lane keeps `_unroll` independent loads/accumulators in flight per k so
        # a warp hides xGMI read latency with fewer warps (mori WarpAccum,
        # VecBytes=4, Unroll=2). Partition each token's hidden across
        # warps_per_tok warps so small batches still fill the grid.
        STEP = _unroll * WAVE
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
        stage3_total = cur_rank_num_token * warps_per_tok
        for stage3_idx in range(global_warp_id, stage3_total, global_warp_num):
            tok_id = stage3_idx // warps_per_tok
            part_id = stage3_idx % warps_per_tok
            unit_base = part_id * units_per_warp
            tok_map_base = tok_id * experts_per_token
            expert_bases = []
            expert_valids = []
            expert_pes = []
            expert_toks = []
            for k_slot in range_constexpr(experts_per_token):
                encoded_k = buffer_load(
                    rsrc_tok_map, tok_map_base + k_slot, vec_width=1, dtype=T.i32()
                )
                dest_pe_k = encoded_k // max_recv  # sentinel: dest_pe == npes
                dest_tok_k = encoded_k % max_recv  # peer-local recv slot
                valid_k = dest_pe_k < npes
                safe_pe = arith.select(valid_k, dest_pe_k, arith.constant(rank))
                safe_tok_k = arith.select(valid_k, dest_tok_k, arith.constant(0))
                # Remote base peer[dest_pe].out_tok[dest_tok]; gather via global
                # loads (no per-expert buffer descriptor) to keep all K loads in
                # flight — see P.load_i32_nt.
                slot_addr = fx.Int64(window.lsa_ptr(safe_pe, off_out_tok)) + fx.Int64(
                    safe_tok_k
                ) * fx.Int64(nbytes)
                expert_bases.append(slot_addr)
                expert_valids.append(valid_k)
                expert_pes.append(safe_pe)
                expert_toks.append(safe_tok_k)

            # Weights (mori UseWeights): once per token (part 0), reduce the K
            # forwarded weight vectors -> out_weights[tok][e]. Reuses the decode
            # above and overlaps with this warp's hidden gather.
            if const_expr(enable_weights):
                if part_id == arith.constant(0):
                    if lane < experts_per_token:
                        weight_acc = arith.constant(0.0, type=T.f32())
                        for k_slot in range_constexpr(experts_per_token):
                            weight_addr = fx.Int64(
                                window.lsa_ptr(expert_pes[k_slot], off_out_wts)
                            ) + (
                                fx.Int64(expert_toks[k_slot])
                                * fx.Int64(experts_per_token)
                                + fx.Int64(lane)
                            ) * fx.Int64(
                                4
                            )
                            weight_val = buffer_load(
                                create_buffer_resource_from_addr(weight_addr),
                                0,
                                vec_width=1,
                                dtype=T.f32(),
                            )
                            weight_acc = weight_acc + arith.select(
                                expert_valids[k_slot],
                                weight_val,
                                arith.constant(0.0, type=T.f32()),
                            )
                        buffer_store(weight_acc, rsrc_out_wts, tok_map_base + lane)
            rem = arith.constant(n_i32) - unit_base
            eff = arith.select(
                rem < units_per_warp, rem, units_per_warp
            )  # i32 units this warp
            out_base = tok_id * out_n_i32

            # Nested fn: closure over expert_bases/valids (lists can't be loop-carried).
            def _one(off):  # reduce k contributions for one i32 unit
                vals = []
                for k_slot in range_constexpr(experts_per_token):
                    v = P.load_i32_nt(expert_bases[k_slot], off)
                    vals.append(
                        arith.select(expert_valids[k_slot], v, arith.constant(0))
                    )
                acc = _zero_accum()
                for k_slot in range_constexpr(experts_per_token):
                    acc = acc + _to_accum2(vals[k_slot])
                buffer_store(
                    _from_accum2(acc), rsrc_out, out_base + off * out_step_mult
                )

            if const_expr(_use_vec4):
                # Vec4 path: load 4 i32 (16B, global_load_dwordx4) per expert
                # per iteration, reducing xGMI load count 4×.
                VEC = 4
                STEP_CHUNK = WAVE * VEC
                STEP_V4 = _unroll * STEP_CHUNK

                def _load_expert_vecs(off):
                    vecs = []
                    valids = []
                    for k_slot in range_constexpr(experts_per_token):
                        vecs.append(P.load_v4i32_nt(expert_bases[k_slot], off))
                        valids.append(expert_valids[k_slot])
                    return vecs, valids

                def _accum_loop():
                    main_end = (eff // STEP_V4) * STEP_V4
                    for u in range(lane * VEC, main_end, STEP_V4):
                        base = unit_base + u
                        # Issue all remote loads first (both unroll rounds) so stores
                        # in the j/r nest cannot block the next load batch.
                        pre_vecs = []
                        pre_valids = []
                        for r in range_constexpr(_unroll):
                            off_r = base + r * STEP_CHUNK
                            vecs_r, valids_r = _load_expert_vecs(off_r)
                            pre_vecs.append(vecs_r)
                            pre_valids.append(valids_r)
                        # Reduce+store: j outer, r inner (unroll innermost).
                        for j in range_constexpr(VEC):
                            for r in range_constexpr(_unroll):
                                off = base + r * STEP_CHUNK
                                acc = _zero_accum()
                                for k_slot in range_constexpr(experts_per_token):
                                    elem = vector.extract(
                                        pre_vecs[r][k_slot], static_position=[j]
                                    )
                                    v = arith.select(
                                        pre_valids[r][k_slot], elem, arith.constant(0)
                                    )
                                    acc = acc + _to_accum2(v)
                                buffer_store(
                                    _from_accum2(acc), rsrc_out, out_base + off + j
                                )
                    for u in range(main_end + lane, eff, WAVE):
                        _one(unit_base + u)

            else:

                def _accum_loop():
                    main_end = (eff // STEP) * STEP
                    for u in range(lane, main_end, STEP):
                        base = unit_base + u
                        pre_vals = []
                        for r in range_constexpr(_unroll):
                            off_r = base + r * WAVE
                            vals_r = []
                            for k_slot in range_constexpr(experts_per_token):
                                v = P.load_i32_nt(expert_bases[k_slot], off_r)
                                vals_r.append(
                                    arith.select(
                                        expert_valids[k_slot], v, arith.constant(0)
                                    )
                                )
                            pre_vals.append(vals_r)
                        for r in range_constexpr(_unroll):
                            off = base + r * WAVE
                            acc = _zero_accum()
                            for k_slot in range_constexpr(experts_per_token):
                                acc = acc + _to_accum2(pre_vals[r][k_slot])
                            buffer_store(
                                _from_accum2(acc),
                                rsrc_out,
                                out_base + off * out_step_mult,
                            )
                    for u in range(main_end + lane, eff, WAVE):
                        _one(unit_base + u)

            _accum_loop()

    @flyc.jit
    def run(
        arena: Int64,
        addr_tok_map: Int64,
        addr_comb_bar: Int64,
        addr_xdb_flag: Int64,
        addr_total_recv: Int64,
        addr_out: Int64,
        addr_out_wts: Int64,
        my_lsa_rank: Int32,
        cur_rank_num_token: Int32,
        stream=fx.Stream(None),
    ):
        ep_combine(
            arena,
            addr_tok_map,
            addr_comb_bar,
            addr_xdb_flag,
            addr_total_recv,
            addr_out,
            addr_out_wts,
            my_lsa_rank,
            cur_rank_num_token,
        ).launch(
            grid=(block_num, 1, 1),
            block=[warp_num_per_block * WAVE, 1, 1],
            stream=stream,
        )

    return run


_FP8_MAX = 240.0  # gfx942 native fp8 is e4m3fnuz: max finite 240 (NOT OCP 448);
# clamping above 240 yields NaN from cvt_pk_fp8_f32 on this arch


def _fabs(f):
    return arith.maximumf(f, arith.negf(f))


def _bf16x2(i32_scalar):  # i32 (2 bf16) -> v2f32
    return vector.bitcast(_V2BF16(), vector.from_elements(_V1I32(), [i32_scalar])).extf(
        _V2F32()
    )


def _warp_amax(lane, v):
    """Max-reduce an f32 across the wavefront (butterfly via ds_bpermute, per-lane
    gather index). Every lane returns the wavefront max; raw values in/out."""
    for off in _BUTTERFLY_OFFSETS:
        idx = arith.unwrap((lane ^ off) * 4)  # byte addr = lane*4
        o = ds_bpermute(T.i32(), idx, arith.bitcast(T.i32(), arith.unwrap(v)))
        v = arith.maximumf(v, arith.bitcast(T.f32(), o))
    return v


def make_combine_scatter(
    *,
    rank,
    npes,
    experts_per_token,
    hidden_dim,
    hidden_elem_size,
    max_tok_per_rank,
    max_recv,
    block_num,
    warp_num_per_block,
    off_out_tok,
    off_comb_inp,
    off_tis,
    off_xdb_mem,
    off_out_wts=0,
    off_comb_wts=0,
    off_comb_scales=0,
    enable_weights=True,
    fp8_direct_cast=False,
    fp8_blockwise=False,
    scale_dim=0,
    reset_total_recv=True,
    _s3_cache=2,
):
    """Scatter combine (mori useExternalInpBuffer / _nop2p path).

    Stage 1  each computing rank P2P-WRITES its post-expert tokens back to the
             ORIGIN rank's comb_inp[computing_rank*M + origin_lid] (origin from
             tis); under fp8_direct_cast the bf16 token is cast to fp8 on write.
    Stage 2  cross-device barrier.
    Stage 3  origin rank LOCAL-reads comb_inp[dest_pe*M + tok] for its token's k
             expert PEs (from tok_map) and reduces (fp8->bf16 dequant if cast).

    vs the gather path: 2 passes (remote write + local read) but compresses the
    transport to fp8; the natural home for fp8_direct_cast (gather has no
    Stage-1 writer to compress at)."""
    if fp8_blockwise and WAVE != 64:
        raise NotImplementedError(
            "fp8_blockwise combine's coalesced path assumes wave64 (one wave == one "
            "128-elem block); wave32 (gfx12) port is TODO"
        )
    # blockwise reuses the fp8->bf16 accum (output bf16, wire fp8) + per-block scale.
    _fp8_out = fp8_direct_cast or fp8_blockwise
    wire_elem_size = 1 if _fp8_out else hidden_elem_size
    to_acc, from_acc, zero_acc = _accum_funcs(wire_elem_size, _fp8_out)
    inp_nbytes = hidden_dim * hidden_elem_size  # source out_tok (bf16/f32)
    wire_nbytes = hidden_dim * wire_elem_size  # comb_inp transport
    wire_n_i32 = wire_nbytes // 4
    out_n_i32 = (hidden_dim * 2) // 4 if _fp8_out else wire_n_i32
    out_step_mult = 2 if _fp8_out else 1
    if fp8_blockwise:
        block_elems = hidden_dim // scale_dim
        assert (
            hidden_dim % scale_dim == 0 and block_elems == 128
        ), "blockwise (coalesced path): block_elems must be 128 (scale_dim = hidden/128)"
        block_i32_fp8 = block_elems // 4  # fp8 i32 units per block (=32)

    @flyc.kernel(known_block_size=[warp_num_per_block * WAVE, 1, 1])
    def ep_combine_s(
        arena: Int64,
        addr_tok_map: Int64,
        addr_comb_bar: Int64,
        addr_xdb_flag: Int64,
        addr_total_recv: Int64,
        addr_out: Int64,
        addr_out_wts: Int64,
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
        rsrc_tok_map = create_buffer_resource_from_addr(addr_tok_map)
        rsrc_comb_bar = create_buffer_resource_from_addr(addr_comb_bar)
        rsrc_total_recv = create_buffer_resource_from_addr(addr_total_recv)
        rsrc_tis = create_buffer_resource_from_addr(
            fx.Int64(window.lsa_ptr(my_lsa_rank, off_tis))
        )
        rsrc_out = create_buffer_resource_from_addr(addr_out)
        rsrc_out_wts = create_buffer_resource_from_addr(addr_out_wts)
        xdb_cur_flag = P.load_i64_acquire(fx.Int64(addr_xdb_flag))
        total_recv = buffer_load(rsrc_total_recv, 0, vec_width=1, dtype=T.i32())

        # ── Stage 1: scatter post-expert tokens back to origin's comb_inp ──
        src_tok_base = fx.Int64(window.lsa_ptr(my_lsa_rank, off_out_tok))
        for recv_slot in range(global_warp_id, total_recv, global_warp_num):
            # tis encodes origin = src_pe*max_tok_per_rank + local_id
            encoded_origin = buffer_load(
                rsrc_tis, recv_slot, vec_width=1, dtype=T.i32()
            )
            origin_pe = encoded_origin // max_tok_per_rank
            origin_lid = encoded_origin % max_tok_per_rank
            dst = fx.Int64(window.lsa_ptr(origin_pe, off_comb_inp)) + (
                fx.Int64(rank * max_tok_per_rank + origin_lid)
            ) * fx.Int64(wire_nbytes)
            src = src_tok_base + fx.Int64(recv_slot) * fx.Int64(inp_nbytes)
            rsrc_src = create_buffer_resource_from_addr(src)
            rsrc_dst = create_buffer_resource_from_addr(dst)
            if const_expr(fp8_blockwise):
                # Blockwise fp8 quant, COALESCED + warp-reduce (block_elems==128
                # so one i32/lane spans a full block). Per block scale_block:
                # lanes load the block coalesced (lane l -> elems 2l,2l+1),
                # ds_bpermute butterfly gives every lane the block amax, then
                # lane-pairs combine their 2 fp8 each into one i32 (even lane
                # writes, coalesced). scale = (amax>MAX)? amax/MAX : 1;
                # quant = clamp(v*MAX/amax). Token sign sentinel: if ANY block
                # scaled, negate block-0 scale.
                scale_dst = fx.Int64(
                    window.lsa_ptr(origin_pe, off_comb_scales)
                ) + fx.Int64(rank * max_tok_per_rank + origin_lid) * fx.Int64(
                    scale_dim
                ) * fx.Int64(
                    4
                )
                rsrc_scales = create_buffer_resource_from_addr(scale_dst)
                fp8max = arith.constant(_FP8_MAX, type=T.f32())
                nlim = arith.constant(-_FP8_MAX, type=T.f32())
                any_scaled = arith.constant(0) != arith.constant(0)  # False (uniform)
                for scale_block in range_constexpr(scale_dim):
                    v2 = _bf16x2(
                        buffer_load(
                            rsrc_src,
                            scale_block * 64 + lane,
                            vec_width=1,
                            dtype=T.i32(),
                        )
                    )
                    e0 = vector.extract(v2, static_position=[0])
                    e1 = vector.extract(v2, static_position=[1])
                    amax = _warp_amax(lane, arith.maximumf(_fabs(e0), _fabs(e1)))
                    scaled = amax > fp8max
                    any_scaled = arith.select(scaled, scaled, any_scaled)
                    scale = arith.select(
                        scaled,
                        arith.divf(amax, fp8max),
                        arith.constant(1.0, type=T.f32()),
                    )
                    inv = arith.select(
                        scaled,
                        arith.divf(fp8max, amax),
                        arith.constant(1.0, type=T.f32()),
                    )
                    if lane == 0:
                        buffer_store(scale, rsrc_scales, scale_block)
                    f0 = fmed3(T.f32(), arith.mulf(e0, inv), fp8max, nlim)
                    f1 = fmed3(T.f32(), arith.mulf(e1, inv), fp8max, nlim)
                    my_packed = cvt_pk_fp8_f32(
                        res=T.i32(),
                        src_a=f0,
                        src_b=f1,
                        old=arith.constant(0, type=T.i32()),
                        word_sel=False,
                    )
                    # neighbour (lane^1)'s 2 fp8 (its low 16) via ds_bpermute.
                    nbr_packed = ds_bpermute(
                        T.i32(),
                        arith.unwrap((lane ^ arith.constant(1)) * arith.constant(4)),
                        arith.unwrap(my_packed),
                    )
                    my_lo16 = my_packed & arith.constant(0xFFFF)
                    packed_pair = my_lo16 | (
                        (nbr_packed & arith.constant(0xFFFF)) << arith.constant(16)
                    )
                    if (lane & arith.constant(1)) == arith.constant(0):
                        buffer_store(
                            packed_pair,
                            rsrc_dst,
                            scale_block * block_i32_fp8 + (lane >> arith.constant(1)),
                        )
                if any_scaled:
                    if lane == 0:
                        s0 = buffer_load(rsrc_scales, 0, vec_width=1, dtype=T.f32())
                        buffer_store(arith.negf(s0), rsrc_scales, 0)
            elif const_expr(fp8_direct_cast):
                # 2 bf16 i32 -> v4f32 -> cvt_pk_fp8 x2 -> 1 fp8 i32
                for elem in range(lane, wire_n_i32, WAVE):
                    bf = buffer_load(rsrc_src, elem * 2, vec_width=2, dtype=T.i32())
                    v4 = vector.bitcast(T.VectorType.get([4], T.bf16()), bf).extf(
                        _V4F32()
                    )
                    f0 = vector.extract(v4, static_position=[0])
                    f1 = vector.extract(v4, static_position=[1])
                    f2 = vector.extract(v4, static_position=[2])
                    f3 = vector.extract(v4, static_position=[3])
                    z = arith.constant(0, type=T.i32())
                    lo = cvt_pk_fp8_f32(
                        res=T.i32(), src_a=f0, src_b=f1, old=z, word_sel=False
                    )
                    fp8 = cvt_pk_fp8_f32(
                        res=T.i32(), src_a=f2, src_b=f3, old=lo, word_sel=True
                    )
                    buffer_store(fp8, rsrc_dst, elem)
            else:
                for elem in range(lane, wire_n_i32, WAVE):
                    v = buffer_load(rsrc_src, elem, vec_width=1, dtype=T.i32())
                    buffer_store(v, rsrc_dst, elem)
            if const_expr(enable_weights):
                # forward this recv slot's weights (dispatch put them in out_wts[recv_slot])
                # to the ORIGIN's comb_wts[computing_rank*M + lid] (dedicated
                # region; reusing out_wts would collide with dispatch's layout).
                weight_src = fx.Int64(
                    window.lsa_ptr(my_lsa_rank, off_out_wts)
                ) + fx.Int64(recv_slot) * fx.Int64(experts_per_token) * fx.Int64(4)
                weight_dst = fx.Int64(
                    window.lsa_ptr(origin_pe, off_comb_wts)
                ) + fx.Int64(rank * max_tok_per_rank + origin_lid) * fx.Int64(
                    experts_per_token
                ) * fx.Int64(
                    4
                )
                if lane < experts_per_token:
                    weight_val = buffer_load(
                        create_buffer_resource_from_addr(weight_src),
                        lane,
                        vec_width=1,
                        dtype=T.i32(),
                    )
                    buffer_store(
                        weight_val, create_buffer_resource_from_addr(weight_dst), lane
                    )

        # ── Stage 2: cross-device barrier ──
        # release Stage-1's P2P comb_inp writes before signaling peers (else
        # Stage-3's acquire races them -> dropped contributions)
        P.fence_system_release()
        fx.barrier()
        if tid == 0:
            P.atomic_add_global(fx.Int64(addr_comb_bar), arith.constant(1))
        if grid_thread_id < npes:
            P.spin_until_eq_i32(fx.Int64(addr_comb_bar), block_num)
            P.fence_system_acquire()
            buffer_store(arith.constant(0), rsrc_comb_bar, 0)
            xdb_remote = fx.Int64(
                window.lsa_ptr(grid_thread_id, off_xdb_mem)
            ) + fx.Int64(rank) * fx.Int64(8)
            P.store_i64_system(xdb_remote, arith.constant(0), xdb_cur_flag)
        if grid_thread_id == 0:
            P.atomic_add_global(
                fx.Int64(addr_xdb_flag), arith.constant(1, type=T.i64())
            )
        if tid < npes:
            xdb_slot = fx.Int64(window.lsa_ptr(my_lsa_rank, off_xdb_mem)) + fx.Int64(
                tid
            ) * fx.Int64(8)
            P.spin_until_eq_i64(xdb_slot, xdb_cur_flag)
            P.fence_system_acquire()
        fx.barrier()
        P.fence_system_acquire()
        if const_expr(reset_total_recv):
            if tid == 0:
                buffer_store(arith.constant(0), rsrc_total_recv, 0)

        # ── Stage 3: local read of comb_inp + reduce ──
        comb_inp_base = fx.Int64(window.lsa_ptr(my_lsa_rank, off_comb_inp))
        safe_tok = arith.select(
            cur_rank_num_token == arith.constant(0),
            arith.constant(1),
            cur_rank_num_token,
        )
        warps_per_tok = (
            arith.constant(global_warp_num) + safe_tok - arith.constant(1)
        ) // safe_tok
        units_per_warp = (
            arith.constant(wire_n_i32) + warps_per_tok - arith.constant(1)
        ) // warps_per_tok
        stage3_total = cur_rank_num_token * warps_per_tok
        for stage3_idx in range(global_warp_id, stage3_total, global_warp_num):
            tok_id = stage3_idx // warps_per_tok
            part_id = stage3_idx % warps_per_tok
            unit_base = part_id * units_per_warp
            tok_map_base = tok_id * experts_per_token
            expert_rsrcs = []
            expert_valids = []
            expert_pes = []
            expert_scales = []
            for k_slot in range_constexpr(experts_per_token):
                encoded_k = buffer_load(
                    rsrc_tok_map, tok_map_base + k_slot, vec_width=1, dtype=T.i32()
                )
                dest_pe = encoded_k // max_recv
                valid = dest_pe < npes
                safe_pe = arith.select(valid, dest_pe, arith.constant(rank))
                # LOCAL comb_inp[computing_pe*M + tok_id]
                src_addr = comb_inp_base + (
                    fx.Int64(safe_pe) * fx.Int64(max_tok_per_rank) + fx.Int64(tok_id)
                ) * fx.Int64(wire_nbytes)
                expert_rsrcs.append(create_buffer_resource_from_addr(src_addr))
                expert_valids.append(valid)
                expert_pes.append(safe_pe)
                if const_expr(fp8_blockwise):
                    scale_addr = fx.Int64(
                        window.lsa_ptr(my_lsa_rank, off_comb_scales)
                    ) + (
                        fx.Int64(safe_pe) * fx.Int64(max_tok_per_rank)
                        + fx.Int64(tok_id)
                    ) * fx.Int64(
                        scale_dim
                    ) * fx.Int64(
                        4
                    )
                    expert_scales.append(create_buffer_resource_from_addr(scale_addr))
            if const_expr(enable_weights):
                if part_id == arith.constant(0):
                    if lane < experts_per_token:
                        weight_acc = arith.constant(0.0, type=T.f32())
                        for k_slot in range_constexpr(experts_per_token):
                            # LOCAL comb_wts[computing_pe*M + tok_id] (scattered in S1)
                            weight_addr = (
                                fx.Int64(window.lsa_ptr(my_lsa_rank, off_comb_wts))
                                + (
                                    fx.Int64(expert_pes[k_slot])
                                    * fx.Int64(max_tok_per_rank)
                                    + fx.Int64(tok_id)
                                )
                                * fx.Int64(experts_per_token)
                                * fx.Int64(4)
                                + fx.Int64(lane) * fx.Int64(4)
                            )
                            weight_val = buffer_load(
                                create_buffer_resource_from_addr(weight_addr),
                                0,
                                vec_width=1,
                                dtype=T.f32(),
                            )
                            weight_acc = weight_acc + arith.select(
                                expert_valids[k_slot],
                                weight_val,
                                arith.constant(0.0, type=T.f32()),
                            )
                        buffer_store(weight_acc, rsrc_out_wts, tok_map_base + lane)
            rem = arith.constant(wire_n_i32) - unit_base
            eff = arith.select(rem < units_per_warp, rem, units_per_warp)
            out_base = tok_id * out_n_i32

            def _one(off):
                acc = zero_acc()
                # blockwise: 4 fp8 per i32 unit are 4 consecutive elements in the
                # same block -> one scale per unit. scale_block = (off*4)//block_elems.
                if const_expr(fp8_blockwise):
                    scale_block = (off * arith.constant(4)) // arith.constant(
                        block_elems
                    )
                    is_b0 = scale_block == arith.constant(0)
                for k_slot in range_constexpr(experts_per_token):
                    v = buffer_load(
                        expert_rsrcs[k_slot],
                        off,
                        vec_width=1,
                        dtype=T.i32(),
                        cache_modifier=_s3_cache,
                    )
                    v = arith.select(expert_valids[k_slot], v, arith.constant(0))
                    if const_expr(fp8_blockwise):
                        scale = buffer_load(
                            expert_scales[k_slot],
                            scale_block,
                            vec_width=1,
                            dtype=T.f32(),
                        )
                        scale = arith.select(
                            is_b0, _fabs(scale), scale
                        )  # undo block-0 sign sentinel
                        # invalid expert -> v already 0; force finite scale so a
                        # stale/garbage comb_scales slot can't make 0*NaN = NaN.
                        scale = arith.select(
                            expert_valids[k_slot],
                            scale,
                            arith.constant(1.0, type=T.f32()),
                        )
                        acc = acc + to_acc(v) * scale
                    else:
                        acc = acc + to_acc(v)
                buffer_store(from_acc(acc), rsrc_out, out_base + off * out_step_mult)

            def _loop():
                for u in range(lane, eff, WAVE):
                    _one(unit_base + u)

            _loop()

    @flyc.jit
    def run(
        arena: Int64,
        addr_tok_map: Int64,
        addr_comb_bar: Int64,
        addr_xdb_flag: Int64,
        addr_total_recv: Int64,
        addr_out: Int64,
        addr_out_wts: Int64,
        my_lsa_rank: Int32,
        cur_rank_num_token: Int32,
        stream=fx.Stream(None),
    ):
        ep_combine_s(
            arena,
            addr_tok_map,
            addr_comb_bar,
            addr_xdb_flag,
            addr_total_recv,
            addr_out,
            addr_out_wts,
            my_lsa_rank,
            cur_rank_num_token,
        ).launch(
            grid=(block_num, 1, 1),
            block=[warp_num_per_block * WAVE, 1, 1],
            stream=stream,
        )

    return run


# ── StdMoE convert ────────────────────────────────────────────────────────


def _to_accum2(i32_scalar):  # i32 (2 bf16) -> v2f32
    return vector.bitcast(_V2BF16(), vector.from_elements(_V1I32(), [i32_scalar])).extf(
        _V2F32()
    )


def _from_accum2(v2f32):  # v2f32 -> i32 (2 bf16)
    return vector.extract(
        vector.bitcast(_V1I32(), v2f32.truncf(_V2BF16())), static_position=[0]
    )


def _splat2(f32_scalar):  # f32 -> v2f32 (broadcast)
    return vector.from_elements(_V2F32(), [f32_scalar, f32_scalar])


def make_convert_dispatch_output(
    *,
    rank,
    experts_per_rank,
    experts_per_token,
    hidden_dim,
    hidden_elem_size,
    max_tok_per_expert,
    block_num,
    warp_num_per_block,
):
    """Per-expert packing of dispatched tokens. One warp per (recv_tok, k_slot):
    lane0 bumps packed_cnt[localExpert] to claim a slot, then the warp copies the
    token embedding into packed_x[slot]."""
    assert hidden_elem_size == 2, "stdmoe convert is bf16-only"
    nbytes = hidden_dim * hidden_elem_size
    n_i32 = nbytes // 4

    @flyc.kernel(known_block_size=[warp_num_per_block * WAVE, 1, 1])
    def convert_disp(
        addr_out_tok: Int64,
        addr_out_idx: Int64,
        addr_tis: Int64,
        addr_total_recv: Int64,
        addr_packed_x: Int64,
        addr_packed_cnt: Int64,
        addr_packed_src: Int64,
        addr_slot_map: Int64,
    ):
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        lane = tid & LANE_MASK
        warp = tid >> LOG2_WAVE
        global_warp_id = bid * warp_num_per_block + warp
        global_warp_num = block_num * warp_num_per_block

        rsrc_out_idx = create_buffer_resource_from_addr(addr_out_idx)
        rsrc_tis = create_buffer_resource_from_addr(addr_tis)
        rsrc_total_recv = create_buffer_resource_from_addr(addr_total_recv)
        rsrc_packed_src = create_buffer_resource_from_addr(addr_packed_src)

        total_recv = buffer_load(rsrc_total_recv, 0, vec_width=1, dtype=T.i32())
        work_limit = total_recv * experts_per_token
        for i in range(global_warp_id, work_limit, global_warp_num):
            recv_tok = i // experts_per_token
            expert = buffer_load(rsrc_out_idx, i, vec_width=1, dtype=T.i32())
            local_expert = expert - arith.constant(rank * experts_per_rank)
            # MUST be unsigned ult: signed slt would treat negative local_expert
            # (non-local experts) as in-range and trigger an OOB copy.
            is_local = arith.cmpi(
                arith.CmpIPredicate.ult, local_expert, arith.constant(experts_per_rank)
            )
            # lane0 claims a per-expert packing slot, then broadcasts.
            slot_lane0 = arith.constant(0)
            if lane == 0:
                if is_local:
                    cnt_addr = fx.Int64(addr_packed_cnt) + fx.Int64(
                        local_expert
                    ) * fx.Int64(4)
                    slot_lane0 = P.atomic_add_global(cnt_addr, fx.Int32(1))
            slot = readlane(T.i32(), slot_lane0, 0)
            safe_local_expert = arith.select(is_local, local_expert, arith.constant(0))
            packed_lin_idx = (
                safe_local_expert * arith.constant(max_tok_per_expert) + slot
            )
            slot_val = arith.select(
                is_local, fx.Int64(packed_lin_idx), arith.constant(-1, type=T.i64())
            )
            if lane == 0:
                P.store_i64_system(
                    fx.Int64(addr_slot_map) + fx.Int64(i) * fx.Int64(8),
                    arith.constant(0),
                    slot_val,
                )
                if is_local:
                    src = buffer_load(rsrc_tis, recv_tok, vec_width=1, dtype=T.i32())
                    buffer_store(src, rsrc_packed_src, packed_lin_idx)
            if is_local:
                dst = fx.Int64(addr_packed_x) + fx.Int64(packed_lin_idx) * fx.Int64(
                    nbytes
                )
                src_t = fx.Int64(addr_out_tok) + fx.Int64(recv_tok) * fx.Int64(nbytes)
                rsrc_dst = create_buffer_resource_from_addr(dst)
                rsrc_src = create_buffer_resource_from_addr(src_t)
                for chunk in range(lane * 4, n_i32, _LANE_STRIDE_I32):
                    v = buffer_load(rsrc_src, chunk, vec_width=4, dtype=T.i32())
                    buffer_store(v, rsrc_dst, chunk)

    @flyc.jit
    def run(
        addr_out_tok: Int64,
        addr_out_idx: Int64,
        addr_tis: Int64,
        addr_total_recv: Int64,
        addr_packed_x: Int64,
        addr_packed_cnt: Int64,
        addr_packed_src: Int64,
        addr_slot_map: Int64,
        stream=fx.Stream(None),
    ):
        convert_disp(
            addr_out_tok,
            addr_out_idx,
            addr_tis,
            addr_total_recv,
            addr_packed_x,
            addr_packed_cnt,
            addr_packed_src,
            addr_slot_map,
        ).launch(
            grid=(block_num, 1, 1),
            block=[warp_num_per_block * WAVE, 1, 1],
            stream=stream,
        )

    return run


def make_convert_combine_input(
    *,
    rank,
    experts_per_rank,
    experts_per_token,
    hidden_dim,
    hidden_elem_size,
    max_tok_per_expert,
    block_num,
    warp_num_per_block,
):
    """Inverse: reduce each recv token's local-expert outputs with routing
    weights back into out_tok (the combine input staging). Warp-partitioned over
    hidden; out_tok[recv_t][e] = sum_{k: slot_k valid} wts[k]*packed_x[slot_k][e]."""
    assert hidden_elem_size == 2, "stdmoe convert is bf16-only"
    nbytes = hidden_dim * hidden_elem_size
    n_i32 = nbytes // 4

    @flyc.kernel(known_block_size=[warp_num_per_block * WAVE, 1, 1])
    def convert_comb(
        addr_out_tok: Int64,
        addr_out_wts: Int64,
        addr_total_recv: Int64,
        addr_packed_x: Int64,
        addr_slot_map: Int64,
    ):
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        lane = tid & LANE_MASK
        warp = tid >> LOG2_WAVE
        global_warp_id = bid * warp_num_per_block + warp
        global_warp_num = block_num * warp_num_per_block

        rsrc_out_wts = create_buffer_resource_from_addr(addr_out_wts)
        rsrc_total_recv = create_buffer_resource_from_addr(addr_total_recv)

        total_recv = buffer_load(rsrc_total_recv, 0, vec_width=1, dtype=T.i32())
        safe_recv = arith.select(
            total_recv == arith.constant(0), arith.constant(1), total_recv
        )
        warps_per_tok = (
            arith.constant(global_warp_num) + safe_recv - arith.constant(1)
        ) // safe_recv
        units_per_warp = (
            arith.constant(n_i32) + warps_per_tok - arith.constant(1)
        ) // warps_per_tok
        stage_total = total_recv * warps_per_tok
        for stage_idx in range(global_warp_id, stage_total, global_warp_num):
            recv_tok = stage_idx // warps_per_tok
            part_id = stage_idx % warps_per_tok
            unit_base = part_id * units_per_warp
            slot_map_base = recv_tok * experts_per_token
            expert_rsrcs = []
            expert_valids = []
            expert_weights = []
            for k_slot in range_constexpr(experts_per_token):
                slot = P.load_i64_acquire(
                    fx.Int64(addr_slot_map)
                    + fx.Int64(slot_map_base + k_slot) * fx.Int64(8)
                )
                valid = slot != arith.constant(-1, type=T.i64())
                safe_slot = arith.select(valid, slot, arith.constant(0, type=T.i64()))
                x_addr = fx.Int64(addr_packed_x) + safe_slot * fx.Int64(nbytes)
                expert_rsrcs.append(create_buffer_resource_from_addr(x_addr))
                expert_valids.append(valid)
                expert_weights.append(
                    buffer_load(
                        rsrc_out_wts, slot_map_base + k_slot, vec_width=1, dtype=T.f32()
                    )
                )
            rsrc_out = create_buffer_resource_from_addr(
                fx.Int64(addr_out_tok) + fx.Int64(recv_tok) * fx.Int64(nbytes)
            )
            rem = arith.constant(n_i32) - unit_base
            eff = arith.select(rem < units_per_warp, rem, units_per_warp)

            def _one(off):
                acc = _to_accum2(arith.constant(0))
                for k_slot in range_constexpr(experts_per_token):
                    v = buffer_load(
                        expert_rsrcs[k_slot], off, vec_width=1, dtype=T.i32()
                    )
                    weight_val = arith.select(
                        expert_valids[k_slot],
                        expert_weights[k_slot],
                        arith.constant(0.0, type=T.f32()),
                    )
                    # v2f32 vector * f32 scalar (broadcast), matching the
                    # FlyDSL reference _weighted_accum_experts.
                    acc = acc + _to_accum2(v) * weight_val
                buffer_store(_from_accum2(acc), rsrc_out, off)

            def _loop():
                for u in range(lane, eff, WAVE):
                    _one(unit_base + u)

            _loop()

    @flyc.jit
    def run(
        addr_out_tok: Int64,
        addr_out_wts: Int64,
        addr_total_recv: Int64,
        addr_packed_x: Int64,
        addr_slot_map: Int64,
        stream=fx.Stream(None),
    ):
        convert_comb(
            addr_out_tok, addr_out_wts, addr_total_recv, addr_packed_x, addr_slot_map
        ).launch(
            grid=(block_num, 1, 1),
            block=[warp_num_per_block * WAVE, 1, 1],
            stream=stream,
        )

    return run


# ── local expert count ────────────────────────────────────────────────────


def make_local_expert_count(
    *, rank, experts_per_rank, experts_per_token, block_num, warp_num_per_block
):
    expert_base = rank * experts_per_rank
    block_size = warp_num_per_block * WAVE

    @flyc.kernel(known_block_size=[block_size, 1, 1])
    def local_expert_count_kernel(
        addr_out_idx: Int64, addr_total_recv: Int64, addr_count: Int64
    ):
        global_thread_id = fx.block_idx.x * block_size + fx.thread_idx.x
        global_thread_num = block_num * block_size
        rsrc_out_idx = create_buffer_resource_from_addr(addr_out_idx)
        rsrc_total_recv = create_buffer_resource_from_addr(addr_total_recv)
        limit = (
            buffer_load(rsrc_total_recv, 0, vec_width=1, dtype=T.i32())
            * experts_per_token
        )
        for i in range(global_thread_id, limit, global_thread_num):
            local_expert = (
                buffer_load(rsrc_out_idx, i, vec_width=1, dtype=T.i32()) - expert_base
            )
            if local_expert >= 0:
                if local_expert < experts_per_rank:
                    P.atomic_add_global(
                        fx.Int64(addr_count) + fx.Int64(local_expert) * fx.Int64(4),
                        fx.Int32(1),
                    )

    @flyc.jit
    def run(
        addr_out_idx: Int64,
        addr_total_recv: Int64,
        addr_count: Int64,
        stream=fx.Stream(None),
    ):
        local_expert_count_kernel(addr_out_idx, addr_total_recv, addr_count).launch(
            grid=(block_num, 1, 1), block=[block_size, 1, 1], stream=stream
        )

    return run
