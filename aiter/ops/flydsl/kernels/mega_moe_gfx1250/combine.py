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
from flydsl.expr import arith, range_constexpr, tdm_ops
from flydsl.expr.rocdl import cvt_scale_pk8_f32_fp8
from flydsl.expr.typing import Int32, Int64, T
from flydsl.expr.typing import Vector as Vec

from aiter.ops.flydsl.kernels import communication_ops_utils as comm_ops
from aiter.ops.flydsl.kernels import vector
from aiter.ops.flydsl.kernels.buffer_ops import (
    buffer_load,
    buffer_store,
    create_buffer_resource_from_addr,
)
from aiter.ops.flydsl.kernels.gemm_common_gfx1250 import (
    make_lds_copy_ops,
    workgroup_barrier,
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

# MXFP8 combine wire format, mirrored from the gemm2 scatter epilogue: the
# hidden dim is cut into 256-element chunks, each carrying its own 8 e8m0 scale
# bytes immediately after its payload (plus 8 bytes of padding to keep every
# chunk 16-byte aligned). Payload and scale being one interval is what lets a
# single TDM descriptor bring both in together.
CHUNK_ELEMS = 256
CHUNK_BYTES = 272
# Bytes each lane dequantizes per round; 16 fp8 sit inside one 32-element MX
# block, so a round needs exactly one scale byte.
LANE_BYTES = 16

# The cross-device xdb barrier is inlined per kernel rather than shared with
# dispatch: FlyDSL only AST-rewrites dynamic `if` in the @flyc.kernel body, not in
# called helpers, so a helper's `if <lane predicate>:` would raise "cannot evaluate
# dynamic Boolean" at trace time.


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


def _make_combine_fused_sync(
    *,
    rank,
    npes,
    off_xdb_mem,
):
    """Stage A: wait until every peer's gemm2 P2P writes into comb_inp land.

    Launch this before the reduce kernel; its retirement is stream-ordered, so
    the reduce needs no in-kernel fence and its grid stays unconstrained.

    One thread per peer pushes the phase and polls that peer's local slot. The
    block is rounded up to whole waves, so rack-scale domains larger than one
    wave are covered without a cross-wave dependency or barrier -- the kernel's
    cost is the peer wait, not thread count.

    Being a rendezvous, it also fences the next dispatch off the regions this
    forward still reads, which is what lets ``Routing.source_token_map`` hand
    gemm2 a live view of ``recv_to_src_token`` instead of a copy.
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
        # `>=` not `==`: a faster peer can lap us and overwrite its monotonic push
        # with a higher call count before we read it.
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


def _make_combine_fused_reduce_mxfp8(
    *,
    experts_per_token,
    hidden_dim,
    block_num,
    warp_num_per_block,
    slot_stride_nbytes,
    tokens_per_block,
    chunks_per_iter,
):
    """Stage B over an MXFP8 wire, moved by TDM on both sides.

    A token's topk slots are ``tok*topk + k``, i.e. consecutive row indices a
    fixed ``slot_stride`` apart, so a group of ``tokens_per_block`` tokens is a
    plain strided 2D region -- no gather mode needed, unlike the scatter side
    where rows belong to different peers. One TDM load therefore brings
    ``T*topk`` slot rows into LDS, payload and scale together because the wire
    interleaves them per chunk. The dequantized bf16 result goes back out
    through a second TDM store.

    Each round a lane owns 16 fp8, which sit inside a single 32-element MX
    block, so it needs exactly one scale byte per expert.
    """
    topk = experts_per_token
    lanes = warp_num_per_block * WAVE
    n_chunks = hidden_dim // CHUNK_ELEMS
    T_TOK = tokens_per_block
    C_CHK = chunks_per_iter

    if hidden_dim % CHUNK_ELEMS:
        raise ValueError(f"hidden_dim must be a multiple of {CHUNK_ELEMS}")
    if n_chunks % C_CHK:
        raise ValueError(
            f"chunks_per_iter={C_CHK} must divide hidden chunks ({n_chunks})"
        )

    IN_ROWS = T_TOK * topk
    IN_ROW_BYTES = C_CHK * CHUNK_BYTES
    OUT_ROW_ELEMS = C_CHK * CHUNK_ELEMS
    OUT_ROW_BYTES = OUT_ROW_ELEMS * 2
    # Lane slots per iteration; one slot is LANE_BYTES fp8 of one token.
    SLOTS = T_TOK * OUT_ROW_ELEMS // LANE_BYTES
    SLOTS_PER_TOK = OUT_ROW_ELEMS // LANE_BYTES
    if SLOTS % lanes:
        raise ValueError(
            f"tokens_per_block*chunks_per_iter*{CHUNK_ELEMS}/{LANE_BYTES}={SLOTS} "
            f"must be a multiple of the block's lane count ({lanes})"
        )
    ROUNDS = SLOTS // lanes
    LDS_BYTES = IN_ROWS * IN_ROW_BYTES + T_TOK * OUT_ROW_BYTES
    if LDS_BYTES > 64 * 1024:
        raise ValueError(f"combine LDS tile is {LDS_BYTES} bytes, over the 64KB budget")

    slot_stride = slot_stride_nbytes
    iters_per_tok = n_chunks // C_CHK

    def _dequant_pk8(lo_i32, hi_i32, e8m0_i32):
        """Native gfx1250 ``v_cvt_scale_pk8_f32_fp8``: 8 fp8 e4m3 -> 8 f32.

        The scale is applied here rather than by the instruction. Letting the HW
        fold in the e8m0 (passing it as the scale operand instead of 127) costs
        an extra ~27x of error -- 61-layer logits_diff 0.618 vs 0.068, where the
        MXFP8 wire format alone only accounts for 0.070 -- so the conversion is
        run unscaled and the exact power of two is multiplied in afterwards.
        """
        src = Vec.from_elements([lo_i32, hi_i32], fx.Int32).ir_value()
        unscaled = Vec(
            cvt_scale_pk8_f32_fp8(
                T.vec(8, T.f32),
                src,
                arith.constant(127),  # e8m0 for 2^0
                0,
            )
        )
        return unscaled * (e8m0_i32 << arith.constant(23)).bitcast(fx.Float32)

    @flyc.kernel(known_block_size=[lanes, 1, 1])
    def ep_combine_fused(
        addr_comb_inp: Int64,
        addr_out: Int64,
        cur_rank_num_token: Int32,
    ):
        tid = fx.thread_idx.x
        bid = fx.block_idx.x

        # These build MLIR layouts/atoms, so they only work inside the kernel
        # body where a Context is established.
        lds_load_b32, _ = make_lds_copy_ops(32)
        lds_load_b128, lds_store_b128 = make_lds_copy_ops(128)

        smem = fx.SharedAllocator(static=False)
        in_ptr = smem.allocate(IN_ROWS * IN_ROW_BYTES)._ptr
        out_ptr = smem.allocate(T_TOK * OUT_ROW_BYTES)._ptr

        def ptr_to_idx(p):
            return fx.index_cast(T.index, fx.ptrtoint(p))

        in_idx = ptr_to_idx(in_ptr)
        out_idx = ptr_to_idx(out_ptr)

        p8_shared = fx.PointerType.get(
            elem_ty=fx.Int8.ir_type,
            address_space=fx.AddressSpace.Shared,
            alignment=16,
        )
        p16_shared = fx.PointerType.get(
            elem_ty=fx.Int16.ir_type,
            address_space=fx.AddressSpace.Shared,
            alignment=16,
        )
        i8_global = fx.PointerType.get(
            elem_ty=fx.Int8.ir_type,
            address_space=fx.AddressSpace.Global,
            alignment=16,
        )
        i16_global = fx.PointerType.get(
            elem_ty=fx.Int16.ir_type,
            address_space=fx.AddressSpace.Global,
            alignment=16,
        )
        inp_iter = fx.inttoptr(i8_global, fx.Int64(addr_comb_inp))
        out_iter = fx.inttoptr(i16_global, fx.Int64(addr_out))

        def global_view(base, off, shape, stride):
            return fx.Tensor(fx.make_view(base + off, fx.make_layout(shape, stride)))

        def lds_view(ptr, shape, stride):
            return fx.Tensor(fx.make_view(ptr, fx.make_layout(shape, stride)))

        lds_in = lds_view(
            fx.recast_iter(p8_shared, in_ptr),
            (IN_ROWS, IN_ROW_BYTES),
            (IN_ROW_BYTES, 1),
        )
        lds_out = lds_view(
            fx.recast_iter(p16_shared, out_ptr),
            (T_TOK, OUT_ROW_ELEMS),
            (OUT_ROW_ELEMS, 1),
        )

        # Lane -> (token, chunk, byte) decode, constant across iterations.
        lane_tok = []
        lane_chunk = []
        lane_byte = []
        for r in range_constexpr(ROUNDS):
            slot = tid + arith.constant(r * lanes)
            lane_tok.append(slot // arith.constant(SLOTS_PER_TOK))
            rem = slot % arith.constant(SLOTS_PER_TOK)
            lane_chunk.append(rem // arith.constant(CHUNK_ELEMS // LANE_BYTES))
            lane_byte.append(
                (rem % arith.constant(CHUNK_ELEMS // LANE_BYTES))
                * arith.constant(LANE_BYTES)
            )

        safe_tok = arith.select(
            cur_rank_num_token == arith.constant(0),
            arith.constant(1),
            cur_rank_num_token,
        )
        n_groups = (
            safe_tok + arith.constant(T_TOK - 1)
        ) // arith.constant(T_TOK)
        total_work = n_groups * arith.constant(iters_per_tok)
        for work in range(bid, total_work, block_num):
            grp = work // arith.constant(iters_per_tok)
            it = work % arith.constant(iters_per_tok)
            tok0 = grp * arith.constant(T_TOK)
            q0 = it * arith.constant(C_CHK)

            # -- global -> LDS: T*topk slot rows, payload and scale together --
            # Clamped to the tile height: the descriptor packs the bound into a
            # narrow field, and (tokens-tok0)*topk overflows it at 16k tokens
            # (98304). Only IN_ROWS rows are ever fetched, so any larger bound
            # means "all rows valid" anyway.
            _rows_left = (cur_rank_num_token - tok0) * arith.constant(topk)
            row_oob = arith.select(
                _rows_left > arith.constant(IN_ROWS),
                arith.constant(IN_ROWS),
                _rows_left,
            )
            g_off = fx.Int64(tok0) * fx.Int64(topk * slot_stride) + fx.Int64(
                q0
            ) * fx.Int64(CHUNK_BYTES)
            gt_in = global_view(
                inp_iter, g_off, (IN_ROWS, IN_ROW_BYTES), (slot_stride, 1)
            )
            atom_in = fx.rocdl.make_tdm_atom(
                gt_in,
                [row_oob, None],
                strides=[slot_stride, None],
                num_warps=warp_num_per_block,
            )
            fx.copy(atom_in, gt_in, lds_in)
            tdm_ops.tensor_wait(0)
            workgroup_barrier()

            # -- dequantize and sum the topk slots out of LDS --
            for r in range_constexpr(ROUNDS):
                t_off = lane_tok[r] * arith.constant(topk * IN_ROW_BYTES)
                base_in = (
                    t_off
                    + lane_chunk[r] * arith.constant(CHUNK_BYTES)
                    + lane_byte[r]
                )
                # Scale byte index within the chunk's 8-byte scale run; read as
                # a dword and shifted out so the byte stays unsigned.
                sc_i = lane_byte[r] // arith.constant(32)
                sc_dw = (
                    t_off
                    + lane_chunk[r] * arith.constant(CHUNK_BYTES)
                    + arith.constant(CHUNK_ELEMS)
                    + (sc_i // arith.constant(4)) * arith.constant(4)
                )
                sc_sh = (sc_i % arith.constant(4)) * arith.constant(8)

                accs = [Vec.filled(8, 0.0, fx.Float32) for _ in range_constexpr(2)]
                for k_slot in range_constexpr(topk):
                    row_b = arith.constant(k_slot * IN_ROW_BYTES)
                    payload = lds_load_b128(in_idx, base_in + row_b)
                    dw = lds_load_b32(in_idx, sc_dw + row_b)[0]
                    e8m0 = (dw >> sc_sh) & arith.constant(0xFF)
                    for j in range_constexpr(2):
                        accs[j] = accs[j] + _dequant_pk8(
                            payload[j * 2], payload[j * 2 + 1], e8m0
                        )

                out_b = (
                    lane_tok[r] * arith.constant(OUT_ROW_BYTES)
                    + lane_chunk[r] * arith.constant(CHUNK_ELEMS * 2)
                    + lane_byte[r] * arith.constant(2)
                )
                for j in range_constexpr(2):
                    lds_store_b128(
                        out_idx,
                        out_b + arith.constant(j * 16),
                        accs[j].to(fx.BFloat16).bitcast(fx.Int32).ir_value(),
                    )
            workgroup_barrier()

            # -- LDS -> global: T bf16 token rows, padding rows clamped away --
            out_off = fx.Int64(tok0) * fx.Int64(hidden_dim) + fx.Int64(
                q0
            ) * fx.Int64(CHUNK_ELEMS)
            gt_out = global_view(
                out_iter, out_off, (T_TOK, OUT_ROW_ELEMS), (hidden_dim, 1)
            )
            _toks_left = cur_rank_num_token - tok0
            atom_out = fx.rocdl.make_tdm_atom(
                gt_out,
                [
                    arith.select(
                        _toks_left > arith.constant(T_TOK),
                        arith.constant(T_TOK),
                        _toks_left,
                    ),
                    None,
                ],
                strides=[hidden_dim, None],
                num_warps=warp_num_per_block,
            )
            fx.copy(atom_out, lds_out, gt_out)
            tdm_ops.tensor_wait(0)
            # Next iteration's TDM load overwrites lds_in; the barrier above
            # already retired every lane's reads of it.

    @flyc.jit
    def run(
        addr_comb_inp: Int64,
        addr_out: Int64,
        cur_rank_num_token: Int32,
        stream=fx.Stream(None),  # noqa: B008
    ):
        ep_combine_fused(
            addr_comb_inp,
            addr_out,
            cur_rank_num_token,
        ).launch(
            grid=(block_num, 1, 1),
            block=[lanes, 1, 1],
            stream=stream,
        )

    return run


def _make_combine_fused_reduce(
    *,
    experts_per_token,
    hidden_dim,
    block_num,
    warp_num_per_block,
    slot_stride_nbytes=None,
    quant=False,
    tokens_per_block=2,
    chunks_per_iter=4,
):
    """Stage B of the GEMM2-fused scatter combine: the per-token topk sum.

    gemm2 has already P2P-written each token's WEIGHTED per-expert result into
    this rank's comb_inp[origin_lid*topk + k] (one contiguous topk-block per
    token), so this is an unweighted sum: out[t] = sum_{k<topk} comb_inp[t*topk
    + k], over a bf16 wire. The dropless full-topk pipeline overwrites every
    active (token, k) slot each call. ``_make_combine_fused_sync`` must have run
    first to make the peers' writes visible.

    ``quant`` switches to the MXFP8 wire, which is a structurally different
    kernel (TDM in and out, LDS staging); the bf16 path below is untouched so
    the default stays bit-identical to the pre-quantization baseline.
    """
    if quant:
        if slot_stride_nbytes is None:
            raise ValueError("MXFP8 combine requires an explicit slot stride")
        return _make_combine_fused_reduce_mxfp8(
            experts_per_token=experts_per_token,
            hidden_dim=hidden_dim,
            block_num=block_num,
            warp_num_per_block=warp_num_per_block,
            slot_stride_nbytes=slot_stride_nbytes,
            tokens_per_block=tokens_per_block,
            chunks_per_iter=chunks_per_iter,
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
        addr_comb_inp: Int64,
        addr_out: Int64,
        cur_rank_num_token: Int32,
    ):
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        lane = tid & LANE_MASK
        warp = tid >> LOG2_WAVE
        global_warp_id = bid * warp_num_per_block + warp
        global_warp_num = block_num * warp_num_per_block

        rsrc_out = create_buffer_resource_from_addr(addr_out)

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

        # No exit barrier: the reduce does no post-completion work, so kernel
        # retirement (stream-ordered) is the only completion signal the host needs.

    @flyc.jit
    def run(
        addr_comb_inp: Int64,
        addr_out: Int64,
        cur_rank_num_token: Int32,
        stream=fx.Stream(None),  # noqa: B008
    ):
        ep_combine_fused(
            addr_comb_inp,
            addr_out,
            cur_rank_num_token,
        ).launch(
            grid=(block_num, 1, 1),
            block=[warp_num_per_block * WAVE, 1, 1],
            stream=stream,
        )

    return run
