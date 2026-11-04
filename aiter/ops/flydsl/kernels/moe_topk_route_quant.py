# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Fused top-k softmax + route/psum + MX quant/preshuffle for small decode MoE.

Replaces three launches of the TP grouped a8w4/fp4 MoE front end
(``_grouped_a8w4_tdm_moe`` at small token counts):

    1. ``topk_softmax`` (HIP ``topkGatingSoftmax``)     logits -> top-k ids/weights
    2. ``moe_route_psum_fused``                         ids -> masked_m, psum, rows
    3. ``moe_fused_quant_preshuffle_routeks``           hidden -> grouped MX payload
                                                        + preshuffled e8m0 scale

with one launch. Two schedules, both built here:

  * ``local`` (up to ``LOCAL_MAX_ROWS`` rows): ``ceil(token_num / 4)``
    independent workgroups; each computes the top-k of every row itself (one
    pass of the block, a few KB of L2-resident logits) and the routing in LDS,
    then quantizes its own 4 tokens. No workgroup waits on another.
  * cluster (up to ``MAX_ROWS``): ``ceil(token_num / 64)`` gfx1250 clusters of
    ``CLUSTER`` workgroups. A cluster's workgroups split the top-k of all rows
    and publish the ids (SCOPE_DEV stores), meet at one cluster barrier, read
    every id back and build the routing in LDS; each then quantizes its share
    of the cluster's tokens. Splitting the top-k keeps it off the critical
    path; a second cluster halves the quant stores each CU issues.

A route's grouped row is assigned deterministically, token-major within an
expert: a per-expert bitmask of the tokens that picked it gives both the count
(popcount) and the route's slot (popcount of the lower tokens). (The split path
assigns slots with LDS atomics, so its order inside an expert is whatever the
atomics did; a GEMM row never depends on its neighbours, so MoE1 / MoE2 see the
same rows either way and produce the same output.)

Semantics reproduced bit for bit:
  * top-k: ``topkGatingSoftmax<..., need_renorm=true>`` -- largest logit first,
    ties to the lower expert index, ``w_k = exp(l_k - l_max)`` summed in k
    order, each multiplied by ``1 / sum``. Rows at or past ``num_valid_rows``
    get ids -1 and weights 0, claim no grouped row and are not quantized.
  * routing: ``masked_m[e]`` (count), ``psum[e]`` (tile-aligned start + count,
    the GEMM's m-tile map), ``topids_to_rows`` (dropped routes -1).
  * quant: ``_emit_quant_block_loop`` (gfx1250 pk8 MXFP4/MXFP8, e8m0 scales in
    the WMMA-preshuffled layout), the emitter the split quant kernels use.

Block: 1024 threads = 32 waves. Top-k: one wave a row, 8 logits a lane, as
repeated u32 min-reductions over order-preserving keys. Quant: one wave a
(token, K-group of 256 columns).
"""

from types import SimpleNamespace

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, ptrtoint, range_constexpr, rocdl
from flydsl.expr.rocdl import cluster
from flydsl.expr.typing import Int32, T

from aiter.ops.flydsl.kernels import buffer_ops
from aiter.ops.flydsl.kernels.moe_fused_route_quant_scatter import (
    _SCALE_RSRC_MAX_BYTES,
    _emit_quant_block_loop,
    _quant_layout,
    _token_multidest_scale_base,
)
from aiter.ops.flydsl.kernels.moe_route_maps import DROPPED_ROUTE_ROW
from aiter.ops.flydsl.kernels.tensor_shim import (
    AITER_FLYDSL_KERNARG_PRELOAD,
    AITER_FLYDSL_KERNARG_PRELOAD_COUNT,
    ptr_buf_tensor,
)

BLOCK_THREADS = 1024
# Workgroups in the cluster (the whole grid); 16 is the gfx1250 maximum.
CLUSTER = 16
# Top-k lanes a row (one wave), 8 logits a lane. Any split gives the same
# result (every key is unique); a whole wave a row halves each selection
# step's dependent chain against the HIP kernel's 16 lanes x 16 logits.
LANES_PER_ROW = 32
# Rows the kernel serves. 128 covers the drafter (13-80 rows) and the target's
# verify step; larger batches keep the split path.
MAX_ROWS = 128
# Up to this many rows every workgroup computes all of the top-k itself (one
# pass of the block), which beats the cluster exchange's barrier + read-back.
LOCAL_MAX_ROWS = 16
# gfx1250 CPOL scope bits: device scope, so a store is visible in L2 to the
# cluster's other workgroups and a load does not hit a stale L0/L1 line.
_CPOL_SCOPE_DEV = 16


_BUTTERFLY = tuple(1 << i for i in range(LANES_PER_ROW.bit_length() - 1))


def _min_tree(vals):
    """Unsigned min of a Python list of values as a 3-input tree (v_min3_u32).

    A plain helper, so the kernel AST rewriter leaves its loops alone."""
    while len(vals) > 1:
        nxt = [fx.min(*vals[q : q + 3]) for q in range(0, len(vals) - 2, 3)]
        rem = len(vals) % 3
        if rem == 2:
            nxt.append(fx.min(vals[-2], vals[-1]))
        elif rem == 1:
            nxt.append(vals[-1])
        vals = nxt
    return vals[0]


def topk_route_quant_tokens_per_block(feat_dim: int, quant_mode: str) -> int:
    """Tokens one workgroup quantizes: one wave per (token, K-group)."""
    L = _quant_layout(feat_dim, quant_mode, 1)
    return (BLOCK_THREADS // L.wave_size) // L.block_iters


def _set_token_bit(lds_bits, expert, token, words):
    """Mark ``token`` in ``expert``'s token bitmask (LDS)."""
    fx.atomic_or(
        lds_bits
        + (fx.Int32(expert) * fx.Int32(words) + fx.Int32(fx.Uint32(token) >> 5)),
        fx.Int32(1) << fx.Int32(fx.Uint32(token) & 31),
        syncscope="workgroup",
    )


def build_moe_topk_route_quant_module(
    *,
    experts: int,
    topk: int,
    feat_dim: int,
    wmma_rep: int,
    align_m: int,
    quant_mode: str = "fp4",
    renormalize: bool = True,
    max_rows: int = MAX_ROWS,
    local: bool = False,
):
    """JIT launcher ``(logits, hidden, num_valid_rows, topk_weights, topk_ids,
    topids_to_rows, masked_m, psum, payload, scale, exchange, token_num,
    row_stride, grid_blocks)``.

    ``local``: every workgroup computes the top-k of every row itself (at most
    ``LOCAL_MAX_ROWS``), so there is no exchange; the grid is plain,
    ``ceil(token_num / tokens_per_block)``. Otherwise the grid is
    ``ceil(token_num / (CLUSTER * tokens_per_block))`` clusters of ``CLUSTER``:
    each cluster splits the top-k of all rows over its workgroups, exchanges the
    ids (cluster 0 through ``topk_ids``, cluster c > 0 through row c - 1 of
    ``exchange``) and quantizes its share of the tokens.
    """
    if experts != 256:
        raise NotImplementedError(
            f"fused top-k/route/quant covers 256 experts, got {experts}"
        )
    if not renormalize:
        # The full-row softmax denominator would need the HIP kernel's
        # per-lane-then-butterfly summation order reproduced; nothing uses it.
        raise NotImplementedError("fused top-k/route/quant renormalizes only")
    if not 1 <= topk <= min(LANES_PER_ROW, 16):
        raise NotImplementedError(f"topk {topk} unsupported")
    if align_m & (align_m - 1) or align_m <= 0:
        raise ValueError(f"align_m {align_m} must be a power of two")
    L = _quant_layout(feat_dim, quant_mode, wmma_rep)
    if not L.use_pk8 or L.lanes_per_mx_block != 4:
        raise NotImplementedError("fused top-k/route/quant requires gfx1250 pk8")
    wave_size = L.wave_size
    waves = BLOCK_THREADS // wave_size
    k_groups = L.block_iters  # one wave iteration (256 columns) a K-group
    if waves % k_groups:
        raise NotImplementedError(f"{k_groups} K-groups do not tile {waves} waves")
    # Tokens a workgroup quantizes: one (token, K-group) task a wave.
    tokens_per_block = waves // k_groups
    rows_per_pass = BLOCK_THREADS // LANES_PER_ROW
    if local:
        max_rows = min(max_rows, rows_per_pass)
    if -(-max_rows // CLUSTER) * LANES_PER_ROW > BLOCK_THREADS:
        raise ValueError(f"max_rows {max_rows} needs more top-k lanes than a block")
    if max_rows * topk > BLOCK_THREADS:
        raise ValueError(f"max_rows {max_rows} * topk exceeds one id a thread")
    tasks_per_wave = 1
    vals_per_lane = experts // LANES_PER_ROW  # 8
    words = -(-max_rows // 32)  # bitmask dwords per expert
    if experts % wave_size:
        raise NotImplementedError("experts must be a multiple of the wave size")
    experts_per_lane = experts // wave_size

    @fx.struct
    class _Storage:
        ids: fx.Array[fx.Int32, max_rows * topk, 16]
        bits: fx.Array[fx.Int32, experts * words, 16]
        starts: fx.Array[fx.Int32, experts, 16]

    module_name = (
        f"moe_topk_route_quant_e{experts}_k{topk}_fd{feat_dim}_r{wmma_rep}"
        f"_a{align_m}_{quant_mode}_{L.native_tag}_rn_mr{max_rows}{'_local' if local else f'_cl{CLUSTER}'}"
    )

    @flyc.kernel(name=module_name, known_block_size=[BLOCK_THREADS, 1, 1])
    def topk_route_quant_kernel(
        logits: fx.Pointer,  # (token_num, row_stride) bf16, routed columns first
        hidden: fx.Pointer,  # (token_num, feat_dim) bf16
        num_valid_rows: fx.Pointer,  # (1,) i32 or null
        topk_weights: fx.Pointer,  # (token_num, topk) f32 out
        topk_ids: fx.Pointer,  # (token_num, topk) i32 out (also the exchange)
        topids_to_rows: fx.Pointer,  # (token_num * topk,) i32 out
        masked_m: fx.Pointer,  # (experts,) i32 out
        psum: fx.Pointer,  # (experts,) i32 out
        grouped_payload: fx.Pointer,  # (contiguous_m, payload_bytes) u8 out
        grouped_scale: fx.Pointer,  # preshuffled e8m0 out
        exchange: fx.Pointer,  # (clusters - 1, token_num * topk) i32 scratch
        token_num: Int32,
        logits_row_stride: Int32,  # elements
    ):
        i32 = T.i32
        f32 = T.f32
        c0_i32 = fx.Int32(0)
        c1_i32 = fx.Int32(1)
        c4_i32 = fx.Int32(4)
        c16_i32 = fx.Int32(16)
        c23_i32 = fx.Int32(23)
        c254_i32 = fx.Int32(254)
        c0_f32 = fx.Float32(0.0)
        c_hi16 = fx.Uint32(0xFFFF0000)
        c_sign = fx.Uint32(0x80000000)
        c_emax = fx.Uint32(experts - 1)
        c_wave = fx.Int32(wave_size)

        tid = fx.Uint32(fx.thread_idx.x)
        bid = fx.Uint32(fx.block_idx.x)
        n_tok = fx.Uint32(token_num)
        c_tpb = fx.Uint32(tokens_per_block)
        if const_expr(local):
            # Top-k rows: all of them; quant tokens: [q_base, q_end).
            tk_base = fx.Uint32(0)
            tk_count = n_tok
            q_base = bid * c_tpb
            cl_id = fx.Uint32(0)
            wg_in_cl = bid
        else:
            cl_id = bid // fx.Uint32(CLUSTER)
            wg_in_cl = bid - cl_id * fx.Uint32(CLUSTER)
            n_cl = (n_tok + fx.Uint32(CLUSTER * tokens_per_block - 1)) // fx.Uint32(
                CLUSTER * tokens_per_block
            )
            # The cluster's workgroups split the top-k of every row ...
            tk_count = (n_tok + fx.Uint32(CLUSTER - 1)) // fx.Uint32(CLUSTER)
            tk_base = wg_in_cl * tk_count
            # ... and the cluster quantizes its share of the tokens.
            per_cl = (n_tok + n_cl - fx.Uint32(1)) // n_cl
            per_wg = (per_cl + fx.Uint32(CLUSTER - 1)) // fx.Uint32(CLUSTER)
            q_base = cl_id * per_cl + wg_in_cl * per_wg
            c_tpb = per_wg
        q_end = fx.min(q_base + c_tpb, n_tok)
        if const_expr(not local):
            q_end = fx.min(q_end, (cl_id + fx.Uint32(1)) * per_cl)
        # cluster 0 publishes through topk_ids itself, the others through
        # their row of `exchange`.
        xchg_addr = (cl_id == fx.Uint32(0)).select(
            fx.Int64(ptrtoint(topk_ids)),
            fx.Int64(ptrtoint(exchange))
            + fx.Int64(cl_id - fx.Uint32(1)) * fx.Int64(n_tok * fx.Uint32(topk * 4)),
        )
        xchg_rsrc = buffer_ops.create_buffer_resource_from_addr(xchg_addr)

        lds = fx.SharedAllocator().allocate(_Storage).peek()
        lds_ids = lds.ids.ptr
        lds_bits = lds.bits.ptr
        lds_starts = lds.starts.ptr

        wid = fx.Uint32(rocdl.readfirstlane(i32, fx.Uint32(tid // c_wave)))
        lane = tid - wid * fx.Uint32(wave_size)

        # Prologue: every global read whose address is known up front is issued
        # here, so the latencies overlap. Reads past the batch go through
        # zero-length descriptors (they return 0) rather than branches.
        #
        # Rows at or past num_valid_rows are padding (signed, as the HIP top-k
        # compares it: a negative count pads every row).
        nvr_is_set = fx.Int64(ptrtoint(num_valid_rows)) != 0
        nvr_raw = fx.Int32(
            buffer_ops.buffer_load(
                buffer_ops.create_buffer_resource_from_addr(
                    fx.Int64(ptrtoint(num_valid_rows)),
                    num_records_bytes=nvr_is_set.select(fx.Int32(4), c0_i32),
                ),
                c0_i32,
                vec_width=1,
                dtype=i32,
            )
        )

        # This wave's quant task: (token, K-group) = divmod(wid, k_groups) of
        # the workgroup's tokens; 8 bf16 of the row a lane.
        hid_base = fx.Int64(ptrtoint(hidden))
        tasks = []
        for i in range_constexpr(tasks_per_wave):
            task = wid + fx.Uint32(i * waves)
            tok_local = task // fx.Uint32(k_groups)
            k_group = task - tok_local * fx.Uint32(k_groups)
            token = q_base + tok_local
            live = token < q_end
            rsrc = buffer_ops.create_buffer_resource_from_addr(
                hid_base + fx.Int64(token) * fx.Int64(feat_dim * 2),
                num_records_bytes=live.select(fx.Int32(feat_dim * 2), c0_i32),
            )
            dwords = buffer_ops.buffer_load(
                rsrc,
                fx.Int32(k_group) * fx.Int32(L.mx_blocks_per_wave_iter * 16)
                + fx.Int32(lane) * fx.Int32(L.elems_per_lane // 2),
                vec_width=4,
                dtype=i32,
            )
            tasks.append((live, token, k_group, dwords))

        # The logits of this lane's row: LANES_PER_ROW lanes a row, 8 logits
        # (one 16-byte load) a lane. One descriptor for the whole block keeps each
        # load a single buffer_load (a per-row one differs between the two rows
        # of a wave and would go through a waterfall). A row offset is only
        # 2-byte aligned (odd row strides), hence the i16 element type.
        grp = tid // fx.Uint32(LANES_PER_ROW)
        g = tid - grp * fx.Uint32(LANES_PER_ROW)
        col0 = fx.Int32(g) * fx.Int32(vals_per_lane)
        row = tk_base + grp
        row_bytes = fx.Int32(logits_row_stride) * fx.Int32(2)
        logits_rsrc = buffer_ops.create_buffer_resource_from_addr(
            fx.Int64(ptrtoint(logits)),
            num_records_bytes=fx.Int32(token_num) * row_bytes,
        )
        row_off = fx.Int32(row) * row_bytes + col0 * fx.Int32(2)
        row_dwords = [
            fx.Vector(
                buffer_ops.buffer_load(
                    logits_rsrc,
                    (row_off + fx.Int32(h * 16)) >> c1_i32,
                    vec_width=8,
                    dtype=T.i16,
                )
            ).bitcast(fx.Int32)
            for h in range_constexpr(vals_per_lane // 8)
        ]

        # Clear the per-expert token bitmask (made visible by the cluster
        # barrier's workgroup barrier).
        for i in range_constexpr(-(-(experts * words) // BLOCK_THREADS)):
            idx = tid + fx.Uint32(i * BLOCK_THREADS)
            if const_expr((i + 1) * BLOCK_THREADS <= experts * words):
                lds_bits[idx] = c0_i32
            else:
                if idx < fx.Uint32(experts * words):
                    lds_bits[idx] = c0_i32

        n_valid = nvr_is_set.select(
            fx.Uint32((nvr_raw < c0_i32).select(c0_i32, nvr_raw)), n_tok
        )

        # Phase 1: top-k of this workgroup's rows.
        if (grp < tk_count) & (row < n_tok):
            is_pad = row >= n_valid
            # One unique u32 key per logit, ordered as the HIP top-k picks: the
            # f32 bits made monotonic (upper 16 bits: a bf16 widens with zero
            # low bits) over 255 - column, so a tie goes to the lower column.
            # -0 is folded into +0 first, as a float compare ties them.
            keys = []
            for h in range_constexpr(vals_per_lane // 8):
                dw = row_dwords[h]
                for j in range_constexpr(8):
                    d = fx.Uint32(dw[j // 2])
                    b = d << fx.Uint32(16) if j % 2 == 0 else d & c_hi16
                    b = fx.Uint32((b.bitcast(fx.Float32) + c0_f32).bitcast(fx.Int32))
                    flip = fx.Uint32(fx.Int32(b) >> fx.Int32(31)) | c_sign
                    col = fx.Uint32(col0) + fx.Uint32(h * 8 + j)
                    keys.append(((b ^ flip) & c_hi16) | (c_emax - col))

            # Repeated max over the keys below the last winner: the distance
            # (last - 1) - key wraps around for every key already taken.
            top_key = []
            bound = fx.Uint32(0xFFFFFFFF)
            for k in range_constexpr(topk):
                dmin = _min_tree([bound - kk for kk in keys])
                for sh in _BUTTERFLY:
                    dmin = fx.min(dmin, dmin.shuffle_xor(fx.Int32(sh), c_wave))
                win = bound - dmin
                top_key.append(win)
                bound = win - fx.Uint32(1)

            top_e = [fx.Int32(c_emax - (kk & fx.Uint32(0xFF))) for kk in top_key]
            top_v = []
            for kk in top_key:
                hi = kk & c_hi16
                pos = hi >= c_sign
                bits = pos.select(hi ^ c_sign, (hi ^ fx.Uint32(0xFFFFFFFF)) & c_hi16)
                top_v.append(fx.Int32(bits).bitcast(fx.Float32))

            # topkGatingSoftmax<need_renorm>: exp(l - l_max), summed in k order,
            # each multiplied by the sum's reciprocal.
            numer = [fx.exp(top_v[k] - top_v[0]) for k in range_constexpr(topk)]
            acc = numer[0]
            for k in range_constexpr(1, topk):
                acc = acc + numer[k]
            nz = acc != c0_f32
            scale = nz.select(fx.Float32(1.0) / acc, fx.Float32(1.0))

            # One route per lane: lane g < topk writes route g of this row.
            my_e = top_e[0]
            my_n = numer[0]
            for k in range_constexpr(1, topk):
                is_k = g == fx.Uint32(k)
                my_e = is_k.select(top_e[k], my_e)
                my_n = is_k.select(numer[k], my_n)
            my_w = nz.select(my_n * scale, my_n)
            route = fx.Int32(row) * fx.Int32(topk) + fx.Int32(g)
            out_e = is_pad.select(fx.Int32(-1), my_e)
            if g < fx.Uint32(topk):
                if const_expr(local):
                    if (row >= q_base) & (row < q_end):
                        ptr_buf_tensor(topk_weights, fx.Float32)[route] = is_pad.select(
                            c0_f32, my_w
                        )
                        ptr_buf_tensor(topk_ids)[route] = out_e
                    lds_ids[route] = out_e
                    if not is_pad:
                        _set_token_bit(lds_bits, my_e, row, words)
                else:
                    if cl_id == fx.Uint32(0):
                        ptr_buf_tensor(topk_weights, fx.Float32)[route] = is_pad.select(
                            c0_f32, my_w
                        )
                    buffer_ops.buffer_store(
                        out_e, xchg_rsrc, route, cache_modifier=_CPOL_SCOPE_DEV
                    )

        if const_expr(local):
            gpu.barrier()
        else:
            # Publish the ids to the cluster: the workgroup-scope release is
            # the s_wait_storecnt that lands them (device scope comes from
            # SCOPE_DEV on the store and on the read-back below).
            fx.memory_fence(ordering=fx.AtomicOrdering.Release, syncscope="workgroup")
            cluster.cluster_barrier()

            # Phase 2: every route of the batch into the per-expert bitmask.
            if tid < n_tok * fx.Uint32(topk):
                e = fx.Int32(
                    buffer_ops.buffer_load(
                        xchg_rsrc,
                        fx.Int32(tid),
                        vec_width=1,
                        dtype=i32,
                        cache_modifier=_CPOL_SCOPE_DEV,
                    )
                )
                lds_ids[tid] = e
                if e >= c0_i32:
                    _set_token_bit(lds_bits, e, tid // fx.Uint32(topk), words)
            gpu.barrier()

        # Phase 3 (wave 0): per-expert counts and the tile-aligned exclusive
        # scan, `experts_per_lane` consecutive experts a lane.
        if wid == fx.Uint32(0):
            e0 = fx.Int32(lane) * fx.Int32(experts_per_lane)
            cnts = []
            for i in range_constexpr(experts_per_lane):
                c = c0_i32
                for w in range_constexpr(words):
                    c = c + fx.ctpop(lds_bits[(e0 + fx.Int32(i)) * fx.Int32(words) + w])
                cnts.append(c)
            padded = [(c + fx.Int32(align_m - 1)) & fx.Int32(-align_m) for c in cnts]
            incl = []
            run = c0_i32
            for i in range_constexpr(experts_per_lane):
                run = run + padded[i]
                incl.append(run)
            lane_incl = run
            for d in (1, 2, 4, 8, 16):
                up = fx.Int32(gpu.shuffle_up(lane_incl, fx.Int32(d), c_wave))
                lane_incl = (lane >= fx.Uint32(d)).select(lane_incl + up, lane_incl)
            lane_excl = lane_incl - run
            is_writer = bid == fx.Uint32(0)
            for i in range_constexpr(experts_per_lane):
                start = lane_excl + incl[i] - padded[i]
                lds_starts[e0 + fx.Int32(i)] = start
                if is_writer:
                    ptr_buf_tensor(masked_m)[e0 + fx.Int32(i)] = cnts[i]
                    ptr_buf_tensor(psum)[e0 + fx.Int32(i)] = start + cnts[i]
        gpu.barrier()

        # Phase 4: quantize this workgroup's tokens into their grouped rows.
        rows_t = ptr_buf_tensor(topids_to_rows)
        scale_rsrc = buffer_ops.create_buffer_resource_from_addr(
            fx.Int64(ptrtoint(grouped_scale)),
            num_records_bytes=_SCALE_RSRC_MAX_BYTES,
        )
        c_lanes_per_block = fx.Int32(L.lanes_per_mx_block)
        block_in_wave = lane // fx.Uint32(c_lanes_per_block)
        lane_in_block = lane - block_in_wave * c_lanes_per_block
        # Every route row of the wave's tasks at once, one lane a route (lane
        # i * topk + k: task i, route k), then broadcast with readlane. Done a
        # route at a time, each row is a chain of dependent LDS round trips.
        if const_expr(tasks_per_wave * topk > wave_size):
            raise NotImplementedError("a wave's task routes must fit one wave")
        l_task = lane // fx.Uint32(topk)
        l_k = lane - l_task * fx.Uint32(topk)
        l_live = fx.Boolean(False)
        l_token = fx.Uint32(0)
        l_writes = fx.Boolean(False)
        for i in range_constexpr(tasks_per_wave):
            live_i, token_i, k_group_i, _ = tasks[i]
            is_i = l_task == fx.Uint32(i)
            l_live = is_i.select(live_i, l_live)
            l_token = is_i.select(token_i, l_token)
            l_writes = is_i.select(live_i & (k_group_i == fx.Uint32(0)), l_writes)
        l_route = fx.Int32(l_token) * fx.Int32(topk) + fx.Int32(l_k)
        l_e = fx.Int32(lds_ids[l_live.select(l_route, c0_i32)])
        l_valid = l_live & (l_token < n_valid) & (l_e >= c0_i32)
        l_base = l_valid.select(l_e, c0_i32) * fx.Int32(words)
        l_word = l_token >> fx.Uint32(5)
        l_below = (c1_i32 << fx.Int32(l_token & fx.Uint32(31))) - c1_i32
        l_slot = c0_i32
        for w in range_constexpr(words):
            bw = lds_bits[l_base + w]
            full = fx.Uint32(w) < l_word
            here = fx.Uint32(w) == l_word
            l_slot = l_slot + fx.ctpop(
                full.select(bw, here.select(bw & l_below, c0_i32))
            )
        l_row = l_valid.select(
            lds_starts[l_valid.select(l_e, c0_i32)] + l_slot,
            fx.Int32(DROPPED_ROUTE_ROW),
        )
        if l_writes:
            rows_t[l_route] = l_row

        for i in range_constexpr(tasks_per_wave):
            live, token, k_group, dwords = tasks[i]
            if live & (token < n_valid):
                rows = [
                    fx.Uint32(rocdl.readlane(i32, l_row, i * topk + k))
                    for k in range_constexpr(topk)
                ]

                scales = [
                    _token_multidest_scale_base(
                        r,
                        fx.Int32(L.rows_per_tile),
                        fx.Int32(L.dst_scale_dwords_per_row),
                        c16_i32,
                    )
                    for r in rows
                ]
                qc = SimpleNamespace(
                    i32=i32,
                    f32=f32,
                    block_iters=1,
                    mx_group_base=fx.Int32(k_group),
                    payload_base=fx.Int64(ptrtoint(grouped_payload)),
                    payload_bytes_per_row=L.payload_bytes_per_row,
                    hidden_base=hid_base,
                    feat_bytes_per_row=feat_dim * 2,
                    feat_row_i32=token,
                    # The prologue already holds this task's hidden slice.
                    hidden_lds_load=lambda _idx, _off, _d=dwords: _d,
                    hidden_lds_idx=None,
                    hidden_lds_row_off=c0_i32,
                    hidden_slot_bytes=0,
                    prequantized=False,
                    payload_dwords_per_lane=L.payload_bytes_per_lane // 4,
                    src_scale_base=fx.Int64(0),
                    src_scale_bytes_per_row=0,
                    mx_blocks_per_wave_iter=L.mx_blocks_per_wave_iter,
                    mx_blocks_per_row=L.mx_blocks_per_row,
                    amax_shuffle_dists=L.amax_shuffle_dists,
                    is_fp8=L.is_fp8,
                    use_native=L.use_native,
                    use_pk8=L.use_pk8,
                    mx_dtype=L.mx_dtype,
                    c0_i32=c0_i32,
                    c1_i32=c1_i32,
                    c4_i32=c4_i32,
                    c23_i32=c23_i32,
                    c254_i32=c254_i32,
                    c0_f32=c0_f32,
                    c_wave=c_wave,
                    c_elems_per_lane=fx.Int32(L.elems_per_lane),
                    c_payload_bytes_per_block=fx.Int32(L.payload_bytes_per_block),
                    c_payload_bytes_per_lane=fx.Int32(L.payload_bytes_per_lane),
                    c_wmma_rep=fx.Int32(wmma_rep),
                    block_in_wave=block_in_wave,
                    lane_in_block=lane_in_block,
                    is_block_lead=lane_in_block == c0_i32,
                    payload_dests=[SimpleNamespace(payload_row_i32=r) for r in rows],
                    dests=[
                        SimpleNamespace(payload_row_i32=r, scale_row_dword_base=s)
                        for r, s in zip(rows, scales)
                    ],
                    scale_rsrc=scale_rsrc,
                    scale_pack_dwords=True,
                )
                _emit_quant_block_loop(qc)

    @flyc.jit
    def launch_topk_route_quant(
        logits: fx.Pointer,
        hidden: fx.Pointer,
        num_valid_rows: fx.Pointer,
        topk_weights: fx.Pointer,
        topk_ids: fx.Pointer,
        topids_to_rows: fx.Pointer,
        masked_m: fx.Pointer,
        psum: fx.Pointer,
        grouped_payload: fx.Pointer,
        grouped_scale: fx.Pointer,
        exchange: fx.Pointer,
        token_num: fx.Int32,
        logits_row_stride: fx.Int32,
        grid_blocks: fx.Int32,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        args = (
            logits,
            hidden,
            num_valid_rows,
            topk_weights,
            topk_ids,
            topids_to_rows,
            masked_m,
            psum,
            grouped_payload,
            grouped_scale,
            exchange,
            token_num,
            logits_row_stride,
        )
        if const_expr(local):
            topk_route_quant_kernel(*args).launch(
                grid=(fx.Int64(grid_blocks), 1, 1),
                block=(BLOCK_THREADS, 1, 1),
                stream=stream,
            )
        else:
            topk_route_quant_kernel(
                *args, value_attrs={"rocdl.cluster_dims": f"{CLUSTER},1,1"}
            ).launch(
                grid=(fx.Int64(grid_blocks), 1, 1),
                block=(BLOCK_THREADS, 1, 1),
                stream=stream,
                cluster=(CLUSTER, 1, 1),
            )

    launch_topk_route_quant.compile_hints = {
        "llvm_options": {
            "amdgpu-kernarg-preload": AITER_FLYDSL_KERNARG_PRELOAD,
            "amdgpu-kernarg-preload-count": AITER_FLYDSL_KERNARG_PRELOAD_COUNT,
        },
    }
    return launch_topk_route_quant
