# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Fused top-k softmax + route/psum + MX quant for prefill-size MoE steps.

At prefill (2k-20k rows) the TP grouped a8w4/fp4 MoE front end is eight
launches: ``topkGatingSoftmax``, the ``num_valid_rows * topk`` multiply, the
route counter fill, ``moe_route``, ``moe_contiguous_psum_remap``, the
``row_to_token`` fill, ``moe_token_multidest_quant_fusepre`` (the MXFP4 payload
to each route's grouped row + compact token-major e8m0 rows) and
``moe_scatter_preshuffle_scale`` (compact rows -> MoE1's WMMA-interleaved
layout). Here: two. ``build_moe_topk_route_quant_prefill_module`` does the first
seven, ``build_moe_prefill_scale_preshuffle_module`` is the preshuffle bounded
by psum (so ``row_to_token`` needs no -1 fill past the experts).

Kernel 1: one workgroup per CU at most (every workgroup resident), each a
contiguous run of ``tokens_per_wg`` tokens, a token a wave (``slots`` a wave):

  A. Each wave TDM-loads its tokens' hidden rows into its own LDS rows (a wave
     waits on its own tensor count, no workgroup barrier), computes each
     token's top-k (``topkGatingSoftmax<renorm>`` bit for bit, the decode
     kernel's code), takes the route's slot inside the workgroup with an LDS
     atomic, quantizes the row (the split kernels' pk8 emitter) and keeps the
     MXFP4 payload in VGPRs; the e8m0 row goes out token-major (one 64 B store).
  B. Thread e claims the workgroup's run inside expert e with one device
     fetch-add on a status word: the low ``COUNT_BITS`` count routes (the old
     value is the workgroup's offset inside the expert: a single-pass prefix in
     arrival order, no scan launch), the bits above count the workgroups that
     have added. The experts' tile-aligned starts need every token's count, so
     thread e spins on its word until all workgroups have added; the last load
     is the total. (That all-workgroup dependency is inherent to the
     contiguous layout: an expert's start depends on every token's routes.)
     Each workgroup scans the 256 totals in LDS; workgroup 0 writes masked_m
     and psum.
  C. Writes topids_to_rows and row_to_token, every route's payload from the
     registers (all of a token's row descriptors first, then whole 1 KB rows)
     and -1 in row_to_token on the experts' tile padding (spread over the grid).
     With ``tdm_store`` each slot's payload goes to its token's LDS row right
     after its quant (A2), and C stores it by TDM gather instead: one op is 8
     tokens x 1 route (8 KB), spread over all waves, so the store rate no longer
     depends on how many slots a wave holds. That lets the host launch 16 waves
     x 4 slots where it would launch 32 x 2: half the wave launches at the same
     store throughput (16384 tokens on 256 CUs: 26.5 -> 24.0 us).

The status words must be zero at launch. The last workgroup out zeroes them
again, so a CUDA graph replays without a memset; the host allocates them zeroed
once per device. A wait that times out sets a sticky error word instead of
hanging the GPU.

Rows inside an expert are in workgroup-arrival order, then LDS-atomic order:
a GEMM row never depends on its neighbours and the gather-reduce sums a token's
routes in k order, so MoE1 / MoE2 / the reduce produce the same bits as with
the split path's order (atomic order as well).

Rows at or past ``num_valid_rows`` get ids -1, weights 0, rows -1, no grouped
row and no quant (the split path's contract).
"""

from types import SimpleNamespace

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, ptrtoint, range_constexpr, rocdl
from flydsl.expr.typing import Int32, T

from aiter.ops.flydsl.kernels import buffer_ops
from aiter.ops.flydsl.kernels import tdm_ops_gfx1250 as tdm_ops
from aiter.ops.flydsl.kernels.gemm_common_gfx1250 import make_lds_copy_ops
from aiter.ops.flydsl.kernels.mega_moe_gfx1250.tdm_gather_shim import (
    make_tensor_gather_descriptor,
    tensor_store_gather,
)
from aiter.ops.flydsl.kernels.moe_fused_route_quant_scatter import (
    _emit_payload_stores,
    _emit_quant_compute,
    _i32_ptr,
    _pack_block_group_dword,
    _quant_layout,
)
from aiter.ops.flydsl.kernels.moe_route_maps import DROPPED_ROUTE_ROW
from aiter.ops.flydsl.kernels.moe_topk_route_quant import (
    _BUTTERFLY,
    LANES_PER_ROW,
    _min_tree,
)
from aiter.ops.flydsl.kernels.tensor_shim import (
    AITER_FLYDSL_KERNARG_PRELOAD,
    AITER_FLYDSL_KERNARG_PRELOAD_COUNT,
    ptr_buf_tensor,
)

# Waves a workgroup (a token a wave a slot); 8 is the floor (a thread an
# expert). The host takes the fewest that hold a workgroup's tokens in one slot.
WAVE_CHOICES = (8, 16, 32)
# Tokens a wave holds (payload in VGPRs across the wait).
MAX_SLOTS = 4
# With the TDM store the payload leaves the registers right after each slot's
# quant, so a wave may hold more tokens.
MAX_SLOTS_TDM = 8
# LDS the kernel may use (hidden staging, plus the TDM store's payload rows).
LDS_BUDGET = 300 * 1024
# Spins (s_sleep 1 each) before a wait gives up and flags the error word.
SPIN_LIMIT = 1 << 22
# A status count word: routes in the low bits, arrived workgroups above.
COUNT_BITS = 20
COUNT_MASK = (1 << COUNT_BITS) - 1


def status_words(experts: int) -> int:
    """Status buffer length: per-expert counts, departures, error, padding."""
    return experts + 4


def _status_depart(experts):
    return experts


def _status_error(experts):
    return experts + 1


def _ptr_at(base_i64, index):
    """Global i32 pointer at ``base + 4 * index``."""
    ptr_type = fx.PointerType.get(fx.Int32.ir_type, fx.AddressSpace.Global, 4)
    return fx.inttoptr(ptr_type, fx.Int64(base_i64) + fx.Int64(index) * 4)


def _lds_barrier():
    """Workgroup barrier for LDS only: gpu.barrier() also waits for every
    outstanding global store (a workgroup release in WGP mode)."""
    rocdl.s_wait_dscnt(0)
    rocdl.s_barrier()


def _load_agent(base_i64, index):
    """Device-coherent i32 load (a spin read)."""
    return fx.Int32(
        fx.generic_load(
            _ptr_at(base_i64, index),
            volatile=True,
            memory_order=fx.AtomicOrdering.Monotonic,
            syncscope="agent",
        )
    )


def tdm_store_fits(feat_dim: int, topk: int, slots: int, waves: int) -> bool:
    """Whether the TDM-store variant serves this geometry: 8-token gather groups
    dealt evenly over the waves, and the payload rows next to the hidden
    staging in LDS."""
    tokens = slots * waves
    if slots > MAX_SLOTS_TDM or tokens % 8 or (topk * tokens // 8) % waves:
        return False
    staging = min(slots, 2) * waves * feat_dim * 2
    payload = tokens * (feat_dim // 2 + 4 * topk)  # MXFP4 rows + route-row table
    return staging + payload <= LDS_BUDGET


def build_moe_topk_route_quant_prefill_module(
    *,
    experts: int,
    topk: int,
    feat_dim: int,
    align_m: int,
    slots: int,
    waves: int = 32,
    quant_mode: str = "fp4",
    tdm_store: bool = False,
):
    """JIT launcher ``(logits, hidden, num_valid_rows, topk_weights, topk_ids,
    topids_to_rows, masked_m, psum, payload, compact_scale, row_to_token,
    status, token_num, logits_row_stride, tokens_per_wg, grid_wgs)``.

    ``tokens_per_wg <= slots * waves``, ``grid_wgs * tokens_per_wg >=
    token_num`` and ``grid_wgs`` at most the workgroups the GPU holds at once
    (one a CU). ``compact_scale`` (token-major e8m0) and ``row_to_token`` feed
    ``build_moe_prefill_scale_preshuffle_module``.
    """
    if experts != 256:
        raise NotImplementedError(
            f"prefill top-k/route/quant: 256 experts, got {experts}"
        )
    if not 1 <= topk <= min(LANES_PER_ROW, 16):
        raise NotImplementedError(f"topk {topk} unsupported")
    if align_m <= 0 or align_m & (align_m - 1):
        raise ValueError(f"align_m {align_m} must be a power of two")
    max_slots = MAX_SLOTS_TDM if tdm_store else MAX_SLOTS
    if not 1 <= slots <= max_slots:
        raise ValueError(f"slots {slots} outside 1..{max_slots}")
    if waves not in WAVE_CHOICES:
        raise ValueError(f"waves {waves} not in {WAVE_CHOICES}")
    L = _quant_layout(feat_dim, quant_mode, 1)
    if not L.use_pk8 or L.lanes_per_mx_block != 4:
        raise NotImplementedError("prefill top-k/route/quant requires gfx1250 pk8")
    wave_size = L.wave_size
    block = waves * wave_size
    if block < experts:
        raise NotImplementedError("a thread an expert")
    k_iters = L.block_iters  # 256 columns a wave iteration
    sd_per_row = L.scale_dwords_per_row  # e8m0 dwords a row (16 at 2048)
    if sd_per_row != 2 * k_iters or sd_per_row > wave_size or sd_per_row % 4:
        raise NotImplementedError("a wave iteration must pack two scale dwords")
    row_bytes = feat_dim * 2
    vals_per_lane = experts // LANES_PER_ROW  # 8 logits a lane
    experts_per_lane = experts // wave_size
    nbuf = min(slots, 2)
    buf_bytes = waves * row_bytes
    if nbuf * buf_bytes > LDS_BUDGET:
        raise NotImplementedError(f"hidden staging {nbuf * buf_bytes} B exceeds LDS")
    if tdm_store and not tdm_store_fits(feat_dim, topk, slots, waves):
        raise NotImplementedError(
            f"TDM store does not fit slots {slots} x waves {waves}"
        )
    n_tok_wg = slots * waves
    n_pad_items = experts * align_m
    st_depart = _status_depart(experts)
    st_error = _status_error(experts)
    n_reset = experts + 1  # everything but the sticky error word

    if tdm_store:

        @fx.struct
        class _Storage:
            cnt: fx.Array[fx.Int32, experts, 16]
            woff: fx.Array[fx.Int32, experts, 16]
            start: fx.Array[fx.Int32, experts, 16]
            tot: fx.Array[fx.Int32, experts, 16]
            misc: fx.Array[fx.Int32, 4, 16]
            # a slot-major token's MXFP4 row (token slot * waves + wave)
            pay: fx.Array[fx.Int32, n_tok_wg * L.payload_bytes_per_row // 4, 16]
            # that token's grouped row of each route (the OOB bound if none)
            rmap: fx.Array[fx.Int32, n_tok_wg * topk, 16]

    else:

        @fx.struct
        class _Storage:
            cnt: fx.Array[fx.Int32, experts, 16]
            woff: fx.Array[fx.Int32, experts, 16]
            start: fx.Array[fx.Int32, experts, 16]
            tot: fx.Array[fx.Int32, experts, 16]
            misc: fx.Array[fx.Int32, 4, 16]

    module_name = (
        f"moe_topk_route_quant_prefill_e{experts}_k{topk}_fd{feat_dim}"
        f"_a{align_m}_s{slots}_w{waves}_{quant_mode}_{L.native_tag}"
    ) + ("_tdm" if tdm_store else "")

    @flyc.kernel(name=module_name, known_block_size=[block, 1, 1])
    def prefill_kernel(
        logits: fx.Pointer,  # (token_num, row_stride) bf16, routed columns first
        hidden: fx.Pointer,  # (token_num, feat_dim) bf16
        num_valid_rows: fx.Pointer,  # (1,) i32 or null
        topk_weights: fx.Pointer,  # (token_num, topk) f32 out
        topk_ids: fx.Pointer,  # (token_num, topk) i32 out
        topids_to_rows: fx.Pointer,  # (token_num * topk,) i32 out
        masked_m: fx.Pointer,  # (experts,) i32 out
        psum: fx.Pointer,  # (experts,) i32 out
        grouped_payload: fx.Pointer,  # (contiguous_m, payload_bytes) u8 out
        compact_scale: fx.Pointer,  # (token_num, feat_dim // 32) u8 out
        row_to_token: fx.Pointer,  # (contiguous_m,) i32 out (-1: no route)
        status: fx.Pointer,  # (status_words,) i32, zero at launch and at exit
        token_num: Int32,
        logits_row_stride: Int32,  # elements
        tokens_per_wg: Int32,
        grid_wgs: Int32,
    ):
        i32 = T.i32
        f32 = T.f32
        c0_i32 = fx.Int32(0)
        c1_i32 = fx.Int32(1)
        c4_i32 = fx.Int32(4)
        c23_i32 = fx.Int32(23)
        c254_i32 = fx.Int32(254)
        c0_f32 = fx.Float32(0.0)
        c_hi16 = fx.Uint32(0xFFFF0000)
        c_sign = fx.Uint32(0x80000000)
        c_emax = fx.Uint32(experts - 1)
        c_wave = fx.Int32(wave_size)

        tid = fx.Uint32(fx.thread_idx.x)
        bid = fx.Uint32(fx.block_idx.x)
        wid = fx.Uint32(rocdl.readfirstlane(i32, fx.Uint32(tid // c_wave)))
        lane = tid - wid * fx.Uint32(wave_size)
        n_tok = fx.Uint32(token_num)
        tpw = fx.Uint32(tokens_per_wg)
        n_wg = fx.Uint32(grid_wgs)
        t0 = bid * tpw
        t_end = fx.min(t0 + tpw, n_tok)
        st_base = fx.Int64(ptrtoint(status))
        st_p = _i32_ptr(status)

        alloc = fx.SharedAllocator()
        lds = alloc.allocate(_Storage).peek()
        lds_cnt = lds.cnt.ptr
        lds_woff = lds.woff.ptr
        lds_start = lds.start.ptr
        lds_tot = lds.tot.ptr
        lds_misc = lds.misc.ptr
        h_lds = alloc.allocate(nbuf * buf_bytes)._ptr
        h_lds_idx = fx.index_cast(T.index, ptrtoint(h_lds))
        lds_load128, _ = make_lds_copy_ops(128)

        # ---- prologue: every load whose address is known -------------------
        # Hidden rows by TDM, a wave its own row (rows past the workgroup:
        # no fetch). Bounded by the workgroup, not by num_valid_rows, so the
        # issue does not wait on that load.
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

        n_valid = nvr_is_set.select(
            fx.Uint32((nvr_raw < c0_i32).select(c0_i32, nvr_raw)), n_tok
        )
        valid_end = fx.min(t_end, n_valid)
        hg = fx.recast_iter(fx.Int8, hidden)

        def _issue_hidden(i):
            base_tok = t0 + fx.Uint32(i * waves)
            shape = (waves, row_bytes)
            desc = tdm_ops.make_tensor_descriptor_2d(
                global_ptr=fx.Tensor(
                    fx.make_view(
                        hg + fx.Int64(base_tok) * fx.Int64(row_bytes),
                        fx.make_layout(shape, (row_bytes, 1)),
                    )
                ),
                lds_memref=fx.Tensor(
                    fx.make_view(
                        fx.add_offset(h_lds, (i % nbuf) * buf_bytes),
                        fx.make_layout(shape, (row_bytes, 1)),
                    )
                ),
                global_offset=(0, 0),
                tensor_shape=shape,
                strides=(row_bytes, 1),
                tile_shape=shape,
                elem_bytes=1,
                num_warps=waves,
                oob_outer_bound=fx.Int32(valid_end) - fx.Int32(base_tok),
            )
            tdm_ops.tensor_load_2d(desc)

        issued = 0
        for i in range_constexpr(nbuf):
            _issue_hidden(i)
            issued += 1

        # The logits of every slot's row: 32 lanes a row, 8 logits a lane
        # (a 2-byte aligned row: odd row strides).
        g = lane
        col0 = fx.Int32(g) * fx.Int32(vals_per_lane)
        row_bytes_l = fx.Int32(logits_row_stride) * fx.Int32(2)
        logits_rsrc = buffer_ops.create_buffer_resource_from_addr(
            fx.Int64(ptrtoint(logits)),
            num_records_bytes=fx.Int32(token_num) * row_bytes_l,
        )
        rows = []
        logit_dw = []
        for i in range_constexpr(slots):
            row = t0 + fx.Uint32(i * waves) + wid
            rows.append(row)
            r_eff = (row < t_end).select(row, fx.Uint32(0))
            off = fx.Int32(r_eff) * row_bytes_l + col0 * fx.Int32(2)
            logit_dw.append(
                fx.Vector(
                    buffer_ops.buffer_load(
                        logits_rsrc, off >> c1_i32, vec_width=8, dtype=T.i16
                    )
                ).bitcast(fx.Int32)
            )

        if tid < fx.Uint32(experts):
            lds_cnt[tid] = c0_i32
        _lds_barrier()

        # ---- A1: top-k of each slot's row --------------------------------
        w_t = ptr_buf_tensor(topk_weights, fx.Float32)
        ids_t = ptr_buf_tensor(topk_ids)
        is_route = g < fx.Uint32(topk)
        my_e = []
        my_ls = []
        for i in range_constexpr(slots):
            row = rows[i]
            live = row < valid_end
            dw = logit_dw[i]
            # One unique u32 key per logit, ordered as the HIP top-k picks:
            # the f32 bits made monotonic over 255 - column (ties to the
            # lower column); -0 folded into +0 first.
            keys = []
            for j in range_constexpr(8):
                d = fx.Uint32(dw[j // 2])
                b = d << fx.Uint32(16) if j % 2 == 0 else d & c_hi16
                b = fx.Uint32((b.bitcast(fx.Float32) + c0_f32).bitcast(fx.Int32))
                flip = fx.Uint32(fx.Int32(b) >> fx.Int32(31)) | c_sign
                col = fx.Uint32(col0) + fx.Uint32(j)
                keys.append(((b ^ flip) & c_hi16) | (c_emax - col))
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
            # topkGatingSoftmax<need_renorm>: exp(l - l_max), summed in k
            # order, each multiplied by the sum's reciprocal.
            numer = [fx.exp(top_v[k] - top_v[0]) for k in range_constexpr(topk)]
            acc = numer[0]
            for k in range_constexpr(1, topk):
                acc = acc + numer[k]
            nz = acc != c0_f32
            scale = nz.select(fx.Float32(1.0) / acc, fx.Float32(1.0))
            e_g = top_e[0]
            n_g = numer[0]
            for k in range_constexpr(1, topk):
                is_k = g == fx.Uint32(k)
                e_g = is_k.select(top_e[k], e_g)
                n_g = is_k.select(numer[k], n_g)
            w_g = nz.select(n_g * scale, n_g)
            route = fx.Int32(row) * fx.Int32(topk) + fx.Int32(g)
            if (row < t_end) & is_route:
                w_t[route] = live.select(w_g, c0_f32)
                ids_t[route] = live.select(e_g, fx.Int32(-1))
            ls = c0_i32
            if live & is_route:
                ls = fx.Int32(
                    fx.atomic_add(lds_cnt + e_g, c1_i32, syncscope="workgroup")
                )
            my_e.append(e_g)
            my_ls.append(ls)

        # ---- A2: quant of each slot's row from its LDS row ---------------
        c_lanes_per_block = fx.Int32(L.lanes_per_mx_block)
        block_in_wave = lane // fx.Uint32(c_lanes_per_block)
        lane_in_block = lane - block_in_wave * c_lanes_per_block
        cs_rsrc = buffer_ops.create_buffer_resource_from_addr(
            fx.Int64(ptrtoint(compact_scale)),
            num_records_bytes=fx.Int32(n_tok * fx.Uint32(sd_per_row * 4)),
        )

        def _qc(row, preloaded):
            return SimpleNamespace(
                i32=i32,
                f32=f32,
                block_iters=k_iters,
                mx_group_base=c0_i32,
                payload_base=fx.Int64(ptrtoint(grouped_payload)),
                payload_bytes_per_row=L.payload_bytes_per_row,
                hidden_base=fx.Int64(ptrtoint(hidden)),
                feat_bytes_per_row=row_bytes,
                feat_row_i32=row,
                hidden_preloaded=preloaded,
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
                c_wmma_rep=c1_i32,
                block_in_wave=block_in_wave,
                lane_in_block=lane_in_block,
                is_block_lead=lane_in_block == c0_i32,
                make_dests=lambda: (None, None),
            )

        quant = []
        it_of = lane >> fx.Uint32(1)
        hi_of = (lane & fx.Uint32(1)) == fx.Uint32(1)
        for i in range_constexpr(slots):
            # This wave's later rows may stay in flight.
            tdm_ops.tensor_wait(issued - (i + 1))
            row_off = fx.Int32((i % nbuf) * buf_bytes) + fx.Int32(wid) * fx.Int32(
                row_bytes
            )
            pre = [
                lds_load128(
                    h_lds_idx,
                    row_off + fx.Int32(it * 512) + fx.Int32(lane) * fx.Int32(16),
                )
                for it in range_constexpr(k_iters)
            ]
            qc = _qc(rows[i], pre)
            results, _ = _emit_quant_compute(qc)
            quant.append((qc, results))
            if const_expr(tdm_store):
                # The payload to this token's LDS row (the byte layout of
                # _emit_payload_stores); it does not stay in VGPRs across B.
                j_tok = fx.Int32(i * waves) + fx.Int32(wid)
                for mx_block, payload_val, _ in results:
                    lds.pay.ptr[
                        j_tok * fx.Int32(L.payload_bytes_per_row // 4)
                        + fx.Int32(mx_block) * fx.Int32(L.payload_bytes_per_block // 4)
                        + fx.Int32(lane_in_block)
                    ] = fx.Int32(payload_val)
            if const_expr(i + nbuf < slots):
                # The buffer is free once this slot's LDS reads have landed.
                rocdl.s_wait_dscnt(0)
                _issue_hidden(i + nbuf)
                issued += 1
            # The e8m0 row, token-major: scale dword 2 * it + h sits on lane
            # 16 h of iteration it; gathered onto lanes 0..15, one 64 B store.
            sel = c0_i32
            for it in range_constexpr(k_iters):
                packed = _pack_block_group_dword(qc, results[it][2])
                lo = fx.Int32(rocdl.readlane(i32, packed, 0))
                hi = fx.Int32(rocdl.readlane(i32, packed, 16))
                sel = (it_of == fx.Uint32(it)).select(hi_of.select(hi, lo), sel)
            if (rows[i] < valid_end) & (lane < fx.Uint32(sd_per_row)):
                buffer_ops.buffer_store(
                    sel,
                    cs_rsrc,
                    fx.Int32(rows[i]) * fx.Int32(sd_per_row) + fx.Int32(lane),
                )

        # ---- B: claim the runs, wait for every workgroup's counts --------
        _lds_barrier()
        if tid < fx.Uint32(experts):
            c = fx.Int32(lds_cnt[tid])
            add = c + fx.Int32(1 << COUNT_BITS)
            old = fx.Int32(fx.atomic_add(st_p + tid, add, syncscope="agent"))
            lds_woff[tid] = old & fx.Int32(COUNT_MASK)
            target = fx.Int32(n_wg) << fx.Int32(COUNT_BITS)
            cur = old + add
            spins = c0_i32
            while (cur < target) & (spins < fx.Int32(SPIN_LIMIT)):
                rocdl.s_sleep(1)
                cur = _load_agent(st_base, tid)
                spins = spins + c1_i32
            if cur < target:
                fx.atomic_or(st_p + fx.Int32(st_error), c1_i32, syncscope="agent")
            lds_tot[tid] = cur & fx.Int32(COUNT_MASK)
        _lds_barrier()

        # Every total of this workgroup is in LDS: depart (the last one out
        # resets the status words at its end).
        dep = c0_i32
        if tid == fx.Uint32(0):
            dep = fx.Int32(
                fx.atomic_add(st_p + fx.Int32(st_depart), c1_i32, syncscope="agent")
            )

        # Tile-aligned exclusive scan of the totals (wave 0, 8 experts a lane).
        if wid == fx.Uint32(0):
            e0 = fx.Int32(lane) * fx.Int32(experts_per_lane)
            cnts = [
                fx.Int32(lds_tot[e0 + fx.Int32(q)])
                for q in range_constexpr(experts_per_lane)
            ]
            padded = [(c + fx.Int32(align_m - 1)) & fx.Int32(-align_m) for c in cnts]
            incl = []
            run = c0_i32
            for q in range_constexpr(experts_per_lane):
                run = run + padded[q]
                incl.append(run)
            lane_incl = run
            for d in (1, 2, 4, 8, 16):
                up = fx.Int32(gpu.shuffle_up(lane_incl, fx.Int32(d), c_wave))
                lane_incl = (lane >= fx.Uint32(d)).select(lane_incl + up, lane_incl)
            lane_excl = lane_incl - run
            is_writer = bid == fx.Uint32(0)
            for q in range_constexpr(experts_per_lane):
                start = lane_excl + incl[q] - padded[q]
                lds_start[e0 + fx.Int32(q)] = start
                if is_writer:
                    ptr_buf_tensor(masked_m)[e0 + fx.Int32(q)] = cnts[q]
                    ptr_buf_tensor(psum)[e0 + fx.Int32(q)] = start + cnts[q]
        _lds_barrier()

        # ---- C: the maps and every route's payload ----------------------
        # TDM gather OOB bound: above every grouped row (a route's row is below
        # token_num * topk + experts * align_m); the hardware drops indices >= it.
        oob_row = fx.Int32(token_num) * fx.Int32(topk) + fx.Int32(experts * align_m)
        rows_t = ptr_buf_tensor(topids_to_rows)
        r2t_t = ptr_buf_tensor(row_to_token)
        pay_base = fx.Int64(ptrtoint(grouped_payload))
        for i in range_constexpr(slots):
            row = rows[i]
            live = row < valid_end
            l_valid = live & is_route
            e_eff = l_valid.select(my_e[i], c0_i32)
            l_row = l_valid.select(
                fx.Int32(lds_start[e_eff]) + fx.Int32(lds_woff[e_eff]) + my_ls[i],
                fx.Int32(DROPPED_ROUTE_ROW),
            )
            if (row < t_end) & is_route:
                rows_t[fx.Int32(row) * fx.Int32(topk) + fx.Int32(g)] = l_row
            if l_valid:
                r2t_t[l_row] = fx.Int32(row)
            if const_expr(tdm_store):
                # Route g's grouped row for the gather stores below.
                if is_route:
                    lds.rmap.ptr[
                        (fx.Int32(i * waves) + fx.Int32(wid)) * fx.Int32(topk)
                        + fx.Int32(g)
                    ] = l_valid.select(l_row, oob_row)
            else:
                # From the registers, a grouped row at a time (whole 1 KB runs).
                # Every route's descriptor first: rebuilding one in the SGPRs the
                # previous row's stores read makes each row wait (s_wait_xcnt).
                qc, results = quant[i]
                rsrcs = []
                for k in range_constexpr(topk):
                    r_k = live.select(fx.Int32(rocdl.readlane(i32, l_row, k)), c0_i32)
                    rsrcs.append(
                        buffer_ops.create_buffer_resource_from_addr(
                            pay_base
                            + fx.Uint64(fx.Uint32(r_k)) * L.payload_bytes_per_row,
                            num_records_bytes=live.select(
                                fx.Int32(L.payload_bytes_per_row), c0_i32
                            ),
                        )
                    )
                for rsrc in rsrcs:
                    for mx_block, payload_val, _ in results:
                        _emit_payload_stores(qc, [rsrc], payload_val, mx_block)

        if const_expr(tdm_store):
            # Every wave stores 8-token x 1-route groups: the LDS rows of tokens
            # 8 grp .. 8 grp + 7 to their route k rows (OOB rows dropped).
            _lds_barrier()
            pay_row = L.payload_bytes_per_row
            pay_view = fx.Tensor(
                fx.make_view(
                    fx.recast_iter(fx.Int8, grouped_payload),
                    fx.make_layout((oob_row, pay_row), (pay_row, 1)),
                )
            )
            pay_lds_idx = fx.index_cast(T.index, ptrtoint(lds.pay.ptr))
            for n in range_constexpr(topk * n_tok_wg // 8 // waves):
                o = fx.Int32(wid) + fx.Int32(n * waves)
                k_r = o % fx.Int32(topk)
                grp = o // fx.Int32(topk)
                idxs = [
                    fx.Int32(
                        rocdl.readfirstlane(
                            i32,
                            fx.Int32(
                                lds.rmap.ptr[
                                    (grp * fx.Int32(8) + fx.Int32(q)) * fx.Int32(topk)
                                    + k_r
                                ]
                            ),
                        )
                    )
                    for q in range_constexpr(8)
                ]
                tensor_store_gather(
                    make_tensor_gather_descriptor(
                        pay_view,
                        pay_lds_idx,
                        idxs,
                        row_width=pay_row,
                        tensor_dim0=pay_row,
                        tensor_dim1=oob_row,
                        stride=pay_row,
                        elem_bytes=1,
                        index_size=32,
                        lds_byte_offset=grp * fx.Int32(8 * pay_row),
                    )
                )
            tdm_ops.tensor_wait(0)

        # row_to_token -1 on each expert's tile padding rows: workgroup w
        # takes items [w * chunk, (w + 1) * chunk) of experts x align_m.
        # (Rows past the last expert stay unwritten: the preshuffle bounds
        # them by psum.)
        for idx in range(
            fx.Int32(bid) * fx.Int32(block) + fx.Int32(tid),
            fx.Int32(n_pad_items),
            fx.Int32(n_wg) * fx.Int32(block),
        ):
            e_p = idx >> fx.Int32(align_m.bit_length() - 1)
            p = idx & fx.Int32(align_m - 1)
            cnt_e = fx.Int32(lds_tot[e_p])
            pad_e = (cnt_e + fx.Int32(align_m - 1)) & fx.Int32(-align_m)
            if cnt_e + p < pad_e:
                r2t_t[fx.Int32(lds_start[e_p]) + cnt_e + p] = fx.Int32(-1)

        # The last workgroup out zeroes the status for the next launch.
        if tid == fx.Uint32(0):
            lds_misc[0] = dep
        gpu.barrier()
        if fx.Int32(lds_misc[0]) == fx.Int32(n_wg) - c1_i32:
            for q in range_constexpr(-(-n_reset // block)):
                w_idx = tid + fx.Uint32(q * block)
                if w_idx < fx.Uint32(n_reset):
                    fx.atomic_xchg(st_p + w_idx, c0_i32, syncscope="agent")

    @flyc.jit
    def launch_prefill(
        logits: fx.Pointer,
        hidden: fx.Pointer,
        num_valid_rows: fx.Pointer,
        topk_weights: fx.Pointer,
        topk_ids: fx.Pointer,
        topids_to_rows: fx.Pointer,
        masked_m: fx.Pointer,
        psum: fx.Pointer,
        grouped_payload: fx.Pointer,
        compact_scale: fx.Pointer,
        row_to_token: fx.Pointer,
        status: fx.Pointer,
        token_num: fx.Int32,
        logits_row_stride: fx.Int32,
        tokens_per_wg: fx.Int32,
        grid_wgs: fx.Int32,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        prefill_kernel(
            logits,
            hidden,
            num_valid_rows,
            topk_weights,
            topk_ids,
            topids_to_rows,
            masked_m,
            psum,
            grouped_payload,
            compact_scale,
            row_to_token,
            status,
            token_num,
            logits_row_stride,
            tokens_per_wg,
            grid_wgs,
        ).launch(
            grid=(fx.Int64(grid_wgs), 1, 1),
            block=(block, 1, 1),
            stream=stream,
        )

    launch_prefill.compile_hints = {
        "llvm_options": {
            "amdgpu-kernarg-preload": AITER_FLYDSL_KERNARG_PRELOAD,
            "amdgpu-kernarg-preload-count": AITER_FLYDSL_KERNARG_PRELOAD_COUNT,
        },
    }
    return launch_prefill


def build_moe_prefill_scale_preshuffle_module(row_bytes: int, wmma_rep: int):
    """``moe_scatter_preshuffle_scale`` (gather form, staged in LDS) bounded by
    the last expert's psum: rows at or past it are zero without a read of
    their ``rows_to_tokens`` entry, which the prefill kernel leaves unwritten
    (it writes the route rows and each expert's tile padding only).

    Launcher ``(src, dst, rows_to_tokens, psum_last, n_tiles)``: ``src`` the
    compact (tokens, row_bytes) e8m0 rows, ``dst`` MoE1's preshuffled
    (contiguous_m // wmma_rep, row_bytes * wmma_rep), ``psum_last`` a (1,)
    int32 view of psum[E - 1]; one workgroup a ``wmma_rep * 16``-row tile.
    """
    threads = 256
    assert row_bytes > 0 and row_bytes % 16 == 0, "whole 16 B runs a row"
    src_dwords = row_bytes // 4
    rows_per_tile = wmma_rep * 16
    units_per_tile = 16 * src_dwords * wmma_rep
    pitch = src_dwords + 1  # odd: the store pass's LDS column spans banks
    vec = 4
    stage_iters = -(-(units_per_tile // vec) // threads)

    @fx.struct
    class _TileStorage:
        buf: fx.Array[fx.Int32, rows_per_tile * pitch, 16]

    @flyc.kernel(
        name=f"moe_prefill_scale_preshuffle_b{row_bytes}_r{wmma_rep}",
        known_block_size=[threads, 1, 1],
    )
    def preshuffle_kernel(
        src: fx.Pointer,
        dst: fx.Pointer,
        rows_to_tokens: fx.Pointer,
        psum_last: fx.Pointer,
    ):
        tile = fx.Uint32(fx.block_idx.x)
        tid = fx.Uint32(fx.thread_idx.x)
        row_base = tile * fx.Uint32(rows_per_tile)
        tile_dword_base = tile * fx.Uint32(units_per_tile)
        map_p = ptr_buf_tensor(rows_to_tokens)
        src_p = ptr_buf_tensor(src)
        dst_p = ptr_buf_tensor(dst)
        bound = fx.Uint32(ptr_buf_tensor(psum_last)[fx.Int32(0)])
        tile_lds = fx.SharedAllocator().allocate(_TileStorage).peek().buf.ptr

        for it in range_constexpr(stage_iters):
            unit = (tid + fx.Uint32(it * threads)) * fx.Uint32(vec)
            if unit < fx.Uint32(units_per_tile):
                row = unit // fx.Uint32(src_dwords)
                sd = unit - row * fx.Uint32(src_dwords)
                grow = row_base + row
                in_rows = grow < bound
                srow = fx.Int32(map_p[in_rows.select(grow, fx.Uint32(0))])
                valid = in_rows & (srow >= fx.Int32(0))
                src_off = valid.select(
                    fx.Uint32(srow) * fx.Uint32(src_dwords) + sd, fx.Uint32(0)
                )
                for j in range_constexpr(vec):
                    v = fx.Uint32(src_p[src_off + fx.Uint32(j)])
                    tile_lds[row * fx.Uint32(pitch) + sd + fx.Uint32(j)] = fx.Int32(
                        valid.select(v, fx.Uint32(0))
                    )
        gpu.barrier()
        for it in range_constexpr(stage_iters):
            unit = (tid + fx.Uint32(it * threads)) * fx.Uint32(vec)
            if unit < fx.Uint32(units_per_tile):
                lane = unit % fx.Uint32(16)
                t2 = unit // fx.Uint32(16)
                w = t2 % fx.Uint32(wmma_rep)
                sd = t2 // fx.Uint32(wmma_rep)
                dst_off = (
                    tile_dword_base
                    + (sd * fx.Uint32(wmma_rep) + w) * fx.Uint32(16)
                    + lane
                )
                for j in range_constexpr(vec):
                    dst_p[dst_off + fx.Uint32(j)] = tile_lds[
                        (w * fx.Uint32(16) + lane + fx.Uint32(j)) * fx.Uint32(pitch)
                        + sd
                    ]

    @flyc.jit
    def launch_preshuffle(
        src: fx.Pointer,
        dst: fx.Pointer,
        rows_to_tokens: fx.Pointer,
        psum_last: fx.Pointer,
        n_tiles: fx.Int32,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        preshuffle_kernel(src, dst, rows_to_tokens, psum_last).launch(
            grid=(fx.Int64(n_tiles), 1, 1), block=(threads, 1, 1), stream=stream
        )

    launch_preshuffle.compile_hints = {
        "llvm_options": {
            "amdgpu-kernarg-preload": AITER_FLYDSL_KERNARG_PRELOAD,
            "amdgpu-kernarg-preload-count": AITER_FLYDSL_KERNARG_PRELOAD_COUNT,
        },
    }
    return launch_preshuffle
