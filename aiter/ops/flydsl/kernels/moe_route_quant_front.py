# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Route + stage-1 MX quant of a contiguous-M grouped MoE as ONE launch (gfx1250).

Small TP batches spend a launch on the route (``moe_route_psum_fused``), one on
the routed MX quant and, with a fused MXFP8 shared expert, one more on its
per-token MXFP8 quant. Each costs its ~2 us launch floor for well under a
microsecond of work. This kernel does all three.

The quant needs each route's grouped row, ``starts[expert] + slot``, and a
slot claimed with an atomic differs between workgroups. So the slot here is a
pure function of the routing instead: the number of lower tokens routed to the
same expert. Every workgroup re-derives the whole table in LDS and they all
agree, with no cross-workgroup wait:

    1. per expert, a bitmap of the tokens routed to it (LDS ``atomic_or``,
       order-independent);
    2. count = popcount of the row, rounded up to the tile;
    3. one warp per token: row of route k = the counts of the experts below
       its expert + the popcount of that expert's bitmap below this token;
    4. quantize the token once and store it to every routed row
       (``moe_token_multidest_quant``'s pass); with ``shared``, the last top-k
       column is the shared expert, whose rows stay in token order: its
       activations go out as MXFP8 into the per-token buffers instead.

Workgroup (0, 0) also scans the counts (the scan of ``moe_route_psum_fused``)
and writes ``masked_m`` / ``psum``; each warp of ``grid.y == 0`` writes its
token's ``topids_to_rows``.

A token routed twice to one expert gets one row for both routes (and the
expert one row less than the split route kernels give it): the two routes
read the same GEMM output, so the weighted sum is unchanged.
"""

from types import SimpleNamespace

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, ptrtoint, range_constexpr, rocdl
from flydsl.expr.typing import Int32, T

from aiter.ops.flydsl.kernels import buffer_ops
from aiter.ops.flydsl.kernels.moe_fused_route_quant_scatter import (
    _SCALE_RSRC_MAX_BYTES,
    BLOCK_THREADS,
    _emit_quant_block_loop,
    _quant_layout,
    _token_multidest_live_rows,
    _token_multidest_scale_base,
)
from aiter.ops.flydsl.kernels.moe_route_maps import DROPPED_ROUTE_ROW
from aiter.ops.flydsl.kernels.tensor_shim import (
    AITER_FLYDSL_KERNARG_PRELOAD,
    AITER_FLYDSL_KERNARG_PRELOAD_COUNT,
    buf_scalar_load,
    ptr_buf_tensor,
)

# One bitmap word per lane in the slot popcount, one expert per thread in the scan.
ROUTE_QUANT_FRONT_MAX_TOKENS = 1024
ROUTE_QUANT_FRONT_MAX_EXPERTS = BLOCK_THREADS


def route_quant_front_token_cap(token_num: int) -> int:
    """Token capacity the kernel is built for: the bitmap is a word per 32."""
    cap = 32
    while cap < token_num:
        cap *= 2
    return cap


def _scan_spare(offset, fresh, previous):
    """Buffer the scan step after ``offset`` writes: its first source is the
    table the warps still read, so the second step gets a fresh one."""
    return fresh if offset == 1 else previous


def _quant_ctx(L, c, *, block_iters, mx_group_base, payload, scale, rows, live):
    """``_emit_quant_block_loop`` context: one token's row to ``rows``."""
    c16_i32 = fx.Int32(16)
    c_rows_per_tile = fx.Int32(L.rows_per_tile)
    c_dst_scale_dwords_per_row = fx.Int32(L.dst_scale_dwords_per_row)
    # A dead token takes zero-length descriptors rather than a branch.
    pay_records = live.select(fx.Int32(L.payload_bytes_per_row), c.c0_i32)
    scale_records = live.select(fx.Int32(_SCALE_RSRC_MAX_BYTES), c.c0_i32)
    scales = [
        _token_multidest_scale_base(
            row, c_rows_per_tile, c_dst_scale_dwords_per_row, c16_i32
        )
        for row in rows
    ]
    return SimpleNamespace(
        i32=c.i32,
        f32=c.f32,
        block_iters=block_iters,
        payload_base=fx.Int64(ptrtoint(payload)),
        payload_bytes_per_row=L.payload_bytes_per_row,
        hidden_base=c.hidden_base,
        feat_bytes_per_row=c.feat_bytes_per_row,
        feat_row_i32=c.token,
        payload_num_records=pay_records,
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
        c0_i32=c.c0_i32,
        c1_i32=fx.Int32(1),
        c4_i32=fx.Int32(4),
        c23_i32=fx.Int32(23),
        c254_i32=fx.Int32(254),
        c0_f32=fx.Float32(0.0),
        c_wave=c.c_wave,
        c_elems_per_lane=fx.Int32(L.elems_per_lane),
        c_payload_bytes_per_block=fx.Int32(L.payload_bytes_per_block),
        c_payload_bytes_per_lane=fx.Int32(L.payload_bytes_per_lane),
        c_wmma_rep=fx.Int32(L.rows_per_tile // 16),
        block_in_wave=c.block_in_wave,
        lane_in_block=c.lane_in_block,
        is_block_lead=c.lane_in_block == c.c0_i32,
        payload_dests=[SimpleNamespace(payload_row_i32=row) for row in rows],
        dests=[
            SimpleNamespace(payload_row_i32=row, scale_row_dword_base=sc)
            for row, sc in zip(rows, scales)
        ],
        scale_t=ptr_buf_tensor(scale, fx.Int8, num_records_bytes=scale_records),
        scale_rsrc=buffer_ops.create_buffer_resource_from_addr(
            fx.Int64(ptrtoint(scale)), num_records_bytes=scale_records
        ),
        scale_pack_dwords=True,
        mx_group_base=mx_group_base,
        hidden_preloaded=c.hidden_rows,
    )


def build_moe_route_quant_front_module(
    feat_dim: int,
    wmma_rep: int,
    topk: int,
    experts: int,
    token_cap: int,
    ksplit: int = 1,
    shared: bool = False,
):
    """Return the JIT launcher of the fused route + stage-1 quant kernel.

    ``token_cap`` (a multiple of 32, see ``route_quant_front_token_cap``) bounds
    the token count of a launch; ``ksplit`` splits a row's K groups over
    ``grid.y``. With ``shared`` the last top-k column is the shared expert
    (``experts - 1``): its routes take the row of their token and its
    activations are quantized to MXFP8 into ``shared_payload`` /
    ``shared_scale`` (row = token); the routed columns stay MXFP4.

    Launcher signature::

        (hidden, grouped_payload, grouped_scale, shared_payload, shared_scale,
         topk_ids, topids_to_rows, masked_m, psum, token_num, tile_m,
         grid_blocks, stream=...)
    """
    L = _quant_layout(feat_dim, "fp4", wmma_rep)
    L8 = _quant_layout(feat_dim, "fp8", wmma_rep)
    if not L.use_pk8 or L.lanes_per_mx_block != 4:
        raise NotImplementedError("route + quant front kernel requires gfx1250 pk8")
    if not 0 < experts <= ROUTE_QUANT_FRONT_MAX_EXPERTS:
        raise ValueError(
            f"experts must be in 1..{ROUTE_QUANT_FRONT_MAX_EXPERTS}, got {experts}"
        )
    if token_cap % 32 or not 0 < token_cap <= ROUTE_QUANT_FRONT_MAX_TOKENS:
        raise ValueError(
            f"token_cap must be a multiple of 32 up to "
            f"{ROUTE_QUANT_FRONT_MAX_TOKENS}, got {token_cap}"
        )
    if L.block_iters % ksplit:
        raise ValueError(
            f"ksplit={ksplit} must divide the row's {L.block_iters} wave iterations"
        )
    if shared and topk < 2:
        raise ValueError("a shared expert column needs topk >= 2")
    wave_size = L.wave_size
    warps_per_block = L.warps_per_block
    block_iters = L.block_iters // ksplit
    n_routed = topk - 1 if shared else topk

    # Bitmap: ``words`` per expert, padded so the slot popcount's 32 lanes stay
    # inside it whatever the expert.
    words = token_cap // 32
    bitmap_sweeps = -(-(experts * words + wave_size) // BLOCK_THREADS)
    id_sweeps = -(-(token_cap * topk) // BLOCK_THREADS)
    count_words = -(-experts // wave_size)

    @fx.struct
    class _RouteStorage:
        bitmap: fx.Array[fx.Int32, bitmap_sweeps * BLOCK_THREADS, 16]
        aligned: fx.Array[fx.Int32, BLOCK_THREADS, 16]
        lds0: fx.Array[fx.Int32, BLOCK_THREADS, 16]
        lds1: fx.Array[fx.Int32, BLOCK_THREADS, 16]

    module_name = (
        f"moe_route_quant_front_k{topk}_e{experts}_fd{feat_dim}_r{wmma_rep}"
        f"_cap{token_cap}{f'_ks{ksplit}' if ksplit > 1 else ''}"
        f"{'_sh8' if shared else ''}"
    )

    @flyc.kernel(name=module_name, known_block_size=[BLOCK_THREADS, 1, 1])
    def front_kernel(
        hidden: fx.Pointer,  # (token_num, feat_dim) bf16
        grouped_payload: fx.Pointer,  # (contiguous_m, feat_dim // 2) MXFP4 out
        grouped_scale: fx.Pointer,  # preshuffled e8m0 out
        shared_payload: fx.Pointer,  # (shared_cap, feat_dim) MXFP8 out, iff shared
        shared_scale: fx.Pointer,  # preshuffled e8m0 out, iff shared
        topk_ids: fx.Pointer,  # (token_num * topk,) i32
        topids_to_rows: fx.Pointer,  # (token_num * topk,) i32 out
        masked_m: fx.Pointer,  # (experts,) i32 out
        psum: fx.Pointer,  # (experts,) i32 out (= m_tile_map)
        token_num: Int32,
        tile_m: Int32,
    ):
        i32 = T.i32
        c0_i32 = fx.Int32(0)
        c_wave = fx.Int32(wave_size)

        tid = fx.Uint32(fx.thread_idx.x)
        bid = fx.Uint32(fx.block_idx.x)
        k_group = fx.Uint32(fx.block_idx.y)
        tokens = fx.Uint32(token_num)
        numel = tokens * fx.Uint32(topk)

        lds = fx.SharedAllocator().allocate(_RouteStorage).peek()
        bitmap = lds.bitmap.ptr
        aligned = lds.aligned.ptr
        lds0 = lds.lds0.ptr
        lds1 = lds.lds1.ptr
        ids_t = ptr_buf_tensor(topk_ids)

        warp_in_block = fx.Uint32(rocdl.readfirstlane(i32, fx.Uint32(tid // c_wave)))
        lane = tid - warp_in_block * c_wave
        token0 = bid * fx.Int32(warps_per_block)
        token = token0 + warp_in_block
        valid = token < tokens
        token_eff = valid.select(token, fx.Uint32(0))
        route0 = token_eff * fx.Int32(topk)
        # This warp's experts. Scalar loads (the route is wave-uniform), issued
        # here so they are back by the time the table below is built.
        warp_ids = [
            fx.Int32(buf_scalar_load(ids_t, route0 + fx.Int32(k)))
            for k in range_constexpr(topk)
        ]

        # This warp's slice of its token's row, loaded once for both quant
        # passes and issued here so it arrives under the route table build.
        block_in_wave = lane // fx.Uint32(L.lanes_per_mx_block)
        lane_in_block = lane - block_in_wave * fx.Uint32(L.lanes_per_mx_block)
        hidden_base = fx.Int64(ptrtoint(hidden))
        hidden_rsrc = buffer_ops.create_buffer_resource_from_addr(
            hidden_base + fx.Uint64(token_eff) * (feat_dim * 2),
            num_records_bytes=feat_dim * 2,
        )
        mx_group_base = (
            k_group * fx.Int32(block_iters) if const_expr(ksplit > 1) else None
        )
        group0 = mx_group_base if const_expr(ksplit > 1) else c0_i32
        hidden_rows = []
        for it in range_constexpr(block_iters):
            mx_block = (group0 + fx.Int32(it)) * fx.Int32(
                L.mx_blocks_per_wave_iter
            ) + block_in_wave
            col_base = mx_block * fx.Int32(32) + lane_in_block * fx.Int32(
                L.elems_per_lane
            )
            hidden_rows.append(
                buffer_ops.buffer_load(
                    hidden_rsrc, col_base >> fx.Int32(1), vec_width=4, dtype=i32
                )
            )

        # ---- Route table, the same in every workgroup -------------------
        # This thread's routes, loaded while the bitmap clears.
        bit_words, bit_masks, routed = [], [], []
        for sweep in range_constexpr(id_sweeps):
            route = tid + sweep * BLOCK_THREADS
            in_range = route < numel
            e = fx.Int32(ids_t[in_range.select(route, fx.Uint32(0))])
            token_of = route // fx.Uint32(topk)
            routed.append(in_range & (e >= c0_i32) & (e < fx.Int32(experts)))
            bit_words.append(e * fx.Int32(words) + fx.Int32(token_of >> 5))
            bit_masks.append(fx.Int32(1) << fx.Int32(token_of & 31))
        for sweep in range_constexpr(bitmap_sweeps):
            bitmap[tid + sweep * BLOCK_THREADS] = c0_i32
        gpu.barrier()

        for sweep in range_constexpr(id_sweeps):
            if routed[sweep]:
                fx.atomic_or(
                    bitmap + bit_words[sweep], bit_masks[sweep], syncscope="workgroup"
                )
        gpu.barrier()

        # Tile-aligned row count of every expert: a warp sums the ones below
        # its experts itself, so only the writer needs the scan.
        in_expert = tid < fx.Uint32(experts)
        tile_v = fx.Uint32(tile_m)
        expert_row = in_expert.select(tid, fx.Uint32(0)) * fx.Uint32(words)
        count = c0_i32
        for word in range_constexpr(words):
            count = count + fx.Int32(fx.math.ctpop(fx.Int32(bitmap[expert_row + word])))
        if in_expert:
            aligned[tid] = fx.Int32((fx.Uint32(count) + tile_v - 1) // tile_v * tile_v)
        gpu.barrier()

        if (bid == 0) & (k_group == 0):
            # Inclusive scan of the aligned counts -> masked_m / psum.
            src = aligned
            dst = lds0
            for offset in range_constexpr(1, BLOCK_THREADS):
                if const_expr((offset & (offset - 1)) != 0):
                    continue
                if in_expert:
                    val = src[tid]
                    has_prev = tid >= offset
                    prev = fx.Int32(0)
                    if has_prev:
                        prev = src[tid - offset]
                    dst[tid] = val + prev
                gpu.barrier()
                # The first step leaves ``aligned`` intact for the warps below.
                src, dst = dst, _scan_spare(offset, lds1, src)
            if in_expert:
                is_not_first = tid != 0
                start = fx.Int32(0)
                if is_not_first:
                    start = src[tid - 1]
                ptr_buf_tensor(masked_m)[tid] = count
                ptr_buf_tensor(psum)[tid] = start + count

        # ---- One warp per token -----------------------------------------
        if token0 < tokens:
            # Lane w holds bitmap word w: the bits of the tokens below this one.
            token_word = token_eff >> 5
            below = (fx.Int32(1) << fx.Int32(token_eff & 31)) - 1
            lane_mask = (lane < token_word).select(
                fx.Int32(-1), (lane == token_word).select(below, c0_i32)
            )
            # ... and aligned count ``32 * j + w`` in ``below_counts[j]``.
            below_counts = [
                fx.Int32(aligned[fx.Uint32(32 * j) + lane])
                for j in range_constexpr(count_words)
            ]
            raw_rows = []
            for k in range_constexpr(topk):
                e = warp_ids[k]
                live = (e >= c0_i32) & (e < fx.Int32(experts))
                e_u = fx.Uint32(live.select(e, c0_i32))
                # Row = the rows of the experts below + the slot in this one:
                # one sum over the wave.
                if const_expr(shared and k == topk - 1):
                    # The shared expert's rows are in token order.
                    part = (lane == 0).select(fx.Int32(token_eff), c0_i32)
                else:
                    word = fx.Int32(bitmap[e_u * fx.Uint32(words) + lane])
                    part = fx.Int32(fx.math.ctpop(word & lane_mask))
                for j in range_constexpr(count_words):
                    part = part + ((fx.Uint32(32 * j) + lane) < e_u).select(
                        below_counts[j], c0_i32
                    )
                for dist in (1, 2, 4, 8, 16):
                    part = part + part.shuffle_xor(fx.Int32(dist), c_wave)
                row = live.select(part, fx.Int32(DROPPED_ROUTE_ROW))
                # Uniform, so each destination descriptor is too.
                raw_rows.append(fx.Int32(rocdl.readfirstlane(i32, row)))

            if (lane == 0) & valid & (k_group == 0):
                rows_t = ptr_buf_tensor(topids_to_rows)
                for k in range_constexpr(topk):
                    rows_t[route0 + fx.Int32(k)] = raw_rows[k]

            common = SimpleNamespace(
                i32=i32,
                f32=T.f32,
                c0_i32=c0_i32,
                c_wave=c_wave,
                hidden_base=hidden_base,
                feat_bytes_per_row=feat_dim * 2,
                token=token_eff,
                block_in_wave=block_in_wave,
                lane_in_block=lane_in_block,
                hidden_rows=hidden_rows,
            )
            rows, any_live = _token_multidest_live_rows(raw_rows[:n_routed])
            _emit_quant_block_loop(
                _quant_ctx(
                    L,
                    common,
                    block_iters=block_iters,
                    mx_group_base=mx_group_base,
                    payload=grouped_payload,
                    scale=grouped_scale,
                    rows=rows,
                    live=valid & any_live,
                )
            )
            if const_expr(shared):
                _emit_quant_block_loop(
                    _quant_ctx(
                        L8,
                        common,
                        block_iters=block_iters,
                        mx_group_base=mx_group_base,
                        payload=shared_payload,
                        scale=shared_scale,
                        rows=[token_eff],
                        live=valid & (raw_rows[topk - 1] >= c0_i32),
                    )
                )

    @flyc.jit
    def launch_front(
        hidden: fx.Pointer,
        grouped_payload: fx.Pointer,
        grouped_scale: fx.Pointer,
        shared_payload: fx.Pointer,
        shared_scale: fx.Pointer,
        topk_ids: fx.Pointer,
        topids_to_rows: fx.Pointer,
        masked_m: fx.Pointer,
        psum: fx.Pointer,
        token_num: fx.Int32,
        tile_m: fx.Int32,
        grid_blocks: fx.Int32,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        front_kernel(
            hidden,
            grouped_payload,
            grouped_scale,
            shared_payload,
            shared_scale,
            topk_ids,
            topids_to_rows,
            masked_m,
            psum,
            token_num,
            tile_m,
        ).launch(
            grid=(fx.Int64(grid_blocks), fx.Int64(ksplit), 1),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    launch_front.compile_hints = {
        "llvm_options": {
            "amdgpu-kernarg-preload": AITER_FLYDSL_KERNARG_PRELOAD,
            "amdgpu-kernarg-preload-count": AITER_FLYDSL_KERNARG_PRELOAD_COUNT,
        },
    }
    return launch_front
