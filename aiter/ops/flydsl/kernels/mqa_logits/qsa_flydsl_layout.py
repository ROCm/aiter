# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""FlyDSL MFMA kernel for the QSA indexer scorer (`qsa_paged_mqa_logits`), prefill.

Per index head the score is a GEMM, S_h = q[:, h, :] @ K^T, on bf16 MFMA
16x16x16; ReLU is applied per head before the heads are summed.

One block = 4 waves (256 threads) owns 64 query rows x 64 compressed groups
(4 pages). Wave 0 derives the block's request range and largest visible count
with a shuffle reduction; a block none of whose rows sees its first column
exits before reading Q (top-k reads each row only over [0, visible), so those
logits are never read). Otherwise each wave keeps its 16 rows of Q for all
heads in VGPRs, and per request present in the block the 4 pages of K are
staged in LDS (row stride 136 bf16) and consumed page by page with one
accumulator. Q and K share a head-dim permutation (lane group g, k-step s ->
dims 32g + 4s + [0..3]) so every lane reads 32 contiguous elements.

Covers index_heads=4, head_dim=128, compressed page size 16, compress_ratio 4,
bf16 Q/K; other inputs raise. int64 index tensors and a strided page table are
converted to int32 / contiguous before the launch. The launch can be captured
into a CUDA graph.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.compiler.protocol import dsl_size_of as sizeof

BLOCK_SIZE = 256
BM, BN = 64, 64
Q_HEADS = 4
K_HEADS = 1
HDIMS = 128
KV_CACHE_BLOCK_SIZE = 64
COMPRESS_RATIO = 4
PAGE_SIZE = KV_CACHE_BLOCK_SIZE // COMPRESS_RATIO


@flyc.kernel(known_block_size=(BLOCK_SIZE, 1, 1))
def _qsa_logits_layout_kernel(
    q: fx.Tensor,  # [rows, 4, 128] bf16, contiguous
    compressed_k_cache: fx.Tensor,  # [pages, 16, 1, 128] bf16, contiguous
    page_table: fx.Tensor,  # [reqs, width] int32
    token_to_request: fx.Tensor,  # [rows] int32
    query_positions: fx.Tensor,  # [rows] int32
    context_lens: fx.Tensor,  # [reqs] int32
    logits: fx.Tensor,  # [rows, num_columns] fp32, out
    visible_out: fx.Tensor,  # [rows] int32, out
    row_starts: fx.Tensor,  # [rows] int32, out
    rows: fx.Int32,
    num_columns: fx.Int32,
    num_pages: fx.Int32,
    num_requests: fx.Int32,
    width: fx.Int32,
    inv_divisor: fx.Constexpr[float],
):
    tid = fx.thread_idx.x
    bid = fx.block_idx.x
    wave_id = tid // 64
    lane_id = tid % 64
    block_layout = fx.make_layout(
        ((rows + BM - 1) // BM, (num_columns + BN - 1) // BN),
        ((num_columns + BN - 1) // BN, 1),
    )
    bidm, bidn = fx.idx2crd(bid, block_layout).unpack()

    def get_buffer_ptr(tensor, num_elem):
        return fx.rocdl.make_buffer_ptr(
            fx.recast_iter(fx.Int8, fx.get_iter(tensor)),
            num_records_bytes=num_elem * sizeof(tensor.dtype),
        )

    def gstore(g_byte_ptr, byte_offset, vec, N, dtype):
        nbytes = N * sizeof(dtype)
        dst = fx.make_view(g_byte_ptr + byte_offset, fx.make_layout(nbytes, 1))
        reg = fx.make_rmem_tensor(fx.make_layout(N, 1), dtype)
        copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy(nbytes * 8), fx.Int8)
        fx.memref_store_vec(vec, reg)
        fx.copy_atom_call(copy_atom, reg, dst)

    def _make_view(memref, shape, stride):
        return fx.make_view(fx.get_iter(memref), fx.make_layout(shape, stride))

    @fx.struct
    class LDS:
        k_tile: fx.Array[fx.BFloat16, BN * (HDIMS + 8), 16]
        row_valid_tag: fx.Array[fx.Int32, BM, 16]
        row_vis: fx.Array[fx.Int32, BM, 16]
        block_info: fx.Array[fx.Int32, 4, 16]  # req_low, req_high, vis_max

    lds = fx.SharedAllocator().allocate(LDS).peek()
    k_tile = lds.k_tile.view(fx.make_layout((BN, HDIMS), (HDIMS + 8, 1)))
    row_valid_tag = lds.row_valid_tag.view(fx.make_layout(BM, 1))
    row_vis = lds.row_vis.view(fx.make_layout(BM, 1))
    block_info = lds.block_info.view(fx.make_layout(4, 1))
    logits_ptr = get_buffer_ptr(logits, rows * num_columns)
    visible_out_ptr = get_buffer_ptr(visible_out, rows)
    row_starts_ptr = get_buffer_ptr(row_starts, rows)

    # 1. per-row metadata: visible count, request table, block reductions
    token_to_request = _make_view(token_to_request, rows, 1)
    query_positions = _make_view(query_positions, rows, 1)
    context_lens = _make_view(context_lens, num_requests, 1)
    row0 = BM * bidm
    row_thread = row0 + tid
    if tid < BM:
        row_valid = (
            row_thread < rows
        )  # rows past `rows` are clamped for loads and never written back
        row_thread = row_valid.select(row_thread, rows - 1)
        req = token_to_request[row_thread]
        req_valid = (req >= 0) & (req < num_requests)
        ctx_len = req_valid.select(context_lens[req_valid.select(req, 0)], 0)
        pos = query_positions[row_thread]
        # row_valid_tag holds the row's request id if the row exists,
        # -999 if the row is past `rows`,
        # and an id outside [0, num_requests) if the row exists but its request is invalid
        row_valid_tag[tid] = row_valid.select(req, -999)
        _1 = (pos + 1) // COMPRESS_RATIO
        _2 = ctx_len // COMPRESS_RATIO
        _3 = (_1 < _2).select(_1, _2)
        vis_out = (0 > _3).select(0, _3)
        row_vis[tid] = vis_out
        # This branch is exactly wave 0, one lane per row: a wave reduction yields
        # req_low / req_high / vis_max; the other waves read just these 3 after the barrier
        lane_ok = row_valid & req_valid
        red_low = lane_ok.select(req, fx.Int32(0x7FFFFFFF))
        red_high = lane_ok.select(req, fx.Int32(-1))
        red_vis = row_valid.select(vis_out, fx.Int32(0))
        for sh in fx.range_constexpr(6):
            peer_low = red_low.shuffle_xor(32 >> sh, 64)
            peer_high = red_high.shuffle_xor(32 >> sh, 64)
            peer_vis = red_vis.shuffle_xor(32 >> sh, 64)
            red_low = (peer_low < red_low).select(peer_low, red_low)
            red_high = (peer_high > red_high).select(peer_high, red_high)
            red_vis = (peer_vis > red_vis).select(peer_vis, red_vis)
        # all 64 lanes store the same value, so no lane-0 guard is needed
        block_info[0] = red_low
        block_info[1] = red_high
        block_info[2] = red_vis
        gstore(
            visible_out_ptr,
            byte_offset=(row_valid & (bidn == 0)).select(
                row_thread * sizeof(fx.Int32), 0x7FFFFFFF
            ),
            vec=fx.Vector.from_elements([vis_out], dtype=fx.Int32),
            N=1,
            dtype=fx.Int32,
        )
        gstore(
            row_starts_ptr,
            byte_offset=(row_valid & (bidn == 0)).select(
                row_thread * sizeof(fx.Int32), 0x7FFFFFFF
            ),
            vec=fx.Vector.from_elements([0], dtype=fx.Int32),
            N=1,
            dtype=fx.Int32,
        )

    fx.gpu.barrier()
    # visible_out and row_starts are written
    # request range of the block's 64 rows, and the largest visible count among them
    req_low = block_info[0]
    req_high = block_info[1]
    vis_max = block_info[2]
    no_valid = req_high < 0
    req_low = no_valid.select(0, req_low)
    req_high = no_valid.select(0, req_high)

    # first column of this block
    col0 = bidn * 64
    page0 = col0 // PAGE_SIZE
    # a block owns 64 rows, a wave 16 rows, each row has 4 heads
    q = fx.make_view(
        fx.get_iter(
            fx.rocdl.make_buffer_tensor(
                q,
                max_size=False,
                num_records_bytes=fx.Int64(rows) * fx.Int64(4 * 128) * fx.Int64(2),
            )
        ),
        fx.make_layout((rows, 4, 128), (4 * 128, 128, 1)),
    )
    logits = fx.make_view(
        fx.get_iter(
            fx.rocdl.make_buffer_tensor(
                logits,
                max_size=False,
                num_records_bytes=fx.Int64(rows) * fx.Int64(num_columns) * fx.Int64(4),
            )
        ),
        fx.make_layout((rows, num_columns), (num_columns, 1)),
    )
    page_table = fx.make_view(
        fx.get_iter(
            fx.rocdl.make_buffer_tensor(
                page_table,
                max_size=False,
                num_records_bytes=fx.Int64(num_requests * width * 4),
            )
        ),
        fx.make_layout((num_requests, width), (width, 1)),
    )
    kc = fx.make_view(
        fx.get_iter(
            fx.rocdl.make_buffer_tensor(
                compressed_k_cache,
                max_size=False,
                num_records_bytes=fx.Int64(num_pages * 16 * 1 * 128 * 2),
            )
        ),
        fx.make_layout((num_pages, 16, 1, 128), (16 * 128 * 1, 128 * 1, 128, 1)),
    )

    mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 16, fx.BFloat16))
    MFMA_K_STEP = 16  # head-dim width of one MFMA step
    MFMA_K_LOOPS = 128 // MFMA_K_STEP
    q_head0123 = [q[None, head, None] for head in fx.range_constexpr(4)]
    # the tiler must cover both modes: (64,128) -> (64,128,ceil(rows/64),1)
    q_head0123_block = [
        fx.flat_divide(qi, (64, 128))[None, None, bidm, 0] for qi in q_head0123
    ]  # [64,128]
    q_head0123_wave = [
        fx.flat_divide(qi, (16, 128))[None, None, wave_id, 0] for qi in q_head0123_block
    ]  # 4 x [16,128]
    # K permutation: at step s lane group g owns dims 32g+4s..32g+4s+3, so over 8 steps each lane reads 32 contiguous elements
    # (16,(4,4),8):(512,(1,32),4) -- row / (4 elements of a step, lane group g) / step s
    q_head0123_wave = [
        fx.make_view(
            fx.get_iter(qi), fx.make_layout((16, (4, 4), 8), (4 * 128, (1, 32), 4))
        )
        for qi in q_head0123_wave
    ]

    tiled_mma = fx.make_tiled_mma(mma_atom, fx.make_layout((1, 1, 1), (0, 1, 2)))
    thr_mma = tiled_mma.thr_slice(lane_id)
    q_frag = [
        [
            thr_mma.make_fragment_A(q_head0123_wave[h][None, None, 0])
            for s in fx.range_constexpr(MFMA_K_LOOPS)
        ]
        for h in fx.range_constexpr(4)
    ]  # q_frag[head][k step]
    # global q -> q frag. After the permutation each thread reads 32 contiguous bf16 over 8 steps, merged into 4 buffer_load_dwordx4
    copy_global_to_frag = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.BFloat16)
    copy_q = fx.make_tiled_copy_A(copy_global_to_frag, tiled_mma).get_slice(lane_id)
    q_global_tile_head0123 = [
        copy_q.partition_S(q_head0123_wave[i]) for i in fx.range_constexpr(4)
    ]

    logits = fx.flat_divide(logits, (64, 64))[None, None, bidm, bidn]
    logits = fx.flat_divide(logits, (16, 64))[
        None, None, wave_id, 0
    ]  # [64,64]->[16,64,4,1], the slice of logits this wave owns
    # split into 4 tiles; every wave computes all 4
    logits = fx.flat_divide(logits, (16, 16))[
        None, None, 0, None
    ]  # [16,64]->[16,16,1,4]->[16,16,4]
    logits_tile0123 = [
        logits[None, None, i] for i in fx.range_constexpr(4)
    ]  # each tile is [16,16]

    # the LDS K tile split into 4 pages, one per compressed page
    k_tile = fx.flat_divide(k_tile, (16, 128))[
        None, None, None, 0
    ]  # [64,128]->[16,128,4,1]->[16,128,4]
    # each [16,128] page gets the same K permutation as Q (LDS row stride HDIMS+8 against bank conflicts): (16,(4,4),8) = [N_TILE_SIZE, K_TILE_SIZE, inner_k_loops]
    k_tile_pages0123 = [
        fx.make_view(
            fx.get_iter(k_tile[None, None, pageid]),
            fx.make_layout((16, (4, 4), 8), (HDIMS + 8, (1, 32), 4)),
        )
        for pageid in fx.range_constexpr(4)
    ]
    # one page at a time: 8 fragments for that page's 8 k steps
    k_frag = [
        thr_mma.make_fragment_B(k_tile_pages0123[0][None, None, 0])
        for kk in fx.range_constexpr(MFMA_K_LOOPS)
    ]

    copy_k_lds_frag = fx.make_copy_atom(fx.UniversalCopy32b(), fx.BFloat16)
    copy_k = fx.make_tiled_copy_B(copy_k_lds_frag, tiled_mma).get_slice(lane_id)
    k_lds_tile0123 = [
        copy_k.partition_S(k_tile_pages0123[i]) for i in fx.range_constexpr(4)
    ]

    # K global -> registers -> LDS: thread t copies chunk t%16 (8 bf16 = 16 bytes) of slot t//16 of each page
    # partition outside the runtime loop; inside it only index page physical_page / page_offset
    g2r_atom = fx.make_copy_atom(fx.rocdl.BufferCopy(128), fx.BFloat16)
    r2s_atom = fx.make_copy_atom(fx.UniversalCopy128b(), fx.BFloat16)
    k_thr_layout = fx.make_layout((16, 16), (16, 1))
    k_val_layout = fx.make_layout((1, 8), (8, 1))
    g2r = fx.make_tiled_copy_tv(g2r_atom, k_thr_layout, k_val_layout).get_slice(tid)
    r2s = fx.make_tiled_copy_tv(r2s_atom, k_thr_layout, k_val_layout).get_slice(tid)
    kc_pages = fx.make_view(
        fx.get_iter(kc), fx.make_layout((16, 128, num_pages), (128, 1, 16 * 128))
    )  # [slot,dim,page]
    kc_thr = g2r.partition_S(kc_pages)  # the 8 bf16 this thread copies from every page
    k_lds_thr = r2s.partition_D(k_tile)  # k_tile: [16,128,4]
    k_reg = [
        fx.make_fragment_like(k_lds_thr[None, None, None, 0])
        for _ in fx.range_constexpr(4)
    ]

    # In a runtime for loop, x.method(...) on an x defined outside makes x loop-carried,
    # and ThrMma / ThrCopy cannot be loop-carried, so these calls stay outside the loop
    # a single accumulator (in a list: acc[0].fill in the loop is not treated as loop-carried)
    acc = [thr_mma.make_fragment_C(logits_tile0123[0])]
    q_frag_retiled = [
        [copy_q.retile(q_frag[h][s]) for s in fx.range_constexpr(MFMA_K_LOOPS)]
        for h in fx.range_constexpr(4)
    ]
    k_frag_retiled = [
        copy_k.retile(k_frag[kk]) for kk in fx.range_constexpr(MFMA_K_LOOPS)
    ]

    # Early exit: no row of the block sees col0 or beyond, and top-k reads each row only over [0, visible), so the block computes and writes nothing.
    # Q is loaded inside the if too, so skipped blocks never read it. Inside the if, methods on outside objects are equally off limits,
    # and loop variables need names not used outside it (qh / qs)
    if col0 < vis_max:
        # Q depends only on the rows: load it into registers once, outside the request loop
        for qh in fx.range_constexpr(4):
            for qs in fx.range_constexpr(MFMA_K_LOOPS):
                fx.copy(
                    copy_global_to_frag,
                    src=q_global_tile_head0123[qh][None, None, None, qs],
                    dst=q_frag_retiled[qh][qs],
                )

        # collective load k to lds
        for reqid in range(req_low, req_high + 1):
            reqid = fx.Int32(reqid)
            page_ok = []
            for page_offset in fx.range_constexpr(4):
                logical_page = page0 + page_offset
                logical_page_valid = logical_page < width
                logical_page = logical_page_valid.select(logical_page, 0)
                physical_page = page_table[reqid, logical_page]
                physical_page_valid = (physical_page >= 0) & (physical_page < num_pages)
                physical_page = physical_page_valid.select(physical_page, 0)
                page_ok.append(physical_page_valid & logical_page_valid)
                fx.copy(
                    g2r_atom,
                    src=kc_thr[None, None, None, physical_page],
                    dst=k_reg[page_offset],
                )
            for page_offset in fx.range_constexpr(4):
                fx.copy(
                    r2s_atom,
                    src=k_reg[page_offset],
                    dst=k_lds_thr[None, None, None, page_offset],
                )
            # all 4 pages of K are in LDS
            # MFMA: each wave's Q fragments against all 4 K pages
            fx.gpu.barrier()
            NEG_INF = fx.Float32(float("-inf"))
            DROP = fx.Int32(0x7FFFFFFF)
            r_grp = lane_id // 16  # C fragment: row = 4*r_grp + jr
            c_in = lane_id % 16  #             column = c_in
            # the 4 rows this lane writes: row, visible count, valid, and whether this pass writes it
            own = []
            for jr in fx.range_constexpr(4):
                row_local = (
                    wave_id * 16 + r_grp * 4 + jr
                )  # row within the block (0..63)
                tag = row_valid_tag[row_local]
                valid = (tag >= 0) & (tag < num_requests)
                invalid = (tag != -999) & ((tag < 0) | (tag >= num_requests))
                owned = (valid & (tag == reqid)) | (invalid & (reqid == req_low))
                own.append((row0 + row_local, row_vis[row_local], valid, owned))

            # page outer, head middle, k step inner: one accumulator at a time, each page written back when done
            for pg in fx.range_constexpr(4):
                for kk in fx.range_constexpr(MFMA_K_LOOPS):
                    fx.copy(
                        copy_k_lds_frag,
                        src=k_lds_tile0123[pg][None, None, None, kk],
                        dst=k_frag_retiled[kk],
                    )
                score = None
                for hd in fx.range_constexpr(4):
                    acc[0].fill(0.0)
                    for kk in fx.range_constexpr(MFMA_K_LOOPS):
                        fx.gemm(thr_mma, acc[0], q_frag[hd][kk], k_frag[kk], acc[0])
                    v = acc[0].load()
                    v = v.maximumf(fx.Vector.zeros_like(v))  # ReLU per head
                    score = v if score is None else score + v  # then sum over heads
                score = score * inv_divisor

                col = col0 + pg * 16 + c_in
                for jr in fx.range_constexpr(4):
                    row, vis, valid, owned = own[jr]
                    keep = valid & (col < vis) & page_ok[pg]
                    val = keep.select(fx.Float32(score[jr]), NEG_INF)
                    off = (owned & (col < num_columns)).select(
                        (row * num_columns + col) * 4, DROP
                    )
                    gstore(
                        logits_ptr,
                        off,
                        fx.Vector.from_elements([val], dtype=fx.Float32),
                        1,
                        fx.Float32,
                    )
            fx.gpu.barrier()  # every wave is done with this pass's K before the next pass overwrites LDS


@flyc.jit
def _qsa_logits_layout_launch(
    q: fx.Tensor,
    compressed_k_cache: fx.Tensor,
    page_table: fx.Tensor,
    token_to_request: fx.Tensor,
    query_positions: fx.Tensor,
    context_lens: fx.Tensor,
    logits: fx.Tensor,
    visible_out: fx.Tensor,
    row_starts: fx.Tensor,
    rows: fx.Int32,
    num_columns: fx.Int32,
    num_pages: fx.Int32,
    num_requests: fx.Int32,
    width: fx.Int32,
    grid: fx.Int32,
    inv_divisor: fx.Constexpr[float],
    stream: fx.Stream,
):
    _qsa_logits_layout_kernel(
        q,
        compressed_k_cache,
        page_table,
        token_to_request,
        query_positions,
        context_lens,
        logits,
        visible_out,
        row_starts,
        rows,
        num_columns,
        num_pages,
        num_requests,
        width,
        inv_divisor,
    ).launch(grid=(grid, 1, 1), block=(BLOCK_SIZE, 1, 1), stream=stream)


# Buffer offsets are i32 bytes.
_MAX_BYTES = 2**31 - 1


def _cdiv(a: int, b: int) -> int:
    return -(-a // b)


def _check_supported(
    q: torch.Tensor,
    compressed_k_cache: torch.Tensor,
    page_table: torch.Tensor,
    compress_ratio: int,
    columns: int,
) -> None:
    """The kernel hard-codes these; anything else would be silently mis-scored."""
    if compress_ratio != COMPRESS_RATIO:
        raise ValueError(
            f"compress_ratio must be {COMPRESS_RATIO}, got {compress_ratio}"
        )
    if q.dtype != torch.bfloat16 or compressed_k_cache.dtype != torch.bfloat16:
        raise ValueError("q and compressed_k_cache must be bf16")
    if tuple(q.shape[1:]) != (Q_HEADS, HDIMS) or not q.is_contiguous():
        raise ValueError(f"q must be a contiguous [tokens, {Q_HEADS}, {HDIMS}] tensor")
    if (
        tuple(compressed_k_cache.shape[1:]) != (PAGE_SIZE, K_HEADS, HDIMS)
        or not compressed_k_cache.is_contiguous()
    ):
        raise ValueError(
            "compressed_k_cache must be a contiguous "
            f"[pages, {PAGE_SIZE}, {K_HEADS}, {HDIMS}] tensor"
        )
    if page_table.shape[0] == 0:
        raise ValueError("page_table must have at least one request")
    if q.shape[0] * columns * 4 > _MAX_BYTES:
        raise ValueError("logits exceed the kernel's 2 GiB buffer addressing")
    if compressed_k_cache.numel() * 2 > _MAX_BYTES:
        raise ValueError(
            "compressed_k_cache exceeds the kernel's 2 GiB buffer addressing"
        )


def qsa_paged_mqa_logits_flydsl(
    q: torch.Tensor,
    compressed_k_cache: torch.Tensor,
    page_table: torch.Tensor,
    token_to_request: torch.Tensor,
    query_positions: torch.Tensor,
    context_lens: torch.Tensor,
    compress_ratio: int,
    divisor: float,
    logits: torch.Tensor,
    visible_groups: torch.Tensor,
    row_starts: torch.Tensor | None,
) -> None:
    """Fill `logits` [rows, columns] fp32 over each row's [0, visible) and
    `visible_groups` [rows] int32 in place."""
    rows, columns = logits.shape
    _check_supported(q, compressed_k_cache, page_table, compress_ratio, columns)
    if row_starts is None:
        row_starts = torch.empty(rows, dtype=torch.int32, device=q.device)
    # The kernel reads int32 indices and uses `width` as the page-table row stride.
    page_table = page_table.to(torch.int32).contiguous()
    token_to_request = token_to_request.to(torch.int32)
    query_positions = query_positions.to(torch.int32)
    context_lens = context_lens.to(torch.int32)
    _qsa_logits_layout_launch(
        q,
        compressed_k_cache,
        page_table,
        token_to_request,
        query_positions,
        context_lens,
        logits,
        visible_groups,
        row_starts,
        rows,
        columns,
        compressed_k_cache.shape[0],
        page_table.shape[0],
        page_table.shape[1],
        _cdiv(rows, BM) * _cdiv(columns, BN),
        1.0 / divisor,
        stream=torch.cuda.current_stream(),
    )
