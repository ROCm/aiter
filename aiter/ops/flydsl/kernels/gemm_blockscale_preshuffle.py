# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FP8 block-scale GEMM for gfx942: preshuffled B, MFMA 16x16x32, fp32 accumulation.

C = (A * scale_a) @ (B * scale_b)^T, with 128-wide K scale blocks.
M is a launch argument; N and K are compiled in.

Inputs/outputs:
  - A: row-major [M, K] e4m3fnuz.
  - B: [N, K] e4m3fnuz stored by shuffle_weight(w, layout=(16, 16)).
  - scale_a: fp32, K-major [K/128, M], per A row.
  - scale_b: fp32, row-major [ceil(N/128), K/128], per 128 x 128 B block.
  - C: row-major [M, N] bf16/fp16.
  - split_k > 1: split_k * M * N fp32 workspace elements and one zeroed Int32
    arrival counter per output tile; both allocations are reusable.

Paths:
  - A reaches LDS through registers or async DMA (use_async_copy); DMA also
    stages A scales when they fill whole waves.
  - Output uses direct stores or LDS CShuffle (use_cshuffle_epilog).
  - split_k > 1 uses direct fp32 partial stores; the last arrival reduces each
    tile into C and resets its counter for the next launch.

Compare the plan_lds result with device LDS capacity before compiling a new tile.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr.typing import Vector as Vec
from flydsl.runtime.device import get_rocm_arch

from aiter.ops.flydsl.gemm_tune.flydsl_gemm_a8w8_blockscale_bpreshuffle_common import (
    MFMA_MN,
    SCALE_BLOCK,
    WAVE,
    tile_validation_error,
)
from aiter.ops.flydsl.kernels.mfma_epilogues import mfma_epilog
from aiter.ops.flydsl.kernels.mfma_preshuffle_pipeline import (
    crd2idx,
    group_m_major,
    lds_store_8b_xor16,
    lds_store_16b_xor16,
    swizzle_xor16,
)
from aiter.ops.flydsl.kernels.splitk_epilogue import (
    CPOL_COHERENT,
    splitk_reduce_epilogue,
)

DEFAULT_NUM_WAVES = 4
# MI300X XCDs have separate L2 caches; fewer XCDs only reduce locality.
NUM_XCDS = 8
# M tiles per L2 locality group.
XCD_GROUP_M = 16
# gfx942 global-to-LDS DMA moves one dword per lane.
DMA_BYTES_PER_LANE = 4
# Placeholder extent; buffer record counts provide the runtime bounds.
FLAT_MAX_ELEMS = 2**31 - 1
# Split-K row-view extent; record counts bound M.
SPLIT_K_M_MAX = 65536
SPLIT_FLAG_BYTES = 4  # Int32 last-arrival flag in LDS


def cshuffle_ping_rows(tile_m):
    """Return ping's half of the output rows, rounded up to a 16-row MFMA boundary."""
    return MFMA_MN * ((tile_m // MFMA_MN + 1) // 2)


def stages_a_scales(tile_m, tile_k, use_async_copy):
    return use_async_copy and ((tile_k // SCALE_BLOCK) * tile_m) % WAVE == 0


def plan_lds(*, tile_m, tile_n, tile_k, use_cshuffle_epilog, use_async_copy, split_k=1):
    """Return total LDS bytes for both buffers and the optional split-K arrival flag."""
    # CShuffle reuses each A buffer for half the 2-byte output tile, rounded up.
    epilog_bytes = 2 * cshuffle_ping_rows(tile_m) * tile_n if use_cshuffle_epilog else 0
    buffer_bytes = max(tile_m * tile_k, epilog_bytes)
    if stages_a_scales(tile_m, tile_k, use_async_copy):
        buffer_bytes += (tile_k // SCALE_BLOCK) * tile_m * 4
    return 2 * buffer_bytes + (SPLIT_FLAG_BYTES if split_k > 1 else 0)


def hot_loop_slots(
    *,
    m_repeat,
    num_acc_n,
    n_per_wave,
    k_unroll,
    kb_per_tile,
    num_a_loads,
    num_a_dma_loads,
    num_scale_a_dmas,
    use_async_copy,
    stage_a_scales,
    spread_loads,
):
    """Return (VMEM slots per iteration, dswr_start); DS writes start at dswr_start - 1.

    With spread_loads, each load gets a slot in program order; async loads use
    only the first half of the step so they can drain before the closing barrier.
    """
    mfma_total = (k_unroll * 2) * m_repeat * num_acc_n
    mfma_per_iter = 2 * num_acc_n
    sche_iters = mfma_total // mfma_per_iter
    if spread_loads:
        # Aligned wave columns share a B scale when n_per_wave divides SCALE_BLOCK.
        num_scale_b_loads = 1 if SCALE_BLOCK % n_per_wave == 0 else num_acc_n
        num_vmem = (
            (num_a_dma_loads if use_async_copy else num_a_loads)
            + num_scale_a_dmas
            + kb_per_tile * ((0 if stage_a_scales else m_repeat) + num_scale_b_loads)
            + k_unroll * num_acc_n
        )
        vmem_iters = max(1, sche_iters // 2) if use_async_copy else sche_iters
        vmem_per_iter = [
            num_vmem // vmem_iters + (1 if i < num_vmem % vmem_iters else 0)
            for i in range(vmem_iters)
        ] + [0] * (sche_iters - vmem_iters)
    else:
        vmem_per_iter = [1] * sche_iters
    dswr_start = sche_iters - min(num_a_loads, sche_iters)
    return vmem_per_iter, dswr_start


def _flat_elems(t, num_records_bytes=None, elem=None, align=None):
    """Build a view sliced as (None, i), optionally bounded by num_records_bytes.

    Out-of-bounds loads return zero and stores drop; omitted bounds use max_size.
    """
    it = fx.get_iter(t)
    if elem is not None:
        # A pointer type overrides the source's byte alignment during recast.
        ptr_ty = fx.PointerType.get(
            elem.ir_type, address_space=fx.AddressSpace.Global, alignment=align
        )
        it = fx.recast_iter(ptr_ty, it)
    view = fx.Tensor(fx.make_view(it, fx.make_layout(FLAT_MAX_ELEMS, 1)))
    if num_records_bytes is None:
        buf = fx.rocdl.make_buffer_tensor(view, max_size=True)
    else:
        buf = fx.rocdl.make_buffer_tensor(
            view, max_size=False, num_records_bytes=num_records_bytes
        )
    return fx.logical_divide(buf, fx.make_layout(1, 1))


def _buffer_load(atom, view, idx, n, dtype, soffset=0):
    """soffset must be wave-uniform and is measured in elements."""
    reg = fx.make_rmem_tensor(fx.make_layout(n, 1), dtype)
    fx.copy(atom, fx.slice(view, (None, fx.Int32(idx))), reg, soffset=fx.Int32(soffset))
    return Vec(fx.memref_load_vec(reg))


def _buffer_store(atom, view, idx, vec, n, dtype):
    reg = fx.make_rmem_tensor(fx.make_layout(n, 1), dtype)
    fx.memref_store_vec(vec, reg)
    fx.copy(atom, reg, fx.slice(view, (None, fx.Int32(idx))))


def _index_coords(idx, layout):
    return tuple(fx.Index(c).ir_value() for c in fx.idx2crd(idx, layout).unpack())


def _xcd_remap_wgid(wgid, grid_m, grid_n):
    """(bx, by) for linear workgroup id ``wgid`` on a grid_m x grid_n grid, XCD-swizzled.

    Local because the shared ``xcd_remap_bx_by`` takes a 2-D block id and this kernel
    launches a 1-D tile grid. The hardware deals consecutive ids round-robin to the
    XCDs; this gives each XCD a contiguous run, then re-tiles the run with
    ``group_m_major``. The first ``count % NUM_XCDS`` XCDs own one extra workgroup, so
    later runs shift by ``min(xcd, rem)``; otherwise the map is not a permutation.
    """
    num_wgs = grid_n * grid_m
    per_xcd = num_wgs // NUM_XCDS
    rem = num_wgs % NUM_XCDS
    xcd = wgid % NUM_XCDS
    in_xcd = wgid // NUM_XCDS
    clip = (xcd < rem).select(xcd, rem)
    rotated = xcd * per_xcd + clip + in_xcd
    return group_m_major(rotated, grid_m, grid_n, fx.Index(XCD_GROUP_M))


def _tile_indices(num_m_tiles, num_n_tiles):
    # Index-typed: signed Int32 division and modulo would add sign fix-ups.
    wgid = fx.Index(fx.block_idx.x)
    return _xcd_remap_wgid(wgid, num_m_tiles, num_n_tiles)


def _layout_b_nk(n, k):
    """Logical (n, k) layout of B after ``shuffle_weight(layout=(16, 16))``, fp8.

    Local because the shared ``make_preshuffle_b_layout`` exposes only the physical
    5-D layout. Each 16-row by 64-byte block holds 4 klane groups of 16 contiguous
    k bytes; blocks run k-fastest within an n0 row of blocks.
    """
    k_blocks = k // 64
    n0 = fx.Int32(n // 16)
    k0 = fx.Int32(k_blocks)
    stride_nlane = fx.Int32(16)
    stride_klane = fx.Int32(16 * 16)
    stride_k0 = fx.Int32(4 * 16 * 16)
    stride_n0 = fx.Int32(k_blocks * 4 * 16 * 16)
    return fx.make_layout(
        ((16, n0), (16, 4, k0)),
        ((stride_nlane, stride_n0), (1, stride_klane, stride_k0)),
    )


def compile_blockscale_preshuffle_gemm(
    *,
    N: int,
    K: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    out_dtype: str = "bf16",
    use_cshuffle_epilog: bool = False,
    use_async_copy: bool = False,
    num_waves: int = DEFAULT_NUM_WAVES,
    split_k: int = 1,
):
    """Compile a launcher with runtime M after C, A, B and the scales.

    For split_k > 1, workspace and semaphore follow M, before the stream.
    """
    gpu_arch = str(get_rocm_arch()).split(":", 1)[0]
    if gpu_arch != "gfx942":
        raise ValueError(
            f"blockscale preshuffle GEMM supports gfx942 only, got {gpu_arch}"
        )
    if out_dtype not in ("fp16", "bf16"):
        raise ValueError(f"out_dtype must be 'fp16' or 'bf16', got {out_dtype!r}")
    validation_error = tile_validation_error(
        n=N,
        k=K,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        num_waves=num_waves,
        use_cshuffle_epilog=use_cshuffle_epilog,
        split_k=split_k,
    )
    if validation_error is not None:
        raise ValueError(validation_error)

    # Each K64 (ku) block supplies two K32 MFMAs: two 8-byte packs per lane.
    scale_k = K // SCALE_BLOCK
    scale_n = -(-N // SCALE_BLOCK)
    k_blocks16 = tile_k // 16
    kb_per_tile = tile_k // SCALE_BLOCK
    ku_per_kb = SCALE_BLOCK // 64
    kpack_elems = 16
    k_unroll = tile_k // 64
    m_repeat = tile_m // 16
    n_per_wave = tile_n // num_waves
    num_acc_n = n_per_wave // 16
    num_n_tiles = N // tile_n
    num_tiles = K // tile_k // split_k

    total_threads = num_waves * WAVE
    bytes_a_per_tile = tile_m * tile_k
    bytes_per_thread_a = bytes_a_per_tile // total_threads
    a_load_bytes = 16 if bytes_per_thread_a % 16 == 0 else 8
    a_chunk_dwords = a_load_bytes // 4
    num_a_loads = bytes_per_thread_a // a_load_bytes
    num_a_dma_loads = bytes_per_thread_a // DMA_BYTES_PER_LANE
    tile_k_dwords = tile_k // 4

    out_elem_cls = fx.BFloat16 if out_dtype == "bf16" else fx.Float16

    stage_a_scales = stages_a_scales(tile_m, tile_k, use_async_copy)
    num_scale_a_dmas = (kb_per_tile * tile_m) // WAVE if stage_a_scales else 0

    # A 16-row step has too few MFMAs to give every load its own scheduling slot.
    spread_loads = tile_m > MFMA_MN
    # Prefetch scales first only when their register footprint is small.
    scales_first = stage_a_scales or kb_per_tile * m_repeat <= 4

    vmem_per_iter, dswr_start = hot_loop_slots(
        m_repeat=m_repeat,
        num_acc_n=num_acc_n,
        n_per_wave=n_per_wave,
        k_unroll=k_unroll,
        kb_per_tile=kb_per_tile,
        num_a_loads=num_a_loads,
        num_a_dma_loads=num_a_dma_loads,
        num_scale_a_dmas=num_scale_a_dmas,
        use_async_copy=use_async_copy,
        stage_a_scales=stage_a_scales,
        spread_loads=spread_loads,
    )
    sche_iters = len(vmem_per_iter)

    cshuffle_nlane = 32
    cshuffle_mlane = total_threads // cshuffle_nlane
    cshuffle_vec = 4 if tile_n % (cshuffle_nlane * 4) == 0 else 2
    cshuffle_m_reps = tile_m // cshuffle_mlane
    cshuffle_n_reps = tile_n // (cshuffle_nlane * cshuffle_vec)

    epilog_tag = "cshuffle" if use_cshuffle_epilog else "direct"
    sas_tag = "_sas" if stage_a_scales else ""
    async_tag = "_ac" if use_async_copy else ""
    waves_tag = f"_w{num_waves}" if num_waves != DEFAULT_NUM_WAVES else ""
    sk_tag = f"_sk{split_k}" if split_k > 1 else ""

    kernel_name = (
        f"bs_gemm_{out_dtype}_{epilog_tag}{async_tag}{waves_tag}{sas_tag}{sk_tag}"
        f"_t{tile_m}x{tile_n}x{tile_k}"
    )

    def lds_buffer_bytes(async_copy):
        return (
            plan_lds(
                tile_m=tile_m,
                tile_n=tile_n,
                tile_k=tile_k,
                use_cshuffle_epilog=use_cshuffle_epilog,
                use_async_copy=async_copy,
            )
            // 2
        )

    # Scales follow the A/CShuffle region in each buffer.
    a_scale_lds_base = lds_buffer_bytes(False)
    buffer_size_bytes = lds_buffer_bytes(use_async_copy)

    # Separate LDS objects prove non-aliasing for the K-loop schedule.
    # They need not be adjacent.
    @fx.struct
    class SharedStorage:
        ping: fx.Array[fx.Uint8, buffer_size_bytes, 16]
        pong: fx.Array[fx.Uint8, buffer_size_bytes, 16]
        if split_k > 1:
            split_flag: fx.Array[fx.Int32, 1, SPLIT_FLAG_BYTES]

    def rows_view(t):
        return fx.Tensor(
            fx.make_view(fx.get_iter(t), fx.make_layout((SPLIT_K_M_MAX, N), (N, 1)))
        )

    def num_workgroups(i32_m):
        # Linearised so the XCD swizzle sees one workgroup id axis.
        return fx.ceildiv(i32_m, tile_m) * num_n_tiles

    @flyc.jit
    def gemm_body(
        arg_c, arg_a, arg_b, arg_scale_a, arg_scale_b, i32_m, split_args=None
    ):
        acc_init = fx.full(4, 0.0, fx.Float32)

        layout_b = _layout_b_nk(N, K)

        # Apply the bank swizzle to the column once per K block, avoiding per-read XORs.
        layout_lds = fx.make_layout((tile_m, tile_k), (tile_k, 1))

        tx = fx.Index(fx.thread_idx.x)

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()

        if const_expr(use_cshuffle_epilog):
            # Reuse ping/pong only after the epilogue barrier ends all operand reads.
            ping_rows = cshuffle_ping_rows(tile_m)
            lds_out_ping = fx.recast_iter(out_elem_cls, lds.ping.ptr)
            lds_out_pong = fx.recast_iter(out_elem_cls, lds.pong.ptr)

            def lds_out_row(row_block_start):
                # const_expr: a plain if's early return does not apply (runtime branch).
                if const_expr(row_block_start < ping_rows):
                    return lds_out_ping, 0
                return lds_out_pong, ping_rows

        # Record bounds zero A rows past M and drop C/workspace stores. Out-of-M
        # A scales may read the next K block, but multiply zero into dropped rows.
        m_index = fx.Index(i32_m)
        fp8 = fx.Float8E4M3FNUZ
        # Dword indexing keeps the A row term out of the K loop and saves registers.
        gA = _flat_elems(
            arg_a, num_records_bytes=fx.Int64(i32_m) * K, elem=fx.Int32, align=4
        )
        gB = _flat_elems(arg_b)
        gSA = _flat_elems(
            arg_scale_a, num_records_bytes=fx.Int64(i32_m) * (scale_k * 4)
        )
        gSB = _flat_elems(arg_scale_b)
        gC = _flat_elems(arg_c, num_records_bytes=fx.Int64(i32_m) * (N * 2))
        layout_a_dw = fx.make_layout((i32_m, K // 4), (K // 4, 1))
        layout_a_bytes = fx.make_layout((i32_m, K), (K, 1))
        layout_scale_a = fx.make_layout(
            (i32_m, (SCALE_BLOCK, scale_k)), (1, (0, i32_m))
        )
        layout_scale_b = fx.make_layout(
            ((SCALE_BLOCK, scale_n), (SCALE_BLOCK, scale_k)), ((0, scale_k), (0, 1))
        )
        layout_c = fx.make_layout((i32_m, N), (N, 1))
        # Tiled coordinates: (row, col, M tile, 1), or (row, col, M tile, N tile) for C.
        a_dw_tiles = fx.flat_divide(layout_a_dw, fx.make_tile(tile_m, K // 4))
        a_byte_tiles = fx.flat_divide(layout_a_bytes, fx.make_tile(tile_m, K))
        scale_a_tiles = fx.flat_divide(layout_scale_a, fx.make_tile(tile_m, K))
        c_tiles = fx.flat_divide(layout_c, fx.make_tile(tile_m, tile_n))
        if const_expr(split_k > 1):
            arg_workspace, arg_semaphore = split_args
            split_id = fx.Index(fx.block_idx.y)
            k_start = split_id * (num_tiles * tile_k)
            # Bound each workspace plane so tail rows cannot overwrite the next split.
            plane_elems = fx.Int64(i32_m) * N
            plane_iter = fx.add_offset(
                fx.get_iter(arg_workspace), fx.Int64(split_id) * plane_elems
            )
            gW = _flat_elems(
                fx.Tensor(fx.make_view(plane_iter, fx.make_layout(1, 1))),
                num_records_bytes=plane_elems * 4,
            )
        else:
            k_start = 0
        a_copy = fx.make_copy_atom(
            (
                fx.rocdl.BufferCopy128b()
                if a_load_bytes == 16
                else fx.rocdl.BufferCopy64b()
            ),
            fx.Int32,
        )
        b_copy = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fp8)
        scale_a_copy = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.Float32)
        scale_b_copy = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
        dma_copy = fx.make_copy_atom(
            fx.rocdl.BufferCopyLDS32b(), DMA_BYTES_PER_LANE * 8
        )
        if const_expr(use_async_copy):
            gA_dma = _flat_elems(
                arg_a, num_records_bytes=fx.Int64(i32_m) * K, elem=fx.Int8, align=4
            )
        if const_expr(stage_a_scales):
            gSA_dma = _flat_elems(
                arg_scale_a,
                num_records_bytes=fx.Int64(i32_m) * (scale_k * 4),
                elem=fx.Int8,
                align=4,
            )

        num_m_tiles = (m_index + (tile_m - 1)) // tile_m
        m_tile, n_tile = _tile_indices(num_m_tiles, num_n_tiles)

        layout_wave_lane = fx.make_layout((num_waves, WAVE), (WAVE, 1))
        wave_id, lane_id = _index_coords(fx.thread_idx.x, layout_wave_lane)

        layout_lane16 = fx.make_layout((4, 16), (16, 1))
        lane_div_16, lane_mod_16 = _index_coords(fx.Int32(lane_id), layout_lane16)

        row_a_lds = lane_mod_16
        lane_k_bytes = lane_div_16 * kpack_elems

        # (element, lane group, MFMA row tile): each lane holds four consecutive rows.
        layout_acc_row = fx.make_layout((4, 4, m_repeat), (1, 4, 16))
        # (lane % 16, MFMA column tile, wave, N tile); waves own consecutive columns.
        layout_n_col = fx.make_layout(
            (16, num_acc_n, num_waves, num_n_tiles), (1, 16, n_per_wave, tile_n)
        )

        # Fold each acc_row_step into an immediate offset from this lane's first row.
        acc_row0 = crd2idx((0, lane_div_16, 0), layout_acc_row)

        def acc_row_step(mi, ii=0):
            return crd2idx((ii, 0, mi), layout_acc_row)

        def n_col(ni, *, lane=lane_mod_16, tile=n_tile):
            return crd2idx(
                (fx.Int32(lane), ni, fx.Int32(wave_id), fx.Int32(tile)), layout_n_col
            )

        def n_col_step(ni):
            return crd2idx((0, ni, 0, 0), layout_n_col)

        b_cols = [n_col(i) for i in range_constexpr(num_acc_n)]

        layout_scale_a_stage = fx.make_layout((tile_m, kb_per_tile), (1, i32_m))
        layout_scale_a_lds = fx.make_layout((tile_m, kb_per_tile), (4, 4 * tile_m))

        layout_a_tile_dw = fx.make_layout((tile_m, tile_k_dwords), (tile_k_dwords, 1))

        def a_chunk_layout(chunk_dwords, num_loads):
            return fx.make_layout(
                (total_threads, num_loads),
                (chunk_dwords, total_threads * chunk_dwords),
            )

        layout_a_chunks = a_chunk_layout(a_chunk_dwords, num_a_loads)
        layout_a_dma_chunks = a_chunk_layout(1, num_a_dma_loads)

        def a_chunk_coord(i, layout_chunks):
            tile_idx = crd2idx((fx.Int32(tx), i), layout_chunks)
            return _index_coords(fx.Int32(tile_idx), layout_a_tile_dw)

        # Hoist invariant A slot offsets out of the K loop.
        a_slot_offs = []
        for i in range_constexpr(num_a_loads):
            row_a_local, col_a_local_i32 = a_chunk_coord(i, layout_a_chunks)
            a_slot_offs.append(
                crd2idx(
                    (
                        fx.Int32(row_a_local),
                        fx.Int32(col_a_local_i32),
                        fx.Int32(m_tile),
                        0,
                    ),
                    a_dw_tiles,
                )
            )

        mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 32, fp8))

        def k_offset_operands(idx, k_off):
            """Use SGPR soffset for uniform K offsets when spreading loads."""
            if const_expr(spread_loads):
                return idx, k_off
            return idx + k_off, 0

        def load_b_packs_k64(base_k, ku: int, ni: int):
            """Return two i64 B packs, one per MFMA, in the same K order as A."""
            k_lane = ku * 64 + lane_k_bytes
            idx_lane = crd2idx((fx.Int32(b_cols[ni]), fx.Int32(k_lane)), layout_b)
            idx_pack, soff = k_offset_operands(
                idx_lane, crd2idx((0, fx.Int32(base_k)), layout_b)
            )
            b_i64x2 = _buffer_load(b_copy, gB, idx_pack, 16, fp8, soff).bitcast(
                fx.Int64
            )
            return b_i64x2[0], b_i64x2[1]

        def load_b_tile(base_k):
            b_tile = []
            for ku in range_constexpr(k_unroll):
                packs0 = []
                packs1 = []
                for ni in range_constexpr(num_acc_n):
                    b0, b1 = load_b_packs_k64(base_k, ku, ni)
                    packs0.append(b0)
                    packs1.append(b1)
                b_tile.append((packs0, packs1))
            return b_tile

        def lds_load_16b(row_a, col_base, lds_buffer):
            col_base_swz = swizzle_xor16(row_a, col_base, k_blocks16)
            idx_a16 = crd2idx((fx.Int32(row_a), fx.Int32(col_base_swz)), layout_lds)
            ptr_off = fx.add_offset(lds_buffer.ptr, fx.make_int_tuple(idx_a16))
            return fx.make_view(ptr_off, fx.make_layout(16, 1)).load()

        def lds_load_packs_k64(row_a, col_base, lds_buffer):
            loaded_a16 = lds_load_16b(row_a, col_base, lds_buffer)
            a_i64x2 = Vec(loaded_a16).bitcast(fx.Int64)
            return a_i64x2[0], a_i64x2[1]

        def prefetch_a0_pack(lds_buffer):
            """Hide first-MFMA LDS latency; call after the publishing barrier."""
            return lds_load_packs_k64(row_a_lds, lane_k_bytes, lds_buffer)

        # Waves redundantly write the same scale slice; lane L lands at M0 + L*4.
        def stage_a_scales_to_lds_async(k_base, lds_buffer):
            scale_a_tile = crd2idx(
                (0, fx.Int32(k_base), fx.Int32(m_tile), 0), scale_a_tiles
            )
            layout_dma_slots = fx.make_layout((WAVE, num_scale_a_dmas), (1, WAVE))
            for it in range_constexpr(num_scale_a_dmas):
                slot = crd2idx((fx.Int32(lane_id), it), layout_dma_slots)
                scale_a_idx = scale_a_tile + crd2idx(
                    fx.Int32(slot), layout_scale_a_stage
                )
                # M0 must point to lane 0's slot.
                first_slot = crd2idx((0, it), layout_dma_slots)
                lds_ptr = fx.add_offset(
                    lds_buffer.ptr,
                    a_scale_lds_base + crd2idx(first_slot, layout_scale_a_lds),
                )
                fx.copy(
                    dma_copy,
                    fx.slice(gSA_dma, (None, fx.Int32(scale_a_idx * 4))),
                    fx.make_view(lds_ptr, fx.make_layout(1, 1)),
                )

        def lds_load_a_scale_vec(kb, mi, lds_buffer):
            off = (
                a_scale_lds_base
                + crd2idx((fx.Int32(acc_row0), kb), layout_scale_a_lds)
                + crd2idx((acc_row_step(mi), 0), layout_scale_a_lds)
            )
            ptr_off = fx.add_offset(lds_buffer.ptr, fx.make_int_tuple(off))
            raw = fx.make_view(ptr_off, fx.make_layout(16, 1)).load()
            return Vec(raw).bitcast(fx.Float32)

        def load_a_tile(base_k):
            parts = []
            for off in a_slot_offs:
                idx, soff = k_offset_operands(off, base_k // 4)
                parts.append(
                    _buffer_load(a_copy, gA, idx, a_load_bytes // 4, fx.Int32, soff)
                )
            return parts

        def store_a_tile_to_lds(vec_a_parts, lds_buffer):
            for i in range_constexpr(num_a_loads):
                row_a_local, col_a_local_i32 = a_chunk_coord(i, layout_a_chunks)
                chunk = {
                    "lds_ptr": lds_buffer.ptr,
                    "layout_lds": layout_lds,
                    "row_local": row_a_local,
                    "col_local_i32": col_a_local_i32,
                    "tx_c4": 4,
                    "k_blocks16": k_blocks16,
                    "lds_base": 0,
                }
                if const_expr(a_load_bytes == 16):
                    lds_store_16b_xor16(**chunk, vec_part_i32x4=vec_a_parts[i])
                else:
                    lds_store_8b_xor16(**chunk, vec_part_i32x2=vec_a_parts[i])

        def dma_a_tile_to_lds(base_k, lds_buffer):
            # The DMA writes LDS lane-linearly, so the swizzle goes on the global
            # address. M0 (an SGPR) holds the LDS base, so readfirstlane makes the
            # wave's first-thread offset uniform.
            wave_first_thread = crd2idx((fx.Int32(wave_id), 0), layout_wave_lane)
            wave_lds_dwords = crd2idx(
                (fx.Int32(wave_first_thread), 0), layout_a_dma_chunks
            )
            wave_ptr = fx.add_offset(
                lds_buffer.ptr,
                fx.rocdl.readfirstlane(
                    fx.Int32.ir_type, fx.Int32(wave_lds_dwords * DMA_BYTES_PER_LANE)
                ),
            )
            for i in range_constexpr(num_a_dma_loads):
                lds_ptr = fx.add_offset(
                    wave_ptr,
                    crd2idx((0, i), layout_a_dma_chunks) * DMA_BYTES_PER_LANE,
                )
                row_a_local, col_a_local_i32 = a_chunk_coord(i, layout_a_dma_chunks)
                col_a_local_sw = swizzle_xor16(
                    row_a_local, col_a_local_i32 * 4, k_blocks16
                )
                src_idx, soff = k_offset_operands(
                    crd2idx(
                        (
                            fx.Int32(row_a_local),
                            fx.Int32(col_a_local_sw),
                            fx.Int32(m_tile),
                            0,
                        ),
                        a_byte_tiles,
                    ),
                    base_k,
                )
                fx.copy(
                    dma_copy,
                    fx.slice(gA_dma, (None, fx.Int32(src_idx))),
                    fx.make_view(lds_ptr, fx.make_layout(1, 1)),
                    soffset=fx.Int32(soff),
                )

        def wait_a_dma():
            # The compiler may issue A's DMAs after the B loads, so drain everything.
            fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0, expcnt=0)

        def load_a_to_lds(base_k, lds_buffer):
            if const_expr(stage_a_scales):
                stage_a_scales_to_lds_async(base_k, lds_buffer)
            if const_expr(use_async_copy):
                dma_a_tile_to_lds(base_k, lds_buffer)
            else:
                store_a_tile_to_lds(load_a_tile(base_k), lds_buffer)

        def load_scales_for_tile(k_base):
            tile_scales = []
            for kb in range_constexpr(kb_per_tile):
                k_block_start = fx.Int32(k_base + kb * SCALE_BLOCK)
                scale_a_vecs = []
                if const_expr(not stage_a_scales):
                    scale_a_row0 = crd2idx(
                        (fx.Int32(acc_row0), k_block_start, fx.Int32(m_tile), 0),
                        scale_a_tiles,
                    )
                    for mi in range_constexpr(m_repeat):
                        scale_a_idx = scale_a_row0 + crd2idx(
                            (acc_row_step(mi), 0, 0, 0), scale_a_tiles
                        )
                        scale_a_vecs.append(
                            _buffer_load(scale_a_copy, gSA, scale_a_idx, 4, fx.Float32)
                        )

                scale_b_vals = []
                for ni in range_constexpr(num_acc_n):
                    scale_b_idx, soff = k_offset_operands(
                        crd2idx((fx.Int32(n_col(ni, lane=0)), 0), layout_scale_b),
                        crd2idx((0, k_block_start), layout_scale_b),
                    )
                    scale_b_vals.append(
                        _buffer_load(
                            scale_b_copy, gSB, scale_b_idx, 1, fx.Float32, soff
                        )[0]
                    )

                # Return raw loads: multiplying here would force a vmcnt(0) per scale
                # block and lose the prefetch distance.
                tile_scales.append((scale_a_vecs, scale_b_vals))
            return tile_scales

        def fp8_frag(word):
            frag = fx.make_rmem_tensor(fx.make_layout(8, 1), fp8)
            fx.memref_store_vec(Vec.from_elements([word], fx.Int64).bitcast(fp8), frag)
            return frag

        def mfma_step(acc_in, a, b):
            acc = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Float32)
            fx.memref_store_vec(Vec(acc_in), acc)
            fx.gemm(mma_atom, acc, fp8_frag(a), fp8_frag(b), acc)
            return fx.memref_load_vec(acc)

        def mfma_k64_bytes(acc_in, a0, a1, b0, b1):
            acc_mid = mfma_step(acc_in, a0, b0)
            return mfma_step(acc_mid, a1, b1)

        def scale_products(scale_a_vecs, scale_b_vals, kb, lds_buffer):
            """Staged A scales are published by the same barrier as the A tile."""
            if const_expr(stage_a_scales):
                scale_a_vecs = [
                    lds_load_a_scale_vec(kb, mi, lds_buffer)
                    for mi in range_constexpr(m_repeat)
                ]
            scale_b_vecs = [
                Vec.filled(4, scale_b_vals[ni], fx.Float32)
                for ni in range_constexpr(num_acc_n)
            ]
            products = []
            for mi in range_constexpr(m_repeat):
                products.append(
                    [
                        scale_a_vecs[mi] * scale_b_vecs[ni]
                        for ni in range_constexpr(num_acc_n)
                    ]
                )
            return products

        def compute_tile(
            running_accs, b_tile_in, lds_buffer, tile_scales, *, a0_prefetch
        ):
            running_accs = list(running_accs)

            for kb in range_constexpr(kb_per_tile):
                scales = scale_products(*tile_scales[kb], kb=kb, lds_buffer=lds_buffer)
                block_accs = [acc_init] * (num_acc_n * m_repeat)

                for ku_local in range_constexpr(ku_per_kb):
                    ku = kb * ku_per_kb + ku_local
                    b_packs0, b_packs1 = b_tile_in[ku]
                    col_base = lane_k_bytes + ku * 64

                    for mi in range_constexpr(m_repeat):
                        row_a_mi = row_a_lds + (mi * 16)
                        if const_expr(kb == 0 and ku_local == 0 and mi == 0):
                            a0, a1 = a0_prefetch
                        else:
                            a0, a1 = lds_load_packs_k64(row_a_mi, col_base, lds_buffer)

                        for ni in range_constexpr(num_acc_n):
                            acc_idx = mi * num_acc_n + ni
                            block_accs[acc_idx] = mfma_k64_bytes(
                                block_accs[acc_idx],
                                a0,
                                a1,
                                b_packs0[ni],
                                b_packs1[ni],
                            )

                for mi in range_constexpr(m_repeat):
                    for ni in range_constexpr(num_acc_n):
                        acc_idx = mi * num_acc_n + ni
                        running_accs[acc_idx] = fx.fma(
                            block_accs[acc_idx],
                            scales[mi][ni],
                            running_accs[acc_idx],
                        )

            return running_accs

        def store_output_cshuffle(final_accs):
            layout_out = fx.make_layout((tile_m, tile_n), (tile_n, 1))
            layout_c_tile = fx.select(c_tiles, [0, 1])

            c_vec_copy = fx.make_copy_atom(
                (
                    fx.rocdl.BufferCopy64b()
                    if cshuffle_vec == 4
                    else fx.rocdl.BufferCopy32b()
                ),
                out_elem_cls,
            )

            def write_row_to_lds(*, mi, ii, row_in_tile, row):
                lds_out, first_row = lds_out_row(mi * 16)
                for ni in range_constexpr(num_acc_n):
                    acc_idx = mi * num_acc_n + ni
                    val = Vec(final_accs[acc_idx])[ii]
                    lds_idx = crd2idx(
                        (
                            fx.Int32(row_in_tile - first_row),
                            fx.Int32(n_col(ni, tile=0)),
                        ),
                        layout_out,
                    )
                    fx.ptr_store(val.to(out_elem_cls), lds_out + fx.Int32(lds_idx))

            gpu.barrier()
            mfma_epilog(
                use_cshuffle=False,
                m_repeat=m_repeat,
                lane_div_16=lane_div_16,
                bx_m=0,
                body_row=write_row_to_lds,
            )
            gpu.barrier()

            tiler_mn, layout_tv = fx.make_layout_tv(
                fx.make_layout((cshuffle_mlane, cshuffle_nlane), (cshuffle_nlane, 1)),
                fx.make_layout((1, cshuffle_vec), (1, 1)),
            )
            tiles_lds = fx.zipped_divide(layout_out, tiler_mn)
            tiles_c = fx.zipped_divide(layout_c_tile, tiler_mn)
            tv_idx = fx.Int32(crd2idx((fx.Int32(tx), 0), layout_tv))
            c_tile_base = crd2idx((0, 0, fx.Int32(m_tile), fx.Int32(n_tile)), c_tiles)
            for mr in range_constexpr(cshuffle_m_reps):
                # cshuffle_mlane divides the 16-row split, so blocks stay in one buffer.
                lds_out, first_row = lds_out_row(mr * cshuffle_mlane)
                mr_lds = mr - first_row // cshuffle_mlane
                for nr in range_constexpr(cshuffle_n_reps):
                    lds_idx_pair = crd2idx((tv_idx, (mr_lds, nr)), tiles_lds)
                    frag = Vec(
                        fx.make_view(
                            lds_out + fx.Int32(lds_idx_pair),
                            fx.make_layout(cshuffle_vec, 1),
                        ).load()
                    )
                    _buffer_store(
                        c_vec_copy,
                        gC,
                        c_tile_base + crd2idx((tv_idx, (mr, nr)), tiles_c),
                        frag,
                        cshuffle_vec,
                        out_elem_cls,
                    )

        def store_output_direct(final_accs):
            if const_expr(split_k > 1):
                # Write through: the reducing split may use another XCD's L2.
                c_cls, c_dst = fx.Float32, gW
                c_copy = fx.make_copy_atom(fx.rocdl.BufferCopy32b(CPOL_COHERENT), c_cls)
            else:
                c_cls, c_dst = out_elem_cls, gC
                c_copy = fx.make_copy_atom(fx.rocdl.BufferCopy16b(), c_cls)

            idx_lane = crd2idx(
                (
                    fx.Int32(acc_row0),
                    fx.Int32(n_col(0, tile=0)),
                    fx.Int32(m_tile),
                    fx.Int32(n_tile),
                ),
                c_tiles,
            )

            def body_row(*, mi, ii, row_in_tile, row):
                for ni in range_constexpr(num_acc_n):
                    acc_idx = mi * num_acc_n + ni
                    acc = final_accs[acc_idx]
                    val = Vec(acc)[ii]
                    val_out = val if const_expr(split_k > 1) else val.to(c_cls)
                    idx_out = idx_lane + crd2idx(
                        (acc_row_step(mi, ii), n_col_step(ni), 0, 0), c_tiles
                    )
                    _buffer_store(
                        c_copy,
                        c_dst,
                        idx_out,
                        Vec.from_elements([val_out], c_cls),
                        1,
                        c_cls,
                    )

            mfma_epilog(
                use_cshuffle=False,
                m_repeat=m_repeat,
                lane_div_16=lane_div_16,
                bx_m=0,
                body_row=body_row,
            )

        def store_output(final_accs):
            if const_expr(use_cshuffle_epilog):
                store_output_cshuffle(final_accs)
            else:
                store_output_direct(final_accs)

        def hot_loop_scheduler():
            mfma_group = num_acc_n
            fx.rocdl.sched_dsrd(2)
            fx.rocdl.sched_mfma(1)
            if const_expr(tile_m == 16):
                fx.rocdl.sched_vmem(1)
            fx.rocdl.sched_mfma(1)
            if const_expr(tile_m == 16):
                fx.rocdl.sched_vmem(1)
            if const_expr(num_acc_n < 4):
                fx.rocdl.sched_dsrd(1)
                fx.rocdl.sched_mfma(1)
                if const_expr(tile_m == 16):
                    fx.rocdl.sched_vmem(1)
                fx.rocdl.sched_dsrd(1)
                fx.rocdl.sched_mfma(1)
                if const_expr(tile_m == 16):
                    fx.rocdl.sched_vmem(1)
                fx.rocdl.sched_mfma(1)

            for sche_i in range_constexpr(sche_iters):
                if const_expr(vmem_per_iter[sche_i] > 0):
                    fx.rocdl.sched_vmem(vmem_per_iter[sche_i])
                fx.rocdl.sched_mfma(mfma_group)
                fx.rocdl.sched_dsrd(1)
                fx.rocdl.sched_mfma(mfma_group)
                # One slot more than A stores, opened an iteration early, so the next
                # tile's LDS stores start among the last MFMAs, not behind them.
                if const_expr(sche_i >= dswr_start - 1):
                    fx.rocdl.sched_dswr(1)
            # Closes this step's groups so they do not reach into the next step.
            fx.rocdl.sched_barrier(0)

        def k_step(next_k, cur_buf, next_buf, running_accs, cur_b, cur_scales, cur_a0):
            """Compute cur_buf while prefetching next_buf."""
            # Program-order VMEM slots prioritize A when scales cost occupancy;
            # A must reach LDS before the barrier; load B last to shorten its lifetime.
            if const_expr(spread_loads and scales_first):
                next_scales = load_scales_for_tile(next_k)
            load_a_to_lds(next_k, next_buf)
            if const_expr(spread_loads):
                if const_expr(not scales_first):
                    next_scales = load_scales_for_tile(next_k)
                next_b = load_b_tile(next_k)
            else:
                next_b = load_b_tile(next_k)
                next_scales = load_scales_for_tile(next_k)

            running_accs = compute_tile(
                running_accs, cur_b, cur_buf, cur_scales, a0_prefetch=cur_a0
            )

            hot_loop_scheduler()
            if const_expr(use_async_copy):
                wait_a_dma()
            gpu.barrier()
            return running_accs, next_b, next_scales, prefetch_a0_pack(next_buf)

        # Include the prologue in the first step's scheduling region, excluding setup.
        fx.rocdl.sched_barrier(0)

        lds_bufs = (lds.ping, lds.pong)
        cur_b = load_b_tile(k_start)
        cur_scales = load_scales_for_tile(k_start)
        load_a_to_lds(k_start, lds_bufs[0])
        if const_expr(use_async_copy):
            wait_a_dma()
        elif const_expr(spread_loads):
            # Drain every prologue load, not only A's: a load still in flight would make
            # every loop iteration wait for all (the loop top has a back edge).
            fx.rocdl.s_waitcnt(vmcnt=0)
        gpu.barrier()
        running_accs = [acc_init] * (num_acc_n * m_repeat)
        cur_a0 = prefetch_a0_pack(lds_bufs[0])

        # Pair steps for static ping/pong identities.
        num_steps = num_tiles - 1
        if const_expr(num_steps >= 2):
            for pair in range(0, num_steps // 2, 1):
                k_pair = k_start + fx.Index(pair) * (2 * tile_k)
                running_accs, cur_b, cur_scales, cur_a0 = k_step(
                    k_pair + tile_k,
                    lds.ping,
                    lds.pong,
                    running_accs,
                    cur_b,
                    cur_scales,
                    cur_a0,
                )
                running_accs, cur_b, cur_scales, cur_a0 = k_step(
                    k_pair + 2 * tile_k,
                    lds.pong,
                    lds.ping,
                    running_accs,
                    cur_b,
                    cur_scales,
                    cur_a0,
                )
        if const_expr(num_steps % 2 == 1):
            running_accs, cur_b, cur_scales, cur_a0 = k_step(
                k_start + num_steps * tile_k,
                lds.ping,
                lds.pong,
                running_accs,
                cur_b,
                cur_scales,
                cur_a0,
            )

        final_accs = compute_tile(
            running_accs,
            cur_b,
            lds_bufs[num_steps % 2],
            cur_scales,
            a0_prefetch=cur_a0,
        )

        store_output(final_accs)

        if const_expr(split_k > 1):
            # Release the write-through partials before counting this split's arrival.
            fx.rocdl.s_waitcnt(vmcnt=0)
            gpu.barrier()
            splitk_reduce_epilogue(
                rows_view(arg_workspace),
                rows_view(arg_c),
                arg_semaphore,
                lds.split_flag.ptr,
                tile_m,
                tile_n,
                total_threads,
                out_elem_cls,
                fx.thread_idx.x,
                # Int32: Index ids crash the tracer when indexing the reduce's view.
                fx.Int32(m_tile),
                fx.Int32(n_tile),
                i32_m,
                N,
                split_k,
            )

    if split_k == 1:

        @flyc.kernel(name=kernel_name)
        def kernel_gemm(
            arg_c: fx.Tensor,
            arg_a: fx.Tensor,
            arg_b: fx.Tensor,
            arg_scale_a: fx.Tensor,
            arg_scale_b: fx.Tensor,
            i32_m: fx.Int32,
        ):
            gemm_body(arg_c, arg_a, arg_b, arg_scale_a, arg_scale_b, i32_m)

        @flyc.jit
        def launch_gemm(
            arg_c: fx.Tensor,
            arg_a: fx.Tensor,
            arg_b: fx.Tensor,
            arg_scale_a: fx.Tensor,
            arg_scale_b: fx.Tensor,
            i32_m: fx.Int32,
            stream: fx.Stream,
        ):
            kernel_gemm(
                arg_c,
                arg_a,
                arg_b,
                arg_scale_a,
                arg_scale_b,
                i32_m,
            ).launch(
                grid=(num_workgroups(i32_m), 1, 1),
                block=(total_threads, 1, 1),
                stream=stream,
            )

        return launch_gemm

    else:

        @flyc.kernel(name=kernel_name)
        def kernel_gemm_split_k(
            arg_c: fx.Tensor,
            arg_a: fx.Tensor,
            arg_b: fx.Tensor,
            arg_scale_a: fx.Tensor,
            arg_scale_b: fx.Tensor,
            i32_m: fx.Int32,
            arg_workspace: fx.Tensor,
            arg_semaphore: fx.Tensor,
        ):
            gemm_body(
                arg_c,
                arg_a,
                arg_b,
                arg_scale_a,
                arg_scale_b,
                i32_m,
                split_args=(arg_workspace, arg_semaphore),
            )

        # The split is the second grid axis, so the first keeps the swizzle's tile ids.
        @flyc.jit
        def launch_gemm_split_k(
            arg_c: fx.Tensor,
            arg_a: fx.Tensor,
            arg_b: fx.Tensor,
            arg_scale_a: fx.Tensor,
            arg_scale_b: fx.Tensor,
            i32_m: fx.Int32,
            arg_workspace: fx.Tensor,
            arg_semaphore: fx.Tensor,
            stream: fx.Stream,
        ):
            kernel_gemm_split_k(
                arg_c,
                arg_a,
                arg_b,
                arg_scale_a,
                arg_scale_b,
                i32_m,
                arg_workspace,
                arg_semaphore,
            ).launch(
                grid=(num_workgroups(i32_m), split_k, 1),
                block=(total_threads, 1, 1),
                stream=stream,
            )

        return launch_gemm_split_k


__all__ = ["compile_blockscale_preshuffle_gemm", "plan_lds"]
