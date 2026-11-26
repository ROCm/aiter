# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""A8W8 (FP8 E4M3 activation x FP8 E4M3 weight) 256x256 / 128x256 MXscale GEMM for gfx1250."""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm as llvm_dialect
from flydsl.expr import const_expr, range_constexpr, rocdl
from flydsl.expr.rocdl import cluster
from flydsl.expr.typing import Constexpr, T, as_ir_value
from flydsl.expr.typing import Vector as Vec
from flydsl.runtime.device import get_rocm_arch

from aiter.jit.utils.lds_capacity import get_lds_capacity_bytes

from . import tdm_ops_gfx1250 as tdm_ops
from .gemm_common_gfx1250 import (
    make_lds_copy_ops,
    pipeline_fence,
    pipeline_fence_signal,
    pipeline_fence_wait,
    prefetch_kernel_code,
    workgroup_barrier,
)
from .gfx1250_cluster import compute_mcast_masks_runtime
from .kernels_common import format_kernel_name
from .splitk_fused_epilogue_gfx1250 import emit_fused_splitk_epilogue
from .tensor_shim import (
    AITER_FLYDSL_KERNARG_PRELOAD,
    AITER_FLYDSL_KERNARG_PRELOAD_COUNT,
)


@flyc.jit
def launch_gemm_a8w8_256x256(
    arg_c: fx.Pointer,
    arg_a: fx.Pointer,
    arg_b: fx.Pointer,
    arg_scale_a: fx.Pointer,
    arg_scale_b: fx.Pointer,
    i32_m: fx.Int32,
    stream: fx.Stream,
    N: fx.Int32,
    K: fx.Int32,
    stride_ascale_k: fx.Int32,
    i32_lda: fx.Int32,
    i32_ldc: fx.Int32,
    arg_out: fx.Pointer,
    i32_cm: fx.Int32,
    tile_m: Constexpr[int],
    tile_n: Constexpr[int],
    tile_k: Constexpr[int],
    m_warp: Constexpr[int],
    n_warp: Constexpr[int],
    out_is_f16: Constexpr[int],
    num_buffers: Constexpr[int],
    use_cluster: Constexpr[bool],
    cluster_n: Constexpr[int],
    is_mxscale: Constexpr[bool],
    block_size: Constexpr[int],
    split_k: Constexpr[int] = 1,
    a_preshuffle: Constexpr[bool] = False,
    persistent_n_tiles: Constexpr[int] = 1,
    fused_splitk: Constexpr[bool] = False,
    a_scale_preshuffled: Constexpr[bool] = False,
    k_tiles_class: Constexpr[int] = 2,
    b_scale_32x32: Constexpr[bool] = False,
    batched: Constexpr[bool] = False,
    batch: fx.Int32 = 1,
    fuse_quant_out: Constexpr[bool] = False,
    bounded_n: Constexpr[bool] = False,
    a_scale_unaligned: Constexpr[bool] = False,
):
    """A launch clusters ``i32_cm`` rows, the cluster_m that fits M, so no ``.cluster_dims`` is
    compiled in; ``use_cluster`` is False only for a launch of one workgroup.

    N must be a multiple of ``tile_n * cluster_n * persistent_n_tiles`` unless
    ``bounded_n`` (split_k=1): N then ends inside the last such group, whose spare
    tiles skip their loads and stores. M is unrestricted (a
    multiple of 2 when ``a_preshuffle``); K must be divisible by 128 and at
    least 512 per split.
    ``a_scale_unaligned`` (block128): the A scale rows do not all start on a 16-byte
    boundary; each K step then loads them from the 16-byte floor of its address.
    ``k_tiles_class`` buckets the K-tiles per split:
    0 for <= 16, 1 for <= 64, 2 above; long K loops keep the early merge
    timeout on A/B and the plain K-loop schedule, which run faster there.
    ``b_scale_32x32``: the mxfp8_32 B scale is row-major ``[N/32, K/32]``.
    ``batched``: A ``[M, batch, K]``, A scale ``[M, batch, K/32]``, B and its
    scale ``[batch, ...]``, C ``[M, batch, N]``.
    ``fuse_quant_out``: the C tile is group-32 MXFP8-quantized in the
    epilogue. ``arg_c`` is then the e4m3 C (same layout and ``ldc``) and
    ``arg_out`` the row-major ``[M, ldc // 32]`` e8m0 scale, bit-matching ATOM
    ``quantize_fp8`` of the bf16 output."""

    assert (tile_m, tile_n, tile_k, m_warp, n_warp, num_buffers) in (
        (256, 256, 128, 2, 2, 4),
        (256, 256, 128, 2, 2, 2),
        (128, 128, 128, 2, 2, 4),
        (128, 256, 128, 2, 2, 4),
        (128, 256, 128, 2, 2, 3),
    ), "only the tuned 2x2-wave profiles are supported"
    assert is_mxscale, "only the MX-scaled profiles are supported"
    mx32 = block_size == 32
    assert mx32 or block_size == 128, f"block_size must be 32 or 128, got {block_size}"
    assert 1 <= cluster_n < 16, f"cluster_n must be 1..15, got {cluster_n}"
    assert split_k in (1, 2, 4, 8), f"split_k must be 1/2/4/8, got {split_k}"
    assert (
        persistent_n_tiles >= 1
    ), f"persistent_n_tiles must be >= 1, got {persistent_n_tiles}"
    assert (
        persistent_n_tiles == 1 or split_k == 1
    ), "persistent_n_tiles>1 requires split_k=1"
    assert not fused_splitk or (
        split_k > 1 and tile_m % split_k == 0
    ), "a fused split-K epilogue needs split_k > 1 dividing tile_m"
    cluster_k = split_k if fused_splitk else 1
    assert (
        use_cluster or cluster_n * cluster_k == 1
    ), "a plain launch has no cluster_n or K splits"
    assert not bounded_n or (
        split_k == 1 and not fuse_quant_out
    ), "bounded_n needs split_k=1 and no fused quant"
    assert not a_scale_preshuffled or mx32, "a_scale_preshuffled is mxfp8_32 only"
    assert not b_scale_32x32 or mx32, "b_scale_32x32 is mxfp8_32 only"
    assert not (a_scale_unaligned and mx32), "a_scale_unaligned is block128 only"
    assert not batched or (
        mx32 and split_k == 1 and not a_preshuffle and not a_scale_preshuffled
    ), "batched needs mxfp8_32, split_k=1 and row-major A and A scale"
    assert not fuse_quant_out or (
        mx32
        and split_k == 1
        and not a_preshuffle
        and not a_scale_preshuffled
        and persistent_n_tiles == 1
        and not fused_splitk
    ), (
        "fuse_quant_out needs mxfp8_32, split_k=1, row-major A and A scale and "
        "no persistent tiles"
    )

    WAVE = 32
    WMMA_M = WMMA_N = 16
    WMMA_K = 128
    num_waves = m_warp * n_warp
    block = num_waves * WAVE
    warp_tile_m = tile_m // m_warp
    warp_tile_n = tile_n // n_warp
    wmma_m_rep = warp_tile_m // WMMA_M
    wmma_n_rep = warp_tile_n // WMMA_N
    half_m, half_n = wmma_m_rep // 2, wmma_n_rep // 2
    n_acc = wmma_m_rep * wmma_n_rep

    # LDS: num_buffers slots per operand, KPAIR K-tiles each; the K loop is unrolled over them.
    KPAIR = 1 if num_buffers == 4 and tile_n == 256 else 2
    UNROLL = KPAIR * num_buffers
    SUPER_K = tile_k * KPAIR
    # A-preshuffle: A is shuffle_mxfp8fp4_a-tiled, [M, K] -> [M/2, K/128, 2, 128]
    A_PAIR = 2 if a_preshuffle else 1
    LDS_PAD_A = 16
    A_SHAPE = (tile_m // A_PAIR, A_PAIR * SUPER_K)
    A_LDS_ROW = A_SHAPE[1] + LDS_PAD_A
    B_SHAPE = (tile_n // 16, 16 * SUPER_K)
    B_LDS_ROW = B_SHAPE[1]
    # mx32 row-major A scale: one slot stages the scales of SA_PANEL_STAGES K slots.
    SA_PANEL_STAGES = (
        (8 if num_buffers == 2 else 4 if num_buffers == 3 else 1)
        if mx32 and not a_scale_preshuffled
        else 1
    )
    SA_PANEL_WIDTH = SA_PANEL_STAGES * (SUPER_K // 32)
    SA_PANEL_PAD = 4 if SA_PANEL_STAGES > 1 else 0
    if const_expr(mx32):
        SA_SHAPE = (
            (tile_m // 32, SUPER_K) if a_scale_preshuffled else (tile_m, SA_PANEL_WIDTH)
        )
        SA_LDS_ROW = SA_SHAPE[1] + SA_PANEL_PAD
        SB_SHAPE = (tile_n // 32, SUPER_K // 32 if b_scale_32x32 else SUPER_K)
        SB_LDS_ROW = SB_SHAPE[1]
    else:
        # block128: one A-scale row per K block, one B-scale row per 128 columns;
        # the TDM lands a slot's rows contiguously. An unaligned A-scale row's
        # bytes start up to 15 (+ 15 per earlier row) bytes into it.
        N_BLOCKS = tile_n // 128
        SA_LDS_ROW = ((tile_m + 15) // 16) * 16 + (
            16 * KPAIR if a_scale_unaligned else 0
        )
        SA_SHAPE = (KPAIR, SA_LDS_ROW if a_scale_unaligned else tile_m)
        SB_SHAPE = (N_BLOCKS, KPAIR)
        SB_LDS_ROW = ((N_BLOCKS + 15) // 16) * 16 // N_BLOCKS
    STAGE_A = A_SHAPE[0] * A_LDS_ROW
    STAGE_B = B_SHAPE[0] * B_LDS_ROW
    STAGE_SA = SA_SHAPE[0] * SA_LDS_ROW
    STAGE_SB = SB_SHAPE[0] * SB_LDS_ROW
    # B first so it keeps a 64-KiB-aligned base; the rest packs behind it.
    PLANAR_B_BASE = 0
    PLANAR_A_BASE = PLANAR_B_BASE + num_buffers * STAGE_B
    PLANAR_SA_BASE = PLANAR_A_BASE + num_buffers * STAGE_A
    PLANAR_SB_BASE = PLANAR_SA_BASE + num_buffers * STAGE_SA
    PLANAR_END = PLANAR_SB_BASE + num_buffers * STAGE_SB

    out_elem = fx.Float16 if out_is_f16 else fx.BFloat16
    partial_elem = fx.Float32 if split_k > 1 else out_elem
    partial_bytes = 4 if split_k > 1 else 2
    C_LDS_ROW = tile_n + 8
    ARENA_B = max(PLANAR_END, tile_m * C_LDS_ROW * partial_bytes)
    # Fused quant: e4m3 C tile (rows padded by 16 B) plus one e8m0 byte per
    # 32 columns, both staged in LDS. A 32-column group is two adjacent 16x16
    # tiles of one wave (wmma_n_rep is even).
    C8_ROW = tile_n + 16
    if const_expr(fuse_quant_out):
        ARENA_B = max(ARENA_B, tile_m * C8_ROW + tile_m * (tile_n // 32))
    # The compile target, not the host: AOT cross-compiles these gfx1250
    # kernels under FLYDSL_GPU_ARCH, which get_rocm_arch() honours.
    arch = get_rocm_arch().split(":", 1)[0]
    lds_cap = get_lds_capacity_bytes(arch)

    # Persistent tiles stage C per wave, clear of LDS slots 0..C_EARLY_SLOTS-1, so
    # the next tile can load those slots while this tile's C store still reads LDS.
    C_EARLY_SLOTS = num_buffers // 2
    Q_LDS_ROW = warp_tile_n + 8
    Q_BYTES = warp_tile_m * Q_LDS_ROW * 2
    Q_FREE_OFF = (PLANAR_END + 255) // 256 * 256
    # Each wave stages rows [0, Q_SPLIT) of its quadrant at Q_OFFS[w] and the rest
    # at Q_TAIL_OFFS[w]: contiguous except for the last wave, whose head fits B slot 3
    # alone so that B slot 2 stays free for the next tile's K-tile 2.
    Q_SPLIT = warp_tile_m - 16
    Q_HEAD_BYTES = Q_SPLIT * Q_LDS_ROW * 2
    if const_expr(num_buffers == 4):
        Q_OFFS = [
            Q_FREE_OFF,
            PLANAR_A_BASE + 3 * STAGE_A,
            PLANAR_A_BASE + 2 * STAGE_A,
            PLANAR_B_BASE + 3 * STAGE_B,
        ]
        Q_TAIL_OFFS = [off + Q_HEAD_BYTES for off in Q_OFFS[:3]] + [
            Q_FREE_OFF + Q_BYTES
        ]
        q_fits = STAGE_A >= Q_BYTES and STAGE_B >= Q_HEAD_BYTES
    else:
        Q_OFFS = [
            Q_FREE_OFF,
            PLANAR_A_BASE + STAGE_A,
            PLANAR_A_BASE + STAGE_A + Q_BYTES,
            PLANAR_B_BASE + STAGE_B,
        ]
        Q_TAIL_OFFS = [off + Q_HEAD_BYTES for off in Q_OFFS]
        q_fits = STAGE_A >= 2 * Q_BYTES and STAGE_B >= Q_BYTES
    Q_END = max(Q_FREE_OFF + Q_BYTES, Q_TAIL_OFFS[3] + Q_BYTES - Q_HEAD_BYTES)
    # An mx32 A-scale panel feeds several K-steps, so its slot is still being read
    # when the drain would prefetch the next tile into it.
    c_relocate = (
        persistent_n_tiles > 1
        and num_buffers in (2, 4)
        and SA_PANEL_STAGES == 1
        and Q_END <= lds_cap
        and q_fits
    )
    if const_expr(c_relocate):
        ARENA_B = max(ARENA_B, Q_END)
    if ARENA_B > lds_cap:
        raise RuntimeError(
            f"Shared Memory Overflow: requested {ARENA_B} bytes on {arch}, "
            f"limit is {lds_cap} bytes"
        )
    # Drain K-steps whose freed slot takes the next tile's matching K-step, and how
    # many of those loads are already in flight when each drain K-step fences.
    DRAIN_PREF = [
        c_relocate and g % KPAIR == KPAIR - 1 and g // KPAIR < C_EARLY_SLOTS
        for g in range(UNROLL)
    ]
    DRAIN_PREF_BEFORE = [sum(DRAIN_PREF[:g]) for g in range(UNROLL)]
    # With four slots and one K-tile per slot, the third drain K-step also lands
    # the next tile's K-tile 2 for every operand but A (whose slot 2 holds C), and
    # that tile waits for the C store only at its first refill.
    BB_OK = c_relocate and num_buffers == 4 and KPAIR == 1
    BB_SLOT = 2
    kernel_name = format_kernel_name(
        f"gemm_a8w8_mx{block_size}_compute_t{tile_m}x{tile_n}x{tile_k}"
        f"_mw{m_warp}_nw{n_warp}_nb{num_buffers}_sk{split_k}"
        f"_cn{cluster_n}"
        + ("" if use_cluster else "_plain")
        + ("_apre" if a_preshuffle else "")
        + (f"_ps{persistent_n_tiles}" if persistent_n_tiles > 1 else "")
        + ("_fsk" if fused_splitk else "")
        + ("_aspre" if a_scale_preshuffled else "")
        + ("_bs32x32" if b_scale_32x32 else "")
        + ("_mbn" if batched else "")
        + ("_fq" if fuse_quant_out else "")
    )

    def _run_tile(
        arg_c: fx.Pointer,
        arg_a: fx.Pointer,
        arg_b: fx.Pointer,
        arg_scale_a: fx.Pointer,
        arg_scale_b: fx.Pointer,
        i32_m: fx.Int32,
        i32_n: fx.Int32,
        i32_k: fx.Int32,
        i32_stride_ascale_k: fx.Int32,
        i32_lda: fx.Int32,
        i32_ldc: fx.Int32,
        arg_out: fx.Pointer,
        i32_cm: fx.Int32,
        tile_idx=0,
    ):
        K_TILES = i32_k // (tile_k * split_k)
        k64 = fx.Int64(i32_k)
        lda64 = fx.Int64(i32_lda)
        ldc64 = fx.Int64(i32_ldc)
        b_row64 = k64 * 16  # global bytes of a 16-column B row

        tid = fx.thread_idx.x
        bid_x, bid_y, bid_z = fx.block_idx
        if const_expr(persistent_n_tiles > 1):
            bid_y = bid_y + tile_idx * fx.grid_dim.y
        if const_expr(split_k > 1):
            m_chunks = fx.grid_dim.z // split_k
            split_idx = bid_z // m_chunks
            m_chunk = bid_z - split_idx * m_chunks
            kt_base = fx.Int64(split_idx) * fx.Int64(K_TILES)
        else:
            m_chunk = fx.Int32(0) if const_expr(batched) else bid_z
            kt_base = fx.Int64(0)
        wave = fx.Int32(rocdl.readfirstlane(T.i32, tid // WAVE))
        lane = tid % WAVE
        lane16 = lane % 16
        kgrp = lane // 16
        wave_m = wave // n_warp
        wave_n = wave % n_warp
        if const_expr(use_cluster):
            local_x, local_y = cluster.compute_cluster_position()
            a_mask, b_mask = compute_mcast_masks_runtime(
                local_x, local_y, i32_cm, cluster_n
            )
        else:
            a_mask, b_mask = 0, 0
        cluster_plane = fx.Int32(0)
        if const_expr(fused_splitk):
            # Multicast inputs only within the matching K split. The hardware
            # numbers workgroups X first, then Y, then Z inside the cluster.
            cluster_plane = fx.Int32(rocdl.cluster_workgroup_id_z()) * (
                i32_cm * cluster_n
            )
            a_mask = fx.Int32(a_mask) << cluster_plane
            b_mask = fx.Int32(b_mask) << cluster_plane
        blk_m = (m_chunk * fx.grid_dim.x + bid_x) * tile_m
        blk_n = bid_y * tile_n
        blk_m64 = fx.Int64(blk_m)
        blk_n64 = fx.Int64(blk_n)
        m_rows = i32_m - blk_m  # valid M rows of the tile (A / C), <= 0 past M

        def _clamp(x, hi):
            """``x`` limited to [0, hi]."""
            return fx.max(fx.min(x, hi), 0)

        if const_expr(bounded_n):
            # Valid N columns of this tile; a spare tile of the last group owns none.
            n_left = i32_n - blk_n
            n_cols = _clamp(n_left, tile_n)
            b_rows = (n_cols + 15) // 16
            sb_blk = 32 if const_expr(mx32) else 128
            sb_rows = (n_cols + (sb_blk - 1)) // sb_blk
            # A load bounded to zero rows still fetches one: a spare tile reads
            # the last real tile's B and B scale, never past the weights.
            last_n = ((i32_n - 1) // tile_n) * tile_n
            ld_n64 = fx.Int64((n_left > 0).select(blk_n, last_n))
        else:
            n_cols = b_rows = sb_rows = None
            ld_n64 = blk_n64
        a_rows = (m_rows + 1) >> 1 if const_expr(a_preshuffle) else m_rows
        if const_expr(mx32):
            sa_rows = (
                (i32_m + 31) // 32 - blk_m // 32
                if const_expr(a_scale_preshuffled)
                else m_rows
            )

        arena = fx.SharedAllocator(static=False)
        arena.allocate(ARENA_B)
        base_ptr = arena.base_ptr

        def _planar_base(offset, stride, stage):
            ptr = fx.add_offset(base_ptr, offset + stage * stride)
            return fx.index_cast(T.index, fx.ptrtoint(ptr))

        def _view(ptr, shape, stride):
            return fx.Tensor(fx.make_view(ptr, fx.make_layout(shape, stride)))

        def _view_at(base, off, shape, stride):
            return _view(fx.add_offset(base, off), shape, stride)

        if const_expr(fused_splitk):
            flat_tile = bid_y * fx.grid_dim.x * fx.grid_dim.z // split_k + (
                m_chunk * fx.grid_dim.x + bid_x
            )

        gA_base = fx.recast_iter(fx.Int8, arg_a)
        gB_base = fx.recast_iter(fx.Int8, arg_b)

        k_elem0 = kt_base * tile_k
        a_off0 = blk_m64 * lda64 + k_elem0 * A_PAIR
        b_off0 = (ld_n64 // 16) * b_row64 + k_elem0 * 16
        if const_expr(mx32):
            sa_row64 = (
                fx.Int64(i32_stride_ascale_k) if const_expr(batched) else k64 // 32
            )
            sa_off0 = (
                (blk_m64 // 32) * k64 + k_elem0
                if const_expr(a_scale_preshuffled)
                else blk_m64 * sa_row64 + k_elem0 // 32
            )
            sb_k64 = k64 // 32 if const_expr(b_scale_32x32) else k64
            sb_off0 = (ld_n64 // 32) * sb_k64 + (
                k_elem0 // 32 if const_expr(b_scale_32x32) else k_elem0
            )
        else:
            sa_off0 = blk_m64 + kt_base * fx.Int64(i32_stride_ascale_k)
            sb_off0 = (ld_n64 // 128) * (k64 // 128) + kt_base
        if const_expr(batched):
            bat = fx.Int64(bid_z)
            n64 = fx.Int64(i32_n)
            a_off0 = a_off0 + bat * k64
            b_off0 = b_off0 + bat * n64 * k64
            sa_off0 = sa_off0 + bat * (k64 // 32)
            sb_off0 = sb_off0 + bat * (n64 // 32) * sb_k64
        b_tile_shift = sb_tile_shift = fx.Int32(0)
        if const_expr(c_relocate):
            # The next persistent tile differs only in its N block. Byte deltas are
            # added unsigned, so the drain prefetches that tile only while B's, the
            # largest, stays below 4 GiB; the shifts below are exact modulo 2**32.
            n_adv = fx.grid_dim.y * tile_n
            b_tile_shift = n_adv * i32_k
            if const_expr(mx32):
                sb_tile_shift = (n_adv // 32) * (
                    i32_k // 32 if const_expr(b_scale_32x32) else i32_k
                )
            else:
                sb_tile_shift = (n_adv // 128) * (i32_k // 128)
            pref_on = fx.Int64(n_adv) * k64 + (
                BB_SLOT if BB_OK else C_EARLY_SLOTS - 1
            ) * B_LDS_ROW < fx.Int64(2**32)

        gA = _view_at(
            gA_base, a_off0, (A_SHAPE[0], A_PAIR * tile_k), (A_PAIR * tile_k, 1)
        )
        gB = _view_at(gB_base, b_off0, (B_SHAPE[0], 16 * tile_k), (16 * tile_k, 1))
        if const_expr(mx32):
            sa_gstride = (
                i32_k
                if const_expr(a_scale_preshuffled)
                else (i32_stride_ascale_k if const_expr(batched) else i32_k // 32)
            )
            sb_gstride = i32_k // 32 if const_expr(b_scale_32x32) else i32_k
        else:
            sa_gstride, sb_gstride = i32_stride_ascale_k, i32_k // 128
            sa_rows = (m_rows > 0).select(fx.Int32(KPAIR), fx.Int32(0))
            sa_step = i32_stride_ascale_k * KPAIR
            sa_cols = m_rows
            if const_expr(a_scale_unaligned):
                # A row off a 4-byte (less so a 16-byte) boundary stalls its load:
                # floor the base and the row stride to 16 bytes, read LDS at the rest.
                sa_drift = (i32_stride_ascale_k >= 16).select(
                    i32_stride_ascale_k & 15, fx.Int32(0)
                )
                sa_gstride = i32_stride_ascale_k - sa_drift
                sa_rem0 = (
                    fx.Int32(fx.ptrtoint(fx.add_offset(arg_scale_a, sa_off0))) & 15
                )
                sa_off0 = sa_off0 - fx.Int64(sa_rem0)
                sa_cols = fx.min(m_rows, tile_m)
                sa_cols = sa_cols + (KPAIR - 1) * sa_drift
        gSA = _view_at(arg_scale_a, sa_off0, SA_SHAPE, (SA_SHAPE[1], 1))
        gSB = _view_at(arg_scale_b, sb_off0, SB_SHAPE, (SB_SHAPE[1], 1))

        def _owner_tdm(owner):
            """Load descriptor of the wave owning A (0), B (1), A scale (2) or B scale (3):
            (dgroup0, dgroup1, LDS bytes per slot, global step per slot, next-tile shift).
            """
            lds_base, shape, lds_row, pad = (
                (PLANAR_A_BASE, A_SHAPE, A_LDS_ROW, LDS_PAD_A),
                (PLANAR_B_BASE, B_SHAPE, B_LDS_ROW, 0),
                (PLANAR_SA_BASE, SA_SHAPE, SA_LDS_ROW, SA_PANEL_PAD),
                (PLANAR_SB_BASE, SB_SHAPE, SB_LDS_ROW, 0),
            )[owner]
            if const_expr(owner == 0):
                tensor, gstride, mask, rows = gA, i32_lda * A_PAIR, a_mask, a_rows
            elif const_expr(owner == 1):
                tensor, gstride, mask, rows = gB, i32_k * 16, b_mask, b_rows
            elif const_expr(owner == 2):
                tensor, gstride, mask, rows = gSA, sa_gstride, a_mask, sa_rows
            else:
                tensor, gstride, mask, rows = gSB, sb_gstride, b_mask, sb_rows
                if const_expr(not mx32 and use_cluster):
                    flat_wg = (
                        fx.Int32(local_x) + fx.Int32(local_y) * i32_cm + cluster_plane
                    )
                    mask = 1 << fx.Int32(rocdl.readfirstlane(T.i32, flat_wg))
            # block128 A scale: a K block is a global row, bounded to the tile's M.
            sa128 = const_expr(owner == 2 and not mx32)
            desc = tdm_ops.make_tensor_descriptor_2d(
                global_ptr=tensor,
                lds_memref=_view(
                    fx.add_offset(base_ptr, lds_base), shape, (lds_row, 1)
                ),
                global_offset=(0, 0),
                tensor_shape=shape,
                strides=(gstride, 1),
                tile_shape=shape,
                elem_bytes=1,
                pad_interval=shape[1] if pad else 0,
                pad_amount=pad,
                num_warps=1,
                workgroup_mask=mask,
                early_timeout=owner > 1 or k_tiles_class > 1,
                oob_outer_bound=rows,
                oob_inner_bound=sa_cols if sa128 else None,
            )
            tile_shift = (
                b_tile_shift
                if const_expr(owner == 1)
                else (sb_tile_shift if const_expr(owner == 3) else fx.Int32(0))
            )
            return (
                Vec(desc.dgroup0),
                Vec(desc.dgroup1),
                fx.Int32(shape[0] * lds_row),
                fx.Int32(sa_step if sa128 else shape[1]),
                tile_shift,
            )

        own_tdm = (
            fx.full(4, 0, fx.Int32),
            fx.full(8, 0, fx.Int32),
            fx.Int32(0),
            fx.Int32(0),
            fx.Int32(0),
        )
        if wave == 0:
            own_tdm = _owner_tdm(0)
        elif wave == 1:
            own_tdm = _owner_tdm(1)
        elif wave == 2:
            own_tdm = _owner_tdm(2)
        else:
            own_tdm = _owner_tdm(3)
        dgroup0, dgroup1, tdm_lds_step, tdm_global_step, tdm_tile_shift = own_tdm

        tdm_desc = tdm_ops.TDMDescriptor2D(as_ir_value(dgroup0), as_ir_value(dgroup1))
        if const_expr(a_scale_unaligned):
            # Only the A-scale owner shifts its load; the other waves add zero.
            sa_own_mask = (wave == 2).select(fx.Int32(15), fx.Int32(0))
            sa_own_rem0 = sa_rem0 & sa_own_mask
        tdm_base_lds, tdm_base_lo, tdm_base_hi = dgroup0[1], dgroup0[2], dgroup0[3]

        def _prepare_tdm(slot, tile_delta, sa_panel=0, sa_refill=1):
            """This wave's descriptor for LDS ``slot`` at ``tile_delta`` global bytes past
            the tile's first K-tile. An mx32 A-scale panel has its own slot and K."""
            if const_expr(SA_PANEL_STAGES > 1):
                slot = (wave == 2).select(sa_panel % num_buffers, fx.Int32(slot))
                tile_delta = (wave == 2).select(sa_panel * SA_PANEL_WIDTH, tile_delta)
            if const_expr(a_scale_unaligned):
                sa_rem = (sa_own_rem0 + tile_delta) & sa_own_mask
                tile_delta = sa_own_rem0 + tile_delta - sa_rem
            desc = tdm_ops.update_tensor_descriptor_2d_lds_addr(
                tdm_desc, tdm_base_lds + tdm_lds_step * fx.Int32(slot)
            )
            desc = tdm_ops.update_tensor_descriptor_2d_addr64(
                desc, tdm_base_lo, tdm_base_hi, tile_delta
            )
            if const_expr(a_scale_unaligned):
                desc = tdm_ops.grow_tensor_descriptor_2d_inner_extent(desc, sa_rem)
            if const_expr(SA_PANEL_STAGES > 1):
                desc = tdm_ops.clamp_tensor_descriptor_2d_inner_extent(
                    desc,
                    i32_k // 32 - fx.Int32(k_elem0 // 32) - sa_panel * SA_PANEL_WIDTH,
                    wave == 2,
                )
                desc = tdm_ops.update_tensor_descriptor_2d_pred(
                    desc,
                    (wave == 2).select(fx.Int32(sa_refill), fx.Int32(1)),
                )
            return desc

        wmb = wave_m * warp_tile_m
        wnb = wave_n * warp_tile_n

        # [B-scale opsel][A-scale opsel]: the scale byte of the 16-row / 16-column tile.
        wmma_atoms = [
            [
                fx.make_mma_atom(
                    fx.rocdl.WMMAScale(
                        WMMA_M,
                        WMMA_N,
                        WMMA_K,
                        fx.Float8E4M3FN,
                        fx.Float8E4M3FN,
                        fx.Float32,
                        opsel_a=opsel_sb,
                        opsel_b=opsel_sa,
                    )
                )
                for opsel_sa in range_constexpr(2)
            ]
            for opsel_sb in range_constexpr(2)
        ]
        # Allocated quadrant by quadrant: the order fixes the accumulators' registers.
        c_frags = [None] * n_acc
        for qm in range_constexpr(2):
            for qn in range_constexpr(2):
                for im in range_constexpr(half_m):
                    for jn in range_constexpr(half_n):
                        c_frags[(qm * half_m + im) * wmma_n_rep + qn * half_n + jn] = (
                            fx.make_rmem_tensor(8, fx.Float32)
                        )
        for cf in c_frags:
            cf.store(fx.constant_vector(0.0, T.vec(8, T.f32)))

        def _frag_reg(v):
            """A register tensor holding the 16-dword A or B fragment ``v``."""
            t = fx.make_rmem_tensor(16, fx.Int32)
            t.store(v)
            return t

        # The previous persistent tile's C store must have read its LDS: a relocating
        # tile waits for it only before the slots that C occupies (see the prologue).
        if const_expr(c_relocate):
            if tile_idx == 0:
                workgroup_barrier(use_cluster=use_cluster)
        else:
            if const_expr(persistent_n_tiles > 1):
                rocdl.sched_barrier(0)
                tdm_ops.tensor_wait(0)
            workgroup_barrier(use_cluster=use_cluster)
        # Per-lane LDS address of every unrolled K-tile, so that fragment
        # displacements stay DS immediates inside the K loop.
        stage_a_addr, stage_b_addr, stage_sa_addr, stage_sb_addr = [], [], [], []
        sa_row, sb_col = wmb + lane, wnb + lane
        if const_expr(a_preshuffle):
            a_pair = wave_m * (warp_tile_m // 2) + (lane16 >> 1)
            a_byte = fx.index_cast(
                T.index,
                a_pair * A_LDS_ROW + (lane16 & 1) * WMMA_K + kgrp * 16,
            )
        else:
            a_byte = fx.index_cast(T.index, (wmb + lane16) * A_LDS_ROW + kgrp * 16)
        b_byte = fx.index_cast(
            T.index,
            (wnb // 16) * B_LDS_ROW + kgrp * 256 + lane16 * 16,
        )
        if const_expr(mx32):
            sa_byte = fx.index_cast(
                T.index,
                (
                    (sa_row // 32) * SUPER_K + (sa_row % 32) * 4
                    if const_expr(a_scale_preshuffled)
                    else sa_row * SA_LDS_ROW
                ),
            )
            sb_byte = fx.index_cast(
                T.index,
                (sb_col // 32) * SB_LDS_ROW
                + (0 if const_expr(b_scale_32x32) else (sb_col % 32) * 4),
            )
        else:
            sa_sel = ((wmb + lane) % 4) * 0x01010101
            sb_sel = fx.Int32(0)
            sa_byte = fx.index_cast(T.index, ((wmb + lane) // 4) * 4)
            sb_byte = fx.index_cast(T.index, (wnb // 128) * KPAIR)
        # K-tile pitch inside a slot: block128 A-scale K blocks are whole rows apart,
        # every other operand's are bytes apart in its row.
        if const_expr(mx32):
            sa_k_pitch = tile_k if const_expr(a_scale_preshuffled) else tile_k // 32
        else:
            sa_k_pitch = SA_LDS_ROW
        for stage in range_constexpr(UNROLL):
            slot, par = stage // KPAIR, stage % KPAIR
            stage_a_addr.append(
                _planar_base(PLANAR_A_BASE, STAGE_A, slot)
                + a_byte
                + par * (A_SHAPE[1] // KPAIR)
            )
            stage_b_addr.append(
                _planar_base(PLANAR_B_BASE, STAGE_B, slot)
                + b_byte
                + par * (B_SHAPE[1] // KPAIR)
            )
            stage_sa_addr.append(
                _planar_base(PLANAR_SA_BASE, STAGE_SA, slot)
                + sa_byte
                + par * sa_k_pitch
            )
            stage_sb_addr.append(
                _planar_base(PLANAR_SB_BASE, STAGE_SB, slot)
                + sb_byte
                + par * (SB_SHAPE[1] // KPAIR)
            )

        lds_load_b32, _ = make_lds_copy_ops(32)
        lds_load_b128, _ = make_lds_copy_ops(128)

        DS_PER_FRAG = 4  # ds_load_b128 per 16-dword fragment
        # Short-K block128 256-row tiles spread single ds_loads over the WMMAs.
        WAR_SCHED = k_tiles_class == 0 and tile_m == 256 and not mx32
        # Load windows (first, last WMMA position a load may follow) per quadrant.
        WAR_Q0, WAR_Q1, WAR_Q2, WAR_Q3 = (4, 15), (0, 11), (4, 11), (3, 10)
        WAR_WAIT_AT = 3
        WAR_PERM_AT = 11

        def _join(v):
            """Four 4-dword loads -> one 16-dword fragment."""
            return (
                v[0]
                .shuffle(v[1], list(range(8)))
                .shuffle(
                    v[2].shuffle(v[3], list(range(8))),
                    list(range(16)),
                )
            )

        def _frag_loads(kind, stage, idx, sink, per=DS_PER_FRAG):
            """Thunks loading A or B fragment ``idx`` of a stage, ``per`` ds_loads each;
            the last one hands the joined fragment to ``sink``."""
            if const_expr(kind == "a"):
                addr, row, span = stage_a_addr[stage], (16 // A_PAIR) * A_LDS_ROW, 32
            else:
                addr, row, span = stage_b_addr[stage], B_LDS_ROW, 512
            parts = []

            def _go(first):
                for j in range_constexpr(first, first + per):
                    parts.append(Vec(lds_load_b128(addr, idx * row + span * j)))
                if const_expr(first + per == DS_PER_FRAG):
                    sink(_join(parts))

            return [
                lambda first=first: _go(first)
                for first in range_constexpr(0, DS_PER_FRAG, per)
            ]

        def _bcast_byte(word, sel):
            """E8M0 byte -> all four bytes of an i32, so the atom's scale opsel is a no-op.

            One v_perm_b32 rather than shift + mask + multiply: each byte of `sel` names the
            source byte for that output byte, so sel = idx * 0x01010101 replicates byte idx.
            """
            return fx.Int32(rocdl.perm_b32(word, word, sel))

        def _load_sa_word(stage, sm, sa_super):
            if const_expr(mx32):
                if const_expr(SA_PANEL_STAGES > 1):
                    panel = sa_super // SA_PANEL_STAGES
                    panel_slot = panel % num_buffers
                    panel_k = (sa_super % SA_PANEL_STAGES) * (SUPER_K // 32)
                    addr = (
                        _planar_base(
                            PLANAR_SA_BASE,
                            STAGE_SA,
                            fx.index_cast(T.index, panel_slot),
                        )
                        + sa_byte
                        + fx.index_cast(T.index, panel_k)
                        + (stage % KPAIR) * sa_k_pitch
                    )
                    return lds_load_b32(addr, sm * 32 * SA_LDS_ROW)[0]
                # sm advances 32 A rows: 32 * (SUPER_K/32) bytes in row-major LDS.
                return lds_load_b32(stage_sa_addr[stage], sm * SUPER_K)[0]
            sa_addr = stage_sa_addr[stage]
            if const_expr(a_scale_unaligned):
                sa_rem = (sa_rem0 + sa_super * sa_step) & 15
                if const_expr(stage % KPAIR == 1):
                    sa_rem = sa_rem + sa_drift
                sa_addr = sa_addr + fx.index_cast(T.index, sa_rem)
            return lds_load_b32(sa_addr, sm * 32)[0]

        def _load_sb_word(stage, sn):
            return lds_load_b32(stage_sb_addr[stage], sn * SB_LDS_ROW)[0]

        def _sa_of(word):
            return word if const_expr(mx32) else _bcast_byte(word, sa_sel)

        def _sb_of(word):
            return word if const_expr(mx32) else _bcast_byte(word, sb_sel)

        N_SA = wmma_m_rep // 2
        N_SB = wmma_n_rep // 2 if const_expr(mx32) else 1

        seed_a = [fx.make_rmem_tensor(16, fx.Int32) for _ in range_constexpr(half_m)]
        seed_b = [fx.make_rmem_tensor(16, fx.Int32) for _ in range_constexpr(half_n)]
        seed_sa = [fx.make_rmem_tensor(1, fx.Int32) for _ in range_constexpr(N_SA)]
        seed_sb = [fx.make_rmem_tensor(1, fx.Int32) for _ in range_constexpr(N_SB)]

        def _store_scale(seed, word):
            seed.store(Vec.from_elements([word], fx.Int32))

        def _scale_seed_loads(stage, sa_super):
            """Thunks seeding a stage's raw scale words: A scales, then B scales."""
            return [
                lambda sm=sm: _store_scale(
                    seed_sa[sm], _load_sa_word(stage, sm, sa_super)
                )
                for sm in range_constexpr(N_SA)
            ] + [
                lambda sn=sn: _store_scale(seed_sb[sn], _load_sb_word(stage, sn))
                for sn in range_constexpr(N_SB)
            ]

        def _broadcast_seed_scales():
            """WAR_SCHED: replicate each seed's E8M0 byte ahead of the K-step using it."""
            for sm in range_constexpr(N_SA):
                _store_scale(seed_sa[sm], _sa_of(seed_sa[sm].load()[0]))
            for sn in range_constexpr(N_SB):
                _store_scale(seed_sb[sn], _sb_of(seed_sb[sn].load()[0]))

        def _frag_seed_loads(kind, stage, per=DS_PER_FRAG):
            """Thunks seeding the first-half fragments of a stage."""
            seeds = seed_a if const_expr(kind == "a") else seed_b
            return [
                thunk
                for i in range_constexpr(len(seeds))
                for thunk in _frag_loads(kind, stage, i, seeds[i].store, per)
            ]

        def _seed_thunks(stage, parity, sa_super):
            """Seed producers of ``stage``, one per WMMA slot, as (early, tail) for a
            K-step of ``parity``: the scales and the fragment kind whose seeds that
            K-step is done with, then the kind its last quadrant still reads."""
            head = _scale_seed_loads(stage, sa_super)
            a_thunks = _frag_seed_loads("a", stage)
            b_thunks = _frag_seed_loads("b", stage)
            early, tail = (
                (a_thunks, b_thunks)
                if const_expr(parity == 0)
                else (b_thunks, a_thunks)
            )
            n_slots = half_m * half_n
            if const_expr(
                mx32
                and N_SA == N_SB
                and len(head) + len(early) - (n_slots - n_slots // 4)
                > n_slots - len(tail)
            ):
                # Too many producers for the slots: pair each A-scale load with a B one.
                head = [
                    (lambda a=head[i], b=head[N_SA + i]: (a(), b()))
                    for i in range_constexpr(N_SA)
                ]
            return head + early, tail

        def _k_step(
            stage,
            parity,
            refill_delta,
            sa_super,
            fence_outstanding,
            has_next,
            prefetch=False,
            fence_extra=0,
            pref_gate=None,
            first_hook=None,
        ):
            """One K-tile, four 64x64 quadrants, at most three fragment groups live.

            The last K-tile of a slot signals the slot free and, given ``refill_delta``,
            reloads it: always in the steady loop, in the drain (``prefetch``) only for
            the next persistent tile, with ``fence_extra`` earlier such loads in flight.
            ``has_next`` seeds the fragments and scales of the following K-step."""
            next_stage = (stage + 1) % UNROLL
            slot = stage // KPAIR
            boundary = stage % KPAIR == KPAIR - 1
            refill = boundary and refill_delta is not None
            n_slots = half_m * half_n

            if const_expr(WAR_SCHED):
                # The previous K-step already broadcast the scale bytes.
                sa_k = [seed_sa[sm].load()[0] for sm in range_constexpr(N_SA)]
                sb_k = [seed_sb[sn].load()[0] for sn in range_constexpr(N_SB)]
            else:
                sa_k = [_sa_of(seed_sa[sm].load()[0]) for sm in range_constexpr(N_SA)]
                sb_k = [_sb_of(seed_sb[sn].load()[0]) for sn in range_constexpr(N_SB)]
            rocdl.sched_barrier(0)

            refill_desc = [None]

            def _prep_refill():
                if const_expr(not refill):
                    return
                if const_expr(SA_PANEL_STAGES > 1):
                    # The A-scale owner refills a panel only after its last K-step.
                    sa_next_panel = sa_super // SA_PANEL_STAGES + num_buffers
                    sa_refill = (sa_super % SA_PANEL_STAGES == SA_PANEL_STAGES - 1) & (
                        sa_next_panel
                        < (SUPERS + SA_PANEL_STAGES - 1) // SA_PANEL_STAGES
                    )
                    refill_desc[0] = _prepare_tdm(
                        slot,
                        refill_delta,
                        sa_next_panel,
                        sa_refill.select(fx.Int32(1), fx.Int32(0)),
                    )
                else:
                    refill_desc[0] = _prepare_tdm(slot, refill_delta)

            if const_expr(not WAR_SCHED):
                _prep_refill()

            def _signal(outstanding):
                pipeline_fence_signal(outstanding=outstanding, use_cluster=False)

            def _signal_free():
                """Retire this K-step's ds_loads, then signal the slot it finished."""
                rocdl.sched_barrier(0)
                rocdl.s_wait_dscnt(0)
                rocdl.sched_barrier(0)
                if const_expr(boundary and has_next):
                    if const_expr(fence_extra > 0):
                        if has_next_tile:
                            _signal(fence_outstanding + fence_extra)
                        else:
                            _signal(fence_outstanding)
                    else:
                        _signal(fence_outstanding)
                rocdl.sched_barrier(0)

            def _wait_refill():
                """Wait for the signalled slot's readers, then start its reload."""
                if const_expr(boundary and has_next):
                    # Without this fence the scheduler hoists the wait up to the
                    # signal, and the WMMAs between them no longer hide it.
                    rocdl.sched_barrier(0)
                    pipeline_fence_wait(use_cluster=False)
                    if const_expr(first_hook is not None):
                        first_hook()
                    if const_expr(refill and pref_gate is not None):
                        if pref_gate == 1:
                            tdm_ops.tensor_load_2d(refill_desc[0])
                    elif const_expr(refill and prefetch):
                        if has_next_tile:
                            tdm_ops.tensor_load_2d(refill_desc[0])
                    elif const_expr(refill):
                        tdm_ops.tensor_load_2d(refill_desc[0])
                rocdl.sched_barrier(0)

            halves = {"a": [seed_a, None], "b": [seed_b, None]}

            def _half_loads(kind):
                """Producers of the second-half fragments. Only the 256-row tile has
                enough WMMA slots for block128 to split per ds_load."""
                n = len(halves[kind][0])
                per = DS_PER_FRAG if const_expr(mx32 or tile_m != 256) else 1
                halves[kind][1] = [None] * n
                out = []
                for i in range_constexpr(n):

                    def _keep(v, i=i):
                        halves[kind][1][i] = _frag_reg(v)

                    out += _frag_loads(kind, stage, n + i, _keep, per)
                return out

            def _wmma(quad, i, j):
                """C += A[i] x B[j] on fragments i, j of quadrant (A half, B half)."""
                qm, qn = quad
                wm, wn = qm * half_m + i, qn * half_n + j
                acc = c_frags[wm * wmma_n_rep + wn]
                fx.gemm(
                    wmma_atoms[wn % 2 if mx32 else 0][wm % 2],
                    acc,
                    halves["b"][qn][j],
                    halves["a"][qm][i],
                    acc,
                    scale_a=sb_k[wn // 2 if mx32 else 0],
                    scale_b=sa_k[wm // 2],
                )

            # Quadrants as (A half, B half): the seeds, then the new half of the kind
            # loaded first, both new halves, the other new half. Even K-steps load B first.
            first, second = ("b", "a") if const_expr(parity == 0) else ("a", "b")
            walk = [(0, 0), (0, 1), (1, 1), (1, 0)]
            if const_expr(parity == 1):
                walk = [(qn, qm) for qm, qn in walk]
            # WMMA order in a quadrant, by the fragment kind that changes fastest.
            fast = {
                "a": [(p % half_m, p // half_m) for p in range(n_slots)],
                "b": [(p // half_n, p % half_n) for p in range(n_slots)],
            }

            if const_expr(WAR_SCHED):
                # Every fragment dies only in the last four WMMAs of a quadrant and
                # every load issues at least four WMMAs after the latest death, so
                # no ds_load overwrites VGPRs that an in-flight WMMA still reads.
                halves["a"][0] = [_frag_reg(t.load()) for t in seed_a]
                halves["b"][0] = [_frag_reg(t.load()) for t in seed_b]

                def _seeds(kind):
                    if const_expr(not has_next):
                        return []
                    return _frag_seed_loads(kind, next_stage, 1)

                def _spread(thunks, lo, hi):
                    out = {}
                    if const_expr(len(thunks) == 0):
                        return out
                    pos = list(range(lo, hi + 1))
                    for t_i, t in enumerate(thunks):
                        out.setdefault(pos[t_i * len(pos) // len(thunks)], []).append(t)
                    return out

                def _run(quad, order, loads, pre=None, at=None, free=0):
                    if const_expr(pre is not None):
                        pre()
                    for pos in range_constexpr(n_slots):
                        _wmma(quad, *order[pos])
                        if const_expr(at is not None and pos in at):
                            at[pos]()
                        for t in loads.get(pos, []):
                            t()
                        if const_expr(pos >= free - 1):
                            rocdl.sched_barrier(0)

                def _wait_and_seed_scales():
                    _wait_refill()
                    if const_expr(has_next):
                        for t in _scale_seed_loads(
                            next_stage, sa_super + int(boundary)
                        ):
                            t()

                diag_last = [
                    (i, j) for i in range(half_m) for j in range(half_n) if i != j
                ] + [(k, k) for k in range(half_m)]
                _run(
                    walk[0],
                    fast[second],
                    _spread(_half_loads(first), *WAR_Q0),
                    pre=_prep_refill,
                    free=WAR_Q0[0],
                )
                _run(walk[1], fast[second], _spread(_half_loads(second), *WAR_Q1))
                _run(
                    walk[2],
                    fast[first],
                    _spread(_seeds(second), *WAR_Q2),
                    pre=_signal_free,
                    at={WAR_WAIT_AT: _wait_and_seed_scales},
                )
                _run(
                    walk[3],
                    diag_last,
                    _spread(_seeds(first), *WAR_Q3),
                    at={WAR_PERM_AT: _broadcast_seed_scales} if has_next else None,
                )
                return

            def _quad(quad, order, produce, pre=None):
                """A WMMA per slot, each followed by the producer placed at that slot."""
                assert (
                    max(produce, default=-1) < n_slots
                ), "producer thunks overflow the quadrant"
                if const_expr(pre is not None):
                    pre()
                last = const_expr(max(produce) if produce else -1)
                for pos in range_constexpr(n_slots):
                    _wmma(quad, *order[pos])
                    if const_expr(pos in produce):
                        produce[pos]()
                    if const_expr(pos == last):
                        for _ in range_constexpr(2 * len(produce)):
                            rocdl.sched_mfma(1)
                            rocdl.sched_dsrd(DS_PER_FRAG // 2)
                        rocdl.sched_barrier(0)

            # Q2 waits for the refill after SLACK WMMAs; Q2 and Q3 then seed the next
            # K-step, the seeds Q3 itself still reads being replaced in its last slots.
            SLACK = n_slots // 4
            early, tail = (
                _seed_thunks(next_stage, parity, sa_super + int(boundary))
                if const_expr(has_next)
                else ([], [])
            )
            n_q2 = n_slots - SLACK
            assert len(early) - n_q2 <= n_slots - len(
                tail
            ), "seed thunks overflow Q2 + Q3"
            q2 = {SLACK - 1: _wait_refill}
            q2.update({SLACK + i: t for i, t in enumerate(early[:n_q2])})
            q3 = dict(enumerate(early[n_q2:]))
            tail_at = [n_slots - len(tail) + i for i in range_constexpr(len(tail))]
            assert not (
                set(tail_at) & set(q3)
            ), "tail seeds collide with the Q3 early seeds"
            q3.update(dict(zip(tail_at, tail)))

            sched_fence = lambda: rocdl.sched_barrier(0)
            _quad(walk[0], fast[second], dict(enumerate(_half_loads(first))))
            _quad(
                walk[1],
                fast[first if mx32 else second],
                dict(enumerate(_half_loads(second))),
                pre=sched_fence,
            )
            _quad(walk[2], fast[first], q2, pre=_signal_free)
            _quad(walk[3], fast[first], q3, pre=sched_fence)

        # K loop: n_full steady revolutions of all slots, then a drain of the last drain_s.
        SUPERS = K_TILES // KPAIR
        last_delta = (SUPERS - 1) * tdm_global_step
        n_full = (SUPERS + num_buffers - 1) // num_buffers - 1
        drain_s = SUPERS - n_full * num_buffers  # 1..num_buffers

        def _seed_delta(i):
            """Global K offset of the prologue's load of slot ``i``."""
            return fx.min(fx.Int32(i) * tdm_global_step, last_delta)

        if const_expr(c_relocate):
            # The drain loads the next tile's first n_pref slots as it frees them,
            # so that tile's prologue skips those loads.
            has_next_tile = (tile_idx < persistent_n_tiles - 1) & pref_on
            had_prev_tile = (tile_idx > 0) & pref_on
            if const_expr(bounded_n):
                # The drain prefetches with this tile's bounds: whole tiles only.
                has_next_tile = has_next_tile & (blk_n + n_adv + tile_n <= i32_n)
                had_prev_tile = had_prev_tile & (blk_n + tile_n <= i32_n)
            n_pref = fx.min(drain_s, C_EARLY_SLOTS)
            prev_pref = had_prev_tile.select(n_pref, fx.Int32(0))
            pref_delta = [
                _seed_delta(s) + tdm_tile_shift
                for s in range_constexpr(BB_SLOT + 1 if BB_OK else C_EARLY_SLOTS)
            ]
        # bb_on: this K split has four drain K-steps and a steady loop. defer: this
        # tile's slots C_EARLY_SLOTS.. wait for the previous C store until its first
        # refill. pref_bb: this wave lands the next tile's K-tile 2 in the drain.
        bb_on = defer = pref_bb = fx.Int32(0)
        if const_expr(BB_OK):
            bb_on = (n_full > 0).select(
                (drain_s == num_buffers).select(fx.Int32(1), fx.Int32(0)), fx.Int32(0)
            )
            defer = had_prev_tile.select(bb_on, fx.Int32(0))
            pref_bb = (wave == 0).select(
                fx.Int32(0), has_next_tile.select(bb_on, fx.Int32(0))
            )

        for i in range_constexpr(num_buffers):
            if const_expr(c_relocate and i == C_EARLY_SLOTS):  # noqa: SIM102
                # Slots from here on hold the previous tile's C until its TDM
                # store, and every peer's, has read it out of LDS.
                if tile_idx > 0:  # noqa: SIM102
                    if defer == 0:
                        rocdl.sched_barrier(0)
                        tdm_ops.tensor_wait(0)
                        workgroup_barrier(use_cluster=False)
            seed_delta = _seed_delta(i)
            if const_expr(c_relocate and i < C_EARLY_SLOTS):
                if fx.Int32(i) >= prev_pref:
                    tdm_ops.tensor_load_2d(_prepare_tdm(i, seed_delta))
            elif const_expr(BB_OK):
                if defer == 0:
                    tdm_ops.tensor_load_2d(_prepare_tdm(i, seed_delta))
            elif const_expr(SA_PANEL_STAGES > 1):
                sa_seed = (fx.Int32(i * SA_PANEL_STAGES) < SUPERS).select(
                    fx.Int32(1), fx.Int32(0)
                )
                tdm_ops.tensor_load_2d(
                    _prepare_tdm(i, seed_delta, fx.Int32(i), sa_seed)
                )
            else:
                tdm_ops.tensor_load_2d(_prepare_tdm(i, seed_delta))
        pipeline_fence(outstanding=num_buffers - 1, use_cluster=False)

        def _deferred_slots():
            """First refill of a deferring tile: wait out every wave's C store in the
            cluster, then load the slots that C occupied (slot 2's non-A operands
            already landed in the previous tile's drain)."""
            rocdl.sched_barrier(0)
            tdm_ops.tensor_wait(0)
            workgroup_barrier(use_cluster=use_cluster)
            if wave == 0:
                tdm_ops.tensor_load_2d(_prepare_tdm(BB_SLOT, _seed_delta(BB_SLOT)))
            for s in range_constexpr(BB_SLOT + 1, num_buffers):
                tdm_ops.tensor_load_2d(_prepare_tdm(s, _seed_delta(s)))
            rocdl.sched_barrier(0)

        for group in _seed_thunks(0, 0, fx.Int32(0)):
            for thunk in group:
                thunk()
        if const_expr(WAR_SCHED):
            _broadcast_seed_scales()

        slot_delta = [
            fx.Int32(c + num_buffers) * tdm_global_step
            for c in range_constexpr(num_buffers)
        ]

        CLUSTER_SYNC_REVS = 8  # steady revolutions between cluster barriers

        def _steady_delta(slot, rev_delta):
            return fx.min(rev_delta + slot_delta[slot], last_delta)

        def _run_steady(parity):
            for rev in range(n_full):
                rev_delta = (rev * num_buffers) * tdm_global_step
                do_sync = rev % CLUSTER_SYNC_REVS == CLUSTER_SYNC_REVS - 1

                def _first_refill(rev=rev):
                    if (rev == 0).select(defer, fx.Int32(0)) == 1:
                        _deferred_slots()

                for g in range_constexpr(UNROLL):
                    _k_step(
                        g,
                        parity,
                        (
                            _steady_delta(g // KPAIR, rev_delta)
                            if const_expr(g % KPAIR == KPAIR - 1)
                            else None
                        ),
                        rev * num_buffers + g // KPAIR,
                        num_buffers - 2,
                        True,
                        first_hook=(
                            _first_refill if const_expr(BB_OK and g == 0) else None
                        ),
                    )
                if const_expr(use_cluster):  # noqa: SIM102
                    if do_sync:
                        cluster.cluster_barrier()

        # Inline asm: the typed ``wave_id() & 1`` does not stay a scalar s_and_b32 and
        # changes the generated ISA.
        wave_parity = fx.Int32(
            llvm_dialect.inline_asm(
                T.i32,
                [as_ir_value(rocdl.wave_id())],
                "s_and_b32 $0, $1, 1",
                "=s,s,~{scc}",
                has_side_effects=True,
            )
        )
        if wave_parity == 0:
            _run_steady(0)
        else:
            _run_steady(1)

        # Retire the last steady producer before the shared drain.
        rocdl.s_wait_dscnt(0)
        rocdl.sched_barrier(0)
        for g in range_constexpr(UNROLL):
            slot = g // KPAIR
            bb_pref = BB_OK and g == BB_SLOT
            prefetch = DRAIN_PREF[g] or bb_pref
            if g < drain_s * KPAIR:
                _k_step(
                    g,
                    0,
                    pref_delta[slot] if const_expr(prefetch) else None,
                    n_full * num_buffers + slot,
                    num_buffers - 2 - slot,
                    g < UNROLL - 1,
                    prefetch=prefetch,
                    fence_extra=DRAIN_PREF_BEFORE[g],
                    pref_gate=pref_bb if const_expr(bb_pref) else None,
                )

        def _tdm_store(g_tile, lds_ptr, tile, n_warps=1, warp=0):
            """Store an LDS tile, (rows, elements per LDS row), to its global view."""
            atom, coord = fx.rocdl.cdna5.make_tiled_tdm_atom(
                fx.rocdl.TensorStore(),
                g_tile,
                fx.make_layout(tile, (tile[1], 1)),
                tile,
                num_warps=n_warps,
            )
            blk = fx.zipped_divide(coord, tile)[None, (0, None)]
            lds = _view(lds_ptr, (tile, 1), ((tile[1], 1), tile[0] * tile[1]))
            src, dst = fx.rocdl.cdna5.tdm_partition(
                atom, warp, fx.make_layout(n_warps, 1), lds, blk
            )
            fx.copy(atom, src[None, 0], dst[None, 0])

        accs = [c_frags[idx].load() for idx in range_constexpr(n_acc)]
        if const_expr(c_relocate):
            # The next tile's n_out prefetches stay in flight across the C store.
            n_out = has_next_tile.select(n_pref, fx.Int32(0))
            for c in range_constexpr(C_EARLY_SLOTS + 1):
                if n_out == c:
                    tdm_ops.tensor_wait(c)
            workgroup_barrier(use_cluster=False)
        else:
            pipeline_fence(outstanding=0, use_cluster=use_cluster)
        gC_base = fx.recast_iter(
            fx.PointerType.get(partial_elem.ir_type, arg_c.address_space),
            arg_c,
        )
        if const_expr(batched):
            gC_base = fx.add_offset(gC_base, bat * n64)
        if const_expr(c_relocate):
            q_offs = []
            for offs in (Q_OFFS, Q_TAIL_OFFS):
                q_off = fx.Int32(offs[num_waves - 1] // 2)
                for w in range_constexpr(num_waves - 1):
                    q_off = (wave == w).select(fx.Int32(offs[w] // 2), q_off)
                q_offs.append(q_off)
            q_ptrs = [base_ptr + q_off * 2 for q_off in q_offs]
            for wm in range_constexpr(wmma_m_rep):
                part = 0 if const_expr(wm * 16 < Q_SPLIT) else 1
                row0 = 0 if const_expr(part == 0) else Q_SPLIT
                for wn in range_constexpr(wmma_n_rep):
                    h = accs[wm * wmma_n_rep + wn].to(out_elem)
                    fx.ptr_store(
                        h.bitcast(fx.Int8),
                        q_ptrs[part]
                        + ((wm * 16 - row0 + lane16) * Q_LDS_ROW + wn * 16 + kgrp * 8)
                        * 2,
                    )
            rocdl.s_wait_dscnt(0)
            rocdl.sched_barrier(0)
            q_rows = _clamp(m_rows - wmb, warp_tile_m)
            q_cols = warp_tile_n
            if const_expr(bounded_n):
                q_cols = _clamp(n_cols - wnb, warp_tile_n)
                q_cols = fx.Int32(rocdl.readfirstlane(T.i32, q_cols))
            for part, row0, n_rows in (
                (0, 0, Q_SPLIT),
                (1, Q_SPLIT, warp_tile_m - Q_SPLIT),
            ):
                # Already uniform; as an SGPR it keeps the C store descriptor out of VGPRs.
                p_rows = _clamp(q_rows - row0, n_rows)
                p_rows = fx.Int32(rocdl.readfirstlane(T.i32, p_rows))
                gtQ = _view_at(
                    gC_base,
                    blk_m64 * ldc64
                    + blk_n64
                    + fx.Int64(wmb + row0) * ldc64
                    + fx.Int64(wnb),
                    (p_rows, q_cols),
                    (ldc64, 1),
                )
                _tdm_store(
                    gtQ,
                    fx.recast_iter(out_elem, base_ptr) + q_offs[part],
                    (n_rows, Q_LDS_ROW),
                )
        elif const_expr(fuse_quant_out):
            # Group-32 MXFP8 quant straight from the accumulators. A lane holds
            # 8 consecutive columns of one row per 16x16 tile (kgrp*8), so a
            # 32-column group is two adjacent tiles across lanes l and l^16:
            # one cross-lane max, no bf16 staging. Values are first rounded to
            # the output dtype, as the bf16 store does, so the codes match
            # quantize_fp8 of that output (amax floored at 1e-10, e8m0 ceil).
            from aiter.ops.flydsl.kernels.quant_utils import (
                emit_cvt_scalef32_pk8_fp8_f32,
                emit_mx_e8m0_scale,
            )
            from aiter.utility.mx_types import MxDtypeInt, MxScaleRoundModeInt

            s8 = fx.recast_iter(fx.Int8, arg_out)
            lds8 = fx.recast_iter(fx.Int8, base_ptr)
            gC8_base = fx.recast_iter(
                fx.PointerType.get(fx.Int8.ir_type, arg_c.address_space), arg_c
            )
            GN = tile_n // 32
            S8_OFF = tile_m * C8_ROW
            v2i32_ty = T.vec(2, T.i32)
            srow = ldc64 // 32  # scale row stride: one byte per 32 C columns
            c8_off = blk_m64 * ldc64 + blk_n64
            sbase_blk = blk_n64 // 32
            if const_expr(batched):
                c8_off = c8_off + bat * n64
                sbase_blk = sbase_blk + bat * (n64 // 32)
            for wm in range_constexpr(wmma_m_rep):
                row_rel = wmb + wm * 16 + lane16
                for wp in range_constexpr(wmma_n_rep // 2):
                    h0 = accs[wm * wmma_n_rep + 2 * wp].to(out_elem).to(fx.Float32)
                    h1 = accs[wm * wmma_n_rep + 2 * wp + 1].to(out_elem).to(fx.Float32)
                    amax = abs(h0[0])
                    for t in range_constexpr(1, 8):
                        amax = fx.maxnumf(amax, abs(h0[t]))
                    for t in range_constexpr(8):
                        amax = fx.maxnumf(amax, abs(h1[t]))
                    amax = fx.maxnumf(amax, fx.Float32(amax).shuffle_xor(16, WAVE))
                    amax = fx.maxnumf(amax, fx.Float32(1e-10))
                    e8 = emit_mx_e8m0_scale(
                        amax,
                        mode=MxScaleRoundModeInt.RoundUp,
                        dtype=MxDtypeInt.FP8_E4M3,
                    )
                    scale_f32 = (fx.Int32(e8) << 23).bitcast(fx.Float32)
                    p0 = emit_cvt_scalef32_pk8_fp8_f32(
                        h0.ir_value(), scale_f32, v2i32_ty=v2i32_ty, rocdl=rocdl
                    )
                    p1 = emit_cvt_scalef32_pk8_fp8_f32(
                        h1.ir_value(), scale_f32, v2i32_ty=v2i32_ty, rocdl=rocdl
                    )
                    gcol = fx.Int64(wnb + wp * 32)
                    # e4m3 bytes go through LDS so the global write is the
                    # tile's coalesced TDM store.
                    l_off = fx.Int64(row_rel) * C8_ROW + gcol + fx.Int64(kgrp * 8)
                    fx.ptr_store(
                        fx.Vector(p0).bitcast(fx.Int8), fx.add_offset(lds8, l_off)
                    )
                    fx.ptr_store(
                        fx.Vector(p1).bitcast(fx.Int8),
                        fx.add_offset(lds8, l_off + 16),
                    )
                    if kgrp == 0:
                        fx.ptr_store(
                            Vec.from_elements([fx.Uint8(e8)], fx.Int8),
                            fx.add_offset(
                                lds8,
                                fx.Int64(S8_OFF + row_rel * GN) + gcol // 32,
                            ),
                        )
            workgroup_barrier(use_cluster=False)
            # One row's GN scale bytes per thread: contiguous in the row-major
            # scale, so each row is a single small store.
            for rr in range_constexpr((tile_m + block - 1) // block):
                srow_rel = fx.Int32(tid) + rr * block
                if (srow_rel < tile_m) and (srow_rel < m_rows):
                    sv = fx.ptr_load(
                        fx.add_offset(lds8, fx.Int64(S8_OFF + srow_rel * GN)),
                        result_type=T.vec(GN, T.i8),
                    )
                    fx.ptr_store(
                        fx.Vector(sv),
                        fx.add_offset(
                            s8, (blk_m64 + fx.Int64(srow_rel)) * srow + sbase_blk
                        ),
                    )
            gtC8 = _view_at(gC8_base, c8_off, (m_rows, tile_n), (ldc64, 1))
            _tdm_store(gtC8, lds8, (tile_m, C8_ROW), num_waves, wave)
            tdm_ops.tensor_wait(0)
        else:
            for wm in range_constexpr(wmma_m_rep):
                row_rel = wmb + wm * 16 + lane16
                if const_expr(fused_splitk):
                    # Put the locally reduced stripe last so all peer rows form
                    # one contiguous TDM store, with no packing copy in LDS.
                    row_rel = (
                        row_rel + tile_m - (split_idx + 1) * (tile_m // split_k)
                    ) & (tile_m - 1)
                for wn in range_constexpr(wmma_n_rep):
                    col_rel = wnb + wn * 16 + kgrp * 8
                    h = accs[wm * wmma_n_rep + wn].to(partial_elem)
                    fx.ptr_store(
                        h.bitcast(fx.Int8),
                        base_ptr + (row_rel * C_LDS_ROW + col_rel) * partial_bytes,
                    )
            workgroup_barrier(use_cluster=False)
            c_off = blk_m64 * ldc64 + blk_n64
            if const_expr(fused_splitk):
                # Peer rows go to a packed scratch tensor, one contiguous plane
                # per split of an output tile; the local stripe stays in LDS.
                emit_fused_splitk_epilogue(
                    elem=out_elem,
                    tid=tid,
                    block=block,
                    tile_m=tile_m,
                    tile_n=tile_n,
                    lds_base_ptr=base_ptr,
                    partials=arg_c,
                    out=arg_out,
                    c_off=c_off,
                    ldc64=ldc64,
                    c_lds_row=C_LDS_ROW,
                    split_k=split_k,
                    split_idx=split_idx,
                    mn_oob=m_rows,
                    flat_tile=flat_tile,
                )
            else:
                if const_expr(split_k > 1):
                    c_off = c_off + fx.Int64(split_idx) * fx.Int64(i32_m) * ldc64
                c_cols = n_cols if const_expr(bounded_n) else tile_n
                _tdm_store(
                    _view_at(gC_base, c_off, (m_rows, c_cols), (ldc64, 1)),
                    fx.recast_iter(partial_elem, base_ptr),
                    (tile_m, C_LDS_ROW),
                    num_waves,
                    wave,
                )
                if const_expr(persistent_n_tiles == 1):
                    tdm_ops.tensor_wait(0)

    @flyc.kernel(name=kernel_name, known_block_size=[block, 1, 1])
    def kernel_gemm_a8w8_256x256(
        arg_c: fx.Pointer,
        arg_a: fx.Pointer,
        arg_b: fx.Pointer,
        arg_scale_a: fx.Pointer,
        arg_scale_b: fx.Pointer,
        i32_m: fx.Int32,
        i32_n: fx.Int32,
        i32_k: fx.Int32,
        i32_stride_ascale_k: fx.Int32,
        i32_lda: fx.Int32,
        i32_ldc: fx.Int32,
        arg_out: fx.Pointer,
        i32_cm: fx.Int32,
    ):
        prefetch_kernel_code(num_waves)
        tile_args = (
            arg_c,
            arg_a,
            arg_b,
            arg_scale_a,
            arg_scale_b,
            i32_m,
            i32_n,
            i32_k,
            i32_stride_ascale_k,
            i32_lda,
            i32_ldc,
            arg_out,
            i32_cm,
        )
        if const_expr(persistent_n_tiles == 1):
            _run_tile(*tile_args)
        else:
            for tile_idx in range(
                fx.Int32(0), fx.Int32(persistent_n_tiles), fx.Int32(1)
            ):
                _run_tile(*tile_args, tile_idx)
            tdm_ops.tensor_wait(0)

    cm = i32_cm if const_expr(use_cluster) else 1
    gx = (i32_m + (tile_m - 1)) // tile_m
    gy = (N + (tile_n - 1)) // tile_n
    if const_expr(bounded_n):
        # Whole cluster / persistent groups of N tiles; the spare ones own no column.
        n_group = cluster_n * persistent_n_tiles
        gy = ((gy + (n_group - 1)) // n_group) * n_group
    gx = (((gx > 0).select(gx, fx.Int32(1)) + (cm - 1)) // cm) * cm
    # Split gx exactly, so no workgroup is left over to recompute a duplicate tile.
    m_run_max, m_run_min = 32, 8
    pow2 = gx & -gx
    capped = (pow2 < m_run_max).select(pow2, fx.Int32(m_run_max))
    # A cluster spans consecutive bid_x, so the x extent must stay a whole number of cluster rows.
    fits_cluster = (capped % fx.Int32(cm)) == 0
    if const_expr(fused_splitk or batched):
        # Keep grid.z for K splits or batches only, so every tile's splits belong
        # to the same hardware cluster, including grids with more than 32 M tiles.
        m_run = gx
    else:
        m_run = ((gx > m_run_max) & (pow2 >= m_run_min) & fits_cluster).select(
            capped, gx
        )
    m_chunks = gx // m_run
    gy_launch = gy // persistent_n_tiles if const_expr(persistent_n_tiles > 1) else gy
    grid_arg = (m_run, gy_launch, batch if const_expr(batched) else m_chunks * split_k)
    # Runtime N/K shape checks belong to the caller.
    cluster_arg = (cm, cluster_n, cluster_k) if use_cluster else None
    kernel_gemm_a8w8_256x256(
        arg_c,
        arg_a,
        arg_b,
        arg_scale_a,
        arg_scale_b,
        i32_m,
        N,
        K,
        stride_ascale_k,
        i32_lda,
        i32_ldc,
        arg_out,
        i32_cm,
    ).launch(
        grid=grid_arg,
        block=(block, 1, 1),
        stream=stream,
        cluster=cluster_arg,
    )


launch_gemm_a8w8_256x256.compile_hints["llvm_options"] = {
    "amdgpu-expert-scheduling-mode": True,
    "amdgpu-kernarg-preload": AITER_FLYDSL_KERNARG_PRELOAD,
    "amdgpu-kernarg-preload-count": AITER_FLYDSL_KERNARG_PRELOAD_COUNT,
}
