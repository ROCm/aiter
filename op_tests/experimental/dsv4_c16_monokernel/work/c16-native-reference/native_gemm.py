# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FP8xFP8 preshuffle GEMM kernels for gfx1250."""

import math

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, range_constexpr, rocdl
from flydsl.expr.rocdl import cluster, tdm_ops
from flydsl.expr.typing import Constexpr, T
from flydsl.expr.typing import Vector as Vec
from flydsl.runtime.device import get_rocm_arch

from aiter.jit.utils.chip_info import get_lds_capacity_bytes
from aiter.ops.flydsl.kernels.gemm_common_gfx1250 import (
    make_lds_copy_ops,
    pipeline_fence,
    workgroup_barrier as _native_workgroup_barrier,
)
from aiter.ops.flydsl.kernels.gfx1250_cluster import compute_mcast_masks
from aiter.ops.flydsl.kernels.kernels_common import format_kernel_name
from aiter.ops.flydsl.kernels.tensor_shim import (
    AITER_FLYDSL_KERNARG_PRELOAD,
    AITER_FLYDSL_KERNARG_PRELOAD_COUNT,
)

# Largest TDM pad interval the gfx1250 descriptor can encode for 1-byte
# elements: log2(interval/4) - 1 must fit 3 bits, so interval <= 1024 B.
_TDM_MAX_PAD_INTERVAL_BYTES = 1024



from flydsl._mlir.dialects import llvm as llvm_dialect
from flydsl._mlir import ir

def _named_qa_barrier_pointer():
    module = ir.InsertionPoint.current.block.owner
    while module.name != "gpu.module":
        module = module.parent
    name = "native_qa_named_barrier"
    block = module.regions[0].blocks[0]
    found = any(op.name == "llvm.mlir.global" and str(op.attributes["sym_name"]) == '"'+name+'"' for op in block.operations)
    if not found:
        with ir.InsertionPoint.at_block_begin(block):
            ty = ir.Type.parse('!llvm.target<"amdgcn.named.barrier", 0>')
            glob = llvm_dialect.GlobalOp(ty, name, ir.Attribute.parse("#llvm.linkage<internal>"), addr_space=3)
            init = glob.regions[0].blocks.append()
            with ir.InsertionPoint(init):
                poison = llvm_dialect.PoisonOp(ty).result
                llvm_dialect.ReturnOp(arg=poison)
    return llvm_dialect.AddressOfOp(ir.Type.parse("!llvm.ptr<3>"), name).result


def build_native_gemm_body(tile_n, tile_k, n_warp, num_buffers, qa_named=True, parent_waves=4):
    """Return (inline FlyDSL body, required LDS bytes).

    QA: call only physical waves0,1; named barrier1 has2participants.
    QB: call all4physicalwaves. Caller supplies sharedarena and logicalNtile.
    No grid barrier or kernel launch is performed by the body.
    """
    if (tile_n,tile_k,n_warp,num_buffers) not in [(32,1024,2,4),(128,512,4,2)]:
        raise ValueError("Only audited native QA/QB profiles are supported")
    tile_m=16
    m_warp=1
    out_is_f16=0
    cluster_m=cluster_n=split_k=1
    is_mxscale=True
    block_size=128
    batched=False
    preload_ks=0
    a_preshuffle=a_scale_preshuffled=False
    mx32 = is_mxscale and block_size == 32
    mx128 = is_mxscale and block_size == 128
    if a_scale_preshuffled and not mx32:
        raise ValueError("a_scale_preshuffled is supported only for mxfp8_32")
    preload = preload_ks > 0
    # A-preshuffle: A is shuffle_mxfp8fp4_a-tiled, [M, K] -> [M/2, K/128, 2, 128]
    A_PAIR = 2 if a_preshuffle else 1
    if a_preshuffle and batched:
        raise ValueError(
            "[FlyDSL gfx1250] a_preshuffle is 2-D only: the batched A layout is "
            "[M, B, K], which shuffle_mxfp8fp4_a does not describe"
        )
    if a_preshuffle and tile_m % 2 != 0:
        raise ValueError(
            f"[FlyDSL gfx1250] a_preshuffle needs an even tile_m, got {tile_m}"
        )
    if batched and not (mx128 and split_k == 1):
        raise ValueError(
            "[FlyDSL gfx1250] the batched path needs mx128 and split_k==1, got "
            f"is_mxscale={is_mxscale}, block_size={block_size}, split_k={split_k}"
        )
    if preload and not (
        batched
        and cluster_m == 1
        and cluster_n == 1
        and preload_ks % (tile_k // 128) == 0
    ):
        raise ValueError(
            f"[FlyDSL gfx1250] preload_ks={preload_ks} needs batched, no cluster "
            f"and a multiple of tile_k/128={tile_k // 128}; got batched={batched}, "
            f"cluster={cluster_m}x{cluster_n}"
        )
    use_cluster = cluster_m > 1 or cluster_n > 1
    WMMA_M = WMMA_N = 16
    WMMA_K = 128
    WAVE = 32
    K_WS = tile_k // WMMA_K
    warp_tile_m = tile_m // m_warp
    warp_tile_n = tile_n // n_warp
    wmma_m_rep = warp_tile_m // WMMA_M
    wmma_n_rep = warp_tile_n // WMMA_N
    sa_opsel = mx32 and not a_scale_preshuffled and wmma_m_rep % 2 == 0
    sb_opsel = False  # B-scale is always n32k4 shuffled on gfx1250.
    sa_loads = wmma_m_rep // 2 if sa_opsel else wmma_m_rep
    sb_loads = wmma_n_rep // 2 if sb_opsel else wmma_n_rep
    n_acc = wmma_m_rep * wmma_n_rep
    num_waves = m_warp * n_warp
    block = num_waves * WAVE

    LDS_PAD_A = 16
    A_LDS_ROWS = tile_m // A_PAIR
    A_TDM_ROW = A_PAIR * tile_k
    A_PAD_SPLIT = max(1, -(-A_TDM_ROW // _TDM_MAX_PAD_INTERVAL_BYTES))
    A_PAD_INTERVAL = A_TDM_ROW // A_PAD_SPLIT
    A_LDS_ROW = A_TDM_ROW + A_PAD_SPLIT * LDS_PAD_A
    B_LDS_ROW = tile_k * 16
    STAGE_A = ((A_LDS_ROWS * A_LDS_ROW + 15) // 16) * 16
    STAGE_B = (((tile_n // 16) * B_LDS_ROW + 15) // 16) * 16
    ALIGNED_N = tile_n % 128 == 0 or 128 % tile_n == 0
    _period = math.lcm(tile_n, 128)
    N_BLOCKS = (
        max(1, tile_n // 128)
        if ALIGNED_N
        else max(
            (bn + tile_n - 1) // 128 - bn // 128 + 1 for bn in range(0, _period, tile_n)
        )
    )
    SC_WORDS = tile_k // 4  # shuffled 32x4 scale words per 32-row K tile
    sa_m_major = batched  # batched A-scale is [M, B, K//128]
    SA_SHAPE = (
        (
            (max(1, tile_m // 32), tile_k)
            if a_scale_preshuffled
            else (tile_m, tile_k // 32)
        )
        if mx32
        else ((tile_m, K_WS) if sa_m_major else (K_WS, tile_m))
    )
    SB_SHAPE = (tile_n // 32, tile_k) if mx32 else (N_BLOCKS, K_WS)
    STAGE_SA = ((SA_SHAPE[0] * SA_SHAPE[1] + 15) // 16) * 16
    STAGE_SB = ((SB_SHAPE[0] * SB_SHAPE[1] + 15) // 16) * 16
    SA_OFF = STAGE_A + STAGE_B
    SB_OFF = SA_OFF + STAGE_SA
    AB_PITCH = ((STAGE_A + STAGE_B + 1023) // 1024) * 1024
    SCALE_PITCH = ((SB_OFF + STAGE_SB + 1023) // 1024) * 1024
    PITCH = AB_PITCH if preload else (SCALE_PITCH if is_mxscale else AB_PITCH)
    PANEL_A = ((tile_m * preload_ks + 15) // 16) * 16 if preload else 0
    PANEL_B = ((N_BLOCKS * preload_ks + 15) // 16) * 16 if preload else 0
    PANEL_OFF = num_buffers * PITCH
    SA_LDS_OFF, SB_LDS_OFF = (0, PANEL_A) if preload else (SA_OFF, SB_OFF)
    SA_LDS_ROW = SB_LDS_ROW = preload_ks if preload else K_WS
    SA_TDM_SHAPE = (tile_m, preload_ks) if preload else SA_SHAPE
    SB_TDM_SHAPE = (N_BLOCKS, preload_ks) if preload else SB_SHAPE
    out_cls = fx.Float32 if split_k > 1 else (fx.Float16 if out_is_f16 else fx.BFloat16)
    c_bytes = 4 if split_k > 1 else 2
    C_PAD = 8 if tile_n >= 128 else 0
    C_LDS_ROW = tile_n + C_PAD
    C_STORE_B = (tile_m * C_LDS_ROW * c_bytes + 127) // 128 * 128
    ARENA_B = max(PANEL_OFF + PANEL_A + PANEL_B, C_STORE_B)
    # The compile target, not the host: AOT cross-compiles these gfx1250
    # kernels under FLYDSL_GPU_ARCH, which get_rocm_arch() honours.
    arch = get_rocm_arch().split(":", 1)[0]
    lds_cap = get_lds_capacity_bytes(arch)
    if ARENA_B > lds_cap:
        raise RuntimeError(
            f"Shared Memory Overflow: requested {ARENA_B} bytes on {arch}, "
            f"limit is {lds_cap} bytes"
        )
    use_quadrant = (wmma_m_rep % 2 == 0) and (wmma_n_rep % 2 == 0) and (n_acc >= 8)
    scale_tag = "mx32" if mx32 else ("mx128" if mx128 else "ptpc")
    kernel_name = format_kernel_name(
        f"{'batched_' if batched else ''}gemm_a8w8_{scale_tag}"
        f"_t{tile_m}x{tile_n}x{tile_k}"
        f"_mw{m_warp}_nw{n_warp}_nb{num_buffers}_sk{split_k}"
        f"_cm{cluster_m}_cn{cluster_n}"
        + ("_mbn" if batched else "")
        + (f"_pre{preload_ks}" if preload else "")
        + ("_apre" if a_preshuffle else "")
        + ("_aspre" if a_scale_preshuffled else "")
    )

    def workgroup_barrier(use_cluster=False):
        assert not use_cluster
        if qa_named and num_waves != parent_waves:
            rocdl.sched_barrier(0)
            rocdl.s_wait_dscnt(0)
            barrier_ptr = _named_qa_barrier_pointer()
            rocdl.s_barrier_join(barrier_ptr)
            rocdl.s_barrier_signal_var(barrier_ptr, num_waves)
            rocdl.s_barrier_wait(1)
            rocdl.sched_barrier(0)
        else:
            _native_workgroup_barrier(use_cluster=False)

    def pipeline_fence(outstanding=0, use_cluster=False):
        tdm_ops.tensor_wait(outstanding)
        workgroup_barrier(use_cluster=use_cluster)

    @flyc.jit
    def native_gemm_body(
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
        logical_n_tile: fx.Int32,
        external_lds: fx.Pointer,
    ):
        K_TILES = i32_k // (tile_k * split_k)
        k64 = fx.Int64(i32_k)
        lda64 = fx.Int64(i32_lda)
        ldc64 = fx.Int64(i32_ldc)

        tid = fx.Int32(fx.thread_idx.x)
        bid_x, bid_y, bid_z = fx.Int32(0), logical_n_tile, fx.Int32(0)
        kt_base = fx.Int64(bid_z) * fx.Int64(K_TILES) if split_k > 1 else None
        bz64 = fx.Int64(bid_z) if const_expr(batched) else None
        wave = rocdl.readfirstlane(T.i32, tid // WAVE)
        lane = tid % WAVE
        lane16 = lane % 16
        kgrp = lane // 16
        wave_m = wave // n_warp
        wave_n = wave % n_warp
        if const_expr(use_cluster):
            local_x, local_y = cluster.compute_cluster_position()
            a_mask, b_mask = compute_mcast_masks(local_x, local_y, cluster_m, cluster_n)
        else:
            a_mask, b_mask = 0, 0
        blk_m = bid_x * tile_m
        blk_n = bid_y * tile_n
        blk_m64 = fx.Int64(blk_m)
        blk_n64 = fx.Int64(blk_n)
        mn_oob = i32_m - blk_m  # valid M rows (A / C, and mx128's per-row A-scale)
        a_oob = (mn_oob + 1) >> 1 if const_expr(a_preshuffle) else mn_oob
        nb_oob = stride_ask64 = sa_oob = None
        if const_expr(mx32):
            sa_oob = (
                (i32_m + 31) // 32 - blk_m // 32
                if const_expr(a_scale_preshuffled)
                else mn_oob
            )
        elif const_expr(mx128):
            nb_oob = (i32_n // 128 - blk_n // 128) if not ALIGNED_N else None
            stride_ask64 = fx.Int64(i32_stride_ascale_k)

        base_ptr = external_lds

        def _bidx(p):
            return fx.Int64(fx.ptrtoint(p))

        def _buf_ptr(s):
            return fx.add_offset(base_ptr, s * PITCH)

        def _gv(base, off, shape, stride):
            return fx.Tensor(
                fx.make_view(fx.add_offset(base, off), fx.make_layout(shape, stride))
            )

        def _lv(ptr, shape, stride):
            return fx.Tensor(fx.make_view(ptr, fx.make_layout(shape, stride)))

        lds_load_b128, _ = make_lds_copy_ops(128)
        lds_load_b32 = make_lds_copy_ops(32)[0] if const_expr(mx32) else None

        gA_base = fx.recast_iter(fx.Int8, arg_a)
        gB_base = fx.recast_iter(fx.Int8, arg_b)
        gC_base = fx.recast_iter(
            fx.PointerType.get(out_cls.ir_type, arg_c.address_space), arg_c
        )
        a_off0 = blk_m64 * lda64
        b_off0 = blk_n64 // 16 * (k64 * 16)
        if const_expr(batched):
            a_off0 = a_off0 + bz64 * k64
            b_off0 = b_off0 + bz64 * fx.Int64(i32_n) * k64

        W_A, W_B = 0, 1
        gA = _gv(gA_base, a_off0, (A_LDS_ROWS, A_TDM_ROW), (A_TDM_ROW, 1))
        atomA = fx.atom_set_value(
            fx.rocdl.make_tdm_atom(
                gA,
                [a_oob, None],
                strides=[lda64 * A_PAIR, None],
                num_warps=1,
                pad_interval=A_PAD_INTERVAL,
                pad_amount=LDS_PAD_A,
                early_timeout=True,
            ),
            "workgroup_mask",
            a_mask,
        )
        gB = _gv(gB_base, b_off0, (tile_n // 16, tile_k * 16), (tile_k * 16, 1))
        atomB = fx.atom_set_value(
            fx.rocdl.make_tdm_atom(
                gB,
                [None, None],
                strides=[k64 * 16, None],
                num_warps=1,
                early_timeout=True,
            ),
            "workgroup_mask",
            b_mask,
        )
        W_SA = W_SB = gSA = atomSA = gSB = atomSB = sa_imm = sb_imm = None
        if const_expr(is_mxscale):

            def _tdm1(
                gt, outer, inner, o_stride, mask=0
            ):  # single-warp 2-D atom, both dims clamped
                atom = fx.rocdl.make_tdm_atom(
                    gt,
                    [outer, inner],
                    strides=[o_stride, None],
                    num_warps=1,
                    early_timeout=True,
                )
                return fx.atom_set_value(atom, "workgroup_mask", mask)

            W_SA, W_SB = 2 % num_waves, 3 % num_waves
            if const_expr(mx32):
                if const_expr(a_scale_preshuffled):
                    sa_off0 = blk_m64 // 32 * k64
                    gSA = _gv(arg_scale_a, sa_off0, SA_SHAPE, (tile_k, 1))
                    atomSA = _tdm1(gSA, sa_oob, None, k64, a_mask)

                    def sa_imm(kt):
                        return fx.Int64(kt) * tile_k

                else:
                    sa_off0 = blk_m64 * (k64 // 32)
                    gSA = _gv(arg_scale_a, sa_off0, SA_SHAPE, (tile_k // 32, 1))
                    atomSA = _tdm1(gSA, sa_oob, None, k64 // 32, a_mask)

                    def sa_imm(kt):
                        return fx.Int64(kt) * (tile_k // 32)

                sb_off0 = blk_n64 // 32 * k64
                gSB = _gv(arg_scale_b, sb_off0, SB_SHAPE, (tile_k, 1))
                atomSB = _tdm1(gSB, None, None, k64, b_mask)

                def sb_imm(kt):
                    return fx.Int64(kt) * tile_k

            else:
                sb_off0 = blk_n64 // 128 * (k64 // 128)
                if const_expr(batched):
                    sb_off0 = sb_off0 + bz64 * (fx.Int64(i32_n) // 128) * (k64 // 128)
                if const_expr(sa_m_major):
                    sa_off0 = blk_m64 * stride_ask64 + bz64 * (k64 // 128)
                    gSA = _gv(arg_scale_a, sa_off0, SA_TDM_SHAPE, (SA_LDS_ROW, 1))
                    atomSA = _tdm1(gSA, mn_oob, None, stride_ask64, a_mask)

                    def sa_imm(kt):
                        return fx.Int64(kt * K_WS)

                else:
                    sa_off0 = blk_m64
                    gSA = _gv(arg_scale_a, sa_off0, SA_SHAPE, (tile_m, 1))
                    atomSA = _tdm1(gSA, None, mn_oob, stride_ask64, a_mask)

                    def sa_imm(kt):
                        return fx.Int64(kt * K_WS) * stride_ask64

                gSB = _gv(arg_scale_b, sb_off0, SB_TDM_SHAPE, (SB_LDS_ROW, 1))
                atomSB = _tdm1(gSB, nb_oob, None, k64 // 128, b_mask)

                def sb_imm(kt):
                    return fx.Int64(kt) * K_WS

        def _wcopy(w, atom, gt, lv, imm_offset):
            if wave == w:
                fx.copy(atom, gt, lv, imm_offset=imm_offset)

        def issue(s, kt):
            pa = _buf_ptr(s)
            ktg = fx.Int64(kt) if kt_base is None else fx.Int64(kt) + kt_base
            _wcopy(
                W_A,
                atomA,
                gA,
                _lv(pa, (A_LDS_ROWS, A_TDM_ROW), (A_LDS_ROW, 1)),
                ktg * A_TDM_ROW,
            )
            _wcopy(
                W_B,
                atomB,
                gB,
                _lv(
                    fx.add_offset(pa, STAGE_A),
                    (tile_n // 16, tile_k * 16),
                    (B_LDS_ROW, 1),
                ),
                ktg * (tile_k * 16),
            )
            if const_expr(is_mxscale and not preload):
                _wcopy(
                    W_SA,
                    atomSA,
                    gSA,
                    _lv(fx.add_offset(pa, SA_OFF), SA_SHAPE, (SA_SHAPE[1], 1)),
                    sa_imm(ktg),
                )
                _wcopy(
                    W_SB,
                    atomSB,
                    gSB,
                    _lv(fx.add_offset(pa, SB_OFF), SB_SHAPE, (SB_SHAPE[1], 1)),
                    sb_imm(ktg),
                )

        panel_ptr = fx.add_offset(base_ptr, PANEL_OFF) if const_expr(preload) else None

        def preload_scales():
            _wcopy(
                W_SA,
                atomSA,
                gSA,
                _lv(panel_ptr, SA_TDM_SHAPE, (SA_LDS_ROW, 1)),
                fx.Int64(0),
            )
            _wcopy(
                W_SB,
                atomSB,
                gSB,
                _lv(fx.add_offset(panel_ptr, PANEL_A), SB_TDM_SHAPE, (SB_LDS_ROW, 1)),
                fx.Int64(0),
            )
            tdm_ops.tensor_wait(0)
            workgroup_barrier(use_cluster=False)

        def _scale_buf(pbuf, cur_kt):
            if const_expr(not preload):
                return pbuf
            return fx.add_offset(panel_ptr, cur_kt * K_WS)

        wmb = wave_m * warp_tile_m
        wnb = wave_n * warp_tile_n

        a_pair_base = (
            fx.Int64(
                (wave_m * (warp_tile_m // 2) + (lane16 >> 1)) * A_LDS_ROW
                + (lane16 & 1) * WMMA_K
                + kgrp * 16
            )
            if const_expr(a_preshuffle)
            else None
        )

        def load_a(buf, wm, ks):
            if const_expr(a_preshuffle):
                b0 = a_pair_base
                off0 = (
                    wm * ((WMMA_M // 2) * A_LDS_ROW)
                    + ks * (2 * WMMA_K)
                    + (ks * (2 * WMMA_K) // A_PAD_INTERVAL) * LDS_PAD_A
                )
            else:
                row = wmb + wm * 16 + lane16
                b0 = fx.Int64(row * A_LDS_ROW + ks * WMMA_K + kgrp * 16)
                off0 = 0
            v = [
                Vec(lds_load_b128(buf, b0 + (off0 + 32 * j)))
                for j in range_constexpr(4)
            ]
            v01 = v[0].shuffle(v[1], list(range(8)))
            v23 = v[2].shuffle(v[3], list(range(8)))
            return v01.shuffle(v23, list(range(16)))

        def load_b(buf, wn, ks):
            nbl = wnb // 16 + wn
            b0 = fx.Int64(
                STAGE_A + nbl * B_LDS_ROW + ks * 2048 + kgrp * 256 + lane16 * 16
            )
            v = [Vec(lds_load_b128(buf, b0 + 512 * j)) for j in range_constexpr(4)]
            v01 = v[0].shuffle(v[1], list(range(8)))
            v23 = v[2].shuffle(v[3], list(range(8)))
            return v01.shuffle(v23, list(range(16)))

        def _bcast_byte(byte):
            w = byte.to(fx.Int32)
            return fx.Int32(rocdl.perm_b32(w, w, 0))

        def load_sa(buf, sbuf, wm, ks):
            row = wmb + wm * 16 + lane16
            if const_expr(not mx32):
                off = (
                    (SA_LDS_OFF + row * SA_LDS_ROW + ks)
                    if const_expr(sa_m_major)
                    else (SA_OFF + ks * tile_m + row)
                )
                byte = fx.Uint8(fx.ptr_load(sbuf + off))
                return _bcast_byte(byte)
            if const_expr(sa_opsel):
                # One i32 per lane holds four K scales. The two 16-row WMMA
                # blocks occupy the low/high lane halves of the same VGPR.
                row = wmb + wm * 32 + lane
            if const_expr(a_scale_preshuffled):
                if const_expr(tile_m < 32):
                    row = row + blk_m % 32
                word = (row // 32) * SC_WORDS + ks * 32 + (row % 32)
            else:
                word = row * (tile_k // 128) + ks
            return lds_load_b32(buf, SA_OFF + word * 4)[0]

        def load_sb(buf, sbuf, wn, ks):
            if const_expr(mx32):
                col = wnb + wn * 16 + lane16
                word = (col // 32) * SC_WORDS + ks * 32 + (col % 32)
                return lds_load_b32(buf, SB_OFF + word * 4)[0]
            if const_expr(ALIGNED_N):
                n_block = (
                    wnb + wn * 16
                ) // 128  # local offset alone folds to a constant here
            else:
                n_block = (
                    blk_n + wnb + wn * 16
                ) // 128 - blk_n // 128  # needs tile's runtime base
            byte = fx.Uint8(
                fx.ptr_load(sbuf + (SB_LDS_OFF + n_block * SB_LDS_ROW + ks))
            )
            return _bcast_byte(byte)

        wmma_atom = None
        wmma_atoms = None
        if const_expr(is_mxscale):
            if const_expr(mx32):
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
                                opsel_a=sb_sel,
                                opsel_b=sa_sel,
                            )
                        )
                        for sa_sel in range_constexpr(2 if sa_opsel else 1)
                    ]
                    for sb_sel in range_constexpr(2 if sb_opsel else 1)
                ]
            else:
                wmma_atom = fx.make_mma_atom(
                    fx.rocdl.WMMAScale(
                        WMMA_M,
                        WMMA_N,
                        WMMA_K,
                        fx.Float8E4M3FN,
                        fx.Float8E4M3FN,
                        fx.Float32,
                    )
                )
        else:
            wmma_atom = fx.make_mma_atom(
                fx.rocdl.WMMA(WMMA_M, WMMA_N, WMMA_K, fx.Float8E4M3FN, fx.Float32)
            )
        c_frags = [fx.make_rmem_tensor(8, fx.Float32) for _ in range_constexpr(n_acc)]
        for cf in c_frags:
            cf.store(Vec.filled(8, 0.0, fx.Float32))

        def _rmem(n, v):
            t = fx.make_rmem_tensor(n, fx.Int32)
            t.store(v)
            return t

        def _mma(idx, wm, wn, wt, act, sa_k, sb_k):
            if const_expr(is_mxscale):
                fx.gemm(
                    (
                        wmma_atoms[wn % 2 if sb_opsel else 0][wm % 2 if sa_opsel else 0]
                        if const_expr(mx32)
                        else wmma_atom
                    ),
                    c_frags[idx],
                    wt,
                    act,
                    c_frags[idx],
                    scale_a=sb_k,
                    scale_b=sa_k,
                )
            else:
                fx.gemm(wmma_atom, c_frags[idx], wt, act, c_frags[idx])

        DS_A = DS_B = 4
        front_wm = (wmma_m_rep + 1) // 2
        _FRONT = list(range(front_wm))
        _BACK = list(range(front_wm, wmma_m_rep))

        def _mma_rows(wm_list, act, wt, sa_k, sb_k):
            for i in range_constexpr(len(wm_list)):
                wm = wm_list[i]
                for wn_raw in range_constexpr(wmma_n_rep):
                    wn = (wmma_n_rep - 1 - wn_raw) if (wm % 2 == 1) else wn_raw
                    idx = wm * wmma_n_rep + wn
                    _mma(
                        idx,
                        wm,
                        wn,
                        wt[wn],
                        act[i],
                        (
                            sa_k[wm // 2 if sa_opsel else wm]
                            if const_expr(is_mxscale)
                            else None
                        ),
                        (
                            sb_k[wn // 2 if sb_opsel else wn]
                            if const_expr(is_mxscale)
                            else None
                        ),
                    )

        _BS_DS = (
            wmma_n_rep * 4 + sb_loads + sa_loads if is_mxscale else wmma_n_rep * DS_B
        )

        def _load_state(buf, sbuf, ks):
            wt = [_rmem(16, load_b(buf, wn, ks)) for wn in range_constexpr(wmma_n_rep)]
            if const_expr(is_mxscale):
                sb_k = [load_sb(buf, sbuf, wn, ks) for wn in range_constexpr(sb_loads)]
                sa_k = [load_sa(buf, sbuf, wm, ks) for wm in range_constexpr(sa_loads)]
                return wt, sb_k, sa_k
            return wt, None, None

        def _kstep(buf, sbuf, ks, state, nxt_ks, prefetch_kt=None):
            wt, sb_k, sa_k = state
            act_f = [_rmem(16, load_a(buf, wm, ks)) for wm in _FRONT]
            if const_expr(len(_BACK) > 0):
                act_b = [_rmem(16, load_a(buf, wm, ks)) for wm in _BACK]
                rocdl.s_wait_dscnt(len(_BACK) * DS_A)
            else:
                rocdl.s_wait_dscnt(0)
            _mma_rows(_FRONT, act_f, wt, sa_k, sb_k)
            if const_expr(prefetch_kt is not None):
                rocdl.sched_barrier(0)
                issue(prefetch_kt % num_buffers, prefetch_kt)
                rocdl.sched_barrier(0)
            if const_expr(len(_BACK) > 0):
                rocdl.s_wait_dscnt(0)
                _mma_rows(_BACK, act_b, wt, sa_k, sb_k)
            return (
                _load_state(buf, sbuf, nxt_ks)
                if const_expr(nxt_ks is not None)
                else None
            )

        def compute_ktile_row(buf, sbuf, prefetch_kt):
            state = _load_state(buf, sbuf, 0)
            for ks in range_constexpr(K_WS):
                nxt_ks = ks + 1 if const_expr(ks + 1 < K_WS) else None
                pk = prefetch_kt if const_expr(ks == 0) else None
                state = _kstep(buf, sbuf, ks, state, nxt_ks, prefetch_kt=pk)
            _fr, _bk = front_wm * wmma_n_rep, len(_BACK) * wmma_n_rep
            for _ks in range_constexpr(K_WS):
                rocdl.sched_dsrd((_BS_DS if _ks == 0 else 0) + front_wm * DS_A)
                rocdl.sched_mfma(_fr)
                rocdl.sched_dsrd(len(_BACK) * DS_A)
                rocdl.sched_mfma(_bk)
                if const_expr(_ks < K_WS - 1):
                    rocdl.sched_dsrd(_BS_DS)
            rocdl.sched_barrier(0)

        HALF_M, HALF_N = wmma_m_rep // 2, wmma_n_rep // 2
        _NXT_DS = HALF_N * DS_B + (sb_loads + sa_loads if is_mxscale else 0)

        def _emit_block(wm0, wn0, a_frags, b_frags, sa_k, sb_k):
            for i in range_constexpr(len(a_frags)):
                wm = wm0 + i
                for j in range_constexpr(len(b_frags)):
                    wn = wn0 + j
                    _mma(
                        wm * wmma_n_rep + wn,
                        wm,
                        wn,
                        b_frags[j],
                        a_frags[i],
                        (
                            sa_k[wm // 2 if sa_opsel else wm]
                            if const_expr(is_mxscale)
                            else None
                        ),
                        (
                            sb_k[wn // 2 if sb_opsel else wn]
                            if const_expr(is_mxscale)
                            else None
                        ),
                    )

        def _load_b_half(buf, wn0, ks):
            return [
                _rmem(16, load_b(buf, wn0 + wn, ks)) for wn in range_constexpr(HALF_N)
            ]

        def _load_scales(buf, sbuf, ks):
            if const_expr(is_mxscale):
                return (
                    [load_sb(buf, sbuf, wn, ks) for wn in range_constexpr(sb_loads)],
                    [load_sa(buf, sbuf, wm, ks) for wm in range_constexpr(sa_loads)],
                )
            return None, None

        QUAD_PREFETCH_EARLY = (not is_mxscale) and K_WS >= 2

        def compute_ktile_quad(buf, sbuf, prefetch_kt):
            b_left = _load_b_half(buf, 0, 0)
            sb_k, sa_k = _load_scales(buf, sbuf, 0)
            for ks in range_constexpr(K_WS):
                nxt_ks = ks + 1 if const_expr(ks + 1 < K_WS) else None
                pf = ks == 0 and prefetch_kt is not None
                a_top = [
                    _rmem(16, load_a(buf, wm, ks)) for wm in range_constexpr(HALF_M)
                ]

                if const_expr(is_mxscale):
                    rocdl.s_wait_dscnt(DS_A * HALF_M + 2 * DS_B * HALF_N)
                rocdl.sched_barrier(0)
                _emit_block(0, 0, a_top, b_left, sa_k, sb_k)
                if const_expr(pf and QUAD_PREFETCH_EARLY):
                    rocdl.sched_barrier(0)
                    issue(prefetch_kt % num_buffers, prefetch_kt)
                    rocdl.sched_barrier(0)
                a_bot = [
                    _rmem(16, load_a(buf, HALF_M + wm, ks))
                    for wm in range_constexpr(HALF_M)
                ]
                b_right = _load_b_half(buf, HALF_N, ks)
                if const_expr(is_mxscale):
                    rocdl.s_wait_dscnt(DS_A * HALF_M + DS_B * HALF_N)
                _emit_block(HALF_M, 0, a_bot, b_left, sa_k, sb_k)
                if const_expr(pf and not QUAD_PREFETCH_EARLY):
                    rocdl.sched_barrier(0)
                    issue(prefetch_kt % num_buffers, prefetch_kt)
                    rocdl.sched_barrier(0)
                if const_expr(nxt_ks is not None):
                    nxt_b_left = _load_b_half(buf, 0, nxt_ks)
                    nxt_sb_k, nxt_sa_k = _load_scales(buf, sbuf, nxt_ks)
                if const_expr(is_mxscale):
                    rocdl.s_wait_dscnt(
                        DS_B * HALF_N if const_expr(nxt_ks is None) else _BS_DS
                    )
                _emit_block(0, HALF_N, a_top, b_right, sa_k, sb_k)
                if const_expr(is_mxscale):
                    rocdl.s_wait_dscnt(0 if const_expr(nxt_ks is None) else _NXT_DS)
                _emit_block(HALF_M, HALF_N, a_bot, b_right, sa_k, sb_k)
                rocdl.sched_barrier(0)
                if const_expr(nxt_ks is not None):
                    b_left, sb_k, sa_k = nxt_b_left, nxt_sb_k, nxt_sa_k

        def compute_ktile(buf, sbuf, prefetch_kt):
            if const_expr(use_quadrant):
                compute_ktile_quad(buf, sbuf, prefetch_kt)
            else:
                compute_ktile_row(buf, sbuf, prefetch_kt)

        def issue_ptpc_scale_loads():
            gSA_base = fx.recast_iter(
                fx.PointerType.get(fx.Float32.ir_type, arg_scale_a.address_space),
                arg_scale_a,
            )
            gSB_base = fx.recast_iter(
                fx.PointerType.get(fx.Float32.ir_type, arg_scale_b.address_space),
                arg_scale_b,
            )
            sa_view = fx.Tensor(fx.make_view(gSA_base, fx.make_layout(i32_m, 1)))
            sb_view = fx.Tensor(fx.make_view(gSB_base, fx.make_layout(i32_n, 1)))
            sa_buf = fx.rocdl.make_buffer_tensor(
                sa_view, max_size=False, num_records_bytes=i32_m * fx.Int32(4)
            )
            sb_buf = fx.rocdl.make_buffer_tensor(
                sb_view, max_size=False, num_records_bytes=i32_n * fx.Int32(4)
            )
            sa_lay, sb_lay = (fx.make_layout(1, 1), fx.make_layout(4, 1))
            sa_tiles = fx.logical_divide(sa_buf, sa_lay)
            sb_tiles = fx.logical_divide(sb_buf, sb_lay)
            sa_atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
            sb_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.Float32)

            def _issue(atom, tiles, lay, tile_idx):
                r = fx.make_rmem_tensor(lay, fx.Float32)
                fx.copy_atom_call(atom, fx.slice(tiles, (None, tile_idx)), r)
                return r

            sa_r = [
                _issue(sa_atom, sa_tiles, sa_lay, blk_m + wmb + wm * 16 + lane16)
                for wm in range_constexpr(wmma_m_rep)
            ]
            col4 = bid_y * (tile_n // 4) + wave_n * (warp_tile_n // 4) + kgrp * 2
            sb_r = [
                (
                    _issue(sb_atom, sb_tiles, sb_lay, col4 + wn * 4),
                    _issue(sb_atom, sb_tiles, sb_lay, col4 + wn * 4 + 1),
                )
                for wn in range_constexpr(wmma_n_rep)
            ]
            return sa_r, sb_r

        def epilogue_apply_ptpc_scale(scale_regs):
            sa_r, sb_r = scale_regs
            accs = [c_frags[idx].load() for idx in range_constexpr(n_acc)]
            sa = [
                Vec.from_elements([sa_r[wm].load()[0]] * 8)
                for wm in range_constexpr(wmma_m_rep)
            ]
            sb = [
                sb_r[wn][0].load().shuffle(sb_r[wn][1].load(), list(range(8)))
                for wn in range_constexpr(wmma_n_rep)
            ]
            for wm in range_constexpr(wmma_m_rep):
                for wn in range_constexpr(wmma_n_rep):
                    idx = wm * wmma_n_rep + wn
                    accs[idx] = accs[idx] * sb[wn] * sa[wm]
            return accs

        if const_expr(preload):
            preload_scales()
        if const_expr(use_cluster):
            cluster.cluster_barrier()
        for i in range_constexpr(num_buffers - 1):
            issue(i, i)
        n_steady = K_TILES - (num_buffers - 1)
        for kt in range(n_steady):
            s = kt % num_buffers
            pbuf = _buf_ptr(s)
            buf = _bidx(pbuf)
            pipeline_fence(outstanding=(num_buffers - 2), use_cluster=False)
            compute_ktile(buf, _scale_buf(pbuf, kt), kt + (num_buffers - 1))
            if const_expr(use_cluster) and kt % num_buffers == num_buffers - 1:
                cluster.cluster_barrier()
        scale_regs = None
        for j in range_constexpr(num_buffers - 1):
            kt = n_steady + j
            s = kt % num_buffers
            pbuf = _buf_ptr(s)
            buf = _bidx(pbuf)
            pipeline_fence(outstanding=(num_buffers - 2 - j), use_cluster=False)
            if const_expr(not is_mxscale and j == num_buffers - 2):
                scale_regs = issue_ptpc_scale_loads()
            compute_ktile(buf, _scale_buf(pbuf, kt), None)

        accs = None
        if const_expr(is_mxscale):
            accs = [c_frags[idx].load() for idx in range_constexpr(n_acc)]
        pipeline_fence(outstanding=0, use_cluster=use_cluster)
        if const_expr(not is_mxscale):
            accs = epilogue_apply_ptpc_scale(scale_regs)
        for wm in range_constexpr(wmma_m_rep):
            row_rel = wmb + wm * 16 + lane16
            for wn in range_constexpr(wmma_n_rep):
                col_rel = wnb + wn * 16 + kgrp * 8
                h = accs[wm * wmma_n_rep + wn].to(out_cls)
                fx.ptr_store(
                    h.bitcast(fx.Int8),
                    base_ptr + (row_rel * C_LDS_ROW + col_rel) * c_bytes,
                )
        workgroup_barrier(use_cluster=False)
        c_off_rt = blk_m64 * ldc64 + blk_n64
        if const_expr(batched):
            c_off_rt = c_off_rt + bz64 * fx.Int64(i32_n)
        if const_expr(split_k > 1):
            c_off_rt = c_off_rt + fx.Int64(bid_z) * fx.Int64(i32_m) * ldc64
        gtC = _gv(gC_base, c_off_rt, (tile_m, C_LDS_ROW), (C_LDS_ROW, 1))
        atomC = fx.rocdl.make_tdm_atom(
            gtC,
            [mn_oob, tile_n if C_PAD else None],
            strides=[ldc64, None],
            num_warps=num_waves,
            early_timeout=False,
        )
        fx.copy(
            atomC,
            _lv(fx.recast_iter(out_cls, base_ptr), (tile_m, C_LDS_ROW), (C_LDS_ROW, 1)),
            gtC,
        )
        tdm_ops.tensor_wait(0)

    return native_gemm_body, ARENA_B

"""Exact native FlyDSL QA/QB launch reference, NOT a fused/Gluon port.

Scale tensors are logical [M,K/128] views with physical M-contiguous storage.
Construct with torch.empty_strided((M,K//128),(1,M),dtype=fp8_e8m0,...).
No scale repacking, quantization, weight packing, or implicit synchronization
is performed here. Inputs must already follow the native storage contract.
"""
QA_KERNEL = 'flydsl_mxfp8_128_bpreshuffle_wmma_t16x32x1024_mw1_nw2_nb4_sk1_cm1_cn1'
QB_KERNEL = 'flydsl_mxfp8_128_bpreshuffle_wmma_t16x128x512_mw1_nw4_nb2_sk1_cm1_cn1'


def _run(xq, packed_weight, xscale, wscale, out, *, n, k, kernel):
    from aiter.ops.flydsl.mxfp8_bpreshuffle_gemm_gfx1250 import (
        run_gemm_a8w8_mxfp8_128_bpreshuffle_gfx1250,
    )
    m = xq.shape[0]
    if m != 16 or tuple(xq.shape) != (m, k):
        raise ValueError(f'This C16 native reference requires XQ[16,{k}]')
    if tuple(packed_weight.shape) != (n, k):
        raise ValueError(f'Packed weight must retain logical shape [{n},{k}]')
    if tuple(out.shape) != (m, n):
        raise ValueError(f'Output must have shape [{m},{n}]')
    if tuple(xscale.shape) != (m, k // 128) or tuple(xscale.stride()) != (1, m):
        raise ValueError('X scale must be an explicit M-contiguous [M,K/128] view with strides (1,M); row-major scales are not native-compatible')
    if not packed_weight.is_contiguous() or not wscale.is_contiguous():
        raise ValueError('Native packed weight and W scale must be contiguous')
    return run_gemm_a8w8_mxfp8_128_bpreshuffle_gfx1250(
        xq, packed_weight, xscale, wscale, out,
        kernel_name=kernel, a_is_preshuffled=False,
        allow_cluster_m_fallback=False,
    )


def qa(xq, packed_weight, xscale, wscale, out):
    """Native 16x1536x4096 QA, two waves, 48 CTAs, no split K."""
    return _run(xq, packed_weight, xscale, wscale, out,
                n=1536, k=4096, kernel=QA_KERNEL)


def qb(xq, packed_weight, xscale, wscale, out):
    """Native 16x32768x1024 QB, four waves, 256 CTAs, no split K."""
    return _run(xq, packed_weight, xscale, wscale, out,
                n=32768, k=1024, kernel=QB_KERNEL)

_QA_BODY, QA_ARENA_BYTES = build_native_gemm_body(32,1024,2,4)
_QA_BODY_REGULAR, _ = build_native_gemm_body(32,1024,2,4,False)
_QB_BODY, QB_ARENA_BYTES = build_native_gemm_body(128,512,4,2)

@flyc.jit
def launch_body_smoke(C:fx.Pointer,A:fx.Pointer,B:fx.Pointer,SA:fx.Pointer,SB:fx.Pointer,
                      stream:fx.Stream, QA:Constexpr[bool], TWO_PHYSICAL:Constexpr[bool]=False):
    @flyc.kernel(known_block_size=[64 if TWO_PHYSICAL else 128,1,1])
    def smoke(C:fx.Pointer,A:fx.Pointer,B:fx.Pointer,SA:fx.Pointer,SB:fx.Pointer):
        arena=fx.SharedAllocator(static=False)
        arena.allocate(QA_ARENA_BYTES if const_expr(QA) else QB_ARENA_BYTES)
        tid=fx.Int32(fx.thread_idx.x)
        tile=fx.Int32(fx.block_idx.x)
        if const_expr(QA):
            if tid<64:
                if const_expr(TWO_PHYSICAL):
                    _QA_BODY_REGULAR(C,A,B,SA,SB,fx.Int32(16),fx.Int32(1536),fx.Int32(4096),fx.Int32(16),fx.Int32(4096),fx.Int32(1536),tile,arena.base_ptr)
                else:
                    _QA_BODY(C,A,B,SA,SB,fx.Int32(16),fx.Int32(1536),fx.Int32(4096),fx.Int32(16),fx.Int32(4096),fx.Int32(1536),tile,arena.base_ptr)
        else:
            _QB_BODY(C,A,B,SA,SB,fx.Int32(16),fx.Int32(32768),fx.Int32(1024),fx.Int32(16),fx.Int32(1024),fx.Int32(32768),tile,arena.base_ptr)
    smoke(C,A,B,SA,SB).launch(grid=(48 if const_expr(QA) else 256,1,1),block=(64 if const_expr(TWO_PHYSICAL) else 128,1,1),stream=stream)

launch_body_smoke.compile_hints['llvm_options'] = {
    'amdgpu-expert-scheduling-mode': True,
    'amdgpu-kernarg-preload': AITER_FLYDSL_KERNARG_PRELOAD,
    'amdgpu-kernarg-preload-count': AITER_FLYDSL_KERNARG_PRELOAD_COUNT,
}
