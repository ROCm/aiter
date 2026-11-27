# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""MLA gather projection, gfx1250: A-resident head loop over pair-type steps.

One workgroup owns a 256-token M tile and loops over its heads in pairs (2j, 2j+1);
each pair is computed as two 256-column steps:
  * K step [nope(2j) | nope(2j+1)], V step [v(2j) | v(2j+1)]: per step the same
    256 x 512 weight slab and GEMM shape as one head, but the V step's row is one
    256 B-aligned 256 B segment (whole 256 B blocks), and the K step writes the pair's
    384 B K span as three 128 B lines L0 = nope(2j), L1 = rope|nope(2j+1)[:64],
    L2 = nope(2j+1)[64:]|rope;
  * the tile's 512-wide FP8 latent is TDM-gathered into LDS ONCE and stays resident;
    per step only the weight slab streams through a 3-slot B ring, two K tiles ahead;
  * one 320 B staging row per token serves both step types:
        K step: [rope | nope(2j+1)[:64] | nope(2j) | nope(2j+1)[64:] of the next row]
        V step: [rope | v(2j) | v(2j+1)]
    RoPE is converted once and parked at [0, 64); L2 is the 128 B window that starts
    64 B before a row (64 B guard before row 0);
  * the epilogue folds the ptpc row scale, activation scale and output descale into
    one factor per column, clamps to E4M3 and converts with
    v_cvt_scalef32_pk8_fp8_f32;
  * stores are TDM stores with the non-temporal hint: a store that covers only part
    of a 256 B block otherwise makes L2 fetch the block from HBM first (one read per
    128 B line written); NT stores skip that fill;
  * SCHED_MODE's WMMA multi-cycle arbitration stall is disabled.
FP8 output only.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.rocdl import cvt_scale_pk8_f32_fp8, tdm_ops
from flydsl.expr.typing import Constexpr, T
from flydsl.expr.typing import Vector as Vec
from flydsl.runtime.device import get_rocm_arch

from aiter.jit.utils.chip_info import get_lds_capacity_bytes
from aiter.ops.flydsl.kernels.gemm_common_gfx1250 import make_lds_copy_ops
from aiter.ops.flydsl.kernels.kernels_common import format_kernel_name
from aiter.ops.flydsl.kernels.mega_moe_gfx1250.tdm_gather_shim import (
    TDMGatherDescriptor,
    make_tensor_gather_descriptor,
)
from aiter.ops.flydsl.kernels.tensor_shim import (
    AITER_FLYDSL_KERNARG_PRELOAD,
    AITER_FLYDSL_KERNARG_PRELOAD_COUNT,
    buf_load_scalar,
    ptr_buf_tensor,
    ptr_rsrc,
)

_K = 512
_CACHE_ROW = 576
_ROPE_OFFSET = 512
_ROPE_WIDTH = 64

_TILE_M = 256
_TILE_N = 256
_TILE_K = 128
_M_WARP = 4
_N_WARP = 2
_NB = 3  # B ring slots
_STG_ROW = 320  # staging pitch (see module docstring)
_STORE_NT = 1  # TDM store cache policy: TH = non-temporal (no write-allocate fill)


def _sat(v):
    """Clamp to the E4M3 range; v_cvt_scalef32_pk8_fp8_f32 does not saturate. NaN stays NaN."""
    return v.maximumf(fx.full(8, -448.0, fx.Float32)).minimumf(
        fx.full(8, 448.0, fx.Float32)
    )


def _pack_fp8x8(v):
    """Eight fp32 values as saturated E4M3 bytes; NaN stays NaN (stock path)."""
    v = _sat(v)
    words = []
    for i in (0, 4):
        lo = rocdl.cvt_pk_fp8_f32(T.i32, v[i], v[i + 1], fx.Int32(0), False)
        hi = rocdl.cvt_pk_fp8_f32(T.i32, v[i + 2], v[i + 3], lo, True)
        words.append(fx.full(1, fx.Int32(hi), fx.Int32))
    return words[0].shuffle(words[1], [0, 1]).bitcast(fx.Int8)


def _cvt8(v):
    """Eight fp32 -> eight E4M3 bytes in one gfx1250 v_cvt_scalef32_pk8_fp8_f32
    (unit scale, RNE) after the clamp; same bytes as _pack_fp8x8."""
    return Vec(
        rocdl.cvt_scalef32_pk8_fp8_f32(
            T.vec(2, T.i32), _sat(v).ir_value(), fx.Float32(1.0).ir_value()
        )
    ).bitcast(fx.Int8)


@flyc.jit
def launch_gather_kv_b_proj_hl_gfx1250(
    arg_k: fx.Pointer,
    arg_v: fx.Pointer,
    arg_cache: fx.Pointer,
    arg_indices: fx.Pointer,
    arg_b: fx.Pointer,
    arg_scale_a: fx.Pointer,
    arg_scale_b: fx.Pointer,
    i32_num_blocks: fx.Int32,
    i32_m: fx.Int32,
    f32_scale_a: fx.Float32,
    arg_k_out_scale: fx.Pointer,
    arg_v_out_scale: fx.Pointer,
    i32_hpg: fx.Int32,
    stream: fx.Stream,
    n_heads: Constexpr[int] = 96,
    scale_a_is_ptr: Constexpr[int] = 1,
):
    tile_m, tile_n, tile_k = _TILE_M, _TILE_N, _TILE_K
    m_warp, n_warp = _M_WARP, _N_WARP
    nb = _NB
    if n_heads % 2 != 0:
        raise ValueError("n_heads must be even")
    ldk = n_heads * 192
    ldv = n_heads * 128
    WMMA_M = WMMA_N = 16
    WMMA_K = 128
    WAVE = 32
    K_TILES = _K // tile_k
    warp_tile_m = tile_m // m_warp
    warp_tile_n = tile_n // n_warp
    wmma_m_rep = warp_tile_m // WMMA_M
    wmma_n_rep = warp_tile_n // WMMA_N
    n_acc = wmma_m_rep * wmma_n_rep
    num_waves = m_warp * n_warp
    block = num_waves * WAVE

    LDS_PAD_A = 16
    A_GATHER_ROWS = 8
    A_GATHER_GROUPS = tile_m // (num_waves * A_GATHER_ROWS)
    A_LDS_ROWS = tile_m // 2
    A_TDM_ROW = 2 * tile_k
    A_PAD_INTERVAL = A_TDM_ROW
    A_LDS_ROW = A_TDM_ROW + LDS_PAD_A
    B_LDS_ROW = tile_k * 16
    STAGE_A = ((A_LDS_ROWS * A_LDS_ROW + 15) // 16) * 16
    STAGE_B = ((tile_n // 16) * B_LDS_ROW + 15) // 16 * 16
    SR = _STG_ROW
    A_BASE = 0
    B_BASE = A_BASE + K_TILES * STAGE_A
    CK_GUARD = B_BASE + nb * STAGE_B  # 64 B: row 0's L2 window starts here
    CK0 = CK_GUARD + 64  # staging row r starts at CK0 + r * SR
    ROPE_TMP = CK0  # raw rope rows (16 KB), converted into place before the first step
    # +64: the dense 320 B TDM window of the V store (LDS offset 64) ends 64 B past row 255
    ARENA_B = CK0 + tile_m * SR + 64
    arch = get_rocm_arch().split(":", 1)[0]
    lds_cap = get_lds_capacity_bytes(arch)
    if ARENA_B > lds_cap:
        raise RuntimeError(
            f"Shared Memory Overflow: requested {ARENA_B} bytes on {arch}, "
            f"limit is {lds_cap} bytes"
        )
    kernel_name = format_kernel_name(
        "gather_kv_b_proj_hl"
        f"_t{tile_m}x{tile_n}x{tile_k}"
        f"_mw{m_warp}_nw{n_warp}_nb{nb}_h{n_heads}_pairstep_nt_fp8"
    )

    @flyc.kernel(name=kernel_name, known_block_size=[block, 1, 1])
    def kernel_gather_kv_b_proj_hl(
        arg_k: fx.Pointer,
        arg_v: fx.Pointer,
        arg_cache: fx.Pointer,
        arg_indices: fx.Pointer,
        arg_b: fx.Pointer,
        arg_scale_a: fx.Pointer,
        arg_scale_b: fx.Pointer,
        i32_num_blocks: fx.Int32,
        i32_m: fx.Int32,
        f32_scale_a: fx.Float32,
        arg_k_out_scale: fx.Pointer,
        arg_v_out_scale: fx.Pointer,
        i32_hpg: fx.Int32,
    ):
        k64 = fx.Int64(_K)

        tid = fx.Int32(fx.thread_idx.x)
        bid_x, bid_y, _ = fx.block_idx
        wave = rocdl.readfirstlane(T.i32, tid // WAVE)
        lane = tid % WAVE
        lane16 = lane % 16
        kgrp = lane // 16
        wave_m = wave // n_warp
        wave_n = wave % n_warp
        blk_m = bid_x * tile_m
        mn_oob = i32_m - blk_m
        head0 = bid_y * i32_hpg

        arena = fx.SharedAllocator(static=False)
        arena.allocate(ARENA_B)
        base_ptr = arena.base_ptr

        def _bidx(p):
            return fx.Int64(fx.ptrtoint(p))

        def _gv(base, off, shape, stride):
            return fx.Tensor(
                fx.make_view(fx.add_offset(base, off), fx.make_layout(shape, stride))
            )

        def _lv(ptr, shape, stride):
            return fx.Tensor(fx.make_view(ptr, fx.make_layout(shape, stride)))

        lds_load_b64, _ = make_lds_copy_ops(64)
        lds_load_b128, _ = make_lds_copy_ops(128)

        gA_base = fx.recast_iter(fx.Int8, arg_cache)
        gB_base = fx.recast_iter(fx.Int8, arg_b)
        gK_base = fx.recast_iter(fx.Int8, arg_k)
        gV_base = fx.recast_iter(fx.Int8, arg_v)
        lds_i8 = fx.recast_iter(fx.Int8, base_ptr)
        lds_base = _bidx(base_ptr)
        cache_rows = fx.Tensor(fx.make_view(gA_base, fx.make_layout(_CACHE_ROW, 1)))
        lds_bytes = fx.Tensor(
            fx.make_view(
                fx.recast_iter(fx.Int8, base_ptr),
                fx.make_layout(ARENA_B, 1),
            )
        )
        index_rsrc = ptr_rsrc(arg_indices, num_records_bytes=fx.Int64(i32_m) * 4)

        # SCHED_MODE: no WMMA multi-cycle arbitration stall (other waves' VALU / LDS
        # work issues while a WMMA is in flight).
        rocdl.disable_xdl_arb_stall()

        # ---------------- prologue: A (all K) + RoPE gathers, once ----------------
        zero_indices = [fx.Int32(0) for _ in range_constexpr(A_GATHER_ROWS)]
        gather_template = make_tensor_gather_descriptor(
            global_ptr=cache_rows,
            lds_memref=lds_bytes,
            row_indices=zero_indices,
            row_width=tile_k,
            tensor_dim0=_CACHE_ROW,
            tensor_dim1=i32_num_blocks,
            stride=_CACHE_ROW,
            elem_bytes=1,
            pad_interval=A_PAD_INTERVAL,
            pad_amount=LDS_PAD_A,
            index_size=32,
            gather_tile_dim1=A_GATHER_ROWS,
            lds_byte_offset=fx.index_cast(T.index, fx.Int32(0)),
            workgroup_mask=0,
        )
        rope_template = make_tensor_gather_descriptor(
            global_ptr=cache_rows,
            lds_memref=lds_bytes,
            row_indices=zero_indices,
            row_width=_ROPE_WIDTH,
            tensor_dim0=_CACHE_ROW,
            tensor_dim1=i32_num_blocks,
            stride=_CACHE_ROW,
            elem_bytes=1,
            pad_interval=_ROPE_WIDTH,
            pad_amount=0,
            index_size=32,
            gather_tile_dim1=A_GATHER_ROWS,
            lds_byte_offset=fx.index_cast(T.index, fx.Int32(ROPE_TMP)),
            global_byte_offset=_ROPE_OFFSET,
            workgroup_mask=0,
        )

        def _gather_indices(group):
            local_base = group * num_waves * A_GATHER_ROWS + wave * A_GATHER_ROWS
            global_base = blk_m + local_base
            lo = Vec(buf_load_scalar(index_rsrc, global_base))
            hi = Vec(buf_load_scalar(index_rsrc, global_base + 4))
            rows = []
            for i in range_constexpr(A_GATHER_ROWS):
                physical = lo[i] if const_expr(i < 4) else hi[i - 4]
                rows.append((local_base + i < mn_oob).select(physical, i32_num_blocks))
            return local_base, rows

        rocdl.sched_barrier(0)
        for group in range_constexpr(A_GATHER_GROUPS):
            local_base, row_indices = _gather_indices(group)
            for kt in range_constexpr(K_TILES):
                desc = make_tensor_gather_descriptor(
                    global_ptr=cache_rows,
                    lds_memref=lds_bytes,
                    row_indices=row_indices,
                    row_width=tile_k,
                    tensor_dim0=_CACHE_ROW,
                    tensor_dim1=i32_num_blocks,
                    stride=_CACHE_ROW,
                    elem_bytes=1,
                    pad_interval=A_PAD_INTERVAL,
                    pad_amount=LDS_PAD_A,
                    index_size=32,
                    gather_tile_dim1=A_GATHER_ROWS,
                    lds_byte_offset=fx.index_cast(
                        T.index,
                        fx.Int32(A_BASE + kt * STAGE_A + (local_base // 2) * A_LDS_ROW),
                    ),
                    global_byte_offset=fx.Int64(kt * tile_k),
                )
                tdm_ops.tensor_load_gather(
                    TDMGatherDescriptor(
                        dgroup0=desc.dgroup0,
                        dgroup1=gather_template.dgroup1,
                        dgroup2=desc.dgroup2,
                        dgroup3=desc.dgroup3,
                    )
                )
            rdesc = make_tensor_gather_descriptor(
                global_ptr=cache_rows,
                lds_memref=lds_bytes,
                row_indices=row_indices,
                row_width=_ROPE_WIDTH,
                tensor_dim0=_CACHE_ROW,
                tensor_dim1=i32_num_blocks,
                stride=_CACHE_ROW,
                elem_bytes=1,
                pad_interval=_ROPE_WIDTH,
                pad_amount=0,
                index_size=32,
                gather_tile_dim1=A_GATHER_ROWS,
                lds_byte_offset=fx.index_cast(
                    T.index, fx.Int32(ROPE_TMP + local_base * _ROPE_WIDTH)
                ),
                global_byte_offset=_ROPE_OFFSET,
            )
            tdm_ops.tensor_load_gather(
                TDMGatherDescriptor(
                    dgroup0=rdesc.dgroup0,
                    dgroup1=rope_template.dgroup1,
                    dgroup2=rdesc.dgroup2,
                    dgroup3=rdesc.dgroup3,
                )
            )
        rocdl.sched_barrier(0)

        # ---------------- B ring: one 256-column weight slab per step ----------------
        # Step s of pair j: columns [nope(2j) | nope(2j+1)] (s even) or [v(2j) | v(2j+1)]
        # (s odd), i.e. 16-column groups 32j + 8 (s & 1) + 16 c + [0, 8) for halves c = 0, 1
        # = 16 s - 8 (s & 1) + 16 c + [0, 8). Wave w loads groups 2 (w % 4) + {0, 1} of
        # half w // 4 into slot rows 2w, 2w + 1.
        B_ROWS_PER_WAVE = (tile_n // 16) // num_waves
        b_wave_row = wave * B_ROWS_PER_WAVE
        b_wave_grp = (wave // 4) * 16 + (wave % 4) * B_ROWS_PER_WAVE
        B_TILE = (B_ROWS_PER_WAVE, tile_k * 16)

        def issue_b(step, kt, slot):
            """One TDM load per wave: this wave's rows of step ``step`` K tile ``kt``."""
            grp = step * 16 - (step & 1) * 8 + b_wave_grp
            gB = _gv(
                gB_base,
                fx.Int64(grp) * (k64 * 16),
                (B_ROWS_PER_WAVE, K_TILES * tile_k * 16),
                (k64 * 16, 1),
            )
            atomB, coordB = fx.rocdl.cdna5.make_tiled_tdm_atom(
                fx.rocdl.TensorLoad(),
                gB,
                fx.make_layout(B_TILE, (B_LDS_ROW, 1)),
                B_TILE,
                num_warps=1,
            )
            blkB = fx.zipped_divide(coordB, B_TILE)[None, (0, None)]
            b_box = _lv(
                fx.add_offset(
                    base_ptr, B_BASE + slot * STAGE_B + b_wave_row * B_LDS_ROW
                ),
                (B_TILE, 1),
                ((B_LDS_ROW, 1), B_ROWS_PER_WAVE * B_LDS_ROW),
            )
            tBs, tBg = fx.rocdl.cdna5.tdm_partition(
                atomB, 0, fx.make_layout(1, 1), b_box, blkB
            )
            fx.copy(atomB, tBg[None, fx.Int32(kt)], tBs[None, 0])

        last_step = head0 + (i32_hpg - 1)
        rocdl.sched_barrier(0)
        issue_b(head0, 0, 0)
        issue_b(head0, 1, 1)
        issue_b(head0, 2, 2)
        rocdl.sched_barrier(0)

        # ---------------- fragments / MMA ----------------
        wmb = wave_m * warp_tile_m
        wnb = wave_n * warp_tile_n
        a_pair_base = fx.Int64(
            (wave_m * (warp_tile_m // 2) + (lane16 >> 1)) * A_LDS_ROW
            + (lane16 & 1) * WMMA_K
            + kgrp * 16
        )

        def load_a(kt, wm):
            offset = (
                a_pair_base + (A_BASE + kt * STAGE_A) + wm * ((WMMA_M // 2) * A_LDS_ROW)
            )
            v = [
                Vec(lds_load_b128(lds_base, offset + 32 * j))
                for j in range_constexpr(4)
            ]
            v01 = v[0].shuffle(v[1], list(range(8)))
            v23 = v[2].shuffle(v[3], list(range(8)))
            return v01.shuffle(v23, list(range(16)))

        def load_b(slot, wn):
            nbl = wnb // 16 + wn
            b0 = fx.Int64(
                B_BASE + slot * STAGE_B + nbl * B_LDS_ROW + kgrp * 256 + lane16 * 16
            )
            v = [Vec(lds_load_b128(lds_base, b0 + 512 * j)) for j in range_constexpr(4)]
            v01 = v[0].shuffle(v[1], list(range(8)))
            v23 = v[2].shuffle(v[3], list(range(8)))
            return v01.shuffle(v23, list(range(16)))

        wmma_atom = fx.make_mma_atom(
            fx.rocdl.WMMA(WMMA_M, WMMA_N, WMMA_K, fx.Float8E4M3FN, fx.Float32)
        )
        c_frags = [fx.make_rmem_tensor(8, fx.Float32) for _ in range_constexpr(n_acc)]
        for cf in c_frags:
            cf.store(fx.full(8, 0.0, fx.Float32))

        def _rmem(n, v):
            t = fx.make_rmem_tensor(n, fx.Int32)
            t.store(v)
            return t

        def _mma(idx, wt, act):
            fx.gemm(wmma_atom, c_frags[idx], wt, act, c_frags[idx])

        HALF_M, HALF_N = wmma_m_rep // 2, wmma_n_rep // 2

        def _emit_block(wm0, wn0, a_frags, b_frags):
            for i in range_constexpr(len(a_frags)):
                wm = wm0 + i
                for j in range_constexpr(len(b_frags)):
                    wn = wn0 + j
                    _mma(wm * wmma_n_rep + wn, b_frags[j], a_frags[i])

        def compute_ktile(kt, slot, prefetch):
            b_left = [_rmem(16, load_b(slot, wn)) for wn in range_constexpr(HALF_N)]
            a_top = [_rmem(16, load_a(kt, wm)) for wm in range_constexpr(HALF_M)]
            rocdl.sched_barrier(0)
            _emit_block(0, 0, a_top, b_left)
            rocdl.sched_barrier(0)
            prefetch()
            rocdl.sched_barrier(0)
            a_bot = [
                _rmem(16, load_a(kt, HALF_M + wm)) for wm in range_constexpr(HALF_M)
            ]
            b_right = [
                _rmem(16, load_b(slot, HALF_N + wn)) for wn in range_constexpr(HALF_N)
            ]
            _emit_block(HALF_M, 0, a_bot, b_left)
            _emit_block(0, HALF_N, a_top, b_right)
            _emit_block(HALF_M, HALF_N, a_bot, b_right)
            rocdl.sched_barrier(0)

        # ---------------- epilogue helpers ----------------
        gSB_base = fx.recast_iter(
            fx.PointerType.get(fx.Float32.ir_type, arg_scale_b.address_space),
            arg_scale_b,
        )
        sb_view = fx.Tensor(fx.make_view(gSB_base, fx.make_layout(n_heads * 256, 1)))
        sb_buf = fx.rocdl.make_buffer_tensor(
            sb_view, max_size=False, num_records_bytes=fx.Int32(n_heads * 256 * 4)
        )
        sb_lay = fx.make_layout(4, 1)
        sb_tiles = fx.logical_divide(sb_buf, sb_lay)
        sb_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.Float32)

        def issue_weight_scale_loads(step):
            def _issue(tile_idx):
                r = fx.make_rmem_tensor(sb_lay, fx.Float32)
                fx.copy_atom_call(sb_atom, fx.slice(sb_tiles, (None, tile_idx)), r)
                return r

            # first column of this wave's half: 16 * (16 s - 8 (s & 1) + 16 wave_n)
            col4 = step * 64 - (step & 1) * 32 + wave_n * 64 + kgrp * 2
            return [
                (_issue(col4 + wn * 4), _issue(col4 + wn * 4 + 1))
                for wn in range_constexpr(wmma_n_rep)
            ]

        activation_scale = (
            ptr_buf_tensor(arg_scale_a, fx.Float32, n=1, num_records_bytes=4)[0]
            if const_expr(scale_a_is_ptr)
            else f32_scale_a
        )
        inv_k, inv_v = (
            1.0 / ptr_buf_tensor(p, fx.Float32, n=1, num_records_bytes=4)[0]
            for p in (arg_k_out_scale, arg_v_out_scale)
        )
        sa_inv_k = fx.Float32(activation_scale * inv_k)
        sa_inv_v = fx.Float32(activation_scale * inv_v)
        wn1 = wave_n == 1

        def epi_layout(is_v):
            """Wave-uniform staging base / odd-nope shift of this step (pitch SR).
            K step: nope(2j) (wave_n 0) at +128, nope(2j+1) (wave_n 1) at +64, its cols
            >= 64 shifted -192 B into the previous row's tail. V step: v(2j) at +64,
            v(2j+1) at +192."""
            base = is_v.select(
                wn1.select(fx.Int32(CK0 + 192), fx.Int32(CK0 + 64)),
                wn1.select(fx.Int32(CK0 + 64), fx.Int32(CK0 + 128)),
            )
            hi_shift = is_v.select(fx.Int32(0), wn1.select(fx.Int32(-192), fx.Int32(0)))
            sainv = fx.full(8, fx.Float32(is_v.select(sa_inv_v, sa_inv_k)), fx.Float32)
            return base, hi_shift, sainv

        def stage_c(sb_r, is_v):
            """Scale, clamp, convert and stage every accumulator; zero them for the next step."""
            base, hi_shift, sainv = epi_layout(is_v)
            fac = [
                sb_r[wn][0].load().shuffle(sb_r[wn][1].load(), list(range(8))) * sainv
                for wn in range_constexpr(wmma_n_rep)
            ]
            for wm in range_constexpr(wmma_m_rep):
                row_off = base + (wmb + wm * 16 + lane16) * SR
                for wn in range_constexpr(wmma_n_rep):
                    idx = wm * wmma_n_rep + wn
                    h = _cvt8(c_frags[idx].load() * fac[wn])
                    col = kgrp * 8 + wn * 16
                    if const_expr(wn * 16 >= 64):
                        off = row_off + col + hi_shift
                    else:
                        off = row_off + col
                    fx.ptr_store(h, base_ptr + off)
            for cf in c_frags:
                cf.store(fx.full(8, 0.0, fx.Float32))

        blk_m64 = fx.Int64(blk_m)
        ST_TILE = (
            tile_m,
            SR,
        )  # dense LDS window, clipped to the row segment by the global view

        def _tdm_store(g_base, g_off, ld, lds_off, width):
            gt = _gv(g_base, g_off, (mn_oob, width), (fx.Int64(ld), 1))
            atom, coord = fx.rocdl.cdna5.make_tiled_tdm_atom(
                fx.rocdl.TensorStore(cache_modifier=_STORE_NT),
                gt,
                fx.make_layout(ST_TILE, (SR, 1)),
                ST_TILE,
                num_warps=num_waves,
            )
            blk = fx.zipped_divide(coord, ST_TILE)[None, (0, None)]
            s = _lv(
                fx.add_offset(lds_i8, lds_off), (ST_TILE, 1), ((SR, 1), tile_m * SR)
            )
            ts, tg = fx.rocdl.cdna5.tdm_partition(
                atom, wave, fx.make_layout(num_waves, 1), s, blk
            )
            fx.copy(atom, ts[None, 0], tg[None, 0])

        def store_c(step, is_v):
            """K step: L0, L1, L2 (128 B lines of the pair's 384 B K span); V step: the
            pair's V as one 256 B-aligned 256 B row segment per token."""
            pair = fx.Int64(step >> 1)
            kg = blk_m64 * ldk + pair * 384
            vg = blk_m64 * ldv + pair * 256
            if is_v:
                _tdm_store(gV_base, vg, ldv, CK0 + 64, 256)
            else:
                _tdm_store(gK_base, kg, ldk, CK0 + 128, 128)
                _tdm_store(gK_base, kg + 128, ldk, CK0, 128)
                _tdm_store(gK_base, kg + 256, ldk, CK0 - 64, 128)

        def rope_to_staging():
            """Raw RoPE rows (ROPE_TMP, 64 B each) -> scaled fp8 at CK0 + SR r + [0, 64).
            ROPE_TMP overlaps the staging rows, so all rows are read before any is written.
            """
            rope_scale = fx.full(8, sa_inv_k, fx.Float32)
            row = tid  # 256 threads, one row each
            words = [
                Vec(
                    lds_load_b64(
                        lds_base, fx.Int64(ROPE_TMP + row * _ROPE_WIDTH + j * 8)
                    )
                )
                for j in range_constexpr(_ROPE_WIDTH // 8)
            ]
            rocdl.s_wait_dscnt(0)
            gpu.barrier()
            for j in range_constexpr(_ROPE_WIDTH // 8):
                values = Vec(
                    cvt_scale_pk8_f32_fp8(
                        T.vec(8, T.f32),
                        words[j].ir_value(),
                        fx.Int32(127).ir_value(),
                        0,
                    )
                )
                packed = _pack_fp8x8(values * rope_scale)
                fx.ptr_store(packed, base_ptr + CK0 + j * 8 + row * SR)

        # ---------------- main loop over steps ----------------
        # Per-wave TDM issue order (steady, step s):
        #   epi(s-1): B(s,2) S(s-1) | c1: B(s,3) | c2: B(s+1,0) | c3: B(s+1,1) | epi(s): B(s+1,2) S(s)
        # fence(s,kt) leaves what was issued after B(s,kt): kt0 -> 2+S, kt1/kt2 -> 1+S, kt3 -> 1,
        # where S = stores of step s-1 (3 after a K step, 1 after a V step).
        # fence(s,3) implies S(s-1) done, so the staging is free after K tile 3.
        LOAD_AFTER = (2, 1, 1, 1)
        S_AFTER_K, S_AFTER_V = 3, 1

        for hh in range(i32_hpg):
            step = head0 + hh
            is_v = (step & 1) == 1
            nxt = step + 1
            nxt = (nxt < last_step).select(nxt, last_step)
            sb_regs = None
            for kt in range_constexpr(K_TILES):
                if const_expr(kt == 3):
                    tdm_ops.tensor_wait(LOAD_AFTER[kt])
                else:
                    if hh == 0:
                        tdm_ops.tensor_wait(LOAD_AFTER[kt])
                    elif is_v:  # previous step was a K step
                        tdm_ops.tensor_wait(LOAD_AFTER[kt] + S_AFTER_K)
                    else:
                        tdm_ops.tensor_wait(LOAD_AFTER[kt] + S_AFTER_V)
                rocdl.s_wait_dscnt(0)
                gpu.barrier()
                if const_expr(kt == 0):  # noqa: SIM102
                    if hh == 0:
                        rope_to_staging()
                g = hh * K_TILES + kt
                slot = g % nb
                pslot = (g + 2) % nb
                p_step = step if const_expr(kt + 2 < K_TILES) else nxt
                p_kt = (kt + 2) % K_TILES

                def _prefetch(p_step=p_step, p_kt=p_kt, pslot=pslot, kt=kt):
                    if const_expr(kt > 0):
                        issue_b(p_step, p_kt, pslot)

                if const_expr(kt == 2):
                    sb_regs = issue_weight_scale_loads(step)
                compute_ktile(kt, slot, _prefetch)
            stage_c(sb_regs, is_v)
            rocdl.s_wait_dscnt(0)
            gpu.barrier()
            issue_b(nxt, 2, (hh * K_TILES + K_TILES + 2) % nb)
            store_c(step, is_v)
        tdm_ops.tensor_wait(0)

    gx = (i32_m + (tile_m - 1)) // tile_m
    kernel_gather_kv_b_proj_hl(
        arg_k,
        arg_v,
        arg_cache,
        arg_indices,
        arg_b,
        arg_scale_a,
        arg_scale_b,
        i32_num_blocks,
        i32_m,
        f32_scale_a,
        arg_k_out_scale,
        arg_v_out_scale,
        i32_hpg,
    ).launch(
        grid=(gx, n_heads // i32_hpg, 1),
        block=(block, 1, 1),
        stream=stream,
    )


launch_gather_kv_b_proj_hl_gfx1250.compile_hints["llvm_options"] = {
    "amdgpu-expert-scheduling-mode": True,
    "amdgpu-kernarg-preload": AITER_FLYDSL_KERNARG_PRELOAD,
    "amdgpu-kernarg-preload-count": AITER_FLYDSL_KERNARG_PRELOAD_COUNT,
}


def heads_per_group(m: int, n_heads: int, num_cus: int) -> int:
    """Heads per workgroup: the largest even divisor of `n_heads` whose grid
    finishes in the fewest workgroup rounds, each round costing its heads plus
    about two heads' worth of prologue (the one-off A / RoPE gather)."""
    m_tiles = (m + _TILE_M - 1) // _TILE_M
    best = None
    for hpg in range(n_heads, 1, -1):
        if n_heads % hpg or hpg % 2:
            continue
        rounds = -(-(m_tiles * (n_heads // hpg)) // num_cus)
        cost = rounds * (hpg + 2)
        if best is None or cost < best[0]:
            best = (cost, hpg)
    return best[1]
