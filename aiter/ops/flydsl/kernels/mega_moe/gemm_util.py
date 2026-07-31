# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
"""A8W4 GEMM utilities for MegaMoE v2 stage1."""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from aiter.ops.flydsl.kernels import buffer_ops as _buffer_ops

_PACK = 2  # fp4 micro-scale pack (per-32 E8M0): pack_M = pack_N = pack_K = 2


def wait_lds_barrier(vmcnt=63):
    """Drain LDS writes and optionally older VMEM while preserving newer loads."""
    waitcnt = (vmcnt & 0xF) | ((vmcnt & 0x30) << 10) | (7 << 4)
    fx.rocdl.s_waitcnt(waitcnt)
    fx.barrier()


class TileScheduler:
    """Per-tile expert / gate-base-row resolver (flat persistent loop + XCD swizzle live in kernel body)."""

    def __init__(self, *, expert_rsrc, inter_dim, expert_offset=0):
        self._expert_rsrc = expert_rsrc
        self._inter_dim = inter_dim
        # Dispatch emits global expert IDs while W1 is rank-local.
        self._expert_offset = int(expert_offset)

    def expert_of(self, m_tile_i32):
        g = _buffer_ops.buffer_load(self._expert_rsrc, m_tile_i32, vec_width=1, dtype=fx.Int32)
        if const_expr(self._expert_offset != 0):
            return g - fx.Int32(self._expert_offset)
        return g

    def gate_base_row(self, expert_i32):
        return expert_i32 * fx.Int32(2 * self._inter_dim)


class ATileLoader:
    """Expert-major A rows (row slot == sorted row) read contiguously gmem->reg (load_regs) then reg->LDS (store)."""

    def __init__(
        self,
        *,
        x_rsrc,
        row_bytes,
        sort_block_m,
        k_step_bytes,
        total_threads,
        swizzle=False,
        x_base_addr=None,
        x_tensor=None,
        async_copy=False,
    ):
        self._x_rsrc = x_rsrc
        self._x_base_addr = x_base_addr
        self._sort_block_m = sort_block_m
        self._k_step_bytes = k_step_bytes
        self._total_threads = total_threads
        self._swizzle = swizzle
        self._row_bytes = row_bytes
        self._tx = fx.thread_idx.x
        self._wave = self._tx // 64
        self._x_tensor = x_tensor
        self._async_copy = bool(async_copy)
        if const_expr(self._async_copy):
            assert x_tensor is not None
            assert total_threads % 64 == 0
            assert (sort_block_m * 16) % total_threads == 0
            assert row_bytes % 16 == 0 and k_step_bytes % 16 == 0
            self._dma_atom = fx.make_copy_atom(
                fx.rocdl.BufferCopyLDS128b(),
                128,
            )

    def for_tile(self, tile_row_base_i32):
        """Precompute LDS and tile-local global offsets for one M tile."""
        if self._x_base_addr is not None:
            tile_base_addr = self._x_base_addr + fx.Int64(tile_row_base_i32) * fx.Int64(self._row_bytes)
            self._tile_rsrc = _buffer_ops.create_buffer_resource_from_addr(
                tile_base_addr,
                num_records_bytes=self._sort_block_m * self._row_bytes,
            )
        if const_expr(self._async_copy):
            tile_iter = fx.add_offset(
                fx.get_iter(self._x_tensor),
                fx.Int64(tile_row_base_i32) * fx.Int64(self._row_bytes),
            )
            tile_view = fx.Tensor(
                fx.make_view(
                    tile_iter,
                    fx.make_layout(
                        self._sort_block_m * self._row_bytes,
                        1,
                    ),
                )
            )
            tile_buffer = fx.rocdl.make_buffer_tensor(
                tile_view,
                max_size=False,
                num_records_bytes=self._sort_block_m * self._row_bytes,
            )
            self._tile_dma = fx.logical_divide(
                tile_buffer,
                fx.make_layout(1, 1),
            )
        chunks_per_row = self._k_step_bytes // 16
        row_stride_i32 = self._k_step_bytes // 4
        total_chunks = self._sort_block_m * chunks_per_row
        self._chunks = []
        for c in range_constexpr(0, total_chunks, self._total_threads):
            lin = fx.Int32(c) + fx.Int32(self._tx)
            row = lin // fx.Int32(chunks_per_row)
            chunk = lin % fx.Int32(chunks_per_row)
            if self._x_base_addr is not None:
                row_byte = row * fx.Int32(self._row_bytes)
            else:
                row_byte = (tile_row_base_i32 + row) * fx.Int32(self._row_bytes)
            if const_expr(self._swizzle):
                col_i32 = chunk * fx.Int32(4)
                swz = row * fx.Int32(row_stride_i32) + (col_i32 ^ ((row & fx.Int32(15)) << fx.Int32(2)))
                lds_byte = swz * fx.Int32(4)
            else:
                lds_byte = lin * fx.Int32(16)
            self._chunks.append((lds_byte, row_byte + chunk * fx.Int32(16)))

    def load_regs(self, k_step_byte_off):
        """Read this K-step's 16-B chunks gmem->reg (VMEM); only the K-step offset varies. Returns (lds_off, vec4)."""
        koff = fx.Int32(k_step_byte_off)
        regs = []
        for lds_byte, chunk_base in self._chunks:
            g_i32 = (chunk_base + koff) // fx.Int32(4)
            x_rsrc = self._tile_rsrc if self._x_base_addr is not None else self._x_rsrc
            regs.append((lds_byte, _buffer_ops.buffer_load(x_rsrc, g_i32, vec_width=4, dtype=fx.Int32)))
        return regs

    def store(self, lds_dst, regs, base_i32=0):
        """Scatter loaded chunks into LDS via ds_write (precomputed lds_byte incl. swizzle). base_i32 = ping/pong."""
        base_bytes = fx.Int32(base_i32) * fx.Int32(4)
        for lds_byte, v in regs:
            ptr = fx.recast_iter(fx.Uint8, fx.add_offset(lds_dst.ptr, fx.make_int_tuple(base_bytes + lds_byte)))
            fx.ptr_store(Vec(v).bitcast(fx.Uint8), ptr)

    def prefetch_to_lds(self, k_step_byte_off, lds_dst, base_i32=0):
        """Issue swizzled direct global-to-LDS copies with a wave-uniform LDS base."""
        koff = fx.Int32(k_step_byte_off)
        base_bytes = fx.Int32(base_i32) * fx.Int32(4)
        lds_f8 = fx.recast_iter(
            fx.Float8E4M3FN,
            lds_dst.ptr,
        )
        total_chunks = self._sort_block_m * 16
        for round_base in range_constexpr(
            0,
            total_chunks,
            self._total_threads,
        ):
            physical = fx.Int32(round_base) + fx.Int32(self._tx)
            row = physical // fx.Int32(16)
            physical_chunk = physical % fx.Int32(16)
            if const_expr(self._swizzle):
                logical_chunk = physical_chunk ^ (row & fx.Int32(15))
            else:
                logical_chunk = physical_chunk
            src_byte = row * fx.Int32(self._row_bytes) + koff + logical_chunk * fx.Int32(16)
            src = fx.slice(
                self._tile_dma,
                (None, src_byte),
            )
            wave_base = base_bytes + fx.Int32((round_base + self._wave * 64) * 16)
            dst = fx.make_view(
                fx.add_offset(lds_f8, wave_base),
                fx.make_layout(1, 1),
            )
            fx.copy(self._dma_atom, src, dst)


class AS2RLoader:
    """Load one FP8 MFMA A operand from row-major LDS."""

    def __init__(self, *, k_step_bytes, swizzle=False):
        self._k_step_bytes = k_step_bytes
        self._swizzle = swizzle
        self._lane = fx.thread_idx.x % 64

    def _load_16b(self, lds_src, i32_off):
        # add_offset+recast preserves the LDS address space.
        byte_off = i32_off * fx.Int32(4)
        ptr_off = fx.add_offset(lds_src.ptr, fx.make_int_tuple(byte_off))
        i8_iter = fx.recast_iter(fx.Uint8, ptr_off)
        v = fx.make_view(i8_iter, fx.make_layout(16, 1)).load()
        return Vec(v).bitcast(fx.Int32)

    def load_operand(self, lds_src, mi, ksub, base_i32=0):
        """A operand for m-tile mi, K=128 sub-block ksub. base_i32 = ping/pong. fp8 32-per-lane = 16@K + 16@K+64."""
        row = fx.Int32(mi * 16) + fx.Int32(self._lane % 16)
        row_i32 = row * fx.Int32(self._k_step_bytes // 4) + fx.Int32(base_i32)
        klane4 = fx.Int32(self._lane // 16) * fx.Int32(4)

        def _c(col):  # per-row XOR bank swizzle (must match ATileLoader.store)
            if const_expr(self._swizzle):
                return col ^ ((row & fx.Int32(15)) << fx.Int32(2))
            return col

        col_lo = klane4 + fx.Int32(ksub * 32)
        lo = self._load_16b(lds_src, row_i32 + _c(col_lo)).bitcast(fx.Int64)
        hi = self._load_16b(lds_src, row_i32 + _c(col_lo + fx.Int32(16))).bitcast(fx.Int64)
        return Vec.from_elements([lo[0], lo[1], hi[0], hi[1]], fx.Int64).bitcast(fx.Int32)


class BWeightLoader:
    """Per-K-step fp4 gate&up weights VMEM->reg (i32x8, 128b widened). shuffle_weight_w4 N-major layout."""

    def __init__(self, *, w_rsrc, num_acc_n, model_dim, cache_modifier=0):
        self._w_rsrc = w_rsrc
        self._num_acc_n = num_acc_n
        self._cache_modifier = int(cache_modifier)
        self._lane = fx.thread_idx.x % 64
        self._stride_nlane = 16
        self._stride_klane = 256
        self._stride_k0 = 1024
        self._stride_n0 = model_dim * 8

    def _load_pack(self, row_base_i32, ni, kstep_i32, ksub):
        lane_row = fx.Int32(self._lane % 16)
        lane_k = fx.Int32(self._lane // 16)
        n_blk = (row_base_i32 // fx.Int32(16)) + fx.Int32(ni)
        k0 = kstep_i32 * fx.Int32(2) + fx.Int32(ksub)
        byte = (
            n_blk * fx.Int32(self._stride_n0)
            + k0 * fx.Int32(self._stride_k0)
            + lane_k * fx.Int32(self._stride_klane)
            + lane_row * fx.Int32(self._stride_nlane)
        )
        i32_off = byte // fx.Int32(4)
        v = _buffer_ops.buffer_load(
            self._w_rsrc,
            i32_off,
            vec_width=4,
            dtype=fx.Int32,
            cache_modifier=self._cache_modifier,
        )
        return Vec(v)

    def load_step(self, row_base_i32, kstep_i32):
        """list[num_acc_n] of [ksub0_i32x4, ksub1_i32x4] for this K-step."""
        return [self.load_ni(row_base_i32, ni, kstep_i32) for ni in range_constexpr(self._num_acc_n)]

    def load_ni(self, row_base_i32, ni, kstep_i32):
        """One N-group's [ksub0, ksub1] fp4 packs (per-ni so the pipe can interleave one next-tile B load)."""
        return [self._load_pack(row_base_i32, ni, kstep_i32, ks) for ks in range_constexpr(_PACK)]


class BScaleLoader:
    """Preshuffled per-1x32 E8M0 B scales; one packed i32 (4 e8m0) per K-step per 32-row pack-group."""

    def __init__(self, *, scale_rsrc, num_acc_n, model_dim):
        assert num_acc_n % _PACK == 0
        self._rsrc = scale_rsrc
        self._n_groups = num_acc_n // _PACK
        self._row_stride = (model_dim // 256) * 64
        lane = fx.thread_idx.x % 64
        self._lane_off = lane

    def load_step(self, base_row_i32, kstep_i32):
        base_group = base_row_i32 // fx.Int32(32)
        kterm = kstep_i32 * fx.Int32(64)
        lane = fx.Int32(self._lane_off)
        out = []
        for g in range_constexpr(self._n_groups):
            off = (base_group + fx.Int32(g)) * fx.Int32(self._row_stride) + kterm + lane
            out.append(_buffer_ops.buffer_load(self._rsrc, off, vec_width=1, dtype=fx.Int32))
        return out


class AScaleLoader:
    """Per-1x32 E8M0 A scales STAGED to LDS once per tile (stage), read via ds_read in K-loop (kills VMEM flood)."""

    def __init__(self, *, scale_rsrc, m_repeat, model_dim, sort_block_m, total_threads):
        self._rsrc = scale_rsrc
        self._n_scale = model_dim // 32
        self._lane = fx.thread_idx.x % 64
        self._n_groups = m_repeat // _PACK
        self._sort_block_m = sort_block_m
        self._total_threads = total_threads
        self._tx = fx.thread_idx.x

    def stage(self, lds_ascale, tile_row_base_i32):
        """Coalesced gmem->LDS copy of this tile's e8m0 A-scale block [sort_block_m, n_scale]. Call before K-loop."""
        total = self._sort_block_m * self._n_scale
        assert total % 16 == 0, "A-scale tile must contain whole 16-byte copy chunks"
        base = tile_row_base_i32 * fx.Int32(self._n_scale)
        n16 = total // 16

        @flyc.jit
        def copy_chunk(lin: fx.Int32):
            if lin < fx.Int32(n16):
                g_i32 = (base + lin * fx.Int32(16)) // fx.Int32(4)
                v = _buffer_ops.buffer_load(
                    self._rsrc,
                    g_i32,
                    vec_width=4,
                    dtype=fx.Int32,
                )
                ptr = fx.recast_iter(
                    fx.Uint8,
                    fx.add_offset(
                        lds_ascale.ptr,
                        fx.make_int_tuple(lin * fx.Int32(16)),
                    ),
                )
                fx.ptr_store(Vec(v).bitcast(fx.Uint8), ptr)

        for c in range_constexpr(0, n16, self._total_threads):
            lin = fx.Int32(c) + fx.Int32(self._tx)
            # M32 has 448 chunks, which is not divisible by a 256/512-thread CTA.
            copy_chunk(lin)

    def load_step(self, lds_ascale, kstep_i32):
        """One packed i32 per pack-group, read from the LDS-staged A-scale (ds_read)."""
        lane_row = fx.Int32(self._lane % 16)
        col0 = kstep_i32 * fx.Int32(8) + fx.Int32(self._lane // 16)  # e8m0 col = kstep*8 + ksub*4 + KLane
        out = []
        for g in range_constexpr(self._n_groups):
            r0 = fx.Int32(g * 32) + lane_row
            r1 = r0 + fx.Int32(16)
            b = []
            for ksub in range_constexpr(_PACK):
                for rr in (r0, r1):
                    b.append(self._read_scale_lds(lds_ascale, rr, col0 + fx.Int32(ksub * 4)))
            packed = b[0] | (b[1] << fx.Int32(8)) | (b[2] << fx.Int32(16)) | (b[3] << fx.Int32(24))
            out.append(packed)
        return out

    def _read_scale_lds(self, lds_ascale, row_i32, col_i32):
        off = row_i32 * fx.Int32(self._n_scale) + col_i32
        ptr = fx.recast_iter(fx.Uint8, fx.add_offset(lds_ascale.ptr, fx.make_int_tuple(off)))
        v = fx.make_view(ptr, fx.make_layout(1, 1)).load()
        return Vec(v)[0].to(fx.Int32)


class MfmaScaleGU:
    """Gate/up scaled-MFMA atoms for FP8 A x FP4 B."""

    def __init__(self, *, m_repeat, num_acc_n):
        self._m_repeat = m_repeat
        self._num_acc_n = num_acc_n
        self._atoms = {
            (osa, osb): fx.make_mma_atom(
                fx.rocdl.cdna4.MFMA_Scale(16, 16, 128, fx.Float8E4M3FN, fx.Float4E2M1FN, opsel_a=osa, opsel_b=osb)
            )
            for osa in range(4)
            for osb in range(4)
        }
        self.zero_value = Vec.filled(4, 0.0, fx.Float32)

    def idx(self, mi, ni):
        return mi * self._num_acc_n + ni

    def _mfma(self, a_op, b_op, acc, sa_v, sb_v, ksub, ia, jb):
        opsel_a = ksub * _PACK + ia
        opsel_b = ksub * _PACK + jb
        a_frag = fx.make_rmem_tensor(8, fx.Int32)
        b_frag = fx.make_rmem_tensor(4, fx.Int32)
        c_frag = fx.make_rmem_tensor(4, fx.Float32)
        a_frag.store(Vec(a_op))
        b_frag.store(Vec(b_op))
        c_frag.store(Vec(acc))
        fx.gemm(self._atoms[(opsel_a, opsel_b)], c_frag, a_frag, b_frag, c_frag, scale_a=sa_v, scale_b=sb_v)
        return Vec(c_frag.load())

    def call(self, a_load, b, acc, sa, sb):
        """Non-pipelined lazy-A K-step: a_load(mi,ksub) on demand (one A operand live -> low VGPR, wins small bs)."""
        for ksub in range_constexpr(_PACK):
            for mi in range_constexpr(self._m_repeat):
                a_op = a_load(mi, ksub)
                sa_v = sa[mi // _PACK]
                ia = mi % _PACK
                for ni in range_constexpr(self._num_acc_n):
                    aidx = self.idx(mi, ni)
                    acc[aidx] = self._mfma(
                        a_op,
                        b[ni][ksub],
                        acc[aidx],
                        sa_v,
                        sb[ni // _PACK],
                        ksub,
                        ia,
                        ni % _PACK,
                    )
        return acc

    def call_pipe(self, a_load, b_prev, acc, sa, sb, load_next):
        """B-major pipe: MFMA b_prev while load_next(ni) issues next-tile B[ni] before its group."""
        a_ops = []
        for mi in range_constexpr(self._m_repeat):
            a_ops.append([a_load(mi, ks) for ks in range_constexpr(_PACK)])
        b_next = []
        for ni in range_constexpr(self._num_acc_n):
            b_next.append(load_next(ni))
            jb = ni % _PACK
            sb_v = sb[ni // _PACK]
            for ksub in range_constexpr(_PACK):
                for mi in range_constexpr(self._m_repeat):
                    aidx = self.idx(mi, ni)
                    acc[aidx] = self._mfma(
                        a_ops[mi][ksub],
                        b_prev[ni][ksub],
                        acc[aidx],
                        sa[mi // _PACK],
                        sb_v,
                        ksub,
                        mi % _PACK,
                        jb,
                    )
        return acc, b_next

    def call_pipe_am(self, a_load, b_prev, acc, sa, sb, load_next):
        """A-major pipe: lazy-A per (mi,ksub) overlaps next A ds_read with MFMA; next-B loads spread over groups."""
        nn = self._num_acc_n
        b_next = [None] * nn
        ngrp = self._m_repeat * _PACK
        g = 0
        nb = 0
        for mi in range_constexpr(self._m_repeat):
            ia = mi % _PACK
            sa_v = sa[mi // _PACK]
            for ksub in range_constexpr(_PACK):
                a_op = a_load(mi, ksub)
                tgt = ((g + 1) * nn) // ngrp
                while nb < tgt:
                    b_next[nb] = load_next(nb)
                    nb += 1
                for ni in range_constexpr(nn):
                    aidx = self.idx(mi, ni)
                    acc[aidx] = self._mfma(
                        a_op,
                        b_prev[ni][ksub],
                        acc[aidx],
                        sa_v,
                        sb[ni // _PACK],
                        ksub,
                        ia,
                        ni % _PACK,
                    )
                g += 1
        while nb < nn:
            b_next[nb] = load_next(nb)
            nb += 1
        return acc, b_next

    def call_pipe_am_final(self, a_load, b_prev, acc, sa, sb):
        """Consume the final prefetched B step without redundant loads."""
        for mi in range_constexpr(self._m_repeat):
            ia = mi % _PACK
            sa_v = sa[mi // _PACK]
            for ksub in range_constexpr(_PACK):
                a_op = a_load(mi, ksub)
                for ni in range_constexpr(self._num_acc_n):
                    aidx = self.idx(mi, ni)
                    acc[aidx] = self._mfma(
                        a_op,
                        b_prev[ni][ksub],
                        acc[aidx],
                        sa_v,
                        sb[ni // _PACK],
                        ksub,
                        ia,
                        ni % _PACK,
                    )
        return acc


class SiluQuantEpilogue:
    """silu(gate)*up -> fp8 + per-32 E8M0 out-scale (aiter/CK swizzled), via inline .ptr cshuffle."""

    # fmt: off
    def __init__(self, *, out_rsrc, out_scale_rsrc, sorted_rsrc, tokens, inter_dim, m_repeat, num_acc_n,
        sort_block_m, tile_n, num_waves, lds_out, always_valid=False, out_base_addr=None):
    # fmt: on
        self._out_rsrc = out_rsrc
        self._out_scale_rsrc = out_scale_rsrc
        self._sorted_rsrc = sorted_rsrc
        self._tokens = tokens
        self._inter_dim = inter_dim
        self._m_repeat = m_repeat
        self._num_acc_n = num_acc_n
        self._sort_block_m = sort_block_m
        self._tile_n = tile_n
        self._num_waves = num_waves
        self._lds_out = lds_out
        self._always_valid = always_valid
        self._out_base_addr = out_base_addr
        self._lane = fx.thread_idx.x % 64
        self._sorted_scale_cols_i32 = (inter_dim // 32 + 7) // 8 * 8

    def _silu(self, g):
        emu = (g * fx.Float32(-1.4426950408889634)).exp2()
        return g * (fx.Float32(1.0) / (fx.Float32(1.0) + emu))

    def _combine(self, acc):
        gui_n = self._num_acc_n // _PACK
        out = []
        for mi in range_constexpr(self._m_repeat):
            for ni in range_constexpr(gui_n):
                g_idx = mi * self._num_acc_n + ni * _PACK
                out.append(self._silu_mul(acc[g_idx], acc[g_idx + 1]))
        return out

    def _silu_mul(self, gate_v4, up_v4):
        gv = Vec(gate_v4)
        uv = Vec(up_v4)
        elems = [self._silu(gv[i]) * uv[i] for i in range_constexpr(4)]
        return Vec.from_elements(elems, fx.Float32)

    def store(self, acc, tile_i32, tile_row_base_i32, n_tile_base_i32):
        combined = self._combine(acc)
        n_per = len(combined) // self._m_repeat
        if self._out_base_addr is not None:
            out_rsrc = _buffer_ops.create_buffer_resource_from_addr(
                self._out_base_addr
                + fx.Int64(tile_i32)
                * fx.Int64(self._sort_block_m * self._inter_dim),
                num_records_bytes=self._sort_block_m * self._inter_dim,
            )
        else:
            out_rsrc = self._out_rsrc

        tx = fx.thread_idx.x
        wave = tx // fx.Int32(64)
        lane = tx % fx.Int32(64)
        l16 = lane % fx.Int32(16)
        ld4 = (lane // fx.Int32(16)) * fx.Int32(4)
        cs_tile_n = self._tile_n // 2
        nwo = cs_tile_n // self._num_waves
        cbase = wave * fx.Int32(nwo)
        cptr = self._lds_out.ptr

        for mi in range_constexpr(self._m_repeat):
            for nj in range_constexpr(n_per):
                v4 = Vec(combined[mi * n_per + nj])
                col = cbase + fx.Int32(nj * 16) + l16
                for ii in range_constexpr(4):
                    row = fx.Int32(mi * 16) + ld4 + fx.Int32(ii)
                    idx = row * fx.Int32(cs_tile_n) + col
                    ptr = fx.add_offset(cptr, fx.make_int_tuple(idx))
                    fx.ptr_store(Vec.from_elements([v4[ii]], fx.Float32), ptr)
        gpu.barrier()

        c64 = fx.Int32(64)
        n32 = fx.Int32(self._sorted_scale_cols_i32 * 32)
        NLANE = 32
        EVEC = 2
        rows_per_iter = self._num_waves * 64 // NLANE
        mlane = tx // fx.Int32(NLANE)
        nlane = tx % fx.Int32(NLANE)
        m_reps = self._sort_block_m // rows_per_iter
        n_reps = cs_tile_n // (NLANE * EVEC)
        out_tile_base = n_tile_base_i32 // fx.Int32(2) - cbase

        for mr in range_constexpr(m_reps):
            row = fx.Int32(mr * rows_per_iter) + mlane
            slot = tile_row_base_i32 + row
            row_g = tile_i32 * fx.Int32(self._sort_block_m) + row
            if const_expr(self._always_valid):
                valid = fx.Boolean(True)
                out_row_base = (
                    row * fx.Int32(self._inter_dim)
                    if self._out_base_addr is not None
                    else row_g * fx.Int32(self._inter_dim)
                )
            else:
                tok = _buffer_ops.buffer_load(self._sorted_rsrc, slot, vec_width=1, dtype=fx.Int32)
                valid = tok < fx.Int32(self._tokens)
                out_row_base = slot * fx.Int32(self._inter_dim)
            for nr in range_constexpr(n_reps):
                col0 = fx.Int32(nr * NLANE * EVEC) + nlane * fx.Int32(EVEC)
                idx = row * fx.Int32(cs_tile_n) + col0
                f32_iter = fx.recast_iter(fx.Float32, fx.add_offset(cptr, fx.make_int_tuple(idx)))
                frag = fx.make_view(f32_iter, fx.make_layout(EVEC, 1)).load()
                v0 = frag[0]
                v1 = frag[1]
                a0 = v0.maximumf(fx.Float32(0.0) - v0)
                a1 = v1.maximumf(fx.Float32(0.0) - v1)
                m = a0.maximumf(a1)
                for off in (1, 2, 4, 8):
                    m = m.maximumf(m.shuffle_xor(fx.Int32(off), c64))
                max_rounded = (m.bitcast(fx.Int32) + fx.Int32(0x400000)) & fx.Int32(0xFF800000)
                _e = (max_rounded >> fx.Int32(23)) - fx.Int32(8)
                e8m0_v = (_e > fx.Int32(0)).select(_e, fx.Int32(0))
                quant_scale = ((fx.Int32(254) - e8m0_v) << fx.Int32(23)).bitcast(fx.Float32)
                gcol = out_tile_base + col0

                scaled0 = v0 * quant_scale
                scaled1 = v1 * quant_scale
                packed = rocdl.cvt_pk_fp8_f32(T.i32, scaled0, scaled1, fx.Int32(0), 0)
                short_raw = fx.Int32(packed).to(fx.Int16)
                out_byte = out_row_base + gcol
                out_byte = valid.select(out_byte, fx.Int32(0x40000000))
                _buffer_ops.buffer_store(short_raw, out_rsrc, out_byte, offset_is_bytes=True)

                col_s = gcol >> fx.Int32(5)
                is_writer = (gcol & fx.Int32(31)) == fx.Int32(0)
                d0 = row_g >> fx.Int32(5)
                d1 = (row_g >> fx.Int32(4)) & fx.Int32(1)
                d2 = row_g & fx.Int32(15)
                d3 = col_s >> fx.Int32(3)
                d4 = (col_s >> fx.Int32(2)) & fx.Int32(1)
                d5 = col_s & fx.Int32(3)
                byte_off = d0 * n32 + d3 * fx.Int32(256) + d5 * fx.Int32(64) + d2 * fx.Int32(4) + d4 * fx.Int32(2) + d1
                byte_off = is_writer.select(byte_off, fx.Int32(0x40000000))
                e8m0_i8 = e8m0_v.to(fx.Int8)
                _buffer_ops.buffer_store(e8m0_i8, self._out_scale_rsrc, byte_off, offset_is_bytes=True)
        wait_lds_barrier()
