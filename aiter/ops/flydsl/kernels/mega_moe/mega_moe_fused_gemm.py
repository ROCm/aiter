# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
"""GEMM1/GEMM2 tiles, buffer helpers and launch configs shared by the MegaMoEV2 small- and large-input kernels."""

from bisect import bisect_left
from dataclasses import dataclass
from functools import cache
from typing import NamedTuple

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import (
    Float4E2M1FN,
    Float8E4M3FN,
    Float32,
    Int32,
    T,
)
from flydsl.expr.typing import Vector as Vec
from flydsl.expr.typing import as_ir_value as _raw

from ..mxfp4_gemm_common import _lds_swizzle_mask as lds_swizzle_mask
from ..mxfp4_gemm_common import (
    flat_buffer_view,
    global_typed_ptr,
    kBS_stride_k0_dw,
    kStages,
    lds_dma_atom_128,
    lds_dma_dst,
    lds_swizzle_mask_f8,
    lds_vec_load,
)
from ..tensor_shim import _fx_value_elem, buf_copy_load, buf_copy_store, ptr_buf_tensor

TOKEN_BUCKETS = (
    1,
    4,
    8,
    16,
    32,
    64,
    128,
    256,
    512,
    1024,
    2048,
    4096,
    8192,
    16384,
    32768,
)
# The small-input single kernel keeps fixed slots up to this MTPR (past it the main path is faster).
SMALL_M_FIXED_SLOT_MAX_MTPR = 512
# The large-input path runs on an instance of this MTPR only.
LARGE_M_MTPR = 8192
REFERENCE_EXPERTS_PER_RANK = 48
EXPERT_CONFIG_GRANULARITY = 64


@dataclass(frozen=True, slots=True)
class Stage1Config:
    sort_block_m: int
    tile_n: int
    num_waves: int
    grid_mult: int
    num_dispatch_cu: int
    mfma_amajor: bool
    async_a_copy: bool
    use_tile_resource: bool
    b_nt: int
    waves_per_eu_hint: int = 2
    tile_k: int = 256
    pipe_weights: bool = True
    swizzle_a: bool = True
    work_shards: int = 8
    payload_chunk_rows: int = 0


@dataclass(frozen=True, slots=True)
class Stage2Config:
    block_m: int
    block_n: int
    persist: bool
    persist_cu: int
    use_nt: bool
    persist_strided: bool = False
    skew_cu: int = 0
    block_k: int = 256
    b_hoist: bool = True
    ascale_prefetch: bool = True
    spatial_partition: int = 402
    bf16_lds: bool = False


@dataclass(frozen=True, slots=True)
class MegaMoEConfig:
    stage1: Stage1Config
    stage2: Stage2Config
    p2p_quant: str

    def __post_init__(self):
        sbm = self.stage1.sort_block_m
        bm = self.stage2.block_m
        if bm > sbm or sbm % bm:
            raise ValueError(
                f"Stage2 block_m={bm} must divide Stage1 sort_block_m={sbm}"
            )
        if self.p2p_quant not in ("none", "fp8_blockwise_1x32"):
            raise ValueError(f"unsupported p2p_quant={self.p2p_quant!r}")
        if self.p2p_quant != "none" and self.stage2.bf16_lds:
            raise ValueError("FP8 P2P requires Stage2 bf16_lds=False")


def nearest_token_bucket(tokens: int) -> int:
    if tokens <= 0:
        raise ValueError(f"tokens must be positive, got {tokens}")
    index = bisect_left(TOKEN_BUCKETS, tokens)
    if index == 0:
        return TOKEN_BUCKETS[0]
    if index == len(TOKEN_BUCKETS):
        return TOKEN_BUCKETS[-1]
    lower, upper = TOKEN_BUCKETS[index - 1], TOKEN_BUCKETS[index]
    return upper if upper - tokens <= tokens - lower else lower


def expert_config_class(experts_per_rank: int) -> int:
    return (
        (experts_per_rank + EXPERT_CONFIG_GRANULARITY - 1)
        // EXPERT_CONFIG_GRANULARITY
        * EXPERT_CONFIG_GRANULARITY
    )


def _scale_dispatch_cu(dispatch_cu: int, experts_per_rank: int) -> int:
    expert_waves = (experts_per_rank + 63) // 64
    return min(224, dispatch_cu * expert_waves)


def _fixed_dispatch_cu(bucket: int) -> int:
    if bucket <= 1:
        return 64
    if bucket <= 8:
        return 128
    if bucket <= 16:
        return 96
    if bucket <= 32:
        return 128
    return min(224, 16 * (bucket.bit_length() + 7))


def _select_fixed_stage1(bucket: int, experts_per_rank: int) -> Stage1Config:
    grid_mult = max(1, bucket // 4) if bucket <= 16 else 3
    return Stage1Config(
        sort_block_m=32,
        tile_n=256 if bucket <= 8 else 128,
        num_waves=4,
        grid_mult=grid_mult,
        num_dispatch_cu=_scale_dispatch_cu(
            _fixed_dispatch_cu(bucket), experts_per_rank
        ),
        mfma_amajor=False,
        async_a_copy=False,
        use_tile_resource=bucket <= 16,
        b_nt=0 if bucket == 1 else 3,
        waves_per_eu_hint=1 if bucket == 16 else 2,
    )


def _select_fixed_stage2(bucket: int, model_dim: int) -> Stage2Config:
    block_n = 256 if bucket in (1, 4, 64) or bucket >= 1024 else 128
    if model_dim < 4096:
        block_n = 128
    persist = bucket >= 128
    return Stage2Config(
        block_m=32,
        block_n=block_n,
        persist=persist,
        persist_cu=128 if bucket == 256 else 240 if persist else 0,
        use_nt=bucket <= 128,
        persist_strided=512 <= bucket <= 1024,
    )


def _large_config(
    experts_per_rank: int, model_dim: int, inter_dim: int
) -> MegaMoEConfig:
    stage1 = Stage1Config(
        sort_block_m=128,
        tile_n=512 if inter_dim >= 2048 else 256,
        num_waves=8,
        grid_mult=1,
        num_dispatch_cu=_scale_dispatch_cu(96, experts_per_rank),
        mfma_amajor=True,
        async_a_copy=True,
        use_tile_resource=True,
        b_nt=0,
        work_shards=4,
        payload_chunk_rows=384,
    )
    stage2 = Stage2Config(
        block_m=64,
        block_n=128 if model_dim < 4096 else 256,
        persist=True,
        persist_cu=240,
        use_nt=False,
        skew_cu=96,
    )
    return MegaMoEConfig(stage1=stage1, stage2=stage2, p2p_quant="fp8_blockwise_1x32")


@cache
def _select_config(
    bucket: int, mtpr: int, experts_per_rank: int, model_dim: int, inter_dim: int
) -> MegaMoEConfig:
    if mtpr == LARGE_M_MTPR:
        return _large_config(experts_per_rank, model_dim, inter_dim)
    stage1 = _select_fixed_stage1(bucket, experts_per_rank)
    stage2 = _select_fixed_stage2(bucket, model_dim)
    return MegaMoEConfig(stage1=stage1, stage2=stage2, p2p_quant="none")


def select_mega_moe_config(
    tokens: int,
    mtpr: int,
    *,
    experts_per_rank: int = REFERENCE_EXPERTS_PER_RANK,
    model_dim: int = 7168,
    inter_dim: int = 3072,
) -> MegaMoEConfig:
    """Small-input (fixed-slot, MTPR <= 512) or large-input (MTPR 8192) config for ``tokens`` per rank."""
    if mtpr <= 0 or mtpr & (mtpr - 1):
        raise ValueError(f"mtpr={mtpr} must be a positive power of two")
    if mtpr > SMALL_M_FIXED_SLOT_MAX_MTPR and mtpr != LARGE_M_MTPR:
        raise ValueError(f"no small- or large-input config for mtpr={mtpr}")
    if tokens > mtpr:
        raise ValueError(f"tokens={tokens} exceeds mtpr={mtpr}")
    if experts_per_rank <= 0 or experts_per_rank > 64:
        raise ValueError(f"experts_per_rank must be in [1, 64], got {experts_per_rank}")
    if model_dim <= 0 or inter_dim <= 0:
        raise ValueError(f"invalid model shape {model_dim}x{inter_dim}")
    return _select_config(
        nearest_token_bucket(tokens),
        mtpr,
        expert_config_class(experts_per_rank),
        model_dim,
        inter_dim,
    )


_PACK = 2  # fp4 micro-scale pack (per-32 E8M0): pack_M = pack_N = pack_K = 2


class AddrBuffer(NamedTuple):
    """A raw device address and an optional hardware OOB bound in bytes."""

    addr: object
    num_records_bytes: object = None


def _addr_view(buf, elem, width):
    # unit_stride=1: a wide access may start at any element offset.
    return ptr_buf_tensor(
        buf.addr,
        elem,
        unit_elems=width,
        unit_stride=1,
        num_records_bytes=buf.num_records_bytes,
    )


def load_at(buf, index, elem=fx.Int32, width=1, cache_modifier=0, mask=None):
    """Load ``width`` x ``elem`` at element ``index`` of ``buf``.

    A false ``mask`` moves the access past the buffer bound.
    """
    if mask is not None:
        index = fx.Boolean(mask).select(
            index, fx.Int32(0x7FFFFFFF // (elem.width // 8))
        )
    return buf_copy_load(
        _addr_view(buf, elem, width), index, elem, width, cache_modifier
    )


def store_at(value, buf, index, cache_modifier=0, offset_is_bytes=False):
    """Store ``value`` (scalar or vector) at element ``index`` of ``buf``."""
    elem, width = _fx_value_elem(value)
    if offset_is_bytes:
        index = fx.Int32(index) // fx.Int32(elem.width // 8)
    buf_copy_store(
        _addr_view(buf, elem, width), index, value, elem, width, cache_modifier
    )


def _make_buffer(tensor, elem_ty, width=1, *, max_size=True, num_records_bytes=None):
    alignment = max(1, elem_ty.width * width // 8)
    ptr_ty = fx.PointerType.get(elem_ty.ir_type, fx.AddressSpace.Global, alignment)
    base = fx.inttoptr(ptr_ty, fx.Int64(fx.ptrtoint(fx.get_iter(tensor))))
    view = fx.Tensor(fx.make_view(base, fx.make_layout((width, 1), (1, width))))
    return fx.rocdl.make_buffer_tensor(
        view, max_size=max_size, num_records_bytes=num_records_bytes
    )


def _make_buffer_from_addr(addr, elem_ty, width=1, *, num_records_bytes=None):
    alignment = max(1, elem_ty.width * width // 8)
    ptr_ty = fx.PointerType.get(elem_ty.ir_type, fx.AddressSpace.Global, alignment)
    base = fx.inttoptr(ptr_ty, fx.Int64(addr))
    view = fx.Tensor(fx.make_view(base, fx.make_layout((width, 1), (1, width))))
    return fx.rocdl.make_buffer_tensor(view, num_records_bytes=num_records_bytes)


def _buffer_load(buffer, group_index, elem_ty, width=1, cache_modifier=0):
    atom = fx.make_copy_atom(
        fx.rocdl.BufferCopy(elem_ty.width * width, cache_modifier), elem_ty
    )
    fragment = fx.make_rmem_tensor(width, elem_ty)
    fx.copy(atom, fx.slice(buffer, (None, group_index)), fragment)
    value = Vec(fragment.load())
    return value[0] if width == 1 else value


def _buffer_store(buffer, group_index, value, elem_ty, width=1, cache_modifier=0):
    atom = fx.make_copy_atom(
        fx.rocdl.BufferCopy(elem_ty.width * width, cache_modifier), elem_ty
    )
    fragment = fx.make_rmem_tensor(width, elem_ty)
    fragment.store(Vec.from_elements([value], elem_ty) if width == 1 else Vec(value))
    fx.copy(atom, fragment, fx.slice(buffer, (None, group_index)))


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
        g = _buffer_load(self._expert_rsrc, m_tile_i32, fx.Int32)
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
        row_bytes,
        sort_block_m,
        k_step_bytes,
        total_threads,
        swizzle=False,
        x_tensor=None,
        async_copy=False,
    ):
        self._sort_block_m = sort_block_m
        self._k_step_bytes = k_step_bytes
        self._total_threads = total_threads
        self._swizzle = swizzle
        self._row_bytes = row_bytes
        self._tx = fx.thread_idx.x
        self._wave = self._tx // 64
        self._x_tensor = x_tensor
        self._async_copy = bool(async_copy)
        assert x_tensor is not None
        if const_expr(self._async_copy):
            assert total_threads % 64 == 0
            assert (sort_block_m * 16) % total_threads == 0
            assert row_bytes % 16 == 0 and k_step_bytes % 16 == 0
            self._dma_atom = fx.make_copy_atom(
                fx.rocdl.BufferCopyLDS128b(),
                128,
            )

    def for_tile(self, tile_row_base_i32, valid_rows_i32=None):
        """Precompute LDS and tile-local global offsets for one M tile.

        valid_rows_i32 clamps the buffer's num_records: padding rows read back as 0 without touching memory.
        """
        tile_iter = fx.add_offset(
            fx.get_iter(self._x_tensor),
            fx.Int64(tile_row_base_i32) * fx.Int64(self._row_bytes),
        )
        tile_view = fx.Tensor(
            fx.make_view(
                tile_iter, fx.make_layout(self._sort_block_m * self._row_bytes, 1)
            )
        )
        if valid_rows_i32 is None:
            nrec = fx.Int32(self._sort_block_m * self._row_bytes)
        else:
            nrec = fx.Int32(valid_rows_i32) * fx.Int32(self._row_bytes)
        self._tile_rsrc = _make_buffer(
            tile_view,
            fx.Int32,
            4,
            max_size=False,
            num_records_bytes=nrec,
        )
        if const_expr(self._async_copy):
            tile_buffer = fx.rocdl.make_buffer_tensor(
                tile_view,
                max_size=False,
                num_records_bytes=nrec,
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
            row_byte = row * fx.Int32(self._row_bytes)
            if const_expr(self._swizzle):
                col_i32 = chunk * fx.Int32(4)
                swz = row * fx.Int32(row_stride_i32) + (
                    col_i32 ^ ((row & fx.Int32(15)) << fx.Int32(2))
                )
                lds_byte = swz * fx.Int32(4)
            else:
                lds_byte = lin * fx.Int32(16)
            self._chunks.append((lds_byte, row_byte + chunk * fx.Int32(16)))

    def load_regs(self, k_step_byte_off):
        """Read this K-step's 16-B chunks gmem->reg (VMEM); only the K-step offset varies. Returns (lds_off, vec4)."""
        koff = fx.Int32(k_step_byte_off)
        regs = []
        for lds_byte, chunk_base in self._chunks:
            group = (chunk_base + koff) // fx.Int32(16)
            regs.append((lds_byte, _buffer_load(self._tile_rsrc, group, fx.Int32, 4)))
        return regs

    def store(self, lds_dst, regs, base_i32=0):
        """Scatter loaded chunks into LDS via ds_write (precomputed lds_byte incl. swizzle). base_i32 = ping/pong."""
        base_bytes = fx.Int32(base_i32) * fx.Int32(4)
        for lds_byte, v in regs:
            dst = fx.make_view(
                fx.add_offset(
                    fx.recast_iter(fx.Int32, lds_dst.ptr),
                    (base_bytes + lds_byte) // fx.Int32(4),
                ),
                fx.make_layout(4, 1),
            )
            fragment = fx.make_rmem_tensor(4, fx.Int32)
            fragment.store(Vec(v))
            fx.copy(fx.make_copy_atom(fx.UniversalCopy128b(), fx.Int32), fragment, dst)

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
            src_byte = (
                row * fx.Int32(self._row_bytes) + koff + logical_chunk * fx.Int32(16)
            )
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
        hi = self._load_16b(lds_src, row_i32 + _c(col_lo + fx.Int32(16))).bitcast(
            fx.Int64
        )
        return Vec.from_elements([lo[0], lo[1], hi[0], hi[1]], fx.Int64).bitcast(
            fx.Int32
        )


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
        return _buffer_load(
            self._w_rsrc,
            byte // fx.Int32(16),
            fx.Int32,
            4,
            self._cache_modifier,
        )

    def load_step(self, row_base_i32, kstep_i32):
        """list[num_acc_n] of [ksub0_i32x4, ksub1_i32x4] for this K-step."""
        return [
            self.load_ni(row_base_i32, ni, kstep_i32)
            for ni in range_constexpr(self._num_acc_n)
        ]

    def load_ni(self, row_base_i32, ni, kstep_i32):
        """One N-group's [ksub0, ksub1] fp4 packs (per-ni so the pipe can interleave one next-tile B load)."""
        return [
            self._load_pack(row_base_i32, ni, kstep_i32, ks)
            for ks in range_constexpr(_PACK)
        ]


class BScaleLoader:
    """Preshuffled per-1x32 E8M0 B scales; one packed i32 (4 e8m0) per K-step per 32-row pack-group."""

    def __init__(self, *, scale_rsrc, num_acc_n, model_dim):
        assert num_acc_n % _PACK == 0
        self._rsrc = scale_rsrc
        self._n_groups = num_acc_n // _PACK
        self._row_stride = (model_dim // 256) * 64
        lane = fx.thread_idx.x % 64
        self._lane_off = lane

    def load_step(self, base_row_i32, kstep_i32, cache_modifier=0):
        base_group = base_row_i32 // fx.Int32(32)
        kterm = kstep_i32 * fx.Int32(64)
        lane = fx.Int32(self._lane_off)
        out = []
        for g in range_constexpr(self._n_groups):
            off = (base_group + fx.Int32(g)) * fx.Int32(self._row_stride) + kterm + lane
            out.append(
                _buffer_load(self._rsrc, off, fx.Int32, cache_modifier=cache_modifier)
            )
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
                v = _buffer_load(
                    self._rsrc,
                    (base + lin * fx.Int32(16)) // fx.Int32(16),
                    fx.Int32,
                    4,
                )
                dst = fx.make_view(
                    fx.add_offset(
                        fx.recast_iter(fx.Int32, lds_ascale.ptr), lin * fx.Int32(4)
                    ),
                    fx.make_layout(4, 1),
                )
                fragment = fx.make_rmem_tensor(4, fx.Int32)
                fragment.store(v)
                fx.copy(
                    fx.make_copy_atom(fx.UniversalCopy128b(), fx.Int32), fragment, dst
                )

        for c in range_constexpr(0, n16, self._total_threads):
            lin = fx.Int32(c) + fx.Int32(self._tx)
            # M32 has 448 chunks, which is not divisible by a 256/512-thread CTA.
            copy_chunk(lin)

    def load_step(self, lds_ascale, kstep_i32):
        """One packed i32 per pack-group, read from the LDS-staged A-scale (ds_read)."""
        lane_row = fx.Int32(self._lane % 16)
        col0 = kstep_i32 * fx.Int32(8) + fx.Int32(
            self._lane // 16
        )  # e8m0 col = kstep*8 + ksub*4 + KLane
        out = []
        for g in range_constexpr(self._n_groups):
            r0 = fx.Int32(g * 32) + lane_row
            r1 = r0 + fx.Int32(16)
            b = []
            for ksub in range_constexpr(_PACK):
                for rr in (r0, r1):
                    b.append(
                        self._read_scale_lds(lds_ascale, rr, col0 + fx.Int32(ksub * 4))
                    )
            packed = (
                b[0]
                | (b[1] << fx.Int32(8))
                | (b[2] << fx.Int32(16))
                | (b[3] << fx.Int32(24))
            )
            out.append(packed)
        return out

    def _read_scale_lds(self, lds_ascale, row_i32, col_i32):
        off = row_i32 * fx.Int32(self._n_scale) + col_i32
        ptr = fx.recast_iter(
            fx.Uint8, fx.add_offset(lds_ascale.ptr, fx.make_int_tuple(off))
        )
        v = fx.make_view(ptr, fx.make_layout(1, 1)).load()
        return Vec(v, dtype=fx.Uint8)[0].to(fx.Int32)


class MfmaScaleGU:
    """Gate/up scaled-MFMA atoms for FP8 A x FP4 B."""

    def __init__(self, *, m_repeat, num_acc_n):
        self._m_repeat = m_repeat
        self._num_acc_n = num_acc_n
        self._atoms = {
            (osa, osb): fx.make_mma_atom(
                fx.rocdl.cdna4.MFMA_Scale(
                    16,
                    16,
                    128,
                    fx.Float8E4M3FN,
                    fx.Float4E2M1FN,
                    opsel_a=osa,
                    opsel_b=osb,
                )
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
        fx.gemm(
            self._atoms[(opsel_a, opsel_b)],
            c_frag,
            a_frag,
            b_frag,
            c_frag,
            scale_a=sa_v,
            scale_b=sb_v,
        )
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
    """SwiGLU followed by FP8 quantization and per-32 E8M0 output scales."""

    # fmt: off
    def __init__(self, *, out_rsrc, out_scale_rsrc, sorted_rsrc, tokens, inter_dim, m_repeat, num_acc_n,
        sort_block_m, tile_n, num_waves, lds_out, swiglu_limit=0.0, always_valid=False, out_tensor=None,
        out_cache_modifier=0):
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
        self._swiglu_limit = float(swiglu_limit)
        self._always_valid = always_valid
        self._out_tensor = out_tensor
        # Default store cache policy; 17 (sc0|sc1) makes the outputs visible to other CUs' consumers.
        self._out_cache_modifier = int(out_cache_modifier)
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
        if self._swiglu_limit <= 0:
            elems = [self._silu(gv[i]) * uv[i] for i in range_constexpr(4)]
            return Vec.from_elements(elems, fx.Float32)
        limit = fx.Float32(self._swiglu_limit)
        elems = [
            self._silu(-(-gv[i]).maximumf(-limit)) * fx.clampf(uv[i], -limit, limit)
            for i in range_constexpr(4)
        ]
        return Vec.from_elements(elems, fx.Float32)

    def store(self, acc, tile_i32, tile_row_base_i32, n_tile_base_i32, valid_rows_i32=None, store_cm=None):
        """valid_rows_i32 drops padding-row output / scale stores (nobody reads them)."""
        if store_cm is None:
            store_cm = self._out_cache_modifier
        combined = self._combine(acc)
        n_per = len(combined) // self._m_repeat
        if self._out_tensor is not None:
            tile_iter = fx.add_offset(
                fx.get_iter(self._out_tensor),
                fx.Int64(tile_i32) * fx.Int64(self._sort_block_m * self._inter_dim),
            )
            tile_view = fx.Tensor(fx.make_view(tile_iter, fx.make_layout(1, 1)))
            out_rsrc = _make_buffer(
                tile_view,
                fx.Int16,
                max_size=False,
                num_records_bytes=(fx.Int32(self._sort_block_m * self._inter_dim) if valid_rows_i32 is None
                                   else valid_rows_i32 * fx.Int32(self._inter_dim)),
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
                    if self._out_tensor is not None
                    else row_g * fx.Int32(self._inter_dim)
                )
            else:
                tok = _buffer_load(self._sorted_rsrc, slot, fx.Int32)
                valid = tok < fx.Int32(self._tokens)
                out_row_base = slot * fx.Int32(self._inter_dim)
            if const_expr(valid_rows_i32 is not None):
                valid = valid & (row < valid_rows_i32)
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
                _buffer_store(out_rsrc, out_byte // fx.Int32(2), short_raw, fx.Int16, cache_modifier=store_cm)

                col_s = gcol >> fx.Int32(5)
                is_writer = (gcol & fx.Int32(31)) == fx.Int32(0)
                if const_expr(valid_rows_i32 is not None):
                    is_writer = is_writer & (row < valid_rows_i32)
                d0 = row_g >> fx.Int32(5)
                d1 = (row_g >> fx.Int32(4)) & fx.Int32(1)
                d2 = row_g & fx.Int32(15)
                d3 = col_s >> fx.Int32(3)
                d4 = (col_s >> fx.Int32(2)) & fx.Int32(1)
                d5 = col_s & fx.Int32(3)
                byte_off = d0 * n32 + d3 * fx.Int32(256) + d5 * fx.Int32(64) + d2 * fx.Int32(4) + d4 * fx.Int32(2) + d1
                byte_off = is_writer.select(byte_off, fx.Int32(0x40000000))
                e8m0_i8 = e8m0_v.to(fx.Int8)
                _buffer_store(self._out_scale_rsrc, byte_off, e8m0_i8, fx.Int8, cache_modifier=store_cm)
        wait_lds_barrier()


class _LdsF32View:
    def __init__(self, ptr):
        self.ptr = ptr


# fmt: off
@flyc.jit
def do_tile(m_tile, n_tile_base, expert, sched, a_gather, a_s2r, b_loader, b_scale, a_scale, mfma, epi, a_buf,
    a_scale_lds, a_lds_i32, K_ITERS, M_REPEAT, NUM_ACC_N, A_K_STEP_BYTES, pipe_weights,
    mfma_amajor, async_a_copy, trb_rsrc, rows_rsrc=None):
# fmt: on
    N_ACC = M_REPEAT * NUM_ACC_N
    NUM_B_SCALE = NUM_ACC_N // _PACK
    NUM_A_SCALE = M_REPEAT // _PACK
    B_STATE_END = (
        N_ACC + NUM_ACC_N * _PACK
    )
    SB_STATE_END = B_STATE_END + NUM_B_SCALE
    last = fx.Int32(K_ITERS - 1)
    tile_row_base = _buffer_load(trb_rsrc, m_tile, fx.Int32)
    b_row = sched.gate_base_row(expert) + n_tile_base
    a_gather.for_tile(tile_row_base, None if rows_rsrc is None else _buffer_load(rows_rsrc, m_tile, fx.Int32))
    if const_expr(pipe_weights):
        if const_expr(async_a_copy):
            a_gather.prefetch_to_lds(
                fx.Int32(0),
                a_buf,
                fx.Int32(0),
            )
        else:
            a_gather.store(
                a_buf,
                a_gather.load_regs(fx.Int32(0)),
                fx.Int32(0),
            )
        a_scale.stage(a_scale_lds, tile_row_base)
        wait_lds_barrier(0 if async_a_copy else 63)
        b0 = b_loader.load_step(b_row, fx.Int32(0))
        init = [mfma.zero_value for _ in range(N_ACC)]
        init += [h for ni_list in b0 for h in ni_list]
        if const_expr(async_a_copy):
            init += b_scale.load_step(
                b_row,
                fx.Int32(0),
            )
            init += a_scale.load_step(
                a_scale_lds,
                fx.Int32(0),
            )
        for sp_i, state in range(0, K_ITERS - 1, 1, init=init):
            sp = fx.Int32(sp_i)
            acc = [Vec(a) for a in state[:N_ACC]]
            b_prev = [
                [Vec(state[N_ACC + ni * _PACK + ks]) for ks in range(_PACK)] for ni in range(NUM_ACC_N)
            ]
            cur_off = (sp & fx.Int32(1)) * fx.Int32(a_lds_i32)
            nxt_off = ((sp + fx.Int32(1)) & fx.Int32(1)) * fx.Int32(a_lds_i32)
            spn = sp + fx.Int32(1)
            if const_expr(async_a_copy):
                sb = [
                    fx.Int32(
                        state[B_STATE_END + group]
                    )
                    for group in range_constexpr(
                        NUM_B_SCALE
                    )
                ]
                sa = [
                    fx.Int32(
                        state[SB_STATE_END + group]
                    )
                    for group in range_constexpr(
                        NUM_A_SCALE
                    )
                ]
            else:
                sb = b_scale.load_step(b_row, sp)
                sa = a_scale.load_step(
                    a_scale_lds,
                    sp,
                )

            def a_load(mi, ks, _base=cur_off):
                return a_s2r.load_operand(a_buf, mi, ks, _base)

            if const_expr(async_a_copy):
                rocdl.sched_barrier(0)
                a_gather.prefetch_to_lds(
                    spn * fx.Int32(A_K_STEP_BYTES),
                    a_buf,
                    nxt_off,
                )
                rocdl.sched_barrier(0)
                sb_next = b_scale.load_step(
                    b_row,
                    spn,
                )
                sa_next = a_scale.load_step(
                    a_scale_lds,
                    spn,
                )
            else:
                a_regs = a_gather.load_regs(
                    spn * fx.Int32(A_K_STEP_BYTES)
                )

            def load_next(ni, _kn=spn):
                return b_loader.load_ni(b_row, ni, _kn)

            call_pipe = (
                mfma.call_pipe_am
                if mfma_amajor
                else mfma.call_pipe
            )
            acc, b_next = call_pipe(
                a_load,
                b_prev,
                acc,
                sa,
                sb,
                load_next,
            )
            if const_expr(async_a_copy):
                wait_lds_barrier(
                    NUM_ACC_N * _PACK
                    + NUM_B_SCALE
                )
            else:
                a_gather.store(a_buf, a_regs, nxt_off)
                wait_lds_barrier()
            yv = list(acc) + [h for ni_list in b_next for h in ni_list]
            if const_expr(async_a_copy):
                yv += sb_next
                yv += sa_next
            state = yield yv
        acc = [Vec(r) for r in state[:N_ACC]]
        b_prev = [
            [Vec(state[N_ACC + ni * _PACK + ks]) for ks in range(_PACK)]
            for ni in range(NUM_ACC_N)
        ]
        # m_tile >= 0, so tile_zero == 0, but it is opaque per tile: without it every tail A ds_read
        # address folds to a loop invariant, is hoisted to the preheader and spilled at the VGPR cap.
        tile_zero = (m_tile < fx.Int32(0)).select(m_tile, fx.Int32(0))
        final_off = (last & fx.Int32(1)) * fx.Int32(a_lds_i32) + tile_zero

        def final_a_load(mi, ks, _base=final_off):
            return a_s2r.load_operand(a_buf, mi, ks, _base)

        if const_expr(async_a_copy):
            sb = [
                fx.Int32(
                    state[B_STATE_END + group]
                )
                for group in range_constexpr(
                    NUM_B_SCALE
                )
            ]
            sa = [
                fx.Int32(
                    state[SB_STATE_END + group]
                )
                for group in range_constexpr(
                    NUM_A_SCALE
                )
            ]
        else:
            sb = b_scale.load_step(b_row, last)
            sa = a_scale.load_step(
                a_scale_lds,
                last,
            )
        acc = mfma.call_pipe_am_final(
            final_a_load,
            b_prev,
            acc,
            sa,
            sb,
        )
    else:
        if const_expr(async_a_copy):
            a_gather.prefetch_to_lds(
                fx.Int32(0),
                a_buf,
                fx.Int32(0),
            )
        else:
            a_gather.store(
                a_buf,
                a_gather.load_regs(fx.Int32(0)),
                fx.Int32(0),
            )
        a_scale.stage(a_scale_lds, tile_row_base)
        wait_lds_barrier(0 if async_a_copy else 63)
        init = [mfma.zero_value for _ in range(N_ACC)]
        for sp_i, state in range(0, K_ITERS, 1, init=init):
            sp = fx.Int32(sp_i)
            acc = [Vec(a) for a in state]
            cur_off = (sp & fx.Int32(1)) * fx.Int32(a_lds_i32)
            nxt_off = ((sp + fx.Int32(1)) & fx.Int32(1)) * fx.Int32(a_lds_i32)
            spn = (sp + fx.Int32(1) < last).select(
                sp + fx.Int32(1),
                last,
            )

            def a_load(mi, ks, _base=cur_off):
                return a_s2r.load_operand(a_buf, mi, ks, _base)

            b = b_loader.load_step(b_row, sp)
            sa = a_scale.load_step(a_scale_lds, sp)
            sb = b_scale.load_step(b_row, sp)
            if const_expr(async_a_copy):
                rocdl.sched_barrier(0)
                a_gather.prefetch_to_lds(
                    spn * fx.Int32(A_K_STEP_BYTES),
                    a_buf,
                    nxt_off,
                )
                rocdl.sched_barrier(0)
            else:
                a_regs = a_gather.load_regs(
                    spn * fx.Int32(A_K_STEP_BYTES)
                )
            acc = mfma.call(
                a_load,
                b,
                acc,
                sa,
                sb,
            )
            if const_expr(async_a_copy):
                wait_lds_barrier(0)
            else:
                a_gather.store(a_buf, a_regs, nxt_off)
                wait_lds_barrier()
            state = yield list(acc)
        acc = [Vec(r) for r in state]
    # The epilogue aliases A_buf as cshuffle LDS after every wave finishes its final A ds_read.
    wait_lds_barrier()
    epi.store(acc, m_tile, tile_row_base, n_tile_base)


# fmt: off
@flyc.jit
def do_tile_la(m_tile, n_tile_base, expert, sched, a_gather, a_s2r, b_loader, b_scale, a_scale, mfma, epi, a_buf,
    a_scale_lds, a_lds_i32, K_ITERS, M_REPEAT, NUM_ACC_N, A_K_STEP_BYTES, trb_rsrc, rows_rsrc, LA):
# fmt: on
    """pipe_weights K-loop with B / B-scale issued LA K-steps ahead, unrolled xLA.

    The baseline x1 loop yields freshly loaded B registers that still interfere with the B registers feeding the
    MFMAs, so LLVM emits phi copies and drains the next-step loads before the back-edge: no real lookahead, one
    full HBM latency per K-step. Unrolling by LA rotates the LA-deep register queue back to identity each
    iteration, so the yielded sets coalesce with the loop arguments and loads stay in flight across the back-edge.
    Per-accumulator MFMA order (ksub 0 then 1, K-steps ascending) is unchanged -> bit-identical output.
    B-scale loads and epilogue stores are nt; the epilogue stores are clamped to the tile's valid rows.
    """
    N_ACC = M_REPEAT * NUM_ACC_N
    NUM_B = NUM_ACC_N * _PACK
    NUM_SB = NUM_ACC_N // _PACK
    sb_cm = 3
    tile_row_base = fx.Int32(rocdl.readfirstlane(T.i32, _raw(_buffer_load(trb_rsrc, m_tile, fx.Int32))))
    b_row = sched.gate_base_row(expert) + n_tile_base
    valid_rows = None
    if const_expr(rows_rsrc is not None):
        valid_rows = fx.Int32(rocdl.readfirstlane(T.i32, _raw(_buffer_load(rows_rsrc, m_tile, fx.Int32))))
    a_gather.for_tile(tile_row_base, valid_rows)
    a_gather.store(a_buf, a_gather.load_regs(fx.Int32(0)), fx.Int32(0))
    a_scale.stage(a_scale_lds, tile_row_base)
    wait_lds_barrier(63)
    init = [mfma.zero_value for _ in range(N_ACC)]
    for q in range_constexpr(LA):
        init += [h for ni_list in b_loader.load_step(b_row, fx.Int32(q)) for h in ni_list]
    for q in range_constexpr(LA):
        init += b_scale.load_step(b_row, fx.Int32(q), sb_cm)
    a_lds_bytes = [lb for lb, _c in a_gather._chunks]

    def unpack(state):
        acc = [Vec(a) for a in state[:N_ACC]]
        bq = []
        for q in range_constexpr(LA):
            base = N_ACC + q * NUM_B
            bq.append([[Vec(state[base + ni * _PACK + ks]) for ks in range(_PACK)] for ni in range(NUM_ACC_N)])
        sbase = N_ACC + LA * NUM_B
        sbq = [[fx.Int32(state[sbase + q * NUM_SB + g]) for g in range(NUM_SB)] for q in range(LA)]
        return acc, bq, sbq

    def pack(acc, bq, sbq):
        out = list(acc)
        for b in bq:
            out += [h for ni_list in b for h in ni_list]
        for sb in sbq:
            out += list(sb)
        return out

    def step(sp, acc, bq, sbq, issue, a_next):
        cur_off = (sp & fx.Int32(1)) * fx.Int32(a_lds_i32)
        nxt_off = ((sp + fx.Int32(1)) & fx.Int32(1)) * fx.Int32(a_lds_i32)
        kl = sp + fx.Int32(LA)
        sb_new = b_scale.load_step(b_row, kl, sb_cm) if issue else None
        sa = a_scale.load_step(a_scale_lds, sp)
        a_regs = None
        if const_expr(a_next):
            a_regs = [v for _lb, v in a_gather.load_regs((sp + fx.Int32(1)) * fx.Int32(A_K_STEP_BYTES))]

        def a_load(mi, ks, _base=cur_off):
            return a_s2r.load_operand(a_buf, mi, ks, _base)

        def load_next(ni, _kn=kl):
            return b_loader.load_ni(b_row, ni, _kn)

        if const_expr(issue):
            acc, b_new = mfma.call_pipe(a_load, bq[0], acc, sa, sbq[0], load_next)
            bq = bq[1:] + [b_new]
            sbq = sbq[1:] + [sb_new]
        else:
            acc = mfma.call(a_load, bq[0], acc, sa, sbq[0])
            bq = bq[1:]
            sbq = sbq[1:]
        if const_expr(a_next):
            a_gather.store(a_buf, list(zip(a_lds_bytes, a_regs)), nxt_off)
            wait_lds_barrier()
        return acc, bq, sbq

    n_loop = (K_ITERS - LA) // LA
    for it_i, state in range(0, n_loop, 1, init=init):
        sp0 = fx.Int32(it_i) * fx.Int32(LA)
        acc, bq, sbq = unpack(state)
        for u in range_constexpr(LA):
            acc, bq, sbq = step(sp0 + fx.Int32(u), acc, bq, sbq, True, True)
        state = yield pack(acc, bq, sbq)
    acc, bq, sbq = unpack(state)
    for k in range_constexpr(n_loop * LA, K_ITERS):
        acc, bq, sbq = step(fx.Int32(k), acc, bq, sbq, k + LA <= K_ITERS - 1, k < K_ITERS - 1)
    wait_lds_barrier()
    epi.store(acc, m_tile, tile_row_base, n_tile_base, valid_rows, 3)


# fmt: off
def build_fused_gemm1(*, x_tensor, w_rsrc, sw_rsrc, sx_rsrc,
    out_rsrc, os_rsrc, trb_rsrc, expert_rsrc, out_tensor, a_buf, a_scale_lds, c_tile,
    model_dim, inter_dim, sort_block_m, tile_n, num_waves, n_per_wave, wave_id,
    m_repeat, num_acc_n, a_k_step_bytes, total_threads, k_iters, a_lds_i32, n_tiles,
    expert_offset, b_cache_modifier, swizzle_a, pipe_weights, mfma_amajor, async_a_copy,
    use_tile_resource, swiglu_limit=0.0, rows_rsrc=None, lookahead=0, out_cache_modifier=0):
    # fmt: on
    """Build the GEMM1 atoms and return its expert resolver and tile runner."""
    sched = TileScheduler(
        expert_rsrc=expert_rsrc,
        inter_dim=inter_dim,
        expert_offset=expert_offset,  # GLOBAL sorted_expert_id -> LOCAL w1 index
    )
    n_wave_base = wave_id * fx.Int32(n_per_wave)

    # fmt: off
    a_gather = ATileLoader(row_bytes=model_dim, sort_block_m=sort_block_m,
        k_step_bytes=a_k_step_bytes, total_threads=total_threads, swizzle=swizzle_a,
        x_tensor=x_tensor, async_copy=async_a_copy)
    # fmt: on
    a_s2r = AS2RLoader(k_step_bytes=a_k_step_bytes, swizzle=swizzle_a)
    b_loader = BWeightLoader(
        w_rsrc=w_rsrc,
        num_acc_n=num_acc_n,
        model_dim=model_dim,
        cache_modifier=b_cache_modifier,
    )
    b_scale = BScaleLoader(scale_rsrc=sw_rsrc, num_acc_n=num_acc_n, model_dim=model_dim)
    a_scale = AScaleLoader(
        scale_rsrc=sx_rsrc,
        m_repeat=m_repeat,
        model_dim=model_dim,
        sort_block_m=sort_block_m,
        total_threads=total_threads,
    )
    mfma = MfmaScaleGU(m_repeat=m_repeat, num_acc_n=num_acc_n)
    # fmt: off
    epi = SiluQuantEpilogue(out_rsrc=out_rsrc, out_scale_rsrc=os_rsrc, sorted_rsrc=trb_rsrc, tokens=0,
        inter_dim=inter_dim, m_repeat=m_repeat, num_acc_n=num_acc_n, sort_block_m=sort_block_m, tile_n=tile_n,
        num_waves=num_waves, lds_out=c_tile, swiglu_limit=swiglu_limit, always_valid=True,
        out_tensor=out_tensor if use_tile_resource else None, out_cache_modifier=out_cache_modifier)
    # fmt: on

    def _decode(flat):
        m_tile = flat // fx.Int32(n_tiles)
        n_tile = flat - m_tile * fx.Int32(n_tiles)
        return m_tile, n_tile

    def expert_of_flat(flat):
        m_tile, _n = _decode(flat)
        return sched.expert_of(m_tile)

    def do_scheduled_tile(flat):
        m_tile, n_tile = _decode(flat)
        n_tile_base = n_wave_base + n_tile * fx.Int32(tile_n)
        expert = sched.expert_of(m_tile)
        # fmt: off
        if const_expr(lookahead > 0):
            assert pipe_weights and not async_a_copy and not mfma_amajor
            do_tile_la(m_tile, n_tile_base, expert, sched, a_gather, a_s2r, b_loader, b_scale, a_scale, mfma, epi,
                a_buf, a_scale_lds, a_lds_i32, k_iters, m_repeat, num_acc_n, a_k_step_bytes, trb_rsrc, rows_rsrc,
                lookahead)
            return
        do_tile(m_tile, n_tile_base, expert, sched, a_gather,
            a_s2r, b_loader, b_scale, a_scale, mfma, epi, a_buf,
            a_scale_lds, a_lds_i32, k_iters, m_repeat, num_acc_n,
            a_k_step_bytes, pipe_weights, mfma_amajor, async_a_copy,
            trb_rsrc, rows_rsrc)
        # fmt: on

    return expert_of_flat, do_scheduled_tile


def scale_view(
    arg_scale, base_dw, K_TILES_TOTAL, k0_stride_dw=64, num_records_bytes=None
):
    """View one e8m0 scale word, optionally bounded to the real buffer extent."""
    base_dw = rocdl.readfirstlane(T.i32, base_dw)
    i32_ptr_ty = fx.PointerType.get(
        T.i32, address_space=fx.AddressSpace.Global, alignment=4
    )
    off_i64 = fx.Int64(base_dw)
    base_iter = fx.inttoptr(i32_ptr_ty, fx.Int64(arg_scale) + off_i64 * fx.Int64(4))
    shape = (4, 16, K_TILES_TOTAL, 1)
    stride = (16, 1, k0_stride_dw, 1)
    view = fx.Tensor(fx.make_view(base_iter, fx.make_layout(shape, stride)))
    if num_records_bytes is not None:
        return fx.rocdl.make_buffer_tensor(view, num_records_bytes=num_records_bytes)
    return fx.rocdl.make_buffer_tensor(view, max_size=False)


def scale_mma_atoms(a_dtype):
    """16 (opselA,opselB) scaled-MFMA atoms; A elem is fp8/fp4, B is fp4."""
    elem_a = Float8E4M3FN if a_dtype == "fp8" else Float4E2M1FN
    return {
        (osa, osb): fx.make_mma_atom(
            fx.rocdl.cdna4.MFMA_Scale(
                16, 16, 128, elem_a, Float4E2M1FN, opsel_a=osa, opsel_b=osb
            )
        )
        for osa in range(4)
        for osb in range(4)
    }


def mma_one_j(
    J,
    in_b,
    sa,
    sb,
    bq_frags_kt,
    a_frags,
    c_frags,
    atoms,
    i0=0,
    single_rg=False,
    rg_off=0,
    k_halves=2,
):
    """Run one scaled-MFMA cluster over a 32-row A-scale group."""
    if const_expr(single_rg):
        steps = tuple(
            (2 * k + rg_off, k, i0, bq_frags_kt[J][k]) for k in range(k_halves)
        )
    else:
        steps = tuple(
            (2 * k + im, k, i0 + im, bq_frags_kt[J][k])
            for k in range(k_halves)
            for im in range(2)
        )
    for osa, k, i, bJ in steps:
        osb = 2 * k + in_b
        fx.gemm(
            atoms[(osa, osb)],
            c_frags[i][J],
            a_frags[i][k],
            bJ,
            c_frags[i][J],
            scale_a=sa,
            scale_b=sb,
        )


def issue_a_load_lds_dt(
    arg_aq,
    s_aq_base,
    slot,
    kt,
    m_row,
    wave,
    lane,
    is_f8,
    KH_TILE_A,
    K_BYTES,
    BM=32,
    valid_rows=None,
    NW=4,
):
    """Load one A tile through a tile-local descriptor so allocations may span the 4 GiB buffer ABI."""
    lanes_per_row = KH_TILE_A // 16  # 8 (fp4) / 16 (fp8)
    rows_per_call = 64 // lanes_per_row  # 8 (fp4) / 4 (fp8)
    a_lane_row = lane // lanes_per_row
    rows_per_wave = BM // NW  # rows each wave loads (NW4: BM32 8, BM64 16; NW8: BM64 8)
    # BM16 fp4: partial-wave round-robin (waves 2,3 re-load, harmless); BM>=32 byte-identical per-wave blocks.
    partial_wave_gather = rows_per_wave < rows_per_call
    if const_expr(partial_wave_gather):
        n_gather_calls = BM // rows_per_call
        gather_base_row = (wave % fx.Int32(n_gather_calls)) * rows_per_call
        n_row_groups = 1
    else:
        gather_base_row = wave * rows_per_wave
        n_row_groups = rows_per_wave // rows_per_call
    lane_col = (lane % lanes_per_row) * 16
    base_i32 = s_aq_base
    atom = lds_dma_atom_128()
    tile_base = fx.Int64(arg_aq) + fx.Int64(m_row) * fx.Int64(K_BYTES)
    src = flat_buffer_view(
        tile_base,
        None,
        T.i32,
        align=16,
        elem_bytes=4,
        fold=False,
        num_records_bytes=fx.Int64(BM if valid_rows is None else valid_rows)
        * fx.Int64(K_BYTES),
    )
    for g in range_constexpr(n_row_groups):
        lds_row = gather_base_row + g * rows_per_call
        mask = (
            lds_swizzle_mask_f8(lds_row + a_lane_row, KH_TILE_A)
            if const_expr(is_f8)
            else lds_swizzle_mask(lds_row + a_lane_row)
        )
        tile_row = lds_row + a_lane_row
        voffset = (lane_col ^ mask) + tile_row * K_BYTES
        off = fx.Int32(slot * (BM * KH_TILE_A)) + lds_row * KH_TILE_A
        v_e = (voffset + kt * KH_TILE_A) // 4  # per-lane i32-elem index
        fx.copy(
            atom, src[v_e, None], lds_dma_dst(base_i32, off, elem_ty=T.i32, align=16)
        )


@flyc.jit
def gemm2_compute_v2(
    lds_base_i32,
    arg_ascale,
    arg_bq,
    arg_bscale,
    arg_eids,
    arg_aq,
    i32_max_m_blocks,
    bx_i32,
    lane,
    wave,
    i32_inter,
    i32_hidden,
    i32_kpad,
    i32_npad,
    *,
    BM,
    BN=256,
    BK=256,
    use_nt,
    INTER_MAX,
    aStages,
    a_dtype,
    has_pad=False,
    SBM=None,
    g2_bhoist=True,
    g2_ascale_pf=True,
    expert_offset=0,
    valid_rows=None,
    NW=4,
    early_b_load=False,
):
    """Run the GEMM2 K-loop and return accumulators for the selected epilogue."""
    # SBM is the sort padding unit; BM is the compute tile and must divide SBM.
    if SBM is None:
        SBM = BM
    kMChunks = BM // 16  # 16-row MFMA row-groups
    kHalves = BK // 128  # 16x16x128 MFMA K-steps per K-tile
    tilesPerScaleChunk = 256 // BK  # K-tiles sharing one 256-K E8M0 word
    numAccN = (BN // NW) // 16  # 16-column MFMA subblocks per wave
    nPairs = max(1, numAccN // 2)  # one B-scale per two 16-column subblocks
    # BM16: single 16-row block owning a 32-row scale chunk (chunk==m_block_idx, rg0-only).
    is_bm16 = BM < 32
    rg_off = 0
    kScaleSubBlocks = max(1, kMChunks // 2)
    is_f8_a = a_dtype == "fp8"  # only the A path differs
    a_pack = 1 if is_f8_a else 2
    KH_TILE_A = BK // a_pack
    slot_bytes = BM * KH_TILE_A
    # Contraction K = inter_dim runtime (i32_inter); INTER_MAX caps compile-time view/fragment bounds.
    K_rt = fx.Int32(i32_inter)
    K_BYTES = K_rt // fx.Int32(a_pack)  # A row stride bytes (runtime)
    kc_rt = K_rt // fx.Int32(256)  # (K//32)//4//2
    K_TILES_RT = K_rt // fx.Int32(BK)  # runtime K-tile trip count
    kAS_per_chunk_dw = kc_rt * fx.Int32(64)
    kBS_stride_n0_dw = kc_rt * fx.Int32(64)
    # N_OUT = model_dim/hidden is the gemm2 output N dim; runtime via i32_hidden (no K-loop dependency).
    N_OUT_rt = fx.Int32(i32_hidden)
    kbs_per_expert_dw = (
        N_OUT_rt // fx.Int32(32)
    ) * kBS_stride_n0_dw  # (N_OUT//16//2)*stride
    num_n_blocks = N_OUT_rt // fx.Int32(BN)
    KH4 = K_rt // fx.Int32(8)  # i32 col stride (= K_HALF//4)
    K_SCALE_CHUNKS_MAX = INTER_MAX // 256

    # Padded shapes mask weight tiles beyond the real K/N extents.
    N_real = None
    if const_expr(has_pad):
        K_real = K_rt - fx.Int32(i32_kpad)
        halves_real = (K_real + fx.Int32(127)) // fx.Int32(128)
        N_real = N_OUT_rt - fx.Int32(i32_npad)

    # Map each compute block to its SBM-padded expert metadata row.
    m_block_idx = bx_i32 // num_n_blocks
    n_block_idx = bx_i32 - m_block_idx * num_n_blocks
    eids_ptr = global_typed_ptr(arg_eids, T.i32)
    if const_expr(SBM == BM):
        e = rocdl.readfirstlane(T.i32, eids_ptr[m_block_idx])
        m_row = m_block_idx * BM
    else:
        m_row = m_block_idx * BM
        e = rocdl.readfirstlane(T.i32, eids_ptr[m_row // fx.Int32(SBM)])
    if const_expr(expert_offset != 0):
        e = e - fx.Int32(expert_offset)

    lane_div_16 = lane // 16
    lane_mod_16 = lane % 16

    s_aq_base = lds_base_i32
    mma_atoms = scale_mma_atoms(a_dtype)

    # A activation: global->LDS DMA (issue_a_load_lds), then LDS->reg ds-read (issue_a_ds_read).
    A_NDW = (
        8 if is_f8_a else 4
    )  # fp8 packs two 128-K halves -> i32<8:1>; fp4 -> i32<4:1>
    a_frags = [
        [fx.make_rmem_tensor(A_NDW, Int32) for _ in range_constexpr(kHalves)]
        for _ in range_constexpr(kMChunks)
    ]

    def issue_a_load_lds(slot, kt):
        issue_a_load_lds_dt(
            arg_aq,
            s_aq_base,
            slot,
            kt,
            m_row,
            wave,
            lane,
            is_f8_a,
            KH_TILE_A,
            K_BYTES,
            BM=BM,
            valid_rows=valid_rows,
            NW=NW,
        )

    def issue_a_ds_read(slot):
        # A ds-read for one slot into a_frags: fp8 -> i32<8:1> (two 128-K halves), fp4 -> i32<4:1>.
        for k in range_constexpr(kHalves):
            for i in range_constexpr(kMChunks):
                lds_row = lane_mod_16 + i * 16
                row_off = fx.Int32(slot * slot_bytes) + lds_row * KH_TILE_A
                if const_expr(is_f8_a):
                    mask = lds_swizzle_mask_f8(lane_mod_16, KH_TILE_A)
                    col0 = lane_div_16 * 16 + k * 128
                    col_lo = col0 ^ mask
                    col_hi = (col0 + 64) ^ mask
                    lo = Vec(
                        lds_vec_load(
                            s_aq_base,
                            row_off + col_lo,
                            Vec.make_type(2, fx.Int64),
                            fx.Int64,
                            align=16,
                        )
                    )
                    hi = Vec(
                        lds_vec_load(
                            s_aq_base,
                            row_off + col_hi,
                            Vec.make_type(2, fx.Int64),
                            fx.Int64,
                            align=16,
                        )
                    )
                    a64 = Vec.from_elements([lo[0], lo[1], hi[0], hi[1]], fx.Int64)
                    a_frags[i][k].store(a64.bitcast(fx.Int32))
                else:
                    mask = lds_swizzle_mask(lane_mod_16)
                    lds_col = (lane_div_16 * 16 + k * 64) ^ mask
                    vec = lds_vec_load(
                        s_aq_base,
                        row_off + lds_col,
                        Vec.make_type(4, fx.Int32),
                        fx.Int32,
                        align=16,
                    )
                    a_frags[i][k].store(Vec(vec))

    # The shared e8m0 scale layout bounds each A-scale view to its remaining bytes.
    sc_copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), 32)

    asc_per_mb = fx.Int32(kScaleSubBlocks) * kAS_per_chunk_dw * fx.Int32(4)
    asc_num = fx.Int64(i32_max_m_blocks) * fx.Int64(asc_per_mb)
    scale_chunk0 = m_block_idx if const_expr(is_bm16) else m_row // 32

    def make_ascale_view(sub):
        base_dw = (scale_chunk0 + fx.Int32(sub)) * kAS_per_chunk_dw
        nrec = asc_num - fx.Int64(base_dw) * fx.Int64(4)
        return scale_view(
            arg_ascale,
            base_dw,
            K_SCALE_CHUNKS_MAX,
            k0_stride_dw=64,
            num_records_bytes=nrec,
        )

    ascale_views = [make_ascale_view(sub) for sub in range_constexpr(kScaleSubBlocks)]
    sc_frag_tmpl = ascale_views[0][0, 0, 0, None]  # i32<1:1> (one e8m0 word)

    def load_a_scale_tile(kt):
        # One i32 A-scale register per 32-row chunk (kScaleSubBlocks).
        chunk_kt = (
            kt
            if const_expr(tilesPerScaleChunk == 1)
            else kt // fx.Int32(tilesPerScaleChunk)
        )
        out = []
        for sub in range_constexpr(kScaleSubBlocks):
            saf = fx.make_fragment_like(sc_frag_tmpl)
            fx.copy(
                sc_copy_atom,
                ascale_views[sub][lane_div_16, lane_mod_16, chunk_kt, None],
                saf,
            )
            out.append(Vec(saf.load())[0])
        return out

    # Stream B weights and scales through registers so use_nt reaches the ISA cache policy.
    bq_rsrc = AddrBuffer(arg_bq)

    bq_base_dw = [
        rocdl.readfirstlane(
            T.i32,
            (e * N_OUT_rt + n_block_idx * BN + wave * (BN // NW) + j * 16) * KH4,
        )
        for j in range_constexpr(numAccN)
    ]

    mni_base = n_block_idx * (BN // 16 // 2) + wave * (BN // (16 * NW) // 2)
    bscale_views = [
        scale_view(
            arg_bscale,
            e * kbs_per_expert_dw + (mni_base + mw) * kBS_stride_n0_dw,
            K_SCALE_CHUNKS_MAX,
            k0_stride_dw=kBS_stride_k0_dw,
        )
        for mw in range_constexpr(nPairs)
    ]

    frag_tmpl = fx.make_rmem_tensor(4, Int32)
    # B-scale word template shares the A-scale layout (sc_frag_tmpl).

    def issue_b_load_into(bqf, bsf, kt_rt):
        # Issue B-weight + B-scale vmem loads for K-tile kt_rt into the given (per-stage) fragments.
        for j in range_constexpr(numAccN):
            for half in range_constexpr(kHalves):
                bq_off_dw = (
                    bq_base_dw[j]
                    + lane_div_16 * fx.Int32(64)
                    + lane_mod_16 * fx.Int32(4)
                    + kt_rt * fx.Int32(kHalves * 256)
                    + fx.Int32(half * 256)
                )
                load_mask = None
                if const_expr(has_pad):
                    col = n_block_idx * BN + wave * (BN // NW) + j * 16
                    load_mask = (col < N_real) & (
                        kt_rt * fx.Int32(kHalves) + fx.Int32(half) < halves_real
                    )
                bq_vec = load_at(
                    bq_rsrc, bq_off_dw, fx.Int32, 4, 2 if use_nt else 0, mask=load_mask
                )
                bqf[j][half].store(Vec(bq_vec))
        chunk_kt = (
            kt_rt
            if const_expr(tilesPerScaleChunk == 1)
            else kt_rt // fx.Int32(tilesPerScaleChunk)
        )
        for mw in range_constexpr(nPairs):
            fx.copy(
                sc_copy_atom,
                bscale_views[mw][lane_div_16, lane_mod_16, chunk_kt, None],
                bsf[mw],
            )

    def stream_b_tile(kt_rt):
        # Fresh per-iter fragments (B streamed, not register-resident) then issue_b_load_into.
        bqf = [
            [fx.make_fragment_like(frag_tmpl) for _ in range_constexpr(kHalves)]
            for _ in range_constexpr(numAccN)
        ]
        bsf = [fx.make_fragment_like(sc_frag_tmpl) for _ in range_constexpr(nPairs)]
        issue_b_load_into(bqf, bsf, kt_rt)
        return bqf, bsf

    # Scaled-MFMA clusters over the loaded A / B / scale fragments.
    def shift_scale_word(scale, kt_rt):
        if const_expr(tilesPerScaleChunk == 1):
            return scale
        scale_shift = (kt_rt % fx.Int32(tilesPerScaleChunk)) * fx.Int32(16)
        return fx.Int32(scale).shrui(scale_shift)

    def mfma_cluster(bqf, bsf, sa, kt_rt):
        # opsel (no gate/up split): mni=J//2, in_b=J%2; sa is a per-32-row-chunk list.
        sa = [
            shift_scale_word(sa[sub], kt_rt) for sub in range_constexpr(kScaleSubBlocks)
        ]
        sb_words = [
            shift_scale_word(Vec(bsf[mni].load())[0], kt_rt)
            for mni in range_constexpr(nPairs)
        ]
        for J in range_constexpr(numAccN):
            mni, in_b = J // 2, J % 2
            sb = sb_words[mni]
            if const_expr(is_bm16):
                mma_one_j(
                    J,
                    in_b,
                    sa[0],
                    sb,
                    bqf,
                    a_frags,
                    c_frags,
                    mma_atoms,
                    i0=0,
                    single_rg=True,
                    rg_off=rg_off,
                    k_halves=kHalves,
                )
                continue
            for sub in range_constexpr(kScaleSubBlocks):
                mma_one_j(
                    J,
                    in_b,
                    sa[sub],
                    sb,
                    bqf,
                    a_frags,
                    c_frags,
                    mma_atoms,
                    i0=2 * sub,
                    k_halves=kHalves,
                )

    # C accumulator: register fragments, zeroed then accumulated in place; (un)packed to K-loop carry.
    zero4 = Vec.filled(4, 0.0, Float32)
    c_frags = [
        [fx.make_rmem_tensor(4, Float32) for _ in range_constexpr(numAccN)]
        for _ in range_constexpr(kMChunks)
    ]
    for i in range_constexpr(kMChunks):
        for J in range_constexpr(numAccN):
            c_frags[i][J].store(zero4)

    def load_c_carry():
        return [c_frags[i][J].load() for i in range(kMChunks) for J in range(numAccN)]

    def store_c_carry(state):
        n = 0
        for i in range_constexpr(kMChunks):
            for J in range_constexpr(numAccN):
                c_frags[i][J].store(state[n])
                n += 1
        return n

    if const_expr(BM == 64 and BN // NW == 64):
        # BM64 with a 64-column per-wave slab (NW4 BN256 / NW8 BN512) uses the 1-stage B path.
        for kt_iv, state in range(
            fx.Int32(0),
            K_TILES_RT,
            fx.Int32(1),
            init=load_c_carry(),
        ):
            store_c_carry(state)
            kt_rt = fx.Int32(kt_iv)
            if const_expr(early_b_load):
                # Issue this K-tile's B and A-scale loads before the barrier and the next A-tile DMA:
                # vmcnt retires in issue order, so waiting for them leaves that DMA in flight.
                bqf, bsf = stream_b_tile(kt_rt)
                sa = load_a_scale_tile(kt_rt)
            gpu.barrier()
            issue_a_ds_read(kt_rt % fx.Int32(aStages))
            nxt = kt_rt + fx.Int32(kStages)
            if nxt < K_TILES_RT:
                issue_a_load_lds(nxt % fx.Int32(aStages), nxt)
            if const_expr(not early_b_load):
                bqf, bsf = stream_b_tile(kt_rt)
                sa = load_a_scale_tile(kt_rt)
            mfma_cluster(bqf, bsf, sa, kt_rt)
            results = yield load_c_carry()
        store_c_carry(results)
    else:
        # 2-stage B pipeline: consume carried "current" B, prefetch next tile into the same fragments via scf.for state.
        cur_bqf = [
            [fx.make_fragment_like(frag_tmpl) for _ in range_constexpr(kHalves)]
            for _ in range_constexpr(numAccN)
        ]
        cur_bsf = [fx.make_fragment_like(sc_frag_tmpl) for _ in range_constexpr(nPairs)]
        nxt_bqf = [
            [fx.make_fragment_like(frag_tmpl) for _ in range_constexpr(kHalves)]
            for _ in range_constexpr(numAccN)
        ]
        nxt_bsf = [fx.make_fragment_like(sc_frag_tmpl) for _ in range_constexpr(nPairs)]
        # g2_ascale_pf: carry the A-scale through scf.for state, same rotating-buffer model as B.
        cur_saf = nxt_saf = None
        if const_expr(g2_ascale_pf):
            cur_saf = [
                fx.make_fragment_like(sc_frag_tmpl)
                for _ in range_constexpr(kScaleSubBlocks)
            ]
            nxt_saf = [
                fx.make_fragment_like(sc_frag_tmpl)
                for _ in range_constexpr(kScaleSubBlocks)
            ]

        def load_b_carry():
            # Flat CURRENT (to-consume) B-weight, B-scale, then (opt) A-scale values.
            out = []
            for j in range_constexpr(numAccN):
                for half in range_constexpr(kHalves):
                    out.append(cur_bqf[j][half].load())
            for mw in range_constexpr(nPairs):
                out.append(cur_bsf[mw].load())
            if const_expr(g2_ascale_pf):
                for sub in range_constexpr(kScaleSubBlocks):
                    out.append(cur_saf[sub].load())
            return out

        def store_b_carry(state, base):
            n = base
            for j in range_constexpr(numAccN):
                for half in range_constexpr(kHalves):
                    cur_bqf[j][half].store(state[n])
                    n += 1
            for mw in range_constexpr(nPairs):
                cur_bsf[mw].store(state[n])
                n += 1
            if const_expr(g2_ascale_pf):
                for sub in range_constexpr(kScaleSubBlocks):
                    cur_saf[sub].store(state[n])
                    n += 1
            return n

        def rotate_b_carry():
            # Yield the PREFETCHED (next-tile) values -> become "current" next iteration.
            out = []
            for j in range_constexpr(numAccN):
                for half in range_constexpr(kHalves):
                    out.append(nxt_bqf[j][half].load())
            for mw in range_constexpr(nPairs):
                out.append(nxt_bsf[mw].load())
            if const_expr(g2_ascale_pf):
                for sub in range_constexpr(kScaleSubBlocks):
                    out.append(nxt_saf[sub].load())
            return out

        def issue_a_scale_load_into(saf, kt_rt):
            # A-scale vmem load(s) for K-tile kt_rt into the given (per-stage) fragment(s).
            sa = load_a_scale_tile(kt_rt)
            for sub in range_constexpr(kScaleSubBlocks):
                saf[sub].store(Vec.filled(1, sa[sub], Int32))

        def load_carry():
            return load_c_carry() + load_b_carry()

        def store_carry(state):
            base = store_c_carry(state)
            store_b_carry(state, base)

        def yield_carry():
            return load_c_carry() + rotate_b_carry()

        # Prologue: prefetch tile 0's B/B-scale into "current" (VALUES enter via init=load_carry()).
        issue_b_load_into(cur_bqf, cur_bsf, fx.Int32(0))
        if const_expr(g2_ascale_pf):
            issue_a_scale_load_into(cur_saf, fx.Int32(0))
        rocdl.sched_barrier(0)

        def prefetch_next_b(kt_rt):
            # Prefetch NEXT tile's B; if none, copy current through (rotate_b_carry state, unused after loop).
            nxt_b = kt_rt + fx.Int32(1)
            if nxt_b < K_TILES_RT:
                issue_b_load_into(nxt_bqf, nxt_bsf, nxt_b)
                if const_expr(g2_ascale_pf):
                    issue_a_scale_load_into(nxt_saf, nxt_b)
            else:
                for j in range_constexpr(numAccN):
                    for half in range_constexpr(kHalves):
                        nxt_bqf[j][half].store(cur_bqf[j][half].load())
                for mw in range_constexpr(nPairs):
                    nxt_bsf[mw].store(cur_bsf[mw].load())
                if const_expr(g2_ascale_pf):
                    for sub in range_constexpr(kScaleSubBlocks):
                        nxt_saf[sub].store(cur_saf[sub].load())

        for kt_iv, state in range(
            fx.Int32(0),
            K_TILES_RT,
            fx.Int32(1),
            init=load_carry(),
        ):
            store_carry(state)
            kt_rt = fx.Int32(kt_iv)
            if const_expr(g2_bhoist):
                prefetch_next_b(kt_rt)
            gpu.barrier()
            issue_a_ds_read(kt_rt % fx.Int32(aStages))
            nxt_a = kt_rt + fx.Int32(kStages)
            if nxt_a < K_TILES_RT:
                issue_a_load_lds(nxt_a % fx.Int32(aStages), nxt_a)
            # A-scale from the prefetch carry (g2_ascale_pf) or loaded synchronously here.
            if const_expr(g2_ascale_pf):
                sa = [
                    Vec(cur_saf[sub].load())[0]
                    for sub in range_constexpr(kScaleSubBlocks)
                ]
            else:
                sa = load_a_scale_tile(kt_rt)
            if const_expr(not g2_bhoist):
                prefetch_next_b(kt_rt)
            # Fence the MFMA chain from the B vmem loads (next-tile loads ride ahead of compute).
            rocdl.sched_barrier(0)
            rocdl.s_setprio(1)
            mfma_cluster(cur_bqf, cur_bsf, sa, kt_rt)
            rocdl.s_setprio(0)
            rocdl.sched_barrier(0)
            results = yield yield_carry()
        store_carry(results)

    # Load the C fragments (fp8/fp4 unified onto the same fx.gemm path) and hand them to the epilog.
    accm_vecs = [
        [c_frags[i][J].load() for J in range(numAccN)] for i in range(kMChunks)
    ]
    return accm_vecs, m_row, n_block_idx, N_OUT_rt


def _spart_output_tile_index(block_1d_id, M0, N0, group_num, m01):
    """Map a 1D block ID to a spatially local output tile."""
    gn = fx.Int32(group_num)
    n0 = fx.Int32(N0)
    m01c = fx.Int32(m01)

    # group_size = ceil(M0*N0 / GroupNum); big_group_num = GroupNum - (group_size*GroupNum - M0*N0)
    mn = M0 * n0
    group_size = (mn + gn - fx.Int32(1)) // gn
    big_group_num = gn - (group_size * gn - mn)

    group_id_y = block_1d_id // gn
    group_id_x = block_1d_id - group_id_y * gn

    # remap = group_id_x < big_group_num ? gx*gs + gy : gx*gs + big - gx + gy
    remap_a = group_id_x * group_size + group_id_y
    remap_b = group_id_x * group_size + big_group_num - group_id_x + group_id_y
    remap = (group_id_x < big_group_num).select(remap_a, remap_b)

    idx_M0 = remap // n0
    idx_N0 = remap - idx_M0 * n0

    # M0_tmp = M0 / M01 ; M0_mod_M01 = M0 - M0_tmp*M01 ; M01_adapt = (idx_M0 < M0 - M0_mod) ? M01 : M0_mod
    M0_tmp = M0 // m01c
    M0_mod = M0 - M0_tmp * m01c
    M01_adapt = (idx_M0 < (M0 - M0_mod)).select(m01c, M0_mod)

    idx_M00 = idx_M0 // m01c
    idx_M01 = idx_M0 - idx_M00 * m01c
    idx_local = idx_N0 + idx_M01 * n0

    N_out = idx_local // M01_adapt
    loc_mod = idx_local - N_out * M01_adapt

    m_block_idx = loc_mod + idx_M00 * m01c
    n_block_idx = N_out
    return m_block_idx, n_block_idx


def _resolve_g2_knobs(g2_bhoist, g2_ascale_pf, g2_spart, g2_bf16_lds, use_reduce):
    """Resolve explicit GEMM2 knobs against deterministic defaults."""
    g2_bhoist = True if g2_bhoist is None else bool(g2_bhoist)
    g2_ascale_pf = True if g2_ascale_pf is None else bool(g2_ascale_pf)
    g2_spart = 402 if g2_spart is None else int(g2_spart)
    g2_group_num = g2_spart // 100 if g2_spart > 0 else 0
    g2_m01 = g2_spart % 100 if g2_spart > 0 else 0
    if g2_spart > 0 and (g2_group_num < 1 or g2_m01 < 1):
        raise AssertionError(
            f"g2_spart={g2_spart} must encode GroupNum>=1,M01>=1 as GroupNum*100+M01 (e.g. 402)"
        )
    g2_bf16_lds = (True if g2_bf16_lds is None else bool(g2_bf16_lds)) and use_reduce
    return g2_bhoist, g2_ascale_pf, g2_spart, g2_group_num, g2_m01, g2_bf16_lds
