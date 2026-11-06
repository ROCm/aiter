# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
"""Launch configs and buffer helpers shared by the MegaMoEV2 small- and large-input kernels."""

from bisect import bisect_left
from dataclasses import dataclass
from functools import cache
from typing import NamedTuple

import flydsl.expr as fx
from flydsl.expr.typing import Vector as Vec

from ..tensor_shim import _fx_value_elem, buf_copy_load, buf_copy_store, ptr_buf_tensor
from .mega_moe_config import (
    LARGE_M_MTPRS,
    REFERENCE_EXPERTS_PER_RANK,
    _fixed_dispatch_cu,
)

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


def _select_fixed_stage1(bucket: int) -> Stage1Config:
    grid_mult = max(1, bucket // 4) if bucket <= 16 else 3
    return Stage1Config(
        sort_block_m=32,
        tile_n=256 if bucket <= 8 else 128,
        num_waves=4,
        grid_mult=grid_mult,
        num_dispatch_cu=_fixed_dispatch_cu(bucket),
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


def _large_config(model_dim: int, inter_dim: int) -> MegaMoEConfig:
    stage1 = Stage1Config(
        sort_block_m=128,
        tile_n=512 if inter_dim >= 2048 else 256,
        num_waves=8,
        grid_mult=1,
        num_dispatch_cu=96,
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
    bucket: int, mtpr: int, model_dim: int, inter_dim: int
) -> MegaMoEConfig:
    if mtpr in LARGE_M_MTPRS:
        return _large_config(model_dim, inter_dim)
    stage1 = _select_fixed_stage1(bucket)
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
    """Small-input (fixed-slot, MTPR <= 512) or large-input (MTPR in LARGE_M_MTPRS) config for ``tokens`` per rank."""
    if mtpr <= 0 or mtpr & (mtpr - 1):
        raise ValueError(f"mtpr={mtpr} must be a positive power of two")
    if mtpr > SMALL_M_FIXED_SLOT_MAX_MTPR and mtpr not in LARGE_M_MTPRS:
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
        model_dim,
        inter_dim,
    )


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


def load_at(buf, index, elem=fx.Int32, width=1, cache_modifier=0):
    """Load ``width`` x ``elem`` at element ``index`` of ``buf``."""
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
