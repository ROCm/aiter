# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
"""Static MegaMoEV2 configurations tuned for eight-GPU MI355X."""

from bisect import bisect_left
from dataclasses import dataclass, replace
from functools import lru_cache

TOKEN_BUCKETS = (1, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768)
P2P_FP8_MIN_TOKENS = 1024
FIXED_SLOT_MAX_MTPR = 255


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
    active_expert_producer: bool = False
    cooperative_payload_copy: bool = False
    work_shards: int = 8
    external_grouping: bool = False
    external_counting: bool = False


@dataclass(frozen=True, slots=True)
class Stage2Config:
    block_m: int
    block_n: int
    persist: bool
    persist_cu: int
    use_nt: bool
    persist_strided: bool = False
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
            raise ValueError(f"Stage2 block_m={bm} must divide Stage1 sort_block_m={sbm}")
        if self.p2p_quant not in ("none", "fp8_blockwise_1x32"):
            raise ValueError(f"unsupported p2p_quant={self.p2p_quant!r}")
        if self.p2p_quant != "none" and self.stage2.bf16_lds:
            raise ValueError("FP8 P2P requires Stage2 bf16_lds=False")


_FIXED_GEOMETRY = {
    1: (1, 64, True, 2, 0),
    4: (1, 128, True, 2, 3),
    8: (2, 128, True, 2, 3),
    16: (4, 96, True, 1, 3),
    32: (3, 128, False, 2, 3),
    64: (3, 208, False, 2, 3),
    128: (3, 224, False, 2, 3),
}

_COMPACT_SMALL_DISPATCH_CU = {1: 224, 4: 128, 8: 192, 16: 64, 32: 128, 64: 192, 128: 128}


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


def _select_stage1(bucket: int, fixed_slot: bool, mtpr: int) -> Stage1Config:
    if fixed_slot:
        grid_mult, dispatch_cu, tile_resource, waves_per_eu, b_nt = _FIXED_GEOMETRY[bucket]
        config = Stage1Config(
            sort_block_m=32,
            tile_n=256 if bucket <= 8 else 128,
            num_waves=4,
            grid_mult=grid_mult,
            num_dispatch_cu=dispatch_cu,
            mfma_amajor=False,
            async_a_copy=False,
            use_tile_resource=tile_resource,
            b_nt=b_nt,
            waves_per_eu_hint=waves_per_eu,
        )
    elif bucket <= 4:
        config = Stage1Config(
            sort_block_m=32,
            tile_n=256,
            num_waves=4,
            grid_mult=1,
            num_dispatch_cu=_COMPACT_SMALL_DISPATCH_CU[bucket],
            mfma_amajor=False,
            async_a_copy=False,
            use_tile_resource=False,
            b_nt=0 if bucket == 1 else 3,
        )
    elif bucket <= 128:
        config = Stage1Config(
            sort_block_m=32,
            tile_n=512,
            num_waves=8,
            grid_mult=1,
            num_dispatch_cu=_COMPACT_SMALL_DISPATCH_CU[bucket],
            mfma_amajor=True,
            async_a_copy=True,
            use_tile_resource=False,
            b_nt=3,
        )
    elif bucket <= 1024:
        config = Stage1Config(
            sort_block_m=64,
            tile_n=512,
            num_waves=8,
            grid_mult=1 if bucket == 256 else 2,
            num_dispatch_cu=160 if bucket == 256 else 128,
            mfma_amajor=True,
            async_a_copy=True,
            use_tile_resource=bucket == 256,
            b_nt=3 if bucket <= 512 else 0,
        )
    elif bucket == 2048:
        config = Stage1Config(
            sort_block_m=64,
            tile_n=512,
            num_waves=8,
            grid_mult=1,
            num_dispatch_cu=32,
            mfma_amajor=True,
            async_a_copy=False,
            use_tile_resource=True,
            b_nt=0,
        )
    else:
        config = Stage1Config(
            sort_block_m=128,
            tile_n=512,
            num_waves=8,
            grid_mult=1,
            num_dispatch_cu=32,
            mfma_amajor=True,
            async_a_copy=True,
            use_tile_resource=bucket >= 16384,
            b_nt=0,
        )
    external_grouping = not fixed_slot and mtpr >= 2048
    return replace(
        config,
        work_shards=4 if mtpr >= 8192 else 8,
        external_grouping=external_grouping,
        external_counting=external_grouping and mtpr >= 8192,
    )


def _select_stage2(bucket: int, fixed_slot: bool) -> Stage2Config:
    block_m = 64 if bucket >= 4096 else 32
    block_n = 256 if bucket in (1, 4, 64) or bucket >= 1024 or (not fixed_slot and bucket < 128) else 128
    persist = bucket >= 128
    persist_cu = 0
    if persist:
        persist_cu = 128 if bucket == 256 else 256 if bucket in (4096, 16384) else 240
    return Stage2Config(
        block_m=block_m,
        block_n=block_n,
        persist=persist,
        persist_cu=persist_cu,
        use_nt=bucket <= 128,
        persist_strided=bucket in (512, 1024, 2048),
    )


@lru_cache(maxsize=None)
def _select_bucket_config(bucket: int, mtpr: int, p2p_quant: str) -> MegaMoEConfig:
    fixed_slot = mtpr <= FIXED_SLOT_MAX_MTPR
    stage1 = _select_stage1(bucket, fixed_slot, mtpr)
    stage2 = _select_stage2(bucket, fixed_slot)
    return MegaMoEConfig(stage1=stage1, stage2=stage2, p2p_quant=p2p_quant)


def select_mega_moe_config(tokens: int, mtpr: int) -> MegaMoEConfig:
    if mtpr <= 0 or mtpr & (mtpr - 1):
        raise ValueError(f"mtpr={mtpr} must be a positive power of two")
    if tokens > mtpr:
        raise ValueError(f"tokens={tokens} exceeds mtpr={mtpr}")
    bucket = nearest_token_bucket(tokens)
    fixed_slot = mtpr <= FIXED_SLOT_MAX_MTPR
    if fixed_slot and bucket not in _FIXED_GEOMETRY:
        raise ValueError(f"fixed-slot does not support token bucket {bucket}")
    p2p_quant = "fp8_blockwise_1x32" if tokens > P2P_FP8_MIN_TOKENS else "none"
    return _select_bucket_config(bucket, mtpr, p2p_quant)
