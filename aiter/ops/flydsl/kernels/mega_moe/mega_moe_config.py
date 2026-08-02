# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
"""Static MegaMoEV2 configurations tuned for eight-GPU MI355X."""

from bisect import bisect_left
from dataclasses import dataclass, replace
from functools import cache

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
P2P_FP8_MIN_MTPR = 1024
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
            raise ValueError(
                f"Stage2 block_m={bm} must divide Stage1 sort_block_m={sbm}"
            )
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

_COMPACT_SMALL_DISPATCH_CU = {
    1: 224,
    4: 128,
    8: 192,
    16: 64,
    32: 128,
    64: 192,
    128: 128,
}

_OVERSIZED_SMALL_DISPATCH_CU = {
    4096: {4: 224, 8: 128},
    16384: {1: 128, 4: 224, 8: 64, 64: 64},
    32768: {4: 224, 8: 128, 128: 64},
}

# work shards, external grouping, external counting
_OVERSIZED_SMALL_PROTOCOL = {
    2048: {
        4: (8, False, False),
        32: (8, False, False),
        128: (8, False, False),
    },
    4096: {
        1: (1, False, False),
        32: (8, False, False),
        64: (1, False, False),
        128: (8, False, False),
    },
    8192: {
        1: (1, False, False),
        4: (1, True, False),
        8: (1, False, False),
        16: (1, False, False),
        32: (1, False, False),
        64: (4, False, False),
        128: (4, False, False),
    },
    16384: {
        1: (1, True, False),
        4: (8, False, False),
        8: (2, True, False),
        16: (2, True, False),
        32: (4, False, False),
        64: (1, False, False),
        128: (4, False, False),
    },
    32768: {
        1: (1, False, False),
        4: (1, True, False),
        8: (1, True, False),
        16: (1, False, False),
        32: (2, False, False),
        64: (4, False, False),
        128: (4, False, False),
    },
}


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
        grid_mult, dispatch_cu, tile_resource, waves_per_eu, b_nt = _FIXED_GEOMETRY[
            bucket
        ]
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

    oversized_capacity = not fixed_slot and mtpr > bucket
    if oversized_capacity:
        dispatch_cu = _OVERSIZED_SMALL_DISPATCH_CU.get(mtpr, {}).get(bucket)
        if dispatch_cu is not None:
            config = replace(config, num_dispatch_cu=dispatch_cu)
        elif bucket == 32:
            config = replace(
                config, num_dispatch_cu=192 if mtpr in (4096, 16384) else 64
            )
        elif bucket == 64:
            config = replace(config, num_dispatch_cu=160)
        elif bucket == 128:
            config = replace(config, num_dispatch_cu=192)
        elif bucket == 256 and mtpr >= 16384:
            config = replace(config, sort_block_m=32, grid_mult=1, num_dispatch_cu=64)
        elif bucket == 512:
            config = replace(
                config,
                sort_block_m=64,
                grid_mult=2 if mtpr >= 32768 else 1,
                num_dispatch_cu=64,
                use_tile_resource=mtpr != 4096,
                b_nt=3 if mtpr >= 16384 else 0,
            )
        elif bucket == 1024:
            config = replace(
                config,
                sort_block_m=64,
                grid_mult=1,
                num_dispatch_cu=64,
                use_tile_resource=True,
                b_nt=0,
            )
        elif bucket == 2048 and mtpr >= 4096:
            config = replace(
                config,
                grid_mult=1 if mtpr >= 16384 else 2,
                num_dispatch_cu=64,
                async_a_copy=mtpr != 8192,
            )
        elif bucket == 4096:
            config = replace(
                config,
                grid_mult=1 if mtpr == 8192 else 2,
                num_dispatch_cu=64 if mtpr == 8192 else 32,
                use_tile_resource=mtpr >= 16384,
            )

    if mtpr >= 16384:
        config = replace(config, use_tile_resource=True)
    external_grouping = not fixed_slot and mtpr >= 2048
    work_shards = 4 if mtpr >= 8192 or (bucket == 1024 and mtpr >= 4096) else 8
    external_counting = external_grouping and (
        mtpr >= 8192 or (bucket == 1024 and mtpr >= 4096)
    )
    if oversized_capacity and bucket <= 128:
        work_shards, external_grouping, external_counting = (
            _OVERSIZED_SMALL_PROTOCOL.get(mtpr, {}).get(
                bucket, (work_shards, external_grouping, external_counting)
            )
        )
    if bucket == 2048 and mtpr == 8192:
        work_shards = 2
    elif bucket == 4096 and mtpr == 16384:
        work_shards, external_counting = 1, False
    return replace(
        config,
        work_shards=work_shards,
        external_grouping=external_grouping,
        external_counting=external_counting,
    )


def _select_stage2(
    bucket: int, fixed_slot: bool, mtpr: int, sort_block_m: int
) -> Stage2Config:
    if not fixed_slot and mtpr > bucket:
        if bucket == 128 and mtpr == 4096:
            persist_cu = 128
        elif bucket == 256 and mtpr >= 32768:
            persist_cu = 256
        else:
            persist_cu = {1024: 224, 2048: 256, 8192: 256, 16384: 192}.get(bucket, 240)
        return Stage2Config(
            block_m=64 if sort_block_m == 128 else 32,
            block_n=128 if bucket == 256 and sort_block_m == 64 else 256,
            persist=True,
            persist_cu=persist_cu,
            use_nt=bucket <= 128,
            persist_strided=bucket in (512, 1024, 2048),
        )
    block_m = 64 if bucket >= 4096 else 32
    block_n = (
        256
        if bucket in (1, 4, 64) or bucket >= 1024 or (not fixed_slot and bucket < 128)
        else 128
    )
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


@cache
def _select_bucket_config(bucket: int, mtpr: int) -> MegaMoEConfig:
    fixed_slot = mtpr <= FIXED_SLOT_MAX_MTPR
    stage1 = _select_stage1(bucket, fixed_slot, mtpr)
    stage2 = _select_stage2(bucket, fixed_slot, mtpr, stage1.sort_block_m)
    # MTPR is rank-invariant; local token counts need not be.
    p2p_quant = "fp8_blockwise_1x32" if mtpr > P2P_FP8_MIN_MTPR else "none"
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
    return _select_bucket_config(bucket, mtpr)
