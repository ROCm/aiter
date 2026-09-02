# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
"""Static MegaMoEV2 configuration rules for MI355X."""

from bisect import bisect_left
from dataclasses import dataclass
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
MAX_MTPR_CLASS = 32768
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
    external_grouping: bool = False
    external_counting: bool = False
    payload_chunk_rows: int = 0
    payload_tile_ready: bool = False


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


def mtpr_config_class(mtpr: int) -> int:
    return mtpr if mtpr <= P2P_FP8_MIN_MTPR else MAX_MTPR_CLASS


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


def _compact_dispatch_cu(bucket: int) -> int:
    if bucket <= 1:
        return 224
    if bucket <= 4:
        return 128
    if bucket <= 8:
        return 192
    if bucket <= 16:
        return 64
    if bucket <= 32:
        return 128
    if bucket <= 64:
        return 192
    return 128


def _large_dispatch_cu(bucket: int) -> int:
    if bucket <= 1:
        return 224
    if bucket <= 4:
        return 128
    if bucket <= 8:
        return 192
    if bucket <= 32:
        return 64
    if bucket <= 64:
        return 160
    if bucket <= 128:
        return 192
    if bucket <= 256:
        return 160
    if bucket == 8192:
        return 96
    if bucket >= 16384:
        return 32
    return 64


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


def _select_bounded_stage1(
    bucket: int, mtpr: int, experts_per_rank: int, inter_dim: int
) -> Stage1Config:
    if bucket <= 4:
        sort_block_m, tile_n, num_waves = 32, 256, 4
        grid_mult, mfma_amajor, async_a_copy = 1, False, False
    elif bucket <= 128:
        sort_block_m = 32
        tile_n, num_waves = (512 if inter_dim >= 2048 else 256), 8
        grid_mult, mfma_amajor, async_a_copy = 1, True, True
    elif bucket <= 1024:
        sort_block_m = 64
        tile_n, num_waves = (512 if inter_dim >= 2048 else 256), 8
        grid_mult, mfma_amajor, async_a_copy = (1 if bucket == 256 else 2), True, True
    else:
        raise ValueError(f"bounded MTPR does not support token bucket {bucket}")

    dispatch_cu = (
        _compact_dispatch_cu(bucket) if bucket <= 128 else 160 if bucket == 256 else 128
    )
    tile_resource = bucket == 256
    b_nt = 0 if bucket == 1 or bucket >= 1024 else 3
    if mtpr > bucket:
        if bucket == 32:
            dispatch_cu = 64
        elif bucket == 64:
            dispatch_cu = 160
        elif bucket == 128:
            dispatch_cu = 192
        elif bucket == 512:
            grid_mult, dispatch_cu, tile_resource, b_nt = 1, 64, True, 0
    return Stage1Config(
        sort_block_m=sort_block_m,
        tile_n=tile_n,
        num_waves=num_waves,
        grid_mult=grid_mult,
        num_dispatch_cu=_scale_dispatch_cu(dispatch_cu, experts_per_rank),
        mfma_amajor=mfma_amajor,
        async_a_copy=async_a_copy,
        use_tile_resource=tile_resource,
        b_nt=b_nt,
    )


def _select_large_stage1(
    bucket: int, experts_per_rank: int, inter_dim: int
) -> Stage1Config:
    if bucket <= 4:
        sort_block_m, tile_n, num_waves = 32, 256, 4
        mfma_amajor, async_a_copy = False, False
    elif bucket <= 128:
        sort_block_m = 32
        tile_n, num_waves = (512 if inter_dim >= 2048 else 256), 8
        mfma_amajor, async_a_copy = True, True
    elif bucket <= 2048:
        sort_block_m = 64
        tile_n, num_waves = (512 if inter_dim >= 2048 else 256), 8
        mfma_amajor, async_a_copy = True, True
    else:
        sort_block_m = 128
        tile_n, num_waves = (512 if inter_dim >= 2048 else 256), 8
        mfma_amajor, async_a_copy = True, True

    work_shards = 1 if bucket <= 32 else 4
    if bucket == 2048:
        work_shards = 8
    return Stage1Config(
        sort_block_m=sort_block_m,
        tile_n=tile_n,
        num_waves=num_waves,
        grid_mult=1,
        num_dispatch_cu=_scale_dispatch_cu(
            _large_dispatch_cu(bucket), experts_per_rank
        ),
        mfma_amajor=mfma_amajor,
        async_a_copy=async_a_copy,
        use_tile_resource=True,
        b_nt=3 if 1 < bucket <= 256 else 0,
        work_shards=work_shards,
        external_grouping=bucket == 4 or bucket >= 256,
        external_counting=bucket >= 256,
        payload_chunk_rows=384,
        payload_tile_ready=True,
    )


def _select_bounded_stage2(
    bucket: int, fixed_slot: bool, mtpr: int, sort_block_m: int, model_dim: int
) -> Stage2Config:
    if not fixed_slot and mtpr > bucket:
        return Stage2Config(
            block_m=64 if sort_block_m == 128 else 32,
            block_n=128 if bucket == 256 and sort_block_m == 64 else 256,
            persist=True,
            persist_cu=240,
            use_nt=bucket <= 128,
            persist_strided=512 <= bucket <= 2048,
        )
    block_n = (
        256
        if bucket in (1, 4, 64) or bucket >= 1024 or not fixed_slot and bucket < 128
        else 128
    )
    if model_dim < 4096:
        block_n = 128
    persist = bucket >= 128
    return Stage2Config(
        block_m=64 if bucket >= 4096 else 32,
        block_n=block_n,
        persist=persist,
        persist_cu=128 if bucket == 256 else 240 if persist else 0,
        use_nt=bucket <= 128,
        persist_strided=512 <= bucket <= 2048,
    )


def _select_large_stage2(
    bucket: int, sort_block_m: int, model_dim: int
) -> Stage2Config:
    if bucket == 1024:
        persist_cu = 224
    elif bucket == 2048:
        persist_cu = 256
    elif bucket == 16384:
        persist_cu = 192
    else:
        persist_cu = 240
    block_n = 128 if bucket == 256 or model_dim < 4096 else 256
    return Stage2Config(
        block_m=64 if sort_block_m == 128 else 32,
        block_n=block_n,
        persist=True,
        persist_cu=persist_cu,
        use_nt=bucket <= 128,
        persist_strided=512 <= bucket <= 2048,
        skew_cu=96 if bucket >= 512 else 0,
    )


@cache
def _select_bucket_config(
    bucket: int, mtpr_class: int, experts_per_rank: int, model_dim: int, inter_dim: int
) -> MegaMoEConfig:
    if mtpr_class == MAX_MTPR_CLASS:
        stage1 = _select_large_stage1(bucket, experts_per_rank, inter_dim)
        stage2 = _select_large_stage2(bucket, stage1.sort_block_m, model_dim)
        return MegaMoEConfig(
            stage1=stage1, stage2=stage2, p2p_quant="fp8_blockwise_1x32"
        )

    fixed_slot = mtpr_class <= FIXED_SLOT_MAX_MTPR
    if fixed_slot:
        stage1 = _select_fixed_stage1(bucket, experts_per_rank)
    else:
        stage1 = _select_bounded_stage1(bucket, mtpr_class, experts_per_rank, inter_dim)
    stage2 = _select_bounded_stage2(
        bucket, fixed_slot, mtpr_class, stage1.sort_block_m, model_dim
    )
    return MegaMoEConfig(stage1=stage1, stage2=stage2, p2p_quant="none")


def select_mega_moe_config(
    tokens: int,
    mtpr: int,
    *,
    experts_per_rank: int = REFERENCE_EXPERTS_PER_RANK,
    model_dim: int = 7168,
    inter_dim: int = 3072,
) -> MegaMoEConfig:
    if mtpr <= 0 or mtpr & (mtpr - 1):
        raise ValueError(f"mtpr={mtpr} must be a positive power of two")
    if tokens > mtpr:
        raise ValueError(f"tokens={tokens} exceeds mtpr={mtpr}")
    if experts_per_rank <= 0:
        raise ValueError(f"experts_per_rank must be positive, got {experts_per_rank}")
    if model_dim <= 0 or inter_dim <= 0:
        raise ValueError(f"invalid model shape {model_dim}x{inter_dim}")
    bucket = nearest_token_bucket(tokens)
    mtpr_class = mtpr_config_class(mtpr)
    if mtpr_class <= FIXED_SLOT_MAX_MTPR and bucket > 128:
        raise ValueError(f"fixed-slot does not support token bucket {bucket}")
    if mtpr_class <= FIXED_SLOT_MAX_MTPR and experts_per_rank > 64:
        raise ValueError("fixed-slot supports at most 64 experts per rank")
    return _select_bucket_config(
        bucket, mtpr_class, expert_config_class(experts_per_rank), model_dim, inter_dim
    )
