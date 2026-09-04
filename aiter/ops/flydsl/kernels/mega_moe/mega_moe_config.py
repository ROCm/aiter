# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
"""Static MegaMoEV2 configuration rules for MI355X."""

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
FIXED_GRID_MULT_VALUES = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32)
P2P_FP8_MIN_MTPR = 1024
FIXED_SLOT_MAX_MTPR = 255
MAX_MTPR_CLASS = 32768
# Source-indexed payload storage cuts the maximum-capacity activation buffer
# by roughly ``topk``.  Keep every smaller capacity on the historical layout.
INDEXED_PAYLOAD_MIN_MTPR = MAX_MTPR_CLASS
INDEXED_PAYLOAD_MIN_SBM = 128
REFERENCE_EXPERTS_PER_RANK = 48
EXPERT_CONFIG_GRANULARITY = 64
# Compact route metadata dedicates ten bits to the global expert/group segment.
# Under the EP8 protocol this admits 8 * 127 expert segments plus 8 group
# segments.  The next expert would require segment 1024 and cannot be encoded.
MAX_FANOUT_SEGMENTS = 1024
MAX_FANOUT_EXPERTS_PER_RANK = 256


def fixed_stage1_epoch_slot(grid_mult: int, num_dispatch_cu: int, num_cu: int) -> int:
    """Return a collision-free fixed-slot epoch counter for one launch geometry."""
    if grid_mult not in FIXED_GRID_MULT_VALUES:
        raise ValueError(f"unsupported fixed-slot grid multiplier {grid_mult}")
    if not 0 < num_dispatch_cu < num_cu:
        raise ValueError(f"num_dispatch_cu={num_dispatch_cu} must be in [1, {num_cu})")
    return FIXED_GRID_MULT_VALUES.index(grid_mult) * (num_cu + 1) + num_dispatch_cu


def fixed_stage1_epoch_slot_count(num_cu: int) -> int:
    if num_cu <= 0:
        raise ValueError(f"num_cu must be positive, got {num_cu}")
    return len(FIXED_GRID_MULT_VALUES) * (num_cu + 1)


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
    payload_tile_ready: bool = False
    prepare_cu: int = 32
    prepare_quant_cu: int = 64


def stage1_bundle_identity(config: Stage1Config) -> Stage1Config:
    """Return the Stage1 kernel identity without prepare-only launch knobs."""
    return replace(config, prepare_cu=0, prepare_quant_cu=0)


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
    aligned_pair: bool = False
    pair_cu: int = 0
    pair_block_m: int = 32
    pair_block_n: int = 256
    pair_work_weight: int = 2
    pair_scatter_vec: int = 16


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


@dataclass(frozen=True, slots=True)
class Stage2BundleKey:
    """Stage2 compile identity, including the Stage1 wire-layout contract."""

    config: Stage2Config
    sbm: int
    p2p_quant: str
    fixed_slot_dispatch: bool

    def __post_init__(self):
        if self.config.block_m > self.sbm or self.sbm % self.config.block_m:
            raise ValueError(
                f"Stage2 block_m={self.config.block_m} must divide bundle SBM={self.sbm}"
            )
        if self.p2p_quant not in ("none", "fp8_blockwise_1x32"):
            raise ValueError(f"unsupported p2p_quant={self.p2p_quant!r}")


@dataclass(frozen=True, slots=True)
class MegaMoEBundleEntry:
    pair_id: int
    token_bucket: int
    config: MegaMoEConfig
    stage1_variant_id: int
    stage2_variant_id: int


@dataclass(frozen=True, slots=True)
class MegaMoEBundlePlan:
    mtpr: int
    fixed_slot_dispatch: bool
    entries: tuple[MegaMoEBundleEntry, ...]
    stage1_variants: tuple[Stage1Config, ...]
    stage2_variants: tuple[Stage2BundleKey, ...]

    def entry_for_tokens(self, tokens: int) -> MegaMoEBundleEntry:
        if tokens < 0 or tokens > self.mtpr:
            raise ValueError(f"tokens={tokens} must be in [0, {self.mtpr}]")
        # Empty DP ranks must still launch the same collective MegaMoE protocol
        # as non-empty ranks.  Use the smallest bundle geometry for that rank;
        # returning early would strand peers that have already entered dispatch.
        bucket = TOKEN_BUCKETS[0] if tokens == 0 else nearest_token_bucket(tokens)
        for entry in self.entries:
            if entry.token_bucket == bucket:
                return entry
        raise ValueError(
            f"token bucket {bucket} is not present in the mtpr={self.mtpr} bundle"
        )


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

    # Compact dispatch uses one preplanned protocol at every bounded size.
    # A 256-row payload chunk needs at most four producer CTAs per peer, so
    # 32 producer CTAs cover all eight peers without an idle prefix ahead of
    # the queued GEMM consumers.
    dispatch_cu = 32
    grid_mult = 1
    tile_resource = True
    b_nt = 0
    return Stage1Config(
        sort_block_m=sort_block_m,
        tile_n=tile_n,
        num_waves=num_waves,
        grid_mult=grid_mult,
        num_dispatch_cu=dispatch_cu,
        mfma_amajor=mfma_amajor,
        async_a_copy=async_a_copy,
        use_tile_resource=tile_resource,
        b_nt=b_nt,
        work_shards=4,
        payload_chunk_rows=256,
        payload_tile_ready=True,
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
    # Quant work grows linearly with tokens.  Keep it below prepare's critical
    # path without occupying every CU: 64 through 8K, 96 at 16K, 192 at 32K.
    prepare_quant_cu = max(64, min(192, (3 * bucket) // 512))
    return Stage1Config(
        sort_block_m=sort_block_m,
        tile_n=tile_n,
        num_waves=num_waves,
        grid_mult=1,
        num_dispatch_cu=32,
        mfma_amajor=mfma_amajor,
        async_a_copy=async_a_copy,
        use_tile_resource=True,
        b_nt=3 if 1 < bucket <= 256 else 0,
        work_shards=work_shards,
        payload_chunk_rows=384,
        payload_tile_ready=True,
        prepare_quant_cu=prepare_quant_cu,
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
    if not persist:
        persist_cu = 0
    elif bucket == 256:
        persist_cu = 128
    elif bucket == 1024:
        persist_cu = 256
    else:
        persist_cu = 240
    return Stage2Config(
        block_m=64 if bucket >= 4096 else 32,
        block_n=block_n,
        persist=persist,
        persist_cu=persist_cu,
        use_nt=bucket <= 128,
        persist_strided=512 <= bucket <= 2048,
    )


def _select_large_stage2(
    bucket: int, sort_block_m: int, model_dim: int
) -> Stage2Config:
    if bucket in (1024, 2048):
        persist_cu = 256
    elif bucket == 16384:
        persist_cu = 192
    else:
        persist_cu = 240
    block_n = 128 if bucket == 256 or model_dim < 4096 else 256
    aligned_pair = bucket == 8192
    return Stage2Config(
        block_m=64 if sort_block_m == 128 else 32,
        block_n=block_n,
        persist=True,
        persist_cu=persist_cu,
        use_nt=bucket <= 128,
        persist_strided=512 <= bucket <= 2048,
        skew_cu=112 if aligned_pair else 96 if bucket >= 512 else 0,
        aligned_pair=aligned_pair,
        pair_cu=722 if aligned_pair else 0,
    )


@cache
def _select_bucket_config(
    bucket: int,
    mtpr_class: int,
    experts_per_rank: int,
    model_dim: int,
    inter_dim: int,
    fixed_slot_dispatch: bool,
) -> MegaMoEConfig:
    if mtpr_class == MAX_MTPR_CLASS:
        stage1 = _select_large_stage1(bucket, experts_per_rank, inter_dim)
        stage2 = _select_large_stage2(bucket, stage1.sort_block_m, model_dim)
        return MegaMoEConfig(
            stage1=stage1, stage2=stage2, p2p_quant="fp8_blockwise_1x32"
        )

    fixed_slot = fixed_slot_dispatch
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
    world_size: int = 8,
) -> MegaMoEConfig:
    if mtpr <= 0 or mtpr & (mtpr - 1):
        raise ValueError(f"mtpr={mtpr} must be a positive power of two")
    if tokens > mtpr:
        raise ValueError(f"tokens={tokens} exceeds mtpr={mtpr}")
    if experts_per_rank <= 0:
        raise ValueError(f"experts_per_rank must be positive, got {experts_per_rank}")
    if not 0 < world_size <= 8:
        raise ValueError(f"world_size must be in [1, 8], got {world_size}")
    if model_dim <= 0 or inter_dim <= 0:
        raise ValueError(f"invalid model shape {model_dim}x{inter_dim}")
    if experts_per_rank > MAX_FANOUT_EXPERTS_PER_RANK:
        raise ValueError(
            "MegaMoE v2 fanout pair ids support at most "
            f"{MAX_FANOUT_EXPERTS_PER_RANK} experts per rank"
        )
    bucket = nearest_token_bucket(tokens)
    mtpr_class = mtpr_config_class(mtpr)
    fixed_slot_dispatch = (
        mtpr_class <= FIXED_SLOT_MAX_MTPR
        and world_size == 8
        and experts_per_rank == REFERENCE_EXPERTS_PER_RANK
    )
    if fixed_slot_dispatch and bucket > 128:
        raise ValueError(f"fixed-slot does not support token bucket {bucket}")
    total_segments = world_size * experts_per_rank + world_size
    if total_segments > MAX_FANOUT_SEGMENTS:
        raise ValueError(
            f"MegaMoE v2 fanout needs {total_segments} segments, exceeding "
            f"the {MAX_FANOUT_SEGMENTS}-segment route metadata limit"
        )
    return _select_bucket_config(
        bucket,
        mtpr_class,
        expert_config_class(experts_per_rank),
        model_dim,
        inter_dim,
        fixed_slot_dispatch,
    )


@cache
def build_mega_moe_bundle_plan(
    mtpr: int,
    *,
    experts_per_rank: int = REFERENCE_EXPERTS_PER_RANK,
    model_dim: int = 7168,
    inter_dim: int = 3072,
    world_size: int = 8,
) -> MegaMoEBundlePlan:
    """Deduplicate variants while keeping Stage1/Stage2 selection atomic."""
    if mtpr <= 0 or mtpr & (mtpr - 1):
        raise ValueError(f"mtpr={mtpr} must be a positive power of two")
    buckets = tuple(bucket for bucket in TOKEN_BUCKETS if bucket <= mtpr)
    if not buckets or buckets[-1] != mtpr:
        raise ValueError(f"mtpr={mtpr} has no exact token bucket")

    fixed_slot_dispatch = (
        mtpr <= FIXED_SLOT_MAX_MTPR
        and world_size == 8
        and experts_per_rank == REFERENCE_EXPERTS_PER_RANK
    )
    stage1_variants: list[Stage1Config] = []
    stage2_variants: list[Stage2BundleKey] = []
    stage1_ids: dict[Stage1Config, int] = {}
    stage2_ids: dict[Stage2BundleKey, int] = {}
    entries: list[MegaMoEBundleEntry] = []
    for bucket in buckets:
        config = select_mega_moe_config(
            bucket,
            mtpr,
            experts_per_rank=experts_per_rank,
            model_dim=model_dim,
            inter_dim=inter_dim,
            world_size=world_size,
        )
        stage1_key = stage1_bundle_identity(config.stage1)
        stage1_id = stage1_ids.setdefault(stage1_key, len(stage1_variants))
        if stage1_id == len(stage1_variants):
            stage1_variants.append(config.stage1)
        stage2_key = Stage2BundleKey(
            config.stage2,
            config.stage1.sort_block_m,
            config.p2p_quant,
            fixed_slot_dispatch,
        )
        stage2_id = stage2_ids.setdefault(stage2_key, len(stage2_variants))
        if stage2_id == len(stage2_variants):
            stage2_variants.append(stage2_key)
        entries.append(
            MegaMoEBundleEntry(
                pair_id=len(entries),
                token_bucket=bucket,
                config=config,
                stage1_variant_id=stage1_id,
                stage2_variant_id=stage2_id,
            )
        )
    return MegaMoEBundlePlan(
        mtpr=mtpr,
        fixed_slot_dispatch=fixed_slot_dispatch,
        entries=tuple(entries),
        stage1_variants=tuple(stage1_variants),
        stage2_variants=tuple(stage2_variants),
    )
