# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
"""Static MegaMoEV2 configuration rules for MI355X."""

from bisect import bisect_left
from dataclasses import dataclass, fields, replace
from enum import Enum, IntEnum
from functools import cache


class TokenBucket(IntEnum):
    BS1 = 1
    BS4 = 4
    BS8 = 8
    BS16 = 16
    BS32 = 32
    BS64 = 64
    BS128 = 128
    BS256 = 256
    BS512 = 512
    BS1024 = 1024
    BS2048 = 2048
    BS4096 = 4096
    BS8192 = 8192
    BS16384 = 16384
    BS32768 = 32768


TOKEN_BUCKETS = tuple(bucket.value for bucket in TokenBucket)
FIXED_GRID_MULT_VALUES = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32)
BOUNDED_COMPACT_MAX_MTPR = 1024
P2P_FP8_MIN_MTPR = BOUNDED_COMPACT_MAX_MTPR
FIXED_SLOT_MAX_MTPR = 255
MAX_MTPR_CLASS = 32768
# Source-indexed payload storage cuts the maximum-capacity activation buffer
# by roughly ``topk``.  Keep every smaller capacity on the historical layout.
INDEXED_PAYLOAD_MIN_MTPR = MAX_MTPR_CLASS
INDEXED_PAYLOAD_MIN_SBM = 128
REFERENCE_EXPERTS_PER_RANK = 48
# Compact route metadata dedicates ten bits to the global expert/group segment.
# Under the EP8 protocol this admits 8 * 127 expert segments plus 8 group
# segments.  The next expert would require segment 1024 and cannot be encoded.
MAX_FANOUT_SEGMENTS = 1024
MAX_FANOUT_EXPERTS_PER_RANK = 256


ACTIVATION_FP4 = "fp4"
ACTIVATION_FP8 = "fp8"
SUPPORTED_ACTIVATION_DTYPES = (ACTIVATION_FP4, ACTIVATION_FP8)

P2P_QUANT_AUTO = "auto"
P2P_QUANT_NONE = "none"
P2P_QUANT_FP8_BLOCKWISE = "fp8_blockwise_1x32"
SUPPORTED_P2P_QUANT_MODES = (P2P_QUANT_NONE, P2P_QUANT_FP8_BLOCKWISE)

FIXED_LARGE_CAPACITY_MTPR = 8192
R1_EXPERTS_PER_RANK = 32
BLOCK_M_SMALL = 32
BLOCK_M_MEDIUM = 64
BLOCK_M_LARGE = 128
TILE_N_NARROW = 128
TILE_N_BASE = 256
TILE_N_WIDE = 512
COMPACT_NUM_WAVES = 4
ASYNC_NUM_WAVES = 8
GRID_MULT_SINGLE_EPOCH = 1
B_NT_DISABLED = 0
B_NT_ENABLED = 3
WAVES_PER_EU_LOW = 1
WAVES_PER_EU_DEFAULT = 2
PERSIST_CU_DEFAULT = 240
WIDE_BATCH_MIN_TOKENS = 256
NO_EXACT_TOKEN_KEY = 0
EXACT_TUNING_TOKENS = frozenset((1, 2, 4, 16))


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
    prepare_quant_cu: int = 64


def stage1_bundle_identity(config: Stage1Config) -> Stage1Config:
    """Return the Stage1 kernel identity without prepare-only launch knobs."""
    return replace(config, prepare_quant_cu=0)


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
    b2stage: bool = True
    ascale_prefetch: bool = True
    spatial_partition: int = 402
    bf16_lds: bool = False
    aligned_pair: bool = False
    pair_cu: int = 0
    pair_block_m: int = 32
    pair_block_n: int = 256
    deep_a_pipeline: bool = False


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
        if self.p2p_quant not in SUPPORTED_P2P_QUANT_MODES:
            raise ValueError(f"unsupported p2p_quant={self.p2p_quant!r}")
        if self.p2p_quant != P2P_QUANT_NONE and self.stage2.bf16_lds:
            raise ValueError("FP8 P2P requires Stage2 bf16_lds=False")
        if self.stage2.deep_a_pipeline and not self.stage2.b2stage:
            raise ValueError("Stage2 deep_a_pipeline requires b2stage=True")


@dataclass(frozen=True, slots=True)
class Stage2BundleKey:
    """Stage2 compile identity, including the Stage1 wire-layout contract."""

    config: Stage2Config
    sbm: int
    p2p_quant: str

    def __post_init__(self):
        if self.config.block_m > self.sbm or self.sbm % self.config.block_m:
            raise ValueError(
                f"Stage2 block_m={self.config.block_m} must divide bundle SBM={self.sbm}"
            )
        if self.p2p_quant not in ("none", "fp8_blockwise_1x32"):
            raise ValueError(f"unsupported p2p_quant={self.p2p_quant!r}")


@dataclass(frozen=True, slots=True)
class MegaMoEBundleEntry:
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


def _select_bounded_stage1(bucket: int, inter_dim: int) -> Stage1Config:
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
    )


def _select_large_stage1(bucket: int, inter_dim: int) -> Stage1Config:
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
    model_dim: int,
    inter_dim: int,
    fixed_slot_dispatch: bool,
) -> MegaMoEConfig:
    if mtpr_class == MAX_MTPR_CLASS:
        stage1 = _select_large_stage1(bucket, inter_dim)
        stage2 = _select_large_stage2(bucket, stage1.sort_block_m, model_dim)
        return MegaMoEConfig(
            stage1=stage1, stage2=stage2, p2p_quant="fp8_blockwise_1x32"
        )

    if fixed_slot_dispatch:
        stage1 = _select_fixed_stage1(bucket)
    else:
        stage1 = _select_bounded_stage1(bucket, inter_dim)
    stage2 = _select_bounded_stage2(
        bucket, fixed_slot_dispatch, mtpr_class, stage1.sort_block_m, model_dim
    )
    return MegaMoEConfig(stage1=stage1, stage2=stage2, p2p_quant="none")


def select_mega_moe_config(
    tokens: int,
    mtpr: int,
    *,
    a_dtype: str = ACTIVATION_FP8,
    experts_per_rank: int = REFERENCE_EXPERTS_PER_RANK,
    model_dim: int = 7168,
    inter_dim: int = 3072,
    world_size: int = 8,
) -> MegaMoEConfig:
    """Return the fp8-safe base MegaMoE config (no A4W4 tuning patches)."""
    if a_dtype != ACTIVATION_FP8:
        raise ValueError(
            "select_mega_moe_config emits fp8-safe base configs only; "
            "use resolve_mega_moe_config() for fp4 (applies async-copy safety)."
        )
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
        model_dim,
        inter_dim,
        fixed_slot_dispatch,
    )


@cache
def build_mega_moe_bundle_plan(
    mtpr: int,
    *,
    a_dtype: str = ACTIVATION_FP8,
    p2p_quant: str = P2P_QUANT_AUTO,
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
        config = resolve_mega_moe_config(
            bucket,
            mtpr,
            p2p_quant,
            a_dtype=a_dtype,
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
        )
        stage2_id = stage2_ids.setdefault(stage2_key, len(stage2_variants))
        if stage2_id == len(stage2_variants):
            stage2_variants.append(stage2_key)
        entries.append(
            MegaMoEBundleEntry(
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


class CapacityMode(str, Enum):
    FIXED_SLOT = "fixed_slot"
    BOUNDED_COMPACT = "bounded_compact"
    LARGE_COMPACT = "large_compact"


@dataclass(frozen=True, slots=True)
class TuningContext:
    tokens: int
    bucket: int
    mtpr: int
    capacity_mode: CapacityMode
    a_dtype: str
    p2p_quant: str
    experts_per_rank: int


@dataclass(frozen=True, slots=True)
class ConfigPatch:
    stage1: tuple[tuple[str, object], ...] = ()
    stage2: tuple[tuple[str, object], ...] = ()


_STAGE1_PATCH_FIELDS = frozenset(field.name for field in fields(Stage1Config))
_STAGE2_PATCH_FIELDS = frozenset(field.name for field in fields(Stage2Config))


def capacity_mode_for_mtpr(mtpr: int) -> CapacityMode:
    if mtpr <= FIXED_SLOT_MAX_MTPR:
        return CapacityMode.FIXED_SLOT
    if mtpr <= BOUNDED_COMPACT_MAX_MTPR:
        return CapacityMode.BOUNDED_COMPACT
    return CapacityMode.LARGE_COMPACT


def _validate_mtpr(mtpr: int) -> None:
    if mtpr <= 0 or mtpr & (mtpr - 1):
        raise ValueError(f"mtpr={mtpr} must be a positive power of two")


def _validate_activation_dtype(a_dtype: str) -> None:
    if a_dtype not in SUPPORTED_ACTIVATION_DTYPES:
        raise ValueError(f"unsupported activation dtype={a_dtype!r}")


# ---- Sparse measured residuals ------------------------------------------------


def _patch(
    *,
    stage1: dict[str, object] | None = None,
    stage2: dict[str, object] | None = None,
) -> ConfigPatch:
    stage1 = stage1 or {}
    stage2 = stage2 or {}
    unknown_stage1 = stage1.keys() - _STAGE1_PATCH_FIELDS
    unknown_stage2 = stage2.keys() - _STAGE2_PATCH_FIELDS
    if unknown_stage1 or unknown_stage2:
        raise ValueError(
            f"invalid tuning patch fields: stage1={sorted(unknown_stage1)}, "
            f"stage2={sorted(unknown_stage2)}"
        )
    return ConfigPatch(
        tuple(stage1.items()),
        tuple(stage2.items()),
    )


# Sparse residuals measured on MI355X after the formula-derived base geometry.
_P2P_FP8_TUNED_BUCKETS = frozenset((TokenBucket.BS256, TokenBucket.BS512))


def _p2p_fp8_patch(bucket: int) -> ConfigPatch | None:
    if bucket not in _P2P_FP8_TUNED_BUCKETS:
        return None
    is_bs512 = bucket == TokenBucket.BS512
    return _patch(
        stage2={
            "block_m": BLOCK_M_MEDIUM,
            "block_n": TILE_N_BASE,
            "persist": True,
            "persist_cu": PERSIST_CU_DEFAULT if is_bs512 else 128,
            "use_nt": False,
            "persist_strided": is_bs512,
            "deep_a_pipeline": True,
        },
    )


def _fixed_slot_token_patch(a_dtype: str, tokens: int) -> ConfigPatch | None:
    if tokens == 2:
        grid_mult = 2
    elif a_dtype == ACTIVATION_FP4 and tokens == 4:
        grid_mult = 4
    elif a_dtype == ACTIVATION_FP4 and tokens == 16:
        grid_mult = 3
    else:
        return None
    return _patch(stage1={"grid_mult": grid_mult})


_A4_ASYNC_COPY_SAFETY_PATCH = _patch(stage1={"sort_block_m": BLOCK_M_MEDIUM})
_A4_WIDE_BATCH_OCCUPANCY_PATCH = _patch(stage1={"waves_per_eu_hint": WAVES_PER_EU_LOW})

_A4_BUCKET_PATCHES = {
    TokenBucket.BS512: _patch(stage1={"b_nt": B_NT_DISABLED, "num_dispatch_cu": 160}),
    TokenBucket.BS1024: _patch(
        stage1={"sort_block_m": BLOCK_M_LARGE, "num_dispatch_cu": 88},
        stage2={"block_m": BLOCK_M_MEDIUM},
    ),
    TokenBucket.BS4096: _patch(
        stage1={
            "num_dispatch_cu": 72,
        },  # payload NOT set: cross-rank, rank-invariant via base mtpr
    ),
    TokenBucket.BS8192: _patch(
        stage1={
            "grid_mult": 2,
            "num_dispatch_cu": 32,
            "swizzle_a": False,
        },  # payload NOT set: cross-rank, rank-invariant via base mtpr
    ),
}

_A4_FP8_P2P_BUCKET_PATCHES = {
    TokenBucket.BS1024: _patch(
        stage1={"grid_mult": GRID_MULT_SINGLE_EPOCH},
        stage2={"persist_cu": 224},
    )
}

_A4_FIXED8192_SINGLE_TOKEN_PATCH = _patch(stage1={"num_dispatch_cu": 160})
_A4_FIXED8192_TWO_TOKEN_PATCH = _patch(
    stage1={
        "num_dispatch_cu": 128,
        # payload NOT set: cross-rank, rank-invariant via base mtpr
    }
)
_A4_FIXED8192_COMPACT_DISPATCH_PATCH = _patch(stage1={"num_dispatch_cu": 32})
_A4_FIXED8192_BS8_STAGE2_PATCH = _patch(stage2={"block_m": BLOCK_M_MEDIUM})
_A4_FIXED8192_TUNED_BUCKETS = frozenset(
    (
        TokenBucket.BS4,
        TokenBucket.BS8,
        TokenBucket.BS16,
        TokenBucket.BS32,
        TokenBucket.BS64,
        TokenBucket.BS128,
        TokenBucket.BS256,
        TokenBucket.BS512,
        TokenBucket.BS4096,
        TokenBucket.BS8192,
    )
)


def _a4_fixed8192_dispatch_cu(bucket: int) -> int | None:
    """Return measured MI355X DCU optima without extrapolating to untested buckets."""
    if bucket == TokenBucket.BS8:
        return 32
    if bucket <= TokenBucket.BS256:
        return 160 - max(32, int(bucket) // 4)
    if bucket == TokenBucket.BS512:
        return 160
    if bucket == TokenBucket.BS4096:
        return 72
    if bucket == TokenBucket.BS8192:
        return 96
    return None


def _a4_fixed8192_tuned_patch(bucket: int) -> ConfigPatch | None:
    if bucket not in _A4_FIXED8192_TUNED_BUCKETS:
        return None
    dispatch_cu = _a4_fixed8192_dispatch_cu(bucket)
    assert dispatch_cu is not None
    # payload_chunk_rows/tile_ready NOT set: cross-rank, kept rank-invariant via base mtpr.
    stage1 = {
        "num_dispatch_cu": dispatch_cu,
    }
    if bucket == TokenBucket.BS256:
        stage1["tile_n"] = TILE_N_BASE
    elif bucket == TokenBucket.BS8192:
        stage1.update(
            grid_mult=GRID_MULT_SINGLE_EPOCH,
            swizzle_a=True,
            waves_per_eu_hint=WAVES_PER_EU_DEFAULT,
            # external_grouping/counting NOT set: cross-rank, must stay rank-invariant.
        )
    return _patch(stage1=stage1)


def _a4_fixed8192_sync_patch(grid_mult: int) -> ConfigPatch:
    return _patch(
        stage1={
            "sort_block_m": BLOCK_M_SMALL,
            "tile_n": TILE_N_BASE,
            "num_waves": COMPACT_NUM_WAVES,
            "grid_mult": grid_mult,
            "mfma_amajor": False,
            "async_a_copy": False,
        },
        stage2={"block_m": BLOCK_M_SMALL},
    )


_A4_FIXED8192_SYNC_BUCKETS = frozenset(
    (TokenBucket.BS16, TokenBucket.BS32, TokenBucket.BS64, TokenBucket.BS128)
)


def _apply_config_patch(config: MegaMoEConfig, patch: ConfigPatch) -> MegaMoEConfig:
    stage1_values = dict(patch.stage1)
    stage2_values = dict(patch.stage2)
    if not stage1_values and not stage2_values:
        return config
    return replace(
        config,
        stage1=(
            replace(config.stage1, **stage1_values) if stage1_values else config.stage1
        ),
        stage2=(
            replace(config.stage2, **stage2_values) if stage2_values else config.stage2
        ),
    )


_R1_SMALL_BS_SYNC_PATCH = _patch(
    stage1={
        "async_a_copy": False,
        "num_waves": ASYNC_NUM_WAVES,
        "sort_block_m": BLOCK_M_SMALL,
    },
)


def _select_tuning_patches(
    config: MegaMoEConfig,
    context: TuningContext,
) -> tuple[ConfigPatch, ...]:
    patches = []

    if context.capacity_mode == CapacityMode.FIXED_SLOT and (
        patch := _fixed_slot_token_patch(context.a_dtype, context.tokens)
    ):
        patches.append(patch)

    if context.a_dtype == ACTIVATION_FP8:
        return tuple(patches)

    if context.bucket <= TokenBucket.BS128 and config.stage1.async_a_copy:
        # FP4 halves the A K-step bytes. The 8-wave compact kernel therefore
        # needs SBM64 so every thread owns one or more 16-byte async copies.
        patches.append(_A4_ASYNC_COPY_SAFETY_PATCH)

    return tuple(patches)


def _apply_tuning_context(
    config: MegaMoEConfig,
    context: TuningContext,
) -> MegaMoEConfig:
    for tuning in _select_tuning_patches(config, context):
        config = _apply_config_patch(config, tuning)
    return config


@cache
def _resolve_tuned_config(
    config: MegaMoEConfig,
    bucket: int,
    token_key: int,
    mtpr: int,
    a_dtype: str,
    experts_per_rank: int,
) -> MegaMoEConfig:
    context = TuningContext(
        tokens=token_key,
        bucket=bucket,
        mtpr=mtpr,
        capacity_mode=capacity_mode_for_mtpr(mtpr),
        a_dtype=a_dtype,
        p2p_quant=config.p2p_quant,
        experts_per_rank=experts_per_rank,
    )
    return _apply_tuning_context(config, context)


def _tuning_token_key(tokens: int) -> int:
    if tokens in EXACT_TUNING_TOKENS:
        return tokens
    return (
        WIDE_BATCH_MIN_TOKENS if tokens >= WIDE_BATCH_MIN_TOKENS else NO_EXACT_TOKEN_KEY
    )


def resolve_mega_moe_config(
    tokens: int,
    mtpr: int,
    p2p_quant: str = P2P_QUANT_AUTO,
    *,
    a_dtype: str = ACTIVATION_FP8,
    experts_per_rank: int = REFERENCE_EXPERTS_PER_RANK,
    model_dim: int = 7168,
    inter_dim: int = 3072,
    world_size: int = 8,
) -> MegaMoEConfig:
    """Apply A4W4/fp4 measured patches on top of the main bundle base config."""
    _validate_activation_dtype(a_dtype)
    tune_tokens = TOKEN_BUCKETS[0] if tokens <= 0 else tokens
    config = select_mega_moe_config(
        tune_tokens,
        mtpr,
        a_dtype=ACTIVATION_FP8,
        experts_per_rank=experts_per_rank,
        model_dim=model_dim,
        inter_dim=inter_dim,
        world_size=world_size,
    )
    if p2p_quant == P2P_QUANT_AUTO:
        use_fp8 = (
            mtpr >= P2P_FP8_MIN_MTPR
            if a_dtype == ACTIVATION_FP4
            else mtpr > P2P_FP8_MIN_MTPR
        )
        desired = P2P_QUANT_FP8_BLOCKWISE if use_fp8 else P2P_QUANT_NONE
    elif p2p_quant in SUPPORTED_P2P_QUANT_MODES:
        desired = p2p_quant
    else:
        raise ValueError(f"unsupported p2p_quant={p2p_quant!r}")
    # Main already selects fp8 P2P for large MTPR. Honour explicit/auto overrides,
    # including fp4 at mtpr==1024 which prefers blockwise P2P.
    if desired != config.p2p_quant:
        config = replace(config, p2p_quant=desired)
    bucket = nearest_token_bucket(tune_tokens)
    return _resolve_tuned_config(
        config,
        bucket,
        _tuning_token_key(tune_tokens),
        mtpr,
        a_dtype,
        experts_per_rank,
    )
