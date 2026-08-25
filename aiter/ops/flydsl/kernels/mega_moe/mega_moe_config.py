# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
"""Static MegaMoE configurations tuned for MI355X multi-GPU execution."""

from bisect import bisect_left
from dataclasses import dataclass, fields, replace
from enum import Enum, IntEnum
from functools import cache

# ---- Domain vocabulary and hardware constants ---------------------------------


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

ACTIVATION_FP4 = "fp4"
ACTIVATION_FP8 = "fp8"
SUPPORTED_ACTIVATION_DTYPES = (ACTIVATION_FP4, ACTIVATION_FP8)

P2P_QUANT_AUTO = "auto"
P2P_QUANT_NONE = "none"
P2P_QUANT_FP8_BLOCKWISE = "fp8_blockwise_1x32"
SUPPORTED_P2P_QUANT_MODES = (P2P_QUANT_NONE, P2P_QUANT_FP8_BLOCKWISE)

BOUNDED_COMPACT_MAX_MTPR = 1024
P2P_FP8_MIN_MTPR = BOUNDED_COMPACT_MAX_MTPR
FIXED_SLOT_MAX_MTPR = 255
MAX_MTPR_CLASS = 32768
FIXED_LARGE_CAPACITY_MTPR = 8192

REFERENCE_EXPERTS_PER_RANK = 48
EXPERT_CONFIG_GRANULARITY = 64
GPU_WAVE_SIZE = 64
MAX_DISPATCH_CU = 224

BLOCK_M_SMALL = 32
BLOCK_M_MEDIUM = 64
BLOCK_M_LARGE = 128
TILE_N_NARROW = 128
TILE_N_BASE = 256
TILE_N_WIDE = 512
TILE_K_DEFAULT = 256

COMPACT_NUM_WAVES = 4
ASYNC_NUM_WAVES = 8
GRID_MULT_SINGLE_EPOCH = 1
B_NT_DISABLED = 0
B_NT_ENABLED = 3
WAVES_PER_EU_LOW = 1
WAVES_PER_EU_DEFAULT = 2
WORK_SHARDS_SINGLE = 1
WORK_SHARDS_GROUPED = 4
WORK_SHARDS_PIPELINED = 8

PERSIST_CU_DEFAULT = 240
LARGE_STAGE2_SKEW_CU = 96
STAGE2_SPATIAL_PARTITION_DEFAULT = 402

DEFAULT_MODEL_DIM = 7168
DEFAULT_INTER_DIM = 3072
WIDE_INTER_DIM_MIN = 2048
WIDE_MODEL_DIM_MIN = 4096

WIDE_BATCH_MIN_TOKENS = 256
NO_EXACT_TOKEN_KEY = 0
EXACT_TUNING_TOKENS = frozenset((1, 2, 4, 16))


# ---- Resolved kernel configuration types --------------------------------------


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
    waves_per_eu_hint: int = WAVES_PER_EU_DEFAULT
    tile_k: int = TILE_K_DEFAULT
    pipe_weights: bool = True
    swizzle_a: bool = True
    work_shards: int = WORK_SHARDS_PIPELINED
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
    block_k: int = TILE_K_DEFAULT
    b_hoist: bool = True
    b2stage: bool = True
    ascale_prefetch: bool = True
    spatial_partition: int = STAGE2_SPATIAL_PARTITION_DEFAULT
    bf16_lds: bool = False
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
            "payload_chunk_rows": 0,
            "payload_tile_ready": False,
        },
    ),
    TokenBucket.BS8192: _patch(
        stage1={
            "grid_mult": 2,
            "num_dispatch_cu": 32,
            "swizzle_a": False,
            "payload_chunk_rows": 0,
            "payload_tile_ready": False,
        },
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
        "payload_chunk_rows": 0,
        "payload_tile_ready": False,
    }
)
_A4_FIXED8192_COMPACT_DISPATCH_PATCH = _patch(stage1={"num_dispatch_cu": 32})
_A4_FIXED8192_BS8_STAGE2_PATCH = _patch(stage2={"block_m": BLOCK_M_MEDIUM})
A4_HOT_ROUTING_TOPK = 6
A4_HOT_ROUTING_TOTAL_EXPERTS = 384
A4_HOT_ROUTING_SKEW_FACTOR = 54
A4_HOT_PAYLOAD_CHUNK_ROWS = 384
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


def _a4_fixed8192_estimated_hot_rows(bucket: int) -> int:
    """Estimate rows owned by the hottest expert for V4-Pro-like routing."""
    numerator = int(bucket) * A4_HOT_ROUTING_TOPK * A4_HOT_ROUTING_SKEW_FACTOR
    return (
        numerator + A4_HOT_ROUTING_TOTAL_EXPERTS - 1
    ) // A4_HOT_ROUTING_TOTAL_EXPERTS


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
    use_chunk_pipeline = (
        _a4_fixed8192_estimated_hot_rows(bucket) >= A4_HOT_PAYLOAD_CHUNK_ROWS
    )
    stage1 = {
        "num_dispatch_cu": dispatch_cu,
        "payload_chunk_rows": A4_HOT_PAYLOAD_CHUNK_ROWS if use_chunk_pipeline else 0,
        "payload_tile_ready": use_chunk_pipeline,
    }
    if bucket == TokenBucket.BS256:
        stage1["tile_n"] = TILE_N_BASE
    elif bucket == TokenBucket.BS8192:
        stage1.update(
            grid_mult=GRID_MULT_SINGLE_EPOCH,
            swizzle_a=True,
            waves_per_eu_hint=WAVES_PER_EU_DEFAULT,
            external_grouping=False,
            external_counting=False,
        )
    return _patch(stage1=stage1)


def _a4_fixed8192_sync_patch(bucket: int, grid_mult: int) -> ConfigPatch:
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


# ---- Formula-derived base configuration ---------------------------------------


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
    return mtpr if mtpr <= BOUNDED_COMPACT_MAX_MTPR else MAX_MTPR_CLASS


def expert_config_class(experts_per_rank: int) -> int:
    return (
        (experts_per_rank + EXPERT_CONFIG_GRANULARITY - 1) // EXPERT_CONFIG_GRANULARITY
    ) * EXPERT_CONFIG_GRANULARITY


def _scale_dispatch_cu(dispatch_cu: int, experts_per_rank: int) -> int:
    expert_waves = (experts_per_rank + GPU_WAVE_SIZE - 1) // GPU_WAVE_SIZE
    return min(MAX_DISPATCH_CU, dispatch_cu * expert_waves)


_FIXED_DISPATCH_CU_SCALE = 16
_FIXED_DISPATCH_CU_BIT_OFFSET = 7
_FIXED_DISPATCH_CU_RESIDUALS = {
    TokenBucket.BS1: 64,
    TokenBucket.BS4: 128,
    TokenBucket.BS8: 128,
    TokenBucket.BS16: 96,
    TokenBucket.BS32: 128,
}


def _fixed_dispatch_cu(bucket: int) -> int:
    if residual := _FIXED_DISPATCH_CU_RESIDUALS.get(bucket):
        return residual
    return min(
        MAX_DISPATCH_CU,
        _FIXED_DISPATCH_CU_SCALE
        * (bucket.bit_length() + _FIXED_DISPATCH_CU_BIT_OFFSET),
    )


_COMPACT_DISPATCH_CU_DEFAULT = 128
_COMPACT_DISPATCH_CU_RESIDUALS = {
    TokenBucket.BS1: MAX_DISPATCH_CU,
    TokenBucket.BS8: 192,
    TokenBucket.BS16: 64,
    TokenBucket.BS64: 192,
}


def _compact_dispatch_cu(bucket: int) -> int:
    return _COMPACT_DISPATCH_CU_RESIDUALS.get(bucket, _COMPACT_DISPATCH_CU_DEFAULT)


_LARGE_DISPATCH_CU_DEFAULT = 64
_LARGE_DISPATCH_CU_RESIDUALS = {
    TokenBucket.BS1: MAX_DISPATCH_CU,
    TokenBucket.BS4: 128,
    TokenBucket.BS8: 192,
    TokenBucket.BS64: 160,
    TokenBucket.BS128: 192,
    TokenBucket.BS256: 160,
    TokenBucket.BS8192: 96,
    TokenBucket.BS16384: 32,
    TokenBucket.BS32768: 32,
}


def _large_dispatch_cu(bucket: int) -> int:
    return _LARGE_DISPATCH_CU_RESIDUALS.get(bucket, _LARGE_DISPATCH_CU_DEFAULT)


def _select_fixed_stage1(bucket: int, experts_per_rank: int) -> Stage1Config:
    grid_mult = (
        max(GRID_MULT_SINGLE_EPOCH, bucket // TokenBucket.BS4)
        if bucket <= TokenBucket.BS16
        else 3
    )
    return Stage1Config(
        sort_block_m=BLOCK_M_SMALL,
        tile_n=TILE_N_BASE if bucket <= TokenBucket.BS8 else TILE_N_NARROW,
        num_waves=COMPACT_NUM_WAVES,
        grid_mult=grid_mult,
        num_dispatch_cu=_scale_dispatch_cu(
            _fixed_dispatch_cu(bucket), experts_per_rank
        ),
        mfma_amajor=False,
        async_a_copy=False,
        use_tile_resource=bucket <= TokenBucket.BS16,
        b_nt=B_NT_DISABLED if bucket == TokenBucket.BS1 else B_NT_ENABLED,
        waves_per_eu_hint=(
            WAVES_PER_EU_LOW if bucket == TokenBucket.BS16 else WAVES_PER_EU_DEFAULT
        ),
    )


_BOUNDED_OVERCAPACITY_DISPATCH_BUCKETS = frozenset(
    (TokenBucket.BS32, TokenBucket.BS64, TokenBucket.BS128, TokenBucket.BS512)
)


def _derive_stage1_geometry(
    bucket: int,
    inter_dim: int,
    *,
    large_compact: bool,
) -> tuple[int, int, int, bool, bool]:
    if bucket <= TokenBucket.BS4:
        return BLOCK_M_SMALL, TILE_N_BASE, COMPACT_NUM_WAVES, False, False
    if large_compact and bucket > TokenBucket.BS2048:
        block_m = BLOCK_M_LARGE
    elif bucket > TokenBucket.BS128:
        block_m = BLOCK_M_MEDIUM
    else:
        block_m = BLOCK_M_SMALL
    tile_n = TILE_N_WIDE if inter_dim >= WIDE_INTER_DIM_MIN else TILE_N_BASE
    return block_m, tile_n, ASYNC_NUM_WAVES, True, True


def _select_bounded_stage1(
    bucket: int, mtpr: int, experts_per_rank: int, inter_dim: int
) -> Stage1Config:
    if bucket > TokenBucket.BS1024:
        raise ValueError(f"bounded MTPR does not support token bucket {bucket}")
    sort_block_m, tile_n, num_waves, mfma_amajor, async_a_copy = (
        _derive_stage1_geometry(
            bucket,
            inter_dim,
            large_compact=False,
        )
    )
    grid_mult = GRID_MULT_SINGLE_EPOCH if bucket <= TokenBucket.BS256 else 2

    dispatch_cu = (
        _compact_dispatch_cu(bucket)
        if bucket <= TokenBucket.BS128
        else _large_dispatch_cu(bucket) if bucket == TokenBucket.BS256 else 128
    )
    tile_resource = bucket == TokenBucket.BS256
    b_nt = (
        B_NT_DISABLED
        if bucket == TokenBucket.BS1 or bucket >= TokenBucket.BS1024
        else B_NT_ENABLED
    )
    if mtpr > bucket:
        if bucket in _BOUNDED_OVERCAPACITY_DISPATCH_BUCKETS:
            dispatch_cu = _large_dispatch_cu(bucket)
        if bucket == TokenBucket.BS512:
            grid_mult = GRID_MULT_SINGLE_EPOCH
            tile_resource, b_nt = True, B_NT_DISABLED
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


_LARGE_PAYLOAD_GEOMETRY_DEFAULT = (384, True)
_LARGE_PAYLOAD_GEOMETRY_BY_BUCKET = {
    TokenBucket.BS2048: (0, False),
    TokenBucket.BS16384: (1536, True),
    TokenBucket.BS32768: (768, True),
}


def _large_payload_geometry(bucket: int) -> tuple[int, bool]:
    return _LARGE_PAYLOAD_GEOMETRY_BY_BUCKET.get(
        bucket, _LARGE_PAYLOAD_GEOMETRY_DEFAULT
    )


def _select_large_stage1(
    bucket: int,
    experts_per_rank: int,
    inter_dim: int,
) -> Stage1Config:
    sort_block_m, tile_n, num_waves, mfma_amajor, async_a_copy = (
        _derive_stage1_geometry(
            bucket,
            inter_dim,
            large_compact=True,
        )
    )

    work_shards = (
        WORK_SHARDS_SINGLE if bucket <= TokenBucket.BS32 else WORK_SHARDS_GROUPED
    )
    if bucket == TokenBucket.BS2048:
        work_shards = WORK_SHARDS_PIPELINED
    payload_chunk_rows, payload_tile_ready = _large_payload_geometry(bucket)
    return Stage1Config(
        sort_block_m=sort_block_m,
        tile_n=tile_n,
        num_waves=num_waves,
        grid_mult=GRID_MULT_SINGLE_EPOCH,
        num_dispatch_cu=_scale_dispatch_cu(
            _large_dispatch_cu(bucket), experts_per_rank
        ),
        mfma_amajor=mfma_amajor,
        async_a_copy=async_a_copy,
        use_tile_resource=True,
        b_nt=(
            B_NT_ENABLED
            if TokenBucket.BS1 < bucket <= TokenBucket.BS256
            else B_NT_DISABLED
        ),
        work_shards=work_shards,
        external_grouping=bucket == TokenBucket.BS4 or bucket >= TokenBucket.BS256,
        external_counting=bucket >= TokenBucket.BS256,
        payload_chunk_rows=payload_chunk_rows,
        payload_tile_ready=payload_tile_ready,
    )


_BOUNDED_BASE_BLOCK_N_BUCKETS = frozenset(
    (TokenBucket.BS1, TokenBucket.BS4, TokenBucket.BS64)
)
_LARGE_PERSIST_CU_RESIDUALS = {
    TokenBucket.BS1024: 224,
    TokenBucket.BS16384: 192,
    TokenBucket.BS32768: 256,
}
_LARGE_STAGE2_NO_SKEW_BUCKETS = frozenset((TokenBucket.BS2048, TokenBucket.BS32768))


def _select_bounded_stage2(
    bucket: int, fixed_slot: bool, mtpr: int, sort_block_m: int, model_dim: int
) -> Stage2Config:
    if not fixed_slot and mtpr > bucket:
        return Stage2Config(
            block_m=BLOCK_M_MEDIUM if sort_block_m == BLOCK_M_LARGE else BLOCK_M_SMALL,
            block_n=(
                TILE_N_NARROW
                if bucket == TokenBucket.BS256 and sort_block_m == BLOCK_M_MEDIUM
                else TILE_N_BASE
            ),
            persist=True,
            persist_cu=PERSIST_CU_DEFAULT,
            use_nt=bucket <= TokenBucket.BS128,
            persist_strided=TokenBucket.BS512 <= bucket <= TokenBucket.BS2048,
        )
    block_n = (
        TILE_N_BASE
        if bucket in _BOUNDED_BASE_BLOCK_N_BUCKETS
        or bucket >= TokenBucket.BS1024
        or not fixed_slot
        and bucket < TokenBucket.BS128
        else TILE_N_NARROW
    )
    if model_dim < WIDE_MODEL_DIM_MIN:
        block_n = TILE_N_NARROW
    persist = bucket >= TokenBucket.BS128
    return Stage2Config(
        block_m=BLOCK_M_MEDIUM if bucket >= TokenBucket.BS4096 else BLOCK_M_SMALL,
        block_n=block_n,
        persist=persist,
        persist_cu=(
            128 if bucket == TokenBucket.BS256 else PERSIST_CU_DEFAULT if persist else 0
        ),
        use_nt=bucket <= TokenBucket.BS128,
        persist_strided=TokenBucket.BS512 <= bucket <= TokenBucket.BS2048,
        deep_a_pipeline=bucket >= TokenBucket.BS1024 and block_n == TILE_N_BASE,
    )


def _select_large_stage2(
    bucket: int,
    sort_block_m: int,
    model_dim: int,
) -> Stage2Config:
    persist_cu = _LARGE_PERSIST_CU_RESIDUALS.get(bucket, PERSIST_CU_DEFAULT)
    block_n = (
        TILE_N_NARROW
        if bucket == TokenBucket.BS256 or model_dim < WIDE_MODEL_DIM_MIN
        else TILE_N_BASE
    )
    block_m = (
        BLOCK_M_MEDIUM
        if sort_block_m == BLOCK_M_LARGE or bucket == TokenBucket.BS2048
        else BLOCK_M_SMALL
    )
    skew_cu = (
        LARGE_STAGE2_SKEW_CU
        if bucket >= TokenBucket.BS512 and bucket not in _LARGE_STAGE2_NO_SKEW_BUCKETS
        else 0
    )
    return Stage2Config(
        block_m=block_m,
        block_n=block_n,
        persist=True,
        persist_cu=persist_cu,
        use_nt=bucket <= TokenBucket.BS128,
        persist_strided=TokenBucket.BS512 <= bucket <= TokenBucket.BS2048,
        skew_cu=skew_cu,
        deep_a_pipeline=bucket >= TokenBucket.BS1024 and block_n == TILE_N_BASE,
    )


@cache
def _select_bucket_config(
    bucket: int,
    mtpr_class: int,
    experts_per_rank: int,
    model_dim: int,
    inter_dim: int,
    p2p_quant: str,
) -> MegaMoEConfig:
    if mtpr_class == MAX_MTPR_CLASS:
        stage1 = _select_large_stage1(
            bucket,
            experts_per_rank,
            inter_dim,
        )
        stage2 = _select_large_stage2(
            bucket,
            stage1.sort_block_m,
            model_dim,
        )
        return MegaMoEConfig(stage1=stage1, stage2=stage2, p2p_quant=p2p_quant)

    fixed_slot = mtpr_class <= FIXED_SLOT_MAX_MTPR
    if fixed_slot:
        stage1 = _select_fixed_stage1(bucket, experts_per_rank)
    else:
        stage1 = _select_bounded_stage1(bucket, mtpr_class, experts_per_rank, inter_dim)
    stage2 = _select_bounded_stage2(
        bucket, fixed_slot, mtpr_class, stage1.sort_block_m, model_dim
    )
    return MegaMoEConfig(stage1=stage1, stage2=stage2, p2p_quant=p2p_quant)


# ---- Request normalization and public resolution ------------------------------


def _normalize_config_request(
    tokens: int,
    mtpr: int,
    p2p_quant: str,
    a_dtype: str,
    experts_per_rank: int,
    model_dim: int,
    inter_dim: int,
) -> tuple[int, int, str]:
    _validate_mtpr(mtpr)
    if tokens > mtpr:
        raise ValueError(f"tokens={tokens} exceeds mtpr={mtpr}")
    _validate_activation_dtype(a_dtype)
    if experts_per_rank <= 0:
        raise ValueError(f"experts_per_rank must be positive, got {experts_per_rank}")
    if model_dim <= 0 or inter_dim <= 0:
        raise ValueError(f"invalid model shape {model_dim}x{inter_dim}")

    bucket = nearest_token_bucket(tokens)
    mtpr_class = mtpr_config_class(mtpr)
    if mtpr_class <= FIXED_SLOT_MAX_MTPR and bucket > TokenBucket.BS128:
        raise ValueError(f"fixed-slot does not support token bucket {bucket}")
    if mtpr_class <= FIXED_SLOT_MAX_MTPR and experts_per_rank > GPU_WAVE_SIZE:
        raise ValueError("fixed-slot supports at most 64 experts per rank")

    if p2p_quant == P2P_QUANT_AUTO:
        # MTPR is rank-invariant; local token counts need not be.
        use_fp8 = (
            mtpr >= P2P_FP8_MIN_MTPR
            if a_dtype == ACTIVATION_FP4
            else mtpr > P2P_FP8_MIN_MTPR
        )
        p2p_quant = P2P_QUANT_FP8_BLOCKWISE if use_fp8 else P2P_QUANT_NONE
    elif p2p_quant not in SUPPORTED_P2P_QUANT_MODES:
        raise ValueError(f"unsupported p2p_quant={p2p_quant!r}")
    return bucket, mtpr_class, p2p_quant


def select_mega_moe_config(
    tokens: int,
    mtpr: int,
    p2p_quant: str = P2P_QUANT_AUTO,
    *,
    a_dtype: str = ACTIVATION_FP8,
    experts_per_rank: int = REFERENCE_EXPERTS_PER_RANK,
    model_dim: int = DEFAULT_MODEL_DIM,
    inter_dim: int = DEFAULT_INTER_DIM,
) -> MegaMoEConfig:
    bucket, mtpr_class, p2p_quant = _normalize_config_request(
        tokens,
        mtpr,
        p2p_quant,
        a_dtype,
        experts_per_rank,
        model_dim,
        inter_dim,
    )
    return _select_bucket_config(
        bucket,
        mtpr_class,
        expert_config_class(experts_per_rank),
        model_dim,
        inter_dim,
        p2p_quant,
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


def _make_tuning_context(
    config: MegaMoEConfig,
    tokens: int,
    a_dtype: str,
    mtpr: int,
) -> TuningContext:
    _validate_activation_dtype(a_dtype)
    _validate_mtpr(mtpr)
    return TuningContext(
        tokens=tokens,
        bucket=nearest_token_bucket(tokens),
        mtpr=mtpr,
        capacity_mode=capacity_mode_for_mtpr(mtpr),
        a_dtype=a_dtype,
        p2p_quant=config.p2p_quant,
    )


def _select_tuning_patches(
    config: MegaMoEConfig,
    context: TuningContext,
) -> tuple[ConfigPatch, ...]:
    patches = []

    if context.p2p_quant == P2P_QUANT_FP8_BLOCKWISE and (
        patch := _p2p_fp8_patch(context.bucket)
    ):
        patches.append(patch)

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

    if context.tokens >= WIDE_BATCH_MIN_TOKENS:
        patches.append(_A4_WIDE_BATCH_OCCUPANCY_PATCH)

    if patch := _A4_BUCKET_PATCHES.get(context.bucket):
        patches.append(patch)

    if context.p2p_quant == P2P_QUANT_FP8_BLOCKWISE and (
        patch := _A4_FP8_P2P_BUCKET_PATCHES.get(context.bucket)
    ):
        patches.append(patch)

    if (
        context.capacity_mode == CapacityMode.LARGE_COMPACT
        and context.mtpr == FIXED_LARGE_CAPACITY_MTPR
    ):
        # A fixed 8192-token allocation still executes decode-sized dynamic
        # batches through the large-compact dispatcher.
        if context.tokens == 1:
            patches.append(_A4_FIXED8192_SINGLE_TOKEN_PATCH)
        elif context.bucket <= TokenBucket.BS1024:
            patches.append(_A4_FIXED8192_COMPACT_DISPATCH_PATCH)

        if (
            context.bucket == TokenBucket.BS8
            and context.p2p_quant == P2P_QUANT_FP8_BLOCKWISE
        ):
            patches.append(_A4_FIXED8192_BS8_STAGE2_PATCH)
        elif context.bucket in _A4_FIXED8192_SYNC_BUCKETS:
            grid_mult = (
                2 if context.bucket == TokenBucket.BS16 else GRID_MULT_SINGLE_EPOCH
            )
            patches.append(_a4_fixed8192_sync_patch(context.bucket, grid_mult))

        if patch := _a4_fixed8192_tuned_patch(context.bucket):
            patches.append(patch)
        if context.tokens == 2:
            # BS2 resolves to the BS1 bucket, but its measured optimum differs
            # from the latency-sensitive single-token path.
            patches.append(_A4_FIXED8192_TWO_TOKEN_PATCH)

    return tuple(patches)


def _apply_mega_moe_quant_config(
    config: MegaMoEConfig,
    tokens: int,
    a_dtype: str,
    *,
    mtpr: int,
) -> MegaMoEConfig:
    """Testing helper for applying tuning to an already selected base config."""
    context = _make_tuning_context(config, tokens, a_dtype, mtpr)
    return _apply_tuning_context(config, context)


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
) -> MegaMoEConfig:
    context = TuningContext(
        tokens=token_key,
        bucket=bucket,
        mtpr=mtpr,
        capacity_mode=capacity_mode_for_mtpr(mtpr),
        a_dtype=a_dtype,
        p2p_quant=config.p2p_quant,
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
    model_dim: int = DEFAULT_MODEL_DIM,
    inter_dim: int = DEFAULT_INTER_DIM,
) -> MegaMoEConfig:
    """Resolve and cache the complete base-plus-tuning MegaMoE config."""
    bucket, mtpr_class, p2p_quant = _normalize_config_request(
        tokens,
        mtpr,
        p2p_quant,
        a_dtype,
        experts_per_rank,
        model_dim,
        inter_dim,
    )
    config = _select_bucket_config(
        bucket,
        mtpr_class,
        expert_config_class(experts_per_rank),
        model_dim,
        inter_dim,
        p2p_quant,
    )
    return _resolve_tuned_config(
        config,
        bucket,
        _tuning_token_key(tokens),
        mtpr,
        a_dtype,
    )
