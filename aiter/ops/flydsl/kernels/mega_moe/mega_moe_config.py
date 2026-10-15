# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
"""Static MegaMoEV2 configuration rules for MI355X."""

from bisect import bisect_left
from collections.abc import Callable
from dataclasses import dataclass, replace
from functools import cache

from . import envs

TOKEN_BUCKETS = (
    1,
    2,
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
# Fixed-slot (direct expert slots, no count exchange) is the default up to MTPR
# 128; AITER_MEGA_FIXED_SLOT_MAX_MTPR extends it (see envs.py).
FIXED_SLOT_MAX_MTPR = envs.AITER_MEGA_FIXED_SLOT_MAX_MTPR
MAX_MTPR_CLASS = 32768
INDEXED_PAYLOAD_MIN_MTPR = MAX_MTPR_CLASS
INDEXED_PAYLOAD_MIN_SBM = 128
# Source-indexed payload storage cuts the maximum-capacity activation buffer
# by roughly ``topk``.  Keep every smaller capacity on the historical layout.
REFERENCE_EXPERTS_PER_RANK = 48
R1_EXPERTS_PER_RANK = 32
KIMI_K3_EXPERTS_PER_RANK = 112
# Per-network tuning is keyed by the full (experts_per_rank, model_dim,
# inter_dim) identity, not by experts-per-rank alone: other networks can share
# an experts-per-rank (e.g. GLM-5.2 also has 32) yet have a different model
# shape, and must not inherit another network's validated table.  Networks that
# do not match any registered identity fall back to the base config until they
# are tuned in their own right.
R1_MODEL_DIM, R1_INTER_DIM = 7168, 2048
V4_MODEL_DIM, V4_INTER_DIM = 7168, 3072
K3_MODEL_DIM, K3_INTER_DIM = 3584, 3072
# EPLB redundant-expert profiles (r32 / r64) share V4-Pro's shape and FP4
# prelude but have no per-bucket tuner of their own.
V4_EPLB_EXPERTS_PER_RANKS = (52, 56)
# Compact route metadata dedicates ten bits to the global expert/group segment.
# Under the EP8 protocol this admits 8 * 127 expert segments plus 8 group
# segments.  The next expert would require segment 1024 and cannot be encoded.
MAX_FANOUT_SEGMENTS = 1024
MAX_FANOUT_EXPERTS_PER_RANK = 256
P2P_QUANT_AUTO = "auto"
P2P_QUANT_NONE = "none"
P2P_QUANT_FP8_BLOCKWISE = "fp8_blockwise_1x32"
SUPPORTED_P2P_QUANT_MODES = (P2P_QUANT_NONE, P2P_QUANT_FP8_BLOCKWISE)

ACTIVATION_FP4 = "fp4"
ACTIVATION_FP8 = "fp8"
SUPPORTED_ACTIVATION_DTYPES = (ACTIVATION_FP4, ACTIVATION_FP8)


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
    b2stage: bool = False
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
        if self.p2p_quant not in ("none", "fp8_blockwise_1x32"):
            raise ValueError(f"unsupported p2p_quant={self.p2p_quant!r}")
        if self.p2p_quant != "none" and self.stage2.bf16_lds:
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
        # LDS-DMA A copies + the paired K loop keep the B prefetch in flight,
        # which matters when few tiles are resident.
        async_a_copy=bucket <= 8,
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
        # payload_chunk_rows is part of the dispatch wire protocol: senders
        # chunk payload by it (emit_dispatch_payload) and receivers predict
        # publisher counts by it (_initialize_section_ready).  Under uneven-rank
        # dynamic prefill each rank picks its own token bucket within this
        # MAX-MTPR (32768) class, so the value MUST be identical for every
        # bucket or Stage1's equality wait deadlocks.  1536 is the rank-invariant
        # choice for the class (tuned on the largest bucket).
        payload_chunk_rows=1536,
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
    # Fixed-slot buckets below 128 also run persistent: the non-persistent grid
    # is sized for the MTPR capacity, and its idle CTAs delay the real tiles.
    persist = bucket >= 128 or fixed_slot
    if not persist:
        persist_cu = 0
    elif bucket < 128 or bucket == 256:
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
        # Non-temporal W2 loads also for fixed-slot 256 (DP-padded decode graphs).
        use_nt=bucket <= 128 or (fixed_slot and bucket <= 256),
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


def _replace_config(
    config: MegaMoEConfig,
    *,
    stage1: dict[str, object] | None = None,
    stage2: dict[str, object] | None = None,
) -> MegaMoEConfig:
    return replace(
        config,
        stage1=replace(config.stage1, **stage1) if stage1 else config.stage1,
        stage2=replace(config.stage2, **stage2) if stage2 else config.stage2,
    )


@dataclass(frozen=True, slots=True)
class _TuningContext:
    """Inputs shared by every per-network tuner."""

    tokens: int
    bucket: int
    mtpr: int
    mtpr_class: int
    a_dtype: str
    apply_fp4: bool
    fixed_slot: bool = False


def _fp4_common_prelude(config: MegaMoEConfig, ctx: _TuningContext) -> MegaMoEConfig:
    """FP4 adjustments shared by every non-K3 network, applied before per-network
    knobs.  Dispatched as a global pre-rule (so the V4-Pro EPLB redundant-expert
    profiles epr 52/56, which have no per-network tuner, still receive it)."""
    bucket, tokens, mtpr = ctx.bucket, ctx.tokens, ctx.mtpr
    if mtpr <= FIXED_SLOT_MAX_MTPR:
        if bucket == 4:
            config = _replace_config(config, stage1={"grid_mult": 4})
        elif bucket == 16:
            config = _replace_config(config, stage1={"grid_mult": 3})

    if bucket <= 128 and config.stage1.async_a_copy and not ctx.fixed_slot:
        # FP4 halves the A K-step bytes. SBM64 gives every thread at least one
        # 16-byte async copy unless a validated synchronous patch below applies.
        # Skipped for fixed-slot (ll_cap=32 at mtpr 1/2/4 is not divisible by
        # SBM64); those tiny profiles keep the legal SBM32 FP4 geometry.
        config = _replace_config(config, stage1={"sort_block_m": 64})

    if 256 <= tokens < 32768:
        config = _replace_config(config, stage1={"waves_per_eu_hint": 1})
    return config


def _tune_r1(config: MegaMoEConfig, ctx: _TuningContext) -> MegaMoEConfig:
    """R1 (32 experts per rank, 7168x2048) A4W4 per-bucket tuning (FP4 only).

    The FP4 common prelude is applied by the orchestrator before dispatch.
    """
    if not ctx.apply_fp4:
        return config
    bucket, mtpr = ctx.bucket, ctx.mtpr
    if bucket <= 128:
        if bucket in (8, 16, 32):
            # SBM32 + num_waves=8 needs the async A-copy DMA path: the
            # synchronous reg->LDS copy races under 8 waves (CUDA-graph
            # replay nondeterminism).  async_a_copy=True makes the w8 + mfma
            # geometry deterministic.
            stage1_patch: dict[str, object] = {
                "sort_block_m": 32,
                "num_waves": 8,
                "async_a_copy": True,
                "mfma_amajor": True,
            }
        elif bucket == 128:
            # SBM64 + w8 on the async DMA path (crash-free).
            stage1_patch = {
                "sort_block_m": 64,
                "num_waves": 8,
                "async_a_copy": True,
                "mfma_amajor": True,
            }
        elif bucket == 64:
            # SBM64 keeps the reg->LDS copy race-free without async (the race
            # only bites SBM32+nw8).  work_shards=2 reduces dispatch scheduling
            # overhead; pairs with the Stage2 deep_a path below.
            stage1_patch = {
                "sort_block_m": 64,
                "num_waves": 8,
                "async_a_copy": False,
                "mfma_amajor": True,
                "work_shards": 2,
            }
        else:
            stage1_patch = {
                "sort_block_m": 32,
                "num_waves": 4,
                "async_a_copy": False,
            }
        config = _replace_config(config, stage1=stage1_patch)
    if mtpr == 8192:
        # Per-bucket dispatch CU; small buckets also raise grid_mult.
        r1_dispatch_cu = {
            1: 192,
            2: 224,
            4: 224,
            8: 224,
            16: 160,
            32: 128,
            64: 160,
            128: 160,
            256: 96,
            512: 160,
            1024: 32,
            2048: 64,
            4096: 72,
            8192: 96,
        }[bucket]
        stage1_patch = {"num_dispatch_cu": r1_dispatch_cu}
        if bucket in (2, 4, 8, 16, 32):
            stage1_patch["grid_mult"] = 2
        config = _replace_config(config, stage1=stage1_patch)
        if bucket == 2048:
            # Match the larger-bucket Stage1 geometry (SBM128 + mfma_amajor).
            config = _replace_config(
                config,
                stage1={
                    "sort_block_m": 128,
                    "num_waves": 8,
                    "tile_n": 512,
                    "mfma_amajor": True,
                },
            )
        if bucket == 64:
            # deep_a on the SBM64+w8 base with bm64.
            config = _replace_config(
                config,
                stage2={
                    "block_m": 64,
                    "b2stage": True,
                    "deep_a_pipeline": True,
                },
            )
        if bucket == 128:
            # deep_a + bm64 to fill the SBM64 sort block.
            config = _replace_config(
                config,
                stage2={
                    "block_m": 64,
                    "b2stage": True,
                    "deep_a_pipeline": True,
                },
            )
        if bucket == 256:
            # Widen Stage2 and enable CU skew; b_nt=0 keeps weights in L2.
            config = _replace_config(
                config,
                stage1={"mfma_amajor": True, "b_nt": 0},
                stage2={"block_n": 256, "block_m": 64, "skew_cu": 96},
            )
    elif bucket == 32768:
        # num_dispatch_cu is a rank-local launch knob (main already tunes it
        # per bucket, e.g. the mtpr==8192 table above), so varying it per rank
        # is safe.  payload_chunk_rows is NOT: it is part of the dispatch wire
        # protocol -- senders chunk payload by it (emit_dispatch_payload) and
        # receivers predict publisher counts by it
        # (_initialize_section_ready).  Each rank selects its own token bucket,
        # so a per-bucket payload_chunk_rows makes a TPR16384 sender emit
        # chunks a TPR128 receiver does not expect, deadlocking Stage1's
        # equality wait.  It is kept at the rank-invariant class base (1536,
        # set in _select_large_stage1) for every bucket; only tune the
        # rank-local dispatch CU count here.
        config = _replace_config(config, stage1={"num_dispatch_cu": 64})
    return config


def _tune_v4_pro(config: MegaMoEConfig, ctx: _TuningContext) -> MegaMoEConfig:
    """V4-Pro (48 experts per rank) A4W4 per-bucket tuning (FP4 only).

    The FP4 common prelude is applied by the orchestrator before dispatch.
    """
    if not ctx.apply_fp4:
        return config
    bucket, mtpr = ctx.bucket, ctx.mtpr
    bucket_stage1: dict[str, object] = {}
    bucket_stage2: dict[str, object] = {}
    if bucket == 512:
        bucket_stage1 = {
            "b_nt": 0,
            "num_dispatch_cu": 160,
        }
    elif bucket == 1024:
        bucket_stage1 = {"sort_block_m": 128, "num_dispatch_cu": 88}
        bucket_stage2 = {"block_m": 64}
    elif bucket == 4096:
        bucket_stage1 = {"num_dispatch_cu": 72}
    if bucket_stage1 or bucket_stage2:
        config = _replace_config(
            config,
            stage1=bucket_stage1,
            stage2=bucket_stage2,
        )

    if config.p2p_quant == P2P_QUANT_FP8_BLOCKWISE and bucket == 1024:
        config = _replace_config(
            config,
            stage1={"grid_mult": 1},
            stage2={"persist_cu": 224},
        )

    if mtpr == 8192:
        if bucket <= 1024:
            config = _replace_config(config, stage1={"num_dispatch_cu": 32})

        if bucket == 8:
            # FP4 SBM32 contains exactly 256 16-byte A chunks. The 8-wave GEMM
            # keeps its compute geometry while waves 0-3 perform async copies.
            config = _replace_config(
                config,
                stage1={
                    "sort_block_m": 32,
                    "num_waves": 8,
                    "async_a_copy": True,
                },
            )

        if bucket in (16, 32, 64, 128):
            config = _replace_config(
                config,
                stage1={
                    "sort_block_m": 32,
                    "tile_n": 256,
                    "num_waves": 4,
                    # Extra grid epoch on bs16/32/64; dispatch has enough work.
                    "grid_mult": 2 if bucket in (16, 32, 64) else 1,
                    "mfma_amajor": False,
                    "async_a_copy": False,
                },
                stage2={"block_m": 32},
            )

        # Per-bucket dispatch CU. Larger compute-bound buckets are left alone.
        dispatch_cu = {
            2: 128,
            4: 128,
            8: 32,
            16: 128,
            32: 224,
            64: 224,
            128: 128,
            256: 224,
            512: 160,
            1024: 96,
            4096: 72,
        }.get(bucket)
        if dispatch_cu is not None:
            stage1_patch: dict[str, object] = {"num_dispatch_cu": dispatch_cu}
            if bucket == 256:
                stage1_patch["tile_n"] = 256
            config = _replace_config(config, stage1=stage1_patch)

    # Restore Stage2 fields that drifted from the pre-rebase tuning, and unblock
    # the deep Stage2 path below.  Scoped to the large-MTPR class.
    if ctx.mtpr_class == MAX_MTPR_CLASS:
        restore_stage2: dict[str, object] = {}
        if bucket == 256:
            # use_nt streams GEMM2 weights past L2; lower persist_cu frees CUs
            # for dispatch. Improves this bucket.
            restore_stage2 = {
                "block_m": 64,
                "block_n": 256,
                "persist_cu": 96,
                "b2stage": True,
                "deep_a_pipeline": True,
                "use_nt": True,
            }
        elif bucket == 512:
            restore_stage2 = {
                "block_m": 64,
                "b2stage": True,
                "deep_a_pipeline": True,
            }
        elif bucket == 2048:
            restore_stage2 = {"block_m": 64}
        elif bucket == 8192 and config.stage2.aligned_pair:
            # Aligned-pair and the deep path are mutually exclusive; prefer the
            # deep path here and give aligned-pair up.
            restore_stage2 = {"aligned_pair": False, "pair_cu": 0, "skew_cu": 96}
        block_m = restore_stage2.get("block_m")
        if block_m is not None:
            sbm = config.stage1.sort_block_m
            if block_m > sbm or sbm % block_m:
                # block_m has to tile the Stage1 sort block.  Skip the field
                # rather than raise if a later Stage1 change breaks that.
                restore_stage2.pop("block_m")
        if restore_stage2:
            config = _replace_config(config, stage2=restore_stage2)
    elif bucket == 256:
        # Bounded MTPR keeps BN128 with the deep pipeline off, so the table
        # above does not carry over.  b2stage is applied at bucket 256 only, not
        # 512.
        config = _replace_config(config, stage2={"b2stage": True})

    # The deep Stage2 path is enabled only for A4 and only where the base
    # geometry uses BN256, so A8 remains byte-for-byte mainline.
    if (
        bucket >= 1024
        and config.stage2.block_n == 256
        and not config.stage2.aligned_pair
    ):
        config = _replace_config(
            config,
            stage2={"b2stage": True, "deep_a_pipeline": True},
        )
    return config


def _tune_kimi_k3(config: MegaMoEConfig, ctx: _TuningContext) -> MegaMoEConfig:
    """Kimi-K3 (112 experts per rank) per-bucket tuning (both activation dtypes)."""
    bucket = ctx.bucket
    # Generic num_dispatch_cu=32 is sized for 32-expert ranks and starves K3
    # dispatch (112 experts/rank). Raise it; keep the ends lower.
    config = _replace_config(
        config,
        stage1={"num_dispatch_cu": 96 if bucket == 1 or bucket >= 2048 else 160},
    )
    # Raise Stage2 persist_cu on large buckets (separate phase from dispatch).
    if bucket >= 2048:
        config = _replace_config(
            config,
            stage2={"persist_cu": 192 if bucket <= 4096 else 256},
        )
    # Widen Stage2 off the generic BN128. Not the earlier deep-A BN256 rule:
    # deep_a fails to build at 8192, and that rule runs before this branch.
    if bucket >= 4096:
        config = _replace_config(
            config,
            stage2={"block_n": 256, "b2stage": True},
        )
    elif 256 <= bucket <= 2048:
        # Stage1 uses SBM64 with a generic M tile of 32; widen block_m to 64
        # so each Stage2 tile fills the sort block.
        config = _replace_config(
            config,
            stage2={"block_m": 64, "block_n": 256, "b2stage": True},
        )
    elif 8 <= bucket <= 128:
        # Wider Stage2 N-tile only; SBM32 keeps block_m=32. Tiny buckets stay
        # on BN128.
        config = _replace_config(
            config,
            stage2={"block_n": 256},
        )
    return config


@dataclass(frozen=True, slots=True)
class NetworkProfile:
    """One tunable network, identified by (experts_per_rank, model_dim, inter_dim).

    To add a network: implement a ``_tune_<name>(config, ctx)`` function and add
    a ``NetworkProfile`` row to ``_NETWORKS``; nothing else in selection changes.
    The tuner decides per-dtype behavior from ``ctx.a_dtype`` / ``ctx.apply_fp4``
    (R1 and V4-Pro only tune FP4; Kimi-K3 tunes both FP4 and FP8).

    ``family_experts_per_rank`` lists extra experts-per-rank values (EPLB
    redundant-expert profiles) that share this network's shape and FP4 prelude
    but have no per-bucket tuner of their own.
    """

    name: str
    experts_per_rank: int
    model_dim: int
    inter_dim: int
    tuner: Callable[[MegaMoEConfig, _TuningContext], MegaMoEConfig]
    family_experts_per_rank: tuple[int, ...] = ()


_NETWORKS = (
    NetworkProfile("r1", R1_EXPERTS_PER_RANK, R1_MODEL_DIM, R1_INTER_DIM, _tune_r1),
    NetworkProfile(
        "v4_pro",
        REFERENCE_EXPERTS_PER_RANK,
        V4_MODEL_DIM,
        V4_INTER_DIM,
        _tune_v4_pro,
        V4_EPLB_EXPERTS_PER_RANKS,
    ),
    NetworkProfile(
        "kimi_k3", KIMI_K3_EXPERTS_PER_RANK, K3_MODEL_DIM, K3_INTER_DIM, _tune_kimi_k3
    ),
)


def _match_network(
    experts_per_rank: int, model_dim: int, inter_dim: int
) -> NetworkProfile | None:
    """Return the registered network matching the full (epr, shape) identity.

    A profile matches when the shape is exact and the experts-per-rank is either
    the network's canonical value or one of its EPLB family profiles.  Networks
    that merely share an experts-per-rank (e.g. GLM-5.2 with 32) do not match and
    stay on the base config.
    """
    for profile in _NETWORKS:
        if profile.model_dim != model_dim or profile.inter_dim != inter_dim:
            continue
        if (
            experts_per_rank == profile.experts_per_rank
            or experts_per_rank in profile.family_experts_per_rank
        ):
            return profile
    return None


def select_mega_moe_config(
    tokens: int,
    mtpr: int,
    *,
    a_dtype: str = ACTIVATION_FP8,
    p2p_quant: str = P2P_QUANT_AUTO,
    experts_per_rank: int = REFERENCE_EXPERTS_PER_RANK,
    model_dim: int = 7168,
    inter_dim: int = 3072,
    world_size: int = 8,
) -> MegaMoEConfig:
    if a_dtype not in SUPPORTED_ACTIVATION_DTYPES:
        raise ValueError(f"unsupported activation dtype={a_dtype!r}")
    if p2p_quant not in (P2P_QUANT_AUTO, *SUPPORTED_P2P_QUANT_MODES):
        raise ValueError(f"unsupported p2p_quant={p2p_quant!r}")
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
    if fixed_slot_dispatch and bucket > FIXED_SLOT_MAX_MTPR:
        raise ValueError(f"fixed-slot does not support token bucket {bucket}")
    total_segments = world_size * experts_per_rank + world_size
    if total_segments > MAX_FANOUT_SEGMENTS:
        raise ValueError(
            f"MegaMoE v2 fanout needs {total_segments} segments, exceeding "
            f"the {MAX_FANOUT_SEGMENTS}-segment route metadata limit"
        )
    # Bucket 2 is an exact tuning identity, while its untuned A8 base remains
    # bucket 1 so adding it does not change existing A8 geometry.
    base_bucket = 1 if bucket == 2 else bucket
    config = _select_bucket_config(
        base_bucket,
        mtpr_class,
        model_dim,
        inter_dim,
        fixed_slot_dispatch,
    )
    # Resolve the network by its full (epr, shape) identity.  Networks that only
    # share an experts-per-rank (e.g. GLM-5.2 with 32) do not match and keep the
    # base config, so they never inherit another network's tuning or base rules.
    profile = _match_network(experts_per_rank, model_dim, inter_dim)
    if profile is not None and profile.name == "r1" and bucket == 8192:
        # Disable the aligned-pair Stage2 for R1 bucket 8192 on BOTH dtypes.
        # Although a8w4's mtpr>=1024 p2p is fp8_blockwise (which aligned-pair
        # nominally requires), the aligned-pair path is non-deterministic for
        # a8w4 here: test_mega_moe_v2's slice_output consistency check
        # (two forwards must return identical valid rows) fails intermittently
        # at bs8192.  Keep the plain Stage2, which is deterministic for both
        # dtypes.  (Do NOT FP4-gate this: that re-enables the flaky a8w4 path.)
        config = _replace_config(
            config,
            stage2={"aligned_pair": False, "pair_cu": 0},
        )
    if (
        a_dtype == ACTIVATION_FP4
        and bucket <= 128
        and config.stage1.async_a_copy
        and not fixed_slot_dispatch
    ):
        # FP4 halves each A K-step. SBM64 keeps at least one 16-byte async
        # copy per thread in the eight-wave Stage1 configuration.
        # Excluded for fixed-slot: its EP8 receive capacity is
        # ll_cap=ceil(8*mtpr/32)*32, which is 32 at mtpr 1/2/4, so promoting
        # sort_block_m to 64 breaks the Stage1 `fuse_cap % tile_m == 0` compile
        # assert (and the fixed-slot required-cap check).  Those tiny profiles
        # keep the legal SBM32 FP4 geometry.
        config = replace(
            config,
            stage1=replace(config.stage1, sort_block_m=64),
        )
    if p2p_quant == P2P_QUANT_AUTO:
        desired_p2p = (
            P2P_QUANT_FP8_BLOCKWISE if mtpr >= P2P_FP8_MIN_MTPR else config.p2p_quant
        )
    else:
        desired_p2p = p2p_quant
    if desired_p2p != config.p2p_quant:
        config = replace(config, p2p_quant=desired_p2p)
    # Per-network tuning is dispatched by the matched profile above.  R1/V4-Pro
    # only tune FP4 (apply_fp4); Kimi-K3 tunes both activation dtypes.  The
    # FP4-only INDEXED_PAYLOAD async-revert below stays gated on FP4 as well (it
    # is specific to FP4's 8x16B K-step packing).
    apply_fp4 = a_dtype == ACTIVATION_FP4
    ctx = _TuningContext(
        tokens=tokens,
        bucket=bucket,
        mtpr=mtpr,
        mtpr_class=mtpr_class,
        a_dtype=a_dtype,
        apply_fp4=apply_fp4,
        fixed_slot=fixed_slot_dispatch,
    )
    # FP4 common prelude applies to every matched non-K3 network, including the
    # V4-Pro EPLB redundant-expert profiles 52/56 that have no tuner; apply it as
    # a pre-rule, then let the per-network tuner add its table.  Only the
    # canonical experts-per-rank carries a tuner (EPLB family profiles get the
    # prelude only).
    is_k3 = profile is not None and profile.name == "kimi_k3"
    if apply_fp4 and profile is not None and not is_k3:
        config = _fp4_common_prelude(config, ctx)
    if profile is not None and experts_per_rank == profile.experts_per_rank:
        config = profile.tuner(config, ctx)
    if (
        a_dtype == ACTIVATION_FP4
        and mtpr >= INDEXED_PAYLOAD_MIN_MTPR
        and config.stage1.sort_block_m >= INDEXED_PAYLOAD_MIN_SBM
    ):
        # FP4 packs one K step into 8 16-byte chunks, so a 16-lane direct
        # global-to-LDS atom spans two rows. Indexed payload gives every
        # row an independent source key, which that atom cannot express.
        config = _replace_config(config, stage1={"async_a_copy": False})
    if config.stage2.aligned_pair and config.p2p_quant != P2P_QUANT_FP8_BLOCKWISE:
        # The aligned-pair Stage2 kernel only emits the FP8 P2P row layout, so
        # reject the combination here instead of at launch.
        raise ValueError(
            f"p2p_quant={config.p2p_quant!r} conflicts with aligned-pair Stage2"
        )
    return config


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
        config = select_mega_moe_config(
            bucket,
            mtpr,
            a_dtype=a_dtype,
            p2p_quant=p2p_quant,
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
