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
# Source-indexed payload storage cuts the maximum-capacity activation buffer by
# roughly ``topk``, so every rank at the max-capacity MTPR class uses it
# regardless of sort_block_m; smaller MTPR classes keep the route-major layout.
INDEXED_PAYLOAD_MIN_MTPR = MAX_MTPR_CLASS
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
# EP4 DeepSeek-V4.1-Flash: (world_size, experts_per_rank, model_dim, inter_dim).
V41_FLASH_EP4_LAYOUT = (4, 96, 5120, 2304)
# (world_size, model_dim, inter_dim, experts, topk) of the layouts the fused (small- and large-input) kernels
# serve: EP8 DeepSeek-V4-Pro and EP4 DeepSeek-V4.1-Flash.
FUSED_LAYOUTS = frozenset(
    {
        (8, V4_MODEL_DIM, V4_INTER_DIM, 8 * REFERENCE_EXPERTS_PER_RANK, 6),
        (4, 5120, 2304, 4 * 96, 6),
    }
)
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
# Execution routes of one MegaMoEV2 instance. Every EP rank must take the same
# route for a call, so a route is chosen from EP-wide inputs only (the plan and
# the call's max_tokens), never from a rank's own row count.
ROUTE_STAGED = "staged"
ROUTE_SMALL_M = "small_m"
ROUTE_LARGE_M = "large_m"
# Single-kernel small-input runtimes: power-of-two capacities up to this bound.
SMALL_M_MAX_MTPR = 512
DEFAULT_SMALL_M_MTPRS = (256, 512)
# Instance capacities the large-input persistent kernel is built for. At 32768
# the staged path stays: it is faster there (large-input E2E +11.7% at 32768
# uniform tokens) and its source-indexed payload needs 1.75 GiB per GPU where
# the large-input route-major payload needs 10.5 GiB.
LARGE_M_MTPRS = (1024, 2048, 4096, 8192, 16384)
# EP4 DeepSeek-V4.1-Flash takes its large-input tune at 8192 and 16384 only: on gfx950 the kernel faults
# (illegal memory access) at 1024, 2048 and 4096.
V41_FLASH_EP4_LARGE_M_MTPRS = (8192, 16384)
# The fused kernels have no 2-token variant: 2 tokens take the 1-token config.
FUSED_TOKEN_BUCKETS = tuple(b for b in TOKEN_BUCKETS if b != 2)
# Below this many rows per rank the staged (or small-input) path is faster than
# the large-input kernel, whose geometry is fixed by its capacity.
LARGE_M_MIN_TOKENS = SMALL_M_MAX_MTPR + 1
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
class FusedStage1Knobs:
    """Dispatch / plan / GEMM1 knobs only the fused (small- and large-input) kernels read.

    All off is the EP8 DeepSeek-V4-Pro kernel; the EP4 DeepSeek-V4.1-Flash
    tune (_tune_v41_flash_ep4) turns them on. The protocol knobs (dedup_rows,
    plan_local, direct_done, tok_done) change what the ranks of a launch
    exchange: every rank of a launch must agree on them, so they are set per
    layout, never per token bucket.
    """

    # Large-input dispatch: at least this many payload producers per destination (0: one per row chunk).
    payload_min_blocks: int = 0
    # With payload_min_blocks: a rank with at most payload_small_tokens tokens gives each destination at least
    # payload_small_blocks producers instead (0: off).
    payload_small_blocks: int = 0
    payload_small_tokens: int = 0
    # One dispatch row per (token, destination rank) instead of per route; GEMM1 reads A through the srcmap.
    # Every rank of a launch must agree.
    dedup_rows: bool = False
    # Small-input fused kernel: every CTA waits on the plan flag itself and invalidates only its L1, instead of one
    # CTA an XCD acquiring and republishing the flag. No CTA reads the plan, srcmap or token rows before the flag, so
    # its XCD's L2 holds none of their lines (the launch's acquire dropped the earlier ones).
    plan_direct: bool = False
    # Small-input fused kernel: the plan CTA writes the plan through to memory, so it publishes the flag without an L2
    # writeback first.
    plan_wt: bool = False
    # Small-input fused kernel: every CTA waits for the sources itself and builds the plan from the expert counts
    # in its LDS; GEMM1 takes its tiles from there, and only GEMM2 reads the plan CTA's metadata.
    plan_local: bool = False
    # Small-input fused kernel: GEMM2 moves its dispatch-table pointers into SGPRs, so its buffer loads need no
    # readfirstlane waterfall loop.
    g2_uniform: bool = False
    # Small-input fused kernel: the dispatch-table and remote pointers of the head (producers, plan finalize,
    # kernel entry) in SGPRs, for the same reason.
    head_uniform: bool = False
    # Small-input fused kernel: the GEMM2 scatter epilogue reads its rows' ids / weights from LDS once into lanes and
    # picks them with readlane, instead of an LDS round trip and readfirstlane per row (2: the peer bases too).
    g2_epi_lean: int = 0
    # Small-input GEMM1: K-steps its W1 / W1-scale loads are issued ahead of their MFMAs.
    g1_lookahead: int = 1
    # Small-input GEMM1: K-steps its A rows are loaded ahead of their MFMAs (2: through registers, see do_tile_la).
    g1_a_ahead: int = 1
    # Small-input fused kernel: when a rank's routes need at most this many producers, each of them adds its share of
    # the source flag to every destination itself after its own drain, instead of the last of all producers storing
    # it behind a rank-local done counter (0: off; at most mega_moe_small_m._DIRECT_DONE_INC).
    direct_done: int = 0
    # Small-input fused kernel: each GEMM2 scatter wave drains its rows and adds one to their tokens' counters on
    # the destination, whose combine reduces a token once its counter reaches valid routes x GEMM2 n-blocks instead
    # of after every rank's grid ticket and flag (BF16 rows; the flags stay as the fallback).
    tok_done: bool = False
    # Small-input fused kernel: the runtime keeps per-token GEMM2 completion counters (two parity halves): the
    # tok_done buckets add to them and every launch clears the previous launch's half. Every bucket of a layout
    # must agree, since a tok_done launch may follow any other.
    tok_counters: bool = False
    # Small-input fused kernel: the GEMM2 rows of the first this-many tokens of a destination are BF16 even when
    # the rank sends FP8 rows, and the combine reduces them as BF16 and leaves them poisoned. Sender and combine
    # must agree, so every bucket of a layout sets the same value.
    bf16_head_tokens: int = 0
    # Large-input GEMM1: m-tiles with at most this many real rows (an expert's padded tail) run a
    # short_tail_rows-row tile (0: every m-tile runs sort_block_m rows).
    short_tail_rows: int = 0
    # With short_tail_rows: a tail tile with at most fold_tail real rows folds into its expert's preceding
    # m-tile, which then runs sort_block_m + fold_tail rows and reads the W1 slab once for both (0: off).
    fold_tail: int = 0
    # With fold_tail: an expert's last m-tile with more than short_tail_rows real rows and at most mid_tail_rows
    # runs a mid_tail_rows-row tile (0: off).
    mid_tail_rows: int = 0
    # Large-input plan: every wave of the plan CTA plans a slice of the local experts (False: one wave plans all).
    parallel_plan: bool = False
    # GEMM1 epilogue: output columns per lane in the quantize/store pass (2 or 8).
    epi_evec: int = 2
    # Large-input dispatch: a payload chunk issues one system release before its tile-ready and chunk_done RMWs
    # (False: a second one before chunk_done).
    merge_release: bool = False
    # Large-input dispatch: an expert's last payload chunk does not publish payload_ready (no reader).
    skip_payload_ready: bool = False
    # Large-input GEMM1 with dedup_rows: one wave issues the acquire once the tile's row stamps landed, then
    # a barrier (False: every wave acquires).
    one_row_acquire: bool = False
    # Large-input dispatch with merge_release: a payload chunk publishes its tiles without a system release; its
    # peer stores go to the uncached symmetric heap and are drained first (False: one L2 writeback per chunk).
    no_chunk_release: bool = False
    # Large-input GEMM1 K-loop reads its A operands through inline asm, so the step's first read does not wait for
    # the next step's A copy (False: the compiler makes it wait).
    a_asm_reads: bool = False
    # Large-input dispatch with dedup_rows: a payload chunk resolves its routes one per thread, then its waves copy
    # the sent token rows with all loads ahead of the stores (False: one route per wave in turn).
    batch_rows: bool = False
    # Large-input parallel plan: the plan CTA's warp 1 sizes the payload geometry (read by the producers only, after
    # pair_ready) next to the count exchange instead of before it, only warp 0 fences around the barriers every wave
    # passes, and each wave loads its slice's counts and the peers' row-base pointers with the prefix counts.
    lean_plan: bool = False
    # Large-input lean parallel plan: a source writes each count word to its destination with this launch's tag in
    # the high half, and every plan wave spins on the words it reads (False: count_done flags + a barrier).
    tag_counts: bool = False
    # Large-input dispatch with batch_rows: a route's histogram RMW result is its slot in a fixed-size per-expert
    # bucket of the pair order, so the order is complete with the histogram (False: prefix + a cursor RMW pass).
    bucket_pairs: bool = False
    # Large-input dispatch with bucket_pairs: the payload producers count their routes before the launch gate, next to
    # the owner's launch handshake (only the group_done RMW stays after it).
    early_hist: bool = False
    # Large-input dispatch: a payload producer reads the destination's rows per tile before the plan wait and its
    # ready-order flag once after it (not in every chunk publish).
    peer_hoist: bool = False
    # Large-input launch: the owner clears its rank-local counters between sending and awaiting the launch handshake.
    early_clear: bool = False
    # Large-input dispatch with batch_rows: before the plan, the consumer CTAs send the token rows whose lowest expert
    # at a destination is one of its first presend_experts local experts (the first payload round), so that round's
    # tasks only send route headers (0: off).
    presend_experts: int = 0
    # With presend_experts: the presend covers presend_rows // tokens-on-this-rank experts instead (at least
    # presend_experts, at most 64), so the first round's rows stay about constant as the rows per expert grow (0: off).
    presend_rows: int = 0
    # GEMM1 A scales staged to LDS in K-loop order: lane l's packed scale word of row group g at K-step s is one
    # dword, so a K-step reads one ds_read_b32 per group (False: four ds_read_u8 per group, packed in VALU).
    ascale_packed: bool = False
    # GEMM1 epilogue (epi_evec 8): the quantize/store pass without serial waits. Its lane offsets come from
    # the tile, so they are not hoisted out of the persistent loop and reloaded from scratch between the output stores
    # (each reload's vmcnt(0) waits for the stores before it); all rows' LDS reads go first; the 4-lane max uses DPP.
    # Bit 1: SiLU as v_exp_f32 + v_rcp_f32 (no IEEE divide, no exp2 range fix).
    epi_lean: int = 0


@dataclass(frozen=True, slots=True)
class FusedStage2Knobs:
    """GEMM2 / combine knobs only the fused kernels read (all off: the EP8 kernels)."""

    # Large-input fused kernel: the GEMM2 units are dealt to the 8 XCDs in chunks of this many; a CTA claims from
    # its own XCD's counter, then from the next XCDs' once that one is spent (0: one claim counter per work shard).
    xcd_chunk: int = 0
    # Large-input fused kernel: P2P row stride rounded up to this many bytes (0: unrounded).
    row_align: int = 0
    # Large-input fused kernel: stage the FP8 row scales rank-locally; the m-block's closing unit sends
    # each row's scales as whole lines (needs row_align to leave the scale lines to the row).
    scale_stage: bool = False
    # Large-input fused kernel: every GEMM2 K-step issues its A copy, so its MFMAs no longer wait for it.
    a_dma_uniform: bool = False
    # Large-input fused kernel: drop a GEMM2 unit whose m-block is all padding at its claim (needs parallel_plan).
    skip_pad: bool = False
    # Large-input fused kernel: the GEMM2 epilogue's per-row chain skips most LDS round trips (readlane row ids and
    # weights, DPP scale max, branch-free staged scale store).
    epi_lean: bool = False
    # Large-input fused kernel: each Combine wave starts its token once that token's rows arrived (per-token
    # counts, cleared at launch) instead of after a generation barrier over every rank's GEMM2.
    comb_tok_wait: bool = False
    # Large-input fused kernel (with xcd_chunk): GEMM2 claims walk groups of this many m-blocks, all n-blocks of a
    # group before the next, instead of one n-block over every m-block.
    m_group: int = 0
    # Large-input fused kernel (with m_group): one CTA in early_k leaves GEMM1 for GEMM2 once its claims pass
    # early_p/16 of the GEMM1 work. Small-input kernel: the same, per XCD slice of the GEMM1 work.
    early_k: int = 0
    early_p: int = 8
    # Large-input fused kernel (with early_k and a Stage1 mid_tail_rows): when the rank's tokens give its experts at
    # most mid_tail_rows rows on average, the early CTAs leave after early_p_mid/16 instead (0: off).
    early_p_mid: int = 0
    # With early_k: pick the early CTAs (and every CTA's GEMM1 claim shard) by workgroup id, so each XCD has the same
    # number of early CTAs and both sets cover every shard (False: by arrival ticket).
    early_xcd: bool = False
    # Large-input fused kernel (BM64 1-stage, with a_dma_uniform): a GEMM2 unit copies its m-block's A scales for
    # every K-step into LDS once (LDS DMA past the A slots); the K-steps read them from LDS, not 2 loads per wave.
    ascale_lds: bool = False
    # Large-input fused kernel (BM64, with skip_pad, ascale_lds, scale_stage and an equal Stage1 fold_tail): a GEMM2
    # unit whose next m-block is its expert's tail of at most fold_tail rows also runs those rows (80-row unit, one
    # more MFMA row group and a 16-row epilogue pass); the tail's own units are dropped at their claim (0: off).
    fold_tail: int = 0
    # Large-input fused kernel (with fold_tail, cached W2): a GEMM2 unit whose m-block (with its folded tail) is its
    # expert's only one streams its W2 slab (nt), as no other m-block rereads it from L2.
    nt_single: bool = False
    # Large-input fused kernel (with epi_lean): the P2P rows carry offset int8 (u8 = q + 128) with the same E8M0
    # scale per 32 values instead of E4M3: same bytes, about half the transport rounding error.
    route_i8: bool = False


@dataclass(frozen=True, slots=True)
class MegaMoEConfig:
    stage1: Stage1Config
    stage2: Stage2Config
    p2p_quant: str
    # Fused-kernel knobs (small- and large-input routes only; the staged kernels never read them).
    fused_stage1: FusedStage1Knobs = FusedStage1Knobs()
    fused_stage2: FusedStage2Knobs = FusedStage2Knobs()

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
    # Fused routes this instance builds: small-input capacities and the large-input kernel.
    small_m_mtprs: tuple[int, ...] = ()
    large_m: bool = False

    def route_for(
        self, max_tokens: int, *, small_m: bool = True, large_m: bool = True
    ) -> tuple[str, int]:
        """(route, capacity) of a call whose largest EP rank has ``max_tokens`` rows.

        Every rank passes the same ``max_tokens``, so every rank takes the same
        route. ``small_m`` / ``large_m`` exclude a route the caller cannot use.
        """
        if not 0 <= max_tokens <= self.mtpr:
            raise ValueError(f"max_tokens={max_tokens} must be in [0, {self.mtpr}]")
        if small_m:
            for capacity in self.small_m_mtprs:
                if max_tokens <= capacity:
                    return ROUTE_SMALL_M, capacity
        if large_m and self.large_m and max_tokens >= LARGE_M_MIN_TOKENS:
            return ROUTE_LARGE_M, self.mtpr
        return ROUTE_STAGED, self.mtpr

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


def fused_layout_supported(
    *,
    arch: str,
    world_size: int,
    model_dim: int,
    inter_dim: int,
    experts: int,
    topk: int,
    quant: str,
    swiglu_limit: float,
    act: str = "silu",
    fanout_masks: tuple[int, ...] = (),
) -> bool:
    """Whether the fused kernels support this layout: one of FUSED_LAYOUTS, a8w4 on gfx950.

    The small-input kernel serves every such layout; the large-input kernel the
    capacities large_m_mtprs_for() gives it.
    """
    return (
        arch.split(":", 1)[0] == "gfx950"
        and (world_size, model_dim, inter_dim, experts, topk) in FUSED_LAYOUTS
        and quant == "a8w4"
        and swiglu_limit == 10.0
        and act == "silu"
        and not fanout_masks
    )


def large_m_mtprs_for(
    *, world_size: int, model_dim: int, inter_dim: int
) -> tuple[int, ...]:
    """Instance capacities the large-input kernel is built for on a FUSED_LAYOUTS layout.

    EP8 DeepSeek-V4-Pro takes LARGE_M_MTPRS; EP4 DeepSeek-V4.1-Flash takes V41_FLASH_EP4_LARGE_M_MTPRS, the
    capacities its one tune was measured at (see _tune_v41_flash_ep4_large).
    """
    if (world_size, model_dim, inter_dim) == (4, 5120, 2304):
        return V41_FLASH_EP4_LARGE_M_MTPRS
    return LARGE_M_MTPRS


def small_m_cu_supported(num_cu: int) -> bool:
    """The small-input kernel splits its work over 8 XCDs by XCC id (256-CU parts)."""
    return num_cu == 256


def small_m_mtprs_for(mtpr: int, mtprs=DEFAULT_SMALL_M_MTPRS) -> tuple[int, ...]:
    """Small-input capacities an instance of capacity ``mtpr`` can reach.

    A call picks the smallest capacity >= its max_tokens, so capacities above
    the first one that covers ``mtpr`` are never used.
    """
    mtprs = sorted({int(m) for m in mtprs})
    for m in mtprs:
        if m <= 0 or m & (m - 1) or m > SMALL_M_MAX_MTPR:
            raise ValueError(
                f"small_m_mtprs: {m} must be a power of two <= {SMALL_M_MAX_MTPR}"
            )
    reachable = [m for m in mtprs if m < mtpr]
    reachable += [m for m in mtprs if m >= mtpr][:1]
    return tuple(reachable)


def nearest_token_bucket(tokens: int, buckets=TOKEN_BUCKETS) -> int:
    if tokens <= 0:
        raise ValueError(f"tokens must be positive, got {tokens}")
    index = bisect_left(buckets, tokens)
    if index == 0:
        return buckets[0]
    if index == len(buckets):
        return buckets[-1]
    lower, upper = buckets[index - 1], buckets[index]
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
    world_size: int = 8


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


def _tune_v41_flash_ep2(config: MegaMoEConfig, ctx: _TuningContext) -> MegaMoEConfig:
    """DeepSeek-V4.1-Flash at EP2 (192 experts per rank), the two buckets swept
    on gfx950 (bench_mega_moe_v2, 384 and 16384 source tokens a rank): SBM32
    halves the decode padding and Stage2's tile with it."""
    if ctx.bucket == 512:
        config = _replace_config(
            config, stage1={"sort_block_m": 32, "tile_n": 512, "num_waves": 8}
        )
    elif ctx.bucket == 16384:
        config = _replace_config(
            config,
            stage1={"sort_block_m": 128, "tile_n": 512, "num_waves": 4},
            stage2={"persist_cu": 256},
        )
    return config


def _tune_v41_flash_ep8(config: MegaMoEConfig, ctx: _TuningContext) -> MegaMoEConfig:
    if ctx.apply_fp4 or ctx.world_size != 8:
        return config
    if ctx.fixed_slot:
        if ctx.bucket in (16, 32):
            return _replace_config(config, stage1={"tile_n": 256, "grid_mult": 2})
        if ctx.bucket == 128:
            return _replace_config(config, stage1={"tile_n": 256})
        return config
    # Finite producers retire into consumers; the wider copy cohort reduces
    # compact decode's payload-readiness tail without changing its wire layout.
    if (ctx.mtpr == 32768 and ctx.bucket <= 512) or (
        ctx.mtpr in (256, 512) and ctx.bucket == ctx.mtpr
    ):
        config = _replace_config(config, stage1={"num_dispatch_cu": 224})
    if ctx.mtpr == 8192 and ctx.bucket == 128:
        return _replace_config(config, stage1={"num_dispatch_cu": 160})
    if ctx.mtpr not in (4096, 8192, 16384, 32768) or ctx.bucket < 4096:
        return config
    # The plain deep pipeline outperforms the default aligned-pair schedule.
    # The existing 32K skew path benefits from the full persistent M cohort.
    return _replace_config(
        config,
        stage2={
            "b2stage": True,
            "deep_a_pipeline": True,
            "aligned_pair": False,
            "pair_cu": 0,
            "skew_cu": 240 if ctx.bucket == 32768 else config.stage2.skew_cu,
        },
    )


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
    fp4_prelude: bool = False


_NETWORKS = (
    NetworkProfile(
        "r1",
        R1_EXPERTS_PER_RANK,
        R1_MODEL_DIM,
        R1_INTER_DIM,
        _tune_r1,
        fp4_prelude=True,
    ),
    NetworkProfile(
        "v4_pro",
        REFERENCE_EXPERTS_PER_RANK,
        V4_MODEL_DIM,
        V4_INTER_DIM,
        _tune_v4_pro,
        V4_EPLB_EXPERTS_PER_RANKS,
        fp4_prelude=True,
    ),
    NetworkProfile(
        "kimi_k3", KIMI_K3_EXPERTS_PER_RANK, K3_MODEL_DIM, K3_INTER_DIM, _tune_kimi_k3
    ),
    NetworkProfile(
        "v41_flash_ep2",
        192,
        5120,
        2304,
        _tune_v41_flash_ep2,
        fp4_prelude=True,
    ),
    NetworkProfile("v41_flash_ep8", 48, 5120, 2304, _tune_v41_flash_ep8),
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
        world_size=world_size,
    )
    # Keep FP4 base adjustments opt-in: adding an A8-only profile must not
    # silently retune its FP4 path. EPLB families inherit their profile's prelude.
    if apply_fp4 and profile is not None and profile.fp4_prelude:
        config = _fp4_common_prelude(config, ctx)
    if profile is not None and experts_per_rank == profile.experts_per_rank:
        config = profile.tuner(config, ctx)
    if (
        a_dtype == ACTIVATION_FP4
        and not fixed_slot_dispatch
        and mtpr >= INDEXED_PAYLOAD_MIN_MTPR
    ):
        # Every rank here uses the source-indexed payload. The async A-copy
        # (ATileLoader.prefetch_to_lds) resolves each row's source key per thread,
        # so it serves indexed FP4 too -- the old "indexed FP4 must be sync" rule
        # was conservative (validated: relL2 unchanged, no OOB). Keep async on: it
        # is the race-safe path (only the SYNC reg->LDS copy races at SBM32+nw8)
        # and it reclaims the small-bucket cost that forcing sync would add.
        config = _replace_config(config, stage1={"async_a_copy": True})
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
    small_m_mtprs: tuple[int, ...] = (),
    large_m: bool = False,
) -> MegaMoEBundlePlan:
    """Deduplicate variants while keeping Stage1/Stage2 selection atomic.

    ``small_m_mtprs`` / ``large_m`` add the fused routes; the caller has checked
    the layout supports them (:func:`fused_layout_supported`).
    """
    if mtpr <= 0 or mtpr & (mtpr - 1):
        raise ValueError(f"mtpr={mtpr} must be a positive power of two")
    if large_m and mtpr not in LARGE_M_MTPRS:
        raise ValueError(
            f"the large-input kernel supports mtpr in {LARGE_M_MTPRS}, got {mtpr}"
        )
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
        small_m_mtprs=small_m_mtprs_for(mtpr, small_m_mtprs) if small_m_mtprs else (),
        large_m=large_m,
    )


# Small- and large-input (fused) routes. Their kernels take the same config
# types as the staged path, tuned on their own for the one layout they support
# (fused_layout_supported): the small-input kernel per token bucket, the
# large-input kernel one config per capacity.

# Small-input token bucket from which the expert m-tile is 64 rows: at ~35+
# rows per expert a 32-row tile splits every expert in two and streams its
# weights twice.
SMALL_M_BM64_MIN_BUCKET = 256


def _small_m_stage1(bucket: int, expert_waves: int = 1) -> Stage1Config:
    grid_mult = max(1, bucket // 4) if bucket <= 16 else 3
    return Stage1Config(
        sort_block_m=64 if bucket >= SMALL_M_BM64_MIN_BUCKET else 32,
        tile_n=256 if bucket <= 8 else 128,
        num_waves=4,
        grid_mult=grid_mult,
        # A layout with more than 64 experts a rank gives every 64-expert wave its own producers.
        num_dispatch_cu=min(224, _fixed_dispatch_cu(bucket) * expert_waves),
        mfma_amajor=False,
        async_a_copy=False,
        use_tile_resource=bucket <= 16,
        b_nt=0 if bucket == 1 else 3,
        waves_per_eu_hint=1 if bucket == 16 else 2,
    )


def _small_m_stage2(bucket: int) -> Stage2Config:
    # GEMM2 runs in the kernel's own grid (never persistent) with NT B loads.
    return Stage2Config(
        block_m=64 if bucket >= SMALL_M_BM64_MIN_BUCKET else 32,
        block_n=256 if bucket in (1, 4, 64) or bucket >= 128 else 128,
        persist=False,
        persist_cu=0,
        use_nt=True,
        persist_strided=bucket == 512,
    )


def _large_m_config(expert_waves: int = 1) -> MegaMoEConfig:
    stage1 = Stage1Config(
        sort_block_m=128,
        tile_n=512,
        num_waves=8,
        grid_mult=1,
        num_dispatch_cu=96 * expert_waves,
        mfma_amajor=True,
        async_a_copy=True,
        use_tile_resource=True,
        b_nt=0,
        work_shards=4,
        payload_chunk_rows=384,
    )
    stage2 = Stage2Config(
        block_m=64,
        block_n=256,
        persist=True,
        persist_cu=240,
        use_nt=False,
        skew_cu=96,
    )
    return MegaMoEConfig(stage1=stage1, stage2=stage2, p2p_quant="fp8_blockwise_1x32")


def _tune_v41_flash_ep4_small(config: MegaMoEConfig, bucket: int) -> MegaMoEConfig:
    """EP4 DeepSeek-V4.1-Flash small-input kernel (96 experts per rank), measured on gfx950."""
    # One dispatch row per (token, destination) instead of per route: the XGMI-bound dispatch sends ~45% fewer
    # rows (-3% at 128 tokens, -6% at 512, neutral at 1-32). The plan reaches every CTA in one hop: the plan CTA
    # writes it through to memory and publishes the flag without an L2 writeback, and each CTA waits on the flag
    # itself instead of behind its XCD's acquire and republish (-1.8% at 1 token, neutral at 8-512). Every CTA
    # builds the plan from the expert counts itself once the sources are done, so GEMM1 starts without the plan
    # CTA's metadata stores and flag hop, and takes its tile metadata from LDS (-3% at 1 token, -2% at 8, -1.2% at
    # 32, -0.3..-0.6% at 128-512); the counters' reset point moves to the combine. With at most 8 route-carrying
    # producers (1-2 tokens), each one adds its share of the source flag to every destination after its own
    # drain, skipping the last-producer counter hop (-1.5% at 1 token, -1% at 2, neutral at 16; 24 producers at 16
    # tokens: +1.3%). These change what the ranks of a launch exchange, and the ranks of a launch may pick
    # different buckets: every bucket sets them, as it sets the runtime's per-token counters (used by the
    # tok_done buckets, cleared by every launch) and the BF16 rows of the first tokens (sender and combine agree).
    stage1 = replace(
        config.fused_stage1,
        dedup_rows=True,
        plan_direct=True,
        plan_wt=True,
        plan_local=True,
        direct_done=8,
        tok_counters=True,
        bf16_head_tokens=4,
        # GEMM2 keeps the pointers it reads from the dispatch table in SGPRs: every load of its K-loop sat in a
        # waterfall loop (-1.6..-2.4% at 1-32 tokens, where the GEMM2 K-loop is latency-bound; -0.7..-0.1% at
        # 64-128). The 64-row tiles of 256-512 tokens spill with it (+5..+7%).
        g2_uniform=bucket <= 128,
    )
    if bucket <= 4:
        # GEMM1 runs one round of tiles, so its epilogue is on the critical path: the large-input lean epilogue
        # (8 columns a lane, no serial store waits, fast SiLU) cuts 3% at 1 and 4 tokens (neutral at 8-16).
        # The combine takes a token once its routes' GEMM2 rows are counted on it, not after every rank's last CTA
        # publishes its flag: the ticket -> flag -> reduce hops leave the tail (-3% at 1 token, -2.2% at 2 and 4
        # with the lane-fed scatter it needs; +0.8% at 8, +1.6% at 16, +2.9% at 128 from the per-unit drains).
        stage1 = replace(stage1, epi_evec=8, epi_lean=3, g2_epi_lean=2, tok_done=True)
    config = replace(config, fused_stage1=stage1)
    if bucket == 1:
        # Streaming GEMM1 weight loads, as the larger buckets (-8%). Head pointers in SGPRs: 160 -> 17 waterfall
        # loops, 244 -> 217 VGPR (-2%; neutral at 4-128 tokens, +4% at 256-512). GEMM2 scatter rows take their ids,
        # weights and peer bases from lanes (-1.1%; -0.7% at 4 tokens, neutral at 16-128). GEMM1 loads a K-step's A
        # rows two steps ahead into registers and W1 two steps ahead in a fully unrolled K-loop, one wave a SIMD
        # (-2.4%; either alone, or W1 ahead in the rolled loop, is slower).
        return replace(
            config,
            stage1=replace(config.stage1, b_nt=3, waves_per_eu_hint=1),
            fused_stage1=replace(
                stage1, head_uniform=True, g1_a_ahead=2, g1_lookahead=2
            ),
        )
    if bucket == 16:
        # Half the GEMM1 n-tiles (tile_n 256, -3.5%) and GEMM2 block_n 256 (-3%): longer weight slabs per tile.
        return replace(
            config,
            stage1=replace(config.stage1, tile_n=256),
            stage2=replace(config.stage2, block_n=256),
        )
    if bucket == 128:
        # Half the GEMM1 n-tiles of tile_n 128: each streams a longer W1 slab (-4%).
        return replace(config, stage1=replace(config.stage1, tile_n=256))
    if bucket == 512:
        # Half the CTAs leave GEMM1 for GEMM2 after a claim past half their XCD's GEMM1 slice: the XGMI-bound
        # GEMM2 scatter starts while GEMM1 still runs (-2.7%; a quarter: +1.1%, neutral to +1.2% at 128 tokens).
        return replace(
            config, fused_stage2=replace(config.fused_stage2, early_k=2, early_p=8)
        )
    return config


def _tune_v41_flash_ep4_large(config: MegaMoEConfig) -> MegaMoEConfig:
    """EP4 DeepSeek-V4.1-Flash large-input kernel (96 experts per rank), measured on gfx950 at MTPR 8192.

    It serves 8192 and 16384 (V41_FLASH_EP4_LARGE_M_MTPRS) unchanged: the thresholds below are rows a rank of the launch
    (payload_small_tokens, presend_rows, the early_p_mid bound) and the payload geometry is sized by the plan, so a
    launch of N rows runs the same schedule at any capacity >= N. The capacity only sizes buffers (pair-order
    buckets, row stamps, P2P rows), all inside the 32-bit buffer offsets up to 16384 (LargeMRuntime checks them).
    At 1024, 2048 and 4096 the kernel faults with an illegal memory access on gfx950, so those stay staged.
    """
    # One payload producer per destination leaves its ~16 MB a ~750 us copy at 2048 tokens a rank;
    # 12 split the destination's experts (-24%). XCD-affine GEMM2 claims share each W2 slab in one
    # XCD's L2 across the expert's m-blocks (-5%). 128-byte P2P rows keep each GEMM2 scatter store
    # off the cache lines its neighbouring rows' stores touch (-1%). An expert just past 128 rows pads a
    # second m-tile that is mostly padding; it runs 32 rows instead of 128 (-6%). Row scales leave once per
    # row as whole lines from the m-block's last n-block, not as ten 16-byte partial-line writes (-2.5%).
    # A tail of at most 16 rows rides on its expert's previous m-tile, sharing its W1 reads (-1.2%; 32
    # rows spill the K-loop). All waves of the plan CTA plan the 96 experts, not one (-1.4%), and write
    # each expert's padding rows together (-1%). The GEMM1 epilogue quantizes 8 columns a lane and stores
    # them as one 8-byte write (-1.4%). A token row goes to each destination once, from the route to its
    # lowest expert there (sent first), and GEMM1 waits for the row's arrival stamp (-2.8%). A payload
    # chunk's chunk_done RMW reuses its tile-ready release (-1.8%), and an expert's last chunk skips the
    # unread payload_ready release and RMW (-1.9%). One wave, not every wave, invalidates the caches once a
    # GEMM1 tile's rows landed (-1.1%). A chunk's drained peer stores (uncached heap) need no release before its
    # tile-ready RMW (-1.7%). GEMM1 reads its A operands in inline asm, so a K-step's first read does not wait for
    # the next step's A copy, which the compiler cannot tell from this step's buffer (-1.5%). GEMM2 drops an
    # all-padding m-block at its claim instead of after its GEMM1 tile, and issues its A copy on every K-step so
    # the MFMAs do not wait for the copy two steps ahead (-1.4% together). A payload chunk resolves its routes one
    # per thread and copies the sent rows with their loads ahead of the stores, so 8 producers a destination
    # keep up and the other 16 CTAs run GEMM1 from the start (-1.3%; 8 with the row-at-a-time copy: +1.1%).
    # The plan sizes the payload geometry off its serial path, fences once instead of in every wave, and loads
    # its inputs in one round (-1%). The token rows of the first payload round (each destination's first 8
    # experts) cross XGMI while the plan runs, so GEMM1 starts sooner (-2.2%; 32 experts crowd the plan: +5%).
    # Each Combine wave starts once its own token's rows arrived, not after every rank's GEMM2 (-1.8%): the ranks
    # that finish GEMM2 before the most loaded one reduce most of their tokens while it still runs.
    # GEMM2 is XGMI-bound and the links idle once the dispatch is done, while GEMM1 still runs: half the CTAs
    # leave GEMM1 for GEMM2 after a claim past a quarter of the GEMM1 work, with GEMM2 claiming 8 m-blocks
    # at a time in all n-blocks, so the early units find their GEMM1 m-tiles done (-7.5%).
    # Count words carry the launch tag, so the plan spins on them instead of a count_done round, and each route's
    # histogram RMW is its slot in a per-expert pair-order bucket, so the producers start without the prefix /
    # cursor pass; warp 1 sizes the geometry while the other waves plan (-1.2%). GEMM1 stages its A scales in K-loop
    # order, so a K-step reads each row group's scale word with one ds_read_b32 instead of four serial ds_read_u8
    # and a VALU pack (-1.8%). Early GEMM2 CTAs are picked per XCD, and the GEMM2 epilogue's per-row chain skips
    # most LDS round trips (readlane row ids, DPP scale max) (-1.2% together). The GEMM1 epilogue quantizes and
    # stores without scratch-reload store drains and with a v_rcp SiLU, and a GEMM2 unit copies its m-block's
    # A scales to LDS once instead of two scale loads per K-step (-1.6% together, each alone below 1%).
    # An expert's GEMM2 tail of at most 16 rows (~40% of the experts have 129-144 rows) rides on its previous
    # m-block as an 80-row unit instead of running ten 64-row units of its own (-1.4%).
    # An expert's last GEMM1 m-tile with 33-64 real rows runs 64 rows, not 128 (-4% at 512 tokens a rank,
    # where most experts have one ~32-row tile; -1.9% at 1024, neutral at 256-384 and 2048).
    # GEMM1 streams its W1 loads (nt), as the small buckets: -2% at 512 and 1024, where the short and mid
    # tiles wait on W1, -0.7% at 2048.
    # A GEMM2 unit whose m-block is its expert's only one streams W2 (nt) as well: no other m-block rereads the
    # slab from L2 (-5% at 512, -4% at 1024, -0.7% at 2048, where most experts keep cached loads).
    # With W2 streamed, up to 64 rows an expert (1024 tokens a rank) the early CTAs leave GEMM1 after 6/16 of its
    # work instead of 4/16 (-2% at 512, -1.4% at 1024; the later start loses 1.2% at 1280-2048, so it stays 4/16
    # there).
    # The P2P rows carry offset int8 instead of E4M3 under the same E8M0 scales: relL2 vs the fp32 reference
    # 0.0583 -> 0.0525 at 512-2048 tokens, at no measurable cost (same bytes; the extra cvt sits off the XGMI bound).
    # The presend covers 16384 // tokens experts (8..64): about the same rows at every token count, where 8
    # experts fit 2048 tokens only (-1% at 512, -1.4% at 1024, -0.7% at 256, unchanged at 2048).
    # Up to 640 tokens a rank each destination gets 12 payload producers, not 8: the GEMM1 tiles that wait on
    # the dispatch are most of the first round there (-1.2% at 512, -0.8% at 256; neutral at 768-2048).
    # The launch head is serial up to the first GEMM1 tile: the producers count their routes before the gate
    # (written through, so group_done needs no L2 writeback), the owner clears its counters while the handshake
    # travels, and producers read the destination's rows per tile / ready order once (-1% at 256 and 512).
    return replace(
        config,
        stage1=replace(config.stage1, b_nt=3),
        fused_stage1=replace(
            config.fused_stage1,
            payload_min_blocks=8,
            batch_rows=True,
            short_tail_rows=32,
            fold_tail=16,
            mid_tail_rows=64,
            parallel_plan=True,
            epi_evec=8,
            dedup_rows=True,
            merge_release=True,
            skip_payload_ready=True,
            one_row_acquire=True,
            no_chunk_release=True,
            a_asm_reads=True,
            payload_small_blocks=12,
            payload_small_tokens=640,
            lean_plan=True,
            presend_experts=8,
            presend_rows=16384,
            tag_counts=True,
            bucket_pairs=True,
            ascale_packed=True,
            epi_lean=3,
            early_hist=True,
            early_clear=True,
            peer_hoist=True,
        ),
        fused_stage2=replace(
            config.fused_stage2,
            xcd_chunk=32,
            row_align=128,
            scale_stage=True,
            a_dma_uniform=True,
            skip_pad=True,
            comb_tok_wait=True,
            m_group=8,
            early_k=2,
            early_p=4,
            early_p_mid=6,
            early_xcd=True,
            epi_lean=True,
            ascale_lds=True,
            fold_tail=16,
            nt_single=True,
            route_i8=True,
        ),
    )


def _is_v41_flash_ep4(world_size, experts_per_rank, model_dim, inter_dim) -> bool:
    return (world_size, experts_per_rank, model_dim, inter_dim) == V41_FLASH_EP4_LAYOUT


@cache
def _fused_config(
    bucket: int,
    mtpr: int,
    world_size: int,
    experts_per_rank: int,
    model_dim: int,
    inter_dim: int,
) -> MegaMoEConfig:
    ep4 = _is_v41_flash_ep4(world_size, experts_per_rank, model_dim, inter_dim)
    expert_waves = (experts_per_rank + 63) // 64
    if mtpr in LARGE_M_MTPRS:
        config = _large_m_config(expert_waves)
        return _tune_v41_flash_ep4_large(config) if ep4 else config
    config = MegaMoEConfig(
        stage1=_small_m_stage1(bucket, expert_waves),
        stage2=_small_m_stage2(bucket),
        p2p_quant="none",
    )
    return _tune_v41_flash_ep4_small(config, bucket) if ep4 else config


def select_fused_config(
    tokens: int,
    mtpr: int,
    *,
    world_size: int = 8,
    experts_per_rank: int = REFERENCE_EXPERTS_PER_RANK,
    model_dim: int = V4_MODEL_DIM,
    inter_dim: int = V4_INTER_DIM,
) -> MegaMoEConfig:
    """Small-input (MTPR <= SMALL_M_MAX_MTPR) or large-input (MTPR in LARGE_M_MTPRS)
    config of a fused_layout_supported() layout (default: EP8 DeepSeek-V4-Pro) for
    ``tokens`` rows on a rank; an idle rank takes the 1-token config."""
    if mtpr <= 0 or mtpr & (mtpr - 1):
        raise ValueError(f"mtpr={mtpr} must be a positive power of two")
    large_mtprs = large_m_mtprs_for(
        world_size=world_size, model_dim=model_dim, inter_dim=inter_dim
    )
    if mtpr > SMALL_M_MAX_MTPR and mtpr not in large_mtprs:
        raise ValueError(f"no small- or large-input config for mtpr={mtpr}")
    if not 0 <= tokens <= mtpr:
        raise ValueError(f"tokens={tokens} must be in [0, {mtpr}]")
    if not 0 < experts_per_rank <= 128:
        raise ValueError(
            f"experts_per_rank must be in [1, 128], got {experts_per_rank}"
        )
    return _fused_config(
        nearest_token_bucket(max(tokens, 1), FUSED_TOKEN_BUCKETS),
        mtpr,
        world_size,
        experts_per_rank,
        model_dim,
        inter_dim,
    )
