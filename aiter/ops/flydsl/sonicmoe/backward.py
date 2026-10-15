# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
# ruff: noqa: B008, SIM102

"""First-stage training backward for the gfx950 SonicMoE operator.

This module implements the dense BF16/FP16 fixed-K and flat ragged-route
training contracts needed by the SonicMoE ROCm adapter. All seven forward
activations and optional expert bias are supported. It does not reuse the
inference workspace. Backward-owned sorting/workspace and invocation-owned
forward state keep retained graphs and overlapping forward calls safe.

The implementation is entirely FlyDSL on device.  Standalone backward
recomputes both forward intermediates; fixed-K BF16 SwiGLU backward can instead
consume an invocation-owned forward state and skip W1 recomputation.  Eligible
no-bias state paths also compute an A16 ``q = dout @ W2`` once and fuse route
score, activation-derivative, and dW2-input work without materializing the W2
projection.  The fused exact-row schedule covers the fully grouped short-token
BM16 path and the tuned T4096/H4096/I2048/E64/K8 BM64 path.  Other grouped W1,
W2, dA, dW1, dW2, and dX contractions use gfx950 MFMA kernels where their
individual policies allow; remaining matrix products use the general A16W16
GEMM.  Small FlyDSL kernels implement routing metadata, gather/scatter,
activation derivatives, and the top-K reduction.  Legacy dtype, activation,
bias, ragged-route, and untuned long-token contracts retain the conservative
host-dispatched fallback.
"""

import functools
import math
from collections.abc import Callable
from typing import TYPE_CHECKING

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr import arith, const_expr, gpu, range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.utils.arith import _to_raw as _raw
from flydsl.runtime.device import get_rocm_arch

from aiter.ops.flydsl.kernels import buffer_ops
from aiter.ops.flydsl.kernels.gemm_a16w16_gfx950 import gemm_a16w16
from aiter.ops.flydsl.kernels.kernels_common import get_warp_size
from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled

from .dynamic_policy import (
    E16RoutePolicy,
    e16_route_policy_representative,
    select_e16_route_policy,
)
from .kernels.grouped_a16w16 import compile_sonic_grouped_a16w16_nn
from .kernels.grouped_da import compile_grouped_da_gfx950
from .kernels.grouped_scheduler import (
    build_compact_m_tile_descriptors,
    build_exact_m_tile_queue,
    exact_m_tile_queue_elements,
    exact_m_tile_queue_upper_bound,
    fixed_compact_m_tile_descriptor_upper_bound,
    ragged_compact_m_tile_descriptor_upper_bound,
)
from .kernels.grouped_tn import (
    active_expert_descriptor_capacity,
    active_expert_queue_elements,
    build_active_expert_queue_flydsl,
    finalize_hot_splitk_flydsl,
    grouped_tn_from_metadata_flydsl,
    grouped_tn_from_queue_flydsl,
    grouped_tn_splitk_from_queue_flydsl,
    hot_split_descriptor_capacity,
    zero_inactive_weight_grads_flydsl,
    zero_weight_grads_adaptive_flydsl,
)
from .kernels.memory import atomic_add
from .kernels.ragged_sorting import moe_ragged_sorting_flydsl
from .kernels.reduction import compile_moe_reduction
from .kernels.sorting import moe_sorting_flydsl, moe_sorting_get_workspace_size
from .kernels.stage1 import (
    _gelu_tanh_f32,
    _relu_f32,
    _sigmoid_f32,
    _tanh_f32,
    compile_gemm1_a16w4_port,
)
from .kernels.stage2 import compile_gemm2_a16w4_port

if TYPE_CHECKING:
    from .op import SonicMoEConfig


_BLOCK_THREADS = 256
# Backward materializes expert segments and contracts across their padded row
# dimension with the generic A16 GEMM below.  That GEMM uses BLOCK_K=64 and has
# no K-tail path, so every non-empty segment must remain a multiple of 64.
# Forward's independently tuned route tile is not a legal substitute here.
_BACKWARD_SORT_UNIT = 64
_TOKEN_MASK = 0x00FFFFFF
_MAX_SIGNED_I32 = (1 << 31) - 1
_MAX_BUFFER_BYTE_OFFSET = (1 << 32) - 1
_WARP_SIZE = get_warp_size()
_RED_SLOTS = max(1, (_BLOCK_THREADS + _WARP_SIZE - 1) // _WARP_SIZE)
_DY_COPY_DWORDS = 4
_DY_COPY_BF16_ELEMENTS = 2 * _DY_COPY_DWORDS
_GLU_ACTIVATIONS = frozenset({"swiglu", "geglu", "reglu"})
_SUPPORTED_ACTIVATIONS = frozenset(
    {"swiglu", "geglu", "reglu", "gelu_tanh_approx", "relu", "silu", "relu_sq"}
)

_GEMM_KWARGS = {
    "block_m": 64,
    "block_n": 64,
    "block_k": 64,
    "stages": 2,
    "split_k": 1,
    "m_waves": 2,
    "n_waves": 2,
    "k_waves": 1,
    "group_m": 0,
    "use_half_tile_interleaved": False,
}

# The first device-driven backward GEMM specialization targets the dominant
# BF16 SwiGLU training shape.  A 16-row compute tile preserves the sorter's
# 64-row expert metadata granularity while four K-waves cover gfx950's short-M
# regime.  These are compile-time policy choices, not public ABI knobs.
_GROUPED_W1_BM = 16
_GROUPED_W1_BN = 64
_GROUPED_W1_BK = 64
_GROUPED_W1_K_WAVE = 4
# At the fixed-K policy boundary, a compact real-M-tile queue exposes long
# expert segments as independent CTAs.  BN128 halves the queue grid, while two
# K waves retain enough N parallelism for the production H3584/I512 shape.
_COMPACT_W1_MIN_TOKENS = 64
_COMPACT_W1_BM = 16
_COMPACT_W1_BN = 128
_COMPACT_W1_BK = 64
_COMPACT_W1_K_WAVE = 2
# This BM16/k-wave4 specialization is tuned for short expert segments.  Fixed-K
# routing guarantees at most one edge per (token, expert), so ``tokens`` is a
# distribution-independent upper bound.  Ragged routing permits duplicates and
# therefore uses the total route count as its conservative bound.  Above this
# limit the existing BM64 GEMM has enough M work to amortize host dispatch and
# is substantially faster (for example, balanced T4096/E64 has 512 rows/expert).
_GROUPED_W1_MAX_EXPERT_ROWS = 128

# W2 recompute has the opposite aspect ratio (K=intermediate, N=hidden).  Keep
# its policy independent so gfx950 tuning can evolve without coupling the two
# kernels. BM32/BN256/BK64 is the measured balanced-routing winner for the
# production I=512 contraction on MI350, while remaining within 1 us of the
# best decode profile.
_GROUPED_W2_BM = 32
_GROUPED_W2_BN = 256
_GROUPED_W2_BK = 64
_GROUPED_W2_MAX_EXPERT_ROWS = 128

# dA consumes public row-major W2 directly: [M, H] @ [H, I] -> [M, I].  The
# NN-specialized kernel transposes B in LDS and uses CDNA4 LDSReadTrans16_64b
# before MFMA.  Three measured gfx950 profiles cover decode, sparse short-M,
# and long/hot expert segments.  Legacy fallbacks reuse their existing host
# frequency read; the short hostless path dispatches from the device queue.
_GROUPED_DA_BN = 64
_GROUPED_DA_MAX_EXPERT_ROWS = 4096
_GROUPED_DA_STAGES = 2
_GROUPED_DA_E896_DENSE_STAGES = 3
_GROUPED_DA_E896_DENSE_GRID_CAP = 256

# Keep the first token-major grouped-dW1 rollout on the production E896
# retained-state contract until its cache behavior has been measured broadly.
_DIRECT_GROUPED_DW1_RHS_SHAPE = (4096, 3584, 512, 896, 16)

# Weight-gradient TN contractions are output-stationary: each CTA owns one
# output tile and reduces all rows for one expert before a single BF16 store.
# BK32 is the measured short/ragged-row winner on gfx950.  Exact resource
# bounds make the unused portion of the final K tile read as zero without
# loading the sorter's full 64-row padding.
_GROUPED_DW2_BK = 32
_GROUPED_DW2_PIPELINE_THRESHOLD = 64
_GROUPED_DW2_SPARSE_EXPERTS = 32

# Qwen3-30B-A3B's EP8 training shard reaches this exact fixed-K bucket.  When
# forward retained route-order preactivation, backward has only four matrix
# contractions left (dA, dW2, dW1, and dX).  Keep the initial rollout exact so
# the 8192-row worst-case bound cannot broaden the grouped policies for other
# distributions or for standalone backward, which still recomputes W1/W2.
_E16_FIXED_STATE_GROUPED_SHAPE = (8192, 2048, 768, 16, 1)

# Flat Qwen3-30B-A3B routing has a dynamic local route count after EP
# all-to-all.  Both retained-state and standalone calls use the same bounded
# device queues for every non-empty R.
_E16_FLAT_GROUPED_SHAPE = (2048, 768, 16)
_E16_EXACT_BM = 128
_E16_EXACT_BK = 64
_E16_EXACT_DA_BN = 192
# Expert-major identity rows can be gathered by the exact dA GEMM itself,
# avoiding a full unscaled grad-output copy before the contraction.
_E16_DIRECT_DA_RHS_ENABLED = True
# BN128 was 15% faster than the screenshot's BN256 tile in the balanced
# single-contraction sweep and also exposes twice as much N parallelism for a
# hot expert.  Both contractions share the same BM128 exact-tile queue.
_E16_EXACT_DX_BN = 128
_E16_EXACT_N_WAVES = 4
_E16_FLAT_SEGMENTED_DX_GRID_CAP = 1024
# A hot E16 shard exposes only 24 output tiles with the default 256x256 dW2
# profile.  The 128x128 profile raises that to 96 tiles and is materially
# faster for one/few active experts, while the wider tile remains better for a
# balanced shard.  Select entirely from the device-resident active queue count
# so dynamic routing never synchronizes to the host.
_E16_DW2_SMALL_TILE_MAX_ACTIVE_EXPERTS = 4
_E16_DW2_SMALL_TILE_N = 64
# In a dense shard, keep isolated long experts on the same high-parallelism
# profile instead of forcing them through the balanced 256x256 tile.  The
# frequency predicate is evaluated from the existing device queue, and is
# combined with the sparse-shard predicate inside the small-profile launch.
_E16_DW2_HOT_EXPERT_MIN_ROWS = 1408
# Keep the small profile selective as the shard grows.  Eleven sixty-fourths
# (17.2%) preserves the measured R8192 crossover and prevents several merely
# above-average experts from overwhelming the high-parallelism profile at
# R16384.  Both operands are host-known shape values; selection of individual
# experts remains device-side.
_E16_DW2_HOT_EXPERT_ROUTE_FRACTION_NUMERATOR = 11
_E16_DW2_HOT_EXPERT_ROUTE_FRACTION_DENOMINATOR = 64
# Quantize runtime row cutoffs to the exact-queue M tile.  The numeric cutoff
# is already a runtime scalar (and therefore not a JIT key); quantization also
# prevents tiny rank-to-rank R differences from reclassifying experts whose
# workloads are otherwise identical at GEMM-tile granularity.
_E16_DW2_HOT_EXPERT_ROW_QUANTUM = 128
# dW2 consumes the same hot descriptors before dW1 and reuses the leading
# portion of dW1's larger FP32 partial workspace on the same stream.  The
# smaller dW2 contraction crosses over later than dW1, so only the rank-shared
# XLARGE policy enables its companion split.
_E16_DW2_SPLITK_ENABLED = True
_E16_DW2_SPLIT_BM = 256
_E16_DW2_SPLIT_BN = 256
_E16_DW2_SPLIT_BK = 64
_E16_DW2_SPLIT_M_WAVES = 4
_E16_DW2_SPLIT_N_WAVES = 4
_E16_DW2_SPLIT_STAGES = 2
# Direct token-major dW1 is limited by output-tile parallelism when only one or
# two experts are live.  BN64 doubles its N grid for those sparse shards, while
# three or more active experts provide enough work for the more efficient
# BN128 profile.  Both profiles consume the same device queue and use disjoint
# guards, so this selection never reads the count on the host.
_E16_DW1_NARROW_MAX_ACTIVE_EXPERTS = 2
_E16_DW1_SPLITK_ENABLED = True
# Keep split-queue setup off the latency-sensitive SMALL/MEDIUM policy classes
# even when the per-expert crossover is tuned below their representative size.
_E16_DW1_SPLIT_ROWS = 8192
# One 8192-row partition has no parallelism benefit.  Start immediately above
# that boundary so the 10K--12K experts in a large skewed EP shard split into
# two independent contractions while an exact 8192-row expert stays cold.
_E16_DW1_SPLIT_MIN_HOT_ROWS = 8193
# With three or more live experts, do not enable split-K until the hottest
# expert reaches the measured gfx950 crossover.  The device queue builder
# evaluates this trigger from the resident frequency vector; once enabled it
# still applies the lower per-expert cutoff below.  Separating the shard-wide
# trigger from per-expert eligibility avoids partial-cohort cliffs such as
# hot4 at R=32769 and R=49151.  Express the crossover as a fraction of the
# adaptive split length so the open-ended XLARGE family scales past 134K.
_E16_DW1_DENSE_SPLIT_TRIGGER_NUMERATOR = 7
_E16_DW1_DENSE_SPLIT_TRIGGER_DENOMINATOR = 4
_E16_DW1_SPLIT_SPARSE_MAX_ACTIVE_EXPERTS = 2
_E16_DW1_SPLIT_BM = 128
_E16_DW1_SPLIT_BN = 256
_E16_DW1_SPLIT_BK = 32
_E16_DW1_SPLIT_M_WAVES = 2
_E16_DW1_SPLIT_N_WAVES = 4
_E16_DW1_SPLIT_STAGES = 2
# Finite per-policy scratch ceilings prevent unbounded split metadata.  Actual
# allocations are tightened from the local R, so a shared larger policy does
# not inflate short ranks.  LARGE never exceeds 32K after automatic promotion;
# XLARGE is open ended and adapts its runtime split length while retaining one
# bounded descriptor ABI.
_E16_DW1_LARGE_SPLIT_CAPACITY = 8
_E16_DW1_XLARGE_SPLIT_CAPACITY = 32
_E16_DW1_SPLIT_ROW_QUANTUM = 128
# dW1 and dX are independent after the retained-state derivative has produced
# dZ.  Large expert-major shards can overlap them profitably; smaller shards
# keep the serial path because the two-event fork/join cost is not amortized.
# Auxiliary-stream overlap made isolated large-R kernels faster, but exposed
# more NCCL on real EP training and amplified rank arrival skew.  Keep the
# implementation available for explicit experiments while the stable dynamic
# policy defaults to one stream on every capacity class.
_E16_DW_DX_OVERLAP_ENABLED = False

_RoutesSorterMetadata = tuple[
    torch.Tensor,  # sorted_token_ids
    torch.Tensor,  # sorted_route_ids
    torch.Tensor,  # sorted_weights
    torch.Tensor,  # sorted_expert_ids
    torch.Tensor,  # num_valid_ids
    torch.Tensor,  # expert_frequency
    int,  # sorter M padding unit
]


def _use_e16_dw1_dual_profile(
    *,
    direct_rhs: bool,
    e16_flat_grouped: bool,
    metadata_direct: bool,
    route_policy: E16RoutePolicy,
) -> bool:
    """Select device-guarded narrow/dense direct-dW1 profiles.

    The medium capacity class uses one minimax BN128 launch to reduce launch
    pressure.  Every rank can select that class through the same scheduling
    hint; the decision never depends on its exact local route count.
    """

    return (
        direct_rhs
        and e16_flat_grouped
        and not metadata_direct
        and route_policy != E16RoutePolicy.MEDIUM
    )


def _use_e16_dw_dx_overlap(
    *,
    e16_flat_grouped: bool,
    reuse_forward_preactivation: bool,
    flat_identity_dx: bool,
    use_grouped_dw1: bool,
    use_grouped_dx: bool,
    route_policy: E16RoutePolicy,
) -> bool:
    """Select the audited large-route dW1/dX stream overlap."""

    return (
        _E16_DW_DX_OVERLAP_ENABLED
        and e16_flat_grouped
        and reuse_forward_preactivation
        and flat_identity_dx
        and use_grouped_dw1
        and use_grouped_dx
        and route_policy == E16RoutePolicy.XLARGE
    )


def _use_e16_fixed_state_grouped_backward(
    *,
    compute_dtype: str,
    activation: str,
    tokens: int,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    topk: int,
    flat_routes: bool,
    has_bias: bool,
    reuse_forward_preactivation: bool,
) -> bool:
    """Select Qwen3 E16's retained-state, four-contraction backward."""

    return (
        not flat_routes
        and not has_bias
        and reuse_forward_preactivation
        and compute_dtype == "bf16"
        and activation == "swiglu"
        and (tokens, hidden_size, intermediate_size, num_experts, topk)
        == _E16_FIXED_STATE_GROUPED_SHAPE
    )


def _use_e16_flat_grouped_backward(
    *,
    compute_dtype: str,
    activation: str,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    routes: int,
    flat_routes: bool,
    has_bias: bool,
    reuse_forward_preactivation: bool = False,
) -> bool:
    """Select the dynamic E16 grouped path for every non-empty route list."""

    return (
        flat_routes
        and not has_bias
        and compute_dtype == "bf16"
        and activation == "swiglu"
        and (hidden_size, intermediate_size, num_experts) == _E16_FLAT_GROUPED_SHAPE
        and routes > 0
    )


def _use_e16_flat_segmented_dx(
    *,
    e16_flat_grouped: bool,
    reuse_forward_preactivation: bool,
    token_indices_sorted: bool = False,
) -> bool:
    """Select the route-order dX epilogue and token-segment reducer."""

    return e16_flat_grouped and reuse_forward_preactivation and token_indices_sorted


def _use_e16_flat_deduplicated_metadata(
    *,
    e16_flat_grouped: bool,
    reuse_forward_preactivation: bool,
) -> bool:
    """Select the single-builder retained E16 flat metadata path."""

    return e16_flat_grouped and reuse_forward_preactivation


def _grouped_dw2_tuning(
    max_expert_rows: int,
    hidden_size: int,
    intermediate_size: int,
    *,
    active_experts: int | None = None,
) -> tuple[int, int, int, int, int, int]:
    """Return ``(BM, BN, BK, K-pad, M-waves, N-waves)`` for dW2."""

    block_m = next(tile for tile in (256, 128, 64) if hidden_size % tile == 0)
    block_n = next(tile for tile in (256, 128, 64) if intermediate_size % tile == 0)
    # Resource-aware persistent grids let smaller output tiles expose useful
    # second/third/fourth resident CTAs on gfx950.  Decode and sparse short-K
    # routes prefer 128x128; a dense 2-4 row/expert batch benefits from twice
    # as many M tiles while retaining BN256's contiguous output stores.
    supports_128 = hidden_size % 128 == 0 and intermediate_size % 128 == 0
    sparse_short = (
        active_experts is not None
        and active_experts <= _GROUPED_DW2_SPARSE_EXPERTS
        and max_expert_rows <= 128
    )
    if supports_128 and (max_expert_rows <= 1 or sparse_short):
        block_m = 128
        block_n = 128
    elif (
        max_expert_rows <= 4 and hidden_size % 128 == 0 and intermediate_size % 256 == 0
    ):
        block_m = 128
        block_n = 256
    m_waves = 4 if block_m == 256 else 2
    n_waves = 4 if block_n == 256 else 2
    return block_m, block_n, _GROUPED_DW2_BK, 0, m_waves, n_waves


def _grouped_dw2_stages(max_expert_rows: int) -> int:
    """Use triple buffering only once an expert spans multiple K tiles."""

    return 3 if max_expert_rows >= _GROUPED_DW2_PIPELINE_THRESHOLD else 2


def _e16_dw2_hot_profile_min_rows(routes: int) -> int:
    """Return a tile-quantized device row cutoff for the hot dW2 profile."""

    scaled = (
        routes * _E16_DW2_HOT_EXPERT_ROUTE_FRACTION_NUMERATOR
        + _E16_DW2_HOT_EXPERT_ROUTE_FRACTION_DENOMINATOR
        - 1
    ) // _E16_DW2_HOT_EXPERT_ROUTE_FRACTION_DENOMINATOR
    cutoff = max(_E16_DW2_HOT_EXPERT_MIN_ROWS, scaled)
    quantum = _E16_DW2_HOT_EXPERT_ROW_QUANTUM
    return ((cutoff + quantum - 1) // quantum) * quantum


def _e16_hot_split_schedule(
    routes: int,
    num_experts: int,
    route_policy: E16RoutePolicy,
) -> tuple[int, int, int, int]:
    """Return capacity, partition, per-expert, and shard trigger thresholds.

    LARGE and XLARGE provide finite *upper* bounds of eight and 32 descriptors
    respectively.  First choose a quantized partition length which is safe for
    that policy, then tighten the allocation to the descriptor count provable
    from this invocation's real route count.  A one-record minimum preserves
    the same split/finalize launch topology when no expert can cross the hot
    threshold, without reserving a policy-sized FP32 partial tensor.

    For the open-ended XLARGE family, increase the runtime K partition length
    just enough that all legal expert distributions fit the finite descriptor
    bound.  The 128-row quantization prevents insignificant R jitter from
    changing the scalar while preserving substantially more hot-expert
    parallelism than a coarse power-of-two split length.
    """

    if route_policy == E16RoutePolicy.LARGE:
        policy_capacity = _E16_DW1_LARGE_SPLIT_CAPACITY
    elif route_policy == E16RoutePolicy.XLARGE:
        policy_capacity = _E16_DW1_XLARGE_SPLIT_CAPACITY
    else:
        raise ValueError("hot split scheduling requires LARGE or XLARGE policy")

    quantum = _E16_DW1_SPLIT_ROW_QUANTUM
    minimum_units = (_E16_DW1_SPLIT_ROWS + quantum - 1) // quantum

    def thresholds_for(units: int) -> tuple[int, int, int]:
        split_rows = units * quantum
        min_hot_rows = split_rows + 1
        if route_policy == E16RoutePolicy.XLARGE:
            # dW2 has a materially later split crossover than dW1.  XLARGE
            # reuses one queue/scratch buffer for both gradients, so admit only
            # experts which clear dW2's load-scaled threshold.  This keeps a
            # balanced 134K shard on the regular grouped kernels while hot1,
            # hot4, and long-tail shards retain split parallelism.
            min_hot_rows = max(
                min_hot_rows,
                _e16_dw2_hot_profile_min_rows(routes),
            )
        split_activation_rows = max(
            min_hot_rows + 1,
            (
                split_rows * _E16_DW1_DENSE_SPLIT_TRIGGER_NUMERATOR
                + _E16_DW1_DENSE_SPLIT_TRIGGER_DENOMINATOR
                - 1
            )
            // _E16_DW1_DENSE_SPLIT_TRIGGER_DENOMINATOR,
        )
        return split_rows, min_hot_rows, split_activation_rows

    def required_for(units: int) -> int:
        split_rows, min_hot_rows, _ = thresholds_for(units)
        return hot_split_descriptor_capacity(
            routes,
            num_experts,
            split_rows,
            min_hot_rows,
        )

    if required_for(minimum_units) <= policy_capacity:
        split_units = minimum_units
    else:
        low = minimum_units + 1
        high = max(low, (routes + quantum - 1) // quantum)
        while low < high:
            middle = (low + high) // 2
            if required_for(middle) <= policy_capacity:
                high = middle
            else:
                low = middle + 1
        split_units = low

    split_rows, min_hot_rows, split_activation_rows = thresholds_for(split_units)
    required = hot_split_descriptor_capacity(
        routes,
        num_experts,
        split_rows,
        min_hot_rows,
    )
    if required > policy_capacity:
        raise RuntimeError(
            "bounded E16 split schedule under-sized its descriptor storage: "
            f"required={required}, capacity={policy_capacity}"
        )
    return max(1, required), split_rows, min_hot_rows, split_activation_rows


def _use_e16_dw2_dual_profile(
    *,
    use_hostless_grouped: bool,
    use_tn_metadata_direct: bool,
    num_experts: int,
    hidden_size: int,
    intermediate_size: int,
    route_policy: E16RoutePolicy,
) -> bool:
    """Select the device-guarded E16 small/wide dW2 profile pair.

    The medium capacity class uses one 256x256 minimax launch on every rank.
    Other classes retain device-guarded sparse/hot and wide profiles.
    """

    return (
        use_hostless_grouped
        and not use_tn_metadata_direct
        and num_experts == 16
        and hidden_size == 2048
        and intermediate_size == 768
        and route_policy != E16RoutePolicy.MEDIUM
    )


def _launch_grouped_dw2(
    dy: torch.Tensor,
    activation: torch.Tensor,
    expert_frequency: torch.Tensor,
    sorted_expert_ids: torch.Tensor,
    num_valid_ids: torch.Tensor,
    dw2: torch.Tensor,
    active_expert_storage: torch.Tensor | None,
    *,
    use_hostless_grouped: bool,
    use_tn_metadata_direct: bool,
    max_expert_rows: int,
    route_policy: E16RoutePolicy,
    hidden_size: int,
    intermediate_size: int,
    active_experts: int,
    stream: torch.cuda.Stream,
    hot_split_state: (
        tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ]
        | None
    ) = None,
    hot_split_min_rows: int | None = None,
) -> None:
    """Launch the tuned grouped dW2 profiles after dy becomes available."""

    use_e16_policy_geometry = (
        use_hostless_grouped
        and not use_tn_metadata_direct
        and int(expert_frequency.numel()) == 16
        and hidden_size == 2048
        and intermediate_size == 768
    )
    use_e16_short_metadata_geometry = (
        use_hostless_grouped
        and use_tn_metadata_direct
        and int(expert_frequency.numel()) == 16
        and hidden_size == 2048
        and intermediate_size == 768
    )
    profile_rows = (
        e16_route_policy_representative(route_policy)
        if use_e16_policy_geometry
        else max_expert_rows
    )
    if hot_split_state is None:
        if hot_split_min_rows is not None:
            raise ValueError("hot_split_min_rows requires hot_split_state")
        split_min_rows = _E16_DW1_SPLIT_MIN_HOT_ROWS
    else:
        split_min_rows = (
            _E16_DW1_SPLIT_MIN_HOT_ROWS
            if hot_split_min_rows is None
            else hot_split_min_rows
        )
        if split_min_rows <= 0:
            raise ValueError("hot_split_min_rows must be positive")

    if use_e16_policy_geometry and route_policy == E16RoutePolicy.MEDIUM:
        # The 4K--16K dynamic bucket uses one rank-consistent profile.  BN128
        # doubles dW2's output-tile parallelism under expert skew, while the
        # capped queue-3D schedule below keeps balanced shards efficient without
        # reading the device queue count or specializing on the exact route count.
        grouped_dw2_profiles = (
            (
                256,
                128,
                _GROUPED_DW2_BK,
                0,
                4,
                2,
                _grouped_dw2_stages(profile_rows),
                0,
                None,
                0,
                None,
                False,
            ),
        )
    elif _use_e16_dw2_dual_profile(
        use_hostless_grouped=use_hostless_grouped,
        use_tn_metadata_direct=use_tn_metadata_direct,
        num_experts=int(expert_frequency.numel()),
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        route_policy=route_policy,
    ):
        # The split-K companion already owns every expert above its boundary.
        # Clamp the regular-profile crossover to that boundary so lowering the
        # small-tile threshold for normal batches cannot retile split-K's cold
        # complement at R65536 and above.
        # The cutoff is a runtime scalar consumed by device guards, but derive
        # its value from the finite policy representative.  Ranks sharing one
        # policy therefore keep the same device-side profile boundary even
        # when their exact local route counts differ.
        load_scaled_hot_rows = _e16_dw2_hot_profile_min_rows(profile_rows)
        hot_profile_min_rows = (
            max(load_scaled_hot_rows, split_min_rows)
            if hot_split_state is not None
            else load_scaled_hot_rows
        )
        wide_dw2 = _grouped_dw2_tuning(
            profile_rows,
            hidden_size,
            intermediate_size,
            active_experts=16,
        )
        grouped_dw2_profiles = (
            (
                128,
                _E16_DW2_SMALL_TILE_N,
                _GROUPED_DW2_BK,
                0,
                2,
                2,
                _grouped_dw2_stages(profile_rows),
                0,
                _E16_DW2_SMALL_TILE_MAX_ACTIVE_EXPERTS,
                hot_profile_min_rows,
                None,
                True,
            ),
            (
                *wide_dw2,
                _grouped_dw2_stages(profile_rows),
                _E16_DW2_SMALL_TILE_MAX_ACTIVE_EXPERTS + 1,
                None,
                0,
                hot_profile_min_rows - 1,
                False,
            ),
        )
    elif (
        use_hostless_grouped
        and not use_tn_metadata_direct
        and int(expert_frequency.numel()) <= _GROUPED_DW2_SPARSE_EXPERTS
    ):
        # The active-count guard can never select the dense profile when the
        # model has at most 32 experts.  Avoid an unconditional empty launch
        # in E16's latency-sensitive training path.
        sparse_dw2 = _grouped_dw2_tuning(
            profile_rows,
            hidden_size,
            intermediate_size,
            active_experts=_GROUPED_DW2_SPARSE_EXPERTS,
        )
        grouped_dw2_profiles = (
            (
                *sparse_dw2,
                _grouped_dw2_stages(profile_rows),
                0,
                None,
                0,
                None,
                False,
            ),
        )
    elif use_hostless_grouped and not use_tn_metadata_direct:
        # The queue count is already produced by compact W1.  Launch disjoint
        # sparse/dense profiles without a host-side active-count readback.
        sparse_dw2 = _grouped_dw2_tuning(
            profile_rows,
            hidden_size,
            intermediate_size,
            active_experts=_GROUPED_DW2_SPARSE_EXPERTS,
        )
        balanced_dw2 = _grouped_dw2_tuning(
            min(profile_rows, 4),
            hidden_size,
            intermediate_size,
        )
        grouped_dw2_profiles = (
            (
                *sparse_dw2,
                _grouped_dw2_stages(profile_rows),
                0,
                _GROUPED_DW2_SPARSE_EXPERTS,
                0,
                None,
                False,
            ),
            (
                *balanced_dw2,
                _grouped_dw2_stages(min(profile_rows, 4)),
                _GROUPED_DW2_SPARSE_EXPERTS + 1,
                None,
                0,
                None,
                False,
            ),
        )
    else:
        grouped_dw2_profiles = (
            (
                *_grouped_dw2_tuning(
                    profile_rows,
                    hidden_size,
                    intermediate_size,
                    active_experts=active_experts,
                ),
                (
                    2
                    if use_e16_short_metadata_geometry
                    else _grouped_dw2_stages(profile_rows)
                ),
                0,
                None,
                0,
                None,
                False,
            ),
        )

    for (
        dw2_bm,
        dw2_bn,
        dw2_bk,
        dw2_k_padding,
        dw2_mw,
        dw2_nw,
        dw2_stages,
        min_active_experts,
        max_active_experts,
        min_profile_rows,
        max_profile_rows,
        active_guard_or_expert_rows,
    ) in grouped_dw2_profiles:
        grouped_dw2_kwargs = {
            "block_m": dw2_bm,
            "block_n": dw2_bn,
            "block_k": dw2_bk,
            "k_padding": dw2_k_padding,
            "m_waves": dw2_mw,
            "n_waves": dw2_nw,
            "stages": dw2_stages,
            "stream": stream,
        }
        if use_e16_policy_geometry and route_policy == E16RoutePolicy.MEDIUM:
            grouped_dw2_kwargs["use_queue_3d"] = True
        effective_max_profile_rows = max_profile_rows
        effective_min_profile_rows = min_profile_rows
        effective_active_guard = active_guard_or_expert_rows
        if hot_split_state is not None:
            # The queue builder may disable split-K for a dense cohort without
            # changing the host launch graph.  Keep the normal small/wide dW2
            # profile predicates, then exclude only the experts present in an
            # active split queue through its device-resident count.
            grouped_dw2_kwargs["hot_expert_storage"] = hot_split_state[1]
            grouped_dw2_kwargs["hot_split_min_rows"] = split_min_rows
        if use_tn_metadata_direct:
            grouped_tn_from_metadata_flydsl(
                dy,
                activation,
                expert_frequency,
                sorted_expert_ids,
                num_valid_ids,
                dw2,
                **grouped_dw2_kwargs,
            )
        else:
            assert active_expert_storage is not None
            grouped_tn_from_queue_flydsl(
                dy,
                activation,
                expert_frequency,
                active_expert_storage,
                dw2,
                min_active_experts=min_active_experts,
                max_active_experts=max_active_experts,
                min_expert_rows=effective_min_profile_rows,
                max_expert_rows=effective_max_profile_rows,
                active_guard_or_expert_rows=effective_active_guard,
                **grouped_dw2_kwargs,
            )

    if hot_split_state is not None:
        split_queue, hot_queue, partials = hot_split_state
        grouped_tn_splitk_from_queue_flydsl(
            dy,
            activation,
            expert_frequency,
            split_queue,
            partials,
            block_m=_E16_DW2_SPLIT_BM,
            block_n=_E16_DW2_SPLIT_BN,
            block_k=_E16_DW2_SPLIT_BK,
            m_waves=_E16_DW2_SPLIT_M_WAVES,
            n_waves=_E16_DW2_SPLIT_N_WAVES,
            stages=_E16_DW2_SPLIT_STAGES,
            stream=stream,
        )
        finalize_hot_splitk_flydsl(
            split_queue,
            hot_queue,
            partials,
            dw2,
            stream=stream,
        )


def _grouped_da_tuning(
    max_expert_rows: int, hidden_size: int
) -> tuple[int, int, int, int, int]:
    """Return ``(BM, BN, BK, m_waves, n_waves)`` for grouped dA."""

    if max_expert_rows <= 1 and hidden_size % 128 == 0:
        return (16, _GROUPED_DA_BN, 128, 1, 4)
    if max_expert_rows <= 16:
        return (32, _GROUPED_DA_BN, 64, 2, 2)
    return (64, _GROUPED_DA_BN, 64, 2, 2)


# dX has the row-major NN shape ``[M, 2I] @ [2I, H]``.  BM16 avoids doing the
# sorter's full 64-row padding for short experts.  The production T4096 shape
# instead benefits from one BM64 tile per sorter block: it cuts repeated W1
# slab loads while retaining enough M work per CTA.  That shape gets its own
# descriptor queue so the latency-sensitive BM16 W1/short-dX queue remains
# unchanged.  A persistent four-workgroup-per-CU launch bound (1024 on
# MI350/MI355X) keeps large descriptor grids resident without a long tail.
_GROUPED_DX_BM = 16
_GROUPED_DX_BK = 64
_GROUPED_DX_STAGES = 2
_GROUPED_DX_GRID_CAP = 1024
_GROUPED_DX_DENSE_EXPERTS = 256
_LARGE_GROUPED_DX_BM = 64
_LARGE_GROUPED_DX_BN = 256
_LARGE_GROUPED_DX_N_WAVES = 4
_GROUPED_DX_M_REUSE_SHAPE = (4096, 3584, 512, 896, 16)
_GROUPED_DX_M_REUSE_MIN_ACTIVE_EXPERTS = 33
_LARGE_GROUPED_DX_SHAPES = frozenset(
    {
        # Dense prefill/training bucket used by the standalone FlyDSL tests.
        (4096, 4096, 2048, 64, 8),
        # Default SonicMoE ROCm adapter bucket.  Without the device queue this
        # shape falls back to one dX GEMM launch per active expert (up to 896).
        (4096, 3584, 512, 896, 16),
    }
)
# The first long-token hostless rollout stays narrower than the large-dX
# policy.  In particular, the E64/H4096 bucket has not yet been audited for
# distribution-independent dA/dW scheduling without the legacy frequency
# readback.  The default ROCm adapter bucket below has a retained-state dataflow
# in which every contraction and row transform already consumes a device queue
# or device-resident extent.
_HOSTLESS_LARGE_STATE_SHAPES = frozenset({(4096, 3584, 512, 896, 16)})


def _e16_exact_dx_grid(logical_grid: int, grid_cap: int) -> int:
    """Bound E16 dX to its finite persistent-grid occupancy cap."""

    return min(grid_cap, logical_grid)


def _e16_exact_tile_launch_bound(
    routes: int,
    num_experts: int,
) -> int:
    """Return the actual-R exact-queue bound used only for launch grids."""

    return exact_m_tile_queue_upper_bound(
        routes,
        num_experts,
        _E16_EXACT_BM,
    )


def _grouped_da_hostless_profiles(
    *,
    tokens: int,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    topk: int,
    max_expert_rows: int,
):
    """Return sparse/dense device-guarded dA profiles for hostless backward."""

    shape = (tokens, hidden_size, intermediate_size, num_experts, topk)
    if shape in _HOSTLESS_LARGE_STATE_SHAPES:
        # The E896 production bucket is dominated by repeated W2 fetches.  A
        # wider M tile amortizes each slab across more routes while preserving
        # the per-wave BM64 profile's two M repeats.  Dense routing uses eight
        # waves (MW4/NW2); sparse hot routing uses BM256 with MW8/NW1 so the B
        # tile remains exactly covered by the 512-thread async-load layout.
        # Triple buffering raises the dense BM128 profile from 48 to 72 KiB
        # LDS.  LDS alone fits two CTAs/CU, but its combined VGPR/AGPR usage
        # quantizes residency to one; a 256-CTA persistent grid fills gfx950.
        # Keep sparse BM256 double-buffered: its 120 KiB stage-3 footprint
        # lowers occupancy and regresses instruction-wait/VMEM-latency PMCs.
        sparse_tuning = (256, _GROUPED_DA_BN, 64, 8, 1)
        dense_tuning = (128, _GROUPED_DA_BN, 64, 4, 2)
        sparse_stages = _GROUPED_DA_STAGES
        dense_stages = _GROUPED_DA_E896_DENSE_STAGES
        dense_grid_cap = _GROUPED_DA_E896_DENSE_GRID_CAP
    else:
        sparse_tuning = _grouped_da_tuning(max_expert_rows, hidden_size)
        dense_tuning = _grouped_da_tuning(min(max_expert_rows, 2), hidden_size)
        sparse_stages = dense_stages = _GROUPED_DA_STAGES
        dense_grid_cap = None
    return (
        (*sparse_tuning, True, 0, _GROUPED_DW2_SPARSE_EXPERTS, sparse_stages, None),
        (
            *dense_tuning,
            False,
            _GROUPED_DW2_SPARSE_EXPERTS + 1,
            None,
            dense_stages,
            dense_grid_cap,
        ),
    )


# Weight gradients are output-stationary TN contractions.  BM/BN128 with BK32
# is the measured throughput winner once an expert can own multiple rows;
# decode prefers BM/BN64 because its output tile count exposes more parallelism
# without increasing per-CTA work.  Exact-frequency buffer resources suppress
# sorter-padding traffic in both profiles.
_GROUPED_DW1_BLOCK_K = 32
_GROUPED_DW1_MAX_EXPERT_ROWS = 4096
# The custom inactive-only fill wins once at least one eighth of the production
# expert set is live.  Below that point nearly the whole 9.9-GiB pair is still
# written and torch's dense memset is faster.  Legacy paths reuse their host
# frequency readback; hostless paths select from the device queue count.
_INACTIVE_WEIGHT_GRAD_ZERO_ACTIVE_RATIO = 8
# Each Qwen3 E16 expert owns roughly 9 MiB of BF16 dW1/dW2 output.  A single
# 1024-thread CTA cannot expose enough independent stores when only a few
# experts are active, so partition each expert slab across one gfx950-sized
# grid wave.  The retained expert-major predicate keeps this tuning isolated
# from E896 and every generic hostless path.
_E16_INACTIVE_WEIGHT_GRAD_ZERO_BLOCKS_PER_EXPERT = 16

# Device-sized row kernels use a host-known allocation bound only to size the
# launch.  Each workgroup then walks the sorter-produced ``num_valid_ids[0]``
# extent in a grid-stride loop.  Four workgroups per gfx950 CU is enough to
# cover latency without launching tens of thousands of idle CTAs for sparse
# hot-expert distributions.
_HOSTLESS_ROW_GRID_CAP = 1024
# Qwen3's non-SMALL expert-major E16 families have substantially more
# independent row work than the E896 path.  A second gfx950 wave is only an
# occupancy cap: every launch first takes the minimum with an actual-R work
# bound, and every row kernel reads its device-produced live extent.  SMALL
# keeps the measured one-wave cap; a rank-shared policy makes this finite
# tuning choice consistent across EP ranks.
_E16_EXACT_ROW_GRID_CAP = 2048


def _inactive_weight_grad_zero_blocks_per_expert(
    *,
    use_e16_deduplicated_metadata: bool,
    num_experts: int,
) -> int:
    """Select the audited E16 expert-slab store partition."""

    if use_e16_deduplicated_metadata and num_experts == 16:
        return _E16_INACTIVE_WEIGHT_GRAD_ZERO_BLOCKS_PER_EXPERT
    return 1


def _hostless_row_grid_cap(
    *,
    use_e16_expert_major: bool,
    route_policy: E16RoutePolicy = E16RoutePolicy.SMALL,
) -> int:
    """Return the row-kernel grid cap for the audited hostless dataflow."""

    if use_e16_expert_major and route_policy != E16RoutePolicy.SMALL:
        return _E16_EXACT_ROW_GRID_CAP
    return _HOSTLESS_ROW_GRID_CAP


def _grouped_dw1_tuning(
    max_expert_rows: int,
    hidden_size: int,
    intermediate_size: int,
    *,
    direct_rhs: bool = False,
    wide_direct_rhs: bool = True,
) -> tuple[int, int, int, int, int, int]:
    """Return ``(BM, BN, BK, K-pad, M-waves, N-waves)`` for grouped dW1."""

    preferred_tile = 64 if max_expert_rows <= 1 else 128
    output_m = 2 * intermediate_size
    block_m = preferred_tile if output_m % preferred_tile == 0 else 64
    if (
        direct_rhs
        and wide_direct_rhs
        and max_expert_rows > 1
        and hidden_size % 256 == 0
    ):
        block_n = 256
        n_waves = 4
    else:
        block_n = preferred_tile if hidden_size % preferred_tile == 0 else 64
        n_waves = 2
    return (block_m, block_n, _GROUPED_DW1_BLOCK_K, 0, 2, n_waves)


def _use_grouped_dw1(
    *,
    compute_dtype: str,
    activation: str,
    hidden_size: int,
    intermediate_size: int,
    tokens: int,
    routes: int,
    flat_routes: bool,
) -> bool:
    """Select gfx950's output-stationary grouped dW1 contraction."""

    max_expert_rows = routes if flat_routes else tokens
    return (
        compute_dtype == "bf16"
        and activation == "swiglu"
        and hidden_size % 64 == 0
        and (2 * intermediate_size) % 64 == 0
        and max_expert_rows <= _GROUPED_DW1_MAX_EXPERT_ROWS
    )


def _grouped_dx_tuning(active_experts: int, hidden_size: int) -> tuple[int, int]:
    """Return ``(BN, n_waves)`` for the observed expert distribution."""

    if active_experts >= _GROUPED_DX_DENSE_EXPERTS and hidden_size % 256 == 0:
        return (256, 4)
    if hidden_size % 128 == 0:
        return (128, 2)
    return (64, 2)


def _use_large_grouped_dx_descriptor_queue(
    *,
    tokens: int,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    topk: int,
    flat_routes: bool,
) -> bool:
    """Select the shape-static BM64 dX queue tuned on gfx950.

    Keep this deliberately exact until other long-token shapes have end-to-end
    measurements.  Every input is host-known, so enabling the queue never
    introduces a routing-statistics readback.
    """

    shape = (tokens, hidden_size, intermediate_size, num_experts, topk)
    return not flat_routes and shape in _LARGE_GROUPED_DX_SHAPES


def _use_grouped_dx_m_reuse_schedule(
    *,
    compute_dtype: str,
    activation: str,
    tokens: int,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    topk: int,
    flat_routes: bool,
    has_bias: bool,
    reuse_forward_preactivation: bool,
    use_large_grouped_dx: bool,
    use_hostless_grouped: bool,
    direct_grouped_dx_routes: bool,
) -> bool:
    """Select E896's device-guarded expert-major dX schedule pair."""

    return (
        compute_dtype == "bf16"
        and activation == "swiglu"
        and not flat_routes
        and not has_bias
        and reuse_forward_preactivation
        and use_large_grouped_dx
        and use_hostless_grouped
        and direct_grouped_dx_routes
        and (tokens, hidden_size, intermediate_size, num_experts, topk)
        == _GROUPED_DX_M_REUSE_SHAPE
    )


def _use_compact_w1_descriptor_queue(
    *,
    tokens: int,
    hidden_size: int,
    intermediate_size: int,
) -> bool:
    """Select the no-readback real-M-tile queue when its tuning is legal."""

    return (
        tokens >= _COMPACT_W1_MIN_TOKENS
        and hidden_size % (_COMPACT_W1_K_WAVE * _COMPACT_W1_BK) == 0
        and intermediate_size % _COMPACT_W1_BN == 0
    )


def _use_standalone_e16_tn_metadata_direct(
    *,
    e16_flat_grouped: bool,
    reuse_forward_preactivation: bool,
    use_hostless_grouped: bool,
    use_grouped_dw1: bool,
    use_grouped_dw2: bool,
    sort_unit: int,
    routes: int,
) -> bool:
    """Use sorter blocks directly while every expert owns at most one.

    ``routes <= sort_unit`` is a distribution-independent proof of the
    one-block-per-expert contract required by grouped TN's metadata ABI.  This
    remains valid at the inclusive boundary even though the independent W1
    recompute policy starts using its compact descriptor queue there.  The
    result is one finite short-route family, never a specialization on exact
    ``routes``.
    """

    return (
        e16_flat_grouped
        and not reuse_forward_preactivation
        and use_hostless_grouped
        and (use_grouped_dw1 or use_grouped_dw2)
        and sort_unit == _BACKWARD_SORT_UNIT
        and 0 < routes <= sort_unit
    )


def _grouped_w1_tuning(
    *,
    tokens: int,
    hidden_size: int,
    intermediate_size: int,
) -> tuple[int, int, int, int, bool]:
    """Return ``(BM, BN, BK, k_wave, compact_grid)`` for grouped W1."""

    compact_grid = _use_compact_w1_descriptor_queue(
        tokens=tokens,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
    )
    if compact_grid:
        return (
            _COMPACT_W1_BM,
            _COMPACT_W1_BN,
            _COMPACT_W1_BK,
            _COMPACT_W1_K_WAVE,
            True,
        )
    return (
        _GROUPED_W1_BM,
        _GROUPED_W1_BN,
        _GROUPED_W1_BK,
        _GROUPED_W1_K_WAVE,
        False,
    )


def _use_grouped_w1_recompute(
    *,
    compute_dtype: str,
    activation: str,
    hidden_size: int,
    intermediate_size: int,
    tokens: int,
    routes: int,
    flat_routes: bool,
) -> bool:
    """Return whether the short-M grouped W1 specialization is applicable."""

    max_expert_rows = routes if flat_routes else tokens
    bm, bn, bk, k_wave, _ = _grouped_w1_tuning(
        tokens=tokens,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
    )
    return (
        compute_dtype == "bf16"
        and activation == "swiglu"
        and hidden_size % (k_wave * bk) == 0
        and intermediate_size % bn == 0
        and _BACKWARD_SORT_UNIT % bm == 0
        and max_expert_rows <= _GROUPED_W1_MAX_EXPERT_ROWS
    )


def _use_grouped_dx(
    *,
    compute_dtype: str,
    activation: str,
    hidden_size: int,
    intermediate_size: int,
    tokens: int,
    num_experts: int,
    topk: int,
    flat_routes: bool,
    compact_w1: bool,
) -> bool:
    """Select the device-scheduled BF16 SwiGLU dX contraction.

    Short compact mode deliberately reuses the already-built W1 descriptor
    queue.  Fixed-K T1 is queue-free because every active expert owns one real
    row.  The tuned T4096 production shape uses an independent BM64 queue.
    """

    return (
        compute_dtype == "bf16"
        and activation == "swiglu"
        and (2 * intermediate_size) % _GROUPED_DX_BK == 0
        and hidden_size % 64 == 0
        and (
            compact_w1
            or (tokens == 1 and not flat_routes)
            or _use_large_grouped_dx_descriptor_queue(
                tokens=tokens,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                num_experts=num_experts,
                topk=topk,
                flat_routes=flat_routes,
            )
        )
    )


def _use_direct_grouped_dx_routes(*, use_grouped_dx: bool, flat_routes: bool) -> bool:
    """Fold fixed-K grouped dX's sorted-row permutation into its epilogue."""

    return use_grouped_dx and not flat_routes


def _use_hostless_grouped_backward(
    *,
    flat_routes: bool,
    has_bias: bool,
    tokens: int,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    topk: int,
    reuse_forward_preactivation: bool,
    use_large_grouped_dx: bool,
    use_grouped_w1: bool,
    use_grouped_w2: bool,
    use_grouped_dw2: bool,
    use_grouped_da: bool,
    use_grouped_dw1: bool,
    use_grouped_dx: bool,
    e16_fixed_state_grouped: bool = False,
    e16_flat_grouped: bool = False,
) -> bool:
    """Select a fully device-dispatched fixed-K backward.

    Every matrix contraction must already have a grouped implementation.  The
    remaining row kernels can then consume the sorter extent directly, so no
    host-side expert-frequency reconstruction is necessary.  The short path
    recomputes W1/W2 and therefore requires both grouped projections.  The
    strict long-shape path instead requires retained W1 state and the fused
    dA/dscore dataflow, so neither projection is part of its backward graph.
    Keep that rollout limited to the audited default E896 adapter bucket.
    """

    short_grouped = tokens <= 128 and use_grouped_w1 and use_grouped_w2
    shape = (tokens, hidden_size, intermediate_size, num_experts, topk)
    retained_large_grouped = (
        reuse_forward_preactivation
        and use_large_grouped_dx
        and (shape in _HOSTLESS_LARGE_STATE_SHAPES or e16_fixed_state_grouped)
    )
    fully_grouped = (
        not flat_routes
        and not has_bias
        and use_grouped_dw2
        and use_grouped_da
        and use_grouped_dw1
        and use_grouped_dx
        and (short_grouped or retained_large_grouped)
    )
    flat_fully_grouped = (
        e16_flat_grouped
        and not has_bias
        and use_grouped_w1
        and use_grouped_w2
        and use_grouped_dw2
        and use_grouped_da
        and use_grouped_dw1
        and use_grouped_dx
    )
    return fully_grouped or flat_fully_grouped


def _use_sorter_native_backward_metadata(
    *,
    flat_routes: bool,
    has_bias: bool,
    tokens: int,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    topk: int,
    reuse_forward_preactivation: bool,
    use_hostless_grouped: bool,
    use_large_grouped_dx: bool,
    use_compact_w1: bool,
    sort_unit: int,
) -> bool:
    """Select sorter-native metadata only for the audited E896 contract."""

    return (
        not flat_routes
        and not has_bias
        and reuse_forward_preactivation
        and use_hostless_grouped
        and use_large_grouped_dx
        and not use_compact_w1
        and sort_unit == _LARGE_GROUPED_DX_BM == 64
        and (tokens, hidden_size, intermediate_size, num_experts, topk)
        == (4096, 3584, 512, 896, 16)
    )


def _use_fused_da_dscore(
    *,
    reuse_forward_preactivation: bool,
    use_hostless_grouped: bool,
    use_compact_w1: bool,
    use_large_grouped_dx: bool,
    flat_routes: bool,
    has_bias: bool,
    compute_dtype: str,
    activation: str,
    e16_flat_grouped: bool = False,
) -> bool:
    """Select the first gfx950 unscaled-dA/route-score fusion rollout.

    The fused dataflow uses retained forward state plus an exact-row descriptor
    schedule.  Short fixed-K shapes reuse the BM16 compact W1/dX queue, E16
    flat routes use their exact BM128 queue at every non-empty R, and the
    production T4096 shape reuses its independently tuned BM64 dX queue.
    ``dy`` initially carries the exact gathered A16 output gradient, grouped
    dA computes ``q = dout @ W2``, and one row kernel subsequently forms dA's
    score scaling, dscore, and the scaled dW2 input.  Bias, legacy host
    dispatch, and non-SwiGLU/dtype contracts retain the projection-recompute
    implementation.
    """

    return (
        reuse_forward_preactivation
        and (
            (use_hostless_grouped and use_compact_w1)
            or use_large_grouped_dx
            or e16_flat_grouped
        )
        and (not flat_routes or e16_flat_grouped)
        and not has_bias
        and compute_dtype == "bf16"
        and activation == "swiglu"
    )


def _use_direct_grouped_dw1_rhs(
    *,
    reuse_forward_preactivation: bool,
    use_fused_forward_state_prepare: bool,
    use_grouped_dw1: bool,
    flat_routes: bool,
    has_bias: bool,
    compute_dtype: str,
    activation: str,
    tokens: int,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    topk: int,
    flat_identity_routes: bool = False,
) -> bool:
    """Select token-major RHS gather for retained-state grouped dW1.

    Select either the exact fixed-K E896 contract or retained E16 flat routes
    whose caller has declared identity token indices.  In both cases fused
    state preparation has no other consumer for the sorter-order hidden-state
    copy.  Standalone, non-identity ragged, bias, and legacy contracts continue
    to materialize ``x_sorted``.
    """

    return (
        reuse_forward_preactivation
        and use_fused_forward_state_prepare
        and use_grouped_dw1
        and not has_bias
        and compute_dtype == "bf16"
        and activation == "swiglu"
        and (
            (
                not flat_routes
                and (tokens, hidden_size, intermediate_size, num_experts, topk)
                == _DIRECT_GROUPED_DW1_RHS_SHAPE
            )
            or (
                flat_routes
                and flat_identity_routes
                and (hidden_size, intermediate_size, num_experts, topk)
                == (2048, 768, 16, 1)
            )
        )
    )


@functools.lru_cache(maxsize=64)
def _compile_grouped_dx(
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    block_m: int,
    block_n: int,
    n_waves: int,
    compact_grid: bool,
    device_index: int,
    min_active_experts: int = 0,
    max_active_experts: int | None = None,
    store_route_slots: bool = False,
    store_route_ids: bool = False,
    top_k: int = 1,
    expert_m_reuse: bool = False,
    expert_m_reuse_threshold: int | None = None,
    exact_tile_queue: bool = False,
):
    """Build the gfx950 grouped ``dZ @ W1`` specialization."""

    return compile_sonic_grouped_a16w16_nn(
        contraction_size=2 * intermediate_size,
        output_size=hidden_size,
        num_experts=num_experts,
        block_m=block_m,
        block_n=block_n,
        block_k=_GROUPED_DX_BK,
        stages=_GROUPED_DX_STAGES,
        n_waves=n_waves,
        sorted_block_m=_BACKWARD_SORT_UNIT,
        compact_grid=compact_grid,
        device_index=device_index,
        min_active_experts=min_active_experts,
        max_active_experts=max_active_experts,
        store_route_slots=store_route_slots,
        store_route_ids=store_route_ids,
        top_k=top_k,
        expert_m_reuse=expert_m_reuse,
        expert_m_reuse_threshold=expert_m_reuse_threshold,
        exact_tile_queue=exact_tile_queue,
    )


@functools.lru_cache(maxsize=16)
def _compile_e16_exact_grouped_da(
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    device_index: int,
    gather_a_rows: bool = False,
):
    """Build Qwen3 E16's exact-queue ``dY @ W2`` contraction."""

    return compile_sonic_grouped_a16w16_nn(
        contraction_size=hidden_size,
        output_size=intermediate_size,
        num_experts=num_experts,
        block_m=_E16_EXACT_BM,
        block_n=_E16_EXACT_DA_BN,
        block_k=_E16_EXACT_BK,
        stages=2,
        n_waves=_E16_EXACT_N_WAVES,
        sorted_block_m=_BACKWARD_SORT_UNIT,
        compact_grid=True,
        device_index=device_index,
        exact_tile_queue=True,
        gather_a_rows=gather_a_rows,
    )


@functools.lru_cache(maxsize=64)
def _compile_grouped_w1_recompute(
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    topk: int,
    has_bias: bool,
    interleaved_w1: bool,
    compact_grid: bool,
    device_index: int,
):
    """Build grouped raw-W1 preactivation recompute for BF16 SwiGLU.

    ``device_index`` intentionally participates in the cache key because a
    loaded compiled function is tied to its ROCm device.
    """

    del device_index
    if compact_grid:
        bm, bn, bk, k_wave = (
            _COMPACT_W1_BM,
            _COMPACT_W1_BN,
            _COMPACT_W1_BK,
            _COMPACT_W1_K_WAVE,
        )
    else:
        bm, bn, bk, k_wave = (
            _GROUPED_W1_BM,
            _GROUPED_W1_BN,
            _GROUPED_W1_BK,
            _GROUPED_W1_K_WAVE,
        )
    return compile_gemm1_a16w4_port(
        BM=bm,
        SORTED_BM=_BACKWARD_SORT_UNIT,
        D_HIDDEN=hidden_size,
        D_INTER=intermediate_size,
        NE=num_experts,
        TOPK=topk,
        TILE_N=bn,
        TILE_K=bk,
        act="swiglu",
        b_cache_mod=0,
        w_dtype="bf16",
        a_dtype="bf16",
        w_layout="interleaved" if interleaved_w1 else "standard",
        k_wave=k_wave,
        round_preact_bf16=False,
        has_bias=has_bias,
        logical_dense_weight=True,
        store_preactivation=True,
        expert_grid=not compact_grid,
        compact_grid=compact_grid,
    )


def _use_grouped_w2_recompute(
    *,
    compute_dtype: str,
    activation: str,
    hidden_size: int,
    intermediate_size: int,
    tokens: int,
    routes: int,
    flat_routes: bool,
) -> bool:
    """Return whether the short-M grouped W2 projection is applicable."""

    max_expert_rows = routes if flat_routes else tokens
    return (
        compute_dtype == "bf16"
        and activation == "swiglu"
        and hidden_size % _GROUPED_W2_BN == 0
        and intermediate_size % _GROUPED_W2_BK == 0
        and max_expert_rows <= _GROUPED_W2_MAX_EXPERT_ROWS
    )


def _use_grouped_dw2(
    *,
    compute_dtype: str,
    hidden_size: int,
    intermediate_size: int,
) -> bool:
    """Return whether the gfx950 grouped BF16 dW2 contraction is legal."""

    return (
        compute_dtype == "bf16"
        and hidden_size % 64 == 0
        and intermediate_size % 64 == 0
    )


def _use_grouped_da(
    *,
    compute_dtype: str,
    activation: str,
    hidden_size: int,
    intermediate_size: int,
    tokens: int,
    routes: int,
    flat_routes: bool,
) -> bool:
    """Return whether the short-M grouped dA specialization is applicable."""

    max_expert_rows = routes if flat_routes else tokens
    return (
        compute_dtype == "bf16"
        and activation == "swiglu"
        and hidden_size % 64 == 0
        and intermediate_size % _GROUPED_DA_BN == 0
        and max_expert_rows <= _GROUPED_DA_MAX_EXPERT_ROWS
    )


@functools.lru_cache(maxsize=64)
def _compile_grouped_da(
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    block_m: int,
    block_n: int,
    block_k: int,
    m_waves: int,
    n_waves: int,
    device_index: int,
    queue_direct: bool = False,
    min_active_experts: int = 0,
    max_active_experts: int | None = None,
    stages: int = _GROUPED_DA_STAGES,
    persistent: bool = False,
):
    """Build the grouped raw-W2 dA contraction for BF16 SwiGLU."""

    del device_index
    return compile_grouped_da_gfx950(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        sorted_block_size=_BACKWARD_SORT_UNIT,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        stages=stages,
        m_waves=m_waves,
        n_waves=n_waves,
        queue_direct=queue_direct,
        persistent=persistent,
        min_active_experts=min_active_experts,
        max_active_experts=max_active_experts,
    )


@functools.lru_cache(maxsize=64)
def _compile_grouped_w2_recompute(
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    topk: int,
    has_bias: bool,
    device_index: int,
):
    """Build grouped raw-W2 sorted projection recompute for BF16 SwiGLU."""

    del device_index
    return compile_gemm2_a16w4_port(
        BM=_GROUPED_W2_BM,
        SORTED_BM=_BACKWARD_SORT_UNIT,
        NE=num_experts,
        N_OUT=hidden_size,
        D_INTER=intermediate_size,
        TILE_N=_GROUPED_W2_BN,
        TILE_K=_GROUPED_W2_BK,
        xcd_swizzle=0,
        b_cache_mod=0,
        w_dtype="bf16",
        a_dtype="bf16",
        persist=False,
        has_bias=has_bias,
        round_projection_bf16=False,
        output_mode="atomic",
        TOPK=topk,
        logical_dense_weight=True,
        store_sorted_projection=True,
        expert_grid=True,
    )


@fx.struct
class _ScoreBackwardSharedStorage:
    reduction: fx.Array[fx.Float32, _RED_SLOTS, 16]


@fx.struct
class _FlatDxSegmentSharedStorage:
    bounds: fx.Array[fx.Int32, 2, 16]


def _gelu_tanh_derivative_f32(x):
    """Derivative of the tanh-approximate GELU used by stage 1."""

    one = fx.Float32(1.0)
    half = fx.Float32(0.5)
    sqrt_2_over_pi = fx.Float32(0.7978845608028654)
    coeff = fx.Float32(0.044715)
    inner = sqrt_2_over_pi * (x + coeff * x * x * x)
    tanh_inner = _tanh_f32(inner)
    dinner = sqrt_2_over_pi * (one + fx.Float32(3.0) * coeff * x * x)
    return (
        half * (one + tanh_inner) + half * x * (one - tanh_inner * tanh_inner) * dinner
    )


def _relu_derivative_f32(x):
    zero = fx.Float32(0.0)
    one = fx.Float32(1.0)
    positive = arith.cmpf(arith.CmpFPredicate.OGT, _raw(x), _raw(zero))
    return fx.Float32(arith.select(positive, _raw(one), _raw(zero)))


def _activation_f32(gate, up, activation: str):
    """Apply a compile-time selected public SonicMoE activation."""

    if activation == "swiglu":
        return gate * _sigmoid_f32(gate) * up
    if activation == "geglu":
        return _gelu_tanh_f32(gate) * up
    if activation == "reglu":
        return _relu_f32(gate) * up
    if activation == "gelu_tanh_approx":
        return _gelu_tanh_f32(gate)
    if activation == "relu":
        return _relu_f32(gate)
    if activation == "silu":
        return gate * _sigmoid_f32(gate)
    if activation == "relu_sq":
        relu = _relu_f32(gate)
        return relu * relu
    raise AssertionError(f"unexpected activation {activation!r}")


def _activation_backward_f32(gate, up, da, activation: str):
    """Apply a compile-time selected activation Jacobian-vector product."""

    one = fx.Float32(1.0)
    if activation == "swiglu":
        sigmoid = _sigmoid_f32(gate)
        activated_gate = gate * sigmoid
        derivative = sigmoid * (one + gate * (one - sigmoid))
        return da * up * derivative, da * activated_gate
    if activation == "geglu":
        return (
            da * up * _gelu_tanh_derivative_f32(gate),
            da * _gelu_tanh_f32(gate),
        )
    if activation == "reglu":
        return da * up * _relu_derivative_f32(gate), da * _relu_f32(gate)
    if activation == "gelu_tanh_approx":
        return da * _gelu_tanh_derivative_f32(gate), fx.Float32(0.0)
    if activation == "relu":
        return da * _relu_derivative_f32(gate), fx.Float32(0.0)
    if activation == "silu":
        sigmoid = _sigmoid_f32(gate)
        return da * sigmoid * (one + gate * (one - sigmoid)), fx.Float32(0.0)
    if activation == "relu_sq":
        return da * fx.Float32(2.0) * _relu_f32(gate), fx.Float32(0.0)
    raise AssertionError(f"unexpected activation {activation!r}")


def _max_padded_routes(
    tokens: int,
    num_experts: int,
    topk: int,
    sort_unit: int,
) -> tuple[int, int]:
    """Return the dense sorter's safe ``(rows, blocks)`` allocation bound."""

    routes = tokens * topk
    active_experts = min(num_experts, routes)
    padding_bound = (routes + active_experts * (sort_unit - 1)) // sort_unit
    per_expert_bound = active_experts * ((tokens + sort_unit - 1) // sort_unit)
    blocks = min(padding_bound, per_expert_bound)
    return blocks * sort_unit, blocks


def _max_padded_flat_routes(
    routes: int,
    num_experts: int,
    sort_unit: int,
) -> tuple[int, int]:
    """Return the ragged sorter's safe ``(rows, blocks)`` allocation bound."""

    active_experts = min(num_experts, routes)
    blocks = (routes + active_experts * (sort_unit - 1)) // sort_unit
    return blocks * sort_unit, blocks


def _materialize_expert_segments(
    expert_frequency: torch.Tensor,
    sort_unit: int,
) -> tuple[list[int], list[tuple[int, int, int]], int, int]:
    """Synchronously reconstruct padded expert slices for legacy fallbacks."""

    frequencies = expert_frequency.cpu().tolist()
    segments: list[tuple[int, int, int]] = []
    offset = 0
    for expert, count in enumerate(frequencies):
        if count:
            padded = ((int(count) + sort_unit - 1) // sort_unit) * sort_unit
            segments.append((expert, offset, padded))
            offset += padded
    return frequencies, segments, offset, max(int(count) for count in frequencies)


@functools.lru_cache(maxsize=128)
def _compile_expert_histogram(num_experts: int, device_index: int):
    """Compile fixed-K expert histogram kernels for one device specialization."""

    del device_index

    @flyc.kernel(known_block_size=[_BLOCK_THREADS, 1, 1])
    def clear_kernel(expert_frequency: fx.Tensor):
        index = gpu.block_idx.x * fx.Int32(_BLOCK_THREADS) + gpu.thread_idx.x
        if index < fx.Int32(num_experts):
            rsrc = buffer_ops.create_buffer_resource(expert_frequency, max_size=True)
            buffer_ops.buffer_store(fx.Int32(0), rsrc, index)

    @flyc.kernel(known_block_size=[_BLOCK_THREADS, 1, 1])
    def histogram_kernel(
        topk_ids: fx.Tensor,
        expert_frequency: fx.Tensor,
        i32_routes: fx.Int32,
    ):
        index = gpu.block_idx.x * fx.Int32(_BLOCK_THREADS) + gpu.thread_idx.x
        if index < i32_routes:
            ids_rsrc = buffer_ops.create_buffer_resource(topk_ids, max_size=True)
            expert = fx.Int32(
                buffer_ops.buffer_load(ids_rsrc, index, vec_width=1, dtype=T.i32)
            )
            # Public validation deliberately avoids a device synchronization;
            # keep this guard so malformed ids cannot write out of bounds.
            if (expert >= fx.Int32(0)) & (expert < fx.Int32(num_experts)):
                atomic_add(expert_frequency, expert, fx.Int32(1), dtype_bytes=4)

    @flyc.jit
    def launch(
        topk_ids: fx.Tensor,
        expert_frequency: fx.Tensor,
        i32_routes: fx.Int32,
        i32_route_grid: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        clear_kernel(expert_frequency).launch(
            grid=((num_experts + _BLOCK_THREADS - 1) // _BLOCK_THREADS, 1, 1),
            block=(_BLOCK_THREADS, 1, 1),
            stream=stream,
        )
        histogram_kernel(topk_ids, expert_frequency, i32_routes).launch(
            grid=(i32_route_grid, 1, 1),
            block=(_BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    return launch


@functools.lru_cache(maxsize=128)
def _compile_gather(
    hidden_size: int,
    compute_dtype: str,
    device_index: int,
    device_padded_rows: bool = False,
):
    """Compile sorted-row gathers for hidden states and output gradients."""

    del device_index
    elem_dtype = fx.Float16 if compute_dtype == "fp16" else fx.BFloat16
    vector_width = 4
    vectors_per_row = hidden_size // vector_width

    @flyc.kernel(known_block_size=[_BLOCK_THREADS, 1, 1])
    def gather_kernel(
        hidden_states: fx.Tensor,
        grad_output: fx.Tensor,
        sorted_token_ids: fx.Tensor,
        x_sorted: fx.Tensor,
        dout_sorted: fx.Tensor,
        num_valid_ids: fx.Tensor,
        i32_tokens: fx.Int32,
        i32_padded_rows: fx.Int32,
    ):
        index = gpu.block_idx.x * fx.Int32(_BLOCK_THREADS) + gpu.thread_idx.x
        stride = gpu.grid_dim.x * fx.Int32(_BLOCK_THREADS)
        ids_rsrc = buffer_ops.create_buffer_resource(sorted_token_ids, max_size=True)
        x_rsrc = buffer_ops.create_buffer_resource(hidden_states, max_size=True)
        dout_rsrc = buffer_ops.create_buffer_resource(grad_output, max_size=True)
        x_sorted_rsrc = buffer_ops.create_buffer_resource(x_sorted, max_size=True)
        dout_sorted_rsrc = buffer_ops.create_buffer_resource(dout_sorted, max_size=True)

        def gather_vector(index):
            row = index // fx.Int32(vectors_per_row)
            column = (index % fx.Int32(vectors_per_row)) * fx.Int32(vector_width)
            packed = fx.Int32(
                buffer_ops.buffer_load(ids_rsrc, row, vec_width=1, dtype=T.i32)
            )
            token = packed & fx.Int32(_TOKEN_MASK)
            valid = (token >= fx.Int32(0)) & (token < i32_tokens)
            safe_token = valid.select(token, fx.Int32(0))
            source = safe_token * fx.Int32(hidden_size) + column
            destination = row * fx.Int32(hidden_size) + column
            x_value = buffer_ops.buffer_load(
                x_rsrc,
                source,
                vec_width=vector_width,
                dtype=elem_dtype,
            )
            dout_value = buffer_ops.buffer_load(
                dout_rsrc,
                source,
                vec_width=vector_width,
                dtype=elem_dtype,
            )
            zero = fx.Vector.filled(vector_width, 0.0, elem_dtype)
            buffer_ops.buffer_store(
                valid.select(fx.Vector(x_value), zero), x_sorted_rsrc, destination
            )
            buffer_ops.buffer_store(
                valid.select(fx.Vector(dout_value), zero), dout_sorted_rsrc, destination
            )

        padded_rows = i32_padded_rows
        if const_expr(device_padded_rows):
            valid_rsrc = buffer_ops.create_buffer_resource(num_valid_ids, max_size=True)
            padded_rows = fx.Int32(
                buffer_ops.buffer_load(
                    valid_rsrc, fx.Int32(0), vec_width=1, dtype=T.i32
                )
            )
        total = padded_rows * fx.Int32(vectors_per_row)
        if const_expr(device_padded_rows):
            for vector_index in range(index, total, stride):
                gather_vector(fx.Int32(vector_index))
        else:
            if index < total:
                gather_vector(index)

    @flyc.jit
    def launch(
        hidden_states: fx.Tensor,
        grad_output: fx.Tensor,
        sorted_token_ids: fx.Tensor,
        x_sorted: fx.Tensor,
        dout_sorted: fx.Tensor,
        num_valid_ids: fx.Tensor,
        i32_tokens: fx.Int32,
        i32_padded_rows: fx.Int32,
        i32_grid: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        gather_kernel(
            hidden_states,
            grad_output,
            sorted_token_ids,
            x_sorted,
            dout_sorted,
            num_valid_ids,
            i32_tokens,
            i32_padded_rows,
        ).launch(
            grid=(i32_grid, 1, 1),
            block=(_BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    return launch


@functools.lru_cache(maxsize=128)
def _compile_activation_prepare(
    hidden_size: int,
    intermediate_size: int,
    activation_name: str,
    compute_dtype: str,
    interleaved_w1: bool,
    device_index: int,
    device_padded_rows: bool = False,
):
    """Compile activation recomputation and routed-dout scaling."""

    del device_index
    elem_dtype = fx.Float16 if compute_dtype == "fp16" else fx.BFloat16
    is_glu = activation_name in _GLU_ACTIVATIONS
    projection_size = intermediate_size * (2 if is_glu else 1)
    projection_column_stride = 2 if is_glu and interleaved_w1 else 1
    up_column_offset = (
        1 if is_glu and interleaved_w1 else intermediate_size if is_glu else 0
    )

    @flyc.kernel(known_block_size=[_BLOCK_THREADS, 1, 1])
    def activation_prepare_kernel(
        preactivation: fx.Tensor,
        activation: fx.Tensor,
        dout_sorted: fx.Tensor,
        dy: fx.Tensor,
        sorted_weights: fx.Tensor,
        sorted_token_ids: fx.Tensor,
        num_valid_ids: fx.Tensor,
        i32_tokens: fx.Int32,
        i32_padded_rows: fx.Int32,
    ):
        tid = gpu.thread_idx.x
        weights_rsrc = buffer_ops.create_buffer_resource(sorted_weights, max_size=True)
        preact_rsrc = buffer_ops.create_buffer_resource(preactivation, max_size=True)
        activation_rsrc = buffer_ops.create_buffer_resource(activation, max_size=True)
        dout_rsrc = buffer_ops.create_buffer_resource(dout_sorted, max_size=True)
        dy_rsrc = buffer_ops.create_buffer_resource(dy, max_size=True)
        ids_rsrc = buffer_ops.create_buffer_resource(sorted_token_ids, max_size=True)

        def prepare_row(row):
            route_weight = fx.Float32(
                buffer_ops.buffer_load(weights_rsrc, row, vec_width=1, dtype=T.f32)
            )

            for base in range_constexpr(0, intermediate_size, _BLOCK_THREADS):
                column = tid + fx.Int32(base)
                if column < fx.Int32(intermediate_size):
                    gate_offset = row * fx.Int32(projection_size) + column * fx.Int32(
                        projection_column_stride
                    )
                    act_offset = row * fx.Int32(intermediate_size) + column
                    gate = buffer_ops.buffer_load(
                        preact_rsrc, gate_offset, vec_width=1, dtype=elem_dtype
                    ).extf(T.f32)
                    up_offset = gate_offset + fx.Int32(up_column_offset)
                    up = buffer_ops.buffer_load(
                        preact_rsrc,
                        up_offset,
                        vec_width=1,
                        dtype=elem_dtype,
                    ).extf(T.f32)
                    activation_f32 = _activation_f32(gate, up, activation_name)
                    buffer_ops.buffer_store(
                        fx.Float32(activation_f32).to(elem_dtype),
                        activation_rsrc,
                        act_offset,
                    )

            # Materialize dy in A16 before its two GEMMs.  This preserves the
            # multiply-before-GEMM dependency while making the standalone
            # FlyDSL backward's A16 input boundary explicit.
            for base in range_constexpr(0, hidden_size, _BLOCK_THREADS):
                column = tid + fx.Int32(base)
                if column < fx.Int32(hidden_size):
                    offset = row * fx.Int32(hidden_size) + column
                    dout_value = buffer_ops.buffer_load(
                        dout_rsrc, offset, vec_width=1, dtype=elem_dtype
                    ).extf(T.f32)
                    buffer_ops.buffer_store(
                        fx.Float32(dout_value * route_weight).to(elem_dtype),
                        dy_rsrc,
                        offset,
                    )

        padded_rows = i32_padded_rows
        if const_expr(device_padded_rows):
            valid_rsrc = buffer_ops.create_buffer_resource(num_valid_ids, max_size=True)
            padded_rows = fx.Int32(
                buffer_ops.buffer_load(
                    valid_rsrc, fx.Int32(0), vec_width=1, dtype=T.i32
                )
            )

        if const_expr(device_padded_rows):
            for row_value in range(gpu.block_idx.x, padded_rows, gpu.grid_dim.x):
                row = fx.Int32(row_value)
                packed = fx.Int32(
                    buffer_ops.buffer_load(ids_rsrc, row, vec_width=1, dtype=T.i32)
                )
                token = packed & fx.Int32(_TOKEN_MASK)
                if token < i32_tokens:
                    prepare_row(row)
        else:
            prepare_row(gpu.block_idx.x)

    @flyc.jit
    def launch(
        preactivation: fx.Tensor,
        activation: fx.Tensor,
        dout_sorted: fx.Tensor,
        dy: fx.Tensor,
        sorted_weights: fx.Tensor,
        sorted_token_ids: fx.Tensor,
        num_valid_ids: fx.Tensor,
        i32_tokens: fx.Int32,
        i32_padded_rows: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        activation_prepare_kernel(
            preactivation,
            activation,
            dout_sorted,
            dy,
            sorted_weights,
            sorted_token_ids,
            num_valid_ids,
            i32_tokens,
            i32_padded_rows,
        ).launch(
            grid=(i32_padded_rows, 1, 1),
            block=(_BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    return launch


@functools.lru_cache(maxsize=128)
def _compile_activation_prepare_from_forward_state(
    hidden_size: int,
    intermediate_size: int,
    topk: int,
    interleaved_w1: bool,
    device_index: int,
    device_padded_rows: bool = False,
):
    """Gather route-order BF16 preactivation and prepare backward rows.

    The state is compact ``[T, K, 2I]`` route-order storage, while every
    downstream backward contraction consumes sorter order.  Folding that
    permutation into activation preparation avoids both W1 recomputation and
    a standalone gather launch.
    """

    del device_index
    elem_dtype = fx.BFloat16
    projection_size = 2 * intermediate_size
    projection_column_stride = 2 if interleaved_w1 else 1
    up_column_offset = 1 if interleaved_w1 else intermediate_size

    @flyc.kernel(known_block_size=[_BLOCK_THREADS, 1, 1])
    def activation_prepare_from_state_kernel(
        route_preactivation: fx.Tensor,
        preactivation: fx.Tensor,
        activation: fx.Tensor,
        dout_sorted: fx.Tensor,
        dy: fx.Tensor,
        sorted_weights: fx.Tensor,
        sorted_token_ids: fx.Tensor,
        num_valid_ids: fx.Tensor,
        i32_tokens: fx.Int32,
        i32_padded_rows: fx.Int32,
    ):
        tid = gpu.thread_idx.x
        route_preact_rsrc = buffer_ops.create_buffer_resource(
            route_preactivation, max_size=True
        )
        preact_rsrc = buffer_ops.create_buffer_resource(preactivation, max_size=True)
        activation_rsrc = buffer_ops.create_buffer_resource(activation, max_size=True)
        dout_rsrc = buffer_ops.create_buffer_resource(dout_sorted, max_size=True)
        dy_rsrc = buffer_ops.create_buffer_resource(dy, max_size=True)
        weights_rsrc = buffer_ops.create_buffer_resource(sorted_weights, max_size=True)
        ids_rsrc = buffer_ops.create_buffer_resource(sorted_token_ids, max_size=True)
        zero = fx.Float32(0.0).to(elem_dtype)

        def prepare_real_row(row, token, slot):
            route_row = token * fx.Int32(topk) + slot
            route_base = route_row * fx.Int32(projection_size)
            sorted_base = row * fx.Int32(projection_size)
            for base in range_constexpr(0, intermediate_size, _BLOCK_THREADS):
                column = tid + fx.Int32(base)
                if column < fx.Int32(intermediate_size):
                    relative_gate = column * fx.Int32(projection_column_stride)
                    route_gate_offset = route_base + relative_gate
                    sorted_gate_offset = sorted_base + relative_gate
                    gate = buffer_ops.buffer_load(
                        route_preact_rsrc,
                        route_gate_offset,
                        vec_width=1,
                        dtype=elem_dtype,
                    )
                    up = buffer_ops.buffer_load(
                        route_preact_rsrc,
                        route_gate_offset + fx.Int32(up_column_offset),
                        vec_width=1,
                        dtype=elem_dtype,
                    )
                    buffer_ops.buffer_store(gate, preact_rsrc, sorted_gate_offset)
                    buffer_ops.buffer_store(
                        up,
                        preact_rsrc,
                        sorted_gate_offset + fx.Int32(up_column_offset),
                    )
                    activation_f32 = _activation_f32(
                        gate.extf(T.f32), up.extf(T.f32), "swiglu"
                    )
                    buffer_ops.buffer_store(
                        fx.Float32(activation_f32).to(elem_dtype),
                        activation_rsrc,
                        row * fx.Int32(intermediate_size) + column,
                    )

            route_weight = fx.Float32(
                buffer_ops.buffer_load(weights_rsrc, row, vec_width=1, dtype=T.f32)
            )
            for base in range_constexpr(0, hidden_size, _BLOCK_THREADS):
                column = tid + fx.Int32(base)
                if column < fx.Int32(hidden_size):
                    offset = row * fx.Int32(hidden_size) + column
                    dout_value = buffer_ops.buffer_load(
                        dout_rsrc, offset, vec_width=1, dtype=elem_dtype
                    ).extf(T.f32)
                    buffer_ops.buffer_store(
                        fx.Float32(dout_value * route_weight).to(elem_dtype),
                        dy_rsrc,
                        offset,
                    )

        def clear_padding_row(row):
            for base in range_constexpr(0, intermediate_size, _BLOCK_THREADS):
                column = tid + fx.Int32(base)
                if column < fx.Int32(intermediate_size):
                    relative_gate = column * fx.Int32(projection_column_stride)
                    sorted_gate_offset = row * fx.Int32(projection_size) + relative_gate
                    buffer_ops.buffer_store(zero, preact_rsrc, sorted_gate_offset)
                    buffer_ops.buffer_store(
                        zero,
                        preact_rsrc,
                        sorted_gate_offset + fx.Int32(up_column_offset),
                    )
                    buffer_ops.buffer_store(
                        zero,
                        activation_rsrc,
                        row * fx.Int32(intermediate_size) + column,
                    )
            for base in range_constexpr(0, hidden_size, _BLOCK_THREADS):
                column = tid + fx.Int32(base)
                if column < fx.Int32(hidden_size):
                    buffer_ops.buffer_store(
                        zero,
                        dy_rsrc,
                        row * fx.Int32(hidden_size) + column,
                    )

        def decode_route(row):
            packed = fx.Int32(
                buffer_ops.buffer_load(ids_rsrc, row, vec_width=1, dtype=T.i32)
            )
            token = packed & fx.Int32(_TOKEN_MASK)
            slot = (packed >> fx.Int32(24)) & fx.Int32(0xFF)
            return token, slot

        padded_rows = i32_padded_rows
        if const_expr(device_padded_rows):
            valid_rsrc = buffer_ops.create_buffer_resource(num_valid_ids, max_size=True)
            padded_rows = fx.Int32(
                buffer_ops.buffer_load(
                    valid_rsrc, fx.Int32(0), vec_width=1, dtype=T.i32
                )
            )

        if const_expr(device_padded_rows):
            for row_value in range(gpu.block_idx.x, padded_rows, gpu.grid_dim.x):
                row = fx.Int32(row_value)
                token, slot = decode_route(row)
                if (token < i32_tokens) & (slot < fx.Int32(topk)):
                    prepare_real_row(row, token, slot)
        else:
            # Legacy contractions consume complete SORTED_BM-padded segments,
            # so their sentinel rows must explicitly materialize zero tails.
            # The hostless branch above skips sentinels throughout and gets
            # dX's short tile tail from its independently zeroed dZ workspace.
            row = gpu.block_idx.x
            token, slot = decode_route(row)
            if (token < i32_tokens) & (slot < fx.Int32(topk)):
                prepare_real_row(row, token, slot)
            else:
                clear_padding_row(row)

    @flyc.jit
    def launch(
        route_preactivation: fx.Tensor,
        preactivation: fx.Tensor,
        activation: fx.Tensor,
        dout_sorted: fx.Tensor,
        dy: fx.Tensor,
        sorted_weights: fx.Tensor,
        sorted_token_ids: fx.Tensor,
        num_valid_ids: fx.Tensor,
        i32_tokens: fx.Int32,
        i32_padded_rows: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        activation_prepare_from_state_kernel(
            route_preactivation,
            preactivation,
            activation,
            dout_sorted,
            dy,
            sorted_weights,
            sorted_token_ids,
            num_valid_ids,
            i32_tokens,
            i32_padded_rows,
        ).launch(
            grid=(i32_padded_rows, 1, 1),
            block=(_BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    return launch


@functools.lru_cache(maxsize=128)
def _compile_fused_forward_state_prepare(
    hidden_size: int,
    intermediate_size: int,
    topk: int,
    interleaved_w1: bool,
    device_index: int,
    schedule_block_m: int = _COMPACT_W1_BM,
    defer_dy_scaling: bool = False,
    store_x_sorted: bool = True,
    flat_routes: bool = False,
    exact_tile_queue: bool = False,
    defer_activation: bool = False,
):
    """Gather the exact live rows and prepare retained-state backward inputs.

    This specialization is restricted to fully grouped BF16 SwiGLU paths.  It
    avoids walking every 64-row sorter pad, never
    materializes a sorted copy of ``grad_output``, and combines the remaining
    hidden-state gather with activation and routed-gradient preparation.

    It consumes an existing counter-first BM16/BM64 descriptor queue.  The E16
    flat retained path can instead reuse its BM128 exact queue, whose records
    carry ``(expert, first_row, valid_rows)`` and therefore need no separate
    compact state schedule.
    """

    del device_index
    elem_dtype = fx.BFloat16
    projection_size = 2 * intermediate_size
    projection_column_stride = 2 if interleaved_w1 else 1
    up_column_offset = 1 if interleaved_w1 else intermediate_size
    vector_dy_copy = (
        defer_dy_scaling
        and not store_x_sorted
        and hidden_size % _DY_COPY_BF16_ELEMENTS == 0
    )
    if exact_tile_queue:
        if schedule_block_m != _E16_EXACT_BM:
            raise ValueError(
                f"exact state-prepare schedule_block_m must be {_E16_EXACT_BM}"
            )
    elif schedule_block_m not in (_COMPACT_W1_BM, _LARGE_GROUPED_DX_BM):
        raise ValueError("state-prepare schedule_block_m must be 16 or 64")

    @flyc.kernel(known_block_size=[_BLOCK_THREADS, 1, 1])
    def fused_prepare_kernel(
        hidden_states: fx.Tensor,
        grad_output: fx.Tensor,
        route_preactivation: fx.Tensor,
        x_sorted: fx.Tensor,
        activation: fx.Tensor,
        dy: fx.Tensor,
        sorted_weights: fx.Tensor,
        sorted_token_ids: fx.Tensor,
        sorted_route_ids: fx.Tensor,
        schedule: fx.Tensor,
        i32_tokens: fx.Int32,
        i32_routes: fx.Int32,
    ):
        tid = gpu.thread_idx.x
        dout_rsrc = buffer_ops.create_buffer_resource(grad_output, max_size=True)
        x_rsrc = dout_rsrc
        if const_expr(store_x_sorted):
            x_rsrc = buffer_ops.create_buffer_resource(hidden_states, max_size=True)
        route_preact_rsrc = buffer_ops.create_buffer_resource(
            route_preactivation, max_size=True
        )
        activation_rsrc = buffer_ops.create_buffer_resource(activation, max_size=True)
        dy_rsrc = buffer_ops.create_buffer_resource(dy, max_size=True)
        x_sorted_rsrc = dy_rsrc
        if const_expr(store_x_sorted):
            x_sorted_rsrc = buffer_ops.create_buffer_resource(x_sorted, max_size=True)
        weights_rsrc = buffer_ops.create_buffer_resource(sorted_weights, max_size=True)
        ids_rsrc = buffer_ops.create_buffer_resource(sorted_token_ids, max_size=True)
        route_ids_rsrc = ids_rsrc
        if const_expr(flat_routes):
            route_ids_rsrc = buffer_ops.create_buffer_resource(
                sorted_route_ids,
                max_size=True,
            )

        def prepare_real_row(row):
            packed = fx.Int32(
                buffer_ops.buffer_load(ids_rsrc, row, vec_width=1, dtype=T.i32)
            )
            token = packed & fx.Int32(_TOKEN_MASK)
            slot = (packed >> fx.Int32(24)) & fx.Int32(0xFF)
            if const_expr(flat_routes):
                route_row = fx.Int32(
                    buffer_ops.buffer_load(
                        route_ids_rsrc,
                        row,
                        vec_width=1,
                        dtype=T.i32,
                    )
                )
                route_valid = (
                    (token < i32_tokens)
                    & (route_row >= fx.Int32(0))
                    & (route_row < i32_routes)
                )
            else:
                route_row = token * fx.Int32(topk) + slot
                route_valid = (token < i32_tokens) & (slot < fx.Int32(topk))
            if route_valid:
                route_weight = fx.Float32(1.0)
                if const_expr(not defer_dy_scaling):
                    route_weight = fx.Float32(
                        buffer_ops.buffer_load(
                            weights_rsrc, row, vec_width=1, dtype=T.f32
                        )
                    )
                source_base = token * fx.Int32(hidden_size)
                destination_base = row * fx.Int32(hidden_size)
                if const_expr(vector_dy_copy):
                    # This path defers route-weight scaling to the fused dA/
                    # dscore consumer and does not materialize x_sorted, so dy
                    # is a bitwise copy.  Move one contiguous 128-bit chunk per
                    # lane instead of serializing eight BF16 load/wait/store
                    # sequences through the vector ALU.
                    dwords_per_row = hidden_size // 2
                    vectors_per_row = hidden_size // _DY_COPY_BF16_ELEMENTS
                    source_dword_base = token * fx.Int32(dwords_per_row)
                    destination_dword_base = row * fx.Int32(dwords_per_row)
                    for base in range_constexpr(0, vectors_per_row, _BLOCK_THREADS):
                        vector_index = tid + fx.Int32(base)
                        if vector_index < fx.Int32(vectors_per_row):
                            dword_offset = vector_index * fx.Int32(_DY_COPY_DWORDS)
                            dout_vector = buffer_ops.buffer_load(
                                dout_rsrc,
                                source_dword_base + dword_offset,
                                vec_width=_DY_COPY_DWORDS,
                                dtype=T.i32,
                            )
                            buffer_ops.buffer_store(
                                dout_vector,
                                dy_rsrc,
                                destination_dword_base + dword_offset,
                            )
                else:
                    for base in range_constexpr(0, hidden_size, _BLOCK_THREADS):
                        column = tid + fx.Int32(base)
                        if column < fx.Int32(hidden_size):
                            source = source_base + column
                            destination = destination_base + column
                            dout_value = buffer_ops.buffer_load(
                                dout_rsrc,
                                source,
                                vec_width=1,
                                dtype=elem_dtype,
                            ).extf(T.f32)
                            if const_expr(store_x_sorted):
                                x_value = buffer_ops.buffer_load(
                                    x_rsrc,
                                    source,
                                    vec_width=1,
                                    dtype=elem_dtype,
                                )
                                buffer_ops.buffer_store(
                                    x_value, x_sorted_rsrc, destination
                                )
                            buffer_ops.buffer_store(
                                fx.Float32(dout_value * route_weight).to(elem_dtype),
                                dy_rsrc,
                                destination,
                            )

                if const_expr(not defer_activation):
                    route_base = route_row * fx.Int32(projection_size)
                    activation_base = row * fx.Int32(intermediate_size)
                    for base in range_constexpr(0, intermediate_size, _BLOCK_THREADS):
                        column = tid + fx.Int32(base)
                        if column < fx.Int32(intermediate_size):
                            relative_gate = column * fx.Int32(projection_column_stride)
                            gate = buffer_ops.buffer_load(
                                route_preact_rsrc,
                                route_base + relative_gate,
                                vec_width=1,
                                dtype=elem_dtype,
                            ).extf(T.f32)
                            up = buffer_ops.buffer_load(
                                route_preact_rsrc,
                                route_base + relative_gate + fx.Int32(up_column_offset),
                                vec_width=1,
                                dtype=elem_dtype,
                            ).extf(T.f32)
                            activation_f32 = _activation_f32(gate, up, "swiglu")
                            buffer_ops.buffer_store(
                                fx.Float32(activation_f32).to(elem_dtype),
                                activation_rsrc,
                                activation_base + column,
                            )

        schedule_rsrc = buffer_ops.create_buffer_resource(schedule, max_size=True)
        total_tiles = fx.Int32(
            buffer_ops.buffer_load(schedule_rsrc, fx.Int32(0), vec_width=1, dtype=T.i32)
        )
        total_rows = total_tiles * fx.Int32(schedule_block_m)
        for task_value in range(gpu.block_idx.x, total_rows, gpu.grid_dim.x):
            task = fx.Int32(task_value)
            descriptor_index = task // fx.Int32(schedule_block_m)
            local_row = task % fx.Int32(schedule_block_m)
            if const_expr(exact_tile_queue):
                record_offset = fx.Int32(1) + descriptor_index * fx.Int32(3)
                first_row = fx.Int32(
                    buffer_ops.buffer_load(
                        schedule_rsrc,
                        record_offset + fx.Int32(1),
                        vec_width=1,
                        dtype=T.i32,
                    )
                )
                valid_rows = fx.Int32(
                    buffer_ops.buffer_load(
                        schedule_rsrc,
                        record_offset + fx.Int32(2),
                        vec_width=1,
                        dtype=T.i32,
                    )
                )
                if local_row < valid_rows:
                    prepare_real_row(first_row + local_row)
            else:
                descriptor = fx.Int32(
                    buffer_ops.buffer_load(
                        schedule_rsrc,
                        descriptor_index + fx.Int32(1),
                        vec_width=1,
                        dtype=T.i32,
                    )
                )
                prepare_real_row(descriptor * fx.Int32(schedule_block_m) + local_row)

    @flyc.jit
    def launch(
        hidden_states: fx.Tensor,
        grad_output: fx.Tensor,
        route_preactivation: fx.Tensor,
        x_sorted: fx.Tensor,
        activation: fx.Tensor,
        dy: fx.Tensor,
        sorted_weights: fx.Tensor,
        sorted_token_ids: fx.Tensor,
        sorted_route_ids: fx.Tensor,
        schedule: fx.Tensor,
        i32_tokens: fx.Int32,
        i32_routes: fx.Int32,
        i32_grid: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        fused_prepare_kernel(
            hidden_states,
            grad_output,
            route_preactivation,
            x_sorted,
            activation,
            dy,
            sorted_weights,
            sorted_token_ids,
            sorted_route_ids,
            schedule,
            i32_tokens,
            i32_routes,
        ).launch(
            grid=(i32_grid, 1, 1),
            block=(_BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    return launch


@functools.lru_cache(maxsize=128)
def _compile_activation_derivative(
    intermediate_size: int,
    activation_name: str,
    compute_dtype: str,
    interleaved_w1: bool,
    device_index: int,
    device_padded_rows: bool = False,
):
    """Compile the selected activation's Jacobian-vector product."""

    del device_index
    elem_dtype = fx.Float16 if compute_dtype == "fp16" else fx.BFloat16
    is_glu = activation_name in _GLU_ACTIVATIONS
    projection_size = intermediate_size * (2 if is_glu else 1)
    projection_column_stride = 2 if is_glu and interleaved_w1 else 1
    up_column_offset = (
        1 if is_glu and interleaved_w1 else intermediate_size if is_glu else 0
    )

    @flyc.kernel(known_block_size=[_BLOCK_THREADS, 1, 1])
    def activation_derivative_kernel(
        preactivation: fx.Tensor,
        da: fx.Tensor,
        dz: fx.Tensor,
        sorted_token_ids: fx.Tensor,
        num_valid_ids: fx.Tensor,
        i32_tokens: fx.Int32,
        i32_padded_rows: fx.Int32,
    ):
        tid = gpu.thread_idx.x
        preact_rsrc = buffer_ops.create_buffer_resource(preactivation, max_size=True)
        da_rsrc = buffer_ops.create_buffer_resource(da, max_size=True)
        dz_rsrc = buffer_ops.create_buffer_resource(dz, max_size=True)
        ids_rsrc = buffer_ops.create_buffer_resource(sorted_token_ids, max_size=True)

        def differentiate_row(row):
            for base in range_constexpr(0, intermediate_size, _BLOCK_THREADS):
                column = tid + fx.Int32(base)
                if column < fx.Int32(intermediate_size):
                    gate_offset = row * fx.Int32(projection_size) + column * fx.Int32(
                        projection_column_stride
                    )
                    up_offset = gate_offset + fx.Int32(up_column_offset)
                    act_offset = row * fx.Int32(intermediate_size) + column
                    gate = buffer_ops.buffer_load(
                        preact_rsrc, gate_offset, vec_width=1, dtype=elem_dtype
                    ).extf(T.f32)
                    up = buffer_ops.buffer_load(
                        preact_rsrc,
                        up_offset,
                        vec_width=1,
                        dtype=elem_dtype,
                    ).extf(T.f32)
                    da_value = buffer_ops.buffer_load(
                        da_rsrc, act_offset, vec_width=1, dtype=elem_dtype
                    ).extf(T.f32)
                    dz_gate, dz_up = _activation_backward_f32(
                        gate,
                        up,
                        da_value,
                        activation_name,
                    )
                    buffer_ops.buffer_store(
                        fx.Float32(dz_gate).to(elem_dtype), dz_rsrc, gate_offset
                    )
                    if const_expr(is_glu):
                        buffer_ops.buffer_store(
                            fx.Float32(dz_up).to(elem_dtype), dz_rsrc, up_offset
                        )

        padded_rows = i32_padded_rows
        if const_expr(device_padded_rows):
            valid_rsrc = buffer_ops.create_buffer_resource(num_valid_ids, max_size=True)
            padded_rows = fx.Int32(
                buffer_ops.buffer_load(
                    valid_rsrc, fx.Int32(0), vec_width=1, dtype=T.i32
                )
            )

        if const_expr(device_padded_rows):
            for row_value in range(gpu.block_idx.x, padded_rows, gpu.grid_dim.x):
                row = fx.Int32(row_value)
                packed = fx.Int32(
                    buffer_ops.buffer_load(ids_rsrc, row, vec_width=1, dtype=T.i32)
                )
                token = packed & fx.Int32(_TOKEN_MASK)
                if token < i32_tokens:
                    differentiate_row(row)
        else:
            differentiate_row(gpu.block_idx.x)

    @flyc.jit
    def launch(
        preactivation: fx.Tensor,
        da: fx.Tensor,
        dz: fx.Tensor,
        sorted_token_ids: fx.Tensor,
        num_valid_ids: fx.Tensor,
        i32_tokens: fx.Int32,
        i32_padded_rows: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        activation_derivative_kernel(
            preactivation,
            da,
            dz,
            sorted_token_ids,
            num_valid_ids,
            i32_tokens,
            i32_padded_rows,
        ).launch(
            grid=(i32_padded_rows, 1, 1),
            block=(_BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    return launch


@functools.lru_cache(maxsize=64)
def _compile_fused_activation_derivative_dscore_scale_dy(
    hidden_size: int,
    intermediate_size: int,
    topk: int,
    interleaved_w1: bool,
    device_index: int,
    schedule_block_m: int = _COMPACT_W1_BM,
    flat_routes: bool = False,
    device_padded_rows: bool = False,
    materialize_activation: bool = False,
    separate_dy_source: bool = False,
):
    """Fuse the post-dA row work for the BF16 SwiGLU state fast path.

    On entry ``da`` contains the unscaled, A16-materialized contraction
    ``q = dout @ W2`` and the dy source contains the unscaled ``dout``.
    Each workgroup owns a sorted route and performs three operations while the
    same row is resident:

    * multiply q by the FP32 route weight before applying the SwiGLU Jacobian;
    * reduce ``dot(q, activation)`` directly into route-order FP32 dscore;
    * scale the dy source into ``dy`` to restore the established A16 dW2 input
      contract.  Expert-major identity routes can read ``grad_output`` directly
      and thereby skip the otherwise redundant pre-dA copy.

    A route is owned by exactly one workgroup, so dscore needs neither an
    initialization launch nor atomics.  This is intentionally separate from
    the general derivative kernel until bias and ragged-route epilogues are
    implemented.
    """

    del device_index
    elem_dtype = fx.BFloat16
    projection_size = 2 * intermediate_size
    projection_column_stride = 2 if interleaved_w1 else 1
    up_column_offset = 1 if interleaved_w1 else intermediate_size
    vector_dy_scale = hidden_size % _DY_COPY_BF16_ELEMENTS == 0
    if not device_padded_rows and schedule_block_m not in (
        _COMPACT_W1_BM,
        _LARGE_GROUPED_DX_BM,
    ):
        raise ValueError("fused derivative schedule_block_m must be 16 or 64")

    @flyc.kernel(known_block_size=[_BLOCK_THREADS, 1, 1])
    def fused_kernel(
        route_preactivation: fx.Tensor,
        activation: fx.Tensor,
        da: fx.Tensor,
        dy_source: fx.Tensor,
        dy: fx.Tensor,
        dz: fx.Tensor,
        sorted_weights: fx.Tensor,
        sorted_token_ids: fx.Tensor,
        sorted_route_ids: fx.Tensor,
        dtopk_weights: fx.Tensor,
        schedule: fx.Tensor,
        i32_tokens: fx.Int32,
        i32_routes: fx.Int32,
    ):
        tid = gpu.thread_idx.x
        zero_f32 = fx.Float32(0.0)
        fm_fast = arith.FastMathFlags.fast
        route_preact_rsrc = buffer_ops.create_buffer_resource(
            route_preactivation, max_size=True
        )
        activation_rsrc = buffer_ops.create_buffer_resource(activation, max_size=True)
        da_rsrc = buffer_ops.create_buffer_resource(da, max_size=True)
        dy_rsrc = buffer_ops.create_buffer_resource(dy, max_size=True)
        dy_source_rsrc = dy_rsrc
        if const_expr(separate_dy_source):
            dy_source_rsrc = buffer_ops.create_buffer_resource(
                dy_source,
                max_size=True,
            )
        dz_rsrc = buffer_ops.create_buffer_resource(dz, max_size=True)
        weights_rsrc = buffer_ops.create_buffer_resource(sorted_weights, max_size=True)
        ids_rsrc = buffer_ops.create_buffer_resource(sorted_token_ids, max_size=True)
        route_ids_rsrc = ids_rsrc
        if const_expr(flat_routes):
            route_ids_rsrc = buffer_ops.create_buffer_resource(
                sorted_route_ids,
                max_size=True,
            )
        ds_rsrc = buffer_ops.create_buffer_resource(dtopk_weights, max_size=True)
        schedule_rsrc = buffer_ops.create_buffer_resource(schedule, max_size=True)
        lds = fx.SharedAllocator().allocate(_ScoreBackwardSharedStorage).peek()
        reduction = lds.reduction.view(fx.make_layout(_RED_SLOTS, 1))

        def wave_reduce_add(value):
            result = value
            with fx.fastmath(fm_fast):
                for shift_index in range_constexpr(int(math.log2(_WARP_SIZE))):
                    offset = _WARP_SIZE // (2 << shift_index)
                    result = result + gpu.shuffle_xor(result, offset, _WARP_SIZE)
            return result

        def process_row(row, route_row, token):
            route_weight = fx.Float32(
                buffer_ops.buffer_load(weights_rsrc, row, vec_width=1, dtype=T.f32)
            )
            thread_dot = zero_f32
            route_base = route_row * fx.Int32(projection_size)
            sorted_base = row * fx.Int32(projection_size)
            for base in range_constexpr(0, intermediate_size, _BLOCK_THREADS):
                column = tid + fx.Int32(base)
                if column < fx.Int32(intermediate_size):
                    relative_gate = column * fx.Int32(projection_column_stride)
                    route_gate_offset = route_base + relative_gate
                    sorted_gate_offset = sorted_base + relative_gate
                    act_offset = row * fx.Int32(intermediate_size) + column
                    gate = buffer_ops.buffer_load(
                        route_preact_rsrc,
                        route_gate_offset,
                        vec_width=1,
                        dtype=elem_dtype,
                    ).extf(T.f32)
                    up = buffer_ops.buffer_load(
                        route_preact_rsrc,
                        route_gate_offset + fx.Int32(up_column_offset),
                        vec_width=1,
                        dtype=elem_dtype,
                    ).extf(T.f32)
                    q = buffer_ops.buffer_load(
                        da_rsrc, act_offset, vec_width=1, dtype=elem_dtype
                    ).extf(T.f32)
                    if const_expr(materialize_activation):
                        sigmoid = _sigmoid_f32(gate)
                        one = fx.Float32(1.0)
                        activated_gate = gate * sigmoid
                        derivative = sigmoid * (one + gate * (one - sigmoid))
                        activation_bf16 = fx.Float32(activated_gate * up).to(elem_dtype)
                        buffer_ops.buffer_store(
                            activation_bf16,
                            activation_rsrc,
                            act_offset,
                        )
                        activation_value = fx.Float32(activation_bf16)
                        scaled_q = q * route_weight
                        dz_gate = scaled_q * up * derivative
                        dz_up = scaled_q * activated_gate
                    else:
                        activation_value = buffer_ops.buffer_load(
                            activation_rsrc,
                            act_offset,
                            vec_width=1,
                            dtype=elem_dtype,
                        ).extf(T.f32)
                        dz_gate, dz_up = _activation_backward_f32(
                            gate,
                            up,
                            q * route_weight,
                            "swiglu",
                        )
                    thread_dot = thread_dot + q * activation_value
                    buffer_ops.buffer_store(
                        fx.Float32(dz_gate).to(elem_dtype), dz_rsrc, sorted_gate_offset
                    )
                    buffer_ops.buffer_store(
                        fx.Float32(dz_up).to(elem_dtype),
                        dz_rsrc,
                        sorted_gate_offset + fx.Int32(up_column_offset),
                    )

            # dW2 retains the existing multiply-before-GEMM A16 boundary.
            if const_expr(vector_dy_scale):
                # Scale eight packed BF16 values per VMEM transaction so the
                # in-place pass does not serialize one load/wait/store chain
                # per scalar element.
                dwords_per_row = hidden_size // 2
                vectors_per_row = hidden_size // _DY_COPY_BF16_ELEMENTS
                source_row = token if separate_dy_source else row
                source_dword_base = source_row * fx.Int32(dwords_per_row)
                destination_dword_base = row * fx.Int32(dwords_per_row)
                for base in range_constexpr(0, vectors_per_row, _BLOCK_THREADS):
                    vector_index = tid + fx.Int32(base)
                    if vector_index < fx.Int32(vectors_per_row):
                        dword_offset = vector_index * fx.Int32(_DY_COPY_DWORDS)
                        dout_raw = buffer_ops.buffer_load(
                            dy_source_rsrc,
                            source_dword_base + dword_offset,
                            vec_width=_DY_COPY_DWORDS,
                            dtype=T.i32,
                        )
                        dout_vector = (
                            fx.Vector(dout_raw).bitcast(fx.BFloat16).to(fx.Float32)
                        )
                        scaled_vector = fx.Vector.from_elements(
                            [
                                dout_vector[index] * route_weight
                                for index in range_constexpr(_DY_COPY_BF16_ELEMENTS)
                            ],
                            fx.Float32,
                        ).to(fx.BFloat16)
                        buffer_ops.buffer_store(
                            scaled_vector.bitcast(fx.Int32),
                            dy_rsrc,
                            destination_dword_base + dword_offset,
                        )
            else:
                for base in range_constexpr(0, hidden_size, _BLOCK_THREADS):
                    column = tid + fx.Int32(base)
                    if column < fx.Int32(hidden_size):
                        source_row = token if separate_dy_source else row
                        source_offset = source_row * fx.Int32(hidden_size) + column
                        destination_offset = row * fx.Int32(hidden_size) + column
                        dout_value = buffer_ops.buffer_load(
                            dy_source_rsrc,
                            source_offset,
                            vec_width=1,
                            dtype=elem_dtype,
                        ).extf(T.f32)
                        buffer_ops.buffer_store(
                            fx.Float32(dout_value * route_weight).to(elem_dtype),
                            dy_rsrc,
                            destination_offset,
                        )

            reduced = wave_reduce_add(thread_dot)
            if const_expr(_RED_SLOTS > 1):
                lane = tid % fx.Int32(_WARP_SIZE)
                wave = tid // fx.Int32(_WARP_SIZE)
                if lane == fx.Int32(0):
                    fx.memref_store(reduced, reduction, wave)
                gpu.barrier()
                if wave == fx.Int32(0):
                    in_range = lane < fx.Int32(_RED_SLOTS)
                    safe_lane = in_range.select(lane, fx.Int32(0))
                    partial = fx.memref_load(reduction, safe_lane)
                    reduced = wave_reduce_add(in_range.select(partial, zero_f32))
                    if lane == fx.Int32(0):
                        fx.memref_store(reduced, reduction, fx.Int32(0))
                gpu.barrier()
                reduced = fx.memref_load(reduction, fx.Int32(0))

            if tid == fx.Int32(0):
                buffer_ops.buffer_store(reduced, ds_rsrc, route_row)

        def process_sorted_row(row):
            packed = fx.Int32(
                buffer_ops.buffer_load(ids_rsrc, row, vec_width=1, dtype=T.i32)
            )
            token = packed & fx.Int32(_TOKEN_MASK)
            slot = (packed >> fx.Int32(24)) & fx.Int32(0xFF)
            if const_expr(flat_routes):
                route_row = fx.Int32(
                    buffer_ops.buffer_load(
                        route_ids_rsrc,
                        row,
                        vec_width=1,
                        dtype=T.i32,
                    )
                )
                route_valid = (
                    (token < i32_tokens)
                    & (route_row >= fx.Int32(0))
                    & (route_row < i32_routes)
                )
            else:
                route_row = token * fx.Int32(topk) + slot
                route_valid = (token < i32_tokens) & (slot < fx.Int32(topk))
            if route_valid:
                process_row(row, route_row, token)

        if const_expr(device_padded_rows):
            padded_rows = fx.Int32(
                buffer_ops.buffer_load(
                    schedule_rsrc,
                    fx.Int32(0),
                    vec_width=1,
                    dtype=T.i32,
                )
            )
            for row_value in range(gpu.block_idx.x, padded_rows, gpu.grid_dim.x):
                process_sorted_row(fx.Int32(row_value))
        else:
            total_tiles = fx.Int32(
                buffer_ops.buffer_load(
                    schedule_rsrc,
                    fx.Int32(0),
                    vec_width=1,
                    dtype=T.i32,
                )
            )
            total_rows = total_tiles * fx.Int32(schedule_block_m)
            for task_value in range(gpu.block_idx.x, total_rows, gpu.grid_dim.x):
                task = fx.Int32(task_value)
                descriptor_index = task // fx.Int32(schedule_block_m)
                local_row = task % fx.Int32(schedule_block_m)
                descriptor = fx.Int32(
                    buffer_ops.buffer_load(
                        schedule_rsrc,
                        descriptor_index + fx.Int32(1),
                        vec_width=1,
                        dtype=T.i32,
                    )
                )
                process_sorted_row(descriptor * fx.Int32(schedule_block_m) + local_row)

    @flyc.jit
    def launch(
        route_preactivation: fx.Tensor,
        activation: fx.Tensor,
        da: fx.Tensor,
        dy_source: fx.Tensor,
        dy: fx.Tensor,
        dz: fx.Tensor,
        sorted_weights: fx.Tensor,
        sorted_token_ids: fx.Tensor,
        sorted_route_ids: fx.Tensor,
        dtopk_weights: fx.Tensor,
        schedule: fx.Tensor,
        i32_tokens: fx.Int32,
        i32_routes: fx.Int32,
        i32_grid: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        fused_kernel(
            route_preactivation,
            activation,
            da,
            dy_source,
            dy,
            dz,
            sorted_weights,
            sorted_token_ids,
            sorted_route_ids,
            dtopk_weights,
            schedule,
            i32_tokens,
            i32_routes,
        ).launch(
            grid=(i32_grid, 1, 1),
            block=(_BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    return launch


@functools.lru_cache(maxsize=128)
def _compile_activation_derivative_from_forward_state(
    intermediate_size: int,
    topk: int,
    interleaved_w1: bool,
    device_index: int,
):
    """Differentiate SwiGLU directly from compact route-order forward state."""

    del device_index
    elem_dtype = fx.BFloat16
    projection_size = 2 * intermediate_size
    projection_column_stride = 2 if interleaved_w1 else 1
    up_column_offset = 1 if interleaved_w1 else intermediate_size
    compact_block_m = _COMPACT_W1_BM

    @flyc.kernel(known_block_size=[_BLOCK_THREADS, 1, 1])
    def derivative_from_state_kernel(
        route_preactivation: fx.Tensor,
        da: fx.Tensor,
        dz: fx.Tensor,
        sorted_token_ids: fx.Tensor,
        schedule: fx.Tensor,
        i32_tokens: fx.Int32,
    ):
        tid = gpu.thread_idx.x
        route_preact_rsrc = buffer_ops.create_buffer_resource(
            route_preactivation, max_size=True
        )
        da_rsrc = buffer_ops.create_buffer_resource(da, max_size=True)
        dz_rsrc = buffer_ops.create_buffer_resource(dz, max_size=True)
        ids_rsrc = buffer_ops.create_buffer_resource(sorted_token_ids, max_size=True)

        def differentiate_real_row(row):
            packed = fx.Int32(
                buffer_ops.buffer_load(ids_rsrc, row, vec_width=1, dtype=T.i32)
            )
            token = packed & fx.Int32(_TOKEN_MASK)
            slot = (packed >> fx.Int32(24)) & fx.Int32(0xFF)
            if (token < i32_tokens) & (slot < fx.Int32(topk)):
                route_row = token * fx.Int32(topk) + slot
                route_base = route_row * fx.Int32(projection_size)
                sorted_base = row * fx.Int32(projection_size)
                activation_base = row * fx.Int32(intermediate_size)
                for base in range_constexpr(0, intermediate_size, _BLOCK_THREADS):
                    column = tid + fx.Int32(base)
                    if column < fx.Int32(intermediate_size):
                        relative_gate = column * fx.Int32(projection_column_stride)
                        gate = buffer_ops.buffer_load(
                            route_preact_rsrc,
                            route_base + relative_gate,
                            vec_width=1,
                            dtype=elem_dtype,
                        ).extf(T.f32)
                        up = buffer_ops.buffer_load(
                            route_preact_rsrc,
                            route_base + relative_gate + fx.Int32(up_column_offset),
                            vec_width=1,
                            dtype=elem_dtype,
                        ).extf(T.f32)
                        da_value = buffer_ops.buffer_load(
                            da_rsrc,
                            activation_base + column,
                            vec_width=1,
                            dtype=elem_dtype,
                        ).extf(T.f32)
                        dz_gate, dz_up = _activation_backward_f32(
                            gate,
                            up,
                            da_value,
                            "swiglu",
                        )
                        buffer_ops.buffer_store(
                            fx.Float32(dz_gate).to(elem_dtype),
                            dz_rsrc,
                            sorted_base + relative_gate,
                        )
                        buffer_ops.buffer_store(
                            fx.Float32(dz_up).to(elem_dtype),
                            dz_rsrc,
                            sorted_base + relative_gate + fx.Int32(up_column_offset),
                        )

        schedule_rsrc = buffer_ops.create_buffer_resource(schedule, max_size=True)
        total_tiles = fx.Int32(
            buffer_ops.buffer_load(schedule_rsrc, fx.Int32(0), vec_width=1, dtype=T.i32)
        )
        total_rows = total_tiles * fx.Int32(compact_block_m)
        for task_value in range(gpu.block_idx.x, total_rows, gpu.grid_dim.x):
            task = fx.Int32(task_value)
            descriptor_index = task // fx.Int32(compact_block_m)
            local_row = task % fx.Int32(compact_block_m)
            descriptor = fx.Int32(
                buffer_ops.buffer_load(
                    schedule_rsrc,
                    descriptor_index + fx.Int32(1),
                    vec_width=1,
                    dtype=T.i32,
                )
            )
            differentiate_real_row(descriptor * fx.Int32(compact_block_m) + local_row)

    @flyc.jit
    def launch(
        route_preactivation: fx.Tensor,
        da: fx.Tensor,
        dz: fx.Tensor,
        sorted_token_ids: fx.Tensor,
        schedule: fx.Tensor,
        i32_tokens: fx.Int32,
        i32_grid: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        derivative_from_state_kernel(
            route_preactivation,
            da,
            dz,
            sorted_token_ids,
            schedule,
            i32_tokens,
        ).launch(
            grid=(i32_grid, 1, 1),
            block=(_BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    return launch


@functools.lru_cache(maxsize=128)
def _compile_bias_gradient_clear(
    projection_size: int,
    hidden_size: int,
    num_experts: int,
    compute_dtype: str,
    device_index: int,
):
    """Compile the exact-zero initialization for all expert bias gradients."""

    del device_index
    elem_dtype = fx.Float16 if compute_dtype == "fp16" else fx.BFloat16
    db1_elements = num_experts * projection_size
    db2_elements = num_experts * hidden_size
    grid_size = (max(db1_elements, db2_elements) + _BLOCK_THREADS - 1) // _BLOCK_THREADS

    @flyc.kernel(known_block_size=[_BLOCK_THREADS, 1, 1])
    def clear_kernel(db1: fx.Tensor, db2: fx.Tensor):
        index = gpu.block_idx.x * fx.Int32(_BLOCK_THREADS) + gpu.thread_idx.x
        db1_rsrc = buffer_ops.create_buffer_resource(db1, max_size=True)
        db2_rsrc = buffer_ops.create_buffer_resource(db2, max_size=True)
        if index < fx.Int32(db1_elements):
            buffer_ops.buffer_store(elem_dtype(0.0), db1_rsrc, index)
        if index < fx.Int32(db2_elements):
            buffer_ops.buffer_store(elem_dtype(0.0), db2_rsrc, index)

    @flyc.jit
    def launch(
        db1: fx.Tensor,
        db2: fx.Tensor,
        stream: fx.Stream = fx.Stream(None),
    ):
        clear_kernel(db1, db2).launch(
            grid=(grid_size, 1, 1),
            block=(_BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    return launch


@functools.lru_cache(maxsize=128)
def _compile_bias_gradient_reduction(
    projection_size: int,
    hidden_size: int,
    compute_dtype: str,
    device_index: int,
):
    """Compile one expert segment's FP32-accumulating bias reductions."""

    del device_index
    elem_dtype = fx.Float16 if compute_dtype == "fp16" else fx.BFloat16
    grid_size = (
        max(projection_size, hidden_size) + _BLOCK_THREADS - 1
    ) // _BLOCK_THREADS

    @flyc.kernel(known_block_size=[_BLOCK_THREADS, 1, 1])
    def reduction_kernel(
        dz: fx.Tensor,
        dy: fx.Tensor,
        db1: fx.Tensor,
        db2: fx.Tensor,
        i32_rows: fx.Int32,
    ):
        column = gpu.block_idx.x * fx.Int32(_BLOCK_THREADS) + gpu.thread_idx.x
        dz_rsrc = buffer_ops.create_buffer_resource(dz, max_size=True)
        dy_rsrc = buffer_ops.create_buffer_resource(dy, max_size=True)
        db1_rsrc = buffer_ops.create_buffer_resource(db1, max_size=True)
        db2_rsrc = buffer_ops.create_buffer_resource(db2, max_size=True)

        if column < fx.Int32(projection_size):
            db1_acc = fx.Float32(0.0)
            for row in range(fx.Int32(0), i32_rows, fx.Int32(1)):
                offset = row * fx.Int32(projection_size) + column
                value = buffer_ops.buffer_load(
                    dz_rsrc, offset, vec_width=1, dtype=elem_dtype
                ).extf(T.f32)
                db1_acc = db1_acc + value
            buffer_ops.buffer_store(db1_acc.to(elem_dtype), db1_rsrc, column)

        if column < fx.Int32(hidden_size):
            db2_acc = fx.Float32(0.0)
            for row in range(fx.Int32(0), i32_rows, fx.Int32(1)):
                offset = row * fx.Int32(hidden_size) + column
                value = buffer_ops.buffer_load(
                    dy_rsrc, offset, vec_width=1, dtype=elem_dtype
                ).extf(T.f32)
                db2_acc = db2_acc + value
            buffer_ops.buffer_store(db2_acc.to(elem_dtype), db2_rsrc, column)

    @flyc.jit
    def launch(
        dz: fx.Tensor,
        dy: fx.Tensor,
        db1: fx.Tensor,
        db2: fx.Tensor,
        i32_rows: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        reduction_kernel(dz, dy, db1, db2, i32_rows).launch(
            grid=(grid_size, 1, 1),
            block=(_BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    return launch


@functools.lru_cache(maxsize=128)
def _compile_score_backward(
    hidden_size: int,
    topk: int,
    compute_dtype: str,
    device_index: int,
    device_padded_rows: bool = False,
    token_major_dout: bool = False,
):
    """Compile ``ds = dot(dout, materialized_down_projection)``."""

    del device_index
    elem_dtype = fx.Float16 if compute_dtype == "fp16" else fx.BFloat16

    @flyc.kernel(known_block_size=[_BLOCK_THREADS, 1, 1])
    def score_backward_kernel(
        dout_sorted: fx.Tensor,
        projection: fx.Tensor,
        sorted_token_ids: fx.Tensor,
        dtopk_weights: fx.Tensor,
        num_valid_ids: fx.Tensor,
        i32_tokens: fx.Int32,
        i32_padded_rows: fx.Int32,
    ):
        tid = gpu.thread_idx.x
        zero_f32 = fx.Float32(0.0)
        fm_fast = arith.FastMathFlags.fast

        ids_rsrc = buffer_ops.create_buffer_resource(sorted_token_ids, max_size=True)
        dout_rsrc = buffer_ops.create_buffer_resource(dout_sorted, max_size=True)
        projection_rsrc = buffer_ops.create_buffer_resource(projection, max_size=True)
        ds_rsrc = buffer_ops.create_buffer_resource(dtopk_weights, max_size=True)
        lds = fx.SharedAllocator().allocate(_ScoreBackwardSharedStorage).peek()
        reduction = lds.reduction.view(fx.make_layout(_RED_SLOTS, 1))

        def wave_reduce_add(value):
            result = value
            with fx.fastmath(fm_fast):
                for shift_index in range_constexpr(int(math.log2(_WARP_SIZE))):
                    offset = _WARP_SIZE // (2 << shift_index)
                    result = result + gpu.shuffle_xor(result, offset, _WARP_SIZE)
            return result

        def reduce_row(row, token, slot):
            thread_dot = zero_f32
            for base in range_constexpr(0, hidden_size, _BLOCK_THREADS):
                column = tid + fx.Int32(base)
                if column < fx.Int32(hidden_size):
                    projection_offset = row * fx.Int32(hidden_size) + column
                    dout_row = token if const_expr(token_major_dout) else row
                    dout_offset = dout_row * fx.Int32(hidden_size) + column
                    dout_value = buffer_ops.buffer_load(
                        dout_rsrc,
                        dout_offset,
                        vec_width=1,
                        dtype=elem_dtype,
                    ).extf(T.f32)
                    projected = buffer_ops.buffer_load(
                        projection_rsrc,
                        projection_offset,
                        vec_width=1,
                        dtype=elem_dtype,
                    ).extf(T.f32)
                    thread_dot = thread_dot + dout_value * projected

            reduced = wave_reduce_add(thread_dot)
            if const_expr(_RED_SLOTS > 1):
                lane = tid % fx.Int32(_WARP_SIZE)
                wave = tid // fx.Int32(_WARP_SIZE)
                if lane == fx.Int32(0):
                    fx.memref_store(reduced, reduction, wave)
                gpu.barrier()
                if wave == fx.Int32(0):
                    in_range = lane < fx.Int32(_RED_SLOTS)
                    safe_lane = in_range.select(lane, fx.Int32(0))
                    partial = fx.memref_load(reduction, safe_lane)
                    reduced = wave_reduce_add(in_range.select(partial, zero_f32))
                    if lane == fx.Int32(0):
                        fx.memref_store(reduced, reduction, fx.Int32(0))
                gpu.barrier()
                reduced = fx.memref_load(reduction, fx.Int32(0))

            if tid == fx.Int32(0):
                destination = token * fx.Int32(topk) + slot
                buffer_ops.buffer_store(reduced, ds_rsrc, destination)

        padded_rows = i32_padded_rows
        if const_expr(device_padded_rows):
            valid_rsrc = buffer_ops.create_buffer_resource(num_valid_ids, max_size=True)
            padded_rows = fx.Int32(
                buffer_ops.buffer_load(
                    valid_rsrc, fx.Int32(0), vec_width=1, dtype=T.i32
                )
            )

        if const_expr(device_padded_rows):
            for row_value in range(gpu.block_idx.x, padded_rows, gpu.grid_dim.x):
                row = fx.Int32(row_value)
                packed = fx.Int32(
                    buffer_ops.buffer_load(ids_rsrc, row, vec_width=1, dtype=T.i32)
                )
                token = packed & fx.Int32(_TOKEN_MASK)
                slot = packed >> fx.Int32(24)
                valid_route = token < i32_tokens
                if valid_route:
                    reduce_row(row, token, slot)
        else:
            row = gpu.block_idx.x
            packed = fx.Int32(
                buffer_ops.buffer_load(ids_rsrc, row, vec_width=1, dtype=T.i32)
            )
            token = packed & fx.Int32(_TOKEN_MASK)
            slot = packed >> fx.Int32(24)
            valid_route = token < i32_tokens
            # Legacy row grids intentionally retain their original
            # execute-then-suppress behavior for padded routes.
            thread_dot = zero_f32
            for base in range_constexpr(0, hidden_size, _BLOCK_THREADS):
                column = tid + fx.Int32(base)
                if column < fx.Int32(hidden_size):
                    offset = row * fx.Int32(hidden_size) + column
                    dout_value = buffer_ops.buffer_load(
                        dout_rsrc,
                        offset,
                        vec_width=1,
                        dtype=elem_dtype,
                    ).extf(T.f32)
                    projected = buffer_ops.buffer_load(
                        projection_rsrc,
                        offset,
                        vec_width=1,
                        dtype=elem_dtype,
                    ).extf(T.f32)
                    thread_dot = thread_dot + dout_value * projected
            reduced = wave_reduce_add(thread_dot)
            if const_expr(_RED_SLOTS > 1):
                lane = tid % fx.Int32(_WARP_SIZE)
                wave = tid // fx.Int32(_WARP_SIZE)
                if lane == fx.Int32(0):
                    fx.memref_store(reduced, reduction, wave)
                gpu.barrier()
                if wave == fx.Int32(0):
                    in_range = lane < fx.Int32(_RED_SLOTS)
                    safe_lane = in_range.select(lane, fx.Int32(0))
                    partial = fx.memref_load(reduction, safe_lane)
                    reduced = wave_reduce_add(in_range.select(partial, zero_f32))
                    if lane == fx.Int32(0):
                        fx.memref_store(reduced, reduction, fx.Int32(0))
                gpu.barrier()
                reduced = fx.memref_load(reduction, fx.Int32(0))
            if tid == fx.Int32(0):
                if valid_route:
                    destination = token * fx.Int32(topk) + slot
                    buffer_ops.buffer_store(reduced, ds_rsrc, destination)

    @flyc.jit
    def launch(
        dout_sorted: fx.Tensor,
        projection: fx.Tensor,
        sorted_token_ids: fx.Tensor,
        dtopk_weights: fx.Tensor,
        num_valid_ids: fx.Tensor,
        i32_tokens: fx.Int32,
        i32_padded_rows: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        score_backward_kernel(
            dout_sorted,
            projection,
            sorted_token_ids,
            dtopk_weights,
            num_valid_ids,
            i32_tokens,
            i32_padded_rows,
        ).launch(
            grid=(i32_padded_rows, 1, 1),
            block=(_BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    return launch


@functools.lru_cache(maxsize=128)
def _compile_route_score_backward(
    hidden_size: int, compute_dtype: str, device_index: int
):
    """Compile flat-route ``ds = dot(dout, down_projection)`` scatter."""

    del device_index
    elem_dtype = fx.Float16 if compute_dtype == "fp16" else fx.BFloat16

    @flyc.kernel(known_block_size=[_BLOCK_THREADS, 1, 1])
    def score_backward_kernel(
        dout_sorted: fx.Tensor,
        projection: fx.Tensor,
        sorted_token_ids: fx.Tensor,
        sorted_route_ids: fx.Tensor,
        droute_weights: fx.Tensor,
        i32_tokens: fx.Int32,
        i32_routes: fx.Int32,
    ):
        row = gpu.block_idx.x
        tid = gpu.thread_idx.x
        zero_f32 = fx.Float32(0.0)
        fm_fast = arith.FastMathFlags.fast

        token_rsrc = buffer_ops.create_buffer_resource(sorted_token_ids, max_size=True)
        route_rsrc = buffer_ops.create_buffer_resource(sorted_route_ids, max_size=True)
        dout_rsrc = buffer_ops.create_buffer_resource(dout_sorted, max_size=True)
        projection_rsrc = buffer_ops.create_buffer_resource(projection, max_size=True)
        ds_rsrc = buffer_ops.create_buffer_resource(droute_weights, max_size=True)
        token = fx.Int32(
            buffer_ops.buffer_load(token_rsrc, row, vec_width=1, dtype=T.i32)
        )
        route = fx.Int32(
            buffer_ops.buffer_load(route_rsrc, row, vec_width=1, dtype=T.i32)
        )
        valid_route = (
            (token >= fx.Int32(0))
            & (token < i32_tokens)
            & (route >= fx.Int32(0))
            & (route < i32_routes)
        )

        thread_dot = zero_f32
        for base in range_constexpr(0, hidden_size, _BLOCK_THREADS):
            column = tid + fx.Int32(base)
            if column < fx.Int32(hidden_size):
                offset = row * fx.Int32(hidden_size) + column
                dout_value = buffer_ops.buffer_load(
                    dout_rsrc,
                    offset,
                    vec_width=1,
                    dtype=elem_dtype,
                ).extf(T.f32)
                projected = buffer_ops.buffer_load(
                    projection_rsrc,
                    offset,
                    vec_width=1,
                    dtype=elem_dtype,
                ).extf(T.f32)
                thread_dot = thread_dot + dout_value * projected

        lds = fx.SharedAllocator().allocate(_ScoreBackwardSharedStorage).peek()
        reduction = lds.reduction.view(fx.make_layout(_RED_SLOTS, 1))

        def wave_reduce_add(value):
            result = value
            with fx.fastmath(fm_fast):
                for shift_index in range_constexpr(int(math.log2(_WARP_SIZE))):
                    offset = _WARP_SIZE // (2 << shift_index)
                    result = result + gpu.shuffle_xor(result, offset, _WARP_SIZE)
            return result

        reduced = wave_reduce_add(thread_dot)
        if const_expr(_RED_SLOTS > 1):
            lane = tid % fx.Int32(_WARP_SIZE)
            wave = tid // fx.Int32(_WARP_SIZE)
            if lane == fx.Int32(0):
                fx.memref_store(reduced, reduction, wave)
            gpu.barrier()
            if wave == fx.Int32(0):
                in_range = lane < fx.Int32(_RED_SLOTS)
                safe_lane = in_range.select(lane, fx.Int32(0))
                partial = fx.memref_load(reduction, safe_lane)
                reduced = wave_reduce_add(in_range.select(partial, zero_f32))
                if lane == fx.Int32(0):
                    fx.memref_store(reduced, reduction, fx.Int32(0))
            gpu.barrier()
            reduced = fx.memref_load(reduction, fx.Int32(0))

        if tid == fx.Int32(0):
            if valid_route:
                buffer_ops.buffer_store(reduced, ds_rsrc, route)

    @flyc.jit
    def launch(
        dout_sorted: fx.Tensor,
        projection: fx.Tensor,
        sorted_token_ids: fx.Tensor,
        sorted_route_ids: fx.Tensor,
        droute_weights: fx.Tensor,
        i32_tokens: fx.Int32,
        i32_routes: fx.Int32,
        i32_padded_rows: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        score_backward_kernel(
            dout_sorted,
            projection,
            sorted_token_ids,
            sorted_route_ids,
            droute_weights,
            i32_tokens,
            i32_routes,
        ).launch(
            grid=(i32_padded_rows, 1, 1),
            block=(_BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    return launch


@functools.lru_cache(maxsize=128)
def _compile_ragged_dx_reduction(
    hidden_size: int, compute_dtype: str, device_index: int
):
    """Compile FP32 atomic accumulation of variable-count route gradients."""

    del device_index
    elem_dtype = fx.Float16 if compute_dtype == "fp16" else fx.BFloat16

    @flyc.kernel(known_block_size=[_BLOCK_THREADS, 1, 1])
    def clear_kernel(dx_accum: fx.Tensor, i32_elements: fx.Int32):
        index = gpu.block_idx.x * fx.Int32(_BLOCK_THREADS) + gpu.thread_idx.x
        if index < i32_elements:
            accum_rsrc = buffer_ops.create_buffer_resource(dx_accum, max_size=True)
            buffer_ops.buffer_store(fx.Float32(0.0), accum_rsrc, index)

    @flyc.kernel(known_block_size=[_BLOCK_THREADS, 1, 1])
    def scatter_kernel(
        dx_sorted: fx.Tensor,
        sorted_token_ids: fx.Tensor,
        dx_accum: fx.Tensor,
        i32_tokens: fx.Int32,
        i32_padded_rows: fx.Int32,
    ):
        index = gpu.block_idx.x * fx.Int32(_BLOCK_THREADS) + gpu.thread_idx.x
        total = i32_padded_rows * fx.Int32(hidden_size)
        if index < total:
            row = index // fx.Int32(hidden_size)
            column = index % fx.Int32(hidden_size)
            token_rsrc = buffer_ops.create_buffer_resource(
                sorted_token_ids, max_size=True
            )
            dx_rsrc = buffer_ops.create_buffer_resource(dx_sorted, max_size=True)
            token = fx.Int32(
                buffer_ops.buffer_load(token_rsrc, row, vec_width=1, dtype=T.i32)
            )
            if (token >= fx.Int32(0)) & (token < i32_tokens):
                value = buffer_ops.buffer_load(
                    dx_rsrc, index, vec_width=1, dtype=elem_dtype
                ).extf(T.f32)
                destination = token * fx.Int32(hidden_size) + column
                atomic_add(dx_accum, destination, value, dtype_bytes=4)

    @flyc.kernel(known_block_size=[_BLOCK_THREADS, 1, 1])
    def finalize_kernel(dx_accum: fx.Tensor, dx: fx.Tensor, i32_elements: fx.Int32):
        index = gpu.block_idx.x * fx.Int32(_BLOCK_THREADS) + gpu.thread_idx.x
        if index < i32_elements:
            accum_rsrc = buffer_ops.create_buffer_resource(dx_accum, max_size=True)
            dx_rsrc = buffer_ops.create_buffer_resource(dx, max_size=True)
            value = buffer_ops.buffer_load(accum_rsrc, index, vec_width=1, dtype=T.f32)
            buffer_ops.buffer_store(fx.Float32(value).to(elem_dtype), dx_rsrc, index)

    @flyc.jit
    def launch(
        dx_sorted: fx.Tensor,
        sorted_token_ids: fx.Tensor,
        dx_accum: fx.Tensor,
        dx: fx.Tensor,
        i32_tokens: fx.Int32,
        i32_padded_rows: fx.Int32,
        i32_clear_grid: fx.Int32,
        i32_scatter_grid: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        elements = i32_tokens * fx.Int32(hidden_size)
        clear_kernel(dx_accum, elements).launch(
            grid=(i32_clear_grid, 1, 1),
            block=(_BLOCK_THREADS, 1, 1),
            stream=stream,
        )
        scatter_kernel(
            dx_sorted,
            sorted_token_ids,
            dx_accum,
            i32_tokens,
            i32_padded_rows,
        ).launch(
            grid=(i32_scatter_grid, 1, 1),
            block=(_BLOCK_THREADS, 1, 1),
            stream=stream,
        )
        finalize_kernel(dx_accum, dx, elements).launch(
            grid=(i32_clear_grid, 1, 1),
            block=(_BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    return launch


@functools.lru_cache(maxsize=16)
def _compile_flat_segmented_dx_reduction(
    hidden_size: int,
    compute_dtype: str,
    device_index: int,
):
    """Reduce route-order dX directly into token-order storage.

    The retained E16 adapter supplies route rows in nondecreasing token order.
    One workgroup therefore owns a token, finds its half-open route segment
    with two fixed-trip lower bounds, accumulates all matching BF16 rows in
    FP32, and writes final BF16 dX once.  This replaces the legacy clear,
    atomic-scatter, and finalize launch sequence for that audited contract.
    """

    del device_index
    if compute_dtype != "bf16":
        raise ValueError(
            "flat segmented dX currently supports compute_dtype='bf16' only"
        )
    if hidden_size <= 0 or hidden_size % (_BLOCK_THREADS * 8):
        raise ValueError(
            "flat segmented dX requires hidden_size to be a positive multiple "
            f"of {_BLOCK_THREADS * 8}"
        )
    dwords_per_row = hidden_size // 2
    vectors_per_thread = hidden_size // (_BLOCK_THREADS * 8)
    # R is a runtime extent and must not create a JIT specialization for every
    # EP shard.  Signed-i32 route validation gives one fixed, sufficient trip
    # count for all legal dynamic shapes; predicates suppress iterations after
    # each lower bound converges.
    lower_bound_steps = _MAX_SIGNED_I32.bit_length()

    @flyc.kernel(known_block_size=[_BLOCK_THREADS, 1, 1])
    def segmented_reduce_kernel(
        dx_route_order: fx.Tensor,
        token_indices: fx.Tensor,
        dx: fx.Tensor,
        i32_tokens: fx.Int32,
        i32_routes: fx.Int32,
    ):
        tid = gpu.thread_idx.x
        route_dx_rsrc = buffer_ops.create_buffer_resource(
            dx_route_order,
            max_size=True,
        )
        token_rsrc = buffer_ops.create_buffer_resource(token_indices, max_size=True)
        dx_rsrc = buffer_ops.create_buffer_resource(dx, max_size=True)
        lds = fx.SharedAllocator().allocate(_FlatDxSegmentSharedStorage).peek()
        bounds = lds.bounds.view(fx.make_layout(2, 1))

        def lower_bound(target):
            lo = fx.Int32(0)
            hi = i32_routes
            for _ in range_constexpr(lower_bound_steps):
                searching = lo < hi
                mid = (lo + hi) // fx.Int32(2)
                safe_mid = searching.select(mid, fx.Int32(0))
                candidate = fx.Int32(
                    buffer_ops.buffer_load(
                        token_rsrc,
                        safe_mid,
                        vec_width=1,
                        dtype=T.i32,
                    )
                )
                move_right = searching & (candidate < target)
                lo = move_right.select(mid + fx.Int32(1), lo)
                move_left = searching & (candidate >= target)
                hi = move_left.select(mid, hi)
            return lo

        token = gpu.block_idx.x
        token_stride = gpu.grid_dim.x
        for token_row in range(token, i32_tokens, token_stride):
            if tid == fx.Int32(0):
                start = lower_bound(token_row)
                end = lower_bound(token_row + fx.Int32(1))
                fx.memref_store(start, bounds, fx.Int32(0))
                fx.memref_store(end, bounds, fx.Int32(1))
            gpu.barrier()

            start = fx.Int32(fx.memref_load(bounds, fx.Int32(0)))
            end = fx.Int32(fx.memref_load(bounds, fx.Int32(1)))
            for vector_iter in range_constexpr(vectors_per_thread):
                dword_column = tid * fx.Int32(4) + fx.Int32(
                    vector_iter * _BLOCK_THREADS * 4
                )
                accum = fx.Vector.filled(8, 0.0, fx.Float32)
                for route_row in range(start, end, fx.Int32(1)):
                    route_dword = route_row * fx.Int32(dwords_per_row) + dword_column
                    packed = buffer_ops.buffer_load(
                        route_dx_rsrc,
                        route_dword,
                        vec_width=4,
                        dtype=T.i32,
                    )
                    accum = accum + fx.Vector(packed).bitcast(fx.BFloat16).to(
                        fx.Float32
                    )

                output_dword = token_row * fx.Int32(dwords_per_row) + dword_column
                buffer_ops.buffer_store(
                    accum.to(fx.BFloat16).bitcast(fx.Int32),
                    dx_rsrc,
                    output_dword,
                )
            # All lanes must finish consuming the current shared bounds before
            # lane zero publishes the next grid-stride token's segment.
            gpu.barrier()

    @flyc.jit
    def launch(
        dx_route_order: fx.Tensor,
        token_indices: fx.Tensor,
        dx: fx.Tensor,
        i32_tokens: fx.Int32,
        i32_routes: fx.Int32,
        i32_grid: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        segmented_reduce_kernel(
            dx_route_order,
            token_indices,
            dx,
            i32_tokens,
            i32_routes,
        ).launch(
            grid=(i32_grid, 1, 1),
            block=(_BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    return launch


@functools.lru_cache(maxsize=128)
def _compile_unsort(
    hidden_size: int,
    topk: int,
    compute_dtype: str,
    device_index: int,
    device_padded_rows: bool = False,
):
    """Compile sorted expert-row to dense ``[tokens, topk, H]`` scatter."""

    del device_index
    elem_dtype = fx.Float16 if compute_dtype == "fp16" else fx.BFloat16
    vector_width = 4
    vectors_per_row = hidden_size // vector_width

    @flyc.kernel(known_block_size=[_BLOCK_THREADS, 1, 1])
    def unsort_kernel(
        dx_sorted: fx.Tensor,
        sorted_token_ids: fx.Tensor,
        dx_routes: fx.Tensor,
        num_valid_ids: fx.Tensor,
        i32_tokens: fx.Int32,
        i32_padded_rows: fx.Int32,
    ):
        index = gpu.block_idx.x * fx.Int32(_BLOCK_THREADS) + gpu.thread_idx.x
        stride = gpu.grid_dim.x * fx.Int32(_BLOCK_THREADS)
        ids_rsrc = buffer_ops.create_buffer_resource(sorted_token_ids, max_size=True)
        source_rsrc = buffer_ops.create_buffer_resource(dx_sorted, max_size=True)
        destination_rsrc = buffer_ops.create_buffer_resource(dx_routes, max_size=True)

        def unsort_vector(index):
            row = index // fx.Int32(vectors_per_row)
            column = (index % fx.Int32(vectors_per_row)) * fx.Int32(vector_width)
            packed = fx.Int32(
                buffer_ops.buffer_load(ids_rsrc, row, vec_width=1, dtype=T.i32)
            )
            token = packed & fx.Int32(_TOKEN_MASK)
            slot = packed >> fx.Int32(24)
            valid = token < i32_tokens
            if valid:
                source = row * fx.Int32(hidden_size) + column
                destination = (
                    (token * fx.Int32(topk) + slot) * fx.Int32(hidden_size)
                ) + column
                value = buffer_ops.buffer_load(
                    source_rsrc,
                    source,
                    vec_width=vector_width,
                    dtype=elem_dtype,
                )
                buffer_ops.buffer_store(value, destination_rsrc, destination)

        padded_rows = i32_padded_rows
        if const_expr(device_padded_rows):
            valid_rsrc = buffer_ops.create_buffer_resource(num_valid_ids, max_size=True)
            padded_rows = fx.Int32(
                buffer_ops.buffer_load(
                    valid_rsrc, fx.Int32(0), vec_width=1, dtype=T.i32
                )
            )
        total = padded_rows * fx.Int32(vectors_per_row)
        if const_expr(device_padded_rows):
            for vector_index in range(index, total, stride):
                unsort_vector(fx.Int32(vector_index))
        else:
            if index < total:
                unsort_vector(index)

    @flyc.jit
    def launch(
        dx_sorted: fx.Tensor,
        sorted_token_ids: fx.Tensor,
        dx_routes: fx.Tensor,
        num_valid_ids: fx.Tensor,
        i32_tokens: fx.Int32,
        i32_padded_rows: fx.Int32,
        i32_grid: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        unsort_kernel(
            dx_sorted,
            sorted_token_ids,
            dx_routes,
            num_valid_ids,
            i32_tokens,
            i32_padded_rows,
        ).launch(
            grid=(i32_grid, 1, 1),
            block=(_BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    return launch


def _validate_backward_inputs(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    grad_output: torch.Tensor,
    config: "SonicMoEConfig",
    b1: torch.Tensor | None,
    b2: torch.Tensor | None,
    interleaved_w1: bool,
) -> tuple[int, int, int, int]:
    dtype_by_name = {"bf16": torch.bfloat16, "fp16": torch.float16}
    if config.compute_dtype not in dtype_by_name:
        raise ValueError(
            "sonic_moe_backward supports compute_dtype='bf16' or 'fp16', "
            f"got {config.compute_dtype!r}"
        )
    expected_dtype = dtype_by_name[config.compute_dtype]
    if config.activation not in _SUPPORTED_ACTIVATIONS:
        raise ValueError(
            f"sonic_moe_backward does not support activation={config.activation!r}; "
            f"expected one of {sorted(_SUPPORTED_ACTIVATIONS)}"
        )
    if not isinstance(interleaved_w1, bool):
        raise TypeError(
            f"interleaved_w1 must be bool, got {type(interleaved_w1).__name__}"
        )
    if interleaved_w1 and config.activation not in _GLU_ACTIVATIONS:
        raise ValueError(
            "interleaved_w1 is valid only for GLU activations "
            f"{sorted(_GLU_ACTIVATIONS)}, got activation={config.activation!r}"
        )

    tokens = int(hidden_states.shape[0]) if hidden_states.ndim == 2 else -1
    hidden_size = int(config.hidden_size)
    intermediate_size = int(config.intermediate_size)
    num_experts = int(config.num_experts)
    topk = int(config.top_k)
    projection_size = intermediate_size * (
        2 if config.activation in _GLU_ACTIVATIONS else 1
    )
    if (b1 is None) != (b2 is None):
        raise ValueError("b1 and b2 must both be None or both be tensors")
    expected = {
        "hidden_states": (tokens, hidden_size),
        "w1": (num_experts, projection_size, hidden_size),
        "w2": (num_experts, hidden_size, intermediate_size),
        "topk_ids": (tokens, topk),
        "topk_weights": (tokens, topk),
        "grad_output": (tokens, hidden_size),
    }
    tensors = {
        "hidden_states": hidden_states,
        "w1": w1,
        "w2": w2,
        "topk_ids": topk_ids,
        "topk_weights": topk_weights,
        "grad_output": grad_output,
    }
    if b1 is not None:
        expected["b1"] = (num_experts, projection_size)
        expected["b2"] = (num_experts, hidden_size)
        tensors["b1"] = b1
        tensors["b2"] = b2
    if tokens <= 0:
        raise ValueError(
            f"hidden_states must be non-empty 2D, got shape {tuple(hidden_states.shape)}"
        )
    if tokens > _TOKEN_MASK:
        raise ValueError(
            f"token count must fit the sorter's 24-bit token field, got {tokens}"
        )
    max_padded, _ = _max_padded_routes(tokens, num_experts, topk, _BACKWARD_SORT_UNIT)
    if max_padded > _MAX_SIGNED_I32:
        raise ValueError(
            f"padded route count exceeds the signed 32-bit limit, got {max_padded}"
        )
    if max_padded * max(hidden_size, projection_size) * 2 > _MAX_BUFFER_BYTE_OFFSET:
        raise ValueError(
            "fixed-K backward workspace exceeds the 32-bit buffer offset limit"
        )
    mesh_stride = (
        (tokens + _BACKWARD_SORT_UNIT - 1) // _BACKWARD_SORT_UNIT
    ) * _BACKWARD_SORT_UNIT
    if num_experts * mesh_stride > _MAX_SIGNED_I32:
        raise ValueError(
            "fixed-K backward sorting mesh exceeds the signed 32-bit index limit"
        )
    for name, tensor in tensors.items():
        if tuple(tensor.shape) != expected[name]:
            raise ValueError(
                f"{name} must have shape {expected[name]}, got {tuple(tensor.shape)}"
            )
        if not tensor.is_cuda or tensor.device != hidden_states.device:
            raise ValueError(f"{name} must be on the same ROCm device as hidden_states")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
    floating_names = ["hidden_states", "w1", "w2", "grad_output"]
    if b1 is not None:
        floating_names.extend(("b1", "b2"))
    for name in floating_names:
        if tensors[name].dtype != expected_dtype:
            raise TypeError(
                f"{name} must be {expected_dtype}, got {tensors[name].dtype}"
            )
    if topk_ids.dtype != torch.int32:
        raise TypeError(f"topk_ids must be int32, got {topk_ids.dtype}")
    if topk_weights.dtype != torch.float32:
        raise TypeError(f"topk_weights must be float32, got {topk_weights.dtype}")
    if hidden_size % 64 != 0 or intermediate_size % 64 != 0:
        raise ValueError("hidden_size and intermediate_size must be multiples of 64")
    return tokens, hidden_size, intermediate_size, num_experts


def _validate_forward_state(
    forward_state: object,
    hidden_states: torch.Tensor,
    config: "SonicMoEConfig",
    interleaved_w1: bool,
    has_bias: bool,
) -> tuple[torch.Tensor, int, torch.cuda.Event]:
    """Validate the fixed-K training-forward state consumed by backward.

    This deliberately uses a structural contract instead of importing the
    forward state's concrete class.  Forward and backward can therefore
    evolve independently while structurally malformed or incompatible state
    is rejected instead of silently falling back to W1 recomputation.
    """

    field_names = (
        "preactivation",
        "tokens",
        "hidden_size",
        "intermediate_size",
        "num_experts",
        "top_k",
        "activation",
        "compute_dtype",
        "interleaved_w1",
        "has_bias",
        "producer_stream",
        "ready_event",
    )
    values: dict[str, object] = {}
    missing: list[str] = []
    for name in field_names:
        try:
            values[name] = getattr(forward_state, name)
        except AttributeError:
            missing.append(name)
    if missing:
        raise ValueError(
            "forward_state is missing required field(s): " + ", ".join(missing)
        )

    tokens = int(hidden_states.shape[0])
    expected_ints = {
        "tokens": tokens,
        "hidden_size": int(config.hidden_size),
        "intermediate_size": int(config.intermediate_size),
        "num_experts": int(config.num_experts),
        "top_k": int(config.top_k),
    }
    for name, expected in expected_ints.items():
        value = values[name]
        if type(value) is not int:
            raise TypeError(
                f"forward_state.{name} must be int, got {type(value).__name__}"
            )
        if value != expected:
            raise ValueError(f"forward_state.{name} must equal {expected}, got {value}")

    expected_strings = {
        "activation": str(config.activation),
        "compute_dtype": str(config.compute_dtype),
    }
    for name, expected in expected_strings.items():
        value = values[name]
        if type(value) is not str:
            raise TypeError(
                f"forward_state.{name} must be str, got {type(value).__name__}"
            )
        if value != expected:
            raise ValueError(
                f"forward_state.{name} must equal {expected!r}, got {value!r}"
            )
    if config.activation != "swiglu" or config.compute_dtype != "bf16":
        raise ValueError(
            "forward_state reuse currently supports only dense BF16 SwiGLU fixed-K backward"
        )

    state_interleaved = values["interleaved_w1"]
    if type(state_interleaved) is not bool:
        raise TypeError(
            "forward_state.interleaved_w1 must be bool, got "
            f"{type(state_interleaved).__name__}"
        )
    if state_interleaved != interleaved_w1:
        raise ValueError(
            "forward_state.interleaved_w1 must match the backward call, got "
            f"{state_interleaved} and {interleaved_w1}"
        )
    state_has_bias = values["has_bias"]
    if type(state_has_bias) is not bool:
        raise TypeError(
            "forward_state.has_bias must be bool, got "
            f"{type(state_has_bias).__name__}"
        )
    if state_has_bias != has_bias:
        raise ValueError(
            "forward_state.has_bias must match the backward call, got "
            f"{state_has_bias} and {has_bias}"
        )

    preactivation = values["preactivation"]
    if not isinstance(preactivation, torch.Tensor):
        raise TypeError(
            "forward_state.preactivation must be a torch.Tensor, got "
            f"{type(preactivation).__name__}"
        )
    expected_shape = (
        tokens,
        int(config.top_k),
        2 * int(config.intermediate_size),
    )
    if tuple(preactivation.shape) != expected_shape:
        raise ValueError(
            f"forward_state.preactivation must have shape {expected_shape}, "
            f"got {tuple(preactivation.shape)}"
        )
    if not preactivation.is_cuda or preactivation.device != hidden_states.device:
        raise ValueError(
            "forward_state.preactivation must be on the same ROCm device as hidden_states"
        )
    if preactivation.dtype != torch.bfloat16:
        raise TypeError(
            "forward_state.preactivation must be torch.bfloat16, got "
            f"{preactivation.dtype}"
        )
    preactivation_bytes = preactivation.numel() * preactivation.element_size()
    if preactivation_bytes > _MAX_SIGNED_I32:
        raise ValueError(
            "forward_state.preactivation byte span exceeds the signed 32-bit "
            f"buffer limit, got {preactivation_bytes}"
        )
    if not preactivation.is_contiguous():
        raise ValueError("forward_state.preactivation must be contiguous")

    producer_stream = values["producer_stream"]
    if type(producer_stream) is not int:
        raise TypeError(
            "forward_state.producer_stream must be int, got "
            f"{type(producer_stream).__name__}"
        )
    if producer_stream < 0:
        raise ValueError(
            f"forward_state.producer_stream must be non-negative, got {producer_stream}"
        )

    ready_event = values["ready_event"]
    if not isinstance(ready_event, torch.cuda.Event):
        raise TypeError(
            "forward_state.ready_event must be torch.cuda.Event, got "
            f"{type(ready_event).__name__}"
        )
    event_device = ready_event.device
    if event_device is None:
        raise ValueError("forward_state.ready_event must already be recorded")
    if torch.device(event_device) != hidden_states.device:
        raise ValueError(
            "forward_state.ready_event must be recorded on the same ROCm device "
            "as hidden_states"
        )

    return preactivation, producer_stream, ready_event


def _validate_routes_forward_state(
    forward_state: object,
    hidden_states: torch.Tensor,
    routes: int,
    config: "SonicMoEConfig",
    interleaved_w1: bool,
    has_bias: bool,
) -> tuple[
    torch.Tensor,
    int,
    torch.cuda.Event,
    _RoutesSorterMetadata | None,
    bool,
    int | None,
]:
    """Validate retained route-order state for dynamic E16 flat routing."""

    field_names = (
        "preactivation",
        "tokens",
        "routes",
        "hidden_size",
        "intermediate_size",
        "num_experts",
        "activation",
        "compute_dtype",
        "interleaved_w1",
        "has_bias",
        "producer_stream",
        "ready_event",
    )
    values: dict[str, object] = {}
    missing: list[str] = []
    for name in field_names:
        try:
            values[name] = getattr(forward_state, name)
        except AttributeError:
            missing.append(name)
    if missing:
        raise ValueError(
            "forward_state is missing required field(s): " + ", ".join(missing)
        )

    tokens = int(hidden_states.shape[0])
    expected_ints = {
        "tokens": tokens,
        "routes": routes,
        "hidden_size": int(config.hidden_size),
        "intermediate_size": int(config.intermediate_size),
        "num_experts": int(config.num_experts),
    }
    for name, expected in expected_ints.items():
        value = values[name]
        if type(value) is not int:
            raise TypeError(
                f"forward_state.{name} must be int, got {type(value).__name__}"
            )
        if value != expected:
            raise ValueError(f"forward_state.{name} must equal {expected}, got {value}")

    expected_strings = {
        "activation": str(config.activation),
        "compute_dtype": str(config.compute_dtype),
    }
    for name, expected in expected_strings.items():
        value = values[name]
        if type(value) is not str:
            raise TypeError(
                f"forward_state.{name} must be str, got {type(value).__name__}"
            )
        if value != expected:
            raise ValueError(
                f"forward_state.{name} must equal {expected!r}, got {value!r}"
            )

    if (
        tokens != routes
        or routes < 0
        or (
            int(config.hidden_size),
            int(config.intermediate_size),
            int(config.num_experts),
        )
        != _E16_FLAT_GROUPED_SHAPE
    ):
        raise ValueError(
            "flat forward_state reuse requires dynamic T==R with " "H2048/I768/E16"
        )
    if config.activation != "swiglu" or config.compute_dtype != "bf16" or has_bias:
        raise ValueError(
            "flat forward_state reuse currently supports only bias-free BF16 "
            "SwiGLU backward"
        )

    state_interleaved = values["interleaved_w1"]
    if type(state_interleaved) is not bool:
        raise TypeError(
            "forward_state.interleaved_w1 must be bool, got "
            f"{type(state_interleaved).__name__}"
        )
    if state_interleaved != interleaved_w1:
        raise ValueError(
            "forward_state.interleaved_w1 must match the backward call, got "
            f"{state_interleaved} and {interleaved_w1}"
        )
    state_has_bias = values["has_bias"]
    if type(state_has_bias) is not bool:
        raise TypeError(
            "forward_state.has_bias must be bool, got "
            f"{type(state_has_bias).__name__}"
        )
    if state_has_bias != has_bias:
        raise ValueError(
            "forward_state.has_bias must match the backward call, got "
            f"{state_has_bias} and {has_bias}"
        )

    preactivation = values["preactivation"]
    if not isinstance(preactivation, torch.Tensor):
        raise TypeError(
            "forward_state.preactivation must be a torch.Tensor, got "
            f"{type(preactivation).__name__}"
        )
    expected_shape = (routes, 2 * int(config.intermediate_size))
    if tuple(preactivation.shape) != expected_shape:
        raise ValueError(
            f"forward_state.preactivation must have shape {expected_shape}, "
            f"got {tuple(preactivation.shape)}"
        )
    if not preactivation.is_cuda or preactivation.device != hidden_states.device:
        raise ValueError(
            "forward_state.preactivation must be on the same ROCm device as "
            "hidden_states"
        )
    if preactivation.dtype != torch.bfloat16:
        raise TypeError(
            "forward_state.preactivation must be torch.bfloat16, got "
            f"{preactivation.dtype}"
        )
    preactivation_bytes = preactivation.numel() * preactivation.element_size()
    if preactivation_bytes > _MAX_SIGNED_I32:
        raise ValueError(
            "forward_state.preactivation byte span exceeds the signed 32-bit "
            f"buffer limit, got {preactivation_bytes}"
        )
    if not preactivation.is_contiguous():
        raise ValueError("forward_state.preactivation must be contiguous")

    producer_stream = values["producer_stream"]
    if type(producer_stream) is not int:
        raise TypeError(
            "forward_state.producer_stream must be int, got "
            f"{type(producer_stream).__name__}"
        )
    if producer_stream < 0:
        raise ValueError(
            f"forward_state.producer_stream must be non-negative, got {producer_stream}"
        )

    ready_event = values["ready_event"]
    if not isinstance(ready_event, torch.cuda.Event):
        raise TypeError(
            "forward_state.ready_event must be torch.cuda.Event, got "
            f"{type(ready_event).__name__}"
        )
    event_device = ready_event.device
    if event_device is None:
        raise ValueError("forward_state.ready_event must already be recorded")
    if torch.device(event_device) != hidden_states.device:
        raise ValueError(
            "forward_state.ready_event must be recorded on the same ROCm device "
            "as hidden_states"
        )

    # Sorter metadata is an optional optimization ABI.  Older state producers
    # carry only preactivation; partially populated or structurally invalid
    # metadata must therefore fall back to a fresh backward sort rather than
    # weakening the existing state compatibility contract.
    metadata_names = (
        "sorted_token_ids",
        "sorted_route_ids",
        "sorted_weights",
        "sorted_expert_ids",
        "num_valid_ids",
        "expert_frequency",
    )
    metadata_values = tuple(
        getattr(forward_state, name, None) for name in metadata_names
    )
    sorter_metadata: _RoutesSorterMetadata | None = None
    if all(isinstance(value, torch.Tensor) for value in metadata_values):
        sort_unit = int(config.route_tile_m)
        max_padded, max_blocks = _max_padded_flat_routes(
            routes,
            int(config.num_experts),
            sort_unit,
        )
        expected_metadata = (
            ((max_padded,), torch.int32),
            ((max_padded,), torch.int32),
            ((max_padded,), torch.float32),
            ((max_blocks,), torch.int32),
            ((2,), torch.int32),
            ((int(config.num_experts),), torch.int32),
        )
        metadata_ok = True
        for value, (shape, dtype) in zip(metadata_values, expected_metadata):
            assert isinstance(value, torch.Tensor)
            metadata_ok = metadata_ok and (
                tuple(value.shape) == shape
                and value.dtype == dtype
                and value.is_cuda
                and value.device == hidden_states.device
                and value.is_contiguous()
            )
        # Invocation-owned fields must not be views into the saved activation
        # or into one another.  This also makes accidental state construction
        # from a reusable packed workspace fail closed to the normal sorter.
        if metadata_ok:
            storage_ptrs = [
                value.untyped_storage().data_ptr()
                for value in metadata_values
                if isinstance(value, torch.Tensor)
            ]
            metadata_ok = (
                preactivation.untyped_storage().data_ptr() not in storage_ptrs
                and len(set(storage_ptrs)) == len(storage_ptrs)
            )
        if metadata_ok:
            (
                sorted_token_ids,
                sorted_route_ids,
                sorted_weights,
                sorted_expert_ids,
                num_valid_ids,
                expert_frequency,
            ) = metadata_values
            assert isinstance(sorted_token_ids, torch.Tensor)
            assert isinstance(sorted_route_ids, torch.Tensor)
            assert isinstance(sorted_weights, torch.Tensor)
            assert isinstance(sorted_expert_ids, torch.Tensor)
            assert isinstance(num_valid_ids, torch.Tensor)
            assert isinstance(expert_frequency, torch.Tensor)
            sorter_metadata = (
                sorted_token_ids,
                sorted_route_ids,
                sorted_weights,
                sorted_expert_ids,
                num_valid_ids,
                expert_frequency,
                sort_unit,
            )

    token_indices_identity = getattr(
        forward_state,
        "token_indices_identity",
        False,
    )
    if type(token_indices_identity) is not bool:
        raise TypeError(
            "forward_state.token_indices_identity must be bool, got "
            f"{type(token_indices_identity).__name__}"
        )

    # New state producers carry a rank-shared performance-policy size.  It is
    # deliberately optional so states produced before this field existed keep
    # selecting from their local runtime R.  Validation is delegated to the
    # common finite-policy mapper; the value is never an allocation bound.
    route_policy_size = getattr(forward_state, "route_policy_size", None)
    select_e16_route_policy(routes, route_policy_size)

    return (
        preactivation,
        producer_stream,
        ready_event,
        sorter_metadata,
        token_indices_identity,
        route_policy_size,
    )


def _validate_backward_route_inputs(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    token_indices: torch.Tensor | None,
    expert_indices: torch.Tensor | None,
    route_weights: torch.Tensor,
    grad_output: torch.Tensor,
    config: "SonicMoEConfig",
    b1: torch.Tensor | None,
    b2: torch.Tensor | None,
    interleaved_w1: bool,
    *,
    implicit_identity_routes: bool = False,
) -> tuple[int, int, int, int, int]:
    dtype_by_name = {"bf16": torch.bfloat16, "fp16": torch.float16}
    if config.compute_dtype not in dtype_by_name:
        raise ValueError(
            "sonic_moe_backward_routes supports compute_dtype='bf16' or 'fp16', "
            f"got {config.compute_dtype!r}"
        )
    expected_dtype = dtype_by_name[config.compute_dtype]
    if config.activation not in _SUPPORTED_ACTIVATIONS:
        raise ValueError(
            f"sonic_moe_backward_routes does not support activation={config.activation!r}; "
            f"expected one of {sorted(_SUPPORTED_ACTIVATIONS)}"
        )
    if not isinstance(interleaved_w1, bool):
        raise TypeError(
            f"interleaved_w1 must be bool, got {type(interleaved_w1).__name__}"
        )
    if interleaved_w1 and config.activation not in _GLU_ACTIVATIONS:
        raise ValueError(
            "interleaved_w1 is valid only for GLU activations "
            f"{sorted(_GLU_ACTIVATIONS)}, got activation={config.activation!r}"
        )
    if not isinstance(implicit_identity_routes, bool):
        raise TypeError("implicit_identity_routes must be bool")
    if implicit_identity_routes:
        if token_indices is not None or expert_indices is not None:
            raise ValueError(
                "implicit identity routes must not supply token_indices or "
                "expert_indices"
            )
        if route_weights.ndim != 1:
            raise ValueError("route_weights must be one-dimensional")
    else:
        if token_indices is None or expert_indices is None:
            raise ValueError(
                "explicit flat routes require token_indices and expert_indices"
            )
        if (
            token_indices.ndim != 1
            or expert_indices.ndim != 1
            or route_weights.ndim != 1
        ):
            raise ValueError(
                "token_indices, expert_indices, and route_weights must be "
                "one-dimensional"
            )

    tokens = int(hidden_states.shape[0]) if hidden_states.ndim == 2 else -1
    routes = int(route_weights.numel())
    hidden_size = int(config.hidden_size)
    intermediate_size = int(config.intermediate_size)
    num_experts = int(config.num_experts)
    projection_size = intermediate_size * (
        2 if config.activation in _GLU_ACTIVATIONS else 1
    )
    if (b1 is None) != (b2 is None):
        raise ValueError("b1 and b2 must both be None or both be tensors")
    expected = {
        "hidden_states": (tokens, hidden_size),
        "w1": (num_experts, projection_size, hidden_size),
        "w2": (num_experts, hidden_size, intermediate_size),
        "route_weights": (routes,),
        "grad_output": (tokens, hidden_size),
    }
    tensors = {
        "hidden_states": hidden_states,
        "w1": w1,
        "w2": w2,
        "route_weights": route_weights,
        "grad_output": grad_output,
    }
    if not implicit_identity_routes:
        assert token_indices is not None
        assert expert_indices is not None
        expected["token_indices"] = (routes,)
        expected["expert_indices"] = (routes,)
        tensors["token_indices"] = token_indices
        tensors["expert_indices"] = expert_indices
    if b1 is not None:
        expected["b1"] = (num_experts, projection_size)
        expected["b2"] = (num_experts, hidden_size)
        tensors["b1"] = b1
        tensors["b2"] = b2
    if tokens < 0 or (tokens == 0 and routes != 0):
        raise ValueError(
            "hidden_states must be a 2D tensor with at least one row when "
            f"routes are non-empty, got shape {tuple(hidden_states.shape)} "
            f"and routes={routes}"
        )
    if tokens > _TOKEN_MASK:
        raise ValueError(
            f"token count must fit the sorter's 24-bit token field, got {tokens}"
        )
    if routes > _MAX_SIGNED_I32:
        raise ValueError(f"route count exceeds the signed 32-bit limit, got {routes}")
    max_padded, _ = _max_padded_flat_routes(routes, num_experts, _BACKWARD_SORT_UNIT)
    if max_padded > _MAX_SIGNED_I32:
        raise ValueError(
            f"padded route count exceeds the signed 32-bit limit, got {max_padded}"
        )
    if max_padded * max(hidden_size, projection_size) * 2 > _MAX_BUFFER_BYTE_OFFSET:
        raise ValueError(
            "ragged backward workspace exceeds the 32-bit buffer offset limit"
        )
    _validate_backward_dx_extent(
        tokens,
        hidden_size,
        requires_fp32_accum=False,
    )
    for name, tensor in tensors.items():
        if tuple(tensor.shape) != expected[name]:
            raise ValueError(
                f"{name} must have shape {expected[name]}, got {tuple(tensor.shape)}"
            )
        if not tensor.is_cuda or tensor.device != hidden_states.device:
            raise ValueError(f"{name} must be on the same ROCm device as hidden_states")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
    floating_names = ["hidden_states", "w1", "w2", "grad_output"]
    if b1 is not None:
        floating_names.extend(("b1", "b2"))
    for name in floating_names:
        if tensors[name].dtype != expected_dtype:
            raise TypeError(
                f"{name} must be {expected_dtype}, got {tensors[name].dtype}"
            )
    if not implicit_identity_routes:
        assert token_indices is not None
        assert expert_indices is not None
        if token_indices.dtype != torch.int32 or expert_indices.dtype != torch.int32:
            raise TypeError(
                "token_indices and expert_indices must be int32, got "
                f"{token_indices.dtype}/{expert_indices.dtype}"
            )
    if route_weights.dtype != torch.float32:
        raise TypeError(f"route_weights must be float32, got {route_weights.dtype}")
    if hidden_size % 64 != 0 or intermediate_size % 64 != 0:
        raise ValueError("hidden_size and intermediate_size must be multiples of 64")
    return tokens, hidden_size, intermediate_size, num_experts, routes


def _validate_backward_dx_extent(
    tokens: int,
    hidden_size: int,
    *,
    requires_fp32_accum: bool,
) -> None:
    """Validate only the dX buffers that a selected route path allocates.

    Every backward path owns a 16-bit token-major ``dx`` tensor.  Only the
    generic flat-route scatter additionally owns an FP32 accumulation tensor;
    retained identity and sorted segmented routes write/reduce in route order
    and must not inherit that stricter, unused allocation limit.
    """

    elements = tokens * hidden_size
    if elements > _MAX_SIGNED_I32 or elements * 2 > _MAX_BUFFER_BYTE_OFFSET:
        raise ValueError(
            "ragged backward 16-bit input-gradient buffer exceeds the "
            "32-bit buffer offset limit"
        )
    if requires_fp32_accum and elements * 4 > _MAX_BUFFER_BYTE_OFFSET:
        raise ValueError(
            "ragged backward FP32 input-gradient workspace exceeds the "
            "32-bit buffer offset limit"
        )


def _ptr(tensor: torch.Tensor):
    return flyc.from_c_void_p(fx.Uint8, tensor.data_ptr())


def _is_current_stream_capturing_conservatively() -> bool:
    """Disable auxiliary launches when capture state cannot be proven safe."""

    try:
        return bool(torch.cuda.is_current_stream_capturing())
    except RuntimeError:
        return True


@functools.cache
def _e16_dx_auxiliary_stream(device_index: int) -> torch.cuda.Stream:
    """Return the process-local high-priority stream for large E16 dX."""

    with torch.cuda.device(device_index):
        return torch.cuda.Stream(device=device_index, priority=-1)


def _record_optional_tensors_on_stream(
    stream: torch.cuda.Stream,
    *tensors: torch.Tensor | None,
) -> None:
    """Extend allocator ownership for tensors consumed by an auxiliary stream."""

    for tensor in tensors:
        if tensor is not None:
            tensor.record_stream(stream)


def _launch_overlapped_e16_dw1_dx(
    launch_dw1: Callable[[torch.cuda.Stream], None],
    launch_dx: Callable[[torch.cuda.Stream], None],
    auxiliary_tensors: tuple[torch.Tensor | None, ...],
    *,
    stream: torch.cuda.Stream,
    device_index: int,
) -> None:
    """Fork dX to an auxiliary stream, run dW1, then join on ``stream``."""

    auxiliary_stream = _e16_dx_auxiliary_stream(device_index)
    fork_event = torch.cuda.Event()
    done_event = torch.cuda.Event()
    fork_event.record(stream)
    auxiliary_stream.wait_event(fork_event)
    _record_optional_tensors_on_stream(auxiliary_stream, *auxiliary_tensors)
    done_recorded = False
    try:
        launch_dx(auxiliary_stream)
        done_event.record(auxiliary_stream)
        done_recorded = True
        launch_dw1(stream)
    finally:
        if done_recorded:
            stream.wait_event(done_event)
        else:
            # If host-side launch validation raised after enqueuing any dX
            # work, still restore current-stream completion semantics before
            # propagating the error.
            stream.wait_stream(auxiliary_stream)


def _sonic_moe_backward_impl(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    expert_ids: torch.Tensor | None,
    route_weights: torch.Tensor,
    grad_output: torch.Tensor,
    config: "SonicMoEConfig",
    *,
    token_indices: torch.Tensor | None,
    dimensions: tuple[int, int, int, int],
    b1: torch.Tensor | None = None,
    b2: torch.Tensor | None = None,
    interleaved_w1: bool = False,
    forward_state_data: tuple[torch.Tensor, int, torch.cuda.Event] | None = None,
    forward_sorter_metadata: _RoutesSorterMetadata | None = None,
    token_indices_sorted: bool = False,
    token_indices_identity: bool = False,
    route_policy_size: int | None = None,
    implicit_identity_routes: bool = False,
) -> tuple[torch.Tensor, ...]:
    """Shared sorted-expert implementation for fixed-K and flat routes."""

    tokens, hidden_size, intermediate_size, num_experts = dimensions
    if implicit_identity_routes:
        if token_indices is not None or expert_ids is not None:
            raise ValueError(
                "implicit identity backward must not receive materialized ids"
            )
        if not token_indices_identity or forward_sorter_metadata is None:
            raise ValueError(
                "implicit identity backward requires identity routing and "
                "retained sorter metadata"
            )
    flat_routes = token_indices is not None or implicit_identity_routes
    routes = int(route_weights.numel())
    route_policy = select_e16_route_policy(routes, route_policy_size)
    topk = int(config.top_k)
    compute_dtype = str(config.compute_dtype)
    activation_name = str(config.activation)
    # Forward exact-E16 metadata uses the forward route tile (BM128 for the
    # tuned Qwen3 configuration).  A fallback backward sort keeps the original
    # BM64 layout.  Never reinterpret metadata produced at one granularity as
    # the other: ``sorted_expert_ids`` is indexed in sorter-block units.
    sort_unit = (
        forward_sorter_metadata[-1]
        if forward_sorter_metadata is not None
        else _BACKWARD_SORT_UNIT
    )
    projection_size = intermediate_size * (
        2 if activation_name in _GLU_ACTIVATIONS else 1
    )
    has_bias = b1 is not None
    reuse_forward_preactivation = forward_state_data is not None
    use_e16_fixed_state_grouped = _use_e16_fixed_state_grouped_backward(
        compute_dtype=compute_dtype,
        activation=activation_name,
        tokens=tokens,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        topk=topk,
        flat_routes=flat_routes,
        has_bias=has_bias,
        reuse_forward_preactivation=reuse_forward_preactivation,
    )
    use_e16_flat_grouped = _use_e16_flat_grouped_backward(
        compute_dtype=compute_dtype,
        activation=activation_name,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        routes=routes,
        flat_routes=flat_routes,
        has_bias=has_bias,
        reuse_forward_preactivation=reuse_forward_preactivation,
    )
    use_flat_identity_dx = (
        use_e16_flat_grouped
        and reuse_forward_preactivation
        and token_indices_identity
        and tokens == routes
    )
    # The segmented route-order reducer uses binary search and therefore
    # requires nondecreasing token indices.  Preserve the generic atomic
    # reducer unless the caller explicitly supplies that ordering contract.
    use_flat_segmented_dx = _use_e16_flat_segmented_dx(
        e16_flat_grouped=use_e16_flat_grouped,
        reuse_forward_preactivation=reuse_forward_preactivation,
        token_indices_sorted=(token_indices_sorted and not use_flat_identity_dx),
    )
    use_flat_route_order_dx = use_flat_identity_dx or use_flat_segmented_dx
    if flat_routes and not use_flat_route_order_dx:
        _validate_backward_dx_extent(
            tokens,
            hidden_size,
            requires_fp32_accum=True,
        )
    use_e16_deduplicated_metadata = _use_e16_flat_deduplicated_metadata(
        e16_flat_grouped=use_e16_flat_grouped,
        reuse_forward_preactivation=reuse_forward_preactivation,
    )
    use_e16_exact_queue = use_e16_fixed_state_grouped or use_e16_flat_grouped
    use_grouped_w1 = _use_grouped_w1_recompute(
        compute_dtype=compute_dtype,
        activation=activation_name,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        tokens=tokens,
        routes=routes,
        flat_routes=flat_routes,
    )
    use_grouped_w1 = use_grouped_w1 or use_e16_flat_grouped
    use_grouped_w2 = _use_grouped_w2_recompute(
        compute_dtype=compute_dtype,
        activation=activation_name,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        tokens=tokens,
        routes=routes,
        flat_routes=flat_routes,
    )
    use_grouped_w2 = use_grouped_w2 or use_e16_flat_grouped
    use_grouped_dw2 = _use_grouped_dw2(
        compute_dtype=compute_dtype,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
    )
    use_grouped_da = _use_grouped_da(
        compute_dtype=compute_dtype,
        activation=activation_name,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        tokens=tokens,
        routes=routes,
        flat_routes=flat_routes,
    )
    use_grouped_da = (
        use_grouped_da or use_e16_fixed_state_grouped or use_e16_flat_grouped
    )
    use_grouped_dw1 = _use_grouped_dw1(
        compute_dtype=compute_dtype,
        activation=activation_name,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        tokens=tokens,
        routes=routes,
        flat_routes=flat_routes,
    )
    use_grouped_dw1 = (
        use_grouped_dw1 or use_e16_fixed_state_grouped or use_e16_flat_grouped
    )
    grouped_w1_bm, grouped_w1_bn, _, _, compact_w1_grid = _grouped_w1_tuning(
        tokens=tokens,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
    )
    use_compact_w1 = use_grouped_w1 and compact_w1_grid
    use_grouped_dx = _use_grouped_dx(
        compute_dtype=compute_dtype,
        activation=activation_name,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        tokens=tokens,
        num_experts=num_experts,
        topk=topk,
        flat_routes=flat_routes,
        compact_w1=use_compact_w1,
    )
    use_grouped_dx = (
        use_grouped_dx or use_e16_fixed_state_grouped or use_e16_flat_grouped
    )
    direct_grouped_dx_routes = _use_direct_grouped_dx_routes(
        use_grouped_dx=use_grouped_dx,
        flat_routes=flat_routes,
    )
    use_large_grouped_dx = use_grouped_dx and (
        use_e16_fixed_state_grouped
        or _use_large_grouped_dx_descriptor_queue(
            tokens=tokens,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            topk=topk,
            flat_routes=flat_routes,
        )
    )
    use_hostless_grouped = _use_hostless_grouped_backward(
        flat_routes=flat_routes,
        has_bias=has_bias,
        tokens=tokens,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        topk=topk,
        reuse_forward_preactivation=reuse_forward_preactivation,
        use_large_grouped_dx=use_large_grouped_dx,
        use_grouped_w1=use_grouped_w1,
        use_grouped_w2=use_grouped_w2,
        use_grouped_dw2=use_grouped_dw2,
        use_grouped_da=use_grouped_da,
        use_grouped_dw1=use_grouped_dw1,
        use_grouped_dx=use_grouped_dx,
        e16_fixed_state_grouped=use_e16_fixed_state_grouped,
        e16_flat_grouped=use_e16_flat_grouped,
    )
    use_standalone_e16_tn_metadata_direct = _use_standalone_e16_tn_metadata_direct(
        e16_flat_grouped=use_e16_flat_grouped,
        reuse_forward_preactivation=reuse_forward_preactivation,
        use_hostless_grouped=use_hostless_grouped,
        use_grouped_dw1=use_grouped_dw1,
        use_grouped_dw2=use_grouped_dw2,
        sort_unit=sort_unit,
        routes=routes,
    )
    hostless_row_grid_cap = _hostless_row_grid_cap(
        use_e16_expert_major=(use_hostless_grouped and use_flat_identity_dx),
        route_policy=route_policy,
    )
    use_sorter_native_backward_metadata = _use_sorter_native_backward_metadata(
        flat_routes=flat_routes,
        has_bias=has_bias,
        tokens=tokens,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        topk=topk,
        reuse_forward_preactivation=reuse_forward_preactivation,
        use_hostless_grouped=use_hostless_grouped,
        use_large_grouped_dx=use_large_grouped_dx,
        use_compact_w1=use_compact_w1,
        sort_unit=sort_unit,
    )
    use_grouped_dx_m_reuse = _use_grouped_dx_m_reuse_schedule(
        compute_dtype=compute_dtype,
        activation=activation_name,
        tokens=tokens,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        topk=topk,
        flat_routes=flat_routes,
        has_bias=has_bias,
        reuse_forward_preactivation=reuse_forward_preactivation,
        use_large_grouped_dx=use_large_grouped_dx,
        use_hostless_grouped=use_hostless_grouped,
        direct_grouped_dx_routes=direct_grouped_dx_routes,
    )
    # Exact-row state preparation reuses either the short BM16 queue or the
    # production T4096 BM64 dX queue.  Decode has only ~18 us of gather/prepare
    # work and is faster on the original one-row kernels.
    use_fused_forward_state_prepare = (
        reuse_forward_preactivation
        and not has_bias
        and (
            (use_hostless_grouped and use_compact_w1)
            or use_large_grouped_dx
            or use_e16_flat_grouped
        )
    )
    use_direct_grouped_dw1_rhs = _use_direct_grouped_dw1_rhs(
        reuse_forward_preactivation=reuse_forward_preactivation,
        use_fused_forward_state_prepare=use_fused_forward_state_prepare,
        use_grouped_dw1=use_grouped_dw1,
        flat_routes=flat_routes,
        has_bias=has_bias,
        compute_dtype=compute_dtype,
        activation=activation_name,
        tokens=tokens,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        topk=topk,
        flat_identity_routes=use_flat_identity_dx,
    )
    use_fused_da_dscore = _use_fused_da_dscore(
        reuse_forward_preactivation=reuse_forward_preactivation,
        use_hostless_grouped=use_hostless_grouped,
        use_compact_w1=use_compact_w1,
        use_large_grouped_dx=use_large_grouped_dx,
        flat_routes=flat_routes,
        has_bias=has_bias,
        compute_dtype=compute_dtype,
        activation=activation_name,
        e16_flat_grouped=use_e16_flat_grouped,
    )
    # Expert-major identity routes already have the same row order as
    # grad_output.  Let grouped dA and the fused derivative read that tensor
    # directly instead of first copying it bit-for-bit into ``dy``.
    use_direct_grouped_da_rhs = (
        _E16_DIRECT_DA_RHS_ENABLED
        and use_flat_identity_dx
        and use_fused_forward_state_prepare
        and use_fused_da_dscore
        and use_e16_exact_queue
    )
    use_e16_hot_dw1_splitk = (
        _E16_DW1_SPLITK_ENABLED
        and use_e16_flat_grouped
        and reuse_forward_preactivation
        and token_indices_identity
        and forward_sorter_metadata is not None
        and use_direct_grouped_dw1_rhs
        and use_e16_deduplicated_metadata
        and route_policy >= E16RoutePolicy.LARGE
    )
    use_e16_hot_dw2_splitk = (
        _E16_DW2_SPLITK_ENABLED
        and use_e16_hot_dw1_splitk
        and route_policy == E16RoutePolicy.XLARGE
    )
    # Generic hostless shapes retain the latency-optimal eager memset for tiny
    # batches.  Dynamic retained E16 always uses the device-adaptive zeroer so
    # R=1/2 cannot change host launch topology or allocator initialization.
    hostless_dense_weight_zero = (
        use_hostless_grouped
        and not use_e16_flat_grouped
        and routes * _INACTIVE_WEIGHT_GRAD_ZERO_ACTIVE_RATIO < num_experts
    )
    device = hidden_states.device
    device_index = device.index or 0
    with torch.cuda.device(device):
        arch = get_rocm_arch()
    if not str(arch).startswith("gfx95"):
        raise RuntimeError(f"SonicMoE backward requires gfx95*, got {arch!r}")
    if flat_routes and routes == 0:
        result = (
            torch.zeros_like(hidden_states, memory_format=torch.contiguous_format),
            torch.zeros_like(w1, memory_format=torch.contiguous_format),
            torch.zeros_like(w2, memory_format=torch.contiguous_format),
            torch.empty_like(route_weights, memory_format=torch.contiguous_format),
        )
        if has_bias:
            return (
                *result,
                torch.zeros_like(b1, memory_format=torch.contiguous_format),
                torch.zeros_like(b2, memory_format=torch.contiguous_format),
            )
        return result
    if flat_routes:
        max_padded, max_blocks = _max_padded_flat_routes(routes, num_experts, sort_unit)
        workspace_elements = num_experts
    else:
        max_padded, max_blocks = _max_padded_routes(
            tokens, num_experts, topk, sort_unit
        )
        workspace_elements = moe_sorting_get_workspace_size(
            tokens,
            num_experts,
            topk,
            unit_size=sort_unit,
        )

    # The retained E16 route state owns its sorter output for the whole
    # autograd invocation.  Reuse those tensors directly; they cannot alias the
    # forward operator's LRU workspace.  Every other path preserves the
    # backward-owned allocation and sorting contract.
    if forward_sorter_metadata is not None:
        (
            sorted_token_ids,
            sorted_route_ids,
            sorted_weights,
            sorted_expert_ids,
            num_valid_ids,
            expert_frequency,
            retained_sort_unit,
        ) = forward_sorter_metadata
        assert retained_sort_unit == sort_unit
        sorting_workspace = None
    else:
        # The ragged sorter's first dispatch initializes every padded metadata
        # row before histogram/scatter.  Allocating with ``full(-1)`` here
        # merely inserts an extra eager fill launch.
        sorted_token_ids = torch.empty(max_padded, dtype=torch.int32, device=device)
        sorted_weights = torch.empty(max_padded, dtype=torch.float32, device=device)
        sorted_route_ids = (
            torch.empty(max_padded, dtype=torch.int32, device=device)
            if flat_routes
            else None
        )
        sorted_expert_ids = torch.empty(max_blocks, dtype=torch.int32, device=device)
        num_valid_ids = torch.empty(2, dtype=torch.int32, device=device)
        sorting_workspace = (
            torch.empty(workspace_elements, dtype=torch.int32, device=device)
            if workspace_elements
            else None
        )
        expert_frequency = torch.empty(num_experts, dtype=torch.int32, device=device)
    sorter_dummy = torch.empty(4, dtype=torch.int32, device=device)
    # The retained E16 family uses one fixed queue ABI for every non-empty R.
    # The live count remains device-produced, but storage and guarded launch
    # bounds no longer shrink at R<num_experts.
    active_expert_capacity = (
        num_experts
        if use_e16_flat_grouped
        else active_expert_descriptor_capacity(routes, num_experts)
    )
    # A compact descriptor builder can emit this queue in its existing two
    # launches.  Other routing regimes select metadata-direct or standalone
    # construction after the already-required frequency readback below.
    active_expert_storage = (
        torch.empty(
            1 + 2 * active_expert_capacity,
            dtype=torch.int32,
            device=device,
        )
        if (use_grouped_dw1 or use_grouped_dw2)
        and (use_compact_w1 or use_large_grouped_dx or use_e16_deduplicated_metadata)
        else None
    )
    if use_e16_exact_queue:
        exact_queue_max_rows = None if flat_routes else tokens
        exact_tile_queue_bound = exact_m_tile_queue_upper_bound(
            routes,
            num_experts,
            _E16_EXACT_BM,
            max_expert_rows=exact_queue_max_rows,
        )
        exact_tile_queue_storage = torch.empty(
            exact_m_tile_queue_elements(
                routes,
                num_experts,
                _E16_EXACT_BM,
                max_expert_rows=exact_queue_max_rows,
            ),
            dtype=torch.int32,
            device=device,
        )
        # Queue storage and runtime launch geometry both follow the true local
        # route count.  Exact dA/dX kernels read the live device counter and
        # grid-stride over it; using the actual host-known bound avoids idle
        # CTAs when a rank shares a larger compile/tuning policy with peers.
        # This scalar controls only launch grids and never enters a JIT key.
        exact_tile_launch_bound = (
            _e16_exact_tile_launch_bound(routes, num_experts)
            if use_e16_flat_grouped
            else exact_tile_queue_bound
        )
    else:
        exact_tile_queue_bound = 0
        exact_tile_launch_bound = 0
        exact_tile_queue_storage = None
    if use_e16_hot_dw1_splitk:
        (
            split_descriptor_capacity,
            hot_split_rows,
            hot_split_min_rows,
            hot_split_activation_rows,
        ) = _e16_hot_split_schedule(
            routes,
            num_experts,
            route_policy,
        )
        # Allocate from the real route-count bound while retaining one record
        # for a device-side no-op.  This keeps the host launch sequence stable
        # across ranks without turning an XLARGE policy hint into hundreds of
        # MiB of needless scratch on a short local shard.
        hot_expert_capacity = max(
            1,
            min(num_experts, routes // hot_split_min_rows),
        )
        hot_split_storage = torch.empty(
            1 + 3 * split_descriptor_capacity,
            dtype=torch.int32,
            device=device,
        )
        hot_expert_storage = torch.empty(
            1 + 3 * hot_expert_capacity,
            dtype=torch.int32,
            device=device,
        )
        hot_dw1_partials = torch.empty(
            (split_descriptor_capacity, projection_size, hidden_size),
            dtype=torch.float32,
            device=device,
        )
        if use_e16_hot_dw2_splitk:
            hot_dw2_elements = (
                split_descriptor_capacity * hidden_size * intermediate_size
            )
            assert hot_dw2_elements <= hot_dw1_partials.numel()
            hot_dw2_partials = hot_dw1_partials.view(-1)[:hot_dw2_elements].view(
                split_descriptor_capacity,
                hidden_size,
                intermediate_size,
            )
        else:
            hot_dw2_partials = None
    else:
        split_descriptor_capacity = 0
        hot_split_rows = 0
        hot_split_min_rows = 0
        hot_split_activation_rows = 0
        hot_expert_capacity = 0
        hot_split_storage = None
        hot_expert_storage = None
        hot_dw1_partials = None
        hot_dw2_partials = None
    if use_compact_w1 and not use_e16_deduplicated_metadata:
        if flat_routes:
            compact_w1_bound = ragged_compact_m_tile_descriptor_upper_bound(
                routes,
                num_experts,
                grouped_w1_bm,
            )
        else:
            compact_w1_bound = fixed_compact_m_tile_descriptor_upper_bound(
                tokens,
                num_experts,
                topk,
                grouped_w1_bm,
            )
        # Entry zero stores the device-produced live count.  Keeping the count
        # at a fixed offset lets one compiled GEMM specialization serve every
        # host-known descriptor-capacity bound.
        compact_w1_storage = torch.empty(
            compact_w1_bound + 1, dtype=torch.int32, device=device
        )
        compact_w1_total = compact_w1_storage[:1]
        compact_w1_descriptors = compact_w1_storage[1:]
    else:
        compact_w1_bound = 0
        compact_w1_storage = None
        compact_w1_descriptors = None
        compact_w1_total = None
    if use_large_grouped_dx:
        large_dx_bound = fixed_compact_m_tile_descriptor_upper_bound(
            tokens,
            num_experts,
            topk,
            _LARGE_GROUPED_DX_BM,
        )
        large_dx_storage = torch.empty(
            large_dx_bound + 1, dtype=torch.int32, device=device
        )
        large_dx_total = large_dx_storage[:1]
        large_dx_descriptors = large_dx_storage[1:]
    else:
        large_dx_bound = 0
        large_dx_storage = None
        large_dx_total = None
        large_dx_descriptors = None
    if use_fused_forward_state_prepare:
        if use_e16_deduplicated_metadata:
            assert exact_tile_queue_storage is not None
            state_row_schedule = exact_tile_queue_storage
            state_schedule_block_m = _E16_EXACT_BM
            state_schedule_bound = exact_tile_launch_bound
            state_schedule_exact_queue = True
        elif use_large_grouped_dx:
            assert large_dx_storage is not None
            state_row_schedule = large_dx_storage
            state_schedule_block_m = _LARGE_GROUPED_DX_BM
            state_schedule_bound = large_dx_bound
            state_schedule_exact_queue = False
        else:
            assert compact_w1_storage is not None
            state_row_schedule = compact_w1_storage
            state_schedule_block_m = _COMPACT_W1_BM
            state_schedule_bound = compact_w1_bound
            state_schedule_exact_queue = False
    else:
        state_row_schedule = None
        state_schedule_block_m = 0
        state_schedule_bound = 0
        state_schedule_exact_queue = False
    sorted_hidden_shape = (max_padded, hidden_size)
    x_sorted = (
        None
        if use_direct_grouped_dw1_rhs
        else torch.empty(sorted_hidden_shape, dtype=hidden_states.dtype, device=device)
    )
    # The retained-state hostless path reads token-major grad_output directly
    # in both dy preparation and dscore, avoiding a large padded sorted copy.
    dout_sorted = (
        None
        if use_fused_forward_state_prepare
        else torch.empty(sorted_hidden_shape, dtype=hidden_states.dtype, device=device)
    )
    dy = torch.empty(sorted_hidden_shape, dtype=hidden_states.dtype, device=device)
    # The fused state path derives route-score gradients from q=dout@W2 and
    # therefore does not materialize the forward down projection.  Fallbacks
    # retain their original padding initialization contract.
    projection = (
        None
        if use_fused_da_dscore
        else (
            torch.empty_like(dy)
            if use_hostless_grouped or not use_grouped_w2
            else torch.zeros_like(dy)
        )
    )
    # Grouped W1 writes ceil(real_rows/BM)*BM rows instead of every
    # SORTED_BM-padded row.  Zero-initialize the untouched suffix: gather makes
    # padded x/dout zero, so its dy/da/dz and therefore dW/db contributions
    # remain exactly zero while all activation inputs stay finite.
    # The same fast path differentiates directly from compact route-order
    # state, so it does not allocate or write a padded sorted preactivation.
    preactivation = (
        None
        if use_fused_forward_state_prepare
        else torch.empty(
            (max_padded, projection_size),
            dtype=hidden_states.dtype,
            device=device,
        )
    )
    if (
        preactivation is not None
        and not reuse_forward_preactivation
        and use_grouped_w1
        and not use_hostless_grouped
    ):
        preactivation.zero_()
    activation = torch.empty(
        (max_padded, intermediate_size), dtype=hidden_states.dtype, device=device
    )
    # Grouped dA writes real expert rows only.  Legacy derivatives consume full
    # sorter-padded segments and therefore need zero tails; exact-row fused
    # derivatives skip those rows and can leave dA uninitialized there.
    da = (
        torch.empty_like(activation)
        if use_fused_da_dscore or use_hostless_grouped or not use_grouped_da
        else torch.zeros_like(activation)
    )
    # Compact descriptor paths may consume their rounded row tails and need
    # zero dZ there.  The exact E16 queue carries a valid-row count per tile;
    # its derivative and both downstream GEMMs mask those tails, so an eager
    # 28 MiB memset is unnecessary.
    dz_needs_zero = (
        use_hostless_grouped or use_fused_da_dscore
    ) and not use_e16_deduplicated_metadata
    dz_factory = torch.zeros if dz_needs_zero else torch.empty
    dz = dz_factory(
        (max_padded, projection_size),
        dtype=hidden_states.dtype,
        device=device,
    )
    dx_sorted = (
        None
        if direct_grouped_dx_routes or use_flat_route_order_dx
        else (torch.zeros_like(dy) if use_e16_flat_grouped else torch.empty_like(dy))
    )
    dx = torch.empty_like(hidden_states, memory_format=torch.contiguous_format)
    if use_flat_identity_dx:
        # Route r is token r, so the grouped route-order epilogue already has
        # the final dX address.  Avoid both the [R,H] temporary and the
        # segmented reduction launch.
        dx_routes = dx
    elif use_flat_segmented_dx:
        dx_routes = torch.empty(
            (routes, hidden_size),
            dtype=hidden_states.dtype,
            device=device,
        )
    elif flat_routes:
        dx_routes = None
    else:
        dx_routes = torch.empty(
            (tokens, topk, hidden_size),
            dtype=hidden_states.dtype,
            device=device,
        )
    dx_accum = (
        torch.empty((tokens, hidden_size), dtype=torch.float32, device=device)
        if flat_routes and not use_flat_route_order_dx
        else None
    )

    # The grouped pair is initialized from routing metadata below, before its
    # first contraction; every other path retains eager zero initialization.
    grouped_weight_grads = use_grouped_dw1 and use_grouped_dw2
    weight_grad_factory = (
        torch.zeros_like
        if not grouped_weight_grads or hostless_dense_weight_zero
        else torch.empty_like
    )
    dw1 = weight_grad_factory(w1, memory_format=torch.contiguous_format)
    dw2 = weight_grad_factory(w2, memory_format=torch.contiguous_format)
    droute_weights = torch.empty_like(
        route_weights, memory_format=torch.contiguous_format
    )
    db1 = (
        torch.empty_like(b1, memory_format=torch.contiguous_format)
        if b1 is not None
        else None
    )
    db2 = (
        torch.empty_like(b2, memory_format=torch.contiguous_format)
        if b2 is not None
        else None
    )

    # Custom kernels consume raw storage only; detach keeps DLPack conversion
    # valid when this function is called from a torch.autograd.Function.
    x_arg = hidden_states.detach()
    w1_arg = w1.detach()
    w2_arg = w2.detach()
    ids_arg = expert_ids.detach() if expert_ids is not None else None
    weights_arg = route_weights.detach()
    token_arg = token_indices.detach() if token_indices is not None else None
    dout_arg = grad_output.detach()
    b1_arg = b1.detach() if b1 is not None else None
    b2_arg = b2.detach() if b2 is not None else None

    with torch.cuda.device(device):
        stream = torch.cuda.current_stream(device)
        if forward_sorter_metadata is not None:
            assert forward_state_data is not None
            route_preactivation, producer_stream, ready_event = forward_state_data
            if int(stream.cuda_stream) != producer_stream:
                stream.wait_event(ready_event)
            # The state, rather than a reusable forward workspace, owns these
            # allocations.  Track their lifetime on the consuming stream just
            # like the retained preactivation tensor.
            route_preactivation.record_stream(stream)
            for metadata_tensor in forward_sorter_metadata[:-1]:
                metadata_tensor.record_stream(stream)
        if has_bias:
            clear_bias_gradients = _compile_bias_gradient_clear(
                projection_size,
                hidden_size,
                num_experts,
                compute_dtype,
                device_index,
            )
            _run_compiled(clear_bias_gradients, db1, db2, stream)

        if flat_routes:
            assert sorted_route_ids is not None
            if forward_sorter_metadata is None:
                assert token_arg is not None
                assert ids_arg is not None
                assert sorting_workspace is not None
                moe_ragged_sorting_flydsl(
                    token_arg,
                    ids_arg,
                    weights_arg,
                    expert_frequency,
                    sorting_workspace,
                    sorted_token_ids,
                    sorted_weights,
                    sorted_expert_ids,
                    num_valid_ids,
                    sorter_dummy,
                    num_experts,
                    tokens=tokens,
                    max_padded_routes=max_padded,
                    unit_size=sort_unit,
                    sorted_route_ids=sorted_route_ids,
                )
        else:
            assert ids_arg is not None
            if not use_sorter_native_backward_metadata:
                route_grid = max(1, (routes + _BLOCK_THREADS - 1) // _BLOCK_THREADS)
                histogram = _compile_expert_histogram(num_experts, device_index)
                _run_compiled(
                    histogram, ids_arg, expert_frequency, routes, route_grid, stream
                )
            else:
                assert active_expert_storage is not None
                assert large_dx_storage is not None
            moe_sorting_flydsl(
                ids_arg,
                weights_arg,
                sorted_token_ids,
                sorted_weights,
                sorted_expert_ids,
                num_valid_ids,
                sorter_dummy,
                num_experts,
                unit_size=sort_unit,
                workspace=sorting_workspace,
                expert_frequency_out=(
                    expert_frequency if use_sorter_native_backward_metadata else None
                ),
                active_expert_storage=(
                    active_expert_storage
                    if use_sorter_native_backward_metadata
                    else None
                ),
                bm64_schedule_storage=(
                    large_dx_storage if use_sorter_native_backward_metadata else None
                ),
            )

        if use_e16_exact_queue:
            assert exact_tile_queue_storage is not None
            build_exact_m_tile_queue(
                expert_frequency,
                sorted_expert_ids,
                num_valid_ids,
                exact_tile_queue_storage,
                block_m=_E16_EXACT_BM,
                sorted_block_m=sort_unit,
                queue_capacity=exact_tile_queue_bound,
                active_expert_storage=(
                    active_expert_storage if use_e16_deduplicated_metadata else None
                ),
                active_expert_capacity=(
                    active_expert_capacity if use_e16_deduplicated_metadata else None
                ),
                hot_split_storage=(
                    hot_split_storage if use_e16_hot_dw1_splitk else None
                ),
                hot_expert_storage=(
                    hot_expert_storage if use_e16_hot_dw1_splitk else None
                ),
                split_rows=(hot_split_rows if use_e16_hot_dw1_splitk else None),
                min_hot_rows=(hot_split_min_rows if use_e16_hot_dw1_splitk else None),
                split_activation_rows=(
                    hot_split_activation_rows if use_e16_hot_dw1_splitk else None
                ),
                sparse_split_max_experts=(
                    _E16_DW1_SPLIT_SPARSE_MAX_ACTIVE_EXPERTS
                    if use_e16_hot_dw1_splitk
                    else None
                ),
                stream=stream,
            )

        if use_large_grouped_dx and not use_sorter_native_backward_metadata:
            assert large_dx_descriptors is not None
            assert large_dx_total is not None
            build_compact_m_tile_descriptors(
                expert_frequency,
                sorted_expert_ids,
                num_valid_ids,
                large_dx_descriptors,
                large_dx_total,
                block_m=_LARGE_GROUPED_DX_BM,
                sorted_block_m=sort_unit,
                descriptor_capacity=large_dx_bound,
                active_expert_storage=active_expert_storage,
                active_expert_capacity=(
                    active_expert_capacity
                    if active_expert_storage is not None
                    else None
                ),
                stream=stream,
            )

        # The grouped W1 kernel derives its live CTA bound and expert mapping
        # exclusively from device-produced sorter metadata.  Its launch grid is
        # only a safe allocation bound, so no expert-frequency readback is
        # required to recompute preactivation.  Keep the generic fallback for
        # non-SwiGLU/FP16 shapes until their epilogues are enabled here.
        if use_grouped_w1:
            if use_compact_w1 and not use_e16_deduplicated_metadata:
                assert compact_w1_descriptors is not None
                assert compact_w1_total is not None
                build_compact_m_tile_descriptors(
                    expert_frequency,
                    sorted_expert_ids,
                    num_valid_ids,
                    compact_w1_descriptors,
                    compact_w1_total,
                    block_m=grouped_w1_bm,
                    sorted_block_m=sort_unit,
                    descriptor_capacity=compact_w1_bound,
                    active_expert_storage=active_expert_storage,
                    active_expert_capacity=(
                        active_expert_capacity
                        if active_expert_storage is not None
                        else None
                    ),
                    stream=stream,
                )
            if not reuse_forward_preactivation:
                assert preactivation is not None
                grouped_w1 = _compile_grouped_w1_recompute(
                    hidden_size,
                    intermediate_size,
                    num_experts,
                    topk,
                    has_bias,
                    interleaved_w1,
                    use_compact_w1,
                    device_index,
                )
                grouped_w1_grid = (
                    compact_w1_bound if use_compact_w1 else num_experts
                ) * (intermediate_size // grouped_w1_bn)
                dummy_ptr = w1_arg.data_ptr()
                _run_compiled(
                    grouped_w1,
                    x_arg.data_ptr(),
                    w1_arg.data_ptr(),
                    (
                        compact_w1_storage.data_ptr()
                        if compact_w1_storage is not None
                        else expert_frequency.data_ptr()
                    ),
                    dummy_ptr if b1_arg is None else b1_arg.data_ptr(),
                    sorted_expert_ids.data_ptr(),
                    num_valid_ids.data_ptr(),
                    sorted_token_ids.data_ptr(),
                    tokens,
                    int(grouped_w1_grid),
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    float("inf"),
                    preactivation.data_ptr(),
                    stream,
                )

        # Fully grouped BF16/SwiGLU paths never reconstruct expert segments on
        # the host.  Fixed-K routing bounds each expert by the token count,
        # while row kernels read a device queue or the exact padded extent from
        # ``num_valid_ids[0]``.  All fallback contracts retain the original
        # frequency readback and segment loop.
        if use_hostless_grouped:
            frequencies = None
            segments: list[tuple[int, int, int]] = []
            padded_rows = 0
            max_expert_rows = routes if flat_routes else tokens
        else:
            frequencies, segments, padded_rows, max_expert_rows = (
                _materialize_expert_segments(expert_frequency, sort_unit)
            )

        if grouped_weight_grads:
            if use_hostless_grouped:
                if not hostless_dense_weight_zero:
                    active_count_storage = (
                        active_expert_storage
                        if active_expert_storage is not None
                        else num_valid_ids
                    )
                    active_count_divisor = (
                        1 if active_expert_storage is not None else sort_unit
                    )
                    zero_weight_grads_adaptive_flydsl(
                        expert_frequency,
                        active_count_storage,
                        dw1,
                        dw2,
                        active_count_divisor=active_count_divisor,
                        dense_active_ratio=_INACTIVE_WEIGHT_GRAD_ZERO_ACTIVE_RATIO,
                        blocks_per_expert=_inactive_weight_grad_zero_blocks_per_expert(
                            use_e16_deduplicated_metadata=use_e16_deduplicated_metadata,
                            num_experts=num_experts,
                        ),
                        stream=stream,
                    )
            else:
                active_experts = len(segments)
                selective_weight_grad_zero = (
                    active_experts * _INACTIVE_WEIGHT_GRAD_ZERO_ACTIVE_RATIO
                    >= num_experts
                )
                if selective_weight_grad_zero and active_experts < num_experts:
                    zero_inactive_weight_grads_flydsl(
                        expert_frequency,
                        dw1,
                        dw2,
                        stream=stream,
                    )
                elif not selective_weight_grad_zero:
                    dw1.zero_()
                    dw2.zero_()

        # One sorter block per active expert is itself a valid schedule and is
        # the lowest-latency path.  Compact W1 regimes already produced the
        # shared queue above; long/non-compact regimes build it once here for
        # both weight-gradient TN contractions.
        use_tn_metadata_direct = use_standalone_e16_tn_metadata_direct or (
            (use_grouped_dw1 or use_grouped_dw2)
            and active_expert_storage is None
            and sort_unit == _BACKWARD_SORT_UNIT
            and (
                routes <= sort_unit
                if use_hostless_grouped
                else max_expert_rows <= sort_unit
            )
        )
        if (
            (use_grouped_dw1 or use_grouped_dw2)
            and not use_tn_metadata_direct
            and active_expert_storage is None
        ):
            active_expert_storage = torch.empty(
                active_expert_queue_elements(routes, num_experts),
                dtype=torch.int32,
                device=device,
            )
            build_active_expert_queue_flydsl(
                expert_frequency,
                sorted_expert_ids,
                num_valid_ids,
                routes=routes,
                queue_storage=active_expert_storage,
                stream=stream,
            )

        if not use_fused_forward_state_prepare:
            assert x_sorted is not None
            assert dout_sorted is not None
            gather = _compile_gather(
                hidden_size,
                compute_dtype,
                device_index,
                use_hostless_grouped,
            )
            gather_work = (max_padded if use_hostless_grouped else padded_rows) * (
                hidden_size // 4
            )
            gather_grid = max(1, (gather_work + _BLOCK_THREADS - 1) // _BLOCK_THREADS)
            if use_hostless_grouped:
                gather_grid = min(hostless_row_grid_cap, gather_grid)
            _run_compiled(
                gather,
                x_arg,
                dout_arg,
                sorted_token_ids,
                x_sorted,
                dout_sorted,
                num_valid_ids,
                tokens,
                padded_rows,
                gather_grid,
                stream,
            )

        # Other activation/dtype combinations retain the original per-expert
        # preactivation path.
        if not use_grouped_w1 and not reuse_forward_preactivation:
            assert preactivation is not None
            assert x_sorted is not None
            for expert, start, rows in segments:
                end = start + rows
                gemm_a16w16(
                    x_sorted[start:end],
                    w1_arg[expert].transpose(0, 1),
                    out=preactivation[start:end],
                    bias=None if b1_arg is None else b1_arg[expert],
                    user_kwargs=_GEMM_KWARGS,
                    stream=stream,
                    layout="nt",
                )

        activation_grid = (
            min(hostless_row_grid_cap, max_padded)
            if use_hostless_grouped
            else padded_rows
        )
        if forward_state_data is None:
            assert preactivation is not None
            assert dout_sorted is not None
            activation_prepare = _compile_activation_prepare(
                hidden_size,
                intermediate_size,
                activation_name,
                compute_dtype,
                interleaved_w1,
                device_index,
                use_hostless_grouped,
            )
            _run_compiled(
                activation_prepare,
                preactivation,
                activation,
                dout_sorted,
                dy,
                sorted_weights,
                sorted_token_ids,
                num_valid_ids,
                tokens,
                activation_grid,
                stream,
            )
        else:
            route_preactivation, producer_stream, ready_event = forward_state_data
            route_preactivation.record_stream(stream)
            if int(stream.cuda_stream) != producer_stream:
                stream.wait_event(ready_event)
            if use_fused_forward_state_prepare:
                if not use_direct_grouped_da_rhs:
                    fused_prepare = _compile_fused_forward_state_prepare(
                        hidden_size,
                        intermediate_size,
                        topk,
                        interleaved_w1,
                        device_index,
                        state_schedule_block_m,
                        use_fused_da_dscore,
                        store_x_sorted=not use_direct_grouped_dw1_rhs,
                        flat_routes=flat_routes,
                        exact_tile_queue=state_schedule_exact_queue,
                        defer_activation=use_e16_exact_queue,
                    )
                    assert state_row_schedule is not None
                    state_prepare_grid = min(
                        hostless_row_grid_cap,
                        max(1, state_schedule_bound * state_schedule_block_m),
                    )
                    _run_compiled(
                        fused_prepare,
                        x_arg,
                        dout_arg,
                        route_preactivation,
                        x_arg if x_sorted is None else x_sorted,
                        activation,
                        dy,
                        sorted_weights,
                        sorted_token_ids,
                        (
                            sorted_route_ids
                            if sorted_route_ids is not None
                            else sorted_token_ids
                        ),
                        state_row_schedule,
                        tokens,
                        routes,
                        state_prepare_grid,
                        stream,
                    )
            else:
                assert preactivation is not None
                assert dout_sorted is not None
                activation_prepare = _compile_activation_prepare_from_forward_state(
                    hidden_size,
                    intermediate_size,
                    topk,
                    interleaved_w1,
                    device_index,
                    use_hostless_grouped,
                )
                _run_compiled(
                    activation_prepare,
                    route_preactivation,
                    preactivation,
                    activation,
                    dout_sorted,
                    dy,
                    sorted_weights,
                    sorted_token_ids,
                    num_valid_ids,
                    tokens,
                    activation_grid,
                    stream,
                )

        if use_grouped_dw2 and not use_fused_da_dscore:
            _launch_grouped_dw2(
                dy,
                activation,
                expert_frequency,
                sorted_expert_ids,
                num_valid_ids,
                dw2,
                active_expert_storage,
                use_hostless_grouped=use_hostless_grouped,
                use_tn_metadata_direct=use_tn_metadata_direct,
                max_expert_rows=max_expert_rows,
                route_policy=route_policy,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                active_experts=len(segments),
                stream=stream,
                hot_split_state=(
                    (hot_split_storage, hot_expert_storage, hot_dw2_partials)
                    if use_e16_hot_dw2_splitk
                    else None
                ),
                hot_split_min_rows=(
                    hot_split_min_rows if use_e16_hot_dw2_splitk else None
                ),
            )

        if use_grouped_da:
            if use_e16_exact_queue:
                assert exact_tile_queue_storage is not None
                grouped_da = _compile_e16_exact_grouped_da(
                    hidden_size,
                    intermediate_size,
                    num_experts,
                    device_index,
                    gather_a_rows=use_direct_grouped_da_rhs,
                )
                grouped_da_grid = max(
                    1,
                    min(
                        _GROUPED_DX_GRID_CAP,
                        exact_tile_launch_bound
                        * (intermediate_size // _E16_EXACT_DA_BN),
                    ),
                )
                grouped_da_args = (
                    (
                        dout_arg.data_ptr()
                        if use_direct_grouped_da_rhs
                        else dy.data_ptr()
                    ),
                    w2_arg.data_ptr(),
                    exact_tile_queue_storage.data_ptr(),
                    sorted_expert_ids.data_ptr(),
                    num_valid_ids.data_ptr(),
                    da.data_ptr(),
                )
                if use_direct_grouped_da_rhs:
                    grouped_da_args = (
                        *grouped_da_args,
                        sorted_token_ids.data_ptr(),
                        tokens,
                    )
                _run_compiled(
                    grouped_da,
                    *grouped_da_args,
                    int(grouped_da_grid),
                    stream,
                )
            else:
                if use_hostless_grouped and active_expert_storage is not None:
                    grouped_da_profiles = _grouped_da_hostless_profiles(
                        tokens=tokens,
                        hidden_size=hidden_size,
                        intermediate_size=intermediate_size,
                        num_experts=num_experts,
                        topk=topk,
                        max_expert_rows=max_expert_rows,
                    )
                    active_count_ptr = active_expert_storage.data_ptr()
                else:
                    grouped_da_profiles = (
                        (
                            *_grouped_da_tuning(max_expert_rows, hidden_size),
                            False,
                            0,
                            None,
                            _GROUPED_DA_STAGES,
                            None,
                        ),
                    )
                    # Unguarded specializations do not dereference this argument.
                    active_count_ptr = expert_frequency.data_ptr()

                for (
                    grouped_da_bm,
                    grouped_da_bn,
                    grouped_da_bk,
                    grouped_da_mw,
                    grouped_da_nw,
                    grouped_da_queue_direct,
                    min_active_experts,
                    max_active_experts,
                    grouped_da_stages,
                    grouped_da_grid_cap,
                ) in grouped_da_profiles:
                    grouped_da = _compile_grouped_da(
                        hidden_size,
                        intermediate_size,
                        num_experts,
                        grouped_da_bm,
                        grouped_da_bn,
                        grouped_da_bk,
                        grouped_da_mw,
                        grouped_da_nw,
                        device_index,
                        grouped_da_queue_direct,
                        min_active_experts,
                        max_active_experts,
                        stages=grouped_da_stages,
                        persistent=grouped_da_grid_cap is not None,
                    )
                    if grouped_da_queue_direct:
                        guarded_capacity = active_expert_capacity
                        if max_active_experts is not None:
                            guarded_capacity = min(guarded_capacity, max_active_experts)
                        grouped_da_grid = guarded_capacity * (
                            intermediate_size // grouped_da_bn
                        )
                    else:
                        grouped_da_grid = num_experts * (
                            intermediate_size // grouped_da_bn
                        )
                        if grouped_da_grid_cap is not None:
                            grouped_da_grid = min(grouped_da_grid, grouped_da_grid_cap)
                    _run_compiled(
                        grouped_da,
                        dy.data_ptr(),
                        w2_arg.data_ptr(),
                        expert_frequency.data_ptr(),
                        sorted_expert_ids.data_ptr(),
                        num_valid_ids.data_ptr(),
                        active_count_ptr,
                        da.data_ptr(),
                        int(grouped_da_grid),
                        stream,
                    )

        if use_grouped_w2 and not use_fused_da_dscore:
            assert projection is not None
            grouped_w2 = _compile_grouped_w2_recompute(
                hidden_size,
                intermediate_size,
                num_experts,
                topk,
                has_bias,
                device_index,
            )
            grouped_w2_grid = num_experts * (hidden_size // _GROUPED_W2_BN)
            dummy_ptr = w2_arg.data_ptr()
            _run_compiled(
                grouped_w2,
                activation.data_ptr(),
                w2_arg.data_ptr(),
                expert_frequency.data_ptr(),
                dummy_ptr if b2_arg is None else b2_arg.data_ptr(),
                sorted_expert_ids.data_ptr(),
                num_valid_ids.data_ptr(),
                sorted_token_ids.data_ptr(),
                sorted_weights.data_ptr(),
                tokens,
                max_blocks,
                int(grouped_w2_grid),
                projection.data_ptr(),
                stream,
            )

        # Fallbacks recompute the down projection for ds and use materialized
        # A16 dy for both da and dW2.  The fused path instead gets dscore from
        # q=dout@W2, so it deliberately skips these per-expert projections.
        for expert, start, rows in segments:
            end = start + rows
            if not use_grouped_w2 and not use_fused_da_dscore:
                assert projection is not None
                gemm_a16w16(
                    activation[start:end],
                    w2_arg[expert].transpose(0, 1),
                    out=projection[start:end],
                    bias=None if b2_arg is None else b2_arg[expert],
                    user_kwargs=_GEMM_KWARGS,
                    stream=stream,
                    layout="nt",
                )
            if not use_grouped_da:
                gemm_a16w16(
                    dy[start:end],
                    w2_arg[expert],
                    out=da[start:end],
                    user_kwargs=_GEMM_KWARGS,
                    stream=stream,
                    layout="nn",
                )
            if not use_grouped_dw2:
                gemm_a16w16(
                    dy[start:end].transpose(0, 1),
                    activation[start:end],
                    out=dw2[expert],
                    user_kwargs=_GEMM_KWARGS,
                    stream=stream,
                    layout="tn",
                )

        derivative_grid = (
            min(hostless_row_grid_cap, max_padded)
            if use_hostless_grouped
            else padded_rows
        )
        if use_fused_da_dscore:
            assert forward_state_data is not None
            route_preactivation = forward_state_data[0]
            fused_derivative = _compile_fused_activation_derivative_dscore_scale_dy(
                hidden_size,
                intermediate_size,
                topk,
                interleaved_w1,
                device_index,
                state_schedule_block_m,
                flat_routes=flat_routes,
                device_padded_rows=use_e16_deduplicated_metadata,
                materialize_activation=use_e16_exact_queue,
                separate_dy_source=use_direct_grouped_da_rhs,
            )
            assert state_row_schedule is not None
            derivative_schedule = (
                num_valid_ids if use_e16_deduplicated_metadata else state_row_schedule
            )
            derivative_work_bound = (
                max_padded
                if use_e16_deduplicated_metadata
                else state_schedule_bound * state_schedule_block_m
            )
            derivative_grid = min(
                hostless_row_grid_cap,
                max(1, derivative_work_bound),
            )
            _run_compiled(
                fused_derivative,
                route_preactivation,
                activation,
                da,
                dout_arg if use_direct_grouped_da_rhs else dy,
                dy,
                dz,
                sorted_weights,
                sorted_token_ids,
                sorted_route_ids if sorted_route_ids is not None else sorted_token_ids,
                droute_weights,
                derivative_schedule,
                tokens,
                routes,
                derivative_grid,
                stream,
            )
            # dy was intentionally left unscaled until the fused row kernel so
            # grouped dA could compute q=dout@W2.  dW2 can start only now.
            _launch_grouped_dw2(
                dy,
                activation,
                expert_frequency,
                sorted_expert_ids,
                num_valid_ids,
                dw2,
                active_expert_storage,
                use_hostless_grouped=use_hostless_grouped,
                use_tn_metadata_direct=use_tn_metadata_direct,
                max_expert_rows=max_expert_rows,
                route_policy=route_policy,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                active_experts=len(segments),
                stream=stream,
                hot_split_state=(
                    (hot_split_storage, hot_expert_storage, hot_dw2_partials)
                    if use_e16_hot_dw2_splitk
                    else None
                ),
                hot_split_min_rows=(
                    hot_split_min_rows if use_e16_hot_dw2_splitk else None
                ),
            )
        elif use_fused_forward_state_prepare:
            assert forward_state_data is not None
            route_preactivation = forward_state_data[0]
            derivative_from_state = _compile_activation_derivative_from_forward_state(
                intermediate_size,
                topk,
                interleaved_w1,
                device_index,
            )
            assert compact_w1_storage is not None
            derivative_grid = min(
                hostless_row_grid_cap,
                max(1, compact_w1_bound * _COMPACT_W1_BM),
            )
            _run_compiled(
                derivative_from_state,
                route_preactivation,
                da,
                dz,
                sorted_token_ids,
                compact_w1_storage,
                tokens,
                derivative_grid,
                stream,
            )
        else:
            assert preactivation is not None
            activation_derivative = _compile_activation_derivative(
                intermediate_size,
                activation_name,
                compute_dtype,
                interleaved_w1,
                device_index,
                use_hostless_grouped,
            )
            _run_compiled(
                activation_derivative,
                preactivation,
                da,
                dz,
                sorted_token_ids,
                num_valid_ids,
                tokens,
                derivative_grid,
                stream,
            )

        def _launch_selected_grouped_dw1(
            launch_stream: torch.cuda.Stream,
        ) -> None:
            grouped_dw1_rhs = x_arg if use_direct_grouped_dw1_rhs else x_sorted
            assert grouped_dw1_rhs is not None
            grouped_dw1_active_storage = active_expert_storage
            (
                grouped_dw1_bm,
                grouped_dw1_bn,
                grouped_dw1_bk,
                grouped_dw1_k_padding,
                grouped_dw1_m_waves,
                grouped_dw1_n_waves,
            ) = _grouped_dw1_tuning(
                (
                    e16_route_policy_representative(route_policy)
                    if use_e16_flat_grouped
                    else max_expert_rows
                ),
                hidden_size,
                intermediate_size,
                direct_rhs=use_direct_grouped_dw1_rhs,
                # E16 flat identity routing trades the E896-oriented BN256
                # profile for enough N-grid parallelism under expert skew.
                wide_direct_rhs=not use_flat_identity_dx,
            )
            if _use_e16_dw1_dual_profile(
                direct_rhs=use_direct_grouped_dw1_rhs,
                e16_flat_grouped=use_e16_flat_grouped,
                metadata_direct=use_tn_metadata_direct,
                route_policy=route_policy,
            ):
                grouped_dw1_profiles = (
                    (
                        128,
                        64,
                        _GROUPED_DW1_BLOCK_K,
                        0,
                        2,
                        2,
                        0,
                        _E16_DW1_NARROW_MAX_ACTIVE_EXPERTS,
                    ),
                    (
                        grouped_dw1_bm,
                        grouped_dw1_bn,
                        grouped_dw1_bk,
                        grouped_dw1_k_padding,
                        grouped_dw1_m_waves,
                        grouped_dw1_n_waves,
                        _E16_DW1_NARROW_MAX_ACTIVE_EXPERTS + 1,
                        None,
                    ),
                )
            else:
                grouped_dw1_profiles = (
                    (
                        grouped_dw1_bm,
                        grouped_dw1_bn,
                        grouped_dw1_bk,
                        grouped_dw1_k_padding,
                        grouped_dw1_m_waves,
                        grouped_dw1_n_waves,
                        0,
                        None,
                    ),
                )

            for (
                profile_bm,
                profile_bn,
                profile_bk,
                profile_k_padding,
                profile_m_waves,
                profile_n_waves,
                min_active_experts,
                max_active_experts,
            ) in grouped_dw1_profiles:
                grouped_dw1_kwargs = {
                    "block_m": profile_bm,
                    "block_n": profile_bn,
                    "block_k": profile_bk,
                    "k_padding": profile_k_padding,
                    "m_waves": profile_m_waves,
                    "n_waves": profile_n_waves,
                    "stream": launch_stream,
                }
                if use_e16_hot_dw1_splitk:
                    assert hot_expert_storage is not None
                    grouped_dw1_kwargs["hot_expert_storage"] = hot_expert_storage
                    grouped_dw1_kwargs["hot_split_min_rows"] = hot_split_min_rows
                if use_tn_metadata_direct:
                    grouped_tn_from_metadata_flydsl(
                        dz,
                        grouped_dw1_rhs,
                        expert_frequency,
                        sorted_expert_ids,
                        num_valid_ids,
                        dw1,
                        sorted_token_ids=(
                            sorted_token_ids if use_direct_grouped_dw1_rhs else None
                        ),
                        **grouped_dw1_kwargs,
                    )
                else:
                    assert grouped_dw1_active_storage is not None
                    grouped_tn_from_queue_flydsl(
                        dz,
                        grouped_dw1_rhs,
                        expert_frequency,
                        grouped_dw1_active_storage,
                        dw1,
                        sorted_token_ids=(
                            sorted_token_ids if use_direct_grouped_dw1_rhs else None
                        ),
                        min_active_experts=min_active_experts,
                        max_active_experts=max_active_experts,
                        **grouped_dw1_kwargs,
                    )

            if use_e16_hot_dw1_splitk:
                assert hot_split_storage is not None
                assert hot_expert_storage is not None
                assert hot_dw1_partials is not None
                grouped_tn_splitk_from_queue_flydsl(
                    dz,
                    x_arg,
                    expert_frequency,
                    hot_split_storage,
                    hot_dw1_partials,
                    sorted_token_ids=sorted_token_ids,
                    block_m=_E16_DW1_SPLIT_BM,
                    block_n=_E16_DW1_SPLIT_BN,
                    block_k=_E16_DW1_SPLIT_BK,
                    m_waves=_E16_DW1_SPLIT_M_WAVES,
                    n_waves=_E16_DW1_SPLIT_N_WAVES,
                    stages=_E16_DW1_SPLIT_STAGES,
                    stream=launch_stream,
                )
                finalize_hot_splitk_flydsl(
                    hot_split_storage,
                    hot_expert_storage,
                    hot_dw1_partials,
                    dw1,
                    stream=launch_stream,
                )

        def _launch_selected_grouped_dx(
            launch_stream: torch.cuda.Stream,
        ) -> None:
            # Compact queues round real expert rows to their selected BM.
            # Every scheduled tail stays inside the sorter's 64-row padding,
            # and the fixed-K route epilogue ignores non-sentinel rows.  Ragged
            # routing retains the separate sorted-output reduction below.
            if use_e16_exact_queue:
                assert exact_tile_queue_storage is not None
                grouped_dx_bm = _E16_EXACT_BM
                grouped_dx_profiles = (
                    (
                        _E16_EXACT_DX_BN,
                        _E16_EXACT_N_WAVES,
                        0,
                        None,
                        False,
                        None,
                    ),
                )
                grouped_dx_m_tiles = exact_tile_launch_bound
                grouped_dx_compact = True
                grouped_dx_schedule = exact_tile_queue_storage
            elif use_large_grouped_dx:
                grouped_dx_bm = _LARGE_GROUPED_DX_BM
                if use_grouped_dx_m_reuse:
                    grouped_dx_profiles = (
                        (
                            _LARGE_GROUPED_DX_BN,
                            _LARGE_GROUPED_DX_N_WAVES,
                            0,
                            None,
                            False,
                            _GROUPED_DX_M_REUSE_MIN_ACTIVE_EXPERTS,
                        ),
                    )
                else:
                    grouped_dx_profiles = (
                        (
                            _LARGE_GROUPED_DX_BN,
                            _LARGE_GROUPED_DX_N_WAVES,
                            0,
                            None,
                            False,
                            None,
                        ),
                    )
                grouped_dx_m_tiles = large_dx_bound
                grouped_dx_compact = True
                grouped_dx_schedule = large_dx_storage
            elif use_hostless_grouped and active_expert_storage is not None:
                grouped_dx_bm = _GROUPED_DX_BM
                sparse_dx = _grouped_dx_tuning(0, hidden_size)
                dense_dx = _grouped_dx_tuning(_GROUPED_DX_DENSE_EXPERTS, hidden_size)
                if sparse_dx == dense_dx:
                    grouped_dx_profiles = ((*sparse_dx, 0, None, False, None),)
                else:
                    grouped_dx_profiles = (
                        (*sparse_dx, 0, _GROUPED_DX_DENSE_EXPERTS - 1, False, None),
                        (*dense_dx, _GROUPED_DX_DENSE_EXPERTS, None, False, None),
                    )
                grouped_dx_m_tiles = compact_w1_bound
                grouped_dx_compact = use_compact_w1
                grouped_dx_schedule = compact_w1_storage
            else:
                grouped_dx_bm = _GROUPED_DX_BM
                active_experts = (
                    min(routes, num_experts) if use_hostless_grouped else len(segments)
                )
                grouped_dx_profiles = (
                    (
                        *_grouped_dx_tuning(active_experts, hidden_size),
                        0,
                        None,
                        False,
                        None,
                    ),
                )
                grouped_dx_m_tiles = (
                    sum(
                        (int(count) + _GROUPED_DX_BM - 1) // _GROUPED_DX_BM
                        for count in frequencies
                    )
                    if use_compact_w1
                    else active_experts
                )
                grouped_dx_compact = use_compact_w1
                grouped_dx_schedule = compact_w1_storage

            for (
                grouped_dx_bn,
                grouped_dx_n_waves,
                min_active_experts,
                max_active_experts,
                expert_m_reuse,
                expert_m_reuse_threshold,
            ) in grouped_dx_profiles:
                grouped_dx = _compile_grouped_dx(
                    hidden_size,
                    intermediate_size,
                    num_experts,
                    grouped_dx_bm,
                    grouped_dx_bn,
                    grouped_dx_n_waves,
                    grouped_dx_compact,
                    device_index,
                    min_active_experts,
                    max_active_experts,
                    store_route_slots=direct_grouped_dx_routes,
                    store_route_ids=use_flat_route_order_dx,
                    top_k=topk,
                    expert_m_reuse=expert_m_reuse,
                    expert_m_reuse_threshold=expert_m_reuse_threshold,
                    exact_tile_queue=use_e16_exact_queue,
                )
                if use_e16_exact_queue:
                    assert exact_tile_queue_storage is not None
                    profile_m_tiles = grouped_dx_m_tiles
                    profile_schedule = exact_tile_queue_storage
                    profile_eids = sorted_expert_ids
                elif expert_m_reuse:
                    assert active_expert_storage is not None
                    profile_m_tiles = 2 * active_expert_capacity
                    profile_schedule = active_expert_storage
                    profile_eids = expert_frequency
                elif expert_m_reuse_threshold is not None:
                    assert active_expert_storage is not None
                    profile_m_tiles = max(
                        grouped_dx_m_tiles, 2 * active_expert_capacity
                    )
                    profile_schedule = grouped_dx_schedule
                    profile_eids = sorted_expert_ids
                else:
                    profile_m_tiles = grouped_dx_m_tiles
                    profile_schedule = grouped_dx_schedule
                    profile_eids = sorted_expert_ids
                grouped_dx_logical_grid = profile_m_tiles * (
                    hidden_size // grouped_dx_bn
                )
                grouped_dx_grid = max(
                    1,
                    (
                        _e16_exact_dx_grid(
                            grouped_dx_logical_grid,
                            hostless_row_grid_cap,
                        )
                        if use_e16_exact_queue
                        else min(_GROUPED_DX_GRID_CAP, grouped_dx_logical_grid)
                    ),
                )
                _run_compiled(
                    grouped_dx,
                    dz.data_ptr(),
                    w1_arg.data_ptr(),
                    (
                        profile_schedule.data_ptr()
                        if profile_schedule is not None
                        else expert_frequency.data_ptr()
                    ),
                    profile_eids.data_ptr(),
                    (
                        active_expert_storage.data_ptr()
                        if (
                            min_active_experts > 0
                            or max_active_experts is not None
                            or expert_m_reuse
                            or expert_m_reuse_threshold is not None
                        )
                        else num_valid_ids.data_ptr()
                    ),
                    *(
                        (expert_frequency.data_ptr(),)
                        if expert_m_reuse_threshold is not None
                        else ()
                    ),
                    (
                        dx_routes
                        if direct_grouped_dx_routes or use_flat_route_order_dx
                        else dx_sorted
                    ).data_ptr(),
                    *(
                        (sorted_token_ids.data_ptr(), tokens)
                        if direct_grouped_dx_routes
                        else (
                            (sorted_route_ids.data_ptr(), routes)
                            if use_flat_route_order_dx
                            else ()
                        )
                    ),
                    grouped_dx_grid,
                    launch_stream,
                )

        use_e16_dw_dx_overlap = _use_e16_dw_dx_overlap(
            e16_flat_grouped=use_e16_flat_grouped,
            reuse_forward_preactivation=reuse_forward_preactivation,
            flat_identity_dx=use_flat_identity_dx,
            use_grouped_dw1=use_grouped_dw1,
            use_grouped_dx=use_grouped_dx,
            route_policy=route_policy,
        )
        if use_e16_dw_dx_overlap and not _is_current_stream_capturing_conservatively():
            _launch_overlapped_e16_dw1_dx(
                _launch_selected_grouped_dw1,
                _launch_selected_grouped_dx,
                (
                    dz,
                    w1_arg,
                    exact_tile_queue_storage,
                    large_dx_storage,
                    compact_w1_storage,
                    sorted_expert_ids,
                    expert_frequency,
                    num_valid_ids,
                    active_expert_storage,
                    sorted_token_ids,
                    sorted_route_ids,
                    dx_routes,
                    dx_sorted,
                ),
                stream=stream,
                device_index=device_index,
            )
        else:
            if use_grouped_dw1:
                _launch_selected_grouped_dw1(stream)
            if use_grouped_dx:
                _launch_selected_grouped_dx(stream)

        # Each expert owns a disjoint output slice, so no atomics or
        # cross-expert reductions are needed for dW1 or routed dX.
        bias_gradient_reduction = (
            _compile_bias_gradient_reduction(
                projection_size,
                hidden_size,
                compute_dtype,
                device_index,
            )
            if has_bias
            else None
        )
        for expert, start, rows in segments:
            end = start + rows
            if not use_grouped_dw1:
                assert x_sorted is not None
                gemm_a16w16(
                    dz[start:end].transpose(0, 1),
                    x_sorted[start:end],
                    out=dw1[expert],
                    user_kwargs=_GEMM_KWARGS,
                    stream=stream,
                    layout="tn",
                )
            if not use_grouped_dx:
                assert dx_sorted is not None
                gemm_a16w16(
                    dz[start:end],
                    w1_arg[expert],
                    out=dx_sorted[start:end],
                    user_kwargs=_GEMM_KWARGS,
                    stream=stream,
                    layout="nn",
                )
            if bias_gradient_reduction is not None:
                _run_compiled(
                    bias_gradient_reduction,
                    dz[start:end],
                    dy[start:end],
                    db1[expert],
                    db2[expert],
                    rows,
                    stream,
                )

        if flat_routes:
            assert sorted_route_ids is not None
            flat_padded_rows = max_padded if use_e16_flat_grouped else padded_rows
            if not use_fused_da_dscore:
                assert dout_sorted is not None
                assert projection is not None
                route_score_backward = _compile_route_score_backward(
                    hidden_size,
                    compute_dtype,
                    device_index,
                )
                _run_compiled(
                    route_score_backward,
                    dout_sorted,
                    projection,
                    sorted_token_ids,
                    sorted_route_ids,
                    droute_weights,
                    tokens,
                    routes,
                    flat_padded_rows,
                    stream,
                )

            if use_flat_identity_dx:
                # The grouped dX epilogue wrote route row r directly to token
                # row r, so no scatter/reduction remains.
                pass
            elif use_flat_segmented_dx:
                assert token_arg is not None
                assert dx_routes is not None
                reduce_routes = _compile_flat_segmented_dx_reduction(
                    hidden_size,
                    compute_dtype,
                    device_index,
                )
                segmented_grid = max(
                    1,
                    min(_E16_FLAT_SEGMENTED_DX_GRID_CAP, tokens),
                )
                _run_compiled(
                    reduce_routes,
                    dx_routes,
                    token_arg,
                    dx,
                    tokens,
                    routes,
                    segmented_grid,
                    stream,
                )
            else:
                assert dx_accum is not None
                assert dx_sorted is not None
                reduce_routes = _compile_ragged_dx_reduction(
                    hidden_size,
                    compute_dtype,
                    device_index,
                )
                output_elements = tokens * hidden_size
                scatter_elements = flat_padded_rows * hidden_size
                clear_grid = max(
                    1, (output_elements + _BLOCK_THREADS - 1) // _BLOCK_THREADS
                )
                scatter_grid = max(
                    1, (scatter_elements + _BLOCK_THREADS - 1) // _BLOCK_THREADS
                )
                _run_compiled(
                    reduce_routes,
                    dx_sorted,
                    sorted_token_ids,
                    dx_accum,
                    dx,
                    tokens,
                    flat_padded_rows,
                    clear_grid,
                    scatter_grid,
                    stream,
                )
        else:
            if not use_fused_da_dscore:
                assert projection is not None
                score_backward = _compile_score_backward(
                    hidden_size,
                    topk,
                    compute_dtype,
                    device_index,
                    use_hostless_grouped,
                    use_fused_forward_state_prepare,
                )
                _run_compiled(
                    score_backward,
                    dout_arg if use_fused_forward_state_prepare else dout_sorted,
                    projection,
                    sorted_token_ids,
                    droute_weights,
                    num_valid_ids,
                    tokens,
                    (
                        min(hostless_row_grid_cap, max_padded)
                        if use_hostless_grouped
                        else padded_rows
                    ),
                    stream,
                )

            assert dx_routes is not None
            assert ids_arg is not None
            if not direct_grouped_dx_routes:
                assert dx_sorted is not None
                unsort = _compile_unsort(
                    hidden_size,
                    topk,
                    compute_dtype,
                    device_index,
                    use_hostless_grouped,
                )
                unsort_work = (max_padded if use_hostless_grouped else padded_rows) * (
                    hidden_size // 4
                )
                unsort_grid = max(
                    1, (unsort_work + _BLOCK_THREADS - 1) // _BLOCK_THREADS
                )
                if use_hostless_grouped:
                    unsort_grid = min(hostless_row_grid_cap, unsort_grid)
                _run_compiled(
                    unsort,
                    dx_sorted,
                    sorted_token_ids,
                    dx_routes,
                    num_valid_ids,
                    tokens,
                    padded_rows,
                    unsort_grid,
                    stream,
                )

            reduction_dtype = "f16" if compute_dtype == "fp16" else "bf16"
            reduce = compile_moe_reduction(
                topk=topk, model_dim=hidden_size, dtype_str=reduction_dtype
            )
            _run_compiled(
                reduce,
                _ptr(dx_routes),
                _ptr(dx),
                _ptr(expert_frequency),
                _ptr(ids_arg),
                tokens,
                stream,
            )

    result = (dx, dw1, dw2, droute_weights)
    if has_bias:
        return (*result, db1, db2)
    return result


def sonic_moe_backward(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    grad_output: torch.Tensor,
    config: "SonicMoEConfig",
    *,
    b1: torch.Tensor | None = None,
    b2: torch.Tensor | None = None,
    interleaved_w1: bool = False,
    forward_state: object | None = None,
) -> tuple[torch.Tensor, ...]:
    """Differentiate dense BF16/FP16 fixed-K SonicMoE, optionally with bias.

    Parameters use logical, expert-major weights: ``w1[E, 2I, H]`` for GLU
    activations, ``w1[E, I, H]`` for pointwise activations, and
    ``w2[E, H, I]``. By default GLU W1 rows are ``[g0..gI, u0..uI]``. Setting
    ``interleaved_w1=True`` instead consumes W1 (and optional B1) in native
    ``[g0, u0, g1, u1, ...]`` row order and returns ``dw1``/``db1`` in that
    same order. Routing tensors are ``topk_ids[int32, T, K]`` and
    ``topk_weights[float32, T, K]``. Without bias, the returned tuple is
    ``(dx, dw1, dw2, dtopk_weights)``. When ``b1`` and ``b2`` are supplied,
    ``(db1, db2)`` are appended. Tensor and bias gradients preserve the A16
    input dtype; routing-score gradients use FP32.

    ``forward_state`` may be the invocation-owned state returned by SonicMoE's
    BF16 SwiGLU fixed-K training forward.  Its route-order preactivation is
    gathered directly into backward sorter order, skipping W1 recomputation.
    Passing ``None`` retains the standalone recompute path.

    State validation is structural.  The state must come from the official
    training forward for this exact ``hidden_states``, ``w1``, ``topk_ids``,
    and optional ``b1`` invocation.  As with the hot-path expert-id value
    contract, proving that semantic identity would require synchronization or
    retaining/hash-reading large inputs and is therefore an unchecked caller
    precondition.

    Expert ids must be in range and distinct within each token. As in the
    inference fixed-K path, value validation is an unchecked hot-path
    precondition so the only synchronization is the bring-up implementation's
    expert-frequency copy used for per-expert GEMM dispatch.
    """

    dimensions = _validate_backward_inputs(
        hidden_states,
        w1,
        w2,
        topk_ids,
        topk_weights,
        grad_output,
        config,
        b1,
        b2,
        interleaved_w1,
    )
    forward_state_data = (
        None
        if forward_state is None
        else _validate_forward_state(
            forward_state,
            hidden_states,
            config,
            interleaved_w1,
            b1 is not None,
        )
    )
    return _sonic_moe_backward_impl(
        hidden_states,
        w1,
        w2,
        topk_ids,
        topk_weights,
        grad_output,
        config,
        token_indices=None,
        dimensions=dimensions,
        b1=b1,
        b2=b2,
        interleaved_w1=interleaved_w1,
        forward_state_data=forward_state_data,
    )


def sonic_moe_backward_routes(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    token_indices: torch.Tensor,
    expert_indices: torch.Tensor,
    route_weights: torch.Tensor,
    grad_output: torch.Tensor,
    config: "SonicMoEConfig",
    *,
    b1: torch.Tensor | None = None,
    b2: torch.Tensor | None = None,
    interleaved_w1: bool = False,
    forward_state: object | None = None,
    token_indices_sorted: bool = False,
    token_indices_identity: bool = False,
    route_policy_size: int | None = None,
) -> tuple[torch.Tensor, ...]:
    """Differentiate SonicMoE over a flat variable-count route list.

    ``token_indices``, ``expert_indices``, and ``route_weights`` are contiguous
    ``[R]`` tensors with int32, int32, and float32 dtype. Every route is tracked
    by its original position, so duplicate ``(token, expert)`` pairs receive
    independent score gradients. Tokens may have zero routes and ``R`` may be
    zero. The result contract matches :func:`sonic_moe_backward`, with
    ``droute_weights`` replacing the fixed-K score gradient.
    ``interleaved_w1`` has the same GLU W1/B1 input and gradient-layout
    contract as :func:`sonic_moe_backward`.

    ``forward_state`` may be the invocation-owned route-order state returned
    by :meth:`SonicMoE.forward_routes_training`.  State reuse is currently
    restricted to the audited bias-free BF16/SwiGLU E16 training bucket and
    removes both forward-projection recomputations from backward.  Newer state
    objects also carry invocation-owned sorter metadata; when all six tensors
    validate, backward reuses them and skips the ragged sorter.  Older or
    partial states safely retain the established backward-owned sort.  Every
    non-empty retained E16 state uses the same exact-queue dataflow, including
    route counts below one sorter tile.

    Set ``token_indices_sorted=True`` only when ``token_indices`` is known to
    be nondecreasing.  The audited E16 retained-state path then uses a
    segmented dX reducer; the default supports arbitrary route order through
    the established atomic scatter.

    ``token_indices_identity=True`` is the stronger contract
    ``routes == tokens`` and ``token_indices[r] == r``.  It lets grouped dX
    write its final token-major result directly.  A state produced by
    ``forward_routes_training(..., token_indices_identity=True)`` carries this
    flag automatically, so callers normally do not need to repeat it.

    ``route_policy_size`` optionally supplies a rank-shared tile/algorithm
    policy floor.  It affects only finite kernel-policy classes, never tensor
    bounds or runtime grids, and an unexpectedly larger local route count
    promotes itself automatically.  When a retained state carries the policy
    from forward, backward reuses it; an explicitly repeated hint must select
    the same effective class.

    Token and expert ids must be in range. Value validation remains an unchecked
    hot-path precondition; the compatibility adapter validates it before launch.
    """

    if not isinstance(token_indices_sorted, bool):
        raise TypeError(
            "token_indices_sorted must be bool, got "
            f"{type(token_indices_sorted).__name__}"
        )
    if not isinstance(token_indices_identity, bool):
        raise TypeError(
            "token_indices_identity must be bool, got "
            f"{type(token_indices_identity).__name__}"
        )

    validated = _validate_backward_route_inputs(
        hidden_states,
        w1,
        w2,
        token_indices,
        expert_indices,
        route_weights,
        grad_output,
        config,
        b1,
        b2,
        interleaved_w1,
    )
    if forward_state is None:
        forward_state_data = None
        forward_sorter_metadata = None
        state_token_indices_identity = False
        select_e16_route_policy(validated[4], route_policy_size)
    else:
        (
            route_preactivation,
            producer_stream,
            ready_event,
            forward_sorter_metadata,
            state_token_indices_identity,
            state_route_policy_size,
        ) = _validate_routes_forward_state(
            forward_state,
            hidden_states,
            validated[4],
            config,
            interleaved_w1,
            b1 is not None,
        )
        if route_policy_size is None:
            route_policy_size = state_route_policy_size
        elif state_route_policy_size is not None and (
            select_e16_route_policy(validated[4], route_policy_size)
            != select_e16_route_policy(validated[4], state_route_policy_size)
        ):
            raise ValueError(
                "route_policy_size must select the same policy as forward_state"
            )
        forward_state_data = (
            route_preactivation,
            producer_stream,
            ready_event,
        )
    effective_token_indices_identity = (
        token_indices_identity or state_token_indices_identity
    )
    if effective_token_indices_identity and validated[0] != validated[4]:
        raise ValueError(
            "token_indices_identity requires routes == tokens, got "
            f"{validated[4]} and {validated[0]}"
        )
    return _sonic_moe_backward_impl(
        hidden_states,
        w1,
        w2,
        expert_indices,
        route_weights,
        grad_output,
        config,
        token_indices=token_indices,
        dimensions=validated[:4],
        b1=b1,
        b2=b2,
        interleaved_w1=interleaved_w1,
        forward_state_data=forward_state_data,
        forward_sorter_metadata=forward_sorter_metadata,
        token_indices_sorted=token_indices_sorted,
        token_indices_identity=effective_token_indices_identity,
        route_policy_size=route_policy_size,
    )


def sonic_moe_backward_expert_major(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    route_weights: torch.Tensor,
    grad_output: torch.Tensor,
    config: "SonicMoEConfig",
    *,
    forward_state: object,
    b1: torch.Tensor | None = None,
    b2: torch.Tensor | None = None,
    interleaved_w1: bool = False,
    route_policy_size: int | None = None,
) -> tuple[torch.Tensor, ...]:
    """Differentiate retained E16 expert-major identity routes without IDs.

    The matching training forward already retained the padded route metadata
    needed by every backward kernel.  Consequently neither an identity
    ``arange`` nor a repeated expert-id vector is part of this API.  Unlike the
    compatibility route entry point, this function fails closed instead of
    falling back to a sorter when the state is incomplete or was not produced
    under the expert-major identity contract.
    """

    validated = _validate_backward_route_inputs(
        hidden_states,
        w1,
        w2,
        None,
        None,
        route_weights,
        grad_output,
        config,
        b1,
        b2,
        interleaved_w1,
        implicit_identity_routes=True,
    )
    (
        route_preactivation,
        producer_stream,
        ready_event,
        forward_sorter_metadata,
        state_token_indices_identity,
        state_route_policy_size,
    ) = _validate_routes_forward_state(
        forward_state,
        hidden_states,
        validated[4],
        config,
        interleaved_w1,
        b1 is not None,
    )
    state_expert_major = getattr(forward_state, "expert_major", False)
    if type(state_expert_major) is not bool:
        raise TypeError(
            "forward_state.expert_major must be bool, got "
            f"{type(state_expert_major).__name__}"
        )
    if (
        not state_expert_major
        or not state_token_indices_identity
        or forward_sorter_metadata is None
    ):
        raise ValueError(
            "implicit expert-major backward requires a complete retained state "
            "with expert_major=True and token_indices_identity=True"
        )
    if route_policy_size is None:
        route_policy_size = state_route_policy_size
    elif state_route_policy_size is not None and (
        select_e16_route_policy(validated[4], route_policy_size)
        != select_e16_route_policy(validated[4], state_route_policy_size)
    ):
        raise ValueError(
            "route_policy_size must select the same policy as forward_state"
        )

    return _sonic_moe_backward_impl(
        hidden_states,
        w1,
        w2,
        None,
        route_weights,
        grad_output,
        config,
        token_indices=None,
        dimensions=validated[:4],
        b1=b1,
        b2=b2,
        interleaved_w1=interleaved_w1,
        forward_state_data=(
            route_preactivation,
            producer_stream,
            ready_event,
        ),
        forward_sorter_metadata=forward_sorter_metadata,
        token_indices_identity=True,
        route_policy_size=route_policy_size,
        implicit_identity_routes=True,
    )


__all__ = [
    "sonic_moe_backward",
    "sonic_moe_backward_expert_major",
    "sonic_moe_backward_routes",
]
