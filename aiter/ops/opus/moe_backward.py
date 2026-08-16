# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Python binding and autograd attachment API for Opus MoE backward.

The module follows the Opus stage-1 wrapper boundary: raw JIT bindings,
output/workspace allocation, validation, and public native wrappers. Forward
routing and projection kernels own out/A_sorted/Z_sorted; this module only
attaches the two backward Functions to that saved state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import torch
from torch import Tensor
from torch.autograd.function import once_differentiable

from ...jit.core import compile_ops


@dataclass(frozen=True)
class OpusMoeFixedMetadata:
    """Versioned fixed-top-k sorting metadata emitted by forward."""

    layout_version: ClassVar[int] = 1

    sorted_token_ids: Tensor
    sorted_expert_ids: Tensor
    num_valid_ids: Tensor
    reverse_sorted: Tensor
    expert_padded_offsets: Tensor
    block_m: int


@dataclass(frozen=True)
class OpusMoeVarlenMetadata:
    """Versioned compact-route sorting metadata emitted by forward."""

    layout_version: ClassVar[int] = 1

    sorted_route_ids: Tensor
    sorted_expert_ids: Tensor
    num_valid_ids: Tensor
    route_to_token: Tensor
    token_route_offsets: Tensor
    expert_padded_offsets: Tensor
    block_m: int


@dataclass(frozen=True)
class OpusMoeBackwardOutput:
    """Complete expert-path gradients and reusable K1 intermediates."""

    d_x: Tensor
    d_w1: Tensor
    d_w2: Tensor
    d_scores: Tensor
    d_z_sorted: Tensor
    a_scaled: Tensor
    d_b1: Tensor | None = None
    d_b2: Tensor | None = None


@dataclass(frozen=True)
class OpusMoeDownBackwardOutput:
    d_z_sorted: Tensor
    a_scaled: Tensor
    d_scores: Tensor
    d_scores_workspace: Tensor


@dataclass(frozen=True)
class OpusMoeRouteBackwardOutput:
    d_x_route: Tensor
    d_x: Tensor

    @property
    def d_x_route_sorted(self) -> Tensor:
        """Compatibility alias; the workspace is logical [token, slot] order."""

        return self.d_x_route


def _select_internal_fixed_route_pair(
    d_z_sorted: Tensor, w1: Tensor, topk: int
) -> tuple[int, int]:
    """Select the internal K2/K3 workspace layout without exact-shape rules."""

    l2_friendly_bytes = 128 * 1024 * 1024
    b_first_min_routes = 250_000
    m5_min_average_routes = 1536
    bn512_min_dz_bytes = 1024 * 1024 * 1024
    working_set_bytes = (
        d_z_sorted.numel() * d_z_sorted.element_size()
        + w1.numel() * w1.element_size()
    )
    enough_route_tiles = d_z_sorted.shape[0] >= 4 * 32
    if (
        topk >= 4
        and w1.shape[2] > 128
        and enough_route_tiles
        and working_set_bytes > l2_friendly_bytes
    ):
        # Match the native full-pipeline selector: the BN256 route kernel
        # halves dZ rereads when D is tile aligned, while BN128 remains the
        # remainder-safe large-working-set path.
        if w1.shape[2] % 256 == 0:
            average_padded_routes = d_z_sorted.shape[0] // w1.shape[0]
            if average_padded_routes >= m5_min_average_routes:
                route_dz_bytes = (
                    d_z_sorted.numel() * d_z_sorted.element_size()
                )
                if (
                    w1.shape[2] % 512 == 0
                    and route_dz_bytes >= bn512_min_dz_bytes
                ):
                    output_n_tiles = w1.shape[2] // 512
                    return (19 if output_n_tiles in (4, 8) else 18), 1
                return 16, 1
            # Once the sorted route stream is this large, issue the wider W1
            # transfer before dZ so its vmcnt-critical tail enters the memory
            # pipeline earlier.  Below the measured crossover retain the
            # original A-first BN256 kernel.
            route_dx_kernel_id = (
                15 if d_z_sorted.shape[0] >= b_first_min_routes else 14
            )
            return route_dx_kernel_id, 1
        return 11, 1
    return -1, -1


def _select_fixed_down_block_n(
    inter_dim: int,
    sorted_expert_ids: Tensor,
    expert_padded_offsets: Tensor,
    kernel_id: int,
) -> int:
    """Mirror native K1 dispatch so workspace shape and kernel stay coupled."""

    if kernel_id in (11, 12, 13, 14, 15, 16):
        if inter_dim % 256 != 0:
            raise ValueError("BN256 down kernels require I divisible by 256")
        return 256
    if kernel_id != -1 or inter_dim < 512 or inter_dim % 256 != 0:
        return 128

    # BN256/M6 groups six adjacent sorter tiles.  The compact launcher adds
    # one boundary group per expert, rounds the result to four-group cohorts,
    # then launches one CTA per 256 intermediate columns.  A wider N tile
    # wins only once this geometry exposes enough work for all gfx950 CUs.
    groups = (sorted_expert_ids.numel() + 5) // 6
    groups += expert_padded_offsets.numel() - 1
    padded_groups = ((groups + 3) // 4) * 4
    candidate_ctas = padded_groups * (inter_dim // 256)
    return 256 if candidate_ctas >= 384 else 128


def _select_saved_a_scaled_down_kernel(
    inter_dim: int,
    sorted_capacity: int,
    sorted_expert_ids: Tensor,
    expert_padded_offsets: Tensor,
    kernel_id: int,
) -> int:
    """Map a BN256 K1 policy to its no-a_scaled-write counterpart."""

    if kernel_id in (13, 14, 15, 16):
        return kernel_id
    if kernel_id == 11:
        return 13
    if kernel_id == 12:
        return 14
    if kernel_id != -1:
        raise ValueError(
            "saved_a_scaled currently supports auto or BN256 down kernels"
        )
    block_n = _select_fixed_down_block_n(
        inter_dim, sorted_expert_ids, expert_padded_offsets, kernel_id
    )
    if block_n != 256:
        raise ValueError(
            "saved_a_scaled currently requires the general BN256 K1 geometry"
        )
    # Long BN256 streams use split-N64 W2 staging and pipeline the next
    # route-group Z transfer under the current group's activation epilogue.
    # Keep kid13 for short streams where deferred-Z scheduling is not selected.
    return 16 if sorted_capacity >= 65_536 else 13


def _reject_saved_a_scaled_down_without_cache(kernel_id: int) -> None:
    """Keep no-store K1 instances away from an uninitialized output cache."""

    if int(kernel_id) in (13, 14, 15, 16):
        raise ValueError(
            "K1 kernel_id=13/14/15/16 requires saved_a_scaled"
        )


def _validate_saved_a_scaled(
    saved_a_scaled: Tensor,
    sorted_capacity: int,
    inter_dim: int,
    device: torch.device,
) -> None:
    """Validate the forward cache container; metadata defines live route rows.

    Live rows must contain the BF16-rounded value
    ``route_weight * SwiGLU(z_sorted)`` in expert-sorted route-major order.
    Every sorter-padding row must be exact zero.  The full pipeline uses this
    producer contract to select a K5 instance that reads padded rows directly.
    """

    if saved_a_scaled.dtype != torch.bfloat16:
        raise TypeError("saved_a_scaled must be BF16")
    if saved_a_scaled.shape != (sorted_capacity, inter_dim):
        raise ValueError("saved_a_scaled must have shape [sorted_capacity,I]")
    if saved_a_scaled.device != device or not saved_a_scaled.is_contiguous():
        raise ValueError("saved_a_scaled must be contiguous on the common device")


def _validate_saved_x_sorted(
    saved_x_sorted: Tensor,
    sorted_capacity: int,
    model_dim: int,
    device: torch.device,
) -> None:
    """Validate an optional forward-saved padded expert-sorted X container.

    The ordinary layout stores logical row ``r`` as ``x[token(r)]``.  The
    blocked-G2 layout uses the same allocation shape but is selected through a
    separate explicit ABI bit.  Both producers must materialize sorter padding
    as exact zero because direct K4 consumes every padded reduction row.
    """

    if saved_x_sorted.dtype != torch.bfloat16:
        raise TypeError("saved_x_sorted must be BF16")
    if saved_x_sorted.shape != (sorted_capacity, model_dim):
        raise ValueError(
            "saved_x_sorted must have shape [sorted_capacity,D]"
        )
    if saved_x_sorted.device != device or not saved_x_sorted.is_contiguous():
        raise ValueError(
            "saved_x_sorted must be contiguous on the common device"
        )


def _select_saved_x_dw1_kernel(
    gate_up_dim: int,
    model_dim: int,
    sorted_capacity: int,
    num_experts: int,
    kernel_id: int,
) -> int:
    """Select direct sorted-X K4 from the production grouped-GEMM geometry.

    The cache-backed instance shares kid 13's BM256xBN128xBK32 output tile,
    reverse cohort-4 schedule.  Auto uses the same long-reduction and
    output-grid crossover instead of a model tuple, then enables a third LDS
    stage while two future BK32 tiles can remain in flight.  Explicit kids
    14/15/17/18/19 remain available for family benchmarking.  Kid 19 is an
    explicit-only physical-LDS-layout experiment; auto retains kid 18.
    """

    if gate_up_dim % 256 != 0 or model_dim % 128 != 0:
        raise ValueError(
            "saved_x_sorted K4 requires 2I divisible by 256 and D by 128"
        )
    if kernel_id not in (-1, 14, 15, 17, 18, 19):
        raise ValueError(
            "saved_x_sorted currently supports auto or K4 "
            "kernel_id=14/15/17/18/19"
        )
    if kernel_id in (14, 15, 17, 18, 19):
        return kernel_id

    if num_experts < 2:
        raise ValueError(
            "saved_x_sorted auto requires at least two experts; use "
            "kernel_id=14/15/17/18/19 only for explicit tuning"
        )
    average_padded_routes = sorted_capacity // num_experts
    inter_dim = gate_up_dim // 2
    output_tiles = (
        num_experts * (gate_up_dim // 256) * (model_dim // 128)
    )
    compact_full_residency_grid = (
        1024 <= output_tiles < 1536 and output_tiles % 512 == 0
    )
    if not (
        average_padded_routes >= 3072
        and inter_dim <= 1024
        and (output_tiles >= 1536 or compact_full_residency_grid)
    ):
        raise ValueError(
            "saved_x_sorted auto requires the production long-reduction "
            "BM256 K4 geometry; use kernel_id=14/15/17/18/19 only for explicit "
            "tuning"
        )
    # Long reductions with sufficiently wide output/reduction grids amortize
    # an eager LDS issue window for both K16 fragments.  Issue sorted-X before
    # dZ so the smaller reused operand enters the direct-to-LDS pipeline first.
    # Keep the lower-register schedule for narrow D or I families.
    if inter_dim >= 768 and model_dim >= 1536:
        # Three 24-KiB stages keep two future BK32 tiles in flight.  The
        # extra LDS/VGPR footprint wins through the tested medium/long route
        # family, but very long per-expert reductions favor the two-stage
        # kernel's lower resource footprint.  This is a reduction-geometry
        # boundary (roughly 200 BK32 tiles), not a model-shape special case.
        if average_padded_routes <= 6400:
            return 18
        return 15
    return 14


def _reject_sorted_x_dw1_without_cache(kernel_id: int) -> None:
    """Keep the direct-row K4 instance away from token-major X."""

    if int(kernel_id) in (14, 15, 17, 18, 19, 20, 21):
        raise ValueError(
            "direct sorted-X K4 kernel_id requires saved_x_sorted"
        )


def _select_blocked_g2_full_pipeline(
    inter_dim: int,
    model_dim: int,
    down_kernel_id: int,
    route_dx_kernel_id: int,
    route_reduce_kernel_id: int,
    dw1_kernel_id: int,
) -> tuple[int, int, int, int]:
    """Select the coupled blocked-dZ/blocked-X producer-consumer ABI.

    The policy is defined by tile divisibility and physical-layout contracts,
    not an exact model tuple.  Explicit incompatible kernel requests fail so
    no tensor can be silently reinterpreted with the wrong layout.
    """

    if inter_dim % 256 != 0 or model_dim % 512 != 0:
        raise ValueError(
            "blocked-G2 full pipeline requires I divisible by 256 and "
            "D divisible by 512"
        )
    # K3 kid3 distributes one complete D=2048 row across route-id owners.
    # Other blocked-K2-compatible D tiles retain the general sorted-route
    # reducer rather than constraining the whole cache ABI to one width.
    route_reduce_kernel_id_selected = 3 if model_dim % 2048 == 0 else 1
    selected = (17, 20, route_reduce_kernel_id_selected, 21)
    requested = (
        int(down_kernel_id),
        int(route_dx_kernel_id),
        int(route_reduce_kernel_id),
        int(dw1_kernel_id),
    )
    names = (
        "down_kernel_id",
        "route_dx_kernel_id",
        "route_reduce_kernel_id",
        "dw1_kernel_id",
    )
    for name, value, expected in zip(names, requested, selected):
        if value not in (-1, expected):
            raise ValueError(
                f"blocked-G2 full pipeline requires {name}={expected}, "
                f"got {value}"
            )
    return selected


@dataclass(frozen=True)
class OpusMoeWeightBackwardOutput:
    d_w1: Tensor
    d_w2: Tensor


@dataclass(frozen=True)
class OpusMoeBiasDownBackwardOutput:
    d_scores: Tensor
    d_b2: Tensor


# Raw fixed-routing JIT bindings.


def _gen_opus_moe_router_bwd_fake_tensors(
    d_scores: Tensor,
    scores: Tensor,
    topk_ids: Tensor,
    d_logits: Tensor,
) -> Tensor:
    return d_logits


@compile_ops(
    "module_opus_moe_backward",
    fc_name="opus_moe_router_bwd",
    gen_fake=_gen_opus_moe_router_bwd_fake_tensors,
    develop=True,
)
def _opus_moe_router_bwd_raw(
    d_scores: Tensor,
    scores: Tensor,
    topk_ids: Tensor,
    d_logits: Tensor,
    kernel_id: int,
) -> Tensor: ...


def _gen_opus_moe_db1_bwd_fake_tensors(
    d_z: Tensor,
    sorted_token_ids: Tensor,
    num_valid_ids: Tensor,
    expert_padded_offsets: Tensor,
    d_b1: Tensor,
) -> Tensor:
    return d_b1


@compile_ops(
    "module_opus_moe_backward",
    fc_name="opus_moe_db1_bwd",
    gen_fake=_gen_opus_moe_db1_bwd_fake_tensors,
    develop=True,
)
def _opus_moe_db1_bwd_raw(
    d_z: Tensor,
    sorted_token_ids: Tensor,
    num_valid_ids: Tensor,
    expert_padded_offsets: Tensor,
    d_b1: Tensor,
    token_num: int,
    topk: int,
    block_m: int,
    kernel_id: int,
) -> Tensor: ...


def _gen_opus_moe_bias_down_bwd_fake_tensors(
    d_out: Tensor,
    scores: Tensor,
    b2: Tensor,
    sorted_token_ids: Tensor,
    sorted_expert_ids: Tensor,
    num_valid_ids: Tensor,
    expert_padded_offsets: Tensor,
    d_scores: Tensor,
    d_b2: Tensor,
) -> Tensor:
    return d_b2


@compile_ops(
    "module_opus_moe_backward",
    fc_name="opus_moe_bias_down_bwd",
    gen_fake=_gen_opus_moe_bias_down_bwd_fake_tensors,
    develop=True,
)
def _opus_moe_bias_down_bwd_raw(
    d_out: Tensor,
    scores: Tensor,
    b2: Tensor,
    sorted_token_ids: Tensor,
    sorted_expert_ids: Tensor,
    num_valid_ids: Tensor,
    expert_padded_offsets: Tensor,
    d_scores: Tensor,
    d_b2: Tensor,
    block_m: int,
    kernel_id: int,
) -> Tensor: ...


def _gen_opus_moe_down_bwd_fake_tensors(
    d_out: Tensor,
    z: Tensor,
    w2: Tensor,
    scores: Tensor,
    sorted_token_ids: Tensor,
    sorted_expert_ids: Tensor,
    num_valid_ids: Tensor,
    expert_padded_offsets: Tensor,
    d_scores_workspace: Tensor,
    d_z: Tensor,
    a_scaled: Tensor,
    d_scores: Tensor,
) -> Tensor:
    return d_z


@compile_ops(
    "module_opus_moe_backward",
    fc_name="opus_moe_down_bwd",
    gen_fake=_gen_opus_moe_down_bwd_fake_tensors,
    develop=True,
)
def _opus_moe_down_bwd_raw(
    d_out: Tensor,
    z: Tensor,
    w2: Tensor,
    scores: Tensor,
    sorted_token_ids: Tensor,
    sorted_expert_ids: Tensor,
    num_valid_ids: Tensor,
    expert_padded_offsets: Tensor,
    d_scores_workspace: Tensor,
    d_z: Tensor,
    a_scaled: Tensor,
    d_scores: Tensor,
    block_m: int,
    kernel_id: int,
) -> Tensor: ...


def _gen_opus_moe_route_bwd_fake_tensors(
    d_z: Tensor,
    w1: Tensor,
    sorted_token_ids: Tensor,
    sorted_expert_ids: Tensor,
    num_valid_ids: Tensor,
    expert_padded_offsets: Tensor,
    reverse_sorted: Tensor,
    d_x_route: Tensor,
    d_x: Tensor,
) -> Tensor:
    return d_x


@compile_ops(
    "module_opus_moe_backward",
    fc_name="opus_moe_route_bwd",
    gen_fake=_gen_opus_moe_route_bwd_fake_tensors,
    develop=True,
)
def _opus_moe_route_bwd_raw(
    d_z: Tensor,
    w1: Tensor,
    sorted_token_ids: Tensor,
    sorted_expert_ids: Tensor,
    num_valid_ids: Tensor,
    expert_padded_offsets: Tensor,
    reverse_sorted: Tensor,
    d_x_route: Tensor,
    d_x: Tensor,
    block_m: int,
    route_dx_kernel_id: int,
    route_reduce_kernel_id: int,
) -> Tensor: ...


def _gen_opus_moe_weight_bwd_fake_tensors(
    x: Tensor,
    d_out: Tensor,
    d_z: Tensor,
    a_scaled: Tensor,
    sorted_token_ids: Tensor,
    num_valid_ids: Tensor,
    expert_padded_offsets: Tensor,
    d_w1: Tensor,
    d_w2: Tensor,
) -> Tensor:
    return d_w1


@compile_ops(
    "module_opus_moe_backward",
    fc_name="opus_moe_weight_bwd",
    gen_fake=_gen_opus_moe_weight_bwd_fake_tensors,
    develop=True,
)
def _opus_moe_weight_bwd_raw(
    x: Tensor,
    d_out: Tensor,
    d_z: Tensor,
    a_scaled: Tensor,
    sorted_token_ids: Tensor,
    num_valid_ids: Tensor,
    expert_padded_offsets: Tensor,
    d_w1: Tensor,
    d_w2: Tensor,
    topk: int,
    block_m: int,
    dw1_kernel_id: int,
    dw2_kernel_id: int,
) -> Tensor: ...


def _gen_opus_moe_dw1_bwd_fake_tensors(
    x: Tensor,
    d_z: Tensor,
    sorted_token_ids: Tensor,
    num_valid_ids: Tensor,
    expert_padded_offsets: Tensor,
    d_w1: Tensor,
) -> Tensor:
    return d_w1


@compile_ops(
    "module_opus_moe_backward",
    fc_name="opus_moe_dw1_bwd",
    gen_fake=_gen_opus_moe_dw1_bwd_fake_tensors,
    develop=True,
)
def _opus_moe_dw1_bwd_raw(
    x: Tensor,
    d_z: Tensor,
    sorted_token_ids: Tensor,
    num_valid_ids: Tensor,
    expert_padded_offsets: Tensor,
    d_w1: Tensor,
    topk: int,
    block_m: int,
    kernel_id: int,
) -> Tensor: ...


def _gen_opus_moe_dw2_bwd_fake_tensors(
    d_out: Tensor,
    a_scaled: Tensor,
    sorted_token_ids: Tensor,
    num_valid_ids: Tensor,
    expert_padded_offsets: Tensor,
    d_w2: Tensor,
) -> Tensor:
    return d_w2


@compile_ops(
    "module_opus_moe_backward",
    fc_name="opus_moe_dw2_bwd",
    gen_fake=_gen_opus_moe_dw2_bwd_fake_tensors,
    develop=True,
)
def _opus_moe_dw2_bwd_raw(
    d_out: Tensor,
    a_scaled: Tensor,
    sorted_token_ids: Tensor,
    num_valid_ids: Tensor,
    expert_padded_offsets: Tensor,
    d_w2: Tensor,
    topk: int,
    block_m: int,
    kernel_id: int,
) -> Tensor: ...


def _gen_opus_moe_gather_x_blocked_g2_fake_tensors(
    x: Tensor,
    sorted_token_ids: Tensor,
    num_valid_ids: Tensor,
    x_sorted: Tensor,
) -> Tensor:
    return x_sorted


@compile_ops(
    "module_opus_moe_backward",
    fc_name="opus_moe_gather_x_blocked_g2",
    gen_fake=_gen_opus_moe_gather_x_blocked_g2_fake_tensors,
    develop=True,
)
def _opus_moe_gather_x_blocked_g2_raw(
    x: Tensor,
    sorted_token_ids: Tensor,
    num_valid_ids: Tensor,
    x_sorted: Tensor,
    block_m: int,
) -> Tensor: ...


def _gen_opus_moe_full_bwd_fake_tensors(
    d_out: Tensor,
    x: Tensor,
    x_dw1: Tensor,
    z: Tensor,
    w1: Tensor,
    w2: Tensor,
    scores: Tensor,
    sorted_token_ids: Tensor,
    sorted_expert_ids: Tensor,
    num_valid_ids: Tensor,
    reverse_sorted: Tensor,
    expert_padded_offsets: Tensor,
    d_scores_workspace: Tensor,
    d_z: Tensor,
    a_scaled: Tensor,
    d_scores: Tensor,
    d_x_route: Tensor,
    d_x: Tensor,
    d_w1: Tensor,
    d_w2: Tensor,
) -> Tensor:
    return d_x


@compile_ops(
    "module_opus_moe_backward",
    fc_name="opus_moe_full_bwd",
    gen_fake=_gen_opus_moe_full_bwd_fake_tensors,
    develop=True,
)
def _opus_moe_full_bwd_raw(
    d_out: Tensor,
    x: Tensor,
    x_dw1: Tensor,
    z: Tensor,
    w1: Tensor,
    w2: Tensor,
    scores: Tensor,
    sorted_token_ids: Tensor,
    sorted_expert_ids: Tensor,
    num_valid_ids: Tensor,
    reverse_sorted: Tensor,
    expert_padded_offsets: Tensor,
    d_scores_workspace: Tensor,
    d_z: Tensor,
    a_scaled: Tensor,
    d_scores: Tensor,
    d_x_route: Tensor,
    d_x: Tensor,
    d_w1: Tensor,
    d_w2: Tensor,
    block_m: int,
    down_kernel_id: int,
    route_dx_kernel_id: int,
    route_reduce_kernel_id: int,
    dw1_kernel_id: int,
    dw2_kernel_id: int,
    x_dw1_blocked_g2: bool = False,
) -> Tensor: ...


@compile_ops(
    "module_opus_moe_backward",
    fc_name="opus_moe_full_bwd_trusted",
    gen_fake=_gen_opus_moe_full_bwd_fake_tensors,
    develop=True,
)
def _opus_moe_full_bwd_trusted_raw(
    d_out: Tensor,
    x: Tensor,
    x_dw1: Tensor,
    z: Tensor,
    w1: Tensor,
    w2: Tensor,
    scores: Tensor,
    sorted_token_ids: Tensor,
    sorted_expert_ids: Tensor,
    num_valid_ids: Tensor,
    reverse_sorted: Tensor,
    expert_padded_offsets: Tensor,
    d_scores_workspace: Tensor,
    d_z: Tensor,
    a_scaled: Tensor,
    d_scores: Tensor,
    d_x_route: Tensor,
    d_x: Tensor,
    d_w1: Tensor,
    d_w2: Tensor,
    block_m: int,
    down_kernel_id: int,
    route_dx_kernel_id: int,
    route_reduce_kernel_id: int,
    dw1_kernel_id: int,
    dw2_kernel_id: int,
    x_dw1_blocked_g2: bool = False,
) -> Tensor:
    """Internal launch after the complete reusable tensor contract is checked."""

    ...


# Checked single-family wrappers used by tests, tuning, and autograd pruning.


def opus_moe_down_backward(
    d_out: Tensor,
    z_sorted: Tensor,
    w2: Tensor,
    topk_weights: Tensor,
    sorted_token_ids: Tensor,
    sorted_expert_ids: Tensor,
    num_valid_ids: Tensor,
    expert_padded_offsets: Tensor,
    *,
    block_m: int,
    kernel_id: int = -1,
    saved_a_scaled: Tensor | None = None,
) -> OpusMoeDownBackwardOutput:
    """Launch the first gfx950 BF16 K1 family on sorted-major saved state.

    ``saved_a_scaled`` is an optional non-differentiable forward cache with
    shape ``[sorted_capacity, I]``.  When supplied, K1 skips recomputing and
    writing ``route_weight * SwiGLU(z_sorted)``; the returned ``a_scaled`` is
    the same tensor and K5 consumes it directly.  Only routed rows identified
    by the sorting metadata are live; every padding row must be exact zero.
    """

    if d_out.dtype != torch.bfloat16 or any(
        tensor.dtype != torch.bfloat16 for tensor in (z_sorted, w2)
    ):
        raise TypeError("Opus K1 currently requires BF16 d_out, z_sorted, and w2")
    if topk_weights.dtype != torch.float32:
        raise TypeError("topk_weights must be float32")
    token_num, model_dim = d_out.shape
    num_experts, w2_model_dim, inter_dim = w2.shape
    del num_experts
    if w2_model_dim != model_dim:
        raise ValueError("w2 model dimension must match d_out")
    if z_sorted.shape != (sorted_token_ids.numel(), 2 * inter_dim):
        raise ValueError("z_sorted must have shape [sorted_capacity,2I]")
    if topk_weights.shape[0] != token_num:
        raise ValueError("topk_weights token dimension must match d_out")
    if model_dim % 64 != 0 or inter_dim % 128 != 0:
        raise ValueError(
            "the first K1 instance requires D divisible by 64 and I divisible by 128"
        )

    route_num = int(topk_weights.numel())
    if saved_a_scaled is not None:
        _validate_saved_a_scaled(
            saved_a_scaled, z_sorted.shape[0], inter_dim, z_sorted.device
        )
        kernel_id = _select_saved_a_scaled_down_kernel(
            inter_dim,
            z_sorted.shape[0],
            sorted_expert_ids,
            expert_padded_offsets,
            kernel_id,
        )
    else:
        _reject_saved_a_scaled_down_without_cache(kernel_id)
    down_block_n = _select_fixed_down_block_n(
        inter_dim, sorted_expert_ids, expert_padded_offsets, kernel_id
    )
    d_scores_parts = (inter_dim + down_block_n - 1) // down_block_n
    d_z = torch.empty_like(z_sorted)
    a_scaled = (
        saved_a_scaled
        if saved_a_scaled is not None
        else torch.empty(
            (z_sorted.shape[0], inter_dim),
            dtype=z_sorted.dtype,
            device=z_sorted.device,
        )
    )
    d_scores = torch.empty_like(topk_weights, dtype=torch.float32)
    d_scores_workspace = torch.empty(
        (route_num, d_scores_parts) if d_scores_parts > 1 else (0, 0),
        dtype=torch.float32,
        device=d_out.device,
    )
    _opus_moe_down_bwd_raw(
        d_out,
        z_sorted,
        w2,
        topk_weights,
        sorted_token_ids,
        sorted_expert_ids,
        num_valid_ids,
        expert_padded_offsets,
        d_scores_workspace,
        d_z,
        a_scaled,
        d_scores,
        int(block_m),
        int(kernel_id),
    )
    return OpusMoeDownBackwardOutput(
        d_z_sorted=d_z,
        a_scaled=a_scaled,
        d_scores=d_scores,
        d_scores_workspace=d_scores_workspace,
    )


def opus_moe_router_backward(
    d_scores: Tensor,
    topk_weights: Tensor,
    topk_ids: Tensor,
    num_experts: int,
    *,
    kernel_id: int = -1,
) -> Tensor:
    """Fuse selected-softmax backward and scatter to ``[T,E]`` logits.

    This API implements only softmax over the already selected top-k logits.
    ``topk_ids`` are the fixed discrete routes from forward and are never
    recomputed.  Full-softmax-before-topk has different gradient semantics and
    intentionally requires a separate future entry.
    """

    if d_scores.dtype != torch.float32 or topk_weights.dtype != torch.float32:
        raise TypeError("router backward requires float32 d_scores and topk_weights")
    if topk_ids.dtype != torch.int32:
        raise TypeError("topk_ids must be int32")
    if d_scores.ndim != 2 or topk_weights.ndim != 2 or topk_ids.ndim != 2:
        raise ValueError("d_scores, topk_weights, and topk_ids must have rank 2")
    if d_scores.shape != topk_weights.shape or topk_ids.shape != topk_weights.shape:
        raise ValueError("d_scores, topk_weights, and topk_ids must share [T,K]")
    if topk_weights.shape[1] not in (1, 2, 4, 8):
        raise ValueError("selected-softmax router backward supports K in {1,2,4,8}")
    num_experts = int(num_experts)
    if num_experts < topk_weights.shape[1]:
        raise ValueError("num_experts must be at least topk")
    if not (d_scores.is_cuda and topk_weights.is_cuda and topk_ids.is_cuda):
        raise ValueError("router backward tensors must be on a GPU")
    if not (
        d_scores.device == topk_weights.device == topk_ids.device
    ):
        raise ValueError("router backward tensors must be on the same GPU")

    d_scores = d_scores.contiguous()
    topk_weights = topk_weights.contiguous()
    topk_ids = topk_ids.contiguous()
    d_logits = torch.empty(
        (topk_weights.shape[0], num_experts),
        dtype=torch.float32,
        device=topk_weights.device,
    )
    _opus_moe_router_bwd_raw(
        d_scores,
        topk_weights,
        topk_ids,
        d_logits,
        int(kernel_id),
    )
    return d_logits


def opus_moe_db1_backward(
    d_z_sorted: Tensor,
    sorted_token_ids: Tensor,
    num_valid_ids: Tensor,
    expert_padded_offsets: Tensor,
    *,
    token_num: int,
    topk: int,
    block_m: int,
    kernel_id: int = -1,
) -> Tensor:
    """Reduce sorted route ``dZ`` into BF16 expert ``db1``."""

    if d_z_sorted.dtype != torch.bfloat16:
        raise TypeError("db1 backward requires BF16 d_z_sorted")
    if d_z_sorted.ndim != 2 or d_z_sorted.shape[1] % 2 != 0:
        raise ValueError("d_z_sorted must have shape [sorted_capacity,2I]")
    if sorted_token_ids.dtype != torch.int32:
        raise TypeError("sorted_token_ids must be int32")
    if num_valid_ids.dtype != torch.int32:
        raise TypeError("num_valid_ids must be int32")
    if expert_padded_offsets.dtype != torch.int32:
        raise TypeError("expert_padded_offsets must be int32")
    if d_z_sorted.shape[0] != sorted_token_ids.numel():
        raise ValueError("d_z_sorted capacity must match sorted_token_ids")
    num_experts = int(expert_padded_offsets.numel()) - 1
    if num_experts <= 0:
        raise ValueError("expert_padded_offsets must contain E+1 entries")
    d_b1 = torch.empty(
        (num_experts, d_z_sorted.shape[1]),
        dtype=torch.bfloat16,
        device=d_z_sorted.device,
    )
    _opus_moe_db1_bwd_raw(
        d_z_sorted.contiguous(),
        sorted_token_ids.contiguous(),
        num_valid_ids.contiguous(),
        expert_padded_offsets.contiguous(),
        d_b1,
        int(token_num),
        int(topk),
        int(block_m),
        int(kernel_id),
    )
    return d_b1


def opus_moe_bias_down_backward(
    d_out: Tensor,
    topk_weights: Tensor,
    b2: Tensor,
    sorted_token_ids: Tensor,
    sorted_expert_ids: Tensor,
    num_valid_ids: Tensor,
    expert_padded_offsets: Tensor,
    d_scores: Tensor,
    *,
    block_m: int,
    kernel_id: int = -1,
) -> OpusMoeBiasDownBackwardOutput:
    """Add the output-bias term to ``d_scores`` and reduce expert ``db2``.

    ``d_scores`` is updated in-place so this launch can directly extend K1's
    existing FP32 result without an intermediate tensor.
    """

    if d_out.dtype != torch.bfloat16 or b2.dtype != torch.bfloat16:
        raise TypeError("bias down backward requires BF16 d_out and b2")
    if topk_weights.dtype != torch.float32 or d_scores.dtype != torch.float32:
        raise TypeError("topk_weights and d_scores must be float32")
    if d_out.ndim != 2 or b2.ndim != 2:
        raise ValueError("d_out and b2 must have shapes [T,D] and [E,D]")
    if b2.shape[1] != d_out.shape[1]:
        raise ValueError("b2 model dimension must match d_out")
    if topk_weights.shape != d_scores.shape:
        raise ValueError("topk_weights and d_scores must share [T,K]")
    if topk_weights.shape[0] != d_out.shape[0]:
        raise ValueError("topk_weights token dimension must match d_out")
    if expert_padded_offsets.numel() != b2.shape[0] + 1:
        raise ValueError("expert_padded_offsets must contain E+1 entries")
    for name, tensor in (
        ("sorted_token_ids", sorted_token_ids),
        ("sorted_expert_ids", sorted_expert_ids),
        ("num_valid_ids", num_valid_ids),
        ("expert_padded_offsets", expert_padded_offsets),
    ):
        if tensor.dtype != torch.int32:
            raise TypeError(f"{name} must be int32")
    d_b2 = torch.empty_like(b2)
    _opus_moe_bias_down_bwd_raw(
        d_out.contiguous(),
        topk_weights.contiguous(),
        b2.contiguous(),
        sorted_token_ids.contiguous(),
        sorted_expert_ids.contiguous(),
        num_valid_ids.contiguous(),
        expert_padded_offsets.contiguous(),
        d_scores,
        d_b2,
        int(block_m),
        int(kernel_id),
    )
    return OpusMoeBiasDownBackwardOutput(d_scores=d_scores, d_b2=d_b2)


class _OpusMoeSelectedSoftmaxFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        router_logits: Tensor,
        topk_ids: Tensor,
        kernel_id: int,
    ) -> Tensor:
        selected_logits = torch.gather(
            router_logits,
            1,
            topk_ids.to(torch.int64),
        )
        topk_weights = torch.softmax(selected_logits, dim=-1).contiguous()
        ctx.save_for_backward(topk_weights, topk_ids)
        ctx.num_experts = int(router_logits.shape[1])
        ctx.kernel_id = int(kernel_id)
        ctx.set_materialize_grads(False)
        return topk_weights

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_scores: Tensor | None):
        if grad_scores is None:
            return None, None, None
        topk_weights, topk_ids = ctx.saved_tensors
        d_logits = opus_moe_router_backward(
            grad_scores.contiguous(),
            topk_weights,
            topk_ids,
            ctx.num_experts,
            kernel_id=ctx.kernel_id,
        )
        return d_logits, None, None


def opus_moe_selected_softmax(
    router_logits: Tensor,
    topk_ids: Tensor,
    *,
    kernel_id: int = -1,
) -> Tensor:
    """Selected-top-k softmax with the fused Opus backward.

    The caller supplies the forward-selected route ids.  This keeps routing
    decisions identical between forward and backward and differentiates only
    the selected-softmax values.
    """

    if router_logits.dtype != torch.float32:
        raise TypeError("router_logits must be float32")
    if topk_ids.dtype != torch.int32:
        raise TypeError("topk_ids must be int32")
    if router_logits.ndim != 2 or topk_ids.ndim != 2:
        raise ValueError("router_logits and topk_ids must have rank 2")
    if router_logits.shape[0] != topk_ids.shape[0]:
        raise ValueError("router_logits and topk_ids token dimensions must match")
    if topk_ids.shape[1] not in (1, 2, 4, 8):
        raise ValueError("selected-softmax supports K in {1,2,4,8}")
    if router_logits.shape[1] < topk_ids.shape[1]:
        raise ValueError("router expert dimension must be at least topk")
    if not router_logits.is_cuda:
        raise ValueError("the Opus router backward requires a GPU tensor")
    if router_logits.device != topk_ids.device:
        raise ValueError("router_logits and topk_ids must be on the same device")
    return _OpusMoeSelectedSoftmaxFunction.apply(
        router_logits,
        topk_ids.contiguous(),
        int(kernel_id),
    )


def opus_moe_route_backward(
    d_z_sorted: Tensor,
    w1: Tensor,
    sorted_token_ids: Tensor,
    sorted_expert_ids: Tensor,
    num_valid_ids: Tensor,
    expert_padded_offsets: Tensor,
    reverse_sorted: Tensor,
    *,
    topk: int,
    block_m: int,
    route_dx_kernel_id: int = -1,
    route_reduce_kernel_id: int = -1,
) -> OpusMoeRouteBackwardOutput:
    """Launch K2 sorted route dX and K3 fixed-top-k token reduction."""

    if d_z_sorted.dtype != torch.bfloat16 or w1.dtype != torch.bfloat16:
        raise TypeError("Opus K2/K3 currently require BF16 d_z_sorted and w1")
    if d_z_sorted.ndim != 2 or w1.ndim != 3:
        raise ValueError("d_z_sorted and w1 must have rank 2 and 3")
    num_experts, gate_up_dim, model_dim = w1.shape
    if expert_padded_offsets.dtype != torch.int32:
        raise TypeError("expert_padded_offsets must be int32")
    if (
        expert_padded_offsets.ndim != 1
        or expert_padded_offsets.numel() != num_experts + 1
    ):
        raise ValueError("expert_padded_offsets must have shape [E+1]")
    if gate_up_dim % 2 != 0 or d_z_sorted.shape != (
        sorted_token_ids.numel(),
        gate_up_dim,
    ):
        raise ValueError("d_z_sorted must have shape [sorted_capacity,2I]")
    topk = int(topk)
    if topk not in (1, 2, 4, 8):
        raise ValueError("the first route reduce supports topk in {1,2,4,8}")
    if reverse_sorted.numel() % topk != 0:
        raise ValueError("reverse_sorted must contain T*topk entries")
    if gate_up_dim % 32 != 0 or model_dim % 128 != 0:
        raise ValueError("the first K2/K3 instances require 2I%32==0 and D%128==0")

    token_num = reverse_sorted.numel() // topk
    d_x_route = torch.empty(
        (sorted_token_ids.numel(), model_dim),
        dtype=torch.bfloat16,
        device=d_z_sorted.device,
    )
    d_x = torch.empty(
        (token_num, model_dim),
        dtype=torch.bfloat16,
        device=d_z_sorted.device,
    )
    _opus_moe_route_bwd_raw(
        d_z_sorted,
        w1,
        sorted_token_ids,
        sorted_expert_ids,
        num_valid_ids,
        expert_padded_offsets,
        reverse_sorted,
        d_x_route,
        d_x,
        int(block_m),
        int(route_dx_kernel_id),
        int(route_reduce_kernel_id),
    )
    return OpusMoeRouteBackwardOutput(
        d_x_route=d_x_route,
        d_x=d_x,
    )


def opus_moe_gather_x_blocked_g2(
    x: Tensor,
    sorted_token_ids: Tensor,
    num_valid_ids: Tensor,
    *,
    block_m: int,
    out: Tensor | None = None,
) -> Tensor:
    """Gather ``x[token]`` directly into K4's private blocked-G2 cache.

    The existing forward sorter metadata is reused verbatim; no TopK or host
    readback is performed.  Sorter-padding rows in the live padded prefix are
    materialized as exact zero.  Supplying ``out`` exposes the graph-safe
    preallocated-output form needed by a forward pipeline.
    """

    if x.dtype != torch.bfloat16 or x.ndim != 2:
        raise TypeError("x must be a rank-2 BF16 tensor")
    if sorted_token_ids.dtype != torch.int32 or sorted_token_ids.ndim != 1:
        raise TypeError("sorted_token_ids must be a rank-1 int32 tensor")
    if num_valid_ids.dtype != torch.int32 or num_valid_ids.ndim != 1:
        raise TypeError("num_valid_ids must be a rank-1 int32 tensor")
    if num_valid_ids.numel() < 1:
        raise ValueError("num_valid_ids must contain at least one entry")
    if int(block_m) != 32:
        raise ValueError("blocked-G2 sorted-X requires block_m=32")
    if x.shape[0] <= 0 or x.shape[1] <= 0 or x.shape[1] % 32 != 0:
        raise ValueError("x must have positive [T,D] with D divisible by 32")
    expected_shape = (sorted_token_ids.numel(), x.shape[1])
    if out is None:
        out = torch.empty(expected_shape, dtype=x.dtype, device=x.device)
    if out.dtype != torch.bfloat16 or out.shape != expected_shape:
        raise ValueError("out must be BF16 with shape [sorted_capacity,D]")
    tensors = (x, sorted_token_ids, num_valid_ids, out)
    if any(tensor.device != x.device for tensor in tensors):
        raise ValueError("x, sorting metadata, and out must share one device")
    if any(not tensor.is_contiguous() for tensor in tensors):
        raise ValueError("x, sorting metadata, and out must be contiguous")
    _opus_moe_gather_x_blocked_g2_raw(
        x, sorted_token_ids, num_valid_ids, out, int(block_m)
    )
    return out


def opus_moe_weight_backward(
    x: Tensor,
    d_out: Tensor,
    d_z_sorted: Tensor,
    a_scaled_sorted: Tensor,
    sorted_token_ids: Tensor,
    num_valid_ids: Tensor,
    expert_padded_offsets: Tensor,
    *,
    topk: int,
    block_m: int,
    dw1_kernel_id: int = -1,
    dw2_kernel_id: int = -1,
) -> OpusMoeWeightBackwardOutput:
    """Launch grouped K4 dW1 and K5 dW2 over padded expert intervals."""

    bf16_tensors = (x, d_out, d_z_sorted, a_scaled_sorted)
    if any(tensor.dtype != torch.bfloat16 for tensor in bf16_tensors):
        raise TypeError("Opus K4/K5 currently require BF16 data tensors")
    if x.ndim != 2 or d_out.ndim != 2:
        raise ValueError("x and d_out must have rank 2")
    if d_z_sorted.ndim != 2 or a_scaled_sorted.ndim != 2:
        raise ValueError("d_z_sorted and a_scaled_sorted must have rank 2")
    if sorted_token_ids.dtype != torch.int32:
        raise TypeError("sorted_token_ids must be int32")
    if num_valid_ids.dtype != torch.int32:
        raise TypeError("num_valid_ids must be int32")
    if expert_padded_offsets.dtype != torch.int32:
        raise TypeError("expert_padded_offsets must be int32")

    token_num, model_dim = x.shape
    if d_out.shape != (token_num, model_dim):
        raise ValueError("d_out must have the same [T,D] shape as x")
    sorted_capacity, gate_up_dim = d_z_sorted.shape
    if gate_up_dim <= 0 or gate_up_dim % 2 != 0:
        raise ValueError("d_z_sorted gate/up dimension must be positive and even")
    inter_dim = gate_up_dim // 2
    if sorted_token_ids.numel() != sorted_capacity:
        raise ValueError("d_z_sorted must have shape [sorted_capacity,2I]")
    if a_scaled_sorted.shape != (sorted_capacity, inter_dim):
        raise ValueError("a_scaled_sorted must have shape [sorted_capacity,I]")
    if expert_padded_offsets.ndim != 1 or expert_padded_offsets.numel() < 2:
        raise ValueError("expert_padded_offsets must contain E+1 entries")
    if num_valid_ids.numel() < 1:
        raise ValueError("num_valid_ids must contain at least one entry")
    topk = int(topk)
    block_m = int(block_m)
    if not 0 < topk <= 256:
        raise ValueError("topk must be in [1,256]")
    if block_m != 32:
        raise ValueError("the first K4/K5 instances require block_m=32")
    if gate_up_dim % 32 != 0 or model_dim % 128 != 0:
        raise ValueError("K4 requires 2I%32==0 and D%128==0")
    if model_dim % 32 != 0 or inter_dim % 128 != 0:
        raise ValueError("K5 requires D%32==0 and I%128==0")
    _reject_sorted_x_dw1_without_cache(dw1_kernel_id)

    num_experts = expert_padded_offsets.numel() - 1
    d_w1 = torch.empty(
        (num_experts, gate_up_dim, model_dim),
        dtype=torch.bfloat16,
        device=x.device,
    )
    d_w2 = torch.empty(
        (num_experts, model_dim, inter_dim),
        dtype=torch.bfloat16,
        device=x.device,
    )
    _opus_moe_weight_bwd_raw(
        x,
        d_out,
        d_z_sorted,
        a_scaled_sorted,
        sorted_token_ids,
        num_valid_ids,
        expert_padded_offsets,
        d_w1,
        d_w2,
        topk,
        block_m,
        int(dw1_kernel_id),
        int(dw2_kernel_id),
    )
    return OpusMoeWeightBackwardOutput(d_w1=d_w1, d_w2=d_w2)


def opus_moe_dw1_backward(
    x: Tensor,
    d_z_sorted: Tensor,
    sorted_token_ids: Tensor,
    num_valid_ids: Tensor,
    expert_padded_offsets: Tensor,
    *,
    topk: int,
    block_m: int,
    kernel_id: int = -1,
    saved_x_sorted: Tensor | None = None,
) -> Tensor:
    """Launch K4 only, allowing the up Function to prune K5.

    ``saved_x_sorted`` may carry the forward input in padded expert-sorted
    route order.  Padding rows must be exact zero.  This removes K4's repeated
    token decode and random gather while leaving the default token-major path
    unchanged.
    """

    if x.dtype != torch.bfloat16 or d_z_sorted.dtype != torch.bfloat16:
        raise TypeError("Opus K4 currently requires BF16 x and d_z_sorted")
    if x.ndim != 2 or d_z_sorted.ndim != 2:
        raise ValueError("x and d_z_sorted must have rank 2")
    if sorted_token_ids.dtype != torch.int32:
        raise TypeError("sorted_token_ids must be int32")
    if num_valid_ids.dtype != torch.int32:
        raise TypeError("num_valid_ids must be int32")
    if expert_padded_offsets.dtype != torch.int32:
        raise TypeError("expert_padded_offsets must be int32")
    token_num, model_dim = x.shape
    del token_num
    sorted_capacity, gate_up_dim = d_z_sorted.shape
    if sorted_token_ids.numel() != sorted_capacity:
        raise ValueError("d_z_sorted must have shape [sorted_capacity,2I]")
    if gate_up_dim <= 0 or gate_up_dim % 2 != 0:
        raise ValueError("d_z_sorted gate/up dimension must be positive and even")
    if expert_padded_offsets.ndim != 1 or expert_padded_offsets.numel() < 2:
        raise ValueError("expert_padded_offsets must contain E+1 entries")
    topk = int(topk)
    block_m = int(block_m)
    if not 0 < topk <= 256:
        raise ValueError("topk must be in [1,256]")
    if block_m != 32:
        raise ValueError("the first K4 instance requires block_m=32")
    if gate_up_dim % 32 != 0 or model_dim % 128 != 0:
        raise ValueError("K4 requires 2I%32==0 and D%128==0")
    x_dw1 = x
    if saved_x_sorted is not None:
        _validate_saved_x_sorted(
            saved_x_sorted, sorted_capacity, model_dim, x.device
        )
        kernel_id = _select_saved_x_dw1_kernel(
            gate_up_dim,
            model_dim,
            sorted_capacity,
            expert_padded_offsets.numel() - 1,
            int(kernel_id),
        )
        x_dw1 = saved_x_sorted
    else:
        _reject_sorted_x_dw1_without_cache(kernel_id)

    d_w1 = torch.empty(
        (expert_padded_offsets.numel() - 1, gate_up_dim, model_dim),
        dtype=torch.bfloat16,
        device=x.device,
    )
    _opus_moe_dw1_bwd_raw(
        x_dw1,
        d_z_sorted,
        sorted_token_ids,
        num_valid_ids,
        expert_padded_offsets,
        d_w1,
        topk,
        block_m,
        int(kernel_id),
    )
    return d_w1


def opus_moe_dw2_backward(
    d_out: Tensor,
    a_scaled_sorted: Tensor,
    sorted_token_ids: Tensor,
    num_valid_ids: Tensor,
    expert_padded_offsets: Tensor,
    *,
    topk: int,
    block_m: int,
    kernel_id: int = -1,
) -> Tensor:
    """Launch K5 only, allowing the down Function to prune K4."""

    if d_out.dtype != torch.bfloat16 or a_scaled_sorted.dtype != torch.bfloat16:
        raise TypeError("Opus K5 currently requires BF16 d_out and a_scaled")
    if d_out.ndim != 2 or a_scaled_sorted.ndim != 2:
        raise ValueError("d_out and a_scaled_sorted must have rank 2")
    if sorted_token_ids.dtype != torch.int32:
        raise TypeError("sorted_token_ids must be int32")
    if num_valid_ids.dtype != torch.int32:
        raise TypeError("num_valid_ids must be int32")
    if expert_padded_offsets.dtype != torch.int32:
        raise TypeError("expert_padded_offsets must be int32")
    token_num, model_dim = d_out.shape
    del token_num
    sorted_capacity, inter_dim = a_scaled_sorted.shape
    if sorted_token_ids.numel() != sorted_capacity:
        raise ValueError("a_scaled_sorted must have shape [sorted_capacity,I]")
    if expert_padded_offsets.ndim != 1 or expert_padded_offsets.numel() < 2:
        raise ValueError("expert_padded_offsets must contain E+1 entries")
    topk = int(topk)
    block_m = int(block_m)
    if not 0 < topk <= 256:
        raise ValueError("topk must be in [1,256]")
    if block_m != 32:
        raise ValueError("the first K5 instance requires block_m=32")
    if model_dim % 32 != 0 or inter_dim % 128 != 0:
        raise ValueError("K5 requires D%32==0 and I%128==0")

    d_w2 = torch.empty(
        (expert_padded_offsets.numel() - 1, model_dim, inter_dim),
        dtype=torch.bfloat16,
        device=d_out.device,
    )
    _opus_moe_dw2_bwd_raw(
        d_out,
        a_scaled_sorted,
        sorted_token_ids,
        num_valid_ids,
        expert_padded_offsets,
        d_w2,
        topk,
        block_m,
        int(kernel_id),
    )
    return d_w2


# Raw compact variable-routing JIT bindings.

def _gen_varlen_down_fake(
    d_out: Tensor,
    z: Tensor,
    w2: Tensor,
    scores: Tensor,
    b2: Tensor,
    sorted_route_ids: Tensor,
    sorted_expert_ids: Tensor,
    num_valid_ids: Tensor,
    route_to_token: Tensor,
    expert_padded_offsets: Tensor,
    d_scores_workspace: Tensor,
    d_z: Tensor,
    a_scaled: Tensor,
    d_scores: Tensor,
    d_w2: Tensor,
    d_b2: Tensor,
) -> Tensor:
    return d_z


@compile_ops(
    "module_opus_moe_backward",
    fc_name="opus_moe_varlen_down_bwd",
    gen_fake=_gen_varlen_down_fake,
    develop=True,
)
def _opus_moe_varlen_down_bwd_raw(
    d_out: Tensor,
    z: Tensor,
    w2: Tensor,
    scores: Tensor,
    b2: Tensor,
    sorted_route_ids: Tensor,
    sorted_expert_ids: Tensor,
    num_valid_ids: Tensor,
    route_to_token: Tensor,
    expert_padded_offsets: Tensor,
    d_scores_workspace: Tensor,
    d_z: Tensor,
    a_scaled: Tensor,
    d_scores: Tensor,
    d_w2: Tensor,
    d_b2: Tensor,
    block_m: int,
    has_bias: bool,
    compute_dz: bool,
    compute_dw2: bool,
    compute_dscore: bool,
    compute_db2: bool,
    down_kernel_id: int,
    dw2_kernel_id: int,
    bias_kernel_id: int,
) -> Tensor: ...


def _gen_varlen_up_fake(
    d_z: Tensor,
    x: Tensor,
    w1: Tensor,
    sorted_route_ids: Tensor,
    sorted_expert_ids: Tensor,
    num_valid_ids: Tensor,
    route_to_token: Tensor,
    token_route_offsets: Tensor,
    expert_padded_offsets: Tensor,
    d_x_route: Tensor,
    d_x: Tensor,
    d_w1: Tensor,
    d_b1: Tensor,
) -> Tensor:
    return d_x


@compile_ops(
    "module_opus_moe_backward",
    fc_name="opus_moe_varlen_up_bwd",
    gen_fake=_gen_varlen_up_fake,
    develop=True,
)
def _opus_moe_varlen_up_bwd_raw(
    d_z: Tensor,
    x: Tensor,
    w1: Tensor,
    sorted_route_ids: Tensor,
    sorted_expert_ids: Tensor,
    num_valid_ids: Tensor,
    route_to_token: Tensor,
    token_route_offsets: Tensor,
    expert_padded_offsets: Tensor,
    d_x_route: Tensor,
    d_x: Tensor,
    d_w1: Tensor,
    d_b1: Tensor,
    block_m: int,
    compute_dx: bool,
    compute_dw1: bool,
    compute_db1: bool,
    route_dx_kernel_id: int,
    route_reduce_kernel_id: int,
    dw1_kernel_id: int,
    bias_kernel_id: int,
) -> Tensor: ...


def _gen_varlen_router_fake(
    d_scores: Tensor,
    scores: Tensor,
    route_expert_ids: Tensor,
    token_route_offsets: Tensor,
    d_logits: Tensor,
) -> Tensor:
    return d_logits


@compile_ops(
    "module_opus_moe_backward",
    fc_name="opus_moe_varlen_router_bwd",
    gen_fake=_gen_varlen_router_fake,
    develop=True,
)
def _opus_moe_varlen_router_bwd_raw(
    d_scores: Tensor,
    scores: Tensor,
    route_expert_ids: Tensor,
    token_route_offsets: Tensor,
    d_logits: Tensor,
    kernel_id: int,
) -> Tensor: ...


# Checked full-chain wrappers and the production autograd attachment.


def _validate_common_backward_contract(
    d_out: Tensor,
    x: Tensor,
    z_sorted: Tensor,
    w1: Tensor,
    w2: Tensor,
    route_weights: Tensor,
    metadata: OpusMoeFixedMetadata | OpusMoeVarlenMetadata,
    b1: Tensor | None,
    b2: Tensor | None,
) -> tuple[int, int, int, int]:
    if not isinstance(metadata, (OpusMoeFixedMetadata, OpusMoeVarlenMetadata)):
        raise TypeError(
            "metadata must be OpusMoeFixedMetadata or OpusMoeVarlenMetadata"
        )
    if metadata.layout_version != 1:
        raise ValueError(f"unsupported Opus MoE metadata v{metadata.layout_version}")
    if any(tensor.dtype != torch.bfloat16 for tensor in (d_out, x, z_sorted, w1, w2)):
        raise TypeError("Opus MoE backward requires BF16 d_out/x/z/w1/w2")
    if d_out.ndim != 2 or x.ndim != 2 or z_sorted.ndim != 2:
        raise ValueError("d_out, x, and z_sorted must have rank 2")
    if w1.ndim != 3 or w2.ndim != 3:
        raise ValueError("w1 and w2 must have rank 3")

    token_num, model_dim = x.shape
    num_experts, gate_up_dim, w1_model_dim = w1.shape
    w2_experts, w2_model_dim, inter_dim = w2.shape
    if d_out.shape != x.shape:
        raise ValueError("d_out must have the same [T,D] shape as x")
    if (
        w1_model_dim != model_dim
        or w2_experts != num_experts
        or w2_model_dim != model_dim
        or gate_up_dim != 2 * inter_dim
    ):
        raise ValueError("expected matching w1=[E,2I,D] and w2=[E,D,I]")
    if model_dim % 128 != 0 or inter_dim % 128 != 0:
        raise ValueError("current gfx950 instances require D%128==0 and I%128==0")
    if int(metadata.block_m) != 32:
        raise ValueError("current gfx950 instances require sorting block_m=32")

    is_varlen = isinstance(metadata, OpusMoeVarlenMetadata)
    sorted_ids = metadata.sorted_route_ids if is_varlen else metadata.sorted_token_ids
    if z_sorted.shape != (sorted_ids.numel(), gate_up_dim):
        raise ValueError("z_sorted must have shape [sorted_capacity,2I]")
    if metadata.expert_padded_offsets.shape != (num_experts + 1,):
        raise ValueError("expert_padded_offsets must have shape [E+1]")
    if b1 is not None and (
        b1.dtype != torch.bfloat16 or b1.shape != (num_experts, gate_up_dim)
    ):
        raise ValueError("b1 must be BF16 with shape [E,2I]")
    if b2 is not None and (
        b2.dtype != torch.bfloat16 or b2.shape != (num_experts, model_dim)
    ):
        raise ValueError("b2 must be BF16 with shape [E,D]")

    if is_varlen:
        if route_weights.dtype != torch.float32 or route_weights.ndim != 1:
            raise TypeError("variable route_weights must be a 1D float32 tensor")
        route_count = route_weights.numel()
        if metadata.route_to_token.shape != (route_count,):
            raise ValueError("route_to_token must have shape [R]")
        if metadata.token_route_offsets.shape != (token_num + 1,):
            raise ValueError("token_route_offsets must have shape [T+1]")
        metadata_tensors = (
            metadata.sorted_route_ids,
            metadata.sorted_expert_ids,
            metadata.num_valid_ids,
            metadata.route_to_token,
            metadata.token_route_offsets,
            metadata.expert_padded_offsets,
        )
    else:
        if route_weights.dtype != torch.float32 or route_weights.ndim != 2:
            raise TypeError("fixed route_weights must be a rank-2 float32 tensor")
        if route_weights.shape[0] != token_num:
            raise ValueError("route_weights must have shape [T,K]")
        topk = int(route_weights.shape[1])
        if topk not in (1, 2, 4, 8):
            raise ValueError("fixed routing supports K in {1,2,4,8}")
        if metadata.reverse_sorted.numel() != token_num * topk:
            raise ValueError("reverse_sorted must contain T*K entries")
        metadata_tensors = (
            metadata.sorted_token_ids,
            metadata.sorted_expert_ids,
            metadata.num_valid_ids,
            metadata.reverse_sorted,
            metadata.expert_padded_offsets,
        )

    for tensor in metadata_tensors:
        if tensor.dtype != torch.int32:
            raise TypeError("all Opus MoE sorting metadata tensors must be int32")
        if tensor.ndim != 1:
            raise ValueError("all Opus MoE sorting metadata tensors must be rank 1")
    if metadata.num_valid_ids.numel() < 1:
        raise ValueError("num_valid_ids must contain at least one entry")
    if metadata.sorted_expert_ids.numel() * int(metadata.block_m) < sorted_ids.numel():
        raise ValueError("sorted_expert_ids does not cover sorted route capacity")

    tensors = (d_out, x, z_sorted, w1, w2, route_weights, *metadata_tensors)
    if b1 is not None:
        tensors += (b1,)
    if b2 is not None:
        tensors += (b2,)
    if any(tensor.device != x.device for tensor in tensors):
        raise ValueError("all Opus MoE tensors must share one device")
    if any(not tensor.is_contiguous() for tensor in tensors):
        raise ValueError("all Opus MoE tensors must be contiguous")
    return token_num, model_dim, num_experts, inter_dim


def _validate_attach_contract(
    out: Tensor,
    a_sorted: Tensor,
    z_sorted: Tensor,
    x: Tensor,
    w1: Tensor,
    w2: Tensor,
    route_weights: Tensor,
    metadata: OpusMoeFixedMetadata | OpusMoeVarlenMetadata,
    b1: Tensor | None,
    b2: Tensor | None,
) -> None:
    token_num, model_dim, _, inter_dim = _validate_common_backward_contract(
        out, x, z_sorted, w1, w2, route_weights, metadata, b1, b2
    )
    sorted_ids = (
        metadata.sorted_route_ids
        if isinstance(metadata, OpusMoeVarlenMetadata)
        else metadata.sorted_token_ids
    )
    if out.shape != (token_num, model_dim) or out.dtype != torch.bfloat16:
        raise ValueError("out must be BF16 with shape [T,D]")
    if a_sorted.shape != (sorted_ids.numel(), inter_dim):
        raise ValueError("a_sorted must have shape [sorted_capacity,I]")
    if a_sorted.dtype != torch.bfloat16:
        raise TypeError("a_sorted must be BF16")
    if a_sorted.device != x.device or not a_sorted.is_contiguous():
        raise ValueError("a_sorted must be contiguous on the common device")


def opus_moe_backward(
    d_out: Tensor,
    x: Tensor,
    z_sorted: Tensor,
    w1: Tensor,
    w2: Tensor,
    route_weights: Tensor,
    metadata: OpusMoeFixedMetadata,
    *,
    b1: Tensor | None = None,
    b2: Tensor | None = None,
    down_kernel_id: int = -1,
    route_dx_kernel_id: int = -1,
    route_reduce_kernel_id: int = -1,
    dw1_kernel_id: int = -1,
    dw2_kernel_id: int = -1,
    db1_kernel_id: int = -1,
    bias_kernel_id: int = -1,
    saved_a_scaled: Tensor | None = None,
    saved_x_sorted: Tensor | None = None,
    saved_x_sorted_blocked_g2: bool = False,
) -> OpusMoeBackwardOutput:
    """Run the complete checked fixed-top-k K1--K5 backward chain.

    A BF16 ``saved_a_scaled=[sorted_capacity,I]`` may be supplied by forward
    to remove K1's scale/pack/global-store epilogue.  It must use the same
    sorting metadata and contain ``route_weight * SwiGLU(z_sorted)`` for each
    routed row; every sorter-padding row must be exact zero.  The cache is
    treated as
    non-differentiable because K1 still computes dZ and dScore explicitly.

    A BF16 ``saved_x_sorted=[sorted_capacity,D]`` may independently preserve
    the forward input in expert-sorted route order.  Its live rows must equal
    ``x[token]`` and every sorter-padding row must be exact zero.  Set
    ``saved_x_sorted_blocked_g2=True`` only for a cache produced by
    :func:`opus_moe_gather_x_blocked_g2`; this couples K1/K2/K4 to their
    blocked-G2 producer-consumer instances and requires ``saved_a_scaled``.
    """

    if not isinstance(metadata, OpusMoeFixedMetadata):
        raise TypeError("opus_moe_backward requires OpusMoeFixedMetadata")
    token_num, model_dim, num_experts, inter_dim = (
        _validate_common_backward_contract(
            d_out, x, z_sorted, w1, w2, route_weights, metadata, b1, b2
        )
    )
    sorted_capacity = metadata.sorted_token_ids.numel()
    saved_x_sorted_blocked_g2 = bool(saved_x_sorted_blocked_g2)
    if saved_x_sorted_blocked_g2 and saved_x_sorted is None:
        raise ValueError(
            "saved_x_sorted_blocked_g2=True requires saved_x_sorted"
        )
    if saved_a_scaled is not None:
        _validate_saved_a_scaled(
            saved_a_scaled, sorted_capacity, inter_dim, x.device
        )
        if not saved_x_sorted_blocked_g2:
            down_kernel_id = _select_saved_a_scaled_down_kernel(
                inter_dim,
                sorted_capacity,
                metadata.sorted_expert_ids,
                metadata.expert_padded_offsets,
                down_kernel_id,
            )
    else:
        if saved_x_sorted_blocked_g2:
            raise ValueError(
                "blocked-G2 full pipeline requires saved_a_scaled because "
                "K1 kernel 17 does not rewrite that forward cache"
            )
        _reject_saved_a_scaled_down_without_cache(down_kernel_id)
    x_dw1 = x
    if saved_x_sorted is not None:
        _validate_saved_x_sorted(
            saved_x_sorted, sorted_capacity, model_dim, x.device
        )
        if saved_x_sorted_blocked_g2:
            (
                down_kernel_id,
                route_dx_kernel_id,
                route_reduce_kernel_id,
                dw1_kernel_id,
            ) = _select_blocked_g2_full_pipeline(
                inter_dim,
                model_dim,
                down_kernel_id,
                route_dx_kernel_id,
                route_reduce_kernel_id,
                dw1_kernel_id,
            )
        else:
            dw1_kernel_id = _select_saved_x_dw1_kernel(
                2 * inter_dim,
                model_dim,
                sorted_capacity,
                num_experts,
                int(dw1_kernel_id),
            )
        x_dw1 = saved_x_sorted
    else:
        _reject_sorted_x_dw1_without_cache(dw1_kernel_id)
    down_block_n = _select_fixed_down_block_n(
        inter_dim,
        metadata.sorted_expert_ids,
        metadata.expert_padded_offsets,
        down_kernel_id,
    )
    d_scores_parts = (inter_dim + down_block_n - 1) // down_block_n
    d_scores_workspace = torch.empty(
        (route_weights.numel(), d_scores_parts)
        if d_scores_parts > 1
        else (0, 0),
        dtype=torch.float32,
        device=x.device,
    )
    d_z = torch.empty_like(z_sorted)
    a_scaled = (
        saved_a_scaled
        if saved_a_scaled is not None
        else torch.empty(
            (sorted_capacity, inter_dim), dtype=torch.bfloat16, device=x.device
        )
    )
    d_scores = torch.empty_like(route_weights)
    d_x_route = torch.empty(
        (sorted_capacity, model_dim), dtype=torch.bfloat16, device=x.device
    )
    d_x = torch.empty_like(x)
    d_w1 = torch.empty_like(w1)
    d_w2 = torch.empty_like(w2)

    _opus_moe_full_bwd_raw(
        d_out,
        x,
        x_dw1,
        z_sorted,
        w1,
        w2,
        route_weights,
        metadata.sorted_token_ids,
        metadata.sorted_expert_ids,
        metadata.num_valid_ids,
        metadata.reverse_sorted,
        metadata.expert_padded_offsets,
        d_scores_workspace,
        d_z,
        a_scaled,
        d_scores,
        d_x_route,
        d_x,
        d_w1,
        d_w2,
        int(metadata.block_m),
        int(down_kernel_id),
        int(route_dx_kernel_id),
        int(route_reduce_kernel_id),
        int(dw1_kernel_id),
        int(dw2_kernel_id),
        saved_x_sorted_blocked_g2,
    )

    d_b1 = None
    if b1 is not None:
        d_b1 = opus_moe_db1_backward(
            d_z,
            metadata.sorted_token_ids,
            metadata.num_valid_ids,
            metadata.expert_padded_offsets,
            token_num=token_num,
            topk=route_weights.shape[1],
            block_m=metadata.block_m,
            kernel_id=db1_kernel_id,
        )
    d_b2 = None
    if b2 is not None:
        bias = opus_moe_bias_down_backward(
            d_out,
            route_weights,
            b2,
            metadata.sorted_token_ids,
            metadata.sorted_expert_ids,
            metadata.num_valid_ids,
            metadata.expert_padded_offsets,
            d_scores,
            block_m=metadata.block_m,
            kernel_id=bias_kernel_id,
        )
        d_b2 = bias.d_b2

    return OpusMoeBackwardOutput(
        d_x=d_x,
        d_w1=d_w1,
        d_w2=d_w2,
        d_scores=d_scores,
        d_z_sorted=d_z,
        a_scaled=a_scaled,
        d_b1=d_b1,
        d_b2=d_b2,
    )


def _varlen_up_backward(
    d_z_sorted: Tensor,
    x: Tensor,
    w1: Tensor,
    metadata: OpusMoeVarlenMetadata,
    *,
    compute_dx: bool,
    compute_dw1: bool,
    compute_db1: bool,
    route_dx_kernel_id: int = -1,
    route_reduce_kernel_id: int = -1,
    dw1_kernel_id: int = -1,
    bias_kernel_id: int = -1,
) -> tuple[Tensor | None, Tensor | None, Tensor | None]:
    if compute_dw1:
        _reject_sorted_x_dw1_without_cache(dw1_kernel_id)
    d_x_route = torch.empty(
        (metadata.route_to_token.numel(), x.shape[1]),
        device=x.device,
        dtype=x.dtype,
    )
    native_d_x = torch.empty_like(x)
    native_d_w1 = torch.empty_like(w1)
    native_d_b1 = (
        torch.empty(
            (w1.shape[0], w1.shape[1]), device=w1.device, dtype=w1.dtype
        )
        if compute_db1
        else w1.new_empty((0, 0))
    )
    _opus_moe_varlen_up_bwd_raw(
        d_z_sorted,
        x,
        w1,
        metadata.sorted_route_ids,
        metadata.sorted_expert_ids,
        metadata.num_valid_ids,
        metadata.route_to_token,
        metadata.token_route_offsets,
        metadata.expert_padded_offsets,
        d_x_route,
        native_d_x,
        native_d_w1,
        native_d_b1,
        int(metadata.block_m),
        bool(compute_dx),
        bool(compute_dw1),
        bool(compute_db1),
        int(route_dx_kernel_id),
        int(route_reduce_kernel_id),
        int(dw1_kernel_id),
        int(bias_kernel_id),
    )
    return (
        native_d_x if compute_dx else None,
        native_d_w1 if compute_dw1 else None,
        native_d_b1 if compute_db1 else None,
    )


def _varlen_down_backward(
    d_out: Tensor,
    z_sorted: Tensor,
    w2: Tensor,
    route_weights: Tensor,
    metadata: OpusMoeVarlenMetadata,
    b2: Tensor | None,
    *,
    compute_dz: bool,
    compute_dw2: bool,
    compute_dscore: bool,
    compute_db2: bool,
    down_kernel_id: int = -1,
    dw2_kernel_id: int = -1,
    bias_kernel_id: int = -1,
) -> tuple[Tensor | None, Tensor | None, Tensor | None, Tensor | None, Tensor]:
    inter_dim = w2.shape[2]
    d_scores_parts = (inter_dim + 127) // 128
    workspace = torch.empty(
        (route_weights.numel(), d_scores_parts),
        device=route_weights.device,
        dtype=torch.float32,
    )
    native_d_z = torch.empty_like(z_sorted)
    a_scaled = torch.empty(
        (z_sorted.shape[0], inter_dim),
        device=z_sorted.device,
        dtype=z_sorted.dtype,
    )
    native_d_scores = torch.empty_like(route_weights)
    native_d_w2 = torch.empty_like(w2)
    saved_b2 = b2 if b2 is not None else w2.new_empty((0, 0))
    # The native contract requires a shape-correct db2 placeholder whenever
    # b2 is present, even when this autograd request prunes the db2 write.
    native_d_b2 = (
        torch.empty((w2.shape[0], w2.shape[1]), device=w2.device, dtype=w2.dtype)
        if b2 is not None
        else w2.new_empty((0, 0))
    )
    _opus_moe_varlen_down_bwd_raw(
        d_out,
        z_sorted,
        w2,
        route_weights,
        saved_b2,
        metadata.sorted_route_ids,
        metadata.sorted_expert_ids,
        metadata.num_valid_ids,
        metadata.route_to_token,
        metadata.expert_padded_offsets,
        workspace,
        native_d_z,
        a_scaled,
        native_d_scores,
        native_d_w2,
        native_d_b2,
        int(metadata.block_m),
        b2 is not None,
        bool(compute_dz),
        bool(compute_dw2),
        bool(compute_dscore),
        bool(compute_db2),
        int(down_kernel_id),
        int(dw2_kernel_id),
        int(bias_kernel_id),
    )
    return (
        native_d_z if compute_dz else None,
        native_d_w2 if compute_dw2 else None,
        native_d_scores if compute_dscore else None,
        native_d_b2 if compute_db2 else None,
        a_scaled,
    )


def opus_moe_varlen_backward(
    d_out: Tensor,
    x: Tensor,
    z_sorted: Tensor,
    w1: Tensor,
    w2: Tensor,
    route_weights: Tensor,
    metadata: OpusMoeVarlenMetadata,
    *,
    b1: Tensor | None = None,
    b2: Tensor | None = None,
    down_kernel_id: int = -1,
    route_dx_kernel_id: int = -1,
    route_reduce_kernel_id: int = -1,
    dw1_kernel_id: int = -1,
    dw2_kernel_id: int = -1,
    db1_kernel_id: int = -1,
    bias_kernel_id: int = -1,
) -> OpusMoeBackwardOutput:
    """Run the complete compact-route backward chain."""

    if not isinstance(metadata, OpusMoeVarlenMetadata):
        raise TypeError(
            "opus_moe_varlen_backward requires OpusMoeVarlenMetadata"
        )
    _validate_common_backward_contract(
        d_out, x, z_sorted, w1, w2, route_weights, metadata, b1, b2
    )
    d_z, d_w2, d_scores, d_b2, a_scaled = _varlen_down_backward(
        d_out,
        z_sorted,
        w2,
        route_weights,
        metadata,
        b2,
        compute_dz=True,
        compute_dw2=True,
        compute_dscore=True,
        compute_db2=b2 is not None,
        down_kernel_id=down_kernel_id,
        dw2_kernel_id=dw2_kernel_id,
        bias_kernel_id=bias_kernel_id,
    )
    d_x, d_w1, d_b1 = _varlen_up_backward(
        d_z,
        x,
        w1,
        metadata,
        compute_dx=True,
        compute_dw1=True,
        compute_db1=b1 is not None,
        route_dx_kernel_id=route_dx_kernel_id,
        route_reduce_kernel_id=route_reduce_kernel_id,
        dw1_kernel_id=dw1_kernel_id,
        bias_kernel_id=db1_kernel_id,
    )
    return OpusMoeBackwardOutput(
        d_x=d_x,
        d_w1=d_w1,
        d_w2=d_w2,
        d_scores=d_scores,
        d_z_sorted=d_z,
        a_scaled=a_scaled,
        d_b1=d_b1,
        d_b2=d_b2,
    )


# The expert path intentionally has exactly two Functions for both layouts.


class _OpusMoeUpProjectionFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        a_sorted: Tensor,
        z_sorted: Tensor,
        x: Tensor,
        w1: Tensor,
        metadata: OpusMoeFixedMetadata | OpusMoeVarlenMetadata,
        b1: Tensor | None,
        saved_x_sorted: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        del b1
        saved_x = (
            saved_x_sorted
            if saved_x_sorted is not None
            else x.new_empty((0, 0))
        )
        ctx.save_for_backward(x, w1, saved_x)
        ctx.opus_metadata = metadata
        ctx.has_saved_x_sorted = saved_x_sorted is not None
        ctx.set_materialize_grads(False)
        ctx.mark_non_differentiable(a_sorted)
        return a_sorted, z_sorted

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_a_sorted: Tensor | None, grad_z_sorted: Tensor | None):
        del grad_a_sorted
        x, w1, saved_x_sorted = ctx.saved_tensors
        metadata = ctx.opus_metadata
        need_dx = ctx.needs_input_grad[2]
        need_dw1 = ctx.needs_input_grad[3]
        need_db1 = ctx.needs_input_grad[5]
        d_x = d_w1 = d_b1 = None
        if grad_z_sorted is not None and (need_dx or need_dw1 or need_db1):
            grad_z_sorted = grad_z_sorted.contiguous()
            if isinstance(metadata, OpusMoeVarlenMetadata):
                d_x, d_w1, d_b1 = _varlen_up_backward(
                    grad_z_sorted,
                    x,
                    w1,
                    metadata,
                    compute_dx=need_dx,
                    compute_dw1=need_dw1,
                    compute_db1=need_db1,
                )
            else:
                topk = metadata.reverse_sorted.numel() // x.shape[0]
                if need_dx:
                    route_dx_kernel_id, route_reduce_kernel_id = (
                        _select_internal_fixed_route_pair(
                            grad_z_sorted, w1, topk
                        )
                    )
                    d_x = opus_moe_route_backward(
                        grad_z_sorted,
                        w1,
                        metadata.sorted_token_ids,
                        metadata.sorted_expert_ids,
                        metadata.num_valid_ids,
                        metadata.expert_padded_offsets,
                        metadata.reverse_sorted,
                        topk=topk,
                        block_m=metadata.block_m,
                        route_dx_kernel_id=route_dx_kernel_id,
                        route_reduce_kernel_id=route_reduce_kernel_id,
                    ).d_x
                if need_dw1:
                    d_w1 = opus_moe_dw1_backward(
                        x,
                        grad_z_sorted,
                        metadata.sorted_token_ids,
                        metadata.num_valid_ids,
                        metadata.expert_padded_offsets,
                        topk=topk,
                        block_m=metadata.block_m,
                        saved_x_sorted=(
                            saved_x_sorted if ctx.has_saved_x_sorted else None
                        ),
                    )
                if need_db1:
                    d_b1 = opus_moe_db1_backward(
                        grad_z_sorted,
                        metadata.sorted_token_ids,
                        metadata.num_valid_ids,
                        metadata.expert_padded_offsets,
                        token_num=x.shape[0],
                        topk=topk,
                        block_m=metadata.block_m,
                    )
        return None, None, d_x, d_w1, None, d_b1, None


class _OpusMoeDownProjectionFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        out: Tensor,
        a_sorted: Tensor,
        z_sorted: Tensor,
        w2: Tensor,
        route_weights: Tensor,
        metadata: OpusMoeFixedMetadata | OpusMoeVarlenMetadata,
        b2: Tensor | None,
        saved_a_scaled: Tensor | None,
    ) -> Tensor:
        saved_b2 = b2 if b2 is not None else w2.new_empty((0, 0))
        saved_scaled = (
            saved_a_scaled
            if saved_a_scaled is not None
            else w2.new_empty((0, 0))
        )
        ctx.save_for_backward(
            z_sorted, w2, route_weights, saved_b2, saved_scaled
        )
        ctx.opus_metadata = metadata
        ctx.has_b2 = b2 is not None
        ctx.has_saved_a_scaled = saved_a_scaled is not None
        ctx.set_materialize_grads(False)
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_out: Tensor | None):
        z_sorted, w2, route_weights, saved_b2, saved_a_scaled = ctx.saved_tensors
        metadata = ctx.opus_metadata
        need_dz = ctx.needs_input_grad[2]
        need_dw2 = ctx.needs_input_grad[3]
        need_dscore = ctx.needs_input_grad[4]
        need_db2 = ctx.needs_input_grad[6]
        d_z = d_w2 = d_scores = d_b2 = None
        if grad_out is not None and (need_dz or need_dw2 or need_dscore or need_db2):
            grad_out = grad_out.contiguous()
            if isinstance(metadata, OpusMoeVarlenMetadata):
                d_z, d_w2, d_scores, d_b2, _ = _varlen_down_backward(
                    grad_out,
                    z_sorted,
                    w2,
                    route_weights,
                    metadata,
                    saved_b2 if ctx.has_b2 else None,
                    compute_dz=need_dz,
                    compute_dw2=need_dw2,
                    compute_dscore=need_dscore,
                    compute_db2=need_db2,
                )
            else:
                result = None
                if need_dz or need_dw2 or need_dscore:
                    result = opus_moe_down_backward(
                        grad_out,
                        z_sorted,
                        w2,
                        route_weights,
                        metadata.sorted_token_ids,
                        metadata.sorted_expert_ids,
                        metadata.num_valid_ids,
                        metadata.expert_padded_offsets,
                        block_m=metadata.block_m,
                        saved_a_scaled=(
                            saved_a_scaled if ctx.has_saved_a_scaled else None
                        ),
                    )
                if need_dz:
                    d_z = result.d_z_sorted
                if need_dw2:
                    d_w2 = opus_moe_dw2_backward(
                        grad_out,
                        result.a_scaled,
                        metadata.sorted_token_ids,
                        metadata.num_valid_ids,
                        metadata.expert_padded_offsets,
                        topk=route_weights.shape[1],
                        block_m=metadata.block_m,
                    )
                if ctx.has_b2 and (need_dscore or need_db2):
                    bias = opus_moe_bias_down_backward(
                        grad_out,
                        route_weights,
                        saved_b2,
                        metadata.sorted_token_ids,
                        metadata.sorted_expert_ids,
                        metadata.num_valid_ids,
                        metadata.expert_padded_offsets,
                        (
                            result.d_scores
                            if result is not None
                            else torch.zeros_like(route_weights)
                        ),
                        block_m=metadata.block_m,
                    )
                    if need_db2:
                        d_b2 = bias.d_b2
                if need_dscore and result is not None:
                    d_scores = result.d_scores
        return None, None, d_z, d_w2, d_scores, None, d_b2, None


def opus_moe_attach_backward(
    out: Tensor,
    a_sorted: Tensor,
    z_sorted: Tensor,
    x: Tensor,
    w1: Tensor,
    w2: Tensor,
    route_weights: Tensor,
    metadata: OpusMoeFixedMetadata | OpusMoeVarlenMetadata,
    *,
    b1: Tensor | None = None,
    b2: Tensor | None = None,
    saved_a_scaled: Tensor | None = None,
    saved_x_sorted: Tensor | None = None,
) -> Tensor:
    """Attach native backward to saved tensors produced by an Opus forward.

    No routing, TopK, projection, activation, or output computation happens
    here. Forward owns out/A_sorted/Z_sorted and passes the exact sorting
    metadata used to produce them.  It may also pass the non-differentiable
    BF16 ``saved_a_scaled=route_weight*SwiGLU(z_sorted)`` cache in the same
    expert-sorted route-major layout; sorter padding rows must be exact zero.
    ``saved_x_sorted`` may similarly preserve ``x[token]`` in padded sorted
    order, but its padding rows must be exact zero because K4 reduces them.
    """

    _validate_attach_contract(
        out, a_sorted, z_sorted, x, w1, w2, route_weights, metadata, b1, b2
    )
    if saved_a_scaled is not None:
        if isinstance(metadata, OpusMoeVarlenMetadata):
            raise ValueError(
                "saved_a_scaled fast path currently supports fixed top-k"
            )
        _validate_saved_a_scaled(
            saved_a_scaled,
            metadata.sorted_token_ids.numel(),
            w2.shape[2],
            x.device,
        )
    if saved_x_sorted is not None:
        if isinstance(metadata, OpusMoeVarlenMetadata):
            raise ValueError(
                "saved_x_sorted fast path currently supports fixed top-k"
            )
        _validate_saved_x_sorted(
            saved_x_sorted,
            metadata.sorted_token_ids.numel(),
            x.shape[1],
            x.device,
        )
    attached_a, attached_z = _OpusMoeUpProjectionFunction.apply(
        a_sorted, z_sorted, x, w1, metadata, b1, saved_x_sorted
    )
    return _OpusMoeDownProjectionFunction.apply(
        out,
        attached_a,
        attached_z,
        w2,
        route_weights,
        metadata,
        b2,
        saved_a_scaled,
    )


# Compact selected-softmax integration.  Its forward uses device-side Torch
# tensor operations; its backward always dispatches the native Opus kernel.


def opus_moe_varlen_router_backward(
    d_scores: Tensor,
    scores: Tensor,
    route_expert_ids: Tensor,
    token_route_offsets: Tensor,
    num_experts: int,
    *,
    kernel_id: int = -1,
) -> Tensor:
    """Selected-softmax backward for compact variable routes."""

    if d_scores.dtype != torch.float32 or scores.dtype != torch.float32:
        raise TypeError("d_scores and scores must be float32")
    if route_expert_ids.dtype != torch.int32:
        raise TypeError("route_expert_ids must be int32")
    if token_route_offsets.dtype != torch.int32:
        raise TypeError("token_route_offsets must be int32")
    if d_scores.ndim != 1 or scores.shape != d_scores.shape:
        raise ValueError("d_scores and scores must have shape [R]")
    if route_expert_ids.shape != scores.shape:
        raise ValueError("route_expert_ids must have shape [R]")
    if token_route_offsets.ndim != 1 or token_route_offsets.numel() < 2:
        raise ValueError("token_route_offsets must have shape [T+1]")
    if num_experts <= 0:
        raise ValueError("num_experts must be positive")
    token_num = token_route_offsets.numel() - 1
    d_logits = torch.empty(
        (token_num, num_experts), device=scores.device, dtype=torch.float32
    )
    _opus_moe_varlen_router_bwd_raw(
        d_scores.contiguous(),
        scores.contiguous(),
        route_expert_ids.contiguous(),
        token_route_offsets.contiguous(),
        d_logits,
        int(kernel_id),
    )
    return d_logits


def _segmented_softmax(
    selected_logits: Tensor,
    route_to_token: Tensor,
    token_num: int,
) -> Tensor:
    if selected_logits.numel() == 0:
        return selected_logits.clone()
    token_ids = route_to_token.to(torch.int64)
    maxima = torch.full(
        (token_num,),
        -torch.inf,
        device=selected_logits.device,
        dtype=selected_logits.dtype,
    )
    maxima.scatter_reduce_(
        0, token_ids, selected_logits, reduce="amax", include_self=True
    )
    numerator = torch.exp(selected_logits - maxima[token_ids])
    denominator = torch.zeros_like(maxima).index_add(0, token_ids, numerator)
    return (numerator / denominator[token_ids]).contiguous()


class _OpusMoeVarlenSelectedSoftmaxFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        router_logits: Tensor,
        route_expert_ids: Tensor,
        route_to_token: Tensor,
        token_route_offsets: Tensor,
        kernel_id: int,
    ) -> Tensor:
        selected_logits = router_logits[
            route_to_token.to(torch.int64), route_expert_ids.to(torch.int64)
        ]
        scores = _segmented_softmax(
            selected_logits, route_to_token, token_route_offsets.numel() - 1
        )
        ctx.save_for_backward(scores, route_expert_ids, token_route_offsets)
        ctx.num_experts = int(router_logits.shape[1])
        ctx.kernel_id = int(kernel_id)
        ctx.set_materialize_grads(False)
        return scores

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_scores: Tensor | None):
        if grad_scores is None:
            return None, None, None, None, None
        scores, route_expert_ids, token_route_offsets = ctx.saved_tensors
        d_logits = opus_moe_varlen_router_backward(
            grad_scores.contiguous(),
            scores,
            route_expert_ids,
            token_route_offsets,
            ctx.num_experts,
            kernel_id=ctx.kernel_id,
        )
        return d_logits, None, None, None, None


def opus_moe_varlen_selected_softmax(
    router_logits: Tensor,
    route_expert_ids: Tensor,
    route_to_token: Tensor,
    token_route_offsets: Tensor,
    *,
    kernel_id: int = -1,
) -> Tensor:
    """Selected-softmax over compact variable-route token segments."""

    if router_logits.dtype != torch.float32 or router_logits.ndim != 2:
        raise TypeError("router_logits must be a rank-2 float32 tensor")
    if route_expert_ids.dtype != torch.int32 or route_to_token.dtype != torch.int32:
        raise TypeError("route_expert_ids and route_to_token must be int32")
    if token_route_offsets.dtype != torch.int32:
        raise TypeError("token_route_offsets must be int32")
    if route_expert_ids.ndim != 1 or route_to_token.shape != route_expert_ids.shape:
        raise ValueError("route_expert_ids and route_to_token must have shape [R]")
    if token_route_offsets.shape != (router_logits.shape[0] + 1,):
        raise ValueError("token_route_offsets must have shape [T+1]")
    tensors = (router_logits, route_expert_ids, route_to_token, token_route_offsets)
    if any(not tensor.is_cuda for tensor in tensors):
        raise ValueError("the Opus varlen router backward requires GPU tensors")
    if any(tensor.device != router_logits.device for tensor in tensors):
        raise ValueError("varlen selected-softmax tensors must share one device")
    return _OpusMoeVarlenSelectedSoftmaxFunction.apply(
        router_logits,
        route_expert_ids.contiguous(),
        route_to_token.contiguous(),
        token_route_offsets.contiguous(),
        int(kernel_id),
    )


__all__ = [
    "OpusMoeBackwardOutput",
    "OpusMoeFixedMetadata",
    "OpusMoeVarlenMetadata",
    "opus_moe_attach_backward",
    "opus_moe_backward",
    "opus_moe_router_backward",
    "opus_moe_selected_softmax",
    "opus_moe_varlen_backward",
    "opus_moe_varlen_router_backward",
    "opus_moe_varlen_selected_softmax",
]
