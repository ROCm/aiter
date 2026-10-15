# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
# ruff: noqa: SIM102

"""SonicMoE-style A16 inference forward for CDNA4 (gfx950).

The hot path is three logical stages:

1. router softmax + top-k + expert sort/token rounding (and output zeroing),
2. grouped stage-1 GEMM with indexed row-gather and fused activation, and
3. grouped down-projection with routing-weighted atomic scatter, or fixed-slot
   stores followed by an FP32 top-k reduction.

No gathered activation tensor is materialized.  Expert rows are represented by
``sorted_token_ids`` and each expert's row count is rounded to ``route_tile_m``
by the sorting kernel.  Stage 1 gathers the original activation rows while
loading A; stage 2 consumes the sorted 16-bit intermediate and either scatters
directly to tokens or writes fixed top-k route rows for reduction.

Weights may be dense BF16/FP16 (A16W16) or per-1x32 E8M0-scaled MXFP4
(A16W4). MXFP4 currently uses BF16 activations; dense weights use the compute
dtype selected by :class:`SonicMoEConfig`.
The reusable :class:`SonicMoE` object retains its inference-forward API and
also exposes an explicitly training-oriented fixed-K state-producing entry
point.  This module exports standalone ``sonic_moe_backward`` and
``sonic_moe_backward_routes`` entry points for dense BF16/FP16 fixed-K and flat
ragged-route training across all supported activations, including optional
expert bias gradients.
"""

from __future__ import annotations

import functools
import math
import threading
from collections import OrderedDict
from collections.abc import Iterable
from dataclasses import dataclass, field, replace

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.runtime.device import get_rocm_arch

from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled

from .backward import (
    sonic_moe_backward,
    sonic_moe_backward_expert_major,
    sonic_moe_backward_routes,
)
from .dynamic_policy import (
    E16_ROUTE_POLICY_SIZES,
    E16RoutePolicy,
    canonical_e16_route_policy_size,
    e16_route_policy_representative,
    select_e16_route_policy,
)
from .kernels.ragged_sorting import (
    moe_expert_major_sorting_flydsl,
    moe_ragged_sorting_flydsl,
)
from .kernels.reduction import compile_moe_reduction
from .kernels.sorting import (
    moe_softmax_sort_flydsl,
    moe_sorting_flydsl,
    moe_sorting_get_workspace_size,
    topk_frequency_flydsl,
)
from .kernels.stage1 import (
    compile_gemm1_a16w4_port,
    gemm1_a16w4_grid,
)
from .kernels.stage2 import (
    compile_gemm2_a16w4_port,
    gemm2_a16w4_grid,
)
from .kernels.topk_gating import supports_topk_gating_layout

_GFX950_LDS_BYTES = 160 * 1024
_MAX_BUFFER_BYTE_OFFSET = 0xFFFFFFFF
_MAX_SIGNED_I32 = 0x7FFFFFFF
_DEFAULT_MAX_CACHED_WORKSPACES = 8
_SUPPORTED_ROUTER_DTYPES = {
    torch.float32: "f32",
    torch.float16: "f16",
    torch.bfloat16: "bf16",
}
_COMPUTE_DTYPES = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
}
_SUPPORTED_ACTIVATIONS = frozenset(
    {"swiglu", "geglu", "reglu", "gelu_tanh_approx", "relu", "silu", "relu_sq"}
)
_SUPPORTED_STAGE2_OUTPUT_MODES = frozenset({"atomic", "reduce"})
_GLU_ACTIVATIONS = frozenset({"swiglu", "geglu", "reglu"})
_STAGE1_PERSISTENT_TOKENS = 4096
_STAGE1_PERSISTENT_SHAPE = (3584, 512, 896, 16)
_STAGE1_PERSISTENT_TILE_MS = frozenset({64, 80, 96, 112})
_GEMM1_ACTIVATIONS = {
    # gemm1's historical ``silu`` spelling means the fused SwiGLU epilogue.
    "swiglu": "silu",
    "geglu": "geglu",
    "reglu": "reglu",
    "gelu_tanh_approx": "gelu_tanh_approx",
    "relu": "relu",
    "silu": "silu_pointwise",
    "relu_sq": "relu_sq",
}


def _validate_training_preactivation_extent(
    tokens: int, top_k: int, intermediate_size: int
) -> None:
    """Keep masked route-state stores below their signed-i32 sentinel.

    The training GEMM uses ``0x7fffffff`` as the byte offset for masked-off
    lanes.  Consequently the state resource itself must end below that
    sentinel, even though ordinary FlyDSL buffer addressing permits the full
    unsigned 32-bit range.
    """

    state_bytes = tokens * top_k * 2 * intermediate_size * 2
    if state_bytes > _MAX_SIGNED_I32:
        raise ValueError(
            "training preactivation exceeds the kernel's signed 32-bit "
            "masked-store byte-offset limit: "
            f"tokens={tokens}, top_k={top_k}, intermediate_size={intermediate_size}"
        )


def _validate_routes_training_preactivation_extent(
    routes: int, intermediate_size: int
) -> None:
    """Keep flat route-order state below the signed-i32 store limit."""

    state_bytes = routes * 2 * intermediate_size * 2
    if state_bytes > _MAX_SIGNED_I32:
        raise ValueError(
            "flat training preactivation exceeds the kernel's signed 32-bit "
            "masked-store byte-offset limit: "
            f"routes={routes}, intermediate_size={intermediate_size}"
        )


@dataclass(frozen=True)
class SonicMoEConfig:
    """Static shape and tile configuration for :class:`SonicMoE`.

    ``tile_m``/``tile_n``/``tile_k`` configure the stage-1 GEMM, and
    ``stage1_k_wave`` optionally partitions its four waves across K.  The
    down-projection defaults to the same tile values and can be tuned
    independently with ``down_tile_m``/``down_tile_n``/``down_tile_k``.
    Routing is padded to the least common multiple of both M tiles so the two
    GEMMs can share one sorted layout.  All tiles are compile-time constants.
    """

    hidden_size: int
    intermediate_size: int
    num_experts: int
    top_k: int
    tile_m: int = 32
    tile_n: int = 128
    tile_k: int = 128
    down_tile_m: int | None = None
    down_tile_n: int | None = None
    down_tile_k: int | None = None
    renormalize: bool = True
    stage1_b_cache_mod: int | None = None
    stage2_b_cache_mod: int | None = None
    stage1_xcd_swizzle: int = 0
    stage1_k_wave: int = 1
    stage2_xcd_swizzle: int = 1
    waves_per_eu: int | None = None
    persistent_stage2: bool = False
    stage2_output_mode: str = "atomic"
    activation: str = "swiglu"
    compute_dtype: str = "bf16"
    # Appended to preserve the positional ABI of every pre-existing field.
    stage2_pipeline_stages: int | None = None
    persistent_stage1: bool = False
    stage1_write_padded_rows: bool = False
    stage1_lds_swizzle: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.persistent_stage1, bool):
            raise TypeError(
                "persistent_stage1 must be bool, got "
                f"{type(self.persistent_stage1).__name__}"
            )
        if not isinstance(self.stage1_write_padded_rows, bool):
            raise TypeError(
                "stage1_write_padded_rows must be bool, got "
                f"{type(self.stage1_write_padded_rows).__name__}"
            )
        if not isinstance(self.stage1_lds_swizzle, bool):
            raise TypeError(
                "stage1_lds_swizzle must be bool, got "
                f"{type(self.stage1_lds_swizzle).__name__}"
            )
        if not isinstance(self.stage2_output_mode, str):
            raise TypeError(
                "stage2_output_mode must be a string, got "
                f"{type(self.stage2_output_mode).__name__}"
            )
        if self.stage2_output_mode not in _SUPPORTED_STAGE2_OUTPUT_MODES:
            raise ValueError(
                f"unsupported stage2_output_mode {self.stage2_output_mode!r}; expected one of "
                f"{sorted(_SUPPORTED_STAGE2_OUTPUT_MODES)}"
            )
        if self.compute_dtype not in _COMPUTE_DTYPES:
            raise ValueError(
                f"unsupported compute_dtype {self.compute_dtype!r}; expected one of "
                f"{sorted(_COMPUTE_DTYPES)}"
            )
        if not isinstance(self.activation, str):
            raise TypeError(
                f"activation must be a string, got {type(self.activation).__name__}"
            )
        if self.activation not in _SUPPORTED_ACTIVATIONS:
            raise ValueError(
                f"unsupported activation {self.activation!r}; expected one of "
                f"{sorted(_SUPPORTED_ACTIVATIONS)}"
            )
        positive = {
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "num_experts": self.num_experts,
            "top_k": self.top_k,
            "tile_m": self.tile_m,
            "tile_n": self.tile_n,
            "tile_k": self.tile_k,
            "down_tile_m": self.stage2_tile_m,
            "down_tile_n": self.stage2_tile_n,
            "down_tile_k": self.stage2_tile_k,
        }
        for name, value in positive.items():
            if int(value) <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.top_k > self.num_experts:
            raise ValueError(
                f"top_k ({self.top_k}) cannot exceed num_experts ({self.num_experts})"
            )
        if self.top_k > 16:
            raise ValueError(
                f"top_k must be <= 16 for the gfx950 router, got {self.top_k}"
            )
        if self.tile_m % 16 != 0 or self.stage2_tile_m % 16 != 0:
            raise ValueError(
                "tile_m and down_tile_m must be multiples of 16, got "
                f"{self.tile_m}/{self.stage2_tile_m}"
            )
        if self.tile_n % 64 != 0 or self.stage2_tile_n % 64 != 0:
            raise ValueError(
                "tile_n and down_tile_n must be multiples of 64, got "
                f"{self.tile_n}/{self.stage2_tile_n}"
            )
        if self.tile_k % 32 != 0 or self.stage2_tile_k % 32 != 0:
            raise ValueError(
                "tile_k and down_tile_k must be multiples of MFMA-K=32, got "
                f"{self.tile_k}/{self.stage2_tile_k}"
            )
        if self.tile_k & (self.tile_k - 1) or self.stage2_tile_k & (
            self.stage2_tile_k - 1
        ):
            raise ValueError(
                "tile_k and down_tile_k must be powers of two for the LDS swizzle, got "
                f"{self.tile_k}/{self.stage2_tile_k}"
            )
        # A direct-to-LDS copy uses 256 threads x 16 bytes. Each workgroup must
        # cover an integral number of those 4096-byte transfer rounds.
        if (self.tile_m * self.tile_k) % 2048 != 0:
            raise ValueError(
                "tile_m * tile_k must be a multiple of 2048 A16 elements for "
                "the stage1 direct-to-LDS copy"
            )
        if (self.stage2_tile_m * self.stage2_tile_k) % 2048 != 0:
            raise ValueError(
                "down_tile_m * down_tile_k must be a multiple of 2048 A16 elements for "
                "the stage2 direct-to-LDS copy"
            )
        if self.hidden_size % 32 != 0 or self.intermediate_size % 32 != 0:
            raise ValueError(
                "hidden_size and intermediate_size must be multiples of 32 for the "
                "16-bit preshuffle"
            )
        if self.hidden_size % self.tile_k != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by tile_k ({self.tile_k})"
            )
        if self.stage1_k_wave not in (1, 2, 4):
            raise ValueError(
                f"stage1_k_wave must be 1, 2, or 4, got {self.stage1_k_wave}"
            )
        stage1_k_span = self.stage1_k_wave * self.tile_k
        if self.hidden_size % stage1_k_span != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by "
                f"stage1_k_wave * tile_k ({self.stage1_k_wave} * {self.tile_k} = {stage1_k_span})"
            )
        if self.intermediate_size % self.tile_n != 0:
            raise ValueError(
                "intermediate_size "
                f"({self.intermediate_size}) must be divisible by tile_n ({self.tile_n})"
            )
        if self.intermediate_size % self.stage2_tile_k != 0:
            raise ValueError(
                "intermediate_size "
                f"({self.intermediate_size}) must be divisible by down_tile_k ({self.stage2_tile_k})"
            )
        if self.hidden_size % self.stage2_tile_n != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by "
                f"down_tile_n ({self.stage2_tile_n})"
            )
        if self.stage1_b_cache_mod not in (None, 0, 2):
            raise ValueError(
                "stage1_b_cache_mod must be None, 0 (cached), or 2 (non-temporal)"
            )
        if self.stage2_b_cache_mod not in (None, 0, 2):
            raise ValueError(
                "stage2_b_cache_mod must be None, 0 (cached), or 2 (non-temporal)"
            )
        if (
            self.stage2_pipeline_stages is not None
            and type(self.stage2_pipeline_stages) is not int
        ):
            raise TypeError(
                f"stage2_pipeline_stages must be None or an integer, got {type(self.stage2_pipeline_stages).__name__}"
            )
        if self.stage2_pipeline_stages not in (None, 1, 2):
            raise ValueError(
                f"stage2_pipeline_stages must be None (auto), 1, or 2, got {self.stage2_pipeline_stages!r}"
            )
        if self.stage1_xcd_swizzle < 0 or self.stage2_xcd_swizzle < 0:
            raise ValueError("XCD swizzle values must be non-negative")
        if self.persistent_stage1:
            static_shape = (
                self.hidden_size,
                self.intermediate_size,
                self.num_experts,
                self.top_k,
            )
            if static_shape != _STAGE1_PERSISTENT_SHAPE:
                raise ValueError(
                    "persistent_stage1 is currently restricted to the validated "
                    "gfx950 H3584/I512/E896/K16 production shape, got "
                    f"H{self.hidden_size}/I{self.intermediate_size}/"
                    f"E{self.num_experts}/K{self.top_k}"
                )
            if self.tile_m not in _STAGE1_PERSISTENT_TILE_MS:
                raise ValueError(
                    "persistent_stage1 requires a validated Stage-1 BM in "
                    f"{sorted(_STAGE1_PERSISTENT_TILE_MS)}, got {self.tile_m}"
                )
            if self.activation != "swiglu" or self.compute_dtype != "bf16":
                raise ValueError(
                    "persistent_stage1 currently requires BF16 SwiGLU, got "
                    f"activation={self.activation!r}, compute_dtype={self.compute_dtype!r}"
                )

        # Fail before JIT compilation if a tile cannot fit the gfx950 160 KiB LDS.
        stage1_k_tiles_per_wave = self.hidden_size // stage1_k_span
        stage1_stages = 2 if stage1_k_tiles_per_wave > 1 else 1
        stage1_a_lds = (
            self.stage1_k_wave * stage1_stages * self.tile_m * self.tile_k * 2
        )
        if self.stage1_k_wave > 1:
            stage1_n_waves = 4 // self.stage1_k_wave
            stage1_acc_n = (self.tile_n // stage1_n_waves) // 16
            stage1_m_repeat = self.tile_m // 16
            stage1_reduce_lds = 4 * (stage1_acc_n * stage1_m_repeat) * 64 * 4 * 4
            stage1_lds = max(stage1_a_lds, stage1_reduce_lds)
        else:
            stage1_lds = stage1_a_lds
        # Stage 2 reuses the A-tile storage for the FP32 epilogue only after
        # the contraction has finished.  Their lifetimes do not overlap, so
        # the kernel reserves the larger region rather than their sum.
        stage2_stages = self.stage2_effective_pipeline_stages
        stage2_lds = max(
            stage2_stages * self.stage2_tile_m * self.stage2_tile_k * 2,
            self.stage2_tile_m * self.stage2_tile_n * 4,
        )
        if stage1_lds > _GFX950_LDS_BYTES:
            raise ValueError(
                f"stage1 tile needs {stage1_lds} LDS bytes, exceeding gfx950's "
                f"{_GFX950_LDS_BYTES} bytes"
            )
        if stage2_lds > _GFX950_LDS_BYTES:
            raise ValueError(
                f"stage2 tile needs {stage2_lds} LDS bytes, exceeding gfx950's "
                f"{_GFX950_LDS_BYTES} bytes"
            )

    @property
    def stage2_tile_n(self) -> int:
        return self.tile_n if self.down_tile_n is None else self.down_tile_n

    @property
    def stage2_tile_m(self) -> int:
        return self.tile_m if self.down_tile_m is None else self.down_tile_m

    @property
    def stage2_tile_k(self) -> int:
        return self.tile_k if self.down_tile_k is None else self.down_tile_k

    @property
    def route_tile_m(self) -> int:
        """Padding/metadata granularity shared by both grouped GEMMs."""

        return math.lcm(self.tile_m, self.stage2_tile_m)

    @property
    def supports_flydsl_router(self) -> bool:
        """Whether the logits-to-top-k FlyDSL layout supports this expert count."""

        return supports_topk_gating_layout(self.num_experts)

    @property
    def is_glu(self) -> bool:
        return self.activation in _GLU_ACTIVATIONS

    @property
    def stage1_projection_size(self) -> int:
        return self.intermediate_size * (2 if self.is_glu else 1)

    @property
    def stage2_auto_pipeline_eligible(self) -> bool:
        """Whether this config matches the token-independent part of the auto gate."""

        common = (
            self.stage2_b_cache_mod in (None, 0)
            and self.waves_per_eu is None
            and not self.persistent_stage2
            and self.stage2_output_mode == "atomic"
            and self.compute_dtype == "bf16"
        )
        if not common:
            return False

        return (
            (
                self.hidden_size == 4096
                and self.intermediate_size == 2048
                and self.num_experts == 64
                and self.top_k == 8
                and (self.tile_m, self.tile_n, self.tile_k) == (128, 256, 64)
                and (self.stage2_tile_m, self.stage2_tile_n, self.stage2_tile_k)
                == (128, 128, 64)
                and self.stage1_xcd_swizzle == 0
                and self.stage2_xcd_swizzle == 8
                and not self.stage1_write_padded_rows
                and not self.stage1_lds_swizzle
            )
            or (
                self.hidden_size == 2048
                and self.intermediate_size == 768
                and self.num_experts == 128
                and self.top_k == 8
                and (self.tile_m, self.tile_n, self.tile_k) == (128, 192, 64)
                and (self.stage2_tile_m, self.stage2_tile_n, self.stage2_tile_k)
                == (64, 256, 64)
                and self.stage1_xcd_swizzle == 8
                and self.stage2_xcd_swizzle == 0
                and self.stage1_write_padded_rows
                and self.stage1_lds_swizzle
            )
            or (
                self.hidden_size == 4096
                and self.intermediate_size == 14336
                and self.num_experts == 8
                and self.top_k == 2
                and (self.tile_m, self.tile_n, self.tile_k) == (128, 256, 64)
                and (self.stage2_tile_m, self.stage2_tile_n, self.stage2_tile_k)
                == (128, 128, 64)
                and self.stage1_xcd_swizzle == 8
                and self.stage2_xcd_swizzle == 8
                and self.stage1_write_padded_rows
                and self.stage1_lds_swizzle
            )
        )

    @property
    def stage2_effective_pipeline_stages(self) -> int:
        """Pipeline depth the builder can realize, excluding token dispatch.

        A single K tile has nothing to overlap, so an explicit request for two
        stages intentionally normalizes to one, matching the GEMM builder.
        ``None`` reserves two-stage LDS only for the exact auto-policy shape.
        """

        requested = self.stage2_pipeline_stages
        if requested is None:
            requested = 2 if self.stage2_auto_pipeline_eligible else 1
        if self.stage2_output_mode != "atomic" or self.compute_dtype != "bf16":
            return 1
        k_tiles = self.intermediate_size // self.stage2_tile_k
        return 2 if requested == 2 and k_tiles > 1 else 1


@dataclass(frozen=True)
class SonicMoEWeights:
    """Prepared weights consumed by the gfx950 grouped MFMA kernels.

    ``weight_dtype`` is ``"bf16"``/``"fp16"`` for dense A16W16 or ``"mxfp4"``
    for per-1x32 E8M0-scaled FP4 weights. MXFP4 currently requires a BF16
    activation/intermediate ABI.
    The logical weight shapes are captured by ``config``; tile-only config
    changes do not require another preshuffle.
    """

    gate_up: torch.Tensor
    down: torch.Tensor
    dummy_scale: torch.Tensor
    config: SonicMoEConfig
    gate_up_scale: torch.Tensor | None = None
    down_scale: torch.Tensor | None = None
    weight_dtype: str = "bf16"
    stage1_bias: torch.Tensor | None = None
    stage2_bias: torch.Tensor | None = None

    @property
    def device(self) -> torch.device:
        return self.gate_up.device

    @property
    def tensors(self) -> tuple[torch.Tensor, ...]:
        tensors = [self.gate_up, self.down, self.dummy_scale]
        if self.stage1_bias is not None:
            tensors.append(self.stage1_bias)
        if self.stage2_bias is not None:
            tensors.append(self.stage2_bias)
        if self.gate_up_scale is not None:
            tensors.append(self.gate_up_scale)
        if self.down_scale is not None:
            tensors.append(self.down_scale)
        return tuple(tensors)

    @property
    def has_bias(self) -> bool:
        return self.stage1_bias is not None

    @property
    def compute_dtype(self) -> torch.dtype:
        return _COMPUTE_DTYPES[self.config.compute_dtype]


@dataclass(frozen=True)
class SonicMoEForwardState:
    """Immutable, invocation-owned state emitted by a training forward.

    ``preactivation`` is compact fixed-K route-order storage with shape
    ``[tokens, top_k, 2 * intermediate_size]``.  Its last dimension is either
    separate gate/up halves or native ``[g0, u0, ...]`` interleaving according
    to ``interleaved_w1``.  The tensor never aliases reusable
    :class:`SonicMoEWorkspace` storage.

    ``ready_event`` is recorded after the complete forward enqueue sequence.
    A future standalone backward consumer can skip a wait on
    ``producer_stream`` and otherwise wait on this event without a host
    synchronization.
    """

    preactivation: torch.Tensor
    tokens: int
    hidden_size: int
    intermediate_size: int
    num_experts: int
    top_k: int
    activation: str
    compute_dtype: str
    interleaved_w1: bool
    has_bias: bool
    producer_stream: int
    ready_event: torch.cuda.Event


@dataclass(frozen=True)
class SonicMoERoutesForwardState:
    """Invocation-owned training state for an arbitrary flat route list.

    ``preactivation`` has shape ``[routes, 2 * intermediate_size]`` and is in
    the original caller route order, even though Stage 1 computes in
    expert-sorted order.  Route metadata supplies the inverse mapping so Stage
    1 can write this compact state directly without a separate gather or
    permutation kernel.  Eligible dynamic Qwen3 E16 calls retain the six
    optional metadata tensors below.  They are invocation-owned and allow
    backward to skip sorting; other shapes leave them as ``None``.
    """

    preactivation: torch.Tensor
    tokens: int
    routes: int
    hidden_size: int
    intermediate_size: int
    num_experts: int
    activation: str
    compute_dtype: str
    interleaved_w1: bool
    has_bias: bool
    producer_stream: int
    ready_event: torch.cuda.Event
    # Eligible Qwen3 E16 training calls retain their forward route metadata so
    # backward can consume the same invocation-owned tensors without running
    # the four-launch ragged sorter.  Generic route shapes leave these optional
    # fields unset and retain the established backward-owned sort.
    sorted_token_ids: torch.Tensor | None = None
    sorted_route_ids: torch.Tensor | None = None
    sorted_weights: torch.Tensor | None = None
    sorted_expert_ids: torch.Tensor | None = None
    num_valid_ids: torch.Tensor | None = None
    expert_frequency: torch.Tensor | None = None
    # These are explicit, unchecked value contracts supplied by the caller.
    # They let backward remove the final route-to-token reduction when each
    # input route is already the unique row for the correspondingly numbered
    # token.  Older state producers simply omit the fields and retain the
    # conservative path.
    expert_major: bool = False
    token_indices_identity: bool = False
    # Rank-shared/coarsened route capacity used only for finite performance
    # policy selection.  Actual ``routes`` remains the allocation and bounds
    # authority.  ``None`` is retained for states from non-route entrypoints.
    route_policy_size: int | None = None


@dataclass
class SonicMoEWorkspace:
    """Reusable routing, intermediate, and output buffers for one route shape."""

    tokens: int
    routes: int | None
    route_tile_m: int
    max_padded_tokens: int
    # Number of route-metadata blocks.  Compute-grid upper bounds are larger
    # when a GEMM uses an M tile smaller than ``route_tile_m``.
    max_m_blocks: int
    stage1_max_m_blocks: int
    stage2_max_m_blocks: int
    sorted_token_ids: torch.Tensor
    sorted_route_ids: torch.Tensor | None
    sorted_weights: torch.Tensor
    sorted_expert_ids: torch.Tensor
    num_valid_ids: torch.Tensor
    sorting_workspace: torch.Tensor | None
    expert_frequency: torch.Tensor
    router_topk_weights: torch.Tensor
    router_topk_ids: torch.Tensor
    router_topk_expert_indices: torch.Tensor
    intermediate: torch.Tensor
    route_output: torch.Tensor | None
    # Exact-shape workspaces own a reusable default result.  Growable
    # expert-major workspaces deliberately do not: their active shape changes
    # between invocations, so the public API already returns invocation-owned
    # storage to keep earlier results alive.
    output: torch.Tensor | None
    _launch_lock: threading.Lock = field(
        default_factory=threading.Lock,
        init=False,
        repr=False,
        compare=False,
    )

    @functools.cached_property
    def storage_ptrs(self) -> frozenset[int]:
        """Storage bases owned by this workspace, cached off the hot path."""

        tensors = [
            self.sorted_token_ids,
            self.sorted_weights,
            self.sorted_expert_ids,
            self.num_valid_ids,
            self.expert_frequency,
            self.router_topk_weights,
            self.router_topk_ids,
            self.router_topk_expert_indices,
            self.intermediate,
        ]
        if self.output is not None:
            tensors.append(self.output)
        if self.sorted_route_ids is not None:
            tensors.append(self.sorted_route_ids)
        if self.route_output is not None:
            tensors.append(self.route_output)
        if self.sorting_workspace is not None:
            tensors.append(self.sorting_workspace)
        return frozenset(tensor.untyped_storage().data_ptr() for tensor in tensors)

    def flat_active_view(
        self,
        config: SonicMoEConfig,
        tokens: int,
        routes: int,
    ) -> SonicMoEWorkspace:
        """Return a current-shape view of a growable flat-route allocation."""

        if self.routes is None:
            raise ValueError("flat_active_view requires a flat-route workspace")
        if tokens < 0 or routes < 0 or tokens > self.tokens or routes > self.routes:
            raise ValueError(
                "active flat shape must fit the workspace capacity, got "
                f"T={tokens}/R={routes} within T={self.tokens}/R={self.routes}"
            )
        active_experts = min(config.num_experts, routes)
        max_blocks = (
            routes + active_experts * (config.route_tile_m - 1)
        ) // config.route_tile_m
        max_padded = max_blocks * config.route_tile_m
        if max_padded > self.max_padded_tokens:
            raise RuntimeError("active padded route bound exceeds workspace capacity")

        view = SonicMoEWorkspace(
            tokens=tokens,
            routes=routes,
            route_tile_m=config.route_tile_m,
            max_padded_tokens=max_padded,
            max_m_blocks=max_blocks,
            stage1_max_m_blocks=max_padded // config.tile_m,
            stage2_max_m_blocks=max_padded // config.stage2_tile_m,
            sorted_token_ids=self.sorted_token_ids[: max(1, max_padded)],
            sorted_route_ids=(
                self.sorted_route_ids[: max(1, max_padded)]
                if self.sorted_route_ids is not None
                else None
            ),
            sorted_weights=self.sorted_weights[: max(1, max_padded)],
            sorted_expert_ids=self.sorted_expert_ids[: max(1, max_blocks)],
            num_valid_ids=self.num_valid_ids,
            sorting_workspace=self.sorting_workspace,
            expert_frequency=self.expert_frequency,
            router_topk_weights=self.router_topk_weights[:tokens],
            router_topk_ids=self.router_topk_ids[:tokens],
            router_topk_expert_indices=self.router_topk_expert_indices[:tokens],
            intermediate=self.intermediate[: max(1, max_padded)],
            route_output=None,
            output=(self.output[:tokens] if self.output is not None else None),
        )
        # Views of one backing allocation must serialize through the same
        # enqueue lock even though each invocation carries its own active
        # shape metadata.
        view._launch_lock = self._launch_lock
        return view

    @classmethod
    def allocate(
        cls,
        config: SonicMoEConfig,
        tokens: int,
        device: torch.device,
        *,
        routes: int | None = None,
        reusable_output: bool = True,
    ) -> SonicMoEWorkspace:
        if tokens < 0:
            raise ValueError(f"tokens must be non-negative, got {tokens}")
        if routes is not None and routes < 0:
            raise ValueError(f"routes must be non-negative, got {routes}")

        # If A experts are active, Q padded route tiles need at least
        # Q*route_tile_m - A*(route_tile_m-1) real routes. Dense top-k routing
        # additionally has at most one edge per (token, expert); flat routing
        # deliberately supports duplicates, so it cannot use that tighter bound.
        route_count = tokens * config.top_k if routes is None else int(routes)
        active_experts = min(config.num_experts, route_count)
        route_tile_m = config.route_tile_m
        padding_bound = (
            route_count + active_experts * (route_tile_m - 1)
        ) // route_tile_m
        if routes is None:
            per_expert_bound = active_experts * (
                (tokens + route_tile_m - 1) // route_tile_m
            )
            max_blocks = min(padding_bound, per_expert_bound)
        else:
            max_blocks = padding_bound
        max_padded = max_blocks * route_tile_m
        if max_padded > _MAX_SIGNED_I32:
            raise ValueError(
                "padded route count exceeds the sorting kernel's signed 32-bit "
                f"index limit: {max_padded}"
            )
        if max_padded * 4 > _MAX_BUFFER_BYTE_OFFSET:
            raise ValueError(
                "sorted route metadata exceeds the kernel's 32-bit byte-offset "
                f"limit: max_padded={max_padded}"
            )
        if max_padded * config.intermediate_size * 2 > _MAX_BUFFER_BYTE_OFFSET:
            raise ValueError(
                "sorted 16-bit intermediate exceeds the kernel's 32-bit byte-offset limit: "
                f"max_padded={max_padded}, intermediate_size={config.intermediate_size}"
            )
        if routes is None:
            sorting_workspace_size = moe_sorting_get_workspace_size(
                tokens,
                config.num_experts,
                config.top_k,
                unit_size=route_tile_m,
            )
            mesh_stride = ((tokens + route_tile_m - 1) // route_tile_m) * route_tile_m
            if config.num_experts * mesh_stride > _MAX_SIGNED_I32:
                raise ValueError(
                    "sorting mesh exceeds the kernel's signed 32-bit byte-index limit: "
                    f"experts={config.num_experts}, mesh_stride={mesh_stride}"
                )
        else:
            # Flat sorting only needs one atomic cursor per expert.  The exact
            # frequency is kept separately because the prefix phase must not
            # destroy the public frequency result.
            sorting_workspace_size = config.num_experts
        if sorting_workspace_size * 4 > _MAX_BUFFER_BYTE_OFFSET:
            raise ValueError(
                "sorting workspace exceeds the kernel's 32-bit byte-offset limit: "
                f"{sorting_workspace_size * 4} bytes"
            )
        return cls(
            tokens=tokens,
            routes=routes,
            route_tile_m=route_tile_m,
            max_padded_tokens=max_padded,
            max_m_blocks=max_blocks,
            stage1_max_m_blocks=max_padded // config.tile_m,
            stage2_max_m_blocks=max_padded // config.stage2_tile_m,
            # Keep a one-element backing allocation for the all-empty ragged
            # case so raw buffer descriptors never receive a null data pointer.
            sorted_token_ids=torch.empty(
                max(1, max_padded), dtype=torch.int32, device=device
            ),
            sorted_route_ids=(
                torch.empty(max(1, max_padded), dtype=torch.int32, device=device)
                if routes is not None
                else None
            ),
            sorted_weights=torch.empty(
                max(1, max_padded), dtype=torch.float32, device=device
            ),
            sorted_expert_ids=torch.empty(
                max(1, max_blocks), dtype=torch.int32, device=device
            ),
            num_valid_ids=torch.empty(2, dtype=torch.int32, device=device),
            sorting_workspace=(
                torch.empty(sorting_workspace_size, dtype=torch.int32, device=device)
                if sorting_workspace_size
                else None
            ),
            expert_frequency=torch.empty(
                config.num_experts, dtype=torch.int32, device=device
            ),
            router_topk_weights=torch.empty(
                (tokens, config.top_k), dtype=torch.float32, device=device
            ),
            router_topk_ids=torch.empty(
                (tokens, config.top_k), dtype=torch.int32, device=device
            ),
            router_topk_expert_indices=torch.empty(
                (tokens, config.top_k), dtype=torch.int32, device=device
            ),
            intermediate=torch.empty(
                (max(1, max_padded), config.intermediate_size),
                dtype=_COMPUTE_DTYPES[config.compute_dtype],
                device=device,
            ),
            route_output=(
                torch.empty(
                    (tokens, config.top_k, config.hidden_size),
                    dtype=_COMPUTE_DTYPES[config.compute_dtype],
                    device=device,
                )
                if config.stage2_output_mode == "reduce" and routes is None
                else None
            ),
            output=(
                torch.empty(
                    (tokens, config.hidden_size),
                    dtype=_COMPUTE_DTYPES[config.compute_dtype],
                    device=device,
                )
                if reusable_output
                else None
            ),
        )


@dataclass(frozen=True)
class _TrainingForwardLaunchPlan:
    """Host-side launch choices which are invariant across one route bucket."""

    stage1: object
    stage2: object
    reduce: object | None
    grid1: int
    grid2: int
    output_mode: str
    route_policy_size: int | None


@dataclass
class _SonicMoEDynamicWorkspacePoolEntry:
    """One shared capacity and enqueue lock for a device/stream/config key."""

    capacity: SonicMoEWorkspace | None = None
    launch_lock: threading.Lock = field(default_factory=threading.Lock)


class SonicMoEDynamicWorkspacePool:
    """Explicitly shared growable flat-route scratch storage.

    A pool may be passed to multiple :class:`SonicMoE` instances.  Entries are
    isolated by the full :class:`SonicMoEConfig`, device, and current stream,
    and each entry owns one launch lock that survives capacity growth.  The
    pool, rather than any participating operator, is the sole long-lived owner
    of the current capacity; this prevents 48-layer models from retaining one
    high-water allocation per layer.

    Sharing is opt-in because it changes scratch lifetime and is deliberately
    unsupported during graph capture.  Call :meth:`clear` only while users of
    the pool are quiescent.  Dropping all operators and the pool releases its
    tensors naturally; a still-live pool can be released explicitly with
    :meth:`clear`.
    """

    def __init__(self, *, max_cached_workspaces: int = _DEFAULT_MAX_CACHED_WORKSPACES):
        if isinstance(max_cached_workspaces, bool) or not isinstance(
            max_cached_workspaces, int
        ):
            raise TypeError("max_cached_workspaces must be an integer")
        if max_cached_workspaces <= 0:
            raise ValueError("max_cached_workspaces must be positive")
        self._max_cached_workspaces = max_cached_workspaces
        self._entries: OrderedDict[
            tuple[int, int, SonicMoEConfig],
            _SonicMoEDynamicWorkspacePoolEntry,
        ] = OrderedDict()
        self._lock = threading.RLock()

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def clear(self) -> None:
        """Drop all cached capacities; callers must first quiesce pool users."""

        with self._lock:
            self._entries.clear()

    def reserve(
        self,
        config: SonicMoEConfig,
        tokens: int,
        routes: int,
        device: torch.device,
    ) -> SonicMoEWorkspace:
        """Return an active view over the key's single growable capacity."""

        if tokens < 0:
            raise ValueError(f"tokens must be non-negative, got {tokens}")
        if routes < 0:
            raise ValueError(f"routes must be non-negative, got {routes}")
        with torch.cuda.device(device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "shared dynamic SonicMoE workspaces are not graph-capture safe; "
                    "use per-operator workspaces for captured expert-major inference"
                )
            stream_id = int(torch.cuda.current_stream(device).cuda_stream)
        key = (device.index or 0, stream_id, config)
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                entry = _SonicMoEDynamicWorkspacePoolEntry()
                self._entries[key] = entry
            else:
                self._entries.move_to_end(key)

            capacity = entry.capacity
            needs_growth = (
                capacity is None
                or tokens > capacity.tokens
                or capacity.routes is None
                or routes > capacity.routes
            )
            if needs_growth:
                old_tokens = 0 if capacity is None else capacity.tokens
                old_routes = (
                    0
                    if capacity is None or capacity.routes is None
                    else capacity.routes
                )
                capacity_tokens = max(tokens, old_tokens + max(1, old_tokens // 2))
                capacity_routes = max(routes, old_routes + max(1, old_routes // 2))
                try:
                    capacity = SonicMoEWorkspace.allocate(
                        config,
                        capacity_tokens,
                        device,
                        routes=capacity_routes,
                        reusable_output=False,
                    )
                except ValueError:
                    capacity = SonicMoEWorkspace.allocate(
                        config,
                        tokens,
                        device,
                        routes=routes,
                        reusable_output=False,
                    )
                capacity._launch_lock = entry.launch_lock
                entry.capacity = capacity

            while len(self._entries) > self._max_cached_workspaces:
                self._entries.popitem(last=False)
            assert capacity is not None
            return capacity.flat_active_view(config, tokens, routes)


def _preshuffle_16bit_weight(
    weight: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Convert ``[..., N, K]`` FP16/BF16 rows to the 16x16 N-major layout."""

    n, k = weight.shape[-2:]
    if n % 16 != 0 or k % 32 != 0:
        raise ValueError(f"weight N/K must be divisible by 16/32, got {n}/{k}")
    if dtype not in (torch.float16, torch.bfloat16):
        raise TypeError(
            f"dense SonicMoE weight dtype must be FP16 or BF16, got {dtype}"
        )
    x = weight.detach().to(dtype=dtype).contiguous()
    leading = x.numel() // (n * k)
    # BK=32, KPack=8 16-bit values (16 bytes), BN=16.
    return (
        x.view(leading, n // 16, 16, k // 32, 4, 8)
        .permute(0, 1, 3, 4, 2, 5)
        .contiguous()
        .view_as(x)
    )


def _validate_weight_inputs(
    w1: torch.Tensor,
    w2: torch.Tensor,
    config: SonicMoEConfig,
) -> None:
    expected_w1 = (
        config.num_experts,
        config.stage1_projection_size,
        config.hidden_size,
    )
    expected_w2 = (config.num_experts, config.hidden_size, config.intermediate_size)
    if tuple(w1.shape) != expected_w1:
        raise ValueError(f"w1 must have shape {expected_w1}, got {tuple(w1.shape)}")
    if tuple(w2.shape) != expected_w2:
        raise ValueError(f"w2 must have shape {expected_w2}, got {tuple(w2.shape)}")
    if not w1.is_cuda or not w2.is_cuda:
        raise ValueError("SonicMoE weights must be on a ROCm device")
    if w1.device != w2.device:
        raise ValueError(f"w1 and w2 must share a device, got {w1.device}/{w2.device}")
    if not (w1.dtype.is_floating_point and w2.dtype.is_floating_point):
        raise TypeError(f"w1/w2 must be floating point, got {w1.dtype}/{w2.dtype}")


def _prepare_biases(
    b1: torch.Tensor | None,
    b2: torch.Tensor | None,
    w1: torch.Tensor,
    config: SonicMoEConfig,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Validate optional logical expert biases and materialize the A16 ABI."""

    if (b1 is None) != (b2 is None):
        raise ValueError("b1 and b2 must either both be provided or both be None")
    if b1 is None:
        return None, None
    assert b2 is not None

    expected_b1 = (config.num_experts, config.stage1_projection_size)
    expected_b2 = (config.num_experts, config.hidden_size)
    if tuple(b1.shape) != expected_b1:
        raise ValueError(f"b1 must have shape {expected_b1}, got {tuple(b1.shape)}")
    if tuple(b2.shape) != expected_b2:
        raise ValueError(f"b2 must have shape {expected_b2}, got {tuple(b2.shape)}")
    if not b1.is_cuda or not b2.is_cuda:
        raise ValueError("SonicMoE biases must be on a ROCm device")
    if b1.device != w1.device or b2.device != w1.device:
        raise ValueError("b1/b2 and expert weights must share a device")
    if not (b1.dtype.is_floating_point and b2.dtype.is_floating_point):
        raise TypeError(f"b1/b2 must be floating point, got {b1.dtype}/{b2.dtype}")
    compute_dtype = _COMPUTE_DTYPES[config.compute_dtype]
    return (
        b1.detach().to(dtype=compute_dtype, copy=True).contiguous(),
        b2.detach().to(dtype=compute_dtype, copy=True).contiguous(),
    )


def _round_up(value: int, multiple: int) -> int:
    return ((int(value) + multiple - 1) // multiple) * multiple


def _mxfp4_scale_storage_numel(experts: int, rows: int, k: int) -> int:
    """Physical uint8 elements in the padded gfx950 E8M0 scale layout."""

    return _round_up(experts * rows, 256) * _round_up(k // 32, 8)


def _validate_dense_resource_limits(config: SonicMoEConfig) -> None:
    # Each expert gets a 64-bit resource base, but offsets within it remain u32.
    gate_up_bytes_per_expert = config.stage1_projection_size * config.hidden_size * 2
    if gate_up_bytes_per_expert > _MAX_BUFFER_BYTE_OFFSET:
        raise ValueError(
            "dense 16-bit gate/up weights for one expert exceed the 32-bit "
            "byte-offset limit: "
            f"{gate_up_bytes_per_expert} bytes"
        )


def _validate_mxfp4_resource_limits(config: SonicMoEConfig) -> None:
    """Guard per-expert packed weights and whole-tensor E8M0 scale spans."""

    packed_gate_up_bytes_per_expert = (
        config.stage1_projection_size * config.hidden_size // 2
    )
    if packed_gate_up_bytes_per_expert > _MAX_BUFFER_BYTE_OFFSET:
        raise ValueError(
            "MXFP4 gate/up weights for one expert exceed the 32-bit byte-offset limit: "
            f"{packed_gate_up_bytes_per_expert} bytes"
        )

    gate_scale_cols = _round_up(config.hidden_size // 32, 8)
    down_scale_cols = _round_up(config.intermediate_size // 32, 8)
    spans = {
        "gate/up E8M0 scales": config.num_experts
        * config.stage1_projection_size
        * gate_scale_cols,
        "down E8M0 scales": config.num_experts * config.hidden_size * down_scale_cols,
    }
    for name, span in spans.items():
        if span > _MAX_BUFFER_BYTE_OFFSET:
            raise ValueError(
                f"{name} require a {span}-byte whole-tensor resource, exceeding "
                "the current 32-bit buffer-offset limit"
            )


def _validate_prepared_weight_storage(
    weights: SonicMoEWeights,
    config: SonicMoEConfig,
) -> None:
    """Validate the exact physical ABI before passing raw pointers to kernels."""

    if not weights.gate_up.is_cuda or not weights.down.is_cuda:
        raise ValueError("prepared SonicMoE weights must be on a ROCm device")
    if weights.gate_up.device != weights.down.device:
        raise ValueError("prepared gate/up and down weights must share a device")
    if any(tensor.device != weights.device for tensor in weights.tensors):
        raise ValueError("all prepared weights and scales must share a device")
    if any(not tensor.is_contiguous() for tensor in weights.tensors):
        raise ValueError("all prepared weights and scales must be contiguous")
    if any(tensor.requires_grad for tensor in weights.tensors):
        raise ValueError(
            "prepared inference weights and scales must not require gradients"
        )
    if weights.dummy_scale.dtype != torch.uint8 or weights.dummy_scale.numel() < 1:
        raise TypeError("dummy_scale must be a non-empty contiguous uint8 tensor")
    if weights.gate_up.data_ptr() % 16 or weights.down.data_ptr() % 16:
        raise ValueError("prepared gate/up and down weights must be 16-byte aligned")
    if (weights.stage1_bias is None) != (weights.stage2_bias is None):
        raise ValueError(
            "prepared stage1_bias and stage2_bias must either both be present or absent"
        )
    if weights.stage1_bias is not None:
        expected_b1 = (config.num_experts, config.stage1_projection_size)
        expected_b2 = (config.num_experts, config.hidden_size)
        if tuple(weights.stage1_bias.shape) != expected_b1:
            raise ValueError(
                f"prepared stage1_bias must have shape {expected_b1}, "
                f"got {tuple(weights.stage1_bias.shape)}"
            )
        if tuple(weights.stage2_bias.shape) != expected_b2:
            raise ValueError(
                f"prepared stage2_bias must have shape {expected_b2}, "
                f"got {tuple(weights.stage2_bias.shape)}"
            )
        expected_dtype = _COMPUTE_DTYPES[config.compute_dtype]
        if (
            weights.stage1_bias.dtype != expected_dtype
            or weights.stage2_bias.dtype != expected_dtype
        ):
            raise TypeError(
                f"prepared SonicMoE biases must use {expected_dtype} storage"
            )
        if weights.stage1_bias.data_ptr() % 2 or weights.stage2_bias.data_ptr() % 2:
            raise ValueError("prepared SonicMoE biases must be 2-byte aligned")

    if weights.weight_dtype in ("bf16", "fp16"):
        expected_gate_up = (
            config.num_experts,
            config.stage1_projection_size,
            config.hidden_size,
        )
        expected_down = (
            config.num_experts,
            config.hidden_size,
            config.intermediate_size,
        )
        expected_dtype = _COMPUTE_DTYPES[config.compute_dtype]
        expected_format = config.compute_dtype
        if weights.weight_dtype != expected_format:
            raise ValueError(
                "dense prepared weight format must match compute_dtype, got "
                f"{weights.weight_dtype!r}/{config.compute_dtype!r}"
            )
        if (
            weights.gate_up.dtype != expected_dtype
            or weights.down.dtype != expected_dtype
        ):
            raise TypeError(
                f"{expected_format.upper()} prepared weights must use "
                f"{expected_dtype} storage"
            )
        if weights.gate_up_scale is not None or weights.down_scale is not None:
            raise ValueError(
                "dense prepared weights must not carry MXFP4 scale buffers"
            )
    else:
        if config.compute_dtype != "bf16":
            raise ValueError(
                "MXFP4 prepared weights currently require compute_dtype='bf16'"
            )
        expected_gate_up = (
            config.num_experts,
            config.stage1_projection_size,
            config.hidden_size // 2,
        )
        expected_down = (
            config.num_experts,
            config.hidden_size,
            config.intermediate_size // 2,
        )
        if weights.gate_up.dtype != torch.uint8 or weights.down.dtype != torch.uint8:
            raise TypeError("MXFP4 prepared weights must use packed uint8 storage")
        if weights.gate_up_scale is None or weights.down_scale is None:
            raise ValueError(
                "MXFP4 prepared weights require gate/up and down E8M0 scales"
            )
        if (
            weights.gate_up_scale.dtype != torch.uint8
            or weights.down_scale.dtype != torch.uint8
        ):
            raise TypeError("MXFP4 E8M0 scales must use uint8 storage")
        expected_gate_scale = _mxfp4_scale_storage_numel(
            config.num_experts,
            config.stage1_projection_size,
            config.hidden_size,
        )
        expected_down_scale = _mxfp4_scale_storage_numel(
            config.num_experts,
            config.hidden_size,
            config.intermediate_size,
        )
        if (
            weights.gate_up_scale.ndim != 1
            or weights.gate_up_scale.numel() != expected_gate_scale
        ):
            raise ValueError(
                "MXFP4 gate/up scale storage has the wrong padded size: "
                f"expected {expected_gate_scale}, got {weights.gate_up_scale.numel()}"
            )
        if (
            weights.down_scale.ndim != 1
            or weights.down_scale.numel() != expected_down_scale
        ):
            raise ValueError(
                "MXFP4 down scale storage has the wrong padded size: "
                f"expected {expected_down_scale}, got {weights.down_scale.numel()}"
            )
        if weights.gate_up_scale.data_ptr() % 4 or weights.down_scale.data_ptr() % 4:
            raise ValueError("MXFP4 E8M0 scale buffers must be 4-byte aligned")

    if tuple(weights.gate_up.shape) != expected_gate_up:
        raise ValueError(
            f"prepared gate/up storage must have shape {expected_gate_up}, "
            f"got {tuple(weights.gate_up.shape)}"
        )
    if tuple(weights.down.shape) != expected_down:
        raise ValueError(
            f"prepared down storage must have shape {expected_down}, "
            f"got {tuple(weights.down.shape)}"
        )


def _f32_to_e8m0(values: torch.Tensor) -> torch.Tensor:
    """Encode positive FP32 scales as E8M0 exponent bytes."""

    values = values.to(torch.float32).contiguous()
    bits = values.view(torch.int32)
    exponent = ((bits >> 23) & 0xFF).to(torch.uint8)
    is_nan_or_inf = exponent == 0xFF
    round_up = ((bits & 0x400000) > 0) & (
        ((bits & 0x200000) > 0) | ((bits & 0x1FFFFF) > 0) | (exponent > 0)
    )
    rounded = (exponent.to(torch.int16) + round_up.to(torch.int16)).clamp_max(0xFE)
    return torch.where(
        is_nan_or_inf,
        torch.full_like(exponent, 0xFF),
        rounded.to(torch.uint8),
    )


def _e8m0_to_f32(scale: torch.Tensor) -> torch.Tensor:
    scale_u8 = scale.view(torch.uint8)
    bits = scale_u8.to(torch.int32) << 23
    bits = torch.where(scale_u8 == 0, torch.full_like(bits, 0x00400000), bits)
    bits = torch.where(scale_u8 == 0xFF, torch.full_like(bits, 0x7F800001), bits)
    return bits.view(torch.float32)


def _quantize_mxfp4_weight(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-1x32 E8M0 quantization to packed E2M1 codes.

    Quantization is row-chunked so preparing a large expert tensor does not
    materialize a second full-size FP32 weight tensor.
    """

    if weight.ndim != 3 or weight.shape[-1] % 32 != 0:
        raise ValueError("MXFP4 weights must be rank-3 with K divisible by 32")
    experts, rows, k = (int(v) for v in weight.shape)
    packed = torch.empty(
        (experts, rows, k // 2), dtype=torch.uint8, device=weight.device
    )
    scales = torch.empty(
        (experts, rows, k // 32), dtype=torch.uint8, device=weight.device
    )
    boundaries = torch.tensor(
        [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0],
        dtype=torch.float32,
        device=weight.device,
    )
    target_chunk_bytes = 64 * 1024 * 1024
    rows_per_chunk = max(1, target_chunk_bytes // (k * 4))

    packed_rows = packed.view(experts * rows, k // 2)
    scale_rows = scales.view(experts * rows, k // 32)

    def quantize_rows(chunk: torch.Tensor, output_row: int) -> None:
        chunk = chunk.to(torch.float32).contiguous()
        blocks = chunk.view(-1, 32)
        amax = blocks.abs().amax(dim=1)
        if not torch.isfinite(amax).all().item():
            raise ValueError("MXFP4 weights must contain only finite values")
        scale_e8m0 = _f32_to_e8m0(amax / 4.0)
        scale_f32 = _e8m0_to_f32(scale_e8m0)
        normalized = blocks / scale_f32[:, None]
        normalized_abs = normalized.abs()
        magnitude = torch.bucketize(normalized_abs, boundaries).to(torch.uint8)
        # torch.bucketize(right=False) selects the lower code at every exact
        # midpoint. MXFP4 uses round-to-nearest-even, so the three midpoints
        # whose lower code is odd must select the upper (even) code instead.
        rne_upper_tie = (
            (normalized_abs == 0.75)
            | (normalized_abs == 1.75)
            | (normalized_abs == 3.5)
        )
        magnitude = magnitude + rne_upper_tie.to(torch.uint8)
        codes = magnitude | (torch.signbit(normalized).to(torch.uint8) << 3)
        packed_chunk = (codes[:, 1::2] << 4) | codes[:, ::2]
        chunk_rows = int(chunk.shape[0])
        packed_rows[output_row : output_row + chunk_rows].copy_(
            packed_chunk.view(chunk_rows, k // 2)
        )
        scale_rows[output_row : output_row + chunk_rows].copy_(
            scale_e8m0.view(chunk_rows, k // 32)
        )

    if weight.is_contiguous():
        source_rows = weight.detach().view(experts * rows, k)
        for row_start in range(0, experts * rows, rows_per_chunk):
            row_end = min(row_start + rows_per_chunk, experts * rows)
            quantize_rows(source_rows[row_start:row_end], row_start)
    else:
        # Preserve the bounded-memory behavior for unusual strided inputs rather
        # than allowing reshape() to materialize the entire expert tensor.
        for expert in range(experts):
            for row_start in range(0, rows, rows_per_chunk):
                row_end = min(row_start + rows_per_chunk, rows)
                quantize_rows(
                    weight[expert, row_start:row_end].detach(),
                    expert * rows + row_start,
                )
    return packed, scales


def _dequantize_mxfp4_weight(
    packed: torch.Tensor,
    scales: torch.Tensor,
) -> torch.Tensor:
    """Decode an unshuffled MXFP4 tensor; used by correctness oracles."""

    if packed.dtype != torch.uint8 or scales.dtype != torch.uint8:
        raise TypeError("packed MXFP4 values and E8M0 scales must be uint8")
    codes = torch.empty(
        (*packed.shape[:-1], packed.shape[-1] * 2),
        dtype=torch.uint8,
        device=packed.device,
    )
    codes[..., ::2] = packed & 0xF
    codes[..., 1::2] = packed >> 4
    values = torch.tensor(
        [
            0.0,
            0.5,
            1.0,
            1.5,
            2.0,
            3.0,
            4.0,
            6.0,
            -0.0,
            -0.5,
            -1.0,
            -1.5,
            -2.0,
            -3.0,
            -4.0,
            -6.0,
        ],
        dtype=torch.float32,
        device=packed.device,
    )
    decoded = values[codes.long()]
    scale_f32 = _e8m0_to_f32(scales).repeat_interleave(32, dim=-1)
    if decoded.shape != scale_f32.shape:
        raise ValueError(
            f"packed/scales shapes are inconsistent: {tuple(decoded.shape)}/{tuple(scales.shape)}"
        )
    return decoded * scale_f32


def _preshuffle_mxfp4_weight(weight: torch.Tensor) -> torch.Tensor:
    """Preshuffle packed FP4 ``[..., N, K/2]`` into 16x16 byte tiles."""

    n, packed_k = (int(v) for v in weight.shape[-2:])
    if n % 16 != 0 or packed_k % 32 != 0:
        raise ValueError(
            f"packed MXFP4 weight N/(K/2) must be divisible by 16/32, got {n}/{packed_k}"
        )
    leading = weight.numel() // (n * packed_k)
    return (
        weight.view(leading, n // 16, 16, packed_k // 32, 2, 16)
        .permute(0, 1, 3, 4, 2, 5)
        .contiguous()
        .view_as(weight)
    )


def _preshuffle_e8m0_scale(scale: torch.Tensor) -> torch.Tensor:
    """Apply the gfx950 per-1x32 scale layout, including 256x8 padding."""

    if scale.ndim != 3:
        raise ValueError(
            f"scale must have shape [E, N, K/32], got {tuple(scale.shape)}"
        )
    rows = int(scale.shape[0] * scale.shape[1])
    cols = int(scale.shape[2])
    padded_rows = ((rows + 255) // 256) * 256
    padded_cols = ((cols + 7) // 8) * 8
    padded = torch.full(
        (padded_rows, padded_cols),
        127,
        dtype=torch.uint8,
        device=scale.device,
    )
    padded[:rows, :cols] = scale.reshape(rows, cols)
    return (
        padded.view(padded_rows // 32, 2, 16, padded_cols // 8, 2, 4)
        .permute(0, 3, 5, 2, 4, 1)
        .contiguous()
        .view(-1)
    )


def prepare_sonic_bf16_weights(
    w1: torch.Tensor,
    w2: torch.Tensor,
    config: SonicMoEConfig,
    *,
    b1: torch.Tensor | None = None,
    b2: torch.Tensor | None = None,
) -> SonicMoEWeights:
    """Validate and preshuffle stage-1 and down-projection weights.

    Parameters
    ----------
    w1:
        ``[num_experts, 2 * intermediate_size, hidden_size]`` in ``[gate | up]``
        order for GLU activations, otherwise
        ``[num_experts, intermediate_size, hidden_size]``. Floating input is
        converted to BF16 once during preparation.
    w2:
        ``[num_experts, hidden_size, intermediate_size]``.
    b1, b2:
        Optional expert-major biases with shapes
        ``[num_experts, stage1_projection_size]`` and
        ``[num_experts, hidden_size]``. They must be provided together and are
        copied to contiguous BF16 storage during preparation.
    """

    _validate_weight_inputs(w1, w2, config)
    if config.compute_dtype != "bf16":
        raise ValueError(
            "prepare_sonic_bf16_weights requires config.compute_dtype='bf16', got "
            f"{config.compute_dtype!r}"
        )
    _validate_dense_resource_limits(config)
    stage1_bias, stage2_bias = _prepare_biases(b1, b2, w1, config)

    return SonicMoEWeights(
        gate_up=_preshuffle_16bit_weight(w1, torch.bfloat16),
        down=_preshuffle_16bit_weight(w2, torch.bfloat16),
        dummy_scale=torch.zeros(1, dtype=torch.uint8, device=w1.device),
        config=config,
        stage1_bias=stage1_bias,
        stage2_bias=stage2_bias,
    )


def prepare_sonic_fp16_weights(
    w1: torch.Tensor,
    w2: torch.Tensor,
    config: SonicMoEConfig,
    *,
    b1: torch.Tensor | None = None,
    b2: torch.Tensor | None = None,
) -> SonicMoEWeights:
    """Validate and preshuffle dense FP16 stage-1/down-projection weights.

    The logical layouts match :func:`prepare_sonic_bf16_weights`. Floating
    inputs and optional expert biases are copied to contiguous FP16 storage.
    """

    if config.compute_dtype != "fp16":
        raise ValueError(
            "prepare_sonic_fp16_weights requires config.compute_dtype='fp16', got "
            f"{config.compute_dtype!r}"
        )
    _validate_weight_inputs(w1, w2, config)
    _validate_dense_resource_limits(config)
    stage1_bias, stage2_bias = _prepare_biases(b1, b2, w1, config)
    return SonicMoEWeights(
        gate_up=_preshuffle_16bit_weight(w1, torch.float16),
        down=_preshuffle_16bit_weight(w2, torch.float16),
        dummy_scale=torch.zeros(1, dtype=torch.uint8, device=w1.device),
        config=config,
        stage1_bias=stage1_bias,
        stage2_bias=stage2_bias,
        weight_dtype="fp16",
    )


def prepare_sonic_mxfp4_weights(
    w1: torch.Tensor,
    w2: torch.Tensor,
    config: SonicMoEConfig,
    *,
    b1: torch.Tensor | None = None,
    b2: torch.Tensor | None = None,
) -> SonicMoEWeights:
    """Quantize and preshuffle weight-only MXFP4 gate/up and down weights.

    Activations and the sorted stage-1 intermediate remain BF16 (A16W4). Each
    32-value weight block receives one E8M0 scale. This is the numerically
    validated low-memory path; it does not quantize activations to MXFP8.
    Optional ``b1``/``b2`` follow the same expert-major BF16 contract as
    :func:`prepare_sonic_bf16_weights`.
    """

    if config.compute_dtype != "bf16":
        raise ValueError("MXFP4 weights currently require config.compute_dtype='bf16'")
    _validate_weight_inputs(w1, w2, config)
    stage1_bias, stage2_bias = _prepare_biases(b1, b2, w1, config)
    if config.hidden_size % 64 != 0 or config.intermediate_size % 64 != 0:
        raise ValueError(
            "MXFP4 weight preshuffle requires hidden/intermediate sizes divisible by 64"
        )
    _validate_mxfp4_resource_limits(config)
    w1_quant, w1_scale = _quantize_mxfp4_weight(w1)
    w2_quant, w2_scale = _quantize_mxfp4_weight(w2)
    return SonicMoEWeights(
        gate_up=_preshuffle_mxfp4_weight(w1_quant),
        down=_preshuffle_mxfp4_weight(w2_quant),
        dummy_scale=torch.zeros(1, dtype=torch.uint8, device=w1.device),
        config=config,
        stage1_bias=stage1_bias,
        stage2_bias=stage2_bias,
        gate_up_scale=_preshuffle_e8m0_scale(w1_scale),
        down_scale=_preshuffle_e8m0_scale(w2_scale),
        weight_dtype="mxfp4",
    )


def _stage1_cache_mod(config: SonicMoEConfig, tokens: int) -> int:
    if config.stage1_b_cache_mod is not None:
        return config.stage1_b_cache_mod
    return 2 if 16 <= tokens <= 1024 else 0


def _stage2_cache_mod(config: SonicMoEConfig, tokens: int) -> int:
    if config.stage2_b_cache_mod is not None:
        return config.stage2_b_cache_mod
    return 0 if tokens <= 16 or tokens >= 2048 else 2


def _stage2_stages(config: SonicMoEConfig, tokens: int) -> int:
    """Select the measured gfx950 Stage-2 A-LDS pipeline depth.

    An explicit ``stage2_pipeline_stages`` value is a tuning override used by
    controlled gfx950 sweeps.  With the default ``None``, the two-stage
    implementation is intentionally gated to the one production
    bucket where paired AB/BA measurements showed a repeatable gain.  All other
    shapes retain the established serial loop.  Route layout, weight-format,
    bias, and actual output-mode checks live at the call site because they are
    properties of the prepared invocation rather than
    :class:`SonicMoEConfig` alone.
    """

    if config.stage2_pipeline_stages is not None:
        return config.stage2_effective_pipeline_stages

    return 2 if tokens == 4096 and config.stage2_auto_pipeline_eligible else 1


def _validate_stage1_persistent_runtime(
    config: SonicMoEConfig,
    tokens: int,
    arch: object,
) -> None:
    """Enforce the exact runtime bucket audited for Stage-1 persistence."""

    if not config.persistent_stage1:
        return
    if tokens != _STAGE1_PERSISTENT_TOKENS:
        raise ValueError(
            "persistent_stage1 is currently restricted to T4096, got "
            f"tokens={tokens}"
        )
    if not str(arch).startswith("gfx950"):
        raise RuntimeError("persistent_stage1 requires gfx950, got " f"{arch!r}")


@functools.lru_cache(maxsize=256)
def _get_stage1_launcher(
    config: SonicMoEConfig,
    b_cache_mod: int,
    weight_dtype: str,
    has_bias: bool,
    device_index: int,
    fused_identity_metadata: bool = False,
    identity_expert_counts_input: bool = False,
):
    # ``device_index`` is intentionally part of the LRU key.  Compiled
    # launchers cache a device-loaded function after first use and cannot be
    # shared across ROCm devices in one process.
    del device_index
    return compile_gemm1_a16w4_port(
        BM=config.tile_m,
        SORTED_BM=config.route_tile_m,
        D_HIDDEN=config.hidden_size,
        D_INTER=config.intermediate_size,
        NE=config.num_experts,
        TOPK=config.top_k,
        TILE_N=config.tile_n,
        TILE_K=config.tile_k,
        act=_GEMM1_ACTIVATIONS[config.activation],
        b_cache_mod=b_cache_mod,
        xcd_swizzle=config.stage1_xcd_swizzle,
        waves_per_eu=config.waves_per_eu,
        w_dtype=weight_dtype,
        a_dtype=config.compute_dtype,
        w_layout="standard",
        k_wave=config.stage1_k_wave,
        persist=config.persistent_stage1,
        round_preact_bf16=True,
        has_bias=has_bias,
        skip_epilogue_id_reload=config.stage1_write_padded_rows,
        a_lds_swizzle=config.stage1_lds_swizzle,
        fused_identity_metadata=fused_identity_metadata,
        identity_expert_counts_input=identity_expert_counts_input,
    )


@functools.lru_cache(maxsize=256)
def _get_stage1_training_launcher(
    config: SonicMoEConfig,
    b_cache_mod: int,
    has_bias: bool,
    interleaved_w1: bool,
    device_index: int,
    tile_n_override: int | None = None,
    waves_per_eu_override: int | None = None,
    route_preactivation_by_route_id: bool = False,
    tile_m_override: int | None = None,
    fused_identity_metadata: bool = False,
    identity_expert_counts_input: bool = False,
):
    """Compile the BF16 dual-output Stage-1 specialization."""

    del device_index
    tile_m = config.tile_m if tile_m_override is None else tile_m_override
    tile_n = config.tile_n if tile_n_override is None else tile_n_override
    waves_per_eu = (
        config.waves_per_eu if waves_per_eu_override is None else waves_per_eu_override
    )
    return compile_gemm1_a16w4_port(
        BM=tile_m,
        SORTED_BM=config.route_tile_m,
        D_HIDDEN=config.hidden_size,
        D_INTER=config.intermediate_size,
        NE=config.num_experts,
        TOPK=config.top_k,
        TILE_N=tile_n,
        TILE_K=config.tile_k,
        act=_GEMM1_ACTIVATIONS[config.activation],
        b_cache_mod=b_cache_mod,
        xcd_swizzle=config.stage1_xcd_swizzle,
        waves_per_eu=waves_per_eu,
        w_dtype="bf16",
        a_dtype="bf16",
        w_layout="standard",
        k_wave=config.stage1_k_wave,
        persist=config.persistent_stage1,
        round_preact_bf16=True,
        has_bias=has_bias,
        store_route_preactivation=True,
        route_preactivation_interleaved=interleaved_w1,
        route_preactivation_by_route_id=route_preactivation_by_route_id,
        a_lds_swizzle=config.stage1_lds_swizzle,
        fused_identity_metadata=fused_identity_metadata,
        identity_expert_counts_input=identity_expert_counts_input,
        identity_emit_route_ids=fused_identity_metadata,
    )


def _training_stage1_tuning(
    config: SonicMoEConfig,
    tokens: int,
    has_bias: bool,
) -> tuple[int, int | None]:
    """Select the measured gfx950 training-state Stage-1 specialization.

    The ordinary throughput profile uses BN256.  Its extra route-state scatter
    pushes this dual-output kernel to 512 total VGPRs and spills loop-carried
    values to scratch.  BN128 with a two-wave occupancy target keeps the same
    reduction order and route padding while cutting the dynamic scratch traffic
    and improving the complete forward/backward step for this production
    bucket.  Bias and user-retuned configurations retain their requested tiles.
    """

    if (
        tokens == 4096
        and config.hidden_size == 4096
        and config.intermediate_size == 2048
        and config.num_experts == 64
        and config.top_k == 8
        and config.tile_m == 128
        and config.tile_n == 256
        and config.tile_k == 64
        and config.route_tile_m == 128
        and config.stage1_k_wave == 1
        and config.stage1_b_cache_mod in (None, 0)
        and config.stage1_xcd_swizzle == 0
        and config.waves_per_eu is None
        and config.activation == "swiglu"
        and config.compute_dtype == "bf16"
        and not has_bias
    ):
        return 128, 2
    return config.tile_n, config.waves_per_eu


def _use_e16_expert_major_forward_tuning(
    config: SonicMoEConfig,
    tokens: int,
    routes: int | None,
    has_bias: bool,
    *,
    token_indices_identity: bool,
) -> bool:
    """Match the Qwen3 E16 expert-major path eligible for finite profiles."""

    return (
        token_indices_identity
        and routes is not None
        and routes == tokens
        and config.hidden_size == 2048
        and config.intermediate_size == 768
        and config.num_experts == 16
        and config.top_k == 1
        and (config.tile_m, config.tile_n, config.tile_k) == (128, 192, 64)
        and (
            config.stage2_tile_m,
            config.stage2_tile_n,
            config.stage2_tile_k,
        )
        == (64, 256, 64)
        and config.route_tile_m == 128
        and config.stage1_k_wave == 1
        and config.stage1_b_cache_mod in (None, 0)
        and config.stage2_b_cache_mod in (None, 0)
        and config.stage1_xcd_swizzle == 8
        and config.stage2_xcd_swizzle == 0
        and config.waves_per_eu is None
        and not config.persistent_stage1
        and not config.persistent_stage2
        and config.stage2_pipeline_stages == 2
        and config.stage2_output_mode == "atomic"
        and config.stage1_write_padded_rows
        and config.stage1_lds_swizzle
        and config.activation == "swiglu"
        and config.compute_dtype == "bf16"
        and not has_bias
    )


def _training_stage1_tile_m(
    config: SonicMoEConfig,
    tokens: int,
    routes: int | None,
    has_bias: bool,
    *,
    token_indices_identity: bool = False,
    route_policy: E16RoutePolicy | None = None,
) -> int:
    """Select Stage-1 M from a finite, rank-shareable route policy.

    The saturated large-route class uses BM64 to reduce register pressure.
    Smaller classes retain BM128: broad sweeps show that extending the old
    8K-only BM64 choice through 16K/32K hurts balanced workloads.  No exact
    route count or production interval participates in this choice.
    """

    if (
        route_policy is not None
        and route_policy == E16RoutePolicy.XLARGE
        and _use_e16_expert_major_forward_tuning(
            config,
            tokens,
            routes,
            has_bias,
            token_indices_identity=token_indices_identity,
        )
    ):
        return 64
    return config.tile_m


def _is_e16_flat_training_shape(
    config: SonicMoEConfig,
    tokens: int,
    routes: int,
    *,
    has_bias: bool,
) -> bool:
    """Match the dynamic Qwen3-30B-A3B EP8 flat-route fast path."""

    return (
        tokens == routes
        and config.hidden_size == 2048
        and config.intermediate_size == 768
        and config.num_experts == 16
        and config.top_k == 1
        and config.activation == "swiglu"
        and config.compute_dtype == "bf16"
        and not has_bias
    )


def _retain_e16_flat_route_metadata(
    config: SonicMoEConfig,
    tokens: int,
    routes: int,
    *,
    has_bias: bool,
) -> bool:
    """Keep forward route metadata for dynamic Qwen3 EP8 backward."""

    return _is_e16_flat_training_shape(
        config,
        tokens,
        routes,
        has_bias=has_bias,
    )


def _can_fuse_e16_identity_metadata_stage1(
    config: SonicMoEConfig,
    *,
    token_indices_identity: bool,
) -> bool:
    """Whether Stage 1 can own the complete identity-metadata production.

    Keep one definitive host predicate for launcher selection and execution:
    selecting an extended-ABI Stage-1 kernel and later falling back to the
    legacy call shape would be unsafe.
    """

    return (
        token_indices_identity
        and config.num_experts == 16
        and config.tile_m <= 256
        and config.route_tile_m % config.tile_m == 0
        and not config.persistent_stage1
    )


def _validate_expert_major_routes(
    expert_offsets: torch.Tensor | None,
    *,
    hidden_states: torch.Tensor,
    routes: int,
    num_experts: int,
    token_indices_identity: bool,
) -> bool:
    """Validate the structural part of the expert-major fast-path contract.

    Offset values deliberately remain an asynchronous caller precondition:
    ``offsets[0] == 0``, ``offsets[-1] == routes``, nondecreasing offsets, and
    expert ``e`` occupying ``[offsets[e], offsets[e + 1])``.  Checking those
    values here would put device-to-host synchronization back on the hot path.
    """

    if not isinstance(token_indices_identity, bool):
        raise TypeError("token_indices_identity must be bool")
    if expert_offsets is None:
        if token_indices_identity:
            raise ValueError(
                "token_indices_identity=True requires expert_offsets so the "
                "expert-major fast path can be selected"
            )
        return False
    if not isinstance(expert_offsets, torch.Tensor):
        raise TypeError("expert_offsets must be a torch.Tensor or None")
    expected_shape = (num_experts + 1,)
    if tuple(expert_offsets.shape) != expected_shape:
        raise ValueError(
            f"expert_offsets must have shape {expected_shape}, got "
            f"{tuple(expert_offsets.shape)}"
        )
    if (
        not expert_offsets.is_cuda
        or expert_offsets.device != hidden_states.device
        or expert_offsets.dtype != torch.int32
        or not expert_offsets.is_contiguous()
    ):
        raise ValueError(
            "expert_offsets must be contiguous int32 on the same ROCm device "
            "as hidden_states"
        )
    if token_indices_identity and routes != int(hidden_states.shape[0]):
        raise ValueError(
            "token_indices_identity=True requires routes == tokens, got "
            f"{routes} and {int(hidden_states.shape[0])}"
        )
    return True


def _validate_expert_major_counts(
    expert_counts: torch.Tensor | None,
    *,
    hidden_states: torch.Tensor,
    routes: int,
    num_experts: int,
) -> bool:
    """Validate device counts for the implicit E16 expert-major contract.

    Count values deliberately remain an asynchronous caller precondition:
    they must be non-negative and sum to ``routes``.  Reading them on the host
    here would serialize the dynamic training path.
    """

    if expert_counts is None:
        return False
    if not isinstance(expert_counts, torch.Tensor):
        raise TypeError("expert_counts must be a torch.Tensor or None")
    expected_shape = (num_experts,)
    if tuple(expert_counts.shape) != expected_shape:
        raise ValueError(
            f"expert_counts must have shape {expected_shape}, got "
            f"{tuple(expert_counts.shape)}"
        )
    if (
        not expert_counts.is_cuda
        or expert_counts.device != hidden_states.device
        or expert_counts.dtype != torch.int32
        or not expert_counts.is_contiguous()
    ):
        raise ValueError(
            "expert_counts must be contiguous int32 on the same ROCm device "
            "as hidden_states"
        )
    if routes != int(hidden_states.shape[0]):
        raise ValueError(
            "implicit expert-major counts require routes == tokens, got "
            f"{routes} and {int(hidden_states.shape[0])}"
        )
    return True


@functools.lru_cache(maxsize=256)
def _get_stage2_launcher(
    config: SonicMoEConfig,
    b_cache_mod: int,
    weight_dtype: str,
    has_bias: bool,
    output_mode: str,
    stages: int,
    device_index: int,
):
    # See _get_stage1_launcher: keep a distinct loaded function per device.
    del device_index
    return compile_gemm2_a16w4_port(
        BM=config.stage2_tile_m,
        SORTED_BM=config.route_tile_m,
        NE=config.num_experts,
        N_OUT=config.hidden_size,
        D_INTER=config.intermediate_size,
        TILE_N=config.stage2_tile_n,
        TILE_K=config.stage2_tile_k,
        xcd_swizzle=config.stage2_xcd_swizzle,
        b_cache_mod=b_cache_mod,
        waves_per_eu=config.waves_per_eu,
        w_dtype=weight_dtype,
        a_dtype=config.compute_dtype,
        persist=config.persistent_stage2,
        has_bias=has_bias,
        round_projection_bf16=True,
        output_mode=output_mode,
        TOPK=config.top_k,
        stages=stages,
    )


class SonicMoE:
    """Reusable gfx950 SonicMoE A16 forward operator.

    Exact-shape and growable workspaces use independent LRU caches, each bounded
    by ``max_cached_workspaces``. Exact entries are keyed by ``(device, stream,
    token_count, route_count)``; dense top-k calls use a dedicated route-count
    sentinel, while every flat-route call uses growable storage. Multiple
    operators may explicitly share growable flat-route
    scratch by receiving the same :class:`SonicMoEDynamicWorkspacePool`.
    Except for growable flat-route workspaces, the returned default output
    aliases that workspace and is
    overwritten by the next call with the same key; pass ``out=`` when the
    caller owns output storage. Growable route workspaces span multiple active
    shapes, so their default outputs are invocation-owned. Independent streams
    receive independent workspaces, while calls sharing a cache key serialize
    their complete kernel enqueue sequence. Eviction only drops the operator's
    reference: an in-progress call retains its workspace locally, and its
    buffers are allocated and used on the keyed stream so PyTorch's stream-aware
    allocator defers unsafe reuse. Call :meth:`clear_workspace` to release all
    cached entries eagerly.
    """

    def __init__(
        self,
        config: SonicMoEConfig,
        weights: SonicMoEWeights,
        *,
        max_cached_workspaces: int = _DEFAULT_MAX_CACHED_WORKSPACES,
        shared_dynamic_workspace_pool: SonicMoEDynamicWorkspacePool | None = None,
    ):
        prepared_shape = (
            weights.config.hidden_size,
            weights.config.intermediate_size,
            weights.config.num_experts,
            weights.config.activation,
            weights.config.compute_dtype,
        )
        requested_shape = (
            config.hidden_size,
            config.intermediate_size,
            config.num_experts,
            config.activation,
            config.compute_dtype,
        )
        if prepared_shape != requested_shape:
            raise ValueError(
                "prepared weights were created for different H/I/E/activation/dtype: "
                f"{prepared_shape} != {requested_shape}"
            )
        if weights.weight_dtype not in ("bf16", "fp16", "mxfp4"):
            raise ValueError(
                f"unsupported prepared weight dtype {weights.weight_dtype!r}"
            )
        if weights.weight_dtype == "mxfp4":
            if config.tile_k < 128 or config.stage2_tile_k < 128:
                raise ValueError(
                    "MXFP4 requires tile_k and down_tile_k >= 128 for packed FP4 loads"
                )
            _validate_mxfp4_resource_limits(config)
        else:
            _validate_dense_resource_limits(config)
        if config.persistent_stage1 and weights.weight_dtype != "bf16":
            raise ValueError(
                "persistent_stage1 is currently validated only for dense BF16 weights"
            )
        _validate_prepared_weight_storage(weights, config)
        if isinstance(max_cached_workspaces, bool) or not isinstance(
            max_cached_workspaces, int
        ):
            raise TypeError("max_cached_workspaces must be an integer")
        if max_cached_workspaces <= 0:
            raise ValueError("max_cached_workspaces must be positive")
        if shared_dynamic_workspace_pool is not None and not isinstance(
            shared_dynamic_workspace_pool,
            SonicMoEDynamicWorkspacePool,
        ):
            raise TypeError(
                "shared_dynamic_workspace_pool must be a "
                "SonicMoEDynamicWorkspacePool or None"
            )
        self.config = config
        self.weights = weights
        self._max_cached_workspaces = max_cached_workspaces
        self._shared_dynamic_workspace_pool = shared_dynamic_workspace_pool
        self._workspaces: OrderedDict[tuple[int, int, int, int], SonicMoEWorkspace] = (
            OrderedDict()
        )
        self._dynamic_route_workspaces: OrderedDict[
            tuple[int, int], SonicMoEWorkspace
        ] = OrderedDict()
        self._dynamic_route_workspace_locks: dict[tuple[int, int], threading.Lock] = {}
        self._workspace_lock = threading.RLock()
        self.workspace: SonicMoEWorkspace | None = None

    def clear_workspace(self) -> None:
        """Clear operator-owned caches, but never a separately owned shared pool."""

        with self._workspace_lock:
            self._workspaces.clear()
            self._dynamic_route_workspaces.clear()
            self._dynamic_route_workspace_locks.clear()
            self.workspace = None

    def reserve(self, tokens: int, *, routes: int | None = None) -> SonicMoEWorkspace:
        if routes is not None and routes < 0:
            raise ValueError(f"routes must be non-negative, got {routes}")
        stream_id = int(torch.cuda.current_stream(self.weights.device).cuda_stream)
        route_key = -1 if routes is None else int(routes)
        key = (self.weights.device.index or 0, stream_id, int(tokens), route_key)
        with self._workspace_lock:
            workspace = self._workspaces.get(key)
            if workspace is None:
                workspace = SonicMoEWorkspace.allocate(
                    self.config,
                    int(tokens),
                    self.weights.device,
                    routes=routes,
                )
                self._workspaces[key] = workspace
                while len(self._workspaces) > self._max_cached_workspaces:
                    self._workspaces.popitem(last=False)
            else:
                self._workspaces.move_to_end(key)
            self.workspace = workspace
            return workspace

    def reserve_dynamic_routes(
        self,
        tokens: int,
        routes: int,
    ) -> SonicMoEWorkspace:
        """Reserve one geometrically growing flat workspace per device/stream.

        Expert-parallel training commonly presents a different local route
        count on almost every microbatch.  Caching exact ``(T, R)`` shapes
        would retain many large intermediate/output tensors and repeatedly hit
        the allocator.  This path keeps one capacity allocation per stream and
        returns cheap current-shape views over it.
        """

        if tokens < 0:
            raise ValueError(f"tokens must be non-negative, got {tokens}")
        if routes < 0:
            raise ValueError(f"routes must be non-negative, got {routes}")
        if self._shared_dynamic_workspace_pool is not None:
            workspace = self._shared_dynamic_workspace_pool.reserve(
                self.config,
                tokens,
                routes,
                self.weights.device,
            )
            # Keeping an active view here would make every participating
            # operator retain the pool's superseded capacity after another
            # operator grows the shared entry.
            self.workspace = None
            return workspace
        stream_id = int(torch.cuda.current_stream(self.weights.device).cuda_stream)
        key = (self.weights.device.index or 0, stream_id)
        with self._workspace_lock:
            capacity = self._dynamic_route_workspaces.get(key)
            launch_lock = self._dynamic_route_workspace_locks.get(key)
            if launch_lock is None:
                launch_lock = threading.Lock()
                self._dynamic_route_workspace_locks[key] = launch_lock
            needs_growth = (
                capacity is None
                or tokens > capacity.tokens
                or capacity.routes is None
                or routes > capacity.routes
            )
            if needs_growth:
                old_tokens = 0 if capacity is None else capacity.tokens
                old_routes = (
                    0
                    if capacity is None or capacity.routes is None
                    else capacity.routes
                )
                capacity_tokens = max(tokens, old_tokens + max(1, old_tokens // 2))
                capacity_routes = max(routes, old_routes + max(1, old_routes // 2))
                try:
                    capacity = SonicMoEWorkspace.allocate(
                        self.config,
                        capacity_tokens,
                        self.weights.device,
                        routes=capacity_routes,
                        reusable_output=False,
                    )
                except ValueError:
                    # Geometric headroom can cross an addressing limit even
                    # though the requested active shape is valid.  Retain the
                    # growable policy at the exact legal boundary.
                    capacity = SonicMoEWorkspace.allocate(
                        self.config,
                        tokens,
                        self.weights.device,
                        routes=routes,
                        reusable_output=False,
                    )
                capacity._launch_lock = launch_lock
                self._dynamic_route_workspaces[key] = capacity
                while len(self._dynamic_route_workspaces) > self._max_cached_workspaces:
                    evicted_key, _ = self._dynamic_route_workspaces.popitem(last=False)
                    self._dynamic_route_workspace_locks.pop(evicted_key, None)
            else:
                self._dynamic_route_workspaces.move_to_end(key)
            assert capacity is not None
            workspace = capacity.flat_active_view(self.config, tokens, routes)
            self.workspace = workspace
            return workspace

    def _validate_hidden(
        self,
        hidden_states: torch.Tensor,
        *,
        allow_empty: bool = False,
    ) -> int:
        if not hidden_states.is_cuda:
            raise ValueError("hidden_states must be on a ROCm device")
        if hidden_states.device != self.weights.device:
            raise ValueError(
                f"hidden_states and weights must share a device, got "
                f"{hidden_states.device}/{self.weights.device}"
            )
        expected_dtype = self.weights.compute_dtype
        if hidden_states.dtype != expected_dtype:
            raise TypeError(
                f"hidden_states must use {expected_dtype}, got {hidden_states.dtype}"
            )
        if hidden_states.ndim != 2 or hidden_states.shape[1] != self.config.hidden_size:
            raise ValueError(
                f"hidden_states must have shape [tokens, {self.config.hidden_size}], "
                f"got {tuple(hidden_states.shape)}"
            )
        if not hidden_states.is_contiguous():
            raise ValueError("hidden_states must be contiguous")
        if hidden_states.data_ptr() % 16:
            raise ValueError(
                "hidden_states must be 16-byte aligned for direct-to-LDS loads"
            )
        if hidden_states.requires_grad:
            raise ValueError(
                "SonicMoE is inference-only; hidden_states must not require gradients"
            )
        tokens = int(hidden_states.shape[0])
        minimum_tokens = 0 if allow_empty else 1
        if tokens < minimum_tokens or tokens > 0xFFFFFF:
            raise ValueError(
                f"tokens must be in [{minimum_tokens}, 2^24-1], got {tokens}"
            )
        if tokens * self.config.hidden_size * 2 > _MAX_BUFFER_BYTE_OFFSET:
            raise ValueError(
                "16-bit atomic output addressing exceeds the 32-bit byte-offset limit: "
                f"tokens={tokens}, hidden_size={self.config.hidden_size}"
            )
        arch = get_rocm_arch()
        if not str(arch).startswith("gfx95"):
            raise RuntimeError(f"SonicMoE CDNA4 forward requires gfx95*, got {arch!r}")
        _validate_stage1_persistent_runtime(self.config, tokens, arch)
        return tokens

    def _validate_training_hidden(
        self,
        hidden_states: torch.Tensor,
        *,
        allow_empty: bool = False,
    ) -> int:
        """Apply the physical forward ABI while permitting autograd inputs."""

        return self._validate_hidden(
            hidden_states.detach() if hidden_states.requires_grad else hidden_states,
            allow_empty=allow_empty,
        )

    def _validate_out(
        self,
        out: torch.Tensor | None,
        workspace: SonicMoEWorkspace,
        *read_tensors: torch.Tensor,
        invocation_owned_default: bool = False,
    ) -> torch.Tensor:
        if out is None:
            if invocation_owned_default:
                out = torch.empty(
                    (workspace.tokens, self.config.hidden_size),
                    dtype=self.weights.compute_dtype,
                    device=self.weights.device,
                )
            else:
                if workspace.output is None:
                    raise RuntimeError(
                        "workspace does not own a reusable default output"
                    )
                out = workspace.output
        expected = (workspace.tokens, self.config.hidden_size)
        if tuple(out.shape) != expected:
            raise ValueError(f"out must have shape {expected}, got {tuple(out.shape)}")
        expected_dtype = self.weights.compute_dtype
        if (
            out.device != self.weights.device
            or out.dtype != expected_dtype
            or not out.is_contiguous()
        ):
            raise ValueError(
                f"out must be contiguous {expected_dtype} on the same ROCm device as the weights"
            )
        if out.data_ptr() % 4:
            raise ValueError(
                "out must be 4-byte aligned for packed 16-bit output stores"
            )
        if out.requires_grad:
            raise ValueError(
                "SonicMoE is inference-only; out must not require gradients"
            )

        workspace_storages = workspace.storage_ptrs
        if any(
            tensor.numel() > 0
            and tensor.untyped_storage().data_ptr() in workspace_storages
            for tensor in read_tensors
        ):
            raise ValueError(
                "inputs and prepared weights must not alias internal workspace storage"
            )

        out_storage = out.untyped_storage().data_ptr()
        if out.numel() > 0 and any(
            tensor.numel() > 0 and out_storage == tensor.untyped_storage().data_ptr()
            for tensor in read_tensors
        ):
            raise ValueError("out must not alias an input or prepared-weight storage")
        workspace_output_storage = (
            workspace.output.untyped_storage().data_ptr()
            if workspace.output is not None
            else None
        )
        if (
            out_storage in workspace_storages
            and out_storage != workspace_output_storage
        ):
            raise ValueError("out must not alias internal workspace scratch storage")
        if (
            workspace.output is not None
            and out_storage == workspace_output_storage
            and out.data_ptr() != workspace.output.data_ptr()
        ):
            raise ValueError(
                "out must start at the internal workspace output base address"
            )
        return out

    def _validate_expert_frequency_out(
        self,
        expert_frequency_out: torch.Tensor,
        workspace: SonicMoEWorkspace,
        output: torch.Tensor,
        *read_tensors: torch.Tensor,
    ) -> torch.Tensor:
        """Validate a caller-owned dense expert-frequency output buffer."""

        if not isinstance(expert_frequency_out, torch.Tensor):
            raise TypeError("expert_frequency_out must be a torch.Tensor")
        expected = (self.config.num_experts,)
        if tuple(expert_frequency_out.shape) != expected:
            raise ValueError(
                f"expert_frequency_out must have shape {expected}, "
                f"got {tuple(expert_frequency_out.shape)}"
            )
        if (
            not expert_frequency_out.is_cuda
            or expert_frequency_out.device != self.weights.device
            or expert_frequency_out.dtype != torch.int32
            or not expert_frequency_out.is_contiguous()
        ):
            raise ValueError(
                "expert_frequency_out must be contiguous int32 on the same ROCm device"
            )

        frequency_storage = expert_frequency_out.untyped_storage().data_ptr()
        if frequency_storage == output.untyped_storage().data_ptr() or any(
            frequency_storage == tensor.untyped_storage().data_ptr()
            for tensor in read_tensors
        ):
            raise ValueError("expert_frequency_out must not alias an input or output")
        if frequency_storage in workspace.storage_ptrs:
            raise ValueError(
                "expert_frequency_out must not alias internal workspace storage"
            )
        return expert_frequency_out

    def _run_grouped_gemms(
        self,
        hidden_states: torch.Tensor,
        workspace: SonicMoEWorkspace,
        out: torch.Tensor,
        *,
        token_indices_identity: bool = False,
        route_policy_size: int | None = None,
        identity_boundaries: torch.Tensor | None = None,
        identity_route_weights: torch.Tensor | None = None,
        identity_frequency: torch.Tensor | None = None,
        identity_frequency_mirror: torch.Tensor | None = None,
        identity_expert_counts_input: bool = False,
    ) -> torch.Tensor:
        cfg = self.config
        tokens = workspace.tokens
        stream = torch.cuda.current_stream(hidden_states.device)
        # Flat routes have no fixed slot dimension and may contain duplicate
        # edges, so only dense fixed-K workspaces may select the reduce path.
        output_mode = (
            cfg.stage2_output_mode
            if workspace.routes is None
            else ("identity" if token_indices_identity else "atomic")
        )
        if output_mode == "reduce" and workspace.route_output is None:
            raise RuntimeError(
                "reduce stage2 output requires a fixed-top-k route workspace"
            )

        # Expert-major inference shares the same finite scheduling policy as
        # training/backward.  The real token count remains the runtime extent
        # and grid bound; only compile-time cache/pipeline choices consult the
        # representative.  Fixed-top-k and generic ragged callers keep their
        # established token-based policy by leaving the hint unset.
        dynamic_e16_flat = (
            route_policy_size is not None
            and workspace.routes is not None
            and token_indices_identity
            and _is_e16_flat_training_shape(
                cfg,
                tokens,
                int(workspace.routes),
                has_bias=self.weights.has_bias,
            )
        )
        policy_tokens = route_policy_size if dynamic_e16_flat else tokens

        fused_identity_metadata = identity_boundaries is not None
        if fused_identity_metadata:
            if (
                identity_route_weights is None
                or identity_frequency is None
                or workspace.sorting_workspace is None
            ):
                raise RuntimeError(
                    "fused identity Stage 1 requires boundaries, weights, "
                    "frequency, and padded-prefix storage"
                )
            if identity_frequency_mirror is None:
                identity_frequency_mirror = identity_frequency

        stage1_args = (
            cfg,
            _stage1_cache_mod(cfg, policy_tokens),
            self.weights.weight_dtype,
            self.weights.has_bias,
            hidden_states.device.index or 0,
        )
        stage1 = (
            _get_stage1_launcher(
                *stage1_args,
                fused_identity_metadata=True,
                identity_expert_counts_input=identity_expert_counts_input,
            )
            if fused_identity_metadata
            else _get_stage1_launcher(*stage1_args)
        )
        grid1 = gemm1_a16w4_grid(
            cfg.tile_m,
            INTER=cfg.intermediate_size,
            TILE_N=cfg.tile_n,
            max_m_blocks=workspace.stage1_max_m_blocks,
            persist=cfg.persistent_stage1,
        )
        stage1_args = (
            hidden_states.data_ptr(),
            self.weights.gate_up.data_ptr(),
            (
                self.weights.dummy_scale
                if self.weights.gate_up_scale is None
                else self.weights.gate_up_scale
            ).data_ptr(),
            (
                self.weights.dummy_scale
                if self.weights.stage1_bias is None
                else self.weights.stage1_bias
            ).data_ptr(),
            workspace.sorted_expert_ids.data_ptr(),
            workspace.num_valid_ids.data_ptr(),
            workspace.sorted_token_ids.data_ptr(),
            tokens,
            max(1, int(grid1)) if fused_identity_metadata else int(grid1),
            1.0,
            1.0,
            1.0,
            1.0,
            float("inf"),
            workspace.intermediate.data_ptr(),
        )
        if fused_identity_metadata:
            assert identity_boundaries is not None
            assert identity_route_weights is not None
            assert identity_frequency is not None
            assert identity_frequency_mirror is not None
            assert workspace.sorting_workspace is not None
            _run_compiled(
                stage1,
                *stage1_args,
                identity_boundaries.data_ptr(),
                identity_route_weights.data_ptr(),
                identity_frequency.data_ptr(),
                identity_frequency_mirror.data_ptr(),
                workspace.sorting_workspace.data_ptr(),
                workspace.sorted_weights.data_ptr(),
                stream,
            )
        else:
            _run_compiled(stage1, *stage1_args, stream)

        # Empty expert-major calls still launch Stage 1 once to publish the
        # all-zero retained/public metadata, but have no GEMM2 work.
        if fused_identity_metadata and workspace.routes == 0:
            return out

        stage2_stages = _stage2_stages(cfg, policy_tokens)
        if (
            (workspace.routes is not None and not dynamic_e16_flat)
            or self.weights.weight_dtype != "bf16"
            or self.weights.has_bias
            or output_mode not in ("atomic", "identity")
        ):
            stage2_stages = 1
        stage2 = _get_stage2_launcher(
            cfg,
            _stage2_cache_mod(cfg, policy_tokens),
            self.weights.weight_dtype,
            self.weights.has_bias,
            output_mode,
            stage2_stages,
            hidden_states.device.index or 0,
        )
        grid2 = gemm2_a16w4_grid(
            cfg.stage2_tile_m,
            N_OUT=cfg.hidden_size,
            TILE_N=cfg.stage2_tile_n,
            max_m_blocks=workspace.stage2_max_m_blocks,
            persist=cfg.persistent_stage2,
        )
        _run_compiled(
            stage2,
            workspace.intermediate.data_ptr(),
            self.weights.down.data_ptr(),
            (
                self.weights.dummy_scale
                if self.weights.down_scale is None
                else self.weights.down_scale
            ).data_ptr(),
            (
                self.weights.dummy_scale
                if self.weights.stage2_bias is None
                else self.weights.stage2_bias
            ).data_ptr(),
            workspace.sorted_expert_ids.data_ptr(),
            workspace.num_valid_ids.data_ptr(),
            workspace.sorted_token_ids.data_ptr(),
            workspace.sorted_weights.data_ptr(),
            tokens,
            workspace.stage2_max_m_blocks,
            int(grid2),
            (workspace.route_output if output_mode == "reduce" else out).data_ptr(),
            stream,
        )
        if output_mode == "reduce":
            assert workspace.route_output is not None
            reduction_dtype = "f16" if cfg.compute_dtype == "fp16" else "bf16"
            reduce = compile_moe_reduction(
                topk=cfg.top_k,
                model_dim=cfg.hidden_size,
                dtype_str=reduction_dtype,
            )
            route_output_ptr = flyc.from_c_void_p(
                fx.Uint8, workspace.route_output.data_ptr()
            )
            out_ptr = flyc.from_c_void_p(fx.Uint8, out.data_ptr())
            unused_ptr = flyc.from_c_void_p(
                fx.Uint8, self.weights.dummy_scale.data_ptr()
            )
            _run_compiled(
                reduce,
                route_output_ptr,
                out_ptr,
                unused_ptr,
                unused_ptr,
                tokens,
                stream,
            )
        return out

    def _prepare_grouped_gemms_training(
        self,
        workspace: SonicMoEWorkspace,
        *,
        interleaved_w1: bool,
        token_indices_identity: bool,
        route_policy_size: int | None = None,
        fused_identity_metadata: bool = False,
        identity_expert_counts_input: bool = False,
    ) -> _TrainingForwardLaunchPlan:
        """Resolve launch choices before expert-major compute dispatch.

        Resolving cached JIT launchers, tuning policy, and grids on the critical
        path can leave a visible device bubble before Stage 1.  This helper does
        that host work first.  The returned plan is invocation-local: real EP
        training has highly variable route counts, so caching exact runtime
        grids here would merely churn a small LRU.
        """

        cfg = self.config
        tokens = workspace.tokens
        flat_routes = workspace.routes is not None
        route_policy = (
            select_e16_route_policy(int(workspace.routes), route_policy_size)
            if workspace.routes is not None
            else None
        )
        policy_representative = (
            e16_route_policy_representative(route_policy)
            if route_policy is not None
            else None
        )
        output_mode = (
            cfg.stage2_output_mode
            if not flat_routes
            else ("identity" if token_indices_identity else "atomic")
        )
        if output_mode == "reduce" and workspace.route_output is None:
            raise RuntimeError(
                "reduce stage2 output requires a fixed-top-k route workspace"
            )

        training_tile_n, training_waves_per_eu = _training_stage1_tuning(
            cfg,
            tokens,
            self.weights.has_bias,
        )
        training_tile_m = _training_stage1_tile_m(
            cfg,
            tokens,
            workspace.routes,
            self.weights.has_bias,
            token_indices_identity=token_indices_identity,
            route_policy=route_policy,
        )
        use_e16_expert_major_tuning = _use_e16_expert_major_forward_tuning(
            cfg,
            tokens,
            workspace.routes,
            self.weights.has_bias,
            token_indices_identity=token_indices_identity,
        )
        use_xlarge_e16_expert_major_tuning = (
            use_e16_expert_major_tuning and route_policy == E16RoutePolicy.XLARGE
        )
        if use_e16_expert_major_tuning and route_policy == E16RoutePolicy.SMALL:
            # This is a capacity-class specialization, not the former exact
            # R==4096 gate.  BN64 supplies enough independent N tiles for the
            # entire <=4K class; the next class returns to the baseline BN192.
            training_tile_n = 64
            training_waves_per_eu = None
        elif use_xlarge_e16_expert_major_tuning:
            training_tile_n = 128
            training_waves_per_eu = None

        # Cache policy is part of the compiled GEMM family.  Once the caller
        # supplies a rank-shared E16 route policy, do not let the exact local
        # route count select a different cache mode (for example around the
        # historical 2K cutoff).  Actual ``tokens`` still controls all tensor
        # bounds and the runtime launch grid below.
        policy_tokens = (
            policy_representative
            if use_e16_expert_major_tuning and policy_representative is not None
            else tokens
        )
        stage1_cache_mod = _stage1_cache_mod(cfg, policy_tokens)
        stage2_cfg = (
            replace(cfg, down_tile_n=128, stage2_pipeline_stages=1)
            if use_xlarge_e16_expert_major_tuning
            else cfg
        )
        stage2_stages = _stage2_stages(stage2_cfg, policy_tokens)
        dynamic_e16_flat = workspace.routes is not None and _is_e16_flat_training_shape(
            cfg,
            tokens,
            int(workspace.routes),
            has_bias=self.weights.has_bias,
        )
        if (
            (flat_routes and not dynamic_e16_flat)
            or self.weights.has_bias
            or output_mode not in ("atomic", "identity")
        ):
            stage2_stages = 1
        stage2_cache_mod = _stage2_cache_mod(stage2_cfg, policy_tokens)
        grid1 = gemm1_a16w4_grid(
            training_tile_m,
            INTER=cfg.intermediate_size,
            TILE_N=training_tile_n,
            max_m_blocks=workspace.max_padded_tokens // training_tile_m,
            persist=cfg.persistent_stage1,
        )
        grid2 = gemm2_a16w4_grid(
            stage2_cfg.stage2_tile_m,
            N_OUT=stage2_cfg.hidden_size,
            TILE_N=stage2_cfg.stage2_tile_n,
            max_m_blocks=workspace.stage2_max_m_blocks,
            persist=stage2_cfg.persistent_stage2,
        )
        device_index = self.weights.device.index or 0
        stage1_args = (
            cfg,
            stage1_cache_mod,
            self.weights.has_bias,
            interleaved_w1,
            device_index,
            training_tile_n,
            training_waves_per_eu,
            flat_routes,
            training_tile_m,
        )
        stage1 = (
            _get_stage1_training_launcher(
                *stage1_args,
                fused_identity_metadata=True,
                identity_expert_counts_input=identity_expert_counts_input,
            )
            if fused_identity_metadata
            else _get_stage1_training_launcher(*stage1_args)
        )
        stage2 = _get_stage2_launcher(
            stage2_cfg,
            stage2_cache_mod,
            "bf16",
            self.weights.has_bias,
            output_mode,
            stage2_stages,
            device_index,
        )
        reduce = (
            compile_moe_reduction(
                topk=cfg.top_k,
                model_dim=cfg.hidden_size,
                dtype_str="bf16",
            )
            if output_mode == "reduce"
            else None
        )
        return _TrainingForwardLaunchPlan(
            stage1=stage1,
            stage2=stage2,
            reduce=reduce,
            grid1=int(grid1),
            grid2=int(grid2),
            output_mode=output_mode,
            route_policy_size=policy_representative,
        )

    def _launch_e16_identity_stage1(
        self,
        hidden_states: torch.Tensor,
        workspace: SonicMoEWorkspace,
        route_preactivation: torch.Tensor,
        route_weights: torch.Tensor,
        expert_boundaries: torch.Tensor,
        frequency: torch.Tensor,
        frequency_mirror: torch.Tensor | None,
        sorted_token_ids: torch.Tensor,
        sorted_route_ids: torch.Tensor,
        sorted_weights: torch.Tensor,
        sorted_expert_ids: torch.Tensor,
        num_valid_ids: torch.Tensor,
        launch_plan: _TrainingForwardLaunchPlan,
        stream: torch.cuda.Stream,
    ) -> None:
        """Run the counts-scheduled Stage 1 which also emits route metadata."""

        if self.config.num_experts != 16:
            raise RuntimeError("fused identity Stage 1 requires exactly 16 experts")
        routes = int(workspace.routes or 0)
        mirror_arg = frequency if frequency_mirror is None else frequency_mirror
        assert workspace.sorting_workspace is not None
        _run_compiled(
            launch_plan.stage1,
            hidden_states.data_ptr(),
            self.weights.gate_up.data_ptr(),
            self.weights.dummy_scale.data_ptr(),
            (
                self.weights.dummy_scale
                if self.weights.stage1_bias is None
                else self.weights.stage1_bias
            ).data_ptr(),
            sorted_expert_ids.data_ptr(),
            num_valid_ids.data_ptr(),
            sorted_token_ids.data_ptr(),
            workspace.tokens,
            max(1, launch_plan.grid1),
            1.0,
            1.0,
            1.0,
            1.0,
            float("inf"),
            workspace.intermediate.data_ptr(),
            route_preactivation.data_ptr(),
            sorted_route_ids.data_ptr(),
            routes,
            expert_boundaries.data_ptr(),
            route_weights.data_ptr(),
            frequency.data_ptr(),
            mirror_arg.data_ptr(),
            workspace.sorting_workspace.data_ptr(),
            sorted_weights.data_ptr(),
            stream,
        )

    def _run_grouped_gemms_training(
        self,
        hidden_states: torch.Tensor,
        workspace: SonicMoEWorkspace,
        out: torch.Tensor,
        route_preactivation: torch.Tensor,
        *,
        interleaved_w1: bool,
        sorted_token_ids: torch.Tensor | None = None,
        sorted_route_ids: torch.Tensor | None = None,
        sorted_weights: torch.Tensor | None = None,
        sorted_expert_ids: torch.Tensor | None = None,
        num_valid_ids: torch.Tensor | None = None,
        token_indices_identity: bool = False,
        launch_plan: _TrainingForwardLaunchPlan | None = None,
        stage1_enqueued: bool = False,
    ) -> torch.Tensor:
        """Training-only grouped forward with the Stage-1 dual output."""

        cfg = self.config
        tokens = workspace.tokens
        stream = torch.cuda.current_stream(hidden_states.device)
        flat_routes = workspace.routes is not None
        if launch_plan is None:
            launch_plan = self._prepare_grouped_gemms_training(
                workspace,
                interleaved_w1=interleaved_w1,
                token_indices_identity=token_indices_identity,
            )
        output_mode = launch_plan.output_mode
        sorted_token_ids = (
            workspace.sorted_token_ids if sorted_token_ids is None else sorted_token_ids
        )
        sorted_route_ids = (
            workspace.sorted_route_ids if sorted_route_ids is None else sorted_route_ids
        )
        sorted_weights = (
            workspace.sorted_weights if sorted_weights is None else sorted_weights
        )
        sorted_expert_ids = (
            workspace.sorted_expert_ids
            if sorted_expert_ids is None
            else sorted_expert_ids
        )
        num_valid_ids = (
            workspace.num_valid_ids if num_valid_ids is None else num_valid_ids
        )
        if flat_routes and sorted_route_ids is None:
            raise RuntimeError("flat-route training requires sorted route ids")

        if not stage1_enqueued:
            _run_compiled(
                launch_plan.stage1,
                hidden_states.data_ptr(),
                self.weights.gate_up.data_ptr(),
                self.weights.dummy_scale.data_ptr(),
                (
                    self.weights.dummy_scale
                    if self.weights.stage1_bias is None
                    else self.weights.stage1_bias
                ).data_ptr(),
                sorted_expert_ids.data_ptr(),
                num_valid_ids.data_ptr(),
                sorted_token_ids.data_ptr(),
                tokens,
                launch_plan.grid1,
                1.0,
                1.0,
                1.0,
                1.0,
                float("inf"),
                workspace.intermediate.data_ptr(),
                route_preactivation.data_ptr(),
                (
                    self.weights.dummy_scale
                    if sorted_route_ids is None
                    else sorted_route_ids
                ).data_ptr(),
                (
                    tokens * cfg.top_k
                    if workspace.routes is None
                    else int(workspace.routes)
                ),
                stream,
            )

        _run_compiled(
            launch_plan.stage2,
            workspace.intermediate.data_ptr(),
            self.weights.down.data_ptr(),
            self.weights.dummy_scale.data_ptr(),
            (
                self.weights.dummy_scale
                if self.weights.stage2_bias is None
                else self.weights.stage2_bias
            ).data_ptr(),
            sorted_expert_ids.data_ptr(),
            num_valid_ids.data_ptr(),
            sorted_token_ids.data_ptr(),
            sorted_weights.data_ptr(),
            tokens,
            workspace.stage2_max_m_blocks,
            launch_plan.grid2,
            (workspace.route_output if output_mode == "reduce" else out).data_ptr(),
            stream,
        )
        if output_mode == "reduce":
            assert workspace.route_output is not None
            reduce = launch_plan.reduce
            assert reduce is not None
            route_output_ptr = flyc.from_c_void_p(
                fx.Uint8, workspace.route_output.data_ptr()
            )
            out_ptr = flyc.from_c_void_p(fx.Uint8, out.data_ptr())
            unused_ptr = flyc.from_c_void_p(
                fx.Uint8, self.weights.dummy_scale.data_ptr()
            )
            _run_compiled(
                reduce,
                route_output_ptr,
                out_ptr,
                unused_ptr,
                unused_ptr,
                tokens,
                stream,
            )
        return out

    def __call__(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        out: torch.Tensor | None = None,
        *,
        expert_frequency_out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run router + grouped expert MLP from router logits.

        ``router_logits`` is ``[tokens, num_experts]`` in FP32/FP16/BF16.
        When supplied, ``expert_frequency_out`` must be caller-owned contiguous
        int32 storage with shape ``[num_experts]``.  It receives the exact
        selected-route count without changing the returned output object.
        """

        if not hidden_states.is_cuda:
            raise ValueError("hidden_states must be on a ROCm device")
        with torch.cuda.device(hidden_states.device):
            return self._forward_from_logits(
                hidden_states,
                router_logits,
                out,
                expert_frequency_out,
            )

    def _launch_prevalidated_logits(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        workspace: SonicMoEWorkspace,
        output: torch.Tensor,
        expert_frequency_out: torch.Tensor | None,
        dtype_str: str,
    ) -> torch.Tensor:
        """Enqueue a validated native-router call while preserving workspace serialization."""

        with workspace._launch_lock:
            moe_softmax_sort_flydsl(
                router_logits,
                workspace.sorted_token_ids,
                workspace.sorted_weights,
                workspace.sorted_expert_ids,
                workspace.num_valid_ids,
                output,
                self.config.num_experts,
                self.config.top_k,
                dtype_str,
                unit_size=self.config.route_tile_m,
                renormalize=self.config.renormalize,
                workspace=workspace.sorting_workspace,
                topk_scratch=(
                    workspace.router_topk_weights,
                    workspace.router_topk_ids,
                    workspace.router_topk_expert_indices,
                ),
                direct_single_token=True,
                expert_frequency_out=expert_frequency_out,
            )
            result = self._run_grouped_gemms(hidden_states, workspace, output)
        if expert_frequency_out is not None:
            expert_frequency_out.record_stream(
                torch.cuda.current_stream(hidden_states.device)
            )
        return result

    def _forward_from_logits_prevalidated(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        out: torch.Tensor,
        expert_frequency_out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Internal trusted fast path for the SonicMoE compatibility adapter.

        The caller must already have established the full public ``__call__``
        contract: tensors are correctly shaped, typed, contiguous, aligned,
        non-aliasing, gradient-free, on this operator's gfx95 device, and the
        config supports the native FlyDSL router.  ``out`` is mandatory and
        remains caller-owned.  This method deliberately skips those duplicate
        Python checks, but retains current-stream workspace selection, launch
        serialization, and frequency lifetime tracking.
        """

        tokens = int(hidden_states.shape[0])
        workspace = self.reserve(tokens)
        return self._launch_prevalidated_logits(
            hidden_states,
            router_logits,
            workspace,
            out,
            expert_frequency_out,
            _SUPPORTED_ROUTER_DTYPES[router_logits.dtype],
        )

    def _forward_from_logits(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        out: torch.Tensor | None,
        expert_frequency_out: torch.Tensor | None,
    ) -> torch.Tensor:
        tokens = self._validate_hidden(hidden_states)
        if not router_logits.is_cuda or router_logits.device != hidden_states.device:
            raise ValueError(
                "router_logits must be on the same ROCm device as hidden_states"
            )
        if tuple(router_logits.shape) != (tokens, self.config.num_experts):
            raise ValueError(
                f"router_logits must have shape {(tokens, self.config.num_experts)}, "
                f"got {tuple(router_logits.shape)}"
            )
        dtype_str = _SUPPORTED_ROUTER_DTYPES.get(router_logits.dtype)
        if dtype_str is None:
            raise TypeError(
                f"router_logits must be FP32/FP16/BF16, got {router_logits.dtype}"
            )
        if not router_logits.is_contiguous():
            raise ValueError("router_logits must be contiguous")
        if router_logits.requires_grad:
            raise ValueError(
                "SonicMoE is inference-only; router_logits must not require gradients"
            )
        if (
            self.config.supports_flydsl_router
            and router_logits.numel() * router_logits.element_size()
            > _MAX_BUFFER_BYTE_OFFSET
        ):
            raise ValueError(
                "router_logits exceed the FlyDSL router's 32-bit byte-offset limit"
            )

        # Exact layouts map an expert row to power-of-two lane groups; VPT may
        # be non-power-of-two (for example E=896 uses VPT=14, TPT=64).  Retain
        # the PyTorch router only for counts without such a layout; expert sort
        # and both grouped GEMMs remain FlyDSL in that fallback.
        if not self.config.supports_flydsl_router:
            output = out
            frequency = None
            if expert_frequency_out is not None:
                workspace = self.reserve(tokens)
                output = self._validate_out(
                    out,
                    workspace,
                    hidden_states,
                    router_logits,
                    *self.weights.tensors,
                )
                frequency = self._validate_expert_frequency_out(
                    expert_frequency_out,
                    workspace,
                    output,
                    hidden_states,
                    router_logits,
                    *self.weights.tensors,
                )
            probs = torch.softmax(router_logits.float(), dim=-1)
            topk_weights, topk_ids = torch.topk(probs, self.config.top_k, dim=-1)
            if self.config.renormalize:
                topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
            topk_ids = topk_ids.to(torch.int32)
            if frequency is None:
                return self.forward_topk(
                    hidden_states,
                    topk_ids,
                    topk_weights.contiguous(),
                    out=out,
                )

            result = self.forward_topk(
                hidden_states,
                topk_ids,
                topk_weights.contiguous(),
                out=output,
            )
            topk_frequency_flydsl(topk_ids, frequency, self.config.num_experts)
            frequency.record_stream(torch.cuda.current_stream(hidden_states.device))
            return result

        workspace = self.reserve(tokens)
        output = self._validate_out(
            out,
            workspace,
            hidden_states,
            router_logits,
            *self.weights.tensors,
        )
        frequency = None
        if expert_frequency_out is not None:
            frequency = self._validate_expert_frequency_out(
                expert_frequency_out,
                workspace,
                output,
                hidden_states,
                router_logits,
                *self.weights.tensors,
            )
        return self._launch_prevalidated_logits(
            hidden_states,
            router_logits,
            workspace,
            output,
            frequency,
            dtype_str,
        )

    def forward_routes(
        self,
        hidden_states: torch.Tensor,
        token_indices: torch.Tensor,
        expert_indices: torch.Tensor,
        route_weights: torch.Tensor,
        out: torch.Tensor | None = None,
        expert_frequency_out: torch.Tensor | None = None,
        *,
        expert_offsets: torch.Tensor | None = None,
        token_indices_identity: bool = False,
        route_policy_size: int | None = None,
    ) -> torch.Tensor:
        """Run the grouped MLP from a flat variable-K route list.

        ``token_indices`` and ``expert_indices`` must be contiguous int32 and
        ``route_weights`` contiguous float32, all with shape ``[routes]``.
        Every edge is consumed independently, including duplicate
        ``(token, expert)`` pairs; weights are used verbatim.  IDs must be in
        range.  This hot path intentionally leaves value validation to the
        caller to avoid a device synchronization.

        When supplied, ``expert_frequency_out`` must be contiguous int32 with
        shape ``[num_experts]`` and receives the number of route occurrences
        for each expert.

        ``expert_offsets`` selects the pre-grouped fast path.  It must be a
        device int32 ``[num_experts + 1]`` cumulative-length tensor describing
        expert-major input segments.  Its values and agreement with
        ``expert_indices`` are an unchecked hot-path precondition.  Set
        ``token_indices_identity=True`` only when ``routes == tokens`` and
        ``token_indices[r] == r``; Stage 2 can then store directly without
        clearing or atomically accumulating the output.

        ``route_policy_size`` is an optional rank-shared performance-policy
        floor for expert-major tile/cache selection.  It is coarsened to a
        finite capacity class, automatically promoted by a larger local route
        count, and never changes allocation, runtime grids, or route bounds.
        """

        if not hidden_states.is_cuda:
            raise ValueError("hidden_states must be on a ROCm device")
        with torch.cuda.device(hidden_states.device):
            return self._forward_routes_on_current_device(
                hidden_states,
                token_indices,
                expert_indices,
                route_weights,
                out,
                expert_frequency_out,
                expert_offsets=expert_offsets,
                expert_counts=None,
                token_indices_identity=token_indices_identity,
                route_policy_size=route_policy_size,
            )

    def forward_expert_major(
        self,
        hidden_states: torch.Tensor,
        expert_offsets: torch.Tensor,
        route_weights: torch.Tensor,
        out: torch.Tensor | None = None,
        expert_frequency_out: torch.Tensor | None = None,
        *,
        route_policy_size: int | None = None,
    ) -> torch.Tensor:
        """Run E16 expert-major identity routes without materialized IDs.

        Rows are already grouped by expert according to ``expert_offsets``;
        route ``r`` is token ``r``.  The E16 Stage-1 kernel derives both token
        and expert IDs from that contract, so allocating
        ``arange(routes)`` and ``repeat_interleave(experts, counts)`` is
        unnecessary.  This entry point deliberately fails closed if that
        specialization is disabled instead of passing synthetic IDs to a
        sorter that could read them.
        """

        if not hidden_states.is_cuda:
            raise ValueError("hidden_states must be on a ROCm device")
        with torch.cuda.device(hidden_states.device):
            return self._forward_routes_on_current_device(
                hidden_states,
                None,
                None,
                route_weights,
                out,
                expert_frequency_out,
                expert_offsets=expert_offsets,
                expert_counts=None,
                token_indices_identity=True,
                route_policy_size=route_policy_size,
            )

    def forward_expert_major_counts(
        self,
        hidden_states: torch.Tensor,
        expert_counts: torch.Tensor,
        route_weights: torch.Tensor,
        out: torch.Tensor | None = None,
        expert_frequency_out: torch.Tensor | None = None,
        *,
        route_policy_size: int | None = None,
    ) -> torch.Tensor:
        """Run E16 expert-major identity routes directly from expert counts.

        ``expert_counts`` is device int32 ``[16]``. Its values must be
        non-negative and sum to the number of rows. The E16 Stage-1 kernel
        computes both raw and padded prefixes while performing the first GEMM,
        eliminating the caller-side ``cumsum`` as well as materialized
        token/expert ids. The value contract is intentionally unchecked to
        avoid a D2H synchronization on the dynamic-route hot path.
        """

        if not hidden_states.is_cuda:
            raise ValueError("hidden_states must be on a ROCm device")
        with torch.cuda.device(hidden_states.device):
            return self._forward_routes_on_current_device(
                hidden_states,
                None,
                None,
                route_weights,
                out,
                expert_frequency_out,
                expert_offsets=None,
                expert_counts=expert_counts,
                token_indices_identity=True,
                route_policy_size=route_policy_size,
            )

    def _forward_routes_on_current_device(
        self,
        hidden_states: torch.Tensor,
        token_indices: torch.Tensor | None,
        expert_indices: torch.Tensor | None,
        route_weights: torch.Tensor,
        out: torch.Tensor | None,
        expert_frequency_out: torch.Tensor | None,
        *,
        expert_offsets: torch.Tensor | None,
        expert_counts: torch.Tensor | None,
        token_indices_identity: bool,
        route_policy_size: int | None,
    ) -> torch.Tensor:
        tokens = self._validate_hidden(hidden_states, allow_empty=True)
        implicit_identity_ids = token_indices is None and expert_indices is None
        if (token_indices is None) != (expert_indices is None):
            raise ValueError(
                "token_indices and expert_indices must either both be tensors "
                "or both be None"
            )
        if route_weights.ndim != 1:
            raise ValueError("route_weights must be one-dimensional")
        routes = int(route_weights.numel())
        if not implicit_identity_ids:
            assert token_indices is not None
            assert expert_indices is not None
            if token_indices.ndim != 1 or expert_indices.ndim != 1:
                raise ValueError(
                    "token_indices and expert_indices must be one-dimensional"
                )
            if (
                int(token_indices.numel()) != routes
                or int(expert_indices.numel()) != routes
            ):
                raise ValueError(
                    "token_indices, expert_indices, and route_weights must have equal length"
                )
        if routes > _MAX_SIGNED_I32:
            raise ValueError(
                f"route count exceeds the sorting kernel's signed 32-bit limit: {routes}"
            )
        policy_representative = e16_route_policy_representative(
            select_e16_route_policy(routes, route_policy_size)
        )
        if not route_weights.is_cuda or route_weights.device != hidden_states.device:
            raise ValueError(
                "route tensors must be on the same ROCm device as hidden_states"
            )
        if not implicit_identity_ids:
            assert token_indices is not None
            assert expert_indices is not None
            if (
                not token_indices.is_cuda
                or not expert_indices.is_cuda
                or token_indices.device != hidden_states.device
                or expert_indices.device != hidden_states.device
            ):
                raise ValueError(
                    "route tensors must be on the same ROCm device as hidden_states"
                )
            if (
                token_indices.dtype != torch.int32
                or expert_indices.dtype != torch.int32
            ):
                raise TypeError(
                    "token_indices/expert_indices must be int32, got "
                    f"{token_indices.dtype}/{expert_indices.dtype}"
                )
        if route_weights.dtype != torch.float32:
            raise TypeError(f"route_weights must be float32, got {route_weights.dtype}")
        if not route_weights.is_contiguous() or (
            not implicit_identity_ids
            and (
                not token_indices.is_contiguous() or not expert_indices.is_contiguous()
            )
        ):
            raise ValueError("route tensors must be contiguous")
        if route_weights.requires_grad:
            raise ValueError(
                "SonicMoE is inference-only; route_weights must not require gradients"
            )

        if expert_offsets is not None and expert_counts is not None:
            raise ValueError("expert_offsets and expert_counts are mutually exclusive")
        counts_major = _validate_expert_major_counts(
            expert_counts,
            hidden_states=hidden_states,
            routes=routes,
            num_experts=self.config.num_experts,
        )
        expert_major = counts_major or _validate_expert_major_routes(
            expert_offsets,
            hidden_states=hidden_states,
            routes=routes,
            num_experts=self.config.num_experts,
            token_indices_identity=token_indices_identity,
        )
        if implicit_identity_ids:
            if not expert_major or not _can_fuse_e16_identity_metadata_stage1(
                self.config,
                token_indices_identity=token_indices_identity,
            ):
                raise NotImplementedError(
                    "implicit expert-major ids require the E16 fused identity "
                    "Stage-1 specialization"
                )

        workspace = self.reserve_dynamic_routes(tokens, routes)
        route_inputs = (
            (route_weights,)
            if implicit_identity_ids
            else (token_indices, expert_indices, route_weights)
        )
        output = self._validate_out(
            out,
            workspace,
            hidden_states,
            *route_inputs,
            *((expert_offsets,) if expert_offsets is not None else ()),
            *((expert_counts,) if expert_counts is not None else ()),
            *self.weights.tensors,
            invocation_owned_default=True,
        )

        frequency = workspace.expert_frequency
        if expert_frequency_out is not None:
            frequency = self._validate_expert_frequency_out(
                expert_frequency_out,
                workspace,
                output,
                hidden_states,
                *route_inputs,
                *((expert_offsets,) if expert_offsets is not None else ()),
                *((expert_counts,) if expert_counts is not None else ()),
                *self.weights.tensors,
            )

        assert workspace.sorting_workspace is not None
        with workspace._launch_lock:
            fuse_identity_stage1 = (
                expert_major
                and _can_fuse_e16_identity_metadata_stage1(
                    self.config,
                    token_indices_identity=token_indices_identity,
                )
            )
            if fuse_identity_stage1:
                expert_boundaries = expert_counts if counts_major else expert_offsets
                assert expert_boundaries is not None
                return self._run_grouped_gemms(
                    hidden_states,
                    workspace,
                    output,
                    token_indices_identity=True,
                    route_policy_size=policy_representative,
                    identity_boundaries=expert_boundaries,
                    identity_route_weights=route_weights,
                    identity_frequency=frequency,
                    identity_expert_counts_input=counts_major,
                )
            if expert_major:
                expert_boundaries = expert_counts if counts_major else expert_offsets
                assert expert_boundaries is not None
                moe_expert_major_sorting_flydsl(
                    token_indices,
                    expert_indices,
                    route_weights,
                    expert_boundaries,
                    frequency,
                    workspace.sorting_workspace,
                    workspace.sorted_token_ids,
                    workspace.sorted_weights,
                    workspace.sorted_expert_ids,
                    workspace.num_valid_ids,
                    output,
                    self.config.num_experts,
                    tokens=tokens,
                    max_padded_routes=workspace.max_padded_tokens,
                    unit_size=self.config.route_tile_m,
                    token_indices_identity=token_indices_identity,
                    clear_output=not token_indices_identity,
                    route_policy_size=policy_representative,
                    expert_counts_input=counts_major,
                )
            else:
                moe_ragged_sorting_flydsl(
                    token_indices,
                    expert_indices,
                    route_weights,
                    frequency,
                    workspace.sorting_workspace,
                    workspace.sorted_token_ids,
                    workspace.sorted_weights,
                    workspace.sorted_expert_ids,
                    workspace.num_valid_ids,
                    output,
                    self.config.num_experts,
                    tokens=tokens,
                    max_padded_routes=workspace.max_padded_tokens,
                    unit_size=self.config.route_tile_m,
                )
            if routes == 0:
                return output
            return self._run_grouped_gemms(
                hidden_states,
                workspace,
                output,
                token_indices_identity=token_indices_identity,
                route_policy_size=(policy_representative if expert_major else None),
            )

    def forward_routes_training(
        self,
        hidden_states: torch.Tensor,
        token_indices: torch.Tensor,
        expert_indices: torch.Tensor,
        route_weights: torch.Tensor,
        out: torch.Tensor | None = None,
        *,
        interleaved_w1: bool = False,
        expert_frequency_out: torch.Tensor | None = None,
        expert_offsets: torch.Tensor | None = None,
        token_indices_identity: bool = False,
        route_policy_size: int | None = None,
    ) -> tuple[torch.Tensor, SonicMoERoutesForwardState]:
        """Run arbitrary flat routes and retain route-order W1 preactivation.

        The saved BF16 tensor has shape ``[routes, 2 * intermediate_size]``.
        Its row ``r`` corresponds exactly to input route ``r`` even though the
        grouped GEMM executes in expert-sorted order.  Duplicate
        ``(token, expert)`` edges and tokens with variable route counts remain
        independent.

        Like :meth:`forward_topk_training`, this low-level raw-pointer API is
        intended for an autograd adapter and does not itself attach a
        ``grad_fn``.  It currently supports dense BF16 SwiGLU weights.

        Passing ``expert_offsets`` selects the same expert-major contract as
        :meth:`forward_routes` and avoids the generic counting sort.  The
        generated padded metadata is owned by the returned state and reused by
        backward.  ``token_indices_identity=True`` additionally selects direct
        Stage-2 output stores and records the identity mapping in that state.
        ``route_policy_size`` may carry one rank-shared tile/cache policy floor; it
        selects only a bounded performance profile and is automatically
        promoted by larger local route counts.  Actual ``routes`` remains
        authoritative for allocation, bounds, and retained state.
        """

        if not isinstance(interleaved_w1, bool):
            raise TypeError("interleaved_w1 must be bool")
        if self.weights.weight_dtype != "bf16" or self.config.compute_dtype != "bf16":
            raise NotImplementedError(
                "forward_routes_training currently supports only dense BF16 weights and compute"
            )
        if self.config.activation != "swiglu":
            raise NotImplementedError(
                "forward_routes_training currently supports only activation='swiglu'"
            )
        if not hidden_states.is_cuda:
            raise ValueError("hidden_states must be on a ROCm device")
        with torch.cuda.device(hidden_states.device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "forward_routes_training does not support graph capture without a "
                    "graph-private preallocated state slot; capture support is not enabled"
                )
            return self._forward_routes_training_on_current_device(
                hidden_states,
                token_indices,
                expert_indices,
                route_weights,
                out,
                interleaved_w1=interleaved_w1,
                expert_frequency_out=expert_frequency_out,
                expert_offsets=expert_offsets,
                expert_counts=None,
                token_indices_identity=token_indices_identity,
                route_policy_size=route_policy_size,
            )

    def forward_expert_major_training(
        self,
        hidden_states: torch.Tensor,
        expert_offsets: torch.Tensor,
        route_weights: torch.Tensor,
        out: torch.Tensor | None = None,
        *,
        interleaved_w1: bool = False,
        expert_frequency_out: torch.Tensor | None = None,
        route_policy_size: int | None = None,
    ) -> tuple[torch.Tensor, SonicMoERoutesForwardState]:
        """Train E16 expert-major identity routes without materialized IDs.

        This is the training counterpart of :meth:`forward_expert_major`.
        The returned state owns all padded route metadata needed by
        :func:`sonic_moe_backward_expert_major`, so an autograd adapter need
        not construct or save flat token/expert ID tensors.  The lifecycle is
        deliberately limited to the audited bias-free BF16 SwiGLU
        H2048/I768/E16/top-k-1 shape; unsupported configurations fail before
        allocating invocation state.
        """

        if not isinstance(interleaved_w1, bool):
            raise TypeError("interleaved_w1 must be bool")
        if self.weights.weight_dtype != "bf16" or self.config.compute_dtype != "bf16":
            raise NotImplementedError(
                "forward_expert_major_training currently supports only dense "
                "BF16 weights and compute"
            )
        if self.config.activation != "swiglu":
            raise NotImplementedError(
                "forward_expert_major_training currently supports only "
                "activation='swiglu'"
            )
        if not hidden_states.is_cuda:
            raise ValueError("hidden_states must be on a ROCm device")
        with torch.cuda.device(hidden_states.device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "forward_expert_major_training does not support graph capture "
                    "without a graph-private preallocated state slot"
                )
            return self._forward_routes_training_on_current_device(
                hidden_states,
                None,
                None,
                route_weights,
                out,
                interleaved_w1=interleaved_w1,
                expert_frequency_out=expert_frequency_out,
                expert_offsets=expert_offsets,
                expert_counts=None,
                token_indices_identity=True,
                route_policy_size=route_policy_size,
            )

    def forward_expert_major_counts_training(
        self,
        hidden_states: torch.Tensor,
        expert_counts: torch.Tensor,
        route_weights: torch.Tensor,
        out: torch.Tensor | None = None,
        *,
        interleaved_w1: bool = False,
        expert_frequency_out: torch.Tensor | None = None,
        route_policy_size: int | None = None,
    ) -> tuple[torch.Tensor, SonicMoERoutesForwardState]:
        """Train E16 expert-major identity routes directly from counts.

        Counts follow :meth:`forward_expert_major_counts`. The retained state
        is consumed by :func:`sonic_moe_backward_expert_major`; neither side
        materializes a length-``routes`` token or expert-id tensor. This has
        the same audited retained-backward shape contract as
        :meth:`forward_expert_major_training`.
        """

        if not isinstance(interleaved_w1, bool):
            raise TypeError("interleaved_w1 must be bool")
        if self.weights.weight_dtype != "bf16" or self.config.compute_dtype != "bf16":
            raise NotImplementedError(
                "forward_expert_major_counts_training currently supports only "
                "dense BF16 weights and compute"
            )
        if self.config.activation != "swiglu":
            raise NotImplementedError(
                "forward_expert_major_counts_training currently supports only "
                "activation='swiglu'"
            )
        if not hidden_states.is_cuda:
            raise ValueError("hidden_states must be on a ROCm device")
        with torch.cuda.device(hidden_states.device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "forward_expert_major_counts_training does not support graph "
                    "capture without a graph-private preallocated state slot"
                )
            return self._forward_routes_training_on_current_device(
                hidden_states,
                None,
                None,
                route_weights,
                out,
                interleaved_w1=interleaved_w1,
                expert_frequency_out=expert_frequency_out,
                expert_offsets=None,
                expert_counts=expert_counts,
                token_indices_identity=True,
                route_policy_size=route_policy_size,
            )

    def _forward_routes_training_on_current_device(
        self,
        hidden_states: torch.Tensor,
        token_indices: torch.Tensor | None,
        expert_indices: torch.Tensor | None,
        route_weights: torch.Tensor,
        out: torch.Tensor | None,
        *,
        interleaved_w1: bool,
        expert_frequency_out: torch.Tensor | None,
        expert_offsets: torch.Tensor | None,
        expert_counts: torch.Tensor | None,
        token_indices_identity: bool,
        route_policy_size: int | None,
    ) -> tuple[torch.Tensor, SonicMoERoutesForwardState]:
        tokens = self._validate_training_hidden(
            hidden_states,
            allow_empty=True,
        )
        implicit_identity_ids = token_indices is None and expert_indices is None
        if (token_indices is None) != (expert_indices is None):
            raise ValueError(
                "token_indices and expert_indices must either both be tensors "
                "or both be None"
            )
        if route_weights.ndim != 1:
            raise ValueError("route_weights must be one-dimensional")
        routes = int(route_weights.numel())
        if not implicit_identity_ids:
            assert token_indices is not None
            assert expert_indices is not None
            if token_indices.ndim != 1 or expert_indices.ndim != 1:
                raise ValueError(
                    "token_indices and expert_indices must be one-dimensional"
                )
            if (
                int(token_indices.numel()) != routes
                or int(expert_indices.numel()) != routes
            ):
                raise ValueError(
                    "token_indices, expert_indices, and route_weights must have equal length"
                )
        if routes > _MAX_SIGNED_I32:
            raise ValueError(
                f"route count exceeds the sorting kernel's signed 32-bit limit: {routes}"
            )
        policy_representative = e16_route_policy_representative(
            select_e16_route_policy(routes, route_policy_size)
        )
        if not route_weights.is_cuda or route_weights.device != hidden_states.device:
            raise ValueError(
                "route tensors must be on the same ROCm device as hidden_states"
            )
        if not implicit_identity_ids:
            assert token_indices is not None
            assert expert_indices is not None
            if (
                not token_indices.is_cuda
                or not expert_indices.is_cuda
                or token_indices.device != hidden_states.device
                or expert_indices.device != hidden_states.device
            ):
                raise ValueError(
                    "route tensors must be on the same ROCm device as hidden_states"
                )
            if (
                token_indices.dtype != torch.int32
                or expert_indices.dtype != torch.int32
            ):
                raise TypeError(
                    "token_indices/expert_indices must be int32, got "
                    f"{token_indices.dtype}/{expert_indices.dtype}"
                )
        if route_weights.dtype != torch.float32:
            raise TypeError(f"route_weights must be float32, got {route_weights.dtype}")
        if not route_weights.is_contiguous() or (
            not implicit_identity_ids
            and (
                not token_indices.is_contiguous() or not expert_indices.is_contiguous()
            )
        ):
            raise ValueError("route tensors must be contiguous")

        if expert_offsets is not None and expert_counts is not None:
            raise ValueError("expert_offsets and expert_counts are mutually exclusive")
        counts_major = _validate_expert_major_counts(
            expert_counts,
            hidden_states=hidden_states,
            routes=routes,
            num_experts=self.config.num_experts,
        )
        expert_major = counts_major or _validate_expert_major_routes(
            expert_offsets,
            hidden_states=hidden_states,
            routes=routes,
            num_experts=self.config.num_experts,
            token_indices_identity=token_indices_identity,
        )
        if implicit_identity_ids:
            if not expert_major or not _can_fuse_e16_identity_metadata_stage1(
                self.config,
                token_indices_identity=token_indices_identity,
            ):
                raise NotImplementedError(
                    "implicit expert-major ids require the E16 fused identity "
                    "Stage-1 specialization"
                )

        if implicit_identity_ids and not _retain_e16_flat_route_metadata(
            self.config,
            tokens,
            routes,
            has_bias=self.weights.has_bias,
        ):
            raise NotImplementedError(
                "implicit expert-major training requires the complete "
                "bias-free BF16 SwiGLU H2048/I768/E16/top_k=1 retained-state "
                "lifecycle with routes == tokens"
            )

        _validate_routes_training_preactivation_extent(
            routes,
            self.config.intermediate_size,
        )
        workspace = self.reserve_dynamic_routes(tokens, routes)
        route_inputs = (
            (route_weights,)
            if implicit_identity_ids
            else (token_indices, expert_indices, route_weights)
        )
        output = self._validate_out(
            out,
            workspace,
            hidden_states,
            *route_inputs,
            *((expert_offsets,) if expert_offsets is not None else ()),
            *((expert_counts,) if expert_counts is not None else ()),
            *self.weights.tensors,
            invocation_owned_default=True,
        )
        retain_route_metadata = _retain_e16_flat_route_metadata(
            self.config,
            tokens,
            routes,
            has_bias=self.weights.has_bias,
        )
        frequency = (
            torch.empty_like(workspace.expert_frequency)
            if retain_route_metadata
            else workspace.expert_frequency
        )
        frequency_mirror = None
        if expert_frequency_out is not None:
            validated_frequency_out = self._validate_expert_frequency_out(
                expert_frequency_out,
                workspace,
                output,
                hidden_states,
                *route_inputs,
                *((expert_offsets,) if expert_offsets is not None else ()),
                *((expert_counts,) if expert_counts is not None else ()),
                *self.weights.tensors,
            )
            if retain_route_metadata:
                frequency_mirror = validated_frequency_out
            else:
                frequency = validated_frequency_out
        if retain_route_metadata:
            sorted_token_ids = torch.empty_like(workspace.sorted_token_ids)
            sorted_route_ids = torch.empty_like(workspace.sorted_route_ids)
            sorted_weights = torch.empty_like(workspace.sorted_weights)
            sorted_expert_ids = torch.empty_like(workspace.sorted_expert_ids)
            num_valid_ids = torch.empty_like(workspace.num_valid_ids)
        else:
            sorted_token_ids = workspace.sorted_token_ids
            sorted_route_ids = workspace.sorted_route_ids
            sorted_weights = workspace.sorted_weights
            sorted_expert_ids = workspace.sorted_expert_ids
            num_valid_ids = workspace.num_valid_ids
        preactivation = torch.empty(
            (routes, 2 * self.config.intermediate_size),
            dtype=torch.bfloat16,
            device=hidden_states.device,
        )
        if preactivation.untyped_storage().data_ptr() in workspace.storage_ptrs:
            raise RuntimeError(
                "training preactivation unexpectedly aliases reusable workspace storage"
            )

        fuse_identity_stage1 = expert_major and _can_fuse_e16_identity_metadata_stage1(
            self.config,
            token_indices_identity=token_indices_identity,
        )
        launch_plan = self._prepare_grouped_gemms_training(
            workspace,
            interleaved_w1=interleaved_w1,
            token_indices_identity=token_indices_identity,
            route_policy_size=policy_representative,
            fused_identity_metadata=fuse_identity_stage1,
            identity_expert_counts_input=counts_major,
        )
        assert workspace.sorting_workspace is not None
        assert workspace.sorted_route_ids is not None
        stream = torch.cuda.current_stream(hidden_states.device)
        ready_event = torch.cuda.Event()
        with workspace._launch_lock:
            stage1_enqueued = False
            if expert_major:
                expert_boundaries = expert_counts if counts_major else expert_offsets
                assert expert_boundaries is not None
                if fuse_identity_stage1:
                    self._launch_e16_identity_stage1(
                        hidden_states,
                        workspace,
                        preactivation,
                        route_weights,
                        expert_boundaries,
                        frequency,
                        frequency_mirror,
                        sorted_token_ids,
                        sorted_route_ids,
                        sorted_weights,
                        sorted_expert_ids,
                        num_valid_ids,
                        launch_plan,
                        stream,
                    )
                    stage1_enqueued = True
                if not stage1_enqueued:
                    moe_expert_major_sorting_flydsl(
                        token_indices,
                        expert_indices,
                        route_weights,
                        expert_boundaries,
                        frequency,
                        workspace.sorting_workspace,
                        sorted_token_ids,
                        sorted_weights,
                        sorted_expert_ids,
                        num_valid_ids,
                        output,
                        self.config.num_experts,
                        tokens=tokens,
                        max_padded_routes=workspace.max_padded_tokens,
                        unit_size=self.config.route_tile_m,
                        sorted_route_ids=sorted_route_ids,
                        expert_frequency_mirror=frequency_mirror,
                        token_indices_identity=token_indices_identity,
                        clear_output=not token_indices_identity,
                        route_policy_size=launch_plan.route_policy_size,
                        expert_counts_input=counts_major,
                    )
            else:
                moe_ragged_sorting_flydsl(
                    token_indices,
                    expert_indices,
                    route_weights,
                    frequency,
                    workspace.sorting_workspace,
                    sorted_token_ids,
                    sorted_weights,
                    sorted_expert_ids,
                    num_valid_ids,
                    output,
                    self.config.num_experts,
                    tokens=tokens,
                    max_padded_routes=workspace.max_padded_tokens,
                    unit_size=self.config.route_tile_m,
                    sorted_route_ids=sorted_route_ids,
                    expert_frequency_mirror=frequency_mirror,
                )
            result = output
            if routes:
                result = self._run_grouped_gemms_training(
                    hidden_states,
                    workspace,
                    output,
                    preactivation,
                    interleaved_w1=interleaved_w1,
                    sorted_token_ids=sorted_token_ids,
                    sorted_route_ids=sorted_route_ids,
                    sorted_weights=sorted_weights,
                    sorted_expert_ids=sorted_expert_ids,
                    num_valid_ids=num_valid_ids,
                    token_indices_identity=token_indices_identity,
                    launch_plan=launch_plan,
                    stage1_enqueued=stage1_enqueued,
                )
            self._record_training_forward_stream(
                stream,
                hidden_states,
                token_indices,
                expert_indices,
                route_weights,
                result,
                preactivation,
                *self.weights.tensors,
                sorted_token_ids,
                sorted_route_ids,
                sorted_weights,
                sorted_expert_ids,
                num_valid_ids,
                workspace.sorting_workspace,
                workspace.intermediate,
                frequency,
                frequency_mirror,
                expert_offsets,
                expert_counts,
            )
            ready_event.record(stream)

        state = SonicMoERoutesForwardState(
            preactivation=preactivation,
            tokens=tokens,
            routes=routes,
            hidden_size=self.config.hidden_size,
            intermediate_size=self.config.intermediate_size,
            num_experts=self.config.num_experts,
            activation=self.config.activation,
            compute_dtype=self.config.compute_dtype,
            interleaved_w1=interleaved_w1,
            has_bias=self.weights.has_bias,
            producer_stream=int(stream.cuda_stream),
            ready_event=ready_event,
            # Empty launches keep one backing element so Stage 1 always
            # receives a valid buffer resource.  Expose only the
            # mathematical active extent in the retained-state ABI; this lets
            # backward validate and reuse an empty rank's metadata exactly as
            # it does for every non-empty dynamic route count.
            sorted_token_ids=(
                sorted_token_ids[: workspace.max_padded_tokens]
                if retain_route_metadata
                else None
            ),
            sorted_route_ids=(
                sorted_route_ids[: workspace.max_padded_tokens]
                if retain_route_metadata
                else None
            ),
            sorted_weights=(
                sorted_weights[: workspace.max_padded_tokens]
                if retain_route_metadata
                else None
            ),
            sorted_expert_ids=(
                sorted_expert_ids[: workspace.max_m_blocks]
                if retain_route_metadata
                else None
            ),
            num_valid_ids=(num_valid_ids if retain_route_metadata else None),
            expert_frequency=(frequency if retain_route_metadata else None),
            expert_major=expert_major,
            token_indices_identity=token_indices_identity,
            route_policy_size=launch_plan.route_policy_size,
        )
        return result, state

    @staticmethod
    def _record_training_forward_stream(
        stream: torch.cuda.Stream,
        *tensors: torch.Tensor | None,
    ) -> None:
        """Tie every raw-pointer tensor in a training launch to its stream."""

        recorded_storages: set[int] = set()
        for tensor in tensors:
            if tensor is None:
                continue
            storage = tensor.untyped_storage().data_ptr()
            if storage in recorded_storages:
                continue
            tensor.record_stream(stream)
            recorded_storages.add(storage)

    def forward_topk_training(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        out: torch.Tensor | None = None,
        *,
        interleaved_w1: bool = False,
        expert_frequency_out: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, SonicMoEForwardState]:
        """Run fixed-K forward and retain exact A16 W1 preactivation.

        This first training-state specialization supports dense BF16 SwiGLU.
        The saved tensor is allocated per invocation and is never backed by a
        reusable workspace.  ``interleaved_w1=True`` changes only the saved
        state's last-dimension layout to ``[g0, u0, ...]``; prepared forward
        weights retain their established preshuffled representation.

        As in :meth:`forward_topk`, each token's expert IDs must be distinct
        and in range.  The hot path does not synchronize to validate values.

        This is a low-level raw-pointer launch API, not a PyTorch autograd
        Function: it permits gradient-bearing inputs so an adapter can save
        them and provide the corresponding backward implementation.  Calling
        it directly does not attach a ``grad_fn`` to ``output`` or state.

        When supplied, ``expert_frequency_out`` follows the same caller-owned
        contiguous int32 ``[num_experts]`` contract as :meth:`__call__` and is
        populated from the fixed-K sorter without changing the return tuple.

        Forward-only graph capture is deliberately rejected in this phase.
        Safe capture requires an explicit graph-private preallocated state slot
        and a paired lifetime protocol, neither of which this API exposes yet.
        """

        if not isinstance(interleaved_w1, bool):
            raise TypeError("interleaved_w1 must be bool")
        if self.weights.weight_dtype != "bf16" or self.config.compute_dtype != "bf16":
            raise NotImplementedError(
                "forward_topk_training currently supports only dense BF16 weights and compute"
            )
        if self.config.activation != "swiglu":
            raise NotImplementedError(
                "forward_topk_training currently supports only activation='swiglu'"
            )
        if not hidden_states.is_cuda:
            raise ValueError("hidden_states must be on a ROCm device")
        with torch.cuda.device(hidden_states.device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "forward_topk_training does not support graph capture without a "
                    "graph-private preallocated state slot; capture support is not enabled"
                )
            return self._forward_topk_training_on_current_device(
                hidden_states,
                topk_ids,
                topk_weights,
                out,
                interleaved_w1=interleaved_w1,
                expert_frequency_out=expert_frequency_out,
            )

    def _forward_topk_training_on_current_device(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        out: torch.Tensor | None,
        *,
        interleaved_w1: bool,
        expert_frequency_out: torch.Tensor | None,
    ) -> tuple[torch.Tensor, SonicMoEForwardState]:
        tokens = self._validate_training_hidden(hidden_states)
        expected = (tokens, self.config.top_k)
        if tuple(topk_ids.shape) != expected or tuple(topk_weights.shape) != expected:
            raise ValueError(
                f"topk_ids/topk_weights must both have shape {expected}, got "
                f"{tuple(topk_ids.shape)}/{tuple(topk_weights.shape)}"
            )
        if (
            not topk_ids.is_cuda
            or not topk_weights.is_cuda
            or topk_ids.device != hidden_states.device
            or topk_weights.device != hidden_states.device
        ):
            raise ValueError(
                "topk ids/weights must be on the same ROCm device as hidden_states"
            )
        if topk_ids.dtype != torch.int32 or topk_weights.dtype != torch.float32:
            raise TypeError(
                f"topk_ids/topk_weights must be int32/float32, got "
                f"{topk_ids.dtype}/{topk_weights.dtype}"
            )
        if not topk_ids.is_contiguous() or not topk_weights.is_contiguous():
            raise ValueError("topk ids/weights must be contiguous")

        _validate_training_preactivation_extent(
            tokens,
            self.config.top_k,
            self.config.intermediate_size,
        )

        workspace = self.reserve(tokens)
        output = self._validate_out(
            out,
            workspace,
            hidden_states,
            topk_ids,
            topk_weights,
            *self.weights.tensors,
        )
        frequency = None
        if expert_frequency_out is not None:
            frequency = self._validate_expert_frequency_out(
                expert_frequency_out,
                workspace,
                output,
                hidden_states,
                topk_ids,
                topk_weights,
                *self.weights.tensors,
            )
        preactivation = torch.empty(
            (tokens, self.config.top_k, 2 * self.config.intermediate_size),
            dtype=torch.bfloat16,
            device=hidden_states.device,
        )
        if preactivation.untyped_storage().data_ptr() in workspace.storage_ptrs:
            raise RuntimeError(
                "training preactivation unexpectedly aliases reusable workspace storage"
            )

        launch_plan = self._prepare_grouped_gemms_training(
            workspace,
            interleaved_w1=interleaved_w1,
            token_indices_identity=False,
        )
        stream = torch.cuda.current_stream(hidden_states.device)
        ready_event = torch.cuda.Event()
        with workspace._launch_lock:
            moe_sorting_flydsl(
                topk_ids,
                topk_weights,
                workspace.sorted_token_ids,
                workspace.sorted_weights,
                workspace.sorted_expert_ids,
                workspace.num_valid_ids,
                output,
                self.config.num_experts,
                unit_size=self.config.route_tile_m,
                workspace=workspace.sorting_workspace,
                direct_single_token=True,
                expert_frequency_out=frequency,
            )
            result = self._run_grouped_gemms_training(
                hidden_states,
                workspace,
                output,
                preactivation,
                interleaved_w1=interleaved_w1,
                launch_plan=launch_plan,
            )
            self._record_training_forward_stream(
                stream,
                hidden_states,
                topk_ids,
                topk_weights,
                result,
                preactivation,
                *self.weights.tensors,
                workspace.sorted_token_ids,
                workspace.sorted_weights,
                workspace.sorted_expert_ids,
                workspace.num_valid_ids,
                workspace.sorting_workspace,
                workspace.intermediate,
                workspace.route_output,
                frequency,
            )
            ready_event.record(stream)

        state = SonicMoEForwardState(
            preactivation=preactivation,
            tokens=tokens,
            hidden_size=self.config.hidden_size,
            intermediate_size=self.config.intermediate_size,
            num_experts=self.config.num_experts,
            top_k=self.config.top_k,
            activation=self.config.activation,
            compute_dtype=self.config.compute_dtype,
            interleaved_w1=interleaved_w1,
            has_bias=self.weights.has_bias,
            producer_stream=int(stream.cuda_stream),
            ready_event=ready_event,
        )
        return result, state

    def forward_topk(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        out: torch.Tensor | None = None,
        *,
        expert_frequency_out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the grouped MLP from precomputed route ids and weights.

        ``topk_ids`` must be contiguous int32 and ``topk_weights`` contiguous
        float32, both with shape ``[tokens, top_k]``.  Every token's expert IDs
        must be distinct and in ``[0, num_experts)``; this hot path intentionally
        avoids a device synchronization to validate their values.  Weights are
        consumed as-is; normalize them before this call when desired.

        When supplied, ``expert_frequency_out`` follows the same caller-owned
        contiguous int32 ``[num_experts]`` contract as :meth:`__call__` and is
        populated directly by the fixed-K sorter.
        """

        if not hidden_states.is_cuda:
            raise ValueError("hidden_states must be on a ROCm device")
        with torch.cuda.device(hidden_states.device):
            return self._forward_topk_on_current_device(
                hidden_states,
                topk_ids,
                topk_weights,
                out,
                expert_frequency_out,
            )

    def _forward_topk_on_current_device(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        out: torch.Tensor | None,
        expert_frequency_out: torch.Tensor | None,
    ) -> torch.Tensor:
        tokens = self._validate_hidden(hidden_states)
        expected = (tokens, self.config.top_k)
        if tuple(topk_ids.shape) != expected or tuple(topk_weights.shape) != expected:
            raise ValueError(
                f"topk_ids/topk_weights must both have shape {expected}, got "
                f"{tuple(topk_ids.shape)}/{tuple(topk_weights.shape)}"
            )
        if (
            not topk_ids.is_cuda
            or not topk_weights.is_cuda
            or topk_ids.device != hidden_states.device
            or topk_weights.device != hidden_states.device
        ):
            raise ValueError(
                "topk ids/weights must be on the same ROCm device as hidden_states"
            )
        if topk_ids.dtype != torch.int32 or topk_weights.dtype != torch.float32:
            raise TypeError(
                f"topk_ids/topk_weights must be int32/float32, got "
                f"{topk_ids.dtype}/{topk_weights.dtype}"
            )
        if not topk_ids.is_contiguous() or not topk_weights.is_contiguous():
            raise ValueError("topk ids/weights must be contiguous")
        if topk_weights.requires_grad:
            raise ValueError(
                "SonicMoE is inference-only; topk_weights must not require gradients"
            )

        workspace = self.reserve(tokens)
        output = self._validate_out(
            out,
            workspace,
            hidden_states,
            topk_ids,
            topk_weights,
            *self.weights.tensors,
        )
        frequency = None
        if expert_frequency_out is not None:
            frequency = self._validate_expert_frequency_out(
                expert_frequency_out,
                workspace,
                output,
                hidden_states,
                topk_ids,
                topk_weights,
                *self.weights.tensors,
            )
        with workspace._launch_lock:
            moe_sorting_flydsl(
                topk_ids,
                topk_weights,
                workspace.sorted_token_ids,
                workspace.sorted_weights,
                workspace.sorted_expert_ids,
                workspace.num_valid_ids,
                output,
                self.config.num_experts,
                unit_size=self.config.route_tile_m,
                workspace=workspace.sorting_workspace,
                direct_single_token=True,
                expert_frequency_out=frequency,
            )
            result = self._run_grouped_gemms(hidden_states, workspace, output)
        if frequency is not None:
            frequency.record_stream(torch.cuda.current_stream(hidden_states.device))
        return result


@torch.no_grad()
def warmup_sonic_e16_training(
    operator: SonicMoE,
    w1: torch.Tensor,
    w2: torch.Tensor,
    *,
    route_policy_sizes: Iterable[int] | None = None,
    interleaved_w1: bool = False,
    synchronize: bool = True,
) -> tuple[int, ...]:
    """Compile the finite dynamic-E16 forward/backward policy families.

    The warmup executes a real retained-state expert-major forward and
    backward because FlyDSL device compilation happens on first launch, not
    when a cached launcher factory is created.  A fixed 16-route probe keeps
    allocations small; only ``route_policy_size`` changes, so this does not
    reintroduce exact-R compiler variants.  The returned tuple contains the
    canonical policy sizes that were warmed.

    This helper is intentionally restricted to the retained expert-major
    identity lifecycle of the tuned bias-free BF16 Qwen3-30B-A3B local-expert
    shape. ``w1`` and ``w2`` are the logical contiguous BF16 backward weights,
    not the operator's preshuffled forward storage; an interleaved W1 must be
    declared consistently.  Each process/device should call it once, then the
    integration should synchronize its EP ranks before timed work.  With
    ``synchronize=False`` it only guarantees compilation and enqueue on the
    current stream, so the caller owns the corresponding synchronization.

    Warmup intentionally uses a tiny probe and does not reserve production-size
    scratch.  An application which knows a high-water route count can call
    ``operator.reserve_dynamic_routes(max_routes, max_routes)`` during startup
    (preferably through a shared dynamic workspace pool).  At runtime, derive
    one policy size from the current EP-wide maximum local route count with
    :func:`canonical_e16_route_policy_size` and pass that same value to every
    rank's forward; backward inherits it from the retained state.  The shared
    maximum must cover every local rank—automatic promotion of an underestimated
    hint preserves correctness, but cannot keep rank launch policies aligned.
    """

    if not isinstance(operator, SonicMoE):
        raise TypeError("operator must be a SonicMoE instance")
    if not isinstance(interleaved_w1, bool):
        raise TypeError("interleaved_w1 must be bool")
    if not isinstance(synchronize, bool):
        raise TypeError("synchronize must be bool")
    config = operator.config
    if (
        operator.weights.weight_dtype != "bf16"
        or not _use_e16_expert_major_forward_tuning(
            config,
            16,
            16,
            operator.weights.has_bias,
            token_indices_identity=True,
        )
    ):
        raise ValueError(
            "warmup_sonic_e16_training requires the tuned bias-free BF16 "
            "H2048/I768/E16/K1 expert-major configuration"
        )

    requested_sizes = (
        E16_ROUTE_POLICY_SIZES
        if route_policy_sizes is None
        else tuple(route_policy_sizes)
    )
    if not requested_sizes:
        raise ValueError("route_policy_sizes must contain at least one value")
    canonical_sizes = tuple(
        dict.fromkeys(canonical_e16_route_policy_size(size) for size in requested_sizes)
    )

    device = operator.weights.device
    routes = config.num_experts
    hidden_states = torch.zeros(
        (routes, config.hidden_size),
        dtype=torch.bfloat16,
        device=device,
    )
    route_weights = torch.ones(routes, dtype=torch.float32, device=device)
    expert_counts = torch.ones(routes, dtype=torch.int32, device=device)
    grad_output = torch.zeros_like(hidden_states)

    for policy_size in canonical_sizes:
        _, state = operator.forward_expert_major_counts_training(
            hidden_states,
            expert_counts,
            route_weights,
            interleaved_w1=interleaved_w1,
            route_policy_size=policy_size,
        )
        sonic_moe_backward_expert_major(
            hidden_states,
            w1,
            w2,
            route_weights,
            grad_output,
            config,
            interleaved_w1=interleaved_w1,
            forward_state=state,
        )

    if synchronize:
        torch.cuda.synchronize(device)
    return canonical_sizes


@torch.no_grad()
def sonic_moe_reference(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    router_logits: torch.Tensor,
    config: SonicMoEConfig,
    *,
    b1: torch.Tensor | None = None,
    b2: torch.Tensor | None = None,
) -> torch.Tensor:
    """Approximate oracle with FP32 GEMMs and A16 rounding boundaries.

    The production kernel uses unordered 16-bit atomic additions in stage 2, while
    this reference accumulates routes in FP32 and casts once at the end.  Small
    last-bit differences are therefore expected.
    """

    if tuple(hidden_states.shape) != (router_logits.shape[0], config.hidden_size):
        raise ValueError("hidden_states shape does not match config/router_logits")
    if tuple(w1.shape) != (
        config.num_experts,
        config.stage1_projection_size,
        config.hidden_size,
    ):
        raise ValueError("w1 shape does not match config")
    if tuple(w2.shape) != (
        config.num_experts,
        config.hidden_size,
        config.intermediate_size,
    ):
        raise ValueError("w2 shape does not match config")
    compute_dtype = _COMPUTE_DTYPES[config.compute_dtype]
    if hidden_states.dtype != compute_dtype:
        raise TypeError(
            f"hidden_states must use {compute_dtype} for this config, got "
            f"{hidden_states.dtype}"
        )
    prepared_b1, prepared_b2 = _prepare_biases(b1, b2, w1, config)

    probs = torch.softmax(router_logits.float(), dim=-1)
    route_weights, route_ids = torch.topk(probs, config.top_k, dim=-1)
    if config.renormalize:
        route_weights = route_weights / route_weights.sum(dim=-1, keepdim=True)

    x = hidden_states.float()
    # Match the prepare API: arbitrary floating-point source weights are first
    # materialized in the configured A16 storage dtype before the kernel reads
    # them.  Keeping that boundary in the oracle matters for FP32 source weights.
    w1f = w1.to(compute_dtype).float()
    w2f = w2.to(compute_dtype).float()
    b1f = None if prepared_b1 is None else prepared_b1.float()
    b2f = None if prepared_b2 is None else prepared_b2.float()
    result = torch.zeros(
        (hidden_states.shape[0], config.hidden_size),
        dtype=torch.float32,
        device=hidden_states.device,
    )
    for slot in range(config.top_k):
        expert = route_ids[:, slot]
        stage1 = torch.bmm(w1f[expert], x.unsqueeze(-1)).squeeze(-1)
        if b1f is not None:
            stage1 = stage1 + b1f[expert]
        # The legacy grouped GEMM materializes A16 preactivation before the
        # activation kernel reloads it in FP32. The fused kernel preserves this
        # observable rounding boundary.
        stage1 = stage1.to(compute_dtype).float()
        # Stage 1 stores an A16 sorted intermediate before stage 2 reloads it.
        if config.activation == "swiglu":
            gate, up = stage1.split(config.intermediate_size, dim=-1)
            activated = torch.nn.functional.silu(gate) * up
        elif config.activation == "geglu":
            gate, up = stage1.split(config.intermediate_size, dim=-1)
            activated = torch.nn.functional.gelu(gate, approximate="tanh") * up
        elif config.activation == "reglu":
            gate, up = stage1.split(config.intermediate_size, dim=-1)
            activated = torch.nn.functional.relu(gate) * up
        elif config.activation == "gelu_tanh_approx":
            activated = torch.nn.functional.gelu(stage1, approximate="tanh")
        elif config.activation == "relu":
            activated = torch.nn.functional.relu(stage1)
        elif config.activation == "silu":
            activated = torch.nn.functional.silu(stage1)
        elif config.activation == "relu_sq":
            activated = torch.nn.functional.relu(stage1).square()
        else:  # guarded by SonicMoEConfig validation
            raise AssertionError(f"unexpected activation {config.activation!r}")
        activated = activated.to(compute_dtype).float()
        projected = torch.bmm(w2f[expert], activated.unsqueeze(-1)).squeeze(-1)
        if b2f is not None:
            projected = projected + b2f[expert]
        projected = projected.to(compute_dtype).float()
        result.add_(projected * route_weights[:, slot, None])
    return result.to(compute_dtype)


@torch.no_grad()
def sonic_moe_mxfp4_reference(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    router_logits: torch.Tensor,
    config: SonicMoEConfig,
    *,
    b1: torch.Tensor | None = None,
    b2: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reference the A16W4 path after canonical per-1x32 weight quantization.

    This intentionally quantizes the logical weights before calling
    :func:`sonic_moe_reference`, separating kernel error from the model-level
    error introduced by MXFP4 weights.
    """

    _validate_weight_inputs(w1, w2, config)
    w1_quant, w1_scale = _quantize_mxfp4_weight(w1)
    w2_quant, w2_scale = _quantize_mxfp4_weight(w2)
    w1_dequant = _dequantize_mxfp4_weight(w1_quant, w1_scale)
    w2_dequant = _dequantize_mxfp4_weight(w2_quant, w2_scale)
    return sonic_moe_reference(
        hidden_states,
        w1_dequant,
        w2_dequant,
        router_logits,
        config,
        b1=b1,
        b2=b2,
    )


__all__ = [
    "E16_ROUTE_POLICY_SIZES",
    "SonicMoE",
    "SonicMoEConfig",
    "SonicMoEDynamicWorkspacePool",
    "SonicMoEForwardState",
    "SonicMoERoutesForwardState",
    "SonicMoEWeights",
    "SonicMoEWorkspace",
    "canonical_e16_route_policy_size",
    "prepare_sonic_bf16_weights",
    "prepare_sonic_fp16_weights",
    "prepare_sonic_mxfp4_weights",
    "sonic_moe_backward",
    "sonic_moe_backward_expert_major",
    "sonic_moe_backward_routes",
    "sonic_moe_mxfp4_reference",
    "sonic_moe_reference",
    "warmup_sonic_e16_training",
]
