# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL MoE router: bf16 gate GEMM + sqrt-softplus biased top-6 (gfx950).

The router of a DeepSeek-V4.1-Flash MoE layer (hidden 5120, 384 routed
experts, top-6, ``e_score_correction_bias``, renormalized and scaled by the
routed scaling factor) in two launches: a split-K bf16 GEMM to fp32 partials,
then one wave per token that sums the partials, scores and picks the top 6.

Given the same logits the ids and weights equal vLLM's
``topk_hash_softplus_sqrt`` (``renormalize=True``, no hash table, no
``bias_vl``) bit for bit. The logits sum k-slices in a fixed order per row
bucket, so their last fp32 bits can differ from another GEMM's; ids can then
differ only where the 6th and 7th ``score + bias`` are within a few ulp.
"""

import struct

import torch

from ...jit.utils.chip_info import get_gfx_runtime
from .kernels.router_sqrtsoftplus_topk import (
    EXPERTS,
    HIDDEN,
    TOPK,
    build_router_gemm_module,
    build_router_topk_module,
)
from .kernels.tensor_shim import _run_compiled, current_stream_arg

_SUPPORTED_GFX = ("gfx950",)
# The fp32 partials ``[ksp, m, 384]`` (ksp <= 16) use 32-bit byte offsets.
_MAX_ROWS = 65536
# Rows the tile table was tuned for. Above it the op is still correct, but a
# hipBLASLt GEMM + ``topk_gating`` is as fast or faster, so the eligibility
# helper declines.
TUNED_MAX_ROWS = 768

# GEMM tiles ``(a, b, ksp, kw, chunk, xm, xn)`` per row bucket (upper bound):
# ``a x b`` 16x16 MFMA tiles per workgroup, ``ksp`` k-slices over workgroups,
# ``kw`` over a workgroup's waves, ``chunk`` k-steps per load batch, XCD grid
# ``xm x xn``. The fastest of a random sweep on MI355X at each bound.
_TILES = (
    (16, (1, 1, 8, 1, 10, 1, 8)),
    (32, (2, 1, 8, 2, 10, 1, 8)),
    (48, (1, 2, 5, 4, 8, 2, 4)),
    (64, (2, 1, 5, 1, 8, 2, 4)),
    (96, (3, 3, 16, 1, 5, 1, 8)),
    (128, (2, 3, 16, 1, 5, 2, 4)),
    (192, (3, 3, 8, 2, 5, 1, 8)),
    (256, (2, 4, 5, 8, 4, 8, 1)),
    (384, (3, 3, 8, 1, 5, 2, 4)),
    (512, (2, 3, 2, 4, 5, 8, 1)),
    (768, (3, 6, 8, 2, 2, 2, 4)),
)
_LARGE = (4, 3, 1, 4, 5, 8, 1)
# The same for the gate in the load-time layout of ``shuffle_router_gate``.
_TILES_SHUFFLED = (
    (16, (1, 1, 8, 1, 10, 1, 8)),
    (32, (1, 2, 8, 4, 5, 2, 4)),
    (48, (1, 3, 10, 4, 4, 1, 8)),
    (64, (1, 2, 10, 2, 8, 2, 4)),
    (96, (3, 2, 8, 1, 5, 2, 4)),
    (128, (2, 3, 8, 4, 5, 4, 2)),
    (192, (2, 4, 5, 2, 4, 4, 2)),
    (256, (2, 3, 4, 4, 5, 8, 1)),
    (384, (3, 2, 8, 1, 5, 2, 4)),
    (512, (2, 3, 4, 2, 5, 8, 1)),
    (768, (3, 3, 8, 1, 5, 8, 1)),
)

_gemm_launchers: dict = {}
_topk_launchers: dict = {}


def _tile(m: int, weight_shuffled: bool) -> tuple[int, ...]:
    for bound, tile in _TILES_SHUFFLED if weight_shuffled else _TILES:
        if m <= bound:
            return tile
    return _LARGE


def _gemm_launcher(tile, weight_shuffled):
    key = (*tile, weight_shuffled)
    if key not in _gemm_launchers:
        _gemm_launchers[key] = build_router_gemm_module(*tile, weight_shuffled)
    return _gemm_launchers[key]


def _topk_launcher(ksp, has_padding):
    key = (ksp, has_padding)
    if key not in _topk_launchers:
        _topk_launchers[key] = build_router_topk_module(ksp, has_padding)
    return _topk_launchers[key]


def router_sqrtsoftplus_topk_workspace_shape(
    num_rows: int, weight_shuffled: bool = False
) -> tuple[int, int, int]:
    """Shape of the fp32 logits scratch (``partials``) for ``num_rows`` rows."""
    return (_tile(num_rows, weight_shuffled)[2], num_rows, EXPERTS)


def shuffle_router_gate(weight: torch.Tensor) -> torch.Tensor:
    """The gate ``[384, 5120]`` bf16 in the load-time layout
    ``[expert tile, k step, lane, 8]`` (lane = 16 g + l holds
    ``w[16 tile + l, 32 step + 8 g : +8]``), so a wave's weight fragment is
    one contiguous 1 KB run. A permutation (bit-exact); run once at weight
    load and pass ``weight_shuffled=True``."""
    if weight.shape != (EXPERTS, HIDDEN) or weight.dtype != torch.bfloat16:
        raise ValueError(f"weight must be bf16 [{EXPERTS}, {HIDDEN}]")
    return (
        weight.reshape(EXPERTS // 16, 16, HIDDEN // 32, 4, 8)
        .permute(0, 2, 3, 1, 4)
        .contiguous()
        .reshape(EXPERTS, HIDDEN)
    )


def _unsupported_reason(x, weight, bias, is_padding) -> str | None:
    if get_gfx_runtime() not in _SUPPORTED_GFX:
        return f"needs one of {_SUPPORTED_GFX}, running on {get_gfx_runtime()}"
    if x.dim() != 2 or x.shape[1] != HIDDEN or x.dtype != torch.bfloat16:
        return f"x must be bf16 [m, {HIDDEN}], got {x.dtype} {tuple(x.shape)}"
    m = x.shape[0]
    if m > _MAX_ROWS:
        return f"at most {_MAX_ROWS} rows, got {m}"
    # A lane loads eight consecutive k (one dwordx4) of a row.
    if x.stride(1) != 1 or x.stride(0) % 8 or x.data_ptr() % 16:
        return "x needs unit column stride, a row stride divisible by 8, 16 B alignment"
    if m * x.stride(0) * x.element_size() >= 2**31:
        return "x spans 2 GiB or more (32-bit buffer offsets)"
    if (
        weight.shape != (EXPERTS, HIDDEN)
        or weight.dtype != torch.bfloat16
        or not weight.is_contiguous()
    ):
        return f"weight must be contiguous bf16 [{EXPERTS}, {HIDDEN}]"
    if bias.shape != (EXPERTS,) or bias.dtype != torch.float32:
        return f"bias must be fp32 [{EXPERTS}]"
    if not bias.is_contiguous():
        return "bias must be contiguous"
    tensors = [x, weight, bias]
    if is_padding is not None:
        if (
            is_padding.shape != (m,)
            or is_padding.dtype != torch.bool
            or not is_padding.is_contiguous()
        ):
            return f"is_padding must be contiguous bool [{m}]"
        tensors.append(is_padding)
    if len({t.device for t in tensors}) != 1 or not x.is_cuda:
        return "every tensor must be on the same GPU"
    return None


def router_sqrtsoftplus_topk_supported(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    top_k: int = TOPK,
    is_padding: torch.Tensor | None = None,
) -> bool:
    """Whether to route these inputs through
    :func:`flydsl_router_sqrtsoftplus_topk`: gfx950, hidden 5120, 384 experts,
    top-6, fp32 bias and at most ``TUNED_MAX_ROWS`` (768) rows. Callers keep
    their router GEMM + top-k otherwise."""
    return (
        top_k == TOPK
        and x.dim() == 2
        and x.shape[0] <= TUNED_MAX_ROWS
        and _unsupported_reason(x, weight, bias, is_padding) is None
    )


def flydsl_router_sqrtsoftplus_topk(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    routed_scaling_factor: float = 1.0,
    *,
    is_padding: torch.Tensor | None = None,
    weight_shuffled: bool = False,
    topk_weights: torch.Tensor | None = None,
    topk_ids: torch.Tensor | None = None,
    partials: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Router of one MoE layer for a block of rows.

    ``logits = x @ weight.T`` (fp32 accumulate), ``score = sqrt(softplus(
    logit))`` (``logit`` above 20), the 6 experts with the largest
    ``score + bias`` (ties: the lowest id), weights ``score`` of the picks
    (computed as ``(score + bias) - bias``) scaled to sum
    ``routed_scaling_factor``.

    Args:
        x: bf16 ``[m, 5120]``, unit column stride, row stride divisible by 8.
        weight: bf16 ``[384, 5120]``, row-major, or
            ``shuffle_router_gate(gate)`` with ``weight_shuffled=True``.
        bias: fp32 ``[384]`` (the score-correction bias).
        routed_scaling_factor: the picks' weights sum to this.
        is_padding: optional bool ``[m]``; those rows get id -1, weight 0.
        topk_weights, topk_ids: optional contiguous fp32 / int32 ``[m, 6]``.
        partials: optional fp32 scratch of
            ``router_sqrtsoftplus_topk_workspace_shape(m, weight_shuffled)``.

    Returns:
        ``(topk_weights, topk_ids)`` in descending ``score + bias`` order.
        CUDA-graph safe, no atomics, bitwise repeatable; the first call of
        each row bucket compiles its kernels.
    """
    reason = _unsupported_reason(x, weight, bias, is_padding)
    if reason is not None:
        raise ValueError(f"flydsl_router_sqrtsoftplus_topk: {reason}")
    m, dev = x.shape[0], x.device
    tile = _tile(m, weight_shuffled)
    ksp = tile[2]
    if topk_weights is None:
        topk_weights = torch.empty((m, TOPK), dtype=torch.float32, device=dev)
    if topk_ids is None:
        topk_ids = torch.empty((m, TOPK), dtype=torch.int32, device=dev)
    if partials is None:
        partials = torch.empty((ksp, m, EXPERTS), dtype=torch.float32, device=dev)
    for name, t, shape, dtype in (
        ("topk_weights", topk_weights, (m, TOPK), torch.float32),
        ("topk_ids", topk_ids, (m, TOPK), torch.int32),
        ("partials", partials, (ksp, m, EXPERTS), torch.float32),
    ):
        if t.shape != shape or t.dtype != dtype or not t.is_contiguous():
            raise ValueError(f"{name} must be contiguous {dtype} {list(shape)}")
        if t.device != dev:
            raise ValueError(f"{name} must be on {dev}")
    if m:
        stream = current_stream_arg()
        # Raw-address launchers: only the device changes a handle.
        _run_compiled(
            _gemm_launcher(tile, weight_shuffled),
            x.data_ptr(),
            weight.data_ptr(),
            partials.data_ptr(),
            m,
            x.stride(0),
            stream,
            specialization_key=dev.index,
        )
        _run_compiled(
            _topk_launcher(ksp, is_padding is not None),
            partials.data_ptr(),
            bias.data_ptr(),
            0 if is_padding is None else is_padding.data_ptr(),
            topk_weights.data_ptr(),
            topk_ids.data_ptr(),
            m,
            struct.unpack("<i", struct.pack("<f", float(routed_scaling_factor)))[0],
            stream,
            specialization_key=dev.index,
        )
    return topk_weights, topk_ids
