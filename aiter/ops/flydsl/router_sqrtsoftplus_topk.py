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

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch

from ...jit.utils.chip_info import get_gfx_runtime
from .kernels.router_sqrtsoftplus_topk import (
    EXPERTS,
    HIDDEN,
    TOPK,
    build_router_gemm_module,
    build_router_topk_module,
)
from .kernels.tensor_shim import _run_compiled, current_stream_arg, ptr_arg

_SUPPORTED_GFX = ("gfx950",)
# The fp32 partials ``[ksp, m, 384]`` (ksp <= 16) use 32-bit byte offsets.
_MAX_ROWS = 65536
# x's buffer view spans ``m * row_stride * 2`` bytes; keeping it under 1 GiB
# leaves the kernels' 32-bit offsets (and their out-of-bound marker) clear.
_MAX_X_BYTES = 1 << 30
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
    (32, (1, 1, 5, 1, 8, 2, 4)),
    (48, (2, 3, 16, 1, 5, 2, 4)),
    (64, (2, 3, 16, 1, 5, 2, 4)),
    (96, (3, 3, 16, 1, 5, 1, 8)),
    (128, (2, 6, 16, 1, 2, 4, 2)),
    (192, (3, 3, 8, 1, 5, 2, 4)),
    (256, (4, 2, 5, 2, 1, 4, 2)),
    (384, (3, 3, 4, 4, 5, 8, 1)),
    (512, (2, 6, 8, 1, 4, 8, 1)),
    (768, (3, 4, 5, 1, 4, 8, 1)),
)
_LARGE = (4, 3, 1, 4, 5, 8, 1)
# The same for the gate in the load-time layout of ``shuffle_router_gate``.
_TILES_SHUFFLED = (
    (16, (1, 1, 8, 1, 10, 1, 8)),
    (32, (1, 2, 10, 2, 8, 2, 4)),
    (48, (1, 3, 10, 4, 4, 1, 8)),
    (64, (1, 2, 5, 1, 8, 4, 2)),
    (96, (1, 3, 8, 1, 2, 2, 4)),
    (128, (1, 3, 8, 1, 1, 4, 2)),
    (192, (1, 3, 5, 1, 8, 4, 2)),
    (256, (2, 4, 5, 1, 4, 8, 1)),
    (384, (3, 3, 8, 1, 5, 8, 1)),
    (512, (4, 3, 4, 1, 4, 2, 4)),
    (768, (3, 6, 4, 1, 2, 2, 4)),
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


def _x_row_stride(x) -> int:
    # One row reads only its own 5120 elements, whatever its stride says.
    return x.stride(0) if x.shape[0] > 1 else HIDDEN


def _unsupported_reason(x, weight, bias, is_padding) -> str | None:
    if get_gfx_runtime() not in _SUPPORTED_GFX:
        return f"needs one of {_SUPPORTED_GFX}, running on {get_gfx_runtime()}"
    if x.dim() != 2 or x.shape[1] != HIDDEN or x.dtype != torch.bfloat16:
        return f"x must be bf16 [m, {HIDDEN}], got {x.dtype} {tuple(x.shape)}"
    m = x.shape[0]
    if m > _MAX_ROWS:
        return f"at most {_MAX_ROWS} rows, got {m}"
    # A lane loads eight consecutive k (one dwordx4) of a row. Rows must not
    # overlap: a zero (expanded) or short row stride would read other rows.
    if x.stride(1) != 1 or x.data_ptr() % 16:
        return "x needs unit column stride and 16 B alignment"
    if m > 1 and (x.stride(0) < HIDDEN or x.stride(0) % 8):
        return f"x row stride must be >= {HIDDEN} and divisible by 8"
    if m * _x_row_stride(x) * x.element_size() > _MAX_X_BYTES:
        return "x spans more than 1 GiB (32-bit buffer offsets)"
    if (
        weight.shape != (EXPERTS, HIDDEN)
        or weight.dtype != torch.bfloat16
        or not weight.is_contiguous()
        or weight.data_ptr() % 16
    ):
        return f"weight must be contiguous, 16 B aligned bf16 [{EXPERTS}, {HIDDEN}]"
    if bias.shape != (EXPERTS,) or bias.dtype != torch.float32:
        return f"bias must be fp32 [{EXPERTS}]"
    # Each lane loads its six biases as three 8 B pairs.
    if not bias.is_contiguous() or bias.data_ptr() % 8:
        return "bias must be contiguous and 8 B aligned"
    tensors = [x, weight, bias]
    if is_padding is not None:
        # Read one byte per row: any offset view of a bool [m] works.
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
        CUDA-graph safe, no atomics, bitwise repeatable. The first call of
        each row bucket compiles its kernels: call
        :func:`router_sqrtsoftplus_topk_warmup` before graph capture.
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
        # The GEMM stores four fp32 partials (16 B) per access.
        if name == "partials" and t.data_ptr() % 16:
            raise ValueError("partials must be 16 B aligned")
        if t.device != dev:
            raise ValueError(f"{name} must be on {dev}")
    if m:
        stream = current_stream_arg()
        # Untyped-pointer launchers: only the device changes a handle.
        _run_compiled(
            _gemm_launcher(tile, weight_shuffled),
            ptr_arg(x),
            ptr_arg(weight),
            ptr_arg(partials),
            m,
            _x_row_stride(x),
            stream,
            specialization_key=dev.index,
        )
        _run_compiled(
            _topk_launcher(ksp, is_padding is not None),
            ptr_arg(partials),
            ptr_arg(bias),
            (
                flyc.from_c_void_p(fx.Uint8, 0)
                if is_padding is None
                else ptr_arg(is_padding)
            ),
            ptr_arg(topk_weights),
            ptr_arg(topk_ids),
            m,
            struct.unpack("<i", struct.pack("<f", float(routed_scaling_factor)))[0],
            stream,
            specialization_key=dev.index,
        )
    return topk_weights, topk_ids


def router_sqrtsoftplus_topk_warmup(
    device: torch.device | int | str | None = None,
    weight_shuffled: bool = False,
    with_padding: bool = False,
    max_rows: int = TUNED_MAX_ROWS,
) -> None:
    """Compile (and run once, on zeros) the kernels of every row bucket up to
    ``max_rows`` for this gate layout and padding flag, so that no call made
    under CUDA-graph capture compiles. Call it once per device before
    capture with the arguments the model will use (``with_padding`` when it
    passes ``is_padding``)."""
    dev = (
        torch.device("cuda", torch.cuda.current_device())
        if device is None
        else (
            torch.device("cuda", device)
            if isinstance(device, int)
            else torch.device(device)
        )
    )
    bounds = [b for b, _ in (_TILES_SHUFFLED if weight_shuffled else _TILES)]
    rows = [b for b in bounds if b <= max_rows]
    if max_rows > bounds[-1]:
        rows.append(bounds[-1] + 1)  # the large-row tile
    elif max_rows not in rows:
        rows.append(max_rows)
    with torch.cuda.device(dev):
        weight = torch.zeros((EXPERTS, HIDDEN), dtype=torch.bfloat16, device=dev)
        bias = torch.zeros(EXPERTS, dtype=torch.float32, device=dev)
        for m in rows:
            x = torch.zeros((m, HIDDEN), dtype=torch.bfloat16, device=dev)
            pad = torch.zeros(m, dtype=torch.bool, device=dev) if with_padding else None
            flydsl_router_sqrtsoftplus_topk(
                x, weight, bias, is_padding=pad, weight_shuffled=weight_shuffled
            )
