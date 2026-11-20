# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL MoE router: bf16 gate GEMM + sqrt-softplus biased top-k (gfx950).

Two launches replace the router GEMM (fp32 out) and the
``topk_softplus_sqrt``-style top-k:

    logits = x @ weight.T                     bf16 products, fp32 accumulate
    score  = sqrt(logit if logit > 20 else log(1 + exp(logit)))
    ids    = the topk experts with the largest score + bias, descending
             (ties: the lowest expert id)
    w      = ((score + bias)[ids] - bias[ids]) * (scale / their sum, in slot order)
    padding rows: ids -1, weights 0

Given the same logits, ids and weights equal vLLM's ``topk_softplus_sqrt`` bit
for bit. The GEMM sums k-slices in a fixed order that depends on the row-count
bucket, so the logits differ from another GEMM's within fp32 accumulation error.

Only hidden size 5120 with 128 experts and top-3 is tuned and compiled;
``router_sqrtsoftplus_topk_supported`` tells a caller when to fall back.
"""

import functools
import struct

import torch

from aiter.jit.utils.chip_info import get_gfx_runtime

from .kernels.router_sqrtsoftplus_topk import H, build_router_gemm, build_router_topk
from .kernels.tensor_shim import _run_compiled

__all__ = [
    "flydsl_router_sqrtsoftplus_topk",
    "router_sqrtsoftplus_topk_partials_shape",
    "router_sqrtsoftplus_topk_supported",
    "shuffle_router_gate",
    "warmup_router_sqrtsoftplus_topk",
]

# (a, b, ksp, kw, chunk, xm, xn) of the GEMM per row-count upper bound, the
# fastest of a random tile sweep at the bound (MI355X, graph replay, cold L2).
# a x b: 16-token x 16-expert MFMA tiles per workgroup; ksp: k-slices over
# workgroups; kw: k-slices over a workgroup's waves; chunk: k-steps per load
# batch; xm x xn: token x expert slabs over the 8 XCDs.
_TILES = {
    (128, 3): {
        False: (
            (16, (2, 1, 16, 2, 5, 1, 8)),
            (48, (1, 2, 16, 1, 10, 2, 4)),
            (96, (1, 2, 20, 1, 8, 2, 4)),
            (192, (1, 2, 5, 1, 8, 4, 2)),
            (384, (1, 2, 5, 1, 8, 4, 2)),
            (768, (4, 2, 4, 1, 5, 2, 4)),
            (None, (4, 2, 4, 1, 5, 2, 4)),
        ),
        # The load-time gate layout (shuffle_router_gate).
        True: (
            (16, (1, 1, 20, 1, 8, 2, 4)),
            (48, (1, 2, 16, 1, 10, 2, 4)),
            (96, (2, 1, 16, 2, 5, 4, 2)),
            (192, (1, 2, 4, 1, 10, 8, 1)),
            (384, (3, 2, 16, 1, 5, 8, 1)),
            (768, (3, 2, 4, 4, 5, 8, 1)),
            (None, (3, 2, 4, 4, 5, 8, 1)),
        ),
    },
}


def _tile(num_experts: int, topk: int, m: int, weight_shuffled: bool):
    for bound, tile in _TILES[(num_experts, topk)][weight_shuffled]:
        if bound is None or m <= bound:
            return tile
    raise AssertionError("unreachable")


@functools.cache
def _gemm_launcher(device_index, experts, tile, wshuf):
    # One compiled handle per device and variant.
    return build_router_gemm(experts, *tile, wshuf)


@functools.cache
def _topk_launcher(device_index, experts, topk, ksp, has_padding):
    return build_router_topk(experts, topk, ksp, has_padding)


def router_sqrtsoftplus_topk_supported(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    topk: int,
) -> bool:
    """Whether ``flydsl_router_sqrtsoftplus_topk`` takes these inputs: gfx950,
    ``x`` bf16 ``[m, 5120]`` with unit column stride, ``weight`` bf16 contiguous
    ``[E, 5120]`` and ``bias`` fp32 contiguous ``[E]`` on x's device, with
    ``(E, topk)`` a tuned pair (today ``(128, 3)``)."""
    return (
        x.is_cuda
        and get_gfx_runtime() == "gfx950"
        and (weight.shape[0] if weight.dim() == 2 else -1, topk) in _TILES
        and x.dtype == torch.bfloat16
        and x.dim() == 2
        and x.shape[1] == H
        and x.stride(1) == 1
        and 0 <= x.stride(0)
        and x.shape[0] * x.stride(0) < 2**30
        and weight.dtype == torch.bfloat16
        and weight.shape[1] == H
        and weight.is_contiguous()
        and weight.device == x.device
        and bias.dtype == torch.float32
        and bias.shape == (weight.shape[0],)
        and bias.is_contiguous()
        and bias.device == x.device
    )


def router_sqrtsoftplus_topk_partials_shape(
    num_tokens: int, num_experts: int, topk: int, weight_shuffled: bool = False
) -> tuple[int, int, int]:
    """Shape of the fp32 GEMM partials scratch ``[ksp, num_tokens, num_experts]``
    for a caller that preallocates it (a CUDA-graph static buffer)."""
    ksp = _tile(num_experts, topk, num_tokens, weight_shuffled)[2]
    return ksp, num_tokens, num_experts


def shuffle_router_gate(weight: torch.Tensor) -> torch.Tensor:
    """The gate ``[E, 5120]`` bf16 in the load-time layout
    ``[expert tile, k step, lane, 8]``: lane ``16 g + l`` of tile ``t`` and k-step
    ``s`` holds ``weight[16 t + l, 32 s + 8 g : 32 s + 8 g + 8]``. A permutation
    (bit-exact); convert once at weight load, before graph capture, and pass
    ``weight_shuffled=True``."""
    assert weight.dtype == torch.bfloat16 and weight.dim() == 2
    e, h = weight.shape
    assert h == H and e % 16 == 0
    return (
        weight.reshape(e // 16, 16, h // 32, 4, 8)
        .permute(0, 2, 3, 1, 4)
        .contiguous()
        .reshape(e, h)
    )


def flydsl_router_sqrtsoftplus_topk(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    topk: int,
    routed_scaling_factor: float = 1.0,
    *,
    is_padding: torch.Tensor | None = None,
    topk_weights: torch.Tensor | None = None,
    topk_ids: torch.Tensor | None = None,
    partials: torch.Tensor | None = None,
    weight_shuffled: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Router of one MoE layer (two launches, CUDA-graph safe, no atomics).

    Args:
        x: bf16 ``[m, 5120]``, unit column stride, any row stride.
        weight: bf16 gate ``[E, 5120]`` (``shuffle_router_gate`` layout when
            ``weight_shuffled``).
        bias: fp32 ``[E]`` score correction bias.
        topk: experts per token.
        routed_scaling_factor: the weights of a token sum to this.
        is_padding: optional bool ``[m]``; those rows get id -1 and weight 0.
        topk_weights / topk_ids: optional fp32 / int32 contiguous ``[m, topk]``
            outputs.
        partials: optional fp32 contiguous scratch of
            ``router_sqrtsoftplus_topk_partials_shape``.

    Returns:
        ``(topk_weights, topk_ids)`` in descending ``score + bias`` order.
    """
    if not router_sqrtsoftplus_topk_supported(x, weight, bias, topk):
        raise ValueError(
            "flydsl_router_sqrtsoftplus_topk needs gfx950, bf16 x [m, 5120] with unit "
            "column stride, a contiguous bf16 gate [E, 5120], fp32 bias [E] and "
            f"(E, topk) in {sorted(_TILES)}"
        )
    m = x.shape[0]
    e = weight.shape[0]
    dev = x.device
    if topk_weights is None:
        topk_weights = torch.empty((m, topk), dtype=torch.float32, device=dev)
    if topk_ids is None:
        topk_ids = torch.empty((m, topk), dtype=torch.int32, device=dev)
    tile = _tile(e, topk, m, weight_shuffled)
    ksp = tile[2]
    if partials is None:
        partials = torch.empty((ksp, m, e), dtype=torch.float32, device=dev)
    for t, dtype, shape, name in (
        (topk_weights, torch.float32, (m, topk), "topk_weights"),
        (topk_ids, torch.int32, (m, topk), "topk_ids"),
        (partials, torch.float32, (ksp, m, e), "partials"),
    ):
        if not (
            t.dtype == dtype
            and tuple(t.shape) == shape
            and t.is_contiguous()
            and t.device == dev
        ):
            raise ValueError(
                f"{name} must be a contiguous {dtype} {shape} on x's device"
            )
    if is_padding is not None and not (
        is_padding.dtype == torch.bool
        and tuple(is_padding.shape) == (m,)
        and is_padding.is_contiguous()
        and is_padding.device == dev
    ):
        raise ValueError("is_padding must be a contiguous bool [m] on x's device")
    if m:
        stream = torch.cuda.current_stream(dev)
        _run_compiled(
            _gemm_launcher(dev.index, e, tile, weight_shuffled),
            x.data_ptr(),
            weight.data_ptr(),
            partials.data_ptr(),
            m,
            x.stride(0),
            stream,
        )
        _run_compiled(
            _topk_launcher(dev.index, e, topk, ksp, is_padding is not None),
            partials.data_ptr(),
            bias.data_ptr(),
            0 if is_padding is None else is_padding.data_ptr(),
            topk_weights.data_ptr(),
            topk_ids.data_ptr(),
            m,
            struct.unpack("<i", struct.pack("<f", routed_scaling_factor))[0],
            stream,
        )
    return topk_weights, topk_ids


def warmup_router_sqrtsoftplus_topk(
    device: torch.device | None = None, num_experts: int = 128, topk: int = 3
) -> None:
    """Compile every row-count bucket, with and without padding, for both gate
    layouts, before CUDA-graph capture."""
    dev = torch.device("cuda") if device is None else torch.device(device)
    if dev.index is None:
        dev = torch.device("cuda", torch.cuda.current_device())
    w = torch.zeros((num_experts, H), dtype=torch.bfloat16, device=dev)
    b = torch.zeros(num_experts, dtype=torch.float32, device=dev)
    for shuffled in (False, True):
        for bound, _ in _TILES[(num_experts, topk)][shuffled]:
            m = 769 if bound is None else bound
            x = torch.zeros((m, H), dtype=torch.bfloat16, device=dev)
            for pad in (None, torch.zeros(m, dtype=torch.bool, device=dev)):
                flydsl_router_sqrtsoftplus_topk(
                    x, w, b, topk, is_padding=pad, weight_shuffled=shuffled
                )
    torch.cuda.synchronize(dev)
