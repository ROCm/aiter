# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
"""Weight-binding and forward-input contracts shared by MegaMoEV2 and its small- and large-input runtimes."""

import torch

from aiter.utility import dtypes

WEIGHT_NAMES = ("w1", "w1_scale", "w2", "w2_scale")


def contiguous_weights(weights):
    """Construction weights as the kernels read them; the caller keeps these alive."""
    return tuple(t if t.is_contiguous() else t.contiguous() for t in weights)


# Packed FP4 values and E8M0 scales: the kernels read the bytes, so these views are interchangeable.
_RAW_BYTE_DTYPES = (torch.uint8, dtypes.fp4x2, dtypes.fp8_e8m0)


def _spec(tensor):
    dtype = torch.uint8 if tensor.dtype in _RAW_BYTE_DTYPES else tensor.dtype
    return tuple(tensor.shape), dtype, tensor.device


def weight_spec(weights):
    """(shape, dtype, device) of each weight tensor, with byte-view dtypes as uint8."""
    return tuple(_spec(t) for t in weights)


def check_bindable(weights, spec):
    """Raise unless ``weights`` can be bound to an instance built for ``spec``.

    Launches keep the bound pointers and read their bytes in the construction layout, so a
    weight must be contiguous (a copy made here would die while a captured graph still uses
    it) and match the construction shape, dtype and device.
    """
    for name, tensor, expected in zip(WEIGHT_NAMES, weights, spec):
        if not isinstance(tensor, torch.Tensor) or not tensor.is_contiguous():
            raise ValueError(f"bind_weights: {name} must be a contiguous tensor")
        got = _spec(tensor)
        if got != expected:
            raise ValueError(
                f"bind_weights: {name} is {got[0]} {got[1]} on {got[2]}; "
                f"the instance was built for {expected[0]} {expected[1]} on {expected[2]}"
            )


def check_forward_inputs(x_bf16, wts, topk_ids, *, max_rows, model_dim, topk, device):
    """Return the row count of one rank's forward inputs; raise unless they are
    contiguous bf16 [rows, model_dim] and float32 / int32 [rows, topk] on ``device``
    with rows <= ``max_rows``."""
    rows = int(x_bf16.shape[0])
    if rows > max_rows:
        raise ValueError(f"run_tokens={rows} > max_tok_per_rank={max_rows}")
    for name, tensor, dtype, shape in (
        ("x_bf16", x_bf16, torch.bfloat16, (rows, model_dim)),
        ("wts", wts, torch.float32, (rows, topk)),
        ("topk_ids", topk_ids, torch.int32, (rows, topk)),
    ):
        if tensor.dtype != dtype or not tensor.is_contiguous():
            raise ValueError(
                f"{name} must be contiguous {str(dtype).removeprefix('torch.')}"
            )
        if tuple(tensor.shape) != shape:
            raise ValueError(f"{name} must have shape {shape}")
        if tensor.device != device:
            # A rank that faults on a foreign pointer would leave its peers waiting.
            raise ValueError(f"x_bf16, wts and topk_ids must be on {device}")
    return rows
