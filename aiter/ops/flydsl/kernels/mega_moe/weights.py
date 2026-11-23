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


class SharedExpertInput:
    """What the small-input kernel needs to compute the model's shared expert itself (``shared_in``), and the
    way back to the dense launches.

    ``x_fp8`` / ``x_scale``: the group-32 MXFP8 pair of the call's rows (E4M3 ``[rows, model_dim]``, row-major E8M0
    ``[rows, model_dim / 32]``); ``gate_up`` / ``gate_up_scale``: the ``[gate | up]`` E4M3 weight ``[2 inter,
    model_dim]`` in the (16, 16) preshuffle with its 32x32-block E8M0 scales; ``down`` / ``down_scale``: the same
    for ``[model_dim, inter]``. ``dense()`` computes the shared expert's BF16 output the usual way; a route that
    cannot fuse it calls that instead (``used_dense`` then tells the caller).
    """

    def __init__(self, x_fp8, x_scale, gate_up, gate_up_scale, down, down_scale, dense):
        self.x_fp8, self.x_scale = x_fp8, x_scale
        self.gate_up, self.gate_up_scale, self.down, self.down_scale = (
            gate_up,
            gate_up_scale,
            down,
            down_scale,
        )
        self._dense = dense
        self.used_dense = False

    def dense(self):
        self.used_dense = True
        return self._dense()

    @staticmethod
    def _readable(sizes) -> bool:
        return all(
            t.numel() == n
            and t.element_size() == 1
            and t.is_contiguous()
            and t.data_ptr() % 16 == 0
            for t, n in sizes
        )

    def weights_fit(self, model_dim: int, inter_dim: int) -> bool:
        """Shapes, layout and alignment of the weights the kernel reads (metadata only)."""
        return self._readable(
            (
                (self.gate_up, 2 * inter_dim * model_dim),
                (self.gate_up_scale, (2 * inter_dim // 32) * (model_dim // 32)),
                (self.down, model_dim * inter_dim),
                (self.down_scale, (model_dim // 32) * (inter_dim // 32)),
            )
        )

    def fits(self, rows: int, model_dim: int, inter_dim: int) -> bool:
        """``weights_fit`` and the same for the pair of a ``rows``-row call (metadata only)."""
        return (
            rows > 0
            and self.weights_fit(model_dim, inter_dim)
            and self._readable(
                (
                    (self.x_fp8, rows * model_dim),
                    (self.x_scale, rows * (model_dim // 32)),
                )
            )
        )
