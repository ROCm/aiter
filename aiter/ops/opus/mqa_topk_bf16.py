# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""BF16 MQA logits fused with top-k (OPUS, gfx950): the DSA lightning indexer for training.

Per query row ``r`` over its window ``[row_starts[r], row_ends[r])`` of key rows::

    score[r, c] = sum_h relu(q[r, h] . k[c]) * weights[r, h]
    indices[r]  = the top-``topk`` keys of row ``r``

The score matrix is never written: the op samples a threshold, emits the keys above it and
selects exactly per row. ``mqa_logits_bf16`` writes the dense scores (reference and debug)::

    cfg = mqa_topk_bf16_config(dev, max_rows=..., max_keys=...)   # once per shape bucket
    indices, values = mqa_topk_bf16(q, k, weights, row_starts, row_ends, cfg=cfg)
    logits = mqa_logits_bf16(q, k, weights, row_starts, row_ends, cfg=cfg)

LAYOUTS (contiguous)::

    q        [rows, 32, 128] bf16     weights  [rows, 32]  fp32 (scales folded in)
    k        [keys, 128]     bf16     row_starts, row_ends [rows] int32, half-open

Windows are in key-row space and are clamped to ``[0, keys)``; causality is never inferred from
the row order, so a context-parallel caller passes its local query rows with the gathered keys.

OUTPUT: ``indices [rows, min(topk, keys)]`` int32 absolute key rows. A row with ``n`` valid keys
holds ``min(n, W)`` of them in ascending order, no duplicates, then ``-1``. Rows with more than
``W`` keys keep the top ``W`` by (score desc, index asc). ``values`` (optional, fp32) are the
selected scores, ``-inf`` at ``-1``. Scores use BF16 MFMA with FP32 accumulation in a fixed
order, so the results are bitwise deterministic. NaN or Inf inputs are undefined.
"""

from dataclasses import dataclass

import torch

from ...jit.utils.chip_info import get_gfx_runtime

SUPPORTED_GFX = ("gfx950",)
HEADS = 32
HEAD_DIM = 128
_PENDING = "gfx950 OPUS mqa_topk_bf16 kernels are pending"


@dataclass(frozen=True)
class MqaTopkConfig:
    """A launch configuration for one shape bucket (:func:`mqa_topk_bf16_config`)."""

    device: torch.device
    heads: int
    head_dim: int
    topk: int
    max_rows: int
    max_keys: int
    relu: bool
    low_memory: bool

    @property
    def width(self) -> int:
        """Columns of ``indices`` / ``values`` for a call with ``max_keys`` keys."""
        return min(self.topk, self.max_keys)

    def workspace(self) -> torch.Tensor:
        """Device workspace for one live call, sized from the bounds (caller-held)."""
        raise NotImplementedError(_PENDING)


def mqa_topk_bf16_config(
    device,
    *,
    max_rows: int,
    max_keys: int,
    heads: int = HEADS,
    head_dim: int = HEAD_DIM,
    topk: int = 2048,
    relu: bool = True,
    low_memory: bool = False,
) -> MqaTopkConfig:
    """Check the bounds against the device and return the configuration."""
    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError(f"mqa_topk_bf16 runs on a GPU, got {device}")
    arch = get_gfx_runtime()
    if arch not in SUPPORTED_GFX:
        raise RuntimeError(f"mqa_topk_bf16 supports {SUPPORTED_GFX}, not {arch}")
    if (heads, head_dim) != (HEADS, HEAD_DIM):
        raise ValueError(f"heads x head_dim must be {HEADS} x {HEAD_DIM}")
    if topk < 1 or max_rows < 0 or not 1 <= max_keys < 2**31:
        raise ValueError(
            f"need topk >= 1, max_rows >= 0, 1 <= max_keys < 2^31; "
            f"got {topk}, {max_rows}, {max_keys}"
        )
    return MqaTopkConfig(
        device, heads, head_dim, topk, max_rows, max_keys, relu, low_memory
    )


def _check_inputs(cfg, q, k, weights, row_starts, row_ends) -> tuple[int, int]:
    rows, keys = q.shape[0], k.shape[0]
    expect = (
        ("q", q, torch.bfloat16, (rows, cfg.heads, cfg.head_dim)),
        ("k", k, torch.bfloat16, (keys, cfg.head_dim)),
        ("weights", weights, torch.float32, (rows, cfg.heads)),
        ("row_starts", row_starts, torch.int32, (rows,)),
        ("row_ends", row_ends, torch.int32, (rows,)),
    )
    for name, t, dtype, shape in expect:
        if t.dtype != dtype or tuple(t.shape) != shape:
            raise ValueError(
                f"{name} must be {dtype} {shape}, got {t.dtype} {tuple(t.shape)}"
            )
        if t.device != cfg.device or not t.is_contiguous():
            raise ValueError(f"{name} must be contiguous on {cfg.device}")
    if rows > cfg.max_rows or keys > cfg.max_keys:
        raise ValueError(
            f"rows {rows} / keys {keys} exceed the config bounds "
            f"{cfg.max_rows} / {cfg.max_keys}"
        )
    return rows, keys


def mqa_topk_bf16(
    q: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    *,
    cfg: MqaTopkConfig,
    workspace: torch.Tensor | None = None,
    indices: torch.Tensor | None = None,
    values: torch.Tensor | None = None,
    return_values: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Top-k key rows of each query row (see the module docstring); returns (indices, values).

    ``values`` is returned when ``return_values`` is set or a ``values`` tensor is passed.
    """
    _check_inputs(cfg, q, k, weights, row_starts, row_ends)
    raise NotImplementedError(_PENDING)


def mqa_logits_bf16(
    q: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    *,
    cfg: MqaTopkConfig,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Dense scores ``[rows, keys]`` fp32, ``-inf`` outside each window; returns ``out``."""
    _check_inputs(cfg, q, k, weights, row_starts, row_ends)
    raise NotImplementedError(_PENDING)
