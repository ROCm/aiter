# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import torch
import triton

from aiter.ops.triton._triton_kernels.fusions.fused_mixed_sample import (
    _fused_mixed_sample_pick_kernel,
    _fused_mixed_sample_tile_kernel,
)
from aiter.ops.triton.utils.config_utils import (
    load_config_json,
    resolve_config_dir,
    select_leq_config,
)
from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER = AiterTritonLogger()

_CONFIG_NAME = "FUSED_MIXED_SAMPLE"


def _get_config(M: int) -> dict:
    """Config for M rows: the smallest ``M_LEQ_<x> >= M``, else ``any``.

    Reads ``configs/<arch>/triton/fusions/fused_mixed_sample/DEFAULT.json``.
    Raises ``FileNotFoundError`` when the running arch has no such file.
    """
    cfg_dir = resolve_config_dir("fusions", _CONFIG_NAME)
    raw = load_config_json(f"{cfg_dir}/DEFAULT.json", required=True)
    return select_leq_config(raw, M, prefix="M_LEQ_")


def fused_mixed_sample(
    input: torch.Tensor,
    exponentials: torch.Tensor,
    temperatures: torch.Tensor,
    eps: float = 1e-10,
    greedy_eps: float = 0.0,
    out: torch.Tensor | None = None,
    config: dict | None = None,
) -> torch.Tensor:
    """Sample one token per row with the exponential race, or take the argmax.

    For each row ``i``:

    - If ``temperatures[i] <= greedy_eps``, the token is the argmax of
      ``input[i]``. Ties go to the lowest id.
    - Else, the token is
      ``argmax_j softmax(input[i] / max(T, 1e-5))_j / (exponentials[i, j] + eps)``.
      This is the race of ``aiter.mixed_sample_outer_exponential``, with the
      same noise.

    The kernel splits each row into vocab tiles (stage 1) and combines the
    tiles (stage 2). With ``greedy_eps=0.0`` and ``temperatures >= 0``, the
    greedy rule is the ``temperature == 0`` rule of
    ``mixed_sample_outer_exponential``.

    Args:
        input: logits, shape [M, V], fp32, fp16 or bf16. The last dim must be
            contiguous.
        exponentials: Exp(1) noise, shape [M, V], fp32. The last dim must be
            contiguous. A broadcast row (stride 0 in dim 0) is accepted.
        temperatures: shape [M], fp32.
        eps: added to the noise before the division.
        greedy_eps: rows with a temperature at or below this value are greedy.
        out: optional output, shape [M], int32, contiguous.
        config: optional config with ``BLOCK_V``, ``num_warps`` and
            ``num_warps_pick``. If it is None, the tuned config is loaded.

    Returns:
        The token ids, shape [M], int32.

    Limits:
        V must be at least 1 and smaller than 2**31. The tuned configs are for
        gfx942 and gfx950. Without ``config``, other archs raise
        ``FileNotFoundError``.
    """
    if input.dim() != 2 or exponentials.shape != input.shape:
        raise ValueError(
            f"input and exponentials must be [M, V] with the same shape, got "
            f"{tuple(input.shape)} and {tuple(exponentials.shape)}"
        )
    if exponentials.dtype != torch.float32 or temperatures.dtype != torch.float32:
        raise ValueError("exponentials and temperatures must be float32")
    if input.stride(1) != 1 or exponentials.stride(1) != 1:
        raise ValueError("the last dim of input and exponentials must be contiguous")
    M, V = input.shape
    if not 1 <= V < 2**31:
        raise ValueError(f"V must be at least 1 and smaller than 2**31, got {V}")
    if temperatures.shape != (M,):
        raise ValueError(f"temperatures must have shape ({M},)")
    if out is None:
        out = torch.empty(M, dtype=torch.int32, device=input.device)
    elif out.shape != (M,) or out.dtype != torch.int32 or out.stride(0) != 1:
        raise ValueError(f"out must be a contiguous int32 tensor with shape ({M},)")
    if M == 0:
        return out

    if config is None:
        config = _get_config(M)
    block_v = config["BLOCK_V"]
    n_tiles = triton.cdiv(V, block_v)
    _LOGGER.info(
        "FUSED_MIXED_SAMPLE: input=%s dtype=%s BLOCK_V=%s n_tiles=%s",
        tuple(input.shape),
        input.dtype,
        block_v,
        n_tiles,
    )
    f32 = {"dtype": torch.float32, "device": input.device}
    part_m = torch.empty((n_tiles, M), **f32)
    part_key = torch.empty((n_tiles, M), **f32)
    part_idx = torch.empty((n_tiles, M), dtype=torch.int32, device=input.device)
    temps = temperatures.contiguous()
    _fused_mixed_sample_tile_kernel[(M, n_tiles)](
        input,
        input.stride(0),
        exponentials,
        exponentials.stride(0),
        temps,
        eps,
        greedy_eps,
        part_m,
        part_key,
        part_idx,
        M,
        V,
        BLOCK_V=block_v,
        num_warps=config["num_warps"],
    )
    _fused_mixed_sample_pick_kernel[(M,)](
        out,
        temps,
        greedy_eps,
        part_m,
        part_key,
        part_idx,
        M,
        n_tiles,
        V,
        BLOCK_T=max(16, triton.next_power_of_2(n_tiles)),
        num_warps=config["num_warps_pick"],
    )
    return out
