# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import torch
import triton

from aiter.jit.utils.torch_guard import torch_compile_guard
from aiter.ops.triton._triton_kernels.gemm.basic.gemm_a16w16x3 import (
    _gemm_a16w16x3_kernel,
    _get_config,
)
from aiter.ops.triton.utils.common_utils import deserialize_str, serialize_dict
from aiter.ops.triton.utils.device_info import get_num_xcds
from aiter.ops.triton.utils.gemm_config_utils import compute_splitk_params
from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER = AiterTritonLogger()

# Split-K arrival counters, one buffer for each (device, stream). The kernel resets
# each counter that it uses, so a buffer is all zeros between launches.
_TILE_COUNTERS = {}


def split_bf16x3(weight: torch.Tensor) -> torch.Tensor:
    """
    Splits an fp32 weight into three bf16 terms whose fp32 sum is the weight.

    Three bf16 terms hold 24 mantissa bits, as fp32 does, so the sum is exact for
    finite normal values. Call it once, when the weights load.

    Args:
        weight (torch.Tensor): FP32 matrix with shape (N, K).

    Returns:
        torch.Tensor: Contiguous BF16 tensor with shape (3, N, K), largest term first.

    Raises:
        ValueError: If the weight has inf or NaN values.
    """
    assert weight.dtype == torch.float32 and weight.dim() == 2, "fp32 (N, K) weight"
    if not torch.isfinite(weight).all():
        raise ValueError("the weight has inf or NaN values")
    terms = []
    rest = weight.detach().clone()
    for _ in range(3):
        term = rest.to(torch.bfloat16)
        terms.append(term)
        rest -= term.to(torch.float32)
    return torch.stack(terms).contiguous()


def _tile_counters(device: torch.device, num_tiles: int) -> torch.Tensor:
    key = (device.index, torch.cuda.current_stream(device).cuda_stream)
    counters = _TILE_COUNTERS.get(key)
    if counters is None or counters.numel() < num_tiles:
        counters = torch.zeros(max(num_tiles, 1024), dtype=torch.int32, device=device)
        _TILE_COUNTERS[key] = counters
    return counters


def gemm_a16w16x3_fake_tensor(
    x: torch.Tensor,
    w: torch.Tensor,
    y: torch.Tensor | None = None,
    config: str | None = None,
) -> torch.Tensor:
    if y is None:
        return torch.empty(
            (x.shape[0], w.shape[1]), dtype=torch.float32, device=x.device
        )
    return y


@torch_compile_guard(gen_fake=gemm_a16w16x3_fake_tensor)
def gemm_a16w16x3_(
    x: torch.Tensor,
    w: torch.Tensor,
    y: torch.Tensor | None = None,
    config: str | None = None,
) -> torch.Tensor:
    """
    Computes Y = X @ W^T in fp32 for an fp32 weight W held as three bf16 terms.

    One launch adds the three terms in one fp32 accumulator, with bf16 MFMA. The
    result matches an fp32 GEMM of the bf16 X and the fp32 weight up to fp32 rounding.
    It does not depend on the order in which split-K workgroups finish.

    Args:
        x (torch.Tensor): BF16 input with shape (M, K).
        w (torch.Tensor): BF16 weight terms with shape (3, N, K), from split_bf16x3().
        y (Optional[torch.Tensor]): Pre-allocated FP32 output with shape (M, N).
        config (Optional[str]): Kernel tuning parameters (BLOCK_SIZE_M, BLOCK_SIZE_N,
            BLOCK_SIZE_K, GROUP_SIZE_M, NUM_KSPLIT, cache_modifier, num_warps,
            num_stages, waves_per_eu, matrix_instr_nonkdim).

    Returns:
        y (torch.Tensor): FP32 output with shape (M, N).

    Limits: x and w must be bf16 and y fp32. Split-K launches use counters of the
    current stream, so a captured graph must not run on two streams at the same time.
    """
    _LOGGER.info(
        "GEMM_A16W16X3: x.shape=%s, w.shape=%s", tuple(x.shape), tuple(w.shape)
    )
    assert x.dtype == torch.bfloat16 and w.dtype == torch.bfloat16, "bf16 x and w"
    assert w.dim() == 3 and w.shape[0] == 3, "w must be (3, N, K)"
    assert x.dim() == 2 and x.shape[1] == w.shape[2], "x must be (M, K)"

    M, K = x.shape
    N = w.shape[1]

    if config is None:
        config, _ = _get_config(M, N, K)
    else:
        config = deserialize_str(config)
    compute_splitk_params(config, K)

    if y is None:
        y = torch.empty((M, N), dtype=torch.float32, device=x.device)
    assert y.dtype == torch.float32 and y.shape == (M, N), "fp32 (M, N) y"

    num_tiles = triton.cdiv(M, config["BLOCK_SIZE_M"]) * triton.cdiv(
        N, config["BLOCK_SIZE_N"]
    )
    counters = _tile_counters(x.device, num_tiles)
    if config["NUM_KSPLIT"] > 1:
        workspace = torch.empty(
            (config["NUM_KSPLIT"], M, N), dtype=torch.float32, device=x.device
        )
    else:
        workspace = y

    grid = (num_tiles * config["NUM_KSPLIT"],)
    _gemm_a16w16x3_kernel[grid](
        x,
        w,
        y,
        workspace,
        counters,
        M,
        N,
        K,
        x.stride(0),
        x.stride(1),
        w.stride(0),
        w.stride(1),
        w.stride(2),
        y.stride(0),
        y.stride(1),
        **config,
        NUM_XCDS=get_num_xcds(),
    )

    return y


def gemm_a16w16x3(
    x: torch.Tensor,
    w: torch.Tensor,
    y: torch.Tensor | None = None,
    config: dict | None = None,
) -> torch.Tensor:
    config_hashable = serialize_dict(config) if config else None
    return gemm_a16w16x3_(x, w, y, config_hashable)
