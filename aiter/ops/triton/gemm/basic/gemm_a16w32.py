# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import torch
import triton

from aiter.jit.utils.torch_guard import torch_compile_guard
from aiter.ops.triton._triton_kernels.gemm.basic.gemm_a16w32 import (
    _gemm_a16w32_bf16x3_kernel,
    _gemm_a16w32_fma_kernel,
    _get_config,
)
from aiter.ops.triton.utils.common_utils import deserialize_str, serialize_dict
from aiter.ops.triton.utils.device_info import get_num_xcds
from aiter.ops.triton.utils.gemm_config_utils import compute_splitk_params
from aiter.ops.triton.utils.logger import AiterTritonLogger
from aiter.utility.graph_alloc import capture_scoped_cache, persistent_alloc

_LOGGER = AiterTritonLogger()

# One split-K arrival counter for each output tile.
_SPLIT_COUNTERS = 1 << 16


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


@capture_scoped_cache
def _split_counters(device: torch.device, stream_id: int) -> torch.Tensor:
    # Zeroed counters for each (device, stream). The kernel resets the ones it uses.
    with persistent_alloc(device):
        return torch.zeros(_SPLIT_COUNTERS, dtype=torch.int32, device=device)


def gemm_a16w32_fake_tensor(
    x: torch.Tensor,
    w: torch.Tensor,
    w_bf16x3: torch.Tensor,
    y: torch.Tensor | None = None,
    config: str | None = None,
) -> torch.Tensor:
    if y is None:
        return torch.empty(
            (x.shape[0], w.shape[0]), dtype=torch.float32, device=x.device
        )
    return y


def _gemm_fma(x, w, y, config):
    M, K = x.shape
    N = w.shape[0]
    grid = (triton.cdiv(M, config["BLOCK_SIZE_M"]) * N,)
    _gemm_a16w32_fma_kernel[grid](
        x,
        w,
        y,
        M,
        N,
        K,
        x.stride(0),
        x.stride(1),
        w.stride(0),
        w.stride(1),
        y.stride(0),
        y.stride(1),
        BLOCK_SIZE_M=config["BLOCK_SIZE_M"],
        BLOCK_SIZE_K=config["BLOCK_SIZE_K"],
        cache_modifier=config["cache_modifier"],
        num_warps=config["num_warps"],
        num_stages=config["num_stages"],
        waves_per_eu=config["waves_per_eu"],
    )


def _gemm_bf16x3(x, w_bf16x3, y, config):
    M, K = x.shape
    N = w_bf16x3.shape[1]
    compute_splitk_params(config, K)
    num_tiles = triton.cdiv(M, config["BLOCK_SIZE_M"]) * triton.cdiv(
        N, config["BLOCK_SIZE_N"]
    )
    workspace, counters = None, None
    if config["NUM_KSPLIT"] > 1:
        assert num_tiles <= _SPLIT_COUNTERS, "too many output tiles for split-K"
        workspace = torch.empty(
            (config["NUM_KSPLIT"], M, N), dtype=torch.float32, device=x.device
        )
        stream_id = torch.cuda.current_stream(x.device).cuda_stream
        counters = _split_counters(x.device, stream_id)

    grid = (num_tiles * config["NUM_KSPLIT"],)
    _gemm_a16w32_bf16x3_kernel[grid](
        x,
        w_bf16x3,
        y,
        workspace,
        counters,
        M,
        N,
        K,
        x.stride(0),
        x.stride(1),
        w_bf16x3.stride(0),
        w_bf16x3.stride(1),
        w_bf16x3.stride(2),
        y.stride(0),
        y.stride(1),
        **config,
        NUM_XCDS=get_num_xcds(),
    )


@torch_compile_guard(gen_fake=gemm_a16w32_fake_tensor)
def gemm_a16w32_(
    x: torch.Tensor,
    w: torch.Tensor,
    w_bf16x3: torch.Tensor,
    y: torch.Tensor | None = None,
    config: str | None = None,
) -> torch.Tensor:
    """
    Computes Y = X @ W^T in fp32 for a bf16 X and an fp32 weight W.

    The config of each M bucket selects one of two kernels with ``kernel_type``:

    - ``"bandwidth_bound"`` (small M): fp32 FMA on W, one row of W in each
      program (``BLOCK_SIZE_N`` is 1). It reads 4 bytes for each weight value.
    - ``"compute_bound"``: bf16 MFMA on the three bf16 terms of W. Each K step
      loads the X tile one time and adds the three dots into one fp32
      accumulator. Split-K is in the kernel. The last split of a tile adds the
      partials in split order.

    Both kernels match an fp32 GEMM of X and W up to fp32 rounding. The result
    does not depend on the order in which workgroups finish.

    Args:
        x (torch.Tensor): BF16 input with shape (M, K).
        w (torch.Tensor): FP32 weight with shape (N, K).
        w_bf16x3 (torch.Tensor): BF16 terms of w with shape (3, N, K), from
            split_bf16x3(w).
        y (Optional[torch.Tensor]): Pre-allocated FP32 output with shape (M, N).
        config (Optional[str]): Kernel tuning parameters (kernel_type, BLOCK_SIZE_M,
            BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE_M, NUM_KSPLIT, cache_modifier,
            num_warps, num_stages, waves_per_eu, matrix_instr_nonkdim).

    Returns:
        y (torch.Tensor): FP32 output with shape (M, N).

    Limits: configs exist for gfx950 only. Split-K launches use counters of the
    current stream, so a captured graph must not run on two streams at the same
    time. With torch < 2.10, run one eager split-K call on a stream before a graph
    capture on that stream.
    """
    _LOGGER.info("GEMM_A16W32: x.shape=%s, w.shape=%s", tuple(x.shape), tuple(w.shape))
    assert x.dtype == torch.bfloat16 and x.dim() == 2, "bf16 (M, K) x"
    assert w.dtype == torch.float32 and w.dim() == 2, "fp32 (N, K) w"
    assert w_bf16x3.dtype == torch.bfloat16, "bf16 w_bf16x3"
    assert w_bf16x3.shape == (3, *w.shape), "w_bf16x3 must be (3, N, K)"
    assert x.shape[1] == w.shape[1], "x and w must have the same K"

    M, K = x.shape
    N = w.shape[0]

    if config is None:
        config, _ = _get_config(M, N, K)
    else:
        config = deserialize_str(config)
    kernel_type = config.pop("kernel_type", "compute_bound")
    assert kernel_type in ("bandwidth_bound", "compute_bound"), kernel_type

    if y is None:
        y = torch.empty((M, N), dtype=torch.float32, device=x.device)
    assert y.dtype == torch.float32 and y.shape == (M, N), "fp32 (M, N) y"

    if kernel_type == "bandwidth_bound":
        _gemm_fma(x, w, y, config)
    else:
        _gemm_bf16x3(x, w_bf16x3, y, config)
    return y


def gemm_a16w32(
    x: torch.Tensor,
    w: torch.Tensor,
    w_bf16x3: torch.Tensor,
    y: torch.Tensor | None = None,
    config: dict | None = None,
) -> torch.Tensor:
    """
    Computes Y = X @ W^T in fp32 for a bf16 X and an fp32 weight W.

    See ``gemm_a16w32_`` for the arguments. Here ``config`` is a dict. The
    wrapper serializes it, so that torch.compile can trace the op.
    """
    config_hashable = serialize_dict(config) if config else None
    return gemm_a16w32_(x, w, w_bf16x3, y, config_hashable)
