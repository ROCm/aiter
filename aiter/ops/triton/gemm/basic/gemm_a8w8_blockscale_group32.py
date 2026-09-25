# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import torch
import triton

from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops.triton._triton_kernels.common.splitk_reduce import (
    _gemm_splitk_reduce_kernel,
)
from aiter.ops.triton._triton_kernels.gemm.basic.gemm_a8w8_blockscale_group32 import (
    _gemm_a8w8_blockscale_group32_kernel,
    _gemm_a8w8_blockscale_group32_packed_kernel,
    _get_config,
)
from aiter.utility.graph_alloc import ROUTES_INSIDE_CAPTURE, persistent_alloc

# One arrival counter per output tile of a fused split-K launch.
_SPLIT_COUNTERS = 1 << 16
_split_counters_by_stream: dict[tuple[torch.device, int], torch.Tensor] = {}


def _split_counters(device: torch.device) -> torch.Tensor | None:
    """Zeroed arrival counters per (device, stream), as for the a16w16 split-K
    semaphores; kernels re-zero what they use. None when a stream's first use
    falls inside a capture that cannot allocate outside the graph pool."""
    key = (device, torch.cuda.current_stream(device).cuda_stream)
    counters = _split_counters_by_stream.get(key)
    if counters is None:
        if not ROUTES_INSIDE_CAPTURE and torch.cuda.is_current_stream_capturing():
            return None
        with persistent_alloc(device):
            counters = torch.zeros(_SPLIT_COUNTERS, dtype=torch.int32, device=device)
        _split_counters_by_stream[key] = counters
    return counters


def _split_k_partition(k: int, splits: int, step: int) -> tuple[int, int]:
    """Partition size rounded up to whole steps, and the non-empty count."""
    part = -(-k // splits)
    size = -(-part // step) * step
    return size, -(-k // size)


def _fused_workspace(y, tiles, splits, block_m, block_n):
    """(grid, partials, counters) for fused split-K, or None to run unfused."""
    assert tiles <= _SPLIT_COUNTERS, "Too many output tiles for fused split-K"
    counters = _split_counters(y.device)
    if counters is None:
        return None
    partials = torch.empty(
        tiles * splits * block_m * block_n, dtype=torch.float32, device=y.device
    )
    return (-(-tiles // 8) * 8 * splits,), partials, counters


def gemm_a8w8_blockscale_group32(
    x: torch.Tensor,
    w: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    dtype: torch.dtype = torch.bfloat16,
    y: torch.Tensor | None = None,
    weight_group_rows: int = 32,
    split_k: int | None = None,
    config: dict | None = None,
) -> torch.Tensor:
    """Compute X @ W.T using native E4M3 operands and E8M0 group32 scales.

    x and w are contiguous (M, K) and (N, K) matrices. Activation scales are
    row-major (M, K // 32); compact weight scales are
    (ceil(N / weight_group_rows), K // 32), with weight_group_rows 1 or 32.
    E8M0 and E4M3 tensors may also be passed as their uint8 views. Weights and
    scales are neither preshuffled nor expanded.

    Accumulation is FP32, with BF16, FP16 or FP32 output. y optionally supplies
    a contiguous (M, N) output buffer. split_k is a positive partition count;
    None uses NUM_KSPLIT and the optional packed-K variant from the tuning table.
    config optionally overrides the table, following the Triton GEMM interface.
    Explicit partitions are rounded to whole K tiles, then empty partitions
    are removed. Split-K uses the shared AITER reduction without atomics;
    a table FUSED_SPLITK (or packed NUM_KSPLIT) instead sums each tile's
    partitions within the launch, on one XCD. A table cache_modifier applies
    to the weight loads.

    This backend requires gfx950 microscaling MFMA. Small M is limited by
    weight bandwidth and occupancy; packing K panels fills MFMA rows without
    a global partial buffer. Larger M uses output tiling and activation reuse.
    """
    assert x.ndim == w.ndim == 2, "Expected two matrix operands"
    M, K = x.shape
    N, weight_k = w.shape
    assert (
        K > 0 and K % 32 == 0 and weight_k == K and N > 0
    ), "Expected matching positive K divisible by 32 and positive N"
    assert weight_group_rows in (1, 32), "Weight scales must be 1x32 or 32x32"
    assert x.dtype in (torch.float8_e4m3fn, torch.uint8), "x must be E4M3"
    assert w.dtype in (torch.float8_e4m3fn, torch.uint8), "w must be E4M3"
    assert x_scale.dtype in (torch.float8_e8m0fnu, torch.uint8), "x_scale must be E8M0"
    assert w_scale.dtype in (torch.float8_e8m0fnu, torch.uint8), "w_scale must be E8M0"
    assert x_scale.shape == (M, K // 32), "Invalid activation scale shape"
    assert w_scale.shape == (
        -(-N // weight_group_rows),
        K // 32,
    ), "Invalid weight scale shape"
    assert all(
        t.is_cuda and t.device == x.device and t.is_contiguous()
        for t in (x, w, x_scale, w_scale)
    ), "Operands must be contiguous on the same GPU"
    assert dtype in (
        torch.bfloat16,
        torch.float16,
        torch.float32,
    ), "Output must be BF16, FP16 or FP32"
    assert split_k is None or (
        type(split_k) is int and split_k > 0
    ), "split_k must be a positive integer or None"
    if y is None:
        y = torch.empty((M, N), dtype=dtype, device=x.device)
    else:
        assert (
            y.shape == (M, N) and y.dtype == dtype and y.device == x.device
        ), "Invalid output shape, dtype or device"
        assert y.is_contiguous(), "Output must be contiguous"
    if M == 0:
        return y
    assert get_gfx_runtime() == "gfx950", "Group32 FP8 GEMM requires gfx950"

    x = x.view(torch.float8_e4m3fn) if x.dtype == torch.uint8 else x
    w = w.view(torch.float8_e4m3fn) if w.dtype == torch.uint8 else w
    x_scale = x_scale.view(torch.uint8)
    w_scale = w_scale.view(torch.uint8)
    if config is None:
        config, _ = _get_config(M, N, K)
    packed = (
        config.get("packed") if weight_group_rows == 32 and split_k is None else None
    )
    launch_config = config if packed is None else packed
    launch_options = {
        key: launch_config[key]
        for key in ("num_warps", "num_stages", "waves_per_eu", "matrix_instr_nonkdim")
    }
    # Triton's repr callback sees constexpr arguments, not compiler options.
    # Carry the same values into the name without a second source of tuning.
    launch_repr = tuple(launch_options.items())
    if packed is not None:
        block_m, block_n = packed["BLOCK_SIZE_M"], packed["BLOCK_SIZE_N"]
        block_k, k_pack = packed["BLOCK_SIZE_K"], packed["K_PACK"]
        grid_m, grid_n = -(-M // block_m), -(-N // block_n)
        split_k_size, num_splits = _split_k_partition(
            K, packed.get("NUM_KSPLIT", 1), block_k * k_pack
        )
        workspace = None
        if num_splits > 1:
            workspace = _fused_workspace(
                y, grid_m * grid_n, num_splits, block_m, block_n
            )
        if workspace is None:
            workspace = (grid_m, grid_n), y, y
            split_k_size, num_splits = 0, 1
        grid, partials, counters = workspace
        _gemm_a8w8_blockscale_group32_packed_kernel[grid](
            x,
            w,
            x_scale,
            w_scale,
            y,
            partials,
            counters,
            M,
            N,
            K,
            block_m,
            block_n,
            block_k,
            k_pack,
            LAUNCH_OPTIONS=launch_repr,
            SPLITK_BLOCK_SIZE=split_k_size,
            FUSED_SPLITS=num_splits,
            B_CACHE_MODIFIER=packed.get("cache_modifier"),
            **launch_options,
        )
        return y

    block_m = config["BLOCK_SIZE_M"]
    block_n = config["BLOCK_SIZE_N"]
    block_k = config["BLOCK_SIZE_K"]
    n_first = config["N_FIRST"]
    grid_m, grid_n = -(-M // block_m), -(-N // block_n)
    fused = split_k is None and config.get("FUSED_SPLITK", False)
    split_k = config["NUM_KSPLIT"] if split_k is None else split_k
    split_k_size, num_splits = _split_k_partition(K, split_k, block_k)
    workspace = None
    if fused and num_splits > 1:
        workspace = _fused_workspace(y, grid_m * grid_n, num_splits, block_m, block_n)
    if workspace is not None:
        grid, partials, counters = workspace
        out, fused_splits = y, num_splits
    else:
        out = (
            y
            if num_splits == 1
            else torch.empty((num_splits, M, N), dtype=torch.float32, device=x.device)
        )
        grid = (grid_n, grid_m, num_splits) if n_first else (grid_m, grid_n, num_splits)
        partials = counters = out
        fused_splits = 1
    _gemm_a8w8_blockscale_group32_kernel[grid](
        x,
        w,
        x_scale,
        w_scale,
        out,
        partials,
        counters,
        M,
        N,
        K,
        weight_group_rows,
        split_k_size,
        block_m,
        block_n,
        block_k,
        N_FIRST=n_first,
        LAUNCH_OPTIONS=launch_repr,
        FUSED_SPLITS=fused_splits,
        B_CACHE_MODIFIER=config.get("cache_modifier"),
        **launch_options,
    )
    if fused_splits == 1 and num_splits > 1:
        reduce_m = config["REDUCE_BLOCK_SIZE_M"]
        reduce_n = config["REDUCE_BLOCK_SIZE_N"]
        _gemm_splitk_reduce_kernel[(-(-M // reduce_m), -(-N // reduce_n))](
            out,
            y,
            None,
            M,
            N,
            M * N,
            N,
            1,
            N,
            1,
            BLOCK_SIZE_M=reduce_m,
            BLOCK_SIZE_N=reduce_n,
            ACTUAL_KSPLIT=num_splits,
            MAX_KSPLIT=triton.next_power_of_2(num_splits),
            ADD_BIAS=False,
            activation=None,
            use_activation=False,
            KERNEL_NAME="_gemm_a8w8_blockscale_group32_reduce_kernel",
        )
    return y
