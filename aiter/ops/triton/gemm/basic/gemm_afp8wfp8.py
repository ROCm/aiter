# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.


import math

import torch
import triton

from aiter.ops.triton._triton_kernels.common.splitk_reduce import (
    _gemm_splitk_reduce_kernel,
)
from aiter.ops.triton._triton_kernels.gemm.basic.gemm_afp8wfp8 import (
    _gemm_afp8wfp8_kernel,
    _gemm_afp8wfp8_preshuffle_kernel,
    _get_config,
)
from aiter.ops.triton.utils._triton.arch_info import get_arch
from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER = AiterTritonLogger()

_GLUON_SUPPORTED_ARCHS = ("gfx1250",)


def _is_gluon_available():
    """Check if the gluon backend is available for the current GPU architecture."""
    try:
        return any(supported in get_arch() for supported in _GLUON_SUPPORTED_ARCHS)
    except Exception:  # noqa: BLE001
        return False


def _resolve_x_scale_strides(
    x_scales: torch.Tensor,
    M: int,
    K: int,
    x_scale_group_size: int,
    is_x_scale_transposed: bool,
) -> tuple[int, int]:
    """Validate the activation-scale buffer and return its (row, group) strides.

    ``x_scale_group_size`` is how many K elements share one e8m0 byte: 32 for MX
    activations, 128 for blockscale activations (what aiter's per-group quant
    emits for ATOM's per_1x128 path).

    ``is_x_scale_transposed`` means the buffer still reads ``(M, K // group)``
    through ``.shape`` but its bytes are laid out column-major -- logically
    ``(K // group, M)``. That is what ``per_group_quant_hip(transpose_scale=True)``
    produces for group sizes other than 32. Folding it into strides here keeps
    both the triton and gluon kernels layout-agnostic.
    """
    assert x_scale_group_size in (
        32,
        128,
    ), f"x_scale_group_size must be 32 (MX) or 128 (blockscale), got {x_scale_group_size}"
    assert (
        K % x_scale_group_size == 0
    ), f"K={K} must be divisible by x_scale_group_size={x_scale_group_size}"

    expected = (M, K // x_scale_group_size)
    assert tuple(x_scales.shape[-2:]) == expected, (
        f"x_scales must have shape {expected} for x_scale_group_size="
        f"{x_scale_group_size}, got {tuple(x_scales.shape)}"
    )

    if is_x_scale_transposed:
        return x_scales.stride(1), x_scales.numel() // x_scales.stride(0)
    return x_scales.stride(0), x_scales.stride(1)


def gemm_afp8wfp8(
    x: torch.Tensor,
    w: torch.Tensor,
    x_scales: torch.Tensor,
    w_scales: torch.Tensor,
    dtype: torch.dtype | None = torch.bfloat16,
    y: torch.Tensor | None = None,
    config: dict | None = None,
    skip_reduce: bool | None = False,
    x_scale_group_size: int = 128,
    is_x_scale_transposed: bool = False,
) -> torch.Tensor:
    """
    Computes matrix multiplication Y = X @ W^T with FP8 activations and FP8
    weights (e8m0 act scales, 128x128 e8m0 weight scales).

    Args:
        x: FP8 e4m3 (or uint8 view) input matrix with shape (M, K).
        w: FP8 e4m3 (or uint8 view) weight matrix with shape (N, K) — internally
           transposed to (K, N) before the kernel call.
        x_scales: e8m0 (uint8) per-group scale for x with shape
           (M, K // x_scale_group_size).
        w_scales: e8m0 (uint8) per-block scale for w with shape (N // 128, K // 128).
        dtype: Output dtype (BF16 or FP16). Default bf16.
        y: Optional pre-allocated output tensor with shape (M, N).
        config: Optional kernel-tuning dict. If None uses defaults.
        x_scale_group_size: K elements per activation scale — 128 for blockscale
           activations (default), 32 for MX activations.
        is_x_scale_transposed: x_scales bytes are column-major, i.e. logically
           (K // group, M). Default False (row-major).

    Returns:
        torch.Tensor: Output with shape (M, N).
    """
    M, K = x.shape
    N, K_w = w.shape
    assert K == K_w, f"K mismatch: x has K={K}, w has K={K_w}"
    stride_asm, stride_ask = _resolve_x_scale_strides(
        x_scales, M, K, x_scale_group_size, is_x_scale_transposed
    )

    # Transpose w to (K, N) for the kernel.
    w_t = w.T

    # tl.dot_scaled with format "e4m3" expects uint8-typed operands; reinterpret
    # the FP8 buffers as uint8 (bit-identical view).
    if x.dtype != torch.uint8:
        x = x.view(torch.uint8)
    if w_t.dtype != torch.uint8:
        w_t = w_t.view(torch.uint8)

    if config is None:
        config, _ = _get_config(M, N, K)

    if y is None and (config["NUM_KSPLIT"] == 1 or not skip_reduce):
        y = torch.empty((M, N), dtype=dtype, device=x.device)

    config["SPLITK_BLOCK_SIZE"] = triton.cdiv(
        K, config["NUM_KSPLIT"]
    )  # How big each split_k partition is
    if config["NUM_KSPLIT"] > 1:
        y_pp = torch.empty(
            (config["NUM_KSPLIT"], M, N),
            dtype=torch.float32,
            device=x.device,
        )
    else:
        y_pp = None

    grid = lambda META: (
        (
            META["NUM_KSPLIT"]
            * triton.cdiv(M, META["BLOCK_SIZE_M"])
            * triton.cdiv(N, META["BLOCK_SIZE_N"])
        ),
    )

    _gemm_afp8wfp8_kernel[grid](
        x,
        w_t,
        y if config["NUM_KSPLIT"] == 1 else y_pp,
        x_scales,
        w_scales,
        M,
        N,
        K,
        x.stride(0),
        x.stride(1),
        w_t.stride(0),
        w_t.stride(1),
        0 if config["NUM_KSPLIT"] == 1 else y_pp.stride(0),
        y.stride(0) if config["NUM_KSPLIT"] == 1 else y_pp.stride(1),
        y.stride(1) if config["NUM_KSPLIT"] == 1 else y_pp.stride(2),
        stride_asm,
        stride_ask,
        w_scales.stride(0),
        w_scales.stride(1),
        A_SCALE_K_GROUP=x_scale_group_size,
        **config,
    )

    if config["NUM_KSPLIT"] > 1:
        if skip_reduce:
            return y_pp

        REDUCE_BLOCK_SIZE_M = 32
        REDUCE_BLOCK_SIZE_N = 32
        ACTUAL_KSPLIT = triton.cdiv(K, config["SPLITK_BLOCK_SIZE"])

        grid_reduce = (
            triton.cdiv(M, REDUCE_BLOCK_SIZE_M),
            triton.cdiv(N, REDUCE_BLOCK_SIZE_N),
        )
        _gemm_splitk_reduce_kernel[grid_reduce](
            y_pp,
            y,
            None,
            M,
            N,
            y_pp.stride(0),
            y_pp.stride(1),
            y_pp.stride(2),
            y.stride(0),
            y.stride(1),
            BLOCK_SIZE_M=REDUCE_BLOCK_SIZE_M,
            BLOCK_SIZE_N=REDUCE_BLOCK_SIZE_N,
            ACTUAL_KSPLIT=ACTUAL_KSPLIT,
            MAX_KSPLIT=triton.next_power_of_2(config["NUM_KSPLIT"]),
            ADD_BIAS=False,
            activation=None,
            use_activation=False,
            KERNEL_NAME="_gemm_afp8wfp8_reduce_kernel",
        )

    return y


def gemm_afp8wfp8_preshuffle(
    x: torch.Tensor,
    w_shuffled: torch.Tensor,
    x_scales: torch.Tensor,
    w_scales: torch.Tensor,
    dtype: torch.dtype | None = torch.bfloat16,
    y: torch.Tensor | None = None,
    config: dict | None = None,
    skip_reduce: bool | None = False,
    x_scale_group_size: int = 128,
    is_x_scale_transposed: bool = False,
    kernel_type: str = "bandwidth_bound",
    backend: str | None = None,
) -> torch.Tensor:
    """
    Preshuffle variant of gemm_afp8wfp8. The weight tensor has already been
    permuted via aiter.ops.shuffle.shuffle_weight(..., layout=(16, 16)). Scales
    are left unshuffled in the compact 128x128 layout.

    Uses the gluon backend automatically on supported architectures (gfx1250)
    and the triton backend everywhere else. Pass ``backend`` to force a choice.

    Args:
        x: FP8 e4m3 activations with shape (M, K).
        w_shuffled: FP8 e4m3 weights, shuffled in place to (N, K) storage
            (same total bytes; bytes rearranged for the kernel's read pattern).
        x_scales: e8m0 (uint8) per-group scale with shape
            (M, K // x_scale_group_size).
        w_scales: e8m0 (uint8) per-block weight scale with shape (N // 128, K // 128).
        dtype: Output dtype.
        y: Optional pre-allocated output (M, N).
        config: Optional kernel-tuning dict.
        x_scale_group_size: K elements per activation scale — 128 for blockscale
            activations (default), 32 for MX activations.
        is_x_scale_transposed: x_scales bytes are column-major, i.e. logically
            (K // group, M). Default False (row-major).
        kernel_type: [gluon only] Kernel variant. Only "bandwidth_bound" exists
            so far.
        backend: "triton", "gluon", or None (auto-detect).

    Returns:
        torch.Tensor: Output with shape (M, N).
    """
    M, K = x.shape
    N, K_w = w_shuffled.shape
    assert K == K_w, f"K mismatch: x={K}, w={K_w}"
    assert N % 16 == 0, f"N must be divisible by 16 for preshuffle, got {N}"
    stride_asm, stride_ask = _resolve_x_scale_strides(
        x_scales, M, K, x_scale_group_size, is_x_scale_transposed
    )

    # The kernel expects to address the shuffled tensor as (N//16, K*16).
    w_view = w_shuffled.view(N // 16, K * 16)

    if x.dtype != torch.uint8:
        x = x.view(torch.uint8)
    if w_view.dtype != torch.uint8:
        w_view = w_view.view(torch.uint8)

    # Resolve the backend up-front so the config is loaded from the backend's
    # config dir (<arch>/<backend>/gemm/) -- the two kernels take different keys.
    if backend is None:
        backend = "gluon" if _is_gluon_available() else "triton"
    backend = backend.lower()
    assert backend in (
        "triton",
        "gluon",
    ), f"Unknown backend '{backend}', must be 'triton' or 'gluon'"
    if backend == "gluon":
        assert (
            _is_gluon_available()
        ), f"Gluon backend requires one of {_GLUON_SUPPORTED_ARCHS}, got '{get_arch()}'"

    if config is None:
        config, _ = _get_config(M, N, K, shuffle=True, backend=backend)

    # CTA-cluster (CGA) multicast, gluon only. CTAS_M x CTAS_N CTAs form one
    # cluster, and each operand fetch is multicast to every CTA in the cluster
    # that wants it.
    #
    # NOTE the convention flip that happens right here, because the two sides
    # disagree on what BLOCK_SIZE_M / BLOCK_SIZE_N mean:
    #
    #   in the JSON   -> the PER-CTA tile. A tuned config keeps its meaning when
    #                    CTAS changes, and the LDS/register budget stays
    #                    readable straight off the file.
    #   in the kernel -> the CLUSTER tile. gl.arange over BLOCK_SIZE_N then
    #                    shards across the cluster on its own, and the grid
    #                    lambda below counts clusters rather than CTAs (triton
    #                    multiplies the grid by num_ctas at launch).
    #
    # So `"BLOCK_SIZE_N": 256, "CTAS_N": 2` is 256 columns per CTA and a
    # 512-column cluster tile -- per-CTA LDS is unchanged from 1x1, which is the
    # whole point. Verified: LDS/CTA is 311536 B at 1x1 and 311792 B at both 1x2
    # and 2x2, rather than halving or quartering.
    if backend == "gluon":
        ctas_m, ctas_n = config["CTAS_M"], config["CTAS_N"]
        num_ctas = ctas_m * ctas_n
        config["BLOCK_SIZE_M"] *= ctas_m
        config["BLOCK_SIZE_N"] *= ctas_n
    else:
        ctas_m, ctas_n, num_ctas = 1, 1, 1

    if y is None and (config["NUM_KSPLIT"] == 1 or not skip_reduce):
        y = torch.empty((M, N), dtype=dtype, device=x.device)

    config["SPLITK_BLOCK_SIZE"] = triton.cdiv(
        K, config["NUM_KSPLIT"]
    )  # How big each split_k partition is
    if config["NUM_KSPLIT"] > 1:
        y_pp = torch.empty(
            (config["NUM_KSPLIT"], M, N),
            dtype=torch.float32,
            device=x.device,
        )
    else:
        y_pp = None

    grid = lambda META: (
        (
            META["NUM_KSPLIT"]
            * triton.cdiv(M, META["BLOCK_SIZE_M"])
            * triton.cdiv(N, META["BLOCK_SIZE_N"])
        ),
    )
    if backend == "gluon":
        from aiter.ops.triton._gluon_kernels.gfx1250.gemm.basic.gemm_mxfp8 import (
            _PRESHUFFLE_KERNEL_MAP,
        )

        kernel_type = config.pop("kernel_type", kernel_type)
        assert kernel_type in _PRESHUFFLE_KERNEL_MAP, (
            f"Unknown kernel_type '{kernel_type}', must be one of "
            f"{list(_PRESHUFFLE_KERNEL_MAP.keys())}"
        )
        assert (
            num_ctas == 1 or kernel_type == "bandwidth_bound"
        ), f"CGA multicast is only wired into bandwidth_bound, got '{kernel_type}'"
        _LOGGER.info(
            f"GEMM_AFP8WFP8 PRESHUFFLE [gluon/gfx1250]: x={tuple(x.shape)} "
            f"w={tuple(w_view.shape)} kernel={kernel_type}"
        )

        # Shape-derived clamp, not a tuning default: the pipeline computes
        # NUM_BUFFERS - 1 tiles outside the main loop, so the depth cannot
        # exceed the K-tile count of a single split.
        num_k_iter = triton.cdiv(config["SPLITK_BLOCK_SIZE"], config["BLOCK_SIZE_K"])
        num_buffers = max(2, min(config["NUM_BUFFERS"], num_k_iter + 1))

        # warp_bases mirrors the a8w8 blockscale gluon wrapper: warp 0 walks N,
        # the remaining log2(num_warps // 2) warps walk M.
        warp_bases = [(0, 1)]
        for i in range(int(math.log2(config["num_warps"] // 2))):
            warp_bases.append((1 << i, 0))

        _PRESHUFFLE_KERNEL_MAP[kernel_type][grid](
            x,
            w_view,
            y if config["NUM_KSPLIT"] == 1 else y_pp,
            x_scales,
            w_scales,
            M,
            N,
            K,
            x.stride(0),
            x.stride(1),
            w_view.stride(0),
            w_view.stride(1),
            0 if config["NUM_KSPLIT"] == 1 else y_pp.stride(0),
            y.stride(0) if config["NUM_KSPLIT"] == 1 else y_pp.stride(1),
            y.stride(1) if config["NUM_KSPLIT"] == 1 else y_pp.stride(2),
            stride_asm,
            stride_ask,
            w_scales.stride(0),
            w_scales.stride(1),
            BLOCK_SIZE_M=config["BLOCK_SIZE_M"],
            BLOCK_SIZE_N=config["BLOCK_SIZE_N"],
            BLOCK_SIZE_K=config["BLOCK_SIZE_K"],
            GROUP_SIZE_M=config["GROUP_SIZE_M"],
            A_SCALE_K_GROUP=x_scale_group_size,
            # Not foldable into strides like the rest of the layout: staging the
            # scales in LDS needs a TDM descriptor whose innermost stride is 1,
            # so the kernel has to know which axis is contiguous.
            A_SCALE_TRANSPOSED=is_x_scale_transposed,
            NUM_KSPLIT=config["NUM_KSPLIT"],
            SPLITK_BLOCK_SIZE=config["SPLITK_BLOCK_SIZE"],
            num_warps=config["num_warps"],
            warp_bases=tuple(warp_bases),
            cache_modifier=config["cache_modifier"],
            NUM_BUFFERS=num_buffers,
            # Every gluon preshuffle config declares this; configs/CLAUDE.md
            # forbids Python-side defaults for tuning values, so a missing key
            # is a config bug and should fail loudly rather than silently tune
            # itself off. The kernel clamps it to the main-loop trip count.
            LOOP_UNROLL_FACTOR=config["LOOP_UNROLL_FACTOR"],
            # Which B-scale fill to use. Tuned per shape: the TDM fill drops the
            # register staging and a barrier, but the global pre-load streams
            # better on some shapes. Both fill the same compact slab and both
            # work at any BLOCK_SIZE_N.
            B_SCALE_TDM=config["B_SCALE_TDM"],
            # Not read by the kernel: triton forwards it to the AMD backend as
            # the amdgpu-waves-per-eu occupancy hint. 0 emits no attribute.
            waves_per_eu=config["waves_per_eu"],
            CTAS_M=ctas_m,
            CTAS_N=ctas_n,
            # Reserved launch option (sets the cluster dim) that the kernel also
            # declares, so it is bound both ways -- like waves_per_eu above.
            num_ctas=num_ctas,
        )
    else:
        _gemm_afp8wfp8_preshuffle_kernel[grid](
            x,
            w_view,
            y if config["NUM_KSPLIT"] == 1 else y_pp,
            x_scales,
            w_scales,
            M,
            N,
            K,
            x.stride(0),
            x.stride(1),
            w_view.stride(0),
            w_view.stride(1),
            0 if config["NUM_KSPLIT"] == 1 else y_pp.stride(0),
            y.stride(0) if config["NUM_KSPLIT"] == 1 else y_pp.stride(1),
            y.stride(1) if config["NUM_KSPLIT"] == 1 else y_pp.stride(2),
            stride_asm,
            stride_ask,
            w_scales.stride(0),
            w_scales.stride(1),
            A_SCALE_K_GROUP=x_scale_group_size,
            **config,
        )

    if config["NUM_KSPLIT"] > 1:
        if skip_reduce:
            return y_pp

        REDUCE_BLOCK_SIZE_M = 32
        REDUCE_BLOCK_SIZE_N = 32
        ACTUAL_KSPLIT = triton.cdiv(K, config["SPLITK_BLOCK_SIZE"])

        grid_reduce = (
            triton.cdiv(M, REDUCE_BLOCK_SIZE_M),
            triton.cdiv(N, REDUCE_BLOCK_SIZE_N),
        )
        _gemm_splitk_reduce_kernel[grid_reduce](
            y_pp,
            y,
            None,
            M,
            N,
            y_pp.stride(0),
            y_pp.stride(1),
            y_pp.stride(2),
            y.stride(0),
            y.stride(1),
            BLOCK_SIZE_M=REDUCE_BLOCK_SIZE_M,
            BLOCK_SIZE_N=REDUCE_BLOCK_SIZE_N,
            ACTUAL_KSPLIT=ACTUAL_KSPLIT,
            MAX_KSPLIT=triton.next_power_of_2(config["NUM_KSPLIT"]),
            ADD_BIAS=False,
            activation=None,
            use_activation=False,
            KERNEL_NAME="_gemm_afp8wfp8_preshuffle_reduce_kernel",
        )

    return y
