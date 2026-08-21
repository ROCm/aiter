# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import functools
import math
import os

import torch
import triton
from packaging.version import Version

from aiter.ops.triton._triton_kernels.common.splitk_reduce import (
    _gemm_splitk_reduce_kernel,
)
from aiter.ops.triton._triton_kernels.gemm.basic.gemm_a8w8_blockscale import (
    _gemm_a8w8_blockscale_kernel as triton_gemm_a8w8_blockscale_kernel,
)
from aiter.ops.triton._triton_kernels.gemm.basic.gemm_a8w8_blockscale import (
    _gemm_a8w8_blockscale_preshuffle_kernel as triton_gemm_a8w8_blockscale_preshuffle_kernel,
)
from aiter.ops.triton._triton_kernels.gemm.basic.gemm_a8w8_blockscale import (
    _get_config,
)
from aiter.ops.triton.utils._triton.arch_info import get_arch
from aiter.ops.triton.utils.core import AITER_TRITON_CONFIGS_PATH, load_config_json
from aiter.ops.triton.utils.gemm_config_utils import compute_splitk_params
from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER = AiterTritonLogger()
_FORCE_GFX1250_EX = os.environ.get("AITER_FORCE_GFX1250_EX", "0") == "1"
_TRITON_VERSION = Version(triton.__version__)

_GLUON_SUPPORTED_ARCHS = ("gfx950", "gfx1250")
_GLUON_PRESHUFFLE_ARCHS = ("gfx1250",)
_GLUON_DEFAULT_ARCHS = ("gfx1250",)


@functools.lru_cache(maxsize=1024)
def _get_gfx950_gluon_config_cached(M: int, N: int, K: int):
    from aiter.ops.triton._gluon_kernels.gfx950.gemm.basic.gemm_a8w8_blockscale import (
        _SUPPORTED_TILES,
    )

    dev = get_arch()

    # Try specialized config first.
    config_dict = load_config_json(
        f"{AITER_TRITON_CONFIGS_PATH}/gemm/gluon/{dev}-GEMM-A8W8_BLOCKSCALE-N={N}-K={K}.json",
        required=False,
    )
    # Fall back to the general config (must exist).
    if config_dict is None:
        config_dict = load_config_json(
            f"{AITER_TRITON_CONFIGS_PATH}/gemm/gluon/{dev}-GEMM-A8W8_BLOCKSCALE.json"
        )

    # Config keys should be named M_LEQ_<bound> or "any"
    bounds = []
    for setting in config_dict:
        potential_block_m = setting.replace("M_LEQ_", "")
        if potential_block_m.isnumeric():
            bounds.append(int(potential_block_m))

    # Walk buckets in ascending-M order; pick the smallest one whose tile
    # the kernel currently supports. Unsupported buckets are skipped (those
    # configs become live again once the kernel grows the corresponding
    # padded-LDS layouts), so we may fall through to "any".
    config = config_dict["any"]
    for bound in sorted(bounds):
        if M > bound or f"M_LEQ_{bound}" not in config_dict:
            continue
        candidate = config_dict[f"M_LEQ_{bound}"]
        if (candidate["BLOCK_SIZE_M"], candidate["BLOCK_SIZE_N"]) in _SUPPORTED_TILES:
            config = candidate
            break

    return config


def _get_gfx950_gluon_config(M: int, N: int, K: int):
    # Fresh copy per call, outside the lru boundary -- the caller writes
    # derived fields (SPLITK_BLOCK_SIZE here, GROUP_K/GROUP_N at the call
    # site) into the returned dict.
    config = _get_gfx950_gluon_config_cached(M, N, K).copy()

    block_size_k = config["BLOCK_SIZE_K"]
    num_k_blocks = triton.cdiv(K, block_size_k)
    num_k_blocks_per_split = triton.cdiv(num_k_blocks, config["NUM_KSPLIT"])
    config["SPLITK_BLOCK_SIZE"] = num_k_blocks_per_split * block_size_k

    return config


def _gemm_a8w8_blockscale_gluon_gfx950(
    x, w, x_scale, w_scale, M, N, K, dtype, y, config
):
    from aiter.ops.triton._gluon_kernels.gfx950.gemm.basic.gemm_a8w8_blockscale import (
        _gemm_a8w8_blockscale_kernel,
        _gemm_a8w8_blockscale_reduce_kernel,
    )

    _LOGGER.info(
        f"GEMM_A8W8 BLOCKSCALE [gluon/gfx950]: x={tuple(x.shape)} w={tuple(w.shape)}"
    )

    if y is None:
        y = torch.empty((M, N), dtype=dtype, device=x.device)

    if config is None:
        config = _get_gfx950_gluon_config(M, N, K)

    # Scale block sizes
    # TODO: need a better way to pass scale block sizes around
    config["GROUP_K"] = triton.next_power_of_2(triton.cdiv(K, w_scale.shape[0]))
    config["GROUP_N"] = triton.next_power_of_2(triton.cdiv(N, w_scale.shape[1]))

    if config["NUM_KSPLIT"] == 1:
        assert (
            config["GROUP_K"] == config["BLOCK_SIZE_K"]
        ), f"GROUP_K: {config['GROUP_K']} must equal BLOCK_SIZE_K: {config['BLOCK_SIZE_K']} when not using KSPLIT"

    if config["NUM_KSPLIT"] > 1:
        y_pp = torch.empty(
            (config["NUM_KSPLIT"], M, N), dtype=torch.float32, device=y.device
        )
    else:
        y_pp = None

    num_stages = config.get("num_stages", 2)
    num_stages = max(num_stages, 2)

    grid = lambda META: (
        (
            META["NUM_KSPLIT"]
            * triton.cdiv(M, META["BLOCK_SIZE_M"])
            * triton.cdiv(N, META["BLOCK_SIZE_N"])
        ),
    )
    _gemm_a8w8_blockscale_kernel[grid](
        x,
        w,
        y if config["NUM_KSPLIT"] == 1 else y_pp,
        x_scale,
        w_scale,
        M,
        N,
        K,
        x.stride(0),
        x.stride(1),
        w.stride(0),
        w.stride(1),
        0 if config["NUM_KSPLIT"] == 1 else y_pp.stride(0),
        y.stride(0) if config["NUM_KSPLIT"] == 1 else y_pp.stride(1),
        y.stride(1) if config["NUM_KSPLIT"] == 1 else y_pp.stride(2),
        x_scale.stride(0),
        x_scale.stride(1),
        w_scale.stride(0),
        w_scale.stride(1),
        NUM_WARPS=config["num_warps"],
        NUM_STAGES=num_stages,
        **config,
    )

    if config["NUM_KSPLIT"] > 1:
        REDUCE_BLOCK_SIZE_M = 32
        REDUCE_BLOCK_SIZE_N = 32
        ACTUAL_KSPLIT = triton.cdiv(K, config["SPLITK_BLOCK_SIZE"])

        grid_reduce = (
            triton.cdiv(M, REDUCE_BLOCK_SIZE_M),
            triton.cdiv(N, REDUCE_BLOCK_SIZE_N),
        )

        _gemm_a8w8_blockscale_reduce_kernel[grid_reduce](
            y_pp,
            y,
            M,
            N,
            y_pp.stride(0),
            y_pp.stride(1),
            y_pp.stride(2),
            y.stride(0),
            y.stride(1),
            REDUCE_BLOCK_SIZE_M,
            REDUCE_BLOCK_SIZE_N,
            ACTUAL_KSPLIT,
            triton.next_power_of_2(config["NUM_KSPLIT"]),
        )

    return y


def gemm_a8w8_blockscale(
    x: torch.Tensor,
    w: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    dtype: float | None = torch.bfloat16,
    y: torch.Tensor | None = None,
    config: dict | None = None,
    skip_reduce: bool | None = False,
    kernel_type: str = "bandwidth_bound",
    backend: str | None = None,
):
    """
    Computes 8 bit matrix multiplication Y = X @ W^T using block-wise quantization scales.
    Each block along K and N dimensions has independent scale factors for fine-grained quantization.

    Args:
        x (torch.Tensor): INT8 input matrix with shape (M, K).
        w (torch.Tensor): INT8 weight matrix with shape (N, K), internally transposed.
        x_scale (torch.Tensor): Block-wise scale for x with shape (M, scale_k).
            scale_k = ceil(K / scale_block_size_k).
        w_scale (torch.Tensor): Block-wise scale for w with shape (scale_n, scale_k).
            scale_n = ceil(N / scale_block_size_n).
        dtype (Optional[torch.dtype]): Output datatype (BF16 or FP16).
        y (Optional[torch.Tensor]): Pre-allocated output tensor with shape (M, N).
        config (Optional[dict]): Kernel tuning parameters (BLOCK_SIZE_M, BLOCK_SIZE_N,
            BLOCK_SIZE_K, GROUP_SIZE_M, NUM_KSPLIT).

    Returns:
        torch.Tensor: Output with shape (M, N).
    """
    _LOGGER.info(
        f"GEMM_A8W8_BLOCKSCALE: x={tuple(x.shape)} w={tuple(w.shape)} x_scale={tuple(x_scale.shape)} w_scale={tuple(w_scale.shape)}"
    )

    M, K = x.shape
    N, K = w.shape

    # Check constraints.
    assert x.shape[1] == w.shape[1], "Incompatible dimensions!!!"

    # Transpose w and w_scale
    w = w.T  # (K, N)
    w_scale = w_scale.T  # (scale_k, scale_n)

    if backend is None:
        backend = "gluon" if get_arch() in _GLUON_DEFAULT_ARCHS else "triton"
    backend = backend.lower()
    assert backend in (
        "triton",
        "gluon",
    ), f"Unknown backend '{backend}', must be 'triton' or 'gluon'"

    if backend == "gluon":
        assert (
            get_arch() in _GLUON_SUPPORTED_ARCHS
        ), f"Gluon backend requires one of {_GLUON_SUPPORTED_ARCHS}, got '{get_arch()}'"
        if get_arch() == "gfx950":
            return _gemm_a8w8_blockscale_gluon_gfx950(
                x, w, x_scale, w_scale, M, N, K, dtype, y, config
            )

    if config is None:
        config, _ = _get_config(M, N, K, backend=backend)

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

    compute_splitk_params(config, K)

    # Scale block sizes
    # TODO: need a better way to pass scale block sizes around
    config["GROUP_K"] = triton.next_power_of_2(
        triton.cdiv(K, w_scale.shape[0])
    )  # scale_block_size_k
    config["GROUP_N"] = triton.next_power_of_2(
        triton.cdiv(N, w_scale.shape[1])
    )  # scale_block_size_n

    assert (
        config["GROUP_K"] == config["BLOCK_SIZE_K"]
    ), "GROUP_K must equal BLOCK_SIZE_K"

    # grid = (config["NUM_KSPLIT"], triton.cdiv(M, config["BLOCK_SIZE_M"]) * triton.cdiv(N, config["BLOCK_SIZE_N"]),)
    grid = lambda META: (
        (
            META["NUM_KSPLIT"]
            * triton.cdiv(M, META["BLOCK_SIZE_M"])
            * triton.cdiv(N, META["BLOCK_SIZE_N"])
        ),  # Effective launch grid dims: [NUM_KSPLIT, NUM_M_BLOCKS, NUM_N_BLOCKS]
    )

    if backend == "gluon":
        from aiter.ops.triton._gluon_kernels.gfx1250.gemm.basic.gemm_a8w8_blockscale import (
            _KERNEL_MAP,
        )

        assert (
            kernel_type in _KERNEL_MAP
        ), f"Unknown kernel_type '{kernel_type}', must be one of {list(_KERNEL_MAP.keys())}"
        _LOGGER.info(
            f"GEMM_A8W8 BLOCKSCALE [gluon/gfx1250]: x={tuple(x.shape)} w={tuple(w.shape)} "
            f"kernel={kernel_type}"
        )

        impl = _KERNEL_MAP[kernel_type]
        extra_constexpr = {}
        warp_bases = [(0, 1)]
        for i in range(int(math.log2(config["num_warps"] // 2))):
            warp_bases.append((1 << i, 0))
        extra_constexpr["warp_bases"] = tuple(warp_bases)
        config["NUM_BUFFERS"] = config.pop("num_stages", 1)

        impl[grid](
            x,
            w,
            y if config["NUM_KSPLIT"] == 1 else y_pp,
            x_scale,
            w_scale,
            M,
            N,
            K,
            x.stride(0),
            x.stride(1),
            w.stride(0),
            w.stride(1),
            0 if config["NUM_KSPLIT"] == 1 else y_pp.stride(0),
            y.stride(0) if config["NUM_KSPLIT"] == 1 else y_pp.stride(1),
            y.stride(1) if config["NUM_KSPLIT"] == 1 else y_pp.stride(2),
            x_scale.stride(0),
            x_scale.stride(1),
            w_scale.stride(0),
            w_scale.stride(1),
            **config,
            **extra_constexpr,
        )
    else:
        impl = triton_gemm_a8w8_blockscale_kernel

        impl[grid](
            x,
            w,
            y if config["NUM_KSPLIT"] == 1 else y_pp,
            x_scale,
            w_scale,
            M,
            N,
            K,
            x.stride(0),
            x.stride(1),
            w.stride(0),
            w.stride(1),
            0 if config["NUM_KSPLIT"] == 1 else y_pp.stride(0),
            y.stride(0) if config["NUM_KSPLIT"] == 1 else y_pp.stride(1),
            y.stride(1) if config["NUM_KSPLIT"] == 1 else y_pp.stride(2),
            x_scale.stride(0),
            x_scale.stride(1),
            w_scale.stride(0),
            w_scale.stride(1),
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
            REDUCE_BLOCK_SIZE_M,
            REDUCE_BLOCK_SIZE_N,
            ACTUAL_KSPLIT,
            triton.next_power_of_2(config["NUM_KSPLIT"]),
            ADD_BIAS=False,
            activation="",
            use_activation=False,
            KERNEL_NAME="_gemm_a8w8_blockscale_reduce_kernel",
        )

    return y


def gemm_a8w8_blockscale_preshuffle(
    x: torch.Tensor,
    w: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    dtype: float | None = torch.bfloat16,
    y: torch.Tensor | None = None,
    config: dict | None = None,
    skip_reduce: bool | None = False,
    is_x_scale_tranposed: bool | None = True,
    kernel_type: str = "bandwidth_bound",
    backend: str | None = None,
):
    """
    Computes 8 bit matrix multiplication Y = X @ W^T using block-wise quantization scales.
    Each block along K and N dimensions has independent scale factors for fine-grained quantization.

    Args:
        x (torch.Tensor): INT8 input matrix with shape (M, K).
        w (torch.Tensor): INT8 weight matrix with shape (N, K), internally transposed.
        x_scale (torch.Tensor): Block-wise scale for x with shape (M, scale_k).
            scale_k = ceil(K / scale_block_size_k).
        w_scale (torch.Tensor): Block-wise scale for w with shape (scale_n, scale_k).
            scale_n = ceil(N / scale_block_size_n).
        dtype (Optional[torch.dtype]): Output datatype (BF16 or FP16).
        y (Optional[torch.Tensor]): Pre-allocated output tensor with shape (M, N).
        config (Optional[dict]): Kernel tuning parameters (BLOCK_SIZE_M, BLOCK_SIZE_N,
            BLOCK_SIZE_K, GROUP_SIZE_M, NUM_KSPLIT).

    Returns:
        torch.Tensor: Output with shape (M, N).
    """
    _LOGGER.info(
        f"GEMM_A8W8_BLOCKSCALE: x={tuple(x.shape)} w={tuple(w.shape)} x_scale={tuple(x_scale.shape)} w_scale={tuple(w_scale.shape)}"
    )

    M, K = x.shape
    N, K = w.shape
    N = N * 16
    K = K // 16

    # Check constraints.
    assert x.shape[1] == w.shape[1] // 16, "Incompatible dimensions!!!"

    # Transpose w and w_scale
    # w = w.T  # (K, N)
    w_scale = w_scale.T  # (scale_k, scale_n)

    # Resolve backend up-front so the config is loaded from the backend's
    # config dir (gemm/<backend>/), falling back to the shared gemm/ dir.
    if backend is None:
        backend = "gluon" if get_arch() in _GLUON_DEFAULT_ARCHS else "triton"
    backend = backend.lower()

    if config is None:
        config, _ = _get_config(M, N, K, True, backend=backend)

    # Triton 3.6 fails TritonAMDGPUConvertToBufferOps for gfx950 preshuffle
    # configs with three pipeline stages. Keep the tuned tile and split-K.
    if (
        backend == "triton"
        and get_arch() == "gfx950"
        and _TRITON_VERSION < Version("3.7.0")
        and config.get("num_stages", 1) > 2
    ):
        config["num_stages"] = 2

    kernel_type_from_config = config.pop("kernel_type", None)
    if kernel_type_from_config is not None:
        kernel_type = kernel_type_from_config

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

    # If block size is greater than split k size, shrink the block size
    if config["BLOCK_SIZE_K"] > config["SPLITK_BLOCK_SIZE"]:
        config["BLOCK_SIZE_K"] = triton.next_power_of_2(config["SPLITK_BLOCK_SIZE"])
        if config["BLOCK_SIZE_K"] > config["SPLITK_BLOCK_SIZE"]:
            config["BLOCK_SIZE_K"] = config["BLOCK_SIZE_K"] // 4
    config["BLOCK_SIZE_K"] = max(
        config["BLOCK_SIZE_K"], 16
    )  # minimum block size is 16 for perf

    # Scale block sizes
    # TODO: need a better way to pass scale block sizes around
    config["GROUP_K"] = triton.next_power_of_2(
        triton.cdiv(K, w_scale.shape[0])
    )  # scale_block_size_k
    config["GROUP_N"] = triton.next_power_of_2(
        triton.cdiv(N, w_scale.shape[1])
    )  # scale_block_size_n

    assert (
        config["GROUP_K"] == config["BLOCK_SIZE_K"]
    ), "GROUP_K must equal BLOCK_SIZE_K"

    if _FORCE_GFX1250_EX:
        config["BLOCK_SIZE_K"] = 64

    # grid = (config["NUM_KSPLIT"], triton.cdiv(M, config["BLOCK_SIZE_M"]) * triton.cdiv(N, config["BLOCK_SIZE_N"]),)
    grid = lambda META: (
        (
            META["NUM_KSPLIT"]
            * triton.cdiv(M, META["BLOCK_SIZE_M"])
            * triton.cdiv(N, META["BLOCK_SIZE_N"])
        ),  # Effective launch grid dims: [NUM_KSPLIT, NUM_M_BLOCKS, NUM_N_BLOCKS]
    )

    extra_constexpr = {}
    if backend is None:
        backend = "gluon" if get_arch() in _GLUON_PRESHUFFLE_ARCHS else "triton"
    backend = backend.lower()
    assert backend in (
        "triton",
        "gluon",
    ), f"Unknown backend '{backend}', must be 'triton' or 'gluon'"

    if backend == "gluon":
        assert (
            get_arch() in _GLUON_PRESHUFFLE_ARCHS
        ), f"Gluon preshuffle requires one of {_GLUON_PRESHUFFLE_ARCHS}, got '{get_arch()}'"
        from aiter.ops.triton._gluon_kernels.gfx1250.gemm.basic.gemm_a8w8_blockscale import (
            _PRESHUFFLE_KERNEL_MAP,
        )

        assert (
            kernel_type in _PRESHUFFLE_KERNEL_MAP
        ), f"Unknown kernel_type '{kernel_type}', must be one of {list(_PRESHUFFLE_KERNEL_MAP.keys())}"
        _LOGGER.info(
            f"GEMM_A8W8 BLOCKSCALE PRESHUFFLE [gluon/gfx1250]: x={tuple(x.shape)} w={tuple(w.shape)} "
            f"kernel={kernel_type}"
        )

        impl = _PRESHUFFLE_KERNEL_MAP[kernel_type]
        warp_bases = [(0, 1)]
        for i in range(int(math.log2(config["num_warps"] // 2))):
            warp_bases.append((1 << i, 0))
        extra_constexpr["warp_bases"] = tuple(warp_bases)
        config["NUM_BUFFERS"] = config.pop("num_stages", 1)
    else:
        impl = triton_gemm_a8w8_blockscale_preshuffle_kernel

    impl[grid](
        x,
        w,
        y if config["NUM_KSPLIT"] == 1 else y_pp,
        x_scale,
        w_scale,
        M,
        N,
        K,
        x.stride(0),
        x.stride(1),
        w.stride(0),
        w.stride(1),
        0 if config["NUM_KSPLIT"] == 1 else y_pp.stride(0),
        y.stride(0) if config["NUM_KSPLIT"] == 1 else y_pp.stride(1),
        y.stride(1) if config["NUM_KSPLIT"] == 1 else y_pp.stride(2),
        x_scale.stride(1) if is_x_scale_tranposed else x_scale.stride(0),
        (
            (x_scale.numel() // x_scale.stride(0))
            if is_x_scale_tranposed
            else x_scale.stride(1)
        ),
        w_scale.stride(0),
        w_scale.stride(1),
        **config,
        **extra_constexpr,
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
            REDUCE_BLOCK_SIZE_M,
            REDUCE_BLOCK_SIZE_N,
            ACTUAL_KSPLIT,
            triton.next_power_of_2(config["NUM_KSPLIT"]),
            ADD_BIAS=False,
            activation="",
            use_activation=False,
            KERNEL_NAME="_gemm_a8w8_blockscale_reduce_kernel",
        )

    return y
