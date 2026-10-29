# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import functools

import torch
import triton

from aiter.jit.utils.torch_guard import torch_compile_guard
from aiter.ops.triton._gluon_kernels.gfx1250.layouts import (
    operand_bundle_cga,
    wmma_layout,
)
from aiter.ops.triton.utils._triton.arch_info import get_arch
from aiter.ops.triton.utils.common_utils import deserialize_str, serialize_dict
from aiter.ops.triton.utils.gemm_config_utils import get_gemm_config
from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER = AiterTritonLogger()

_SUPPORTED_ARCHS = ("gfx1250",)


@functools.lru_cache(maxsize=8)
def _num_cus(device_index):
    return torch.cuda.get_device_properties(device_index).multi_processor_count


@functools.lru_cache(maxsize=128)
def _build_layouts(BLOCK_M, BLOCK_N, BLOCK_K, ctas_per_cga, num_warps, TRANSPOSE_B):
    """Build the WMMA, operand and shared layouts for one cluster-wide tile.

    Cached: the layout objects cost more to build than the launch itself, and they depend only on
    the config, not on the shape.
    """
    per_cta_m = BLOCK_M // ctas_per_cga[0]
    per_cta_n = BLOCK_N // ctas_per_cga[1]

    wmma = wmma_layout((per_cta_m, per_cta_n), ctas_per_cga, num_warps)
    dot_a, shared_a = operand_bundle_cga(wmma, 0, (per_cta_m, BLOCK_K), ctas_per_cga)
    if TRANSPOSE_B:
        dot_b, shared_b = operand_bundle_cga(
            wmma, 1, (per_cta_n, BLOCK_K), ctas_per_cga, transpose_b=True
        )
    else:
        dot_b, shared_b = operand_bundle_cga(
            wmma, 1, (BLOCK_K, per_cta_n), ctas_per_cga
        )
    return wmma, dot_a, dot_b, shared_a, shared_b


def gemm_a16w16_streamk_fake_tensor(
    x: torch.Tensor,
    w: torch.Tensor,
    dtype: torch.dtype | None = None,
    y: torch.Tensor | None = None,
    config: str | None = None,
) -> torch.Tensor:
    if y is not None:
        return y
    return torch.empty(
        (x.shape[0], w.shape[0]), dtype=dtype or x.dtype, device=x.device
    )


@torch_compile_guard(gen_fake=gemm_a16w16_streamk_fake_tensor)
def gemm_a16w16_streamk_(
    x: torch.Tensor,
    w: torch.Tensor,
    dtype: torch.dtype | None = None,
    y: torch.Tensor | None = None,
    config: str | None = None,
) -> torch.Tensor:
    """WG cluster StreamK GEMM on gfx1250: y = x @ w.T, one tile per workgroup cluster.

    x is (M, K) row major and w is (N, K). A K-major w (w.stride(1) == 1) keeps B in its
    transposed form; an N-major w takes the native path, which also unlocks the aliased C store.
    """
    assert any(
        a in get_arch() for a in _SUPPORTED_ARCHS
    ), f"gemm_a16w16_streamk requires one of {_SUPPORTED_ARCHS}, got '{get_arch()}'"
    assert x.shape[1] == w.shape[1], f"incompatible shapes {x.shape} and {w.shape}"
    assert x.stride(1) == 1, f"x must be row major (M, K), got strides {x.stride()}"

    M, K = x.shape
    N = w.shape[0]
    if dtype is None:
        dtype = x.dtype

    cfg = deserialize_str(config) if config else None
    if cfg is None:
        cfg, _ = get_gemm_config("GEMM-A16W16-STREAMK", M, N, K, backend="gluon")

    BLOCK_M = cfg["BLOCK_M"]
    BLOCK_N = cfg["BLOCK_N"]
    BLOCK_K = cfg["BLOCK_K"]
    NUM_BUFFERS = cfg["NUM_BUFFERS"]
    GROUP_SIZE_M = cfg["GROUP_SIZE_M"]
    CTAS_M = cfg["CTAS_M"]
    CTAS_N = cfg["CTAS_N"]
    num_warps = cfg["num_warps"]
    waves_per_eu = cfg["waves_per_eu"]

    # BLOCK_M and BLOCK_N are cluster-wide, so each CTA must get a whole sub-tile.
    assert (
        BLOCK_M % CTAS_M == 0 and BLOCK_N % CTAS_N == 0
    ), f"BLOCK_M={BLOCK_M} BLOCK_N={BLOCK_N} must divide by the cluster {CTAS_M}x{CTAS_N}"
    num_ctas = CTAS_M * CTAS_N

    # The kernel swaps the two strides itself for a transposed B, so it always gets the (K, N)
    # pair; only the orientation flag changes.
    w_t = w.T
    if w_t.stride(1) == 1:
        TRANSPOSE_B = False
    elif w_t.stride(0) == 1:
        TRANSPOSE_B = True
    else:
        raise ValueError(
            f"w must be contiguous in one dimension, got strides {w.stride()}"
        )
    stride_bk, stride_bn = w_t.stride(0), w_t.stride(1)

    # The steady loop needs a full ring before it runs, so clamp the depth to the K tile count.
    num_k_tiles = triton.cdiv(K, BLOCK_K)
    NUM_BUFFERS = max(2, min(NUM_BUFFERS, num_k_tiles))

    if y is None:
        y = torch.empty((M, N), dtype=dtype, device=x.device)

    wmma, dot_a, dot_b, shared_a, shared_b = _build_layouts(
        BLOCK_M, BLOCK_N, BLOCK_K, (CTAS_M, CTAS_N), num_warps, TRANSPOSE_B
    )

    num_tiles = triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)
    NUM_SMS = min(_num_cus(x.device.index) // num_ctas, num_tiles)

    # KTAIL is decided here, from the real K, so only one drain path is compiled.
    KTAIL = 1 if (K % BLOCK_K) == (BLOCK_K // 2) else 0

    _LOGGER.info("GEMM_A16W16_STREAMK [gluon/gfx1250]: x=%s w=%s", x.shape, w.shape)

    from aiter.ops.triton._gluon_kernels.gfx1250.gemm.basic.gemm_a16w16_streamk import (
        gemm_a16w16_streamk_kernel_,
    )

    gemm_a16w16_streamk_kernel_[(NUM_SMS,)](
        x,
        w_t,
        y,
        M,
        N,
        K,
        num_tiles,
        x.stride(0),
        x.stride(1),
        stride_bk,
        stride_bn,
        y.stride(0),
        y.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        NUM_BUFFERS=NUM_BUFFERS,
        TRANSPOSE_B=TRANSPOSE_B,
        SHARED_LAYOUT_A=shared_a,
        SHARED_LAYOUT_B=shared_b,
        OPERAND_LAYOUT_A=dot_a,
        OPERAND_LAYOUT_B=dot_b,
        WMMA_LAYOUT=wmma,
        NUM_SMS=NUM_SMS,
        CTAS_N=CTAS_N,
        GROUP_SIZE_M=GROUP_SIZE_M,
        KTAIL=KTAIL,
        num_warps=num_warps,
        num_ctas=num_ctas,
        waves_per_eu=waves_per_eu,
    )
    return y


def gemm_a16w16_streamk(
    x: torch.Tensor,
    w: torch.Tensor,
    dtype: torch.dtype | None = None,
    y: torch.Tensor | None = None,
    config: dict | None = None,
) -> torch.Tensor:
    """Public entry point. See gemm_a16w16_streamk_ for the argument contract."""
    return gemm_a16w16_streamk_(
        x, w, dtype, y, serialize_dict(config) if config else None
    )
