# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Torch launch wrapper for the gfx1250 BF16/FP16 TDM GEMM."""

from typing import Optional

import torch


def gemm_a16w16_tdm(
    a: torch.Tensor,
    b: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    *,
    block_m: int = 256,
    block_n: int = 256,
    block_k: int = 128,
    stages: int = 2,
    m_waves: int = 4,
    n_waves: int = 4,
    group_m: int = 0,
    mma_m: int = 16,
    mma_n: int = 16,
    mma_k: int = 32,
    split_k: int = 1,
    stream: Optional[torch.cuda.Stream] = None,
    layout: str = "nt",
    interleave: bool = False,
    split_k_atomic: bool = False,
    split_k_policy: str | None = None,
) -> torch.Tensor:
    """C = A[M,K] @ B[K,N]; FP16/BF16 output, optional bias.

    ``layout`` is ``nt`` or ``nn``. N means that dimension is contiguous
    (stride 1); T means the other dimension is contiguous. The other stride
    is the leading stride and may be larger than the contiguous size, so
    padded rows are allowed. C is always row-major. M and N may end on a
    partial tile; K must contain at least ``stages - 1`` complete tiles.
    No split-K tail: K must be divisible by split_k * block_k. split_k is the
    cluster size along the K split and must be in 1..9.
    ``interleave=True`` uses four quadrant tiles per wave and opposite serpentine
    orders on even/odd waves. It requires NT B and sufficiently large even wave
    tiles; the default schedule and split-K reduction semantics are unchanged.

    ``split_k_atomic=True`` opts into workspace-free cluster reduction using
    packed BF16 atomic adds. It requires split_k > 1, even N and an even output
    leading stride. Each atomic update rounds to BF16 and rank arrival order
    is unspecified, so results may differ between runs and from the default
    FP32 sum of rounded partials. Workspace partials are stored in the
    input/output dtype (BF16 or FP16, two bytes), not FP32. Bias is added once
    after the completed sum. ``split_k_policy`` is a compatibility alias;
    ``"workspace"`` selects the default, ``"atomic"`` selects BF16 atomics.
    """
    import flydsl.expr as fx
    from aiter.jit.utils.chip_info import get_gfx_runtime
    from .kernels.gemm_a16w16_tdm_gfx1250 import (
        GEMM_DTYPE_BF16,
        GEMM_DTYPE_FP16,
        build_gemm_gfx1250,
        validate_gemm_gfx1250_config,
    )
    from .kernels.tensor_shim import _run_compiled, ptr_arg

    if split_k_policy is not None:
        if split_k_policy not in ("atomic", "workspace"):
            raise ValueError("split_k_policy must be atomic or workspace")
        requested_atomic = split_k_policy == "atomic" and split_k > 1
        if split_k_atomic and not requested_atomic:
            raise ValueError("conflicting split-K policy arguments")
        split_k_atomic = requested_atomic
    if split_k_atomic and torch.are_deterministic_algorithms_enabled():
        raise RuntimeError("BF16 atomic split-K is nondeterministic")
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError("expected A[M,K] and B[K,N]")
    if a.dtype not in (torch.float16, torch.bfloat16) or b.dtype != a.dtype:
        raise ValueError("A and B must have the same FP16 or BF16 dtype")
    dtype_id = GEMM_DTYPE_FP16 if a.dtype is torch.float16 else GEMM_DTYPE_BF16
    if a.device != b.device or a.device.type != "cuda":
        raise ValueError("A and B must be on the same ROCm device")
    layout = layout.lower()
    if layout not in ("nn", "nt"):
        raise ValueError("layout must be 'nn' or 'nt'")
    m, k = a.shape
    n = b.shape[1]
    validate_gemm_gfx1250_config(
        m,
        n,
        k,
        block_m,
        block_n,
        block_k,
        stages,
        m_waves,
        n_waves,
        group_m,
        dtype_id,
        mma_m,
        mma_n,
        mma_k,
        layout == "nt",
        split_k,
        interleave,
        split_k_atomic,
    )
    a_leading_stride = _leading_stride(
        a, 1, 0, k, f"A does not satisfy the {layout.upper()} layout"
    )
    if layout == "nt":
        b_leading_stride = _leading_stride(
            b, 0, 1, k, "B does not satisfy the NT layout"
        )
    else:
        b_leading_stride = _leading_stride(
            b, 1, 0, n, "B does not satisfy the NN layout"
        )
    if out is None:
        out = torch.empty((m, (n + 7) // 8 * 8), device=a.device, dtype=a.dtype)[:, :n]
    if out.shape != (m, n) or out.dtype != a.dtype or out.device != a.device:
        raise ValueError("out must be [M,N] with the input dtype and device")
    c_leading_stride = _leading_stride(out, 1, 0, n, "out must be row-major [M,N]")
    if split_k_atomic and c_leading_stride % 2:
        raise ValueError("split_k_atomic requires an even output leading stride")
    if bias is not None and (
        bias.shape != (n,)
        or bias.dtype != a.dtype
        or bias.device != a.device
        or not bias.is_contiguous()
    ):
        raise ValueError("bias must be contiguous [N] with the input dtype and device")
    tensors = [a, b, out] + ([] if bias is None else [bias])
    if any(t.data_ptr() % 16 for t in tensors):
        raise ValueError("all pointers must be 16-byte aligned")
    b_rows = n if layout == "nt" else k
    if (
        max(m * a_leading_stride, b_rows * b_leading_stride, m * c_leading_stride)
        > (2**32 - 1) // 2
    ):
        raise ValueError("tensor offsets exceed the supported 32-bit byte range")
    out_lo, out_hi = _storage_span(out)
    if any(
        _overlaps(out_lo, out_hi, t) for t in [a, b] + ([] if bias is None else [bias])
    ):
        raise ValueError("out must not overlap the inputs")
    with torch.cuda.device(a.device):
        arch = get_gfx_runtime()
        if arch != "gfx1250":
            raise ValueError(f"TDM GEMM requires gfx1250, got {arch}")
        if stream is None:
            stream = torch.cuda.current_stream()
        if stream.device != a.device:
            raise ValueError("stream must belong to the input device")
        dtype = fx.Float16 if dtype_id == GEMM_DTYPE_FP16 else fx.BFloat16
        mn_tiles = ((m + block_m - 1) // block_m) * ((n + block_n - 1) // block_n)
        # Partials use the input/output dtype: BF16 GEMM has a 2-byte BF16
        # workspace. Only the post-barrier register sum is FP32.
        workspace = (
            out
            if split_k == 1 or split_k_atomic
            else torch.empty(
                (mn_tiles * split_k * block_m * block_n,),
                device=a.device,
                dtype=a.dtype,
            )
        )
        has_bias = bias is not None
        b_k_contiguous = layout == "nt"
        launch = build_gemm_gfx1250(
            dtype_id,
            block_m,
            block_n,
            block_k,
            stages,
            m_waves,
            n_waves,
            group_m,
            has_bias,
            b_k_contiguous,
            mma_m,
            mma_n,
            mma_k,
            split_k,
            interleave,
            split_k_atomic,
            device_id=torch.cuda.current_device(),
        )
        _run_compiled(
            launch,
            ptr_arg(a, dtype),
            ptr_arg(b, dtype),
            ptr_arg(out, dtype),
            ptr_arg(out if bias is None else bias, dtype),
            ptr_arg(workspace, dtype),
            m,
            n,
            k,
            a_leading_stride,
            b_leading_stride,
            c_leading_stride,
            fx.Stream(stream.cuda_stream),
        )
        for tensor in tensors + [workspace]:
            tensor.record_stream(stream)
    return out


def _leading_stride(tensor, unit_dim, leading_dim, min_leading, message):
    unit, leading = tensor.stride(unit_dim), tensor.stride(leading_dim)
    if unit != 1 or leading < min_leading or leading * tensor.element_size() % 16 != 0:
        raise ValueError(
            f"{message}: expected stride({unit_dim}) == 1 and a 16-byte-aligned "
            f"leading stride({leading_dim}) >= {min_leading}; got stride={tuple(tensor.stride())}"
        )
    return int(leading)


def _storage_span(tensor):
    last = (tensor.shape[0] - 1) * tensor.stride(0) + (
        tensor.shape[1] - 1
    ) * tensor.stride(1)
    start = tensor.data_ptr()
    return start, start + (last + 1) * tensor.element_size()


def _overlaps(out_lo, out_hi, tensor):
    lo, hi = (
        _storage_span(tensor)
        if tensor.ndim == 2
        else (
            tensor.data_ptr(),
            tensor.data_ptr() + tensor.numel() * tensor.element_size(),
        )
    )
    return out_lo < hi and lo < out_hi
