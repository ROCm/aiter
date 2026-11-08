# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Explicit gfx1250 TDM GEMM with atomic or FP32-workspace cluster split-K."""

import torch


def gemm_a16w16_tdm(
    a: torch.Tensor,
    b: torch.Tensor,
    out: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    *,
    block_m: int = 128,
    block_n: int = 128,
    block_k: int = 128,
    stages: int = 2,
    m_waves: int = 2,
    n_waves: int = 2,
    split_k: int = 1,
    split_k_policy: str = "atomic",
    group_m: int = 0,
    layout: str = "nt",
    interleave: bool = True,
    stream: torch.cuda.Stream | None = None,
) -> torch.Tensor:
    """C[M,N] = A[M,K] @ B[K,N], optionally plus bias[N].

    B is a logical [K,N] view: NT has stride(0)=1, NN stride(1)=1.
    Explicit entrypoint only: existing GEMM dispatch/tuning is unchanged.

    split_k_policy="workspace" uses FP32 LDS/global partials and a fixed-order
    FP32 sum before the final output rounding. It supports odd N and deterministic
    mode, but can differ from split_k=1 due to FP32 accumulation grouping.

    split_k_policy="atomic" (backward-compatible default), when split_k>1,
    clears C inside cluster rank 0, synchronizes the cluster, then
    each rank atomically adds its rounded BF16/FP16 partial, using packed pairs.
    It is nondeterministic and NOT equivalent to FP32 partial reduction:
    cancellation and overflow may magnify low-precision rounding. Use split_k=1
    when deterministic accumulation or small absolute cancellation error matters.
    With bias, a second cluster barrier precedes one final bias addition.

    K must be divisible by split_k*block_k. Odd N is supported with split_k=1 or workspace policy.
    Leading strides must be 16-byte aligned; all tensors remain caller-owned.
    Inputs produced on another stream require the caller's normal stream ordering.
    """
    import flydsl.expr as fx

    from .kernels.gemm_a16w16_tdm_gfx1250 import compile_gemm, validate_config
    from .kernels.tensor_shim import _run_compiled, ptr_arg

    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError("expected A[M,K] and B[K,N]")
    if a.dtype not in (torch.bfloat16, torch.float16) or b.dtype != a.dtype:
        raise ValueError("A and B must have the same BF16/FP16 dtype")
    if a.device.type != "cuda" or b.device != a.device:
        raise ValueError("A and B must be on the same ROCm device")
    if type(interleave) is not bool:
        raise ValueError("interleave must be bool")
    validate_config(
        block_m,
        block_n,
        block_k,
        stages,
        m_waves,
        n_waves,
        split_k,
        group_m,
        layout,
        interleave,
        split_k_policy,
    )
    m, k = a.shape
    n = b.shape[1]
    if min(m, n, k) <= 0 or k % (split_k * block_k):
        raise ValueError(
            "positive dimensions and K divisible by split_k*block_k required"
        )
    if k // (split_k * block_k) < stages:
        raise ValueError("each K split must contain at least stages full tiles")
    if group_m * ((n + block_n - 1) // block_n) >= 2**31:
        raise ValueError("grouped swizzle exceeds the supported 31-bit range")
    if split_k > 1 and split_k_policy == "atomic" and n % 2:
        raise ValueError("packed split-K requires even N")
    if (
        torch.are_deterministic_algorithms_enabled()
        and split_k > 1
        and split_k_policy == "atomic"
    ):
        raise RuntimeError("packed atomic split-K is nondeterministic; use split_k=1")

    def leading(t, unit, major, minimum, name):
        stride = t.stride(major)
        if t.stride(unit) != 1 or stride < minimum or stride * 2 % 16:
            raise ValueError(
                f"{name} requires unit minor stride and aligned leading stride"
            )
        return stride

    lda = leading(a, 1, 0, k, "A")
    ldb = leading(b, 0, 1, k, "B NT") if layout == "nt" else leading(b, 1, 0, n, "B NN")
    if out is None:
        # Odd N uses a padded leading stride without exposing padding as output.
        out = torch.empty((m, (n + 7) // 8 * 8), device=a.device, dtype=a.dtype)[:, :n]
    if out.shape != (m, n) or out.dtype != a.dtype or out.device != a.device:
        raise ValueError("out must be [M,N] with input dtype/device")
    ldc = leading(out, 1, 0, n, "C")
    if bias is not None and (
        bias.shape != (n,)
        or bias.dtype != a.dtype
        or bias.device != a.device
        or not bias.is_contiguous()
    ):
        raise ValueError("bias must be contiguous [N] with input dtype/device")
    tensors = [a, b, out] + ([] if bias is None else [bias])
    if any(t.data_ptr() % 16 for t in tensors):
        raise ValueError("all tensor pointers must be 16-byte aligned")

    def span(t):
        size = 1 + sum((s - 1) * st for s, st in zip(t.shape, t.stride()))
        return t.data_ptr(), t.data_ptr() + size * t.element_size()

    lo, hi = span(out)
    if any(
        lo < span(t)[1] and span(t)[0] < hi
        for t in [a, b] + ([] if bias is None else [bias])
    ):
        raise ValueError("out must not overlap A, B or bias")
    # Packed buffer atomics use 32-bit byte offsets. Restrict all shapes to the
    # same safe range; TDM/global pointer calculations otherwise use i64.
    if max(m * lda, n * ldb if layout == "nt" else k * ldb, m * ldc) * 2 >= 2**31:
        raise ValueError("tensor footprint exceeds the supported 31-bit byte range")
    with torch.cuda.device(a.device):
        arch = torch.cuda.get_device_properties(a.device).gcnArchName.split(":")[0]
        if arch != "gfx1250":
            raise ValueError(f"TDM GEMM requires gfx1250, got {arch}")
        stream = torch.cuda.current_stream(a.device) if stream is None else stream
        if stream.device != a.device:
            raise ValueError("stream must belong to the input device")
        launch = compile_gemm(
            "bf16" if a.dtype == torch.bfloat16 else "fp16",
            block_m,
            block_n,
            block_k,
            stages,
            m_waves,
            n_waves,
            split_k,
            group_m,
            layout,
            interleave,
            bias is not None,
            split_k_policy,
        )
        if split_k > 1 and split_k_policy == "workspace":
            size = ((m + block_m - 1) // block_m) * ((n + block_n - 1) // block_n)
            size *= split_k * block_m * block_n
            workspace = torch.empty(size, dtype=torch.float32, device=a.device)
        else:
            workspace = out
        elem = fx.BFloat16 if a.dtype == torch.bfloat16 else fx.Float16
        _run_compiled(
            launch,
            ptr_arg(a, elem),
            ptr_arg(b, elem),
            ptr_arg(out, elem),
            ptr_arg(out if bias is None else bias, elem),
            ptr_arg(workspace, fx.Float32 if workspace is not out else elem),
            m,
            n,
            k,
            lda,
            ldb,
            ldc,
            fx.Stream(stream.cuda_stream),
        )
        for t in tensors + [workspace]:
            t.record_stream(stream)
    return out
