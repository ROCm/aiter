# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Explicit BF16 HGEMM with FP32 split-K accumulation and a final BF16 copy."""

import torch

from aiter.jit.utils.chip_info import get_cu_num, get_gfx_runtime

from .gemm_kernels import flydsl_hgemm


def flydsl_hgemm_fp32_workspace(
    a: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor,
    workspace: torch.Tensor,
    *,
    stream: torch.cuda.Stream | None = None,
) -> torch.Tensor:
    """Compute BF16 ``a @ weight.T`` using caller-owned FP32 scratch.

    This inference-only, opt-in path is qualified for exact ``(M,N,K) =
    (128,7168,12288)`` on gfx950 with 256 CUs. All operands must be aligned,
    contiguous row-major tensors on one device. Bias, scaling, prepacked
    weights and autograd are not supported. Unsupported calls raise before
    launching; this entry does not change the normal HGEMM/CSV dispatch.

    The existing FlyDSL FP32-output split-K4 GEMM is followed by a PyTorch
    FP32-to-BF16 copy on the same stream: TWO kernels, not a fused epilogue.
    FP32 atomic accumulation avoids per-split BF16 rounding but is not
    bitwise deterministic. Deterministic-algorithms mode is rejected.

    ``workspace`` is FP32[128,7168] (3.5 MiB); ``out`` is BF16[128,7168].
    Neither may overlap inputs or each other. Contents may be uninitialized:
    the GEMM initializes scratch on every invocation. No output/scratch
    allocation or host read of tensor contents is performed here.

    Sequential calls on a stream may reuse scratch. Concurrent streams or
    graph replays require distinct scratch/output buffers or explicit
    dependencies. Order input producers and output consumers when using an
    explicit stream, and keep all buffers alive until work completes.
    Warm up before graph capture; graph replays must retain the buffers.

    The non-temporal B policy targets rotating weights. It can be slower
    than cached HGEMM or hipBLASLt for repeatedly reused hot weights.
    """
    m, n, k = 128, 7168, 12288
    specs = (
        ("a", a, (m, k), torch.bfloat16),
        ("weight", weight, (n, k), torch.bfloat16),
        ("out", out, (m, n), torch.bfloat16),
        ("workspace", workspace, (m, n), torch.float32),
    )
    for name, tensor, shape, dtype in specs:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if (
            tensor.layout != torch.strided
            or tuple(tensor.shape) != shape
            or tensor.dtype != dtype
            or tensor.device.type != "cuda"
            or tensor.device != a.device
            or not tensor.is_contiguous()
            or tensor.is_conj()
            or tensor.is_neg()
            or tensor.requires_grad
        ):
            raise ValueError(
                f"{name} must be contiguous {shape} {dtype} on the input ROCm "
                "device, without autograd or lazy conjugate/negative views"
            )
        if tensor.data_ptr() % 16:
            raise ValueError(f"{name} must be 16-byte aligned")

    if getattr(weight, "is_shuffled", False):
        raise ValueError("prepacked weights are not supported")
    if stream is not None and not isinstance(stream, torch.cuda.Stream):
        raise TypeError("stream must be a torch.cuda.Stream")

    def overlaps(lhs, rhs):
        lb, rb = lhs.data_ptr(), rhs.data_ptr()
        return (
            lb < rb + rhs.numel() * rhs.element_size()
            and rb < lb + lhs.numel() * lhs.element_size()
        )

    for target in (out, workspace):
        if any(overlaps(target, source) for source in (a, weight)):
            raise ValueError("out/workspace must not overlap a or weight")
    if overlaps(out, workspace):
        raise ValueError("out and workspace must not overlap")
    if torch.are_deterministic_algorithms_enabled():
        raise RuntimeError("FP32 atomic split-K is nondeterministic")

    with torch.cuda.device(a.device):
        if get_gfx_runtime() != "gfx950" or get_cu_num() != 256:
            raise ValueError("This HGEMM path requires gfx950 with 256 CUs")
        if stream is None:
            stream = torch.cuda.current_stream(a.device)
        if stream.device != a.device:
            raise ValueError("stream must be on the input device")
        with torch.cuda.stream(stream):
            flydsl_hgemm(
                a,
                weight,
                out=workspace,
                out_dtype=torch.float32,
                block_m=128,
                block_n=128,
                block_k=64,
                stages=4,
                split_k=4,
                m_waves=2,
                n_waves=4,
                k_waves=1,
                group_m=0,
                policy="ft",
                k_stagger=0,
                b_non_temporal=True,
                stream=stream,
            )
            out.copy_(workspace)
    return out
