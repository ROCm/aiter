# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""SonicMoE grouped GEMM backend selection.

Triton remains the default. hipBLASLt is opt-in and lives outside the Triton
wrappers, matching the dense GEMM dispatch in ``aiter.tuned_gemm``.
"""

import os
import weakref
from collections import OrderedDict

import torch

_HOST_CU_SEQLENS_CACHE_MAX_ENTRIES = 4096
_HOST_CU_SEQLENS_CACHE: OrderedDict[
    tuple[int, int], tuple[weakref.ReferenceType, torch.Tensor]
] = OrderedDict()


def _cu_seqlens_cache_key(cu_seqlens: torch.Tensor) -> tuple[int, int]:
    device_index = cu_seqlens.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    return device_index, cu_seqlens.data_ptr()


def register_host_cu_seqlens(
    cu_seqlens: torch.Tensor, host_cu_seqlens: torch.Tensor
) -> None:
    """Cache host offsets to avoid GPU-to-CPU syncs in the multi-stream backend."""
    if cu_seqlens.device.type != "cuda":
        raise ValueError("cu_seqlens cache keys must be GPU tensors")
    host_cu_seqlens = host_cu_seqlens.to(
        device="cpu", dtype=torch.int64, copy=False
    ).contiguous()
    key = _cu_seqlens_cache_key(cu_seqlens)
    _HOST_CU_SEQLENS_CACHE[key] = (weakref.ref(cu_seqlens), host_cu_seqlens)
    _HOST_CU_SEQLENS_CACHE.move_to_end(key)
    while len(_HOST_CU_SEQLENS_CACHE) > _HOST_CU_SEQLENS_CACHE_MAX_ENTRIES:
        _HOST_CU_SEQLENS_CACHE.popitem(last=False)


def _registered_host_cu_seqlens(
    cu_seqlens: torch.Tensor,
) -> torch.Tensor | None:
    if cu_seqlens.device.type != "cuda":
        return cu_seqlens
    key = _cu_seqlens_cache_key(cu_seqlens)
    entry = _HOST_CU_SEQLENS_CACHE.get(key)
    if entry is None:
        return None
    tensor_ref, host_cu_seqlens = entry
    if tensor_ref() is not cu_seqlens:
        _HOST_CU_SEQLENS_CACHE.pop(key, None)
        return None
    _HOST_CU_SEQLENS_CACHE.move_to_end(key)
    return host_cu_seqlens


def clear_registered_host_cu_seqlens(cu_seqlens: torch.Tensor) -> None:
    """Discard a stale host-offset entry for a newly allocated GPU tensor."""
    if cu_seqlens.device.type == "cuda":
        _HOST_CU_SEQLENS_CACHE.pop(_cu_seqlens_cache_key(cu_seqlens), None)


def _local_tensor(tensor: torch.Tensor | None) -> torch.Tensor | None:
    if tensor is not None and hasattr(tensor, "to_local"):
        return tensor.to_local()
    return tensor


def _grouped_output_dtype(
    A: torch.Tensor,
    B: torch.Tensor,
    out: torch.Tensor | None,
    out_dtype: torch.dtype | None,
) -> torch.dtype:
    if out_dtype is not None:
        return out_dtype
    if out is not None:
        return out.dtype
    return torch.promote_types(A.dtype, B.dtype)


def grouped_gemm(
    A: torch.Tensor,
    B: torch.Tensor,
    cu_seqlens: torch.Tensor,
    out: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    A_idx: torch.Tensor | None = None,
    scatter_idx: torch.Tensor | None = None,
    A_is_transposed: bool = False,
    B_is_transposed: bool = False,
    A_scale: torch.Tensor | None = None,
    B_scale: torch.Tensor | None = None,
    block_size: int = 128,
    out_dtype: torch.dtype | None = None,
):
    """Run grouped GEMM on the backend named by ``SONIC_MOE_GROUPED_GEMM_BACKEND``."""
    if (A_scale is None) != (B_scale is None):
        raise ValueError("A_scale and B_scale must be provided together")
    if A_scale is not None and block_size != 128:
        raise ValueError("Sonic blockwise FP8 requires block_size=128")
    # The hipBLASLt backends do not accept block scales.
    backend = (
        "triton"
        if A_scale is not None
        else os.environ.get("SONIC_MOE_GROUPED_GEMM_BACKEND", "triton").lower()
    )
    if backend not in {"triton", "hipblaslt", "multistream", "auto"}:
        raise ValueError(
            "SONIC_MOE_GROUPED_GEMM_BACKEND must be triton, hipblaslt, "
            "multistream, or auto"
        )
    if A_is_transposed:
        if B_is_transposed:
            raise ValueError("a grouped wgrad does not support a transposed B")
        if bias is not None:
            raise ValueError("bias is invalid for a grouped wgrad")
        if scatter_idx is not None:
            raise ValueError("scatter_idx is invalid for a grouped wgrad")
    if backend == "multistream":
        result = _grouped_gemm_multistream(
            A,
            B,
            cu_seqlens,
            out,
            bias,
            A_idx,
            scatter_idx,
            A_is_transposed,
            B_is_transposed,
            out_dtype,
        )
        return out if out is not None else result
    if backend in {"hipblaslt", "auto"}:
        try:
            result = _grouped_gemm_hipblaslt(
                A,
                B,
                cu_seqlens,
                out,
                bias,
                A_idx,
                scatter_idx,
                A_is_transposed,
                B_is_transposed,
                out_dtype,
            )
        except (RuntimeError, ValueError):
            if backend == "hipblaslt":
                raise
        else:
            return out if out is not None else result

    from aiter.ops.triton.moe.sonicmoe import grouped_gemm as triton_grouped_gemm

    return triton_grouped_gemm(
        A,
        B,
        cu_seqlens,
        out,
        bias,
        A_idx,
        scatter_idx,
        A_is_transposed,
        B_is_transposed,
        A_scale,
        B_scale,
        block_size,
        out_dtype,
    )


def _grouped_gemm_hipblaslt(
    A: torch.Tensor,
    B: torch.Tensor,
    cu_seqlens: torch.Tensor,
    out: torch.Tensor | None,
    bias: torch.Tensor | None,
    A_idx: torch.Tensor | None,
    scatter_idx: torch.Tensor | None,
    A_is_transposed: bool,
    B_is_transposed: bool,
    out_dtype: torch.dtype | None = None,
):
    from aiter.ops.gradlib import hipb_grouped_mm

    A = _local_tensor(A)
    B = _local_tensor(B)
    cu_seqlens = _local_tensor(cu_seqlens)
    bias = _local_tensor(bias)
    out = _local_tensor(out)
    A_idx = _local_tensor(A_idx)
    scatter_idx = _local_tensor(scatter_idx)
    if scatter_idx is not None:
        scatter_idx = scatter_idx.to(dtype=torch.int64)
    work_a = A.index_select(0, A_idx) if A_idx is not None else A
    work_a = work_a.contiguous()
    work_b = (
        B.contiguous()
        if A_is_transposed or B_is_transposed
        else B.transpose(1, 2).contiguous()
    )
    counts = cu_seqlens.contiguous()
    requested_dtype = _grouped_output_dtype(work_a, work_b, out, out_dtype)
    # Grouped hipBLASLt has FP16 algorithms. BF16 and FP32 queries do not, so a
    # requested FP32 output is computed in the input dtype and cast. BF16 is
    # submitted as requested and raises when the library has no algorithm.
    compute_dtype = (
        requested_dtype
        if requested_dtype in (torch.float16, torch.bfloat16)
        else work_a.dtype
    )
    if work_a.dtype != compute_dtype or work_b.dtype != compute_dtype:
        work_a = work_a.to(dtype=compute_dtype)
        work_b = work_b.to(dtype=compute_dtype)
    result_dtype = compute_dtype

    if A_is_transposed:
        if scatter_idx is not None:
            raise ValueError("scatter_idx is invalid for a grouped wgrad")
        E = counts.numel() - 1
        shape = (E, work_a.shape[1], work_b.shape[1])
    else:
        shape = (work_a.shape[0], work_b.shape[1])

    direct_out = (
        out is not None
        and out.is_contiguous()
        and out.dtype == result_dtype
        and scatter_idx is None
        and tuple(out.shape) == shape
    )
    work_out = (
        out if direct_out else torch.empty(shape, dtype=result_dtype, device=A.device)
    )

    hipb_grouped_mm(
        work_a,
        work_b,
        counts,
        work_out,
        A_is_transposed,
        (bias.to(dtype=result_dtype).contiguous() if bias is not None else None),
    )

    if out is None:
        if scatter_idx is None:
            if work_out.dtype == requested_dtype:
                return work_out
            return work_out.to(dtype=requested_dtype)
        out = torch.empty(work_out.shape, dtype=requested_dtype, device=work_out.device)
    if work_out.dtype != out.dtype:
        work_out = work_out.to(dtype=out.dtype)
    if scatter_idx is not None:
        out.index_copy_(0, scatter_idx, work_out)
    elif work_out is not out:
        out.copy_(work_out)
    return out


def _grouped_gemm_multistream(
    A: torch.Tensor,
    B: torch.Tensor,
    cu_seqlens: torch.Tensor,
    out: torch.Tensor | None,
    bias: torch.Tensor | None,
    A_idx: torch.Tensor | None,
    scatter_idx: torch.Tensor | None,
    A_is_transposed: bool,
    B_is_transposed: bool,
    out_dtype: torch.dtype | None = None,
):
    from aiter.ops.gradlib import hipb_multistream_mm

    A = _local_tensor(A)
    B = _local_tensor(B)
    cu_seqlens = _local_tensor(cu_seqlens)
    bias = _local_tensor(bias)
    out = _local_tensor(out)
    A_idx = _local_tensor(A_idx)
    scatter_idx = _local_tensor(scatter_idx)
    if scatter_idx is not None:
        scatter_idx = scatter_idx.to(dtype=torch.int64)
    work_a = A.index_select(0, A_idx) if A_idx is not None else A
    work_a = work_a.contiguous()
    work_b = B.contiguous()
    host_cu_seqlens = _registered_host_cu_seqlens(cu_seqlens)
    counts = host_cu_seqlens if host_cu_seqlens is not None else cu_seqlens.contiguous()
    if A_is_transposed:
        if scatter_idx is not None:
            raise ValueError("scatter_idx is invalid for a grouped wgrad")
        shape = (counts.numel() - 1, work_a.shape[1], work_b.shape[1])
    else:
        shape = (
            work_a.shape[0],
            work_b.shape[1] if B_is_transposed else work_b.shape[2],
        )

    input_dtype = torch.promote_types(work_a.dtype, work_b.dtype)
    work_a = work_a.to(dtype=input_dtype)
    work_b = work_b.to(dtype=input_dtype)
    result_dtype = _grouped_output_dtype(work_a, work_b, out, out_dtype)
    # FP32 outputs are stored directly. A narrower output with a wider input
    # has no hipBLASLt algorithm, so compute in the input dtype and cast.
    compute_dtype = (
        result_dtype if result_dtype in (input_dtype, torch.float32) else input_dtype
    )
    direct_out = (
        out is not None
        and out.is_contiguous()
        and out.dtype == compute_dtype
        and compute_dtype == result_dtype
        and scatter_idx is None
        and tuple(out.shape) == shape
    )
    work_out = (
        out if direct_out else torch.empty(shape, dtype=compute_dtype, device=A.device)
    )
    hipb_multistream_mm(
        work_a,
        work_b,
        counts,
        work_out,
        A_is_transposed,
        bias.to(dtype=compute_dtype).contiguous() if bias is not None else None,
        B_is_transposed,
    )
    if work_out.dtype != result_dtype:
        work_out = work_out.to(dtype=result_dtype)

    if out is None:
        if scatter_idx is None:
            return work_out
        out = torch.empty_like(work_out)
    if scatter_idx is not None:
        out.index_copy_(0, scatter_idx, work_out)
    elif work_out is not out:
        out.copy_(work_out)
    return out
