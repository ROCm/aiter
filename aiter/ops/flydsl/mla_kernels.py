# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.runtime.device import get_rocm_arch

from .utils import addressable_lds_bytes_for_gfx

__all__ = [
    "flydsl_mla_pagesize1_fp8_fp8",
    "flydsl_mla_pagesize64_fp8_fp8",
]


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _require_runtime(condition, message):
    if not condition:
        raise RuntimeError(message)


def _require_tensor(name, value):
    _require(
        isinstance(value, torch.Tensor),
        f"{name}: expected torch.Tensor, got {type(value).__name__}",
    )


def _require_cuda_tensor(name, tensor, device):
    _require(tensor.is_cuda, f"{name}: expected CUDA tensor")
    _require(
        tensor.device == device,
        f"{name}: expected device {device}, got {tensor.device}",
    )


def _validate_pagesize1_inputs(
    split_data,
    split_lse,
    q,
    kv_buffer,
    kv_page_indices,
    work_indptr,
    work_info,
    softmax_scale,
):
    arch = str(get_rocm_arch() or "").split(":", 1)[0]
    _require_runtime(
        arch == "gfx1250",
        f"expected gfx1250, got {arch or 'unknown'}",
    )

    for name, value in (
        ("q", q),
        ("kv_buffer", kv_buffer),
        ("split_data", split_data),
        ("split_lse", split_lse),
        ("kv_page_indices", kv_page_indices),
        ("work_indptr", work_indptr),
        ("work_info", work_info),
    ):
        _require_tensor(name, value)

    _require(
        not isinstance(softmax_scale, torch.Tensor),
        "softmax_scale: expected an ordinary Python real number, got a "
        "torch.Tensor (reading its value here would force a host sync)",
    )
    softmax_scale = float(softmax_scale)

    _require(q.is_cuda, "q: expected CUDA tensor")
    device = q.device
    for name, tensor in (
        ("split_data", split_data),
        ("split_lse", split_lse),
        ("kv_buffer", kv_buffer),
        ("kv_page_indices", kv_page_indices),
        ("work_indptr", work_indptr),
        ("work_info", work_info),
    ):
        _require_cuda_tensor(name, tensor, device)

    batch = q.size(0)
    _require(
        q.ndim == 3 and q.shape == (batch, 128, 576),
        f"q: expected shape [{batch}, 128, 576], got {list(q.shape)}",
    )
    _require(
        q.dtype == torch.float8_e4m3fn,
        f"q: expected torch.float8_e4m3fn, got {q.dtype}",
    )
    _require(
        q.is_contiguous(),
        f"q: expected contiguous tensor, got stride {list(q.stride())}",
    )

    _require(
        kv_buffer.ndim == 4 and tuple(kv_buffer.shape[1:]) == (1, 1, 576),
        "kv_buffer: expected shape [num_pages, 1, 1, 576], "
        f"got {list(kv_buffer.shape)}",
    )
    _require(
        kv_buffer.dtype == torch.float8_e4m3fn,
        f"kv_buffer: expected torch.float8_e4m3fn, got {kv_buffer.dtype}",
    )
    _require(
        kv_buffer.is_contiguous(),
        f"kv_buffer: expected contiguous tensor, got stride {list(kv_buffer.stride())}",
    )

    _require(
        split_data.ndim == 3 and tuple(split_data.shape[1:]) == (128, 512),
        f"split_data: expected shape [num_partials, 128, 512], got {list(split_data.shape)}",
    )
    _require(
        split_data.dtype == torch.float32,
        f"split_data: expected torch.float32, got {split_data.dtype}",
    )
    _require(
        split_data.is_contiguous(),
        f"split_data: expected contiguous tensor, got stride {list(split_data.stride())}",
    )

    _require(
        split_lse.shape == (split_data.shape[0], 128),
        "split_lse: expected shape "
        f"[{split_data.shape[0]}, 128], got {list(split_lse.shape)}",
    )
    _require(
        split_lse.dtype == torch.float32,
        f"split_lse: expected torch.float32, got {split_lse.dtype}",
    )
    _require(
        split_lse.is_contiguous(),
        f"split_lse: expected contiguous tensor, got stride {list(split_lse.stride())}",
    )

    _require(
        kv_page_indices.ndim == 1,
        f"kv_page_indices: expected 1D tensor, got {kv_page_indices.ndim}D",
    )
    _require(
        kv_page_indices.dtype == torch.int32,
        f"kv_page_indices: expected torch.int32, got {kv_page_indices.dtype}",
    )
    _require(
        kv_page_indices.is_contiguous(),
        "kv_page_indices: expected contiguous tensor, got stride "
        f"{list(kv_page_indices.stride())}",
    )
    _require(
        work_indptr.ndim == 1,
        f"work_indptr: expected 1D tensor, got {work_indptr.ndim}D",
    )
    _require(
        work_indptr.dtype == torch.int32,
        f"work_indptr: expected torch.int32, got {work_indptr.dtype}",
    )
    _require(
        work_indptr.is_contiguous(),
        f"work_indptr: expected contiguous tensor, got stride {list(work_indptr.stride())}",
    )
    _require(
        work_info.ndim == 2 and work_info.shape[1] == 8,
        f"work_info: expected shape [num_works, 8], got {list(work_info.shape)}",
    )
    _require(
        work_info.dtype == torch.int32,
        f"work_info: expected torch.int32, got {work_info.dtype}",
    )
    _require(
        work_info.is_contiguous(),
        f"work_info: expected contiguous tensor, got stride {list(work_info.stride())}",
    )

    num_cus = work_indptr.numel() - 1
    _require(
        num_cus > 0,
        "work_indptr: expected work_indptr.numel() - 1 (num_cus) to be "
        f"positive, got {work_indptr.numel()} entries",
    )

    properties = torch.cuda.get_device_properties(device)
    lds_size = getattr(properties, "shared_memory_per_multiprocessor", None)
    lds_size = (
        int(lds_size) if lds_size is not None else addressable_lds_bytes_for_gfx(arch)
    )
    return num_cus, lds_size, softmax_scale


def flydsl_mla_pagesize1_fp8_fp8(
    split_data,
    split_lse,
    q,
    kv_buffer,
    kv_page_indices,
    work_indptr,
    work_info,
    softmax_scale,
    *,
    stream=None,
):
    num_cus, lds_size, softmax_scale = _validate_pagesize1_inputs(
        split_data,
        split_lse,
        q,
        kv_buffer,
        kv_page_indices,
        work_indptr,
        work_info,
        softmax_scale,
    )
    from .kernels.mla_gfx1250.mla_pagesize1_fp8_fp8 import (
        launch_mla_pagesize1_fp8_fp8,
    )

    if stream is None:
        stream = torch.cuda.current_stream(q.device)
    launch_mla_pagesize1_fp8_fp8(
        flyc.from_c_void_p(fx.Float32, split_data.data_ptr()),
        flyc.from_c_void_p(fx.Float32, split_lse.data_ptr()),
        flyc.from_c_void_p(fx.BFloat16, None),
        flyc.from_c_void_p(fx.Int8, q.data_ptr()),
        flyc.from_c_void_p(fx.Int8, kv_buffer.data_ptr()),
        flyc.from_c_void_p(fx.Int32, kv_page_indices.data_ptr()),
        flyc.from_c_void_p(fx.Int32, work_indptr.data_ptr()),
        flyc.from_c_void_p(fx.Int32, work_info.data_ptr()),
        softmax_scale,
        kv_buffer.size(0),
        num_cus,
        lds_size,
        stream=stream,
    )


def _validate_pagesize64_inputs(
    split_data,
    split_lse,
    q,
    kv_buffer,
    kv_indptr,
    kv_page_indices,
    kv_last_page_lens,
    qo_indptr,
    num_kv_splits_indptr,
    q_scale,
    kv_scale,
    softmax_scale,
    num_splits,
    page_size,
):
    arch = str(get_rocm_arch() or "").split(":", 1)[0]
    _require(arch == "gfx1250", f"expected gfx1250, got {arch or 'unknown'}")
    _require(
        isinstance(num_splits, int) and not isinstance(num_splits, bool),
        f"num_splits: expected int, got {type(num_splits).__name__}",
    )
    _require(num_splits > 0, f"num_splits: expected positive value, got {num_splits}")

    from .kernels.mla_gfx1250.mla_pagesize64_fp8_fp8 import (
        KV_PAGE_ELEMENTS,
        NUM_Q_HEADS,
        PAGE_SIZE,
        Q_HEAD_STRIDE,
        Q_ROW_STRIDE,
        QK_HEAD_DIM,
        V_HEAD_DIM,
    )

    _require(
        page_size == PAGE_SIZE, f"page_size: expected {PAGE_SIZE}, got {page_size}"
    )
    _require(
        q.dtype == torch.float8_e4m3fn,
        f"q: expected torch.float8_e4m3fn, got {q.dtype}",
    )
    _require(
        q.ndim == 3 and tuple(q.shape[1:]) == (NUM_Q_HEADS, QK_HEAD_DIM),
        f"q: expected [total_q, {NUM_Q_HEADS}, {QK_HEAD_DIM}], got {list(q.shape)}",
    )
    expected_q_stride = (Q_ROW_STRIDE, Q_HEAD_STRIDE, 1)
    _require(
        tuple(q.stride()) == expected_q_stride,
        f"q: expected padded stride {list(expected_q_stride)}, got {list(q.stride())}",
    )
    batch = q.size(0)
    _require(batch > 0, "q: total_q/batch must be positive")

    _require(
        kv_buffer.dtype == torch.float8_e4m3fn,
        f"kv_buffer: expected torch.float8_e4m3fn, got {kv_buffer.dtype}",
    )
    _require(
        kv_buffer.ndim == 2 and kv_buffer.size(1) == KV_PAGE_ELEMENTS,
        "kv_buffer: expected a segmented 2D page view "
        f"[num_pages, {KV_PAGE_ELEMENTS}]; token-major 4D tensors are not accepted",
    )
    _require(kv_buffer.size(0) > 0, "kv_buffer: num_pages must be positive")
    _require(
        tuple(kv_buffer.stride()) == (KV_PAGE_ELEMENTS, 1),
        "kv_buffer: expected contiguous segmented pages with "
        f"page stride 0x{KV_PAGE_ELEMENTS:x}, got stride {list(kv_buffer.stride())}",
    )

    int32_inputs = {
        "kv_indptr": (kv_indptr, (batch + 1,)),
        "kv_page_indices": (kv_page_indices, None),
        "kv_last_page_lens": (kv_last_page_lens, (batch,)),
        "qo_indptr": (qo_indptr, (batch + 1,)),
        "num_kv_splits_indptr": (num_kv_splits_indptr, (batch + 1,)),
    }
    for name, (tensor, expected_shape) in int32_inputs.items():
        _require(
            tensor.dtype == torch.int32,
            f"{name}: expected torch.int32, got {tensor.dtype}",
        )
        _require(
            tensor.ndim == 1,
            f"{name}: expected a 1D tensor, got shape {list(tensor.shape)}",
        )
        if expected_shape is not None:
            _require(
                tuple(tensor.shape) == expected_shape,
                f"{name}: expected shape {list(expected_shape)}, got {list(tensor.shape)}",
            )
        _require(tensor.is_contiguous(), f"{name}: expected a contiguous tensor")
    _require(kv_page_indices.numel() > 0, "kv_page_indices: must not be empty")

    for name, scale in (("q_scale", q_scale), ("kv_scale", kv_scale)):
        _require(
            scale.dtype == torch.float32,
            f"{name}: expected torch.float32, got {scale.dtype}",
        )
        _require(
            tuple(scale.shape) == (1,),
            f"{name}: expected shape [1], got {list(scale.shape)}",
        )
        _require(scale.is_contiguous(), f"{name}: expected a contiguous tensor")

    if num_splits == 1:
        expected_data_shape = (batch, NUM_Q_HEADS, V_HEAD_DIM)
        expected_data_dtype = torch.bfloat16
    else:
        expected_data_shape = (batch, num_splits, NUM_Q_HEADS, V_HEAD_DIM)
        expected_data_dtype = torch.float32
    _require(
        split_data.dtype == expected_data_dtype,
        f"split_data: expected {expected_data_dtype} for num_splits={num_splits}, "
        f"got {split_data.dtype}",
    )
    _require(
        tuple(split_data.shape) == expected_data_shape,
        f"split_data: expected shape {list(expected_data_shape)}, "
        f"got {list(split_data.shape)}",
    )
    _require(split_data.is_contiguous(), "split_data: expected a contiguous tensor")

    expected_lse_shape = (batch, num_splits, NUM_Q_HEADS, 1)
    _require(
        split_lse.dtype == torch.float32,
        f"split_lse: expected torch.float32, got {split_lse.dtype}",
    )
    _require(
        tuple(split_lse.shape) == expected_lse_shape,
        f"split_lse: expected shape {list(expected_lse_shape)}, "
        f"got {list(split_lse.shape)}",
    )
    _require(split_lse.is_contiguous(), "split_lse: expected a contiguous tensor")
    return batch


def flydsl_mla_pagesize64_fp8_fp8(
    split_data,
    split_lse,
    q,
    kv_buffer,
    kv_indptr,
    kv_page_indices,
    kv_last_page_lens,
    qo_indptr,
    num_kv_splits_indptr,
    q_scale,
    kv_scale,
    softmax_scale,
    num_splits,
    *,
    page_size=64,
    stream=None,
):
    batch = _validate_pagesize64_inputs(
        split_data,
        split_lse,
        q,
        kv_buffer,
        kv_indptr,
        kv_page_indices,
        kv_last_page_lens,
        qo_indptr,
        num_kv_splits_indptr,
        q_scale,
        kv_scale,
        softmax_scale,
        num_splits,
        page_size,
    )
    from .kernels.mla_gfx1250.mla_pagesize64_fp8_fp8 import (
        launch_mla_pagesize64_fp8_fp8,
    )

    if stream is None:
        stream = torch.cuda.current_stream(q.device)
    output_type = fx.BFloat16 if num_splits == 1 else fx.Float32
    launch_mla_pagesize64_fp8_fp8(
        flyc.from_c_void_p(output_type, split_data.data_ptr()),
        flyc.from_c_void_p(fx.Float32, split_lse.data_ptr()),
        flyc.from_c_void_p(fx.Int8, q.data_ptr()),
        flyc.from_c_void_p(fx.Int8, kv_buffer.data_ptr()),
        flyc.from_c_void_p(fx.Int32, kv_indptr.data_ptr()),
        flyc.from_c_void_p(fx.Int32, kv_page_indices.data_ptr()),
        flyc.from_c_void_p(fx.Int32, kv_last_page_lens.data_ptr()),
        flyc.from_c_void_p(fx.Int32, qo_indptr.data_ptr()),
        flyc.from_c_void_p(fx.Int32, num_kv_splits_indptr.data_ptr()),
        flyc.from_c_void_p(fx.Float32, q_scale.data_ptr()),
        flyc.from_c_void_p(fx.Float32, kv_scale.data_ptr()),
        float(softmax_scale),
        batch,
        num_splits,
        int(num_splits == 1),
        stream=stream,
    )
