# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import ctypes

import torch

from aiter.jit import core as _aiter_jit_core  # noqa: F401  # puts csrc on sys.path
from csrc.cpp_itfs.torch_utils import torch_to_c_types


def _as_base_stream(cuda_stream):
    stream = torch.Stream(
        stream_id=cuda_stream.stream_id,
        device_index=cuda_stream.device_index,
        device_type=cuda_stream.device_type,
    )
    assert type(stream) is torch.Stream
    assert not isinstance(stream, torch.cuda.Stream)
    return stream


def _assert_same_handle(stream_ptr, native_handle):
    assert isinstance(stream_ptr, ctypes.c_void_p)
    # ctypes.c_void_p(0).value is None; the default CUDA stream handle is 0.
    assert (stream_ptr.value or 0) == (native_handle or 0)


def test_torch_to_c_types_cuda_stream():
    stream = torch.cuda.Stream()
    (stream_ptr,) = torch_to_c_types(stream)
    _assert_same_handle(stream_ptr, stream.cuda_stream)


def test_torch_to_c_types_base_stream():
    cuda_stream = torch.cuda.Stream()
    expected = cuda_stream.cuda_stream
    (stream_ptr,) = torch_to_c_types(_as_base_stream(cuda_stream))
    _assert_same_handle(stream_ptr, expected)


def test_torch_to_c_types_current_stream_as_base_stream():
    # pa_ragged passes current_stream(); Dynamo may rebuild it as torch.Stream.
    cuda_stream = torch.cuda.current_stream()
    expected = cuda_stream.cuda_stream
    (stream_ptr,) = torch_to_c_types(_as_base_stream(cuda_stream))
    _assert_same_handle(stream_ptr, expected)


def test_torch_to_c_types_cpu_stream_rejected():
    stream = torch.Stream(device="cpu")
    try:
        torch_to_c_types(stream)
    except ValueError as err:
        assert "Unsupported type" in str(err)
        return
    raise AssertionError("expected ValueError for a CPU torch.Stream")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("A GPU is required")
    test_torch_to_c_types_cuda_stream()
    if hasattr(torch, "Stream"):
        test_torch_to_c_types_base_stream()
        test_torch_to_c_types_current_stream_as_base_stream()
        test_torch_to_c_types_cpu_stream_rejected()
    print("ALL_PASS")
