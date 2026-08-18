# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import ctypes

import pytest
import torch

from csrc.cpp_itfs.torch_utils import torch_to_c_types


@pytest.mark.skipif(not torch.cuda.is_available(), reason="A GPU is required")
def test_torch_to_c_types_cuda_stream():
    stream = torch.cuda.Stream()

    (stream_ptr,) = torch_to_c_types(stream)

    assert isinstance(stream_ptr, ctypes.c_void_p)
    assert stream_ptr.value == stream.cuda_stream


@pytest.mark.skipif(
    not torch.cuda.is_available() or not hasattr(torch, "Stream"),
    reason="A GPU and torch.Stream are required",
)
def test_torch_to_c_types_base_stream():
    stream = torch.Stream(device="cuda")
    assert type(stream) is torch.Stream

    (stream_ptr,) = torch_to_c_types(stream)
    cuda_stream = torch.cuda.Stream(
        stream_id=stream.stream_id,
        device_index=stream.device_index,
        device_type=stream.device_type,
    )

    assert isinstance(stream_ptr, ctypes.c_void_p)
    assert stream_ptr.value == cuda_stream.cuda_stream
