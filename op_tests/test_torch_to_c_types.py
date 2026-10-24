# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""torch_to_c_types must pass the raw HIP stream handle for every stream type.

torch.cuda.Stream exposes it as .cuda_stream; a device-agnostic torch.Stream
(e.g. from torch.accelerator.current_stream()) does not and is rebuilt from
its (stream_id, device_index, device_type).
"""

import ctypes

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a HIP device"
)


def _handle(stream) -> int:
    # torch_utils detects the GPU arch at import, so import it only when a
    # test actually runs; the skipif marker cannot guard a module-level import.
    from csrc.cpp_itfs.torch_utils import torch_to_c_types

    (c_arg,) = torch_to_c_types(stream)
    assert isinstance(c_arg, ctypes.c_void_p)
    return c_arg.value or 0


def test_default_cuda_stream_keeps_null_handle():
    # The default stream's handle is 0; it must not be mistaken for a missing one.
    stream = torch.cuda.default_stream()
    assert stream.cuda_stream == 0
    assert _handle(stream) == 0


def test_cuda_stream_handle():
    stream = torch.cuda.Stream()
    assert stream.cuda_stream != 0
    assert _handle(stream) == stream.cuda_stream


def test_generic_stream_matches_cuda_stream():
    cuda_stream = torch.cuda.Stream()
    with torch.cuda.stream(cuda_stream):
        generic = torch.accelerator.current_stream()
    assert type(generic) is torch.Stream
    assert not hasattr(generic, "cuda_stream")
    assert _handle(generic) == cuda_stream.cuda_stream


def test_generic_default_stream():
    with torch.cuda.stream(torch.cuda.default_stream()):
        generic = torch.accelerator.current_stream()
    assert type(generic) is torch.Stream
    assert _handle(generic) == 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
