# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""The FlyDSL AOT dry run answers the device queries of the host code it replays."""

from aiter.aot.flydsl.dry_run import dry_run
from aiter.ops.flydsl.kernels import tensor_shim


def test_current_stream_arg_stays_off_the_device(monkeypatch):
    def no_device(_device):
        raise RuntimeError("no ROCm-capable device is detected")

    # A wheel build host has no GPU, where the raw stream query fails.
    monkeypatch.setattr(tensor_shim, "_get_current_raw_stream", no_device)
    with dry_run("gfx1250", 256):
        tensor_shim.current_stream_arg()
