# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Adapted from ROCm/ATOM (https://github.com/ROCm/ATOM, MIT License) at 97359d6:
# atom/mono/device/stamps.py
"""A timeline build's per-CTA phase stamps (profiling only): the s_memrealtime
(100 MHz) at which the CTA passes point k, kept in LDS until the kernel ends -- a global store per
stamp would sit on every later ``s_waitcnt vmcnt(0)``.

``on`` is the build's compile-time switch: off, every call traces to nothing.
"""

import flydsl.expr as fx
from flydsl.expr import const_expr

from aiter.ops.flydsl.kernels.dsv41_mono.common.device.ops import memrealtime, traced


@traced
def stamp(on, tls, tid, k):
    # compile-time gate outside, traced condition inside: they cannot be one `and`
    if const_expr(on):  # noqa: SIM102
        if tid == 0:
            fx.ptr_store(memrealtime(), tls + k)
