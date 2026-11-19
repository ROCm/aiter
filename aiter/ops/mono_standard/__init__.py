# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Explicit gfx1250 MONO operators, built lazily on first invocation.

The standalone entry points use grid (16, 16, 1), block (128, 1, 1).
See docs/mono-rope.md for the supported shapes and device-call ABI.
"""


def rope_quant_window(*args, **kwargs):
    from .loader import load_a05
    return load_a05().rope_quant_window(*args, **kwargs)


def inverse_rope_group_quant(*args, **kwargs):
    from .loader import load_a07
    return load_a07()(*args, **kwargs)


KERNELS = {"A05": rope_quant_window, "A07": inverse_rope_group_quant}
