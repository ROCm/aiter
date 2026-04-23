# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Method registry — single source of truth for kernel dispatch."""

from __future__ import annotations
from collections import namedtuple
from aiter.ops.triton.conv.conv2d import (
    conv2d_nchw,
    conv2d_nchw_cblocked,
    conv2d_winograd_f4x3,
    conv2d_winograd_f4x3_fused,
    conv2d_winograd_f4x3_cblocked,
)
from aiter.ops.triton.conv._utils import _is_3x3_conv, _is_winograd_eligible

MethodEntry = namedtuple(
    "MethodEntry", ["kernel_fn", "guard_fn", "is_winograd", "bench_tag", "short_name"]
)


def _3x3_guard(R, S, stride, dilation, C):
    return _is_3x3_conv(R, S)


def _wino_guard(R, S, stride, dilation, C):
    return _is_winograd_eligible(R, S, stride, dilation, C)


METHOD_REGISTRY = {
    "default": MethodEntry(conv2d_nchw, None, False, "", "default"),
    "cblocked": MethodEntry(
        conv2d_nchw_cblocked, _3x3_guard, False, "[cblocked]", "cblocked"
    ),
    "winograd_f4x3": MethodEntry(
        conv2d_winograd_f4x3, _wino_guard, True, "[winograd_f4x3]", "WF(4,3)"
    ),
    "winograd_f4x3_fused": MethodEntry(
        conv2d_winograd_f4x3_fused, _wino_guard, True, "[wino_f4x3_fused]", "WF4fused"
    ),
    "winograd_f4x3_cblocked": MethodEntry(
        conv2d_winograd_f4x3_cblocked,
        _wino_guard,
        True,
        "[winograd_f4x3_cblocked]",
        "WF4cb",
    ),
}

ORDERED_METHODS = list(METHOD_REGISTRY.keys())
ALL_METHODS = ORDERED_METHODS + ["all"]
