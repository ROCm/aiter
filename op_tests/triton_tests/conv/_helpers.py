# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Test-side library: TestSuite, method registry, and runners.

Library code (no ``test_`` prefix) — pytest does not collect this file.
``test_conv2d.py`` imports the runners and registry from here.

Public surface:

- ``TestSuite`` / ``TestResult``
    Correctness collector. ``check_close`` records pass/fail per case;
    ``failed_results`` returns the list at end of run.

- ``METHOD_REGISTRY`` (+ ``ORDERED_METHODS`` / ``ALL_METHODS``)
    Kernel dispatch table. Each entry maps a method name to its
    public ``conv2d_*`` callable, applicability guard, winograd flag,
    and bench tag.

- ``run_all_methods(...)``
    Main dispatch. For a given (x, w, b, stride, padding, dilation):
    runs the selected NCHW kernel (or all of them, if method="all"),
    runs ``conv2d_nhwc`` if layout_mode includes nhwc, and checks every
    output against ``F.conv2d`` within method-appropriate tolerance.

- ``run_edge_cases``, ``run_random_fuzzing``, ``run_no_bias``,
  ``run_activations``, ``run_cross_method``
    Test runners called by ``test_conv2d.py``. Each iterates a shape
    set and delegates to ``run_all_methods``.

- ``COMMON_SHAPES``, ``get_edge_case_shapes()``
    Shared shape data — 3x3 stride-1 shapes routable by every kernel,
    and the 12-shape edge-case list respectively.
"""

import random
import traceback
from collections import namedtuple
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn.functional as F

from aiter.ops.triton.conv._utils import (
    dynamic_conv_tolerances,
    _out_hw,
    _is_1x1_conv,
    _is_3x3_conv,
    _winograd_tolerances,
    apply_activation,
)
from aiter.ops.triton.conv._launch import _select_3x3_method
from aiter.ops.triton.conv.conv2d import (
    conv2d_nchw,
    conv2d_nchw_cblocked,
    conv2d_nhwc,
    conv2d_winograd_f4x3,
    conv2d_winograd_f4x3_fused,
    conv2d_winograd_f4x3_cblocked,
)

# -- Architecture gating ------------------------------------------------------
SUPPORTED_ARCHS = {
    "RDNA": {"gfx1200", "gfx1201"},
    "CDNA": set(),
}

# Flat union for arch-check use sites.
ALL_SUPPORTED_ARCHS = set().union(*SUPPORTED_ARCHS.values())


# -- Method registry ----------------------------------------------------------

MethodEntry = namedtuple(
    "MethodEntry", ["kernel_fn", "guard_fn", "is_winograd", "bench_tag", "short_name"]
)


def _3x3_guard(R, S, stride, dilation, C):
    return _is_3x3_conv(R, S)


def _wino_guard(R, S, stride, dilation, C):
    # _is_winograd_eligible signature varies by upstream — keep the flag tight
    from aiter.ops.triton.conv._utils import _is_winograd_eligible

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


# -- Result + suite -----------------------------------------------------------


@dataclass
class TestResult:
    name: str
    passed: bool
    max_abs_error: float
    rel_error: float
    message: str = ""


class TestSuite:
    """Correctness-only test runner. No bench records, no MIOpen tables."""

    __test__ = False  # not a pytest TestCase — leading "Test" is incidental

    def __init__(
        self,
        device: str,
        dtype: torch.dtype,
        verbose: bool = False,
        print_shapes: bool = False,
        layout_mode: str = "both",
    ):
        self.device = torch.device(device)
        self.dtype = dtype
        self.verbose = verbose
        self.print_shapes = print_shapes
        self.layout_mode = layout_mode
        self.results: List[TestResult] = []

    def check_close(
        self,
        name: str,
        got: torch.Tensor,
        ref: torch.Tensor,
        K_red: Optional[int] = None,
        rtol: Optional[float] = None,
        atol: Optional[float] = None,
    ) -> TestResult:
        got32 = got.float()
        ref32 = ref.float()
        diff = (got32 - ref32).abs()
        max_abs = float(diff.max().item()) if diff.numel() else 0.0
        rel = max_abs / (float(ref32.abs().max().item()) + 1e-6)
        if rtol is None or atol is None:
            K_est = int(K_red) if K_red is not None else 1024
            rtol_calc, atol_calc = dynamic_conv_tolerances(self.dtype, K_est, ref32)
            rtol = rtol if rtol is not None else rtol_calc
            atol = atol if atol is not None else atol_calc
        try:
            torch.testing.assert_close(got32, ref32, rtol=rtol, atol=atol)
            passed = True
            msg = "OK"
        except AssertionError as e:
            passed = False
            msg = str(e).split("\n")[0]
        res = TestResult(name, passed, max_abs, rel, msg)
        self.results.append(res)
        if self.verbose:
            mark = "✓" if passed else "✗"
            print(f"  {mark} {name:<40} | max_abs={max_abs:.3e} rel={rel:.3e}")
        return res

    def all_passed(self) -> bool:
        return all(r.passed for r in self.results)

    def failed_results(self) -> List[TestResult]:
        return [r for r in self.results if not r.passed]


# -- Tolerance + dispatch -----------------------------------------------------


def _get_tolerances(
    method_name, entry, suite, y_ref, N, C, H, W, K_out, R, S, stride, dilation
):
    if entry.is_winograd:
        return _winograd_tolerances(suite.dtype, C * R * S, y_ref, "f4x3")
    if method_name == "default" and _is_3x3_conv(R, S):
        routed = _select_3x3_method(N, C, H, W, K_out, stride, dilation)
        if routed and "winograd" in routed:
            return _winograd_tolerances(suite.dtype, C * R * S, y_ref, "f4x3")
    return dynamic_conv_tolerances(suite.dtype, C * R * S, y_ref)


def run_all_methods(
    suite: TestSuite,
    x: torch.Tensor,
    w: torch.Tensor,
    b: Optional[torch.Tensor],
    stride,
    padding,
    dilation,
    name: str,
    method: str = "default",
    activation: str = "none",
):
    """Correctness-only dispatch: run selected method(s) and check vs F.conv2d."""
    N, C, H, W_in = x.shape
    K_out, _, R, S = w.shape

    y_ref = F.conv2d(
        x,
        w,
        b.to(dtype=suite.dtype) if b is not None else None,
        stride=stride,
        padding=padding,
        dilation=dilation,
    )
    y_ref = apply_activation(y_ref, activation)

    if suite.print_shapes:
        if _is_1x1_conv(R, S, dilation):
            kernel_type = "[1x1]"
        elif _is_3x3_conv(R, S):
            kernel_type = "[3x3]"
        else:
            kernel_type = "[general]"
        print(
            f"    {name} {kernel_type}: X{tuple(x.shape)} W{tuple(w.shape)} -> Y{tuple(y_ref.shape)}"
        )

    if suite.layout_mode in ("nchw", "both"):
        methods_to_run = ORDERED_METHODS if method == "all" else [method]
        for m in methods_to_run:
            entry = METHOD_REGISTRY[m]
            if entry.guard_fn and not entry.guard_fn(R, S, stride, dilation, C):
                continue
            y_tri = entry.kernel_fn(
                x,
                w,
                b,
                stride,
                padding,
                dilation,
                activation=activation,
            )
            rtol, atol = _get_tolerances(
                m, entry, suite, y_ref, N, C, H, W_in, K_out, R, S, stride, dilation
            )
            suite.check_close(
                f"{name} {entry.bench_tag or '[NCHW]'}",
                y_tri,
                y_ref,
                rtol=rtol,
                atol=atol,
            )

    if suite.layout_mode in ("nhwc", "both"):
        y_nhwc = conv2d_nhwc(
            x,
            w,
            b,
            stride,
            padding,
            dilation,
            activation=activation,
        )
        if _is_3x3_conv(R, S):
            nhwc_method = _select_3x3_method(N, C, H, W_in, K_out, stride, dilation)
            if nhwc_method in ("winograd_f4x3", "winograd_f4x3_cblocked"):
                _r, _a = _winograd_tolerances(suite.dtype, C * R * S, y_ref, "f4x3")
                suite.check_close(f"{name} [NHWC]", y_nhwc, y_ref, rtol=_r, atol=_a)
            else:
                suite.check_close(f"{name} [NHWC]", y_nhwc, y_ref, K_red=C * R * S)
        else:
            suite.check_close(f"{name} [NHWC]", y_nhwc, y_ref, K_red=C * R * S)


# -- Shape sets ---------------------------------------------------------------

# Shapes routable by ALL 5 NCHW kernels (3x3, stride=1, padding=1, dilation=1,
# C >= 4). Used by test_cross_method to verify every kernel produces the same
# result (within tolerance) on the same input.
COMMON_SHAPES = [
    (1, 64, 56, 56, 64, 3, 3, (1, 1), (1, 1), (1, 1), "common 64ch/56sp"),
    (1, 128, 28, 28, 128, 3, 3, (1, 1), (1, 1), (1, 1), "common 128ch/28sp"),
    (1, 256, 14, 14, 256, 3, 3, (1, 1), (1, 1), (1, 1), "common 256ch/14sp"),
]


# -- Edge case shapes ---------------------------------------------------------


def get_edge_case_shapes():
    return [
        (1, 3, 7, 7, 8, 3, 3, (1, 1), (1, 1), (1, 1), "3x3 same padding"),
        (1, 3, 8, 8, 16, 1, 1, (1, 1), (0, 0), (1, 1), "1x1 stride1"),
        (2, 16, 32, 32, 32, 3, 3, (2, 2), (1, 1), (1, 1), "stride2"),
        (2, 32, 17, 23, 64, 5, 5, (2, 2), (2, 2), (1, 1), "odd dims + pad"),
        (4, 64, 28, 28, 128, 3, 3, (1, 1), (0, 0), (2, 2), "dilation2"),
        (2, 512, 7, 7, 1024, 1, 1, (1, 1), (0, 0), (1, 1), "1x1 large channels"),
        (1, 3, 112, 112, 64, 7, 7, (2, 2), (3, 3), (1, 1), "7x7 large spatial"),
        (1, 1, 16, 16, 16, 3, 3, (1, 1), (1, 1), (1, 1), "single input channel"),
        (2, 64, 8, 8, 64, 3, 3, (1, 1), (1, 1), (1, 1), "small spatial 3x3"),
        (1, 128, 4, 4, 256, 1, 1, (1, 1), (0, 0), (1, 1), "1x1 tiny spatial"),
        (2, 32, 32, 32, 32, 3, 3, (1, 1), (0, 0), (1, 1), "3x3 no padding"),
        (2, 64, 28, 28, 128, 3, 3, (2, 2), (1, 1), (1, 1), "3x3 stride2 standard"),
    ]


# -- Test runners (no test_ prefix; pytest will not collect this file) --------


def run_edge_cases(suite: TestSuite, activation: str = "none", method: str = "default"):
    for (
        N,
        C,
        H,
        W,
        K_out,
        R,
        S,
        stride,
        padding,
        dilation,
        desc,
    ) in get_edge_case_shapes():
        P, Q = _out_hw(H, W, R, S, stride, padding, dilation)
        if P < 1 or Q < 1:
            continue
        x = torch.randn((N, C, H, W), device=suite.device, dtype=suite.dtype)
        w = torch.randn((K_out, C, R, S), device=suite.device, dtype=suite.dtype)
        b = torch.randn((K_out,), device=suite.device, dtype=suite.dtype)
        run_all_methods(
            suite,
            x,
            w,
            b,
            stride,
            padding,
            dilation,
            name=desc,
            method=method,
            activation=activation,
        )


def run_activations(
    suite: TestSuite, method: str = "default", activation: str = "relu"
):
    N, C, H, W, K_out = 2, 32, 16, 16, 64
    R, S = 3, 3
    stride, padding, dilation = (1, 1), (1, 1), (1, 1)
    x = torch.randn((N, C, H, W), device=suite.device, dtype=suite.dtype)
    w = torch.randn((K_out, C, R, S), device=suite.device, dtype=suite.dtype)
    b = torch.randn((K_out,), device=suite.device, dtype=suite.dtype)
    run_all_methods(
        suite,
        x,
        w,
        b,
        stride,
        padding,
        dilation,
        name=f"activation_{activation}_{method}",
        method=method,
        activation=activation,
    )


def run_no_bias(suite: TestSuite, method: str = "default"):
    shapes = [
        (1, 64, 8, 8, 128, 1, 1, (1, 1), (0, 0), (1, 1), "1x1 no bias"),
        (2, 32, 16, 16, 64, 3, 3, (1, 1), (1, 1), (1, 1), "3x3 no bias"),
        (1, 16, 8, 8, 32, 5, 5, (1, 1), (2, 2), (1, 1), "5x5 no bias"),
    ]
    for N, C, H, W, K_out, R, S, stride, padding, dilation, desc in shapes:
        x = torch.randn((N, C, H, W), device=suite.device, dtype=suite.dtype)
        w = torch.randn((K_out, C, R, S), device=suite.device, dtype=suite.dtype)
        run_all_methods(
            suite,
            x,
            w,
            None,
            stride,
            padding,
            dilation,
            name=desc,
            method=method,
        )


def run_cross_method(suite: TestSuite):
    """Run every NCHW-applicable kernel on shapes that all 5 can handle.

    Each kernel is checked against F.conv2d. Transitivity gives us
    cross-kernel equivalence: if kernel A and B both match the same
    F.conv2d output within tolerance, they match each other within ~2x.
    """
    for N, C, H, W, K_out, R, S, stride, padding, dilation, desc in COMMON_SHAPES:
        x = torch.randn((N, C, H, W), device=suite.device, dtype=suite.dtype)
        w = torch.randn((K_out, C, R, S), device=suite.device, dtype=suite.dtype)
        b = torch.randn((K_out,), device=suite.device, dtype=suite.dtype)
        run_all_methods(
            suite,
            x,
            w,
            b,
            stride,
            padding,
            dilation,
            name=desc,
            method="all",  # iterates every kernel in ORDERED_METHODS
        )


def run_random_fuzzing(
    suite: TestSuite,
    num_tests: int = 10,
    activation: str = "none",
    method: str = "default",
    seed: int = 42,
):
    """Bounded random shape sweep, seeded for reproducibility.

    Default num_tests=10 keeps CI cheap; callers can pass a larger value
    for ad-hoc development sweeps.
    """
    random.seed(seed)
    for i in range(num_tests):
        N = random.randint(1, 8)
        C = random.choice([1, 3, 16, 32, 64, 128, 256])
        H = random.randint(4, 64)
        W = random.randint(4, 64)
        K_out = random.choice([16, 32, 64, 128, 256])
        R = random.randint(1, min(7, H))
        S = random.randint(1, min(7, W))
        sh = random.randint(1, 3)
        sw = random.randint(1, 3)
        ph = random.randint(0, R // 2)
        pw = random.randint(0, S // 2)
        dh = random.randint(1, 2)
        dw = random.randint(1, 2)
        P, Q = _out_hw(H, W, R, S, (sh, sw), (ph, pw), (dh, dw))
        if P < 1 or Q < 1:
            continue
        try:
            x = torch.randn((N, C, H, W), device=suite.device, dtype=suite.dtype)
            w = torch.randn((K_out, C, R, S), device=suite.device, dtype=suite.dtype)
            b = torch.randn((K_out,), device=suite.device, dtype=suite.dtype)
            tag = f"Random[{i}] ({N},{C},{H},{W})->({N},{K_out},{P},{Q})"
            run_all_methods(
                suite,
                x,
                w,
                b,
                (sh, sw),
                (ph, pw),
                (dh, dw),
                name=tag,
                method=method,
                activation=activation,
            )
        except Exception as e:
            tb = traceback.format_exc()
            suite.results.append(
                TestResult(
                    f"Random[{i}]",
                    False,
                    float("inf"),
                    float("inf"),
                    f"{type(e).__name__}: {e}\n{tb}",
                )
            )
