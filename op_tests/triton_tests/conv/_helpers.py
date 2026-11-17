# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Shared convolution generators, assertions, method registry, and runners.

Library code (no ``test_`` prefix) — pytest does not collect this file.
Conv2D and Conv3D tests and benchmarks import shared code from here.

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

- ``assert_weight_pack_cache_clear_is_scoped(...)``
    Shared setup and assertions for the dimension-specific Conv2D and Conv3D
    weight-pack cache-clear tests.

- ``generate_conv_input(...)``, ``generate_conv_weights(...)``
    GPU input generators shared by the tests and benchmarks.

- ``assert_prepacked_pool_sizes_only_rotating_packed_inputs(...)``,
  ``assert_conv_benchmark_graph_packing_and_outputs(...)``
    Shared checks for rotating pools, packing, and graph replay.
"""

import csv
import math
import random
import traceback
from collections import namedtuple
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from aiter import logger
from aiter.ops.triton.conv._utils import (
    _is_1x1_conv,
    _is_3x3_conv,
    _is_winograd_2d_eligible,
    _out_hw,
)
from aiter.ops.triton.conv.conv2d import (
    _select_3x3_method,
    conv2d_nchw,
    conv2d_nchw_3x3_direct,
    conv2d_nchw_cblocked,
    conv2d_nhwc,
    conv2d_winograd_f4x3,
    conv2d_winograd_f4x3_cblocked,
)
from aiter.ops.triton.utils.conv_config_utils import has_conv_config

CONV2D_WEIGHT_PACK_CACHE_NAMES = (
    "_PACK_CACHE",
    "_PACK_CACHE_3x3",
    "_PACK_CACHE_WINOGRAD_F4X3",
)
CONV3D_WEIGHT_PACK_CACHE_NAMES = (
    "_PACK_CACHE_3D_GENERAL",
    "_PACK_CACHE_3D_3X3X3",
    "_PACK_CACHE_3D_WINOGRAD_HW",
)
_ALL_WEIGHT_PACK_CACHE_NAMES = (
    *CONV2D_WEIGHT_PACK_CACHE_NAMES,
    *CONV3D_WEIGHT_PACK_CACHE_NAMES,
)


def generate_conv_input(shape, dtype, layout):
    """Make one input in the requested physical layout on the GPU."""
    memory_format = {
        "nchw": torch.contiguous_format,
        "nhwc": torch.channels_last,
        "ncdhw": torch.contiguous_format,
        "ndhwc": torch.channels_last_3d,
    }[layout]
    input_tensor = torch.empty(
        shape, device="cuda", dtype=dtype, memory_format=memory_format
    ).normal_()
    return (input_tensor,)


def generate_conv_weights(shape, dtype, use_bias):
    """Make ordinary tensors so the convolution weight-pack cache can be used."""
    with torch.inference_mode(False):
        weight = torch.randn(shape, device="cuda", dtype=dtype)
        bias = (
            torch.randn((shape[0],), device="cuda", dtype=dtype) if use_bias else None
        )
    return weight, bias


def assert_weight_pack_cache_clear_is_scoped(
    monkeypatch,
    prepack_module,
    clear_caches,
    cleared_names,
):
    preserved_names = tuple(
        name for name in _ALL_WEIGHT_PACK_CACHE_NAMES if name not in cleared_names
    )
    caches = {}
    for name in (*cleared_names, *preserved_names):
        cache = prepack_module._LRUPackCache(maxsize=1)
        cache.put(name, object(), object())
        monkeypatch.setattr(prepack_module, name, cache)
        caches[name] = cache

    clear_caches()

    assert all(
        not caches[name]._d for name in cleared_names
    ), f"expected caches to be cleared: {cleared_names}"
    assert all(
        caches[name]._d for name in preserved_names
    ), f"expected caches to be preserved: {preserved_names}"


def dynamic_conv_tolerances(dtype: torch.dtype, K_red: int):
    eps = {
        torch.float16: 2**-10,
        torch.bfloat16: 2**-7,
        torch.float32: 2**-23,
    }.get(dtype, 2**-10)
    # rtol relaxes in steps as the reduction depth K_red (= C*R*S) grows: more
    # accumulated terms means more rounding, so the relative-error budget widens.
    # Breakpoints (1024, 4096) and values (6e-3 / 8e-3 / 1.2e-2) are empirical —
    # the lowest rtol that still holds across the fuzzer shape sweep at each depth.
    # See DESIGN.md section 8 for the full numerical model.
    rtol = 6e-3 if K_red < 1024 else (8e-3 if K_red < 4096 else 1.2e-2)
    # Error model: fp16 inputs multiplied pairwise have eps relative error per product.
    # Accumulated in fp32 over K_red terms, max absolute error grows as ~eps * sqrt(K_red).
    # The 10x multiplier covers worst-case accumulation ordering differences
    # between our Triton kernels and PyTorch reference.
    atol = max(eps * 8, 10.0 * eps * (K_red**0.5))
    return rtol, atol


def _winograd_tolerances(dtype, K_red, variant="f4x3"):
    """Return (rtol, atol) for Winograd F(4x4,3x3) correctness checks.
    Winograd transforms amplify fp16 rounding errors:
    - F(4x4,3x3): coefficients up to ±8, significant amplification
    """
    rtol, atol = dynamic_conv_tolerances(dtype, K_red)
    if variant == "f4x3":
        rtol *= 6.0
        atol = max(atol * 6.0, 0.6)
    return rtol, atol


def apply_activation(y: torch.Tensor, activation: str):
    if activation == "relu":
        return F.relu(y)
    if activation == "relu6":
        return torch.clamp(y, 0, 6)
    if activation == "gelu":
        return F.gelu(y, approximate="tanh")
    return y


def conv2d_padding_reference(
    source,
    weight,
    bias=None,
    stride=(1, 1),
    padding=(0, 0),
    dilation=(1, 1),
    padding_mode="zeros",
):
    """Reference convolution without using production padding preparation."""
    if padding_mode != "zeros":
        pad_height, pad_width = (
            (padding, padding) if isinstance(padding, int) else padding
        )
        source = F.pad(
            source, (pad_width, pad_width, pad_height, pad_height), mode=padding_mode
        )
        padding = (0, 0)
    return F.conv2d(source, weight, bias, stride, padding, dilation)


# -- Architecture gating ------------------------------------------------------
# NOTE: CDNA configs are not tuned at all - these are included only so
# correctness tests can run on CDNA hardware without being skipped.
# AITER Triton CI relies on CDNA runners.
SUPPORTED_ARCHS = {
    "RDNA": {
        "gfx1100",
        "gfx1101",
        "gfx1150",
        "gfx1151",
        "gfx1200",
        "gfx1201",
        "gfx1250",
    },
    "CDNA": {"gfx942", "gfx950"},
}

# Flat union for arch-check use sites.
ALL_SUPPORTED_ARCHS = set().union(*SUPPORTED_ARCHS.values())


# -- Method registry ----------------------------------------------------------

MethodEntry = namedtuple(
    "MethodEntry", ["kernel_fn", "guard_fn", "is_winograd", "bench_tag", "short_name"]
)


def _3x3_guard(R, S, stride, dilation, C):
    return _is_3x3_conv(R, S)


def _direct_3x3_guard(R, S, stride, dilation, C):
    return _is_3x3_conv(R, S) and has_conv_config("CONV-3X3-NCHW")


METHOD_REGISTRY = {
    "default": MethodEntry(conv2d_nchw, None, False, "", "default"),
    "direct": MethodEntry(
        conv2d_nchw_3x3_direct, _direct_3x3_guard, False, "[direct]", "direct"
    ),
    "cblocked": MethodEntry(
        conv2d_nchw_cblocked, _3x3_guard, False, "[cblocked]", "cblocked"
    ),
    "winograd_f4x3": MethodEntry(
        conv2d_winograd_f4x3,
        _is_winograd_2d_eligible,
        True,
        "[winograd_f4x3]",
        "WF(4,3)",
    ),
    "winograd_f4x3_cblocked": MethodEntry(
        conv2d_winograd_f4x3_cblocked,
        _is_winograd_2d_eligible,
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
        self.results: list[TestResult] = []

    def check_close(
        self,
        name: str,
        got: torch.Tensor,
        ref: torch.Tensor,
        K_red: int | None = None,
        rtol: float | None = None,
        atol: float | None = None,
    ) -> TestResult:
        got32 = got.float()
        ref32 = ref.float()
        diff = (got32 - ref32).abs()
        max_abs = float(diff.max().item()) if diff.numel() else 0.0
        rel = max_abs / (float(ref32.abs().max().item()) + 1e-6)
        if rtol is None or atol is None:
            K_est = int(K_red) if K_red is not None else 1024
            rtol_calc, atol_calc = dynamic_conv_tolerances(self.dtype, K_est)
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
            log_result = logger.info if passed else logger.warning
            log_result(
                "  %s %-40s | max_abs=%.3e rel=%.3e",
                mark,
                name,
                max_abs,
                rel,
            )
        return res

    def all_passed(self) -> bool:
        return all(r.passed for r in self.results)

    def failed_results(self) -> list[TestResult]:
        return [r for r in self.results if not r.passed]


# -- Tolerance + dispatch -----------------------------------------------------


def _get_tolerances(
    method_name, entry, suite, y_ref, N, C, H, W, K_out, R, S, stride, dilation
):
    if entry.is_winograd:
        return _winograd_tolerances(suite.dtype, C * R * S, "f4x3")
    if method_name == "default" and _is_3x3_conv(R, S):
        routed = _select_3x3_method(N, C, H, W, K_out, stride, dilation)
        if routed and "winograd" in routed:
            return _winograd_tolerances(suite.dtype, C * R * S, "f4x3")
    return dynamic_conv_tolerances(suite.dtype, C * R * S)


def run_all_methods(
    suite: TestSuite,
    x: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor | None,
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
        logger.info(
            "    %s %s: X%s W%s -> Y%s",
            name,
            kernel_type,
            tuple(x.shape),
            tuple(w.shape),
            tuple(y_ref.shape),
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
                _r, _a = _winograd_tolerances(suite.dtype, C * R * S, "f4x3")
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
    torch.manual_seed(0)
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
    torch.manual_seed(0)
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
    torch.manual_seed(0)
    shapes = [
        (1, 64, 8, 8, 128, 1, 1, (1, 1), (0, 0), (1, 1), "1x1 no bias"),
        (2, 32, 16, 16, 64, 3, 3, (1, 1), (1, 1), (1, 1), "3x3 no bias"),
        (1, 16, 8, 8, 32, 5, 5, (1, 1), (2, 2), (1, 1), "5x5 no bias"),
        (4096, 3, 14, 14, 1024, 14, 14, (14, 14), (0, 0), (1, 1), "kimi_k3 patch"),
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
    torch.manual_seed(0)
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
    torch.manual_seed(seed)
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
        except Exception as e:  # noqa: BLE001
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


def assert_prepacked_pool_sizes_only_rotating_packed_inputs(monkeypatch, dimensions):
    from op_tests.op_benchmarks.triton import bench_conv2d, bench_conv3d
    from op_tests.op_benchmarks.triton.utils.rotating_buffer_bench import (
        RotatingBufferBench,
    )

    bencher = RotatingBufferBench.from_dict(
        {
            "target_rot_pool_sz_mb": 0.01,
            "min_pool_len": 1,
            "max_pool_len": 32,
        }
    )
    inspected = []

    def inspect_pool(function):
        pool = bencher.rot_bufs
        packed = "x_blocked" in pool[0]
        input_key = "x_blocked" if packed else "input_tensor"
        input_bytes = pool[0][input_key].untyped_storage().nbytes()
        expected_length = min(32, math.ceil(0.01 * 1024**2 / input_bytes))
        assert len(pool) == expected_length
        assert len({entry[input_key].data_ptr() for entry in pool}) == len(pool)
        if packed:
            assert len({entry["input_tensor"].data_ptr() for entry in pool}) == 1
        assert len({entry["weight"].data_ptr() for entry in pool}) == 1
        assert len({entry["bias"].data_ptr() for entry in pool}) == 1
        inspected.append(packed)
        return 1.0

    monkeypatch.setattr(bencher, "bench", inspect_pool)
    monkeypatch.setattr(
        RotatingBufferBench,
        "from_dict",
        classmethod(lambda _cls, _config: bencher),
    )
    if dimensions == 2:
        bench_conv2d.main(
            [
                "--N",
                "1",
                "--C",
                "3",
                "--H",
                "3",
                "--W",
                "3",
                "--K",
                "8",
                "--R",
                "3",
                "--S",
                "3",
                "--pad-h",
                "1",
                "--pad-w",
                "1",
                "--method",
                "cblocked",
            ]
        )
    else:
        bench_conv3d.main(
            [
                "--shape",
                "1",
                "3",
                "3",
                "3",
                "3",
                "8",
                "3",
                "3",
                "3",
                "--padding",
                "1",
                "1",
                "1",
                "--method",
                "cblocked",
            ]
        )
    assert inspected == [True, False]
    assert bencher.rot_bufs is None


def assert_conv_benchmark_graph_packing_and_outputs(
    monkeypatch,
    tmp_path,
    dimensions,
    method,
    layout,
    kernel_size,
    channels,
):
    from aiter.ops.triton.conv import _prepack as conv_prepack
    from op_tests.op_benchmarks.triton import bench_conv2d, bench_conv3d
    from op_tests.op_benchmarks.triton.utils.rotating_buffer_bench import (
        RotatingBufferBench,
    )

    monkeypatch.chdir(tmp_path)
    captured_packs = []
    captured_weight_packs = []
    launch_name = (
        "_launch_nchw_to_cblocked" if dimensions == 2 else "_launch_ncdhw_to_cblocked"
    )
    original_launch = getattr(conv_prepack, launch_name)

    def observed_pack(*args, **kwargs):
        if torch.cuda.is_current_stream_capturing():
            captured_packs.append(True)
        return original_launch(*args, **kwargs)

    monkeypatch.setattr(conv_prepack, launch_name, observed_pack)
    weight_packers = (
        "prepack_oihw_to_kmajor",
        "prepack_oihw_to_3x3",
        "prepack_winograd_filter_f4x3",
        "prepack_oidhw_to_kmajor",
        "prepack_oidhw_to_3x3x3",
        "prepack_winograd_hw_filter_f4x3",
    )
    for name in weight_packers:
        original_packer = getattr(conv_prepack, name)

        def observed_weight_pack(*args, packer=original_packer, **kwargs):
            if torch.cuda.is_current_stream_capturing():
                captured_weight_packs.append(True)
            return packer(*args, **kwargs)

        monkeypatch.setattr(conv_prepack, name, observed_weight_pack)

    original_bench = RotatingBufferBench.bench
    measurements = []
    used_benchers = []

    def checked_bench(bencher, function, **kwargs):
        used_benchers.append(bencher)
        assert bencher.kernels_per_graph == 4
        assert bencher.num_replays == 2
        assert bencher.warmup_iters == 1
        assert bencher.warmup_replays == 1
        assert bencher.min_pool_len == 2
        assert bencher.max_pool_len == 4
        assert bencher.target_rot_pool_sz_mb == 1
        pool = bencher.rot_bufs
        packed = "x_blocked" in pool[0]
        input_key = "x_blocked" if packed else "input_tensor"
        assert len({entry[input_key].data_ptr() for entry in pool}) == len(pool)
        assert len({entry["weight"].data_ptr() for entry in pool}) == 1
        captured = []
        previous_packs = len(captured_packs)

        def observed(**entry):
            output = function(**entry)
            if torch.cuda.is_current_stream_capturing():
                captured.append((entry, output))
            return output

        milliseconds = original_bench(bencher, observed, **kwargs)
        assert math.isfinite(milliseconds) and milliseconds > 0
        assert len(captured) == 4
        assert not captured_weight_packs
        measurements.append(len(captured_packs) - previous_packs)

        for entry, output in captured:
            input_tensor = entry["input_tensor"]
            if packed:
                blocked = entry["x_blocked"]
                permutation = (0, 1, blocked.ndim - 1, *range(2, blocked.ndim - 1))
                input_tensor = blocked.permute(permutation).reshape(
                    input_tensor.shape[0], -1, *input_tensor.shape[2:]
                )[:, :channels]
            reference_fn = F.conv2d if dimensions == 2 else F.conv3d
            bias = entry["bias"]
            reference = reference_fn(
                input_tensor.float(),
                entry["weight"].float(),
                bias.float() if bias is not None else None,
                padding=kernel_size // 2,
            )
            reduction = channels * kernel_size**dimensions
            tolerances = (
                _winograd_tolerances
                if "winograd" in method
                else dynamic_conv_tolerances
            )
            relative, absolute = tolerances(torch.float16, reduction)
            torch.testing.assert_close(
                output.float(),
                reference,
                rtol=relative,
                atol=absolute,
            )
        return milliseconds

    monkeypatch.setattr(RotatingBufferBench, "bench", checked_bench)
    common = [
        "--method",
        method,
        "--layout",
        layout,
        "--metric",
        "time",
        "--graph-kernels-per-graph",
        "4",
        "--graph-num-replays",
        "2",
        "--graph-warmup-iters",
        "1",
        "--graph-warmup-replays",
        "1",
        "--rot-bufs-min-pool-len",
        "2",
        "--rot-bufs-target-mb",
        "1",
    ]
    if dimensions == 2:
        bench_conv2d.main(
            common
            + [
                "--N",
                "1",
                "--C",
                str(channels),
                "--H",
                "7",
                "--W",
                "7",
                "--K",
                "8",
                "--R",
                str(kernel_size),
                "--S",
                str(kernel_size),
                "--pad-h",
                str(kernel_size // 2),
                "--pad-w",
                str(kernel_size // 2),
            ]
        )
    else:
        bench_conv3d.main(
            common
            + [
                "--shape",
                "1",
                str(channels),
                "2",
                "7",
                "7",
                "8",
                *([str(kernel_size)] * 3),
                "--padding",
                *([str(kernel_size // 2)] * 3),
                "-o",
            ]
        )
        (output_file,) = tmp_path.glob("conv3d_*.csv")
        with output_file.open() as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == len(measurements)
        assert all(row["weight_policy"] == "fixed" for row in rows)
        assert all(row["provider"] != "torch" for row in rows)
    assert measurements == ([0, 4] if "cblocked" in method else [0])
    assert all(bencher.rot_bufs is None for bencher in used_benchers)
