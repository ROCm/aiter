# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Pytest unit tests for aiter.ops.triton.conv.conv2d.

Correctness only. All tests compare Triton kernels against
torch.nn.functional.conv2d on synthetic tensors. No model loading,
no network, no torchvision.

Test matrix (uniform across the four primary test families):

    NCHW × {fp16, bf16} × every kernel  (5 kernels)         = 10
    NHWC × {fp16, bf16}                  (single dispatch)  =  2
                                                            ---
                                  base cases per test family  12

test_edge, test_fuzz, test_no_bias use the base matrix as-is.
test_activations multiplies the base matrix by 3 (relu/relu6/gelu) -> 36.

Plus test_cross_method (differential correctness) that runs every NCHW
kernel on shapes routable by all of them and verifies they all match
F.conv2d. NCHW-only by design; 2 cases (one per dtype).

Total: 12 + 12 + 12 + 36 + 2 = 74 cases.

Where a kernel's guard rejects a shape (e.g. winograd on a 5x5), the
shape is silently skipped inside run_all_methods.

Performance benchmarking lives in
op_tests/op_benchmarks/triton/bench_conv2d.py (and, for real-model
shapes, in op_benchmarks/triton/model_benchmarking_tool/bench_models.py).
"""

from __future__ import annotations

import pytest
import torch

from aiter.ops.triton.utils._triton.arch_info import get_arch

from ._helpers import (
    ALL_SUPPORTED_ARCHS,
    TestSuite,
    ORDERED_METHODS,
    run_edge_cases,
    run_activations,
    run_no_bias,
    run_random_fuzzing,
    run_cross_method,
)

# Module-level arch gate. Skip the whole test module on unsupported archs
# rather than fail per-test. Extend SUPPORTED_ARCHS in _helpers.py when
# adding CDNA (or other RDNA) support.
_current_arch = get_arch()
if _current_arch not in ALL_SUPPORTED_ARCHS:
    pytest.skip(
        f"aiter.ops.triton.conv tests run on {sorted(ALL_SUPPORTED_ARCHS)}; "
        f"current arch {_current_arch!r} not supported",
        allow_module_level=True,
    )


# Build the (dtype, layout, method) matrix once. NHWC entries only pair with
# method="default" because conv2d_nhwc is single-dispatch — the method param
# is a no-op there, so re-running for every method id would just duplicate work.
def _build_matrix():
    matrix = []
    for dtype, dtype_id in [(torch.float16, "fp16"), (torch.bfloat16, "bf16")]:
        for method in ORDERED_METHODS:
            matrix.append(((dtype, "nchw", method), f"{dtype_id}_nchw_{method}"))
        matrix.append(((dtype, "nhwc", "default"), f"{dtype_id}_nhwc"))
    return matrix


_MATRIX = _build_matrix()
PARAMS = [params for params, _ in _MATRIX]
IDS = [tid for _, tid in _MATRIX]


def _make_suite(dtype, layout):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return TestSuite(device="cuda", dtype=dtype, layout_mode=layout)


def _assert_suite(suite: TestSuite):
    failed = suite.failed_results()
    assert not failed, f"{len(failed)} tests failed: {[r.name for r in failed]}"


# -- The four primary test families, all on the same matrix ------------------


@pytest.mark.parametrize("dtype,layout,method", PARAMS, ids=IDS)
def test_edge(dtype, layout, method):
    suite = _make_suite(dtype, layout)
    run_edge_cases(suite, method=method)
    _assert_suite(suite)


@pytest.mark.parametrize("dtype,layout,method", PARAMS, ids=IDS)
def test_fuzz(dtype, layout, method):
    suite = _make_suite(dtype, layout)
    run_random_fuzzing(suite, num_tests=10, method=method)
    _assert_suite(suite)


@pytest.mark.parametrize("dtype,layout,method", PARAMS, ids=IDS)
def test_no_bias(dtype, layout, method):
    suite = _make_suite(dtype, layout)
    run_no_bias(suite, method=method)
    _assert_suite(suite)


@pytest.mark.parametrize("activation", ["relu", "relu6", "gelu"])
@pytest.mark.parametrize("dtype,layout,method", PARAMS, ids=IDS)
def test_activations(dtype, layout, method, activation):
    suite = _make_suite(dtype, layout)
    run_activations(suite, method=method, activation=activation)
    _assert_suite(suite)


# -- Differential correctness across all 5 NCHW kernels (NCHW-only) ----------


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
def test_cross_method(dtype):
    suite = _make_suite(dtype, "nchw")
    run_cross_method(suite)
    _assert_suite(suite)
