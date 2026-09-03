# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Pytest unit tests for aiter.ops.triton.conv.conv2d.

Kernel correctness tests compare Triton against torch.nn.functional.conv2d on
synthetic tensors. Routing tests exercise configuration lookup without launching
kernels. No model loading, network access, or torchvision.

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

Plus 7 exact-route and configuration-precedence regression cases.

Total: 12 + 12 + 12 + 36 + 2 + 7 = 81 cases.

Where a kernel's guard rejects a shape (e.g. winograd on a 5x5), the
shape is silently skipped inside run_all_methods.

Performance benchmarking lives in
op_tests/op_benchmarks/triton/bench_conv2d.py (and, for real-model
shapes, in op_benchmarks/triton/model_benchmarking_tool/bench_models.py).
"""

import pytest
import torch

import aiter.ops.triton.conv.conv2d as conv2d_module
from aiter.ops.triton.utils import conv_config_utils
from aiter.ops.triton.utils._triton.arch_info import get_arch

from ._helpers import (
    ALL_SUPPORTED_ARCHS,
    ORDERED_METHODS,
    TestSuite,
    run_activations,
    run_cross_method,
    run_edge_cases,
    run_no_bias,
    run_random_fuzzing,
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


# -- Configuration lookup and routing (no kernel launches) -------------------

_GFX1100_PINNED = {
    "N": 1,
    "C": 64,
    "H": 56,
    "W": 56,
    "K": 64,
    "stride": (1, 1),
    "padding": (1, 1),
}

_GFX1151_PINNED = {
    "N": 1,
    "C": 256,
    "H": 28,
    "W": 28,
    "K": 256,
    "stride": (2, 2),
    "padding": (1, 1),
}

_UNPINNED = {
    "N": 1,
    "C": 64,
    "H": 55,
    "W": 55,
    "K": 64,
    "stride": (1, 1),
    "padding": (1, 1),
}


@pytest.fixture
def isolated_conv_config_cache():
    """Keep mocked architectures and synthetic config tables test-local."""
    conv_config_utils._get_conv_config_cached.cache_clear()
    conv_config_utils.has_conv_config.cache_clear()
    yield
    conv_config_utils._get_conv_config_cached.cache_clear()
    conv_config_utils.has_conv_config.cache_clear()


def _use_arch(monkeypatch, arch):
    monkeypatch.setattr(conv_config_utils.arch_info, "get_arch", lambda: arch)


def _resolve_nchw_3x3(shape, *, stride=None, padding=None):
    return conv2d_module._resolve_route(
        R=3,
        S=3,
        stride=shape["stride"] if stride is None else stride,
        dilation=(1, 1),
        N=shape["N"],
        C=shape["C"],
        H=shape["H"],
        W_in=shape["W"],
        K_out=shape["K"],
        layout="nchw",
        padding=shape["padding"] if padding is None else padding,
    )


@pytest.mark.parametrize(
    "arch,shape",
    [("gfx1100", _GFX1100_PINNED), ("gfx1151", _GFX1151_PINNED)],
)
def test_exact_nchw_pin_selects_direct(
    monkeypatch, isolated_conv_config_cache, arch, shape
):
    _use_arch(monkeypatch, arch)

    assert _resolve_nchw_3x3(shape) is conv2d_module.Route.DIRECT_NCHW_3X3


@pytest.mark.parametrize("arch", ["gfx1100", "gfx1151"])
def test_unpinned_nchw_shape_falls_back_to_cblocked(
    monkeypatch, isolated_conv_config_cache, arch
):
    _use_arch(monkeypatch, arch)

    assert _resolve_nchw_3x3(_UNPINNED) is conv2d_module.Route.CBLOCKED_NCHW


@pytest.mark.parametrize(
    "route_override",
    [{"padding": (0, 0)}, {"stride": (2, 2)}],
    ids=["padding", "stride"],
)
def test_exact_nchw_pin_uses_complete_shape_key(
    monkeypatch, isolated_conv_config_cache, route_override
):
    _use_arch(monkeypatch, "gfx1100")

    assert (
        _resolve_nchw_3x3(_GFX1100_PINNED, **route_override)
        is conv2d_module.Route.CBLOCKED_NCHW
    )


def test_conv_config_layout_variant_precedence(monkeypatch, isolated_conv_config_cache):
    shape_key = "test-shape"
    config = {
        "shapes_nhwc": {shape_key: {"source": "layout"}},
        "shapes": {shape_key: {"source": "generic"}},
        "M_LEQ_64": {"source": "bucket"},
        "any": {"source": "any"},
    }
    monkeypatch.setattr(
        conv_config_utils, "_get_conv_config_file", lambda _config_name: config
    )

    def selected(*variants, key=shape_key, M=32):
        return conv_config_utils.get_conv_config(
            "TEST-CONV-VARIANTS", shape_key=key, M=M, variants=variants
        )["source"]

    assert selected("nhwc") == "layout"
    assert selected() == "generic"
    assert selected("nhwc", key="missing") == "bucket"
    assert selected("nhwc", key="missing", M=65) == "any"
