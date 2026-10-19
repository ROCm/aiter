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

Plus 7 exact-route and configuration-precedence regression cases, 2
scalar-parameter cases, 1 cache-clear case, and 2 inference-tensor cases.

Convolution cases: 12 + 12 + 12 + 36 + 2 + 7 + 2 + 1 + 2 = 86.
Padding preparation adds 20 focused cases for validation, boundary amounts,
layout conversion, empty batches, no-op reuse, and delegation to F.pad.

The whole module is skipped without a supported GPU. Padding preparation itself
does not require architecture-specific padding configurations.

Where a kernel's guard rejects a shape (e.g. winograd on a 5x5), the
shape is silently skipped inside run_all_methods.

Performance benchmarking lives in
op_tests/op_benchmarks/triton/bench_conv2d.py (and, for real-model
shapes, in op_benchmarks/triton/model_benchmarking_tool/bench_models.py).
"""

import pytest
import torch
import torch.nn.functional as F

import aiter.ops.triton.conv.conv2d as conv2d_module
from aiter.ops.triton.conv import _launch as padding_ops
from aiter.ops.triton.conv import _prepack as conv_prepack
from aiter.ops.triton.conv._prepack import clear_conv2d_weight_pack_caches
from aiter.ops.triton.utils import conv_config_utils
from aiter.ops.triton.utils._triton.arch_info import get_arch
from op_tests.triton_tests.conv._helpers import (
    ALL_SUPPORTED_ARCHS,
    CONV2D_WEIGHT_PACK_CACHE_NAMES,
    ORDERED_METHODS,
    TestSuite,
    assert_weight_pack_cache_clear_is_scoped,
    dynamic_conv_tolerances,
    run_activations,
    run_cross_method,
    run_edge_cases,
    run_no_bias,
    run_random_fuzzing,
)

_current_arch = get_arch() if torch.cuda.is_available() else None
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


@pytest.mark.parametrize("layout", ["nchw", "nhwc"])
def test_scalar_parameters_and_noncontiguous_input(layout):
    """Conv2D accepts scalar parameters and materializes sliced inputs."""
    torch.manual_seed(0)
    x_base = torch.randn(1, 32, 12, 18, device="cuda", dtype=torch.float16)
    x = x_base[..., ::2]
    assert not x.is_contiguous(), f"expected sliced input, got strides={x.stride()}"
    w = torch.randn(48, 32, 1, 1, device="cuda", dtype=torch.float16)

    y = conv2d_module.conv2d(x, w, stride=1, padding=0, dilation=1, layout=layout)
    ref = F.conv2d(x.float(), w.float())
    rtol, atol = dynamic_conv_tolerances(torch.float16, 32)
    torch.testing.assert_close(y.float(), ref, rtol=rtol, atol=atol)
    if layout == "nhwc":
        assert y.is_contiguous(
            memory_format=torch.channels_last
        ), f"expected channels-last output, got strides={y.stride()}"
    else:
        assert (
            y.is_contiguous()
        ), f"expected contiguous output, got strides={y.stride()}"


def test_conv2d_weight_pack_cache_clear_is_scoped(monkeypatch):
    assert_weight_pack_cache_clear_is_scoped(
        monkeypatch,
        conv_prepack,
        clear_conv2d_weight_pack_caches,
        CONV2D_WEIGHT_PACK_CACHE_NAMES,
    )


def test_inference_weight_bypasses_pack_cache_and_observes_updates(monkeypatch):
    cache = conv_prepack._LRUPackCache(maxsize=2)
    monkeypatch.setattr(conv_prepack, "_PACK_CACHE_3x3", cache)

    with torch.inference_mode():
        weight = torch.arange(18, dtype=torch.float16, device="cpu").reshape(2, 1, 3, 3)
        first, _ = conv_prepack.get_or_make_weight_pack_3x3(weight, block_c=4)
        second, _ = conv_prepack.get_or_make_weight_pack_3x3(weight, block_c=4)
        weight.add_(10)
        refreshed, _ = conv_prepack.get_or_make_weight_pack_3x3(weight, block_c=4)

    assert not cache._d, "inference weights must not enter the global cache"
    assert second is not first, "inference weight unexpectedly reused a pack"
    assert refreshed is not second, "updated inference weight reused a stale pack"
    expected = weight.reshape(2, 1, 9).permute(0, 2, 1)
    torch.testing.assert_close(refreshed[:, :, :1], expected)


def test_conv2d_inference_weight_observes_updates():
    torch.manual_seed(0)
    with torch.inference_mode():
        x = torch.randn(1, 64, 16, 16, device="cuda", dtype=torch.float16)
        weight = torch.randn(32, 64, 3, 3, device="cuda", dtype=torch.float16)

        output = conv2d_module.conv2d(x, weight, padding=1)
        reference = F.conv2d(x.float(), weight.float(), padding=1)

        weight.add_(0.25)
        output_after_update = conv2d_module.conv2d(x, weight, padding=1)
        reference_after_update = F.conv2d(x.float(), weight.float(), padding=1)

    rtol, atol = dynamic_conv_tolerances(torch.float16, 64 * 3 * 3)
    torch.testing.assert_close(output.float(), reference, rtol=rtol, atol=atol)
    torch.testing.assert_close(
        output_after_update.float(), reference_after_update, rtol=rtol, atol=atol
    )


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

    route = _resolve_nchw_3x3(shape)
    assert (
        route is conv2d_module.Route.DIRECT_NCHW_3X3
    ), f"expected direct NCHW route for pinned {arch} shape, got {route}"


@pytest.mark.parametrize("arch", ["gfx1100", "gfx1151"])
def test_unpinned_nchw_shape_falls_back_to_cblocked(
    monkeypatch, isolated_conv_config_cache, arch
):
    _use_arch(monkeypatch, arch)

    route = _resolve_nchw_3x3(_UNPINNED)
    assert (
        route is conv2d_module.Route.CBLOCKED_NCHW
    ), f"expected NCHWc fallback for unpinned {arch} shape, got {route}"


@pytest.mark.parametrize(
    "route_override",
    [{"padding": (0, 0)}, {"stride": (2, 2)}],
    ids=["padding", "stride"],
)
def test_exact_nchw_pin_uses_complete_shape_key(
    monkeypatch, isolated_conv_config_cache, route_override
):
    _use_arch(monkeypatch, "gfx1100")

    route = _resolve_nchw_3x3(_GFX1100_PINNED, **route_override)
    assert (
        route is conv2d_module.Route.CBLOCKED_NCHW
    ), f"expected NCHWc after shape-key change {route_override}, got {route}"


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

    assert selected("nhwc") == "layout", "layout-specific config was not preferred"
    assert selected() == "generic", "generic shape config was not selected"
    assert (
        selected("nhwc", key="missing") == "bucket"
    ), "M bucket was not used after a layout-specific shape miss"
    assert (
        selected("nhwc", key="missing", M=65) == "any"
    ), "generic fallback was not used after shape and bucket misses"


@pytest.mark.parametrize(
    "padding_mode,layout,dtype",
    [
        pytest.param("reflect", "nchw", torch.float16, id="reflect-nchw-fp16"),
        pytest.param("replicate", "nhwc", torch.bfloat16, id="replicate-nhwc-bf16"),
        pytest.param("circular", "nhwc", torch.float16, id="circular-nhwc-fp16"),
    ],
)
def test_prepared_padding_delegates_to_pytorch(
    monkeypatch, padding_mode, layout, dtype
):
    """Check our forwarding/layout contract, not PyTorch's padding arithmetic."""
    input_format = torch.channels_last if layout == "nchw" else torch.contiguous_format
    output_format = torch.channels_last if layout == "nhwc" else torch.contiguous_format
    source = torch.randn(1, 64, 56, 56, device="cuda", dtype=dtype).contiguous(
        memory_format=input_format
    )
    original = source.clone()
    original_pad = F.pad
    calls = []

    def recording_pad(input_tensor, padding, mode):
        calls.append((input_tensor, padding, mode))
        return original_pad(input_tensor, padding, mode=mode)

    def unexpected_config(*args, **kwargs):
        pytest.fail("F.pad must not need Triton padding launch configurations")

    monkeypatch.setattr(padding_ops.F, "pad", recording_pad)
    monkeypatch.setattr(conv_config_utils, "_get_conv_config_file", unexpected_config)
    actual, conv_padding = padding_ops._prepare_conv2d_padding(
        source, (1, 2), padding_mode, layout=layout
    )

    assert len(calls) == 1
    assert calls[0][0] is source
    assert calls[0][1:3] == ((2, 2, 1, 1), padding_mode)
    assert actual.shape == (1, 64, 58, 60)
    assert actual.is_contiguous(memory_format=output_format)
    assert actual.dtype == source.dtype
    assert actual.device == source.device
    assert actual.data_ptr() != source.data_ptr()
    assert conv_padding == (0, 0)
    torch.testing.assert_close(source, original, rtol=0, atol=0)


@pytest.mark.parametrize(
    "padding_mode,height,width,padding",
    [
        pytest.param("reflect", 3, 5, (2, 4), id="reflect-limit"),
        pytest.param("reflect", 1, 5, (0, 4), id="reflect-singleton-axis"),
        pytest.param("replicate", 1, 1, 3, id="replicate-beyond-size-scalar"),
        pytest.param("circular", 3, 5, (3, 5), id="circular-limit"),
    ],
)
def test_prepared_padding_accepts_boundary_amounts(
    padding_mode, height, width, padding
):
    source = torch.ones(2, 3, height, width, device="cuda", dtype=torch.float16)
    actual, conv_padding = padding_ops._prepare_conv2d_padding(
        source, padding, padding_mode
    )
    pad_height, pad_width = (padding, padding) if isinstance(padding, int) else padding
    assert actual.shape == (2, 3, height + 2 * pad_height, width + 2 * pad_width)
    assert actual.is_contiguous()
    assert conv_padding == (0, 0)


@pytest.mark.parametrize(
    "padding_mode,padding,layout",
    [
        ("zeros", (1, 3), "nhwc"),
        ("reflect", 0, "nchw"),
        ("replicate", (0, 0), "nchw"),
    ],
)
def test_prepared_padding_noop_does_not_materialize(
    monkeypatch, padding_mode, padding, layout
):
    source = torch.ones(2, 3, 5, 7).transpose(2, 3)

    def unexpected_materialization(*args, **kwargs):
        pytest.fail("no-op padding must not pad or convert the input layout")

    monkeypatch.setattr(padding_ops.F, "pad", unexpected_materialization)
    monkeypatch.setattr(padding_ops, "_ensure_layout", unexpected_materialization)
    actual, conv_padding = padding_ops._prepare_conv2d_padding(
        source, padding, padding_mode, layout=layout
    )
    expected_padding = (padding, padding) if isinstance(padding, int) else padding
    assert actual is source
    assert conv_padding == expected_padding


@pytest.mark.parametrize(
    "shape,padding,padding_mode,layout,message",
    [
        ((3, 5, 7), 0, "zeros", "nchw", "must be 4-D NCHW"),
        ((2, 3, 5, 7), 0, "reflect", "invalid", "layout must be"),
        ((2, 3, 5, 7), 0, [], "nchw", "padding_mode must be one of"),
        ((2, 3, 0, 7), 0, "zeros", "nchw", "spatial dimensions must be positive"),
        ((2, 3, 5, 7), (-1, 0), "replicate", "nchw", "must be non-negative"),
        ((2, 3, 5, 7), (1, 2, 3), "replicate", "nchw", "length-2 sequence"),
        ((2, 3, 5, 7), (5, 0), "reflect", "nchw", "reflect padding"),
        ((2, 3, 5, 7), (0, 8), "circular", "nchw", "circular padding"),
        ((2, 3, 5, 7), 1, "replicate", "nchw", "requires a CUDA/ROCm tensor"),
    ],
    ids=[
        "rank",
        "layout",
        "mode",
        "spatial-dimension",
        "negative-padding",
        "padding-length",
        "reflect-limit",
        "circular-limit",
        "cpu-materialization",
    ],
)
def test_prepared_padding_validates_before_materialization(
    shape, padding, padding_mode, layout, message, monkeypatch
):
    source = torch.empty(shape)

    def unexpected_padding(*args, **kwargs):
        pytest.fail("invalid padding must be rejected before calling F.pad")

    monkeypatch.setattr(padding_ops.F, "pad", unexpected_padding)
    with pytest.raises(ValueError, match=message):
        padding_ops._prepare_conv2d_padding(
            source, padding, padding_mode, layout=layout
        )


def test_prepared_padding_handles_empty_batch():
    source = torch.empty(0, 3, 5, 7, device="cuda", dtype=torch.float16)
    actual, conv_padding = padding_ops._prepare_conv2d_padding(
        source, (1, 2), "replicate", layout="nhwc"
    )
    assert actual.shape == (0, 3, 7, 11)
    assert actual.is_contiguous(memory_format=torch.channels_last)
    assert actual.dtype == source.dtype
    assert actual.device == source.device
    assert conv_padding == (0, 0)
