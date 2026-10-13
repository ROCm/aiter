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
Padding primitives add 684 validation, mapping, reference, materialization, and
configuration cases, for a total of 770 cases.

Convolution tests retain architecture gating. Padding validation and reference
tests run on CPU. GPU tests that read padding configurations require a configured
architecture; coordinate-mapping and no-op checks do not require padding configs.

Where a kernel's guard rejects a shape (e.g. winograd on a 5x5), the
shape is silently skipped inside run_all_methods.

Performance benchmarking lives in
op_tests/op_benchmarks/triton/bench_conv2d.py (and, for real-model
shapes, in op_benchmarks/triton/model_benchmarking_tool/bench_models.py).
"""

import pytest
import torch
import torch.nn.functional as F
import triton
import triton.language as tl

import aiter.ops.triton.conv.conv2d as conv2d_module
from aiter.ops.triton._triton_kernels.conv.padding import _map_padding_coordinate
from aiter.ops.triton.conv import _launch as padding_ops
from aiter.ops.triton.conv import _prepack as conv_prepack
from aiter.ops.triton.conv._prepack import clear_conv2d_weight_pack_caches
from aiter.ops.triton.conv._utils import _validate_conv2d_padding_mode
from aiter.ops.triton.utils import conv_config_utils
from aiter.ops.triton.utils._triton.arch_info import get_arch
from op_tests.triton_tests.conv._helpers import (
    ALL_SUPPORTED_ARCHS,
    CONV2D_WEIGHT_PACK_CACHE_NAMES,
    ORDERED_METHODS,
    TestSuite,
    assert_weight_pack_cache_clear_is_scoped,
    conv2d_padding_reference,
    dynamic_conv_tolerances,
    run_activations,
    run_cross_method,
    run_edge_cases,
    run_no_bias,
    run_random_fuzzing,
)

_current_arch = get_arch() if torch.cuda.is_available() else None
_requires_conv_arch = pytest.mark.skipif(
    _current_arch not in ALL_SUPPORTED_ARCHS,
    reason=(
        f"aiter.ops.triton.conv tests run on {sorted(ALL_SUPPORTED_ARCHS)}; "
        f"current arch {_current_arch!r} not supported"
    ),
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
@_requires_conv_arch
def test_edge(dtype, layout, method):
    suite = _make_suite(dtype, layout)
    run_edge_cases(suite, method=method)
    _assert_suite(suite)


@pytest.mark.parametrize("dtype,layout,method", PARAMS, ids=IDS)
@_requires_conv_arch
def test_fuzz(dtype, layout, method):
    suite = _make_suite(dtype, layout)
    run_random_fuzzing(suite, num_tests=10, method=method)
    _assert_suite(suite)


@pytest.mark.parametrize("dtype,layout,method", PARAMS, ids=IDS)
@_requires_conv_arch
def test_no_bias(dtype, layout, method):
    suite = _make_suite(dtype, layout)
    run_no_bias(suite, method=method)
    _assert_suite(suite)


@pytest.mark.parametrize("activation", ["relu", "relu6", "gelu"])
@pytest.mark.parametrize("dtype,layout,method", PARAMS, ids=IDS)
@_requires_conv_arch
def test_activations(dtype, layout, method, activation):
    suite = _make_suite(dtype, layout)
    run_activations(suite, method=method, activation=activation)
    _assert_suite(suite)


# -- Differential correctness across all 5 NCHW kernels (NCHW-only) ----------


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@_requires_conv_arch
def test_cross_method(dtype):
    suite = _make_suite(dtype, "nchw")
    run_cross_method(suite)
    _assert_suite(suite)


@pytest.mark.parametrize("layout", ["nchw", "nhwc"])
@_requires_conv_arch
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


@_requires_conv_arch
def test_conv2d_weight_pack_cache_clear_is_scoped(monkeypatch):
    assert_weight_pack_cache_clear_is_scoped(
        monkeypatch,
        conv_prepack,
        clear_conv2d_weight_pack_caches,
        CONV2D_WEIGHT_PACK_CACHE_NAMES,
    )


@_requires_conv_arch
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


@_requires_conv_arch
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
@_requires_conv_arch
def test_exact_nchw_pin_selects_direct(
    monkeypatch, isolated_conv_config_cache, arch, shape
):
    _use_arch(monkeypatch, arch)

    route = _resolve_nchw_3x3(shape)
    assert (
        route is conv2d_module.Route.DIRECT_NCHW_3X3
    ), f"expected direct NCHW route for pinned {arch} shape, got {route}"


@pytest.mark.parametrize("arch", ["gfx1100", "gfx1151"])
@_requires_conv_arch
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
@_requires_conv_arch
def test_exact_nchw_pin_uses_complete_shape_key(
    monkeypatch, isolated_conv_config_cache, route_override
):
    _use_arch(monkeypatch, "gfx1100")

    route = _resolve_nchw_3x3(_GFX1100_PINNED, **route_override)
    assert (
        route is conv2d_module.Route.CBLOCKED_NCHW
    ), f"expected NCHWc after shape-key change {route_override}, got {route}"


@_requires_conv_arch
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


_PADDING_MODES = ("zeros", "reflect", "replicate", "circular")
_PADDING_CONFIG_ARCHS = ("gfx1201", "gfx942", "gfx950")


def _valid_padding_cases():
    cases = []
    for padding_mode in _PADDING_MODES:
        for height, width in ((1, 1), (1, 7), (5, 1), (3, 7)):
            if padding_mode == "reflect":
                max_height, max_width = height - 1, width - 1
            elif padding_mode == "circular":
                max_height, max_width = height, width
            else:
                max_height, max_width = height + 2, width + 3
            paddings = {
                (0, 0),
                (min(1, max_height), min(2, max_width)),
                (max_height, 0),
                (0, max_width),
                (max_height, max_width),
            }
            for padding in sorted(paddings):
                cases.append(
                    pytest.param(
                        padding_mode,
                        height,
                        width,
                        padding,
                        id=(
                            f"{padding_mode}-{height}x{width}-"
                            f"pad{padding[0]}x{padding[1]}"
                        ),
                    )
                )
    return cases


_VALID_PADDING_CASES = _valid_padding_cases()


@pytest.mark.parametrize("padding_mode,height,width,padding", _VALID_PADDING_CASES)
def test_padding_validation_accepts_legal_amounts(padding_mode, height, width, padding):
    _validate_conv2d_padding_mode(padding_mode, height, width, padding)


@pytest.mark.parametrize(
    "padding_mode", ["", "constant", "Reflect", "edge", None, 0, []]
)
@pytest.mark.parametrize("padding", [(0, 0), (1, 2)])
def test_padding_validation_rejects_unknown_modes(padding_mode, padding):
    with pytest.raises(ValueError, match="padding_mode must be one of"):
        _validate_conv2d_padding_mode(padding_mode, 3, 7, padding)


@pytest.mark.parametrize("padding_mode", _PADDING_MODES)
@pytest.mark.parametrize("padding", [(-1, 0), (0, -1), (-1, -1)])
def test_padding_validation_rejects_negative_amounts(padding_mode, padding):
    with pytest.raises(ValueError, match="padding values must be non-negative"):
        _validate_conv2d_padding_mode(padding_mode, 3, 7, padding)


@pytest.mark.parametrize("padding_mode", _PADDING_MODES)
@pytest.mark.parametrize("height,width", [(0, 3), (3, 0), (-1, 3), (3, -1)])
def test_padding_validation_rejects_nonpositive_dimensions(padding_mode, height, width):
    with pytest.raises(ValueError, match="input spatial dimensions must be positive"):
        _validate_conv2d_padding_mode(padding_mode, height, width, (0, 0))


@pytest.mark.parametrize(
    "padding_mode,height,width,padding",
    [
        ("reflect", 3, 5, (3, 0)),
        ("reflect", 3, 5, (0, 5)),
        ("reflect", 1, 5, (1, 0)),
        ("reflect", 5, 1, (0, 1)),
        ("circular", 3, 5, (4, 0)),
        ("circular", 3, 5, (0, 6)),
        ("circular", 1, 5, (2, 0)),
        ("circular", 5, 1, (0, 2)),
    ],
)
def test_padding_validation_matches_pytorch_limits(
    padding_mode, height, width, padding
):
    with pytest.raises(ValueError, match=f"{padding_mode} padding"):
        _validate_conv2d_padding_mode(padding_mode, height, width, padding)

    source = torch.ones(1, 1, height, width)
    pad_height, pad_width = padding
    with pytest.raises(RuntimeError):
        F.pad(source, (pad_width, pad_width, pad_height, pad_height), mode=padding_mode)


@triton.jit
def _padding_gather_kernel(
    source,
    output,
    height: tl.constexpr,
    width: tl.constexpr,
    pad_height,
    pad_width,
    output_width,
    overhang: tl.constexpr,
    PADDING_MODE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    columns = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    input_row, row_valid = _map_padding_coordinate(
        row - pad_height - overhang, height, pad_height, PADDING_MODE
    )
    input_columns, columns_valid = _map_padding_coordinate(
        columns - pad_width - overhang, width, pad_width, PADDING_MODE
    )
    output_valid = columns < output_width
    values = tl.load(
        source + input_row * width + input_columns,
        mask=output_valid & row_valid & columns_valid,
        other=0.0,
    )
    tl.store(output + row * output_width + columns, values, mask=output_valid)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize("padding_mode,height,width,padding", _VALID_PADDING_CASES)
def test_padding_mapping_matches_pytorch(padding_mode, height, width, padding, dtype):
    """Check scalar rows, vector columns, and zeroed partial-tile overhang."""
    _validate_conv2d_padding_mode(padding_mode, height, width, padding)
    source = torch.arange(1, height * width + 1, dtype=torch.float32).reshape(
        1, 1, height, width
    )
    pad_height, pad_width = padding
    overhang = 3
    output_height = height + 2 * (pad_height + overhang)
    output_width = width + 2 * (pad_width + overhang)
    output = torch.empty(1, 1, output_height, output_width, device="cuda", dtype=dtype)
    _padding_gather_kernel[(output_height, triton.cdiv(output_width, 64))](
        source.to(device="cuda", dtype=dtype),
        output,
        height,
        width,
        pad_height,
        pad_width,
        output_width,
        overhang,
        PADDING_MODE=padding_mode,
        BLOCK_SIZE=64,
        num_warps=4,
    )

    expected = F.pad(
        source,
        (pad_width, pad_width, pad_height, pad_height),
        mode="constant" if padding_mode == "zeros" else padding_mode,
    )
    expected = F.pad(expected, (overhang, overhang, overhang, overhang))
    torch.testing.assert_close(output.cpu().float(), expected, rtol=0, atol=0)


@pytest.mark.parametrize("padding_mode", _PADDING_MODES)
@pytest.mark.parametrize("layout", ["nchw", "nhwc"])
@pytest.mark.parametrize("has_bias", [False, True])
@pytest.mark.parametrize(
    "padding,stride,dilation",
    [(0, 1, 1), (1, 2, 1), ((1, 2), (2, 1), (1, 2))],
    ids=["unpadded", "scalar-strided", "per-axis-dilated"],
)
def test_padding_reference_matches_module(
    padding_mode, layout, has_bias, padding, stride, dilation
):
    source = torch.arange(140, dtype=torch.float32).reshape(2, 2, 5, 7) / 16
    if layout == "nhwc":
        source = source.contiguous(memory_format=torch.channels_last)
    weight = torch.arange(36, dtype=torch.float32).reshape(3, 2, 3, 2) / 16
    bias = torch.arange(3, dtype=torch.float32) / 16 if has_bias else None
    module = torch.nn.Conv2d(
        2,
        3,
        (3, 2),
        stride=stride,
        padding=padding,
        dilation=dilation,
        padding_mode=padding_mode,
        bias=has_bias,
    )
    with torch.no_grad():
        module.weight.copy_(weight)
        if has_bias:
            module.bias.copy_(bias)
        expected = module(source)
        actual = conv2d_padding_reference(
            source,
            weight,
            bias,
            stride=stride,
            padding=padding,
            dilation=dilation,
            padding_mode=padding_mode,
        )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize("layout", ["nchw", "nhwc"])
@pytest.mark.parametrize("padding_mode,height,width,padding", _VALID_PADDING_CASES)
def test_prepared_padding_matches_pytorch(
    padding_mode, height, width, padding, layout, dtype
):
    if (
        padding_mode != "zeros"
        and padding != (0, 0)
        and _current_arch not in _PADDING_CONFIG_ARCHS
    ):
        pytest.skip(f"No padding configuration for {_current_arch!r}")
    memory_format = torch.channels_last if layout == "nhwc" else torch.contiguous_format
    source = torch.arange(6 * height * width, device="cuda").reshape(
        2, 3, height, width
    )
    source = source.to(dtype=dtype, memory_format=memory_format)
    original = source.clone()
    actual, conv_padding = padding_ops._prepare_conv2d_padding(
        source, padding, padding_mode, layout=layout
    )

    if padding_mode == "zeros":
        expected = source
        assert actual is source
        assert conv_padding == padding
    else:
        pad_height, pad_width = padding
        expected = F.pad(
            source, (pad_width, pad_width, pad_height, pad_height), mode=padding_mode
        )
        assert conv_padding == (0, 0)
        if padding == (0, 0):
            assert actual is source
        else:
            assert actual.data_ptr() != source.data_ptr()
    assert actual.is_contiguous(memory_format=memory_format)
    assert actual.dtype == source.dtype
    assert actual.device == source.device
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(source, original, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize("layout", ["nchw", "nhwc"])
@pytest.mark.parametrize("padding_mode", ["reflect", "replicate", "circular"])
@pytest.mark.parametrize(
    "source_kind", ["sliced", "transposed", "broadcast", "channels_last"]
)
@pytest.mark.skipif(
    _current_arch not in _PADDING_CONFIG_ARCHS,
    reason=f"No padding configuration for {_current_arch!r}",
)
def test_prepared_padding_reads_source_strides(
    padding_mode, layout, dtype, source_kind
):
    source = torch.arange(2 * 5 * 9 * 13, device="cuda").reshape(2, 5, 9, 13).to(dtype)
    if source_kind == "sliced":
        source = source[:, 1::2, 1::2, 2::3]
        assert source.storage_offset() > 0
    elif source_kind == "transposed":
        source = source.transpose(1, 2).transpose(2, 3)
    elif source_kind == "broadcast":
        source = source[:1, :1, :1, :].expand(2, 5, 9, 13)
        assert source.stride()[:3] == (0, 0, 0)
    else:
        source = source.contiguous(memory_format=torch.channels_last)
    actual, conv_padding = padding_ops._prepare_conv2d_padding(
        source, (1, 2), padding_mode, layout=layout
    )
    expected = F.pad(source, (2, 2, 1, 1), mode=padding_mode)
    memory_format = torch.channels_last if layout == "nhwc" else torch.contiguous_format
    assert actual.is_contiguous(memory_format=memory_format)
    assert conv_padding == (0, 0)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("layout", ["nchw", "nhwc"])
@pytest.mark.parametrize(
    "padding_mode,padding",
    [(mode, 0) for mode in _PADDING_MODES]
    + [(mode, (0, 0)) for mode in _PADDING_MODES]
    + [("zeros", 2), ("zeros", (1, 3))],
)
def test_prepared_padding_noop_does_not_allocate_or_launch(
    monkeypatch, padding_mode, padding, layout
):
    source = torch.ones(2, 3, 5, 7).transpose(2, 3)

    def unexpected_allocation(*args, **kwargs):
        pytest.fail("no-op padding must not allocate an output")

    monkeypatch.setattr(padding_ops, "_alloc_output", unexpected_allocation)
    monkeypatch.setattr(padding_ops, "_get_config_padding", None)
    monkeypatch.setattr(padding_ops, "_pad_conv2d_kernel", None)
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
        ((1, 2, 3, 5, 7), 0, "zeros", "nchw", "must be 4-D NCHW"),
        ((2, 3, 0, 7), 0, "zeros", "nchw", "spatial dimensions must be positive"),
        ((2, 3, 5, 7), 0, "invalid", "nchw", "padding_mode must be one of"),
        ((2, 3, 5, 7), 0, "reflect", "invalid", "layout must be"),
        ((2, 3, 5, 7), -1, "replicate", "nchw", "must be non-negative"),
        ((2, 3, 5, 7), (1, 2, 3), "replicate", "nchw", "length-2 sequence"),
        ((2, 3, 5, 7), (5, 0), "reflect", "nchw", "reflect padding"),
        ((2, 3, 5, 7), (0, 8), "circular", "nchw", "circular padding"),
        ((2, 3, 5, 7), 1, "replicate", "nchw", "requires a CUDA/ROCm tensor"),
    ],
)
def test_prepared_padding_validates_before_launch(
    shape, padding, padding_mode, layout, message, monkeypatch
):
    source = torch.empty(shape)
    monkeypatch.setattr(padding_ops, "_get_config_padding", None)
    monkeypatch.setattr(padding_ops, "_pad_conv2d_kernel", None)
    with pytest.raises(ValueError, match=message):
        padding_ops._prepare_conv2d_padding(
            source, padding, padding_mode, layout=layout
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
@pytest.mark.parametrize("layout", ["nchw", "nhwc"])
@pytest.mark.parametrize("padding_mode", ["reflect", "replicate", "circular"])
def test_prepared_padding_empty_batch_does_not_launch(
    padding_mode, layout, monkeypatch
):
    source = torch.empty(0, 3, 5, 7, device="cuda", dtype=torch.float16)
    monkeypatch.setattr(padding_ops, "_get_config_padding", None)
    monkeypatch.setattr(padding_ops, "_pad_conv2d_kernel", None)
    actual, conv_padding = padding_ops._prepare_conv2d_padding(
        source, (1, 2), padding_mode, layout=layout
    )
    memory_format = torch.channels_last if layout == "nhwc" else torch.contiguous_format
    assert actual.shape == (0, 3, 7, 11)
    assert actual.is_contiguous(memory_format=memory_format)
    assert actual.dtype == source.dtype
    assert actual.device == source.device
    assert conv_padding == (0, 0)


@pytest.mark.parametrize("arch", _PADDING_CONFIG_ARCHS)
def test_padding_config_defaults(monkeypatch, isolated_conv_config_cache, arch):
    _use_arch(monkeypatch, arch)
    config = padding_ops._get_config_padding(M=1024)
    assert config == {"BLOCK_SIZE": 256, "num_warps": 4}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
@pytest.mark.parametrize("layout", ["nchw", "nhwc"])
def test_padding_launch_uses_json_parameters(
    monkeypatch, isolated_conv_config_cache, layout
):
    parameters = {"BLOCK_SIZE": 128, "num_warps": 2}
    config_reads = []
    launches = []
    kernel = padding_ops._pad_conv2d_kernel

    def config_file(config_name):
        config_reads.append(config_name)
        return {"any": parameters}

    class RecordingKernel:
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                compiled_kernel = kernel[grid](*args, **kwargs)
                launches.append((grid, kwargs, compiled_kernel.name))
                return compiled_kernel

            return launch

    monkeypatch.setattr(conv_config_utils, "_get_conv_config_file", config_file)
    monkeypatch.setattr(padding_ops, "_pad_conv2d_kernel", RecordingKernel())
    source = torch.arange(210, device="cuda", dtype=torch.float16).reshape(2, 3, 5, 7)
    actual, conv_padding = padding_ops._prepare_conv2d_padding(
        source, (1, 2), "reflect", layout=layout
    )

    assert config_reads == ["CONV-PADDING"]
    assert len(launches) == 1
    grid, launch_kwargs, kernel_name = launches[0]
    assert grid == (triton.cdiv(actual.numel(), parameters["BLOCK_SIZE"]),)
    assert launch_kwargs["BLOCK_SIZE"] == parameters["BLOCK_SIZE"]
    assert launch_kwargs["num_warps"] == parameters["num_warps"]
    assert kernel_name == (
        f"_pad_conv2d_kernel_BLOCK_SIZE_{parameters['BLOCK_SIZE']}_"
        f"PADDING_MODE_REFLECT_CHANNELS_LAST_{int(layout == 'nhwc')}"
    )
    assert conv_padding == (0, 0)
    expected = F.pad(source, (2, 2, 1, 1), mode="reflect")
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
