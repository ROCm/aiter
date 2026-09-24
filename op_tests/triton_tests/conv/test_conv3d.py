# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Pytest unit tests for aiter.ops.triton.conv.conv3d.

Checks numerical correctness, packing, caching, routing, and supported public
API validation. Numerical tests compare the auto-routed Triton conv3d kernels
against torch.nn.functional.conv3d on synthetic tensors, in both NCDHW and
NDHWC (channels-last-3d) layouts. No model loading, no network.

The headline shapes are the Wan-style VAE encoder/decoder 3x3x3 convs
(stride 1, pad 1) the kernel was first built for; encoder and decoder
collapse to the same 4 distinct shapes. A few small extra shapes exercise
stride/pad/dilation and non-3x3x3 kernels through the same general path.

Tolerances reuse the shared dynamic model with K_red = C*T*R*S. The same
module also covers configuration lookup.
"""

import pytest
import torch
import torch.nn.functional as F

from aiter.ops.triton.conv import _launch as conv_launch
from aiter.ops.triton.conv import _prepack as conv_prepack
from aiter.ops.triton.conv import conv3d as conv3d_module
from aiter.ops.triton.conv._prepack import (
    clear_conv2d_weight_pack_caches,
    clear_conv3d_weight_pack_caches,
    prepack_ncdhw_to_cblocked,
    prepack_oidhw_to_3x3x3,
    prepack_oidhw_to_kmajor,
    prepack_winograd_hw_filter_f4x3,
)
from aiter.ops.triton.conv._utils import (
    _conv3d_dims,
    _winograd_transform_storage_dtype,
)
from aiter.ops.triton.conv.conv3d import (
    Route3D,
    _resolve_route,
    conv3d,
    conv3d_1x1x1,
    conv3d_general,
    conv3d_ncdhw_cblocked,
    conv3d_ndhwc_3x3x3,
    conv3d_winograd_hw_f4x3,
    conv3d_winograd_hw_f4x3_cblocked,
)
from aiter.ops.triton.utils import conv_config_utils
from aiter.ops.triton.utils._triton.arch_info import get_arch
from aiter.ops.triton.utils.conv_config_utils import (
    format_prepack_shape_key_3d,
    format_shape_key_3d,
)

from ._helpers import (
    ALL_SUPPORTED_ARCHS,
    _winograd_tolerances,
    apply_activation,
    dynamic_conv_tolerances,
)

_current_arch = get_arch()
if _current_arch not in ALL_SUPPORTED_ARCHS:
    pytest.skip(
        f"aiter.ops.triton.conv tests run on {sorted(ALL_SUPPORTED_ARCHS)}; "
        f"current arch {_current_arch!r} not supported",
        allow_module_level=True,
    )


# (name, N, C, D, H, W, K) — all 3x3x3, stride 1, pad 1, dil 1.
VAE_SHAPES = [
    ("vae_384_3x46x51", 1, 384, 3, 46, 51, 384),
    ("vae_96_6x354x394", 1, 96, 6, 354, 394, 96),
    ("vae_192_6x178x198", 1, 192, 6, 178, 198, 192),
    ("vae_384_4x90x100", 1, 384, 4, 90, 100, 384),
]

# (name, N, C, D, H, W, K, T, R, S, stride, pad, dil) — covers both the general
# path and the specialized 1x1x1 path (routing verified in test_routing).
EXTRA_SHAPES = [
    ("stride2", 1, 32, 8, 32, 32, 32, 3, 3, 3, (2, 2, 2), (1, 1, 1), (1, 1, 1)),
    ("pad0", 1, 16, 6, 24, 24, 16, 3, 3, 3, (1, 1, 1), (0, 0, 0), (1, 1, 1)),
    ("k1x1x1", 1, 64, 4, 16, 16, 128, 1, 1, 1, (1, 1, 1), (0, 0, 0), (1, 1, 1)),
    ("k1x1x1_proj", 1, 384, 3, 46, 51, 192, 1, 1, 1, (1, 1, 1), (0, 0, 0), (1, 1, 1)),
    ("k1x1x1_s2", 1, 96, 6, 32, 32, 96, 1, 1, 1, (2, 2, 2), (0, 0, 0), (1, 1, 1)),
    ("k5x5x5", 1, 16, 8, 20, 20, 16, 5, 5, 5, (1, 1, 1), (2, 2, 2), (1, 1, 1)),
    ("dilated", 1, 16, 8, 24, 24, 16, 3, 3, 3, (1, 1, 1), (2, 2, 2), (2, 2, 2)),
]

DTYPES = [(torch.float16, "fp16"), (torch.bfloat16, "bf16")]
_WINOGRAD_NORMALIZED_MAX_ERROR = 0.025
_WINOGRAD_RELATIVE_L2_ERROR = 0.01


def _assert_conv3d_result(
    y, ref, dtype, K_red, *, is_winograd=False, normalization_reference=None
):
    """Check the public result contract and guard against permissive tolerances."""
    assert y.dtype == dtype, f"expected output dtype {dtype}, got {y.dtype}"
    assert y.shape == ref.shape, f"output shape {y.shape} != reference {ref.shape}"
    assert torch.isfinite(y).all(), "Conv3D output contains non-finite values"
    assert torch.isfinite(ref).all(), "reference output contains non-finite values"

    if is_winograd:
        rtol, atol = _winograd_tolerances(dtype, K_red)
    else:
        rtol, atol = dynamic_conv_tolerances(dtype, K_red)
    y32 = y.float()
    ref32 = ref.float()
    torch.testing.assert_close(
        y32,
        ref32,
        rtol=rtol,
        atol=atol,
        msg=lambda msg: (
            f"Conv3D result mismatch: dtype={dtype}, K_red={K_red}, "
            f"is_winograd={is_winograd}\n\n{msg}"
        ),
    )

    if is_winograd:
        error = (y32 - ref32).abs()
        scale_tensor = (
            ref32
            if normalization_reference is None
            else normalization_reference.float()
        )
        reference_scale = (
            scale_tensor.abs().max().clamp_min(torch.finfo(torch.float32).tiny)
        )
        reference_norm = torch.linalg.vector_norm(ref32).clamp_min(
            torch.finfo(torch.float32).tiny
        )
        normalized_max_error = error.max() / reference_scale
        relative_l2_error = torch.linalg.vector_norm(error) / reference_norm
        assert normalized_max_error < _WINOGRAD_NORMALIZED_MAX_ERROR, (
            f"Winograd normalized max error {normalized_max_error} exceeds "
            f"{_WINOGRAD_NORMALIZED_MAX_ERROR}"
        )
        assert relative_l2_error < _WINOGRAD_RELATIVE_L2_ERROR, (
            f"Winograd relative L2 error {relative_l2_error} exceeds "
            f"{_WINOGRAD_RELATIVE_L2_ERROR}"
        )


def _run_case(
    N,
    C,
    D,
    H,
    W,
    K,
    T,
    R,
    S,
    stride,
    pad,
    dil,
    dtype,
    bias,
    act,
    layout="ncdhw",
    force_winograd=False,
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    torch.manual_seed(0)
    x = torch.randn(N, C, D, H, W, device="cuda", dtype=dtype)
    w = torch.randn(K, C, T, R, S, device="cuda", dtype=dtype)
    b = torch.randn(K, device="cuda", dtype=dtype) if bias else None

    if force_winograd:
        assert layout == "ncdhw", f"forced Winograd requires NCDHW, got {layout}"
        y = conv3d_winograd_hw_f4x3(
            x, w, b, stride=stride, padding=pad, dilation=dil, activation=act
        )
    else:
        y = conv3d(
            x,
            w,
            b,
            stride=stride,
            padding=pad,
            dilation=dil,
            activation=act,
            layout=layout,
        )

    ref = F.conv3d(
        x.float(),
        w.float(),
        b.float() if b is not None else None,
        stride=stride,
        padding=pad,
        dilation=dil,
    )
    normalization_reference = ref
    ref = apply_activation(ref, act)

    K_red = C * T * R * S
    # The Winograd path amplifies rounding — use the 6x-bumped tolerance when
    # this shape/layout actually routes there.
    route = (
        Route3D.WINOGRAD_HW
        if force_winograd
        else _resolve_route(
            T,
            R,
            S,
            stride,
            dil,
            N,
            C,
            D,
            H,
            W,
            K,
            layout,
            padding=pad,
        )
    )
    _assert_conv3d_result(
        y,
        ref,
        dtype,
        K_red,
        is_winograd=route in (Route3D.WINOGRAD_HW, Route3D.WINOGRAD_HW_CBLOCKED),
        normalization_reference=normalization_reference,
    )
    if layout == "ndhwc":
        assert y.is_contiguous(
            memory_format=torch.channels_last_3d
        ), f"expected channels-last-3d output, got strides={y.stride()}"
    else:
        assert (
            y.is_contiguous()
        ), f"expected contiguous NCDHW output, got strides={y.stride()}"


LAYOUTS = ["ncdhw", "ndhwc"]


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("dtype,dtype_id", DTYPES, ids=[d[1] for d in DTYPES])
@pytest.mark.parametrize("shape", VAE_SHAPES, ids=[s[0] for s in VAE_SHAPES])
def test_vae_shapes(shape, dtype, dtype_id, layout):
    _, N, C, D, H, W, K = shape
    _run_case(
        N,
        C,
        D,
        H,
        W,
        K,
        3,
        3,
        3,
        (1, 1, 1),
        (1, 1, 1),
        (1, 1, 1),
        dtype,
        bias=True,
        act="none",
        layout=layout,
    )


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("dtype,dtype_id", DTYPES, ids=[d[1] for d in DTYPES])
@pytest.mark.parametrize("act", ["relu", "relu6", "gelu"])
def test_bias_and_activation(dtype, dtype_id, layout, act):
    # Exercise every fused epilogue on the Winograd/NCDHW and direct/NDHWC
    # 3x3x3 routes without paying the full VAE-shape cost for each activation.
    _run_case(
        1,
        64,
        3,
        16,
        16,
        64,
        3,
        3,
        3,
        (1, 1, 1),
        (1, 1, 1),
        (1, 1, 1),
        dtype,
        bias=True,
        act=act,
        layout=layout,
        force_winograd=layout == "ncdhw",
    )


@pytest.mark.parametrize("layout", LAYOUTS)
def test_scalar_parameters_and_noncontiguous_input(layout):
    """Public Conv3D accepts scalar parameters and materializes sliced inputs."""
    torch.manual_seed(0)
    x_base = torch.randn(1, 32, 4, 12, 18, device="cuda", dtype=torch.float16)
    x = x_base[..., ::2]
    assert not x.is_contiguous(), f"expected sliced input, got strides={x.stride()}"
    w = torch.randn(48, 32, 1, 1, 1, device="cuda", dtype=torch.float16)

    y = conv3d(x, w, stride=1, padding=0, dilation=1, layout=layout)
    ref = F.conv3d(x.float(), w.float())
    rtol, atol = dynamic_conv_tolerances(torch.float16, 32)
    torch.testing.assert_close(
        y.float(),
        ref,
        rtol=rtol,
        atol=atol,
        msg=lambda msg: (
            f"Scalar-parameter Conv3D mismatch: layout={layout}, "
            f"input_strides={x.stride()}\n\n{msg}"
        ),
    )
    assert y.dtype == torch.float16, f"expected float16 output, got {y.dtype}"
    if layout == "ndhwc":
        assert y.is_contiguous(
            memory_format=torch.channels_last_3d
        ), f"expected channels-last-3d output, got strides={y.stride()}"
    else:
        assert (
            y.is_contiguous()
        ), f"expected contiguous NCDHW output, got strides={y.stride()}"


def test_ncdhw_to_cblocked_mapping_and_zero_padding():
    N, C, D, H, W, block_c = 2, 5, 2, 2, 3, 4
    x = torch.arange(N * C * D * H * W, device="cuda", dtype=torch.float16).reshape(
        N, C, D, H, W
    )

    packed, C_pad = prepack_ncdhw_to_cblocked(x, block_c)

    C_blocks = (C + block_c - 1) // block_c
    expected = torch.zeros(
        (N, C_blocks, D, H, W, block_c), device=x.device, dtype=x.dtype
    )
    for c in range(C):
        expected[:, c // block_c, ..., c % block_c] = x[:, c]

    assert (
        C_pad == C_blocks * block_c
    ), f"expected padded channels {C_blocks * block_c}, got {C_pad}"
    assert (
        packed.shape == expected.shape
    ), f"packed shape {packed.shape} != expected {expected.shape}"
    assert (
        packed.is_contiguous()
    ), f"packed strides are not contiguous: {packed.stride()}"
    torch.testing.assert_close(
        packed,
        expected,
        rtol=0,
        atol=0,
        msg=lambda msg: (
            f"NCDHW-to-cblocked mapping mismatch: C={C}, block_c={block_c}, "
            f"C_pad={C_pad}\n\n{msg}"
        ),
    )
    assert (
        torch.count_nonzero(packed[:, -1, ..., C % block_c :]) == 0
    ), "channel-padding region contains nonzero values"


def test_oidhw_to_kmajor_prepack_mapping_and_zero_padding():
    K_out, C, T, R, S, block_k = 2, 3, 2, 1, 2, 16
    w = torch.arange(K_out * C * T * R * S, dtype=torch.float16).reshape(
        K_out, C, T, R, S
    )

    packed, K_pad = prepack_oidhw_to_kmajor(w, block_k)

    K_red = C * T * R * S
    assert K_pad == block_k, f"expected K padding {block_k}, got {K_pad}"
    assert packed.shape == (
        K_out,
        K_pad,
    ), f"packed shape {packed.shape} != expected {(K_out, K_pad)}"
    assert (
        packed.is_contiguous()
    ), f"packed strides are not contiguous: {packed.stride()}"
    torch.testing.assert_close(
        packed[:, :K_red],
        w.reshape(K_out, K_red),
        msg=lambda msg: (
            f"OIDHW-to-K-major mapping mismatch: K_red={K_red}, "
            f"block_k={block_k}, K_pad={K_pad}\n\n{msg}"
        ),
    )
    assert (
        torch.count_nonzero(packed[:, K_red:]) == 0
    ), "K-padding region contains nonzero values"


def test_oidhw_to_3x3x3_prepack_mapping_and_zero_padding():
    K_out, C, block_c = 2, 3, 4
    w = torch.arange(K_out * C * 3 * 3 * 3, dtype=torch.float16).reshape(
        K_out, C, 3, 3, 3
    )

    packed, C_pad = prepack_oidhw_to_3x3x3(w, block_c)

    expected = w.reshape(K_out, C, 27).permute(0, 2, 1)
    assert C_pad == block_c, f"expected channel padding {block_c}, got {C_pad}"
    assert packed.shape == (
        K_out,
        27,
        C_pad,
    ), f"packed shape {packed.shape} != expected {(K_out, 27, C_pad)}"
    assert (
        packed.is_contiguous()
    ), f"packed strides are not contiguous: {packed.stride()}"
    torch.testing.assert_close(
        packed[:, :, :C],
        expected,
        msg=lambda msg: (
            f"OIDHW-to-3x3x3 mapping mismatch: C={C}, block_c={block_c}, "
            f"C_pad={C_pad}\n\n{msg}"
        ),
    )
    assert (
        torch.count_nonzero(packed[:, :, C:]) == 0
    ), "channel-padding region contains nonzero values"


def test_weight_prepack_cache_reuses_and_invalidates(monkeypatch):
    cache = conv_prepack._LRUPackCache(maxsize=2)
    monkeypatch.setattr(conv_prepack, "_PACK_CACHE_3D_GENERAL", cache)
    w = torch.arange(3, dtype=torch.float16).reshape(1, 1, 1, 1, 3)

    first, first_pad = conv_prepack.get_or_make_weight_pack_3d(w, block_k=4)
    cached, cached_pad = conv_prepack.get_or_make_weight_pack_3d(w, block_k=4)

    assert cached is first, "identical weight did not reuse its cached pack"
    assert (
        cached_pad == first_pad
    ), f"cached padding {cached_pad} != original {first_pad}"

    w.add_(10)
    refreshed, refreshed_pad = conv_prepack.get_or_make_weight_pack_3d(w, block_k=4)

    assert refreshed is not first, "in-place weight update reused a stale pack"
    assert (
        refreshed_pad == first_pad
    ), f"refreshed padding {refreshed_pad} != original {first_pad}"
    torch.testing.assert_close(
        refreshed[:, :3],
        w.reshape(1, 3),
        msg=lambda msg: f"refreshed weight pack contains stale data\n\n{msg}",
    )
    assert (
        torch.count_nonzero(refreshed[:, 3:]) == 0
    ), "refreshed K-padding region contains nonzero values"


def test_weight_prepack_cache_uses_lru_eviction(monkeypatch):
    cache = conv_prepack._LRUPackCache(maxsize=2)
    monkeypatch.setattr(conv_prepack, "_PACK_CACHE_3D_GENERAL", cache)
    weights = [
        (torch.arange(3, dtype=torch.float16) + offset).reshape(1, 1, 1, 1, 3)
        for offset in (0, 10, 20)
    ]

    first, _ = conv_prepack.get_or_make_weight_pack_3d(weights[0], block_k=4)
    second, _ = conv_prepack.get_or_make_weight_pack_3d(weights[1], block_k=4)
    first_hit, _ = conv_prepack.get_or_make_weight_pack_3d(weights[0], block_k=4)
    conv_prepack.get_or_make_weight_pack_3d(weights[2], block_k=4)

    assert first_hit is first, "cache hit did not return the original packed weight"
    assert len(cache._d) == 2, f"expected two cached entries, got {len(cache._d)}"
    assert (
        conv_prepack.get_or_make_weight_pack_3d(weights[0], block_k=4)[0] is first
    ), "recently used entry was unexpectedly evicted"
    assert (
        conv_prepack.get_or_make_weight_pack_3d(weights[1], block_k=4)[0] is not second
    ), "least-recently-used entry was not evicted"


@pytest.mark.parametrize(
    "clear_caches,cleared_names,preserved_names",
    [
        (
            clear_conv2d_weight_pack_caches,
            ("_PACK_CACHE", "_PACK_CACHE_3x3", "_PACK_CACHE_WINOGRAD_F4X3"),
            (
                "_PACK_CACHE_3D_GENERAL",
                "_PACK_CACHE_3D_3X3X3",
                "_PACK_CACHE_3D_WINOGRAD_HW",
            ),
        ),
        (
            clear_conv3d_weight_pack_caches,
            (
                "_PACK_CACHE_3D_GENERAL",
                "_PACK_CACHE_3D_3X3X3",
                "_PACK_CACHE_3D_WINOGRAD_HW",
            ),
            ("_PACK_CACHE", "_PACK_CACHE_3x3", "_PACK_CACHE_WINOGRAD_F4X3"),
        ),
    ],
    ids=["conv2d", "conv3d"],
)
def test_weight_pack_cache_clear_is_dimension_specific(
    monkeypatch, clear_caches, cleared_names, preserved_names
):
    caches = {}
    for name in (*cleared_names, *preserved_names):
        cache = conv_prepack._LRUPackCache(maxsize=1)
        cache.put(name, object(), object())
        monkeypatch.setattr(conv_prepack, name, cache)
        caches[name] = cache

    clear_caches()

    assert all(
        not caches[name]._d for name in cleared_names
    ), f"expected caches to be cleared: {cleared_names}"
    assert all(
        caches[name]._d for name in preserved_names
    ), f"expected caches to be preserved: {preserved_names}"


def test_general_masks_weight_tail_when_block_k_exceeds_pack_granularity(monkeypatch):
    config = {
        "BLOCK_M": 32,
        "BLOCK_N": 32,
        "BLOCK_K": 128,
        "GROUP_SIZE_M": 4,
        "num_warps": 4,
        "num_stages": 1,
    }
    monkeypatch.setattr(conv_launch, "_get_config_general_3d", lambda **_: config)
    torch.manual_seed(0)
    x = torch.randn((1, 3, 2, 8, 8), device="cuda", dtype=torch.float16)
    w = torch.randn((5, 3, 2, 3, 3), device="cuda", dtype=torch.float16)
    bias = torch.randn((5,), device="cuda", dtype=torch.float16)

    # K_red=54 and the normal 64-lane prepack gives K_pad=64. BLOCK_K=128
    # therefore exercises the masked second half of the final weight load.
    y = conv3d_general(x, w, bias, padding=(0, 1, 1), block_k=64)
    reference = F.conv3d(x.float(), w.float(), bias.float(), padding=(0, 1, 1))
    rtol, atol = dynamic_conv_tolerances(torch.float16, 54)
    torch.testing.assert_close(
        y.float(),
        reference,
        rtol=rtol,
        atol=atol,
        msg=lambda msg: (
            "general Conv3D tail masking mismatch: K_red=54, block_k=64\n\n" f"{msg}"
        ),
    )
    assert y.dtype == torch.float16, f"expected float16 output, got {y.dtype}"


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"stride": (1, 1)}, "stride must be an int or a length-3 sequence"),
        ({"stride": 0}, "stride values must be positive"),
        ({"padding": -1}, "padding values must be non-negative"),
        ({"dilation": 0}, "dilation values must be positive"),
    ],
    ids=["stride-length", "stride-zero", "padding-negative", "dilation-zero"],
)
def test_invalid_spatial_parameters(kwargs, match):
    x = torch.empty((1, 4, 3, 4, 4), dtype=torch.float16)
    w = torch.empty((4, 4, 1, 1, 1), dtype=torch.float16)

    with pytest.raises(ValueError, match=match):
        conv3d(x, w, **kwargs)


def test_invalid_layout():
    x = torch.empty((1, 4, 3, 4, 4), dtype=torch.float16)
    w = torch.empty((4, 4, 1, 1, 1), dtype=torch.float16)

    with pytest.raises(ValueError, match="layout must be 'ncdhw' or 'ndhwc'"):
        conv3d(x, w, layout="nchw")


def test_conv3d_dims_supports_cpu_tensors():
    x = torch.empty((2, 4, 5, 6, 7), dtype=torch.float16)
    w = torch.empty((8, 4, 3, 3, 3), dtype=torch.float16)

    dimensions = _conv3d_dims(
        x,
        w,
        stride=(1, 1, 1),
        padding=(1, 1, 1),
        dilation=(1, 1, 1),
    )

    expected = (2, 4, 5, 6, 7, 8, 3, 3, 3, 5, 6, 7)
    assert dimensions == expected, f"dimensions {dimensions} != expected {expected}"


def test_conv3d_execution_rejects_cpu_tensors():
    x = torch.empty((1, 4, 3, 4, 4), dtype=torch.float16)
    w = torch.empty((4, 4, 1, 1, 1), dtype=torch.float16)

    with pytest.raises(ValueError, match="requires CUDA input and weight tensors"):
        conv3d_general(x, w)


def test_input_weight_channel_mismatch():
    x = torch.empty((1, 4, 3, 4, 4), device="cuda", dtype=torch.float16)
    w = torch.empty((4, 3, 1, 1, 1), device="cuda", dtype=torch.float16)

    with pytest.raises(ValueError, match="weight in-channels 3 != input channels 4"):
        conv3d(x, w)


def test_nonpositive_output_shape():
    x = torch.empty((1, 4, 2, 4, 4), device="cuda", dtype=torch.float16)
    w = torch.empty((4, 4, 3, 3, 3), device="cuda", dtype=torch.float16)

    with pytest.raises(ValueError, match="calculated conv3d output is non-positive"):
        conv3d(x, w)


def test_3x3x3_prepack_rejects_other_kernel_shapes():
    w = torch.empty((4, 4, 1, 3, 3), dtype=torch.float16)

    with pytest.raises(ValueError, match="3x3x3 prepack requires a 3x3x3 weight"):
        prepack_oidhw_to_3x3x3(w)


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("dtype,dtype_id", DTYPES, ids=[d[1] for d in DTYPES])
@pytest.mark.parametrize("shape", EXTRA_SHAPES, ids=[s[0] for s in EXTRA_SHAPES])
def test_extra_shapes(shape, dtype, dtype_id, layout):
    _name, N, C, D, H, W, K, T, R, S, stride, pad, dil = shape
    _run_case(
        N,
        C,
        D,
        H,
        W,
        K,
        T,
        R,
        S,
        stride,
        pad,
        dil,
        dtype,
        bias=False,
        act="none",
        layout=layout,
    )


ADVERSARIAL_SHAPES = [
    (
        "asymmetric_general_n2_c5_k7",
        2,
        5,
        5,
        9,
        13,
        7,
        2,
        3,
        1,
        (1, 2, 3),
        (0, 1, 2),
        (2, 1, 3),
    ),
    (
        "specialized_3x3x3_n2_c65_k67",
        2,
        65,
        4,
        9,
        11,
        67,
        3,
        3,
        3,
        (1, 1, 1),
        (1, 1, 1),
        (1, 1, 1),
    ),
]


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("dtype,dtype_id", DTYPES, ids=[d[1] for d in DTYPES])
@pytest.mark.parametrize(
    "shape", ADVERSARIAL_SHAPES, ids=[shape[0] for shape in ADVERSARIAL_SHAPES]
)
def test_adversarial_shapes(shape, dtype, dtype_id, layout):
    _name, N, C, D, H, W, K, T, R, S, stride, pad, dil = shape
    _run_case(
        N,
        C,
        D,
        H,
        W,
        K,
        T,
        R,
        S,
        stride,
        pad,
        dil,
        dtype,
        bias=True,
        act="none",
        layout=layout,
    )


DIRECT_METHOD_CASES = [
    pytest.param(
        conv3d_1x1x1,
        "ncdhw",
        (2, 5, 3, 5, 7, 7, 1, 1, 1),
        (1, 2, 1),
        (0, 0, 0),
        (1, 1, 1),
        False,
        id="1x1x1",
    ),
    pytest.param(
        conv3d_general,
        "ndhwc",
        (2, 5, 5, 9, 13, 7, 2, 3, 1),
        (1, 2, 3),
        (0, 1, 2),
        (2, 1, 3),
        False,
        id="general",
    ),
    pytest.param(
        conv3d_ncdhw_cblocked,
        "ncdhw",
        (2, 65, 4, 9, 11, 67, 3, 3, 3),
        (1, 1, 1),
        (1, 1, 1),
        (1, 1, 1),
        False,
        id="cblocked",
    ),
    pytest.param(
        conv3d_ndhwc_3x3x3,
        "ndhwc",
        (2, 65, 4, 9, 11, 67, 3, 3, 3),
        (1, 1, 1),
        (1, 1, 1),
        (1, 1, 1),
        False,
        id="ndhwc-3x3x3",
    ),
    pytest.param(
        conv3d_winograd_hw_f4x3,
        "ncdhw",
        (2, 17, 4, 9, 11, 19, 3, 3, 3),
        (1, 1, 1),
        (1, 1, 1),
        (1, 1, 1),
        True,
        id="winograd",
    ),
    pytest.param(
        conv3d_winograd_hw_f4x3_cblocked,
        "ncdhw",
        (2, 17, 4, 9, 11, 19, 3, 3, 3),
        (1, 1, 1),
        (1, 1, 1),
        (1, 1, 1),
        True,
        id="winograd-cblocked",
    ),
]

EPILOGUE_CASES = [
    pytest.param(False, "none", id="no-bias-none"),
    pytest.param(True, "relu", id="bias-relu"),
    pytest.param(False, "relu6", id="no-bias-relu6"),
    pytest.param(True, "gelu", id="bias-gelu"),
]


@pytest.mark.parametrize("bias,activation", EPILOGUE_CASES)
@pytest.mark.parametrize("dtype,dtype_id", DTYPES, ids=[d[1] for d in DTYPES])
@pytest.mark.parametrize(
    "method,layout,shape,stride,padding,dilation,is_winograd", DIRECT_METHOD_CASES
)
def test_direct_methods_support_bias_and_activations(
    method,
    layout,
    shape,
    stride,
    padding,
    dilation,
    is_winograd,
    dtype,
    dtype_id,
    bias,
    activation,
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    torch.manual_seed(0)
    N, C, D, H, W, K, T, R, S = shape
    x = torch.randn((N, C, D, H, W), device="cuda", dtype=dtype)
    if layout == "ndhwc":
        x = x.contiguous(memory_format=torch.channels_last_3d)
    w = torch.randn((K, C, T, R, S), device="cuda", dtype=dtype)
    if is_winograd:
        w = w * 0.1
    b = torch.randn((K,), device="cuda", dtype=dtype) if bias else None
    kwargs = {
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "activation": activation,
    }
    if method in (conv3d_1x1x1, conv3d_general):
        kwargs["layout"] = layout

    try:
        y = method(x, w, b, **kwargs)
        ref = F.conv3d(
            x.float(),
            w.float(),
            b.float() if b is not None else None,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        normalization_reference = ref
        ref = apply_activation(ref, activation)
        _assert_conv3d_result(
            y,
            ref,
            dtype,
            C * T * R * S,
            is_winograd=is_winograd,
            normalization_reference=normalization_reference,
        )
        if layout == "ndhwc":
            assert y.is_contiguous(
                memory_format=torch.channels_last_3d
            ), f"expected channels-last-3d output, got strides={y.stride()}"
        else:
            assert (
                y.is_contiguous()
            ), f"expected contiguous NCDHW output, got strides={y.stride()}"
    finally:
        clear_conv3d_weight_pack_caches()


@pytest.mark.parametrize(
    "wino_fn",
    [conv3d_winograd_hw_f4x3, conv3d_winograd_hw_f4x3_cblocked],
    ids=["plain", "cblocked"],
)
@pytest.mark.parametrize("dtype,dtype_id", DTYPES, ids=[d[1] for d in DTYPES])
def test_winograd_variants(wino_fn, dtype, dtype_id):
    """Both Winograd input variants (plain NCDHW read, cblocked NCDHWc read)
    match F.conv3d within the 6x Winograd tolerance. The cblocked variant is not
    routed by default (never faster here) but is kept available, so it needs
    explicit coverage; the plain variant is also covered via routing."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    torch.manual_seed(0)
    for N, C, D, H, W, K in [(1, 96, 4, 32, 40, 96), (1, 192, 3, 24, 28, 192)]:
        x = torch.randn(N, C, D, H, W, device="cuda", dtype=dtype)
        w = torch.randn(K, C, 3, 3, 3, device="cuda", dtype=dtype) * 0.1
        b = torch.randn(K, device="cuda", dtype=dtype)
        y = wino_fn(x, w, b, stride=(1, 1, 1), padding=(1, 1, 1))
        ref = F.conv3d(x.float(), w.float(), b.float(), stride=1, padding=1)
        _assert_conv3d_result(y, ref, dtype, C * 27, is_winograd=True)


def test_bf16_winograd_uses_fp16_transform_storage():
    """BF16 I/O uses FP16 Winograd operands to avoid a second BF16 rounding."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    assert (
        _winograd_transform_storage_dtype(torch.bfloat16) == torch.float16
    ), "BF16 Winograd weights should use FP16 transform storage"
    assert (
        _winograd_transform_storage_dtype(torch.float16) == torch.float16
    ), "FP16 Winograd weights should retain FP16 transform storage"

    torch.manual_seed(0)
    N, C, D, H, W, K = 1, 96, 4, 32, 40, 96
    x = torch.randn(N, C, D, H, W, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(K, C, 3, 3, 3, device="cuda", dtype=torch.bfloat16) * 0.1
    b = torch.randn(K, device="cuda", dtype=torch.bfloat16)

    transformed_w, _ = prepack_winograd_hw_filter_f4x3(w)
    assert (
        transformed_w.dtype == torch.float16
    ), f"expected FP16 transformed weights, got {transformed_w.dtype}"
    assert torch.isfinite(
        transformed_w
    ).all(), "transformed Winograd weights contain non-finite values"

    y = conv3d_winograd_hw_f4x3(x, w, b, padding=(1, 1, 1))
    ref = F.conv3d(x.float(), w.float(), b.float(), padding=1)
    error = (y.float() - ref).abs()
    normalized_max_error = error.max() / ref.abs().max()
    relative_l2_error = torch.linalg.vector_norm(error) / torch.linalg.vector_norm(ref)

    assert y.dtype == torch.bfloat16, f"expected BF16 output, got {y.dtype}"
    assert torch.isfinite(y).all(), "BF16 Winograd output contains non-finite values"
    assert (
        normalized_max_error < 0.015
    ), f"normalized max error {normalized_max_error} exceeds 0.015"
    assert (
        relative_l2_error < 0.006
    ), f"relative L2 error {relative_l2_error} exceeds 0.006"


def test_routing(monkeypatch):
    s1 = (1, 1, 1)

    def route(
        T,
        R,
        S,
        stride=s1,
        dilation=s1,
        C=64,
        D=4,
        H=32,
        W=32,
        K=64,
        layout="ncdhw",
        padding=s1,
    ):
        return _resolve_route(
            T,
            R,
            S,
            stride,
            dilation,
            1,
            C,
            D,
            H,
            W,
            K,
            layout,
            padding=padding,
        )

    # 1x1x1 -> specialized channel-GEMM (layout-independent).
    assert (
        route(1, 1, 1, C=128) is Route3D.ONE_X_ONE_X_ONE
    ), "NCDHW 1x1x1 convolution should use the specialized route"
    assert (
        route(1, 1, 1, C=128, layout="ndhwc") is Route3D.ONE_X_ONE_X_ONE
    ), "NDHWC 1x1x1 convolution should use the specialized route"
    # The measured wave32 policy accounts for depth utilization, channel
    # expansion/compression, tile count, and activation-pack cost.
    monkeypatch.setattr("aiter.ops.triton.conv.conv3d._is_amd_wave32", lambda: True)
    assert (
        route(3, 3, 3, C=384, D=3, H=46, W=51, K=384) is Route3D.WINOGRAD_HW
    ), "expected Wan decoder shape to use Winograd"
    assert (
        route(3, 3, 3, C=192, D=6, H=178, W=198, K=192) is Route3D.WINOGRAD_HW
    ), "expected large low-depth shape to use Winograd"
    assert (
        route(3, 3, 3, C=384, D=4, H=90, W=100, K=384) is Route3D.WINOGRAD_HW
    ), "expected medium-depth shape to use Winograd"
    assert (
        route(3, 3, 3, C=96, D=6, H=354, W=394, K=96) is Route3D.CBLOCKED_NCDHW
    ), "expected low-channel shape to use NCDHWc"
    assert (
        route(3, 3, 3, C=64) is Route3D.CBLOCKED_NCDHW
    ), "expected C=64 shape to use NCDHWc"
    assert (
        route(3, 3, 3, C=32) is Route3D.CBLOCKED_NCDHW
    ), "expected C=32 shape to use NCDHWc"
    assert (
        route(3, 3, 3, C=12, D=3, H=354, W=642, K=160, padding=(0, 0, 0))
        is Route3D.GENERAL
    ), "expected low-channel unpadded shape to use the general route"
    assert (
        route(3, 3, 3, C=384, D=3, H=46, W=51, K=384, padding=(0, 0, 0))
        is Route3D.CBLOCKED_NCDHW
    ), "expected unpadded C=384 shape to use NCDHWc"
    assert (
        route(3, 3, 3, C=640, D=3, H=46, W=82, K=640, padding=(0, 0, 0))
        is Route3D.WINOGRAD_HW_CBLOCKED
    ), "expected wide C=640 shape to use cblocked Winograd"
    assert (
        route(3, 3, 3, C=512, D=33, H=34, W=60, K=4096, padding=(0, 1, 1))
        is Route3D.CBLOCKED_NCDHW
    ), "expected high-depth expansion shape to use NCDHWc"
    assert (
        route(3, 3, 3, stride=(2, 2, 2), C=384) is Route3D.CBLOCKED_NCDHW
    ), "expected strided 3x3x3 shape to use NCDHWc"
    # Wave64 uses the conservative 512-channel crossover.
    monkeypatch.setattr("aiter.ops.triton.conv.conv3d._is_amd_wave32", lambda: False)
    assert (
        route(3, 3, 3, C=384, D=4, H=64, W=64, K=384) is Route3D.CBLOCKED_NCDHW
    ), "expected wave64 C=384 shape to use NCDHWc"
    assert (
        route(3, 3, 3, C=512, D=4, H=64, W=64, K=512) is Route3D.WINOGRAD_HW
    ), "expected wave64 C=512 shape to use Winograd"
    # 3x3x3 NDHWC -> channels-last kernel.
    assert (
        route(3, 3, 3, C=384, layout="ndhwc") is Route3D.NDHWC_3X3X3
    ), "expected NDHWC 3x3x3 shape to use the channels-last kernel"
    # No specialized kernel -> general.
    assert (
        route(5, 5, 5, C=384) is Route3D.GENERAL
    ), "expected 5x5x5 shape to use the general route"
    assert (
        route(3, 3, 3, dilation=(2, 2, 2), C=384) is Route3D.GENERAL
    ), "expected dilated 3x3x3 shape to use the general route"
    assert (
        route(1, 1, 1, dilation=(2, 2, 2), C=384) is Route3D.GENERAL
    ), "expected dilated 1x1x1 shape to use the general route"


@pytest.mark.parametrize(
    "route,expected_wrapper,layout",
    [
        (Route3D.ONE_X_ONE_X_ONE, "conv3d_1x1x1", "ncdhw"),
        (Route3D.WINOGRAD_HW, "conv3d_winograd_hw_f4x3", "ncdhw"),
        (
            Route3D.WINOGRAD_HW_CBLOCKED,
            "conv3d_winograd_hw_f4x3_cblocked",
            "ncdhw",
        ),
        (Route3D.CBLOCKED_NCDHW, "conv3d_ncdhw_cblocked", "ncdhw"),
        (Route3D.NDHWC_3X3X3, "conv3d_ndhwc_3x3x3", "ndhwc"),
        (Route3D.GENERAL, "conv3d_general", "ndhwc"),
    ],
)
def test_route_and_run_dispatches_to_selected_wrapper(
    monkeypatch, route, expected_wrapper, layout
):
    calls = []
    sentinel = object()

    monkeypatch.setattr(
        conv3d_module, "_resolve_route", lambda *_args, **_kwargs: route
    )

    def wrapper(name):
        def run(*args, **kwargs):
            calls.append((name, args, kwargs))
            return sentinel

        return run

    wrapper_names = [
        "conv3d_1x1x1",
        "conv3d_winograd_hw_f4x3",
        "conv3d_winograd_hw_f4x3_cblocked",
        "conv3d_ncdhw_cblocked",
        "conv3d_ndhwc_3x3x3",
        "conv3d_general",
    ]
    for name in wrapper_names:
        monkeypatch.setattr(conv3d_module, name, wrapper(name))

    x = torch.empty((1, 4, 3, 5, 7))
    w = torch.empty((6, 4, 3, 3, 3))
    result = conv3d_module._route_and_run(
        x,
        w,
        None,
        (1, 1, 1),
        (1, 1, 1),
        (1, 1, 1),
        "relu",
        64,
        layout,
    )

    assert result is sentinel, "dispatcher did not return the selected wrapper result"
    called_wrappers = [name for name, _args, _kwargs in calls]
    assert called_wrappers == [
        expected_wrapper
    ], f"called wrappers {called_wrappers}, expected only {expected_wrapper}"
    if expected_wrapper in ("conv3d_1x1x1", "conv3d_general"):
        assert (
            calls[0][2]["layout"] == layout
        ), f"forwarded layout {calls[0][2]['layout']} != expected {layout}"


# -- Configuration lookup and launcher-key regression tests -----------------


@pytest.fixture
def isolated_conv3d_config_cache():
    """Keep synthetic configuration tables and their cache test-local."""
    conv_config_utils._get_conv_config_cached.cache_clear()
    conv_config_utils.has_conv_config.cache_clear()
    yield
    conv_config_utils._get_conv_config_cached.cache_clear()
    conv_config_utils.has_conv_config.cache_clear()


def test_conv3d_config_layout_variant_precedence(
    monkeypatch, isolated_conv3d_config_cache
):
    shape_key = "test-shape"
    config = {
        "shapes_ndhwc": {shape_key: {"source": "layout"}},
        "shapes": {shape_key: {"source": "generic"}},
        "M_LEQ_64": {"source": "bucket"},
        "any": {"source": "any"},
    }
    monkeypatch.setattr(
        conv_config_utils, "_get_conv_config_file", lambda _config_name: config
    )

    def selected(*variants, key=shape_key, M=32):
        return conv_config_utils.get_conv_config(
            "TEST-CONV3D-VARIANTS", shape_key=key, M=M, variants=variants
        )["source"]

    assert selected("ndhwc") == "layout", "layout-specific config was not preferred"
    assert selected() == "generic", "generic shape config was not selected"
    assert (
        selected("ndhwc", key="missing") == "bucket"
    ), "M bucket was not used after a layout-specific shape miss"
    assert (
        selected("ndhwc", key="missing", M=65) == "any"
    ), "generic fallback was not used after shape and bucket misses"


class _KernelSpy:
    def __init__(self):
        self.grid = None
        self.calls = []

    def __getitem__(self, grid):
        self.grid = grid
        return self

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))


def test_prepack_launcher_uses_complete_3d_key(monkeypatch):
    seen = []
    kernel = _KernelSpy()
    monkeypatch.setattr(
        conv_launch,
        "_get_config_prepack_3d",
        lambda **kwargs: seen.append(kwargs) or {},
    )
    monkeypatch.setattr(conv_launch, "_ncx_to_cblocked_kernel", kernel)

    conv_launch._launch_ncdhw_to_cblocked(
        torch.empty(1), torch.empty(1), 2, 65, 3, 5, 7, 128, 64
    )

    expected = [
        {
            "shape_key": format_prepack_shape_key_3d(2, 65, 3, 5, 7, 64),
            "M": 3 * 5 * 7,
        }
    ]
    assert seen == expected, f"prepack config lookup {seen} != expected {expected}"
    assert (
        len(kernel.calls) == 1
    ), f"expected one prepack kernel launch, got {len(kernel.calls)}"


def test_general_launcher_uses_complete_3d_key_and_layout(monkeypatch):
    seen = []
    kernel = _KernelSpy()
    monkeypatch.setattr(
        conv_launch,
        "_get_config_general_3d",
        lambda **kwargs: seen.append(kwargs) or {},
    )
    monkeypatch.setattr(conv_launch, "_conv3d_general_kernel", kernel)

    conv_launch._launch_general_3d(
        torch.empty(1),
        torch.empty(1),
        None,
        torch.empty(1),
        2,
        5,
        5,
        9,
        13,
        7,
        2,
        3,
        1,
        3,
        5,
        6,
        64,
        (1, 2, 3),
        (0, 1, 2),
        (2, 1, 3),
        64,
        "none",
        layout="ndhwc",
    )

    expected_key = format_shape_key_3d(
        2, 5, 5, 9, 13, 7, 2, 3, 1, 1, 2, 3, 0, 1, 2, 2, 1, 3
    )
    expected = [{"shape_key": expected_key, "M": 2 * 3 * 5 * 6, "variants": ("ndhwc",)}]
    assert seen == expected, f"general config lookup {seen} != expected {expected}"
    assert (
        len(kernel.calls) == 1
    ), f"expected one general kernel launch, got {len(kernel.calls)}"


def test_1x1x1_launcher_uses_complete_3d_key_and_layout(monkeypatch):
    seen = []
    kernel = _KernelSpy()
    monkeypatch.setattr(
        conv_launch,
        "_get_config_1x1x1_3d",
        lambda **kwargs: seen.append(kwargs) or {},
    )
    monkeypatch.setattr(conv_launch, "_conv3d_1x1x1_kernel", kernel)

    conv_launch._launch_1x1x1_3d(
        torch.empty(1),
        torch.empty((7, 5, 1, 1, 1)),
        None,
        torch.empty(1),
        2,
        5,
        3,
        5,
        7,
        7,
        3,
        3,
        7,
        (1, 2, 1),
        (0, 0, 0),
        "none",
        layout="ncdhw",
    )

    expected_key = format_shape_key_3d(
        2, 5, 3, 5, 7, 7, 1, 1, 1, 1, 2, 1, 0, 0, 0, 1, 1, 1
    )
    expected = [{"shape_key": expected_key, "M": 2 * 3 * 3 * 7, "variants": ("ncdhw",)}]
    assert seen == expected, f"1x1x1 config lookup {seen} != expected {expected}"
    assert (
        len(kernel.calls) == 1
    ), f"expected one 1x1x1 kernel launch, got {len(kernel.calls)}"


@pytest.mark.parametrize("layout", ["ndhwc", "cblocked"])
def test_3x3x3_launchers_use_complete_3d_key(monkeypatch, layout):
    seen = []
    kernel = _KernelSpy()
    common = [
        torch.empty(1),
        torch.empty(1),
        None,
        torch.empty(1),
        2,
        65,
        4,
        9,
        11,
        67,
        4,
        5,
        6,
        128,
    ]
    tail = [(1, 2, 3), (0, 1, 2), (2, 1, 3), "none"]
    if layout == "ndhwc":
        monkeypatch.setattr(
            conv_launch,
            "_get_config_3x3x3_ndhwc",
            lambda **kwargs: seen.append(kwargs) or {},
        )
        monkeypatch.setattr(conv_launch, "_conv3d_3x3x3_ndhwc_kernel", kernel)
        conv_launch._launch_3x3x3_ndhwc(*common, *tail)
    else:
        monkeypatch.setattr(
            conv_launch,
            "_get_config_3x3x3_cblocked",
            lambda **kwargs: seen.append(kwargs) or {},
        )
        monkeypatch.setattr(conv_launch, "_conv3d_3x3x3_cblocked_kernel", kernel)
        conv_launch._launch_3x3x3_cblocked(*common, 64, *tail)

    expected_key = format_shape_key_3d(
        2, 65, 4, 9, 11, 67, 3, 3, 3, 1, 2, 3, 0, 1, 2, 2, 1, 3
    )
    expected = [{"shape_key": expected_key, "M": 2 * 4 * 5 * 6}]
    assert seen == expected, f"3x3x3 config lookup {seen} != expected {expected}"
    assert (
        len(kernel.calls) == 1
    ), f"expected one {layout} 3x3x3 kernel launch, got {len(kernel.calls)}"


@pytest.mark.parametrize(
    "input_variant,use_cblocked", [("ncdhw", False), ("cblocked", True)]
)
def test_winograd_launchers_use_complete_3d_key_and_axes(
    monkeypatch, input_variant, use_cblocked
):
    input_seen = []
    gemm_seen = []
    output_seen = []
    kernels = [_KernelSpy(), _KernelSpy(), _KernelSpy()]
    monkeypatch.setattr(
        conv_launch,
        "_get_config_wino_hw_input",
        lambda **kwargs: input_seen.append(kwargs) or {},
    )
    monkeypatch.setattr(
        conv_launch,
        "_get_config_wino_hw_gemm",
        lambda **kwargs: gemm_seen.append(kwargs) or {},
    )
    monkeypatch.setattr(
        conv_launch,
        "_get_config_wino_hw_output",
        lambda **kwargs: output_seen.append(kwargs) or {},
    )
    monkeypatch.setattr(
        conv_launch, "_winograd_hw_f4x3_input_transform_kernel", kernels[0]
    )
    monkeypatch.setattr(
        conv_launch, "_winograd_hw_f4x3_batched_gemm_kernel", kernels[1]
    )
    monkeypatch.setattr(
        conv_launch, "_winograd_hw_f4x3_output_transform_kernel", kernels[2]
    )
    x = torch.empty((2, 65, 4, 9, 11), dtype=torch.float16)
    x_blocked = (
        torch.empty((2, 2, 4, 9, 11, 64), dtype=torch.float16) if use_cblocked else None
    )

    conv_launch._launch_winograd_hw_f4x3(
        x,
        torch.empty(1),
        None,
        torch.empty(1),
        2,
        65,
        4,
        9,
        11,
        67,
        4,
        9,
        11,
        128,
        (1, 1, 1),
        "none",
        block_k=64,
        x_blocked=x_blocked,
    )

    expected_key = format_shape_key_3d(
        2, 65, 4, 9, 11, 67, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1
    )
    tile_count = 3 * 3
    expected_input = [
        {
            "shape_key": expected_key,
            "M": 2 * 4 * tile_count,
            "variants": (input_variant,),
        }
    ]
    expected_stage = [{"shape_key": expected_key, "M": 2 * 4 * tile_count}]
    assert (
        input_seen == expected_input
    ), f"Winograd input config lookup {input_seen} != expected {expected_input}"
    assert (
        gemm_seen == expected_stage
    ), f"Winograd GEMM config lookup {gemm_seen} != expected {expected_stage}"
    assert (
        output_seen == expected_stage
    ), f"Winograd output config lookup {output_seen} != expected {expected_stage}"
    launch_counts = [len(kernel.calls) for kernel in kernels]
    assert all(
        count == 1 for count in launch_counts
    ), f"expected one launch per Winograd stage, got {launch_counts}"


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
