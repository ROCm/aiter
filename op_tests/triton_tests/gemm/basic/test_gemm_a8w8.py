# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import functools
import importlib
import os
import sys

import pytest
import torch
import torch.nn.functional as F

from aiter.ops.shuffle import shuffle_weight
from aiter.ops.triton.gemm.basic.gemm_a8w8 import _is_gluon_available
from aiter.ops.triton.gemm.basic.gemm_a8w8 import gemm_a8w8 as triton_gemm_a8w8
from aiter.ops.triton.gemm.basic.gemm_a8w8 import gemm_a8w8_preshuffle
from aiter.ops.triton.utils._triton import arch_info
from aiter.ops.triton.utils.config_utils import resolve_config_dir
from aiter.ops.triton.utils.gemm_config_utils import (
    compute_splitk_params,
    get_gemm_config,
)
from aiter.ops.triton.utils.types import get_fp8_dtypes, str_to_torch_dtype

DEVICE_ARCH = arch_info.get_arch()
IS_GFX1250 = "gfx1250" in (DEVICE_ARCH or "")


def is_gluon_supported():
    """gluon a8w8 kernels only exist on some archs (gfx950, gfx1250)."""
    return _is_gluon_available()


def _skip_if_triton_has_no_config(backend):
    """Some archs ship only gluon-format configs for a family, leaving the
    triton backend with nothing usable (see test_gemm_a16w16's
    _skip_if_triton_on_gfx1250).  a8w8 still ships tuned triton configs on
    gfx1250, so this is a no-op there today; the probe keeps the test honest
    if that ever changes."""
    if backend != "triton":
        return
    cfg_dir = resolve_config_dir("gemm", "GEMM-A8W8", backend="triton")
    if not os.path.exists(f"{cfg_dir}/DEFAULT.json"):
        pytest.skip(f"triton backend has no {DEVICE_ARCH} a8w8 config")


def _skip_if_backend_unsupported(backend, in_dtype, k, layout="TN"):
    """Arch/shape-gated backend combinations."""
    _skip_if_triton_has_no_config(backend)
    if backend != "gluon":
        return
    if not is_gluon_supported():
        pytest.skip("Gluon backend is not supported on this architecture")
    if IS_GFX1250:
        # The gfx1250 gluon a8w8 kernel is an fp8 TDM/WMMA kernel. int8
        # inputs, non-K-contiguous operands and K <= 128 (fewer than two
        # 128-wide WMMA K tiles) have no gluon path there -- gemm_a8w8's
        # auto-dispatch silently falls back to triton for those, and an
        # explicit backend="gluon" raises.
        if in_dtype == torch.int8:
            pytest.skip("gfx1250 gluon a8w8 is fp8-only")
        if k <= 128:
            pytest.skip("gfx1250 gluon a8w8 needs K > 128 (two WMMA K tiles)")
        if layout != "TN":
            pytest.skip("gfx1250 gluon a8w8 needs K-contiguous x and w")


def _skip_if_preshuffle_unsupported(n, k):
    if DEVICE_ARCH != "gfx950":
        pytest.skip("Preshuffled gluon a8w8 requires gfx950.")
    if n % 16 != 0 or k % 32 != 0:
        pytest.skip(
            "For preshuffle, N must be multiple of 16 and K must be multiple of 32."
        )


def run_torch(x, weight, x_scale, w_scale, bias=None, dtype=torch.bfloat16):
    x = F.linear(x.to(torch.float32), weight.to(torch.float32))
    scale = torch.matmul(x_scale, w_scale)
    out = torch.mul(x, scale)
    if bias is not None:
        out = out.to(bias) + bias
    return out.to(dtype)


def run_triton(
    x, weight, x_scale, w_scale, bias=None, dtype=torch.bfloat16, y=None, impl=None
):
    return impl(x, weight, x_scale, w_scale, bias, dtype, y)


e5m2_type, e4m3_type = get_fp8_dtypes()


dtype_max = {
    dtype: (torch.finfo(dtype) if dtype.is_floating_point else torch.iinfo(dtype)).max
    for dtype in [
        e5m2_type,
        e4m3_type,
        torch.int8,
    ]
}


def get_x_vals():
    x_vals = [(1, 1, 1)]  # minimal case
    x_vals += [(3, 5, 2)]  # irregular shape
    x_vals += [(1024 * v, 1024 * v, 1024 * v) for v in (1, 2, 4, 5, 8)]
    x_vals += [(v, 106496, 16384) for v in (190, 256, 4096)]  # LL3 405B FC1
    x_vals += [
        (v, 10240, 8192) for v in (256, 4096, 8000)
    ]  # LL3 70B QKV input projection
    return x_vals


def get_splitk_x_vals():
    return [
        (1, 1280, 8192),
        (32, 1280, 8192),
        (64, 1280, 8192),
        (128, 1280, 8192),
        (256, 1280, 8192),
        (1, 8192, 1024),
        (32, 8192, 1024),
        (64, 8192, 1024),
        (128, 8192, 1024),
        (256, 8192, 1024),
        (1024, 1024, 1000),
        (1024, 1024, 1024),
        (1024, 1024, 4096),
        (1024, 4096, 4096),
        (2048, 2048, 2048),
        (4096, 4096, 4096),
    ]


def generate_gemm_a8w8_inputs(
    M: int,
    N: int,
    K: int,
    in_dtype: torch.dtype | str,
    out_dtype: torch.dtype | str,
    layout: str = "TN",
    output: bool = False,
    shuffle: bool = False,
):
    """
    The GEMM kernel expects:
    - x: (M, K) -> row-major format
    - w: (N, K) -> column-major format
    """
    torch.manual_seed(0)
    if layout[0] == "T":
        # T (transposed) in Fortran notation equals row-major
        x = torch.randn((M, K), dtype=torch.float32, device="cuda")
    else:
        x = torch.randn((K, M), dtype=torch.float32, device="cuda").T

    if layout[1] == "N":
        weight = torch.randn((N, K), dtype=torch.float32, device="cuda")
    else:
        weight = torch.randn((K, N), dtype=torch.float32, device="cuda").T

    max_x = x.abs().float().amax(dim=1, keepdim=True)
    x_scale = max_x / dtype_max[in_dtype]
    x = x / x_scale
    x = x.to(in_dtype)

    max_weight = weight.abs().float().amax(dim=1, keepdim=True).T.contiguous()
    w_scale = max_weight / dtype_max[in_dtype]
    weight = weight / w_scale.T
    weight = weight.to(in_dtype)

    bias = torch.rand([1, N], dtype=torch.float32, device="cuda") * 10

    if shuffle:
        weight_shuffle_layout = (16, 16)
        weight_shuffled = shuffle_weight(weight, weight_shuffle_layout).reshape(
            weight.shape[0] // weight_shuffle_layout[0],
            weight.shape[1] * weight_shuffle_layout[0],
        )
    else:
        weight_shuffled = weight

    y = None
    if output:
        y = torch.empty((M, N), dtype=out_dtype, device="cuda")

    return x, weight, weight_shuffled, x_scale, w_scale, bias, y


def get_fewer_x_vals():
    x_vals = [(16, 1024, 1024)]
    x_vals += [(128, 8192, 512)]
    x_vals += [(256, 512, 8192)]
    x_vals += [(1024 * v, 1024 * v, 1024 * v) for v in (1, 5, 8)]
    return x_vals


@pytest.mark.parametrize(
    "in_dtype, m, n, k",
    [
        (in_dtype, *shape)
        for in_dtype in ["fp8e4m3", "fp8e5m2"]
        for shape in get_x_vals()
    ],
)
@pytest.mark.parametrize("backend", ["triton", "gluon"])
@pytest.mark.parametrize("preshuffle", [False, True])
def test_gemm_fp8(in_dtype, m, n, k, backend: str, preshuffle: bool):

    torch.cuda.empty_cache()

    if preshuffle:
        # gemm_a8w8_preshuffle is a gluon-only entry point.
        if backend != "gluon":
            pytest.skip("preshuffled weights are a gluon-only path")
        _skip_if_preshuffle_unsupported(n, k)
    else:
        _skip_if_backend_unsupported(backend, str_to_torch_dtype[in_dtype], k)

    in_dtype = str_to_torch_dtype[in_dtype]
    out_dtype = str_to_torch_dtype["bf16"]
    x, weight, weight_triton, x_scale, w_scale, bias, y = generate_gemm_a8w8_inputs(
        M=m,
        N=n,
        K=k,
        in_dtype=in_dtype,
        out_dtype=out_dtype,
        layout="TN",
        output=False,
        shuffle=preshuffle,
    )

    a = run_torch(x, weight, x_scale, w_scale, bias, out_dtype)
    if preshuffle:
        impl = gemm_a8w8_preshuffle
    else:
        impl = functools.partial(triton_gemm_a8w8, backend=backend)
    b = run_triton(x, weight_triton, x_scale, w_scale, bias, out_dtype, y, impl)

    torch.testing.assert_close(a, b, atol=0.02, rtol=1e-2)


@pytest.mark.parametrize(
    "out_dtype, m, n, k, layout, output",
    [
        (out_dtype, *shape, layout, output)
        for out_dtype in ["fp16", "fp32", "int32"]
        for shape in get_fewer_x_vals()
        for layout in ["TN", "TT", "NN", "NT"]
        for output in [True, False]
    ],
)
@pytest.mark.parametrize("backend", ["triton", "gluon"])
@pytest.mark.parametrize("preshuffle", [False, True])
def test_gemm_int8(out_dtype, m, n, k, layout, output, backend: str, preshuffle: bool):

    torch.cuda.empty_cache()

    if preshuffle:
        if backend != "gluon":
            pytest.skip("preshuffled weights are a gluon-only path")
        _skip_if_preshuffle_unsupported(n, k)
    else:
        _skip_if_backend_unsupported(backend, torch.int8, k, layout=layout)

    in_dtype = str_to_torch_dtype["int8"]
    out_dtype = str_to_torch_dtype[out_dtype]
    x, weight, weight_triton, x_scale, w_scale, bias, y = generate_gemm_a8w8_inputs(
        M=m,
        N=n,
        K=k,
        in_dtype=in_dtype,
        out_dtype=out_dtype,
        layout=layout,
        output=output,
        shuffle=preshuffle,
    )

    a = run_torch(x, weight, x_scale, w_scale, bias, out_dtype)
    if preshuffle:
        impl = gemm_a8w8_preshuffle
    else:
        impl = functools.partial(triton_gemm_a8w8, backend=backend)
    b = run_triton(x, weight_triton, x_scale, w_scale, bias, out_dtype, y, impl)

    if out_dtype in [torch.int8, torch.int32]:
        torch.testing.assert_close(a, b, atol=1, rtol=1e-2)
    else:
        torch.testing.assert_close(a, b, atol=0.03, rtol=1e-2)


@pytest.mark.parametrize(
    "in_dtype, out_dtype, m, n, k",
    [
        (in_dtype, out_dtype, *shape)
        for in_dtype in ["fp8e4m3", "fp8e5m2", "int8"]
        for out_dtype in ["bf16", "fp32"]
        for shape in get_splitk_x_vals()
    ],
)
@pytest.mark.parametrize("num_ksplit", [2, 4, 8])
@pytest.mark.parametrize("has_bias", [True, False])
def test_gemm_splitk(in_dtype, out_dtype, m, n, k, num_ksplit, has_bias):

    torch.cuda.empty_cache()

    if out_dtype == "int32" and in_dtype in ["fp8e4m3", "fp8e5m2"]:
        pytest.skip(
            "This kernel is not supported for in_dtype of float and out_dtype of int."
        )

    in_dtype = str_to_torch_dtype[in_dtype]
    out_dtype = str_to_torch_dtype[out_dtype]
    x, weight, weight_triton, x_scale, w_scale, bias, _ = generate_gemm_a8w8_inputs(
        M=m,
        N=n,
        K=k,
        in_dtype=in_dtype,
        out_dtype=out_dtype,
        layout="TN",
        output=False,
    )

    if not has_bias:
        bias = None

    # split-K is a triton-backend feature; load the triton-format config.
    config, _ = get_gemm_config("GEMM-A8W8", m, n, k, backend="triton")
    config["NUM_KSPLIT"] = num_ksplit
    compute_splitk_params(config, k)

    a = run_torch(x, weight, x_scale, w_scale, bias, out_dtype)
    b = triton_gemm_a8w8(
        x,
        weight_triton,
        x_scale,
        w_scale,
        bias,
        out_dtype,
        config=config,
        backend="triton",
    )

    if out_dtype in [torch.int8, torch.int32]:
        torch.testing.assert_close(a, b, atol=1, rtol=1e-2)
    else:
        torch.testing.assert_close(a, b, atol=0.02, rtol=1e-2)


@pytest.mark.parametrize(
    "in_dtype, out_dtype, m, n, k",
    [
        (in_dtype, out_dtype, *shape)
        for in_dtype in ["fp8e4m3"]
        for out_dtype in ["bf16"]
        for shape in [
            (64, 1280, 8192),
            (128, 1280, 8192),
            (1024, 4096, 4096),
        ]
    ],
)
@pytest.mark.parametrize("num_ksplit", [2, 4])
def test_gemm_splitk_skip_reduce(in_dtype, out_dtype, m, n, k, num_ksplit):

    torch.cuda.empty_cache()

    in_dtype = str_to_torch_dtype[in_dtype]
    out_dtype = str_to_torch_dtype[out_dtype]
    x, weight, weight_triton, x_scale, w_scale, _bias, _ = generate_gemm_a8w8_inputs(
        M=m,
        N=n,
        K=k,
        in_dtype=in_dtype,
        out_dtype=out_dtype,
        layout="TN",
        output=False,
    )

    # split-K is a triton-backend feature; load the triton-format config.
    config, _ = get_gemm_config("GEMM-A8W8", m, n, k, backend="triton")
    config["NUM_KSPLIT"] = num_ksplit
    compute_splitk_params(config, k)

    a = run_torch(x, weight, x_scale, w_scale, None, out_dtype)

    y_pp = triton_gemm_a8w8(
        x,
        weight_triton,
        x_scale,
        w_scale,
        None,
        out_dtype,
        config=config,
        skip_reduce=True,
        backend="triton",
    )

    assert y_pp.dim() == 3, f"Expected 3D tensor, got {y_pp.dim()}D"
    assert (
        y_pp.shape[1] == m and y_pp.shape[2] == n
    ), f"Expected shape (*, {m}, {n}), got {y_pp.shape}"

    b = y_pp.sum(dim=0).to(out_dtype)
    torch.testing.assert_close(a, b, atol=0.02, rtol=1e-2)


def test_legacy_gluon_import_path_warns():
    """The pre-move path still resolves here, but tells callers to move on."""
    legacy = "aiter.ops.triton.gluon.gemm_a8w8"
    sys.modules.pop(legacy, None)

    with pytest.warns(DeprecationWarning, match="has moved to"):
        mod = importlib.import_module(legacy)

    assert mod.gemm_a8w8.__module__ == "aiter.ops.triton.gemm.basic.gemm_a8w8"
