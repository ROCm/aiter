# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import functools
import importlib
import sys

import pytest
import torch
import torch.nn.functional as F

from aiter.ops.shuffle import shuffle_weight
from aiter.ops.triton.gemm.basic.gemm_a8w8 import gemm_a8w8 as triton_gemm_a8w8
from aiter.ops.triton.gemm.basic.gemm_a8w8 import (
    gemm_a8w8_negate_w_k_tail,
    gemm_a8w8_preshuffle,
)
from aiter.ops.triton.utils._triton import arch_info
from aiter.ops.triton.utils.gemm_config_utils import (
    compute_splitk_params,
    get_gemm_config,
)
from aiter.ops.triton.utils.types import get_fp8_dtypes, str_to_torch_dtype
from op_tests.triton_tests.gemm.gemm_shapes import get_shapes

DEVICE_ARCH = arch_info.get_arch()


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


@pytest.mark.parametrize(
    "in_dtype, m, n, k",
    [
        (in_dtype, *shape)
        for in_dtype in ["fp8e4m3", "fp8e5m2"]
        for shape in get_shapes()
    ],
)
@pytest.mark.parametrize(
    "impl",
    [
        "triton",
        "gluon",
        "gluon_shuffle",
        "triton_shuffle",
    ],
)
def test_gemm_fp8(in_dtype, m, n, k, impl: str):

    torch.cuda.empty_cache()

    if impl in ["gluon", "gluon_shuffle"] and DEVICE_ARCH != "gfx950":
        pytest.skip(
            "Gluon implementation is not supported on this device (requires gfx950)."
        )

    if "_shuffle" in impl and (n % 16 != 0 or k % 32 != 0):
        pytest.skip(
            "For preshuffle, N must be multiple of 16 and K must be multiple of 32."
        )
    if impl == "gluon_shuffle" and k % 256 != 0:
        pytest.skip("Gluon preshuffle needs K to be a multiple of BLOCK_SIZE_K (256).")

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
        shuffle=("_shuffle" in impl),
    )

    a = run_torch(x, weight, x_scale, w_scale, bias, out_dtype)
    if impl == "triton":
        impl = triton_gemm_a8w8
    elif impl == "gluon":
        impl = functools.partial(triton_gemm_a8w8, backend="gluon")
    elif impl == "triton_shuffle":
        impl = gemm_a8w8_preshuffle
    elif impl == "gluon_shuffle":
        impl = functools.partial(gemm_a8w8_preshuffle, backend="gluon")
    else:
        raise ValueError(f"Unknown implementation: {impl}")
    b = run_triton(x, weight_triton, x_scale, w_scale, bias, out_dtype, y, impl)

    torch.testing.assert_close(a, b, atol=0.02, rtol=1e-2)


def get_ptpc_x_vals():
    x_vals = [(1, 16, 128), (3, 5, 2)]  # one tile, irregular shape
    x_vals += [(8, 80, 1000)]  # partial M, N and K tiles
    x_vals += [(m, 12448, 7168) for m in (8, 128, 1024)]  # in_proj, TP4
    x_vals += [(m, 6288, 7168) for m in (16, 600)]  # in_proj, TP8
    x_vals += [(4096, 1024, 2048)]
    return x_vals


@pytest.mark.parametrize("m, n, k", get_ptpc_x_vals())
@pytest.mark.parametrize("output", [True, False])
@pytest.mark.parametrize("has_bias", [True, False])
def test_gemm_fp8_ptpc_gluon_gfx1250(m, n, k, output, has_bias):
    """Per-token and per-channel scales through the gfx1250 Gluon kernel."""
    if DEVICE_ARCH not in ("gfx1250", "gfx1250-strict"):
        pytest.skip("The Gluon PTPC kernel requires gfx1250.")

    torch.cuda.empty_cache()
    out_dtype = str_to_torch_dtype["bf16"]
    x, weight, _, x_scale, w_scale, bias, y = generate_gemm_a8w8_inputs(
        M=m,
        N=n,
        K=k,
        in_dtype=e4m3_type,
        out_dtype=out_dtype,
        layout="TN",
        output=output,
    )
    if not has_bias:
        bias = None

    a = run_torch(x, weight, x_scale, w_scale, bias, out_dtype)
    b = triton_gemm_a8w8(
        x, weight, x_scale, w_scale, bias, out_dtype, y, backend="gluon"
    )
    if output:
        assert b is y

    torch.testing.assert_close(a, b, atol=0.02, rtol=1e-2)


@pytest.mark.parametrize("m, n, k", get_ptpc_x_vals())
@pytest.mark.parametrize("has_bias", [True, False])
def test_gemm_fp8_ptpc_gluon_gfx1250_w_negated(m, n, k, has_bias):
    """The gfx1250 Gluon kernel with a weight whose K tail is negated one time."""
    if DEVICE_ARCH not in ("gfx1250", "gfx1250-strict"):
        pytest.skip("The Gluon PTPC kernel requires gfx1250.")

    torch.cuda.empty_cache()
    out_dtype = str_to_torch_dtype["bf16"]
    x, weight, _, x_scale, w_scale, bias, y = generate_gemm_a8w8_inputs(
        M=m,
        N=n,
        K=k,
        in_dtype=e4m3_type,
        out_dtype=out_dtype,
        layout="TN",
        output=True,
    )
    if not has_bias:
        bias = None

    a = run_torch(x, weight, x_scale, w_scale, bias, out_dtype)
    b = triton_gemm_a8w8(
        x,
        gemm_a8w8_negate_w_k_tail(weight),
        x_scale,
        w_scale,
        bias,
        out_dtype,
        y,
        backend="gluon",
        w_k_tail_negated=True,
    )

    torch.testing.assert_close(a, b, atol=0.02, rtol=1e-2)


@pytest.mark.parametrize(
    "out_dtype, m, n, k, layout, output",
    [
        (out_dtype, *shape, layout, output)
        for out_dtype in ["fp16", "fp32", "int32"]
        for shape in get_shapes(max_size=8192 * 8192 * 8192)
        for layout in ["TN", "TT", "NN", "NT"]
        for output in [True, False]
    ],
)
@pytest.mark.parametrize(
    "impl",
    [
        "triton",
        "gluon",
        "gluon_shuffle",
        "triton_shuffle",
    ],
)
def test_gemm_int8(out_dtype, m, n, k, layout, output, impl: str):

    torch.cuda.empty_cache()

    in_dtype = "int8"

    if impl in ["gluon", "gluon_shuffle"] and DEVICE_ARCH != "gfx950":
        pytest.skip(
            "Gluon implementation is not supported on this device (requires gfx950)."
        )

    if "_shuffle" in impl and (n % 16 != 0 or k % 32 != 0):
        pytest.skip(
            "For preshuffle, N must be multiple of 16 and K must be multiple of 32."
        )
    if impl == "gluon_shuffle" and k % 256 != 0:
        pytest.skip("Gluon preshuffle needs K to be a multiple of BLOCK_SIZE_K (256).")

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
        shuffle=("_shuffle" in impl),
    )

    a = run_torch(x, weight, x_scale, w_scale, bias, out_dtype)
    if impl == "triton":
        impl = triton_gemm_a8w8
    elif impl == "gluon":
        impl = functools.partial(triton_gemm_a8w8, backend="gluon")
    elif impl == "triton_shuffle":
        impl = gemm_a8w8_preshuffle
    elif impl == "gluon_shuffle":
        impl = functools.partial(gemm_a8w8_preshuffle, backend="gluon")
    else:
        raise ValueError(f"Unknown implementation: {impl}")
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
        for shape in get_shapes(max_size=4096 * 4096 * 4096)
    ],
)
@pytest.mark.parametrize("num_ksplit", [2, 4, 8])
@pytest.mark.parametrize("has_bias", [True, False])
@pytest.mark.parametrize("preshuffle", [False, True])
def test_gemm_splitk(in_dtype, out_dtype, m, n, k, num_ksplit, has_bias, preshuffle):

    torch.cuda.empty_cache()

    if out_dtype == "int32" and in_dtype in ["fp8e4m3", "fp8e5m2"]:
        pytest.skip(
            "This kernel is not supported for in_dtype of float and out_dtype of int."
        )

    if preshuffle and (n % 16 != 0 or k % 32 != 0):
        pytest.skip(
            "For preshuffle, N must be multiple of 16 and K must be multiple of 32."
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
        shuffle=preshuffle,
    )

    if not has_bias:
        bias = None

    if preshuffle:
        config, _ = get_gemm_config("GEMM-A8W8_PRESHUFFLED", m, n, k)
        fn = gemm_a8w8_preshuffle
    else:
        config, _ = get_gemm_config("GEMM-A8W8", m, n, k)
        fn = triton_gemm_a8w8
    config["NUM_KSPLIT"] = num_ksplit
    compute_splitk_params(config, k)

    a = run_torch(x, weight, x_scale, w_scale, bias, out_dtype)
    b = fn(
        x,
        weight_triton,
        x_scale,
        w_scale,
        bias,
        out_dtype,
        config=config,
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
@pytest.mark.parametrize("preshuffle", [False, True])
def test_gemm_splitk_skip_reduce(in_dtype, out_dtype, m, n, k, num_ksplit, preshuffle):

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
        shuffle=preshuffle,
    )

    if preshuffle:
        config, _ = get_gemm_config("GEMM-A8W8_PRESHUFFLED", m, n, k)
        fn = gemm_a8w8_preshuffle
    else:
        config, _ = get_gemm_config("GEMM-A8W8", m, n, k)
        fn = triton_gemm_a8w8
    config["NUM_KSPLIT"] = num_ksplit
    compute_splitk_params(config, k)

    a = run_torch(x, weight, x_scale, w_scale, None, out_dtype)

    y_pp = fn(
        x,
        weight_triton,
        x_scale,
        w_scale,
        None,
        out_dtype,
        config=config,
        skip_reduce=True,
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
