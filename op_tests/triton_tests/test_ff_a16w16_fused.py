import torch
import pytest
from aiter.ops.triton.ff_a16w16_fused_gated import ff_a16w16_fused_gated
from aiter.ops.triton.ff_a16w16_fused_ungated import ff_a16w16_fused_ungated
from op_tests.triton_tests.ff_test_utils import ff_gated_test, ff_ungated_test
from aiter.ops.triton.utils.types import str_to_torch_dtype


def basic_shape_set():
    shapes = [(1, 1, 1)]  # minimal case
    shapes += [(3, 5, 2)]  # irregular shape
    shapes += [
        (32, 32, 32),
        (128, 128, 128),
        (512, 512, 512),
        (1024, 1024, 1024),
        (4864, 4096, 8192),
    ]
    shapes = [(2**i, 256, 7168) for i in range(1, 4)]
    return shapes


def extended_shape_set():
    shapes = [(2**i, 256, 7168) for i in range(5, 9)]
    shapes += [(1024 * v, 1024 * v, 1024 * v) for v in range(2, 9)]
    shapes = [(9728, 8192, 65536), (4864, 8192, 4160)]
    shapes += [
        (1, 1280, 8192),
        (32, 1280, 8192),
        (64, 1280, 8192),
        (128, 1280, 8192),
        (192, 1280, 8192),
        (256, 1280, 8192),
        (320, 1280, 8192),
        (512, 1280, 8192),
        (1024, 1280, 8192),
        (2048, 1280, 8192),
        (4096, 1280, 8192),
        (8192, 1280, 8192),
        (16384, 1280, 8192),
        (1, 8192, 1024),
        (32, 8192, 1024),
        (64, 8192, 1024),
        (128, 8192, 1024),
        (192, 8192, 1024),
        (256, 8192, 1024),
        (320, 8192, 1024),
        (512, 8192, 1024),
        (1024, 8192, 1024),
        (2048, 8192, 1024),
        (4096, 8192, 1024),
        (8192, 8192, 1024),
        (16384, 8192, 1024),
    ]
    return shapes


@pytest.mark.parametrize(
    "batch, hidden_dim, intermediate_dim, dtype, output, activation",
    [
        (*shape, dtype_str, output, activation)
        for shape in basic_shape_set()
        for dtype_str in ["bf16"]
        for output in [True]
        for activation in ["silu_exp2", "gelu_tanh", "relu", None]
    ]
    + [
        pytest.param(*shape, dtype_str, output, activation, marks=pytest.mark.extended)
        for shape in extended_shape_set()
        for dtype_str in ["bf16", "fp16"]
        for output in [True, False]  # TODO: Debug False fails
        for activation in ["silu_exp2", "gelu_tanh", "relu", None]
    ],
)
def test_ff_a16w16_fused_ungated(
    batch: int, hidden_dim: int, intermediate_dim: int, dtype, output, activation
):
    if (batch * intermediate_dim * hidden_dim) > 5000 * 5000 * 5000:
        pytest.skip(
            "Small differences in implementation between Triton & Torch activations accumulate to beyond test bounds w/large matrices."
        )
    dtype = str_to_torch_dtype[dtype]
    ff_ungated_test(
        ff_a16w16_fused_ungated,
        batch=batch,
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
        dtype=dtype,
        output=output,
        activation=activation,
        y_init="zeros",
    )


@pytest.mark.parametrize(
    "batch, hidden_dim, intermediate_dim, dtype, output, activation",
    [
        (*shape, dtype_str, output, activation)
        for shape in basic_shape_set()
        for dtype_str in ["bf16"]
        for output in [True]
        for activation in ["silu_exp2", "gelu_tanh", "relu", None]
    ]
    + [
        pytest.param(*shape, dtype_str, output, activation, marks=pytest.mark.extended)
        for shape in extended_shape_set()
        for dtype_str in ["bf16", "fp16"]
        for output in [True, False]  # TODO: Debug False fails
        for activation in ["silu_exp2", "gelu_tanh", "relu", None]
    ],
)
def test_ff_a16w16_fused_gated(
    batch: int, hidden_dim: int, intermediate_dim: int, dtype, output, activation
):
    if (batch * intermediate_dim * hidden_dim) > 5000 * 5000 * 5000:
        pytest.skip(
            "Small differences in implementation between Triton & Torch activations accumulate to beyond test bounds w/large matrices."
        )

    dtype = str_to_torch_dtype[dtype]
    ff_gated_test(
        ff_a16w16_fused_gated,
        batch=batch,
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
        dtype=dtype,
        output=output,
        activation=activation,
        y_init="zeros",
    )
