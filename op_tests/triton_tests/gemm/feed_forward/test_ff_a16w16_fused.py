import torch
import pytest
from aiter.ops.triton.ff_a16w16_fused_gated import ff_a16w16_fused_gated
from aiter.ops.triton.ff_a16w16_fused_ungated import ff_a16w16_fused_ungated
from op_tests.triton_tests.gemm.basic.test_gemm_a16w16 import get_x_vals
from op_tests.triton_tests.gemm.feed_forward.ff_test_utils import (
    ff_gated_test,
    ff_ungated_test,
)


@pytest.mark.parametrize("activation", ["silu_exp2", "gelu_tanh", "relu", None])
@pytest.mark.parametrize("batch, hidden_dim, intermediate_dim", get_x_vals())
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("output", [True, False])
def test_ff_a16w16_fused_ungated(
    batch: int, hidden_dim: int, intermediate_dim: int, dtype, output, activation
):
    if (batch * intermediate_dim * hidden_dim) > 5000 * 5000 * 5000:
        pytest.skip(
            "Small differences in implementation between Triton & Torch activations accumulate to beyond test bounds w/large matrices."
        )
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


@pytest.mark.parametrize("activation", ["silu_exp2", "gelu_tanh", "relu", None])
@pytest.mark.parametrize("batch, hidden_dim, intermediate_dim", get_x_vals())
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("output", [True, False])
def test_ff_a16w16_fused_gated(
    batch: int, hidden_dim: int, intermediate_dim: int, dtype, output, activation
):
    if (batch * intermediate_dim * hidden_dim) > 5000 * 5000 * 5000:
        pytest.skip(
            "Small differences in implementation between Triton & Torch activations accumulate to beyond test bounds w/large matrices."
        )

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
