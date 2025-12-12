import torch
import pytest
from aiter.ops.triton.ff_a16w16 import ff_a16w16_gated, ff_a16w16_nogate
from op_tests.triton_tests.gemm.feed_forward.ff_test_utils import (
    ff_gated_test,
    ff_ungated_test,
)
from op_tests.triton_tests.gemm.basic.test_gemm_a16w16 import get_x_vals


@pytest.mark.parametrize("activation", ["gelu_tanh", "silu_exp2", "relu", None])
@pytest.mark.parametrize("batch, hidden_dim, intermediate_dim", get_x_vals())
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("output", [True, False])
def test_ff_a16w16_ungated(
    batch: int, hidden_dim: int, intermediate_dim: int, dtype, output, activation
):
    ff_ungated_test(
        ff_a16w16_nogate,
        batch=batch,
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
        dtype=dtype,
        output=output,
        activation=activation,
        y_init="empty",
    )


@pytest.mark.parametrize("activation", ["gelu_tanh", "silu_exp2", "relu", None])
@pytest.mark.parametrize("batch, hidden_dim, intermediate_dim", get_x_vals())
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("output", [True, False])
def test_ff_a16w16_gated(
    batch: int, hidden_dim: int, intermediate_dim: int, dtype, output, activation
):
    ff_gated_test(
        ff_a16w16_gated,
        batch=batch,
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
        dtype=dtype,
        output=output,
        activation=activation,
        y_init="empty",
    )
