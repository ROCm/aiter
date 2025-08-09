import torch
import pytest
from aiter.ops.triton.ff_a16w16_fused_gated import ff_a16w16_fused_gated
from aiter.ops.triton.ff_a16w16_fused_ungated import ff_a16w16_fused_ungated
from op_tests.triton_tests.utils import minimal_x_vals
from op_tests.triton_tests.test_gemm_a16w16 import get_x_vals
from op_tests.triton_tests.ff_test_utils import ff_gated_test, ff_ungated_test


@pytest.mark.parametrize("activation", ["silu_exp2", "gelu_tanh", "relu", None])
@pytest.mark.parametrize(
    "batch, hidden_dim, intermediate_dim", minimal_x_vals(get_x_vals(), sample=5)
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("output", [True, False])
def test_ff_a16w16_ungated(
    batch: int, hidden_dim: int, intermediate_dim: int, dtype, output, activation
):
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
@pytest.mark.parametrize(
    "batch, hidden_dim, intermediate_dim", minimal_x_vals(get_x_vals(), sample=5)
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("output", [True, False])
def test_ff_a16w16_gated(
    batch: int, hidden_dim: int, intermediate_dim: int, dtype, output, activation
):
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
