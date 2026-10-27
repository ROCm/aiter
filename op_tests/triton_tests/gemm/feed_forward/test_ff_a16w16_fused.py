import pytest
import torch

from aiter.ops.triton.gemm.feed_forward.ff_a16w16_fused_gated import (
    ff_a16w16_fused_gated,
)
from aiter.ops.triton.gemm.feed_forward.ff_a16w16_fused_ungated import (
    ff_a16w16_fused_ungated,
)
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
    torch.manual_seed(0)
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
    torch.manual_seed(0)

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


# gfx950 M<=8 launch regression (PR #5399). get_x_vals() jumps from M=3 to M>=32,
# so M=5-8 -- the only range that selects the DEFAULT M_LEQ_8 tile -- is otherwise
# uncompiled by CI. A DEFAULT M_LEQ_8 that exceeds LDS (the bug this PR fixes)
# raises OutOfResources at launch; these cases take the config=None default path
# at M=8 so that regression is caught. One shape resolves to a specialized gated
# config, one is untuned and falls back to DEFAULT; both gated and ungated run.
_M8_SHAPES = [
    (8, 8192, 8192),  # gated -> specialized N=16384,K=8192 ; ungated -> DEFAULT
    (8, 2048, 3072),  # untuned for both -> DEFAULT M_LEQ_8
]


def _skip_unless_gfx950():
    from aiter.ops.triton.utils._triton.arch_info import get_arch

    if "gfx950" not in (get_arch() or ""):
        pytest.skip("FF-A16W16-fused configs are tuned for gfx950")


@pytest.mark.parametrize("batch, hidden_dim, intermediate_dim", _M8_SHAPES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_ff_a16w16_fused_gated_m8_launch(batch, hidden_dim, intermediate_dim, dtype):
    _skip_unless_gfx950()
    torch.manual_seed(0)
    ff_gated_test(
        ff_a16w16_fused_gated,
        batch=batch,
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
        dtype=dtype,
        output=True,
        activation="silu_exp2",
        y_init="zeros",
    )


@pytest.mark.parametrize("batch, hidden_dim, intermediate_dim", _M8_SHAPES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_ff_a16w16_fused_ungated_m8_launch(batch, hidden_dim, intermediate_dim, dtype):
    _skip_unless_gfx950()
    torch.manual_seed(0)
    ff_ungated_test(
        ff_a16w16_fused_ungated,
        batch=batch,
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
        dtype=dtype,
        output=True,
        activation="silu_exp2",
        y_init="zeros",
    )
