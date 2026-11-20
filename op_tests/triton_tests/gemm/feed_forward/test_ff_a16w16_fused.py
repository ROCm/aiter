import pytest
import torch

from aiter.ops.triton.gemm.feed_forward.ff_a16w16_fused_gated import (
    ff_a16w16_fused_gated,
)
from aiter.ops.triton.gemm.feed_forward.ff_a16w16_fused_ungated import (
    ff_a16w16_fused_ungated,
)
from op_tests.triton_tests.gemm.feed_forward.ff_test_utils import (
    ff_gated_test,
    ff_ungated_test,
)
from op_tests.triton_tests.gemm.gemm_shapes import get_shapes


@pytest.mark.parametrize("activation", ["silu_exp2", "gelu_tanh", "relu", None])
@pytest.mark.parametrize("batch, hidden_dim, intermediate_dim", get_shapes())
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


# gfx950's tuned gated configs for N=16384 (intermediate_dim=8192) at K=1280 and
# K=8192 use a 32x128x256 tile with num_stages=2 for M <= 64. That needs 168864 B
# of LDS (the limit is 163840), so these cases cannot launch.
_GATED_LDS_REASON = "gfx950 tuned gated config for N=16384, K=1280/8192 exceeds LDS"


def _gated_config_exceeds_lds(batch, hidden_dim, intermediate_dim):
    from aiter.ops.triton.utils._triton.arch_info import get_arch

    return (
        get_arch() == "gfx950"
        and batch <= 64
        and intermediate_dim == 8192
        and hidden_dim in (1280, 8192)
    )


@pytest.mark.parametrize("activation", ["silu_exp2", "gelu_tanh", "relu", None])
@pytest.mark.parametrize("batch, hidden_dim, intermediate_dim", get_shapes())
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("output", [True, False])
def test_ff_a16w16_fused_gated(
    batch: int, hidden_dim: int, intermediate_dim: int, dtype, output, activation
):
    if (batch * intermediate_dim * hidden_dim) > 5000 * 5000 * 5000:
        pytest.skip(
            "Small differences in implementation between Triton & Torch activations accumulate to beyond test bounds w/large matrices."
        )
    if _gated_config_exceeds_lds(batch, hidden_dim, intermediate_dim):
        pytest.skip(_GATED_LDS_REASON)
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


# gfx950 M<=8 launch regression (PR #5399). get_shapes() has only M=7 between M=3
# and M=16, so M=5-8 -- the only range that selects the DEFAULT M_LEQ_8 tile -- is
# barely compiled by CI otherwise. A DEFAULT M_LEQ_8 that exceeds LDS (the bug this PR fixes)
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
    if _gated_config_exceeds_lds(batch, hidden_dim, intermediate_dim):
        pytest.skip(_GATED_LDS_REASON)
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
