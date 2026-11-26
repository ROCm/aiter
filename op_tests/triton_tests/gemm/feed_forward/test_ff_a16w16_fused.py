import pytest
import torch
import torch.nn.functional as F
import triton

from aiter.ops.triton.gemm.feed_forward.ff_a16w16_fused_gated import (
    ff_a16w16_fused_gated,
)
from aiter.ops.triton.gemm.feed_forward.ff_a16w16_fused_ungated import (
    ff_a16w16_fused_ungated,
)
from op_tests.triton_tests.gemm.feed_forward.ff_test_utils import (
    ff_gated_test,
    ff_ungated_test,
    generate_ff_inputs,
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


# Retuned gfx950 buckets (M_LEQ_4, M_LEQ_8, M_LEQ_64) that get_x_vals() does not reach.
_RETUNED_GATED_SHAPES = [
    (4, 8192, 8192),  # N=16384,K=8192 M_LEQ_4
    (64, 8192, 8192),  # N=16384,K=8192 M_LEQ_64
]
_RETUNED_UNGATED_SHAPES = [
    (8, 1280, 16384),  # N=16384,K=1280 M_LEQ_8
    (64, 1280, 16384),  # N=16384,K=1280 M_LEQ_64
]


def _skip_unless_gfx950():
    from aiter.ops.triton.utils._triton.arch_info import get_arch

    if "gfx950" not in (get_arch() or ""):
        pytest.skip("FF-A16W16-fused configs are tuned for gfx950")


@pytest.mark.parametrize(
    "batch, hidden_dim, intermediate_dim", _M8_SHAPES + _RETUNED_GATED_SHAPES
)
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


@pytest.mark.parametrize(
    "batch, hidden_dim, intermediate_dim", _M8_SHAPES + _RETUNED_UNGATED_SHAPES
)
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


# Optional b_up / b_down of the ungated FF. The two small shapes are the action
# FFN of a vision-language-action model (M=8, 32 -> 256 -> 32) and a generic
# small FFN (M=16, 64 -> 512 -> 64); they take the one-program-along-N path.
# The others reach the atomic path with several programs along N and M, an odd
# hidden dim and a ragged K tail.
_BIAS_SHAPES = [
    (8, 32, 256),
    (16, 64, 512),
    (1, 128, 512),
    (3, 5, 2),
    (17, 200, 384),
    (7, 512, 1120),
    (64, 1024, 4096),
    (128, 4096, 1024),
]


def _ff_ungated_bias_ref(x, w1, w2, b_up, b_down, activation):
    h = F.linear(x.float(), w1.float(), None if b_up is None else b_up.float())
    if activation in ("gelu", "gelu_tanh"):
        h = F.gelu(h, approximate="tanh")
    elif activation in ("silu", "silu_exp2"):
        h = F.silu(h)
    elif activation == "relu":
        h = F.relu(h)
    y = h.to(x.dtype).float() @ w2.float()
    if b_down is not None:
        y = y + b_down.float()
    return y.to(x.dtype)


@pytest.mark.parametrize("bias", ["up", "down", "both"])
@pytest.mark.parametrize("activation", ["gelu_tanh", "silu_exp2", None])
@pytest.mark.parametrize("batch, hidden_dim, intermediate_dim", _BIAS_SHAPES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("output", [True, False])
def test_ff_a16w16_fused_ungated_bias(
    batch, hidden_dim, intermediate_dim, dtype, output, activation, bias
):
    torch.manual_seed(0)
    x, w1, w2, out_dtype, _, y = generate_ff_inputs(
        batch,
        hidden_dim,
        intermediate_dim,
        dtype,
        gating=False,
        output=output,
        y_init="zeros",
    )
    b_up = (
        torch.randn(intermediate_dim, dtype=dtype, device="cuda")
        if bias in ("up", "both")
        else None
    )
    b_down = (
        torch.randn(hidden_dim, dtype=dtype, device="cuda")
        if bias in ("down", "both")
        else None
    )
    ref = _ff_ungated_bias_ref(x, w1, w2, b_up, b_down, activation)
    out = ff_a16w16_fused_ungated(
        x, w1, w2, out_dtype, y=y, activation=activation, b_up=b_up, b_down=b_down
    )
    if output:
        assert out is y
    torch.testing.assert_close(out, ref, atol=5e-2, rtol=5e-2)


# With y not given and one program along N, the kernel stores y instead of adding
# to a zero-filled y. Each element gets one partial product either way, so the
# two paths must give the same bits. The last shape has four programs along M.
@pytest.mark.parametrize(
    "batch, hidden_dim, intermediate_dim",
    [(8, 32, 256), (16, 64, 512), (1, 128, 512), (3, 5, 2), (64, 64, 512)],
)
@pytest.mark.parametrize("with_bias", [False, True])
def test_ff_a16w16_fused_ungated_store_matches_atomic(
    batch, hidden_dim, intermediate_dim, with_bias
):
    torch.manual_seed(0)
    dtype = torch.bfloat16
    x, w1, w2, out_dtype, _, y = generate_ff_inputs(
        batch, hidden_dim, intermediate_dim, dtype, output=True, y_init="zeros"
    )
    b_up = b_down = None
    if with_bias:
        b_up = torch.randn(intermediate_dim, dtype=dtype, device="cuda")
        b_down = torch.randn(hidden_dim, dtype=dtype, device="cuda")
    config = {
        "BLOCK_SIZE_M": 16,
        "BLOCK_SIZE_N": max(16, triton.next_power_of_2(intermediate_dim)),
        "BLOCK_SIZE_K": max(16, triton.next_power_of_2(hidden_dim)),
        "GROUP_SIZE_M": 1,
        "num_warps": 4,
        "num_stages": 1,
        "waves_per_eu": 0,
        "matrix_instr_nonkdim": 16,
        "cache_modifier": ".cg",
    }
    kwargs = {
        "activation": "gelu_tanh",
        "config": config,
        "b_up": b_up,
        "b_down": b_down,
    }
    stored = ff_a16w16_fused_ungated(x, w1, w2, out_dtype, **kwargs)
    added = ff_a16w16_fused_ungated(x, w1, w2, out_dtype, y=y, **kwargs)
    assert torch.equal(stored, added)


# The gated kernel reads the same FF-A16W16-fused config files as the ungated one,
# keyed by the shape of w_up: (N, K) = (2 * intermediate_dim, hidden_dim). These
# gated shapes therefore select the one-program-along-N entries of the
# N=256-K=32 and N=512-K=64 files on gfx950.
@pytest.mark.parametrize(
    "batch, hidden_dim, intermediate_dim",
    [(1, 32, 128), (8, 32, 128), (64, 32, 128), (1, 64, 256), (16, 64, 256)],
)
@pytest.mark.parametrize("activation", ["silu_exp2", "gelu_tanh"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("output", [True, False])
def test_ff_a16w16_fused_gated_small_shared_config(
    batch, hidden_dim, intermediate_dim, dtype, output, activation
):
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


def test_ff_a16w16_fused_ungated_bias_rejects_wrong_shape():
    x, w1, w2, out_dtype, _, _ = generate_ff_inputs(
        8, 32, 256, torch.bfloat16, output=False
    )
    b = torch.zeros(32, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(AssertionError):
        ff_a16w16_fused_ungated(x, w1, w2, out_dtype, b_up=b)
    with pytest.raises(AssertionError):
        ff_a16w16_fused_ungated(x, w1, w2, out_dtype, b_down=b.new_zeros(256))
