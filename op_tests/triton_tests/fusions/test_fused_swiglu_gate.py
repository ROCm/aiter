import torch
import pytest

from aiter.ops.triton.fusions.fused_swiglu_gate import fused_swiglu_gate

# MiniMax-M3 TP4: local intermediate d = 1536 // 4 = 384, gate-up last dim = 768.
_MINIMAX_TP4_LAST = 768
_MINIMAX_TOP_K = 8
_MINIMAX_ALPHA = 1.702
_MINIMAX_LIMIT = 7.0


def torch_swiglu_gate_ref(
    x: torch.Tensor,
    alpha: float = _MINIMAX_ALPHA,
    limit: float = _MINIMAX_LIMIT,
    add_residual: bool = True,
) -> torch.Tensor:
    gate, up = x.chunk(2, dim=-1)
    if limit > 0:
        gate = gate.clamp(max=limit)
        up = up.clamp(min=-limit, max=limit)
    out = gate * torch.sigmoid(gate * alpha)
    if add_residual:
        out = out * (up + 1)
    else:
        out = out * up
    return out.to(x.dtype)


@pytest.mark.parametrize(
    "shape",
    [
        (4, 64),
        (128, 256),
        (31, 500),
        (2, 16, 128),
        (1, 3, 7, 32),
    ],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("swiglu_limit", [0.0, 7.0])
@pytest.mark.parametrize("add_residual", [True, False])
@pytest.mark.parametrize("use_explicit_out", [False, True])
def test_fused_swiglu_gate(
    shape, dtype, swiglu_limit, add_residual, use_explicit_out
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    x = torch.randn(shape, dtype=dtype, device="cuda")
    ref = torch_swiglu_gate_ref(
        x, limit=swiglu_limit, add_residual=add_residual
    )
    if use_explicit_out:
        out = torch.empty_like(ref)
        fused_swiglu_gate(
            x,
            out,
            swiglu_limit=swiglu_limit,
            add_residual=add_residual,
        )
    else:
        out = fused_swiglu_gate(
            x,
            swiglu_limit=swiglu_limit,
            add_residual=add_residual,
        )
    torch.testing.assert_close(out, ref, rtol=2e-2, atol=0.15)


def test_fused_swiglu_gate_requires_even_last_dim():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    x = torch.randn(2, 3, device="cuda")
    with pytest.raises(AssertionError, match="even"):
        fused_swiglu_gate(x)


@pytest.mark.parametrize(
    "n_rows,last_dim",
    [
        pytest.param(4 * _MINIMAX_TOP_K, _MINIMAX_TP4_LAST, id="minimax_tp4_decode4"),
        pytest.param(
            256 * _MINIMAX_TOP_K, _MINIMAX_TP4_LAST, id="minimax_tp4_rows256x8"
        ),
        pytest.param(
            (8190 + 3) * _MINIMAX_TOP_K,
            _MINIMAX_TP4_LAST,
            id="minimax_tp4_pref8190_dec3",
        ),
        pytest.param(
            (7235 + 3) * _MINIMAX_TOP_K,
            _MINIMAX_TP4_LAST,
            id="minimax_tp4_pref7235_dec3",
        ),
    ],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_fused_swiglu_gate_minimax_m3_shapes(n_rows, last_dim, dtype):
    """MoE / MLP gate activation as (tokens * top_k, 2 * local_d) under TP4."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    x = torch.randn(n_rows, last_dim, dtype=dtype, device="cuda")
    ref = torch_swiglu_gate_ref(x)
    out = fused_swiglu_gate(x)
    # exp2-based sigmoid can differ slightly from torch.sigmoid at bf16 scale.
    torch.testing.assert_close(out, ref, rtol=2e-2, atol=0.15)
