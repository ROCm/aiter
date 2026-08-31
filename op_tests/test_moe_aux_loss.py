# op_tests/test_moe_aux_loss.py
import pytest
import torch

torch.set_default_device("cuda")


def aux_loss_ref(probs, tokens_per_expert, coeff_scaled):
    """Pure PyTorch reference."""
    aggregated = probs.sum(dim=0)
    return (aggregated * tokens_per_expert).sum() * coeff_scaled


@pytest.mark.parametrize("N", [1, 32, 512, 4096])
@pytest.mark.parametrize("E", [8, 64, 128])
def test_moe_aux_loss_fwd(N, E):
    from aiter.ops.triton.moe.moe_aux_loss import moe_aux_loss_fwd

    torch.manual_seed(42)
    probs = torch.rand(N, E, dtype=torch.float32)
    tokens_per_expert = torch.randint(0, N, (E,), dtype=torch.float32)
    coeff_scaled = 0.01

    result = moe_aux_loss_fwd(probs, tokens_per_expert, coeff_scaled)
    expected = aux_loss_ref(probs, tokens_per_expert, coeff_scaled)

    torch.testing.assert_close(result, expected, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("N", [1, 32, 512])
@pytest.mark.parametrize("E", [8, 64])
def test_moe_aux_loss_bwd(N, E):
    from aiter.ops.triton.moe.moe_aux_loss import moe_aux_loss_bwd

    torch.manual_seed(42)
    tokens_per_expert = torch.randint(0, N, (E,), dtype=torch.float32)
    coeff_scaled = 0.01
    grad_aux_loss = torch.tensor(1.0)

    result = moe_aux_loss_bwd(tokens_per_expert, coeff_scaled, grad_aux_loss, N, E)

    expected = tokens_per_expert.unsqueeze(0).expand(N, E) * coeff_scaled
    torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-6)


def test_moe_aux_loss_fwd_n_zero():
    from aiter.ops.triton.moe.moe_aux_loss import moe_aux_loss_fwd

    probs = torch.empty(0, 8, dtype=torch.float32)
    tokens_per_expert = torch.ones(8, dtype=torch.float32)
    result = moe_aux_loss_fwd(probs, tokens_per_expert, 0.01)
    assert result.item() == 0.0


def test_moe_aux_loss_bwd_n_zero():
    from aiter.ops.triton.moe.moe_aux_loss import moe_aux_loss_bwd

    tokens_per_expert = torch.ones(8, dtype=torch.float32)
    result = moe_aux_loss_bwd(tokens_per_expert, 0.01, torch.tensor(1.0), 0, 8)
    assert result.shape == (0, 8)
