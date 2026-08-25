# aiter/ops/triton/moe/moe_aux_loss.py
import torch
import triton
import triton.language as tl


@triton.jit
def _moe_aux_loss_fwd_kernel(
    probs_ptr,
    tokens_per_expert_ptr,
    out_ptr,
    coeff_scaled,
    N: tl.constexpr,
    E: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    expert_id = tl.program_id(0)
    col_sum = tl.zeros([], dtype=tl.float32)
    for start in range(0, N, BLOCK_N):
        row_ids = start + tl.arange(0, BLOCK_N)
        mask = row_ids < N
        vals = tl.load(probs_ptr + row_ids * E + expert_id, mask=mask, other=0.0)
        col_sum += tl.sum(vals)

    tpe = tl.load(tokens_per_expert_ptr + expert_id)
    partial = col_sum * tpe * coeff_scaled
    tl.atomic_add(out_ptr, partial)


def moe_aux_loss_fwd(
    probs: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    coeff_scaled: float,
) -> torch.Tensor:
    N, E = probs.shape
    out = torch.zeros((), dtype=torch.float32, device=probs.device)
    if N == 0:
        return out
    BLOCK_N = min(triton.next_power_of_2(N), 1024)
    _moe_aux_loss_fwd_kernel[(E,)](
        probs,
        tokens_per_expert,
        out,
        coeff_scaled,
        N,
        E,
        BLOCK_N,
    )
    return out


@triton.jit
def _moe_aux_loss_bwd_kernel(
    tokens_per_expert_ptr,
    grad_probs_ptr,
    scale,
    N: tl.constexpr,
    E: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    expert_id = tl.program_id(0)
    tpe_val = tl.load(tokens_per_expert_ptr + expert_id)
    fill_val = tpe_val * scale

    for start in range(0, N, BLOCK_N):
        row_ids = start + tl.arange(0, BLOCK_N)
        mask = row_ids < N
        tl.store(grad_probs_ptr + row_ids * E + expert_id, fill_val, mask=mask)


def moe_aux_loss_bwd(
    tokens_per_expert: torch.Tensor,
    coeff_scaled: float,
    grad_aux_loss: torch.Tensor,
    num_tokens: int,
    num_experts: int,
) -> torch.Tensor:
    if num_tokens == 0:
        return torch.empty(
            0, num_experts, dtype=torch.float32, device=tokens_per_expert.device
        )
    grad_probs = torch.empty(
        num_tokens, num_experts, dtype=torch.float32, device=tokens_per_expert.device
    )
    scale = coeff_scaled * grad_aux_loss.item()
    BLOCK_N = min(triton.next_power_of_2(num_tokens), 1024)
    _moe_aux_loss_bwd_kernel[(num_experts,)](
        tokens_per_expert,
        grad_probs,
        scale,
        num_tokens,
        num_experts,
        BLOCK_N,
    )
    return grad_probs
