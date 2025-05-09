import torch
import pytest
from typing import Tuple

from aiter.ops.triton.moe_align_block_size import moe_align_block_size_triton
from op_tests.triton_tests.utils.moe_align_block_size_ref import torch_moe_align_block_size
from op_benchmarks.triton.utils.moe import generate_moe_logits


def triton_moe_align_block_size(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    max_num_tokens_padded = topk_ids.numel() + num_experts * (block_size - 1)
    sorted_ids = torch.full((max_num_tokens_padded,), topk_ids.numel(),
                            dtype=torch.int32, device="cuda")
    max_num_m_blocks = (max_num_tokens_padded + block_size - 1) // block_size
    expert_ids = torch.empty(max_num_m_blocks, dtype=torch.int32, device="cuda")
    num_tokens_post_pad = torch.empty((1), dtype=torch.int32, device="cuda")

    moe_align_block_size_triton(topk_ids,
                                num_experts,
                                block_size,
                                sorted_ids,
                                expert_ids,
                                num_tokens_post_pad,
                            )

    return sorted_ids, expert_ids, num_tokens_post_pad


@pytest.mark.parametrize("M, E, top_k, block_size",
    [(1, 2, 1, 16),
    (4, 2, 1, 16),
    (8, 2, 1, 16),
    (16, 2, 1, 16),
    (16, 8, 4, 16),
    (16, 8, 4, 32),
    (16, 8, 4, 128),
    (16, 48, 4, 64),
    (32, 224, 12, 128),
    ],
)
def test_correctness(M: int, E: int, top_k: int, block_size: int):
    _, topk_ids = generate_moe_logits(M, E, top_k)

    tri_sorted_ids, tri_expert_ids, tri_num_tokens_post_pad = triton_moe_align_block_size(topk_ids, block_size, E)
    torch_sorted_ids, torch_expert_ids, torch_num_tokens_post_pad = torch_moe_align_block_size(topk_ids, E, block_size)

    torch.eq(tri_sorted_ids, torch_sorted_ids)
    torch.eq(tri_num_tokens_post_pad, torch_num_tokens_post_pad)
    torch.eq(tri_expert_ids[:E], torch_expert_ids[:E])
