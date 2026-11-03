import pytest
import torch

import aiter
from aiter.ops.triton.moe.moe_sort_small import moe_sort_small


def _alloc(m, topk, num_experts, block_size, device):
    max_padded = m * topk + num_experts * block_size - topk
    num_blocks = (max_padded + block_size - 1) // block_size
    max_padded = num_blocks * block_size
    return (
        torch.empty(max_padded, dtype=torch.int32, device=device),
        torch.empty(max_padded, dtype=torch.float32, device=device),
        torch.empty(num_blocks, dtype=torch.int32, device=device),
        torch.empty(2, dtype=torch.int32, device=device),
    )


@pytest.mark.parametrize("m", [1, 3, 16, 107, 224])
@pytest.mark.parametrize("block_size", [16, 32])
@pytest.mark.parametrize("buf_cols", [6144, 12288])
def test_moe_sort_small_matches_opus(m, block_size, buf_cols):
    num_experts, topk, device = 128, 4, "cuda"
    torch.manual_seed(m)
    topk_weights, topk_ids = torch.rand(m, num_experts, device=device).topk(topk)
    topk_ids = topk_ids.to(torch.int32).contiguous()
    topk_weights = topk_weights.contiguous()

    ref = _alloc(m, topk, num_experts, block_size, device)
    ref_buf = torch.ones(m, buf_cols, dtype=torch.bfloat16, device=device)
    ws_size = aiter.moe_sorting_opus_get_workspace_size(
        m, num_experts, topk, 0, topk_ids.get_device()
    )
    ws = torch.empty(ws_size, dtype=torch.uint8, device=device) if ws_size else None
    aiter.moe_sorting_opus_fwd(
        topk_ids, topk_weights, *ref, ref_buf, num_experts, block_size,
        None, None, ws, 0, None, None, None,
    )

    out = _alloc(m, topk, num_experts, block_size, device)
    buf = torch.ones(m, buf_cols, dtype=torch.bfloat16, device=device)
    moe_sort_small(topk_ids, topk_weights, *out, buf, num_experts, block_size)

    torch.testing.assert_close(out[3], ref[3], rtol=0, atol=0)
    n = int(ref[3][0])
    torch.testing.assert_close(out[0][:n], ref[0][:n], rtol=0, atol=0)
    torch.testing.assert_close(out[1][:n], ref[1][:n], rtol=0, atol=0)
    torch.testing.assert_close(out[2][: n // block_size], ref[2][: n // block_size], rtol=0, atol=0)
    assert bool((buf == 0).all()) and bool((ref_buf == 0).all())
