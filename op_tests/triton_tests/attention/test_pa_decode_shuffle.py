# SPDX-License-Identifier: MIT

import pytest
import torch

import aiter
from aiter.ops.triton.attention.pa_decode_shuffle import (
    paged_attention_decode_shuffle_swa,
)
from aiter.ops.triton.utils.types import e4m3_dtype

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a ROCm GPU"
)


def _plain_cache(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    blocks, heads, _, block_size, _width = key_cache.shape
    head_size = value_cache.shape[3]
    key = (
        key_cache.permute(0, 3, 1, 2, 4)
        .contiguous()
        .view(blocks, block_size, heads, head_size)
    )
    value = (
        value_cache.permute(0, 2, 4, 1, 3)
        .contiguous()
        .view(blocks, block_size, heads, head_size)
    )
    return key, value


@pytest.mark.parametrize(
    "block_size,context_values",
    [(16, [0, 1, 31]), (128, [0, 1, 255])],
)
@torch.inference_mode()
def test_per_token_cache_producer_to_shuffled_decode(
    block_size: int, context_values: list[int]
) -> None:
    torch.manual_seed(7)
    device = "cuda"
    num_seqs, num_q_heads, num_kv_heads = 3, 16, 1
    head_size, num_blocks = 128, 6
    width = 16 // torch.tensor([], dtype=e4m3_dtype).element_size()
    context_lens = torch.tensor(context_values, dtype=torch.int32, device=device)
    block_tables = torch.tensor(
        [[0, 1], [2, 3], [4, 5]], dtype=torch.int32, device=device
    )

    slot_mapping = torch.cat(
        (
            torch.tensor([2 * block_size], device=device),
            torch.arange(
                4 * block_size,
                4 * block_size + context_values[-1],
                device=device,
            ),
        )
    ).to(torch.int64)
    key = torch.randn(
        slot_mapping.numel(),
        num_kv_heads,
        head_size,
        dtype=torch.bfloat16,
        device=device,
    )
    value = torch.randn_like(key)
    key_cache = torch.full(
        (num_blocks, num_kv_heads, head_size // width, block_size, width),
        float("nan"),
        dtype=e4m3_dtype,
        device=device,
    )
    value_cache = torch.full(
        (num_blocks, num_kv_heads, block_size // width, head_size, width),
        float("nan"),
        dtype=e4m3_dtype,
        device=device,
    )
    k_scale = torch.full(
        (num_blocks, num_kv_heads, block_size),
        float("nan"),
        dtype=torch.float32,
        device=device,
    )
    v_scale = torch.full_like(k_scale, float("nan"))
    aiter.reshape_and_cache_with_pertoken_quant(
        key,
        value,
        key_cache,
        value_cache,
        k_scale,
        v_scale,
        slot_mapping,
        True,
    )

    query = torch.randn(
        num_seqs,
        num_q_heads,
        head_size,
        dtype=torch.bfloat16,
        device=device,
    )
    output = torch.empty_like(query)
    softmax_scale = head_size**-0.5
    paged_attention_decode_shuffle_swa(
        output,
        query,
        key_cache,
        value_cache,
        block_tables,
        context_lens,
        softmax_scale,
        k_scale,
        v_scale,
        sliding_window=-1,
    )

    key_plain, value_plain = _plain_cache(key_cache, value_cache)
    reference = torch.zeros_like(output, dtype=torch.float32)
    for seq, context_len in enumerate(context_lens.tolist()):
        if context_len == 0:
            continue
        pages = block_tables[seq, : (context_len + block_size - 1) // block_size]
        seq_key = key_plain[pages].reshape(-1, num_kv_heads, head_size)[:context_len]
        seq_value = value_plain[pages].reshape(-1, num_kv_heads, head_size)[
            :context_len
        ]
        seq_k_scale = (
            k_scale[pages].permute(0, 2, 1).reshape(-1, num_kv_heads)[:context_len]
        )
        seq_v_scale = (
            v_scale[pages].permute(0, 2, 1).reshape(-1, num_kv_heads)[:context_len]
        )
        seq_key = seq_key.float() * seq_k_scale[..., None]
        seq_value = seq_value.float() * seq_v_scale[..., None]
        group = num_q_heads // num_kv_heads
        kv_head_for_q = torch.arange(num_q_heads, device=device) // group
        seq_key = seq_key[:, kv_head_for_q]
        seq_value = seq_value[:, kv_head_for_q]
        scores = torch.einsum("hd,khd->hk", query[seq].float(), seq_key)
        scores.mul_(softmax_scale)
        reference[seq] = torch.einsum(
            "hk,khd->hd", torch.softmax(scores, dim=-1), seq_value
        )

    assert torch.isfinite(output).all()
    torch.testing.assert_close(output[0], torch.zeros_like(output[0]))
    cosine = torch.nn.functional.cosine_similarity(
        output[1:].float().flatten(),
        reference[1:].flatten(),
        dim=0,
    )
    assert cosine.item() >= 0.999
