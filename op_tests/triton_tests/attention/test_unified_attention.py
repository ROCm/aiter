# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Optional

import pytest
import torch

from aiter.ops.triton.attention.unified_attention import unified_attention
from aiter.ops.triton.utils.types import e4m3_dtype

NUM_HEADS = [(4, 4), (8, 2), (16, 2)]
HEAD_SIZES = [128, 256]
BLOCK_SIZES = [16, 64]

DTYPES = [torch.float16, torch.bfloat16]
QDTYPES = [None, e4m3_dtype]
# one value large enough to test overflow in index calculation.
# one value small enough to test the schema op check
NUM_BLOCKS = [32768, 2048]


def ref_paged_attn(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    query_lens: list[int],
    kv_lens: list[int],
    block_tables: torch.Tensor,
    scale: float,
    sliding_window: Optional[int] = None,
    soft_cap: Optional[float] = None,
    sinks: Optional[torch.Tensor] = None,
    q_descale: Optional[torch.Tensor] = None,
    k_descale: Optional[torch.Tensor] = None,
    v_descale: Optional[torch.Tensor] = None,
    output_scale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    num_seqs = len(query_lens)
    block_tables = block_tables.cpu().numpy()
    _, block_size, num_kv_heads, head_size = key_cache.shape
    orig_dtype = query.dtype
    outputs: list[torch.Tensor] = []
    start_idx = 0
    query = query.to(torch.float32)
    key_cache = key_cache.to(torch.float32)
    value_cache = value_cache.to(torch.float32)
    if q_descale is not None:
        query = query * q_descale
    if k_descale is not None:
        key_cache = key_cache * k_descale
    if v_descale is not None:
        value_cache = value_cache * v_descale
    for i in range(num_seqs):
        query_len = query_lens[i]
        kv_len = kv_lens[i]
        q = query[start_idx : start_idx + query_len]
        q *= scale

        num_kv_blocks = (kv_len + block_size - 1) // block_size
        block_indices = block_tables[i, :num_kv_blocks]

        k = key_cache[block_indices].view(-1, num_kv_heads, head_size)
        k = k[:kv_len]
        v = value_cache[block_indices].view(-1, num_kv_heads, head_size)
        v = v[:kv_len]

        if q.shape[1] != k.shape[1]:
            k = torch.repeat_interleave(k, q.shape[1] // k.shape[1], dim=1)
            v = torch.repeat_interleave(v, q.shape[1] // v.shape[1], dim=1)
        attn = torch.einsum("qhd,khd->hqk", q, k).float()
        empty_mask = torch.ones(query_len, kv_len, device=q.device)
        mask = torch.triu(empty_mask, diagonal=kv_len - query_len + 1).bool()
        if sliding_window is not None:
            sliding_window_mask = (
                torch.triu(
                    empty_mask, diagonal=kv_len - (query_len + sliding_window) + 1
                )
                .bool()
                .logical_not()
            )
            mask |= sliding_window_mask
        if soft_cap is not None and soft_cap > 0:
            attn = soft_cap * torch.tanh(attn / soft_cap)
        attn.masked_fill_(mask, float("-inf"))
        if sinks is not None:
            s_aux = sinks[:, None, None].repeat_interleave(attn.shape[-2], dim=-2)
            attn = torch.cat((attn, s_aux), dim=-1)
        attn = torch.softmax(attn, dim=-1).to(v.dtype)
        if sinks is not None:
            attn = attn[..., :-1]
        out = torch.einsum("hqk,khd->qhd", attn, v)

        outputs.append(out)
        start_idx += query_len

    out = torch.cat(outputs, dim=0)
    if output_scale is not None:
        out = out / output_scale
    return out.to(orig_dtype)


@pytest.mark.parametrize(
    "seq_lens", [[(1, 1328), (5, 18), (129, 463)], [(1, 523), (1, 37), (1, 2011)]]
)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("sliding_window", [None, 256])
@pytest.mark.parametrize("soft_cap", [None, 50.0])
@pytest.mark.parametrize("num_blocks", NUM_BLOCKS)
@pytest.mark.parametrize(
    "q_dtype, kv_dtype, use_kv_descale, use_out_scale, use_q_descale",
    [
        (torch.bfloat16, torch.bfloat16, False, False, False),
        (e4m3_dtype, e4m3_dtype, True, True, True),
        (e4m3_dtype, e4m3_dtype, True, False, True),
        (torch.bfloat16, e4m3_dtype, True, False, False),
        (torch.bfloat16, e4m3_dtype, True, False, False),
    ],
)
@torch.inference_mode()
def test_triton_unified_attn(
    seq_lens: list[tuple[int, int]],
    num_heads: tuple[int, int],
    head_size: int,
    sliding_window: Optional[int],
    block_size: int,
    soft_cap: Optional[float],
    num_blocks: int,
    q_dtype: torch.dtype,
    kv_dtype: torch.dtype,
    use_kv_descale: bool,
    use_out_scale: bool,
    use_q_descale: bool,
) -> None:
    if q_dtype is not None and q_dtype.itemsize < 2 and block_size < 32:
        pytest.skip("block size must be at least 32 for fp8")

    torch.manual_seed(0)
    num_seqs = len(seq_lens)
    query_lens = [x[0] for x in seq_lens]
    kv_lens = [x[1] for x in seq_lens]
    num_query_heads = num_heads[0]
    num_kv_heads = num_heads[1]
    assert num_query_heads % num_kv_heads == 0
    max_query_len = max(query_lens)
    max_kv_len = max(kv_lens)
    window_size = (sliding_window - 1, 0) if sliding_window is not None else (-1, -1)
    scale = head_size**-0.5

    query = torch.randn(
        sum(query_lens), num_query_heads, head_size, dtype=torch.float32, device="cuda"
    )
    key_cache = torch.randn(
        num_blocks,
        block_size,
        num_kv_heads,
        head_size,
        dtype=torch.float32,
        device="cuda",
    )
    value_cache = torch.randn_like(key_cache)
    query = query.to(q_dtype)
    key_cache = key_cache.to(kv_dtype)
    value_cache = value_cache.to(kv_dtype)
    cu_query_lens = torch.tensor(
        [0] + query_lens, dtype=torch.int32, device="cuda"
    ).cumsum(dim=0, dtype=torch.int32)
    kv_lens = torch.tensor(kv_lens, dtype=torch.int32, device="cuda")

    max_num_blocks_per_seq = (max_kv_len + block_size - 1) // block_size
    block_tables = torch.randint(
        0,
        num_blocks,
        (num_seqs, max_num_blocks_per_seq),
        dtype=torch.int32,
        device="cuda",
    )
    sinks = torch.randn(num_query_heads, dtype=torch.bfloat16, device="cuda")
    output = torch.empty_like(query)

    maybe_quantized_query = query
    maybe_quantized_key_cache = key_cache
    maybe_quantized_value_cache = value_cache
    q_descale = None  # Not yet supported
    k_descale = None
    v_descale = None
    out_scale = None
    # QKV are drawn from N(0, 1): no need to calculate the descales from the original tensors
    # generate random descales for testing
    if use_kv_descale:
        k_descale = torch.rand(1, dtype=torch.float32, device="cuda")
        v_descale = torch.rand(1, dtype=torch.float32, device="cuda")
    if use_out_scale:
        out_scale = 1 / torch.rand(1, dtype=torch.float32, device="cuda")
    if use_q_descale:
        q_descale = torch.rand(1, dtype=torch.float32, device="cuda")

    unified_attention(
        q=maybe_quantized_query,
        k=maybe_quantized_key_cache,
        v=maybe_quantized_value_cache,
        out=output,
        cu_seqlens_q=cu_query_lens,
        seqused_k=kv_lens,
        max_seqlen_q=max_query_len,
        max_seqlen_k=max_kv_len,
        softmax_scale=scale,
        causal=True,
        window_size=window_size,
        block_table=block_tables,
        softcap=soft_cap if soft_cap is not None else 0,
        q_descale=q_descale,
        k_descale=k_descale,
        v_descale=v_descale,
        sinks=sinks,
        output_scale=out_scale,
    )

    ref_output = ref_paged_attn(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        query_lens=query_lens,
        kv_lens=kv_lens,
        block_tables=block_tables,
        scale=scale,
        sliding_window=sliding_window,
        soft_cap=soft_cap,
        sinks=sinks,
        q_descale=q_descale,
        k_descale=k_descale,
        v_descale=v_descale,
        output_scale=out_scale,
    )
    atol, rtol = 1.5e-2, 1e-2
    if kv_dtype.itemsize == 1:
        atol, rtol = 1.5e-1, 1.5e-1
    output = output.to(torch.float32)
    ref_output = ref_output.to(torch.float32)
    torch.testing.assert_close(
        output, ref_output, atol=atol, rtol=rtol
    ), f"{torch.max(torch.abs(output - ref_output))}"
