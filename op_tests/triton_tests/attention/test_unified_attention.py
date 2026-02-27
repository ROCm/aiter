# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Optional

import pytest
import torch

from aiter.ops.triton.attention.unified_attention import unified_attention
from aiter.ops.triton.utils.types import e4m3_dtype
from aiter.ops.triton.utils._triton import arch_info
from aiter.ops.triton.utils.device_info import get_num_sms

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
    k_block_scale: Optional[torch.Tensor] = None,  # [num_blocks, num_kv_heads]
    v_block_scale: Optional[torch.Tensor] = None,  # [num_blocks, num_kv_heads]
    second_dot_dtype: torch.dtype = torch.float32,
    run_on: str = "cuda",
) -> torch.Tensor:
    num_seqs = len(query_lens)
    block_tables_np = block_tables.cpu().numpy()
    _, block_size, num_kv_heads, head_size = key_cache.shape

    outputs: list[torch.Tensor] = []
    start_idx = 0
    query = query.to(torch.float32)
    prev_device = query.device
    query = query.to(run_on)
    key_cache = key_cache.to(run_on).float()
    value_cache = value_cache.to(run_on).float()
    if k_block_scale is not None:
        k_block_scale = k_block_scale.to(run_on)
        v_block_scale = v_block_scale.to(run_on)
    if sinks is not None:
        sinks = sinks.to(run_on)
    for i in range(num_seqs):
        query_len = query_lens[i]
        kv_len = kv_lens[i]
        q = query[start_idx : start_idx + query_len]
        q *= scale

        num_kv_blocks = (kv_len + block_size - 1) // block_size
        block_indices = block_tables_np[i, :num_kv_blocks]

        k = key_cache[block_indices]
        v = value_cache[block_indices]

        if k_block_scale is not None and v_block_scale is not None:
            my_k_block_scale = k_block_scale[block_indices, :]
            my_v_block_scale = v_block_scale[block_indices, :]
            k = k.to(torch.float32) * my_k_block_scale[:, None, :, None]
            v = v.to(torch.float32) * my_v_block_scale[:, None, :, None]

        k = k.view(-1, num_kv_heads, head_size).float()
        k = k[:kv_len]
        v = v.view(-1, num_kv_heads, head_size).float()
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
        # to capture the precision loss
        attn = attn.to(second_dot_dtype).to(torch.float32)
        v = v.to(second_dot_dtype).to(torch.float32)
        out = torch.einsum("hqk,khd->qhd", attn, v)

        outputs.append(out)
        start_idx += query_len

    return torch.cat(outputs, dim=0).to(prev_device)


def gen_testdata(
    seq_lens: list[tuple[int, int]],
    num_heads: tuple[int, int],
    head_size: int,
    dtype: torch.dtype,
    block_size: int,
    num_blocks: int,
    q_dtype: Optional[torch.dtype],
    device: str = "cuda",
):
    num_seqs = len(seq_lens)
    query_lens = [x[0] for x in seq_lens]
    kv_lens_list = [x[1] for x in seq_lens]
    num_query_heads = num_heads[0]
    num_kv_heads = num_heads[1]
    max_query_len = max(query_lens)
    max_kv_len = max(kv_lens_list)

    query = torch.randn(
        sum(query_lens), num_query_heads, head_size, dtype=torch.float32, device=device
    ).to(dtype)
    key_cache = torch.randn(
        num_blocks,
        block_size,
        num_kv_heads,
        head_size,
        dtype=torch.float32,
        device=device,
    )
    value_cache = torch.randn_like(key_cache)

    key_cache = key_cache.to(dtype)
    value_cache = value_cache.to(dtype)

    cu_query_lens = torch.tensor(
        [0] + query_lens, dtype=torch.int32, device=device
    ).cumsum(dim=0, dtype=torch.int32)
    kv_lens = torch.tensor(kv_lens_list, dtype=torch.int32, device=device)

    max_num_blocks_per_seq = (max_kv_len + block_size - 1) // block_size
    block_tables = torch.randint(
        0,
        num_blocks,
        (num_seqs, max_num_blocks_per_seq),
        dtype=torch.int32,
        device=device,
    )
    sinks = torch.randn(num_query_heads, dtype=torch.bfloat16, device=device)
    output = torch.empty_like(query)

    maybe_quantized_query = query
    maybe_quantized_key_cache = key_cache
    maybe_quantized_value_cache = value_cache
    q_descale = None
    k_descale = None
    v_descale = None

    if q_dtype is not None:
        maybe_quantized_query = query.to(q_dtype)
        maybe_quantized_key_cache = key_cache.to(q_dtype)
        maybe_quantized_value_cache = value_cache.to(q_dtype)
        scale_shape = (num_seqs, num_kv_heads)
        # Q scale not yet supported, keep as None
        k_descale = torch.rand(scale_shape, dtype=torch.float32, device=device)
        v_descale = torch.rand(scale_shape, dtype=torch.float32, device=device)

    return dict(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        query_lens=query_lens,
        kv_lens=kv_lens,
        num_query_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
        max_query_len=max_query_len,
        max_kv_len=max_kv_len,
        block_tables=block_tables,
        sinks=sinks,
        output=output,
        maybe_quantized_query=maybe_quantized_query,
        maybe_quantized_key_cache=maybe_quantized_key_cache,
        maybe_quantized_value_cache=maybe_quantized_value_cache,
        q_descale=q_descale,
        k_descale=k_descale,
        v_descale=v_descale,
        cu_query_lens=cu_query_lens,
    )


@pytest.mark.parametrize(
    "seq_lens", [[(1, 1328), (5, 18), (129, 463)], [(1, 523), (1, 37), (1, 2011)]]
)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("sliding_window", [None, 256])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("soft_cap", [None, 10.0, 50.0])
@pytest.mark.parametrize("num_blocks", NUM_BLOCKS)
@pytest.mark.parametrize("q_dtype", QDTYPES)
@torch.inference_mode()
def test_triton_unified_attn(
    seq_lens: list[tuple[int, int]],
    num_heads: tuple[int, int],
    head_size: int,
    sliding_window: Optional[int],
    dtype: torch.dtype,
    block_size: int,
    soft_cap: Optional[float],
    num_blocks: int,
    q_dtype: Optional[torch.dtype],
) -> None:
    if q_dtype is not None and q_dtype.itemsize < 2 and block_size < 32:
        pytest.skip("block size must be at least 32 for fp8")
    # TODO: fix this failing test for gfx942
    if arch_info.get_arch() == "gfx942" and get_num_sms() == 80 and block_size < 32:
        pytest.skip(f"block size must be at least 32 for {arch_info.get_arch()}")

    torch.manual_seed(0)
    data = gen_testdata(
        seq_lens, num_heads, head_size, dtype, block_size, num_blocks, q_dtype
    )
    window_size = (sliding_window - 1, 0) if sliding_window is not None else (-1, -1)
    scale = head_size**-0.5

    unified_attention(
        q=data["maybe_quantized_query"],
        k=data["maybe_quantized_key_cache"],
        v=data["maybe_quantized_value_cache"],
        out=data["output"],
        cu_seqlens_q=data["cu_query_lens"],
        seqused_k=data["kv_lens"],
        max_seqlen_q=data["max_query_len"],
        max_seqlen_k=data["max_kv_len"],
        softmax_scale=scale,
        causal=True,
        window_size=window_size,
        block_table=data["block_tables"],
        softcap=soft_cap if soft_cap is not None else 0,
        q_descale=data["q_descale"],
        k_descale=data["k_descale"],
        v_descale=data["v_descale"],
        sinks=data["sinks"],
    )

    ref_output = ref_paged_attn(
        query=data["query"],
        key_cache=data["key_cache"],
        value_cache=data["value_cache"],
        query_lens=data["query_lens"],
        kv_lens=data["kv_lens"],
        block_tables=data["block_tables"],
        scale=scale,
        sliding_window=sliding_window,
        soft_cap=soft_cap,
        sinks=data["sinks"],
        run_on="cpu",
    )
    ref_output = ref_output.to(data["output"].device).to(data["output"].dtype)
    atol, rtol = 1.5e-2, 1e-2
    if q_dtype is not None:
        atol, rtol = 1.5e-1, 1.5e-1
    torch.testing.assert_close(
        data["output"], ref_output, atol=atol, rtol=rtol
    ), f"{torch.max(torch.abs(data['output'] - ref_output))}"


@pytest.mark.parametrize(
    "seq_lens", [[(1, 1328), (5, 18), (129, 463)], [(1, 523), (1, 37), (1, 2011)]]
)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("block_size", [512, 1024])
@pytest.mark.parametrize("sliding_window", [None, 256])
@pytest.mark.parametrize("soft_cap", [None, 10.0, 50.0])
@pytest.mark.parametrize("num_blocks", [1024, 64])
@torch.inference_mode()
def test_triton_unified_attn_blockscale(
    seq_lens: list[tuple[int, int]],
    num_heads: tuple[int, int],
    head_size: int,
    sliding_window: Optional[int],
    block_size: int,
    soft_cap: Optional[float],
    num_blocks: int,
) -> None:
    q_dtype = e4m3_dtype
    dtype = e4m3_dtype
    if block_size < 32:
        pytest.skip("block size must be at least 32 for fp8")
    # TODO: fix this failing test for gfx942
    if arch_info.get_arch() == "gfx942" and get_num_sms() == 80 and block_size < 32:
        pytest.skip(f"block size must be at least 32 for {arch_info.get_arch()}")

    torch.manual_seed(0)
    num_kv_heads = num_heads[1]
    data = gen_testdata(
        seq_lens, num_heads, head_size, dtype, block_size, num_blocks, q_dtype
    )
    kv_scale = 0.5 + torch.rand(
        num_blocks, num_kv_heads, 2, dtype=torch.float32, device="cuda"
    )
    k_block_scale = kv_scale[:, :, 0]
    v_block_scale = kv_scale[:, :, 1]
    window_size = (sliding_window - 1, 0) if sliding_window is not None else (-1, -1)
    scale = head_size**-0.5

    unified_attention(
        q=data["maybe_quantized_query"],
        k=data["maybe_quantized_key_cache"],
        v=data["maybe_quantized_value_cache"],
        out=data["output"],
        cu_seqlens_q=data["cu_query_lens"],
        seqused_k=data["kv_lens"],
        max_seqlen_q=data["max_query_len"],
        max_seqlen_k=data["max_kv_len"],
        softmax_scale=scale,
        causal=True,
        window_size=window_size,
        block_table=data["block_tables"],
        softcap=soft_cap if soft_cap is not None else 0,
        q_descale=None,
        k_descale=None,
        v_descale=None,
        sinks=None,
        k_block_scale=k_block_scale,
        v_block_scale=v_block_scale,
    )

    ref_output = ref_paged_attn(
        query=data["query"],
        key_cache=data["key_cache"],
        value_cache=data["value_cache"],
        query_lens=data["query_lens"],
        kv_lens=data["kv_lens"],
        block_tables=data["block_tables"],
        scale=scale,
        sliding_window=sliding_window,
        soft_cap=soft_cap,
        sinks=None,
        k_block_scale=k_block_scale,
        v_block_scale=v_block_scale,
    )
    ref_output = ref_output.to(data["output"].device).to(torch.float32)
    data["output"] = data["output"].to(torch.float32)
    atol, rtol = 1.5e-1, 1.5e-1
    torch.testing.assert_close(
        data["output"], ref_output, atol=atol, rtol=rtol
    ), f"{torch.max(torch.abs(data['output'] - ref_output))}"
