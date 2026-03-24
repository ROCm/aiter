# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Optional
import pytest
import torch

from aiter.ops.triton.utils.types import e4m3_dtype
from aiter.ops.triton.attention.unified_attention import unified_attention
import aiter.ops.triton.utils._triton.arch_info as arch_info

from aiter.ops.triton.quant.sage_attention_quant_wrappers import sage_quant_v2

NUM_HEADS = [(4, 4), (8, 2), (16, 2)]
HEAD_SIZES = [128, 256]
BLOCK_SIZES = [16, 64, 48]

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
) -> torch.Tensor:
    num_seqs = len(query_lens)
    block_tables = block_tables.cpu().numpy()
    _, block_size, num_kv_heads, head_size = key_cache.shape
    head_size_v = value_cache.shape[-1]

    outputs: list[torch.Tensor] = []
    start_idx = 0
    for i in range(num_seqs):
        query_len = query_lens[i]
        kv_len = kv_lens[i]
        q = query[start_idx : start_idx + query_len]
        q *= scale

        num_kv_blocks = (kv_len + block_size - 1) // block_size
        block_indices = block_tables[i, :num_kv_blocks]

        k = key_cache[block_indices].view(-1, num_kv_heads, head_size)
        k = k[:kv_len]
        v = value_cache[block_indices].view(-1, num_kv_heads, head_size_v)
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
        
        out = torch.einsum("hqk,khc->qhc", attn, v)

        outputs.append(out)
        start_idx += query_len

    return torch.cat(outputs, dim=0)

# All the parametrization stays here on the "base" test
BASE_SEQ_LENS = [
    [(1, 1328), (5, 18), (129, 463)],
    [(1, 523), (1, 37), (1, 2011)],
]
QUANT_SCHEMES = [None, "fp8", "sage_mxfp4"]  # only used in base test


def unified_attn_unpack(d):
    return (
        d["query"],
        d["key_cache"],
        d["value_cache"],
        d["cu_query_lens"],
        d["kv_lens"],
        d["block_tables"],
        d["sinks"],
        d["output"],
        d["max_query_len"],
        d["max_kv_len"],
        d["window_size"],
        d["scale"],
        d["rope_size"],
        d["ref_kwargs"],
    )

@pytest.fixture
def unified_attn_inputs(
    seq_lens: list[tuple[int, int]],
    num_heads: tuple[int, int],
    head_size_qk: int,
    sliding_window: Optional[int],
    dtype: torch.dtype,
    block_size: int,
    soft_cap: Optional[float],
    num_blocks: int,
    rope_size: int,
):
    torch.cuda.empty_cache()
    torch.manual_seed(0)

    num_seqs = len(seq_lens)
    query_lens = [x[0] for x in seq_lens]
    kv_lens_list = [x[1] for x in seq_lens]
    num_query_heads = num_heads[0]
    num_kv_heads = num_heads[1]
    assert num_query_heads % num_kv_heads == 0

    max_query_len = max(query_lens)
    max_kv_len = max(kv_lens_list)
    window_size = (sliding_window - 1, 0) if sliding_window is not None else (-1, -1)
    scale = head_size_qk**-0.5

    query = torch.randn(
        sum(query_lens), num_query_heads, head_size_qk, dtype=dtype, device="cuda"
    )
    key_cache = torch.randn(
        num_blocks,
        block_size,
        num_kv_heads,
        head_size_qk,
        dtype=dtype,
        device="cuda",
    )

    if rope_size > 0:
        lora_rank = head_size_qk - rope_size
        value_cache = key_cache[:, :, :, :lora_rank]
    else:
        value_cache = torch.randn_like(key_cache)

    cu_query_lens = torch.tensor(
        [0] + query_lens, dtype=torch.int32, device="cuda"
    ).cumsum(dim=0, dtype=torch.int32)
    kv_lens = torch.tensor(kv_lens_list, dtype=torch.int32, device="cuda")

    max_num_blocks_per_seq = (max_kv_len + block_size - 1) // block_size
    block_tables = torch.randint(
        0,
        num_blocks,
        (num_seqs, max_num_blocks_per_seq),
        dtype=torch.int32,
        device="cuda",
    )

    sinks = torch.randn(num_query_heads, dtype=torch.bfloat16, device="cuda")

    output = torch.empty(
        sum(query_lens),
        num_query_heads,
        head_size_qk - rope_size,
        dtype=dtype,
        device="cuda",
    )

    common_kwargs = dict(
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
    )

    return dict(
        # raw tensors
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        cu_query_lens=cu_query_lens,
        kv_lens=kv_lens,
        block_tables=block_tables,
        sinks=sinks,
        output=output,
        # meta
        max_query_len=max_query_len,
        max_kv_len=max_kv_len,
        window_size=window_size,
        scale=scale,
        rope_size=rope_size,
        # for ref
        ref_kwargs=common_kwargs,
    )



@pytest.mark.parametrize("seq_lens", BASE_SEQ_LENS)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size_qk", HEAD_SIZES)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("sliding_window", [None, 256])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("soft_cap", [None, 10.0, 50.0])
@pytest.mark.parametrize("num_blocks", NUM_BLOCKS)
@pytest.mark.parametrize("rope_size", [0])
@torch.inference_mode()
def test_triton_unified_attn(
    unified_attn_inputs,
):
    (
        query,
        key_cache,
        value_cache,
        cu_query_lens,
        kv_lens,
        block_tables,
        sinks,
        output,
        max_query_len,
        max_kv_len,
        window_size,
        scale,
        rope_size,
        ref_kwargs,
    ) = unified_attn_unpack(unified_attn_inputs)

    unified_attention(
        q=query,
        k=key_cache,
        v=value_cache,
        out=output,
        cu_seqlens_q=cu_query_lens,
        seqused_k=kv_lens,
        max_seqlen_q=max_query_len,
        max_seqlen_k=max_kv_len,
        softmax_scale=scale,
        causal=True,
        window_size=window_size,
        block_table=block_tables,
        softcap=ref_kwargs["soft_cap"] if ref_kwargs["soft_cap"] is not None else 0,
        q_descale=None,
        k_descale=None,
        v_descale=None,
        sinks=sinks,
        sage_mxfp4=False,
    )

    ref_output = ref_paged_attn(**ref_kwargs)
    atol, rtol = 1.5e-2, 1e-2
    torch.testing.assert_close(output, ref_output, atol=atol, rtol=rtol)

@pytest.mark.parametrize("seq_lens", BASE_SEQ_LENS)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size_qk", HEAD_SIZES)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("sliding_window", [None, 256])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("soft_cap", [None, 10.0, 50.0])
@pytest.mark.parametrize("num_blocks", NUM_BLOCKS)
@pytest.mark.parametrize("rope_size", [64])  # MLA only
@torch.inference_mode()
def test_triton_unified_attn_mla(unified_attn_inputs):
    (
        query,
        key_cache,
        value_cache,
        cu_query_lens,
        kv_lens,
        block_tables,
        sinks,
        output,
        max_query_len,
        max_kv_len,
        window_size,
        scale,
        rope_size,
        ref_kwargs,
    ) = unified_attn_unpack(unified_attn_inputs)

    # same as normal, but implicitly exercising rope_size > 0 layout
    unified_attention(
        q=query,
        k=key_cache,
        v=value_cache,
        out=output,
        cu_seqlens_q=cu_query_lens,
        seqused_k=kv_lens,
        max_seqlen_q=max_query_len,
        max_seqlen_k=max_kv_len,
        softmax_scale=scale,
        causal=True,
        window_size=window_size,
        block_table=block_tables,
        softcap=ref_kwargs["soft_cap"] if ref_kwargs["soft_cap"] is not None else 0,
        q_descale=None,
        k_descale=None,
        v_descale=None,
        sinks=sinks,
        sage_mxfp4=False,
    )

    ref_output = ref_paged_attn(**ref_kwargs)
    atol, rtol = 1.5e-2, 1e-2
    torch.testing.assert_close(output, ref_output, atol=atol, rtol=rtol)


@pytest.mark.parametrize("seq_lens", BASE_SEQ_LENS)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size_qk", HEAD_SIZES)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("sliding_window", [None, 256])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("soft_cap", [None, 10.0, 50.0])
@pytest.mark.parametrize("num_blocks", NUM_BLOCKS)
@pytest.mark.parametrize("rope_size", [0, 64])
@torch.inference_mode()
def test_triton_unified_attn_fp8(unified_attn_inputs):
    (
        query,
        key_cache,
        value_cache,
        cu_query_lens,
        kv_lens,
        block_tables,
        sinks,
        output,
        max_query_len,
        max_kv_len,
        window_size,
        scale,
        rope_size,
        ref_kwargs,
    ) = unified_attn_unpack(unified_attn_inputs)

    FP8_TYPE = e4m3_dtype
    fp8_max = torch.finfo(FP8_TYPE).max

    q_abs_max = query.abs().amax().clamp(min=1e-9)
    q_descale = (q_abs_max / fp8_max).to(torch.float32).unsqueeze(0).cuda()
    q_fp8 = (query * (fp8_max / q_abs_max)).to(FP8_TYPE)

    k_abs_max = key_cache.abs().amax().clamp(min=1e-9)
    k_descale = (k_abs_max / fp8_max).to(torch.float32).unsqueeze(0).cuda()
    k_fp8 = (key_cache * (fp8_max / k_abs_max)).to(FP8_TYPE)

    v_abs_max = value_cache.abs().amax().clamp(min=1e-9)
    v_descale = (v_abs_max / fp8_max).to(torch.float32).unsqueeze(0).cuda()
    v_fp8 = (value_cache * (fp8_max / v_abs_max)).to(FP8_TYPE)

    unified_attention(
        q=q_fp8,
        k=k_fp8,
        v=v_fp8,
        out=output,
        cu_seqlens_q=cu_query_lens,
        seqused_k=kv_lens,
        max_seqlen_q=max_query_len,
        max_seqlen_k=max_kv_len,
        softmax_scale=scale,
        causal=True,
        window_size=window_size,
        block_table=block_tables,
        softcap=ref_kwargs["soft_cap"] if ref_kwargs["soft_cap"] is not None else 0,
        q_descale=q_descale,
        k_descale=k_descale,
        v_descale=v_descale,
        sinks=sinks,
        sage_mxfp4=False,
    )

    ref_output = ref_paged_attn(**ref_kwargs)
    atol, rtol = 1.5e-1, 1.5e-1
    torch.testing.assert_close(output, ref_output, atol=atol, rtol=rtol)


@pytest.mark.parametrize("seq_lens", BASE_SEQ_LENS)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size_qk", HEAD_SIZES)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("sliding_window", [None, 256])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("soft_cap", [None, 10.0, 50.0])
@pytest.mark.parametrize("num_blocks", NUM_BLOCKS)
@pytest.mark.parametrize("rope_size", [0])  # ROPE not supported
@torch.inference_mode()
def test_triton_unified_attn_mxfp4(unified_attn_inputs):
    if not arch_info.is_fp4_avail():
        pytest.skip("FP4 dot product is not supported on this GPU")

    (
        query,
        key_cache,
        value_cache,
        cu_query_lens,
        kv_lens,
        block_tables,
        sinks,
        output,
        max_query_len,
        max_kv_len,
        window_size,
        scale,
        rope_size,
        ref_kwargs,
    ) = unified_attn_unpack(unified_attn_inputs)

    (
        q_fp4,
        q_descale,
        k_fp4,
        k_descale,
        v_fp4,
        v_descale,
    ) = sage_quant_v2(
        query,
        key_cache,
        value_cache,
        hadamard_rotation=True,
        R=None,
        BLOCK_R=128,
        layout_k="cache",
        v_descale=None,
    )

    unified_attention(
        q=q_fp4,
        k=k_fp4,
        v=v_fp4,
        out=output,
        cu_seqlens_q=cu_query_lens,
        seqused_k=kv_lens,
        max_seqlen_q=max_query_len,
        max_seqlen_k=max_kv_len,
        softmax_scale=scale,
        causal=True,
        window_size=window_size,
        block_table=block_tables,
        softcap=ref_kwargs["soft_cap"] if ref_kwargs["soft_cap"] is not None else 0,
        q_descale=q_descale,
        k_descale=k_descale,
        v_descale=v_descale,
        sinks=sinks,
        sage_mxfp4=True,
    )

    ref_output = ref_paged_attn(**ref_kwargs)
    atol, rtol = 3.5e-1, 2.5e-1
    torch.testing.assert_close(output, ref_output, atol=atol, rtol=rtol)
    mae = (output - ref_output).abs().mean().item()
    assert mae < 0.1, f"MXFP4 mean absolute error too high: {mae}"
