# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Optional
import pytest
import torch

from aiter.ops.triton.utils.types import e4m3_dtype
from aiter.ops.triton.attention.unified_attention import unified_attention
import aiter.ops.triton.utils._triton.arch_info as arch_info

from aiter.ops.triton.quant.sage_attention_quant_wrappers import sage_quant_v2

NUM_HEADS = [(4, 4), (8, 2), (16, 1)]
HEAD_SIZES = [(128, 128), (256, 256), (192, 128)]
BLOCK_SIZES = [16, 64, 48]

DTYPES = [torch.float16, torch.bfloat16]
QDTYPES = [None, e4m3_dtype]
# one value large enough to test overflow in index calculation.
# one value small enough to test the schema op check
NUM_BLOCKS = [32768, 2048]
BASE_SEQ_LENS = [
    [(1, 1328), (5, 18), (129, 463)],
    [(1, 523), (1, 37), (1, 2011)],
]
QUANT_SCHEMES = [None, "fp8", "sage_mxfp4"]  # only used in base test


def ref_attn(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    query_lens: list[int],
    kv_lens: list[int],
    block_tables: torch.Tensor,  # (num_seqs, max_num_blocks_per_seq) if kv_layout == "cache" or cu_seqlens_k[:-1] if kv_layout == "thd"
    scale: float,
    sliding_window: Optional[int] = None,
    soft_cap: Optional[float] = None,
    sinks: Optional[torch.Tensor] = None,
    causal: bool = True,
    kv_layout: str = "cache",
) -> torch.Tensor:
    num_seqs = len(query_lens)

    if kv_layout not in ("cache", "thd"):
        raise ValueError(f"Invalid kv_layout: {kv_layout}")

    kv_is_thd = kv_layout == "thd"

    if kv_is_thd:
        _, num_kv_heads, head_size = key_cache.shape
    else:
        assert (
            block_tables is not None
        ), "block_tables must be provided for cache layout"
        block_tables = block_tables.cpu().numpy()
        _, block_size, num_kv_heads, head_size = key_cache.shape
    head_size_v = value_cache.shape[-1]

    outputs: list[torch.Tensor] = []
    start_idx = 0
    kv_start_idx = 0
    for i in range(num_seqs):
        query_len = query_lens[i]
        kv_len = kv_lens[i]
        q = query[start_idx : start_idx + query_len]
        q *= scale

        if kv_is_thd:
            kv_end_idx = kv_start_idx + kv_len
            token_indices = torch.arange(kv_start_idx, kv_end_idx, device=query.device)
            k = key_cache[token_indices].view(-1, num_kv_heads, head_size)
            v = value_cache[token_indices].view(-1, num_kv_heads, head_size_v)
        else:
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
        if causal:
            mask = torch.triu(empty_mask, diagonal=kv_len - query_len + 1).bool()
        else:  # no causal masking
            mask = torch.zeros_like(empty_mask).bool()
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


def make_unified_attn_inputs(
    seq_lens: list[tuple[int, int]],
    num_heads: tuple[int, int],
    head_size_qk: int,
    head_size_v: int,
    dtype: torch.dtype,
    kv_layout: str = "cache",
    block_size: int = None,
    num_blocks: int = None,
):
    torch.cuda.empty_cache()
    torch.manual_seed(0)

    num_seqs = len(seq_lens)
    query_lens = [x[0] for x in seq_lens]
    kv_lens_list = [x[1] for x in seq_lens]
    num_query_heads, num_kv_heads = num_heads
    assert num_query_heads % num_kv_heads == 0

    max_query_len = max(query_lens)
    max_kv_len = max(kv_lens_list)

    scale = head_size_qk**-0.5

    query = torch.randn(
        sum(query_lens), num_query_heads, head_size_qk, dtype=dtype, device="cuda"
    )

    if kv_layout == "thd":
        key_cache = torch.randn(
            sum(kv_lens_list),
            num_kv_heads,
            head_size_qk,
            dtype=dtype,
            device="cuda",
        )
        value_cache = torch.randn(
            sum(kv_lens_list),
            num_kv_heads,
            head_size_v,
            dtype=dtype,
            device="cuda",
        )
    elif kv_layout == "cache":
        key_cache = torch.randn(
            num_blocks,
            block_size,
            num_kv_heads,
            head_size_qk,
            dtype=dtype,
            device="cuda",
        )
        value_cache = torch.randn(
            num_blocks,
            block_size,
            num_kv_heads,
            head_size_v,
            dtype=dtype,
            device="cuda",
        )
    else:
        raise ValueError(f"Invalid kv_layout: {kv_layout}")

    cu_query_lens = torch.tensor(
        [0] + query_lens, dtype=torch.int32, device="cuda"
    ).cumsum(dim=0, dtype=torch.int32)
    cu_key_lens = torch.tensor(
        [0] + kv_lens_list, dtype=torch.int32, device="cuda"
    ).cumsum(dim=0, dtype=torch.int32)
    kv_lens = torch.tensor(kv_lens_list, dtype=torch.int32, device="cuda")
    query_lens = torch.tensor(query_lens, dtype=torch.int32, device="cuda")

    max_num_blocks_per_seq = (max_kv_len + block_size - 1) // block_size
    block_tables = torch.randint(
        0,
        num_blocks,
        (num_seqs, max_num_blocks_per_seq),
        dtype=torch.int32,
        device="cuda",
    )

    output = torch.empty(
        sum(query_lens),
        num_query_heads,
        head_size_v,
        dtype=dtype,
        device="cuda",
    )

    return (
        query,
        key_cache,
        value_cache,
        cu_query_lens,
        cu_key_lens,
        query_lens,
        kv_lens,
        block_tables,
        output,
        max_query_len,
        max_kv_len,
        scale,
    )


@pytest.mark.parametrize("seq_lens", BASE_SEQ_LENS)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_sizes", HEAD_SIZES)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("sliding_window", [None, 256])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("soft_cap", [None, 10.0, 50.0])
@pytest.mark.parametrize("num_blocks", NUM_BLOCKS)
@pytest.mark.parametrize("kv_layout", ["cache"])
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("quant_scheme", [None, "fp8", "sage_mxfp4"])
@torch.inference_mode()
def test_triton_unified_attn(
    seq_lens,
    num_heads,
    head_sizes,
    sliding_window,
    dtype,
    block_size,
    soft_cap,
    num_blocks,
    kv_layout,
    causal,
    quant_scheme,
):
    if quant_scheme == "sage_mxfp4":
        if not arch_info.is_fp4_avail():
            pytest.skip("FP4 dot product is not supported on this GPU")

    head_size_qk = head_sizes[0]
    head_size_v = head_sizes[1]

    (
        query,
        key_cache,
        value_cache,
        cu_query_lens,
        cu_key_lens,
        query_lens,
        kv_lens,
        block_tables,
        output,
        max_query_len,
        max_kv_len,
        scale,
    ) = make_unified_attn_inputs(
        seq_lens=seq_lens,
        num_heads=num_heads,
        head_size_qk=head_size_qk,
        head_size_v=head_size_v,
        dtype=dtype,
        block_size=block_size,
        num_blocks=num_blocks,
        kv_layout=kv_layout,
    )

    num_query_heads = num_heads[0]
    window_size = (sliding_window - 1, 0) if sliding_window is not None else (-1, -1)
    sinks = torch.randn(num_query_heads, dtype=torch.bfloat16, device="cuda")

    q_input, k_input, v_input = query, key_cache, value_cache
    q_descale = k_descale = v_descale = None

    if quant_scheme == "fp8":
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

        q_input, k_input, v_input = q_fp8, k_fp8, v_fp8
    elif quant_scheme == "sage_mxfp4":
        if not arch_info.is_fp4_avail():
            pytest.skip("FP4 dot product is not supported on this GPU")

        q_input, q_descale, k_input, k_descale, v_input, v_descale = sage_quant_v2(
            query,
            key_cache,
            value_cache,
            hadamard_rotation=True,
            R=None,
            BLOCK_R=32,
            layout_k=kv_layout,
            v_descale=None,
        )

    unified_attention(
        q=q_input,
        k=k_input,
        v=v_input,
        out=output,
        cu_seqlens_q=cu_query_lens,
        seqused_k=kv_lens,
        max_seqlen_q=max_query_len,
        max_seqlen_k=max_kv_len,
        softmax_scale=scale,
        causal=causal,
        window_size=window_size,
        block_table=block_tables,
        softcap=soft_cap if soft_cap is not None else 0,
        q_descale=q_descale,
        k_descale=k_descale,
        v_descale=v_descale,
        sinks=sinks,
        sage_mxfp4=quant_scheme == "sage_mxfp4",
        kv_layout=kv_layout,
        cu_seqlens_k=cu_key_lens,
    )

    ref_output = ref_attn(
        query,
        key_cache,
        value_cache,
        query_lens,
        kv_lens,
        block_tables,
        scale,
        sliding_window,
        soft_cap,
        sinks,
        causal=causal,
        kv_layout=kv_layout,
    )

    if quant_scheme == "fp8":
        atol, rtol = 1.5e-1, 1.5e-1
    elif quant_scheme == "sage_mxfp4":
        atol, rtol = 3.5e-1, 2.5e-1
        # mae = (output - ref_output).abs().mean().item()
        # assert mae < 0.1, f"MXFP4 mean absolute error too high: {mae}"
    else:
        atol, rtol = 1.5e-2, 1e-2
    torch.testing.assert_close(output, ref_output, atol=atol, rtol=rtol)
