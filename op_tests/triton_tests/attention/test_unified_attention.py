# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Optional

import pytest
import torch

from aiter.ops.triton.attention.unified_attention import SAGE_VERSION, unified_attention, get_config
from aiter.ops.triton.utils.types import e4m3_dtype
import aiter.ops.triton.utils._triton.arch_info as arch_info

NUM_HEADS = [(4, 4), (8, 2), (16, 2)]
HEAD_SIZES = [128, 256]
BLOCK_SIZES = [512]

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
    kv_layout: str = "cache", # "cache" i.e (num_blocks, block_size, num_kv_heads, head_size) as default. "bshd", "bhsd", "thd"
) -> torch.Tensor:
    num_seqs = len(query_lens)
    block_tables = block_tables.cpu().numpy()
    if kv_layout == "thd":
        _, num_kv_heads, head_size = key_cache.shape
        block_size = kv_lens.max().item()
    else:
        # we are only checking correctness, so we can permute the cache to the most convenient layout
        if kv_layout == "bhsd":
            key_cache = key_cache.permute(0, 2, 1, 3)
            value_cache = value_cache.permute(0, 2, 1, 3)
        _, block_size, num_kv_heads, head_size = key_cache.shape

    outputs: list[torch.Tensor] = []
    start_idx = 0
    for i in range(num_seqs):
        query_len = query_lens[i]
        kv_len = kv_lens[i]
        q = query[start_idx : start_idx + query_len]
        q *= scale

        num_kv_blocks = (kv_len + block_size - 1) // block_size
        block_indices = block_tables[i, :num_kv_blocks]

        if kv_layout == "thd":
            start_token_idx = block_indices[0]
            end_token_idx = start_token_idx + kv_len
            k = key_cache[start_token_idx:end_token_idx].view(-1, num_kv_heads, head_size)
            v = value_cache[start_token_idx:end_token_idx].view(-1, num_kv_heads, head_size)
        else:
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

    return torch.cat(outputs, dim=0)



def input_helper(BATCH, HQ, HK, N_CTX_Q, N_CTX_K, D_HEAD, DECODE_P, dtype, causal, equal_seqlens=False, softcap=None, kv_layout="cache", num_blocks=0, block_size=512):

    varlen = not equal_seqlens

    if not varlen:
        seqlens_q = torch.tensor(
            [N_CTX_Q for _ in range(BATCH)], dtype=torch.int32, device="cuda"
        )
        seqlens_k = torch.tensor(
            [N_CTX_K for _ in range(BATCH)], dtype=torch.int32, device="cuda"
        )
    else:
        seqlens_q = torch.randint(
            1, N_CTX_Q + 1, (BATCH,), dtype=torch.int32, device="cuda"
        )
        seqlens_k = torch.randint(
            N_CTX_Q, N_CTX_K + 1, (BATCH,), dtype=torch.int32, device="cuda"
        )

    # turn DECODE_P of the samples to decode samples (seqlen_q == 1)
    if DECODE_P > 0.0:
        num_decode = int(round(DECODE_P * BATCH))
        if num_decode > 0:
            decode_idx = torch.randperm(BATCH, device=seqlens_q.device)[:num_decode]
            seqlens_q[decode_idx] = 1

    if causal:
        if (seqlens_k < seqlens_q).any():
            print(
                f"Warning: clamping seqlens_k to be >= seqlens_q for config "
                f"(BATCH={BATCH}, HQ={HQ}, HK={HK}, N_CTX_Q={N_CTX_Q}, N_CTX_K={N_CTX_K})"
            )
        seqlens_k = torch.maximum(seqlens_k, seqlens_q)

    cu_seqlens_q = torch.zeros(len(seqlens_q) + 1, dtype=torch.int32, device="cuda")
    cu_seqlens_q[1:] = seqlens_q.cumsum(dim=0, dtype=torch.int32)
    cu_seqlens_k = torch.zeros(len(seqlens_k) + 1, dtype=torch.int32, device="cuda")
    cu_seqlens_k[1:] = seqlens_k.cumsum(dim=0, dtype=torch.int32)
    
    num_seqs = BATCH
    num_query_heads = HQ
    num_kv_heads = HK
    head_size = D_HEAD
    assert num_query_heads % num_kv_heads == 0
    max_query_len = max(seqlens_q).item()
    max_kv_len = max(seqlens_k).item()
    soft_cap = softcap

    query = torch.randn(
        sum(seqlens_q), num_query_heads, head_size, dtype=dtype, device="cuda"
    )

    block_tables = None
    block_size = None

    if kv_layout == "cache":
        block_size = block_size if block_size else 512
        max_num_blocks_per_seq = (max_kv_len + block_size - 1) // block_size
        min_required_blocks = BATCH * max_num_blocks_per_seq
        num_blocks = (
            num_blocks if num_blocks else max(min_required_blocks, 2048)
        )
        num_blocks = max(num_blocks, min_required_blocks)
        key_cache = torch.randn(
            num_blocks, block_size, num_kv_heads, head_size, dtype=dtype, device="cuda"
        )
        value_cache = torch.randn_like(key_cache)
        block_tables = torch.randint(
            0,
            num_blocks,
            (num_seqs, max_num_blocks_per_seq),
            dtype=torch.int32,
            device="cuda",
        )
    elif kv_layout in ("bshd", "bhsd"):
        # assert equal_seqlens, "equal_seqlens assumed if kv_cache_layout is bshd"
        block_size = N_CTX_K
        num_blocks = BATCH
        block_tables = torch.arange(num_blocks, dtype=torch.int32, device="cuda").unsqueeze(1)
        key_cache = torch.randn(
            BATCH, N_CTX_K, num_kv_heads, head_size, dtype=dtype, device="cuda"
        )
        if kv_layout == "bhsd":
            key_cache = key_cache.transpose(1, 2).contiguous()  # to bhsd

        value_cache = torch.randn_like(key_cache)
    else:  # thd
        block_size = max_kv_len
        block_tables = cu_seqlens_k.unsqueeze(1)
        key_cache = torch.randn(
            sum(seqlens_k), num_kv_heads, head_size, dtype=dtype, device="cuda"
        )
        value_cache = torch.randn_like(key_cache)

   

    return (
        query,
        key_cache,
        value_cache,
        cu_seqlens_q,
        cu_seqlens_k,
        seqlens_q,
        seqlens_k,
        max_query_len,
        max_kv_len,
        block_tables,
        block_size,
        num_query_heads,
        num_kv_heads,
        head_size,
        soft_cap,
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
@pytest.mark.parametrize("quant_scheme", ["v1", "v2", None])
@pytest.mark.parametrize("kv_layout", ["cache", "bshd", "bhsd", "thd"])
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
    quant_scheme: Optional[str],
    kv_layout: str,
) -> None:
    # if q_dtype is not None and q_dtype.itemsize < 2 and block_size < 32:
    #     pytest.skip("block size must be at least 32 for fp8")
    is_reduced_precision = quant_scheme in ("v1", "v2", "fp8")
    # TODO: better block size checking
    if is_reduced_precision and block_size < 32:
        pytest.skip("block size must be at least 32 for fp8")
    if quant_scheme not in ("v1", "v2") and q_dtype is not None:
        pytest.skip("SAGE quant and fp8 q_dtype are mutually exclusive")
    if quant_scheme == "v2" and not arch_info.is_fp4_avail():
        pytest.skip("FP4 dot product is not supported on this GPU")

    torch.cuda.empty_cache()
    torch.manual_seed(0)

    num_seqs = len(seq_lens)
    query_lens = torch.tensor([x[0] for x in seq_lens], dtype=torch.int32)
    kv_lens = torch.tensor([x[1] for x in seq_lens], dtype=torch.int32)

    (
        query,
        key_cache,
        value_cache,
        cu_query_lens,
        cu_key_lens,
        seqlens_q,
        seqlens_k,
        max_query_len,
        max_kv_len,
        block_tables,
        block_size,
        num_query_heads,
        num_kv_heads,
        head_size,
        soft_cap,
    ) = input_helper(
        BATCH=num_seqs,
        HQ=num_heads[0],
        HK=num_heads[1],
        N_CTX_Q=max(query_lens).item(),
        N_CTX_K=max(kv_lens).item(),
        D_HEAD=head_size,
        DECODE_P=0.0,
        dtype=dtype,
        causal=True,
        block_size=block_size,
        softcap=soft_cap,
        kv_layout=kv_layout,
    )

    window_size = (
            (sliding_window - 1, 0)
            if sliding_window is not None
            else (-1, -1)
        )
    
    scale = query.shape[-1]**-0.5
    sinks = torch.randn(num_query_heads, dtype=torch.bfloat16, device="cuda")
    output = torch.empty_like(query)

    maybe_quantized_query = query
    maybe_quantized_key_cache = key_cache
    maybe_quantized_value_cache = value_cache
    q_descale = None
    k_descale = None
    v_descale = None
    
    sagev1 = quant_scheme=="v1"
    sagev2 = quant_scheme=="v2"
    fp8_full = quant_scheme == "fp8"

    if sagev1 or sagev2:
        num_queries_per_kv = num_query_heads // num_kv_heads
        config = get_config(
            query.shape[0], len(cu_query_lens),
            num_queries_per_kv, num_kv_heads, head_size,
            window_size, max_query_len, max_kv_len,
            block_size, query.element_size()
        )
        BLOCK_M = config["BLOCK_M"]
        BLOCK_N = config["TILE_SIZE"]
        if sagev1:
            from aiter.ops.triton.quant.sage_attention_quant_wrappers import sage_quant_v1
            maybe_quantized_query, q_descale, maybe_quantized_key_cache, k_descale, maybe_quantized_value_cache, v_descale = sage_quant_v1(
                query, key_cache, value_cache, BLOCK_M, BLOCK_N,
                layout_k=kv_layout,
                v_descale=None,
                cu_seqlens_q=cu_query_lens, cu_seqlens_k=cu_key_lens,
                block_table=block_tables, block_size=block_size
            )
            
        else:
            from aiter.ops.triton.quant.sage_attention_quant_wrappers import sage_quant_v2
            maybe_quantized_query, q_descale, maybe_quantized_key_cache, k_descale, maybe_quantized_value_cache, v_descale = sage_quant_v2(
                query, key_cache, value_cache, BLOCK_M, BLOCK_N,
                hadamard_rotation=True, R=None, BLOCK_R=128,
                layout_k=kv_layout, v_descale=None
            )
    elif fp8_full:
            from aiter.ops.triton.utils.types import e4m3_dtype

            FP8_TYPE = e4m3_dtype
            fp8_max = torch.finfo(FP8_TYPE).max

            q_abs_max = query.abs().amax().clamp(min=1e-9)
            q_descale = (q_abs_max / fp8_max).to(torch.float32).unsqueeze(0).cuda()
            maybe_quantized_query = (query * (fp8_max / q_abs_max)).to(FP8_TYPE)

            k_abs_max = key_cache.abs().amax().clamp(min=1e-9)
            k_descale = (k_abs_max / fp8_max).to(torch.float32).unsqueeze(0).cuda()
            maybe_quantized_key_cache = (key_cache * (fp8_max / k_abs_max)).to(FP8_TYPE)

            v_abs_max = value_cache.abs().amax().clamp(min=1e-9)
            v_descale = (v_abs_max / fp8_max).to(torch.float32).unsqueeze(0).cuda()
            maybe_quantized_value_cache = (value_cache * (fp8_max / v_abs_max)).to(FP8_TYPE)
    else:
        q_descale, k_descale, v_descale = None, None, None

    if sagev1:
        sage_version = SAGE_VERSION.SAGE
    elif sagev2:
        sage_version = SAGE_VERSION.SAGE_MXFP4
    else:
        sage_version = None


    unified_attention(
        q=maybe_quantized_query,
        k=maybe_quantized_key_cache,
        v=maybe_quantized_value_cache,
        out=output,
        cu_seqlens_q=cu_query_lens,
        cu_seqlens_k=cu_key_lens,
        seqused_k=seqlens_k,
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
        sage_version=sage_version,
        kv_layout=kv_layout
    )

    ref_output = ref_paged_attn(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        query_lens=seqlens_q,
        kv_lens=seqlens_k,
        block_tables=block_tables,
        scale=scale,
        sliding_window=sliding_window,
        soft_cap=soft_cap,
        sinks=sinks,
        kv_layout=kv_layout
    )

    atol, rtol = 1.5e-2, 1e-2

    if is_reduced_precision:
        atol, rtol = 1.5e-1, 1.5e-1
    torch.testing.assert_close(
        output, ref_output, atol=atol, rtol=rtol
    ), f"{torch.max(torch.abs(output - ref_output))}"



if __name__ == "__main__":
    # Pick one concrete test case
    seq_lens = [(1, 1328), (5, 18), (129, 463)]
    num_heads = (16, 8)          # example from NUM_HEADS
    head_size = 128              # example from HEAD_SIZES
    block_size = 128             # example from BLOCK_SIZES
    sliding_window = None        # or 256
    dtype = torch.bfloat16       # example from DTYPES
    soft_cap = None              # or 10.0 / 50.0
    num_blocks = 32768              # example from NUM_BLOCKS
    q_dtype = None               # or torch.float8_e4m3fn, etc.
    quant_scheme = "v1"

    test_triton_unified_attn(
        seq_lens=seq_lens,
        num_heads=num_heads,
        head_size=head_size,
        sliding_window=sliding_window,
        dtype=dtype,
        block_size=block_size,
        soft_cap=soft_cap,
        num_blocks=num_blocks,
        q_dtype=q_dtype,
        quant_scheme=quant_scheme,
        kv_layout="cache"
    )