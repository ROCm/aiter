# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import functools
import random
import sys
from types import SimpleNamespace

import pytest
import torch
import triton

from aiter.ops.triton.attention.unified_attention import (
    _is_gluon_available,
    _length_aware_split,
    is_2d_gluon_available,
    unified_attention,
)
from aiter.ops.triton.utils._triton import arch_info
from aiter.ops.triton.utils.shuffle import shuffle_scale_batched, shuffle_weight
from aiter.ops.triton.utils.types import e4m3_dtype
from aiter.ops.triton.utils.unified_attention_utils import get_unified_attention_config
from aiter.test_common import checkAllclose
from op_tests.triton_tests.quant.test_quant_mxfp4 import (
    torch_dynamic_mxfp4_quant,
)

DEVICE_ARCH = arch_info.get_arch()
IS_DEVICE_ARCH_GFX12 = DEVICE_ARCH in ("gfx1250",)


def shuffle_kv_cache(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
):
    """
    Shuffle key and value cache layout for optimized memory access.

        layout: (num_lanes, num_elements_per_thread)
            gfx1250: (16, 8) for BF16 and FP8.
            gfx950: (16, 8) for BF16 and (16, 16) for FP8.

        WMMA/MFMA instruction shape:
            BF16: 16x16x32
            FP8: 16x16x64
    """
    dtype = key_cache.dtype
    assert value_cache.dtype == dtype
    assert dtype in (torch.bfloat16, e4m3_dtype)

    num_blocks, block_size, num_kv_heads, head_size = key_cache.shape
    num_blocks_v, block_size_v, num_kv_heads_v, head_size_v = value_cache.shape
    assert block_size >= 16
    assert num_blocks == num_blocks_v
    assert num_kv_heads == num_kv_heads_v
    assert head_size == head_size_v
    assert block_size == block_size_v

    k_width = 16 // key_cache.element_size()
    key_cache_shuffled = key_cache.view(
        -1, block_size, num_kv_heads, head_size
    ).permute(0, 2, 3, 1)
    key_cache_shuffled = key_cache_shuffled.view(
        -1,
        num_kv_heads,
        head_size // k_width,
        k_width,
        block_size,
    )
    key_cache_shuffled = key_cache_shuffled.permute(0, 1, 2, 4, 3).contiguous()

    value_cache_shuffled = value_cache.view(
        -1, block_size, num_kv_heads, head_size
    ).permute(0, 2, 1, 3)
    value_cache_shuffled = value_cache_shuffled.view(
        -1,
        num_kv_heads,
        block_size // k_width,
        k_width,
        head_size,
    )
    value_cache_shuffled = value_cache_shuffled.permute(0, 1, 2, 4, 3).contiguous()

    return key_cache_shuffled, value_cache_shuffled


def dynamic_nvfp4_quant_kv_cache(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
):
    dtype = key_cache.dtype
    assert value_cache.dtype == dtype
    assert dtype == torch.bfloat16

    num_blocks, block_size, num_kv_heads, head_size = key_cache.shape
    num_blocks_v, block_size_v, num_kv_heads_v, head_size_v = value_cache.shape
    assert block_size >= 128
    assert num_blocks == num_blocks_v
    assert num_kv_heads == num_kv_heads_v
    assert head_size == head_size_v
    assert block_size == block_size_v

    key_cache_shuffled = key_cache.view(
        -1, block_size, num_kv_heads, head_size
    ).permute(0, 2, 1, 3)
    value_cache_shuffled = value_cache.view(
        -1, block_size, num_kv_heads, head_size
    ).permute(0, 2, 1, 3)

    quant_head_size = head_size // 2
    scale_width = head_size // 16

    def quant_and_shuffle(key_or_value_cache):
        cache_shuffled, cache_shuffled_scale = torch_dynamic_mxfp4_quant(
            key_or_value_cache, is_nvfp4=True
        )
        cache_shuffled_scale = cache_shuffled_scale.view(
            -1, num_kv_heads, block_size, scale_width
        )
        cache_shuffled = shuffle_weight(cache_shuffled, arch="gfx950").view(
            -1, num_kv_heads, block_size * quant_head_size
        )
        cache_shuffled_scale = shuffle_scale_batched(cache_shuffled_scale).view(
            -1, num_kv_heads, block_size * scale_width
        )
        cache_shuffled = torch.cat(
            [
                cache_shuffled.view(torch.uint8),
                cache_shuffled_scale.view(torch.uint8),
            ],
            dim=-1,
        ).contiguous()
        cache_shuffled = cache_shuffled.view(
            -1, num_kv_heads, block_size, quant_head_size + scale_width
        )
        return cache_shuffled

    key_cache_quant_and_shuffled = quant_and_shuffle(key_cache_shuffled)
    value_cache_quant_and_shuffled = quant_and_shuffle(value_cache_shuffled)

    return key_cache_quant_and_shuffled, value_cache_quant_and_shuffled


def uniform_random(shape, start=0, end=1, dtype=torch.float32, device="cuda"):
    return (end - start) * torch.rand(shape, dtype=dtype, device=device) + start


def generate_data(
    seq_lens,
    num_blocks=32768,
    block_size=32,
    head_size=64,
    v_head_size=None,
    num_heads=(16, 2),
    sliding_window=None,
    q_dtype=torch.bfloat16,
    kv_dtype=torch.bfloat16,
    out_dtype=torch.bfloat16,
    shuffled_kv_cache=False,
    use_q_descale=None,
    use_kv_descale=None,
    use_out_scale=False,
    device="cuda",
):
    torch.manual_seed(0)
    num_seqs = len(seq_lens)
    query_lens = [x[0] for x in seq_lens]
    kv_lens = [x[1] for x in seq_lens]
    num_query_heads = num_heads[0]
    num_kv_heads = num_heads[1]
    assert num_query_heads % num_kv_heads == 0
    if v_head_size is None:
        v_head_size = head_size
    max_query_len = max(query_lens)
    max_kv_len = max(kv_lens)
    if sliding_window is not None and sliding_window > 0:
        window_size = (sliding_window - 1, 0)
    else:
        window_size = (-1, -1)
    scale = head_size**-0.5

    # Descales default to "on for any non-bf16 input" unless the caller overrides.
    if use_q_descale is None:
        use_q_descale = q_dtype != torch.bfloat16
    if use_kv_descale is None:
        use_kv_descale = kv_dtype != torch.bfloat16

    # ---- query ----
    query = torch.randn(
        sum(query_lens), num_query_heads, head_size, dtype=torch.float32, device=device
    )
    query_scales = None
    if q_dtype == torch.uint8:
        query = query / 10
        maybe_quant_query = query.view(-1, head_size)
        maybe_quant_query, query_scales = torch_dynamic_mxfp4_quant(
            maybe_quant_query, is_nvfp4=True
        )
        maybe_quant_query = maybe_quant_query.view(-1, num_query_heads, head_size // 2)
        query_scales = query_scales.view(-1, num_query_heads, head_size // 16)
        query = query.to(e4m3_dtype)
    else:
        query = query.to(q_dtype)
        maybe_quant_query = query

    # ---- kv cache ----
    key_cache = torch.randn(
        num_blocks,
        block_size,
        num_kv_heads,
        head_size,
        dtype=torch.float32,
        device=device,
    )
    value_cache = torch.randn(
        num_blocks,
        block_size,
        num_kv_heads,
        v_head_size,
        dtype=torch.float32,
        device=device,
    )
    if kv_dtype == torch.uint8:
        key_cache_orig = key_cache.to(e4m3_dtype)
        value_cache_orig = value_cache.to(e4m3_dtype)
        key_cache, value_cache = dynamic_nvfp4_quant_kv_cache(
            key_cache.to(torch.bfloat16), value_cache.to(torch.bfloat16)
        )
    else:
        key_cache_orig = key_cache.to(kv_dtype)
        value_cache_orig = value_cache.to(kv_dtype)
        if shuffled_kv_cache:
            key_cache, value_cache = shuffle_kv_cache(key_cache_orig, value_cache_orig)
        else:
            key_cache, value_cache = key_cache_orig, value_cache_orig

    cu_query_lens = torch.tensor(
        [0] + query_lens, dtype=torch.int32, device=device
    ).cumsum(dim=0, dtype=torch.int32)
    kv_lens = torch.tensor(kv_lens, dtype=torch.int32, device=device)

    max_num_blocks_per_seq = (max_kv_len + block_size - 1) // block_size
    max_num_blocks_per_seq = (
        min(max_num_blocks_per_seq * num_seqs, num_blocks) // num_seqs
    )
    total_ind_count = num_seqs * max_num_blocks_per_seq
    values = torch.arange(0, total_ind_count, dtype=torch.int, device=device)
    values = values[torch.randperm(total_ind_count, device=device)]
    block_tables = values.view(num_seqs, max_num_blocks_per_seq).contiguous()

    sinks = torch.randn(num_query_heads, dtype=torch.float32, device=device)

    output = torch.empty(
        sum(query_lens), num_query_heads, v_head_size, dtype=out_dtype, device=device
    )

    # ---- descales / output scale ----
    q_descale = None
    k_descale = None
    v_descale = None
    output_scale = None
    if use_q_descale:
        q_descale = uniform_random(
            1, start=1e-4, end=1.0, dtype=torch.float32, device=device
        )
    if use_kv_descale:
        k_descale = uniform_random(
            1, start=1e-4, end=1.0, dtype=torch.float32, device=device
        )
        v_descale = uniform_random(
            1, start=1e-4, end=1.0, dtype=torch.float32, device=device
        )
    if use_out_scale:
        output_scale = 1.0 / uniform_random(
            1, start=1e-4, end=1.0, dtype=torch.float32, device=device
        )

    return (
        query,
        key_cache_orig,
        value_cache_orig,
        key_cache,
        value_cache,
        sinks,
        output,
        cu_query_lens,
        kv_lens,
        max_query_len,
        max_kv_len,
        scale,
        window_size,
        block_tables,
        maybe_quant_query,
        query_scales,
        q_descale,
        k_descale,
        v_descale,
        output_scale,
    )


def ref_paged_attn(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    query_lens: list[int],
    kv_lens: list[int],
    block_tables: torch.Tensor,
    scale: float,
    out_dtype: torch.dtype,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    sinks: torch.Tensor | None = None,
    q_descale: torch.Tensor | None = None,
    k_descale: torch.Tensor | None = None,
    v_descale: torch.Tensor | None = None,
    output_scale: torch.Tensor | None = None,
    causal: int = 1,
    qq_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    num_seqs = len(query_lens)
    _, block_size, num_kv_heads, head_size = key_cache.shape
    v_head_size = value_cache.shape[-1]
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
        v = value_cache[block_indices].view(-1, num_kv_heads, v_head_size)
        v = v[:kv_len]

        if q.shape[1] != k.shape[1]:
            k = torch.repeat_interleave(k, q.shape[1] // k.shape[1], dim=1)
            v = torch.repeat_interleave(v, q.shape[1] // v.shape[1], dim=1)
        attn = torch.einsum("qhd,khd->hqk", q, k).float()
        empty_mask = torch.ones(query_len, kv_len, device=q.device, dtype=torch.float32)
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
        if qq_bias is not None:
            attn[:, :, kv_len - query_len :] += qq_bias[:query_len, :query_len]
        if causal:
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

    return out.to(out_dtype)


def _assert_request_rel_l2(output, ref_output, seq_lens, bound=2e-2):
    """Bound each request's relative L2 error against the reference.

    Long-KV outputs are small, so a fixed atol would pass all-zero rows.
    gfx950 measures ~0.0025; one dropped 64-token KV tile gives ~0.06.
    A zero-length request (a padded slot) must give zero rows.
    """
    start = 0
    for q_len, kv_len in seq_lens:
        out_rows = output[start : start + q_len].float()
        ref_rows = ref_output[start : start + q_len].float()
        start += q_len
        if kv_len == 0:
            assert torch.equal(out_rows, torch.zeros_like(out_rows)), "kv_len 0"
            continue
        rel_err = ((out_rows - ref_rows).norm() / ref_rows.norm()).item()
        assert rel_err < bound, f"kv_len {kv_len}: relative L2 error {rel_err:.4f}"


@pytest.mark.parametrize(
    "seq_lens",
    [
        [(1, 1328)],
        [(1, 8192)] * 32,
        [(1, 523), (1, 37), (1, 2011)],
        [(1, 1328), (1, 523), (1, 37), (1, 2011), (1, 8192)],
        [(2, 8704), (7, 8704)],
        [(8, 8704)] * 32,
    ],
)
@pytest.mark.parametrize("num_heads", [(64, 8), (8, 1)])
@pytest.mark.parametrize("head_size", [64])
@pytest.mark.parametrize("sliding_window", [None, 2048])
@pytest.mark.parametrize(
    "q_dtype, kv_dtype, o_dtype, block_size, use_out_scale",
    [
        (torch.bfloat16, torch.bfloat16, torch.bfloat16, 64, False),
        (torch.bfloat16, e4m3_dtype, torch.bfloat16, 128, False),
        (e4m3_dtype, e4m3_dtype, torch.bfloat16, 128, False),
        (e4m3_dtype, e4m3_dtype, e4m3_dtype, 128, True),
        (e4m3_dtype, torch.uint8, torch.bfloat16, 128, False),
        (torch.uint8, torch.uint8, torch.bfloat16, 128, False),
    ],
)
@pytest.mark.parametrize("soft_cap", [None])
@pytest.mark.parametrize("num_blocks", [32768])
@pytest.mark.parametrize("shuffled_kv_cache", [True, False])
@torch.inference_mode()
def test_triton_unified_attn_3d(
    seq_lens: list[tuple[int, int]],
    num_heads: tuple[int, int],
    head_size: int,
    sliding_window: int | None,
    block_size: int,
    soft_cap: float | None,
    num_blocks: int,
    q_dtype: torch.dtype,
    kv_dtype: torch.dtype,
    o_dtype: torch.dtype,
    shuffled_kv_cache: bool,
    use_out_scale: bool,
) -> None:
    torch.cuda.empty_cache()

    if DEVICE_ARCH not in (
        "gfx950",
        "gfx1250",
    ):
        # gfx1250 -> Gluon
        # gfx950 -> Gluon
        pytest.skip(f"skip {DEVICE_ARCH}")

    if kv_dtype == torch.uint8:
        if DEVICE_ARCH not in ("gfx1250",):
            pytest.skip(f"NVFP4 KV cache requires {DEVICE_ARCH}")
        if not shuffled_kv_cache:
            pytest.skip("NVFP4 KV cache requires shuffled KV cache")

    if shuffled_kv_cache:
        if q_dtype == e4m3_dtype and kv_dtype == e4m3_dtype and block_size < 32:
            pytest.skip(
                "For A8W8 Unified Attention with pre-shuffled KV cache, only block_size >= 32 is supported"
            )

        num_stage_assume = 2
        kv_cache_shared_mem_size = (
            2 * num_stage_assume * block_size * head_size * kv_dtype.itemsize
        )
        LDS_limit = 327680 if IS_DEVICE_ARCH_GFX12 else 262144
        if kv_cache_shared_mem_size > LDS_limit:
            pytest.skip(
                f"Skipping test for KV cache LDS required memory = {kv_cache_shared_mem_size / 1024} kB > 320 kB"
            )

    # TODO: Uncomment after pytorch adds support for manual_seed
    torch.manual_seed(0)
    query_lens = [x[0] for x in seq_lens]

    (
        query,
        key_cache,
        value_cache,
        maybe_shuffled_key_cache,
        maybe_shuffled_value_cache,
        sinks,
        output,
        cu_query_lens,
        kv_lens,
        max_query_len,
        max_kv_len,
        scale,
        window_size,
        block_tables,
        maybe_quant_query,
        query_scales,
        q_descale,
        k_descale,
        v_descale,
        output_scale,
    ) = generate_data(
        seq_lens=seq_lens,
        num_blocks=num_blocks,
        block_size=block_size,
        head_size=head_size,
        num_heads=num_heads,
        sliding_window=sliding_window,
        q_dtype=q_dtype,
        kv_dtype=kv_dtype,
        out_dtype=o_dtype,
        shuffled_kv_cache=shuffled_kv_cache,
        use_out_scale=use_out_scale,
        device="cuda",
    )

    unified_attention(
        q=maybe_quant_query,
        k=maybe_shuffled_key_cache,
        v=maybe_shuffled_value_cache,
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
        q_scales=query_scales,
        output_scale=output_scale,
        sinks=sinks,
        shuffled_kv_cache=shuffled_kv_cache,
    )

    ref_output = ref_paged_attn(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        query_lens=query_lens,
        kv_lens=kv_lens,
        block_tables=block_tables,
        scale=scale,
        out_dtype=o_dtype,
        q_descale=q_descale,
        k_descale=k_descale,
        v_descale=v_descale,
        output_scale=output_scale,
        sliding_window=sliding_window,
        soft_cap=soft_cap,
        sinks=sinks,
    )

    atol, rtol = 1.5e-2, 1e-2
    if q_dtype != torch.bfloat16 or kv_dtype != torch.bfloat16:
        atol, rtol = 1.5e-1, 1.5e-1
    tol_err_ratio = 0.01
    assert (
        checkAllclose(
            output.to(torch.bfloat16),
            ref_output.to(torch.bfloat16),
            atol=atol,
            rtol=rtol,
            tol_err_ratio=tol_err_ratio,
            msg="unified_attn_3d output",
        )
        <= tol_err_ratio
    )


@pytest.mark.parametrize("captured_max_seqlen_k", [False, True])
@torch.inference_mode()
def test_triton_unified_attn_3d_long_kv(captured_max_seqlen_k: bool) -> None:
    """Split-KV verify on the Kimi-K3 DSpark draft shape (head dim 64, bf16 Q, fp8 KV).

    max_seqlen_k is the real maximum (20000) or the 1M bound a CUDA graph captures.
    """
    if DEVICE_ARCH not in ("gfx950", "gfx1250"):
        pytest.skip(f"skip {DEVICE_ARCH}")
    torch.cuda.empty_cache()
    seq_lens = [(4, 20000), (4, 9000), (4, 300), (4, 13000)]
    (
        query,
        key_cache,
        value_cache,
        _,
        _,
        _,
        output,
        cu_query_lens,
        kv_lens,
        max_query_len,
        max_kv_len,
        scale,
        window_size,
        block_tables,
        _,
        _,
        q_descale,
        k_descale,
        v_descale,
        _,
    ) = generate_data(
        seq_lens=seq_lens,
        num_blocks=1024,
        block_size=128,
        head_size=64,
        num_heads=(8, 2),
        q_dtype=torch.bfloat16,
        kv_dtype=e4m3_dtype,
        out_dtype=torch.bfloat16,
    )

    unified_attention(
        q=query,
        k=key_cache,
        v=value_cache,
        out=output,
        cu_seqlens_q=cu_query_lens,
        seqused_k=kv_lens,
        max_seqlen_q=max_query_len,
        max_seqlen_k=1 << 20 if captured_max_seqlen_k else max_kv_len,
        softmax_scale=scale,
        causal=True,
        window_size=window_size,
        block_table=block_tables,
        softcap=0,
        q_descale=q_descale,
        k_descale=k_descale,
        v_descale=v_descale,
    )

    ref_output = ref_paged_attn(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        query_lens=[x[0] for x in seq_lens],
        kv_lens=kv_lens,
        block_tables=block_tables,
        scale=scale,
        out_dtype=torch.bfloat16,
        q_descale=q_descale,
        k_descale=k_descale,
        v_descale=v_descale,
    )

    _assert_request_rel_l2(output, ref_output, seq_lens)


def _length_aware_seq_lens(batch: str, query_len: int):
    if batch in ("skewed", "skewed_300", "skewed_empty"):
        # one long request among short ones, as a long-context draft batch; 300
        # requests take more than one step over seq_lens in the kernels
        rng = random.Random(7)
        short = [
            rng.randint(200, 3000) for _ in range(299 if batch == "skewed_300" else 31)
        ]
        if batch == "skewed_empty":
            # padded slots of a CUDA graph batch have no KV
            short = [0 if i % 8 == 0 else kv for i, kv in enumerate(short)]
        return [(query_len, 120_000)] + [(query_len, kv) for kv in short]
    return [(query_len, 8192)] * 32


def _length_aware_pages(page_counts):
    """KV cache size and block table holding only the pages each request uses.

    A cache of len(page_counts) x max(page_counts) pages gives every request the
    long request's length (~37 GB of fp32 K/V for skewed_300). Each request gets
    distinct random pages; the table entries past them are never read and name
    random pages of the cache.
    """
    num_blocks = sum(page_counts) + 64  # spare pages no request owns
    rng = torch.Generator(device="cuda").manual_seed(0)
    block_tables = torch.randint(
        num_blocks,
        (len(page_counts), max(page_counts)),
        generator=rng,
        dtype=torch.int32,
        device="cuda",
    )
    counts = torch.tensor(page_counts, device="cuda")
    owned = torch.arange(block_tables.shape[1], device="cuda") < counts[:, None]
    pages = torch.randperm(num_blocks, generator=rng, device="cuda")
    block_tables[owned] = pages[: sum(page_counts)].to(torch.int32)
    return num_blocks, block_tables


@pytest.mark.parametrize(
    "batch, num_heads, max_query_len, option, seqused_k_dtype",
    [
        (batch, *shape, torch.int32)
        for shape in [
            ((8, 2), 4, None),  # Kimi-K3 DSpark draft verify
            ((8, 2), 1, None),
            ((8, 1), 1, "sinks"),  # GPT-OSS TP8 full-attention decode
        ]
        for batch in ["skewed", "uniform", "skewed_empty"]
    ]
    + [("skewed_300", (8, 2), 1, None, torch.int32)]
    # the widest verify query that still fits one block of the q >= 4 entry
    + [("skewed", (8, 2), 8, None, torch.int32)]
    # sglang passes int64 sequence lengths
    + [("skewed", (8, 2), 4, None, torch.int64)]
    # the length-aware split does not exclude these inputs
    + [
        ("skewed", (8, 2), 4, option, torch.int32)
        for option in ("shuffled_kv_cache", "softcap", "qq_bias")
    ],
)
@torch.inference_mode()
def test_triton_unified_attn_3d_length_aware_split(
    batch: str,
    num_heads: tuple[int, int],
    max_query_len: int,
    option: str | None,
    seqused_k_dtype: torch.dtype,
    monkeypatch,
) -> None:
    """Length-aware KV split (kv_split SPLIT_PROGRAMS_PER_CU), head dim 64 bf16-Q / fp8-KV.

    max_seqlen_k is the 1M bound a CUDA graph captures, so the grid's segment
    count only knows the batch size; each sequence must take its own share of
    the segments from seqused_k, and the reduce must merge exactly those.
    """
    if DEVICE_ARCH != "gfx950":
        pytest.skip("the length-aware kv_split entry is tuned for gfx950")
    torch.cuda.empty_cache()
    block_size = 128
    seq_lens = _length_aware_seq_lens(batch, max_query_len)
    num_blocks, block_tables = _length_aware_pages(
        [triton.cdiv(kv, block_size) for _, kv in seq_lens]
    )
    shuffled_kv_cache = option == "shuffled_kv_cache"
    (
        query,
        key_cache,
        value_cache,
        maybe_shuffled_key_cache,
        maybe_shuffled_value_cache,
        sinks,
        output,
        cu_query_lens,
        kv_lens,
        max_seqlen_q,
        _,
        scale,
        window_size,
        _,  # block_tables come from _length_aware_pages
        _,
        _,
        q_descale,
        k_descale,
        v_descale,
        _,
    ) = generate_data(
        seq_lens=seq_lens,
        num_blocks=num_blocks,
        block_size=block_size,
        head_size=64,
        num_heads=num_heads,
        q_dtype=torch.bfloat16,
        kv_dtype=e4m3_dtype,
        out_dtype=torch.bfloat16,
        shuffled_kv_cache=shuffled_kv_cache,
    )
    sinks = sinks if option == "sinks" else None
    # the scores here are ~0.1; a cap of 0.1 moves each request ~5%, past the bound
    soft_cap = 0.1 if option == "softcap" else None
    qq_bias = None
    if option == "qq_bias":
        qq_bias = torch.randn(
            max_query_len, max_query_len, dtype=torch.float32, device="cuda"
        )

    # record the split the launcher picks
    module = sys.modules[unified_attention.__module__]
    launches = []
    launch_3d = module._unified_attention_3d_triton

    def spy(params, *args):
        launches.append(args[3:])  # NUM_SEGMENTS, TILE_SIZE, length-aware split
        return launch_3d(params, *args)

    monkeypatch.setattr(module, "_unified_attention_3d_triton", spy)

    output.fill_(float("nan"))  # the reduce must write every row
    unified_attention(
        q=query,
        k=maybe_shuffled_key_cache,
        v=maybe_shuffled_value_cache,
        out=output,
        cu_seqlens_q=cu_query_lens,
        seqused_k=kv_lens.to(seqused_k_dtype),
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=1 << 20,
        softmax_scale=scale,
        causal=True,
        window_size=window_size,
        block_table=block_tables,
        softcap=soft_cap or 0,
        q_descale=q_descale,
        k_descale=k_descale,
        v_descale=v_descale,
        sinks=sinks,
        qq_bias=qq_bias,
        shuffled_kv_cache=shuffled_kv_cache,
    )

    # the split is length-aware, and the per-sequence counts (the kernels'
    # seq_num_segments) differ from the cap the grid was sized for
    assert len(launches) == 1, "expected the 3D split-KV path"
    num_segments, _, split = launches[0]
    assert num_segments > 1 and split is not None
    split_budget = split.split_budget
    kv = [k for _, k in seq_lens]
    segment_tokens = max(triton.cdiv(sum(kv), split_budget), 1)
    used = [min(max(k // segment_tokens, 1), num_segments) for k in kv]
    assert sum(used) <= split_budget + len(kv) - 1
    if batch != "uniform":
        assert used[0] > max(used[1:]) and min(used) < num_segments, used
    else:
        assert len(set(used)) == 1 and used[0] < num_segments, used

    ref_output = ref_paged_attn(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        query_lens=[q for q, _ in seq_lens],
        kv_lens=kv_lens,
        block_tables=block_tables,
        scale=scale,
        out_dtype=torch.bfloat16,
        soft_cap=soft_cap,
        sinks=sinks,
        q_descale=q_descale,
        k_descale=k_descale,
        v_descale=v_descale,
        qq_bias=qq_bias,
    )

    # a dropped or doubly merged segment moves it far past the bound
    _assert_request_rel_l2(output, ref_output, seq_lens)


@torch.inference_mode()
def test_triton_unified_attn_3d_length_aware_split_graph_replay() -> None:
    """A CUDA graph captured at the 1M bound splits by the seqused_k of each replay.

    A serving engine changes the lengths in place between replays, from a skewed
    batch to a uniform one, to one with padded (zero-length) slots, and back.
    """
    if DEVICE_ARCH != "gfx950":
        pytest.skip("the length-aware kv_split entry is tuned for gfx950")
    torch.cuda.empty_cache()
    block_size, query_len = 128, 4
    skewed = _length_aware_seq_lens("skewed", query_len)
    batches = [
        skewed,
        _length_aware_seq_lens("uniform", query_len),
        _length_aware_seq_lens("skewed_empty", query_len),
        skewed,
    ]
    # each request keeps its pages across the replays
    num_blocks, block_tables = _length_aware_pages(
        [max(triton.cdiv(kv, block_size) for _, kv in r) for r in zip(*batches)]
    )
    (
        query,
        key_cache,
        value_cache,
        _,
        _,
        _,
        output,
        cu_query_lens,
        kv_lens,
        max_seqlen_q,
        _,
        scale,
        window_size,
        _,  # block_tables come from _length_aware_pages
        _,
        _,
        q_descale,
        k_descale,
        v_descale,
        _,
    ) = generate_data(
        seq_lens=skewed,
        num_blocks=num_blocks,
        block_size=block_size,
        head_size=64,
        num_heads=(8, 2),
        q_dtype=torch.bfloat16,
        kv_dtype=e4m3_dtype,
        out_dtype=torch.bfloat16,
    )

    def run():
        unified_attention(
            q=query,
            k=key_cache,
            v=value_cache,
            out=output,
            cu_seqlens_q=cu_query_lens,
            seqused_k=kv_lens,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=1 << 20,
            softmax_scale=scale,
            causal=True,
            window_size=window_size,
            block_table=block_tables,
            softcap=0,
            q_descale=q_descale,
            k_descale=k_descale,
            v_descale=v_descale,
        )

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run()  # compile outside the capture
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()

    for batch in batches:
        kv = [k for _, k in batch]
        kv_lens.copy_(torch.tensor(kv, dtype=kv_lens.dtype, device="cuda"))
        output.fill_(float("nan"))
        graph.replay()
        ref_output = ref_paged_attn(
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            query_lens=[q for q, _ in batch],
            kv_lens=kv,
            block_tables=block_tables,
            scale=scale,
            out_dtype=torch.bfloat16,
            q_descale=q_descale,
            k_descale=k_descale,
            v_descale=v_descale,
        )
        _assert_request_rel_l2(output, ref_output, batch)


def _gfx950_split(**overrides):
    """gfx950 Triton kv_split config and params, Kimi-K3 draft decode at the 1M bound."""
    p = {
        "head_size": 64,
        "head_size_v": 64,
        "max_seqlen_q": 1,
        "max_seqlen_k": 1 << 20,
        "sliding_window": 0,
        "shuffled_kv_cache": False,
        "q_dtype": torch.bfloat16,
        "kv_cache_dtype": e4m3_dtype,
        "block_size": 128,
        "num_sms": 256,
        "num_seqs": 64,
        "num_kv_heads": 2,
        "num_queries_per_kv": 4,
        "all_decode": True,
        "skip_reduce": False,
    }
    p.update(overrides)
    params = SimpleNamespace(**p, num_2d_prgms=p["num_seqs"] * p["num_kv_heads"])
    config = get_unified_attention_config("kv_split", params, arch="gfx950")
    return config, params


def test_length_aware_split_scope(monkeypatch) -> None:
    """The length-aware entry leaves every call it does not split as before."""
    # _length_aware_split reads the attn_3d and reduce tables; pin them to gfx950
    # like the rest
    module = sys.modules[unified_attention.__module__]
    gfx950_config = functools.partial(get_unified_attention_config, arch="gfx950")
    monkeypatch.setattr(module, "get_unified_attention_config", gfx950_config)
    config, params = _gfx950_split()
    split = _length_aware_split(config, params)
    assert split.seq_block == triton.next_power_of_2(params.num_seqs)
    assert split.num_segments == config["SPLIT_NUM_SEGMENTS"] > config["NUM_SEGMENTS"]
    # calls left out keep the uniform split of the head <= 128 entry
    head_128, _ = _gfx950_split(head_size=128, head_size_v=128)
    assert config["NUM_SEGMENTS"] == head_128["NUM_SEGMENTS"]
    for left_out in (
        {"q_dtype": torch.float16},
        {"skip_reduce": True},
        {"head_size": 32, "head_size_v": 64},
        {"head_size": 64, "head_size_v": 32},
    ):
        fallback, fallback_params = _gfx950_split(**left_out)
        assert _length_aware_split(fallback, fallback_params) is None
        assert fallback["NUM_SEGMENTS"] == head_128["NUM_SEGMENTS"]
        assert fallback["TILE_SIZE"] == head_128["TILE_SIZE"]
    # a sliding window keeps the uniform split even where the entry opts in
    windowed = _gfx950_split(sliding_window=4096)
    assert "SPLIT_PROGRAMS_PER_CU" in windowed[0]
    assert _length_aware_split(*windowed) is None
    # head 64 only
    head_32, _ = _gfx950_split(head_size=32, head_size_v=32)
    assert "SPLIT_PROGRAMS_PER_CU" not in head_32
    # the kernels round sum(seq_len) up by the budget in int32
    edge = {"num_seqs": 1023, "num_kv_heads": 1}
    assert _length_aware_split(*_gfx950_split(**edge, max_seqlen_k=1 << 21))
    assert _length_aware_split(*_gfx950_split(**edge, max_seqlen_k=2_099_202)) is None
    # and round one sequence up by its segments' tiles in int32
    one = {"num_seqs": 1, "num_kv_heads": 1}
    assert _length_aware_split(*_gfx950_split(**one, max_seqlen_k=2**31 - 2048)) is None
    # find_work_item unpacks the sequences before an item from 15 bits
    wide = {**vars(params), "max_seqlen_k": 1 << 15}
    assert _length_aware_split(
        config, SimpleNamespace(**{**wide, "num_seqs": 2**15 - 1})
    )
    assert (
        _length_aware_split(config, SimpleNamespace(**{**wide, "num_seqs": 2**15}))
        is None
    )

    # the grid holds split_budget + num_seqs - 1 work items; the shares reach that
    # only when empty sequences sit beside one that fills the budget
    def num_items(kv, budget):
        segment_tokens = max(triton.cdiv(sum(kv), budget), 1)
        return sum(max(k // segment_tokens, 1) for k in kv)

    rng = random.Random(0)
    for _ in range(2000):
        kv = [
            rng.choice((0, 1, rng.randint(1, 1 << 17)))
            for _ in range(rng.randint(1, 40))
        ]
        budget = rng.randint(1, 1024)
        assert num_items(kv, budget) <= budget + len(kv) - 1, (kv, budget)
    assert num_items([512 * 64, 0], 512) == 512 + 2 - 1

    # an entry without one of the split's keys keeps the uniform split
    for key in module._LENGTH_AWARE_KV_SPLIT_KEYS:
        partial = {k: v for k, v in config.items() if k != key}
        assert _length_aware_split(partial, params) is None, key
    for section, key in (
        ("attn_3d", "SPLIT_RUNS_PER_XCD"),
        ("reduce", "SPLIT_SEGMENT_BLOCK_MIN"),
    ):

        def without_key(op, *args, section=section, key=key, **kwargs):
            entry = gfx950_config(op, *args, **kwargs)
            if op == section:
                entry.pop(key)
            return entry

        with monkeypatch.context() as m:
            m.setattr(module, "get_unified_attention_config", without_key)
            assert _length_aware_split(config, params) is None, key

    def verify(q):
        return SimpleNamespace(
            **{**vars(params), "max_seqlen_q": q, "all_decode": q == 1}
        )

    # a verify query longer than one query block (BLOCK_Q 8 at q >= 4) keeps the
    # uniform split
    assert _length_aware_split(config, verify(9)) is None
    # the 3D launch and the reduce read their length-aware keys from these entries,
    # for decode and for verify queries up to one query block (q <= 8 here)
    for q in (1, 4, 8):
        verify_q = verify(q)
        assert _length_aware_split(config, verify_q)
        attn_3d = get_unified_attention_config("attn_3d", verify_q, arch="gfx950")
        assert "SPLIT_RUNS_PER_XCD" in attn_3d
        if attn_3d["BLOCK_M"] > 16:
            # a wider tuned block is cut down to the query rows of one sequence
            assert attn_3d["SPLIT_BLOCK_M"] == 16
        assert "SPLIT_SEGMENT_BLOCK_MIN" in get_unified_attention_config(
            "reduce", verify_q, arch="gfx950"
        )


@pytest.mark.parametrize(
    "seq_lens",
    [
        [(512, 512)],
        [
            (1, 15),
            (12, 133),
            (12, 87),
            (1, 133),
            (2, 343),
            (567, 275),
            (34, 345),
            (777, 777),
            (454, 345),
            (1, 134),
        ],
    ],
)
@pytest.mark.parametrize("num_heads", [(8, 8), (8, 1)])
@pytest.mark.parametrize("head_size", [64, 128, 256, 512])
@pytest.mark.parametrize("block_size", [16, 64])
@pytest.mark.parametrize("sliding_window", [None, 256])
@pytest.mark.parametrize(
    "soft_cap",
    [None, 50.0],
)
@pytest.mark.parametrize(
    "num_blocks",
    [2048, 32768],
)
@pytest.mark.parametrize(
    "q_dtype, kv_dtype, out_dtype, use_q_descale, use_kv_descale, use_out_scale",
    [
        (torch.bfloat16, torch.bfloat16, torch.bfloat16, False, False, False),
        (torch.bfloat16, e4m3_dtype, torch.bfloat16, False, True, False),
        (e4m3_dtype, e4m3_dtype, torch.bfloat16, True, True, False),
        (torch.float16, torch.float16, torch.float16, False, False, False),
    ],
)
@pytest.mark.parametrize(
    "shuffled_kv_cache",
    [
        True,
        False,
    ],
)
@pytest.mark.parametrize("backend", ["triton", "gluon"])
@torch.inference_mode()
def test_triton_unified_attn(
    seq_lens: list[tuple[int, int]],
    num_heads: tuple[int, int],
    head_size: int,
    sliding_window: int | None,
    block_size: int,
    soft_cap: float | None,
    num_blocks: int,
    q_dtype: torch.dtype,
    kv_dtype: torch.dtype,
    out_dtype: torch.dtype,
    use_q_descale: bool,
    use_kv_descale: bool,
    use_out_scale: bool,
    shuffled_kv_cache: bool,
    backend: str,  # "triton" | "gluon"
) -> None:
    if backend == "gluon" and not _is_gluon_available():
        pytest.skip(f"skip gluon backend, not available on {DEVICE_ARCH}")
    use_gluon_2d = is_2d_gluon_available(
        SimpleNamespace(
            q_dtype=q_dtype,
            kv_cache_dtype=kv_dtype,
            softcap=soft_cap,
            use_qq_bias=False,
            use_alibi_slopes=False,
            head_size=head_size,
            head_size_v=head_size,
            shuffled_kv_cache=shuffled_kv_cache,
            block_size=block_size,
            k_width=16 // kv_dtype.itemsize,
        ),
        backend,
    )
    torch.manual_seed(0)
    # shuffling only supported for gfx1250 gluon kernels
    if shuffled_kv_cache and not use_gluon_2d:
        pytest.skip("skip shuffled_kv_cache, 2d gluon not available")
    query_lens = [x[0] for x in seq_lens]
    kv_lens_list = [x[1] for x in seq_lens]
    (
        query,
        key_cache_orig,
        value_cache_orig,
        key_cache,
        value_cache,
        sinks,
        output,
        cu_query_lens,
        kv_lens,
        max_query_len,
        max_kv_len,
        scale,
        window_size,
        block_tables,
        _maybe_quant_query,
        _query_scales,
        q_descale,
        k_descale,
        v_descale,
        output_scale,
    ) = generate_data(
        seq_lens=seq_lens,
        num_blocks=num_blocks,
        block_size=block_size,
        head_size=head_size,
        num_heads=num_heads,
        sliding_window=sliding_window,
        q_dtype=q_dtype,
        kv_dtype=kv_dtype,
        out_dtype=out_dtype,
        shuffled_kv_cache=shuffled_kv_cache,
        use_q_descale=use_q_descale,
        use_kv_descale=use_kv_descale,
        use_out_scale=use_out_scale,
        device="cuda",
    )

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
        softcap=soft_cap if soft_cap is not None else 0,
        q_descale=q_descale,
        k_descale=k_descale,
        v_descale=v_descale,
        sinks=sinks,
        output_scale=output_scale,
        shuffled_kv_cache=shuffled_kv_cache,
        backend=backend,
    )

    ref_output = ref_paged_attn(
        query=query,
        key_cache=key_cache_orig,
        value_cache=value_cache_orig,
        query_lens=query_lens,
        kv_lens=kv_lens_list,
        block_tables=block_tables,
        scale=scale,
        out_dtype=out_dtype,
        sliding_window=sliding_window,
        soft_cap=soft_cap,
        sinks=sinks,
        q_descale=q_descale,
        k_descale=k_descale,
        v_descale=v_descale,
        output_scale=output_scale,
    )

    atol, rtol = 1.5e-2, 1e-2
    is_fp8 = kv_dtype.itemsize == 1 or q_dtype.itemsize == 1
    if is_fp8:
        atol, rtol = 1.5e-1, 1.5e-1
    output = output.to(torch.float32)
    ref_output = ref_output.to(torch.float32)
    if is_fp8 and use_gluon_2d and (use_kv_descale or use_q_descale):
        # For fp8 allow up to 1% of elements to fall outside tolerance.
        # NOTE: fp8 + q/kv scaling causes around 0.1% mismatch with gluon kernel
        # Might be related to softmax trick to use pk_fma
        mismatch = torch.abs(output - ref_output) > atol + rtol * torch.abs(ref_output)
        mismatch_fraction = mismatch.float().mean().item()
        assert mismatch_fraction < 0.005, (
            f"fp8 mismatch fraction {mismatch_fraction:.4%} exceeds 0.5% "
            f"(max abs diff {torch.max(torch.abs(output - ref_output))})"
        )
    else:
        (
            torch.testing.assert_close(output, ref_output, atol=atol, rtol=rtol),
            f"{torch.max(torch.abs(output - ref_output))}",
        )


# Query lengths that leave padded rows in the last query block (BLOCK_Q of 2/4/8),
# on the 3D split-KV path (long KV) and the 2D path (KV <= 512).
@pytest.mark.parametrize(
    "num_heads, query_len, kv_len",
    [
        ((32, 8), 3, 4096),
        ((32, 8), 4, 4096),
        ((16, 1), 3, 4096),
        ((32, 8), 3, 256),
        ((32, 8), 5, 256),
    ],
)
@torch.inference_mode()
def test_triton_unified_attn_qq_bias_padded_query_rows(
    num_heads: tuple[int, int], query_len: int, kv_len: int
) -> None:
    if DEVICE_ARCH == "gfx1250":
        pytest.skip("gfx1250 Gluon kernel does not mask padded qq_bias rows")
    seq_lens = [(query_len, kv_len)] * 2
    (
        query,
        key_cache_orig,
        value_cache_orig,
        key_cache,
        value_cache,
        _sinks,
        output,
        cu_query_lens,
        kv_lens,
        max_query_len,
        max_kv_len,
        scale,
        window_size,
        block_tables,
        *_,
    ) = generate_data(
        seq_lens=seq_lens,
        num_blocks=2048,
        block_size=16,
        head_size=128,
        num_heads=num_heads,
        device="cuda",
    )
    qq_bias = torch.randn(query_len, query_len, dtype=torch.float32, device="cuda")

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
        softcap=0,
        q_descale=None,
        k_descale=None,
        v_descale=None,
        qq_bias=qq_bias,
        backend="triton",
    )

    ref_output = ref_paged_attn(
        query=query,
        key_cache=key_cache_orig,
        value_cache=value_cache_orig,
        query_lens=[x[0] for x in seq_lens],
        kv_lens=[x[1] for x in seq_lens],
        block_tables=block_tables,
        scale=scale,
        out_dtype=output.dtype,
        qq_bias=qq_bias,
    )
    torch.testing.assert_close(
        output.to(torch.float32), ref_output.to(torch.float32), atol=1.5e-2, rtol=1e-2
    )


@pytest.mark.parametrize(
    "seq_lens, num_heads, block_size, sliding_window",
    [
        ([(4, 3000), (8, 40000), (1, 9000)], (16, 2), 16, 2048),
        ([(8, 8704)], (16, 2), 128, 2048),
        ([(8, 582), (1, 582)], (16, 2), 64, 512),
        ([(1, 8704)] * 4, (16, 2), 16, 2048),
        ([(128, 1100)], (32, 32), 64, 1024),
        ([(17, 1100)], (32, 32), 64, 1024),
    ],
)
@torch.inference_mode()
def test_triton_unified_attn_sw_verify_split(
    seq_lens, num_heads, block_size, sliding_window
):
    if DEVICE_ARCH != "gfx950" or not _is_gluon_available():
        pytest.skip("gfx950 gluon split dispatch")
    query_lens = [q for q, _ in seq_lens]
    kv_lens = [kv for _, kv in seq_lens]
    max_q = max(query_lens)
    q, k, v, _, _, sinks, out, cu_q, kv_lens_t, _, max_k, scale, window, table = (
        generate_data(
            seq_lens,
            num_blocks=len(seq_lens) * -(-max(kv_lens) // block_size),
            block_size=block_size,
            num_heads=num_heads,
            sliding_window=sliding_window,
        )[:14]
    )
    sinks = sinks + 8.0
    ref = ref_paged_attn(
        q,
        k,
        v,
        query_lens,
        kv_lens,
        table,
        scale,
        torch.bfloat16,
        sliding_window,
        sinks=sinks,
    ).float()

    first = [max(0, kv - ql - sliding_window + 1) // block_size for ql, kv in seq_lens]
    width = (sliding_window + max_q + block_size - 2) // block_size + 1
    index = torch.tensor(first, device="cuda")[:, None] + torch.arange(
        width, device="cuda"
    )
    trimmed_lens = [kv - fb * block_size for kv, fb in zip(kv_lens, first)]
    assert max(trimmed_lens) <= sliding_window + max_q + block_size - 2

    for block_table, lens, max_seqlen_k in [
        (table, kv_lens_t, max_k),
        (
            table.gather(1, index.clamp(max=table.shape[1] - 1)),
            torch.tensor(trimmed_lens, dtype=torch.int32, device="cuda"),
            max(trimmed_lens),
        ),
    ]:
        out = torch.empty_like(out)
        unified_attention(
            q=q,
            k=k,
            v=v,
            out=out,
            cu_seqlens_q=cu_q,
            max_seqlen_q=max_q,
            seqused_k=lens,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=scale,
            causal=True,
            window_size=window,
            block_table=block_table,
            softcap=0,
            q_descale=None,
            k_descale=None,
            v_descale=None,
            sinks=sinks,
        )
        torch.testing.assert_close(out.float(), ref, atol=1.5e-2, rtol=1e-2)


@pytest.mark.parametrize(
    "seq_lens, num_heads, block_size",
    [
        ([(8, 10246)], (16, 2), 8192),
        ([(8, 6150), (1, 6150)], (8, 1), 4096),
    ],
)
@torch.inference_mode()
def test_triton_unified_attn_sw_split_no_sinks(seq_lens, num_heads, block_size):
    # a page much longer than a tile leaves 32+ empty leading segments, a whole
    # reduce chunk at head 256, which without sinks has no finite max
    if DEVICE_ARCH != "gfx950" or not _is_gluon_available():
        pytest.skip("gfx950 gluon split dispatch")

    sliding_window = 2048
    query_lens = [q for q, _ in seq_lens]
    kv_lens = [kv for _, kv in seq_lens]
    q, k, v, _, _, _, out, cu_q, kv_lens_t, max_q, max_k, scale, window, table = (
        generate_data(
            seq_lens,
            num_blocks=len(seq_lens) * -(-max(kv_lens) // block_size),
            block_size=block_size,
            head_size=256,
            num_heads=num_heads,
            sliding_window=sliding_window,
        )[:14]
    )
    ref = ref_paged_attn(
        q, k, v, query_lens, kv_lens, table, scale, torch.bfloat16, sliding_window
    ).float()

    unified_attention(
        q=q,
        k=k,
        v=v,
        out=out,
        cu_seqlens_q=cu_q,
        max_seqlen_q=max_q,
        seqused_k=kv_lens_t,
        max_seqlen_k=max_k,
        softmax_scale=scale,
        causal=True,
        window_size=window,
        block_table=table,
        softcap=0,
        q_descale=None,
        k_descale=None,
        v_descale=None,
    )
    torch.testing.assert_close(out.float(), ref, atol=1.5e-2, rtol=1e-2)


@pytest.mark.parametrize(
    "seq_lens",
    [
        [(1, 1328)],
        [(1, 523), (1, 37), (1, 2011)],
        [(256, 256)],
        [(1, 137), (5, 1024), (1, 19)],
    ],
)
@pytest.mark.parametrize("num_heads", [(64, 8), (8, 1)])
@pytest.mark.parametrize(
    "head_size, v_head_size",
    [
        # MiMo decode: qk head wider than the value head
        (192, 128),
        # symmetric, must stay identical to the single-mask path
        (128, 128),
        # value head wider than qk, the reverse ordering
        (128, 192),
    ],
)
@pytest.mark.parametrize("block_size", [16, 64])
# fp8 prefill takes the SPLIT_UNMASKED_LOOP path on gfx950
@pytest.mark.parametrize(
    "q_dtype, kv_dtype",
    [(torch.bfloat16, torch.bfloat16), (e4m3_dtype, e4m3_dtype)],
)
@pytest.mark.parametrize("backend", ["triton", "gluon"])
@torch.inference_mode()
def test_triton_unified_attn_asym_head(
    seq_lens: list[tuple[int, int]],
    num_heads: tuple[int, int],
    head_size: int,
    v_head_size: int,
    block_size: int,
    q_dtype: torch.dtype,
    kv_dtype: torch.dtype,
    backend: str,
) -> None:
    if backend == "gluon" and not _is_gluon_available():
        pytest.skip(f"skip gluon backend, not available on {DEVICE_ARCH}")
    torch.manual_seed(0)
    query_lens = [x[0] for x in seq_lens]
    kv_lens_list = [x[1] for x in seq_lens]

    (
        query,
        key_cache_orig,
        value_cache_orig,
        key_cache,
        value_cache,
        sinks,
        output,
        cu_query_lens,
        kv_lens,
        max_query_len,
        max_kv_len,
        scale,
        window_size,
        block_tables,
        _maybe_quant_query,
        _query_scales,
        q_descale,
        k_descale,
        v_descale,
        output_scale,
    ) = generate_data(
        seq_lens=seq_lens,
        block_size=block_size,
        head_size=head_size,
        v_head_size=v_head_size,
        num_heads=num_heads,
        q_dtype=q_dtype,
        kv_dtype=kv_dtype,
        device="cuda",
    )

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
        softcap=0,
        q_descale=q_descale,
        k_descale=k_descale,
        v_descale=v_descale,
        sinks=sinks,
        output_scale=output_scale,
        backend=backend,
    )

    ref_output = ref_paged_attn(
        query=query,
        key_cache=key_cache_orig,
        value_cache=value_cache_orig,
        query_lens=query_lens,
        kv_lens=kv_lens_list,
        block_tables=block_tables,
        scale=scale,
        out_dtype=torch.bfloat16,
        sinks=sinks,
        q_descale=q_descale,
        k_descale=k_descale,
        v_descale=v_descale,
    )

    atol, rtol = 1.5e-2, 1e-2
    if q_dtype != torch.bfloat16 or kv_dtype != torch.bfloat16:
        atol, rtol = 1.5e-1, 1.5e-1
    tol_err_ratio = 0.01
    assert (
        checkAllclose(
            output.to(torch.bfloat16),
            ref_output.to(torch.bfloat16),
            atol=atol,
            rtol=rtol,
            tol_err_ratio=tol_err_ratio,
            msg=f"unified_attn asym head qk={head_size} v={v_head_size}",
        )
        <= tol_err_ratio
    )


@pytest.mark.parametrize("case", ["shuffled_kv_cache", "out_qk_width"])
@torch.inference_mode()
def test_triton_unified_attn_asym_head_guards(case: str) -> None:
    head_size, v_head_size = 192, 128
    (
        query,
        _key_cache_orig,
        _value_cache_orig,
        key_cache,
        value_cache,
        sinks,
        output,
        cu_query_lens,
        kv_lens,
        max_query_len,
        max_kv_len,
        scale,
        window_size,
        block_tables,
        _maybe_quant_query,
        _query_scales,
        _q_descale,
        _k_descale,
        _v_descale,
        _output_scale,
    ) = generate_data(
        seq_lens=[(1, 523)],
        num_blocks=64,
        block_size=64,
        head_size=head_size,
        v_head_size=v_head_size,
        num_heads=(8, 1),
        device="cuda",
    )
    shuffled = case == "shuffled_kv_cache"
    if shuffled:
        # shuffle_kv_cache wants matching heads, so shuffle k and v on their own
        key_cache = shuffle_kv_cache(key_cache, key_cache)[0]
        value_cache = shuffle_kv_cache(value_cache, value_cache)[1]
    else:
        # the old convention: out sized by the qk head
        output = output.new_empty((*output.shape[:-1], head_size))

    with pytest.raises(AssertionError, match="value head size"):
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
            softcap=0,
            q_descale=None,
            k_descale=None,
            v_descale=None,
            sinks=sinks,
            shuffled_kv_cache=shuffled,
            backend="triton",
        )


@pytest.mark.parametrize("backend", ["triton", "gluon"])
@torch.inference_mode()
def test_triton_unified_attn_packed_kv_over_2gib(backend: str) -> None:
    # vLLM passes k and v as strided views of one cache that packs k and v in
    # the last dim. Each view has fewer than 2 GiB of elements, but the view
    # spans more than 2 GiB, so blocks at the end need 64-bit offsets.
    if backend == "gluon" and not _is_gluon_available():
        pytest.skip(f"skip gluon backend, not available on {DEVICE_ARCH}")
    torch.manual_seed(0)
    num_blocks, block_size, num_kv_heads, head_size = 40000, 16, 8, 128
    num_query_heads = 64
    dtype = torch.bfloat16
    kv_cache = torch.empty(
        num_blocks, num_kv_heads, block_size, 2 * head_size, dtype=dtype, device="cuda"
    )
    kv_cache.uniform_(-1, 1)
    key_cache, value_cache = kv_cache.transpose(1, 2).split(head_size, dim=-1)
    assert key_cache.nelement() * key_cache.element_size() < 2**31
    assert key_cache[-1].data_ptr() - kv_cache.data_ptr() >= 2**31

    seq_lens = [(1, 1328), (4, 2011), (1, 37), (16, 523)]
    query_lens = [x[0] for x in seq_lens]
    kv_lens_list = [x[1] for x in seq_lens]
    max_blocks = max((kv + block_size - 1) // block_size for kv in kv_lens_list)
    # The blocks past 2 GiB, at the end of the cache.
    block_tables = torch.arange(
        num_blocks - len(seq_lens) * max_blocks,
        num_blocks,
        dtype=torch.int32,
        device="cuda",
    ).view(len(seq_lens), max_blocks)

    query = torch.randn(
        sum(query_lens), num_query_heads, head_size, dtype=dtype, device="cuda"
    )
    output = torch.empty_like(query)
    cu_query_lens = torch.tensor(
        [0] + query_lens, dtype=torch.int32, device="cuda"
    ).cumsum(dim=0, dtype=torch.int32)
    kv_lens = torch.tensor(kv_lens_list, dtype=torch.int32, device="cuda")
    scale = head_size**-0.5
    unified_attention(
        q=query,
        k=key_cache,
        v=value_cache,
        out=output,
        cu_seqlens_q=cu_query_lens,
        seqused_k=kv_lens,
        max_seqlen_q=max(query_lens),
        max_seqlen_k=max(kv_lens_list),
        softmax_scale=scale,
        causal=True,
        window_size=(-1, -1),
        block_table=block_tables,
        softcap=0,
        q_descale=None,
        k_descale=None,
        v_descale=None,
        backend=backend,
    )

    # Give the reference only the blocks in use.
    used = block_tables.flatten().long()
    ref_output = ref_paged_attn(
        query=query,
        key_cache=key_cache[used].contiguous(),
        value_cache=value_cache[used].contiguous(),
        query_lens=query_lens,
        kv_lens=kv_lens_list,
        block_tables=torch.arange(used.numel(), device=block_tables.device).view(
            block_tables.shape
        ),
        scale=scale,
        out_dtype=dtype,
    )
    torch.testing.assert_close(output, ref_output, atol=1.5e-2, rtol=1e-2)


def _spec_verify_seq_lens(batch: int, max_query_len: int, seed: int):
    """Verify-step batch: mixed query lengths up to 1 + num_spec, mixed contexts."""
    rng = random.Random(seed)
    return [
        (rng.randint(1, max_query_len), rng.randint(max_query_len + 1, 3000))
        for _ in range(batch)
    ]


@pytest.mark.parametrize(
    "path, sliding_window", [("3d", None), ("2d", None), ("2d", 256), ("auto", None)]
)
@pytest.mark.parametrize("batch, seed", [(7, 0), (31, 1), (61, 2), (128, 3)])
@pytest.mark.parametrize("max_query_len", [4, 8])
@pytest.mark.parametrize("num_heads", [(16, 2), (16, 4), (16, 8), (16, 16)])
@pytest.mark.parametrize("head_size", [128, 256])
@torch.inference_mode()
def test_unified_attn_spec_verify_gfx1250(
    path: str,
    sliding_window: int | None,
    batch: int,
    seed: int,
    max_query_len: int,
    num_heads: tuple[int, int],
    head_size: int,
    monkeypatch,
) -> None:
    """Short multi-token queries (speculative-decode verify) on gfx1250: the 3D
    split-KV kernel, where query rows of one block have different causal horizons
    so a split segment can be partly or wholly past a row's horizon, and the 2D
    kernel with a BLOCK_M sized to the query rows."""
    if DEVICE_ARCH != "gfx1250":
        pytest.skip("gfx1250 Gluon kernels")
    if path != "auto":
        monkeypatch.setattr(
            sys.modules[unified_attention.__module__],
            "use_2d_kernel",
            lambda params, backend="triton": path == "2d",
        )
    torch.manual_seed(seed)
    seq_lens = _spec_verify_seq_lens(batch, max_query_len, seed)
    block_size = 16
    (
        query,
        key_cache,
        value_cache,
        shuffled_key_cache,
        shuffled_value_cache,
        _sinks,
        output,
        cu_query_lens,
        kv_lens,
        max_seqlen_q,
        max_kv_len,
        scale,
        window_size,
        block_tables,
        *_,
    ) = generate_data(
        seq_lens=seq_lens,
        num_blocks=batch * (3000 // block_size + 1) + 64,
        block_size=block_size,
        head_size=head_size,
        num_heads=num_heads,
        sliding_window=sliding_window,
        shuffled_kv_cache=True,
        device="cuda",
    )
    unified_attention(
        q=query,
        k=shuffled_key_cache,
        v=shuffled_value_cache,
        out=output,
        cu_seqlens_q=cu_query_lens,
        seqused_k=kv_lens,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_kv_len,
        softmax_scale=scale,
        causal=True,
        window_size=window_size,
        block_table=block_tables,
        softcap=0,
        q_descale=None,
        k_descale=None,
        v_descale=None,
        sinks=None,
        shuffled_kv_cache=True,
    )
    ref_output = ref_paged_attn(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        query_lens=[q for q, _ in seq_lens],
        kv_lens=kv_lens,
        block_tables=block_tables,
        scale=scale,
        out_dtype=torch.bfloat16,
        sliding_window=sliding_window,
        soft_cap=None,
        sinks=None,
    )
    assert not output.isnan().any()
    torch.testing.assert_close(output, ref_output, atol=1.5e-2, rtol=1e-2)
