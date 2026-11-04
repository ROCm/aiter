# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import random
import sys
from types import SimpleNamespace

import pytest
import torch
import triton
import triton.language as tl

from aiter.ops.triton.attention.unified_attention import (
    _is_gluon_available,
    is_2d_gluon_available,
    unified_attention,
)
from aiter.ops.triton.utils._triton import arch_info
from aiter.ops.triton.utils.shuffle import shuffle_scale_batched, shuffle_weight
from aiter.ops.triton.utils.types import e4m3_dtype
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


@pytest.mark.parametrize(
    "seq_lens",
    [
        [(1, 1328)],
        [(1, 8192)] * 32,
        [(1, 523), (1, 37), (1, 2011)],
        [(1, 1328), (1, 523), (1, 37), (1, 2011), (1, 8192)],
    ],
)
@pytest.mark.parametrize("num_heads", [(64, 8), (8, 1)])
@pytest.mark.parametrize("head_size", [64])
@pytest.mark.parametrize("sliding_window", [None])
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
@pytest.mark.parametrize("force_fp8_pv", [False, True])
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
    force_fp8_pv: bool,
) -> None:
    torch.cuda.empty_cache()

    if DEVICE_ARCH not in (
        "gfx950",
        "gfx1250",
    ):
        # gfx1250 -> Gluon
        # gfx950 -> Triton
        pytest.skip(f"skip {DEVICE_ARCH}")

    if force_fp8_pv and kv_dtype != torch.uint8:
        pytest.skip("force_fp8_pv_on_fp4_kv_cache only applies to an NVFP4 KV cache")
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
        force_fp8_pv_on_fp4_kv_cache=force_fp8_pv,
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


@pytest.mark.parametrize("num_seqs", [1, 8, 112, 160, 1025])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("num_kv_heads", [1, 2, 8])
@pytest.mark.parametrize("tile_size", [16, 64, 128])
def test_gfx12_token_split_tiles_fused(num_seqs, dtype, num_kv_heads, tile_size):
    """AITER_GFX12_TOKEN_SPLIT_FUSED: the one-program token split gives bitwise the
    nine torch ops' segm_tiles, eagerly and on CUDA-graph replay with live lengths."""
    from aiter.ops.triton.attention import unified_attention as ua

    gen = torch.Generator().manual_seed(num_seqs * 31 + tile_size)
    for hi in (1, 4096, 190_000, 2_000_000):
        lens = torch.randint(0, hi + 1, (num_seqs,), generator=gen).to(dtype)
        lens[torch.rand(num_seqs, generator=gen) < 0.2] = 0
        lens = lens.cuda()
        ref = ua._gfx12_token_split_tiles_torch(lens, num_kv_heads, tile_size)
        out = ua._gfx12_token_split_tiles_fused(lens, num_kv_heads, tile_size)
        assert out.dtype == ref.dtype == torch.int32 and out.shape == ref.shape == (1,)
        assert torch.equal(out, ref), (hi, out.item(), ref.item())
        # a strided view (every other entry of a wider buffer)
        wide = torch.repeat_interleave(lens, 2)
        wide[1::2] = 7_777_777
        out = ua._gfx12_token_split_tiles_fused(wide[::2], num_kv_heads, tile_size)
        assert torch.equal(out, ref), (hi, out.item(), ref.item())

    live = torch.zeros(num_seqs, dtype=dtype, device="cuda")
    ua._gfx12_token_split_tiles_fused(live, num_kv_heads, tile_size)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = ua._gfx12_token_split_tiles_fused(live, num_kv_heads, tile_size)
    for _ in range(8):
        lens = torch.randint(0, 262_144, (num_seqs,), generator=gen).to(dtype)
        live.copy_(lens.cuda())
        graph.replay()
        ref = ua._gfx12_token_split_tiles_torch(live, num_kv_heads, tile_size)
        assert torch.equal(captured, ref), (captured.item(), ref.item())


@pytest.mark.parametrize("path", ["2d", "3d", "auto"])
@pytest.mark.parametrize("block_size", [64, 128])
@pytest.mark.parametrize("head_size", [128, 256])
@pytest.mark.parametrize("num_heads", [(4, 1), (16, 2)])
@pytest.mark.parametrize(
    "seq_lens",
    [
        [(1, 1328), (1, 523), (1, 37), (1, 2011), (1, 8192)],  # decode
        [(4, 3000)] * 8,  # spec-decode verify
        [(512, 512), (1024, 2048), (37, 1500)],  # prefill, with a cached prefix
    ],
)
@torch.inference_mode()
def test_triton_unified_attn_k4v8(
    path: str,
    block_size: int,
    head_size: int,
    num_heads: tuple[int, int],
    seq_lens: list[tuple[int, int]],
    monkeypatch,
) -> None:
    """A8K4V8 on gfx1250: fp8 Q, NVFP4 K (e2m1 values, one e4m3 scale per 16, shuffled; scales preshuffled per
    min(128, page) rows) and fp8 V (shuffled), through the 2D and 3D kernels. The reference reads the dequantized
    K and the fp8 V with its descale; P is fp8 in the PV WMMA, as for A8W8, so the bound is on the relative L2."""
    if DEVICE_ARCH != "gfx1250":
        pytest.skip("A8K4V8 runs on the gfx1250 Gluon kernels")
    from op_tests.op_benchmarks.triton import bench_unified_attention as bench

    if path != "auto":
        monkeypatch.setattr(
            sys.modules[unified_attention.__module__],
            "use_2d_kernel",
            lambda params, backend="triton": path == "2d",
        )
    torch.manual_seed(0)
    num_blocks = len(seq_lens) * max((k + block_size - 1) // block_size for _, k in seq_lens) + 16
    inp = bench.make_inputs(
        seq_lens=seq_lens,
        num_heads=num_heads,
        head_size_qk=head_size,
        head_size_v=head_size,
        block_size=block_size,
        num_blocks=num_blocks,
        fp8_q=True,
        fp8_kv=True,
        fp8_output=False,
        out_scale_value=1.0,
        nvfp4_kv=True,
        nvfp4_v=False,
    )
    value_cache = shuffle_kv_cache(inp["k_fp8"], inp["v_fp8"])[1]
    output = torch.full_like(inp["output"], float("nan"))
    unified_attention(
        q=inp["q_fp8"],
        k=inp["k_nvfp4"],
        v=value_cache,
        out=output,
        cu_seqlens_q=inp["cu_query_lens"],
        seqused_k=inp["kv_lens"],
        max_seqlen_q=inp["max_query_len"],
        max_seqlen_k=inp["max_kv_len"],
        softmax_scale=inp["scale"],
        causal=True,
        window_size=(-1, -1),
        block_table=inp["block_tables"],
        softcap=0,
        q_descale=inp["q_descale"],
        k_descale=inp["k_descale"],
        v_descale=inp["v_descale"],
        shuffled_kv_cache=True,
        backend="gluon",
    )
    ref = ref_paged_attn(
        query=inp["query_ref"],
        key_cache=inp["key_ref"],
        value_cache=inp["value_ref"],
        query_lens=inp["query_lens"],
        kv_lens=inp["kv_lens"],
        block_tables=inp["block_tables"],
        scale=inp["scale"],
        sliding_window=None,
        soft_cap=None,
        q_descale=None,
        k_descale=inp["k_descale"],
        v_descale=inp["v_descale"],
        output_scale=None,
        out_dtype=torch.bfloat16,
    ).float()
    out = output.float()
    assert not out.isnan().any()
    rel = ((out - ref).norm() / ref.norm()).item()
    assert rel < 0.04, f"relative L2 error {rel:.4f}"
    assert (out - ref).abs().max().item() < 0.15


@pytest.mark.parametrize("loop_variant", [1, 2])
@pytest.mark.parametrize("tile_size, block_m", [(128, 128), (256, 64)])
@pytest.mark.parametrize("block_size", [64, 128])
@pytest.mark.parametrize("num_heads", [(4, 1), (16, 2)])
@pytest.mark.parametrize(
    "seq_lens",
    [
        [(512, 512), (1024, 2048), (37, 1500)],  # prefill on cached prefixes
        [(2000, 9000), (129, 129), (700, 4097)],  # ragged tails, page-crossing lengths
    ],
)
@torch.inference_mode()
def test_triton_unified_attn_k4v8_loop_variant(
    loop_variant: int,
    tile_size: int,
    block_m: int,
    block_size: int,
    num_heads: tuple[int, int],
    seq_lens: list[tuple[int, int]],
    monkeypatch,
) -> None:
    """A8K4V8 prefill through the 2D kernel with the loop variant forced (1: reordered; 2: tensor-subtile split
    with the scaled QK WMMA on each key half), tiles of one or several pages, causal masks."""
    if DEVICE_ARCH != "gfx1250":
        pytest.skip("A8K4V8 runs on the gfx1250 Gluon kernels")
    from op_tests.op_benchmarks.triton import bench_unified_attention as bench

    mod = sys.modules[unified_attention.__module__]
    monkeypatch.setattr(mod, "use_2d_kernel", lambda params, backend="triton": True)
    orig = mod.get_unified_attention_config

    def forced(op, params, backend="triton", arch=None):
        cfg = orig(op, params, backend=backend, arch=arch)
        if op == "attn_2d":
            cfg.update(
                LOOP_VARIANT=loop_variant,
                TILE_SIZE=max(tile_size, block_size),
                BLOCK_M=block_m,
                NUM_BUFFERS=2,
                num_warps=4,
            )
        return cfg

    monkeypatch.setattr(mod, "get_unified_attention_config", forced)
    torch.manual_seed(0)
    num_blocks = len(seq_lens) * max((k + block_size - 1) // block_size for _, k in seq_lens) + 16
    inp = bench.make_inputs(
        seq_lens=seq_lens,
        num_heads=num_heads,
        head_size_qk=256,
        head_size_v=256,
        block_size=block_size,
        num_blocks=num_blocks,
        fp8_q=True,
        fp8_kv=True,
        fp8_output=False,
        out_scale_value=1.0,
        nvfp4_kv=True,
        nvfp4_v=False,
    )
    value_cache = shuffle_kv_cache(inp["k_fp8"], inp["v_fp8"])[1]
    output = torch.full_like(inp["output"], float("nan"))
    unified_attention(
        q=inp["q_fp8"],
        k=inp["k_nvfp4"],
        v=value_cache,
        out=output,
        cu_seqlens_q=inp["cu_query_lens"],
        seqused_k=inp["kv_lens"],
        max_seqlen_q=inp["max_query_len"],
        max_seqlen_k=inp["max_kv_len"],
        softmax_scale=inp["scale"],
        causal=True,
        window_size=(-1, -1),
        block_table=inp["block_tables"],
        softcap=0,
        q_descale=inp["q_descale"],
        k_descale=inp["k_descale"],
        v_descale=inp["v_descale"],
        shuffled_kv_cache=True,
        backend="gluon",
    )
    ref = ref_paged_attn(
        query=inp["query_ref"],
        key_cache=inp["key_ref"],
        value_cache=inp["value_ref"],
        query_lens=inp["query_lens"],
        kv_lens=inp["kv_lens"],
        block_tables=inp["block_tables"],
        scale=inp["scale"],
        sliding_window=None,
        soft_cap=None,
        q_descale=None,
        k_descale=inp["k_descale"],
        v_descale=inp["v_descale"],
        output_scale=None,
        out_dtype=torch.bfloat16,
    ).float()
    out = output.float()
    assert not out.isnan().any()
    rel = ((out - ref).norm() / ref.norm()).item()
    assert rel < 0.04, f"relative L2 error {rel:.4f}"
    assert (out - ref).abs().max().item() < 0.15


@triton.jit
def _sigmoid_mul_ref_kernel(x_ptr, g_ptr, out_ptr, stride_g, WIDTH: tl.constexpr):
    """x * sigmoid(gate) rounded as the unfused Qwen3.5 output gate (ATOM's
    sigmoid_mul): the sigmoid to the gate dtype, then the product to bf16."""
    row = tl.program_id(0)
    cols = tl.arange(0, WIDTH)
    x = tl.load(x_ptr + row * WIDTH + cols).to(tl.float32)
    g = tl.load(g_ptr + row * stride_g + cols)
    g = tl.sigmoid(g.to(tl.float32)).to(g.dtype).to(tl.float32)
    tl.store(out_ptr + row * WIDTH + cols, (x * g).to(tl.bfloat16))


@pytest.mark.parametrize("path", ["2d", "3d", "auto"])
@pytest.mark.parametrize("token_split", [False, True])
@pytest.mark.parametrize("select", [False, True])
@pytest.mark.parametrize(
    "seq_lens",
    [
        [(1, 1328), (1, 523), (1, 37), (1, 2011), (1, 8192)] * 3,  # decode
        [(5, 3000), (5, 129), (5, 7000)] * 5,  # MTP-4 verify rows
    ],
)
@torch.inference_mode()
def test_unified_attn_output_gate_rows_k4v8(
    path: str,
    token_split: bool,
    select: bool,
    seq_lens: list[tuple[int, int]],
    monkeypatch,
) -> None:
    """The output epilogue (Qwen3.5: out * sigmoid(gate), and only some rows) is
    bitwise the unfused attention followed by the sigmoid-mul and the row gather,
    on every path the attention can end on (2D, 3D + reduce_segments, 3D with one
    segment)."""
    if DEVICE_ARCH != "gfx1250":
        pytest.skip("A8K4V8 runs on the gfx1250 Gluon kernels")
    from op_tests.op_benchmarks.triton import bench_unified_attention as bench

    module = sys.modules[unified_attention.__module__]
    if path != "auto":
        monkeypatch.setattr(
            module,
            "use_2d_kernel",
            lambda params, backend="triton": path == "2d",
        )
    monkeypatch.setattr(module, "_GFX12_TOKEN_SPLIT", token_split)
    torch.manual_seed(0)
    block_size, head_size, num_heads = 128, 256, (4, 1)
    num_blocks = (
        len(seq_lens) * max((k + block_size - 1) // block_size for _, k in seq_lens)
        + 16
    )
    inp = bench.make_inputs(
        seq_lens=seq_lens,
        num_heads=num_heads,
        head_size_qk=head_size,
        head_size_v=head_size,
        block_size=block_size,
        num_blocks=num_blocks,
        fp8_q=True,
        fp8_kv=True,
        fp8_output=False,
        out_scale_value=1.0,
        nvfp4_kv=True,
        nvfp4_v=False,
    )
    value_cache = shuffle_kv_cache(inp["k_fp8"], inp["v_fp8"])[1]
    tokens = inp["q_fp8"].shape[0]
    width = num_heads[0] * head_size
    # The gate is a column slice of the QKV-gate projection's output.
    qkvg = 3 * torch.randn(tokens, 2 * width + 512, dtype=torch.bfloat16, device="cuda")
    gate = qkvg[:, :width]
    rows = None
    if select:
        # One row a sequence (its last), as the MTP drafter keeps.
        rows = (inp["cu_query_lens"][1:] - 1).to(torch.int64)

    def run(out, **kw):
        return unified_attention(
            q=inp["q_fp8"],
            k=inp["k_nvfp4"],
            v=value_cache,
            out=out,
            cu_seqlens_q=inp["cu_query_lens"],
            seqused_k=inp["kv_lens"],
            max_seqlen_q=inp["max_query_len"],
            max_seqlen_k=inp["max_kv_len"],
            softmax_scale=inp["scale"],
            causal=True,
            window_size=(-1, -1),
            block_table=inp["block_tables"],
            softcap=0,
            q_descale=inp["q_descale"],
            k_descale=inp["k_descale"],
            v_descale=inp["v_descale"],
            shuffled_kv_cache=True,
            backend="gluon",
            **kw,
        )

    plain = run(
        torch.empty(
            tokens, num_heads[0], head_size, dtype=torch.bfloat16, device="cuda"
        )
    )
    ref = torch.empty(tokens, width, dtype=torch.bfloat16, device="cuda")
    _sigmoid_mul_ref_kernel[(tokens,)](
        plain.reshape(tokens, width),
        gate,
        ref,
        gate.stride(0),
        WIDTH=width,
        enable_fp_fusion=False,
    )
    if rows is not None:
        ref = ref[rows]
    out = torch.full(
        (ref.shape[0], num_heads[0], head_size),
        float("nan"),
        dtype=torch.bfloat16,
        device="cuda",
    )
    fused = run(out, output_gate=gate, output_rows=rows)
    assert fused.data_ptr() == out.data_ptr()
    assert torch.equal(out.reshape(ref.shape), ref)

@pytest.mark.parametrize("batch, seed", [(1, 0), (13, 1), (64, 2)])
@pytest.mark.parametrize("block_size", [64, 128])
@torch.inference_mode()
def test_gfx12_token_split_tiles_precomputed(batch, seed, block_size, monkeypatch):
    """gfx12_token_split_tiles: a caller's tiles per segment at this launch's tile
    size and KV heads replace the split's own (bitwise the same output, nothing
    computed); at another tile size the split computes its own."""
    if DEVICE_ARCH != "gfx1250":
        pytest.skip("gfx1250 Gluon kernels")
    ua = sys.modules[unified_attention.__module__]
    monkeypatch.setattr(ua, "_GFX12_TOKEN_SPLIT", True)
    monkeypatch.setattr(ua, "use_2d_kernel", lambda params, backend="triton": False)
    torch.manual_seed(seed)
    seq_lens = [(1, int(n)) for n in torch.randint(1, 20000, (batch,))]
    (
        query,
        _key_cache,
        _value_cache,
        key_cache,
        value_cache,
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
        num_blocks=batch * (20000 // block_size + 1) + 8,
        block_size=block_size,
        head_size=256,
        num_heads=(4, 1),
        sliding_window=None,
        shuffled_kv_cache=True,
        device="cuda",
    )

    def run(**kwargs):
        out = torch.empty_like(output)
        unified_attention(
            q=query,
            k=key_cache,
            v=value_cache,
            out=out,
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
            **kwargs,
        )
        return out

    ref = run()
    tiles = ua._gfx12_token_split_tiles_torch(kv_lens, 1, block_size)
    wrong_tile = (tiles, block_size * 2, 1)
    assert torch.equal(run(gfx12_token_split_tiles=wrong_tile), ref)

    def computed(*args):
        raise AssertionError("the split computed its own tiles")

    monkeypatch.setattr(ua, "_gfx12_token_split_tiles_torch", computed)
    monkeypatch.setattr(ua, "_gfx12_token_split_tiles_fused", computed)
    out = run(gfx12_token_split_tiles=(tiles, block_size, 1))
    assert torch.equal(out, ref)
