# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Optional

import pytest
import torch

from aiter.ops.triton.attention.unified_attention import (
    unified_attention,
    is_2d_gluon_available,
)
from aiter.ops.triton.utils.shuffle import shuffle_weight, shuffle_scale_batched
from op_tests.triton_tests.quant.test_quant_mxfp4 import (
    torch_dynamic_mxfp4_quant,
)
from aiter.ops.triton.utils.types import e4m3_dtype
import aiter.ops.triton.utils._triton.arch_info as arch_info
from aiter.test_common import checkAllclose

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


def uniform_random(shape, start=0, end=1, dtype=None, device=None):
    return (end - start) * torch.rand(shape, dtype=dtype, device=device) + start


def generate_data(
    seq_lens,
    num_blocks=32768,
    block_size=32,
    head_size=64,
    num_heads=(16, 2),
    sliding_window=None,
    q_dtype=torch.bfloat16,
    kv_dtype=torch.bfloat16,
    out_dtype=torch.bfloat16,
    shuffled_kv_cache=False,
    use_q_descale=None,
    use_kv_descale=None,
    use_out_scale=False,
    device="cpu",
):
    torch.manual_seed(0)
    num_seqs = len(seq_lens)
    query_lens = [x[0] for x in seq_lens]
    kv_lens = [x[1] for x in seq_lens]
    num_query_heads = num_heads[0]
    num_kv_heads = num_heads[1]
    assert num_query_heads % num_kv_heads == 0
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
        # NVFP4 query: the kernel consumes packed fp4 + scales, the reference uses e4m3.
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
    value_cache = torch.randn_like(key_cache)
    if kv_dtype == torch.uint8:
        # NVFP4 KV cache: kernel consumes packed+shuffled cache, reference uses e4m3.
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
    values = torch.arange(0, total_ind_count, dtype=torch.int)
    values = values[torch.randperm(total_ind_count)]
    block_tables = values.view(num_seqs, max_num_blocks_per_seq).contiguous().to(device)

    sinks = torch.randn(num_query_heads, dtype=torch.float32, device=device)

    output = torch.empty(
        sum(query_lens), num_query_heads, head_size, dtype=out_dtype, device=device
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
    sliding_window: Optional[int] = None,
    soft_cap: Optional[float] = None,
    sinks: Optional[torch.Tensor] = None,
    q_descale: Optional[torch.Tensor] = None,
    k_descale: Optional[torch.Tensor] = None,
    v_descale: Optional[torch.Tensor] = None,
    output_scale: Optional[torch.Tensor] = None,
    causal: int = 1,
) -> torch.Tensor:
    num_seqs = len(query_lens)
    block_tables = block_tables.cpu().numpy()
    _, block_size, num_kv_heads, head_size = key_cache.shape
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
        # skip NVFP4 KV cache for now as ds_load_tr4 is not yet supported
        # (e4m3_dtype, torch.uint8, torch.bfloat16, 128, False),
        # (torch.uint8, torch.uint8, torch.bfloat16, 128, False),
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
    sliding_window: Optional[int],
    block_size: int,
    soft_cap: Optional[float],
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
        # gfx950 -> Triton
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
                f"Skipping test for KV cache LDS required memory = {kv_cache_shared_mem_size/1024} kB > 320 kB"
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
@pytest.mark.parametrize("head_size", [64, 128])
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
    out_dtype: torch.dtype,
    use_q_descale: bool,
    use_kv_descale: bool,
    use_out_scale: bool,
    shuffled_kv_cache: bool,
) -> None:
    use_gluon_2d = is_2d_gluon_available(q_dtype, kv_dtype, soft_cap, False, False)
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
        torch.testing.assert_close(
            output, ref_output, atol=atol, rtol=rtol
        ), f"{torch.max(torch.abs(output - ref_output))}"
