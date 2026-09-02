# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from types import SimpleNamespace

import pytest
import torch

from aiter.ops.triton.attention.unified_attention import (
    _fasttile_modes,
    _is_gluon_available,
    is_2d_gluon_available,
    unified_attention,
    unified_attention_get_fasttile,
    unified_attention_set_fasttile,
    use_2d_kernel,
)
from aiter.ops.triton.utils._triton import arch_info
from aiter.ops.triton.utils.device_info import get_num_sms
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


# ---------------------------------------------------------------------------
# fasttile: the restructured 2-D prefill tile loop
#
# The schedule only touches the 2-D launch, and `use_2d_kernel` is picky: the
# suite's usual shapes route to the 3-D kernel, where a fasttile test would
# pass without executing a single restructured tile. Every shape below is
# therefore checked with `_require_2d_kernel` before it is trusted.
# ---------------------------------------------------------------------------

# "short" takes the 2-D path deterministically via the max_seqlen_k <= 512
# rule and still resolves to the BLOCK_M=128 prefill entry, so the bulk loop
# runs several tiles. "long" is the production geometry: it qualifies on
# program count on any gfx950 part. "ragged" and "head512" use head_size 512,
# which always takes the 2-D path.
#
# "ragged" is a regression shape: its (567, 275) entry carries more query
# tokens than cached keys, which makes context_len negative. A `bulk_end` that
# is not clamped up to `tile_start` then floors below zero and hands the masked
# loop a negative first tile.
_RAGGED_SEQ_LENS = [
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
]

_FASTTILE_SHAPES = {
    "short": {
        "seq_lens": [(512, 512)],
        "num_blocks": 512,
        "block_size": 64,
        "head_size": 64,
        "num_heads": (64, 8),
        "device": "cuda",
    },
    "long": {
        "seq_lens": [(4096, 16384)],
        "num_blocks": 512,
        "block_size": 64,
        "head_size": 64,
        "num_heads": (64, 8),
        "device": "cuda",
    },
    "ragged": {
        "seq_lens": _RAGGED_SEQ_LENS,
        "num_blocks": 2048,
        "block_size": 16,
        "head_size": 512,
        "num_heads": (8, 8),
        "device": "cuda",
    },
    "head512": {
        "seq_lens": [(512, 512)],
        "num_blocks": 2048,
        "block_size": 16,
        "head_size": 512,
        "num_heads": (8, 8),
        "device": "cuda",
    },
}

# bf16 carries 8 mantissa bits, so one ULP at magnitude m is m * 2**-8. The
# output dtype is bf16 for every case here, including the fp8 ones, so this is
# the spacing that governs regardless of how q and the cache are stored.
_BF16_ULP = 2**-8

# The gate does not discriminate on dtype, so the fp8 prefill path takes the
# restructured loop too and is covered here rather than excluded from it.
_FASTTILE_DTYPES = (
    ("bf16", torch.bfloat16, torch.bfloat16),
    ("fp8", e4m3_dtype, e4m3_dtype),
)


@pytest.fixture
def fasttile():
    """Select a fasttile schedule for one test, restoring it afterwards."""
    original = unified_attention_get_fasttile()
    yield unified_attention_set_fasttile
    unified_attention_set_fasttile(original)


def _require_2d_kernel(shape, sliding_window=0):
    """Skip rather than silently pass if this machine routes the shape to 3-D."""
    num_heads = shape["num_heads"]
    max_q = max(x[0] for x in shape["seq_lens"])
    max_k = max(x[1] for x in shape["seq_lens"])
    num_tokens = sum(x[0] for x in shape["seq_lens"])
    # BLOCK_Q is bounded below by 1, so this over-counts programs at most by
    # the tuned BLOCK_M; it only feeds the "enough programs" branch below.
    num_2d_prgms = (num_tokens + len(shape["seq_lens"])) * num_heads[1]
    params = SimpleNamespace(
        head_size=shape["head_size"],
        sliding_window=sliding_window,
        all_decode=False,
        max_seqlen_q=max_q,
        max_seqlen_k=max_k,
        num_2d_prgms=num_2d_prgms,
        target_num_prgms=get_num_sms() * 4,
    )
    if not use_2d_kernel(params):
        pytest.skip("shape routes to the 3-D kernel on this machine")


def _run_unified_attention(data, shuffled_kv_cache=False):
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
        q_descale,
        k_descale,
        v_descale,
        output_scale,
    ) = data
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
        shuffled_kv_cache=shuffled_kv_cache,
        backend="triton",
    )
    return output.clone()


def _drop_sinks(data):
    data = list(data)
    data[5] = None
    return tuple(data)


def _assert_ulp_close(stock, other, label, max_ulps=4.0):
    """Assert `other` is within a few bf16 ULP of the stock kernel, elementwise.

    The tolerance is deliberately mixed, because neither half works alone:

    * ``rtol`` is the real contract -- each element must stay within `max_ulps`
      of *its own* magnitude. A tensor-wide budget derived from max|out| would
      let a small element drift thousands of its own ULPs unnoticed.
    * ``atol`` is one ULP of the accumulation scale, and it is not slack. An
      output element is a softmax-weighted sum of V, so its absolute error is
      bounded by the scale of that sum, not by the element. Elements that come
      out near zero do so by cancellation between O(max|out|) terms and retain
      no significant digits, so a relative bound on them is meaningless. Only
      such elements fall through to this floor; everything at a magnitude worth
      measuring is governed by rtol.

    Degenerate rows (a query position with no keys in range and no sink) are
    NaN in the *stock* kernel too. ``equal_nan`` makes the NaN pattern part of
    the contract: NaN against a number fails from either side.
    """
    finite = stock[~torch.isnan(stock)]
    scale = finite.abs().max().item() if finite.numel() else 0.0
    torch.testing.assert_close(
        other,
        stock,
        rtol=max_ulps * _BF16_ULP,
        atol=scale * _BF16_ULP,
        equal_nan=True,
        msg=lambda m: f"{label}: {m}",
    )


@pytest.mark.skipif(DEVICE_ARCH != "gfx950", reason="fasttile is gfx950-only")
@pytest.mark.parametrize("shape_key", sorted(_FASTTILE_SHAPES))
@pytest.mark.parametrize("dtypes", _FASTTILE_DTYPES, ids=lambda d: d[0])
@pytest.mark.parametrize("mode", ["nofuse", "on"])
@pytest.mark.parametrize("use_sinks", [False, True])
@torch.inference_mode()
def test_fasttile_matches_stock(fasttile, shape_key, dtypes, mode, use_sinks) -> None:
    """The restructured tile loop must stay within a bf16 ULP of the stock path.

    Comparing against `ref_paged_attn` at the suite's atol would not catch a
    real bulk/tail-boundary bug: the bf16 quantisation floor is two orders of
    magnitude larger than the difference the restructure is allowed to
    introduce. So this pins the restructure against the *stock kernel*.
    """
    shape = _FASTTILE_SHAPES[shape_key]
    _require_2d_kernel(shape)

    _, q_dtype, kv_dtype = dtypes
    data = generate_data(q_dtype=q_dtype, kv_dtype=kv_dtype, **shape)
    if not use_sinks:
        data = _drop_sinks(data)

    # Both states are set explicitly, so this holds however the environment
    # variable happened to be set for the run.
    fasttile("off")
    stock = _run_unified_attention(data).float()

    fasttile(mode)
    fast = _run_unified_attention(data).float()

    _assert_ulp_close(stock, fast, f"fasttile({mode}, {dtypes[0]}, {shape_key})")


@pytest.mark.skipif(DEVICE_ARCH != "gfx950", reason="fasttile is gfx950-only")
@pytest.mark.parametrize("shape_key", sorted(_FASTTILE_SHAPES))
@pytest.mark.parametrize("mode", ["nofuse", "on"])
@torch.inference_mode()
def test_fasttile_matches_reference(fasttile, shape_key, mode) -> None:
    """Both modes must still satisfy the suite's accuracy contract."""
    shape = _FASTTILE_SHAPES[shape_key]
    _require_2d_kernel(shape)
    data = generate_data(**shape)
    ref_output = ref_paged_attn(
        query=data[0],
        key_cache=data[1],
        value_cache=data[2],
        query_lens=[x[0] for x in shape["seq_lens"]],
        kv_lens=[x[1] for x in shape["seq_lens"]],
        block_tables=data[13],
        scale=data[11],
        out_dtype=torch.bfloat16,
        sinks=data[5],
        q_descale=data[16],
        k_descale=data[17],
        v_descale=data[18],
        output_scale=data[19],
    ).float()

    fasttile(mode)
    out = _run_unified_attention(data).float()
    torch.testing.assert_close(out, ref_output, atol=1.5e-2, rtol=1e-2)


@pytest.mark.parametrize("all_decode", [True, False])
@pytest.mark.parametrize("sliding_window", [0, 256])
@pytest.mark.parametrize("mode", ["nofuse", "on"])
def test_fasttile_gate_declines_uncertified_paths(
    fasttile, all_decode, sliding_window, mode
) -> None:
    """The restructure must stay off for decode and for windowed layers.

    The bulk loop drops the sliding-window bound as well as the causal one, so
    a gate that let a windowed layer through would be silently wrong rather
    than merely uncertified. On any architecture but gfx950 the gate declines
    outright, which is what makes this safe to enable globally.
    """
    fasttile(mode)
    split, fuse = _fasttile_modes(
        SimpleNamespace(
            sliding_window=sliding_window,
            all_decode=all_decode,
            softmax_scale=0.125,
        )
    )
    expected = DEVICE_ARCH == "gfx950" and not all_decode and sliding_window <= 0
    assert split is expected
    assert fuse is (expected and mode == "on")


@pytest.mark.parametrize("softmax_scale", [-0.125, 0.0, 0.125])
def test_fasttile_fold_requires_positive_scale(fasttile, softmax_scale) -> None:
    """The fold must decline a non-positive scale, keeping the split.

    It takes the maximum of the unscaled logits and scales that, which is the
    true maximum only because scaling by a positive constant is monotone. Under
    a negative scale it would pick the minimum and hand exp2 large positive
    arguments; the stock path takes the maximum of the already-scaled logits
    and is stable either way.
    """
    fasttile("on")
    split, fuse = _fasttile_modes(
        SimpleNamespace(sliding_window=0, all_decode=False, softmax_scale=softmax_scale)
    )
    on_gfx950 = DEVICE_ARCH == "gfx950"
    assert split is on_gfx950
    assert fuse is (on_gfx950 and softmax_scale > 0)


def test_fasttile_rejects_unknown_mode(fasttile) -> None:
    """An unusable value must fail loudly rather than silently mean 'off'."""
    with pytest.raises(ValueError, match="Invalid fasttile value"):
        fasttile("fastest")


@pytest.mark.skipif(DEVICE_ARCH != "gfx950", reason="fasttile is gfx950-only")
@pytest.mark.parametrize("dtypes", _FASTTILE_DTYPES, ids=lambda d: d[0])
@pytest.mark.parametrize("mode", ["nofuse", "on"])
@torch.inference_mode()
def test_fasttile_matches_stock_shuffled_kv(fasttile, dtypes, mode) -> None:
    """The restructure must hold for the pre-shuffled cache layout too.

    Shuffled pages address K and V through a different branch of the tile body
    -- the block table is indexed once per tile rather than per token, because
    a shuffled tile is exactly one page -- so the bulk pass exercises code the
    non-shuffled shapes never reach. The rest of the suite skips shuffled off
    gfx1250, which left this branch uncovered on the arch the schedule targets.
    """
    _, q_dtype, kv_dtype = dtypes
    shape = dict(_FASTTILE_SHAPES["short"])
    _require_2d_kernel(shape)
    data = generate_data(
        q_dtype=q_dtype, kv_dtype=kv_dtype, shuffled_kv_cache=True, **shape
    )

    fasttile("off")
    stock = _run_unified_attention(data, shuffled_kv_cache=True).float()

    fasttile(mode)
    fast = _run_unified_attention(data, shuffled_kv_cache=True).float()

    _assert_ulp_close(stock, fast, f"fasttile({mode}, {dtypes[0]}, shuffled)")
