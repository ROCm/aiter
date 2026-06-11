# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inlined, self-contained port of `test_triton_unified_attn` from
aiter/op_tests/triton_tests/attention/test_unified_attention.py.

It targets the LOCAL gluon kernel in `unified_attention_w_3d.py` (not the aiter
package). All helpers (`shuffle_kv_cache`, `generate_data`, `ref_paged_attn`)
are inlined so the test has no dependency on the aiter op_tests harness.

The aiter `unified_attention` dispatches between a 2d gluon kernel and a triton
fallback. The local module implements ONLY the 2d gluon kernel, so configs the
gluon path doesn't cover (mixed q/kv dtype, softcap, non-gfx1250) are skipped
via the same `is_2d_gluon_available` predicate the upstream uses to route them.
"""

import os
import sys
from typing import Optional

import pytest
import torch

#sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from aiter.ops.triton.utils.types import e4m3_dtype  # noqa: E402
import aiter.ops.triton.utils._triton.arch_info as arch_info  # noqa: E402
from unified_attention_w_3d import unified_attention  # noqa: E402

DEVICE_ARCH = arch_info.get_arch()
IS_DEVICE_ARCH_GFX12 = DEVICE_ARCH in ("gfx1250",)


def is_2d_gluon_available(q_dtype, kv_cache_dtype, softcap, use_qq_bias, use_alibi_slopes):
    """Inlined copy of aiter's predicate. The local module IS the 2d gluon
    kernel, so this also decides which configs it can run at all."""
    return (
        IS_DEVICE_ARCH_GFX12
        and not softcap
        and not use_qq_bias
        and not use_alibi_slopes
        and q_dtype != torch.uint8
        and kv_cache_dtype != torch.uint8
        and q_dtype == kv_cache_dtype
    )


def shuffle_kv_cache(key_cache: torch.Tensor, value_cache: torch.Tensor):
    """Shuffle key/value cache layout for optimized memory access.

    layout: (num_lanes, num_elements_per_thread)
        gfx1250: (16, 8) for BF16 and FP8.
    WMMA instruction shape: BF16 16x16x32, FP8 16x16x64.
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
        -1, num_kv_heads, head_size // k_width, k_width, block_size
    )
    key_cache_shuffled = key_cache_shuffled.permute(0, 1, 2, 4, 3).contiguous()

    value_cache_shuffled = value_cache.view(
        -1, block_size, num_kv_heads, head_size
    ).permute(0, 2, 1, 3)
    value_cache_shuffled = value_cache_shuffled.view(
        -1, num_kv_heads, block_size // k_width, k_width, head_size
    )
    value_cache_shuffled = value_cache_shuffled.permute(0, 1, 2, 4, 3).contiguous()

    return key_cache_shuffled, value_cache_shuffled


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
    """bf16 / e4m3 paged-attention inputs. Trimmed from the aiter generate_data
    (the nvfp4 / torch.uint8 quant path is dropped — unused by this test)."""
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
    ).to(q_dtype)

    # ---- kv cache ----
    key_cache = torch.randn(
        num_blocks, block_size, num_kv_heads, head_size, dtype=torch.float32, device=device
    )
    value_cache = torch.randn_like(key_cache)
    key_cache_orig = key_cache.to(kv_dtype)
    value_cache_orig = value_cache.to(kv_dtype)
    if shuffled_kv_cache:
        key_cache, value_cache = shuffle_kv_cache(key_cache_orig, value_cache_orig)
    else:
        key_cache, value_cache = key_cache_orig, value_cache_orig

    cu_query_lens = torch.tensor(
        [0] + query_lens, dtype=torch.int32, device=device
    ).cumsum(dim=0, dtype=torch.int32)
    kv_lens_t = torch.tensor(kv_lens, dtype=torch.int32, device=device)

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
    q_descale = k_descale = v_descale = output_scale = None
    if use_q_descale:
        q_descale = uniform_random(1, 1e-4, 1.0, torch.float32, device)
    if use_kv_descale:
        k_descale = uniform_random(1, 1e-4, 1.0, torch.float32, device)
        v_descale = uniform_random(1, 1e-4, 1.0, torch.float32, device)
    if use_out_scale:
        output_scale = 1.0 / uniform_random(1, 1e-4, 1.0, torch.float32, device)

    return (
        query,
        key_cache_orig,
        value_cache_orig,
        key_cache,
        value_cache,
        sinks,
        output,
        cu_query_lens,
        kv_lens_t,
        max_query_len,
        max_kv_len,
        scale,
        window_size,
        block_tables,
        q_descale,
        k_descale,
        v_descale,
        output_scale,
    )


def ref_paged_attn(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    query_lens: list,
    kv_lens: list,
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
    outputs = []
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

        k = key_cache[block_indices].view(-1, num_kv_heads, head_size)[:kv_len]
        v = value_cache[block_indices].view(-1, num_kv_heads, head_size)[:kv_len]

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
        [(1, 128), (1, 77), (1, 397)]
    ],
)
@pytest.mark.parametrize("num_heads", [(8, 1)])
@pytest.mark.parametrize("head_size", [64, 128])
@pytest.mark.parametrize("block_size", [64, ])
@pytest.mark.parametrize("sliding_window", [None, 256])
@pytest.mark.parametrize("soft_cap", [None,])
@pytest.mark.parametrize("num_blocks", [2048,])
@pytest.mark.parametrize(
    "q_dtype, kv_dtype, out_dtype, use_q_descale, use_kv_descale, use_out_scale",
    [
        (torch.bfloat16, torch.bfloat16, torch.bfloat16, False, False, False),
        #(torch.bfloat16, e4m3_dtype, torch.bfloat16, False, True, False),
        (e4m3_dtype, e4m3_dtype, torch.bfloat16, True, True, False),
    ],
)
@pytest.mark.parametrize("shuffled_kv_cache", [True, False])
@torch.inference_mode()
def test_triton_unified_attn(
    seq_lens: list,
    num_heads: tuple,
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
    # The local module implements ONLY the 2d gluon kernel; skip configs that
    # upstream would route to the (unavailable here) triton fallback.
    if not use_gluon_2d:
        pytest.skip("local module only implements the 2d gluon kernel for this config")

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
        device="cpu",
    )

    def to_cuda(t):
        return t.to("cuda") if t is not None else None

    output = to_cuda(output)
    unified_attention(
        q=to_cuda(query),
        k=to_cuda(key_cache),
        v=to_cuda(value_cache),
        out=output,
        cu_seqlens_q=to_cuda(cu_query_lens),
        seqused_k=to_cuda(kv_lens),
        max_seqlen_q=max_query_len,
        max_seqlen_k=max_kv_len,
        softmax_scale=scale,
        causal=True,
        window_size=window_size,
        block_table=to_cuda(block_tables),
        softcap=soft_cap if soft_cap is not None else 0,
        q_descale=to_cuda(q_descale),
        k_descale=to_cuda(k_descale),
        v_descale=to_cuda(v_descale),
        sinks=to_cuda(sinks),
        output_scale=to_cuda(output_scale),
        shuffled_kv_cache=shuffled_kv_cache,
    )

    # The reference runs on CPU using the unshuffled KV
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
    output = output.to(torch.float32).cpu()
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
