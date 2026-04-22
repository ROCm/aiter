# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from re import T
from typing import Optional

import pytest
import torch

from unified_attention_2d_merged import (
    unified_attention as gluon_unified_attention_2d,
)
from aiter.ops.triton.utils.types import e4m3_dtype
import aiter.ops.triton.utils._triton.arch_info as arch_info

DEVICE_ARCH = arch_info.get_arch()

NUM_HEADS = [(64, 8), (8, 1)]
HEAD_SIZES = [64, 128]
BLOCK_SIZES = [16, 64]


DTYPES = [torch.bfloat16]
QDTYPES = [None]
# one value large enough to test overflow in index calculation.
# one value small enough to test the schema op check
NUM_BLOCKS = [
    4096,
]
SLIDING_WINDOWS = [None]

IS_DEVICE_ARCH_GFX12 = DEVICE_ARCH == "gfx1250" 


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

    if IS_DEVICE_ARCH_GFX12:
        if dtype == torch.bfloat16:
            layout = (16, 8)
        else:
            # Caution: in gfx1250, the 16-bit and 8-bit layout should both be (16, 8), however, in order to enable ds_load_b128 for 8-bit WMMA,
            # we use (16, 16) here, noted that you must set k_width to 16 in the corresponding DotOperandLayout, the math will be equivalent.
            layout = (16, 16)
    else:
        if dtype == torch.bfloat16:
            layout = (16, 8)
        else:
            layout = (16, 16)

    num_blocks, block_size, num_kv_heads, head_size = key_cache.shape
    num_blocks_v, block_size_v, num_kv_heads_v, head_size_v = value_cache.shape
    assert block_size >= 16
    assert num_blocks == num_blocks_v
    assert num_kv_heads == num_kv_heads_v
    assert head_size == head_size_v
    assert block_size == block_size_v

    num_lanes, num_elements_per_thread = layout
    key_cache_shuffled = key_cache.view(
        -1, block_size, num_kv_heads, head_size
    ).permute(0, 2, 1, 3)
    key_cache_shuffled = key_cache_shuffled.view(
        -1,
        num_kv_heads,
        block_size // num_lanes,
        num_lanes,
        head_size // (2 * num_elements_per_thread),
        2,  # there are 2 groups of threads, t0 ~ t15 and t16 ~ t31
        num_elements_per_thread,
    )
    key_cache_shuffled = key_cache_shuffled.permute(0, 1, 2, 4, 5, 3, 6).contiguous()
    key_cache_shuffled = key_cache_shuffled.view(
        -1, num_kv_heads, block_size // 16, head_size * 16
    )

    value_cache_shuffled = value_cache.view(
        -1, block_size, num_kv_heads, head_size
    ).permute(0, 2, 1, 3)
    value_cache_shuffled = value_cache_shuffled.view(
        -1,
        num_kv_heads,
        block_size // (2 * num_elements_per_thread),
        2,
        num_elements_per_thread,
        head_size // num_lanes,
        num_lanes,
    )
    value_cache_shuffled = value_cache_shuffled.permute(
        0, 1, 5, 2, 3, 6, 4
    ).contiguous()
    value_cache_shuffled = value_cache_shuffled.view(
        -1, num_kv_heads, head_size // 16, block_size * 16
    )

    return key_cache_shuffled, value_cache_shuffled


def generate_data(
    seq_lens,
    num_blocks=32768,
    block_size=32,
    head_size=64,
    num_heads=(16, 2),
    sliding_window=None,
    q_dtype=torch.float8_e4m3fn,
    kv_dtype=torch.bfloat16,
    shuffled_kv_cache=False,
    remove_indirect_access=False,
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

    query = torch.randn(
        sum(query_lens), num_query_heads, head_size, dtype=torch.float32, device="cpu"
    ).to(q_dtype)
    key_cache = torch.randn(
        num_blocks,
        block_size,
        num_kv_heads,
        head_size,
        dtype=torch.float32,
        device="cpu",
    )
    value_cache_orig = torch.randn_like(key_cache).to(kv_dtype)
    key_cache_orig = key_cache.to(kv_dtype)
    cu_query_lens = torch.tensor(
        [0] + query_lens, dtype=torch.int32, device="cpu"
    ).cumsum(dim=0, dtype=torch.int32)
    kv_lens = torch.tensor(kv_lens, dtype=torch.int32, device="cpu")

    max_num_blocks_per_seq = (max_kv_len + block_size - 1) // block_size
    max_num_blocks_per_seq = (
        min(max_num_blocks_per_seq * num_seqs, num_blocks) // num_seqs
    )
    total_ind_count = num_seqs * max_num_blocks_per_seq
    values = torch.arange(0, total_ind_count, dtype=torch.int)
    values = values[torch.randperm(total_ind_count)]
    block_tables = values.view(num_seqs, max_num_blocks_per_seq).contiguous().to("cpu")
    if remove_indirect_access:
        inds = torch.arange(max_num_blocks_per_seq)
        block_tables[:] = inds


    sinks = torch.randn(num_query_heads, dtype=torch.float32, device="cpu")

    output = torch.empty_like(query)

    q_descale = None
    k_descale = None
    v_descale = None
    output_scale = None
    if q_dtype == e4m3_dtype:
        q_descale = torch.tensor(1.0).to("cpu")
        output_scale = torch.tensor(1.0).to("cpu")
    if kv_dtype == e4m3_dtype:
        k_descale = torch.tensor(1.0).to("cpu")
        v_descale = torch.tensor(1.0).to("cpu")

    if shuffled_kv_cache:
        key_cache, value_cache = shuffle_kv_cache(key_cache_orig, value_cache_orig)
    else:
        key_cache, value_cache = key_cache_orig, value_cache_orig

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

    return out.to(orig_dtype)


@pytest.mark.parametrize(
    "seq_lens",
    [
        [
            (512, 512),
        ],
        [(567, 275), (34, 345)],
        [(1, 411), (12, 701), (1,456), (1, 133), (2, 343)],
        [(777, 777)],
    ],
)
@pytest.mark.parametrize(
    "num_heads",
    [
        [8, 1],
    ],
)
@pytest.mark.parametrize(
    "head_size",
    [
        64,
    ],
)
@pytest.mark.parametrize(
    "block_size",
    [
        64,
    ],
)
@pytest.mark.parametrize(
    "sliding_window",
    [
        None,
    ],
)
@pytest.mark.parametrize(
    "dtype",
    [
        torch.bfloat16,
    ],
)
@pytest.mark.parametrize(
    "soft_cap",
    [
        None,
    ],
)
@pytest.mark.parametrize("num_blocks", NUM_BLOCKS)
@pytest.mark.parametrize(
    "q_dtype",
    [
        torch.bfloat16,
    ],
)
@torch.inference_mode()
@pytest.mark.parametrize(
    "use_tdm, num_kv_blocks",
    [
        # (False, 1),
        (True, 1),
        # (True, 4),
    ],
)
@pytest.mark.parametrize(
    "shuffled_kv_cache",
    [
        True, False,
    ],
)
@pytest.mark.parametrize(
    "check_ref",
    [
        True,
    ],
)
@pytest.mark.parametrize(
    "loop_variant",
    [
        3,
    ],
)
@torch.inference_mode()
def test_gluon_unified_attn_2d_noncausal(
    seq_lens: list[tuple[int, int]],
    num_heads: tuple[int, int],
    head_size: int,
    sliding_window: Optional[int],
    dtype: torch.dtype,
    block_size: int,
    soft_cap: Optional[float],
    num_blocks: int,
    q_dtype: Optional[torch.dtype],
    use_tdm: bool,
    num_kv_blocks: int,
    shuffled_kv_cache: bool,
    check_ref: bool,
    loop_variant: int,
) -> None:

    causal = False

    if DEVICE_ARCH not in (
        "gfx950",
        "gfx1250",
    ):
        pytest.skip(f"{DEVICE_ARCH} is not supported")
    if DEVICE_ARCH not in ("gfx1250",) and use_tdm == True:
        pytest.skip(f"{DEVICE_ARCH} does not have TDM")
    if num_kv_blocks > 1 and DEVICE_ARCH not in ("gfx1250",):
        pytest.skip(f"{DEVICE_ARCH} does not have TDM gather")
    if q_dtype is not None and q_dtype.itemsize < 2 and block_size < 32:
        pytest.skip("block size must be at least 32 for fp8")
    if shuffled_kv_cache and (not use_tdm or num_kv_blocks != 1):
        pytest.skip(
            "Shuffled KV cache is only supported with TDM and num_kv_blocks == 1"
        )
    torch.manual_seed(0)
    num_seqs = len(seq_lens)
    query_lens = [x[0] for x in seq_lens]
    kv_lens = [x[1] for x in seq_lens]
    num_query_heads = num_heads[0]
    num_kv_heads = num_heads[1]
    assert num_query_heads % num_kv_heads == 0
    num_buffers = 3
    if loop_variant == 0:
        num_buffers = 2
    # NOTE: for now, skip paged access
    remove_indirect_access=False

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
        seq_lens,
        num_blocks=num_blocks,
        block_size=block_size,
        head_size=head_size,
        num_heads=(num_query_heads, num_kv_heads),
        sliding_window=sliding_window,
        q_dtype=q_dtype,
        kv_dtype=dtype,
        shuffled_kv_cache=shuffled_kv_cache,
        remove_indirect_access=remove_indirect_access, 
    )
    output_cuda = output.cuda()
    gluon_unified_attention_2d(
        q=query.cuda(),
        k=key_cache.cuda(),
        v=value_cache.cuda(),
        out=output_cuda,
        cu_seqlens_q=cu_query_lens.cuda(),
        seqused_k=kv_lens.cuda(),
        max_seqlen_q=max_query_len,
        max_seqlen_k=max_kv_len,
        softmax_scale=scale,
        causal=causal,
        window_size=window_size,
        block_table=block_tables.cuda(),
        softcap=soft_cap if soft_cap is not None else 0,
        q_descale=q_descale.cuda() if q_descale is not None else None,
        k_descale=k_descale.cuda() if k_descale is not None else None,
        v_descale=v_descale.cuda() if v_descale is not None else None,
        output_scale=output_scale.cuda() if output_scale is not None else None,
        sinks=sinks.cuda() if sinks is not None else None,
        new_kv_layout=num_kv_blocks > 1,
        num_kv_blocks=num_kv_blocks,
        use_tdm=use_tdm,
        shuffled_kv_cache=shuffled_kv_cache,
        remove_indirect_access=remove_indirect_access,
        loop_variant=loop_variant,
        num_buffers=num_buffers,
    )
    ref_output = ref_paged_attn(
        query=query,
        key_cache=key_cache_orig,
        value_cache=value_cache_orig,
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
        output_scale=output_scale,
        causal=causal,
    )
    atol, rtol = 1.5e-2, 1e-2
    if q_dtype is not None:
        atol, rtol = 1.5e-1, 1.5e-1
    output = output_cuda.cpu().to(torch.float32)
    ref_output = ref_output.cpu().to(torch.float32)
    torch.testing.assert_close(
        output, ref_output, atol=atol, rtol=rtol
    ), f"{torch.max(torch.abs(output - ref_output))}"
