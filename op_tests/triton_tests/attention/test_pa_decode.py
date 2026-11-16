# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import random

import pytest
import torch
import triton
import triton.language as tl

import aiter.ops.triton.attention.pa_decode as pa_decode_mod
from aiter import logger, pertoken_quant
from aiter.ops.triton.attention.pa_decode import paged_attention_decode
from aiter.ops.triton.utils._triton.arch_info import get_arch
from aiter.ops.triton.utils.config_utils import load_config_json, resolve_config_dir


def paged_attention_decode_ref(
    output,  # [num_seqs, num_q_heads, head_sz]
    query,  # [num_seqs, num_kv_heads, head_sz],
    k_cache,  # [num_seqs, num_kv_heads, head_sz/x, blk_sz, x]
    v_cache,  # [num_seqs, num_kv_heads, head_sz, blk_sz]
    blk_tables,  # [num_seq, max_num_blks_per_seq]
    ctx_lens,  # [num_seqs]
) -> None:
    num_q_heads = query.shape[1]
    num_kv_heads = v_cache.shape[1]
    q_grp_sz = num_q_heads // num_kv_heads
    head_sz = v_cache.shape[2]
    kv_blk_sz = v_cache.shape[3]

    num_seqs = query.shape[0]
    for s in range(num_seqs):
        q = query[s].unsqueeze(0)
        blk_tbl = blk_tables[s]
        ctx_len = ctx_lens[s]

        keys = []
        values = []
        for j in range(ctx_len):
            blk_number = int(blk_tbl[j // kv_blk_sz])
            blk_offset = j % kv_blk_sz

            k = k_cache[blk_number, :, :, blk_offset, :]
            k = k.reshape(num_kv_heads, head_sz)
            if q_grp_sz != 1:
                k = k.repeat_interleave(q_grp_sz, 0)
            keys.append(k)

            v = v_cache[blk_number, :, :, blk_offset]
            if q_grp_sz != 1:
                v = v.repeat_interleave(q_grp_sz, 0)
            values.append(v)

        keys = torch.stack(keys, dim=0)
        values = torch.stack(values, dim=0)

        scale = 1.0 / (head_sz**0.5)
        q = q * scale
        attn = torch.einsum("qhd,khd->hqk", q, keys)
        attn = torch.softmax(attn, dim=-1)
        out = torch.einsum("hqk, khd->qhd", attn, values)
        out = out.view(num_q_heads, head_sz)
        output[s].copy_(out, non_blocking=True)


tl_to_torch_dtype = {tl.bfloat16: torch.bfloat16, tl.float16: torch.float16}


def input_helper(
    B,
    H_Q,
    H_KV,
    D,
    KV_BLK_SZ,
    SEQ_LEN,
    dtype,
    kv_cache_dtype,
    output_type,
    num_blocks=4,
    random_seed: int = 0,
):
    """Helper function to generate input tensors for paged attention testing."""
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    random.seed(random_seed)

    # Query tensor generation
    if dtype not in (torch.bfloat16, torch.float16, torch.float32):
        query = torch.randn(
            B, H_Q, D, dtype=torch.float16, device="cuda"
        )  # assumption dtype is 8bits or lower
        query = query.to(dtype=dtype)
    else:
        query = torch.randn(B, H_Q, D, dtype=dtype, device="cuda")

    if kv_cache_dtype not in (torch.bfloat16, torch.float16, torch.float32):
        x = min(
            D, 16 // torch.tensor([], dtype=torch.float16, device="cuda").element_size()
        )
        key_cache = torch.randn(
            num_blocks, H_KV, D // x, KV_BLK_SZ, x, dtype=torch.float16, device="cuda"
        )
        value_cache = torch.randn(
            num_blocks, H_KV, D, KV_BLK_SZ, dtype=torch.float16, device="cuda"
        )
        key_cache = torch.clamp(
            key_cache, min=1e-3
        )  # For FP8 case, this is needed to prevent NANs
        value_cache = torch.clamp(
            value_cache, min=1e-3
        )  # For FP8 case, this is needed to prevent NANs

        # torch doesn't have randn for fp8 data type, so we convert here
        key_cache = key_cache.to(dtype=kv_cache_dtype)
        value_cache = value_cache.to(dtype=kv_cache_dtype)
    else:
        x = min(
            D,
            16 // torch.tensor([], dtype=kv_cache_dtype, device="cuda").element_size(),
        )
        key_cache = torch.randn(
            num_blocks, H_KV, D // x, KV_BLK_SZ, x, dtype=kv_cache_dtype, device="cuda"
        )
        value_cache = torch.randn(
            num_blocks, H_KV, D, KV_BLK_SZ, dtype=kv_cache_dtype, device="cuda"
        )
        key_cache = torch.clamp(
            key_cache, min=1e-3
        )  # For FP8 case, this is needed to prevent NANs
        value_cache = torch.clamp(
            value_cache, min=1e-3
        )  # For FP8 case, this is needed to prevent NANs

    key_cache_tri = key_cache.permute(0, 1, 3, 2, 4).flatten(3, 4).contiguous()
    value_cache_tri = value_cache.permute(0, 1, 3, 2).contiguous()

    context_lens = torch.full((B,), SEQ_LEN, device="cuda")
    max_context_len = max(context_lens)
    max_num_blks_per_seq = (max_context_len + KV_BLK_SZ - 1) // KV_BLK_SZ

    block_tables = []
    for i in range(B):
        block_table = [
            random.randint(0, num_blocks - 1) for _ in range(max_num_blks_per_seq)
        ]
        block_tables.append(block_table)
    block_tables = torch.tensor(block_tables, dtype=torch.int32, device="cuda")

    output = torch.zeros(B, H_Q, D, dtype=output_type, device="cuda")

    return (
        query,
        output,
        key_cache,
        value_cache,
        key_cache_tri,
        value_cache_tri,
        context_lens,
        block_tables,
        max_context_len,
    )


@pytest.mark.parametrize("B", [1, 4])
@pytest.mark.parametrize("H_Q, H_KV", [(1, 1), (8, 1)])
@pytest.mark.parametrize("KV_BLK_SZ", [1, 16])
@pytest.mark.parametrize("SEQ_LEN", [1, 57, 1024])
@pytest.mark.parametrize("NUM_BLK", [4, 16])
# q_dtype, kv_dtype, compute_type, output_type
# INT8xINT8 -> BF16-> BF16
# FP8xFP8 -> BF16-> FP8
# BF16xINT8 -> BF16-> BF16
# BF16xFP8 -> BF16-> BF16
# BF16xBF16->BF16->BF16
# FP16xFP16->FP16->FP16
@pytest.mark.parametrize(
    "dtype, kv_cache_dtype, compute_type, output_type",
    [
        (torch.bfloat16, torch.bfloat16, tl.bfloat16, torch.bfloat16),
        (torch.bfloat16, torch.float8_e4m3fnuz, tl.bfloat16, torch.bfloat16),
        (torch.bfloat16, torch.int8, tl.bfloat16, torch.bfloat16),
        (torch.float8_e4m3fnuz, torch.float8_e4m3fnuz, tl.bfloat16, torch.bfloat16),
        (torch.int8, torch.int8, tl.bfloat16, torch.bfloat16),
    ],
)
def test_paged_attn(
    B,
    H_Q,
    H_KV,
    KV_BLK_SZ,
    SEQ_LEN,
    NUM_BLK,
    dtype,
    kv_cache_dtype,
    compute_type,
    output_type,
):

    head_size = 128

    if SEQ_LEN >= 8192 and B >= 16:
        pytest.skip("B>={4} and SEQ_LEN>={8192} tests are too slow")

    torch.cuda.empty_cache()  # Helps avoid hangs in large tests

    num_blocks = NUM_BLK

    (
        query,
        triton_output,
        key_cache,
        value_cache,
        key_cache_tri,
        value_cache_tri,
        context_lens,
        block_tables,
        max_context_len,
    ) = input_helper(
        B,
        H_Q,
        H_KV,
        head_size,
        KV_BLK_SZ,
        SEQ_LEN,
        dtype,
        kv_cache_dtype,
        output_type,
        num_blocks,
    )

    attn_scale = 1.0 / (head_size**0.5)

    paged_attention_decode(
        triton_output,
        query,
        key_cache_tri,
        value_cache_tri,
        context_lens,
        block_tables,
        attn_scale,
        max_context_len,
        compute_type,
        k_scale=torch.tensor([1.0], dtype=torch.float32, device="cuda"),
        v_scale=torch.tensor([1.0], dtype=torch.float32, device="cuda"),
        num_seq_partitions=0,
        alibi_slopes=None,
    )

    # torch doesn't have support for fp8 data type, so we convert here
    if dtype not in (torch.bfloat16, torch.float16, torch.float32):
        query = query.to(tl_to_torch_dtype[compute_type])

    if kv_cache_dtype not in (torch.bfloat16, torch.float16):
        key_cache = key_cache.to(dtype=tl_to_torch_dtype[compute_type])
        value_cache = value_cache.to(dtype=tl_to_torch_dtype[compute_type])
    torch_output = torch.zeros(B, H_Q, head_size, dtype=output_type, device="cuda")
    paged_attention_decode_ref(
        torch_output, query, key_cache, value_cache, block_tables, context_lens
    )

    torch.testing.assert_close(triton_output, torch_output, rtol=1e-02, atol=1e-02)


# test_paged_attn stops at SEQ_LEN=1024, a single 1024-token partition, so it
# only ever reaches the V1 kernels. SEQ_LEN=4096 spans four partitions, which
# dispatches to the V2 (partitioned + reduce) kernels; the spy fails the test
# if a dispatch change ever routes these cases back to V1.
@pytest.mark.parametrize("B", [1, 4])
@pytest.mark.parametrize("H_Q, H_KV", [(1, 1), (8, 1)])
@pytest.mark.parametrize(
    "dtype, kv_cache_dtype, compute_type, output_type",
    [
        (torch.bfloat16, torch.bfloat16, tl.bfloat16, torch.bfloat16),
        (torch.bfloat16, torch.float8_e4m3fnuz, tl.bfloat16, torch.bfloat16),
    ],
)
def test_paged_attn_v2(
    B, H_Q, H_KV, dtype, kv_cache_dtype, compute_type, output_type, monkeypatch
):
    # paged_attention_decode looks up paged_attn_decode_v2 in its module globals.
    v2_calls = []
    paged_attn_decode_v2 = paged_attention_decode.__globals__["paged_attn_decode_v2"]

    def spy(*args, **kwargs):
        v2_calls.append(True)
        return paged_attn_decode_v2(*args, **kwargs)

    monkeypatch.setitem(paged_attention_decode.__globals__, "paged_attn_decode_v2", spy)

    test_paged_attn(
        B,
        H_Q,
        H_KV,
        KV_BLK_SZ=16,
        SEQ_LEN=4096,
        NUM_BLK=16,
        dtype=dtype,
        kv_cache_dtype=kv_cache_dtype,
        compute_type=compute_type,
        output_type=output_type,
    )
    assert v2_calls, "SEQ_LEN=4096 was expected to dispatch to the V2 kernels"


@pytest.mark.parametrize("B, H_Q, H_KV", [(1, 8, 1), (4, 16, 16), (64, 16, 16)])
@pytest.mark.parametrize("SEQ_LEN", [2048, 8192])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_paged_attn_multi_partition_dispatch(B, H_Q, H_KV, SEQ_LEN, dtype, monkeypatch):
    """Check output and V1/V2 choice for bf16/fp16 caches above 1024 tokens."""
    head_size, kv_blk_sz = 128, 16
    compute_type = {torch.bfloat16: tl.bfloat16, torch.float16: tl.float16}[dtype]
    num_blocks = (SEQ_LEN + kv_blk_sz - 1) // kv_blk_sz
    (
        query,
        triton_output,
        key_cache,
        value_cache,
        key_cache_tri,
        value_cache_tri,
        context_lens,
        block_tables,
        max_context_len,
    ) = input_helper(
        B, H_Q, H_KV, head_size, kv_blk_sz, SEQ_LEN, dtype, dtype, dtype, num_blocks
    )

    called = []
    for name in ("paged_attn_decode_v1", "paged_attn_decode_v2"):
        fn = getattr(pa_decode_mod, name)
        monkeypatch.setattr(
            pa_decode_mod,
            name,
            lambda *a, _fn=fn, _name=name, **k: (called.append(_name), _fn(*a, **k))[1],
        )

    paged_attention_decode(
        triton_output,
        query,
        key_cache_tri,
        value_cache_tri,
        context_lens,
        block_tables,
        1.0 / head_size**0.5,
        int(max_context_len),
        compute_type,
        k_scale=torch.tensor([1.0], dtype=torch.float32, device="cuda"),
        v_scale=torch.tensor([1.0], dtype=torch.float32, device="cuda"),
    )

    num_partitions = -(-SEQ_LEN // pa_decode_mod._SEQ_PARTITION_SIZE)
    rule = pa_decode_mod._get_dispatch_config(
        {torch.bfloat16: "bf16", torch.float16: "fp16"}[dtype]
    )
    if rule is not None:
        expect_v1 = num_partitions <= rule["v1_max_partitions"]
        busy_min = rule.get("v1_min_programs")
        if busy_min is not None and B * H_KV >= busy_min:
            expect_v1 = expect_v1 or num_partitions <= rule["v1_busy_max_partitions"]
    else:
        expect_v1 = SEQ_LEN <= 8192 and (num_partitions == 1 or B * H_Q > 512)
    if get_arch() == "gfx950":
        assert rule is not None and not expect_v1, "gfx950 bf16/fp16 must use V2 here"
    assert called == ["paged_attn_decode_v1" if expect_v1 else "paged_attn_decode_v2"]

    torch_output = torch.zeros(B, H_Q, head_size, dtype=dtype, device="cuda")
    paged_attention_decode_ref(
        torch_output, query, key_cache, value_cache, block_tables, context_lens
    )
    torch.testing.assert_close(triton_output, torch_output, rtol=1e-02, atol=1e-02)


@pytest.mark.parametrize("B", [1, 4, 57, 64])
# @pytest.mark.parametrize("H_Q, H_KV", [(1,1), (16, 16), (2,1), (24,4)]) #TODO: GQA failing
@pytest.mark.parametrize("H_Q, H_KV", [(1, 1), (16, 16)])
@pytest.mark.parametrize("D", [1, 64, 128])
@pytest.mark.parametrize("KV_BLK_SZ", [1, 4, 512])
@pytest.mark.parametrize("SEQ_LEN", [1, 32, 57, 512, 10000])
@pytest.mark.parametrize("NUM_BLK", [4, 32])
@pytest.mark.parametrize(
    "dtype, kv_cache_dtype, compute_type, output_type",
    [
        (torch.float16, torch.float16, tl.float16, torch.float16),
        # (torch.bfloat16, torch.bfloat16, tl.bfloat16, torch.bfloat16),
    ],
)
def test_paged_attn_per_token_quant(
    B,
    H_Q,
    H_KV,
    D,
    KV_BLK_SZ,
    SEQ_LEN,
    NUM_BLK,
    dtype,
    kv_cache_dtype,
    compute_type,
    output_type,
):
    if D == 128 and KV_BLK_SZ == 512:  # Causes Shared Memory out of resources on Mi300
        pytest.skip("D={128} and KV_BLK_SZ={512} causes shared memory out of resources")

    if SEQ_LEN >= 8192 and B >= 16:
        pytest.skip("B>={4} and SEQ_LEN>={8192} tests are too slow")

    torch.cuda.empty_cache()  # Helps avoid hangs in large tests

    num_blocks = NUM_BLK

    (
        query,
        triton_output,
        key_cache,
        value_cache,
        key_cache_tri,
        value_cache_tri,
        context_lens,
        block_tables,
        max_context_len,
    ) = input_helper(
        B,
        H_Q,
        H_KV,
        D,
        KV_BLK_SZ,
        SEQ_LEN,
        dtype,
        kv_cache_dtype,
        output_type,
        num_blocks,
    )

    attn_scale = 1.0 / (D**0.5)

    (
        key_cache_tri_quant,
        k_scale,
    ) = pertoken_quant(
        key_cache_tri, scale_dtype=torch.float32, quant_dtype=torch.float8_e4m3fnuz
    )
    (
        value_cache_tri_quant,
        v_scale,
    ) = pertoken_quant(
        value_cache_tri, scale_dtype=torch.float32, quant_dtype=torch.float8_e4m3fnuz
    )

    paged_attention_decode(
        triton_output,
        query,
        key_cache_tri_quant,
        value_cache_tri_quant,
        context_lens,
        block_tables,
        attn_scale,
        max_context_len,
        compute_type,
        k_scale=k_scale,
        v_scale=v_scale,
        num_seq_partitions=0,
        alibi_slopes=None,
    )

    logger.debug(
        "B=%d H_Q=%d, H_KV=%d D=%d, KV_BLK_SZ=%d, SEQ_LEN=%d, NUM_BLK=%d",
        B,
        H_Q,
        H_KV,
        D,
        KV_BLK_SZ,
        SEQ_LEN,
        NUM_BLK,
    )
    logger.debug("query=%s", query)
    logger.debug(
        "key_cache_tri.shape=%s key_cache_tri=%s", key_cache_tri.shape, key_cache_tri
    )
    logger.debug("k_scale.shape=%s k_scale=%s", k_scale.shape, k_scale)
    logger.debug(
        "key_cache_tri_quant.shape=%s key_cache_tri_quant=%s",
        key_cache_tri_quant.shape,
        key_cache_tri_quant,
    )
    logger.debug("v_scale.shape=%s v_scale=%s", v_scale.shape, v_scale)
    logger.debug(
        "value_cache_tri.shape=%s value_cache_tri=%s",
        value_cache_tri.shape,
        value_cache_tri,
    )
    logger.debug(
        "value_cache_tri_quant.shape=%s value_cache_tri_quant=%s",
        value_cache_tri_quant.shape,
        value_cache_tri_quant,
    )
    logger.debug("triton_output=%s", triton_output)
    # torch doesn't have support for fp8 data type, so we convert here
    if dtype not in (torch.bfloat16, torch.float16, torch.float32):
        query = query.to(tl_to_torch_dtype[compute_type])

    if kv_cache_dtype not in (torch.bfloat16, torch.float16):
        key_cache = key_cache.to(dtype=tl_to_torch_dtype[compute_type])
        value_cache = value_cache.to(dtype=tl_to_torch_dtype[compute_type])
    torch_output = torch.zeros(B, H_Q, D, dtype=output_type, device="cuda")
    paged_attention_decode_ref(
        torch_output, query, key_cache, value_cache, block_tables, context_lens
    )
    logger.debug("torch_output=%s", torch_output)

    torch.testing.assert_close(triton_output, torch_output, rtol=2.5e-1, atol=2.5e-1)


_PAGES = (16, 64, 128)


def _config(arch, **kwargs):
    cfg_dir = resolve_config_dir(
        "attention", "PA-DECODE", backend="triton", arch=arch, **kwargs
    )
    return load_config_json(f"{cfg_dir}/DEFAULT.json", required=False) or {}


@pytest.mark.parametrize("B", [1, 4])
@pytest.mark.parametrize("KV_BLK_SZ", [_PAGES[0], _PAGES[-1]])
@pytest.mark.parametrize("SEQ_LEN", [512, 4096])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_paged_attn_gqa_split_output(B, KV_BLK_SZ, SEQ_LEN, dtype):
    """Check GQA decode output when the sequences have different lengths."""
    H_Q, H_KV, head_size = 32, 8, 128
    compute_type = {torch.bfloat16: tl.bfloat16, torch.float16: tl.float16}[dtype]
    num_blocks = B * triton.cdiv(SEQ_LEN, KV_BLK_SZ)
    (
        query,
        triton_output,
        key_cache,
        value_cache,
        key_cache_tri,
        value_cache_tri,
        _,
        block_tables,
        _,
    ) = input_helper(
        B, H_Q, H_KV, head_size, KV_BLK_SZ, SEQ_LEN, dtype, dtype, dtype, num_blocks
    )
    context_lens = torch.randint(1, SEQ_LEN + 1, (B,), device="cuda")
    context_lens[0] = SEQ_LEN
    paged_attention_decode(
        triton_output,
        query,
        key_cache_tri,
        value_cache_tri,
        context_lens,
        block_tables,
        1.0 / head_size**0.5,
        SEQ_LEN,
        compute_type,
        k_scale=torch.tensor([1.0]),
        v_scale=torch.tensor([1.0]),
    )
    torch_output = torch.zeros(B, H_Q, head_size, dtype=dtype, device="cuda")
    paged_attention_decode_ref(
        torch_output, query, key_cache, value_cache, block_tables, context_lens
    )
    torch.testing.assert_close(triton_output, torch_output, rtol=1e-02, atol=1e-02)


_TDM_LENS = [8192, 1, 127, 128, 129, 255, 257, 4000, 7777, 12345]


def _gqa_tdm_case(
    dtype, H_KV, group, lens_list, poison=False, reference=True, head_size=128
):
    """Random paged cache, an fp32 reference and optional NaN/Inf in unused rows."""
    page = 128
    B = len(lens_list)
    max_pages = triton.cdiv(max(lens_list), page)
    num_blocks = B * max_pages + 3
    query = torch.randn(B, H_KV * group, head_size, device="cuda").to(dtype)
    key_cache = torch.randn(num_blocks, H_KV, page, head_size, device="cuda").to(dtype)
    value_cache = torch.randn(num_blocks, H_KV, page, head_size, device="cuda").to(
        dtype
    )
    perm = torch.randperm(num_blocks, device="cuda").to(torch.int32)
    block_tables = perm[: B * max_pages].view(B, max_pages)
    lens = torch.tensor(lens_list, device="cuda")
    expected = None
    if reference:
        expected = torch.empty(
            B, H_KV * group, head_size, dtype=torch.float32, device="cuda"
        )
        for b, n in enumerate(lens_list):
            blocks = block_tables[b, : triton.cdiv(n, page)].long()
            k = key_cache[blocks].permute(1, 0, 2, 3).reshape(H_KV, -1, head_size)
            v = value_cache[blocks].permute(1, 0, 2, 3).reshape(H_KV, -1, head_size)
            q = query[b].float().view(H_KV, group, head_size)
            scores = torch.einsum("hgd,hld->hgl", q, k[:, :n].float())
            probs = (scores / head_size**0.5).softmax(-1)
            expected[b] = torch.einsum("hgl,hld->hgd", probs, v[:, :n].float()).reshape(
                H_KV * group, head_size
            )
    if poison:
        for b, n in enumerate(lens_list):
            if n % page:
                blk = int(block_tables[b, n // page])
                key_cache[blk, :, n % page :] = float("nan")
                value_cache[blk, :, n % page :] = float("inf")
    return query, key_cache, value_cache, block_tables, lens, expected


@pytest.mark.parametrize("poison", [False, True], ids=["clean", "nan_inf_tail"])
@pytest.mark.parametrize(
    "H_KV, group, num_warps", [(8, 4, 2), (8, 16, 4), (1, 16, 1), (2, 1, 2)]
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_paged_attn_gqa_tdm(dtype, H_KV, group, num_warps, poison):
    """TDM GQA kernel against an fp32 reference, with partial pages and garbage rows."""
    if get_arch() != "gfx1250":
        pytest.skip("the TDM kernel needs gfx1250")
    query, key_cache, value_cache, block_tables, lens, expected = _gqa_tdm_case(
        dtype, H_KV, group, _TDM_LENS, poison
    )
    output = torch.zeros_like(query)
    pa_decode_mod.paged_attn_decode_gqa_tdm(
        output, query, key_cache, value_cache, block_tables, lens, 128**-0.5, num_warps
    )
    atol = 3e-2 if dtype == torch.bfloat16 else 4e-3
    torch.testing.assert_close(output.float(), expected, rtol=0, atol=atol)


@pytest.mark.parametrize("poison", [False, True], ids=["clean", "nan_inf_tail"])
@pytest.mark.parametrize(
    "H_KV, group, head_size, num_warps",
    [(128, 1, 56, 1), (8, 4, 64, 2), (4, 16, 56, 4)],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_paged_attn_gqa_tdm_head_sizes(
    dtype, H_KV, group, head_size, num_warps, poison
):
    """Head sizes below the padded tile (56) and 64, including MHA-like groups of 1."""
    if get_arch() != "gfx1250":
        pytest.skip("the TDM kernel needs gfx1250")
    query, key_cache, value_cache, block_tables, lens, expected = _gqa_tdm_case(
        dtype, H_KV, group, _TDM_LENS[:6], poison, head_size=head_size
    )
    output = torch.zeros_like(query)
    pa_decode_mod.paged_attn_decode_gqa_tdm(
        output,
        query,
        key_cache,
        value_cache,
        block_tables,
        lens,
        head_size**-0.5,
        num_warps,
    )
    atol = 3e-2 if dtype == torch.bfloat16 else 4e-3
    torch.testing.assert_close(output.float(), expected, rtol=0, atol=atol)


def test_paged_attn_gqa_tdm_empty_sequence():
    """A sequence of length 0 writes zeros and does not disturb its neighbours."""
    if get_arch() != "gfx1250":
        pytest.skip("the TDM kernel needs gfx1250")
    lens_list = [300, 0, 129]
    query, key_cache, value_cache, block_tables, lens, expected = _gqa_tdm_case(
        torch.float16, 8, 4, [300, 1, 129]
    )
    lens = torch.tensor(lens_list, device="cuda")
    output = torch.full_like(query, 7.0)
    pa_decode_mod.paged_attn_decode_gqa_tdm(
        output, query, key_cache, value_cache, block_tables, lens, 128**-0.5, 2
    )
    assert torch.count_nonzero(output[1]) == 0
    for b in (0, 2):
        torch.testing.assert_close(output[b].float(), expected[b], rtol=0, atol=4e-3)


def _tdm_dispatch_calls(
    monkeypatch,
    rule,
    *,
    scale=1.0,
    alibi=None,
    head=128,
    kv=8,
    seqs=2,
    max_seq_len=1024,
    compute_type=None,
):
    calls = []
    monkeypatch.setattr(pa_decode_mod, "_get_dispatch_config", lambda dtype: rule)
    monkeypatch.setattr(pa_decode_mod, "_pa_decode_gqa_tdm_kernel", object())
    monkeypatch.setattr(
        pa_decode_mod,
        "paged_attn_decode_gqa_tdm",
        lambda *a, **k: calls.append(("tdm", k)),
    )
    for name in ("v1", "v2"):
        monkeypatch.setattr(
            pa_decode_mod,
            f"paged_attn_decode_{name}",
            lambda *a, _name=name, **k: calls.append((_name, k)),
        )
    # The launchers are mocked, so CPU tensors are enough.
    paged_attention_decode(
        output=torch.empty(seqs, 4 * kv, head, dtype=torch.float16, device="cpu"),
        query=torch.empty(seqs, 4 * kv, head, dtype=torch.float16, device="cpu"),
        key_cache=torch.empty(1, kv, 128, head, dtype=torch.float16, device="cpu"),
        value_cache=torch.empty(1, kv, 128, head, dtype=torch.float16, device="cpu"),
        seq_lens=None,
        block_tables=None,
        attn_scale=1.0,
        max_seq_len=max_seq_len,
        compute_type=compute_type,
        k_scale=torch.tensor([scale]),
        v_scale=torch.tensor([1.0]),
        alibi_slopes=alibi,
    )
    return [name for name, _ in calls], calls


_TDM_RULE = {
    "v1_max_partitions": 1,
    "tdm_num_warps": 4,
}


@pytest.mark.parametrize(
    "rule_change, kwargs",
    [
        ({"tdm_num_warps": None}, {}),
        ({}, {"scale": 2.0}),
        ({}, {"alibi": torch.ones(32)}),
        ({}, {"head": 80}),
        ({}, {"compute_type": tl.float32}),
    ],
    ids=["disabled", "k_scale", "alibi", "head_size", "compute_type"],
)
def test_dispatch_skips_tdm_when_unsupported(rule_change, kwargs, monkeypatch):
    names, _ = _tdm_dispatch_calls(monkeypatch, {**_TDM_RULE, **rule_change}, **kwargs)
    assert "tdm" not in names


@pytest.mark.parametrize("poison", [False, True], ids=["clean", "nan_inf_tail"])
@pytest.mark.parametrize(
    "H_KV, group, head_size, seq_partition_size",
    [(8, 4, 128, 128), (1, 16, 128, 256), (2, 1, 128, 1024), (4, 1, 56, 4096)],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_paged_attn_gqa_tdm_split(
    dtype, H_KV, group, head_size, seq_partition_size, poison
):
    """Split TDM kernel plus the v2 reduce kernel, with partitions past short sequences."""
    if get_arch() != "gfx1250":
        pytest.skip("the TDM kernel needs gfx1250")
    query, key_cache, value_cache, block_tables, lens, expected = _gqa_tdm_case(
        dtype, H_KV, group, _TDM_LENS, poison, head_size=head_size
    )
    output = torch.zeros_like(query)
    pa_decode_mod.paged_attn_decode_gqa_tdm(
        output,
        query,
        key_cache,
        value_cache,
        block_tables,
        lens,
        head_size**-0.5,
        2,
        seq_partition_size,
        max(_TDM_LENS),
    )
    atol = 3e-2 if dtype == torch.bfloat16 else 4e-3
    torch.testing.assert_close(output.float(), expected, rtol=0, atol=atol)


def test_paged_attn_gqa_tdm_is_repeatable():
    """A race in the LDS ring would make repeated runs differ."""
    if get_arch() != "gfx1250":
        pytest.skip("the TDM kernel needs gfx1250")
    query, key_cache, value_cache, block_tables, lens, _ = _gqa_tdm_case(
        torch.float16, 8, 4, [8192] * 128, reference=False
    )
    first = torch.zeros_like(query)
    args = (query, key_cache, value_cache, block_tables, lens, 128**-0.5)
    pa_decode_mod.paged_attn_decode_gqa_tdm(first, *args, 1)
    output = torch.zeros_like(query)
    for _ in range(20):
        output.zero_()
        pa_decode_mod.paged_attn_decode_gqa_tdm(output, *args, 1)
        assert torch.equal(output, first)


_V1_V2_NAMES = {torch.bfloat16: "bf16", torch.float16: "fp16"}


def _page128_rule(dtype):
    """gfx1250 rule without the Gluon kernel, so the Triton V1/V2 path is chosen."""
    return {
        **_config("gfx1250")["dispatch"][_V1_V2_NAMES[dtype]],
        "tdm_num_warps": None,
    }


@pytest.mark.parametrize(
    "min_programs, seq, expect",
    [(8, 8192, "v1"), (8, 9216, "v2"), (9, 8192, "v2")],
    ids=["busy_v1", "over_partition_limit", "not_busy"],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_paged_attn_page128_busy_v1_output(
    dtype, min_programs, seq, expect, monkeypatch
):
    """V1/V2 output on 128-token pages; 8 programs stand in for the 2048-program limit."""
    if get_arch() != "gfx1250":
        pytest.skip("the 128-token launch settings are tuned for gfx1250")
    rule = {k: v for k, v in _page128_rule(dtype).items() if not k.startswith("split_")}
    rule["v1_min_programs"] = min_programs
    assert rule["v1_busy_max_partitions"] == 8
    monkeypatch.setattr(pa_decode_mod, "_get_dispatch_config", lambda _: rule)
    called, launches = [], []
    for name in ("paged_attn_decode_v1", "paged_attn_decode_v2"):
        fn = getattr(pa_decode_mod, name)
        monkeypatch.setattr(
            pa_decode_mod,
            name,
            lambda *a, _fn=fn, _name=name, **k: (called.append(_name), _fn(*a, **k))[1],
        )
    get_launch = pa_decode_mod._get_launch_config
    monkeypatch.setattr(
        pa_decode_mod,
        "_get_launch_config",
        lambda kernel, blk: (launches.append((kernel, blk)), get_launch(kernel, blk))[
            1
        ],
    )
    B, H_KV, group = 4, 2, 4
    query, key_cache, value_cache, block_tables, lens, expected = _gqa_tdm_case(
        dtype, H_KV, group, [seq] * B
    )
    output = torch.zeros_like(query)
    paged_attention_decode(
        output,
        query,
        key_cache,
        value_cache,
        lens,
        block_tables,
        128**-0.5,
        seq,
        {torch.bfloat16: tl.bfloat16, torch.float16: tl.float16}[dtype],
        k_scale=torch.tensor([1.0], dtype=torch.float32, device="cuda"),
        v_scale=torch.tensor([1.0], dtype=torch.float32, device="cuda"),
    )
    assert called == [f"paged_attn_decode_{expect}"]
    assert launches == [(f"{expect}_w_dot", 128)]
    atol = 3e-2 if dtype == torch.bfloat16 else 4e-3
    torch.testing.assert_close(output.float(), expected, rtol=0, atol=atol)
