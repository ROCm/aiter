# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Paged gfx950 FMHA against the unified-attention torch oracle."""

import itertools

import pytest
import torch

from aiter.ops.flydsl.kernels.flash_attn_func_fp8_gfx950 import (
    _fp8_auto_block_m,
    _fp8_rescale_threshold,
    _num_cu,
)
from aiter.ops.flydsl.kernels.fmha_gfx950.flash_attn_fp8_gfx950 import (
    build_flash_attn_dualwave_swp_fp8_module,
)
from aiter.test_common import checkAllclose

PAGE, H, HKV, D = 64, 64, 4, 128


def _arch():
    return (
        torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
        if torch.cuda.is_available()
        else ""
    )


pytestmark = pytest.mark.skipif(
    _arch() != "gfx950", reason="paged fp8 FMHA requires gfx950"
)


def q8(x):
    s = x.abs().amax().clamp(min=1e-4) / 448.0
    return (x / s).to(torch.float8_e4m3fn), s.reshape(1).float().to(x.device)


def ref_paged_attn(
    query,
    key_cache,
    value_cache,
    query_lens,
    kv_lens,
    block_tables,
    scale,
    out_dtype=torch.float32,
    sliding_window=None,
    soft_cap=None,
    sinks=None,
    q_descale=None,
    k_descale=None,
    v_descale=None,
    output_scale=None,
    causal=1,
):
    # Local copy of the established paged oracle; never import a test suite.
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
        outputs.append(torch.einsum("hqk,khd->qhd", attn, v))
        start_idx += query_len
    out = torch.cat(outputs, dim=0)
    if output_scale is not None:
        out = out / output_scale
    return out.to(out_dtype)


def make_case(query_lens, kv_lens, dtype, causal=True, seed=3, group=None):
    heads = H if group is None else HKV * group
    g = torch.Generator(device="cuda").manual_seed(seed)
    pages = [max(0, (n + PAGE - 1) // PAGE) for n in kv_lens]
    n_pages = max(1, sum(pages))
    q, qs = q8(
        torch.randn(
            sum(query_lens), heads, D, device="cuda", dtype=torch.bfloat16, generator=g
        )
    )
    k, ks = q8(
        torch.randn(
            n_pages, PAGE, HKV, D, device="cuda", dtype=torch.bfloat16, generator=g
        )
    )
    v, vs = q8(
        torch.randn(
            n_pages, PAGE, HKV, D, device="cuda", dtype=torch.bfloat16, generator=g
        )
    )
    perm = torch.randperm(
        n_pages, generator=torch.Generator().manual_seed(seed + 1)
    ).tolist()
    bt = torch.zeros(len(pages), max(1, max(pages)), device="cuda", dtype=torch.int32)
    offset = 0
    for i, count in enumerate(pages):
        bt[i, :count] = torch.tensor(
            perm[offset : offset + count], device="cuda", dtype=torch.int32
        )
        offset += count
    cu_q = torch.tensor(
        [0] + list(itertools.accumulate(query_lens)), device="cuda", dtype=torch.int32
    )
    return {
        "q": q,
        "k": k,
        "v": v,
        "out": torch.full(q.shape, float("nan"), device="cuda", dtype=dtype),
        "cu_seqlens_q": cu_q,
        "max_seqlen_q": max(query_lens),
        "seqused_k": torch.tensor(kv_lens, device="cuda", dtype=torch.int32),
        "max_seqlen_k": max(kv_lens),
        "softmax_scale": D**-0.5,
        "causal": causal,
        "window_size": (-1, -1),
        "block_table": bt,
        "softcap": 0,
        "q_descale": qs,
        "k_descale": ks,
        "v_descale": vs,
    }


def reference(case, query_lens, kv_lens):
    return ref_paged_attn(
        case["q"],
        case["k"],
        case["v"],
        query_lens,
        kv_lens,
        case["block_table"],
        case["softmax_scale"],
        q_descale=case["q_descale"],
        k_descale=case["k_descale"],
        v_descale=case["v_descale"],
        causal=case["causal"],
        soft_cap=case["softcap"],
        sliding_window=(
            None if case["window_size"][0] < 0 else case["window_size"][0] + 1
        ),
    )


def direct_candidate(
    case,
    kv_lens,
    block_m=None,
    layout="linear",
    body_variant="default",
    gqa_pack_m=False,
):
    heads = case["q"].shape[1]
    b, max_q = len(kv_lens), case["max_seqlen_q"]
    if block_m is None:
        block_m = _fp8_auto_block_m(
            b, H, max_q, max(kv_lens), _num_cu(case["q"].device)
        )
    mod = build_flash_attn_dualwave_swp_fp8_module(
        num_heads=heads,
        num_kv_heads=HKV,
        head_dim=D,
        causal=case["causal"],
        paged=True,
        varlen=True,
        cross_seqlen=True,
        kv_cache_layout="shuffled" if layout != "linear" else "linear",
        _k_shuffled_only=layout == "k_shuffled",
        out_dtype="bf16" if case["out"].dtype == torch.bfloat16 else "f16",
        rescale_threshold=_fp8_rescale_threshold(max(kv_lens)),
        block_m=block_m,
        body_variant=body_variant,
        gqa_pack_m=gqa_pack_m,
    )
    mod(
        case["q"].view(torch.int8).reshape(-1),
        case["k"].view(torch.int8).reshape(case["k"].shape[0], -1),
        case["v"].view(torch.int8).reshape(case["v"].shape[0], -1),
        case["out"].reshape(-1),
        b,
        max_q,
        stride_q_n=heads * D,
        stride_kv_n=HKV * D,
        seq_len_kv=max(kv_lens),
        softmax_scale=case["softmax_scale"],
        cu_seqlens_q=case["cu_seqlens_q"],
        cu_seqlens_kv=case["seqused_k"],
        q_descale=case["q_descale"],
        k_descale=case["k_descale"],
        v_descale=case["v_descale"],
        block_table=case["block_table"].reshape(-1),
        block_table_stride=case["block_table"].stride(0),
        stream=torch.cuda.current_stream(),
    )
    return case["out"]


def _check(want, got, atol=None):
    if atol is None:
        atol = 0.08 * want.abs().max().item()
    assert torch.isfinite(got).all()
    assert (
        checkAllclose(want.float(), got.float(), rtol=0, atol=atol, tol_err_ratio=0)
        == 0
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("layout", ["linear", "shuffled"])
@pytest.mark.parametrize("body_variant", ["default", "conventional_bn64"])
def test_paged_prefill(dtype, causal, layout, body_variant):
    qlens, klens = [512, 256], [512, 256]
    case = make_case(qlens, klens, dtype, causal)
    want = reference(case, qlens, klens)
    shuffle_case(case, layout)
    _check(
        want, direct_candidate(case, klens, layout=layout, body_variant=body_variant)
    )


@pytest.mark.parametrize("block_m", [128, 256])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("layout", ["linear", "shuffled"])
@pytest.mark.parametrize("body_variant", ["default", "conventional_bn64"])
def test_paged_ragged_stale_nan(block_m, causal, layout, body_variant):
    qlens, klens = [513, 1, 129, 257], [577, 0, 65, 1]
    if body_variant == "conventional_bn64" and block_m != 128:
        pytest.skip("conventional_bn64 fixes block_m=128")
    case = make_case(qlens, klens, torch.bfloat16, causal)
    # Tail bytes and page-zero prefetches must not reach PV, while valid
    # payload is unchanged. The longest Q supplies excess blocks for every row.
    for row, n in enumerate(klens):
        if n % PAGE:
            page = int(case["block_table"][row, n // PAGE])
            case["v"].view(torch.uint8)[page, n % PAGE :] = 0x7F
    storage = torch.full(
        (sum(qlens) + block_m, H, D),
        float("nan"),
        device="cuda",
        dtype=case["out"].dtype,
    )
    case["out"] = storage[: sum(qlens)]
    want = torch.nan_to_num(reference(case, qlens, klens))
    shuffle_case(case, layout)
    got = direct_candidate(case, klens, block_m, layout, body_variant)
    _check(want, got)
    assert torch.count_nonzero(got[qlens[0] : qlens[0] + qlens[1]]) == 0
    assert torch.isnan(storage[sum(qlens) :]).all()


@pytest.mark.parametrize("block_m", [128, 256])
@pytest.mark.parametrize("layout", ["linear", "shuffled"])
@pytest.mark.parametrize("body_variant", ["default", "conventional_bn64"])
def test_paged_empty_segment(block_m, layout, body_variant):
    if body_variant == "conventional_bn64" and block_m != 128:
        pytest.skip("conventional_bn64 fixes block_m=128")
    case = make_case([513], [0], torch.float16)
    case["k"].view(torch.uint8).fill_(0x7F)
    case["v"].view(torch.uint8).fill_(0xFF)
    shuffle_case(case, layout)
    got = direct_candidate(case, [0], block_m, layout, body_variant)
    assert torch.count_nonzero(got) == 0


@pytest.mark.parametrize("group", [1, 4, 8, 16])
@pytest.mark.parametrize("layout", ["linear", "shuffled"])
@pytest.mark.parametrize(
    "causal,dtype",
    [
        (False, torch.bfloat16),
        (True, torch.bfloat16),
        (True, torch.float16),
    ],
)
def test_paged_gqa_pack_m(group, layout, causal, dtype):
    qlens, klens = [93, 1, 17, 9], [157, 0, 5, 1]
    case = make_case(qlens, klens, dtype, causal, group=group)
    for row, n in enumerate(klens):
        if n % PAGE:
            page = int(case["block_table"][row, n // PAGE])
            case["v"].view(torch.uint8)[page, n % PAGE :] = 0x7F
    storage = torch.full(
        (sum(qlens) + 128, HKV * group, D),
        float("nan"),
        device="cuda",
        dtype=dtype,
    )
    case["out"] = storage[: sum(qlens)]
    want = torch.nan_to_num(reference(case, qlens, klens))
    shuffle_case(case, layout)
    got = direct_candidate(
        case,
        klens,
        layout=layout,
        body_variant="conventional_bn64",
        gqa_pack_m=True,
    )
    _check(want, got)
    assert torch.count_nonzero(got[qlens[0] : qlens[0] + qlens[1]]) == 0
    assert torch.isnan(storage[sum(qlens) :]).all()


@pytest.mark.parametrize("layout", ["linear", "shuffled"])
def test_paged_gqa_pack_m_empty(layout):
    case = make_case([17], [0], torch.float16, group=16)
    case["k"].view(torch.uint8).fill_(0x7F)
    case["v"].view(torch.uint8).fill_(0xFF)
    shuffle_case(case, layout)
    got = direct_candidate(
        case,
        [0],
        layout=layout,
        body_variant="conventional_bn64",
        gqa_pack_m=True,
    )
    assert torch.count_nonzero(got) == 0


@pytest.mark.parametrize("layout", ["linear", "shuffled"])
@pytest.mark.parametrize("gqa_pack_m", [False, True])
@pytest.mark.parametrize("causal", [False, True])
def test_paged_bn64_concurrent_v_reads(layout, gqa_pack_m, causal):
    qlens, klens = [65] * 40, [64] * 40
    case = make_case(qlens, klens, torch.bfloat16, causal)
    want = torch.nan_to_num(reference(case, qlens, klens))
    shuffle_case(case, layout)
    # A single KV tile excludes ring reuse; concurrent waves expose missing
    # completion waits that low-concurrency cases can hide behind LDS latency.
    for _ in range(5):
        got = direct_candidate(
            case,
            klens,
            block_m=128,
            layout=layout,
            body_variant="conventional_bn64",
            gqa_pack_m=gqa_pack_m,
        )
        _check(want, got)


@pytest.mark.parametrize("poison_tail", [False, True])
def test_paged_bn64_shuffled_full_tiles_and_tail(poison_tail):
    qlens, klens = [65] * 40, [197] * 40
    case = make_case(qlens, klens, torch.bfloat16, causal=False)
    # Distinct token values expose page/quad permutations across the branch.
    tokens = torch.arange(PAGE, device="cuda", dtype=torch.float32)
    values = ((tokens % 13) - 6).view(1, PAGE, 1, 1).expand_as(case["v"])
    case["v"] = values.to(case["v"].dtype).contiguous()
    case["v_descale"].fill_(1)
    if poison_tail:
        for row, n in enumerate(klens):
            page = int(case["block_table"][row, n // PAGE])
            case["v"].view(torch.uint8)[page, n % PAGE :] = 0x7F
    want = reference(case, qlens, klens)
    shuffle_case(case, "shuffled")
    got = direct_candidate(
        case,
        klens,
        layout="shuffled",
        body_variant="conventional_bn64",
        gqa_pack_m=True,
    )
    _check(want, got)


@pytest.mark.parametrize("layout", ["linear", "shuffled"])
def test_paged_addressing(layout):
    qlens, klens = [128], [256]
    original = make_case(qlens, klens, torch.bfloat16)
    want = reference(original, qlens, klens)
    outputs = []
    pool_blocks, count = 65535, original["k"].shape[0]
    for logical_ids in (list(range(count)), list(reversed(range(count)))):
        ids = [pool_blocks - count + i for i in logical_ids]
        case = dict(original, out=torch.empty_like(original["out"]))
        for key in ("k", "v"):
            pool = torch.zeros(
                pool_blocks, PAGE, HKV, D, device="cuda", dtype=original[key].dtype
            )
            for logical, pid in enumerate(ids):
                pool[pid] = original[key][int(original["block_table"][0, logical])]
            case[key] = pool
        case["block_table"] = torch.tensor([ids], device="cuda", dtype=torch.int32)
        shuffle_case(case, layout)
        outputs.append(direct_candidate(case, klens, layout=layout).clone())
        _check(want, outputs[-1], atol=0.1)
        del case, pool
    assert torch.equal(outputs[0], outputs[1])


def shuffle_k(k):
    nb = k.shape[0]
    return (
        k.view(torch.uint8)
        .reshape(nb, PAGE, HKV, D // 16, 16)
        .permute(0, 2, 3, 1, 4)
        .contiguous()
        .view(k.dtype)
    )


@pytest.mark.parametrize("block_m", [128, 256])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("layout", ["k_shuffled", "shuffled"])
def test_paged_shuffled_matches_linear(block_m, causal, layout):
    qlens, klens = [257, 129], [321, 65]
    case = make_case(qlens, klens, torch.bfloat16, causal)
    want = torch.nan_to_num(reference(case, qlens, klens))
    linear = direct_candidate(case, klens, block_m).clone()
    shuffle_case(case, layout)
    got = direct_candidate(case, klens, block_m, layout)
    _check(want, got)
    # The shuffled V reader permutes the PV reduction order, so results match linear only to rounding.
    _check(linear, got)


def shuffle_case(case, layout):
    if layout != "linear":
        case["k"] = shuffle_k(case["k"])
    if layout == "shuffled":
        v = case["v"]
        case["v"] = (
            v.view(torch.uint8)
            .reshape(v.shape[0], PAGE // 16, 16, HKV, D)
            .permute(0, 3, 1, 4, 2)
            .contiguous()
            .view(v.dtype)
        )
