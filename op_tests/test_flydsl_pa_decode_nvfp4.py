# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""NVFP4 K/V FlyDSL paged decode on gfx1250 against a torch reference.

NVFP4 here is the MiniMax-M3 sparse KV pool layout: per (page, KV head) a
token-major [128, D // 2] e2m1 payload (element 2j in the low nibble of byte j)
and [128, D // 16] E4M3 block scales, no outer scale. The reference dequantizes
the cache to fp32 and runs softmax attention; the kernel widens K/V to BF16
exactly, so the remaining difference is BF16 Q/P rounding.

The sparse cases follow MiniMax-M3.1 at TP4 (16 query heads on one KV head,
head_dim 128, top-16 blocks of 128 tokens): every row (a decode or MTP-verify
token, or a prefill token) is its own sequence with a compacted table of its
selected pages, the partial tail page last, as ATOM builds it.

Run with: python -m pytest -q op_tests/test_flydsl_pa_decode_nvfp4.py
"""

import importlib
import importlib.util

import pytest
import torch

from aiter.jit.utils.chip_info import get_gfx_runtime

PAGE = 128
D = 128
E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


@pytest.fixture
def device(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("ROCm is not available")
    if get_gfx_runtime() != "gfx1250":
        pytest.skip("NVFP4 pa_decode runs on gfx1250")
    if importlib.util.find_spec("flydsl") is None:
        pytest.skip("FlyDSL is not installed")
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    return torch.device("cuda")


def nvfp4_quantize(x):
    """[..., D] float -> (payload uint8 [..., D // 2], scales e4m3 [..., D // 16]),
    as ATOM's ``_nvfp4_store_kernel`` writes them: scale = e4m3(amax / 6), codes
    rounded half to even and saturated at 6."""
    blocks = x.float().reshape(*x.shape[:-1], -1, 16)
    scale = (blocks.abs().amax(-1) / 6.0).to(torch.float8_e4m3fn)
    scale_f = scale.float()
    inv = torch.where(scale_f > 0, 1.0 / scale_f, torch.zeros_like(scale_f))
    y = blocks * inv[..., None]
    a = y.abs()
    mag = (
        (a > 0.25).int()
        + (a >= 0.75).int()
        + (a > 1.25).int()
        + (a >= 1.75).int()
        + (a > 2.5).int()
        + (a >= 3.5).int()
        + (a > 5.0).int()
    )
    codes = (mag | torch.where(y < 0, 8, 0)).reshape(*x.shape[:-1], -1, 2)
    payload = (codes[..., 0] | (codes[..., 1] << 4)).to(torch.uint8)
    return payload, scale


def nvfp4_dequantize(payload, scales):
    """Inverse of the packing: fp32 [..., D]."""
    table = torch.tensor(E2M1, device=payload.device)
    codes = torch.stack([payload & 15, payload >> 4], -1).flatten(-2).long()
    values = table[codes & 7] * torch.where(codes >= 8, -1.0, 1.0)
    values = values.reshape(*values.shape[:-1], -1, 16)
    return (values * scales.float()[..., None]).flatten(-2)


def make_cache(num_pages, num_kv_heads, device, seed=0):
    g = torch.Generator(device=device).manual_seed(seed)
    k = torch.randn(num_pages, num_kv_heads, PAGE, D, device=device, generator=g)
    v = torch.randn(num_pages, num_kv_heads, PAGE, D, device=device, generator=g)
    # Head-row magnitudes spread over E4M3's range, as real activations do.
    k = k * torch.exp2(torch.randint(-6, 4, (num_pages, 1, PAGE, 1), device=device))
    kp, ks = nvfp4_quantize(k)
    vp, vs = nvfp4_quantize(v)
    return kp, ks, vp, vs


def poison_tails(cache, table, lengths):
    """Bytes past each row's context in its last page (the never-written part of a
    tail block) become NaN scales and random codes: the kernel must not read them."""
    kp, ks, vp, vs = cache
    for row, n in enumerate(lengths.tolist()):
        if n % PAGE == 0:
            continue
        page = int(table[row, (n - 1) // PAGE])
        tail = n % PAGE
        for t in (kp, vp):
            t[page, :, tail:] = torch.randint_like(t[page, :, tail:], 0, 256)
        for t in (ks, vs):
            t[page, :, tail:] = torch.tensor(float("nan")).to(torch.float8_e4m3fn)


def sparse_rows(num_rows, topk, num_pages, device, seed, max_len=None):
    """Per-row compacted page tables (distinct pages, tail last) and lengths."""
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(num_pages, generator=g)
    assert num_rows * topk <= num_pages
    table = perm[: num_rows * topk].reshape(num_rows, topk).to(torch.int32)
    full = topk * PAGE if max_len is None else max_len
    lengths = torch.randint(1, full + 1, (num_rows,), generator=g)
    lengths[::7] = full  # top-k windows: every selected block full
    lengths[3::11] = 0  # padding rows
    lengths[5::13] = torch.randint(1, 129, (len(lengths[5::13]),), generator=g)
    return table.to(device), lengths.to(torch.int32).to(device)


def reference(query, cache, table, lengths, scale, query_length=1):
    """fp32 attention of each row over the dequantized pages of its table."""
    kp, ks, vp, vs = cache
    rows, heads, _ = query.shape
    num_kv_heads = kp.shape[1]
    group = heads // num_kv_heads
    out = torch.zeros(rows, heads, D, device=query.device)
    for seq in range(lengths.shape[0]):
        n = int(lengths[seq])
        if n == 0:
            continue
        pages = table[seq, : (n + PAGE - 1) // PAGE].long()
        k = nvfp4_dequantize(kp[pages], ks[pages])  # [p, Hkv, PAGE, D]
        v = nvfp4_dequantize(vp[pages], vs[pages])
        k = k.permute(1, 0, 2, 3).reshape(num_kv_heads, -1, D)[:, :n]
        v = v.permute(1, 0, 2, 3).reshape(num_kv_heads, -1, D)[:, :n]
        for qi in range(query_length):
            row = seq * query_length + qi
            visible = n - query_length + qi + 1
            q = query[row].float().reshape(num_kv_heads, group, D)
            s = torch.einsum("hgd,htd->hgt", q, k[:, :visible]) * scale
            p = torch.softmax(s, -1)
            o = torch.einsum("hgt,htd->hgd", p, v[:, :visible])
            out[row] = o.reshape(heads, D)
    return out


def run(
    device,
    monkeypatch,
    num_rows,
    *,
    num_kv_heads=1,
    group=16,
    topk=16,
    query_dtype=torch.bfloat16,
    query_length=1,
    max_partitions=None,
    workgroup_budget=None,
    max_len=None,
    num_pages=None,
    payload_dtype=torch.uint8,
    expect=None,
):
    pa = importlib.import_module("aiter.ops.flydsl.pa_decode")
    planner = importlib.import_module("aiter.ops.flydsl.kernels.pa_decode_plan")
    num_seqs = num_rows // query_length
    num_pages = num_pages or max(64, num_seqs * topk + 8)
    cache = make_cache(num_pages, num_kv_heads, device)
    table, lengths = sparse_rows(num_seqs, topk, num_pages, device, 1, max_len)
    if query_length > 1:
        lengths = torch.clamp(lengths, min=query_length)
    poison_tails(cache, table, lengths)
    g = torch.Generator(device=device).manual_seed(2)
    query = torch.randn(
        num_rows, num_kv_heads * group, D, device=device, generator=g
    ).to(query_dtype)
    output = torch.full_like(query, float("nan"))
    scale = D**-0.5
    kp, ks, vp, vs = cache
    plan = planner.plan_pa_decode(
        lengths,
        num_kv_heads,
        max_partitions=max_partitions,
        workgroup_budget=workgroup_budget,
        query_length=query_length,
    )
    compiled = []
    compile_tile = pa.compile_pa_decode_tile

    def spy(**kwargs):
        compiled.append(kwargs)
        return compile_tile(**kwargs)

    monkeypatch.setattr(pa, "compile_pa_decode_tile", spy)
    expected = reference(query, cache, table, lengths, scale, query_length)
    for _ in range(2):  # the second call takes the recorded-launch path
        output.fill_(float("nan"))
        pa.pa_decode(
            output,
            query,
            kp.view(payload_dtype),
            vp.view(payload_dtype),
            lengths,
            table,
            scale,
            query_length,
            key_scale=ks,
            value_scale=vs,
            work_plan=plan,
            max_context_length=topk * PAGE,
        )
        torch.cuda.synchronize()
        assert torch.isfinite(output).all(), "non-finite output"
        err = (output.float() - expected).abs()
        tol = 2e-2 + 2e-2 * expected.abs()
        assert (err <= tol).all(), (
            f"max abs err {err.max().item():.4g} at "
            f"{torch.nonzero(err > tol)[:4].tolist()}"
        )
        cos = torch.nn.functional.cosine_similarity(
            output.float().flatten(), expected.flatten(), dim=0
        )
        assert cos > 0.9999, f"cosine {cos.item():.6f}"
    assert compiled and compiled[-1]["kv_scale_dtype"] == torch.float8_e4m3fn
    if expect is not None:
        mode = (
            f"cluster{compiled[-1]['cluster_merge']}"
            if compiled[-1]["cluster_merge"]
            else ("direct" if compiled[-1]["direct_output"] else "reduce")
        )
        assert mode == expect, (mode, expect)


# (rows, kwargs): M3.1 TP4 decode shapes. MTP rows are flattened per token,
# exactly as the FP8 sparse path does (query_length 1, own table per token).
CASES = {
    "plain-decode-c4": (4, {}),
    "plain-decode-c64": (64, {}),
    "mtp8-verify-c4": (32, {}),
    "mtp8-verify-c64": (512, {}),
    "mtp8-verify-c64-cluster8": (512, {"max_partitions": 8}),
    "one-partition-direct": (32, {"max_partitions": 1, "expect": "direct"}),
    "last-arriver-merge": (8, {"max_partitions": 8, "workgroup_budget": 1}),
    "reducer": (300, {"max_partitions": 8, "workgroup_budget": 4096}),
    "prefill-rows": (1024, {"max_len": 1024}),
    "short-rows": (96, {"max_len": 200}),
    "f16-query": (32, {"query_dtype": torch.float16}),
    "fp4x2-view": (16, {"payload_dtype": None}),
    "two-kv-heads-gqa8": (32, {"num_kv_heads": 2, "group": 8}),
}


@pytest.mark.parametrize("case", list(CASES))
def test_pa_decode_nvfp4(case, device, monkeypatch):
    rows, kwargs = CASES[case]
    kwargs = dict(kwargs)
    if "payload_dtype" in kwargs and kwargs["payload_dtype"] is None:
        from aiter.utility import dtypes

        kwargs["payload_dtype"] = dtypes.fp4x2
    run(device, monkeypatch, rows, **kwargs)


def test_pa_decode_nvfp4_rejects(device):
    """Shapes outside the NVFP4 tile raise instead of computing garbage."""
    pa = importlib.import_module("aiter.ops.flydsl.pa_decode")
    planner = importlib.import_module("aiter.ops.flydsl.kernels.pa_decode_plan")
    kp, ks, vp, vs = make_cache(8, 1, device)
    lengths = torch.full((2,), 256, dtype=torch.int32, device=device)
    table = torch.arange(4, dtype=torch.int32, device=device).reshape(2, 2)
    query = torch.randn(2, 16, D, device=device, dtype=torch.bfloat16)
    plan = planner.plan_pa_decode(lengths, 1)
    with pytest.raises(NotImplementedError):
        pa.pa_decode(
            torch.empty_like(query),
            query,
            kp,
            vp,
            lengths,
            table,
            0.1,
            1,
            key_scale=ks,
            value_scale=vs,
            work_plan=plan,
            sinks=torch.zeros(16, device=device),
        )
    with pytest.raises(ValueError):
        pa.pa_decode(
            torch.empty_like(query),
            query,
            kp,
            vp,
            lengths,
            table,
            0.1,
            1,
            key_scale=ks[:, :, :, :4],
            value_scale=vs[:, :, :, :4],
            work_plan=plan,
        )
    # Two query tokens x GQA16 = 32 rows: more than the wave tile's one query
    # tile.
    query2 = torch.randn(4, 16, D, device=device, dtype=torch.bfloat16)
    plan2 = planner.plan_pa_decode(lengths, 1, query_length=2)
    with pytest.raises(ValueError):
        pa.pa_decode(
            torch.empty_like(query2),
            query2,
            kp,
            vp,
            lengths,
            table,
            0.1,
            2,
            key_scale=ks,
            value_scale=vs,
            work_plan=plan2,
        )


# (rows, plan kwargs, output mode): the FP8 output epilogue (direct and cluster8)
# and the other modes, which quantize after the BF16 output inside pa_decode.
QUANT_CASES = {
    "direct-prefill-rows": (1024, {"max_partitions": 1, "max_len": 1024}, "direct"),
    "direct-decode": (32, {"max_partitions": 1}, "direct"),
    "cluster8-decode-c64": (
        64,
        {"max_partitions": 8, "workgroup_budget": 512},
        "cluster8",
    ),
    "cluster8-mtp8-c16": (
        128,
        {"max_partitions": 8, "workgroup_budget": 1024},
        "cluster8",
    ),
    "reducer-fallback": (
        300,
        {"max_partitions": 8, "workgroup_budget": 4096},
        "reduce",
    ),
}


@pytest.mark.parametrize("store_output", [True, False], ids=["store", "quantonly"])
@pytest.mark.parametrize("query_dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("case", list(QUANT_CASES))
def test_pa_decode_nvfp4_fp8_output(
    case, query_dtype, store_output, device, monkeypatch
):
    """quantized_output / output_scale with NVFP4 K/V: bit for bit the BF16 (FP16)
    output of the same call quantized per token by dynamic_per_token_scaled_quant,
    the semantics of the FP8 page16 epilogue; padded bucket rows are FP8 zeros
    with a unit descale."""
    from aiter.ops.flydsl.kernels import pa_decode_kernel as kernels
    from aiter.ops.quant import dynamic_per_token_scaled_quant
    from aiter.utility import dtypes

    pa = importlib.import_module("aiter.ops.flydsl.pa_decode")
    planner = importlib.import_module("aiter.ops.flydsl.kernels.pa_decode_plan")
    rows, plan_kwargs, mode = QUANT_CASES[case]
    plan_kwargs = dict(plan_kwargs)
    max_len = plan_kwargs.pop("max_len", None)
    topk = 16
    num_pages = rows * topk + 8
    cache = make_cache(num_pages, 1, device)
    table, lengths = sparse_rows(rows, topk, num_pages, device, 1, max_len)
    poison_tails(cache, table, lengths)
    kp, ks, vp, vs = cache
    g = torch.Generator(device=device).manual_seed(3)
    query = torch.randn(rows, 16, D, device=device, generator=g).to(query_dtype)
    plan = planner.plan_pa_decode(lengths, 1, **plan_kwargs)
    common = {"key_scale": ks, "value_scale": vs, "work_plan": plan}
    bound = topk * PAGE

    reference_out = torch.full_like(query, float("nan"))
    pa.pa_decode(
        reference_out, query, kp, vp, lengths, table, D**-0.5, 1,
        max_context_length=bound, **common,
    )  # fmt: skip
    ref_q = torch.empty(rows, 16 * D, dtype=dtypes.fp8, device=device)
    ref_s = torch.empty(rows, 1, dtype=torch.float32, device=device)
    dynamic_per_token_scaled_quant(ref_q, reference_out.view(rows, -1), ref_s)

    compiled = []
    compile_tile = pa.compile_pa_decode_tile

    def spy(**kwargs):
        compiled.append(kwargs)
        return compile_tile(**kwargs)

    monkeypatch.setattr(pa, "compile_pa_decode_tile", spy)
    pad = 5
    for _ in range(2):  # the second call takes the recorded-launch path
        output = torch.full_like(query, float("nan"))
        quantized = torch.full(
            (rows + pad, 16, D), 0x7F, dtype=torch.uint8, device=device
        ).view(dtypes.fp8)
        scales = torch.full((rows + pad, 1), float("nan"), device=device)
        pa.pa_decode(
            output, query, kp, vp, lengths, table, D**-0.5, 1,
            max_context_length=bound, quantized_output=quantized,
            output_scale=scales, store_output=store_output,
            quantized_pad_rows=pad, **common,
        )  # fmt: skip
        torch.cuda.synchronize()
        got_q = quantized[:rows].view(rows, -1).view(torch.uint8)
        assert torch.equal(got_q, ref_q.view(torch.uint8)), "FP8 bytes differ"
        assert torch.equal(scales[:rows], ref_s), "descales differ"
        pad_bytes = quantized[rows:].view(torch.uint8)
        assert torch.equal(pad_bytes, torch.zeros_like(pad_bytes))
        assert torch.equal(scales[rows:], torch.ones_like(scales[rows:]))
        if store_output:
            assert torch.equal(output, reference_out), "BF16 output differs"
    want = "nvfp4w"
    if mode != "reduce":
        assert compiled[-1]["quant_output"]
        assert compiled[-1]["store_output"] == store_output
        got_mode = (
            f"cluster{compiled[-1]['cluster_merge']}"
            if compiled[-1]["cluster_merge"]
            else ("direct" if compiled[-1]["direct_output"] else "reduce")
        )
        assert got_mode == mode, (got_mode, mode)
    else:
        assert "quant_output" not in compiled[-1]
    tile = compile_tile(**compiled[-1])
    traits = [t for t, v in kernels._GFX1250_PA_DECODE_CACHE.items() if v is tile]
    assert traits and traits[0].compute_dtype == want, (traits, want)
