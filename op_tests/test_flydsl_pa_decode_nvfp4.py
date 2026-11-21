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
from aiter.test_common import benchmark

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


# --- MXFP8 output (o_proj's native MXFP8 input) ---------------------------------
#
# An E8M0 ``output_scale`` [rows, H * D / 32] selects it. The bytes must equal the
# split path the model ran before: the BF16 PA output of the same call, then the
# group-32 quant ATOM's MXFP8 o_proj applies to it (``atom.model_ops.blockscale.
# quantize_fp8``): ceil-power-of-two E8M0 scales of max(amax, 1e-10) / 448.


def mxfp8_reference(x):
    """[rows, K] BF16/FP16 -> (E4M3 bytes [rows, K], E8M0 bytes [rows, K / 32])."""
    rows, k = x.shape
    groups = x.float().view(rows, k // 32, 32)
    amax = groups.abs().amax(-1).clamp_min(1e-10)
    bits = (amax * torch.tensor(1.0 / 448.0, dtype=torch.float32)).view(torch.int32)
    code = (bits >> 23) + ((bits & 0x7FFFFF) != 0).int()
    reciprocal = ((254 - code).clamp_min(0) << 23).view(torch.float32)
    q = (groups * reciprocal[..., None]).clamp(-448.0, 448.0)
    q = q.to(torch.float8_e4m3fn).view(torch.uint8).view(rows, k)
    return q, code.to(torch.uint8)


def _split_quant_kernel():
    """The split path's standalone quant: a copy of ATOM's ``quantize_fp8_kernel``
    (one launch, 32 groups a program), timed as the kernel the fusion removes."""
    import triton
    import triton.language as tl

    @triton.jit
    def quantize_fp8_kernel(X, Y, S, SIZE):
        groups = tl.program_id(0) * 32 + tl.arange(0, 32)
        offsets = groups[:, None] * 32 + tl.arange(0, 32)[None, :]
        x = tl.load(X + offsets, offsets < SIZE, other=0).to(tl.float32)
        amax = tl.maximum(tl.max(tl.abs(x), 1), 1e-10)
        bits = (amax * (1.0 / 448.0)).to(tl.uint32, bitcast=True)
        code = (bits >> 23) + ((bits & 0x7FFFFF) != 0).to(tl.uint32)
        reciprocal = (tl.maximum(254 - code.to(tl.int32), 0) << 23).to(
            tl.float32, bitcast=True
        )
        q = tl.minimum(tl.maximum(x * reciprocal[:, None], -448.0), 448.0)
        tl.store(Y + offsets, q.to(tl.float8e4nv), offsets < SIZE)
        tl.store(S + groups, code.to(tl.uint8), groups < SIZE // 32)

    def quantize(x, y, s):
        quantize_fp8_kernel[(triton.cdiv(x.numel(), 1024),)](x, y, s, x.numel())

    return quantize


# Every output mode the plan can pick, as ATOM's sparse PA plans them: prefill
# rows one partition (direct), decode rows a cluster of 8 while they fit one
# resident wave of clusters, else the default split (clusters shared by several
# rows, or the reducer). The other cluster sizes and the last-arriver merge come
# from short contexts and one slot per row.
MXFP8_CASES = {
    "direct-prefill": (2048, {"max_partitions": 1, "max_len": 1024}, "direct"),
    "direct-decode": (32, {"max_partitions": 1}, "direct"),
    "cluster8-c4-mtp8": (8, {"max_partitions": 8, "workgroup_budget": 64}, "cluster8"),
    "cluster8-c24-mtp8": (
        192,
        {"max_partitions": 8, "workgroup_budget": 1536},
        "cluster8",
    ),
    "cluster4-short": (
        64,
        {"max_partitions": 8, "workgroup_budget": 512, "max_len": 1024},
        "cluster4",
    ),
    "last-arriver-merge": (8, {"max_partitions": 8, "workgroup_budget": 1}, "merge"),
    # 512 rows exceed one resident wave of clusters: four rows a cluster of 8.
    "shared-cluster-c64-mtp8": (
        512,
        {"max_partitions": 2, "workgroup_budget": 1024},
        "cluster2x4",
    ),
    "shared-cluster3-c64-mtp8": (
        512,
        {"max_partitions": 3, "workgroup_budget": 1536},
        "cluster3x4",
    ),
    "shared-cluster8": (
        300,
        {"max_partitions": 8, "workgroup_budget": 4096},
        "cluster8x2",
    ),
    # Past two sequences a cluster of 8: the reducer.
    "reducer-np8": (600, {"max_partitions": 8, "workgroup_budget": 4800}, "reduce"),
}


def _mode(kwargs):
    if kwargs.get("cluster_merge"):
        shared = kwargs.get("cluster_seqs", 1)
        return f"cluster{kwargs['cluster_merge']}" + (
            f"x{shared}" if shared > 1 else ""
        )
    if kwargs.get("direct_output"):
        return "direct" if kwargs.get("num_partitions") == 1 else "merge"
    return "reduce"


@pytest.mark.parametrize("store_output", [True, False], ids=["store", "quantonly"])
@pytest.mark.parametrize("query_dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("case", list(MXFP8_CASES))
def test_pa_decode_nvfp4_mxfp8_output(
    case, query_dtype, store_output, device, monkeypatch
):
    """E8M0 ``output_scale``: the MXFP8 bytes and scales equal the BF16 (FP16)
    output of the same call quantized group-32 as the split path does, in every
    output mode (the wave tile's epilogue, its merges, the reducer's); padded
    rows are FP8 zeros with E8M0 1.0 scales."""
    from aiter.utility import dtypes

    pa = importlib.import_module("aiter.ops.flydsl.pa_decode")
    rows, plan_kwargs, mode = MXFP8_CASES[case]
    plan_kwargs = dict(plan_kwargs)
    max_len = plan_kwargs.pop("max_len", None)
    topk = 16
    num_pages = rows * topk + 8
    cache = make_cache(num_pages, 1, device)
    table, lengths = sparse_rows(rows, topk, num_pages, device, 1, max_len)
    poison_tails(cache, table, lengths)
    kp, ks, vp, vs = cache
    g = torch.Generator(device=device).manual_seed(4)
    query = torch.randn(rows, 16, D, device=device, generator=g).to(query_dtype)
    # Outputs of very different magnitude per head, as o_proj inputs have.
    query[:, ::5] *= 8
    plan = pa.plan_pa_decode(lengths, 1, **plan_kwargs)
    common = {"key_scale": ks, "value_scale": vs, "work_plan": plan}
    bound = max_len or topk * PAGE

    reference_out = torch.full_like(query, float("nan"))
    pa.pa_decode(
        reference_out, query, kp, vp, lengths, table, D**-0.5, 1,
        max_context_length=bound, **common,
    )  # fmt: skip
    ref_q, ref_s = mxfp8_reference(reference_out.view(rows, -1))
    split_q = torch.empty(rows, 16 * D, dtype=dtypes.fp8, device=device)
    split_s = torch.empty(rows, 16 * D // 32, dtype=torch.uint8, device=device)
    _split_quant_kernel()(reference_out.view(rows, -1), split_q, split_s)
    assert torch.equal(split_q.view(torch.uint8), ref_q), "reference != split quant"
    assert torch.equal(split_s, ref_s), "reference != split quant scales"

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
        scales = torch.full(
            (rows + pad, 16 * D // 32), 0xFF, dtype=torch.uint8, device=device
        ).view(dtypes.fp8_e8m0)
        pa.pa_decode(
            output, query, kp, vp, lengths, table, D**-0.5, 1,
            max_context_length=bound, quantized_output=quantized,
            output_scale=scales, store_output=store_output,
            quantized_pad_rows=pad, **common,
        )  # fmt: skip
        torch.cuda.synchronize()
        got_q = quantized[:rows].view(rows, -1).view(torch.uint8)
        got_s = scales[:rows].view(torch.uint8)
        bad = (got_q != ref_q).sum().item()
        assert bad == 0, f"{bad} FP8 bytes differ"
        assert torch.equal(got_s, ref_s), "E8M0 scales differ"
        pad_bytes = quantized[rows:].view(torch.uint8)
        assert torch.equal(pad_bytes, torch.zeros_like(pad_bytes))
        assert (scales[rows:].view(torch.uint8) == 127).all()
        if store_output:
            assert torch.equal(output, reference_out), "BF16 output differs"
    got_mode = _mode(compiled[-1])
    assert got_mode == mode, (got_mode, mode)
    assert bool(compiled[-1].get("quant_mx")) == (mode != "reduce")


def test_pa_decode_mxfp8_rejects(device):
    """A wrongly shaped E8M0 scale is refused."""
    from aiter.utility import dtypes

    pa = importlib.import_module("aiter.ops.flydsl.pa_decode")
    kp, ks, vp, vs = make_cache(8, 1, device)
    lengths = torch.full((2,), 256, dtype=torch.int32, device=device)
    table = torch.arange(4, dtype=torch.int32, device=device).reshape(2, 2)
    query = torch.randn(2, 16, D, device=device, dtype=torch.bfloat16)
    plan = pa.plan_pa_decode(lengths, 1)
    with pytest.raises(ValueError):
        pa.pa_decode(
            torch.empty_like(query), query, kp, vp, lengths, table, 0.1, 1,
            key_scale=ks, value_scale=vs, work_plan=plan,
            quantized_output=torch.empty_like(query, dtype=dtypes.fp8),
            output_scale=torch.empty(2, 16, dtype=dtypes.fp8_e8m0, device=device),
        )  # fmt: skip


# (rows, max_partitions, workgroup_budget): MiniMax-M3.1 TP4 sparse layers as ATOM
# plans them (top-16 blocks a row, so the context length beyond the window does
# not matter). Decode rows are DSpark K7 verify tokens (8 a request).
# None: ATOM's default split (``get_recommended_splits``) and budget.
BENCH_SHAPES = {
    "c4-decode-bs1": (8, 8, 64),
    "c64-decode-bs24": (192, 8, 1536),
    "c64-decode-bs64": (512, None, None),
    # The traced c64 step: three partitions a row, merged by the reducer.
    "c64-decode-bs64-np3": (512, 3, 1536),
    "prefill-2048": (2048, 1, None),
    "prefill-4536": (4536, 1, None),
}


@benchmark()
def bench_mxfp8_output(shape, rows, query_dtype=torch.bfloat16):
    """The MXFP8-output PA against the split path it replaces (BF16 PA, then
    the group-32 quant) and the BF16 PA alone; the fused bytes must equal the
    split path's."""
    pa = importlib.import_module("aiter.ops.flydsl.pa_decode")
    from aiter.ops.triton.gluon.pa_decode_gluon import get_recommended_splits
    from aiter.test_common import run_perftest
    from aiter.utility import dtypes

    _, max_partitions, budget = BENCH_SHAPES[shape]
    device = torch.device("cuda")
    if max_partitions is None:
        max_partitions = get_recommended_splits(rows, 1)
    topk = 16
    num_pages = rows * topk + 8
    kp, ks, vp, vs = make_cache(num_pages, 1, device)
    table, lengths = sparse_rows(rows, topk, num_pages, device, 1)
    lengths.fill_(topk * PAGE)  # decode and long-prefix rows attend a full window
    query = torch.randn(rows, 16, D, device=device).to(query_dtype)
    plan = pa.plan_pa_decode(
        lengths,
        1,
        max_partitions=max_partitions,
        **({} if budget is None else {"workgroup_budget": budget}),
        # ATOM raises the planner's batch bound for long prefills.
        **({"max_batch_size": 8192} if rows > 4096 else {}),
    )
    common = {
        "key_scale": ks,
        "value_scale": vs,
        "work_plan": plan,
        "max_context_length": topk * PAGE,
    }
    output = torch.empty_like(query)
    quantized = torch.empty(rows, 16, D, dtype=dtypes.fp8, device=device)
    scales = torch.empty(rows, 16 * D // 32, dtype=dtypes.fp8_e8m0, device=device)
    split_q = torch.empty(rows, 16 * D, dtype=dtypes.fp8, device=device)
    split_s = torch.empty(rows, 16 * D // 32, dtype=torch.uint8, device=device)
    quantize = _split_quant_kernel()

    def pa_bf16():
        pa.pa_decode(output, query, kp, vp, lengths, table, D**-0.5, 1, **common)

    def split():
        pa_bf16()
        quantize(output.view(rows, -1), split_q, split_s)

    def fused():
        pa.pa_decode(
            output, query, kp, vp, lengths, table, D**-0.5, 1,
            quantized_output=quantized, output_scale=scales, store_output=False,
            **common,
        )  # fmt: skip

    candidates = {"pa_bf16": pa_bf16, "split": split, "fused": fused}
    ret = {"gfx": get_gfx_runtime()}
    # KV bytes the rows read (payload + scales) plus the output written.
    kv_bytes = rows * topk * PAGE * 2 * (D // 2 + D // 16)
    for name, fn in candidates.items():
        _, us = run_perftest(fn)
        out_bytes = rows * 16 * D * (2 if name == "pa_bf16" else 1)
        ret[f"{name} us"] = us
        ret[f"{name} TB/s"] = (kv_bytes + out_bytes) / us / 1e6
    split()
    fused()
    torch.cuda.synchronize()
    bad = quantized.view(rows, -1).view(torch.uint8) != split_q.view(torch.uint8)
    err = bad.float().mean().item()
    err += (scales.view(torch.uint8) != split_s).float().mean().item()
    assert err == 0, f"fused MXFP8 differs from the split path ({err:.2e})"
    ret["fused err"] = err
    return ret


def main():
    import argparse

    import pandas as pd

    import aiter

    if get_gfx_runtime() != "gfx1250":
        aiter.logger.warning("NVFP4 pa_decode runs on gfx1250; skipping")
        return
    parser = argparse.ArgumentParser(description="NVFP4 PA MXFP8 output vs split")
    parser.add_argument(
        "--shape", nargs="*", default=list(BENCH_SHAPES), choices=list(BENCH_SHAPES)
    )
    args = parser.parse_args()
    from aiter.ops.flydsl.kernels import pa_decode_kernel as kernels

    pa = importlib.import_module("aiter.ops.flydsl.pa_decode")
    original = kernels.compile_pa_decode_tile
    compiled = []

    def spy(**kwargs):
        compiled.append(kwargs)
        return original(**kwargs)

    df = []
    for shape in args.shape:
        compiled.clear()
        pa.compile_pa_decode_tile = spy
        try:
            row = bench_mxfp8_output(shape, BENCH_SHAPES[shape][0])
        finally:
            pa.compile_pa_decode_tile = original
        # The mode of the MXFP8 call: a tile with the epilogue, else the reducer.
        mx = [k for k in compiled if k.get("quant_mx")]
        row["mode"] = _mode(mx[-1]) if mx else "reduce"
        df.append(row)
    df = pd.DataFrame(df)
    aiter.logger.info("NVFP4 PA MXFP8 output summary:\n%s", df.to_markdown(index=False))


if __name__ == "__main__":
    main()
