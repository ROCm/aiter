# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""fused_qknorm_idxrqknorm writing NVFP4 K/V in the row layout (gfx950, gfx1250).

kv_cache_dtype="nvfp4" with byte k_scale / v_scale planes: per (page, kv head) an e2m1
payload [128, 64] and E4M3 scales [128, 8], the layout of MiniMax-M3's sparse NVFP4
pool and of the FlyDSL paged decode's NVFP4 K/V. Every check is bitwise against the
same call with kv_cache_dtype="auto" into page-128 model-dtype caches, quantized by a
torch reference of the rule (scale = e4m3(amax / 6), codes = e2m1(x * (1 / scale))
half to even, IEEE divisions): the K/V bytes, and q_out / index_q_out / the index
cache unchanged. Covers bf16 / fp16, one and two kv heads, an fp8 and a bf16 index,
pad slots (-1), strided pool views (K/V and scales of one entry-major pool), the
index-less (skip_index_branch) call, split-K partials and the multi-row launch.

index_cache_dtype="nvfp4" with index_scale writes the index key in the same row
layout (payload [pages, 128, 64] + E4M3 [pages, 128, 8], strided pool fields): bitwise
against "auto" into a page-128 index cache quantized by the same rule, a code's sign
taken from x * (1 / scale) (ATOM's index_fp4_store; a zero-scale block stores +0
codes there, where the K/V rows store -0).

Run with: python -m pytest -q op_tests/test_fused_qknorm_idxrqknorm_nvfp4_rows.py
"""

import itertools
import os
import subprocess
import sys
import tempfile

import pytest
import torch

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx_runtime

D, PAGE = 128, 128


@pytest.fixture(scope="module", autouse=True)
def _gpu():
    if not torch.cuda.is_available():
        pytest.skip("ROCm is not available")
    if get_gfx_runtime() not in ("gfx950", "gfx1250"):
        pytest.skip("the NVFP4 row layout runs on gfx950 and gfx1250")


def nvfp4_rows_ref(x, sign_of_scaled=False):
    """[..., D] model-dtype rows -> (payload uint8 [..., D/2], scales uint8 [..., D/16]).
    ``sign_of_scaled``: a code's sign from x * (1 / scale) (the index rows) rather
    than x (K/V)."""
    blocks = x.float().unflatten(-1, (-1, 16))
    scale = (blocks.abs().amax(-1) / 6.0).clamp(max=448.0).to(torch.float8_e4m3fn)
    s = scale.float()
    inv = torch.where(s > 0, 1.0 / s, torch.zeros_like(s))
    y = blocks * inv[..., None]
    a = y.abs()
    mag = sum((a > t).int() for t in (0.25, 1.25, 2.5, 5.0)) + sum(
        (a >= t).int() for t in (0.75, 1.75, 3.5)
    )
    sign = torch.where((y if sign_of_scaled else blocks) < 0, 8, 0)
    codes = (mag | sign).flatten(-2).unflatten(-1, (-1, 2))
    payload = (codes[..., 0] | (codes[..., 1] << 4)).to(torch.uint8)
    return payload, scale.view(torch.uint8)


def make_pool(pages, nkv, device):
    """K / V payloads and scale planes as fields of one entry-major pool (one entry a
    page), the strides MhaKvPool's field views have."""
    pay, sc = nkv * PAGE * (D // 2), nkv * PAGE * (D // 16)
    entry = 2 * pay + 2 * sc + 64  # a pad: page strides wider than the fields
    pool = torch.full((pages, entry), 0xEE, dtype=torch.uint8, device=device)

    def field(offset, width):
        return pool[:, offset : offset + nkv * PAGE * width].unflatten(
            1, (nkv, PAGE, width)
        )

    kp = field(0, D // 2)
    vp = field(pay, D // 2)
    ks = field(2 * pay, D // 16)
    vs = field(2 * pay + sc, D // 16)
    return pool, kp, ks, vp, vs


def make_layer(m, nq, nkv, dtype, device, seed, rotary_dim=64):
    g = torch.Generator(device=device).manual_seed(seed)
    row = (nq + 2 * nkv + nkv + 1) * D
    qkv = torch.randn(m, row, generator=g, device=device).to(dtype)
    # Spread the V rows (raw) and the K rows (normed) over E4M3's range.
    mags = torch.exp2(
        torch.randint(-8, 6, (m, row // D, 1), generator=g, device=device)
    )
    qkv = (qkv.view(m, -1, D) * mags).view(m, row).to(dtype)
    qkv[0, (nq + nkv) * D : (nq + nkv + 1) * D] = 0  # an all-zero V row
    w = [
        (torch.randn(D, generator=g, device=device) * 0.5 + 1).to(dtype)
        for _ in range(4)
    ]
    cos_sin = torch.randn(4096, rotary_dim, generator=g, device=device).to(dtype)
    pos = torch.randint(0, 4096, (m,), generator=g, device=device)
    return {
        "qkv": qkv,
        "w": w,
        "cos_sin": cos_sin,
        "pos": pos,
        "nq": nq,
        "nkv": nkv,
        "rot": rotary_dim,
    }


def call(layer, slots, k, v, index, *, kv_dtype, k_scale=None, v_scale=None,
         index_dtype="auto", skip_index=False, partials=None, index_scale=None):  # fmt: skip
    m, nq, nkv = layer["qkv"].shape[0], layer["nq"], layer["nkv"]
    qkv = layer["qkv"].clone()
    q_out = torch.empty(m, nq * D, dtype=qkv.dtype, device=qkv.device)
    iq_out = torch.empty(m, nkv * D, dtype=qkv.dtype, device=qkv.device)
    w = layer["w"]
    aiter.fused_qknorm_idxrqknorm(
        qkv, w[0], w[1], layer["cos_sin"], layer["pos"], nq, nkv, layer["rot"],
        1e-6, w[2], w[3], nkv, slots, k, v, index, PAGE, q_out, iq_out, slots,
        kv_cache_dtype=kv_dtype, index_cache_dtype=index_dtype,
        k_scale=k_scale, v_scale=v_scale, asm_layout=False,
        skip_index_branch=skip_index,
        **({} if partials is None else {"qkv_partials": partials}),
        **({} if index_scale is None else {"index_scale": index_scale}),
    )  # fmt: skip
    torch.cuda.synchronize()
    return q_out, iq_out


CASES = list(
    itertools.product(
        [torch.bfloat16, torch.float16],  # model dtype
        [1, 2],  # kv heads (M3.1 TP4 has one)
        ["fp8", "auto"],  # index cache
        [37, 3000],  # tokens: one row a warp, and the multi-row launch on gfx1250
    )
)


@pytest.mark.parametrize("dtype, nkv, index_dtype, m", CASES)
def test_nvfp4_rows_equal_auto_then_quantize(dtype, nkv, index_dtype, m):
    dev = torch.device("cuda")
    layer = make_layer(m, 16 * nkv, nkv, dtype, dev, seed=m + nkv)
    pages = (m + PAGE - 1) // PAGE + 3
    slots = torch.randperm(pages * PAGE, device=dev)[:m].to(torch.int64)
    slots[1::17] = -1  # pad rows: nothing written
    idx_t = dtypes.fp8 if index_dtype == "fp8" else dtype

    # The unfused reference: "auto" K/V into page-128 caches, then quantize.
    ref_k = torch.zeros(pages, PAGE, nkv, D, dtype=dtype, device=dev)
    ref_v = torch.zeros_like(ref_k)
    ref_index = torch.zeros(pages * PAGE, D, dtype=idx_t, device=dev)
    ref_q, ref_iq = call(
        layer, slots, ref_k, ref_v, ref_index, kv_dtype="auto", index_dtype=index_dtype
    )

    pool, kp, ks, vp, vs = make_pool(pages, nkv, dev)
    untouched = pool.clone()
    index = torch.zeros_like(ref_index)
    # The scale planes may be uint8 or float8_e4m3fn views.
    sdt = torch.float8_e4m3fn if index_dtype == "fp8" else torch.uint8
    q, iq = call(
        layer, slots, kp, vp, index, kv_dtype="nvfp4",
        k_scale=ks.view(sdt), v_scale=vs.view(sdt), index_dtype=index_dtype,
    )  # fmt: skip
    assert torch.equal(q, ref_q) and torch.equal(iq, ref_iq)
    assert torch.equal(index.view(torch.uint8), ref_index.view(torch.uint8))
    live = slots >= 0
    page, pos = slots[live] // PAGE, slots[live] % PAGE
    for cache, payload, scales in ((ref_k, kp, ks), (ref_v, vp, vs)):
        want_p, want_s = nvfp4_rows_ref(cache[page, pos])  # [rows, nkv, ...]
        assert torch.equal(payload[page, :, pos], want_p)
        assert torch.equal(scales[page, :, pos], want_s)
    # Nothing else of the pool moved: pad slots, unused rows, the entry pad.
    written = torch.zeros_like(pool, dtype=torch.bool)
    for t in (kp, ks, vp, vs):
        mask = torch.zeros(t.shape, dtype=torch.bool, device=dev)
        mask[page, :, pos] = True
        written.as_strided(t.shape, t.stride(), t.storage_offset())[mask] = True
    assert torch.equal(pool[~written], untouched[~written])


def test_nvfp4_rows_skip_index_branch():
    dev = torch.device("cuda")
    m, pages = 200, 4
    layer = make_layer(m, 16, 1, torch.bfloat16, dev, seed=7)
    slots = torch.randperm(pages * PAGE, device=dev)[:m].to(torch.int64)
    ref_k = torch.zeros(pages, PAGE, 1, D, dtype=torch.bfloat16, device=dev)
    ref_v = torch.zeros_like(ref_k)
    ref_q, _ = call(layer, slots, ref_k, ref_v, None, kv_dtype="auto", skip_index=True)
    _, kp, ks, vp, vs = make_pool(pages, 1, dev)
    q, _ = call(
        layer, slots, kp, vp, None, kv_dtype="nvfp4", k_scale=ks, v_scale=vs,
        skip_index=True,
    )  # fmt: skip
    assert torch.equal(q, ref_q)
    page, pos = slots // PAGE, slots % PAGE
    for cache, payload, scales in ((ref_k, kp, ks), (ref_v, vp, vs)):
        want_p, want_s = nvfp4_rows_ref(cache[page, pos])
        assert torch.equal(payload[page, :, pos], want_p)
        assert torch.equal(scales[page, :, pos], want_s)


@pytest.mark.parametrize("splits", [2, 4, 8])
def test_nvfp4_rows_with_splitk_partials(splits):
    """qkv_partials: the reduced row, as the FP8 path takes it; same bytes as the
    call on the reduced qkv."""
    dev = torch.device("cuda")
    m, pages, nkv = 64, 2, 1
    layer = make_layer(m, 16, nkv, torch.bfloat16, dev, seed=splits)
    row = layer["qkv"].shape[1]
    g = torch.Generator(device=dev).manual_seed(11)
    partials = torch.randn(splits, m, row, generator=g, device=dev)
    reduced = layer["qkv"].clone()
    acc = partials[0].clone()
    for s in range(1, splits):
        acc += partials[s]
    reduced.copy_(acc.to(torch.bfloat16))
    slots = torch.randperm(pages * PAGE, device=dev)[:m].to(torch.int64)
    index_ref = torch.zeros(pages * PAGE, D, dtype=dtypes.fp8, device=dev)
    _, kp0, ks0, vp0, vs0 = make_pool(pages, nkv, dev)
    q0, _ = call(
        dict(layer, qkv=reduced), slots, kp0, vp0, index_ref, kv_dtype="nvfp4",
        k_scale=ks0, v_scale=vs0, index_dtype="fp8",
    )  # fmt: skip
    index = torch.zeros_like(index_ref)
    _, kp, ks, vp, vs = make_pool(pages, nkv, dev)
    q, _ = call(
        layer, slots, kp, vp, index, kv_dtype="nvfp4", k_scale=ks, v_scale=vs,
        index_dtype="fp8", partials=partials,
    )  # fmt: skip
    assert torch.equal(q, q0)
    assert torch.equal(index.view(torch.uint8), index_ref.view(torch.uint8))
    for a, b in ((kp, kp0), (ks, ks0), (vp, vp0), (vs, vs0)):
        assert torch.equal(a, b)


def make_index_pool(pages, device):
    """The NVFP4 index rows as two fields of one entry-major pool, as ATOM binds
    them (payload [pages, 128, 64], scales [pages, 128, 8])."""
    pay, sc = PAGE * (D // 2), PAGE * (D // 16)
    pool = torch.full((pages, pay + sc + 48), 0xEE, dtype=torch.uint8, device=device)
    ip = pool[:, :pay].unflatten(1, (PAGE, D // 2))
    isc = pool[:, pay : pay + sc].unflatten(1, (PAGE, D // 16))
    return pool, ip, isc


@pytest.mark.parametrize(
    "dtype, m, splits",
    [
        (torch.bfloat16, 37, 0),
        (torch.bfloat16, 3000, 0),  # the multi-row launch on gfx1250
        (torch.float16, 37, 0),
        (torch.float16, 3000, 0),
        (torch.bfloat16, 64, 4),  # split-K partials
    ],
)
def test_nvfp4_index_rows_equal_auto_then_quantize(dtype, m, splits):
    """The index key in the NVFP4 row layout, K/V too: index bytes as index_fp4_store
    makes them of the "auto" row, K/V bytes and q / index q as the fp8-index call."""
    dev = torch.device("cuda")
    nkv = 1
    layer = make_layer(m, 16, nkv, dtype, dev, seed=m + splits)
    # Index keys near zero in some rows: zero-scale and subnormal-scale blocks.
    ik = slice((16 + 2 * nkv + nkv) * D, (16 + 2 * nkv + nkv + 1) * D)
    layer["qkv"][2::5, ik] = (layer["qkv"][2::5, ik].float() * 1e-4).to(dtype)
    pages = (m + PAGE - 1) // PAGE + 3
    slots = torch.randperm(pages * PAGE, device=dev)[:m].to(torch.int64)
    slots[1::17] = -1
    partials = None
    if splits:
        g = torch.Generator(device=dev).manual_seed(5)
        partials = torch.randn(splits, m, layer["qkv"].shape[1], generator=g, device=dev)
        acc = partials[0].clone()
        for i in range(1, splits):
            acc += partials[i]
        layer = dict(layer, qkv=acc.to(dtype))  # the reference reads the reduced row

    ref_k = torch.zeros(pages, PAGE, nkv, D, dtype=dtype, device=dev)
    ref_v = torch.zeros_like(ref_k)
    ref_index = torch.zeros(pages * PAGE, D, dtype=dtype, device=dev)
    ref_q, ref_iq = call(layer, slots, ref_k, ref_v, ref_index, kv_dtype="auto")

    _, kp, ks, vp, vs = make_pool(pages, nkv, dev)
    ipool, ip, isc = make_index_pool(pages, dev)
    untouched = ipool.clone()
    fused_layer = layer if not splits else dict(layer, qkv=torch.zeros_like(layer["qkv"]))
    q, iq = call(
        fused_layer, slots, kp, vp, ip, kv_dtype="nvfp4", k_scale=ks, v_scale=vs,
        index_dtype="nvfp4", index_scale=isc, partials=partials,
    )  # fmt: skip
    assert torch.equal(q, ref_q) and torch.equal(iq, ref_iq)
    live = slots >= 0
    page, pos = slots[live] // PAGE, slots[live] % PAGE
    want_p, want_s = nvfp4_rows_ref(ref_index[slots[live]], sign_of_scaled=True)
    assert torch.equal(ip[page, pos], want_p)
    assert torch.equal(isc[page, pos], want_s)
    for cache, payload, scales in ((ref_k, kp, ks), (ref_v, vp, vs)):
        wp, ws = nvfp4_rows_ref(cache[page, pos])
        assert torch.equal(payload[page, :, pos], wp)
        assert torch.equal(scales[page, :, pos], ws)
    written = torch.zeros_like(ipool, dtype=torch.bool)
    for t in (ip, isc):
        mask = torch.zeros(t.shape, dtype=torch.bool, device=dev)
        mask[page, pos] = True
        written.as_strided(t.shape, t.stride(), t.storage_offset())[mask] = True
    assert torch.equal(ipool[~written], untouched[~written])


REJECT_CHILD = (
    "import sys, torch, aiter; a, kw = torch.load(sys.argv[1]); "
    "aiter.fused_qknorm_idxrqknorm(*a, **kw); torch.cuda.synchronize()"
)


def test_nvfp4_rows_rejects():
    """Mismatched K / V scale-plane strides, a page other than 128 and an fp4 index
    with row-layout K/V fail on the host. A failed host check aborts the process, so
    each call runs in a child ("good" shows the harness is sound)."""
    dev = torch.device("cuda")
    m, nq, nkv = 4, 16, 1
    layer = make_layer(m, nq, nkv, torch.bfloat16, dev, seed=3)
    slots = torch.arange(m, dtype=torch.int64, device=dev)
    _, kp, ks, vp, vs = make_pool(2, nkv, dev)
    index = torch.zeros(2 * PAGE, D, dtype=dtypes.fp8, device=dev)
    w = layer["w"]

    _, ip, isc = make_index_pool(2, dev)

    def args(v_scale=vs, block_size=PAGE, index_dtype="fp8", index=index, index_scale=None):
        q_out = torch.empty(m, nq * D, dtype=torch.bfloat16, device=dev)
        iq = torch.empty(m, nkv * D, dtype=torch.bfloat16, device=dev)
        a = (
            layer["qkv"].clone(), w[0], w[1], layer["cos_sin"], layer["pos"], nq,
            nkv, layer["rot"], 1e-6, w[2], w[3], nkv, slots, kp, vp, index,
            block_size, q_out, iq, slots,
        )  # fmt: skip
        kw = {
            "kv_cache_dtype": "nvfp4", "index_cache_dtype": index_dtype, "k_scale": ks,
            "v_scale": v_scale, "asm_layout": False,
        }  # fmt: skip
        if index_scale is not None:
            kw["index_scale"] = index_scale
        return a, kw

    cases = {
        "good": args(),
        "scale_strides": args(v_scale=vs.contiguous()),
        "page_64": args(block_size=64),
        "fp4_index": args(index_dtype="nvfp4"),
        "good_fp4_index_rows": args(index_dtype="nvfp4", index=ip, index_scale=isc),
        # index_scale with an fp8 index, and a scale plane of the wrong width.
        "fp4_index_rows_fp8": args(index_dtype="fp8", index_scale=isc),
        "fp4_index_rows_shape": args(
            index_dtype="nvfp4", index=ip, index_scale=isc[:, :, :4]
        ),
    }
    with tempfile.TemporaryDirectory() as tmp:
        for name, call_args in cases.items():
            path = os.path.join(tmp, name)
            torch.save(call_args, path)
            cmd = [sys.executable, "-c", REJECT_CHILD, path]
            r = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if name.startswith("good"):
                assert r.returncode == 0, r.stderr[-500:]
            else:
                assert r.returncode != 0, f"{name}: accepted"
                assert "fp4" in r.stderr or "128" in r.stderr, r.stderr[-500:]
