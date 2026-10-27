# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""fused_qknorm_idxrqknorm writing an MXFP4 MAIN K/V cache and an fp8 main Q.

``kv_cache_dtype="mxfp4"``: K (post norm + RoPE) and V (the raw row) go into uint8
``[pages, kv_heads, 8704]`` caches, one 128-token page per (page, head),
``[128 x 64 B e2m1][128 x 4 B e8m0]`` with the low nibble the even element, the layout
``pa_decode_mxfp4`` reads (and the same page as the MXFP4 index cache).

The judge is the "auto" mode of the SAME op in its plain page-128 layout: it writes the
post-norm/rope K and the raw V as bf16/fp16, and ``per_1x32_f4_quant`` (aiter's torch
MXFP4 reference) applied to those rows is what the "mxfp4" bytes must equal, bit for bit.
Everything else the op produces must not move. ``q_fp8_out`` must equal the bf16 ``q_out``
rounded to e4m3 as ``bf16(q) / q_scale``, saturating at 448.
"""

import pytest
import torch

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.quant import per_1x32_f4_quant

NQ, NIQ, HD, ROT = 16, 1, 128, 64  # MiniMax-M3 TP4
PAGE_BYTES = 128 * (64 + 4)
SENTINEL = 0xA5
requires_gfx950 = pytest.mark.skipif(
    not torch.cuda.is_available() or get_gfx() != "gfx950",
    reason="requires gfx950",
)


def make_inputs(T, nkv, dtype, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    row = (NQ + 2 * nkv + NIQ + 1) * HD
    qkv = torch.randn(T, row, dtype=dtype, device="cuda", generator=g)
    w = (torch.randn(HD, dtype=dtype, device="cuda", generator=g) * 0.1).to(dtype)
    cs = torch.randn(max(T, 64), ROT, dtype=dtype, device="cuda", generator=g)
    return qkv, w, cs, torch.arange(T, dtype=torch.int64, device="cuda")


def pages_for(slots, extra=0):
    return -(-(int(slots.max()) + 1 + extra) // 128)


def run(inputs, nkv, slots, kv, index_dtype, dtype, *, skip_index=False, **extra):
    """One call of the fused op in page-128 layout. kv: 'auto' (bf16 rows) or 'mxfp4'."""
    qkv, w, cs, pos = inputs
    T = qkv.shape[0]
    nb = pages_for(slots, extra=T)
    if kv == "mxfp4":
        kc = torch.full(
            (nb, nkv, PAGE_BYTES), SENTINEL, dtype=torch.uint8, device="cuda"
        )
        vc = torch.full_like(kc, SENTINEL)
    else:
        kc = torch.zeros(nb, 128, nkv, HD, dtype=dtype, device="cuda")
        vc = torch.zeros_like(kc)
    if index_dtype == "mxfp4":
        ic = torch.full((nb, PAGE_BYTES), SENTINEL, dtype=torch.uint8, device="cuda")
    elif index_dtype == "fp8":
        ic = torch.zeros(nb, 128, HD, dtype=dtypes.fp8, device="cuda")
    else:
        ic = torch.zeros(nb, 128, HD, dtype=dtype, device="cuda")
    qo = torch.empty(T, NQ * HD, dtype=dtype, device="cuda")
    iq = torch.empty(T, NIQ * HD, dtype=dtype, device="cuda")
    aiter.fused_qknorm_idxrqknorm(
        qkv.clone(), w, w, cs, pos, NQ, nkv, ROT, 1e-6, w, w, NIQ,
        slots, kc, vc, ic, 128, extra.pop("q_out", qo), iq, slots,
        kv_cache_dtype=kv, index_cache_dtype=index_dtype, skip_index_branch=skip_index,
        **extra,
    )  # fmt: skip
    return kc, vc, ic, qo, iq


def expected_pages(rows_by_slot, slots, nb, nkv):
    """Reference bytes: per_1x32_f4_quant of each written row, scattered to its slot."""
    want = torch.full((nb, nkv, PAGE_BYTES), SENTINEL, dtype=torch.uint8, device="cuda")
    flat = rows_by_slot.reshape(-1, HD)
    q, s = per_1x32_f4_quant(flat)
    data = q.view(torch.uint8).reshape(-1, 64)
    scale = s.view(torch.uint8).reshape(-1, 4)
    # An all-zero group yields e8m0 0 in the torch reference; the writer's rule (copied
    # from indexer_fp4_scale_e8m0) clamps amax to eps and yields 1. Data bytes are 0.
    zero = flat.float().reshape(-1, 4, 32).abs().amax(-1) == 0
    scale = torch.where(zero & (scale == 0), torch.ones_like(scale), scale)
    ok = slots >= 0
    for t in range(slots.numel()):
        if not ok[t]:
            continue
        p, tok = int(slots[t]) // 128, int(slots[t]) % 128
        for h in range(nkv):
            i = t * nkv + h
            want[p, h, tok * 64 : (tok + 1) * 64] = data[i]
            want[p, h, 128 * 64 + tok * 4 : 128 * 64 + (tok + 1) * 4] = scale[i]
    return want


def rows_from_auto(cache, slots, nkv):
    """[T, nkv, 128] rows the "auto" page-128 cache holds for each token (slot < 0: zeros)."""
    out = torch.zeros(slots.numel(), nkv, HD, dtype=cache.dtype, device="cuda")
    ok = slots >= 0
    out[ok] = cache[slots[ok] // 128, slots[ok] % 128]
    return out


def slot_cases(T):
    seq = torch.arange(T, dtype=torch.int64, device="cuda")
    perm = torch.randperm(((T + 127) // 128 + 1) * 128, device="cuda")[:T].to(
        torch.int64
    )
    skipped = perm.clone()
    skipped[::3] = -1
    return {"sequential": seq, "shuffled": perm, "skipped": skipped}


@requires_gfx950
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
@pytest.mark.parametrize("nkv", [1, 2])
@pytest.mark.parametrize("T", [1, 7, 128, 300, 1031])
@pytest.mark.parametrize("case", ["sequential", "shuffled", "skipped"])
@pytest.mark.parametrize("index_dtype", ["mxfp4", "fp8", "auto"])
def test_main_kv_bytes_equal_the_reference_quantizer(dtype, nkv, T, case, index_dtype):
    torch.manual_seed(T)
    slots = slot_cases(T)[case]
    inputs = make_inputs(T, nkv, dtype, seed=T)
    kc_a, vc_a, ic_a, q_a, iq_a = run(inputs, nkv, slots, "auto", index_dtype, dtype)
    kc_f, vc_f, ic_f, q_f, iq_f = run(inputs, nkv, slots, "mxfp4", index_dtype, dtype)
    nb = kc_f.shape[0]
    assert torch.equal(
        kc_f, expected_pages(rows_from_auto(kc_a, slots, nkv), slots, nb, nkv)
    )
    assert torch.equal(
        vc_f, expected_pages(rows_from_auto(vc_a, slots, nkv), slots, nb, nkv)
    )
    # Only the main K/V storage changed: q, index q and the index cache are identical.
    assert torch.equal(q_a, q_f) and torch.equal(iq_a, iq_f)
    assert torch.equal(ic_a.view(torch.uint8), ic_f.view(torch.uint8))


@requires_gfx950
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
def test_skip_index_branch_writes_main_kv(dtype):
    """A layer that reuses another layer's top-k skips the index slots, not the main KV."""
    T, nkv = 300, 1
    slots = slot_cases(T)["shuffled"]
    inputs = make_inputs(T, nkv, dtype)
    kc_a, vc_a, *_ = run(inputs, nkv, slots, "auto", "auto", dtype, skip_index=True)
    kc_f, vc_f, *_ = run(inputs, nkv, slots, "mxfp4", "auto", dtype, skip_index=True)
    nb = kc_f.shape[0]
    assert torch.equal(
        kc_f, expected_pages(rows_from_auto(kc_a, slots, nkv), slots, nb, nkv)
    )
    assert torch.equal(
        vc_f, expected_pages(rows_from_auto(vc_a, slots, nkv), slots, nb, nkv)
    )


@requires_gfx950
def test_views_into_one_pool():
    """K and V as strided views of one pool ([pages, 2, nkv, 8704]) land in the right bytes."""
    T, nkv, dtype = 300, 2, torch.bfloat16
    slots = slot_cases(T)["shuffled"]
    inputs = make_inputs(T, nkv, dtype)
    kc_a, vc_a, *_ = run(inputs, nkv, slots, "auto", "auto", dtype)
    nb = pages_for(slots, extra=T)
    pool = torch.full(
        (nb, 2, nkv, PAGE_BYTES), SENTINEL, dtype=torch.uint8, device="cuda"
    )
    qkv, w, cs, pos = inputs
    ic = torch.zeros(nb, 128, HD, dtype=dtype, device="cuda")
    aiter.fused_qknorm_idxrqknorm(
        qkv.clone(), w, w, cs, pos, NQ, nkv, ROT, 1e-6, w, w, NIQ,
        slots, pool[:, 0], pool[:, 1], ic, 128,
        torch.empty(T, NQ * HD, dtype=dtype, device="cuda"),
        torch.empty(T, NIQ * HD, dtype=dtype, device="cuda"), slots,
        kv_cache_dtype="mxfp4", index_cache_dtype="auto",
    )  # fmt: skip
    assert torch.equal(
        pool[:, 0], expected_pages(rows_from_auto(kc_a, slots, nkv), slots, nb, nkv)
    )
    assert torch.equal(
        pool[:, 1], expected_pages(rows_from_auto(vc_a, slots, nkv), slots, nb, nkv)
    )


def q_fp8_reference(q, scale):
    x = q.float() / scale
    return x.clamp(-448.0, 448.0).to(dtypes.fp8)


@requires_gfx950
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
@pytest.mark.parametrize("kv", ["auto", "mxfp4"])
@pytest.mark.parametrize("scale", [None, 0.25, 3.0])
@pytest.mark.parametrize("bf16_q_too", [True, False])
def test_q_fp8_out(dtype, kv, scale, bf16_q_too):
    T, nkv = 300, 1
    slots = slot_cases(T)["shuffled"]
    inputs = make_inputs(T, nkv, dtype)
    # blow the first rows up so some q elements saturate at 448 / scale
    inputs[0][:8, : NQ * HD] *= 600.0
    q8 = torch.empty(T, NQ * HD, dtype=dtypes.fp8, device="cuda")
    q_scale = (
        None
        if scale is None
        else torch.tensor([scale], dtype=torch.float32, device="cuda")
    )
    kwargs = {"q_fp8_out": q8, "q_scale": q_scale}
    if not bf16_q_too:
        kwargs["q_out"] = None
    _, _, _, q_bf, _ = run(inputs, nkv, slots, kv, "auto", dtype, **kwargs)
    ref_in = inputs
    _, _, _, q_ref, _ = run(ref_in, nkv, slots, kv, "auto", dtype)  # bf16 q, no fp8
    want = q_fp8_reference(q_ref, 1.0 if scale is None else scale)
    assert torch.equal(q8.view(torch.uint8), want.view(torch.uint8))
    if bf16_q_too:
        assert torch.equal(q_bf, q_ref)
