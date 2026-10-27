# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""fused_qknorm_idxrqknorm writing an MXFP4 index cache (index_cache_dtype="mxfp4").

The judge is the "auto" mode of the SAME op: it writes the post-norm/rope index K as
bf16/fp16, and `per_1x32_f4_quant` (aiter's torch MXFP4 reference, RoundUp) applied to
that row is what the "mxfp4" bytes must equal, bit for bit. Everything else the op
produces (q, index q, main K/V and scales) must not move at all, since MXFP4 only
changes how the index-K slot is stored.

Page layout under test: one 128-token page is [128 x 64 B e2m1][128 x 4 B e8m0]
(8704 B), low nibble the even element; it is what `score_flydsl_fp4` reads.
"""

import argparse
import itertools
import sys

import pandas as pd
import pytest
import torch

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.quant import per_1x32_f4_quant
from aiter.test_common import benchmark, run_perftest
from aiter.utility import fp4_utils

NQ, NKV, NIQ, HD, ROT = 16, 1, 1, 128, 64  # MiniMax-M3 TP4
PAGE_BYTES = 128 * (64 + 4)
SENTINEL = 0xA5
requires_gfx950 = pytest.mark.skipif(
    not torch.cuda.is_available() or get_gfx() != "gfx950",
    reason="requires gfx950",
)


def make_inputs(T, dtype, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    row = (NQ + 2 * NKV + NIQ + 1) * HD
    qkv = torch.randn(T, row, dtype=dtype, device="cuda", generator=g)
    w = (torch.randn(HD, dtype=dtype, device="cuda", generator=g) * 0.1).to(dtype)
    cs = torch.randn(max(T, 64), ROT, dtype=dtype, device="cuda", generator=g)
    return qkv, w, cs, torch.arange(T, dtype=torch.int64, device="cuda")


def run_op(inputs, slots, index_slots, kv, index_dtype, dtype):
    """One call of the fused op. Returns the cache tensors and every other output."""
    qkv, w, cs, pos = inputs
    T = qkv.shape[0]
    cap = int(max(slots.max(), index_slots.max())) + 1
    nb16 = -(-max(cap, T) // 16)
    nb128 = -(-max(cap, T) // 128)
    kvdt = dtypes.fp8 if kv == "fp8" else dtype
    x = 16 // torch.empty(0, dtype=kvdt).element_size()
    kc = torch.zeros(nb16, NKV, HD // x, 16, x, dtype=kvdt, device="cuda")
    vc = torch.zeros(nb16, NKV, 16 // x, HD, x, dtype=kvdt, device="cuda")
    ks = (
        torch.zeros(nb16, NKV, 16, dtype=torch.float32, device="cuda")
        if kv == "fp8"
        else None
    )
    vs = (
        torch.zeros(nb16, NKV, 16, dtype=torch.float32, device="cuda")
        if kv == "fp8"
        else None
    )
    if index_dtype == "mxfp4":
        ic = torch.full((nb128, PAGE_BYTES), SENTINEL, dtype=torch.uint8, device="cuda")
    elif index_dtype == "fp8":
        ic = torch.zeros(nb128, 128, HD, dtype=dtypes.fp8, device="cuda")
    else:
        ic = torch.zeros(nb128, 128, HD, dtype=dtype, device="cuda")
    qo = torch.empty(T, NQ * HD, dtype=dtype, device="cuda")
    iq = torch.empty(T, NIQ * HD, dtype=dtype, device="cuda")
    aiter.fused_qknorm_idxrqknorm(
        qkv.clone(), w, w, cs, pos, NQ, NKV, ROT, 1e-6, w, w, NIQ,
        slots, kc, vc, ic, 16, qo, iq, index_slots,
        kv_cache_dtype="fp8" if kv == "fp8" else "auto",
        index_cache_dtype=index_dtype, k_scale=ks, v_scale=vs, asm_layout=True,
    )  # fmt: skip
    return ic, {"q": qo, "iq": iq, "kc": kc, "vc": vc, "ks": ks, "vs": vs}


def expected_pages(ic_auto, index_slots, nb128):
    """The page bytes the MXFP4 write must produce: the reference quantizer on the
    "auto" rows, scattered to the written slots; every other byte stays SENTINEL."""
    rows = ic_auto.reshape(-1, HD)
    q, s = per_1x32_f4_quant(rows)
    data = q.view(torch.uint8).reshape(-1, 64)
    scale = s.view(torch.uint8).reshape(-1, 4)
    # An all-zero group: per_1x32_f4_quant yields e8m0 0, which the fp4 scorer does not
    # support; the HIP rule (indexer_fp4_scale_e8m0, copied into the op) clamps amax
    # to eps and yields 1. Data bytes are 0 either way.
    zero_group = rows.float().reshape(-1, 4, 32).abs().amax(-1) == 0
    scale = torch.where(zero_group & (scale == 0), torch.ones_like(scale), scale)
    want = torch.full((nb128, PAGE_BYTES), SENTINEL, dtype=torch.uint8, device="cuda")
    ok = index_slots[index_slots >= 0]
    page, tok = ok // 128, ok % 128
    for p, t, slot in zip(page.tolist(), tok.tolist(), ok.tolist()):
        want[p, t * 64 : (t + 1) * 64] = data[slot]
        want[p, 128 * 64 + t * 4 : 128 * 64 + (t + 1) * 4] = scale[slot]
    return want


def slot_cases(T):
    seq = torch.arange(T, dtype=torch.int64, device="cuda")
    perm = torch.randperm(((T + 127) // 128 + 1) * 128, device="cuda")[:T].to(
        torch.int64
    )
    skipped = perm.clone()
    skipped[::3] = -1
    return {
        "sequential": (seq, seq),
        "shuffled_slots": (seq, perm),
        "skipped_slots": (seq, skipped),
        "main_and_index_differ": (perm.flip(0).clone(), seq),
    }


@requires_gfx950
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
@pytest.mark.parametrize("kv", ["fp8", "auto"])
@pytest.mark.parametrize("T", [1, 7, 128, 300, 1031])
@pytest.mark.parametrize(
    "case", ["sequential", "shuffled_slots", "skipped_slots", "main_and_index_differ"]
)
def test_bytes_equal_the_reference_quantizer(dtype, kv, T, case):
    torch.manual_seed(T)
    slots, index_slots = slot_cases(T)[case]
    inputs = make_inputs(T, dtype, seed=T)
    ic_auto, out_auto = run_op(inputs, slots, index_slots, kv, "auto", dtype)
    ic_fp4, out_fp4 = run_op(inputs, slots, index_slots, kv, "mxfp4", dtype)
    # MXFP4 changes how the index-K slot is stored and nothing else.
    for name, ref in out_auto.items():
        if ref is not None:
            assert torch.equal(
                ref.view(torch.uint8) if ref.dtype != torch.uint8 else ref,
                (
                    out_fp4[name].view(torch.uint8)
                    if out_fp4[name].dtype != torch.uint8
                    else out_fp4[name]
                ),
            ), name
    want = expected_pages(ic_auto, index_slots, ic_fp4.shape[0])
    assert torch.equal(ic_fp4, want)


@requires_gfx950
def test_existing_index_modes_unchanged():
    """ "auto" and "fp8" write what they always did: the fp8 rows are the cast of the
    "auto" rows, and nothing else moved between the two."""
    T, dtype = 300, torch.bfloat16
    slots = torch.arange(T, dtype=torch.int64, device="cuda")
    inputs = make_inputs(T, dtype)
    ic_auto, out_auto = run_op(inputs, slots, slots, "fp8", "auto", dtype)
    ic_fp8, out_fp8 = run_op(inputs, slots, slots, "fp8", "fp8", dtype)
    assert torch.equal(
        ic_fp8.view(torch.uint8), ic_auto.to(dtypes.fp8).view(torch.uint8)
    )
    assert torch.equal(out_auto["q"], out_fp8["q"]) and torch.equal(
        out_auto["iq"], out_fp8["iq"]
    )


@requires_gfx950
def test_rewriting_a_slot_replaces_all_68_bytes():
    """Overwriting a token with different content leaves no stale data or scale byte."""
    T, dtype = 300, torch.bfloat16
    slots = torch.arange(T, dtype=torch.int64, device="cuda")
    a, b = make_inputs(T, dtype, seed=1), make_inputs(T, dtype, seed=2)
    # Scale the second batch so its e8m0 bytes differ from the first's everywhere.
    b = (b[0] * 8, *b[1:])
    nb128 = -(-T // 128)
    ic = torch.full((nb128, PAGE_BYTES), SENTINEL, dtype=torch.uint8, device="cuda")
    ic_b_auto, _ = run_op(b, slots, slots, "auto", "auto", dtype)
    for inputs in (a, b):  # write a, then b over it, into the same pages
        qkv, w, cs, pos = inputs
        nb16 = -(-T // 16)
        kc = torch.zeros(nb16, NKV, HD // 8, 16, 8, dtype=dtype, device="cuda")
        vc = torch.zeros(nb16, NKV, 16 // 8, HD, 8, dtype=dtype, device="cuda")
        aiter.fused_qknorm_idxrqknorm(
            qkv.clone(), w, w, cs, pos, NQ, NKV, ROT, 1e-6, w, w, NIQ,
            slots, kc, vc, ic, 16, torch.empty(T, NQ * HD, dtype=dtype, device="cuda"),
            torch.empty(T, NIQ * HD, dtype=dtype, device="cuda"), slots,
            kv_cache_dtype="auto", index_cache_dtype="mxfp4", asm_layout=True,
        )  # fmt: skip
    assert torch.equal(ic, expected_pages(ic_b_auto, slots, nb128))


@requires_gfx950
def test_zero_and_extreme_rows():
    """amax 0 (eps clamp), sparse rows and a single outlier go through the same e8m0
    rule. (The op RMS-normalises the row first, so magnitude cannot be swept here.)"""
    T, dtype = 64, torch.bfloat16
    slots = torch.arange(T, dtype=torch.int64, device="cuda")
    qkv, w, cs, pos = make_inputs(T, dtype)
    ik = qkv[:, -HD:]
    ik[0] = 0
    ik[2, ::2] = 0  # half the groups' elements zero
    ik[3, 5] = 3e4  # one outlier in an otherwise ordinary row
    inputs = (qkv, w, cs, pos)
    ic_auto, _ = run_op(inputs, slots, slots, "auto", "auto", dtype)
    ic_fp4, _ = run_op(inputs, slots, slots, "auto", "mxfp4", dtype)
    assert torch.equal(ic_fp4, expected_pages(ic_auto, slots, ic_fp4.shape[0]))


@requires_gfx950
def test_end_to_end_with_the_fp4_decode_scorer():
    """Write with the op, read with `score_flydsl_fp4`; the scores are those of
    quantize -> dequantize -> bf16 scoring of the "auto" rows."""
    from aiter.ops.flydsl.kernels.minimax_m3_index_score_fp4 import score_flydsl_fp4

    T, dtype = 300, torch.bfloat16
    slots = torch.arange(T, dtype=torch.int64, device="cuda")
    inputs = make_inputs(T, dtype)
    ic_auto, _ = run_op(inputs, slots, slots, "auto", "auto", dtype)
    ic_fp4, _ = run_op(inputs, slots, slots, "auto", "mxfp4", dtype)
    pages = ic_fp4.shape[0]
    kd = torch.as_strided(ic_fp4, (pages, 128, 64), (PAGE_BYTES, 64, 1))
    ks = torch.as_strided(ic_fp4, (pages, 128, 4), (PAGE_BYTES, 4, 1), 128 * 64)
    q = torch.randn(1, 1, HD, dtype=torch.bfloat16, device="cuda")
    bt = torch.arange(pages, dtype=torch.int32, device="cuda")[None]
    seq = torch.tensor([T], dtype=torch.int32, device="cuda")
    got = score_flydsl_fp4(q, kd, ks, bt, seq, 1, 1, HD**-0.5, pages)
    rows = ic_auto.reshape(-1, HD)[:T]
    qz, s = per_1x32_f4_quant(ic_auto.reshape(-1, HD))
    f = fp4_utils.mxfp4_to_f32(qz.view(torch.uint8).reshape(-1, 64).clone())
    e = fp4_utils.e8m0_to_f32(s.view(torch.uint8).reshape(-1, 4)).repeat_interleave(
        32, -1
    )
    deq = (f * e).to(torch.bfloat16)[:T]
    assert deq.shape == rows.shape
    z = (deq.float() @ q.reshape(HD).float()) * (HD**-0.5 * 1.4426950409)
    z = torch.cat([z, torch.full((pages * 128 - T,), float("-inf"), device="cuda")])
    ref = z.reshape(pages, 128).amax(-1)
    assert torch.allclose(got.reshape(-1), ref, atol=3e-5, rtol=3e-5)


@requires_gfx950
def test_writer_fp8_index_q_is_the_scorers_e4m3_query():
    """The index-Q the op writes as e4m3 is exactly the bf16 row rounded to e4m3 with
    saturation, and `score_flydsl_fp4` consumes that buffer as is: the scores are
    those of the dequantized index K against that e4m3 query."""
    from aiter.ops.flydsl.kernels.minimax_m3_index_score_fp4 import score_flydsl_fp4

    T, dtype = 300, torch.bfloat16
    slots = torch.arange(T, dtype=torch.int64, device="cuda")
    qkv, w, cs, pos = make_inputs(T, dtype)
    nb = -(-T // 16)
    kc = torch.zeros(nb, NKV, HD // 8, 16, 8, dtype=dtype, device="cuda")
    vc = torch.zeros(nb, NKV, 16 // 8, HD, 8, dtype=dtype, device="cuda")
    ic = torch.full(
        (-(-T // 128), PAGE_BYTES), SENTINEL, dtype=torch.uint8, device="cuda"
    )
    iq_bf16 = torch.empty(T, NIQ * HD, dtype=dtype, device="cuda")
    iq_e4m3 = torch.empty(T, NIQ * HD, dtype=dtypes.fp8, device="cuda")
    qo = torch.empty(T, NQ * HD, dtype=dtype, device="cuda")
    for iq in (iq_bf16, iq_e4m3):
        aiter.fused_qknorm_idxrqknorm(
            qkv.clone(), w, w, cs, pos, NQ, NKV, ROT, 1e-6, w, w, NIQ,
            slots, kc, vc, ic, 16, qo, iq, slots,
            kv_cache_dtype="auto", index_cache_dtype="mxfp4", asm_layout=True,
        )  # fmt: skip
    want = iq_bf16.float().clamp(-448, 448).to(dtypes.fp8)
    assert torch.equal(iq_e4m3.view(torch.uint8), want.view(torch.uint8))
    # ...and the decode scorer reads it: row t of request 0 is token t's query.
    pages = ic.shape[0]
    kd = torch.as_strided(ic, (pages, 128, 64), (PAGE_BYTES, 64, 1))
    ks = torch.as_strided(ic, (pages, 128, 4), (PAGE_BYTES, 4, 1), 128 * 64)
    bt = torch.arange(pages, dtype=torch.int32, device="cuda")[None]
    seq = torch.tensor([T], dtype=torch.int32, device="cuda")
    q8 = iq_e4m3[T - 1 : T].view(1, 1, HD)
    got = score_flydsl_fp4(q8, kd, ks, bt, seq, 1, 1, HD**-0.5, pages)
    keys = torch.empty(pages * 128, HD, dtype=dtype, device="cuda")
    ic_auto, _ = run_op((qkv, w, cs, pos), slots, slots, "auto", "auto", dtype)
    qz, s = per_1x32_f4_quant(ic_auto.reshape(-1, HD))
    f = fp4_utils.mxfp4_to_f32(qz.view(torch.uint8).reshape(-1, 64).clone())
    e = fp4_utils.e8m0_to_f32(s.view(torch.uint8).reshape(-1, 4)).repeat_interleave(
        32, -1
    )
    keys.copy_((f * e).to(dtype))
    z = (keys[:T].float() @ q8.reshape(HD).to(dtype).float()) * (
        HD**-0.5 * 1.4426950409
    )
    z = torch.cat([z, torch.full((pages * 128 - T,), float("-inf"), device="cuda")])
    assert torch.allclose(
        got.reshape(-1), z.reshape(pages, 128).amax(-1), atol=3e-5, rtol=3e-5
    )


def test_reference_scale_rule_is_the_hip_rule():
    """The e8m0 rule copied into the op (mxfp4ScaleE8m0) is the one in
    cache_kernels.cu's indexer_fp4_scale_e8m0; this pins the torch quantizer, which
    the byte-equality tests above use, to that rule on its bit-level definition."""
    torch.manual_seed(0)
    x = (
        torch.randn(4096, HD) * torch.exp2(torch.randint(-12, 12, (4096, 1)).float())
    ).to(torch.bfloat16)
    _, s = per_1x32_f4_quant(x)
    amax = x.float().reshape(-1, 32).abs().amax(-1)
    eps = 6.0 * torch.tensor(2.0**-126)
    bits = (
        torch.maximum(amax, eps) * torch.tensor(1.0 / 6.0, dtype=torch.float32)
    ).view(torch.int32)
    e = ((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0).to(torch.int32)
    assert torch.equal(e.to(torch.uint8), s.view(torch.uint8).reshape(-1))


def rejection_call(case):
    """One mxfp4 call that is (or, for "good", is not) a contract violation."""
    T, dtype = 32, torch.bfloat16
    slots = torch.arange(T, dtype=torch.int64, device="cuda")
    qkv, w, cs, pos = make_inputs(T, dtype)
    kc = torch.zeros(2, NKV, HD // 8, 16, 8, dtype=dtype, device="cuda")
    vc = torch.zeros(2, NKV, 16 // 8, HD, 8, dtype=dtype, device="cuda")
    qo = torch.empty(T, NQ * HD, dtype=dtype, device="cuda")
    iq = torch.empty(T, NIQ * HD, dtype=dtype, device="cuda")
    ic = {
        "good": torch.zeros(1, PAGE_BYTES, dtype=torch.uint8, device="cuda"),
        "not_uint8": torch.zeros(1, 128, HD, dtype=dtype, device="cuda"),
        "partial_page": torch.zeros(PAGE_BYTES + 128, dtype=torch.uint8, device="cuda"),
    }.get(case, torch.zeros(1, PAGE_BYTES, dtype=torch.uint8, device="cuda"))
    aiter.fused_qknorm_idxrqknorm(
        qkv, w, w, cs, pos, NQ, NKV, ROT, 1e-6, w, w, NIQ,
        slots, kc, vc, ic, 16, qo, iq, slots,
        kv_cache_dtype="auto", index_cache_dtype="mxfp4", asm_layout=True,
        skip_index_branch=(case == "skip_index_branch"),
    )  # fmt: skip
    torch.cuda.synchronize()


@requires_gfx950
@pytest.mark.parametrize(
    "case", ["good", "not_uint8", "partial_page", "skip_index_branch"]
)
def test_rejections(case):
    """AITER_CHECK aborts the process rather than raising, so each case runs in its
    own interpreter; "good" proves the harness itself is sound."""
    import os
    import subprocess

    here = os.path.dirname(os.path.abspath(__file__))
    code = (
        f"import sys; sys.path.insert(0, {here!r}); "
        f"import test_fused_qknorm_idxrqknorm_mxfp4 as t; t.rejection_call({case!r})"
    )
    r = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    if case == "good":
        assert r.returncode == 0, r.stderr[-500:]
    else:
        assert r.returncode != 0
        assert "mxfp4" in r.stderr or "index" in r.stderr, r.stderr[-500:]


def write_runner(T, kv, name, dtype=torch.bfloat16):
    """A closure running the op once with index_cache_dtype=`name`."""
    slots = torch.arange(T, dtype=torch.int64, device="cuda")
    qkv, w, cs, pos = make_inputs(T, dtype)
    nb16, nb128 = -(-T // 16), -(-T // 128)
    kvdt = dtypes.fp8 if kv == "fp8" else dtype
    x = 16 // torch.empty(0, dtype=kvdt).element_size()
    kc = torch.zeros(nb16, NKV, HD // x, 16, x, dtype=kvdt, device="cuda")
    vc = torch.zeros(nb16, NKV, 16 // x, HD, x, dtype=kvdt, device="cuda")
    ks = torch.zeros(nb16, NKV, 16, device="cuda") if kv == "fp8" else None
    vs = torch.zeros(nb16, NKV, 16, device="cuda") if kv == "fp8" else None
    shape, idt = {
        "auto": ((nb128, 128, HD), dtype),
        "fp8": ((nb128, 128, HD), dtypes.fp8),
        "mxfp4": ((nb128, PAGE_BYTES), torch.uint8),
    }[name]
    ic = torch.zeros(shape, dtype=idt, device="cuda")
    qo = torch.empty(T, NQ * HD, dtype=dtype, device="cuda")
    iq = torch.empty(T, NIQ * HD, dtype=dtype, device="cuda")

    def run():
        aiter.fused_qknorm_idxrqknorm(
            qkv, w, w, cs, pos, NQ, NKV, ROT, 1e-6, w, w, NIQ,
            slots, kc, vc, ic, 16, qo, iq, slots,
            kv_cache_dtype="fp8" if kv == "fp8" else "auto",
            index_cache_dtype=name, k_scale=ks, v_scale=vs, asm_layout=True,
        )  # fmt: skip

    return run


@pytest.mark.skip(
    reason="perf sweep: run via python op_tests/test_fused_qknorm_idxrqknorm_mxfp4.py"
)
@benchmark()
def test_write_perf(T, kv):
    ret = {"gfx": get_gfx()}
    for name in ("auto", "fp8", "mxfp4"):
        _, us = run_perftest(write_runner(T, kv, name), num_iters=20, num_warmup=5)
        ret[f"{name} us"] = us
    return ret


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="fused_qknorm_idxrqknorm index-cache write sweep"
    )
    ap.add_argument(
        "--tokens", type=int, nargs="*", default=[1, 32, 128, 1024, 4096, 16384]
    )
    ap.add_argument("--kv", nargs="*", choices=["fp8", "auto"], default=["fp8", "auto"])
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if not torch.cuda.is_available() or get_gfx() != "gfx950":
        aiter.logger.warning("requires gfx950; skipping sweep")
        return 0
    rows = [test_write_perf(T, kv) for kv, T in itertools.product(args.kv, args.tokens)]
    aiter.logger.info(
        "index-cache write summary (markdown):\n%s",
        pd.DataFrame(rows).to_markdown(index=False),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
