# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""fused_qknorm_idxrqknorm writing MXFP4 caches and an e4m3 main Q.

``index_cache_dtype="mxfp4"`` and ``kv_cache_dtype="mxfp4"`` store the post norm + RoPE
index K, main K and the raw V as uint8 pages of 128 tokens, ``[128 x 64 B e2m1][128 x 4 B
e8m0]`` (8704 B), the low nibble the even element. The judge is the "auto" mode of the SAME
op: ``per_1x32_f4_quant`` (aiter's torch MXFP4 reference) applied to the rows "auto" writes
is what the MXFP4 bytes must equal, bit for bit, and every other output must not move.
An e4m3 ``q_out`` must equal the input-dtype q cast to e4m3, saturating at its max.
"""

import argparse
import itertools
import os
import subprocess
import sys

import pandas as pd
import torch

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.quant import per_1x32_f4_quant
from aiter.test_common import benchmark, run_perftest
from aiter.utility import fp4_utils

torch.set_default_device("cuda")

SUPPORTED_GFX = ["gfx950"]
NQ, NIQ, HD, ROT = 16, 1, 128, 64  # MiniMax-M3 TP4
PAGE_BYTES = 128 * (64 + 4)
SENTINEL = 0xA5
DEFAULT_DTYPES = [dtypes.bf16, dtypes.fp16]
DEFAULT_TOKENS = [1, 300, 1031]
DEFAULT_PERF_TOKENS = [1, 32, 128, 1024, 4096, 16384]

INDEX_CASES = ("sequential", "index_skipped", "differ")
KV_CASES = ("sequential", "shuffled", "main_skipped", "differ")
# (kv_cache_dtype, block_size, index_cache_dtype, slot case, kv heads)
CONFIGS = [
    (kv, bs, index, case, nkv)
    for kv, bs, index, cases, nkvs in (
        ("fp8", 16, "mxfp4", INDEX_CASES, (1,)),
        ("auto", 16, "mxfp4", INDEX_CASES, (1,)),
        ("mxfp4", 128, "mxfp4", KV_CASES, (1, 2)),
        ("mxfp4", 128, "fp8", KV_CASES, (1, 2)),
        ("mxfp4", 128, "auto", KV_CASES, (1, 2)),
    )
    for case in cases
    for nkv in nkvs
]


def make_inputs(T, nkv, dtype, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    row = (NQ + 2 * nkv + NIQ + 1) * HD
    qkv = torch.randn(T, row, dtype=dtype, generator=g)
    w = (torch.randn(HD, dtype=dtype, generator=g) * 0.1).to(dtype)
    cs = torch.randn(max(T, 64), ROT, dtype=dtype, generator=g)
    return qkv, w, cs, torch.arange(T, dtype=torch.int64)


def slot_cases(T):
    """(main slots, index slots) per case; -1 marks a skipped token."""
    seq = torch.arange(T, dtype=torch.int64)
    perm = torch.randperm(((T + 127) // 128 + 1) * 128)[:T].to(torch.int64)
    skipped = perm.clone()
    skipped[::3] = -1
    return {
        "sequential": (seq, seq),
        "shuffled": (perm, perm),
        "index_skipped": (seq, skipped),
        "main_skipped": (skipped, skipped),
        "differ": (perm.flip(0).clone(), seq),
    }


def make_call(
    inputs,
    nkv,
    slots,
    kv,
    bs,
    index_dtype,
    dtype,
    index_slots=None,
    skip_index=False,
    ic=None,
    pool=False,
    q_out=None,
):
    """A closure running the op once, and the tensors it writes (caches, q, index q).

    kv "auto"/"fp8" with bs 16 is the page-16 asm layout, "auto" with bs 128 the plain
    page-128 layout, "mxfp4" the 8704 B pages (pool: K and V as views of one pool)."""
    qkv, w, cs, pos = inputs
    T = qkv.shape[0]
    index_slots = slots if index_slots is None else index_slots
    n = max(int(max(slots.max(), index_slots.max())) + 1, T)
    nb, nb128 = -(-n // bs), -(-n // 128)
    ks = vs = None
    if kv == "mxfp4":
        kvp = torch.full((nb, 2, nkv, PAGE_BYTES), SENTINEL, dtype=torch.uint8)
        kc, vc = (
            (kvp[:, 0], kvp[:, 1]) if pool else (kvp[:, 0].clone(), kvp[:, 1].clone())
        )
    elif bs == 16:
        kvdt = dtypes.fp8 if kv == "fp8" else dtype
        x = 16 // torch.empty(0, dtype=kvdt).element_size()
        kc = torch.zeros(nb, nkv, HD // x, 16, x, dtype=kvdt)
        vc = torch.zeros(nb, nkv, 16 // x, HD, x, dtype=kvdt)
        if kv == "fp8":
            ks = torch.zeros(nb, nkv, 16, dtype=torch.float32)
            vs = torch.zeros_like(ks)
    else:
        kc = torch.zeros(nb, 128, nkv, HD, dtype=dtype)
        vc = torch.zeros_like(kc)
    if ic is None:
        ic = {
            "mxfp4": lambda: torch.full(
                (nb128, PAGE_BYTES), SENTINEL, dtype=torch.uint8
            ),
            "fp8": lambda: torch.zeros(nb128, 128, HD, dtype=dtypes.fp8),
            "auto": lambda: torch.zeros(nb128, 128, HD, dtype=dtype),
        }[index_dtype]()
    out = {
        "q": torch.empty(T, NQ * HD, dtype=dtype),
        "iq": torch.empty(T, NIQ * HD, dtype=dtype),
        "ic": ic, "kc": kc, "vc": vc, "ks": ks, "vs": vs,
    }  # fmt: skip
    q_out = out["q"] if q_out is None else q_out

    def call():
        aiter.fused_qknorm_idxrqknorm(
            qkv, w, w, cs, pos, NQ, nkv, ROT, 1e-6, w, w, NIQ,
            slots, out["kc"], out["vc"], out["ic"], bs, q_out, out["iq"], index_slots,
            kv_cache_dtype=kv, index_cache_dtype=index_dtype, k_scale=ks, v_scale=vs,
            asm_layout=bs == 16, skip_index_branch=skip_index,
        )  # fmt: skip

    return call, out


def run_op(*args, **kwargs):
    call, out = make_call(*args, **kwargs)
    call()
    return out


def expected_pages(auto_cache, slots):
    """The pages the MXFP4 write must produce for the rows an "auto" cache
    ([pages, 128, kv heads, 128]) holds at `slots`; every other byte stays SENTINEL."""
    nb, _, nkv, _ = auto_cache.shape
    want = torch.full((nb, nkv, PAGE_BYTES), SENTINEL, dtype=torch.uint8)
    ok = slots[slots >= 0]
    if ok.numel() == 0:  # per_1x32_f4_quant cannot take zero rows
        return want
    rows = auto_cache[ok // 128, ok % 128].reshape(-1, HD)
    q, s = per_1x32_f4_quant(rows)
    data = q.view(torch.uint8).reshape(-1, nkv, 64)
    scale = s.view(torch.uint8).reshape(-1, 4)[: rows.shape[0]].reshape(-1, nkv, 4)
    # An all-zero group: per_1x32_f4_quant yields e8m0 0, which the fp4 scorer does not
    # support; the op's rule (indexer_fp4_scale_e8m0 in cache_kernels.cu) clamps amax to eps
    # and yields 1. Data bytes are 0 either way.
    zero = rows.float().reshape(-1, nkv, 4, 32).abs().amax(-1) == 0
    scale = torch.where(zero & (scale == 0), torch.ones_like(scale), scale)
    strides = (nkv * PAGE_BYTES, PAGE_BYTES)
    d = torch.as_strided(want, (nb, nkv, 128, 64), (*strides, 64, 1))
    sc = torch.as_strided(want, (nb, nkv, 128, 4), (*strides, 4, 1), 128 * 64)
    d[ok // 128, :, ok % 128] = data
    sc[ok // 128, :, ok % 128] = scale
    return want


def index_pages(ic_auto, slots):
    return expected_pages(ic_auto[:, :, None], slots).view(-1, PAGE_BYTES)


def same(a, b):
    return torch.equal(a.view(torch.uint8), b.view(torch.uint8))


@benchmark()
def test_write_bytes(T, nkv, dtype, kv, bs, index_dtype, case):
    torch.manual_seed(T)
    slots, index_slots = slot_cases(T)[case]
    inputs = make_inputs(T, nkv, dtype, seed=T)
    auto = lambda d: "auto" if d == "mxfp4" else d
    base = run_op(
        inputs, nkv, slots, auto(kv), bs, auto(index_dtype), dtype, index_slots
    )
    got = run_op(inputs, nkv, slots, kv, bs, index_dtype, dtype, index_slots)
    for name, ref in base.items():
        if ref is None:
            continue
        if name == "ic" and index_dtype == "mxfp4":
            ref = index_pages(ref, index_slots)
        elif name in ("kc", "vc") and kv == "mxfp4":
            ref = expected_pages(ref, slots)
        assert same(got[name], ref), f"{name} differs from the auto-mode reference"
    return {"result": "PASS"}


@benchmark()
def test_skip_index_branch(dtype):
    """A layer that reuses another layer's top-k skips the index slots, not the main KV."""
    T = 300
    slots = slot_cases(T)["shuffled"][0]
    inputs = make_inputs(T, 1, dtype)
    base = run_op(inputs, 1, slots, "auto", 128, "auto", dtype, skip_index=True)
    got = run_op(inputs, 1, slots, "mxfp4", 128, "auto", dtype, skip_index=True)
    for name in ("kc", "vc"):
        assert same(got[name], expected_pages(base[name], slots)), name
    return {"result": "PASS"}


def test_views_into_one_pool():
    """K and V as strided views of one pool ([pages, 2, kv heads, 8704]) land right."""
    T, nkv, dtype = 300, 2, torch.bfloat16
    slots = slot_cases(T)["shuffled"][0]
    inputs = make_inputs(T, nkv, dtype)
    base = run_op(inputs, nkv, slots, "auto", 128, "auto", dtype)
    got = run_op(inputs, nkv, slots, "mxfp4", 128, "auto", dtype, pool=True)
    for name in ("kc", "vc"):
        assert same(got[name], expected_pages(base[name], slots)), name
    aiter.logger.info("test_views_into_one_pool: PASS")


@benchmark()
def test_q_fp8_out(dtype, kv, q_dtype):
    """An e4m3 q_out gets the input-dtype q cast to e4m3, unit scale and saturating (+-inf too, NaN
    stays NaN), and replaces the bf16 q: the packed qkv row is not written back either.
    """
    T = 300
    slots = slot_cases(T)["shuffled"][0]
    fp8_max = torch.finfo(
        dtypes.fp8
    ).max  # not at import: dtypes.fp8 is uint8 off gfx94x/950
    qkv, w, cs, pos = make_inputs(T, 1, dtype)
    w = w * 5000  # the norm erases a scaled qkv, so scale its weight to pass 448
    w[5], w[70] = float("inf"), float("nan")  # and put +-inf and NaN in two columns
    inputs = (qkv, w, cs, pos)
    qkv = qkv.clone()
    q8 = torch.empty(T, NQ * HD, dtype=q_dtype)
    run_op(inputs, 1, slots, kv, 128, "auto", dtype, q_out=q8)
    q_ref = run_op(inputs, 1, slots, kv, 128, "auto", dtype)["q"]
    qf = q_ref.float()
    assert (qf.abs() > fp8_max).any() and qf.isinf().any() and qf.isnan().any()
    want, nan = qf.clamp(-fp8_max, fp8_max).to(dtypes.fp8), qf.isnan()
    assert same(q8[~nan], want[~nan])
    assert (
        q8.view(dtypes.fp8)[nan].float().isnan().all()
    )  # NaN stays NaN, its sign is free
    assert torch.equal(inputs[0], qkv)
    return {"result": "PASS"}


def test_existing_index_modes_unchanged():
    """ "auto" and "fp8" write what they always did: the fp8 rows are the cast of the
    "auto" rows, and nothing else moved between the two."""
    T, dtype = 300, torch.bfloat16
    slots = torch.arange(T, dtype=torch.int64)
    inputs = make_inputs(T, 1, dtype)
    auto = run_op(inputs, 1, slots, "fp8", 16, "auto", dtype)
    fp8 = run_op(inputs, 1, slots, "fp8", 16, "fp8", dtype)
    assert same(fp8["ic"], auto["ic"].to(dtypes.fp8))
    assert same(auto["q"], fp8["q"]) and same(auto["iq"], fp8["iq"])
    aiter.logger.info("test_existing_index_modes_unchanged: PASS")


def test_rewriting_a_slot_replaces_all_68_bytes():
    """Overwriting a token with different content leaves no stale data or scale byte."""
    T, dtype = 300, torch.bfloat16
    slots = torch.arange(T, dtype=torch.int64)
    a = make_inputs(T, 1, dtype, seed=1)
    b = make_inputs(T, 1, dtype, seed=2)
    b = (b[0] * 8, *b[1:])  # so b's e8m0 bytes differ from a's everywhere
    ic_b = run_op(b, 1, slots, "auto", 16, "auto", dtype)["ic"]
    ic = run_op(a, 1, slots, "auto", 16, "mxfp4", dtype)["ic"]
    run_op(b, 1, slots, "auto", 16, "mxfp4", dtype, ic=ic)  # b over a, same pages
    assert torch.equal(ic, index_pages(ic_b, slots))
    aiter.logger.info("test_rewriting_a_slot_replaces_all_68_bytes: PASS")


def test_zero_and_extreme_rows():
    """amax 0 (eps clamp), sparse rows and a single outlier go through the same e8m0
    rule. (The op RMS-normalises the row first, so magnitude cannot be swept here.)"""
    T, dtype = 64, torch.bfloat16
    slots = torch.arange(T, dtype=torch.int64)
    inputs = make_inputs(T, 1, dtype)
    ik = inputs[0][:, -HD:]
    ik[0] = 0
    ik[2, ::2] = 0  # half of every group's elements zero
    ik[3, 5] = 3e4  # one outlier in an otherwise ordinary row
    auto = run_op(inputs, 1, slots, "auto", 16, "auto", dtype)["ic"]
    got = run_op(inputs, 1, slots, "auto", 16, "mxfp4", dtype)["ic"]
    assert torch.equal(got, index_pages(auto, slots))
    aiter.logger.info("test_zero_and_extreme_rows: PASS")


def test_pages_dequantize_to_the_reference():
    """Read the pages back the way a consumer addresses them (data at token * 64, scale
    at 128 * 64 + token * 4), dequantize in torch, and compare with the dequantized
    reference quantization of the "auto" rows."""
    T, dtype = 300, torch.bfloat16
    slots = torch.arange(T, dtype=torch.int64)
    inputs = make_inputs(T, 1, dtype)
    auto = run_op(inputs, 1, slots, "auto", 16, "auto", dtype)["ic"]
    ic = run_op(inputs, 1, slots, "auto", 16, "mxfp4", dtype)["ic"]
    pages = ic.shape[0]
    kd = torch.as_strided(ic, (pages, 128, 64), (PAGE_BYTES, 64, 1))
    ks = torch.as_strided(ic, (pages, 128, 4), (PAGE_BYTES, 4, 1), 128 * 64)

    def dequant(data, scale):
        f = fp4_utils.mxfp4_to_f32(data.reshape(-1, 64).clone())
        e = fp4_utils.e8m0_to_f32(scale.reshape(-1, 4)).repeat_interleave(32, -1)
        return f * e

    qz, s = per_1x32_f4_quant(auto.reshape(-1, HD))
    ref = dequant(qz.view(torch.uint8), s.view(torch.uint8))[:T]
    assert torch.equal(dequant(kd, ks)[:T], ref)
    aiter.logger.info("test_pages_dequantize_to_the_reference: PASS")


def test_reference_scale_rule_is_the_hip_rule():
    """The e8m0 rule in the op (mxfp4ScaleE8m0) is the one in cache_kernels.cu's
    indexer_fp4_scale_e8m0; this pins the torch quantizer the byte-equality tests use
    to that rule on its bit-level definition."""
    torch.manual_seed(0)
    x = torch.randn(4096, HD) * torch.exp2(torch.randint(-12, 12, (4096, 1)).float())
    x = x.to(torch.bfloat16)
    _, s = per_1x32_f4_quant(x)
    amax = x.float().reshape(-1, 32).abs().amax(-1)
    eps = 6.0 * torch.tensor(2.0**-126)
    bits = (torch.maximum(amax, eps) * torch.tensor(1.0 / 6.0)).view(torch.int32)
    e = ((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0).to(torch.int32)
    assert torch.equal(e.to(torch.uint8), s.view(torch.uint8).reshape(-1))
    aiter.logger.info("test_reference_scale_rule_is_the_hip_rule: PASS")


def rejection_call(case):
    """One mxfp4 call that is (or, for "good", is not) a contract violation."""
    T, dtype = 32, torch.bfloat16
    slots = torch.arange(T, dtype=torch.int64)
    kv, bs = ("mxfp4", 128) if case.startswith("kv_mxfp4") else ("auto", 16)
    if case in ("kv_mxfp4_few_index_pages", "kv_mxfp4_v_pages"):
        slots += 200  # two kv pages
    ic = {
        "not_uint8": torch.zeros(1, 128, HD, dtype=dtype),
        "partial_page": torch.zeros(PAGE_BYTES + 128, dtype=torch.uint8),
        "kv_mxfp4_small_index": torch.zeros(128, dtype=torch.uint8),
        "kv_mxfp4_few_index_pages": torch.zeros(PAGE_BYTES, dtype=torch.uint8),
    }.get(case)
    call, out = make_call(
        make_inputs(T, 1, dtype), 1, slots, kv, bs, "mxfp4", dtype,
        ic=ic, skip_index=case == "skip_index_branch",
    )  # fmt: skip
    if case == "kv_mxfp4_v_pages":
        out["vc"] = out["vc"][:1]  # one V page for two K pages
    call()
    torch.cuda.synchronize()


def test_arch_guards():
    """Where opus converts to fp4/fp8 as zeros, those stores must be refused up front."""
    import aiter.ops.fused_qknorm_idxrqknorm as op

    T, bf16 = 32, torch.bfloat16
    slots = torch.arange(T, dtype=torch.int64)
    q8 = torch.empty(T, NQ * HD, dtype=dtypes.fp8)
    runtime = op.get_gfx_runtime
    try:
        for gfx, kv, bs, index, q_out, msg in (
            ("gfx942", "auto", 16, "mxfp4", None, "gfx950"),
            ("gfx942", "mxfp4", 128, "auto", None, "gfx950"),
            ("gfx1100", "auto", 128, "auto", q8, "fp8 q_out"),
        ):
            op.get_gfx_runtime = lambda gfx=gfx: gfx
            try:
                run_op(
                    make_inputs(T, 1, bf16), 1, slots, kv, bs, index, bf16, q_out=q_out
                )
            except RuntimeError as e:
                assert msg in str(e), e
            else:
                raise AssertionError(
                    f"accepted on {gfx}: {kv} {index} {q_out is not None}"
                )
    finally:
        op.get_gfx_runtime = runtime
    aiter.logger.info("test_arch_guards: PASS")


@benchmark()
def test_rejections(case):
    """AITER_CHECK aborts the process rather than raising, so each case runs in its own
    interpreter; "good" proves the harness itself is sound."""
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
    return {"result": "PASS"}


@benchmark()
def test_write_perf(T, kv):
    """Cost of the index-K write by index cache dtype (M3 TP4 shape)."""
    dtype = torch.bfloat16
    slots = torch.arange(T, dtype=torch.int64)
    kv_row = HD * (1 if kv == "fp8" else 2)
    ret = {"gfx": get_gfx()}
    for name, index_row in (("auto", 2 * HD), ("fp8", HD), ("mxfp4", 68)):
        call, _ = make_call(make_inputs(T, 1, dtype), 1, slots, kv, 16, name, dtype)
        _, us = run_perftest(call, num_iters=20, num_warmup=5)
        # bytes: qkv read; q, index q, K, V and index K written
        nbytes = T * (2 * (NQ + 2 + NIQ + 1) * HD + 2 * (NQ + NIQ) * HD + 2 * kv_row)
        ret[f"{name} us"] = us
        ret[f"{name} TB/s"] = (nbytes + T * index_row) / us / 1e6
    return ret


def summary(name, rows):
    df = pd.DataFrame(rows)
    aiter.logger.info("%s summary (markdown):\n%s", name, df.to_markdown(index=False))


def sweep(dtype_list, tokens, perf_tokens):
    summary(
        "write bytes",
        [
            test_write_bytes(T, nkv, dtype, kv, bs, index, case)
            for dtype, T, (kv, bs, index, case, nkv) in itertools.product(
                dtype_list, tokens, CONFIGS
            )
        ],
    )
    summary("skip_index_branch", [test_skip_index_branch(d) for d in dtype_list])
    summary(
        "e4m3 q_out",
        [
            test_q_fp8_out(*cfg)
            for cfg in itertools.product(
                dtype_list, ("auto", "mxfp4"), (dtypes.fp8, torch.uint8)
            )
        ],
    )
    test_views_into_one_pool()
    test_existing_index_modes_unchanged()
    test_rewriting_a_slot_replaces_all_68_bytes()
    test_zero_and_extreme_rows()
    test_pages_dequantize_to_the_reference()
    test_reference_scale_rule_is_the_hip_rule()
    test_arch_guards()
    summary(
        "rejections",
        [
            test_rejections(c)
            for c in (
                "good",
                "not_uint8",
                "partial_page",
                "skip_index_branch",
                "kv_mxfp4_small_index",
                "kv_mxfp4_few_index_pages",
                "kv_mxfp4_v_pages",
            )
        ],
    )
    summary(
        "index-K write",
        [
            test_write_perf(T, kv)
            for kv, T in itertools.product(("fp8", "auto"), perf_tokens)
        ],
    )


def main():
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning(
            "fused_qknorm_idxrqknorm mxfp4 unsupported on %s; skipping", get_gfx()
        )
        return

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="Test fused_qknorm_idxrqknorm MXFP4 caches and e4m3 q_out",
    )
    parser.add_argument(
        "-d",
        "--dtype",
        type=dtypes.str2Dtype,
        choices=[dtypes.bf16, dtypes.fp16],
        nargs="*",
        default=DEFAULT_DTYPES,
        help="Input dtype. e.g.: -d bf16 fp16",
    )
    parser.add_argument(
        "-t",
        "--tokens",
        type=int,
        nargs="*",
        default=DEFAULT_TOKENS,
        help="Tokens of the byte-equality sweep. e.g.: -t 1 300",
    )
    parser.add_argument(
        "--perf_tokens",
        type=int,
        nargs="*",
        default=DEFAULT_PERF_TOKENS,
        help="Tokens of the index-K write timing. e.g.: --perf_tokens 1 4096",
    )
    args = parser.parse_args()
    sweep(args.dtype, args.tokens, args.perf_tokens)


if __name__ == "__main__":
    main()
