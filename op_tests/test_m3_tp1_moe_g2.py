# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Bytes and timing of the MiniMax-M3.1 TP1 ragged-tile MoE GEMM2
(aiter.ops.m3_tp1_prefill.moe_g2_p6 and its 8-wave moe_g2_w8).

The reference is aiter's v2 "reduce" GEMM2 writing per-route rows through
``_flydsl_v2_stage2_wrapper(per_slot_out=...)``, on the deployed shape (128
experts, top-4, H 6144, I 3072, sort block 128) and the same inputs: aiter's
moe_sorting of a routing, random MXFP4 ``a2`` codes and e8m0 scales in GEMM1's
output layout (any GEMM1 output is such bytes), random MXFP4 W2 (shuffled for
aiter, ``relayout_w2`` for this kernel). Random FP4 codes make the fp32 sums
order-sensitive (reversing the K order changes about 5e-5 of the route values),
so equal bytes also mean the same accumulation order; ``wide`` spreads the ``a2``
scales over +-24 binades on top, ``same4`` routes every token to experts 0..3,
``half`` leaves experts 64.. idle and ``one`` puts expert 5 in every top-4.
Every route row and its moe_reduction must be equal, byte for byte; outputs start
as NaN. w8 is also run under HBM contention, and its A-tile publish waits are
checked in the disassembly (``isa``): each must come at least W8_PUBCNT VMEM ops
after the DMA it publishes.

    python3 op_tests/test_m3_tp1_moe_g2.py
    python3 op_tests/test_m3_tp1_moe_g2.py --quick
"""

import argparse
import os
import re
import subprocess
import sys
import tempfile

import pandas as pd
import torch

import aiter
from aiter import dtypes
from aiter.fused_moe import _flydsl_v2_stage2_wrapper, moe_sorting
from aiter.jit.core import get_user_jit_dir
from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops.flydsl.moe_kernels import _run_moe_reduction
from aiter.ops.m3_tp1_prefill import (
    MOE_G2_W8_MAX_ROUTES,
    moe_g2_p6,
    moe_g2_w8,
    relayout_w2,
)
from aiter.ops.shuffle import moe_shuffle_scale, moe_shuffle_weight
from aiter.test_common import benchmark, run_perftest

torch.set_default_device("cuda")

E, H, I, TOPK, SBM = 128, 6144, 3072, 4, 128
KN2 = "flydsl_moe2_layout_afp4_wfp4_bf16_t128x256x256_reduce_persist_sbm128"
G2 = {"p6": moe_g2_p6, "w8": moe_g2_w8}
W8_PUBCNT = 36  # moe_g2.cu: the vmcnt of w8's publish wait


def make_w2(seed):
    """Random MXFP4 W2: aiter's shuffled copy and the relayout_w2 one."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    w2_q = torch.randint(0, 256, (E, H, I // 2), dtype=torch.uint8, generator=g)
    w2_s = torch.randint(118, 127, (E, H, I // 32), dtype=torch.uint8, generator=g)
    w2 = moe_shuffle_weight(
        w2_q.view(dtypes.fp4x2), experts_cnt=E, is_guinterleave=False, gate_up=False
    )
    s2 = moe_shuffle_scale(
        w2_s.view(-1, I // 32), E, is_guinterleave=False, gate_up=False
    )
    b, bs = relayout_w2(w2_q, w2_s)
    return w2.view(torch.uint8), s2, b, bs


def make_inputs(m, seed, routing, scales):
    g = torch.Generator(device="cuda").manual_seed(seed * 7919 + m)
    logits = torch.randn(m, E, generator=g)
    if routing == "skew":
        logits += 1.5 * torch.randn(E, generator=g)
    elif routing == "same4":  # the longest expert runs
        logits[:, :TOPK] += 100.0
    elif routing == "half":
        logits[:, E // 2 :] = -1e9
    elif routing == "one":  # one expert with every token
        logits[:, 5] += 10.0
    tw, tid = torch.topk(torch.softmax(logits, -1), TOPK, dim=-1)
    sorted_ids, sorted_w, sorted_e, nvalid, _ = moe_sorting(
        tid.to(torch.int32), tw.float(), E, H, torch.bfloat16, block_size=SBM
    )
    rows = sorted_ids.numel()
    a2 = torch.randint(0, 256, (rows, I // 2), dtype=torch.uint8, generator=g)
    lo, hi = (118, 131) if scales == "narrow" else (103, 152)
    a2_scale = torch.randint(
        lo, hi, ((rows + 31) // 32 * 32, I // 32), dtype=torch.uint8, generator=g
    )
    return a2, a2_scale, sorted_ids, sorted_w, sorted_e, nvalid


def run_v2(inp, m, out=None):
    a2, a2_scale, sorted_ids, sorted_w, sorted_e, nvalid = inp
    w2, s2, _, _ = W2
    if out is None:
        out = torch.full((m, TOPK, H), float("nan"), dtype=torch.bfloat16)
    _flydsl_v2_stage2_wrapper(
        inter_states=a2,
        w1=None,
        w2=w2,
        sorted_token_ids=sorted_ids,
        sorted_expert_ids=sorted_e,
        num_valid_ids=nvalid,
        out=torch.empty((m, H), dtype=torch.bfloat16),
        topk=TOPK,
        kernelName=KN2,
        model_dim=H,
        inter_dim=I,
        num_experts=E,
        w2_scale=s2,
        a2_scale=a2_scale,
        sorted_weights=sorted_w,
        bias2=None,
        block_m=SBM,
        per_slot_out=out,
    )
    return out


def run_g2(inp, m, kernel="p6"):
    a2, a2_scale, sorted_ids, sorted_w, sorted_e, nvalid = inp
    _, _, b, bs = W2
    out = torch.full((m, TOPK, H), float("nan"), dtype=torch.bfloat16)
    G2[kernel](
        a2, a2_scale, b, bs, sorted_ids, sorted_w, sorted_e, nvalid, out, m, TOPK
    )
    return out


def ndiff(a, b):
    return int((a.view(torch.uint16) != b.view(torch.uint16)).sum())


def reduce(rows, m):
    out = torch.empty((m, H), dtype=torch.bfloat16)
    _run_moe_reduction(rows, out, m, TOPK, H)
    return out


@benchmark()
def test_moe_g2(m, seed, routing="random", scales="narrow"):
    inp = make_inputs(m, seed, routing, scales)
    ref = run_v2(inp, m)
    d = {}
    for kernel in G2:
        got = run_g2(inp, m, kernel)
        again = run_g2(inp, m, kernel)
        d[f"{kernel} diff values"] = ndiff(got, ref)
        d[f"{kernel} unwritten"] = int(torch.isnan(got).sum())
        d[f"{kernel} rerun diff"] = ndiff(again, got)
        d[f"{kernel} reduce diff"] = ndiff(reduce(got, m), reduce(ref, m))
    assert int(torch.isnan(ref).sum()) == 0, "the reference left route rows unwritten"
    assert all(v == 0 for v in d.values()), f"m={m} seed={seed} {routing} {scales}: {d}"
    return {**d, "sort blocks": int(inp[5][0]) // SBM}


@benchmark()
def perf_moe_g2(m):
    inp = make_inputs(m, 0, "random", "narrow")
    a2, a2_scale, sorted_ids, sorted_w, sorted_e, nvalid = inp
    _, _, b, bs = W2
    out = torch.empty((m, TOPK, H), dtype=torch.bfloat16)
    args = (a2, a2_scale, b, bs, sorted_ids, sorted_w, sorted_e, nvalid, out, m, TOPK)
    _, us = run_perftest(moe_g2_p6, *args, num_iters=20, num_warmup=3)
    _, us_w8 = run_perftest(moe_g2_w8, *args, num_iters=20, num_warmup=3)
    _, us_v2 = run_perftest(
        run_v2, inp, m, torch.empty_like(out), num_iters=20, num_warmup=3
    )
    tflop = 2.0 * m * TOPK * H * I / 1e12
    return {
        "p6 us (with tile list)": us,
        "w8 us (with tile list)": us_w8,
        "v2 reduce GEMM2 us": us_v2,
        "p6 speedup": us_v2 / us,
        "w8 speedup": us_v2 / us_w8,
        "w8 PF": tflop / us_w8 * 1e3,
    }


def expect_error(fn, *args, match):
    try:
        fn(*args)
        torch.cuda.synchronize()
    except RuntimeError as e:
        assert match in str(e), f"unexpected error: {e}"
        return
    raise AssertionError(f"{fn.__name__} accepted invalid arguments ({match})")


def test_arg_checks():
    m = 64
    inp = make_inputs(m, 0, "random", "narrow")
    a2, a2_scale, sorted_ids, sorted_w, sorted_e, nvalid = inp
    _, _, b, bs = W2
    out = torch.empty((m, TOPK, H), dtype=torch.bfloat16)
    big = torch.zeros(4097 * SBM, dtype=torch.int32)
    expect_error(
        moe_g2_p6,
        a2,
        a2_scale,
        b,
        bs,
        big,
        sorted_w,
        sorted_e,
        nvalid,
        out,
        m,
        TOPK,
        match="at most 4096 sort blocks",
    )
    out_h = torch.empty((m, TOPK, 4096), dtype=torch.bfloat16)
    expect_error(
        moe_g2_p6,
        a2,
        a2_scale,
        b,
        bs,
        sorted_ids,
        sorted_w,
        sorted_e,
        nvalid,
        out_h,
        m,
        TOPK,
        match="out must be",
    )
    a2_i = torch.empty((a2.shape[0], 2048), dtype=torch.uint8)
    expect_error(
        moe_g2_p6,
        a2_i,
        a2_scale,
        b,
        bs,
        sorted_ids,
        sorted_w,
        sorted_e,
        nvalid,
        out,
        m,
        TOPK,
        match="a2 must be",
    )
    big = torch.empty((MOE_G2_W8_MAX_ROUTES // TOPK + 1, TOPK, H), dtype=torch.bfloat16)
    expect_error(
        moe_g2_w8,
        a2,
        a2_scale,
        b,
        bs,
        sorted_ids,
        sorted_w,
        sorted_e,
        nvalid,
        big,
        big.shape[0],
        TOPK,
        match="32-bit store offsets",
    )
    aiter.logger.info("m3_tp1 moe_g2 argument checks passed")


@benchmark()
def test_w8_contention(m, reps):
    """w8 against p6 while another stream streams 1 GB copies through HBM."""
    inp = make_inputs(m, 3, "skew", "wide")
    ref = run_g2(inp, m, "p6")
    src = torch.empty(1 << 30, dtype=torch.uint8)
    dst = torch.empty_like(src)
    side = torch.cuda.Stream()
    diffs = 0
    for _ in range(reps):
        with torch.cuda.stream(side):
            for _ in range(4):
                dst.copy_(src)
        diffs += ndiff(run_g2(inp, m, "w8"), ref)
    torch.cuda.synchronize()
    assert diffs == 0, f"m={m}: w8 differs from p6 in {diffs} route values"
    return {"runs": reps, "diff values": diffs}


def w8_publish_counts():
    """VMEM ops issued between the A DMAs each publish wait of moe_g2_gemm_w8 covers
    (those of two K-steps earlier) and the wait, in the steady-state K loop: its body
    twice in a row, so the waits of the first two steps count across the back-edge.
    Also the VMEM ops a conditional branch inside the loop can skip (must be 0)."""
    llvm = "/opt/rocm/llvm/bin"
    so = os.path.join(get_user_jit_dir(), "module_m3_tp1_moe_g2.so")
    asm = ""
    with tempfile.TemporaryDirectory() as d:
        fat = os.path.join(d, "fatbin")
        subprocess.run(
            [
                f"{llvm}/llvm-objcopy",
                f"--dump-section=.hip_fatbin={fat}",
                so,
                fat + ".so",
            ],
            check=True,
        )
        with open(fat, "rb") as f:
            data = f.read()
        # one clang offload bundle per source file of the module
        magic, starts = b"__CLANG_OFFLOAD_BUNDLE__", []
        while (i := data.find(magic, starts[-1] + 1 if starts else 0)) >= 0:
            starts.append(i)
        for k, (a, b) in enumerate(zip(starts, starts[1:] + [len(data)])):
            part, co = os.path.join(d, f"bundle{k}"), os.path.join(d, f"co{k}")
            with open(part, "wb") as f:
                f.write(data[a:b])
            subprocess.run(
                [
                    f"{llvm}/clang-offload-bundler",
                    "--unbundle",
                    "--type=o",
                    f"--input={part}",
                    "--targets=hipv4-amdgcn-amd-amdhsa--gfx950",
                    f"--output={co}",
                ],
                check=True,
            )
            asm += subprocess.run(
                [f"{llvm}/llvm-objdump", "-d", "--no-show-raw-insn", co],
                check=True,
                capture_output=True,
                text=True,
            ).stdout
    insts, on = [], False
    for line in asm.splitlines():
        head = re.match(r"^[0-9a-f]+ <(.+)>:$", line)
        if head:
            on = "moe_g2_gemm_w8" in head.group(1)
            continue
        m = re.match(r"^\s+(\S+)\s*(.*?)\s*//\s*([0-9A-Fa-f]+):", line)
        if on and m:
            insts.append((int(m.group(3), 16), m.group(1), m.group(2)))

    def dest(x):
        off = int(x[2].split()[0])
        return x[0] + 4 + 4 * (off - 65536 if off >= 32768 else off)

    def vmem(x):
        return x[1].startswith(("buffer_", "global_", "scratch_", "flat_"))

    def dma(x):
        return x[1].startswith("buffer_load") and x[2].rstrip().endswith("lds")

    def publish(x):
        return x[1] == "s_waitcnt" and f"vmcnt({W8_PUBCNT})" in x[2]

    waits = [i for i, x in enumerate(insts) if publish(x)]
    assert (
        len(waits) == 4
    ), f"{len(waits)} publish waits, expected one per unrolled K-step"
    end = next(
        i
        for i, x in enumerate(insts)
        if i > waits[-1]
        and x[1].startswith(("s_branch", "s_cbranch"))
        and dest(x) <= insts[waits[0]][0]
    )
    start = next(i for i, x in enumerate(insts) if x[0] == dest(insts[end]))
    body = insts[start : end + 1]
    at = {x[0]: i for i, x in enumerate(body)}
    skipped = sum(
        sum(map(vmem, body[i + 1 : at[dest(x)]]))
        for i, x in enumerate(body)
        if x[1].startswith("s_cbranch") and x[0] < dest(x) and dest(x) in at
    )
    seq, counts = body + body, []
    for w in range(len(body), len(seq)):
        if publish(seq[w]):
            # 4 DMAs a K-step: before the wait come step kt's, kt-1's, then kt-2's
            last = [i for i in range(w) if dma(seq[i])][-9]
            counts.append(sum(map(vmem, seq[last + 1 : w])))
    return counts, skipped


def test_w8_publish_isa():
    counts, skipped = w8_publish_counts()
    assert skipped == 0, f"a branch in the K loop can skip {skipped} VMEM ops"
    assert min(counts) >= W8_PUBCNT, f"publish waits {counts} VMEM ops after their DMAs"
    aiter.logger.info(f"w8 publish waits: {counts} VMEM ops after their DMAs (>= 36)")


parser = argparse.ArgumentParser(
    formatter_class=argparse.RawTextHelpFormatter,
    description="Test the MiniMax-M3.1 TP1 ragged-tile MoE GEMM2.",
)
parser.add_argument(
    "--only",
    nargs="*",
    choices=["numerics", "args", "isa", "stress", "perf"],
    default=None,
)
parser.add_argument("--quick", action="store_true", help="fewer and smaller cases")
args = parser.parse_args()

if get_gfx_runtime() != "gfx950":
    aiter.logger.info("m3_tp1 moe_g2 targets gfx950; skipping")
    sys.exit(0)


def wanted(part):
    return args.only is None or part in args.only


W2 = make_w2(0)  # module-level: @benchmark would log a tensor argument
tables = {}
if wanted("args"):
    test_arg_checks()
if wanted("isa"):
    test_w8_publish_isa()
if wanted("numerics"):
    rows = []
    sizes = [1000, 8192] if args.quick else [1000, 4096, 5000, 8192, 16384, 32768]
    for m in sizes:
        for seed in [0] if args.quick else [0, 1, 2]:
            rows.append(test_moe_g2(m, seed))
    for seed in (0, 1):
        rows.append(test_moe_g2(8192, seed, routing="skew"))
    for m in [8192] if args.quick else [8192, 16384]:
        rows.append(test_moe_g2(m, 0, scales="wide"))
    for m in [1, 129] if args.quick else [1, 2, 17, 37, 129, 255, 513]:
        rows.append(test_moe_g2(m, 0))
    for m in (1, 300, 8192):
        rows.append(test_moe_g2(m, 0, routing="same4"))
    rows.append(test_moe_g2(4096, 0, routing="half"))
    rows.append(test_moe_g2(8192, 0, routing="one"))
    tables["moe_g2_p6 / moe_g2_w8 vs v2 reduce GEMM2 (per-route rows)"] = pd.DataFrame(
        rows
    )
if wanted("stress"):
    rows = [test_w8_contention(m, 4 if args.quick else 16) for m in (8192, 16384)]
    tables["moe_g2_w8 under HBM contention"] = pd.DataFrame(rows)
if wanted("perf"):
    rows = [perf_moe_g2(m) for m in (4096, 8192, 16384, 32768)]
    tables["moe_g2_p6 / moe_g2_w8 timing"] = pd.DataFrame(rows)
for title, df in tables.items():
    aiter.logger.info(f"{title}:\n{df.to_markdown(index=False)}")
