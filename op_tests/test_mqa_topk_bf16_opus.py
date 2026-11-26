# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Tests for aiter.ops.opus.mqa_topk_bf16 on packed causal documents seen from one context-parallel
rank: the output contract, the dense logits against an FP32 torch reference, the fused top-k bit for
bit against a stable sort of the dense logits and its recall against the FP32 reference, and
(--perf) the time of both ops. Every option takes a list; the run is their cartesian product.

    python3 op_tests/test_mqa_topk_bf16_opus.py                          # the default matrix
    # GLM-5.3 (Lumen#37 profile): one 69632-token document, CP 4 -> 17408 rows x 69632 keys
    python3 op_tests/test_mqa_topk_bf16_opus.py -d 69632 --cp 4 --cp-rank 0 -k 2048 --relu 1 \\
        -i randn --perf
    # Kimi-K3.1 (Lumen#35): two 16384-token documents, CP 4
    python3 op_tests/test_mqa_topk_bf16_opus.py -d 16384,16384 --cp 4 --cp-rank 0 -k 2048 \\
        --relu 1 -i randn --perf

A document pack is its lengths, comma-separated; each document is cut into 2 x cp chunks and rank
r holds chunks r and 2 cp - 1 - r (the zigzag split), its rows' windows [doc start, position + 1)
in the pack's key space. Inputs (-i):
  randn           q, k ~ N(0, 1) in bf16, weights ~ N(0, 1) / sqrt(32 x 128)
  dup_keys        every key the same row: every score of a row ties (the fallback path)
  zero_rows       every third q row zero: those rows score 0 everywhere (the fallback path)
  neg_weights     weights all negative
  random_windows  windows drawn at random over [-100, keys + 100): out of range, inverted, empty
                  (they ignore the CP split, so they run at cp 1 only)
"""

import argparse
import itertools

import pandas as pd
import pytest
import torch

import aiter
from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops.opus.mqa_topk_bf16 import (
    HEAD_DIM,
    HEADS,
    mqa_logits_bf16,
    mqa_topk_bf16,
    mqa_topk_bf16_config,
)
from aiter.test_common import benchmark, run_perftest

INPUTS = ("randn", "dup_keys", "zero_rows", "neg_weights", "random_windows")


def make_windows(doc_lens, cp_size=1, cp_rank=0):
    """Rows of one CP rank over packed causal documents, in global key-row space.

    Each document is cut into 2 * cp_size chunks; the rank holds chunks cp_rank and
    2 * cp_size - 1 - cp_rank. Returns int32 (row_starts, row_ends) and the key count.
    """
    starts, ends, base = [], [], 0
    for n in doc_lens:
        assert n % (2 * cp_size) == 0, "pad documents to a multiple of 2 * cp"
        c = n // (2 * cp_size)
        for j in (cp_rank, 2 * cp_size - 1 - cp_rank):
            for p in range(base + j * c, base + (j + 1) * c):
                starts.append(base)
                ends.append(p + 1)
        base += n
    i32 = torch.int32
    return torch.tensor(starts, dtype=i32), torch.tensor(ends, dtype=i32), base


def make_inputs(doc_lens, cp_size, cp_rank, device, inputs="randn", seed=0):
    row_starts, row_ends, keys = make_windows(doc_lens, cp_size, cp_rank)
    g = torch.Generator(device="cpu").manual_seed(seed)
    rows = row_starts.shape[0]
    q = torch.randn(rows, HEADS, HEAD_DIM, generator=g).bfloat16()
    k = torch.randn(keys, HEAD_DIM, generator=g).bfloat16()
    w = torch.randn(rows, HEADS, generator=g) * (HEADS * HEAD_DIM) ** -0.5
    if inputs == "dup_keys":
        k = k[:1].expand_as(k).contiguous()
    elif inputs == "zero_rows":
        q[::3] = 0
    elif inputs == "neg_weights":
        w = -w.abs()
    elif inputs == "random_windows":
        row_starts = torch.randint(-100, keys + 100, (rows,), generator=g).int()
        row_ends = torch.randint(-100, keys + 100, (rows,), generator=g).int()
    else:
        assert inputs == "randn", f"unknown inputs {inputs}"
    return [t.to(device) for t in (q, k, w, row_starts, row_ends)]


def clamp(row_starts, row_ends, keys):
    lo = row_starts.long().clamp(0, keys)[:, None]
    return lo, torch.maximum(row_ends.long().clamp(max=keys)[:, None], lo)


def reference_logits(q, k, weights, row_starts, row_ends, relu=True):
    """FP32 scores [rows, keys], -inf outside each window."""
    keys = k.shape[0]
    kf = k.float().t()
    s = torch.zeros(q.shape[0], keys, device=q.device)
    for h in range(q.shape[1]):
        d = q[:, h].float() @ kf
        s += weights[:, h, None] * (d.relu() if relu else d)
    cols = torch.arange(keys, device=q.device)
    lo, hi = clamp(row_starts, row_ends, keys)
    return s.masked_fill((cols < lo) | (cols >= hi), float("-inf"))


def topk_of(scores, row_starts, row_ends, topk):
    """The top-W keys of dense scores by (score desc, index asc), stored ascending, -1 padded."""
    keys = scores.shape[1]
    width = min(topk, keys)
    lo, hi = clamp(row_starts, row_ends, keys)
    s = scores.clone()
    s[s == 0] = 0.0  # -0 orders (and reads back) as +0
    val, idx = s.sort(dim=1, descending=True, stable=True)
    keep = torch.arange(width, device=s.device) < (hi - lo).clamp(max=width)
    idx = torch.where(keep, idx[:, :width], keys)
    idx, order = idx.sort(dim=1)
    val = val[:, :width].gather(1, order)
    keep = idx < keys
    return torch.where(keep, idx, -1).int(), torch.where(keep, val, float("-inf"))


def reference(q, k, weights, row_starts, row_ends, topk, relu=True, chunk=256):
    """The FP32 reference top-k, a chunk of rows at a time."""
    parts = [
        topk_of(
            reference_logits(
                q[r : r + chunk],
                k,
                weights[r : r + chunk],
                row_starts[r : r + chunk],
                row_ends[r : r + chunk],
                relu,
            ),
            row_starts[r : r + chunk],
            row_ends[r : r + chunk],
            topk,
        )
        for r in range(0, q.shape[0], chunk)
    ]
    return torch.cat([p[0] for p in parts]), torch.cat([p[1] for p in parts])


def check_contract(indices, values, row_starts, row_ends, keys, topk):
    """Valid prefix of min(n, W) ascending rows inside [lo, hi), then -1; -inf values there."""
    rows, width = row_starts.shape[0], min(topk, keys)
    assert indices.dtype == torch.int32 and tuple(indices.shape) == (rows, width)
    lo, hi = clamp(row_starts, row_ends, keys)
    valid = torch.arange(width, device=indices.device) < (hi - lo).clamp(max=width)
    idx = indices.long()
    assert torch.equal(idx >= 0, valid), "valid entries first"
    assert (idx[~valid] == -1).all(), "padding is -1"
    assert ((idx >= lo) & (idx < hi))[valid].all(), "an index lies outside its window"
    ascending = idx[:, 1:] > idx[:, :-1]
    assert ascending[valid[:, 1:]].all(), "valid entries ascend, no duplicates"
    if values is not None:
        assert torch.isneginf(values[~valid]).all(), "-1 slots hold -inf"
        assert torch.isfinite(values[valid]).all(), "valid slots hold finite scores"


@pytest.mark.parametrize("cp_size,cp_rank", [(1, 0), (2, 1), (4, 0), (4, 3)])
def test_reference_contract(cp_size, cp_rank):
    q, k, w, rs, re = make_inputs([64, 256, 32, 512], cp_size, cp_rank, "cpu")
    idx, val = reference(q, k, w, rs, re, topk=48)
    check_contract(idx, val, rs, re, k.shape[0], 48)


def test_reference_ties():
    # Equal keys give equal scores: the smallest indices of each window win.
    q, k, w, rs, re = make_inputs([256], 1, 0, "cpu")
    k = k[:1].expand_as(k).contiguous()
    idx, _ = reference(q, k, w, rs, re, topk=16)
    want = rs[:, None] + torch.arange(16, dtype=torch.int32)
    want = torch.where(want < re[:, None], want, -1)
    assert torch.equal(idx, want)


def row_hits(got, want, keys):
    """Per row, how many of got's keys are among want's: both ascending, -1 padded."""
    big = torch.full_like(want, keys)
    sorted_want = torch.where(
        want >= 0, want, big
    )  # -1 padding to the end, past every key
    pos = torch.searchsorted(sorted_want, got.contiguous()).clamp(max=want.shape[1] - 1)
    return ((sorted_want.gather(1, pos) == got) & (got >= 0)).sum(1)


def check(q, k, w, rs, re, cfg, idx, val, topk, relu):
    """The op's dense logits against FP32, its top-k against a stable sort of them (bit for bit)
    and against the FP32 top-k (recall); a chunk of rows at a time, about 1 GiB of scores each.
    """
    rows, keys = q.shape[0], k.shape[0]
    chunk = max(1, (1 << 28) // keys)
    err, hits, total = 0.0, 0, 0
    for r0 in range(0, rows, chunk):
        sl = slice(r0, min(r0 + chunk, rows))
        part = (q[sl], k, w[sl], rs[sl], re[sl])
        logits = mqa_logits_bf16(*part, cfg=cfg)
        want = reference_logits(*part, relu)
        fin = torch.isfinite(want)
        assert torch.equal(
            fin, torch.isfinite(logits)
        ), "-inf exactly outside the windows"
        if fin.any():
            scale = max(want[fin].abs().max().item(), 1.0)
            err = max(err, (logits[fin] - want[fin]).abs().max().item() / scale)
        ref_idx, ref_val = topk_of(logits, rs[sl], re[sl], topk)
        assert torch.equal(
            idx[sl], ref_idx
        ), "indices: a stable sort of the dense logits"
        assert torch.equal(val[sl], ref_val), "values: the dense logits"
        fp32_idx, _ = topk_of(want, rs[sl], re[sl], topk)
        hits += int(row_hits(idx[sl], fp32_idx, keys).sum())
        total += int((fp32_idx >= 0).sum())
    assert err <= 1e-5, f"logits max |err| {err:.3e} of max |s| against FP32"
    recall = hits / max(total, 1)
    assert recall >= 0.9995, f"recall {recall:.6f} against the FP32 reference"
    return err, recall


@benchmark()
def run_case(
    docs, cp, cp_rank, topk, relu, inputs, low_memory=False, do_check=True, perf=False
):
    q, k, w, rs, re = make_inputs(docs, cp, cp_rank, "cuda", inputs)
    rows, keys = q.shape[0], k.shape[0]
    cfg = mqa_topk_bf16_config(
        q.device,
        topk=topk,
        max_rows=rows,
        max_keys=keys,
        relu=bool(relu),
        low_memory=bool(low_memory),
    )
    ws = cfg.workspace()
    lo, hi = clamp(rs, re, keys)
    pairs = int((hi - lo).sum())
    ret = {"rows": rows, "keys": keys, "pairs (M)": round(pairs / 1e6, 1)}

    idx, val = mqa_topk_bf16(q, k, w, rs, re, cfg=cfg, workspace=ws, return_values=True)
    check_contract(idx, val, rs, re, keys, topk)
    idx2, val2 = mqa_topk_bf16(
        q, k, w, rs, re, cfg=cfg, workspace=ws, return_values=True
    )
    assert torch.equal(idx, idx2) and torch.equal(val, val2), "bitwise deterministic"
    if do_check:
        err, recall = check(q, k, w, rs, re, cfg, idx, val, topk, bool(relu))
        ret.update({"logits rel err": f"{err:.1e}", "recall": f"{recall:.6f}"})
    if perf:
        # Rotated copies keep the inputs cold (about 1 GiB of them, past the 256 MB MALL), but
        # run_perftest copies the outputs and the workspace too: a quarter of free memory at most.
        def rotations(*outs):
            inputs = sum(t.nbytes for t in (q, k, w, rs, re))
            each = inputs + sum(t.nbytes for t in outs)
            fit = int(torch.cuda.mem_get_info()[0] * 0.25 // each)
            return max(1, min(-(-(1 << 30) // inputs), fit, 32))

        _, us = run_perftest(
            mqa_topk_bf16,
            q,
            k,
            w,
            rs,
            re,
            cfg=cfg,
            workspace=ws,
            indices=idx,
            num_rotate_args=rotations(ws, idx),
        )
        del idx2, val2
        out = torch.empty(rows, keys, device=q.device)
        _, us_dense = run_perftest(
            mqa_logits_bf16,
            q,
            k,
            w,
            rs,
            re,
            cfg=cfg,
            out=out,
            num_rotate_args=rotations(out),
        )
        flops = pairs * HEADS * HEAD_DIM * 2
        ret.update(
            {
                "topk us": round(us, 1),
                "topk TFLOPS": round(flops / us / 1e6),
                "logits us": round(us_dense, 1),
                "workspace MiB": round(cfg.workspace_bytes / 2**20),
            }
        )
    return ret


def _gfx950():
    return torch.cuda.is_available() and get_gfx_runtime() == "gfx950"


@pytest.mark.parametrize(
    "docs,cp,cp_rank,topk,relu,inputs",
    [
        ((256, 512, 64, 1024, 128), 4, 0, 64, 1, "randn"),
        ((1000, 776), 1, 0, 64, 1, "randn"),
        ((40, 8), 1, 0, 64, 1, "randn"),
        ((16384, 8192), 4, 3, 2048, 0, "randn"),
        ((16384, 8192), 2, 0, 4096, 1, "randn"),
        ((16384,), 4, 0, 2048, 1, "dup_keys"),
        ((16384,), 2, 1, 2048, 1, "zero_rows"),
        ((12288,), 1, 0, 2048, 1, "random_windows"),
    ],
)
def test_mqa_topk_bf16(docs, cp, cp_rank, topk, relu, inputs):
    if not _gfx950():
        pytest.skip("requires gfx950")
    run_case(docs, cp, cp_rank, topk, relu, inputs)


def check_graph():
    """Graph replay equals an eager call, also with new inputs."""
    q, k, w, rs, re = make_inputs((16384, 8192), 4, 1, "cuda")
    rows, keys = q.shape[0], k.shape[0]
    cfg = mqa_topk_bf16_config(q.device, max_rows=rows, max_keys=keys)
    ws = cfg.workspace()
    idx = torch.empty(rows, cfg.width, dtype=torch.int32, device=q.device)
    val = torch.empty(rows, cfg.width, dtype=torch.float32, device=q.device)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):  # warm up outside the capture
        mqa_topk_bf16(q, k, w, rs, re, cfg=cfg, workspace=ws, indices=idx, values=val)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        mqa_topk_bf16(q, k, w, rs, re, cfg=cfg, workspace=ws, indices=idx, values=val)
    for _ in range(2):
        q.copy_(torch.randn(q.shape, device=q.device).bfloat16())
        idx.fill_(7)
        graph.replay()
        want_idx, want_val = mqa_topk_bf16(q, k, w, rs, re, cfg=cfg, return_values=True)
        assert torch.equal(idx, want_idx) and torch.equal(val, want_val), "graph replay"


def test_mqa_topk_bf16_graph():
    if not _gfx950():
        pytest.skip("requires gfx950")
    check_graph()


def test_registry_beside_mxfp4():
    # Stub names shared with the MXFP4 module would mix the two registries.
    if not _gfx950():
        pytest.skip("requires gfx950")
    from aiter.ops.opus import pa_mqa_logits_mxfp4

    dev = torch.device("cuda", torch.cuda.current_device())
    mqa_topk_bf16_config(dev, max_rows=1, max_keys=1)
    assert pa_mqa_logits_mxfp4._instances(dev), "MXFP4 registry"


def doc_pack(v):
    """'16384,16384' -> (16384, 16384): the lengths of one pack of documents."""
    return tuple(int(x) for x in v.strip("()").split(",") if x.strip())


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter, description=__doc__
    )
    parser.add_argument(
        "-d",
        "--docs",
        type=doc_pack,
        nargs="*",
        default=[(256, 512, 64, 1024, 128), (1000, 776), (40, 8), (8192, 4096)],
        help="document packs, each its lengths comma-separated (e.g. 69632 16384,16384)",
    )
    parser.add_argument(
        "--cp", type=int, nargs="*", default=[1, 4], help="context-parallel sizes"
    )
    parser.add_argument(
        "--cp-rank",
        type=int,
        nargs="*",
        default=[0, -1],
        help="the rank whose rows are scored; negative counts from the last",
    )
    parser.add_argument("-k", "--topk", type=int, nargs="*", default=[64, 2048, 4096])
    parser.add_argument("--relu", type=int, nargs="*", default=[1, 0], choices=[0, 1])
    parser.add_argument(
        "-i", "--inputs", nargs="*", default=list(INPUTS), choices=INPUTS
    )
    parser.add_argument(
        "--low-memory", type=int, nargs="*", default=[0], choices=[0, 1]
    )
    parser.add_argument("--perf", action="store_true", help="also time both ops")
    parser.add_argument(
        "--no-check",
        action="store_true",
        help="skip the dense and FP32 comparisons (the contract is still checked)",
    )
    args = parser.parse_args()

    test_reference_ties()
    for cp_size, cp_rank in ((1, 0), (4, 0), (4, 3)):
        test_reference_contract(cp_size, cp_rank)
    aiter.logger.info("mqa_topk_bf16 reference contract: ok")
    if not _gfx950():
        aiter.logger.warning("mqa_topk_bf16 requires gfx950; skipping the op")
        return
    check_graph()
    aiter.logger.info("mqa_topk_bf16 HIP graph capture: ok")

    rows, skipped, seen = [], [], set()
    for docs, cp, rank, topk, relu, inputs, low_mem in itertools.product(
        args.docs,
        args.cp,
        args.cp_rank,
        args.topk,
        args.relu,
        args.inputs,
        args.low_memory,
    ):
        rank = rank % cp if -cp <= rank < 0 else rank
        if rank >= cp or any(n % (2 * cp) for n in docs):
            continue  # no such rank, or a document the zigzag split cannot cut
        if inputs == "random_windows" and cp > 1:
            continue  # its windows ignore the split: cp 1 covers it
        case = (docs, cp, rank, topk, relu, inputs, low_mem)
        if case in seen:  # --cp-rank 0 -1 at cp 1
            continue
        seen.add(case)
        try:
            rows.append(
                run_case(
                    docs,
                    cp,
                    rank,
                    topk,
                    relu,
                    inputs,
                    low_mem,
                    not args.no_check,
                    args.perf,
                )
            )
        except torch.OutOfMemoryError:
            skipped.append(case)
            torch.cuda.empty_cache()
    df = pd.DataFrame(rows).drop(columns=["do_check", "perf"], errors="ignore")
    aiter.logger.info(
        "mqa_topk_bf16 summary (markdown):\n%s", df.to_markdown(index=False)
    )
    if skipped:
        aiter.logger.warning(
            "mqa_topk_bf16: %d cases skipped, out of memory: %s", len(skipped), skipped
        )


if __name__ == "__main__":
    main()
