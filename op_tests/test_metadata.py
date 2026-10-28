# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import argparse
import itertools
import os
import random

import pandas as pd
import torch

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.test_common import benchmark, checkAllclose, run_perftest

torch.set_default_device("cuda")
torch.set_printoptions(sci_mode=False)

# ---------------------------------------------------------------------------
# Kimi-K2.5-MXFP4 decode MLA configuration (per TP rank).
# ---------------------------------------------------------------------------
KIMI_TOTAL_QO_HEADS = 64  # Kimi-K2.5 num_attention_heads
MLA_MIN_HEADS = 16  # atom _MLA_MIN_HEADS: AITER MLA kernels need >= 16 q-heads
KIMI_NHEAD_KV = 1  # MLA: a single latent KV head

# Decode metadata knobs, identical to atom's persistent worker buffers.
PAGE_SIZE = 1
KV_GRANULARITY = max(PAGE_SIZE, 16)
MAX_SEQLEN_QO = 1  # pure decode, no MTP/spec tokens
UNI_SEQLEN_QO = 1
MAX_SPLIT_PER_BATCH = 16
IS_CAUSAL = True

# Default serving sweep (kimi Makefile / perf_sweep.sh), plus the batch counts
# that only prefill reaches. num_batches is the concurrency for decode, so the
# kimi row tops out at 128 -- but a DSA prefill chunk carries one batch per query
# token, which puts --max-num-batched-tokens on this axis instead. Those large
# counts are where the parallel planner chunks its LDS scratch, and 4096 / 8192 /
# 16384 sit at one chunk, the old 8191-batch LDS cliff, and several chunks. Below
# 4096 none of that code runs, so without these rows the default sweep cannot
# tell a working chunk carry from a broken one.
DEFAULT_BATCHES = [4, 8, 16, 32, 64, 128, 4096, 8192, 16384]
DEFAULT_CTX_LENS = [2048, 4096, 8192]

_PARALLEL_ENV = "AITER_MLA_META_USE_PARALLEL"
# Lowering the planner's chunk ceiling forces the runtime-chunk kernel. A card
# whose LDS fits the full chunk always gets the folded instantiation, so this is
# the only way a gfx950 run reaches the path gfx942 takes in production.
_CHUNK_ENV = "AITER_MLA_META_BATCH_CHUNK"
# Mirrors MLA_V12_PARALLEL_BATCH_CHUNK; only ever a label in the table.
MLA_V12_DEFAULT_CHUNK = 4096

# Every card the MLA metadata planner is built and validated for.
SUPPORTED_GFX = ["gfx942", "gfx950"]

# Non-uniform batches (uni_seqlen_qo=-1). gfx942 has no bf16 persistent kernel
# for 96 heads, so the planner folds each request into 96 / 16 pseudo-batches
# while qo_indptr keeps one entry per request. A trailing 0 is a CUDA-graph
# padding row; 7 decode tokens plus one is the Kimi-K3 TP8/DCP8 tail shape.
DEFAULT_QO_LENS = [(1, 1, 1, 1, 1, 1, 1, 0), (2, 1, 3), (2, 2, 2, 0), (4, 4, 4, 0)]
DEFAULT_QO_NHEADS = [16, 96]


def kimi_nhead(tp: int) -> int:
    """q-heads per rank for Kimi-K2.5 at the given TP, padded to MLA_MIN_HEADS."""
    assert KIMI_TOTAL_QO_HEADS % tp == 0, f"TP{tp} does not divide 64 heads evenly"
    return max(KIMI_TOTAL_QO_HEADS // tp, MLA_MIN_HEADS)


def build_decode_inputs(batch_size, ctx_len, dtype, kvtype, nhead, *, jitter, seed):
    """
    Build the decode-time inputs for one (batch_size, ctx_len) shape, mirroring
    what atom feeds get_mla_metadata_v1:

      * cu_seqlens_q : arange(batch+1)  -- 1 query token per sequence.
      * kv_indptr    : cumulative KV page counts (page_size == 1 => 1 page/token),
                       so each sequence holds ``seqlen_kv`` pages.
      * kv_last_page_lens : ones (page_size == 1).

    With ``jitter`` the per-sequence KV length is drawn from
    [ctx_len // 2, ctx_len] to emulate the spread of a real in-flight batch;
    otherwise every sequence is exactly ``ctx_len`` long.
    """
    if jitter:
        rng = random.Random(seed)
        kv_lens = [
            rng.randint(max(1, ctx_len // 2), ctx_len) for _ in range(batch_size)
        ]
    else:
        kv_lens = [ctx_len] * batch_size

    qo_indptr = torch.arange(batch_size + 1, dtype=torch.int32, device="cuda")

    kv_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device="cuda")
    kv_indptr[1:] = torch.tensor(kv_lens, dtype=torch.int32, device="cuda").cumsum(0)

    kv_last_page_lens = torch.ones(batch_size, dtype=torch.int32, device="cuda")

    # Output buffers, sized exactly as atom does via get_mla_metadata_info_v1.
    (
        (work_meta_data_size, work_meta_data_type),
        (work_indptr_size, work_indptr_type),
        (work_info_set_size, work_info_set_type),
        (reduce_indptr_size, reduce_indptr_type),
        (reduce_final_map_size, reduce_final_map_type),
        (reduce_partial_map_size, reduce_partial_map_type),
    ) = aiter.get_mla_metadata_info_v1(
        batch_size,
        MAX_SEQLEN_QO,
        nhead,
        dtype,
        kvtype,
        is_sparse=False,
        fast_mode=True,
    )

    inputs = {
        "qo_indptr": qo_indptr,
        "kv_indptr": kv_indptr,
        "kv_last_page_lens": kv_last_page_lens,
        "nhead": nhead,
        "max_seqlen_qo": MAX_SEQLEN_QO,
        "uni_seqlen_qo": UNI_SEQLEN_QO,
    }
    out_meta = {
        "work_meta_data": (work_meta_data_size, work_meta_data_type),
        "work_indptr": (work_indptr_size, work_indptr_type),
        "work_info_set": (work_info_set_size, work_info_set_type),
        "reduce_indptr": (reduce_indptr_size, reduce_indptr_type),
        "reduce_final_map": (reduce_final_map_size, reduce_final_map_type),
        "reduce_partial_map": (reduce_partial_map_size, reduce_partial_map_type),
    }
    return inputs, out_meta, kv_lens


def alloc_outputs(out_meta):
    return {
        name: torch.empty(size, dtype=t, device="cuda")
        for name, (size, t) in out_meta.items()
    }


def call_metadata(inputs, outs, dtype, kvtype):
    """Run get_mla_metadata_v1 into the provided output buffers (in place)."""
    aiter.get_mla_metadata_v1(
        inputs["qo_indptr"],
        inputs["kv_indptr"],
        inputs["kv_last_page_lens"],
        inputs["nhead"] // KIMI_NHEAD_KV,
        KIMI_NHEAD_KV,
        IS_CAUSAL,
        outs["work_meta_data"],
        outs["work_info_set"],
        outs["work_indptr"],
        outs["reduce_indptr"],
        outs["reduce_final_map"],
        outs["reduce_partial_map"],
        page_size=PAGE_SIZE,
        kv_granularity=KV_GRANULARITY,
        max_seqlen_qo=inputs["max_seqlen_qo"],
        uni_seqlen_qo=inputs["uni_seqlen_qo"],
        fast_mode=True,
        max_split_per_batch=MAX_SPLIT_PER_BATCH,
        dtype_q_nope=dtype,
        dtype_kv_nope=kvtype,
    )


def run_path(inputs, out_meta, dtype, kvtype, use_parallel):
    """Allocate fresh buffers, force a planner via env, and run it once."""
    prev = os.environ.get(_PARALLEL_ENV)
    os.environ[_PARALLEL_ENV] = "1" if use_parallel else "0"
    try:
        outs = alloc_outputs(out_meta)
        call_metadata(inputs, outs, dtype, kvtype)
        torch.cuda.synchronize()
    finally:
        if prev is None:
            os.environ.pop(_PARALLEL_ENV, None)
        else:
            os.environ[_PARALLEL_ENV] = prev
    return outs


def compare_metadata(golden, test):
    """
    Compare the meaningful (written) regions of two metadata buffer sets.

    The buffers are over-allocated (worst case), so only the populated prefixes
    are deterministic; tails come from torch.empty. We derive the valid prefix
    lengths from the planner's own indptr outputs:

      * work_indptr is fully written (size #cu+1) and ends with the total work
        count; work_info_set[:num_works] is the populated work region.
      * reduce_indptr is fully written; each split qo-tile contributes a strictly
        increasing step, so the number of reduce groups is the count of positive
        steps and the total partial-tile count is reduce_indptr's final value.

    Returns (ok, details) where details maps a field name to its mismatch count.
    """
    details = {}

    def exact(name, g, t, mask=None):
        # Planner output is indices, not arithmetic: any difference at all is a
        # divergence, so compare to the bit (rtol=atol=tol_err_ratio=0).
        details[name] = checkAllclose(
            g, t, rtol=0, atol=0, tol_err_ratio=0, mask=mask, printLog=False
        )

    wi_g = golden["work_indptr"]
    wi_t = test["work_indptr"]
    exact("work_indptr", wi_g, wi_t)

    num_works = int(wi_g[-1].item())
    exact(
        "work_info_set",
        golden["work_info_set"][:num_works],
        test["work_info_set"][:num_works],
    )

    ri_g = golden["reduce_indptr"]
    ri_t = test["reduce_indptr"]
    exact("reduce_indptr", ri_g, ri_t)

    # Valid regions for the reduce maps, derived from the golden reduce_indptr.
    steps = ri_g[1:] - ri_g[:-1]
    num_groups = int((steps > 0).sum().item())
    num_partial = int(ri_g[-1].item())

    # reduce_final_map is NOT a packed prefix. Both planners write row i only
    # when tile i is split, at that tile's own index, so the written rows are
    # scattered over the whole buffer. Taking the first num_groups rows compared
    # memory neither planner wrote -- at batch=4096 with --jitter, 207 of those
    # 218 rows, which is 414 of the elements. Since alloc_outputs() hands each
    # run a fresh torch.empty buffer, that read whatever the two allocations
    # happened to hold and reported a mismatch that no kernel produced. A split
    # tile is exactly one whose reduce_indptr step is positive, so compare those
    # rows and leave the untouched ones alone.
    rfm_g = golden["reduce_final_map"].reshape(-1, 2)
    rfm_t = test["reduce_final_map"].reshape(-1, 2)
    written = (steps > 0)[: rfm_g.shape[0]].unsqueeze(1)
    exact("reduce_final_map", rfm_g, rfm_t, mask=written)

    exact(
        "reduce_partial_map",
        golden["reduce_partial_map"][:num_partial],
        test["reduce_partial_map"][:num_partial],
    )

    ok = all(v == 0 for v in details.values())
    return ok, details, num_works, num_groups, num_partial


def planner_nbytes(golden, batch_size, num_works, num_groups, num_partial):
    # The planner does no floating-point at all, so TFLOPS would be a column of
    # zeros. Bytes are the metric that means something here: it reads three
    # per-batch indptrs and writes the work/reduce descriptors, and the GB/s
    # column against a card that does TB/s is the whole diagnosis -- this kernel
    # is bound by a single-lane scan, not by traffic.
    wis = golden["work_info_set"]
    work_bytes = wis.element_size() * (wis[0].numel() if wis.dim() > 1 else 1)
    return (
        3 * (batch_size + 1) * 4  # qo_indptr, kv_indptr, kv_last_page_lens
        + num_works * work_bytes  # work_info_set
        + (batch_size + 1) * 4  # reduce_indptr
        + num_groups * 2 * 4  # reduce_final_map, split tiles only
        + num_partial * 4  # reduce_partial_map, one entry per partial fragment
    )


def time_planners(inputs, out_meta, dtype, kvtype, num_iters, nbytes, ret):
    """Time both planners into ``ret`` and return their us by name."""
    # The serial planner is both the reference and a kernel under test: it is
    # what runs today whenever the parallel path bails out, so it is timed.
    candidates = {"serial": "0", "parallel": "1"}
    us = {}
    for name, env in candidates.items():
        os.environ[_PARALLEL_ENV] = env
        _, us[name] = run_perftest(
            call_metadata,
            inputs,
            alloc_outputs(out_meta),
            dtype,
            kvtype,
            num_iters=num_iters,
        )
        ret[f"{name} us"] = round(us[name], 3)
        ret[f"{name} GB/s"] = round(nbytes / us[name] / 1e3, 2)
    os.environ.pop(_PARALLEL_ENV, None)
    return us


@benchmark()
def test_metadata(
    batch_size, ctx_len, dtype, kvtype, nhead, jitter, seed, num_iters, batch_chunk=0
):
    if batch_chunk:
        os.environ[_CHUNK_ENV] = str(batch_chunk)
    else:
        os.environ.pop(_CHUNK_ENV, None)

    inputs, out_meta, _kv_lens = build_decode_inputs(
        batch_size, ctx_len, dtype, kvtype, nhead, jitter=jitter, seed=seed
    )

    # Golden (serial planner) vs parallel planner -- must be bit identical.
    golden = run_path(inputs, out_meta, dtype, kvtype, use_parallel=False)
    parallel = run_path(inputs, out_meta, dtype, kvtype, use_parallel=True)
    ok, mism, num_works, num_groups, num_partial = compare_metadata(golden, parallel)

    if not ok:
        print(f"  [MISMATCH] bs={batch_size} ctx={ctx_len} nhead={nhead}: {mism}")

    nbytes = planner_nbytes(golden, batch_size, num_works, num_groups, num_partial)

    ret = {
        "gfx": get_gfx(),
        "chunk": batch_chunk or MLA_V12_DEFAULT_CHUNK,
        "num_works": num_works,
        "num_split_groups": num_groups,
    }
    us = time_planners(inputs, out_meta, dtype, kvtype, num_iters, nbytes, ret)

    # Only the parallel path can diverge; serial is the reference it is checked
    # against, so its err is 0 by construction and would be a noise column.
    ret["parallel err"] = max(mism.values()) if mism else 0
    ret["match"] = ok
    ret["speedup"] = (
        round(us["serial"] / us["parallel"], 3) if us["parallel"] else float("nan")
    )
    return ret


def bf16_fold_ratio(nhead):
    """Host mirror of the bf16 head fold in get_mla_metadata_v1_2_device."""
    if get_gfx() == "gfx942" and nhead != 16:
        return nhead // 16
    return 1


def build_nonuniform_inputs(qo_lens, nhead, dtype, kvtype):
    kv_lens = [0 if qo_len == 0 else 32 + 16 * i for i, qo_len in enumerate(qo_lens)]
    qo_indptr = torch.zeros(len(qo_lens) + 1, dtype=torch.int32, device="cuda")
    qo_indptr[1:] = torch.tensor(qo_lens, dtype=torch.int32, device="cuda").cumsum(0)
    kv_indptr = torch.zeros(len(qo_lens) + 1, dtype=torch.int32, device="cuda")
    kv_indptr[1:] = torch.tensor(kv_lens, dtype=torch.int32, device="cuda").cumsum(0)

    names = (
        "work_meta_data",
        "work_indptr",
        "work_info_set",
        "reduce_indptr",
        "reduce_final_map",
        "reduce_partial_map",
    )
    info = aiter.get_mla_metadata_info_v1(
        len(qo_lens),
        max(qo_lens),
        nhead,
        dtype,
        kvtype,
        is_sparse=False,
        fast_mode=True,
    )
    inputs = {
        "qo_indptr": qo_indptr,
        "kv_indptr": kv_indptr,
        "kv_last_page_lens": torch.ones(len(qo_lens), dtype=torch.int32, device="cuda"),
        "nhead": nhead,
        "max_seqlen_qo": max(qo_lens),
        "uni_seqlen_qo": -1,
    }
    return inputs, dict(zip(names, info))


def folded_qo_range_errors(outs, qo_lens, ratio):
    """
    Check every work's [qo_start, qo_end) against the folded Q layout, where
    pseudo-batch ``b * ratio + g`` owns rows ``(indptr[b] * ratio + g * len_b)``
    onward. Each non-empty pseudo-batch must be covered exactly, and no range
    may run past the folded Q tensor.
    """
    begins = [0]
    for qo_len in qo_lens:
        begins.append(begins[-1] + qo_len)
    spans = [
        (begins[b] * ratio + g * qo_len, begins[b] * ratio + (g + 1) * qo_len)
        for b, qo_len in enumerate(qo_lens)
        for g in range(ratio)
    ]
    total_rows = begins[-1] * ratio
    pieces = {span: [] for span in spans if span[1] > span[0]}

    num_works = int(outs["work_indptr"][-1].item())
    work_info = outs["work_info_set"].view(-1, 8)
    if not 0 < num_works <= work_info.shape[0]:
        return [f"{num_works} works for a {work_info.shape[0]}-entry work_info_set"]

    errors = []
    for row in work_info[:num_works].tolist():
        qo_start, qo_end = row[2], row[3]
        if not 0 <= qo_start <= qo_end <= total_rows:
            errors.append(f"[{qo_start}, {qo_end}) is outside {total_rows} Q rows")
            continue
        if qo_start == qo_end:
            continue
        owners = [s for s in pieces if s[0] <= qo_start and qo_end <= s[1]]
        if len(owners) != 1:
            errors.append(f"[{qo_start}, {qo_end}) spans pseudo-batches {owners}")
            continue
        pieces[owners[0]].append((qo_start, qo_end))

    for span, covered in pieces.items():
        merged = []
        for start, end in sorted(covered):
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))
        if merged != [span]:
            errors.append(f"pseudo-batch {span} covered by {merged}")
    return errors


def nonempty_works(outs):
    num_works = int(outs["work_indptr"][-1].item())
    works = outs["work_info_set"].view(-1, 8)[:num_works]
    return works[works[:, 3] > works[:, 2]]


@benchmark()
def test_nonuniform_metadata(qo_lens, nhead, dtype, kvtype, num_iters):
    inputs, out_meta = build_nonuniform_inputs(qo_lens, nhead, dtype, kvtype)

    golden = run_path(inputs, out_meta, dtype, kvtype, use_parallel=False)
    parallel = run_path(inputs, out_meta, dtype, kvtype, use_parallel=True)
    _, mism, num_works, num_groups, num_partial = compare_metadata(golden, parallel)
    # The serial planner gives an empty request one zero-length work and the
    # parallel planner skips it, so only the works that carry queries must agree.
    del mism["work_indptr"]
    works_g, works_p = nonempty_works(golden), nonempty_works(parallel)
    mism["work_info_set"] = (
        checkAllclose(works_g, works_p, rtol=0, atol=0, tol_err_ratio=0, printLog=False)
        if works_g.shape == works_p.shape
        else 1.0
    )
    ok = all(v == 0 for v in mism.values())
    ratio = bf16_fold_ratio(nhead)
    errors = folded_qo_range_errors(golden, qo_lens, ratio)

    if not ok:
        print(f"  [MISMATCH] qo_lens={qo_lens} nhead={nhead}: {mism}")
    for error in errors:
        print(f"  [QO RANGE] qo_lens={qo_lens} nhead={nhead}: {error}")

    nbytes = planner_nbytes(golden, len(qo_lens), num_works, num_groups, num_partial)
    ret = {
        "gfx": get_gfx(),
        "fold": ratio,
        "num_works": num_works,
        "num_split_groups": num_groups,
    }
    time_planners(inputs, out_meta, dtype, kvtype, num_iters, nbytes, ret)
    ret["parallel err"] = max(mism.values()) if mism else 0
    ret["qo range err"] = len(errors)
    ret["match"] = ok and not errors
    return ret


def main():
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning(
            "mla metadata planner unsupported on %s; skipping", get_gfx()
        )
        return

    parser = argparse.ArgumentParser(
        description=(
            "MLA metadata planner microbench/correctness test, with shapes "
            "aligned to ATOM serving Kimi-K2.5-MXFP4 decode."
        )
    )
    parser.add_argument(
        "-tp",
        "--tensor-parallel",
        type=int,
        default=8,
        help="TP degree (Kimi-K2.5-MXFP4 recipe uses TP8). Sets q-heads per rank.",
    )
    parser.add_argument(
        "--nhead",
        type=int,
        default=None,
        help="Override q-heads per rank (default: derived from --tp).",
    )
    parser.add_argument(
        "-b",
        "--batch",
        type=int,
        nargs="+",
        default=DEFAULT_BATCHES,
        help="Batch sizes (== serving concurrency).",
    )
    parser.add_argument(
        "-c",
        "--ctx-len",
        type=int,
        nargs="+",
        default=DEFAULT_CTX_LENS,
        help="KV context lengths (== serving ISL).",
    )
    parser.add_argument(
        "--jitter",
        action="store_true",
        help="Randomize per-sequence KV length in [ctx/2, ctx] (decode spread).",
    )
    parser.add_argument(
        "--batch-chunk",
        type=int,
        nargs="+",
        default=[0],
        help="Planner chunk ceilings to sweep; 0 leaves the built-in one. A value\n"
        "below it forces the runtime-chunk kernel, which is what gfx942 runs.",
    )
    parser.add_argument(
        "-q",
        "--qo-lens",
        type=dtypes.str2tuple,
        nargs="*",
        default=DEFAULT_QO_LENS,
        help="Per-request query lengths for the non-uniform bf16 sweep.\n"
        "e.g.: -q 2,1,3 4,4,4,0",
    )
    parser.add_argument(
        "--qo-nhead",
        type=int,
        nargs="*",
        default=DEFAULT_QO_NHEADS,
        help="q-heads for the non-uniform bf16 sweep (96 folds on gfx942).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-iters", type=int, default=101)
    args = parser.parse_args()

    nhead = args.nhead if args.nhead is not None else kimi_nhead(args.tensor_parallel)
    # MXFP4 recipe: --kv_cache_dtype fp8, q quantized to fp8 to match KV.
    dtype = dtypes.fp8
    kvtype = dtypes.fp8

    print(
        f"gfx={get_gfx()} cu={torch.cuda.get_device_properties(0).multi_processor_count} "
        f"| Kimi-K2.5-MXFP4 decode | nhead={nhead} nhead_kv={KIMI_NHEAD_KV} "
        f"dtype={dtype} kv={kvtype} page_size={PAGE_SIZE} "
        f"kv_gran={KV_GRANULARITY} max_split_per_batch={MAX_SPLIT_PER_BATCH} "
        f"jitter={args.jitter}"
    )

    rows = []
    all_match = True
    for batch_chunk, ctx_len, batch_size in itertools.product(
        args.batch_chunk, args.ctx_len, args.batch
    ):
        row = test_metadata(
            batch_size,
            ctx_len,
            dtype,
            kvtype,
            nhead,
            args.jitter,
            args.seed,
            args.num_iters,
            batch_chunk,
        )
        rows.append(row)
        all_match = all_match and row["match"]

    df = pd.DataFrame(rows)
    aiter.logger.info(
        "mla metadata planner summary (markdown):\n%s", df.to_markdown(index=False)
    )

    # The fold is a bf16 decision on gfx942, so this sweep pins bf16.
    nonuniform_rows = [
        test_nonuniform_metadata(
            qo_lens, qo_nhead, dtypes.bf16, dtypes.bf16, args.num_iters
        )
        for qo_nhead, qo_lens in itertools.product(args.qo_nhead, args.qo_lens)
    ]
    if nonuniform_rows:
        aiter.logger.info(
            "mla metadata non-uniform qo summary (markdown):\n%s",
            pd.DataFrame(nonuniform_rows).to_markdown(index=False),
        )

    assert rows, (
        "the sweep ran no shapes -- an empty axis would otherwise report a pass "
        "without comparing anything"
    )
    assert all_match, "parallel MLA metadata planner diverged from serial reference"
    assert all(
        row["match"] for row in nonuniform_rows
    ), "non-uniform MLA metadata left the folded Q layout or diverged across planners"


if __name__ == "__main__":
    main()
