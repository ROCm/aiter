# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import os
import argparse
import random

import torch
import pandas as pd

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.test_common import benchmark, run_perftest

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

# Default serving sweep (kimi Makefile / perf_sweep.sh).
DEFAULT_BATCHES = [4, 8, 16, 32, 64, 128]
DEFAULT_CTX_LENS = [2048, 4096, 8192]

_PARALLEL_ENV = "AITER_MLA_META_USE_PARALLEL"


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

    inputs = dict(
        qo_indptr=qo_indptr,
        kv_indptr=kv_indptr,
        kv_last_page_lens=kv_last_page_lens,
        nhead=nhead,
    )
    out_meta = dict(
        work_meta_data=(work_meta_data_size, work_meta_data_type),
        work_indptr=(work_indptr_size, work_indptr_type),
        work_info_set=(work_info_set_size, work_info_set_type),
        reduce_indptr=(reduce_indptr_size, reduce_indptr_type),
        reduce_final_map=(reduce_final_map_size, reduce_final_map_type),
        reduce_partial_map=(reduce_partial_map_size, reduce_partial_map_type),
    )
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
        max_seqlen_qo=MAX_SEQLEN_QO,
        uni_seqlen_qo=UNI_SEQLEN_QO,
        fast_mode=True,
        max_split_per_batch=MAX_SPLIT_PER_BATCH,
        dtype_q=dtype,
        dtype_kv=kvtype,
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

    wi_g = golden["work_indptr"]
    wi_t = test["work_indptr"]
    details["work_indptr"] = int((wi_g != wi_t).sum().item())

    num_works = int(wi_g[-1].item())
    wis_g = golden["work_info_set"][:num_works]
    wis_t = test["work_info_set"][:num_works]
    details["work_info_set"] = int((wis_g != wis_t).sum().item())

    ri_g = golden["reduce_indptr"]
    ri_t = test["reduce_indptr"]
    details["reduce_indptr"] = int((ri_g != ri_t).sum().item())

    # Valid prefixes for the reduce maps, derived from the golden reduce_indptr.
    steps = ri_g[1:] - ri_g[:-1]
    num_groups = int((steps > 0).sum().item())
    num_partial = int(ri_g[-1].item())

    rfm_g = golden["reduce_final_map"][:num_groups]
    rfm_t = test["reduce_final_map"][:num_groups]
    details["reduce_final_map"] = int((rfm_g != rfm_t).sum().item())

    rpm_g = golden["reduce_partial_map"][:num_partial]
    rpm_t = test["reduce_partial_map"][:num_partial]
    details["reduce_partial_map"] = int((rpm_g != rpm_t).sum().item())

    ok = all(v == 0 for v in details.values())
    return ok, details, num_works, num_groups


@benchmark()
def test_metadata(batch_size, ctx_len, dtype, kvtype, nhead, jitter, seed, num_iters):
    inputs, out_meta, kv_lens = build_decode_inputs(
        batch_size, ctx_len, dtype, kvtype, nhead, jitter=jitter, seed=seed
    )

    # Golden (serial planner) vs parallel planner -- must be bit identical.
    golden = run_path(inputs, out_meta, dtype, kvtype, use_parallel=False)
    parallel = run_path(inputs, out_meta, dtype, kvtype, use_parallel=True)
    ok, mism, num_works, num_groups = compare_metadata(golden, parallel)

    if not ok:
        print(f"  [MISMATCH] bs={batch_size} ctx={ctx_len} nhead={nhead}: {mism}")

    # Microbench both planners.
    serial_outs = alloc_outputs(out_meta)
    parallel_outs = alloc_outputs(out_meta)

    os.environ[_PARALLEL_ENV] = "0"
    _, us_serial = run_perftest(
        call_metadata, inputs, serial_outs, dtype, kvtype, num_iters=num_iters
    )
    os.environ[_PARALLEL_ENV] = "1"
    _, us_parallel = run_perftest(
        call_metadata, inputs, parallel_outs, dtype, kvtype, num_iters=num_iters
    )
    os.environ.pop(_PARALLEL_ENV, None)

    speedup = us_serial / us_parallel if us_parallel > 0 else float("nan")

    return {
        "match": ok,
        "num_works": num_works,
        "num_split_groups": num_groups,
        "serial_us": round(us_serial, 3),
        "parallel_us": round(us_parallel, 3),
        "speedup": round(speedup, 3),
    }


def main():
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
        nargs="*",
        default=DEFAULT_BATCHES,
        help="Batch sizes (== serving concurrency).",
    )
    parser.add_argument(
        "-c",
        "--ctx-len",
        type=int,
        nargs="*",
        default=DEFAULT_CTX_LENS,
        help="KV context lengths (== serving ISL).",
    )
    parser.add_argument(
        "--jitter",
        action="store_true",
        help="Randomize per-sequence KV length in [ctx/2, ctx] (decode spread).",
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
    for ctx_len in args.ctx_len:
        for batch_size in args.batch:
            row = test_metadata(
                batch_size,
                ctx_len,
                dtype,
                kvtype,
                nhead,
                args.jitter,
                args.seed,
                args.num_iters,
            )
            rows.append(row)
            all_match = all_match and row["match"]

    df = pd.DataFrame(rows)
    cols = [
        "batch_size",
        "ctx_len",
        "nhead",
        "num_works",
        "num_split_groups",
        "match",
        "serial_us",
        "parallel_us",
        "speedup",
    ]
    cols = [c for c in cols if c in df.columns]
    print(df[cols].to_string(index=False))

    assert all_match, "parallel MLA metadata planner diverged from serial reference"
    print("\nAll shapes: parallel planner matches serial reference. ✓")


if __name__ == "__main__":
    main()
