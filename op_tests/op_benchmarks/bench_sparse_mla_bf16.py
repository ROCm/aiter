# SPDX-License-Identifier: MIT
"""Compare HIP v1/v2 and same-checkout Gluon on disjoint per-query KV pools.

python op_tests/op_benchmarks/bench_sparse_mla_bf16.py --verify --output result.jsonl
Events inside captured graphs time main + reducer, excluding Python/JIT overhead.
The cold run flushes 512 MiB before the start event. Logical GB/s is not measured
HBM traffic. Sampled FP64 accuracy is checked before and after every timed case.
"""

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

import torch
import triton

import aiter
from aiter.jit.core import get_user_jit_dir
from aiter.ops.triton.attention import sparse_mla as gluon
from op_tests.test_sparse_mla_bf16 import check, reference


def ints(value):
    return [int(x) for x in value.split(",")]


def tensor_hash(tensor):
    data = tensor.detach().contiguous().view(torch.uint8).flatten()
    digest = hashlib.sha256()
    for start in range(0, data.numel(), 8 << 20):
        digest.update(data[start : start + (8 << 20)].cpu().numpy().tobytes())
    return digest.hexdigest()


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--heads", type=ints, default=[16, 64])
    parser.add_argument(
        "--queries", type=ints, default=[1, 32, 128, 256, 512, 1024, 2048]
    )
    parser.add_argument("--topk", type=ints, default=[2048, 2051])
    parser.add_argument("--splits", type=ints, default=[1, 2, 4, 8, 16, 32])
    parser.add_argument("--context", type=int, default=32768)
    parser.add_argument("--samples", type=int, default=60)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--check-rows", type=int, default=4)
    parser.add_argument("--cache-modes", default="warm,cold")
    parser.add_argument("--return-lse", action="store_true")
    parser.add_argument("--purpose", choices=["timing", "trace"], default="timing")
    parser.add_argument("--reverse", action="store_true")
    parser.add_argument("--verify", action="store_true", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("output already exists; use a fresh run")
    assert args.samples >= 1 and args.check_rows >= 1
    assert args.purpose == "trace" or (args.samples >= 30 and args.warmup >= 10)
    assert all(q > 0 for q in args.queries)
    assert set(args.heads) <= {16, 64} and set(args.splits) <= {1, 2, 4, 8, 16, 32}
    assert set(args.cache_modes.split(",")) <= {"warm", "cold"}
    assert all(0 < k <= args.context for k in args.topk)
    assert torch.version.hip and torch.cuda.device_count() == 1
    torch.cuda.set_device(0)
    torch.set_num_threads(4)
    props = torch.cuda.get_device_properties(0)
    assert props.gcnArchName.split(":")[0] == "gfx950"

    def emit(row):
        with args.output.open("a") as output:
            output.write(json.dumps(row, allow_nan=False) + "\n")
        keys = (
            "type",
            "heads",
            "queries",
            "topk",
            "variant",
            "splits",
            "cache",
            "median_us",
            "status",
        )
        print(json.dumps({k: row[k] for k in keys if k in row}), flush=True)

    source = Path(__file__).resolve()
    jit = Path(get_user_jit_dir()) / "module_sparse_mla_bf16.so"
    emit(
        {
            "type": "config",
            "command": sys.argv,
            "device": str(props),
            "torch": str(torch.__version__),
            "triton": triton.__version__,
            "hip": torch.version.hip,
            "benchmark_sha256": file_hash(source),
            "aiter": aiter.__file__,
            "gluon_source_sha256": file_hash(Path(gluon.__file__)),
            "native_library_sha256": file_hash(jit) if jit.exists() else None,
            "scope": "captured attention main+reduce; not model/engine latency",
            "pool_layout": "disjoint",
            "output_contract": "O+LSE" if args.return_lse else "O-only",
            "args": vars(args) | {"output": str(args.output)},
        }
    )
    stream = torch.cuda.Stream()
    flush = torch.empty(512 << 20, device="cuda", dtype=torch.uint8)
    cases = [(h, q, k) for h in args.heads for q in args.queries for k in args.topk]
    if args.reverse:
        cases.reverse()
    total = failed = 0
    for heads, queries, topk in cases:
        with torch.cuda.stream(stream):
            torch.manual_seed(1701 + heads + queries + topk)
            q = (
                torch.randn((queries, heads, 512), device="cuda", dtype=torch.bfloat16)
                * 0.5
            )
            kv = torch.empty(
                (queries * args.context, 512), device="cuda", dtype=torch.bfloat16
            )
            kv.normal_().mul_(0.5)
            idx = torch.cat(
                [
                    torch.randperm(args.context, device="cuda")[:topk].to(torch.int32)
                    + row * args.context
                    for row in range(queries)
                ]
            )
            ptr = torch.arange(queries + 1, device="cuda", dtype=torch.int32) * topk
            out = torch.empty_like(q)
            lse = torch.empty(q.shape[:2], device="cuda") if args.return_lse else None
            max_s = max(args.splits)
            po = torch.empty((queries * max_s * heads * 512,), device="cuda")
            pl = torch.empty((queries * max_s * heads,), device="cuda")
            rows = sorted(
                {
                    round(i * (queries - 1) / max(1, min(queries, args.check_rows) - 1))
                    for i in range(min(queries, args.check_rows))
                }
            )
            selected = torch.tensor(rows, device="cuda", dtype=torch.long)
            oracle_ptr = (
                torch.arange(len(rows) + 1, device="cuda", dtype=torch.int32) * topk
            )
            expected = reference(
                q[selected],
                kv,
                oracle_ptr,
                idx.view(queries, topk)[selected].flatten(),
                1 / 16,
            )
            tensors = {"q": q, "kv": kv, "indptr": ptr, "indices": idx}
            hashes = {name: tensor_hash(value) for name, value in tensors.items()}
            case = {
                "heads": heads,
                "queries": queries,
                "topk": topk,
                "pool_bytes": kv.numel() * 2,
                "input_sha256": hashes,
                "input_addresses": {
                    name: value.data_ptr() for name, value in tensors.items()
                },
                "out_address": out.data_ptr(),
                "checked_rows": rows,
                "output_contract": "O+LSE" if args.return_lse else "O-only",
            }
            variants = [(f"v{v}", s) for s in args.splits for v in (1, 2)]
            variants += [("gluon", s) for s in args.splits] + [("gluon_default", None)]
            if args.reverse:
                variants.reverse()
            runners = []
            for variant, splits in variants:
                state = [None]
                if variant.startswith("v"):
                    workspace = (
                        (
                            po[: queries * splits * heads * 512].view(
                                queries, splits, heads, 512
                            ),
                            pl[: queries * splits * heads].view(queries, splits, heads),
                        )
                        if splits > 1
                        else None
                    )

                    def invoke(
                        v=int(variant[1]),
                        s=splits,
                        workspace=workspace,
                        state=state,
                        inputs=(q, kv, ptr, idx, out, lse),
                    ):
                        q, kv, ptr, idx, out, lse = inputs
                        state[0] = aiter.sparse_mla_bf16_fwd(
                            q,
                            kv,
                            ptr,
                            idx,
                            1 / 16,
                            version=v,
                            kv_splits=s,
                            out=out,
                            return_lse=args.return_lse,
                            lse=lse,
                            workspace=workspace,
                        )

                    actual_splits = splits
                else:

                    def invoke(s=splits, state=state, inputs=(q, kv, ptr, idx, out)):
                        q, kv, ptr, idx, out = inputs
                        state[0] = gluon.sparse_mla_fwd(
                            q,
                            kv,
                            ptr,
                            idx,
                            1 / 16,
                            kv_lora_rank=512,
                            qk_rope_head_dim=0,
                            kv_splits=s,
                            has_invalid=False,
                            out=out,
                            return_lse=args.return_lse,
                        )

                    actual_splits = splits or gluon._mla_num_splits(
                        queries, heads // 16, topk
                    )
                key = {"variant": variant, "splits": actual_splits}
                invoke()
                torch.cuda.synchronize()
                try:
                    before = check(
                        (
                            state[0][0][selected],
                            state[0][1][selected] if state[0][1] is not None else None,
                        ),
                        expected,
                    )
                except AssertionError as error:
                    failed += 1
                    emit(
                        {
                            "type": "accuracy_failure",
                            **case,
                            **key,
                            "status": "excluded",
                            "error": str(error),
                        }
                    )
                    if variant.startswith("v"):
                        raise
                    continue
                start = torch.cuda.Event(enable_timing=True, external=True)
                end = torch.cuda.Event(enable_timing=True, external=True)
                start.record()
                end.record()
                stream.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    start.record()
                    invoke()
                    end.record()
                graph.replay()
                stream.synchronize()
                check(
                    (
                        state[0][0][selected],
                        state[0][1][selected] if state[0][1] is not None else None,
                    ),
                    expected,
                )
                runners.append(
                    {
                        **key,
                        "graph": graph,
                        "start": start,
                        "end": end,
                        "state": state,
                        "before_rms": before,
                    }
                )
            for mode in args.cache_modes.split(","):
                for i in range(args.warmup):
                    for r in (runners if i % 2 == 0 else runners[::-1]):
                        if mode == "cold":
                            flush.fill_(1)
                        r["graph"].replay()
                stream.synchronize()
                measured = [[] for _ in runners]
                for i in range(args.samples):
                    order = (
                        range(len(runners))
                        if i % 2 == 0
                        else reversed(range(len(runners)))
                    )
                    for j in order:
                        r = runners[j]
                        if mode == "cold":
                            flush.fill_(1)
                        r["graph"].replay()
                        r["end"].synchronize()
                        value = r["start"].elapsed_time(r["end"]) * 1000
                        assert math.isfinite(value) and value > 0
                        measured[j].append(value)
                for r, samples in zip(runners, measured):
                    r["graph"].replay()
                    stream.synchronize()
                    output, output_lse = r["state"][0]
                    after = check(
                        (
                            output[selected],
                            output_lse[selected] if output_lse is not None else None,
                        ),
                        expected,
                    )
                    median = statistics.median(samples)
                    logical_bytes = (
                        queries * (heads * 512 * 4 + topk * (512 * 2 + 4))
                        + (queries + 1) * 4
                    )
                    if args.return_lse:
                        logical_bytes += queries * heads * 4
                    emit(
                        {
                            "type": (
                                "timing"
                                if args.purpose == "timing"
                                else "diagnostic_timing"
                            ),
                            **case,
                            "variant": r["variant"],
                            "splits": r["splits"],
                            "cache": mode,
                            "samples_us": samples,
                            "median_us": median,
                            "p90_us": sorted(samples)[
                                math.ceil(len(samples) * 0.9) - 1
                            ],
                            "logical_bytes": logical_bytes,
                            "logical_gbps": logical_bytes / median / 1000,
                            "qk_pv_tflops": 4
                            * queries
                            * heads
                            * topk
                            * 512
                            / median
                            / 1e6,
                            "before_rms": r["before_rms"],
                            "after_rms": after,
                            "status": "passed",
                        }
                    )
                    total += 1
            assert hashes == {
                name: tensor_hash(value) for name, value in tensors.items()
            }
            emit(
                {
                    "type": "case_complete",
                    **case,
                    "status": "passed",
                    "inputs_unchanged": True,
                }
            )
            runners.clear()
            del (
                q,
                kv,
                ptr,
                idx,
                out,
                lse,
                po,
                pl,
                tensors,
                invoke,
                state,
                graph,
                r,
                output,
                output_lse,
            )
        gc.collect()
        torch.cuda.empty_cache()
    emit(
        {
            "type": "complete",
            "status": "passed",
            "timing_records": total,
            "accuracy_exclusions": failed,
        }
    )


if __name__ == "__main__":
    main()
