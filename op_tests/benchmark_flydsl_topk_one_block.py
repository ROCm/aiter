# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Compare one-block TopK kernel sources using alternating GPU graph timings.

Example: python op_tests/benchmark_flydsl_topk_one_block.py --baseline /tmp/before.py
Use --stable 0, --values, and --decode to cover the other kernel modes.
"""

import argparse
import importlib.util
import json
import statistics
import sys
import tempfile
from functools import partial
from pathlib import Path

import torch

from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.kernels.kernels_common import get_warp_size
from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled
from aiter.ops.flydsl.topk_per_row import (
    _ONE_BLOCK_LDS_BUDGET_BYTES,
    _SHORT_ROWS_1024_THREAD_MAX_ROWS,
    _should_use_one_block,
)

ROOT = Path(__file__).resolve().parents[1]
KERNEL = ROOT / "aiter/ops/flydsl/kernels/radix_topk_one_block.py"


def load_variant(tag, source_path, scratch):
    # Give A/B modules distinct kernel symbols and source files for JIT caching.
    source = source_path.read_text().replace(
        'f"radix_topk_one_block_', f'f"radix_ob_benchmark_{tag}_'
    )
    path = Path(scratch) / f"variant_{tag}.py"
    path.write_text(source)
    name = f"aiter.ops.flydsl.kernels._ob_benchmark_{tag}"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def ordered_keys(values):
    """Match one-block's existing bitwise ordering, including signed zero/NaNs."""
    raw = values.view(torch.int32).to(torch.int64)
    return (raw ^ ((raw >> 31) & 0x7FFFFFFF) ^ -2147483648) & 0xFFFFFFFF


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--baseline",
        type=Path,
        help="Previous kernel source for an alternating A/B comparison",
    )
    ap.add_argument(
        "--shapes",
        nargs="+",
        default=[
            "1x4096",
            "4x32768",
            "16x49152",
            "32x65535",
            "64x131072",
            "128x262144",
            "1024x53248",
        ],
    )
    ap.add_argument(
        "--snapshot",
        type=Path,
        help="Prefill torch snapshot with logits/rowStarts/rowEnds",
    )
    ap.add_argument("--current", type=Path, default=KERNEL)
    ap.add_argument("--cases", nargs="+", default=["randn", "concentrated", "equal"])
    ap.add_argument("--k", type=int, default=2048)
    ap.add_argument("--stable", type=int, default=1)
    ap.add_argument("--values", action="store_true")
    ap.add_argument("--decode", action="store_true")
    ap.add_argument("--iters", type=int, default=32)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--output", default="/tmp/flydsl_topk_one_block_results.json")
    args = ap.parse_args()
    arch = get_gfx()
    wave = get_warp_size(arch)
    scratch = tempfile.TemporaryDirectory(prefix="flydsl_topk_benchmark_")
    sources = {"base": args.baseline} if args.baseline else {}
    sources["current"] = args.current
    modules = {
        tag: load_variant(tag, path, scratch.name) for tag, path in sources.items()
    }
    stream = torch.cuda.Stream()
    torch.cuda.set_stream(stream)
    print("CONFIG", arch, torch.__version__, vars(args), flush=True)
    results = []
    for shape in args.shapes:
        rows, width = map(int, shape.split("x"))
        for case in args.cases:
            if case == "real":
                if shape != args.shapes[0]:
                    continue
                if args.snapshot is None or args.decode:
                    ap.error("--cases real requires --snapshot and prefill mode")
                data = torch.load(args.snapshot, map_location="cuda", weights_only=True)
                x = data["logits"]
                starts, ends = data["rowStarts"].cuda(), data["rowEnds"].cuda()
                rows, width = x.shape
                assert bool((ends - starts >= args.k).all())
                del data
            else:
                rows, width = map(int, shape.split("x"))
                torch.manual_seed(123)
                x = torch.randn(rows, width, device="cuda")
                if case == "concentrated":
                    x = 68.0 + x * 2.7
                elif case == "equal":
                    x.fill_(68.0)
                elif case == "uniform":
                    x = torch.rand_like(x) * 0.1 + 0.9
                elif case == "ties":
                    x = (x * 8).round() / 8
                starts = torch.zeros(rows, dtype=torch.int32, device="cuda")
                ends = torch.full((rows,), width, dtype=torch.int32, device="cuda")
            keys = ordered_keys(x)
            columns = torch.arange(width, device=x.device)[None, :]
            keys.masked_fill_(
                (columns < starts[:, None]) | (columns >= ends[:, None]), -1
            )
            want = torch.argsort(keys, dim=1, descending=True, stable=True)[:, : args.k]
            want = want.sort(dim=1).values.to(torch.int32)
            del keys, columns
            assert _should_use_one_block(arch, rows, width), (arch, rows, width)
            graphs = {}
            buffers = []
            for tag, module in modules.items():
                out = torch.empty((rows, args.k), dtype=torch.int32, device="cuda")
                val = torch.empty_like(out, dtype=torch.float32) if args.values else x
                short_rows = width <= 4096
                block_threads = (
                    1024
                    if not short_rows or rows <= _SHORT_ROWS_1024_THREAD_MAX_ROWS
                    else 256
                )
                launcher = module.build_radix_topk_one_block_module(
                    args.k,
                    block_threads=block_threads,
                    stable=bool(args.stable),
                    wave_size=wave,
                    short_rows=short_rows,
                    write_values=args.values,
                    is_decode=args.decode,
                    arch=arch,
                    lds_budget_bytes=_ONE_BLOCK_LDS_BUDGET_BYTES.get(arch, 0),
                )

                run = partial(
                    _run_compiled,
                    launcher,
                    x,
                    starts,
                    ends,
                    out,
                    val,
                    width,
                    1,
                    rows,
                    stream,
                )
                print("COMPILE", shape, case, tag, flush=True)
                run()
                torch.cuda.synchronize()
                if args.stable:
                    assert torch.equal(out, want), (shape, case, tag, "indices")
                else:
                    assert bool(((out >= 0) & (out < width)).all()), (
                        shape,
                        case,
                        tag,
                        "bounds",
                    )
                    assert bool(
                        (
                            out.sort(dim=1).values[:, 1:]
                            != out.sort(dim=1).values[:, :-1]
                        ).all()
                    ), (shape, case, tag, "unique")
                    assert torch.equal(
                        ordered_keys(x.gather(1, out.long())).sort(dim=1).values,
                        ordered_keys(x.gather(1, want.long())).sort(dim=1).values,
                    ), (shape, case, tag, "members")
                if args.values:
                    assert torch.equal(
                        val.view(torch.int32), x.gather(1, out.long()).view(torch.int32)
                    ), (
                        shape,
                        case,
                        tag,
                        "values",
                    )
                for _ in range(3):
                    run()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    for _ in range(args.iters):
                        run()
                graphs[tag] = graph
                buffers.append((out, val, launcher))
            measurements = {tag: [] for tag in modules}
            for rep in range(args.rounds):
                tags = list(modules)
                if rep % 2:
                    tags.reverse()
                for tag in tags:
                    graphs[tag].replay()
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                        enable_timing=True
                    )
                    start.record()
                    graphs[tag].replay()
                    end.record()
                    end.synchronize()
                    measurements[tag].append(
                        start.elapsed_time(end) * 1000 / args.iters
                    )
            result = {
                "case": case,
                "rows": rows,
                "width": width,
                "k": args.k,
                "stable": bool(args.stable),
                "values": args.values,
                "decode": args.decode,
                "us": {tag: statistics.median(v) for tag, v in measurements.items()},
                "rounds_us": measurements,
            }
            results.append(result)
            Path(args.output).write_text(json.dumps(results, indent=2))
            print("RESULT", json.dumps(result), flush=True)
            del graphs


if __name__ == "__main__":
    main()
