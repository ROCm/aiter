# SPDX-License-Identifier: MIT
"""Pinned 11-10 recipe, real public fused_moe A/B with four weight buffers.

python bench_moe.py --aiter-root /app/aiter --baseline /tmp/aiter_configs/tuned_grouped_fmoe.csv --output result.json
The fixture uses the pinned recipe's input-only builder; timed calls are real ops.
"""

import argparse
import importlib
import importlib.util
import json
import os
import pathlib
import statistics

from make_overlay import check_source, make_overlay


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--aiter-root", type=pathlib.Path, required=True)
    ap.add_argument("--baseline", type=pathlib.Path, required=True)
    ap.add_argument("--output", type=pathlib.Path, required=True)
    ap.add_argument("--seed", type=int, default=393)
    args = ap.parse_args()
    check_source(args.aiter_root)
    make_overlay(args.baseline.read_text())  # validates the exact baseline row
    os.environ["AITER_CONFIG_GROUPED_FMOE"] = str(args.baseline)
    os.environ["AITER_LOG_LEVEL"] = "WARNING"
    import torch

    gm = importlib.import_module("aiter.ops.flydsl.grouped_moe_gfx1250")
    fm = importlib.import_module("aiter.fused_moe")
    from make_overlay import UPDATE

    assert pathlib.Path(gm.__file__).resolve().is_relative_to(args.aiter_root.resolve())
    helper = load(
        "_m3_fixture", args.aiter_root / "op_tests/test_moe_prefill_route_in_quant.py"
    )
    route = load(
        "_m3_routes", args.aiter_root / "op_tests/test_moe_tp_route_quant_fused.py"
    )
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    fixture, kw = helper._moe_args(16)
    sets = [(fixture[1], fixture[2], kw["w1_scale"], kw["w2_scale"])]
    sets += [tuple(x.clone() for x in sets[0]) for _ in range(3)]
    original = gm._find_grouped_config
    enabled = False

    def config(**k):
        row = original(**k)
        if enabled and k["token_num"] == 16 and not k.get("ep_fused", False):
            return dict(row, **UPDATE)
        return row

    gm._find_grouped_config = config
    rows = []
    try:
        for m in range(9, 17):
            inputs = []
            for slot in range(4):
                x = (torch.randn(m, 6144, device="cuda") * 0.25).bfloat16()
                ids = route.m3_routes(m, 129, args.seed + slot)
                weights = torch.rand(m, 5, device="cuda")
                weights[:, :4] /= weights[:, :4].sum(1, keepdim=True)
                weights[:, -1] = 1.0
                inputs.append((x, ids, weights))

            def run(slot):
                x, ids, weights = inputs[slot]
                w1, w2, s1, s2 = sets[slot]
                return fm.fused_moe(
                    x, w1, w2, weights, ids, **dict(kw, w1_scale=s1, w2_scale=s2)
                )

            graphs, outputs = [], []
            # A/A controls are independently captured just like B.
            for mode in (False, False, True):
                enabled = mode
                for slot in range(4):
                    run(slot)
                torch.cuda.synchronize()
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    for _ in range(8):
                        out = [run(slot) for slot in range(4)]
                g.replay()
                torch.cuda.synchronize()
                graphs.append(g)
                outputs.append(out)
            for out in outputs[1:]:
                for x, y in zip(outputs[0], out):
                    torch.testing.assert_close(x, y, atol=0, rtol=0)
            for slot, (x, ids, weights) in enumerate(inputs):
                x.neg_()
                ids.copy_(
                    route.m3_routes(m, 129, args.seed + 50 + slot, real=max(1, m - 3))
                )
            for g in graphs:
                g.replay()
            torch.cuda.synchronize()
            for out in outputs[1:]:
                for x, y in zip(outputs[0], out):
                    torch.testing.assert_close(x, y, atol=0, rtol=0)
            for slot, (x, ids, weights) in enumerate(inputs):
                x.neg_()
                ids.copy_(route.m3_routes(m, 129, args.seed + slot))
            times = [[] for _ in graphs]
            for rep in range(25):
                for i in (0, 1, 2) if rep % 2 == 0 else (2, 1, 0):
                    start, end = (
                        torch.cuda.Event(enable_timing=True),
                        torch.cuda.Event(enable_timing=True),
                    )
                    start.record()
                    graphs[i].replay()
                    end.record()
                    end.synchronize()
                    times[i].append(start.elapsed_time(end) * 1000 / 32)
            us = [statistics.median(v) for v in times]
            row = dict(
                m=m,
                seed=args.seed,
                base_us=us[0],
                aa_us=us[1],
                candidate_us=us[2],
                speedup=us[0] / us[2],
                aa_speedup=us[0] / us[1],
                exact=True,
                samples_us=times,
            )
            rows.append(row)
            print(
                json.dumps({k: v for k, v in row.items() if k != "samples_us"}),
                flush=True,
            )
            for g in graphs:
                g.reset()
    finally:
        gm._find_grouped_config = original
    args.output.write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
