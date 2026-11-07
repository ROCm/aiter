# SPDX-License-Identifier: MIT
"""Additional random signed-FP4/scale/input regression for the pinned recipe."""

import argparse
import importlib
import importlib.util
import json
import os
from pathlib import Path

import torch
from make_overlay import UPDATE, check_source, make_overlay

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--aiter-root", type=Path, required=True)
ap.add_argument("--baseline", type=Path, required=True)
ap.add_argument("--output", type=Path, required=True)
args = ap.parse_args()
check_source(args.aiter_root)
make_overlay(args.baseline.read_text())
os.environ["AITER_CONFIG_GROUPED_FMOE"] = str(args.baseline)
os.environ["AITER_LOG_LEVEL"] = "WARNING"

torch.set_num_threads(1)


def module(name, path):
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


gm = importlib.import_module("aiter.ops.flydsl.grouped_moe_gfx1250")
fm = importlib.import_module("aiter.fused_moe")
helper = module(
    "inputs", args.aiter_root / "op_tests/test_moe_prefill_route_in_quant.py"
)
route = module("routes", args.aiter_root / "op_tests/test_moe_tp_route_quant_fused.py")
fixed, kw = helper._moe_args(16)
original = gm._find_grouped_config
override = {}


def getcfg(**k):
    return dict(original(**k), **override)


gm._find_grouped_config = getcfg
configs = [UPDATE]
rows = []
for seed in (401, 402):
    torch.manual_seed(seed)
    for t in fixed[1:3]:
        t.view(torch.uint8).random_(0, 256)
    for exp in (116, 119, 122, 125):
        for key in ("w1_scale", "w2_scale"):
            kw[key].view(torch.uint8).random_(exp - 1, exp + 2)
        for m, amp in ((9, 0.125), (12, 0.5), (16, 1.0), (16, 4.0)):
            x = (torch.randn(m, 6144, device="cuda") * amp).bfloat16()
            ids = route.m3_routes(m, 129, seed + m)
            weight = torch.rand(m, 5, device="cuda")
            weight[:, :4] /= weight[:, :4].sum(1, keepdim=True)
            weight[:, -1] = 1

            def run():
                return fm.fused_moe(x, fixed[1], fixed[2], weight, ids, **kw)

            override = {}
            old = run().clone()
            for i, c in enumerate(configs):
                override = c
                new = run()
                a, b = old.float(), new.float()
                delta = a - b
                den = (a.square() + b.square()).sum().item()
                row = dict(
                    seed=seed,
                    scale_exp=exp,
                    m=m,
                    amp=amp,
                    config=i,
                    exact=torch.equal(old, new),
                    finite=bool(torch.isfinite(new).all()),
                    logits_diff=delta.square().sum().item() / max(den, 1e-30),
                    maxerr=delta.abs().max().item(),
                    out_rms=b.square().mean().sqrt().item(),
                    mismatched=int((old != new).sum().item()),
                    elements=old.numel(),
                )
                rows.append(row)
                print(json.dumps(row), flush=True)
assert all(r["finite"] and r["logits_diff"] < 1e-5 for r in rows)
args.output.write_text(json.dumps(rows, indent=2))
