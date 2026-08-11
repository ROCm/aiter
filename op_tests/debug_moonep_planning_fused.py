# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Localise a fused-planner mismatch by diffing its scratch against the multi one.

The two planners share an algorithm but not a layout: ``local_hist`` is
``[vblock][expert]`` in the three-launch planner and ``[expert][vblock]`` in the
fused one, so it is compared after transposing.
"""

import torch

from aiter.ops.flydsl.moonep import (
    MoonEPFusedPlanner,
    MoonEPGpuPlanner,
    MoonEPPlanConfig,
    build_reference_plan,
)

R, S, K, E, TP = 4, 256, 4, 32, 16


def main() -> int:
    device = torch.device("cuda")
    g = torch.Generator(device="cpu").manual_seed(0)
    topk_all = [
        torch.rand(S, E, generator=g).topk(K, dim=1).indices.to(torch.int32).to(device)
        for _ in range(R)
    ]
    tpe = torch.stack(
        [
            torch.bincount(t.reshape(-1).to(torch.int64), minlength=E).to(torch.int32)
            for t in topk_all
        ]
    ).to(device)

    config = MoonEPPlanConfig(
        rank=0, world_size=R, num_tokens=S, top_k=K, num_experts=E, token_padding=TP
    )
    topk = topk_all[0]

    multi = MoonEPGpuPlanner(config, device)
    fused = MoonEPFusedPlanner(config, device)
    multi.build(topk, tpe)
    fused.build(topk, tpe)
    torch.cuda.synchronize()

    print(f"multi geo: NV={multi.geo.NV} EPV={multi.geo.EPV}")
    print(f"fused geo: NV={fused.geo.NV} EPV={fused.geo.EPV} blocks={fused.geo.blocks}")

    def report(name, a, b):
        same = torch.equal(a.cpu(), b.cpu())
        msg = "OK " if same else "DIFF"
        extra = ""
        if not same:
            bad = (a.cpu() != b.cpu()).nonzero()
            i = tuple(bad[0].tolist())
            extra = (
                f"  {bad.shape[0]}/{a.numel()} slots, first {i} "
                f"multi={a.cpu()[i].item()} fused={b.cpu()[i].item()}"
            )
        print(f"  [{msg}] {name}{extra}")

    print("scratch:")
    report("tpe_prefix", multi._tpe_prefix, fused._tpe_prefix)
    report("alloc_cumsum", multi._alloc_cumsum, fused._alloc_cumsum)
    report("expert_off", multi._expert_off, fused._expert_off)
    report("order", multi._order, fused._order)

    # local_hist layouts differ; compare on the common [vblock][expert] view.
    nv_m, nv_f = multi.geo.NV, fused.geo.NV
    hm = multi._local_hist.view(nv_m, E)
    hf = fused._local_hist.view(E, nv_f).t().contiguous()
    n = min(nv_m, nv_f)
    report(f"local_hist[:{n}] (prefix)", hm[:n], hf[:n])

    # An order mismatch is the likeliest single-slot off-by-one; show its shape.
    om, of = multi._order.cpu(), fused._order.cpu()
    if not torch.equal(om, of):
        d = (of - om)
        vals, counts = torch.unique(d, return_counts=True)
        print(f"  order delta histogram: {dict(zip(vals.tolist(), counts.tolist()))}")
        idx = (d != 0).nonzero().flatten()[:8].tolist()
        print(f"  first differing flat indices: {idx}")
        print(f"  experts there: {topk.reshape(-1)[idx].tolist()}")

    # Golden vblock histogram prefix, computed on the host straight from topk.
    flat = topk.reshape(-1).cpu()
    golden = torch.zeros(fused.geo.NV, E, dtype=torch.int32)
    for v in range(fused.geo.NV):
        lo = v * fused.geo.EPV
        hi = min(lo + fused.geo.EPV, flat.numel())
        if hi > lo:
            golden[v] = torch.bincount(
                flat[lo:hi].to(torch.int64), minlength=E
            ).to(torch.int32)
    golden_prefix = torch.cumsum(golden, dim=0) - golden
    print("vs host golden vblock prefix:")
    report("  fused local_hist", golden_prefix, hf)
    report("  multi local_hist", golden_prefix[:nv_m], hm)

    # A broken grid barrier shows up as run-to-run drift.
    snap = [fused.dst.clone(), fused._local_hist.clone()]
    for _ in range(5):
        fused.build(topk, tpe)
    torch.cuda.synchronize()
    print("determinism over 5 more fused builds:")
    report("  dst", snap[0], fused.dst)
    report("  local_hist", snap[1], fused._local_hist)

    ref = build_reference_plan(config, topk, tpe)
    for name in ("dst", "cu_seqlens", "alloc", "experts_to_copy", "remote_stats"):
        print(f"vs reference ({name}):")
        report(f"  multi.{name}", getattr(ref, name), getattr(multi, name))
        report(f"  fused.{name}", getattr(ref, name), getattr(fused, name))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
