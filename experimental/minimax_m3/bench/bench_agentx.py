"""MiniMax-M3 mono microbenchmark at the AgentX decode shape, TP4: ``reqs`` requests (c=1: 1) of
``ql`` rows (EAGLE3 verify, 1 + 3 draft tokens) after a ``ctx``-token prefill (the caches hold
random keys for the whole context). One call = one decode step through ``layers`` sparse
layers (57 = MiniMax-M3's 3..59); ``steps`` calls back to back in one HIP graph, replayed REPS
times. Reports us per step (median replay / steps, slowest rank) for each backend.

    torchrun --nproc_per_node 4 bench/bench_agentx.py [--backends flydsl,gluon] [--ctx 327680]
        [--ql 4] [--reqs 1] [--layers 57] [--steps 10] [--check]

--check: each backend's step output vs FlyDSL's on the same inputs (nRMSE), first eager step."""
import argparse
import os
import sys
import time

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_layer_tp import make_runner, nrmse  # noqa: E402
from m3_inputs import make_layers, make_spec_step  # noqa: E402

REPS = 20


def bench_steps(fn, group, steps):
    fn()
    torch.cuda.synchronize()
    dist.barrier(group=group)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(steps):
            fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(REPS):
        dist.barrier(group=group)
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        g.replay()
        b.record()
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(b) * 1000 / steps)
    ts.sort()
    t = torch.tensor([ts[len(ts) // 2], ts[0], ts[-1]])
    dist.all_reduce(t, op=dist.ReduceOp.MAX, group=group)
    del g
    return t.tolist()


def sparse_tables(backend, r, rows):
    """The last selecting layer's per-row selection: [rows, 17] (16 block ids by slot, the sparse
    context length) -- FlyDSL's sparse_table (page-16 ids, 8 a block) or the Gluon stab."""
    if backend == "flydsl":
        from vllm.models.minimax_m3.amd.mono.config import PAGE_STRIDE
        from vllm.models.minimax_m3.amd.mono.layout import SCRATCH, SPARSE_TABLE_WORDS
        w = r.scratch4[SCRATCH["sparse_table"]:].view(torch.int32)[: rows * SPARSE_TABLE_WORDS].view(rows, -1)
        return torch.cat([w[:, 0:128:8] // PAGE_STRIDE, w[:, 128:129]], 1).cpu()
    from vllm.models.minimax_m3.amd.mono.gluon import layout as LY
    w = r.scratch[LY.StepLayout().stab():].view(torch.int32)[: rows * LY.STAB_WORDS].view(rows, -1)
    return w[:, :17].cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backends", default="flydsl")
    ap.add_argument("--ctx", default="327680")
    ap.add_argument("--ql", type=int, default=4)
    ap.add_argument("--reqs", type=int, default=1)
    ap.add_argument("--layers", type=int, default=57)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--timeline", action="store_true")
    ap.add_argument("--eager", type=int, default=0, help="N eager steps (no graph): for rocprofv3 / ATT")
    a = ap.parse_args()
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)
    dist.init_process_group("gloo")
    group = dist.new_group(backend="gloo")
    ctxs = [int(x) for x in a.ctx.split(",")]
    rows = a.reqs * a.ql
    need = (max(ctxs) + a.ql + 127) // 128
    nb = a.reqs * need + 8
    t0 = time.time()
    layers = make_layers(a.layers, nb, rank, dev, max_pos=max(ctxs) + a.ql + 16)
    torch.cuda.synchronize()
    if rank == 0:
        print(f"layers {a.layers} nb {nb} built in {time.time() - t0:.1f} s, "
              f"{torch.cuda.memory_allocated(dev) / 2**30:.1f} GiB", flush=True)
    backends = a.backends.split(",")
    runners = {b: make_runner(b, layers, rank, group, dev, a.debug, a.timeline) for b in backends}
    hidden0 = torch.randn(rows, 6144, generator=torch.Generator(device=dev).manual_seed(5), device=dev).to(torch.bfloat16)
    resid0 = (torch.randn(rows, 6144, generator=torch.Generator(device=dev).manual_seed(6), device=dev) * 4).to(torch.bfloat16)
    kv0 = [(lw.attn.kv_cache.clone(), lw.index_cache.kv_cache.clone()) for lw in layers] if a.check else None
    for ctx in ctxs:
        res, outs, tabs = {}, {}, {}
        for b, r in runners.items():
            if kv0 is not None:  # --check: every backend's first step from the same pristine caches
                for lw, (kc, ic) in zip(layers, kv0):
                    lw.attn.kv_cache.copy_(kc)
                    lw.index_cache.kv_cache.copy_(ic)
            if b == "gluon":
                r.ql = a.ql
            md, positions = make_spec_step(a.reqs, a.ql, ctx, nb, dev)
            r.md = md
            if not r.prepare(rows):
                raise RuntimeError(f"{b}: width {rows} refused")
            t1 = time.time()
            h, rr = r.forward(positions, hidden0, resid0)
            torch.cuda.synchronize()
            if rank == 0:
                print(f"{b} ctx {ctx}: first step {(time.time() - t1) * 1e3:.1f} ms", flush=True)
            if hasattr(r, "raise_on_errors"):
                r.raise_on_errors()
            outs[b] = (h.clone(), rr.clone())
            if os.environ.get("M3_OUT_SAVE") and rank == 0:  # determinism checks across runs
                torch.save({"h": h.cpu(), "r": rr.cpu()}, f"{os.environ['M3_OUT_SAVE']}_{b}_{ctx}.pt")
            if a.check:
                tabs[b] = sparse_tables(b, r, rows)
            if a.eager:
                for _ in range(a.eager):
                    r.forward(positions, hidden0, resid0)
                torch.cuda.synchronize()
                dist.barrier(group=group)
                res[b] = [0.0, 0.0, 0.0]
            else:
                res[b] = bench_steps(lambda: r.forward(positions, hidden0, resid0), group, a.steps)
            torch.cuda.synchronize()
            if a.timeline and rank == 0:
                from bench_layer_tp import timeline_summary, FLY_POINTS
                tl = r.tl_buf.cpu()
                if os.environ.get("M3_TL_SAVE"):
                    torch.save(tl, f"{os.environ['M3_TL_SAVE']}_{b}_{ctx}.pt")
                    if getattr(r, "hw_buf", None) is not None:
                        torch.save(r.hw_buf.cpu(), f"{os.environ['M3_TL_SAVE']}_{b}_{ctx}_hw.pt")
                for li in range(min(2, a.layers)):
                    if os.environ.get("M3_TL_TOP"):
                        sl = int(os.environ["M3_TL_TOP"])
                        t = tl[li].double()
                        t0 = t[:, 0][t[:, 0] > 0].min()
                        v = (t[:, sl] - t0) / 100.0
                        order = torch.argsort(v, descending=True)[:12].tolist()
                        print(f"  slot {sl} latest CTAs: " + " ".join(f"{c}:{v[c]:.1f}" for c in order)
                              + f" | earliest: " + " ".join(f"{c}:{v[c]:.1f}" for c in torch.argsort(v)[:6].tolist()))
                    print(f"  {b} ctx {ctx} timeline (last replay) layer {li} ({'selects' if li % 4 == 0 else 'reuses'}):\n"
                          + timeline_summary(tl, li, FLY_POINTS if b == "flydsl" else r.TL_NAMES), flush=True)
        if rank == 0:
            line = f"ctx={ctx} reqs={a.reqs} ql={a.ql} L={a.layers} steps={a.steps}:"
            for b, (med, lo, hi) in res.items():
                line += f"  {b} {med:8.1f} us/step ({med / a.layers:5.1f} us/layer, min {lo:.1f} max {hi:.1f})"
            if "flydsl" in res:
                for b in res:
                    if b != "flydsl":
                        line += f"  {b}/flydsl {res[b][0] / res['flydsl'][0]:.3f}"
                        if a.check:
                            line += (f"  nRMSE h {nrmse(outs[b][0], outs['flydsl'][0]):.2e}"
                                     f" r {nrmse(outs[b][1], outs['flydsl'][1]):.2e}")
                            if ctx + a.ql > 2048 and tabs.get(b) is not None and tabs.get("flydsl") is not None:
                                tb, tf = tabs[b], tabs["flydsl"]
                                line += f"  tables {'EQUAL' if torch.equal(tb, tf) else 'DIFFER'}"
                                if not torch.equal(tb, tf):
                                    bad = (tb != tf).any(1).nonzero().flatten().tolist()
                                    line += f" rows {bad} e.g. gluon {tb[bad[0]].tolist()} flydsl {tf[bad[0]].tolist()}"
            print(line, flush=True)
    dist.barrier(group=group)
    for r in runners.values():
        r.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
