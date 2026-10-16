"""MiniMax-M3 mono microbenchmark, TP4: a decode step of L synthetic sparse layers (K1 fused
into K4, one launch a layer) replayed as a HIP graph; us per layer (median of reps, slowest
rank) and the 57-layer projection. Backends: flydsl (ATOM's kernels, vendored), gluon.

    torchrun --nproc_per_node 4 bench/bench_layer_tp.py [--backends flydsl,gluon]
        [--rows 1,2,4,8,16] [--ctx 1536] [--layers 16] [--check]

--check: each backend's step output vs FlyDSL's on the same inputs (nRMSE)."""
import argparse
import os
import time
import sys

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from m3_inputs import StandaloneRunner, make_layers, make_step  # noqa: E402

REPS = 30
SPARSE_LAYERS = 57


def bench(fn, group, n_inner=1):
    fn()
    torch.cuda.synchronize()
    dist.barrier(group=group)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(n_inner):
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
        ts.append(a.elapsed_time(b) * 1000 / n_inner)
    ts.sort()
    t = torch.tensor([ts[len(ts) // 2]])
    dist.all_reduce(t, op=dist.ReduceOp.MAX, group=group)
    del g
    return t.item()


def nrmse(a, b):
    a, b = a.float(), b.float()
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


FLY_POINTS = {  # FlyDSL K4 stamp points (fuse_k1: K1's on 27..31)
    0: "begin", 27: "k1 norm", 28: "k1 x staged", 29: "k1 qkv gemv", 30: "k1 head got", 31: "k1 done",
    18: "sel a", 20: "sel b", 19: "sel done", 17: "split blocks", 12: "split q", 13: "split qk", 14: "split softmax",
    15: "split pv", 1: "split end", 3: "o got parts", 4: "o end", 22: "router got a", 24: "router ss",
    25: "router xn", 26: "router 26", 23: "router 23", 5: "router end", 21: "route got scores", 2: "routed",
    6: "route sync", 7: "ug first load", 9: "ug mfma", 8: "ug end", 11: "down end", 10: "mids staged",
}


def timeline_summary(tl, layer, points):
    """Per stamp point: CTAs reaching it, median / max us after the layer's first CTA start."""
    t = tl[layer].double()
    t0 = t[:, 0][t[:, 0] > 0].min()
    rows = []
    for k in range(t.shape[1]):
        v = t[:, k]
        v = v[v > 0]
        if len(v):
            us = (v - t0) / 100.0  # 100 MHz
            rows.append((us.median().item(), us.max().item(), len(v), k))
    rows.sort()
    return "\n".join(f"    {med:7.2f} {mx:7.2f} us  n={n:3d}  p{k:2d} {points.get(k, '')}" for med, mx, n, k in rows)


def make_runner(backend, layers, rank, group, dev, debug, timeline=False):
    if backend == "flydsl":
        return StandaloneRunner(layers, rank, group, dev, debug=debug, timeline=timeline)
    if backend == "gluon":
        from vllm.models.minimax_m3.amd.mono.gluon.runner import GluonStandalone

        return GluonStandalone(layers, rank, group, dev, debug=debug, timeline=timeline)
    raise ValueError(backend)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backends", default="flydsl")
    ap.add_argument("--rows", default="1,2,4,8,16")
    ap.add_argument("--ctx", default="1536")
    ap.add_argument("--layers", type=int, default=16)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--timeline", action="store_true")
    a = ap.parse_args()
    rank, tp = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)
    dist.init_process_group("gloo")
    group = dist.new_group(backend="gloo")
    rows_list = [int(x) for x in a.rows.split(",")]
    ctxs = [int(x) for x in a.ctx.split(",")]
    nb = max(rows_list) * ((max(ctxs) + 127) // 128) + 8
    layers = make_layers(a.layers, nb, rank, dev)
    backends = a.backends.split(",")
    runners = {b: make_runner(b, layers, rank, group, dev, a.debug, a.timeline) for b in backends}
    hidden0 = (torch.randn(16, 6144, generator=torch.Generator(device=dev).manual_seed(5), device=dev)
               ).to(torch.bfloat16)
    resid0 = (torch.randn(16, 6144, generator=torch.Generator(device=dev).manual_seed(6), device=dev) * 4
              ).to(torch.bfloat16)
    for ctx in ctxs:
        for rows in rows_list:
            md, positions = make_step(rows, ctx, nb, dev)
            res = {}
            outs = {}
            for b, r in runners.items():
                r.md = md
                if not r.prepare(rows):
                    raise RuntimeError(f"{b}: width {rows} refused")
                t_first = time.time()
                h, rr = r.forward(positions, hidden0[:rows], resid0[:rows])
                t_launch = time.time() - t_first
                torch.cuda.synchronize()
                print(f"rank {rank} {b} rows {rows}: first forward launch {t_launch * 1e3:.1f} ms, "
                      f"done {(time.time() - t_first) * 1e3:.1f} ms", flush=True)
                if a.debug:
                    r.raise_on_given_up_waits(rows)
                if hasattr(r, "raise_on_errors"):
                    try:
                        r.raise_on_errors()
                    except RuntimeError as e:
                        print(f"rank {rank} {b} rows {rows}: first forward: {e}", flush=True)
                        r.errors.zero_()
                outs[b] = (h.clone(), rr.clone())
                res[b] = bench(lambda: r.forward(positions, hidden0[:rows], resid0[:rows]), group) / a.layers
                if hasattr(r, "raise_on_errors"):
                    torch.cuda.synchronize()
                    try:
                        r.raise_on_errors()
                    except RuntimeError as e:
                        print(f"rank {rank} {b} rows {rows}: during the timed replays: {e}", flush=True)
                        r.errors.zero_()
            if rank == 0:
                line = f"rows={rows:2d} ctx={ctx:5d} L={a.layers}:"
                for b, t in res.items():
                    line += f"  {b} {t:6.1f} us/layer ({t * SPARSE_LAYERS / 1000:5.2f} ms/57)"
                if "flydsl" in res:
                    for b in res:
                        if b != "flydsl":
                            line += f"  {b}/flydsl {res[b] / res['flydsl']:.2f}"
                if a.check and "flydsl" in outs:
                    for b in outs:
                        if b != "flydsl":
                            line += (f"  nRMSE h {nrmse(outs[b][0], outs['flydsl'][0]):.2e}"
                                     f" r {nrmse(outs[b][1], outs['flydsl'][1]):.2e}")
                print(line, flush=True)
            if a.timeline:
                for b, r in runners.items():
                    tl = r.tl_buf.cpu()
                    os.makedirs("bench/logs", exist_ok=True)
                    torch.save(tl, f"bench/logs/tl_{b}_r{rank}_rows{rows}_ctx{ctx}.pt")
                    if rank == 0:
                        pts = FLY_POINTS if b == "flydsl" else getattr(r, "TL_NAMES", {})
                        for li in (0, 1):
                            print(f"  {b} timeline layer {li} ({'selects' if li % 4 == 0 else 'reuses'}):\n"
                                  + timeline_summary(tl, li, pts), flush=True)
    dist.barrier(group=group)
    for r in runners.values():
        r.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
