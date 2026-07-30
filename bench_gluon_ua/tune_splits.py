"""Measure the best decode split-KV count, and score candidate heuristics against it.

Un-split decode launches num_seqs * num_kv_heads workgroups, which under-fills the GPU at
low batch. Splitting the KV range adds workgroups but costs a second (reduce) kernel and
partial-buffer traffic, so there is an optimum per shape. This sweeps NUM_SPLITS through
the gluon wrapper's `num_splits=` override, finds the measured optimum per cell, then
reports how close each candidate formula gets.

Timing is attention + reduce, from torch.profiler filtered by kernel name (an L2 flush runs
between iterations and must not be counted -- it is ~93 us, several times the smaller
cells).

    UA_HEAD_SIZE={64,128} python tune_splits.py
"""
import sys, os, math, json, itertools
import torch, triton
sys.path.insert(0, "/app/aiter/bench_gluon_ua")
import bench_ua as B
from aiter.ops.triton.utils.types import e4m3_dtype
from aiter.ops.triton._gluon_kernels.gfx950.attention.unified_attention import (
    unified_attention as glu_ua, _select_num_splits)

DEV, HS, CU = B.DEV, B.HEAD_SIZE, B.CU
VER = triton.__version__.split("+")[0]
TILE = 64
one = torch.ones(1, dtype=torch.float32, device=DEV)
torch.manual_seed(0)
SPLITS = [1, 2, 4, 8, 16, 32, 64, 128]
HEADS = [(8, 1), (64, 8), (8, 8), (64, 64)]
CS = [int(x) for x in os.environ.get("CS", "1,8,32,64,128").split(",")]
CTXS = [int(x) for x in os.environ.get("CTXS", "1024,8192").split(",")]
DTYPES = [("bf16", torch.bfloat16), ("fp8", e4m3_dtype)]
MAX_KV_GB = float(os.environ.get("MAX_KV_GB", 40))


def kernel_us(fn):
    """attention + reduce only; the 512 MB L2 flush is excluded by name filtering."""
    res = B.profile_kernels(fn)
    return B.pick(res, "unified_attention", "reduce_segments")


def sweep(C, ctx, Hq, Hkv, dname, dt):
    kv_gb = 2 * C * ctx * Hkv * HS * dt.itemsize / 1e9
    if kv_gb > MAX_KV_GB:
        return None
    scale = 1.0 / math.sqrt(HS)
    q = torch.randn(C, Hq, HS, dtype=torch.float32, device=DEV).to(dt)
    k, v, bt = B.make_paged_kv(ctx, C, TILE, Hkv, dtype=dt)
    cu = torch.arange(0, C + 1, dtype=torch.int32, device=DEV)
    seqk = torch.full((C,), ctx, dtype=torch.int32, device=DEV)
    out = torch.empty(C, Hq, HS, dtype=torch.bfloat16, device=DEV)
    d = (one, one.clone(), one.clone()) if dt.itemsize == 1 else (None, None, None)
    num_tiles = ctx // TILE
    times = {}
    for S in SPLITS:
        if S > num_tiles:
            break
        try:
            fn = lambda: glu_ua(q, k, v, out, cu, seqk, 1, ctx, scale, True, (-1, -1), bt,
                                0, d[0], d[1], d[2], None, num_splits=S)
            fn(); torch.cuda.synchronize()
            times[S] = kernel_us(fn)
        except Exception:
            pass
    del q, k, v, bt, out
    torch.cuda.empty_cache()
    return times


recs = []
print(f"=== split-KV tuning | triton {VER} | HEAD_SIZE={HS} | {CU} CUs ===", flush=True)
for dname, dt in DTYPES:
    for (Hq, Hkv) in HEADS:
        print(f"\n--- {dname} {Hq}/{Hkv} ---", flush=True)
        for ctx in CTXS:
            for C in CS:
                t = sweep(C, ctx, Hq, Hkv, dname, dt)
                if not t:
                    continue
                best_S = min(t, key=t.get)
                nw = 1 if dname == "bf16" else 1     # decode num_warps (nqpk<=mfma_dim)
                wpe = 2 if HS >= 128 else 1
                heur = _select_num_splits(C, Hkv, ctx // TILE, nw, wpe)
                rec = dict(ver=VER, head_size=HS, dtype=dname, C=C, ctx=ctx, Hq=Hq, Hkv=Hkv,
                           base_wgs=C * Hkv, num_tiles=ctx // TILE,
                           times={str(s): v for s, v in t.items()},
                           best_S=best_S, best_us=t[best_S], heur_S=heur,
                           heur_us=t.get(heur), s1_us=t.get(1))
                recs.append(rec)
                loss = (t[heur] / t[best_S] - 1) * 100 if heur in t else float("nan")
                grid = " ".join(f"{s}:{v:.0f}" for s, v in sorted(t.items()))
                print(f"  C{C:<4d} ctx{ctx:<5d} wgs{C*Hkv:<6d}: best S={best_S:<3d} "
                      f"{t[best_S]:8.1f}us | heur S={heur:<3d} {loss:+5.1f}% | S1 "
                      f"{t[1]:8.1f}us ({t[1]/t[best_S]:.2f}x) | {grid}", flush=True)

path = os.environ.get("TUNE_OUT",
                      f"/app/aiter/bench_gluon_ua/matrix_scan/splits_hs{HS}_{VER}.json")
os.makedirs(os.path.dirname(path), exist_ok=True)
json.dump(recs, open(path, "w"), indent=0)
ok = [r for r in recs if r["heur_us"]]
if ok:
    losses = [(r["heur_us"] / r["best_us"] - 1) * 100 for r in ok]
    print(f"\ncurrent heuristic: mean {sum(losses)/len(losses):+.1f}% vs best, "
          f"worst {max(losses):+.1f}% ({len(ok)} cells)")
print("wrote", path)
