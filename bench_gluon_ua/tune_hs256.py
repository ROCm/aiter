"""Tune the gfx950 gluon unified-attention knobs at HEAD_SIZE=256 (bf16 + fp8).

HEAD_SIZE=256 is the first size the wrapper has never been calibrated for, and it is the
one where the LDS budget actually bites: the KV tile is [TILE_SIZE, HEAD_SIZE] per buffer,
so double-buffering costs twice what it does at 128. The wrapper's decode branch already
drops to NUM_BUFFERS=1 for HEAD_SIZE>=128, but its prefill branch hardcodes 2 -- whether
that still fits (and still wins) at 256 is the main question here.

Same staged shape as tune_hs64.py so the compile count stays sane:
  stage 1 - (BLOCK_M, MFMA_DIM, num_warps) tile at the shipped waves_per_eu
  stage 2 - waves_per_eu at the best tile
  stage 3 - NUM_BUFFERS at the best (tile, wpe)

Every config records LDS, VGPR/spills and a correctness check against an fp32 torch
reference, so a config that is fast because it is wrong (or that silently spills) is
visible rather than winning. Time is attention + reduce via torch.profiler filtered by
kernel name; decode uses the wrapper's own split-KV count.

    UA_HEAD_SIZE=256 python tune_hs256.py
"""
import sys, os, math, json, re
import torch, triton
sys.path.insert(0, "/app/aiter/bench_gluon_ua")
import bench_ua as B
from aiter.ops.triton.utils.types import e4m3_dtype
from aiter.ops.triton._gluon_kernels.gfx950.attention.unified_attention import (
    _select_num_splits)

DEV, HS = B.DEV, B.HEAD_SIZE
assert HS == 256, "run with UA_HEAD_SIZE=256"
TILE = int(os.environ.get("TILE", 64))
VER = triton.__version__.split("+")[0]
one = torch.ones(1, dtype=torch.float32, device=DEV)
D8 = (one, one.clone(), one.clone())
DTYPES = [("bf16", torch.bfloat16), ("fp8", e4m3_dtype)]
torch.manual_seed(0)

# Hq=64 with a GQA-8 and an MQA point, plus a small-batch decode where splits matter.
DEC_SHAPES = [(128, 8192, 64, 8), (8, 8192, 64, 8), (32, 1024, 8, 1)]
PRE_SHAPES = [(8, 1024, 64, 8), (1, 8192, 64, 8)]
# BLOCK_M must be a multiple of MFMA_DIM * num_warps.
DEC_TILES = {"bf16": [(16, 16, 1), (32, 32, 1), (32, 16, 2), (64, 32, 2)],
             "fp8": [(32, 32, 1), (64, 32, 2)]}      # 16x16 fp8 needs TILE>=128 (see study)
PRE_TILES = {"bf16": [(128, 32, 4), (64, 32, 2), (64, 16, 4), (128, 16, 8)],
             "fp8": [(128, 32, 4), (64, 32, 2)]}
WPES = [1, 2, 3, 4]
NBUFS = [1, 2]


def desc_of(dt):
    return D8 if dt.itemsize == 1 else (None, None, None)


def torch_ref(q, k, v, bt, ctx, Hq, Hkv, qlen, s=0):
    nqpk = Hq // Hkv
    blocks = bt[s, :ctx // TILE].long()
    kk = k[blocks].permute(2, 0, 1, 3).reshape(Hkv, -1, HS).float()[:, :ctx]
    vv = v[blocks].permute(2, 0, 1, 3).reshape(Hkv, -1, HS).float()[:, :ctx]
    qi = s * qlen + qlen - 1
    out = torch.empty(Hq, HS, dtype=torch.float32, device=DEV)
    for h in range(Hq):
        p = torch.softmax((kk[h // nqpk] @ q[qi, h].float()) * (1.0 / math.sqrt(HS)), dim=0)
        out[h] = p @ vv[h // nqpk]
    del kk, vv
    return out, qi


def asm_stats(ck):
    a = ck.asm["amdgcn"]
    g = lambda k: int((re.search(rf"\.{k}:\s*(\d+)", a) or [None, -1])[1])
    return g("vgpr_count"), g("vgpr_spill_count"), ck.metadata.shared


def run(phase, N, L, Hq, Hkv, dt, BM, mf, nw, nb, wpe):
    """One config -> dict(us, rel, lds, vgpr, spill, splits) or dict(error=...)."""
    nqpk = Hq // Hkv
    scale = 1.0 / math.sqrt(HS)
    qlen = 1 if phase == "decode" else L
    q = torch.randn(N * qlen, Hq, HS, dtype=torch.float32, device=DEV).to(dt)
    k, v, bt = B.make_paged_kv(L, N, TILE, Hkv, dtype=dt)
    cu = torch.arange(0, (N + 1) * qlen, qlen, dtype=torch.int32, device=DEV)
    seqk = torch.full((N,), L, dtype=torch.int32, device=DEV)
    out = torch.empty(N * qlen, Hq, HS, dtype=torch.bfloat16, device=DEV)
    desc = desc_of(dt)
    try:
        ref, qi = torch_ref(q, k, v, bt, L, Hq, Hkv, qlen)
        if phase == "decode":
            S = _select_num_splits(N, Hkv, L // TILE, nw)
            if S == 1:
                def f():
                    B.launch_glu_2d(q, k, v, out, cu, seqk, bt, scale, BM, TILE, nw, wpe,
                                    NUM_SPLITS=1, ALL_DECODE=True, MFMA_DIM=mf,
                                    NUM_BUFFERS=nb, descales=desc)
            else:
                so, sm, se = B.alloc_segm(N, Hq, S)
                def f():
                    B.launch_glu_2d(q, k, v, out, cu, seqk, bt, scale, BM, TILE, nw, wpe,
                                    NUM_SPLITS=S, ALL_DECODE=True, partials=(sm, se, so),
                                    MFMA_DIM=mf, NUM_BUFFERS=nb, descales=desc)
                    B.launch_reduce(out, cu, seqk, bt, TILE, S, 2, max(1, BM // nqpk),
                                    (so, sm, se))
        else:
            S = 1
            def f():
                B.launch_glu_2d(q, k, v, out, cu, seqk, bt, scale, BM, TILE, nw, wpe,
                                MFMA_DIM=mf, NUM_BUFFERS=nb, descales=desc)
        ck = f()
        torch.cuda.synchronize()
        rel = ((ref - out[qi].float()).abs().mean() / ref.abs().mean()).item()
        us = B.pick(B.profile_kernels(f), "unified_attention", "reduce_segments")
        vgpr, spill, lds = asm_stats(ck) if ck is not None else (-1, -1, -1)
        r = dict(us=us, rel=rel, ok=bool(rel < 0.06), lds=lds, vgpr=vgpr, spill=spill,
                 splits=S)
    except Exception as exc:
        r = dict(error=f"{type(exc).__name__}: {str(exc)[:90]}")
    del q, k, v, bt, out
    torch.cuda.empty_cache()
    return r


def sweep(phase, shapes, dt, cfgs, label):
    """Score each config by geomean time over the shapes; skip any that fails a shape."""
    scored = []
    for (BM, mf, nw, nb, wpe) in cfgs:
        times, meta, bad = [], None, None
        for (N, L, Hq, Hkv) in shapes:
            r = run(phase, N, L, Hq, Hkv, dt, BM, mf, nw, nb, wpe)
            if "error" in r:
                bad = r["error"]
                break
            if not r["ok"]:
                bad = f"WRONG rel={r['rel']:.1%}"
                break
            times.append(r["us"])
            meta = r
        if bad:
            print(f"    BM{BM:<3d} mfma{mf:<2d} nw{nw} nb{nb} wpe{wpe}: {bad}", flush=True)
            continue
        geo = math.exp(sum(math.log(t) for t in times) / len(times))
        scored.append((geo, BM, mf, nw, nb, wpe, meta))
        print(f"    BM{BM:<3d} mfma{mf:<2d} nw{nw} nb{nb} wpe{wpe}: {geo:8.1f}us  "
              f"LDS {meta['lds']//1024:3d}KB  VGPR {meta['vgpr']:3d}"
              f"{' SPILL ' + str(meta['spill']) if meta['spill'] > 0 else ''}"
              f"  rel {meta['rel']:.2%}", flush=True)
    scored.sort()
    return scored


results = {}
print(f"=== HEAD_SIZE=256 tuning | triton {VER} | TILE={TILE} | {B.CU} CUs ===", flush=True)
for phase, shapes, tiles in (("decode", DEC_SHAPES, DEC_TILES),
                             ("prefill", PRE_SHAPES, PRE_TILES)):
    for dname, dt in DTYPES:
        print(f"\n--- {phase} {dname} ---", flush=True)
        base_wpe = 2
        base_nb = 1 if phase == "decode" else 2
        print(f"  stage 1: tile (wpe={base_wpe}, nb={base_nb})", flush=True)
        s1 = sweep(phase, shapes, dt,
                   [(bm, mf, nw, base_nb, base_wpe) for (bm, mf, nw) in tiles[dname]],
                   "tile")
        if not s1:
            print("    no working tile -- skipping", flush=True)
            continue
        _, BM, mf, nw, _, _, _ = s1[0]
        print(f"  stage 2: waves_per_eu (BM{BM} mfma{mf} nw{nw} nb{base_nb})", flush=True)
        s2 = sweep(phase, shapes, dt, [(BM, mf, nw, base_nb, w) for w in WPES], "wpe")
        wpe = s2[0][5] if s2 else base_wpe
        print(f"  stage 3: NUM_BUFFERS (BM{BM} mfma{mf} nw{nw} wpe{wpe})", flush=True)
        s3 = sweep(phase, shapes, dt, [(BM, mf, nw, nb, wpe) for nb in NBUFS], "nbuf")
        best = (s3 or s2 or s1)[0]
        results[f"{phase}_{dname}"] = dict(
            block_m=best[1], mfma_dim=best[2], num_warps=best[3], num_buffers=best[4],
            waves_per_eu=best[5], geo_us=best[0], lds=best[6]["lds"],
            vgpr=best[6]["vgpr"], spill=best[6]["spill"])
        print(f"  => BEST {phase} {dname}: BLOCK_M={best[1]} MFMA_DIM={best[2]} "
              f"num_warps={best[3]} NUM_BUFFERS={best[4]} waves_per_eu={best[5]} "
              f"({best[0]:.1f}us, LDS {best[6]['lds']//1024}KB)", flush=True)

path = os.environ.get("TUNE256_OUT",
                      f"/app/aiter/bench_gluon_ua/matrix_scan/tune_hs256_{VER}.json")
json.dump(results, open(path, "w"), indent=1)
print("\nwrote", path)
for k_, v_ in results.items():
    print(f"  {k_:14s} {v_}")
