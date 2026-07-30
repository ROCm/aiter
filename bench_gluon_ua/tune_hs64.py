"""Tune the gfx950 gluon unified-attention knobs at HEAD_SIZE=64 (bf16 + fp8).

Staged sweep so the compile count stays sane:
  stage 1 - waves_per_eu at the default tile
  stage 2 - (BLOCK_M, MFMA_DIM, num_warps) tile at the best wpe
  stage 3 - NUM_BUFFERS at the best tile (decode only; the single-buffer path is decode-only)

Prints a ranked table per (phase, dtype) and the winning config. Run per triton version:
    UA_HEAD_SIZE=64 python tune_hs64.py
"""
import sys, math, json, os
import torch, triton
sys.path.insert(0, "/app/aiter/bench_gluon_ua")
import bench_ua as B
from aiter.ops.triton.utils.types import e4m3_dtype

DEV, HS, RCP = B.DEV, B.HEAD_SIZE, B.RCP_LN2
assert HS == 64, "run with UA_HEAD_SIZE=64"
TILE = 64
VER = triton.__version__.split("+")[0]
one = torch.ones(1, dtype=torch.float32, device=DEV)
D8 = (one, one.clone(), one.clone())
DTYPES = [("bf16", torch.bfloat16), ("fp8", e4m3_dtype)]
torch.manual_seed(0)

DEC_SHAPES = [(128, 8192, 64, 8), (32, 1024, 8, 1)]
PRE_SHAPES = [(8, 1024, 64, 8), (1, 8192, 64, 8)]
# BLOCK_M must equal MFMA_DIM * num_warps.
DEC_TILES = {"bf16": [(16, 16, 1), (32, 32, 1), (32, 16, 2), (64, 32, 2)],
             "fp8": [(32, 32, 1), (64, 32, 2)]}          # 16x16 fp8 does not lower
PRE_TILES = {"bf16": [(128, 32, 4), (64, 32, 2), (64, 16, 4), (128, 16, 8)],
             "fp8": [(128, 32, 4), (64, 32, 2)]}
cache = {}


def bits(dt):
    return ((1, D8, 16) if dt.itemsize == 1 else (2, (None, None, None), 8))


def decode_time(C, ctx, Hq, Hkv, dt, BM, mf, nw, nb, wpe):
    """gluon decode time (attn + reduce) for one config; None if it does not compile."""
    key = ("d", C, ctx, Hq, Hkv, dt, BM, mf, nw, nb, wpe)
    if key in cache:
        return cache[key]
    nqpk = Hq // Hkv; scale = 1.0 / math.sqrt(HS); e, desc, _ = bits(dt)
    q = torch.randn(C, Hq, HS, dtype=torch.float32, device=DEV).to(dt)
    k, v, bt = B.make_paged_kv(ctx, C, TILE, Hkv, dtype=dt)
    cu = torch.arange(0, C + 1, dtype=torch.int32, device=DEV)
    seqk = torch.full((C,), ctx, dtype=torch.int32, device=DEV)
    og = torch.empty(C, Hq, HS, dtype=torch.bfloat16, device=DEV)
    Sg = B.select_gluon_num_splits(C, Hkv, ctx // TILE)
    try:
        if Sg == 1:
            def f():
                B.launch_glu_2d(q, k, v, og, cu, seqk, bt, scale, BM, TILE, nw, wpe,
                                NUM_SPLITS=1, ALL_DECODE=True, MFMA_DIM=mf,
                                NUM_BUFFERS=nb, descales=desc)
            f(); torch.cuda.synchronize()
            t = B.pick(B.profile_kernels(f), "unified_attention_2d")
        else:
            so, sm, se = B.alloc_segm(C, Hq, Sg)
            def f():
                B.launch_glu_2d(q, k, v, og, cu, seqk, bt, scale, BM, TILE, nw, wpe,
                                NUM_SPLITS=Sg, ALL_DECODE=True, partials=(sm, se, so),
                                MFMA_DIM=mf, NUM_BUFFERS=nb, descales=desc)
                B.launch_reduce(og, cu, seqk, bt, TILE, Sg, 2, 16 // nqpk, (so, sm, se))
            f(); torch.cuda.synchronize()
            r = B.profile_kernels(f)
            t = B.pick(r, "unified_attention_2d") + B.pick(r, "reduce_segments")
    except Exception:
        t = None
    del q, k, v, bt, og; torch.cuda.empty_cache()
    cache[key] = t
    return t


def prefill_time(bs, N, Hq, Hkv, dt, BM, mf, nw, nb, wpe):
    key = ("p", bs, N, Hq, Hkv, dt, BM, mf, nw, nb, wpe)
    if key in cache:
        return cache[key]
    scale = 1.0 / math.sqrt(HS); e, desc, _ = bits(dt); nt = bs * N
    q = torch.randn(nt, Hq, HS, dtype=torch.float32, device=DEV).to(dt)
    k, v, bt = B.make_paged_kv(N, bs, TILE, Hkv, dtype=dt)
    cu = torch.arange(0, (bs + 1) * N, N, dtype=torch.int32, device=DEV)
    seqk = torch.full((bs,), N, dtype=torch.int32, device=DEV)
    og = torch.empty(nt, Hq, HS, dtype=torch.bfloat16, device=DEV)
    try:
        def f():
            B.launch_glu_2d(q, k, v, og, cu, seqk, bt, scale, BM, TILE, nw, wpe,
                            MFMA_DIM=mf, NUM_BUFFERS=nb, descales=desc)
        f(); torch.cuda.synchronize()
        t = B.pick(B.profile_kernels(f), "unified_attention")
    except Exception:
        t = None
    del q, k, v, bt, og; torch.cuda.empty_cache()
    cache[key] = t
    return t


def score(phase, dt, tile, nb, wpe):
    """geomean time over the phase's shapes; inf if any shape fails."""
    BM, mf, nw = tile
    shapes = DEC_SHAPES if phase == "decode" else PRE_SHAPES
    fn = decode_time if phase == "decode" else prefill_time
    ts = [fn(*sh, dt, BM, mf, nw, nb, wpe) for sh in shapes]
    if any(t is None or t <= 0 for t in ts):
        return float("inf"), ts
    g = math.exp(sum(math.log(t) for t in ts) / len(ts))
    return g, ts


results = {}
print(f"=== tuning HEAD_SIZE=64 on triton {VER} | {B.CU} CUs ===", flush=True)
for phase in ("decode", "prefill"):
    tiles_by_dt = DEC_TILES if phase == "decode" else PRE_TILES
    for dname, dt in DTYPES:
        tiles = tiles_by_dt[dname]
        base_tile = tiles[0]
        base_nb = 2
        # stage 1: waves_per_eu at the default tile
        s1 = [(score(phase, dt, base_tile, base_nb, w)[0], w) for w in (1, 2, 3, 4)]
        s1 = [(g, w) for g, w in s1 if g < float("inf")]
        best_wpe = min(s1)[1]
        print(f"\n[{phase} {dname}] stage1 wpe @ tile{base_tile} nb{base_nb}: "
              + "  ".join(f"wpe{w}:{g:.1f}" for g, w in sorted(s1, key=lambda x: x[1]))
              + f"  -> best wpe={best_wpe}", flush=True)
        # stage 2: tile at the best wpe
        s2 = [(score(phase, dt, t, base_nb, best_wpe)[0], t) for t in tiles]
        s2 = [(g, t) for g, t in s2 if g < float("inf")]
        best_tile = min(s2)[1]
        print(f"[{phase} {dname}] stage2 tile @ wpe{best_wpe}: "
              + "  ".join(f"BM{t[0]}/mfma{t[1]}/nw{t[2]}:{g:.1f}" for g, t in s2)
              + f"  -> best {best_tile}", flush=True)
        # stage 3: NUM_BUFFERS (decode only)
        best_nb = base_nb
        if phase == "decode":
            s3 = [(score(phase, dt, best_tile, nb, best_wpe)[0], nb) for nb in (1, 2)]
            s3 = [(g, nb) for g, nb in s3 if g < float("inf")]
            best_nb = min(s3)[1]
            print(f"[{phase} {dname}] stage3 nb @ tile{best_tile} wpe{best_wpe}: "
                  + "  ".join(f"nb{nb}:{g:.1f}" for g, nb in sorted(s3, key=lambda x: x[1]))
                  + f"  -> best nb={best_nb}", flush=True)
        # re-check wpe at the winning tile/nb, in case the tile changed the optimum
        s4 = [(score(phase, dt, best_tile, best_nb, w)[0], w) for w in (1, 2, 3, 4)]
        s4 = [(g, w) for g, w in s4 if g < float("inf")]
        final_wpe = min(s4)[1]
        g_final, ts = score(phase, dt, best_tile, best_nb, final_wpe)
        print(f"[{phase} {dname}] recheck wpe @ tile{best_tile} nb{best_nb}: "
              + "  ".join(f"wpe{w}:{g:.1f}" for g, w in sorted(s4, key=lambda x: x[1]))
              + f"  -> FINAL BM{best_tile[0]}/mfma{best_tile[1]}/nw{best_tile[2]} "
                f"nb{best_nb} wpe{final_wpe}  geomean {g_final:.1f}us", flush=True)
        results[f"{phase}/{dname}"] = dict(
            block_m=best_tile[0], mfma_dim=best_tile[1], num_warps=best_tile[2],
            num_buffers=best_nb, waves_per_eu=final_wpe, geomean_us=g_final,
            per_shape_us=ts)

out = os.environ.get("TUNE_OUT", f"/app/aiter/bench_gluon_ua/tune_hs64_{VER}.json")
json.dump(dict(ver=VER, head_size=HS, results=results), open(out, "w"), indent=1)
print("\nwrote", out)
for k, r in results.items():
    print(f"  {k:14s}: BM{r['block_m']} mfma{r['mfma_dim']} nw{r['num_warps']} "
          f"nb{r['num_buffers']} wpe{r['waves_per_eu']}  ({r['geomean_us']:.1f}us)")
