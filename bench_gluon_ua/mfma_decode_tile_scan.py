"""Per-shape decode comparison of the two bf16 MFMA shapes (16x16x32 vs 32x32x16).

The MFMA shape study sweeps K_WIDTH on one large decode shape; that shape turns out to be
bandwidth-saturated, where the two tiles tie. This walks the decode shape range instead, which
is where the M-lane-utilisation difference (50% vs 25% at NUM_QUERIES_PER_KV=8) actually shows.

    UA_HEAD_SIZE={64,128} python mfma_decode_tile_scan.py
"""
import sys, os, math, json
import torch, triton
sys.path.insert(0, "/app/aiter/bench_gluon_ua")
import bench_ua as B

DEV, HS, RCP = B.DEV, B.HEAD_SIZE, B.RCP_LN2
TILE = 64
VER = triton.__version__.split("+")[0]
WPE = 1 if HS < 128 else 2          # tuned per head size; see tune_hs64.py
SHAPES = [(16, 1024, 64, 8), (32, 1024, 8, 1), (64, 1024, 8, 1), (128, 1024, 64, 8),
          (16, 8192, 8, 1), (128, 8192, 64, 8), (128, 8192, 8, 1)]
torch.manual_seed(0)


def run(C, ctx, Hq, Hkv, BM, mf):
    nqpk = Hq // Hkv; scale = 1.0 / math.sqrt(HS)
    q = torch.randn(C, Hq, HS, dtype=torch.bfloat16, device=DEV)
    k, v, bt = B.make_paged_kv(ctx, C, TILE, Hkv, dtype=torch.bfloat16)
    cu = torch.arange(0, C + 1, dtype=torch.int32, device=DEV)
    seqk = torch.full((C,), ctx, dtype=torch.int32, device=DEV)
    o = torch.empty(C, Hq, HS, dtype=torch.bfloat16, device=DEV)
    Sg = B.select_gluon_num_splits(C, Hkv, ctx // TILE)
    if Sg == 1:
        def f():
            B.launch_glu_2d(q, k, v, o, cu, seqk, bt, scale, BM, TILE, 1, WPE, NUM_SPLITS=1,
                            ALL_DECODE=True, MFMA_DIM=mf, NUM_BUFFERS=2)
        f(); torch.cuda.synchronize()
        t = B.pick(B.profile_kernels(f), "unified_attention_2d")
    else:
        so, sm, se = B.alloc_segm(C, Hq, Sg)
        def f():
            B.launch_glu_2d(q, k, v, o, cu, seqk, bt, scale, BM, TILE, 1, WPE, NUM_SPLITS=Sg,
                            ALL_DECODE=True, partials=(sm, se, so), MFMA_DIM=mf, NUM_BUFFERS=2)
            B.launch_reduce(o, cu, seqk, bt, TILE, Sg, 2, 16 // nqpk, (so, sm, se))
        f(); torch.cuda.synchronize()
        r = B.profile_kernels(f)
        t = B.pick(r, "unified_attention_2d") + B.pick(r, "reduce_segments")
    del q, k, v, bt, o; torch.cuda.empty_cache()
    return t


recs = []
print(f"=== decode tile scan | triton {VER} | HEAD_SIZE={HS} | wpe={WPE} ===", flush=True)
for (C, ctx, Hq, Hkv) in SHAPES:
    t16, t32 = run(C, ctx, Hq, Hkv, 16, 16), run(C, ctx, Hq, Hkv, 32, 32)
    recs.append(dict(ver=VER, head_size=HS, C=C, ctx=ctx, Hq=Hq, Hkv=Hkv,
                     label=f"C{C} ctx{ctx} {Hq}/{Hkv}", t_16x16=t16, t_32x32=t32))
    print(f"  C{C:<3d} ctx{ctx:<5d} {Hq}/{Hkv}: 16x16 {t16:8.1f}  32x32 {t32:8.1f}  "
          f"16x16/32x32 = {t16 / t32:.2f}", flush=True)

out = os.environ.get("DEC_TILE_OUT",
                     f"/app/aiter/bench_gluon_ua/mfma_shape_study_7_28/dectile_hs{HS}_{VER}.json")
json.dump(recs, open(out, "w"), indent=0)
print("wrote", out)
