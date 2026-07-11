"""Full decode perf scan for the CURRENTLY-INSTALLED triton version.
Grid: C in {16,32,64,128} x ctx in {1024,8192} x heads in {64/8 GQA, 8/1 MQA}.
Both impls: triton (3d attn + reduce, heuristic split) and gluon (2d, right-sized
split; S=1 => non-split, no reduce). Records time (us/iter), TFLOP/s, GB/s.
Writes scan_<major.minor.patch>.json. Run once per version (dir-swap between)."""
import sys, math, json
import triton
sys.path.insert(0, "/app/aiter/bench_gluon_ua")
import torch
import bench_ua as B

DEV, DT, HS, RCP = B.DEV, B.DT, B.HEAD_SIZE, B.RCP_LN2
VER_FULL = triton.__version__
VER = VER_FULL.split("+")[0]
TILE = 64
torch.manual_seed(0)

CS = [16, 32, 64, 128]
CTXS = [1024, 8192]
HEADS = [(64, 8), (8, 1)]
records = []


def scan_one(C, ctx, Hq, Hkv):
    nqpk = Hq // Hkv; scale = 1.0 / math.sqrt(HS); num_tiles = ctx // TILE
    q = torch.randn(C, Hq, HS, dtype=DT, device=DEV)
    k, v, bt = B.make_paged_kv(ctx, C, TILE, Hkv)
    cu = torch.arange(0, C + 1, dtype=torch.int32, device=DEV)
    seqk = torch.full((C,), ctx, dtype=torch.int32, device=DEV)
    flops, byts = B.decode_flops_bytes(C, ctx, Hq, Hkv)

    # ---- triton: 3d attn + reduce at the heuristic split ----
    attn, red = B.select_3d_config(HS, TILE, ctx, B.TARGET_PRGMS, C * Hkv, DT, DT, False, 1, 0)
    S = attn["NUM_SEGMENTS_PER_SEQ"]; a_nw = attn["num_warps"]; wpe = attn["waves_per_eu"]
    nstg = attn["num_stages"]; r_nw = red["num_warps"]
    seg = B.alloc_segm(C, Hq, S); ot = torch.empty_like(q)
    def tf():
        B.launch_tri_3d(q, k, v, cu, seqk, bt, scale, 16, 16 // nqpk, TILE, a_nw, S, wpe, nstg, seg)
        B.launch_reduce(ot, cu, seqk, bt, TILE, S, r_nw, 16 // nqpk, seg)
    rt = B.profile_kernels(tf)
    t_attn = B.pick(rt, "unified_attention_3d"); t_red = B.pick(rt, "reduce_segments")
    t_tot = t_attn + t_red

    # ---- gluon: 2d at the right-sized split (S=1 => non-split, no reduce) ----
    Sg = B.select_gluon_num_splits(C, Hkv, num_tiles)
    og = torch.empty_like(q)
    if Sg == 1:
        def gf():
            B.launch_glu_2d(q, k, v, og, cu, seqk, bt, scale, 16, TILE, 1, wpe,
                            NUM_SPLITS=1, ALL_DECODE=True, MFMA_DIM=16, NUM_BUFFERS=2)
        gf(); torch.cuda.synchronize()
        xc = (ot.float() - og.float()).abs().max().item()
        rg = B.profile_kernels(gf)
        g_attn = B.pick(rg, "unified_attention_2d"); g_red = 0.0
    else:
        so, sm, se = B.alloc_segm(C, Hq, Sg)
        # correctness pass (prescaled M); timed pass uses raw M -> identical reduce cost
        B.launch_glu_2d(q, k, v, og, cu, seqk, bt, scale, 16, TILE, 1, wpe,
                        NUM_SPLITS=Sg, ALL_DECODE=True, partials=(sm, se, so), MFMA_DIM=16, NUM_BUFFERS=2)
        B.launch_reduce(og, cu, seqk, bt, TILE, Sg, r_nw, 16 // nqpk, (so, sm * (RCP * scale), se))
        torch.cuda.synchronize()
        xc = (ot.float() - og.float()).abs().max().item()
        def gf():
            B.launch_glu_2d(q, k, v, og, cu, seqk, bt, scale, 16, TILE, 1, wpe,
                            NUM_SPLITS=Sg, ALL_DECODE=True, partials=(sm, se, so), MFMA_DIM=16, NUM_BUFFERS=2)
            B.launch_reduce(og, cu, seqk, bt, TILE, Sg, r_nw, 16 // nqpk, (so, sm, se))
        rg = B.profile_kernels(gf)
        g_attn = B.pick(rg, "unified_attention_2d"); g_red = B.pick(rg, "reduce_segments")
    g_tot = g_attn + g_red

    for impl, sp, tot, at, rd in [("triton", S, t_tot, t_attn, t_red),
                                  ("gluon", Sg, g_tot, g_attn, g_red)]:
        records.append(dict(ver=VER, ver_full=VER_FULL, C=C, ctx=ctx, Hq=Hq, Hkv=Hkv,
                            impl=impl, split=sp, time_us=tot, attn_us=at, red_us=rd,
                            tflops=B.tflops(flops, tot), gbps=B.gbps(byts, tot), xc=xc))
    print(f"C{C:>3} ctx{ctx} {Hq}/{Hkv}: "
          f"tri {t_tot:6.1f}us {B.gbps(byts,t_tot):5.0f}GB/s {B.tflops(flops,t_tot):4.0f}TF S{S:<3} | "
          f"glu {g_tot:6.1f}us {B.gbps(byts,g_tot):5.0f}GB/s {B.tflops(flops,g_tot):4.0f}TF S{Sg:<3} "
          f"{t_tot/g_tot:.2f}x xc{xc:.0e}")


print(f"=== triton {VER_FULL} ===")
for ctx in CTXS:
    for (Hq, Hkv) in HEADS:
        for C in CS:
            scan_one(C, ctx, Hq, Hkv)

path = f"/app/aiter/bench_gluon_ua/scan_{VER}.json"
json.dump(records, open(path, "w"), indent=0)
print("wrote", path)
