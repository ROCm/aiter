"""Decode: MFMA 16x16/BLOCK_M=16 (matches Triton BLOCK_M) vs 32x32/BLOCK_M=32.
Grid order already matched (seq-fastest via ALL_DECODE). Triton = reference."""
import sys, math
sys.path.insert(0, "/app/aiter/bench_gluon_ua")
import torch
import bench_ua as B

DEV, DT, HS, RCP = B.DEV, B.DT, B.HEAD_SIZE, B.RCP_LN2
rows = []


def run(C, ctx, Hq, Hkv):
    nqpk = Hq // Hkv; scale = 1.0 / math.sqrt(HS); TILE = 64
    attn, red = B.select_3d_config(HS, 64, ctx, B.TARGET_PRGMS, C * Hkv, DT, DT, False, 1, 0)
    S, a_nw, wpe, nstg, r_nw = (attn["NUM_SEGMENTS_PER_SEQ"], attn["num_warps"],
                               attn["waves_per_eu"], attn["num_stages"], red["num_warps"])
    q = torch.randn(C, Hq, HS, dtype=DT, device=DEV)
    k, v, bt = B.make_paged_kv(ctx, C, TILE, Hkv)
    cu = torch.arange(0, C + 1, dtype=torch.int32, device=DEV)
    seqk = torch.full((C,), ctx, dtype=torch.int32, device=DEV)
    _, byts = B.decode_flops_bytes(C, ctx, Hq, Hkv)

    segm = B.alloc_segm(C, Hq, S); ot = torch.empty_like(q)
    def tf():
        B.launch_tri_3d(q, k, v, cu, seqk, bt, scale, 16, 16 // nqpk, TILE, a_nw, S, wpe, nstg, segm)
        B.launch_reduce(ot, cu, seqk, bt, TILE, S, r_nw, 16 // nqpk, segm)
    tt = B.pick(B.profile_kernels(tf), "unified_attention_3d") + B.pick(B.profile_kernels(tf), "reduce_segments")

    res = {}
    for mf, bm in [(32, 32), (16, 16)]:
        so, sm, se = B.alloc_segm(C, Hq, S); og = torch.empty_like(q)
        B.launch_glu_2d(q, k, v, og, cu, seqk, bt, scale, bm, TILE, 1, wpe,
                        NUM_SPLITS=S, ALL_DECODE=True, partials=(sm, se, so), MFMA_DIM=mf)
        B.launch_reduce(og, cu, seqk, bt, TILE, S, r_nw, bm // nqpk, (so, sm * (RCP * scale), se))
        torch.cuda.synchronize()
        xc = (ot.float() - og.float()).abs().max().item()
        def gf():
            B.launch_glu_2d(q, k, v, og, cu, seqk, bt, scale, bm, TILE, 1, wpe,
                            NUM_SPLITS=S, ALL_DECODE=True, partials=(sm, se, so), MFMA_DIM=mf)
            B.launch_reduce(og, cu, seqk, bt, TILE, S, r_nw, bm // nqpk, (so, sm, se))
        gt = B.pick(B.profile_kernels(gf), "unified_attention_2d") + B.pick(B.profile_kernels(gf), "reduce_segments")
        res[mf] = (gt, B.gbps(byts, gt), xc)
    r = dict(shape=f"C{C} ctx{ctx} {Hq}/{Hkv}", S=S, tt=tt, tgb=B.gbps(byts, tt),
             m32=res[32], m16=res[16])
    rows.append(r)
    print(f"{r['shape']:18s} S{S:<3d} tri {tt:6.1f}us {r['tgb']:5.0f} | "
          f"MFMA32/BM32 {res[32][0]:6.1f}us {res[32][1]:5.0f} ({tt/res[32][0]:.2f}x) | "
          f"MFMA16/BM16 {res[16][0]:6.1f}us {res[16][1]:5.0f} ({tt/res[16][0]:.2f}x) xc{res[16][2]:.0e}")


for sh in [(16, 1024, 64, 8), (64, 8192, 64, 8), (128, 8192, 64, 8),
           (32, 8192, 8, 1), (64, 8192, 8, 1), (16, 8192, 8, 1), (128, 8192, 8, 1)]:
    run(*sh)

L = ["# Decode: 16x16/BLOCK_M=16 vs 32x32/BLOCK_M=32 (grid matched, Triton=ref)\n",
     "| shape | S | Triton GB/s | 32×32 GB/s (spd) | 16×16 GB/s (spd) | xcheck |",
     "|---|--:|--:|--:|--:|--:|"]
for r in rows:
    L.append(f"| {r['shape']} | {r['S']} | {r['tgb']:.0f} | {r['m32'][1]:.0f} ({r['tt']/r['m32'][0]:.2f}×) | "
             f"{r['m16'][1]:.0f} ({r['tt']/r['m16'][0]:.2f}×) | {r['m16'][2]:.0e} |")
open("/app/aiter/bench_gluon_ua/decode_mfma16.md", "w").write("\n".join(L) + "\n")
print("wrote decode_mfma16.md")
