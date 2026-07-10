"""
Compare 16x16 vs 32x32 MFMA in the gfx950 gluon unified-attention kernel.

32x32 forces BLOCK_M = 32*num_warps; 16x16 allows BLOCK_M = 16*num_warps
(so decode can use BLOCK_M=16, matching Triton and halving GQA row waste).

Triton (its own heuristic) is the reference / correctness oracle.
Writes bench_gluon_ua/mfma_results.md
"""
import sys, math
sys.path.insert(0, "/app/aiter/bench_gluon_ua")
import torch, triton
import bench_ua as B

DEV, DT, RCP_LN2, HS = B.DEV, B.DT, B.RCP_LN2, B.HEAD_SIZE
OUT_MD = "/app/aiter/bench_gluon_ua/mfma_results.md"

PREFILL = [dict(Bt=1, N=8192, Hq=64, Hkv=8), dict(Bt=8, N=1024, Hq=8, Hkv=1)]
DECODE = [
    dict(C=16, ctx=1024, Hq=64, Hkv=8),
    dict(C=64, ctx=8192, Hq=64, Hkv=8),
    dict(C=128, ctx=8192, Hq=64, Hkv=8),
    dict(C=64, ctx=8192, Hq=8, Hkv=1),
]


def reduce_gluon(so, sm, se, out, cu, seqk, bt, TILE, S, r_nw, BQ, scale):
    # triton reduce needs prescaled (log2) max; gluon stores raw max
    B.launch_reduce(out, cu, seqk, bt, TILE, S, r_nw, BQ, (so, sm * (RCP_LN2 * scale), se))


def run_prefill(sh, rows):
    Bt, N, Hq, Hkv = sh["Bt"], sh["N"], sh["Hq"], sh["Hkv"]
    nqpk = Hq // Hkv; scale = 1.0 / math.sqrt(HS)
    c = B.select_2d_config(64, HS, 0, False, N, N, nqpk, 1, DT, DT, False)
    TILE, wpe = c["TILE_SIZE"], c["waves_per_eu"]
    bs = TILE
    q = torch.randn(Bt * N, Hq, HS, dtype=DT, device=DEV)
    k, v, bt = B.make_paged_kv(N, Bt, bs, Hkv)
    cu = torch.arange(0, (Bt + 1) * N, N, dtype=torch.int32, device=DEV)
    seqk = torch.full((Bt,), N, dtype=torch.int32, device=DEV)
    flops, _ = B.prefill_flops_bytes(Bt, N, Hq, Hkv)
    ot = torch.empty_like(q)
    B.launch_tri_2d(q, k, v, ot, cu, seqk, bt, scale, 128, 128 // nqpk, TILE, 4, 1, wpe)

    # (label, MFMA_DIM, BLOCK_M, num_warps)
    variants = [("gluon 32x32 BM128/nw4", 32, 128, 4),
                ("gluon 16x16 BM64/nw4", 16, 64, 4),
                ("gluon 16x16 BM128/nw8", 16, 128, 8)]
    t_us = B.pick(B.profile_kernels(lambda: B.launch_tri_2d(
        q, k, v, ot, cu, seqk, bt, scale, 128, 128 // nqpk, TILE, 4, 1, wpe)), "unified_attention")
    rows.append(dict(shape=f"b{Bt} {N}/{N} {Hq}/{Hkv}", label="triton 2d",
                     us=t_us, metric=B.tflops(flops, t_us), xc=0.0))
    for label, mf, bm, nw in variants:
        og = torch.empty_like(q)
        B.launch_glu_2d(q, k, v, og, cu, seqk, bt, scale, bm, TILE, nw, wpe, MFMA_DIM=mf)
        torch.cuda.synchronize()
        xc = (ot.float() - og.float()).abs().max().item()
        us = B.pick(B.profile_kernels(lambda: B.launch_glu_2d(
            q, k, v, og, cu, seqk, bt, scale, bm, TILE, nw, wpe, MFMA_DIM=mf)), "unified_attention")
        rows.append(dict(shape=f"b{Bt} {N}/{N} {Hq}/{Hkv}", label=label,
                         us=us, metric=B.tflops(flops, us), xc=xc))
    del q, k, v, bt, ot; torch.cuda.empty_cache()


def run_decode(sh, rows):
    C, ctx, Hq, Hkv = sh["C"], sh["ctx"], sh["Hq"], sh["Hkv"]
    nqpk = Hq // Hkv; scale = 1.0 / math.sqrt(HS)
    BM0 = 16 if nqpk <= 16 else triton.next_power_of_2(nqpk)
    attn, red = B.select_3d_config(HS, 64, ctx, B.TARGET_PRGMS, C * Hkv, DT, DT, False, 1, 0)
    TILE, S, a_nw, wpe, nstg = (attn["TILE_SIZE"], attn["NUM_SEGMENTS_PER_SEQ"],
                                attn["num_warps"], attn["waves_per_eu"], attn["num_stages"])
    r_nw = red["num_warps"]; bs = TILE
    q = torch.randn(C, Hq, HS, dtype=DT, device=DEV)
    k, v, bt = B.make_paged_kv(ctx, C, bs, Hkv)
    cu = torch.arange(0, C + 1, dtype=torch.int32, device=DEV)
    seqk = torch.full((C,), ctx, dtype=torch.int32, device=DEV)
    flops, byts = B.decode_flops_bytes(C, ctx, Hq, Hkv)

    # triton ref (3d + reduce)
    segm_t = B.alloc_segm(C, Hq, S)
    ot = torch.empty_like(q)

    def tri_fn():
        B.launch_tri_3d(q, k, v, cu, seqk, bt, scale, BM0, BM0 // nqpk, TILE, a_nw, S, wpe, nstg, segm_t)
        B.launch_reduce(ot, cu, seqk, bt, TILE, S, r_nw, BM0 // nqpk, segm_t)

    tri_fn(); torch.cuda.synchronize()
    ta = B.pick(B.profile_kernels(tri_fn), "unified_attention_3d") + B.pick(B.profile_kernels(tri_fn), "reduce_segments")
    rows.append(dict(shape=f"C{C} ctx{ctx} {Hq}/{Hkv}", S=S, label="triton 3d+red",
                     us=ta, metric=B.gbps(byts, ta), xc=0.0))

    for label, mf, bm, nw in [("gluon 32x32 BM32/nw1", 32, 32, 1),
                              ("gluon 16x16 BM16/nw1", 16, 16, 1)]:
        BQ = bm // nqpk
        so, sm, se = B.alloc_segm(C, Hq, S)
        og = torch.empty_like(q)
        # correctness: prescaled reduce
        B.launch_glu_2d(q, k, v, og, cu, seqk, bt, scale, bm, TILE, nw, wpe,
                        NUM_SPLITS=S, ALL_DECODE=True, partials=(sm, se, so), MFMA_DIM=mf)
        reduce_gluon(so, sm, se, og, cu, seqk, bt, TILE, S, r_nw, BQ, scale)
        torch.cuda.synchronize()
        xc = (ot.float() - og.float()).abs().max().item()

        def glu_fn():
            B.launch_glu_2d(q, k, v, og, cu, seqk, bt, scale, bm, TILE, nw, wpe,
                            NUM_SPLITS=S, ALL_DECODE=True, partials=(sm, se, so), MFMA_DIM=mf)
            B.launch_reduce(og, cu, seqk, bt, TILE, S, r_nw, BQ, (so, sm, se))

        us = B.pick(B.profile_kernels(glu_fn), "unified_attention_2d") + B.pick(B.profile_kernels(glu_fn), "reduce_segments")
        rows.append(dict(shape=f"C{C} ctx{ctx} {Hq}/{Hkv}", S=S, label=label,
                         us=us, metric=B.gbps(byts, us), xc=xc))
        del so, sm, se, og; torch.cuda.empty_cache()
    del q, k, v, bt, ot, segm_t; torch.cuda.empty_cache()


def main():
    torch.manual_seed(0)
    pre, dec = [], []
    for sh in PREFILL:
        run_prefill(sh, pre)
        for r in pre[-4:]:
            print(f"[P] {r['shape']:20s} {r['label']:24s} {r['us']:9.1f}us {r['metric']:7.0f} TF  xc {r['xc']:.1e}")
    for sh in DECODE:
        run_decode(sh, dec)
        for r in dec[-3:]:
            print(f"[D] {r['shape']:20s} S? {r['label']:24s} {r['us']:8.1f}us {r['metric']:6.0f} GB/s xc {r['xc']:.1e}")

    L = ["# gfx950 gluon: 16x16 vs 32x32 MFMA\n",
         f"- bf16, HEAD_SIZE={HS}, causal; Triton = reference. speedup vs triton in ().",
         f"- profiler {B.ITERS} iters, {B.FLUSH_MB}MB flush.\n",
         "## Prefill (TFLOP/s)\n",
         "| shape | variant | us | TFLOP/s | xcheck |", "|---|---|--:|--:|--:|"]
    tri = {}
    for r in pre:
        if r["label"].startswith("triton"): tri[r["shape"]] = r["us"]
    for r in pre:
        sp = f" ({tri[r['shape']]/r['us']:.2f}x)" if not r["label"].startswith("triton") else ""
        L.append(f"| {r['shape']} | {r['label']}{sp} | {r['us']:.1f} | {r['metric']:.0f} | {r['xc']:.1e} |")
    L += ["\n## Decode (attn+reduce, GB/s)\n",
          "| shape | S | variant | us | GB/s | xcheck |", "|---|--:|---|--:|--:|--:|"]
    tri = {}
    for r in dec:
        if r["label"].startswith("triton"): tri[r["shape"]] = r["us"]
    for r in dec:
        sp = f" ({tri[r['shape']]/r['us']:.2f}x)" if not r["label"].startswith("triton") else ""
        L.append(f"| {r['shape']} | {r['S']} | {r['label']}{sp} | {r['us']:.1f} | {r['metric']:.0f} | {r['xc']:.1e} |")
    open(OUT_MD, "w").write("\n".join(L) + "\n")
    print("wrote", OUT_MD)


if __name__ == "__main__":
    main()
