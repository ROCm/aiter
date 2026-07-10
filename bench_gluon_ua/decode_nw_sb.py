"""8/1 decode: single-buffer (32KB) with num_warps 1 (BM16) vs 2 (BM32).
nw=2 -> 4 waves/CU (matches triton) at the cost of BM=32 row-waste. Triton = ref."""
import sys, math
sys.path.insert(0, "/app/aiter/bench_gluon_ua")
import torch
import bench_ua as B

DEV, DT, HS, RCP = B.DEV, B.DT, B.HEAD_SIZE, B.RCP_LN2


def run(C, ctx, Hq, Hkv):
    nqpk = Hq // Hkv; scale = 1.0 / math.sqrt(HS); TILE = 64
    attn, red = B.select_3d_config(HS, 64, ctx, B.TARGET_PRGMS, C * Hkv, DT, DT, False, 1, 0)
    S, a_nw, wpe, nstg, r_nw = (attn["NUM_SEGMENTS_PER_SEQ"], attn["num_warps"],
                               attn["waves_per_eu"], attn["num_stages"], red["num_warps"])
    q = torch.randn(C, Hq, HS, dtype=DT, device=DEV); k, v, bt = B.make_paged_kv(ctx, C, TILE, Hkv)
    cu = torch.arange(0, C + 1, dtype=torch.int32, device=DEV); seqk = torch.full((C,), ctx, dtype=torch.int32, device=DEV)
    _, byts = B.decode_flops_bytes(C, ctx, Hq, Hkv)
    seg = B.alloc_segm(C, Hq, S); ot = torch.empty_like(q)
    def tf():
        B.launch_tri_3d(q, k, v, cu, seqk, bt, scale, 16, 16 // nqpk, TILE, a_nw, S, wpe, nstg, seg)
        B.launch_reduce(ot, cu, seqk, bt, TILE, S, r_nw, 16 // nqpk, seg)
    tt = B.pick(B.profile_kernels(tf), "unified_attention_3d") + B.pick(B.profile_kernels(tf), "reduce_segments")
    out = f"C{C} ctx{ctx} {Hq}/{Hkv} S{S}: tri {B.gbps(byts,tt):.0f}"
    for nw in (1, 2):
        BM = 16 * nw; BQ = BM // nqpk
        so, sm, se = B.alloc_segm(C, Hq, S); og = torch.empty_like(q)
        B.launch_glu_2d(q, k, v, og, cu, seqk, bt, scale, BM, TILE, nw, wpe,
                        NUM_SPLITS=S, ALL_DECODE=True, partials=(sm, se, so), MFMA_DIM=16, NUM_BUFFERS=1)
        B.launch_reduce(og, cu, seqk, bt, TILE, S, r_nw, BQ, (so, sm * (RCP * scale), se))
        torch.cuda.synchronize(); xc = (ot.float() - og.float()).abs().max().item()
        def gf():
            B.launch_glu_2d(q, k, v, og, cu, seqk, bt, scale, BM, TILE, nw, wpe,
                            NUM_SPLITS=S, ALL_DECODE=True, partials=(sm, se, so), MFMA_DIM=16, NUM_BUFFERS=1)
            B.launch_reduce(og, cu, seqk, bt, TILE, S, r_nw, BQ, (so, sm, se))
        gt = B.pick(B.profile_kernels(gf), "unified_attention_2d") + B.pick(B.profile_kernels(gf), "reduce_segments")
        out += f" | nw{nw}/BM{BM} {B.gbps(byts,gt):.0f} ({tt/gt:.2f}x) xc{xc:.0e}"
    print(out)


for sh in [(64, 8192, 8, 1), (32, 8192, 8, 1), (16, 8192, 8, 1), (128, 8192, 8, 1),
           (64, 8192, 64, 8), (16, 1024, 64, 8)]:
    run(*sh)
