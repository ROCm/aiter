"""Small-ctx (1024) 8/1 decode: sweep NUM_SPLITS (capped at num_tiles=16).
KV is tiny (fits in cache) -> overhead/latency-bound, not BW. Report us + GB/s.
gluon nb1/nw1 vs triton@heuristic."""
import sys, math
sys.path.insert(0, "/app/aiter/bench_gluon_ua")
import torch
import bench_ua as B

DEV, DT, HS, RCP = B.DEV, B.DT, B.HEAD_SIZE, B.RCP_LN2


def run(C, ctx, Hq, Hkv):
    nqpk = Hq // Hkv; scale = 1.0 / math.sqrt(HS); TILE = 64
    num_tiles = ctx // TILE
    attn, red = B.select_3d_config(HS, 64, ctx, B.TARGET_PRGMS, C * Hkv, DT, DT, False, 1, 0)
    S_heur, a_nw, wpe, nstg, r_nw = (attn["NUM_SEGMENTS_PER_SEQ"], attn["num_warps"],
                                     attn["waves_per_eu"], attn["num_stages"], red["num_warps"])
    q = torch.randn(C, Hq, HS, dtype=DT, device=DEV); k, v, bt = B.make_paged_kv(ctx, C, TILE, Hkv)
    cu = torch.arange(0, C + 1, dtype=torch.int32, device=DEV); seqk = torch.full((C,), ctx, dtype=torch.int32, device=DEV)
    _, byts = B.decode_flops_bytes(C, ctx, Hq, Hkv)
    seg = B.alloc_segm(C, Hq, S_heur); ot = torch.empty_like(q)
    def tf():
        B.launch_tri_3d(q, k, v, cu, seqk, bt, scale, 16, 16 // nqpk, TILE, a_nw, S_heur, wpe, nstg, seg)
        B.launch_reduce(ot, cu, seqk, bt, TILE, S_heur, r_nw, 16 // nqpk, seg)
    tt = B.pick(B.profile_kernels(tf), "unified_attention_3d") + B.pick(B.profile_kernels(tf), "reduce_segments")
    out = f"C{C} ctx{ctx} {Hq}/{Hkv}: tri {tt:.1f}us {B.gbps(byts,tt):.0f}GB(S{S_heur})"
    for S in [s for s in (2, 4, 8, 16) if s <= num_tiles]:
        wgs = C * Hkv * S
        so, sm, se = B.alloc_segm(C, Hq, S); og = torch.empty_like(q)
        B.launch_glu_2d(q, k, v, og, cu, seqk, bt, scale, 16, TILE, 1, wpe,
                        NUM_SPLITS=S, ALL_DECODE=True, partials=(sm, se, so), MFMA_DIM=16, NUM_BUFFERS=1)
        B.launch_reduce(og, cu, seqk, bt, TILE, S, r_nw, 16 // nqpk, (so, sm * (RCP * scale), se))
        torch.cuda.synchronize(); xc = (ot.float() - og.float()).abs().max().item()
        def gf():
            B.launch_glu_2d(q, k, v, og, cu, seqk, bt, scale, 16, TILE, 1, wpe,
                            NUM_SPLITS=S, ALL_DECODE=True, partials=(sm, se, so), MFMA_DIM=16, NUM_BUFFERS=1)
            B.launch_reduce(og, cu, seqk, bt, TILE, S, r_nw, 16 // nqpk, (so, sm, se))
        gt = B.pick(B.profile_kernels(gf), "unified_attention_2d") + B.pick(B.profile_kernels(gf), "reduce_segments")
        star = "*" if S == S_heur else " "
        out += f" | S{S}{star} {gt:.1f}us({tt/gt:.2f}x,{wgs}WG)"
    print(out + f"  xc{xc:.0e}")


for sh in [(16, 1024, 8, 1), (32, 1024, 8, 1), (64, 1024, 8, 1), (128, 1024, 8, 1), (256, 1024, 8, 1)]:
    run(*sh)
