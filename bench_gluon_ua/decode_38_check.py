"""3.8.0 (native distributed path) decode: heuristic vs right-sized splits, ctx 1024 & 8192.
Decode is nb=2 here (thread_barrier absent on 3.8 => single-buffer nb=1 won't compile)."""
import sys, math
sys.path.insert(0, "/app/aiter/bench_gluon_ua")
import torch
import bench_ua as B

DEV, DT, HS, RCP, CU = B.DEV, B.DT, B.HEAD_SIZE, B.RCP_LN2, B.CU
TILE = 64
TARGET_WGS = CU * 4  # ~1024, the §11 right-sizing target


def run(C, ctx, Hq, Hkv):
    nqpk = Hq // Hkv; scale = 1.0 / math.sqrt(HS); num_tiles = ctx // TILE
    q = torch.randn(C, Hq, HS, dtype=DT, device=DEV); k, v, bt = B.make_paged_kv(ctx, C, TILE, Hkv)
    cu = torch.arange(0, C + 1, dtype=torch.int32, device=DEV); seqk = torch.full((C,), ctx, dtype=torch.int32, device=DEV)
    _, byts = B.decode_flops_bytes(C, ctx, Hq, Hkv)
    # triton @ heuristic
    attn, red = B.select_3d_config(HS, TILE, ctx, B.TARGET_PRGMS, C * Hkv, DT, DT, False, 1, 0)
    S_h, a_nw, wpe, nstg, r_nw = (attn["NUM_SEGMENTS_PER_SEQ"], attn["num_warps"],
                                  attn["waves_per_eu"], attn["num_stages"], red["num_warps"])
    seg = B.alloc_segm(C, Hq, S_h); ot = torch.empty_like(q)
    def tf():
        B.launch_tri_3d(q, k, v, cu, seqk, bt, scale, 16, 16 // nqpk, TILE, a_nw, S_h, wpe, nstg, seg)
        B.launch_reduce(ot, cu, seqk, bt, TILE, S_h, r_nw, 16 // nqpk, seg)
    tt = B.pick(B.profile_kernels(tf), "unified_attention_3d") + B.pick(B.profile_kernels(tf), "reduce_segments")

    def glu(S):
        so, sm, se = B.alloc_segm(C, Hq, S); og = torch.empty_like(q)
        def gf():
            B.launch_glu_2d(q, k, v, og, cu, seqk, bt, scale, 16, TILE, 1, wpe, NUM_SPLITS=S,
                            ALL_DECODE=True, partials=(sm, se, so), MFMA_DIM=16, NUM_BUFFERS=2)
            B.launch_reduce(og, cu, seqk, bt, TILE, S, r_nw, 16 // nqpk, (so, sm, se))
        return B.pick(B.profile_kernels(gf), "unified_attention_2d") + B.pick(B.profile_kernels(gf), "reduce_segments")

    S_rs = max(1, min(num_tiles, round(TARGET_WGS / (C * Hkv))))
    g_h = glu(S_h)
    g_rs = glu(S_rs)
    print(f"C{C} ctx{ctx} {Hq}/{Hkv}: tri {B.gbps(byts,tt):.0f}(S{S_h}) | "
          f"gluon heuristicS{S_h} {B.gbps(byts,g_h):.0f}({tt/g_h:.2f}x) | "
          f"gluon rightsizedS{S_rs} {B.gbps(byts,g_rs):.0f}({tt/g_rs:.2f}x)")


print("=== ctx=1024 ===")
for C in (16, 32, 64, 128):
    run(C, 1024, 8, 1)
run(64, 1024, 64, 8)
print("=== ctx=8192 ===")
for C in (16, 32, 64, 128):
    run(C, 8192, 8, 1)
run(64, 8192, 64, 8)
