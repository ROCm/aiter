"""Cross triton-version comparison (run on 3.6.0 / 3.7.1 / 3.8.0).
Uses the kernel's NATIVE layout gating (no monkeypatch): the KV-load layout is
chosen by ASYNC_COPY_SUPPORTS_DISTRIBUTED -> BlockedLayout on 3.6/3.7, the
distributed offset_bases path on 3.8+. Decode is double-buffered (nb=2) since
gl.thread_barrier (single-buffer) is still absent on 3.7/3.8. Writes
compare_371_<ver>.txt.
"""
import sys, math
import triton
sys.path.insert(0, "/app/aiter/bench_gluon_ua")
import torch
import bench_ua as B

DEV, DT, HS, RCP = B.DEV, B.DT, B.HEAD_SIZE, B.RCP_LN2
VER = triton.__version__
torch.manual_seed(0)
out = [f"triton {VER}  (native layout gating (distributed on 3.8+, Blocked on 3.6/3.7), decode nb=2)"]
print(out[0])

# ---- prefill b1 8k 64/8 (nb=2, 32x32) ----
r = B.run_prefill(dict(B=1, N=8192, Hq=64, Hkv=8))
line = f"PREFILL b1 8k 64/8: gluon {r['g_tf']:.0f} TFLOP/s ({r['speedup']:.2f}x vs triton {r['t_tf']:.0f})"
out.append(line); print(line)


# ---- decode nb=2 (double-buffered), MFMA=16/BM=16/nw1 ----
def decode_nb2(C, ctx, Hq, Hkv):
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
    # gluon: right-sized splits (Triton stays on its heuristic S above)
    Sg = B.select_gluon_num_splits(C, Hkv, ctx // TILE)
    og = torch.empty_like(q)
    if Sg == 1:
        # Enough WGs already -> no split: kernel writes output directly
        # (store_output), no reduce kernel.
        def gf():
            B.launch_glu_2d(q, k, v, og, cu, seqk, bt, scale, 16, TILE, 1, wpe,
                            NUM_SPLITS=1, ALL_DECODE=True, MFMA_DIM=16, NUM_BUFFERS=2)
        gf(); torch.cuda.synchronize()
        xc = (ot.float() - og.float()).abs().max().item()
        gt = B.pick(B.profile_kernels(gf), "unified_attention_2d")
    else:
        so, sm, se = B.alloc_segm(C, Hq, Sg)
        # correctness with prescaled M; timed run uses raw M (reduce cost identical)
        B.launch_glu_2d(q, k, v, og, cu, seqk, bt, scale, 16, TILE, 1, wpe,
                        NUM_SPLITS=Sg, ALL_DECODE=True, partials=(sm, se, so), MFMA_DIM=16, NUM_BUFFERS=2)
        B.launch_reduce(og, cu, seqk, bt, TILE, Sg, r_nw, 16 // nqpk, (so, sm * (RCP * scale), se))
        torch.cuda.synchronize()
        xc = (ot.float() - og.float()).abs().max().item()
        def gf():
            B.launch_glu_2d(q, k, v, og, cu, seqk, bt, scale, 16, TILE, 1, wpe,
                            NUM_SPLITS=Sg, ALL_DECODE=True, partials=(sm, se, so), MFMA_DIM=16, NUM_BUFFERS=2)
            B.launch_reduce(og, cu, seqk, bt, TILE, Sg, r_nw, 16 // nqpk, (so, sm, se))
        gt = B.pick(B.profile_kernels(gf), "unified_attention_2d") + B.pick(B.profile_kernels(gf), "reduce_segments")
    line = f"DECODE C{C} ctx{ctx} {Hq}/{Hkv}: gluon {B.gbps(byts,gt):.0f} GB/s (S{Sg}, {tt/gt:.2f}x vs triton {B.gbps(byts,tt):.0f} S{S})  xc{xc:.0e}"
    out.append(line); print(line)


for sh in [(64, 8192, 64, 8), (128, 8192, 64, 8), (32, 8192, 8, 1), (16, 8192, 8, 1),
           (64, 1024, 64, 8), (128, 1024, 64, 8), (32, 1024, 8, 1), (16, 1024, 8, 1)]:
    decode_nb2(*sh)

open(f"/app/aiter/bench_gluon_ua/compare_371_{VER}.txt", "w").write("\n".join(out) + "\n")
print("wrote compare_371_" + VER + ".txt")
