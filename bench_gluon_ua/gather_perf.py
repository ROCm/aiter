"""Gather overhead: gluon TILE=64 with page=64 (non-gather, AsyncKVLoader) vs
page=16 (gather x4, AsyncGatherKVLoader). Same TILE/LDS/occupancy => delta is
the per-element block-table gather cost. ratio = gather / non-gather."""
import sys, math
sys.path.insert(0, "/app/aiter/bench_gluon_ua")
import torch
import bench_ua as B

DEV, DT, HS, RCP, CU = B.DEV, B.DT, B.HEAD_SIZE, B.RCP_LN2, B.CU
TILE = 64


def prefill(page):
    N, Hq, Hkv = 8192, 64, 8; nqpk = Hq // Hkv; scale = 1.0 / math.sqrt(HS)
    q = torch.randn(N, Hq, HS, dtype=DT, device=DEV); k, v, bt = B.make_paged_kv(N, 1, page, Hkv)
    cu = torch.tensor([0, N], dtype=torch.int32, device=DEV); seqk = torch.tensor([N], dtype=torch.int32, device=DEV)
    flops, _ = B.prefill_flops_bytes(1, N, Hq, Hkv)
    og = torch.empty_like(q)
    B.launch_glu_2d(q, k, v, og, cu, seqk, bt, scale, 128, TILE, 4, 2, MFMA_DIM=32)  # nb=2 default
    torch.cuda.synchronize()
    us = B.pick(B.profile_kernels(lambda: B.launch_glu_2d(q, k, v, og, cu, seqk, bt, scale, 128, TILE, 4, 2, MFMA_DIM=32)), "unified_attention")
    return B.tflops(flops, us)


def decode(C, ctx, Hq, Hkv, page):
    nqpk = Hq // Hkv; scale = 1.0 / math.sqrt(HS)
    num_tiles = ctx // TILE
    S = max(1, min(num_tiles, round(CU * 4 / (C * Hkv))))
    q = torch.randn(C, Hq, HS, dtype=DT, device=DEV); k, v, bt = B.make_paged_kv(ctx, C, page, Hkv)
    cu = torch.arange(0, C + 1, dtype=torch.int32, device=DEV); seqk = torch.full((C,), ctx, dtype=torch.int32, device=DEV)
    _, byts = B.decode_flops_bytes(C, ctx, Hq, Hkv)
    so, sm, se = B.alloc_segm(C, Hq, S); og = torch.empty_like(q)
    def gf():
        B.launch_glu_2d(q, k, v, og, cu, seqk, bt, scale, 16, TILE, 1, 2, NUM_SPLITS=S,
                        ALL_DECODE=True, partials=(sm, se, so), MFMA_DIM=16, NUM_BUFFERS=1)
        B.launch_reduce(og, cu, seqk, bt, TILE, S, 1, 16 // nqpk, (so, sm, se))
    us = B.pick(B.profile_kernels(gf), "unified_attention_2d") + B.pick(B.profile_kernels(gf), "reduce_segments")
    return B.gbps(byts, us), S


p64, p16 = prefill(64), prefill(16)
print(f"PREFILL b1 8192 64/8 TILE64: page64(non-gather) {p64:.0f}TF | page16(gather x4) {p16:.0f}TF  ({p16/p64:.2f}x)")
for C, ctx, Hq, Hkv in [(64, 8192, 64, 8), (128, 8192, 64, 8), (64, 8192, 8, 1), (16, 8192, 8, 1)]:
    g64, S = decode(C, ctx, Hq, Hkv, 64)
    g16, _ = decode(C, ctx, Hq, Hkv, 16)
    print(f"DECODE C{C} ctx{ctx} {Hq}/{Hkv} S{S} TILE64: page64(non-gather) {g64:.0f} | page16(gather x4) {g16:.0f} GB/s  ({g16/g64:.2f}x)")
