"""Correctness for the new AsyncGatherKVLoader (TILE_SIZE > page_size).
Compares gluon (gather, TILE=64) vs Triton 3d, both on the same paged KV cache."""
import sys, math
sys.path.insert(0, "/app/aiter/bench_gluon_ua")
import torch
import bench_ua as B

DEV, DT, HS, RCP = B.DEV, B.DT, B.HEAD_SIZE, B.RCP_LN2


def ref_vs_gluon(C, ctx, Hq, Hkv, page, TILE_g):
    nqpk = Hq // Hkv; scale = 1.0 / math.sqrt(HS)
    k, v, bt = B.make_paged_kv(ctx, C, page, Hkv)
    q = torch.randn(C, Hq, HS, dtype=DT, device=DEV)
    cu = torch.arange(0, C + 1, dtype=torch.int32, device=DEV)
    seqk = torch.full((C,), ctx, dtype=torch.int32, device=DEV)
    # Triton reference at this page (uses its own TILE from the heuristic)
    attn, red = B.select_3d_config(HS, page, ctx, B.TARGET_PRGMS, C * Hkv, DT, DT, False, 1, 0)
    TILE_t, S_t = attn["TILE_SIZE"], attn["NUM_SEGMENTS_PER_SEQ"]
    seg = B.alloc_segm(C, Hq, S_t); ot = torch.empty_like(q)
    B.launch_tri_3d(q, k, v, cu, seqk, bt, scale, 16, 16 // nqpk, TILE_t, attn["num_warps"], S_t,
                    attn["waves_per_eu"], attn["num_stages"], seg)
    B.launch_reduce(ot, cu, seqk, bt, TILE_t, S_t, red["num_warps"], 16 // nqpk, seg)
    # gluon at this page with TILE_g (gather auto-selected when TILE_g != page)
    num_tiles = ctx // TILE_g
    S_g = min(num_tiles, 4)
    so, sm, se = B.alloc_segm(C, Hq, S_g); og = torch.empty_like(q)
    B.launch_glu_2d(q, k, v, og, cu, seqk, bt, scale, 16, TILE_g, 1, 2,
                    NUM_SPLITS=S_g, ALL_DECODE=True, partials=(sm, se, so), MFMA_DIM=16, NUM_BUFFERS=1)
    B.launch_reduce(og, cu, seqk, bt, TILE_g, S_g, red["num_warps"], 16 // nqpk, (so, sm * (RCP * scale), se))
    torch.cuda.synchronize()
    return (ot.float() - og.float()).abs().max().item(), TILE_t


# regression: TILE == page (AsyncKVLoader path)
e, _ = ref_vs_gluon(4, 512, 64, 8, 64, 64)
print(f"TILE==page  (page64, gluon TILE64, non-gather): max_abs={e:.2e}")
# gather: TILE > page
e, tt = ref_vs_gluon(4, 512, 64, 8, 32, 64)
print(f"TILE>page   (page32, gluon TILE64 gather x2; triton TILE{tt}): max_abs={e:.2e}")
e, tt = ref_vs_gluon(4, 512, 64, 8, 16, 64)
print(f"TILE>page   (page16, gluon TILE64 gather x4; triton TILE{tt}): max_abs={e:.2e}")
# MQA gather
e, tt = ref_vs_gluon(4, 512, 8, 1, 32, 64)
print(f"TILE>page   (page32, 8/1, gluon TILE64 gather x2): max_abs={e:.2e}")
print("OK")
