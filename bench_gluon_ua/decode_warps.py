"""Decode: gluon num_warps sweep (16x16 MFMA => BLOCK_M = 16*num_warps).
Isolates the waves-per-CU effect vs Triton (num_warps=2). Grid seq-fastest, split."""
import sys, math
sys.path.insert(0, "/app/aiter/bench_gluon_ua")
import torch
import bench_ua as B

DEV, DT, HS, RCP = B.DEV, B.DT, B.HEAD_SIZE, B.RCP_LN2


def glu_launch(q, k, v, out, cu, seqk, bt, scale, TILE, S, nw, partials, capture=False):
    NKV = k.shape[2]; NS = seqk.shape[0]; nqpk = q.shape[1] // NKV
    BM = 16 * nw; BQ = BM // nqpk
    sm, se, so = partials
    ck = B.glu_2d[(NS, NKV, S)](  # seq-fastest
        query_ptr=q, key_cache_ptr=k, value_cache_ptr=v, sink_ptr=None, output_ptr=out,
        block_tables_ptr=bt, seq_lens_ptr=seqk, query_start_len_ptr=cu,
        query_stride_0=q.stride(0), query_stride_1=q.stride(1),
        output_stride_0=out.stride(0), output_stride_1=out.stride(1),
        k_descale_ptr=None, v_descale_ptr=None, q_descale_ptr=None, out_scale_ptr=None,
        USE_SINKS=False, SLIDING_WINDOW=0, num_blocks=k.shape[0],
        stride_k_cache_0=k.stride(0), stride_k_cache_1=k.stride(1), stride_k_cache_2=k.stride(2), stride_k_cache_3=k.stride(3),
        stride_v_cache_0=v.stride(0), stride_v_cache_1=v.stride(1), stride_v_cache_2=v.stride(2), stride_v_cache_3=v.stride(3),
        block_table_stride=bt.stride(0), num_seqs=NS, SCALE=scale,
        NUM_QUERY_HEADS=q.shape[1], NUM_KV_HEADS=NKV, BLOCK_SIZE=TILE, TILE_SIZE=TILE, HEAD_SIZE=HS,
        BLOCK_Q=BQ, BLOCK_M=BM, ARCH_NAME="gfx950", waves_per_eu=2,
        USE_LOAD_BUFFER_OP=True, USE_STORE_BUFFER_OP=True, num_warps=nw, ALL_DECODE=True,
        CAUSAL=True, REMOVE_INDIRECT_ACCESS=False, NUM_BUFFERS=2, MFMA_DIM=16,
        NUM_SPLITS=S, partial_m_ptr=sm, partial_l_ptr=se, partial_acc_ptr=so)
    if capture:
        md = ck.metadata
        return getattr(ck, "n_regs", None), getattr(ck, "n_spills", None), getattr(md, "shared", None)
    return None


def run(C, ctx, Hq, Hkv, show_occ=False):
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
    out = f"C{C} ctx{ctx} {Hq}/{Hkv} S{S}: tri(nw2/BM16) {tt:5.1f}us {B.gbps(byts,tt):5.0f}"
    for nw in (1, 2, 4):
        so, sm, se = B.alloc_segm(C, Hq, S); og = torch.empty_like(q)
        occ = glu_launch(q, k, v, og, cu, seqk, bt, scale, TILE, S, nw, (sm, se, so), capture=show_occ)
        B.launch_reduce(og, cu, seqk, bt, TILE, S, r_nw, (16 * nw) // nqpk, (so, sm * (RCP * scale), se))
        torch.cuda.synchronize()
        xc = (ot.float() - og.float()).abs().max().item()
        def gf():
            glu_launch(q, k, v, og, cu, seqk, bt, scale, TILE, S, nw, (sm, se, so))
            B.launch_reduce(og, cu, seqk, bt, TILE, S, r_nw, (16 * nw) // nqpk, (so, sm, se))
        gt = B.pick(B.profile_kernels(gf), "unified_attention_2d") + B.pick(B.profile_kernels(gf), "reduce_segments")
        tag = f"nw{nw}/BM{16*nw}"
        occstr = f" [VGPR {occ[0]} sp{occ[1]} LDS{occ[2]//1024}KB]" if show_occ and occ else ""
        out += f" | {tag} {gt:5.1f}us {B.gbps(byts,gt):5.0f} ({tt/gt:.2f}x){occstr}"
    print(out + (f"  xc{xc:.0e}" if not show_occ else ""))


run(64, 8192, 8, 1, show_occ=True)   # MQA, with occupancy dump
for sh in [(32, 8192, 8, 1), (16, 8192, 8, 1), (128, 8192, 8, 1),
           (16, 1024, 64, 8), (64, 8192, 64, 8), (128, 8192, 64, 8)]:
    run(*sh)
