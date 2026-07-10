"""Prototype eval: single-buffered 32KB decode LDS (NUM_BUFFERS=1) vs double (2).
16x16 MFMA / BLOCK_M=16 / nw1 / split, seq-fastest grid. Triton = reference."""
import sys, math
sys.path.insert(0, "/app/aiter/bench_gluon_ua")
import torch
import bench_ua as B

DEV, DT, HS, RCP = B.DEV, B.DT, B.HEAD_SIZE, B.RCP_LN2


def glu(q, k, v, out, cu, seqk, bt, scale, TILE, S, nb, partials, capture=False):
    NKV = k.shape[2]; NS = seqk.shape[0]; nqpk = q.shape[1] // NKV
    sm, se, so = partials
    ck = B.glu_2d[(NS, NKV, S)](
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
        BLOCK_Q=16 // nqpk, BLOCK_M=16, ARCH_NAME="gfx950", waves_per_eu=2,
        USE_LOAD_BUFFER_OP=True, USE_STORE_BUFFER_OP=True, num_warps=1, ALL_DECODE=True,
        CAUSAL=True, REMOVE_INDIRECT_ACCESS=False, NUM_BUFFERS=nb, MFMA_DIM=16,
        NUM_SPLITS=S, partial_m_ptr=sm, partial_l_ptr=se, partial_acc_ptr=so)
    if capture:
        md = ck.metadata
        return getattr(ck, "n_regs", None), getattr(ck, "n_spills", None), getattr(md, "shared", None)


def run(C, ctx, Hq, Hkv, occ=False):
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
    out = f"C{C} ctx{ctx} {Hq}/{Hkv} S{S}: tri {tt:5.1f}us {B.gbps(byts,tt):5.0f}"
    for nb in (2, 1):
        so, sm, se = B.alloc_segm(C, Hq, S); og = torch.empty_like(q)
        occres = glu(q, k, v, og, cu, seqk, bt, scale, TILE, S, nb, (sm, se, so), capture=occ)
        B.launch_reduce(og, cu, seqk, bt, TILE, S, r_nw, 16 // nqpk, (so, sm * (RCP * scale), se))
        torch.cuda.synchronize()
        xc = (ot.float() - og.float()).abs().max().item()
        def gf():
            glu(q, k, v, og, cu, seqk, bt, scale, TILE, S, nb, (sm, se, so))
            B.launch_reduce(og, cu, seqk, bt, TILE, S, r_nw, 16 // nqpk, (so, sm, se))
        gt = B.pick(B.profile_kernels(gf), "unified_attention_2d") + B.pick(B.profile_kernels(gf), "reduce_segments")
        os = f" [LDS{occres[2]//1024}KB VGPR{occres[0]} sp{occres[1]}]" if occ and occres else ""
        out += f" | nb{nb} {gt:5.1f}us {B.gbps(byts,gt):5.0f} ({tt/gt:.2f}x) xc{xc:.0e}{os}"
    print(out)


run(64, 8192, 8, 1, occ=True)
for sh in [(32, 8192, 8, 1), (16, 8192, 8, 1), (16, 1024, 64, 8),
           (64, 8192, 64, 8), (128, 8192, 64, 8), (128, 8192, 8, 1)]:
    run(*sh)
