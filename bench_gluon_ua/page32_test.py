"""Decode: page_size 64 vs 32 (gluon TILE=page). page=32 => 16KB LDS (occ 4) +
2x num_tiles (more splittable WGs). Triton@heuristic vs gluon@right-sized-splits, per page.
8/1 (MQA) focus + one GQA regression check."""
import sys, math
sys.path.insert(0, "/app/aiter/bench_gluon_ua")
import torch
import bench_ua as B

DEV, DT, HS, RCP, CU = B.DEV, B.DT, B.HEAD_SIZE, B.RCP_LN2, B.CU
TARGET_WGS = CU * 4  # ~1024


def rightsized(C, Hkv, num_tiles):
    return max(1, min(num_tiles, round(TARGET_WGS / (C * Hkv))))


def tri(q, k, v, cu, seqk, bt, scale, TILE, nqpk, C, ctx, Hkv):
    attn, red = B.select_3d_config(HS, TILE, ctx, B.TARGET_PRGMS, C * Hkv, DT, DT, False, 1, 0)
    S, a_nw, wpe, nstg, r_nw = (attn["NUM_SEGMENTS_PER_SEQ"], attn["num_warps"],
                               attn["waves_per_eu"], attn["num_stages"], red["num_warps"])
    seg = B.alloc_segm(C, q.shape[1], S); ot = torch.empty_like(q)
    def f():
        B.launch_tri_3d(q, k, v, cu, seqk, bt, scale, 16, 16 // nqpk, TILE, a_nw, S, wpe, nstg, seg)
        B.launch_reduce(ot, cu, seqk, bt, TILE, S, r_nw, 16 // nqpk, seg)
    return B.pick(B.profile_kernels(f), "unified_attention_3d") + B.pick(B.profile_kernels(f), "reduce_segments"), S, wpe, r_nw


def glu(q, k, v, cu, seqk, bt, scale, TILE, S, nqpk, C, wpe, r_nw, cap_lds=[None]):
    Hq = q.shape[1]
    so, sm, se = B.alloc_segm(C, Hq, S); og = torch.empty_like(q)
    ck = B.glu_2d[(C, k.shape[2], S)](
        query_ptr=q, key_cache_ptr=k, value_cache_ptr=v, sink_ptr=None, output_ptr=og,
        block_tables_ptr=bt, seq_lens_ptr=seqk, query_start_len_ptr=cu,
        query_stride_0=q.stride(0), query_stride_1=q.stride(1),
        output_stride_0=og.stride(0), output_stride_1=og.stride(1),
        k_descale_ptr=None, v_descale_ptr=None, q_descale_ptr=None, out_scale_ptr=None,
        USE_SINKS=False, SLIDING_WINDOW=0, num_blocks=k.shape[0],
        stride_k_cache_0=k.stride(0), stride_k_cache_1=k.stride(1), stride_k_cache_2=k.stride(2), stride_k_cache_3=k.stride(3),
        stride_v_cache_0=v.stride(0), stride_v_cache_1=v.stride(1), stride_v_cache_2=v.stride(2), stride_v_cache_3=v.stride(3),
        block_table_stride=bt.stride(0), num_seqs=C, SCALE=scale,
        NUM_QUERY_HEADS=Hq, NUM_KV_HEADS=k.shape[2], BLOCK_SIZE=TILE, TILE_SIZE=TILE, HEAD_SIZE=HS,
        BLOCK_Q=16 // nqpk, BLOCK_M=16, ARCH_NAME="gfx950", waves_per_eu=wpe,
        USE_LOAD_BUFFER_OP=True, USE_STORE_BUFFER_OP=True, num_warps=1, ALL_DECODE=True,
        CAUSAL=True, REMOVE_INDIRECT_ACCESS=False, NUM_BUFFERS=1, MFMA_DIM=16,
        NUM_SPLITS=S, partial_m_ptr=sm, partial_l_ptr=se, partial_acc_ptr=so)
    cap_lds[0] = ck.metadata.shared
    B.launch_reduce(og, cu, seqk, bt, TILE, S, r_nw, 16 // nqpk, (so, sm * (RCP * scale), se))
    def f():
        B.glu_2d[(C, k.shape[2], S)](
            query_ptr=q, key_cache_ptr=k, value_cache_ptr=v, sink_ptr=None, output_ptr=og,
            block_tables_ptr=bt, seq_lens_ptr=seqk, query_start_len_ptr=cu,
            query_stride_0=q.stride(0), query_stride_1=q.stride(1),
            output_stride_0=og.stride(0), output_stride_1=og.stride(1),
            k_descale_ptr=None, v_descale_ptr=None, q_descale_ptr=None, out_scale_ptr=None,
            USE_SINKS=False, SLIDING_WINDOW=0, num_blocks=k.shape[0],
            stride_k_cache_0=k.stride(0), stride_k_cache_1=k.stride(1), stride_k_cache_2=k.stride(2), stride_k_cache_3=k.stride(3),
            stride_v_cache_0=v.stride(0), stride_v_cache_1=v.stride(1), stride_v_cache_2=v.stride(2), stride_v_cache_3=v.stride(3),
            block_table_stride=bt.stride(0), num_seqs=C, SCALE=scale,
            NUM_QUERY_HEADS=Hq, NUM_KV_HEADS=k.shape[2], BLOCK_SIZE=TILE, TILE_SIZE=TILE, HEAD_SIZE=HS,
            BLOCK_Q=16 // nqpk, BLOCK_M=16, ARCH_NAME="gfx950", waves_per_eu=wpe,
            USE_LOAD_BUFFER_OP=True, USE_STORE_BUFFER_OP=True, num_warps=1, ALL_DECODE=True,
            CAUSAL=True, REMOVE_INDIRECT_ACCESS=False, NUM_BUFFERS=1, MFMA_DIM=16,
            NUM_SPLITS=S, partial_m_ptr=sm, partial_l_ptr=se, partial_acc_ptr=so)
        B.launch_reduce(og, cu, seqk, bt, TILE, S, r_nw, 16 // nqpk, (so, sm, se))
    return B.pick(B.profile_kernels(f), "unified_attention_2d") + B.pick(B.profile_kernels(f), "reduce_segments")


def run(C, ctx, Hq, Hkv):
    nqpk = Hq // Hkv; scale = 1.0 / math.sqrt(HS)
    _, byts = B.decode_flops_bytes(C, ctx, Hq, Hkv)
    cu = torch.arange(0, C + 1, dtype=torch.int32, device=DEV); seqk = torch.full((C,), ctx, dtype=torch.int32, device=DEV)
    q = torch.randn(C, Hq, HS, dtype=DT, device=DEV)
    res = {}
    for page in (64, 32):
        k, v, bt = B.make_paged_kv(ctx, C, page, Hkv)
        tt, Sh, wpe, r_nw = tri(q, k, v, cu, seqk, bt, scale, page, nqpk, C, ctx, Hkv)
        S_rs = rightsized(C, Hkv, ctx // page)
        lds = [None]
        gt = glu(q, k, v, cu, seqk, bt, scale, page, S_rs, nqpk, C, wpe, r_nw, lds)
        res[page] = (B.gbps(byts, tt), B.gbps(byts, gt), tt / gt, Sh, S_rs, lds[0] // 1024)
        del k, v, bt; torch.cuda.empty_cache()
    p64, p32 = res[64], res[32]
    print(f"C{C} ctx{ctx} {Hq}/{Hkv}: "
          f"page64[tri {p64[0]:.0f} | glu {p64[1]:.0f}({p64[2]:.2f}x,S{p64[4]})]  "
          f"page32[tri {p32[0]:.0f} | glu {p32[1]:.0f}({p32[2]:.2f}x,S{p32[4]},LDS{p32[5]}KB)]")


for sh in [(16, 1024, 8, 1), (64, 1024, 8, 1), (128, 1024, 8, 1),
           (16, 8192, 8, 1), (64, 8192, 8, 1), (128, 8192, 8, 1),
           (64, 8192, 64, 8)]:
    run(*sh)
