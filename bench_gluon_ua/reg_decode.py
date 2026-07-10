"""Register/LDS/occupancy: current gluon decode (single-buf, MFMA=16, BM=16, nw=1)
vs Triton 3d decode (BM=16, nw=2), for an 8/1 (MQA) shape. C64 ctx8192 8/1 S64."""
import sys, math
sys.path.insert(0, "/app/aiter/bench_gluon_ua")
import torch
import bench_ua as B

DEV, DT, HS = B.DEV, B.DT, B.HEAD_SIZE
LDS_PER_CU = 64 * 1024  # gfx950


def show(ck, name, nw):
    md = ck.metadata
    regs = getattr(ck, "n_regs", None); sp = getattr(ck, "n_spills", None)
    lds = getattr(md, "shared", 0)
    wg_cu = LDS_PER_CU // lds if lds else 0
    print(f"{name:26s} VGPR={regs:4d} spills={sp}  LDS={lds//1024}KB  num_warps={nw}  "
          f"-> wg/CU(LDS)~{wg_cu}  waves/CU~{wg_cu*nw}")


def main():
    C, ctx, Hq, Hkv, S = 64, 8192, 8, 1, 64
    nqpk = Hq // Hkv; scale = 1.0 / math.sqrt(HS); TILE = 64
    q = torch.randn(C, Hq, HS, dtype=DT, device=DEV)
    k, v, bt = B.make_paged_kv(ctx, C, TILE, Hkv)
    cu = torch.arange(0, C + 1, dtype=torch.int32, device=DEV)
    seqk = torch.full((C,), ctx, dtype=torch.int32, device=DEV)

    # gluon current decode default: single-buffer, MFMA=16, BM=16, nw=1
    so, sm, se = B.alloc_segm(C, Hq, S); og = torch.empty_like(q)
    ckg = B.glu_2d[(C, Hkv, S)](
        query_ptr=q, key_cache_ptr=k, value_cache_ptr=v, sink_ptr=None, output_ptr=og,
        block_tables_ptr=bt, seq_lens_ptr=seqk, query_start_len_ptr=cu,
        query_stride_0=q.stride(0), query_stride_1=q.stride(1),
        output_stride_0=og.stride(0), output_stride_1=og.stride(1),
        k_descale_ptr=None, v_descale_ptr=None, q_descale_ptr=None, out_scale_ptr=None,
        USE_SINKS=False, SLIDING_WINDOW=0, num_blocks=k.shape[0],
        stride_k_cache_0=k.stride(0), stride_k_cache_1=k.stride(1), stride_k_cache_2=k.stride(2), stride_k_cache_3=k.stride(3),
        stride_v_cache_0=v.stride(0), stride_v_cache_1=v.stride(1), stride_v_cache_2=v.stride(2), stride_v_cache_3=v.stride(3),
        block_table_stride=bt.stride(0), num_seqs=C, SCALE=scale,
        NUM_QUERY_HEADS=Hq, NUM_KV_HEADS=Hkv, BLOCK_SIZE=TILE, TILE_SIZE=TILE, HEAD_SIZE=HS,
        BLOCK_Q=16 // nqpk, BLOCK_M=16, ARCH_NAME="gfx950", waves_per_eu=2,
        USE_LOAD_BUFFER_OP=True, USE_STORE_BUFFER_OP=True, num_warps=1, ALL_DECODE=True,
        CAUSAL=True, REMOVE_INDIRECT_ACCESS=False, NUM_BUFFERS=1, MFMA_DIM=16,
        NUM_SPLITS=S, partial_m_ptr=sm, partial_l_ptr=se, partial_acc_ptr=so)
    show(ckg, "gluon decode (nb1/16x16)", 1)

    segm = B.alloc_segm(C, Hq, S)
    ckt = B.tri_3d[(C, Hkv, S)](
        segm_output_ptr=segm[0], segm_max_ptr=segm[1], segm_expsum_ptr=segm[2],
        query_ptr=q, key_cache_ptr=k, value_cache_ptr=v, sink_ptr=None,
        block_tables_ptr=bt, seq_lens_ptr=seqk, alibi_slopes_ptr=None, qq_bias_ptr=None,
        scale=scale, q_descale_ptr=None, k_descale_ptr=None, v_descale_ptr=None,
        out_scale_ptr=None, softcap=0.0, num_query_heads=Hq, num_queries_per_kv=nqpk,
        block_table_stride=bt.stride(0), query_stride_0=q.stride(0), query_stride_1=q.stride(1),
        qq_bias_stride_0=0, BLOCK_SIZE=TILE, HEAD_SIZE=HS, HEAD_SIZE_PADDED=HS,
        USE_ALIBI_SLOPES=False, USE_QQ_BIAS=False, USE_SOFTCAP=False, USE_SINKS=False, SLIDING_WINDOW=0,
        stride_k_cache_0=k.stride(0), stride_k_cache_1=k.stride(1), stride_k_cache_2=k.stride(2), stride_k_cache_3=k.stride(3),
        stride_v_cache_0=v.stride(0), stride_v_cache_1=v.stride(1), stride_v_cache_2=v.stride(2), stride_v_cache_3=v.stride(3),
        query_start_len_ptr=cu, BLOCK_Q=16 // nqpk, num_seqs=C, BLOCK_M=16,
        ALL_DECODE=True, SHUFFLED_KV_CACHE=False, K_WIDTH=8, IS_Q_FP8=False, IS_KV_FP8=False,
        TILE_SIZE=TILE, NUM_SEGMENTS_PER_SEQ=S, num_warps=2, waves_per_eu=2, num_stages=2)
    show(ckt, "triton 3d decode (nw2)", 2)


main()
