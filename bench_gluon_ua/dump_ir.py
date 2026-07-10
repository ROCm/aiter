"""Compare TTGIR + register/LDS usage: Triton 3d decode vs gfx950 gluon decode.
Current decode config: gluon 16x16 / BLOCK_M=16 / nw1 / split, seq-fastest grid.
Shape: C64 ctx8192 64/8 S8."""
import sys, math, re, collections
sys.path.insert(0, "/app/aiter/bench_gluon_ua")
import torch, triton
import bench_ua as B

DEV, DT, HS = B.DEV, B.DT, B.HEAD_SIZE
C, ctx, Hq, Hkv, S = 64, 8192, 64, 8, 8
nqpk = Hq // Hkv
scale = 1.0 / math.sqrt(HS)
TILE = 64
outdir = "/app/aiter/bench_gluon_ua"

q = torch.randn(C, Hq, HS, dtype=DT, device=DEV)
k, v, bt = B.make_paged_kv(ctx, C, TILE, Hkv)
cu = torch.arange(0, C + 1, dtype=torch.int32, device=DEV)
seqk = torch.full((C,), ctx, dtype=torch.int32, device=DEV)

TTG_TOKENS = ["tt.dot", "tt.load", "amdgpu.buffer_load", "buffer_load_to_local",
              "async_copy_global_to_local", "async_wait", "async_commit",
              "ttg.local_load", "ttg.local_alloc", "ttg.local_store",
              "tt.reduce", "tt.store", "amdgpu.buffer_store",
              "ttg.convert_layout", "scf.for", "tt.exp2", "math.exp2"]


def analyze(ck, name):
    md = ck.metadata
    regs = getattr(ck, "n_regs", None)
    spills = getattr(ck, "n_spills", None)
    shared = getattr(md, "shared", None)
    nw = getattr(md, "num_warps", None)
    ttgir = ck.asm.get("ttgir", "")
    with open(f"{outdir}/ir_{name}.ttgir", "w") as f:
        f.write(ttgir)
    hist = {t: len(re.findall(re.escape(t), ttgir)) for t in TTG_TOKENS}
    hist = {t: c for t, c in hist.items() if c}
    mma = sorted(set(re.findall(r"#(?:ttg\.)?(?:amd_)?(?:mfma|wmma)[^>]*", ttgir)))
    print(f"== {name} ==  VGPR={regs} spills={spills} LDS={shared}B num_warps={nw} ttgir_lines={len(ttgir.splitlines())}")
    print(f"   ops: {hist}")
    for m in mma[:3]:
        print(f"   mma-enc: {m[:110]}")
    return regs, spills, shared, nw


# ---- Triton 3d decode (BM=16, nw=2) ----
segm = B.alloc_segm(C, Hq, S)
ck_t = B.tri_3d[(C, Hkv, S)](
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
    TILE_SIZE=TILE, NUM_SEGMENTS_PER_SEQ=S, num_warps=2, waves_per_eu=2, num_stages=2,
)
analyze(ck_t, "triton3d_dec")

# ---- Gluon decode: 16x16 / BM16 / nw1 / split, seq-fastest grid (ALL_DECODE) ----
so, sm, se = B.alloc_segm(C, Hq, S)
og = torch.empty_like(q)
ck_g = B.glu_2d[(C, Hkv, S)](  # (tqb=NS, NKV, S) seq-fastest
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
    CAUSAL=True, REMOVE_INDIRECT_ACCESS=False, NUM_BUFFERS=2, MFMA_DIM=16,
    NUM_SPLITS=S, partial_m_ptr=sm, partial_l_ptr=se, partial_acc_ptr=so,
)
analyze(ck_g, "gluon_dec16")
print("wrote ir_triton3d_dec.ttgir / ir_gluon_dec16.ttgir")
