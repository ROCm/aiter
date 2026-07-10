"""
Benchmark: existing Triton unified-attention kernels vs the gfx950 Gluon kernel.

Config is chosen by the *Triton dispatcher heuristics* (select_2d_config /
select_3d_config) rather than hand-picked. The heuristic drives the shared knobs
(TILE_SIZE, block_size, and for decode NUM_SEGMENTS/NUM_SPLITS); Triton also uses
its heuristic BLOCK_M / num_warps. The gluon kernel is architecturally tied to
BLOCK_M = 32*num_warps (32x32 MFMA), so:
    prefill : heuristic gives BLOCK_M=128,num_warps=4 -> gluon uses it verbatim.
    decode  : gluon uses its minimal M-tile BLOCK_M=32,num_warps=1 (Triton stays
              on heuristic BLOCK_M=16,num_warps=2); TILE_SIZE + NUM_SEGMENTS shared.

Decode uses split-KV + Triton reduce_segments for BOTH paths (identical reduce).

Timing: torch.profiler aggregated per GPU-kernel name, L2 flush between iters.
Buffers use torch.empty (like the original 3d dispatch).

Run:  python bench_gluon_ua/bench_ua.py     # writes bench_gluon_ua/results.md
"""
import math
import torch
import triton

from aiter.ops.triton.attention.unified_attention import select_2d_config, select_3d_config
from aiter.ops.triton.utils.device_info import get_num_sms
from aiter.ops.triton._triton_kernels.attention.unified_attention import (
    kernel_unified_attention_2d as tri_2d,
    kernel_unified_attention_3d as tri_3d,
    reduce_segments,
)
from aiter.ops.triton._gluon_kernels.gfx950.attention.unified_attention import (
    kernel_unified_attention_2d as glu_2d,
)
from torch.profiler import profile, ProfilerActivity

DEV = "cuda"
DT = torch.bfloat16
RCP_LN2 = 1.4426950408889634
HEAD_SIZE = 128
CU = get_num_sms()
TARGET_PRGMS = CU * 4
WARMUP, ITERS, FLUSH_MB = 8, 30, 512
OUT_MD = "/app/aiter/bench_gluon_ua/results.md"

# heads given as (num_query_heads, num_kv_heads); both cases are nqpk=8
PREFILL_SHAPES = [
    dict(B=1, N=8192, Hq=64, Hkv=8),
    dict(B=8, N=1024, Hq=64, Hkv=8),
    dict(B=8, N=1024, Hq=8, Hkv=1),   # TP8
]
DECODE_SHAPES = [
    dict(C=C, ctx=ctx, Hq=Hq, Hkv=Hkv)
    for ctx in (1024, 4096, 8192)
    for C in (16, 32, 64, 128)
    for (Hq, Hkv) in ((64, 8), (8, 1))
]

_flush = torch.empty(FLUSH_MB * 1024 * 1024 // 4, dtype=torch.float32, device=DEV)
BT_PAD = 8
SEGM_TOK_PAD = 256


# ---------------------------------------------------------------- input setup
def make_paged_kv(ctx_per_seq, num_seqs, block_size, Hkv):
    nb_per_seq = ctx_per_seq // block_size
    num_blocks = nb_per_seq * num_seqs
    k = torch.randn(num_blocks, block_size, Hkv, HEAD_SIZE, dtype=DT, device=DEV) * 0.5
    v = torch.randn(num_blocks, block_size, Hkv, HEAD_SIZE, dtype=DT, device=DEV) * 0.5
    bt = torch.zeros(num_seqs, nb_per_seq + BT_PAD, dtype=torch.int32, device=DEV)
    bt[:, :nb_per_seq] = torch.arange(num_blocks, dtype=torch.int32, device=DEV).view(num_seqs, nb_per_seq)
    return k, v, bt


def alloc_segm(nt, Hq, S):
    n = nt + SEGM_TOK_PAD
    so = torch.empty(n, Hq, S, HEAD_SIZE, dtype=torch.float32, device=DEV)
    sm = torch.empty(n, Hq, S, dtype=torch.float32, device=DEV)
    se = torch.empty(n, Hq, S, dtype=torch.float32, device=DEV)
    return so, sm, se


# ---------------------------------------------------------------- launchers
def launch_tri_2d(q, k, v, out, cu, seqk, bt, scale, BM, BQ, TILE, nw, ns, wpe):
    nt, Hq, D = q.shape; NKV = k.shape[2]; NS = seqk.shape[0]; nqpk = Hq // NKV
    tqb = nt // BQ + NS
    tri_2d[(NKV, tqb)](
        output_ptr=out, query_ptr=q, key_cache_ptr=k, value_cache_ptr=v, sink_ptr=None,
        block_tables_ptr=bt, seq_lens_ptr=seqk, alibi_slopes_ptr=None, qq_bias_ptr=None,
        scale=scale, q_descale_ptr=None, k_descale_ptr=None, v_descale_ptr=None,
        out_scale_ptr=None, softcap=0.0, num_query_heads=Hq, num_queries_per_kv=nqpk,
        block_table_stride=bt.stride(0), query_stride_0=q.stride(0), query_stride_1=q.stride(1),
        output_stride_0=out.stride(0), output_stride_1=out.stride(1), qq_bias_stride_0=0,
        BLOCK_SIZE=k.shape[1], HEAD_SIZE=D, HEAD_SIZE_PADDED=triton.next_power_of_2(D),
        USE_ALIBI_SLOPES=False, USE_QQ_BIAS=False, USE_SOFTCAP=False, USE_SINKS=False,
        SLIDING_WINDOW=0,
        stride_k_cache_0=k.stride(0), stride_k_cache_1=k.stride(1),
        stride_k_cache_2=k.stride(2), stride_k_cache_3=k.stride(3),
        stride_v_cache_0=v.stride(0), stride_v_cache_1=v.stride(1),
        stride_v_cache_2=v.stride(2), stride_v_cache_3=v.stride(3),
        query_start_len_ptr=cu, num_seqs=NS, ALL_DECODE=False, SHUFFLED_KV_CACHE=False,
        K_WIDTH=8, BLOCK_M=BM, BLOCK_Q=BQ, TILE_SIZE=TILE,
        num_warps=nw, num_stages=ns, waves_per_eu=wpe,
    )


def launch_glu_2d(q, k, v, out, cu, seqk, bt, scale, BM, TILE, nw, wpe,
                  NUM_SPLITS=1, ALL_DECODE=False, partials=None, MFMA_DIM=32, NUM_BUFFERS=2):
    nt, Hq, D = q.shape; NKV = k.shape[2]; NS = seqk.shape[0]; nqpk = Hq // NKV
    BQ = BM // nqpk
    # ALL_DECODE fast path launches one block per sequence (no BLOCK_Q packing).
    tqb = NS if ALL_DECODE else nt // BQ + NS
    # Decode uses Triton's 3d grid order (seq/q-block fastest-varying); prefill keeps
    # the kv_head-fastest 2d order. The kernel derives the mapping from ALL_DECODE.
    grid = (tqb, NKV, NUM_SPLITS) if ALL_DECODE else (NKV, tqb, NUM_SPLITS)
    pm, pl, pa = (None, None, None) if partials is None else partials
    glu_2d[grid](
        query_ptr=q, key_cache_ptr=k, value_cache_ptr=v, sink_ptr=None, output_ptr=out,
        block_tables_ptr=bt, seq_lens_ptr=seqk, query_start_len_ptr=cu,
        query_stride_0=q.stride(0), query_stride_1=q.stride(1),
        output_stride_0=out.stride(0), output_stride_1=out.stride(1),
        k_descale_ptr=None, v_descale_ptr=None, q_descale_ptr=None, out_scale_ptr=None,
        USE_SINKS=False, SLIDING_WINDOW=0, num_blocks=k.shape[0],
        stride_k_cache_0=k.stride(0), stride_k_cache_1=k.stride(1),
        stride_k_cache_2=k.stride(2), stride_k_cache_3=k.stride(3),
        stride_v_cache_0=v.stride(0), stride_v_cache_1=v.stride(1),
        stride_v_cache_2=v.stride(2), stride_v_cache_3=v.stride(3),
        block_table_stride=bt.stride(0), num_seqs=NS, SCALE=scale,
        NUM_QUERY_HEADS=Hq, NUM_KV_HEADS=NKV, BLOCK_SIZE=k.shape[1],
        TILE_SIZE=TILE, HEAD_SIZE=D, BLOCK_Q=BQ, BLOCK_M=BM,
        ARCH_NAME="gfx950", waves_per_eu=wpe, USE_LOAD_BUFFER_OP=True,
        USE_STORE_BUFFER_OP=True, num_warps=nw, ALL_DECODE=ALL_DECODE,
        CAUSAL=True, REMOVE_INDIRECT_ACCESS=False, NUM_BUFFERS=NUM_BUFFERS, MFMA_DIM=MFMA_DIM,
        NUM_SPLITS=NUM_SPLITS,
        partial_m_ptr=pm, partial_l_ptr=pl, partial_acc_ptr=pa,
    )


def launch_tri_3d(q, k, v, cu, seqk, bt, scale, BM, BQ, TILE, nw, S, wpe, nstg, segm):
    nt, Hq, D = q.shape; NKV = k.shape[2]; NS = seqk.shape[0]; nqpk = Hq // NKV
    so, sm, se = segm
    tri_3d[(NS, NKV, S)](
        segm_output_ptr=so, segm_max_ptr=sm, segm_expsum_ptr=se,
        query_ptr=q, key_cache_ptr=k, value_cache_ptr=v, sink_ptr=None,
        block_tables_ptr=bt, seq_lens_ptr=seqk, alibi_slopes_ptr=None, qq_bias_ptr=None,
        scale=scale, q_descale_ptr=None, k_descale_ptr=None, v_descale_ptr=None,
        out_scale_ptr=None, softcap=0.0, num_query_heads=Hq, num_queries_per_kv=nqpk,
        block_table_stride=bt.stride(0), query_stride_0=q.stride(0), query_stride_1=q.stride(1),
        qq_bias_stride_0=0, BLOCK_SIZE=k.shape[1], HEAD_SIZE=D,
        HEAD_SIZE_PADDED=triton.next_power_of_2(D), USE_ALIBI_SLOPES=False,
        USE_QQ_BIAS=False, USE_SOFTCAP=False, USE_SINKS=False, SLIDING_WINDOW=0,
        stride_k_cache_0=k.stride(0), stride_k_cache_1=k.stride(1),
        stride_k_cache_2=k.stride(2), stride_k_cache_3=k.stride(3),
        stride_v_cache_0=v.stride(0), stride_v_cache_1=v.stride(1),
        stride_v_cache_2=v.stride(2), stride_v_cache_3=v.stride(3),
        query_start_len_ptr=cu, BLOCK_Q=BQ, num_seqs=NS, BLOCK_M=BM,
        ALL_DECODE=True, SHUFFLED_KV_CACHE=False, K_WIDTH=8, IS_Q_FP8=False, IS_KV_FP8=False,
        TILE_SIZE=TILE, NUM_SEGMENTS_PER_SEQ=S, num_warps=nw,
        waves_per_eu=wpe, num_stages=nstg,
    )


def launch_reduce(out, cu, seqk, bt, TILE, S, nw, BQ, segm):
    nt, Hq, D = out.shape; NS = seqk.shape[0]
    so, sm, se = segm
    reduce_segments[(nt, Hq)](
        output_ptr=out, segm_output_ptr=so, segm_max_ptr=sm, segm_expsum_ptr=se,
        seq_lens_ptr=seqk, num_seqs=NS, num_query_heads=Hq, out_scale_ptr=None,
        output_stride_0=out.stride(0), output_stride_1=out.stride(1),
        block_table_stride=bt.stride(0), HEAD_SIZE=D,
        HEAD_SIZE_PADDED=triton.next_power_of_2(D), query_start_len_ptr=cu, BLOCK_Q=BQ,
        TILE_SIZE=TILE, NUM_SEGMENTS_PER_SEQ=S, num_warps=nw, waves_per_eu=2, num_stages=1,
    )


# ---------------------------------------------------------------- profiling
def profile_kernels(fn):
    for _ in range(WARMUP):
        _flush.zero_(); fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(ITERS):
            _flush.zero_(); fn()
        torch.cuda.synchronize()
    res = {}
    for e in prof.key_averages():
        t = getattr(e, "cuda_time_total", 0) or getattr(e, "device_time_total", 0)
        if t:
            res[e.key] = res.get(e.key, 0.0) + float(t)
    return res


def pick(res, *subs):
    return sum(v for k, v in res.items() if any(s in k for s in subs)) / ITERS  # us/iter


# ---------------------------------------------------------------- flops / bytes
def prefill_flops_bytes(B, N, Hq, Hkv):
    pairs = N * (N + 1) / 2
    flops = B * 4 * Hq * HEAD_SIZE * pairs
    e = 2
    byts = B * (2 * N * Hq * HEAD_SIZE * e + 2 * N * Hkv * HEAD_SIZE * e)
    return flops, byts


def decode_flops_bytes(C, ctx, Hq, Hkv):
    flops = 4 * C * Hq * HEAD_SIZE * ctx
    e = 2
    byts = 2 * C * ctx * Hkv * HEAD_SIZE * e + 2 * C * Hq * HEAD_SIZE * e
    return flops, byts


def tflops(flops, us):
    return flops / (us / 1e6) / 1e12


def gbps(byts, us):
    return byts / (us / 1e6) / 1e9


# ---------------------------------------------------------------- runners
def run_prefill(sh):
    B, N, Hq, Hkv = sh["B"], sh["N"], sh["Hq"], sh["Hkv"]
    nqpk = Hq // Hkv
    scale = 1.0 / math.sqrt(HEAD_SIZE)
    BM0 = 16 if nqpk <= 16 else triton.next_power_of_2(nqpk)
    BQ0 = BM0 // nqpk
    nt = B * N
    tqb = nt // BQ0 + B
    c = select_2d_config(64, HEAD_SIZE, 0, False, N, N, nqpk, tqb * Hkv, DT, DT, False)
    BM, BQ, TILE, nw, nstg, wpe = (c["BLOCK_M"], c["BLOCK_Q"], c["TILE_SIZE"],
                                   c["num_warps"], c["num_stages"], c["waves_per_eu"])
    bs = TILE  # gluon needs TILE == block_size

    q = torch.randn(nt, Hq, HEAD_SIZE, dtype=DT, device=DEV)
    k, v, bt = make_paged_kv(N, B, bs, Hkv)
    cu = torch.arange(0, (B + 1) * N, N, dtype=torch.int32, device=DEV)
    seqk = torch.full((B,), N, dtype=torch.int32, device=DEV)
    ot, og = torch.empty_like(q), torch.empty_like(q)

    launch_tri_2d(q, k, v, ot, cu, seqk, bt, scale, BM, BQ, TILE, nw, nstg, wpe)
    # gluon: BLOCK_M must be 32*num_warps; heuristic large-prefill gives 128/4 -> ok
    g_nw = nw; g_BM = BM
    assert g_BM == 32 * g_nw, f"gluon needs BLOCK_M=32*num_warps, got {g_BM}/{g_nw}"
    launch_glu_2d(q, k, v, og, cu, seqk, bt, scale, g_BM, TILE, g_nw, wpe)
    torch.cuda.synchronize()
    xcheck = (ot.float() - og.float()).abs().max().item()

    flops, byts = prefill_flops_bytes(B, N, Hq, Hkv)
    t_us = pick(profile_kernels(lambda: launch_tri_2d(q, k, v, ot, cu, seqk, bt, scale, BM, BQ, TILE, nw, nstg, wpe)), "unified_attention")
    g_us = pick(profile_kernels(lambda: launch_glu_2d(q, k, v, og, cu, seqk, bt, scale, g_BM, TILE, g_nw, wpe)), "unified_attention")
    row = dict(
        shape=f"b{B} {N}/{N} {Hq}/{Hkv}", cfg=f"BM {BM}/{g_BM} TILE {TILE} nw {nw}/{g_nw}",
        t_us=t_us, t_tf=tflops(flops, t_us), g_us=g_us, g_tf=tflops(flops, g_us),
        speedup=t_us / g_us, xcheck=xcheck,
    )
    del q, k, v, bt, ot, og; torch.cuda.empty_cache()
    return row


def run_decode(sh):
    C, ctx, Hq, Hkv = sh["C"], sh["ctx"], sh["Hq"], sh["Hkv"]
    nqpk = Hq // Hkv
    scale = 1.0 / math.sqrt(HEAD_SIZE)
    BM0 = 16 if nqpk <= 16 else triton.next_power_of_2(nqpk)
    BQ0 = BM0 // nqpk
    attn, red = select_3d_config(HEAD_SIZE, 64, ctx, TARGET_PRGMS, C * Hkv, DT, DT, False, 1, 0)
    TILE, S, a_nw, wpe, nstg = (attn["TILE_SIZE"], attn["NUM_SEGMENTS_PER_SEQ"],
                                attn["num_warps"], attn["waves_per_eu"], attn["num_stages"])
    r_nw = red["num_warps"]
    bs = TILE
    # gluon decode: minimal M-tile (BLOCK_M=32, num_warps=1); TILE & S shared w/ triton
    # decode: 16x16 MFMA, BLOCK_M=16 (matches Triton's BLOCK_M; best on all decode shapes)
    g_nw, g_BM, g_mfma = 1, 16, 16
    g_BQ = g_BM // nqpk

    q = torch.randn(C, Hq, HEAD_SIZE, dtype=DT, device=DEV)
    k, v, bt = make_paged_kv(ctx, C, bs, Hkv)
    cu = torch.arange(0, C + 1, dtype=torch.int32, device=DEV)
    seqk = torch.full((C,), ctx, dtype=torch.int32, device=DEV)

    # ---- correctness cross-check: both produce correct output ----
    segm_t = alloc_segm(C, Hq, S)
    launch_tri_3d(q, k, v, cu, seqk, bt, scale, BM0, BQ0, TILE, a_nw, S, wpe, nstg, segm_t)
    ot = torch.empty_like(q)
    launch_reduce(ot, cu, seqk, bt, TILE, S, r_nw, BQ0, segm_t)
    so, sm, se = alloc_segm(C, Hq, S)
    launch_glu_2d(q, k, v, q.clone(), cu, seqk, bt, scale, g_BM, TILE, g_nw, wpe,
                  NUM_SPLITS=S, ALL_DECODE=True, partials=(sm, se, so), MFMA_DIM=g_mfma, NUM_BUFFERS=1)
    og = torch.empty_like(q)
    launch_reduce(og, cu, seqk, bt, TILE, S, r_nw, g_BQ, (so, sm * (RCP_LN2 * scale), se))
    torch.cuda.synchronize()
    xcheck = (ot.float() - og.float()).abs().max().item()

    flops, byts = decode_flops_bytes(C, ctx, Hq, Hkv)

    def tri_fn():
        launch_tri_3d(q, k, v, cu, seqk, bt, scale, BM0, BQ0, TILE, a_nw, S, wpe, nstg, segm_t)
        launch_reduce(ot, cu, seqk, bt, TILE, S, r_nw, BQ0, segm_t)

    def glu_fn():
        # partials = (partial_m=max, partial_l=expsum, partial_acc=acc) = (sm, se, so)
        # reduce consumes (segm_output=acc, segm_max=max, segm_expsum=expsum) = (so, sm, se)
        launch_glu_2d(q, k, v, og, cu, seqk, bt, scale, g_BM, TILE, g_nw, wpe,
                      NUM_SPLITS=S, ALL_DECODE=True, partials=(sm, se, so), MFMA_DIM=g_mfma, NUM_BUFFERS=1)
        launch_reduce(og, cu, seqk, bt, TILE, S, r_nw, g_BQ, (so, sm, se))

    rt = profile_kernels(tri_fn)
    ta, tr = pick(rt, "unified_attention_3d"), pick(rt, "reduce_segments")
    rg = profile_kernels(glu_fn)
    ga, gr = pick(rg, "unified_attention_2d"), pick(rg, "reduce_segments")

    row = dict(
        shape=f"C{C} ctx{ctx} {Hq}/{Hkv}", S=S,
        cfg=f"BM {BM0}/{g_BM} nw {a_nw}/{g_nw}",
        t_attn=ta, t_red=tr, t_tot=ta + tr, t_gb=gbps(byts, ta + tr),
        g_attn=ga, g_red=gr, g_tot=ga + gr, g_gb=gbps(byts, ga + gr),
        speedup=(ta + tr) / (ga + gr), xcheck=xcheck,
    )
    del q, k, v, bt, ot, og, segm_t, so, sm, se; torch.cuda.empty_cache()
    return row


# ---------------------------------------------------------------- md
def write_md(pre_rows, dec_rows):
    L = []
    L.append("# Unified Attention: Triton vs gfx950 Gluon\n")
    L.append(f"- device: **gfx950**, {CU} CUs · dtype **bf16** · HEAD_SIZE={HEAD_SIZE} · causal")
    L.append(f"- config from Triton heuristics (`select_2d_config` / `select_3d_config`); "
             f"same TILE_SIZE / NUM_SEGMENTS on both sides")
    L.append(f"- decode: split-KV attn + Triton `reduce_segments` (identical reduce both sides); "
             f"gluon uses minimal M-tile BLOCK_M=32/num_warps=1")
    L.append(f"- profiler: {ITERS} iters, {WARMUP} warmup, {FLUSH_MB} MB L2 flush per iter; "
             f"speedup = triton_time / gluon_time (>1 = gluon faster)")
    L.append("- `cfg` column shows `<triton>/<gluon>` where they differ (BLOCK_M, num_warps)\n")

    L.append("## Prefill  (TFLOP/s = headline, compute-bound)\n")
    L.append("| shape | cfg | Triton us | Triton TFLOP/s | Gluon us | Gluon TFLOP/s | speedup | xcheck |")
    L.append("|---|---|--:|--:|--:|--:|--:|--:|")
    for r in pre_rows:
        L.append(f"| {r['shape']} | {r['cfg']} | {r['t_us']:.1f} | {r['t_tf']:.0f} | "
                 f"{r['g_us']:.1f} | {r['g_tf']:.0f} | **{r['speedup']:.2f}x** | {r['xcheck']:.1e} |")

    L.append("\n## Decode  (GB/s = headline, memory-bound; attn+reduce total)\n")
    L.append("| shape | S | cfg | Triton attn/red/tot us | Triton GB/s | Gluon attn/red/tot us | Gluon GB/s | speedup | xcheck |")
    L.append("|---|--:|---|--:|--:|--:|--:|--:|--:|")
    for r in dec_rows:
        L.append(f"| {r['shape']} | {r['S']} | {r['cfg']} | "
                 f"{r['t_attn']:.1f}/{r['t_red']:.1f}/{r['t_tot']:.1f} | {r['t_gb']:.0f} | "
                 f"{r['g_attn']:.1f}/{r['g_red']:.1f}/{r['g_tot']:.1f} | {r['g_gb']:.0f} | "
                 f"**{r['speedup']:.2f}x** | {r['xcheck']:.1e} |")

    L.append("\n_xcheck = max abs diff between gluon and triton output (bf16, expect ~1e-2)._")
    with open(OUT_MD, "w") as f:
        f.write("\n".join(L) + "\n")


def main():
    torch.manual_seed(0)
    print(f"gfx950 {CU} CUs | prefill={len(PREFILL_SHAPES)} decode={len(DECODE_SHAPES)} shapes")
    pre_rows = []
    for sh in PREFILL_SHAPES:
        r = run_prefill(sh)
        pre_rows.append(r)
        print(f"[prefill] {r['shape']:18s} T {r['t_us']:7.1f}us {r['t_tf']:4.0f}TF | "
              f"G {r['g_us']:7.1f}us {r['g_tf']:4.0f}TF | {r['speedup']:.2f}x  xc {r['xcheck']:.1e}")
    dec_rows = []
    for sh in DECODE_SHAPES:
        r = run_decode(sh)
        dec_rows.append(r)
        print(f"[decode ] {r['shape']:16s} S{r['S']:<3d} T {r['t_tot']:6.1f}us {r['t_gb']:5.0f}GB/s | "
              f"G {r['g_tot']:6.1f}us {r['g_gb']:5.0f}GB/s | {r['speedup']:.2f}x  xc {r['xcheck']:.1e}")
    write_md(pre_rows, dec_rows)
    print(f"\nwrote {OUT_MD}")


if __name__ == "__main__":
    main()
