"""Decode/prefill MFMA tile choice (16x16x32 vs 32x32x16, bf16) across the GQA ratio.

The rest of the MFMA study only ever measures NUM_QUERIES_PER_KV = 8 (both `64/8` and
`8/1` are nqpk=8), which is one point on the axis that actually drives the choice. The M
tile holds `nqpk` valid rows out of `BLOCK_M` in decode, and the wrapper sizes the tile as

    block_m   = max(MFMA_DIM, next_pow2(nqpk))
    num_warps = block_m // MFMA_DIM

so the two shapes trade off differently at each end of the range:

    nqpk    BLOCK_M/nw  16x16      BLOCK_M/nw  32x32     valid M rows
     1        16 / 1                 32 / 1              1/16  vs 1/32   (MHA)
     8        16 / 1                 32 / 1              8/16  vs 8/32
    16        16 / 1                 32 / 1             16/16  vs 16/32
    32        32 / 2                 32 / 1             32/32  vs 32/32   <- 16x16 pays 2 warps

i.e. 16x16 should win by the most at MHA and should *lose* once nqpk reaches 32, where both
tiles are fully packed but 16x16 needs two warps to get there. This walks nqpk in
{1,2,4,8,16,32} at fixed NUM_QUERY_HEADS=64 (so Hkv varies, and with it the KV traffic --
that is the real MHA-vs-GQA tradeoff, not an artefact) and reports both tiles per cell.

Prefill is included at nqpk 1 and 8: BLOCK_Q = BLOCK_M/nqpk tokens keep the M tile 100% full
either way, so the prediction is that prefill is nqpk-independent and stays 32x32.

    UA_HEAD_SIZE={64,128} python mfma_nqpk_scan.py
"""
import sys, os, math, json
import torch, triton
sys.path.insert(0, "/app/aiter/bench_gluon_ua")
import bench_ua as B

DEV, HS, RCP = B.DEV, B.HEAD_SIZE, B.RCP_LN2
TILE = 64
VER = triton.__version__.split("+")[0]
WPE_DEC = 1 if HS < 128 else 2       # kernel wrapper's own values (see tune_hs64.py)
WPE_PRE = 3 if HS < 128 else 2
# Grids are env-overridable so a subset can be smoke-tested without touching the file:
#   NQPK=1,32  DEC=16x1024  PRE=8x1024  python mfma_nqpk_scan.py
NQPK = [int(x) for x in os.environ.get("NQPK", "1,2,4,8,16,32").split(",")]
HQ = int(os.environ.get("HQ", 64))
DEC_SHAPES = [tuple(int(v) for v in s.split("x"))
              for s in os.environ.get(
                  "DEC", "1x1024,16x1024,128x1024,1x8192,16x8192,128x8192").split(",") if s]
PRE_SHAPES = [tuple(int(v) for v in s.split("x"))
              for s in os.environ.get("PRE", "8x1024,1x8192").split(",") if s]
PRE_NQPK = [int(x) for x in os.environ.get("PRE_NQPK", "1,8").split(",")]
torch.manual_seed(0)


def tile_of(mfma, nqpk):
    """The wrapper's own tile rule, so each cell is a choice the heuristic could make."""
    block_m = max(mfma, triton.next_power_of_2(nqpk))
    return block_m, block_m // mfma


# ------------------------------------------------------------------ decode
def dec_ref(q, k, v, cu, seqk, bt, C, ctx, Hq, Hkv):
    """Triton 3d + reduce on the same inputs -> correctness baseline."""
    nqpk = Hq // Hkv
    scale = 1.0 / math.sqrt(HS)
    BM0 = 16 if nqpk <= 16 else triton.next_power_of_2(nqpk)
    BQ0 = BM0 // nqpk
    a, r = B.select_3d_config(HS, TILE, ctx, B.TARGET_PRGMS, C * Hkv,
                              torch.bfloat16, torch.bfloat16, False, 1, 0)
    S = a["NUM_SEGMENTS_PER_SEQ"]
    segm = B.alloc_segm(C, Hq, S)
    o = torch.empty(C, Hq, HS, dtype=torch.bfloat16, device=DEV)
    B.launch_tri_3d(q, k, v, cu, seqk, bt, scale, BM0, BQ0, TILE, a["num_warps"], S,
                    a["waves_per_eu"], a["num_stages"], segm)
    B.launch_reduce(o, cu, seqk, bt, TILE, S, r["num_warps"], BQ0, segm)
    torch.cuda.synchronize()
    del segm
    return o


def dec_run(q, k, v, cu, seqk, bt, C, ctx, Hq, Hkv, mfma, ref):
    nqpk = Hq // Hkv
    scale = 1.0 / math.sqrt(HS)
    BM, nw = tile_of(mfma, nqpk)
    o = torch.empty(C, Hq, HS, dtype=torch.bfloat16, device=DEV)
    Sg = B.select_gluon_num_splits(C, Hkv, ctx // TILE)
    try:
        if Sg == 1:
            def f():
                B.launch_glu_2d(q, k, v, o, cu, seqk, bt, scale, BM, TILE, nw, WPE_DEC,
                                NUM_SPLITS=1, ALL_DECODE=True, MFMA_DIM=mfma, NUM_BUFFERS=2)
            f(); torch.cuda.synchronize()
            # rel must be read here: in the split path below the timing loop leaves `o`
            # un-rescaled, so measuring it after profiling reports a bogus ~190% error.
            rel = ((ref.float() - o.float()).abs().mean() / ref.float().abs().mean()).item()
            t = B.pick(B.profile_kernels(f), "unified_attention_2d")
        else:
            so, sm, se = B.alloc_segm(C, Hq, Sg)
            BQ = BM // nqpk
            # correctness pass: the reduce consumes ln-domain max, the kernel emits log2
            B.launch_glu_2d(q, k, v, o, cu, seqk, bt, scale, BM, TILE, nw, WPE_DEC,
                            NUM_SPLITS=Sg, ALL_DECODE=True, partials=(sm, se, so),
                            MFMA_DIM=mfma, NUM_BUFFERS=2)
            B.launch_reduce(o, cu, seqk, bt, TILE, Sg, 2, BQ, (so, sm, se))
            torch.cuda.synchronize()
            rel = ((ref.float() - o.float()).abs().mean() / ref.float().abs().mean()).item()

            def f():
                B.launch_glu_2d(q, k, v, o, cu, seqk, bt, scale, BM, TILE, nw, WPE_DEC,
                                NUM_SPLITS=Sg, ALL_DECODE=True, partials=(sm, se, so),
                                MFMA_DIM=mfma, NUM_BUFFERS=2)
                B.launch_reduce(o, cu, seqk, bt, TILE, Sg, 2, BQ, (so, sm, se))
            r = B.profile_kernels(f)
            t = B.pick(r, "unified_attention_2d") + B.pick(r, "reduce_segments")
            del so, sm, se
        return dict(time_us=t, rel=rel, block_m=BM, num_warps=nw, splits=Sg)
    except Exception as exc:
        return dict(error=f"{type(exc).__name__}: {str(exc)[:90]}", block_m=BM,
                    num_warps=nw, splits=Sg)
    finally:
        del o
        torch.cuda.empty_cache()


# ------------------------------------------------------------------ prefill
def pre_run(q, k, v, cu, seqk, bt, bs, N, Hq, Hkv, mfma, ref):
    nqpk = Hq // Hkv
    scale = 1.0 / math.sqrt(HS)
    BM, nw = 128, 128 // mfma          # prefill pins BLOCK_M=128 for both tiles
    o = torch.empty(bs * N, Hq, HS, dtype=torch.bfloat16, device=DEV)
    try:
        def f():
            B.launch_glu_2d(q, k, v, o, cu, seqk, bt, scale, BM, TILE, nw, WPE_PRE,
                            MFMA_DIM=mfma, NUM_BUFFERS=2)
        f(); torch.cuda.synchronize()
        t = B.pick(B.profile_kernels(f), "unified_attention")
        rel = ((ref.float() - o.float()).abs().mean() / ref.float().abs().mean()).item()
        return dict(time_us=t, rel=rel, block_m=BM, num_warps=nw)
    except Exception as exc:
        return dict(error=f"{type(exc).__name__}: {str(exc)[:90]}", block_m=BM, num_warps=nw)
    finally:
        del o
        torch.cuda.empty_cache()


def pre_ref(q, k, v, cu, seqk, bt, bs, N, Hq, Hkv):
    nqpk = Hq // Hkv
    scale = 1.0 / math.sqrt(HS)
    c = B.select_2d_config(TILE, HS, 0, False, N, N, nqpk, 1, torch.bfloat16,
                           torch.bfloat16, False)
    o = torch.empty(bs * N, Hq, HS, dtype=torch.bfloat16, device=DEV)
    B.launch_tri_2d(q, k, v, o, cu, seqk, bt, scale, c["BLOCK_M"], c["BLOCK_Q"], TILE,
                    c["num_warps"], c["num_stages"], c["waves_per_eu"])
    torch.cuda.synchronize()
    return o


recs = []
print(f"=== nqpk MFMA tile scan | triton {VER} | HEAD_SIZE={HS} | {B.CU} CUs | "
      f"wpe dec{WPE_DEC}/pre{WPE_PRE} ===", flush=True)

print("\n--- decode (Hq=64, Hkv=64/nqpk) ---", flush=True)
for (C, ctx) in DEC_SHAPES:
    for nqpk in NQPK:
        Hkv = HQ // nqpk
        q = torch.randn(C, HQ, HS, dtype=torch.bfloat16, device=DEV)
        k, v, bt = B.make_paged_kv(ctx, C, TILE, Hkv, dtype=torch.bfloat16)
        cu = torch.arange(0, C + 1, dtype=torch.int32, device=DEV)
        seqk = torch.full((C,), ctx, dtype=torch.int32, device=DEV)
        ref = dec_ref(q, k, v, cu, seqk, bt, C, ctx, HQ, Hkv)
        cells = {}
        for mfma in (16, 32):
            cells[mfma] = dec_run(q, k, v, cu, seqk, bt, C, ctx, HQ, Hkv, mfma, ref)
        _, byts = B.decode_flops_bytes(C, ctx, HQ, Hkv)
        rec = dict(ver=VER, head_size=HS, phase="decode", C=C, ctx=ctx, Hq=HQ, Hkv=Hkv,
                   nqpk=nqpk, label=f"C{C} ctx{ctx} {HQ}/{Hkv}", bytes=byts,
                   r16=cells[16], r32=cells[32])
        recs.append(rec)
        a, b = cells[16], cells[32]
        if "error" in a or "error" in b:
            def cell(x):
                return x["error"] if "error" in x else f"{x['time_us']:.1f}us"
            print(f"  nqpk{nqpk:<3d} C{C:<4d} ctx{ctx:<5d} {HQ}/{Hkv:<3d}: "
                  f"16x16 {cell(a)} | 32x32 {cell(b)}", flush=True)
        else:
            g = a["time_us"] / b["time_us"]
            tag = ("16x16 +%.0f%%" % ((1 / g - 1) * 100) if g < 0.98 else
                   ("32x32 +%.0f%%" % ((g - 1) * 100) if g > 1.02 else "tie"))
            print(f"  nqpk{nqpk:<3d} C{C:<4d} ctx{ctx:<5d} {HQ}/{Hkv:<3d}: "
                  f"BM{a['block_m']}/nw{a['num_warps']} {a['time_us']:8.1f}  "
                  f"BM{b['block_m']}/nw{b['num_warps']} {b['time_us']:8.1f}  "
                  f"{tag:<12s} S{a['splits']}  "
                  f"{B.gbps(byts, a['time_us']):5.0f}/{B.gbps(byts, b['time_us']):5.0f} GB/s  "
                  f"rel {a['rel']:.1%}/{b['rel']:.1%}", flush=True)
        del q, k, v, bt, ref
        torch.cuda.empty_cache()

print("\n--- prefill (Hq=64, Hkv=64/nqpk) ---", flush=True)
for (bs, N) in PRE_SHAPES:
    for nqpk in PRE_NQPK:
        Hkv = HQ // nqpk
        q = torch.randn(bs * N, HQ, HS, dtype=torch.bfloat16, device=DEV)
        k, v, bt = B.make_paged_kv(N, bs, TILE, Hkv, dtype=torch.bfloat16)
        cu = torch.arange(0, (bs + 1) * N, N, dtype=torch.int32, device=DEV)
        seqk = torch.full((bs,), N, dtype=torch.int32, device=DEV)
        ref = pre_ref(q, k, v, cu, seqk, bt, bs, N, HQ, Hkv)
        cells = {m: pre_run(q, k, v, cu, seqk, bt, bs, N, HQ, Hkv, m, ref) for m in (16, 32)}
        flops, _ = B.prefill_flops_bytes(bs, N, HQ, Hkv)
        recs.append(dict(ver=VER, head_size=HS, phase="prefill", B=bs, N=N, Hq=HQ, Hkv=Hkv,
                         nqpk=nqpk, label=f"b{bs} {N} {HQ}/{Hkv}", flops=flops,
                         r16=cells[16], r32=cells[32]))
        a, b = cells[16], cells[32]
        if "error" in a or "error" in b:
            print(f"  nqpk{nqpk:<3d} b{bs} N{N:<5d} {HQ}/{Hkv:<3d}: FAIL "
                  f"{a.get('error', '')} {b.get('error', '')}", flush=True)
        else:
            g = a["time_us"] / b["time_us"]
            tag = ("16x16 +%.0f%%" % ((1 / g - 1) * 100) if g < 0.98 else
                   ("32x32 +%.0f%%" % ((g - 1) * 100) if g > 1.02 else "tie"))
            print(f"  nqpk{nqpk:<3d} b{bs} N{N:<5d} {HQ}/{Hkv:<3d}: "
                  f"16x16 {a['time_us']:8.1f}  32x32 {b['time_us']:8.1f}  {tag:<12s} "
                  f"{B.tflops(flops, a['time_us']):4.0f}/{B.tflops(flops, b['time_us']):4.0f} TF  "
                  f"rel {a['rel']:.1%}/{b['rel']:.1%}", flush=True)
        del q, k, v, bt, ref
        torch.cuda.empty_cache()

out = os.environ.get("NQPK_OUT",
                     f"/app/aiter/bench_gluon_ua/mfma_shape_study_7_28/nqpk_hs{HS}_{VER}.json")
json.dump(recs, open(out, "w"), indent=0)
print("\nwrote", out)
