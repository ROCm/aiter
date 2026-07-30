"""Full decode + prefill perf scan (bf16 and fp8) for the CURRENTLY-INSTALLED triton.

Grid:
  decode  : C in {16,32,64,128} x ctx in {1024,8192} x heads in {64/8 GQA, 8/1 MQA}
  prefill : (B,N) in {(1,1024),(4,1024),(8,1024),(1,8192)} x the same two head configs
  dtypes  : bf16 and fp8 (q, k and v all e4m3; output stays bf16)

Both impls per cell: triton (3d attn + reduce for decode, 2d for prefill) and gluon
(2d; decode right-sizes the split, S=1 => non-split, no reduce). Records time (us/iter),
TFLOP/s, GB/s. Writes scanfull_<major.minor.patch>.json into $SCAN_DIR (default: this
folder). Run once per triton version (PYTHONPATH-shadow or dir-swap between).

gluon config follows the gfx950 gluon wrapper's own heuristics:
  decode  : num_warps=1, BLOCK_M=16, MFMA_DIM=16
  prefill : num_warps=4, BLOCK_M=128, MFMA_DIM=32
  waves_per_eu: 2 (the wrapper's bf16 value, used for both dtypes here). The wrapper's fp8
  branch picks 3, which is neutral on 3.6/3.7 but collapses on 3.8.0 (up to 2.3x slower);
  GLU_FP8_WPE=3 reproduces it for the sensitivity comparison.
Exception: **fp8 decode falls back to MFMA_DIM=32 / BLOCK_M=32**. The wrapper's 16x16
decode tile does not lower with fp8 operands — `ConvertTritonAMDGPUToLLVM` fails
("PassManager::run failed") on 3.6.0, 3.7.0 and 3.8.0 alike, for nb=1 and nb=2 and for
BLOCK_M 16 or 32. 32x32 is the smallest fp8 decode tile that compiles.
NUM_BUFFERS=2 everywhere (the wrapper prefers 1 for decode; 2 keeps all three triton
versions and both dtypes apples-to-apples, as the bf16-only decode scan already did).
"""
import os, sys, math, json, traceback
import triton
sys.path.insert(0, "/app/aiter/bench_gluon_ua")
import torch
import bench_ua as B
from aiter.ops.triton.utils.types import e4m3_dtype

DEV, HS, RCP = B.DEV, B.HEAD_SIZE, B.RCP_LN2
VER_FULL = triton.__version__
VER = VER_FULL.split("+")[0]
TILE = 64
OUT_DT = torch.bfloat16          # output stays bf16 in every configuration
DTYPES = [("bf16", torch.bfloat16), ("fp8", e4m3_dtype)]
# Overrides for the fp8 waves_per_eu sensitivity pass:
#   GLU_FP8_WPE=<n>       gluon waves_per_eu on the fp8 rows, both phases (overrides the
#                         defaults below; 3 = the wrapper's old shipped fp8 heuristic)
#   GLU_WPE_DEC / GLU_WPE_PRE=<n>   per-phase waves_per_eu override
#   ONLY_DTYPE=fp8        restrict the scan to one dtype
#   OUT_SUFFIX=_fp8wpe3   suffix for the output json name
ONLY_DTYPE = os.environ.get("ONLY_DTYPE")
if ONLY_DTYPE:
    DTYPES = [(n, d) for (n, d) in DTYPES if n == ONLY_DTYPE]
torch.manual_seed(0)

DECODE_SHAPES = [(C, ctx, Hq, Hkv) for ctx in (1024, 8192)
                 for (Hq, Hkv) in ((64, 8), (8, 1)) for C in (16, 32, 64, 128)]
PREFILL_SHAPES = [(b, n, Hq, Hkv) for (b, n) in ((1, 1024), (4, 1024), (8, 1024), (1, 8192))
                  for (Hq, Hkv) in ((64, 8), (8, 1))]
records = []


def gluon_wpe(phase, dt):
    """gluon waves_per_eu, tuned per HEAD_SIZE and phase (see tune_hs64.py).

    HEAD_SIZE>=128: 2 for both phases/dtypes (the value the gfx950 wrapper picks for bf16).
    HEAD_SIZE<128 (calibrated at 64): decode 1, prefill 3. wpe=4 — what the wrapper picks
    below 128 — is the worst of {1,2,3,4} on 3.7/3.8 (fp8 prefill +51% on 3.8).
    """
    if dt.itemsize == 1 and os.environ.get("GLU_FP8_WPE"):
        return int(os.environ["GLU_FP8_WPE"])
    env = os.environ.get("GLU_WPE_DEC" if phase == "decode" else "GLU_WPE_PRE")
    if env:
        return int(env)
    if HS < 128:
        return 1 if phase == "decode" else 3
    return 2


def fp8_bits(dt, phase):
    """Per-dtype knobs: (element bytes, descale triple, triton K_WIDTH, gluon waves_per_eu)."""
    if dt.itemsize == 1:
        one = torch.ones(1, dtype=torch.float32, device=DEV)
        # fixed at 1.0: descale magnitude does not affect kernel time, and identical
        # descales on both sides keep the gluon-vs-triton cross-check meaningful.
        return 1, (one, one.clone(), one.clone()), 16, gluon_wpe(phase, dt)
    return 2, (None, None, None), 8, gluon_wpe(phase, dt)


def gluon_decode_tile(dt):
    """(BLOCK_M, MFMA_DIM) for the gluon decode kernel.

    bf16 uses the wrapper's 16x16 minimal M-tile. fp8 must fall back to 32x32: the
    16x16 fp8 dot fails to lower (ConvertTritonAMDGPUToLLVM) on all three versions.
    """
    return (32, 32) if dt.itemsize == 1 else (16, 16)


def add(phase, dname, label, n1, n2, Hq, Hkv, impl, split, tot, at, rd, flops, byts, xc, cfg):
    records.append(dict(ver=VER, ver_full=VER_FULL, head_size=HS, phase=phase, dtype=dname, label=label,
                        n1=n1, n2=n2, Hq=Hq, Hkv=Hkv, impl=impl, split=split,
                        time_us=tot, attn_us=at, red_us=rd,
                        tflops=B.tflops(flops, tot), gbps=B.gbps(byts, tot), xc=xc, cfg=cfg))


# ------------------------------------------------------------------ decode
def decode_one(C, ctx, Hq, Hkv, dname, dt):
    nqpk = Hq // Hkv; scale = 1.0 / math.sqrt(HS); num_tiles = ctx // TILE
    e, desc, kw, g_wpe = fp8_bits(dt, "decode")
    is_fp8 = dt.itemsize == 1
    label = f"C{C} ctx{ctx} {Hq}/{Hkv}"

    q = (torch.randn(C, Hq, HS, dtype=torch.float32, device=DEV)).to(dt)
    k, v, bt = B.make_paged_kv(ctx, C, TILE, Hkv, dtype=dt)
    cu = torch.arange(0, C + 1, dtype=torch.int32, device=DEV)
    seqk = torch.full((C,), ctx, dtype=torch.int32, device=DEV)
    flops, byts = B.decode_flops_bytes(C, ctx, Hq, Hkv, e=e)

    # ---- triton: 3d attn + reduce at the heuristic split ----
    attn, red = B.select_3d_config(HS, TILE, ctx, B.TARGET_PRGMS, C * Hkv, dt, dt, False, 1, 0)
    assert attn["TILE_SIZE"] == TILE, f"heuristic TILE {attn['TILE_SIZE']} != {TILE}"
    S = attn["NUM_SEGMENTS_PER_SEQ"]; a_nw = attn["num_warps"]; wpe = attn["waves_per_eu"]
    nstg = attn["num_stages"]; r_nw = red["num_warps"]
    seg = B.alloc_segm(C, Hq, S); ot = torch.empty(C, Hq, HS, dtype=OUT_DT, device=DEV)

    def tf():
        B.launch_tri_3d(q, k, v, cu, seqk, bt, scale, 16, 16 // nqpk, TILE, a_nw, S, wpe, nstg, seg,
                        descales=desc, K_WIDTH=kw, IS_Q_FP8=is_fp8, IS_KV_FP8=is_fp8)
        B.launch_reduce(ot, cu, seqk, bt, TILE, S, r_nw, 16 // nqpk, seg)
    rt = B.profile_kernels(tf)
    t_attn = B.pick(rt, "unified_attention_3d"); t_red = B.pick(rt, "reduce_segments")
    t_tot = t_attn + t_red

    # ---- gluon: 2d at the right-sized split (S=1 => non-split, no reduce) ----
    Sg = B.select_gluon_num_splits(C, Hkv, num_tiles)
    g_BM, g_mfma = gluon_decode_tile(dt)
    og = torch.empty(C, Hq, HS, dtype=OUT_DT, device=DEV)
    if Sg == 1:
        def gf():
            B.launch_glu_2d(q, k, v, og, cu, seqk, bt, scale, g_BM, TILE, 1, g_wpe,
                            NUM_SPLITS=1, ALL_DECODE=True, MFMA_DIM=g_mfma, NUM_BUFFERS=2,
                            descales=desc)
        gf(); torch.cuda.synchronize()
        xc = (ot.float() - og.float()).abs().max().item()
        rg = B.profile_kernels(gf)
        g_attn = B.pick(rg, "unified_attention_2d"); g_red = 0.0
    else:
        so, sm, se = B.alloc_segm(C, Hq, Sg)
        # correctness pass (prescaled M); timed pass uses raw M -> identical reduce cost
        B.launch_glu_2d(q, k, v, og, cu, seqk, bt, scale, g_BM, TILE, 1, g_wpe,
                        NUM_SPLITS=Sg, ALL_DECODE=True, partials=(sm, se, so), MFMA_DIM=g_mfma,
                        NUM_BUFFERS=2, descales=desc)
        B.launch_reduce(og, cu, seqk, bt, TILE, Sg, r_nw, 16 // nqpk, (so, sm, se))
        torch.cuda.synchronize()
        xc = (ot.float() - og.float()).abs().max().item()

        def gf():
            B.launch_glu_2d(q, k, v, og, cu, seqk, bt, scale, g_BM, TILE, 1, g_wpe,
                            NUM_SPLITS=Sg, ALL_DECODE=True, partials=(sm, se, so), MFMA_DIM=g_mfma,
                            NUM_BUFFERS=2, descales=desc)
            B.launch_reduce(og, cu, seqk, bt, TILE, Sg, r_nw, 16 // nqpk, (so, sm, se))
        rg = B.profile_kernels(gf)
        g_attn = B.pick(rg, "unified_attention_2d"); g_red = B.pick(rg, "reduce_segments")
    g_tot = g_attn + g_red

    add("decode", dname, label, C, ctx, Hq, Hkv, "triton", S, t_tot, t_attn, t_red,
        flops, byts, xc, f"BM16 nw{a_nw} wpe{wpe} TILE{TILE}")
    add("decode", dname, label, C, ctx, Hq, Hkv, "gluon", Sg, g_tot, g_attn, g_red,
        flops, byts, xc, f"BM{g_BM} nw1 mfma{g_mfma} nb2 wpe{g_wpe} TILE{TILE}")
    print(f"[dec {dname:4s}] {label:18s}: "
          f"tri {t_tot:7.1f}us {B.gbps(byts, t_tot):5.0f}GB/s {B.tflops(flops, t_tot):4.0f}TF S{S:<3} | "
          f"glu {g_tot:7.1f}us {B.gbps(byts, g_tot):5.0f}GB/s {B.tflops(flops, g_tot):4.0f}TF S{Sg:<3} "
          f"{t_tot / g_tot:.2f}x xc{xc:.0e}", flush=True)
    del q, k, v, bt, ot, og, seg; torch.cuda.empty_cache()


# ------------------------------------------------------------------ prefill
def prefill_one(bs, N, Hq, Hkv, dname, dt):
    nqpk = Hq // Hkv; scale = 1.0 / math.sqrt(HS)
    e, desc, kw, g_wpe = fp8_bits(dt, "prefill")
    label = f"b{bs} {N} {Hq}/{Hkv}"
    nt = bs * N

    BM0 = 16 if nqpk <= 16 else triton.next_power_of_2(nqpk)
    tqb0 = nt // (BM0 // nqpk) + bs
    c = B.select_2d_config(TILE, HS, 0, False, N, N, nqpk, tqb0 * Hkv, dt, dt, False)
    BM, BQ, T, nw, nstg, wpe = (c["BLOCK_M"], c["BLOCK_Q"], c["TILE_SIZE"],
                                c["num_warps"], c["num_stages"], c["waves_per_eu"])
    assert T == TILE, f"heuristic TILE {T} != {TILE} (gluon needs TILE == block_size)"

    q = (torch.randn(nt, Hq, HS, dtype=torch.float32, device=DEV)).to(dt)
    k, v, bt = B.make_paged_kv(N, bs, TILE, Hkv, dtype=dt)
    cu = torch.arange(0, (bs + 1) * N, N, dtype=torch.int32, device=DEV)
    seqk = torch.full((bs,), N, dtype=torch.int32, device=DEV)
    ot = torch.empty(nt, Hq, HS, dtype=OUT_DT, device=DEV)
    og = torch.empty_like(ot)
    flops, byts = B.prefill_flops_bytes(bs, N, Hq, Hkv, e=e)

    # gluon prefill: the wrapper's own choice (BLOCK_M=128 / num_warps=4 / 32x32 MFMA),
    # which is what the triton large-prefill heuristic lands on too.
    g_nw, g_BM = nw, BM
    assert g_BM == 32 * g_nw, f"gluon needs BLOCK_M=32*num_warps, got {g_BM}/{g_nw}"

    def tf():
        B.launch_tri_2d(q, k, v, ot, cu, seqk, bt, scale, BM, BQ, TILE, nw, nstg, wpe,
                        descales=desc, K_WIDTH=kw)

    def gf():
        B.launch_glu_2d(q, k, v, og, cu, seqk, bt, scale, g_BM, TILE, g_nw, g_wpe,
                        MFMA_DIM=32, NUM_BUFFERS=2, descales=desc)

    tf(); gf(); torch.cuda.synchronize()
    xc = (ot.float() - og.float()).abs().max().item()
    t_tot = B.pick(B.profile_kernels(tf), "unified_attention")
    g_tot = B.pick(B.profile_kernels(gf), "unified_attention")

    add("prefill", dname, label, bs, N, Hq, Hkv, "triton", 1, t_tot, t_tot, 0.0,
        flops, byts, xc, f"BM{BM} nw{nw} wpe{wpe} TILE{TILE}")
    add("prefill", dname, label, bs, N, Hq, Hkv, "gluon", 1, g_tot, g_tot, 0.0,
        flops, byts, xc, f"BM{g_BM} nw{g_nw} mfma32 nb2 wpe{g_wpe} TILE{TILE}")
    print(f"[pre {dname:4s}] {label:18s}: "
          f"tri {t_tot:7.1f}us {B.tflops(flops, t_tot):4.0f}TF | "
          f"glu {g_tot:7.1f}us {B.tflops(flops, g_tot):4.0f}TF "
          f"{t_tot / g_tot:.2f}x xc{xc:.0e}", flush=True)
    del q, k, v, bt, ot, og; torch.cuda.empty_cache()


def guarded(fn, phase, dname, label, args):
    try:
        fn(*args, dname, DT_OF[dname])
    except Exception as exc:  # a cell that fails to compile must not kill the scan
        records.append(dict(ver=VER, ver_full=VER_FULL, head_size=HS, phase=phase, dtype=dname, label=label,
                            error=f"{type(exc).__name__}: {exc}"))
        print(f"[{phase[:3]} {dname:4s}] {label:18s}: FAILED {type(exc).__name__}: {exc}",
              flush=True)
        traceback.print_exc()
        torch.cuda.empty_cache()


DT_OF = dict(DTYPES)

print(f"=== triton {VER_FULL} | {B.CU} CUs | "
      f"{len(DECODE_SHAPES)} decode + {len(PREFILL_SHAPES)} prefill shapes x {len(DTYPES)} dtypes ===",
      flush=True)
for dname, dt in DTYPES:
    for (C, ctx, Hq, Hkv) in DECODE_SHAPES:
        guarded(decode_one, "decode", dname, f"C{C} ctx{ctx} {Hq}/{Hkv}", (C, ctx, Hq, Hkv))
    for (bs, N, Hq, Hkv) in PREFILL_SHAPES:
        guarded(prefill_one, "prefill", dname, f"b{bs} {N} {Hq}/{Hkv}", (bs, N, Hq, Hkv))

SCAN_DIR = os.environ.get("SCAN_DIR", "/app/aiter/bench_gluon_ua")
path = f"{SCAN_DIR}/scanfull{os.environ.get('OUT_SUFFIX', '')}_{VER}.json"
json.dump(records, open(path, "w"), indent=0)
nfail = sum(1 for r in records if "error" in r)
print(f"wrote {path} ({len(records)} records, {nfail} failed cells)")
