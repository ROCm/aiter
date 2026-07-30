"""MFMA instruction-shape study for the gfx950 gluon unified-attention kernel.

gfx950 exposes two MFMA shapes per dtype that this kernel can use:

    bf16   32x32x16  (v_mfma_f32_32x32x16_bf16)      vs  16x16x32  (v_mfma_f32_16x16x32_bf16)
    fp8    32x32x64  (v_mfma_scale_f32_32x32x64_f8f6f4) vs 16x16x128 (..._16x16x128_f8f6f4)

Both members of a pair do the same MACs per instruction *pair-wise* (32*32*16 == 2 * 16*16*32),
so the 32x32 form is 2x the work per instruction. What differs is the M granularity: BLOCK_M
must be a multiple of MFMA_DIM * num_warps, and in decode only NUM_QUERIES_PER_KV of the M rows
carry real queries, so the wider tile wastes more lanes.

This sweeps, per (phase, dtype, MFMA_DIM), the dot-operand K_WIDTH for the QK and PV dots
(the kernel hard-codes them per shape) and records compile / correctness / time. It loads
patched *copies* of the kernel module, so the checked-in kernel is never modified.

    python mfma_shape_study.py            # HEAD_SIZE=128 (set UA_HEAD_SIZE=64 for the other)
"""
import sys, os, math, json, importlib.util
import torch, triton
sys.path.insert(0, "/app/aiter/bench_gluon_ua")
import bench_ua as B
from aiter.ops.triton.utils.types import e4m3_dtype

DEV, HS, RCP = B.DEV, B.HEAD_SIZE, B.RCP_LN2
TILE = 64
VER = triton.__version__.split("+")[0]
SCRATCH = os.environ.get("MFMA_SCRATCH", "/tmp/mfma_variants")
KERNEL_SRC = ("/app/aiter/aiter/ops/triton/_gluon_kernels/gfx950/attention/"
              "unified_attention.py")
one = torch.ones(1, dtype=torch.float32, device=DEV)
D8 = (one, one.clone(), one.clone())
torch.manual_seed(0)
os.makedirs(SCRATCH, exist_ok=True)

ORIG_BLOCK = """        if MFMA_DIM == 32:
            mfma_instr = [32, 32, 16] if not self.DOT_FP8 else [32, 32, 64]
            self.K_WIDTH_QK = gl.constexpr(16) if self.DOT_FP8 else gl.constexpr(8)
            self.K_WIDTH_PV = gl.constexpr(16) if self.DOT_FP8 else gl.constexpr(4)
        else:
            mfma_instr = [16, 16, 32] if not self.DOT_FP8 else [16, 16, 128]
            self.K_WIDTH_QK = gl.constexpr(16) if self.DOT_FP8 else gl.constexpr(8)
            self.K_WIDTH_PV = gl.constexpr(16) if self.DOT_FP8 else gl.constexpr(8)"""
PATCH_BLOCK = """        if MFMA_DIM == 32:
            mfma_instr = [32, 32, 16] if not self.DOT_FP8 else [32, 32, 64]
            self.K_WIDTH_QK = gl.constexpr({kq})
            self.K_WIDTH_PV = gl.constexpr({kp})
        else:
            mfma_instr = [16, 16, 32] if not self.DOT_FP8 else [16, 16, 128]
            self.K_WIDTH_QK = gl.constexpr({kq})
            self.K_WIDTH_PV = gl.constexpr({kp})"""

_src = open(KERNEL_SRC).read()
assert ORIG_BLOCK in _src, "kernel K_WIDTH block moved - update ORIG_BLOCK"
_mods = {}


def variant_kernel(kq, kp):
    """Load a copy of the kernel module with K_WIDTH_QK/PV pinned to (kq, kp)."""
    key = (kq, kp)
    if key in _mods:
        return _mods[key]
    name = f"ua_kw{kq}_{kp}"
    path = f"{SCRATCH}/{name}.py"
    open(path, "w").write(_src.replace(ORIG_BLOCK, PATCH_BLOCK.format(kq=kq, kp=kp)))
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    _mods[key] = mod.kernel_unified_attention_2d
    return _mods[key]


# ------------------------------------------------------------------ references
def triton_ref(phase, dt, shape):
    """Triton output + time for the same inputs (correctness baseline)."""
    if phase == "decode":
        C, ctx, Hq, Hkv = shape; nqpk = Hq // Hkv; scale = 1.0 / math.sqrt(HS)
        f8 = dt.itemsize == 1
        d = D8 if f8 else (None, None, None)
        a, r = B.select_3d_config(HS, TILE, ctx, B.TARGET_PRGMS, C * Hkv, dt, dt, False, 1, 0)
        S = a["NUM_SEGMENTS_PER_SEQ"]; seg = B.alloc_segm(C, Hq, S)
        o = torch.empty(C, Hq, HS, dtype=torch.bfloat16, device=DEV)
        q, k, v, bt, cu, seqk = INPUTS[(phase, dt, shape)]
        def f():
            B.launch_tri_3d(q, k, v, cu, seqk, bt, scale, 16, 16 // nqpk, TILE, a["num_warps"],
                            S, a["waves_per_eu"], a["num_stages"], seg, descales=d,
                            K_WIDTH=16 if f8 else 8, IS_Q_FP8=f8, IS_KV_FP8=f8)
            B.launch_reduce(o, cu, seqk, bt, TILE, S, r["num_warps"], 16 // nqpk, seg)
        f(); torch.cuda.synchronize()
        rt = B.profile_kernels(f)
        return o, B.pick(rt, "unified_attention_3d") + B.pick(rt, "reduce_segments")
    bs, N, Hq, Hkv = shape; nqpk = Hq // Hkv; scale = 1.0 / math.sqrt(HS)
    f8 = dt.itemsize == 1
    d = D8 if f8 else (None, None, None)
    q, k, v, bt, cu, seqk = INPUTS[(phase, dt, shape)]
    c = B.select_2d_config(TILE, HS, 0, False, N, N, nqpk, 1, dt, dt, False)
    o = torch.empty(bs * N, Hq, HS, dtype=torch.bfloat16, device=DEV)
    def f():
        B.launch_tri_2d(q, k, v, o, cu, seqk, bt, scale, c["BLOCK_M"], c["BLOCK_Q"], TILE,
                        c["num_warps"], c["num_stages"], c["waves_per_eu"],
                        descales=d, K_WIDTH=16 if f8 else 8)
    f(); torch.cuda.synchronize()
    return o, B.pick(B.profile_kernels(f), "unified_attention")


def make_inputs(phase, dt, shape):
    if phase == "decode":
        C, ctx, Hq, Hkv = shape
        q = torch.randn(C, Hq, HS, dtype=torch.float32, device=DEV).to(dt)
        k, v, bt = B.make_paged_kv(ctx, C, TILE, Hkv, dtype=dt)
        cu = torch.arange(0, C + 1, dtype=torch.int32, device=DEV)
        seqk = torch.full((C,), ctx, dtype=torch.int32, device=DEV)
    else:
        bs, N, Hq, Hkv = shape
        q = torch.randn(bs * N, Hq, HS, dtype=torch.float32, device=DEV).to(dt)
        k, v, bt = B.make_paged_kv(N, bs, TILE, Hkv, dtype=dt)
        cu = torch.arange(0, (bs + 1) * N, N, dtype=torch.int32, device=DEV)
        seqk = torch.full((bs,), N, dtype=torch.int32, device=DEV)
    return q, k, v, bt, cu, seqk


def gluon_run(phase, dt, shape, mfma, BM, nw, wpe, kq, kp, ref):
    """Compile+time one (shape, MFMA_DIM, K_WIDTH) variant. Returns (time_us, xc) or (None, err)."""
    kern = variant_kernel(kq, kp)
    saved = B.glu_2d
    B.glu_2d = kern
    q, k, v, bt, cu, seqk = INPUTS[(phase, dt, shape)]
    d = D8 if dt.itemsize == 1 else (None, None, None)
    scale = 1.0 / math.sqrt(HS)
    try:
        if phase == "decode":
            C, ctx, Hq, Hkv = shape; nqpk = Hq // Hkv
            o = torch.empty(C, Hq, HS, dtype=torch.bfloat16, device=DEV)
            Sg = B.select_gluon_num_splits(C, Hkv, ctx // TILE)
            if Sg == 1:
                def f():
                    B.launch_glu_2d(q, k, v, o, cu, seqk, bt, scale, BM, TILE, nw, wpe,
                                    NUM_SPLITS=1, ALL_DECODE=True, MFMA_DIM=mfma,
                                    NUM_BUFFERS=2, descales=d)
                f(); torch.cuda.synchronize()
                t = B.pick(B.profile_kernels(f), "unified_attention_2d")
            else:
                so, sm, se = B.alloc_segm(C, Hq, Sg)
                B.launch_glu_2d(q, k, v, o, cu, seqk, bt, scale, BM, TILE, nw, wpe,
                                NUM_SPLITS=Sg, ALL_DECODE=True, partials=(sm, se, so),
                                MFMA_DIM=mfma, NUM_BUFFERS=2, descales=d)
                B.launch_reduce(o, cu, seqk, bt, TILE, Sg, 2, 16 // nqpk,
                                (so, sm, se))
                torch.cuda.synchronize()
                def f():
                    B.launch_glu_2d(q, k, v, o, cu, seqk, bt, scale, BM, TILE, nw, wpe,
                                    NUM_SPLITS=Sg, ALL_DECODE=True, partials=(sm, se, so),
                                    MFMA_DIM=mfma, NUM_BUFFERS=2, descales=d)
                    B.launch_reduce(o, cu, seqk, bt, TILE, Sg, 2, 16 // nqpk, (so, sm, se))
                r = B.profile_kernels(f)
                t = B.pick(r, "unified_attention_2d") + B.pick(r, "reduce_segments")
        else:
            bs, N, Hq, Hkv = shape
            o = torch.empty(bs * N, Hq, HS, dtype=torch.bfloat16, device=DEV)
            def f():
                B.launch_glu_2d(q, k, v, o, cu, seqk, bt, scale, BM, TILE, nw, wpe,
                                MFMA_DIM=mfma, NUM_BUFFERS=2, descales=d)
            f(); torch.cuda.synchronize()
            t = B.pick(B.profile_kernels(f), "unified_attention")
        xc = (ref.float() - o.float()).abs().max().item()
        rel = ((ref.float() - o.float()).abs().mean() / ref.float().abs().mean()).item()
        return dict(time_us=t, xc=xc, rel=rel)
    except Exception as exc:
        return dict(error=f"{type(exc).__name__}: {str(exc)[:80]}")
    finally:
        B.glu_2d = saved
        torch.cuda.empty_cache()


DEC_SHAPE = (128, 8192, 64, 8)
PRE_SHAPE = (8, 1024, 64, 8)
DTYPES = [("bf16", torch.bfloat16), ("fp8", e4m3_dtype)]
KW = {"bf16": [4, 8, 16], "fp8": [8, 16, 32]}
# decode: BLOCK_M = MFMA_DIM (one warp, minimal M tile). prefill: BLOCK_M = 128.
CASES = [("decode", DEC_SHAPE, lambda m: (m, 1)),
         ("prefill", PRE_SHAPE, lambda m: (128, 128 // m))]
WPE = {"decode": 1 if HS < 128 else 2, "prefill": 3 if HS < 128 else 2}

INPUTS = {}
for phase, shape, _ in CASES:
    for _, dt in DTYPES:
        INPUTS[(phase, dt, shape)] = make_inputs(phase, dt, shape)

records = []
print(f"=== MFMA shape study | triton {VER} | HEAD_SIZE={HS} | {B.CU} CUs ===", flush=True)
for phase, shape, tile_of in CASES:
    for dname, dt in DTYPES:
        ref, tref = triton_ref(phase, dt, shape)
        print(f"\n[{phase} {dname}] shape {shape}  triton ref {tref:.1f}us", flush=True)
        for mfma in (16, 32):
            BM, nw = tile_of(mfma)
            for kq in KW[dname]:
                for kp in KW[dname]:
                    r = gluon_run(phase, dname and dt, shape, mfma, BM, nw,
                                  WPE[phase], kq, kp, ref)
                    rec = dict(ver=VER, head_size=HS, phase=phase, dtype=dname, mfma=mfma,
                               block_m=BM, num_warps=nw, kq=kq, kp=kp, triton_us=tref, **r)
                    records.append(rec)
                    if "error" in r:
                        print(f"  mfma{mfma:<2d} BM{BM:<3d} kq{kq:<2d} kp{kp:<2d}: "
                              f"FAIL {r['error']}", flush=True)
                    else:
                        ok = "ok " if r["rel"] < 0.05 else "BAD"
                        print(f"  mfma{mfma:<2d} BM{BM:<3d} kq{kq:<2d} kp{kp:<2d}: "
                              f"{r['time_us']:8.1f}us  {tref / r['time_us']:.2f}x vs triton  "
                              f"{ok} rel {r['rel']:.1%}", flush=True)

out = os.environ.get("MFMA_OUT", f"/app/aiter/bench_gluon_ua/mfma_study_hs{HS}_{VER}.json")
json.dump(records, open(out, "w"), indent=0)
print("\nwrote", out)
