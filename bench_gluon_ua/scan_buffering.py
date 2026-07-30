"""Decode buffering comparison for the current triton version: gluon KV staging
as direct-to-registers (nb=0), single LDS buffer (nb=1), double LDS buffer (nb=2),
plus the triton baseline. Records time (us), GB/s, xcheck, and kernel resources
(LDS bytes / VGPRs / spills). Writes buffering_<ver>.json. Run once per version."""
import sys, math, json
import triton
sys.path.insert(0, "/app/aiter/bench_gluon_ua")
import torch
import bench_ua as B
from aiter.ops.triton._gluon_kernels.gfx950.attention.unified_attention import (
    kernel_unified_attention_2d as glu,
)

DEV, DT, HS, RCP = B.DEV, B.DT, B.HEAD_SIZE, B.RCP_LN2
VER_FULL = triton.__version__
VER = VER_FULL.split("+")[0]
TILE = 64
torch.manual_seed(0)
SHAPES = [(C, ctx, Hq, Hkv) for ctx in (1024, 8192) for (Hq, Hkv) in ((64, 8), (8, 1))
          for C in (16, 32, 64, 128)]
records = []


def _cache_keys():
    keys = set()
    for dc in glu.device_caches.values():
        cache = dc[0] if isinstance(dc, tuple) else dc
        try:
            keys |= set(cache.keys())
        except AttributeError:
            pass
    return keys


def launch_and_stat(fn):
    """Run fn once; return (LDS_bytes, n_regs, n_spills) of the kernel it compiled
    (diff the JIT cache before/after). Returns (None,)*3 on a cache hit."""
    before = _cache_keys()
    fn()
    torch.cuda.synchronize()
    for dc in glu.device_caches.values():
        cache = dc[0] if isinstance(dc, tuple) else dc
        try:
            items = list(cache.items())
        except AttributeError:
            continue
        for kkey, ck in items:
            if kkey not in before:
                md = getattr(ck, "metadata", None)
                return (getattr(md, "shared", None), getattr(ck, "n_regs", None),
                        getattr(ck, "n_spills", None))
    return (None, None, None)


def scan_one(C, ctx, Hq, Hkv):
    nqpk = Hq // Hkv; scale = 1.0 / math.sqrt(HS)
    q = torch.randn(C, Hq, HS, dtype=DT, device=DEV)
    k, v, bt = B.make_paged_kv(ctx, C, TILE, Hkv)
    cu = torch.arange(0, C + 1, dtype=torch.int32, device=DEV)
    seqk = torch.full((C,), ctx, dtype=torch.int32, device=DEV)
    flops, byts = B.decode_flops_bytes(C, ctx, Hq, Hkv)
    a, r = B.select_3d_config(HS, TILE, ctx, B.TARGET_PRGMS, C * Hkv, DT, DT, False, 1, 0)
    S = a["NUM_SEGMENTS_PER_SEQ"]; anw = a["num_warps"]; wpe = a["waves_per_eu"]; nst = a["num_stages"]; rnw = r["num_warps"]

    # triton baseline (its own heuristic split)
    seg = B.alloc_segm(C, Hq, S); ref = torch.empty_like(q)
    def tf():
        B.launch_tri_3d(q, k, v, cu, seqk, bt, scale, 16, 16 // nqpk, TILE, anw, S, wpe, nst, seg)
        B.launch_reduce(ref, cu, seqk, bt, TILE, S, rnw, 16 // nqpk, seg)
    rt = B.profile_kernels(tf)
    t_tot = B.pick(rt, "unified_attention_3d") + B.pick(rt, "reduce_segments")
    records.append(dict(ver=VER, C=C, ctx=ctx, Hq=Hq, Hkv=Hkv, mode="triton", split=S,
                        time_us=t_tot, gbps=B.gbps(byts, t_tot), tflops=B.tflops(flops, t_tot),
                        xc=0.0, lds=None, regs=None, spills=None))

    # gluon: right-sized split, three buffering modes
    Sg = B.select_gluon_num_splits(C, Hkv, ctx // TILE)
    for nb, mode in ((0, "registers"), (1, "single_lds"), (2, "double_lds")):
        try:
            og = torch.empty_like(q)
            if Sg == 1:
                def gf():
                    B.launch_glu_2d(q, k, v, og, cu, seqk, bt, scale, 16, TILE, 1, wpe,
                                    NUM_SPLITS=1, ALL_DECODE=True, MFMA_DIM=16, NUM_BUFFERS=nb)
                lds, regs, spills = launch_and_stat(gf)
                xc = (og.float() - ref.float()).abs().max().item()
            else:
                so, sm, se = B.alloc_segm(C, Hq, Sg)
                def gf():
                    B.launch_glu_2d(q, k, v, og, cu, seqk, bt, scale, 16, TILE, 1, wpe,
                                    NUM_SPLITS=Sg, ALL_DECODE=True, partials=(sm, se, so), MFMA_DIM=16, NUM_BUFFERS=nb)
                    B.launch_reduce(og, cu, seqk, bt, TILE, Sg, rnw, 16 // nqpk, (so, sm, se))
                # correctness (prescaled) + resource capture on the attn kernel
                lds, regs, spills = launch_and_stat(lambda: B.launch_glu_2d(
                    q, k, v, og, cu, seqk, bt, scale, 16, TILE, 1, wpe, NUM_SPLITS=Sg,
                    ALL_DECODE=True, partials=(sm, se, so), MFMA_DIM=16, NUM_BUFFERS=nb))
                B.launch_reduce(og, cu, seqk, bt, TILE, Sg, rnw, 16 // nqpk, (so, sm, se))
                torch.cuda.synchronize()
                xc = (og.float() - ref.float()).abs().max().item()
            rg = B.profile_kernels(gf)
            g_tot = B.pick(rg, "unified_attention_2d") + (B.pick(rg, "reduce_segments") if Sg > 1 else 0.0)
            records.append(dict(ver=VER, C=C, ctx=ctx, Hq=Hq, Hkv=Hkv, mode=mode, split=Sg,
                                time_us=g_tot, gbps=B.gbps(byts, g_tot), tflops=B.tflops(flops, g_tot),
                                xc=xc, lds=lds, regs=regs, spills=spills))
            tag = f"{mode} {B.gbps(byts, g_tot):.0f}GB/s(LDS{lds} sp{spills}) {t_tot/g_tot:.2f}x"
        except Exception as e:
            records.append(dict(ver=VER, C=C, ctx=ctx, Hq=Hq, Hkv=Hkv, mode=mode, split=Sg,
                                time_us=None, gbps=None, tflops=None, xc=None,
                                lds=None, regs=None, spills=None, error=type(e).__name__))
            tag = f"{mode} FAIL:{type(e).__name__}"
        print(f"C{C:>3} ctx{ctx} {Hq}/{Hkv} S{Sg}: tri {B.gbps(byts,t_tot):.0f} | {tag}")


print(f"=== triton {VER_FULL} : decode buffering (registers / single-LDS / double-LDS) ===")
for sh in SHAPES:
    scan_one(*sh)
path = f"/app/aiter/bench_gluon_ua/buffering_{VER}.json"
json.dump(records, open(path, "w"), indent=0)
print("wrote", path)
