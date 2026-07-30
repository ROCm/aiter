"""Full gfx950 unified-attention matrix scan: Triton vs Gluon, across Triton versions.

Grid, per Triton version:
  decode  : C in {1,8,32,64,128} x ctx in {1024,8192}
  prefill : (B,N) in {(1,1024),(4,1024),(8,1024),(1,8192)}
  heads   : 8/1 (MQA), 64/8 (GQA-8), 8/8 (MHA small), 64/64 (MHA wide)
  dtypes  : bf16, fp8 (q/k/v all e4m3; output stays bf16)
  head_sz : 64, 128

Both implementations are driven through their **public wrappers**, so each one picks its
own configuration exactly as it would in production:
    triton -> aiter.ops.triton.attention.unified_attention.unified_attention
    gluon  -> aiter.ops.triton._gluon_kernels.gfx950...unified_attention

Correctness is checked per cell against an fp32 torch reference computed for one sequence
(cheap even at 34 GB caches), so a wrong-but-fast cell can never be read as a win.

Writes matrix_<HS>_<ver>.json into $SCAN_DIR. Run once per Triton version; switch with
PYTHONPATH=/home/mekaymak/tritons/t3{6,7,8} (see that folder's README).

    UA_HEAD_SIZE={64,128} SCAN_DIR=... python scan_matrix.py
"""
import sys, os, math, json, time, traceback
import torch, triton
sys.path.insert(0, "/app/aiter/bench_gluon_ua")
import bench_ua as B
from aiter.ops.triton.utils.types import e4m3_dtype
from aiter.ops.triton.attention.unified_attention import unified_attention as tri_ua
from aiter.ops.triton._gluon_kernels.gfx950.attention.unified_attention import (
    unified_attention as glu_ua)

DEV, HS = B.DEV, B.HEAD_SIZE
VER = triton.__version__.split("+")[0]
TILE = 64                      # page size == TILE_SIZE (the gluon kernel requires it)
SCAN_DIR = os.environ.get("SCAN_DIR", "/app/aiter/bench_gluon_ua/matrix_scan")
one = torch.ones(1, dtype=torch.float32, device=DEV)
torch.manual_seed(0)
os.makedirs(SCAN_DIR, exist_ok=True)

HEADS = [(8, 1, "MQA 8/1"), (64, 8, "GQA-8 64/8"), (8, 8, "MHA 8/8"), (64, 64, "MHA 64/64")]
DECODE = [(C, ctx) for ctx in (1024, 8192) for C in (1, 8, 32, 64, 128)]
PREFILL = [(1, 1024), (4, 1024), (8, 1024), (1, 8192)]
DTYPES = [("bf16", torch.bfloat16), ("fp8", e4m3_dtype)]
# Cap the KV cache so one cell cannot eat the GPU: 64/64 at C128/ctx8192/HS128 is 34 GB.
MAX_KV_GB = float(os.environ.get("MAX_KV_GB", 40))


def torch_ref(q, k, v, bt, ctx, Hq, Hkv, causal, qlen, s=0):
    """fp32 reference for the LAST query row of sequence `s` (the causal-hardest row)."""
    nqpk = Hq // Hkv
    blocks = bt[s, :ctx // TILE].long()
    kk = k[blocks].permute(2, 0, 1, 3).reshape(Hkv, -1, HS).float()[:, :ctx]
    vv = v[blocks].permute(2, 0, 1, 3).reshape(Hkv, -1, HS).float()[:, :ctx]
    qi = s * qlen + qlen - 1
    out = torch.empty(Hq, HS, dtype=torch.float32, device=DEV)
    for h in range(Hq):
        lg = (kk[h // nqpk] @ q[qi, h].float()) * (1.0 / math.sqrt(HS))
        p = torch.softmax(lg, dim=0)
        out[h] = p @ vv[h // nqpk]
    del kk, vv
    return out, qi


def timeit(fn):
    """Attention + reduce only, via torch.profiler filtered by kernel name.

    Wall clock would be wrong here: bench_ua flushes 512 MB of L2 between iterations and
    that memset is ~93 us -- several times larger than the smaller decode cells, which
    would flatten every ratio toward 1. Name-filtering keeps the flush out and picks up
    both the attention kernel and (when the split path runs) reduce_segments.
    """
    return B.pick(B.profile_kernels(fn), "unified_attention", "reduce_segments")


def cell(phase, N, L, Hq, Hkv, dname, dt):
    nqpk = Hq // Hkv
    qlen = 1 if phase == "decode" else L
    ntok = N * qlen
    scale = 1.0 / math.sqrt(HS)
    kv_gb = 2 * N * L * Hkv * HS * dt.itemsize / 1e9
    rec = dict(ver=VER, head_size=HS, phase=phase, N=N, L=L, Hq=Hq, Hkv=Hkv, nqpk=nqpk,
               dtype=dname, kv_gb=kv_gb)
    if kv_gb > MAX_KV_GB:
        rec["skipped"] = f"kv {kv_gb:.1f}GB > cap"
        return rec
    q = torch.randn(ntok, Hq, HS, dtype=torch.float32, device=DEV).to(dt)
    k, v, bt = B.make_paged_kv(L, N, TILE, Hkv, dtype=dt)
    cu = torch.arange(0, (N + 1) * qlen, qlen, dtype=torch.int32, device=DEV)
    seqk = torch.full((N,), L, dtype=torch.int32, device=DEV)
    d = (one, one.clone(), one.clone()) if dt.itemsize == 1 else (None, None, None)
    ref, qi = torch_ref(q, k, v, bt, L, Hq, Hkv, phase == "prefill", qlen)

    for impl in ("triton", "gluon"):
        out = torch.empty(ntok, Hq, HS, dtype=torch.bfloat16, device=DEV)
        try:
            if impl == "triton":
                fn = lambda: tri_ua(q, k, v, out, cu, qlen, seqk, L, scale, True, (-1, -1),
                                    bt, 0, d[0], d[1], d[2])
            else:
                fn = lambda: glu_ua(q, k, v, out, cu, seqk, qlen, L, scale, True, (-1, -1),
                                    bt, 0, d[0], d[1], d[2], None)
            fn(); torch.cuda.synchronize()
            rel = ((ref - out[qi].float()).abs().mean() / ref.abs().mean()).item()
            rec[impl] = dict(us=timeit(fn), rel=rel, ok=bool(rel < 0.06))
        except Exception as exc:
            rec[impl] = dict(error=f"{type(exc).__name__}: {str(exc)[:110]}")
        del out
        torch.cuda.empty_cache()
    del q, k, v, bt, ref
    torch.cuda.empty_cache()
    return rec


recs = []
print(f"=== matrix scan | triton {VER} | HEAD_SIZE={HS} | {B.CU} CUs | "
      f"{B.WARMUP}+{B.ITERS} iters ===", flush=True)
for dname, dt in DTYPES:
    for (Hq, Hkv, hlabel) in HEADS:
        print(f"\n--- {dname} {hlabel} ---", flush=True)
        for phase, grid in (("decode", DECODE), ("prefill", PREFILL)):
            for (a, b) in grid:
                N, L = (a, b)
                r = cell(phase, N, L, Hq, Hkv, dname, dt)
                recs.append(r)
                tag = f"{phase[:3]} N{N:<4d} L{L:<5d}"
                if "skipped" in r:
                    print(f"  {tag}: skipped ({r['skipped']})", flush=True)
                    continue
                def fmt(x):
                    if "error" in x:
                        return f"ERR {x['error'][:34]}"
                    return f"{x['us']:9.1f}us{'' if x['ok'] else ' BAD'}"
                t, g = r["triton"], r["gluon"]
                sp = (f"{t['us']/g['us']:.2f}x" if "error" not in t and "error" not in g
                      else "–")
                print(f"  {tag}: triton {fmt(t)} | gluon {fmt(g)} | speedup {sp}", flush=True)
                json.dump(recs, open(f"{SCAN_DIR}/matrix_hs{HS}_{VER}.json", "w"), indent=0)

path = f"{SCAN_DIR}/matrix_hs{HS}_{VER}.json"
json.dump(recs, open(path, "w"), indent=0)
nbad = sum(1 for r in recs for i in ("triton", "gluon")
           if i in r and "error" not in r[i] and not r[i]["ok"])
nerr = sum(1 for r in recs for i in ("triton", "gluon") if i in r and "error" in r[i])
print(f"\n{len(recs)} cells | {nbad} numerically wrong | {nerr} errored")
print("wrote", path)
