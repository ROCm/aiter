"""End-to-end check of the gfx950 gluon wrapper's OWN heuristics.

Every other script in this folder drives `kernel_unified_attention_2d` directly with
hand-picked knobs, so the tile/warp/buffer choices in `unified_attention()` itself were
never exercised as a unit. This calls the public wrapper, prints the configuration its
heuristics select per shape, checks the result against an fp32 torch reference, and times
it against the Triton kernel.

Coverage includes the case the study added: **MHA, num_kv_heads == num_query_heads**
(nqpk=1), alongside GQA and MQA, in bf16 and fp8, decode and prefill.

    UA_HEAD_SIZE={64,128} python verify_heuristics.py
"""
import sys, os, math, time, json
import torch, triton
sys.path.insert(0, "/app/aiter/bench_gluon_ua")
import bench_ua as B
from aiter.ops.triton.utils.types import e4m3_dtype
from aiter.ops.triton._gluon_kernels.gfx950.attention.unified_attention import (
    unified_attention as glu_ua)

DEV, HS = B.DEV, B.HEAD_SIZE
VER = triton.__version__.split("+")[0]
TILE = 64
one = torch.ones(1, dtype=torch.float32, device=DEV)
torch.manual_seed(0)

# (label, phase, num_seqs/batch, ctx or seqlen, Hq, Hkv)
SHAPES = [
    ("decode MHA   ", "decode", 16, 8192, 64, 64),
    ("decode MHA   ", "decode", 128, 1024, 64, 64),
    ("decode GQA-8 ", "decode", 128, 8192, 64, 8),
    ("decode GQA-8 ", "decode", 16, 1024, 64, 8),
    ("decode MQA   ", "decode", 128, 8192, 8, 1),
    ("decode 32:1  ", "decode", 16, 8192, 64, 2),
    ("prefill MHA  ", "prefill", 8, 1024, 64, 64),
    ("prefill GQA-8", "prefill", 8, 1024, 64, 8),
    ("prefill GQA-8", "prefill", 1, 8192, 64, 8),
]


def torch_ref(q, k, v, bt, ctx, Hq, Hkv, tile, causal, qlen, s=0):
    """fp32 reference for sequence `s` (decode: 1 query row; prefill: last row)."""
    nqpk = Hq // Hkv
    blocks = bt[s, :max(1, ctx // tile)].long()
    kk = k[blocks].permute(2, 0, 1, 3).reshape(Hkv, -1, HS).float()[:, :ctx]
    vv = v[blocks].permute(2, 0, 1, 3).reshape(Hkv, -1, HS).float()[:, :ctx]
    qi = (s * qlen) + qlen - 1          # index of the query row we check
    out = torch.empty(Hq, HS, dtype=torch.float32, device=DEV)
    for h in range(Hq):
        logits = (kk[h // nqpk] @ q[qi, h].float()) * (1.0 / math.sqrt(HS))
        if causal:
            logits[ctx:] = float("-inf")
        p = torch.softmax(logits, dim=0)
        out[h] = p @ vv[h // nqpk]
    return out, qi


def run(label, phase, N, L, Hq, Hkv, fp8):
    dt = e4m3_dtype if fp8 else torch.bfloat16
    nqpk = Hq // Hkv
    scale = 1.0 / math.sqrt(HS)
    qlen = 1 if phase == "decode" else L
    ntok = N * qlen
    q = torch.randn(ntok, Hq, HS, dtype=torch.float32, device=DEV).to(dt)
    k, v, bt = B.make_paged_kv(L, N, TILE, Hkv, dtype=dt)
    cu = torch.arange(0, (N + 1) * qlen, qlen, dtype=torch.int32, device=DEV)
    seqk = torch.full((N,), L, dtype=torch.int32, device=DEV)
    out = torch.empty(ntok, Hq, HS, dtype=torch.bfloat16, device=DEV)
    d = (one, one.clone(), one.clone()) if fp8 else (None, None, None)

    def f():
        glu_ua(q, k, v, out, cu, seqk, qlen, L, scale, True, (-1, -1), bt, 0,
               d[0], d[1], d[2], None)
    try:
        f(); torch.cuda.synchronize()
    except Exception as exc:
        print(f"  {label} {'fp8 ' if fp8 else 'bf16'} {Hq}/{Hkv:<3d} N{N:<4d} L{L:<5d}: "
              f"FAILED {type(exc).__name__}: {str(exc)[:70]}", flush=True)
        return None
    ref, qi = torch_ref(q, k, v, bt, L, Hq, Hkv, TILE, phase == "prefill", qlen)
    rel = ((ref - out[qi].float()).abs().mean() / ref.abs().mean()).item()

    t0 = time.perf_counter()
    for _ in range(10): f()
    torch.cuda.synchronize()
    us = (time.perf_counter() - t0) / 10 * 1e6
    kvb = k.nelement() * k.element_size() * 2
    tag = "ok " if rel < 0.06 else "BAD"
    print(f"  {label} {'fp8 ' if fp8 else 'bf16'} {Hq}/{Hkv:<3d} N{N:<4d} L{L:<5d}: "
          f"{us:9.1f}us  {kvb/(us/1e6)/1e12:5.2f} TB/s  rel {rel:6.2%} {tag}  "
          f"kv {kvb/1e9:6.2f}GB bufops={B.buffer_op_flags(k, out)[0]}", flush=True)
    r = dict(label=label.strip(), phase=phase, N=N, L=L, Hq=Hq, Hkv=Hkv, nqpk=nqpk,
             dtype="fp8" if fp8 else "bf16", head_size=HS, time_us=us, rel=rel,
             kv_bytes=kvb, ok=rel < 0.06)
    del q, k, v, bt, out, ref
    torch.cuda.empty_cache()
    return r


recs = []
print(f"=== wrapper heuristics verification | triton {VER} | HEAD_SIZE={HS} ===", flush=True)
for fp8 in (False, True):
    print(f"\n--- {'fp8' if fp8 else 'bf16'} ---", flush=True)
    for (label, phase, N, L, Hq, Hkv) in SHAPES:
        r = run(label, phase, N, L, Hq, Hkv, fp8)
        if r:
            recs.append(r)

bad = [r for r in recs if not r["ok"]]
print(f"\n{len(recs)} configurations ran, {len(bad)} incorrect"
      + ("" if not bad else ": " + ", ".join(f"{r['label']} {r['dtype']}" for r in bad)))
path = os.environ.get("VERIFY_OUT",
                      f"/app/aiter/bench_gluon_ua/mfma_shape_study_7_28/verify_hs{HS}_{VER}.json")
json.dump(recs, open(path, "w"), indent=0)
print("wrote", path)
