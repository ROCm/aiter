"""Supporting probes for the MFMA shape study: register pressure, the buffer-op path,
and the fp8 16x16x128 TILE_SIZE constraint.

The nqpk/decode/prefill scans answer *which* tile wins; these answer *why*, and they are
what turns the decode result from "16x16 by 0-5%" into a rule with a mechanism behind it:

  A. reg-pressure  compile-only VGPR / spill audit per (dtype, TILE, BLOCK_M, MFMA_DIM,
                   num_warps, USE_LOAD_BUFFER_OP). Each config compiles in its own
                   subprocess because the fp8 16x16 path at TILE=64 aborts the compiler
                   with an LLVM assertion (not a catchable Python exception).
  B. buffer-op     the same configs, timed with buffer ops forced on and off, on a cache
                   small enough that BOTH settings are legal -- isolating the code-path
                   effect from the cache-size effect.
  C. fp8-tile      does v_mfma_scale_f32_16x16x128_f8f6f4 lower, and is it correct/faster,
                   at TILE_SIZE 64 vs 128? Both dots must reach K=128: the QK dot's K is
                   HEAD_SIZE, the PV dot's K is TILE_SIZE.

    UA_HEAD_SIZE={64,128} python mfma_tile_probes.py
"""
import sys, os, re, json, math, time, subprocess
import torch, triton
sys.path.insert(0, "/app/aiter/bench_gluon_ua")
import bench_ua as B
from aiter.ops.triton.utils.types import e4m3_dtype
from aiter.ops.triton._gluon_kernels.gfx950.attention.unified_attention import (
    kernel_unified_attention_2d as glu)

DEV, HS = B.DEV, B.HEAD_SIZE
VER = triton.__version__.split("+")[0]
_one = torch.ones(1, dtype=torch.float32, device=DEV)
# (BLOCK_M, MFMA_DIM, num_warps). BLOCK_M must be a multiple of MFMA_DIM*num_warps, so
# (16,32,*) is not representable -- MFMA_DIM=32 floors BLOCK_M at 32.
CONFIGS = [(16, 16, 1), (32, 32, 1), (32, 16, 2), (64, 32, 2), (64, 16, 4)]


def _mk(C, ctx, Hkv, tile, dt, Hq=64):
    q = torch.randn(C, Hq, HS, dtype=torch.float32, device=DEV).to(dt)
    k, v, bt = B.make_paged_kv(ctx, C, tile, Hkv, dtype=dt)
    cu = torch.arange(0, C + 1, dtype=torch.int32, device=DEV)
    seqk = torch.full((C,), ctx, dtype=torch.int32, device=DEV)
    o = torch.empty(C, Hq, HS, dtype=torch.bfloat16, device=DEV)
    return q, k, v, bt, cu, seqk, o


def launch(q, k, v, o, cu, seqk, bt, tile, BM, nw, mfma, use_buf, fp8, nbuf=2):
    Hq = q.shape[1]; NKV = k.shape[2]; NS = seqk.shape[0]; nqpk = Hq // NKV
    d = _one if fp8 else None
    return glu[(NS, NKV, 1)](
        query_ptr=q, key_cache_ptr=k, value_cache_ptr=v, sink_ptr=None, output_ptr=o,
        block_tables_ptr=bt, seq_lens_ptr=seqk, query_start_len_ptr=cu,
        query_stride_0=q.stride(0), query_stride_1=q.stride(1),
        output_stride_0=o.stride(0), output_stride_1=o.stride(1),
        k_descale_ptr=d, v_descale_ptr=d, q_descale_ptr=d, out_scale_ptr=None,
        USE_SINKS=False, SLIDING_WINDOW=0, num_blocks=k.shape[0],
        stride_k_cache_0=k.stride(0), stride_k_cache_1=k.stride(1),
        stride_k_cache_2=k.stride(2), stride_k_cache_3=k.stride(3),
        stride_v_cache_0=v.stride(0), stride_v_cache_1=v.stride(1),
        stride_v_cache_2=v.stride(2), stride_v_cache_3=v.stride(3),
        block_table_stride=bt.stride(0), num_seqs=NS, SCALE=1.0 / math.sqrt(HS),
        NUM_QUERY_HEADS=Hq, NUM_KV_HEADS=NKV, BLOCK_SIZE=tile,
        TILE_SIZE=tile, HEAD_SIZE=HS, BLOCK_Q=max(1, BM // nqpk), BLOCK_M=BM,
        ARCH_NAME="gfx950", waves_per_eu=2, USE_LOAD_BUFFER_OP=use_buf,
        USE_STORE_BUFFER_OP=True, num_warps=nw, ALL_DECODE=True,
        CAUSAL=True, REMOVE_INDIRECT_ACCESS=False, NUM_BUFFERS=nbuf, MFMA_DIM=mfma,
        NUM_SPLITS=1, partial_m_ptr=None, partial_l_ptr=None, partial_acc_ptr=None)


# ------------------------------------------------------ A: one compile, in a subprocess
if os.environ.get("PROBE_ONE"):
    tile = int(os.environ["P_TILE"]); BM = int(os.environ["P_BM"])
    mfma = int(os.environ["P_MFMA"]); nw = int(os.environ["P_NW"])
    use_buf = os.environ["P_BUF"] == "1"; fp8 = os.environ["P_FP8"] == "1"
    dt = e4m3_dtype if fp8 else torch.bfloat16
    q, k, v, bt, cu, seqk, o = _mk(2, tile * 8, 8, tile, dt)
    ck = launch(q, k, v, o, cu, seqk, bt, tile, BM, nw, mfma, use_buf, fp8)
    a = ck.asm["amdgcn"]
    g = lambda key: int((re.search(rf"\.{key}:\s*(\d+)", a) or [None, -1])[1])
    kinds = sorted(set(re.findall(r"v_mfma[\w]+", a)))
    print("PROBE_JSON " + json.dumps(dict(
        ok=True, vgpr=g("vgpr_count"), agpr=g("agpr_count"), spill=g("vgpr_spill_count"),
        scratch=g("private_segment_fixed_size"), n_mfma=len(re.findall(r"v_mfma", a)),
        instr=[x.replace("v_mfma_", "") for x in kinds])))
    sys.exit(0)


def reg_probe(tile, BM, mfma, nw, use_buf, fp8):
    env = dict(os.environ, PROBE_ONE="1", P_TILE=str(tile), P_BM=str(BM), P_MFMA=str(mfma),
               P_NW=str(nw), P_BUF="1" if use_buf else "0", P_FP8="1" if fp8 else "0")
    try:
        r = subprocess.run([sys.executable, __file__], env=env, capture_output=True,
                           text=True, timeout=600)
    except subprocess.TimeoutExpired:
        return dict(ok=False, reason="timeout")
    for line in r.stdout.splitlines():
        if line.startswith("PROBE_JSON "):
            return json.loads(line[len("PROBE_JSON "):])
    err = r.stderr
    reason = ("compiler-assertion" if "Assertion" in err or "PassManager" in err
              else "error")
    return dict(ok=False, reason=reason)


out = {"ver": VER, "head_size": HS, "reg": [], "bufop": [], "fp8_tile": []}
print(f"=== MFMA tile probes | triton {VER} | HEAD_SIZE={HS} ===", flush=True)

print("\n--- A. register pressure / spills (compile-only) ---", flush=True)
print(f"{'dtype':>5s} {'TILE':>4s} {'BM':>4s} {'MFMA':>5s} {'nw':>3s} | "
      f"{'bufop=T vgpr/spill':>20s} | {'bufop=F vgpr/spill':>20s}", flush=True)
for fp8 in (False, True):
    for tile in (64, 128):
        for (BM, mfma, nw) in CONFIGS:
            cells = {}
            for use_buf in (True, False):
                r = reg_probe(tile, BM, mfma, nw, use_buf, fp8)
                cells[use_buf] = r
                out["reg"].append(dict(dtype="fp8" if fp8 else "bf16", tile=tile,
                                       block_m=BM, mfma=mfma, num_warps=nw,
                                       buffer_ops=use_buf, **r))
            def fmt(r):
                return (f"{r['vgpr']:4d}/{r['spill']:<3d}" if r.get("ok")
                        else f"{r['reason']:>8s}")
            flag = ""
            if cells[False].get("ok") and cells[False].get("spill", 0) > 0:
                flag = "  <== SPILLS"
            print(f"{'fp8' if fp8 else 'bf16':>5s} {tile:4d} {BM:4d} {mfma:5d} {nw:3d} | "
                  f"{fmt(cells[True]):>20s} | {fmt(cells[False]):>20s}{flag}", flush=True)

print("\n--- B. buffer-op path, timed (bf16, kv small enough that both are legal) ---",
      flush=True)
C, ctx, Hkv, tile = 64, 8192, 8, 64
q, k, v, bt, cu, seqk, o = _mk(C, ctx, Hkv, tile, torch.bfloat16)
kv_bytes = k.nelement() * k.element_size() * 2
print(f"C{C} ctx{ctx} 64/{Hkv} TILE{tile}: kv {kv_bytes/1e9:.2f}GB "
      f"(production buffer ops = {B.buffer_op_flags(k, o)[0]})", flush=True)
for (BM, mfma, nw) in CONFIGS:
    res = {}
    for use_buf in (True, False):
        try:
            f = lambda: launch(q, k, v, o, cu, seqk, bt, tile, BM, nw, mfma, use_buf, False)
            f(); torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(10): f()
            torch.cuda.synchronize()
            res[use_buf] = (time.perf_counter() - t0) / 10 * 1e6
        except Exception:
            res[use_buf] = float("nan")
    out["bufop"].append(dict(block_m=BM, mfma=mfma, num_warps=nw, C=C, ctx=ctx, Hkv=Hkv,
                             tile=tile, bytes=kv_bytes, us_bufop=res[True],
                             us_nobufop=res[False]))
    print(f"  BM{BM:<3d} MFMA{mfma:<3d} nw{nw}: bufop=T {res[True]:8.1f}us  "
          f"bufop=F {res[False]:8.1f}us  ratio {res[False]/res[True]:5.2f}x", flush=True)
del q, k, v, bt, o; torch.cuda.empty_cache()

print("\n--- C. fp8 16x16x128 vs TILE_SIZE (QK dot K=HEAD_SIZE, PV dot K=TILE_SIZE) ---",
      flush=True)


def torch_ref(q, k, v, bt, ctx, Hq, Hkv, tile, s=0):
    nqpk = Hq // Hkv
    blocks = bt[s, :ctx // tile].long()
    kk = k[blocks].permute(2, 0, 1, 3).reshape(Hkv, ctx, HS).float()
    vv = v[blocks].permute(2, 0, 1, 3).reshape(Hkv, ctx, HS).float()
    r = torch.empty(Hq, HS, dtype=torch.float32, device=DEV)
    for h in range(Hq):
        p = torch.softmax((kk[h // nqpk] @ q[s, h].float()) * (1.0 / math.sqrt(HS)), dim=0)
        r[h] = p @ vv[h // nqpk]
    return r


for tile in (64, 128):
    C, ctx, Hq, Hkv = 64, 8192, 64, 8
    nqpk = Hq // Hkv
    q, k, v, bt, cu, seqk, o = _mk(C, ctx, Hkv, tile, e4m3_dtype)
    ref = torch_ref(q, k, v, bt, ctx, Hq, Hkv, tile)
    nbytes = k.nelement() + v.nelement()
    for mfma in (16, 32):
        BM = max(mfma, triton.next_power_of_2(nqpk)); nw = BM // mfma
        rec = dict(tile=tile, mfma=mfma, block_m=BM, num_warps=nw, C=C, ctx=ctx,
                   Hq=Hq, Hkv=Hkv, m_util=nqpk / BM)
        try:
            f = lambda: B.launch_glu_2d(q, k, v, o, cu, seqk, bt, 1.0 / math.sqrt(HS), BM,
                                        tile, nw, 2, NUM_SPLITS=1, ALL_DECODE=True,
                                        MFMA_DIM=mfma, NUM_BUFFERS=2,
                                        descales=(_one, _one, _one))
            f(); torch.cuda.synchronize()
            rel = ((ref - o[0].float()).abs().mean() / ref.abs().mean()).item()
            t0 = time.perf_counter()
            for _ in range(10): f()
            torch.cuda.synchronize()
            us = (time.perf_counter() - t0) / 10 * 1e6
            rec.update(ok=True, time_us=us, rel=rel, tbs=nbytes / (us / 1e6) / 1e12)
            print(f"  TILE{tile:<4d} mfma{mfma:<3d} BM{BM:<3d}: {us:8.1f}us  "
                  f"{rec['tbs']:5.2f} TB/s  rel {rel:6.2%}  M-util {rec['m_util']:.0%}",
                  flush=True)
        except Exception as exc:
            rec.update(ok=False, reason=f"{type(exc).__name__}")
            print(f"  TILE{tile:<4d} mfma{mfma:<3d} BM{BM:<3d}: FAILED "
                  f"({type(exc).__name__} -- does not lower)", flush=True)
        out["fp8_tile"].append(rec)
    del q, k, v, bt, o, ref; torch.cuda.empty_cache()

path = os.environ.get("PROBE_OUT",
                      f"/app/aiter/bench_gluon_ua/mfma_shape_study_7_28/probes_hs{HS}_{VER}.json")
json.dump(out, open(path, "w"), indent=0)
print("\nwrote", path)
