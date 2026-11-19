#!/usr/bin/env python3
"""Standalone benchmark of the gpt-oss prefill attention call on gfx1250 (aiter unified_attention).

Reproduces what SGLang's aiter backend (_forward_extend_unified) passes in the 70000/300 CC=4 run:
  * one 16k-token prefill chunk at a KV prefix of P tokens (P = 0, 16k, 32k, 48k = chunks 1..4)
  * plus D decode rows (q_len 1, long context): --enable-mixed-chunk batches them into the same call
  * GQA 64 q heads / 8 kv heads, head dim 64, bf16, attention sinks, paged KV (page 64, shuffled
    block table), max_seqlen_k = pages * page_size, window (sw-1, 0) for sliding layers.

Reference per-call times from the server traces (full-attention layer, mixed bs=3/4):
  chunk 1 ~3.7 ms, chunk 2 ~10.3 ms, chunk 3 ~17.0 ms   (GB300 fmhaSm103a: 1.9 / 5.5 / 9.2 ms)

Usage (inside the container, aiter importable):
  python3 bench_prefill_attention.py                                  # table config, full layer
  python3 bench_prefill_attention.py --sw 127                         # sliding-window layer
  python3 bench_prefill_attention.py --cfg '{"BLOCK_M": 512, "num_warps": 8}'
  python3 bench_prefill_attention.py --sweep --prefix 32768           # sweep launch configs
  python3 bench_prefill_attention.py --fp8 ...                        # fp8 KV + fp8 q (the TP1 launch)

--cfg / --sweep swap the launch config inside this process only (they wrap aiter's
get_unified_attention_config); the server is unaffected.  The winner goes into
configs/gfx1250/gluon/attention/unified_attention/DEFAULT.json (the script prints the entry).
"""
import argparse
import itertools
import json
import math
import statistics
import sys

import torch

# launch-config override for this process (None = the table config)
_CFG_OVERRIDE = {"cfg": None}
_TABLE_KEY = {
    ("full", "bf16"): "D_LEQ_64.Q_GEQ_9216.DT_bf16_bf16", ("sw", "bf16"): "D_LEQ_64.Q_GEQ_9216.SW.DT_bf16_bf16",
    ("full", "fp8"): "D_LEQ_64.Q_GEQ_9216.DT_fp8_fp8", ("sw", "fp8"): "D_LEQ_64.Q_GEQ_9216.SW.DT_fp8_fp8",
}
_TILE = {"bf16": 64, "fp8": 128}   # TILE_SIZE_MIN/MAX of the table entries (non-shuffled KV runs one 64-token page)


def install_config_patch():
    """Wrap aiter's config lookup so --cfg / --sweep can replace the gfx1250 2D prefill config."""
    import aiter.ops.triton.attention.unified_attention as ua

    orig = ua.get_unified_attention_config
    seen = {}

    def patched(op, params, backend="triton", arch=None):
        c = orig(op, params, backend=backend, arch=arch)
        if op == "attn_2d" and backend == "gluon" and not params.all_decode:
            if _CFG_OVERRIDE["cfg"]:
                c.update(_CFG_OVERRIDE["cfg"])
            key = (params.sliding_window > 0, json.dumps(c, sort_keys=True))
            if key not in seen:
                seen[key] = 1
                print(f"  [launch config] {'sw' if params.sliding_window > 0 else 'full'}: "
                      + ", ".join(f"{k}={c[k]}" for k in ("BLOCK_M", "num_warps", "LOOP_VARIANT",
                                                          "NUM_BUFFERS", "waves_per_eu", "TILE_SIZE") if k in c),
                      flush=True)
        return c

    ua.get_unified_attention_config = patched


def install_kernel_file(path):
    """Swap the gfx1250 2D kernel aiter's dispatcher launches for the one in `path` (this process only)."""
    import importlib.util
    import os
    import aiter.ops.triton.attention.unified_attention as ua

    name = "ua2d_alt_" + os.path.splitext(os.path.basename(path))[0]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    ua._unified_attention_kernel_2d_gfx1250 = mod._unified_attention_gluon_kernel_2d
    print(f"  [kernel file] {path}", flush=True)


def build_inputs(prefix, chunk, decode_rows, decode_ctx, hq, hkv, d, page, seed=0, device="cuda", fp8_dtype=None):
    """Inputs laid out like SGLang's _forward_extend_unified call.

    fp8_dtype: fp8 KV cache (--kv-cache-dtype fp8_e4m3) and static fp8 query (SGLANG_6003_PREFILL_Q=static):
    q/k/v saturate-cast to fp8 with descale 1.0 (SGLang's static q scale is 1.0; the KV scales default to 1.0).
    """
    g = torch.Generator(device=device).manual_seed(seed)
    q_lens = [chunk] + [1] * decode_rows
    kv_lens = [prefix + chunk] + [decode_ctx] * decode_rows
    pages_per_seq = [math.ceil(k / page) for k in kv_lens]
    max_pages = max(pages_per_seq)
    num_blocks = sum(pages_per_seq) + 1
    perm = torch.randperm(num_blocks, generator=g, device=device).to(torch.int32)
    block_table = torch.zeros(len(q_lens), max_pages, dtype=torch.int32, device=device)
    off = 0
    for i, n in enumerate(pages_per_seq):
        block_table[i, :n] = perm[off:off + n]
        off += n
    k_cache = torch.randn(num_blocks, page, hkv, d, generator=g, device=device, dtype=torch.bfloat16)
    v_cache = torch.randn(num_blocks, page, hkv, d, generator=g, device=device, dtype=torch.bfloat16)
    q = torch.randn(sum(q_lens), hq, d, generator=g, device=device, dtype=torch.bfloat16)
    sinks = (torch.randn(hq, generator=g, device=device) * 2.0).to(torch.bfloat16)
    cu = torch.zeros(len(q_lens) + 1, dtype=torch.int32, device=device)
    cu[1:] = torch.cumsum(torch.tensor(q_lens, device=device, dtype=torch.int32), 0)
    seqused_k = torch.tensor(kv_lens, dtype=torch.int32, device=device)
    out = torch.empty_like(q)                                   # output stays bf16
    descale = {"q": None, "k": None, "v": None}
    q_dec_bf16 = None
    if fp8_dtype is not None:
        fmax = torch.finfo(fp8_dtype).max
        to8 = lambda t: t.float().clamp(-fmax, fmax).to(fp8_dtype)  # noqa: E731  (saturating, like scaled_fp8_quant)
        q_dec_bf16 = q[chunk:].clone()                          # decode rows as forward_decode sees them
        q, k_cache, v_cache = to8(q), to8(k_cache), to8(v_cache)
        descale = {n: torch.ones(1, dtype=torch.float32, device=device) for n in ("q", "k", "v")}
    return dict(q=q, k=k_cache, v=v_cache, out=out, cu=cu, seqused_k=seqused_k,
                block_table=block_table, sinks=sinks, q_lens=q_lens, kv_lens=kv_lens,
                max_seqlen_q=max(q_lens), max_seqlen_k=max_pages * page, descale=descale, q_dec_bf16=q_dec_bf16)


def run(unified_attention, x, sw, scale, split_decode=False):
    window = (sw - 1, 0) if sw > 0 else (-1, -1)
    n_dec = len(x["q_lens"]) - 1
    if split_decode and n_dec > 0:
        # SGLang _forward_extend_unified on a MIXED batch: the prefill seq, then the 1-token decode tails
        # through the decode route (max_seqlen_q=1, q per SGLANG_6003_DECODE_Q = bf16, no q descale).
        T0 = x["q_lens"][0]
        unified_attention(
            q=x["q"][:T0], k=x["k"], v=x["v"], out=x["out"][:T0],
            cu_seqlens_q=x["cu"][:2], max_seqlen_q=T0,
            seqused_k=x["seqused_k"][:1], max_seqlen_k=x["max_seqlen_k"],
            softmax_scale=scale, causal=True, window_size=window,
            block_table=x["block_table"][:1], softcap=0,
            q_descale=x["descale"]["q"], k_descale=x["descale"]["k"], v_descale=x["descale"]["v"],
            sinks=x["sinks"],
        )
        q_dec = x["q_dec_bf16"] if x["q_dec_bf16"] is not None else x["q"][T0:]
        unified_attention(
            q=q_dec, k=x["k"], v=x["v"], out=x["out"][T0:],
            cu_seqlens_q=x["cu_dec"], max_seqlen_q=1,
            seqused_k=x["seqused_k"][1:], max_seqlen_k=x["max_seqlen_k"],
            softmax_scale=scale, causal=True, window_size=window,
            block_table=x["block_table"][1:], softcap=0,
            q_descale=None, k_descale=x["descale"]["k"], v_descale=x["descale"]["v"],
            sinks=x["sinks"],
        )
        return
    unified_attention(
        q=x["q"], k=x["k"], v=x["v"], out=x["out"],
        cu_seqlens_q=x["cu"], max_seqlen_q=x["max_seqlen_q"],
        seqused_k=x["seqused_k"], max_seqlen_k=x["max_seqlen_k"],
        softmax_scale=scale, causal=True, window_size=window,
        block_table=x["block_table"], softcap=0,
        q_descale=x["descale"]["q"], k_descale=x["descale"]["k"], v_descale=x["descale"]["v"],
        sinks=x["sinks"],
    )


def time_ms(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(b))
    return statistics.median(ts)


def attn_flops(x, hq, d, sw):
    """Useful (unmasked) QK + PV FLOPs of the call."""
    total = 0
    for ql, kl in zip(x["q_lens"], x["kv_lens"]):
        ctx = kl - ql
        visible = torch.arange(ql, dtype=torch.int64) + ctx + 1   # keys a causal query sees
        if sw > 0:
            visible = visible.clamp(max=sw)
        total += 4 * d * hq * int(visible.sum())
    return total


def gather_seq_kv(x, i, hkv, d):
    page = x["k"].shape[1]
    n = math.ceil(x["kv_lens"][i] / page)
    idx = x["block_table"][i, :n].long()
    kd = x["descale"]["k"].float().cpu().item() if x["descale"]["k"] is not None else 1.0
    vd = x["descale"]["v"].float().cpu().item() if x["descale"]["v"] is not None else 1.0
    k = x["k"][idx].reshape(-1, hkv, d)[: x["kv_lens"][i]].float() * kd
    v = x["v"][idx].reshape(-1, hkv, d)[: x["kv_lens"][i]].float() * vd
    return k, v


def check(x, hq, hkv, d, sw, scale, rows=48, seed=1):
    """fp32 reference on sampled query rows of every sequence (all heads); worst RELATIVE error.

    Per (row, head): max |out - ref| / max(max |ref|, 0.1).  Relative, because the error is relative: the fp8
    path multiplies P (cast to e4m3, 3 mantissa bits, ~6% rounding) by V, so a row that sees only 1-2 keys
    outputs ~one V vector (|ref| ~ 1-3) with a few % error, while long-context rows average thousands of keys
    to |ref| ~ 0.01-0.05.  The 0.1 floor keeps near-zero rows from inflating the ratio.
    """
    g = torch.Generator().manual_seed(seed)
    nq = hq // hkv
    sinks = x["sinks"].float()
    qd = x["descale"]["q"].float().cpu().item() if x["descale"]["q"] is not None else 1.0
    worst = 0.0
    start = 0
    for i, (ql, kl) in enumerate(zip(x["q_lens"], x["kv_lens"])):
        k, v = gather_seq_kv(x, i, hkv, d)
        if ql == 1:
            pos = [0]
        else:
            pos = sorted({0, 1, ql - 1, ql - 2, *torch.randint(0, ql, (rows,), generator=g).tolist()})
        for p in pos:
            qa = kl - ql + p                                      # absolute key position of this query
            lo = max(0, qa - sw + 1) if sw > 0 else 0
            if i > 0 and x.get("split") and x["q_dec_bf16"] is not None:
                qv = x["q_dec_bf16"][i - 1].float().view(hkv, nq, d)  # decode row, bf16 q (split route)
            else:
                qv = x["q"][start + p].float().view(hkv, nq, d) * qd  # [hkv, nq, d]
            kk, vv = k[lo:qa + 1].permute(1, 0, 2), v[lo:qa + 1].permute(1, 0, 2)  # [hkv, n, d]
            s = torch.einsum("hqd,hkd->hqk", qv, kk) * scale      # [hkv, nq, n]
            s = torch.cat([s, sinks.view(hkv, nq, 1)], dim=-1)
            pr = torch.softmax(s, dim=-1)[..., :-1]
            ref = torch.einsum("hqk,hkd->hqd", pr, vv).reshape(hq, d)
            diff = (x["out"][start + p].float().cpu() - ref.cpu()).abs().amax(dim=-1)   # [hq]
            den = ref.cpu().abs().amax(dim=-1).clamp(min=0.1)
            worst = max(worst, (diff / den).max().item())
        start += ql
    return worst


def sweep_grid():
    seen = set()
    for bm, nw, lv, nb, wpe in itertools.product(
        (64, 128, 256, 512), (4, 8), (0, 1, 2, 3), (2, 3), (1, 2, 4)
    ):
        if bm < 16 * nw or (lv == 0 and nb == 3):
            continue
        key = (bm, nw, lv, nb, wpe)
        if key in seen:
            continue
        seen.add(key)
        yield {"BLOCK_M": bm, "num_warps": nw, "LOOP_VARIANT": lv, "NUM_BUFFERS": nb, "waves_per_eu": wpe}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prefix", type=str, default="0,16384,32768,49152",
                    help="comma list of KV prefix lengths before the chunk")
    ap.add_argument("--chunk", type=int, default=16384)
    ap.add_argument("--decode-rows", type=int, default=3, help="decode rows in the mixed batch (0 = pure EXTEND)")
    ap.add_argument("--decode-ctx", type=int, default=62000)
    ap.add_argument("--sw", type=int, default=0, help="sliding window (gpt-oss sliding layers: 127)")
    ap.add_argument("--hq", type=int, default=64)
    ap.add_argument("--hkv", type=int, default=8)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--page", type=int, default=64)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--cfg", type=str, default="", help="JSON launch config for this layer type")
    ap.add_argument("--sweep", action="store_true", help="sweep launch configs (first prefix only)")
    ap.add_argument("--no-check", action="store_true", help="skip the accuracy check")
    ap.add_argument("--kernel-file", default="",
                    help="use the 2D kernel from this file (e.g. an ablation from make_ablations.py) instead of "
                         "the installed one; this process only")
    ap.add_argument("--split-decode", action="store_true",
                    help="run the decode rows as a separate decode-route call, as the changed SGLang "
                         "_forward_extend_unified does for MIXED batches")
    ap.add_argument("--fp8", action="store_true",
                    help="fp8 KV cache + static fp8 query, as the TP1 launch (--kv-cache-dtype fp8_e4m3, "
                         "SGLANG_6003_PREFILL_Q=static)")
    ap.add_argument("--tol", type=float, default=None,
                    help="max relative error tolerated (default 0.02 bf16, 0.1 fp8: fp8 P has ~6%% rounding); the sweep also accepts up to "
                         "1.5x the table config's own error")
    args = ap.parse_args()

    install_config_patch()
    if args.kernel_file:
        install_kernel_file(args.kernel_file)
    from aiter.ops.triton.attention.unified_attention import unified_attention
    from aiter.ops.triton.utils.types import e4m3_dtype

    dt = "fp8" if args.fp8 else "bf16"
    tol = args.tol if args.tol is not None else (0.1 if args.fp8 else 0.02)
    scale = args.d ** -0.5
    layer = "sw" if args.sw > 0 else "full"
    prefixes = [int(p) for p in args.prefix.split(",")]
    print(f"gpt-oss prefill attention | layer={layer} q/kv={dt} chunk={args.chunk} decode_rows={args.decode_rows}{' (split)' if args.split_decode else ''} "
          f"hq={args.hq} hkv={args.hkv} d={args.d} page={args.page} sinks=on", flush=True)
    if args.chunk < 9216:
        print(f"  NOTE: chunk {args.chunk} < 9216, so the table uses its regular entries, "
              f"not the new {_TABLE_KEY[(layer, dt)]} entry", flush=True)

    def one(prefix, cfg, do_check):
        _CFG_OVERRIDE["cfg"] = cfg
        x = build_inputs(prefix, args.chunk, args.decode_rows, args.decode_ctx,
                         args.hq, args.hkv, args.d, args.page, fp8_dtype=e4m3_dtype if args.fp8 else None)
        x["split"] = args.split_decode
        x["cu_dec"] = torch.arange(args.decode_rows + 1, dtype=torch.int32, device="cuda")
        f = lambda: run(unified_attention, x, args.sw, scale, args.split_decode)  # noqa: E731
        ms = time_ms(f, args.warmup, args.iters)
        err = None
        if do_check:
            f()
            torch.cuda.synchronize()
            err = check(x, args.hq, args.hkv, args.d, args.sw, scale)
        tf = attn_flops(x, args.hq, args.d, args.sw) / (ms * 1e-3) / 1e12
        del x
        torch.cuda.empty_cache()
        return ms, tf, err

    if not args.sweep:
        cfg = json.loads(args.cfg) if args.cfg else None
        print(f"{'prefix':>8} {'ms':>9} {'TFLOP/s':>9} {'rel_err':>9}")
        bad = False
        for p in prefixes:
            ms, tf, err = one(p, cfg, not args.no_check)
            ok = "" if err is None else ("  ok" if err <= tol else "  FAIL")
            bad |= err is not None and err > tol
            print(f"{p:>8} {ms:>9.3f} {tf:>9.1f} {('-' if err is None else f'{err:.2e}'):>9}{ok}", flush=True)
        sys.exit(1 if bad else 0)

    p = prefixes[0]
    print(f"sweep at prefix={p}; each config: time, then accuracy" + ("" if not args.no_check else " (skipped)"))
    base_ms, base_tf, base_err = one(p, None, not args.no_check)
    ok_err = None if base_err is None else max(tol, 1.5 * base_err)
    print(f"  table config (installed DEFAULT.json): {base_ms:.3f} ms, {base_tf:.1f} TFLOP/s"
          + ("" if base_err is None else f", rel_err {base_err:.2e} -> accept <= {ok_err:.2e}"), flush=True)
    results = []
    for cfg in sweep_grid():
        tag = json.dumps(cfg, separators=(",", ":"))
        try:
            ms, tf, err = one(p, cfg, not args.no_check)
            status = "ok" if err is None or err <= ok_err else f"BAD_ACC({err:.1e})"
            results.append((ms, tf, status, tag))
            print(f"  {ms:9.3f} ms {tf:8.1f} TFLOP/s  {status:12s} {tag}", flush=True)
        except Exception as e:  # compile failures (LDS / registers), static_asserts
            msg = str(e).strip().splitlines()[-1][:100] if str(e).strip() else type(e).__name__
            results.append((float("inf"), 0.0, "ERROR", tag))
            print(f"  {'-':>9}    {'-':>8}          ERROR        {tag}  {msg}", flush=True)
    good = sorted(r for r in results if r[2] == "ok")
    print(f"\ntable config (installed DEFAULT.json): {base_ms:.3f} ms, {base_tf:.1f} TFLOP/s")
    print("top 8:")
    for ms, tf, st, tag in good[:8]:
        print(f"  {ms:9.3f} ms  {tf:8.1f} TFLOP/s  x{base_ms / ms:4.2f}  {tag}")
    if good:
        best = json.loads(good[0][3])
        entry = {**best, "TILE_SIZE_MIN": _TILE[dt], "TILE_SIZE_MAX": _TILE[dt]}
        key = _TABLE_KEY[(layer, dt)]
        print(f"\nDEFAULT.json entry for this layer (replace the values of \"{key}\"):")
        print(f'    "{key}": {json.dumps(entry)}')


if __name__ == "__main__":
    main()
