#!/usr/bin/env python3
"""Effect of disabling `packed-fp32-ops` on the Gluon fp8 MQA-logits kernel.

LLVM's SLP vectorizer pairs the kernel's independent per-row FP32 reduction
chains into `v_pk_fma_f32`. On gfx950 a packed FP32 op is the one VALU form that
can never co-issue in an MFMA shadow, so every one of them is exposed latency
between MFMAs -- work the scheduler would otherwise have hidden for free.

Disabling the `packed-fp32-ops` subtarget feature on the kernel function leaves
ISel nothing to select and makes the SLP cost model see no cheap `<2 x float>`
op, so the pairing never forms. This script compiles the same kernel both ways,
dumps every IR stage, and times it.

  python3 repro_fp8_mqa_disable_pk.py --both
  python3 repro_fp8_mqa_disable_pk.py --packing off      # one arm

IRs land in `<triton version>_triton/{with,without}_packing/`, timings in
`<triton version>_triton/result_{with,without}_packing.json`.

Each arm runs in its own process with its own `TRITON_CACHE_DIR`. That is not
tidiness: Triton's cache key does not include the target-feature change, so a
shared cache hands back whichever binary was compiled first and the flag looks
like it did nothing.
"""
import argparse
import json
import os
import re
import statistics
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

# glm-shaped: 32 query heads, head_size 128. One long-context launch, which is
# where the exposed packed ops cost the most because the KV walk dominates.
NUM_HEADS = 32
HEAD_SIZE = 128
BATCH, SEQ_Q, SEQ_KV = 1, 8192, 32768

KERNEL_NAME = "_gluon_fp8_mqa_logits_kernel"


def _build_id():
    """Something that identifies the compiler beyond its version string.

    A wheel carries its commit in the local version (`+amd.rocm7.2.0.gitXXXX`);
    a source build does not, so fall back to the git HEAD of the tree it was
    built from if that is still around.
    """
    import triton

    v = triton.__version__
    if "+" in v:
        return v
    import os
    import subprocess

    root = os.path.dirname(os.path.dirname(os.path.abspath(triton.__file__)))
    for guess in (root, "/opt/triton-tot"):
        try:
            out = subprocess.run(["git", "-C", guess, "rev-parse", "HEAD"],
                                 capture_output=True, text=True, timeout=10)
            if out.returncode == 0:
                return f"{v}+git{out.stdout.strip()[:10]}"
        except Exception:
            pass
    return v


def print_irs_to_files(compiled_kernel, prefix):
    """Every textual IR stage, one file each. `hsaco` is the ELF, so it is
    written as bytes rather than as its Python repr."""
    for key in compiled_kernel.asm.keys():
        val = compiled_kernel.asm[key]
        if isinstance(val, (bytes, bytearray)):
            with open(f"{prefix}_{key}.bin", "wb") as fptr:
                fptr.write(val)
            continue
        with open(f"{prefix}_{key}.txt", "w") as fptr:
            print(val, file=fptr)


def build_inputs():
    import torch

    s_q, s_k = BATCH * SEQ_Q, BATCH * SEQ_KV
    torch.manual_seed(0)
    q = torch.randn(s_q, NUM_HEADS, HEAD_SIZE, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(s_k, HEAD_SIZE, device="cuda", dtype=torch.bfloat16)
    weights = torch.randn(s_q, NUM_HEADS, device="cuda", dtype=torch.float32)

    # causal windows inside each batch element, concatenated
    cu_starts = torch.zeros(s_q, dtype=torch.int, device="cuda")
    cu_ends = torch.zeros(s_q, dtype=torch.int, device="cuda")
    ar = torch.arange(SEQ_Q, dtype=torch.int, device="cuda")
    for b in range(BATCH):
        qs, kvs = b * SEQ_Q, b * SEQ_KV
        cu_starts[qs : qs + SEQ_Q] = kvs
        cu_ends[qs : qs + SEQ_Q] = kvs + (SEQ_KV - SEQ_Q) + ar + 1
    flops = 2.0 * float((cu_ends - cu_starts).clamp(min=0).sum()) * NUM_HEADS * HEAD_SIZE

    # per-token fp8 scaling along the KV rows, as the kernel expects
    amax = kv.abs().amax(dim=1, keepdim=True).float().clamp(min=1e-4)
    scale = amax / 448.0
    kv8 = (kv / scale).to(torch.float8_e4m3fn)
    kv_scales = scale.squeeze(1).contiguous()
    q8 = q.to(torch.float8_e4m3fn)
    del q, kv
    torch.cuda.empty_cache()
    return q8, kv8, kv_scales, weights, cu_starts, cu_ends, flops


# Instruction classes, in priority order -- the first pattern that matches an
# instruction owns it, so `v_pk_*` and the move ops are taken out of the VALU
# bucket before the catch-all sees them.
CLASSES = (
    (r"v_mfma[a-z0-9_]*", "v_mfma"),
    (r"v_pk_[a-z0-9_]+", "v_pk_*"),
    (r"v_accvgpr_(read|write)[a-z0-9_]*", "v_accvgpr move"),
    (r"v_mov_b[0-9]+", "v_mov"),
    (r"v_[a-z0-9_]+", "VALU (other)"),
    (r"buffer_load[a-z0-9_]*", "buffer_load"),
    (r"buffer_store[a-z0-9_]*", "buffer_store"),
    (r"ds_(read|write)[a-z0-9_]*", "ds_read/write"),
    (r"scratch_(load|store)[a-z0-9_]*", "scratch (spill)"),
    (r"s_waitcnt[a-z0-9_]*", "s_waitcnt"),
    (r"s_barrier[a-z0-9_]*", "s_barrier"),
    (r"s_(?!waitcnt|barrier|nop|endpgm|branch|cbranch)[a-z0-9_]+", "SALU"),
)


def classify(line):
    body = line.strip()
    for pat, name in CLASSES:
        if re.match(pat + r"\b", body):
            return name
    return None


def loop_body(asm):
    """The steady-state loop: the longest span closed by a backward branch."""
    lines = asm.splitlines()
    labels = {}
    for i, ln in enumerate(lines):
        m = re.match(r"^(\.\S+):", ln)
        if m:
            labels[m.group(1)] = i
    best = None
    for i, ln in enumerate(lines):
        m = re.search(r"s_cbranch\w*\s+(\.\S+)", ln)
        if m and m.group(1) in labels and labels[m.group(1)] < i:
            span = i - labels[m.group(1)]
            if best is None or span > best[0]:
                best = (span, labels[m.group(1)], i)
    return lines[best[1] : best[2] + 1] if best else lines


def tally(lines):
    out = {name: 0 for _p, name in CLASSES}
    for ln in lines:
        c = classify(ln)
        if c:
            out[c] += 1
    out["VALU total"] = (out["v_pk_*"] + out["v_accvgpr move"] + out["v_mov"]
                         + out["VALU (other)"])
    return out


def count_insts(asm):
    out = {}
    whole = tally(asm.splitlines())
    body = loop_body(asm)
    loop = tally(body)
    out["loop_lines"] = len(body)
    for k, v in whole.items():
        out[k] = v
    out["loop"] = loop
    # kept flat for the summary table
    out["v_pk_*"] = whole["v_pk_*"]
    out["v_mfma"] = whole["v_mfma"]
    for key, pat in (("vgpr", r"\.vgpr_count:\s*(\d+)"),
                     ("agpr", r"\.agpr_count:\s*(\d+)"),
                     ("spill", r"\.vgpr_spill_count:\s*(\d+)")):
        m = re.search(pat, asm)
        out[key] = int(m.group(1)) if m else -1
    return out


def run_one(packing, warmup, iters, reps, unroll=1, label=None):
    """Compile + dump + time one arm. Returns the result dict."""
    import triton  # noqa: F401  (import after TRITON_CACHE_DIR is set)
    import torch

    sys.path.insert(0, HERE)
    # Pin the loop shape first: at aiter's shipped UNROLL=2 the two packing arms
    # do not compile to the same body on 3.7.0, which would leave the unroll
    # factor varying alongside the thing under test.
    import force_unroll

    force_unroll.enable(unroll)
    if not packing:
        import no_packed_fp32

        no_packed_fp32.enable(KERNEL_NAME)

    from aiter.ops.triton.attention.fp8_mqa_logits import fp8_mqa_logits
    import aiter.ops.triton.attention.fp8_mqa_logits as mod

    ver = label or triton.__version__.split("+")[0]
    tag = "with_packing" if packing else "without_packing"
    outdir = os.path.join(HERE, f"{ver}_triton", tag)
    os.makedirs(outdir, exist_ok=True)

    q8, kv8, kv_scales, weights, cu_starts, cu_ends, flops = build_inputs()

    def call():
        return fp8_mqa_logits(q8, kv8, kv_scales, weights, cu_starts, cu_ends, True)

    out = call()
    torch.cuda.synchronize()

    # the compiled kernel, straight out of the JIT cache
    compiled = None
    kern = getattr(mod, KERNEL_NAME, None)
    if kern is not None:
        for _dev, cache in kern.device_caches.items():
            for _k, c in cache[0].items():
                compiled = c
    if compiled is None:
        raise RuntimeError(f"{KERNEL_NAME} not found in the JIT cache")
    print_irs_to_files(compiled, os.path.join(outdir, KERNEL_NAME))

    asm = compiled.asm["amdgcn"]
    stats = count_insts(asm)

    from torch.profiler import ProfilerActivity, profile

    def timed():
        for _ in range(warmup):
            call()
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(iters):
                call()
            torch.cuda.synchronize()
        ev = [e for e in prof.key_averages()
              if KERNEL_NAME in e.key and e.self_device_time_total > 0]
        if len(ev) != 1:
            raise RuntimeError(f"expected 1 {KERNEL_NAME}, got {[e.key for e in ev]}")
        return ev[0].self_device_time_total / ev[0].count

    us = statistics.median([timed() for _ in range(reps)])

    res = {
        "triton": triton.__version__,
        "triton_build": _build_id(),
        "packing": packing,
        "shape": f"{BATCH}x{SEQ_Q}x{SEQ_KV}",
        "unroll": unroll,
        "num_heads": NUM_HEADS,
        "head_size": HEAD_SIZE,
        "us": us,
        "tflops": flops / us / 1e6,
        "out_checksum": float(out[out != float("-inf")].double().abs().sum()),
        **stats,
    }
    with open(os.path.join(HERE, f"{ver}_triton", f"result_{tag}.json"), "w") as f:
        json.dump(res, f, indent=1)

    print(f"\n  triton         {res['triton']}")
    print(f"  kernel         {KERNEL_NAME}")
    print(f"  packed-fp32    {'ENABLED (default)' if packing else 'DISABLED'}")
    print(f"  shape          {res['shape']}, {NUM_HEADS} heads, head_size {HEAD_SIZE}")
    print(f"  UNROLL         {unroll}")
    print(f"  time           {us:.1f} us   ({res['tflops']:.0f} TFLOP/s)")
    print(f"  vgpr / agpr    {stats['vgpr']} / {stats['agpr']}   spill {stats['spill']}")
    print(f"  IRs            {outdir}/")
    lp = stats["loop"]
    print(f"\n  steady-state loop body: {stats['loop_lines']} lines")
    for k in ("v_mfma", "v_pk_*", "VALU (other)", "v_mov", "v_accvgpr move",
              "VALU total", "buffer_load", "buffer_store", "ds_read/write",
              "scratch (spill)", "s_waitcnt", "SALU"):
        print(f"    {k:18} {lp[k]:>5}")
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--packing", choices=["on", "off"], default="on")
    ap.add_argument("--label", default=None,
                    help="output folder name instead of the triton version "
                         "(a source build's version string does not say which "
                         "commit it is)")
    ap.add_argument("--unroll", type=int, default=1,
                    help="pin the KV loop unroll factor (1 = every arm gets the "
                         "same 4-MFMA body; 2 = what aiter ships)")
    ap.add_argument("--both", action="store_true",
                    help="run both arms, each in its own process and cache")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--rounds", type=int, default=3,
                    help="with --both: alternating A/B rounds, median over them")
    args = ap.parse_args()

    if args.both:
        import triton
        ver = args.label or triton.__version__.split("+")[0]
        # Alternate the arms across rounds instead of running each once. This
        # box is shared, and identical shapes have measured 30% apart on it
        # within half an hour; alternating means a drift in load lands on both
        # arms rather than on whichever went second.
        seen = {"on": [], "off": []}
        last = {}
        for rnd in range(args.rounds):
            for packing in (("on", "off") if rnd % 2 == 0 else ("off", "on")):
                env = dict(os.environ)
                # a cache per arm: the cache key does not see the target feature
                env["TRITON_CACHE_DIR"] = os.path.join(
                    HERE, ".triton_cache", f"packing_{packing}")
                cmd = [sys.executable, os.path.abspath(__file__),
                       "--packing", packing, "--warmup", str(args.warmup),
                       "--iters", str(args.iters), "--reps", str(args.reps),
                       "--unroll", str(args.unroll)]
                if args.label:
                    cmd += ["--label", args.label]
                label = "ENABLED" if packing == "on" else "DISABLED"
                print(f"\n=== round {rnd + 1}/{args.rounds}: packed-fp32-ops "
                      f"{label} ===", flush=True)
                r = subprocess.run(cmd, env=env)
                if r.returncode:
                    return r.returncode
                tag = "with_packing" if packing == "on" else "without_packing"
                f = os.path.join(HERE, f"{ver}_triton", f"result_{tag}.json")
                rec = json.load(open(f))
                seen[packing].append(rec["us"])
                last[packing] = rec
        results = []
        for packing in ("on", "off"):
            if seen[packing]:
                rec = dict(last[packing])
                rec["us_rounds"] = seen[packing]
                rec["us"] = statistics.median(seen[packing])
                rec["tflops"] = rec["tflops"] * last[packing]["us"] / rec["us"]
                tag = "with_packing" if packing == "on" else "without_packing"
                json.dump(rec, open(os.path.join(
                    HERE, f"{ver}_triton", f"result_{tag}.json"), "w"), indent=1)
                results.append(rec)
        if len(results) == 2:
            a, b = results
            print(f"\n=== summary, triton {a['triton']} ===")
            print(f"{'':16} {'with packing':>14} {'without':>14}")
            print(f"{'time (us)':16} {a['us']:>14.1f} {b['us']:>14.1f}")
            print(f"{'TFLOP/s':16} {a['tflops']:>14.0f} {b['tflops']:>14.0f}")
            print(f"{'v_pk_*':16} {a['v_pk_*']:>14} {b['v_pk_*']:>14}")
            print(f"{'vgpr':16} {a['vgpr']:>14} {b['vgpr']:>14}")
            print(f"{'spill':16} {a['spill']:>14} {b['spill']:>14}")
            print(f"\n{'in steady loop':16} {'with packing':>14} {'without':>14} "
                  f"{'delta':>8}")
            for k in ("v_mfma", "v_pk_*", "VALU (other)", "v_mov",
                      "v_accvgpr move", "VALU total", "buffer_load",
                      "scratch (spill)", "s_waitcnt"):
                x, y = a["loop"][k], b["loop"][k]
                print(f"{k:16} {x:>14} {y:>14} {y - x:>+8}")
            # The two arms can land on different unroll factors, so raw loop
            # counts are not comparable -- normalise by the MFMA count, which
            # is the work the loop actually does.
            ma, mb = a["loop"]["v_mfma"], b["loop"]["v_mfma"]
            if ma and mb:
                print(f"\n{'per v_mfma':16} {'with packing':>14} {'without':>14} "
                      f"{'change':>8}")
                for k in ("VALU total", "VALU (other)", "v_pk_*", "buffer_load"):
                    x, y = a["loop"][k] / ma, b["loop"][k] / mb
                    ch = f"{y / x:>7.2f}x" if x else "     --"
                    print(f"{k:16} {x:>14.2f} {y:>14.2f} {ch}")
                # unpacking one v_pk_* costs exactly one extra scalar op; any
                # VALU growth past that is the compiler doing something else
                pred = (a["loop"]["VALU total"] + a["loop"]["v_pk_*"]) / ma
                got = b["loop"]["VALU total"] / mb
                print(f"\n  VALU/MFMA predicted from unpacking alone: {pred:.2f}")
                print(f"  VALU/MFMA actually emitted:               {got:.2f}"
                      f"   ({got - pred:+.2f} beyond unpacking)")
            print(f"{'us per round':16} {str(a['us_rounds']):>14} "
                  f"{str(b['us_rounds']):>14}")
            print(f"\nspeedup from disabling packed-fp32-ops: "
                  f"{a['us'] / b['us']:.3f}x")
            if a["out_checksum"] != b["out_checksum"]:
                print(f"WARNING: outputs differ "
                      f"({a['out_checksum']} vs {b['out_checksum']})")
            else:
                print("outputs identical")
        return 0

    if "TRITON_CACHE_DIR" not in os.environ:
        os.environ["TRITON_CACHE_DIR"] = os.path.join(
            HERE, ".triton_cache", f"packing_{args.packing}")
    run_one(args.packing == "on", args.warmup, args.iters, args.reps,
            args.unroll, args.label)
    return 0


if __name__ == "__main__":
    sys.exit(main())
