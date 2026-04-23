#!/usr/bin/env python3
"""GDN prefill kernel benchmark: original (FLA) vs opt vs opt_vk.

Measures pure chunk-kernel CUDA time via torch.profiler, with per-kernel
breakdown.  L2-norm is applied once before the timed region so the profiler
only captures chunk kernels themselves.

Backends (all from aiter):
  - chunk_gated_delta_rule        (original, mirrors FLA implementation)
  - chunk_gated_delta_rule_opt    (fused K12/K34, transposed intermediate)
  - chunk_gated_delta_rule_opt_vk (same fused K12/K34, h in [V,K] layout)

Usage:
    python gdr_prefill_kernel_bench.py
    python gdr_prefill_kernel_bench.py --tp 1
    python gdr_prefill_kernel_bench.py --model 35B --seqlen 2500
"""

import argparse

import torch
import torch.nn.functional as F
from torch.profiler import profile, ProfilerActivity

from aiter.ops.triton.gated_delta_net import (
    chunk_gated_delta_rule,
    chunk_gated_delta_rule_opt,
    chunk_gated_delta_rule_opt_vk,
)

# -- Model configs ------------------------------------------------------------

MODELS = {
    "Qwen3.5-35B-A3B": {
        "num_k_heads": 16,
        "num_v_heads": 32,
        "head_k_dim": 128,
        "head_v_dim": 128,
    },
    "Qwen3.5-397B-A17B": {
        "num_k_heads": 16,
        "num_v_heads": 64,
        "head_k_dim": 128,
        "head_v_dim": 128,
    },
}

BACKENDS = {
    "original": chunk_gated_delta_rule,
    "opt": chunk_gated_delta_rule_opt,
    "opt_vk": chunk_gated_delta_rule_opt_vk,
}

# Kernel names that belong to the chunk algorithm itself (exclude l2norm,
# memcpy, fill, dtype-cast elementwise, etc.)
CHUNK_KERNEL_PREFIXES = [
    "chunk_gated_delta_rule_fwd_kernel_h",
    "chunk_fwd_kernel_o",
    "chunk_scaled_dot_kkt_fwd_kernel",
    "chunk_local_cumsum",
    "merge_16x16_to_64x64",
    "recompute_w_u_fwd_kernel",
    "fused_solve_tril_recompute_w_u_kernel",
    "fused_chunk_local_cumsum_scaled_dot_kkt_fwd_kernel",
]


def is_chunk_kernel(name):
    return any(name.startswith(p) for p in CHUNK_KERNEL_PREFIXES)


# -- Helpers ------------------------------------------------------------------


def make_inputs(B, T, H, K, V, dtype):
    """Create prefill inputs with q/k already L2-normalized."""
    q = torch.randn(B, T, H, K, dtype=dtype, device="cuda")
    k = torch.randn(B, T, H, K, dtype=dtype, device="cuda")
    # Pre-normalize so the kernel doesn't need to
    q = F.normalize(q.float(), p=2, dim=-1).to(dtype)
    k = F.normalize(k.float(), p=2, dim=-1).to(dtype)
    v = torch.randn(B, T, H, V, dtype=dtype, device="cuda")
    beta = torch.rand(B, T, H, dtype=dtype, device="cuda").sigmoid()
    g = F.logsigmoid(torch.rand(B, T, H, dtype=torch.float32, device="cuda"))
    h0 = torch.zeros(B, H, K, V, dtype=torch.float32, device="cuda")
    scale = float(K**-0.5)
    return q, k, v, g, beta, h0, scale


def run_fn(backend_name, fn, q, k, v, g, beta, h0, scale):
    if backend_name == "opt_vk":
        state = h0.transpose(-1, -2).contiguous()
    else:
        state = h0.clone()
    o, _ = fn(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=state,
        output_final_state=False,
        use_qk_l2norm_in_kernel=False,
    )
    return o


def profile_backend(backend_name, fn, q, k, v, g, beta, h0, scale, warmup, niters):
    """Profile and return (kernel_list, total_us, chunk_only_us)."""
    for _ in range(warmup):
        run_fn(backend_name, fn, q, k, v, g, beta, h0, scale)
    torch.cuda.synchronize()

    with profile(activities=[ProfilerActivity.CUDA], record_shapes=False) as prof:
        for _ in range(niters):
            run_fn(backend_name, fn, q, k, v, g, beta, h0, scale)
    torch.cuda.synchronize()

    kernels = []
    total_us = 0.0
    chunk_us = 0.0
    for evt in prof.key_averages():
        if evt.device_type is not None and "cuda" in str(evt.device_type).lower():
            avg_us = evt.self_device_time_total / niters
            is_chunk = is_chunk_kernel(evt.key)
            kernels.append((evt.key, evt.count, avg_us, is_chunk))
            total_us += avg_us
            if is_chunk:
                chunk_us += avg_us

    kernels.sort(key=lambda x: x[2], reverse=True)
    return kernels, total_us, chunk_us


def validate(q, k, v, g, beta, h0, scale):
    o_orig = run_fn("original", BACKENDS["original"], q, k, v, g, beta, h0, scale)
    o_opt = run_fn("opt", BACKENDS["opt"], q, k, v, g, beta, h0, scale)
    o_vk = run_fn("opt_vk", BACKENDS["opt_vk"], q, k, v, g, beta, h0, scale)
    torch.cuda.synchronize()
    diff_opt = (o_orig.float() - o_opt.float()).abs().max().item()
    diff_vk = (o_orig.float() - o_vk.float()).abs().max().item()
    return diff_opt, diff_vk


# -- Main ---------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(
        description="GDN prefill benchmark: original vs opt vs opt_vk"
    )
    p.add_argument(
        "--model", nargs="+", default=["35B", "397B"], choices=["35B", "397B", "all"]
    )
    p.add_argument("--tp", type=int, nargs="+", default=[1, 2])
    p.add_argument("--seqlen", type=int, nargs="+", default=[2500, 60000])
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--niters", type=int, default=20)
    p.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Show top-K kernels in per-kernel breakdown",
    )
    return p.parse_args()


def main():
    cli = parse_args()
    if "all" in cli.model:
        model_keys = list(MODELS.keys())
    else:
        model_keys = [k for k in MODELS if any(m in k for m in cli.model)]

    B = 1
    dtype = torch.bfloat16
    gpu = torch.cuda.get_device_name(0)

    print(f"GPU: {gpu}")
    print(f"Batch: {B}  |  Warmup: {cli.warmup}  |  Measure iters: {cli.niters}")
    print("Backends: original (FLA), opt, opt_vk")
    print("L2-norm applied BEFORE profiling (use_qk_l2norm_in_kernel=False)")
    print()

    all_summary = []  # for markdown

    for model_name in model_keys:
        cfg = MODELS[model_name]
        for tp in cli.tp:
            H = cfg["num_v_heads"] // tp
            K = cfg["head_k_dim"]
            V = cfg["head_v_dim"]
            if H < 1:
                print(f"[SKIP] {model_name} TP={tp}: heads < 1 after split\n")
                continue

            for T in cli.seqlen:
                header = (
                    f"{model_name}  TP={tp}  (H={H}, K={K}, V={V})  " f"B={B}  T={T}"
                )
                sep = 110
                print("=" * sep)
                print(header)
                print("=" * sep)

                # Validate
                torch.manual_seed(42)
                q, k_, v_, g, beta, h0, scale = make_inputs(B, T, H, K, V, dtype)
                try:
                    diff_opt, diff_vk = validate(q, k_, v_, g, beta, h0, scale)
                    print(
                        f"Validation vs original:  opt={diff_opt:.2e}  opt_vk={diff_vk:.2e}"
                    )
                except Exception as e:
                    print(f"Validation FAILED: {e}")
                    torch.cuda.empty_cache()
                    print()
                    continue

                # Profile each backend
                row = {"model": model_name, "tp": tp, "H": H, "T": T}
                for bname, fn in BACKENDS.items():
                    torch.cuda.empty_cache()
                    torch.manual_seed(42)
                    q, k_, v_, g, beta, h0, scale = make_inputs(B, T, H, K, V, dtype)
                    try:
                        kernels, total_us, chunk_us = profile_backend(
                            bname,
                            fn,
                            q,
                            k_,
                            v_,
                            g,
                            beta,
                            h0,
                            scale,
                            cli.warmup,
                            cli.niters,
                        )
                        row[bname] = chunk_us

                        print(
                            f"\n  [{bname}]  chunk kernels: {chunk_us:.1f} us   "
                            f"(all CUDA: {total_us:.1f} us)"
                        )
                        print(
                            f"  {'Kernel':<70s} {'Calls':>6} {'Avg(us)':>9} {'%chunk':>7}"
                        )
                        print(f"  {'-'*70} {'-'*6} {'-'*9} {'-'*7}")
                        for kname, cnt, avg, is_ck in kernels[: cli.top_k]:
                            tag = "" if is_ck else " *"
                            pct = avg / chunk_us * 100 if chunk_us > 0 and is_ck else 0
                            pct_s = f"{pct:>5.1f}%" if is_ck else "   ---"
                            print(
                                f"  {kname[:68]:<70s} {cnt:>6} {avg:>9.1f} {pct_s}{tag}"
                            )
                        if len(kernels) > cli.top_k:
                            rest_ck = sum(a for _, _, a, c in kernels[cli.top_k :] if c)
                            rest_ot = sum(
                                a for _, _, a, c in kernels[cli.top_k :] if not c
                            )
                            if rest_ck > 0:
                                print(
                                    f"  {'... (other chunk kernels)':<70s} {'':>6} "
                                    f"{rest_ck:>9.1f} {rest_ck/chunk_us*100:>5.1f}%"
                                )
                            if rest_ot > 0:
                                print(
                                    f"  {'... (other non-chunk kernels)':<70s} {'':>6} "
                                    f"{rest_ot:>9.1f}    --- *"
                                )
                        print("  (* = overhead, not counted in chunk total)")
                    except Exception as e:
                        print(f"\n  [{bname}] ERROR: {e}")
                        row[bname] = None
                        torch.cuda.empty_cache()

                # Summary
                orig = row.get("original")
                print("\n  --- Chunk kernel time (us/iter) ---")
                for bname in ["original", "opt", "opt_vk"]:
                    t = row.get(bname)
                    if t is None:
                        print(f"  {bname:<12s}  ERROR")
                    elif orig and orig > 0 and bname != "original":
                        print(
                            f"  {bname:<12s}  {t:>9.1f} us   "
                            f"speedup vs original: {orig/t:.2f}x"
                        )
                    else:
                        print(f"  {bname:<12s}  {t:>9.1f} us   (baseline)")
                print()
                all_summary.append(row)

    # Markdown
    print("=" * 100)
    print("Markdown Summary (chunk kernel only, us/iter)")
    print("=" * 100)
    print(
        "| Model | TP | H | T | original (us) | opt (us) | opt_vk (us) "
        "| opt spdup | opt_vk spdup |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in all_summary:
        orig = r.get("original")
        opt = r.get("opt")
        vk = r.get("opt_vk")

        def fmt(v):
            return f"{v:.1f}" if v is not None else "ERR"

        def spd(base, v):
            if base and v and v > 0:
                return f"{base/v:.2f}x"
            return "-"

        print(
            f"| {r['model']} | {r['tp']} | {r['H']} | {r['T']} "
            f"| {fmt(orig)} | {fmt(opt)} | {fmt(vk)} "
            f"| {spd(orig, opt)} | {spd(orig, vk)} |"
        )
    print()


if __name__ == "__main__":
    main()
