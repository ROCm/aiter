# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
#
# Standalone example / test harness for the jagged_dense_bmm BACKWARD pass.
#
# Given the upstream gradient dOut (L, N) of the forward
#     Out[s:e, :] = Jagged[s:e, :] @ Dense[b] + Bias[b][None, :]
# it computes, per group b over its packed row slice [s, e):
#     dJagged[s:e, :] = dOut[s:e, :] @ Dense[b].T        (M_b x K)
#     dDense[b]       = Jagged[s:e, :].T @ dOut[s:e, :]   (K x N)
#     dBias[b]        = sum_m dOut[s:e, :]                (N,)
#
# This harness is built test-first: it provides torch references for all three
# gradients, cross-checks those references against torch.autograd on the forward
# (so the references are themselves trusted), and wires up the FlyDSL launchers.
# Until the kernels exist, the FlyDSL path reports SKIPPED (NotImplementedError)
# while the reference-vs-autograd check still runs and must pass.
#
# Run inside the project venv:
#     source flydsl_venv/bin/activate
#     python aiter/aiter/ops/flydsl/kernels/example_jagged_dense_bmm_bwd.py
#
# Lives next to jagged_dense_bmm.py / jagged_dense_bmm_bwd.py and imports them as
# sibling modules so it does not pull in the full aiter package.

from __future__ import annotations

import argparse
import sys
import time

import torch

import flydsl.compiler as flyc

# Sibling imports (script dir is on sys.path[0]); avoids importing the aiter pkg.
from example_jagged_dense_bmm import make_seq_offsets
from jagged_dense_bmm import BLOCK_M, K, N
from jagged_dense_bmm_bwd import SPLIT, grad_dense_bias, grad_jagged

GRADS = ("djagged", "ddense", "dbias")

# Timing targets map the launchers onto the two distinct kernels: dJagged is a GEMM,
# while dDense + dBias share the single fused grad_dense_bias launcher.
BENCH_TARGETS = ("djagged", "dense_bias")


def make_inputs(n_groups, max_seq_len, regime, seed, device):
    """Build forward inputs plus the upstream gradient dOut (L, N)."""
    torch.manual_seed(0)
    seq_offsets = make_seq_offsets(n_groups, max_seq_len, regime, seed, device)
    total_rows = int(seq_offsets[-1].item())
    jagged = torch.randn(max(total_rows, 1), K, dtype=torch.bfloat16, device=device)
    dense = torch.randn(n_groups, K, N, dtype=torch.bfloat16, device=device)
    bias = torch.randn(n_groups, N, dtype=torch.bfloat16, device=device)
    d_out = torch.randn(max(total_rows, 1), N, dtype=torch.bfloat16, device=device)
    return jagged, dense, bias, d_out, seq_offsets, total_rows


# --- Trusted torch references (fp32 accumulate, matching the kernel math) ------


def ref_grad_jagged(d_out, dense, seq_offsets, n_groups):
    L = d_out.shape[0]
    dj = torch.zeros((L, K), dtype=torch.float32, device=d_out.device)
    for b in range(n_groups):
        s = int(seq_offsets[b].item())
        e = int(seq_offsets[b + 1].item())
        if e > s:
            # (M_b, N) @ (N, K) = (M_b, K)
            dj[s:e] = d_out[s:e].float() @ dense[b].float().t()
    return dj


def ref_grad_dense(jagged, d_out, seq_offsets, n_groups):
    dd = torch.zeros((n_groups, K, N), dtype=torch.float32, device=jagged.device)
    for b in range(n_groups):
        s = int(seq_offsets[b].item())
        e = int(seq_offsets[b + 1].item())
        if e > s:
            # (K, M_b) @ (M_b, N) = (K, N)
            dd[b] = jagged[s:e].float().t() @ d_out[s:e].float()
    return dd


def ref_grad_bias(d_out, seq_offsets, n_groups):
    db = torch.zeros((n_groups, N), dtype=torch.float32, device=d_out.device)
    for b in range(n_groups):
        s = int(seq_offsets[b].item())
        e = int(seq_offsets[b + 1].item())
        if e > s:
            db[b] = d_out[s:e].float().sum(dim=0)
    return db


def autograd_grads(jagged, dense, bias, d_out, seq_offsets, n_groups):
    """Ground-truth gradients via torch.autograd on the forward, used to verify
    the hand-written references above. Everything in fp32."""
    j = jagged.float().clone().requires_grad_(True)
    d = dense.float().clone().requires_grad_(True)
    bi = bias.float().clone().requires_grad_(True)
    loss = j.new_zeros(())
    for b in range(n_groups):
        s = int(seq_offsets[b].item())
        e = int(seq_offsets[b + 1].item())
        if e > s:
            out_b = j[s:e] @ d[b] + bi[b][None, :]
            # <out, dOut> so that d(loss)/d(out) == dOut exactly.
            loss = loss + (out_b * d_out[s:e].float()).sum()
    loss.backward()
    return j.grad, d.grad, bi.grad


# --- Validation helpers -------------------------------------------------------


def cosine_maxabs(ref, got):
    ref = ref.float().flatten()
    got = got.float().flatten()
    cos = torch.nn.functional.cosine_similarity(ref, got, dim=0).item()
    max_abs = (ref - got).abs().max().item()
    return cos, max_abs


def report(name, ref, got, cos_thresh=0.999):
    cos, max_abs = cosine_maxabs(ref, got)
    ok = cos > cos_thresh
    print(f"  {name:<10} {'PASS' if ok else 'FAIL'}  cosine={cos:.6f}  max_abs_err={max_abs:.4f}")
    return ok


# --- FlyDSL host wiring (calls launchers; SKIPPED until kernels exist) ---------


def run_flydsl_bwd(which, jagged, dense, bias, d_out, seq_offsets, n_groups, max_seq_len):
    """Prepare device buffers and invoke the FlyDSL backward launchers.

    Returns a dict {grad_name: tensor or None}. A None entry means the kernel is
    not implemented yet (NotImplementedError) and the FlyDSL path is skipped.
    """
    device = jagged.device
    total_rows = jagged.shape[0]
    stream = torch.cuda.current_stream()
    results = {g: None for g in GRADS}

    tDOut = flyc.from_dlpack(d_out).mark_layout_dynamic(leading_dim=1, divisibility=8)

    if "djagged" in which:
        # RHS is Dense[b] in its (K, N) orientation: pass the plain dense as a
        # tall (n_groups * K, N) matrix (K-major per group).
        dense_kn = dense.reshape(n_groups * K, N).contiguous()
        # Pad output rows by BLOCK_M so a partial tail-tile store stays in-bounds.
        d_jagged = torch.zeros(total_rows + BLOCK_M, K, dtype=torch.bfloat16, device=device)
        tDJ = flyc.from_dlpack(d_jagged).mark_layout_dynamic(leading_dim=1, divisibility=8)
        try:
            grad_jagged(tDJ, tDOut, dense_kn, seq_offsets, n_groups, max_seq_len, stream=stream)
            torch.cuda.synchronize()
            results["djagged"] = d_jagged[:total_rows]
        except NotImplementedError as ex:
            print(f"  [skip] grad_jagged: {ex}")

    # dDense and dBias are produced by a single fused launcher: the dBias column-sums
    # piggyback on the dDense partials pass (both reduce over m). Run it once if either
    # gradient is requested, then hand back whichever pieces were asked for.
    if ("ddense" in which) or ("dbias" in which):
        d_dense = torch.zeros(n_groups, K, N, dtype=torch.bfloat16, device=device)
        d_bias = torch.zeros(n_groups, N, dtype=torch.bfloat16, device=device)
        # fp32 split-reduction scratch: dDense (n_groups*SPLIT*K, N), dBias (n_groups*SPLIT, N).
        dense_partials = torch.zeros(n_groups * SPLIT * K, N, dtype=torch.float32, device=device)
        bias_partials = torch.zeros(n_groups * SPLIT, N, dtype=torch.float32, device=device)
        tJagged = flyc.from_dlpack(jagged).mark_layout_dynamic(leading_dim=1, divisibility=8)
        try:
            grad_dense_bias(
                d_dense.view(n_groups * K, N), d_bias, tJagged, tDOut, seq_offsets,
                dense_partials, bias_partials, n_groups, max_seq_len, stream=stream,
            )
            torch.cuda.synchronize()
            if "ddense" in which:
                results["ddense"] = d_dense
            if "dbias" in which:
                results["dbias"] = d_bias
        except NotImplementedError as ex:
            print(f"  [skip] grad_dense_bias: {ex}")

    return results


def _bench_launchers(which, jagged, dense, d_out, seq_offsets, n_groups, max_seq_len):
    """Build zero-arg launch closures for the requested timing target(s).

    Buffers are allocated once; the closure issues only the kernel(s) under test so a
    warmup + timed loop measures steady-state kernel cost (compile happens in warmup).
    """
    device = jagged.device
    total_rows = jagged.shape[0]
    stream = torch.cuda.current_stream()
    tDOut = flyc.from_dlpack(d_out).mark_layout_dynamic(leading_dim=1, divisibility=8)
    launchers = {}

    if "djagged" in which:
        dense_kn = dense.reshape(n_groups * K, N).contiguous()
        d_jagged = torch.zeros(total_rows + BLOCK_M, K, dtype=torch.bfloat16, device=device)
        tDJ = flyc.from_dlpack(d_jagged).mark_layout_dynamic(leading_dim=1, divisibility=8)

        def run_djagged():
            grad_jagged(tDJ, tDOut, dense_kn, seq_offsets, n_groups, max_seq_len, stream=stream)

        launchers["djagged"] = run_djagged

    if "dense_bias" in which:
        d_dense = torch.zeros(n_groups, K, N, dtype=torch.bfloat16, device=device)
        d_bias = torch.zeros(n_groups, N, dtype=torch.bfloat16, device=device)
        dense_partials = torch.zeros(n_groups * SPLIT * K, N, dtype=torch.float32, device=device)
        bias_partials = torch.zeros(n_groups * SPLIT, N, dtype=torch.float32, device=device)
        tJagged = flyc.from_dlpack(jagged).mark_layout_dynamic(leading_dim=1, divisibility=8)
        d_dense_v = d_dense.view(n_groups * K, N)

        def run_dense_bias():
            grad_dense_bias(d_dense_v, d_bias, tJagged, tDOut, seq_offsets, dense_partials,
                            bias_partials, n_groups, max_seq_len, stream=stream)

        launchers["dense_bias"] = run_dense_bias

    return launchers


def bench_flydsl_bwd(which, jagged, dense, d_out, seq_offsets, n_groups, max_seq_len,
                     warmup=10, iters=50):
    """Time each FlyDSL backward target and print us/iter + TFLOP/s.

    dJagged and dDense are each a 2*L*K*N GEMM; the fused dense_bias target's dBias
    sum (~L*N adds) is negligible against the dDense GEMM, so both targets are scored
    on 2*L*K*N. L = packed rows (sum of M_b), so skew is scored on the work it does.
    """
    L = jagged.shape[0]
    launchers = _bench_launchers(which, jagged, dense, d_out, seq_offsets, n_groups, max_seq_len)
    print(f"timing (warmup={warmup}, iters={iters}):")
    for name in BENCH_TARGETS:
        if name not in launchers:
            continue
        launch = launchers[name]
        for _ in range(warmup):
            launch()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            launch()
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / iters
        tflops = (2.0 * L * K * N) / dt / 1e12
        print(f"  {name:<11} {dt * 1e6:9.2f} us/iter   {tflops:8.2f} TFLOP/s")


def main(argv=None):
    p = argparse.ArgumentParser(description="jagged_dense_bmm backward: validate references + kernels")
    p.add_argument("-b", "--n-groups", type=int, default=64, help="number of groups (batch)")
    p.add_argument("-m", "--max-seq-len", type=int, default=512, help="max rows per group")
    p.add_argument("--regime", choices=["uniform", "skew"], default="uniform")
    p.add_argument("--seed", type=int, default=1234, help="skew RNG seed")
    p.add_argument("--only", default="all", help="comma list of {djagged,ddense,dbias} or 'all'")
    p.add_argument("--bench", action="store_true", help="after validation, time each kernel + report TFLOP/s")
    p.add_argument("--warmup", type=int, default=10, help="bench warmup iterations")
    p.add_argument("--iters", type=int, default=50, help="bench timed iterations")
    args = p.parse_args(argv)

    if not torch.cuda.is_available():
        print("CUDA/ROCm device not available; this example requires a GPU.")
        return 1

    which = GRADS if args.only == "all" else tuple(s.strip() for s in args.only.split(","))
    for w in which:
        if w not in GRADS:
            print(f"unknown gradient '{w}'; choose from {GRADS} or 'all'")
            return 2

    device = "cuda"
    print(f"shape: n_groups={args.n_groups}, max_seq_len={args.max_seq_len}, "
          f"K={K}, N={N}, regime={args.regime}, split={SPLIT}")

    jagged, dense, bias, d_out, seq_offsets, total_rows = make_inputs(
        args.n_groups, args.max_seq_len, args.regime, args.seed, device
    )
    print(f"packed rows L = {total_rows}")

    refs = {
        "djagged": ref_grad_jagged(d_out, dense, seq_offsets, args.n_groups),
        "ddense": ref_grad_dense(jagged, d_out, seq_offsets, args.n_groups),
        "dbias": ref_grad_bias(d_out, seq_offsets, args.n_groups),
    }

    # --- References must match autograd (validates the references themselves) ---
    print("reference vs autograd:")
    ag_j, ag_d, ag_b = autograd_grads(jagged, dense, bias, d_out, seq_offsets, args.n_groups)
    ref_ok = True
    ref_ok &= report("dJagged", ag_j, refs["djagged"], cos_thresh=0.99999)
    ref_ok &= report("dDense", ag_d, refs["ddense"], cos_thresh=0.99999)
    ref_ok &= report("dBias", ag_b, refs["dbias"], cos_thresh=0.99999)
    if not ref_ok:
        print("references do NOT match autograd; aborting before kernel validation.")
        return 1

    # --- FlyDSL kernels vs references (SKIPPED until kernels are implemented) ---
    print("flydsl vs reference:")
    got = run_flydsl_bwd(
        which, jagged, dense, bias, d_out, seq_offsets, args.n_groups, args.max_seq_len
    )
    name_map = {"djagged": "dJagged", "ddense": "dDense", "dbias": "dBias"}
    kernel_ok = True
    any_run = False
    for g in which:
        if got[g] is None:
            continue
        any_run = True
        kernel_ok &= report(name_map[g], refs[g], got[g])
    if not any_run:
        print("  (no kernels implemented yet)")

    # --- Optional timing/TFLOPs summary -------------------------------------------
    if args.bench and any_run:
        bench_which = tuple(
            t for t in BENCH_TARGETS
            if (t == "djagged" and "djagged" in which)
            or (t == "dense_bias" and (("ddense" in which) or ("dbias" in which)))
        )
        bench_flydsl_bwd(
            bench_which, jagged, dense, d_out, seq_offsets, args.n_groups, args.max_seq_len,
            warmup=args.warmup, iters=args.iters,
        )

    return 0 if (ref_ok and kernel_ok) else 1


if __name__ == "__main__":
    sys.exit(main())
