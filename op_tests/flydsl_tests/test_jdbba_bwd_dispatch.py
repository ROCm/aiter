# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""End-to-end test of the jagged_dense_bmm BACKWARD dispatch layer.

Validates ``jagged_dense_bmm_bwd_dispatched`` (aiter.ops.flydsl) for the headline
shapes (B in {120, 1024}, D in {256, 512}) across regimes (genrec + skew), plus:

  1. per-shape config resolution (winner @ North-Star Mi=7680, D-bucketed
     heuristic elsewhere), and that the resolved gj_stages_a is actually pinned;
  2. cos > 0.999 vs a torch-eager reference for ALL THREE grads (dJagged, dDense,
     dBias), reusing the bench's reference;
  3. a regression guard for the grad_jagged last-group tail over-read that GPU-
     faulted before 2026-07-06 (B=32, D=512, Mi=2048, genrec, seed=1234) -- this
     shape must simply run to completion (a fault aborts the process).

Because D (= K = N) and the grad_jagged schedule knobs are compile-time constants
snapshotted on the first launch -- AND the FlyDSL cache key ignores COARSEN_M /
GJ_STAGES_A -- this test runs ONE D PER SUBPROCESS with ~/.flydsl/cache cleared
between them. Invoked with no args it orchestrates the per-D worker subprocesses.

Run (inside the venv):
    HIP_VISIBLE_DEVICES=6 python op_tests/flydsl_tests/test_jdbba_bwd_dispatch.py
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

# Make `aiter` (and the sibling bench module) importable when run as a script.
_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Headline correctness cases per D: (B, Mi, regime). Mi is kept modest (correctness
# is shape-topology-dependent, not Mi-magnitude-dependent) to bound the eager
# reference cost; the North-Star Mi is exercised via resolve_config only.
_HEADLINE = [
    (120, 512, "genrec"),
    (120, 512, "skew"),
    (1024, 512, "genrec"),
]
# grad_jagged last-group tail over-read regression (must keep exact shape/seed).
_REGRESSION = dict(B=32, Mi=2048, regime="genrec", seed=1234, sparsity=0.95)
_COS_THRESH = 0.999


def _cos(a, b):
    import torch

    return torch.nn.functional.cosine_similarity(
        a.float().flatten(), b.float().flatten(), dim=0
    ).item()


def _eager_grads(jagged, dense, bias, seq_offsets, grad_out, B):
    """Analytic (torch-autograd) grads of the eager fp32 forward, ground truth for
    the FlyDSL autograd op. loss = <Out, grad_out> so d(loss)/dOut == grad_out."""
    import torch

    j = jagged.float().detach().clone().requires_grad_(True)
    d = dense.float().detach().clone().requires_grad_(True)
    bi = bias.float().detach().clone().requires_grad_(True)
    go = grad_out.float()
    loss = j.new_zeros(())
    for b in range(B):
        s, e = int(seq_offsets[b]), int(seq_offsets[b + 1])
        if e > s:
            out_b = j[s:e] @ d[b] + bi[b][None, :]
            loss = loss + (out_b * go[s:e]).sum()
    loss.backward()
    return j.grad, d.grad, bi.grad


def _run_autograd_case(D, B, Mi, regime, *, seed=1234, sparsity=0.95):
    """Run the fwd+bwd autograd op through out.backward(); check grads vs eager."""
    import torch

    from aiter.ops.flydsl import jagged_dense_bmm_autograd
    from bench_jagged_dense_bmm_bwd_perf import _make_inputs

    jagged, dense, grad_out, seq_offsets, L, N, K = _make_inputs(
        B, D, D, Mi, regime=regime, seed=seed, sparsity=sparsity
    )
    bias = torch.randn(B, N, dtype=torch.bfloat16, device=jagged.device)

    jf = jagged.detach().clone().requires_grad_(True)
    df = dense.detach().clone().requires_grad_(True)
    bf = bias.detach().clone().requires_grad_(True)
    out = jagged_dense_bmm_autograd(jf, df, bf, seq_offsets, n_groups=B, max_seq_len=Mi)
    out.backward(grad_out)
    torch.cuda.synchronize()

    gj_e, gd_e, gb_e = _eager_grads(jagged, dense, bias, seq_offsets, grad_out, B)
    c_j, c_d, c_b = _cos(jf.grad, gj_e), _cos(df.grad, gd_e), _cos(bf.grad, gb_e)
    ok = min(c_j, c_d, c_b) > _COS_THRESH
    tag = f"[autograd] B={B} D={D} Mi={Mi} {regime:6s} L={L}"
    return ok, f"[{'PASS' if ok else 'FAIL'}] {tag}  grad cos(dJ={c_j:.5f}, dD={c_d:.5f}, dB={c_b:.5f})"


def _run_case(D, B, Mi, regime, *, seed=1234, sparsity=0.95, label=""):
    """Run the dispatched backward for one shape; return (ok, msg)."""
    import torch

    from aiter.ops.flydsl import jagged_dense_bmm_bwd_dispatched
    from aiter.ops.flydsl.kernels import jagged_dense_bmm_bwd as _bwd

    # Reuse the bench's input builder + eager reference (single source of truth).
    from bench_jagged_dense_bmm_bwd_perf import _make_inputs, _torch_reference

    jagged, dense, d_out, seq_offsets, L, N, K = _make_inputs(
        B, D, D, Mi, regime=regime, seed=seed, sparsity=sparsity
    )
    dj, dd, db = jagged_dense_bmm_bwd_dispatched(
        jagged, dense, d_out, seq_offsets, n_groups=B, max_seq_len=Mi
    )
    torch.cuda.synchronize()
    rj, rd, rb = _torch_reference(jagged, dense, d_out, seq_offsets, N, K)
    c_dj, c_dd, c_db = _cos(dj, rj), _cos(dd, rd), _cos(db, rb)
    ok = min(c_dj, c_dd, c_db) > _COS_THRESH
    tag = f"{label}B={B} D={D} Mi={Mi} {regime:6s} L={L} gj={_bwd.GJ_STAGES_A} split={_bwd.SPLIT}"
    return ok, f"[{'PASS' if ok else 'FAIL'}] {tag}  cos(dJ={c_dj:.5f}, dD={c_dd:.5f}, dB={c_db:.5f})"


def _worker(D: int) -> int:
    """Run every case for a single D in this (fresh) process. Returns exit code."""
    from aiter.ops.flydsl.jagged_dense_bmm_bwd_dispatch import resolve_config
    from aiter.ops.flydsl.kernels import jagged_dense_bmm_bwd as _bwd

    ok = True

    # (1) resolution: North-Star Mi=7680 hits a JSON winner; expected per-D gj.
    expected_gj = 1 if D <= 256 else 2
    cfg = resolve_config(n_groups=1024, reduction_k=D, output_n=D, max_seq_len=7680)
    res_ok = cfg["gj_stages_a"] == expected_gj
    ok &= res_ok
    print(f"[{'PASS' if res_ok else 'FAIL'}] resolve winner D={D}: gj_stages_a={cfg['gj_stages_a']} "
          f"(expected {expected_gj})")

    # (2) headline correctness (all three grads).
    for (B, Mi, regime) in _HEADLINE:
        case_ok, msg = _run_case(D, B, Mi, regime)
        ok &= case_ok
        print(msg)

    # After the first launch the resolved gj must be pinned into the kernel module.
    pin_ok = _bwd.GJ_STAGES_A == expected_gj
    ok &= pin_ok
    print(f"[{'PASS' if pin_ok else 'FAIL'}] pinned GJ_STAGES_A={_bwd.GJ_STAGES_A} (expected {expected_gj})")

    # (3) regression guard (D=512 only): must run to completion without faulting.
    if D == 512:
        r = _REGRESSION
        case_ok, msg = _run_case(D, r["B"], r["Mi"], r["regime"], seed=r["seed"],
                                 sparsity=r["sparsity"], label="[regression] ")
        ok &= case_ok
        print(msg)

    # (4) end-to-end autograd (out.backward()) vs eager grads. D=256 only: D=512 is
    # blocked at large L by the FORWARD int32-offset overflow (integration plan Risks).
    if D <= 256:
        for (B, Mi, regime) in [(120, 512, "genrec"), (120, 512, "skew")]:
            case_ok, msg = _run_autograd_case(D, B, Mi, regime)
            ok &= case_ok
            print(msg)
    else:
        print(f"[SKIP] autograd D={D}: forward int32-overflow at large L (separate forward fix)")

    print(f"\nD={D} RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def _orchestrate() -> int:
    """Spawn one worker subprocess per D, clearing the FlyDSL cache between them."""
    cache = Path.home() / ".flydsl" / "cache"
    overall = True
    for D in (256, 512):
        if cache.exists():
            shutil.rmtree(cache, ignore_errors=True)
        print(f"\n===== D={D} (fresh process, cache cleared) =====")
        rc = subprocess.run(
            [sys.executable, os.path.abspath(__file__), "--worker-d", str(D)],
            env=os.environ.copy(),
        ).returncode
        overall &= (rc == 0)
    print("\n==================== OVERALL:", "PASS" if overall else "FAIL", "====================")
    return 0 if overall else 1


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="jdbba backward dispatch test")
    p.add_argument("--worker-d", type=int, default=None,
                   help="internal: run all cases for this single D in-process")
    args = p.parse_args(argv)
    if args.worker_d is not None:
        return _worker(args.worker_d)
    return _orchestrate()


if __name__ == "__main__":
    sys.exit(main())
