# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""End-to-end test of the jagged_dense_bmm BACKWARD dispatch layer.

Validates ``jagged_dense_bmm_bwd_dispatched`` (aiter.ops.flydsl) for the headline
shapes (B in {120, 1024}, D in {256, 512}) across all three regimes (uniform +
genrec + skew), plus:

  1. per-shape config resolution (winner @ North-Star Mi=7680, D-bucketed
     heuristic elsewhere), and that the resolved gj_stages_a is actually pinned;
  2. cos > 0.999 vs a torch-eager reference for ALL THREE grads (dJagged, dDense,
     dBias), reusing the bench's reference;
  3. a regression guard for the grad_jagged last-group tail over-read that GPU-
     faulted before 2026-07-06 (B=32, D=512, Mi=2048, genrec, seed=1234) -- this
     shape must simply run to completion (a fault aborts the process);
  4. a forced split=2 reduce-path case (D in {512, 384}) that exercises the
     SPLIT>=2 grad_dense_reduce / grad_bias_reduce kernels with NRED_COL_TILES>=2
     -- a branch UNREACHABLE via the D-derived split policy (SPLIT=1 for D>256;
     NRED_COL_TILES=1 for D<=256). D=384 also covers a partial last column-tile
     (384 % 256 != 0), guarding the reduce kernels' col < N store bound.

Phase 4: the backward bakes D and the schedule knobs into a memoized per-shape
build (jagged_dense_bmm_bwd.build_backward), so multiple D coexist in ONE process.
This test therefore runs BOTH D in a single process (D=256 then D=512) — which is
itself the multi-D regression — with no per-D subprocess isolation and no cache
clearing. ``--worker-d D`` still runs a single D in-process for manual debugging.

Run (inside the venv), either as a script or under pytest (GPU required; the
pytest cases skip cleanly when no ROCm/CUDA device or flydsl is available):
    HIP_VISIBLE_DEVICES=6 python op_tests/flydsl_tests/test_jdbba_bwd_dispatch.py
    HIP_VISIBLE_DEVICES=6 pytest op_tests/flydsl_tests/test_jdbba_bwd_dispatch.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytest

# Make `aiter` (and the sibling bench module) importable both as a script and under
# pytest. parents[2] is the aiter repo root (for `import aiter`); the test's own
# directory carries the sibling `bench_jagged_dense_bmm_bwd_perf` module.
_REPO_ROOT = str(Path(__file__).resolve().parents[2])
_TEST_DIR = str(Path(__file__).resolve().parent)
for _p in (_REPO_ROOT, _TEST_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Headline correctness cases per D: (B, Mi, regime). Mi is kept modest (correctness
# is shape-topology-dependent, not Mi-magnitude-dependent) to bound the eager
# reference cost; the North-Star Mi is exercised via resolve_config only. Covers
# B in {120, 1024} and all three regimes (uniform / genrec / skew); uniform is run
# at B=120 only (uniform is topology-trivial, so B=1024 adds no new coverage).
_HEADLINE = [
    (120, 512, "uniform"),
    (120, 512, "genrec"),
    (120, 512, "skew"),
    (1024, 512, "genrec"),
    (1024, 512, "skew"),
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
    from aiter.ops.flydsl.jagged_dense_bmm_bwd_dispatch import resolve_config
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
    # Report the ACTUAL per-shape build config (not module globals, which no longer
    # track it — the build bakes D + knobs as closure constants).
    cfg = resolve_config(n_groups=B, reduction_k=D, output_n=D, max_seq_len=Mi)
    bw = _bwd.build_backward(D, split=cfg["split"], gj_stages_a=cfg["gj_stages_a"], coarsen_m=cfg["coarsen_m"])
    tag = f"{label}B={B} D={D} Mi={Mi} {regime:6s} L={L} gj={cfg['gj_stages_a']} split={bw.split}"
    return ok, f"[{'PASS' if ok else 'FAIL'}] {tag}  cos(dJ={c_dj:.5f}, dD={c_dd:.5f}, dB={c_db:.5f})"


def _run_reduce_path_case(D, B, Mi, regime, *, seed=1234, sparsity=0.95):
    """Force split=2 to exercise the SPLIT>=2 reduce kernels' NRED_COL_TILES>=2
    column-tiling path at N>256.

    The D-derived split policy uses SPLIT=1 for D>256, so grad_dense_reduce /
    grad_bias_reduce never launch there -- and at D<=256 (SPLIT=2) N<=256 gives
    NRED_COL_TILES=1. So the multi-column-tile reduce branch is only reachable via
    a forced split kwarg at N>256. D=512 (N=512) exercises NRED_COL_TILES=2 with a
    full last tile; D=384 (N=384) exercises the PARTIAL last tile (384 % 256 != 0),
    guarding the reduce kernels' col < N store bound.
    """
    import torch

    from aiter.ops.flydsl import jagged_dense_bmm_bwd_dispatched
    from bench_jagged_dense_bmm_bwd_perf import _make_inputs, _torch_reference

    jagged, dense, d_out, seq_offsets, L, N, K = _make_inputs(
        B, D, D, Mi, regime=regime, seed=seed, sparsity=sparsity
    )
    dj, dd, db = jagged_dense_bmm_bwd_dispatched(
        jagged, dense, d_out, seq_offsets, n_groups=B, max_seq_len=Mi, split=2
    )
    torch.cuda.synchronize()
    rj, rd, rb = _torch_reference(jagged, dense, d_out, seq_offsets, N, K)
    c_dj, c_dd, c_db = _cos(dj, rj), _cos(dd, rd), _cos(db, rb)
    ok = min(c_dj, c_dd, c_db) > _COS_THRESH
    ncols = (N + (N if N <= 256 else 256) - 1) // (N if N <= 256 else 256)
    tag = f"[reduce-path] B={B} D={D} Mi={Mi} {regime:6s} split=2 NRED_COL_TILES={ncols} L={L}"
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

    # The memoized build resolves the D-derived SPLIT (2 @ D<=256, 1 @ D>256).
    expected_split = 2 if D <= 256 else 1
    bw = _bwd.build_backward(D, split=None, gj_stages_a=expected_gj, coarsen_m=None)
    split_ok = bw.split == expected_split
    ok &= split_ok
    print(f"[{'PASS' if split_ok else 'FAIL'}] build_backward(D={D}).split={bw.split} (expected {expected_split})")

    # (2) headline correctness (all three grads).
    for (B, Mi, regime) in _HEADLINE:
        case_ok, msg = _run_case(D, B, Mi, regime)
        ok &= case_ok
        print(msg)

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
    """Phase 4: run BOTH D in ONE process. The backward now bakes D + knobs into a
    memoized per-shape build (jagged_dense_bmm_bwd.build_backward), so multiple D
    coexist — no single-D-per-process constraint, no per-D subprocess isolation,
    no cache clearing. Running D=256 then D=512 here IS the multi-D regression."""
    overall = True
    for D in (256, 512):
        print(f"\n===== D={D} (same process, multi-D) =====")
        overall &= (_worker(D) == 0)

    # Forced split=2 reduce-path coverage: the SPLIT>=2 reduce kernels with
    # NRED_COL_TILES>=2 are unreachable via the D-derived policy (SPLIT=1 @ D>256;
    # NRED_COL_TILES=1 @ D<=256). D=512 -> full last tile; D=384 -> partial last
    # tile (guards the col < N store bound). Same process (multi-D build).
    print("\n===== forced split=2 reduce path (NRED_COL_TILES>=2) =====")
    for (D, B, Mi, regime) in [(512, 120, 512, "genrec"), (384, 120, 512, "genrec")]:
        case_ok, msg = _run_reduce_path_case(D, B, Mi, regime)
        overall &= case_ok
        print(msg)

    print("\n==================== OVERALL:", "PASS" if overall else "FAIL", "====================")
    return 0 if overall else 1


# --------------------------------------------------------------------------- #
# pytest entry points (collectable by CI). These require a GPU + flydsl and skip #
# cleanly otherwise; the `__main__` script path below is unchanged for manual   #
# GPU runs. Each per-D worker is its own test so a failure localizes to a D.     #
# --------------------------------------------------------------------------- #

def _backend_ready() -> bool:
    """True iff a ROCm/CUDA device and an importable flydsl are both present."""
    try:
        import torch

        from aiter.ops.flydsl import is_flydsl_available

        return torch.cuda.is_available() and is_flydsl_available()
    except Exception:
        return False


_requires_backend = pytest.mark.skipif(
    not _backend_ready(),
    reason="jdbba backward requires a ROCm/CUDA GPU and an importable flydsl",
)


@_requires_backend
@pytest.mark.parametrize("D", [256, 512])
def test_jdbba_bwd_dispatch_worker(D):
    """Per-D: config resolution + SPLIT, headline correctness (all 3 grads, all
    regimes), the tail-over-read regression (D=512), and autograd (D<=256)."""
    assert _worker(D) == 0


@_requires_backend
@pytest.mark.parametrize("D", [512, 384])
def test_jdbba_bwd_reduce_path(D):
    """Forced split=2 reduce-path coverage (NRED_COL_TILES>=2). D=384 exercises a
    partial last column-tile (384 % 256 != 0), guarding the reduce col < N bound."""
    ok, msg = _run_reduce_path_case(D, 120, 512, "genrec")
    assert ok, msg


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
