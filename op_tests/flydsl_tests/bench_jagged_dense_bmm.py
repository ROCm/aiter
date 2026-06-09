# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
#
# Benchmark: FlyDSL jagged_dense_bmm_broadcast_add (jdbba) prototype vs the
# upstream Meta/HSTU Triton kernel.
#
#   Out[s:e, :] = Jagged[s:e, :] @ Dense[b] + Bias[b][None, :]   per group b
#
# Run inside the devcontainer (torch/triton/flydsl live in the container venv):
#   docker exec -w /home/anguyenh/aiter anguyenh-dev \
#     python op_tests/flydsl_tests/bench_jagged_dense_bmm.py
#
# IMPORTANT -- two different timings, do not confuse them:
#   * Default (CUDA-event wall-clock) measures END-TO-END latency, which at these
#     shapes is ~90% fixed host launch/dispatch overhead (~70 us) and only ~10%
#     actual GPU work. It is the right number for "how long does one call take in
#     a Python loop" but it MASKS kernel-level changes -- e.g. the C-store
#     scalarization and a BLOCK_M sweep both left this number unchanged while the
#     device time moved by 30%. Treat cross-kernel comparisons here as unreliable.
#   * --device-time re-runs each shape under rocprofv3 and reports the true
#     per-kernel GPU duration (the number to optimize against). Requires rocprofv3
#     on PATH. There is no pure-Python substitute: CUDA-graph capture of the
#     FlyDSL launch path produces an empty graph (replay yields zeros), and
#     batched event timing is still dispatch-starved.
#
# The Triton reference comes from the upstream generative-recommenders repo.

from __future__ import annotations

import argparse
import os
import statistics
import subprocess
import sys
from typing import List

import torch

import flydsl.compiler as flyc
from aiter.ops.flydsl.kernels import jagged_dense_bmm as jdbba

# ---- Upstream Triton reference --------------------------------------------------
_HAS_TRITON = True
try:
    from generative_recommenders.ops.triton.triton_jagged import (
        triton_jagged_dense_bmm_add_fwd,
    )
except Exception as exc:  # pragma: no cover - environment dependent
    _HAS_TRITON = False
    _TRITON_ERR = exc

BLOCK_M = jdbba.BLOCK_M
N = jdbba.N
K = jdbba.K


def torch_reference(jagged, dense, bias, seq_offsets):
    # dense: (B, K, N) un-transposed; bias: (B, N)
    B_groups = dense.shape[0]
    L = jagged.shape[0]
    out = torch.zeros((L, dense.shape[2]), dtype=torch.bfloat16, device=jagged.device)
    for b in range(B_groups):
        s = int(seq_offsets[b].item())
        e = int(seq_offsets[b + 1].item())
        if e > s:
            acc = jagged[s:e].float() @ dense[b].float() + bias[b].float()[None, :]
            out[s:e] = acc.to(torch.bfloat16)
    return out


def _cos_mae(expected, actual):
    ef, af = expected.float(), actual.float()
    cos = torch.nn.functional.cosine_similarity(ef.flatten(), af.flatten(), dim=0).item()
    mae = (ef - af).abs().mean().item()
    return cos, mae


def _time_ms(fn, warmup=10, iters=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.median(samples)


def _make_inputs(row_counts: List[int]):
    B_groups = len(row_counts)
    seq_offsets = torch.zeros(B_groups + 1, dtype=torch.int32)
    for i, m in enumerate(row_counts):
        seq_offsets[i + 1] = seq_offsets[i] + m
    L = int(seq_offsets[-1].item())
    max_seq_len = max(row_counts) if row_counts else 0

    jagged_pad = torch.randn(L + BLOCK_M, K, dtype=torch.bfloat16).cuda()
    jagged = jagged_pad[:L].contiguous() if L > 0 else jagged_pad[:0].contiguous()
    dense = torch.randn(B_groups, K, N, dtype=torch.bfloat16).cuda()
    bias = torch.randn(B_groups, N, dtype=torch.bfloat16).cuda()
    seq_offsets_d = seq_offsets.cuda()
    return jagged, dense, bias, seq_offsets, seq_offsets_d, L, max_seq_len, B_groups


def _run_flydsl(jagged, dense, bias, seq_offsets_d, L, max_seq_len, B_groups):
    # FlyDSL wants Dense as a tall (B_groups*N, K) matrix (host pre-transpose),
    # bias flattened to (B_groups*N,), and a padded output to absorb tail tiles.
    dense_tall = dense.transpose(1, 2).reshape(B_groups * N, K).contiguous()  # (B*N, K)
    bias_flat = bias.reshape(B_groups * N).contiguous()
    out = torch.zeros(L + BLOCK_M, N, dtype=torch.bfloat16).cuda()

    tA = flyc.from_dlpack(jagged).mark_layout_dynamic(leading_dim=1, divisibility=8)
    tC = flyc.from_dlpack(out).mark_layout_dynamic(leading_dim=1, divisibility=8)

    def launch():
        jdbba.jagged_dense_bmm(
            tC, tA, dense_tall, bias_flat, seq_offsets_d, B_groups, max_seq_len,
            stream=torch.cuda.current_stream(),
        )

    launch()
    torch.cuda.synchronize()
    return out[:L], launch


def _run_triton(jagged, dense, bias, seq_offsets_d, max_seq_len):
    # Triton wrapper wants seq_offsets as int64 and dense as (B, K, N).
    so64 = seq_offsets_d.to(torch.int64)

    def launch():
        return triton_jagged_dense_bmm_add_fwd(max_seq_len, so64, jagged, dense, bias)

    out, _, _, _ = launch()
    torch.cuda.synchronize()
    return out, launch


def run_case(name, row_counts):
    jagged, dense, bias, seq_offsets, seq_offsets_d, L, max_seq_len, B_groups = _make_inputs(
        row_counts
    )
    expected = torch_reference(jagged, dense, bias, seq_offsets)

    # FlyDSL
    fly_out, fly_launch = _run_flydsl(
        jagged, dense, bias, seq_offsets_d, L, max_seq_len, B_groups
    )
    fly_cos, fly_mae = _cos_mae(expected, fly_out)
    fly_ms = _time_ms(fly_launch)

    # Triton (if available)
    if _HAS_TRITON and L > 0:
        tri_out, tri_launch = _run_triton(jagged, dense, bias, seq_offsets_d, max_seq_len)
        tri_cos, tri_mae = _cos_mae(expected, tri_out)
        tri_ms = _time_ms(tri_launch)
        speedup = tri_ms / fly_ms if fly_ms > 0 else float("nan")
        tri_str = f"{tri_ms:8.4f} ms (cos={tri_cos:.4f})"
        sp_str = f"{speedup:5.2f}x"
    else:
        tri_str = "    n/a"
        sp_str = "  n/a"

    print(
        f"[{name:14s}] L={L:5d} groups={str(row_counts):24s} "
        f"FlyDSL {fly_ms:8.4f} ms (cos={fly_cos:.4f})  "
        f"Triton {tri_str}  speedup(tri/fly)={sp_str}"
    )


# --- Test cases (label -> per-group row counts) ---------------------------------
CASES = {
    "exact-multiple": [128, 256, 128],
    "partial-bottom": [200, 100],
    "empty-group": [128, 0, 128],
    "skewed": [512, 16, 32, 128],
    # Larger / more realistic batched shapes for timing.
    "uniform-32x256": [256] * 32,
    "uniform-64x512": [512] * 64,
    "ragged-128grp": [((i * 37) % 480) + 32 for i in range(128)],
}


# --- Device-time path (rocprofv3) -----------------------------------------------
# The default wall-clock is dominated by host launch overhead. To get true GPU
# time we re-invoke this script in a hidden per-shape worker mode under rocprofv3,
# then read the per-kernel durations out of its sqlite db.


def _profile_worker(row_counts, which):
    """Hidden mode: build inputs and fire N launches of one impl, nothing else,
    so rocprofv3 records clean per-kernel device durations."""
    jagged, dense, bias, _, seq_offsets_d, L, max_seq_len, B_groups = _make_inputs(row_counts)
    if L == 0:
        return
    if which == "flydsl":
        _, launch = _run_flydsl(jagged, dense, bias, seq_offsets_d, L, max_seq_len, B_groups)
    else:
        if not _HAS_TRITON:
            return
        _, launch = _run_triton(jagged, dense, bias, seq_offsets_d, max_seq_len)
    for _ in range(10):  # warmup
        launch()
    torch.cuda.synchronize()
    for _ in range(50):  # measured launches
        launch()
    torch.cuda.synchronize()


def _read_kernel_us(db_path, name_substr):
    """Median device duration (us) of kernels whose name contains name_substr."""
    import sqlite3

    db = sqlite3.connect(db_path)
    tabs = [r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")]
    disp = next(t for t in tabs if "kernel_dispatch" in t)
    sym = next(t for t in tabs if "kernel_symbol" in t)
    rows = db.execute(
        f"SELECT s.kernel_name, d.end-d.start FROM {disp} d JOIN {sym} s ON d.kernel_id=s.id"
    ).fetchall()
    durs = sorted(dur for nm, dur in rows if dur and dur > 0 and name_substr in (nm or ""))
    return statistics.median(durs) / 1000.0 if durs else float("nan")


def _device_time_us(label, row_counts, which, name_substr):
    """Re-invoke this script under rocprofv3 for one shape/impl, parse the db."""
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        cmd = [
            "rocprofv3", "--kernel-trace", "-d", td, "-o", "trace", "--",
            sys.executable, os.path.abspath(__file__),
            "--worker", which, "--case", label,
        ]
        env = dict(os.environ, FLYDSL_RUNTIME_ENABLE_CACHE="1")
        subprocess.run(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        dbs = []
        for root, _, files in os.walk(td):
            dbs += [os.path.join(root, f) for f in files if f.endswith(".db")]
        if not dbs:
            return float("nan")
        return _read_kernel_us(dbs[0], name_substr)


def run_device_time():
    if subprocess.run(["which", "rocprofv3"], stdout=subprocess.DEVNULL).returncode != 0:
        print("ERROR: --device-time requires rocprofv3 on PATH")
        return
    print(f"FlyDSL jdbba vs Triton  DEVICE TIME (rocprofv3)  (N={N}, K={K}, bf16)\n")
    for label, rc in CASES.items():
        fly = _device_time_us(label, rc, "flydsl", "jdbba")
        tri = (
            _device_time_us(label, rc, "triton", "jagged_dense_bmm_broadcast_add")
            if _HAS_TRITON
            else float("nan")
        )
        sp = tri / fly if fly == fly and fly > 0 else float("nan")
        print(
            f"[{label:14s}] L={sum(rc):5d}  "
            f"FlyDSL {fly:7.3f} us  Triton {tri:7.3f} us  speedup(tri/fly)={sp:5.2f}x"
        )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--device-time", action="store_true",
                    help="report true per-kernel GPU time via rocprofv3 (recommended)")
    ap.add_argument("--worker", choices=["flydsl", "triton"], help=argparse.SUPPRESS)
    ap.add_argument("--case", help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.worker:  # hidden rocprof worker mode
        _profile_worker(CASES[args.case], args.worker)
        sys.exit(0)

    if args.device_time:
        run_device_time()
        sys.exit(0)

    if not _HAS_TRITON:
        print(f"WARNING: upstream Triton kernel unavailable: {_TRITON_ERR}")
    print(f"FlyDSL jdbba vs Triton  END-TO-END wall-clock  (N={N}, K={K}, bf16)")
    print("NOTE: ~90% of this is host launch overhead; use --device-time for GPU time.\n")
    for _label, _rc in CASES.items():
        run_case(_label, _rc)
