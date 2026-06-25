#!/usr/bin/env python3
"""Poison-tail correctness checker for the FlyDSL K=512 decode TopK kernel.

Runs the FlyDSL unordered (set-output) path for one or more shapes and compares
the returned index set against ``torch.topk`` over the *masked* row region. The
padded tail ``[row_len:max_width)`` is filled with a large poison value
(``+1e30`` by default) so any kernel that reads past ``row_len`` is caught: the
poison columns would dominate the selection and break set-equivalence.

The FlyDSL variant is selected entirely through the ``FLYDSL_TOPK_*``
environment so this checker exercises whatever tier/scan/barrier configuration
the caller sets (same contract as ``topk_decode_rocprof_runner.py``).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _add_path(path) -> None:
    if not path:
        return
    resolved = Path(path).expanduser().resolve()
    if resolved.exists() and str(resolved) not in sys.path:
        sys.path.insert(0, str(resolved))


def load_flydsl():
    _add_path(REPO_ROOT.parent / "FlyDSL" / "python")
    configured = os.environ.get("FLYDSL_PATH")
    if configured:
        _add_path(Path(configured) / "python")
        _add_path(configured)
    _add_path(REPO_ROOT.parent / ".r1_flydsl_pkgs")
    from aiter.ops.flydsl.topk_per_row_decode import flydsl_top_k_per_row_decode

    return flydsl_top_k_per_row_decode


def make_logits(num_rows, max_width, row_len, distribution, seed, device, poison):
    torch.manual_seed(seed)
    shape = (num_rows, max_width)
    if distribution == "random":
        logits = torch.randn(shape, dtype=torch.float32, device=device)
    elif distribution == "10LSBits":
        fixed_top = 0x3F900000
        top_mask = 0xFFFFFC00
        low_mask = 0x000003FF
        low = torch.randint(0, 2**10, shape, dtype=torch.int32, device=device)
        bits = (fixed_top & top_mask) | (low & low_mask)
        logits = bits.view(torch.float32)
    elif distribution == "ties":
        logits = torch.randint(-16, 16, shape, dtype=torch.int32, device=device).to(
            torch.float32
        )
    else:
        raise ValueError(f"unknown distribution: {distribution}")
    if row_len < max_width:
        logits[:, row_len:] = poison
    return logits


def check_one(op, k, L, num_rows, max_width, distribution, next_n, seed, poison):
    device = torch.device("cuda")
    seq_lens = torch.full((num_rows // next_n,), L, dtype=torch.int32, device=device)
    logits = make_logits(num_rows, max_width, L, distribution, seed, device, poison)
    indices = torch.empty((num_rows, k), dtype=torch.int32, device=device)
    op(
        logits,
        next_n,
        seq_lens,
        indices,
        num_rows,
        logits.stride(0),
        logits.stride(1),
        k=k,
        ordered=False,
    )
    torch.cuda.synchronize()

    ref_k = min(k, L)
    expected = torch.topk(logits[:, :L], ref_k, dim=-1).indices.to(torch.int32)
    actual = indices.detach().cpu()
    expected = expected.cpu()
    logits_cpu = logits.detach().cpu()
    for row in range(num_rows):
        valid = min(k, L)
        a = actual[row, :valid]
        e = expected[row, :valid]
        bad = (a < 0) | (a >= L)
        if bad.any():
            return False, f"row {row}: out-of-range/poison indices {a[bad][:4].tolist()}"
        a_set, e_set = set(a.tolist()), set(e.tolist())
        if len(a_set) != len(a.tolist()):
            return False, f"row {row}: duplicate indices"
        if a_set == e_set:
            continue
        a_only = sorted(a_set - e_set)
        e_only = sorted(e_set - a_set)
        if len(a_only) != len(e_only):
            return False, f"row {row}: set size mismatch ({len(a_only)} vs {len(e_only)})"
        av = torch.tensor([logits_cpu[row, i].item() for i in a_only])
        ev = torch.tensor([logits_cpu[row, i].item() for i in e_only])
        if not torch.allclose(av.sort().values, ev.sort().values, rtol=0, atol=0):
            return False, f"row {row}: value mismatch on tie boundary"
    return True, "ok"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=512)
    ap.add_argument("--L", type=int, nargs="+", required=True)
    ap.add_argument("--num-rows", type=int, nargs="+", default=[1])
    ap.add_argument("--max-width", type=int, default=256000)
    ap.add_argument(
        "--distribution", nargs="+", default=["random"],
        choices=["random", "10LSBits", "ties"],
    )
    ap.add_argument("--next-n", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--poison", type=float, default=1e30)
    ap.add_argument(
        "--unpadded", action="store_true",
        help="Use max_width == L (no padded tail) instead of --max-width.",
    )
    args = ap.parse_args()

    op = load_flydsl()
    fails = 0
    for dist in args.distribution:
        for rows in args.num_rows:
            for L in args.L:
                mw = L if args.unpadded else max(args.max_width, L)
                ok, msg = check_one(
                    op, args.k, L, rows, mw, dist, args.next_n, args.seed, args.poison
                )
                status = "PASS" if ok else "FAIL"
                if not ok:
                    fails += 1
                print(
                    f"[{status}] dist={dist:8s} rows={rows} L={L:7d} mw={mw} : {msg}",
                    flush=True,
                )
    if fails:
        print(f"\n{fails} FAILED")
        sys.exit(1)
    print("\nall passed")


if __name__ == "__main__":
    main()
