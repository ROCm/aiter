r"""Poison-tail correctness checker for the FlyDSL K=512 decode TopK kernel.

Runs the FlyDSL unordered (set-output) path for one or more shapes and compares
the returned index set against ``torch.topk`` over the *masked* row region. The
padded tail ``[row_len:max_width)`` is filled with a large poison value
(``+1e30`` by default) so any kernel that reads past ``row_len`` is caught: the
poison columns would dominate the selection and break set-equivalence.

The FlyDSL variant is selected entirely through the ``FLYDSL_TOPK_*``
environment so this checker exercises whatever tier/scan/barrier configuration
the caller sets (same contract as ``topk_decode_rocprof_runner.py``).

Every flag accepts multiple values (``nargs="+"``) and the checker runs the full
Cartesian product of ``distribution x k x num-rows x L``. Exit code is non-zero
if any shape fails. Run ``-h`` for the complete flag list.

Usage examples::

    # Regime A (uniform row length == max_width): k/rows/L/dist sweep
    python op_tests/topk_decode_correctness.py \
        --k 512 2048 --num-rows 1 4 16 --L 8192 65536 131072 \
        --distribution random ties 10LSBits

    # Regime B (per-row random causal length in [0.3*L, L], poison tail)
    python op_tests/topk_decode_correctness.py \
        --k 2048 --num-rows 64 128 256 --L 65536 \
        --distribution random ties --seq-rand-min-frac 0.3

    # Unpadded (max_width == L, no poison tail) single shape
    python op_tests/topk_decode_correctness.py --k 512 --L 120000 --unpadded
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


def _has_built_flydsl(path) -> bool:
    """True only if `path` holds a *built* flydsl (with the generated _mlir ext)."""
    if not path:
        return False
    p = Path(path).expanduser().resolve()
    if not (p / "flydsl").exists():
        return False
    return (p / "flydsl" / "_mlir").exists() or bool(
        list((p / "flydsl").glob("_mlir*"))
    )


def _add_path(path) -> None:
    if not path:
        return
    resolved = Path(path).expanduser().resolve()
    if resolved.exists() and str(resolved) not in sys.path:
        sys.path.insert(0, str(resolved))


def load_flydsl():
    # Only prepend a candidate FlyDSL path if it actually contains the compiled
    # MLIR extension; a source-only checkout must not shadow the installed runtime.
    configured = os.environ.get("FLYDSL_PATH")
    for cand in (
        REPO_ROOT.parent / "FlyDSL" / "python",
        (Path(configured) / "python") if configured else None,
        Path(configured) if configured else None,
        REPO_ROOT.parent / ".r1_flydsl_pkgs",
    ):
        if _has_built_flydsl(cand):
            _add_path(cand)
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


def check_one(
    op,
    k,
    L,
    num_rows,
    max_width,
    distribution,
    next_n,
    seed,
    poison,
    seq_rand_min_frac=0.0,
):
    device = torch.device("cuda")
    batch = num_rows // next_n
    if seq_rand_min_frac and seq_rand_min_frac > 0.0:
        # Regime B: random per-row causal length in [frac*L, L].
        gen = torch.Generator(device=device).manual_seed(seed)
        lo = max(1, int(seq_rand_min_frac * L))
        seq_lens = torch.randint(
            lo, L + 1, (batch,), generator=gen, device=device, dtype=torch.int32
        )
    else:
        seq_lens = torch.full((batch,), L, dtype=torch.int32, device=device)

    # Per-row valid length (decode causal geometry: row r -> batch r//next_n at
    # slot r%next_n -> valid length seq_len - next_n + slot + 1).
    row_ids = torch.arange(num_rows, device=device)
    batch_ids = row_ids // next_n
    slots = row_ids % next_n
    row_lens = (seq_lens[batch_ids].to(torch.int64) - next_n + slots + 1).clamp_min(0)
    row_lens_cpu = row_lens.cpu().tolist()

    # Base logits then poison each row's own tail [row_len:max_width).
    base = make_logits(
        num_rows, max_width, max_width, distribution, seed, device, poison
    )
    for r in range(num_rows):
        rl = row_lens_cpu[r]
        if rl < max_width:
            base[r, rl:] = poison
    logits = base

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

    actual = indices.detach().cpu()
    logits_cpu = logits.detach().cpu()
    for row in range(num_rows):
        rl = row_lens_cpu[row]
        valid = min(k, rl)
        if valid <= 0:
            continue
        ref = torch.topk(logits_cpu[row, :rl], valid).indices.to(torch.int32)
        a = actual[row, :valid]
        e = ref
        bad = (a < 0) | (a >= rl)
        if bad.any():
            return (
                False,
                f"row {row}(len={rl}): out-of-range/poison indices {a[bad][:4].tolist()}",
            )
        a_set, e_set = set(a.tolist()), set(e.tolist())
        if len(a_set) != len(a.tolist()):
            return False, f"row {row}: duplicate indices"
        if a_set == e_set:
            continue
        a_only = sorted(a_set - e_set)
        e_only = sorted(e_set - a_set)
        if len(a_only) != len(e_only):
            return (
                False,
                f"row {row}: set size mismatch ({len(a_only)} vs {len(e_only)})",
            )
        av = torch.tensor([logits_cpu[row, i].item() for i in a_only])
        ev = torch.tensor([logits_cpu[row, i].item() for i in e_only])
        if not torch.allclose(av.sort().values, ev.sort().values, rtol=0, atol=0):
            return False, f"row {row}: value mismatch on tie boundary"
    return True, "ok"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, nargs="+", default=[512])
    ap.add_argument("--L", type=int, nargs="+", required=True)
    ap.add_argument("--num-rows", type=int, nargs="+", default=[1])
    ap.add_argument("--max-width", type=int, default=256000)
    ap.add_argument(
        "--distribution",
        nargs="+",
        default=["random"],
        choices=["random", "10LSBits", "ties"],
    )
    ap.add_argument("--next-n", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--poison", type=float, default=1e30)
    ap.add_argument(
        "--unpadded",
        action="store_true",
        help="Use max_width == L (no padded tail) instead of --max-width.",
    )
    ap.add_argument(
        "--seq-rand-min-frac",
        type=float,
        default=0.0,
        help="Regime B: per-row seq_len random in [frac*L, L] (0 = uniform L = regime A).",
    )
    args = ap.parse_args()

    op = load_flydsl()
    fails = 0
    total = 0
    for dist in args.distribution:
        for k in args.k:
            for rows in args.num_rows:
                for L in args.L:
                    mw = L if args.unpadded else max(args.max_width, L)
                    ok, msg = check_one(
                        op,
                        k,
                        L,
                        rows,
                        mw,
                        dist,
                        args.next_n,
                        args.seed,
                        args.poison,
                        args.seq_rand_min_frac,
                    )
                    total += 1
                    status = "PASS" if ok else "FAIL"
                    if not ok:
                        fails += 1
                    print(
                        f"[{status}] dist={dist:8s} k={k:5d} rows={rows:3d} "
                        f"L={L:7d} mw={mw} : {msg}",
                        flush=True,
                    )
    regime = "B(rand seq)" if args.seq_rand_min_frac else "A(seq==max)"
    if fails:
        print(f"\n{fails}/{total} FAILED  [regime {regime}]")
        sys.exit(1)
    print(f"\nall {total} passed  [regime {regime}]")


if __name__ == "__main__":
    main()
