# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Does folding local_start into the out base drop leading window tokens?

`KNOWN_ISSUE_out_store_alignment.md` (opus-ops) documents a correctness bug in the FP8 /
16x16 kernels with exactly that shape:

    out_base = ptr_out + row*stride + local_start   ->  byte address 4*(row*stride+local_start)
    is not 16-byte aligned when local_start % 4 != 0, and the gfx950 raw_buffer_store_b32
    OOB check discards the leading sub-16B partial group at the base, dropping the first
    (4 - local_start%4) % 4 in-window tokens. Measured for local_start >= 16; below 16 was
    an incidental exemption that the doc says not to rely on.

The 32x32x64 kernel's OPUS_LOGITS_FP4_WINDOW_VBUF=1 store does fold local_start into the
base, so it is on exactly that footing. The doc also argues the two properties are
mutually exclusive ("a lower-bound token compare is unavoidable"), which if true here
makes that build silently wrong on non-4-aligned windows.

So reproduce the doc's own experiment rather than reasoning about it: one row, fixed
window width, sweep local_start over every residue and past the <16 exemption, and check
BOTH directions --

  dropped : in-window cells left at the -inf pre-fill  (the documented bug)
  leaked  : out-of-window cells written                (the failure mode the doc's
            rejected "round the base down" fix would have introduced)

    python3 op_tests/test_fp4_store_alignment.py [--width 40] [--max-start 68]
"""

import argparse
import importlib.util
from pathlib import Path

import torch

_SPEC = importlib.util.spec_from_file_location(
    "_opus_optest_align", Path(__file__).with_name("test_pa_mqa_logits_opus.py")
)
T = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(T)

from aiter.ops.opus.pa_mqa_logits_opus import (  # noqa: E402
    pa_mqa_logits_mxfp4_prefill,
)

dev = "cuda"


def one_case(start, end, block_k, seed):
    """Returns (n_dropped, n_leaked, max_rel_err) for a single-row window [start, end)."""
    inp = T.build_inputs(1, end, 1, block_k, seed=seed)
    rb = torch.zeros(1, dtype=torch.int32, device=dev)
    ls = torch.tensor([start], dtype=torch.int32, device=dev)
    le = torch.tensor([end], dtype=torch.int32, device=dev)

    out = pa_mqa_logits_mxfp4_prefill(
        inp.q_packed, inp.q_scale, inp.kv_cache, inp.kv_scale, inp.block_tables,
        inp.weights, rb, ls, le, inp.max_seq_len,
        weight_scale=T.WEIGHT_SCALE, block_k=block_k, kv_block_size=T.KV_BLOCK_SIZE,
    )  # fmt: skip
    torch.cuda.synchronize()

    row = out[0]
    col = torch.arange(inp.max_seq_len, device=dev)
    inside = (col >= start) & (col < end)

    dropped = int(torch.isneginf(row[inside]).sum().item())
    leaked = int((~torch.isneginf(row[~inside])).sum().item())

    ref = T.ref_rows(inp, [0], rb, ls, le)[0][2]
    got = row[start:end].float()
    finite = torch.isfinite(got)
    err = (
        ((got[finite] - ref[finite]).abs().max() / ref.abs().max().clamp(min=1e-6)).item()
        if finite.any()
        else float("nan")
    )
    return dropped, leaked, err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=40, help="window width, as in the doc")
    ap.add_argument("--max-start", type=int, default=68)
    ap.add_argument("--block-k", type=int, nargs="*", default=[64, 256])
    a = ap.parse_args()

    print(f"window width {a.width}, local_start 0..{a.max_start - 1}, "
          f"single row; expecting dropped=0 and leaked=0 everywhere")  # fmt: skip
    bad = 0
    for block_k in a.block_k:
        rows = []
        for start in range(a.max_start):
            d, lk, err = one_case(start, start + a.width, block_k, seed=start + 1)
            rows.append((start, d, lk, err))
            if d or lk or not (err < 2e-5):
                bad += 1
        print(f"\n--- block_k={block_k} ---")
        print(f"{'start':>5} {'%4':>3} {'dropped':>8} {'leaked':>7} {'err':>10}")
        for start, d, lk, err in rows:
            flag = "" if (d == 0 and lk == 0 and err < 2e-5) else "   <== FAIL"
            # Print every residue-interesting start plus anything that failed.
            if start % 4 != 0 or start < 20 or flag or start % 16 == 0:
                print(f"{start:5d} {start % 4:3d} {d:8d} {lk:7d} {err:10.2e}{flag}")
        worst = max(r[3] for r in rows)
        print(f"  {sum(1 for r in rows if r[1] == 0 and r[2] == 0 and r[3] < 2e-5)}"
              f"/{len(rows)} starts clean, worst err {worst:.2e}")  # fmt: skip

    print(f"\n{'PASS' if bad == 0 else f'FAIL ({bad} bad cases)'}")
    raise SystemExit(0 if bad == 0 else 1)


if __name__ == "__main__":
    main()
