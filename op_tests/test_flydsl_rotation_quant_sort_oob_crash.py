# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Deterministic out-of-bounds crash repro for the fused rotation + per-1x32
MXFP4 quant + MoE scale-sort kernel
(``per_1x32_fp4_quant_block_rotation_mfma_moe_sorting_kernel``).

Root cause under test
---------------------
The kernel traverses source rows in blocks of ``GROUP_QUANT_BLOCK_SIZE`` (64).
The launch grid is ``ceil(rows / 64)`` (further divided by ``persist_m * W``),
so for a token count that is **not** a multiple of 64 the final block(s) address
rows ``>= rows`` -- i.e. past the end of the ``input`` tensor. Because the input
buffer resource is built with ``max_size=True`` (a 4 GB descriptor), the hardware
bounds check is effectively disabled, so the staged-X HBM->LDS DMA
(``buffer_load_dwordx4 ... offen lds``) reads unmapped memory and faults with
``MEMORY_VIOLATION`` / ``hipErrorIllegalAddress``.

Why this is normally *intermittent*: with the default caching allocator the input
is sub-allocated inside a large segment, so the tail read usually lands on a
mapped (neighbouring) page and silently returns garbage. It only crashes when the
read happens to cross a segment/page boundary into unmapped memory.

Making it deterministic
-----------------------
1. ``PYTORCH_NO_CUDA_MEMORY_CACHING=1`` -- every tensor is a direct hipMalloc, so
   ``input`` is its own mapping with unmapped VA right after it.
2. Allocate ``input`` *last* (outputs first) so nothing is mapped immediately
   after it.
3. Pick a shape whose tail / grid-overrun read reaches several MB past the input
   end -- far beyond any 2 MB allocation-rounding slack -- guaranteeing the read
   hits an unmapped page.

Usage
-----
    # baseline (unpatched): expect an illegal-memory-access crash
    HIP_VISIBLE_DEVICES=0 python op_tests/test_flydsl_rotation_quant_sort_oob_crash.py

    # after the fix: expect it to run clean
    HIP_VISIBLE_DEVICES=0 python op_tests/test_flydsl_rotation_quant_sort_oob_crash.py --expect clean
"""
import os

# Must be set before torch initialises the HIP context.
os.environ.setdefault("PYTORCH_NO_CUDA_MEMORY_CACHING", "1")
os.environ.setdefault("HIP_LAUNCH_BLOCKING", "1")
os.environ.setdefault("AMD_SERIALIZE_KERNEL", "3")

import argparse
import sys

import torch

from aiter import dtypes
from aiter.ops.flydsl import (
    flydsl_per_1x32_fp4_quant_block_rotation_mfma_sort_inplace,
)

G = 32  # FP4 / E8M0 group size


def _make_sorted_ids(token_num: int, topk: int):
    """Minimal identity ``sorted_ids`` (each expanded row routed once), padded
    to a multiple of 32 with the ``token_num`` sentinel, mirroring
    ``moe_sort_block_fwd`` output."""
    rows = token_num * topk
    src = torch.arange(rows, dtype=torch.int64)
    token_idx = src // topk
    topk_id = src % topk
    info = ((topk_id << 24) | token_idx).to(torch.int32)
    sorted_len = (rows + 31) // 32 * 32
    sorted_ids = torch.full((sorted_len,), token_num, dtype=torch.int32)
    sorted_ids[:rows] = info
    return (
        sorted_ids.to("cuda"),
        torch.tensor([rows], dtype=torch.int32, device="cuda"),
        rows,
    )


def run_once(token_num: int, topk: int, cols: int, dtype, waves: int) -> bool:
    """Launch the fused per-block rotation+quant+sort kernel once with a
    non-64-aligned row count. Returns True if it completed without an illegal
    memory access, False if it crashed."""
    assert cols % G == 0
    scale_n = cols // G
    rows = token_num * topk
    assert rows % 64 != 0, "row count must be non-64-aligned to trigger the tail OOB"

    sorted_ids, num_valid_ids, rows = _make_sorted_ids(token_num, topk)

    row_bytes = cols * 2  # bf16 / fp16 input element = 2 bytes
    valid_bytes = rows * row_bytes
    num_row_blocks = (rows + 63) // 64
    # With persist_m=1 and W waves/WG, the grid covers
    # ceil(num_row_blocks / W) * W blocks; every wave runs the staged-X DMA for
    # its 64-row block, so blocks beyond num_row_blocks are read fully OOB. Using
    # W here overshoots the input end by up to (W*64 - rows) rows -- many MB,
    # far past any 2 MB allocation-rounding slack -> a guaranteed page fault.
    blocks_launched = ((num_row_blocks + waves - 1) // waves) * waves
    max_row_read = blocks_launched * 64 - 1
    max_read_bytes = (max_row_read + 1) * row_bytes
    oob_bytes = max_read_bytes - valid_bytes

    print(
        f"config: token_num={token_num} topk={topk} rows={rows} (rows%64="
        f"{rows % 64}) cols={cols} scale_n={scale_n} dtype={str(dtype).split('.')[-1]}\n"
        f"  input valid size   = {valid_bytes/1e6:8.3f} MB\n"
        f"  max kernel read to = {max_read_bytes/1e6:8.3f} MB (row {max_row_read})\n"
        f"  => reads up to     ~{oob_bytes/1e6:8.3f} MB PAST the input end",
        flush=True,
    )

    # Allocate outputs FIRST so `x` is the last device mapping; the tail OOB read
    # then lands in unmapped VA right after `x`.
    out_u8 = torch.empty(rows, cols // 2, dtype=torch.uint8, device="cuda")
    out_scale = torch.empty(
        (int(sorted_ids.shape[0]) + 31) // 32 * 32,
        cols // 32,
        dtype=dtypes.fp8_e8m0,
        device="cuda",
    )
    # Per-block R (scale_n, 32, 32) -> selects the per-block kernel that faults
    # in the field log (per_1x32_fp4_quant_block_rotation_mfma_moe_sorting_kernel).
    R = torch.empty(scale_n, G, G, dtype=dtype, device="cuda")
    R.normal_()
    # `x` allocated LAST, exact size.
    x = torch.empty(rows, cols, dtype=dtype, device="cuda")
    x.normal_()
    torch.cuda.synchronize()

    try:
        flydsl_per_1x32_fp4_quant_block_rotation_mfma_sort_inplace(
            out_u8, x, R, out_scale, sorted_ids, num_valid_ids, token_num,
            persist_m=1, waves_per_wg=waves,
        )
        torch.cuda.synchronize()
    except RuntimeError as e:
        print(f"\n>>> CRASH: {type(e).__name__}: {e}", flush=True)
        return False
    print("\n>>> completed without illegal memory access", flush=True)
    return True


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                description=__doc__)
    p.add_argument("-t", "--token", type=int, default=1,
                   help="token_num (rows = token_num * topk). Default 1.")
    p.add_argument("-topk", "--topk", type=int, default=1)
    p.add_argument("-dim", "--cols", type=int, default=16384,
                   help="cols (<=16384 for mxfp4_moe_sort_hip).")
    p.add_argument("-w", "--waves", type=int, default=4,
                   help="waves_per_wg; more waves -> more fully-OOB blocks -> "
                        "larger overshoot past the input end.")
    p.add_argument("-d", "--dtype", choices=["bf16", "fp16"], default="bf16")
    p.add_argument("--expect", choices=["crash", "clean"], default="crash",
                   help="crash = baseline (bug present); clean = after the fix.")
    args = p.parse_args()

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    completed = run_once(args.token, args.topk, args.cols, dtype, args.waves)

    crashed = not completed
    if args.expect == "crash":
        if crashed:
            print("\nRESULT: crash reproduced as expected (bug present).")
            sys.exit(0)
        print("\nRESULT: expected a crash but the kernel ran clean "
              "(bug not triggered / already patched).")
        sys.exit(3)
    else:  # expect clean
        if not crashed:
            print("\nRESULT: ran clean as expected (fix works).")
            sys.exit(0)
        print("\nRESULT: expected clean but the kernel crashed (fix missing/broken).")
        sys.exit(4)


if __name__ == "__main__":
    main()
