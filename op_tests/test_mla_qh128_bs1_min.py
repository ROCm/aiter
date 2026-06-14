# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""batch=1 MINIMAL stage1-only NaN reproducer for qh128-q1-16mx4-64nx1-np.

Motivation
----------
The input-side probe (MLA_ZERO_UNUSED_V) proved the stage1 NaN is NOT caused by
invalid input V in the tail-page padding: that input region is finite and
zeroing it changes nothing. The NaN appears even at num_kv_splits=1 in splitData
while splitLse stays finite -> it is produced INSIDE the kernel (masked-lane V/P
staging), not read from global memory.

ROOT CAUSE (proven by --repeat): the NaN is GPU-STATE dependent, not input
dependent. Repeating the *identical* launch shows:

    iter 0 (cold): nan=0          <- registers/LDS start clean -> finite
    iter 1..N:     nan=763904     <- inherits stale NaN from the prior launch

AMD VGPR/LDS are NOT cleared between kernel launches. The kernel computes a
NaN/Inf in some masked / out-of-range lane that does not reach this launch's
output but lingers in a VGPR/LDS slot; the next launch's masked V-load is
predicated off, leaves that VGPR unwritten, and P*V does 0*staleNaN = NaN. This
reproduces at num_kv_splits=1 too, so it is the common core_loop V/P staging,
not a multi-pass-only path.

To give that bug a minimal carrier for asm inspection / GPU-side probes, this
script slices a real ATOM seg dump down to a SINGLE sequence (batch=1) and
replays stage1-only, optionally repeating the launch to expose the cold/warm
split.

It does NOT modify the existing test file; it reuses the building blocks
(`_stage1_only`, `_stage1_diag`, `_torch_reduce`, `_normalize_dump`) from
test_mla_qh128_e2e.py so the input construction is byte-for-byte the same.

Slicing keeps the full `kv_compact` buffer intact (the kernel indexes it via
kv_indices) and only narrows the per-batch metadata:
  q          -> q[b:b+1]
  kv_indices -> kv_indices[kv_indptr[b]:kv_indptr[b+1]]   (this batch's pages)
  kv_indptr  -> [0, n_pages_b]
  kv_last    -> [kv_last_page_lens[b]]
  qo_indptr  -> [0, 1]

Usage (ff_mla container):

    docker exec ff_mla bash -lc 'cd /home/carhuang/feifei/aiter && \\
      ENABLE_CK=0 ENABLE_FLYDSL=0 \\
      AITER_ASM_DIR=/home/carhuang/feifei/aiter/hsa \\
      MLA_DECODE_DUMP_DIR=/root/mla_decode_dump \\
      python3 op_tests/test_mla_qh128_bs1_min.py --splits 1'

    # only one specific batch index:
    python3 op_tests/test_mla_qh128_bs1_min.py --batch 3 --splits 1 2
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_mla_qh128_e2e as e2e  # noqa: E402  (reuse exact building blocks)

import aiter  # noqa: E402
from aiter.jit.utils.chip_info import get_gfx  # noqa: E402

DUMP_DIR = os.environ.get("MLA_DECODE_DUMP_DIR", "/root/mla_decode_dump")


def _first_dump() -> str:
    paths = sorted(
        glob.glob(os.path.join(DUMP_DIR, "mla_decode_*.pt"))
        + glob.glob(os.path.join(DUMP_DIR, "seg_decode_*.pt"))
    )
    if not paths:
        raise SystemExit(f"no dumps under {DUMP_DIR}")
    return paths[0]


def _slice_batch(b: dict, batch_idx: int, dev: torch.device):
    """Return a single-sequence (batch=1) view of dump dict `b` for `batch_idx`.

    Keeps kv_compact whole; only the index metadata + q row are narrowed."""
    kv_indptr = b["kv_indptr"].to(torch.int64).cpu()
    kv_indices_full = b["kv_indices"].to(torch.int64).cpu()
    kv_last_full = b["kv_last_page_lens"].to(torch.int64).cpu()
    page_size = int(b["page_size"])

    lo = int(kv_indptr[batch_idx])
    hi = int(kv_indptr[batch_idx + 1])
    n_pages = hi - lo
    pages = kv_indices_full[lo:hi]
    last = int(kv_last_full[batch_idx])
    ctx_len = (n_pages - 1) * page_size + last if n_pages > 0 else 0
    tail_len = last if 0 < last < page_size else 0

    seg_kv = b["kv_compact"].to(dev).contiguous()
    q_row = b["q"][batch_idx : batch_idx + 1].to(dev).contiguous()
    o_server_row = b["o_server"][batch_idx : batch_idx + 1].to(dev)

    sliced = {
        "seg_kv": seg_kv,
        "q": q_row,
        "kv_indices": pages.to(torch.int32).to(dev),
        "kv_indptr": torch.tensor([0, n_pages], dtype=torch.int32, device=dev),
        "kv_last_page_lens": torch.tensor(
            [last], dtype=torch.int32, device=dev
        ),
        "qo_indptr": torch.tensor([0, 1], dtype=torch.int32, device=dev),
        "o_server": o_server_row,
        "n_pages": n_pages,
        "ctx_len": ctx_len,
        "tail_len": tail_len,
        "full_pages": n_pages - (1 if tail_len else 0),
    }
    return sliced


def _run_one(b: dict, sliced: dict, num_kv_splits: int, dev: torch.device):
    o_server = sliced["o_server"]
    total_s, nhead, v_head_dim = o_server.shape
    q_scale = b["q_scale"].to(dev) if b.get("q_scale") is not None else None
    kv_scale = b["kv_scale"].to(dev) if b.get("kv_scale") is not None else None
    num_kv_splits_indptr = torch.tensor(
        [0, num_kv_splits], dtype=torch.int32, device=dev
    )

    logits, attn_lse = e2e._stage1_only(
        q=sliced["q"],
        kv_buffer=sliced["seg_kv"],
        qo_indptr=sliced["qo_indptr"],
        kv_indptr=sliced["kv_indptr"],
        kv_indices=sliced["kv_indices"],
        kv_last_page_lens=sliced["kv_last_page_lens"],
        num_kv_splits_indptr=num_kv_splits_indptr,
        num_kv_splits=num_kv_splits,
        max_seqlen_q=int(b["max_q_len"]),
        page_size=int(b["page_size"]),
        sm_scale=float(b["sm_scale"]),
        nhead=nhead,
        v_head_dim=v_head_dim,
        q_scale=q_scale,
        kv_scale=kv_scale,
    )
    lg_fin = bool(torch.isfinite(logits.float().cpu()).all().item())
    lse_fin = bool(torch.isfinite(attn_lse.float().cpu()).all().item())
    finite = lg_fin and lse_fin
    nan_logits = int(torch.isnan(logits.float()).sum().item())
    _, diag = e2e._stage1_diag(logits, attn_lse)
    cos = float("nan")
    if finite:
        out = e2e._torch_reduce(logits, attn_lse)
        cos = e2e._cos(out[..., :v_head_dim], o_server[..., :v_head_dim])
    return finite, diag, cos, nan_logits


def main() -> None:
    parser = argparse.ArgumentParser(
        description="batch=1 minimal stage1-only NaN reproducer (qh128)."
    )
    parser.add_argument(
        "--dump",
        default="first",
        help="dump path, or 'first' (default) for the first dump in MLA_DECODE_DUMP_DIR.",
    )
    parser.add_argument(
        "--batch",
        default="scan",
        help="batch index to slice, or 'scan' (default) to find the first "
        "single-sequence slice that reproduces NaN.",
    )
    parser.add_argument(
        "--splits",
        type=int,
        nargs="+",
        default=[1],
        help="num_kv_splits values to run per sliced batch (default: 1).",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="re-launch the IDENTICAL kernel call N times per (batch,split) and "
        "print nan-per-iter. Use >=2 to expose the cold(finite)/warm(NaN) split "
        "that proves the uninitialized VGPR/LDS read. Default 1.",
    )
    args = parser.parse_args()

    if get_gfx() != "gfx1250":
        raise SystemExit(f"requires gfx1250 (mi400), got {get_gfx()}")

    dev = torch.device("cuda")
    path = _first_dump() if args.dump == "first" else args.dump
    b = e2e._normalize_dump(torch.load(path, map_location="cpu"))
    bs = int(b["kv_indptr"].numel() - 1)
    aiter.logger.info(
        "bs1-min: dump=%s total_batches=%d splits=%s",
        os.path.basename(path),
        bs,
        args.splits,
    )

    if args.batch == "scan":
        batch_indices = list(range(bs))
    else:
        batch_indices = [int(args.batch)]

    repro = []  # (batch_idx, ctx, splits)
    first_repro_geom = None
    for bi in batch_indices:
        sliced = _slice_batch(b, bi, dev)
        for nks in args.splits:
            nan_seq = []
            finite = True
            diag = ""
            cos = float("nan")
            for _ in range(max(1, args.repeat)):
                finite, diag, cos, nan_logits = _run_one(b, sliced, nks, dev)
                nan_seq.append(nan_logits)
            tag = "FINITE" if finite else "NaN/Inf"
            seq_str = (
                ("  nan-per-iter=%s" % nan_seq) if args.repeat > 1 else ""
            )
            aiter.logger.info(
                "  batch=%-3d splits=%d ctx=%d n_pages=%d tail_len=%d "
                "full_pages=%d -> %s cos=%s%s %s",
                bi,
                nks,
                sliced["ctx_len"],
                sliced["n_pages"],
                sliced["tail_len"],
                sliced["full_pages"],
                tag,
                ("%.4f" % cos) if cos == cos else "nan",
                seq_str,
                "" if finite else diag,
            )
            # cold(finite)->warm(NaN) transition is the proof signature.
            if args.repeat > 1 and nan_seq[0] == 0 and any(n > 0 for n in nan_seq[1:]):
                aiter.logger.info(
                    "    ^ COLD launch finite, WARM launches NaN => uninitialized "
                    "VGPR/LDS read (stale NaN inherited across launches)."
                )
            if not finite:
                repro.append((bi, sliced["ctx_len"], nks))
                if first_repro_geom is None:
                    first_repro_geom = (bi, sliced, nks, diag)

    aiter.logger.info("=== bs1-min summary ===")
    aiter.logger.info(
        "  reproducing single-sequence slices: %d (out of %d batches x %d splits)",
        len(repro),
        len(batch_indices),
        len(args.splits),
    )
    if first_repro_geom is not None:
        bi, sliced, nks, diag = first_repro_geom
        aiter.logger.info(
            "  MINIMAL REPRO: batch=%d splits=%d ctx_len=%d n_pages=%d "
            "tail_len=%d full_pages=%d",
            bi,
            nks,
            sliced["ctx_len"],
            sliced["n_pages"],
            sliced["tail_len"],
            sliced["full_pages"],
        )
        aiter.logger.info("    diag: %s", diag)
        aiter.logger.info(
            "    -> batch=1 single-pass stage1 produces NaN with CLEAN finite "
            "input => kernel-internal (masked-lane V/P staging), not input data."
        )
    else:
        aiter.logger.info("  no single-sequence slice reproduced NaN.")


if __name__ == "__main__":
    main()
