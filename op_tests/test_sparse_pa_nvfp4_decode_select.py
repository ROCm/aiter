# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""MiniMax-M3 NVFP4 sparse paged decode with the index top-16 selected in the attention
workgroup (``sparse_pa_nvfp4_decode(index_score=...)``, FlyDSL, gfx950).

Checks, against a torch reference of the selection contract and against the unfused chain
(the same kernel fed the reference ids):

  * the stored ``topk_idx`` equals the reference selection: descending score, ties to the
    larger block, NaN above +inf, ``init_blocks`` / ``local_blocks`` pins, -1 past the live
    count;
  * the attention output is bit-identical to the unfused kernel on those ids (same splits).

Rows cover the window path (<= 1792 live blocks), the window loop (> 1792) and the exact
path (fewer than 16 live blocks, constant / heavily tied rows).

Run (exits nonzero if any check fails):
    python op_tests/test_sparse_pa_nvfp4_decode_select.py
"""

import sys

import numpy as np
import torch
from test_flydsl_sparse_pa_nvfp4 import BLK, SM_SCALE, make_case

from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops.flydsl.kernels.sparse_pa_nvfp4_decode import (
    decode_select_supported,
    sparse_pa_nvfp4_decode,
)

TOPK = 16
INIT_PIN, LOCAL_PIN = 1e30, 1e29


def order_keys(v):
    """int32 order keys of fp32 ``v`` (numpy): the float's bits with negatives' magnitude bits
    flipped (so -0.0 sits one step below 0.0), NaN on top."""
    bits = v.astype(np.float32).view(np.int32)
    key = bits ^ ((bits >> 31) & 0x7FFFFFFF)
    return np.where(np.isnan(v), np.int32(2**31 - 1), key)


def reference_ids(scores, live, init_blocks, local_blocks):
    """[rows, 16] ids of ``scores`` [rows, width] (fp32 cpu), ``live`` blocks per row."""
    rows = scores.shape[0]
    out = torch.full((rows, TOPK), -1, dtype=torch.int32)
    for r in range(rows):
        n = int(live[r])
        if n == 0:
            continue
        v = scores[r, :n].numpy().astype(np.float32).copy()
        col = np.arange(n)
        if init_blocks or local_blocks:
            v[col < init_blocks] = np.float32(INIT_PIN)
            v[col >= max(0, n - local_blocks)] = np.float32(LOCAL_PIN)
        key = order_keys(v).astype(np.int64)
        sel = np.lexsort((-col, -key))[:TOPK]  # key descending, ties to the larger id
        out[r, : len(sel)] = torch.from_numpy(sel.astype(np.int32))
    return out


def make_scores(rows, width, live, kind, seed):
    g = torch.Generator().manual_seed(seed)
    s = torch.randn(rows, width, generator=g)
    if kind == "ties":  # many equal scores, including at the 16th place
        s = (s * 2).round() / 2
    elif kind == "const":
        s = torch.zeros(rows, width)
    elif kind == "special":  # +-inf, NaN, +-0 among live blocks
        for r in range(rows):
            n = int(live[r])
            for j, val in zip(
                torch.randint(0, max(n, 1), (6,), generator=g).tolist(),
                [float("inf"), float("-inf"), float("nan"), 0.0, -0.0, float("nan")],
            ):
                if j < n:
                    s[r, j] = val
    elif (
        kind == "rising"
    ):  # scores rise toward the row's end (more windows hold candidates)
        s = s * 0.1 + torch.arange(width).float() / 64
    # past the live blocks the scorer writes nothing: leave garbage the kernel must ignore
    for r in range(rows):
        s[r, int(live[r]) :] = float("nan") if r % 2 else 1e38
    return s


def run_case(
    name, seq_lens, ql, width, kind, init_blocks=0, local_blocks=0, splits=4, xcds=None
):
    c = make_case(seq_lens, [ql] * len(seq_lens), 1, 16, seed=len(name))
    rows = c["q"].shape[0]
    live = ((c["row_kv"] + BLK - 1) // BLK).cpu()
    scores = make_scores(rows, width, live, kind, seed=rows + width)
    ref = reference_ids(scores, live, init_blocks, local_blocks)
    topk = torch.full((1, rows, TOPK), -7, dtype=torch.int32, device="cuda")
    out = torch.zeros_like(c["q"])
    sparse_pa_nvfp4_decode(
        c["q"], c["cache"], topk, c["bt"], c["seq_lens"], SM_SCALE, out,
        c["k_scale"], c["v_scale"], query_len=ql, splits=splits, xcds=xcds,
        index_score=scores.cuda().view(1, rows, width),
        index_max_seq_len=max(seq_lens), init_blocks=init_blocks, local_blocks=local_blocks,
    )  # fmt: skip
    ref_out = torch.zeros_like(c["q"])
    sparse_pa_nvfp4_decode(
        c["q"], c["cache"], ref.cuda().view(1, rows, TOPK), c["bt"], c["seq_lens"], SM_SCALE,
        ref_out, c["k_scale"], c["v_scale"], query_len=ql, splits=splits, xcds=xcds,
    )  # fmt: skip
    torch.cuda.synchronize()
    ids_ok = torch.equal(topk.cpu()[0], ref)
    out_ok = torch.equal(out, ref_out)
    print(
        f"{'PASS' if ids_ok and out_ok else 'FAIL'}  {name:28s} rows={rows:3d} width={width:5d} "
        f"splits={splits} xcds={xcds} ids={'ok' if ids_ok else 'MISMATCH'} out={'ok' if out_ok else 'MISMATCH'}"
    )
    if not ids_ok:
        bad = (topk.cpu()[0] != ref).any(1).nonzero().flatten()[:3].tolist()
        for r in bad:
            print(
                "   row",
                r,
                "live",
                int(live[r]),
                "got",
                topk.cpu()[0, r].tolist(),
                "ref",
                ref[r].tolist(),
            )
    return ids_ok and out_ok


def check_overlap_rejected():
    """topk_idx is written when index_score is given: an expanded / overlapping layout is refused,
    while the same broadcast view stays valid for the read-only call (no index_score).
    """
    seq_lens, ql, width = [1000, 700], 1, 64
    c = make_case(seq_lens, [ql] * 2, 1, 16, seed=3)
    rows = c["q"].shape[0]
    scores = torch.randn(1, rows, width, device="cuda")
    call = lambda topk, **kw: sparse_pa_nvfp4_decode(
        c["q"], c["cache"], topk, c["bt"], c["seq_lens"], SM_SCALE, None,
        c["k_scale"], c["v_scale"], query_len=ql, splits=1, **kw,
    )  # fmt: skip
    ids = torch.zeros(1, 1, TOPK, dtype=torch.int32, device="cuda")
    select_kw = {"index_score": scores, "index_max_seq_len": max(seq_lens)}
    ok = True
    for name, topk in (
        ("rows expanded", ids.expand(1, rows, TOPK)),
        ("heads expanded", torch.zeros(1, rows, TOPK, dtype=torch.int32, device="cuda").expand(2, rows, TOPK)),
    ):  # fmt: skip
        try:
            call(topk, **select_kw)
            refused = False
        except ValueError:
            refused = True
        print(f"{'PASS' if refused else 'FAIL'}  overlap refused: {name}")
        ok &= refused
    return ok


def main():
    if get_gfx_runtime() != "gfx950":
        print("SKIP: gfx950 only")
        return 0
    ok = check_overlap_rejected()
    # (name, seq_lens, query_len, width)
    shapes = [
        ("short_ctx", [1, 100, 1500, 2047, 2048, 2049], 1, 64),
        ("k7_32K", [32768, 20000], 8, 256),
        ("k7_175K", [175000, 90001, 4097], 8, 2048),
        ("window_edge", [229376 - 5, 229376, 229376 + 3], 1, 2048),
        ("loop_300K", [300000, 100000], 4, 4096),
    ]
    for name, lens, ql, width in shapes:
        for kind in ("random", "ties", "const", "special", "rising"):
            ok &= run_case(f"{name}/{kind}", lens, ql, width, kind)
    for init, local in ((1, 2), (0, 1), (3, 0)):
        ok &= run_case(
            f"k7_175K/pins{init},{local}", [175000, 90001], 8, 2048, "ties", init, local
        )
    for splits, xcds in ((1, None), (2, 8), (4, 8), (8, 8)):
        ok &= run_case(
            f"k7_175K/s{splits}",
            [175000, 120000, 33000],
            8,
            2048,
            "random",
            splits=splits,
            xcds=xcds,
        )
    assert decode_select_supported(8) and not decode_select_supported(512)
    print("ALL PASS" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
