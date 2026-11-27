# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""MiniMax-M3 decode index block scores over the NVFP4 index cache (gfx1250).

``aiter.ops.flydsl.index_score_decode`` against an FP32 torch reference:
score[h, b * q + t, blk] = max over the block's keys before the row's cutoff
(seq_len - q + t + 1) of (hi + lo / 16) . dequant(K), times sm_scale * log2(e),
hi / lo the query's two E4M3 terms. The kernel's dots are the hardware's
128-deep scaled WMMA, the reference's an FP32 matmul, so they agree to a few
1e-3 relative, not bit for bit: a score is graded against the WMMA's error
envelope (``SCALED_DOT_REL`` of the sum of the products' magnitudes). Bitwise
equality with ATOM's Triton scorer, which issues the same WMMA, is checked on
the ATOM side
(``tests/test_minimax_m3_decode_score_flydsl.py``).

Shapes are the model's: 4 index heads x 8 query rows a request at TP1 (DSpark
K = 7), 1 x 8 on a TP4 rank, plain decode (1 row), contexts of 50K-300K tokens,
the 9216 B pages of ATOM's NVFP4 index pool (payload and scales two byte views
of one buffer).

Run with: python op_tests/test_flydsl_index_score_decode_nvfp4.py
"""

import argparse
import itertools

import pandas as pd
import torch

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.test_common import benchmark, checkAllclose, run_perftest

SUPPORTED_GFX = ["gfx1250"]
E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
PAGE, D = 128, 128
PAGE_BYTES = PAGE * (D // 2 + D // 16)


def dequant(payload, scale):
    """[..., 128, 64] e2m1 pairs and [..., 128, 8] E4M3 scales -> fp32 keys."""
    table = torch.tensor(E2M1, device=payload.device)
    codes = torch.stack([payload & 15, payload >> 4], -1).flatten(-2).long()
    values = table[codes & 7] * torch.where(codes >= 8, -1.0, 1.0)
    values = values.reshape(*values.shape[:-1], -1, 16)
    return (values * scale.view(torch.float8_e4m3fn).float()[..., None]).flatten(-2)


def make_inputs(lens, query_len, heads, seed=0):
    """The pool as ATOM lays it out: one [pages, 9216] byte buffer, the payload
    its first 8192 bytes a page and the scales the last 1024; pages of a
    request scattered by the block table."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    nbs = [-(-n // PAGE) for n in lens]
    pages = sum(nbs) + 2
    pool = torch.randint(
        0, 256, (pages, PAGE_BYTES), dtype=torch.uint8, device="cuda", generator=g
    )
    payload = pool[:, : PAGE * D // 2].unflatten(1, (PAGE, D // 2))
    scale = pool[:, PAGE * D // 2 :].unflatten(1, (PAGE, D // 16))
    # Finite scales of mixed magnitude: a constant scale would hide a scale-row
    # mix-up, a NaN one the comparison.
    scale.copy_(
        (torch.rand(pages, PAGE, D // 16, device="cuda", generator=g) * 2 + 0.01)
        .to(torch.float8_e4m3fn)
        .view(torch.uint8)
    )
    # A few queries outside E4M3's range, so the clamp is exercised.
    q = torch.randn(len(lens) * query_len, heads, D, device="cuda", generator=g) * 3
    q[::7, :, ::31] = 600.0
    q = q.to(torch.bfloat16)
    perm = torch.randperm(pages, device="cuda", generator=g).int()
    bt = torch.zeros(len(lens), max(nbs), dtype=torch.int32, device="cuda")
    s = 0
    for i, nb in enumerate(nbs):
        bt[i, :nb] = perm[s : s + nb]
        s += nb
    return q, payload, scale, bt, torch.tensor(lens, dtype=torch.int32, device="cuda")


# The scaled WMMA does not sum its 128 products exactly in fp32: on MI455X a dot
# lands ~2^-12 of the sum of the products' magnitudes from the exact one. A
# score is graded against this fraction of that sum (ATOM's
# tests/test_minimax_m3_fp4_index_scorers.py uses the same bound).
SCALED_DOT_REL = 2.0**-9


def run_torch(q, payload, scale, bt, seq_lens, sm_scale, query_len, max_block):
    """Reference only: FP32 math. Returns (score, live, tol) -- the value of
    each block the kernel must write (the rest of ``score`` is NaN), which
    those are, and the error a WMMA score may have there."""
    from aiter.ops.flydsl.kernels.index_score_decode_nvfp4_gfx1250 import (
        sm_scale_log2e,
    )

    rows, heads, _ = q.shape
    q32 = q.float().clamp(-448, 448)
    hi = q32.to(torch.float8_e4m3fn).float()
    lo = ((q32 - hi) * 16).to(torch.float8_e4m3fn).float()
    q_eff = hi + lo / 16
    q_mag = hi.abs() + lo.abs() / 16
    c = sm_scale_log2e(sm_scale)
    score = torch.full((heads, rows, max_block), float("nan"), device=q.device)
    tol = torch.zeros_like(score)
    live = torch.zeros_like(score, dtype=torch.bool)
    for b, n in enumerate(seq_lens.tolist()):
        nb = min(-(-n // PAGE), max_block)
        if nb == 0:
            continue
        pages = bt[b, :nb].long()
        k = dequant(payload[pages], scale[pages]).reshape(-1, D)
        pos = torch.arange(nb * PAGE, device=q.device)
        for t in range(query_len):
            r = b * query_len + t
            hidden = pos[None] >= n - query_len + t + 1
            dots = (q_eff[r] @ k.T).masked_fill(hidden, float("-inf"))
            mags = (q_mag[r] @ k.abs().T).masked_fill(hidden, 0.0)
            score[:, r, :nb] = dots.view(heads, nb, PAGE).amax(-1) * c
            tol[:, r, :nb] = mags.view(heads, nb, PAGE).amax(-1) * c * SCALED_DOT_REL
            live[:, r, :nb] = True
    return score, live, tol + 1e-6


@benchmark()
def test_index_score_decode_nvfp4(
    batch, context, query_len, heads, workgroups, ring, lens
):
    from aiter.ops.flydsl.index_score_decode import (
        index_score_decode_nvfp4,
        index_score_decode_nvfp4_supported,
    )

    sm_scale = D**-0.5
    q, payload, scale, bt, seq_lens = make_inputs(lens, query_len, heads)
    max_block = bt.shape[1]
    assert index_score_decode_nvfp4_supported(q, payload, scale, query_len)
    ref, live, tol = run_torch(
        q, payload, scale, bt, seq_lens, sm_scale, query_len, max_block
    )
    # A row whose cutoff is not positive sees no key: -inf, exactly.
    hidden = live & (ref == float("-inf"))
    graded = live & ~hidden
    # NaN-poisoned: a block the kernel skips, or writes past the live ones,
    # shows.
    score = torch.full_like(ref, float("nan"))

    def flydsl():
        index_score_decode_nvfp4(
            q,
            payload,
            scale,
            score,
            bt,
            seq_lens,
            sm_scale,
            query_len,
            workgroups=workgroups or None,
            ring=ring,
        )
        return score

    candidates = {"flydsl": flydsl}
    blocks = sum(min(-(-n // PAGE), max_block) for n in lens)
    nbytes = blocks * PAGE_BYTES + q.numel() * 2 + int(live.sum()) * 4
    # Two scaled dots (hi, lo) a (row, head, key).
    flops = 2 * 2 * heads * query_len * blocks * PAGE * D
    ret = {"gfx": get_gfx_runtime()}
    for name, fn in candidates.items():
        out, us = run_perftest(fn)
        assert torch.isnan(
            out[~live]
        ).all(), f"{name}: wrote a block past the live ones"
        assert not torch.isnan(out[live]).any(), f"{name}: a live block not written"
        assert (out[hidden] == float("-inf")).all(), f"{name}: a hidden block scored"
        # The error in units of the WMMA's envelope at each block: within 1.
        err = checkAllclose(
            torch.zeros_like(ref),
            ((out - ref) / tol).nan_to_num(0.0, posinf=2.0, neginf=-2.0),
            rtol=0,
            atol=1.0,
            mask=graded,
            msg=f"{name}: index_score_decode_nvfp4 b{batch} ctx{context} "
            f"q{query_len} h{heads}",
        )
        assert err == 0, f"{name}: {err:.2%} of the scores outside the WMMA bound"
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = err
    return ret


def request_lens(batch, context, ragged):
    """``batch`` request lengths: all ``context``, or ragged below it (down to
    one token, block boundaries included)."""
    if not ragged:
        return [context] * batch
    edge = [context, context - 1, 129, 1, context // 2 + 1, 128, 7, 127]
    g = torch.Generator().manual_seed(batch + context)
    rest = torch.randint(1, context + 1, (max(0, batch - len(edge)),), generator=g)
    return (edge + rest.tolist())[:batch]


def main():
    if get_gfx_runtime() not in SUPPORTED_GFX:
        aiter.logger.warning(
            "index_score_decode_nvfp4 unsupported on %s; skipping", get_gfx_runtime()
        )
        return
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="config input of test",
    )
    parser.add_argument("-b", "--batch", type=int, nargs="*", default=[4, 8, 16])
    parser.add_argument(
        "-c",
        "--context",
        type=int,
        nargs="*",
        default=[50000, 150000, 300000],
        help="context tokens a request",
    )
    parser.add_argument(
        "-q",
        "--rows",
        type=dtypes.str2tuple,
        nargs="*",
        default=[(8, 4), (8, 1), (1, 4), (1, 1), (3, 5)],
        help="(query rows a request, index heads), e.g. 8,4",
    )
    parser.add_argument(
        "-w",
        "--workgroups",
        type=int,
        nargs="*",
        default=[0],
        help="workgroup budget of the launch (0: the op's default)",
    )
    parser.add_argument("-r", "--ring", type=int, nargs="*", default=[4])
    parser.add_argument(
        "--ragged", type=int, nargs="*", default=[0, 1], help="1: mixed lengths"
    )
    args = parser.parse_args()

    cells = list(
        itertools.product(
            args.ragged, args.batch, args.context, args.rows, args.workgroups, args.ring
        )
    )
    # Launch knobs off their defaults: one workgroup a 64-block window (the
    # widest chunk), a large budget, the shortest and a long ring.
    cells += [
        (1, 8, 64 * 18 * PAGE, (8, 4), workgroups, ring)
        for workgroups, ring in ((1, 4), (1, 2), (96, 6), (4096, 3))
    ]
    df = []
    for (
        ragged,
        batch,
        context,
        (query_len, heads),
        workgroups,
        ring,
    ) in cells:
        df.append(
            test_index_score_decode_nvfp4(
                batch,
                context,
                query_len,
                heads,
                workgroups,
                ring,
                request_lens(batch, context, ragged),
            )
        )
    df = pd.DataFrame(df).drop(columns=["lens"], errors="ignore")
    aiter.logger.info(
        "index_score_decode_nvfp4 summary (markdown):\n%s", df.to_markdown(index=False)
    )


if __name__ == "__main__":
    main()
