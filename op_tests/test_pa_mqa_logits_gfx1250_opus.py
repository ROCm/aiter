# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""MXFP4 paged MQA logits (OPUS, 32x16x128 WMMA) -- correctness and perf on gfx1250.

Emits three markdown tables: the corner cases, a causal prefill sweep and ATOM's two
CSA-compressed prefill regimes.

Every launch goes through ``aiter.ops.opus.pa_mqa_logits_mxfp4``, the arch dispatcher, and not
through the gfx1250 arm directly -- so the path under test is the path a caller gets, layout
check included.

    python3 op_tests/test_pa_mqa_logits_gfx1250_opus.py            # the full default sweep
    python3 op_tests/test_pa_mqa_logits_gfx1250_opus.py -b 1 2     # a quick subset

THE DATA CARRIES A PER-ROW MAGNITUDE SPREAD (``randn_spread``) AND THAT IS LOAD-BEARING. Plain
``torch.randn`` gives every 32-element block nearly the same ``amax``, hence nearly the same
E8M0 exponent -- so a scale routed to the WRONG token reads an exponent that happens to be
right. A deliberately misrouted ``b_scale_sel`` passed the standalone suite at
``cos = 0.999952`` on ``randn`` data and only failed at 0.396 once each KV token and Q head got
its own power-of-two multiplier. Uniform MAGNITUDES hide a scale-routing bug exactly as uniform
values hide a K permutation, which is also why no case here uses uniform data.
``test_scale_spread_has_teeth`` asserts the instrument still spreads.

There is no second implementation to cross-check against on this target -- gfx1250 has no
FlyDSL fp4 MQA-logits kernel -- so the dequantized fp32 reference is the only judge. It runs
over the same quantized values the kernel sees, so ``err`` is expected at ~1e-6 (fp32
accumulation order), not at fp4 resolution. The K-permutation and scale-routing probes a
reference cannot give live in the opus-ops standalone harness.
"""

import argparse
import itertools
import random
from dataclasses import dataclass

import pandas as pd
import torch

import aiter
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.opus.pa_mqa_logits_mxfp4 import (  # BLOCK_K / Q_PER_BLOCK are internal, and
    BLOCK_K,  # reached for on purpose: only a test sizes block_tables or checks a tile's rows
    Q_PER_BLOCK,
    pa_mqa_logits_mxfp4,
    pa_mqa_logits_mxfp4_plan,
)
from aiter.test_common import benchmark, checkAllclose, run_perftest

dev = "cuda"

SUPPORTED_GFX = ["gfx1250"]  # the kernel is gfx1250-only; the wrapper enforces it too

HEADS = 64
HEAD_DIM = 128
KV_BLOCK_SIZE = 64  # page size
SCALE_BLOCK = 32  # E8M0 block
WEIGHT_SCALE = 1.5
BLOCKS_ROW = HEAD_DIM // SCALE_BLOCK  # 4 natural E8M0 blocks per row

CSA_RATIO = 4  # ATOM's compression ratio: row n sees floor((pos + 1) / 4)

# Window ends around the KV_TILE = 128 boundary, plus the 1-tile and 2-tile pipeline corners --
# which is where the accumulator ping-pong's peeled first phase and its epilogue run ALONE,
# rather than as the steady loop's two halves.
TILE_EDGE_ENDS = (1, 63, 64, 65, 127, 128, 129, 191, 255, 256, 257, 383, 384, 385)
PREFILL_TOTAL_QLEN = 16384
PREFILL_QMIN = 800
N_COS_SAMPLE = 8

# Not a command-line knob on purpose: readings taken at different iteration counts are not
# comparable on this kernel, so the budget is pinned here.
PERF_ITERS = 50
PERF_WARMUP = 10

# Per-row / per-page byte counts for the traffic denominator. ALL NATURAL on this target, so
# each is just the product -- no permutation and no padding, unlike the gfx950 sibling.
Q_ROW_BYTES = HEADS * HEAD_DIM // 2  # 4096: one packed fp4 query row
QS_ROW_BYTES = HEADS * BLOCKS_ROW  # 256: its E8M0 scales, [H, 4]
W_ROW_BYTES = HEADS * 2  # 128: bf16 per-head weights
KV_PAGE_BYTES = KV_BLOCK_SIZE * HEAD_DIM // 2  # 4096: one packed fp4 page
KVS_PAGE_BYTES = KV_BLOCK_SIZE * BLOCKS_ROW  # 256: its E8M0 scales, [PAGE, 4]

FP4_E2M1_MAX = 6.0
_FP4_GRID_VALUES = [
    -6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0,
    0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
]  # fmt: skip
_E2M1_LUT = [0xF, 0xE, 0xD, 0xC, 0xB, 0xA, 0x9, 0x0, 0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7]
_E2M1_INV_LUT = [7, 8, 9, 10, 11, 12, 13, 14, 7, 6, 5, 4, 3, 2, 1, 0]

# The magnitude spread, in EXPONENTS of two. Wide enough that a misrouted scale is a clean
# factor of two on that column and narrow enough to stay far from E8M0's ends.
MAG_SPREAD_LO, MAG_SPREAD_HI = -3, 3


# ── MXFP4 quant / dequant ─────────────────────────────────────────────────────
def fp4_quant(x, block_size=SCALE_BLOCK):
    """[..., d] float -> (packed nibbles [..., d/2] uint8, e8m0 [..., d/block] uint8).

    Low nibble = even element, matching the kernel."""
    *prefix, d = x.shape
    assert d % block_size == 0
    x_blk = x.float().reshape(*prefix, d // block_size, block_size)
    amax = x_blk.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    exp_biased = (
        (torch.ceil(torch.log2(amax / FP4_E2M1_MAX)) + 127.0)
        .clamp(0.0, 255.0)
        .to(torch.uint8)
    )
    e8m0 = exp_biased.squeeze(-1).contiguous()
    x_scaled = x_blk / torch.pow(2.0, exp_biased.float() - 127.0)
    grid = torch.tensor(_FP4_GRID_VALUES, dtype=torch.float32, device=x.device)
    idx = (x_scaled.unsqueeze(-1) - grid).abs().argmin(dim=-1)
    lut = torch.tensor(_E2M1_LUT, dtype=torch.uint8, device=x.device)
    nibbles = lut[idx].reshape(*prefix, d)
    packed = (nibbles[..., 0::2] | (nibbles[..., 1::2] << 4)).to(torch.uint8)
    return packed.contiguous(), e8m0


def fp4_dequant(packed, e8m0, block_size=SCALE_BLOCK):
    *prefix, d_half = packed.shape
    d = d_half * 2
    nibbles = torch.empty(*prefix, d, dtype=torch.uint8, device=packed.device)
    nibbles[..., 0::2] = packed & 0xF
    nibbles[..., 1::2] = (packed >> 4) & 0xF
    inv = torch.tensor(_E2M1_INV_LUT, dtype=torch.long, device=packed.device)
    grid = torch.tensor(_FP4_GRID_VALUES, dtype=torch.float32, device=packed.device)
    vals = grid[inv[nibbles.long()]]
    scale = torch.pow(2.0, e8m0.float() - 127.0)
    return (
        vals.reshape(*prefix, d // block_size, block_size) * scale.unsqueeze(-1)
    ).reshape(*prefix, d)


def randn_spread(rows, g):
    """``[rows, HEAD_DIM]`` normal values, each ROW scaled by its own power of two.

    A PER-ROW multiplier, not per-element: that is what moves the block's ``amax`` and hence its
    E8M0, while leaving the quantized nibbles distributed as they would be otherwise. See the
    module docstring for why the suite is blind without it."""
    x = torch.randn(rows, HEAD_DIM, generator=g, device=dev, dtype=torch.float32)
    e = torch.randint(
        MAG_SPREAD_LO, MAG_SPREAD_HI + 1, (rows, 1), generator=g, device=dev
    )
    return x * torch.pow(2.0, e.float())


# ── input builders: every buffer NATURAL ──────────────────────────────────────
@dataclass
class Inputs:
    q_packed: torch.Tensor  # [T, H, D/2]            natural
    q_scale: torch.Tensor  # [T, H, 4]               natural
    q_dq: torch.Tensor  # [T, H, D]  dequantized, for the reference
    weights: torch.Tensor  # [T, H] bf16             natural
    kv_cache: torch.Tensor  # [num_blocks, PAGE, D/2] natural
    kv_scale: torch.Tensor  # [num_blocks, PAGE, 4]   natural
    kv_dq: torch.Tensor  # [bs, t_max, D] dequantized
    block_tables: torch.Tensor
    max_seq_len: int


def pages_for(max_end):
    """Pages per sequence, rounded so a CTA never indexes past the table.

    Rounded to a whole KV TILE and not to a page: a CTA covers its window in whole tiles and
    reads ``block_tables`` at every page of the last one, even where the window stops inside
    it.
    """
    tiles = max(1, (max_end + BLOCK_K - 1) // BLOCK_K)
    return tiles * (BLOCK_K // KV_BLOCK_SIZE)


def build_inputs(bs, max_end, total_tokens, seed):
    g = torch.Generator(device=dev).manual_seed(seed)
    mbps = pages_for(max_end)
    t_max = mbps * KV_BLOCK_SIZE
    num_blocks = bs * mbps

    # --- KV. The two "layouts" are reshapes: natural is what the kernel reads. ---
    kv = randn_spread(bs * t_max, g)
    kv_packed, kv_e8 = fp4_quant(kv)
    kv_dq = fp4_dequant(kv_packed, kv_e8).reshape(bs, t_max, HEAD_DIM)
    kv_cache = kv_packed.reshape(num_blocks, KV_BLOCK_SIZE, HEAD_DIM // 2).contiguous()
    kv_scale = kv_e8.reshape(num_blocks, KV_BLOCK_SIZE, BLOCKS_ROW).contiguous()
    block_tables = torch.arange(num_blocks, dtype=torch.int32, device=dev).reshape(
        bs, mbps
    )

    # --- Q + weights ---
    q = randn_spread(total_tokens * HEADS, g)
    q_packed_flat, q_e8 = fp4_quant(q)
    q_dq = fp4_dequant(q_packed_flat, q_e8).reshape(total_tokens, HEADS, HEAD_DIM)
    q_packed = q_packed_flat.reshape(total_tokens, HEADS, HEAD_DIM // 2).contiguous()
    q_scale = q_e8.reshape(total_tokens, HEADS, BLOCKS_ROW).contiguous()
    weights = torch.randn(
        total_tokens, HEADS, generator=g, device=dev, dtype=torch.float32
    ).to(torch.bfloat16)

    return Inputs(
        q_packed=q_packed,
        q_scale=q_scale,
        q_dq=q_dq,
        weights=weights,
        kv_cache=kv_cache,
        kv_scale=kv_scale,
        kv_dq=kv_dq,
        block_tables=block_tables,
        max_seq_len=t_max,
    )


# ── reference ─────────────────────────────────────────────────────────────────
def ref_rows(inp, rows, rb, ls, le):
    """Reference logits for a few sampled rows: {row: (start, end, [values])}."""
    w = inp.weights.float()
    ref = {}
    for r in rows:
        b, s, e = int(rb[r]), int(ls[r]), int(le[r])
        if e <= s:
            ref[r] = (s, e, None)
            continue
        k = inp.kv_dq[b, s:e]  # [n, D]
        scores = torch.relu(inp.q_dq[r] @ k.T)  # [H, n]
        ref[r] = (s, e, (scores * w[r, :, None]).sum(0) * WEIGHT_SCALE)
    return ref


def check_rows(out, ref, msg):
    """``checkAllclose`` over the in-window cells of the sampled rows, concatenated into one
    flat pair because every row has a different window.

    The bound is relative: a logit is a signed sum over 64 heads of a 128-long dot product, so
    values run to ~1e4 and ``atol`` covers only the cells the weights cancel to near zero.
    """
    got, want = [], []
    for r, (s, e, vals) in ref.items():
        if vals is None:
            continue
        got.append(out[r, s:e].float())
        want.append(vals.float())
    if not got:
        return 0.0
    return checkAllclose(
        torch.cat(want), torch.cat(got), rtol=2e-5, atol=1e-2, msg=msg, printLog=False
    )


def roofline_bytes(rb, le, total_q, n_logits):
    """Bytes a launch must move at least once -- NOT one K vector per output logit.

    The per-logit count is meaningless here: every query row of a batch scores against the same
    KV pages, so on a long-context shape it implies a ~14000x reuse factor and reports a figure
    above the card's HBM peak. Counting each page once makes this a lower bound on real
    traffic, which is a reading that can be held against a hardware limit."""
    if rb.numel() == 0:
        return 0
    ends = torch.zeros(int(rb.max().item()) + 1, dtype=torch.int64, device=rb.device)
    ends.scatter_reduce_(0, rb.long(), le.long().clamp(min=0), reduce="amax")
    pages = int(((ends + KV_BLOCK_SIZE - 1) // KV_BLOCK_SIZE).sum().item())
    return (
        total_q * (Q_ROW_BYTES + QS_ROW_BYTES + W_ROW_BYTES)
        + pages * (KV_PAGE_BYTES + KVS_PAGE_BYTES)
        + n_logits * 4
    )


def oob_is_neginf(out, ls, le):
    """Every cell outside ``[local_start, local_end)`` must be left at the -inf pre-fill.

    Not redundant with ``check_rows``: this is what caught a store path writing ``own_start``
    columns BELOW every window while the cosine stayed clean."""
    col = torch.arange(out.shape[1], device=out.device).unsqueeze(0)
    inside = (col >= ls.unsqueeze(1)) & (col < le.unsqueeze(1))
    return bool(torch.isneginf(out[~inside]).all().item())


def window_is_written(out, ls, le):
    """Every cell inside ``[local_start, local_end)`` must have been stored to.

    ``check_rows`` sees a dropped token too, but only on the rows it sampled. This scans every
    row, which is what makes it the check that catches a ``num_groups`` short of the tail.
    """
    col = torch.arange(out.shape[1], device=out.device).unsqueeze(0)
    inside = (col >= ls.unsqueeze(1)) & (col < le.unsqueeze(1))
    return bool(torch.isfinite(out[inside]).all().item())


def sample_rows(total, le, n=N_COS_SAMPLE, seed=0):
    nonempty = torch.nonzero(le > 0).flatten().tolist()
    if not nonempty:
        return []
    rng = random.Random(seed)
    return sorted(rng.sample(nonempty, min(n, len(nonempty))))


# ── correctness ───────────────────────────────────────────────────────────────
def assert_qshare_windows(cu_tiles, num_tiles, local_starts, local_ends):
    """Check the one condition the schedule takes on faith: within a tile the window rule is
    NON-DECREASING, so the union is the first row's start and the LAST row's end -- two loads
    instead of a reduction. A violation makes the builder compute a union that is not one, so
    rows whose window reaches past it are scored short.

    Host-side and synchronising, which is why it lives here and not in the op: it is worth three
    device-to-host copies in a correctness sweep and nothing in a hot path.

    "A tile is contiguous rows of one batch" is not checked, because the tile cut inside
    ``pa_mqa_logits_mxfp4_plan`` is what produces the array and guarantees it.
    """
    ct = cu_tiles[: num_tiles + 1].tolist()
    ls = local_starts.tolist()
    le = local_ends.tolist()
    for t in range(num_tiles):
        lo, hi = ct[t], ct[t + 1]
        if hi == lo:
            continue  # an empty tile; its CTA gets a zero-count record
        if not (0 < hi - lo <= Q_PER_BLOCK):
            raise AssertionError(
                f"qshare: tile {t} spans rows [{lo},{hi}), which is not 1..{Q_PER_BLOCK} rows"
            )
        for r in range(lo + 1, hi):
            if ls[r] < ls[r - 1] or le[r] < le[r - 1]:
                raise AssertionError(
                    f"qshare: tile {t} window is not non-decreasing (row {r - 1}: "
                    f"[{ls[r - 1]},{le[r - 1]}) then row {r}: [{ls[r]},{le[r]}))"
                )


def run_one(inp, qlens, rb, ls, le, label, seed, check_windows=True):
    """Launch one case over explicit per-row windows and score it."""
    total_q = int(rb.numel())
    cu = torch.tensor(
        [0] + list(itertools.accumulate(qlens)), dtype=torch.int32, device=dev
    )
    # `local_starts` is passed because these cases carry non-zero window starts; ATOM's paths do
    # not and leave it None. `row_to_batch` is passed because `block_tables` here is per
    # SEQUENCE -- leaving it None would make the kernel read the table by query row.
    plan = pa_mqa_logits_mxfp4_plan(
        cu, le, total_q=total_q, local_starts=ls, row_to_batch=rb
    )
    if check_windows:
        # The condition the kernel cannot check. Host-side and synchronising, so it runs in the
        # correctness path only -- and it is worth running, because breaking it DEADLOCKS the
        # CTA rather than returning a wrong answer.
        assert_qshare_windows(plan.cu_tiles, plan.num_tiles, ls, le)

    out = pa_mqa_logits_mxfp4(
        inp.q_packed, inp.q_scale, inp.kv_cache, inp.kv_scale, inp.block_tables,
        inp.weights, plan, inp.max_seq_len,
        weight_scale=WEIGHT_SCALE, kv_block_size=KV_BLOCK_SIZE,
    )  # fmt: skip
    torch.cuda.synchronize()

    rows = sample_rows(total_q, le, seed=seed)
    err = check_rows(out, ref_rows(inp, rows, rb, ls, le), f"{label}")
    oob = oob_is_neginf(out, ls, le)
    wr = window_is_written(out, ls, le)
    return {
        "case": label, "rows": total_q, "tiles": plan.num_tiles, "ctas": plan.num_ctas,
        "max_win": int(le.max()), "err": err, "oob -inf": oob,
        "window written": wr, "pass": err == 0 and oob and wr,
    }  # fmt: skip


def check_prefill(windows_per_batch, seed, label):
    """One ragged-prefill case from explicit per-row ``(start, end)`` windows, random data."""
    qlens = [len(w) for w in windows_per_batch]
    total_q = sum(qlens)
    max_end = max(e for w in windows_per_batch for (_, e) in w)
    inp = build_inputs(len(qlens), max_end, total_q, seed)

    rb, ls, le = [], [], []
    for b, w in enumerate(windows_per_batch):
        for s, e in w:
            rb.append(b)
            ls.append(s)
            le.append(e)
    rb = torch.tensor(rb, dtype=torch.int32, device=dev)
    ls = torch.tensor(ls, dtype=torch.int32, device=dev)
    le = torch.tensor(le, dtype=torch.int32, device=dev)

    ret = run_one(inp, qlens, rb, ls, le, label, seed)
    del inp
    torch.cuda.empty_cache()
    return ret


def _g4(windows_per_batch):
    """Replicate each row Q_PER_BLOCK times so a group's rows share a window exactly -- the
    EASY regime. The CSA rules below are the hard one, where adjacent rows of a group differ by
    a column and the loop bound has to be their union."""
    return [[r for r in b for _ in range(Q_PER_BLOCK)] for b in windows_per_batch]


def _csa_fresh(qlen):
    """Fresh sequence: row n sees ``(n + 1) // RATIO``; runs start at n = 3 mod 4."""
    return [(0, (n + 1) // CSA_RATIO) for n in range(qlen)]


def _csa_chunked(qlen, kvlen):
    """``kvlen`` compressed rows committed and this chunk is the tail: row n sees
    ``kvlen - (qlen - 1 - n) // RATIO``, so runs align to the tail rather than to n."""
    return [(0, kvlen - (qlen - 1 - n) // CSA_RATIO) for n in range(qlen)]


def run_corner():
    """The cases the qshare contract is made of: short groups at every residue mod
    Q_PER_BLOCK, windows that do not start at 0, the KV_TILE = 128 boundary neighbourhood, every
    window start mod 128, and both ATOM CSA regimes where a group's rows differ by a column.
    """
    cases = [
        (_g4([[(0, 50), (0, 120), (0, 200)], [(0, 40), (0, 100)]]), 0, "ragged/2b"),
        (_g4([[(0, 30)], [(0, 200)], [(0, 100), (0, 150)]]), 2, "ragged/3b"),
        (_g4([[(10, 50), (64, 200)], [(0, 100), (130, 256)]]), 4, "offset starts"),
        (_g4([[(0, 2048)], [(0, 4096)]]), 8, "long/2b"),
        (_g4([[(0, 512), (0, 1024), (0, 1536)], [(0, 2000)]]), 10, "mixed long"),
        (_g4([[(100, 2048), (512, 4096)], [(0, 8192)]]), 12, "offset long"),
        (_g4([[(0, 1), (17, 33)], [(63, 65), (255, 257)]]), 34, "tiny windows"),
        (_g4([[(0, e) for e in TILE_EDGE_ENDS]]), 40, "tile edges"),
        (_g4([[(s, s + 96) for s in range(130)]]), 52, "start sweep mod 128"),
        ([_csa_fresh(10), _csa_fresh(37), _csa_fresh(64)], 60, "csa fresh"),
        (
            [_csa_chunked(10, 200), _csa_chunked(37, 71), _csa_chunked(63, 1000)],
            62,
            "csa chunked",
        ),
        (
            [_csa_fresh(1), _csa_fresh(2), _csa_fresh(3), _csa_chunked(2, 129)],
            64,
            "csa short groups",
        ),
        (
            [_csa_chunked(8, 300), _csa_chunked(3, 300), _csa_fresh(2049)],
            66,
            "csa mixed",
        ),
    ]
    rows = [check_prefill(w, seed, label) for w, seed, label in cases]
    ok = all(r["pass"] for r in rows)
    aiter.logger.info(
        "MXFP4 MQA logits gfx1250, corner cases, random data with a per-row magnitude "
        "spread (markdown):\n%s",
        pd.DataFrame(rows).to_markdown(index=False),
    )
    return ok


def test_scale_spread_has_teeth():
    """The instrument check: assert the test DATA still spreads the E8M0 exponents.

    The suite's ability to see a misrouted scale rests entirely on neighbouring rows carrying
    DIFFERENT exponents. Check the premise rather than trust it -- if ``randn_spread`` ever
    loses its multiplier, this fails here instead of turning the whole suite into decoration.
    """
    g = torch.Generator(device=dev).manual_seed(0)
    _, e8 = fp4_quant(randn_spread(4096, g))
    distinct = int(torch.unique(e8).numel())
    # 7 exponents of spread, minus the odd block whose amax rounds into a neighbour.
    want = MAG_SPREAD_HI - MAG_SPREAD_LO + 1
    ok = distinct >= want
    aiter.logger.info(
        "scale spread: %d distinct E8M0 exponents over 4096 rows (want >= %d) -- %s",
        distinct,
        want,
        "ok" if ok else "FAIL: the suite can no longer see a misrouted scale",
    )
    # The misroute distance is 16 tokens (b_scale_sel picks a lane half), so also require that
    # rows 16 apart disagree often. A spread that happened to be periodic in 16 would be as
    # blind as no spread at all -- which is exactly how DIAG=4's first version passed a broken
    # kernel at cos 1.000000.
    per_row = e8.reshape(4096, BLOCKS_ROW)[:, 0]
    disagree = float((per_row[:-16] != per_row[16:]).float().mean().item())
    ok = ok and disagree > 0.5
    aiter.logger.info(
        "  rows 16 apart disagree on their exponent %.1f%% of the time (want > 50%%) -- %s",
        100.0 * disagree,
        "ok" if disagree > 0.5 else "FAIL: the spread is blind to a lane-half misroute",
    )
    return ok


# ── perf ──────────────────────────────────────────────────────────────────────
def gen_prefill_qlens(bs, total=PREFILL_TOTAL_QLEN, qmin=PREFILL_QMIN, seed=0):
    g = random.Random(seed)
    extra = total - bs * qmin
    w = [g.random() for _ in range(bs)]
    s = sum(w) or 1.0
    parts = [qmin + int(extra * wi / s) for wi in w]
    parts[0] += total - sum(parts)
    return parts


def tail_causal_windows(qlens, ctxs):
    """The MTP tail-causal windows, in packed (b, n) row order.

    Batch ``b``'s ``n``-th row sees ``[0, ctx[b] - (qlen[b] - 1 - n))``, plain causal when
    ``qlen == ctx``. The harness's rule, not the kernel's: the schedule takes whatever windows
    it is handed, and a compressed cache's ``floor((pos+1)/R)`` is not expressible here.
    """
    rb, ls, le = [], [], []
    for b, (q, c) in enumerate(zip(qlens, ctxs)):
        for n in range(q):
            rb.append(b)
            ls.append(0)
            le.append(max(c - (q - 1 - n), 0))

    def t(v):
        return torch.tensor(v, dtype=torch.int32, device=dev)

    return t(rb), t(ls), t(le)


def score(fn, inp, rb, ls, le, total_q, n_logits, seed):
    """Time the launch, then score it against the sampled-row reference.

    Scoring runs AFTER the timed region and frees its temporaries: the reference materializes a
    ``[heads, window]`` score matrix per sampled row -- ~1 GB on the longest shape -- and the
    caching allocator charges that churn to whatever is timed next, worth 4-9% here. An
    agreement check in FRONT of a timed region is the same mistake."""
    flops = 2 * HEADS * HEAD_DIM * n_logits
    nbytes = roofline_bytes(rb, le, total_q, n_logits)
    out, us = run_perftest(fn, num_iters=PERF_ITERS, num_warmup=PERF_WARMUP)
    ref = ref_rows(inp, sample_rows(total_q, le, seed=seed), rb, ls, le)
    ret = {
        "us": round(us, 2),
        "TFLOPS": round(flops / us / 1e6, 1),
        "GB per s": round(nbytes / us / 1e3, 1),
        "err": check_rows(out, ref, "perf"),
    }
    del ref, out
    torch.cuda.empty_cache()
    return ret


@benchmark()
def test_prefill_causal(bs):
    """One causal prefill shape: 16384 query rows split across ``bs`` batches, ctx == qlen."""
    qlens = gen_prefill_qlens(bs, seed=bs)
    total_q = sum(qlens)
    inp = build_inputs(bs, max(qlens), total_q, seed=bs)
    cu = torch.tensor(
        [0] + list(itertools.accumulate(qlens)), dtype=torch.int32, device=dev
    )
    rb, ls, le = tail_causal_windows(qlens, qlens)
    # Per FORWARD against a per-layer kernel, so built OUTSIDE the timed region: `run_perftest`
    # sums every CUDA event in it, so a metadata kernel left inside lands in the reported time.
    plan = pa_mqa_logits_mxfp4_plan(
        cu, le, total_q=total_q, local_starts=ls, row_to_batch=rb
    )
    out = torch.full(
        (total_q, inp.max_seq_len), float("-inf"), dtype=torch.float32, device=dev
    )

    # Bound as DEFAULTS, not captured: this is rebuilt per shape over a name the sweep reuses,
    # so late binding would read the next shape's buffers. The plan is passed IN because
    # `run_perftest` sums every CUDA event in the region, so a builder left inside the timed
    # call lands in the reported time.
    def ours(inp=inp, plan=plan, out=out):
        return pa_mqa_logits_mxfp4(
            inp.q_packed, inp.q_scale, inp.kv_cache, inp.kv_scale, inp.block_tables,
            inp.weights, plan, inp.max_seq_len,
            weight_scale=WEIGHT_SCALE, kv_block_size=KV_BLOCK_SIZE, out=out,
        )  # fmt: skip

    n_logits = int((le - ls).clamp(min=0).sum().item())
    ret = {
        "gfx": get_gfx(),
        "total_q": total_q,
        "tiles": plan.num_tiles,
        "ctas": plan.num_ctas,
        "max_win": int(le.max()),
        "n_logits": n_logits,
    }
    ret.update(score(ours, inp, rb, ls, le, total_q, n_logits, seed=bs))
    del inp, out
    torch.cuda.empty_cache()
    return ret


@benchmark()
def test_prefill_csa(regime, bs, qlen, kvlen):
    """ATOM's two prefill regimes at CSA ratio 4, where a group's rows differ by a column.

    The window rule is an INPUT, never derived: ``tail_causal_windows`` is the other shape this
    harness builds and it cannot express this one, since
    ``floor((x - d) / R) != floor(x / R) - d``."""
    make = _csa_fresh if regime == "fresh" else (lambda q: _csa_chunked(q, kvlen))
    per_batch = [make(qlen) for _ in range(bs)]
    qlens = [len(w) for w in per_batch]
    total_q = sum(qlens)
    max_end = max(e for w in per_batch for (_, e) in w)
    inp = build_inputs(bs, max_end, total_q, seed=bs + qlen)

    rb = torch.tensor(
        [b for b, w in enumerate(per_batch) for _ in w], dtype=torch.int32, device=dev
    )
    ls = torch.tensor(
        [s for w in per_batch for (s, _) in w], dtype=torch.int32, device=dev
    )
    le = torch.tensor(
        [e for w in per_batch for (_, e) in w], dtype=torch.int32, device=dev
    )
    cu = torch.tensor(
        [0] + list(itertools.accumulate(qlens)), dtype=torch.int32, device=dev
    )
    # Per FORWARD against a per-layer kernel, so built OUTSIDE the timed region: `run_perftest`
    # sums every CUDA event in it, so a metadata kernel left inside lands in the reported time.
    plan = pa_mqa_logits_mxfp4_plan(
        cu, le, total_q=total_q, local_starts=ls, row_to_batch=rb
    )
    out = torch.full(
        (total_q, inp.max_seq_len), float("-inf"), dtype=torch.float32, device=dev
    )

    def ours(inp=inp, plan=plan, out=out):
        return pa_mqa_logits_mxfp4(
            inp.q_packed, inp.q_scale, inp.kv_cache, inp.kv_scale, inp.block_tables,
            inp.weights, plan, inp.max_seq_len,
            weight_scale=WEIGHT_SCALE, kv_block_size=KV_BLOCK_SIZE, out=out,
        )  # fmt: skip

    n_logits = int((le - ls).clamp(min=0).sum().item())
    ret = {
        "gfx": get_gfx(),
        "regime": regime,
        "bs": bs,
        "qlen": qlen,
        "tiles": plan.num_tiles,
        "ctas": plan.num_ctas,
        "max_win": int(le.max()),
        "n_logits": n_logits,
    }
    ret.update(score(ours, inp, rb, ls, le, total_q, n_logits, seed=bs + qlen))
    del inp, out
    torch.cuda.empty_cache()
    return ret


def main():
    # Whole-op arch gate, here rather than inside the @benchmark fns: CI discovers every
    # op_tests/test_*.py and runs it on the other shards too, where the wrapper's own gfx1250
    # check would raise and fail the shard. Positive allow-list, so an unknown new card skips.
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning(
            "this MXFP4 MQA-logits kernel is gfx1250-only; skipping on %s", get_gfx()
        )
        return

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="config input of test",
    )
    parser.add_argument(
        "-b", "--batch", type=int, nargs="*", default=[1, 2, 4, 8, 16],
        help="causal prefill batch sizes; total_q is fixed at 16384 and split across them",
    )  # fmt: skip
    parser.add_argument(
        "--skip-corner", action="store_true", help="perf only, no correctness sweep"
    )
    args = parser.parse_args()

    ok = True
    if not args.skip_corner:
        ok = test_scale_spread_has_teeth()
        ok = run_corner() and ok

    rows = [test_prefill_causal(bs) for bs in args.batch]
    aiter.logger.info(
        "MXFP4 MQA logits gfx1250 prefill, causal (ctx == qlen), RANDOM data (markdown):\n%s",
        pd.DataFrame(rows).to_markdown(index=False),
    )

    csa_shapes = [
        ("fresh", 1, 16384, 0),
        ("fresh", 2, 8192, 0),
        ("fresh", 4, 4096, 0),
        ("chunked", 1, 4096, 4096),
        ("chunked", 1, 8192, 12288),
        ("chunked", 1, 16384, 12288),
        ("chunked", 2, 8192, 25000),
    ]
    rows = [test_prefill_csa(*s) for s in csa_shapes]
    aiter.logger.info(
        "MXFP4 MQA logits gfx1250 prefill, CSA ratio %d, RANDOM data (markdown):\n%s",
        CSA_RATIO,
        pd.DataFrame(rows).to_markdown(index=False),
    )

    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
