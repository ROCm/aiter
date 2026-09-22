# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""MXFP4 paged MQA logits (OPUS) -- correctness and perf, through the arch dispatcher.

Emits five tables -- the corner cases, a causal prefill sweep, ATOM's two CSA-compressed
prefill regimes (fresh, and chunked at PR #5332's shapes so the two PRs are comparable), and
an MTP decode sweep -- as markdown for a reader and as one-line JSON records for a benchmark
driver.

Every launch goes through ``aiter.ops.opus.pa_mqa_logits_mxfp4``, the ONE module both arches
build into, so the path under test is the path a caller gets, layout check included. This file
runs on BOTH targets: it builds the NATURAL (3-D) scale/cache layouts on gfx1250 and the
MFMA-permuted (4-D) ones on gfx950 (``_is_permuted`` / ``_scale_to_opus`` below), while the
dequantized reference reads the natural E8M0 either way. The kernel INSTANCES differ per arch --
gfx1250 packs up to four query rows per tile (``qlen4_kv64``, ``qlen1_kv64``), gfx950 runs one
and adds a wider 256-token KV tile (``qlen1_kv64``, ``qlen1_kv256``) -- and ``_variants()``
returns the right set. The two tile-cut probes
(``assert_qshare_windows``, ``check_row_id_bound``) are gfx1250-only and gate on ``q_per_block``.

    python3 op_tests/test_pa_mqa_logits_mxfp4_opus.py             # the full default sweep
    python3 op_tests/test_pa_mqa_logits_mxfp4_opus.py -b 1 2      # a quick subset
    python3 op_tests/test_pa_mqa_logits_mxfp4_opus.py \\
        --data-init constant uniform --scale-init constant auto   # two paired init regimes

DATA AND SCALE ARE TWO INDEPENDENT AXES, not "sample the data then derive the scale by
quantizing it". The reference dequantizes whatever ``(nibbles, E8M0)`` pair comes out, which is
defined for any pair, so nothing downstream needs the two to be consistent -- and the exponents
are then a property of the SCALE axis alone.

THAT MATTERS BECAUSE THE SPREAD IS LOAD-BEARING. Where neighbouring KV tokens carry the same
exponent, a scale routed to the WRONG token reads one that happens to be right: a deliberately
misrouted ``b_scale_sel`` passed the standalone suite at ``cos = 0.999952`` on flat-magnitude
data and only failed at 0.396 once the exponents spread. ``b_scale_sel`` misroutes across a
lane HALF, so the number that matters is how often rows 16 apart disagree -- a spread periodic
in 16 would be as blind as none. ``scale_spread`` measures both, and ``correctness_blindness``
SKIPS the correctness sweep with the reason in the table for a pair that cannot judge, because
a probe that cannot fail gets quoted as evidence.

There is no second implementation to cross-check against on this target -- gfx1250 has no
FlyDSL fp4 MQA-logits kernel -- so the dequantized fp32 reference is the only judge. It runs
over the same quantized values the kernel sees, so ``err`` is expected at ~1e-6 (fp32
accumulation order), not at fp4 resolution. The K-permutation and scale-routing probes a
reference cannot give live in the opus-ops standalone harness.
"""

import argparse
import functools
import itertools
import random
from dataclasses import dataclass

import pandas as pd
import torch

import aiter
from aiter.benchmark_data_init import (
    DATA_DISTS,
    E8M0_SCALE_DISTS,
    fill,
    fill_fp4,
    fill_scale_e8m0,
    make_generator,
)
from aiter.benchmark_reporting import print_json_table
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.opus.pa_mqa_logits_mxfp4 import (
    pa_mqa_logits_mxfp4,
    pa_mqa_logits_mxfp4_block_table_width,
    pa_mqa_logits_mxfp4_plan,
    pa_mqa_logits_mxfp4_plan_buffers,
    pa_mqa_logits_mxfp4_variants,
)
from aiter.test_common import benchmark, checkAllclose, run_perftest
from aiter.utility.fp4_utils import e8m0_to_f32, mxfp4_to_f32

dev = "cuda"

# BOTH targets of the one module. build_inputs picks the layout per arch (`_is_permuted`); the
# dispatcher checks it. The perf/correctness code never names an arch -- only the layout builder
# and the two gfx1250-only tile-cut probes do, and those gate on `q_per_block`.
SUPPORTED_GFX = ["gfx1250", "gfx950"]

HEADS = 64
HEAD_DIM = 128
KV_BLOCK_SIZE = 64  # page size
SCALE_BLOCK = 32  # E8M0 block
WEIGHT_SCALE = 1.5
BLOCKS_ROW = HEAD_DIM // SCALE_BLOCK  # 4 natural E8M0 blocks per row

# gfx950 MFMA-permuted scale geometry, used only when `_is_permuted()`. A lane's 4 E8M0 bytes
# land in one aligned dword indexed [.., g(K_CHUNKS), m(MFMA_N), byte]; the shuffle is a pure
# permutation of the natural array, so the reference (which reads the natural E8M0) is unmoved.
MFMA_N = 32
K_TILES = HEAD_DIM // 64  # 2  (MFMA_K = 64)
K_CHUNKS = 64 // SCALE_BLOCK  # 2  (32-K chunks per k-tile)
SCALE_BYTES = 4  # K_TILES * n_tiles per lane dword

CSA_RATIO = 4  # ATOM's compression ratio: row n sees floor((pos + 1) / 4)

# Decode: a fixed 32 concurrent sequences, and the windows below are COMPRESSED column counts
# rather than raw context -- the CSA divide is already applied. A long row scans 25000 columns,
# which at ratio 4 stands for roughly 100k raw tokens; a short row scans 100.
DECODE_SEQS = 32
DECODE_WIN_LONG = 25000
DECODE_WIN_SHORT = 100

# Window ends around the KV_TILE = 64 boundary, plus the 1-tile and 2-tile pipeline corners --
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

# `fill_fp4`'s `constant` fills the packed BYTE and its own default is 0, which would make
# `--data-init constant` the same vacuous all-zero buffer as `zero`. 0x22 is both nibbles at
# 0x2 = 1.0, so with constant scales and constant weights every in-window cell lands on exactly
# HEADS * HEAD_DIM * WEIGHT_SCALE = 12288 -- the standalone harness's DIAG=2 probe.
FP4_CONSTANT_BYTE = 0x22

# How often rows 16 apart must carry DIFFERENT exponents for a misrouted `b_scale_sel` to be
# visible at all. Not a tolerance -- below this the correctness sweep does not run.
MISROUTE_DISAGREE_MIN = 0.2

# The fp32 reassociation bound, as a fraction of the checked cells' own magnitude RANGE.
#
# A logit is a signed sum over 64 heads, so a cell the weights cancel to near zero still
# carries the rounding of terms as large as the row's biggest logit: the error scales with the
# TERMS and not with the result. A constant `atol` cannot express that, and the 1e-2 this
# replaces was written for values running to ~1e4 -- it went silently too tight once the scale
# axis moved to the library's pow2_binomial spread and the range reached ~4.5e5.
#
# Measured on the 25000-column decode shape, which is where the tail shows up: 4 cells in
# 200000 land outside a constant 1e-2, every one of them cancelled to |ref| 204..1210 inside a
# row whose largest logit is 451775, worst delta 0.109 -- 2.4e-7 of the range. Identical to the
# last bit with the chunk split forced off, so this is the reference and the kernel summing the
# same 64 terms in different orders, not a kernel result.
#
# 1e-6 is that with margin and still ~1000x tighter than the bug class this suite exists to
# catch: a misrouted scale moves a cell by a FACTOR OF TWO, not by 0.09% of the median cell.
REASSOC_ATOL_REL = 1e-6


# ── the two init axes, and the dequant that joins them ────────────────────────
def fill_nibbles(rows, data_init, gen):
    """``[rows, HEAD_DIM/2]`` packed e2m1, low nibble = even element."""
    return fill_fp4(
        (rows, HEAD_DIM), data_init, gen, device=dev, constant=FP4_CONSTANT_BYTE
    )


def fill_exponents(shape, scale_init, gen):
    """E8M0 on-wire bytes at the library's own pow2 spread.

    ``benchmark_data_init`` offers a narrower ``n`` for a reference that cannot follow the
    default's 2^-11..2^10, and narrowing is the wrong lever here: it would cost teeth (n=10
    spreads 18 exponents with rows 16 apart disagreeing 88.8%, against 8 and 78.0% at n=3) to
    buy back the fp32 headroom ``REASSOC_ATOL_REL`` already accounts for. The default stays.
    """
    return fill_scale_e8m0(shape, scale_init, gen, device=dev)


def fp4_dequant(packed, e8m0, block_size=SCALE_BLOCK):
    """``[..., d/2]`` packed e2m1 + ``[..., d/block]`` E8M0 -> ``[..., d]`` fp32.

    Defined for ANY pair, which is what lets the two init axes stay independent: nothing here
    assumes the exponents were derived by quantizing these nibbles.

    BOTH decodes come from ``aiter.utility.fp4_utils`` rather than from arithmetic here, so the
    reference cannot drift from what ``fill_fp4`` / ``fill_scale_e8m0`` write. That matters at
    the two E8M0 encodings a plain ``2^(byte - 127)`` gets wrong: ``0xFF`` is the NaN sentinel,
    which the kernel propagates through the relu, where the power would give +inf. Neither
    generator emits it today -- ``pow2_binomial`` spans bytes 116..137 -- so this is about the
    reference staying canonical rather than about a case the suite currently reaches.
    """
    *prefix, d_half = packed.shape
    d = d_half * 2
    vals = mxfp4_to_f32(packed).reshape(*prefix, d // block_size, block_size)
    scale = e8m0_to_f32(e8m0)
    return (vals * scale.unsqueeze(-1)).reshape(*prefix, d)


# ── per-arch scale/cache layout ───────────────────────────────────────────────
@functools.cache
def _is_permuted() -> bool:
    """True on gfx950 (MFMA-permuted 4-D scales + 4-chunk kv_cache), False on gfx1250 (natural
    3-D). Lazy for the same reason as ``_variants``: the arch probe inits HIP at import, which
    ``main()``'s gate is meant to run before."""
    return get_gfx() == "gfx950"


def _scale_to_opus(e8_nat, rows_per_group):
    """``[rows, 4]`` natural E8M0 -> ``[rows/rpg, K_CHUNKS, MFMA_N, SCALE_BYTES]``, gfx950's
    layout. ``rows_per_group`` is ``MFMA_N * n_tiles``: 64 heads per query row for q_scale, 64
    page tokens per block for kv_scale. A pure permutation -- no byte dropped or duplicated.
    """
    n_tiles = rows_per_group // MFMA_N
    groups = e8_nat.shape[0] // rows_per_group
    return (
        e8_nat.reshape(groups, n_tiles, MFMA_N, K_TILES, K_CHUNKS)
        .permute(0, 4, 2, 3, 1)  # [group, g, m, kt, tile]
        .reshape(groups, K_CHUNKS, MFMA_N, SCALE_BYTES)
        .contiguous()
    )


# ── input builders: NATURAL on gfx1250, MFMA-permuted on gfx950 ────────────────
@dataclass
class Inputs:
    q_packed: torch.Tensor  # [T, H, D/2]            natural
    q_scale: torch.Tensor  # [T, H, 4] nat / [T, 2, 32, 4] permuted
    q_dq: torch.Tensor  # [T, H, D]  dequantized, for the reference
    weights: torch.Tensor  # [T, H] bf16             natural
    kv_cache: torch.Tensor  # [nb, PAGE, D/2] nat / [nb, 4, PAGE, 16] permuted
    kv_scale: torch.Tensor  # [nb, PAGE, 4] nat / [nb, 2, 32, 4] permuted
    kv_dq: torch.Tensor  # [bs, t_max, D] dequantized
    block_tables: torch.Tensor
    max_seq_len: int


def pages_for(max_end):
    """Pages per sequence, rounded so a CTA never indexes past the table.

    Rounded to a whole KV TILE and not to a page: a CTA covers its window in whole tiles and
    reads ``block_tables`` at every page of the last one, even where the window stops inside
    it.
    """
    # Sized for EVERY compiled variant, not just the one this shape will pick: a test holds one
    # block_tables per case and the plan chooses from the shape. That is what the op's own
    # sizing helper is for, and open-coding the rounding is how it drifts.
    return pa_mqa_logits_mxfp4_block_table_width(
        max(max_end, 1), kv_block_size=KV_BLOCK_SIZE
    )


def build_inputs(bs, max_end, total_tokens, seed, data_init, scale_init):
    """Every buffer from ONE seeded generator, so a ``--seed`` reproduces the case bit for bit.

    The nibbles and the exponents are drawn separately and never reconciled -- see the module
    docstring for why that is the point rather than a shortcut.
    """
    gen = make_generator(seed, device=dev)
    mbps = pages_for(max_end)
    t_max = mbps * KV_BLOCK_SIZE
    num_blocks = bs * mbps

    permuted = _is_permuted()

    # --- KV. The reference always reads the NATURAL (kv_packed, kv_e8); the kernel reads the
    # arch's own scale/cache layout, both pure reshapes/permutations of the same bytes. ---
    kv_packed = fill_nibbles(bs * t_max, data_init, gen)
    kv_e8 = fill_exponents((bs * t_max, BLOCKS_ROW), scale_init, gen)
    kv_dq = fp4_dequant(kv_packed, kv_e8).reshape(bs, t_max, HEAD_DIM)
    if permuted:
        # gfx950: kv_cache[blk, b, o, :] holds K[token o][32b:32b+32]'s 16 packed bytes.
        kv_cache = (
            kv_packed.reshape(
                num_blocks, KV_BLOCK_SIZE, BLOCKS_ROW, HEAD_DIM // 2 // BLOCKS_ROW
            )
            .permute(0, 2, 1, 3)
            .contiguous()
        )
        kv_scale = _scale_to_opus(kv_e8, KV_BLOCK_SIZE)
    else:
        kv_cache = kv_packed.reshape(
            num_blocks, KV_BLOCK_SIZE, HEAD_DIM // 2
        ).contiguous()
        kv_scale = kv_e8.reshape(num_blocks, KV_BLOCK_SIZE, BLOCKS_ROW).contiguous()
    block_tables = torch.arange(num_blocks, dtype=torch.int32, device=dev).reshape(
        bs, mbps
    )

    # --- Q + weights ---
    q_packed_flat = fill_nibbles(total_tokens * HEADS, data_init, gen)
    q_e8 = fill_exponents((total_tokens * HEADS, BLOCKS_ROW), scale_init, gen)
    q_dq = fp4_dequant(q_packed_flat, q_e8).reshape(total_tokens, HEADS, HEAD_DIM)
    q_packed = q_packed_flat.reshape(total_tokens, HEADS, HEAD_DIM // 2).contiguous()
    q_scale = (
        _scale_to_opus(q_e8, HEADS)
        if permuted
        else q_e8.reshape(total_tokens, HEADS, BLOCKS_ROW).contiguous()
    )
    weights = fill(
        (total_tokens, HEADS), data_init, gen, dtype=torch.bfloat16, device=dev
    )

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
    flat pair because every row has a different window. Returns the MISMATCH RATIO.

    ``atol`` is taken from the checked cells' own range rather than fixed, for the reason
    ``REASSOC_ATOL_REL`` gives: the cells that need it are the ones the 64-head sum cancels,
    and what bounds their error is the size of the TERMS, which no constant can know.
    """
    got, want = [], []
    for r, (s, e, vals) in ref.items():
        if vals is None:
            continue
        got.append(out[r, s:e].float())
        want.append(vals.float())
    if not got:
        return 0.0
    want, got = torch.cat(want), torch.cat(got)
    atol = max(1e-2, REASSOC_ATOL_REL * float(want.abs().max()))
    return checkAllclose(want, got, rtol=2e-5, atol=atol, msg=msg, printLog=False)


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
@functools.cache
def _variants():
    """The kernel instances this build compiled, looked up LAZILY.

    Never at module scope: the probe reads the device's properties, so it initializes the HIP
    context at IMPORT and raises outright on an arch ``main()`` is meant to skip -- and CI
    discovers every ``op_tests/test_*.py`` on the other shards. The gate has to run first.
    """
    return pa_mqa_logits_mxfp4_variants()


def _qpb_max():
    """The widest ``Q_PER_BLOCK`` compiled. ``_g4`` replicates rows by it so a group's rows
    share a window exactly on the widest instance; a narrower one cuts the same data finer,
    which is a regime the cases want covered rather than avoided."""
    return max((v.q_per_block for v in _variants()), default=4)


def _resolve_variant(name):
    """The perf decode sweep names instances by the shared ``qlen{Q}_kv{K}`` convention. Both
    arches compile ``qlen1_kv64``, so it resolves natively on each; a name the running arch did
    NOT compile (gfx1250's ``qlen4_kv64`` on gfx950, or gfx950's ``qlen1_kv256`` on gfx1250)
    falls back to its default (``None``), so one shape list drives both targets."""
    if name is None:
        return None
    return name if name in {v.name for v in _variants()} else None


def assert_qshare_windows(cu_tiles, num_tiles, local_starts, local_ends, q_per_block):
    """Check the one condition the schedule takes on faith: within a tile the window rule is
    NON-DECREASING, so the union is the first row's start and the LAST row's end -- two loads
    instead of a reduction. A violation makes the builder compute a union that is not one, so
    rows whose window reaches past it are scored short.

    Host-side and synchronising, which is why it lives here and not in the op: it is worth three
    device-to-host copies in a correctness sweep and nothing in a hot path.

    ``q_per_block`` must be THIS plan's instance and never the widest one the build compiled:
    the one-row instance cuts every tile to a single row, so a bound taken from the widest
    would be satisfied by anything and the span check would stop checking.

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
        if not (0 < hi - lo <= q_per_block):
            raise AssertionError(
                f"qshare: tile {t} spans rows [{lo},{hi}), which is not 1..{q_per_block} rows"
            )
        for r in range(lo + 1, hi):
            if ls[r] < ls[r - 1] or le[r] < le[r - 1]:
                raise AssertionError(
                    f"qshare: tile {t} window is not non-decreasing (row {r - 1}: "
                    f"[{ls[r - 1]},{le[r - 1]}) then row {r}: [{ls[r]},{le[r]}))"
                )


def run_one(inp, qlens, rb, ls, le, label, seed, variant, check_windows=True):
    """Launch one case over explicit per-row windows, on a PINNED instance, and score it."""
    total_q = int(rb.numel())
    cu = torch.tensor(
        [0] + list(itertools.accumulate(qlens)), dtype=torch.int32, device=dev
    )
    # The buffers carry the instance, so allocating them is the only way to run a case on
    # something other than what the plan's own `total_q // batch` rule would choose. It is also
    # the only path in this file that touches `plan_buffers` at all.
    buffers = pa_mqa_logits_mxfp4_plan_buffers(
        dev, total_q, len(qlens), variant=variant
    )
    # `local_starts` is passed because these cases carry non-zero window starts; ATOM's paths do
    # not and leave it None. `row_to_batch` is passed because `block_tables` here is per
    # SEQUENCE -- leaving it None would make the kernel read the table by query row.
    plan = pa_mqa_logits_mxfp4_plan(
        cu, le, buffers=buffers, total_q=total_q, local_starts=ls, row_to_batch=rb
    )
    if check_windows and plan.variant.q_per_block > 1:
        # The condition the kernel cannot check. Host-side and synchronising, so it runs in the
        # correctness path only -- and it is worth running, because breaking it DEADLOCKS the
        # CTA rather than returning a wrong answer. Only a tile WIDER than one row can violate it,
        # so it is moot at q_per_block == 1 (gfx950, and gfx1250's one-row instance).
        assert_qshare_windows(
            plan.cu_tiles, plan.num_tiles, ls, le, plan.variant.q_per_block
        )

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
        "case": label, "variant": plan.variant.name,
        "rows": total_q, "tiles": plan.num_tiles, "ctas": plan.num_ctas,
        "max_win": int(le.max()), "err": err, "oob -inf": oob,
        "window written": wr, "pass": err == 0 and oob and wr,
    }  # fmt: skip


def check_prefill(windows_per_batch, seed, label, data_init, scale_init, variant):
    """One ragged-prefill case from explicit per-row ``(start, end)`` windows."""
    qlens = [len(w) for w in windows_per_batch]
    total_q = sum(qlens)
    max_end = max(e for w in windows_per_batch for (_, e) in w)
    inp = build_inputs(len(qlens), max_end, total_q, seed, data_init, scale_init)

    rb, ls, le = [], [], []
    for b, w in enumerate(windows_per_batch):
        for s, e in w:
            rb.append(b)
            ls.append(s)
            le.append(e)
    rb = torch.tensor(rb, dtype=torch.int32, device=dev)
    ls = torch.tensor(ls, dtype=torch.int32, device=dev)
    le = torch.tensor(le, dtype=torch.int32, device=dev)

    ret = run_one(inp, qlens, rb, ls, le, label, seed, variant)
    del inp
    torch.cuda.empty_cache()
    return {"data_init": data_init, "scale_init": scale_init, "seed": seed, **ret}


def _g4(windows_per_batch):
    """Replicate each row ``_qpb_max()`` times so a group's rows share a window exactly -- the
    EASY regime. The CSA rules below are the hard one, where adjacent rows of a group differ by
    a column and the loop bound has to be their union."""
    return [[r for r in b for _ in range(_qpb_max())] for b in windows_per_batch]


def _csa_fresh(qlen):
    """Fresh sequence: row n sees ``(n + 1) // RATIO``; runs start at n = 3 mod 4."""
    return [(0, (n + 1) // CSA_RATIO) for n in range(qlen)]


def _csa_chunked(qlen, kvlen):
    """``kvlen`` compressed rows committed and this chunk is the tail: row n sees
    ``kvlen - (qlen - 1 - n) // RATIO``, so runs align to the tail rather than to n."""
    return [(0, kvlen - (qlen - 1 - n) // CSA_RATIO) for n in range(qlen)]


# A `cu_seq_q` claiming more rows than `q` holds. 14 against 10 puts BOTH of the kernel's
# row_id clauses on the path: the cut is [0,4) [4,8) [8,12) [12,14), so tile 2 straddles the
# real row count (the per-wave clause) and tile 3 is past it entirely (the CTA-uniform one).
ROWID_REAL_ROWS, ROWID_CLAIMED_ROWS, ROWID_WIN = 10, 14, 200


def check_row_id_bound(data_init, scale_init, seed, variant):
    """The one caller inconsistency the kernel has to survive rather than diagnose.

    ``cu_seq_q[batch]`` is device data, so no launcher check can compare it against
    ``q.shape[0]`` -- reading it host-side is the sync this whole design exists to avoid.
    Unbounded, the chain ``cu_seq_q -> cu_tiles -> rec.row_id -> row_id`` puts a CTA's reads
    past ``q`` / ``q_scale`` / ``weights`` and its WRITES past ``out``. The kernel bounds
    ``row_id`` against ``num_rows`` instead, so the surplus rows are dropped and the real ones
    stay right.

    THIS CASE IS THE ONLY THING THAT WALKS THAT PATH -- every other case here builds
    ``cu_seq_q`` and ``q`` from one ``qlens``, so the rows always agree.

    **What gives it teeth is the oversized `out`, not the hope of a fault.** With a
    right-sized one the unbounded kernel overruns by 4 KB, which the caching allocator usually
    absorbs: no fault, no wrong answer in the rows that are checked, and the probe passes while
    establishing nothing. So `out` is allocated for all ``claimed`` rows and the surplus ones
    are required to stay at their -inf pre-fill. That is the same `row_id` the overrun would
    have used, so it tests the bound and not a symptom.

    Both clauses need a tile WIDER than one row, so only the qshare instances put the per-wave
    half on the path; at one row per CTA the cut is `[0,1) .. [13,14)`, no tile straddles
    ``real`` and the CTA-uniform clause is the only one that can fire. Run on both anyway --
    the CTA-uniform half is the one that bounds the store, and it is the whole probe there.
    """
    real, claimed = ROWID_REAL_ROWS, ROWID_CLAIMED_ROWS
    inp = build_inputs(1, ROWID_WIN, real, seed, data_init, scale_init)

    def t(v):
        return torch.tensor(v, dtype=torch.int32, device=dev)

    # The window arrays describe `claimed` rows; `q` holds `real`. Every launcher check passes:
    # num_rows is q.shape[0], local_ends is longer than it, and out is longer still.
    rb = torch.zeros(claimed, dtype=torch.int32, device=dev)
    ls = torch.zeros(claimed, dtype=torch.int32, device=dev)
    le = t([ROWID_WIN] * claimed)
    buffers = pa_mqa_logits_mxfp4_plan_buffers(dev, claimed, 1, variant=variant)
    plan = pa_mqa_logits_mxfp4_plan(
        t([0, claimed]),
        le,
        buffers=buffers,
        total_q=claimed,
        local_starts=ls,
        row_to_batch=rb,
    )
    out = torch.full(
        (claimed, inp.max_seq_len), float("-inf"), dtype=torch.float32, device=dev
    )
    pa_mqa_logits_mxfp4(
        inp.q_packed, inp.q_scale, inp.kv_cache, inp.kv_scale, inp.block_tables,
        inp.weights, plan, inp.max_seq_len,
        weight_scale=WEIGHT_SCALE, kv_block_size=KV_BLOCK_SIZE, out=out,
    )  # fmt: skip
    torch.cuda.synchronize()

    # Scored over the rows `q` actually holds -- the surplus ones have no Q to be right about.
    rows = list(range(real))
    err = check_rows(out, ref_rows(inp, rows, rb, ls, le), "row_id bound")
    oob = oob_is_neginf(out[:real], ls[:real], le[:real])
    wr = window_is_written(out[:real], ls[:real], le[:real])
    # The clause that fails without the kernel's bound: rows past `q` were never scheduled.
    untouched = bool(torch.isneginf(out[real:]).all().item())
    ret = {
        "data_init": data_init, "scale_init": scale_init, "seed": seed,
        "case": f"cu_seq_q {claimed} > q {real}", "variant": plan.variant.name,
        "rows": real,
        "tiles": plan.num_tiles, "ctas": plan.num_ctas, "max_win": ROWID_WIN,
        "err": err, "oob -inf": oob and untouched, "window written": wr,
        "pass": err == 0 and oob and untouched and wr,
    }  # fmt: skip
    del inp, out
    torch.cuda.empty_cache()
    return ret


def run_corner(data_init, scale_init, seed):
    """The cases the qshare contract is made of: short groups at every residue mod
    the widest Q_PER_BLOCK, windows not starting at 0, the KV_TILE = 64 neighbourhood, every
    window start mod 128, and both ATOM CSA regimes where a group's rows differ by a column.

    Every case runs on every compiled instance, so the count below is the case list times
    ``len(_variants())``.

    The per-case seeds below are OFFSETS from ``--seed``: the cases stay distinct from one
    another while the whole sweep moves with the flag.
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
    # EVERY compiled instance, and not the one the plan would pick on its own. The default rule
    # is `total_q // batch` and no case above has fewer than two rows per sequence -- the
    # shortest is `csa short groups` at 8 rows over 4 batches -- so left to itself this suite
    # builds the qshare instance 14 times and the one-row instance never. The probes' coverage
    # is stated in TILE units, so it has to be re-established per instance rather than
    # inherited: a tile is four rows for one of them and one row for the other.
    variants = _variants()
    if not variants:
        raise RuntimeError(
            "no compiled kernel instances to run the corner suite on; an empty sweep reports "
            "`pass` having tested nothing"
        )
    # `check_row_id_bound` walks the gfx1250 cu_seq_q -> cu_tiles -> row_id overrun path and its
    # tile straddle -- a gfx1250-only guarantee. gfx950 cuts one row per tile and its callers
    # build cu_seq_q from the same qlens as q, so that path is never taken; skip it there.
    row_id = (
        []
        if _is_permuted()
        else [check_row_id_bound(data_init, scale_init, seed + 70, v) for v in variants]
    )
    return [
        check_prefill(w, seed + case_seed, label, data_init, scale_init, v)
        for v in variants
        for w, case_seed, label in cases
    ] + row_id


SPREAD_PROBE_ROWS = 4096


def scale_spread(scale_init, seed=0, rows=SPREAD_PROBE_ROWS):
    """What a ``--scale-init`` can see, measured on the array the sweep would actually build.

    Returns ``(distinct exponents, fraction of rows 16 apart that disagree)``. The second is
    the one that decides: ``b_scale_sel`` misroutes across a lane HALF, so a spread periodic in
    16 hides a misroute as completely as a flat one.
    """
    e8 = fill_exponents(
        (rows, BLOCKS_ROW), scale_init, make_generator(seed, device=dev)
    )
    block0 = e8[:, 0]
    distinct = int(torch.unique(e8).numel())
    disagree = float((block0[:-16] != block0[16:]).float().mean().item())
    return distinct, disagree


def correctness_blindness(data_init, scale_init):
    """Why this init pair cannot judge a wrong answer, or ``None`` when it can.

    There is no second implementation on this target, so the dequantized reference is the only
    judge and it shares this file's understanding of the layout. That leaves two ways for the
    sweep to pass while establishing nothing, and both are properties of the INIT PAIR rather
    than of the kernel -- so they skip the sweep with the reason in the table instead of
    passing it.
    """
    if data_init == "zero":
        return "data-init zero makes every fp4 nibble 0, so the reference agrees with anything"
    distinct, disagree = scale_spread(scale_init)
    if distinct < 3 or disagree < MISROUTE_DISAGREE_MIN:
        return (
            f"scale-init {scale_init} spreads {distinct} exponents and rows 16 apart disagree "
            f"{disagree:.1%} (want >= 3 and >= {MISROUTE_DISAGREE_MIN:.0%}); a misrouted "
            "b_scale_sel would read an exponent that happens to be right"
        )
    return None


# ── NaN E8M0 scale propagation ────────────────────────────────────────────────
def poison_kv_rows(kv_scale, block_tables, row_in_seq):
    """A copy of ``kv_scale`` with KV row ``row_in_seq`` of EVERY batch set to 0xFF (E8M0 NaN).

    Every batch, because the block tables differ per batch and poisoning only batch 0's would
    leave the others as an untested control. Poisoned by MARKING in the NATURAL layout and pushing
    the mark through the arch's own permutation (``_scale_to_opus`` on gfx950, identity on
    gfx1250), rather than by computing where those bytes land -- the permutation is the thing
    under test's own ABI, so rederiving it here would let the test and the kernel agree on a wrong
    offset.
    """
    nb = block_tables.numel()  # one entry per (batch, page)
    mark = torch.zeros(nb * KV_BLOCK_SIZE, BLOCKS_ROW, dtype=torch.uint8, device=dev)
    blk = block_tables[:, row_in_seq // KV_BLOCK_SIZE].long()
    mark[blk * KV_BLOCK_SIZE + row_in_seq % KV_BLOCK_SIZE] = 1
    out = kv_scale.clone()
    if _is_permuted():
        out[_scale_to_opus(mark, KV_BLOCK_SIZE) != 0] = 0xFF
    else:
        out[mark.reshape(out.shape) != 0] = 0xFF
    return out


def check_nan_scale(entry, bs, next_n, ends, seed, variant, label, kv_row=0):
    """A NaN E8M0 scale must reach the logits of every row that attends its KV row, and nowhere
    else. E8M0 0xFF is NaN and an E2M1 nibble cannot encode one, so a NaN scale is the only way a
    non-finite value enters this kernel -- and the relu decides whether it comes out again. The
    kernel promises the IEEE ``maximum`` that PROPAGATES it (not a NaN-swallowing select), and
    only under ``-fno-finite-math-only``. Asserted as an exact SET, so NaN smeared too far fails.
    """
    total_q = bs * next_n
    assert len(ends) == total_q
    inp = build_inputs(bs, max(max(ends), 1), total_q, seed, "norm", "auto")

    def t(v):
        return torch.tensor(v, dtype=torch.int32, device=dev)

    rb = t([b for b in range(bs) for _ in range(next_n)])
    ls = torch.zeros(total_q, dtype=torch.int32, device=dev)
    le = t(list(ends))
    cu = t([0] + list(itertools.accumulate([next_n] * bs)))
    kvs = poison_kv_rows(inp.kv_scale, inp.block_tables, kv_row)

    # `entry` is which TABLE this is: prefill hands the builder the window starts, decode leaves
    # them None. Both are zero here so the schedules coincide -- what differs is the builder's
    # `local_starts` branch, which reads the array instead of assuming 0.
    buffers = pa_mqa_logits_mxfp4_plan_buffers(dev, total_q, bs, variant=variant)
    plan = pa_mqa_logits_mxfp4_plan(
        cu,
        le,
        buffers=buffers,
        total_q=total_q,
        local_starts=ls if entry == "prefill" else None,
        row_to_batch=rb,
    )
    out = pa_mqa_logits_mxfp4(
        inp.q_packed, inp.q_scale, inp.kv_cache, kvs, inp.block_tables,
        inp.weights, plan, inp.max_seq_len,
        weight_scale=WEIGHT_SCALE, kv_block_size=KV_BLOCK_SIZE,
    )  # fmt: skip
    torch.cuda.synchronize()

    col = torch.arange(out.shape[1], device=dev).unsqueeze(0)
    inside = (col >= ls.unsqueeze(1)) & (col < le.unsqueeze(1))
    want = inside & (col == kv_row)
    got = inside & ~torch.isfinite(out)
    exact = bool(torch.equal(want, got))
    oob = oob_is_neginf(out, ls, le)
    return {
        "case": label, "entry": entry, "variant": plan.variant.name, "rows": total_q,
        "max_win": int(le.max()), "expect nan": int(want.sum()), "got nan": int(got.sum()),
        "exact set": exact, "oob -inf": oob, "pass": exact and oob,
    }  # fmt: skip


def run_nan_scale(seed):
    """:func:`check_nan_scale` over both entry points and every compiled instance.

    Separate from :func:`run_corner` because its pass condition is the opposite one: these cases
    REQUIRE non-finite in-window cells, so ``window_is_written`` -- which every corner case asserts
    -- is deliberately false here.
    """
    oks = []
    for v in _variants():
        # Ragged windows, all containing KV row 0.
        oks.append(check_nan_scale("prefill", 2, 3, [50, 120, 200, 40, 100, 180],
                                   seed + 80, v, "prefill, nan at kv row 0"))  # fmt: skip
        # A window that EXCLUDES the poisoned row -- the control that says the NaN is not smeared.
        oks.append(check_nan_scale("prefill", 1, 4, [0, 1, 2, 3],
                                   seed + 81, v, "prefill, nan at kv row 2", kv_row=2))  # fmt: skip
        # Decode, where a window spans several KV splits and only one holds the row.
        oks.append(check_nan_scale("decode", 2, 4,
                                   [200, 201, 202, 203, v.block_k * 2 + 1] + [130] * 3,
                                   seed + 82, v, "decode, nan at kv row 0"))  # fmt: skip
    df = pd.DataFrame(oks)
    aiter.logger.info(
        "MXFP4 MQA logits NaN E8M0 scale propagation, %d/%d pass (markdown):\n%s",
        int(df["pass"].sum()), len(df), df.to_markdown(index=False),
    )  # fmt: skip
    return bool(df["pass"].all())


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
        "TB/s": round(nbytes / us / 1e6, 3),
        "err": check_rows(out, ref, "perf"),
    }
    del ref, out
    torch.cuda.empty_cache()
    return ret


@benchmark()
def test_prefill_causal(bs, data_init, scale_init, seed):
    """One causal prefill shape: 16384 query rows split across ``bs`` batches, ctx == qlen.

    The qlen split is seeded by ``bs`` rather than by ``--seed``, so the SHAPE is a function of
    the sweep point alone and two seeds stay comparable; ``--seed`` moves the data only.
    """
    qlens = gen_prefill_qlens(bs, seed=bs)
    total_q = sum(qlens)
    inp = build_inputs(bs, max(qlens), total_q, seed, data_init, scale_init)
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
        # Whatever the op settled on its own: these rows pass no `variant`, which is the other
        # half of the decode table's coverage and the only place the DEFAULT is exercised.
        "variant": plan.variant.name,
        "total_q": total_q,
        "tiles": plan.num_tiles,
        "ctas": plan.num_ctas,
        "max_win": int(le.max()),
        "n_logits": n_logits,
    }
    ret.update(score(ours, inp, rb, ls, le, total_q, n_logits, seed=seed))
    del inp, out
    torch.cuda.empty_cache()
    return ret


def run_windowed_case(per_batch, data_init, scale_init, seed):
    """Launch and score one prefill case from explicit per-row ``(start, end)`` windows.

    The shared half of the two CSA regimes -- they differ only in how ``per_batch`` is built.
    The window rule is an INPUT either way and is never derived: ``tail_causal_windows`` is the
    other shape this harness builds and it cannot express a CSA one, since
    ``floor((x - d) / R) != floor(x / R) - d``.
    """
    bs = len(per_batch)
    qlens = [len(w) for w in per_batch]
    total_q = sum(qlens)
    max_end = max(e for w in per_batch for (_, e) in w)
    inp = build_inputs(bs, max_end, total_q, seed, data_init, scale_init)

    def t(v):
        return torch.tensor(v, dtype=torch.int32, device=dev)

    rb = t([b for b, w in enumerate(per_batch) for _ in w])
    ls = t([s for w in per_batch for (s, _) in w])
    le = t([e for w in per_batch for (_, e) in w])
    cu = t([0] + list(itertools.accumulate(qlens)))
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
        # As in `test_prefill_causal`: the op's own default, passed no `variant`.
        "variant": plan.variant.name,
        "total_q": total_q,
        "tiles": plan.num_tiles,
        "ctas": plan.num_ctas,
        "min_win": int((le - ls).min()),
        "max_win": int(le.max()),
        "max_seq_len": inp.max_seq_len,
        "n_logits": n_logits,
    }
    ret.update(score(ours, inp, rb, ls, le, total_q, n_logits, seed=seed))
    del inp, out
    torch.cuda.empty_cache()
    return ret


@benchmark()
def test_prefill_fresh(bs, qlen, data_init, scale_init, seed):
    """Fresh CSA prefill: the sequence starts empty, so row n sees ``(n + 1) // CSA_RATIO``.

    ``bs`` sequences of ``qlen`` rows each, which is where a group's four rows differ by a
    column rather than sharing one.
    """
    return run_windowed_case(
        [_csa_fresh(qlen) for _ in range(bs)], data_init, scale_init, seed
    )


@benchmark()
def test_prefill_chunked(bs, kvlen, data_init, scale_init, seed):
    """Chunked CSA prefill at PR #5332's shapes, so the two PRs' tables are comparable.

    A sequence ends this forward with ``kvlen`` COMPRESSED rows committed and this chunk is its
    tail, so row n of a ``qlen``-row sequence sees ``kvlen - (qlen - 1 - n) // CSA_RATIO``.
    ``kvlen = 25000`` stands for roughly a 100k-raw-token context.

    ``PREFILL_TOTAL_QLEN`` rows split RAGGEDLY across ``bs`` by ``gen_prefill_qlens`` -- the
    generator ``test_prefill_causal`` already uses, and the one #5332's own sweep uses. That is
    what makes the shapes identical rather than merely similar: it reproduces #5332's
    ``n_logits`` and ``min_win`` to the digit (376053760 / 20905 at bs=1, 392830720 / 22945 at
    bs=2), where an even split lands 256 logits away at bs=2 and 1.3M away at bs=4.
    """
    return run_windowed_case(
        [_csa_chunked(q, kvlen) for q in gen_prefill_qlens(bs, seed=bs)],
        data_init,
        scale_init,
        seed,
    )


@benchmark()
def test_decode(mtp, seqs, n_long, variant, data_init, scale_init, seed):
    """MTP decode over ``seqs`` sequences, ``n_long`` of them long, on ``variant``.

    ``mtp`` is the query rows per sequence -- ``next_n`` in the schedule's own tables -- so the
    row count is ``seqs * mtp`` and the tile count is ``seqs * ceil(mtp/QPB)``. Within one
    ``(mtp, seqs)`` the tile count is therefore CONSTANT across the ragged shapes, which is what
    makes them a clean read on load balance: same rows, same tiles, only the work per tile moves.

    Every row of a sequence takes that sequence's whole window. The tail-causal alternative --
    row ``j`` ending at ``ctx - (mtp - 1 - j)`` -- would read the same here, because the loop
    bound is the TILE's union and that is the last row's end either way; the per-row ends only
    move the store mask, by three columns out of 25000.

    ``variant`` is the kernel instance, or ``None`` for the op's own default. It is a per-shape
    CHOICE made by the driver and never inferred here -- see ``decode_shapes``.
    """
    variant = _resolve_variant(variant)  # gfx1250 names; the arch default elsewhere
    n_short = seqs - n_long
    ctxs = [DECODE_WIN_LONG] * n_long + [DECODE_WIN_SHORT] * n_short
    qlens = [mtp] * seqs
    total_q = seqs * mtp
    inp = build_inputs(seqs, max(ctxs), total_q, seed, data_init, scale_init)

    def t(v):
        return torch.tensor(v, dtype=torch.int32, device=dev)

    rb = t([b for b in range(seqs) for _ in range(mtp)])
    ls = torch.zeros(total_q, dtype=torch.int32, device=dev)
    le = t([ctxs[b] for b in range(seqs) for _ in range(mtp)])
    cu = t([0] + list(itertools.accumulate(qlens)))
    # `local_starts` stays None rather than an array of zeros, because that is the call ATOM
    # makes on this path and a per-row start load is not part of it.
    buffers = pa_mqa_logits_mxfp4_plan_buffers(dev, total_q, seqs, variant=variant)
    plan = pa_mqa_logits_mxfp4_plan(
        cu, le, buffers=buffers, total_q=total_q, row_to_batch=rb
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

    n_logits = int(le.sum().item())
    ret = {
        "gfx": get_gfx(),
        "regime": "uniform" if n_short == 0 else "ragged",
        "n_short": n_short,
        "variant": plan.variant.name,
        "rows": total_q,
        "tiles": plan.num_tiles,
        "ctas": plan.num_ctas,
        "max_win": int(le.max()),
        "n_logits": n_logits,
    }
    ret.update(score(ours, inp, rb, ls, le, total_q, n_logits, seed=seed))
    del inp, out
    torch.cuda.empty_cache()
    return ret


def summarize(name, rows):
    """One result table, twice: markdown to read and one-line JSON to forward.

    The JSON is what a combined benchmark driver validates and re-renders; the markdown is what
    makes a standalone run readable. A driver keeps only the JSON lines, so emitting both costs
    nothing there.
    """
    df = pd.DataFrame(rows)
    aiter.logger.info("%s (markdown):\n%s", name, df.to_markdown(index=False))
    print_json_table(name, df)


def init_pairs(data_init, scale_init):
    """Pair the two axes POSITION-WISE, a length-1 side broadcasting.

    Paired and not crossed, because that is the rule the combined driver applies on its side;
    crossing here would turn one requested pair into four cases and quietly quadruple a sweep.
    """
    data, scale = list(data_init), list(scale_init)
    if len(data) == 1:
        data *= len(scale)
    if len(scale) == 1:
        scale *= len(data)
    if len(data) != len(scale):
        raise ValueError(
            "--data-init and --scale-init must have equal length (a length-1 side broadcasts)"
        )
    return list(zip(data, scale))


def main():
    # Whole-op arch gate, here rather than inside the @benchmark fns: CI discovers every
    # op_tests/test_*.py and runs it on the other shards too, where the wrapper's own arch
    # check would raise and fail the shard. Positive allow-list, so an unknown new card skips.
    if get_gfx() not in SUPPORTED_GFX:
        why = f"built for {'/'.join(SUPPORTED_GFX)}, skipped on {get_gfx()}"
        aiter.logger.warning("MXFP4 MQA logits: %s", why)
        # Still a table: a driver that recognises none at all reports a broken extractor
        # rather than a skip, and the two want different answers from whoever reads it.
        summarize("pa_mqa_logits_mxfp4 (not run)", [{"err_msg": why}])
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
        "--no-verify", action="store_true", help="skip the correctness sweep, perf only"
    )
    # Hand-rolled rather than `add_data_init_args`, whose --scale-init offers the four FLOAT
    # scale dists. These scales are E8M0 on the wire, so the choices have to be
    # E8M0_SCALE_DISTS or a driver passing the MX default `auto` is rejected by argparse
    # before the test runs at all.
    parser.add_argument(
        "--data-init", nargs="+", choices=list(DATA_DISTS), default=["norm"],
        help="DATA init for the fp4 nibbles and the weights, paired position-wise\n"
             "with --scale-init (a length-1 side broadcasts)",
    )  # fmt: skip
    parser.add_argument(
        "--scale-init", nargs="+", choices=list(E8M0_SCALE_DISTS), default=["auto"],
        help="E8M0 SCALE init, sampled INDEPENDENTLY of the data. This axis alone\n"
             "decides whether the suite can see a misrouted scale -- see scale_spread",
    )  # fmt: skip
    parser.add_argument(
        "--seed", type=int, default=0,
        help="RNG seed for the data and the sampled reference rows; shapes are pinned to\n"
             "the sweep point, so two seeds stay comparable",
    )  # fmt: skip
    args = parser.parse_args()

    pairs = init_pairs(args.data_init, args.scale_init)

    corner, not_judged, ok = [], [], True
    for data_init, scale_init in pairs:
        distinct, disagree = scale_spread(scale_init)
        aiter.logger.info(
            "scale-init %s spreads %d E8M0 exponents; rows 16 apart disagree %.1f%%",
            scale_init,
            distinct,
            100.0 * disagree,
        )
        if args.no_verify:
            continue
        blind = correctness_blindness(data_init, scale_init)
        if blind is not None:
            # Its OWN table, not a row in the one below. A skip shares none of that table's
            # columns, and one NaN turns every bool column in it into a float -- so `pass`
            # would reach a reader as 1.0 rather than true.
            not_judged.append(
                {
                    "data_init": data_init,
                    "scale_init": scale_init,
                    "seed": args.seed,
                    "err_msg": blind,
                }
            )
            continue
        rows = run_corner(data_init, scale_init, args.seed)
        ok = all(r["pass"] for r in rows) and ok
        corner += rows
    if not_judged:
        summarize("pa_mqa_logits_mxfp4 corner (not judged)", not_judged)
    if corner:
        summarize("pa_mqa_logits_mxfp4 corner", corner)

    # NaN E8M0 propagation: its own pass condition (REQUIRES non-finite in-window cells), and its
    # own random data -- independent of the init-pair sweep above.
    if not args.no_verify:
        ok = run_nan_scale(args.seed) and ok

    summarize(
        "pa_mqa_logits_mxfp4 prefill causal",
        [
            test_prefill_causal(bs, data_init, scale_init, args.seed)
            for data_init, scale_init in pairs
            for bs in args.batch
        ],
    )

    fresh_shapes = [(1, 16384), (2, 8192), (4, 4096)]
    summarize(
        f"pa_mqa_logits_mxfp4 prefill fresh (csa ratio {CSA_RATIO})",
        [
            test_prefill_fresh(bs, qlen, data_init, scale_init, args.seed)
            for data_init, scale_init in pairs
            for bs, qlen in fresh_shapes
        ],
    )

    # PR #5332's three chunked rows: PREFILL_TOTAL_QLEN rows raggedly split across bs, every
    # sequence carrying 25000 committed compressed rows.
    chunked_shapes = [(1, 25000), (2, 25000), (4, 25000)]
    summarize(
        f"pa_mqa_logits_mxfp4 prefill chunked (csa ratio {CSA_RATIO}, #5332 shapes)",
        [
            test_prefill_chunked(bs, kvlen, data_init, scale_init, args.seed)
            for data_init, scale_init in pairs
            for bs, kvlen in chunked_shapes
        ],
    )

    # (mtp, seqs, n_long, variant).
    #
    # **MTP = 1 carries the batch sweep, because it is the regime the framework runs.** At one
    # row per sequence `rows == seqs == TILES`, so `seqs` IS the tile count and the schedule has
    # to spread 1..128 tiles over 3072 CTAs -- the axis it lives or dies on, and the one MTP > 1
    # cannot be read on, because packing four rows into a tile hides it. Its two ragged rows hold
    # the tile count and move only the long fraction, for the same reason the mtp = 4 ones do.
    #
    # The MTP > 1 rows stay at `DECODE_SEQS`: two uniform ones extend the row count from the
    # mtp = 1 anchor, and the last three hold mtp = 4 and 128 rows so the uniform mtp = 4 line is
    # their same-row-count anchor.
    #
    # **The variant is a per-shape CHOICE, not a rule.** The op defaults every shape to the
    # four-row instance and infers nothing, so `mtp = 1` -- where that instance masks three of
    # its four waves off -- names the one-row instance here, which is what a caller that knows
    # its own regime does. Left to the default those rows read about twice the time, and the
    # `variant` column is what makes the choice visible in the table rather than implied.
    decode_shapes = [
        (1, 1, 1, "qlen1_kv64"),
        (1, 8, 8, "qlen1_kv64"),
        (1, DECODE_SEQS, DECODE_SEQS, "qlen1_kv64"),
        (1, 128, 128, "qlen1_kv64"),
        (1, DECODE_SEQS, 4, "qlen1_kv64"),
        (1, 128, 16, "qlen1_kv64"),
        (4, DECODE_SEQS, DECODE_SEQS, None),
        (8, DECODE_SEQS, DECODE_SEQS, None),
        (4, DECODE_SEQS, 16, None),
        (4, DECODE_SEQS, 4, None),
        (4, DECODE_SEQS, 28, None),
    ]
    summarize(
        f"pa_mqa_logits_mxfp4 decode "
        f"(win {DECODE_WIN_LONG}/{DECODE_WIN_SHORT} compressed cols)",
        [
            test_decode(mtp, seqs, n_long, variant, data_init, scale_init, args.seed)
            for data_init, scale_init in pairs
            for mtp, seqs, n_long, variant in decode_shapes
        ],
    )

    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
